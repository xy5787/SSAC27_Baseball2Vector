"""Raw FanGraphs exports -> the two populations this package models.

Two things that were previously conflated are kept apart here, and the
separation is the whole point of the module:

* **The history pool** (:func:`load_history_pool`) -- *every* season a player
  actually played, with no plate-appearance minimum, 2019-2025. Marcel needs
  this: a season absent from the table is treated as zero PA, which is right for
  "did not play" and wrong for "played a little".
* **The modelling cohort** (:func:`load_extended_cohort`) -- 2019-2025 seasons
  with PA >= 100, the population the tool scores are standardized inside and the
  only rows that are ever a focal player. This keeps the qualifier that Findings
  1 and 2 were built on.

Inputs, both manual FanGraphs leaderboard exports with no PA filter:

    data/raw/fangraphs-leaderboards (1).csv   2021-2026, 465 columns
    data/raw/batting_stats_2019_2020.csv      2019-2020, 465 columns

2026 is dropped: it is later than every outcome season in the study, so it can
only ever be a forward reference.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import (
    CONSTITUENT_COLS,
    CONSTITUENT_STATS,
    ENCODINGS,
    REPO_ROOT,
    TOOL_NAMES,
    apply_b2v_scale,
    tool_score_columns,
)
from baseball2vec.baselines import pca_tool_scores, zscore_tool_scores
from baseball2vec.data import preprocess
from baseball2vec.tools import apply_direction, build_final_groups, season_zscore

EXPORT_2021_2025 = REPO_ROOT / "data" / "raw" / "fangraphs-leaderboards (1).csv"
EXPORT_2019_2020 = REPO_ROOT / "data" / "raw" / "batting_stats_2019_2020.csv"

STUDY_SEASONS = list(range(2019, 2026))
QUALIFIER_PA = 100

def _read_export(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # FanGraphs' combined-report export repeats the identifier columns.
    for base in ("Name", "Team", "Season"):
        duplicate = f"{base}.1"
        if duplicate in df.columns:
            assert (df[base] == df[duplicate]).all(), f"{base} / {duplicate} mismatch"
            df = df.drop(columns=[duplicate])
    return df


def load_raw_exports() -> pd.DataFrame:
    """Both exports, concatenated, restricted to the study window."""
    frames = [_read_export(EXPORT_2019_2020), _read_export(EXPORT_2021_2025)]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Season"].isin(STUDY_SEASONS)].reset_index(drop=True)
    df["player_id"] = "fg:" + df["PlayerId"].astype("Int64").astype(str)
    assert df["player_id"].notna().all()
    assert not df.duplicated(["player_id", "Season"]).any()
    return df


def load_history_pool() -> pd.DataFrame:
    """Every played season, no PA minimum -- the lookback pool Marcel reads.

    Pitchers who batted (2019 and 2021, before the universal designated hitter)
    are left in. They are never focal players and never appear in another
    player's history, and the primary league baseline is wRC+'s definitional 100
    rather than a population mean, so their presence changes no projection.
    """
    df = load_raw_exports()
    pool = df.loc[df["PA"] > 0, ["player_id", "Name", "Season", "PA", "wRC+", "WAR", "Age"]]
    pool = pool.copy()
    assert pool.notna().all().all()
    return pool.sort_values(["player_id", "Season"], kind="mergesort").reset_index(
        drop=True
    )


def load_extended_cohort(verbose: bool = True) -> pd.DataFrame:
    """2019-2025 seasons with PA >= 100, with tool scores recomputed from raw.

    The transform chain is the published one, applied season by season:
    Bayesian stabilization toward the row's own season mean, within-season
    z-scoring, direction correction, then the tool score as the group mean.
    This is season-local for Z-score scores. The historical PCA below fits
    all seasons and is NOT forecast-safe; Task 8 implements train-window PCA.

    Unlike :func:`common.load_player_seasons`, which reuses the archived
    2021-2025 tool scores, this recomputes them so that 2019 and 2020 can join
    the study. The two agree to Pearson r >= 0.9999 on the overlapping seasons;
    `task6_extended_window.py` asserts that agreement on every run.
    """
    df = load_raw_exports()
    cohort = df[df["PA"] >= QUALIFIER_PA].reset_index(drop=True)

    processed = preprocess(cohort.copy())
    groups = build_final_groups(processed)
    stats = [column for columns in groups.values() for column in columns]
    assert stats == CONSTITUENT_STATS, (stats, CONSTITUENT_STATS)

    aligned = apply_direction(season_zscore(processed, stats))
    out = processed[
        ["player_id", "Name", "Team", "Season", "PA", "WAR", "wRC+", "OPS", "Age"]
    ].copy()
    out["age_t"] = out["Age"].astype(float)

    for source, target in zip(CONSTITUENT_STATS, CONSTITUENT_COLS):
        out[target] = aligned[source].to_numpy()

    zscore = zscore_tool_scores(aligned, groups)
    pca, _ = pca_tool_scores(aligned, groups)
    for prefix, scores in (("ZScore", zscore), ("PCA", pca)):
        for tool in TOOL_NAMES:
            column = f"{prefix}_{tool}"
            out[column] = scores[tool].to_numpy()
            # Season-local scale only; does NOT make the globally fitted PCA safe.
            out[f"safe_{column}"] = out.groupby("Season")[column].transform(
                lambda x: (x - x.mean()) / x.std(ddof=0)
            )

    out = apply_b2v_scale(out, tool_score_columns(out))

    if verbose:
        counts = out.groupby("Season").size()
        print("  extended cohort (PA >= 100) by season:")
        print("   ", dict(counts))
    assert not out.duplicated(["player_id", "Season"]).any()
    return out.sort_values(["player_id", "Season"], kind="mergesort").reset_index(
        drop=True
    )


def extended_encodings() -> dict[str, str]:
    """Encodings available on the recomputed cohort.

    JointVAE is deliberately absent: reproducing it for 2019-2020 would require
    retraining the VAE on the widened cohort, which changes the encoder rather
    than extending it. Task 5 cross-checks JointVAE on the archived seasons.
    """
    return {name: prefix for name, prefix in ENCODINGS.items() if name != "vae"}


# --------------------------------------------------------------------------- #
# Transitions on the extended cohort
# --------------------------------------------------------------------------- #
def build_extended_transitions(cohort: pd.DataFrame) -> pd.DataFrame:
    """Adjacent (t, t+1) season pairs on the recomputed 2019-2025 cohort.

    Same construction as :func:`common.build_transitions`, but keyed on the
    FanGraphs ``PlayerId`` rather than Chadwick-matched names, and carrying the
    recomputed tool scores instead of the archived ones.
    """
    feature_cols = [
        f"safe_{prefix}_{tool}"
        for prefix in extended_encodings().values()
        for tool in TOOL_NAMES
    ]
    carry = ["Name", "Season", "PA", "WAR", "wRC+", "OPS", "age_t"]
    keep = [*carry, *feature_cols, *CONSTITUENT_COLS]

    rows: list[dict] = []
    for player_id, player in cohort.groupby("player_id", sort=True):
        records = player.sort_values("Season", kind="mergesort")[keep].to_dict("records")
        for left, right in zip(records, records[1:]):
            if int(right["Season"]) != int(left["Season"]) + 1:
                continue
            row = {"player_id": player_id, "Name": left["Name"]}
            row.update({f"{k}_t": v for k, v in left.items()})
            row.update({f"{k}_t1": v for k, v in right.items()})
            rows.append(row)

    out = pd.DataFrame(rows)
    assert not out.duplicated(["player_id", "Season_t", "Season_t1"]).any()
    assert (out["Season_t1"] == out["Season_t"] + 1).all()
    return out.sort_values(["Season_t", "player_id"], kind="mergesort").reset_index(
        drop=True
    )


def extended_rolling_folds(transitions: pd.DataFrame) -> dict[str, tuple[list[int], int]]:
    """Expanding-window folds: every input season that has at least one earlier one.

    Fold label is the test transition, e.g. ``"2022->23"``.
    """
    inputs = sorted(int(s) for s in transitions["Season_t"].unique())
    folds: dict[str, tuple[list[int], int]] = {}
    for index, season in enumerate(inputs):
        if index == 0:
            continue  # nothing earlier to train on
        label = f"{season}->{str(season + 1)[2:]}"
        folds[label] = (inputs[:index], season)
    return folds
