"""Shared transforms and evaluation utilities for the public SSAC reproduction."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SSAC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SSAC_ROOT
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseball2vec.baselines import zscore_tool_scores  # noqa: E402
from baseball2vec.data import preprocess  # noqa: E402
from baseball2vec.tools import (  # noqa: E402
    FEATURE_GROUPS,
    apply_direction,
    build_final_groups,
    grade_2080,
    season_zscore,
)

# Scale of the B2V tool scores handed to every model. "sd10" (default, the SSAC
# submission) puts them on the 20-80 scouting scale (50 + 10 z, clipped to
# [20, 80]) inside each input season, the convention SD10Pipeline/ introduced;
# outputs go to results_sd10/. "zscore" keeps the season-standardized scores
# behind the earlier results/.
B2V_SCALE = os.environ.get("B2V_SCALE", "sd10")
assert B2V_SCALE in ("zscore", "sd10"), f"unknown B2V_SCALE={B2V_SCALE!r}"
RESULTS_DIR = SSAC_ROOT / ("results_sd10" if B2V_SCALE == "sd10" else "results")
INTERMEDIATE_DIR = SSAC_ROOT / "data" / "processed"

RAW_EXPORT_PATH = REPO_ROOT / "data" / "raw" / "batting_stats_2021_2025.csv"
TRANSITIONS_PATH = INTERMEDIATE_DIR / "transitions.csv"

SEED = 42
N_BOOT = 2000
# Wide grid, used for every model so that no model gets a search advantage.
ALPHA_GRID_WIDE = np.logspace(-3, 4, 50)
# The grid the published Finding 2 run used; reported as a secondary variant.
ALPHA_GRID_ARCHIVED = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
INNER_FOLDS = 5

TARGET = "wRC+_t1"  # default target; see TARGETS for the full registry
GROUP_COL = "player_id"


@dataclass(frozen=True)
class Target:
    """One prediction target and the scalar baseline it is compared against.

    ``excludes_2020`` marks targets that are *counting* quantities. The 2020
    season was 60 games, so a counting outcome for 2020 lives on a different
    scale than every other season and a model cannot know that in advance. Such
    targets drop every transition touching 2020 rather than pretending the
    shortened season is comparable. Rate targets are unaffected -- a rate is a
    rate whatever the schedule.
    """

    key: str
    column: str  # the t+1 column being predicted
    baseline: str  # the t column used as the M0 scalar baseline
    label: str
    is_rate: bool
    excludes_2020: bool


TARGETS: dict[str, Target] = {
    "wrc_plus": Target(
        key="wrc_plus",
        column="wRC+_t1",
        baseline="wRC+_t",
        label="next-season wRC+",
        is_rate=True,
        excludes_2020=False,
    ),
    # The five tools include Defense and Speed, which wRC+ does not price at all.
    # WAR does, so it is the target on which the tool structure has something to
    # gain that a batting-only index cannot show.
    "war": Target(
        key="war",
        column="WAR_t1",
        baseline="WAR_t",
        label="next-season WAR (counting)",
        is_rate=False,
        excludes_2020=True,
    ),
    # WAR per 600 PA separates tool quality from playing time. WAR correlates
    # with PA at r = 0.674 against 0.449 for wRC+, so a large part of predicting
    # next-season WAR is predicting next-season playing time -- a roster and
    # injury question, not a question about the profile.
    #
    # ``excludes_2020=False`` because WAR per 600 PA is a rate, and the rule
    # stated on ``Target`` applies to it exactly as it applies to wRC+: a rate is
    # a rate whatever the schedule. Carrying ``True`` here (as an earlier version
    # did) silently evaluated the rate WAR target on three origins while wRC+ was
    # read on five, so the two headline claims did not rest on the same folds or
    # the same test cases. Task 6 still reports the 2020-free window for every
    # target, so the shortened season remains available as a sensitivity rather
    # than as an undisclosed restriction.
    "war_rate": Target(
        key="war_rate",
        column="WAR_per_600_t1",
        baseline="WAR_per_600_t",
        label="next-season WAR per 600 PA",
        is_rate=True,
        excludes_2020=False,
    ),
}

PRIMARY_TARGET = "wrc_plus"
WAR_RATE_DENOMINATOR = 600


def add_rate_columns(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Attach WAR per 600 PA, computed from the row's own season only."""
    out = player_seasons.copy()
    out["WAR_per_600"] = (
        WAR_RATE_DENOMINATOR * out["WAR"].astype(float) / out["PA"].astype(float)
    )
    return out


def drop_2020(transitions: pd.DataFrame) -> pd.DataFrame:
    """Remove every transition that touches the 60-game 2020 season."""
    return transitions[
        ~transitions["Season_t"].eq(2020) & ~transitions["Season_t1"].eq(2020)
    ].reset_index(drop=True)


def prepare_for_target(transitions: pd.DataFrame, target: Target) -> pd.DataFrame:
    """Apply the target's season restrictions."""
    return drop_2020(transitions) if target.excludes_2020 else transitions

TOOL_NAMES = ["Contact", "Power", "Discipline", "Defense", "Speed"]
# Encoding -> archived column prefix. All three exist in the archived table.
ENCODINGS = {"zscore": "ZScore", "pca": "PCA", "vae": "JointVAE"}

# The 22 constituent statistics, in tool order, exactly as tools.FEATURE_GROUPS
# defines them. Direction handling is delegated to tools.apply_direction.
CONSTITUENT_STATS: list[str] = [
    stat for group in FEATURE_GROUPS.values() for stat in group
]

# Display-name variants between the archived pybaseball fetch and the manual
# FanGraphs export. Each pair was confirmed to be the same player-season by an
# exact PA match; see decisions_log.md entry D3.
NAME_ALIASES: dict[str, str] = {
    "andrewyoung": "andyyoung",
    "clintfrazier": "jacksonfrazier",
    "josefmiranda": "josemiranda",
    "calmitchell": "calvinmitchell",
}

# Published Finding 2 anchors that this package must keep reproducing.
EXPECTED_N_TRAIN_FOLD_C = 1056
EXPECTED_N_TEST_FOLD_C = 358
EXPECTED_M0_MAE = 19.2195
EXPECTED_M1_MAE = 18.0267


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    """Accent-strip, lowercase and de-suffix a display name into a join key."""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", text)
    return re.sub(r"[^a-z0-9]", "", text)


def sanitize(stat: str) -> str:
    """Turn a FanGraphs stat label into a safe feature-column suffix."""
    return (
        stat.replace("%", "_pct")
        .replace("/", "_per_")
        .replace("-", "_")
        .replace(" ", "_")
    )


CONSTITUENT_COLS: list[str] = [f"cs_{sanitize(s)}" for s in CONSTITUENT_STATS]


def apply_b2v_scale(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Map season-standardized tool scores onto the configured B2V scale."""
    if B2V_SCALE == "zscore":
        return frame
    out = frame.copy()
    for column in columns:
        out[column] = grade_2080(out[column])
    return out


def tool_score_columns(frame: pd.DataFrame) -> list[str]:
    """Season-standardized encoder columns (``safe_<encoding>_<tool>...``)."""
    return [c for c in frame.columns if c.startswith("safe_")]


# --------------------------------------------------------------------------- #
# Player-season table
# --------------------------------------------------------------------------- #
def load_player_seasons(verbose: bool = True) -> pd.DataFrame:
    """Reconstruct player seasons from the committed, public transition table.

    Tasks 1--5 consume only adjacent transitions.  Keeping the transition table
    as their canonical input removes the former cross-project dependency and
    on a private raw export while preserving every modeled row and value.
    """
    transitions = pd.read_csv(TRANSITIONS_PATH)
    frames = []
    for suffix in ("_t", "_t1"):
        columns = {c: c[: -len(suffix)] for c in transitions if c.endswith(suffix)}
        frame = transitions[["player_id", *columns]].rename(columns=columns)
        frames.append(frame)
    players = pd.concat(frames, ignore_index=True)
    players = players.sort_values(
        ["player_id", "Season", "source_row"], kind="mergesort"
    ).drop_duplicates(["player_id", "Season"], keep="first")
    assert not players.duplicated(["player_id", "Season"]).any()
    players = apply_b2v_scale(players, tool_score_columns(players))
    if verbose:
        print(
            f"  loaded {len(transitions)} public transitions from "
            f"{TRANSITIONS_PATH.relative_to(SSAC_ROOT)}"
        )
    return add_rate_columns(players).reset_index(drop=True)


def load_transitions() -> pd.DataFrame:
    """Load the canonical public transition table without rebuilding it."""
    frame = pd.read_csv(TRANSITIONS_PATH)
    frame = apply_b2v_scale(frame, tool_score_columns(frame))
    assert len(frame) == 1414
    assert not frame.duplicated(["player_id", "Season_t", "Season_t1"]).any()
    assert (frame["Season_t1"] == frame["Season_t"] + 1).all()
    return frame


# --------------------------------------------------------------------------- #
# Season-to-season transitions
# --------------------------------------------------------------------------- #
CARRY_COLS = [
    "Name", "Season", "PA", "WAR", "WAR_per_600", "wRC+", "OPS", "age_t", "birth_year",
]


def build_transitions(players: pd.DataFrame) -> pd.DataFrame:
    """Adjacent (t, t+1) season pairs keyed on the stable player ID."""
    feature_cols = [
        f"safe_{prefix}_{tool}" for prefix in ENCODINGS.values() for tool in TOOL_NAMES
    ]
    keep = [*CARRY_COLS, "source_row", *feature_cols, *CONSTITUENT_COLS]
    rows: list[dict] = []
    for player_id, player in players.groupby("player_id", sort=True):
        player = player.sort_values(["Season", "source_row"], kind="mergesort")
        records = player[keep].to_dict("records")
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
    assert (out["birth_year_t"] == out["birth_year_t1"]).all()
    assert (out["age_t_t"] == out["Season_t"] - out["birth_year_t"]).all()
    return out.sort_values(
        ["Season_t", "player_id"], kind="mergesort"
    ).reset_index(drop=True)


def assert_no_future_columns(features: list[str]) -> None:
    """Guard: a predictor may never come from the outcome season."""
    for name in features:
        assert not name.endswith("_t1"), f"future-season predictor: {name}"
        assert "2025" not in name, f"hard-coded outcome season in predictor: {name}"


# --------------------------------------------------------------------------- #
# Feature sets
# --------------------------------------------------------------------------- #
def tool_features(encoding: str = "zscore") -> list[str]:
    prefix = ENCODINGS[encoding]
    return [f"safe_{prefix}_{tool}_t" for tool in TOOL_NAMES]


def constituent_features() -> list[str]:
    return [f"{col}_t" for col in CONSTITUENT_COLS]


CONTEXT_FEATURES = ["age_t_t", "PA_t"]


def model_specs(
    encoding: str = "zscore", target: Target | None = None
) -> dict[str, list[str]]:
    """The six Task 1 models, in report order.

    Model names stay target-agnostic ("M0 scalar baseline" rather than
    "M0 current wRC+") so the same labels line up across targets in every table.
    """
    target = target or TARGETS[PRIMARY_TARGET]
    tools = tool_features(encoding)
    stats = constituent_features()
    return {
        "M0 scalar baseline": [target.baseline],
        "M1 B2V 5 tools": tools,
        "M2 constituent stats": stats,
        "M3 scalar + age + PA": [target.baseline, *CONTEXT_FEATURES],
        "M4 B2V + age + PA": [*tools, *CONTEXT_FEATURES],
        "M5 constituents + age + PA": [*stats, *CONTEXT_FEATURES],
    }


# --------------------------------------------------------------------------- #
# Model fitting
# --------------------------------------------------------------------------- #
def ridge_pipeline(features: list[str], alpha: float) -> Pipeline:
    """Train-only median imputation -> train-only standardization -> Ridge."""
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("ridge", Ridge(alpha=alpha))])


def select_alpha(
    train: pd.DataFrame,
    features: list[str],
    alphas: np.ndarray,
    target: str = TARGET,
) -> tuple[float, pd.DataFrame]:
    # `target` is a column name here, not a Target object.
    """Player-grouped 5-fold CV on the training window only. Ties -> smaller alpha."""
    splitter = GroupKFold(n_splits=INNER_FOLDS)
    folds = list(splitter.split(train, train[target], train[GROUP_COL]))
    rows: list[dict] = []
    for alpha in alphas:
        fold_mae = []
        for fold, (fit_idx, valid_idx) in enumerate(folds):
            model = ridge_pipeline(features, float(alpha))
            model.fit(train.iloc[fit_idx][features], train.iloc[fit_idx][target])
            pred = model.predict(train.iloc[valid_idx][features])
            mae = float(mean_absolute_error(train.iloc[valid_idx][target], pred))
            fold_mae.append(mae)
            rows.append({"alpha": float(alpha), "inner_fold": str(fold), "mae": mae})
        rows.append(
            {"alpha": float(alpha), "inner_fold": "mean", "mae": float(np.mean(fold_mae))}
        )
    tuning = pd.DataFrame(rows)
    means = tuning[tuning["inner_fold"].eq("mean")]
    best = float(
        means.sort_values(["mae", "alpha"], kind="mergesort").iloc[0]["alpha"]
    )
    tuning["selected"] = tuning["alpha"].eq(best) & tuning["inner_fold"].eq("mean")
    return best, tuning


@dataclass
class FoldResult:
    fold: str
    model: str
    features: list[str]
    n_features: int
    n_train: int
    n_test: int
    train_period: str
    test_period: str
    selected_alpha: float
    predictions: pd.DataFrame  # player_id, Season_t, y_true, prediction


def run_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: dict[str, list[str]],
    alphas: np.ndarray,
    fold_label: str,
    target: str = TARGET,
) -> tuple[list[FoldResult], pd.DataFrame]:
    """Fit each spec on *train*, predict *test*. Nothing is fitted on *test*."""
    train_period = "; ".join(
        f"{a}->{b}"
        for a, b in sorted(set(zip(train["Season_t"], train["Season_t1"])))
    )
    test_period = "; ".join(
        f"{a}->{b}" for a, b in sorted(set(zip(test["Season_t"], test["Season_t1"])))
    )
    results: list[FoldResult] = []
    tuning_frames: list[pd.DataFrame] = []
    for model_name, features in specs.items():
        assert_no_future_columns(features)
        alpha, tuning = select_alpha(train, features, alphas, target=target)
        tuning.insert(0, "outer_fold", fold_label)
        tuning.insert(1, "model", model_name)
        tuning_frames.append(tuning)

        fitted = ridge_pipeline(features, alpha)
        fitted.fit(train[features], train[target])
        preds = test[["player_id", "Name", "Season_t", "Season_t1"]].copy()
        preds["y_true"] = test[target].to_numpy(float)
        preds["prediction"] = fitted.predict(test[features])
        preds["fold"] = fold_label
        preds["model"] = model_name
        results.append(
            FoldResult(
                fold=fold_label,
                model=model_name,
                features=features,
                n_features=len(features),
                n_train=len(train),
                n_test=len(test),
                train_period=train_period,
                test_period=test_period,
                selected_alpha=alpha,
                predictions=preds,
            )
        )
    return results, pd.concat(tuning_frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Metrics and clustered paired bootstrap
# --------------------------------------------------------------------------- #
def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def cluster_bootstrap_indices(
    ids: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> list[np.ndarray]:
    """Resample focal players with replacement, then take all their rows.

    Clustering on the focal player keeps the resampling unit independent when a
    player contributes test rows to more than one rolling-origin fold.
    """
    unique = np.unique(ids)
    lookup = {value: np.flatnonzero(ids == value) for value in unique}
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        samples.append(np.concatenate([lookup[value] for value in drawn]))
    return samples


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(clean, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap_difference(
    y: np.ndarray,
    pred_candidate: np.ndarray,
    pred_baseline: np.ndarray,
    ids: np.ndarray,
    samples: list[np.ndarray] | None = None,
) -> dict:
    """Baseline MAE minus candidate MAE; positive means the candidate is better."""
    if samples is None:
        samples = cluster_bootstrap_indices(ids)
    point = mae(y, pred_baseline) - mae(y, pred_candidate)
    draws = np.array(
        [
            mae(y[idx], pred_baseline[idx]) - mae(y[idx], pred_candidate[idx])
            for idx in samples
        ]
    )
    low, high = percentile_ci(draws)
    return {
        "improvement_mae": point,
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": bool(low > 0 or high < 0),
        "bootstrap_unit": "focal player (cluster)",
        "n_boot": len(samples),
        "direction": "baseline MAE minus candidate MAE",
        "positive_means_candidate_better": True,
    }


#: A fold whose training set is thinner than this is flagged in result tables.
THIN_TRAIN_THRESHOLD = 300


def attach_fit_metadata(
    predictions: pd.DataFrame, results: list[FoldResult]
) -> pd.DataFrame:
    """Copy each model's fitting metadata onto its prediction rows."""
    out = predictions.copy()
    for column in ("n_train", "n_features", "selected_alpha", "train_period",
                   "test_period"):
        out[column] = out["model"].map(
            {result.model: getattr(result, column) for result in results}
        )
    return out


def evaluate_predictions(
    predictions: pd.DataFrame,
    label: str,
    pairs: list[tuple[str, str]],
    pooled: bool = False,
    extra: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model MAE/RMSE and paired contrasts over one set of test rows.

    *predictions* is long-form: one row per (test case, model) with columns
    ``player_id``, ``Season_t``, ``y_true``, ``prediction``, ``model``, plus
    whatever fitting metadata :func:`attach_fit_metadata` added.

    When *pooled* is set the rows come from separately fitted models, so the
    per-fold fitting metadata has no single value and is reported as such rather
    than silently inheriting the first fold's.
    """
    pivot = predictions.pivot_table(
        index=["player_id", "Season_t"], columns="model", values="prediction"
    ).sort_index()
    truth = (
        predictions.drop_duplicates(["player_id", "Season_t"])
        .set_index(["player_id", "Season_t"])
        .loc[pivot.index, "y_true"]
    )
    assert pivot.notna().all().all()
    y = truth.to_numpy(float)

    # Cluster on the focal player: pooled across folds, one player can supply
    # more than one test row, so rows are not independent.
    focal = np.asarray([player_id for player_id, _ in pivot.index])
    samples = cluster_bootstrap_indices(focal)

    meta = predictions.drop_duplicates("model").set_index("model")

    def field(model: str, column: str):
        if column not in meta.columns:
            return pd.NA
        value = meta.loc[model, column]
        return pd.NA if (isinstance(value, float) and not np.isfinite(value)) else value

    base = dict(extra or {})
    rows: list[dict] = []
    for model in pivot.columns:
        pred = pivot[model].to_numpy(float)
        mae_draws = np.array([mae(y[i], pred[i]) for i in samples])
        rmse_draws = np.array([rmse(y[i], pred[i]) for i in samples])
        mae_low, mae_high = percentile_ci(mae_draws)
        rmse_low, rmse_high = percentile_ci(rmse_draws)
        n_train = field(model, "n_train")
        if pooled:
            fit_meta = {
                "n_features": field(model, "n_features"),
                "selected_alpha": pd.NA,
                "n_train": pd.NA,
                "thin_training_set": pd.NA,
                "train_period": "re-fitted per fold",
                "test_period": "all rolling-origin test seasons",
            }
        else:
            fit_meta = {
                "n_features": field(model, "n_features"),
                "selected_alpha": field(model, "selected_alpha"),
                "n_train": n_train,
                "thin_training_set": pd.NA
                if pd.isna(n_train)
                else int(n_train) < THIN_TRAIN_THRESHOLD,
                "train_period": field(model, "train_period"),
                "test_period": field(model, "test_period"),
            }
        rows.append(
            {
                **base,
                "fold": label,
                "model": model,
                **fit_meta,
                "n_test": len(y),
                "mae": mae(y, pred),
                "mae_ci_low": mae_low,
                "mae_ci_high": mae_high,
                "rmse": rmse(y, pred),
                "rmse_ci_low": rmse_low,
                "rmse_ci_high": rmse_high,
            }
        )

    paired: list[dict] = []
    for candidate, baseline in pairs:
        record = paired_bootstrap_difference(
            y,
            pivot[candidate].to_numpy(float),
            pivot[baseline].to_numpy(float),
            focal,
            samples=samples,
        )
        paired.append(
            {
                **base,
                "fold": label,
                "candidate": candidate,
                "baseline": baseline,
                "n_test": len(y),
                "sign": "candidate better"
                if record["improvement_mae"] > 0
                else "baseline better",
                **record,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(paired)


def sign_consistency(
    paired: pd.DataFrame,
    pairs: list[tuple[str, str]],
    by: str = "alpha_grid",
    folds: list[str] | None = None,
) -> pd.DataFrame:
    """Does every rolling-origin fold agree on the direction of each contrast?"""
    folds = list(ROLLING_FOLDS) if folds is None else list(folds)
    rows: list[dict] = []
    for key in paired[by].unique():
        for candidate, baseline in pairs:
            block = paired[
                paired[by].eq(key)
                & paired["candidate"].eq(candidate)
                & paired["baseline"].eq(baseline)
                & paired["fold"].isin(folds)
            ].set_index("fold")
            deltas = {f: float(block.loc[f, "improvement_mae"]) for f in folds}
            excludes = {f: bool(block.loc[f, "excludes_zero"]) for f in folds}
            signs = {f: int(np.sign(v)) for f, v in deltas.items()}
            rows.append(
                {
                    by: key,
                    "candidate": candidate,
                    "baseline": baseline,
                    **{f"delta_mae_fold_{f}": deltas[f] for f in folds},
                    **{f"ci_excludes_zero_fold_{f}": excludes[f] for f in folds},
                    "n_folds": len(folds),
                    "n_folds_candidate_better": sum(v > 0 for v in signs.values()),
                    "all_folds_same_sign": len(set(signs.values())) == 1,
                    "all_three_same_sign": len(set(signs.values())) == 1,
                    "n_folds_ci_excludes_zero": sum(excludes.values()),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
#: Rolling-origin folds. Training uses only transitions whose *input* season is
#: strictly earlier than the test fold's input season.
ROLLING_FOLDS: dict[str, tuple[list[int], int]] = {
    "A": ([2021], 2022),
    "B": ([2021, 2022], 2023),
    "C": ([2021, 2022, 2023], 2024),
}


def split_by_input_season(
    transitions: pd.DataFrame, train_inputs: list[int], test_input: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on transitions whose input season is strictly earlier than *test_input*."""
    assert max(train_inputs) < test_input, (train_inputs, test_input)
    train = transitions[transitions["Season_t"].isin(train_inputs)].copy()
    test = transitions[transitions["Season_t"].eq(test_input)].copy()
    assert int(train["Season_t1"].max()) <= test_input, "training outcome leaks"
    assert test["player_id"].nunique() == len(test)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_fold(
    transitions: pd.DataFrame, fold: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_inputs, test_input = ROLLING_FOLDS[fold]
    train = transitions[transitions["Season_t"].isin(train_inputs)].copy()
    test = transitions[transitions["Season_t"].eq(test_input)].copy()
    assert int(train["Season_t1"].max()) <= test_input, "training outcome leaks"
    assert test["player_id"].nunique() == len(test)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def environment_stamp() -> dict:
    import platform

    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "seed": SEED,
        "n_boot": N_BOOT,
        "b2v_scale": B2V_SCALE,
        "alpha_grid": "np.logspace(-3, 4, 50)",
        "inner_cv": f"GroupKFold(n_splits={INNER_FOLDS}) on player_id",
        "input_snapshot_sha256": sha256_file(INTERMEDIATE_DIR / "submission_inputs_2019_2025.csv"),
    }
