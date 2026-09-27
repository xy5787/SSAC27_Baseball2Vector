"""Deterministic same-season scalar matching and focal-player clustered uncertainty."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

N_BOOT = 2000
SEED = 42
TOOL_COLS = ['ZScore_Contact', 'ZScore_Power', 'ZScore_Discipline', 'ZScore_Defense', 'ZScore_Speed']

def percentile_ci(values):
    return tuple(float(v) for v in np.quantile(values, [.025, .975]))

WRC_TOLERANCE = 5.0
WAR_TOLERANCE = 0.2


def pa_stratum(pa: float) -> str:
    if pa < 250:
        return "100-249"
    if pa < 500:
        return "250-499"
    return "500+"


def profile_distance(left: pd.Series, right: pd.Series, stds: np.ndarray) -> float:
    delta = (left[TOOL_COLS].to_numpy(float) - right[TOOL_COLS].to_numpy(float)) / stds
    return float(np.linalg.norm(delta) / math.sqrt(len(TOOL_COLS)))


def pair_row(pair_type: str, focal: pd.Series, comparison: pd.Series, stds: np.ndarray, gap: float = np.nan) -> dict:
    row = {
        "pair_type": pair_type,
        "focal_player_id": focal["player_id"], "comparison_player_id": comparison["player_id"],
        "focal_name": focal["Name"], "comparison_name": comparison["Name"],
        "focal_source_row": int(focal["source_row"]), "comparison_source_row": int(comparison["source_row"]),
        "focal_season": int(focal["Season"]), "comparison_season": int(comparison["Season"]),
        "focal_pa": float(focal["PA"]), "comparison_pa": float(comparison["PA"]),
        "pa_stratum": pa_stratum(float(focal["PA"])),
        "focal_wrc": float(focal["wRC+"]), "comparison_wrc": float(comparison["wRC+"]),
        "focal_war": float(focal["WAR"]), "comparison_war": float(comparison["WAR"]),
        "abs_delta_wrc": float(abs(focal["wRC+"] - comparison["wRC+"])),
        "abs_delta_war": float(abs(focal["WAR"] - comparison["WAR"])),
        "standardized_scalar_gap": float(gap),
        "profile_distance": profile_distance(focal, comparison, stds),
    }
    for col in TOOL_COLS:
        row[f"focal_{col}"] = float(focal[col])
        row[f"comparison_{col}"] = float(comparison[col])
    return row


def deterministic_matches(players: pd.DataFrame, stds: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs: list[dict] = []
    flow: list[dict] = []
    for season, cohort in players.groupby("Season", sort=True):
        cohort = cohort.sort_values(["player_id", "source_row"], kind="mergesort")
        wrc_sd = float(cohort["wRC+"].std(ddof=1))
        war_sd = float(cohort["WAR"].std(ddof=1))
        successful = 0
        for _, focal in cohort.iterrows():
            candidates = cohort[~cohort["player_id"].eq(focal["player_id"])].copy()
            candidates["_dwrc"] = (candidates["wRC+"] - focal["wRC+"]).abs()
            candidates["_dwar"] = (candidates["WAR"] - focal["WAR"]).abs()
            candidates = candidates[
                (candidates["_dwrc"] <= WRC_TOLERANCE) & (candidates["_dwar"] <= WAR_TOLERANCE)
            ].copy()
            if candidates.empty:
                continue
            candidates["_gap"] = np.sqrt((candidates["_dwrc"] / wrc_sd) ** 2 + (candidates["_dwar"] / war_sd) ** 2)
            candidates["_gap_sort"] = candidates["_gap"].round(12)
            selected = candidates.sort_values(
                ["_gap_sort", "player_id", "source_row"], kind="mergesort"
            ).iloc[0]
            pairs.append(pair_row("scalar_matched", focal, selected, stds, selected["_gap"]))
            successful += 1
        flow.append({"season": int(season), "candidate_focal_rows": len(cohort), "successfully_matched": successful, "unmatched_no_joint_candidate": len(cohort) - successful})
    return pd.DataFrame(pairs), pd.DataFrame(flow)


def random_pairs_for_matched_focals(players: pd.DataFrame, matched_pairs: pd.DataFrame, stds: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    lookup = players.set_index(["player_id", "Season"], drop=False)
    rows: list[dict] = []
    for matched in matched_pairs.sort_values(["focal_season", "focal_player_id"], kind="mergesort").itertuples(index=False):
        focal = lookup.loc[(matched.focal_player_id, matched.focal_season)]
        cohort = players[
            players["Season"].eq(matched.focal_season)
            & ~players["player_id"].eq(matched.focal_player_id)
            & players["PA"].map(pa_stratum).eq(matched.pa_stratum)
        ].sort_values(["player_id", "source_row"], kind="mergesort")
        assert not cohort.empty
        comparison = cohort.iloc[int(rng.integers(0, len(cohort)))]
        rows.append(pair_row("random_same_season_pa_stratum", focal, comparison, stds))
    return pd.DataFrame(rows)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cutoff = q * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def clustered_summary(pairs: pd.DataFrame, all_player_ids: np.ndarray, *, return_draws: bool = False):
    groups = ["same_player_adjacent", "scalar_matched", "random_same_season_pa_stratum"]
    rng = np.random.default_rng(SEED)
    bootstrap = {group: [] for group in groups}
    for _ in range(N_BOOT):
        sampled = rng.choice(all_player_ids, len(all_player_ids), replace=True)
        ids, counts = np.unique(sampled, return_counts=True)
        weight_map = dict(zip(ids, counts))
        for group in groups:
            sub = pairs[pairs["pair_type"].eq(group)]
            selected = sub[sub["focal_player_id"].isin(weight_map)]
            weights = selected["focal_player_id"].map(weight_map).to_numpy(float)
            bootstrap[group].append(weighted_quantile(selected["profile_distance"].to_numpy(float), weights, 0.5))
    summary = []
    for group in groups:
        sub = pairs[pairs["pair_type"].eq(group)]
        low, high = percentile_ci(bootstrap[group])
        summary.append({
            "pair_type": group, "n_pairs": len(sub),
            "n_unique_focal_players": sub["focal_player_id"].nunique(),
            "n_unique_comparison_players": sub["comparison_player_id"].nunique(),
            "median": sub["profile_distance"].median(),
            "q1": sub["profile_distance"].quantile(0.25), "q3": sub["profile_distance"].quantile(0.75),
            "ci_low": low, "ci_high": high,
            "bootstrap_unit": "focal stable player ID", "n_boot": N_BOOT,
        })
    differences = []
    for left, right in [("scalar_matched", "same_player_adjacent"), ("random_same_season_pa_stratum", "scalar_matched")]:
        values = np.asarray(bootstrap[left]) - np.asarray(bootstrap[right])
        low, high = percentile_ci(values)
        point = pairs.loc[pairs["pair_type"].eq(left), "profile_distance"].median() - pairs.loc[pairs["pair_type"].eq(right), "profile_distance"].median()
        differences.append({"left": left, "right": right, "median_difference": point, "ci_low": low, "ci_high": high, "n_boot": N_BOOT})
    result = (pd.DataFrame(summary), pd.DataFrame(differences))
    # Optional joint draws preserve the exact resamples for ratio statistics.
    return (*result, pd.DataFrame(bootstrap)) if return_draws else result

