"""Data loading, caching, and preprocessing for Baseball2Vec."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

RAW_CACHE = Path(__file__).parents[2] / "data" / "raw" / "batting_stats_2021_2025.csv"
YEARS = [2021, 2022, 2023, 2024, 2025]

# Bayesian stabilization regression-to-the-mean thresholds (PA equivalents).
# Zero means no stabilization (metric is already stable or scale-invariant).
STABILIZATION_THRESHOLDS: dict[str, int] = {
    "Contact%": 100,
    "K%": 60,
    "BB%": 120,
    "AVG": 500,
    "SLG": 400,
    "ISO": 160,
    "HardHit%": 50,
    "Barrel%": 50,
    "O-Swing%": 100,
    "Swing%": 100,
    "Spd": 0,
    "MaxEV": 0,
    "Def": 0,
    "BsR": 0,
    "UBR": 0,
    "wSB": 0,
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns (except Name/Team) to float, stripping '%' signs."""
    for col in df.columns:
        if df[col].dtype == object and col not in ("Name", "Team"):
            try:
                df[col] = df[col].astype(str).str.replace("%", "").astype(float)
            except (ValueError, TypeError):
                pass
    return df


def _stabilize_row(
    row: pd.Series,
    season_means: pd.DataFrame,
    thresholds: dict[str, int],
) -> pd.Series:
    """Apply Bayesian (regression-to-the-mean) stabilization to one row."""
    pa = row["PA"]
    means = season_means.loc[row["Season"]]
    for col, threshold in thresholds.items():
        if col not in row.index or threshold == 0:
            continue
        league_avg = means.get(col, 0)
        original_val = row[col]
        # Handle cases where the raw value is on a 0-100 scale but the mean is 0-1
        if original_val > 1 and 0 <= league_avg < 1:
            league_avg *= 100
        row[col] = (original_val * pa + league_avg * threshold) / (pa + threshold)
    return row


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, apply Bayesian stabilization, and add UniqueName column.

    Args:
        raw_df: Output of :func:`load_raw`.

    Returns:
        Preprocessed DataFrame ready for tool scoring.
    """
    df = _coerce_numeric(raw_df.copy())

    print("  Applying Bayesian stabilization...")
    season_means = df.groupby("Season").mean(numeric_only=True)
    df = df.apply(
        _stabilize_row,
        axis=1,
        season_means=season_means,
        thresholds=STABILIZATION_THRESHOLDS,
    )
    df["UniqueName"] = df["Name"] + " (" + df["Season"].astype(str) + ")"
    return df
