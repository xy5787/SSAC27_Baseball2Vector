"""Five-tool feature definitions, direction normalization, and scaling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 5-Tool feature groups. Columns are filtered at runtime against the actual DataFrame.
FEATURE_GROUPS: dict[str, list[str]] = {
    "Contact": ["Contact%", "K%", "AVG", "xBA", "SwStr%"],
    "Power": ["ISO", "SLG", "HardHit%", "Barrel%", "maxEV", "EV", "HR/FB"],
    "Speed": ["Spd", "BsR", "UBR", "wSB"],
    "Defense": ["Def", "Fld"],
    "Discipline": ["BB%", "O-Swing%", "Swing%", "BB/K"],
}

TOOL_NAMES: list[str] = list(FEATURE_GROUPS.keys())

# Metrics where lower is better → sign-flip so that "higher = better" throughout.
REVERSE_COLS: list[str] = ["K%", "SwStr%", "O-Swing%", "Swing%"]

# One representative stat per tool used to verify latent alignment direction.
ALIGN_ANCHORS: dict[str, str] = {
    "Contact": "Contact%",
    "Power": "ISO",
    "Speed": "Spd",
    "Defense": "Def",
    "Discipline": "BB%",
}


def build_final_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Filter FEATURE_GROUPS to columns that exist in *df*.

    Args:
        df: Preprocessed batting stats DataFrame.

    Returns:
        Dict mapping tool name → list of available column names.

    Raises:
        AssertionError: If fewer than 5 tools survive filtering.
    """
    final: dict[str, list[str]] = {}
    for group, cols in FEATURE_GROUPS.items():
        valid = [c for c in cols if c in df.columns]
        if valid:
            final[group] = valid
    assert len(final) == 5, (
        f"Expected 5 tools after filtering, got {len(final)}: {list(final)}"
    )
    return final


def season_zscore(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Apply within-season Z-Score normalization to *features*.

    Args:
        df: Preprocessed DataFrame (must have a ``Season`` column).
        features: Column names to normalize.

    Returns:
        Copy of *df* with only the Z-scored feature columns, NaN → 0.
    """
    X = df.copy()
    for col in features:
        if col in X.columns:
            X[col] = X.groupby("Season")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
    return X[features].fillna(0)


def apply_direction(X_df: pd.DataFrame) -> pd.DataFrame:
    """Flip sign of REVERSE_COLS so that all features are "higher = better".

    Args:
        X_df: Z-scored feature DataFrame.

    Returns:
        Copy with reversed columns negated.
    """
    X = X_df.copy()
    for c in REVERSE_COLS:
        if c in X.columns:
            X[c] = -X[c]
    return X


def scale_to_2080(scores_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Min-max scale tool scores to the 20–80 scouting scale.

    Args:
        scores_df: DataFrame with raw tool score columns.
        cols: Column names to scale.

    Returns:
        New DataFrame with the same columns scaled to [20, 80].
    """
    scaler = MinMaxScaler(feature_range=(20, 80))
    return pd.DataFrame(
        scaler.fit_transform(scores_df[cols]),
        columns=cols,
        index=scores_df.index,
    )


def grade_2080(z: pd.Series, mean: float = 50.0, sd: float = 10.0) -> pd.Series:
    """Map standardized scores onto the 20–80 scouting scale (50 + 10 z, clipped).

    Same convention as ``SD10Pipeline/scaling_sd10.py``, but it takes scores
    already standardized within their season, so no season sees another's
    mean or SD.

    Args:
        z: Scores standardized within their season (mean 0, SD 1).
        mean: Grade assigned to an average player.
        sd: Grade points per standard deviation.

    Returns:
        Grades clipped to [20, 80].
    """
    return (mean + sd * z.astype(float)).clip(20, 80)


def pentagon_area(row: pd.Series, axes: list[str]) -> float:
    """Compute the area of a regular pentagon from five axis values.

    Uses the formula: A = (1/2) * Σ r_i * r_{i+1} * sin(2π/5)

    Args:
        row: Series containing the five axis values.
        axes: Ordered list of five column names.

    Returns:
        Pentagon area.
    """
    assert len(axes) == 5
    sine_val = np.sin(2 * np.pi / 5)
    values = [row[col] for col in axes]
    return sum(0.5 * values[i] * values[(i + 1) % 5] * sine_val for i in range(5))
