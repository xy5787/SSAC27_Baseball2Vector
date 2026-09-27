"""Z-Score averaging and PCA baseline tool-score methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .tools import ALIGN_ANCHORS, TOOL_NAMES, pentagon_area, scale_to_2080


def zscore_tool_scores(
    X_aligned: pd.DataFrame,
    final_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute tool scores by averaging within-group Z-Scores.

    Args:
        X_aligned: Direction-corrected Z-scored feature DataFrame.
        final_groups: Tool → column mapping from :func:`tools.build_final_groups`.

    Returns:
        DataFrame with one column per tool.
    """
    raw: dict[str, np.ndarray] = {}
    for group, cols in final_groups.items():
        raw[group] = X_aligned[cols].mean(axis=1).values
    return pd.DataFrame(raw)


def pca_tool_scores(
    X_aligned: pd.DataFrame,
    final_groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute tool scores as the first principal component of each group.

    Sign is flipped when PC1 correlates negatively with the anchor statistic
    so that all tools point "higher = better".

    Args:
        X_aligned: Direction-corrected Z-scored feature DataFrame.
        final_groups: Tool → column mapping from :func:`tools.build_final_groups`.

    Returns:
        Tuple of (tool scores DataFrame, dict of explained variance ratios).
    """
    raw: dict[str, np.ndarray] = {}
    explained: dict[str, float] = {}

    for group, cols in final_groups.items():
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_aligned[cols]).flatten()
        explained[group] = float(pca.explained_variance_ratio_[0])

        anchor = ALIGN_ANCHORS.get(group)
        if anchor and anchor in X_aligned.columns:
            if np.corrcoef(pc1, X_aligned[anchor].values)[0, 1] < 0:
                pc1 = -pc1

        raw[group] = pc1

    return pd.DataFrame(raw), explained


def build_scaled_results(
    methods: dict[str, pd.DataFrame],
    tool_names: list[str] = TOOL_NAMES,
) -> dict[str, pd.DataFrame]:
    """Scale each method's raw tool scores to 20–80 and compute Pentagon areas.

    Args:
        methods: Dict mapping method name → raw tool score DataFrame.
        tool_names: Ordered list of tool column names.

    Returns:
        Dict mapping method name → DataFrame with prefixed tool columns and an
        ``{method}_Area`` column.
    """
    results: dict[str, pd.DataFrame] = {}
    for name, raw in methods.items():
        scaled = scale_to_2080(raw, tool_names)
        out = scaled.copy()
        out.columns = [f"{name}_{c}" for c in tool_names]
        out[f"{name}_Area"] = scaled.apply(pentagon_area, axes=tool_names, axis=1)
        results[name] = out
    return results


def correlation_table(
    result_df: pd.DataFrame,
    method_names: list[str],
    validation_metrics: list[str] = ("WAR", "wRC+", "OPS"),
) -> dict[str, dict[str, float]]:
    """Pearson correlation of each method's Pentagon Area against performance metrics.

    Args:
        result_df: Combined result DataFrame (must contain ``{method}_Area`` columns).
        method_names: Method names to evaluate.
        validation_metrics: Performance columns to correlate against.

    Returns:
        Nested dict: method → metric → Pearson r.
    """
    table: dict[str, dict[str, float]] = {}
    for method in method_names:
        area_col = f"{method}_Area"
        table[method] = {
            metric: float(result_df[area_col].corr(result_df[metric]))
            for metric in validation_metrics
            if metric in result_df.columns
        }
    return table
