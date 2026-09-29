"""Descriptive scientific checks; no abstract or manuscript assets required."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results_sd10'
TOOLS=['Power','Contact','Discipline','Defense','Speed']
pytestmark=pytest.mark.skipif(not (RESULTS/'diversity/pairs.csv').exists(), reason='Run reproduce.py --stage primary with local inputs')

def test_diversity_has_a_common_focal_sample_and_valid_matches():
    p = pd.read_csv(RESULTS / 'diversity/pairs.csv')
    keys = ['focal_player_id', 'focal_season']
    groups = {name: block for name, block in p.groupby('pair_type')}
    sets = [set(map(tuple, block[keys].to_numpy())) for block in groups.values()]
    assert sets[0] == sets[1] == sets[2]
    assert all(len(block) == len(sets[0]) for block in groups.values())
    matched = groups['scalar_matched']
    assert matched.abs_delta_war.le(.2 + 1e-12).all()
    assert matched.abs_delta_wrc.le(5 + 1e-12).all()
    for group in ['scalar_matched', 'random_same_season_pa_stratum']:
        b = groups[group]
        assert b.focal_season.eq(b.comparison_season).all()
        assert b.focal_player_id.ne(b.comparison_player_id).all()
    b = groups['same_player_adjacent']
    assert b.focal_player_id.eq(b.comparison_player_id).all()
    assert b.comparison_season.eq(b.focal_season + 1).all()

def test_all_distances_reconstruct_from_current_sd10_cohort():
    cohort = pd.read_csv(RESULTS / 'task8/cohort.csv').set_index(['player_id', 'Season'])
    pairs = pd.read_csv(RESULTS / 'diversity/pairs.csv')
    cols = [f'b2v_{t}' for t in TOOLS]
    sd = cohort[cols].std(ddof=1).to_numpy()
    left = cohort.loc[list(zip(pairs.focal_player_id, pairs.focal_season)), cols].to_numpy()
    right = cohort.loc[list(zip(pairs.comparison_player_id, pairs.comparison_season)), cols].to_numpy()
    np.testing.assert_allclose(np.sqrt(np.mean(((left-right)/sd)**2, axis=1)), pairs.profile_distance, atol=1e-12)
    for i, tool in enumerate(TOOLS):
        np.testing.assert_allclose(left[:, i], pairs[f'focal_ZScore_{tool}'], atol=1e-12)
        np.testing.assert_allclose(right[:, i], pairs[f'comparison_ZScore_{tool}'], atol=1e-12)

def test_diversity_statistics_and_example_selection():
    pairs = pd.read_csv(RESULTS / 'diversity/pairs.csv')
    summary = pd.read_csv(RESULTS / 'diversity/summary.csv').set_index('pair_type')
    diff = pd.read_csv(RESULTS / 'diversity/differences.csv')
    for group, block in pairs.groupby('pair_type'):
        r = summary.loc[group]
        np.testing.assert_allclose([r.q1, r['median'], r.q3], block.profile_distance.quantile([.25, .5, .75]))
        assert r.n_pairs == len(block)
        assert r.ci_low < r['median'] < r.ci_high
    for row in diff.itertuples():
        np.testing.assert_allclose(row.median_difference, summary.loc[row.left, 'median']-summary.loc[row.right, 'median'])
    pool = pairs[pairs.pair_type.eq('scalar_matched') & pairs.focal_pa.ge(400) & pairs.comparison_pa.ge(400)]
    example = pd.read_csv(RESULTS / 'diversity/illustrative_pair.csv').iloc[0]
    best = (pool.profile_distance-pool.profile_distance.quantile(.9)).abs().min()
    np.testing.assert_allclose(example.distance_to_q90, best)

def test_relative_distance_reduction_uses_joint_cluster_draws():
    pairs = pd.read_csv(RESULTS / 'diversity/pairs.csv')
    summary = pd.read_csv(RESULTS / 'diversity/summary.csv').set_index('pair_type')
    draws = pd.read_csv(RESULTS / 'diversity/bootstrap_medians.csv.gz')
    result = pd.read_csv(RESULTS / 'diversity/relative_distance_reduction.csv').iloc[0]
    random = 'random_same_season_pa_stratum'
    expected = 1 - summary.loc['scalar_matched', 'median'] / summary.loc[random, 'median']
    np.testing.assert_allclose(result.estimate, expected, atol=1e-12)
    values = 1 - draws.scalar_matched / draws[random]
    np.testing.assert_allclose([result.ci_low, result.ci_high], values.quantile([.025, .975]), atol=1e-12)
    assert result.n_boot == len(draws) == 2000
    # Independently reconstruct joint player resampling using explicit repeats.
    ids = np.sort(pairs.focal_player_id.unique())
    rng = np.random.default_rng(42)
    for i in range(20):
        sampled = rng.choice(ids, len(ids), replace=True)
        counts = pd.Series(sampled).value_counts()
        for group, block in pairs.groupby('pair_type'):
            weights = block.focal_player_id.map(counts).fillna(0).to_numpy(int)
            repeated = np.repeat(block.profile_distance.to_numpy(), weights)
            median = np.quantile(repeated, .5, method='inverted_cdf')
            np.testing.assert_allclose(draws.loc[i, group], median, atol=1e-12)
    # The new statistic must not change the existing distance estimates/CIs.
    differences = pd.read_csv(RESULTS / 'diversity/differences.csv')
    for r in differences.itertuples():
        np.testing.assert_allclose([r.ci_low, r.ci_high],
                                  (draws[r.left]-draws[r.right]).quantile([.025, .975]), atol=1e-12)

def test_reference_gap_closure_uses_unrounded_medians_and_joint_draws():
    summary = pd.read_csv(RESULTS / 'diversity/summary.csv').set_index('pair_type')['median']
    draws = pd.read_csv(RESULTS / 'diversity/bootstrap_medians.csv.gz')
    result = pd.read_csv(RESULTS / 'diversity/reference_gap_closure.csv').iloc[0]
    r, m, h = 'random_same_season_pa_stratum', 'scalar_matched', 'same_player_adjacent'
    expected = (summary[r] - summary[m]) / (summary[r] - summary[h])
    denom = draws[r] - draws[h]
    assert denom.gt(0).all()
    samples = (draws[r] - draws[m]) / denom
    np.testing.assert_allclose(result.estimate, expected, atol=1e-12)
    np.testing.assert_allclose([result.ci_low, result.ci_high], samples.quantile([.025,.975]), atol=1e-12)
    assert result.n_boot == len(draws) == 2000
