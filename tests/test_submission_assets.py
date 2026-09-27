"""Check the submission's scientific claims against independent stored inputs."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results_sd10'
TOOLS = ['Power', 'Contact', 'Discipline', 'Defense', 'Speed']


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


def test_table_rows_use_latest_predictions_and_signed_paired_intervals():
    table = pd.read_csv(ROOT / 'abstract/tables/table1_prediction_tradeoffs.csv')
    assert len(table) == 24
    assert {'history+representative5', 'train_PCA5', 'full_PCA5', 'selected5', 'history+selected5'}.issubset(table.model)
    for row in table.itertuples():
        if row.model in ['selected5', 'history+selected5']:
            context = 'history' if row.model.startswith('history') else 'standalone'
            p = pd.read_csv(RESULTS / 'task10/performance.csv')
            p = p[p.target.eq(row.target) & p.window.eq('all') & p.context.eq(context) & p.fold.eq('pooled')].set_index('model')
            c = pd.read_csv(RESULTS / 'task10/contrasts.csv')
            r = c[c.target.eq(row.target) & c.window.eq('all') & c.context.eq(context) & c.candidate.eq('B2V') & c.baseline.eq('selected5')].iloc[0]
            np.testing.assert_allclose(row.mae, p.loc['selected5', 'mae'], atol=1e-12)
            np.testing.assert_allclose(row.B2V_improvement, p.loc['selected5', 'mae']-p.loc['B2V', 'mae'], atol=1e-10)
            np.testing.assert_allclose([row.B2V_improvement, row.ci_low, row.ci_high],
                                      [r.improvement_mae, r.ci_low, r.ci_high], atol=1e-12)
            assert row.n_test == 1706
            continue
        task = 'task9' if row.model == 'full_PCA5' else 'task8'
        p = pd.read_csv(RESULTS / task / 'model_performance.csv')
        p = p[p.target.eq(row.target) & p.window.eq('all') & p.fold.eq('pooled')].set_index('model')
        np.testing.assert_allclose(row.mae, p.loc[row.model, 'mae'], atol=1e-12)
        if row.model == row.reference:
            assert np.isnan(row.B2V_improvement)
            continue
        np.testing.assert_allclose(row.B2V_improvement, p.loc[row.model, 'mae']-p.loc[row.reference, 'mae'], atol=1e-10)
        c = pd.read_csv(RESULTS / task / 'model_contrasts.csv')
        c = c[c.target.eq(row.target) & c.window.eq('all') & c.fold.eq('pooled')]
        forward = c[c.candidate.eq(row.reference) & c.baseline.eq(row.model)]
        if len(forward):
            expected = forward.iloc[0][['improvement_mae', 'ci_low', 'ci_high']].to_numpy(float)
        else:
            r = c[c.candidate.eq(row.model) & c.baseline.eq(row.reference)].iloc[0]
            expected = [-r.improvement_mae, -r.ci_high, -r.ci_low]
        np.testing.assert_allclose([row.B2V_improvement, row.ci_low, row.ci_high], expected, atol=1e-12)


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
