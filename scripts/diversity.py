"""Descriptive matching and joint player-cluster uncertainty; no manuscript output."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results_sd10'
OUT=RESULTS/'diversity'
TOOLS=['Power','Contact','Discipline','Defense','Speed']
SEED=42
N_BOOT=2000

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def relative_distance_reduction(summary, draws):
    """Ratio of group medians, with CI from joint focal-player draws.

    This is a descriptive comparison, not an effect caused by matching or a
    fraction of information/variance removed. Random pairs alone enforce PA strata.
    """
    medians = summary.set_index('pair_type')['median']
    random = 'random_same_season_pa_stratum'
    assert medians[random] > 0 and draws[random].gt(0).all()
    reduction = 1 - draws['scalar_matched'] / draws[random]
    low, high = np.quantile(reduction, [.025, .975])
    return dict(estimate=float(1 - medians['scalar_matched'] / medians[random]),
                ci_low=float(low), ci_high=float(high), n_boot=len(draws),
                definition='1 - matched median / random median',
                bootstrap_unit='focal stable player ID; joint resamples across groups',
                interpretation='relative difference of medians, not median of pairwise ratios or causal removal')

def reference_gap_closure(summary, draws):
    """Post-feedback descriptive ratio of medians using joint player draws.

    Not an equivalence statistic, causal matching effect, or explained variance.
    No resampling, refitting, truncation, or selection of favorable draws.
    """
    m = summary.set_index('pair_type')['median']
    random, matched, same = 'random_same_season_pa_stratum', 'scalar_matched', 'same_player_adjacent'
    denominator = m[random] - m[same]
    denominators = draws[random] - draws[same]
    assert denominator > 0 and denominators.gt(0).all()
    fraction = (draws[random] - draws[matched]) / denominators
    low, high = np.quantile(fraction, [.025, .975])
    return dict(estimate=float((m[random]-m[matched])/denominator),
                ci_low=float(low), ci_high=float(high), n_boot=len(draws),
                denominator=float(denominator), min_bootstrap_denominator=float(denominators.min()),
                definition='(random median - matched median) / (random median - same-hitter median)',
                bootstrap_unit='focal stable player ID; existing joint draws across all three groups',
                interpretation='post-feedback descriptive gap fraction; not causal or explained information')

def build_diversity(cohort):
    # Reuse the original deterministic matching, random pairing and bootstrap.
    # This script does not import SSAC's same-named baseball2vec package.
    sys.path.insert(0, str(ROOT / 'src'))
    from baseball2vec import matching as m
    players = cohort.sort_values(['Season', 'player_id']).reset_index(drop=True).copy()
    players['source_row'] = np.arange(len(players))
    for tool in TOOLS:
        players[f'ZScore_{tool}'] = players[f'b2v_{tool}']
    stds = players[m.TOOL_COLS].std(ddof=1).to_numpy(float)
    matches, flow = m.deterministic_matches(players, stds)
    lookup = players.set_index(['player_id', 'Season'], drop=False)
    # All three groups use exactly the same focal player-seasons. A later season
    # is needed for the within-player reference, not to construct the profile.
    has_next = [(r.focal_player_id, r.focal_season + 1) in lookup.index
                for r in matches.itertuples()]
    matched = matches.loc[has_next].copy()
    random = m.random_pairs_for_matched_focals(players, matched, stds)
    adjacent = pd.DataFrame([
        m.pair_row('same_player_adjacent', lookup.loc[(r.focal_player_id, r.focal_season)],
                   lookup.loc[(r.focal_player_id, r.focal_season + 1)], stds)
        for r in matched.itertuples()
    ])
    pairs = pd.concat([adjacent, matched, random], ignore_index=True)
    summary, differences, draws = m.clustered_summary(
        pairs, np.sort(matched.focal_player_id.unique()), return_draws=True)
    draws.to_csv(OUT / 'bootstrap_medians.csv.gz', index=False, compression='gzip')
    ratio = relative_distance_reduction(summary, draws)
    closure = reference_gap_closure(summary, draws)
    pd.DataFrame([closure]).to_csv(OUT / 'reference_gap_closure.csv', index=False)
    pd.DataFrame([ratio]).to_csv(OUT / 'relative_distance_reduction.csv', index=False)
    for frame, name in [(pairs, 'pairs'), (summary, 'summary'), (differences, 'differences'), (flow, 'matching_flow')]:
        frame.to_csv(OUT / f'{name}.csv', index=False)
    pool = matched[matched.focal_pa.ge(400) & matched.comparison_pa.ge(400)].copy()
    q90 = pool.profile_distance.quantile(.90)
    pool['distance_to_q90'] = (pool.profile_distance - q90).abs()
    example = pool.sort_values(['distance_to_q90', 'standardized_scalar_gap', 'focal_player_id', 'comparison_player_id']).head(1)
    example.to_csv(OUT / 'illustrative_pair.csv', index=False)
    audit = {
        'cohort_player_seasons': len(players), 'cohort_years': [int(players.Season.min()), int(players.Season.max())],
        'all_matched_focals': len(matches), 'common_focals': len(matched),
        'common_focal_players': int(matched.focal_player_id.nunique()),
        'excluded_without_match': len(players) - len(matches),
        'matched_without_next_season': len(matches) - len(matched),
        'distance': 'RMS of five score differences divided by each tool pooled sample SD (ddof=1); clipped 20–80 coordinates',
        'tool_order': m.TOOL_COLS, 'pooled_tool_sds': stds.tolist(),
        'matching': 'Different same-season hitter; abs WAR gap <=0.2 and abs wRC+ gap <=5. Nearest season-standardized scalar gap; deterministic ID tie break; reuse allowed.',
        'random': 'One random same-season, different-ID hitter in the focal PA stratum (100–249, 250–499, 500+); same common focals.',
        'bootstrap': '2,000 joint focal-player cluster resamples; percentile CI of median differences; fixed matches and normalization; reused comparator dependence not separately clustered.',
        'illustration': 'Closest to the 90th percentile of matched distance among pairs with both PA>=400, then scalar gap and IDs; illustrative high-distance case, not typical or validation data.',
        'relative_distance_reduction': ratio,
        'reference_gap_closure': closure,
        'seed': SEED, 'n_boot': N_BOOT,
        'source_sha256': digest(RESULTS / 'task8/cohort.csv'),
        'matching_code_sha256': digest(ROOT / 'src/baseball2vec/matching.py'),
    }
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return summary, differences, example.iloc[0], audit
