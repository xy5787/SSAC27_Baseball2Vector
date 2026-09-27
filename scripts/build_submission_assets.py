"""Build the current submission figure/table from the frozen sd10 results.

No predictive model is refitted. Diversity is recomputed on a common focal
sample, using the existing SaberSeminar matching rules on the current cohort.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results_sd10'
OUT = RESULTS / 'diversity'
FIG = ROOT / 'abstract/figures'
TAB = ROOT / 'abstract/tables'
TOOLS = ['Power', 'Contact', 'Discipline', 'Defense', 'Speed']
COLS = [f'b2v_{t}' for t in TOOLS]
SEED = 42
N_BOOT = 2000


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
        'seed': SEED, 'n_boot': N_BOOT,
        'source_sha256': digest(RESULTS / 'task8/cohort.csv'),
        'matching_code_sha256': digest(ROOT / 'src/baseball2vec/matching.py'),
    }
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return summary, differences, example.iloc[0], audit


def save(fig, directory, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(directory / f'{name}.{ext}', dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def diversity_figure(summary, differences, example, audit):
    blue, orange, ink, gray = '#176387', '#BE6336', '#182C3B', '#61717D'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.8, 6.6), gridspec_kw={'width_ratios': [1.1, 1]})
    fig.subplots_adjust(left=.14, right=.96, top=.69, bottom=.28, wspace=.52)
    fig.text(.035, .965, 'Figure 1 | Similar value, distinct skill profiles', fontsize=18, weight='bold', color=ink)
    fig.text(.035, .91, 'B2V describes differences hidden by WAR and wRC+ summaries.', fontsize=12, color=gray)
    ax.set_title('A   An illustrative matched pair', loc='left', pad=66, weight='bold', fontsize=12)
    for side, color, y in [('focal', blue, 1.15), ('comparison', orange, 1.045)]:
        ax.text(-.27, y, f"{example[f'{side}_name']} · {int(example[f'{side}_season'])}  |  "
                f"WAR {example[f'{side}_war']:.1f} · wRC+ {example[f'{side}_wrc']:.1f}",
                color=color, fontsize=10.5, transform=ax.transAxes, weight='bold')
    a = [example[f'focal_ZScore_{t}'] for t in TOOLS]
    b = [example[f'comparison_ZScore_{t}'] for t in TOOLS]
    for i, (v, w) in enumerate(zip(a, b)):
        ax.plot([v, w], [i, i], color='#CAD3DA', lw=4, zorder=1)
        ax.text(v, i-.16, f'{v:.1f}', ha='center', va='bottom', color=blue, fontsize=9)
        ax.text(w, i+.16, f'{w:.1f}', ha='center', va='top', color=orange, fontsize=9)
    ax.scatter(a, range(5), color=blue, s=65, zorder=3)
    ax.scatter(b, range(5), color=orange, marker='s', s=48, zorder=3)
    ax.set_yticks(range(5), ['Power', 'Contact', 'Plate discipline', 'Defense', 'Speed'])
    ax.set_ylim(4.6, -.6); ax.set_xlim(18, 82); ax.set_xticks(range(20, 81, 10))
    ax.axvline(50, color=gray, ls=':', lw=1)
    ax.set_xlabel('B2V score (20–80)', labelpad=10)
    bx.set_title('B   Distances across the common sample', loc='left', pad=66, weight='bold', fontsize=12)
    bx.text(0, 1.08, f"{audit['common_focals']:,} focal seasons · {audit['common_focal_players']:,} hitters", transform=bx.transAxes, fontsize=10.5, color=gray)
    groups = ['same_player_adjacent', 'scalar_matched', 'random_same_season_pa_stratum']
    names = ['Same hitter\nnext season', 'Different hitter\nmatched WAR + wRC+', 'Random hitter\nsame season + PA stratum']
    sm = summary.set_index('pair_type')
    for y, group in enumerate(groups):
        r = sm.loc[group]; color = blue if group == 'scalar_matched' else gray
        bx.plot([r.q1, r.q3], [y, y], color=color, alpha=.22, lw=15, solid_capstyle='butt')
        bx.errorbar(r['median'], y, xerr=[[r['median']-r.ci_low], [r.ci_high-r['median']]], fmt='o', color=color, capsize=5, markersize=6, lw=1.7)
        bx.text(r.q3+.055, y, f"{r['median']:.2f}", va='center', color=ink, weight='bold')
    bx.set_yticks(range(3), names); bx.set_ylim(2.65, -.65); bx.set_xlim(0, max(sm.q3)+.32)
    bx.set_xlabel('Standardized profile distance (RMS)', labelpad=10)
    for axis in [ax, bx]:
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.tick_params(axis='y', length=0); axis.grid(axis='x', color='#E5EAEE', lw=.7); axis.set_axisbelow(True)
    r = audit['relative_distance_reduction']
    fig.text(.035, .14, f"Matched-pair median distance was {100*r['estimate']:.1f}% lower than random (95% CI {100*r['ci_low']:.1f}–{100*r['ci_high']:.1f}%).", fontsize=11, weight='bold', color=ink)
    fig.text(.035, .10, 'B: dots = medians; thin capped bars = 95% cluster CIs; shaded bands = interquartile ranges. All groups share the same focal seasons.', fontsize=9, color=gray)
    fig.text(.035, .057, 'A: high-distance illustration selected near the 90th percentile among matched pairs with both PA ≥ 400; not a typical pair.', fontsize=9, color=gray)
    fig.text(.035, .014, '2019–2025 cohort, PA ≥ 100. Scores are constructed statistical profiles; scouting-grade validity and uniqueness versus other profiles are untested.', fontsize=9, color=gray)
    save(fig, FIG, 'figure1_player_diversity')


def prediction_table():
    perf = [pd.read_csv(RESULTS / t / 'model_performance.csv') for t in ['task8', 'task9']]
    cont = pd.concat([pd.read_csv(RESULTS / t / 'model_contrasts.csv') for t in ['task8', 'task9']])
    pooled = lambda f: f[f.window.eq('all') & f.fold.eq('pooled')]
    p8, p9 = [pooled(p).set_index(['target', 'model']) for p in perf]
    shared = p8.index.intersection(p9.index)
    np.testing.assert_allclose(p8.loc[shared, 'mae'], p9.loc[shared, 'mae'], atol=1e-9, rtol=0)
    p = pd.concat([p8, p9.loc[p9.index.get_level_values('model') == 'full_PCA5']])
    c = pooled(cont)
    p10 = pooled(pd.read_csv(RESULTS / 'task10/performance.csv'))
    c10 = pd.read_csv(RESULTS / 'task10/contrasts.csv')
    added_performance, added_contrasts = [], []
    for context, name, reference in [('standalone', 'selected5', 'B2V'),
                                     ('history', 'history+selected5', 'history+B2V')]:
        block = p10[p10.context.eq(context)]
        for target in ['wrc_plus', 'war_rate']:
            stored_reference = block[block.target.eq(target) & block.model.eq('B2V')].iloc[0]
            np.testing.assert_allclose(stored_reference.mae, p.loc[(target, reference), 'mae'], atol=1e-9)
            assert stored_reference.n_test == 1706
        chosen = block[block.model.eq('selected5')].copy()
        chosen['model'] = name
        added_performance.append(chosen.set_index(['target', 'model']))
        contrast = c10[c10.window.eq('all') & c10.context.eq(context) &
                       c10.candidate.eq('B2V') & c10.baseline.eq('selected5')].copy()
        assert len(contrast) == 2 and contrast.n_test.eq(1706).all()
        contrast['candidate'], contrast['baseline'] = reference, name
        added_contrasts.append(contrast)
    p = pd.concat([p, *added_performance])
    c = pd.concat([c, *added_contrasts])
    models = [('scalar', 'Current outcome (calibrated)', 1), ('representative5', 'Fixed representative statistics', 5),
              ('selected5', 'Training-selected representatives', 5),
              ('B2V', 'B2V', 5), ('stats22', 'All 22 inputs', 22),
              ('train_PCA5', 'Per-tool PCA', 5), ('full_PCA5', 'Unrestricted PCA', 5),
              ('history', 'History-based projection', 1), ('history+B2V', 'Projection + B2V', 6),
              ('history+representative5', 'Projection + fixed representatives', 6),
              ('history+selected5', 'Projection + selected representatives', 6),
              ('history+stats22', 'Projection + all 22 inputs', 23)]
    records, cells = [], []
    for model, label, k in models:
        candidate = 'history+B2V' if model.startswith('history') else 'B2V'
        cell = [label]
        for target in ['wrc_plus', 'war_rate']:
            row = p.loc[(target, model)]; assert row.n_test == 1706
            if model == candidate:
                delta, lo, hi = np.nan, np.nan, np.nan
            else:
                a = c[c.target.eq(target) & c.candidate.eq(candidate) & c.baseline.eq(model)]
                b = c[c.target.eq(target) & c.candidate.eq(model) & c.baseline.eq(candidate)]
                if len(a):
                    delta, lo, hi = a.iloc[0][['improvement_mae', 'ci_low', 'ci_high']]
                else:
                    assert len(b), (target, candidate, model)
                    v = b.iloc[0]; delta, lo, hi = -v.improvement_mae, -v.ci_high, -v.ci_low
                np.testing.assert_allclose(delta, row.mae - p.loc[(target, candidate), 'mae'], atol=1e-9)
            records.append(dict(model=model, label=label, predictors=k, target=target, reference=candidate,
                                mae=row.mae, B2V_improvement=delta, ci_low=lo, ci_high=hi, n_test=int(row.n_test)))
            digits = 2 if target == 'wrc_plus' else 3
            def signed(value):
                precision = max(digits, int(np.ceil(-np.log10(abs(value))))) if value else digits
                return f'{value:+.{precision}f}'
            cell += [f'{row.mae:.{digits}f}', 'reference' if model == candidate else f'{signed(delta)} [{signed(lo)}, {signed(hi)}]']
        cells.append(cell)
    df = pd.DataFrame(records); df.to_csv(TAB / 'table1_prediction_tradeoffs.csv', index=False)
    headers = ['Model inputs', 'wRC+\nMAE', 'wRC+ gain [95% CI]', 'WAR/600\nMAE', 'WAR/600 gain [95% CI]']
    md = '| ' + ' | '.join(h.replace('\n', ' ') for h in headers) + ' |\n| ' + ' | '.join(['---']*5) + ' |\n'
    md += '\n'.join('| ' + ' | '.join(row) + ' |' for row in cells)
    md += '\n\nGain = row MAE − reference MAE; positive favors B2V. First seven rows use B2V; final five use projection+B2V. Five forecast years (2021–2025), 1,706 transitions; 95% paired player-cluster bootstrap CIs (2,000 resamples). Per-tool PCA: one component per tool; unrestricted PCA: five across all 22 inputs. Fixed representatives: ISO, Contact%, BB%, BsR, Def. Selected representatives: training-only cross-validation among the fixed set and 17 within-tool single-replacement sets, separately by target, forecast origin and standalone/history context. History-based projections are recalibrated Marcel-style projections. The history+B2V versus history+selected-representatives wRC+ difference is inconclusive after refitting without 2020-related transitions.\n'
    (TAB / 'table1_prediction_tradeoffs.md').write_text(md)
    fig, ax = plt.subplots(figsize=(14, 8.4)); ax.axis('off')
    fig.subplots_adjust(left=.025, right=.975, top=.86, bottom=.28)
    fig.text(.025, .956, 'Table 1 | Predictive gains and the cost of compression', fontsize=18, weight='bold', color='#182C3B')
    fig.text(.025, .898, 'Primary: next-season wRC+   ·   Secondary: WAR per 600 PA   ·   Full sample: five forecast years, 1,706 transitions', fontsize=11, color='#61717D')
    table = ax.table(cellText=cells, colLabels=headers, colWidths=[.29, .07, .255, .08, .305], cellLoc='left', colLoc='left', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#DDE4E9'); cell.set_linewidth(.55)
        if i == 0:
            cell.set_facecolor('#183F54'); cell.set_text_props(color='white', weight='bold')
        elif i in [4, 9]:
            cell.set_facecolor('#E3EFF5'); cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#F6F8FA' if i > 7 else 'white')
    fig.text(.025, .227, 'Gain = row MAE − reference MAE; positive favors B2V. Rows 1–7: B2V. Rows 8–12: projection + B2V.', fontsize=10, color='#182C3B')
    fig.text(.025, .187, 'Lower MAE is better. Paired player-cluster bootstrap CIs (2,000 resamples); intervals crossing zero do not establish equivalence.', fontsize=9.5, color='#61717D')
    fig.text(.025, .147, 'Per-tool PCA: one component per tool. Unrestricted PCA: five across 22 inputs. History-based projections: recalibrated Marcel-style.', fontsize=9.5, color='#61717D')
    fig.text(.025, .107, 'Fixed representatives: ISO, Contact%, BB%, BsR, Def. Selected: training-only CV among the fixed set and 17 within-tool replacements.', fontsize=9.5, color='#61717D')
    fig.text(.025, .067, 'Selection is separate for each target, forecast origin and standalone/history context; it never uses the held-out forecast outcomes.', fontsize=9.5, color='#61717D')
    fig.text(.025, .027, 'The history+B2V versus history+selected-representatives wRC+ difference is inconclusive without 2020-related transitions.', fontsize=9.5, color='#61717D')
    save(fig, TAB, 'table1_prediction_tradeoffs')
    return df


def main():
    for path in [OUT, FIG, TAB]: path.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(RESULTS / 'task8/cohort.csv')
    assert len(cohort) == 3070 and not cohort.duplicated(['player_id', 'Season']).any()
    assert cohort[COLS].ge(20).all().all() and cohort[COLS].le(80).all().all()
    summary, differences, example, audit = build_diversity(cohort)
    diversity_figure(summary, differences, example, audit)
    prediction_table()
    sources = [RESULTS / t / f for t in ['task8', 'task9'] for f in ['cohort.csv', 'model_performance.csv', 'model_contrasts.csv'] if (RESULTS / t / f).exists()]
    sources += [RESULTS / 'task10' / f for f in ['performance.csv', 'contrasts.csv', 'selected_variants.csv', 'PROTOCOL.md']]
    manifest = {'scale': 'sd10: season-specific 50 + 10z, clipped to 20–80',
                'sources_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
                'script_sha256': digest(Path(__file__)), 'diversity': audit,
                'prediction_model_refits': False}
    (ROOT / 'abstract/submission_assets_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(summary.to_string(index=False)); print(differences.to_string(index=False))
    print(example[['focal_name', 'comparison_name', 'focal_season', 'focal_war', 'comparison_war', 'focal_wrc', 'comparison_wrc']].to_string())


if __name__ == '__main__': main()
