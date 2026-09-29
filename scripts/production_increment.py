"""Projection-system analysis, relocated from the frozen research implementation.
Numerical fitting/scoring functions are unchanged. Run via reproduce.py.
"""
from pathlib import Path
import hashlib, json, os, platform
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import task8_submission_strengthening as t8
sc, sd = t8.sc, t8.sd
ROOT = sc.SSAC_ROOT
OUT = ROOT / 'results_sd10/projection'
PRIVATE = ROOT / 'data/raw/generated/projection'
SOURCE = ROOT / 'data/raw/BattingStats_2018_2026.csv'
KEY = ['player_id', 'Season_t']
REPL_PAIRS = [('scalar','B2V'),('B2V','stats22'),('history','history+B2V'),('history+B2V','history+stats22')]
RUN_COMMAND = 'python scripts/reproduce.py --stage projections'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def source(year_start):
    raw = pd.read_csv(SOURCE, low_memory=False)
    for col in ['Name', 'Team', 'Season']:
        assert raw[col].equals(raw[col+'.1'])
    raw['player_id'] = 'fg:' + raw.PlayerId.astype('Int64').astype(str)
    cols = ['player_id', 'Name', 'Team', 'Season', 'PA', 'WAR', 'wRC+', 'OPS', 'Age', *sc.CONSTITUENT_STATS]
    raw = raw.loc[raw.Season.between(year_start, 2025), cols].copy()
    assert not raw.duplicated(['player_id', 'Season']).any()
    assert raw.player_id.notna().all() and not raw.player_id.str.contains('<NA>').any()
    return raw.sort_values(['Season', 'player_id'], kind='stable').reset_index(drop=True)

def target_frame(raw, base, tk):
    h = raw[raw.PA.gt(0)].copy()
    assert h[['PA', 'WAR', 'wRC+', 'Age']].notna().all().all()
    p = t8.marcel.project(h, metric=t8.METRIC[tk], league_baseline='definition', recenter=False)
    frame = base.merge(p.rename(columns={'marcel_pred': 'history_t'}), left_on=KEY,
                       right_on=['player_id', 'origin_season'], validate='one_to_one', how='left')
    assert frame.history_t.notna().all()
    assert frame.marcel_seasons_used.between(1, 3).all()
    assert np.isfinite(frame[sc.TARGETS[tk].column]).all()
    return frame

def load_projection(system):
    suffix = {'zips': 'z', 'steamer': 's'}[system]
    parts = []
    for year in [2019, 2021, 2022, 2023, 2024, 2025]:
        path = ROOT/'ProjectionDataset'/f'fangraphs-leaderboard-projections_{year}_{suffix}.csv'
        p = pd.read_csv(path, dtype={'PlayerId': 'string'})
        assert {'wRC+', 'WAR', 'PA', 'PlayerId'} <= set(p)
        assert not p.PlayerId.dropna().duplicated().any()
        p = p[p.PlayerId.notna()].copy()
        p['player_id'] = 'fg:' + p.PlayerId
        p['Season_t1'] = year
        p['projection_year'] = year
        p['projection_file'] = path.name
        p = p.rename(columns={'wRC+': 'projection_wrc', 'WAR': 'projection_war', 'PA': 'projection_pa'})
        parts.append(p[['player_id', 'Season_t1', 'projection_year', 'projection_file', 'projection_wrc', 'projection_war', 'projection_pa']])
    p = pd.concat(parts, ignore_index=True)
    assert not p.duplicated(['player_id', 'Season_t1']).any()
    return p

def join_projection(frame, projection, tk):
    frame = frame[frame.Season_t1.ne(2020)].copy()
    joined = frame.merge(projection, on=['player_id', 'Season_t1'], how='left', validate='one_to_one', indicator=True)
    matched = joined._merge.eq('both')
    good = matched & np.isfinite(joined.projection_wrc)
    if tk == 'war_rate':
        good &= np.isfinite(joined.projection_war) & np.isfinite(joined.projection_pa) & joined.projection_pa.gt(0)
    flow = []
    for year, b in joined.assign(usable=good).groupby('Season_t1'):
        dropped = b[~b.usable]
        flow.append(dict(target=tk, outcome_year=int(year), candidate_n=len(b), matched_n=int(b._merge.eq('both').sum()),
            usable_n=int(b.usable.sum()), excluded_n=len(dropped),
            excluded_PA_median=dropped.PA_t.median(), excluded_age_median=dropped.age_t.median(),
            excluded_depth1=int(dropped.marcel_seasons_used.eq(1).sum()), excluded_depth2=int(dropped.marcel_seasons_used.eq(2).sum()),
            excluded_depth3=int(dropped.marcel_seasons_used.eq(3).sum())))
    kept = joined[good].copy()
    kept['projection_t'] = kept.projection_wrc if tk == 'wrc_plus' else kept.projection_war/kept.projection_pa*600
    assert np.isfinite(kept.projection_t).all()
    assert kept.projection_year.eq(kept.Season_t1).all()
    assert kept.Season_t1.eq(kept.Season_t+1).all()
    return kept, flow

def fit_window(frame, tk, window, specs, raw_model=None, coefficient_model=None):
    target = sc.TARGETS[tk]
    predictions, tuning, coefficients, folds = [], [], [], []
    for label, (train_inputs, test_input) in sd.extended_rolling_folds(frame).items():
        train, test = sc.split_by_input_season(frame, train_inputs, test_input)
        assert train.Season_t1.max() < test.Season_t1.min()
        assert test.Season_t1.nunique() == 1
        year = int(test.Season_t1.iloc[0])
        folds.append(dict(target=tk, window=window, origin=year, n_train=len(train), n_test=len(test),
                          train_outcomes=','.join(map(str, sorted(train.Season_t1.unique())))))
        for model, features in specs.items():
            sc.assert_no_future_columns(features)
            if 'projection_t' in features:
                assert np.isfinite(train.projection_t).all() and np.isfinite(test.projection_t).all()
            pred, alpha, scores = t8.fit_fast(train, test, features, target.column)
            assert np.isfinite(pred).all()
            meta = test[[*KEY, 'Season_t1', 'marcel_seasons_used']].copy()
            meta['y_true'] = test[target.column].to_numpy(float)
            meta['prediction'] = pred
            meta['model'], meta['target'], meta['window'], meta['origin'] = model, tk, window, year
            meta['selected_alpha'], meta['n_train'], meta['n_features'] = alpha, len(train), len(features)
            predictions.append(meta)
            tuning.extend(dict(target=tk, window=window, origin=year, model=model, alpha=float(a),
                               inner_mae=float(s), selected=bool(a == alpha)) for a, s in zip(sc.ALPHA_GRID_WIDE, scores))
            if model == coefficient_model:
                pipe = Pipeline([('transform', t8.transform(features)), ('ridge', Ridge(alpha=alpha))])
                pipe.fit(train[features], train[target.column])
                np.testing.assert_allclose(pipe.predict(test[features]), pred, atol=1e-10, rtol=0)
                ridge = pipe.named_steps['ridge']
                for feature, value in zip(features, ridge.coef_):
                    coefficients.append(dict(target=tk, window=window, origin=year, model=model, feature=feature,
                        coefficient=float(value), intercept=float(ridge.intercept_), selected_alpha=alpha, n_train=len(train)))
        if raw_model:
            meta = predictions[-1].copy()
            meta['model'], meta['prediction'], meta['selected_alpha'], meta['n_features'] = raw_model, test.projection_t.to_numpy(), np.nan, 1
            predictions.append(meta)
        print(f'{tk} {window} outcome={year}: train={len(train)} test={len(test)} models={len(specs)+bool(raw_model)}', flush=True)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(tuning), pd.DataFrame(coefficients), pd.DataFrame(folds)

def summarize_block(block, pairs, origin):
    assert not block.duplicated([*KEY, 'model']).any()
    assert block.groupby(KEY).y_true.nunique().eq(1).all()
    pv = block.pivot(index=KEY, columns='model', values='prediction').sort_index()
    assert pv.notna().all().all()
    meta = block.drop_duplicates(KEY).set_index(KEY).loc[pv.index]
    y = meta.y_true.to_numpy(float)
    ids = pv.index.get_level_values('player_id').to_numpy()
    errors = np.abs(pv.to_numpy()-y[:, None])
    sq = (pv.to_numpy()-y[:, None])**2
    samples = sc.cluster_bootstrap_indices(ids)
    boot = np.array([errors[ix].mean(axis=0) for ix in samples])
    basic = dict(target=str(block.target.iloc[0]), window=str(block.window.iloc[0]), origin=origin,
                 n_test=len(pv), n_players=int(len(np.unique(ids))))
    performance = [dict(**basic, model=model, mae=float(errors[:,j].mean()), rmse=float(np.sqrt(sq[:,j].mean())))
                   for j, model in enumerate(pv.columns)]
    contrasts = []
    for baseline, candidate in pairs:
        a, b = pv.columns.get_loc(baseline), pv.columns.get_loc(candidate)
        low, high = sc.percentile_ci(boot[:,a]-boot[:,b])
        contrasts.append(dict(**basic, contrast=f'{baseline} − {candidate}', baseline=baseline, candidate=candidate,
            improvement_mae=float(errors[:,a].mean()-errors[:,b].mean()), ci_low=low, ci_high=high,
            excludes_zero=bool(low>0 or high<0), n_boot=sc.N_BOOT))
    return performance, contrasts

def summaries(p, production=False, prefix='Z'):
    perf, contrasts, strata = [], [], []
    for (tk, window), block in p.groupby(['target', 'window'], sort=True):
        pairs = ([(prefix+'1', prefix+'2'), (prefix+'2', prefix+'3'), ('R1','R2')] +
                 ([(prefix+'0',prefix+'1')] if tk=='wrc_plus' else [])) if production else REPL_PAIRS
        for origin in [*sorted(block.origin.unique()), 'pooled']:
            b = block if origin == 'pooled' else block[block.origin.eq(origin)]
            a, c = summarize_block(b, pairs, origin)
            perf.extend(a); contrasts.extend(c)
        if window == 'all':
            for depth in [1, 2, 3]:
                b = block[block.marcel_seasons_used.eq(depth)]
                pair = [(prefix+'1',prefix+'2')] if production else [('history','B2V')]
                _, c = summarize_block(b, pair, 'pooled')
                strata.extend(dict(**row, prior_seasons=depth) for row in c)
    c = pd.DataFrame(contrasts)
    signs=[]
    for (tk, window, contrast), b in c[c.origin.ne('pooled')].groupby(['target','window','contrast']):
        signs.append(dict(target=tk,window=window,contrast=contrast,n_origins=len(b),
                          positive_origins=int(b.improvement_mae.gt(0).sum()),negative_origins=int(b.improvement_mae.lt(0).sum()),
                          signs=','.join(f'{int(r.origin)}:{"+" if r.improvement_mae>0 else "-" if r.improvement_mae<0 else "0"}' for r in b.itertuples())))
    return {'performance':pd.DataFrame(perf),'contrasts':c,'history_strata':pd.DataFrame(strata),'origin_signs':pd.DataFrame(signs)}

def environment():
    return dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
                sklearn=sklearn.__version__,seed=sc.SEED,n_boot=sc.N_BOOT,b2v_scale=sc.B2V_SCALE,
                openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'))

def write_arm(arm,p,tuning,folds,extras=None):
    public=OUT/arm;private=PRIVATE/arm
    public.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    p.to_csv(private/'predictions.csv',index=False)
    tuning.to_csv(public/'alpha_tuning.csv',index=False)
    folds.to_csv(public/'sample_flow.csv',index=False)
    tables=summaries(p,production=arm in ['zips','steamer'],prefix='S' if arm=='steamer' else 'Z')
    for name,table in (tables | (extras or {})).items():table.to_csv(public/f'{name}.csv',index=False)
    dump(public/'environment.json',dict(**environment(),source_sha256=sha(SOURCE),protocol_sha256=sha(OUT/'PRESPEC.md'),
         script_sha256=sha(__file__),input_manifest_sha256=sha(OUT/'prespec_inputs.json'),
         predictions_sha256=sha(private/'predictions.csv'),prediction_path=str((private/'predictions.csv').relative_to(ROOT)),
         command=RUN_COMMAND,execution='independent reproduction',complete=True))
    return tables

def alignment_check(kept,system):
    rows=[]
    for year,b in kept.groupby('Season_t1'):
        for row in b.sort_values('player_id').head(3).itertuples():
            path=ROOT/'ProjectionDataset'/row.projection_file
            raw=pd.read_csv(path,dtype={'PlayerId':'string'})
            selected=raw[raw.PlayerId.eq(row.player_id.removeprefix('fg:'))]
            assert len(selected)==1
            selected=selected.iloc[0]
            assert row.Season_t+1==row.Season_t1==row.projection_year==int(path.stem.split('_')[-2])
            for col,value in [('wRC+',row.projection_wrc),('WAR',row.projection_war),('PA',row.projection_pa)]:
                assert float(selected[col])==float(value)
            rows.append(dict(player_id=row.player_id,input_year=int(row.Season_t),outcome_year=int(row.Season_t1),
                projection_file=row.projection_file,projection_wrc=float(row.projection_wrc),verified=True))
    private=PRIVATE/system;private.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(private/'alignment_checks.csv',index=False)
    return pd.DataFrame(rows).groupby('outcome_year').agg(checked_rows=('verified','size'),all_passed=('verified','all')).reset_index()

def run_production(system='zips'):
    raw=source(2018);cohort=t8.cohort_from_raw(raw);base=t8.transitions(cohort)
    assert len(raw)==11462 and len(cohort)==3518 and len(base)==2328
    projection=load_projection(system);prefix='Z' if system=='zips' else 'S'
    parts=[];tunes=[];coefs=[];folds=[];flows=[];alignment_frame=None
    for tk in ['wrc_plus','war_rate']:
        frame,flow=join_projection(target_frame(raw,base,tk),projection,tk)
        assert len(frame)==2053
        flows.extend(flow)
        if tk=='wrc_plus':alignment_frame=frame
        original=t8.specs(sc.TARGETS[tk])
        specs={prefix+'1':['projection_t'],prefix+'2':['projection_t',*t8.TOOLS],prefix+'3':['projection_t',*t8.STATS],
               'R1':original['history'],'R2':original['history+B2V'],'R3':original['B2V']}
        for window,b in [('all',frame),('no2020',frame[frame.Season_t.ne(2020)].copy())]:
            p,t,c,f=fit_window(b,tk,window,specs,prefix+'0' if tk=='wrc_plus' else None,prefix+'2')
            expected=1706 if window=='all' else 1427
            assert p.drop_duplicates(KEY).shape[0]==expected
            parts.append(p);tunes.append(t);coefs.append(c);folds.append(f)
    p=pd.concat(parts,ignore_index=True)
    tables=write_arm(system,p,pd.concat(tunes,ignore_index=True),pd.concat(folds,ignore_index=True),
              {'standardized_coefficients':pd.concat(coefs,ignore_index=True),'projection_matching':pd.DataFrame(flows)})
    alignment=alignment_check(alignment_frame,system)
    alignment.to_csv(OUT/system/'alignment_checks.csv',index=False)
    c=tables['contrasts'];perf=tables['performance'];ratios=[]
    for (tk,window), b in perf[perf.origin.eq('pooled')].groupby(['target','window']):
        v=b.set_index('model').mae
        numerator=float(v[prefix+'1']-v[prefix+'2']);denominator=float(v[prefix+'1']-v[prefix+'3'])
        ratios.append(dict(target=tk,window=window,b2v_increment=numerator,constituent_increment=denominator,
                           retained_fraction=numerator/denominator if denominator>0 else None,denominator_positive=denominator>0))
    pd.DataFrame(ratios).to_csv(OUT/system/'increment_retention.csv',index=False)
    main=c[(c.target=='wrc_plus')&(c.window=='all')&(c.origin=='pooled')&(c.baseline==prefix+'1')&(c.candidate==prefix+'2')].iloc[0]
    triggered=bool(main.improvement_mae>0.40)
    if triggered:
        # Repeat raw-file alignment checks and review temporal-fold bounds, regardless of sign of other contrasts.
        second=alignment_check(alignment_frame,system)
        pd.testing.assert_frame_equal(alignment,second)
        assert alignment_frame.Season_t1.eq(alignment_frame.Season_t+1).all()
    dump(OUT/system/'integrity_checks.json',dict(all_passed=True,alignment_rows=int(alignment.checked_rows.sum()),
        increment_above_historical_040=triggered,alignment_rechecked=triggered,
        train_outcomes_strictly_before_test=True,projection_year_equals_outcome=True,
        note='Source vintage is retrospective; CSVs do not independently certify historical preseason publication timestamps.'))
    print(system,'complete',flush=True)
