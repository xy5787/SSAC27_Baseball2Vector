"""Follow-up controls and repaired retrieval. See results/task8/PROTOCOL.md.

python SSAC27/scripts/task8_submission_strengthening.py --stage all
Stages: export, models, retrieval, summarize. --input allows a portable CSV.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import common as sc
import data as sd
from baseball2vec import marcel
from baseball2vec.data import preprocess
from baseball2vec.tools import FEATURE_GROUPS, season_zscore, apply_direction

OUT = sc.RESULTS_DIR / 'task8'
INPUT = sc.INTERMEDIATE_DIR / 'submission_inputs_2019_2025.csv'
REP = ['ISO', 'Contact%', 'BB%', 'BsR', 'Def']
METRIC = {'wrc_plus': 'wRC+', 'war_rate': 'WAR_per_600', 'war': 'WAR'}
KEY = ['player_id', 'Season_t']
TOOLS = [f'b2v_{t}_t' for t in sc.TOOL_NAMES]
OFF = [f'b2v_{t}_t' for t in ['Power','Contact','Discipline']]
STATS = [f'{c}_t' for c in sc.CONSTITUENT_COLS]
REPS = [f'cs_{sc.sanitize(s)}_t' for s in REP]


def export_inputs():
    raw = sd.load_raw_exports()
    cols = ['player_id','Name','Team','Season','PA','WAR','wRC+','OPS','Age', *sc.CONSTITUENT_STATS]
    raw = raw[cols].sort_values(['Season','player_id']).reset_index(drop=True)
    assert raw.player_id.notna().all() and not raw.player_id.str.contains('<NA>').any()
    raw.to_csv(INPUT, index=False)
    manifest = {'inputs_sha256': sc.sha256_file(INPUT), 'rows': len(raw),
        'source_hashes': {p.name: sc.sha256_file(p) for p in [sd.EXPORT_2019_2020,sd.EXPORT_2021_2025]},
        'source': 'FanGraphs manual leaderboard exports, see data/raw and original manifests',
        'scope': 'minimal unfiltered 2019–2025 input columns; locally prepared, not externally published',
        'third_party_license': 'No license granted or inferred by this export.'}
    (sc.INTERMEDIATE_DIR/'submission_inputs_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return raw


def cohort_from_raw(raw):
    c = raw.loc[raw.PA.ge(100)].copy().reset_index(drop=True)
    a = apply_direction(season_zscore(preprocess(c.copy()), sc.CONSTITUENT_STATS))
    for s,col in zip(sc.CONSTITUENT_STATS, sc.CONSTITUENT_COLS): c[col] = a[s].to_numpy()
    def z(values):
        return values.groupby(c.Season).transform(lambda x:(x-x.mean())/(x.std(ddof=0)+1e-12)).fillna(0)
    for t,stats in FEATURE_GROUPS.items(): c[f'b2v_{t}'] = z(a[stats].mean(axis=1))
    variants = {'fld': a.Fld, 'def':a.Def, 'bsr':a.BsR,
        'speed_no_ubr':a[['Spd','BsR','wSB']].mean(axis=1),
        'fld_per_pa':c.Fld*600/c.PA, 'bsr_per_pa':c.BsR*600/c.PA}
    for name,values in variants.items(): c[f'v_{name}'] = z(values)
    for col in ['WAR','wRC+']:
        c[f'search_{col}'] = z(c[col])
    c = sc.apply_b2v_scale(c, [col for col in c if col.startswith(('b2v_', 'v_'))])
    c['WAR_per_600'] = c.WAR*600/c.PA
    c['age'] = c.Age
    return c


def transitions(c):
    left = c.add_suffix('_t'); right = c.add_suffix('_t1')
    left['player_id'] = left.player_id_t; right['player_id'] = right.player_id_t1
    right['Season_t'] = right.Season_t1-1
    t = left.merge(right,on=KEY,validate='one_to_one')
    t['Name'] = t.Name_t
    assert not t.duplicated(KEY).any()
    return t.sort_values(['Season_t','player_id']).reset_index(drop=True)


def specs(target):
    base = {'scalar':[target.baseline], 'B2V':TOOLS, 'Offense3':OFF,
        'stats22':STATS, 'representative5':REPS, 'history':['history_t'],
        'history+B2V':['history_t',*TOOLS], 'history+stats22':['history_t',*STATS],
        'history+representative5':['history_t',*REPS],
        'history+B2V+context':['history_t',*TOOLS,'age_t','PA_t'],
        'history+stats22+context':['history_t',*STATS,'age_t','PA_t']}
    full = {t:f'b2v_{t}_t' for t in sc.TOOL_NAMES}
    replacements = {'DefenseFld':{'Defense':'v_fld_t'}, 'DefenseDef':{'Defense':'v_def_t'},
        'SpeedBsR':{'Speed':'v_bsr_t'}, 'SpeedNoUBR':{'Speed':'v_speed_no_ubr_t'},
        'FldBsR':{'Defense':'v_fld_t','Speed':'v_bsr_t'},
        'PerPA':{'Defense':'v_fld_per_pa_t','Speed':'v_bsr_per_pa_t'}}
    for name,replace in replacements.items(): base[name] = list((full|replace).values())
    return base


def transform(features, pca=False):
    # True preserves Task 8's groupwise PCA; "full" is the Task 9 control.
    if pca == "full":
        return Pipeline([('impute', SimpleImputer(strategy='median')),
            ('input_scale', StandardScaler()),
            ('pca', PCA(n_components=5, svd_solver='full')),
            ('scale', StandardScaler())])
    if not pca:
        return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    groups = [(t,Pipeline([('impute',SimpleImputer(strategy='median')),
        ('pca',PCA(n_components=1,svd_solver='full'))]),
        [f'cs_{sc.sanitize(s)}_t' for s in stats]) for t,stats in FEATURE_GROUPS.items()]
    return Pipeline([('groups',ColumnTransformer(groups)),('scale',StandardScaler())])


def fit_fast(train,test,features,target,pca=False):
    """Exact Ridge SVD path; transforms fitted separately in each inner split.

    Mathematically the same mean-centered Ridge objective as sklearn. This
    avoids fitting identical preprocessors 50 times per inner split.
    """
    grid = sc.ALPHA_GRID_WIDE
    scores = np.empty((5,len(grid)))
    for j,(tr,va) in enumerate(GroupKFold(5).split(train,groups=train.player_id)):
        proc = transform(features,pca)
        x = proc.fit_transform(train.iloc[tr][features]); xv = proc.transform(train.iloc[va][features])
        y = train.iloc[tr][target].to_numpy(float); ym = y.mean(); xm = x.mean(axis=0)
        u,s,vt = np.linalg.svd(x-xm,full_matrices=False)
        coefs = vt.T @ ((s[:,None]/(s[:,None]**2+grid))*(u.T@(y-ym))[:,None])
        preds = (xv-xm)@coefs+ym
        scores[j] = np.abs(preds-train.iloc[va][target].to_numpy()[:,None]).mean(axis=0)
    means = scores.mean(axis=0); alpha = float(grid[np.argmin(means)])
    pipe = Pipeline([('transform',transform(features,pca)),('ridge',Ridge(alpha=alpha))])
    pipe.fit(train[features],train[target])
    return pipe.predict(test[features]), alpha, means


def summarize_block(p,pairs,extra):
    assert not p.duplicated([*KEY,'model']).any()
    assert p.groupby(KEY).y_true.nunique().eq(1).all()
    perfs=[]; contrasts=[]
    for fold in [*sorted(p.fold.unique()),'pooled']:
        b = p if fold=='pooled' else p[p.fold.eq(fold)]
        perf,contrast = sc.evaluate_predictions(b,fold,pairs,pooled=fold=='pooled',extra=extra)
        perfs.append(perf); contrasts.append(contrast)
    return pd.concat(perfs),pd.concat(contrasts)


def model_pairs():
    return [('B2V','scalar'),('B2V','stats22'),('B2V','representative5'),
        ('B2V','Offense3'),('history+B2V','history'),('history+B2V','history+stats22'),
        ('history+B2V','history+representative5'),('history+B2V+context','history+stats22+context'),
        ('train_PCA5','B2V'), *[(v,'B2V') for v in ['DefenseFld','DefenseDef','SpeedBsR','SpeedNoUBR','FldBsR','PerPA']]]


def run_models(raw,c):
    base=transitions(c); history=raw[raw.PA.gt(0)]
    output=[]; tuning=[]
    for tk in ['wrc_plus','war_rate','war']:
        target=sc.TARGETS[tk]
        projection=marcel.project(history,metric=METRIC[tk],league_baseline='definition',recenter=False)
        t=base.merge(projection.rename(columns={'marcel_pred':'history_t'}),
            left_on=KEY,right_on=['player_id','origin_season'],validate='one_to_one')
        windows={'no2020':sc.drop_2020(t)} if tk=='war' else {'all':t,'no2020':sc.drop_2020(t)}
        for window,frame in windows.items():
            for fold,(tr,te) in sd.extended_rolling_folds(frame).items():
                train,test=sc.split_by_input_season(frame,tr,te)
                assert train.Season_t1.max() <= te
                for model,features in (specs(target)|{'train_PCA5':STATS}).items():
                    pred,alpha,scores=fit_fast(train,test,features,target.column,model=='train_PCA5')
                    b=test[[*KEY,'Name','Season_t1','PA_t1','marcel_seasons_used']].copy()
                    b['y_true']=test[target.column].to_numpy(); b['prediction']=pred
                    b['model']=model; b['fold']=fold; b['target']=tk; b['window']=window
                    b['selected_alpha']=alpha; b['n_train']=len(train);b['n_features']=5 if model=='train_PCA5' else len(features)
                    output.append(b)
                    tuning.extend({'target':tk,'window':window,'fold':fold,'model':model,
                        'alpha':float(a),'inner_mae':float(s),'selected':a==alpha} for a,s in zip(sc.ALPHA_GRID_WIDE,scores))
                print('models',tk,window,fold,flush=True)
                pd.concat(output).to_csv(OUT/'model_predictions.csv',index=False)
    pd.DataFrame(tuning).to_csv(OUT/'model_tuning.csv',index=False)


def nearest_rows(candidates,query,features,focal_id,policy,k=10):
    """Stable ordering; primary neighbors are distinct OTHER players."""
    can=candidates if policy=='allow_self_rows' else candidates[candidates.player_id.ne(focal_id)]
    distances=np.linalg.norm(can[features].to_numpy()-np.asarray(query,dtype=float),axis=1)
    ranked=can.assign(distance=distances).sort_values(['distance','player_id','Season'],kind='mergesort')
    if policy=='other_players': ranked=ranked.drop_duplicates('player_id')
    return ranked.head(k)


def run_retrieval(c):
    t=transitions(c)
    methods={'scalar':['search_WAR','search_wRC+'], 'B2V':[f'b2v_{x}' for x in sc.TOOL_NAMES],
        'stats22':sc.CONSTITUENT_COLS, 'representative5':[f'cs_{sc.sanitize(s)}' for s in REP],
        'Offense3':[f'b2v_{x}' for x in ['Power','Contact','Discipline']]}
    # Coordinates come from c, not from a returner subset of transitions.
    lookup=c.set_index(['player_id','Season'])
    candidates=c.merge(t[[*KEY,'wRC+_t1','WAR_t1','WAR_per_600_t1']],
        left_on=['player_id','Season'],right_on=KEY,validate='one_to_one')
    predictions=[];neighbors=[]
    for tk in ['wrc_plus','war_rate','war']:
        target=sc.TARGETS[tk];frame=sc.prepare_for_target(t,target)
        for fold,(tr,te) in sd.extended_rolling_folds(frame).items():
            test=frame[frame.Season_t.eq(te)]
            can=candidates[candidates.Season.isin(tr)].copy()
            assert (can.Season+1<=te).all()
            for policy in ['other_players','exclude_self_rows','allow_self_rows']:
                for method,features in methods.items():
                    for row in test.itertuples(index=False):
                        pid=row.player_id; src=lookup.loc[(pid,te)]
                        near=nearest_rows(can,src[features].to_numpy(),features,pid,policy)
                        assert len(near)==10
                        for rank,(_,n) in enumerate(near.iterrows(),1):
                            neighbors.append({'target':tk,'fold':fold,'policy':policy,'method':method,
                                'player_id':pid,'Season_t':te,'neighbor_id':n.player_id,
                                'neighbor_season':int(n.Season),'rank':rank,'distance':n.distance})
                        truth=float(test.loc[test.player_id.eq(pid),target.column].iloc[0])
                        for k in [3,5,10]:
                            predictions.append({'target':tk,'fold':fold,'policy':policy,'k':k,
                                'player_id':pid,'Season_t':te,'Season_t1':te+1,'model':method,
                                'y_true':truth,'prediction':near[target.column].head(k).mean()})
            print('retrieval',tk,fold,flush=True)
    pd.DataFrame(predictions).to_csv(OUT/'retrieval_predictions.csv',index=False)
    pd.DataFrame(neighbors).to_csv(OUT/'retrieval_neighbors.csv.gz',index=False,compression='gzip')


def summarize():
    p=pd.read_csv(OUT/'model_predictions.csv'); perfs=[];pairs=[];loo=[]
    for (target,window),b in p.groupby(['target','window']):
        perf,pair=summarize_block(b,model_pairs(),{'target':target,'window':window})
        perfs.append(perf);pairs.append(pair)
        for omit in sorted(b.fold.unique()):
            e=b[b.fold.ne(omit)].assign(error=lambda d:abs(d.y_true-d.prediction)).groupby('model').error.mean()
            for candidate,baseline in model_pairs():
                loo.append({'target':target,'window':window,'omitted_origin':omit,'candidate':candidate,
                    'baseline':baseline,'improvement_mae':e[baseline]-e[candidate]})
    pd.concat(perfs).to_csv(OUT/'model_performance.csv',index=False)
    pd.concat(pairs).to_csv(OUT/'model_contrasts.csv',index=False)
    pd.DataFrame(loo).to_csv(OUT/'leave_one_origin_out.csv',index=False)
    p=pd.read_csv(OUT/'retrieval_predictions.csv'); perfs=[];pairs=[]
    for (target,policy,k),b in p.groupby(['target','policy','k']):
        perf,pair=summarize_block(b,[('B2V',r) for r in ['scalar','stats22','representative5','Offense3']],
            {'target':target,'policy':policy,'k':int(k)})
        perfs.append(perf);pairs.append(pair)
    pd.concat(perfs).to_csv(OUT/'retrieval_performance.csv',index=False)
    pd.concat(pairs).to_csv(OUT/'retrieval_contrasts.csv',index=False)
    # Direct history-depth interaction: improvement in depth1 minus depth3.
    p=pd.read_csv(OUT/'model_predictions.csv'); b=p[(p.target=='wrc_plus')&(p.window=='all')]
    v=b.pivot(index=KEY,columns='model',values='prediction')
    meta=b.drop_duplicates(KEY).set_index(KEY).reindex(v.index)
    e=v.sub(meta.y_true,axis=0).abs();delta=(e['history']-e['B2V']).to_numpy()
    depth=meta.marcel_seasons_used.to_numpy(); ids=v.index.get_level_values(0).to_numpy()
    samples=sc.cluster_bootstrap_indices(ids); draws=[]
    for ix in samples:
        d=depth[ix]; z=delta[ix]
        if (d==1).any() and (d==3).any(): draws.append(z[d==1].mean()-z[d==3].mean())
    record={'n_depth1':int((depth==1).sum()),'n_depth3':int((depth==3).sum()),
        'depth1_improvement':float(delta[depth==1].mean()),'depth3_improvement':float(delta[depth==3].mean()),
        'interaction':float(delta[depth==1].mean()-delta[depth==3].mean()),
        'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),
        'interpretation':'exploratory difference in history-minus-B2V MAE advantages; depth is observed lookback, not career tenure'}
    (OUT/'history_depth_interaction.json').write_text(json.dumps(record,indent=2)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','export','models','retrieval','summarize'],default='all')
    parser.add_argument('--input',type=Path,default=INPUT)
    args=parser.parse_args(argv);OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='export' or (args.stage=='all' and not args.input.exists()):
        if args.input != INPUT: raise FileNotFoundError(args.input)
        export_inputs()
    if args.stage in ['all','models','retrieval']:
        raw=pd.read_csv(args.input);c=cohort_from_raw(raw)
        c.to_csv(OUT/'cohort.csv',index=False)
        missing=raw[raw.PA.ge(100)].groupby('Season')[sc.CONSTITUENT_STATS].agg(lambda x:x.isna().mean())
        missing.to_csv(OUT/'input_missing_fraction.csv')
        (OUT/'environment.json').write_text(json.dumps({**sc.environment_stamp(),
            'input_sha256':sc.sha256_file(args.input),'script_sha256':sc.sha256_file(Path(__file__)),
            'protocol_sha256':sc.sha256_file(OUT/'PROTOCOL.md'),'representative_stats':REP,
            'cohort_rows':len(c),'transition_rows':len(transitions(c))},indent=2)+'\n')
        if args.stage in ['all','models']:run_models(raw,c)
        if args.stage in ['all','retrieval']:run_retrieval(c)
    if args.stage in ['all','summarize']:summarize()

if __name__=='__main__':main()
