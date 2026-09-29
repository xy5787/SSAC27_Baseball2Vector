"""Refreshed primary forecasts and descriptive matching; no manuscript assets."""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
import production_increment as pi
import task10_representative_sensitivity as t10
sc,t8,sd=pi.sc,pi.t8,pi.sd
R=sc.SSAC_ROOT/'results_sd10'
INPUT=sc.SSAC_ROOT/'data/processed/submission_inputs_2019_2025.csv'
assert sc.B2V_SCALE=='sd10'

def run():
    R.mkdir(exist_ok=True)
    for d in ['task8','task9','task10','diversity']:
        (R/d).mkdir(exist_ok=True)
    raw=pi.source(2019); c=t8.cohort_from_raw(raw)
    INPUT.parent.mkdir(parents=True,exist_ok=True)
    raw.to_csv(INPUT,index=False); c.to_csv(R/'task8/cohort.csv',index=False)
    t8.INPUT=INPUT;t8.OUT=R/'task8';t10.OUT=R/'task10'
    shutil.copy(sc.SSAC_ROOT/'protocols/task10.md',t10.OUT/'PROTOCOL.md')
    for tool,stats in t8.FEATURE_GROUPS.items():
        q=c[[f'cs_{sc.sanitize(s)}' for s in stats]].mean(axis=1)
        c[f'pre_{tool}']=q.groupby(c.Season).transform(lambda x:(x-x.mean())/(x.std(ddof=0)+1e-12))
    base=t8.transitions(c);parts=[];tunes=[]
    for tk in ['wrc_plus','war_rate']:
        frame=pi.target_frame(raw,base,tk)
        specs=t8.specs(sc.TARGETS[tk])|{'train_PCA5':t8.STATS,'full_PCA5':t8.STATS,
            'history+Offense3':['history_t',*t8.OFF],
            'minus_Defense':[x for x in t8.TOOLS if 'Defense' not in x],
            'minus_Speed':[x for x in t8.TOOLS if 'Speed' not in x],
            'B2V_preclip':[f'pre_{x}_t' for x in sc.TOOL_NAMES],
            'history+B2V_preclip':['history_t',*[f'pre_{x}_t' for x in sc.TOOL_NAMES]]}
        for window,f in [('all',frame),('no2020',sc.drop_2020(frame))]:
            for fold,(tr,te) in sd.extended_rolling_folds(f).items():
                train,test=sc.split_by_input_season(f,tr,te)
                assert train.Season_t1.max()<test.Season_t1.min()
                for model,features in specs.items():
                    mode=True if model=='train_PCA5' else 'full' if model=='full_PCA5' else False
                    pred,alpha,scores=t8.fit_fast(train,test,features,sc.TARGETS[tk].column,mode)
                    b=test[[*t8.KEY,'Name','Season_t1','PA_t1','marcel_seasons_used']].copy()
                    b['y_true']=test[sc.TARGETS[tk].column].to_numpy(); b['prediction']=pred
                    b['model']=model;b['target']=tk;b['window']=window;b['fold']=fold;b['origin']=b.Season_t1
                    b['selected_alpha']=alpha;b['n_train']=len(train)
                    parts.append(b)
                    tunes.extend(dict(target=tk,window=window,fold=fold,model=model,alpha=float(a),inner_mae=float(s),selected=bool(a==alpha)) for a,s in zip(sc.ALPHA_GRID_WIDE,scores))
                print(tk,window,fold,'refitted',flush=True)
    p=pd.concat(parts,ignore_index=True);p.to_csv(R/'task8/model_predictions.csv',index=False)
    pd.DataFrame(tunes).to_csv(R/'task8/model_tuning.csv',index=False)
    pairs=t8.model_pairs()+[('B2V','full_PCA5'),('full_PCA5','train_PCA5'),('full_PCA5','B2V'),('history+B2V','history+Offense3'),('B2V','minus_Defense'),('B2V','minus_Speed')]
    pairs += [(m,b) for m in ['B2V_preclip','history+B2V_preclip'] for b in (['B2V','stats22','train_PCA5','full_PCA5'] if m=='B2V_preclip' else ['history','history+B2V'])]
    perfs=[];conts=[]
    # pi summarizes shared bootstrap draws once per model block.
    for (tk,w),b in p.groupby(['target','window']):
        for fold in [*sorted(b.fold.unique()),'pooled']:
            q=b if fold=='pooled' else b[b.fold.eq(fold)]
            perf,cont=pi.summarize_block(q,[(b,a) for a,b in pairs],fold)
            perfs.extend(dict(**x,fold=fold) for x in perf);conts.extend(dict(**x,fold=fold) for x in cont)
    perf=pd.DataFrame(perfs);cont=pd.DataFrame(conts)
    for task in ['task8','task9']:
        perf.to_csv(R/task/'model_performance.csv',index=False);cont.to_csv(R/task/'model_contrasts.csv',index=False)
    t10.run()
    import diversity
    diversity.build_diversity(c)
