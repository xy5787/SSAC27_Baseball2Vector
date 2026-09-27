"""Representative-statistic ablations with outer-training-only subset selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import task8_submission_strengthening as t8

sc, sd = t8.sc, t8.sd
OUT = sc.RESULTS_DIR / 'task10'
ORDER = ['Power', 'Contact', 'Discipline', 'Speed', 'Defense']
ORIGINAL = dict(zip(ORDER, ['ISO', 'Contact%', 'BB%', 'BsR', 'Def']))


def variants():
    out = [('original5', 'original', None, list(ORIGINAL.values()))]
    for tool in ORDER:
        for stat in t8.FEATURE_GROUPS[tool]:
            if stat != ORIGINAL[tool]:
                choice = ORIGINAL | {tool: stat}
                out.append((f'swap_{tool}_{sc.sanitize(stat)}', 'swap', tool, list(choice.values())))
    for tool in ORDER:
        out.append((f'drop_{tool}', 'drop', tool, [v for k, v in ORIGINAL.items() if k != tool]))
    return out


def select_variant(tuning):
    eligible = tuning[tuning.kind.ne('drop')]
    return eligible.sort_values(['inner_mae', 'variant_order'], kind='stable').iloc[0]


def run():
    assert sc.B2V_SCALE == 'sd10'
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(t8.INPUT)
    cohort = pd.read_csv(t8.OUT / 'cohort.csv')
    rebuilt = t8.cohort_from_raw(raw)
    np.testing.assert_allclose(rebuilt[[f'b2v_{t}' for t in ORDER]], cohort[[f'b2v_{t}' for t in ORDER]], atol=1e-10)
    base = t8.transitions(cohort)
    old = pd.read_csv(t8.OUT / 'model_predictions.csv')
    specs = variants()
    assert len(specs) == 23
    pd.DataFrame([dict(variant=n, kind=k, tool=t, statistics=json.dumps(v), variant_order=i)
                  for i, (n,k,t,v) in enumerate(specs)]).to_csv(OUT/'variants.csv',index=False)
    output, tuning, selections = [], [], []
    for tk in ['wrc_plus', 'war_rate']:
        target = sc.TARGETS[tk]
        projection = t8.marcel.project(raw[raw.PA.gt(0)], metric=t8.METRIC[tk], league_baseline='definition', recenter=False)
        frame = base.merge(projection.rename(columns={'marcel_pred':'history_t'}), left_on=t8.KEY,
                           right_on=['player_id','origin_season'],validate='one_to_one')
        for window, data in [('all', frame), ('no2020', sc.drop_2020(frame))]:
            for fold, (tr, te) in sd.extended_rolling_folds(data).items():
                train, test = sc.split_by_input_season(data, tr, te)
                assert train.Season_t1.max() <= test.Season_t.min()
                for context in ['standalone','history']:
                    block, tune = [], []
                    for i, (name, kind, tool, stats) in enumerate(specs):
                        features = [f'cs_{sc.sanitize(v)}_t' for v in stats]
                        if context == 'history': features = ['history_t', *features]
                        pred, alpha, scores = t8.fit_fast(train, test, features, target.column)
                        b = test[[*t8.KEY, 'Season_t1']].copy()
                        b['y_true'] = test[target.column].to_numpy()
                        b['prediction'] = pred; b['model'] = name
                        b['target'] = tk; b['window'] = window; b['fold'] = fold; b['context'] = context
                        block.append(b)
                        tune.append(dict(target=tk,window=window,fold=fold,context=context,model=name,kind=kind,
                                         variant_order=i,alpha=alpha,inner_mae=float(np.min(scores)),n_train=len(train),n_test=len(test)))
                        tuning.extend(dict(target=tk,window=window,fold=fold,context=context,model=name,kind=kind,
                                           variant_order=i,alpha=float(a),inner_mae=float(v),selected_alpha=bool(a==alpha))
                                      for a,v in zip(sc.ALPHA_GRID_WIDE,scores))
                    chosen = select_variant(pd.DataFrame(tune))
                    selected = next(b for b in block if b.model.iloc[0] == chosen.model).copy()
                    selected['model'] = 'selected5'
                    selections.append(chosen.to_dict())
                    # Reproduction of the original set is mandatory, on row keys.
                    original_name = 'representative5' if context == 'standalone' else 'history+representative5'
                    mask = old.target.eq(tk)&old.window.eq(window)&old.fold.eq(fold)
                    saved = old[mask&old.model.eq(original_name)].set_index(t8.KEY)
                    orig = block[0].set_index(t8.KEY)
                    assert set(orig.index)==set(saved.index)
                    np.testing.assert_allclose(orig.prediction,saved.loc[orig.index,'prediction'],atol=1e-8,rtol=0)
                    bname = 'B2V' if context == 'standalone' else 'history+B2V'
                    reference = old[mask&old.model.eq(bname)].copy()
                    reference['model']='B2V';reference['context']=context
                    assert set(reference.set_index(t8.KEY).index)==set(orig.index)
                    output.extend([*block, selected, reference[selected.columns]])
                print(tk,window,fold,'completed',flush=True)
    pd.concat(output).to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(tuning).to_csv(OUT/'inner_tuning.csv.gz',index=False,compression='gzip')
    pd.DataFrame(selections).to_csv(OUT/'selected_variants.csv',index=False)
    env = {**sc.environment_stamp(),'protocol_sha256':sc.sha256_file(OUT/'PROTOCOL.md'),
           'input_sha256':sc.sha256_file(t8.INPUT),'task8_predictions_sha256':sc.sha256_file(t8.OUT/'model_predictions.csv'),
           'script_sha256':sc.sha256_file(Path(__file__)),'original5_predictions_reproduced':True,
           'scope':'23 fixed variants plus training-only selection among 18 five-statistic sets; B2V predictions reused'}
    (OUT/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    summarize()


def summarize():
    predictions=pd.read_csv(OUT/'predictions.csv.gz')
    meta=pd.read_csv(OUT/'variants.csv').set_index('variant')
    perfs, contrasts=[] , []
    for (tk,window,context),b in predictions.groupby(['target','window','context']):
        pivot=b.pivot(index=t8.KEY,columns='model',values='prediction')
        y=b.drop_duplicates(t8.KEY).set_index(t8.KEY).loc[pivot.index,'y_true']
        assert b.groupby(t8.KEY).y_true.nunique().eq(1).all() and pivot.notna().all().all()
        errors=pivot.sub(y,axis=0).abs()
        folds=b.drop_duplicates(t8.KEY).set_index(t8.KEY).loc[pivot.index,'fold']
        ids=pivot.index.get_level_values('player_id').to_numpy()
        draws=sc.cluster_bootstrap_indices(ids)
        # All comparisons share resampling draws; compute each draw's model MAE once.
        arr=errors.to_numpy()
        boot=np.array([arr[ix].mean(axis=0) for ix in draws])
        for fold in [*sorted(folds.unique()),'pooled']:
            e=errors if fold=='pooled' else errors.loc[folds.eq(fold)]
            for model in e:
                perfs.append(dict(target=tk,window=window,context=context,fold=fold,model=model,n_test=len(e),mae=float(e[model].mean())))
        for model in errors:
            if model=='B2V': continue
            candidate='original5' if model.startswith('drop_') else 'B2V'
            candidates=[candidate]
            if model not in ['original5'] and not model.startswith('drop_'): candidates.append('original5')
            for cand in candidates:
                values=boot[:,errors.columns.get_loc(model)]-boot[:,errors.columns.get_loc(cand)]
                lo,hi=sc.percentile_ci(values)
                delta=errors[model]-errors[cand]
                by_year=delta.groupby(folds).mean()
                contrasts.append(dict(target=tk,window=window,context=context,candidate=cand,baseline=model,
                    kind='selected' if model=='selected5' else meta.loc[model,'kind'],n_test=len(errors),
                    improvement_mae=float(delta.mean()),ci_low=lo,ci_high=hi,
                    positive_years=int(by_year.gt(0).sum()),n_years=len(by_year)))
        print('summary',tk,window,context,flush=True)
    pd.DataFrame(perfs).to_csv(OUT/'performance.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False)
    print(pd.DataFrame(contrasts).query("baseline == 'selected5' and candidate == 'B2V'").to_string(index=False))


if __name__=='__main__': run()
