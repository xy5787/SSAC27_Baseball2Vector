"""Unrestricted PCA comparison; protocol and outputs in results/task9.

Run: OPENBLAS_NUM_THREADS=1 python SSAC27/scripts/task9_full_pca.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import task8_submission_strengthening as t8

sc, sd = t8.sc, t8.sd
OUT = sc.RESULTS_DIR / 'task9'
PAIRS = [('full_PCA5', m) for m in ['B2V', 'train_PCA5', 'stats22', 'representative5', 'scalar']]
PAIRS += [('B2V', 'full_PCA5')]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(t8.INPUT)
    cohort = t8.cohort_from_raw(raw)
    base = t8.transitions(cohort)
    saved = pd.read_csv(t8.OUT / 'model_predictions.csv')
    predictions, tuning, variance, loadings = [], [], [], []
    for tk in ['wrc_plus', 'war_rate', 'war']:
        target = sc.TARGETS[tk]
        windows = {'no2020': sc.drop_2020(base)} if tk == 'war' else {'all': base, 'no2020': sc.drop_2020(base)}
        for window, frame in windows.items():
            for fold, (tr, te) in sd.extended_rolling_folds(frame).items():
                train, test = sc.split_by_input_season(frame, tr, te)
                assert train.Season_t1.max() <= te
                models = {'full_PCA5': (t8.STATS, 'full'), 'train_PCA5': (t8.STATS, True),
                          'B2V': (t8.TOOLS, False), 'stats22': (t8.STATS, False),
                          'representative5': (t8.REPS, False), 'scalar': ([target.baseline], False)}
                for model, (features, mode) in models.items():
                    pred, alpha, scores = t8.fit_fast(train, test, features, target.column, mode)
                    b = test[[*t8.KEY, 'Name', 'Season_t1', 'PA_t1']].copy()
                    b['y_true'], b['prediction'] = test[target.column].to_numpy(), pred
                    b['model'], b['fold'], b['target'], b['window'] = model, fold, tk, window
                    b['selected_alpha'], b['n_train'] = alpha, len(train)
                    b['n_features'] = 5 if mode else len(features)
                    predictions.append(b)
                    tuning.extend({'target': tk, 'window': window, 'fold': fold, 'model': model,
                                   'alpha': float(a), 'inner_mae': float(s), 'selected': a == alpha}
                                  for a, s in zip(sc.ALPHA_GRID_WIDE, scores))
                proc = t8.transform(t8.STATS, 'full').fit(train[t8.STATS])
                pca = proc.named_steps['pca']
                for k, ratio in enumerate(pca.explained_variance_ratio_, 1):
                    variance.append({'target': tk, 'window': window, 'fold': fold, 'component': k,
                                     'explained_variance_ratio': float(ratio)})
                    loadings.extend({'target': tk, 'window': window, 'fold': fold, 'component': k,
                                     'feature': f, 'loading': float(v)}
                                    for f, v in zip(t8.STATS, pca.components_[k-1]))
                print('fitted', tk, window, fold, flush=True)
    p = pd.concat(predictions, ignore_index=True)
    keys = ['target', 'window', 'model', *t8.KEY]
    shared = p[p.model.ne('full_PCA5')]
    expected = saved[saved.model.isin(shared.model.unique())]
    j = shared.merge(expected, on=keys, validate='one_to_one', suffixes=('_new', '_old'))
    assert len(j) == len(shared) == len(expected)
    np.testing.assert_allclose(j.y_true_new, j.y_true_old, atol=1e-10, rtol=0)
    np.testing.assert_allclose(j.prediction_new, j.prediction_old, atol=1e-8, rtol=0)
    assert j.fold_new.eq(j.fold_old).all()
    np.testing.assert_allclose(j.selected_alpha_new, j.selected_alpha_old, atol=1e-12, rtol=1e-14)
    p.to_csv(OUT/'model_predictions.csv', index=False)
    pd.DataFrame(tuning).to_csv(OUT/'model_tuning.csv', index=False)
    pd.DataFrame(variance).to_csv(OUT/'pca_variance.csv', index=False)
    pd.DataFrame(loadings).to_csv(OUT/'pca_loadings.csv', index=False)
    perfs, contrasts, loo = [], [], []
    for (target, window), b in p.groupby(['target', 'window']):
        perf, contrast = t8.summarize_block(b, PAIRS, {'target': target, 'window': window})
        perfs.append(perf); contrasts.append(contrast)
        for omit in sorted(b.fold.unique()):
            errors = b[b.fold.ne(omit)].assign(error=lambda d: abs(d.y_true-d.prediction)).groupby('model').error.mean()
            for candidate, baseline in PAIRS:
                loo.append({'target': target, 'window': window, 'omitted_origin': omit,
                            'candidate': candidate, 'baseline': baseline,
                            'improvement_mae': errors[baseline]-errors[candidate]})
        print('summarized', target, window, flush=True)
    perf, contrast = pd.concat(perfs), pd.concat(contrasts)
    perf.to_csv(OUT/'model_performance.csv', index=False)
    contrast.to_csv(OUT/'model_contrasts.csv', index=False)
    pd.DataFrame(loo).to_csv(OUT/'leave_one_origin_out.csv', index=False)
    metadata = {**sc.environment_stamp(), 'input_sha256': sc.sha256_file(t8.INPUT),
                'script_sha256': sc.sha256_file(Path(__file__)),
                'shared_script_sha256': sc.sha256_file(Path(t8.__file__)),
                'protocol_sha256': sc.sha256_file(OUT/'PROTOCOL.md'),
                'task8_predictions_sha256': sc.sha256_file(t8.OUT/'model_predictions.csv'),
                'shared_predictions_verified': len(j),
                'max_shared_prediction_difference': float(abs(j.prediction_new-j.prediction_old).max())}
    (OUT/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(contrast[contrast.fold.eq('pooled')].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
