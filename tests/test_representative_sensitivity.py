"""Verify the stronger five-statistic comparator and its held-out comparisons."""
from pathlib import Path
import json
import os
import pytest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_sd10/task10'
pytestmark=pytest.mark.skipif(not (OUT/'predictions.csv.gz').exists(), reason='Run reproduce.py --stage primary with local inputs')
KEY=['target','window','context','fold']
ROW=['player_id','Season_t']


def test_variant_pool_is_local_five_statistic_sensitivity():
    v=pd.read_csv(OUT/'variants.csv')
    original=set(json.loads(v.loc[v.variant.eq('original5'),'statistics'].iloc[0]))
    assert len(v)==23 and v.kind.eq('swap').sum()==17 and v.kind.eq('drop').sum()==5
    for r in v.itertuples():
        stats=set(json.loads(r.statistics))
        if r.kind=='swap':
            assert len(stats)==5 and len(stats-original)==1 and len(original-stats)==1
        if r.kind=='drop':
            assert len(stats)==4 and stats<original


def test_subset_selection_uses_inner_cv_and_selected_predictions():
    tuning=pd.read_csv(OUT/'inner_tuning.csv.gz')
    selection=pd.read_csv(OUT/'selected_variants.csv')
    pred=pd.read_csv(OUT/'predictions.csv.gz')
    assert len(selection)==32  # 2 targets x 2 contexts x (5 + 3 origins)
    for row in selection.itertuples():
        mask=np.ones(len(tuning),dtype=bool);pmask=np.ones(len(pred),dtype=bool)
        for key in KEY:
            mask &= tuning[key].eq(getattr(row,key)).to_numpy()
            pmask &= pred[key].eq(getattr(row,key)).to_numpy()
        eligible=tuning[mask & tuning.kind.ne('drop')]
        best=eligible.sort_values(['inner_mae','variant_order','alpha'],kind='stable').iloc[0]
        assert row.model==best.model
        np.testing.assert_allclose(row.alpha,best.alpha)
        np.testing.assert_allclose(row.inner_mae,best.inner_mae)
        block=pred[pmask].set_index(['model',*ROW])
        selected=block.loc['selected5'].sort_index()
        source=block.loc[row.model].sort_index()
        pd.testing.assert_index_equal(selected.index,source.index)
        np.testing.assert_array_equal(selected.prediction,source.prediction)


def test_original_predictions_and_reported_mae_differences_reproduce():
    pred=pd.read_csv(OUT/'predictions.csv.gz')
    old=pd.read_csv(ROOT/'results_sd10/task8/model_predictions.csv')
    for (target,window,context),block in pred.groupby(['target','window','context']):
        model='representative5' if context=='standalone' else 'history+representative5'
        expected=old[old.target.eq(target)&old.window.eq(window)&old.model.eq(model)].set_index(ROW)
        observed=block[block.model.eq('original5')].set_index(ROW)
        np.testing.assert_allclose(observed.prediction,expected.loc[observed.index,'prediction'],atol=1e-8,rtol=0)
    p=pd.read_csv(OUT/'performance.csv').query("fold == 'pooled'").set_index(['target','window','context','model'])
    c=pd.read_csv(OUT/'contrasts.csv')
    for row in c.itertuples():
        base=(row.target,row.window,row.context)
        expected=p.loc[(*base,row.baseline),'mae']-p.loc[(*base,row.candidate),'mae']
        np.testing.assert_allclose(row.improvement_mae,expected,atol=1e-10)
        observed=pred[pred.target.eq(row.target)&pred.window.eq(row.window)&pred.context.eq(row.context)&pred.model.eq(row.candidate)]
        assert row.n_test==len(observed)
        if os.environ.get('B2V_INPUT_MODE', 'exact') == 'exact':
            assert row.n_test==(1706 if row.window=='all' else 1073)
        assert row.ci_low<row.improvement_mae<row.ci_high
