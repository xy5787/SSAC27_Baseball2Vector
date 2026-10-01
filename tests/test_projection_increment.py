"""Independent checks of the scale amendment on saved predictions and sources."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import pytest
import os
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results_sd10/projection'
PRIVATE=ROOT/'data/raw/generated'
KEY=['target','window','model','player_id','Season_t']
pytestmark=pytest.mark.skipif(not (PRIVATE/'projection_unclipped/steamer/predictions.csv').exists(),reason='Local prediction artifacts required')

@lru_cache(None)
def predictions(system,scale):
    folder='projection' if scale=='clipped' else 'projection_unclipped'
    return pd.read_csv(PRIVATE/folder/system/'predictions.csv')

def bootstrap(delta,ids):
    # Aggregate errors by player, then resample player sums/counts rather than rows.
    sums=pd.DataFrame({'id':ids,'delta':delta}).groupby('id',sort=True).delta.agg(['sum','size'])
    rng=np.random.default_rng(42);a=sums['sum'].to_numpy();n=sums['size'].to_numpy();draws=[]
    for _ in range(2000):
        ix=rng.choice(len(sums),len(sums),replace=True)
        draws.append(a[ix].sum()/n[ix].sum())
    return np.quantile(draws,[.025,.975])

def get(p,target,window,model):
    return p[p.target.eq(target)&p.window.eq(window)&p.model.eq(model)].set_index(['player_id','Season_t']).sort_index()

def test_only_b2v_models_change_and_all_keys_targets_splits_match():
    for system,prefix in [('zips','Z'),('steamer','S')]:
        c=predictions(system,'clipped');u=predictions(system,'unclipped')
        m=c.merge(u,on=KEY,suffixes=('_c','_u'),validate='one_to_one');assert len(m)==len(c)==len(u)
        for col in ['y_true','Season_t1','origin','n_train','marcel_seasons_used']:
            np.testing.assert_allclose(m[col+'_c'],m[col+'_u'],atol=0,rtol=0)
        b=m[m.model.isin([prefix+'0',prefix+'1',prefix+'3','R1'])]
        np.testing.assert_allclose(b.prediction_c,b.prediction_u,atol=0,rtol=0)
        assert c.Season_t1.ne(2020).all() and c.Season_t1.le(2025).all()
        assert c[c.window.eq('no2020')].Season_t.ne(2020).all()
    z=predictions('zips','clipped');s=predictions('steamer','clipped')
    z=z[z.model.str.startswith('R')].set_index(KEY).sort_index();s=s[s.model.str.startswith('R')].set_index(KEY).sort_index()
    pd.testing.assert_series_equal(z.prediction,s.prediction)

def test_clipped_increments_and_cis_reconstruct_from_errors():
    summary=pd.read_csv(RESULT/'baseline_comparison.csv');assert len(summary)==12
    for r in summary.itertuples():
        system='steamer' if r.system=='Steamer' else 'zips'
        prefix={'Steamer':'S','ZiPS':'Z','Marcel':'R'}[r.system]
        p=predictions(system,'clipped');a=get(p,r.target,r.window,prefix+'1');b=get(p,r.target,r.window,prefix+'2')
        assert a.index.equals(b.index) and len(a)==r.n_test
        if os.environ.get('B2V_INPUT_MODE', 'exact') == 'exact':
            assert r.n_test==(1706 if r.window=='all' else 1427)
        ea=abs(a.y_true-a.prediction);eb=abs(b.y_true-b.prediction)
        np.testing.assert_allclose([ea.mean(),eb.mean(),(ea-eb).mean()],[r.calibrated_baseline_mae,r.plus_B2V_mae,r.B2V_increment],atol=1e-12,rtol=0)
        np.testing.assert_allclose(bootstrap((ea-eb).to_numpy(),a.index.get_level_values('player_id')),[r.ci_low,r.ci_high],atol=1e-12,rtol=0)

def test_scale_sensitivity_uses_paired_errors_not_subtracted_interval_bounds():
    summary=pd.read_csv(RESULT/'scale_sensitivity.csv');assert len(summary)==12
    for r in summary.itertuples():
        system='steamer' if r.system=='Steamer' else 'zips';prefix={'Steamer':'S','ZiPS':'Z','Marcel':'R'}[r.system]
        c=get(predictions(system,'clipped'),r.target,r.window,prefix+'2');u=get(predictions(system,'unclipped'),r.target,r.window,prefix+'2')
        assert c.index.equals(u.index)
        delta=abs(c.y_true-c.prediction)-abs(u.y_true-u.prediction)
        np.testing.assert_allclose(delta.mean(),r.improvement_mae,atol=1e-12,rtol=0)
        np.testing.assert_allclose(bootstrap(delta.to_numpy(),c.index.get_level_values('player_id')),[r.ci_low,r.ci_high],atol=1e-12,rtol=0)
