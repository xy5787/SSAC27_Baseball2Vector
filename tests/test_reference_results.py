"""Check independent reruns against aggregate frozen references."""
from pathlib import Path
import pandas as pd
import pytest
import os
ROOT=Path(__file__).resolve().parents[1]
REFERENCES=sorted((ROOT/'reference_results').rglob('*.csv'))

@pytest.mark.parametrize('reference',REFERENCES,ids=lambda p:str(p.relative_to(ROOT/'reference_results')))
def test_aggregate_reproduction(reference):
    if os.environ.get('B2V_INPUT_MODE') == 'new':
        pytest.skip('New snapshot: differences are recorded in reproduction_report.json')
    relative=reference.relative_to(ROOT/'reference_results')
    if relative.parts[0]=='projection_unclipped':
        actual=ROOT/'results_zscore/projection'/Path(*relative.parts[1:])
    else:
        actual=ROOT/'results_sd10'/relative
    if not actual.exists(): pytest.skip('Run scripts/reproduce.py with local inputs first')
    a,b=pd.read_csv(actual),pd.read_csv(reference)
    assert set(a.columns)==set(b.columns)
    keys=[c for c in b if pd.api.types.is_string_dtype(b[c].dtype)]
    if keys:
        a=a.sort_values(keys,kind='stable',na_position='last').reset_index(drop=True)
        b=b.sort_values(keys,kind='stable',na_position='last').reset_index(drop=True)
    pd.testing.assert_frame_equal(a[b.columns],b,check_exact=False,rtol=1e-8,atol=1e-8)
