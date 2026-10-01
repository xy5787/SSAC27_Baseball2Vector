"""Exercise downloader-facing schema errors and explicit snapshot selection."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from input_validation import inspect_inputs, validate_table

@pytest.fixture
def inputs(tmp_path):
    schema=json.loads((ROOT/'data/input_schema.json').read_text())
    rows=[]
    for year in range(2018,2026):
        for player,pa in [(1,20),(2,200)]:
            row={c:.5 for c in schema['source_fields']}
            row.update(PlayerId=player,Name=f'Example {player}',Team='EX',Season=year,PA=pa,Age=28)
            rows.append(row)
    path=tmp_path/'data/raw/BattingStats_2018_2026.csv'
    path.parent.mkdir(parents=True);pd.DataFrame(rows).to_csv(path,index=False)
    (tmp_path/'data/input_schema.json').write_text(json.dumps(schema))
    item=dict(path=str(path.relative_to(tmp_path)),required=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path/'data/input_manifest.json').write_text(json.dumps(dict(files=[item])))
    return tmp_path,path,schema


def test_exact_and_new_downloads_are_distinguished(inputs):
    root,path,_=inputs
    assert inspect_inputs(root)[0]['matches_snapshot']
    # Equivalent CSV bytes need not retain the original file hash.
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(SystemExit,match='Snapshot mismatch'):
        inspect_inputs(root)
    checked=inspect_inputs(root,mode='new')
    assert not checked[0]['matches_snapshot']
    assert checked[0]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('problem',['missing_column','duplicate','percent_units','filtered_history','duplicate_identifier_column'])
def test_schema_errors_fail_before_fitting(inputs,problem):
    root,path,schema=inputs;frame=pd.read_csv(path)
    if problem=='missing_column':frame=frame.drop(columns='UBR')
    elif problem=='duplicate':frame=pd.concat([frame,frame.iloc[:1]])
    elif problem=='percent_units':frame['Contact%']=80
    elif problem=='filtered_history':frame=frame[frame.PA.ge(100)]
    else:frame['Season.1']=frame.Season+1
    frame.to_csv(path,index=False)
    with pytest.raises(SystemExit,match='Invalid'):
        inspect_inputs(root,mode='new')


def test_missing_files_are_reported_together(inputs):
    root,path,_=inputs;path.unlink()
    manifest=json.loads((root/'data/input_manifest.json').read_text())
    manifest['files'].append(dict(path='ProjectionDataset/missing.csv',required=True,sha256='none'))
    (root/'data/input_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(SystemExit) as error:inspect_inputs(root)
    assert 'BattingStats_2018_2026.csv' in str(error.value) and 'ProjectionDataset/missing.csv' in str(error.value)


def test_optional_duplicate_columns_and_zero_pa_missing_outcome(inputs):
    _,path,schema=inputs;frame=pd.read_csv(path)
    frame.loc[0,['PA','wRC+']]=[0,float('nan')]
    for c in ['Name','Team','Season']:frame[c+'.1']=frame[c]
    frame.to_csv(path,index=False)
    assert validate_table(path,schema)['rows']==16


def test_projection_duplicate_ids_are_rejected(inputs):
    _,path,schema=inputs
    pd.DataFrame({'PlayerId':[1,1],'wRC+':[100,100],'WAR':[1,1],'PA':[200,200]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='duplicate projection'):validate_table(path,schema,True)


def test_verification_rejects_missing_outputs_before_tests(inputs,monkeypatch):
    import reproduce
    root,_,_=inputs
    (root/'results_sd10').mkdir()
    ref=root/'reference_results';ref.mkdir();(ref/'missing.csv').write_text('value\n1\n')
    monkeypatch.setattr(reproduce,'ROOT',root)
    with pytest.raises(SystemExit,match='Missing output'):
        reproduce.verify('exact')
    report=json.loads((root/'results_sd10/reproduction_report.json').read_text())
    assert report['status']=='failed' and not report['all_outputs_present']


def test_verification_rejects_inputs_changed_after_fitting(inputs,monkeypatch):
    import reproduce
    root,path,_=inputs
    initial=inspect_inputs(root)
    (root/'results_sd10').mkdir()
    (root/'results_sd10/run_inputs.json').write_text(json.dumps(dict(files=initial)))
    path.write_bytes(path.read_bytes()+b'\n')
    monkeypatch.setattr(reproduce,'ROOT',root)
    monkeypatch.setenv('B2V_INPUT_MODE','new')
    with pytest.raises(SystemExit,match='Inputs changed since fitting'):
        reproduce.verify('new')
