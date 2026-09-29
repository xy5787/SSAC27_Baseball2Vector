"""Checks that do not require privately acquired inputs."""
from pathlib import Path
import importlib.util
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def test_manifest_hashes_and_publication_boundary():
    spec=importlib.util.spec_from_file_location('release_check',ROOT/'scripts/check_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.validate()


def test_projection_reference_arithmetic_and_cohorts():
    for directory in ['projection','projection_unclipped']:
        frame=pd.read_csv(ROOT/'reference_results'/directory/'baseline_comparison.csv')
        assert len(frame)==12 and set(frame.system)=={'ZiPS','Steamer','Marcel'}
        assert (frame.B2V_increment-(frame.calibrated_baseline_mae-frame.plus_B2V_mae)).abs().max()<1e-12
        assert frame[frame.window.eq('all')].n_test.eq(1706).all()
        assert frame[frame.window.eq('no2020')].n_test.eq(1427).all()
        assert frame.ci_low.le(frame.ci_high).all()


def test_input_inventory_has_twelve_required_projections_and_one_batting_file():
    manifest=json.loads((ROOT/'data/input_manifest.json').read_text())
    required=[x for x in manifest['files'] if x['required']]
    assert len(required)==13
    projections=[x['path'] for x in required if x['path'].startswith('ProjectionDataset/')]
    expected={f'ProjectionDataset/fangraphs-leaderboard-projections_{y}_{s}.csv' for y in [2019,2021,2022,2023,2024,2025] for s in ['z','s']}
    assert set(projections)==expected
    assert manifest['historical_preseason_publication_verified'] is False
