"""Reproduce refreshed B2V analyses from local FanGraphs exports, without writing papers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ['all', 'inputs', 'primary', 'projections', 'projection-unclipped', 'compare-scales', 'verify']


def check_inputs(projections=True):
    from input_validation import inspect_inputs
    return inspect_inputs(ROOT, projections, os.environ.get('B2V_INPUT_MODE', 'exact'))


def projection_table(pi):
    import numpy as np
    import pandas as pd
    rows = []
    for system, prefix in [('zips', 'Z'), ('steamer', 'S')]:
        perf = pd.read_csv(pi.OUT/system/'performance.csv')
        ct = pd.read_csv(pi.OUT/system/'contrasts.csv')
        perf = perf[perf.origin.eq('pooled')]
        ct = ct[ct.origin.eq('pooled')]
        for tk in ['wrc_plus', 'war_rate']:
            for window in ['all', 'no2020']:
                models = [('ZiPS' if system == 'zips' else 'Steamer', prefix+'1', prefix+'2')]
                if system == 'zips':
                    models.append(('Marcel', 'R1', 'R2'))
                for label, baseline, candidate in models:
                    q = perf[perf.target.eq(tk) & perf.window.eq(window)].set_index('model')
                    r = ct[ct.target.eq(tk) & ct.window.eq(window) & ct.baseline.eq(baseline) & ct.candidate.eq(candidate)].iloc[0]
                    rows.append(dict(target=tk, window=window, system=label, n_test=r.n_test, n_players=r.n_players,
                        calibrated_baseline_mae=q.loc[baseline,'mae'], plus_B2V_mae=q.loc[candidate,'mae'],
                        B2V_increment=r.improvement_mae, ci_low=r.ci_low, ci_high=r.ci_high,
                        raw_projection_mae=q.loc[prefix+'0','mae'] if label!='Marcel' and tk=='wrc_plus' else np.nan))
    pd.DataFrame(rows).to_csv(pi.OUT/'baseline_comparison.csv', index=False)


def projections(pi, unclipped=False):
    pi.OUT = ROOT / ('results_zscore/projection' if unclipped else 'results_sd10/projection')
    pi.PRIVATE = ROOT / 'data/raw/generated' / ('projection_unclipped' if unclipped else 'projection')
    if pi.OUT.exists() or pi.PRIVATE.exists():
        raise SystemExit('Projection output already exists. Preserve it and use a fresh checkout to rerun.')
    inputs = check_inputs()
    pi.OUT.mkdir(parents=True)
    pi.PRIVATE.mkdir(parents=True)
    protocol = ROOT / 'protocols/reproduction.md'
    (pi.OUT/'PRESPEC.md').write_bytes(protocol.read_bytes())
    pi.dump(pi.OUT/'prespec_inputs.json', dict(inputs=inputs, scale=pi.sc.B2V_SCALE,
        protocol_sha256=pi.sha(protocol), note='Reproduction run; not a new preregistration.'))
    pi.RUN_COMMAND = 'python scripts/reproduce.py --stage ' + ('projection-unclipped' if unclipped else 'projections')
    for system in ['zips', 'steamer']:
        pi.run_production(system)
    projection_table(pi)


def compare_scales(pi):
    import numpy as np
    import pandas as pd
    rows = []
    keys = ['target','window','model',*pi.KEY]
    validation = {}
    clipped_frames = {}
    for system, prefix in [('zips','Z'),('steamer','S')]:
        new = pd.read_csv(ROOT/'data/raw/generated/projection'/system/'predictions.csv')
        old = pd.read_csv(ROOT/'data/raw/generated/projection_unclipped'/system/'predictions.csv')
        clipped_frames[system] = new
        j = new.merge(old, on=keys, validate='one_to_one', suffixes=('_clipped','_unclipped'))
        assert len(j)==len(new)==len(old)
        for col in ['y_true','Season_t1','origin','n_train','marcel_seasons_used']:
            np.testing.assert_allclose(j[col+'_clipped'], j[col+'_unclipped'], atol=1e-10, rtol=0)
        unchanged = j[j.model.isin([prefix+'0',prefix+'1',prefix+'3','R1'])]
        np.testing.assert_allclose(unchanged.prediction_clipped, unchanged.prediction_unclipped, atol=1e-10, rtol=0)
        assert unchanged.selected_alpha_clipped.fillna(-1).eq(unchanged.selected_alpha_unclipped.fillna(-1)).all()
        validation[system] = dict(matched_rows=len(j), unchanged_rows=len(unchanged),
            max_unchanged_difference=float(abs(unchanged.prediction_clipped-unchanged.prediction_unclipped).max()))
        for tk in ['wrc_plus','war_rate']:
            for window in ['all','no2020']:
                models = [('ZiPS' if system=='zips' else 'Steamer',prefix+'2')]
                if system=='zips': models.append(('Marcel','R2'))
                for label, model in models:
                    a = new[new.target.eq(tk)&new.window.eq(window)&new.model.eq(model)].assign(model='clipped')
                    b = old[old.target.eq(tk)&old.window.eq(window)&old.model.eq(model)].assign(model='unclipped')
                    _, direct = pi.summarize_block(pd.concat([a,b]), [('clipped','unclipped')], 'pooled')
                    rows.append(dict(system=label, **direct[0]))
    a,b = (clipped_frames[s][clipped_frames[s].model.str.startswith('R')] for s in ['zips','steamer'])
    j=a.merge(b,on=keys,validate='one_to_one',suffixes=('_z','_s'))
    assert len(j)==len(a)==len(b)
    np.testing.assert_allclose(j.prediction_z,j.prediction_s,atol=1e-12,rtol=0)
    assert j.selected_alpha_z.eq(j.selected_alpha_s).all()
    out=ROOT/'results_sd10/projection'
    pd.DataFrame(rows).to_csv(out/'scale_sensitivity.csv',index=False)
    pi.dump(out/'scale_validation.json',dict(systems=validation,common_reference_rows=len(j),all_passed=True))


def verify(input_mode):
    """Compare regenerated aggregate tables with the published references."""
    import pandas as pd
    import platform
    inputs = check_inputs()
    saved=ROOT/'results_sd10/run_inputs.json'
    if saved.exists():
        before=json.loads(saved.read_text())['files']
        if {x['path']:x['sha256'] for x in before} != {x['path']:x['sha256'] for x in inputs}:
            raise SystemExit('Inputs changed since fitting. Use a fresh checkout; existing results cannot verify these inputs.')
    out = ROOT/'results_sd10'
    if not out.is_dir():
        raise SystemExit('No results. Run scripts/reproduce.py first.')
    references=sorted((ROOT/'reference_results').rglob('*.csv'))
    if not references:
        raise SystemExit('No reference tables found in reference_results/.')
    records=[]
    for reference in references:
        relative=reference.relative_to(ROOT/'reference_results')
        actual=(ROOT/'results_zscore/projection'/Path(*relative.parts[1:])
                if relative.parts[0]=='projection_unclipped' else out/relative)
        record=dict(path=str(relative), present=actual.is_file(), matches_reference=False)
        if actual.is_file():
            a,b=pd.read_csv(actual),pd.read_csv(reference)
            record.update(rows=len(a), reference_rows=len(b))
            try:
                assert set(a.columns)==set(b.columns), 'Column sets differ'
                keys=[c for c in b if pd.api.types.is_string_dtype(b[c].dtype)]
                if keys:
                    a=a.sort_values(keys,kind='stable',na_position='last').reset_index(drop=True)
                    b=b.sort_values(keys,kind='stable',na_position='last').reset_index(drop=True)
                pd.testing.assert_frame_equal(a[b.columns],b,check_exact=False,rtol=1e-8,atol=1e-8)
                record['matches_reference']=True
            except AssertionError as exc:
                record['difference']=str(exc)[:1000]
        records.append(record)
    complete=all(x['present'] for x in records)
    matched=all(x['matches_reference'] for x in records)
    report=dict(input_mode=input_mode, exact_input_hashes=all(x['matches_snapshot'] for x in inputs),
                python=platform.python_version(), inputs=inputs, aggregates=records,
                all_outputs_present=complete, all_references_match=matched, status='failed')
    report_path=out/'reproduction_report.json'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    if not complete:
        raise SystemExit('Missing output tables. See results_sd10/reproduction_report.json.')
    good=matched or input_mode=='new'
    report['status']=('exact_reproduction_verified' if report['exact_input_hashes'] and matched
                      else 'new_snapshot_analysis_completed') if good else 'failed'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(f"Verification: {report['status']}. Report: {report_path}",flush=True)
    if not good:
        raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=STAGES,default='all')
    parser.add_argument('--input-mode', choices=['exact', 'new'], default='exact',
                        help='exact: require published hashes; new: validate a newly downloaded snapshot and report differences')
    args=parser.parse_args()
    os.environ['B2V_INPUT_MODE']=args.input_mode
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ['B2V_SCALE']='zscore' if args.stage=='projection-unclipped' else 'sd10'
    if args.stage=='inputs':
        print(f'Verified {len(check_inputs())} required input files.');return
    if args.stage == 'verify':
        verify(args.input_mode); return
    # Fail before any expensive fitting if any required input is absent or invalid.
    inputs = check_inputs(projections=args.stage != 'primary')
    if args.stage == 'all':
        for name in ['results_sd10', 'results_zscore', 'data/raw/generated']:
            if (ROOT/name).exists():
                raise SystemExit(f'Existing {name}; use a fresh checkout to preserve prior runs.')
        (ROOT/'results_sd10').mkdir()
        (ROOT/'results_sd10/run_inputs.json').write_text(json.dumps(
            dict(input_mode=args.input_mode, files=inputs), indent=2)+'\n')
    import production_increment as pi
    if args.stage in ['all','primary']:
        check_inputs(projections=False)
        if (ROOT/'results_sd10/task8/model_predictions.csv').exists():
            raise SystemExit('Primary output already exists; use a fresh checkout to preserve it.')
        import primary_analysis
        primary_analysis.run()
    if args.stage in ['all','projections','projection-unclipped']:
        projections(pi,unclipped=args.stage=='projection-unclipped')
    if args.stage=='all':
        subprocess.run([sys.executable,__file__,'--stage','projection-unclipped','--input-mode',args.input_mode],check=True,env=os.environ.copy())
    if args.stage in ['all','compare-scales']:
        compare_scales(pi)
    if args.stage == 'all':
        verify(args.input_mode)


if __name__=='__main__': main()
