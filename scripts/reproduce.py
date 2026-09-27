"""Reproduce submission results using locally acquired FanGraphs inputs."""
import argparse
import os
from pathlib import Path
import shutil
os.environ['B2V_SCALE'] = 'sd10'
import pandas as pd
import task8_submission_strengthening as t8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Existing private portable input CSV')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    t8.INPUT.parent.mkdir(parents=True, exist_ok=True)
    for task in ['task8', 'task9', 'task10']:
        out = t8.sc.RESULTS_DIR / task
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / 'protocols' / (task + '.md'), out / 'PROTOCOL.md')
    if args.input:
        source = args.input.resolve()
        if t8.INPUT.exists() and t8.sc.sha256_file(source) != t8.sc.sha256_file(t8.INPUT):
            raise SystemExit('Existing input differs. Use a fresh checkout to preserve it.')
        if source != t8.INPUT.resolve():
            shutil.copyfile(source, t8.INPUT)
    if not t8.INPUT.exists():
        t8.export_inputs()
    raw = pd.read_csv(t8.INPUT)
    cohort = t8.cohort_from_raw(raw)
    cohort.to_csv(t8.OUT / 'cohort.csv', index=False)
    t8.run_models(raw, cohort)
    predictions = pd.read_csv(t8.OUT / 'model_predictions.csv')
    perfs, contrasts = [], []
    for (target, window), block in predictions.groupby(['target', 'window']):
        perf, contrast = t8.summarize_block(block, t8.model_pairs(), {'target': target, 'window': window})
        perfs.append(perf)
        contrasts.append(contrast)
    pd.concat(perfs).to_csv(t8.OUT / 'model_performance.csv', index=False)
    pd.concat(contrasts).to_csv(t8.OUT / 'model_contrasts.csv', index=False)
    import task9_full_pca
    import task10_representative_sensitivity
    import build_submission_assets
    task9_full_pca.main()
    task10_representative_sensitivity.run()
    build_submission_assets.main()


if __name__ == '__main__':
    main()
