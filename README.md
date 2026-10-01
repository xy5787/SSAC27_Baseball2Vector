# Baseball2Vector — SSAC 2027

Baseball2Vector represents hitters with five interpretable tool scores. This repository reproduces the profile comparisons, next-season forecasts, and incremental information above Marcel, ZiPS, and Steamer.

## Reproduction

Use Python 3.14.6 and the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Obtain the inputs described in data/DATA_RECIPE.md.
python scripts/reproduce.py
```

The analysis requires one FanGraphs batting export and 12 historical projection exports. Follow the [data recipe](data/DATA_RECIPE.md) for seasons, fields, and file locations. FanGraphs inputs require separately authorized access and are not included.

The default run checks input hashes and schemas, recomputes all analyses, and compares 12 aggregate tables with `reference_results/`. Results and the comparison report are written to `results_sd10/`; unclipped projection results are written to `results_zscore/projection/`. Use a fresh checkout for a new run, since existing outputs are protected.

For a newly acquired snapshot with different file hashes:

```bash
python scripts/reproduce.py --input-mode new
```

This checks the input schema, runs the same analysis, and records differences from the reference tables. Completion does not establish reproduction of the submitted numbers. Historical inputs may have changed or become unavailable.

To check inputs only, use `--stage inputs`. To compare an existing run with the references, use `--stage verify` with the same input mode. Individual analysis stages, in order, are `primary`, `projections`, `projection-unclipped`, and `compare-scales`. The primary stage needs only the batting export.

## Methods and results

The [analysis protocol](protocols/reproduction.md) describes the cohorts, model comparisons, and sensitivity analyses. Primary forecasts use clipped 20–80 scores; unclipped scores provide a sensitivity analysis.

| Analysis | Reference tables |
| --- | --- |
| Profile distances and gap closure | `reference_results/diversity/` |
| Forecast comparisons | `reference_results/task8/` |
| Training-selected representative statistics | `reference_results/task10/` |
| Marcel, ZiPS, and Steamer augmentation | `reference_results/projection/` |
| Unclipped projection sensitivity | `reference_results/projection_unclipped/` |

Detailed outputs, including player-level predictions, are generated locally and ignored by Git.

Project code is released under the [MIT License](LICENSE). Third-party inputs have separate [data access and rights conditions](DATA_RIGHTS.md).
