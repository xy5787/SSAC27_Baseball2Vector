# Baseball2Vector — SSAC 2027 analysis

Reproduce transparent five-tool profiles, primary forecast comparisons and incremental information above Marcel, ZiPS and Steamer. This revision uses the refreshed batting export, clipped 20–80 grades for primary forecasts, and unclipped coordinates as a sensitivity.

This repository contains analysis code, protocols, acquisition metadata, aggregate reference results and tests. Current abstract/paper drafts and their figures/tables are excluded. No manuscript artifacts are required or regenerated.

## Reproduction

Tested with Python 3.14.6 and the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
# Place the 13 exports at the paths below, following data/DATA_RECIPE.md.
python scripts/reproduce.py
```

Required local layout (2020 projection files are optional):

```text
data/raw/BattingStats_2018_2026.csv
ProjectionDataset/fangraphs-leaderboard-projections_{YEAR}_{SYSTEM}.csv
```

Use YEAR = 2019, 2021, 2022, 2023, 2024, 2025 and SYSTEM = `z` (ZiPS) or `s` (Steamer). Install `requirements-dev.txt`: the final verification uses pytest. The runner checks all files and schemas before fitting, computes every analysis below, compares all aggregate CSVs with published references, runs the tests, and writes `results_sd10/reproduction_report.json`. A nonzero exit means reproduction was not verified. Numerical-library thread counts default to one unless already configured.

By default, `--input-mode exact` requires the published file hashes. For a fresh provider download whose bytes or values differ, use a fresh checkout and explicitly run:

```bash
python scripts/reproduce.py --input-mode new
```

This applies the same analysis to the new snapshot, records actual input hashes and all reference differences, and runs scientific consistency tests. Frozen-reference tests are explicitly skipped in this mode; a successful new-snapshot run does not establish that the submitted numbers were reproduced. Revised data must still satisfy the recipe/schema and the projection systems' common-cohort checks. Inputs are never silently rewritten, and reference results are never replaced. Some historical snapshots may no longer be obtainable.

To check inputs only, use `--stage inputs`; to verify an already completed run, use `--stage verify`. Pass the same `--input-mode` used for fitting. Existing outputs are protected; use a fresh checkout for another run.

The end-to-end runner was validated in an independent temporary directory with only the release files and 13 input CSVs: **30/30 aggregate tables matched, 57 tests passed, none skipped**. A second complete run with deliberately modified input values and no duplicate identifier columns verified new-snapshot handling: 27 tests passed; 30 frozen-reference tests were intentionally skipped, with differences recorded separately. These are local validation runs, not evidence that the provider still serves the exact historical snapshot. Details are in `VALIDATION.json` under `end_to_end_validation`.

The recorded 2026-09-29 standalone reproduction passed all 44 tests; 574,208 prediction rows matched the research run exactly (maximum difference 0). See [VALIDATION.json](VALIDATION.json).

No other Baseball2Vector checkout, old export, stored prediction, manuscript or network call is needed. Exact source hashes are checked before analysis. If historical inputs are unavailable or revised, see [the recipe](data/DATA_RECIPE.md); a hash cannot recover an unavailable snapshot.

The main study contains 3,070 qualifying seasons and 1,706 test transitions, with 1,774 common focal seasons in descriptive matching. The projection extension adds 2018 history and uses the same main test transitions under a different training/history protocol. [Protocol notes](protocols/reproduction.md) distinguish the two 2020 treatments, reference models and the post-result scale amendment.

Stages can be run separately, in order:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py --stage primary
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py --stage projections
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py --stage projection-unclipped
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py --stage compare-scales
```

`primary` needs only the batting export. Projection stages need the batting export plus 12 required historical projection files. The default `all` runs all four stages followed by verification. Existing completed outputs are protected from overwrite. `reference_results/` contains only aggregate summaries; the runner creates new results in ignored directories. Tests compare them to these references and independently reconstruct paired uncertainty from local predictions. Without local outputs, data-dependent tests skip explicitly; package and reference-consistency tests still run.

Use `scripts/reproduce.py` as the analysis entry point; the retired manuscript and legacy PCA entry points have been removed. See [COMMIT_SCOPE.md](COMMIT_SCOPE.md) for the explicit commit boundary and safe staging command.

Before preparing a release, run `python scripts/check_release.py` to verify file hashes and the Git publication boundary. On a checkout without local input data or generated results, `python -m pytest -q tests` runs the package checks and explicitly skips reproduction-dependent checks. The full 44-test reproduction recorded above requires the frozen inputs and generated outputs.

Project-authored code uses the [MIT License](LICENSE). No FanGraphs data license is granted: [DATA_RIGHTS.md](DATA_RIGHTS.md).

## Find the evidence

| Submitted analysis | Aggregate results under `reference_results/` |
| --- | --- |
| Profile distances and gap closure | `diversity/summary.csv`, `diversity/reference_gap_closure.csv` |
| B2V, current outcome, fixed representatives, 22 inputs, grouped/full PCA | `task8/model_performance.csv`, `task8/model_contrasts.csv` (also mirrored under `task9/`) |
| Training-selected representatives | `task10/performance.csv`, `task10/contrasts.csv`, `task10/selected_variants.csv` |
| Marcel, ZiPS, Steamer augmentation | `projection/baseline_comparison.csv`, system-specific `origin_signs.csv` |
| Unclipped sensitivity and paired scale comparisons | `projection_unclipped/`, `projection/scale_sensitivity.csv` |

Generated tables have matching relative paths under `results_sd10/`, except unclipped projection outputs under `results_zscore/projection/`. Player-level outputs stay local and ignored. The end-to-end scope is the submitted descriptive and predictive analyses, not the historical retrieval experiments, user-study materials, exploratory counting WAR, PCA-loading exports, or manuscript/figure rendering. Historical protocols retain those original plans for provenance; `protocols/reproduction.md` defines this release's scope.
