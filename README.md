# Baseball2Vector — SSAC 2027 analysis

Reproduce transparent five-tool profiles, primary forecast comparisons and incremental information above Marcel, ZiPS and Steamer. This revision uses the refreshed batting export, clipped 20–80 grades for primary forecasts, and unclipped coordinates as a sensitivity.

The commit contains analysis code, protocols, acquisition metadata, aggregate reference results and tests. Current abstract/paper drafts and their figures/tables are excluded. The remote removal of the historical abstract directory is preserved; no manuscript artifacts are required or regenerated.

## Reproduce

Tested with Python 3.14.6 and the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
# Obtain local exports following data/DATA_RECIPE.md.
python scripts/reproduce.py --stage inputs
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/reproduce.py
python -m pytest -q tests
```

A clean standalone reproduction passed all 44 tests; 574,208 prediction rows matched the research run exactly (maximum difference 0). See [VALIDATION.json](VALIDATION.json).

No other Baseball2Vector checkout, old export, stored prediction, manuscript or network call is needed. Exact source hashes are checked before analysis. If historical inputs are unavailable or revised, see [the recipe](data/DATA_RECIPE.md); a hash cannot recover an unavailable snapshot.

The main study contains 3,070 qualifying seasons and 1,706 test transitions, with 1,774 common focal seasons in descriptive matching. The projection extension adds 2018 history and uses the same main test transitions under a different training/history protocol. [Protocol notes](protocols/reproduction.md) distinguish the two 2020 treatments, reference models and the post-result scale amendment.

Stages can be run separately, in order:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py --stage primary
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py --stage projections
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py --stage projection-unclipped
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py --stage compare-scales
```

`primary` needs only the batting export. Projection stages need the batting export plus 12 required historical projection files. The default `all` runs all four stages. Existing completed outputs are protected from overwrite. `reference_results/` contains only aggregate summaries; the runner creates new results in ignored directories. Tests compare them to these references and independently reconstruct paired uncertainty from local predictions. Without local outputs, data-dependent tests skip explicitly; package and reference-consistency tests still run.

The former two-export `--input` workflow and manuscript asset entry points are historical; use `scripts/reproduce.py` for this revision. See [COMMIT_SCOPE.md](COMMIT_SCOPE.md) for the explicit commit boundary and safe staging command.

Project-authored code uses the [MIT License](LICENSE). No FanGraphs data license is granted: [DATA_RIGHTS.md](DATA_RIGHTS.md).
