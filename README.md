# Baseball2Vector — SSAC 2027

**Interpretable Five-Tool Profiles with Predictive Utility** — SSAC 2027 submission materials.

This repository contains the final abstract, one figure, one table, and the code needed to reproduce their analyses from locally acquired FanGraphs data. It excludes development notebooks, presentation archives, and player-level data.

This is a standalone submission snapshot. All required project code is included here; no other Baseball2Vector repository or presentation archive is needed to run the analysis.

## Contents

- `abstract/`: submission text, Figure 1 (PDF), and Table 1 (PDF/CSV/Markdown).
- `scripts/`, `src/`: input preparation, prediction comparisons, sensitivity analyses, diversity analysis, and artifact generation.
- `data/`: acquisition recipe, schema, and frozen-snapshot checks; no observations.
- `protocols/`: analysis specifications.
- `reference_results/`: aggregate results used to check reproduction.
- `tests/`: checks for comparison construction and submission artifacts.

## Reproduce

Use Python 3.14.6 (the tested environment), preferably in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Follow [the acquisition recipe](data/README.md) to obtain the required exports through authorized access. FanGraphs CSV exports may require membership. Then run from this directory:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py
python -m pytest -q tests
```

If you already have the private 31-column portable snapshot described in the schema, use `python scripts/reproduce.py --input /path/to/snapshot.csv`. No network download or scraping is performed. Generated inputs, cohorts, predictions, and results remain in ignored `data/raw/`, `data/processed/`, and `results_sd10/` directories. The script regenerates the figure and table in `abstract/`.

The frozen study covers 2019–2025, with 3,070 qualifying player-seasons and 1,706 forecast transitions. The diversity analysis uses 1,777 common focal seasons from 626 players. The artifact builder checks the frozen cohort size; if newly acquired data differ, investigate and document the differences rather than silently treating them as the submitted snapshot. Historical provider revisions can prevent exact reproduction; hashes do not recover unavailable data.

## Rights and submission status

Project-authored code is provided under the [MIT License](LICENSE). This grants no rights to FanGraphs data; see [DATA_RIGHTS.md](DATA_RIGHTS.md). Raw exports and player-level derived files are intentionally absent. SSAC confirmation that source links and a reconstruction recipe satisfy its current data-sharing requirement is pending.
