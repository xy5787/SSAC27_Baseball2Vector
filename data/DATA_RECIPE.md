# FanGraphs data recipe

This recipe accompanies the SSAC 2027 Baseball2Vector analysis. Obtain the inputs independently from [FanGraphs MLB batting leaderboards](https://www.fangraphs.com/leaders/major-league) through its [member export interface](https://blogs.fangraphs.com/contact/). No FanGraphs rows are distributed here.

## Acquire

Export MLB regular-season batting data for **2019–2025**, all players and teams, **PA/IP minimum 0**, one full-season total per player and season (including players who changed teams). Preserve missing values and source units. The required 31 fields and fractional rate columns are listed in [input_schema.json](input_schema.json).

Save the 2019–2020 export as data/raw/batting_stats_2019_2020.csv and the 2021–2025 export as data/raw/batting_stats_2021_2025_all_players.csv. The second name is required by the loader. The author's historical second file also contained 2026, which the code excludes. Record your download date, exact URL, filters, and any changes in field availability or units. The original download date was not recorded.

## Reproduce

From the release directory, install the environment as described in [README.md](../README.md), then run:

~~~bash
mkdir -p data/processed
python scripts/task8_submission_strengthening.py --stage export
OPENBLAS_NUM_THREADS=1 python scripts/reproduce.py
~~~

The first script reads local exports, checks unique player-season keys, and creates the 31-column portable input. It **overwrites** any existing local portable input and manifest. The second script runs the analysis without downloading data. If you already have the exact portable input, use the --input option described in [README.md](../README.md).

## Identify the study snapshot

The original portable input has **10,083 rows** and SHA-256 **ae5c148dd9a5b79b8154facc88967119e56b210cd5aab0adc38a6c238edbeb5f**. Its analysis cohort has **3,070** player-seasons with at least 100 PA. Original export hashes, season counts, and column definitions are in [input_schema.json](input_schema.json). Compare a new export's keys, values, units, missingness, and counts before treating it as the same snapshot.

The SSAC organizers advised that source links, acquisition instructions, code, and snapshot identifiers can support reproducibility **when researchers can independently obtain the exact inputs**. Historical revisions or restricted access can prevent exact reproduction; hashes identify the original files but cannot restore unavailable data. Document any differences rather than adjusting new data to match the reported counts. FanGraphs redistribution permission has not been established; see [DATA_RIGHTS.md](../DATA_RIGHTS.md).
