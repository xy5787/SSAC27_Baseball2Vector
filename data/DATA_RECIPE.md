# Data acquisition

Obtain the following exports through authorized FanGraphs access. Raw exports and player-level derived data are not distributed. `input_schema.json` lists required fields and units; `input_manifest.json` records file hashes and row counts, including two optional outcome-2020 projection files.

## Batting statistics

Source: [FanGraphs MLB batting leaderboards](https://www.fangraphs.com/leaders/major-league).

1. Select MLB regular-season batting, all teams and players, seasons **2018–2026**, individual season rows, and **PA minimum 0**. Use player-season totals across teams, not separate team stints. Low-PA rows are needed for Marcel history.
2. Include the identifiers, outcomes, and 22 constituent statistics in `input_schema.json`. Use the supported CSV export. The study snapshot contains 12,934 rows and 465 columns.
3. Preserve missing values and fractional rates (`0.80`, not `80`, for 80%). Do not replace missing UBR or tracking fields with zero. Duplicate `Name.1`, `Team.1`, and `Season.1` columns are optional but must agree with the corresponding identifiers when present.
4. Save the file as `data/raw/BattingStats_2018_2026.csv`.

Study snapshot SHA-256: `2130d2bc07ad64f3429c60a85628a1da9938b9dc4777ddc98277855bd1289e82`.

The primary analysis uses 2019–2025; the projection extension uses 2018–2025. The 2026 rows are excluded from all fitting, history, and evaluation. Cohorts and filtering are described in the [analysis protocol](../protocols/reproduction.md).

## Historical projections

Source: [FanGraphs projections](https://www.fangraphs.com/projections).

The acquisition route was **Projections → Batters → Historical Preseason Projections (Members Exclusive) → Dashboard → Data Export**. Membership is required; interface labels and historical availability may change.

- Export both **ZiPS** and **Steamer** for outcome years **2019, 2021, 2022, 2023, 2024, 2025**: 12 required files. Outcome-2020 exports are optional and unused.
- Include all hitters and the original `PlayerId`, `wRC+`, `WAR`, and `PA` columns. Do not substitute rest-of-season, depth-chart, or in-season projections.
- Save as `ProjectionDataset/fangraphs-leaderboard-projections_{YEAR}_{SYSTEM}.csv`, using `z` for ZiPS and `s` for Steamer. YEAR is the projected outcome season; a 2021 projection joins to 2020 batting inputs.

Matching uses PlayerId and outcome year. Missing IDs are excluded; duplicate IDs fail validation. Projection wRC+ must be finite; projection WAR/600 requires finite WAR and positive PA.

## Input checks and reproduction

After installing the dependencies in the root README:

```bash
python scripts/reproduce.py --stage inputs
python scripts/reproduce.py
```

The default mode requires the study's file hashes and validates schemas before fitting. The full run compares the 12 published aggregate tables and saves `results_sd10/reproduction_report.json`. Generated inputs and outputs remain in Git-ignored directories.

For newly downloaded files with different hashes, use a fresh checkout and run `python scripts/reproduce.py --input-mode new`. This records the actual hashes and reference differences without replacing the published results. Successful completion does not establish exact reproduction of the study snapshot.

Download dates for the study files were not recorded, and their historical preseason publication times remain unverified. Record dates and export settings for new acquisitions. Provider revisions may prevent recovery of the exact inputs; hashes identify files but cannot restore unavailable snapshots.

See [DATA_RIGHTS.md](../DATA_RIGHTS.md) for access and redistribution conditions.
