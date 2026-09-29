# FanGraphs data acquisition and reconstruction

This revision uses the refreshed 2018–2026 batting export and historical ZiPS/Steamer exports. Obtain the files through your own authorized FanGraphs access. Raw data and player-level derived files are not distributed. Acquisition dates were not recorded; file modification times are not substitutes. The historical preseason publication times of the projection files remain unverified.

## 1. Batting actuals

Source: [FanGraphs MLB batting leaderboards](https://www.fangraphs.com/leaders/major-league).

1. Select MLB regular-season **batting**, all teams and players, seasons **2018–2026**, individual season rows, and **PA minimum 0**. Do not select qualified hitters or apply PA ≥100 during export: Marcel requires low-PA history.
2. Export player-season totals across teams, not separate team stints. Include the combined batting fields in `input_schema.json`: identifiers, outcomes and all 22 constituent statistics. The recorded export has 465 columns and 12,934 rows. Use the supported CSV export; this package performs no scraping or downloads.
3. Preserve missing values and fractional rate units (0.80 means 80%). Do not fill absent UBR or tracking fields with zeros. The source contains `Name.1`, `Team.1`, and `Season.1`; the importer checks these against the original columns. If your export schema differs, document that difference before adapting the importer. Do not remove other repeated-looking columns indiscriminately.
4. Save as `data/raw/BattingStats_2018_2026.csv` relative to this repository's root. Keep it untracked.

Frozen source SHA-256: `2130d2bc07ad64f3429c60a85628a1da9938b9dc4777ddc98277855bd1289e82`.

The primary analysis retains 2019–2025: 10,083 source rows, 3,070 PA-qualified seasons and 1,706 test transitions. The projection-system extension retains 2018–2025: 11,462 source rows and 3,518 qualified seasons. **2026 is never used for fitting, history or evaluation.** The full export is retained locally only to identify the supplied snapshot. Primary diversity uses 1,774 common focal seasons from 626 hitters. The older two-export input and its results are superseded for this revision.

## 2. Historical ZiPS and Steamer projections

Source: [FanGraphs projections](https://www.fangraphs.com/projections). The supplied acquisition route is **Projections → Batters → Historical Preseason Projections (Members Exclusive) → Dashboard → Data Export**. Obtain access to the historical member export, then select each system and projected season. Interface labels and availability can change; this records the author's acquisition route rather than certifying current UI behavior.

- Required seasons: **2019, 2021, 2022, 2023, 2024, 2025**, for **both ZiPS and Steamer**: 12 files.
- The author also inventoried 2020 exports for both systems (14 in total). Their hashes are recorded as optional: outcome-2020 projections are excluded and are not required to run the analysis.
- Retain the exported `PlayerId`, `wRC+`, `WAR`, and `PA` columns, all hitters, and the provider's values. Save the original CSV without spreadsheet reformatting; do not replace it with rest-of-season, depth-chart or in-season projections.
- Save each file as `ProjectionDataset/fangraphs-leaderboard-projections_{YEAR}_{z|s}.csv`, where `z` is ZiPS and `s` is Steamer. For example, `fangraphs-leaderboard-projections_2021_z.csv` forecasts outcome season 2021 and joins to 2020 input statistics.
- Exact FanGraphs `PlayerId` (internally `fg:<PlayerId>`) and outcome year determine matches. No name matching or MLBAM-ID substitution is used. Missing IDs are omitted from the join view; duplicate IDs fail validation. Usable observations require finite projected wRC+, and for WAR/600 finite WAR and positive projected PA. WAR/600 is `600 × projected WAR / projected PA`.

Record the actual download date, URL/settings, system and target year for any new acquisition. The filenames identify target seasons; they do **not** prove that these exact values were available at the historical preseason origin. These are retrospective file-based evaluations, not certified real-time vintage backtests.

## 3. Verify inputs and reproduce

Install the pinned environment from the root README. The file inventory in `input_manifest.json` supplies exact hashes, row counts and optional-file flags. Then run:

```bash
python scripts/reproduce.py --stage inputs
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/reproduce.py
python -m pytest -q tests
```

The first command checks the batting file and the 12 required projections. The full run recomputes the primary analysis, clipped projection augmentation, unclipped sensitivity and direct paired scale comparisons. It never reads the author's old prediction files or another repository. Generated observations and outcomes stay in ignored `data/processed/`, `data/raw/generated/`, `results_sd10/`, and `results_zscore/`. Existing completed outputs are protected; use a fresh checkout for a clean rerun. No abstract, paper, figure or submission table is regenerated.

The loader keeps low-PA history, constructs unique `fg:PlayerId`/season keys, then applies the PA ≥100 cohort threshold. B2V uses shrinkage, within-season orientation/standardization, equal within-tool aggregation and clipped `50 + 10z` grades. The optional scale run uses unclipped standardized coordinates under the same fitting protocol.

| Analysis | Source seasons | Main test outcomes | 2020 handling | Sensitivity test outcomes |
| --- | --- | --- | --- | --- |
| Primary | 2019–2025 | 2021–2025, n=1,706 | 2020 outcomes supply initial training | Remove transitions involving 2020; 2023–2025, n=1,073 |
| Projection extension | 2018–2025 | 2021–2025, n=1,706 | Exclude 2020 outcomes; train initially on 347 outcome-2019 transitions | Also remove 2020 inputs; 2022–2025, n=1,427 |

Both retain observed 2020 records in Marcel history lookbacks. Test transitions agree across main analyses, but training/history differ. Compare projection systems under the extension's common protocol rather than interchanging its Marcel result with the primary Marcel result. Recalibration and tuning use training folds only. The clipped primary specification was adopted after the original unclipped results, for representational consistency; it was not prospectively preregistered.

## 4. Historical snapshot differences

The exact-reproduction entry point stops on a missing file or hash mismatch. Provider revisions, formatting and field changes can prevent recovery of the frozen snapshot. Hashes identify files; they cannot recover unavailable data. Do not edit newly obtained values to force a match. A new-vintage analysis requires a separately recorded input manifest, review of schema/cohort differences and new aggregate reference results; it is not exact reproduction of this snapshot.

No redistribution permission is conferred by this recipe or the project's software license. See [DATA_RIGHTS.md](../DATA_RIGHTS.md). The organizers' conditional source-link/recipe response does not guarantee access to the exact historical inputs.
