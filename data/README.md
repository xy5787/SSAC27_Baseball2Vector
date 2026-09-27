# FanGraphs input acquisition and reconstruction recipe

Status: local reconstruction instructions verified against the current workspace. No FanGraphs rows are licensed for redistribution by this document. Data users must obtain access under the provider's applicable terms. See [data rights notice](../DATA_RIGHTS.md).

## Acquire the inputs

Source: [FanGraphs MLB batting leaderboards](https://www.fangraphs.com/leaders/major-league). Use the supported member CSV export rather than an undocumented endpoint or scraper; see [FanGraphs export support](https://blogs.fangraphs.com/contact/).

1. Select MLB regular-season batting data, all teams and players, with **PA/IP minimum 0**, not qualified hitters and not PA>=100. Keep individual season rows, not a single cumulative career/multiyear row. Include players who batted only a little; Marcel lookbacks require those seasons.
2. Obtain the 2019–2020 and 2021–2025 seasons. The author's second historical export also included 2026, which the code excludes; do not use 2026 to fit or assess this study.
3. Use full-season player totals across teams. Avoid separate team-stint rows for traded players. The import asserts a unique `(PlayerId, Season)` key and does not silently choose a stint.
4. Include the fields listed in `input_schema.json`. The historical inputs were 465-column combined leaderboard exports. Extra fields are discarded; repeated `Name.1`, `Team.1`, `Season.1` identifier columns must agree if present. If separate leaderboard exports are needed to obtain all fields, first join them on PlayerId/Season with explicit uniqueness checks. Current UI/export availability, access tier and field definitions may change.
5. Preserve missing values. In particular, do not invent absent UBR or tracking measurements, drop every row with a missing field, or replace missing values with zero at the raw-export stage. Preserve the provider's numeric rate units. The historical rate columns are fractions (e.g. a contact rate of 0.8 represents 80%); a differently formatted export needs explicit conversion before it can be treated as comparable.
6. Save the first CSV as `data/raw/batting_stats_2019_2020.csv` and the second as `data/raw/fangraphs-leaderboards (1).csv`. The latter is a historical filename expected by the current loader. Use actual local files, not the author's machine-specific symlinks. Keep both untracked.

The exported snapshot's collection date was not recorded in the original manifest. Do not invent a download timestamp. Record the date, exact source URL/settings and field availability for any new acquisition.

## Reconstruct the portable input

In the existing workspace, the original transformation is:

```bash
python scripts/task8_submission_strengthening.py --stage export
```

This command reads user-supplied local CSVs; it does not fetch FanGraphs. **It overwrites the local input snapshot and its manifest.** Run it in a fresh reproduction checkout or after preserving the analysis snapshot, not over the only copy used for the submitted results.

The code concatenates the two exports, retains 2019–2025, constructs `player_id = "fg:" + PlayerId`, asserts unique player-season keys, selects the required 31 columns, and sorts by Season/player_id. Names are descriptive labels; identity is based on FanGraphs PlayerId. It produces:

```text
data/processed/submission_inputs_2019_2025.csv       # local only
data/processed/submission_inputs_manifest.json      # provenance; no player rows
```

Use `input_schema.json` for the source fields and frozen checks. The original portable input has **10,083 rows**. The 3,070-row analysis cohort is selected at PA>=100 only later; Marcel history uses rows with PA>0. The large raw row count includes zero-PA/missing-stat rows and is not itself the modeled population.

| Season | Frozen snapshot rows |
| --- | ---: |
| 2019 | 1,410 |
| 2020 | 1,289 |
| 2021 | 1,508 |
| 2022 | 1,495 |
| 2023 | 1,457 |
| 2024 | 1,454 |
| 2025 | 1,470 |

## Verify and interpret differences

Frozen portable snapshot SHA-256:

```text
ae5c148dd9a5b79b8154facc88967119e56b210cd5aab0adc38a6c238edbeb5f
```

Hashes identify the exact files used; they are not a way to recover undistributed data. Later provider corrections, field changes, export formatting or row-order differences can change hashes. Never change a new export just to force the old row counts. Compare keys, units, missingness and derived counts, and record any divergence transparently. Source access plus a recipe supports procedural reproduction but cannot guarantee bit-for-bit recovery of a historical snapshot.

The next analysis stage is `cohort_from_raw` in `scripts/task8_submission_strengthening.py`: PA filtering, shrinkage, season-standardization/direction, equal within-tool averaging, season restandardization, and 50+10z clipping to 20–80. Adjacent-season joins require both seasons to qualify. Keep `B2V_SCALE=sd10` for the current submission. Five forecast years yield 1,706 evaluation transitions; the descriptive common-focal matching sample has 1,777 focal seasons from 626 players, with different inclusion rules.

## Publication boundary

Publish this recipe, schema, hashes and source links with the analysis code. Keep raw exports and value-preserving row-level snapshots, cohorts, transitions and predictions with observed outcomes out of the release until redistribution authority is established. A processed filename or project code license is not a data license. Obtain current SSAC confirmation of the source-link/recipe route if relying on it instead of hosting inputs.

The current workspace can reconstruct the input from authorized local exports. Use `python scripts/reproduce.py` from the repository root for the analysis pipeline; it reads locally supplied data and regenerates ignored player-level outputs.
