# Task 8 protocol — fixed before new results, 2026-09-17

This is a follow-up analysis after inspecting Tasks 1–7, not a preregistration of the original study. All new specifications below are frozen before running Task 8. No test-based selection of representative statistics, variants, k, or noninferiority margin.

- Inputs: existing local FanGraphs exports; 2019–2025 only. Input and outcome PA >=100; returning-hitter conditional estimand. No 2026 outcomes.
- Same expanding origins as Task 6; wRC+ primary, WAR/600 secondary; counting WAR exploratory, excluding transitions touching 2020. Rate targets also rerun excluding those transitions. History lookbacks still retain 2020.
- Same Ridge grid, player-grouped inner CV, seed 42 and 2,000 player-cluster paired bootstrap. Positive improvement = reference MAE minus candidate MAE. CIs are nominal, conditional on fitted predictions, not corrected for multiple comparisons or season shocks.
- Primary additions: history+22 vs history+5D; representative5 vs B2V; history+representative5 vs history+B2V. Representative5 chosen by meaning, not test performance: ISO, Contact%, BB%, BsR, Def (same season-standardized/stabilized constituent columns).
- Fixed sensitivity variants: Defense=Fld only; Defense=Def only; Speed=BsR only; Speed excluding UBR; Defense=Fld and Speed=BsR together; replace Defense and Speed with Fld/PA*600 and BsR/PA*600. Last variant is PA normalization, NOT defensive-innings normalization. Add age+PA to both history+5D and history+22 as context sensitivity.
- Offense3 and Full5 baselines repeated to reproduce Task 7. No claims that five dimensions identify five independent abilities.
- PCA cross-check: one PC per group, fit inside every CV split and outer training window using a pipeline. Global historical PCA/JointVAE scores excluded.
- Search: coordinates computed over ALL input-season PA>=100 players, before future eligibility filtering; past candidates only, outcomes known by forecast origin. Primary policy excludes all focal IDs and takes distinct other-player IDs. k=10 primary; k=3,5 sensitivities. Controls: scalar WAR+wRC+, 22 constituents, representative5, Offense3. Two secondary policies allow repeated player-seasons, with/without focal ID, to isolate self-history and duplicate-neighbor effects. Save neighbor IDs, seasons, distances and tie-break by distance, ID, season.
- Leave-one-origin-out summaries use existing predictions (no refit). Short-history contrast is exploratory; compare effects directly between history-depth 1 and 3; never equate lookback depth to rookie status.
- Export minimal raw-derived numerical inputs and hashes locally for portable reproduction. No external publication or third-party data license claim. Verify a raw-free isolated copy.
- Human utility: prepare randomized study materials and response/scoring scripts; no recruited participants, fabricated responses, or claim of measured utility.
