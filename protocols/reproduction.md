# Reproduction scope — 2026-09-29

The primary analysis repeats the submitted forecast comparisons from Tasks 8–10 with the refreshed 2018–2026 source restricted to 2019–2025. The 2020 outcomes provide the initial training window; tests are 2021–2025 (1,706 transitions). Excluding transitions involving 2020 leaves 2023–2025 tests (1,073). Its history pool begins in 2019.

The projection-system extension uses 2018–2025 history, excludes 2020 outcomes, and starts with 347 outcome-2019 training rows. It has the same 1,706 main test transitions. Removing 2020 inputs leaves 1,427 tests in 2022–2025. Known 2020 history remains in Marcel lookbacks. No analysis uses 2026.

`projection_original.md` preserves the historical specification before the original unclipped ZiPS/Steamer results. Its approval gates and workspace paths record that execution history; they are not dependencies of the portable runner. Both systems subsequently ran. `projection_clipped.md` records the explicit post-result amendment: clipped SD10 is now the primary representation, with original unclipped results retained as sensitivity. This is not a new preregistration.

The portable runner retains the numerical fitting, cohort, split, ridge tuning (50 alphas, training-player GroupKFold), projection matching and 2,000 paired player-bootstrap routines. It replaces machine-specific approval files and comparisons to private archived predictions with input hash checks, local independent fits and aggregate reference comparisons. No manuscript output is needed. Historical projection publication timestamps remain unverified.

Reference results contain aggregate task8/task9 performance and contrasts, task10 performance/contrasts/selection, diversity summaries and ratio intervals, both projection systems at both scales, and paired clipping comparisons. All prediction and matching rows remain local. Confidence intervals are nominal, conditional on fitted predictions; they omit multiplicity correction and shared season shocks.

The gap-closure ratio in diversity was added after feedback using the same unrounded medians and joint resamples. It is descriptive, not explained variance or a causal matching effect. Model selection and evaluation definitions are unchanged.

Historical Task 8–10 protocols also describe retrieval experiments, human-study preparation, exploratory counting WAR and PCA-loading exports. Those are not run or claimed as reproduced by this release. The current runner generates descriptive matching, rate-outcome forecasts, selected-representative comparisons, both projection scales and their paired comparisons. The default full command then verifies aggregate references and executes the tests; new-snapshot mode explicitly reports reference differences.
