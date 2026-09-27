# Task 9: unrestricted five-component PCA comparison

Specified 2026-09-17 before Task 9 results, following the user's request. This is a follow-up after inspecting Task 8, not a preregistration of the original study.

- Use the frozen Task 8 2019–2025 input snapshot, PA>=100 in both seasons, identical expanding-window splits, targets and exclusions. wRC+ is primary; WAR/600 secondary; counting WAR exploratory. Rate targets include a sensitivity refit excluding transitions touching 2020.
- Full PCA: all 22 season-stabilized, direction-aligned constituent columns jointly enter median imputation, training-fold standardization, PCA(n_components=5), component standardization, then Ridge. All learned transforms and alpha tuning are fitted separately in each inner training split and outer training window. Input-season cohort transformations retain the Task 8 convention.
- Fixed five-component count; no test-based choice of dimension or loading. Same 50-alpha grid, five player-grouped inner folds, seed 42. Standardize PCA inputs to match the constituent Ridge comparator's training-fold input scaling; standardize resulting components to match downstream Ridge treatment.
- Refit B2V, grouped PCA (one PC per existing group), 22-statistic Ridge, representative5 and scalar comparators. Verify all shared model predictions and sample keys against saved Task 8 results. Existing results remain unchanged.
- Compare full PCA to all five comparators, and B2V to full PCA. Positive improvement = baseline MAE minus candidate MAE. Report pooled and yearly results with the existing 2,000 player-cluster paired bootstrap, plus leave-one-origin-out point differences.
- Save PCA loadings and explained input variance per outer window. Explained input variance is not predictive signal retention. No equivalence/noninferiority or human-interpretability claim; CIs are nominal and conditional on fitted predictions, without adjustment for multiple comparisons or common season shocks.
- History-augmented PCA is outside this standalone representation comparison.
