# Clipped production extension: authorized scale amendment

On September 29, 2026 the user requested a unified clipped 20–80 B2V primary
analysis for Marcel, ZiPS and Steamer. Original unclipped results were already
observed; this is an explicitly post-result scale amendment, not a new prospective
preregistration. The same five coordinates are transformed to clip(50 + 10z,20,80).
All other input hashes, cohorts, expanding splits, inner GroupKFold tuning,
50-value alpha grid, targets, matching, reference models, 2020 handling and
2,000 player bootstrap draws (seed 42) follow the original production protocol.

Primary window: 2018–2025 source, 2019 initial training outcomes, no 2020 outcomes,
2021–2025 evaluation (1,706 transitions). Sensitivity also drops 2020 inputs,
evaluating 2022–2025 (1,427). Observed 2020 Marcel history is retained. 2026 is
excluded everywhere. No new features, model selection criteria or subgroups.

The same raw projection, calibrated baseline, projection + B2V, and projection
+ 22 inputs are evaluated on identical keys. Marcel references are fitted in both
systems and must match exactly. Unchanged baseline/22-input predictions must
reproduce the archived unclipped-run values. Clipped B2V results must be reported
regardless of sign or significance. Original unclipped results remain a scale
sensitivity. Direct clipped-minus-unclipped MAE CIs are paired on the same keys;
positive differences favor the unclipped model. These nominal, fixed-prediction
CIs omit multiplicity and shared season shocks. Historical vintages remain
retrospective and are not certified as preseason snapshots.
