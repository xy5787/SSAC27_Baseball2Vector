# Analysis protocol

## Profiles and cohorts

B2V uses 22 FanGraphs statistics grouped into Power, Contact, Plate Discipline, Defense, and Speed. Constituents receive PA-based shrinkage, within-season standardization, and direction alignment. Equal-weight group means are standardized again and mapped to `clip(50 + 10z, 20, 80)`. The implementation and shrinkage constants are in `src/baseball2vec/tools.py` and `scripts/common.py`.

Profiles use hitters with at least 100 PA. Forecast transitions require at least 100 PA in both adjacent seasons. Lower-PA observations remain available for Marcel history. Targets are next-season wRC+ and WAR/600 (`600 × WAR / PA`). No analysis uses 2026.

| Analysis | Source years | Main test outcomes | Training and sensitivity |
| --- | --- | --- | --- |
| Primary | 2019–2025 | 2021–2025; 1,706 transitions | Initial training uses 2020 outcomes. Excluding transitions involving 2020 leaves 1,073 tests in 2023–2025. |
| Projection extension | 2018–2025 | 2021–2025; 1,706 transitions | Initial training uses 347 outcome-2019 transitions; all 2020 outcomes are excluded. Also excluding 2020 inputs leaves 1,427 tests in 2022–2025. |

Both analyses retain observed 2020 records in Marcel lookbacks. Their main test transitions coincide, but training samples and history differ. Marcel results should therefore be compared within the corresponding analysis.

## Forecasts

Outer folds expand through time, with training outcomes strictly earlier than test outcomes. Ridge models use training-fold median imputation and standardization. The penalty is selected from 50 log-spaced values between 0.001 and 10,000 by five-fold player-grouped inner CV, minimizing mean MAE. Ties favor the smaller penalty. PCA is fitted within each training split.

Primary comparisons include current performance, B2V, all 22 constituents, grouped and unrestricted five-component PCA, fixed and training-selected five-statistic summaries, and Marcel-style history projections. Defense and Speed ablations and history-augmented models use the same splits and tuning.

The fixed representatives are ISO, Contact%, BB%, BsR, and Def. The selected comparator chooses between this set and 17 single-statistic replacements within the same tool groups, using inner-CV MAE on outer-training data only. Ties follow variant order, then penalty order. Five leave-one-out variants are also evaluated. This follow-up was specified after viewing earlier comparisons; it was not an original-study preregistration.

The projection extension matches FanGraphs PlayerId and outcome year to historical ZiPS or Steamer files. Each system compares the original projection, ridge recalibration, and augmentation with B2V or all 22 inputs. Marcel reference models share the same eligible rows. Projection WAR/600 uses projected WAR and PA. The two systems use identical training and evaluation cohorts.

## Profile comparisons

The descriptive cohort contains 3,070 player-seasons. Each focal hitter is matched to another same-season hitter within 0.2 WAR and five wRC+ points, minimizing the season-standardized scalar gap with deterministic ID tie-breaking. Comparators may be reused. Restricting to focals with an adjacent-season observation leaves 1,774 focal seasons from 626 hitters.

The same focals supply matched, same-hitter next-season, and random same-season pairs. Random comparators are different hitters in the focal PA stratum: 100–249, 250–499, or 500+. Profile distance is the root mean square of five score differences divided by each tool's pooled sample standard deviation.

Gap closure is `(random median − matched median) / (random median − same-hitter median)`. It was added after feedback and is descriptive, not explained variance or a causal effect.

## Uncertainty and sensitivity

All runs use seed 42 and 2,000 paired player-cluster bootstrap draws. Forecast intervals condition on fitted predictions; descriptive intervals use joint focal-player resamples with fixed matches and normalization. Intervals are nominal 95% percentile intervals without multiplicity or shared-season-shock correction. Reused descriptive comparators are not separately clustered.

Clipped 20–80 scores became the primary specification after the original unclipped results were observed. Unclipped forecasts and paired scale comparisons remain sensitivity analyses; this change was not prospectively preregistered. The 2020 sensitivities refit preprocessing, selection, and models after excluding the specified transitions.

Historical projection publication times remain unverified. These analyses use retrospective files and do not certify real-time preseason performance.
