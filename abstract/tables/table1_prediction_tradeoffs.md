| Model inputs | wRC+ MAE | wRC+ gain [95% CI] | WAR/600 MAE | WAR/600 gain [95% CI] |
| --- | --- | --- | --- | --- |
| Current outcome (calibrated) | 19.98 | +0.92 [+0.58, +1.26] | 1.642 | +0.031 [+0.005, +0.058] |
| Fixed representative statistics | 19.81 | +0.75 [+0.45, +1.06] | 1.617 | +0.007 [-0.016, +0.029] |
| Training-selected representatives | 19.71 | +0.66 [+0.39, +0.92] | 1.611 | +0.0004 [-0.017, +0.018] |
| B2V | 19.06 | reference | 1.611 | reference |
| All 22 inputs | 18.82 | -0.24 [-0.43, -0.05] | 1.571 | -0.040 [-0.061, -0.019] |
| Per-tool PCA | 19.00 | -0.06 [-0.15, +0.0005] | 1.607 | -0.003 [-0.009, +0.001] |
| Unrestricted PCA | 19.11 | +0.05 [-0.05, +0.15] | 1.614 | +0.003 [-0.004, +0.009] |
| History-based projection | 19.21 | +0.37 [+0.17, +0.57] | 1.589 | +0.017 [+0.002, +0.031] |
| Projection + B2V | 18.84 | reference | 1.572 | reference |
| Projection + fixed representatives | 19.07 | +0.23 [+0.09, +0.37] | 1.577 | +0.005 [-0.006, +0.015] |
| Projection + selected representatives | 18.60 | -0.24 [-0.43, -0.06] | 1.570 | -0.002 [-0.018, +0.014] |
| Projection + all 22 inputs | 18.55 | -0.29 [-0.51, -0.07] | 1.546 | -0.026 [-0.048, -0.006] |

Gain = row MAE − reference MAE; positive favors B2V. First seven rows use B2V; final five use projection+B2V. Five forecast years (2021–2025), 1,706 transitions; 95% paired player-cluster bootstrap CIs (2,000 resamples). Per-tool PCA: one component per tool; unrestricted PCA: five across all 22 inputs. Fixed representatives: ISO, Contact%, BB%, BsR, Def. Selected representatives: training-only cross-validation among the fixed set and 17 within-tool single-replacement sets, separately by target, forecast origin and standalone/history context. History-based projections are recalibrated Marcel-style projections. The history+B2V versus history+selected-representatives wRC+ difference is inconclusive after refitting without 2020-related transitions.
