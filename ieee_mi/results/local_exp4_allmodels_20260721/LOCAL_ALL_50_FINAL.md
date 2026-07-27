# Combined local Exp4 benchmark audit

> **Development-only, post-selection result.** Every R4 recording was opened during prior project development. This is not independent confirmation evidence.

**Numerical winner:** `tcformer` — 92.542% mean balanced accuracy.

Runner-up: `orbit_v3`. Paired subject margin: +0.083 points; 95% paired subject-bootstrap interval [-1.458, +1.625] points; wins/ties/losses 4/1/3; exact two-sided sign-flip p=0.968750 (descriptive).

The inferential unit is the participant (n=8): five seeds are averaged within each participant, then the eight participants are weighted equally.

## Common-recipe neural configurations (43)

These are configurations/conditions, not 43 independent architecture families.

| Overall rank | Condition | Mean BA | Subject SD | 95% subject-bootstrap CI | Median | Minimum | Params |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `tcformer` | 92.542% | 6.568% | [88.125%, 96.542%] | 93.167% | 81.667% | 74,516 |
| 4 | `cardinal_fbc_micro_extended` | 91.333% | 8.930% | [85.167%, 96.667%] | 94.167% | 75.333% | 14,642 |
| 5 | `cardinal_dynamics_extended` | 91.292% | 6.644% | [86.875%, 95.417%] | 91.833% | 81.000% | 34,402 |
| 6 | `cardinal_fbc_corr_extended` | 91.250% | 9.302% | [84.792%, 96.667%] | 94.167% | 74.333% | 14,906 |
| 7 | `cardinal_dynamics_sinc_extended` | 91.208% | 8.361% | [85.500%, 96.208%] | 93.500% | 77.000% | 13,331 |
| 8 | `cardinal_fbc_compactdyn_scale010_extended` | 91.208% | 8.670% | [85.208%, 96.292%] | 93.667% | 75.000% | 31,850 |
| 9 | `cardinal_fbc_compactdyn_scale025_extended` | 91.042% | 8.316% | [85.333%, 95.958%] | 92.333% | 75.667% | 31,850 |
| 10 | `fbmsnet` | 91.000% | 8.134% | [85.333%, 95.792%] | 93.500% | 75.667% | 9,713 |
| 11 | `cardinal_fbc_extended` | 90.875% | 8.574% | [84.958%, 95.958%] | 93.333% | 75.333% | 12,098 |
| 12 | `cardinal_fbms_extended` | 90.875% | 8.194% | [85.167%, 95.708%] | 93.500% | 75.667% | 14,321 |
| 13 | `fbcnet` | 90.792% | 8.504% | [84.917%, 95.875%] | 93.000% | 75.333% | 7,490 |
| 14 | `cardinal_fbc_physical_extended` | 90.750% | 9.184% | [84.333%, 96.083%] | 94.000% | 73.333% | 18,607 |
| 16 | `cardinal_fbc_compactdyn_extended` | 90.250% | 7.968% | [84.792%, 95.000%] | 91.500% | 76.667% | 31,850 |
| 18 | `cardinal_fbc_corr` | 89.750% | 9.639% | [83.042%, 95.417%] | 92.333% | 71.667% | 12,026 |
| 19 | `cardinal_fbc_compactdyn_scale010` | 89.667% | 9.625% | [83.000%, 95.375%] | 92.500% | 73.333% | 23,210 |
| 22 | `cardinal_fbc_compactdyn_scale025` | 89.417% | 9.618% | [82.833%, 95.208%] | 92.000% | 73.667% | 23,210 |
| 23 | `cardinal_fbc_micro` | 89.292% | 9.434% | [82.875%, 94.833%] | 92.167% | 72.667% | 11,122 |
| 24 | `cardinal_fbms` | 89.292% | 10.105% | [82.375%, 95.417%] | 92.333% | 73.000% | 11,441 |
| 25 | `cardinal_fbc` | 89.167% | 9.568% | [82.583%, 94.833%] | 91.833% | 72.333% | 9,218 |
| 26 | `cardinal_fbc_physical` | 89.125% | 9.320% | [82.750%, 94.750%] | 92.667% | 73.000% | 13,807 |
| 27 | `eegconformer` | 89.042% | 7.861% | [83.917%, 94.042%] | 90.167% | 78.000% | 419,426 |
| 28 | `eegconformer_compact` | 88.708% | 9.040% | [82.625%, 94.208%] | 90.833% | 73.667% | 379,986 |
| 29 | `cardinal_fbc_compactdyn` | 88.542% | 9.769% | [81.917%, 94.458%] | 91.167% | 73.333% | 23,210 |
| 30 | `cardinal_dynamics_sinc_residual` | 88.208% | 10.246% | [81.208%, 94.375%] | 91.167% | 72.667% | 9,739 |
| 31 | `cardinal_dynamics` | 87.917% | 8.901% | [82.000%, 93.458%] | 88.667% | 74.333% | 24,162 |
| 32 | `cardinal_dynamics_compact` | 87.917% | 8.993% | [81.875%, 93.417%] | 88.833% | 73.667% | 13,994 |
| 33 | `free_cardinal` | 87.875% | 9.248% | [81.583%, 93.500%] | 89.500% | 71.333% | 12,835 |
| 34 | `atcnet_aggressive_pool` | 87.833% | 12.227% | [79.458%, 95.167%] | 92.667% | 66.333% | 104,490 |
| 35 | `shallow` | 87.833% | 10.527% | [80.625%, 94.250%] | 89.833% | 72.333% | 26,722 |
| 36 | `cardinal_dynamics_sinc` | 87.333% | 10.745% | [80.167%, 94.000%] | 91.000% | 71.333% | 9,491 |
| 38 | `ctnet` | 87.042% | 12.787% | [78.458%, 94.833%] | 93.000% | 68.000% | 150,022 |
| 39 | `eegnet` | 86.625% | 15.633% | [75.250%, 94.875%] | 91.500% | 51.333% | 1,602 |
| 40 | `eegsym_wide` | 85.125% | 11.939% | [77.333%, 92.667%] | 86.333% | 69.000% | 159,776 |
| 41 | `atcnet` | 84.875% | 12.748% | [76.541%, 92.958%] | 87.667% | 69.667% | 66,936 |
| 42 | `ctnet_compact` | 83.125% | 15.100% | [73.125%, 92.542%] | 88.500% | 62.333% | 26,218 |
| 43 | `cardinal` | 83.000% | 12.764% | [74.542%, 90.917%] | 87.333% | 64.333% | 15,907 |
| 44 | `cardinal_mix_drop` | 81.958% | 16.117% | [71.167%, 91.833%] | 89.000% | 57.000% | 11,153 |
| 45 | `cardinal_mix` | 81.833% | 15.652% | [71.292%, 91.458%] | 88.167% | 60.000% | 11,153 |
| 46 | `free_scope` | 78.375% | 17.198% | [67.167%, 89.083%] | 82.167% | 54.667% | 41,125 |
| 47 | `eegtcnet` | 69.583% | 13.174% | [61.125%, 78.125%] | 69.333% | 51.667% | 3,918 |
| 48 | `deep4` | 69.458% | 20.890% | [56.500%, 83.375%] | 62.333% | 49.667% | 141,927 |
| 49 | `scope` | 67.792% | 15.386% | [58.250%, 78.000%] | 67.000% | 51.000% | 39,589 |
| 50 | `eegsym` | 62.000% | 17.945% | [51.500%, 75.250%] | 52.500% | 48.667% | 19,252 |

## Current-cache geometric controls (3)

These controls are non-transductive. Their five formal seed identities are exact deterministic replicas and are not five independent fits.

| Overall rank | Condition | Mean BA | Subject SD | 95% subject-bootstrap CI | Median | Minimum | Params |
|---:|---|---:|---:|---:|---:|---:|---:|
| 17 | `tangent_anchor` | 89.792% | 8.928% | [83.333%, 94.583%] | 91.667% | 70.000% | — |
| 21 | `riemann` | 89.583% | 8.986% | [83.125%, 94.583%] | 91.667% | 70.000% | — |
| 37 | `ea_fbcsp` | 87.292% | 9.675% | [80.833%, 93.333%] | 90.000% | 73.333% | — |

## Full-procedure comparators (4)

These use the same outer rows but a separate label-free covariance view and pinned procedure-specific configurations; they are not raw-architecture apples-to-apples comparisons.

| Overall rank | Condition | Mean BA | Subject SD | 95% subject-bootstrap CI | Median | Minimum | Params |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | `orbit_v3` | 92.458% | 6.887% | [87.750%, 96.625%] | 93.333% | 79.000% | 18,211 |
| 3 | `cameo` | 91.333% | 7.928% | [85.917%, 96.083%] | 91.667% | 76.000% | 18,115 |
| 15 | `hemiparity` | 90.500% | 8.192% | [84.833%, 95.417%] | 92.333% | 75.000% | 43,609 |
| 20 | `parity_fuse` | 89.583% | 8.265% | [84.042%, 94.667%] | 91.000% | 75.000% | 19,468 |

## Protocol and audit boundary

- Local participant-specific binary left/right MI: R1–2 fit, R3 selection, fresh R1–3 refit, and all 60 balanced R4 trials prediction-only.
- Primary metric: subject-equal mean balanced accuracy after seed averaging.
- Raw prediction records validated: 2,000.
- Neural trial-level audit: all 1,720 raw neural prediction records validated.
- Old R4-calibrated Riemann/EA-FBCSP numbers and post-hoc fusion scores are outside this ranking.

## Claims this result does not support

- independent confirmation or population-level superiority.
- state of the art or top global performance.
- generalization to other datasets, subjects, montages, or sessions.
- asynchronous, closed-loop, or clinical efficacy.
- performance in disabled or paralyzed users.
- literature novelty based on accuracy alone.
