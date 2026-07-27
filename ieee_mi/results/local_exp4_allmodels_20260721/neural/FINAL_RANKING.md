# Local Exp4 all-architecture tournament

> Development-only result. Recording 4 has previously been inspected; this is not independent confirmation evidence.

**Numerical winner:** `tcformer` — 92.542% mean balanced accuracy.

Runner-up: `cardinal_fbc_micro_extended`. Mean paired margin: +1.208 percentage points; 95% paired subject-bootstrap interval [-0.625, +3.208] points.

Winner-versus-runner-up statistics are post-selection and descriptive.

| Rank | Model | Mean BA | Subject SD | 95% bootstrap CI | Parameters |
|---:|---|---:|---:|---:|---:|
| 1 | `tcformer` | 92.542% | 6.568% | [88.125%, 96.583%] | 74,516 |
| 2 | `cardinal_fbc_micro_extended` | 91.333% | 8.930% | [85.208%, 96.625%] | 14,642 |
| 3 | `cardinal_dynamics_extended` | 91.292% | 6.644% | [86.833%, 95.417%] | 34,402 |
| 4 | `cardinal_fbc_corr_extended` | 91.250% | 9.302% | [84.792%, 96.625%] | 14,906 |
| 5 | `cardinal_dynamics_sinc_extended` | 91.208% | 8.361% | [85.458%, 96.250%] | 13,331 |
| 6 | `cardinal_fbc_compactdyn_scale010_extended` | 91.208% | 8.670% | [85.167%, 96.292%] | 31,850 |
| 7 | `cardinal_fbc_compactdyn_scale025_extended` | 91.042% | 8.316% | [85.333%, 96.042%] | 31,850 |
| 8 | `fbmsnet` | 91.000% | 8.134% | [85.333%, 95.792%] | 9,713 |
| 9 | `cardinal_fbc_extended` | 90.875% | 8.574% | [84.958%, 95.958%] | 12,098 |
| 10 | `cardinal_fbms_extended` | 90.875% | 8.194% | [85.167%, 95.750%] | 14,321 |
| 11 | `fbcnet` | 90.792% | 8.504% | [84.917%, 95.833%] | 7,490 |
| 12 | `cardinal_fbc_physical_extended` | 90.750% | 9.184% | [84.333%, 96.083%] | 18,607 |
| 13 | `cardinal_fbc_compactdyn_extended` | 90.250% | 7.968% | [84.833%, 95.042%] | 31,850 |
| 14 | `cardinal_fbc_corr` | 89.750% | 9.639% | [83.042%, 95.375%] | 12,026 |
| 15 | `cardinal_fbc_compactdyn_scale010` | 89.667% | 9.625% | [82.999%, 95.375%] | 23,210 |
| 16 | `cardinal_fbc_compactdyn_scale025` | 89.417% | 9.618% | [82.875%, 95.125%] | 23,210 |
| 17 | `cardinal_fbc_micro` | 89.292% | 9.434% | [82.792%, 94.917%] | 11,122 |
| 18 | `cardinal_fbms` | 89.292% | 10.105% | [82.417%, 95.375%] | 11,441 |
| 19 | `cardinal_fbc` | 89.167% | 9.568% | [82.583%, 94.833%] | 9,218 |
| 20 | `cardinal_fbc_physical` | 89.125% | 9.320% | [82.708%, 94.750%] | 13,807 |
| 21 | `eegconformer` | 89.042% | 7.861% | [83.875%, 94.000%] | 419,426 |
| 22 | `eegconformer_compact` | 88.708% | 9.040% | [82.625%, 94.250%] | 379,986 |
| 23 | `cardinal_fbc_compactdyn` | 88.542% | 9.769% | [81.917%, 94.458%] | 23,210 |
| 24 | `cardinal_dynamics_sinc_residual` | 88.208% | 10.246% | [81.208%, 94.375%] | 9,739 |
| 25 | `cardinal_dynamics` | 87.917% | 8.901% | [82.042%, 93.417%] | 24,162 |
| 26 | `cardinal_dynamics_compact` | 87.917% | 8.993% | [81.875%, 93.417%] | 13,994 |
| 27 | `free_cardinal` | 87.875% | 9.248% | [81.583%, 93.542%] | 12,835 |
| 28 | `atcnet_aggressive_pool` | 87.833% | 12.227% | [79.458%, 95.167%] | 104,490 |
| 29 | `shallow` | 87.833% | 10.527% | [80.625%, 94.250%] | 26,722 |
| 30 | `cardinal_dynamics_sinc` | 87.333% | 10.745% | [80.167%, 94.000%] | 9,491 |
| 31 | `ctnet` | 87.042% | 12.787% | [78.500%, 94.833%] | 150,022 |
| 32 | `eegnet` | 86.625% | 15.633% | [75.333%, 94.875%] | 1,602 |
| 33 | `eegsym_wide` | 85.125% | 11.939% | [77.292%, 92.708%] | 159,776 |
| 34 | `atcnet` | 84.875% | 12.748% | [76.541%, 92.958%] | 66,936 |
| 35 | `ctnet_compact` | 83.125% | 15.100% | [73.125%, 92.583%] | 26,218 |
| 36 | `cardinal` | 83.000% | 12.764% | [74.542%, 90.917%] | 15,907 |
| 37 | `cardinal_mix_drop` | 81.958% | 16.117% | [71.083%, 91.792%] | 11,153 |
| 38 | `cardinal_mix` | 81.833% | 15.652% | [71.333%, 91.458%] | 11,153 |
| 39 | `free_scope` | 78.375% | 17.198% | [67.000%, 89.125%] | 41,125 |
| 40 | `eegtcnet` | 69.583% | 13.174% | [61.042%, 78.083%] | 3,918 |
| 41 | `deep4` | 69.458% | 20.890% | [56.500%, 83.375%] | 141,927 |
| 42 | `scope` | 67.792% | 15.386% | [58.250%, 77.917%] | 39,589 |
| 43 | `eegsym` | 62.000% | 17.945% | [51.500%, 75.250%] | 19,252 |
