# Common-grid v6 compact summary

This view cites only sealed v6 analysis values. Balanced accuracy is the primary outcome; all results are opened-development evidence, not confirmation or a SOTA claim.

## Top 10 equal-dataset rankings

| Rank | Model | Equal-dataset macro BA |
|---|---|---|
| 1 | `cardinal_fbc_compactdyn_scale025_extended` | 73.990% |
| 2 | `cardinal_fbc_compactdyn_extended` | 73.757% |
| 3 | `cardinal_dynamics_sinc_extended` | 73.667% |
| 4 | `cardinal_dynamics_extended` | 73.631% |
| 5 | `cardinal_fbc_compactdyn_scale010_extended` | 73.538% |
| 6 | `cardinal_fbc_physical_extended` | 73.535% |
| 7 | `cardinal_fbc_compactdyn` | 73.464% |
| 8 | `cardinal_fbc_compactdyn_scale025` | 73.420% |
| 9 | `cardinal_fbms_extended` | 73.365% |
| 10 | `cardinal_dynamics_compact` | 73.318% |

## Per-dataset winners

| Dataset | Model | Balanced accuracy | Participants |
|---|---|---|---|
| local_exp4 | `cardinal_dynamics_sinc_extended` | 91.625% | 8 |
| bnci2014_001 | `cardinal_fbc_compactdyn_extended` | 75.733% | 9 |
| bnci2014_004 | `cardinal_dynamics_sinc_extended` | 78.962% | 9 |
| cho2017 | `cardinal_fbc_compactdyn_scale025_extended` | 71.246% | 52 |
| physionet_mi | `cardinal_dynamics` | 58.841% | 54 |

Complete tables: `all_models_balanced_accuracy.csv`/`.md` and `all_models_accuracy.csv`/`.md`.
