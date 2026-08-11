# All 43 common-grid models: standard accuracy

Values are direct presentation transforms of the sealed `analysis/dataset_summary.csv` and `analysis/overall_summary.csv`; no metric was recomputed. The descriptive accuracy rank is obtained by sorting the sealed equal-dataset accuracy column, using frozen plan architecture order only for exact ties. Balanced accuracy remains the prespecified primary outcome.

| Accuracy rank | Primary BA rank | Model | Local Exp4 accuracy | BNCI 2014-001 accuracy | BNCI 2014-004 accuracy | Cho2017 accuracy | PhysioNet MI accuracy | Equal-dataset accuracy |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | `cardinal_fbc_compactdyn_scale025_extended` | 91.250% | 75.463% | 77.924% | 71.246% | 54.222% | 74.021% |
| 2 | 2 | `cardinal_fbc_compactdyn_extended` | 90.208% | 75.733% | 77.612% | 71.149% | 54.123% | 73.765% |
| 3 | 3 | `cardinal_dynamics_sinc_extended` | 91.625% | 71.019% | 78.962% | 69.235% | 57.531% | 73.674% |
| 4 | 4 | `cardinal_dynamics_extended` | 90.625% | 70.610% | 78.135% | 70.214% | 58.741% | 73.665% |
| 5 | 5 | `cardinal_fbc_compactdyn_scale010_extended` | 91.000% | 74.807% | 77.338% | 69.994% | 54.749% | 73.578% |
| 6 | 6 | `cardinal_fbc_physical_extended` | 90.792% | 74.784% | 77.617% | 70.107% | 54.543% | 73.569% |
| 7 | 7 | `cardinal_fbc_compactdyn` | 88.417% | 75.355% | 78.187% | 71.117% | 54.313% | 73.478% |
| 8 | 8 | `cardinal_fbc_compactdyn_scale025` | 89.625% | 75.486% | 77.102% | 70.992% | 54.091% | 73.459% |
| 9 | 9 | `cardinal_fbms_extended` | 90.875% | 74.390% | 77.608% | 68.946% | 55.309% | 73.426% |
| 10 | 11 | `fbmsnet` | 90.958% | 74.221% | 77.480% | 68.931% | 55.300% | 73.378% |
| 11 | 12 | `tcformer` | 91.208% | 72.384% | 78.000% | 69.466% | 55.745% | 73.361% |
| 12 | 13 | `cardinal_fbc_corr_extended` | 90.333% | 74.799% | 75.941% | 70.036% | 55.588% | 73.340% |
| 13 | 10 | `cardinal_dynamics_compact` | 88.917% | 69.838% | 78.812% | 70.279% | 58.815% | 73.332% |
| 14 | 14 | `cardinal_fbc_micro` | 89.083% | 75.131% | 77.615% | 69.962% | 54.815% | 73.321% |
| 15 | 15 | `cardinal_fbc_micro_extended` | 90.750% | 75.085% | 76.220% | 70.235% | 54.272% | 73.312% |
| 16 | 16 | `cardinal_dynamics` | 88.833% | 70.077% | 78.200% | 70.438% | 58.955% | 73.301% |
| 17 | 17 | `cardinal_fbc_physical` | 89.500% | 75.131% | 77.533% | 70.032% | 54.181% | 73.275% |
| 18 | 18 | `cardinal_fbc_extended` | 90.667% | 74.622% | 76.704% | 69.660% | 54.601% | 73.251% |
| 19 | 19 | `fbcnet` | 90.625% | 74.560% | 76.816% | 69.599% | 54.601% | 73.240% |
| 20 | 20 | `cardinal_fbc` | 89.833% | 74.992% | 77.184% | 69.537% | 54.634% | 73.236% |
| 21 | 21 | `cardinal_fbc_corr` | 88.875% | 74.892% | 76.442% | 70.157% | 55.465% | 73.166% |
| 22 | 22 | `cardinal_fbc_compactdyn_scale010` | 89.417% | 75.108% | 76.722% | 69.952% | 54.551% | 73.150% |
| 23 | 23 | `cardinal_fbms` | 89.208% | 74.414% | 77.327% | 68.941% | 55.292% | 73.037% |
| 24 | 24 | `cardinal_dynamics_sinc_residual` | 87.125% | 70.640% | 78.921% | 69.539% | 57.926% | 72.830% |
| 25 | 25 | `cardinal_dynamics_sinc` | 87.208% | 70.424% | 78.799% | 68.874% | 56.938% | 72.449% |
| 26 | 26 | `free_cardinal` | 88.542% | 70.031% | 76.044% | 67.783% | 58.420% | 72.164% |
| 27 | 27 | `cardinal_mix_drop` | 83.167% | 72.091% | 77.699% | 68.095% | 57.152% | 71.641% |
| 28 | 28 | `cardinal_mix` | 82.125% | 71.975% | 77.677% | 67.194% | 56.897% | 71.174% |
| 29 | 29 | `shallow` | 87.667% | 65.440% | 77.389% | 68.170% | 56.724% | 71.078% |
| 30 | 30 | `cardinal` | 83.917% | 70.100% | 74.619% | 67.848% | 58.461% | 70.989% |
| 31 | 31 | `eegconformer` | 88.958% | 61.806% | 78.200% | 65.923% | 57.218% | 70.421% |
| 32 | 32 | `eegconformer_compact` | 87.375% | 62.600% | 77.706% | 65.474% | 57.531% | 70.137% |
| 33 | 33 | `ctnet` | 86.917% | 66.003% | 78.653% | 65.106% | 52.988% | 69.933% |
| 34 | 34 | `atcnet` | 83.917% | 62.770% | 78.150% | 65.242% | 55.918% | 69.199% |
| 35 | 35 | `atcnet_aggressive_pool` | 86.708% | 61.921% | 77.963% | 63.474% | 54.774% | 68.968% |
| 36 | 36 | `ctnet_compact` | 82.917% | 58.696% | 77.774% | 61.719% | 52.889% | 66.799% |
| 37 | 37 | `eegsym_wide` | 86.208% | 51.435% | 78.296% | 60.642% | 51.745% | 65.665% |
| 38 | 38 | `free_scope` | 77.417% | 55.201% | 73.846% | 59.786% | 54.403% | 64.131% |
| 39 | 39 | `eegnet` | 83.625% | 47.531% | 72.787% | 63.313% | 52.420% | 63.935% |
| 40 | 40 | `eegtcnet` | 70.833% | 47.731% | 73.133% | 56.739% | 52.305% | 60.148% |
| 41 | 41 | `scope` | 70.792% | 43.588% | 74.312% | 56.950% | 52.337% | 59.596% |
| 42 | 42 | `deep4` | 66.625% | 31.265% | 70.035% | 55.480% | 51.531% | 54.987% |
| 43 | 43 | `eegsym` | 62.625% | 38.318% | 63.066% | 53.600% | 50.395% | 53.601% |
