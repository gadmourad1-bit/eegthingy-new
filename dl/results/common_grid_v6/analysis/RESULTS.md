# Exact 43-model common-grid development analysis

> Exploratory opened-development-cohort evidence only. These results are **not confirmation evidence and not a global/SOTA claim**.

- Plan SHA-256: `65b93b7e5d09cfc30fb6ce28156368e66aec1503f4efa37faa3189e25ebbc3b1`
- Pre-result analysis contract SHA-256: `6eb975790bf7213e614159172d673b5e18ebda554f674e01c5b159099273393e`
- Descriptive same-grid leader: `cardinal_fbc_compactdyn_scale025_extended`
- Prespecified context comparator: `tcformer`
- Bootstrap: 100,000 deterministic resamples (seed 20260729)

## Equal-dataset macro results

| Rank | Model | Accuracy | Balanced accuracy | Chance-normalized BA | Macro F1 | Kappa | AUROC | NLL | Brier | ECE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `cardinal_fbc_compactdyn_scale025_extended` | 74.02% | 73.99% | 51.25% | 72.59% | 51.25% | 82.09% | 0.6261 | 0.3692 | 14.29% |
| 2 | `cardinal_fbc_compactdyn_extended` | 73.77% | 73.76% | 50.75% | 72.24% | 50.75% | 81.88% | 0.6641 | 0.3804 | 14.79% |
| 3 | `cardinal_dynamics_sinc_extended` | 73.67% | 73.67% | 51.20% | 73.07% | 51.20% | 81.30% | 0.5149 | 0.3296 | 9.08% |
| 4 | `cardinal_dynamics_extended` | 73.66% | 73.63% | 51.18% | 72.90% | 51.19% | 81.72% | 0.5204 | 0.3322 | 9.40% |
| 5 | `cardinal_fbc_compactdyn_scale010_extended` | 73.58% | 73.54% | 50.44% | 72.09% | 50.44% | 81.71% | 0.6326 | 0.3733 | 14.32% |
| 6 | `cardinal_fbc_physical_extended` | 73.57% | 73.53% | 50.43% | 71.96% | 50.43% | 81.70% | 0.6423 | 0.3775 | 14.68% |
| 7 | `cardinal_fbc_compactdyn` | 73.48% | 73.46% | 50.21% | 71.93% | 50.22% | 81.77% | 0.6600 | 0.3802 | 14.72% |
| 8 | `cardinal_fbc_compactdyn_scale025` | 73.46% | 73.42% | 50.11% | 71.82% | 50.11% | 81.86% | 0.6372 | 0.3756 | 14.54% |
| 9 | `cardinal_fbms_extended` | 73.43% | 73.37% | 50.14% | 72.09% | 50.15% | 81.40% | 0.6218 | 0.3684 | 13.39% |
| 10 | `cardinal_dynamics_compact` | 73.33% | 73.32% | 50.66% | 72.60% | 50.66% | 81.54% | 0.5276 | 0.3372 | 9.30% |
| 11 | `fbmsnet` | 73.38% | 73.32% | 50.07% | 72.07% | 50.08% | 81.41% | 0.6233 | 0.3692 | 13.39% |
| 12 | `tcformer` | 73.36% | 73.32% | 50.31% | 70.59% | 50.32% | 80.72% | 0.5032 | 0.3212 | 6.98% |
| 13 | `cardinal_fbc_corr_extended` | 73.34% | 73.30% | 49.96% | 71.88% | 49.97% | 81.81% | 0.6258 | 0.3734 | 14.05% |
| 14 | `cardinal_fbc_micro` | 73.32% | 73.29% | 49.90% | 71.89% | 49.90% | 81.50% | 0.6418 | 0.3777 | 14.36% |
| 15 | `cardinal_fbc_micro_extended` | 73.31% | 73.28% | 49.88% | 71.65% | 49.88% | 81.70% | 0.6434 | 0.3795 | 14.61% |
| 16 | `cardinal_dynamics` | 73.30% | 73.28% | 50.55% | 72.50% | 50.55% | 81.47% | 0.5330 | 0.3406 | 9.84% |
| 17 | `cardinal_fbc_physical` | 73.28% | 73.25% | 49.81% | 71.84% | 49.81% | 81.48% | 0.6480 | 0.3800 | 14.59% |
| 18 | `cardinal_fbc_extended` | 73.25% | 73.22% | 49.81% | 71.68% | 49.82% | 81.50% | 0.6381 | 0.3769 | 14.30% |
| 19 | `fbcnet` | 73.24% | 73.21% | 49.80% | 71.74% | 49.80% | 81.48% | 0.6407 | 0.3777 | 14.29% |
| 20 | `cardinal_fbc` | 73.24% | 73.20% | 49.74% | 71.74% | 49.74% | 81.38% | 0.6365 | 0.3761 | 14.19% |
| 21 | `cardinal_fbc_corr` | 73.17% | 73.13% | 49.61% | 71.49% | 49.61% | 81.63% | 0.6358 | 0.3776 | 14.27% |
| 22 | `cardinal_fbc_compactdyn_scale010` | 73.15% | 73.11% | 49.55% | 71.56% | 49.55% | 81.54% | 0.6380 | 0.3776 | 14.55% |
| 23 | `cardinal_fbms` | 73.04% | 72.98% | 49.36% | 71.72% | 49.37% | 81.14% | 0.6292 | 0.3739 | 13.56% |
| 24 | `cardinal_dynamics_sinc_residual` | 72.83% | 72.81% | 49.54% | 72.28% | 49.54% | 80.93% | 0.5265 | 0.3372 | 9.13% |
| 25 | `cardinal_dynamics_sinc` | 72.45% | 72.42% | 48.78% | 71.96% | 48.78% | 80.31% | 0.5317 | 0.3409 | 9.31% |
| 26 | `free_cardinal` | 72.16% | 72.12% | 48.24% | 71.17% | 48.25% | 80.89% | 0.5791 | 0.3700 | 12.17% |
| 27 | `cardinal_mix_drop` | 71.64% | 71.61% | 46.94% | 70.20% | 46.94% | 79.97% | 0.5877 | 0.3762 | 12.69% |
| 28 | `cardinal_mix` | 71.17% | 71.15% | 46.03% | 69.82% | 46.04% | 79.25% | 0.6198 | 0.3904 | 13.35% |
| 29 | `shallow` | 71.08% | 71.05% | 46.72% | 70.35% | 46.72% | 79.54% | 0.6389 | 0.3942 | 13.25% |
| 30 | `cardinal` | 70.99% | 70.95% | 45.88% | 69.73% | 45.89% | 80.01% | 0.6024 | 0.3861 | 12.94% |
| 31 | `eegconformer` | 70.42% | 70.39% | 45.88% | 68.41% | 45.88% | 79.49% | 0.7107 | 0.3999 | 13.12% |
| 32 | `eegconformer_compact` | 70.14% | 70.09% | 45.17% | 67.96% | 45.17% | 79.24% | 0.7053 | 0.4026 | 13.37% |
| 33 | `ctnet` | 69.93% | 69.92% | 44.38% | 68.06% | 44.38% | 77.83% | 0.5761 | 0.3723 | 10.19% |
| 34 | `atcnet` | 69.20% | 69.17% | 43.30% | 67.86% | 43.30% | 77.29% | 0.5772 | 0.3698 | 8.34% |
| 35 | `atcnet_aggressive_pool` | 68.97% | 68.93% | 42.94% | 67.74% | 42.94% | 76.64% | 0.5757 | 0.3686 | 8.36% |
| 36 | `ctnet_compact` | 66.80% | 66.72% | 38.94% | 64.76% | 38.94% | 74.99% | 0.5971 | 0.3906 | 8.82% |
| 37 | `eegsym_wide` | 65.67% | 65.60% | 37.67% | 62.24% | 37.68% | 74.67% | 0.6251 | 0.3954 | 8.84% |
| 38 | `free_scope` | 64.13% | 64.11% | 34.19% | 61.99% | 34.19% | 73.26% | 0.7660 | 0.4793 | 17.39% |
| 39 | `eegnet` | 63.94% | 63.85% | 34.69% | 59.77% | 34.70% | 73.16% | 0.6098 | 0.3915 | 5.85% |
| 40 | `eegtcnet` | 60.15% | 60.13% | 27.22% | 53.30% | 27.22% | 68.43% | 0.6682 | 0.4401 | 7.07% |
| 41 | `scope` | 59.60% | 59.59% | 26.70% | 57.06% | 26.69% | 68.24% | 0.8720 | 0.5386 | 19.72% |
| 42 | `deep4` | 54.99% | 54.91% | 18.99% | 48.85% | 18.99% | 65.97% | 0.7747 | 0.5038 | 11.77% |
| 43 | `eegsym` | 53.60% | 53.56% | 15.35% | 45.72% | 15.35% | 62.01% | 0.7489 | 0.4927 | 6.41% |

Primary aggregation: concatenate disjoint folds within each subject/seed, average seeds within subject, average subjects within dataset, then give each dataset equal weight.

## Descriptive leader and TCFormer by dataset

| Dataset | Descriptive leader | TCFormer |
|---|---:|---:|
| `local_exp4` | 91.25% | 91.21% |
| `bnci2014_001` | 75.46% | 72.38% |
| `bnci2014_004` | 77.92% | 78.00% |
| `cho2017` | 71.25% | 69.47% |
| `physionet_mi` | 54.07% | 55.52% |

## TCFormer reference context

| Model | Effect (BA points) | Descriptive fixed-suite 95% interval |
|---|---:|---:|
| `scope` | -13.73 | [-16.02, -11.52] |
| `free_scope` | -9.21 | [-11.36, -7.26] |
| `cardinal` | -2.37 | [-3.77, -1.04] |
| `free_cardinal` | -1.19 | [-2.33, +0.02] |
| `cardinal_dynamics` | -0.04 | [-1.21, +1.12] |
| `cardinal_dynamics_compact` | +0.00 | [-1.06, +1.05] |
| `cardinal_dynamics_extended` | +0.32 | [-0.86, +1.58] |
| `cardinal_dynamics_sinc` | -0.90 | [-1.99, +0.13] |
| `cardinal_dynamics_sinc_residual` | -0.50 | [-1.57, +0.54] |
| `cardinal_dynamics_sinc_extended` | +0.35 | [-0.78, +1.52] |
| `eegnet` | -9.47 | [-12.34, -6.78] |
| `shallow` | -2.26 | [-3.26, -1.32] |
| `deep4` | -18.40 | [-21.81, -15.01] |
| `eegconformer` | -2.92 | [-4.14, -1.83] |
| `eegconformer_compact` | -3.23 | [-4.34, -2.10] |
| `atcnet` | -4.15 | [-6.11, -2.42] |
| `atcnet_aggressive_pool` | -4.38 | [-5.90, -3.11] |
| `fbcnet` | -0.11 | [-1.22, +1.09] |
| `cardinal_fbc` | -0.12 | [-1.20, +1.03] |
| `cardinal_fbc_extended` | -0.10 | [-1.22, +1.11] |
| `cardinal_fbc_corr` | -0.19 | [-1.28, +0.96] |
| `cardinal_fbc_corr_extended` | -0.01 | [-1.05, +1.08] |
| `cardinal_fbc_compactdyn` | +0.15 | [-0.98, +1.27] |
| `cardinal_fbc_compactdyn_extended` | +0.44 | [-0.62, +1.56] |
| `cardinal_fbc_compactdyn_scale010` | -0.20 | [-1.26, +0.94] |
| `cardinal_fbc_compactdyn_scale010_extended` | +0.22 | [-0.86, +1.39] |
| `cardinal_fbc_compactdyn_scale025` | +0.10 | [-0.94, +1.21] |
| `cardinal_fbc_compactdyn_scale025_extended` | +0.67 | [-0.43, +1.90] |
| `cardinal_fbc_micro` | -0.02 | [-1.12, +1.13] |
| `cardinal_fbc_micro_extended` | -0.04 | [-1.11, +1.09] |
| `cardinal_fbc_physical` | -0.07 | [-1.18, +1.13] |
| `cardinal_fbc_physical_extended` | +0.22 | [-0.83, +1.38] |
| `eegtcnet` | -13.19 | [-15.09, -11.31] |
| `fbmsnet` | +0.00 | [-1.12, +1.20] |
| `cardinal_fbms` | -0.34 | [-1.44, +0.81] |
| `cardinal_fbms_extended` | +0.05 | [-1.03, +1.23] |
| `cardinal_mix` | -2.17 | [-3.89, -0.70] |
| `cardinal_mix_drop` | -1.71 | [-3.55, -0.12] |
| `ctnet` | -3.39 | [-5.14, -1.96] |
| `ctnet_compact` | -6.60 | [-8.36, -5.00] |
| `eegsym` | -19.75 | [-23.08, -16.41] |
| `eegsym_wide` | -7.72 | [-9.31, -6.17] |

No p-values, null-hypothesis decisions, or model-selection claims are computed. Every interval is descriptive context for these opened development cohorts.

## Calibration, complexity, and trial-micro outputs

`trial_micro_summary.csv` is intentionally separate: it pools repeated seed predictions and weights cohorts by trial count. Cross-dataset balanced accuracy, macro F1, kappa, and AUROC are left undefined because class semantics differ. `calibration_summary.csv` freezes 15-bin ECE, NLL, and multiclass Brier outputs. `complexity_summary.csv` and `resource_timing_summary.csv` report parameters, selected epochs, fit/inference time, inference time per trial, and peak CUDA allocation.

## Limitations

- Only previously opened development cohorts are analyzed.
- Architecture ranking occurred in the development evidence stream; independent sealed confirmation remains required.
- The hierarchical bootstrap samples datasets and then paired subject clusters. It does not treat trials, folds, or random seeds as independent.
- Runtime comparisons inherit shared-workstation load variation recorded during each job.
