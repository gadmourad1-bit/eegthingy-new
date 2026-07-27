# EEG model accuracy inventory

Generated from the result artifacts present in this repository on 2026-07-21.

Unless a section explicitly says otherwise, every number is mean subject-level
**balanced accuracy (BA)**. Values must only be compared inside the same table or
protocol block. The project contains different datasets, subject partitions,
session splits, preprocessing pipelines, augmentation policies, seed counts, and
confirmation states; putting every value into one ranked leaderboard would be
scientifically invalid.

This is an architecture/configuration-level inventory, not a dump of every
subject, fold, seed, or checkpoint. Pure unit tests with synthetic tensors are
not accuracy experiments. Duplicate copies of identical result artifacts are
collapsed. Representative engineering smoke tests and superseded headline
results are retained at the end but are not scientific evidence.

## 0. Current local all-condition tournament

This is the current apples-within-protocol local development benchmark. It uses
eight participants (S1, S3, S4, S5, S6, S7, S8, and S10), five seeds (7, 17,
27, 37, and 47), and one chronological outer split. Recordings 1--2 fit the
model, recording 3 selects training duration or the procedure-specific frozen
choice, a fresh reset refits recordings 1--3, and all 60 balanced recording-4
trials are prediction-only. Seeds are averaged within participant before the
eight participants are weighted equally. Recording 4 had already been opened
during project development, so these are post-selection development results,
not independent confirmation.

The numerical winner across 43 common-recipe neural configurations, three
current-cache geometric controls, and four full-procedure comparators is
**TCFormer at 92.542% BA**. ORBIT-v3 is second at 92.458%. Their paired
participant-level margin is only +0.083 percentage points (95% paired bootstrap
interval -1.458 to +1.625 points; wins/ties/losses 4/1/3; exact two-sided
sign-flip p=0.96875), so they are not clearly separated.

| Overall rank | Condition | Category | Mean BA |
|---:|---|---|---:|
| 1 | **TCFormer** | Common-recipe neural configuration | **92.542%** |
| 2 | ORBIT-v3 | Full-procedure comparator | 92.458% |
| 3 | CAMEO | Full-procedure comparator | 91.333% |
| 4 | CardinalFBC micro, extended | Common-recipe neural configuration | 91.333% |
| 5 | CardinalDynamics, extended | Common-recipe neural configuration | 91.292% |
| 6 | CardinalFBC correlation, extended | Common-recipe neural configuration | 91.250% |
| 7 | CardinalDynamics Sinc, extended | Common-recipe neural configuration | 91.208% |
| 8 | CardinalFBC compact dynamics scale 0.10, extended | Common-recipe neural configuration | 91.208% |
| 9 | CardinalFBC compact dynamics scale 0.25, extended | Common-recipe neural configuration | 91.042% |
| 10 | FBMSNet | Common-recipe neural configuration | 91.000% |

The full 50-condition tables, subject-bootstrap intervals, medians, minima,
parameter counts, protocol boundary, and exclusions are in
`ieee_mi/results/local_exp4_allmodels_20260721/LOCAL_ALL_50_FINAL.md`. The
associated JSON and 103-file evidence bundle include 2,000 validated raw
prediction records. The four full-procedure comparators use the same outer rows
but a separate label-free covariance view and pinned procedure-specific
configurations; they are not raw-architecture apples-to-apples comparisons.

## 1. Current authoritative cross-dataset CardinalFBMS development result

This is the most complete current cross-dataset experiment: Cho2017 S16--52 over five disjoint
outer folds, PhysioNet S1--54 over three disjoint outer folds, and target-stage
seeds 7, 17, 27, 37, and 47. All target seeds use one source checkpoint trained
with source seed 7. This passed the locked **development** gate; it is not sealed
confirmation or a SOTA result.

| Condition | Cho2017 BA | PhysioNet BA | Equal-dataset macro |
|---|---:|---:|---:|
| **Pretrained CardinalFBMS** | **71.029%** | **61.128%** | **66.079%** |
| Pretrained indexed FBMSNet + spherical spline | 69.547% | 60.187% | 64.867% |
| Scratch CardinalFBMS, canonical seeded | 64.086% | 53.923% | 59.004% |
| Scratch CardinalFBMS, native projected | 65.855% | 54.488% | 60.171% |
| Scratch native indexed FBMSNet | 68.553% | 54.825% | 61.689% |

Source: `ieee_mi/results/NATIVE_FBMS_FULL_GRID_SUMMARY.md`.

## 2. Native-transfer advancement screens

These are earlier fold-0, seed-7 development screens. They are useful for showing
the model-development sequence but are superseded by the full grid above.

### 2.1 CardinalFBMS advancement screen

| Condition | Cho2017 BA | PhysioNet BA | Macro |
|---|---:|---:|---:|
| Pretrained CardinalFBMS | 69.043% | 61.558% | 65.300% |
| Pretrained indexed FBMSNet + spline | 67.444% | 61.177% | 64.310% |
| Scratch CardinalFBMS, canonical seeded | 61.768% | 54.828% | 58.298% |
| Scratch CardinalFBMS, native projected | 61.081% | 54.795% | 57.938% |
| Scratch native indexed FBMSNet | 67.083% | 55.308% | 61.195% |

Source: `ieee_mi/results/NATIVE_FBMS_TRANSFER_FOLD0_SEED7_SUMMARY.md`.

### 2.2 Predecessor CardinalFBC screen

| Condition | Cho2017 BA | PhysioNet BA | Macro |
|---|---:|---:|---:|
| Pretrained CardinalFBC | 66.441% | 59.110% | 62.776% |
| Pretrained indexed FBCNet + spline | 64.696% | 58.433% | 61.564% |
| Scratch CardinalFBC | 62.928% | 52.927% | 57.927% |
| Scratch FBMSNet | 67.083% | 55.308% | 61.195% |
| Fixed 50/50 direct/indexed fusion | 68.097% | 58.251% | 63.174% |

This candidate failed its lead-model gate. Source:
`ieee_mi/results/NATIVE_TRANSFER_FOLD0_SEED7_SUMMARY.md`.

## 3. Cardinal architecture-search models

These are single-seed, fold-0 development screens on local Exp4, BNCI2014-001,
BNCI2014-004, and Cho2017 S1--15. The macro weights the four datasets equally.
The overlapping v5/v6 controls are listed once because their stored scores are
identical.

| Model/configuration | Phase | Local Exp4 | BNCI001 | BNCI004 | Cho S1--15 | Macro |
|---|---|---:|---:|---:|---:|---:|
| CardinalFBC correlation, 21 anchors | v5 | 88.750% | 75.656% | 75.645% | 72.694% | 78.186% |
| CardinalFBC correlation, 31 anchors | v5 | 91.667% | 75.039% | 79.003% | 72.833% | 79.635% |
| CardinalFBC ordered-physical, 21 anchors | v5 | 89.167% | 75.347% | 76.771% | 74.028% | 78.828% |
| CardinalFBC ordered-physical, 31 anchors | v5 | 91.667% | 75.617% | 77.153% | 72.528% | 79.241% |
| CardinalFBC compact dynamics, 21 anchors | v5 | 88.750% | 75.694% | 76.528% | 74.167% | 78.785% |
| CardinalFBC compact dynamics, 31 anchors | v5/v6 | 91.667% | 75.579% | 76.424% | 73.556% | 79.306% |
| CardinalFBC plain, 31 anchors | v5 | 91.250% | 74.614% | 76.806% | 73.139% | 78.952% |
| CardinalFBC micro, 31 anchors | v5/v6 | 92.083% | 74.961% | 77.882% | 71.306% | 79.058% |
| CardinalFBC compact dynamics, scale 0.10 | v6 | 89.375% | 74.537% | 75.868% | 73.167% | 78.237% |
| CardinalFBC compact dynamics, scale 0.25 | v6 | 88.958% | 75.116% | 77.535% | 74.528% | 79.034% |
| CardinalFBC compact dynamics, scale 0.25, 31 anchors | v6 | 91.667% | 75.540% | 77.257% | 75.000% | **79.866%** |
| CardinalFBMS, 21 anchors | v6 | 88.542% | 74.498% | 78.313% | 74.306% | 78.915% |
| CardinalFBMS, 31 anchors | v6 | 91.458% | 74.807% | 76.691% | 74.167% | 79.281% |
| FBCNet | v5/v6 control | 91.458% | 74.267% | 76.806% | 72.972% | 78.876% |
| FBMSNet | v5/v6 control | 91.667% | 74.537% | 77.247% | 74.333% | 79.446% |

The highest number, 79.866%, belonged to a non-selectable 31-anchor ablation and
missed the predeclared advancement margin. No model in this screen was selected.
Sources: `ieee_mi/results/dev_*_continuations_screen_v5.json`,
`ieee_mi/results/dev_*_cardinal_fbms_screen_v6.json`, and
`ieee_mi/EXPERIMENT_LEDGER.md`.

## 4. Uniform deep-architecture benchmark

### 4.1 Local eight-subject architecture-only protocol

Raw task BA, recordings 1--2 fit, recording 3 selects, recording 4 tests;
neural results average seeds 7/17/27. Both classical rows called their
calibration routine on the complete scored recording 4 and are therefore
transductive and advantaged. The separately measured non-transductive Riemann
value is 89.94%; no matched non-transductive EA-FBCSP value is available here.

| Architecture | No augmentation | Swap augmentation |
|---|---:|---:|
| Riemann tangent-space LR | 90.060% transductive / 89.94% non-transductive | -- |
| EA-FBCSP | 90.010% transductive | -- |
| GeoAdaptNet convex-head tangent anchor | 88.462% | -- |
| GeoAdaptNet | 85.635% | 87.576% |
| ShallowConvNet | 85.866% | 87.044% |
| EEG-Conformer | 85.731% | 86.396% |
| EEGNet | 70.513% | 83.962% |
| ATCNet | 55.069% | 75.248% |
| DeepConvNet | 50.146% | 54.104% |

### 4.2 Cho2017, all 52 subjects, within-subject five-fold protocol

| Architecture | No augmentation | Swap augmentation |
|---|---:|---:|
| ShallowConvNet | 62.912% | **63.162%** |
| Riemann tangent-space LR, non-transductive | 59.683% | -- |
| GeoAdaptNet convex-head tangent anchor | 59.588% | -- |
| EEGNet | 56.880% | 61.575% |
| EEG-Conformer | 56.939% | 58.907% |
| GeoAdaptNet-FB, learned filter bank | 57.667% | 58.752% |
| GeoAdaptNet | 57.688% | 58.442% |
| ATCNet | 50.327% | 51.325% |
| DeepConvNet | 50.893% | 51.159% |

Sources: `deepnet/results/full_local_{noaug,aug}.json` and
`deepnet/results/full_cho2017_{noaug,aug}.json`.

### 4.3 Cho2017 30-subject filter-bank diagnostic

| Architecture/configuration | BA |
|---|---:|
| ShallowConvNet + swap | **62.392%** |
| GeoAdaptNet-FB, four learned bands + swap | 58.981% |
| GeoAdaptNet, four fixed bands + swap | 58.181% |
| GeoAdaptNet-FBSP, learned temporal and spatial filters + swap | 58.122% |
| GeoAdaptNet-FB, nine learned bands + swap | 56.320% |
| GeoAdaptNet-FB, four learned bands without swap | 57.828% |
| Riemann fixed-band control | 58.875% |

Sources: `deepnet/results/dev/cho_*.json` and `deepnet/RESULTS.md`.

## 5. GeoAdaptNet deployment-protocol results

These include the causal adapter and confidence/intent behavior and therefore
must not be compared directly with the architecture-only table.

| Protocol | GeoAdaptNet | Riemann TS+LR | EA-FBCSP |
|---|---:|---:|---:|
| Chronological schema-v2, baseline | 84.860% | 84.235% | 87.530% |
| Chronological + swap/blend | **86.622%** | 84.235% | 87.530% |
| Strict LOSO, seed 7 | 83.538% | 81.610% | 85.672% |
| Strict LOSO + swap/blend | **85.019%** | 81.610% | 85.672% |

Additional GeoAdaptNet chronological development/ablation runs:

| Run/configuration | BA | Status |
|---|---:|---|
| Seed-7 baseline | 85.089% | Development |
| Augmentation | 86.535% | Development |
| Augmentation + weight decay variant | 86.535% | Development |
| Swap probability 0.5 | 86.577% | Development |
| Swap + blend | 86.622% | Adopted configuration |
| Swap + blend + adaptive OAS | 85.567% | Rejected |
| Blend, seed 7 | 86.652% | Development |
| Symmetry blend | 86.305% | Development |
| Symmetry blend, longer optimizer | 86.036% | Rejected |
| Symmetry-only longer optimizer | 85.242% | Rejected |
| Symmetry scale 0.75 | 84.896% | Rejected |
| Mixup 0.2 | 85.110% | Rejected |
| CSP warm start | 85.120% | Rejected |
| CSP + deep supervision | 85.119% | Rejected |
| CSP + high deep-supervision weight | 84.579% | Rejected |
| Deep supervision 0.5 | 84.848% | Rejected |
| Deep supervision 0.5 + gate | 84.608% | Rejected |
| Deep supervision 1.0 + gate | 85.120% | Rejected |
| Full output, frozen target updates | 84.853% | Diagnostic |
| Full output, causal target updates | 84.860% | Diagnostic |
| Anchor logits only | 84.690% | Diagnostic |

Sources: `deepnet/results/dev/*.json`, `deepnet/results/chronological_*.json`,
and `deepnet/RESULTS.md`.

## 6. CAMEO experiments

CAMEO used separate local chronological and Cho2017 nested-CV protocols. The
local and Cho values are not directly comparable.

| CAMEO run | Cohort/protocol | BA | Status |
|---|---|---:|---|
| CAMEO v1 | Local development | 88.212% | Development |
| CAMEO v1, no mirror | Local development | 88.056% | Development |
| **Frozen CAMEO v1** | Local recording-4 confirmation, 8 subjects, 3 seeds | **90.208%** | Confirmation |
| CAMEO v1 | Cho development S1--8 | 68.677% | Development |
| CAMEO v1, exact variant | Cho development S1--8 | 69.250% | Development |
| CAMEO v1, no mirror | Cho development S1--8 | **70.448%** | Development |
| CAMEO v1, no mirror | Cho development S9--26 | 63.931% | Development |
| **Frozen CAMEO v1** | Cho confirmation S27--52 | **63.920%** | Confirmation |

Source: `deepnet/results/cameo/*.json`.

## 7. HemiParity, ParityFuse, and ORBIT experiments

These results span three different development protocols and one BNCI2014-001
confirmation. Each row states its cohort.

| Architecture/configuration | Cohort | BA | Status |
|---|---|---:|---|
| HemiParityNet v1 | Local development | 86.580% | Development |
| HemiParityNet v1 | Cho S1--8 development | 60.552% | Development |
| HemiParityNet v1 | BNCI001 S1--4 development | 77.778% | Development |
| ParityFuse v1 | Local development | 84.825% | Development |
| ParityFuse v1 | Cho S1--8 development | 59.365% | Development |
| ParityFuse v1 | BNCI001 S1--4 development | 77.604% | Development |
| ORBIT v1 | Local development | 87.739% | Development |
| ORBIT v1 | Cho S1--8 development | 68.833% | Development |
| ORBIT v1 | BNCI001 S1--4 development | 78.819% | Development |
| ORBIT v2 batch | Cho S1--8 development | 69.292% | Development |
| ORBIT v3 batch + auxiliary loss | Local development | **89.042%** | Development |
| ORBIT v3 batch + auxiliary loss | Cho S1--8 development | 69.302% | Development |
| ORBIT v3 batch + auxiliary loss | Cho S9--17 development | 64.556% | Development |
| ORBIT v3 batch + auxiliary loss | Cho S18--26 development | 66.222% | Development |
| ORBIT v3 batch + auxiliary loss | BNCI001 S1--4 development | 77.951% | Development |
| ORBIT v4 transport-only | Cho S1--8 development | 69.438% | Development |
| ORBIT v5 raw-mean-only | Cho S1--8 development | 69.854% | Development |
| Frozen OrbitTransportNet | BNCI001 S1--4 development | 77.951% | Frozen development |
| **Frozen OrbitTransportNet** | BNCI001 S5--9 confirmation | **76.389%** | Confirmation |
| Tangent anchor | BNCI001 S5--9 confirmation | **81.111%** | Confirmation control |
| Riemann tangent LR | BNCI001 S5--9 confirmation | 80.694% | Confirmation control |
| ShallowConvNet + swap | BNCI001 S5--9 confirmation | 71.944% | Confirmation control |

OrbitTransportNet did not beat the strongest confirmation controls. Sources:
`deepnet/results/parity/*.json`.

## 8. HemiQ-FieldNet experiments

HemiQ is a separate three-channel BNCI2014-004 architecture. Development uses
S1--4; one-shot confirmation uses S5--9.

### 8.1 Final frozen comparison

| Model | Development S1--4 | Confirmation S5--9 |
|---|---:|---:|
| **HemiQ-FieldNet v2** | **70.759%** | **82.500%** |
| ShallowConvNet + swap | 64.598% | 82.375% |
| Riemann tangent LR | 67.422% | 75.125% |
| Frozen tangent anchor | 67.422% | 75.000% |
| Shrinkage FBCSP-LDA | 67.478% | 74.500% |

### 8.2 HemiQ configuration sweep

All rows below are development-only on BNCI004 S1--4.

| HemiQ configuration | BA |
|---|---:|
| Initial/default single-scale field | 67.891% |
| 01 mid context | 67.600% |
| 02 long context | 66.964% |
| 03 compact | 66.920% |
| 04 compact mid context | 68.147% |
| 05 conservative optimization | 67.790% |
| 06 brief teacher | 68.125% |
| 07 no teacher | 67.723% |
| 08 light teacher | 67.801% |
| 09 strong teacher | 68.527% |
| 10 long teacher | 68.449% |
| 11 long kernel | 67.969% |
| 12 dense spectrum | 67.288% |
| 13 wide | 67.734% |
| 14 short context | 68.839% |
| 15 compact mid brief | 67.891% |
| 16 short strong | 68.527% |
| 17 short very strong | 68.605% |
| 18 very short strong | 68.638% |
| 19 short wide strong | 67.958% |
| 20 short slow strong | 68.292% |
| v2 compact-mid | 68.917% |
| v2 default, seed 7 | 70.759% |
| v2 short | 70.647% |
| v2 strong | 70.513% |
| v2 short-strong, seed 7 | **70.982%** |
| v2 default, three additional seeds (3/11/19) | 69.650% |
| v2 short-strong, three additional seeds (3/11/19) | 69.550% |
| v2 default, seed 7 plus seeds 3/11/19 | approximately 69.93% |
| v2 short-strong, seed 7 plus seeds 3/11/19 | approximately 69.91% |

The seed-7 short-strong score was numerically highest in development, but the
multi-seed comparison slightly favored the simpler default, which was frozen.
Sources: `deepnet/results/hemi_q/*.json` and `deepnet/HEMIQ_FIELD_REPORT.md`.

## 9. Post-hoc fusion results

These combinations were selected after observing test scores and are therefore
optimistically biased. They are experiments, not publishable confirmation.

| Fusion | Cohort | BA |
|---|---|---:|
| Best tested fusion | Local eight-subject architecture protocol | 91.61% |
| Best tested fusion | Cho2017 external protocol | 64.77% |
| All-member ensemble | Cho2017 external protocol | 62.25% |
| Riemann + ShallowConvNet | Cho2017 external protocol | 63.64% |

Source: `deepnet_CHATGPT_HANDOFF.md`.

## 10. Superseded, smoke, and legacy numbers

### 10.1 Superseded schema-v1 result -- do not cite

| Model | BA |
|---|---:|
| GeoAdaptNet | 86.402% |
| Riemann TS+LR | 86.190% |
| EA-FBCSP | 83.937% |

The audit invalidated this table because its refit schedule and temperature
transfer contract were incorrect.

### 10.2 One-subject engineering smoke tests -- not cohort evidence

| Smoke protocol | Model | BA |
|---|---|---:|
| Chronological | GeoAdaptNet | 60.737% |
| Chronological | Riemann TS+LR | 69.071% |
| Chronological | EA-FBCSP | 68.910% |
| LOSO | GeoAdaptNet | 74.660% |
| LOSO | Riemann TS+LR | 74.624% |
| LOSO | EA-FBCSP | 81.058% |

### 10.3 Older RA-L study -- ordinary accuracy, not balanced accuracy

These legacy transductive results use a different study and should not be
compared with any table above.

| Decoder | Personalized accuracy | Pooled/generalized accuracy |
|---|---:|---:|
| EA + FB-CSP | 91.1% | 84.8% |
| Riemann TS+LR | 89.5% | 77.6% |

Source: `paper/main.tex` and `deepnet/PROJECT_AUDIT.md`.

The same RA-L study also reports task-level corner/turn accuracy, which is not
offline decoder BA:

| Condition | Corner accuracy | Turn accuracy |
|---|---:|---:|
| EA + FB-CSP personalized | 92.5% | 77.5% |
| Riemann personalized | 95.6% | 91.0% |
| EA + FB-CSP generalized | 93.1% | 85.8% |
| Riemann generalized | 86.9% | 73.7% |
| Pooled | 92.0% | 81.3% |

## 11. Implemented or forward-tested models with no scientific accuracy result

The current IEEE-MI code can construct or recipe-wrap additional models, but no
cohort-level accuracy artifact in the repository supports quoting a result for
them under the current protocol:

- author-faithful TCFormer;
- CTNet;
- EEGTCNet;
- EEGSym;
- AttentionBaseNet;
- author-faithful FBCNet and TCFormer reference-recipe runs.

Their presence in code or passing unit tests must not be represented as an
accuracy comparison. Likewise, the EEGNet, Shallow/DeepConvNet, EEG-Conformer,
and ATCNet scores above come from the older GeoAdaptNet benchmark, not the new
CardinalFBMS full-grid protocol.

## Bottom line

- CardinalFBMS is the strongest condition in its current locked **development**
  protocol: 66.079% macro BA versus 64.867% for the strongest matched control.
- HemiQ-FieldNet has the strongest genuinely sealed neural confirmation mean in
  the repository: 82.500% on BNCI2014-004 S5--9, statistically tied with
  ShallowConvNet at 82.375%.
- CAMEO has frozen confirmations of 90.208% on the small local protocol and
  63.920% on Cho2017 S27--52, but those protocols differ from CardinalFBMS.
- OrbitTransportNet reached 76.389% on its BNCI2014-001 confirmation and lost to
  tangent and Riemannian controls.
- No existing result supports an unqualified global-SOTA claim.
