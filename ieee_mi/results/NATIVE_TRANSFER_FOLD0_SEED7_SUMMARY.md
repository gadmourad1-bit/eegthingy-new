# Native-montage transfer screen: fold 0, seed 7

Status: **development-only, exploratory, nonconfirmatory**. These opened target
cohorts may inform the next bounded candidate, but they cannot support final
paper inference. No sealed confirmation subject was accessed.

## Frozen inputs

- Source checkpoint: `native_pretrain_seed7_partition_scaled`
- Checkpoint file SHA-256:
  `445495e4b78e5f444d7321d4252c68c85600a94332d4c60e93ade1a08b144249`
- Cho2017 target cohort: S16--S52, outer fold 0, seed 7
- PhysionetMI target cohort: S1--S54, outer fold 0, seed 7
- Conditions: pretrained CardinalFBC, scratch CardinalFBC, scratch FBMSNet,
  and pretrained indexed FBCNet after fixed spherical-spline normalization
- Metric: subject-level binary balanced accuracy; dataset means weight subjects
  equally and the two-dataset macro weights datasets equally

## Artifact integrity

The independent `native_transfer_audit` validator reloaded every prediction
archive and checked exact schemas, file and array SHA-256 values, finite
normalized probabilities, test split identity, checkpoint identity, source
hashes, cache hashes, and cross-condition consistency.

| Dataset | Expected records | Validated | Missing |
|---|---:|---:|---:|
| Cho2017 | 148 | 148 | 0 |
| PhysionetMI | 216 | 216 | 0 |

Raw write-once artifacts are preserved in:

- `results/native_transfer_cho_s16_52_fold0_seed7/`
- `results/native_transfer_physionet_s1_54_fold0_seed7/`

The full GPU-side EEG suite passed after the recovery: 196 passed, 0 failed.

## Descriptive results

| Condition | Cho2017 BA | PhysionetMI BA | Equal-dataset macro |
|---|---:|---:|---:|
| Pretrained CardinalFBC | 66.441% | 59.110% | 62.776% |
| Pretrained indexed FBCNet + spline | 64.696% | 58.433% | 61.564% |
| Scratch CardinalFBC | 62.928% | 52.927% | 57.927% |
| Scratch FBMSNet | 67.083% | 55.308% | 61.195% |

Pretraining improved CardinalFBC by 3.514 points on Cho and 6.184 points on
Physionet, or 4.849 points in the equal-dataset macro. On Physionet, a paired
subject bootstrap of pretrained minus scratch CardinalFBC (200,000 resamples,
NumPy PCG64 seed 20260719) gave a one-sided 95% lower bound of +3.621 points.
The corresponding pretrained-minus-FBMSNet lower bound was +0.761 point.
These are development diagnostics, not confirmatory intervals.

## Locked gate decision

The reference envelope takes the better indexed-FBC/FBMSNet dataset mean:
67.083% on Cho and 58.433% on Physionet, for a 62.758% macro.

| Gate component | Observed | Threshold | Result |
|---|---:|---:|---|
| Pretrained vs scratch Cardinal macro | +4.849 pp | at least +1.000 pp | Pass |
| Pretrained vs scratch Cardinal by dataset | +3.514 / +6.184 pp | positive on both | Pass |
| Physionet one-sided bootstrap lower bound | +3.621 pp | greater than 0 | Pass |
| Worst envelope deficit | -0.642 pp | no worse than -1.000 pp | Pass |
| Improvement over per-dataset FBC/FBMS envelope | **+0.018 pp** | at least +0.500 pp | **Fail** |

Therefore this fixed CardinalFBC candidate is rejected as the paper lead. The
result establishes useful heterogeneous-pretraining transfer, but it does not
establish a decisive advantage for the novel coordinate field over the
controls.

## Fixed dual-view diagnostic

A non-tuned 50/50 average of the direct and indexed prediction probabilities
reached 68.097% on Cho and 58.251% on Physionet. Its equal-dataset macro was
63.174%, or +0.416 point over the reference envelope. Equal log-probability
pooling produced identical binary decisions. This is evidence of complementary
views and motivates a coupled dual-view successor, but the fixed fusion itself
also misses the +0.500-point gate and is not advanced.

## Required next step

The next candidate must be declared before execution, evaluated only on these
opened development cohorts, and pass the explicitly encoded numeric gate. Full
folds and all five frozen seeds are authorized only after that screen passes.
Final paper claims still require author-faithful reference recipes, classical
controls, montage-robustness tests, and one-shot untouched confirmation after a
complete freeze.
