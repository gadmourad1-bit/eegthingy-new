# CardinalFBMS native-transfer screen: fold 0, seed 7

Status: **development-only, exploratory, nonconfirmatory**. The locked candidate
passed its advancement gate, which authorizes the prespecified all-fold,
five-seed development expansion. It is not yet a final-paper or SOTA result.
No sealed confirmation subject was accessed.

## Candidate and immutable source

CardinalFBMS keeps FBMSNet's filter bank, mixed temporal block, normalization,
activation, temporal log-variance views, and constrained classifier. It replaces
only the grouped channel-indexed spatial convolution with a regularized
21-anchor cardinal-RBF scalp-weight field. One learned spatial state can
therefore be evaluated directly on different native electrode geometries.

- Canonical trainable parameters: 11,441 (exact FBMSNet parity on 21 channels)
- Source checkpoint file SHA-256:
  `7378450eee169a56f0bfc1bf6ad78ad8014b5c63ae998267b4cc0ddee0c712cb`
- Source checkpoint tensor-state SHA-256:
  `e045e8c15c1b8319da0282e5cfa3342f1ffc105fd42a4197e99b8ebd5ea0fff5`
- Seeded initial-state SHA-256:
  `382d4e18385904b885e4140963705939b7d935ae95731ac402d3a447eabef507`
- Frozen source-corpus SHA-256:
  `e5ce98c41184ff42db6e9976d3369834ac83adb2c8dadacd560ba422ea2a360e`
- Frozen subject-partition SHA-256:
  `ca77219d9394bc8e790f933a5333171f27e89bbad8d86e3f2764a8222cbe7ac9`
- Selected source duration: 27 epochs (best epoch zero-based 26)
- Selection equal-dataset validation CE: 0.550800

The pretraining artifact is preserved under
`results/native_pretrain_cardinal_fbms_seed7/`.

## Locked screen

- Cho2017: S16--S52, outer fold 0, seed 7
- PhysionetMI: S1--S54, outer fold 0, seed 7
- Five conditions: pretrained CardinalFBMS; canonical-seeded scratch
  CardinalFBMS; native-geometry-projected scratch CardinalFBMS; native indexed
  scratch FBMSNet; and checkpoint-derived indexed FBMSNet after fixed Perrin
  spherical-spline transport to canonical21
- Metric: subject-level binary balanced accuracy; subjects and datasets are
  weighted equally

Before the full grid, the real CUDA smoke audit exposed that spline provenance
had hashed normalized float64 working coordinates while the condition and cache
records hashed raw float32 boundary coordinates. The writer was corrected to
bind the exact raw input coordinate arrays, a regression test was added, and a
fresh five-condition smoke grid passed. The earlier smoke grid remains isolated
and was not mixed into this screen. The corrected transfer-writer SHA-256 is
`b89b7af272937899f523e71d84e5a69247ceffa43dd6cb2ec7b44c6d8c47ba13`.

The complete run produced and independently validated 455/455 write-once
records, with zero missing and zero failed jobs:

| Dataset | Expected | Validated | Missing |
|---|---:|---:|---:|
| Cho2017 | 185 | 185 | 0 |
| PhysionetMI | 270 | 270 | 0 |

The auditor checks prediction bytes and array digests, cache and split identity,
checkpoint/source identity, exact model class/configuration/parameter count,
condition initialization, native/spline geometry, train-only and refit scalers,
frozen optimization configuration, producer-valid early stopping and learning
rate histories, BatchNorm reset semantics, and cross-record/cross-dataset
consistency before any aggregate is computed.

Raw artifacts are preserved in:

- `results/native_fbms_transfer_cho_s16_52_fold0_seed7/`
- `results/native_fbms_transfer_physionet_s1_54_fold0_seed7/`

The atomically published gate artifact is
`results/native_fbms_transfer_fold0_seed7_gate.json`, SHA-256
`365531860ca4c558268fc987c247b29b62705cec618589c408edc407bd06a774`.
The copied local artifacts were reopened and produced exactly the same metrics,
checks, bootstrap, and decision as the GPU artifact.

## Development results

| Condition | Cho2017 BA | PhysionetMI BA | Equal-dataset macro |
|---|---:|---:|---:|
| **Pretrained CardinalFBMS** | **69.043%** | **61.558%** | **65.300%** |
| Pretrained indexed FBMSNet + spline | 67.444% | 61.177% | 64.310% |
| Scratch CardinalFBMS, canonical-seeded | 61.768% | 54.828% | 58.298% |
| Scratch CardinalFBMS, native projection | 61.081% | 54.795% | 57.938% |
| Scratch native indexed FBMSNet | 67.083% | 55.308% | 61.195% |

The pretrained continuous candidate improved over its canonical-seeded scratch
control by 7.275 points on Cho and 6.729 points on Physionet, or 7.002 points
in the equal-dataset macro. It improved over the strongest per-dataset new or
historical reference envelope by 1.599 points on Cho and 0.380 point on
Physionet, or 0.990 point in the equal-dataset macro.

## Locked advancement gate

| Gate component | Observed | Threshold | Result |
|---|---:|---:|---|
| Candidate vs canonical-seeded scratch macro | +7.002 pp | at least +1.000 pp | Pass |
| Candidate vs scratch by dataset | +7.275 / +6.729 pp | positive on both | Pass |
| Physionet paired-subject one-sided 95% bootstrap lower bound | +3.919 pp | greater than 0 | Pass |
| Candidate vs reference-envelope macro | +0.990 pp | at least +0.500 pp | Pass |
| Worst dataset reference-envelope delta | +0.380 pp | no worse than -1.000 pp | Pass |

Overall gate decision: **PASS**. The bootstrap used exactly 200,000 paired
subject resamples with NumPy PCG64 seed 20260719.

## What this does and does not establish

This screen supports advancing CardinalFBMS to the prespecified all-fold,
five-seed development evaluation. It is encouraging evidence that
heterogeneous native-montage pretraining benefits the continuous coordinate
field and that the direct native representation can outperform its indexed
spline conversion.

It does not establish IEEE Transactions readiness, SOTA, generalization to an
untouched cohort, or statistical independence of the opened development data.
Those claims still require the full repeated-fold stage, author-faithful and
classical controls, montage/channel robustness tests, a final method freeze,
and one-shot sealed confirmation.
