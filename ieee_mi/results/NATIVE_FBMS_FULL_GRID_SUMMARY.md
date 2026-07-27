# CardinalFBMS all-fold, five-seed development result

Status: **locked development gate PASS**. This is strong development evidence,
not confirmation evidence, a state-of-the-art claim, or proof of benefit in a
disabled or paralyzed clinical population. No sealed confirmation subject was
accessed.

## Frozen evaluation

- Cho2017: subjects 16--52, five disjoint outer folds
- PhysioNet MI: subjects 1--54, three disjoint outer folds
- Target-stage seeds: 7, 17, 27, 37, and 47
- Five frozen conditions per subject/fold/seed
- Total: 8,675 write-once records, all freshly validated
- Scientific unit: subject
- Fold handling: concatenate disjoint OOF predictions before scoring
- Seed handling: average five target-stage scores within subject
- Dataset handling: weight subjects equally within dataset and datasets equally
- Inference: paired subject bootstrap, 200,000 fixed-seed replicates

All target-stage seeds used the same source-pretraining checkpoint from source
seed 7. The reported intervals therefore quantify development-subject
uncertainty after averaging the five target-stage seeds; they do not quantify
source-pretraining randomness.

## Balanced accuracy

| Condition | Cho2017 | PhysioNet MI | Equal-dataset macro |
|---|---:|---:|---:|
| **Pretrained CardinalFBMS** | **71.029%** | **61.128%** | **66.079%** |
| Pretrained indexed FBMSNet + fixed spherical spline | 69.547% | 60.187% | 64.867% |
| Scratch CardinalFBMS, canonical seeded | 64.086% | 53.923% | 59.004% |
| Scratch CardinalFBMS, native projected | 65.855% | 54.488% | 60.171% |
| Scratch native indexed FBMSNet | 68.553% | 54.825% | 61.689% |

CardinalFBMS improved over its canonical-seeded scratch control by 6.944
points on Cho2017 and 7.205 points on PhysioNet, or 7.074 points in the
equal-dataset macro. Its paired-subject one-sided 95% lower bounds were +5.242,
+5.309, and +5.794 points for Cho2017, PhysioNet, and the macro respectively.

The primary architectural comparison was the checkpoint-matched indexed
FBMSNet plus fixed spherical-spline control. CardinalFBMS improved over it by
1.482 points on Cho2017, 0.941 point on PhysioNet, and 1.211 points in the
equal-dataset macro. The prespecified macro one-sided 95% lower bound was
+0.213 point. The macro two-sided percentile interval was +0.027 to +2.417
points. Per-dataset intervals were descriptive rather than gate criteria: the
one-sided lower bounds were +0.038 point on Cho2017 and -0.431 point on
PhysioNet.

The indexed+spline control was also the strongest reference-envelope source on
both datasets. The candidate exceeded that envelope by 1.211 points in the
macro, and its worst dataset margin was +0.941 point.

## Prespecified gate

| Component | Observed | Frozen threshold | Result |
|---|---:|---:|---|
| Candidate minus canonical scratch, macro | +7.074 pp | at least +1.000 pp | Pass |
| Candidate minus scratch, Cho / PhysioNet | +6.944 / +7.205 pp | positive on both | Pass |
| Scratch-bootstrap one-sided lower, Cho / PhysioNet / macro | +5.242 / +5.309 / +5.794 pp | positive for all | Pass |
| Candidate minus indexed+spline, macro one-sided lower | +0.213 pp | greater than 0 | Pass |
| Candidate minus reference envelope, macro | +1.211 pp | at least +0.500 pp | Pass |
| Worst dataset reference-envelope margin | +0.941 pp | at least -1.000 pp | Pass |

Overall locked development decision: **PASS**.

## Artifact integrity and independent verification

- Final post-completion score-blind preflight: 8,675 validated, zero pending,
  zero recoverable transients, decision `GO`; SHA-256
  `c3d4bff53a4b0132812fc42661b122430ac8ebf9c95743cef185fd4975ab7ffb`.
- Frozen gate artifact:
  `native_fbms_transfer_full_grid_gate.json`, 4,921,892 bytes, SHA-256
  `eab9f6ee1b548000eed823025330aa13c20616b8ed266cf5dd6b521a7ef80496`.
- Embedded freshly revalidated manifest SHA-256:
  `5e9a2bd93828603e2cafa671158074c7f707047112d867f8504ae86393f7fbbe`.
- Standalone post-outcome verifier SHA-256:
  `918ee23a785b449688f39929f6fdde7778f81652622ed25bec8b5c1422dfbe18`.
- Independent raw-OOF report SHA-256:
  `7c1302f17f22a4e895c7e42257f5f0a8ab1badf0d59b93ea6b4ee314489f2562`.

The standalone verifier imports neither the frozen gate nor its full-grid
aggregation helpers. It rehashed and reopened all 8,675 prediction and
provenance files, enforced exact root membership and OOF row/label identity,
recomputed every subject/seed score, reconciled nested and top-level means,
reproduced both bootstraps, and independently obtained all six passing checks.
The report contains no verification error.

A separate JSON-level audit performed 694 consistency checks and also
reproduced both bootstraps. Five quantiles differed on the independent local
NumPy environment by only 8.67e-19 to 3.47e-17 (1--22 float64 ULPs); no
threshold or decision changed. On the frozen GPU environment, the raw verifier
matched the serialized quantiles exactly.

## What this establishes

The result supports the development claim that a learned continuous
cardinal-RBF scalp field can transfer one pretrained FBMS-style spatial state
across heterogeneous native electrode geometries better than a fixed
indexed+spherical-spline projection of the same checkpoint, while retaining
exact parameter parity on the canonical 21-channel model.

It also shows a large and statistically stable benefit from the frozen source
pretraining protocol relative to training the same continuous architecture
from scratch on these development cohorts.

## What remains before an IEEE Transactions claim

1. Replicate source pretraining with multiple independently trained source
   seeds; the present five seeds vary target fitting only.
2. Evaluate untouched datasets or a genuinely sealed cohort after freezing the
   complete method and analysis.
3. Add author-faithful, equally tuned modern and classical MI baselines. The
   primary indexed+spline control is checkpoint matched but explicitly not an
   author-faithful FBMSNet reproduction.
4. Add montage/channel-removal robustness, calibration, computational-cost,
   and ablation studies isolating the coordinate field, cardinal anchors,
   geometry regularization, and pretraining.
5. Report subject-only uncertainty precisely; do not present folds, target
   seeds, or the two opened datasets as independent inferential replicates.
6. Do not claim clinical effectiveness for disabled or paralyzed users without
   data from the intended population and an appropriate prospective protocol.

Because the candidate and protocol were selected using opened development
data, even this complete repeated-fold result is not independent confirmation.
It is a strong basis for the next frozen validation stage, not the final paper
claim by itself.
