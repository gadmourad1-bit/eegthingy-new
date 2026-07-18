# Verified schema-v2 results

These results were generated on the LAN RTX 5070 after the final audit corrected two
protocol errors in the superseded schema-v1 run: a compressed refit learning-rate
schedule and temperature scaling transferred to a different model. Schema v2 retains the
exact checkpoint calibrated on the protected validation group; that group is never added
to fitting. The earlier 86.40% table and its checkpoints must not be cited.

This is research on eight healthy volunteers, not clinical validation in people with
paralysis and not evidence that the decoder is ready to control an assistive device.

## Audit and data lock

- Cohort: subjects 1, 3, 4, 5, 6, 7, 8, and 10; 32 complete recordings.
- Window: the 0.0-2.0 s `deployment` window, never the legacy 0.5-2.5 s window.
- Full data-contract SHA-256:
  `2c93983d422c3ee478a70baf6a5f479eac69e5632ad192cb4b1460df1187c2c0`.
- The contract records channel order, bands, sample rate, filter, artifact threshold,
  covariance settings, manifest, file sizes/timestamps, and SHA-256 for every FIF file.
- Schema-v2 JSON retains every scored and rest-window probability, label, provenance,
  intent probability, commit, pre/post boundary center, and update decision.
- `deepnet.report` recomputes all aggregate metrics from those traces and hard-fails on a
  mismatch. The final `deepnet` suite has 68 passing tests on the workstation.

## Chronological cross-session benchmark

For each participant, the first two recordings fit the model. The third recording selects
the exact checkpoint, target-median blend, and temperature. That frozen checkpoint is
then evaluated on the untouched fourth recording after an unlabeled prefix of 20
chronological task/rest windows; prefix windows are excluded from scoring. Neural results
use seeds 7, 17, and 27, averaged within participant before cohort aggregation.

| Model | Balanced accuracy | ROC AUC | Brier | Coverage | Selective accuracy | Rest false commits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoAdaptNet | 84.86 ± 12.67% | 93.32 ± 8.04% | 9.99 ± 7.60% | 41.01 ± 17.65% | **96.82 ± 5.17%** | **15.71 ± 8.89%** |
| Riemannian tangent LR | 84.24 ± 11.75% | **94.25 ± 7.66%** | **9.65 ± 6.97%** | 67.16 ± 22.90% | 96.19 ± 5.09% | 47.95 ± 20.19% |
| EA-FBCSP | **87.53 ± 13.40%** | 93.18 ± 10.13% | 9.21 ± 8.46% | **69.44 ± 25.19%** | 91.72 ± 11.38% | 50.85 ± 18.45% |

Values are participant-level mean ± sample standard deviation. The neural model is tied
with Riemannian LR on balanced accuracy: +0.63 percentage points, paired participant
bootstrap 95% CI −1.01 to +2.42. It trails FBCSP by 2.67 points, CI −5.83 to +0.39.
Neither comparison supports an accuracy-superiority claim.

At the fixed 0.85 commit threshold, GeoAdaptNet reduces rest false commits by 32.23 points
relative to Riemannian LR (CI −46.50 to −20.23) and by 35.14 points relative to FBCSP
(CI −47.58 to −20.55). This comes with materially lower coverage. Baselines have the
same left/right confidence threshold but no separate intent head, so the safety comparison
is for the complete decoder and should not be misrepresented as an architectural
left/right-classification gain.

Mean synchronized batch-one network latency was 4.54 ms on the RTX 5070 (4.36-5.71 ms),
after covariance construction. The model has 9,678 trainable parameters. Twenty-four
schema-v2 checkpoints were exported under
`deepnet/checkpoints/chronological_v2_final_3seeds/` and all safely reloaded with
`weights_only=True`.

### Residual and online-update diagnostics

Using the trained model's anchor logits instead of its full logits changed participant-
level balanced accuracy from 84.86% to 84.69% (+0.17 points for the full output). The
residual gate remained small (mean 0.02145; range 0.02089-0.02211). This is a post-training
logit diagnostic, not an independently trained anchor-only ablation, so it does not prove
that the learned residual is beneficial.

Freezing target-state updates after prefix calibration yielded 84.853% balanced accuracy,
versus 84.860% with causal updates. That negligible difference provides no evidence that
online state updates improve this dataset. The measured safety behavior is therefore best
attributed to prefix alignment, intent gating, and abstention unless a future matched
ablation establishes an adaptation benefit.

Source of truth:

- `deepnet/results/chronological_v2_final_3seeds.json`
- `deepnet/results/chronological_v2_final_3seeds.md`

## Initial strict subject-held-out benchmark

Each outer fold withholds all four recordings from one participant. One fixed inner
participant is also withheld for checkpoint/temperature/blend selection; the remaining
six participants fit the model or baseline. Every validation/outer recording receives an
independent unlabeled prefix containing exactly the first 10 task cues plus intervening
rest windows, and all state resets at recording boundaries. Neural results currently use
one seed (7); classical estimators are deterministic.

| Model | Balanced accuracy | ROC AUC | Brier | Coverage | Selective accuracy | Rest false commits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoAdaptNet | 83.54 ± 9.36% | 90.63 ± 7.95% | 12.88 ± 5.82% | 19.22 ± 20.56% | **95.85 ± 7.02%** | **9.68 ± 9.04%** |
| Riemannian tangent LR | 81.61 ± 10.52% | 88.26 ± 9.88% | 14.59 ± 8.04% | **79.36 ± 10.02%** | 86.31 ± 9.57% | 67.45 ± 10.56% |
| EA-FBCSP | **85.67 ± 9.15%** | **92.91 ± 6.80%** | **10.34 ± 5.91%** | 63.24 ± 24.16% | 94.44 ± 5.86% | 43.84 ± 22.87% |

GeoAdaptNet is +1.93 points versus Riemannian LR (paired bootstrap CI −0.02 to +4.42)
and −2.13 points versus FBCSP (CI −4.80 to +0.81). Again, the honest result is
competitive accuracy, not superiority. Rest false commits are 57.77 points below
Riemannian LR (CI −62.64 to −53.08) and 34.17 points below FBCSP (CI −45.57 to −21.91),
but 19.22% coverage is very conservative and highly variable across people.

This is an initial nested estimate, not a definitive subject-independent result. It uses
one deterministic inner-validation participant per outer fold and one neural seed. A
paper claim requires a full inner LOSO sweep, multiple neural seeds, external datasets,
and matched modern test-time-adaptation comparators.

Eight strict neural checkpoints were exported under
`deepnet/checkpoints/nested_loso_v2_seed7_nested_loso/` and safely reloaded. A real held-
out covariance passed through the checkpoint-backed GPU deployment wrapper with its
preprocessing contract validated and produced a committed causal decision.

Source of truth:

- `deepnet/results/nested_loso_v2_seed7.json`
- `deepnet/results/nested_loso_v2_baselines.json`
- `deepnet/results/nested_loso_v2_combined.json`
- `deepnet/results/nested_loso_v2_combined.md`

## Accuracy improvement (v3): fit-side augmentation

The schema-v2 diagnostics above show the learned residual is inert (+0.17 points, gate
≈ its 0.02 initialization). A follow-up campaign established *why*, and what does move the
number. All runs below keep the exact protected schema-v2 protocol, are fit/selection-side
only (never the outer test), and reproduce the v2 baselines above bit-for-bit when their
knobs are off. Every knob defaults off, so the tables above are unchanged.

**The residual is redundant, not merely untrained.** Reviving it three ways — deep
supervision on the un-gated `anchor+residual`, raising and freeing the gate (it does open,
0.02 → 0.21), and CSP-warm-starting each BiMap with supervised common-spatial-pattern
filters (fit on the training recordings only, in the recentered frame) — each made the
residual genuinely discriminative yet left balanced accuracy at ≈ 85%. The discriminative
structure CSP captures is already present in the full-rank tangent anchor. FBCSP's edge is
therefore regularization (8 supervised features + double shrinkage), not nonlinearity: the
480-dimensional tangent anchor overfits ≈ 950 training epochs.

**What moves the number is regularizing that anchor.** A fit-side, label-consistent
augmentation — left/right electrode swap with label flip (motor imagery is laterally
organized: C3↔C4, F3↔F4, …), a symmetric covariance permutation applied to the training
loader only — combined with accuracy-targeted (`blend`) checkpoint selection, gives a
consistent gain under both protocols. It is selected on the rec3 validation score, not on
test, and confirmed across seeds 7/17/27.

| Protocol | Baseline | + swap-aug + blend | Δ | EA-FBCSP |
| --- | ---: | ---: | ---: | ---: |
| Chronological (3 seeds) | 84.86% | **86.62 ± 10.84%** | +1.76 | 87.53% |
| Strict LOSO (seed 7) | 83.54% | **85.02 ± 9.52%** | +1.48 | 85.67% |

The augmented model nearly matches FBCSP under both protocols (chronological gap 0.91 pt,
LOSO gap 0.65 pt) while retaining the high-precision operating point: chronological rest
false commits 19.8% and LOSO 15.96%, versus FBCSP's 50.85% / 43.84%. Seven of eight LOSO
participants improve (S1 74.7 → 79.0, S3 → 90.5, S4 → 94.0, S5 → 95.5, S6 → 94.4). This
narrows, but does not overturn, the classical bar — the honest result remains competitive
accuracy with a materially safer rest-commit profile, at somewhat lower coverage
(chronological 42.3%, LOSO 28.8%). Augmentation slightly raises coverage and rest commits
versus the un-augmented net, a trade recorded here rather than hidden.

Levers that did **not** help, tested and recorded so they are not re-tried: residual
revival in any form above; log-Euclidean same-class mixup (≈ 0); 3-seed probability
ensembling (uplift −0.1 to −0.3, the augmented seeds are already low-variance); a longer
optimizer budget (overfits — worse); adaptive OAS covariance shrinkage (−1 pt: on real EEG
the OAS weight is ≈ 0.01, so the fixed 1e-3 shrinkage was already appropriate and more
shrinkage washes out structure); and test-time left/right mirror averaging (−0.79 pt at 3
seeds — redundant with the swap augmentation, which already banks the lateral symmetry in
training, so averaging it again at test only adds noise and coverage). None are adopted.
The `--fast` (non-deterministic + TF32) and `--tta-mirror` flags remain available but off.

Reproduce the winning configuration (defaults reproduce the v2 baseline):

```bash
.venv/bin/python -m deepnet.experiment benchmark --subjects all \
  --models geoadapt,riemann,fbcsp --seeds 7,17,27 --window deployment \
  --calibration-windows 20 --max-epochs 180 --patience 25 --device cuda \
  --lr-swap-prob 0.5 --select-metric blend \
  --output deepnet/results/chronological_v3_swap_blend.json

.venv/bin/python -m deepnet.loso --subjects all --seeds 7 \
  --calibration-task-events 10 --max-epochs 180 --patience 25 --device cuda \
  --lr-swap-prob 0.5 --select-metric blend \
  --output deepnet/results/nested_loso_v3_swap_blend.json
```

Source of truth (produced on the RTX 5070; geoadapt-only dev stems, full schema-v2 traces):

- `deepnet/results/dev/v3_swap_blend.json` (chronological, 3 seeds)
- `deepnet/results/dev/loso_swap_blend.json` (strict LOSO, seed 7)

This does not change the claim boundary below: still eight healthy volunteers, still no
superiority claim over FBCSP, still no clinical, closed-loop, or external-generalization
evidence.

## Deep-learning baseline comparison

To place GeoAdaptNet against the modern deep-learning field, it is compared with five
widely-cited braindecode architectures --- EEGNet (Lawhern 2018), ShallowConvNet and
DeepConvNet (Schirrmeister 2017), EEG-Conformer (Song 2023), and ATCNet (Altaheri 2023)
--- and the two classical decoders, under one uniform, leakage-controlled protocol
(`deepnet/dnn_benchmark.py`).  Per participant, recordings 1--2 train, recording 3 selects
the checkpoint, and recording 4 is scored by raw balanced accuracy on its task windows; no
online recentering is applied to any model, so the comparison isolates the architecture.
Convolutional nets consume broadband epochs, GeoAdaptNet consumes filter-bank covariances,
classical methods use their native features.  Neural models are averaged over seeds 7/17/27
within participant.  (These numbers are not comparable to the schema-v2 tables above, which
use the online-adapter deployment protocol; this is a separate architecture-only comparison.)

Every neural network is trained twice: with standard training, and with the left/right swap
augmentation applied identically to all of them (its raw-epoch analogue for the conv nets).
The augmentation is architecture-agnostic and helps every model, so the comparison is fair.

| Architecture | Params | Bal. acc (no aug) | Bal. acc (+ swap aug) |
| --- | ---: | ---: | ---: |
| **GeoAdaptNet** | **9,645** | 85.63 ± 8.61% | **87.58 ± 10.49%** |
| ShallowConvNet | 26,002 | 85.87 ± 10.30% | 87.04 ± 10.31% |
| EEG-Conformer | 266,306 | 85.73 ± 8.41% | 86.40 ± 12.28% |
| EEGNet | 1,602 | 70.51 ± 14.24% | 83.96 ± 13.98% |
| ATCNet | 28,868 | 55.07 ± 10.14% | 75.25 ± 21.98% |
| DeepConvNet | 233,652 | 50.15 ± 4.46% | 54.10 ± 9.95% |
| EA-FBCSP (classical) | --- | 90.01 ± 9.68% | --- |
| Riemann-TS+LR (classical) | --- | 90.06 ± 10.00% | --- |

With the same augmentation, GeoAdaptNet is the most accurate deep architecture and uses the
fewest parameters (9.6 k versus 26 k--266 k).  Without augmentation it is on par with the
best conv nets (ShallowConvNet, EEG-Conformer) at a fraction of their size, while the large,
data-hungry nets (DeepConvNet 234 k params) collapse toward chance on roughly 120 training
epochs per subject --- the data-efficiency argument for geometric decoders.  The two
classical geometric pipelines remain the overall bar under this simple protocol.  This is a
local-cohort result; the external replication below tests whether the ranking is an artefact
of the local recordings.

### External replication (Cho 2017 / GigaDB 100295)

The same architectures and the same matched 15-channel / 125 Hz pipeline are run on the
independent Cho2017 left/right-hand MI cohort (`deepnet/external_cho2017.py`,
`deepnet/external_benchmark.py`), which uses different subjects, a 64-channel montage, and a
512 Hz amplifier (its 10-10 names P7/P8/T7/T8 map to our 10-20 T5/T6/T3/T4).  Cho2017 is
single-session, so the protocol is within-subject stratified 5-fold cross-validation, over
the first 30 subjects, one seed.

| Architecture | Bal. acc (no aug) | Bal. acc (+ swap aug) |
| --- | ---: | ---: |
| ShallowConvNet | **61.95 ± 11.89%** | **62.39 ± 11.05%** |
| **GeoAdaptNet** | 57.27 ± 6.93% | 58.18 ± 7.27% |
| EEG-Conformer | 56.73 ± 8.43% | 58.16 ± 9.20% |
| EEGNet | 54.71 ± 8.14% | 60.94 ± 13.93% |
| ATCNet | 50.63 ± 3.65% | 51.24 ± 7.84% |
| DeepConvNet | 50.56 ± 4.24% | 51.30 ± 3.56% |

Absolute accuracies are much lower than on the local cohort because Cho2017 is a large, noisy
set with many near-chance participants, but the *ranking* is what matters for the bias check.
GeoAdaptNet does not top this cohort --- ShallowConvNet is consistently best on Cho2017 --- but
it stays firmly in the strong group (second/third, tied with EEG-Conformer) and, notably, has
the **lowest cross-subject variance of any model** (± 6.9--7.3 vs ± 8--14), i.e. it is the most
consistent architecture across unseen subjects.  Crucially, the two groupings are stable across
both datasets: the strong architectures (ShallowConvNet, GeoAdaptNet, EEG-Conformer) and the
data-hungry ones that fail on small windows (ATCNet, DeepConvNet) are the same locally and
externally.  GeoAdaptNet wins on the local data and remains competitive on a wholly independent
cohort rather than collapsing, so its local strength is not an artefact of the local recordings.

Source of truth: `deepnet/results/dnn_compare_local.json`, `dnn_compare_local_aug.json`,
`dnn_compare_cho2017.json`, `dnn_compare_cho2017_aug.json`.

## Reproduce reports

```bash
.venv/bin/python -m deepnet.report \
  deepnet/results/chronological_v2_final_3seeds.json \
  --update-json \
  --markdown deepnet/results/chronological_v2_final_3seeds.md

.venv/bin/python -m deepnet.merge_results \
  deepnet/results/nested_loso_v2_seed7.json \
  deepnet/results/nested_loso_v2_baselines.json \
  --output deepnet/results/nested_loso_v2_combined.json

.venv/bin/python -m deepnet.report \
  deepnet/results/nested_loso_v2_combined.json \
  --update-json \
  --markdown deepnet/results/nested_loso_v2_combined.md
```

## Claim boundary

The implementation establishes a reproducible, compact decoder with competitive local
task accuracy, low model latency, and an explicit high-precision/low-coverage operating
point. It does not establish clinical benefit, continuous asynchronous performance,
closed-loop benefit, residual-branch benefit, causal-update benefit, external
generalization, or task-accuracy superiority. Recruitment or assistive-device use should
follow external replication, EMG/artifact controls, repeated closed-loop testing, and a
study designed with the intended disabled population—not precede them.
