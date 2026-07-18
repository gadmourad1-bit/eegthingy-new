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
