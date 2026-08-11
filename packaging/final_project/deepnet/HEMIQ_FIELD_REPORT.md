# HemiQ-FieldNet v2 technical report

Date: 2026-07-18  
Status: implementation complete; development and one-shot confirmation complete; post-run audit passed

## Executive result

HemiQ-FieldNet is a new 11,354-parameter, single-logit neural decoder for
left-versus-right motor imagery from the three bipolar C3/Cz/C4 recordings in
BNCI2014-004 (BCI Competition IV data set 2b). Its internal feature field is
constructed so that sagittal reflection has a prescribed action:

\[
f(Mx)=-f(x),
\]

where \(M\) swaps C3 and C4. This is an architectural identity, not an
augmentation loss. The maximum measured value of
\(|f(x)+f(Mx)|\) was exactly `0.0` in every final validation and test cohort.

The final configuration was selected using subjects 1--4. Subjects 5--9 were
absent from the machine until source, configuration, environment, output path,
and permanent one-shot receipts had been frozen and independently audited.
The protected confirmation result was:

- HemiQ-FieldNet: **82.50%** mean balanced accuracy.
- ShallowConvNet plus left/right swap augmentation: 82.375%.
- Riemannian tangent-space logistic regression: 75.125%.
- Frozen tangent anchor: 75.00%.
- shrinkage FBCSP-LDA: 74.50%.

HemiQ has the highest observed mean. It is effectively tied with the matched
ShallowConvNet (+0.125 percentage point, exact paired `p=1.0`) and is 7.375 to
8.00 points above the three classical baselines. With only five confirmation
subjects, none of the exact subject-level comparisons reaches `p < .05`.
Accordingly, this is evidence of strong performance and a useful inductive
bias, not evidence of population superiority, clinical efficacy, or a new
state of the art.

## Intended role and claim boundary

Motor-imagery BCIs may eventually provide an additional control channel for
people who cannot reliably use their limbs or speech. This experiment only
addresses the offline signal-decoding component. The original data were
recorded from nine healthy volunteers, not from disabled or paralyzed users.
No assistive device, asynchronous control loop, safety study, or clinical
outcome was tested.

The intended technical contribution is narrow:

> A binary motor-imagery decoder built from an invariant-conditioned,
> multiscale reflection-odd complex cross-spectral field, with exact label
> anti-equivariance enforced through every learned odd path and the final
> attention readout.

## Architecture

### 1. Parity coordinates

Let \(x_3,x_z,x_4\) denote the C3, Cz, and C4 signals. Define

\[
e=(x_3+x_4)/\sqrt{2},\qquad
o=(x_3-x_4)/\sqrt{2},\qquad
z=x_z.
\]

Under sagittal reflection \(M\), which swaps C3 and C4,

\[
e\mapsto e,\qquad z\mapsto z,\qquad o\mapsto-o.
\]

The source-only scaler uses one shared mean and standard deviation for C3 and
C4, so scaling commutes with the same reflection.

### 2. Constrained quadrature Gabor bank

A shared bank of 12 learned Gaussian-windowed quadrature filters analyzes
8--30 Hz. Filter centers are represented by positive cumulative gaps, making
them strictly ordered and interior to the band. Bandwidths are bounded to
1.5--8 Hz. Cosine and sine components share one envelope, have their finite
window DC component removed, and are L2-normalized.

The default temporal kernel has 63 samples at 125 Hz and stride 2. Sharing the
same bank across all parity signals preserves their transformation type. Let
the resulting complex coefficients be \(q_e,q_z,q_o,q_3,q_4\). Then

\[
q_e\mapsto q_e,\quad q_z\mapsto q_z,\quad q_o\mapsto-q_o,
\quad q_3\leftrightarrow q_4.
\]

### 3. Local and full-epoch spectral tokens

Tokens are computed at two temporal scales:

- overlapping local windows; and
- one full-epoch token per learned frequency.

For either averaging operator \(E_s\), the first four odd coordinates are the
real and imaginary parts of normalized cross spectra:

\[
c_{oe}=\frac{E_s[q_o q_e^*]}
 {\sqrt{E_s|q_o|^2E_s|q_e|^2+\epsilon}},\qquad
c_{oz}=\frac{E_s[q_o q_z^*]}
 {\sqrt{E_s|q_o|^2E_s|q_z|^2+\epsilon}}.
\]

Both complex quantities negate under reflection. Four additional robust odd
coordinates explicitly encode hemispheric power asymmetry. First define

\[
u=\tanh\left(
\frac{\log(|q_3|^2+\epsilon)-\log(|q_4|^2+\epsilon)}{2}
\right).
\]

The field contains

\[
E_s[u],\qquad E_s[u|u|],\qquad E_s[u^3],\qquad
\tanh\left(\frac{\log P_3-\log P_4}{2}\right),
\]

where \(P_3=E_s|q_3|^2\) and \(P_4=E_s|q_4|^2\). Each coordinate is bounded
and reflection-odd.

The invariant context contains three log powers, three coherence magnitudes,
and normalized frequency, time, and log temporal-scale coordinates. The
magnitude of an odd cross spectrum is invariant, so the context is unchanged
by \(M\).

### 4. Invariant-conditioned odd network

An unconstrained context MLP maps each invariant token to \(h\). Odd token
features \(r\) pass through two conditioned blocks of the schematic form

\[
r' = \tanh\left[W_2\left(
\tanh(W_0r)+\sigma(C(h))\odot\tanh(W_1r)
\right)\right].
\]

All \(W_i\) on odd paths are bias-free. The conditioner \(C\) may have biases
because it consumes only invariant values. `tanh` is odd and the gate is
invariant; therefore \(r(Mx)=-r(x)\) is preserved. The second block is an odd
residual block.

Invariant attention weights are

\[
\alpha_i=\operatorname{softmax}_i A(h_i),
\]

and token scores are produced by a bias-free odd readout \(s_i=w^Tr_i\). The
sole network output is

\[
f(x)=\sum_i\alpha_i s_i.
\]

Because \(\alpha_i(Mx)=\alpha_i(x)\) and \(s_i(Mx)=-s_i(x)\),
\(f(Mx)=-f(x)\) exactly. There is no candidate head, router, test-time
ensemble, batch normalization, dropout, inference-time tangent branch, or
learned decision threshold.

### 5. Training-only teacher

A source-only projected tangent logistic model supplies soft targets during
early optimization. Its reflection-even component is projected out, and its
coefficient starts at 0.25 and decays linearly on the scheduled training
horizon. It never sees validation or test rows, is absent from inference, and
cannot be selected as an alternative output. Checkpoint selection uses only
the neural logit's validation binary cross-entropy. The fixed decision
threshold is zero.

The neural model has 11,354 trainable parameters. The training-only tangent
teacher has 25 fitted logistic parameters and is not included in the neural
parameter count.

## Data and locked protocol

The official description identifies nine subjects, five sessions, three
bipolar C3/Cz/C4 EEG recordings at 250 Hz, and left/right hand imagery. It also
states that the first two sessions are screening sessions and the last three
use feedback. See the [official data-set description](https://bbci.de/competition/iv/desc_2b.pdf)
and [MOABB BNCI2014-004 documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_004.html).

The study protocol was:

- development subjects: S1--S4;
- one-shot confirmation subjects: S5--S9;
- fit sessions: `0train`, `1train`;
- validation/checkpoint session: `2train`;
- prediction-only outer test sessions: `3test`, `4test`;
- epoch: cue +0.5 s through +2.5 s, inclusive;
- downsampled frequency: 125 Hz, 251 samples;
- broadband range: 8--30 Hz;
- channels: the provided bipolar C3, Cz, C4 signals; no CAR rereference;
- fixed seed: 7;
- optimizer: AdamW, learning rate `7e-4`, weight decay `5e-4`;
- batch size: 64; maximum 240 epochs; patience 40;
- selection metric: validation BCE only;
- no test calibration, adaptation, threshold tuning, or test-time augmentation.

Released sessions contain balanced official counts of 120, 140, or 160
trials. The loader accepts only those values, verifies exact per-session class
balance, finite arrays, positive-definite covariances, protocol IDs, and cache
hashes, and denies S5--S9 unless invoked by the locked confirmation runner.

Before the first S5 row could be loaded, each study required:

1. exact ordered subjects and seed;
2. a validated development artifact;
3. current source hashes matching the frozen source map;
4. the exact model/baseline contract;
5. exact CUDA, package, and deterministic-runtime environment;
6. an absolute predeclared output path;
7. an explicit confirmation token; and
8. a permanent `O_EXCL` receipt claimed before loader access.

Receipts are never removed. A completed study cannot be rerun; an interrupted
study could only resume the same artifact, source, environment, and path.

## Development history

The initial single-scale field reached 67.89% development balanced accuracy,
just above the strongest fixed baseline at 67.48%. Development-only tests of
pooling scale, width, optimization strength, teacher duration, filter density,
and fixed seeds did not justify freezing that version.

Adding the full-epoch tokens and explicit bounded hemispheric power moments
produced the decisive development improvement. The default v2 configuration
reached 70.76% across S1--S4. A shorter-window/stronger-teacher setting reached
70.98% at seed 7, but a four-seed robustness check slightly favored the simpler
default (approximately 69.93% versus 69.91%). The default was therefore frozen.
All exploration was confined to S1--S4.

| Method | Development BA, S1--S4 |
| --- | ---: |
| **HemiQ-FieldNet v2, frozen default** | **70.7589%** |
| FBCSP-LDA | 67.4777% |
| Riemann tangent logistic | 67.4219% |
| tangent anchor | 67.4219% |
| ShallowConvNet + swap | 64.5982% |

## One-shot confirmation results

Balanced accuracy and HemiQ-minus-baseline deltas are shown below.

| Subject | HemiQ | Shallow / delta | Riemann / delta | Tangent / delta | FBCSP / delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| S5 | 89.0625% | 84.3750 / +4.6875 | 83.7500 / +5.3125 | 83.7500 / +5.3125 | 65.3125 / +23.7500 |
| S6 | 81.5625% | 80.0000 / +1.5625 | 75.0000 / +6.5625 | 75.0000 / +6.5625 | 77.8125 / +3.7500 |
| S7 | 78.7500% | 77.1875 / +1.5625 | 67.8125 / +10.9375 | 67.5000 / +11.2500 | 70.9375 / +7.8125 |
| S8 | 79.6875% | 86.2500 / -6.5625 | 77.1875 / +2.5000 | 76.8750 / +2.8125 | 86.5625 / -6.8750 |
| S9 | 83.4375% | 84.0625 / -0.6250 | 71.8750 / +11.5625 | 71.8750 / +11.5625 | 71.8750 / +11.5625 |
| **Mean** | **82.5000%** | **82.3750 / +0.1250** | **75.1250 / +7.3750** | **75.0000 / +7.5000** | **74.5000 / +8.0000** |

Exact two-sided subject-level sign-permutation tests enumerate all 32 sign
assignments:

- HemiQ versus ShallowConvNet: `p = 1.0000`;
- HemiQ versus Riemann: `p = .0625`;
- HemiQ versus tangent anchor: `p = .0625`;
- HemiQ versus FBCSP-LDA: `p = .1875`.

The smallest attainable nonzero two-sided value with five nonzero paired
differences is .0625. The consistent HemiQ advantage over Riemann and the
tangent anchor is encouraging, but this experiment is underpowered for a
population-level superiority claim.

## Novelty boundary

This is not the first work to use lateral structure, cross spectra, learnable
wavelets, attention, or equivariance:

- [Mirror CNN](https://www.cjig.cn/en/article/doi/10.11834/jig.210072/)
  swaps left/right channels with flipped labels and averages source/mirror CNN
  predictions.
- A [dual-hemisphere difference CNN](https://doi.org/10.3389/fnins.2022.865594)
  subtracts features from two parameter-shared hemispheric CNNs.
- [MCL-SWT](https://arxiv.org/abs/2409.00130) uses mirror contrastive loss in a
  sliding-window Transformer.
- [GREEN](https://pmc.ncbi.nlm.nih.gov/articles/PMC11963017/) uses learnable
  complex Gabor wavelets in a lightweight EEG architecture.
- [KCS-FCNet](https://www.mdpi.com/2075-4418/13/6/1122) learns from kernel
  cross-spectral functional connectivity for motor imagery.
- [RatioWaveNet](https://doi.org/10.1016/j.array.2026.100961) combines a
  learnable wavelet transform with multi-window attention and temporal
  convolution.
- Exact group-equivariant networks are a broad established field; see
  [Kondor and Trivedi](https://proceedings.mlr.press/v80/kondor18a.html).

After a targeted literature search, no located work used HemiQ's particular
combination of a multiscale complex cross-spectral field with explicit
reflection-odd power moments, invariant-conditioned bias-free odd blocks, and
an invariant attention sum to guarantee a label-negating EEG logit throughout
the network. That supports a narrow “to our knowledge” architecture claim. It
is not a formal exhaustive, peer-reviewed, or patent novelty determination.

## Verification and audit

The full GPU suite completed with **242 passed** and five non-failing warnings.
A MacBook CPU smoke test produced a `(4,)` logit tensor, counted 11,354
parameters, and measured reflection error `0.0`.

An independent post-run audit recomputed all metrics from 8,000 stored trial
predictions, all 60 confirmation cache-array digests, session/order/label
traces, covariance checks, summaries, source hashes, manifests, environments,
receipts, and exact paired tests. It found no metric, data, receipt, source, or
protocol blocker.

| Artifact | SHA-256 |
| --- | --- |
| HemiQ development | `9c13debaa25c4cc91eddb1a1ea6b337abeaa5ab079cf16f79e4ef79e844684d3` |
| HemiQ frozen manifest | `2bbe28d4415a74cb007328579cf8153585ceaaa28219246aecd0622398987401` |
| HemiQ confirmation | `d287571b5309213fe88408d86f25fbbb20d905682cf61a3d6b7180e89720bcf8` |
| HemiQ permanent receipt | `1271c1441ecbf16ae0dcd75fbe45c205b8fb0190b3382c12affb5f7921b35b14` |
| baseline development | `3e2cf01193726f2b8eaf0e94d0943fea9c3bb78e816073a71cc9bd77d5db71a1` |
| baseline frozen manifest | `890f9dbe8e3951bf3929324a92646810fbeff90fcd53cbdf05f87535684ae667` |
| baseline confirmation | `15c707118284173f32841c7a7ef2efb655925895d07070db7027d1d644ffe248` |
| baseline permanent receipt | `62e70e1c6385ceb162dfdd9092d781b09e79599936ac1b1c3c159d01ff35b897` |

Additional canonical hashes:

- HemiQ source manifest: `a6ab2c7363325b3da187f6fa1decc67b7ee143e4e25b71346bcdb7c96f6554f6`;
- baseline source mapping: `e477edf67d1e15b771dc298409607742dd27babdb02268060fe4dd1ba025dc9b`;
- HemiQ environment: `ed7c928c53942f14330b0608bbcf9ca19a35086eb55ba8905b5c5b9a0fe8c4e5`;
- baseline environment: `baff1f017d6cd1da3b0a1d62c7d1f763965ff099bf0e1a9801879a5ec113044c`.

## Reproducibility map

Core implementation:

- `deepnet/hemi_q_field_net.py`
- `deepnet/hemi_q_benchmark.py`
- `deepnet/external_bnci2014_004.py`
- `deepnet/bnci004_baseline_benchmark.py`

Tests:

- `deepnet/tests/test_hemi_q_field_net.py`
- `deepnet/tests/test_hemi_q_benchmark.py`
- `deepnet/tests/test_external_bnci2014_004.py`
- `deepnet/tests/test_bnci004_baseline_benchmark.py`

Final artifacts:

- `deepnet/results/hemi_q/hemi_q_final_bnci004_dev_s1_4.json`
- `deepnet/results/hemi_q/hemi_q_final_frozen_manifest.json`
- `deepnet/results/hemi_q/hemi_q_final_bnci004_confirmation_s5_9.json`
- `deepnet/results/hemi_q/.confirmation-receipts/eegthingy-hemiq-bnci2014-004-s5-9-final-v1.json`
- `deepnet/results/parity/bnci004_fixed_baselines_final_dev_s1_4.json`
- `deepnet/results/parity/bnci004_fixed_baselines_final_frozen_manifest.json`
- `deepnet/results/parity/bnci004_fixed_baselines_final_confirmation_s5_9.json`
- `deepnet/results/parity/.confirmation-receipts/eegthingy-fixed-baselines-bnci2014-004-s5-9-final-v1.json`

Historical GPU commands are recorded inside the JSON artifacts. The two
confirmation commands now intentionally refuse to run again because their
permanent receipts are complete. The following remains safe and reproducible:

```bash
# Full test suite on the isolated GPU workspace
env -C /home/user/Desktop/eegthingy_codex_arch_20260718 \
  ./.venv/bin/python -m pytest -q deepnet/tests

# Mac/CPU architecture smoke test
.venv/bin/python -c "import torch; from deepnet.hemi_q_field_net import HemiQFieldNet; \
m=HemiQFieldNet().eval(); x=torch.randn(4,3,251); \
print(m(x).shape, sum(p.numel() for p in m.parameters()), \
float((m(x)+m(m.reflect(x))).abs().max()))"
```

## Limitations and next experiments

1. The confirmation cohort has only five subjects; exact inference is coarse.
2. This is within-subject decoding with labeled calibration sessions, not
   subject-independent or zero-calibration decoding.
3. The data come from healthy volunteers and only distinguish left/right hand
   imagery from three bipolar channels.
4. Offline balanced accuracy does not establish asynchronous false-activation
   behavior, closed-loop utility, latency under a real acquisition stack, or
   clinical benefit.
5. ShallowConvNet was frozen at one seed and its artifact declares
   `deterministic=false`; its 0.125-point difference from HemiQ is meaningless
   at this sample size.
6. The sealed baseline JSON reports Riemann `parameter_count=0`. Its actual
   four-band, three-channel tangent vector has 24 features and the logistic
   model has 24 coefficients plus one intercept, so the corrected fitted count
   is 25. Scores and predictions are unaffected.
7. The cache identity hashes the loader and numeric preprocessing module but
   not every transitive configuration/package dependency. Both benchmark
   manifests separately pin `config.py`, source mappings, and the execution
   environment, and the audit verified the actual arrays; a future cache
   schema should make that dependency graph explicit.
8. The isolated GPU workspace did not contain Git metadata. Full per-file
   source hashes compensate for numeric provenance but do not identify a
   repository commit.
9. The novelty assessment was targeted, not exhaustive.

The next defensible work is a preregistered replication on a larger untouched
cohort, followed by artifact/EMG controls and a closed-loop study designed for
the intended disabled population. Confirmation subjects in this study must not
be recycled for further architecture selection.
