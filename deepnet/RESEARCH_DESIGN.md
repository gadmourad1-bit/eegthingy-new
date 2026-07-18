# Research design and claim boundary

## Intended contribution

The project asks whether a very small geometric neural decoder can preserve a strong
low-data EEG baseline while improving robustness under session drift, then adapt online
without labels, gradients, replay, or future-window access.

The defensible contribution is the evaluated system, not any one ingredient:

1. A full-resolution tangent-space anchor that the learned branch can only modify through
   a near-zero-initialized residual gate.
2. A compact filter-bank SPD residual with learned semi-orthogonal projections and band
   attention.
3. Two strictly causal, constant-memory, backpropagation-free deployment states: a robust
   log-Euclidean covariance reference and a rest-gated scalar output boundary.
4. Explicit abstention and an optional intent-versus-rest head designed for command safety.
5. Leakage-resistant chronological and subject-held-out tests, plus closed-loop outcomes
   and latency rather than offline epoch accuracy alone.

Do not claim this is the first SPD EEG network, the first online EEG adaptation method,
the first backpropagation-free EEG adaptation method, or the first geometry-aware
recentring method. All of those lanes already contain prior work. Novelty is provisional
until the ablations and external comparisons below show that this particular combination
adds measurable value.

## Research questions

Primary questions:

- Does the protected geometric residual improve chronological cross-session and LOSO
  balanced accuracy over its full tangent-space anchor under the same preprocessing and
  tuning budget?
- Does causal covariance recentering improve later-session performance relative to both
  no alignment and a frozen calibration reference?
- Does causal boundary recentering add value after covariance recentering, rather than
  merely compensating for a poorly calibrated classifier?
- Can confidence gating improve command accuracy at useful coverage, and reduce false
  commits during rest, without unacceptable latency?
- Do any gains survive participant-level aggregation and seed variation?

Secondary questions:

- Is the learned SPD residual more data-efficient than a generic EEG deep network?
- Which filter bands receive attention across people and sessions, and are the patterns
  stable enough to interpret cautiously?
- Does performance in the offline epoch task predict closed-loop navigation success?

## Model hypothesis

### Input and reference alignment

Each task window yields four 15 x 15 shrinkage covariance matrices. For band `b`, the
calibration reference is maintained in log coordinates:

```text
M_b = mean_i(log(C_i,b))
R_b = exp(M_b)
C_aligned = R_b^(-1/2) C R_b^(-1/2)
```

The congruence preserves positive definiteness and maps the current reference to identity.
Eigenvalue floors are relative to each covariance scale, which matters because raw EEG
covariances in volts squared are many orders of magnitude below aligned covariances.

### Anchor path

The matrix logarithm of every aligned covariance is vectorized with square-root-of-two
weighting on off-diagonal entries, preserving the symmetric matrix Frobenius norm. Four
120-element vectors are concatenated, standardized with source-only running statistics,
and passed through a linear two-class head:

```text
anchor: 4 x vech(log(C_aligned)) -> source standardization -> 480 -> 2
```

This is intentionally close to a strong tangent-space logistic classifier. It is both a
baseline inside the network and a guard against replacing a useful small-data solution
with an overfit feature extractor.

### Geometric residual path

For each band:

```text
15 x 15 SPD
  -> semi-orthogonal BiMap (15 -> 8)
  -> ReEig
  -> LogEig and norm-preserving vech (36 values)
  -> LayerNorm -> Linear(36, 24) -> GELU -> Dropout
```

The four 24-wide embeddings receive learned band embeddings. A scalar attention score
forms a weighted band summary. The flattened band embeddings and summary are fused by a
32-wide multilayer block and projected to two residual logits.

```text
logits = anchor_logits + sigmoid(g) * residual_logits
```

The residual gate starts at 0.02. The default model has 9,678 trainable parameters and an
enforced ceiling of 50,000. The optional intent head is scalar and separate: rest examples
contribute to its binary loss but never to the left-versus-right loss.

### Why the architecture is falsifiable

If anchor-only performs as well as the full model, the learned branch is not justified. If
the residual helps only random within-session folds but not chronological or LOSO folds,
it has learned session-specific detail rather than useful invariance. If covariance
adaptation alone accounts for the gain, the appropriate conclusion may be that a simple
geometric classifier is preferable to the network.

## Strictly causal deployment adaptation

The stream contract is `predict with state_t, then update state_t+1`. Tests should fail if
the current sample can alter the state used for its own prediction.

### Covariance state

An unlabeled chronological calibration prefix seeds one log reference per band. For a live
window, the current reference aligns the covariance first. Only after prediction, a robust
EMA may update the reference in matrix-log space. The default covariance rate is 0.001;
the per-band Frobenius innovation is clipped, so a single artifact has bounded influence.

Required covariance conditions:

- `raw`: alignment and online updates disabled.
- `frozen`: calibrate once, align, and never update.
- `causal`: calibrate, align, and predict-then-update online.
- Optional `rest-only causal`: update only when the independent intent/rest signal permits
  it.

### Boundary state

The initial center is the median unlabeled calibration log-odds. At time `t`:

```text
margin_t = score_t - center_t
confidence_t = sigmoid(abs(margin_t) / temperature)
prediction_t = sign(margin_t)
```

The default commit threshold is 0.85. A low-confidence score can be treated as rest-like;
if an independent rest probability is available, it must also pass its predeclared gate.
Only then may a slow EMA update the center. The center remains clamped around its seed, so
an imbalanced or corrupted stream cannot drift without bound. Temperature, confidence
thresholds, update rates, and clamps must be selected using source/validation data only.

Required boundary conditions:

- `none`: score is used without centering or abstention.
- `frozen`: seed the median center but disable online updates.
- `causal`: seed and predict-then-update with internal rest gating.
- `causal + intent`: also require the independent intent/rest gate.
- `causal + abstain`: report coverage and selective accuracy, not accuracy only on the
  retained subset.

Model weights remain frozen throughout target streaming. The adapter retains no EEG
window, label, gradient, optimizer, or replay buffer; state size is constant in stream
length and can be serialized to JSON.

## Dataset and preprocessing lock

The primary local cohort contains exactly these 32 sessions:

```text
S01: R01 R02 R03 R04
S03: R01 R02 R03 R04
S04: R01 R02 R03 R04
S05: R01 R02 R03 R04
S06: R01 R02 R03 R04
S07: R01 R02 R03 R04
S08: R01 R02 R03 R04
S10: R05 R06 R07 R08
```

Subject 9 and subject 10 runs 1-4 are excluded before any split is formed. The main task
uses 15 channels, four overlapping 8-30 Hz bands, 125 Hz sampling, and left/right labels
0/1. Artifact rejection and covariance shrinkage are fixed in `DataConfig`, included in
the cache fingerprint, and reported with results.

The primary epoch is the 0.0-2.0 s `deployment` window. The older 0.5-2.5 s `legacy`
window includes part of the following rest phase and exists only as a named sensitivity
analysis. Never pool the two windows or select between them using test accuracy.

## Leakage-resistant evaluation

Random epoch-level K-fold validation is not evidence of session or user generalization.
EEG epochs from one recording can share slow drift, artifacts, and acquisition state, so a
complete recording is the minimum protected group.

### 1. Within-subject session holdout

For each participant, hold out each complete recording once and train on the other three.
This yields 32 outer folds. It estimates personalization across recordings, not
subject-independent performance.

Hyperparameters and early stopping require another complete source recording as an inner
validation group. If temperature or another model-specific output calibration is fitted
there, either retain that exact frozen checkpoint or obtain calibration for a refit model
through a protected source-only cross-fitting design. Never transfer a temperature to a
different from-scratch model merely because it used the same architecture.

### 2. Chronological cross-session

This is the primary local test because it resembles later-session deployment:

- Subjects 1, 3, 4, 5, 6, 7, and 8: train on runs 1-2, validate on run 3, test on run 4.
- Subject 10: train on runs 5-6, validate on run 7, test on run 8.
- Select the exact checkpoint, temperature, and all hyperparameters on the third recording
  without test data.
- Freeze that same checkpoint and evaluate the untouched fourth recording. The third
  recording remains excluded from fitting so its temperature is model-specific.

Report one result per participant before macro-averaging. A trial-weighted pooled score
may be supplementary but must not replace participant-level aggregation.

### 3. Leave-one-subject-out

Hold out all four sessions from one participant. This yields eight outer folds.
Hyperparameter selection must itself be subject-disjoint. The implemented initial runner
uses one fixed inner-validation participant, trains on the remaining six, and evaluates
the exact selected checkpoint on the outer participant. A definitive runner should
aggregate a full inner leave-one-source-subject-out sweep; a seven-source refit is valid
only if its model-specific calibration is also obtained without reusing its fitted labels.

No target-subject label may affect preprocessing, normalization, epoch selection,
thresholds, or model choice. Calling a model `generalized` is insufficient; only this
subject-disjoint protocol supports a subject-independent claim on the local cohort.

### Unlabeled target calibration

Report source-only and adapted performance separately. If an adapted condition uses a
calibration prefix:

- Define its length before examining target labels.
- Use the first chronological target windows only.
- Do not shuffle the stream.
- Exclude prefix windows from scored test metrics.
- Use no target labels for either covariance or boundary calibration.
- Continue in original acquisition order with predict-then-update adaptation.

An offline reference estimated from the entire target test set is a transductive upper
bound, not an online result. If retained for comparison, label it explicitly and never mix
it into the primary causal table.

### Validation invariants

Every run must programmatically assert:

- During selection, no row overlap between the inner training and validation groups. In
  the locked runners the validation group remains excluded from fitting; no source row
  may overlap the target calibration prefix or scored test rows.
- No row overlap between the target calibration prefix and scored target rows.
- No session overlap where session holdout is claimed.
- No subject overlap where LOSO is claimed.
- No target-label access before final scoring.
- No state update before the current prediction.
- All methods receive the same primary epoch window and scored rows.

## Metrics and inference units

For each outer fold report:

- Number of scored task windows and, when enabled, rest windows.
- Accuracy, balanced accuracy, Cohen's kappa, and ROC AUC.
- Brier score and 10-bin expected calibration error.
- Coverage and selective accuracy at the predeclared commit threshold.
- Rest false-commit rate and commands per minute when rest is evaluated.
- Batch-one synchronized latency, parameter count, and peak memory.
- Calibration-prefix length, calibration time, and adapter update counts.

Aggregate at the participant level with mean, standard deviation, median, and a
participant-resampled 95% confidence interval. Preserve seed-level results and use paired
participant-level comparisons for ablations. With only eight participants, effect sizes
and interval widths are more informative than isolated p-values. Correct families of
multiple ablation tests.

Closed-loop evaluation should report task completion, correct-command or correct-corner
rate, intervention/timeout rate, time to completion, command latency, abstention rate,
and repeated-command failures. One run per condition is not enough to estimate
reliability; repeat conditions and model the participant as the inference unit.

## Required baselines and ablations

### Baselines

All baselines must use the same folds, target-calibration policy, scored windows, and
tuning budget:

- Full tangent-space logistic anchor alone.
- Riemannian tangent-space logistic regression.
- EA-FBCSP with logistic regression.
- A small raw-EEG network such as EEGNet, with source-only and matched adaptation rules.
- TSMNet/SPDDSMBN, using the authors' implementation or a verified faithful port.
- Where practical, representative online TTA comparators such as OTTA-MI or T-TIME and
  a backpropagation-free comparator such as BFT.

If an external method cannot support the exact input or online contract, document the
difference rather than silently changing the proposed method's protocol.

### Architecture ablations

- Anchor only.
- Residual only.
- Anchor plus residual with a fixed gate.
- Anchor plus learned gate.
- Learned BiMap versus fixed projection.
- Band attention versus uniform band averaging.
- Intent/rest head on versus off.
- Parameter-matched Euclidean residual.

### Adaptation ablations

Cross covariance state (`raw`, `frozen`, `causal`) with boundary state (`none`, `frozen`,
`causal`) and report the full factorial comparison where sample size permits. Also test:

- Robust covariance clipping on/off.
- Covariance update rate and rest-only gating.
- Boundary clamp on/off.
- Internal rest gating versus independent intent gating.
- Abstention on/off with the accuracy-coverage curve.
- Calibration-prefix length sensitivity.

### Preprocessing and protocol sensitivity

- `deployment` versus `legacy` window, clearly secondary.
- Filter bank versus a single broad sensorimotor band.
- Within-session versus chronological cross-session versus LOSO.
- Source-only versus chronological unlabeled calibration and causal adaptation.
- At least several fixed random seeds for neural training.

## Positioning against prior work

The following are primary sources and define claims this project must not appropriate.

### TSMNet and SPDDSMBN

Kobler et al. introduced SPD domain-specific momentum batch normalization and TSMNet for
interpretable unsupervised domain adaptation, explicitly including multi-source,
multi-target, and online UDA scenarios:

- Paper: <https://arxiv.org/abs/2206.01323>
- Authors' code: <https://github.com/rkobler/TSMNet>

Therefore, `SPD normalization for online EEG domain adaptation` is not novel here. The
distinguishing hypothesis is a protected full tangent anchor plus a small learned residual
and an explicitly separate, rest-gated output-boundary state under a frozen-weight,
predict-then-update deployment contract. TSMNet is a required direct comparator.

### OTTA-MI

Wimpff et al. studied calibration-free online test-time adaptation for MI decoding using
alignment, adaptive batch normalization, and entropy-based updates:

- Paper: <https://arxiv.org/abs/2311.18520>

Thus, causal online MI test-time adaptation is not a new category. This package instead
tests whether bounded covariance and scalar-boundary state updates can provide a simpler,
constant-memory alternative without target-time gradients or parameter updates.

### T-TIME

Li et al. proposed an online test-time information-maximization ensemble that immediately
predicts each incoming target trial and updates classifiers using conditional entropy and
adaptive marginal regularization:

- Paper: <https://arxiv.org/abs/2412.07228>

The contrast is computational and structural, not chronological: the proposed adapter
uses one tiny frozen model and two backpropagation-free states. Any comparison must also
measure accuracy, calibration, memory, and latency rather than accuracy alone.

### BFT

Li et al. proposed Backpropagation-Free Transformations for lightweight EEG test-time
adaptation, using multiple knowledge-guided or approximate-Bayesian transformations and
a learned ranking/aggregation mechanism:

- Paper: <https://arxiv.org/abs/2601.07556>

Consequently, `backpropagation-free EEG TTA` is not a novelty claim. The proposed system
is different in mechanism: it adapts a persistent SPD reference and a bounded scalar
boundary rather than aggregating multiple transformed predictions. BFT should be compared
where its released implementation and dataset contract allow.

### Geometry-aware EEG networks and recentering

Geometry-aware covariance processing and learned congruence maps are an active, populated
area. Relevant recent examples include:

- Geometry-Aware Deep Congruence Networks for cross-subject MI:
  <https://arxiv.org/abs/2511.18940>
- RUNet, which combines Riemannian and unsupervised representation learning for
  zero-calibration cross-domain MI:
  <https://doi.org/10.1109/TBME.2026.3653024>
- A recent benchmark arguing that simple geometric test-time recentering can rival deeper
  sequence models across public MI datasets:
  <https://doi.org/10.64898/2026.07.07.736991>
- Tensor-CSPNet, an earlier filter-bank SPD deep-learning framework:
  <https://arxiv.org/abs/2202.02472>

These works mean that BiMap/ReEig/LogEig, covariance alignment, and recentering are not
individually sufficient novelty. A publishable claim requires controlled evidence that the
anchor-residual design and dual causal state improve the low-data reliability/safety/latency
tradeoff, ideally on external public multi-session data as well as the local cohort.

## Why the four local papers are weak primary comparators

The PDFs in `papers/` are useful project history but do not establish the bar for a modern
cross-domain neural decoder.

| Local paper | What it evaluates | Why it is insufficient as the main comparator |
| --- | --- | --- |
| *A Robust EEG Brain-Computer Interface Approach for Decoding Hand and Foot Motor Imagery* (PETRA 2025, <https://doi.org/10.1145/3733155.3733220>) | Six healthy participants; left/right hand and left/right foot MI; CSP/ICA and ten classical classifiers; two separately routed binary tasks; within-session splits | Small per-session test partitions, no subject-held-out test, no chronological primary result, and not a true four-way classifier. Peak accuracies from tiny folds are unstable and do not demonstrate online or cross-user robustness. |
| *Assessment of BCI Performance for Human-Robot Interaction* (PETRA 2024, <https://doi.org/10.1145/3652037.3663957>) | Ten healthy participants physically alternate relaxed and arm-curl states; FB-CSP plus classical classifiers; robot-arm mirroring | This is motor execution, not motor imagery. EEG can be confounded by EMG and movement artifacts, so its roughly 89% validation and 80% small live test are not an appropriate MI accuracy target. |
| *Comparative Study of One-Handed vs Two-Handed EEG Intent Recognition for Applications in Human-Robot Interaction* (PETRA 2024, <https://doi.org/10.1145/3652037.3663937>) | Eleven healthy, right-handed participants perform physical unimanual, coordinated, and bimanual movements; CSP plus logistic regression | Again a movement/EMG-confounded motor-execution paradigm, with a different task and acquisition contract. The reported best mean accuracy (81.5%) cannot validate MI, paralysis use, cross-session adaptation, or subject independence. |
| *A Dual-Validation Framework for Temporal Robustness Assessment in Brain-Computer Interfaces for Motor Imagery* (Technologies 2025, <https://doi.org/10.3390/technologies13120595>) | Six healthy participants, three sessions 1-2 days apart, four MI tasks handled as hand and foot binary problems; CSP and ten classifiers; within- and bidirectional cross-session tests | It is the strongest local comparator, but remains small, personalized, short-term, and classical. It uses only 10 trials per class per session, does not test a held-out person, and does not isolate causal online recentering. Bidirectional session pairs can quantify transfer but are not the same as prospective chronological deployment. |

The current study should reproduce comparable classical baselines on the exact new cohort,
then add direct modern external baselines. It should not treat healthy-volunteer performance
as evidence for people with paralysis, and it should not call a personalized later-session
test `generalized`.

## Limitations and minimum evidence before a paper claim

Known limitations:

- Only eight valid local participants and approximately 1,900 task trials before artifact
  rejection.
- Healthy-volunteer data only; no disabled or paralyzed participant data.
- One acquisition montage, sampling rate, cue design, laboratory, and hardware stack.
- Binary left/right hand imagery only in this package's main head.
- Offline cue-locked epochs do not reproduce all asynchronous BCI failure modes.
- Calibration and abstention can improve apparent accuracy by reducing coverage; both must
  be reported.
- The model's band attention is not proof of neurophysiological mechanism.
- Closed-loop simulator success does not imply safe physical-robot or clinical use.

Before a strong research claim, require all of the following:

1. Complete chronological results for all eight participants with preserved predictions.
2. Nested subject-disjoint LOSO results with no target-label tuning.
3. EA-FBCSP, tangent logistic, anchor-only, TSMNet, and at least one representative online
   TTA comparison under matched folds.
4. The full covariance-by-boundary adaptation ablation.
5. Calibration, coverage, rest false-commit, and latency results in addition to accuracy.
6. Multiple neural seeds and participant-level uncertainty/effect sizes.
7. A repeated closed-loop study that distinguishes decisions, commands, and completed
   tasks.
8. External validation on at least one public multi-session MI dataset.
9. Separate future validation with the intended disabled population before any clinical or
   accessibility-effectiveness claim.

The appropriate current language is `intended for assistive BCI research` and `evaluated
on healthy volunteers`, not `validated for disabled users`, `clinically ready`, or `safe for
autonomous control`.
