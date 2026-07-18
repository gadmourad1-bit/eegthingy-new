# GeoAdaptNet — synthesized ChatGPT/Codex conversation report for Claude

This document is a high-fidelity chronological report of the work completed after the earlier
`deepnet_chat_export.md` handoff. It is intentionally synthesized rather than a verbatim raw
conversation dump: repetitive tool logs are omitted, while decisions, commands, corrections,
artifacts, measured values, and claim boundaries are retained.

Generated: 2026-07-18, America/New_York

Companion operational prompt: `deepnet_CLAUDE_HANDOFF.md`

Earlier source context preserved unchanged:

- `deepnet_HANDOFF.md`
- `deepnet_chat_export.md`

---

## Turn 1 — user asks ChatGPT/Codex to take over the entire project

### User request

The user said they were tired of the previous model abandoning the task. They supplied the earlier
handoff and chat export, then asked ChatGPT/Codex to:

- Review the entire repository.
- Review both the prior-work `papers/` directory and current-draft `paper/` directory.
- Use the LAN Blackwell GPU described in the Markdown handoff.
- Build a novel, well-performing EEG motor-imagery neural network intended to support future work
  for disabled and paralyzed people.

### Codex interpretation

The request was treated as an end-to-end research-engineering task, not a scaffold-only task.
Before writing a model, Codex committed to auditing:

- The data and recording manifest.
- Existing evaluation leakage.
- Historical deep-learning work.
- Current paper claims.
- Prior local publications and modern literature.
- Runtime/deployment constraints.
- GPU accessibility and exact software compatibility.

The clinical motivation was retained, but the analysis immediately separated motivation from
evidence: the current recordings are from healthy volunteers and cannot support a clinical claim.

---

## Phase 1 — full project and paper audit

Codex reviewed the two supplied transcript files and the repository’s acquisition, classifier,
GUI, simulator, SBC, historical deep-learning, current paper, and recording code. It also reviewed
the four PDFs under `papers/` and the LaTeX/PDF draft under `paper/`.

### Data findings

- There are 40 FIF files in the repository.
- The valid protected cohort is 32 recordings:
  - Subjects 1, 3, 4, 5, 6, 7, and 8: runs 1–4.
  - Subject 10: replacement runs 5–8.
- Subject 9 is excluded.
- Subject 10 runs 1–4 are excluded as bad recordings.
- The fixed acquisition is 15 channels at 125 Hz.
- Four overlapping filter-bank bands cover 8–30 Hz.
- A typical valid session contains about 60 balanced motor-imagery trials before rejection.

### Window mismatch discovered

The older analysis uses a 0.5–2.5 s task window. Task annotations last about 2.1 s, so this window
contains roughly 0.4 s of the following rest period. It also does not match the intended live
two-second decoder window. Codex therefore made 0.0–2.0 s from task cue the primary `deployment`
window and retained 0.5–2.5 s only as a named `legacy` ablation.

### Leakage/optimism discovered in the existing classical evaluation

The prior classical path pools sessions and uses shuffled epoch-level stratified folds. This lets
epochs from the same recording appear in both training and validation. It also estimates the
target alignment reference from all target windows before scoring the same target set. Although
that alignment is label-free, it is transductive and uses future target data.

Consequences recorded by Codex:

- The older 91.1% personalized and 84.8% pooled EA results are legacy transductive analyses.
- The older 89.5%/77.6% Riemannian values are also not comparable with prospective chronological
  schema-v2 evaluation.
- Existing code merely warns on train/test file overlap in one path; protected evaluation should
  fail on overlap.

### Current-paper risks recorded

- The simulator study has four users and one run per user/condition cell, so strong condition
  inference is not supported.
- A condition called “generalized” may include the driving user in training; this is pooled or
  personalized transfer, not LOSO generalization.
- The reported 0.20 s response time needs a precise start/end definition given the streaming,
  consensus, and dwell rules.
- Healthy-volunteer results cannot be presented as validation in paralysis.
- Subject-held-out testing and recentering ablation were already listed as future work and needed
  to remain limitations until measured.

### Runtime risks outside the neural package

Codex documented but did not change several existing runtime issues:

- Closing the cue window can destroy it without stopping/saving an active stream.
- Live/training sample-rate mismatch only warns.
- Filtering and inference execute synchronously in an acquisition callback.
- After dwell is reached, the smoother can repeatedly expose the same nonzero command, so a robot
  consumer must edge-trigger.
- Four simulator tests fail because filename capitalization/prefix expectations differ from the
  implementation.

All audit findings were written to `deepnet/PROJECT_AUDIT.md`.

---

## Phase 2 — literature positioning and architecture decision

Codex reviewed the prior-work lane represented by SPDNet, TSMNet/SPDDSMBN, Tensor-CSPNet,
Graph-CSPNet, OTTA-MI, T-TIME, BFT, and a July 2026 preprint competitor.

### Claim correction

The earlier handoff suggested that online covariance alignment was offline-only and implied a
larger novelty gap than the literature supports. The corrected position is:

- TSMNet/SPDDSMBN already covers geometric/domain-specific normalization and online UDA cases.
- OTTA-MI and related work already study online/calibration-free MI adaptation.
- Backpropagation-free and test-time adaptation are also occupied categories.

Therefore the project must not claim first online EEG adaptation, first SPD normalization, first
geometric recentering, or first backpropagation-free adaptation.

### Chosen model hypothesis

Codex designed `GeoAdaptNet` around a strong protected baseline rather than a generic CNN:

1. Full-resolution four-band tangent vectors form a 480-feature anchor.
2. Source-only standardization conditions a direct two-logit linear anchor.
3. A learned lower-rank SPD branch maps each 15×15 covariance to 8×8, applies ReEig/LogEig,
   encodes each band, attends across bands, and produces a small residual.
4. The residual is multiplied by a gate initialized to 0.02 before it can alter the anchor.
5. A separate scalar intent/rest head provides an independent rest signal.
6. Frozen weights are deployed with bounded covariance-reference and scalar-boundary states.
7. Confidence/intent gating permits explicit abstention rather than forcing a command.

This construction was chosen because it is falsifiable. If a separately trained anchor matches the
full model, the residual is not justified. If frozen calibration matches causal updates, the
online-update claim should be dropped.

The full design and required ablations were written to `deepnet/RESEARCH_DESIGN.md`.

---

## Phase 3 — data and protocol implementation

Codex built a new isolated `deepnet/` package rather than modifying the existing live classifier.

### `config.py`

- Fixed subject/run manifest.
- Fixed channel order, bands, labels, sample rate, and named epoch windows.
- Immutable `DataConfig` with validation.

### `data.py`

- Explicit FIF resolution—no permissive discovery glob.
- Continuous filter-bank and artifact-band filtering before epoch extraction.
- Optional rest windows for the separate intent task.
- Float64 covariance construction with trace-scaled shrinkage and relative SPD floors.
- Per-window subject, run, session, sample/onset, trial, annotation, and source-file provenance.
- Content/fingerprint-aware cache.
- Full dataset contract with SHA-256 for every FIF.

The locked full contract is:

`2c93983d422c3ee478a70baf6a5f479eac69e5632ad192cb4b1460df1187c2c0`

### `protocols.py`

- Group-disjoint within-subject split support.
- Chronological cross-session split support.
- LOSO support.
- Assertions that hard-fail subject/session overlap.

### Important numerical failure rejected during development

An early absolute eigenvalue floor near `1e-5` was incompatible with real EEG covariance scale,
which is roughly around `1e-10 V²`. It collapsed meaningful covariance structure. Codex replaced
absolute spectral handling with scale-relative normalization/floors and regression tests.

---

## Phase 4 — neural, causal-adaptation, baseline, and reporting implementation

### `spd.py`

Implemented:

- Symmetrization and scale-relative eigendecomposition.
- Stable spectral clip, square root, inverse square root, exponential, and logarithm.
- Divided-difference spectral backward to avoid raw eigenvector-gradient singularity at repeated
  eigenvalues.
- Norm-preserving upper-triangle vectorization.
- Semi-orthogonal `BiMap` with QR parameterization/retraction.
- ReEig and LogEig layers.
- Log-Euclidean momentum reference normalization.

### `model.py`

Implemented the anchor-preserving `GeoAdaptNet` described above. The benchmark configuration has
9,678 trainable parameters and an enforced ceiling below 50,000.

### `engine.py`

Implemented:

- Rest-aware covariance dataset.
- Deterministic seeding.
- AdamW training and cosine schedule.
- Protected validation/early stopping.
- Fixed-epoch training helper for safe uses.
- Branch-specific probability diagnostics.
- Synchronized batch-one latency.
- Safe `weights_only=True` checkpoint loading.
- Later, Mac-safe CUDA/MPS/CPU resolution.

### `recenter.py`

Implemented two serializable, label-free causal states:

- Robust log-Euclidean covariance-reference calibration/update.
- Scalar boundary calibration/update around the median calibration log-odds.

Every stream step obeys predict-then-update ordering. Updates can be disabled, limited to rest-like
windows, weighted by rest probability, and bounded by innovation/clamp rules.

### `baselines.py`

Implemented matched:

- Euclidean-aligned filter-bank CSP with shrinkage LDA.
- Per-band Riemannian alignment/tangent features with standardized logistic regression.

Later, the imports were made compatible with both pyRiemann 0.11 (`pyriemann.utils`) and 0.12
(`pyriemann.geometry`).

### Experiment/report infrastructure

Implemented:

- `experiment.py`: chronological schema-v2 runner.
- `loso.py`: strict neural 6/1/1 subject-held-out runner.
- `loso_baselines.py`: matched classical 6/1/1 runner.
- `deployment.py`: frozen checkpoint-backed streaming wrapper.
- `metrics.py`: task, calibration, coverage, selective, and rest metrics.
- `report.py`: full trace validation and participant bootstrap summaries.
- `merge_results.py`: compatibility-checked result combination.

Schema-v2 JSON stores every scored/rest label, probability, intent score, commit, boundary center,
update decision, and provenance row. Reports are reconstructed from those traces and fail if stored
aggregates disagree.

---

## Phase 5 — Blackwell GPU execution

Codex used the existing Mac SSH alias `gpu` to operate the Ubuntu workstation at
`~/Desktop/eegthingy`.

Verified workstation facts:

- Host `user-desktop-ryzen9`
- RTX 5070, 12 GB, Blackwell compute capability `(12, 0)`
- Driver 595.71.05
- PyTorch 2.11.0+cu128
- CUDA kernel/matmul execution succeeded

The exact CUDA 12.8 requirements were recorded in `deepnet/requirements-cu128.txt` because a
default PyPI TorchAudio package can pull a different CUDA runtime and fail on
`libcudart.so.13`.

The GPU suite reached 68 passing `deepnet` tests before later Mac-only portability tests were
added.

---

## Phase 6 — initial experiment, audit intervention, and schema-v2 correction

### Initial result

An early chronological result produced a headline around 86.40%. Codex did not accept it at face
value. A final code/protocol audit found four issues:

1. Model selection used a 180-epoch cosine horizon, but the subsequent refit compressed the cosine
   into the selected epoch count.
2. Temperature scaling was fitted on the selection model and transferred to a newly refit model
   with a different logit scale.
3. Result/checkpoint artifacts did not fully encode raw/preprocessing identity.
4. Per-window traces were incomplete, preventing independent reconstruction of every aggregate.

### Correction

Codex changed both chronological and LOSO logic to retain the exact protected-validation
checkpoint. The validation session/participant remains excluded from model fitting, and its
temperature applies only to the same model that produced the validation logits. Full data
contracts and traces were added.

The 86.40% schema-v1 result was marked superseded in both `README.md` and `RESULTS.md`. It must not
be cited.

---

## Phase 7 — final chronological schema-v2 benchmark

### Protocol

For every participant:

- Fit recordings 1–2.
- Use recording 3 to select the exact checkpoint, temperature, and target-median blend.
- Test the untouched fourth recording after an unlabeled prefix of 20 chronological task/rest
  windows.
- Exclude calibration-prefix events from scoring.
- Process future windows in order with no target labels.

GeoAdaptNet used seeds 7, 17, and 27. Classical estimators were deterministic and run once per
participant.

### Final metrics

| Model | Balanced accuracy | ROC AUC | Brier | Coverage | Selective accuracy | Rest false commits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoAdaptNet | 84.86 ± 12.67% | 93.32 ± 8.04% | 9.99 ± 7.60% | 41.01 ± 17.65% | 96.82 ± 5.17% | 15.71 ± 8.89% |
| Riemannian LR | 84.24 ± 11.75% | 94.25 ± 7.66% | 9.65 ± 6.97% | 67.16 ± 22.90% | 96.19 ± 5.09% | 47.95 ± 20.19% |
| EA-FBCSP | 87.53 ± 13.40% | 93.18 ± 10.13% | 9.21 ± 8.46% | 69.44 ± 25.19% | 91.72 ± 11.38% | 50.85 ± 18.45% |

Paired participant bootstrap comparisons:

- GeoAdaptNet minus Riemannian balanced accuracy: +0.63 points, CI −1.01 to +2.42.
- GeoAdaptNet minus FBCSP: −2.67 points, CI −5.83 to +0.39.
- GeoAdaptNet minus Riemannian rest false commits: −32.23 points,
  CI −46.50 to −20.23.
- GeoAdaptNet minus FBCSP rest false commits: −35.14 points,
  CI −47.58 to −20.55.

Interpretation retained by Codex:

- Task accuracy is competitive but not superior.
- The separate intent gate plus abstention greatly reduces annotated-rest commits.
- That safety behavior comes at materially lower coverage.
- Baselines lack a separate intent head, so the false-commit comparison is a complete-decoder
  comparison, not proof that the left/right neural classifier is inherently safer.

Latency and size:

- 9,678 parameters.
- Mean synchronized batch-one network latency 4.54 ms on RTX 5070 after covariance construction.
- Range 4.36–5.71 ms.

Artifacts:

- `deepnet/results/chronological_v2_final_3seeds.json`
- `deepnet/results/chronological_v2_final_3seeds.md`
- 24 checkpoints under `deepnet/checkpoints/chronological_v2_final_3seeds/`

### Diagnostic findings that limited the claim

- Full-output balanced accuracy: 84.860%.
- Anchor-logit diagnostic: 84.693%.
- Difference: only +0.17 points.
- Residual gate mean: 0.02145; range 0.02089–0.02211.
- Frozen target updates: 84.853%.
- Causal target updates: 84.860%.

Codex explicitly concluded that these are not evidence that the learned residual or ongoing causal
updates help. The current safety behavior is better attributed to calibration, intent gating, and
abstention until matched ablations say otherwise.

---

## Phase 8 — strict initial LOSO benchmark

### Protocol

For each of eight outer participants:

- Exclude all four outer recordings from every fitting/selection operation.
- Choose the next participant in fixed cohort order as the protected inner validation/calibration
  participant.
- Fit on the remaining six participants.
- Select checkpoint, temperature, and blend only on the inner participant.
- Retain that exact checkpoint.
- For each validation/outer recording separately, calibrate on the first 10 task cues plus
  intervening rest, exclude the prefix, and reset state at the recording boundary.

The current neural LOSO uses seed 7 only.

### Final metrics

| Model | Balanced accuracy | ROC AUC | Brier | Coverage | Selective accuracy | Rest false commits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoAdaptNet | 83.54 ± 9.36% | 90.63 ± 7.95% | 12.88 ± 5.82% | 19.22 ± 20.56% | 95.85 ± 7.02% | 9.68 ± 9.04% |
| Riemannian LR | 81.61 ± 10.52% | 88.26 ± 9.88% | 14.59 ± 8.04% | 79.36 ± 10.02% | 86.31 ± 9.57% | 67.45 ± 10.56% |
| EA-FBCSP | 85.67 ± 9.15% | 92.91 ± 6.80% | 10.34 ± 5.91% | 63.24 ± 24.16% | 94.44 ± 5.86% | 43.84 ± 22.87% |

Paired comparisons:

- GeoAdaptNet minus Riemannian balanced accuracy: +1.93 points,
  CI −0.02 to +4.42.
- GeoAdaptNet minus FBCSP: −2.13 points, CI −4.80 to +0.81.
- GeoAdaptNet minus Riemannian rest false commits: −57.77 points,
  CI −62.64 to −53.08.
- GeoAdaptNet minus FBCSP rest false commits: −34.17 points,
  CI −45.57 to −21.91.

Per-participant GeoAdaptNet balanced accuracy:

| Participant | Balanced accuracy |
| --- | ---: |
| 1 | 74.66% |
| 3 | 88.44% |
| 4 | 92.51% |
| 5 | 94.04% |
| 6 | 93.35% |
| 7 | 74.09% |
| 8 | 73.58% |
| 10 | 77.63% |

Interpretation:

- Again there is no supported accuracy-superiority claim.
- The high-precision operating point has extremely low and variable coverage; participant 7 had
  zero committed task windows at the fixed threshold in the retained traces.
- One deterministic inner participant and one neural seed make this preliminary, not definitive
  subject-independent evidence.

Artifacts:

- `deepnet/results/nested_loso_v2_seed7.json`
- `deepnet/results/nested_loso_v2_baselines.json`
- `deepnet/results/nested_loso_v2_combined.json`
- `deepnet/results/nested_loso_v2_combined.md`
- 8 checkpoints under `deepnet/checkpoints/nested_loso_v2_seed7_nested_loso/`

---

## Phase 9 — deployment and artifact validation

Codex validated all 24 chronological and 8 current LOSO schema-v2 checkpoints using the safe
weights-only loader. Model type, schema version, contract, and 9,678-parameter configuration were
checked.

A real held-out covariance was passed through the checkpoint-backed GPU deployment wrapper. The
wrapper validated the preprocessing contract, froze weights, calibrated causal state, and emitted
a finite decision. The recorded example was:

```text
prediction = right
p_left = 0.14179
p_right = 0.85821
intent = 0.58747
rest = 0.41253
confidence = 0.85821
committed = true
margin = 1.8005
```

This is a functional deployment check, not an additional performance result.

The combined `deepnet` and simulator test run produced 109 passes and four failures. All four
failures were the previously identified simulator filename-case mismatch and were unrelated to
the new neural package.

---

## Turn 2 — user asks whether the entire review is complete

### User question

The user asked whether Codex had reviewed everything and had everything needed.

### Codex answer

Codex answered yes and summarized:

- Full repository/paper/prior-work audit complete.
- Working GeoAdaptNet package implemented.
- GPU experiments complete.
- Corrected schema-v2 chronological and initial LOSO results available.
- No supported accuracy-superiority claim.
- Stronger rest false-commit behavior at lower coverage.
- Current package and reports remain uncommitted.

Codex also stated the highest-value next work: definitive nested LOSO, external data, matched
modern comparators, independent ablations, continuous replay, and closed-loop validation.

---

## Turn 3 — user asks whether it runs on the MacBook

### User request

The user asked: “can this run on the macbook? test it locally”.

### Environment audit

Codex identified:

- Host: Apple-silicon M5 Pro.
- macOS 26.5.2, arm64.
- Repository `.venv`: Python 3.12.13.
- PyTorch 2.12.1.
- MNE 1.12.1, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.8.0,
  pyRiemann 0.11.
- Local `.venv` initially lacked pytest.
- MNE’s normal home-config lock was blocked by the Codex sandbox; testing used an isolated
  temporary MNE/cache configuration without rewriting the user’s home.

### Compatibility issue 1 — pyRiemann import layout

The GPU environment uses pyRiemann 0.12 paths under `pyriemann.geometry`, while the Mac has
pyRiemann 0.11 paths under `pyriemann.utils`. Codex added a dual-version import fallback in
`deepnet/baselines.py`.

### Compatibility issue 2 — MPS spectral coverage

Outside the sandbox, PyTorch reported:

```text
mps_built = true
mps_available = true
basic Metal matmul = successful
```

However, the actual saved model failed on MPS because `aten::_linalg_eigh.eigenvalues` is not
implemented for MPS. Codex then verified `PYTORCH_ENABLE_MPS_FALLBACK=1`; the full checkpoint ran,
but each eigendecomposition moved back to CPU.

Measured batch-one network latency:

- Native CPU: 1.00 ms.
- MPS with CPU fallback: 19.53 ms.

Codex therefore changed `resolve_device("auto")` to probe the required eigendecomposition and use
CPU when native MPS cannot execute it. Explicit MPS now produces a clear explanatory error unless
fallback was enabled before Python startup. LOSO’s default device was changed from hardcoded CUDA
to `auto`, and experiment environment metadata now records CUDA/MPS availability.

### Local test results

After compatibility changes:

- 71/71 `deepnet` tests passed in 1.87 s.
- Python compilation passed.
- `git diff --check` passed.

Real-data inference smoke:

- Loaded S01-R04 directly from FIF with the deployment/rest preprocessing contract.
- Produced 118 windows of shape `(118, 4, 15, 15)`, float32.
- Preprocessing time: 0.41 s.
- Calibrated a saved chronological checkpoint on 20 windows.
- Streamed the remaining 98 windows.
- Complete decoder time: 1.33 ms per window.
- Every probability was finite and left/right probabilities summed to one.

The ad hoc smoke calibration did not exactly reproduce the locked benchmark prefix protocol, so
Codex deliberately did not present its task accuracy as a research result.

Real-data training smoke:

- 119 S01-R01 windows for training.
- 118 S01-R02 windows for validation.
- Two epochs on CPU.
- 0.49 s total, about 0.25 s/epoch.
- Finite training/validation history.

Codex added:

- `deepnet/requirements-macos.txt`
- Mac install/testing notes and measured values to `deepnet/README.md`
- Device capability tests to `deepnet/tests/test_engine.py`

Conclusion given to the user:

- The MacBook is suitable for development, live deployment, and smaller training runs.
- The Blackwell workstation remains preferable for complete multi-seed benchmarks.

---

## Turn 4 — user requests Claude handoff and chat reports

### User request

The user asked for a handoff report and chat report for Claude, analogous to the earlier reports
provided to ChatGPT/Codex.

### Action

Codex preserved the original exports and created two new files:

- `deepnet_CLAUDE_HANDOFF.md` — an actionable takeover prompt with exact state, results,
  commands, guardrails, and next priorities.
- `deepnet_CLAUDE_CHAT_REPORT.md` — this chronological report.

The originals were not overwritten because they are source context owned by the user.

---

## Current source-of-truth artifact index

Read in this order:

1. `deepnet_CLAUDE_HANDOFF.md`
2. `deepnet/README.md`
3. `deepnet/RESULTS.md`
4. `deepnet/RESEARCH_DESIGN.md`
5. `deepnet/PROJECT_AUDIT.md`
6. `deepnet/results/chronological_v2_final_3seeds.json`
7. `deepnet/results/nested_loso_v2_combined.json`

Current checkpoints:

- 24 files in `deepnet/checkpoints/chronological_v2_final_3seeds/`
- 8 files in `deepnet/checkpoints/nested_loso_v2_seed7_nested_loso/`

Important ignored artifacts:

- `deepnet/cache/`
- `deepnet/checkpoints/`
- `deepnet/results/`
- `deepnet/**/*.pt`

The results and checkpoints exist locally but are deliberately not tracked by the current
`.gitignore` change; `deepnet/cache/` does not currently exist. Do not assume another clone
contains any ignored artifact. At audit time, results comprised 18 files/about 41 MiB and
checkpoints comprised 40 `.pt` files/about 2.6 MB, including eight superseded schema-v1 files.

---

## Current worktree and reproducibility risk

At report generation:

```text
branch: main
HEAD: 284085895521ef642f4dcf7b26cf86eb59526599
tracked modification: .gitignore
untracked: deepnet/ and the Markdown handoff/export reports
staged: nothing
committed/pushed: nothing from this work
```

This is the largest immediate operational risk: benchmark JSON records the base commit and a dirty
worktree, but the complete new source is not captured by Git because `deepnet/` is untracked. The
result/checkpoint artifacts are also ignored. Before another agent performs broad refactoring or
cleanup, the user should decide how to version the code and archive large results.

Do not “fix” this by blindly staging everything: the original transcript exports are user-owned,
and result/checkpoint inclusion requires an explicit storage decision.

---

## Decisions that must survive the handoff

1. **Deployment window is primary.** Use 0.0–2.0 s. Label 0.5–2.5 s as legacy.
2. **Exact manifest only.** Never rediscover subject 9 or S10 bad runs with a glob.
3. **No outer-label tuning.** Calibration prefixes are unlabeled and excluded from scoring.
4. **Exact selected checkpoint.** Do not transfer temperature to a different refit model.
5. **Per-window traces are mandatory.** Aggregate-only results are insufficient.
6. **Participant is the inferential unit.** Aggregate seeds within participant first.
7. **Accuracy is competitive, not superior.** The current value is the safety/availability tradeoff.
8. **Residual/causal benefit is unproven.** Run independent matched ablations.
9. **Healthy cohort is not clinical evidence.** Do not recruit or deploy based on these results.
10. **Mac uses CPU by default.** MPS fallback is functional but slower for this SPD model.

---

## Recommended continuation

The most valuable scientific continuation is:

1. Full inner-LOSO selection across participants with multiple neural seeds.
2. Independently trained anchor-only versus full residual comparison.
3. Raw/frozen/causal covariance and boundary-state factorial ablation.
4. Intent-gate/commit-threshold coverage-risk curves.
5. TSMNet and matched modern online-adaptation comparators.
6. External public multi-session MI replication.
7. Continuous asynchronous replay and a validated raw-board-to-covariance bridge.
8. Repeated closed-loop simulation.
9. Only then, a carefully designed study with the intended disabled population.

The correct negative result remains scientifically useful: if the simple tangent anchor with
prefix calibration matches the full network, simplify the deployment and report that outcome.
