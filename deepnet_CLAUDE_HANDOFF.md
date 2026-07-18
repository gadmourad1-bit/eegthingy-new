# Claude handoff — GeoAdaptNet EEG project

Copy this entire document into Claude together with the repository. This is a handoff of a
working, audited research package—not a request to scaffold a new project.

The earlier `deepnet_HANDOFF.md` is historical and operationally obsolete. It describes a future
scaffold, uses the old `{1, 2}` label convention and 0.5–2.5 s window, overstates parts of the
novelty gap, and requests files/baselines that are not the implemented package. Preserve it as
history, but follow this handoff and the current `deepnet/` documentation instead.

## 1. Your role and immediate operating rules

You are taking over as a senior ML research engineer and scientific reviewer for an EEG
motor-imagery project intended to support future assistive control research. Continue from the
implemented state described below.

Before changing anything:

1. Read `deepnet/README.md`, `deepnet/RESULTS.md`, `deepnet/RESEARCH_DESIGN.md`, and
   `deepnet/PROJECT_AUDIT.md` completely.
2. Inspect `git status --short --branch` and preserve every unrelated or user-owned change.
3. Do not delete, overwrite, stage, or commit `deepnet_HANDOFF.md`,
   `deepnet_chat_export.md`, or the two Claude reports unless the user explicitly asks.
4. Treat schema-v2 result JSON files as the source of truth. Never revive or cite the
   superseded schema-v1 86.40% result.
5. Do not claim clinical validation, deployment readiness, accuracy superiority, or a proven
   benefit from the learned residual/causal updates. The current participants are healthy
   volunteers.
6. Never tune on the outer test labels or use future target windows. The deployment contract is
   strictly `predict with state_t`, then optionally update state for `t+1`.

## 2. Repository and machine state

- Mac repository: `/Users/admin/Documents/GitHub/eegthingy`
- Branch: `main`
- Base HEAD when this handoff was generated:
  `284085895521ef642f4dcf7b26cf86eb59526599`
- HEAD subject/date: `paper`, 2026-07-17 21:52:25 -04:00
- Current worktree is intentionally dirty:
  - `.gitignore` is modified to ignore deepnet caches, checkpoints, results, `.pt` files,
    and `.pytest_cache`.
  - `deepnet/` is untracked as a package at this base revision.
  - The earlier transcript/handoff exports are untracked user files.
- No changes have been staged, committed, pushed, or placed in a PR.
- Do not use `git reset --hard`, `git clean`, or checkout-based destructive restoration.

This is an immediate reproducibility risk: the schema-v2 artifacts record base commit `2840858`
and `repository.dirty=true`, but that commit does not contain the untracked implementation that
produced them. Preserve/version the exact source before cleanup or broad refactoring.

LAN GPU workstation:

- SSH alias from the Mac: `gpu`
- Host: `user-desktop-ryzen9`
- Checkout: `~/Desktop/eegthingy`
- OS: Ubuntu 24.04
- CPU: Ryzen 9 9900X
- GPU: NVIDIA GeForce RTX 5070, 12 GB, Blackwell `sm_120`
- Driver: 595.71.05
- Verified CUDA environment: PyTorch 2.11.0+cu128, TorchAudio 2.11.0+cu128,
  Braindecode 1.6.1, MNE 1.12.1, pyRiemann 0.12, scikit-learn 1.9.0
- Exact GPU pins: `deepnet/requirements-cu128.txt`

Mac execution environment:

- Apple-silicon M5 Pro, macOS 26.5.2, arm64
- Repository `.venv`: Python 3.12.13, PyTorch 2.12.1
- Exact tested Mac pins: `deepnet/requirements-macos.txt`
- Native MPS is visible, but PyTorch 2.12.1 does not implement the
  `torch.linalg.eigh` Metal kernel required by this SPD network.
- `deepnet.engine.resolve_device("auto")` now probes the required spectral operation and
  selects CPU on this Mac. Explicit MPS gives a clear error unless
  `PYTORCH_ENABLE_MPS_FALLBACK=1` was set before Python started.
- The MPS fallback was verified but is substantially slower than CPU for this model.

The final Mac compatibility changes may not yet be synchronized to the GPU checkout. Compare or
sync the two `deepnet/` trees before treating a new workstation run as authoritative.

## 3. Research objective and honest novelty boundary

The task is binary left-versus-right hand motor-imagery classification, with a separate
intent-versus-rest signal for safe command gating. The long-term purpose is assistive control for
disabled or paralyzed users, but the present dataset contains only healthy volunteers.

The defensible research contribution is the complete evaluated system:

1. A strong, full-resolution tangent-space linear anchor.
2. A compact learned filter-bank SPD residual protected by a near-zero initialized gate.
3. Frozen network weights plus two bounded, label-free, constant-memory causal states:
   covariance-manifold recentering and scalar output-boundary recentering.
4. A separate intent/rest head and explicit abstention for command safety.
5. Leakage-resistant chronological and subject-held-out protocols with per-window audit traces,
   participant aggregation, calibration metrics, coverage, false rest commits, and latency.

Do **not** call this the first SPD EEG network, first online EEG adaptation system, first
backpropagation-free adaptation method, or first covariance recentering method. Prior work already
occupies those categories. The provisional novelty is the protected anchor/residual construction
plus dual bounded causal state, separate intent gating, abstention, and rigorous deployment audit.
It remains provisional because the learned residual and online-state benefit have not yet been
established by matched ablations.

Primary positioning sources already reviewed:

- TSMNet/SPDDSMBN: <https://arxiv.org/abs/2206.01323>
- OTTA-MI: <https://arxiv.org/abs/2311.18520>
- T-TIME: <https://arxiv.org/abs/2412.07228>
- BFT: <https://arxiv.org/abs/2601.07556>
- 2026 direct preprint competitor:
  <https://www.biorxiv.org/content/10.64898/2026.07.07.736991v1>

TSMNet and current online/test-time-adaptation methods are required comparators before making a
strong novelty claim.

## 4. Project review already completed

The prior agent reviewed:

- Both earlier exported Markdown transcripts.
- Acquisition, GUI, classical classifier, smoothing, SBC, and simulator code.
- The historical `deep-learning` branch and its Braindecode wrappers.
- All 40 FIF recordings.
- The current paper draft under `paper/`.
- All four prior PDFs under `papers/`:
  - `ARobust EEGBrain-Computer Interface.pdf`
  - `Assessment of BCI Performance.pdf`
  - `Comparative Study.pdf`
  - `technologies-13-00595-v2.pdf`

The detailed findings are in `deepnet/PROJECT_AUDIT.md`. Important conclusions:

- The old classical evaluation pools sessions and uses shuffled epoch-level CV, so epochs from the
  same recording can occur on both sides of a fold.
- It also estimates target recentering from all target windows before scoring them. This is
  label-free but transductive, not prospective.
- Older paper figures such as 91.1%/84.8% EA and 89.5%/77.6% Riemannian must be described as legacy
  transductive analyses and cannot be compared directly with the protected schema-v2 results.
- The current paper was intentionally not rewritten during the neural build. It still requires a
  careful schema-v2 revision before submission.

## 5. Fixed data contract

The repository contains 40 FIF files, but the protected cohort is exactly 32 recordings:

| Subject | Valid runs |
| --- | --- |
| 1 | 1, 2, 3, 4 |
| 3 | 1, 2, 3, 4 |
| 4 | 1, 2, 3, 4 |
| 5 | 1, 2, 3, 4 |
| 6 | 1, 2, 3, 4 |
| 7 | 1, 2, 3, 4 |
| 8 | 1, 2, 3, 4 |
| 10 | 5, 6, 7, 8 |

Subject 9 is excluded. Subject 10 runs 1–4 are bad recordings and excluded.

Primary preprocessing:

- Sampling rate: 125 Hz
- Channel order: `Cz, Pz, C3, C4, T5, T6, Fz, F7, F8, F3, F4, T3, T4, P3, P4`
- Bands: 8–12, 11–15, 14–20, 20–30 Hz
- Primary deployment epoch: 0.0–2.0 s from task cue, 251 inclusive samples
- Legacy-only epoch: 0.5–2.5 s; it leaks roughly 0.4 s of the following rest interval
- Neural tensor per window: four 15×15 SPD covariances, shape `(4, 15, 15)`
- Main labels: left `0`, right `1`; optional rest `-1` only trains the intent head
- Covariances are computed in float64, symmetrized, shrunk toward a trace-scaled identity,
  scale-floored, then emitted at the configured precision.

Locked data-contract SHA-256:

`2c93983d422c3ee478a70baf6a5f479eac69e5632ad192cb4b1460df1187c2c0`

The schema-v2 contract includes preprocessing settings, exact manifest, file sizes/timestamps, and
SHA-256 for every source FIF. Do not silently regenerate or resume across a contract mismatch.

## 6. Implemented model and deployment state

`GeoAdaptNet` is implemented and has 9,678 trainable parameters in the benchmark configuration.

Anchor path:

```text
4 × 15×15 aligned SPD covariance
  -> matrix log
  -> norm-preserving upper-triangle vectorization (4 × 120 = 480)
  -> source-only running standardization
  -> Linear(480, 2)
```

Residual path, independently per band:

```text
15×15 SPD
  -> semi-orthogonal BiMap(15 -> 8)
  -> ReEig
  -> LogEig
  -> norm-preserving 36-value vector
  -> LayerNorm -> Linear(36, 24) -> GELU -> Dropout
```

Four residual band embeddings receive learned attention, a 32-wide fusion block, and two residual
logits. Final logits are:

```text
anchor_logits + sigmoid(residual_gate_logit) * residual_logits
```

The gate initializes at 0.02. A separate scalar intent head estimates task versus rest and never
turns the main problem into an artificial three-class task.

Deployment uses `deepnet.deployment.GeoAdaptDecoder`:

- Checkpoints load with `torch.load(..., weights_only=True)`.
- Model weights are frozen.
- A chronological unlabeled prefix calibrates covariance and boundary state.
- Each stream window is predicted before it may update either state.
- Covariance updates are bounded, robust, and normally rest-gated.
- Boundary updates are slow, clamped, rest-like-only, and label-free.
- A command is emitted only when left/right confidence exceeds the fixed threshold and the
  independent intent gate permits it.
- The wrapper requires a precomputed `(4, 15, 15)` covariance. Raw channel selection, causal
  filtering, scheduling, and robot edge-triggering remain outside the wrapper.

## 7. Package map

| File | Responsibility |
| --- | --- |
| `deepnet/config.py` | Immutable cohort and preprocessing contract |
| `deepnet/data.py` | FIF loading, continuous filtering, cache, provenance, SPD covariance |
| `deepnet/protocols.py` | Group-disjoint split generation and leakage assertions |
| `deepnet/spd.py` | Stable differentiable SPD algebra and manifold layers |
| `deepnet/model.py` | Anchor-preserving GeoAdaptNet |
| `deepnet/engine.py` | Training, inference, checkpoint loading, latency, device selection |
| `deepnet/recenter.py` | Causal covariance and scalar-boundary state |
| `deepnet/baselines.py` | EA-FBCSP and Riemannian tangent logistic baselines |
| `deepnet/experiment.py` | Chronological schema-v2 benchmark runner |
| `deepnet/loso.py` | Strict neural 6-train/1-validation/1-target LOSO runner |
| `deepnet/loso_baselines.py` | Matched LOSO classical runners |
| `deepnet/deployment.py` | Checkpoint-backed streaming decoder |
| `deepnet/metrics.py` | Accuracy, calibration, coverage, selective metrics |
| `deepnet/report.py` | Recompute/validate all metrics from retained traces |
| `deepnet/merge_results.py` | Compatibility-checked result merge |
| `deepnet/tests/` | Numerical, leakage, causality, report, and deployment tests |

## 8. Protocol correction history—critical

Early schema-v1 runs are invalid for citation because the audit found:

1. Selection trained with a cosine horizon of 180 epochs, but the refit compressed the cosine
   schedule into the selected epoch count, so it was not the validated optimizer trajectory.
2. Temperature was fitted on the selected model and transferred to a fresh refit model with a
   different logit scale.
3. Artifacts did not lock all raw/preprocessing identity.
4. Per-window predictions were not retained for independent metric reconstruction.

Schema v2 fixes these issues by retaining the exact protected-validation checkpoint instead of
refitting it, leaving the validation group excluded from fitting, storing the full data contract,
and retaining all scored/rest traces. `deepnet.report` recomputes metrics and hard-fails if an
aggregate does not match its traces.

The earlier 86.40% schema-v1 result and checkpoints are explicitly superseded. Do not cite them,
merge them, or use them as a baseline.

Only the schema-v2 artifact set named in the next section is authoritative. Every other JSON under
`deepnet/results/` is pilot/schema-v1 history; despite its filename,
`chronological_seed7_v2.json` declares schema version 1. The eight checkpoints under
`deepnet/checkpoints/chronological_seed7_artifacts/` safely deserialize but remain scientifically
superseded. Safe loading is a security/integrity check, not protocol validation.

## 9. Current verified results

### 9.1 Chronological cross-session, schema v2

Protocol per participant:

- Fit on recordings 1–2.
- Select/calibrate the exact checkpoint on recording 3.
- Freeze it and test recording 4 after an unlabeled chronological prefix of 20 task/rest windows.
- Exclude prefix windows from scoring.
- Neural seeds: 7, 17, 27; aggregate seeds within participant before cohort statistics.

Source of truth:

- `deepnet/results/chronological_v2_final_3seeds.json` (about 6.6 MB)
- `deepnet/results/chronological_v2_final_3seeds.md`
- 24 safe-loadable checkpoints in
  `deepnet/checkpoints/chronological_v2_final_3seeds/`

Artifact SHA-256 for transfer verification:

`b8ce6cfdf9915b410a6ab9e6e2eee9b7bec3fd6617072185dc11512bcb15cca0`

| Model | Balanced accuracy | Coverage | Selective accuracy | Rest false commits |
| --- | ---: | ---: | ---: | ---: |
| GeoAdaptNet | 84.86 ± 12.67% | 41.01 ± 17.65% | 96.82 ± 5.17% | 15.71 ± 8.89% |
| Riemannian LR | 84.24 ± 11.75% | 67.16 ± 22.90% | 96.19 ± 5.09% | 47.95 ± 20.19% |
| EA-FBCSP | 87.53 ± 13.40% | 69.44 ± 25.19% | 91.72 ± 11.38% | 50.85 ± 18.45% |

Accuracy comparisons for GeoAdaptNet:

- Versus Riemannian LR: +0.63 percentage points, participant bootstrap 95% CI
  −1.01 to +2.42.
- Versus EA-FBCSP: −2.67 points, CI −5.83 to +0.39.
- Therefore there is no accuracy-superiority result.

Rest false-commit comparisons:

- Versus Riemannian LR: −32.23 points, CI −46.50 to −20.23.
- Versus EA-FBCSP: −35.14 points, CI −47.58 to −20.55.
- This is a complete-decoder safety/availability comparison: baselines do not have the separate
  intent head, and GeoAdaptNet pays for lower false commits with lower coverage.

Other locked facts:

- ROC AUC: GeoAdaptNet 93.32%, Riemannian 94.25%, FBCSP 93.18%.
- Mean synchronized batch-one network latency: 4.54 ms on RTX 5070 after covariance construction,
  range 4.36–5.71 ms.
- Full-output balanced accuracy 84.860% versus anchor-logit diagnostic 84.693%, only +0.17 points.
- Residual gate mean 0.02145, range 0.02089–0.02211.
- Frozen target-state updates: 84.853% versus causal updates 84.860%.
- The post-training anchor diagnostic is not an independently trained ablation, and the nearly
  identical frozen/causal score is no evidence of an online-update benefit.

### 9.2 Initial strict subject-held-out LOSO, schema v2

Protocol per outer participant:

- Withhold all four recordings from the outer participant.
- Use the next participant in fixed cohort order as protected inner validation/calibration.
- Fit only on the remaining six participants.
- Retain the exact selected model; never transfer calibration to a refit.
- For every validation/outer recording, calibrate on the first 10 task cues plus intervening rest,
  reset state at each recording, and exclude prefix events from scoring.
- Neural result currently has only seed 7.

Source of truth:

- `deepnet/results/nested_loso_v2_seed7.json`
- `deepnet/results/nested_loso_v2_seed7.md`
- `deepnet/results/nested_loso_v2_baselines.json`
- `deepnet/results/nested_loso_v2_baselines.md`
- `deepnet/results/nested_loso_v2_combined.json` (about 17 MB)
- `deepnet/results/nested_loso_v2_combined.md`
- 8 safe-loadable checkpoints in
  `deepnet/checkpoints/nested_loso_v2_seed7_nested_loso/`

Artifact SHA-256 values, in the same neural/baseline/combined order:

- `5116f27cf644c19fa55b5b22d76cb4c10efac7c04f103174113e1356b7b0421c`
- `29405caf83a32603492ab122cb45057dedd635c32077f168ad4640c0f664c592`
- `67a4c7fc4860976144110f1921a3790e01c629997bacb333bc86d91de8a97576`

| Model | Balanced accuracy | Coverage | Selective accuracy | Rest false commits |
| --- | ---: | ---: | ---: | ---: |
| GeoAdaptNet | 83.54 ± 9.36% | 19.22 ± 20.56% | 95.85 ± 7.02% | 9.68 ± 9.04% |
| Riemannian LR | 81.61 ± 10.52% | 79.36 ± 10.02% | 86.31 ± 9.57% | 67.45 ± 10.56% |
| EA-FBCSP | 85.67 ± 9.15% | 63.24 ± 24.16% | 94.44 ± 5.86% | 43.84 ± 22.87% |

Accuracy comparisons for GeoAdaptNet:

- Versus Riemannian LR: +1.93 points, CI −0.02 to +4.42.
- Versus EA-FBCSP: −2.13 points, CI −4.80 to +0.81.
- Again, this is competitive accuracy—not superiority.

Rest false-commit comparisons:

- Versus Riemannian LR: −57.77 points, CI −62.64 to −53.08.
- Versus EA-FBCSP: −34.17 points, CI −45.57 to −21.91.
- Coverage is only 19.22% and varies substantially between participants.

Per-participant GeoAdaptNet balanced accuracy:

- S1 74.66%, S3 88.44%, S4 92.51%, S5 94.04%
- S6 93.35%, S7 74.09%, S8 73.58%, S10 77.63%

This LOSO estimate is preliminary because it uses one deterministic inner participant and one
neural seed. Do not call it a definitive subject-independent result.

## 10. Validation already performed

GPU workstation before the final Mac-only portability edits:

- `deepnet` suite: 68/68 passing.
- Combined `deepnet` + simulator suite: 109 passed, 4 failed.
- All four failures are pre-existing simulator filename-case mismatches: implementation produces
  `MAZE_..._SEED...`; tests expect `tiago_maze_..._seed...`.
- 24 chronological and 8 LOSO checkpoints safely reloaded with `weights_only=True`.
- A real held-out covariance successfully passed through `GeoAdaptDecoder` on the RTX 5070 with
  contract validation and produced a finite committed decision.

Mac after portability changes:

- Current `deepnet` suite: 71/71 passing in 1.87 s.
- A real S01-R04 FIF produced 118 windows of shape `(118, 4, 15, 15)` in 0.41 s.
- After a 20-window smoke calibration, 98 windows streamed with finite normalized probabilities at
  1.33 ms per complete decoder step.
- Native CPU network latency: 1.00 ms per covariance.
- MPS with PyTorch CPU fallback: 19.53 ms per network call; it is not recommended.
- A two-epoch real-data training smoke test used 119 training and 118 validation windows and
  completed in 0.49 s with finite history.
- The tested checkpoint is about 65 KB; trainable parameter storage is about 39 KB.
- `python3 -m py_compile deepnet/*.py deepnet/tests/*.py` passes.
- `git diff --check` passes.

The Mac smoke calibration was a functional portability test, not a locked research protocol; do
not report its ad hoc task accuracy. The current `.venv` still lacks an installed `pytest`; the
71-test run used cached temporary test tooling, and the Mac measurements were not written to a
machine-readable artifact. Install `requirements-macos.txt` before attempting to reproduce them,
and treat the timings as documented observations rather than locked benchmark evidence.

## 11. Reproduction commands

Mac setup/test:

```bash
cd /Users/admin/Documents/GitHub/eegthingy
uv pip install --python .venv/bin/python -r deepnet/requirements-macos.txt
.venv/bin/python -m pytest -q deepnet/tests
```

GPU verification/test:

```bash
ssh gpu 'cd ~/Desktop/eegthingy && .venv/bin/python -c "import torch; x=torch.randn(1024,1024,device=\"cuda\"); print(torch.__version__,torch.cuda.get_device_name(),torch.cuda.get_device_capability(),torch.isfinite(x@x).all().item())"'
ssh gpu 'cd ~/Desktop/eegthingy && .venv/bin/python -m pytest -q deepnet/tests'
```

Chronological benchmark:

```bash
.venv/bin/python -m deepnet.experiment benchmark \
  --subjects all \
  --models geoadapt,riemann,fbcsp \
  --seeds 7,17,27 \
  --window deployment \
  --calibration-windows 20 \
  --max-epochs 180 \
  --patience 25 \
  --device cuda \
  --output deepnet/results/chronological_v2_final_3seeds.json
```

Initial neural LOSO:

```bash
.venv/bin/python -m deepnet.loso \
  --subjects all \
  --seeds 7 \
  --calibration-task-events 10 \
  --max-epochs 180 \
  --patience 25 \
  --device cuda \
  --output deepnet/results/nested_loso_v2_seed7.json
```

Matched LOSO baselines and report reconstruction:

```bash
.venv/bin/python -m deepnet.loso_baselines \
  --subjects all \
  --models riemann,fbcsp \
  --output deepnet/results/nested_loso_v2_baselines.json

.venv/bin/python -m deepnet.merge_results \
  deepnet/results/nested_loso_v2_seed7.json \
  deepnet/results/nested_loso_v2_baselines.json \
  --output deepnet/results/nested_loso_v2_combined.json

.venv/bin/python -m deepnet.report \
  deepnet/results/nested_loso_v2_combined.json \
  --update-json \
  --markdown deepnet/results/nested_loso_v2_combined.md
```

Do not casually rerun and overwrite locked result files. Use a new output stem for new protocols,
seeds, or ablations, and retain the full JSON/checkpoints.

## 12. Known limitations and untouched runtime risks

Scientific limitations:

- Eight healthy participants only.
- No validation in the intended disabled/paralyzed population.
- No external public dataset replication.
- No full inner-LOSO/multi-seed subject-held-out estimate.
- No independently trained anchor-only model.
- No matched TSMNet/OTTA-MI/T-TIME/BFT comparison.
- No evidence that causal target updates improve task accuracy.
- No demonstrated benefit from the learned residual branch.
- No continuous asynchronous replay or repeated closed-loop test of this decoder.
- Safety advantage is coupled to very low coverage, especially in LOSO.

Runtime risks outside `deepnet`, reviewed but not fixed:

- Closing the experiment cue window can destroy it without stopping/saving an active stream.
- Live/training sample-rate mismatch only warns.
- Filtering/inference run synchronously inside the acquisition callback.
- The smoother can emit a nonzero command repeatedly after dwell; robot control must edge-trigger.
- The current deployment wrapper accepts covariances, not raw board samples. A validated live
  preprocessing bridge is still required.

## 13. Recommended next work, in order

1. **Preserve and version the current baseline.** Review the untracked package, decide what results
   should remain external/ignored, then create a dedicated `codex/` or user-chosen branch and
   commit only with explicit user approval.
2. **Re-run the 71-test suite on the GPU checkout after syncing Mac compatibility changes.** Do
   not retrain merely for this sync unless code affecting numerical outputs changed.
3. **Run a definitive nested LOSO study:** full inner-participant sweep, several neural seeds,
   protected selection, participant-level aggregation, and retained traces.
4. **Run matched ablations:** independently train anchor-only, full residual, raw/frozen/causal
   covariance state, boundary off/frozen/causal, and intent gate on/off at the same calibration and
   tuning budget.
5. **Add direct modern comparators:** TSMNet first, then matched online/test-time adaptation such as
   OTTA-MI/T-TIME/BFT where feasible.
6. **Replicate on an external public multi-session MI dataset.** Lock its own preprocessing/data
   contract; never quietly pool it with this cohort.
7. **Build continuous replay and a raw-to-covariance live bridge,** validate cadence/edge-triggering,
   then run repeated closed-loop simulation before involving the intended clinical population.
8. **Revise `paper/` only from schema-v2 artifacts.** State competitive accuracy, lower false rest
   commits at lower coverage, and every limitation. Remove or clearly label legacy transductive
   numbers.

## 14. Definition of a responsible continuation

A successful continuation does not merely improve one aggregate. It preserves the data contract,
keeps protected groups isolated, produces per-window auditable traces, reports participant-level
uncertainty and coverage, compares against strong classical and modern adaptation baselines, and
keeps the healthy-volunteer/clinical boundary explicit.

If a future experiment shows no residual or causal-adaptation benefit, report that result honestly
and simplify the system rather than searching for a favorable split.
