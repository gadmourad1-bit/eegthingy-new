# Causal geometric EEG decoder

`deepnet` is the research package for a compact, geometry-aware left-versus-right
motor-imagery decoder. It is deliberately separate from the acquisition GUI, the
existing classical classifier, and robot-control code. That boundary keeps offline
experiments reproducible and prevents training code from depending on live hardware
state.

This is an experimental assistive-BCI research system. The recordings in this repository
come from healthy volunteers; the package has not been evaluated in people with paralysis
or other motor disabilities and is not a medical device. Do not describe a result from this
dataset as clinical validation or deployment readiness.

The detailed hypotheses, evaluation rules, ablations, and novelty boundary are in
[`RESEARCH_DESIGN.md`](RESEARCH_DESIGN.md). The current locked chronological result and
its limitations are in [`RESULTS.md`](RESULTS.md).

## What is implemented

- A fixed, auditable 32-session data manifest with epoch-level provenance.
- Continuous FIR filter-bank preprocessing and scale-aware SPD covariance estimation.
- Group-disjoint within-subject, chronological cross-session, and leave-one-subject-out
  (LOSO) split generators that hard-fail on protected-group overlap.
- `GeoAdaptNet`, a 9,678-parameter anchor-preserving SPD residual network.
- Differentiable, scale-aware SPD operations with stable spectral gradients.
- A strictly causal, label-free, constant-memory online covariance and output-boundary
  adapter.
- Explicit confidence gating and optional intent-versus-rest prediction.
- Classical EA-FBCSP and Riemannian tangent-space logistic-regression baselines.
- Accuracy, balanced accuracy, kappa, AUC, Brier score, calibration error, coverage,
  selective accuracy, rest false-commit rate, and synchronized batch-one latency.

No performance claim is implied by the presence of the implementation. Only results
produced by the locked protocols in this package should be reported.

## Current verified results

In the three-seed chronological benchmark, GeoAdaptNet reached 84.86 ± 12.67% balanced
accuracy, statistically tied with Riemannian LR (84.24%) and below FBCSP (87.53%) by a
paired difference whose interval crosses zero. Its separate intent gate reduced annotated-
rest false commits to 15.71%, versus 47.95% and 50.85%, at lower 41.01% coverage.

In the initial one-seed strict subject-held-out benchmark, GeoAdaptNet reached 83.54 ±
9.36%, between Riemannian LR (81.61%) and FBCSP (85.67%). Rest false commits were 9.68%,
but coverage was only 19.22%. These are high-precision/low-availability operating points,
not task-accuracy superiority. Full intervals and limitations are in `RESULTS.md`.

The earlier 86.40% schema-v1 table is superseded because its refit schedule and
temperature-transfer contract were invalid. Do not cite it.

## Fixed data contract

The cohort is not discovered with a permissive glob. `deepnet.config` contains the exact
manifest:

| Subject | Included runs | Session IDs |
| --- | --- | --- |
| 1 | 1, 2, 3, 4 | S01-R01 through S01-R04 |
| 3 | 1, 2, 3, 4 | S03-R01 through S03-R04 |
| 4 | 1, 2, 3, 4 | S04-R01 through S04-R04 |
| 5 | 1, 2, 3, 4 | S05-R01 through S05-R04 |
| 6 | 1, 2, 3, 4 | S06-R01 through S06-R04 |
| 7 | 1, 2, 3, 4 | S07-R01 through S07-R04 |
| 8 | 1, 2, 3, 4 | S08-R01 through S08-R04 |
| 10 | 5, 6, 7, 8 | S10-R05 through S10-R08 |

This is 8 participants and 32 recordings. Subject 9 is excluded. Subject 10 runs 1-4
are known bad recordings and are excluded; runs 5-8 are the valid replacement sessions.
Every cached row retains subject, run, session, event sample/onset, trial, annotation, and
source-file provenance.

The model input is a filter bank of four spatial covariance matrices:

- Sampling frequency: 125 Hz.
- Channels, in fixed order: `Cz, Pz, C3, C4, T5, T6, Fz, F7, F8, F3, F4, T3, T4, P3, P4`.
- Bands: 8-12, 11-15, 14-20, and 20-30 Hz.
- Tensor per trial: `(4, 15, 15)` SPD covariances.
- Main labels: left hand `0`, right hand `1`.
- Optional neutral/rest label: `-1`, used only by the separate intent head.

Covariances are formed in float64 arithmetic, symmetrized, and shrunk toward a scaled
identity before conversion to the configured output precision. The cache filename hashes
all preprocessing settings and the source file metadata. Cached arrays live under
`deepnet/cache/` and are not version-controlled.

### Deployment and legacy windows

The default `deployment` epoch is 0.0-2.0 s from task-cue onset. It matches the intended
online decoder and contains 251 samples because MNE includes both endpoints.

The `legacy` epoch is 0.5-2.5 s. It is retained only to reproduce or ablate older paper
numbers. The task annotation lasts about 2.1 s, so this window includes roughly 0.4 s of
the following rest phase and does not match the live 2 s window. New primary results must
use `deployment`; any `legacy` result must be labeled as such.

## Model at a glance

`GeoAdaptNet` begins with a deliberately strong full-resolution tangent-space linear
classifier. A small learned geometric branch adds a gated residual:

1. A per-band log-Euclidean reference aligns each 15 x 15 covariance.
2. A source-only running standardizer conditions all four full tangent vectors
   (4 x 120 = 480 features), and the anchor maps them directly to two logits.
3. Each residual band uses a semi-orthogonal `BiMap` from 15 to 8 dimensions, eigenvalue
   rectification, matrix logarithm, norm-preserving upper-triangle vectorization, and a
   24-wide encoder.
4. Learned band attention and a 32-wide fusion layer produce two residual logits.
5. Final logits are `anchor_logits + sigmoid(gate) * residual_logits`; the gate initializes
   at 0.02, so training starts close to the hard-to-break anchor.
6. An optional scalar head estimates intent versus rest without changing the main task
   into an artificial three-class problem.

The default model has 9,678 trainable parameters, below the package's enforced 50,000
parameter ceiling. SPD spectral functions use divided-difference gradients, relative
eigenvalue floors, and fp32 spectral algebra under low-precision autocast to remain stable
at physical EEG covariance scales.

## Blackwell GPU environment

The configured desktop is `user-desktop-ryzen9` with an NVIDIA RTX 5070 (12 GB,
Blackwell `sm_120`), Ubuntu 24.04, and driver 595.71.05. The Mac SSH alias is `gpu`; the
desktop checkout is `~/Desktop/eegthingy`.

Blackwell support is not optional: use the CUDA 12.8 builds of both PyTorch and
TorchAudio. A default PyPI TorchAudio build can pull a different CUDA runtime and fail
with `libcudart.so.13` errors. The exact tested environment is recorded in
`deepnet/requirements-cu128.txt`.

On the desktop, create or refresh the environment with `uv`:

```bash
cd ~/Desktop/eegthingy
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r deepnet/requirements-cu128.txt
```

Verify that PyTorch sees the Blackwell device and can execute a kernel:

```bash
.venv/bin/python -c "import torch; x=torch.randn(1024,1024,device='cuda'); print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.isfinite(x@x).all().item())"
```

Expected identifiers are `2.11.0+cu128`, `NVIDIA GeForce RTX 5070`, and `(12, 0)`.

From the Mac, the same checks and test suite can be driven over SSH:

```bash
ssh gpu 'cd ~/Desktop/eegthingy && .venv/bin/python -m pytest -q deepnet/tests'
ssh gpu 'cd ~/Desktop/eegthingy && nvidia-smi'
```

## MacBook execution

GeoAdaptNet also runs locally on Apple silicon. The tested Mac environment is Python
3.12 on an M5 Pro; its exact CPU dependencies are in `requirements-macos.txt`:

```bash
uv pip install --python .venv/bin/python -r deepnet/requirements-macos.txt
.venv/bin/python -m pytest -q deepnet/tests
```

Use `--device auto` (the default) or `--device cpu`. Although PyTorch reports the MPS
device as available, PyTorch 2.12.1 does not implement the `torch.linalg.eigh` Metal
kernel required by the SPD network. The device resolver probes that operation and keeps
the model on CPU when it is unavailable. `PYTORCH_ENABLE_MPS_FALLBACK=1` makes explicit
MPS execution possible, but it moves each eigendecomposition back to CPU and was much
slower in the measured batch-one test.

On the M5 Pro smoke test, all 71 package tests passed, a saved schema-v2 checkpoint
produced finite probabilities from a real 118-window FIF session, native CPU model
latency was 1.00 ms per covariance, and the complete causal decoder averaged 1.33 ms per
post-calibration window. A two-epoch real-data training smoke test used 119 training and
118 validation windows and completed in 0.49 s. These timings demonstrate local
execution, not a replacement for the locked Blackwell benchmark or a clinical latency
claim.

## Run the chronological benchmark

The experiment entry point implements the primary prospective local protocol: train on
the first two recordings, select and calibrate the exact frozen checkpoint on the third,
and score the fourth after an unlabeled chronological calibration prefix. The validation
recording is never added to model fitting because temperature is specific to that
checkpoint's logit scale. It compares `geoadapt`, `riemann`, and `fbcsp` on identical
training, validation, calibration, and outer rows.

First build the fingerprinted preprocessing cache for all 32 sessions. The default includes
rest windows for intent training and rest-gated adaptation. Classification metrics are
computed on task windows, while post-calibration rest windows are processed causally and
reported separately through the rest false-commit rate:

```bash
.venv/bin/python -m deepnet.experiment prepare --window deployment
```

Run a one-participant smoke benchmark before committing to the full cohort:

```bash
.venv/bin/python -m deepnet.experiment benchmark \
  --subjects 1 \
  --models geoadapt,riemann,fbcsp \
  --seeds 7 \
  --window deployment \
  --calibration-windows 20 \
  --device cuda \
  --output deepnet/results/chronological_v2_smoke.json
```

Then run every participant with several fixed neural seeds:

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

The runner atomically checkpoints each completed model/participant/seed fold and resumes a
compatible output file automatically. Use `--no-resume` only when intentionally replacing
an experiment. Classical baselines are deterministic and run once even when multiple
neural seeds are supplied. `--task-only` removes rest windows; keep it as a named ablation,
not the primary safety-oriented setting.

From the Mac, prefix either command with:

```bash
ssh gpu 'cd ~/Desktop/eegthingy && .venv/bin/python -m deepnet.experiment benchmark --subjects 1 --models geoadapt,riemann,fbcsp --seeds 7 --device cuda --output deepnet/results/chronological_v2_smoke.json'
```

Results are JSON and include the protocol identifier, complete benchmark configuration,
environment, per-fold metrics, update counts, epoch-selection details, latency, and a
summary. Neural runs also export safe-loadable checkpoints under
`deepnet/checkpoints/<result-stem>/`. Treat the JSON as the source of truth for tables;
do not copy terminal summaries by hand.

## Run the initial nested LOSO benchmark

`deepnet.loso` keeps the outer participant completely absent from training. For each outer
fold it uses the next participant in the fixed cohort order as the single inner-validation
participant, trains on the remaining six, selects and freezes that exact checkpoint, and
resets unlabeled calibration/adaptation state for each of the four outer-target recordings.
The inner-validation participant is never added to fitting, so its selected temperature
is applied to the same model that produced the validation logits:

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

This is an initial nested estimate, not the final subject-independent protocol: one
deterministically chosen inner participant selects the checkpoint, temperature, and
boundary blend. A definitive paper result should average selection over a full inner LOSO
sweep, use multiple neural seeds, and compare matched modern adaptation methods. The
runner records that limitation in every result artifact and checkpoint.

Run the two matched classical baselines under the same 6/1/1 subjects, independent
per-recording calibration prefixes, temperature/blend grid, and outer rows:

```bash
.venv/bin/python -m deepnet.loso_baselines \
  --subjects all \
  --models riemann,fbcsp \
  --output deepnet/results/nested_loso_v2_baselines.json
```

These binary baselines have no intent/rest head. Their task commits and rest false commits
therefore use left/right confidence only; the result artifact records that asymmetry.

Load an exported model without permitting arbitrary pickle objects:

```python
from deepnet.engine import load_model_checkpoint

model, metadata = load_model_checkpoint(
    "deepnet/checkpoints/chronological_v2_final_3seeds/subject-01_seed-7.pt",
    map_location="cuda",
)
print(metadata["temperature"], metadata["target_median_blend"])
```

For causal covariance-level streaming, use the deployment wrapper. It freezes every model
weight, calibrates both states without labels, applies the learned intent gate, and
enforces predict-then-update ordering:

```python
from deepnet.deployment import GeoAdaptDecoder

decoder = GeoAdaptDecoder.from_checkpoint(
    checkpoint_path,
    device="cuda",
    preprocessing_contract=live_preprocessing_contract,
)
decoder.calibrate(unlabeled_calibration_covariances)

decision = decoder.step(live_filterbank_covariance)
if decision.committed:
    send_robot_command(decision.prediction)
```

The wrapper expects shrinkage covariances with shape `(4, 15, 15)`. Raw-board channel
selection, resampling, causal filtering, window scheduling, and robot edge-triggering
remain the responsibility of the acquisition runtime; do not feed raw voltage windows
directly to it.

## Stable Python APIs

Load the fixed cohort with the deployment window:

```python
from deepnet.config import DataConfig
from deepnet.data import load_sessions, valid_sessions

config = DataConfig(window_name="deployment", include_rest=False)
data = load_sessions(valid_sessions(config), config)

print(data.covariances.shape)  # (epochs, 4, 15, 15)
print(data.epochs.shape)       # (epochs, 4, 15, 251)
print(data.session_ids[:3])
```

Materialize protected outer folds:

```python
from deepnet.protocols import (
    chronological_cross_session_splits,
    loso_splits,
    within_subject_splits,
)

within = within_subject_splits(data)
chronological = chronological_cross_session_splits(data)
loso = loso_splits(data)
```

Train with a caller-supplied, complete held-out validation group and retain the exact
checkpoint whose logit calibration is measured there:

```python
from deepnet.engine import CovarianceDataset, TrainConfig, train_model
from deepnet.model import GeoAdaptNet

train_set = CovarianceDataset(train_covariances, train_labels, log_references=train_refs)
validation_set = CovarianceDataset(validation_covariances, validation_labels,
                                   log_references=validation_refs)

selection = train_model(
    GeoAdaptNet(auxiliary_intent=False),
    train_set,
    validation_set,
    TrainConfig(device="cuda"),
)
frozen_model = selection.model
```

`train_fixed_epochs` remains available for ablations that do not transfer model-specific
temperature calibration, or when final-model calibration is obtained by a protected
source-only cross-fitting design. The locked primary runners do not transfer a validation
temperature to a separately refit model.

The outer test labels must not choose the epoch count, hyperparameters, reference, or
abstention threshold. The exact nested rules are in `RESEARCH_DESIGN.md`.

### Causal online adapter

`DualLevelOnlineAdapter` holds only a per-band covariance reference and a scalar log-odds
center. Calibration is unlabeled. Every stream step follows a predict-then-update rule:
the current window is aligned and classified using state at time `t`; only after the
decision is formed may it update state for `t + 1`.

```python
from deepnet.recenter import BoundaryRecenter, DualLevelOnlineAdapter

adapter = DualLevelOnlineAdapter(
    boundary=BoundaryRecenter(class1_label=0, class2_label=1),
)
adapter.calibrate(calibration_covariances, scorer=frozen_model_scorer)

result = adapter.step(
    live_covariance,
    scorer=frozen_model_scorer,
    rest_probability=rest_probability,
)
if result.committed:
    command = result.prediction
```

The model scorer must return one log-odds scalar or one pair of logits. Model weights stay
frozen; no target label, optimizer, replay buffer, or future window is used. Adapter state
is JSON-serializable for deterministic pause/resume tests.

## Tests and reproducibility

Run focused tests from the repository root:

```bash
.venv/bin/python -m pytest -q deepnet/tests
```

The tests cover data manifest and provenance, cache invalidation, covariance SPD
properties, group leakage detection, spectral gradients, semi-orthogonal maps, model
shape/parameter limits, causal state transitions, metrics, baselines, and training helpers.

For every reported experiment, retain at least:

- Git commit and dirty-worktree state.
- Full command and serialized configuration.
- Preprocessing fingerprint and exact session manifest.
- Outer and inner fold IDs.
- Random seed, selected checkpoint, and whether any refit occurred.
- Calibration-prefix policy and whether those windows were excluded from scoring.
- Model parameter count, latency method, and hardware/software versions.
- Per-fold predictions and probabilities, not only aggregate metrics.

## Package map

| File | Responsibility |
| --- | --- |
| `config.py` | Immutable cohort and preprocessing contract |
| `data.py` | FIF loading, annotation mapping, caching, provenance, covariance construction |
| `protocols.py` | Group-disjoint split generation and leakage validation |
| `spd.py` | Differentiable SPD algebra and manifold layers |
| `model.py` | Anchor-preserving geometric residual network |
| `engine.py` | Deterministic training, optional fixed-epoch refit, inference, latency |
| `recenter.py` | Strictly causal covariance and boundary adaptation |
| `baselines.py` | EA-FBCSP and tangent-logistic comparators |
| `loso.py` | Protected 6/1/1 neural subject-held-out benchmark |
| `loso_baselines.py` | Matched strict subject-held-out classical benchmarks |
| `deployment.py` | Checkpoint-backed causal frozen-weight decoder |
| `report.py` | Prediction-trace validation, participant summaries, and intervals |
| `merge_results.py` | Compatibility-checked combination of benchmark artifacts |
| `metrics.py` | Classification, calibration, and selective-prediction metrics |
| `tests/` | Unit and numerical regression tests |
