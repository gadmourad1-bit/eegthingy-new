# EEG motor-imagery and SSVEP platform

A research-focused EEG project for motor-imagery decoding and visual steady-state
stimulation experiments. The codebase includes a live MI dashboard, offline
analysis scripts, a maze-control workflow, and a standalone SSVEP data-collection
interface.

## Quick start

```powershell
uv sync --frozen
uv run python main.py
```

From the main menu you can access:

- MI motor-imagery workflows
- offline and online MI decoding
- MIRepNet training and checkpoint selection
- SSVEP data collection
- maze replay and live control loops

## Project at a glance

This repository contains two main EEG pipelines:

1. MI motor-imagery pipeline
   - classical EA + FB-CSP and Riemannian decoders
   - project deep decoders
   - pretrained MIRepNet foundation model
   - live and offline decoding workflows

2. SSVEP pipeline
   - four-command flicker interface
   - automated phase labeling
   - FIF, CSV, and JSON export
   - separate documentation in [ssvep/README.md](ssvep/README.md)

## Repository layout

- [main.py](main.py) — app entry point and menu launcher
- [config.py](config.py) — project-wide settings, labels, and device choices
- [classifier/](classifier) — MI decoders and online/offline logic
- [scripts/](scripts) — evaluation, training, and benchmark scripts
- [ssvep/](ssvep) — SSVEP task, GUI, and recording contract
- [models/](models) — saved checkpoints and model metadata
- [results/](results) — offline experiment summaries and JSON outputs
- [tests/](tests) — validation for core functionality, including MIRepNet

## MI motor-imagery workflow

### Overview

The MI stack supports:

- classical offline decoding
- online GUI and headless decoding
- target-patient adaptation and calibration
- MIRepNet fine-tuning and inference
- replay and maze-control evaluation

The top-level MIRepNet Workbench exposes six guided paths:

1. offline selected-subject testing
2. online GUI decoding, with or without unlabeled startup calibration
3. online headless decoding, with or without unlabeled startup calibration
4. fine-tuning on selected whole subjects or exact recording files
5. recorded-subject replay on one of the original fixed mazes
6. live OpenBCI headset control on a fixed maze

### Label convention

The project label convention is explicit in [config.py](config.py):

- `1 = left_hand`
- `2 = right_hand`

This same ordering is used in live probability logs and class indexing, so the
UI numeric labels correspond to left/right motor imagery rather than unrelated
experiment labels.

### MIRepNet device behavior

MIRepNet fine-tuning defaults to `MIREPNET_DEVICE = 'auto'`.

- If a compatible CUDA build of PyTorch is installed, training uses CUDA.
- If not, it falls back to CPU.
- Explicit device overrides such as `cpu` or `cuda` are accepted by the training
  scripts.

### MIRepNet preprocessing contract

The MIRepNet pipeline bridges the headset to the official model input contract as
follows:

1. Pick the 15 usable EEG electrodes and rename legacy T3/T4/T5/T6 to T7/T8/P7/P8.
2. Filter 8--30 Hz and resample 125 Hz recordings to 250 Hz.
3. Use the audited 0--2 second MI task window and repeat it to form the model's
   required 4-second / 1000-sample input.
4. Fit Euclidean Alignment on each training recording independently.
5. Interpolate the observed montage to MIRepNet's 45-channel template using the
   inverse-distance rule.
6. Replace the pretraining three-class head with the project's left-vs-right head,
   warm it up, and fine-tune the encoder with whole-recording validation.

The default MIRepNet window mode is `task-repeat`, which takes the valid task
window and repeats it to fill the required 4-second segment. This is the default
used by the training scripts and the live workbench.

### How Euclidean Alignment works

Euclidean Alignment (EA) normalizes the spatial covariance of the measured EEG
channels before the signal is interpolated. For each trial $X_k$ with shape
`(15 channels, samples)`, the pipeline first removes its per-channel temporal
mean and computes its channel covariance:

$$
C_k = \frac{(X_k - \bar{X}_k)(X_k - \bar{X}_k)^T}{T-1}
$$

The reference covariance is the average over the calibration or training
trials:

$$
C_{ref} = \frac{1}{N}\sum_{k=1}^{N} C_k
$$

EA computes the inverse square root of this reference covariance,
$R = C_{ref}^{-1/2}$, using an eigenvalue decomposition. Each trial is then
aligned by left-multiplying its channels:

$$
X_{k,EA} = R X_k
$$

This makes the average aligned covariance approximately the identity matrix,
reducing differences in channel scale, correlation, and subject-specific
recording conditions. The same calibration-derived whitener is reused for
subsequent live windows. EA does not create new electrodes or interpolate
signals; it only transforms the genuinely observed channels. The resulting
15-channel signal is then passed to the inverse-distance projection described
below.

### How 15 channels become 45

The headset does not measure all 45 electrodes expected by the pretrained
MIRepNet encoder. It provides 15 usable electrodes after excluding the known
disconnected input. The project therefore uses a fixed spatial interpolation
matrix with shape `(45, 15)`:

```text
15 measured channels -> Euclidean Alignment -> 45-channel MIRepNet template
```

The 45 channels are the official MIRepNet template, arranged over frontal,
frontocentral, central, centroparietal, and parietal rows. The 15 measured
channels are first renamed to the template's modern names where necessary:
`T3 -> T7`, `T4 -> T8`, `T5 -> P7`, and `T6 -> P8`.

This does not create 30 new independent sensors. It creates a model-compatible
representation of the existing measurements. For every target template
electrode:

- if the target electrode was measured, its output is copied directly from the
  corresponding source channel;
- otherwise, its value is a weighted average of the measured channels, with
  nearby electrodes receiving larger weights.

For a target position $p$ and measured source positions $s_i$, the inverse
distance rule is:

$$
w_i = \frac{1}{\lVert p-s_i\rVert + \epsilon}, \qquad
\hat{x}(p,t) = \sum_i \frac{w_i}{\sum_j w_j} x_i(t)
$$

Here, $x_i(t)$ is the signal at source channel $i$, $\lVert p-s_i\rVert$ is
the distance between the target and source positions in the template's 2-D
layout, and $\epsilon$ is a small numerical stabilizer. The weights are
non-negative and normalized to sum to one, so each interpolated channel stays
within the local weighted signal range. Exact source channels use a one-hot
weight instead of interpolation.

The order matters. Euclidean Alignment is fitted and applied in the original
15-channel space first, where the channels are genuinely measured. Only then
does the pipeline apply the `(45, 15)` interpolation matrix. Interpolating
first would manufacture a 45-channel covariance matrix from only 15
independent signals and can make the alignment rank-deficient.

### Original MIRepNet structure

The released MIRepNet model is a pretrained motor-imagery representation
model. Its main downstream path is:

```text
45-channel EEG, 1000 samples
  |
  v
convolutional patch embedding
  |
six transformer blocks
  |
mean over temporal tokens
  |
linear classification head
```

In the converted checkpoint used here, the patch embedding applies a temporal
convolution, a convolution spanning all 45 channels, batch normalization,
ELU, average pooling, dropout, and a projection to 256-dimensional tokens.
The six repeated transformer blocks each contain pre-normalized eight-head
self-attention and a four-times-expanded GELU feed-forward network, with
residual connections and dropout. The token sequence is averaged before the
final linear layer. The resulting encoder has approximately 5.14 million
parameters.

The original pretraining recipe combines masked-token reconstruction with
supervised MI classification. That pretraining teaches the encoder general
motor-imagery features; downstream use normally replaces or fine-tunes the
classification head for the target task.

### What this project changed, and why

The project preserves the pretrained spatial-temporal encoder and its weight
names so the official checkpoint can be loaded. The changes are an adaptation
layer around that encoder, plus a new downstream head:

| Original MIRepNet contract | This project | Reason |
| --- | --- | --- |
| 45-channel template input | 15 measured channels aligned, then interpolated to 45 | The Cyton+Daisy montage does not contain all template electrodes |
| Official dataset preprocessing | 8--30 Hz filtering, 125 -> 250 Hz resampling, project channel aliases, and Euclidean Alignment | Match the local Exp4 recordings and reduce subject/session distribution differences |
| Native MI trial timing | Audited Exp4 0--2 s task window, repeated to 4 s | The encoder requires 1000 samples at 250 Hz, while the useful local task segment is 2 seconds |
| Original pretrained output classes | Two outputs: `left_hand` and `right_hand` | The live project has a two-class control vocabulary |
| General downstream fine-tuning | Recording-held-out validation, short head warm-up, then encoder fine-tuning | Avoid leakage between recordings and adapt efficiently to the local domain |

The classification head is therefore the part intentionally discarded from the
pretrained checkpoint: its weights are not reused when the target classes do
not match. The convolutional patch embedding and transformer encoder are
loaded from the official weights, while the project's two-class `final_layer`
is trained for labels `1 = left_hand` and `2 = right_hand`. This keeps the
learned MI representation while making the model compatible with the local
montage, timing, labels, and live-control workflow.

### MIRepNet checkpoints and tuning

Fine-tuning can start either from the official pretrained weights or from a saved
checkpoint. The selector accepts whole patients such as `1,5,8` or `S01,S05,S08`,
or individual files such as `1,3-6`.

Saved checkpoints include:

- model weights
- class labels
- training history
- validation metrics
- provenance metadata

The reusable checkpoint registry is recorded under `models/mirepnet/WEIGHTS.json`.
The recommended validated checkpoint is:

- `mirepnet__4subjects_16runs_5ea48c54.pt`

### MIRepNet validated results

The repository includes a frozen Local Exp4 validation contract with eight
participants: S1, S3, S4, S5, S6, S7, and S8 runs 1--4, plus S10 runs 5--8.
The project explicitly excludes S9 and marks S10 runs 1--4 invalid.

Using each participant's first two valid recordings for calibration and the next
two for testing produced:

| Participant | Later-recording test accuracy |
| --- | ---: |
| S1 | 89.92% |
| S3 | 88.33% |
| S4 | 88.33% |
| S5 | 98.31% |
| S6 | 98.33% |
| S7 | 73.33% |
| S8 | 86.55% |
| S10 | 89.74% |
| **Macro average** | **89.11%** |

The three patient-handling modes on that same split were:

| Mode | Macro accuracy | Patients >=80% | Patients >=90% |
| --- | ---: | ---: | ---: |
| Strict zero-shot | 80.36% | 3/8 | 2/8 |
| Unlabeled EA | 86.79% | 6/8 | 2/8 |
| Labeled calibration | 88.63% | 7/8 | 3/8 |

### Reproduction commands

The local validated MIRepNet workflow can be reproduced with:

```powershell
uv run python scripts/evaluate_mirepnet_personalized.py `
  --checkpoint models/mirepnet/mirepnet__4subjects_16runs_5ea48c54.pt `
  --calibration-runs 2 --validated-local-cohort --window-mode task-repeat `
  --device cpu `
  --output results/mirepnet/validated-local-personalized-task-repeat.json
```

For a held-out-patient benchmark:

```powershell
uv run python scripts/evaluate_mirepnet.py --holdout-subject 22
```

For the final adaptive ladder reported in the project notes:

```powershell
uv run python scripts/evaluate_mirepnet.py --train-subjects 3 4 6 7 --holdout-subject 5 --epochs 10 --batch-size 32 --device cpu
```

To audit a checkpoint across all available participants without altering the model:

```powershell
uv run python scripts/evaluate_mirepnet_all.py --checkpoint models/mirepnet/mirepnet__4subjects_16runs_5ea48c54.pt --device cpu
```

## SSVEP workflow

### Overview

The SSVEP collector is a separate module launched from the main menu and is
intentionally self-contained. It is designed for visual steady-state tasks with
four targets and automatic timing/annotation handling.

### Default command mapping

The SSVEP interface uses the following default stimulus mapping:

- top: FORWARD at 8 Hz
- right: RIGHT at 10 Hz
- bottom: BACKWARD at 12 Hz
- left: LEFT at 15 Hz

Each randomized block contains every command exactly once. During action periods,
all four flicker patches are active, while the subject attends the cued target.

### Safety and usage notes

- Synthetic mode is only for UI and file-writing validation.
- Real-board operation includes a safety confirmation before starting.
- Flickering light can trigger symptoms in photosensitive individuals; use the
  lab's approved screening and stop procedure.

### Saved outputs

Every session creates timestamped files in the session output folder:

- `*_raw.fif` — EEG data, accelerometer channels when available, and annotations
- `*_events.csv` — per-phase event log with command and frequency
- `*_metadata.json` — timing, settings, channel order, and protocol details

Annotation names follow the pattern:

```text
ssvep/<command>/<frequency>Hz/action/t001
```

The physical channel 8 is excluded because it is the known disconnected input in
this 15-electrode setup. Legacy T3/T4/T5/T6 names are stored as their modern
10-20 equivalents T7/T8/P7/P8.

### Launching the collector

```powershell
uv run python main.py
```

Then choose:

- `5) SSVEP Data Collector`

More detail is in [ssvep/README.md](ssvep/README.md).

## Maze and live-control workflow

The live MI path opens one of the original fixed mazes, connects the live
Cyton+Daisy stream, and maps left-hand imagery to LEFT and right-hand imagery to
RIGHT. At each wall, the simulator discards decisions made while driving, waits
for a clean post-stop vote window, and uses only the final committed decision.

This flow saves:

- raw EEG stream
- per-window decision CSV/Excel logs
- maze CSV reports
- pose traces

The project includes a plain-language walkthrough in [MIRepNet_SIMPLE_GUIDE.md](MIRepNet_SIMPLE_GUIDE.md).

## Benchmarks and notes

The project keeps benchmark results and negative controls in the repo for
transparency:

- validated Local Exp4 results above
- held-out participant benchmarks
- all-patient audit results
- negative-control comparisons against broader, less validated recordings

Important caution: the all-patient audit is a negative control and should not be
compared directly with validated Local Exp4 results because it mixes valid and
invalid recordings.

## Training and implementation notes

- Fine-tuning a 5.14-million-parameter transformer is slow on CPU.
- CUDA is selected automatically when present.
- Training epochs, batch size, seed, and device are configurable in [config.py](config.py).
- The implementation lives in [classifier/mirepnet.py](classifier/mirepnet.py).
- Focused compatibility tests are in [tests/test_mirepnet.py](tests/test_mirepnet.py).

## Alternative MIRepNet adapter pipeline

The independent adapter implementation remains preserved under the
`mirepnet_pipeline/` directory, including pretrained and fine-tuned checkpoints.
It can be trained directly with:

```powershell
uv run python mirepnet_pipeline/offline_train.py
```

The main application continues to use the validated `classifier/mirepnet.py`
workflow exposed through the MIRepNet Workbench.

## Troubleshooting and environment notes

- If the environment is missing CUDA support, the project falls back to CPU for
  MIRepNet training.
- Use an explicit `--device cuda` or `--device cpu` to force behavior when needed.
- If the virtual environment has drifted, refresh it with `uv sync --frozen` or
  reinstall the environment in a clean location.

## Summary

This project combines a research-grade MI decoder stack with a dedicated SSVEP
collection app, giving you both motor-imagery and visual-evoked experimental
workflows under one codebase.

