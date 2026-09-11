# EEG motor-imagery platform

Run the main menu with:

```powershell
uv sync --frozen
uv run python main.py
```

The menu provides the data collector, live/offline classifiers, and TIAGo maze
simulation. The classifier menu contains the classical EA + FB-CSP and
Riemannian decoders, the project Cardinal networks, and the pretrained MIRepNet
foundation model.

Option `5` opens the separate four-command SSVEP data collector. It presents
top/right/bottom/left flicker targets for forward/right/backward/left, records
the 15 usable Cyton+Daisy EEG channels, labels every protocol phase
automatically, and writes FIF, CSV, and JSON files. Its timing, frequencies,
layout, repetitions, hardware port, and output folder are controlled from one
GUI. See `ssvep/README.md` for the recording contract.

The top-level `MIRepNet Workbench` option provides six guided paths:

1. offline selected-subject testing;
2. online GUI decoding, with or without unlabeled startup calibration;
3. online headless decoding, with or without unlabeled startup calibration;
4. fine-tuning on selected whole subjects or exact recording files;
5. a recorded-subject replay on one of the four original fixed mazes;
6. **live OpenBCI headset control on a fixed maze**.

The live path opens one of the four original mazes, connects the live
Cyton+Daisy stream to MIRepNet, and maps left-hand imagery to LEFT and
right-hand imagery to RIGHT. At every wall the simulator discards decisions
made while driving, waits for a complete fresh post-stop vote window, and uses
only MIRepNet's final committed decision. Closing the maze saves raw EEG,
per-window CSV/Excel decisions, the maze CSV report, and its pose trace.

The maze test uses the four historical 40-corner layouts exactly: standard
maze 1 through 4 are the original seeds 11 through 14. At each fixed-maze
corner, the program chooses an unused recorded EEG epoch whose instruction
matches the turn required there. MIRepNet inference runs at that moment and its
new guess steers the robot; predictions are not precomputed and replayed. A correct guess
follows the corridor. A wrong guess visibly faces a wall, is counted as wrong,
and is then corrected automatically without consuming another EEG epoch. The
HUD and CSV report show the final model score. See
`MIRepNet_SIMPLE_GUIDE.md` for the plain-language explanation.

The offline test can run one or several selected patients in four modes:

1. strict zero-shot, with no target-patient adaptation;
2. unlabeled EA, which uses target EEG signals but never target labels;
3. calibrated, where the first recordings fit a small patient-specific head;
4. a side-by-side comparison of all three on the same later recordings.

Every offline test writes both a detailed JSON result and a formatted Excel
workbook under `results/mirepnet`. The workbook has `Run Summary`,
`Subject Results`, `Epoch Predictions`, and `Protocol Notes` sheets; every
tested EEG epoch records its source file, patient, protocol, true/predicted
label, confidence, and correctness. MIRepNet online sessions also write an
Excel workbook with every live decision window when the session stops.

Fine-tuning can start from the official pretrained weights or continue from a
saved checkpoint. In the interactive selector, choose whole patients using
entries such as `1,5,8` or `S01,S05,S08`, or choose individual files using
entries such as `1,3-6`. New timestamped weights and their complete epoch-by-
epoch training history are saved without overwriting the source checkpoint.

The reusable weights and their SHA-256 hashes are recorded in `models/mirepnet/WEIGHTS.json`.
The recommended checkpoint for the validated workflow is
`mirepnet__4subjects_16runs_5ea48c54.pt`.

## MIRepNet foundation decoder

Choose `Classifier`, then offline or online mode, then decoder choice `5`.
The first new training run downloads the official MIT-licensed checkpoint from
`braindecode/mirepnet-pretrained`; saved fine-tuned models can then be selected
from the same menu without downloading or retraining.

MIRepNet's fixed input contract is bridged to this headset as follows:

1. Pick the 15 usable EEG electrodes and rename legacy T3/T4/T5/T6 positions to
   T7/T8/P7/P8.
2. Filter 8--30 Hz and resample 125 Hz recordings to 250 Hz.
3. Use the audited 0--2 second motor-imagery task interval and repeat it to form
   MIRepNet's required 4-second/1,000-sample input. This avoids feeding the
   model the non-imagery planning interval.
4. Fit Euclidean Alignment on each training recording independently.
5. Interpolate the observed montage to MIRepNet's official 45-channel template
   with its inverse-distance rule.
6. Replace the undocumented three-class pretraining head with the project's
   left-vs-right two-class head, warm it up, then fine-tune the encoder with
   whole-recording validation and early stopping.

For fast patient adaptation, choose an existing MIRepNet checkpoint in offline
mode. The selected training recordings become labeled calibration data for a
small logistic head while the 5.14-million-parameter transformer remains
frozen. This path finishes in seconds rather than fine-tuning the transformer
again.

### Validated local patient-calibration result

The repository's pre-existing frozen Local Exp4 contract contains eight
participants: S1, S3, S4, S5, S6, S7, and S8 runs 1--4, plus S10 runs 5--8.
It explicitly excludes S9 and marks S10 runs 1--4 invalid. Recordings from
S11--S22 are available for exploration but are outside this validated cohort.

Using each participant's first two valid chronological recordings only for
labeled calibration and the remaining two only for testing produced:

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

Thus 7/8 validated participants exceeded 80%, and 2/8 exceeded 90%. The
transformer weights were not changed in this calibration pass. Target test
labels were never used for fitting or adaptation; the completed test block's
unlabeled EEG was used for Euclidean Alignment and feature scaling. The reported
balanced-block threshold also uses the experiment's known 50/50 left/right
trial schedule, so it is an offline balanced-protocol metric rather than an
unconstrained streaming claim.

On the same later-recording test split, the three patient-handling modes were:

| Mode | Macro accuracy | Patients >=80% | Patients >=90% |
| --- | ---: | ---: | ---: |
| Strict zero-shot | 80.36% | 3/8 | 2/8 |
| Unlabeled EA | 86.79% | 6/8 | 2/8 |
| Labeled calibration | 88.63% | 7/8 | 3/8 |

These eight rows include four checkpoint-training patients (S3, S4, S6, S7)
and four checkpoint-unseen patients (S1, S5, S8, S10). Among the genuinely
unseen patients, S5 reached 97.46% strict zero-shot. The other unseen patients
did not consistently reach 80% without adaptation, so zero-shot and calibrated
results must remain separate.

Reproduce the result with:

```powershell
uv run python scripts/evaluate_mirepnet_personalized.py `
  --checkpoint models/mirepnet/mirepnet__4subjects_16runs_5ea48c54.pt `
  --calibration-runs 2 --validated-local-cohort --window-mode task-repeat `
  --device cpu `
  --output results/mirepnet/validated-local-personalized-task-repeat.json
```

The full machine-readable result, including every confusion matrix and protocol
flag, is `results/mirepnet/validated-local-personalized-task-repeat.json`.

To measure generalization without allowing target-participant labels into
training or model selection:

```powershell
uv run python scripts/evaluate_mirepnet.py --holdout-subject 22
```

The result JSON reports two protocols separately:

- strict zero-shot, which uses no target EEG at all;
- deployment EA, which uses target EEG only to estimate an unlabeled alignment
  reference, matching online calibration.

### Measured held-out-patient benchmark

On 2026-09-01, the adaptive CPU ladder below kept participant 5 completely out
of fitting and recording-held-out validation. Each added participant contributes
four recordings; every run used seed 7, 10 epochs, and batch size 32.

| Training participants | Runtime | Strict zero-shot | Unlabeled deployment EA |
| --- | ---: | ---: | ---: |
| 3, 4 | 5.94 min | 90.87% | 91.30% |
| 3, 4, 6 | 8.39 min | 89.57% | 91.74% |
| 3, 4, 6, 7 | 15.75 min | 91.30% | 93.48% |

The ladder stopped at four participants because its runtime exceeded the
15-minute ceiling. The corresponding checkpoint is
`models/mirepnet/mirepnet__4subjects_16runs_5ea48c54.pt`, and the complete
machine-readable report is
`results/mirepnet/train-3-4-6-7__test-5__seed-7.json`.

Reproduce that final stage with:

```powershell
uv run python scripts/evaluate_mirepnet.py --train-subjects 3 4 6 7 --holdout-subject 5 --epochs 10 --batch-size 32 --device cpu
```

Because the requested ladder repeatedly consulted participant 5's test score to
decide whether to add another training participant, these numbers describe this
held-out participant but are not an unbiased final model-selection estimate.
Confirm publication claims with a fresh, untouched participant or nested
participant-level evaluation.

To audit an existing checkpoint on every participant without changing its
weights:

```powershell
uv run python scripts/evaluate_mirepnet_all.py --checkpoint models/mirepnet/mirepnet__4subjects_16runs_5ea48c54.pt --device cpu
```

That checkpoint was tested serially on all 21 available participants. The 17
participants absent from fine-tuning achieved 61.98% macro strict zero-shot
accuracy and 65.01% macro accuracy with unlabeled deployment EA. Two of the 17
unseen participants reached at least 90% with deployment EA. The full per-file,
per-participant report is
`results/mirepnet/all-patients__mirepnet__4subjects_16runs_5ea48c54.json`.

This all-files audit is deliberately retained as a negative control. It must
not be compared directly with the validated Local Exp4 result because it mixes
the frozen benchmark cohort with excluded, explicitly invalid, and unvalidated
recordings. A random within-patient split over those same files reached only
69.34%, confirming that additional fine-tuning alone cannot make every stored
recording exceed 80% when many contain little decodable current-window signal.

Fine-tuning a 5.14-million-parameter transformer is slow on CPU. CUDA is
selected automatically when available. Training epochs, batch size, seed, and
device are configurable in `config.py` or through the evaluation script.

The implementation is in `classifier/mirepnet.py`; focused compatibility tests
are in `tests/test_mirepnet.py`.

## Alternative MIRepNet adapter pipeline

The destination repository's independent adapter implementation is preserved
under `mirepnet_pipeline/`, including its pretrained and fine-tuned checkpoints.
It can be trained directly with:

```powershell
uv run python mirepnet_pipeline/offline_train.py
```

The main application continues to use the validated `classifier/mirepnet.py`
workflow exposed through the MIRepNet Workbench.
