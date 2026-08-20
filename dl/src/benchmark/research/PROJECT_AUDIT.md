# Project audit before the neural rebuild

This audit covers the exported development transcripts, the main and historical branches,
the acquisition/classifier/simulator code, all recordings, the current `paper/` draft, and
the four prior publications in `papers/`. It explains why the new `deepnet` results must
not be mixed with several older numbers.

## Data and cohort

- The repository contains 40 FIF files, but the valid research cohort is 32 recordings:
  subjects 1, 3, 4, 5, 6, 7, 8 with runs 1-4, plus subject 10 with replacement runs 5-8.
- Subject 9 and subject 10 runs 1-4 are excluded before splitting.
- A valid task recording normally has 60 motor-imagery trials, balanced 30 left and 30
  right before artifact rejection.
- The 15-channel, 125 Hz acquisition and four overlapping 8-30 Hz bands are consistent
  across the current classical and new geometric pipelines.
- The task annotation is about 2.1 s. The older 0.5-2.5 s epoch includes roughly 0.4 s of
  the following rest period, whereas the new primary 0.0-2.0 s window matches intended
  deployment timing.

## Why older offline accuracy is optimistic

The current classical command-line evaluation in `classifier/run.py` is useful for
interactive exploration, but not for a paper generalization claim:

- Lines 445-450 use shuffled epoch-level `StratifiedKFold` after pooling recording files.
  Epochs from the same recording therefore occur on both sides of a fold.
- Each validation fold supplies all of its unlabeled windows to `set_reference` before it
  is scored. This is a transductive fold, not prospective online validation.
- Lines 455-457 estimate the final target reference from the entire test set before
  predicting that same set. The operation is label-free but uses future target windows.
- Lines 429-431 merely warn when the same file is selected for training and testing; a
  protected evaluation should fail.

Consequently, the paper's older 91.1% personalized and 84.8% pooled EA figures, and its
89.5%/77.6% Riemannian figures, should be labeled as legacy transductive analyses. They
are not comparable to the chronological calibration-prefix result in `RESULTS.md`.

Historical deep-learning code on the `deep-learning` branch wraps several Braindecode
models but evaluates a small subset of participants and depends on experiment scripts or
outputs that were not committed consistently. Those tables are useful debugging history,
not reproducible primary baselines.

## Current paper claim risks

- The simulator study has four users and one run per user/condition cell. Completion and
  corner/command percentages are descriptive; there is not enough repetition for strong
  condition inference.
- The paper calls a condition “generalized” even though the driving user can be part of
  training. That is pooled or personalized transfer, not leave-one-subject-out evidence.
- The reported median 0.20 s response time needs a precise start/end definition. A 5 Hz
  stream plus 4-of-5 consensus and a three-window dwell cannot generally commit a fresh
  command in one stride.
- The current data are from healthy volunteers. Neither the paper nor the new package can
  claim validation for people with paralysis.
- The paper already lists subject-held-out evaluation and recentering ablation as future
  work; those should remain limitations until measured.

The four local publications are summarized in `RESEARCH_DESIGN.md`. Collectively they use
small healthy cohorts, classical CSP-family methods, and in two cases overt movement that
can contain EMG/movement information. They do not establish clinical, subject-independent,
or modern online-adaptation evidence.

## Runtime and acquisition risks outside the new package

These issues were reviewed but deliberately not changed by the isolated neural rebuild:

- Closing the subject cue window destroys it without stopping/saving an active stream
  (`gui/exp4/subject/root.py`, lines 72-75); normal completion performs that cleanup at
  lines 102-110.
- A live/training sample-rate mismatch only emits a warning (`classifier/run.py`, lines
  559-566) while reusing training window counts and filter assumptions.
- Filtering and inference run synchronously inside the acquisition callback
  (`classifier/run.py`, lines 604 onward). An overrun can collapse nominal strides and
  make decision cadence irregular.
- The smoother returns a nonzero `final` on every later window after dwell is reached
  until consensus changes (`classifier/smoother.py`, line 74). A downstream robot consumer
  must edge-trigger or it can repeat the same command.
- The repository-wide simulator suite currently has four pre-existing filename-case
  failures: tests expect `tiago_maze_..._seed...`, while implementation returns
  `MAZE_..._SEED...`. This is unrelated to `deepnet`; the neural suite passes separately.

## What the rebuild changes

- An explicit manifest and provenance-preserving cache replace file globs.
- Complete recording/subject split generators hard-fail leakage.
- Inner checkpoint/calibration selection and the outer chronological session are distinct;
  the exact calibrated checkpoint is frozen instead of transferring its temperature to a
  differently scaled refit.
- Only a fixed unlabeled target prefix initializes target state; prefix task events are
  excluded from scoring and future windows are processed causally.
- Physical EEG covariance scale is handled by relative spectral floors. The original
  absolute `1e-5` prototype collapsed real `~1e-10 V^2` covariances and was rejected.
- The primary table includes calibration, abstention, rest false commits, participant-level
  uncertainty, and latency rather than accuracy alone.
- Neural checkpoints and the result JSON encode the preprocessing/training/deployment
  settings needed to reproduce a subject model.

An initial protected 6/1/1 LOSO estimate is now complete. The remaining high-value work is
a full inner-LOSO/multi-seed estimate, an independently trained anchor ablation, external
public multi-session datasets, matched modern TTA baselines, continuous replay, and
repeated closed-loop evaluation before recruiting the intended clinical population.
