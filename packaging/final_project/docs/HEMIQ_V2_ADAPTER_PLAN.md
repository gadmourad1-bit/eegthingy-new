# HemiQ-Field harmonized-v2 adapter plan

Status: **model adapter plus the score-blind formal runner/analyzer implemented;
the first independent audit's findings are repaired and adversarially tested;
independent re-audit, formal plan freeze, and benchmark execution still
pending**.

This document defines the boundary for evaluating
`architecture.hemi_q_field` in the comprehensive benchmark without rewriting
the historical HemiQ confirmation study. The historical development artifact,
frozen manifest, confirmation result, and permanent confirmation receipt under
`deepnet/results/hemi_q/` are read-only provenance. A harmonized-v2 run is a
new, development-only procedure and must be reported separately from that
confirmation result.

## Eligible scope

HemiQ-Field is eligible only for `bnci2014_004`. Its inputs are the three
provided bipolar derivations named `C3`, `Cz`, and `C4`; their nominal
coordinates must never be interpreted as three independent point electrodes.
All other benchmark datasets are principled N/A cells, not missing scores.

The proposed grid is:

- subjects 1 through 9;
- official chronological fold 0;
- seeds 7, 17, 27, 37, and 47; and
- exactly 45 atomic prediction records.

Every subject has already been opened during prior project work. Consequently,
all 45 records are development evidence. Subjects 5 through 9 do not regain
confirmation status in this adapter.

## Harmonized input adaptation

The source is the immutable `ieee-mi-cache-v2/harmonized` BNCI2014-004 cache:
128 Hz, the half-open 0.5-3.0 s interval, 320 samples, supplied bipolar order,
and no point-electrode re-reference. The adapter must require the exact channel
tuple `("C3", "Cz", "C4")` and must ignore the nominal coordinate array for
spatial interpolation or continuity.

Starting from each cached trial independently, construct:

1. an 8-30 Hz broadband view using a frozen fourth-order
   `scipy.signal.butter(..., output="sos", fs=128)` band-pass followed by
   `scipy.signal.sosfiltfilt` on the time axis and post-filter trial demeaning;
2. four teacher bands at 8-12, 11-15, 14-20, and 20-30 Hz using the same
   order, implementation, application direction, and demeaning rule; and
3. trace-shrunk SPD covariance matrices from each teacher band, with
   positivity verified after conversion to the stored float32 dtype.

No transform may use labels, cross-trial statistics, validation/test
statistics, or a prediction batch. The filter coefficients, SciPy version,
view schema, and implementation source hashes belong in the immutable plan.

This is not an exact reproduction of the legacy preprocessing. The historical
study resampled continuous recordings to 125 Hz, applied minimum-phase MNE
filters before epoching, and used an inclusive 0.5-2.5 s, 251-sample window.
The v2 adapter instead retains the common 128 Hz, 320-sample benchmark window
and applies a deterministic trial-local transform. Reports must call it the
**HemiQ-Field harmonized-v2 adaptation**, not the historical HemiQ recipe.

## Frozen model adaptation

The parent architecture and hyperparameters come from
`deepnet/results/hemi_q/hemi_q_final_frozen_manifest.json`. Only the input
sampling contract changes:

- `sfreq`: 125.0 to 128.0;
- `n_times`: 251 to 320; and
- `device` and `seed`: bound per formal worker/job.

The filter count, kernel size, stride, local pooling, width, learned frequency
and bandwidth limits, optimizer, teacher weight, teacher fraction,
regularization, gradient clipping, maximum epoch count, patience, and
classification threshold remain unchanged. The adapted configuration must
exist as an outcome-free JSON file with exact schema, byte count, SHA-256, and
the parent-manifest SHA-256. Runtime code must not open any historical result
JSON to obtain configuration.

The implemented score-free configuration is
`configs/hemiq/hemiq_field_harmonized_v2.json`: 1,054 bytes, SHA-256
`0474b53b06c1f4bc56659f0cc75ff3f908aac67306652714f533c83289af128f`.
It cites the historical parent manifest by path, 6,688-byte size, and SHA-256,
but `ieee_mi.hemiq_v2_model.load_frozen_config` does not open that
outcome-bearing artifact at runtime. The copied parent settings are validated
inside the outcome-free source/config closure, with only `sfreq`, `n_times`,
`device`, and `seed` allowed to differ.

## Two-phase evaluation

For every subject/seed:

1. **Selection phase.** Fit the symmetric raw scaler and optional tangent
   teacher on training rows only. Initialize the neural network from the
   job seed, optimize training rows only, and select one duration by validation
   binary cross-entropy. Validation balanced accuracy is diagnostic only.
2. **Reset/refit phase.** Reconstruct the exact seeded neural initialization.
   Refit the symmetric scaler and teacher on train plus validation only.
   Optimize for exactly the selected number of epochs. Preserve the original
   240-epoch teacher-decay horizon so refit epoch `e` uses the same teacher
   coefficient as selection epoch `e`; do not compress the schedule to the
   selected duration.
3. **Prediction phase.** Use the sole HemiQ logit and fixed zero threshold.
   Call the probability predictor exactly once on held-out sessions
   `3test` and `4test`. Publish row indices and probabilities, but no labels or
   scores.

The initial untrained checkpoint is an allowed selection result. If it wins,
the selected refit duration is zero epochs and that outcome must be reported
rather than silently coerced to one.

## Artifact and analysis boundary

The runner must use the same fail-closed standards as the other formal tracks:

- exact 45-job semantic plan validation on create, resume, worker, audit, and
  analysis paths;
- UV-managed virtual environment and locked direct dependencies;
- full transitive source, configuration, cache, split, CUDA, and environment
  identities;
- no-follow containment and rejection of symlinks, FIFOs, sockets, devices,
  hardlink aliases, and unexpected files;
- fenced claims with live ownership checks and power-cut recovery;
- atomic score-blind predictions and immutable completion receipts;
- a quiescent audit before any label is opened; and
- fresh metric recomputation under a worker-respected publication fence.

The implemented formal producer is
`ieee_mi.hemiq_v2_grid`. It accepts only the explicit 45-job roster above.
`ieee_mi.hemiq_v2_analysis` is an aggregate-only consumer: it obtains the
exclusive worker-respected fence, persists and fsyncs the exact label-free
roster audit, and only then indexes held-out labels. Its publication contains
per-subject/seed, per-subject, and equal-participant summaries, not trial-level
labels. `scripts/hemiq_v2_cuda_replay.py` is a synthetic-only leased CUDA
replay utility and has no benchmark loading or scoring path.

Every worker requires a physical UUID-specific synthetic CUDA preflight. The
preflight, claim publication, failure publication, claim release, and final
result rename are protected by the shared project GPU lease. Result directories
are staged under the destination parent, fsynced, changed to mode `0555`, and
then renamed without replacement. Their stage inode is embedded in the
completion object. Prediction leaves are mode `0400`; completion validation
rejects a writable or replaced directory.

The plan binds the complete transitive HemiQ source closure, exact outcome-free
configuration, UV project/lock/Python identities, direct installed versions,
SciPy filter contract, all nine cache archives and split row hashes, Torch/CUDA
versions, NVIDIA driver, and the complete physical GPU UUID roster. Workers
require exactly one UUID through `CUDA_VISIBLE_DEVICES`, reject DDP variables,
enforce the project-wide three-worker cap, and retain the 50 GiB hard disk
floor. Cache and prediction NPZ readers validate the exact ordered raw ZIP
member roster before also validating NumPy's `archive.files`; duplicate ZIP
members are rejected.

Resume recognizes exact hidden stages for plan, preflight, failure, claim, and
result authorities. Real subprocess SIGKILL tests leave each of the first four
stages without executing exception cleanup and prove that restart quarantines
the exact unlocked inode. Malformed or live stages fail closed. When a result
was published before a crash but its original claim survived, resume validates
the result first, refuses a still-locked claim, and only then validates and
quarantines an unlocked claim whose complete receipt exactly matches the
completion.

Completion validation reads record, prediction, and completion leaves from
the descriptor-held result directory. The byte snapshots used to decode
probabilities also generate the audit ledger digests. Metric-time reads must
match every job's persisted three-leaf audit entries before classification
metrics run, which rejects a transient A-to-B read even if an attacker restores
A before the final audit. The decoded prediction snapshot must also match the
plan's exact test trial count and ordered row digest, native row/probability
dtypes, binary probability shape, unique rows, finite `[0,1]` values, and unit
row sums; a matching forged record and completion seal cannot relax those
contracts.

Analysis must keep this procedure separate from the common raw-trial,
author-recipe, deterministic-control, historical-confirmation, and native
transfer tables. It may report per-subject/seed accuracy, balanced accuracy,
macro-F1, Cohen's kappa, NLL, Brier score, and calibration error; the primary
aggregation is fold concatenation, seed mean within participant, and equal
participant mean for BNCI2014-004.

The analyzer verifies the current HemiQ grid, analysis, model, outcome-free
configuration, complete planned source closure, active UV venv, Python,
lockfiles, and exact metric dependencies before opening labels and again under
the fence immediately before rename. It requires a same-parent `0555`
publication rename. The published inode is recorded immediately after rename;
any subsequent fsync, verification, final-audit, fence-exit, or release failure
atomically hides that exact inode and leaves no visible destination. It retains
the published directory and all artifact descriptors, then rebinds the exact
canonical inode, parent and directory fingerprints, exact roster, and every
member after each read and immediately before return. Coherent A-to-B and
A-to-B-to-A swaps fail closed, while exact-inode rollback leaves a replacement
B untouched.

## Promotion requirements

Before execution, an independent audit must verify:

- exact anti-equivariance of the adapted 320-sample model;
- bitwise repeatability of forward, backward, and one optimizer step on CUDA;
- reset/refit initial-state equality;
- source-only scaler and teacher fitting;
- returned float32 covariance positivity on zero, constant, rank-deficient,
  and tiny-amplitude inputs;
- exactly one held-out prediction call;
- all path, plan-forgery, result-forgery, outcome-alias, race, and power-cut
  adversarial tests; and
- no write to or interpretation change of the historical confirmation
  artifacts or receipt.

The current lab UV suite has 113 passing HemiQ tests: 14 model-adapter tests
and 99 formal-runner/analyzer tests. The model tests cover configuration identity,
historical-result isolation, symlink/hardlink rejection, an adversarial
ancestor-path swap against the descriptor-pinned configuration reader,
degenerate float32-SPD views, trial-local transforms, zero-epoch refit, reset
equality, source-only teacher fitting, exact reflection action, and
deterministic forward/backward/optimizer replay. A seed-7 regression also
covers the case where a permitted runtime override equals the historical
parent value.

The runner/analyzer tests additionally cover the exact 45-job cardinality,
strict typed numeric identities, forged GPU driver/index/memory/name rows,
immutable-plan drift, symlink/hardlink/ancestor rejection, raw-ZIP duplicate
and order ambiguity, source closure, preflight and claim lease loss,
live/stale/completed claims, analysis exclusion, score aliases, zero-epoch
propagation, source-only reset/refit, exactly one held-out predictor call,
power loss before result rename, post-commit lease loss, writable/replaced
final directories, and a complete synthetic 45-directory quiescent audit
without label access. Real subprocess SIGKILL tests cover the plan, preflight,
failure, and claim atomic stages. A descriptor-ledger attack test performs a
valid transient A-to-B metric read and restores A; the persisted roster rejects
it. Parameterized tests inject failures after analysis rename at parent fsync,
publication verification, final fence validation, final audit, and fence
release, and require the exact published inode to be hidden. Separate tests
force source/UV drift before labels and before rename. Second-pass regressions
also cover JSON parent-fsync and move-then-raise rollback, claim-before-stage
binding, canonical run-root replacement, hide-before-chmod quarantine, worker
source/UV/config rebinds, simultaneous held result leaves with a coherent
package swap, and exact persisted metric/trial/aggregate consistency. The
third-pass regressions add post-yield A-to-B replacement at preflight, claim,
and failure cleanup; coherently resealed NaN/Inf/count/shape/dtype/range/sum/
duplicate-row prediction attacks; and coherent analysis publication A-to-B
and A-to-B-to-A substitution. The suites ran inside the existing UV-created
virtual environment with `--offline --no-sync`; no package was installed or
changed. These tests are implementation evidence, not a substitute for the
required independent re-audit or formal execution.

The non-formal synthetic CUDA replay also passed under the shared project GPU
lease on physical UUID
`GPU-576ef18e-b48f-c787-e91d-f93658b27b8e` (one lab RTX A5000) with Torch
2.6.0+cu124/CUDA 12.4. It verified seeded reset, bitwise forward/backward/
optimizer replay, bitwise probability replay, exact reflection
anti-equivariance, float32 SPD positivity, the supplied-bipolar order, and
zero-epoch preservation. It used synthetic balanced signals, produced no
benchmark score, and remains distinct from formal execution.

No formal HemiQ-v2 plan, benchmark prediction, score, or aggregate has been
created. The track must remain unlaunched until a separate no-edit audit
approves the frozen source and tests.
