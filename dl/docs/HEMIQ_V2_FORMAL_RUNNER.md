# HemiQ-Field harmonized-v2 formal runner

The retained token `eeg-mi-cache-v2` is a versioned cache-schema identifier,
not the active package or distribution name.

Status: second independent-audit findings repaired and adversarially tested;
final no-edit re-audit pending and deliberately unlaunched.

This document describes the reproducible execution boundary for
`architecture.hemi_q_field` on the harmonized-v2 BNCI2014-004 cache. It is not
the historical HemiQ confirmation protocol. Every subject in this adapter has
already been opened during project development, so every eventual result is
development evidence.

## Fixed scientific scope

The formal roster is exactly:

- dataset `bnci2014_004`;
- supplied bipolar derivations in the exact order `C3`, `Cz`, `C4`;
- subjects 1 through 9;
- official chronological fold 0;
- seeds 7, 17, 27, 37, and 47; and
- 45 atomic score-blind prediction records.

Nominal cache coordinates are interface metadata only. The runner never uses
them as point-electrode geometry, for interpolation, or as evidence of
cross-montage continuity. All other benchmark datasets are principled N/A
cells for this architecture.

The cache contract is `eeg-mi-cache-v2/harmonized`: 128 Hz, the half-open
0.5-3.0 s interval, and 320 samples. Each trial independently produces an
8-30 Hz neural view plus four 8-12, 11-15, 14-20, and 20-30 Hz SPD teacher
views. The transform is deterministic and label-free. It does not estimate
statistics across trials or prediction batches.

## Frozen two-phase fit

For each subject and seed, selection fits the symmetric raw scaler and tangent
teacher on training rows only. The neural model starts from the job seed and
validation binary cross-entropy selects the duration. The untrained initial
state is a valid candidate, so the selected duration may be zero.

Reset/refit reconstructs the exact seeded neural initialization. It refits the
scaler and teacher on training plus validation rows and optimizes for exactly
the selected duration. The teacher coefficient retains the historical
240-epoch decay horizon; a shorter selected duration does not compress it.

After refit, the runner calls `predict_proba` exactly once on sessions `3test`
and `4test`. It publishes test row numbers and two-class probabilities. It
does not publish labels, decisions, correctness, confusion matrices, or
metrics.

## Immutable plan

`benchmark.hemiq_v2_grid build-plan` creates the run directories and publishes a
canonical mode-`0400` `plan.json` without replacement. The semantic validator
requires the exact 45-job roster on creation, load, worker, audit, and analysis
paths.

The plan binds:

- the outcome-free harmonized-v2 configuration byte count and SHA-256;
- the historical parent manifest identity without opening the outcome-bearing
  historical result at runtime;
- the complete transitive HemiQ runner/model/data/lease source closure;
- the SciPy SOS coefficient hashes and view schema;
- all nine cache archive byte counts and SHA-256 values;
- exact cache array identities, shapes, dtypes, channel order, and split row
  hashes;
- the UV project, lock, Python version file, executable, active venv, UV
  version, and exact direct installed dependency versions;
- Torch, CUDA, cuDNN, NumPy, and SciPy identities;
- the NVIDIA driver and complete physical GPU UUID roster; and
- the three-worker ceiling, DDP prohibition, one-UUID visibility contract,
  deterministic cuBLAS setting, and 50 GiB hard disk floor.

The plan is outcome-free. It contains no benchmark prediction, validation
outcome, test label, metric, or historical confirmation score.

## Shared workstation isolation

Formal workers run only from the UV-created project venv. They do not invoke
`uv sync`, install a package, write a system environment, use DDP, or expose
more than one physical GPU. Each worker must be launched with
`CUDA_VISIBLE_DEVICES` equal to one planned physical UUID and uses logical
device `cuda:0`.

Every worker acquires `benchmark.project_gpu_leases` with track scope
`hemiq-v2`. The shared registry enforces at most three active project GPU
workers, leaving one of the four lab GPUs available to other users. The
runner holds the authoritative lease guard across these mutation boundaries:

- physical-UUID preflight receipt publication;
- claim publication and stale partial recovery;
- failure receipt publication;
- claim release; and
- final result-directory publication and claim retirement.

The lease receipt binds the project root, run root, plan SHA-256, physical GPU
UUID, track scope, owner, nonce, lease inode, and persistent registry fence.
A post-publication guard failure may quarantine only the exact published
device/inode and canonical bytes. If another process replaces the canonical
name after the guard yields, cleanup reports contention and preserves both the
replacement authority and any displaced owned inode.

## Synthetic CUDA preflight

Every physical GPU used by the run requires a mode-`0400`, UUID-specific
preflight receipt. The preflight revalidates the complete static plan identity,
the active UV environment, all cache and split identities, the CUDA/driver
identity, UUID isolation, and the 50 GiB disk floor. It then uses only
synthetic balanced signals to verify:

- exact outcome-free configuration identity;
- CUDA availability;
- deterministic seeded reset;
- bitwise forward/backward/optimizer replay;
- bitwise probability replay;
- exact sagittal reflection anti-equivariance;
- finite float32 SPD covariance positivity;
- exact supplied-bipolar input order; and
- preservation of the zero-epoch path.

A worker cannot claim a benchmark job unless the exact preflight receipt for
its physical UUID exists and validates.

## Claims, recovery, and power loss

A claim is a canonical mode-`0400` JSON leaf. Its descriptor is exclusively
locked for the job lifetime and its inode, nonce, bytes, owner, preflight
receipt, disk guard, and GPU lease receipt are revalidated before commit.

If a process dies, the lock is released by the kernel. A later lease-holding
worker may move that exact unlocked claim and its job-owned abandoned stage to
quarantine before publishing a new claim. A locked claim is never stolen.
Unexpected names, object types, aliases, or replaced inodes fail closed.
Recovery always opens and validates the canonical claim before inspecting any
job-owned result stage. Thus a second worker cannot quarantine a live writer's
stage: the live claim blocks recovery first, and a completed result must bind
the exact claim inode, nonce, bytes, and receipts.

If the result rename completed but the process died before deleting its
original claim, resume first validates the complete immutable result. It then
locks and validates the surviving claim, requires its bytes, inode, nonce, GPU
receipt, and full authority to equal the claim receipt embedded in that
completion, and atomically quarantines that exact inode as `completed_claim`.
A still-locked claim remains live even when a result directory is visible and
is never stolen.

Plan, preflight, failure, and claim publication each have an exact hidden-stage
grammar. Restart scans those grammars through anchored descriptors under the
applicable coordination and GPU guards. Exact unlocked stages left by
SIGKILL are quarantined by inode; live, malformed, aliased, or unexpected
stages fail closed. Real subprocess SIGKILL tests exercise all four restart
paths rather than depending on exception cleanup. Generic JSON stages are
mode `0400` before rename. If rename succeeds but the parent fsync fails, or a
rename helper moves and then raises, the exact inode is moved back to its
same-parent stage name before cleanup; no writable canonical JSON authority is
left behind. Preflight, claim, failure, generic-stage, and failed-result-stage
cleanup all carry their exact published inode identity; regular JSON cleanup
also verifies the exact bytes through its held descriptor after the canonical
name has first been hidden. Post-yield A-to-B regressions require replacement
authorities to survive without being mislabeled as owned quarantine evidence.

The producer builds `record.json`, `predictions.npz`, and `completion.json` in
a unique directory under the final `records` parent. It fsyncs all three
mode-`0400` leaves, fsyncs the stage directory, changes the directory to mode
`0555`, fsyncs it again, and then performs an anchored no-replace same-parent
rename. The stage inode and device are embedded in the completion record.
Validation rejects a writable or replaced final directory.

Immediately before the result rename, the worker rebinds the complete planned
source closure, active UV identity, and frozen HemiQ configuration. It repeats
those checks after descriptor-bound completion validation and before returning
success. A failure after rename, including parent fsync, inode inspection,
move-then-raise behavior, lease exit, validation, or the second identity
rebind, hides the exact result inode from its canonical name.

The completion record seals:

- exact record and prediction byte counts and SHA-256 values;
- the exact planned held-out row count, ordered row digest, and unique rows;
- exact `int64` row and `float64 [n_test, 2]` probability array contracts,
  including finite `[0,1]` entries and unit row sums;
- the full deleted-claim authority and its canonical byte hash;
- claim and commit GPU lease receipts;
- the claim and commit 50 GiB disk guards;
- the physical GPU UUID;
- the plan and job identity; and
- the immutable final directory inode, device, and mode.

On the lab filesystem, cross-parent rename of a read-only directory is denied.
For that reason, result stages live under the final `records` parent and use a
same-parent atomic rename. Quarantine code changes permissions only on the
descriptor-bound artifact after it has already been excluded from publication.
The canonical source is first renamed to a hidden name within its current
parent. Only then may quarantine create or open a cross-parent destination or
temporarily change directory permissions.

## Exact NPZ handling

NumPy accepts duplicate ZIP member names and `archive.files` alone is
ambiguous. The HemiQ runner therefore validates both layers:

1. the raw ZIP `infolist()` must have the exact ordered member names and
   cardinality, with no duplicate, directory, encrypted, or empty member; and
2. NumPy's `archive.files` must match the exact ordered key list.

Cache archives require, in order, `x`, `y`, `positions`, `channel_names`,
`sessions`, `runs`, and `identity`. Prediction archives require `test_rows`
then `probabilities`. The latter arrays must be int64 and float64 with exact
plan-bound shapes and hashes.

## Label boundary and aggregate-only analysis

`benchmark.hemiq_v2_analysis` has no training entry point. It first acquires an
exclusive lock on the same coordination file respected by worker claims and
commits. While holding it, the analyzer requires:

- the exact plan and run-root directory roster;
- no claims, failures, partials, or unreviewed quarantine;
- exactly 45 non-writable result directories;
- exact result leaf rosters and completion seals;
- valid claim, commit, preflight, cache, split, source, UV, CUDA, driver, and
  physical-UUID bindings; and
- an exact descriptor-read file roster SHA-256.

The fence retains open descriptors and full identities for the run root,
authority directory, coordination lock, and analysis marker. Each assertion
binds those descriptors back to the live canonical namespace before and after
the marker snapshot. Audit traversal duplicates the pinned root descriptor,
holds every run-directory descriptor, and verifies exact rosters and full stat
fingerprints at the end.

The analyzer writes and fsyncs `audit.json` into its private output stage
before it indexes a held-out label. It then recomputes metrics from the sealed
probabilities and the exact plan-bound test rows. No trial-level label or
correctness vector is published.

Result validation starts from the pinned run-root descriptor and holds the
`records` directory, job directory, and UUID-bound preflight receipt. It opens
all three result leaves before reading any, then rebinds each held descriptor
to its canonical name and rechecks full directory and leaf stat fingerprints
after all reads. The exact bytes used to decode probabilities also produce the
record, prediction, and completion digests returned to the audit. During
validation, decoded arrays from that same pinned byte snapshot must match the
plan's exact test-row manifest and trial count, native dtypes, binary shape,
unique increasing rows, finite probability range, and unit simplex. Coherently
resealed NaN, infinity, wrong-count, wrong-shape, wrong-dtype, duplicate-row,
out-of-range, and bad-sum packages are rejected before metrics. During
metric computation, every job's new descriptor-held snapshot must equal the
corresponding persisted pre-label audit entries. Thus a transient A-to-B
metric read, leaf toggle, or coherent live package-B swap cannot detach
metrics from ledger bytes.

Before the first held-out label is indexed, and again under the same exclusive
fence immediately before publication, the analyzer revalidates the current
HemiQ grid, analysis, model, outcome-free configuration, full planned source
closure, active UV venv, Python, lockfiles, and exact metric dependency
versions.

The aggregate output includes:

- per-subject/seed fold-concatenated metrics;
- the arithmetic seed mean within each participant; and
- the equal-participant BNCI2014-004 mean.

Metrics are accuracy, balanced accuracy, macro-F1, Cohen's kappa, negative
log-likelihood, Brier score, and 15-bin expected calibration error. This
procedure-specific table must not be pooled with common-recipe, native
transfer, author-recipe, deterministic-control, or historical-confirmation
tables.

Persisted analysis validation enforces exact row schemas and order, plan-bound
trial counts, valid metric ranges, exact seed means within each participant,
the exact equal-participant recomputation, and the frozen aggregation
description. Impossible or internally inconsistent JSON cannot be republished
as a valid analysis.

The output directory is staged under its destination parent. `manifest.json`
seals `analysis.json`, `audit.json`, both CSV tables, and the Markdown report
before the stage is changed to mode `0555` and renamed without replacement.
The analyzer rechecks the exclusive fence and the complete input roster both
before and after publication.

The analysis rename explicitly verifies that the stage and destination
descriptors are the same parent. The expected published device/inode is
recorded as the first operation after the rename returns. Any later parent
fsync, publication verification, final audit, fence exit, or lock-release
failure atomically hides that exact inode under the same parent, leaving no
visible destination. Publication verification retains the opened published
directory descriptor and all six member descriptors through fence release and
the final return boundary. After every member read and immediately before
return, it rebinds the exact canonical directory name/inode, parent
fingerprint, full directory fingerprint, exact roster, and every member
fingerprint. Coherent A-to-B and A-to-B-to-A substitutions are rejected. If B
is the current canonical authority, rollback refuses to rename or quarantine
B. Parameterized fault and substitution tests cover these post-rename points.
Move-then-raise injection at the analysis rename is also rebound to the known
stage inode and hidden exactly.

## Reproducible commands after independent approval

The following forms document the intended interface. They are not authority
to launch the track, and no command in this section has been used to create a
formal plan or benchmark result.

```bash
uv run --active --offline --no-sync python -m benchmark.hemiq_v2_grid \
  build-plan \
  --project-root /home/hanafy/neuralnetwork \
  --cache-root /path/to/cache \
  --run-root /path/to/hemiq-v2-run
```

Each approved worker would use one different planned physical UUID:

```bash
CUDA_VISIBLE_DEVICES=GPU-... \
uv run --active --offline --no-sync python -m benchmark.hemiq_v2_grid \
  worker \
  --project-root /home/hanafy/neuralnetwork \
  --cache-root /path/to/cache \
  --run-root /path/to/hemiq-v2-run \
  --gpu-uuid GPU-... \
  --device cuda:0
```

Only after 45 completions and a separate audit approval:

```bash
uv run --active --offline --no-sync python -m benchmark.hemiq_v2_analysis \
  --project-root /home/hanafy/neuralnetwork \
  --cache-root /path/to/cache \
  --run-root /path/to/hemiq-v2-run \
  --destination /path/to/fresh-hemiq-v2-analysis
```

## Current evidence and remaining gate

The existing lab UV environment passes 113 HemiQ tests: 14 adapter tests and
99 runner/analyzer tests. The runner suite includes real subprocess SIGKILL
recovery for plan, preflight, failure, and claim stages; completed-result/live-
claim exclusion and exact unlocked-claim recovery; descriptor-snapshot ledger
enforcement with a transient A-to-B-to-A attack; forged and boolean runtime
identity rejection; source/UV drift at both analysis boundaries; and
parameterized exact-inode cleanup after every post-rename fault. The second
repair adds parent-fsync and move-then-raise JSON rollback, live-claim/stage
binding, run-root replacement rejection, hide-before-chmod quarantine,
pre/post result identity rebinds, coherent package-swap rejection, and
persisted-metric consistency attacks. The third repair adds exact
inode-and-byte post-yield cleanup, coherently resealed prediction-array
corruption, and retained-descriptor analysis publication substitution tests.
Twelve execution-matrix tests remain a separate required suite. A leased
synthetic CUDA replay is rerun after every final source freeze. No dependency
was installed or changed. No formal plan, benchmark prediction, score, or
aggregate has been created.

Before launch, an independent no-edit audit must examine the frozen hashes,
rerun all tests with CUDA visible, and attempt plan, path, lease, cache,
duplicate-member, power-cut, result, and analysis forgery. A passing
implementation audit does not turn later development scores into confirmation
evidence and does not support an SOTA or clinical claim.
