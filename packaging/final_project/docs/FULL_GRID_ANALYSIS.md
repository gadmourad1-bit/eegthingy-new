# Exact 43-model common-grid protocol

`ieee_mi.full_grid` and `ieee_mi.full_grid_analysis` define the formal
common-recipe benchmark. The scope is exactly 43 registered architectures,
five already opened EEG motor-imagery datasets, their frozen folds, and five
training seeds. It produces exploratory development evidence only. It is not
independent confirmation evidence and does not establish a global or
state-of-the-art claim.

## Frozen scope

The formal Cartesian product contains:

- 43 architectures;
- 448 dataset/subject/fold units across `local_exp4`, `bnci2014_001`,
  `bnci2014_004`, `cho2017`, and `physionet_mi`; and
- seeds `7`, `17`, `27`, `37`, and `47`.

Therefore the exact job count is:

```text
43 × 448 × 5 = 96,320 atomic prediction jobs
```

The immutable track label is `common_recipe_43_only`. A formal plan cannot
append another architecture, replace the executor, change a training
hyperparameter, add a seed, or change the dataset/fold product. Separate
author-recipe, transfer, procedure, or control experiments must remain in
separately labelled result tables.

The 43 architecture identifiers, in frozen tie-break order, are:

```text
scope
free_scope
cardinal
free_cardinal
cardinal_dynamics
cardinal_dynamics_compact
cardinal_dynamics_extended
cardinal_dynamics_sinc
cardinal_dynamics_sinc_residual
cardinal_dynamics_sinc_extended
eegnet
shallow
deep4
eegconformer
eegconformer_compact
atcnet
atcnet_aggressive_pool
fbcnet
cardinal_fbc
cardinal_fbc_extended
cardinal_fbc_corr
cardinal_fbc_corr_extended
cardinal_fbc_compactdyn
cardinal_fbc_compactdyn_extended
cardinal_fbc_compactdyn_scale010
cardinal_fbc_compactdyn_scale010_extended
cardinal_fbc_compactdyn_scale025
cardinal_fbc_compactdyn_scale025_extended
cardinal_fbc_micro
cardinal_fbc_micro_extended
cardinal_fbc_physical
cardinal_fbc_physical_extended
eegtcnet
fbmsnet
cardinal_fbms
cardinal_fbms_extended
cardinal_mix
cardinal_mix_drop
ctnet
ctnet_compact
eegsym
eegsym_wide
tcformer
```

## Dedicated UV environment

The benchmark never installs into the workstation's system Python. From the
project root, create and synchronize the project-local virtual environment:

```bash
uv venv --python 3.12 .venv
uv sync --frozen --no-default-groups
```

The formal source check requires `pyproject.toml` and `uv.lock` to contain the
complete runtime closure for Braindecode, MNE, MOABB, NumPy, scikit-learn,
SciPy, Torch, and the matching TorchAudio build. Pytest belongs only to the
test group, and
`[tool.uv] default-groups=[]` prevents test tooling from leaking into formal
runs. Plan creation fails closed if this UV contract or its lock is
incomplete.

After that explicit synchronization, formal commands add `--no-sync`; they do
not install, remove, or repair packages while an experiment is running. Every
formal command uses the interpreter selected by `uv run`; worker subprocesses
inherit that exact `sys.executable`. The source identity hashes the runner,
model/training/data/config sources, the TCFormer implementation, the shared
physical-GPU lease implementation, `pyproject.toml`, and `uv.lock`.

## Mandatory no-score CUDA attestation

Before `run`, invoke the construction/backpropagation preflight on one idle
GPU. If the run root is new, this command first creates its immutable
`plan.json`; otherwise it validates the existing plan and runtime:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --frozen --no-default-groups --no-sync \
python -m ieee_mi.full_grid preflight \
  --run-root /absolute/path/to/common-grid-run \
  --cache-root /absolute/path/to/data_cache \
  --gpu 0
```

The preflight performs exactly 172 checks: 43 architectures against four
canonical shape/class contracts. Each check constructs the model, verifies
finite logits and nonzero finite gradients, takes one AdamW step, and verifies
a finite post-step forward pass. The 172-check CUDA child opens only cache
metadata needed for shape, channel, and position validation; it does not open
labels, reconstruct test outcomes, or compute a score. The parent command's
plan creation/resume gate separately validates the complete cache and frozen
split identities before spawning that child. That planning step may read
labels needed to reconstruct splits, but it still computes no outcome metric.

A successful formal gate is published once at fixed paths:

```text
common-grid-run/preflight/report.json
common-grid-run/preflight/receipt.json
```

Both files are canonical, read-only, single-link regular files. The report
binds the exact ordered 172 rows to the immutable plan hash, source and
environment identities, four cache-array digests, physical GPU UUID, and a
guarded project-GPU lease receipt. The second file binds the report SHA-256 to
the plan and a completion-time guarded lease receipt. A power loss after the
report rename may leave only `report.json`; rerunning `preflight` validates
those exact old bytes and creates only the missing receipt under a new active
lease. Exact `.report.json.stage-*` and `.receipt.json.stage-*` leaves from a
write-stage power loss are descriptor-checked and can be removed repeatedly;
an unknown entry prevents all stage cleanup and is preserved. A receipt
without a report, any changed report, any unknown preflight entry, or any
digest/schema mismatch fails closed.

A direct formal preflight call must see exactly `cuda:0`, with both
`CUDA_VISIBLE_DEVICES` and `FULL_GRID_GPU_UUID` equal to the leased physical
UUID. Torch must expose exactly one CUDA device and its name must equal the
frozen inventory row. If the shared GPU-lease guard fails while exiting after
publication, cleanup reacquires the persistent project registry fence and
moves only the exact report/receipt inodes and payloads demonstrably published
by that invocation. During report-only recovery only the exact new receipt is
invalidated, preserving the previously fsynced report for a later safe retry.
The initial canonical preflight snapshot is taken only after that global guard
is held. A serialized loser validates and accepts a completed winner; cleanup
treats any inode or payload mismatch as contention and never moves the winner.
The orchestrator, every worker, every formal claim and commit, the exact
auditor, and the analyzer all require the completed canonical pair.

## Formal run and resume

Run no more than GPUs 0, 1, and 2:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --frozen --no-default-groups --no-sync \
python -m ieee_mi.full_grid run \
  --run-root /absolute/path/to/common-grid-run \
  --cache-root /absolute/path/to/data_cache \
  --gpus 0,1,2 \
  --cpu-threads 4
```

The same command is the resume command after an orderly pause or power loss.
The canonical plan is reused, completed atomic jobs are validated rather than
rerun, and durable claim tombstones repair the narrow rename-before-release
power-loss boundary. A malformed or foreign-host claim is never stolen based
on age.

Useful read-only checks are:

```bash
uv run --frozen --no-default-groups --no-sync \
python -m ieee_mi.full_grid status \
  --run-root /absolute/path/to/common-grid-run

uv run --frozen --no-default-groups --no-sync \
python -m ieee_mi.full_grid audit \
  --run-root /absolute/path/to/common-grid-run \
  --output /absolute/path/to/new/common-grid-audit.json
```

### Shared-workstation safety

- The orchestrator resolves indexes to physical NVIDIA UUIDs and gives every
  worker exactly one CUDA-visible physical device.
- All formal experiment tracks share the verified project-root registry
  `.ieee-mi-project-gpu-leases`. It permits one worker per physical UUID and
  at most three active GPU workers across the entire project, not merely per
  run directory.
- Preflight publication, every final claim write, and every final record
  rename occur while the shared registry guard is held. Claims and records
  persist the same canonical lease receipt; records also persist the exact
  preflight-report digest.
- GPU 3 or any fourth device is outside the common-grid launch surface.
- A new claim requires an idle/safe GPU snapshot, no foreign compute process,
  and acceptable foreign memory use. Publication repeats the resource probe.
- Every claim and commit enforces an exact, non-overridable 50 GiB free-space
  floor.
- Claims hold a shared persistent run-publication fence for their complete
  lifetime. Analysis requires the exclusive fence.
- Persistent fences, claims, completions, failure receipts, preflight files,
  and published artifacts are descriptor/inode-checked, non-symlink,
  single-link regular files with exact schemas. Raw cache and prediction NPZ
  files additionally require an exact ordered ZIP-member contract before
  NumPy can resolve logical names; duplicates, traversal names, reordering,
  or extra members fail closed.
- A guard failure after a claim, record, or preflight rename atomically moves
  the exact new artifact to its plan-bound quarantine category. Structurally
  safe, immutable, plan-bound quarantine evidence is retained in a sorted
  forensic ledger and does not permanently block an otherwise exact rerun.
  Unknown, writable, aliased, special, or schema-invalid quarantine evidence
  remains unresolved and makes the audit fail.
- Claim publication records the new inode at the atomic writer boundary.
  Failures during strict reread, schema validation, read-only checking, fence
  checking, or guard exit quarantine only that exact inode, including failures
  before the in-memory `Claim` object exists. A replacement race winner is
  never moved.
- Corrupt-record, stale/completed-claim, and stale-partial recovery carries
  the validated `(st_dev, st_ino)` through the quarantine boundary. A missing
  or different canonical inode is normal contention, never authority to move
  the replacement.
- Forensic-ledger construction walks through held run-root, quarantine,
  category, and artifact descriptors. It rebinds every artifact name after
  traversal and re-digests the anchored candidates at the end of the complete
  audit; a category or artifact-name swap cannot pair one path with another
  inode's digest.

## Score-blind production records

Each atomic job is identified by
`dataset/model/subject/fold/seed`. Training uses only the frozen source split:
train rows fit the selection model and scaler, validation rows select the
epoch count, then a reset model is refit on train plus validation. Test rows
are used once for prediction.

Production records contain row identities, normalized class probabilities,
source-only fitting provenance, model-state hashes, parameter count, timing,
peak CUDA allocation, physical GPU UUID, and both resource snapshots. They do
not contain test labels or test performance. Every formal claim and record
also binds the guarded project-GPU lease receipt and the immutable preflight
report SHA-256. Record and completion schemas are exact and recursively reject
outcome-like metadata aliases.

Every commit:

1. revalidates source, environment, cache, and split identities;
2. verifies the exact owned claim and persistent fence;
3. writes an invisible partial directory;
4. fsyncs its leaves, fsyncs the directory, changes the staged directory to
   `0555`, and fsyncs that mode before publication;
5. atomically renames the same staged inode without replacement;
6. requires the final directory to be that exact inode and non-writable,
   reopens and validates all completion bytes, then rechecks claim ownership;
   and
7. quarantines the canonical record if any post-rename claim, lease, inode,
   mode, or validation postcondition fails.

Completion validation opens and retains descriptors for exactly
`completion.json`, `predictions.npz`, and `record.json` before reading any
member. After all three reads it rebinds every held descriptor to its exact
name, rechecks the exact entry set, and compares the complete directory
fingerprint including nanosecond modification and change times. A coherent
held-package/live-package substitution therefore fails rather than returning
bytes detached from the canonical record.

Darwin does not permit renaming a directory whose own owner-write bit is
absent. Local macOS validation therefore performs a fenced, descriptor-held
compatibility transition after proving the exact `0555` inode and leaf set.
Linux first attempts the continuously sealed rename. Some Linux/NFS mounts
reject that operation with `EACCES` or `EPERM`; only either of those errors
permits the same transition after a second exact-tree and inode check. All
success and failure paths restore and fsync `0555` through the original held
descriptor. Other errors never retry. Final record publication requires
exactly `record.json`, `predictions.npz`, and `completion.json`; quarantine
also verifies the exact source inode and recursively seals it after the move.

## Fail-closed analysis

Analysis is a separate command and has no formal setting overrides:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
uv run --frozen --no-default-groups --no-sync \
python -m ieee_mi.full_grid_analysis \
  --run-root /absolute/path/to/common-grid-run \
  --cache-root /absolute/path/to/data_cache \
  --output-dir /absolute/path/to/new/common-grid-analysis
```

The analyzer first validates the complete 96,320-job score-blind tree without
opening labels. It rejects missing, corrupt, extra, aliased, special,
partially published, claimed, or otherwise non-quiescent state. Only then does
it reopen plan-bound caches, reconstruct the frozen splits, and join test-row
identities to labels in memory.

The public analysis API accepts only the run root, cache root, and a new output
directory; it does not accept caller-supplied results or statistical settings.
Before computing any result it acquires the exclusive persistent run fence,
then validates the preflight pair and recomputes the entire analysis under
that lock while revalidating source/environment/cache/split identities and
every input checksum. Each job's ledger hashes are computed from the same
descriptor-held completion, record, and prediction byte buffers used for its
metrics; all three file descriptors remain open together until every byte,
name binding, exact-entry check, and directory fingerprint has been
revalidated. Ledger construction never rereads a detached pathname snapshot. At
the close of computation and again after publication, every ledger row is
checked in exact plan order by one full descriptor-held completion validation,
including the frozen preflight digest. Independent per-file path hashes are
not used, preventing a toggle-package attack from mixing three transient
versions. Publication uses a second persistent output fence, an invisible
staging directory, fsync, and an atomic no-replace directory rename.
After that rename, while both fences are still held, the publisher reopens the
same staged inode, requires the exact `ANALYSIS_FILENAMES` leaf set, re-reads
every unique read-only file, compares every byte to the staged payload,
reparses the canonical manifest, and rechecks both fence identities. Analysis
publication and invalidation use the same NFS-safe held-descriptor transition.
If a move succeeds but resealing, fsync, or final verification raises, source
name disappearance binds the held inode to the canonical destination so the
outer failure path still invalidates it. A postcondition or context-exit
failure atomically renames the known directory to an
`.analysis-invalid-*` forensic name, so the canonical output can never be
consumed. Existing output is never overwritten. All final artifacts and their
directory are read-only.

Every published CSV table has an exact per-column Python type and admissible
range. Integer counters, folds, ranks, epochs, and seeds reject booleans and
floats; metric/timing columns require finite exact floats (or the narrowly
allowed `null` metrics); hashes and strings have their own exact contracts.
The analysis summary and manifest bind the sorted forensic-ledger SHA-256 plus
its resolved and invalid artifact counts.

The plan freezes:

- 100,000 deterministic bootstrap resamples;
- bootstrap seed `20260729`;
- 15 equal-width top-label ECE bins;
- TCFormer as the sole prespecified descriptive context comparator; and
- the equal-dataset primary aggregation.

## Metrics and aggregation

Every atomic prediction set receives accuracy, balanced accuracy,
chance-normalized balanced accuracy, macro F1, Cohen's kappa, one-vs-rest
macro AUROC when defined, negative log likelihood, multiclass Brier score, and
15-bin top-label ECE. Undefined values are written as JSON `null` and empty
CSV cells, never NaN.

The primary value is equal-dataset macro balanced accuracy:

1. concatenate disjoint folds within subject and seed;
2. average seeds within subject;
3. average subjects within dataset; and
4. give each of the five datasets equal weight.

The analyzer reports every model in frozen order, an outcome-ranked
development-only table, calibration and complexity summaries, per-dataset and
overall aggregates, and model-minus-TCFormer descriptive bootstrap intervals.
It produces no null-hypothesis tests, p-values, multiplicity decisions, or
publication-selection rule. Trial-micro results remain separately labelled
because they weight cohorts by trial count and repeat each trial across seeds.

## Published artifacts

The atomic analysis directory contains:

- `analysis.json` and human-readable `RESULTS.md`;
- `job_metrics.csv`, `subject_seed_metrics.csv`, `subject_metrics.csv`,
  `dataset_summary.csv`, and `overall_summary.csv`;
- `trial_micro_summary.csv`;
- `resource_timing_summary.csv`;
- `model_ranking.csv`, `calibration_summary.csv`, and
  `complexity_summary.csv`;
- `tcformer_dataset_context.csv`, `tcformer_overall_context.csv`, and
  `tcformer_seed_context.csv`;
- `input_checksum_ledger.csv`; and
- `manifest.json`, containing the plan/analysis identities and SHA-256 of
  every other artifact.

No analysis artifact contains per-trial labels, probabilities, or row
identities.

## Interpretation boundary

These five cohorts were already opened during development. The descriptive
leader, per-dataset table, and TCFormer intervals are useful for engineering
and paper reporting when labelled accurately, but architecture ranking on
these same cohorts creates selection bias. A sealed untouched cohort or a
prospectively collected dataset remains necessary for confirmation. Training
seeds are repeated algorithmic measurements, not independent human subjects,
and shared-workstation wall-clock measurements retain load-related variation.
