# Author-recipe neural reference grid

`benchmark.author_recipe_grid` is the executable adapter for the two
author-recipe adapted registry procedures:

- `reference.tcformer`
- `reference.fbcnet`

“Adapted” is deliberate: the procedures preserve released optimization logic
where applicable, but integrate it with this study's locked caches, splits,
scalers, and cross-dataset protocol. They are not exact author-reported
reproductions.

This is a separate author-recipe track. Its results must never be substituted
for, appended to, or ranked as common-recipe model rows. The different
optimization schedules answer a different scientific question.

## Exact formal grid

The immutable plan contains the five opened development cohorts and no sealed
subjects:

| Dataset | Subjects | Folds | Seeds | Jobs per reference |
|---|---:|---:|---:|---:|
| `local_exp4` | 8 | 1 | 5 | 40 |
| `bnci2014_001` | 9 | 1 | 5 | 45 |
| `bnci2014_004` | 9 | 1 | 5 | 45 |
| `cho2017` | 52 | 5 | 5 | 1,300 |
| `physionet_mi` | 54 | 3 | 5 | 810 |
| **Total** |  |  |  | **2,240** |

The two references therefore create exactly 4,480 atomic records.

Every record is one dataset/reference/subject/fold/seed fit. The plan freezes
the cache-array hash, exact train/validation/source/test row-vector hashes,
full UV package inventory, `pyproject.toml`, `uv.lock`, in-repository execution
closure, recipe configuration, analysis source, and official-source
provenance. Formal planning requires:

- an active UV virtual environment;
- the sole audited executor
  `benchmark.author_recipe_grid:execute_author_recipe_job` (there is no formal
  CLI or worker-callable override);
- exact direct and locked pins for `torch`, `braindecode`, `einops`, `mne`,
  `numpy`, `scikit-learn`, and `scipy`;
- the project-local `.venv`, with `uv lock --check`,
  `uv sync --check --frozen --no-default-groups` (including rejection of extra
  packages), and `uv pip check` all succeeding;
- `braindecode==1.6.1`;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`; and
- `EEG_MI_TCFORMER_ROOT` pointing to the reviewed compact TCFormer runtime
  snapshot from commit
  `74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`.

The snapshot contains the upstream MIT `LICENSE`, the four runtime-loaded
Python files, and an exact canonical `SOURCE.json`. The adapter independently
pins every file's byte size and SHA-256; modifying both code and manifest does
not bypass the upstream pin. The TCFormer plan records this complete identity.
The FBCNet record identifies the original source commit
`de1bbdd8a54cb1e466830e3d47070e0e56761a37` and the pinned Braindecode adapter;
the original FBCNet checkout is not executed.

The registry contract is source-hashed and validated at plan creation and
resume. Only the stable procedure IDs `reference.tcformer` and
`reference.fbcnet` are accepted. Bare `tcformer` and `fbcnet` are common-grid
architecture names and cannot be used as author-recipe job identities.

## Recipe semantics

### TCFormer

`fit_tcformer_reference` installs the seed before construction and uses the
released Adam parameters, warmup/cosine schedule, and one-for-one
eight-segment reconstruction augmentation.

TCFormer has no validation-selected epoch in this adapter. The duration is
fixed before outcomes are seen:

- 1,000 epochs for BNCI2014-001;
- 500 epochs for BNCI2014-004; and
- a declared 1,000-epoch IV-2a-derived study adaptation for local Exp4,
  Cho2017, and PhysioNet.

Because there is no outcome selection to repeat, its final prescribed fit uses
train+validation rows, with a scaler fitted on those same source rows. This is
not described as a released cross-dataset recipe for the three adapted
cohorts.

### FBCNet

Raw trials are scaled from training rows and passed through the original
nine-band, one-pass causal Chebyshev-II filter bank. Stage 1 trains on the
training partition and uses validation inaccuracy for the released relative
no-decrease patience rule. The fitter restores the best model and Adam state.
Stage 2 optimizes train+validation and stops when the original validation
subset reaches the released loss threshold or the cap.

Every FBCNet record persists the numeric stage-1 transition threshold, its
origin (`last_stage1_epoch_before_best_restore`), and the exact origin epoch,
in addition to the restored model/optimizer hashes.

The same training-row scaler is used for test inference because stage 2
continues the selected fitted model rather than resetting and fitting a new
preprocessor. This is explicitly recorded, not silently treated as a
common-recipe reset/refit.

Neither reference fitter has a test-array argument.

## Score-blind atomic output

After source fitting, the executor calls `predict_probabilities` once on the
test tensor. The record contains:

- job and plan identities;
- source-only fitting, recipe, scaler, state-hash, timing, and runtime
  provenance;
- the test-row-vector hash; and
- a compressed archive containing only `rows` and `probabilities`.

It does not compute a test metric, persist a test outcome, or persist a
predicted class. Recursive exact-schema and forbidden-key validation is
applied before publication and again on every resume/audit. Outcome aliases
such as `outcomes`, `truth`, `ground_truth`, `y_true`, and `y_test` are
explicitly forbidden.

Each job is assembled under `partials/`, fsynced, and checksummed. The final
directory move uses the platform's native directory-FD no-replace primitive:
`renameat2(RENAME_NOREPLACE)` on Linux or
`renameatx_np(RENAME_EXCL)` on macOS. There is no check-then-overwrite
fallback. The exact moved directory FD is immediately sealed and fsynced as
`0555`; a canonical completion is never accepted unless the directory is
`0555` and every child is unique and read-only. Any failure after the move
quarantines only that exact published inode. A raced replacement remains
untouched.

Completion validation opens the sealed directory and all three children at
the same time. It retains every FD until all bytes are read and the full
directory and sorted name-to-child fingerprints have been rebound. This
rejects leaf A→B and A→B→A substitution while allowing unrelated sibling
results to publish concurrently. `predictions.npz` is first inspected as a
raw ZIP: its exact members must be `rows.npy` then `probabilities.npy`, with
no duplicate names, before NumPy decoding is allowed.

Plan, record, claim, partial, and publication paths are checked without
following symlinks; special files, hard-linked leaves, unknown nodes, and
inode swaps fail closed. Cache NPZ leaves are decoded only from a contained
`O_NOFOLLOW`, regular, single-link byte snapshot. Exclusive claim files
prevent duplicate workers. Immediately before and immediately after a record
move, the runner rebinds the live plan, source closure, UV environment,
recipe configuration, cache and split identity, requested GPU, exact claim,
and held shared publication-gate lease. A move-then-raise outcome is
recognized only from the destination's exact staged inode and is rolled back.

Dead claims, interrupted partials, release tombstones, corrupt completions,
and publication fences are moved to `quarantine/` only with the exact inode
captured by their validation. Recovery and release never delete or move a
replacement. Read-only directories are made owner-writable only through a
held exact FD for their quarantine move, then resealed `0555`. Historical
failure receipts remain available under `failures/`.

## Shared-workstation execution

Create and verify the clean environment without modifying system Python:

```bash
cd /home/hanafy/neuralnetwork
uv sync --frozen --no-default-groups
uv run --frozen --no-default-groups --no-sync \
  python -c "import sys; print(sys.prefix)"
```

Do not use `sudo`, system `pip`, or a global Conda environment.

Set the deterministic and official-source environment:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export EEG_MI_TCFORMER_ROOT=/home/hanafy/neuralnetwork/third_party/TCFormer
```

Initialize or resume with at most three GPUs by default, leaving one GPU
available to other workstation users:

```bash
uv run --frozen --no-default-groups --no-sync \
  python -m benchmark.author_recipe_grid run \
  --run-root /home/hanafy/scratchpad/author_recipe_grid_v1 \
  --cache-root /home/hanafy/scratchpad/eeg_novel_20260729/data_cache \
  --gpus 0,1,2 \
  --space-check-path /home/hanafy/scratchpad \
  --min-free-gib 50 \
  --cpu-threads 4
```

Workers use `sys.executable`, so the UV environment is preserved. Before each
new claim, the runner verifies its physical GPU is cooperatively idle and that
both output and optional scratch filesystems remain above the low-water mark.
It never uses DDP. GPU identities may be supplied as indices or UUIDs; worker
subprocesses bind the resolved UUID with `CUDA_VISIBLE_DEVICES`.

Status and exact audit:

```bash
uv run python -m benchmark.author_recipe_grid status \
  --run-root /home/hanafy/scratchpad/author_recipe_grid_v1

uv run python -m benchmark.author_recipe_grid audit \
  --run-root /home/hanafy/scratchpad/author_recipe_grid_v1
```

After a power cut, rerun the original `run` command. Plan, source, cache,
split, UV environment, and official checkout drift all fail closed. Valid
completed records are reused. Stale recovery is enabled by default.

`--allow-busy-gpu` is an explicit unsafe override and should not be used on the
shared workstation. The filesystem floor is stricter: this publication runner
has no low-disk bypass, and a threshold below 50 GiB is always rejected.

## Label-joining analysis

Only after the exact 4,480-job audit reports a quiescent, checksum-valid
Cartesian grid may the separate analyzer load cache outcomes:

```bash
uv run python -m benchmark.author_recipe_analysis \
  --run-root /home/hanafy/scratchpad/author_recipe_grid_v1 \
  --cache-root /home/hanafy/scratchpad/eeg_novel_20260729/data_cache \
  --output-dir /home/hanafy/neuralnetwork/results/author_recipe_analysis_v1
```

The output directory must be outside both the run root and cache root and must
either be absent or be an exact previously published output. An existing
byte-identical, exact-schema output is validated and returned idempotently;
any other existing output fails closed. Analysis staging uses the same native
directory-FD no-replace move and exact-inode rollback. The moved output is
sealed `0555`, every child is unique/read-only, and the analyzer retains the
published directory plus every child FD through its final input, fence, lock,
and byte validation. A post-move failure therefore cannot leave a writable or
partially validated canonical output.

The output lock contains a canonical owner/host/PID/boot/start/nonce payload.
Its liveness decision and canonical validation come from one inode-bound byte
snapshot. A dead crash-left lock and stage are quarantined only by their
captured identity, while a live owner or a raced replacement is never
displaced.

Publication also holds an exclusive fence inside the run root from the first
final audit through runtime/source/cache/split/official-checkout
revalidation, full table recomputation, staging, fsync, and atomic rename.
Job claims hold the matching shared gate for their whole lifetime, so workers
cannot acquire or publish work during that window. The supplied result must
be byte-identical to the fresh recomputation, and the exact input window is
closed both immediately before and after rename. Each analysis ledger row is
hashed from the same coherent completion snapshot used for scoring, rather
than reopening artifact paths. Crash-left run-root fences carry the same
owner/boot/start identity and are safely quarantined only after their owner is
proven stale.

Aggregation is fixed as:

1. concatenate disjoint folds within subject and seed;
2. compute subject-seed metrics;
3. mean seeds within subject;
4. mean subjects within dataset; and
5. give each of the five datasets equal weight.

The analyzer publishes scalar job, subject-seed, subject, dataset, overall,
and timing tables. ECE is frozen at 15 equal-width bins; the CLI rejects any
other value. The immutable analysis contract also freezes metric definitions,
table schemas/order, aggregation strings, timing summaries, and undefined
metric handling. It never publishes labels, row vectors, probability
matrices, or outcome-selected rankings. The interpretation remains
exploratory evidence on already opened development cohorts, not confirmation
or a global state-of-the-art claim.
