# Harmonized-v2 binary neural-procedure grid

The retained `eeg-mi-cache-v2` and `.eeg-mi-project-*` tokens are versioned
cache and coordination-schema identifiers. They are not active Python package
or distribution names.

`benchmark.procedure_grid` is the cache/protocol-bound outer adapter for the four
binary neural procedures that have an exact, outcome-free configuration file:

| Stable ID | Runtime key | Frozen config |
|---|---|---|
| `architecture.cameo` | `cameo` | `configs/local_procedures/cameo_v1.json` |
| `architecture.hemiparity` | `hemiparity` | `configs/local_procedures/hemiparity_v1.json` |
| `architecture.parity_fuse` | `parity_fuse` | `configs/local_procedures/parity_fuse_v1.json` |
| `architecture.orbit_v3` | `orbit_v3` | `configs/local_procedures/orbit_v3.json` |

This is a **separate neural-procedure track**. Its results must not be inserted
into the common raw-trial training-recipe table. ORBIT-v1/v2/v4/v5 are not
silently aliased to v3 and are not reconstructed from result artifacts. Their
read-only provenance boundary is documented in
[`ORBIT_CONFIG_PROVENANCE.md`](ORBIT_CONFIG_PROVENANCE.md).

## Exact formal cardinality

The opened development cohorts and harmonized-v2 splits are:

| Dataset | Subjects | Folds | Seeds | Records per procedure |
|---|---:|---:|---:|---:|
| local Exp4 | 8 | 1 fixed chronological | 5 | 40 |
| BNCI2014-004 | 9 | 1 official chronological | 5 | 45 |
| Cho2017 | 52 | 5 rotating acquisition blocks | 5 | 1,300 |
| PhysioNet MI development | 54 | 3 rotating imagery runs | 5 | 810 |
| **Per procedure** | | | | **2,195** |
| **Four procedures** | | | | **8,780** |

The exact seed order is `7, 17, 27, 37, 47`. BNCI2014-001 is intentionally
absent because these four procedures have binary heads.

`test_formal_grid_is_exact_8780_jobs_and_excludes_blocked_orbits` reconstructs
this Cartesian product without opening EEG and proves that every job ID is
unique.

## Paired input representation

The adapter owns one deterministic representation contract for every dataset:

1. Start from the unaltered broadband trial in `eeg-mi-cache-v2`.
2. Construct the sagittally reflected broadband view with the same odd/even
   10-20 channel permutation used by the procedure implementations.
3. Apply four fourth-order zero-phase Butterworth filters independently to
   each trial: 8-12, 11-15, 14-20, and 20-30 Hz.
4. Construct a fixed-shrinkage SPD covariance independently for each trial and
   band (`shrinkage=1e-3`, demeaning enabled).
5. Construct the reflected SPD view as \(P C P^\mathsf{T}\).
6. During phase A, fit each procedure's tangent reference/standardizer/anchor
   using training rows and their deterministic reflections only.
7. During phase B, discard that fitted preprocessing and refit a fresh tangent
   representation using train+validation rows and their reflections only.
8. Apply the phase-B tangent transform to held-out rows only after phase B is
   complete.

The plan pins the filter coefficients, transform semantics, source code, and
config hashes. Each score-blind record stores only array shapes/dtypes/hashes
for the original/reflected broadband, SPD, and tangent views. It stores no raw
trial, test label, or test score.

The transform supports both current harmonized lengths: 256 samples for local
Exp4 and 320 samples for the public cohorts. Unit tests prove deterministic
byte identity, an involutive reflection, and positive-definite covariance
outputs at both lengths.

## Selection, reset, refit, and one-shot prediction

Every atomic job has this boundary:

- **Phase A:** fit preprocessing and neural weights using the training
  partition; use validation only to choose the epoch count and, where the
  original method declares one, the CAMEO mixture/rho or ORBIT candidate.
- **Reset:** reconstruct the estimator with the same seed. The existing
  procedure implementation hashes its initial trainable state. Only explicitly
  declared data-dependent tangent initialization entries may differ; the
  remaining initialization hash must match exactly.
- **Phase B:** fit fresh preprocessing and weights using train+validation for
  exactly the selected epoch count and frozen selected route.
- **Prediction:** call `predict_proba` exactly once on the held-out rows. No
  mirror diagnostic, test metric, or second estimator call is made by the
  producer.

The adapter delegates the method-specific fitting operations to the already
tested `benchmark.local_outer_refit_benchmark` primitives. Therefore CAMEO,
HemiParity, PARITY-Fuse, and ORBIT-v3 retain their own exact configuration,
route selection, tangent fitting, and reset exclusions rather than being
forced into a generic trainer.

The prediction directory contains:

```text
record.json       # score-blind identity and source-only fitting provenance
predictions.npz   # int64 held-out row IDs + float64 class probabilities
completion.json   # hashes of the preceding two files
```

The NPZ container is itself part of the exact format. Its raw ordered
`ZipInfo.filename` sequence must be exactly `rows.npy, probabilities.npy`, and
NumPy must expose exactly `archive.files == ["rows", "probabilities"]` in that
order. Duplicate, extra, reordered, or path-bearing members are rejected even
if an attacker recomputes the outer completion checksum.

No producer record contains a held-out label, predicted class, accuracy,
balanced accuracy, AUC, or other test metric.

## Immutable plan and UV environment

Production commands require a UV virtual environment for which the canonical
`VIRTUAL_ENV` path equals the active `sys.prefix`, plus a working
`uv --version`. The formal plan, resume/load, worker, analysis, and publication
entry points all enforce that gate. They never call `pip`, `uv add`, or
`uv sync`.

The immutable plan pins:

- the analyzer decision code itself (`src/benchmark/procedure_grid_analysis.py`);
- both executed package initializers and the authoritative model registry;
- every runner/model/data/SPD/augmentation transitive source file;
- all four outcome-free configuration files by byte count and SHA-256;
- the complete UV distribution inventory and its digest;
- Python executable, prefix, platform, UV version, PyTorch/TorchAudio/CUDA runtime;
- NVIDIA UUID, PCI bus ID, model, and driver inventory;
- `pyproject.toml` and `uv.lock`;
- every cache identity and every train/validation/source/test row digest; and
- the blocked ORBIT provenance boundary.

The formal plan also has an exact recursive schema. Recomputing its checksum
cannot legitimize a changed roster, protocol, view definition, executor,
analysis setting, blocked-method boundary, or resource threshold. Synthetic
plans are explicitly marked `test_nonpublishable`; they can exercise the
mechanics but can never satisfy formal analysis.

Each formal cache identity is read once through one no-follow, nonblocking
descriptor; the same verified bytes are parsed and hashed. It binds the exact
dataset, preprocessing, coordinate, channel-order, array-shape, trial-count,
channel-count, position-array, and (for local Exp4) source-file manifest
schemas. Recursive outcome aliases such as `dataset.hidden_labels` fail even
if an attacker recomputes the outer plan digest. Config files use the same
single-snapshot parse-and-hash rule.

The release UV manifests are an exact contract, not a presence check. Every
direct requirement must use one `==` pin. Runtime must include NumPy, SciPy,
scikit-learn, MNE, pyRiemann, PyTorch, and the matching TorchAudio build; the
non-default `test` group must
pin pytest; and the non-default `docs` group must pin ReportLab, pdfplumber,
and pypdf. `[tool.uv] default-groups = []` is mandatory. `uv.lock` must contain
the same direct pins, both named groups, their exact metadata ledgers, and no
package outside the runtime+test+docs dependency graph. Formal execution then
requires the active environment to equal the marker-selected runtime
transitive closure exactly: a test/docs tool or any unrelated installed
distribution makes the release gate fail. Dependency installation remains a
separate, explicit release-environment step; the runner never repairs or syncs
the environment itself.

Create the final environment first. Do not change or sync that environment
after plan creation.

Recommended shared-workstation environment:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
```

Create or verify the plan:

```bash
uv run --frozen --no-sync --no-default-groups python -m benchmark.procedure_grid plan \
  --run-root /path/to/runs/binary_procedure_grid_v1 \
  --cache-root /path/to/data_cache \
  --cpu-threads 4
```

Plan assembly is deterministic and carries an unsealed timestamp sentinel.
The first publication replaces that sentinel once and recomputes the digest.
Both `plan.json` and `plan.sha256` are first fsynced as non-authoritative files
in the sibling forensic tree, then installed with atomic no-replace renames.
Thus a power cut cannot expose a torn authoritative plan leaf. Reassembly
compares the timestamp-independent semantic plan and preserves the original
sealed timestamp. If power is lost after `plan.json` is installed but before
`plan.sha256`, rerunning validates the orphaned canonical plan and atomically
installs only its missing receipt. No other final-path state is repaired.

## Cooperative workers and resumption

Formal workers use the project-wide registry
`.eeg-mi-project-gpu-leases`, shared by the procedure, common-recipe, and
GeoAdapt tracks. Its persistent, tokenized `.registry.fence` protects exact
one-file-per-physical-UUID receipts and enforces a global maximum of three
workers. Worker index 3 and GPU 3 are never selected by default. Pin exactly
one physical GPU UUID per process. `--gpu`, `CUDA_VISIBLE_DEVICES`, the GPU
guard, PyTorch-visible device 0, record metadata, and both resource receipts
must all resolve to that same UUID:

```bash
CUDA_VISIBLE_DEVICES=GPU-... uv run --frozen --no-sync --no-default-groups \
  python -m benchmark.procedure_grid worker \
  --run-root /path/to/runs/binary_procedure_grid_v1 \
  --cache-root /path/to/data_cache \
  --gpu GPU-... --device cuda:0 --worker-index 0 \
  --cpu-threads 4 --recover-stale
```

Run worker indices 0, 1, and 2 on three different idle UUIDs. Logs belong
outside the immutable run root.

The registry’s final fence and lease files are never written in place.
Complete receipts are fsynced in the sibling
`.eeg-mi-project-gpu-lease-forensics` tree and atomically renamed into the
registry without replacement. Release and verified-dead local recovery hold
the exact lease descriptor, rename that inode to an external released/stale
archive, and recheck its nonce, owner, content, and inode at the destination.
A foreign valid owner is treated as live indefinitely; malformed or unknown
active state fails closed and is never stolen.

Every acquired handle also pins the registry-fence inode. Claim publication
and final result-directory publication each run inside a held registry guard,
not between two instantaneous assertions. The guard serializes threads in the
same process and processes through the registry fence, validates the exact
fence and active-lease inode/content/owner before yielding a canonical
receipt, and repeats that validation before unlocking. The claim embeds that
receipt, and commit must obtain the identical receipt from its held guard.
Replacement, disappearance, or release during either guarded publication
fails closed; a post-publication guard failure quarantines the result.

Before work begins, a worker revalidates the exact source, UV environment,
cache, split, config, CPU-thread, device, GPU-UUID, and 50 GiB threshold
identities. Before each new claim it
checks:

- the assigned GPU has no foreign compute process, at most 10% utilization,
  and at most 1,024 MiB unexplained memory;
- the run filesystem remains above the 50 GiB low-water mark.

The disk threshold, physical GPU binding, and cooperative `nvidia-smi`
foreign-process guard are probed again after training and immediately before
commit. The GPU probe strictly parses the selected GPU identity/counters and
every row in the global compute-process table; a malformed row for any GPU,
non-finite counter, duplicate process identity, foreign process on the
selected UUID, or unexplained memory fails closed. Independently of that
cooperative receipt, formal code measures free bytes through a descriptor
bound to the target filesystem immediately before each authoritative write
and again immediately before each claim, result, or analysis rename. The hard
floor is 50 GiB and cannot be lowered by a caller.

Claim files use host/PID/boot/process-start identity. Recovery is conservative:

- a live claim is never stolen;
- an owner must have an exact schema, positive PID, nonempty host, valid
  non-null boot UUID, and positive process-start token; malformed owners are
  never considered live, and malformed claim files fail closed rather than
  being auto-recovered;
- a claim from another host remains live indefinitely unless a future
  distributed lease protocol can prove expiry; wall-clock age alone is never
  evidence of death;
- every claim holds a shared tokenized publication-fence descriptor for its
  whole lifetime;
- commit holds and rereads the same claim descriptor, verifies nonce, owner,
  plan, job, resource receipt, device, and inode before the final rename, then
  rechecks the same path/inode/content and publication fence after the rename;
- each record embeds an exact claim receipt and the completion binds its hash;
- the complete score-blind directory is first fsynced, moved to a
  non-authoritative stage adjacent to its final path, fsynced, changed to mode
  `0555`, and fsynced again before a same-parent atomic no-replace final
  rename;
- completion validation holds the final directory descriptor throughout its
  snapshot, requires that pathname to retain the same inode and exact
  nonwritable `0555` mode, and accepts only the three exact immutable leaves;
- if that post-rename recheck fails, the destination is moved out of the run
  tree, resealed in quarantine, and cannot be counted as complete;
- a stale claim and interrupted build or adjacent publication stage are moved
  to the sibling forensic tree, never deleted or retained as publishable run
  leaves;
- if power fails after the complete record directory rename but before claim
  release, a resumed worker validates the immutable completion first and then
  quarantines the stale post-rename claim; and
- a completed record is never trained again.

Failure receipts, released claims, staging files, and quarantine evidence live
under the sibling `.<run-name>.procedure-forensics` tree, not under the
publishable run. The run root permits only the exact plan/fence leaves plus
`records`, `claims`, and `partials`. Live/stale claims, residual partials,
missing/corrupt records, extra record directories, or unknown root entries
invalidate it. Every run/cache/source/output ancestor and leaf is inspected
with no-follow semantics; regular inputs are opened with `O_NOFOLLOW` and
`O_NONBLOCK`, so FIFOs fail immediately. Symlinks, special nodes, unexpected
names, writable immutable leaves, and hard-linked files fail the audit.

## Quiescent audit and separate analysis

Audit without opening labels for scoring:

```bash
uv run --frozen --no-sync --no-default-groups python -m benchmark.procedure_grid audit \
  --run-root /path/to/runs/binary_procedure_grid_v1
```

For a formal plan, exit status is zero only for all 8,780 checksum-valid
records with no claim or partial path and no unexpected, aliased, hard-linked,
or special record/root entry. A complete synthetic audit remains explicitly
nonpublishable.

Only after that audit succeeds may the separate analyzer join labels:

```bash
uv run --frozen --no-sync --no-default-groups python -m benchmark.procedure_grid_analysis \
  --run-root /path/to/runs/binary_procedure_grid_v1 \
  --cache-root /path/to/data_cache \
  --output-root /path/to/runs/binary_procedure_grid_v1_analysis
```

The output must be outside both the immutable run and cache trees. ECE is
frozen at exactly 15 bins. The analyzer:

1. proves exact completion and quiescence before its first label load;
2. rebuilds every cache and split identity;
3. joins labels to rows only in memory;
4. concatenates disjoint folds within subject/seed;
5. averages five seeds within subject;
6. equally weights subjects within dataset;
7. equally weights the four dataset summaries;
8. validates exact summary/table schemas, identities, aggregation formulas,
   descriptive ranks, and formal row cardinalities;
9. acquires an exclusive publication fence respected by every worker claim
   and commit, pinning and rechecking the fence inode and token;
10. freshly recomputes the complete semantic analysis from immutable records
    and bound labels while holding that fence, refusing any caller-supplied
    scalar or claim mutation;
11. validates output containment before creating any parent, rejects every
    symlinked ancestor, and publishes read-only scalar CSV/JSON files with an
    atomic no-replace rename;
12. rebuilds the full plan/cache/job checksum ledger before and after writing
    the temporary publication while the fence is still held; each validated
    record file is hashed and decoded from the same descriptor snapshot, and
    each output manifest hash is computed from the exact byte string passed to
    its exclusive writer rather than by reopening the path; and
13. moves the just-published destination to a nonpublishable quarantine name
    if the exclusive fence inode/token is replaced before unlock.

The reported rank is descriptive within this four-procedure track. There is no
prespecified superiority test in this runner revision and no confirmation,
common-recipe, global, or SOTA claim.

## Focused verification

The local test suite is entirely injected/synthetic and performs no EEG
scoring:

```bash
uv run --frozen --no-sync --no-default-groups --group test python -m pytest -q \
  tests/core/test_project_gpu_leases.py \
  tests/core/test_procedure_grid.py \
  tests/core/test_procedure_grid_analysis.py
```

It covers formal cardinality, outcome-free config isolation, both input
lengths, SPD/reflection invariants, canonical plan repair, score-blind atomic
publication, stale claim/partial recovery on both sides of the final rename,
source-only reset/refit ordering, one-shot held-out inference, injected worker
resumption, audit-before-label access, scalar-only aggregation, checksum-ledger
revalidation, exact config/candidate/reset schemas, resealed outcome aliases,
claim-token/inode loss, foreign-claim age, symlink/FIFO/hardlink containment,
publication-fence replacement before and after final renames, claim loss after
record publication, semantic recomputation, containment-before-mkdir,
destination no-replace races, cache/run output containment, torn
registry/lease bootstrap, global physical-GPU cap and duplicate exclusion,
lease/fence reassertion, receipt binding, assertion-time path replacement,
lease-release inode replacement, held-guard publication serialization,
descriptor-bound result-directory replacement, writable-directory rejection,
sealed pre-rename record publication, duplicate/extra/reordered/path-bearing
NPZ member rejection, strict global `nvidia-smi` row parsing, hard 50 GiB
prewrite/prerename checks, fixed ECE, and exact runtime/test/docs UV
dependency-closure fixtures.
