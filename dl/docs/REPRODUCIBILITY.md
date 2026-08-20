# Reproducibility guide

## Reproduction levels

This project distinguishes three different claims:

1. **Bundle verification** checks source and result bytes without opening EEG.
2. **Software verification** recreates isolated UV environments and runs
   tests/model-construction checks without training the full grid.
3. **Scientific reproduction** lawfully reacquires the exact datasets,
   rebuilds provenance-bound caches, executes 96,320 GPU jobs, audits the
   complete score-blind tree, and regenerates the aggregate analysis.

Passing a lower level does not prove a higher one. A different dataset mirror,
preprocessing choice, library lock, split, model roster, seed set, or GPU
environment is a new experiment rather than an exact reproduction.

## Sanitized evidence identity

The distributed common result is under `results/common_grid_v6/`.
The scientific score tables and prediction checksums are unchanged. Project
identifiers and the metadata objects that contain them were reissued, so the
identities below are the authoritative identities for this clean release.

| Evidence | Identity |
|---|---|
| Internal formal plan identity | `65b93b7e5d09cfc30fb6ce28156368e66aec1503f4efa37faa3189e25ebbc3b1` |
| Raw `plan.json` SHA-256 | `548c0bb707ac99debfafbc9082dd322aec7eb799e69f1ad69e25231eeb75873f` |
| Preflight `report.json` SHA-256 | `e9fe4b055ec4bd8171042492ba2832c22a083c1347ec244312ce331442df7a4f` |
| Final `final_audit.json` SHA-256 | `21908c268e15907806271245a1d14857b4e6704383b9bd6aadd5d253824af212` |
| Analysis `manifest.json` SHA-256 | `ed1070f69367e36109c473c193b4f898de40ad689a8d367ac3d9e7471a82a1cd` |
| Pre-result analysis-contract SHA-256 | `9d7eba89ca00621a8a3a681fc1ab408818e1df099ef1040685934eb1c7a82dfa` |
| Analysis input-ledger SHA-256 | `3f68621a316a6a90cf18ce073e05f70b453a13fa7ef4407f66a90f54562dcbbf` |
| Forensic-ledger SHA-256 | `962531ea433232d90d9869a2d78529f2e867cd7373b883dd78e8cf01ca31f626` |

The final audit has `exact_cartesian_complete=true`, 96,320 expected and
complete records, zero missing, extra, failed, live-claim, stale-claim, or
partial records, and a valid preflight. Six stale claim artifacts left by
interrupted workers were preserved in quarantine and resolved by the forensic
audit; they are not six failed scientific jobs.

The analysis directory is a sealed 17-file publication. `manifest.json`
declares SHA-256 values for the other 16 files. It includes pseudonymous
job-level, subject-seed-level, and subject-level derived metrics; it excludes
raw EEG and trial-level predictions. Those participant-derived tables require
governance review before public release. Generated convenience tables outside
that directory are views; the sealed analyzer output remains the authority.

## 1. Verify the distributed bundle

From the project root:

```bash
scripts/reproduce.sh verify
```

The verifier checks required paths, the sanitized namespace, the source
provenance inventory, banned generated state, the analysis manifest, the
reviewer catalog, and every entry in `results/SHA256SUMS`. It does not open EEG
or recompute metrics from raw atomic predictions because those prediction
packages are intentionally external to the clean release.

For an independent checksum pass on Linux:

```bash
cd results
sha256sum --check SHA256SUMS
```

## 2. Create isolated UV environments

Prerequisites:

- Linux x86-64;
- UV (the audited execution used UV 0.12.0);
- network access for the first lock-frozen synchronization, or a complete
  reviewed UV cache for offline synchronization;
- sufficient private disk space; and
- for formal execution, an NVIDIA driver compatible with the locked
  `torch==2.6.0+cu124` and `torchaudio==2.6.0+cu124` wheels.

The project pins CPython 3.12.13 in `.python-version`. The three environments
are intentionally separate:

```bash
scripts/reproduce.sh setup runtime
scripts/reproduce.sh setup test
scripts/reproduce.sh setup docs
```

They create `.venv`, `.venv-test`, and `.venv-docs` from the same frozen
`uv.lock`. No command uses `sudo`, system `pip`, or another user's environment.
To select a particular UV executable:

```bash
UV_BIN=/absolute/path/to/uv scripts/reproduce.sh setup runtime
```

After `setup docs`, validate the sealed report inputs and generate the three
deterministic PDFs. The default destination is `output/pdf/`; an alternate
destination must be explicit:

```bash
scripts/reproduce.sh reports
scripts/reproduce.sh reports --output-dir /absolute/report-directory
scripts/reproduce.sh reports --validate-only
```

`--validate-only` creates no PDFs. A successful build does not replace the
required page-by-page visual review before release.

After setup:

```bash
uv lock --check
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv pip check
UV_PROJECT_ENVIRONMENT="$PWD/.venv-test" uv pip check
```

Do not run `uv lock --upgrade`, `uv add`, an unfrozen sync, or an automatic
environment repair after a formal plan is created. A changed dependency set
requires a new project identity and run root.

## 3. Run the software tests

```bash
scripts/reproduce.sh test
```

The unit suite is not the 96,320-job benchmark. It exercises model geometry,
loaders, deterministic training, strict schemas, source identity, the pinned
TCFormer snapshot, GPU leases, crash/recovery behavior, exact auditing, and
analysis publication. CUDA-specific cases require the reviewed Linux GPU
environment. The frozen UV lock is Linux x86-64 only: macOS can run the
standard-library bundle verifiers, but it cannot use the documented setup/test
commands without creating a separately documented, nonformal environment.

## 4. Acquire data and build private caches

Read `docs/DATA_ACCESS.md` before downloading anything. The exact common grid
requires all five datasets, including authorized Local Exp4 access. Create
cache and raw-data roots outside this Git tree:

```bash
scripts/reproduce.sh prepare-data \
  /absolute/private/cache \
  /absolute/private/local-exp4
```

This requests only the opened development cohorts:

- BNCI2014-001 S1–S9;
- BNCI2014-004 S1–S9;
- Cho2017 S1–S52;
- PhysioNet MI S1–S54; and
- Local Exp4 participant codes 1, 3, 4, 5, 6, 7, 8, and 10.

It does not authorize access to the sealed cohorts. If the Local Exp4 path is
omitted, the helper skips that private dataset; the resulting cache cannot run
or be labelled as the exact five-dataset common grid.

Preserve acquisition URL/version, access date, original file hashes, exact
terms, preprocessing identity, ordered channels, cache hashes, and split
hashes. The formal planner validates those identities rather than accepting a
shape-compatible replacement.

## 5. Choose new external run paths

Use previously unused absolute directories on a filesystem with at least 50
GiB free. Do not run inside `results/common_grid_v6`, a data cache, the source
tree, another user's directory, or an earlier formal root.

```bash
export MI_CACHE_ROOT=/absolute/private/cache
export MI_RUN_ROOT=/absolute/new/common-grid-run
export MI_ANALYSIS_ROOT=/absolute/new/common-grid-analysis
```

The examples use task-specific variables intentionally; do not repurpose
`HOME` or another system variable.

## 6. Execute the mandatory CUDA preflight

Select one approved idle physical GPU:

```bash
scripts/reproduce.sh preflight "$MI_CACHE_ROOT" "$MI_RUN_ROOT" 0
```

Preflight creates or validates the immutable plan and performs exactly 172
score-blind CUDA checks: 43 configurations against four canonical
shape/class contracts. Each constructs the model, verifies finite logits and
nonzero finite gradients, takes one AdamW step, and verifies a finite
post-step forward pass. The attestation binds plan, source, environment,
cache identities, GPU UUID, and shared GPU-lease receipt; it contains no test
score.

Formal scripts set `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the compact TCFormer source root.

## 7. Run or resume the exact grid

On a shared four-GPU workstation, the audited policy used GPUs 0–2 and left
GPU 3 outside the launcher:

```bash
scripts/reproduce.sh run "$MI_CACHE_ROOT" "$MI_RUN_ROOT" 0,1,2
```

After an orderly pause, process interruption, or power loss, invoke the same
command:

```bash
scripts/reproduce.sh resume "$MI_CACHE_ROOT" "$MI_RUN_ROOT" 0,1,2
```

The runner validates completed records and recovers only schema-recognized,
identity-bound stale state. Do not edit claims, quarantine, records, plan,
preflight files, or permissions by hand. Do not launch more than three
project workers or include GPU 3 merely to shorten runtime on a shared host.

Read-only progress:

```bash
scripts/reproduce.sh status "$MI_RUN_ROOT"
```

The run is expensive and can take days depending on hardware, stalls, and
shared-workstation load. Runtime measurements are workload-dependent even
when predictions remain deterministic.

## 8. Require the exact final audit

```bash
scripts/reproduce.sh audit \
  "$MI_RUN_ROOT" \
  /absolute/new/common-grid-final-audit.json
```

Do not analyze unless the audit reports the exact Cartesian product complete,
the preflight valid, and no missing, extra, failed, unsafe, partial, or live
claim state. There is no available-case fallback.

## 9. Generate a new analysis publication

The destination must not exist:

```bash
scripts/reproduce.sh analyze \
  "$MI_RUN_ROOT" \
  "$MI_CACHE_ROOT" \
  "$MI_ANALYSIS_ROOT"
```

Analysis reacquires the exclusive publication fence, revalidates the complete
prediction tree, joins labels, computes participant-level metrics and 100,000
deterministic bootstrap resamples, and publishes a new sealed 17-file
directory. Compare its plan, analysis-contract, input-ledger, environment,
source, and manifest identities before comparing numbers with the distributed
analysis.

## Reproduction limitations

- Raw atomic prediction records are not distributed, so the sealed aggregate
  bundle can be checksum-verified but not independently rescored without
  rerunning or obtaining the authorized external result root.
- Local Exp4 is private and its release authorization is not documented in
  this project. A public-data-only rerun is scientifically useful but is not
  the same fixed five-dataset suite.
- Exact dataset-file licenses and hashes must follow the bytes actually
  acquired; MOABB metadata is not a substitute.
- A driver, GPU, library, compiler, or filesystem change can alter runtime
  identity and possibly numerical bytes. Report the new identity rather than
  relabelling it as the original run.
- The sealed confirmation cohorts remain unopened. Reproduction of
  development results does not authorize confirmation access.
