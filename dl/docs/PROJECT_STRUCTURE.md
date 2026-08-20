# Project structure and namespace boundary

## Purpose

This document separates the live code namespace from the sanitized metadata
reissue of the completed common-grid v6 experiment. The distinction matters:
the reissue preserves scientific tables and numeric result values, but its
namespace-bearing plan, source-map, and checksum identities are new. Recorded
predecessor digests provide lineage; they are not current execution identity.

The supported command surface is `scripts/reproduce.sh`. Direct module
commands are implementation interfaces unless a procedure document explicitly
prescribes one for a separate track.

## Top-level map

```text
dl/
├── README.md                         project overview
├── pyproject.toml                    active dependency contract
├── uv.lock                           active Linux x86-64 lock
├── src/
│   └── benchmark/                    single live Python namespace
│       ├── *.py                      data, models, training, protocols, analysis
│       ├── research/                 exploratory model implementations
│       ├── shared/                   dependency-light numerical primitives
│       └── verification/             independent artifact verification
├── tests/
│   ├── core/                         benchmark/protocol tests
│   ├── research/                     exploratory-model tests
│   └── shared/                       numerical primitive tests
├── historical/                        read-only sanitized source reissues
│   ├── common_grid_v6/source/eeg_mi_v6/       v6 replay closure
│   └── native_fbms_full_grid_v1/source/eeg_mi_native_v1/
│                                              native-transfer v1 closure
├── configs/                           reviewed procedure configurations
├── scripts/                           supported command surface and tools
├── reviewer/                          deterministic score-cell replay catalog
├── results/                           distributed, checksummed evidence
├── output/pdf/                        publication reports
├── docs/                              methods and audit documentation
├── third_party/TCFormer/              pinned external runtime and license
└── SOURCE_PROVENANCE.json              release source/repair ledger
```

The project uses a conventional `src/` application layout and sets
`tool.uv.package = false`. The supported wrapper fixes `PYTHONPATH` to the
project's own `src/` directory before launching Python, so an ambient package
with the generic name `benchmark` cannot shadow this source tree. Commands run
from the project root in project-private UV environments. Both the distribution
name and Python namespace are `benchmark`.

Historical reissue packages have distinct names, are outside the ordinary
import path, and are not exposed through compatibility aliases.

## Live package roles

### `benchmark`: formal benchmark implementation

The benchmark layer owns:

- dataset, cache, and split contracts;
- the 43-configuration common-grid roster and external-model adapters;
- deterministic training and model construction;
- plan construction, GPU coordination, execution, audit, and analysis;
- bounded reviewer replay; and
- protocol-separated auxiliary research tracks.

`reviewer_replay.py` is a post-publication verification extension. It projects
a selected job, participant, dataset cell, or model row from the sealed v6
authority. It does not modify the 96,320-job result or independently rerank
models.

### `benchmark.research`: exploratory implementations

The research layer contains GeoAdapt, parity, CAMEO, ORBIT, HemiQ, local-data
procedures, and classical controls. These implementations may be called by a
benchmark-layer procedure, but they do not own formal data splits, result
aggregation, or publication claims.

### `benchmark.shared`: reusable numerical primitives

The shared layer contains SPD operations, augmentation, CSP initialization,
and metric helpers. It is intentionally dependency-light and does not import
either of the higher layers.

The enforced dependency direction is:

```text
scripts/reproduce.sh
        │
        ▼
    benchmark ─────► research
        │               │
        └──────► shared ◄┘
```

There is no reverse `research -> benchmark` import and no upward import from
`shared`. Tests mirror these boundaries under the top-level `tests/` tree.

## Sanitized common-grid v6 source reissue

The distributed v6 metadata reissue retains its historical logical executor
and `eeg_mi/...` source-map keys. Those strings identify the isolated replay
closure; they are not active import paths. The matching historical
`pyproject.toml`, `uv.lock`, and source hashes are validated under the archive.

The scientific analysis CSVs remain byte-identical to their completed-run
counterparts, and scientific JSON numeric leaves are unchanged. Namespace,
schema, source-map, plan, and checksum strings are reissued. The matching
sanitized source files and manifests are preserved under:

```text
historical/common_grid_v6/source/
├── eeg_mi_v6/     isolated sanitized package used only by the v6 adapter
├── pyproject.toml
└── uv.lock
```

The supported reviewer replay loads `eeg_mi_v6` under a private internal
package name without adding its parent to `sys.path`. The replay records the
resulting source/runtime assessment. Users should not add the historical
directory to `PYTHONPATH` or treat it as the live package.

The separate `historical/native_fbms_full_grid_v1/source/` snapshot contains
the sanitized 11-file `eeg_mi_native_v1/` closure bound by the reissued
native-FBMS checkpoint/source manifest. It is used only to validate those
completed artifacts. Predecessor source hashes are lineage records and are
not attributed to code executed from the live benchmark layer or the
sanitized archive.

The cleanup is a documented post-publication identity transformation.
Consequently:

- the reissued plan and source closure have a new identity;
- the workflow must not be described as byte-identical execution of the
  predecessor plan;
- a replay on changed manifests or hardware can be a successful
  fixed-protocol replication without matching exact execution identity; and
- predecessor hashes authenticate lineage only, while the current provenance
  and result ledgers authenticate the reissued files.

## Evidence, generated reports, and vendored code

- `results/common_grid_v6/` is immutable distributed evidence: plan,
  preflight, exact audit, sealed analysis, and checksummed convenience views.
- `reviewer/reviewer_score_cells.csv` is a generated, provenance-bound index
  from each accuracy/balanced-accuracy table cell to a bounded replay command.
  It remains outside the immutable `results/` bundle.
- `output/pdf/` contains derived reports. Numerical authority remains the
  validated source tables and manifests.
- `third_party/TCFormer/` is external code at a pinned upstream commit with
  its license and source manifest; it remains separate from in-house code.
- Private EEG, transformed caches, prediction archives, checkpoints, and new
  run trees belong outside the project directory.
- Reviewer roots also stay outside the project directory. Their scalar
  metrics, confusion matrices, pseudonymous job identities, timings, and
  runtime provenance are derived research records and are not automatically
  cleared for public release.

## Supported commands

Start at the project root and use the wrapper:

```bash
scripts/reproduce.sh help
scripts/reproduce.sh setup runtime
scripts/reproduce.sh setup test
scripts/reproduce.sh setup docs
scripts/reproduce.sh identity
scripts/reproduce.sh verify
scripts/reproduce.sh test
scripts/reproduce.sh reports
```

For bounded post-publication verification:

```bash
scripts/reproduce.sh reviewer list
scripts/reproduce.sh reviewer estimate --scope model --model MODEL_ID
scripts/reproduce.sh reviewer run \
  --scope dataset --model MODEL_ID --dataset DATASET_ID \
  --cache-root /absolute/private/cache \
  --run-root /absolute/reviewer-run \
  --gpu 0
scripts/reproduce.sh reviewer status --run-root /absolute/reviewer-run
scripts/reproduce.sh reviewer compare \
  --run-root /absolute/reviewer-run \
  --cache-root /absolute/private/cache
```

The exact `job`, `subject`, `dataset`, and `model` scopes are documented in
[the reviewer replay guide](REVIEWER_REPLAY.md). Reviewer replay is the
supported v6 score-replay path; it is not the formal score-blind producer and
cannot establish a 43-model rank from one model.

The wrapper also exposes `prepare-data`, `preflight`, `run` / `resume`,
`status`, `audit`, and `analyze` for compatible live-namespace run roots.
Because those operations use `benchmark`, a newly planned
complete run is a new
fixed-recipe evidence run, not a byte-identical continuation of the
predecessor v6 plan.

The file now located at `src/benchmark/requirements-cu128.txt` is retained only for
older research provenance. Do not install it: the active environment authority
is the root `pyproject.toml` plus `uv.lock`, used through the wrapper.

## Frozen randomness contract

Common-grid v6 fixes the training seeds:

```text
7, 17, 27, 37, 47
```

Its 96,320 jobs are 43 configurations × 448 participant/fold units × all
five seeds. The analysis concatenates folds within participant and seed, then
averages seeds within participant before participant and equal-dataset
aggregation. Replacing the schedule with one seed produces a different
experiment. Statistical analysis separately fixes bootstrap seed `20260729`
and 100,000 resamples.

Common-grid v6 did not record `PYTHONHASHSEED`; it must not be retroactively
claimed as part of that execution contract. It may be introduced only in a new
protocol version with a new identity.

## Reproduction terminology

Use the following terms precisely:

1. **Sealed-result verification** checks the distributed plans, manifests,
   source/provenance ledgers, audits, and aggregate tables without retraining.
2. **Fixed-protocol replication** keeps the model, recipe, data/cache, split,
   seed, and selection identities while explicitly recording source or runtime
   drift.
3. **Exact execution identity** additionally requires the original source and
   environment byte identities plus the reviewed hardware/software identity.
4. **Byte-identical artifact reproduction** is stronger again and excludes
   naturally changing operational fields such as timestamps, process IDs, and
   elapsed time unless those fields are normalized by the artifact builder.

The post-publication namespace migration means the active source tree alone
cannot claim the published v6 exact execution identity. A matching score on
new hardware or manifests is useful fixed-protocol evidence, not a retroactive
rewrite of the original run.

## Source-identity consequence

This cleanup deliberately removed the former cross-package cycle and created
one coherent namespace. It therefore creates a new live source identity. New
formal runs require a new plan, source ledger, and run root; the historical v6
snapshot and distributed result bundle remain separate evidence and must not
be relabelled as products of the reorganized live sources.
