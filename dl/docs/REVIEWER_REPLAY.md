# Reviewer-scale model replay

## Purpose and evidence boundary

The 96,320-job common grid is the evidence-generating experiment. A peer
reviewer does not need to repeat all 43 configurations to inspect one result.
The `reviewer` command runs an immutable projection of the reissued plan on one
CUDA GPU, ranging from one atomic job to one complete 2,240-job model row.

Reviewer replay is post-publication verification, not a replacement for the
formal score-blind run. Each replay job trains with the frozen recipe, forms
its predictions, joins the selected cache labels in memory, and retains only
scalar accuracy, balanced accuracy, a confusion matrix, checksums, and
provenance. Trial row identities and probability arrays are discarded rather
than published in the replay tree. Before discarding them, the runner hashes
the exact in-memory prediction serialization and records whether its digest
matches the corresponding published input-ledger entry. A digest difference is
recorded rather than aborting training, so a cross-hardware score comparison
remains possible. After the last selected job, `run` invokes comparison
automatically. The separate `compare` command repeats that validation for an
existing complete run root without retraining: it validates the selected
receipt set, reconstructs the applicable aggregates, and compares them with
the reissued v6 result tables.

The replay does not change the reissued plan, the 96,320-job audit, any published
score, or any rank.

## Active and reissued namespaces

The supported wrapper enters the live `benchmark` layer. The
sanitized v6 plan retains historical logical `eeg_mi/...` source keys and its
historical executor identity. Reviewer replay authenticates
those reissued identities and loads the matching sanitized snapshot from
`historical/common_grid_v6/source/eeg_mi_v6/` through an isolated adapter.
The snapshot package is not an active import alias.

This post-publication layer is a sanitized metadata reissue, not a byte-exact
copy of the predecessor source identity. Scientific tables and numeric result
values are preserved; namespace-bearing plan, source-map, schema, and checksum
identities are reissued. A replay can pass fixed-protocol and score comparisons
while correctly reporting that predecessor digests are lineage only.

## Requirements

- Linux x86-64 and the project-private UV runtime environment;
- one compatible NVIDIA CUDA GPU;
- the unmodified reissued source and vendored TCFormer snapshot;
- a private cache whose selected dataset identities match the frozen plan; and
- lawful access to every selected dataset.

Create the runtime environment before replaying training:

```bash
scripts/reproduce.sh setup runtime
```

Run roots and caches must remain outside the source tree and must be distinct
from one another. `scripts/reproduce.sh prepare-data` prepares the complete
opened public cohorts rather than only a later reviewer selection. Follow
[the data-access requirements](DATA_ACCESS.md); shape-compatible or
independently preprocessed files do not satisfy the frozen cache identity.

## Exact scopes and selectors

All scopes require one stable common-grid model ID. Inspect the accepted IDs:

```bash
scripts/reproduce.sh reviewer list
```

The selector contract is exact. A missing selector or a selector that belongs
to a narrower scope is rejected.

| Scope | Required selectors | Forbidden additional selectors | Jobs | What it can compare |
|---|---|---|---:|---|
| `job` | `--model`, `--dataset`, `--subject`, `--fold`, `--seed` | none | 1 | One sealed job-metric row |
| `subject` | `--model`, `--dataset`, `--subject` | `--fold`, `--seed` | 5, 15, or 25 | One subject aggregate |
| `dataset` | `--model`, `--dataset` | `--subject`, `--fold`, `--seed` | 40–1,300 | One published model-by-dataset cell |
| `model` | `--model` | `--dataset`, `--subject`, `--fold`, `--seed` | 2,240 | All five dataset cells and the equal-dataset model row |

The fixed training seeds are `7`, `17`, `27`, `37`, and `47`. The `subject`,
`dataset`, and `model` scopes always run all five seeds and every fold defined
by the selected dataset; they deliberately provide no fold or seed override.
Only the diagnostic `job` scope selects one already-planned fold and seed. A
one-job or one-subject replay does not confirm a dataset or overall aggregate.

The exact cardinalities are:

| Dataset ID | Subjects | Folds per subject | Jobs per subject | Jobs per model-dataset cell |
|---|---:|---:|---:|---:|
| `local_exp4` | 8 | 1 | 5 | 40 |
| `bnci2014_001` | 9 | 1 | 5 | 45 |
| `bnci2014_004` | 9 | 1 | 5 | 45 |
| `cho2017` | 52 | 5 | 25 | 1,300 |
| `physionet_mi` | 54 | 3 | 15 | 810 |
| **Complete `model` scope** |  |  |  | **2,240** |

## CSV catalog for every table score cell

[`reviewer/reviewer_score_cells.csv`](../reviewer/reviewer_score_cells.csv)
contains one row for each of the 43 models on each of the five datasets plus
its equal-dataset overall value: 258 replay targets covering both accuracy and
balanced accuracy, or 516 published numeric cells. It is generated from the
checksummed full-precision authority tables rather than transcribed from the
PDF.

Validate or deterministically rebuild it from the project root:

```bash
scripts/reproduce.sh reviewer catalog --check
scripts/reproduce.sh reviewer catalog
```

Each CSV row includes the expected scores and table-display percentages,
primary and descriptive rank context, exact subject/fold/seed/job counts,
historical A5000 summed job time, data-access flags, authority hashes, and
copy/paste-safe estimate/run/status/compare commands. Set `CACHE_ROOT`,
`REVIEWER_ROOT`, and `GPU` before using a command; their fail-closed shell
expressions reject an unset variable.

Dataset rows support inexpensive targeted cell checks. Overall rows use model
scope and therefore also check that model's five dataset cells. To rerun both
complete tables without duplicating jobs, filter
`recommended_for_complete_table_run=true` and execute only those 43 model
rows. Running all 258 commands would repeat dataset work and is not the
recommended complete-table procedure.

Catalog ranks are sealed-table context. Replaying one row confirms its scores,
not its rank. The four public-source dataset rows require lawful source access
and a private cache; Local Exp4 and all five-dataset overall rows additionally
require institutional authorization.

## Estimate before running

`estimate` is read-only. It resolves the frozen selection and reports its job
count and expected workload without training:

```bash
scripts/reproduce.sh reviewer estimate \
  --scope model \
  --model cardinal_fbc_compactdyn_scale025_extended

scripts/reproduce.sh reviewer estimate \
  --scope dataset \
  --model cardinal_fbc_compactdyn_scale025_extended \
  --dataset bnci2014_004

scripts/reproduce.sh reviewer estimate \
  --scope subject \
  --model cardinal_fbc_compactdyn_scale025_extended \
  --dataset cho2017 \
  --subject 1

scripts/reproduce.sh reviewer estimate \
  --scope job \
  --model cardinal_fbc_compactdyn_scale025_extended \
  --dataset cho2017 \
  --subject 1 \
  --fold 0 \
  --seed 7
```

Runtime depends strongly on model, dataset, GPU, and shared-machine load. In
particular, TCFormer is materially slower than most compact configurations;
job count alone is not a wall-clock estimate.

## Run, resume, and inspect

Use one GPU and a dedicated external run root. For a complete published table
cell:

```bash
scripts/reproduce.sh reviewer run \
  --scope dataset \
  --model cardinal_fbc_compactdyn_scale025_extended \
  --dataset bnci2014_004 \
  --cache-root /absolute/private/cache \
  --run-root /absolute/reviewer-runs/cardinal-bnci004 \
  --gpu 0
```

Use the same command and identical selectors after an interruption. The
immutable selection bound to the run root prevents resuming it as another
model, dataset, subject, fold, or seed.

Read progress without changing the run:

```bash
scripts/reproduce.sh reviewer status \
  --run-root /absolute/reviewer-runs/cardinal-bnci004
```

For an entire model row, omit all narrower selectors:

```bash
scripts/reproduce.sh reviewer run \
  --scope model \
  --model cardinal_fbc_compactdyn_scale025_extended \
  --cache-root /absolute/private/cache \
  --run-root /absolute/reviewer-runs/cardinal-all-five \
  --gpu 0
```

## Compare with the reissued tables

Comparison is permitted only after the exact selected Cartesian set is
complete and valid:

```bash
scripts/reproduce.sh reviewer compare \
  --run-root /absolute/reviewer-runs/cardinal-bnci004 \
  --cache-root /absolute/private/cache
```

The comparison level follows the immutable scope. Dataset scope checks the
selected cell in the reissued dataset summary. Model scope checks all five cells
and the equal-dataset overall row. Subject and job scopes check their
corresponding published participant/job evidence. Accuracy and balanced accuracy
retain the formal fold, seed, participant, and equal-dataset aggregation order.

The comparison reports three deliberately separate conclusions:

- `strict_score_match` compares the applicable replayed accuracy and balanced
  accuracy values with the published full-precision values and is the command's
  success criterion;
- `paper_display_match` asks only whether the value agrees at the precision
  printed in the paper-facing table; and
- `bit_exact_prediction_match` requires the retained prediction digest for
  every selected job to match the published input ledger.

These must not be conflated. For example, a different GPU may reproduce every
predicted class and aggregate score while producing slightly different
floating-point probabilities, yielding a strict score match without a
bit-exact prediction match.

`exact_execution_identity_match` is stronger still: it requires the exact
reissued formal sources and their execution-environment identity. Runtime
drift is recorded rather than silently treated as the same environment. This
field never asserts byte identity with the predecessor release.
`fixed_protocol_replication` means that the immutable model, recipe,
data/cache, split, and selection identities held and the selected replay
completed; it does not claim the same GPU, driver, CUDA, or other
software/hardware execution stack.

A one-model replay cannot independently confirm the model's rank. Rank is a
relationship among all 43 rows; confirming it requires all 43 model results.
Likewise, reproducing one dataset winner requires the other 42 values on that
dataset. The replay may display the published rank as context, but that is not
a newly recomputed ranking.

## Recorded pre-reissue dataset-scope validation (2026-08-19)

This chronological validation predates the sanitized metadata reissue. Its
scores and prediction comparisons remain numerical lineage evidence, but the
artifact hashes below identify that earlier public-package state and are not
current reissue hashes. Release verification must record new hashes after the
sanitized source and metadata are finalized.

Release validation replayed the complete `bnci2014_004` dataset scope for
`cardinal_fbc_compactdyn_scale025_extended`. All 45 planned jobs completed:
45 were valid and there were zero missing, invalid, extra, partial,
quarantined, or unsafe entries. Every one of the 45 replay prediction hashes
matched its published prediction hash.

Both replayed accuracy and balanced accuracy were
`0.7792361111111111`, identical to the published values with absolute delta zero.
The comparison therefore reported `strict_score_match=true`,
`paper_display_match=true`, and `bit_exact_prediction_match=true`.

The locked-runtime assessment compared a 102-package release closure with all
102 installed packages, found no missing, extra, or version-drifted package,
and reported `release_locked_stack_match=true`. It reported
`exact_execution_identity_match=false` solely because the release
`pyproject.toml` and `uv.lock` have documented repaired source identities
rather than the predecessor manifest identities. This is a successful
`fixed_protocol_replication`, not a claim that the repaired release manifests
are the predecessor execution identity.

| Validation artifact or measurement | Value |
|---|---|
| Replay-plan identity | `2bdc42246540dee93a7f736c371ddd308c4dd78039ae21c5c5cb4c9b16eefb9b` |
| `replay_plan.json` SHA-256 | `45a179cd43ea837086ae9b13072a0f44d1da9ac7cf16373c6982651f47014b11` |
| `environment.json` SHA-256 | `b388994d48db035394b0815b0d5369dd101735f062bc55d21aef813bcd0009df` |
| `comparison.json` SHA-256 | `47b0f1b8bcde245047ed1638e752c24a8ee430ef181d167bbe5666b251a48294` |
| Summed recorded job time | `408.60359093360603` seconds |
| Completed run-tree size | 612 KiB |

Private host paths and physical GPU identifiers are intentionally omitted.

## Public and private data

The BNCI2014-001, BNCI2014-004, Cho2017, and PhysioNet cells use public-source
datasets subject to their own terms and provenance requirements. A reviewer
can select any one of those dataset cells without Local Exp4.

`local_exp4` is private and remains blocked for general redistribution pending
institutional governance. Its 40-job cell requires authorized access. The
published equal-dataset model value weights all five datasets equally, so the
complete 2,240-job `model` scope also requires Local Exp4. An average over only
the four public datasets is useful supplementary evidence, but it is not the
published five-dataset overall value and must not be labelled as its
reproduction.

## Replay-output privacy

The replay's no-prediction-retention rule reduces disclosure risk, but it does
not turn a run root into an automatically public artifact. Each immutable job
receipt identifies a pseudonymous dataset participant, fold, and seed and
contains a confusion matrix, accuracy, balanced accuracy, fit/state hashes,
timing, and execution provenance. The comparison artifact aggregates those
derived participant records. This information can still be sensitive,
especially for Local Exp4 and small cohorts.

Keep replay and cache roots access-controlled and outside the Git project. Do
not commit, upload, or attach replay receipts or comparisons to a manuscript
submission until the responsible institution has approved the corresponding
derived-data release. A prediction-archive digest supports equality checking;
it is not permission to distribute the underlying prediction archive or EEG.

## Exact identity versus cross-hardware replication

Fixed seeds and deterministic Torch settings control randomness within the
documented execution contract; they do not guarantee identical floating-point
predictions across GPU architectures, drivers, CUDA builds, Torch builds, or
dependency versions.

Call a replay an exact execution-identity reproduction only when source,
dataset/cache bytes, split identities, Python and dependency lock, Torch/CUDA
stack, driver, and GPU identity all match the reissued formal plan. On a
different GPU or software stack, report it as an independent fixed-protocol
replication and report the observed score and its difference from the
published score. A
matching aggregate on different hardware is valuable evidence, but it does
not make the prediction bytes or execution identity identical.

Operational fields such as timestamps and elapsed time are measurements and
are never expected to match the original run byte for byte.
