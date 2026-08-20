# Reproducible cross-dataset EEG benchmark

This project packages a general EEG benchmarking codebase together with the
frozen environment, methods documentation, and audited aggregate outputs for
its completed 43-configuration, five-dataset motor-imagery study. The active
package is task-neutral; the released experiment remains explicitly identified
as motor imagery because that is the classification target evaluated in the
published table. The exact common grid completed all **96,320 of 96,320**
planned jobs.

The primary result is deliberately modest: under one shared training recipe
on previously opened development cohorts,
`cardinal_fbc_compactdyn_scale025_extended` was the observed descriptive
leader at **73.9900% equal-dataset balanced accuracy**. TCFormer ranked 12th at
73.3163%. The selected leader's +0.674-point difference from TCFormer had a
descriptive fixed-suite 95% bootstrap interval of **[-0.427, +1.899] points**.
The interval includes zero.

These results are not independent confirmation, a global/state-of-the-art
claim, proof of architectural novelty, or clinical evidence. Most data came
from healthy volunteers; the benchmark does not demonstrate benefit or safety
for disabled or paralyzed people.

## Supported interface and project layout

Use `scripts/reproduce.sh` for release operations, read-only common-grid v6
verification, bounded reviewer replay, and new-run workflows. Separate
auxiliary research tracks may have reviewed module-specific CLIs in their
dedicated procedure documents; those commands do not alter or extend the
common-grid leaderboard.

The implementation uses one Python namespace, `benchmark`, under a standard
`src/` layout with three explicit layers:

- `src/benchmark/` owns datasets, model adapters, training protocols, runners,
  audits, analysis, and reviewer replay;
- `research/` owns exploratory and historical model implementations; and
- `shared/` owns dependency-light numerical primitives used across layers.

Tests live separately under `tests/{core,research,shared}`. The dependency
direction is `benchmark -> research -> shared`; `shared` has no upward import,
and `research` does not import `benchmark`. This removes the former package
cycle and makes ownership visible from the directory tree.

The completed common-grid v6 evidence predates this source reorganization.
Its replay closure remains under `historical/common_grid_v6/source/eeg_mi_v6/`.
The scientific tables and numeric values are unchanged, while any new run
created from the live package necessarily has a new source identity. See
`docs/PROJECT_STRUCTURE.md` for the complete boundary.

## What is included

- `src/benchmark/`: the formal benchmark and protocol layer.
- `src/benchmark/research/`: exploratory GeoAdapt, parity, CAMEO, ORBIT,
  HemiQ, and comparison implementations.
- `src/benchmark/shared/`: SPD algebra, augmentation, CSP initialization,
  and common metric primitives.
- `tests/`: tests organized by the same three responsibility layers.
- `historical/common_grid_v6/source/eeg_mi_v6/`: read-only, token-free source
  reissue for interpreting and replaying the common-grid v6 evidence. It is
  isolated from the active package and carries its own checksummed identity.
- `historical/native_fbms_full_grid_v1/source/`: read-only 11-file source
  closure, under `eeg_mi_native_v1/`, for validating the separate native-FBMS
  v1 checkpoint and completed transfer artifacts. It is evidence, not
  executable live code.
- `third_party/TCFormer/`: compact byte-pinned TCFormer runtime snapshot at
  upstream commit `74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`, with its MIT
  license and source manifest.
- `results/common_grid_v6/`: sanitized plan, preflight, audit, and analysis
  metadata reissue for the completed run. Its scientific analysis CSVs are
  byte-preserved and its JSON numeric values are unchanged; namespace-bearing
  identities and checksums are reissued. The publication includes pseudonymous
  job-, subject-seed-, and subject-level derived metrics.
- `results/gauge_gate1/`: the separate, prespecified Gauge Gate‑1 negative
  result. Gauge is not in the 43-model leaderboard.
- `results/cardinal_fbms_transfer/`: independent verification report for a
  separate historical native-transfer artifact, not a common-grid score.
- `scripts/`: one command surface for environment setup, tests, private data
  caching, preflight, run/resume, status, audit, and analysis.
- `reviewer/`: deterministic score-cell catalog linking every published
  accuracy/balanced-accuracy cell to its bounded replay commands.
- `docs/`: methodology, protocol boundaries, result interpretation,
  reproducibility, data rights, ethics, and release blockers.

No raw EEG, transformed cache, trial-level prediction, direct participant
identifier, demographic/clinical metadata, embedding, or checkpoint is
included. The pseudonymous participant keys and derived metrics in
`job_metrics.csv`, `subject_seed_metrics.csv`, and `subject_metrics.csv` still
require privacy and Local Exp4 governance review before public release.

## Read the result correctly

The primary aggregation concatenates folds within participant and seed,
averages seeds within participant, averages participants within dataset, and
then weights the five datasets equally. Trials, folds, and seeds are repeated
measurements, not independent samples.

The observed Local Exp4 leader was a different configuration,
`cardinal_dynamics_sinc_extended`, at 91.6250% balanced accuracy. The overall
leader scored 91.2500% locally and TCFormer scored 91.2083%. Selecting a local
winner does not make it an all-dataset winner, and selecting either on opened
development outcomes makes subsequent superiority claims exploratory.

See:

- `docs/RESULTS_INTERPRETATION.md` for the verified ranking, uncertainty,
  calibration, and negative-result interpretation;
- `docs/METHODOLOGY_AND_ARCHITECTURES.md` for the shared recipe and all
  in-house architecture families;
- `docs/PROTOCOL_BOUNDARIES.md` before combining any score tables; and
- `results/common_grid_v6/analysis/RESULTS.md` for the analyzer-generated
  complete 43-model summary.

## Quick verification

The clean bundle can be checked without downloading EEG, creating a UV
environment, or running training (Python 3.11 or newer is sufficient):

```bash
scripts/reproduce.sh verify
```

This runs both the whole-release and protocol-separated results verifiers. It
checks the sanitized v6 source reissue, the complete live-source inventory,
the predecessor-digest lineage, required release structure, the common analysis
manifest, and `results/SHA256SUMS`. Auditors with Python 3.11+
may invoke the two underlying standard-library verifier scripts directly, but
the wrapper is preferred because it isolates Python and runs both. The
canonical aggregate analysis manifest has SHA-256:

```text
960ca912d058a2ad3b60def4116dbba436f8ae9162916cde42a4ab265bf01f5d
```

The exact final audit reports 96,320 complete jobs, zero missing/extra/failed
jobs, zero live or stale claims, zero partials, a valid score-blind preflight,
and six resolved stale-claim artifacts left by interrupted workers. Its file
SHA-256 is:

```text
010b44542157efb6ba5a95376198b9b691e9803ce3fe605e77cb5cdb9739d063
```

## Frozen randomness and reproducibility

Common-grid v6 does not rely on an unspecified default seed. Its immutable
plan fixes five training seeds: **7, 17, 27, 37, and 47**. Each
dataset/model/participant/fold combination is evaluated under all five seeds;
the reported aggregation averages seeds within participant. Replacing this
schedule with one seed would produce a different experiment and would not
reproduce the 96,320-job benchmark. The statistical analysis separately fixes
the 100,000-resample bootstrap seed at **20260729**.

The supported launcher fixes the recorded CUDA determinism environment.
Training seeds Python, NumPy, Torch, and all CUDA generators before model
construction, disables cuDNN benchmarking, enables deterministic algorithms,
and uses explicitly seeded augmentation. Common-grid v6 did not record
`PYTHONHASHSEED`; adding it retroactively would change the execution contract.
It is a future-version hardening option, not part of the completed v6 claim.

“Same output” has two precise meanings here:

- On the same reviewed hardware/software execution identity, with the exact
  reissued cache, plan, dependency lock, and seed schedule, scientific
  predictions and derived scores are intended to reproduce.
- Wall-clock timestamps, process identifiers, resource timings, and similar
  operational metadata naturally change between executions, so a newly run
  artifact tree is not promised to be byte-for-byte identical. Different GPU,
  driver, CUDA, Torch, dependency, or data identities are also new execution
  identities even when their aggregate scores happen to agree.

The publication PDFs are a separate derived product. Given the exact same
reissued input bytes, report-input documentation and registry bytes, report
builder bytes, and locked documentation environment, their invariant builder
is intended to emit the same PDF bytes.

Inspect and machine-check this reissued identity at any time:

```bash
scripts/reproduce.sh identity
```

The command recomputes the canonical plan digest and verifies its sidecar,
five training seeds, analysis seed and resample count, planned job count,
recorded and source cuBLAS contract, and all three publication-PDF hashes
without opening EEG data.

## Environment setup

The project targets Linux x86-64, CPython 3.12.13, and the matching CUDA 12.4
builds of PyTorch and TorchAudio 2.6.0 (`2.6.0+cu124`). Both packages are
pinned to the explicit PyTorch cu124 index. The project uses project-private
UV environments and never needs a system-Python install.

```bash
uv --version
scripts/reproduce.sh setup runtime
scripts/reproduce.sh setup test
scripts/reproduce.sh setup docs
scripts/reproduce.sh test
```

After the separate docs environment is ready, validate the reissued inputs and
build the three deterministic publication PDFs under `output/pdf/`:

```bash
scripts/reproduce.sh reports
scripts/reproduce.sh reports --output-dir /absolute/report-directory
scripts/reproduce.sh reports --validate-only
```

The frozen lock and scripted test environments support Linux x86-64 only.
macOS can run the standard-library bundle verifiers and inspect the source, but
it cannot synchronize this lock, run the documented test command, reproduce
the formal CUDA execution identity, or reproduce the recorded GPU results.

## Reviewer-scale replay

A reviewer with one CUDA GPU can rerun one model or one smaller reissued-plan
projection without launching the 96,320-job grid:

```bash
scripts/reproduce.sh reviewer catalog --check
scripts/reproduce.sh reviewer list
scripts/reproduce.sh reviewer estimate \
  --scope dataset \
  --model cardinal_fbc_compactdyn_scale025_extended \
  --dataset bnci2014_004
scripts/reproduce.sh reviewer run \
  --scope dataset \
  --model cardinal_fbc_compactdyn_scale025_extended \
  --dataset bnci2014_004 \
  --cache-root /absolute/private/cache \
  --run-root /absolute/reviewer-run \
  --gpu 0
scripts/reproduce.sh reviewer status --run-root /absolute/reviewer-run
scripts/reproduce.sh reviewer compare \
  --run-root /absolute/reviewer-run \
  --cache-root /absolute/private/cache
```

The four exact scopes are `job` (1 job), `subject` (all folds and all five
seeds; 5, 15, or 25 jobs), `dataset` (one complete table cell; 40–1,300 jobs),
and `model` (all five dataset cells; 2,240 jobs). Dataset, subject, and model
scopes never reduce the five-seed schedule. A one-model run can confirm its
scores but cannot independently recompute its rank among 43 models.

The deterministic [`reviewer/reviewer_score_cells.csv`](reviewer/reviewer_score_cells.csv)
contains all 258 model-by-target entries and both table metrics, with exact
expected values, job counts, access flags, timing context, source hashes, and
copy/paste-safe commands. Select one dataset row for a single table cell, or
filter `recommended_for_complete_table_run=true` to obtain the 43 nonduplicated
model runs that cover both complete tables.

Four dataset cells use public-source data. The Local Exp4 cell and the complete
five-dataset model row require authorized private-data access. Runs on another
GPU/software identity are fixed-protocol replications, not promises of
bit-identical predictions. See the
[reviewer replay guide](docs/REVIEWER_REPLAY.md) for the exact selector
contract, counts, commands, comparison meaning, and data boundary.

## New full-grid replication

Running the complete fixed recipe is expensive: 43 models × 448
participant/fold units × five seeds. It requires lawfully acquired copies
of all four public datasets, authorized access to private Local Exp4,
substantial storage, and one to three approved NVIDIA GPUs.

```bash
scripts/reproduce.sh prepare-data /absolute/private/cache /absolute/private/local-exp4
scripts/reproduce.sh preflight /absolute/private/cache /absolute/new-run 0
scripts/reproduce.sh run /absolute/private/cache /absolute/new-run 0,1,2
scripts/reproduce.sh status /absolute/new-run
scripts/reproduce.sh audit /absolute/new-run /absolute/new-final-audit.json
scripts/reproduce.sh analyze \
  /absolute/new-run \
  /absolute/private/cache \
  /absolute/new-analysis
```

These commands create and execute a plan under the active
`benchmark` source
identity. They are suitable for a new fixed-recipe replication, but they are
not a continuation of the completed predecessor execution. The bounded
`reviewer` command is the supported v6 score-replay interface.

`run` and `resume` are the same idempotent command for a compatible new run
root. Do not use the distributed result directory as a run root, modify a
frozen plan, bypass preflight, or silently omit Local Exp4 while retaining the
five-dataset label. Read `docs/REPRODUCIBILITY.md` before starting a run.

## Data, ethics, and privacy

Data access is intentionally external. Public availability does not authorize
redistributing transformed EEG, and the supplied materials do not establish
the Local Exp4 IRB/consent/data-use boundary. PhysioNet S55–S109, Lee2019,
Weibo2014, and Zhou2016 remain sealed for a future prespecified confirmation
protocol.

See `docs/DATA_ACCESS.md` and `docs/ETHICS_AND_PRIVACY.md`. Do not commit raw
data, trial predictions, embeddings, weights, direct identifiers, or private
metadata. Treat the included pseudonymous participant-level metric tables as
sensitive derived data until their release is explicitly approved.

## License and citation status

The in-house project license, copyright holders, authorship, affiliations,
ORCIDs, funding, conflicts, and final citation metadata have not been supplied
and are not inferred here. Public redistribution is blocked until the
responsible people and institution resolve them.

- `LICENSE_DECISION_REQUIRED.md` explains the code-license blocker.
- `AUTHORS_TEMPLATE.md` is an authorship/CRediT decision worksheet.
- `CITATION.cff.template` must not be renamed until its placeholders are
  replaced and validated.
- `THIRD_PARTY_NOTICES.md` records the current third-party audit, including
  TCFormer's preserved license.

The lack of a project license does not alter third-party licenses or dataset
terms.
