# Reproducible motor-imagery EEG benchmark

This project packages the code, frozen environment, methods documentation,
and audited aggregate outputs for a 43-configuration, five-dataset
motor-imagery EEG benchmark. The exact common grid completed all **96,320 of
96,320** planned jobs.

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

## What is included

- `ieee_mi/`: common-grid models, loaders, deterministic training, formal
  runner/auditor/analyzer, model registry, and separate research tracks.
- `deepnet/`: earlier geometric, parity, CAMEO, ORBIT, and HemiQ research
  implementations.
- `third_party/TCFormer/`: compact byte-pinned TCFormer runtime snapshot at
  upstream commit `74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`, with its MIT
  license and source manifest.
- `results/common_grid_v6/`: plan, preflight attestation, final exact audit,
  and the sealed 17-file analysis publication. The publication includes
  pseudonymous job-, subject-seed-, and subject-level derived metrics.
- `results/gauge_gate1/`: the separate, prespecified Gauge Gate‑1 negative
  result. Gauge is not in the 43-model leaderboard.
- `results/cardinal_fbms_transfer/`: independent verification report for a
  separate historical native-transfer artifact, not a common-grid score.
- `scripts/`: one command surface for environment setup, tests, private data
  caching, preflight, run/resume, status, audit, and analysis.
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

The clean bundle can be checked without downloading EEG or running training:

```bash
python3 scripts/verify_release.py
```

This verifies the frozen core hashes, required release structure, common
analysis manifest, and `results/SHA256SUMS`. The canonical aggregate analysis
manifest has SHA-256:

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

After the separate docs environment is ready, validate the sealed inputs and
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

## Full reproduction

Exact rerunning is expensive: 43 models × 448 participant/fold units × five
seeds. It requires lawfully acquired copies of all four public datasets,
authorized access to private Local Exp4, substantial storage, and one to three
approved NVIDIA GPUs.

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

`run` and `resume` are the same idempotent command. Do not use the distributed
result directory as a new run root, modify a frozen plan, bypass preflight,
or silently omit Local Exp4 while retaining the five-dataset label. Read
`docs/REPRODUCIBILITY.md` before starting a run.

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
