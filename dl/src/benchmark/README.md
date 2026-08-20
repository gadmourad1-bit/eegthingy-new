# `benchmark` package

`benchmark` is the single active Python package. Its root modules form the
formal protocol layer and
owns dataset and split contracts, the 43-configuration roster, deterministic
training, external-model adapters, full-grid execution, GPU coordination,
auditing, aggregate analysis, and bounded reviewer replay.

It is one layer of the single `benchmark` namespace—not a separate
project or installable package. Exploratory model implementations live in
`benchmark.research`; dependency-light numerical primitives live in
`benchmark.shared`.

## Supported interface

Run reviewed workflows from the project root through:

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

The wrapper also exposes `prepare-data`, `preflight`, `run` / `resume`,
`status`, `audit`, and `analyze`. Separate research-track module CLIs are
supported only when their procedure document explicitly prescribes them.

For reviewer-scale verification, use:

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

The exact selector and comparison contract is documented in
[`docs/REVIEWER_REPLAY.md`](../../docs/REVIEWER_REPLAY.md).

## Internal map

| Area | Representative modules |
| --- | --- |
| Dataset and split contract | `config.py`, `data.py` |
| Training and model construction | `training.py`, `models.py`, `baselines.py` |
| Model metadata | `model_registry.py`, `tcformer_source.py` |
| Formal common grid | `full_grid.py`, `project_gpu_leases.py` |
| Aggregate analysis | `full_grid_analysis.py` |
| Bounded result replay | `reviewer_replay.py` |
| Separate protocol tracks | `control_grid*`, `procedure_grid*`, `gauge_*`, `hemiq_v2_*`, `geoadapt_v2*`, `native_*` |

Not every module belongs to the common 43-model leaderboard. Read
[`docs/PROTOCOL_BOUNDARIES.md`](../../docs/PROTOCOL_BOUNDARIES.md) before
combining results.

## Frozen result boundary

The completed common-grid v6 evidence predates the current source
organization. Its sanitized replay closure is retained under
`historical/common_grid_v6/source/eeg_mi_v6/`, outside the active import path.
The distributed result bundle remains the authority for the published scores.

The live `benchmark` sources create a new source identity.
They must not be described as a byte-identical continuation of the completed
execution, even when the fixed protocol and numerical results are reproduced.

## Randomness contract

The common grid fixes five training seeds: `7`, `17`, `27`, `37`, and `47`.
All five are required for every planned dataset/model/participant/fold unit.
The analysis separately fixes bootstrap seed `20260729` for 100,000
resamples. Reducing the schedule to one seed is a different experiment.

The active environment authority is the project-root `pyproject.toml` plus
`uv.lock`. The local `requirements-cu128.txt` is historical provenance and
must not be installed for the current release.
