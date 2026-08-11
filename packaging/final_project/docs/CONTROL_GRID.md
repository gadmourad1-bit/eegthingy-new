# Deterministic classical-control grid

`ieee_mi.control_grid` is the audited CPU-only prediction producer for the
three classical controls in the model registry:

| Stable ID | Registered implementation | Input |
|---|---|---|
| `control.riemann` | `deepnet.baselines.RiemannianTangentLogistic` | four-band SPD covariances |
| `control.tangent_anchor` | `deepnet.tangent_anchor.TangentAnchorClassifier` | four-band SPD covariances |
| `control.ea_fbcsp` | `deepnet.baselines.EAFilterBankCSP` | four-band filtered epochs |

The runner checks the live registry before planning and before every worker
starts. It fails with `RegistryDriftError` if a control is removed, made
ineligible, changed to binary-only, or rebound to another implementation.
All three implementations support binary and multiclass classification. In
particular, BNCI2014-001 remains a four-class task; it is never collapsed to a
binary task.

## Formal cardinality

A deterministic estimator has no meaningful optimization-seed replication.
The atomic key is therefore
`(control, dataset, subject, fold)` and contains no seed field.

| Dataset | Subjects | Folds | Atomic executions per control | Records across 3 controls |
|---|---:|---:|---:|---:|
| local Exp4 | 8 | 1 | 8 | 24 |
| BNCI2014-001 | 9 | 1 | 9 | 27 |
| BNCI2014-004 | 9 | 1 | 9 | 27 |
| Cho2017 | 52 | 5 | 260 | 780 |
| PhysioNet MI development S1–54 | 54 | 3 | 162 | 486 |
| **Total** |  |  | **448** | **1,344** |

Older local artifacts emitted five rows which explicitly aliased the same
deterministic fit to five neural seed IDs. Those aliases are not independent
runs and are not reproduced by this runner. The current track matrix and
`NON_COMMON_TRACK_EXECUTION.md` use the same seedless 448-execution,
448-record-per-control contract. A comprehensive table should ingest one
control record per subject-fold.

## Leakage boundary

Each atomic job executes the following fixed protocol.

1. Fit the channel scaler on training rows.
2. Apply the fixed four-band FIR transform and trace-shrunk covariance
   transform. Neither transform estimates cross-trial state.
3. Fit each prespecified candidate on training rows and select by validation
   negative log likelihood. Validation losses are not persisted.
4. Construct a new estimator. Refit the channel scaler and every estimator
   state on train plus validation.
5. Call the final estimator's `predict_proba` once on the held-out features.
   `calibrate` is never called. A before/after state digest rejects an
   estimator that mutates during held-out inference.

The grids are inherited from the existing local control protocol:

- Riemann: `C ∈ {0.01, 0.1, 1, 10}`;
- tangent anchor: `C ∈ {0.01, 0.1, 1, 10}`; and
- EA-FBCSP: `n_components ∈ {2, 4, 6}` subject to the prespecified feasibility
  rule `n_components <= n_channels`. Thus the three-channel BNCI2014-004 view
  uses `n_components=2`.

Every selection candidate must expose the complete canonical class order.
Returned probabilities must be finite, normalized, and have exactly the
registered class count. A real implementation which cannot return four
BNCI2014-001 columns raises `EstimatorCapabilityError`; the runner does not
remap, duplicate, or fabricate a class.

## Score-blind artifact

Every completed directory contains exactly:

- `record.json`: job identity, immutable plan digest, source-only fit
  provenance, scaler/state hashes, timings, and resource contract;
- `predictions.npz`: only `rows` (`int64`) and `probabilities` (`float64`); and
- `completion.json`: SHA-256 digests of the other two files.

No trial outcome, predicted class, validation loss, test score, or performance
metric is written. JSON objects use exact field allowlists in addition to
recursive outcome/score-key rejection, so a rehashed artifact cannot hide an
outcome under an unrecognized metadata field. The built-in audit validates
identity, canonical JSON, checksums, row hashes, probability
shape/range/normalization, exact cardinality, missing jobs, corruption,
unexpected record paths, abandoned partials, and residual claims/CPU slots
without joining outcomes or computing a score.

A later analysis adapter may join an audited `rows` array to the frozen cache
outcomes. That analysis is intentionally outside this prediction producer.

## Immutable identities and resume

`plan.json` and `plan.sha256` freeze:

- the exact five opened development cohorts, subjects, folds, classes, and
  preprocessing contracts;
- each harmonized-v2 cache array identity and every train/validation/source/test
  row count and SHA-256;
- source hashes for the runner, data/config/feature code, registry, both
  estimator implementations, TangentAnchor's transitive `deepnet/spd.py`,
  `pyproject.toml`, and `uv.lock`;
- Python executable/prefix/version, platform, UV version, and the complete
  installed-package identity;
- estimator grids and output protocol; and
- CPU budget, per-worker thread cap, slot count, and 50 GiB disk low-water mark.

`dataset_order` is the explicit prespecified scientific order. Canonical JSON
may sort object keys on disk, so plan loading restores the `datasets` mapping
to that frozen order before rebuilding and validating the complete plan. It
never substitutes alphabetical object-key order for the experimental order.

Workers refuse source, environment, cache, split, registry, or plan-digest
drift. Job claims and CPU slots are exclusive filesystem records containing a
host/PID/process-start identity. A dead local owner is quarantined before
recovery; a claim from another host is treated as live, and an ambiguous start
marker is never permission to delete a live local owner's record. Every job
claim must prove ownership of a current plan-bound CPU slot. Completed files
are published through a unique partial directory and one atomic directory
rename. Failures are recorded separately and fail the worker rather than
silently marking a job complete. The supervisor stops only its own sibling
workers as soon as one child fails; it never signals foreign processes.

After a power cut, rerun the same `run` command with the same plan digest.
Checksummed completed jobs are skipped, stale local claims are quarantined, and
unfinished jobs resume.

## Shared-workstation and UV usage

The CLI requires an isolated UV-created virtual environment with
`include-system-site-packages = false` and UV on `PATH`. It never invokes an
installer and never writes to the system Python. Planning also refuses a
project unless NumPy, SciPy, scikit-learn, MNE, pyRiemann, Torch, and
threadpoolctl are direct dependencies present in `uv.lock`. Create the final
environment with the project's locked dependencies, for example:

```bash
cd /home/hanafy/neuralnetwork
export PATH="/home/hanafy/.local/bin:$PATH"
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync --frozen
uv pip check
```

Create the immutable plan:

```bash
uv run --frozen --no-sync python -m ieee_mi.control_grid plan \
  --run-root /home/hanafy/neuralnetwork/results/control_grid \
  --cache-root /home/hanafy/scratchpad/eeg_novel_20260729/data_cache \
  --cpu-budget-threads 16 \
  --threads-per-worker 4 \
  --minimum-free-gib 50
```

Copy the printed `plan_sha256` exactly, then start or resume:

```bash
uv run --frozen --no-sync python -m ieee_mi.control_grid run \
  --run-root /home/hanafy/neuralnetwork/results/control_grid \
  --cache-root /home/hanafy/scratchpad/eeg_novel_20260729/data_cache \
  --workers 4 \
  --plan-sha256 <printed-plan-sha256>
```

Run the score-blind audit:

```bash
uv run --frozen --no-sync python -m ieee_mi.control_grid audit \
  --run-root /home/hanafy/neuralnetwork/results/control_grid \
  --output /home/hanafy/neuralnetwork/results/control_grid/audit.json
```

Inside the immutable run tree the audit writer accepts only `audit.json`; it
cannot be pointed at `plan.json`, a record, or another run artifact.

The default cooperative budget is 16 CPU threads: four workers with four
threads each. Workers acquire one of the plan's filesystem CPU slots and set
OpenMP, MKL, OpenBLAS, BLIS, TBB, NumExpr, Numba, Accelerate, and Torch CPU
limits. `CUDA_VISIBLE_DEVICES` is empty in worker processes. There is no
`nvidia-smi` probe, GPU claim, DDP process, or GPU allocation. Before each new
job claim and again immediately before atomic publication, the output
filesystem must have at least the plan's frozen 50 GiB free; a manually
rehashed plan cannot lower that floor.

## Verification

The focused test module is:

```bash
MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/eegthingy-mpl \
UV_CACHE_DIR=/tmp/eegthingy-uv-cache \
uv run --frozen --no-sync \
python -m pytest -q ieee_mi/tests/test_control_grid.py
```

It covers exact production cardinality, registry binding, immutable plan
checksums, seedless jobs, the EA channel-feasibility rule, atomic score-blind
publication, exact schemas, completion checksums, outcome aliases, low-disk
failure, stale/foreign ownership, CPU-slot proof, fail-fast child supervision,
the train/validation/refit/test call boundary, and real four-class estimator
probabilities. A machine without Torch skips only the real TangentAnchor
multiclass smoke test; the lab UV environment must run that test before formal
execution because TangentAnchor's implementation imports Torch.
