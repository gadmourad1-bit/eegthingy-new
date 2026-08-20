# Classical-control score-joining analysis

`benchmark.control_grid_analysis` is the fail-closed analysis companion to the
score-blind deterministic control grid. It is a separate classical,
non-neural track. Its outputs must not be merged into the neural
common-recipe roster or presented as stochastic seed replications.

## Audit and leakage boundary

The analyzer opens no cache until all of the following checks pass:

1. `plan.json` and `plan.sha256` are regular, read-only files accepted by the
   canonical control-plan validator.
2. The plan's complete dataset contract exactly equals the live sealed
   contract, including dataset order, subjects, folds, class counts,
   protocols, and preprocessing. Cardinality alone is not accepted.
3. The plan contains exactly 1,344 unique
   `(control, dataset, subject, fold)` jobs: 448 per control and no seed
   dimension.
4. The score-blind audit validates every record, completion checksum, row
   identity, probability archive, and exact records-tree path.
5. There are no missing or corrupt jobs, unexpected record/root paths,
   unpublished partials, job claims, or CPU slots.

Only then does the analyzer invoke the runner's live runtime-identity
verification and open a plan-bound cache. That verifier rechecks the control
registry, runner source closure, complete UV environment, every cache
identity, and every reconstructed split identity. Cache loading recomputes the
harmonized array digest. The analyzer also checks class range, channel/trial
counts, and every reconstructed train/validation/source/test row digest.
Labels are joined to held-out row indices in memory only.

Before returning, the analyzer rehashes:

- both immutable plan files;
- all 132 loaded cache identities and their split-identity groups; and
- every `completion.json`, `record.json`, and `predictions.npz` file.

It then repeats both the exact control audit and live runtime verification.
The private checksum ledger remains inside the `AnalysisResult`; only its
SHA-256 is published.

The analysis summary binds the analyzer, metric-reference implementation,
complete control-runner source closure, `pyproject.toml`, and `uv.lock` by
SHA-256. Environment provenance includes the full package inventory and its
digest, Python executable/prefix/platform identity, UV version, and explicit
versions for MNE, NumPy, pyRiemann, scikit-learn, SciPy, threadpoolctl, and
PyTorch.

## Metrics and aggregation

The metric definitions are byte-for-byte tested against
`full_grid_analysis.classification_metrics`:

- accuracy;
- balanced accuracy;
- chance-normalized balanced accuracy,
  `(BA - 1/K) / (1 - 1/K)`;
- macro F1 over the planned class set;
- Cohen kappa;
- one-vs-rest macro AUROC;
- negative log likelihood with the true-class probability clipped at
  `1e-15`;
- multiclass Brier score as the mean sum of squared class-probability errors;
  and
- top-label ECE with 15 equal-width confidence bins by default.

Aggregation proceeds in this order:

1. Compute one scalar job/fold row for each held-out prediction artifact.
2. Concatenate disjoint folds within a control/subject and recompute metrics
   from the concatenated held-out predictions.
3. Average subject metrics equally within each dataset.
4. Average the five dataset means equally for each control.

This yields the following formal table cardinalities:

| Published table | Rows | Unit |
|---|---:|---|
| `job_metrics.csv` | 1,344 | control/dataset/subject/fold |
| `fold_metrics.csv` | 1,344 | control/dataset/subject/fold |
| `subject_metrics.csv` | 396 | control/dataset/subject |
| `dataset_summary.csv` | 15 | control/dataset |
| `overall_summary.csv` | 3 | control |
| `timing_summary.csv` | 18 | 15 control/dataset plus 3 all-dataset rows |

The job table also reports candidate count, estimator-fit calls, parameter
count, selection/refit/inference/job time, and inference milliseconds per
trial. These are scalar provenance summaries, not optimization-seed
replicates.

## No invented inference or ranking

There is no prespecified candidate-versus-control contrast in this track.
Consequently, the analyzer publishes no bootstrap interval, p-value,
multiple-testing adjustment, or seed-variance estimate. Controls remain in
immutable registry order in `RESULTS.md`; they are not sorted into a winner
ranking.

Absolute values can be compared descriptively in the comprehensive paper
table, but their protocol and classical-track label must remain visible.

## Aggregate-only publication

Publication is allowed only outside both `run_root` and `cache_root`, including
through resolved symlink aliases. The publisher obtains an exclusive
destination lock and, while owning that lock:

1. repeats the complete quiescent audit;
2. repeats the live source, registry, UV environment, cache, and split
   identity verification;
3. reopens every cache and prediction;
4. recomputes all six scalar tables; and
5. requires the recomputed summary, tables, and private ledger to be
   byte-identical to the supplied sealed result.

It then writes a unique staging directory, fsyncs every file, and atomically
renames the directory into place. Immediately before that rename it repeats
the complete audit, live runtime verification, and private-ledger rehash,
closing the staging-time input window. Lock cleanup verifies the lock's
device, inode, and random token, so a failed publisher cannot delete another
publisher's replacement lock.

The destination contains exactly:

- `analysis.json`;
- `RESULTS.md`;
- the six aggregate CSV tables listed above; and
- `manifest.json` with hashes of every published artifact.

There is no published trial label, row-index vector, class-decision vector,
probability matrix, or checksum-ledger table. Exact schemas and recursive
forbidden-key checks reject attempts to add them. No published JSON or CSV
schema contains a seed field.

## UV execution

The analyzer requires the same isolated UV-created virtual environment policy
as the producer and never installs a package:

```bash
cd /home/hanafy/neuralnetwork
export PATH="/home/hanafy/.local/bin:$PATH"
source .venv/bin/activate

uv run --frozen --no-sync python -m benchmark.control_grid_analysis \
  --run-root /home/hanafy/neuralnetwork/results/control_grid \
  --cache-root /home/hanafy/scratchpad/eeg_novel_20260729/data_cache \
  --output-dir /home/hanafy/neuralnetwork/reports/control_grid_analysis
```

The output directory must not already exist; publication never overwrites an
earlier analysis.

## Verification

Run the bounded analyzer tests without changing the environment:

```bash
MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/eegthingy-mpl \
UV_CACHE_DIR=/tmp/eegthingy-uv-cache \
uv run --frozen --no-sync \
python -m pytest -q tests/core/test_control_grid_analysis.py
```

The tests cover four-class metric parity, a seedless end-to-end mini-grid
aggregation, exact sealed formal roster enforcement (including an
unchanged-cardinality PhysioNet S54-to-S55 substitution), source/registry/UV
environment/cache/split drift, audit-before-label ordering, publication-time
runtime revalidation, full source and MNE provenance, raw/label-table
fabrication, ordinary and rehashed checksum-ledger corruption, incomplete and
stray runs, output paths under the run/cache roots, symlink aliases, absence
of seed fields, aggregate-only file publication, and foreign replacement-lock
preservation.
