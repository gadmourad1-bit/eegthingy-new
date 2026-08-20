# Reviewer score-cell catalog

`reviewer_score_cells.csv` is the machine-readable execution catalog for the
two complete 43-model accuracy tables. It contains 258 rows: every model on
each of the five datasets plus its equal-dataset overall value. Each row carries
both standard accuracy and balanced accuracy, so the catalog covers all 516
numeric score cells without duplicating one replay command per metric.

The values are generated from the checksummed `dataset_summary.csv` and
`overall_summary.csv` authorities. Rank, timing, plan, and analysis-manifest
context is independently bound to its source SHA-256. Rebuild or validate the
catalog from the project root:

```bash
scripts/reproduce.sh reviewer catalog
scripts/reproduce.sh reviewer catalog --check
```

`--check` is read-only and fails if the distributed CSV differs from the
deterministic rendering. The catalog is intentionally outside `results/`;
that directory is an immutable sealed bundle and must not receive new files.

## Running a catalog entry

Prepare the locked runtime, then define the three placeholders used verbatim in
the command columns:

```bash
scripts/reproduce.sh setup runtime
export CACHE_ROOT=/absolute/private/cache
export REVIEWER_ROOT=/absolute/reviewer-runs
export GPU=0
mkdir -p "$REVIEWER_ROOT"
```

Run commands from the project root. Cache and reviewer roots must be existing,
real, nonsymlinked, mutually disjoint directories outside the project. Copy the
selected row's `estimate_command`, `run_command`, `status_command`, and
`compare_command`. The `${VARIABLE:?set VARIABLE}` expressions deliberately
fail before training if a placeholder was not set.

- A `dataset` row runs 40, 45, 45, 1,300, or 810 jobs and confirms that one
  model-by-dataset accuracy/ balanced-accuracy pair.
- An `overall` row uses model scope, runs 2,240 jobs, and confirms all five
  dataset pairs plus the equal-dataset overall pair for that model.
- For a complete-table rerun without duplication, filter
  `recommended_for_complete_table_run=true` and run only those 43 model rows.
  Together they cover the original 96,320 jobs. Running every dataset row and
  every overall row separately would duplicate work.

Four dataset rows use public-source data under their original terms. Local
Exp4 rows and every five-dataset overall row require institutional
authorization for the private local data. No transformed cache is
redistributable through this project.

## Interpretation boundary

Successful comparison confirms the selected full-precision accuracy and
balanced-accuracy values. It does not by itself recompute a rank, establish an
independent confirmation cohort, or imply exact execution identity or
bit-identical predictions on different hardware. The catalog does not cover
calibration, complexity, uncertainty intervals, Gauge Gate 1, or the separate
native-transfer panel.

The complete selector, privacy, and comparison contract is documented in
[`docs/REVIEWER_REPLAY.md`](../docs/REVIEWER_REPLAY.md).
