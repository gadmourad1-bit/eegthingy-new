# Comprehensive MI benchmark statistical analysis plan

Status: prespecified before any new formal-grid test labels are joined to
predictions. The CHSD robustness screen is already complete and is governed by
its own earlier decision contract; it is not a candidate in this plan.

## Scientific boundary

The benchmark contains several deliberately separate experiment families.
Their scores may appear in one coverage report, but unlike protocols are not
pooled into one inferential leaderboard:

| Track | Planned or existing atomic records | Comparison boundary |
|---|---:|---|
| Common raw-trial recipe, 43 configurations | 96,320 planned | The only exact five-dataset architecture leaderboard |
| Author-recipe adapted TCFormer and FBCNet | 4,480 planned | Separate two-method reference table |
| GeoAdapt harmonized-v2 procedures | 6,495 planned | Compare only on datasets eligible for both conditions being contrasted |
| CAMEO, HemiParity, PARITY-Fuse, ORBIT-v3 procedures | 8,780 planned | Binary-task procedure table; BNCI2014-001 is principled N/A |
| Deterministic classical controls | 1,344 planned | Seedless control table |
| HemiQ-Field harmonized-v2 adaptation | 45 planned | BNCI2014-004 only |
| CardinalFBMS native-transfer family | 8,675 complete | Cho/PhysioNet target-transfer table |
| CardinalFBC native-transfer family | 6,940 total; 364 development records already exist | Cho/PhysioNet target-transfer table after the remaining 6,576 records |

Historical confirmation artifacts, candidate-search screens, common-recipe
results, author-adapted recipes, native-montage transfer, and harmonized-v2
procedures retain their original labels. A number from one family cannot fill
a missing cell in another.

All newly opened public and local cohorts in these grids are development
evidence. No result from this plan is independent confirmation, clinical
validation, or by itself evidence of global state of the art.

## Units, folds, and seeds

The participant is the independent sampling unit. Trials, folds, and random
seeds are repeated measurements within a participant and must never be treated
as independent replicates.

For stochastic tracks:

1. concatenate prediction-only folds within participant and seed;
2. compute each metric on that concatenated participant-seed prediction;
3. average the five seeds within participant;
4. average participants within dataset with equal participant weight; and
5. for an overall five-dataset value, average dataset means with equal dataset
   weight.

For the deterministic controls, omit the seed step rather than inventing five
copies. For single-dataset HemiQ, stop after equal participant averaging. For
native-transfer tracks, use their frozen target-cohort and fold contracts.

## Outcomes

The primary outcome is balanced accuracy. The primary common-grid aggregate is
equal-dataset macro balanced accuracy over the fixed five-dataset suite.

Secondary outcomes are:

- raw accuracy;
- macro F1;
- Cohen's kappa;
- negative log likelihood;
- multiclass Brier score; and
- expected calibration error with exactly 15 equal-width confidence bins.

ROC AUC may be reported only where its binary or multiclass definition was
fixed in the track contract and every required class is present. Timing,
trainable parameter count, peak CUDA memory, and prediction latency are
engineering outcomes, not substitutes for accuracy.

## Winner language

The common-recipe “winner” is the configuration with the largest observed
equal-dataset macro balanced accuracy, with ties broken only by the immutable
roster order for deterministic table construction. Because the winner is
outcome-selected from 43 configurations, its comparison with the runner-up or
with another outcome-selected model is descriptive. Bootstrap intervals around
such a selected contrast are labelled descriptive intervals, not
selection-adjusted confirmatory confidence intervals.

TCFormer is the prespecified public common-recipe reference. Subject-paired
differences against TCFormer may be tabulated for every fixed architecture,
but the family is exploratory and multiplicity is controlled with Holm's
method within each dataset and metric family. The same data cannot be used to
select a best model and then present an unadjusted superiority test for it.

The author-recipe table may compare its two fixed references directly.
Procedure and transfer tables may make paired contrasts only when both rows
share the exact dataset, participants, folds, seeds, inputs, and evaluation
contract. All other cross-track comparisons are numerical context only.

## Uncertainty and sensitivity

Participant-level paired mean differences are the default contrast. Report:

- the observed mean and median paired difference;
- a participant bootstrap interval with 100,000 deterministic resamples;
- per-dataset effects before any overall aggregate;
- seed-specific sensitivity rows for stochastic tracks; and
- the number of participants with positive, zero, and negative paired effects.

For the fixed five-dataset suite, resample participants independently inside
each dataset and retain equal dataset weights. A second hierarchical bootstrap
that also resamples datasets may be shown as a sensitivity analysis and must be
labelled a dataset-superpopulation extrapolation. With only five datasets, it
is not the primary interval.

Missing jobs, corrupt records, live claims, source/environment drift, or a
failed exact audit prevent analysis publication. There is no available-case
fallback. A failed seed is rerun under the same immutable plan or the whole
affected comparison remains incomplete.

## Coverage and N/A reporting

The final 70-model by 5-dataset coverage table uses exactly four cell states:

- completed under the named track;
- planned but incomplete;
- blocked by a stated adapter/provenance decision; or
- principled N/A.

Binary-only procedures are N/A for the four-class BNCI2014-001 task. HemiQ is
N/A outside the supplied three-channel BNCI2014-004 bipolar derivations.
Unpromoted ORBIT variants and CardinalSplineDualView remain blocked rather than
receiving post-hoc reconstructed configurations. No missing or N/A cell is
imputed into an overall average.

## Publication outputs

The curated release regenerates, from immutable audited track outputs:

- `results/benchmark_summary.csv` and `.json`;
- `results/model_dataset_coverage.csv`;
- `results/statistical_tests.csv`; and
- `reports/benchmark_report.pdf`.

The exact table columns, row order, coverage vocabulary, numeric encoding, and
separate protocol sections are frozen in
`docs/RELEASE_OUTPUT_SCHEMA.md`.

The PDF includes protocol labels beside every table, the CHSD negative result,
all dataset-specific common scores, the equal-dataset common ranking, separate
procedure/reference/control/transfer tables, uncertainty, resource use,
limitations, and provenance hashes. It contains aggregate statistics only.
Raw EEG, labels, trial-level predictions, and private participant metadata stay
outside the clean project.
