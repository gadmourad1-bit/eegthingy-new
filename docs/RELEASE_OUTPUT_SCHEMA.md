# Curated benchmark release output schema

Status: prespecified before any newly planned formal-grid predictions are
joined to labels. This schema controls the final aggregate files; it does not
authorize available-case analysis or pooling across unlike protocols.

## `results/benchmark_summary.csv`

This is a long table with one row per model, track, and eligible dataset, plus
an optional fixed-suite aggregate row. Column order is immutable:

```text
track
protocol_label
model_id
display_name
ownership
dataset
row_kind
coverage_state
coverage_reason
participants
folds_per_participant
seeds
atomic_records
accuracy
balanced_accuracy
macro_f1
cohen_kappa
nll
brier
ece_15
trainable_parameters
fit_seconds_mean
prediction_seconds_mean
peak_cuda_bytes_max
source_result_sha256
```

`row_kind` is exactly `dataset` or `equal_dataset_macro`. The aggregate row is
permitted only when the track contract names a fixed multi-dataset suite and
all required dataset rows are complete. Metric cells are empty for
`planned_incomplete`, `blocked`, and `not_applicable`; they are never imputed.

All metric values are fractions in `[0, 1]`, not percentages. Timing and memory
columns are engineering summaries and do not participate in winner selection.
The canonical CSV uses UTF-8, LF line endings, RFC 4180 quoting, a header, and
decimal numbers generated from finite IEEE-754 float64 values with Python's
shortest round-trip representation.

`results/benchmark_summary.json` contains the same rows, in the same order,
under schema `ieee-mi-curated-benchmark-summary-v1`. It additionally records
the exact input analysis artifacts, source manifest, environment identity,
generation command, and SHA-256 of the CSV bytes.

## `results/model_dataset_coverage.csv`

This file has exactly one row for every stable registry entry and dataset
pair: 70 models by 5 datasets, or 350 data rows. Column order is:

```text
registry_order
model_id
display_name
track
dataset
eligibility
coverage_state
coverage_reason
result_track
result_artifact_sha256
```

`coverage_state` is exactly one of:

- `completed`
- `planned_incomplete`
- `blocked`
- `not_applicable`

The table does not silently convert historical results from another protocol
into completed cells. CHSD scratch variants are unregistered negative
experiments and therefore appear in the report's separate development-screen
section, not as extra registry rows.

## `results/statistical_tests.csv`

This file contains only prespecified or explicitly labelled exploratory
participant-level contrasts. Column order is:

```text
track
contrast_id
analysis_label
metric
scope
dataset
candidate_model_id
comparator_model_id
participants
positive_participants
zero_participants
negative_participants
mean_difference
median_difference
bootstrap_repetitions
bootstrap_seed
confidence_level
interval_kind
ci_lower
ci_upper
p_value
multiplicity_family
p_value_adjusted_holm
source_result_sha256
```

Differences are candidate minus comparator. `analysis_label` is exactly
`prespecified`, `descriptive_selected_winner`, `exploratory_holm`, or
`sensitivity`. Empty inferential cells remain empty where a comparison is
invalid or only numerical context is allowed. Folds, trials, and seeds are
never counted as independent participants.

The primary participant bootstrap uses 100,000 deterministic resamples. The
fixed five-dataset suite resamples participants independently within each
dataset and retains equal dataset weights. A dataset-resampling hierarchical
bootstrap is a separately labelled sensitivity row.

## Ordering and completeness

Rows are ordered by track order from
`docs/BENCHMARK_STATISTICAL_ANALYSIS_PLAN.md`, then immutable registry order,
then dataset order:

```text
local_exp4
bnci2014_001
bnci2014_004
cho2017
physionet_mi
__equal_dataset_macro__
```

Publication generation fails unless every required track audit is quiescent,
every input artifact matches its recorded digest, all completed cells have the
exact expected participant/fold/seed counts, and no live claim, unrecognized
file, quarantine entry, or publication fence exists. There is no partial-table
fallback.

## Human-readable accuracy table

The PDF renders balanced accuracy as a percentage with two decimals, but the
underlying CSV/JSON retains full float precision. Every table title includes
its protocol label. The report may highlight the observed common-recipe winner
only after the complete 43-model grid; it must label that comparison
outcome-selected and descriptive.

Separate tables are mandatory for:

1. the common raw-trial architecture grid;
2. author-recipe adapted references;
3. GeoAdapt harmonized-v2 procedures;
4. local binary procedures;
5. deterministic classical controls;
6. HemiQ-Field harmonized-v2;
7. native CardinalFBMS transfer;
8. native CardinalFBC transfer; and
9. the CHSD robustness-screen negative result.

No overall number is computed across these nine protocol families.
