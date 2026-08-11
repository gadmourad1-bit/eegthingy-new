# Notes on the sealed common-grid output

The 17 files under `analysis/` are an immutable, manifest-bound publication.
This note is outside that directory. It clarifies two sentences in the sealed
human-readable `analysis/RESULTS.md`; it does not change any result, interval,
manifest entry, or authoritative machine-readable field.

## Fixed-suite and dataset-superpopulation intervals

The interval table displayed in `analysis/RESULTS.md` reports the descriptive
**fixed-suite** comparison with TCFormer. For every bootstrap replicate, all
five planned datasets are retained and weighted equally, while paired
participant clusters are resampled independently within each dataset. This is
the method identified by `analysis.json` as `headline_fixed_suite` and by the
`descriptive_fixed_suite_bootstrap_interval95_*` columns in
`analysis/tcformer_overall_context.csv`.

The limitation sentence saying that the “hierarchical bootstrap samples
datasets and then paired subject clusters” describes only the separate
dataset-superpopulation sensitivity. Those values are stored in the
`descriptive_dataset_superpopulation_interval95_*` columns; they are not the
intervals displayed in the sealed Markdown table and target a different
estimand.

## Deterministic seeds

`20260729` is the frozen **base** bootstrap seed in the plan and analysis
contract. The analyzer derives seeds deterministically in frozen architecture
order, using a zero-based comparison index:

- per-dataset participant interval: base + comparison index × 1,000 + dataset
  ordinal, where the ordinal is 1 through 5 in frozen dataset order;
- fixed-suite interval: base + comparison index × 10,000; and
- dataset-superpopulation sensitivity: base + comparison index × 10,000 + 500.

Accordingly, the selected leader's row in
`analysis/tcformer_overall_context.csv` records fixed-suite seed `20530729` and
dataset-superpopulation seed `20531229`. Both use 100,000 resamples. The
published interval remains `[-0.4272597383, +1.8992515953]` percentage points
before presentation rounding; only the shorthand description of its seed and
resampling level required clarification.
