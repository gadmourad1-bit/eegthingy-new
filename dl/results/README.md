# Results bundle

This is a protocol-separated release of aggregate and pseudonymous derived
results. The sealed analysis includes job-level metrics and subject/subject-
seed aggregate rows keyed by pseudonymous dataset subject indices. It contains
no EEG datasets, raw labels, trial-level predictions, caches, claims,
checkpoints, worker logs, direct identifiers, or clinical/demographic
participant metadata. Because the subject-level rows remain participant-
derived, institutional data-governance review is required before any public
disclosure of this bundle.

## Start here

- `protocol_summary.md`/`.csv` give the completion state and comparison
  boundary of every included or pending track.
- `common_grid_v6/SUMMARY.md` gives the top common-grid rankings and five
  per-dataset winners.
- `common_grid_v6/all_models_balanced_accuracy.csv`/`.md` are the complete
  43-row primary table. The matching standard-accuracy table is
  `common_grid_v6/all_models_accuracy.csv`/`.md`.
- `common_grid_v6/analysis/` is the exact 17-file sealed analysis publication;
  the wide tables are presentation-only pivots outside that sealed directory.
- `common_grid_v6/SEALED_OUTPUT_NOTES.md` clarifies the fixed-suite bootstrap
  wording and deterministic base/per-comparison seeds without altering the
  sealed publication.
- `gauge_gate1/` and `cardinal_fbms_transfer/` are separate protocol evidence
  and must not be inserted into the common-grid ranking.
- `legacy/README.md` records why legacy HemiQ/GeoAdapt JSONs were not imported.

## Integrity

`IMPORT_MANIFEST.csv` records a generic source-authority label, a stable
nonidentifying logical source locator, and the source-side SHA-256 for all 27
imported files. It intentionally contains no machine hostname, account name,
or absolute source path. Every hash was compared with the corresponding local
copy. `SHA256SUMS` covers every regular file below `results/` except itself.
Verify the full bundle from the project root with:

```bash
python results/verify_bundle.py
```

Regenerate the presentation-only common-grid views with:

```bash
python results/build_views.py
```

The sealed common-grid manifest and imported files are never rewritten by the
view builder.

## Interpretation boundary

Balanced accuracy is the prespecified primary metric. The 96,320-job common
grid is complete and exactly audited, but all five cohorts are opened
development evidence. The observed winner is descriptive and outcome-selected
from 43 configurations. It is not independent confirmation, a clinical
validation, or sufficient support for a global state-of-the-art claim.
