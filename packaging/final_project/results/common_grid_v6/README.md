# Common-grid v6 results

This directory contains the exact, audited common raw-trial benchmark and
presentation-only views derived from its sealed CSVs.

- `analysis/` is the untouched 17-file sealed publication. Its
  `manifest.json` hashes the other 16 files in that directory. The sealed
  `job_metrics.csv`, `subject_metrics.csv`, and `subject_seed_metrics.csv`
  contain participant-derived results keyed only by pseudonymous job IDs and
  dataset subject indices; they contain no raw EEG or trial predictions.
- `SEALED_OUTPUT_NOTES.md` records a precise reading of the sealed human-facing
  bootstrap wording and its base/per-comparison seeds. It sits outside
  `analysis/` and does not alter or supersede the sealed CSV/JSON authority.
- `plan.json` and `plan.sha256` are the frozen execution plan and its canonical
  plan digest. The raw bytes of `plan.json` are separately covered by the
  bundle checksum and import manifest.
- `preflight/` contains the 172/172 CUDA preflight report and receipt.
- `final_audit.json` records 96,320/96,320 complete jobs, no missing, extra,
  failed, partial, live/stale-claim, unsafe-path, or unexpected-root entries,
  and `exact_cartesian_complete: true`.
- `all_models_balanced_accuracy.csv`/`.md` contain all 43 models, all five
  dataset balanced accuracies, the equal-dataset overall value, and sealed
  primary rank.
- `all_models_accuracy.csv`/`.md` are the corresponding standard-accuracy
  views. Their accuracy rank is descriptive and is obtained only by sorting
  the sealed equal-dataset accuracy column.
- `SUMMARY.md`, `top_10_balanced_accuracy.csv`, and
  `per_dataset_balanced_accuracy_winners.csv` are compact views.

Run `python results/build_views.py` from the project root to regenerate only
the presentation views. That script pivots sealed values; it does not open
predictions or recompute metrics.

All scores are opened-development evidence. The observed leader is not an
independent confirmation, clinical result, or by itself a global SOTA claim.
Institutional data-governance review is required before publicly disclosing
the pseudonymous subject-level tables.
