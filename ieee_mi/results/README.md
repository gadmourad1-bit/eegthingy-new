# Development result artifacts

Large JSON artifacts copied back from the isolated GPU snapshots are ignored by
Git but retained here for local audit. They contain exploratory development
predictions and histories, not confirmation evidence. The authoritative
experiment status and cohort firewall are documented in
`../EXPERIMENT_LEDGER.md`.

The completed CardinalFBMS all-fold, five-target-seed development result is
summarized in `NATIVE_FBMS_FULL_GRID_SUMMARY.md`. Its large ignored gate JSON,
post-completion score-blind preflight, and independent raw-OOF verification
report are retained in this directory and bound by the hashes in that summary.
The independent verifier source is
`../verification/verify_native_fbms_full_grid.py`; it is explicitly
post-outcome verification code and is not part of the frozen decision gate.
