# Sanitized common-grid v6 replay source

`source/eeg_mi_v6/` is the frozen, sanitized source authority used to replay
the common-grid v6 result bundle. It is not the active project package and its
parent is not on the normal Python import path.

The logical source keys use `eeg_mi/...`, while the distinct physical directory
name prevents this archive from shadowing the active package. The reissued
plan binds these sanitized source bytes and the two dependency manifests under
`source/`; earlier external artifacts are read-only and are not accepted as
current replay records.

The live implementation is `src/benchmark/`, the UV distribution and Python
namespace are both `benchmark`, and supported commands are exposed through
`scripts/reproduce.sh`. The bounded reviewer adapter loads this replay archive
under a private module name through an isolated, explicit path.

The root `SOURCE_PROVENANCE.json` inventories the distributed active and replay
source. `reports/` contains the three sanitized report reissues and their
checksum ledger. Their scientific tables are unchanged, but their document
bytes and identity metadata are intentionally new.
