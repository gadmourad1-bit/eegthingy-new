# Sanitized native-FBMS v1 source closure

`source/eeg_mi_native_v1/` contains the 11 sanitized source files used by the
native-FBMS compatibility reader. Every file SHA-256 is checked against
`eeg_mi.legacy_native_identity.PINNED_SOURCE_MANIFEST`.

This directory is historical evidence, not an active Python package. Its
parent is absent from the normal import path. `eeg_mi.legacy_native_identity`
reads and verifies the archive only when validating sanitized source identity.
Earlier external checkpoint metadata is retired and fails closed; active
native-FBMS v1 writers cannot attribute this closure to a different execution.

The active source files are independently inventoried by release provenance
and are not asserted to be byte-equivalent to this archive.
