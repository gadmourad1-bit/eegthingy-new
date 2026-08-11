# CardinalFBMS lab preservation and raw-result reverification

Status: **PASS**. This is an integrity and reproducibility check of an existing
development result. It is not a new experiment, an independent cohort, or a
confirmation result.

## Preservation event

On 2026-07-29 at 21:32:37-04:00, the completed CardinalFBMS full-grid artifacts
were copied without modifying the source from the old GPU workstation into:

`/home/hanafy/scratchpad/eeg_novel_20260729/imported_prior_fbms_v2`

The preserved payload contains the Cho2017 result root, the PhysioNet MI result
root, and the frozen full-grid gate JSON. Source and destination were hashed
independently using the lexicographically sorted relative path and SHA-256 of
every regular file.

| Integrity item | Source | Lab copy |
|---|---:|---:|
| Regular files | 17,351 | 17,351 |
| Total regular-file bytes | 667,239,624 | 667,239,624 |
| Canonical tree SHA-256 | `958801527c039c734b07b2b9a1593df0b0e64c206f9485335c199eb5508f5840` | `958801527c039c734b07b2b9a1593df0b0e64c206f9485335c199eb5508f5840` |
| Frozen gate SHA-256 | `eab9f6ee1b548000eed823025330aa13c20616b8ed266cf5dd6b521a7ef80496` | `eab9f6ee1b548000eed823025330aa13c20616b8ed266cf5dd6b521a7ef80496` |

## Independent raw-result verification

The preserved standalone verifier,
`ieee_mi/verification/verify_native_fbms_full_grid.py`, has SHA-256
`918ee23a785b449688f39929f6fdde7778f81652622ed25bec8b5c1422dfbe18`.
It was run from the project scratch source with Python 3.12.13 inside the
UV-created virtual environment on Linux 6.8.0-101-generic x86_64. UV itself was
version 0.12.0. No dependency was installed or changed for this verification.

The verifier reopened and rehashed every provenance and prediction file,
reconstructed the exact out-of-fold rows, recomputed all subject/seed balanced
accuracies, reproduced the two 200,000-replicate paired-subject bootstraps, and
re-evaluated the frozen decision checks.

| Verification item | Result |
|---|---|
| Status | PASS |
| Validated prediction records | 8,675 |
| Embedded manifest SHA-256 | `5e9a2bd93828603e2cafa671158074c7f707047112d867f8504ae86393f7fbbe` |
| Generated report SHA-256 | `7c1302f17f22a4e895c7e42257f5f0a8ab1badf0d59b93ea6b4ee314489f2562` |
| Frozen gate checks | 6 of 6 passed |

The independently recomputed equal-dataset macro balanced accuracies were
66.079% for pretrained CardinalFBMS, 64.867% for the checkpoint-matched indexed
FBMSNet plus fixed spherical-spline control, 59.004% for canonical-seeded
scratch CardinalFBMS, 60.171% for native-projected scratch CardinalFBMS, and
61.689% for scratch native indexed FBMSNet.

The generated report is byte-identical to the previously preserved
`ieee_mi/results/native_fbms_full_grid_independent_raw_verification.json`.
This closes the artifact-preservation check, but it does not change the
development-only interpretation of the result.
