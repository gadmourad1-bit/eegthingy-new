# Third-party software license audit

## Status and scope

This is a preliminary engineering inventory for the clean benchmark release.
It is not legal advice and is not the final `THIRD_PARTY_NOTICES.md`.

The project owner has not selected a license for the in-house source. That
choice is a release blocker and must not be inferred from a manuscript submission,
a dependency license, or a dataset license.

This audit covers the proposed direct runtime dependencies observed in the
isolated lab UV environment and the compact pinned TCFormer source snapshot.
The final notice must additionally enumerate every distribution in the
reviewed `uv.lock`, preserve all required license/notice files, and be checked
against the exact clean environments rather than copied from this table.

## Observed direct runtime distributions

The metadata below was read without installing anything from
`/home/hanafy/scratchpad/eeg_novel_20260729/.venv-uv` on 2026-07-30.

| Distribution | Exact version | Installed metadata signal | Distributed license/notice files |
|---|---|---|---|
| Braindecode | 1.6.1 | BSD-3-Clause | `LICENSE.txt`, `NOTICE.txt` |
| einops | 0.8.2 | MIT | `LICENSE` |
| matplotlib | 3.10.9 | matplotlib license agreement | metadata did not enumerate a `License-File` |
| MNE | 1.12.1 | SPDX `BSD-3-Clause` | `LICENSE.txt` |
| MOABB | 1.5.0 | BSD-3-Clause | `LICENSE` |
| NumPy | 2.4.4 | SPDX composite: BSD-3-Clause, 0BSD, MIT, Zlib, and CC0-1.0 | top-level `LICENSE.txt` plus bundled component license files |
| pandas | 3.0.3 | BSD 3-Clause | metadata did not enumerate a `License-File` |
| pyRiemann | 0.11 | BSD 3-clause | `LICENSE` |
| scikit-learn | 1.8.0 | SPDX `BSD-3-Clause` | `COPYING` |
| SciPy | 1.18.0 | SciPy license text in metadata | metadata did not enumerate a `License-File` |
| threadpoolctl | 3.6.0 | BSD-3-Clause | `LICENSE` |
| PyTorch | 2.6.0+cu124 | BSD-3-Clause | `LICENSE`, `NOTICE` |

The test environment additionally proposes pytest 9.0.2, whose installed
metadata reports SPDX `MIT` and distributes `LICENSE`.

The PDF documentation group (`reportlab`, `pypdf`, and `pdfplumber`) was not
installed in the training environment at the time of this audit. Its licenses
must be captured from the final `.venv-docs`; do not guess them from package
names or older versions.

An installed wheel can bundle components under additional licenses even when
the top-level project uses a permissive license. NumPy explicitly reports a
composite expression and many component files. PyTorch and Braindecode ship
notices in addition to licenses. The release generator must copy or reproduce
the required texts from the exact installed distributions and retain their
attribution, rather than reducing this table to a single blanket statement.

## Pinned TCFormer source

The release strategy is a compact, byte-pinned runtime snapshot from
`https://github.com/Altaheri/TCFormer` at upstream commit
`74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`.

The upstream snapshot declares the MIT License and attributes:

```text
Copyright (c) 2025 Hamdi Altaheri
```

Exact snapshot identities:

| Path | SHA-256 |
|---|---|
| `SOURCE.json` | `06e04b81c05930721fd049b12bbff61e277438db2a9f94d9552864c3af4173ea` |
| `LICENSE` | `daea0e9f8596568cf5aae39b69f45ffc9f1b00d600d5315c09e90668cd99ee97` |
| `models/tcformer.py` | `755cba76838ad325ffd35a75c2e555235f1541a4bb409634105f10f3537acd66` |
| `models/modules.py` | `cb24a800c47cf3e417864b9928ba8da61017b1e928ae53706879c6af8a910c14` |
| `models/channel_group_attention.py` | `5623b68c9e1524faad00a8450a8f2fb909d3f39c7d4d20bef7ae769b8de8aabb` |
| `utils/weight_initialization.py` | `e92148c28b1ae39dc7c7a0d52704594545fb09cd9eaa6b1b819c4a3836ad01e3` |

The compact snapshot must preserve the upstream `LICENSE` verbatim and the
paper/repository citation. The local adapter and shared training recipe do not
turn a common-track result into an exact author reproduction.

## Distinct data and paper rights

Software licenses do not authorize redistribution of EEG recordings,
harmonized caches, participant-level predictions, or paper text. Dataset
rights and privacy boundaries are tracked separately in
`docs/DATASETS_AND_LICENSES.md`. Paper quotation and figure reuse require their
own review.

## Release-blocking actions

Before public promotion:

1. choose and record the in-house project license and copyright holders;
2. generate the final Linux x86-64 `uv.lock`;
3. enumerate all runtime, test, and documentation distributions from their
   separate clean UV environments;
4. hash and archive every installed license/notice file used to generate the
   notice;
5. preserve TCFormer's exact MIT text, attribution, repository, commit, and
   citation;
6. verify Braindecode/PyTorch notices and bundled component licenses;
7. have the institution or qualified counsel review the proposed
   `THIRD_PARTY_NOTICES.md`; and
8. fail release verification if a locked distribution is missing from the
   notice inventory.
