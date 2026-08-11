# Third-party notices and release audit status

This file records the third-party material known to be in the clean research
project. It is an engineering notice, not legal advice. The in-house project
license is unresolved; see `LICENSE_DECISION_REQUIRED.md`.

## Vendored source

### TCFormer

- Upstream repository: `https://github.com/Altaheri/TCFormer`
- Pinned upstream commit:
  `74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`
- Upstream license: MIT
- Upstream attribution: `Copyright (c) 2025 Hamdi Altaheri`
- Local provenance: `third_party/TCFormer/SOURCE.json`
- Preserved license: `third_party/TCFormer/LICENSE`

The bundled source is a compact, byte-pinned runtime snapshot, not an
unmodified checkout of the entire upstream repository. Its adapter and use
under the shared common recipe do not make TCFormer an in-house architecture
or an exact reproduction of the authors' training recipe.

## Direct Python dependencies

The reviewed Linux x86-64 lock declares the following direct runtime
distributions. License labels below reproduce the installed metadata observed
in the audited lab environment; the final distributor must preserve the exact
license and notice files shipped by every resolved distribution and its
bundled components.

| Distribution | Version | Observed top-level license signal |
|---|---:|---|
| Braindecode | 1.6.1 | BSD-3-Clause; wheel included `LICENSE.txt` and `NOTICE.txt` |
| einops | 0.8.2 | MIT |
| matplotlib | 3.10.9 | matplotlib license agreement |
| MNE | 1.12.1 | BSD-3-Clause |
| MOABB | 1.5.0 | BSD-3-Clause |
| NumPy | 2.4.4 | Composite metadata including BSD-3-Clause, 0BSD, MIT, Zlib, and CC0-1.0 |
| pandas | 3.0.3 | BSD 3-Clause |
| pyRiemann | 0.11 | BSD 3-Clause |
| scikit-learn | 1.8.0 | BSD-3-Clause |
| SciPy | 1.18.0 | SciPy license |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| PyTorch | 2.6.0+cu124 | BSD-3-Clause; distribution included `LICENSE` and `NOTICE` |

The test group declares pytest 9.0.2 (MIT metadata). The documentation group
declares pdfplumber 0.11.10, pypdf 6.14.2, and ReportLab 5.0.0. Their exact
installed license inventories must still be captured from the final
`.venv-docs`; this file intentionally does not guess them.

Transitive packages and binary wheels may include separately licensed code.
In particular, a top-level dependency label is not a complete notice for
NumPy, PyTorch, Braindecode, CUDA user-space wheels, or their bundled
components. `uv.lock` is the dependency identity, but it is not a substitute
for reproducing required notices.

## External model implementations and papers

EEGNet, ShallowFBCSPNet, Deep4Net, EEG-Conformer, ATCNet, FBCNet, EEG-TCNet,
FBMSNet, CTNet, and EEGSym are external architectures. Except for the pinned
TCFormer snapshot described above, they are instantiated through reviewed
dependencies or local compatibility wrappers. A compact/wide/pooling wrapper
or a shared training recipe does not transfer architecture ownership. The
manuscript must cite the original method paper for every external model it
reports, as well as the software distribution that supplied the
implementation.

CardinalFBC is an in-house derivative of FBCNet; CardinalFBMS is an in-house
derivative of FBMSNet; CardinalMixedTemporal reuses FBMSNet components. Those
relationships require prominent upstream citations and license review.

## Data and publication rights are separate

Nothing in this notice authorizes redistribution of EEG, transformed caches,
participant-level predictions, embeddings, checkpoints, paper figures, or
paper text. See `docs/DATA_ACCESS.md` and `docs/ETHICS_AND_PRIVACY.md`.

## Remaining release work

Before public distribution:

1. choose the in-house source license and identify its copyright holders;
2. enumerate every distribution resolved by the final `uv.lock`, including
   transitive and documentation dependencies;
3. archive and hash the exact license/notice files from those distributions;
4. preserve the TCFormer license, attribution, commit, source manifest, and
   method citation;
5. add the full paper bibliography for external architectures and datasets;
6. check compatibility of derivatives and vendored material with the chosen
   project license; and
7. obtain institutional or qualified legal review when required.

