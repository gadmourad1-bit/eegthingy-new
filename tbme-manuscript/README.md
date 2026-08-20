# TBME manuscript working package

This directory is an internal, reproducible working package for a TBME regular
paper. It is not submission-ready. The manuscript, cover letter, figures,
references, disclosures, and checklist require independent author review and
approval before anything is uploaded.

The current regular-paper requirements were checked on 2026-08-20 against the
official Information for Authors, DOI
[10.1109/TBME.2026.3699307](https://doi.org/10.1109/TBME.2026.3699307).
Recheck the live submission portal immediately before submission because
editorial rules and charges can change.

## Contents

- `main.tex`: manuscript source. The paper covers the full platform
  (acquisition + cue paradigm + online pipeline), the deployed classical
  decoders (deployment-protocol evaluation, Table II), and the 43-model
  common-grid benchmark (Table III and figures).
- `references.bib`: bibliography (26 entries; IEEE journal names carry the
  full "IEEE" prefix).
- `figures/`: five deterministic figures (`platform.pdf`, `workflow.pdf`,
  `architecture.pdf`, `efficiency_scatter.pdf`, `dataset_deltas.pdf`).
- `scripts/build_assets.py`: deterministic figure/table asset builder; also
  validates that the pinned Table III rows and calibration sentences in
  `main.tex` match the sealed statistics (edit both in lockstep or the build
  fails).
- `scripts/deployment_classical.py`: deployment-protocol classical benchmark
  (EA-FBCSP and Riemannian tangent decoders on the 8-participant formal local
  cohort; personalized 5-fold CV and leave-one-participant-out). Its frozen
  output is `generated/deployment_classical.csv`, cited by Table II. Rerun it
  from the repository root with the repository `.venv` to regenerate.
- `scripts/build.sh`: one-command PDF build and compliance check.
- `scripts/check_manuscript.py`: structural, length, size, page, biography,
  and TODO checks.
- `template/`: byte-preserved official 2025 template assets plus a documented
  Tectonic compatibility class that differs only by requesting the generated
  PNG logo instead of the unsupported EPS form. Vendor-provided names and
  bytes may contain legacy branding and are the only allowed exception to the
  project's namespace-cleanup rule.
- `output/TBME_Internal_Working_Draft.pdf`: generated working PDF.
- `SUBMISSION_CHECKLIST.md`: human decisions and institutional approvals that
  automation cannot supply.

## Build

The default build uses the repository's locked `.venv` Python when it is
available and the staged Tectonic 0.16.9 binary. The asset builder is bound to
the Python/NumPy/Matplotlib versions recorded in `generated/statistics.json`;
an ambient environment with different numerical or rendering versions must
fail closed rather than silently replace the checked assets. Override either
tool explicitly when needed:

```bash
scripts/build.sh
PYTHON_BIN=/absolute/python TECTONIC_BIN=/absolute/tectonic scripts/build.sh
```

The script first runs `scripts/build_assets.py`, compiles `main.tex` in a
temporary output directory, installs the PDF at the fixed path above, and then
runs the manuscript checker. It never writes generated LaTeX auxiliaries next
to `main.tex`. It also exports a fixed `SOURCE_DATE_EPOCH` and UTC time zone so
the final PDF metadata and bytes are reproducible when the locked inputs,
Tectonic build, fonts, and environment are unchanged.

Run the checker independently with:

```bash
python3 scripts/check_manuscript.py
python3 scripts/check_manuscript.py --skip-pdf
python3 scripts/check_manuscript.py --fail-on-todo
```

The normal checker reports TODOs as warnings so an internal draft can build.
The final pre-submission invocation must use `--fail-on-todo` and return zero.

## Authorship and AI boundary

AI-assisted prose in this directory is an internal scaffold only. Every claim,
number, citation, limitation, ethics statement, and disclosure must be checked
against primary evidence. The named authors must independently rewrite and
approve the prose in their own scholarly voice. If any generated content
remains within the scope of the publishing policy, the submitted manuscript
must disclose the system, affected sections, and level of use in the
Acknowledgment section.

Automation cannot decide authorship, authorize participant-data disclosure,
infer ethics approval, select a license, or provide author consent. Those items
remain blocking TODOs in the submission checklist.
