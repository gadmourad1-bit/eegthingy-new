# RA-L manuscript — build & submission notes

## Files
- `main.tex` — IEEEtran (journal), double-anonymous draft.
- `refs.bib` — 38 references; all external DOIs verified against Crossref.
- `figures/` — `system`, `method_recenter`, `results_2x2`, `latency`, `trajectory`, `center_drift` (PNG, 200 dpi). Regenerated from `simulation/reports/` and `recordings/`.

## Build
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or:  latexmk -pdf main.tex
```
(No LaTeX on this machine; source is lint-clean — cites/labels/figures/environments all resolve.)
Docker one-liner if you don't have TeX locally:
```bash
docker run --rm -v "$PWD":/w -w /w texlive/texlive latexmk -pdf main.tex
```

## MUST DO before submission (`\todo{...}` markers in main.tex)
1. **IRB protocol number** (title footnote + Sec. Methodology).
2. **Subject demographics**: N, age/sex, sessions per subject, prior BCI experience.
3. **Subject-count reconciliation**: dataset shows 8 usable offline (S1,3,4,5,6,7,8,10; S9 excluded; S10 uses training files 5–8); 4 users in the loop (reported as U1–U4 = S3,S5,S6,S10). Confirm the "10 subjects" figure.

## Anonymization (double-anonymous review)
- No author block (already `Anonymous Author(s)`).
- **Self-citations** `ownRobustMI2025`, `ownDualValidation2025` are the authors' own — the text already refers to them in the third person ("prior work in our line"). For submission, either keep as-is (third person) or replace with "[Anonymous]" per the RA-L policy you follow. Do **not** reveal the lab/institution.
- Remove any identifying strings from figures/repo URLs.

## Known gaps a reviewer will probe (see Stage-4 audit)
- Simulation-only (kinematic, minimal autonomy) — physical/Gazebo run is the top upgrade.
- N=4 in-loop, one trial/cell — no inferential statistics yet.
- No in-loop ablation of recentering, and no keyboard/oracle baseline.
- Offline decoder-accuracy table not yet included (loop results only).
