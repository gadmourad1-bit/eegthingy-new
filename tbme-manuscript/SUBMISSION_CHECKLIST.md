# TBME regular-paper submission checklist

Status: **INTERNAL WORKING DRAFT — NOT READY FOR SUBMISSION**

Requirements were checked on 2026-08-20 against the official Information for
Authors, DOI
[10.1109/TBME.2026.3699307](https://doi.org/10.1109/TBME.2026.3699307).
The corresponding author must recheck the live portal immediately before
submission and record the date below.

- [ ] TODO: live requirements rechecked on `YYYY-MM-DD` by `NAME`.
- [ ] TODO: article type confirmed as **Regular Paper**.

## Automated format gates

- [ ] Manuscript uses the official double-column, single-spaced journal
  template; figures and tables are embedded and cited in order.
- [ ] Structured abstract is under 250 words and contains, in order:
  **Objective, Methods, Results, Conclusion, Significance**.
- [ ] Index terms follow the abstract and are in alphabetical order.
- [ ] Main sections appear in order: **Introduction, Methods, Results,
  Discussion, Conclusion**; references form a separate final section.
- [ ] Conclusion is under 300 words.
- [ ] Review PDF is under 10,000,000 bytes.
- [ ] Regular-paper target is at most 8 pages. Pages 9–12 require an explicit
  author funding decision because overlength charges apply. More than 12 pages
  requires prior Editor-in-Chief permission and fails the local checker.
- [ ] No author biographies or author photographs are included.
- [ ] TODO: prepare the graphical abstract required at submission (the portal
  requests one; `figures/platform.pdf` is a candidate basis but must meet the
  portal's format/size rules).
- [ ] `scripts/build.sh` succeeds.
- [ ] `python3 scripts/check_manuscript.py --fail-on-todo` reports zero TODOs
  and exits successfully.

## Cover letter and scope

- [ ] Cover letter is under 250 words after deleting the internal scaffold
  banner and all bracketed instructions.
- [ ] Cover letter states the verified innovation and significance to
  biomedical research without claiming state of the art, clinical efficacy,
  or independent confirmation beyond the evidence.
- [ ] TODO: all authors confirm the manuscript is original, not previously
  published in substantially the same form, and not under concurrent
  consideration.
- [ ] TODO: disclose every prior conference paper, abstract, preprint, thesis,
  public draft, and overlapping or concurrently prepared manuscript; explain
  the added contribution and provide identifiers where applicable.

## Authors and institutional approvals

- [ ] TODO: finalize author list, order, contribution eligibility, full names,
  affiliations, institutional emails, and corresponding author.
- [ ] TODO: add and verify an ORCID for every author.
- [ ] TODO: every author has reviewed the complete manuscript, references,
  figures, supplements, cover letter, and disclosures and has explicitly
  consented to submission.
- [ ] TODO: complete the contributor-role statement using the journal's
  accepted taxonomy.
- [ ] TODO: record funding sources and grant numbers, or a verified statement
  that no specific funding supported the work.
- [ ] TODO: record conflicts of interest for every author, or a verified
  no-conflict statement.

## Local Exp4 human-participant boundary

- [ ] TODO: identify the responsible institution and provide the exact Local
  Exp4 ethics/IRB committee name, protocol number, approval date, and approval
  or exemption status. Do not infer approval from possession of the data.
- [ ] TODO: verify and state the informed-consent process, including consent
  for research use and any intended release of derived participant-level data.
- [ ] TODO: verify participant demographics, inclusion/exclusion criteria,
  recruitment, sample attrition, and missing-data handling. Do not invent
  unavailable demographics.
- [ ] TODO: document acquisition hardware, electrode montage and reference,
  sampling chain, filtering, event timing, cue/controller software, recording
  environment, session structure, and operator procedure.
- [ ] TODO: institutional privacy/governance review completed for pseudonymous
  participant keys and derived metrics before any public upload.

## Scientific and statistical verification

- [ ] Every manuscript number is programmatically traced to the frozen result
  table or named source artifact; displayed rounding is distinguished from
  full-precision comparisons.
- [ ] The five training seeds (`7, 17, 27, 37, 47`), fold structure,
  participant-aware aggregation, and equal-dataset overall statistic are
  described correctly.
- [ ] The 43-configuration, five-dataset, 96,320-job common grid is not mixed
  with local-only, author-recipe, transfer, geometric, or confirmation tracks.
- [ ] Rankings are described as fixed-suite descriptive results. The reported
  uncertainty interval that includes zero is not presented as significant
  superiority.
- [ ] Standard errors accompany every reported estimated quality metric, and
  their participant-stratified bootstrap computation is reproduced from the
  frozen participant rows.
- [ ] The post-hoc all-42 sign-flip analysis is labelled exploratory; its raw
  and multiplicity-adjusted values are independently verified, and it is not
  substituted for a preregistered confirmation test.
- [ ] Claims explicitly exclude global state of the art, clinical safety or
  benefit, and demonstrated performance in disabled or paralyzed populations.
- [ ] Public-comparator implementations, versions, deviations from original
  papers, and source licenses are checked.
- [ ] References are verified against primary sources, cited numerically in
  order, and contain no fabricated or placeholder entry.

## Code, data, and licensing

- [ ] TODO: project owner selects and approves a first-party code license.
- [ ] TODO: confirm third-party notices and licenses cover every distributed
  dependency, vendored source, template asset, figure, and dataset-derived
  artifact.
- [ ] TODO: provide a code-availability statement with immutable release
  identifier and repository location, or state the justified restriction.
- [ ] TODO: provide separate availability statements for public datasets,
  Local Exp4 raw data, transformed caches, checkpoints, predictions, and
  aggregate results; do not imply redistribution rights that were not granted.
- [ ] TODO: decide whether participant-level derived tables may be released
  after ethics/privacy review.

## AI-assisted drafting

- [ ] Treat every AI-assisted sentence in this package as an internal scaffold.
  Authors independently rewrite it in their own scholarly voice and verify all
  facts, calculations, citations, interpretations, and limitations.
- [ ] TODO: maintain an internal record of systems used, dates, affected
  sections, prompts/data boundaries, and human verification.
- [ ] TODO: determine the required submission disclosure under the policy in
  effect on the submission date. If generated content remains in scope, add an
  Acknowledgment that identifies the system, affected sections, and level of
  use. Editing-only assistance must still be checked carefully.
- [ ] No confidential manuscript, private participant data, credentials, or
  peer-review material was provided to a public AI service.

## Final upload

- [ ] Internal banners, TODOs, placeholder names, comments, and private paths
  removed.
- [ ] Title, abstract, cover letter, author metadata, submission-system fields,
  PDF, and source archive agree exactly.
- [ ] PDF visually inspected page by page for clipping, unreadable figures,
  broken equations, missing references, and accidental disclosure.
- [ ] All source files, bibliography files, and production-quality figure
  assets needed to compile are included; temporary build files are excluded.
- [ ] Final authorship, ethics, originality, funding, conflict, AI-use,
  licensing, and data/code statements approved by all authors.
