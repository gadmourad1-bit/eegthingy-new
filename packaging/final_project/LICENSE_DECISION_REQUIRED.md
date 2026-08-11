# Project license decision required

No license has been selected for the in-house source in this project. Until
the copyright holders and responsible institution make that decision, the
absence of a license must be treated as **no public permission granted to
copy, modify, or redistribute the in-house code**. This file is not itself a
license.

An IEEE manuscript submission, a paper's publication terms, a dependency's
license, and TCFormer's MIT license do not license this project's original
source. Dataset terms likewise govern data, not the source code.

## Decision owners

Before public release, identify and obtain written approval from:

1. every person or institution that owns copyright in the in-house code;
2. the laboratory or university office responsible for software releases;
3. the principal investigator or data steward where local participant data
   influenced code, checkpoints, or released artifacts; and
4. qualified legal or technology-transfer reviewers when required.

Do not infer ownership from commit authors, repository access, employment,
funding, or the author list.

## Questions the decision must answer

- Is the intended release open source, source-available, internal-only, or
  governed by a separate research-use agreement?
- Which exact files are original works, derivatives, vendored third-party
  source, generated outputs, or documentation?
- Are patent disclosures, sponsor rights, export controls, or institutional
  commercialization rules implicated?
- May model weights be released? Weights can have separate privacy,
  contractual, and dataset-derived restrictions even when code is licensed.
- Must a notice, contributor agreement, or copyright header be added?
- Are commercial use, redistribution, and derivative works permitted?

If an OSI-approved license is chosen, use its unmodified canonical text unless
counsel directs otherwise. If custom restrictions are required, do not call
the result “open source.”

## Required release changes

After approval:

- add the complete license text as `LICENSE`;
- replace this file with a short pointer or remove it;
- add the approved SPDX identifier to `pyproject.toml` and the final
  `CITATION.cff` when applicable;
- record copyright holders without guessing;
- update `THIRD_PARTY_NOTICES.md` without changing third-party terms;
- verify that all distributed files are compatible with the selected terms;
  and
- create a new release manifest and checksums after those changes.

The compact TCFormer snapshot under `third_party/TCFormer` remains governed by
its own preserved MIT license and attribution. Other dependencies retain their
own licenses and notices. Selecting a project license never replaces those
obligations.

