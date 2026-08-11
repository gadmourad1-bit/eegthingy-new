# Data access, provenance, and redistribution boundary

## No EEG is distributed with this project

This repository contains code, configuration, documentation, and a sealed
development analysis. That analysis includes pseudonymous job-level,
subject-seed-level, and subject-level derived metric tables. It does not
include raw EEG, transformed/harmonized caches, trial-level predictions,
direct participant identifiers, demographic/clinical metadata, features,
embeddings, checkpoints, or private acquisition manifests.

Pseudonymous participant keys and metrics are still participant-derived data.
Their inclusion in an internal release candidate is not authorization for
public disclosure. In particular, the Local Exp4 rows in `job_metrics.csv`,
`subject_seed_metrics.csv`, and `subject_metrics.csv` require a written
governance decision before this bundle may be published.

Reproduction therefore requires the researcher to obtain each public dataset
from its authoritative record, comply with its terms, create a private cache,
and retain a provenance manifest. Local Exp4 is not public and may be used only
under its controlling institutional authorization.

Filtering, resampling, rereferencing, epoching, pseudonymizing, or converting
files does not erase the original dataset's license, confidentiality, or
human-subject obligations.

## Cohorts used by the verified common grid

All 132 participant caches in the completed common grid were already opened
during model development. The resulting scores are development evidence, not
independent confirmation.

| Dataset | Common-grid cohort and task | Frozen evaluation boundary | Access/redistribution status |
|---|---|---|---|
| Local Exp4 | 8 pseudonymous participants; binary left/right imagery | Four chronological recordings: first two train, third validation, fourth prediction-only test | Private. No supplied document establishes the governing IRB/consent/data-use decision. Do not distribute any participant-derived artifact without written approval. |
| BNCI2014-001 / BCI Competition IV 2a | 9 participants; four classes: left hand, right hand, feet, tongue | Official evaluation session held out; final source-training run within the training session is validation | Official BNCI record reports CC BY-ND 4.0. Do not distribute transformed caches without written permission or a documented legal determination. |
| BNCI2014-004 / BCI Competition IV 2b | 9 participants; binary left/right; three supplied bipolar derivations | Official chronological/session holdout | Official BNCI record reports CC BY-ND 4.0. The three signals are bipolar derivations, not interchangeable with C3/Cz/C4 point electrodes. Do not distribute transformed caches. |
| Cho2017 | 52 participants; binary left/right | Five rotating, class-stratified acquisition-order block folds | Publicly accessible through the dataset DOI, but the exact license attached to the acquired bytes must be captured. Current records are not sufficiently uniform to authorize redistribution from this project. |
| PhysioNet EEG Motor Movement/Imagery v1.0.0 | Participants S1–S54; imagined unilateral-fist runs 4, 8, and 12; binary left/right | Three rotating complete-run folds | Official files are labelled ODC-By 1.0. This project nevertheless does not bundle originals or transformed caches. S55–S109 remain sealed confirmation subjects. |

The common public preprocessing profile is 4–40 Hz, 0.5–3.0 s as a half-open
interval, 128 Hz, 320 samples, epoch demeaning, and symmetric common-average
reference except for the supplied BNCI2014-004 bipolar signals. The normal
profile uses a declared 21-channel montage; BNCI2014-004 retains three bipolar
derivations. Local Exp4 uses its separate, task-valid 0.0–2.0 s half-open
window, 128 Hz/256 samples after resampling from 125 Hz, and common-average
reference over 15 point-EEG channels.

Source-only channel scaling is fitted inside each split. The selection scaler
uses training rows only; the final-refit scaler uses train plus validation;
test rows never determine scaling.

## Authoritative public records and required citations

- **BNCI2014-001:** obtain from the official [BNCI data-set
  registry](https://bnci-horizon-2020.eu/database/data-sets). Cite the registry
  and Tangermann et al., “Review of the BCI Competition IV,” *Frontiers in
  Neuroscience* (2012), DOI
  [10.3389/fnins.2012.00055](https://doi.org/10.3389/fnins.2012.00055).
- **BNCI2014-004:** obtain from the same official BNCI registry. Cite the
  registry and Leeb et al., “A brain-computer interface based on imagined hand
  movements,” *IEEE TNSRE* (2007), DOI
  [10.1109/TNSRE.2007.906956](https://doi.org/10.1109/TNSRE.2007.906956).
- **Cho2017:** use the supporting-data record
  [GigaDB 100295](https://doi.org/10.5524/100295) and cite Cho et al., “EEG
  datasets for motor imagery brain-computer interface,” *GigaScience* (2017),
  DOI [10.1093/gigascience/gix034](https://doi.org/10.1093/gigascience/gix034).
  Record the exact mirror, version, terms, and hashes actually used.
- **PhysioNet:** use [EEG Motor Movement/Imagery Database
  v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), DOI
  [10.13026/C28G6P](https://doi.org/10.13026/C28G6P), and preserve the official
  page's requested citations and ODC-By notice.

MOABB is an acquisition interface, not the licensor. Its metadata does not
replace the exact terms served with the acquired files.

## Sealed cohorts not used in the common grid

The following are not missing common-grid results. They were deliberately
reserved from the opened development suite:

- PhysioNet S55–S109;
- all 54 Lee2019/OpenBMI MI participants;
- all 10 Weibo2014 participants; and
- all 4 Zhou2016 participants.

Do not download, inspect, cache, tune on, or score these cohorts merely to
complete a table. Opening them requires a prespecified confirmation protocol,
a frozen model/configuration, rights review, and authorized unsealing. Once
opened, they cannot continue to be described as untouched confirmation data.

## Local Exp4 release blocker

The repository does not establish the institution, protocol title or number,
approval/exemption status, consent language, participant population,
recruitment procedure, data controller, or permitted artifact classes for
Local Exp4. No such fact may be inferred from the project's assistive-BCI
motivation.

Before any external sharing or manuscript release, the principal
investigator, data steward, and institutional privacy/IRB authority must
document whether the approved boundary permits:

- this secondary machine-learning analysis;
- publication of aggregate statistics;
- external collaboration and public code;
- per-participant or trial-level outputs;
- trained weights or embeddings;
- retention, withdrawal, correction, and deletion procedures; and
- commercial or assistive-device use.

Until then, keep every Local Exp4-derived artifact, including the bundled
pseudonymous metric rows and any weights, within the approved environment.
Even dataset-level aggregate values require the research team's publication
authorization. If participant-level release is not permitted, generate a new
privacy-minimized publication and new manifest/checksums from an authorized
pipeline; do not delete or edit rows inside the sealed analysis directory.

## Reproduction cache requirements

Use an external private cache root; never place it inside the Git project.
For every subject, preserve a machine-readable manifest containing:

1. canonical dataset identifier and version;
2. exact acquisition record/URL and access date;
3. acquisition library and locked version;
4. original filename, byte size, and SHA-256;
5. exact license or terms captured with those bytes;
6. preprocessing schema, ordered channels, events, coordinate convention,
   and split identity;
7. cache array and metadata digests emitted by the loader; and
8. an aggregate manifest digest referenced by the run plan.

Missing identity information is a provenance failure. Do not substitute a
nearby mirror or silently rebuild a cache after a plan is frozen.

## Secure handling minimum

- Store raw and derived EEG only on institutionally approved encrypted
  storage with least-privilege access.
- Keep participant-code linkage outside the analysis project and under a
  different access boundary.
- Never put credentials, tokens, private hostnames, or identifying paths in
  manifests intended for release.
- Treat EEG, predictions, embeddings, and checkpoints as potentially
  identifying; review re-identification and membership-inference risk.
- Record access, exports, incidents, withdrawals, retention, and destruction
  according to the controlling protocol.
- Release dataset-level or participant-level tables only after checking
  small-cell, outlier, linkage, and singling-out risks and obtaining the
  required approval.

The longer source audit in `docs/DATASETS_AND_LICENSES.md` records additional
dataset-level caveats. If it conflicts with newly captured authoritative
terms, stop release work, preserve both records, and obtain a documented
decision rather than choosing the more permissive interpretation.
