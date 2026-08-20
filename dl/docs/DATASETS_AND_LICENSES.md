# Datasets, licenses, ethics, and release boundaries

**Audit date:** 2026-07-29  
**Status:** release-blocking data-governance inventory  
**Scope:** the exact development and sealed-confirmation cohorts registered in
[`src/benchmark/config.py`](../src/benchmark/config.py)

The retained token `eeg-mi-cache-v2` is a versioned cache-schema identifier,
not the active package or distribution name.

This document is a conservative engineering and research-governance record, not
legal advice. A public download page is evidence of access, not by itself
permission to republish human-participant EEG. If a term is not supported by
the cited repository record, it is marked unresolved rather than inferred.

## Non-negotiable distinctions

1. **Project code, third-party code, papers, and data are different works.**
   A code license does not license EEG files. An article's open-access license
   ordinarily licenses the article, not automatically its supporting data.
   A dataset license does not choose the license for this project's source
   code. The final project code license is not yet selected and is a release
   blocker.
2. **Access is not redistribution.** “Open access,” a working downloader, or
   unrestricted research use does not automatically authorize this repository
   to mirror raw files, transformed epochs, or participant-level derivatives.
3. **Scientific sealing is independent of legal access.** A public dataset can
   remain scientifically inaccessible to this study. A sealed subject or
   dataset must not be downloaded, inspected, cached, or scored until the
   prespecified unsealing gate is satisfied.
4. **Copyright permission does not erase privacy obligations.** CC0,
   CC BY-ND, and ODC-By each leave privacy, publicity, personality, contractual,
   or data-protection rights outside at least part of their grants. EEG and
   linked demographic/questionnaire data must be treated as potentially
   re-identifiable human-participant data.
5. **Default release rule: no dataset bytes.** Raw recordings, harmonized
   caches, extracted trials, features, covariances, and trial-level predictions
   stay in approved external storage. The repository may contain official
   source links, acquisition code, immutable manifests, and hashes. Bundling
   data requires a dataset-specific written approval that overrides this
   conservative default.

The official [CC BY-ND 4.0 deed](https://creativecommons.org/licenses/by-nd/4.0/)
prohibits distribution of transformed material and warns that privacy rights
may still constrain use. The official
[CC0 deed](https://creativecommons.org/publicdomain/zero/1.0/) likewise says
privacy and publicity rights are unaffected. The
[ODC-By 1.0 text](https://opendatacommons.org/licenses/by/1-0/) covers database
rights subject to attribution, while expressly warning that individual
contents and privacy/data-protection rights may require separate clearance.
All three sources were accessed 2026-07-29.

## Exact project tracks

“Opened” below means already used in model development; it is not a licensing
conclusion. “Sealed” is a scientific status enforced even when the repository
is public.

| Dataset key | Exact project task and subjects | Scientific status | Repository/license signal | Repository release decision |
|---|---|---|---|---|
| `local_exp4` | Binary left/right imagery; pseudonymous S1, S3, S4, S5, S6, S7, S8, S10; fixed chronological recording holdout | All 8 opened development/replication; no confirmation subjects | No controlling consent, IRB, data-use, ownership, or redistribution record has been supplied | **Blocked. No raw, derived, participant-level, or trained-model release pending lab approval** |
| `bnci2014_001` | BCI Competition IV 2a; all 9 participants; left hand, right hand, feet, tongue; official-session holdout | All 9 opened development only | Official BNCI record: CC BY-ND 4.0 | No raw bundle; no transformed cache distribution |
| `bnci2014_004` | BCI Competition IV 2b; all 9 participants; binary left/right; official-session holdout; supplied bipolar C3/Cz/C4 derivations | All 9 opened development only | Official BNCI record: CC BY-ND 4.0 | No raw bundle; no transformed cache distribution |
| `cho2017` | All 52 participants; binary left/right; rotating acquisition-order block holdout | All 52 opened development only | Original GigaDB DOI; publisher policy expects CC0, while current NEMAR/MOABB metadata says CC BY 4.0 | Exact acquired-byte license unresolved; no bundle |
| `physionet_mi` | Imagined unilateral-fist runs 4, 8, 12 only; binary left/right; rotating complete-run holdout | S1-S54 opened development; S55-S109 sealed confirmation | PhysioNet v1.0.0 files: ODC-By 1.0 | No raw bundle by project policy; preserve attribution/version/hash |
| `lee2019_mi` | OpenBMI MI only; all 54 participants; binary left/right; official-session holdout | Entire dataset sealed confirmation | GigaDB DOI; article is CC BY 4.0 and OpenBMI code is GPL 3.0, but those are not proof of the EEG-file license | Dataset-file license unresolved; no access and no bundle |
| `weibo2014` | All 10 participants; project uses only left/right from the upstream seven-class experiment; rotating-run holdout | Entire dataset sealed confirmation | PLOS paper points to Harvard Dataverse and says raw data are available without restriction; current MOABB metadata reports CC0 | Exact Dataverse record terms must be captured at unseal; no access and no bundle |
| `zhou2016` | All 4 participants; project uses only left/right from the upstream three-class experiment; session holdout | Entire dataset sealed confirmation | Original Figshare v2 is CC0; current MOABB BIDS metadata reports CC BY 4.0 | Acquisition-source/license divergence must be resolved at unseal; no access and no bundle |

The validated scratch environment used MOABB 1.5.0, as recorded in
[`docs/UV_REPRODUCIBILITY.md`](UV_REPRODUCIBILITY.md). The clean promoted
project must pin MOABB and every acquisition dependency in its own `uv.lock`;
the scratch environment is evidence, not the final lock.

## Shared preprocessing and data provenance

The common public-dataset cache contract is `eeg-mi-cache-v2`: 4-40 Hz,
0.5-3.0 seconds as a half-open interval relative to the declared dataset
interval, resampled to 128 Hz (320 samples), epoch demeaned, and symmetrically
common-average referenced except for the supplied bipolar BNCI2014-004
derivations. The common montage profile uses 21 channels, except that
BNCI2014-004 retains its 3 bipolar derivations and Zhou2016 uses its declared
8-channel subset. Native-montage tracks are separate and preserve the native
EEG montage under the constraints in
[`src/benchmark/config.py`](../src/benchmark/config.py).

Local Exp4 has a distinct frozen profile because its task ends earlier:
4-40 Hz, 0.0-2.0 seconds half-open, 125 Hz source resampled to 128 Hz
(256 samples), epoch demeaned, and common-average referenced over 15
point-EEG channels.

Filtering, rereferencing, resampling, epoching, channel selection, and cache
serialization are transformations. They do not make the resulting arrays
license-free or anonymous. In particular, redistributed BNCI transformed
caches must be treated as prohibited adapted material under CC BY-ND unless
the licensor gives written permission or qualified counsel documents a
different conclusion.

The repository does not contain a complete release cache-manifest digest for
these cohorts. Before promotion, the external immutable result root must
provide, for every subject actually used:

- canonical dataset identifier and repository version;
- exact acquisition URL/record and access date;
- acquisition library and locked version;
- original file name, byte size, and SHA-256;
- the license/terms captured from the exact record serving those bytes;
- cache schema, preprocessing profile, ordered channels, events, and split;
- raw/cache identity hashes already emitted by the runner; and
- a manifest SHA-256 referenced by the aggregate result, without copying data.

Missing acquisition or cache identity is a provenance blocker, not permission
to substitute a nearby mirror.

## Dataset-specific records

### Local Exp4

The project registers eight pseudonymous participants and uses all of them as
opened development/replication evidence. It does not reserve a local
confirmation cohort. The public record must not infer participant diagnosis,
disability status, recruitment population, consent scope, or institutional
approval from the project goal.

No repository document supplied to this audit establishes the controlling
human-subject or data-use boundary. The manuscript notes also list the IRB
protocol number and demographics as missing. Before any external release or
sharing, the researchers, principal investigator, IRB/privacy office, and data
steward must supply and approve:

- institution, protocol title and number, approval/exemption status, effective
  dates, amendments, and whether this secondary analysis is in scope;
- consent-form version and whether it permits secondary ML research,
  publication, external collaboration, public code, trained weights, raw or
  derived data sharing, and commercial use;
- the data controller/owner, sponsor terms, data-use agreement, and any
  institutional or cross-border restrictions;
- recruitment population and approved de-identification/pseudonymization
  procedure, including the re-identification risk of EEG and linked metadata;
- withdrawal, correction, retention, destruction, access-control, incident,
  and audit-log procedures;
- whether aggregate statistics, per-participant metrics, trial predictions,
  embeddings, and trained weights may leave the approved environment; and
- a named approver and dated written release decision for each artifact class.

Until that evidence exists, do not publish raw Exp4, harmonized caches,
participant metadata, per-participant or trial-level outputs, features,
embeddings, or weights trained with Exp4. Even aggregate results require the
research team's confirmation that publication is within the approved protocol.

Local sources: the frozen
[`dataset specification`](../src/benchmark/config.py) and
historical manuscript submission notes at `paper/README.md` in the broader
research tree, reviewed 2026-07-29. The manuscript source is not distributed
in this clean benchmark bundle.

### BNCI2014-001 / BCI Competition IV 2a

The project uses all nine participants and all four official classes
(left hand, right hand, feet, tongue), with the official evaluation session
held out from training. Every participant has already been opened, so this is
development evidence only.

The official BNCI database lists 9 participants, 22 EEG plus 3 EOG channels,
Graz University of Technology's Institute for Knowledge Discovery as licensor,
and CC BY-ND 4.0. BNCI also instructs users to cite associated publications.

Required citation: Tangermann et al., “Review of the BCI Competition IV,”
*Frontiers in Neuroscience* (2012),
[doi:10.3389/fnins.2012.00055](https://doi.org/10.3389/fnins.2012.00055).

Release decision:

- access and analysis under the official terms: supported;
- redistribution of unchanged originals: the license can permit sharing with
  attribution, but this project still will not bundle the files;
- redistribution of filtered, resampled, rereferenced, epoched, or otherwise
  transformed caches: blocked by the NoDerivatives condition unless written
  permission or documented legal review says otherwise; and
- citation/attribution: cite the official dataset entry, primary paper,
  licensor, and CC BY-ND 4.0.

Sources: [BNCI data-set registry](https://bnci-horizon-2020.eu/database/data-sets)
and [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/), accessed
2026-07-29.

### BNCI2014-004 / BCI Competition IV 2b

The project uses all nine participants for binary left/right imagery under the
official chronological session split. Its three EEG signals are supplied
bipolar derivations centered nominally around C3, Cz, and C4; they must not be
described as interchangeable with three point electrodes. Every participant
has already been opened, so this is development evidence only.

The official BNCI database lists 9 participants, 3 EEG plus 3 EOG signals, the
same Graz licensor, and CC BY-ND 4.0.

Required citation: Leeb et al., “A brain-computer interface based on imagined
hand movements,” *Transactions on Neural Systems and Rehabilitation
Engineering* (2007),
[doi:10.1109/TNSRE.2007.906956](https://doi.org/10.1109/TNSRE.2007.906956).

The access, attribution, no-bundle, and no-transformed-cache decisions are the
same as BNCI2014-001.

Sources: [BNCI data-set registry](https://bnci-horizon-2020.eu/database/data-sets)
and [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/), accessed
2026-07-29.

### Cho2017

The project uses the binary left/right MI recordings from all 52 participants
with a rotating, class-stratified acquisition-order block holdout. All 52 have
already contributed to development and are not confirmation evidence.

Primary citation: Cho et al., “EEG datasets for motor imagery brain-computer
interface,” *GigaScience* 6(7), gix034 (2017),
[doi:10.1093/gigascience/gix034](https://doi.org/10.1093/gigascience/gix034).
The supporting-data identifier is
[GigaDB doi:10.5524/100295](https://doi.org/10.5524/100295).
The paper reports 52 participants, Gwangju Institute of Science and Technology
IRB approval, written informed consent, and research-purpose collection.

License finding:

- the article is CC BY 4.0, which is the article license;
- current GigaScience editorial policy says accompanying datasets are expected
  under CC0 unless a documented exception applies;
- the current NEMAR derivative record identifies itself as derived from the
  GigaDB DOI but reports CC BY 4.0; current MOABB catalog metadata also reports
  CC BY 4.0; and
- the exact terms attached to the bytes acquired for this project have not
  been captured in a release manifest.

Therefore, analysis access is supported by the public dataset DOI, but
redistribution is unresolved. Do not bundle originals or transformed caches.
At promotion, record whether acquisition came from GigaDB or a particular
mirror, that record's version/license, exact file hashes, and all required
citations.

Sources: [primary article](https://academic.oup.com/gigascience/article/6/7/gix034/3796323),
[GigaScience data policy](https://academic.oup.com/gigascience/pages/editorial_policies_and_reporting_standards),
[NEMAR derivative record](https://ww2.nemar.org/dataset/nm000245), and
[current MOABB dataset documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Cho2017.html),
accessed 2026-07-29.

### PhysioNet EEG Motor Movement/Imagery Database

The canonical source is the
[EEG Motor Movement/Imagery Dataset v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/),
published 2009, with
[doi:10.13026/C28G6P](https://doi.org/10.13026/C28G6P). It contains 109
participants, 64-channel EEG, 160 Hz recordings, and 14 runs in EDF+ format.

This project uses only imagined unilateral-fist runs 4, 8, and 12, mapping
their run-specific T1/T2 events to left/right. It does not use executed
movement or bilateral hands/feet for this track. S1-S54 are opened development
subjects. S55-S109 are scientifically sealed confirmation subjects and remain
inaccessible even though PhysioNet publicly lists them.

PhysioNet explicitly labels the files ODC-By 1.0 and allows anyone to access
them subject to that license. ODC-By permits sharing, modification, and use of
the database subject to attribution and notice requirements, but does not
necessarily license each individual content item or clear privacy rights.

Required citations from the official page:

- Schalk, *EEG Motor Movement/Imagery Dataset*, version 1.0.0, PhysioNet
  (2009), [doi:10.13026/C28G6P](https://doi.org/10.13026/C28G6P);
- Schalk et al., “BCI2000: A General-Purpose Brain-Computer Interface (BCI)
  System,” *Transactions on Biomedical Engineering* (2004),
  [doi:10.1109/TBME.2004.827072](https://doi.org/10.1109/TBME.2004.827072); and
- the standard PhysioNet citation shown on the dataset page at the time of
  release.

Release decision: analysis of S1-S54 is supported. S55-S109 remain sealed.
Although ODC-By can permit redistribution with attribution, this project will
not bundle raw files or caches. A reproducible release should point to
PhysioNet, pin v1.0.0, preserve the ODC-By notice and citations, and record the
official and acquired file hashes. Sources:
[official dataset record](https://physionet.org/content/eegmmidb/1.0.0/) and
[ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/), accessed
2026-07-29.

### Lee2019 / OpenBMI MI — sealed

All 54 participants and both sessions are dataset-level confirmation evidence.
The project uses only binary left/right MI. No participant recording from this
dataset may be downloaded, inspected, cached, or scored before unsealing.

Primary citation: Lee et al., “EEG dataset and OpenBMI toolbox for three BCI
paradigms: an investigation into BCI illiteracy,” *GigaScience* 8(5), giz002
(2019), [doi:10.1093/gigascience/giz002](https://doi.org/10.1093/gigascience/giz002).
The supporting-data identifier is
[GigaDB doi:10.5524/100542](https://doi.org/10.5524/100542). The paper reports
54 healthy participants, Korea University IRB
1040548-KUIRB-16-159-A-2, and written informed consent.

The paper's CC BY 4.0 notice licenses the article. Its “Availability of source
code” section specifies GPL 3.0 for the OpenBMI toolbox. Neither statement,
alone, proves the license governing every EEG data file. Current MOABB metadata
displaying GPL 3.0 must not be treated as a data redistribution grant. Although
GigaScience's general data policy expects CC0 absent an exception, the exact
GigaDB record/file terms must be captured when the dataset is lawfully
unsealed.

Release decision: public paper/metadata review only; dataset access and all
redistribution remain blocked. Sources:
[primary article](https://academic.oup.com/gigascience/article/8/5/giz002/5304369),
[GigaScience data policy](https://academic.oup.com/gigascience/pages/editorial_policies_and_reporting_standards),
and [current MOABB dataset documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html),
accessed 2026-07-29.

### Weibo2014 — sealed

All 10 participants are dataset-level confirmation evidence. The upstream
experiment contains seven class labels; this project uses only left-hand and
right-hand imagery. No participant recording may be downloaded, inspected,
cached, or scored before unsealing.

Primary citation: Yi et al., “Evaluation of EEG Oscillatory Patterns and
Cognitive Process during Simple and Compound Limb Motor Imagery,” *PLOS ONE*
9(12):e114853 (2014),
[doi:10.1371/journal.pone.0114853](https://doi.org/10.1371/journal.pone.0114853).
The paper reports 10 participants, Tianjin University ethics committee
approval, informed consent, and a data-availability statement pointing all raw
files to [Harvard Dataverse doi:10.7910/DVN/27306](https://doi.org/10.7910/DVN/27306).

The PLOS article is CC BY, but that is an article license. The paper describes
the raw data as available without restriction, and current MOABB metadata
reports CC0. Harvard Dataverse uses CC0 as a default but permits dataset owners
to choose custom terms, so repository-wide defaults are not proof of this
specific record's terms.

Release decision: public paper/metadata review only. At scientific unseal,
capture the exact Dataverse version, per-file restrictions, license/custom
terms, citations, file hashes, and access date before acquisition. Do not
bundle raw or derived data. Sources:
[primary PLOS article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0114853),
[current MOABB dataset documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Weibo2014.html),
and [Harvard Dataverse terms](https://support.dataverse.harvard.edu/harvard-dataverse-general-terms-use),
accessed 2026-07-29.

### Zhou2016 — sealed

All four participants are dataset-level confirmation evidence. The upstream
experiment has left-hand, right-hand, and foot imagery; this project uses only
left/right and holds out complete sessions. No participant recording may be
downloaded, inspected, cached, or scored before unsealing.

Primary citation: Zhou et al., “A Fully Automated Trial Selection Method for
Optimization of Motor Imagery Based Brain-Computer Interface,” *PLOS ONE*
11(9):e0162657 (2016),
[doi:10.1371/journal.pone.0162657](https://doi.org/10.1371/journal.pone.0162657).
The paper reports four participants, Anhui University IRB approval, written
informed consent, and supporting data at Figshare.

The original
[Figshare record v2, doi:10.6084/m9.figshare.2061654](https://figshare.com/articles/dataset/data_zip/2061654)
identifies the raw three-class EEG archive and displays CC0. Current MOABB
catalog metadata identifies its BIDS dataset as CC BY 4.0. These may be
different representations or mirrors; a license attached to one record must
not be silently applied to bytes obtained from another.

Release decision: public paper/metadata review only. At scientific unseal,
record the exact MOABB version and acquisition URL, repository record/version,
license and notices attached to those bytes, and file hashes. Do not bundle raw
or derived data. Sources:
[primary PLOS article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162657),
[original Figshare record](https://figshare.com/articles/dataset/data_zip/2061654),
and [current MOABB dataset documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Zhou2016.html),
accessed 2026-07-29.

## Scientific sealing contract

For PhysioNet S55-S109 and all Lee2019, Weibo2014, and Zhou2016 subjects,
“sealed” means:

- no downloader or dataset-loader invocation;
- no raw-file, directory, cache, manifest, checksum, or metadata probe against
  acquired storage;
- no subject preview, shape check, trial count, quality check, preprocessing,
  feature extraction, training, inference, or score;
- no use of outcomes to choose architecture, hyperparameters, preprocessing,
  stopping rules, exclusions, seeds, or analysis methods; and
- no delegation to another researcher or process to inspect the data.

Reviewing public papers and repository landing-page metadata for this
governance document does not unseal participant data.

Unsealing requires all of the following to be recorded before access:

1. frozen candidate architecture, source manifest, configuration, seeds,
   preprocessing, splits, exclusions, metrics, and statistical plan;
2. immutable development results and a written confirmation-analysis plan;
3. authorization from the study lead and the designated data/privacy steward;
4. an approved external storage location and access controls;
5. exact dataset terms, citations, version, loader version, and planned hashes;
6. a dated unseal event in the experiment ledger; and
7. a rule that confirmation results cannot trigger another development cycle
   while retaining their “confirmation” label.

## Public artifact decision table

| Artifact | Default public-release status | Required condition |
|---|---|---|
| Project source and configurations | Blocked pending project-license choice and third-party notices | No embedded data/private paths; dependency licenses recorded |
| Official-source downloader | Permitted after review | Points to canonical source, surfaces terms/citations, never bypasses access controls, and respects the scientific seal |
| Raw EEG or repository archives | **Do not publish** | Exceptional dataset-specific written legal, privacy, and governance approval |
| Filtered/resampled/rereferenced epochs or caches | **Do not publish** | Same as raw; BNCI adapted data additionally require resolution of CC BY-ND |
| Trial arrays, features, covariances, embeddings | **Do not publish** | Dataset-specific privacy and redistribution approval |
| Trial-level labels, probabilities, or predictions | **Do not publish** | Dataset-specific privacy approval and documented scientific need |
| Per-participant metadata or local-ID mapping | **Do not publish** | Explicit consent/IRB/data-steward approval; direct identifiers remain prohibited |
| Per-participant performance table | Hold for privacy review | Public-dataset IDs only, minimum necessary detail, no local linkability |
| Aggregate statistics and plots | Potentially publishable | Approved protocol/consent for Exp4, suppression review, citations, and no recoverable participant data |
| Trained weights | Hold for dataset/privacy review | Consent/DUA compatibility, license analysis, model leakage assessment, and written release decision |
| Dataset/cache manifest | Publish only a sanitized version | Dataset/version/source/file hashes and processing identity; no credentials, private paths, or protected metadata |

## Open blockers

The clean benchmark and any public paper artifact remain blocked until these
items are closed with dated evidence:

- **D01 — Local ethics:** Exp4 IRB approval/exemption, protocol number, dates,
  amendments, and secondary-analysis scope.
- **D02 — Local consent and data rights:** consent language, ownership,
  sponsor/DUA terms, privacy assessment, artifact-by-artifact release decision,
  and named approver.
- **D03 — Code licensing:** owner-selected project license plus complete
  third-party code notices. Dataset and article licenses cannot fill this gap.
- **D04 — Acquisition provenance:** clean UV lock, exact MOABB/acquisition
  versions, canonical source/version, per-file hashes, and captured terms for
  every file used.
- **D05 — Cache provenance:** immutable per-subject raw/cache identities and a
  release cache-manifest SHA-256 from the external result root.
- **D06 — BNCI derivatives:** written licensor permission or qualified legal
  review before any transformed BNCI material is distributed.
- **D07 — Cho license identity:** reconcile the original GigaDB policy with the
  CC BY metadata on current derivative mirrors and bind the answer to the exact
  acquired bytes.
- **D08 — Lee data license:** capture the GigaDB EEG-file terms; do not
  substitute the article's CC BY or the toolbox's GPL license.
- **D09 — Weibo record terms:** capture the exact Harvard Dataverse
  version/license/custom terms and file restrictions at authorized unseal.
- **D10 — Zhou source identity:** resolve original Figshare CC0 versus current
  MOABB BIDS CC BY metadata for the exact acquisition path.
- **D11 — Participant/model privacy:** review re-identification and model
  leakage risks, including weights, embeddings, per-subject metrics, and
  trial-level outputs.
- **D12 — Sealed confirmation authorization:** satisfy and log the complete
  unsealing contract before accessing any sealed subject.

Until these blockers are closed, reproducibility means publishing code,
protocols, citations, and sanitized provenance—not republishing participant
data.
