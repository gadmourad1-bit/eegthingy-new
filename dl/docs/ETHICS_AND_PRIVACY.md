# Ethics, privacy, and intended-use statement

## Research status

This is research software for offline motor-imagery EEG classification. It is
not a medical device, clinical decision system, communication aid approved for
patient use, or evidence that an assistive intervention benefits disabled or
paralyzed people.

The motivating goal is important, but the verified common-grid evidence comes
from previously opened development cohorts and primarily healthy volunteers.
No included experiment establishes safety, effectiveness, usability,
calibration, or quality-of-life benefit for the intended clinical population.

## Human-subject boundary

EEG is human-subject data and can remain sensitive after names are removed.
Signals, session patterns, rare phenotypes, linked metadata, model embeddings,
and trained weights may permit singling out, membership inference, or
re-identification. Pseudonyms are not anonymization.

For the private Local Exp4 cohort, the supplied project materials do not state
an IRB protocol number, approval/exemption status, consent version,
recruitment population, demographics, or data-use authorization. This project
therefore makes no claim about those facts. External release is blocked until
the responsible institution supplies and approves them. See
`docs/DATA_ACCESS.md`.

Public availability of the four external datasets does not remove the duties
to follow their terms, cite them, minimize data, protect participants, and
respect any withdrawal or use restrictions.

## Permitted present use

Subject to the user's institutional authorization and each dataset's terms,
the present defensible use is:

- offline methods research;
- reproducibility checks against the frozen development protocol;
- aggregate, protocol-labelled comparison of the 43 common-recipe models; and
- engineering evaluation of runtime, memory, and calibration.

The software must not be used to control mobility, communication, stimulation,
clinical care, or another safety-critical system without a separate approved
study and an appropriate real-time safety architecture.

## Foreseeable harms and limitations

- **False activation and non-response:** offline balanced accuracy does not
  measure asynchronous false commands, abstention, dwell time, fatigue, or
  failure during real-world assistive use.
- **Population shift:** performance can change with disability, paralysis,
  medication, lesions, implants, age, skin/hair characteristics, language,
  attention, fatigue, montage, amplifier, or care setting.
- **Unequal performance:** aggregate means can conceal participants for whom
  the decoder performs at or below chance. No demographic fairness analysis
  is available because approved demographic metadata were not supplied.
- **Calibration:** the descriptive common-grid leader had worse aggregate
  calibration than TCFormer in this development analysis; high confidence
  must not be interpreted as clinical reliability.
- **Privacy leakage:** participant-level outputs and weights can reveal more
  than aggregate tables.
- **Automation bias:** a high development score may encourage unsafe trust,
  especially when the observed winner was selected from 43 configurations.

## Requirements before a patient-facing study

At minimum, a responsible clinical and engineering team must establish:

1. IRB/ethics approval and consent specifically covering the target
   population, intended task, device interaction, data flows, and failure
   modes;
2. participatory design with disabled users, caregivers, clinicians, and
   accessibility specialists, including compensation and burden review;
3. a prospective protocol with a frozen model and independent participants;
4. real-time metrics: false activations per unit time, latency, information
   transfer, abstention, recalibration burden, fatigue, session drift, and
   participant-specific failure rates;
5. human override, fail-safe states, command confirmation, monitoring, and an
   incident-response plan;
6. security threat modeling, access controls, encryption, audit logging, and
   a retention/deletion schedule;
7. subgroup and worst-case analysis with a plan for non-users and declining
   performance;
8. clinical, regulatory, and institutional review of the claimed intended
   use; and
9. transparent communication that participation or poor decoder performance
   is not a measure of cognition, effort, competence, or eligibility for care.

## Publication and release checklist

Before releasing any artifact, obtain a written decision for each row:

| Artifact class | Current status | Required authority |
|---|---|---|
| In-house source code | License unresolved | Copyright holders and institutional release authority |
| Aggregate public-dataset statistics | Development-only; dataset citations required | Research team and dataset terms |
| Aggregate Local Exp4 statistics | Included in the internal sealed analysis; public-release authorization not documented here | PI, data steward, and IRB/privacy authority |
| Raw EEG or transformed caches | Not included and not authorized by this project | Dataset terms plus institutional/data-controller approval |
| Pseudonymous job/subject-seed/subject metrics | Included in the sealed analysis; still participant-derived and not approved here for public release | Explicit privacy/IRB/data-use and disclosure-control approval |
| Trial-level predictions | Not included | Explicit privacy/IRB/data-use approval |
| Embeddings or trained weights | Not included; potentially participant-derived | Privacy, IP, dataset, sponsor, and IRB/data-use review |
| Participant demographics | Not supplied | Consent, minimization, and disclosure-control review |

Do not backfill missing approvals, demographics, disability status, funding,
or consent language from assumptions. A manuscript should report unknowns as
unknowns until the responsible authority supplies them.

## Scientific claim boundary

The completed common grid supports one narrow statement: under one frozen
training recipe and five opened development datasets, the observed
equal-dataset balanced-accuracy leader was an in-house CardinalFBC
configuration. The descriptive uncertainty interval for its difference from
TCFormer included zero. The result is not independent confirmation, a global
state-of-the-art result, proof of novelty, or evidence of clinical benefit.
