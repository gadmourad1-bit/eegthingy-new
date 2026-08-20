# In-house neural model methods, provenance, and evidence

Status: repository audit completed 2026-07-29.

> **Namespace and archive note.** Historical logical identifiers may use
> `eeg_mi`. The active implementation lives under `benchmark`;
> `pre-migration-package/` is a token-free display alias for a
> predecessor namespace in historical path records; it is not a literal or
> current filesystem path. Other unavailable broader-tree paths are likewise
> shown only as provenance. Those artifacts are not part of this clean
> release.

This document is a methods-and-provenance crosswalk, not a leaderboard and not
a novelty determination. Its authoritative identity source is
[`src/benchmark/model_registry.py`](../src/benchmark/model_registry.py). Architecture
details were checked against the source implementations, the current registry
documentation, the experiment ledger, the accuracy inventory, the GeoAdaptNet
handoff and reports, the HemiQ report, and the current paper draft.

## Scope and ownership boundary

The registry contains 70 records. Exactly 46 have the literal ownership value
`in_house`; those 46 are the subject of the exact crosswalk below:

- 28 common-recipe architecture/configuration records;
- 13 separately evaluated architecture/procedure records; and
- 5 in-house native-transfer procedure records.

“In-house” means that this project owns the repository implementation or the
project-specific derivative/procedure. It does **not** by itself mean that the
method is legally protectable, literature-novel, clinically validated, or
state of the art. The repository does not currently declare a project license.

The following are external architectures and must never be described as ours:
EEGNet, ShallowFBCSPNet, Deep4Net, EEG-Conformer, ATCNet, FBCNet, EEG-TCNet,
FBMSNet, CTNet, EEGSym, and TCFormer. Their compact/wide/pooling variants,
corrected wrappers, local adapters, shared-recipe training, and
author-recipe runners do not transfer architecture ownership to this project.
In particular, TCFormer is official external source pinned at commit
`74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5` and retained as a verified compact
MIT-licensed runtime snapshot; the common FBCNet/FBMSNet implementations come
from `braindecode==1.6.1`. See
[`docs/MODEL_REGISTRY.md`](MODEL_REGISTRY.md) for citations, URLs, pins, and
licenses.

CardinalFBC is our coordinate-field derivative of external FBCNet;
CardinalFBMS is our coordinate-field derivative of external FBMSNet; and
CardinalMixedTemporal reuses FBMSNet's filter-bank/mixed-temporal components.
Only the explicitly described replacement, continuation, or procedure is
in-house. The underlying public architecture must still be cited.

The three registry controls (`control.riemann`, `control.tangent_anchor`, and
`control.ea_fbcsp`) are excluded from the 46-count because they are non-neural
classical methods with ownership
`in_house_implementation_of_classical_method`. The code is local, but the
methods are not original to this project.

The RA-L draft at historical path `paper/main.tex` is a separate
decision-point shared-control paper. Its evaluated decoders are classical
EA-FBCSP and Riemannian tangent-space logistic regression. It is not evidence
for any neural model in this document, and
the historical handoff `deepnet_CHATGPT_HANDOFF.md` explicitly says
the neural classifier work was not to be folded into that manuscript.

## Evidence vocabulary

The short evidence labels in the crosswalk mean:

- **C** — Five-seed local Exp4 common-recipe development tournament:
  `pre-migration-package/results/local_exp4_allmodels_20260721/neural/final_ranking.json`
  and
  `pre-migration-package/results/local_exp4_allmodels_20260721/LOCAL_ALL_50_FINAL.md`.
  Recording 4 had informed development, so this is post-selection development,
  not independent confirmation.

- **S** — Single-seed four-dataset Cardinal screens summarized in historical
  inventory `MODEL_ACCURACY_INVENTORY.md` and
  [`src/benchmark/EXPERIMENT_LEDGER.md`](../src/benchmark/EXPERIMENT_LEDGER.md). No v4-v6
  candidate passed the prespecified architecture-advancement gate.

- **G** — GeoAdaptNet-family experiments in historical summary
  `src/benchmark/research/RESULTS.md` and the JSONs under
  `predecessor-workspace/research-results/`. These mix deployment, architecture-only, and diagnostic
  protocols and must not be pooled into one score.

- **L** — Exact current local outer-refit procedure artifact at historical
  location `pre-migration-package/results/local_exp4_allmodels_20260721/procedures`,
  with the same outer rows as the local tournament but a separate
  procedure-specific covariance/reflection path.

- **D** — Development artifacts only, usually under `predecessor-workspace/research-results/cameo`
  or `predecessor-workspace/research-results/parity`. The cohort/protocol named in the artifact
  remains part of the result identity.

- **Q** — HemiQ's one-time BNCI2014-004 confirmation, frozen manifest, and
  permanent receipt, documented in
  [`src/benchmark/research/HEMIQ_FIELD_REPORT.md`](../src/benchmark/research/HEMIQ_FIELD_REPORT.md). This is
  five healthy confirmation subjects, not a population-superiority or
  clinical result. The frozen
  manifest `predecessor-workspace/hemi_q/hemi_q_final_frozen_manifest.json`
  hashes to
  `2bbe28d4415a74cb007328579cf8153585ceaaa28219246aecd0622398987401`;
  the
  confirmation
  `predecessor-workspace/hemi_q/hemi_q_final_bnci004_confirmation_s5_9.json`
  and permanent
  receipt
  `predecessor-workspace/hemi_q/.confirmation-receipts/eegthingy-hemiq-bnci2014-004-s5-9-final-v1.json`
  hashes are recorded in the report.

- **FBC-T** — CardinalFBC native-transfer fold-0/seed-7 development screen in
  historical artifact `pre-migration-package/results/NATIVE_TRANSFER_FOLD0_SEED7_SUMMARY.md`.
  The candidate failed its paper-lead gate.

- **FBMS-T** — Complete five-seed/all-fold CardinalFBMS development grid in
  historical artifact `pre-migration-package/results/NATIVE_FBMS_FULL_GRID_SUMMARY.md`.
  The frozen development gate passed; this is still opened development
  evidence, not sealed confirmation or SOTA evidence. The historical
  full-grid gate
  `pre-migration-package/results/native_fbms_transfer_full_grid_gate.json`
  hashes to
  `eab9f6ee1b548000eed823025330aa13c20616b8ed266cf5dd6b521a7ef80496`.

- **P** — Implementation/unit-test prototype only; no formal aggregate
  accuracy artifact was identified in the repository.

Dataset abbreviations are **L** = local Exp4, **A** = BNCI2014-001,
**B** = BNCI2014-004, **C** = Cho2017, and **P** = PhysioNet MI. The current
common A task is four-class. Binary-only procedure artifacts made from an
older left/right subset of A cannot fill that four-class cell.

## Shared common-recipe protocol

All 28 common entries are constructed by
`benchmark.baselines.make_model(<common_roster_name>, ...)`, implemented in
[`src/benchmark/baselines.py`](../src/benchmark/baselines.py), and trained as raw-trial
architectures under the common track. The classes themselves are in
[`src/benchmark/models.py`](../src/benchmark/models.py). They accept variable output counts
and are registry-eligible for L/A/B/C/P. This eligibility is a model contract,
not proof of completed five-dataset results.

The current comprehensive local artifact used recordings 1-2 for fitting,
recording 3 for selection, a fresh reset and recordings 1-3 for refit, and
recording 4 for prediction only. It used five seeds and participant-equal
balanced accuracy. The common five-dataset full grid is a future experiment;
the local artifact and the older single-seed screens must not be represented
as that grid.

### SCOPE family

`ScopeNet` learns ordered physical-frequency Sinc filters, evaluates continuous
spherical-polynomial scalp fields at electrode coordinates, and computes
direct log energy plus multi-resolution energy/dynamics moments. Axial
frequency/source/feature mixing is a zero-started residual over the direct
floor. `FreeScopeNet` retains the temporal and decoder design but replaces the
coordinate field with a channel-indexed spatial matrix. It is the continuity
control, not a second proposed continuous model.

Evidence is **C** only. In the current local development tournament SCOPE and
FreeSCOPE were near the bottom of the 50-condition table. This is an important
negative result: the current implementation is not a lead model and has no
cross-dataset confirmation artifact.

### CardinalField family

`CardinalFieldNet` uses 21 fixed standard-1005 inducing positions. A
regularized spherical Gaussian cardinal basis turns learned anchor
coefficients into a continuous scalp-weight field, followed by 16 ordered Sinc
bands, 32 sources, four segmented log-variance views, and one classifier.
`FreeCardinalFieldNet` keeps the filter/decoder pattern but uses fixed-order
channel-indexed weights. The interpolation acts on weights, not trials or
labels. Mathematical invariants and failure cases are in
[`src/benchmark/CARDINAL_FIELD_THEORY.md`](../src/benchmark/CARDINAL_FIELD_THEORY.md).

Evidence is **C** only. FreeCardinal exceeded CardinalField locally, so the
available artifact does not establish an advantage for coordinate continuity.

### CardinalDynamics family

`CardinalDynamicsNet` uses a full-rank continuous spatiotemporal cardinal
tensor. Its fixed classifier receives pooled latent-source log energy and
signed temporal-dynamics summaries. The compact, extended-atlas, ordered-Sinc,
and gated multiscale-residual entries are configurations of this one family:

- default: 32 free FIRs, 32 sources, 16 dynamics channels, 21 anchors;
- compact: 24 FIRs, 24 sources, 12 dynamics channels, dropout 0.35;
- extended: the default sampled from the 31-anchor union atlas;
- Sinc: 16 ordered length-65 Sinc filters, 24 sources, 12 dynamics channels;
- Sinc residual: the Sinc configuration plus a gated rank-four multiscale
  temporal residual; and
- Sinc extended: the ordered-Sinc configuration with 31 anchors.

Evidence is **C**. Extended-atlas variants were much stronger than their
21-anchor counterparts locally, but the 31-anchor atlas is an anatomical
support ablation, not a separately selectable architecture or proof of montage
generalization.

### CardinalFBC family

`CardinalFBCNet` retains the public Braindecode FBCNet spectral filter bank,
normalization, activation, four temporal log-variance views, and constrained
classifier. It replaces FBCNet's grouped channel-indexed spatial convolution
with a regularized cardinal-RBF weight field. With the inducing montage,
fresh indexed weights can be projected into the field so the initial mapping
is function-preserving.

All continuation models keep the CardinalFBC floor and one shared constrained
head. New head columns start at zero:

- correlation: low-rank, shrinkage-regularized signed-source correlations and
  off-diagonal Fisher-z features;
- compact dynamics: a compact CardinalDynamics energy/dynamics continuation;
- scales 0.10/0.25: fixed multipliers on the same compact continuation;
- micro dynamics: eight learned zero-DC Gabor-initialized FIRs, sources,
  energy windows, and dynamics channels; and
- ordered physical dynamics: ordered Sinc filters with source-energy and
  dynamics summaries.

Every `extended` entry changes the inducing atlas from 21 to 31 sites. It does
not create a new family.

Evidence is **C+S**. The 31-anchor micro configuration was the strongest
in-house common entry in the current local tournament, but external TCFormer
was numerically higher. Separately, TCFormer and the procedure-track ORBIT-v3
were not statistically separated in the participant-level local comparison.
In the earlier four-dataset v4-v6 screens no
CardinalFBC continuation passed the advancement gate. In particular, the
best v5 margin over FBMSNet was small and dataset-inconsistent, and the best
v6 number belonged to a non-selectable 31-anchor ablation. These results do
not support a global-best or SOTA claim.

### CardinalFBMS and CardinalMixedTemporal

`CardinalFBMSNet` is the FBMSNet analogue of CardinalFBC: it retains the public
FBMSNet filter bank, mixed-scale temporal block, normalization, activation,
temporal log-variance views, and constrained classifier, replacing only its
grouped indexed spatial convolution with a cardinal field. The common default
uses 21 anchors; `extended` uses 31.

`CardinalMixedTemporalNet` also starts from the public FBMSNet spectral and
mixed-temporal blocks, but uses a full-rank cardinal spatial field and a simple
four-segment log-variance decoder. It is not function-preserving with respect
to the complete FBMSNet classifier. `cardinal_mix_drop` changes only decoder
dropout from 0 to 0.25.

The common models have **C+S** for CardinalFBMS and **C** for
CardinalMixedTemporal. Neither common CardinalFBMS configuration passed the
v6 architecture gate; both CardinalMixedTemporal configurations performed
poorly and were highly variable locally. Do not transfer the later success of
the *pretrained CardinalFBMS procedure* to the from-scratch common architecture
rows.

## Separate architecture/procedure families

These models are registry track `procedure`, not additional columns in the
common raw-trial architecture table. Most are binary-only and therefore N/A
for the registered four-class A task. The current execution audit is
[`docs/NON_COMMON_TRACK_EXECUTION.md`](NON_COMMON_TRACK_EXECUTION.md).

### GeoAdaptNet, GeoAdaptNet-FB, and GeoAdaptNet-FBSP

`benchmark.research.model.GeoAdaptNet` consumes four fixed-band SPD covariance matrices.
It combines a full tangent-space linear anchor with a near-zero-gated
BiMap-ReEig-LogEig residual and band attention. An optional scalar intent/rest
head is auxiliary; it never changes the two-class MI target. Its deployment
procedure additionally includes label-free covariance recentering, a separate
causal decision-boundary state, confidence gating, and abstention.

`benchmark.research.filterbank_net.FilterBankSPDNet` instead learns a Sinc bandpass bank
from broadband EEG before covariance formation and log-Euclidean tangent
classification. GeoAdaptNet-FB keeps full channel covariance
(`reduced_dim=None`); GeoAdaptNet-FBSP applies a learned per-band spatial BiMap
(`reduced_dim=8`) before ReEig/LogEig.

Evidence is **G**. The main negative result is well established: the
GeoAdaptNet deep residual stayed near its initialization and contributed
negligibly; deep supervision, gate opening, and CSP warm starts did not repair
it. A convex tangent head moved performance back toward the classical tangent
decoder. On Cho2017, four learned temporal bands recovered only a small amount,
FBSP hurt, and nine learned bands overfit. GeoAdaptNet did not beat
ShallowConvNet externally. The handoff's publication conclusion is explicit:
GeoAdaptNet is not defensibly a novel classifier; its stronger potential
contribution is the deployed safety/abstention procedure.

### CAMEO

`benchmark.research.cameo_net.CAMEOClassifier` combines a frozen source-only convex
log-Euclidean tangent anchor with shared raw log-energy and signed-dynamics
experts. Original and sagittally reflected trials form explicit
counterfactual views. Training penalizes reflection-even nuisance logits;
protected validation chooses the subtraction strength and one predeclared
expert mixture. That routing is part of the procedure, so CAMEO is not a
common-recipe architecture-only result.

Evidence is **D+L**. The repository contains frozen local and Cho confirmation
artifacts
(`predecessor-workspace/research-results/cameo/cameo_v1_frozen_local_confirm_3seeds.json` and
`predecessor-workspace/research-results/cameo/cameo_v1_frozen_cho_confirm27_52_f5.json`), as
well as the exact current local outer-refit artifact at
`pre-migration-package/results/local_exp4_allmodels_20260721/procedures/cameo.json`.
These are different protocols and cannot be pooled. The local test recording
had been used during later project development; the current local table is
therefore development-only. CAMEO remains binary and has no current
all-dataset common result.

### HemiParity and PARITY-Fuse

`benchmark.research.parity_net.HemiParityClassifier` forms internal even and odd raw and
tangent representations from a trial and its sagittal reflection. Odd
readouts are conditioned only by invariant context; an invariant reliability
gate fuses the logits. The source-only tangent estimator initializes, but does
not freeze, the geometric odd readout.

`benchmark.research.parity_fuse_net.ParityFuseClassifier` tightens the same group action:
shared deterministic raw/tangent encoders process paired views, a
coordinate-wise invariant gate fuses odd latents, and a bias-free residual
starts at zero over a projected convex tangent anchor. Both target the binary
action “reflection swaps left and right”; neither is a four-class architecture.

Evidence is **D+L**. Historical local/Cho/legacy-binary-A development screens
exist, plus current local outer-refit artifacts for HemiParity at
`pre-migration-package/results/local_exp4_allmodels_20260721/procedures/hemiparity.json` and
PARITY-Fuse at
`pre-migration-package/results/local_exp4_allmodels_20260721/procedures/parity_fuse.json`.
The legacy binary A artifacts are incompatible with the current four-class A
benchmark. PARITY-Fuse and HemiParity did not lead the current local
50-condition table.

### HemiQ-FieldNet

`benchmark.research.hemi_q_field_net.HemiQFieldClassifier` is specific to the three
supplied bipolar C3/Cz/C4 derivations in BNCI2014-004. Parity coordinates split
hemispheric even, midline, and C3-C4 odd signals. A shared constrained
quadrature Gabor bank supplies local and full-epoch cross-spectral tokens.
Invariant context conditions bias-free odd blocks, and invariant attention
sums odd token scores. This enforces an exactly label-negating output under
C3/C4 reflection. A projected tangent teacher is training-only.

Evidence is **Q**. The frozen development manifest and one-time S5-S9
confirmation receipt passed post-run audit. HemiQ's 82.500% confirmation mean
was effectively tied with ShallowConvNet's 82.375%; five subjects make exact
inference coarse, and no superiority claim is supported. The data are healthy
volunteers, within-subject with labeled calibration sessions. Point-electrode
transfer would be a new procedure because the source signals are bipolar
derivations.

### CardinalSplineDualView

`benchmark.dual_view.CardinalSplineDualViewNet` evaluates one shared CardinalFBC
backbone in two deterministic views: directly on the native montage and after
fixed geometry-only Perrin spherical-spline transport to the canonical
montage. A tiny detached-disagreement gate mixes the logits and starts at exact
50/50 fusion. The leakage-safe fitting boundary is
`benchmark.dual_view_training.fit_dual_view_target`.

Evidence is **P**. The model is a bounded development prototype, not a promoted
transfer condition. It has no formal aggregate result, source checkpoint,
atomic artifact writer, or operations auditor. It is eligible only for the
registered Cho/PhysioNet target-transfer cohorts.

### OrbitTransportNet

`benchmark.research.orbit_transport_net.OrbitTransportClassifier` takes arbitrary expert
logits for a trial and its reflection, decomposes them into even and odd
components, and learns an odd orientation score whose coefficients depend only
on invariant `(even, odd^2)` context. The transported logit is therefore
exactly odd even if the underlying expert is not equivariant. Energy,
dynamics, and a projected tangent anchor are fused by an invariant gate.

ORBIT-v1 is the distinct registered architecture. v2 adds view-symmetric
BatchNorm; v3 uses BatchNorm with stronger transport/view auxiliary losses and
is the frozen current local procedure; v4 removes the view auxiliary loss; v5
restricts the candidate set to raw-mean. v2-v5 are configurations/ablations,
not four new architectures.

Evidence is **D**, with **L** additionally for v3. In its sealed legacy-binary-A
confirmation
`predecessor-workspace/research-results/parity/orbit_transport_final_bnci_confirmation_s5_9.json`,
frozen OrbitTransportNet scored below the tangent and Riemannian controls. The
current local ORBIT-v3 artifact at
`pre-migration-package/results/local_exp4_allmodels_20260721/procedures/orbit_v3.json`
was numerically close to TCFormer, but the difference was negligible and the
protocols differ (procedure versus common recipe). There is no global-top
claim.

## Native-transfer procedures

These entries are training/transfer procedures, not new network families.
Their frozen target cohorts are Cho2017 S16-S52 and PhysioNet S1-S54. They use
native montage caches, a source checkpoint where applicable, source-only
target scaling, validation-only epoch selection, a reset/refit from the exact
initial state on train+validation, and one prediction-only outer test pass.

The predecessor implementation is
[`src/benchmark/native_transfer.py`](../src/benchmark/native_transfer.py), with the atomic
call `run_record(..., condition=<key>)`. The CardinalFBMS implementation is
[`src/benchmark/native_fbms_transfer.py`](../src/benchmark/native_fbms_transfer.py), also
through `run_record`.

`Pretrained CardinalFBC` fine-tunes a canonical source checkpoint on a native
target montage; `Scratch CardinalFBC` uses the matched target recipe without
learned source weights. Evidence is **FBC-T**. Pretraining beat scratch, but the
candidate exceeded the strongest per-dataset reference envelope by only a
negligible macro margin and failed the frozen 0.5-point lead gate.

`Pretrained CardinalFBMS` fine-tunes the byte-pinned canonical source
checkpoint. The two in-house scratch controls distinguish a canonical
coordinate-field initialization without source weights from projection of a
fresh native indexed layer into the coordinate field. Evidence is **FBMS-T**.
The pretrained procedure passed all six frozen development checks and led its
matched five-condition transfer table, but it remains development-only. The
single source seed, absence of sealed confirmation, and use of healthy public
cohorts preclude SOTA, population, and clinical claims.

## Exact 46-entry registry crosswalk

All rows below have registry ownership exactly `in_house`. “Source binding”
gives the implementation class/module and, where registry identity depends on
configuration, the exact factory or runtime key that disambiguates it.

### Common track: 28 entries

| Stable ID / display name | Identity and ownership dependency | Exact source binding | Intended role / architectural change | Evidence; material limitation |
|---|---|---|---|---|
| `common.scope` — SCOPE-Net | distinct in-house architecture | `benchmark.models.ScopeNet`; `make_model("scope")` | Coordinate-continuous Sinc/scalp-field energy-dynamics model | **C**; weak local result, no confirmation |
| `common.free_scope` — FreeSCOPE | SCOPE configuration/control | `benchmark.models.FreeScopeNet`; `make_model("free_scope")` | Channel-indexed control for coordinate continuity | **C**; control exceeded SCOPE locally |
| `common.cardinal` — CardinalField | distinct in-house architecture | `benchmark.models.CardinalFieldNet`; `make_model("cardinal")` | 21-anchor cardinal-RBF field plus ordered Sinc/log variance | **C**; did not lead locally |
| `common.free_cardinal` — FreeCardinal | CardinalField configuration/control | `benchmark.models.FreeCardinalFieldNet`; `make_model("free_cardinal")` | Channel-indexed CardinalField control | **C**; exceeded coordinate model locally |
| `common.cardinal_dynamics` — CardinalDynamics | distinct in-house architecture | `benchmark.models.CardinalDynamicsNet`; `make_model("cardinal_dynamics")` | Default full-rank continuous energy+dynamics field | **C**; 21-anchor configuration only |
| `common.cardinal_dynamics_compact` — CardinalDynamics compact | family configuration | same class; `make_model("cardinal_dynamics_compact")` | 24-filter/source, 12-dynamics compact ablation | **C**; not a separate family |
| `common.cardinal_dynamics_extended` — CardinalDynamics extended | family configuration | same class; `make_model("cardinal_dynamics_extended")` | Default model with 31-anchor union atlas | **C**; extended support is non-selectable ablation |
| `common.cardinal_dynamics_sinc` — CardinalDynamics Sinc | family configuration | same class; `make_model("cardinal_dynamics_sinc")` | Replace free FIRs with 16 ordered Sinc filters | **C**; large local deficit versus extended form |
| `common.cardinal_dynamics_sinc_residual` — CardinalDynamics Sinc residual | family configuration | same class; `make_model("cardinal_dynamics_sinc_residual")` | Add gated low-rank multiscale residual | **C**; residual did not lead |
| `common.cardinal_dynamics_sinc_extended` — CardinalDynamics Sinc extended | family configuration | same class; `make_model("cardinal_dynamics_sinc_extended")` | Ordered-Sinc model with 31 anchors | **C**; no independent cross-dataset confirmation |
| `common.cardinal_fbc` — CardinalFBC | distinct in-house **FBCNet derivative** | `benchmark.models.CardinalFBCNet`; `make_model("cardinal_fbc")` | Replace indexed FBCNet spatial convolution with 21-anchor field | **C+S**; public FBCNet must be cited; no gate pass |
| `common.cardinal_fbc_extended` — CardinalFBC extended | family configuration, FBCNet derivative | same class; `make_model("cardinal_fbc_extended")` | 31-anchor atlas | **C+S**; atlas ablation, not new family |
| `common.cardinal_fbc_corr` — CardinalFBC correlation | family configuration, FBCNet derivative | `benchmark.models.CardinalFBCCorrelationNet`; `make_model("cardinal_fbc_corr")` | Zero-started low-rank Fisher-z correlation continuation | **C+S**; dataset-inconsistent screen gains |
| `common.cardinal_fbc_corr_extended` — CardinalFBC correlation extended | family configuration, FBCNet derivative | same class; `make_model("cardinal_fbc_corr_extended")` | Correlation continuation, 31 anchors | **C+S**; strong local development only |
| `common.cardinal_fbc_compactdyn` — CardinalFBC compact dynamics | family configuration, FBCNet derivative | `benchmark.models.CardinalFBCCompactDynamicsNet`; `make_model("cardinal_fbc_compactdyn")` | Zero-started compact energy/dynamics continuation | **C+S**; no advancement |
| `common.cardinal_fbc_compactdyn_extended` — CardinalFBC compact dynamics extended | family configuration, FBCNet derivative | same class; `make_model("cardinal_fbc_compactdyn_extended")` | Compact continuation, 31 anchors | **C+S**; atlas ablation |
| `common.cardinal_fbc_compactdyn_scale010` — CardinalFBC compact dynamics scale 0.10 | family configuration, FBCNet derivative | same class; `make_model("cardinal_fbc_compactdyn_scale010")` | Fixed 0.10 continuation multiplier | **C+S**; bounded scale ablation |
| `common.cardinal_fbc_compactdyn_scale010_extended` — CardinalFBC compact dynamics scale 0.10 extended | family configuration, FBCNet derivative | same class; `make_model("cardinal_fbc_compactdyn_scale010_extended")` | 0.10 multiplier and 31 anchors | **C+S**; non-selectable atlas ablation |
| `common.cardinal_fbc_compactdyn_scale025` — CardinalFBC compact dynamics scale 0.25 | family configuration, FBCNet derivative | same class; `make_model("cardinal_fbc_compactdyn_scale025")` | Fixed 0.25 continuation multiplier | **C+S**; canonical variant trailed envelope |
| `common.cardinal_fbc_compactdyn_scale025_extended` — CardinalFBC compact dynamics scale 0.25 extended | family configuration, FBCNet derivative | same class; `make_model("cardinal_fbc_compactdyn_scale025_extended")` | 0.25 multiplier and 31 anchors | **C+S**; best v6 number was non-selectable and below gate |
| `common.cardinal_fbc_micro` — CardinalFBC micro dynamics | family configuration, FBCNet derivative | `benchmark.models.CardinalFBCMicroDynamicsNet`; `make_model("cardinal_fbc_micro")` | Zero-started learned-Gabor micro continuation | **C+S**; no gate pass |
| `common.cardinal_fbc_micro_extended` — CardinalFBC micro dynamics extended | family configuration, FBCNet derivative | same class; `make_model("cardinal_fbc_micro_extended")` | Micro continuation, 31 anchors | **C+S**; best in-house common local row, still development only |
| `common.cardinal_fbc_physical` — CardinalFBC ordered physical dynamics | family configuration, FBCNet derivative | `benchmark.models.CardinalFBCPhysicalDynamicsNet`; `make_model("cardinal_fbc_physical")` | Zero-started ordered-Sinc energy/dynamics continuation | **C+S**; no gate pass |
| `common.cardinal_fbc_physical_extended` — CardinalFBC ordered physical dynamics extended | family configuration, FBCNet derivative | same class; `make_model("cardinal_fbc_physical_extended")` | Ordered-physical continuation, 31 anchors | **C+S**; strong local development only |
| `common.cardinal_fbms` — CardinalFBMS | distinct in-house **FBMSNet derivative** | `benchmark.models.CardinalFBMSNet`; `make_model("cardinal_fbms")` | Replace indexed FBMSNet spatial convolution with 21-anchor field | **C+S**; common model failed v6 gate |
| `common.cardinal_fbms_extended` — CardinalFBMS extended | family configuration, FBMSNet derivative | same class; `make_model("cardinal_fbms_extended")` | 31-anchor atlas | **C+S**; not the pretrained transfer procedure |
| `common.cardinal_mix` — CardinalMixedTemporal | distinct in-house **FBMSNet-component derivative** | `benchmark.models.CardinalMixedTemporalNet`; `make_model("cardinal_mix")` | FBMS temporal components + full-rank cardinal field + new decoder | **C**; weak/high-variance local result |
| `common.cardinal_mix_drop` — CardinalMixedTemporal dropout | family configuration, FBMSNet-component derivative | same class; `make_model("cardinal_mix_drop")` | Decoder dropout 0.25 only | **C**; no meaningful local improvement |

### Separate architecture/procedure track: 13 entries

| Stable ID / display name | Identity | Exact source binding | Eligible data / intended role | Evidence; material limitation |
|---|---|---|---|---|
| `architecture.geoadaptnet` — GeoAdaptNet | distinct architecture within a deployment procedure | `benchmark.research.model.GeoAdaptNet`; `benchmark.research.engine.train_model` | L/B/C/P binary; tangent anchor plus gated SPD residual and optional intent head | **G**; residual inert, classical/deep baselines often stronger |
| `architecture.geoadaptnet_fb` — GeoAdaptNet-FB | distinct learnable-filter geometric architecture | `benchmark.research.filterbank_net.FilterBankSPDNet`; `FilterBankSPDClassifier(reduced_dim=None).fit(...)` | L/B/C/P binary; learn Sinc bands before full covariance/log tangent | **G**; modest Cho gain, still below ShallowConvNet |
| `architecture.geoadaptnet_fbsp` — GeoAdaptNet-FBSP | GeoAdaptNet-FB configuration | same class/fitter with `FilterBankSPDClassifier(reduced_dim=8)` | L/B/C/P binary; add learned per-band spatial BiMap | **G**; spatial reduction hurt in diagnostic |
| `architecture.cameo` — CAMEO-Net | distinct architecture plus validation routing | `benchmark.research.cameo_net.CAMEOClassifier`; local `benchmark.local_outer_refit_benchmark.run_one(model_name="cameo", ...)` | L/B/C/P binary; counterfactual tangent/energy/dynamics experts | **D+L**; procedure-specific, protocols cannot be pooled |
| `architecture.hemiparity` — HemiParityNet | distinct binary equivariant architecture | `benchmark.research.parity_net.HemiParityClassifier`; local `run_one(model_name="hemiparity", ...)` | L/B/C/P binary; internal even/odd raw+tangent fusion | **D+L**; legacy A is binary and incompatible with current A |
| `architecture.parity_fuse` — PARITY-Fuse | distinct binary equivariant architecture | `benchmark.research.parity_fuse_net.ParityFuseClassifier`; local `run_one(model_name="parity_fuse", ...)` | L/B/C/P binary; deterministic latent parity fusion | **D+L**; did not lead local table |
| `architecture.hemi_q_field` — HemiQ-FieldNet | distinct dataset-specific architecture | `benchmark.research.hemi_q_field_net.HemiQFieldClassifier`; `benchmark.research.hemi_q_benchmark.run(...)` | B only; exact odd cross-spectral field on supplied bipolar signals | **Q**; five-subject confirmation, tied strong public baseline |
| `architecture.cardinal_spline_dual_view` — CardinalSplineDualView | distinct bounded transfer prototype | `benchmark.dual_view.CardinalSplineDualViewNet`; `benchmark.dual_view_training.fit_dual_view_target` | C/P target cohorts; shared native/canonical view fusion | **P**; no aggregate result or production writer |
| `architecture.orbit_v1` — ORBIT-v1 | distinct OrbitTransportNet architecture | `benchmark.research.orbit_transport_net.OrbitTransportClassifier`; v1 artifact config | L/B/C/P binary; exact odd transport of arbitrary expert pairs | **D**; no current all-dataset adapter |
| `architecture.orbit_v2` — ORBIT-v2 batch | family configuration | same class; v2 view-symmetric-BN config artifact | L/B/C/P binary; BN variant | **D**; S1-S8 Cho screen only |
| `architecture.orbit_v3` — ORBIT-v3 batch auxiliary | family configuration / frozen local procedure | same class; v3 auxiliary-loss config; local `run_one(model_name="orbit_v3", ...)` | L/B/C/P binary; frozen strongest ORBIT procedure | **D+L**; close locally, lost legacy A confirmation to controls |
| `architecture.orbit_v4` — ORBIT-v4 transport-only | family ablation | same class; v4 artifact config | L/B/C/P binary; remove view auxiliary loss | **D**; S1-S8 Cho screen only |
| `architecture.orbit_v5` — ORBIT-v5 raw-mean-only | family ablation | same class; v5 artifact config | L/B/C/P binary; restrict to raw-mean candidate | **D**; S1-S8 Cho screen only |

### In-house native-transfer procedures: 5 entries

| Stable ID / display name | Identity | Exact source binding | Intended comparison | Evidence; material limitation |
|---|---|---|---|---|
| `procedure.pretrained_cardinal_fbc` — Pretrained CardinalFBC | procedure configuration | `benchmark.native_transfer.run_record(condition="pretrained_cardinal_fbc")` | Source-pretrained coordinate field fine-tuned on native C/P target | **FBC-T**; fold-0/seed-7 only, failed lead gate |
| `procedure.scratch_cardinal_fbc` — Scratch CardinalFBC | procedure configuration | `benchmark.native_transfer.run_record(condition="scratch_cardinal_fbc")` | Matched from-scratch CardinalFBC target control | **FBC-T**; not an independent architecture |
| `procedure.pretrained_cardinal_fbms` — Pretrained CardinalFBMS | procedure configuration | `benchmark.native_fbms_transfer.run_record(condition="pretrained_cardinal_fbms")` | Byte-pinned source checkpoint fine-tuned on native C/P target | **FBMS-T**; development gate pass, no sealed confirmation |
| `procedure.scratch_cardinal_fbms_canonical_seeded` — Scratch CardinalFBMS, canonical seeded | procedure configuration | `benchmark.native_fbms_transfer.run_record(condition="scratch_cardinal_fbms_canonical_seeded")` | Scratch target from canonical field parameterization, no source weights | **FBMS-T**; scratch control, not a new family |
| `procedure.scratch_cardinal_fbms_native_projected` — Scratch CardinalFBMS, native projected | procedure configuration | `benchmark.native_fbms_transfer.run_record(condition="scratch_cardinal_fbms_native_projected")` | Project a fresh native indexed layer into the field | **FBMS-T**; scratch initialization ablation |

## Unregistered CHSD experimental family

The repository also contains a new **unregistered** in-house research family.
It is not part of the 46-entry crosswalk, cannot yet join formal results by a
stable ID, and must not be described as frozen:

| Scratch key | Source | Mechanism and intended role | Repository evidence status |
|---|---|---|---|
| `chsdnet` | `benchmark.chsd.CHSDNet`; `make_chsd_model` | Native-montage multitaper complex coherency HPD surface; Hermitian matrix-log state plus explicit time, frequency, and mixed finite differences; factorized decoder plus separate power/ERD branch | Implementation and tests; no result artifact stored in this repository |
| `chsdnet_direct` | `benchmark.chsd_direct.CHSDDirectNet`; `make_chsd_direct_model` | Flattened native HSD fields with an active state head and zero-started differential/power heads | Implementation and tests; unregistered development ablation |
| `chsdnet_hybrid` | `benchmark.chsd_hybrid.CHSDResidualNet`; `make_chsd_hybrid_model` | CardinalFBC-micro-extended logits plus a bounded zero-started CHSD residual logit | Implementation and tests; unregistered development ablation |
| `chsdnet_joint` | `benchmark.chsd_joint.CHSDJointNet`; `make_chsd_joint_model` | Append CHSD summaries as zero columns in CardinalFBC's one constrained head | Implementation and tests; unregistered development ablation |
| `chsdnet_conditioned_005` | `benchmark.chsd_conditioned.CHSDSurfaceConditionedCardinalFBC`; `make_chsd_conditioned_model` | HSD-conditioned multiplicative modulation of FBC band-by-view floor, bounded at 0.05 | Completed 115-participant, five-dataset disjoint-subject robustness amendment; decision `stop_chsd_promotion` |
| `chsdnet_conditioned_010` | same class/factory | Same conditioning, bound 0.10 | Prespecified scratch screen code; no repository result artifact |
| `chsdnet_conditioned_020` | same class/factory | Same conditioning, bound 0.20 | Prespecified scratch screen code; no repository result artifact |

The proposed novelty is the narrow HSD representation/decoder conjunction,
not matrix logarithms, complex connectivity, filter banks, coordinate
projection, or attention individually. The bounded literature assessment in
[`docs/CHSD_NOVELTY_AUDIT.md`](CHSD_NOVELTY_AUDIT.md) is amber, not a proof of
novelty. It also prescribes kill criteria: complex features must beat real,
magnitude, and power controls; differential fields must beat state-only on
more than one dataset; and an ordinary equal-capacity 2-D decoder matching the
full HSD invalidates the representation advantage. The conditioned 0.05
candidate subsequently reached 73.543% equal-dataset balanced accuracy in the
post-v3 disjoint-subject amendment, versus 74.087% for CardinalFBC micro
extended, 73.628% for FBCNet, and 72.862% for TCFormer. It failed the frozen
dataset-breadth and paired-subject win-rate clauses. Therefore it remains an
unregistered negative result and cannot be promoted from code existence or
numerical competitiveness.

## Important global limitations

1. Nearly all experiments use healthy volunteers. None establishes benefit in
   disabled or paralyzed people, asynchronous false-activation safety, or
   assistive-device efficacy.
2. Local Exp4 has only eight development participants. The current common and
   procedure rankings are post-selection because the outer recording informed
   project development.
3. Scores from common, procedure, transfer, deployment, author-recipe, and
   classical-control tracks answer different questions. They cannot be merged
   into one unlabeled leaderboard.
4. A compact/wide/extended/scale/loss/initialization variant is not a new
   architecture. This audit identifies 46 registry records, not 46 novel
   networks.
5. The strongest confirmed in-house neural artifact, HemiQ, is
   dataset-specific and statistically tied with a strong public baseline on
   five confirmation subjects. The strongest complete transfer result is
   development-only. No repository artifact supports an unqualified SOTA or
   “top globally” statement.
6. Extended 31-anchor models may help the local deployment montage, but
   current data do not identify all added inducing degrees of freedom. They
   are ablations, not evidence of montage invariance.
7. Public-backbone derivatives require both the public citation/license and an
   exact description of our change.

## Registry ambiguities and audit findings

The 46-record coverage is exact against the current registry, but several
identity bindings remain underspecified in registry metadata:

1. `architecture.geoadaptnet_fb` and
   `architecture.geoadaptnet_fbsp` name the same class. Their identity is
   `reduced_dim=None` versus `reduced_dim=8`; the registry implementation
   string alone is insufficient.
2. ORBIT-v1 through v5 name one classifier class. Their identity depends on
   historical configuration artifacts; a stable-ID-to-hash-pinned
   configuration resolver is required before a new all-dataset run.
3. Native-transfer registry fields name modules rather than the exact
   `run_record` callable and condition key. The crosswalk above supplies the
   missing runtime binding.
4. Every common family configuration shares a class with one or more siblings.
   `common_roster_name` plus `benchmark.baselines.make_model` is therefore part
   of the executable identity.
5. CardinalSplineDualView's registry path identifies the model, while
   `fit_dual_view_target` is the leakage-safe training boundary. Neither is an
   atomic production runner.
6. CAMEO, HemiParity, PARITY-Fuse, and ORBIT are labeled `architecture.*` but
   correctly live in the `procedure` track because their paired views,
   covariance anchors, selection, or auxiliary losses are inseparable from
   the reported procedure.
7. The project license is unknown/not declared for every in-house record.
8. CHSD has seven executable scratch keys but no stable registry IDs. Until
   one candidate is selected, frozen, and registered, a formal grid must not
   invent an ID or merge the variants.
9. The registry's evidence strings are useful summaries, not cryptographic
   result manifests. Publication tables should join stable IDs to validated
   artifact hashes rather than rely on display names or free-text coverage.

Coverage result: **46 of 46 exact-`in_house` registry records documented**.
Additionally documented but intentionally excluded from that count: seven
unregistered CHSD scratch variants, 21 external neural/reference/transfer
records, and three locally implemented classical controls.
