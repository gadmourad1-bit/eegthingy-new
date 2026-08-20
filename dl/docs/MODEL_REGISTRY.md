# Canonical EEG motor-imagery model registry

> **Namespace and archive note.** Historical logical identifiers may use
> `eeg_mi`. The active implementation lives under `benchmark`;
> `pre-migration-package/` is a token-free display alias for a
> predecessor namespace in historical path records; it is not a literal or
> current filesystem path.

This document is the human-readable companion to
[`src/benchmark/model_registry.py`](../src/benchmark/model_registry.py). It records what
each name means, whether it denotes a distinct architecture or only a
configuration/procedure, what data representation it consumes, where its code
comes from, and where existing results can be found.

The registry contains 70 stable entries:

- 43 common-recipe neural configurations: exactly 28 in-house and 15 public;
- 8 additional in-house architectures or prototypes;
- 5 ORBIT configurations;
- 2 author-faithful reference procedures;
- 9 native-transfer procedures; and
- 3 classical controls.

The count of registry entries is not a count of novel architectures. Compact,
wide, extended-atlas, continuation-scale, loss, and transfer variants remain
configurations of their stated family. A transfer procedure is not relabeled
as a new network.

## Registry semantics

The stable ID is the join key for future result tables. The legacy common
factory name is retained separately as `common_roster_name`. Results should
never be joined by display name because typography and version labels can
change.

The four tracks are:

- `common`: one common raw-trial training recipe. This isolates architecture
  and configuration differences, but is not necessarily faithful to an
  author's published optimizer or augmentation.
- `author_faithful`: a separately declared attempt to reproduce important
  released training choices. “Author-faithful” does not mean bit-for-bit
  reproduction when inputs, loaders, or software differ.
- `procedure`: an architecture coupled to additional preprocessing,
  pretraining, adaptation, candidate selection, or a specialized protocol.
  These rows must not be ranked as if they differed only by architecture.
- `control`: deterministic classical estimators used as non-neural reference
  points.

`distinct_architecture` identifies the first registered computational design
of a family. `family_configuration` identifies a width, atlas, pooling,
normalization, loss, or continuation variant. `procedure_configuration`
identifies a training/transfer recipe. `classical_control` is self-explanatory.

Dataset eligibility covers the five opened benchmark datasets:
`local_exp4`, `bnci2014_001`, `bnci2014_004`, `cho2017`, and `physionet_mi`.
Eligibility is not proof that a result exists. It means that the registered
model contract is appropriate for the task. Existing coverage is a separate
field. In particular, the common BNCI2014-001 task is four-class; binary
left/right procedures are marked N/A there even if an older artifact used a
two-class subset of BNCI2014-001.

## Common 43-entry roster

The order below is frozen by the existing local tournament and must remain the
column/order contract for reproducible reruns.

| # | Stable ID / factory name | Display name | Family | Identity | Ownership |
|---:|---|---|---|---|---|
| 1 | `common.scope` / `scope` | SCOPE-Net | SCOPE | distinct | in-house |
| 2 | `common.free_scope` / `free_scope` | FreeSCOPE | SCOPE | configuration | in-house |
| 3 | `common.cardinal` / `cardinal` | CardinalField | CardinalField | distinct | in-house |
| 4 | `common.free_cardinal` / `free_cardinal` | FreeCardinal | CardinalField | configuration | in-house |
| 5 | `common.cardinal_dynamics` / `cardinal_dynamics` | CardinalDynamics | CardinalDynamics | distinct | in-house |
| 6 | `common.cardinal_dynamics_compact` / same | CardinalDynamics compact | CardinalDynamics | configuration | in-house |
| 7 | `common.cardinal_dynamics_extended` / same | CardinalDynamics extended | CardinalDynamics | configuration | in-house |
| 8 | `common.cardinal_dynamics_sinc` / same | CardinalDynamics Sinc | CardinalDynamics | configuration | in-house |
| 9 | `common.cardinal_dynamics_sinc_residual` / same | CardinalDynamics Sinc residual | CardinalDynamics | configuration | in-house |
| 10 | `common.cardinal_dynamics_sinc_extended` / same | CardinalDynamics Sinc extended | CardinalDynamics | configuration | in-house |
| 11 | `common.eegnet` / `eegnet` | EEGNet | EEGNet | distinct | public |
| 12 | `common.shallow` / `shallow` | ShallowFBCSPNet | ShallowFBCSPNet | distinct | public |
| 13 | `common.deep4` / `deep4` | Deep4Net | Deep4Net | distinct | public |
| 14 | `common.eegconformer` / `eegconformer` | EEG-Conformer | EEGConformer | distinct | public |
| 15 | `common.eegconformer_compact` / same | EEG-Conformer compact | EEGConformer | configuration | public architecture/local config |
| 16 | `common.atcnet` / `atcnet` | ATCNet | ATCNet | distinct | public |
| 17 | `common.atcnet_aggressive_pool` / same | ATCNet aggressive pooling | ATCNet | configuration | public architecture/local config |
| 18 | `common.fbcnet` / `fbcnet` | FBCNet | FBCNet | distinct | public |
| 19 | `common.cardinal_fbc` / `cardinal_fbc` | CardinalFBC | CardinalFBC | distinct | in-house |
| 20 | `common.cardinal_fbc_extended` / same | CardinalFBC extended | CardinalFBC | configuration | in-house |
| 21 | `common.cardinal_fbc_corr` / same | CardinalFBC correlation | CardinalFBC | configuration | in-house |
| 22 | `common.cardinal_fbc_corr_extended` / same | CardinalFBC correlation extended | CardinalFBC | configuration | in-house |
| 23 | `common.cardinal_fbc_compactdyn` / same | CardinalFBC compact dynamics | CardinalFBC | configuration | in-house |
| 24 | `common.cardinal_fbc_compactdyn_extended` / same | CardinalFBC compact dynamics extended | CardinalFBC | configuration | in-house |
| 25 | `common.cardinal_fbc_compactdyn_scale010` / same | CardinalFBC compact dynamics scale 0.10 | CardinalFBC | configuration | in-house |
| 26 | `common.cardinal_fbc_compactdyn_scale010_extended` / same | Previous + extended | CardinalFBC | configuration | in-house |
| 27 | `common.cardinal_fbc_compactdyn_scale025` / same | CardinalFBC compact dynamics scale 0.25 | CardinalFBC | configuration | in-house |
| 28 | `common.cardinal_fbc_compactdyn_scale025_extended` / same | Previous + extended | CardinalFBC | configuration | in-house |
| 29 | `common.cardinal_fbc_micro` / same | CardinalFBC micro dynamics | CardinalFBC | configuration | in-house |
| 30 | `common.cardinal_fbc_micro_extended` / same | CardinalFBC micro dynamics extended | CardinalFBC | configuration | in-house |
| 31 | `common.cardinal_fbc_physical` / same | CardinalFBC ordered physical dynamics | CardinalFBC | configuration | in-house |
| 32 | `common.cardinal_fbc_physical_extended` / same | Previous + extended | CardinalFBC | configuration | in-house |
| 33 | `common.eegtcnet` / `eegtcnet` | EEG-TCNet | EEGTCNet | distinct | public |
| 34 | `common.fbmsnet` / `fbmsnet` | FBMSNet | FBMSNet | distinct | public |
| 35 | `common.cardinal_fbms` / `cardinal_fbms` | CardinalFBMS | CardinalFBMS | distinct | in-house |
| 36 | `common.cardinal_fbms_extended` / same | CardinalFBMS extended | CardinalFBMS | configuration | in-house |
| 37 | `common.cardinal_mix` / `cardinal_mix` | CardinalMixedTemporal | CardinalMixedTemporal | distinct | in-house |
| 38 | `common.cardinal_mix_drop` / same | CardinalMixedTemporal dropout | CardinalMixedTemporal | configuration | in-house |
| 39 | `common.ctnet` / `ctnet` | CTNet, corrected wrapper | CTNet | distinct | public architecture/local wrapper |
| 40 | `common.ctnet_compact` / same | CTNet compact, corrected wrapper | CTNet | configuration | public architecture/local config |
| 41 | `common.eegsym` / `eegsym` | EEGSym | EEGSym | distinct | public |
| 42 | `common.eegsym_wide` / same | EEGSym wide | EEGSym | configuration | public architecture/local config |
| 43 | `common.tcformer` / `tcformer` | TCFormer | TCFormer | distinct | public official source/local adapter |

Every common entry accepts a variable output count and is registered for all
five opened datasets. All 43 have completed local Exp4 common-recipe coverage.
That coverage is development evidence: the local outer recording had already
been seen during model development.

## In-house common architectures

All code in this section is in
[`src/benchmark/models.py`](../src/benchmark/models.py), with construction in
[`src/benchmark/baselines.py`](../src/benchmark/baselines.py). The project repository
does not currently declare a license; that must be resolved before public code
release. Public backbones retain their own citations and implementation
licenses.

### Family: SCOPE

SCOPE-Net (Scalp-Coordinate Operator with Physical-frequency Energy) accepts
`(batch, channels, time)` raw trials plus one unit-sphere coordinate per
channel. A shared ordered Sinc bank learns bounded physical center frequencies
and bandwidths. A continuous spherical-polynomial scalp field is evaluated at
the observed electrodes and normalized geometrically, so learned spatial
parameters are not assigned to channel indices. For each source-frequency
pair, the decoder computes a direct log-energy floor and multi-resolution
energy/dynamics moments. Axial residual blocks mix frequency, source, and
feature dimensions without quadratic full-token attention. The residual head
is zero-initialized, so optimization starts at the direct log-energy model.

`free_scope` is not a second proposed continuous model. It replaces the
coordinate field with a fixed-order learned spatial matrix while keeping the
filter bank and decoder. Its purpose is to isolate the value of coordinate
continuity.

### Family: CardinalField

CardinalField uses 21 fixed standard-1005 inducing positions. A regularized
spherical Gaussian cardinal basis maps learned coefficients to any observed
montage. Sixteen ordered Sinc bands and 32 continuous sources feed four
segmented log-variance views and one linear classifier. The default common
configuration has no dropout. Cardinal interpolation is a weight-field
parameterization; it is not interpolation of trials or labels.

`free_cardinal` is the channel-indexed spatial control. It shares the ordered
filter bank and segmented log-variance decoder but has no coordinate-based
montage contract.

The longer theory and numerical invariants are documented in
[`src/benchmark/CARDINAL_FIELD_THEORY.md`](../src/benchmark/CARDINAL_FIELD_THEORY.md).

### Family: CardinalDynamics

CardinalDynamics replaces separate temporal and spatial stages with a
full-rank spatiotemporal cardinal tensor. It then forms two complementary
feature groups:

1. pooled log energy from continuous latent sources; and
2. signed dynamics from a depthwise temporal convolution, pointwise projection,
   and per-trial mean/standard-deviation summary.

One fixed classifier receives their concatenation; no validation-selected
router or post-hoc ensemble is part of the architecture.

| Common configuration | Temporal path | Sources | Dynamics | Atlas/extra |
|---|---|---:|---:|---|
| `cardinal_dynamics` | 32 free FIRs, kernel 25 | 32 | 16 | 21 anchors |
| `cardinal_dynamics_compact` | 24 free FIRs, kernel 25 | 24 | 12 | dropout 0.35 |
| `cardinal_dynamics_extended` | default | 32 | 16 | 31 anchors |
| `cardinal_dynamics_sinc` | 16 ordered Sinc filters, kernel 65 | 24 | 12 | 21 anchors |
| `cardinal_dynamics_sinc_residual` | previous + gated rank-four multiscale residual | 24 | 12 | 21 anchors |
| `cardinal_dynamics_sinc_extended` | ordered Sinc | 24 | 12 | 31 anchors |

The extended atlas is a configuration, not a distinct network family. It adds
ten inducing sites needed by the project's 15-channel deployment montage.

### Family: CardinalFBC

CardinalFBC retains the pinned Braindecode FBCNet spectral filter bank, batch
normalization, SiLU, four temporal log-variance views, and constrained
classifier. It replaces only FBCNet's grouped channel-indexed spatial
convolution with a cardinal-RBF weight field. On the 21-channel inducing
montage, a pseudoinverse initialization can reproduce the freshly seeded
indexed layer. On other montages, the same learned field is sampled at the
provided electrode coordinates.

The base family is therefore an in-house derivative with an explicit public
FBCNet dependency, not an original reimplementation of every FBCNet
component. Cite [Mane et al., FBCNet](https://arxiv.org/abs/2104.01233) when
using it.

All continuations preserve the FBC mapping at initialization by expanding the
single max-norm classifier with zero columns. They do not add a separately
trained classifier or a validation-selected ensemble.

| Configuration group | Added representation |
|---|---|
| base / extended | no continuation; 21- or 31-anchor field |
| correlation / extended | per-band rank-6 projections, four temporal windows, shrinkage correlations, off-diagonal Fisher-z features |
| compact dynamics / extended | 24 free FIRs, 24 continuous sources, log-energy pooling, 12 signed-dynamics channels |
| compact scale 0.10 / 0.25 | identical compact features multiplied by the stated scale before the shared head |
| micro / extended | eight learned zero-DC Gabor-initialized FIRs, eight sources, eight energy windows, eight dynamics channels |
| physical / extended | 16 ordered Sinc filters, 12 sources, eight energy windows, eight dynamics channels |

Every `/extended` suffix changes only the inducing atlas from 21 to 31 sites.
The scale variants change only a feature multiplier. They are ablations of the
same CardinalFBC architecture.

### Family: CardinalFBMS

CardinalFBMS is the FBMS analogue of the function-preserving coordinate-field
replacement. It retains FBMSNet's filter bank, mixed-scale temporal
convolution, normalization, activation, temporal log-variance views, and
constrained classifier. Only the grouped indexed spatial convolution becomes a
cardinal field. The default uses 36 temporal views, dilatability 8, stride
factor 4, and 21 anchors; the extended configuration uses 31 anchors.

This is an in-house derivative of
[Liu et al.'s FBMSNet](https://braindecode.org/stable/generated/braindecode.models.FBMSNet.html).
The Braindecode documentation explicitly states that its FBMSNet is a
reimplementation not checked by the original authors, so neither the public
control nor CardinalFBMS should be described as a bit-identical author release.

### Family: CardinalMixedTemporal

CardinalMixedTemporal takes the public FBMSNet filter bank and mixed temporal
block, then replaces the indexed spatial stage with a full-rank cardinal field
and a simple four-segment log-variance decoder. It is not
function-preserving with respect to the complete FBMSNet classifier. The
`cardinal_mix_drop` entry changes only decoder dropout from 0 to 0.25.

## Public models

The common public wrappers use `braindecode==1.6.1` except TCFormer. The
Braindecode implementation is BSD-3-Clause. Original architecture repositories
may have a different license. The common track uses the study's shared
optimizer and selection/refit protocol, so those scores are architecture
comparisons rather than author-recipe reproductions.

| Family | Registered configurations | Concise provenance and local adaptation |
|---|---|---|
| EEGNet | `eegnet` | [Lawhern et al. (2018)](https://braindecode.org/stable/generated/braindecode.models.EEGNet.html); Braindecode model with default compact temporal/depthwise/separable convolutions. |
| ShallowFBCSPNet | `shallow` | [Schirrmeister et al. (2017)](https://doi.org/10.1002/hbm.23730); local temporal and pooling lengths are 13 and 38/8. |
| Deep4Net | `deep4` | Same Schirrmeister paper; local temporal kernels are shortened to length 5 for the harmonized windows. |
| EEGConformer | `eegconformer`, `eegconformer_compact` | [Song et al.](https://doi.org/10.1109/TNSRE.2022.3230250); six Transformer layers versus a local four-layer compact configuration. |
| ATCNet | `atcnet`, `atcnet_aggressive_pool` | [Altaheri et al., DOI 10.1109/TII.2022.3197419](https://doi.org/10.1109/TII.2022.3197419); second pooling size 7 versus local aggressive size 4. |
| FBCNet | `fbcnet` | [Mane et al.](https://arxiv.org/abs/2104.01233); Braindecode implementation. An official-source author-recipe procedure is separately registered. |
| EEG-TCNet | `eegtcnet` | [Ingolfsson et al.](https://doi.org/10.48550/arXiv.2006.00622); local front-end kernel length 33. |
| FBMSNet | `fbmsnet` | [Liu et al.](https://braindecode.org/stable/generated/braindecode.models.FBMSNet.html); Braindecode reimplementation with 9 bands, 36 spatial filters, four mixed temporal kernels, and log variance. |
| CTNet | `ctnet`, `ctnet_compact` | [Zhao et al.](https://braindecode.org/stable/generated/braindecode.models.CTNet.html); local `CorrectedCTNet` restores configured pre-classifier dropout. Compact uses 8 temporal filters, embedding 16, and two heads instead of 20/40/four. |
| EEGSym | `eegsym`, `eegsym_wide` | [Perez-Velasco et al.](https://braindecode.org/stable/generated/braindecode.models.EEGSym.html); explicit channel pairs/midline list and a deterministic equivalent of the supported half-pooling operation. Wide changes filters per branch to 24. |
| TCFormer | `tcformer` | [Altaheri et al. (2025)](https://doi.org/10.1038/s41598-025-16219-7); official MIT source from commit `74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`, preserved as a hash-pinned compact runtime snapshot and loaded without modifying the upstream files. |

## Additional in-house architectures and procedures

### GeoAdaptNet family

`architecture.geoadaptnet` consumes four fixed-band SPD covariance matrices.
Its non-compressed anchor is a linear classifier over the full tangent feature
space. A learned residual performs per-band BiMap, ReEig, LogEig, encoding and
attention before fusion. A sigmoid gate begins near zero, protecting the
convex-like geometric floor from an initially random deep branch. This model
has a binary main task; an optional intent/rest head is auxiliary and never
silently changes the MI task to three classes.

`architecture.geoadaptnet_fb` instead begins with a learnable Sinc filter bank
on broadband trials, forms one full channel covariance per band, regularizes it
to the SPD cone, takes the matrix log at the identity, and fits a compact
linear head. `architecture.geoadaptnet_fbsp` adds a per-band learned BiMap
spatial reduction before ReEig/LogEig. FBSP is a configuration of the
learnable-filter geometric family, not another unrelated architecture.

Existing architecture/deployment results are in `predecessor-workspace/research-results` and are
summarized in the historical broader-tree inventory
`MODEL_ACCURACY_INVENTORY.md`, which is not distributed in this clean bundle.
Those historical protocols differ from the new common grid and must not be
merged as if they were missing cells of the same experiment.

### CAMEO

CAMEO combines three low-capacity experts:

1. a frozen, source-only, reflection-balanced log-Euclidean tangent logistic
   anchor;
2. a ShallowConvNet-inspired raw log-energy head; and
3. a signed raw temporal-dynamics head sharing the temporal/spatial bank.

Original and sagittally reflected trials form explicit counterfactual views.
Training penalizes even (nuisance) logit components. The protected validation
partition selects counterfactual subtraction strength and one predeclared
expert mixture; that routing is part of the procedure. CAMEO is binary and
therefore N/A for the four-class BNCI2014-001 common task.

### HemiParity and PARITY-Fuse

HemiParity constructs internal even and odd representations
`(h(x)+h(Mx))/2` and `(h(x)-h(Mx))/2` for sagittal reflection `M`. Raw and
tangent readouts are odd functions whose coefficients depend only on invariant
context. An invariant reliability gate fuses their logits. The tangent
estimator initializes the geometric odd readout, which remains trainable.

PARITY-Fuse tightens the construction: deterministic shared raw/tangent
encoders contain no dropout or BatchNorm, paired views pass in one call, and a
coordinate-wise invariant gate fuses odd latents. A bias-free residual starts
at zero over the projected convex tangent anchor. Both designs target the
binary group action “sagittal reflection swaps left and right labels”; they are
not valid four-class models without a new group action and head.

### OrbitTransportNet

Orbit transport begins with arbitrary paired expert logits `l(x)` and `l(Mx)`.
It decomposes them into invariant `e` and odd `o` components. An orientation
score is constrained to be odd in `o`, with coefficients conditioned only on
`(e, o²)`. This transports each raw expert to an exactly odd signed logit even
when the underlying expert is not equivariant. Energy, dynamics, and a
projected tangent anchor are fused by a gate that observes only invariant
statistics.

The five registered entries are one architecture plus procedure ablations:

| Stable ID | Difference | Existing coverage |
|---|---|---|
| `architecture.orbit_v1` | GroupNorm paired raw expert, initial auxiliary recipe | local, Cho S1--8, legacy binary BNCI001 screens |
| `architecture.orbit_v2` | view-symmetric BatchNorm | Cho S1--8 |
| `architecture.orbit_v3` | BatchNorm, no orientation penalty, stronger transport/view auxiliaries | local, Cho, legacy binary BNCI001; exact local outer-refit |
| `architecture.orbit_v4` | v3 with view auxiliary removed | Cho S1--8 |
| `architecture.orbit_v5` | v4 procedure restricted to raw-mean candidate | Cho S1--8 |

Only ORBIT-v3 is the frozen exact outer-refit comparator in the current local
50-condition report. The version suffixes do not mean five unrelated neural
architectures.

### HemiQ-FieldNet

HemiQ-FieldNet is deliberately specialized to BNCI2014-004's supplied-bipolar
C3/Cz/C4 strip. C3+C4 and Cz are even under reflection; C3-C4 is odd.
Constrained ordered quadrature Gabor pairs form local complex coefficients.
Power and coherence magnitudes provide invariant context; signed cross-products
and bounded log-power-asymmetry moments provide the odd field. Even-conditioned
blocks use bias-free maps on odd features, and invariant attention combines
bias-free odd token scores. The output is exactly anti-equivariant.

A source-only tangent logistic model may be used only as a decaying training
teacher. Validation and inference use the neural logit alone. The frozen
contract is not extended to point-electrode datasets: doing so would be a new
reference/montage experiment, not a missing score.

### CardinalSplineDualView

This bounded prototype evaluates one shared CardinalFBC backbone twice:
directly on the native electrodes and on raw voltages transported to the
canonical atlas by a fixed geometry-only spherical-spline matrix. Each view
has source-partition-only scaling. A 27-dimensional descriptor summarizes
per-band feature disagreement; a tiny gate predicts the canonical weight.
The descriptor is detached for the gate, and the final gate layer starts at
zero, giving exact 50/50 logit fusion at initialization. The encoder and
classifier are shared across views.

It is registered for the Cho/Physio native-transfer cohorts. The repository
contains implementation and objective tests but no identified formal
aggregate result, so it must not appear in a result table as a zero or failed
run.

## Author-faithful reference recipes

Two reference procedures live in
[`src/benchmark/reference_training.py`](../src/benchmark/reference_training.py):

- `reference.tcformer` uses official TCFormer commit
  `74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`, Adam with coupled L2,
  linear warmup followed by cosine decay, segmentation-and-reconstruction
  augmentation, and the fixed final-epoch horizon. The IV-2a/IV-2b horizons are
  released settings; local/Cho/Physio horizons are declared study adaptations.
- `reference.fbcnet` uses official FBCNet recipe commit
  `de1bbdd8a54cb1e466830e3d47070e0e56761a37`, the released causal
  Chebyshev-II filter design, Adam, and two-stage stopping around a
  Braindecode 1.6.1 FBCNet.

Both are explicitly “recipe-faithful on study-adapted inputs,” not bit-for-bit
reproductions. No formal all-dataset result artifact was identified for either
at registry creation.

## Native-transfer procedures

Native transfer is evaluated only on Cho2017 S16--52 and PhysioNet S1--54.
Cho S1--15 belongs to source pretraining, and PhysioNet S55--109 remains sealed
confirmation. The procedures are binary and are N/A for all other current
dataset cells.

The predecessor CardinalFBC screen has four registered conditions:

| Stable ID | Condition |
|---|---|
| `procedure.pretrained_cardinal_fbc` | fine-tune a canonical source-pretrained CardinalFBC |
| `procedure.scratch_cardinal_fbc` | matched scratch CardinalFBC |
| `procedure.scratch_fbmsnet_legacy_transfer` | native indexed FBMSNet scratch control |
| `procedure.pretrained_indexed_fbcnet_spherical_spline` | checkpoint-derived indexed FBCNet after fixed spherical-spline transport |

The CardinalFBMS full grid has five:

| Stable ID | Condition |
|---|---|
| `procedure.pretrained_cardinal_fbms` | fine-tune the byte-pinned canonical CardinalFBMS checkpoint |
| `procedure.scratch_cardinal_fbms_canonical_seeded` | scratch, canonical coordinate-field seed |
| `procedure.scratch_cardinal_fbms_native_projected` | scratch native indexed layer projected into the field |
| `procedure.scratch_fbmsnet_native` | fresh native indexed FBMSNet |
| `procedure.pretrained_indexed_fbmsnet_spherical_spline` | checkpoint-matched indexed FBMSNet after fixed spline transport |

The last condition is a checkpoint/transfer control, not an author-faithful
FBMSNet reproduction. Complete full-grid statistics are in the historical
artifact `pre-migration-package/results/NATIVE_FBMS_FULL_GRID_SUMMARY.md`.
Predecessor and fold-0 screens are in the adjacent
`NATIVE_TRANSFER_FOLD0_SEED7_SUMMARY.md` and
`NATIVE_FBMS_TRANSFER_FOLD0_SEED7_SUMMARY.md`.

## Classical controls

The three current controls consume a model-side four-band transform, fit all
states on source rows, select a predeclared hyperparameter on validation, reset
and refit on source+validation, and predict the held-out test only afterward.

- `control.riemann`: per-band affine-invariant Riemannian whitening and tangent
  maps, feature standardization, and L2 logistic regression.
- `control.tangent_anchor`: a train-only log-Euclidean reference, exact
  congruence recentering, matrix logarithm, upper-triangle vectorization,
  standardization, and L2 logistic regression.
- `control.ea_fbcsp`: train-only Euclidean alignment, per-band Ledoit-Wolf CSP,
  concatenated log-power features, and shrinkage LDA.

They are deterministic. Repeated seed IDs in the current local artifact are
declared replications of the same deterministic fit, not independent random
runs.

## Results and reporting boundary

The authoritative existing-result inventory in the broader research tree is
`MODEL_ACCURACY_INVENTORY.md`. The exact
current local 50-condition audit is the historical artifact
`pre-migration-package/results/local_exp4_allmodels_20260721/LOCAL_ALL_50_FINAL.md`.

Future comprehensive tables should use:

1. stable ID as the row key;
2. one column per dataset with `N/A — reason` for ineligible contracts and
   `not run` for eligible but incomplete cells;
3. balanced accuracy as the primary per-dataset metric;
4. equal-subject aggregation within each dataset;
5. an explicitly named equal-dataset macro rather than pooling trials across
   cohorts of very different size; and
6. separate table panels for common, author-faithful, procedure, and control
   tracks.

Do not convert missing results to chance accuracy or zero. Do not compare an
older transductive/calibrated score to a newer non-transductive score as if
they shared a protocol. Do not count compact, extended, or transfer entries as
independent architectural novelty. Finally, neither a development leaderboard
win nor failure to find a close paper is sufficient for “state of the art,”
clinical efficacy, or guaranteed novelty.

## Reproducibility checks

[`tests/core/test_model_registry.py`](../tests/core/test_model_registry.py)
checks:

- unique stable IDs;
- exact ordered coverage of all 43 common factory names;
- the declared 28/15 in-house/public split;
- exactly one eligibility decision for every opened dataset;
- mandatory N/A reasons; and
- strict lookup behavior.

The registry is metadata, not an execution factory. This is intentional: it
can be imported by report builders and audit tools without importing PyTorch,
Braindecode, MNE, or loading data.
