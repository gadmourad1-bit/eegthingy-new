# Methodology and architecture catalogue

## Scope of this document

This is the publication-facing methods summary for the clean project. The
source of truth for executable common-grid identities is
`ieee_mi.full_grid.COMMON_ARCHITECTURES` plus
`ieee_mi.baselines.make_model`. The broader ownership/provenance crosswalk is
`docs/INHOUSE_MODEL_METHODS.md` and the 70-entry registry is
`ieee_mi/model_registry.py`.

“In-house” means that this repository owns the implementation or the
project-specific derivative/procedure. It does not establish legal ownership,
patentability, literature novelty, clinical validity, or state of the art.
A compact, extended, wide, scale, pooling, dropout, loss, or initialization
variant is normally a configuration or ablation, not a new architecture.

## Verified common-recipe experiment

The completed common grid compared 43 raw-trial configurations on five
previously opened development datasets. Its immutable product was:

```text
43 models × 448 dataset/participant/fold units × 5 seeds = 96,320 jobs
```

The participant/fold dimensions were:

| Dataset | Participants | Folds per participant | Classes |
|---|---:|---:|---:|
| Local Exp4 | 8 | 1 fixed chronological fold | 2 |
| BNCI2014-001 | 9 | 1 official-session fold | 4 |
| BNCI2014-004 | 9 | 1 official-session fold | 2 |
| Cho2017 | 52 | 5 acquisition-order block folds | 2 |
| PhysioNet MI development | 54 | 3 imagery-run folds | 2 |

The seeds were 7, 17, 27, 37, and 47. Every common configuration used the same
data split, scaling, augmentation, optimizer, stopping, reset/refit, and test
prediction contract. Consequently, the table answers a shared-recipe
architecture/configuration question. It is not an author-recipe reproduction
for every external model.

### Preprocessing

Public-dataset trials use the `ieee-mi-cache-v2` harmonized profile: 4–40 Hz,
0.5–3.0 s half-open epochs, 128 Hz, 320 samples, epoch demeaning, and symmetric
common-average reference, except that BNCI2014-004 retains its three supplied
bipolar derivations. The common montage profile uses 21 declared channels
where available. Local Exp4 uses 4–40 Hz, 0.0–2.0 s half-open, 128 Hz/256
samples, epoch demeaning, and common-average reference over 15 point
electrodes. Details and rights boundaries are in `docs/DATA_ACCESS.md`.

Channel standardization is source-only. Selection training rows determine the
selection scaler. Train plus validation rows determine a new final-refit
scaler. Test rows are excluded from both.

### Selection, reset/refit, and prediction

Within every participant/fold/seed job:

1. install the seed before model construction;
2. optimize only on training rows and select the epoch count by validation
   cross-entropy;
3. reconstruct the exact seeded initialization;
4. refit from that initialization on train plus validation for the selected
   number of epochs; and
5. predict the held-out test rows once.

The score-blind production package stores row identities and normalized class
probabilities but not labels or performance. The analyzer joins labels only
after an exact 96,320-job audit.

### Shared optimization recipe

| Setting | Frozen value |
|---|---:|
| Maximum epochs | 200 |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 0.0008 |
| Weight decay | 0.0005 |
| Scheduler | Cosine annealing over the 200-epoch horizon |
| Early-stopping patience | 35 |
| Minimum validation-loss improvement | 0.0001 |
| Label smoothing | 0.05 |
| Gradient-norm clip | 5.0 |
| Segment reconstruction | 8 segments, batch-level probability 0.5 |
| Temporal shift | Uniform integer shift up to ±8 samples, zero padded |
| Additive Gaussian noise | Standard deviation 0.01 after channel scaling |
| Reflection augmentation/loss | Disabled in the common recipe |

Torch deterministic algorithms were required and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` was bound into formal execution. This is a
deterministic software contract for the recorded environment, not a promise
that different hardware or library versions will produce byte-identical
floating-point results.

### Analysis

The primary outcome is balanced accuracy. Folds are concatenated within each
participant and seed; seeds are averaged within participant; participants are
weighted equally within each dataset; and the five dataset means are weighted
equally. Trials, folds, and seeds are not treated as independent participants.

Secondary outcomes are accuracy, chance-normalized balanced accuracy, macro
F1, Cohen's kappa, one-vs-rest macro AUROC, negative log likelihood,
multiclass Brier score, and 15-bin expected calibration error. Parameter
count, selected epochs, fit/prediction time, and peak CUDA allocation are
engineering outcomes. Trial-micro results remain separate because they weight
datasets by trial count and pool repeated seed predictions.

The fixed-suite descriptive intervals use 100,000 deterministic
participant-cluster bootstrap resamples. Seeds are derived reproducibly from
the frozen base seed 20260729 and the comparison's frozen model index; the
reported leader-versus-TCFormer interval uses seed 20530729. Because the
leader was selected from 43 outcomes, its contrast with TCFormer is
descriptive and not a selection-adjusted confirmatory confidence interval.

## Common-grid in-house architecture families

### SCOPE

`ScopeNet` combines ordered physical-frequency Sinc filters with continuous
spherical-polynomial scalp fields and multiresolution energy/dynamics moments.
An axial frequency/source/feature mixer is a zero-started residual over the
direct energy floor. `FreeScopeNet` replaces the coordinate field with an
indexed spatial matrix and is the continuity control. Common keys: `scope`,
`free_scope`.

The control substantially outperformed SCOPE in the verified grid, so these
data do not demonstrate a benefit from SCOPE's coordinate-continuous field.

### CardinalField

`CardinalFieldNet` maps coefficients at 21 fixed standard-1005 inducing sites
through a regularized spherical Gaussian cardinal basis, then applies 16
ordered Sinc bands, 32 latent sources, four log-variance views, and a
classifier. `FreeCardinalFieldNet` uses fixed-order indexed channel weights.
Common keys: `cardinal`, `free_cardinal`.

The interpolation is applied to spatial weights, not EEG trials or labels.
The free control exceeded the cardinal version in the verified grid; this is
not evidence of a coordinate-field advantage.

### CardinalDynamics

`CardinalDynamicsNet` learns a continuous spatiotemporal cardinal tensor and
decodes latent-source log energy plus signed temporal-dynamics summaries.
The six common configurations are:

- `cardinal_dynamics`: default free-FIR, 21-anchor form;
- `cardinal_dynamics_compact`: fewer temporal/source/dynamics channels;
- `cardinal_dynamics_extended`: default form with a 31-anchor union atlas;
- `cardinal_dynamics_sinc`: ordered-Sinc compact form;
- `cardinal_dynamics_sinc_residual`: Sinc form plus a gated low-rank
  multiscale temporal residual; and
- `cardinal_dynamics_sinc_extended`: ordered-Sinc form with 31 anchors.

The Sinc-extended configuration was the observed Local Exp4 leader and ranked
third overall under the common recipe. The extended atlas is an ablation; the
result does not prove montage invariance or that all 31 inducing degrees of
freedom are identifiable.

### CardinalFBC

`CardinalFBCNet` is an in-house derivative of external FBCNet. It preserves
FBCNet's spectral filter bank, normalization, activation, temporal
log-variance views, and constrained classifier while replacing the grouped
indexed spatial convolution with a regularized cardinal-RBF weight field. A
fresh indexed layer can be projected into that field to preserve the initial
mapping.

Continuation variants keep the CardinalFBC floor and one constrained head;
new head columns start at zero:

- base: `cardinal_fbc`, `cardinal_fbc_extended`;
- shrinkage/Fisher-z correlation features: `cardinal_fbc_corr` and its
  `extended` form;
- compact energy/dynamics continuation: `cardinal_fbc_compactdyn` and its
  `extended` form;
- fixed continuation scales 0.10 and 0.25, each with 21- and 31-anchor forms;
- learned zero-DC Gabor micro-dynamics: `cardinal_fbc_micro` and its
  `extended` form; and
- ordered-Sinc physical dynamics: `cardinal_fbc_physical` and its `extended`
  form.

`cardinal_fbc_compactdyn_scale025_extended` was the observed overall
common-grid leader at 73.9900% equal-dataset balanced accuracy. This is an
outcome-selected configuration and an FBCNet derivative, not evidence of a
wholly independent architecture, confirmed superiority, or global SOTA.

### CardinalFBMS

`CardinalFBMSNet` is the analogous in-house derivative of external FBMSNet. It
preserves the public spectral bank, mixed-scale temporal block, normalization,
temporal views, and constrained classifier while replacing the indexed
spatial convolution with a cardinal field. Common keys: `cardinal_fbms` and
`cardinal_fbms_extended` (21 versus 31 anchors).

Do not transfer evidence from the separately trained pretrained-CardinalFBMS
procedure to these from-scratch common rows.

### CardinalMixedTemporal

`CardinalMixedTemporalNet` reuses FBMSNet's spectral and mixed-temporal blocks,
then uses a full-rank cardinal spatial field and four-segment log-variance
decoder. `cardinal_mix_drop` differs from `cardinal_mix` only by decoder
dropout 0.25. These are FBMSNet-component derivatives and performed below the
leading common configurations.

## In-house families outside the common recipe

These families must not be appended to the common leaderboard. Their paired
views, SPD features, transfer checkpoints, routing, dataset-specific signals,
or author-specific training rules define different scientific questions.

| Family | Mechanism and identity | Evidence boundary |
|---|---|---|
| GeoAdaptNet / GeoAdaptNet-FB / FBSP | Fixed-band SPD tangent anchor plus gated deep residual, or learned-Sinc filter-bank covariance models with optional rank-8 BiMap | Separate binary geometric/deployment protocol. Residual was largely inert and the family did not establish classifier novelty or superiority. |
| CAMEO | Convex log-Euclidean tangent anchor plus raw energy/dynamics experts on original/reflected views; validation selects subtraction and mixture | Binary procedure; routing is inseparable from the reported result. Historical and current-local protocols cannot be pooled. |
| HemiParity | Even/odd raw and tangent representations with invariant-context odd readout and reliability gate | Binary reflection-equivariant procedure; not eligible for the four-class task. |
| PARITY-Fuse | Deterministic paired raw/tangent encoders, invariant odd-latent fusion, and zero-start residual over a convex tangent anchor | Binary procedure; a related but distinct parity family from HemiParity. |
| HemiQ-FieldNet | Exact reflection-odd cross-spectral field for the three supplied BNCI2014-004 bipolar signals | Dataset-specific. Historical five-subject confirmation was effectively tied with ShallowConvNet; no superiority claim. |
| CardinalSplineDualView | One CardinalFBC backbone evaluated on native and fixed spherical-spline-transported views, mixed by a detached-disagreement gate | Cho/PhysioNet transfer prototype; no promoted aggregate result or production writer. |
| OrbitTransportNet | Converts arbitrary paired expert logits into an exactly odd transported score using invariant context; fuses energy, dynamics, and tangent anchor | ORBIT-v1 is the distinct architecture; v2–v5 are configurations/ablations. The local v3 procedure is separate from common training. |

### Native-transfer procedures

Pretrained/scratch CardinalFBC and CardinalFBMS entries are training
procedures, not additional architectures. They use native-montage target
caches and, where applicable, a byte-pinned source checkpoint. The target
cohorts are Cho2017 S16–S52 and PhysioNet S1–S54; source-only scaling,
validation-only epoch selection, initialization reset, train+validation
refit, and one-shot outer-test prediction remain mandatory.

The five in-house registry conditions are `pretrained_cardinal_fbc`,
`scratch_cardinal_fbc`, `pretrained_cardinal_fbms`,
`scratch_cardinal_fbms_canonical_seeded`, and
`scratch_cardinal_fbms_native_projected`. The last two distinguish a fresh
canonical coordinate-field initialization from projection of a fresh native
indexed layer; neither is a new architecture family.

The CardinalFBC transfer family has only limited predecessor coverage under
its own protocol. The five-condition CardinalFBMS transfer grid is a separate,
audited development artifact; it is not a common-grid substitute and provides
no sealed confirmation or clinical evidence.

## Negative experimental families

### CHSD

The unregistered CHSD family represents native-montage complex coherency as a
Hermitian positive-definite surface, uses matrix-log states and explicit
time/frequency/mixed differences, and tests direct, residual, joint, and
conditioned decoders. Matrix logarithms, complex connectivity, filter banks,
coordinate projection, and attention are not individually claimed as novel.

Its executable scratch keys are `chsdnet`, `chsdnet_direct`,
`chsdnet_hybrid`, `chsdnet_joint`, and conditioned bounds
`chsdnet_conditioned_005`, `chsdnet_conditioned_010`, and
`chsdnet_conditioned_020`. They are intentionally unregistered; code
existence is not a completed result.

The bounded 0.05 conditioned candidate reached 73.543% equal-dataset balanced
accuracy in its separate 115-participant robustness amendment, below the
74.087% CardinalFBC-micro-extended reference, and failed its frozen
dataset-breadth and participant-win-rate clauses. The decision was
`stop_chsd_promotion`. Other CHSD variants remain code/screen variants, not
promoted models.

### GaugeQuotientCrossMomentNet

Gauge combines a preserved CardinalFBC path with learned sources,
band/window/lag cross moments, mask-specific coordinate quotient features,
antisymmetric lag wedges, and a zero-start continuation. The defensible
candidate claim was the complete neural composition, not invention of
lagged covariance, the exterior product, time asymmetry, average-reference
projection, causality, or global invariance. Close equation-level prior art is
documented in `docs/GAUGE_QUOTIENT_NOVELTY_PLAN.md`.

Its prespecified 17-subject Gate 1 produced 75.0704% equal-dataset balanced
accuracy versus 74.3988% for CardinalFBC-micro-extended, a +0.6716-point
margin, and 11/17 subject wins. Nevertheless, dataset deltas were nonnegative
on only two of five datasets; the frozen rule required at least three. Gate 1
therefore failed and the persisted decision was
`kill_candidate_before_disjoint_gate`. Gate 2, tuning, renamed retries,
promotion, and SOTA language are forbidden. Gauge is an informative separate
negative result and is not one of the 43 common-grid models.

## External common-grid models

The 15 external common configurations are EEGNet, ShallowFBCSPNet, Deep4Net,
EEG-Conformer (default and compact), ATCNet (default and aggressive pooling),
FBCNet, EEG-TCNet, FBMSNet, CTNet (default and compact), EEGSym (default and
wide), and TCFormer. These names, their architectures, and their original
papers belong to their respective authors. Local compatibility corrections,
shape variants, and use under this project's shared recipe do not change that
ownership.

TCFormer uses the compact pinned upstream source at commit
`74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`; all other public implementations
must be traced through the locked software and cited to their original method
papers in the manuscript.

## Novelty boundary

The completed benchmark can establish reproducible behavior of implemented
configurations under its protocol. It cannot by itself establish patent or
literature novelty. The winning configuration is a derivative plus an
extended-atlas/continuation setting, and its descriptive advantage over
TCFormer is uncertain. Any paper claim should be limited to the precisely
implemented coordinate-field continuation and complete evaluation procedure,
with equation-level prior-art comparison and ablations. Avoid “first,”
“unique,” “reference invariant,” “top globally,” and “state of the art” unless
separate systematic evidence supports the exact statement.
