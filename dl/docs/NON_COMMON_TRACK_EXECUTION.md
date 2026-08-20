# Non-common track execution audit

This document is the human-readable companion to
`benchmark.track_matrix`. The module produces an exact 70-model × 5-dataset
matrix (350 cells) without importing PyTorch, opening a cache, or starting a
training process.

The matrix exists to prevent a scientifically invalid shortcut: procedure,
transfer, author-recipe, and classical-control scores are not missing cells of
the common raw-trial architecture grid. They remain separately labelled
experiments even when they use the same subjects, folds, seeds, metrics, and
cache schema.

Live commands in this document use `benchmark`.
`pre-migration-package/` is a token-free display alias for a predecessor
namespace in one historical artifact path; it is not a literal or current
filesystem path. `eeg-mi-cache-v2` is the active versioned cache schema.

Run the read-only validator with the project UV environment:

```bash
uv run python -m benchmark.track_matrix
```

Emit the complete machine-readable manifest with all 350 cells, execution
descriptors, drift findings, and backlog:

```bash
uv run python -m benchmark.track_matrix --json
```

No command in `benchmark.track_matrix` loads EEG, builds a cache, imports a
training framework, or executes a benchmark.

## Matrix outcome

The validated status partition is:

| Cell status | Count | Meaning |
|---|---:|---|
| `common_full_grid` | 215 | One of 43 common configurations on one of five datasets; execute only under the common raw-trial recipe. |
| `separate_eligible` | 28 | Registry-eligible and backed by a cache/protocol-bound executor, but report in its author-faithful/procedure/control track. |
| `registry_na` | 47 | Principled N/A copied exactly from the registry, usually binary versus the four-class BNCI2014-001 task, a frozen dataset-specific procedure, or an architecture whose frozen spatial rank exceeds the supplied three-channel bipolar input. |
| `missing_adapter_or_blocker` | 60 | Scientifically eligible in the registry, but no current executor binds the implementation to the locked cache, split, reset/refit, and artifact contracts. |
| **Total** | **350** | Exact 70 × 5 Cartesian product. |

The 43 common configurations require 96,320 atomic execution units and result
records (`43 × 2,240`). The 27 non-common entries require 47,274 proposed
atomic execution units and result records across eligible cells. The
deterministic controls contribute 1,344 of those units: one seedless
subject/fold execution produces one score-blind prediction record. The entire
registered matrix therefore represents 143,594 proposed atomic execution
units and 143,594 result records. These counts include blocked proposals but
do not include CHSD or another new candidate until that candidate is frozen
and added to the registry.

Of the total, 17,119 non-common units are currently executable, 30,155 are
blocked by a missing adapter or protocol decision, and 96,320 belong to the
common grid. Registry-N/A cells contribute zero.

No new candidate was frozen into the common runner: CHSD stopped at its
prespecified robustness gate. The formal common scope therefore remains
exactly 43 models and 96,320 records. This does not execute ready non-common
records, resolve blocked records, or turn the common result table into an “all
models” table; those tracks remain scientifically and operationally separate.

## Frozen dimension contracts

Dataset abbreviations below are L = local Exp4, A = BNCI2014-001, B =
BNCI2014-004, C = Cho2017, and P = PhysioNet MI development.

| Grid kind | L | A | B | C | P | Per-entry total |
|---|---:|---:|---:|---:|---:|---:|
| Common or author-faithful stochastic units/records | 40 | 45 | 45 | 1,300 | 810 | 2,240 |
| Binary procedure stochastic units/records | 40 | N/A | 45 | 1,300 | 810 | 2,195 |
| Native-transfer target units/records | N/A | N/A | N/A | 925 | 810 | 1,735 |
| Deterministic-control seedless units/records | 8 | 9 | 9 | 260 | 162 | 448 |

The common dimensions are:

- local Exp4: 8 subjects × 1 fixed chronological fold × 5 seeds;
- BNCI2014-001: 9 subjects × 1 official-session fold × 5 seeds;
- BNCI2014-004: 9 subjects × 1 official-session fold × 5 seeds;
- Cho2017: 52 subjects × 5 rotating acquisition-order folds × 5 seeds; and
- PhysioNet development: 54 subjects × 3 rotating imagery-run folds × 5
  seeds.

Native transfer deliberately uses Cho S16–S52, not S1–S52, because Cho S1–S15
belong to the source-pretraining corpus. Its exact dimensions are 37 × 5 × 5 =
925 Cho records and 54 × 3 × 5 = 810 PhysioNet records per condition.

For each deterministic control, a subject/fold cell is executed once and
emits one score-blind prediction record. The new all-dataset control runner
has no seed dimension and creates no synthetic seed aliases. Historical local
artifacts that repeated deterministic outcomes under five seed identities
must not be used to inflate the apparent sample size.

## The 27 non-common entries

“Ready cells” means that a current cache/protocol-bound callable exists. It
does not mean the score belongs in the common table, and it does not imply that
a rerun is preferable to auditing an already complete artifact.

| Stable ID | Eligible datasets | Ready cells now | Proposed records |
|---|---|---|---:|
| `architecture.geoadaptnet` | L, B, C, P | none; adapter blocked | 2,195 |
| `architecture.geoadaptnet_fb` | L, B, C, P | none; adapter blocked | 2,195 |
| `architecture.geoadaptnet_fbsp` | L, B, C, P | none; adapter blocked | 2,195 |
| `architecture.cameo` | L, B, C, P | L | 2,195 |
| `architecture.hemiparity` | L, B, C, P | L | 2,195 |
| `architecture.parity_fuse` | L, B, C, P | L | 2,195 |
| `architecture.hemi_q_field` | B only | none; v2 protocol blocked | 45 |
| `architecture.cardinal_spline_dual_view` | C, P target cohorts | none; promotion blocked | 1,735 |
| `architecture.orbit_v1` | L, B, C, P | none; config/adapter blocked | 2,195 |
| `architecture.orbit_v2` | L, B, C, P | none; config/adapter blocked | 2,195 |
| `architecture.orbit_v3` | L, B, C, P | L | 2,195 |
| `architecture.orbit_v4` | L, B, C, P | none; config/adapter blocked | 2,195 |
| `architecture.orbit_v5` | L, B, C, P | none; config/adapter blocked | 2,195 |
| `reference.tcformer` | L, A, B, C, P | none; atomic runner blocked | 2,240 |
| `reference.fbcnet` | L, A, B, C, P | none; atomic runner blocked | 2,240 |
| `procedure.pretrained_cardinal_fbc` | C, P target cohorts | C, P | 1,735 |
| `procedure.scratch_cardinal_fbc` | C, P target cohorts | C, P | 1,735 |
| `procedure.scratch_fbmsnet_legacy_transfer` | C, P target cohorts | C, P | 1,735 |
| `procedure.pretrained_indexed_fbcnet_spherical_spline` | C, P target cohorts | C, P | 1,735 |
| `procedure.pretrained_cardinal_fbms` | C, P target cohorts | C, P | 1,735 |
| `procedure.scratch_cardinal_fbms_canonical_seeded` | C, P target cohorts | C, P | 1,735 |
| `procedure.scratch_cardinal_fbms_native_projected` | C, P target cohorts | C, P | 1,735 |
| `procedure.scratch_fbmsnet_native` | C, P target cohorts | C, P | 1,735 |
| `procedure.pretrained_indexed_fbmsnet_spherical_spline` | C, P target cohorts | C, P | 1,735 |
| `control.riemann` | L, A, B, C, P | L, A, B, C, P | 448 seedless records |
| `control.tangent_anchor` | L, A, B, C, P | L, A, B, C, P | 448 seedless records |
| `control.ea_fbcsp` | L, A, B, C, P | L, A, B, C, P | 448 seedless records |

The nine native-transfer procedures account for 15,615 records. The five
CardinalFBMS-family conditions already have a complete audited 8,675-record
grid. The four predecessor CardinalFBC-family conditions have an executable
per-record writer and fold-0/seed-7 development coverage (91 targets per
condition, 364 records total), but no equivalent score-blind full-grid
operations layer; completing that family would require 6,576 additional
records. Existing complete artifacts should be audited and ingested rather
than automatically recomputed.

The four executable local neural-procedure cells are already represented by
formal development artifacts: CAMEO, HemiParity, PARITY-Fuse, and ORBIT-v3.
Historical local deterministic-control artifacts also exist under
`pre-migration-package/results/local_exp4_allmodels_20260721`, but their five repeated seed
identities are not the new runner's record contract. Existing artifacts should
be validated and imported only when their protocol matches the intended table;
they are not a reason to spend compute on duplicate reruns.

## Callable and input contracts

The JSON manifest contains an individual descriptor for every non-common
stable ID. The grouped contracts are:

### GeoAdaptNet family

- `architecture.geoadaptnet` constructs `benchmark.research.model.GeoAdaptNet` and is
  optimized by `benchmark.research.engine.train_model`. It consumes four deterministic
  bandwise SPD covariances and has a hard binary main head.
- `architecture.geoadaptnet_fb` and
  `architecture.geoadaptnet_fbsp` use
  `benchmark.research.filterbank_net.FilterBankSPDClassifier.fit` on broadband trials.
  FB uses `reduced_dim=None`; FBSP uses `reduced_dim=8`.
- Harmonized v2 trials can safely supply these inputs, directly or through a
  deterministic derived covariance view. The blocker is not data
  incompatibility; it is the absence of a v2 atomic runner that pins
  constructor values, applies the current split IDs, resets/refits from the
  same initialization, predicts test once, and writes auditable records.

### CAMEO, parity, and ORBIT families

- CAMEO, HemiParity, PARITY-Fuse, and ORBIT accept paired original/reflected
  broadband trials and a paired four-band SPD/tangent representation.
- The exact local v2 path is
  `benchmark.local_outer_refit_benchmark.run_benchmark`. It owns cache loading,
  the fixed local split, source-only selection/reset/refit, resumable artifact
  publication, and calls the in-memory `run_one` only inside that boundary. It
  supports CAMEO, HemiParity, PARITY-Fuse, and ORBIT-v3 with frozen
  configuration-only files documented in
  [`LOCAL_PROCEDURE_CONFIGS.md`](LOCAL_PROCEDURE_CONFIGS.md); no historical
  result JSON is a runtime configuration dependency. Those four local cells
  are executable separate-procedure results.
- Historical Cho and BNCI runners use older loaders, preprocessing, and split
  contracts. Their scores cannot fill current matrix cells. Shape-general v2
  adapters are required for B, C, and P.
- ORBIT-v1/v2/v4/v5 share the same classifier class with ORBIT-v3. Their stable
  identities live in distinct historical configuration artifacts, so a
  stable-ID-to-config resolver must be frozen before any backfill.

### HemiQ-Field

`benchmark.research.hemi_q_field_net.HemiQFieldClassifier.fit` is valid only for the
three supplied bipolar BNCI2014-004 derivations. A new v2 adapter may apply a
frozen per-trial 8–30 Hz transform, but it must never treat the nominal
C3/Cz/C4 coordinates as independent point electrodes. The historical runner
uses S1–S4 development and S5–S9 one-time confirmation. A new all-nine-subject
run would be development-only and must not rewrite or supersede the historical
confirmation receipt.

### Author-faithful references

- `reference.tcformer` calls
  `benchmark.reference_training.fit_tcformer_reference` with the pinned official
  factory. It uses released Adam, warmup/cosine scheduling, one-for-one
  eight-segment reconstruction augmentation, and a fixed 1,000-epoch horizon.
  Both 256- and 320-sample v2 windows divide by eight.
- `reference.fbcnet` calls
  `benchmark.reference_training.fit_fbcnet_reference`. Raw v2 trials first pass
  through the pinned nine-band causal Chebyshev-II filter bank. Stage 1 uses
  released validation-inaccuracy patience; Stage 2 restores the best model and
  Adam state, optimizes train+validation, and follows the released stopping
  threshold.
- Both train-only callables exist and support variable class counts. What is
  missing is a cache/split-bound runner that constructs the right factory,
  owns one-shot prediction, captures parameters/timing/provenance, and resumes
  atomically. These scores belong in author-recipe columns, never as extra
  common architectures.

### Native transfer and CardinalSplineDualView

- `benchmark.native_transfer.run_record` resolves four explicit runtime
  conditions. The stable ID
  `procedure.scratch_fbmsnet_legacy_transfer` maps to the shorter runtime key
  `scratch_fbmsnet`.
- Five CardinalFBMS-family conditions have a completed historical grid of
  4,625 Cho records plus 4,050 PhysioNet records. Those results remain
  auditable, but their source-pinned v1 writer was retired at the namespace
  migration boundary. They are now `missing_adapter_or_blocker` until a new
  versioned writer and artifact schema are reviewed.
- Every native-transfer row requires `eeg-mi-cache-v2/native`, a byte-pinned
  source checkpoint, and the locked source/target subject boundary. A
  harmonized-profile cache is not a substitute: it has already discarded the
  native montage whose transfer behavior is the scientific object.
- `CardinalSplineDualViewNet` and
  `benchmark.dual_view_training.fit_dual_view_target` implement a leakage-safe
  in-memory prototype. It is blocked because it has no promoted source
  checkpoint, atomic target writer, auditor, or operations runner.

### Classical controls

The three estimator components support binary or multiclass labels:

- `benchmark.research.baselines.RiemannianTangentLogistic.fit`;
- `benchmark.research.tangent_anchor.TangentAnchorClassifier.fit`; and
- `benchmark.research.baselines.EAFilterBankCSP.fit`.

The cache/protocol-bound outer entrypoint is `benchmark.control_grid.main`. Its
immutable plan covers all five opened datasets, all declared subjects and
folds, binary and multiclass probabilities, source-only selection by
validation NLL, reset/refit on train+validation, and one-shot score-blind test
prediction. The runner creates exactly one seedless atomic execution and one
prediction record per control/dataset/subject/fold. It does not install
packages, reserve a GPU, or create five aliases for a deterministic result.

The manifest keeps three names separate:

- `registry_implementation` is the registry's identity text and may name a
  class or module rather than an executable boundary;
- `component_callable` is a source-resolved estimator/fitter symbol; and
- `adapter_callable` is the cache/protocol-bound outer executor when one
  exists.

Only the latter can make a cell `separate_eligible`. Common-grid rows leave
`component_callable` empty because their registry implementation text is not a
uniform callable; their actual outer executor is
`benchmark.full_grid.execute_benchmark_job`.

## Registry and implementation drift

The planner reports these issues rather than resolving them by name guessing:

1. GeoAdaptNet-FB and FBSP name the same class. Their identity depends on
   `reduced_dim=None` versus `reduced_dim=8`, which the registry implementation
   string does not bind.
2. ORBIT-v1 through ORBIT-v5 name one classifier. A stable-ID-specific,
   hash-pinned configuration resolver is mandatory.
3. Native-transfer registry implementation fields name modules, not the
   `run_record` callable and runtime condition. The matrix supplies this exact
   mapping.
4. `scratch_fbmsnet_legacy_transfer` maps to runtime condition
   `scratch_fbmsnet`.
5. `scratch_fbmsnet_legacy_transfer` and `scratch_fbmsnet_native` are not
   duplicate results: they are matched controls inside two different frozen
   source/target procedures.
6. CardinalSplineDualView's registry path names the model, whereas the
   leakage-safe fitting boundary is `fit_dual_view_target`; neither is a
   production artifact runner.
7. HemiQ's available runner is bound to a legacy cache and a historical
   confirmation contract.
8. Common TCFormer/FBCNet and their author-faithful rows are intentional
   architecture-family aliases under different training tracks. They must
   remain separate result columns.

Shared callable groups are emitted in `duplicate_callable_groups`; joining
outputs by display name, family name, implementation string, or runtime
condition is forbidden. The only result join key is the registry stable ID.

## Ordered execution backlog

The machine-readable backlog is ordered by expected scientific value and
implementation risk:

1. Execute and audit the existing score-blind deterministic-control v2 runner
   across all five datasets. Preserve its seedless one-record-per-subject/fold
   contract.
2. Build the two author-faithful atomic runners. TCFormer and FBCNet are the
   most important public recipe sensitivity checks.
3. Audit and ingest the already complete 8,675-record CardinalFBMS transfer
   grid into a separate transfer table.
4. Generalize the proven local CAMEO/HemiParity/PARITY-Fuse/ORBIT-v3 adapter to
   B, C, and P.
5. Add the development-only HemiQ v2 adapter for B.
6. Add a score-blind operations/audit layer for the four predecessor native
   CardinalFBC conditions only if the older transfer question remains useful.
7. Promote CardinalSplineDualView only after a prespecified development gate
   and checkpoint freeze.
8. Add GeoAdaptNet-family v2 reset/refit runners.
9. Backfill ORBIT-v1/v2/v4/v5 only if ORBIT-v3 remains competitive enough to
   justify 8,780 ablation jobs.

This order intentionally prioritizes interpretable controls and strong public
references over expensive historical ablation completion.
