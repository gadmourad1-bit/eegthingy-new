# IEEE-MI development ledger

This ledger separates exploratory model development from confirmation evidence.
All results listed here are development feedback. They must not be used for
confirmatory confidence intervals, superiority tests, or state-of-the-art
claims.

## Cohort firewall

Opened development cohorts:

- `local_exp4`: S1, S3--S8, S10.
- `bnci2014_001`: S1--S9.
- `bnci2014_004`: S1--S9.
- `cho2017`: S1--S15 in the current screens; S1--S52 are registered for
  development because they appeared in prior project experiments.
- `physionet_mi`: S1--S54 have been opened and cached for development-only
  screens and transfer experiments. They are not confirmation evidence.

Sealed until an architecture, training recipe, comparator family, seeds,
statistics, and source manifest are frozen:

- `physionet_mi`: S55--S109.
- `lee2019_mi`: S1--S54.
- `weibo2014`: S1--S10.
- `zhou2016`: S1--S4.

No sealed cohort had been accessed when this ledger was created on
2026-07-19.

## Evaluation contract used by the development screens

- Phase A fits the scaler on train only, optimizes on train only, and selects
  an epoch using validation cross-entropy.
- Phase B restores the exact seeded initialization, fits the scaler on
  train+validation, refits for `best_epoch + 1`, and evaluates test once.
- The common screen recipe is AdamW, 200 epochs maximum, patience 35, batch 64,
  seed 7, and the common source-only augmentation configuration in
  `training.py`.
- Subject balanced accuracies are averaged within each dataset. Dataset macro
  scores weight datasets equally.
- Single-seed and incomplete-fold results are screens, not formal evidence.

## Corrected v4 screen

The v4 screen was the first run after correcting the 31-anchor coordinate
frame. Its four-dataset equal-weight macro leaders were:

| Model | Macro balanced accuracy |
|---|---:|
| FBMSNet | 79.446% |
| CardinalFBC micro, 31 anchors | 79.058% |
| CardinalFBC, 31 anchors | 78.952% |
| FBCNet | 78.876% |

The proposed models were competitive but did not lead. The 31-anchor micro
continuation gained on BNCI2014-001 and local Exp4 but lost roughly 3.0 points
to FBMSNet on Cho S1--S15. The result did not justify freezing a candidate.

## v5 continuation screen

GPU source snapshot:
`/home/user/Desktop/eegthingy_ieee_arch_20260719`

Artifacts:

- `results/dev_bnci001_continuations_screen_v5.json`
- `results/dev_bnci004_continuations_screen_v5.json`
- `results/dev_cho15_continuations_screen_v5.json`
- `results/dev_local8_continuations_screen_v5.json`

Models include correlation, ordered-physical, compact-dynamics, micro, plain
CardinalFBC, FBCNet, and FBMSNet controls. The final equal-dataset macro was
79.635% for the numerically best 31-anchor correlation model versus 79.446%
for FBMSNet: a development difference of only +0.189 point. That model gained
1.76 points on BNCI2014-004, tied FBMSNet on local Exp4, lost 1.50 points on
Cho, and gained 0.50 point on BNCI2014-001. The best canonical continuation
macro was 78.828%, below the reference envelope. No continuation passed the
advancement gate, and no v5 model was selected.

## v6 bounded final architecture screen

GPU source snapshot:
`/home/user/Desktop/eegthingy_ieee_arch_v6_20260719`

The snapshot passed 89 tests before launch. It adds:

- `cardinal_fbms`: an exact seeded-function-preserving replacement of only
  FBMSNet's grouped spatial convolution with a 21-anchor RBF scalp-weight
  field. It has exact 21-channel parameter parity with FBMSNet.
- `cardinal_fbms_extended`: the 31-anchor anatomical-support ablation.
- fixed continuation feature scales 0.10 and 0.25 as a final bounded
  regularization screen. No further scale, rank, kernel, or branch sweep is
  permitted after v6.

Artifacts:

- `results/dev_bnci001_cardinal_fbms_screen_v6.json`
- `results/dev_bnci004_cardinal_fbms_screen_v6.json`
- `results/dev_cho15_cardinal_fbms_screen_v6.json`
- `results/dev_local8_cardinal_fbms_screen_v6.json`

The completed four-dataset screen did not select a model. The strongest
canonical candidate, `cardinal_fbc_compactdyn_scale025`, averaged 79.034%
balanced accuracy and trailed the per-dataset FBCNet/FBMSNet envelope by
0.412 point; its local Exp4 deficit was 2.708 points. `cardinal_fbms`
averaged 78.915% and trailed the envelope by 0.531 point, including a 3.125
point local Exp4 deficit. The numerically strongest v6 entry was the
non-selectable 31-anchor `cardinal_fbc_compactdyn_scale025_extended` ablation
at 79.866%, only 0.420 point above the envelope, below the predeclared 0.50
point advancement threshold. Therefore v6 closes the bounded architecture
sweep without a lead candidate; no additional scale, rank, kernel, or branch
search is authorized.

The SHA-256 digest of the last-completed BNCI2014-001 artifact is
`a23792abc489bda7e4d336d94070589a7f83a1453d4361d0a30b598860cccf8b`.

## Candidate advancement gate

For dataset `d`, the reference envelope is the better balanced accuracy of
FBCNet and FBMSNet. A canonical 21-anchor continuation may advance only if:

1. its equal-dataset mean improvement over the envelope is at least 0.50
   percentage point;
2. it is nonnegative on at least three of four datasets;
3. its worst dataset deficit is no worse than 1.00 point; and
4. it improves over plain CardinalFBC by at least 0.50 point in equal-dataset
   macro.

Differences below 0.25 point are treated as practical ties and resolved in
favor of the smaller model. The 31-anchor atlas is an ablation, not a separately
selectable lead, because current montages do not identify all of its added
degrees of freedom.

If no continuation passes, the next and final development hypothesis is
heterogeneous-montage pretraining of canonical CardinalFBC, followed by
source-only target personalization. Its locked development gate is documented
in the implementation and must be evaluated on Physionet S1--S54 and Cho
S16--S52 before any confirmation access.

## Native-montage pretraining pilot

The locked source corpus contains BNCI2014-001 S1--S9 (left/right rows only),
Cho2017 S1--S15, and local Exp4 S1, S3--S8, S10: 32 subjects and 7,592 binary
trials on 22-channel/320-sample, 64-channel/320-sample, and
15-channel/256-sample native montages. BNCI2014-004 is excluded. The raw-cache
corpus digest is
`e5ce98c41184ff42db6e9976d3369834ac83adb2c8dadacd560ba422ea2a360e`;
the deterministic subject partition digest is
`ca77219d9394bc8e790f933a5333171f27e89bbad8d86e3f2764a8222cbe7ac9`.

`native_pretrain_seed7_locked` is retained only as an invalidated engineering
artifact. Its first implementation fitted a label-free scaler independently
on every source subject, including the validation-role subjects. No target or
confirmation data were involved, but validation-distribution statistics could
enter preprocessing, so this checkpoint is forbidden from advancement or
comparison.

The corrected write-once artifact is
`results/native_pretrain_seed7_partition_scaled`. During Phase A it fits one
scaler per dataset using only the 25 training-role subjects; during Phase B it
fits new per-dataset scalers on all 32 authorized source subjects and regenerates
the checkpoint from the exact seeded initialization. Seed 7 selected 61 epochs
with equal-dataset validation CE 0.582139. Its checkpoint file SHA-256 is
`445495e4b78e5f444d7321d4252c68c85600a94332d4c60e93ade1a08b144249`;
its tensor-state SHA-256 is
`ff96c1058ca041e33b9fe90402aa447dddcd278256300f5eed743cca614f1b71`.
Both hashes were independently revalidated after publication. This remains a
development checkpoint, not evidence of target benefit.

The exact corrected checkpoint also passed a local Apple-silicon CPU inference
smoke with one model instance on both 21-channel/320-sample and
15-channel/256-sample batches. Four-trial batches produced finite `(4, 2)`
logits in approximately 2.10 ms and 1.88 ms respectively in that non-benchmark
smoke. These timings establish compatibility only; they are not efficiency
evidence.

## Known publication blockers

- Current test blocks have informed architecture development; all inference
  must therefore use untouched confirmation cohorts after a freeze.
- A manifest-gated, one-shot confirmation runner and prediction-before-scoring
  receipt do not yet exist.
- Full cohorts, all prescribed folds, and at least five frozen seeds are still
  required for formal results.
- Common-recipe comparisons must be complemented by author-faithful FBCNet and
  TCFormer reference recipes plus classical CSP/FBCSP/Riemannian controls.
- Same-montage accuracy does not establish montage transfer. Native/nested
  montages, channel loss, coordinate perturbation, interpolation, and density
  controls remain required.
- BNCI2014-004 uses supplied bipolar derivations and cannot substantiate a
  point-electrode continuity claim.
- GPU timing from concurrent screens is not an efficiency result.

## Native-transfer fold-0 screen after the GPU outage

The interrupted cache and transfer work was recovered without accepting any
partial result. PhysionetMI S1--S54 native caches were fully reloaded and
validated after rebuilding S24--S28 and S38--S41. The immutable CardinalFBC
checkpoint with file SHA-256
`445495e4b78e5f444d7321d4252c68c85600a94332d4c60e93ade1a08b144249`
then generated the complete development-only fold-0/seed-7 grids:

- Cho2017 S16--S52: 148/148 artifacts validated, zero missing;
- PhysionetMI S1--S54: 216/216 artifacts validated, zero missing.

The independent artifact audit checked file and array digests, exact schemas,
finite normalized probabilities, test split identities, cache/checkpoint/source
provenance, and cross-record consistency before any aggregate was computed.
Raw artifacts and the full descriptive report are under
`results/native_transfer_*_fold0_seed7` and
`results/NATIVE_TRANSFER_FOLD0_SEED7_SUMMARY.md`.

Equal-subject balanced accuracies were:

| Condition | Cho2017 | PhysionetMI | Equal-dataset macro |
|---|---:|---:|---:|
| pretrained CardinalFBC | 66.441% | 59.110% | 62.776% |
| pretrained indexed FBCNet + spline | 64.696% | 58.433% | 61.564% |
| scratch CardinalFBC | 62.928% | 52.927% | 57.927% |
| scratch FBMSNet | 67.083% | 55.308% | 61.195% |

The candidate improved over scratch CardinalFBC by 4.849 points in the
equal-dataset macro and did so on both datasets. The Physionet paired-subject
bootstrap lower bound was positive. Its per-dataset FBC/FBMS reference
envelope, however, was 67.083% on Cho and 58.433% on Physionet, or 62.758% in
the equal-dataset macro. The candidate exceeded that envelope by only 0.018
point, below the locked 0.500-point threshold. The fixed candidate is therefore
rejected as a paper lead.

The numeric decision rule is now executable and fail-closed in
`native_transfer_gate.py`. A fixed 50/50 direct/indexed probability fusion
reached 68.097% on Cho and 58.251% on Physionet, +0.416 point over the envelope
macro. Equal log-probability pooling produced the same binary decisions. This
second diagnostic also fails the 0.500-point gate, but the low cross-view
subject-score correlation motivates exactly one bounded shared-weight
dual-coordinate candidate; no fusion-weight sweep is authorized.

## CardinalFBMS heterogeneous-pretraining advancement screen

The next bounded candidate applies the heterogeneous native-montage protocol to
`CardinalFBMS`, the function-preserving continuous-coordinate replacement of
FBMSNet's grouped spatial convolution. The locked seed-7 checkpoint has 11,441
trainable parameters, selected 27 source epochs, and is pinned by file SHA-256
`7378450eee169a56f0bfc1bf6ad78ad8014b5c63ae998267b4cc0ddee0c712cb`.
Its source corpus and partition remain exactly
`e5ce98c41184ff42db6e9976d3369834ac83adb2c8dadacd560ba422ea2a360e`
and
`ca77219d9394bc8e790f933a5333171f27e89bbad8d86e3f2764a8222cbe7ac9`.

A real five-condition CUDA smoke audit caught and rejected an ambiguity between
raw float32 montage-coordinate hashes and the spline's normalized float64
working copies. The writer was corrected before the full screen, a direct
regression was added, and no pre-correction record entered the grid. The final
transfer writer is pinned by SHA-256
`b89b7af272937899f523e71d84e5a69247ceffa43dd6cb2ec7b44c6d8c47ba13`.

The exact Cho2017 S16--S52 and PhysionetMI S1--S54 fold-0/seed-7 screen then
completed 455/455 write-once records with no failures or missing artifacts.
The locked auditor reopened all records before the gate computed:

| Condition | Cho2017 | PhysionetMI | Equal-dataset macro |
|---|---:|---:|---:|
| pretrained CardinalFBMS | 69.043% | 61.558% | 65.300% |
| pretrained indexed FBMSNet + spline | 67.444% | 61.177% | 64.310% |
| canonical-seeded scratch CardinalFBMS | 61.768% | 54.828% | 58.298% |
| native-projected scratch CardinalFBMS | 61.081% | 54.795% | 57.938% |
| native scratch FBMSNet | 67.083% | 55.308% | 61.195% |

The candidate improved over canonical-seeded scratch by 7.002 points in the
equal-dataset macro and over the strongest per-dataset new/historical reference
envelope by 0.990 point. The Physionet paired-subject one-sided 95% bootstrap
lower bound was +3.919 points. Every locked gate component passed, including
positive scratch effects on both datasets and a +0.380-point worst-dataset
reference-envelope delta. The full decision artifact is
`results/native_fbms_transfer_fold0_seed7_gate.json` (SHA-256
`365531860ca4c558268fc987c247b29b62705cec618589c408edc407bd06a774`),
and the complete report is
`results/NATIVE_FBMS_TRANSFER_FOLD0_SEED7_SUMMARY.md`.

This pass authorizes only the prespecified all-fold, five-frozen-seed
development expansion. It is not confirmation evidence and does not authorize
a SOTA or publication-readiness claim.

## CardinalFBMS all-fold target-stage expansion (frozen before execution)

The advancement pass above authorizes one exact, development-only expansion:
Cho2017 S16--S52 on outer folds 0--4 and PhysionetMI S1--S54 on outer folds
0--2, with target-stage seeds 7, 17, 27, 37, and 47 and the same five locked
conditions. This is 4,625 Cho records plus 4,050 Physionet records, or 8,675
write-once records total. Directories use the collision-free template
`s{subject:03d}_f{fold}_seed{seed}_{condition}`. The already audited 455
fold-0/seed-7 records may be reused only by byte-identical, atomically published
copies after both their source roots and the new roots pass the same artifact
audits; the original roots remain unchanged.

All five target-stage seeds use the single immutable source-pretraining
checkpoint with source seed 7. Therefore this grid estimates outer-fold and
target-fitting variability, not source-pretraining variability. Independent
source-pretraining seed replication remains required before a publication
stability claim.

The x86/GPU writer's canonical-scratch initial-state SHA-256 values were
generated without reading outcomes and frozen before any new grid record:

- seed 7: `382d4e18385904b885e4140963705939b7d935ae95731ac402d3a447eabef507`;
- seed 17: `876e995189902d14e1f9424edeafd132be8185f0caf8a60ff6dd8304824f09a3`;
- seed 27: `76b023e24ac63260e3ae3fa1d22709787f7b52d78e496c0a761342aed82ab079`;
- seed 37: `486aa5fe51783925228d0fc4438ad3415ce82ca714aee23341bbc4c76f0c006b`;
- seed 47: `2a2943e16de9117d0f2bfbfe6ea957cb7be67710aae385f4b20a90b87b5b6af1`.

No aggregate may be opened until all 8,675 records pass the fail-closed audit.
For each dataset/subject/target-seed/condition, disjoint outer-test predictions
are concatenated before balanced accuracy is computed. Seed scores are then
averaged within subject, subjects are weighted equally within dataset, and the
two datasets are weighted equally. Folds and seeds are never inferential
units. The paired bootstrap resamples subjects within dataset and uses 200,000
replicates with fixed seed 20260720 for the candidate-versus-scratch analysis.

The primary full-grid architectural estimand is the equal-dataset paired
subject balanced-accuracy difference between pretrained CardinalFBMS and
`pretrained_indexed_fbmsnet_spherical_spline`. This is a checkpoint-matched
indexed projection control, not an author-faithful FBMSNet reproduction. A
second 200,000-replicate paired bootstrap uses fixed seed 20260721. Its
equal-dataset macro one-sided 95% lower bound must be greater than zero;
per-dataset point estimates and one-/two-sided intervals are reported but are
not separate gate criteria. The previously defined per-dataset oracle envelope
remains a secondary point-estimate check.

The frozen full-grid gate requires all of the following: at least +1.000 point
over canonical-seeded scratch in the equal-dataset macro; a positive scratch
delta on both datasets; positive one-sided 95% paired-subject scratch-bootstrap
lower bounds on both datasets and their equal-dataset macro; a positive
one-sided 95% equal-dataset macro lower bound over the checkpoint-matched
indexed+spline control; at least +0.500 point over the per-dataset reference
envelope in the equal-dataset macro; and no dataset more than 1.000 point below
its reference envelope. Passing remains development evidence only and cannot
authorize a SOTA or confirmatory claim.

The full-grid auditor and decision gate independently bind the exact source
checkpoint file/state, source corpus and subject partition, source initial
state, 11-file writer manifest, 91-subject target-cache manifest, and the
recorded RTX 5070/CUDA/Python/package environment. Those pins are part of the
gate artifact, not merely assumptions made by the operations runner.

Before any new full-grid outcome was produced, the analysis and operations
code was frozen by SHA-256:

- `native_fbms_full_grid_audit.py`:
  `114d783f114f50cf1896a834eccb56761e2269dd60268206c7b22a88c75bb673`;
- `native_fbms_full_grid_gate.py`:
  `995eeb56457c3f2af12e67fb51b4209a78f5707efc43c2d7ab51ff9f68aeca65`;
- `native_fbms_full_grid_ops.py`:
  `386d897db060d6e896d57e4ae91f253a1ec43f7694f4915b65b409c598634d33`.

The corresponding protocol and operations test files were
`e50a06742c637acc5a401c7ed3668c333e682d109e44c939466fdbfef716c80e`
and
`66493db957bdfe03c290752089ef729aead14df4bd73caded7be4774aba562f4`.
The complete local `ieee_mi/tests` suite passed 312 tests before remote sync.

### Score-blind nondefault-seed audit correction before resume

The first execution attempt at 2026-07-19 20:45 UTC completed the 455
byte-identical seed copies and allowed exactly four new Cho2017 S16, fold-0,
target-seed-17 writers to publish before the operations runner stopped:

- `pretrained_cardinal_fbms`;
- `pretrained_indexed_fbmsnet_spherical_spline`;
- `scratch_cardinal_fbms_canonical_seeded`;
- `scratch_cardinal_fbms_native_projected`.

All four writers returned zero, but the post-publication auditor rejected every
record with `target TrainConfig changed frozen field seed`. No prediction
array, aggregate, accuracy, probability, or model comparison was inspected;
only the structural journal error and process logs were read. No further job
was launched. The frozen transfer writer remained exactly
`b89b7af272937899f523e71d84e5a69247ceffa43dd6cb2ec7b44c6d8c47ba13`.

The cause was confined to the new full-grid audit wrapper: it reused the
fold-0/seed-7 screen semantic checker, whose internal default-config comparison
assumes seed 7, without first applying the same compatibility normalization
already used for the source-initial state. The correction independently
requires the artifact's real target seed to be an exact integer, a member of
7/17/27/37/47, and equal to the original record key. It then passes a shallow,
non-mutating seed-7 compatibility copy of both `protocol.train_config.seed` and
the private screen-checker key, composes the canonical-scratch source-state
compatibility copy, and restores the real `target_seed` into the full-grid
semantic identity for cross-record validation.

Before resume, the corrected files superseded only the earlier full-grid audit
and protocol-test hashes:

- `native_fbms_full_grid_audit.py`:
  `de6d02d45efb6004569984ee2a71a88309223aab69fca8aa8b0a195fc79f1f30`;
- `test_native_fbms_full_grid_protocol.py`:
  `73c3a091b14920eded0e728d0727122dd57e583921274c2fa04f077e70d5913c`.

The gate and operations hashes remained
`995eeb56457c3f2af12e67fb51b4209a78f5707efc43c2d7ab51ff9f68aeca65`
and
`386d897db060d6e896d57e4ae91f253a1ec43f7694f4915b65b409c598634d33`.
The corrected complete local suite passed 323 tests. The four published
records are retained only if the corrected remote auditor independently
accepts their bytes during preflight; otherwise resume remains fail-closed.

The corrected GPU suite subsequently passed 47 focused and 323 complete tests.
The score-blind preflight accepted exactly 459 records (the 455 byte-identical
screen copies plus the four records above), reported exactly 8,216 pending,
zero recovery transients, and decision `GO`. The detached four-worker CUDA
runner resumed at 2026-07-19 20:55:23 UTC as PID 186299. The first newly
completed seed-27 records were post-publication validated with `validated=true`
and the scheduler continued launching jobs. Aggregate analysis remains blocked
until all 8,675 records pass the final audit.

### Completed full grid and locked development decision

The runner was intentionally stopped at 2026-07-20 04:29 UTC before a normal
machine shutdown. At that pause, a score-blind preflight freshly validated
3,382 records, reported 5,293 pending, and classified exactly three interrupted
writer traces as safely recoverable. No aggregate was opened. After reboot, the
same recovery command revalidated the 3,382 published records, removed only
those three proven-stale writer traces, and resumed four fresh CUDA workers at
2026-07-20 16:55 UTC.

The operations journal emitted `run_complete` at 2026-07-20 23:24:34 UTC with
exactly 8,675 fully validated records, zero pending records, and zero recovery
transients. A separate post-completion operations preflight again returned
decision `GO`, 8,675 validated, zero pending, zero transients, and
`scores_computed=false`. Its archived JSON SHA-256 is
`c3d4bff53a4b0132812fc42661b122430ac8ebf9c95743cef185fd4975ab7ffb`.

Only after that final score-blind preflight, the still-frozen full-grid gate
(`995eeb56457c3f2af12e67fb51b4209a78f5707efc43c2d7ab51ff9f68aeca65`)
was invoked with only the two complete artifact roots. It atomically produced
`results/native_fbms_transfer_full_grid_gate.json` (4,921,892 bytes), SHA-256
`eab9f6ee1b548000eed823025330aa13c20616b8ed266cf5dd6b521a7ef80496`.
No sealed confirmation record was accessed.

The locked result was **PASS**. Equal-dataset balanced accuracy was 66.079% for
pretrained CardinalFBMS, 64.867% for the checkpoint-matched indexed+spline
control, and 59.004% for canonical-seeded scratch CardinalFBMS. Candidate minus
scratch was +7.074 points in the macro (+6.944 Cho2017, +7.205 PhysioNet), with
one-sided 95% paired-subject lower bounds of +5.794, +5.242, and +5.309 points
for the macro, Cho2017, and PhysioNet respectively.

The primary architectural delta versus the checkpoint-matched indexed+spline
control was +1.211 points in the equal-dataset macro (+1.482 Cho2017, +0.941
PhysioNet). Its prespecified macro one-sided 95% lower bound was +0.213 point,
so the primary criterion passed. Per-dataset primary intervals were descriptive
only; the PhysioNet one-sided lower bound was -0.431 point. The candidate also
exceeded the per-dataset reference envelope by +1.211 points in the macro, and
the worst dataset envelope delta was +0.941 point. All six frozen checks passed.

Post-outcome verification was kept separate from the frozen decision. A
standalone verifier (SHA-256
`918ee23a785b449688f39929f6fdde7778f81652622ed25bec8b5c1422dfbe18`)
imported neither the gate nor its aggregation helpers. It rehashed and reopened
all 8,675 raw OOF records, reconstructed every subject/seed balanced accuracy,
enforced exact fold-row coverage and cross-condition label identity,
reconciled every nested/top-level mean, reproduced both 200,000-replicate
bootstraps, and independently returned all six checks true. Its report SHA-256
is `7c1302f17f22a4e895c7e42257f5f0a8ab1badf0d59b93ea6b4ee314489f2562`.
A separate JSON-level audit completed 694 checks with no substantive
discrepancy; five quantiles varied across NumPy environments only at 1--22
float64 ULPs and did not affect a threshold.

The authoritative interpretation is in
`results/NATIVE_FBMS_FULL_GRID_SUMMARY.md`. This remains opened development
evidence. It does not establish SOTA, independent confirmation, source-seed
stability, or clinical effectiveness in disabled or paralyzed users.

## CHSD post-v3 disjoint-subject robustness amendment

On 2026-07-29, the fixed `chsdnet_conditioned_005` configuration entered the
transparent post-v3 exploratory amendment documented in
`docs/ROBUSTNESS_SCREEN.md`. This amendment did not rewrite the earlier
conditioned-v3 decision `kill_conditioned_family`. It used only opened
development participants that were absent from the v2/v3 screens: 5 local,
6 BNCI2014-001, 6 BNCI2014-004, 48 Cho2017, and 50 PhysioNet participants.
The fixed four-model, fold-0, seed-7 Cartesian grid therefore contained 460
records.

All four workers completed exactly 115 records, emitted one normal completion
summary, logged zero errors, and exited. The frozen status validator reported
460 complete, zero missing, and zero corrupt records. The immutable plan
SHA-256 was
`1e4009ae67e7b11ae60c76c5d110ec49a27246422d2617c9c818199ca799b9ae`;
the frozen runner and analyzer SHA-256 values were
`93bc0c7828b78688620316f1612708a1a8fdb2f369612ea5da573931cc47b4d4`
and
`8cf2e388401679bac662bc5a470807edb99b9a805b93f8c6f29145f4ff02f0fa`.

Only after exact completion was validated did the analyzer open the aggregate
records. Equal-dataset balanced accuracy was:

| Rank | Model | Balanced accuracy |
|---:|---|---:|
| 1 | CardinalFBC micro extended | 74.087% |
| 2 | FBCNet | 73.628% |
| 3 | CHSD-conditioned 0.05 | 73.543% |
| 4 | TCFormer | 72.862% |

The frozen decision was **`stop_chsd_promotion`**. The candidate was within
0.544 percentage point of the strongest reference and exceeded TCFormer by
0.680 point, but it did not satisfy every paired robustness clause:

- versus CardinalFBC micro extended, it was nonnegative on only 1/5 datasets
  and had a 13.91% strict paired-subject win rate;
- versus FBCNet, it was nonnegative on 3/5 datasets but had only a 38.26%
  strict paired-subject win rate; and
- versus TCFormer, it passed the individual comparison with 4/5 nonnegative
  datasets and a 47.83% strict paired-subject win rate.

The aggregate analysis artifact SHA-256 is
`ba834b159dae53781e44f19682a6c032d4674f46fb5b0f5c752fedd9dc84e943`.
This is a retained negative development result. CHSD-conditioned must not be
inserted into the formal common grid, described as selected, or used for a
state-of-the-art, confirmation, or clinical claim.
