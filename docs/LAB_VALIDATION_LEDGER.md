# Lab validation ledger

## Scope

This ledger records read-only or scratch-only verification performed on the
shared lab workstation for the clean EEG motor-imagery benchmark. It is not a
benchmark result table and contains no claim of model superiority. Formal
training may begin only after the relevant runner also passes an independent
no-edit audit.

All commands used the project-owned UV virtual environment at
`/home/hanafy/scratchpad/eeg_novel_20260729/.venv-uv`, the project-owned UV
cache at `/home/hanafy/scratchpad/eeg_novel_20260729/uv-cache`, and source
copies under `/home/hanafy/scratchpad/eeg_novel_20260729/source`. No system
Python, global package set, NVIDIA driver, or another user's environment was
modified.

## Workstation identity

Observed on 2026-07-29:

| Component | Value |
|---|---|
| Host access | SSH alias `larri-ai` |
| GPUs | 4 x NVIDIA RTX A5000, 24,564 MiB reported per GPU |
| Torch | `2.6.0+cu124` |
| Torch CUDA availability | `True` |
| CUDA-visible device count | 4 |
| Determinism variable | `CUBLAS_WORKSPACE_CONFIG=:4096:8` for CUDA tests |
| Python environment | UV-created project virtual environment |
| Worker policy | At most three project GPU workers; GPU 3 remains available |
| Disk low-water mark | 50 GiB; formal writers fail closed below this value |

The read-only safety check immediately before the runner replays showed no
training process owned by the project, zero percent GPU utilization on all
four devices, and 77 GiB free on the containing filesystem. Scratch usage was
7.7 GiB.

## Shared GPU lease helper

Frozen source identities:

| Path | SHA-256 |
|---|---|
| `ieee_mi/project_gpu_leases.py` | `b0d12a8689430c00984bc4fe95800ee626cb37f20434b578591b3d26f0d91908` |
| `ieee_mi/tests/test_project_gpu_leases.py` | `43a509eb90907cbb1b07b691caad2787de50e2f31c6ae6e4a0afed2f0fb31b26` |

The helper's independent lab suite passed 18 of 18 cases. The tests include
four processes contending atomically for three global slots and a held
registry guard that prevents a concurrent release. The repeated
`multiprocessing` fork deprecation warning is a Python warning from the test
harness; it did not change any assertion or test outcome.

## Procedure Grid replay

Frozen identities:

| Path | SHA-256 |
|---|---|
| `ieee_mi/procedure_grid.py` | `256e56b93c9c261c6f9f9954e3e3599f3dd8b42f4d729c7e678b5d69e8de6205` |
| `ieee_mi/procedure_grid_analysis.py` | `36a45b85c8d4f6060d2554363fcac5541d97b9ee5698b4702c61173ffde916cc` |
| `ieee_mi/tests/test_procedure_grid.py` | `6927159622148226867485d8fa231731c7b207d838bf23fab70a1b54ad3d92c4` |
| `ieee_mi/tests/test_procedure_grid_analysis.py` | `9c5b0e2d8582f23192158542ba493821dbf13d87365a357462f35a69a4e621e5` |

The first independent no-edit audit rejected an earlier analysis hash after
proving that an unmanifested read-only file and a CSV replaced after its
intended hash had been computed could both be published successfully. The
repair binds the staging-directory descriptor through publication, requires
the exact eight-file roster, hashes the actual descriptor-read bytes, seals
and fsyncs before rename, and repeats the checks after rename.

The repaired frozen files were copied to the scratch source tree and their
hashes were recomputed there before testing. The combined Procedure Grid,
analysis, and shared-lease closure passed 76 of 76 lab tests in 13.79 seconds.
This includes the two Torch-dependent cases that could not execute in the
CPU-only Mac harness and three new publication fault-injection regressions. No
training job or formal plan was created. A final no-edit re-audit of the
repair replayed both original attacks: each now raises, leaves the requested
final output absent, and moves the bound failed stage to quarantine. The
auditor approved this frozen Procedure Grid boundary.

A later cross-runner audit identified a separate ambiguity not exercised by
that review: `set(archive.files)` does not reject duplicate ZIP members in an
NPZ file. The same audit established that completion directories, not only
their leaves, must be sealed read-only before rename and validated as such
afterward.

The new runner hash enforces exact raw `ZipInfo` and NumPy member order and
cardinality, moves a writable build directory adjacent to the destination,
fsyncs it, seals the held inode to mode 0555, fsyncs again, and only then
performs the same-parent no-replace rename. Final validation binds that inode,
requires the directory to remain nonwritable, and retains working quarantine
for post-guard failure. The updated combined Procedure, analysis, and lease
suite passed 79 of 79 tests in the lab UV/CUDA environment. Independent
no-edit review of these two additions and replay against the final shared
cache-reader hash remain pending.

The later expected Procedure freeze retained the identities in the table above
and expanded to an exact 8,780-job roster, but its canonical lab mirror was
found stale before replay: the runner and runner-test bytes differed and
`docs/PROCEDURE_GRID.md` was absent. Independent static review of the expected
local bytes also **rejected launch**. Completion and analysis read related
children sequentially instead of from one coherent descriptor-held package;
record, claim, and analysis publication set ownership only after rename and
post-rename fsync returned; and release-failure cleanup could quarantine a
replacement completion without an expected-inode guard. The tests lacked
package/leaf A-to-B-to-A, move-then-raise, post-move fsync, claim rollback,
and replacement-survival attacks. No Procedure plan, training, prediction,
or score exists. The next candidate must repair those exact boundaries in an
isolated builder copy, pass a new lab replay, then receive a different no-edit
verdict before the canonical mirror is changed.

The isolated repair candidate now proposes:

| Path | SHA-256 |
|---|---|
| `ieee_mi/procedure_grid.py` | `2ef83e67b025936a46bf3a9c0f668c6f772dce01be8c9834b0971632e60c1617` |
| `ieee_mi/procedure_grid_analysis.py` | `9ab89f07b232290bc646b156f66c9e8c8275892b744a0e165ad79f0892f0492e` |
| `ieee_mi/tests/test_procedure_grid.py` | `23d929b9ff31507523c9b864ab9ff9e091b62c16c259d8210dbaf36cc5836bd0` |
| `ieee_mi/tests/test_procedure_grid_analysis.py` | `8bc2b534d8cc96346f293e426e4425f457d6df60a27eb9648dde3805e1c243c6` |
| `docs/PROCEDURE_GRID.md` | `c781477bc149529de2685bb565f0af13da25712845f36afa4eafe7f3b9ce2864` |

Its exact private offline lab closure passed 95 tests. Runner completions use
one descriptor-held three-file snapshot; analysis inputs use one
descriptor-held eight-file snapshot; publication and claim state is tracked
by exact inode before any post-move operation; and cleanup preserves unrelated
replacement entries after move or fsync failures. The builder tree is clean,
but these bytes have not been copied into the stale canonical mirror. A
different no-edit audit remains mandatory. No formal artifact or score was
created.

That fresh no-edit audit subsequently **rejected** the candidate. It found
remaining parent/ancestor A-to-B-to-A gaps across atomic publication,
completion/analyzer snapshots, fencing, and claims. The 77-test focused replay
passed, but the missing ancestor-history guarantees are release blockers. No
Procedure plan or score was created; another bounded repair and audit are
required.

## GeoAdapt v2 replay

Frozen identities:

| Path | SHA-256 |
|---|---|
| `ieee_mi/geoadapt_v2.py` | `cd13a2d6a1a4e96234f6389d99e71b46f2b209af05b6edaff16e163feb873943` |
| `ieee_mi/geoadapt_v2_analysis.py` | `44164be459fdc106b022f78b6f414c9449537639ed258a668958777c9de73b88` |
| `ieee_mi/tests/test_geoadapt_v2.py` | `a7d1de57bbc45bbc4edebbe49ab9d13fca5f2762cd47e6f4e0ec9aee1c2f9b7b` |
| `ieee_mi/data.py` | `280c6bb51d02ff8d57fefa77abd5e4e11960593558dbd13867134fe0df3c9fac` |
| `ieee_mi/tests/test_data.py` | `68860a599e04176a7b5aeaae586d50879d249f21867e95e8086047d28932bb83` |

An earlier CPU-only test pass had hidden a Linux procfs incompatibility:
`/proc/<pid>/stat` and the boot-ID pseudo-file may report a zero `st_size`.
The frozen replacement uses a bounded, descriptor-anchored,
`O_NOFOLLOW | O_NONBLOCK` reader, validates the live PID and field 22 from
`/proc/<pid>/stat`, and validates the exact boot-ID syntax. Its CUDA test also
sets the production-required CUBLAS determinism variable explicitly.

After independently copying and hashing the corrected sources, the combined
GeoAdapt, cache-reader, and shared-lease closure passed 128 of 128 lab tests in
12.22 seconds with CUDA visible. No training job or formal plan was created.

The subsequent independent no-edit audit nevertheless rejected this candidate
for four untested release blockers: direct writes to final authority filenames
were not crash-atomic; exact plan validation accepted several wrong JSON
scalar types; GPU receipts were validated against a persisted path rather than
the actual current run root; and analysis lacked a post-commit
gate/publication-lock check with quarantine on release failure. These hashes
therefore remain evidence for the procfs repair but are not approved launch
hashes. A corrected candidate, lab replay, and new no-edit audit are pending.

A consolidated core repair subsequently froze the following candidate:

| Path | SHA-256 |
|---|---|
| `ieee_mi/geoadapt_v2.py` | `b61c0bb79330f56304e5953c6106e0626d8d7e8dd789b8b0c6c4d567677d9fef` |
| `ieee_mi/geoadapt_v2_analysis.py` | `7f370f42d8fd6b1476b7d2a472e89eff333fb26e8dde5712b5815027d2232203` |
| `ieee_mi/tests/test_geoadapt_v2.py` | `b95133a49f38fb2c76db66f4f1bab35f7dc29b29b121447624b79edbd1cebc5a` |

Its 78-test lab closure passed and covered strict scalar types, atomic
authority writes, run-root-bound GPU receipts, post-commit lock checks,
exact-order NPZ validation, and sealed record directories. A second no-edit
audit then found a narrower analysis race: metrics could be computed from one
descriptor snapshot while a later path reread supplied the hashes stored in
the input ledger. The candidate was therefore rejected before launch.

The same-snapshot repair now freezes:

| Path | SHA-256 |
|---|---|
| `ieee_mi/geoadapt_v2.py` | `b61c0bb79330f56304e5953c6106e0626d8d7e8dd789b8b0c6c4d567677d9fef` |
| `ieee_mi/geoadapt_v2_analysis.py` | `a8d38aa80dd13a0bc479e45e99ca16736d3bd8c577ddfd39aa198cd189785fd9` |
| `ieee_mi/tests/test_geoadapt_v2.py` | `f87524f403ae06682bfa952cb4d450af4e6c3da9be59841707584c40fa0c5c98` |
| `docs/GEOADAPT_V2_PROCEDURE.md` | `b35fb3178a670d437f8de35da7fd5e4e58ac6240059a1014f57559afcb24d060` |

Each completion is now captured from one held directory descriptor. The exact
bytes used to decode rows and probabilities also supply the artifact hashes
stored in the ledger. Live files are used only to verify that captured
identity before and after scoring and publication. A fault test replaces a
perfect-score prediction package with a valid zero-score package after metric
calculation; publication now aborts and leaves no output. The local closure
passed 96 tests with one expected platform skip, and the lab UV/CUDA closure
passed 97 of 97 tests in 19.00 seconds. No plan, job, prediction, or score was
created. Final approval remains conditional on replay against the ultimately
frozen shared `data.py` and a separate no-edit review.

That final no-edit review matched all four local/lab hashes and replayed 79
focused and 145 wider tests successfully, but **rejected** the candidate on
publication boundaries. A disposable coherent A-to-B-to-A attack was accepted
because completion, record, and prediction leaves were opened and closed
sequentially; the analysis retained detached package-A bytes while valid
package B was canonical. A move-then-raise fault also left a valid canonical
record after publication reported failure because the committed flag was set
only after rename returned. The same ordering is present in analysis
publication. Per-commit source/environment/cache/split rebinding and
expected-inode recovery quarantine are also incomplete.

The 6,495-job procedure grid itself remains correct. A coherent-snapshot and
post-move repair, lab replay, and another independent audit are required before
launch. No GeoAdapt plan or score was created.

The fourth repair is approved at:

| Path | SHA-256 |
|---|---|
| `ieee_mi/geoadapt_v2.py` | `1dcb104184370e8083b1c116c9124375f57949cdca2ff06ac617d57a888d3564` |
| `ieee_mi/geoadapt_v2_analysis.py` | `e7ab42a5ea7bda56ecd9e913b2760e5eeb381866f6d1dbef9e67a32f72f26eec` |
| `ieee_mi/tests/test_geoadapt_v2.py` | `0735a5e9cf72306197b87265adf598cdeb65200a22e6e70a67f4fad2e665b977` |
| `docs/GEOADAPT_V2_PROCEDURE.md` | `e0481faabe66dcba97ffebd573b50dde13fb08daa048d9e8b2ee13ff36069374` |

The exact private offline lab closure passed 156 tests. Fresh no-edit review
confirmed one descriptor-held completion snapshot supplies validation,
hashing, decoding, and analysis bytes; exact `(device,inode)` identity is the
sole publication state; move-then-raise cleanup performs post-commit authority
rebinding before exact-inode quarantine; lock-A/lock-B and claim-record
replacements preserve the replacement and its partial; and all source,
registry, UV, cache, split, device, and lease bindings close around
publication. The exact 6,495 jobs and per-model counts
2,150/2,195/2,150 remain unchanged.

Verdict: **APPROVE_FOR_LAUNCH for the frozen GeoAdapt v2 track**. This is an
artifact/protocol integrity approval, not a performance, novelty, SOTA, or
clinical claim. No GeoAdapt plan or score existed when approval was issued.

## Prior CardinalFBMS evidence

The prior CardinalFBMS result tree was copied read-only from the earlier GPU
box and independently verified on the lab workstation. The source and
destination canonical tree hashes matched:

`958801527c039c734b07b2b9a1593df0b0e64c206f9485335c199eb5508f5840`

The standalone verifier checked all 8,675 raw records and reproduced all six
gate checks. Its report SHA-256 is:

`7c1302f17f22a4e895c7e42257f5f0a8ab1badf0d59b93ea6b4ee314489f2562`

The full interpretation and aggregate values are recorded separately in
`docs/NATIVE_FBMS_LAB_REVERIFICATION.md`.

## Common Grid replay

Frozen identities:

| Path | SHA-256 |
|---|---|
| `ieee_mi/full_grid.py` | `a94c9bfec8ff35687615d0f4a45c61757e9bcfba926d5d68362c9a15ffd69897` |
| `ieee_mi/full_grid_analysis.py` | `5017a51af2bd0a86a2ae1e62599b8d70d0a288146b355bd2b4b3484e42467711` |
| `ieee_mi/baselines.py` | `a31f0891484f3db3e23b0b11727045b98cc0d2ae2099b7499d1e6ed6858ead88` |
| `ieee_mi/tcformer_source.py` | `140960093538f163083bbe84a76f7fe9681b40c0f4a1be74246e964af4e2534e` |
| `ieee_mi/tests/test_full_grid.py` | `b48fa2fa2f3974da3c13aac9e45610d23256d8d94730d974454740a4db989cd9` |
| `ieee_mi/tests/test_full_grid_analysis.py` | `41596767779dab82b0dfd83545131712c630cd810443f4867771e86958273271` |
| `ieee_mi/tests/test_tcformer_source.py` | `f125f87d2781bf1e85a7c7fb888f3ade97f0550116035418385aa77cd6c9a99b` |

The formal preflight is now an immutable, mandatory two-file attestation bound
to the plan, source, environment, cache, and guarded GPU receipt. It contains
exactly 172 ordered architecture/shape checks, survives a report-first power
cut without replacement, and is required by workers, claims, commits, audit,
and analysis.

The scratch source initially contained a full TCFormer checkout but not the
compact release `SOURCE.json`. The repository's audited, offline vendoring
tool created a new scratch-only compact snapshot from upstream commit
`74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`. Its manifest SHA-256 is
`06e04b81c05930721fd049b12bbff61e277438db2a9f94d9552864c3af4173ea`;
all five pinned runtime file sizes and hashes validated.

With that exact snapshot selected, the full Common Grid, analysis, TCFormer
provenance, baseline, cache-reader, and shared-lease closure passed 164 of 164
lab tests in 15.56 seconds. No benchmark plan or training job was created.

The subsequent independent no-edit audit rejected these hashes. It found:

- post-yield GPU-lease guard failures could leave claims, renamed records, or
  preflight attestations usable;
- duplicate ZIP/NPZ members were reduced to a set and accepted;
- valid power-cut quarantine evidence permanently prevented formal completion;
- tombstone removal and worker-log creation had ancestor-swap windows;
- record directories remained writable after publication;
- the direct formal preflight API did not prove that the leased GPU was the
  device on which CUDA checks executed;
- analysis scalar schemas still accepted booleans through Python coercion;
- post-rename analysis failures could leave the destination visible; and
- repeated preflight-stage power cuts could make recovery impossible.

These identities therefore remain lab-replay evidence but are not approved
launch hashes. A consolidated repair, new lab replay, and new no-edit audit
are pending.

The consolidated replacement candidate was then frozen with these critical
identities:

| Path | SHA-256 |
|---|---|
| `ieee_mi/data.py` | `6dfc3be77f315c60acb1aa26c2e2ca70a11e9480390646e351f55429100ad4aa` |
| `ieee_mi/full_grid.py` | `121e26bdfd27af633a87b7c664ee13ae592fff770a36a2f2e2282b450ef70e92` |
| `ieee_mi/full_grid_analysis.py` | `e6e81194ae92816b55196d930a72c70dd5635bbac429cea4c5c11200f3be5b5b` |
| `ieee_mi/tests/test_data.py` | `978d812c55ec0a6618fe75182cf36649acb10a6247db7882a6cdfb9205ead180` |
| `ieee_mi/tests/test_full_grid.py` | `59d0bc9b4d5339034f6f92e883506ab49d7f677b412a0e48112b3f4986da570c` |
| `ieee_mi/tests/test_full_grid_analysis.py` | `88dcae299f2d7afa9732e3a621ef6996b99a1ceb713dddf76bb9b0ff3263efdb` |

Local focused and wider closures passed 160 and 208 tests respectively. The
files were copied to the authoritative lab scratch source and all six hashes
were recomputed there before replay. The lab result was **191 passed, 17
failed**. Every failure was in publication/quarantine behavior on the actual
Linux filesystem: a continuously sealed mode-0555 directory can be renamed
within one parent, but a cross-parent rename returns `EACCES`. A direct
disposable probe confirmed that distinction. The record stage moves from
`partials` to `records`, and its failure cleanup moves the same sealed inode to
quarantine, so neither action could complete.

The independent no-edit audit rejected this candidate and demonstrated five
additional boundary faults:

- Common analysis publication and invalidation contain equivalent sealed
  directory rename assumptions.
- A quarantined directory could be swapped after its descriptor was opened;
  the forensic ledger then described the detached old inode while its recorded
  path resolved to a replacement.
- Three independent path hashes did not form one live completion snapshot. A
  fault alternated a valid package into place for each read, restored another
  package afterward, and the ledger verifier accepted it.
- A failure after exclusive claim publication but before constructing the
  returned `Claim` handle left a live authoritative claim that stranded the
  job.
- Analysis set its `published` identity only after rename returned, so a
  move-then-raise fault could bypass output invalidation.

The auditor otherwise approved the exact 43-model/96,320-job cardinality,
ordered 172-check CUDA preflight, physical leased-GPU binding, strict scalar
schemas, exact NPZ contract, anchored tombstones/logs, repeated preflight
recovery, sealed record directories, and ordinary post-yield invalidation
logic. A new candidate must repair the six demonstrated blockers, pass the
complete lab closure, and receive another no-edit audit before any Common plan
may be created.

The NFS/audit repair candidate was subsequently frozen with these identities:

| Path | SHA-256 |
|---|---|
| `ieee_mi/full_grid.py` | `67e25f47bafeb29d57ded5cc0fa6581dcef12cfccfa1d20c13a90649c304ff36` |
| `ieee_mi/full_grid_analysis.py` | `bcacbf8c8b36e0ea353e690bece38eabc8712c288b0d553a940dbf68d8a22e40` |
| `ieee_mi/tests/test_full_grid.py` | `54604c287b93520f4e0235e96d47ef91c48929cbf114ac66a6bb6db8187565fa` |
| `ieee_mi/tests/test_full_grid_analysis.py` | `6099bcb748cc5a51a67372f0b6d484d664744eb27d8dc09cf840f02796573548` |
| `docs/FULL_GRID_ANALYSIS.md` | `dd0699ee9588eac6c79e6c89473262ce816623c176a0d9fcc88a3f859613dead` |

The repair adds an `EACCES`/`EPERM`-only held-descriptor transition for the
lab filesystem, followed by unconditional resealing, directory synchronization,
and exact flat-tree/inode verification. It also binds quarantine receipts to
the moved inode, closes move-then-failure publication windows, constructs each
analysis-ledger row from one descriptor-held completion snapshot, re-digests
forensic paths at the end of the audit, and cleans an exclusively published
claim if handle construction fails.

Those exact bytes were copied to the authoritative lab scratch source and
re-hashed there. The complete Common runner, analysis, TCFormer provenance,
baseline, cache-reader, independent-verifier, and shared-lease closure passed
**223 of 223 tests in 29.05 seconds**. The output contained the two expected
duplicate-ZIP warnings and 52 numerical-model warnings; it contained no test
failure. No benchmark plan, training job, prediction artifact, metric, or
aggregate was created. A fresh independent no-edit re-audit is in progress, so
this remains a tested candidate rather than an approved launch identity.

That no-edit audit subsequently rejected the candidate despite reproducing the
223-test pass. It demonstrated three uncovered concurrency faults:

- a valid three-file result package could be substituted coherently while its
  children were read sequentially and then replaced again, causing validation
  to return bytes detached from the live package;
- a concurrent preflight loser could quarantine a legitimate winner because
  the initial directory snapshot was taken before the global publication
  guard and cleanup did not prove inode ownership; and
- stale-claim recovery validated one inode but invoked quarantine without that
  expected identity, allowing a replacement live claim to be moved instead.

The record-tree fallback itself passed real lab `EACCES` probes for both
three-file record and 17-file analysis trees. A third repair must hold every
child descriptor in one snapshot, bind cleanup to exact inodes, and make
preflight races non-destructive before another no-edit review.

The third repair candidate is frozen at:

| Path | SHA-256 |
|---|---|
| `ieee_mi/full_grid.py` | `cf84f69945a0de7466a0e56459323758d4a7d37f05a1cc6bcec9d8258a71e3ac` |
| `ieee_mi/full_grid_analysis.py` | `bcacbf8c8b36e0ea353e690bece38eabc8712c288b0d553a940dbf68d8a22e40` |
| `ieee_mi/tests/test_full_grid.py` | `203ae70f86e0c89f12cac367e4b66ad6dc696d54cd6933ab3966e43eb79e8fbb` |
| `ieee_mi/tests/test_full_grid_analysis.py` | `772d77df0f977ae2d7381d1c9e5646ad39207b57e502f1ce5c2c372a14f9abcf` |
| `docs/FULL_GRID_ANALYSIS.md` | `414379a19ca653e63dacec40cdf6edf6b98849c083da066abfc1a18a95cf253b` |

Completion validation now holds all three child descriptors simultaneously,
rebinds each descriptor to its exact live name, and checks the complete job
directory fingerprint. Preflight snapshots and loser cleanup now occur under
the project guard and accept a valid concurrent winner. Every recognized
claim, stage, partial, and corrupt-record recovery passes the exact validated
inode into quarantine. Coherent package substitution, directory metadata
mutation, concurrent winner preservation, and replacement-survival attacks
were added.

The local focused and integrated closures passed 136 and 204 tests. The exact
offline lab UV closure passed **232 tests with 54 expected warnings in 32.89
seconds**. No plan, training job, prediction, or score was created.

A fresh no-edit audit independently matched all five local/lab hashes,
replayed the correctly configured offline UV closure at 232 of 232 tests,
passed the focused runner/analyzer at 136 of 136, and passed all 18 shared
lease tests. Its additional coherent A-to-B-to-A package attack was rejected,
and real lab cross-parent fallbacks preserved the exact inode and sealed modes
for both 3-file records and 17-file analyses. It confirmed the exact
43-by-448-by-5 = 96,320 jobs and 43-by-4 = 172 preflight checks.

Verdict: **APPROVE_FOR_LAUNCH for the frozen 43-model Common core**. Launch
must explicitly select the compact audited TCFormer snapshot rather than the
unmanifested full checkout. It still waits for the Gauge decision: a Gauge
promotion would require a 44-model roster/cardinality change and another
targeted review. This is runner-integrity approval, not a performance or
state-of-the-art claim.

Gauge later failed its frozen Gate 1, so the Common roster remained exactly 43
models. Before initialization, one further bounded compatibility repair
allowed the exact, public cache label-provenance literal only at its documented
cache-identity path while leaving the global outcome-alias scanner unchanged:

| Path | SHA-256 |
|---|---|
| `ieee_mi/full_grid.py` | `d2e0f53ab0010609c8c9d576225caa570e4d8c9768c1b94a16af6c8b4dfb4b5b` |
| `ieee_mi/tests/test_full_grid.py` | `39d58d27776bf8743f2d8bc094b27588fb3b4ad1aa810365dfeed61e9400bcfc` |

Independent review passed 253 tests, reconstructed all 43 models, 132 caches,
448 splits, and 96,320 unique jobs, and rejected 36 of 36 cache-identity
mutations. The other approved Common identities remained unchanged.

A dedicated scratch experiment project was then assembled around the
project-private UV environment. Its reproducibility boundary included:

| Path | SHA-256 |
|---|---|
| `pyproject.toml` | `1fdb188881e3554f69c0009337a1848592042e10cd6104866dd91a71e589baa0` |
| `.python-version` | `aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b` |
| `uv.lock` | `fceb6d13f67d4ade2a8aab9da0b0dd4db65527c59eb65bd41e347e4233d2f576` |
| `third_party/TCFormer/SOURCE.json` | `06e04b81c05930721fd049b12bbff61e277438db2a9f94d9552864c3af4173ea` |

The Linux x86-64 lock contains 109 packages and pins the CUDA 12.4 build of
Torch 2.6.0. Offline lock validation and `uv pip check` passed; all 111
packages installed in the private venv were compatible. After adding the
canonical, read-only TCFormer provenance helper, the sealed project passed
284 of 284 tests.

The first formal preflight attempt at
`runs/common_grid_43_formal_v1` wrote only the immutable score-blind plan and
its sidecar, then stopped before acquiring a GPU lease or executing a CUDA
check. The plan's internal SHA-256 is
`eb3d930523764761e0069aa0ce72d242c61d0e266b6efef0c02122f45f373553`;
the raw canonical `plan.json` SHA-256 is
`c9fbf04b64228e63bf10c17bc89e944f6ae72b1dd958a655541961a68280fd7a`.
The failure exposed a self-inconsistent validator: canonical JSON sorts
mapping keys, while the validator required the decoded `datasets` mapping
insertion order to equal the separate scientific `dataset_order`. All five
datasets, 43 models, 132 caches, 448 splits, and 96,320 jobs were otherwise
intact. No preflight report, training, prediction, metric, or score was
created, and this v1 root is preserved as failure evidence.

A disposable v2 repair makes `dataset_order` the sole ordering authority,
requires its entries to be nonempty strings, and requires exact set membership
from the `datasets` mapping. It does not sort or otherwise change the
scientific order:

| Path | SHA-256 |
|---|---|
| `ieee_mi/full_grid.py` | `c99bd94fbfd031dd5334f4241bcde70499459cc55c67b3d46e7a5989c1d8e2e0` |
| `ieee_mi/tests/test_full_grid.py` | `283927e793412eedd0839255e3c759edeac1fba17fd4b6492e4372e23d3c3495` |

The repaired scratch closure passed 293 of 293 tests, including nonlexical
canonical round trips, arbitrary mapping insertion order, exact job-order
preservation, and missing, extra, duplicate, unknown, non-string, empty, and
unhashable order attacks. It also loaded the untouched failed v1 plan and
enumerated all 96,320 jobs. Two independent no-edit audits matched the exact
hashes, repeated the 293-test closure and real-plan reconstruction, and
approved creation of a separately sealed v2 project.

The v2 preflight executed the 172 score-blind CUDA checks, but correctly
refused to publish an attestation because the environment identity observed
after model construction differed from the plan. Its plan SHA-256 is
`ffb4816d03b9ffd1eb9fc40977940d2bfab9db1902eb41c73893494932ed0510`;
the raw canonical `plan.json` SHA-256 is
`42d27d349a101037c987585f9c011b792ad2d22fff6bc9a0443e3658d2145203`.
The v2 root contains only that plan and its sidecar. It has no report, receipt,
training record, prediction, metric, or score and is preserved unchanged.

A lease-bound diagnostic traced the exact identity drift. Default
`importlib.metadata.distributions()` consults mutable `sys.path`. Model
construction eventually exposed `setuptools/_vendor`, after which 12 vendored
metadata entries appeared to be newly installed distributions even though the
private UV venv had not changed. The proposed repair inventories distributions
only from the venv's fixed `sysconfig` `purelib`/`platlib` roots:

| Path | SHA-256 |
|---|---|
| `ieee_mi/full_grid.py` | `4418947a8ecf46deefd8529adaaa3977b1fa156c26aec82224fc865d1a4ad738` |
| `ieee_mi/tests/test_full_grid.py` | `a32334338d250e7258bc636773328727e6fdf0b28b176544b534cc525aedf6ca` |

The sealed candidate passed 294 of 294 tests. An explicit reproduction made
the same 12 vendored entries appear under default discovery while the repaired
inventory stayed at 111 distributions with package SHA-256
`445c4281ffc54200dcec6869c2bb38131ba45d9875d7586e99bf3e255d2edae7`.
A scratch-only score-blind CUDA replay then published all 172 of 172 checks.
The attestation records no trial features, labels, splits, accuracy, or score.
Its plan, report, and receipt hashes are respectively:

- plan identity:
  `f553c06030b64619e3fe40a36e54b51c2a489fa87fbb305f205a6c97cc4ae950`;
- raw `report.json`:
  `db18746e993a0c518e997478c4586ea536042e80fcba67e35b25490637cc7729`;
- raw `receipt.json`:
  `9da49528dfdc2855c6b6aca10493f1f221684bce5b38f5b2f0a4c98dc913ea52`.

Independent no-edit review of this exact repair and scratch attestation is
mandatory before creating a v3 formal project or plan.

That review subsequently approved the scoped environment inventory. It
independently matched the candidate hashes, reproduced the complete 294-test
offline UV closure, and validated a sealed 172-of-172 CUDA attestation. The
review confirmed that the environment inventory is rooted only in the private
venv's resolved `purelib`/`platlib`, remains stable when vendored metadata is
added to `sys.path`, and records no trials, labels, splits, predictions,
accuracy, or score.

The separately sealed v3 project was then preflighted at
`formal_projects/common_grid_43_v3`, with run root
`runs/common_grid_43_formal_v3`. Its immutable plan identity is
`5d12e6efaf5e4f36d6508d1ea9f16029c656ab365a01c10b9a37cc9d0b5c6e49`.
All 172 ordered score-blind CUDA checks passed, and the source, private UV
environment, cache, GPU lease, plan, report, receipt, and run/project
bindings validated.

The three-GPU v3 launch produced no completed benchmark record. Each worker
failed its first publication because the post-training resource check sampled
the worker's own residual GPU utilization immediately after its CUDA kernels:
observed values were 13--42%, above the unchanged 10% idle threshold. There
was no foreign compute process and no model/scientific failure. The exact
v3 process group was terminated, all four GPUs returned idle, and the run was
preserved as failure evidence. It must not be resumed with changed source.

A narrowly scoped v4 candidate adds a bounded publication cooldown while
retaining the same resource thresholds:

| Path | SHA-256 |
|---|---|
| `ieee_mi/full_grid.py` | `962d82e55efe6d7bd0d1ebac347c05139fe6cdeb63e239cfedff83eb1c9e1c35` |
| `ieee_mi/tests/test_full_grid.py` | `1a6ddb9dd450b53766cec0068cedd3c0fe62f804385e0f6750bab56708631fc8` |

The worker may wait for at most 30 seconds, polling every 250 ms, only when
the sole blocker is above-threshold utilization attributable to its own
positive compute-memory footprint. A foreign process, excess foreign memory,
missing own-memory evidence, malformed counters, or a disk-floor failure
returns unsafe immediately. Timeout returns the final unsafe status and never
relaxes the 10% threshold. The one-shot pre-claim check is unchanged; only
the pre-publication check uses the cooldown.

The exact candidate passed the complete private, offline UV closure at
**300 of 300 tests** with 93 expected warnings in 33.14 seconds. Added
regressions cover successful own-utilization cooling, immediate foreign
process rejection, absent own-memory rejection, excess foreign-memory
rejection, fail-closed timeout, and immediate disk-floor rejection. A
pair of independent, no-edit static audits then rehashed the exact two files
and returned **APPROVE**. Both confirmed the unchanged hard thresholds,
fresh per-iteration GPU/disk probes, immediate foreign-activity/disk failure,
fail-closed timeout, direct worker callback wiring, and the second resource
check under the owned-claim lock. Both noted only that the added cooldown
tests are helper-level rather than a dedicated end-to-end worker integration
test. A bounded, one-GPU scratch publication pilot therefore remains mandatory
before constructing any v4 formal project or plan. No v4 plan, benchmark
output, prediction, metric, or score has been created.

The clean v2 pilot project was then preflighted on GPU 0 and published a valid
172-of-172 score-blind attestation. Its bounded replay produced **11 complete
prediction records, 0 failed jobs, 0 live claims, 0 stale claims, and 0
partials** before the exact worker process group was terminated. The runner's
audit reported `exact_cartesian_complete=false` only because this was
intentionally a bounded sample (11 of 96,320 jobs); it opened no labels and
computed no accuracy or score. Two claims left by the bounded shutdown were
recovered through the runner's inode-validated stale-claim quarantine, not
manual deletion. All four GPUs were idle afterward. The preflight, pilot
records, quarantine receipts, and audit report remain preserved under
`diagnostics/common_publication_cooldown_pilot_v2`; formal v4 launch is now
eligible for a fresh plan and run root.

The fresh formal v4 project passed the full 300-test UV closure and published
plan SHA `7eec34d3583ea6c0ca0e05b0872581cfb040fc446385004a58f65fd50a352841`
with a valid 172-of-172 score-blind preflight. Its three-worker launch on
GPUs 0--2 was then stopped after worker 1 hit a new shared-publication race:
`staged immutable artifact failed descriptor checks` while creating the
publication fence. Workers 0 and 2 had produced 24 score-blind prediction
records; there were no job-level model failures, but the launch is not valid
because one worker exited. The exact process group was terminated, two
interrupted claims were recovered through inode-validated quarantine, all
GPUs returned idle, and no label, metric, accuracy, or score was produced.
The formal v4 run is preserved as failure evidence and must not be resumed;
the publication-fence race requires a new repair, closure, and independent
review before another formal launch.

The formal-v4 concurrency failure was reproduced from the source. The shared
immutable writer removed all sibling `.stage-*` files after one publication;
that deleted a live second worker's stage before its descriptor check. A
lease-free v5 builder applies a narrow repair: an advisory exclusive lock on
the held parent directory now covers stage creation, publication, and stale
stage cleanup. The exact candidate identities are:

| Path | SHA-256 |
|---|---|
| `ieee_mi/full_grid.py` | `b1d81621e2dde8fab3fb68cf426816f2aeef8c18f98876ce6410cc73ab763f27` |
| `ieee_mi/tests/test_full_grid.py` | `35d4c33ed5845bda09218921d146ef5cedc96331f429e87bac0a879e2791fa55` |

The clean v5 builder passes **302 tests** (the prior 300 plus distinct-leaf
and same-target concurrent immutable-write regressions) with 93 expected
warnings. Two fresh independent no-edit audits are pending. No v5 plan, preflight, training,
prediction, metric, or score exists.

Those audits approved the same-target regression and the lock implementation.
The clean formal v5 project then passed **302 tests**, published plan SHA
`0d3109e6764a96dfc80b148beb5ad7565f61b56466485a984ac4d6a1d69c3b31`, and
passed a fresh 172-of-172 score-blind preflight. Its launch was stopped after
32,659 records when 895 deterministic EEGNet attempts were rejected by the
unchanged score-blind alias scanner for Braindecode's legitimate architecture
parameter `F1=8`. No scores were computed; the run and failures are preserved
and cannot be resumed under a changed source identity.

A v6 compatibility repair now permits only `F1=8` at the exact EEGNet
architecture-scalar path and `F1=32` for the documented TCFormer path; the
record validator rejects any other model, value, or path. The exact candidate
identities are:

| Path | SHA-256 |
|---|---|
| `ieee_mi/full_grid.py` | `5e687a5257754da301d2d42984948dc315de1f979fd1e4bd7c27c507857e67f5` |
| `ieee_mi/tests/test_full_grid.py` | `731c82eb55bba37202da6a9e21b75f00c5c237d4517592cf2d22e8f9def9d006` |

The clean v6 builder passed **303 tests** with 93 expected warnings. The
independent reviews approved the narrowly scoped `F1` compatibility rule and
the frozen formal project. The production plan SHA-256 was
`d4dd852fc9e7b80c8c10c54df03d63ce0ae7e7555ee1b7abdee8d79dca28b61a`;
all 172 score-blind CUDA preflight cells passed.

Three single-GPU workers on GPUs 0--2 subsequently completed the exact
43-model by 448-split by 5-seed Cartesian grid. The run contains **96,320 of
96,320** immutable prediction packages, with zero missing, extra, failed,
partial, live-claim, or stale-claim entries. Six claims left by interrupted
workers were recovered through the runner's quarantine mechanism and all six
forensic entries are resolved. The independent final audit reported
`exact_cartesian_complete=true`, zero unsafe paths, zero unexpected root
entries, and a valid preflight attestation. Its SHA-256 is
`010b44542157efb6ba5a95376198b9b691e9803ce3fe605e77cb5cdb9739d063`.

The separately invoked frozen analysis then joined labels to the score-blind
predictions and atomically published all 17 declared artifacts. Every output
hash matches manifest SHA-256
`960ca912d058a2ad3b60def4116dbba436f8ae9162916cde42a4ab265bf01f5d`;
the tables contain 96,320 job rows, 28,380 subject-seed rows, 5,676 subject
rows, 215 dataset-model rows, and 43 overall model rows. The descriptive
equal-dataset leader is `cardinal_fbc_compactdyn_scale025_extended` at
73.9900% balanced accuracy. TCFormer is rank 12 at 73.3163%; the paired
hierarchical-bootstrap difference is +0.6737 percentage point with a
fixed-five-dataset 95% descriptive interval of -0.4273 to +1.8993 points.
The local-data leader is `cardinal_dynamics_sinc_extended` at 91.6250%.

These are opened-development-cohort results. The leader was selected after
ranking 43 models, the interval against the prespecified TCFormer context
comparator crosses zero, and no p-value or confirmatory decision was computed.
The result supports an exploratory common-grid ranking, not statistically
confirmed superiority, independent generalization, clinical efficacy, or a
global/SOTA claim.

## HemiQ harmonized-v2 runner

The pre-audit HemiQ candidate used:

| Path | SHA-256 |
|---|---|
| `ieee_mi/hemiq_v2_grid.py` | `5b5519fd80069caab75767531b36cab603afd93d9baf387f3109bc519538e53c` |
| `ieee_mi/hemiq_v2_analysis.py` | `6920d620190913d29af71e7c805743cda3f1662266ede915d1c364bc92a5dd00` |
| `ieee_mi/tests/test_hemiq_v2_grid.py` | `8c2ad104668e4a0eab43c02d56b52adf27e477a39b808c8bf2aa9d11a3a6960b` |

The isolated lab closure passed 96 tests, and the exact grid remained 45 jobs
(9 BNCI2014-004 participants by 5 seeds). Independent no-edit review rejected
the candidate before scores. Post-rename filesystem errors could leave
canonical results or plan files visible; a second worker could quarantine the
first worker's live stage; the analysis fence was not bound to one live
run-root namespace; failed quarantine could expose writable authority; worker
publication did not rebind source, UV, and configuration identity; and
persisted analysis accepted impossible metric ranges and inconsistent trial
and aggregate counts. A separate repair and new independent audit are in
progress. No HemiQ plan or score was created.

The second repair candidate is frozen at:

| Path | SHA-256 |
|---|---|
| `ieee_mi/hemiq_v2_grid.py` | `7813686c953fe9c9a240c6f51f29b3798c126a20e6b78b1c775fb5defc90f822` |
| `ieee_mi/hemiq_v2_analysis.py` | `281371d7ec5a2bc38af6751fe5c3ace5ca87e617effb4761ec7adde85a64ca4b` |
| `ieee_mi/tests/test_hemiq_v2_grid.py` | `a188608ccf402b82c44c2d734ceb4a7296f672445aa91c65edec99328a5f76af` |
| `docs/HEMIQ_V2_FORMAL_RUNNER.md` | `01a5b2ea1b1c7b8bd2dd93ce88219db9d003218018aeba5e33da26488e900be9` |
| `docs/HEMIQ_V2_ADAPTER_PLAN.md` | `4ec1a04abfb39b8751d65b435890e38a406874fe8e4e1399b3a693970fef784c` |

This candidate addresses all seven audit findings, holds the completion,
record, and prediction children in one descriptor-pinned snapshot, and rejects
a coherent live package-B substitution. Its final isolated lab closure passed
110 runner/model/matrix tests in 35.04 seconds, and a leased synthetic-only
CUDA replay passed all 9 checks without opening benchmark data. No score was
created.

The fresh no-edit audit reproduced those positive checks but **rejected** the
candidate on three narrower boundaries:

- post-guard cleanup could quarantine a replacement claim inode instead of
  the exact inode published by the worker;
- persisted prediction validation accepted a wrong trial count,
  out-of-range probabilities, and NaNs; and
- final analysis verification reread artifact paths instead of rebinding the
  held published-directory descriptor, so a coherent directory-A-to-B
  substitution was accepted.

A third repair and another independent review are required. The 45-job HemiQ
plan remains uncreated.

The third repair candidate is frozen at:

| Path | SHA-256 |
|---|---|
| `ieee_mi/hemiq_v2_grid.py` | `161854112df5f080b67df3d65ef8c8569231e6fee3019e2e3841712bb9b1b3d5` |
| `ieee_mi/hemiq_v2_analysis.py` | `a2e8831b2206c3cd3f9c67e7cf8c064450a4d099e224cc749f866cb517ecaad8` |
| `ieee_mi/tests/test_hemiq_v2_grid.py` | `d4a4415e386abd917d29609ba154583e06a740d2a06092472119d5ffe63f0615` |
| `docs/HEMIQ_V2_FORMAL_RUNNER.md` | `e02cbaf0ecf85cb683714253b381b7b7ea068db40a5bd68e56b2ee5b43d44e33` |
| `docs/HEMIQ_V2_ADAPTER_PLAN.md` | `fafff596c6f9f6ff86b6f37fc102562d83966a47b83bd659b682bb3de69cce1c` |

It binds cleanup to exact claim/result inodes, strictly validates persisted
prediction count, shape, dtype, finite probability range, row sums, and member
uniqueness, and retains the published analysis directory and all child
descriptors through final live-name rebinding. The private offline lab closure
passed 125 model/runner/matrix tests plus three final targeted regressions; a
leased synthetic-only CUDA replay passed 9 of 9 checks. No benchmark data,
score, formal plan, system package, or final-project path was changed. A fresh
independent no-edit verdict remains required before launch.

That fresh review matched all requested local/lab hashes and replayed 113
HemiQ plus 12 execution-matrix tests successfully, but **rejected** the third
repair. In `acquire_claim`, the staged claim is renamed to its canonical name
before `created_claim` is assigned. Both a move-then-raise rename fault and a
successful rename followed by parent-fsync failure left the exact canonical
claim visible while the call reported failure. Cleanup searched only for the
vanished staged name, then skipped canonical cleanup because the claim object
had not yet been constructed. The next repair must carry the staged inode as
the sole publication state and hide only that exact inode on every
post-rename exception, with replacement-survival tests. The formal 45-job plan
remains uncreated.

The bounded claim repair now freezes grid
`9e41c7eda2d5625c86ba618855e4948c5a04dea70d1b3bb58140c6d130a3d62a`
and grid tests
`50815b482e85f2d45964706cf4f2edb9d505f27bc3471f6311867ff695b18b7d`;
the analysis and two documents retain the third-repair hashes above. It
captures the staged claim payload and exact inode before rename and hides only
that inode after move-then-raise or parent-fsync failure. Four focused cleanup
and replacement-survival regressions passed, the original two probes now
leave no canonical claim, and the exact private offline lab replay passed 117
HemiQ plus 12 execution-matrix tests. A different fresh no-edit auditor must
still approve these hashes before the 45-job plan can be created.

That fresh no-edit audit matched all five canonical hashes and passed the
frozen 103 runner/analyzer tests plus 14 adapter tests under the exact
restricted-`PATH`, private-UV, offline launch envelope. It nevertheless
**rejected launch** before plan creation on real, unmocked environment
identity checks. `collect_uv_identity` required a `.python-version` file that
was absent from both the canonical scratch source and the documented final
project root; it then invoked bare `uv --version`, although the prescribed
restricted `PATH=/usr/bin:/bin` deliberately excludes the actual
`/home/hanafy/.local/bin/uv` executable. The document also pointed at the
still-empty final project root and reported 113 rather than the observed 117
HemiQ tests. The exact 45-job roster remains scoreless. A repair must bind the
explicit absolute UV executable, make the Python-version/project-root
contract executable rather than aspirational, correct the receipt, and pass a
real launch-envelope test before another independent review.

The isolated UV-identity repair candidate now proposes:

| Path | SHA-256 |
|---|---|
| `ieee_mi/hemiq_v2_grid.py` | `20017c6f9496770370a017ecf5196a3d97ba7ba4454082cda4be6c1528f97b90` |
| `ieee_mi/hemiq_v2_analysis.py` | `a2e8831b2206c3cd3f9c67e7cf8c064450a4d099e224cc749f866cb517ecaad8` |
| `ieee_mi/hemiq_v2_model.py` | `495f9c76fa8d10acb0cd1adeb1b576b2a3ed8486e981b5320550c3008fcb5a3c` |
| `ieee_mi/tests/test_hemiq_v2_grid.py` | `8ce2ad8a64d07a81ff4e0a20678a807a8a89abeda5da92319d68d596e31d1026` |
| `ieee_mi/tests/test_hemiq_v2_launch_envelope.py` | `7ddd791ecf514e9d95f8b0bdf75dec53d980f3b5e98f445e0896ffdd7f24c172` |
| `ieee_mi/tests/test_hemiq_v2_model.py` | `cfc43cb09f60b7fa0cd808cab95a539fe3de05b6cc3e7b2ca9ccef41aa088eaf` |
| `docs/HEMIQ_V2_ADAPTER_PLAN.md` | `8cf09bf2d868b8cc7826ad33f503d1e38d1d4d767f32d714fdffa26f125bf8ae` |
| `docs/HEMIQ_V2_FORMAL_RUNNER.md` | `09614f1032704578597d5911c8c596cfc2f7776183321c16b0be6caf97285725` |
| `.python-version` | `aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b` |

Its exact private offline closure passed 134 tests, including a real
unmocked restricted-`PATH` launch in which `uv` was absent from `PATH` and the
runner succeeded only through the bound absolute executable. The schema-v2
project package separates the current scratch source from the eventual formal
root and rejects relocation; the exact 45-job roster and model/config bytes
remain unchanged. This proposed freeze has not been copied into either
canonical tree and still requires a different no-edit auditor. No formal
artifact or score was created.

That different auditor matched all nine hashes and passed 122 frozen HemiQ
tests under the exact restricted private-UV envelope, but **rejected launch**
on a cleanup race shared by `_quarantine_leaf` and `_atomic_bytes` rollback.
Both paths checked an expected inode and then performed a separate
path-addressed rename. A name replacement in that interval caused cleanup to
move or unlink the replacement, leave the expected inode displaced, and
remove the canonical name. The failure affects claim, authority, result,
stage, and atomic-JSON cleanup. The UV/root repair remains valid evidence, but
the proposed freeze is not launchable until all such cleanup uses one
directory-anchored move/verify/restore primitive and receives another
independent review. No formal artifact or score was created.

The unified descriptor-transaction repair now freezes:

| Path | SHA-256 |
|---|---|
| `ieee_mi/hemiq_v2_grid.py` | `b9d357290358b7e53a453679eb81b52ae385d46d4528d51e7124ee1933e962ca` |
| `ieee_mi/hemiq_v2_analysis.py` | `08b46b04ec93f364a05f680cb1331df186313025fffbab485ee603a3cda51e16` |
| `ieee_mi/tests/test_hemiq_v2_grid.py` | `81cafcca0b18f3df02827f8b8dd2804ac612bdb669b48a4d3484a4b935bd4188` |
| `ieee_mi/tests/test_hemiq_v2_launch_envelope.py` | `7ddd791ecf514e9d95f8b0bdf75dec53d980f3b5e98f445e0896ffdd7f24c172` |
| `docs/HEMIQ_V2_ADAPTER_PLAN.md` | `8cf09bf2d868b8cc7826ad33f503d1e38d1d4d767f32d714fdffa26f125bf8ae` |
| `docs/HEMIQ_V2_FORMAL_RUNNER.md` | `622d2644ad5ea9e63e856a7c830257cc8163f9a8a959236ce2faec80d1b97e0b` |
| `.python-version` | `aa0d6581054e6e4ff3f91839deca7a854ad37221b8784d060b42d0f847ff1a3b` |

All production no-replace relocation, atomic publication, quarantine, claim,
result, analysis, rollback, and exact-unlink paths use one
descriptor-relative transaction. Actions retain live object receipts;
reporting receives closed-descriptor-safe immutable location metadata rather
than bare actionable paths. The exact restricted private-UV closure passed
130 tests and revalidated the unchanged 45-job scientific roster.

A new auditor who authored neither repair matched the frozen hashes before and
after a sealed no-edit replay, passed all 130 tests, and independently checked
ordinary process interleavings, post-move/fsync faults, parent rebind and
A-to-B-to-A history, unrelated-entry survival, atomic JSON rollback, exact
claim/result/analysis cleanup, SIGKILL recovery, closed-receipt reporting,
absolute UV identity, and scratch/formal relocation rejection. Verdict:
**APPROVE_FOR_HEMIQ_LAUNCH**. The seven changed or missing files were copied
backup-preservingly into canonical scratch source; prior files remain under
`backups/hemiq_integrity_freeze_20260730_0328`. No HemiQ plan, training,
prediction, analysis, or score existed at approval or promotion.

## Gauge-quotient experimental candidate

The first GaugeQuotient candidate used:

| Path | SHA-256 |
|---|---|
| `ieee_mi/gauge_quotient.py` | `7430fc0ef73630add3822816b9973bd459752e1eb08e72319b107f5cdcfc6855` |
| `ieee_mi/gauge_quotient_screen.py` | `0a32b26b3045a9ec34865d28a5cb994f6c9442bc3b940ea5e3389df21e4acb57` |
| `ieee_mi/tests/test_gauge_quotient.py` | `a6b7701570538cf652e411707c4829f024db3d8b179769d011541edd54884362` |
| `docs/GAUGE_QUOTIENT_NOVELTY_PLAN.md` | `9556162742c741490d9a5d43e2be561e32eb4793afed0fce57b473d463c0355f` |

Its 16-test isolated lab property suite passed and confirmed the intended
function-preserving zero-start continuation, parameter budget, quotient
projection, masking, time-reversal parity, and gradients. Independent
scientific review nevertheless rejected scoring those hashes: malformed
inputs were validated only after the native stateful branch and therefore
advanced both BatchNorm counters before raising. The review also found close
equation-level prior art for antisymmetric lagged covariance and established
average-reference projectors. The defensible boundary is a new neural
composition, not a new covariance statistic, reference principle, causal
estimator, or globally invariant predictor. A state-safety repair and revised
novelty document are in progress; the candidate remains absent from the formal
registry and has no score.

The repaired exploratory candidate is frozen at:

| Path | SHA-256 |
|---|---|
| `ieee_mi/gauge_quotient.py` | `49179d934215dfe4ce6c6ecf23f6cc05acab20f32a3b3398299787141ae0599e` |
| `ieee_mi/gauge_quotient_screen.py` | `60062362bc23180aefdb5bc26bfa735e548fc5a8a24926326e50966ce639f15c` |
| `ieee_mi/tests/test_gauge_quotient.py` | `721de6dc180dc225eb6cad739667b6ac16b797437a963322d754c8495f3f99b1` |
| `docs/GAUGE_QUOTIENT_NOVELTY_PLAN.md` | `caf0cbc1ccfcf0f69a962d81a5e49c7d1c68d5c3ffcf16cc41983a9c1e243888` |

The repair front-loads every input check before the native stateful path. Its
isolated lab UV closure passed 20 tests with 20 expected numerical-library
warnings. Independent no-edit attacks exercised malformed inputs through both
public entry points on CPU and initialized CUDA, confirmed unchanged
parameters, buffers, hooks, CPU/CUDA RNG state, and CUDA initialization state,
and reproduced exact native logits and retained native state for every opened
channel/time/class shape. The auditor also confirmed the 16,730 binary and
20,540 four-class trainable-parameter counts, the fixed-filter tolerances,
mask/common-offset/sign/time-reversal properties, and the exact frozen
17-subject and disjoint-115-subject cohorts and decision rules.

A later equation-level search found an additional decisive collision:
Hindriks et al. (2025) derive the same bivariate exterior product and its
lagged cross-correlation asymmetry for EEG. The novelty document was tightened
and independently re-audited at the hash above. The verdict is
**APPROVE_FOR_GATE1_ONLY**: it authorizes construction and review of the
prespecified exploratory 17-subject falsification runner, not Gate 2,
promotion, publication novelty, state-of-the-art, confirmation, clinical, or
causal claims. No Gauge plan, training job, prediction, or score exists yet.

The first frozen Gate 1 runner candidate used:

| Path | SHA-256 |
|---|---|
| `ieee_mi/gauge_gate1_runner.py` | `afd5907577de24c7b69a94a377f91fdebd1684247fe930579ab9c2b0bc14fe27` |
| `ieee_mi/gauge_gate1_analysis.py` | `1393e04fb24f6a894ed58889810bc86ec50c42ba2eae10d090d1f9a0c8520a87` |
| `ieee_mi/tests/test_gauge_gate1_runner.py` | `30f4da2f0617d0408992b070c5aa3f209f51c0c636f820e5a372981f786073c6` |

Its Mac synthetic closure passed 44 tests, but the first exact Linux replay
rejected it: ext4 requires write permission on a moved directory when a
cross-parent rename updates its `..` entry, so publication of the presealed
mode-0555 package failed with `EACCES`. This was a launch blocker, not a model
or CUDA result. The canonical Gate run remained absent.

The inode-bound Linux repair then froze runner
`f37f27c34ebcb0a92c553dc33f97e40eca137d3fe4985530455d11918ef8bac4`
and runner test
`c936602c65ab22a2602674683b2ec4ee12f59b1321aa37e517d0b1a31ee68168`;
the analyzer and model hashes above were unchanged. It temporarily grants
owner-write only to the exact held private-stage inode, restores mode 0555
after success or failure, and preserves no-replace and move-then-raise
cleanup. The exact offline lab model-plus-runner replay passed 67 tests.

A fresh no-edit audit nevertheless **rejected launch** before plan creation.
The real cache sequence duplicated one open descriptor for ZIP-member
validation, which advanced the shared open-file-description offset to the end
of the file; the next duplicated NumPy reader inherited that offset and raised
`EOFError`. The auditor reproduced the failure through the real
`local_exp4` subject-1 loader and established that explicit offset reset made
all 17 cache and split contracts load correctly. It also found that the
documented absolute-UV invocation must either bind the executing UV binary
directly or explicitly include `/home/hanafy/.local/bin` in `PATH`, because
environment fingerprinting currently calls `which("uv")`. A descriptor-offset
repair, real-cache replay, and new no-edit approval are required. No Gauge
plan, claim, training job, prediction, analysis, or score exists.

The descriptor-offset/UV repair then froze:

| Path | SHA-256 |
|---|---|
| `ieee_mi/gauge_gate1_runner.py` | `6982601e3ef910bef6f7618bbf85e99962d21d347869512eb28af61ad36ab8bc` |
| `ieee_mi/gauge_gate1_analysis.py` | `1393e04fb24f6a894ed58889810bc86ec50c42ba2eae10d090d1f9a0c8520a87` |
| `ieee_mi/tests/test_gauge_gate1_runner.py` | `ea14ecfebd8970120de04141d102f323c105da4eeab620c33a3bd9321be7acd1` |
| `docs/GAUGE_QUOTIENT_NOVELTY_PLAN.md` | `a95e0438314a891a70a243765cb7f507fb6237adadd4841634a1f0712977ba3f` |

Its independent open-file-description readers left a nonzero held-descriptor
offset unchanged, explicit absolute UV binding worked with UV omitted from
`PATH`, and all 17 real cache/metadata/split/selected-row loads passed. The
exact Linux closure passed 70 tests and all 51 read-only reference records
validated.

Fresh no-edit review still **rejected launch**. A coherent namespace
A-to-B-to-A attack could rename an opened file or package A aside, install B,
remove B, restore the original A, and then pass revalidation. The held leaf
inode, bytes, and stat fingerprint were once again correct, but the snapshot
had not retained the parent/ancestor directory mutation history. The next
repair must hold and revalidate the relevant parent and ancestor descriptors,
fingerprints, and component bindings, with file, package, candidate-analysis,
and reference-analysis round-trip attacks. The formal Gate run remains absent;
no plan, training job, prediction, analysis, or score exists.

The namespace-history and concurrent-layout repair is now frozen at:

| Path | SHA-256 |
|---|---|
| `ieee_mi/gauge_gate1_runner.py` | `7013a378445d8cbb0032effd2f868201aab3d8c4eb1ff97b399c9642bd215b45` |
| `ieee_mi/gauge_gate1_analysis.py` | `b9dd5a1b7f281b850dedb434b63f636cd7770f4a61e40bafe74daaf918106b31` |
| `ieee_mi/tests/test_gauge_gate1_runner.py` | `bd5d17e8b73568c00ce96ccb61c0e6f1a2701bfecd86037e63f2c4676b8928bb` |
| `docs/GAUGE_QUOTIENT_NOVELTY_PLAN.md` | `480416dfef995083dbcac229656d47e2bb3bff53577f3f94d6040665df6a9031` |

Every file and sealed package now retains the exact directory-descriptor chain
from its declared stable root through its target parent. Revalidation checks
full directory mutation fingerprints, component bindings, the target binding,
held leaf/package identity, and exact bytes. The plan, results, and analysis
use `run_root`; source, comparator, cache, active UV environment, and bound UV
binary use their explicit project, reference, cache, environment-prefix, and
UV-parent roots. Analysis is published below a precreated
`run_root/analysis` subtree. All 17 exact result parents are durably precreated
under run authority before plan publication, and workers subsequently validate
those parents without creating shared layout.

The Mac closure compiled and passed 48 tests with 12 expected Linux skips. An
isolated lab copy compiled and passed 60 runner tests and 80 combined
candidate-plus-runner tests. The real read-only replay loaded all 17 planned
dataset jobs, validated all 51 comparator records, matched comparator snapshot
`c5ecf183...`, bound the explicit UV executable, and revalidated the final
16-file source closure with mapping digest
`97fb37f838685919626dbb2a7b993f81b87e1b978f14a63a32536df96cb4aa44`.
Adversarial coverage includes same-parent file/package A-to-B-to-A,
ancestor-directory A-to-B-to-A, persisted-analysis candidate/reference
restoration, unrelated-sibling allowance, three-worker sibling publication,
result-then-analysis publication, and descriptor lifecycle checks.

The four frozen files were copied backup-preservingly into the canonical lab
scratch source and all seven runner/model/test hashes were recomputed there.
The formal Gate run root and `/home/hanafy/neuralnetwork` remained absent or
empty. This is a tested candidate only: a genuinely different no-edit auditor
must still approve these exact bytes before Gate 1 initialization.

The independent no-edit audit matched all seven hashes, passed 80 of 80
candidate-plus-runner tests under the exact private offline UV envelope, and
replayed the real 17 candidate cache cells and 51 bound comparator records.
It still **rejected launch** on one untested cleanup race. Cleanup first
checked a canonical entry's inode, then separately passed its absolute path to
Linux `renameat2`. An adversarial substitution in that window caused the
replacement inode to be moved to quarantine, the expected inode to remain
displaced, and the canonical name to disappear. The defect affects
claim/result/analysis rollback callers and violates exact-inode replacement
survival on a shared workstation. All audit descriptors and temporary paths
were closed or removed; the formal run remains absent. The next repair must
use an anchored directory-descriptor transaction, verify after the move, and
restore a mismatched moved inode without clobbering concurrent state, with a
deterministic in-window substitution regression and a new independent
no-edit verdict.

The isolated exact-transaction repair now proposes:

| Path | SHA-256 |
|---|---|
| `ieee_mi/gauge_gate1_runner.py` | `8edeb5b59122f2cd9e732141af93aee4d5dd509e363840785b16d390ef72e7ee` |
| `ieee_mi/gauge_gate1_analysis.py` | `b9dd5a1b7f281b850dedb434b63f636cd7770f4a61e40bafe74daaf918106b31` |
| `ieee_mi/tests/test_gauge_gate1_runner.py` | `1224e2ba82f84b154a37c4e40834cb4eb5f3bd1fbf914ad5468b86950a5f40e8` |
| `docs/GAUGE_QUOTIENT_NOVELTY_PLAN.md` | `480416dfef995083dbcac229656d47e2bb3bff53577f3f94d6040665df6a9031` |

The model, screen, and model-test hashes remain the previously frozen
`49179d93...`, `60062362...`, and `721de6dc...` identities. Linux cleanup now
uses an anchored directory-descriptor move, verifies the moved inode, and
restores a mismatched replacement without overwriting concurrent canonical
state. The deterministic A-to-B substitution, concurrent-C, move-then-raise,
retained-parent, and path-rebind regressions passed; the exact private offline
closure passed 85 tests. The real read-only replay again validated all 17
candidate cells, 51 comparator records, 17 cache/split cells, and the same
frozen environment and decision rule. The formal run remains absent. This is
only a proposed freeze until a different no-edit auditor rehashes and attacks
it.

That different auditor matched all seven hashes, passed all 85 tests, and
replayed the 17 cache cells and 51 comparator records, but **rejected launch**
on recovery-name identity. The move and moved-inode verification were
correctly anchored to a retained parent descriptor; however, the function
returned an unchecked absolute path. If the parent name was rebound after the
descriptor opened, that returned path named an unrelated entry in the new
parent. Direct cleanup, claim cleanup, and orphan cleanup reproduced the
problem; orphan cleanup also sealed the unrelated entry. The next repair must
carry descriptor-bound recovery identity through every caller and rebind the
live parent namespace before any path-based report, permission change, or
subsequent recovery action. The formal run remains absent.

The descriptor-receipt repair now proposes:

| Path | SHA-256 |
|---|---|
| `ieee_mi/gauge_gate1_runner.py` | `ba18f32368f6938b283c2b3c70cfa4a497baa5b6ee10de3cfdbc1c53cefd2a2d` |
| `ieee_mi/gauge_gate1_analysis.py` | `624f01a2b5cddb992631460b6fefcfc6a312c7cace374d20b54df10b9a470b32` |
| `ieee_mi/tests/test_gauge_gate1_runner.py` | `6bf11d877749c31deb417072481feae4d03e559e90aaec1f70eac3093c42745f` |

The frozen model, screen, model tests, and novelty document are unchanged.
Cleanup actions now require a live descriptor-bound `RecoveryReceipt`;
helpers that must report recovery state return only immutable identity
metadata with no path action. Stable-ancestor and mutable-parent bindings are
checked around relocation, sealing is descriptor-relative, and name changes
after helper return are covered. The exact private offline closure passed 96
tests and the real read-only replay again passed all 17 candidate cells, 51
comparator records, and 3,384 selected rows. The 16-file source mapping digest
is `62859433cf44dff8ceacb047a23b6162385851b1bcae065b9e002d085a88fcb9`.
The formal run remains absent. A fresh no-edit auditor who authored neither
repair must still approve these exact bytes.

That fresh auditor matched all seven hashes, passed the exact 96-test private
offline UV closure, replayed all 17 candidate cells, 51 comparator records,
and 3,384 selected rows, and independently passed the final recovery-rebind,
descriptor-lifecycle, immutable-metadata, unrelated-entry, fsync-failure, and
Linux-only checks. Its verdict was **APPROVE_FOR_GATE1_LAUNCH**. The three
changed files were then copied backup-preservingly into the canonical scratch
source; the prior bytes are retained under
`backups/gauge_recovery_freeze_20260730_0305`.

The immutable formal plan
`a0c88ed17737201c43122e4eefd4b515a1387a9f536e3a8303a913575b88e2c2`
created exactly 17 candidate-only jobs. Three UV workers used only the first
three full GPU UUIDs and completed 6/6/5 jobs; GPU 3 remained unassigned.
Post-run audit found 17 complete, zero missing, zero corrupt, and no claims.
The persisted decision was reproduced idempotently (`created: false` on the
second analysis call):

| Model | Equal-dataset balanced accuracy |
|---|---:|
| GaugeQuotientCrossMomentNet | 75.0704% |
| CardinalFBC micro extended | 74.3988% |
| FBCNet | 74.1071% |
| TCFormer | 72.6250% |

Gauge exceeded the primary reference by 0.6716 percentage point, won 11/17
subjects, and had a worst dataset deficit of 1.6667 points. However, its
dataset deltas were negative on Local Exp4 (-1.6667), BNCI2014-001 (-1.3889),
and Cho2017 (-0.6250), and positive only on BNCI2014-004 (+1.4583) and
PhysioNet MI (+5.5804). It therefore satisfied three of four Gate 1 conditions
but failed the prespecified requirement for nonnegative deltas on at least
three of five datasets.

Verdict: **Gate 1 failed; execute
`kill_candidate_before_disjoint_gate`**. Gate 2, ablations, tuning, renamed
retries, SOTA language, and promotion are forbidden. This is retained as an
informative opened-development negative result. The committed analysis
package hashes are:

| Artifact | SHA-256 |
|---|---|
| `analysis.json` | `d89480531b4f37f98aa1faae27f976038ec5e09322368d02a9626bfec9d29eff` |
| `analysis.sha256` | `4b59cb2f59c53c301587165826df82d6c053d45d0227d59a658f59b92a74d343` |
| `COMMITTED` | `89f1d7ce9ae5d94d7e39f85e3d8e96d8c4b8b048bd977fdf11684cc71f3eff2e` |

## Author-recipe reference runner

The hardened Author Recipe candidate is frozen at:

| Path | SHA-256 |
|---|---|
| `ieee_mi/author_recipe_grid.py` | `41b7eddf9ce2fb9984a6eeb5fc7e40aff9d8b36c902494a398e2822f8e17a44b` |
| `ieee_mi/author_recipe_analysis.py` | `e237e5deff213a6d210c6aec9010e61b9c9fd12bdeef42ab2469d1c0b97971b8` |
| `ieee_mi/tests/test_author_recipe_grid.py` | `c3b7990422dfa25173827ca17a99f7dbe7c29ca63553f85713e24153c97a114c` |
| `docs/AUTHOR_RECIPE_GRID.md` | `2948120c58396ddea7ae913e568abe66c7a76419e59f4b9e300a222f39315ca7` |

The exact 4,480-job roster remains two reference procedures by 2,240 jobs:
TCFormer with its pinned official-source identity and FBCNet under its separate
released-recipe approximation. The hardening adds native directory-relative
no-replace publication, exact-inode rollback/quarantine, claim-owned
publication state, coherent directory-plus-all-child-descriptor snapshots,
raw duplicate-NPZ rejection, mode-0444 children and mode-0555 completion
directories, and source/environment/config/cache/split/device/lease rebinding
around publication. The analyzer retains the same pinned input snapshots
through output publication and final live-name rebinding.

The final canonical lab replay used the explicit project-private UV
environment and cache, `--active --offline --no-project`, an isolated
basetemp, disabled user-site and bytecode writes, and passed 59 of 59 tests.
Coverage includes concurrent no-replace winners and sibling creation, leaf and
ancestor A-to-B-to-A attacks, move-then-raise and seal-then-raise faults,
post-rename fsync residue, replacement survival, writable/symlink/hardlink/
special-file rejection, and analyzer lock/output rollback. Local and lab
hashes matched; no Author plan, run, cache, training, prediction, or score
artifact was created, and the final project remained empty.

The subsequent independent no-edit audit matched all four hashes, replayed 59
of 59 tests in the restricted private UV environment, reconstructed 4,480
unique jobs with the exact 2,240/2,240 track split, and verified the compact
TCFormer snapshot at upstream commit
`74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`. It also replayed ancestor and
leaf replacement, concurrent publication, exact-inode rollback,
move-then-raise, post-move fsync, and analysis rollback attacks.

Verdict: **APPROVE_FOR_AUTHOR_RECIPE_LAUNCH**. This approves artifact and
procedure integrity only; it is not a claim that the two recipes are
scientifically identical to every implementation detail in their source
papers. No plan, training, prediction, metric, or score existed at approval.

## Pending entries

The ledger must be extended after, and only after, each of these remaining
events:

1. execute the approved Author Recipe, HemiQ, and GeoAdapt formal tracks in
   their separate procedure tables; none of those proposed grids has production
   score completions yet;
2. finish native CardinalFBC and Control Grid hardening and repair the rejected
   Procedure Grid boundary;
3. freeze one candidate and evaluate it once on genuinely sealed or prospective
   confirmation data under a preregistered primary comparison;
4. finish assembling `/home/hanafy/neuralnetwork` from the explicit release
   allowlist, then record an isolated online/offline UV rebuild and full-test
   proof;
5. complete PDF generation and rendered-page inspection for the clean release;
   and
6. record the user/institution decisions for licensing, authorship/CITATION,
   local-data governance, IRB/consent language, and any trained-weight release.
