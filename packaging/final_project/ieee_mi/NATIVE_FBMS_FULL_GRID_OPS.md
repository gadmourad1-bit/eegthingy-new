# CardinalFBMS full-grid operations

This is a development-only expansion. It never authorizes a sealed subject and
the operations runner never computes a score.

## Immutable workload

- Cho2017: subjects 16--52, folds 0--4, seeds 7/17/27/37/47, five conditions
  = 4,625 records.
- Physionet MI: subjects 1--54, folds 0--2, the same five seeds and conditions
  = 4,050 records.
- Total = 8,675 records.
- The 455 already-audited fold-0/seed-7 records are fully re-audited and then
  byte-copied into fresh full-grid roots. Their old roots remain unchanged.
- Missing work after seeding = 8,220 records.
- Every full-grid directory uses the frozen uniform name
  `s{subject:03d}_f{fold}_seed{seed}_{condition}`.

The audited screen measured 50.08 GPU-worker-hours when extrapolated to the
entire grid. Four concurrent workers imply a 12.52-hour evaluation-only lower
estimate from an empty grid, or about 12.0 hours after seeding the existing 455
records. Allow 13--16 wall-hours for fresh Python process launches, checkpoint
and cache validation, uneven job tails, and final full-grid re-audit.

## Safety and recovery

`native_fbms_full_grid_ops` and the independent full-grid auditor pin the
checkpoint bytes and tensor state, source corpus, partition, initial state,
transfer source, complete 11-file writer manifest, all 91 target-cache hashes,
and the exact RTX 5070/CUDA/Python/package environment. A valid record with a
different pin is rejected as stale. CUDA is not caller-configurable. Any unknown
entry in an audited root is rejected.

Before seeding or scheduling, preflight retains each old screen cache path only
as absolute historical provenance, derives the live path from
`--cache-root` plus the locked dataset/subject identity, deduplicates it to one
file for each of the 91 target subjects, and rehashes every current cache file.
A missing, rebuilt, or byte-different cache is a pre-launch `NO-GO`; moving an
identical cache into the current locked root is allowed.

Each record gets a fresh Python process, so no model, optimizer, RNG, or batch
normalization state can cross records. Exactly four processes run at once. The
writer publishes via fsynced staging plus atomic rename. Child processes inherit
the runner's OS file lock, so a replacement runner cannot remove their claims if
only the scheduler dies. After a machine power cut, rerunning the same command:

1. fully audits every existing output and all cross-record semantics;
2. removes only recognized writer lock/staging transients;
3. revalidates or completes the byte-copy seed step;
4. launches only missing records.

Each record has a non-configurable 3,600-second watchdog (the observed maximum
was under two minutes). A timeout is journaled, terminated, and eventually
killed; no new records launch after a failure.

Logs and the append-only operations journal are written only under `--ops-root`,
which must be disjoint from both audited roots.

## GPU-box commands

First synchronize and test the code. Then run the score-blind preflight (no
`--execute`):

```bash
cd /home/user/Desktop/eegthingy_ieee_arch_v7_20260719
mkdir -p artifacts/native_fbms_transfer_full_cho_s16_52 \
  artifacts/native_fbms_transfer_full_physionet_s1_54 \
  ops/native_fbms_full_grid
PYTHONPATH=. .venv/bin/python -m ieee_mi.native_fbms_full_grid_ops \
  --screen-cho-root artifacts/native_fbms_transfer_cho_s16_52_fold0_seed7 \
  --screen-physionet-root artifacts/native_fbms_transfer_physionet_s1_54_fold0_seed7 \
  --cho-root artifacts/native_fbms_transfer_full_cho_s16_52 \
  --physionet-root artifacts/native_fbms_transfer_full_physionet_s1_54 \
  --ops-root ops/native_fbms_full_grid \
  --cache-root data_cache \
  --checkpoint artifacts/native_pretrain_cardinal_fbms_seed7/checkpoint.pt
```

Preflight must emit `"decision": "GO"`, `"scores_computed": false`, the exact
pins, and only expected completed/pending counts. To execute, rerun the identical
command with `--execute`. The same execute command is also the recovery command
after an interruption.

Do not launch when preflight reports `NO-GO`, when the full GPU test suite is not
clean, or when the source manifest includes an extra optional repository-root
`requirements.txt`; that would make new provenance differ from the 455 seed
records.

## Frozen primary comparison

Before outcome generation, the indexed checkpoint projection
`pretrained_indexed_fbmsnet_spherical_spline` was frozen as the single primary
checkpoint-matched comparator. The gate computes a separate paired subject
bootstrap after concatenating outer-fold predictions and averaging seeds within
subject. Its only new criterion is an equal-dataset macro one-sided 95% lower
bound above zero; dataset deltas and intervals are reported but their individual
lower bounds are not gate criteria. This is a development-only checkpoint
control, not an author-faithful FBMSNet reproduction. The prespecified oracle
reference-envelope point checks remain secondary checks.
