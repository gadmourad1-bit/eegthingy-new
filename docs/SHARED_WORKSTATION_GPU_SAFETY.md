# Shared-workstation GPU safety contract

Status: shared lease helper frozen and independently tested; individual formal
runners remain subject to their own audits.

## Purpose

The lab host has four physical RTX A5000 GPUs and is used by other people. The
benchmark therefore limits itself to at most three simultaneous GPU workers
and leaves the fourth device unclaimed. Every formal runner uses the same
project-wide registry implemented by `ieee_mi/project_gpu_leases.py`.

The lease layer is necessary but not sufficient. Before claiming a job, each
runner must also use `nvidia-smi` to reject a GPU with an unrelated compute
process, excessive foreign memory, or utilization above the frozen threshold.
A lease coordinates this project; the cooperative probe protects users outside
this project.

## Physical-device identity

Leases name a complete NVIDIA physical UUID, never a process-local ordinal.
The runner resolves requested indices to UUIDs before planning, freezes the
ordered physical inventory, and binds a worker with `CUDA_VISIBLE_DEVICES` only
after acquiring the matching UUID lease.

The registry lives below the verified clean project root in:

```text
.ieee-mi-project-gpu-leases/
```

Its persistent fence and active lease records use exact JSON schemas. A lease
binds the clean project, run root, immutable plan SHA-256, track scope, physical
GPU UUID, owner host/PID/boot/start identity, nonce, creation time, and the
lease and fence inode identities.

## Three-worker cap and exclusivity

Under one locked registry fence, acquisition:

1. validates every existing registry entry;
2. archives only a lease whose owner is provably stale;
3. rejects an already leased physical UUID;
4. rejects acquisition when three live leases already exist;
5. publishes a staged, fsynced, no-replace lease;
6. revalidates the registry fence immediately after publication; and
7. returns an inode- and nonce-bound handle.

Release reopens and verifies the exact lease, owner, fence, inode, nonce, and
payload before archiving it. It never unlinks an unverified pathname.

## Guarded authoritative mutations

A standalone liveness check followed by a filesystem write has a race: another
cooperative process could release the lease between those two operations. The
public `guard_gpu_lease` context closes that gap by holding the same registry
fence and lock while the runner publishes exactly one authoritative claim or
performs exactly one final result rename.

The guard:

- validates the exact lease before yielding;
- yields a canonical detached receipt for the immutable claim or result;
- keeps the registry lock held across the caller's authoritative mutation; and
- validates the lease and fence again before unlocking.

Formal runners must quarantine a newly published object if post-publication
validation fails. Claims and records persist the exact receipt and validate
that it matches the project, run, plan, track, and GPU identities.

Long model fitting does not hold the registry lock. The worker retains its
lease, reasserts it around boundaries, and discards rather than publishes work
if the capability is lost.

## Power-cut behavior

Lease and fence files are staged, fsynced, renamed without replacement, and
followed by directory fsync. On restart, recovery distinguishes a live owner
from a stale owner using host, PID, Linux boot ID, and the parenthesis-aware
`/proc/<pid>/stat` process-start marker. Unknown, malformed, linked, or special
entries fail closed and are preserved under the forensic directory rather than
silently deleted.

Rerunning an audited command after a power cut may recover recognized stale
claims and non-authoritative stages. It must not reuse a record whose source,
environment, cache, split, plan, lease receipt, or completion checksum differs.

## Frozen helper evidence

The frozen helper identities are:

| File | SHA-256 |
|---|---|
| `ieee_mi/project_gpu_leases.py` | `b0d12a8689430c00984bc4fe95800ee626cb37f20434b578591b3d26f0d91908` |
| `ieee_mi/tests/test_project_gpu_leases.py` | `43a509eb90907cbb1b07b691caad2787de50e2f31c6ae6e4a0afed2f0fb31b26` |

All 18 helper tests passed independently on the lab host in the existing
UV-created Python 3.12.13 virtual environment. The suite covers cap-three
concurrency, duplicate UUID rejection, stale recovery, active-owner
protection, symlink/hardlink/special-file rejection, fence and lease
replacement, no-replace interleavings, exact receipt binding, assert-time
identity, a concurrent foreign-process release attempt, and a legitimate
same-owner release attempt from another thread while the publication guard is
held. No package was installed or changed for this verification.
