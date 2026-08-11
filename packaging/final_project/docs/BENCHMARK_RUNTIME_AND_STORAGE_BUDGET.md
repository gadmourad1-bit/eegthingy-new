# Preliminary benchmark runtime and storage budget

## Status

This is a capacity-planning estimate, not a benchmark result and not a promise
of completion time. No score was produced to prepare it. The estimate must be
replaced by a measured pilot after the relevant runner passes independent
adversarial review.

The shared workstation must remain usable by other researchers:

- use only the project UV virtual environment and project UV cache;
- run at most three GPU workers;
- leave physical GPU 3 unreserved;
- use no distributed training;
- stop before available disk space falls below 50 GiB;
- never install into system Python or change the NVIDIA driver or CUDA toolkit.

## Grid size

The common-recipe track contains 43 models and 96,320 atomic jobs:

| Dataset | Subjects | Folds | Seeds | Jobs per model |
|---|---:|---:|---:|---:|
| Local experiment 4 | 8 | 1 | 5 | 40 |
| BNCI2014-001 | 9 | 1 | 5 | 45 |
| BNCI2014-004 | 9 | 1 | 5 | 45 |
| Cho2017 | 52 | 5 | 5 | 1,300 |
| PhysioNet MI | 54 | 3 | 5 | 810 |
| **Total per model** |  |  |  | **2,240** |
| **43-model total** |  |  |  | **96,320** |

This count is only the common-recipe track. Author-faithful, local-procedure,
geometry-adaptation, HemiQ, native-transfer, deterministic-control, and any
candidate screen belong to separate protocol tables and separate run roots.

## Timing evidence

Two existing, score-independent timing sources were used.

1. The completed local 43-model journal contains one start/completion pair for
   each model, with 40 jobs per invocation. Its SHA-256 is
   `d9681d6027806b67a934e4fcaa59e7c9379d025bd75f277e50f45f6a29cd4133`.
2. The frozen CHSD opened-development runs contain 596 records with
   `timing.total_seconds`. A canonical sorted sequence of
   `(model, dataset, subject, fold, seed, total_seconds)` has SHA-256
   `2bb9682a343c8ac7096a910c5327443dbd46f7d895ce4d2d266f0d980e830fab`.
   These records total 8,138.837968 seconds.

The local journal was produced on different hardware. Its per-job model
timings were scaled by the observed A5000/local ratio for
`cardinal_fbc_micro_extended`. Dataset multipliers were then estimated from
that model's A5000 records. TCFormer used its own observed A5000 mean for each
dataset because its compute profile differs substantially from the compact
models.

This extrapolation estimates approximately:

- 115.7 aggregate GPU-hours for the 43-model common track;
- 38.6 hours at perfect utilization across three A5000 GPUs;
- roughly 58-96 wall-clock hours after a provisional 1.5-2.5 multiplier for
  scheduling imbalance, initialization, analysis, contention, retries, and
  conservative pauses.

The planning range is therefore about 2-4 days of wall time after preflight.
It is not yet measured on the formal runner. TCFormer alone accounts for an
estimated 23.1 GPU-hours and is the largest known contributor. A pilot must
report medians and high quantiles by model and dataset before this estimate is
used operationally.

At the user's observed average of approximately 200 W per active GPU, the
115.7 GPU-hour estimate corresponds to approximately 23.1 kWh of GPU energy
for the common track. This excludes CPU, memory, storage, cooling, conversion
losses, idle intervals, retries, and every non-common track. The GPU-only
electricity cost is therefore `23.1 × local_price_per_kWh`; for example, it is
about $4.63 at $0.20/kWh. A wall-socket measurement is required for a defensible
whole-system energy or cost report.

## Storage evidence

At the 2026-07-29 capacity check:

- `/dev/nvme0n1p2` had 77 GiB available;
- the mandatory low-water mark was 50 GiB;
- usable headroom above the floor was therefore 27 GiB;
- the project scratch tree occupied approximately 7.7 GiB;
- the isolated UV environment accounted for approximately 5.6 GiB;
- the opened dataset cache accounted for approximately 480 MiB.

A 200-directory sample of existing atomic record trees averaged 13.08 KiB and
had a maximum of 16 KiB. Naive linear extrapolation for 96,320 common jobs is
about 1.20 GiB. Logs, filesystem block rounding, quarantine evidence,
analysis artifacts, and non-common tracks require additional headroom, but
the observed record size is compatible with the current 27 GiB margin.

No cleanup of other users' files is authorized. If the 50 GiB floor is
approached, workers must stop cleanly and the project owner must choose a new
project-owned result volume or explicitly authorize removal of identified,
recoverable project artifacts.

## Required measured pilot

After runner approval and before the full grid:

1. run the exact CUDA/source/environment preflight;
2. execute a small, prespecified, score-blind timing pilot spanning the fastest,
   median, and slowest architecture families and all five datasets;
3. record fit, inference, publication, retry, peak GPU memory, and output-byte
   distributions without changing training hyperparameters;
4. recompute per-track wall-time and storage confidence intervals;
5. confirm that GPU 3 remained unreserved and the disk floor was never crossed.

The pilot may refine scheduling only. It must not be used to tune models,
select subjects, change the statistical plan, or inspect held-out outcomes.
