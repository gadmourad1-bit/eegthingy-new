# CHSD disjoint-subject robustness amendment

The retained token `eeg-mi-cache-v2` is a versioned cache-schema identifier,
not the active package or distribution name.

## Status and evidential meaning

This stage is a **transparent post-v3 exploratory amendment**. It was defined
after the conditioned v3 screen returned `kill_conditioned_family`. It does
not rewrite that result, does not convert opened development data into
confirmation data, and cannot support a state-of-the-art claim.

The amendment asks one narrow question: does the fixed
`chsdnet_conditioned_005` configuration remain competitive on the opened
development subjects that were not used in either the v2 or v3 screen? A pass
only permits a later prespecified stage. A failure stops CHSD promotion under
the encoded rule.

No workload is launched by importing either module or by the source changes
that introduced this stage.

## Frozen historical lineage

The immutable robustness plan must verify and bind all three completed
artifacts:

| Artifact | Required identity |
|---|---|
| Corrected v2 plan | `57298f7c18b8841e741d72194cb601faf3ac71cb81b1209b1c3964ddf43c0fc6` |
| Conditioned v3 plan | `1359ea1913d76d624272f8194dfb361e5d919892783003e1f06e89d179f0e784` |
| Conditioned v3 analysis file | `69d616c2e2fe93a9b7f4464c31d8e3e8be048be39cb505eaea89023372135f0b` |

The v3 analysis must say `kill_conditioned_family` and must have no selected
model. The new manifest stores `prior_decision_is_rewritten: false`.

## Cohort and exact cardinality

Only already-opened `eeg-mi-cache-v2/harmonized` development caches are
eligible. Fold 0 and seed 7 are fixed.

| Dataset | Robustness subjects | Count | Subjects excluded because v2/v3 used them |
|---|---|---:|---|
| `local_exp4` | 3, 5, 6, 8, 10 | 5 | 1, 4, 7 |
| `bnci2014_001` | 2, 3, 4, 6, 7, 8 | 6 | 1, 5, 9 |
| `bnci2014_004` | 2, 3, 4, 6, 7, 8 | 6 | 1, 5, 9 |
| `cho2017` | 1-52 except 1, 18, 35, 52 | 48 | 1, 18, 35, 52 |
| `physionet_mi` | 1-54 except 1, 18, 36, 54 | 50 | 1, 18, 36, 54 |

This is exactly 115 subjects. The four fixed models are:

1. `chsdnet_conditioned_005`
2. `cardinal_fbc_micro_extended`
3. `fbcnet`
4. `tcformer`

The Cartesian product is therefore exactly 460 jobs. Each of four static
workers owns one model and all 115 subjects for that model.

The Cho fold-0 implementation derives two equal class sequences from the
class-contiguous cache contract and splits each sequence into five contiguous
blocks. It supports the usual 200-trial caches and the 240-trial caches for
S7, S9, and S46. Only training and validation row indices are returned or
materialized. The worker does not use ``numpy.load(npz)["x"][rows]`` because
that expression first decodes the complete compressed member. It instead
parses each NPY header and streams only the selected C-order rows from the ZIP
member; excluded EEG and label rows are skipped and are never allocated as
NumPy arrays.

## Frozen decision rule

Balanced accuracy is averaged over subjects within each dataset and then
equally over the five datasets. The candidate passes only when **every**
condition below is true:

1. Its equal-dataset balanced accuracy is no more than 1.0 percentage point
   below the strongest of the three references.
2. Its equal-dataset balanced accuracy is strictly greater than TCFormer.
3. Against each reference separately, its dataset-mean balanced-accuracy
   delta is nonnegative on at least three of five datasets.
4. Against each reference separately, no dataset-mean deficit is worse than
   5.0 percentage points.
5. Against each reference separately, strict subject-level wins divided by
   all 115 paired subjects is at least 0.45. Ties are not wins.

If any condition fails, the sole decision is `stop_chsd_promotion`. Otherwise
it is `pass_robustness_gate`. The rule is serialized in the plan, and
`robustness_analysis.py` is itself included in the plan's source hash closure
so the implementation cannot be changed after results exist.

Every inclusive floating-point boundary uses the frozen `1e-12` tolerance in
the permissive direction. A strict subject win or strict TCFormer exceed must
be greater than `1e-12`; values within that tolerance are ties. Tests cover
the exact 1.0-point and -5.0-point boundaries and values just outside them.

## Reproducible shared-workstation execution

Use only the existing UV-managed virtual environment. Do not run `pip` as
root, install system packages, or run `uv sync` against the legacy scratch
lock. The examples below intentionally use `--active --no-sync`.

The plan stores both the inherited core-package/CUDA/driver identity and the
complete sorted output of the read-only `uv pip freeze --python ...` command,
plus its SHA-256. This closes over TCFormer's direct dependencies, including
`einops`; each worker and the analyzer reproduce the same freeze and fail
closed on drift. The command fingerprints the existing environment and does
not install, remove, or synchronize packages.

```bash
cd /home/hanafy/scratchpad/eeg_novel_20260729/source
export PATH=/home/hanafy/.local/bin:$PATH
export UV_CACHE_DIR=/home/hanafy/scratchpad/eeg_novel_20260729/uv-cache
export EEG_MI_TCFORMER_ROOT=/home/hanafy/scratchpad/eeg_novel_20260729/source/third_party/TCFormer
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source /home/hanafy/scratchpad/eeg_novel_20260729/.venv-uv/bin/activate
```

First query the machine and use only four idle full physical UUIDs:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory --format=csv
```

Freeze the plan once. Replace the analysis path with the exact completed v3
`analysis.json` whose file hash is listed above.

```bash
uv run --active --no-sync python -m benchmark.robustness_screen plan \
  --run-root /home/hanafy/scratchpad/eeg_novel_20260729/runs/chsd_disjoint_robustness_v1 \
  --cache-root /home/hanafy/scratchpad/eeg_novel_20260729/data_cache \
  --reference-v2-run-root /home/hanafy/scratchpad/eeg_novel_20260729/runs/chsd_validation_screen_v2 \
  --v3-run-root /home/hanafy/scratchpad/eeg_novel_20260729/runs/chsd_conditioned_screen_v3 \
  --v3-analysis-path /absolute/path/to/completed/v3/analysis.json \
  --gpu-uuid GPU-UUID-0 \
  --gpu-uuid GPU-UUID-1 \
  --gpu-uuid GPU-UUID-2 \
  --gpu-uuid GPU-UUID-3
```

Launch one process per worker only after the plan succeeds. Each process must
see exactly its plan-bound UUID:

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID-0 uv run --active --no-sync python -m benchmark.robustness_screen worker \
  --run-root /home/hanafy/scratchpad/eeg_novel_20260729/runs/chsd_disjoint_robustness_v1 \
  --cache-root /home/hanafy/scratchpad/eeg_novel_20260729/data_cache \
  --worker-index 0 --gpu-uuid GPU-UUID-0
```

Repeat for worker indices 1-3 and their corresponding UUIDs. The fixed mapping
is candidate, CardinalFBC, FBCNet, TCFormer. Every new job repeats:

- the exact source, external TCFormer, UV environment, and driver checks;
- the full physical UUID and one-visible-GPU check;
- rejection of any foreign compute PID on that GPU;
- the 50-GiB free-space low-water guard.

An existing valid JSON record is resumed without rerunning. An existing
corrupt or plan-inconsistent record causes a hard failure and is never
overwritten.

Inspect progress:

```bash
uv run --active --no-sync python -m benchmark.robustness_screen status \
  --run-root /home/hanafy/scratchpad/eeg_novel_20260729/runs/chsd_disjoint_robustness_v1
```

Only after status reports 460 valid records may the analysis run:

```bash
uv run --active --no-sync python -m benchmark.robustness_analysis \
  --run-root /home/hanafy/scratchpad/eeg_novel_20260729/runs/chsd_disjoint_robustness_v1 \
  --output-dir /home/hanafy/scratchpad/eeg_novel_20260729/runs/chsd_disjoint_robustness_v1_analysis
```

The analyzer never opens an EEG cache. It validates all expected record paths,
schemas, plan hashes, cache hashes, split hashes, source/environment bindings,
training settings, GPU UUIDs, PID guards, disk guards, and metric ranges before
it computes or writes a decision. It rejects every unexpected file and every
symlink under the records tree.

The deterministic output set is:

- `analysis.json`
- `dataset_scores.csv`
- `overall_scores.csv`
- `paired_comparisons.csv`
- `REPORT.md`

Existing different output files are treated as corruption and are not
overwritten. The output directory must be a true sibling of the immutable run
directory; unrelated paths, descendants, symlinks, and unexpected pre-existing
artifacts are rejected.

## Completed outcome

The run completed on 2026-07-29. Each worker produced exactly 115 records,
emitted one normal completion summary, logged zero errors, and exited. The
frozen status command validated 460/460 records with zero missing and zero
corrupt records before analysis. No aggregate score was inspected earlier.

The immutable analysis decision was **`stop_chsd_promotion`**:

| Rank | Model | Equal-dataset balanced accuracy |
|---:|---|---:|
| 1 | `cardinal_fbc_micro_extended` | 74.087% |
| 2 | `fbcnet` | 73.628% |
| 3 | `chsdnet_conditioned_005` | 73.543% |
| 4 | `tcformer` | 72.862% |

The equal-subject dataset means that formed those fixed-suite macros were:

| Dataset | CHSD-conditioned 0.05 | CardinalFBC micro ext. | FBCNet | TCFormer |
|---|---:|---:|---:|---:|
| Local Exp4 | 94.000% | 94.333% | 94.000% | 95.000% |
| BNCI2014-001 | 69.792% | 70.833% | 72.222% | 69.792% |
| BNCI2014-004 | 73.542% | 73.854% | 72.396% | 72.917% |
| Cho2017 | 69.201% | 70.486% | 69.592% | 68.854% |
| PhysioNet MI | 61.179% | 60.929% | 59.929% | 57.750% |

The candidate met the within-one-point and strictly-above-TCFormer clauses. It
failed the required broad paired performance against the two stronger
references: 1/5 nonnegative datasets and a 13.91% strict subject win rate
against CardinalFBC micro extended; 3/5 and 38.26% against FBCNet. Its
comparison against TCFormer passed.

Evidence hashes:

- plan:
  `1e4009ae67e7b11ae60c76c5d110ec49a27246422d2617c9c818199ca799b9ae`;
- runner:
  `93bc0c7828b78688620316f1612708a1a8fdb2f369612ea5da573931cc47b4d4`;
- analyzer:
  `8cf2e388401679bac662bc5a470807edb99b9a805b93f8c6f29145f4ff02f0fa`;
- `analysis.json`:
  `ba834b159dae53781e44f19682a6c032d4674f46fb5b0f5c752fedd9dc84e943`;
- `REPORT.md`:
  `558dca9617d56f7fd533e3d6a9ab05c6d49c5ada4efb544f2224b87c467a8b25`;
- canonical sorted-path/SHA-256 tree of the immutable raw run:
  `d64fa73efb6c27c7ffda78c9bb62959d863290464a8c9598df0c8170fad6bee2`;
  and
- canonical sorted-path/SHA-256 tree of the five-file analysis output:
  `fdf3d3c31481b584adbe388184285787fbd22f44a166954b86d4f02bbe28e55d`.

This is a negative opened-development result. It preserves the v3 kill
decision, excludes CHSD-conditioned from the formal common grid, and supports
no state-of-the-art, confirmation, or clinical claim.
