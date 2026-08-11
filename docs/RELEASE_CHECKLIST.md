# Clean benchmark release checklist

## Status and boundary

This is the promotion contract for the clean benchmark project at
`/home/hanafy/neuralnetwork`. The user has authorized scratch research,
read-only preservation of prior artifacts, isolated UV environment work, and
eventual construction of that clean project. The final project root remains
unpromoted while source repairs and independent runner audits continue.

Promotion is blocked until every required item below is complete. A checked box
must point to immutable evidence; it must not mean that someone remembers
running the step.

## Exact minimum source manifest

The current formal common grid, controls, candidate-lineage screens, local
outer-refit procedures, native-transfer procedures, registry, and reference
training callables have this minimum Python source closure:

```text
deepnet/__init__.py
deepnet/augment.py
deepnet/baselines.py
deepnet/cameo_net.py
deepnet/config.py
deepnet/data.py
deepnet/local_outer_refit_benchmark.py
deepnet/orbit_transport_net.py
deepnet/parity_fuse_net.py
deepnet/parity_net.py
deepnet/spd.py
deepnet/tangent_anchor.py
ieee_mi/__init__.py
ieee_mi/baselines.py
ieee_mi/benchmark.py
ieee_mi/chsd.py
ieee_mi/chsd_conditioned.py
ieee_mi/chsd_direct.py
ieee_mi/chsd_hybrid.py
ieee_mi/chsd_joint.py
ieee_mi/conditioned_analysis.py
ieee_mi/conditioned_screen.py
ieee_mi/config.py
ieee_mi/control_grid.py
ieee_mi/control_grid_analysis.py
ieee_mi/data.py
ieee_mi/development_analysis.py
ieee_mi/development_screen.py
ieee_mi/full_grid.py
ieee_mi/full_grid_analysis.py
ieee_mi/hemiq_v2_model.py
ieee_mi/local_geometric_controls.py
ieee_mi/model_registry.py
ieee_mi/models.py
ieee_mi/native_fbms_full_grid_audit.py
ieee_mi/native_fbms_full_grid_gate.py
ieee_mi/native_fbms_full_grid_ops.py
ieee_mi/native_fbms_pretraining_cli.py
ieee_mi/native_fbms_transfer.py
ieee_mi/native_fbms_transfer_audit.py
ieee_mi/native_fbms_transfer_gate.py
ieee_mi/native_pretraining.py
ieee_mi/native_pretraining_cli.py
ieee_mi/native_transfer.py
ieee_mi/native_transfer_audit.py
ieee_mi/native_transfer_gate.py
ieee_mi/reference_training.py
ieee_mi/robustness_analysis.py
ieee_mi/robustness_screen.py
ieee_mi/tcformer_source.py
ieee_mi/track_matrix.py
ieee_mi/training.py
ieee_mi/verification/__init__.py
ieee_mi/verification/verify_native_fbms_full_grid.py
```

- [ ] Recompute this closure after the final candidate is selected and after
      every authorized source repair.
- [ ] Record byte size and SHA-256 for every path.
- [ ] Confirm that every imported local module is present. In particular,
      `deepnet/config.py` must be in the local outer-refit runner's explicit
      source manifest.
- [ ] Do not describe this closure as all 70 registry entries. Any model whose
      matrix cell is blocked by a missing adapter remains explicitly blocked
      until its implementation, cache/split-bound runner, tests, and provenance
      closure are added and reviewed.

## Exact minimum test manifest

```text
deepnet/tests/test_baselines.py
deepnet/tests/test_cameo_net.py
deepnet/tests/test_data.py
deepnet/tests/test_local_outer_refit_benchmark.py
deepnet/tests/test_orbit_transport_net.py
deepnet/tests/test_parity_fuse_net.py
deepnet/tests/test_parity_net.py
deepnet/tests/test_spd.py
ieee_mi/tests/test_baselines.py
ieee_mi/tests/test_benchmark.py
ieee_mi/tests/test_chsd.py
ieee_mi/tests/test_chsd_conditioned.py
ieee_mi/tests/test_chsd_direct.py
ieee_mi/tests/test_chsd_hybrid.py
ieee_mi/tests/test_chsd_joint.py
ieee_mi/tests/test_conditioned_analysis.py
ieee_mi/tests/test_conditioned_screen.py
ieee_mi/tests/test_control_grid.py
ieee_mi/tests/test_control_grid_analysis.py
ieee_mi/tests/test_data.py
ieee_mi/tests/test_development_analysis.py
ieee_mi/tests/test_development_screen.py
ieee_mi/tests/test_full_grid.py
ieee_mi/tests/test_full_grid_analysis.py
ieee_mi/tests/test_geometry_regression.py
ieee_mi/tests/test_hemiq_v2_model.py
ieee_mi/tests/test_independent_full_grid_verify.py
ieee_mi/tests/test_local_geometric_controls.py
ieee_mi/tests/test_model_registry.py
ieee_mi/tests/test_models.py
ieee_mi/tests/test_native_fbms_full_grid_ops.py
ieee_mi/tests/test_native_fbms_full_grid_protocol.py
ieee_mi/tests/test_native_fbms_pretraining_cli.py
ieee_mi/tests/test_native_fbms_transfer.py
ieee_mi/tests/test_native_fbms_transfer_audit.py
ieee_mi/tests/test_native_pretraining.py
ieee_mi/tests/test_native_pretraining_cli.py
ieee_mi/tests/test_native_transfer.py
ieee_mi/tests/test_native_transfer_audit.py
ieee_mi/tests/test_native_transfer_gate.py
ieee_mi/tests/test_reference_training.py
ieee_mi/tests/test_robustness_analysis.py
ieee_mi/tests/test_robustness_screen.py
ieee_mi/tests/test_tcformer_source.py
ieee_mi/tests/test_track_matrix.py
ieee_mi/tests/test_training.py
```

- [ ] Add release tests for the source manifest, provenance generator,
      outcome-free local-procedure configurations, TCFormer source identity,
      third-party notices, and the unified runner.
- [ ] Run tests only from the isolated `.venv-test` environment declared in
      [the UV contract](UV_REPRODUCIBILITY.md).

## Required support and documentation manifest

The following release-level files must exist before promotion. Several are not
present yet and must not be represented as complete:

```text
.gitignore
.python-version
README.md
LICENSE
CITATION.cff
THIRD_PARTY_NOTICES.md
pyproject.toml
uv.lock
docs/CARDINAL_FIELD_THEORY.md
docs/BENCHMARK_STATISTICAL_ANALYSIS_PLAN.md
docs/BENCHMARK_RUNTIME_AND_STORAGE_BUDGET.md
docs/CHSD_NOVELTY_AUDIT.md
docs/CONTROL_GRID.md
docs/CONTROL_GRID_ANALYSIS.md
docs/DATASETS_AND_LICENSES.md
docs/EXPERIMENT_LEDGER.md
docs/FULL_GRID_ANALYSIS.md
docs/INHOUSE_MODEL_METHODS.md
docs/MODEL_REGISTRY.md
docs/NATIVE_FBMS_FULL_GRID_OPS.md
docs/NON_COMMON_TRACK_EXECUTION.md
docs/RELEASE_CHECKLIST.md
docs/ROBUSTNESS_SCREEN.md
docs/THIRD_PARTY_LICENSE_AUDIT.md
docs/UV_REPRODUCIBILITY.md
reports/benchmark_report.pdf
results/benchmark_summary.csv
results/benchmark_summary.json
results/model_dataset_coverage.csv
results/statistical_tests.csv
scripts/build_report.py
scripts/generate_provenance.py
scripts/robustness_cuda_smoke.py
scripts/run_all.py
scripts/vendor_tcformer.py
scripts/verify_release.py
provenance/environment.json
provenance/environment.cdx.json
provenance/requirements.locked.txt
provenance/release_manifest.json
provenance/third_party_sources.json
provenance/SHA256SUMS
```

- [ ] The final `pyproject.toml` and `uv.lock` are newly generated benchmark
      files, not copies of the acquisition/GUI project metadata.
- [ ] `.python-version` is exactly `3.12.13`.
- [ ] A project owner chooses the project license; do not infer one from
      dependencies or from an IEEE submission.
- [ ] `CITATION.cff` contains the final authorship and citation metadata.
- [ ] `scripts/run_all.py` provides one documented entry point but preserves
      the scientific separation among common, author-faithful, procedure,
      transfer, and control tracks. It must not merge unlike scores.
- [ ] The five files under `results/` and `reports/` above are regenerated
      curated aggregates and documentation, not copied development artifacts.
      Their atomic inputs remain in an external immutable result root.

## Outcome-free local-procedure configurations

Prior result JSON files must not be runtime configuration inputs. Extract only
the prespecified architecture and training parameters into:

```text
configs/local_procedures/cameo_v1.json
configs/local_procedures/hemiparity_v1.json
configs/local_procedures/parity_fuse_v1.json
configs/local_procedures/orbit_v3.json
configs/hemiq/hemiq_field_harmonized_v2.json
```

- [ ] Each file has a schema version, stable model ID, explicit parameter
      values, byte size, and SHA-256.
- [ ] No file contains accuracy, loss, predictions, selected test outcomes,
      subject-level scores, or other outcome-derived values.
- [ ] The runner no longer opens development-result JSON at execution time.
- [ ] Tests prove that unknown keys, missing keys, a digest mismatch, or
      outcome-bearing keys fail closed.
- [ ] The HemiQ harmonized-v2 configuration remains separate from the four
      general binary procedures, changes only the declared sampling,
      device, and seed fields, and never opens the historical result manifest
      at runtime.

## TCFormer provenance and license

The expected upstream repository is `https://github.com/Altaheri/TCFormer` at
commit `74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5`. The selected release strategy is
a reviewed compact vendored runtime snapshot containing the upstream
`LICENSE`, `models/tcformer.py`, `models/modules.py`,
`models/channel_group_attention.py`, and
`utils/weight_initialization.py`.

- [x] The adapter verifies an exact canonical `SOURCE.json` and independently
      compiled-in byte sizes and SHA-256 values for all five files. It no
      longer depends on Git metadata. The verifier and its adversarial unit
      tests are `ieee_mi/tcformer_source.py` and
      `ieee_mi/tests/test_tcformer_source.py`.
- [ ] Preserve the upstream MIT license and citation.
- [ ] Record the exact Braindecode wheel version and its distributed
      `LICENSE`/`NOTICE`; do not assume one blanket license for separately
      licensed components.
- [ ] Any adapter change occurs before the final source freeze and is followed
      by CPU, CUDA, determinism, and formal-runner tests.

## Dataset and privacy review

- [ ] `docs/DATASETS_AND_LICENSES.md` records, for every dataset: canonical
      identifier/version, source URL, citation, license or access terms,
      redistribution permission, task/classes, subject boundary, MOABB loader
      and version, preprocessing profile, and cache-manifest digest.
- [ ] Local data records the controlling consent/IRB/data-use boundary and
      whether code and aggregate statistics may be released.
- [ ] No participant identifiers, credentials, private paths, protected
      metadata, raw EEG, or derived trial-level data are committed.
- [ ] Raw downloads and harmonized caches stay in approved external storage.
      The project records manifests and hashes, not redistributed data, unless
      the applicable terms explicitly permit redistribution.

## Clean source and immutable identity

- [ ] The final candidate and its frozen configuration are selected before the
      release source manifest is generated.
- [ ] Legacy active references to `requirements-cu128.txt` are removed from
      current runners or explicitly isolated as historical provenance.
- [ ] Every documentation link and command is checked against the promoted
      tree.
- [ ] `git status --porcelain=v1` is empty.
- [ ] Record the clean commit ID, source-manifest digest, registry digest, and
      TCFormer identity before creating any immutable run plan.
- [ ] A later source, dependency, configuration, dataset, or environment
      change creates a new commit, lock digest, provenance manifest, and run
      root. Never append mixed identities to an old plan.

## Fresh UV environments

Follow `docs/UV_REPRODUCIBILITY.md`. The required environment split is:

- `.venv`: formal runtime, `--no-default-groups`;
- `.venv-test`: runtime plus `--group test`;
- `.venv-docs`: runtime plus `--group docs`.

- [ ] Verify the recorded UV executable is version 0.12.0 and hash the
      executable. Do not install into or alter system Python.
- [ ] Pin UV-managed CPython 3.12.13 locally.
- [ ] Generate a fresh lock and pass `uv lock --check`.
- [ ] Review the lock's package sources and hashes; Torch must resolve to the
      explicit cu124 index and runtime version `2.6.0+cu124`.
- [ ] Perform the first exact online sync into a fresh project `.venv` with
      `uv sync --frozen --no-default-groups`.
- [ ] Pass `uv pip check`, import checks, interpreter isolation checks, and
      Torch/CUDA/cuDNN identity checks.
- [ ] Allocate a uniquely named, previously unused scratch directory and run a
      second exact sync with `UV_OFFLINE=1`, the reviewed shared cache, the
      reviewed lock, and `--no-default-groups`.
- [ ] Run checks from the offline-rebuilt interpreter. A missing cached
      artifact is a failed offline proof, not permission to weaken the lock or
      install ad hoc.
- [ ] Preserve the approved UV cache, or an immutable cache snapshot and
      checksum inventory, for as long as offline recreation is required.

## CPU and CUDA acceptance

- [ ] The offline-created `.venv-test` passes the exact test manifest above.
- [ ] CPU controls run with CUDA hidden, `threadpoolctl`, and every declared
      thread cap enforced.
- [ ] One approved idle physical GPU UUID is exposed for CUDA tests; no ordinal
      or all-GPU assumption is accepted.
- [ ] Torch reports `2.6.0+cu124`, CUDA build 12.4, the reviewed cuDNN value,
      one visible device, the expected GPU identity, and finite outputs and
      gradients.
- [ ] Candidate and roster CUDA smoke tests are bitwise repeatable where the
      protocol requires bitwise determinism.
- [ ] Formal plan/status/audit dry runs pass without editing source,
      dependencies, caches, or the remote host's system configuration.

## Provenance, SBOM, checksums, and reports

- [ ] A checked-in provenance generator writes the environment, source,
      dependency, hardware, cache, split, plan, command, and third-party fields
      required by `docs/UV_REPRODUCIBILITY.md`.
- [ ] Export the formal no-default-groups dependency view to
      `provenance/requirements.locked.txt`.
- [ ] Export a CycloneDX 1.5 SBOM to
      `provenance/environment.cdx.json`.
- [ ] `provenance/third_party_sources.json` records source URLs, pins,
      licenses, and vendored byte hashes.
- [ ] `provenance/SHA256SUMS` covers project metadata, source, configurations,
      documentation, SBOM, aggregate result tables, report source, and rendered
      PDF.
- [ ] Final statistics are regenerated from immutable atomic artifacts. The
      report records coverage and blocked/N/A cells rather than silently
      dropping them.
- [ ] Render and visually inspect every page of the final PDF in the separate
      docs environment before freezing its checksum.

## Files and state forbidden from the clean project

Do not copy:

```text
.venv/
.venv-test/
.venv-docs/
__pycache__/
*.pyc
deepnet/cache/
deepnet/checkpoints/
deepnet/results/
ieee_mi/results/
raw EEG or downloaded dataset archives
scratch logs, lock files, temporary journals, and worker leases
credentials, SSH material, tokens, or user-specific shell configuration
```

Curated aggregate tables and a rendered report may be added only after their
immutable source artifacts, generation command, provenance, and checksums are
recorded. Atomic results stay in a separately managed immutable result root or
archive with a checked manifest.

## Promotion decision

Promotion is allowed only when all applicable checks above pass under one clean
commit and one reviewed lock. Copying a partially prepared tree and planning to
repair it in place is not an acceptable release process.
