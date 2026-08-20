from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from benchmark import procedure_grid, procedure_grid_analysis


def _tiny_plan() -> dict[str, object]:
    spec = procedure_grid.PROCEDURE_BY_ID["architecture.cameo"]
    inventory = procedure_grid._config_inventory()
    inventory["available"] = [
        value
        for value in inventory["available"]
        if value["stable_id"] == spec.stable_id
    ]
    return procedure_grid.assemble_plan(
        dataset_contracts={
            "tiny": {
                "subjects": [1],
                "folds": [0],
                "n_classes": 2,
                "events": ["left", "right"],
                "protocol": "synthetic",
                "preprocessing": {"sfreq_hz": 128.0, "n_times": 256},
            }
        },
        cache_identity={
            "tiny:s001": {
                "array_sha256": "a" * 64,
                "trial_count": 8,
                "n_channels": 3,
            }
        },
        split_identity={
            "tiny:s001:f00": procedure_grid._one_split_identity(
                dataset="tiny",
                subject=1,
                fold=0,
                cache_array_sha256="a" * 64,
                trial_count=8,
                train_rows=[0, 1],
                validation_rows=[2, 3],
                test_rows=[4, 5, 6, 7],
            )
        },
        source_identity={"runner.py": "b" * 64},
        environment_identity={"uv_version": "uv test"},
        config_inventory=inventory,
        procedure_ids=(spec.stable_id,),
        seeds=(7,),
        cpu_threads=1,
        publishable=False,
    )


def _metadata(plan: dict[str, object], job: procedure_grid.Job) -> dict[str, object]:
    expected_config = plan["config_inventory"]["available"][0]
    digests = iter("3456789abcdef" * 8)

    def manifest(shape):
        return {
            "shape": list(shape),
            "dtype": "<f4",
            "sha256": next(digests) * 64,
        }

    preprocessing = {
        name: manifest([1])
        for name in (
            "raw_mean",
            "raw_std",
            "anchor_log_reference",
            "anchor_scaler_mean",
            "anchor_scaler_scale",
            "anchor_logistic_coef",
            "anchor_logistic_intercept",
        )
    }

    def pair(count):
        return {
            "original": manifest([count, 24]),
            "reflected": manifest([count, 24]),
        }

    return {
        "cache_array_sha256": "a" * 64,
        "channels": ["C3", "Cz", "C4"],
        "split": copy.deepcopy(plan["split_identity"]["tiny:s001:f00"]),
        "frozen_config": {
            "schema": "eeg-mi-local-procedure-config-v2",
            "version": 1,
            "model": job.stable_id,
            "config_file": expected_config["config_path"],
            "config_file_bytes": expected_config["config_bytes"],
            "config_file_sha256": expected_config["config_sha256"],
            "settings_sha256": expected_config["settings_sha256"],
            "config": copy.deepcopy(expected_config["settings"]),
            "runtime_overrides": {"seed": job.seed, "device": "cpu"},
        },
        "view_pipeline": {
            "schema": procedure_grid.VIEW_SCHEMA,
            "mirror_index": [2, 1, 0],
            "raw": {
                "original": manifest([8, 3, 256]),
                "reflected": manifest([8, 3, 256]),
            },
            "four_band_spd": {
                "original": manifest([8, 4, 3, 3]),
                "reflected": manifest([8, 4, 3, 3]),
            },
            "selection_tangent": {
                "train": pair(2),
                "validation": pair(2),
            },
            "refit_tangent": {
                "source": pair(4),
                "test": pair(4),
            },
        },
        "fit": {
            "seed_installed_before_construction": True,
            "selected_epoch_index": 1,
            "selection_epochs_run": 2,
            "selection_decision": {
                "selected_mixture": "geo",
                "selected_rho": 0.0,
            },
            "selection_start_reset_sha256": "1" * 64,
            "refit_start_reset_sha256": "1" * 64,
            "refit_state_sha256": "2" * 64,
            "reset_hash_exclusions": [],
            "reset_verified": True,
            "selected_epoch_count": 2,
            "refit_epochs_run": 2,
            "parameter_count": 123,
            "trainable_parameter_count": 123,
            "selection_preprocessing": copy.deepcopy(preprocessing),
            "refit_preprocessing": copy.deepcopy(preprocessing),
        },
        "protocol": {
            "selection_preprocessing_rows": "train_only",
            "selection_optimization_rows": "train_only",
            "selection_decision_rows": "validation_only",
            "refit_preprocessing_rows": "train_plus_validation",
            "refit_optimization_rows": "train_plus_validation",
            "test_use": "single_predict_proba_call_only",
            "test_performance_computed": False,
            "transductive_calibration": False,
        },
        "timing_seconds": {
            "selection_fit": 1.0,
            "refit_fit": 2.0,
            "test_inference": 0.04,
            "job_total": 3.04,
        },
        "runtime": {
            "cpu_threads": 1,
            "torch_interop_threads": 1,
            "device": "cpu",
            "physical_gpu_uuid": None,
            "cuda_visible_devices": None,
            "thread_environment": {
                name: "1" for name in procedure_grid.THREAD_ENVIRONMENT_VARIABLES
            },
            "cuda_peak_memory_bytes": 0,
        },
    }


def _complete_tiny_run(tmp_path: Path) -> tuple[dict[str, object], np.ndarray]:
    plan = _tiny_plan()
    plan = procedure_grid.write_or_validate_plan(tmp_path, plan)
    job = next(procedure_grid.iter_jobs(plan))
    resource = {
        "safe": True,
        "gpu": {
            "safe": True,
            "gpu": "test",
            "utilization_percent": 0.0,
            "memory_used_mib": 0.0,
            "own_compute_memory_mib": 0.0,
            "foreign_processes": [],
            "reason": "test",
            "gpu_uuid": None,
            "pci_bus_id": None,
            "name": None,
        },
        "disk": {
            "safe": True,
            "path": str(tmp_path),
            "free_bytes": 100 * 1024**3,
            "total_bytes": 200 * 1024**3,
            "free_gib": 100.0,
            "minimum_free_gib": procedure_grid.DEFAULT_MIN_FREE_GIB,
            "reason": "test",
        },
        "reason": "test",
    }
    claim = procedure_grid.acquire_claim(
        tmp_path,
        plan,
        job,
        resource_guard=resource,
        recover_stale=False,
    )
    probabilities = np.asarray(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
        dtype=np.float64,
    )
    procedure_grid.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=_metadata(plan, job),
        test_rows=np.arange(4, 8, dtype=np.int64),
        probabilities=probabilities,
    )
    procedure_grid.release_claim(claim)
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    return plan, labels


def _computed_tiny_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, procedure_grid_analysis.AnalysisResult]:
    run_root = tmp_path / "run"
    plan, labels = _complete_tiny_run(run_root)
    cache_root = tmp_path / "cache"

    def bound_labels(plan_arg, selected_cache):
        assert plan_arg == plan
        assert selected_cache == cache_root
        return (
            {("tiny", 1): labels.copy()},
            [
                {
                    "kind": "cache",
                    "identity": "tiny:s001",
                    "sha256_a": "a" * 64,
                    "sha256_b": "b" * 64,
                    "sha256_c": "c" * 64,
                }
            ],
        )

    monkeypatch.setattr(
        procedure_grid_analysis, "_bound_labels", bound_labels
    )
    result = procedure_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=cache_root,
        require_formal=False,
        verify_runtime=False,
    )
    return run_root, cache_root, result


def test_metric_definitions_cover_probability_and_classification_outputs() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    probabilities = np.asarray(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
        dtype=np.float64,
    )
    metrics = procedure_grid_analysis.classification_metrics(labels, probabilities)
    assert set(metrics) == set(procedure_grid_analysis.METRIC_NAMES)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["chance_normalized_balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["ovr_macro_auroc"] == 1.0
    assert metrics["nll"] > 0.0
    assert metrics["multiclass_brier"] > 0.0


def test_incomplete_grid_fails_before_label_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _tiny_plan()
    plan = procedure_grid.write_or_validate_plan(tmp_path, plan)
    touched = {"labels": False}

    def forbidden(*args, **kwargs):
        touched["labels"] = True
        raise AssertionError("label loader must not run")

    monkeypatch.setattr(procedure_grid_analysis, "_bound_labels", forbidden)
    with pytest.raises(
        procedure_grid_analysis.ProcedureAnalysisError,
        match="refusing label access",
    ):
        procedure_grid_analysis.compute_analysis(
            run_root=tmp_path,
            cache_root=tmp_path / "cache",
            require_formal=False,
            verify_runtime=False,
        )
    assert touched["labels"] is False


def test_tiny_injected_analysis_aggregates_and_publishes_only_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan, labels = _complete_tiny_run(run_root)

    def bound_labels(plan_arg, cache_root):
        assert plan_arg == plan
        return (
            {("tiny", 1): labels.copy()},
            [
                {
                    "kind": "cache",
                    "identity": "tiny:s001",
                    "sha256_a": "a" * 64,
                    "sha256_b": "b" * 64,
                    "sha256_c": "c" * 64,
                }
            ],
        )

    monkeypatch.setattr(procedure_grid_analysis, "_bound_labels", bound_labels)
    result = procedure_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=tmp_path / "cache",
        require_formal=False,
        verify_runtime=False,
    )
    assert result.summary["expected_jobs"] == 1
    assert result.summary["confirmation_evidence"] is False
    assert result.summary["common_recipe_result"] is False
    assert result.tables["overall_summary"][0]["balanced_accuracy"] == 1.0
    assert result.tables["overall_summary"][0]["descriptive_rank"] == 1
    serialized = json.dumps(
        {"summary": result.summary, "tables": result.tables}
    ).lower()
    for forbidden in (
        '"labels"',
        '"probabilities"',
        '"rows"',
        '"predictions"',
    ):
        assert forbidden not in serialized

    output = tmp_path / "analysis"
    published = procedure_grid_analysis.publish_analysis(
        result,
        run_root=run_root,
        cache_root=tmp_path / "cache",
        output_root=output,
        require_formal=False,
        verify_runtime=False,
    )
    assert published == output
    assert {path.name for path in output.iterdir()} == {
        "summary.json",
        "job_metrics.csv",
        "subject_seed_metrics.csv",
        "subject_metrics.csv",
        "dataset_summary.csv",
        "overall_summary.csv",
        "timing_summary.csv",
        "manifest.json",
    }
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["raw_labels_or_probabilities_published"] is False
    assert manifest["formal_publication"] is False
    assert manifest["publication_mode"] == procedure_grid.TEST_PUBLICATION_MODE
    assert manifest["ece_bins"] == procedure_grid_analysis.DEFAULT_ECE_BINS
    assert "manifest.json" not in manifest["files"]


def test_publication_rejects_read_only_extra_file_in_bound_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cache_root, result = _computed_tiny_analysis(
        tmp_path, monkeypatch
    )
    output = tmp_path / "analysis-extra"
    original = procedure_grid_analysis._write_artifact_at
    injected = {"done": False}

    def inject_extra(
        descriptor: int,
        name: str,
        payload: bytes,
    ) -> str:
        digest = original(descriptor, name, payload)
        if name == "summary.json" and not injected["done"]:
            injected["done"] = True
            original(descriptor, "EXTRA.txt", b"read-only extra\n")
        return digest

    monkeypatch.setattr(
        procedure_grid_analysis, "_write_artifact_at", inject_extra
    )
    with pytest.raises(
        procedure_grid_analysis.ProcedureAnalysisError,
        match="roster",
    ):
        procedure_grid_analysis.publish_analysis(
            result,
            run_root=run_root,
            cache_root=cache_root,
            output_root=output,
            require_formal=False,
            verify_runtime=False,
        )
    assert injected["done"] is True
    assert not output.exists()


def test_publication_rejects_payload_replacement_after_intended_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cache_root, result = _computed_tiny_analysis(
        tmp_path, monkeypatch
    )
    output = tmp_path / "analysis-replaced"
    original = procedure_grid_analysis._write_artifact_at
    injected = {"done": False}

    def replace_after_hash(
        descriptor: int,
        name: str,
        payload: bytes,
    ) -> str:
        digest = original(descriptor, name, payload)
        if name == "job_metrics.csv" and not injected["done"]:
            injected["done"] = True
            replacement = "job_metrics.replacement"
            original(descriptor, replacement, payload + b"tampered\n")
            os.rename(
                replacement,
                name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            os.fsync(descriptor)
        return digest

    monkeypatch.setattr(
        procedure_grid_analysis, "_write_artifact_at", replace_after_hash
    )
    with pytest.raises(
        procedure_grid_analysis.ProcedureAnalysisError,
        match="manifest",
    ):
        procedure_grid_analysis.publish_analysis(
            result,
            run_root=run_root,
            cache_root=cache_root,
            output_root=output,
            require_formal=False,
            verify_runtime=False,
        )
    assert injected["done"] is True
    assert not output.exists()


def test_publication_seals_exact_snapshot_before_final_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, cache_root, result = _computed_tiny_analysis(
        tmp_path, monkeypatch
    )
    output = tmp_path / "analysis-sealed"
    original = procedure_grid._atomic_rename_noreplace
    observed = {"sealed": False}

    def require_sealed(source: Path, destination: Path) -> None:
        if destination == output:
            directory_stat = procedure_grid._anchored_lstat(source)
            assert stat.S_ISDIR(directory_stat.st_mode)
            assert not directory_stat.st_mode & 0o222
            assert {path.name for path in source.iterdir()} == (
                procedure_grid_analysis.PUBLICATION_FILENAMES
            )
            for path in source.iterdir():
                leaf = procedure_grid._anchored_lstat(path)
                assert stat.S_ISREG(leaf.st_mode)
                assert leaf.st_nlink == 1
                assert not leaf.st_mode & 0o222
            observed["sealed"] = True
        original(source, destination)

    monkeypatch.setattr(
        procedure_grid, "_atomic_rename_noreplace", require_sealed
    )
    published = procedure_grid_analysis.publish_analysis(
        result,
        run_root=run_root,
        cache_root=cache_root,
        output_root=output,
        require_formal=False,
        verify_runtime=False,
    )
    assert published == output
    assert observed["sealed"] is True
    assert not procedure_grid._anchored_lstat(output).st_mode & 0o222


def test_publication_refuses_output_inside_immutable_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    _plan, labels = _complete_tiny_run(run_root)
    monkeypatch.setattr(
        procedure_grid_analysis,
        "_bound_labels",
        lambda *args, **kwargs: (
            {("tiny", 1): labels.copy()},
            [
                {
                    "kind": "cache",
                    "identity": "tiny:s001",
                    "sha256_a": "a" * 64,
                    "sha256_b": "b" * 64,
                    "sha256_c": "c" * 64,
                }
            ],
        ),
    )
    result = procedure_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=tmp_path / "cache",
        require_formal=False,
        verify_runtime=False,
    )
    with pytest.raises(
        procedure_grid_analysis.ProcedureAnalysisError,
        match="outside",
    ):
        procedure_grid_analysis.publish_analysis(
            result,
            run_root=run_root,
            cache_root=tmp_path / "cache",
            output_root=run_root / "analysis",
            require_formal=False,
            verify_runtime=False,
        )


def test_analysis_exact_schema_rejects_claims_and_outcome_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    _plan, labels = _complete_tiny_run(run_root)
    monkeypatch.setattr(
        procedure_grid_analysis,
        "_bound_labels",
        lambda *args, **kwargs: (
            {("tiny", 1): labels.copy()},
            [
                {
                    "kind": "cache",
                    "identity": "tiny:s001",
                    "sha256_a": "a" * 64,
                    "sha256_b": "b" * 64,
                    "sha256_c": "c" * 64,
                }
            ],
        ),
    )
    result = procedure_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=tmp_path / "cache",
        require_formal=False,
        verify_runtime=False,
    )
    false_claim = copy.deepcopy(result)
    false_claim.summary["global_sota_claim"] = True
    with pytest.raises(procedure_grid_analysis.ProcedureAnalysisError):
        procedure_grid_analysis._validate_result(false_claim)
    leaked = copy.deepcopy(result)
    leaked.summary["ground_truth_vector"] = [0, 1, 0, 1]
    with pytest.raises(procedure_grid_analysis.ProcedureAnalysisError):
        procedure_grid_analysis._validate_result(leaked)


def test_publication_recomputes_semantics_and_rejects_consistent_scalar_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    _plan, labels = _complete_tiny_run(run_root)
    monkeypatch.setattr(
        procedure_grid_analysis,
        "_bound_labels",
        lambda *args, **kwargs: (
            {("tiny", 1): labels.copy()},
            [
                {
                    "kind": "cache",
                    "identity": "tiny:s001",
                    "sha256_a": "a" * 64,
                    "sha256_b": "b" * 64,
                    "sha256_c": "c" * 64,
                }
            ],
        ),
    )
    result = procedure_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=tmp_path / "cache",
        require_formal=False,
        verify_runtime=False,
    )
    tampered = copy.deepcopy(result)
    tampered.tables["subject_seed_metrics"][0]["balanced_accuracy"] = 0.5
    tampered.tables["subject_metrics"][0]["balanced_accuracy"] = 0.5
    tampered.tables["dataset_summary"][0]["balanced_accuracy"] = 0.5
    tampered.tables["overall_summary"][0]["balanced_accuracy"] = 0.5
    procedure_grid_analysis._validate_result(tampered)
    with pytest.raises(
        procedure_grid_analysis.ProcedureAnalysisError,
        match="fresh semantic recomputation",
    ):
        procedure_grid_analysis.publish_analysis(
            tampered,
            run_root=run_root,
            cache_root=tmp_path / "cache",
            output_root=tmp_path / "tampered-analysis",
            require_formal=False,
            verify_runtime=False,
        )


def test_publication_refuses_cache_destination_and_fence_blocks_new_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    plan, labels = _complete_tiny_run(run_root)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setattr(
        procedure_grid_analysis,
        "_bound_labels",
        lambda *args, **kwargs: (
            {("tiny", 1): labels.copy()},
            [
                {
                    "kind": "cache",
                    "identity": "tiny:s001",
                    "sha256_a": "a" * 64,
                    "sha256_b": "b" * 64,
                    "sha256_c": "c" * 64,
                }
            ],
        ),
    )
    result = procedure_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=cache_root,
        require_formal=False,
        verify_runtime=False,
    )
    with pytest.raises(
        procedure_grid_analysis.ProcedureAnalysisError,
        match="cache",
    ):
        procedure_grid_analysis.publish_analysis(
            result,
            run_root=run_root,
            cache_root=cache_root,
            output_root=cache_root / "analysis",
            require_formal=False,
            verify_runtime=False,
        )

    job = next(procedure_grid.iter_jobs(plan))
    resource = procedure_grid.validate_completion(run_root, plan, job)["record"][
        "claim_resource_guard"
    ]
    with procedure_grid.publication_fence(
        run_root,
        plan=plan,
        exclusive=True,
    ):
        with pytest.raises(procedure_grid.ClaimUnavailable):
            procedure_grid.acquire_claim(
                run_root,
                plan,
                job,
                resource_guard=resource,
                recover_stale=False,
            )


def test_analysis_freezes_ece_at_fifteen_before_label_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="15"):
        procedure_grid_analysis.compute_analysis(
            run_root=tmp_path,
            cache_root=tmp_path / "cache",
            ece_bins=14,
            require_formal=False,
            verify_runtime=False,
        )


def test_containment_rejection_happens_before_output_parent_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    _plan, labels = _complete_tiny_run(run_root)
    monkeypatch.setattr(
        procedure_grid_analysis,
        "_bound_labels",
        lambda *args, **kwargs: (
            {("tiny", 1): labels.copy()},
            [
                {
                    "kind": "cache",
                    "identity": "tiny:s001",
                    "sha256_a": "a" * 64,
                    "sha256_b": "b" * 64,
                    "sha256_c": "c" * 64,
                }
            ],
        ),
    )
    result = procedure_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=tmp_path / "cache",
        require_formal=False,
        verify_runtime=False,
    )
    parent = run_root / "must-not-be-created"
    with pytest.raises(
        procedure_grid_analysis.ProcedureAnalysisError, match="outside"
    ):
        procedure_grid_analysis.publish_analysis(
            result,
            run_root=run_root,
            cache_root=tmp_path / "cache",
            output_root=parent / "analysis",
            require_formal=False,
            verify_runtime=False,
        )
    assert not parent.exists()


def test_publication_no_replace_race_preserves_competing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    _plan, labels = _complete_tiny_run(run_root)
    monkeypatch.setattr(
        procedure_grid_analysis,
        "_bound_labels",
        lambda *args, **kwargs: (
            {("tiny", 1): labels.copy()},
            [
                {
                    "kind": "cache",
                    "identity": "tiny:s001",
                    "sha256_a": "a" * 64,
                    "sha256_b": "b" * 64,
                    "sha256_c": "c" * 64,
                }
            ],
        ),
    )
    result = procedure_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=tmp_path / "cache",
        require_formal=False,
        verify_runtime=False,
    )
    output = tmp_path / "analysis-race"
    original = procedure_grid._atomic_rename_noreplace
    injected = {"done": False}

    def compete(source: Path, destination: Path) -> None:
        if destination == output and not injected["done"]:
            injected["done"] = True
            output.mkdir()
            (output / "owner.txt").write_text("competitor", encoding="utf-8")
        original(source, destination)

    monkeypatch.setattr(
        procedure_grid, "_atomic_rename_noreplace", compete
    )
    with pytest.raises(FileExistsError):
        procedure_grid_analysis.publish_analysis(
            result,
            run_root=run_root,
            cache_root=tmp_path / "cache",
            output_root=output,
            require_formal=False,
            verify_runtime=False,
        )
    assert (output / "owner.txt").read_text(encoding="utf-8") == "competitor"


def test_publication_fence_replacement_quarantines_published_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    plan, labels = _complete_tiny_run(run_root)
    monkeypatch.setattr(
        procedure_grid_analysis,
        "_bound_labels",
        lambda *args, **kwargs: (
            {("tiny", 1): labels.copy()},
            [
                {
                    "kind": "cache",
                    "identity": "tiny:s001",
                    "sha256_a": "a" * 64,
                    "sha256_b": "b" * 64,
                    "sha256_c": "c" * 64,
                }
            ],
        ),
    )
    result = procedure_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=tmp_path / "cache",
        require_formal=False,
        verify_runtime=False,
    )
    output = tmp_path / "analysis-fence-race"
    original = procedure_grid._atomic_rename_noreplace
    injected = {"done": False}

    def replace_fence_after_publish(
        source: Path, destination: Path
    ) -> None:
        original(source, destination)
        if destination == output and not injected["done"]:
            injected["done"] = True
            fence = procedure_grid._publication_fence_path(run_root)
            fence.unlink()
            procedure_grid._write_json_exclusive(
                fence, procedure_grid._publication_fence_value(plan)
            )

    monkeypatch.setattr(
        procedure_grid,
        "_atomic_rename_noreplace",
        replace_fence_after_publish,
    )
    with pytest.raises(
        procedure_grid_analysis.ProcedureAnalysisError,
        match="fence ownership",
    ):
        procedure_grid_analysis.publish_analysis(
            result,
            run_root=run_root,
            cache_root=tmp_path / "cache",
            output_root=output,
            require_formal=False,
            verify_runtime=False,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".analysis-fence-race.publication-fence-lost-*"))


def test_analysis_rejects_symlinked_cache_and_output_ancestors(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(procedure_grid.ProcedureGridError, match="symlink"):
        procedure_grid_analysis.compute_analysis(
            run_root=tmp_path,
            cache_root=alias / "cache",
            require_formal=False,
            verify_runtime=False,
        )
