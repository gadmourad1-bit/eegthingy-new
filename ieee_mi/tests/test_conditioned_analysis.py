from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import ieee_mi.conditioned_screen as conditioned_screen
from ieee_mi.conditioned_analysis import (
    ConditionedAnalysisError,
    analyze_run,
    write_analysis_bundle,
)
from ieee_mi.conditioned_screen import (
    CONDITIONED_MODELS,
    REFERENCE_MODELS,
    SOURCE_FILES,
    ScreenJob,
    _record_path,
    assemble_manifest as assemble_conditioned_manifest,
    conditioned_jobs,
)
from ieee_mi.development_screen import (
    CANDIDATE_MODELS,
    CURATED_SPLITS,
    EVALUATION_SCOPE,
    RECORD_SCHEMA as REFERENCE_RECORD_SCHEMA,
    _record_path as reference_record_path,
    assemble_manifest as assemble_reference_manifest,
    curated_jobs as reference_jobs,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


GPU_UUIDS = (
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
    "GPU-00000000-0000-0000-0000-000000000003",
)


def _conditioned_environment_identity() -> dict[str, object]:
    return {
        "python": "3.synthetic",
        "executable": "/synthetic/.venv/bin/python",
        "platform": "synthetic",
        "packages": {"torch": "synthetic"},
        "torch_cuda_version": "synthetic",
        "torch_cudnn_version": 999,
        "nvidia_driver_versions": ["synthetic"],
        "required_cublas_workspace_config": ":4096:8",
        "cpu_threads_per_worker": 4,
        "minimum_free_gib": 50.0,
        "uv_executable": "/synthetic/bin/uv",
        "uv_version": "uv 0.synthetic",
        "virtualenv_prefix": "/synthetic/.venv",
        "pyvenv_cfg_sha256": _digest("pyvenv"),
    }


def _identity_maps() -> tuple[dict[str, object], dict[str, object]]:
    cache_identity: dict[str, object] = {}
    split_identity: dict[str, object] = {}
    for dataset, subjects in CURATED_SPLITS:
        for subject in subjects:
            subject_key = f"{dataset}:s{subject:03d}"
            split_key = f"{subject_key}:f00"
            cache_identity[subject_key] = {
                "relative_path": f"cache/{dataset}/subject_{subject:03d}.npz",
                "file_size_bytes": 123,
                "file_sha256": _digest(f"file-{subject_key}"),
                "array_sha256": _digest(f"array-{subject_key}"),
            }
            split_identity[split_key] = {
                "train": {
                    "count": 20,
                    "rows_sha256": _digest(f"train-{split_key}"),
                },
                "validation": {
                    "count": 10,
                    "rows_sha256": _digest(f"validation-{split_key}"),
                },
            }
    return cache_identity, split_identity


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _conditioned_record(
    job: ScreenJob,
    plan: dict[str, object],
    score: float,
) -> dict[str, object]:
    subject_key = f"{job.dataset}:s{job.subject:03d}"
    split_key = f"{subject_key}:f00"
    return {
        "schema": conditioned_screen.RECORD_SCHEMA,
        "created_at": "2026-07-29T00:00:00+00:00",
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "cache_identity": {
            "array_sha256": plan["cache_identity"][subject_key]["array_sha256"],  # type: ignore[index]
            "file_sha256": plan["cache_identity"][subject_key]["file_sha256"],  # type: ignore[index]
        },
        "source_identity": plan["source_identity"],
        "split": plan["split_identity"][split_key],  # type: ignore[index]
        "scaler": {
            "fit_scope": "training_only",
            "mean_sha256": _digest("mean"),
            "std_sha256": _digest("std"),
            "contract": plan["channel_scaling"],
        },
        "model": {
            "identity": {"requested_name": job.model},
            "parameter_count": 1000,
            "constructed_state_sha256": _digest("constructed"),
            "initial_state_sha256": _digest("initial"),
            "selected_state_sha256": _digest("selected"),
            "source_statistics": {
                "available": False,
                "ran": False,
                "fit_scope": "training_rows_only",
                "outcome_aware": False,
            },
        },
        "fit": {
            "best_epoch": 4,
            "epochs_run": 9,
            "best_validation_loss": 0.4,
            "early_stopping": True,
            "config": plan["train_config"],
        },
        "validation_metrics": {
            "accuracy": score,
            "balanced_accuracy": score,
            "cohen_kappa": 2.0 * score - 1.0,
            "negative_log_likelihood": 1.0 - score,
            "roc_auc": min(1.0, score + 0.05),
        },
        "timing": {
            "construction_seconds": 0.1,
            "fit_seconds": 1.0,
            "evaluation_seconds": 0.1,
            "total_seconds": 1.2,
        },
        "resource_preflight": {
            "gpu": {
                "gpu_uuid": GPU_UUIDS[CONDITIONED_MODELS.index(job.model)],
                "foreign_compute_processes": [],
            },
            "disk": {"free_gib": 100.0, "minimum_free_gib": 50.0},
        },
    }


def _reference_record(
    job: ScreenJob,
    plan: dict[str, object],
    score: float,
) -> dict[str, object]:
    subject_key = f"{job.dataset}:s{job.subject:03d}"
    split_key = f"{subject_key}:f00"
    return {
        "schema": REFERENCE_RECORD_SCHEMA,
        "created_at": "2026-07-29T00:00:00+00:00",
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "cache_identity": {
            "array_sha256": plan["cache_identity"][subject_key]["array_sha256"],  # type: ignore[index]
            "file_sha256": plan["cache_identity"][subject_key]["file_sha256"],  # type: ignore[index]
        },
        "source_identity": plan["source_identity"],
        "split": plan["split_identity"][split_key],  # type: ignore[index]
        "scaler": {
            "fit_scope": "training_only",
            "mean_sha256": _digest("mean"),
            "std_sha256": _digest("std"),
            "contract": {"fit_rows": "training_only"},
        },
        "model": {
            "identity": {"requested_name": job.model},
            "parameter_count": 900,
            "constructed_state_sha256": _digest("constructed"),
            "initial_state_sha256": _digest("initial"),
            "selected_state_sha256": _digest("selected"),
            "source_statistics": {
                "available": False,
                "ran": False,
                "fit_scope": "training_rows_only",
                "outcome_aware": False,
            },
        },
        "fit": {
            "best_epoch": 4,
            "epochs_run": 9,
            "best_validation_loss": 0.4,
            "early_stopping": True,
            "config": plan["train_config"],
        },
        "validation_metrics": {
            "accuracy": score,
            "balanced_accuracy": score,
            "cohen_kappa": 2.0 * score - 1.0,
            "negative_log_likelihood": 1.0 - score,
            "roc_auc": min(1.0, score + 0.05),
        },
        "timing": {
            "construction_seconds": 0.1,
            "fit_seconds": 1.0,
            "evaluation_seconds": 0.1,
            "total_seconds": 1.2,
        },
        "resource_preflight": {
            "gpu": {"gpu_uuid": "GPU-synthetic"},
            "disk": {"free_gib": 100.0},
        },
    }


def _complete_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    conditioned_root = tmp_path / "conditioned"
    reference_root = tmp_path / "reference"
    cache_identity, split_identity = _identity_maps()
    reference_plan = assemble_reference_manifest(
        cache_root="/synthetic/cache",
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_hashes={"ieee_mi/development_screen.py": _digest("reference")},
        environment_identity={"python": "synthetic"},
    )
    monkeypatch.setattr(
        conditioned_screen,
        "EXPECTED_REFERENCE_PLAN_SHA256",
        reference_plan["plan_sha256"],
    )
    binding = conditioned_screen._reference_binding(
        reference_root, reference_plan
    )
    conditioned_plan = assemble_conditioned_manifest(
        cache_root="/synthetic/cache",
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_hashes={
            relative: _digest(f"conditioned-{relative}")
            for relative in SOURCE_FILES
        },
        environment_identity=_conditioned_environment_identity(),
        worker_gpu_uuids=GPU_UUIDS,
        reference_binding=binding,
    )
    _write_json(reference_root / "plan.json", reference_plan)
    _write_json(conditioned_root / "plan.json", conditioned_plan)

    conditioned_score = {
        "chsdnet_conditioned_005": 0.72,
        "chsdnet_conditioned_010": 0.78,
        "chsdnet_conditioned_020": 0.74,
    }
    reference_score = {
        "cardinal_fbc_micro_extended": 0.70,
        "tcformer": 0.71,
        "fbcnet": 0.68,
    }
    dataset_offset = {
        dataset: index * 0.001
        for index, (dataset, _) in enumerate(CURATED_SPLITS)
    }
    for job in conditioned_jobs():
        score = conditioned_score[job.model] + dataset_offset[job.dataset]
        _write_json(
            _record_path(conditioned_root, job),
            _conditioned_record(job, conditioned_plan, score),
        )
    for job in reference_jobs():
        if job.model not in REFERENCE_MODELS:
            continue
        score = reference_score[job.model] + dataset_offset[job.dataset]
        _write_json(
            reference_record_path(reference_root, job),
            _reference_record(job, reference_plan, score),
        )
    return conditioned_root, reference_root, conditioned_plan, reference_plan


def test_complete_read_only_join_validates_51_plus_51_and_chooses_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditioned_root, reference_root, _, _ = _complete_runs(
        tmp_path, monkeypatch
    )
    before_conditioned = {
        path.relative_to(conditioned_root): path.read_bytes()
        for path in conditioned_root.rglob("*")
        if path.is_file()
    }
    before_reference = {
        path.relative_to(reference_root): path.read_bytes()
        for path in reference_root.rglob("*")
        if path.is_file()
    }

    # The analyzer must not call the EEG archive loader.
    import numpy as np

    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("analysis opened an EEG cache")
        ),
    )
    analysis = analyze_run(conditioned_root, reference_root)
    assert analysis["audit"] == {
        "conditioned_records_valid": 51,
        "reference_records_valid": 51,
        "paired_subject_records": 51,
        "complete": True,
    }
    assert len(analysis["jobs"]) == 102
    assert len(analysis["dataset_scores"]) == 30
    assert len(analysis["overall_scores"]) == 6
    assert len(analysis["paired_comparisons"]) == 54
    assert analysis["selection"]["decision"] == (
        "choose_one_for_next_prespecified_stage"
    )
    assert analysis["selection"]["selected_model"] == (
        "chsdnet_conditioned_010"
    )
    assert before_conditioned == {
        path.relative_to(conditioned_root): path.read_bytes()
        for path in conditioned_root.rglob("*")
        if path.is_file()
    }
    assert before_reference == {
        path.relative_to(reference_root): path.read_bytes()
        for path in reference_root.rglob("*")
        if path.is_file()
    }


def test_missing_or_corrupt_conditioned_record_fails_before_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditioned_root, reference_root, _, _ = _complete_runs(
        tmp_path, monkeypatch
    )
    first = conditioned_jobs()[0]
    _record_path(conditioned_root, first).unlink()
    with pytest.raises(ConditionedAnalysisError, match="all 51 conditioned"):
        analyze_run(conditioned_root, reference_root)


def test_reference_plan_binding_and_output_read_only_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditioned_root, reference_root, _, _ = _complete_runs(
        tmp_path, monkeypatch
    )
    other_reference = tmp_path / "other-reference"
    other_reference.mkdir()
    (other_reference / "plan.json").write_bytes(
        (reference_root / "plan.json").read_bytes()
    )
    with pytest.raises(ConditionedAnalysisError, match="reference root"):
        analyze_run(conditioned_root, other_reference)

    analysis = analyze_run(conditioned_root, reference_root)
    with pytest.raises(ConditionedAnalysisError, match="must not modify"):
        write_analysis_bundle(
            analysis,
            reference_root / "analysis",
            reference_run_root=reference_root,
        )


def test_tied_or_failed_gate_kills_family() -> None:
    from ieee_mi.conditioned_analysis import _decision

    overall = [
        {
            "model": model,
            "equal_dataset_balanced_accuracy": (
                0.75 if model in CONDITIONED_MODELS else 0.74
            ),
        }
        for model in (*CONDITIONED_MODELS, *REFERENCE_MODELS)
    ]
    paired = []
    for candidate in CONDITIONED_MODELS:
        for reference in REFERENCE_MODELS:
            paired.append(
                {
                    "scope": "equal_dataset",
                    "dataset": None,
                    "candidate": candidate,
                    "reference": reference,
                    "balanced_accuracy_delta": 0.01,
                    "nonnegative_datasets": 5,
                    "minimum_dataset_delta": 0.0,
                    "win_rate": 0.6,
                }
            )
    decision = _decision(overall, paired)
    assert decision["decision"] == "kill_conditioned_family"
    assert decision["selected_model"] is None
