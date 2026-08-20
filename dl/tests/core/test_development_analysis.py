from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.development_analysis import (
    DevelopmentAnalysisError,
    NEW_CANDIDATES,
    analyze_and_write,
    analyze_run,
    write_analysis_bundle,
)
from benchmark.development_screen import (
    CANDIDATE_MODELS,
    CURATED_SPLITS,
    EVALUATION_SCOPE,
    RECORD_SCHEMA,
    ScreenJob,
    _record_path,
    assemble_manifest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plan() -> dict[str, object]:
    cache_identity: dict[str, object] = {}
    split_identity: dict[str, object] = {}
    for dataset, subjects in CURATED_SPLITS:
        for subject in subjects:
            subject_key = f"{dataset}:s{subject:03d}"
            split_key = f"{subject_key}:f00"
            cache_identity[subject_key] = {
                "relative_path": f"{dataset}/subject_{subject:03d}.npz",
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
    return assemble_manifest(
        cache_root="/never/opened/by-analysis",
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_hashes={"src/benchmark/development_screen.py": _digest("source")},
        environment_identity={"python": "synthetic"},
    )


_MODEL_SCORE = {
    "chsdnet": 0.74,
    "chsdnet_hybrid": 0.68,
    "chsdnet_direct": 0.71,
    "chsdnet_joint": 0.73,
    "cardinal_fbc_micro_extended": 0.70,
    "tcformer": 0.72,
    "fbcnet": 0.65,
    "eegnet": 0.60,
}
_DATASET_OFFSET = {
    dataset: index * 0.002 for index, (dataset, _) in enumerate(CURATED_SPLITS)
}


def _record(
    job: ScreenJob,
    plan: dict[str, object],
) -> dict[str, object]:
    score = _MODEL_SCORE[job.model] + _DATASET_OFFSET[job.dataset]
    score += job.subject / 100_000.0
    direct = job.model == "chsdnet_direct"
    subject_key = f"{job.dataset}:s{job.subject:03d}"
    split_key = f"{subject_key}:f{job.fold:02d}"
    return {
        "schema": RECORD_SCHEMA,
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
            "parameter_count": 1000 + CANDIDATE_MODELS.index(job.model),
            "constructed_state_sha256": _digest("constructed"),
            "initial_state_sha256": _digest("initial"),
            "selected_state_sha256": _digest("selected"),
            "source_statistics": {
                "available": direct,
                "ran": direct,
                "fit_scope": "training_rows_only",
                "outcome_aware": False,
            },
        },
        "fit": {
            "best_epoch": 5,
            "epochs_run": 10,
            "best_validation_loss": 0.4,
            "early_stopping": True,
            "config": plan["train_config"],
        },
        "validation_metrics": {
            "accuracy": score - 0.01,
            "balanced_accuracy": score,
            "cohen_kappa": 2.0 * score - 1.0,
            "negative_log_likelihood": 1.0 - score,
            "roc_auc": min(score + 0.05, 1.0),
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


def _complete_run(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    run_root = tmp_path / "run"
    plan = _plan()
    run_root.mkdir()
    (run_root / "plan.json").write_text(
        json.dumps(plan, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    for item in plan["jobs"]:  # type: ignore[union-attr]
        job = ScreenJob.from_mapping(item)
        path = _record_path(run_root, job)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_record(job, plan), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return run_root, plan


def test_complete_analysis_has_exact_coverage_pairing_ranks_and_gates(
    tmp_path: Path,
) -> None:
    run_root, _ = _complete_run(tmp_path)
    before = {
        path.relative_to(run_root): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }
    analysis = analyze_run(run_root)
    after = {
        path.relative_to(run_root): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }

    assert before == after
    assert analysis["audit"] == {
        "expected_records": 136,
        "valid_records": 136,
        "missing_records": 0,
        "corrupt_records": 0,
        "unexpected_json_records": 0,
        "analysis_complete": True,
        "missing_job_ids": [],
        "corrupt_job_errors": {},
        "unexpected_json_paths": [],
    }
    assert len(analysis["jobs"]) == 136
    assert len(analysis["subject_model_means"]) == 136
    assert len(analysis["dataset_model_means"]) == 40
    assert len(analysis["overall_model_means"]) == 8
    assert len(analysis["paired_deltas"]) == len(NEW_CANDIDATES) * 2 * 6
    assert len(analysis["ranks"]) == 48

    overall = {
        row["model"]: row for row in analysis["overall_model_means"]
    }
    assert overall["chsdnet"]["balanced_accuracy_rank"] == 1.0
    assert overall["chsdnet"]["complete_coverage"] is True
    pair = next(
        row
        for row in analysis["paired_deltas"]
        if row["candidate"] == "chsdnet"
        and row["reference"] == "cardinal_fbc_micro_extended"
        and row["scope"] == "equal_dataset"
    )
    assert pair["valid_pairs"] == 17
    assert pair["balanced_accuracy_delta"] == pytest.approx(0.04)
    assert pair["accuracy_delta"] == pytest.approx(0.04)
    assert pair["win_rate"] == 1.0

    diagnostics = {
        row["candidate"]: row for row in analysis["freeze_kill_diagnostics"]
    }
    assert diagnostics["chsdnet"]["recommendation"] == (
        "freeze_for_next_prespecified_stage"
    )
    assert diagnostics["chsdnet_hybrid"]["recommendation"] == (
        "kill_or_redesign_current_variant"
    )
    assert diagnostics["chsdnet_joint"]["recommendation"] == (
        "retain_as_ablation_not_frozen"
    )


def test_incomplete_and_corrupt_records_fail_closed_without_opening_cache(
    tmp_path: Path,
) -> None:
    run_root, plan = _complete_run(tmp_path)
    first = ScreenJob.from_mapping(plan["jobs"][0])  # type: ignore[index]
    second = ScreenJob.from_mapping(plan["jobs"][1])  # type: ignore[index]
    _record_path(run_root, first).unlink()
    corrupt_path = _record_path(run_root, second)
    corrupt = json.loads(corrupt_path.read_text(encoding="utf-8"))
    corrupt["plan_sha256"] = _digest("wrong-plan")
    corrupt_path.write_text(json.dumps(corrupt), encoding="utf-8")
    unexpected = run_root / "records" / "orphan.json"
    unexpected.write_text("{}", encoding="utf-8")

    analysis = analyze_run(run_root)
    audit = analysis["audit"]
    assert audit["valid_records"] == 134
    assert audit["missing_records"] == 1
    assert audit["corrupt_records"] == 1
    assert audit["unexpected_json_records"] == 1
    assert audit["analysis_complete"] is False
    assert first.job_id in audit["missing_job_ids"]
    assert second.job_id in audit["corrupt_job_errors"]
    assert audit["unexpected_json_paths"] == ["records/orphan.json"]
    assert all(
        row["recommendation"] == "insufficient_valid_records"
        for row in analysis["freeze_kill_diagnostics"]
    )
    assert all(
        row["equal_dataset_macro_balanced_accuracy"] is None
        for row in analysis["overall_model_means"]
        if row["model"] in {first.model, second.model}
    )


def test_bundle_writes_json_csv_and_cautious_markdown(tmp_path: Path) -> None:
    run_root, _ = _complete_run(tmp_path)
    analysis = analyze_run(run_root)
    output = tmp_path / "analysis"
    paths = write_analysis_bundle(analysis, output)

    assert set(paths) == {
        "analysis_json",
        "jobs",
        "subject_model_means",
        "dataset_model_means",
        "overall_model_means",
        "paired_deltas",
        "ranks",
        "freeze_kill_diagnostics",
        "audit",
        "summary_markdown",
    }
    loaded = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert loaded["audit"]["valid_records"] == 136
    assert len((output / "jobs.csv").read_text(encoding="utf-8").splitlines()) == 137
    assert (
        len(
            (output / "dataset_model_means.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        == 41
    )
    markdown = (output / "summary.md").read_text(encoding="utf-8")
    assert "model-selection evidence only" in markdown
    assert "does not establish novelty" in markdown
    assert "state-of-the-art" not in markdown.lower()
    assert "sota" not in markdown.lower()
    assert not tuple(output.glob("*.partial"))

    with pytest.raises(DevelopmentAnalysisError):
        analyze_and_write(run_root, run_root / "records" / "analysis")
