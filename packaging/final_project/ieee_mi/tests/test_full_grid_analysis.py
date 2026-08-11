from __future__ import annotations

import copy
import errno
import hashlib
import inspect
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ieee_mi import full_grid, full_grid_analysis


DATASETS = ("d1", "d2")
MODELS = ("model_a", "model_b", "tcformer")
SEEDS = (7, 17)
LABELS = np.asarray([0, 1, 0, 1], dtype=np.int64)
TEST_ROWS = np.asarray([2, 3], dtype=np.int64)


def _synthetic_plan() -> dict[str, Any]:
    contracts: dict[str, Any] = {}
    cache_identity: dict[str, Any] = {}
    split_identity: dict[str, Any] = {}
    for index, dataset in enumerate(DATASETS, start=1):
        digest = f"{index:x}" * 64
        contracts[dataset] = {
            "subjects": [1],
            "folds": [0],
            "n_classes": 2,
            "protocol": "synthetic-test-only",
            "preprocessing": {"sfreq_hz": 128.0, "n_times": 4},
        }
        cache_identity[f"{dataset}:s001"] = {"array_sha256": digest}
        split_identity[f"{dataset}:s001:f00"] = full_grid._one_split_identity(
            dataset=dataset,
            subject=1,
            fold=0,
            cache_array_sha256=digest,
            trial_count=4,
            train_rows=np.asarray([0], dtype=np.int64),
            validation_rows=np.asarray([1], dtype=np.int64),
            test_rows=TEST_ROWS,
        )
    return full_grid.assemble_plan(
        dataset_contracts=contracts,
        architectures=MODELS,
        seeds=SEEDS,
        train_config={"epochs": 2},
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_identity={"tests/synthetic.py": "a" * 64},
        environment_identity={},
        executor="tests.synthetic:executor",
        worker_cpu_threads=2,
        analysis_contract=None,
        publishable=False,
    )


def _audit(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": full_grid.AUDIT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "expected": plan["n_jobs"],
        "complete": plan["n_jobs"],
        "missing": 0,
        "corrupt": 0,
        "extra": 0,
        "live_claims": 0,
        "stale_claims": 0,
        "unknown_claims": 0,
        "partials": 0,
        "claim_tombstones": 0,
        "unsafe_paths": 0,
        "unexpected_root_entries": 0,
        "failed_jobs": 0,
        "quarantine_artifacts": 0,
        "resolved_forensic_artifacts": 0,
        "invalid_forensic_artifacts": 0,
        "forensic_ledger_sha256": hashlib.sha256(b"[]").hexdigest(),
        "preflight_attestation_valid": True,
        "preflight_report_sha256": None,
        "exact_cartesian_complete": True,
        "details": {"forensic_ledger": []},
    }


def _record(job: full_grid.Job) -> dict[str, Any]:
    parameter_counts = {"model_a": 100, "model_b": 80, "tcformer": 120}
    return {
        "test_count": 2,
        "metadata": {
            "fit": {
                "source_selected_epoch": 1,
                "selection_epochs_run": 2,
                "refit_epochs_run": 2,
                "parameter_count": parameter_counts[job.model],
            },
            "timing_seconds": {
                "selection_fit": 0.2,
                "refit_fit": 0.1,
                "test_inference": 0.01,
                "job_total": 0.35,
            },
            "cuda_peak_memory_bytes": 2048,
        },
    }


def _probabilities(job: full_grid.Job) -> np.ndarray:
    if job.model == "model_a":
        return np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    if job.model == "model_b":
        return np.asarray([[0.7, 0.3], [0.3, 0.7]], dtype=np.float64)
    return np.asarray([[0.8, 0.2], [0.7, 0.3]], dtype=np.float64)


def _install_compute_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    run_root = tmp_path / "run"
    cache_root = tmp_path / "cache"
    run_root.mkdir(parents=True)
    cache_root.mkdir()
    plan = _synthetic_plan()
    jobs = tuple(full_grid.iter_jobs(plan))
    audit = _audit(plan)

    monkeypatch.setattr(
        full_grid_analysis,
        "audit_exact_grid",
        lambda unused: (plan, jobs, audit),
    )
    monkeypatch.setattr(
        full_grid_analysis,
        "_bound_cache_labels",
        lambda unused_plan, unused_cache: {
            (dataset, 1): LABELS.copy() for dataset in DATASETS
        },
    )

    def fake_completion(
        unused_root: Path,
        unused_plan: dict[str, Any],
        job: full_grid.Job,
        *,
        expected_preflight_report_sha256: str | None = None,
    ) -> dict[str, Any]:
        assert expected_preflight_report_sha256 is None
        return {
            "rows": TEST_ROWS.copy(),
            "probabilities": _probabilities(job),
            "record": _record(job),
            "completion": {
                "files": {
                    "record.json": "b" * 64,
                    "predictions.npz": "c" * 64,
                }
            },
        }

    monkeypatch.setattr(
        full_grid,
        "_validate_completion_bound",
        fake_completion,
    )
    monkeypatch.setattr(
        full_grid_analysis,
        "_input_ledger_row",
        lambda unused_root, job, unused_payload: {
            "job_id": job.job_id,
            "completion_sha256": "a" * 64,
            "record_sha256": "b" * 64,
            "predictions_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        full_grid_analysis,
        "_verify_input_ledger",
        lambda *unused_args, **unused_kwargs: None,
    )
    monkeypatch.setattr(full_grid, "load_plan", lambda unused: plan)
    monkeypatch.setattr(
        full_grid_analysis,
        "_strict_record_tree",
        lambda *unused: None,
    )
    monkeypatch.setattr(
        full_grid_analysis,
        "_require_quiescent_auxiliary_state",
        lambda *unused: None,
    )
    return run_root, cache_root, plan


def _compute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    resamples: int = 64,
) -> tuple[full_grid_analysis.AnalysisResult, Path, Path, dict[str, Any]]:
    run_root, cache_root, plan = _install_compute_mocks(monkeypatch, tmp_path)
    result = full_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=cache_root,
        bootstrap_resamples=resamples,
        bootstrap_seed=123,
        ece_bins=15,
        tcformer_name="tcformer",
    )
    return result, run_root, cache_root, plan


def test_classification_metrics_and_ece15_are_exact() -> None:
    metrics = full_grid_analysis.classification_metrics(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64),
        n_classes=2,
        ece_bins=15,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["chance_normalized_balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["ece"] == pytest.approx(0.15)
    with pytest.raises(full_grid_analysis.GridAnalysisError, match="invalid arrays"):
        full_grid_analysis.classification_metrics(
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([[0.9, 0.2], [0.1, 0.9]], dtype=np.float64),
            n_classes=2,
        )


def test_bootstraps_are_deterministic_and_preserve_fixed_dataset_weighting() -> None:
    values = {"d1": np.asarray([0.1, 0.3]), "d2": np.asarray([-0.1])}
    first = full_grid_analysis.fixed_suite_subject_bootstrap(
        values,
        n_resamples=100,
        seed=9,
    )
    second = full_grid_analysis.fixed_suite_subject_bootstrap(
        values,
        n_resamples=100,
        seed=9,
    )
    assert first == second
    assert first["observed"] == pytest.approx(0.05)
    assert first["datasets_fixed"] == 2
    assert first["resamples"] == 100
    subject = full_grid_analysis.bootstrap_subject_mean(
        np.asarray([0.1, 0.2]),
        n_resamples=50,
        seed=4,
    )
    assert subject["observed"] == pytest.approx(0.15)


def test_compute_analysis_produces_complete_descriptive_all_model_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _, plan = _compute(monkeypatch, tmp_path)
    assert result.summary["schema"] == full_grid_analysis.ANALYSIS_SCHEMA
    assert result.summary["plan_sha256"] == plan["plan_sha256"]
    assert result.summary["descriptive_leader"] == "model_a"
    assert result.summary["confirmation_evidence"] is False
    assert result.summary["tcformer_comparator"] == "tcformer"
    assert set(result.tables) == full_grid_analysis._TABLE_NAMES
    assert result.summary["table_row_counts"] == {
        "job_metrics": 12,
        "subject_seed_metrics": 12,
        "subject_metrics": 6,
        "dataset_summary": 6,
        "overall_summary": 3,
        "trial_micro_summary": 9,
        "resource_timing_summary": 9,
        "model_ranking": 3,
        "calibration_summary": 9,
        "complexity_summary": 3,
        "tcformer_dataset_context": 4,
        "tcformer_overall_context": 2,
        "tcformer_seed_context": 12,
        "input_checksum_ledger": 12,
    }
    assert [row["model"] for row in result.tables["model_ranking"]] == [
        "model_a",
        "model_b",
        "tcformer",
    ]
    assert all(
        row["comparator"] == "tcformer"
        for row in result.tables["tcformer_overall_context"]
    )
    assert all(
        row["status"].startswith("descriptive")
        for row in result.tables["tcformer_overall_context"]
    )
    assert all(
        row["ece_bins"] == 15
        for row in result.tables["calibration_summary"]
    )


def test_compute_is_deterministic_for_frozen_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_root, cache_root, _ = _compute(monkeypatch, tmp_path)
    repeated = full_grid_analysis.compute_analysis(
        run_root=run_root,
        cache_root=cache_root,
        bootstrap_resamples=64,
        bootstrap_seed=123,
        ece_bins=15,
        tcformer_name="tcformer",
    )
    assert result.summary == repeated.summary
    assert result.tables == repeated.tables
    assert result.input_ledger == repeated.input_ledger
    assert result._seal == repeated._seal


def test_artifacts_have_no_raw_labels_probabilities_rows_or_inference_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _, _ = _compute(monkeypatch, tmp_path)
    full_grid_analysis._reject_forbidden_artifact_keys(result.summary)
    full_grid_analysis._reject_forbidden_artifact_keys(result.tables)
    serialized = json.dumps(
        {"summary": result.summary, "tables": result.tables},
        sort_keys=True,
    ).lower()
    for token in ('"p_value"', '"holm_adjusted"', '"promotion_decision"'):
        assert token not in serialized


def test_markdown_reports_all_models_and_explicitly_limits_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _, _ = _compute(monkeypatch, tmp_path)
    report = full_grid_analysis._markdown_report(result)
    assert "# Exact 43-model common-grid development analysis" in report
    assert "`model_a`" in report
    assert "`model_b`" in report
    assert "`tcformer`" in report
    assert "No p-values" in report
    assert "not a global/SOTA claim" in report


def test_tcformer_context_is_model_minus_comparator_and_seed_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, _, _ = _compute(monkeypatch, tmp_path)
    overall = {
        row["model"]: row
        for row in result.tables["tcformer_overall_context"]
    }
    assert overall["model_a"][
        "equal_dataset_macro_balanced_accuracy_difference"
    ] == pytest.approx(0.5)
    assert overall["model_b"][
        "equal_dataset_macro_balanced_accuracy_difference"
    ] == pytest.approx(0.5)
    observed = {
        (row["model"], row["seed"], row["dataset"])
        for row in result.tables["tcformer_seed_context"]
    }
    expected = {
        (model, seed, dataset)
        for model in ("model_a", "model_b")
        for seed in SEEDS
        for dataset in (*DATASETS, "EQUAL_DATASET_OVERALL")
    }
    assert observed == expected


def test_test_only_grid_can_be_computed_but_never_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_root, _, plan = _compute(monkeypatch, tmp_path)
    jobs = tuple(full_grid.iter_jobs(plan))
    monkeypatch.setattr(
        full_grid_analysis,
        "audit_exact_grid",
        lambda unused: (plan, jobs, _audit(plan)),
    )
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="cannot be published",
    ):
        full_grid_analysis._validate_result_for_publication(
            result,
            run_root=run_root,
        )


def test_result_schema_and_seal_reject_post_compute_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_root, _, _ = _compute(monkeypatch, tmp_path)
    result.tables["overall_summary"][0]["hidden"] = 1
    with pytest.raises(full_grid_analysis.GridAnalysisError, match="seal"):
        full_grid_analysis._validate_result_for_publication(
            result,
            run_root=run_root,
        )
    result._seal = full_grid_analysis._analysis_result_seal(
        result.summary,
        result.tables,
        result.input_ledger,
    )
    with pytest.raises(full_grid_analysis.GridAnalysisError, match="exact schema"):
        full_grid_analysis._validate_result_for_publication(
            result,
            run_root=run_root,
        )


def test_input_ledger_uses_same_validated_snapshot_and_rechecks_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = full_grid.Job(
        dataset="d1",
        model="model_a",
        subject=1,
        fold=0,
        seed=7,
    )
    payloads = {
        "completion.json": b"completion-a",
        "record.json": b"record-a",
        "predictions.npz": b"predictions-a",
    }
    snapshot_digests = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in payloads.items()
    }
    monkeypatch.setattr(
        full_grid_analysis,
        "_sha256_file",
        lambda unused: pytest.fail(
            "ledger construction must not reread a detached path snapshot"
        ),
    )
    row = full_grid_analysis._input_ledger_row(
        tmp_path,
        job,
        {"artifact_sha256": snapshot_digests},
    )
    assert row == {
        "job_id": job.job_id,
        "completion_sha256": snapshot_digests["completion.json"],
        "record_sha256": snapshot_digests["record.json"],
        "predictions_sha256": snapshot_digests["predictions.npz"],
    }
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="same validated snapshot|exact digests",
    ):
        full_grid_analysis._input_ledger_row(
            tmp_path,
            job,
            {"artifact_sha256": {"completion.json": "a" * 64}},
        )

    # The later close-of-computation check must use one descriptor-held
    # completion snapshot, never three independently toggleable path hashes.
    current_digests = dict(snapshot_digests)
    plan = {"publication_mode": full_grid.TEST_PUBLICATION_MODE}
    monkeypatch.setattr(
        full_grid,
        "iter_jobs",
        lambda unused: iter((job,)),
    )

    def validated_snapshot(*unused: Any, **unused_keywords: Any) -> dict[str, Any]:
        return {"artifact_sha256": dict(current_digests)}

    monkeypatch.setattr(
        full_grid,
        "_validate_completion_bound",
        validated_snapshot,
    )
    monkeypatch.setattr(
        full_grid_analysis,
        "_sha256_file",
        lambda unused: pytest.fail(
            "detached per-file hashes permit toggle-package attacks"
        ),
    )
    full_grid_analysis._verify_input_ledger(tmp_path, plan, [row])
    current_digests["predictions.npz"] = hashlib.sha256(
        b"predictions-b"
    ).hexdigest()
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="changed during analysis",
    ):
        full_grid_analysis._verify_input_ledger(tmp_path, plan, [row])
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="exact plan job order",
    ):
        full_grid_analysis._verify_input_ledger(tmp_path, plan, [row, row])


@pytest.mark.parametrize(
    ("table_name", "field"),
    (
        ("job_metrics", "fold"),
        ("job_metrics", "test_count"),
        ("job_metrics", "selected_epoch_index"),
        ("model_ranking", "rank"),
        ("subject_metrics", "balanced_accuracy"),
    ),
)
def test_table_contract_rejects_boolean_numeric_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    table_name: str,
    field: str,
) -> None:
    result, _, _, _ = _compute(monkeypatch, tmp_path)
    rows = copy.deepcopy(result.tables[table_name])
    rows[0][field] = False
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="must not be boolean",
    ):
        full_grid_analysis._validate_scalar_table(table_name, rows)


@pytest.mark.parametrize(
    ("table_name", "field", "invalid", "message"),
    (
        ("job_metrics", "fold", 0.0, "exact integer"),
        ("job_metrics", "accuracy", 1, "exact float"),
        ("job_metrics", "nll", -0.1, "out of range"),
        ("calibration_summary", "ece", 1.1, "out of range"),
        (
            "calibration_summary",
            "multiclass_brier",
            2.1,
            "out of range",
        ),
        ("model_ranking", "rank", 0, "out of range"),
        ("subject_metrics", "accuracy_defined_count", 3, "impossible"),
    ),
)
def test_table_contract_enforces_exact_column_types_and_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    table_name: str,
    field: str,
    invalid: Any,
    message: str,
) -> None:
    result, _, _, _ = _compute(monkeypatch, tmp_path)
    rows = copy.deepcopy(result.tables[table_name])
    rows[0][field] = invalid
    with pytest.raises(full_grid_analysis.GridAnalysisError, match=message):
        full_grid_analysis._validate_scalar_table(table_name, rows)


def test_recursive_forbidden_artifact_keys_are_rejected() -> None:
    with pytest.raises(full_grid_analysis.GridAnalysisError, match="forbidden key"):
        full_grid_analysis._reject_forbidden_artifact_keys(
            {"safe": [{"nested": {"probabilities": [0.5, 0.5]}}]}
        )


def test_exact_audit_accepts_new_schema_without_run_local_gpu_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    plan = _synthetic_plan()
    jobs = tuple(full_grid.iter_jobs(plan))
    monkeypatch.setattr(full_grid, "load_plan", lambda unused: plan)
    monkeypatch.setattr(full_grid, "audit_grid", lambda *unused: _audit(plan))
    monkeypatch.setattr(
        full_grid_analysis,
        "_strict_record_tree",
        lambda *unused: None,
    )
    monkeypatch.setattr(
        full_grid_analysis,
        "_require_quiescent_auxiliary_state",
        lambda *unused: None,
    )
    observed_plan, observed_jobs, observed_audit = (
        full_grid_analysis.audit_exact_grid(root)
    )
    assert observed_plan == plan
    assert observed_jobs == jobs
    assert observed_audit["exact_cartesian_complete"]
    assert "gpu_leases" not in observed_audit


def test_exact_audit_accepts_and_hash_binds_resolved_forensic_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    plan = _synthetic_plan()
    jobs = tuple(full_grid.iter_jobs(plan))
    forensic_ledger = [
        {
            "category": "claims",
            "reason": "stale",
            "identity": jobs[0].job_id,
            "path": (
                f"quarantine/claims/{jobs[0].job_id}.json.stale."
                f"20260729T120000.{'a' * 10}"
            ),
            "artifact_sha256": "b" * 64,
        }
    ]
    resolved_audit = _audit(plan)
    resolved_audit["quarantine_artifacts"] = 1
    resolved_audit["resolved_forensic_artifacts"] = 1
    resolved_audit["details"]["forensic_ledger"] = forensic_ledger
    resolved_audit["forensic_ledger_sha256"] = hashlib.sha256(
        full_grid._canonical_bytes(forensic_ledger)
    ).hexdigest()
    monkeypatch.setattr(full_grid, "load_plan", lambda unused: plan)
    monkeypatch.setattr(
        full_grid,
        "audit_grid",
        lambda *unused: resolved_audit,
    )
    monkeypatch.setattr(
        full_grid_analysis,
        "_strict_record_tree",
        lambda *unused: None,
    )
    monkeypatch.setattr(
        full_grid_analysis,
        "_require_quiescent_auxiliary_state",
        lambda *unused: None,
    )
    _, observed_jobs, observed_audit = full_grid_analysis.audit_exact_grid(
        root
    )
    assert observed_jobs == jobs
    assert observed_audit["exact_cartesian_complete"]
    assert observed_audit["resolved_forensic_artifacts"] == 1

    forged = copy.deepcopy(resolved_audit)
    forged["forensic_ledger_sha256"] = "0" * 64
    monkeypatch.setattr(full_grid, "audit_grid", lambda *unused: forged)
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="not an exact",
    ):
        full_grid_analysis.audit_exact_grid(root)


def test_exact_audit_rejects_boolean_counter_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    plan = _synthetic_plan()
    forged_audit = _audit(plan)
    forged_audit["missing"] = False
    monkeypatch.setattr(full_grid, "load_plan", lambda unused: plan)
    monkeypatch.setattr(full_grid, "audit_grid", lambda *unused: forged_audit)
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="exact nonnegative integers",
    ):
        full_grid_analysis.audit_exact_grid(root)


def test_quiescence_rejects_unknown_claim_partial_and_tombstone(
    tmp_path: Path,
) -> None:
    for name in ("claims", "partials", full_grid.CLAIM_TOMBSTONE_DIRECTORY):
        root = tmp_path / name
        root.mkdir()
        state = root / name
        state.mkdir()
        (state / "unknown").write_text("x", encoding="utf-8")
        with pytest.raises(full_grid_analysis.GridAnalysisError, match=name):
            full_grid_analysis._require_quiescent_auxiliary_state(root)


def test_quiescence_rejects_even_an_empty_symlinked_claim_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    target = tmp_path / "outside"
    run_root.mkdir()
    target.mkdir()
    (run_root / "claims").symlink_to(target, target_is_directory=True)
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="unsafe claims",
    ):
        full_grid_analysis._require_quiescent_auxiliary_state(run_root)


def test_formal_grid_requires_a_pre_result_analysis_contract() -> None:
    plan = {
        "publication_mode": full_grid.FORMAL_PUBLICATION_MODE,
        "dataset_order": list(full_grid.OPENED_DATASETS),
        "architectures": list(full_grid.COMMON_ARCHITECTURES),
        "n_jobs": full_grid.FORMAL_EXPECTED_JOBS,
        "analysis_contract": None,
    }
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="no pre-result analysis contract",
    ):
        full_grid_analysis._verify_frozen_analysis_contract(
            plan,
            bootstrap_resamples=100_000,
            bootstrap_seed=full_grid_analysis.DEFAULT_BOOTSTRAP_SEED,
            ece_bins=15,
            tcformer_name="tcformer",
        )


def _install_publication_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: full_grid_analysis.AnalysisResult,
) -> dict[str, Any]:
    bound_plan = {
        "analysis_contract": {
            "source_identity": {
                "ieee_mi/full_grid_analysis.py": "a" * 64,
            },
            "environment_identity_sha256": "b" * 64,
        }
    }
    monkeypatch.setattr(full_grid, "load_plan", lambda unused: bound_plan)
    monkeypatch.setattr(
        full_grid,
        "verify_runtime_identity",
        lambda *unused_args, **unused_kwargs: None,
    )
    monkeypatch.setattr(
        full_grid_analysis,
        "compute_analysis",
        lambda **unused: result,
    )
    monkeypatch.setattr(
        full_grid_analysis,
        "_validate_result_for_publication",
        lambda unused_result, *, run_root: (bound_plan, (), {}),
    )
    return bound_plan


def test_publish_recomputes_under_lock_and_writes_readonly_atomic_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_root, cache_root, _ = _compute(monkeypatch, tmp_path / "compute")
    _install_publication_mocks(monkeypatch, result=result)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    destination = full_grid_analysis.publish_analysis(
        run_root=run_root,
        cache_root=cache_root,
        output_dir=output_parent / "analysis",
    )
    assert destination.is_dir()
    assert destination.stat().st_mode & 0o222 == 0
    expected_files = {
        "analysis.json",
        "RESULTS.md",
        "manifest.json",
        *(f"{name}.csv" for name in full_grid_analysis._TABLE_NAMES),
    }
    assert {path.name for path in destination.iterdir()} == expected_files
    assert all(path.stat().st_mode & 0o222 == 0 for path in destination.iterdir())
    manifest = json.loads((destination / "manifest.json").read_text("utf-8"))
    assert manifest["schema"] == full_grid_analysis.MANIFEST_SCHEMA
    assert manifest["plan_sha256"] == result.summary["plan_sha256"]
    with pytest.raises(FileExistsError):
        full_grid_analysis.publish_analysis(
            run_root=run_root,
            cache_root=cache_root,
            output_dir=destination,
        )


def test_postpublish_live_input_recheck_invalidates_visible_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_root, cache_root, _ = _compute(
        monkeypatch,
        tmp_path / "compute",
    )
    _install_publication_mocks(monkeypatch, result=result)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    destination = output_parent / "analysis"
    rechecks = 0

    def reject_changed_live_package(*unused: Any, **unused_kwargs: Any) -> None:
        nonlocal rechecks
        rechecks += 1
        raise full_grid.FullGridError(
            "coherent completion package changed during post-analysis recheck"
        )

    monkeypatch.setattr(
        full_grid_analysis,
        "_verify_input_ledger",
        reject_changed_live_package,
    )
    with pytest.raises(
        full_grid.FullGridError,
        match="post-analysis recheck",
    ):
        full_grid_analysis.publish_analysis(
            run_root=run_root,
            cache_root=cache_root,
            output_dir=destination,
        )
    assert rechecks == 1
    assert not destination.exists()
    invalid = list(output_parent.glob(".analysis.analysis-invalid-*"))
    assert len(invalid) == 1 and invalid[0].is_dir()


def test_publish_api_accepts_no_result_or_analysis_setting_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh, run_root, cache_root, _ = _compute(monkeypatch, tmp_path / "compute")
    signature = inspect.signature(full_grid_analysis.publish_analysis)
    assert tuple(signature.parameters) == ("run_root", "cache_root", "output_dir")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    with pytest.raises(TypeError):
        full_grid_analysis.publish_analysis(
            fresh,
            run_root=run_root,
            cache_root=cache_root,
            output_dir=tmp_path / "analysis",
        )


def test_output_publication_gate_inode_replacement_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_root, cache_root, _ = _compute(monkeypatch, tmp_path / "compute")
    _install_publication_mocks(monkeypatch, result=result)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    destination = output_parent / "analysis"
    original_report = full_grid_analysis._markdown_report

    def replace_gate(selected: full_grid_analysis.AnalysisResult) -> str:
        fence = output_parent / ".analysis.analysis-publish.fence"
        displaced = output_parent / ".analysis.analysis-publish.displaced"
        os.replace(fence, displaced)
        fence.write_bytes(b"replacement")
        fence.chmod(0o444)
        return original_report(selected)

    monkeypatch.setattr(full_grid_analysis, "_markdown_report", replace_gate)
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="fence was replaced",
    ):
        full_grid_analysis.publish_analysis(
            run_root=run_root,
            cache_root=cache_root,
            output_dir=destination,
        )
    assert not destination.exists()


def test_output_ancestor_swap_never_publishes_into_replacement_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_root, cache_root, _ = _compute(monkeypatch, tmp_path / "compute")
    _install_publication_mocks(monkeypatch, result=result)
    output_parent = tmp_path / "outputs"
    displaced = tmp_path / "outputs-displaced"
    output_parent.mkdir()
    destination = output_parent / "analysis"
    original = full_grid._atomic_rename_noreplace_at
    swapped = False

    def swap_parent_before_publish(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        nonlocal swapped
        if destination_name == destination.name and not swapped:
            output_parent.rename(displaced)
            output_parent.mkdir()
            swapped = True
        original(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        full_grid,
        "_atomic_rename_noreplace_at",
        swap_parent_before_publish,
    )
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="directory path changed",
    ):
        full_grid_analysis.publish_analysis(
            run_root=run_root,
            cache_root=cache_root,
            output_dir=destination,
        )
    assert not destination.exists()
    assert not (displaced / "analysis").exists()
    invalid = list(displaced.glob(".analysis.analysis-invalid-*"))
    assert len(invalid) == 1 and invalid[0].is_dir()


def test_analysis_publish_uses_linux_nfs_permission_fallback_and_exact_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_root, cache_root, _ = _compute(
        monkeypatch,
        tmp_path / "compute",
    )
    _install_publication_mocks(monkeypatch, result=result)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    destination = output_parent / "analysis"
    original_rename = full_grid._atomic_rename_noreplace_at
    denied = False
    monkeypatch.setattr(full_grid.sys, "platform", "linux")

    def deny_canonical_once(*args: Any) -> None:
        nonlocal denied
        if str(args[3]) == destination.name and not denied:
            denied = True
            raise OSError(errno.EACCES, "synthetic NFS analysis denial")
        original_rename(*args)

    monkeypatch.setattr(
        full_grid,
        "_atomic_rename_noreplace_at",
        deny_canonical_once,
    )
    published = full_grid_analysis.publish_analysis(
        run_root=run_root,
        cache_root=cache_root,
        output_dir=destination,
    )
    assert denied
    assert published == destination
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert {path.name for path in destination.iterdir()} == set(
        full_grid_analysis.ANALYSIS_FILENAMES
    )


def test_analysis_move_then_helper_failure_invalidates_canonical_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_root, cache_root, _ = _compute(
        monkeypatch,
        tmp_path / "compute",
    )
    _install_publication_mocks(monkeypatch, result=result)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    destination = output_parent / "analysis"
    original_publish = full_grid._publish_sealed_directory_noreplace_at
    failed = False

    def move_then_fail(*args: Any, **kwargs: Any) -> None:
        nonlocal failed
        original_publish(*args, **kwargs)
        if str(args[4]) == destination.name and not failed:
            failed = True
            raise OSError(
                errno.EIO,
                "synthetic post-move analysis reseal failure",
            )

    monkeypatch.setattr(
        full_grid,
        "_publish_sealed_directory_noreplace_at",
        move_then_fail,
    )
    with pytest.raises(OSError, match="post-move analysis reseal"):
        full_grid_analysis.publish_analysis(
            run_root=run_root,
            cache_root=cache_root,
            output_dir=destination,
        )
    assert failed
    assert not destination.exists()
    invalid = list(output_parent.glob(".analysis.analysis-invalid-*"))
    assert len(invalid) == 1
    assert stat.S_IMODE(invalid[0].stat().st_mode) == 0o555
    assert {path.name for path in invalid[0].iterdir()} == set(
        full_grid_analysis.ANALYSIS_FILENAMES
    )


def test_analysis_move_then_actual_reseal_failure_recovers_and_invalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, run_root, cache_root, _ = _compute(
        monkeypatch,
        tmp_path / "compute",
    )
    _install_publication_mocks(monkeypatch, result=result)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    destination = output_parent / "analysis"
    original_fchmod = full_grid.os.fchmod
    directory_reseals = 0

    def fail_first_postmove_reseal(descriptor: int, mode: int) -> None:
        nonlocal directory_reseals
        if mode == 0o555:
            directory_reseals += 1
            if directory_reseals == 2:
                raise OSError(errno.EIO, "synthetic actual reseal failure")
        original_fchmod(descriptor, mode)

    monkeypatch.setattr(
        full_grid.os,
        "fchmod",
        fail_first_postmove_reseal,
    )
    with pytest.raises(OSError, match="actual reseal"):
        full_grid_analysis.publish_analysis(
            run_root=run_root,
            cache_root=cache_root,
            output_dir=destination,
        )
    assert not destination.exists()
    invalid = list(output_parent.glob(".analysis.analysis-invalid-*"))
    assert len(invalid) == 1
    assert stat.S_IMODE(invalid[0].stat().st_mode) == 0o555
    assert directory_reseals >= 3


@pytest.mark.parametrize("fence_kind", ("run", "output"))
def test_postrename_fence_loss_invalidates_canonical_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fence_kind: str,
) -> None:
    result, run_root, cache_root, _ = _compute(
        monkeypatch,
        tmp_path / "compute",
    )
    _install_publication_mocks(monkeypatch, result=result)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    destination = output_parent / "analysis"
    original_validate = (
        full_grid_analysis._validate_published_analysis_tree_at
    )

    def validate_then_replace_fence(**kwargs: Any) -> None:
        original_validate(**kwargs)
        if fence_kind == "run":
            fence = run_root / full_grid.PUBLICATION_FENCE_FILENAME
        else:
            fence = output_parent / ".analysis.analysis-publish.fence"
        displaced_fence = fence.with_name(f"{fence.name}.displaced")
        os.replace(fence, displaced_fence)
        fence.write_bytes(b"replacement\n")
        fence.chmod(0o444)

    monkeypatch.setattr(
        full_grid_analysis,
        "_validate_published_analysis_tree_at",
        validate_then_replace_fence,
    )
    with pytest.raises(
        (full_grid.FullGridError, full_grid_analysis.GridAnalysisError),
        match="fence",
    ):
        full_grid_analysis.publish_analysis(
            run_root=run_root,
            cache_root=cache_root,
            output_dir=destination,
        )
    assert not destination.exists()
    invalid = list(output_parent.glob(".analysis.analysis-invalid-*"))
    assert len(invalid) == 1 and invalid[0].is_dir()


@pytest.mark.parametrize("release_kind", ("run", "output"))
def test_postrename_context_release_failure_invalidates_canonical_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_kind: str,
) -> None:
    result, run_root, cache_root, _ = _compute(
        monkeypatch,
        tmp_path / "compute",
    )
    _install_publication_mocks(monkeypatch, result=result)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    destination = output_parent / "analysis"
    if release_kind == "run":
        original_context = full_grid.publication_fence

        @contextmanager
        def failing_run_release(*args: Any, **kwargs: Any):
            with original_context(*args, **kwargs) as descriptor:
                yield descriptor
            raise full_grid.FullGridError("injected run fence release failure")

        monkeypatch.setattr(
            full_grid,
            "publication_fence",
            failing_run_release,
        )
    else:
        original_output_context = full_grid_analysis._output_publication_lock

        @contextmanager
        def failing_output_release(selected: Path):
            with original_output_context(selected) as binding:
                yield binding
            raise full_grid_analysis.GridAnalysisError(
                "injected output fence release failure"
            )

        monkeypatch.setattr(
            full_grid_analysis,
            "_output_publication_lock",
            failing_output_release,
        )
    with pytest.raises(
        (full_grid.FullGridError, full_grid_analysis.GridAnalysisError),
        match="release failure",
    ):
        full_grid_analysis.publish_analysis(
            run_root=run_root,
            cache_root=cache_root,
            output_dir=destination,
        )
    assert not destination.exists()
    invalid = list(output_parent.glob(".analysis.analysis-invalid-*"))
    assert len(invalid) == 1 and invalid[0].is_dir()


def test_output_path_rejects_symlinked_parent_and_no_replace_is_atomic(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(
        full_grid_analysis.GridAnalysisError,
        match="real directory",
    ):
        full_grid_analysis._safe_output_destination(
            alias / "analysis",
            run_root=run_root,
        )

    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    full_grid_analysis._rename_directory_noreplace(source, destination)
    assert destination.is_dir() and not source.exists()
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(FileExistsError):
        full_grid_analysis._rename_directory_noreplace(second, destination)
    assert second.is_dir()


def test_csv_and_parser_surface_are_fixed_and_scalar_only() -> None:
    payload = full_grid_analysis._csv_bytes([{"model": "x", "value": 1.0}])
    assert payload.startswith(b"model,value\r\n")
    parser = full_grid_analysis._parser()
    parsed = parser.parse_args(
        [
            "--run-root",
            "run",
            "--cache-root",
            "cache",
            "--output-dir",
            "output",
        ]
    )
    assert parsed.run_root == Path("run")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--run-root",
                "run",
                "--cache-root",
                "cache",
                "--output-dir",
                "output",
                "--bootstrap-resamples",
                "10",
            ]
        )


def test_all_published_table_schemas_are_exact_scalar_whitelists() -> None:
    assert set(full_grid_analysis._TABLE_SCHEMAS) == full_grid_analysis._TABLE_NAMES
    assert all(schema for schema in full_grid_analysis._TABLE_SCHEMAS.values())
    for forbidden in full_grid_analysis._FORBIDDEN_ARTIFACT_KEYS:
        assert all(
            forbidden not in schema
            for schema in full_grid_analysis._TABLE_SCHEMAS.values()
        )
    assert stat.S_IWUSR & 0o444 == 0
