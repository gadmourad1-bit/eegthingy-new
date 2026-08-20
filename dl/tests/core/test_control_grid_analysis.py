from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmark import control_grid
from benchmark import control_grid_analysis as analysis
from benchmark import full_grid_analysis


def _metrics() -> dict[str, float]:
    return {
        "accuracy": 0.75,
        "balanced_accuracy": 0.75,
        "chance_normalized_balanced_accuracy": 0.5,
        "macro_f1": 0.74,
        "cohen_kappa": 0.5,
        "ovr_macro_auroc": 0.8,
        "nll": 0.6,
        "multiclass_brier": 0.4,
        "ece": 0.1,
    }


def _aggregate_metrics() -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name, value in _metrics().items():
        result[name] = value
        result[f"{name}_defined_count"] = 1
    return result


def _timing_row() -> dict[str, Any]:
    row: dict[str, Any] = {
        "track": analysis.TRACK,
        "dataset": "ALL_DATASETS",
        "control": "control.riemann",
        "jobs": 1,
    }
    values = {
        "candidate_count": 4.0,
        "estimator_fit_calls": 5.0,
        "parameter_count": 12.0,
        "selection_fit_seconds": 1.0,
        "refit_fit_seconds": 0.5,
        "test_inference_seconds": 0.1,
        "job_total_seconds": 1.7,
        "inference_ms_per_trial": 25.0,
    }
    for base, value in values.items():
        for suffix in ("mean", "median", "p95", "min", "max"):
            row[f"{base}_{suffix}"] = value
    return row


def _minimal_result() -> analysis.AnalysisResult:
    metrics = _metrics()
    tables = {
        "job_metrics": [
            {
                "track": analysis.TRACK,
                "dataset": "tiny",
                "control": "control.riemann",
                "subject": 1,
                "fold": 0,
                "job_id": "control-" + "a" * 24,
                "test_count": 4,
                **metrics,
                "candidate_count": 4,
                "estimator_fit_calls": 5,
                "parameter_count": 12,
                "selection_fit_seconds": 1.0,
                "refit_fit_seconds": 0.5,
                "test_inference_seconds": 0.1,
                "job_total_seconds": 1.7,
                "inference_ms_per_trial": 25.0,
            }
        ],
        "fold_metrics": [
            {
                "track": analysis.TRACK,
                "dataset": "tiny",
                "control": "control.riemann",
                "subject": 1,
                "fold": 0,
                "test_count": 4,
                **metrics,
            }
        ],
        "subject_metrics": [
            {
                "track": analysis.TRACK,
                "dataset": "tiny",
                "control": "control.riemann",
                "subject": 1,
                "folds_concatenated": 1,
                "test_trials": 4,
                **metrics,
            }
        ],
        "dataset_summary": [
            {
                "track": analysis.TRACK,
                "dataset": "tiny",
                "control": "control.riemann",
                "subjects_averaged": 1,
                "aggregation": (
                    "concatenate_disjoint_folds_within_subject_then_"
                    "mean_subjects"
                ),
                **_aggregate_metrics(),
            }
        ],
        "overall_summary": [
            {
                "track": analysis.TRACK,
                "control": "control.riemann",
                "datasets_equal_weighted": 1,
                "aggregation": (
                    "fold_concatenation_then_subject_mean_then_"
                    "equal_dataset_mean"
                ),
                **_aggregate_metrics(),
            }
        ],
        "timing_summary": [_timing_row()],
    }
    ledger = [
        {
            "kind": "plan",
            "identity": "b" * 64,
            "sha256_a": "1" * 64,
            "sha256_b": "2" * 64,
            "sha256_c": "b" * 64,
        }
    ]
    summary = {
        "schema": analysis.ANALYSIS_SCHEMA,
        "plan_sha256": "b" * 64,
        "input_ledger_sha256": analysis._sha256_bytes(
            analysis._canonical_bytes(ledger)
        ),
        "track": analysis.TRACK,
        "evidence_scope": "test",
        "confirmation_evidence": False,
        "interpretation": "aggregate-only test result",
        "controls": ["control.riemann"],
        "datasets": ["tiny"],
        "expected_jobs": 1,
        "ece_bins": 15,
        "aggregation_definition": "test",
        "metric_definitions": {"balanced_accuracy": "test"},
        "inferential_comparisons": "omitted",
        "audit_snapshot": {},
        "analysis_source_identity": {},
        "analysis_environment": {},
        "table_row_counts": {
            name: len(rows) for name, rows in tables.items()
        },
    }
    result = analysis.AnalysisResult(
        summary=summary,
        tables=tables,
        input_ledger=ledger,
    )
    result._seal = analysis._analysis_result_seal(
        summary, tables, ledger
    )
    return result


def _reseal(result: analysis.AnalysisResult) -> None:
    result._seal = analysis._analysis_result_seal(
        result.summary,
        result.tables,
        result.input_ledger,
    )


def _mock_audit(
    job: control_grid.Job,
    *,
    complete: bool = True,
    stray: bool = False,
) -> dict[str, Any]:
    return {
        "schema": control_grid.AUDIT_SCHEMA,
        "created_at": "test",
        "plan_sha256": "b" * 64,
        "score_blind": True,
        "expected_jobs": 1,
        "complete_jobs": 1 if complete else 0,
        "missing_jobs": 0 if complete else 1,
        "corrupt_jobs": 0,
        "extra_directories": [],
        "unexpected_record_paths": ["records/rogue"] if stray else [],
        "residual_partial_paths": [],
        "residual_claim_paths": [],
        "residual_cpu_slot_paths": [],
        "unexpected_root_entries": [],
        "complete_job_ids": [job.job_id] if complete else [],
        "missing_job_ids": [] if complete else [job.job_id],
        "corrupt_job_ids": {},
        "expected_per_control": {"control.riemann": 1},
        "complete_per_control": {"control.riemann": 1 if complete else 0},
        "complete_per_dataset": {"tiny": 1 if complete else 0},
        "complete": complete and not stray,
    }


def _sealed_formal_plan(
    contracts: dict[str, Any],
) -> dict[str, Any]:
    caches: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    for dataset, contract in contracts.items():
        for raw_subject in contract["subjects"]:
            subject = int(raw_subject)
            subject_key = control_grid._subject_key(dataset, subject)
            digest = control_grid._sha256_bytes(
                subject_key.encode("utf-8")
            )
            caches[subject_key] = {
                "array_sha256": digest,
                "trial_count": 6,
                "n_channels": 2,
            }
            for raw_fold in contract["folds"]:
                fold = int(raw_fold)
                split_key = control_grid._split_key(
                    dataset, subject, fold
                )
                splits[split_key] = control_grid._one_split_identity(
                    dataset=dataset,
                    subject=subject,
                    fold=fold,
                    cache_array_sha256=digest,
                    trial_count=6,
                    train_rows=np.asarray([0, 1], dtype=np.int64),
                    validation_rows=np.asarray([2], dtype=np.int64),
                    test_rows=np.asarray([3, 4, 5], dtype=np.int64),
                )
    return control_grid.assemble_plan(
        dataset_contracts=contracts,
        controls=control_grid.CONTROLS,
        cache_identity=caches,
        split_identity=splits,
        source_identity={"sealed-test-source": "a" * 64},
        environment_identity={"sealed-test-environment": "uv"},
    )


def _write_readonly_plan(run_root: Path, plan: dict[str, Any]) -> None:
    control_grid.write_or_validate_plan(run_root, plan)
    (run_root / "plan.json").chmod(0o444)
    (run_root / "plan.sha256").chmod(0o444)


def test_multiclass_metrics_match_full_grid_definitions() -> None:
    y = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.70, 0.10, 0.10, 0.10],
            [0.10, 0.65, 0.15, 0.10],
            [0.15, 0.15, 0.55, 0.15],
            [0.10, 0.10, 0.20, 0.60],
            [0.20, 0.50, 0.20, 0.10],
            [0.10, 0.60, 0.20, 0.10],
            [0.10, 0.10, 0.70, 0.10],
            [0.10, 0.20, 0.10, 0.60],
        ],
        dtype=np.float64,
    )
    observed = analysis.classification_metrics(
        y, probabilities, n_classes=4, ece_bins=5
    )
    expected = full_grid_analysis.classification_metrics(
        y, probabilities, n_classes=4, ece_bins=5
    )
    assert observed.keys() == expected.keys()
    for name in observed:
        assert observed[name] == pytest.approx(expected[name])
    assert observed["chance_normalized_balanced_accuracy"] == pytest.approx(
        (observed["balanced_accuracy"] - 0.25) / 0.75
    )


def test_seedless_mini_grid_aggregates_fold_subject_dataset_overall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = control_grid.Job(
        dataset="tiny",
        control="control.riemann",
        subject=1,
        fold=0,
    )
    plan = {
        "plan_sha256": "b" * 64,
        "n_jobs": 1,
        "evidence_scope": "test",
        "dataset_order": ["tiny"],
        "controls": ["control.riemann"],
        "datasets": {
            "tiny": {
                "subjects": [1],
                "folds": [0],
                "n_classes": 4,
            }
        },
    }
    audit = {"complete": True}
    y = np.asarray([0, 1, 2, 3], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ],
        dtype=np.float64,
    )
    cache_ledger = {
        "kind": "cache",
        "identity": "tiny:s001",
        "sha256_a": "1" * 64,
        "sha256_b": "2" * 64,
        "sha256_c": "3" * 64,
    }
    plan_ledger = {
        "kind": "plan",
        "identity": "b" * 64,
        "sha256_a": "4" * 64,
        "sha256_b": "5" * 64,
        "sha256_c": "b" * 64,
    }
    job_ledger = {
        "kind": "job",
        "identity": job.job_id,
        "sha256_a": "6" * 64,
        "sha256_b": "7" * 64,
        "sha256_c": "8" * 64,
    }
    payload = {
        "rows": np.arange(4, dtype=np.int64),
        "probabilities": probabilities,
        "record": {
            "test_count": 4,
            "metadata": {
                "fit": {
                    "selection_candidate_count": 4,
                    "estimator_fit_calls": 5,
                    "parameter_count": 12,
                },
                "timing_seconds": {
                    "selection_fit": 1.0,
                    "refit_fit": 0.5,
                    "test_inference": 0.1,
                    "job_total": 1.7,
                },
            },
        },
    }
    monkeypatch.setattr(
        analysis.control_grid,
        "_require_uv_virtual_environment",
        lambda: None,
    )
    monkeypatch.setattr(
        analysis,
        "audit_exact_grid",
        lambda *unused_args, **unused_kwargs: (plan, (job,), audit),
    )
    monkeypatch.setattr(
        analysis,
        "_verify_live_runtime_identity",
        lambda unused_plan, unused_cache: None,
    )
    monkeypatch.setattr(
        analysis,
        "_bound_cache_labels",
        lambda unused_plan, unused_cache: (
            {("tiny", 1): y},
            [cache_ledger],
        ),
    )
    monkeypatch.setattr(
        analysis.control_grid,
        "validate_completion",
        lambda *unused_args, **unused_kwargs: payload,
    )
    monkeypatch.setattr(
        analysis,
        "_plan_ledger_row",
        lambda unused_root, unused_plan: plan_ledger,
    )
    monkeypatch.setattr(
        analysis,
        "_job_ledger_row",
        lambda *unused_args, **unused_kwargs: job_ledger,
    )
    monkeypatch.setattr(
        analysis, "_verify_input_ledger", lambda **unused: None
    )
    monkeypatch.setattr(
        analysis,
        "_source_identity",
        lambda: {"analysis.py": "9" * 64},
    )
    monkeypatch.setattr(
        analysis,
        "_analysis_environment",
        lambda: {"python": "test"},
    )
    result = analysis.compute_analysis(
        run_root=tmp_path / "run",
        cache_root=tmp_path / "cache",
        ece_bins=5,
    )
    assert {name: len(rows) for name, rows in result.tables.items()} == {
        "job_metrics": 1,
        "fold_metrics": 1,
        "subject_metrics": 1,
        "dataset_summary": 1,
        "overall_summary": 1,
        "timing_summary": 2,
    }
    assert result.tables["overall_summary"][0]["balanced_accuracy"] == 1.0
    assert result.tables["overall_summary"][0][
        "chance_normalized_balanced_accuracy"
    ] == 1.0
    analysis._validate_result_structure(result)


def test_formal_physionet_subject_substitution_is_rejected_before_audit_or_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracts = copy.deepcopy(control_grid._dataset_contracts())
    assert contracts["physionet_mi"]["subjects"][-1] == 54
    contracts["physionet_mi"]["subjects"][-1] = 55
    plan = _sealed_formal_plan(contracts)
    assert plan["n_jobs"] == control_grid.EXPECTED_JOB_COUNT
    _write_readonly_plan(tmp_path, plan)

    def forbidden(*unused_args: object, **unused_kwargs: object) -> None:
        raise AssertionError("structural failure must precede audit/cache access")

    monkeypatch.setattr(analysis.control_grid, "iter_jobs", forbidden)
    monkeypatch.setattr(analysis.control_grid, "audit_grid", forbidden)
    monkeypatch.setattr(
        analysis.control_grid, "_cache_and_split_identity", forbidden
    )
    with pytest.raises(
        analysis.ControlGridAnalysisError,
        match="dataset contracts",
    ):
        analysis.audit_exact_grid(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ("folds", "n_classes", "protocol", "preprocessing"),
)
def test_every_formal_dataset_contract_field_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    contracts = copy.deepcopy(control_grid._dataset_contracts())
    contract = contracts["physionet_mi"]
    if mutation == "folds":
        contract["folds"] = [0, 1, 3]
    elif mutation == "n_classes":
        contract["n_classes"] = int(contract["n_classes"]) + 1
    elif mutation == "protocol":
        contract["protocol"] = f"{contract['protocol']}_drift"
    else:
        contract["preprocessing"] = copy.deepcopy(
            contract["preprocessing"]
        )
        contract["preprocessing"]["sfreq_hz"] = (
            float(contract["preprocessing"]["sfreq_hz"]) + 1.0
        )
    plan = _sealed_formal_plan(contracts)
    assert plan["n_jobs"] == control_grid.EXPECTED_JOB_COUNT
    _write_readonly_plan(tmp_path, plan)
    monkeypatch.setattr(
        analysis.control_grid,
        "iter_jobs",
        lambda unused: (_ for _ in ()).throw(
            AssertionError("contract drift must fail before job iteration")
        ),
    )
    with pytest.raises(
        analysis.ControlGridAnalysisError,
        match="dataset contracts",
    ):
        analysis.audit_exact_grid(tmp_path)


@pytest.mark.parametrize(
    "drift",
    ("registry", "source", "environment", "cache", "splits"),
)
def test_live_runtime_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    contracts = control_grid._dataset_contracts()
    source = {"runner.py": "1" * 64}
    environment = {"python": "uv-test"}
    caches = {"cache": {"array_sha256": "2" * 64}}
    splits = {"split": {"rows_sha256": "3" * 64}}
    plan = {
        "dataset_order": list(control_grid.OPENED_DATASETS),
        "controls": list(control_grid.CONTROLS),
        "datasets": contracts,
        "source_identity": source,
        "environment_identity": environment,
        "cache_identity": caches,
        "split_identity": splits,
        "n_jobs": control_grid.EXPECTED_JOB_COUNT,
    }
    monkeypatch.setattr(
        control_grid, "_require_uv_virtual_environment", lambda: None
    )
    monkeypatch.setattr(
        control_grid,
        "_validate_registry_contract",
        lambda: (
            (_ for _ in ()).throw(
                control_grid.RegistryDriftError("registry drift")
            )
            if drift == "registry"
            else None
        ),
    )
    monkeypatch.setattr(
        control_grid, "_dataset_contracts", lambda: contracts
    )
    monkeypatch.setattr(
        control_grid,
        "_source_identity",
        lambda: {"runner.py": "f" * 64} if drift == "source" else source,
    )
    monkeypatch.setattr(
        control_grid,
        "_environment_identity",
        lambda: (
            {"python": "other"}
            if drift == "environment"
            else environment
        ),
    )
    monkeypatch.setattr(
        control_grid,
        "_cache_and_split_identity",
        lambda unused_root, unused_contracts: (
            {"cache": {"array_sha256": "f" * 64}}
            if drift == "cache"
            else caches,
            {"split": {"rows_sha256": "f" * 64}}
            if drift == "splits"
            else splits,
        ),
    )
    with pytest.raises(
        analysis.ControlGridAnalysisError,
        match="runtime identity",
    ):
        analysis._verify_live_runtime_identity(plan, tmp_path)


def test_runtime_verification_follows_score_blind_audit_and_precedes_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    plan = {"plan_sha256": "b" * 64}
    monkeypatch.setattr(
        control_grid, "_require_uv_virtual_environment", lambda: None
    )
    monkeypatch.setattr(
        analysis,
        "audit_exact_grid",
        lambda *unused_args, **unused_kwargs: (
            events.append("score_blind_audit") or (plan, (), {})
        ),
    )

    def drift(unused_plan: object, unused_cache: object) -> None:
        events.append("runtime_identity")
        raise analysis.ControlGridAnalysisError("forced runtime drift")

    monkeypatch.setattr(analysis, "_verify_live_runtime_identity", drift)
    monkeypatch.setattr(
        analysis,
        "_bound_cache_labels",
        lambda *unused: events.append("labels_opened"),
    )
    with pytest.raises(
        analysis.ControlGridAnalysisError,
        match="runtime drift",
    ):
        analysis.compute_analysis(
            run_root=tmp_path / "run",
            cache_root=tmp_path / "cache",
        )
    assert events == ["score_blind_audit", "runtime_identity"]


def test_publication_window_repeats_runtime_identity_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _minimal_result()
    audit = {"complete": True}
    result.summary["audit_snapshot"] = audit
    _reseal(result)
    events: list[str] = []
    monkeypatch.setattr(
        analysis,
        "audit_exact_grid",
        lambda *unused_args, **unused_kwargs: (
            events.append("score_blind_audit") or ({}, (), audit)
        ),
    )
    monkeypatch.setattr(
        analysis,
        "_verify_live_runtime_identity",
        lambda unused_plan, unused_cache: events.append("runtime_identity"),
    )
    monkeypatch.setattr(
        analysis,
        "_validate_against_plan",
        lambda *unused: events.append("result_binding"),
    )
    monkeypatch.setattr(
        analysis,
        "_verify_input_ledger",
        lambda **unused: events.append("ledger_rehash"),
    )
    analysis._close_publication_input_window(
        result,
        run_root=tmp_path / "run",
        cache_root=tmp_path / "cache",
    )
    assert events == [
        "score_blind_audit",
        "runtime_identity",
        "result_binding",
        "ledger_rehash",
    ]


def test_analysis_provenance_covers_full_uv_and_mne_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert {
        "src/benchmark/control_grid_analysis.py",
        "src/benchmark/full_grid_analysis.py",
        "pyproject.toml",
        "uv.lock",
        *control_grid.SOURCE_FILES,
    }.issubset(analysis.ANALYSIS_SOURCE_FILES)
    packages = [
        [name, f"{index}.0"]
        for index, name in enumerate(
            sorted(control_grid.REQUIRED_DIRECT_DEPENDENCIES), start=1
        )
    ]
    monkeypatch.setattr(
        control_grid,
        "_environment_identity",
        lambda: {
            "python_version": "test",
            "uv_version": "uv test",
            "packages": packages,
            "packages_sha256": "a" * 64,
        },
    )
    observed = analysis._analysis_environment()
    assert observed["full_runtime_identity"]["uv_version"] == "uv test"
    assert observed["required_package_versions"]["mne"] is not None
    assert set(observed["required_package_versions"]) == (
        control_grid.REQUIRED_DIRECT_DEPENDENCIES
    )
    monkeypatch.setattr(
        control_grid,
        "_environment_identity",
        lambda: {
            "python_version": "test",
            "uv_version": None,
            "packages": packages,
            "packages_sha256": "a" * 64,
        },
    )
    with pytest.raises(
        analysis.ControlGridAnalysisError,
        match="UV version",
    ):
        analysis._analysis_environment()


def test_fabricated_raw_or_label_table_is_rejected_even_if_resealed() -> None:
    raw_result = _minimal_result()
    raw_result.tables["job_metrics"][0]["raw_labels"] = [0, 1, 0, 1]
    _reseal(raw_result)
    with pytest.raises(
        analysis.ControlGridAnalysisError, match="exact schema"
    ):
        analysis._validate_result_structure(raw_result)

    label_result = _minimal_result()
    label_result.summary["labels"] = [0, 1]
    _reseal(label_result)
    with pytest.raises(
        analysis.ControlGridAnalysisError,
        match="summary|forbidden",
    ):
        analysis._validate_result_structure(label_result)


def test_corrupted_input_ledger_is_rejected_even_if_resealed() -> None:
    result = _minimal_result()
    result.input_ledger[0]["sha256_a"] = "f" * 64
    _reseal(result)
    with pytest.raises(
        analysis.ControlGridAnalysisError, match="ledger digest"
    ):
        analysis._validate_result_structure(result)


def test_rehashed_corrupted_ledger_cannot_match_fresh_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = _minimal_result()
    corrupt = copy.deepcopy(clean)
    corrupt.input_ledger[0]["sha256_a"] = "f" * 64
    corrupt.summary["input_ledger_sha256"] = analysis._sha256_bytes(
        analysis._canonical_bytes(corrupt.input_ledger)
    )
    _reseal(corrupt)
    monkeypatch.setattr(
        analysis,
        "_close_publication_input_window",
        lambda *unused_args, **unused_kwargs: None,
    )
    monkeypatch.setattr(
        analysis,
        "compute_analysis",
        lambda **unused_kwargs: clean,
    )
    with pytest.raises(
        analysis.ControlGridAnalysisError, match="fresh"
    ):
        analysis._validate_result_for_publication(
            corrupt,
            run_root=tmp_path / "run",
            cache_root=tmp_path / "cache",
        )


def test_no_seed_fields_exist_in_any_publishable_schema_or_result() -> None:
    result = _minimal_result()
    analysis._validate_result_structure(result)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            assert all("seed" not in str(key).lower() for key in value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result.summary)
    visit(result.tables)
    assert all(
        all("seed" not in key for key in schema)
        for schema in analysis._TABLE_SCHEMAS.values()
    )


@pytest.mark.parametrize(
    ("complete", "stray"),
    ((False, False), (True, True)),
)
def test_incomplete_or_stray_run_is_rejected_before_cache_join(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
    stray: bool,
) -> None:
    job = control_grid.Job(
        dataset="tiny",
        control="control.riemann",
        subject=1,
        fold=0,
    )
    plan = {
        "plan_sha256": "b" * 64,
        "n_jobs": 1,
        "controls": ["control.riemann"],
        "dataset_order": ["tiny"],
    }
    monkeypatch.setattr(
        analysis, "_require_regular_readonly", lambda *unused: None
    )
    monkeypatch.setattr(
        analysis.control_grid, "load_plan", lambda unused: plan
    )
    monkeypatch.setattr(
        analysis.control_grid, "iter_jobs", lambda unused: iter((job,))
    )
    monkeypatch.setattr(
        analysis.control_grid,
        "audit_grid",
        lambda *unused_args, **unused_kwargs: _mock_audit(
            job, complete=complete, stray=stray
        ),
    )
    with pytest.raises(
        analysis.ControlGridAnalysisError,
        match="incomplete|stray|corrupt",
    ):
        analysis.audit_exact_grid(
            tmp_path, require_formal_grid=False
        )


def test_output_under_run_or_cache_root_is_rejected(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    cache_root = tmp_path / "cache"
    run_root.mkdir()
    cache_root.mkdir()
    with pytest.raises(
        analysis.ControlGridAnalysisError, match="outside"
    ):
        analysis.publish_analysis(
            _minimal_result(),
            run_root=run_root,
            cache_root=cache_root,
            output_dir=run_root / "analysis",
        )
    with pytest.raises(
        analysis.ControlGridAnalysisError, match="outside"
    ):
        analysis.publish_analysis(
            _minimal_result(),
            run_root=run_root,
            cache_root=cache_root,
            output_dir=cache_root / "analysis",
        )
    alias = tmp_path / "run-alias"
    alias.symlink_to(run_root, target_is_directory=True)
    with pytest.raises(
        analysis.ControlGridAnalysisError, match="outside"
    ):
        analysis.publish_analysis(
            _minimal_result(),
            run_root=run_root,
            cache_root=cache_root,
            output_dir=alias / "analysis",
        )


def test_publisher_emits_only_aggregate_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    cache_root = tmp_path / "cache"
    run_root.mkdir()
    cache_root.mkdir()
    monkeypatch.setattr(
        analysis,
        "_validate_result_for_publication",
        lambda *unused_args, **unused_kwargs: None,
    )
    monkeypatch.setattr(
        analysis,
        "_close_publication_input_window",
        lambda *unused_args, **unused_kwargs: None,
    )
    destination = analysis.publish_analysis(
        _minimal_result(),
        run_root=run_root,
        cache_root=cache_root,
        output_dir=tmp_path / "published",
    )
    expected = {
        "analysis.json",
        "RESULTS.md",
        "manifest.json",
        *{f"{name}.csv" for name in analysis.TABLE_NAMES},
    }
    assert {path.name for path in destination.iterdir()} == expected
    summary = json.loads(
        (destination / "analysis.json").read_text(encoding="utf-8")
    )
    analysis._reject_forbidden_artifact_keys(summary)
    for name in analysis.TABLE_NAMES:
        header = (
            destination / f"{name}.csv"
        ).read_text(encoding="utf-8").splitlines()[0].split(",")
        assert not set(header).intersection(
            analysis._FORBIDDEN_ARTIFACT_KEYS
        )
        assert all("seed" not in column.lower() for column in header)


def test_failed_publisher_does_not_delete_replacement_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    cache_root = tmp_path / "cache"
    output = tmp_path / "published"
    run_root.mkdir()
    cache_root.mkdir()
    lock = (
        tmp_path
        / ".control-analysis-locks"
        / "published.control-analysis-publish.lock"
    )

    def replace_lock(*unused_args: object, **unused_kwargs: object) -> None:
        assert lock.exists()
        lock.unlink()
        lock.write_text("foreign-owner\n", encoding="ascii")
        raise analysis.ControlGridAnalysisError("forced validation failure")

    monkeypatch.setattr(
        analysis, "_validate_result_for_publication", replace_lock
    )
    monkeypatch.setattr(
        analysis,
        "_close_publication_input_window",
        lambda *unused_args, **unused_kwargs: None,
    )
    with pytest.raises(
        analysis.ControlGridAnalysisError, match="forced"
    ):
        analysis.publish_analysis(
            _minimal_result(),
            run_root=run_root,
            cache_root=cache_root,
            output_dir=output,
        )
    assert lock.read_text(encoding="ascii") == "foreign-owner\n"
    assert not output.exists()
