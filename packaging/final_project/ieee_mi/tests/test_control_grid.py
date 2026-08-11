from __future__ import annotations

import copy
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ieee_mi import control_grid


def _split_identity(
    *,
    dataset: str,
    subject: int,
    fold: int,
    digest: str,
    trial_count: int = 8,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "subject": subject,
        "fold": fold,
        "cache_array_sha256": digest,
        "trial_count": trial_count,
        "partitions": {
            "train": {
                "count": 2,
                "rows_sha256": control_grid._rows_sha256([0, 1]),
            },
            "validation": {
                "count": 2,
                "rows_sha256": control_grid._rows_sha256([2, 3]),
            },
            "source": {
                "count": 4,
                "rows_sha256": control_grid._rows_sha256([0, 1, 2, 3]),
            },
            "test": {
                "count": trial_count - 4,
                "rows_sha256": control_grid._rows_sha256(
                    np.arange(4, trial_count, dtype=np.int64)
                ),
            },
        },
    }


def _tiny_plan(
    *,
    n_classes: int = 2,
    controls: tuple[str, ...] = ("control.riemann",),
) -> dict[str, object]:
    digest = "a" * 64
    return control_grid.assemble_plan(
        dataset_contracts={
            "tiny": {
                "subjects": [1],
                "folds": [0],
                "n_classes": n_classes,
                "protocol": "synthetic",
                "preprocessing": {"sfreq_hz": 128.0, "n_times": 96},
            }
        },
        controls=controls,
        cache_identity={
            "tiny:s001": {
                "array_sha256": digest,
                "trial_count": 8,
                "n_channels": 6,
            }
        },
        split_identity={
            "tiny:s001:f00": _split_identity(
                dataset="tiny",
                subject=1,
                fold=0,
                digest=digest,
            )
        },
        source_identity={"ieee_mi/control_grid.py": "b" * 64},
        environment_identity={"python": "synthetic", "uv_version": "uv test"},
        cpu_budget_threads=4,
        threads_per_worker=2,
    )


def _tiny_metadata(
    plan: dict[str, object],
    job: control_grid.Job,
    *,
    n_channels: int = 6,
) -> dict[str, object]:
    candidates = control_grid._admissible_grid(job.control, n_channels)
    threads = plan["execution_config"]["threads_per_worker"]
    return {
        "cache_array_sha256": plan["cache_identity"]["tiny:s001"][
            "array_sha256"
        ],
        "n_channels": n_channels,
        "n_classes": plan["datasets"]["tiny"]["n_classes"],
        "split": copy.deepcopy(plan["split_identity"]["tiny:s001:f00"]),
        "scalers": {
            "selection_mean_sha256": "1" * 64,
            "selection_std_sha256": "2" * 64,
            "refit_mean_sha256": "3" * 64,
            "refit_std_sha256": "4" * 64,
        },
        "fit": {
            "deterministic": True,
            "seed_identity": "none",
            "candidate_grid": [dict(value) for value in candidates],
            "selection_candidate_count": len(candidates),
            "selected_parameters": dict(candidates[0]),
            "selection_state_sha256": "5" * 64,
            "refit_state_sha256": "6" * 64,
            "estimator_fit_calls": len(candidates) + 1,
            "parameter_count": 5,
        },
        "protocol": {
            "selection_scaler_rows": "train_only",
            "selection_estimator_rows": "train_only",
            "selection_decision_rows": "validation_only",
            "selection_objective": "negative_log_likelihood_not_persisted",
            "refit_scaler_rows": "train_plus_validation",
            "refit_estimator_rows": "train_plus_validation",
            "test_use": "single_predict_proba_call_only",
            "transductive_calibration": False,
            "test_performance_computed": False,
        },
        "timing_seconds": {
            "selection_fit": 0.1,
            "refit_fit": 0.1,
            "test_inference": 0.01,
            "job_total": 0.21,
        },
        "runtime": {
            "cpu_threads": threads,
            "thread_environment": {
                name: str(threads)
                for name in control_grid.THREAD_ENVIRONMENT_VARIABLES
            },
            "gpu_reservation": "none_cpu_only",
        },
    }


def _claim(
    tmp_path: Path, plan: dict[str, object], job: control_grid.Job
) -> control_grid.Claim:
    path = control_grid._claim_path(tmp_path, job)
    owner = control_grid._process_identity()
    nonce = "f" * 32
    resource = {
        "disk": {
            "safe": True,
            "path": str(tmp_path.resolve()),
            "free_bytes": 100 * 1024**3,
            "total_bytes": 200 * 1024**3,
            "free_gib": 100.0,
            "minimum_free_gib": 50.0,
            "reason": "safe test disk",
        },
        "cpu": {
            "slot": 0,
            "threads": 2,
            "slot_nonce": "slot",
            "cpu_budget_threads": 4,
        },
        "gpu_reservation": "none_cpu_only",
    }
    control_grid._write_json_exclusive(
        path,
        {
            "schema": control_grid.CLAIM_SCHEMA,
            "created_at": control_grid._utc_now(),
            "plan_sha256": plan["plan_sha256"],
            "job_id": job.job_id,
            "job": job.identity(),
            "nonce": nonce,
            "owner": owner,
            "resource_guard": resource,
        },
    )
    descriptor, observed, payload = control_grid._open_locked_leaf(
        path, nonblocking=True
    )
    return control_grid.Claim(
        job=job,
        path=path,
        nonce=nonce,
        owner=owner,
        resource_guard=resource,
        descriptor=descriptor,
        st_dev=int(observed.st_dev),
        st_ino=int(observed.st_ino),
        payload=payload,
    )


def test_production_grid_has_exact_1344_unique_seedless_jobs(
    tmp_path: Path,
) -> None:
    contracts = control_grid._dataset_contracts()
    caches: dict[str, object] = {}
    splits: dict[str, object] = {}
    for dataset, contract in contracts.items():
        for subject in contract["subjects"]:
            key = control_grid._subject_key(dataset, subject)
            digest = control_grid._sha256_bytes(key.encode("utf-8"))
            caches[key] = {
                "array_sha256": digest,
                "trial_count": 8,
                "n_channels": 6,
            }
            for fold in contract["folds"]:
                splits[control_grid._split_key(dataset, subject, fold)] = (
                    _split_identity(
                        dataset=dataset,
                        subject=subject,
                        fold=fold,
                        digest=digest,
                    )
                )
    plan = control_grid.assemble_plan(
        dataset_contracts=contracts,
        controls=control_grid.CONTROLS,
        cache_identity=caches,
        split_identity=splits,
        source_identity={"runner.py": "c" * 64},
        environment_identity={"python": "test"},
    )
    jobs = tuple(control_grid.iter_jobs(plan))
    assert plan["n_jobs"] == len(jobs) == control_grid.EXPECTED_JOB_COUNT == 1_344
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert Counter(job.control for job in jobs) == {
        control: 448 for control in control_grid.CONTROLS
    }
    assert Counter(job.dataset for job in jobs) == {
        "local_exp4": 24,
        "bnci2014_001": 27,
        "bnci2014_004": 27,
        "cho2017": 780,
        "physionet_mi": 486,
    }
    assert all("seed" not in job.identity() for job in jobs)
    assert plan["seed_policy"] == (
        "one deterministic record; no seed field or seed aliases"
    )
    observed = control_grid.write_or_validate_plan(tmp_path, plan)
    assert observed == plan
    assert observed["dataset_order"] == list(control_grid.OPENED_DATASETS)
    assert list(observed["datasets"]) == list(control_grid.OPENED_DATASETS)
    assert control_grid.load_plan(tmp_path) == plan


def test_registry_controls_are_all_dataset_and_multiclass_eligible() -> None:
    control_grid._validate_registry_contract()


def test_plan_is_canonical_checksummed_and_immutable(tmp_path: Path) -> None:
    plan = _tiny_plan()
    assert plan["plan_sha256"] == control_grid.plan_sha256(plan)
    assert control_grid.write_or_validate_plan(tmp_path, plan) == plan
    assert control_grid.load_plan(tmp_path) == plan
    assert (tmp_path / "plan.json").read_bytes() == (
        control_grid._canonical_bytes(plan) + b"\n"
    )

    changed = copy.deepcopy(plan)
    changed["execution_config"]["cpu_budget_threads"] = 6
    changed["plan_sha256"] = control_grid.plan_sha256(changed)
    with pytest.raises(control_grid.ControlGridError, match="differs"):
        control_grid.write_or_validate_plan(tmp_path, changed)


def test_plan_publication_recovers_either_power_cut_boundary(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    plan_only = tmp_path / "plan-only"
    plan_only.mkdir()
    control_grid._write_json_exclusive(plan_only / "plan.json", plan)
    assert control_grid.write_or_validate_plan(plan_only, plan) == plan
    assert control_grid.load_plan(plan_only) == plan

    digest_only = tmp_path / "digest-only"
    digest_only.mkdir()
    control_grid._write_bytes_exclusive(
        digest_only / "plan.sha256",
        f"{plan['plan_sha256']}\n".encode("ascii"),
    )
    assert control_grid.write_or_validate_plan(digest_only, plan) == plan
    assert control_grid.load_plan(digest_only) == plan


def test_rehashed_plan_cannot_lower_disk_floor_or_change_protocol(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    changed = copy.deepcopy(plan)
    changed["execution_config"]["minimum_free_gib"] = 1.0
    changed["plan_sha256"] = control_grid.plan_sha256(changed)
    control_grid._write_json_exclusive(tmp_path / "plan.json", changed)
    control_grid._write_bytes_exclusive(
        tmp_path / "plan.sha256",
        f"{changed['plan_sha256']}\n".encode("ascii"),
    )
    with pytest.raises(control_grid.ControlGridError, match="minimum_free_gib"):
        control_grid.load_plan(tmp_path)


def test_ea_grid_has_prespecified_channel_feasibility_rule() -> None:
    assert control_grid._admissible_grid("control.ea_fbcsp", 3) == (
        {"n_components": 2},
    )
    assert control_grid._admissible_grid("control.ea_fbcsp", 4) == (
        {"n_components": 2},
        {"n_components": 4},
    )
    with pytest.raises(control_grid.EstimatorCapabilityError):
        control_grid._admissible_grid("control.ea_fbcsp", 1)


def test_atomic_artifact_is_score_blind_and_auditable(tmp_path: Path) -> None:
    plan = _tiny_plan(n_classes=4)
    control_grid.write_or_validate_plan(tmp_path, plan)
    job = next(control_grid.iter_jobs(plan))
    claim = _claim(tmp_path, plan, job)
    rows = np.arange(4, 8, dtype=np.int64)
    probabilities = np.full((4, 4), 0.25, dtype=np.float64)
    output = control_grid.commit_job_output(
        tmp_path,
        plan,
        claim,
        metadata=_tiny_metadata(plan, job),
        test_rows=rows,
        probabilities=probabilities,
    )
    validated = control_grid.validate_completion(tmp_path, plan, job)
    assert np.array_equal(validated["rows"], rows)
    assert np.array_equal(validated["probabilities"], probabilities)
    assert {path.name for path in output.iterdir()} == control_grid.FINAL_FILENAMES
    assert all(path.stat().st_mode & 0o222 == 0 for path in output.iterdir())

    record = json.loads((output / "record.json").read_text(encoding="utf-8"))
    assert control_grid._recursive_forbidden_keys(record) == []
    assert "seed" not in record["job"]
    with np.load(output / "predictions.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"rows", "probabilities"}
    control_grid.release_claim(claim)
    audit = control_grid.audit_grid(tmp_path, plan)
    assert audit["complete"] is True
    assert audit["expected_jobs"] == audit["complete_jobs"] == 1
    assert control_grid._recursive_forbidden_keys(audit) == []


def test_completion_checksum_detects_prediction_corruption(tmp_path: Path) -> None:
    plan = _tiny_plan()
    control_grid.write_or_validate_plan(tmp_path, plan)
    job = next(control_grid.iter_jobs(plan))
    output = control_grid.commit_job_output(
        tmp_path,
        plan,
        _claim(tmp_path, plan, job),
        metadata=_tiny_metadata(plan, job),
        test_rows=np.arange(4, 8, dtype=np.int64),
        probabilities=np.full((4, 2), 0.5, dtype=np.float64),
    )
    prediction_path = output / "predictions.npz"
    prediction_path.chmod(0o600)
    with prediction_path.open("ab") as handle:
        handle.write(b"corruption")
    prediction_path.chmod(0o444)
    with pytest.raises(control_grid.ControlGridError, match="checksums"):
        control_grid.validate_completion(tmp_path, plan, job)


def test_corrupt_staged_prediction_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _tiny_plan()
    control_grid.write_or_validate_plan(tmp_path, plan)
    job = next(control_grid.iter_jobs(plan))

    def corrupt_writer(handle: object, **unused_arrays: np.ndarray) -> None:
        handle.write(b"not-an-npz")

    monkeypatch.setattr(control_grid.np, "savez_compressed", corrupt_writer)
    with pytest.raises(control_grid.ControlGridError, match="cannot load"):
        control_grid.commit_job_output(
            tmp_path,
            plan,
            _claim(tmp_path, plan, job),
            metadata=_tiny_metadata(plan, job),
            test_rows=np.arange(4, 8, dtype=np.int64),
            probabilities=np.full((4, 2), 0.5, dtype=np.float64),
        )
    assert not control_grid._record_directory(tmp_path, job).exists()
    assert not list((tmp_path / "partial").iterdir())
    assert list(
        (tmp_path / "quarantine/abandoned_job_stages").iterdir()
    )


def test_forbidden_outcome_or_score_keys_are_rejected(tmp_path: Path) -> None:
    plan = _tiny_plan()
    control_grid.write_or_validate_plan(tmp_path, plan)
    job = next(control_grid.iter_jobs(plan))
    claim = _claim(tmp_path, plan, job)
    metadata = _tiny_metadata(plan, job)
    metadata["labels"] = [0, 1]
    with pytest.raises(control_grid.ControlGridError, match="score-blind"):
        control_grid.commit_job_output(
            tmp_path,
            plan,
            claim,
            metadata=metadata,
            test_rows=np.arange(4, 8, dtype=np.int64),
            probabilities=np.full((4, 2), 0.5, dtype=np.float64),
        )


def test_low_disk_guard_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    block = 4096
    monkeypatch.setattr(
        control_grid.os,
        "statvfs",
        lambda unused: SimpleNamespace(
            f_bavail=(49 * 1024**3) // block,
            f_blocks=(100 * 1024**3) // block,
            f_frsize=block,
        ),
    )
    status = control_grid.probe_disk(tmp_path, minimum_free_gib=50.0)
    assert status.safe is False
    assert status.free_gib == pytest.approx(49.0)


def test_publication_rechecks_low_disk_after_a_long_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _tiny_plan()
    control_grid.write_or_validate_plan(tmp_path, plan)
    job = next(control_grid.iter_jobs(plan))
    unsafe = control_grid.DiskStatus(
        safe=False,
        path=str(tmp_path.resolve()),
        free_bytes=49 * 1024**3,
        total_bytes=100 * 1024**3,
        free_gib=49.0,
        minimum_free_gib=50.0,
        reason="disk fell below the frozen floor",
    )
    monkeypatch.setattr(
        control_grid,
        "probe_disk",
        lambda unused_path, *, minimum_free_gib: unsafe,
    )
    with pytest.raises(control_grid.DiskUnavailable, match="below"):
        control_grid.commit_job_output(
            tmp_path,
            plan,
            _claim(tmp_path, plan, job),
            metadata=_tiny_metadata(plan, job),
            test_rows=np.arange(4, 8, dtype=np.int64),
            probabilities=np.full((4, 2), 0.5, dtype=np.float64),
        )


def test_cpu_slots_and_job_claims_are_cooperative_and_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _tiny_plan()
    control_grid.write_or_validate_plan(tmp_path, plan)
    safe_disk = control_grid.DiskStatus(
        safe=True,
        path=str(tmp_path),
        free_bytes=100 * 1024**3,
        total_bytes=200 * 1024**3,
        free_gib=100.0,
        minimum_free_gib=50.0,
        reason="safe",
    )
    monkeypatch.setattr(
        control_grid,
        "probe_disk",
        lambda unused_path, *, minimum_free_gib: safe_disk,
    )
    first_slot = control_grid.acquire_cpu_slot(tmp_path, plan)
    second_slot = control_grid.acquire_cpu_slot(tmp_path, plan)
    try:
        with pytest.raises(control_grid.CPUUnavailable):
            control_grid.acquire_cpu_slot(tmp_path, plan)
        job = next(control_grid.iter_jobs(plan))
        claim = control_grid.acquire_claim(
            tmp_path, plan, job, cpu_slot=first_slot
        )
        try:
            with pytest.raises(control_grid.ClaimUnavailable, match="live claim"):
                control_grid.acquire_claim(
                    tmp_path, plan, job, cpu_slot=second_slot
                )
        finally:
            control_grid.release_claim(claim)
    finally:
        control_grid.release_cpu_slot(second_slot)
        control_grid.release_cpu_slot(first_slot)


def test_claim_cannot_bypass_cpu_budget_with_an_unowned_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _tiny_plan()
    control_grid.write_or_validate_plan(tmp_path, plan)
    safe_disk = control_grid.DiskStatus(
        safe=True,
        path=str(tmp_path.resolve()),
        free_bytes=100 * 1024**3,
        total_bytes=200 * 1024**3,
        free_gib=100.0,
        minimum_free_gib=50.0,
        reason="safe",
    )
    monkeypatch.setattr(
        control_grid,
        "probe_disk",
        lambda unused_path, *, minimum_free_gib: safe_disk,
    )
    owner = control_grid._process_identity()
    forged = control_grid.CPUSlot(
        index=0,
        path=control_grid._cpu_slot_path(tmp_path, 0),
        nonce="not-owned",
        owner=owner,
        threads=2,
    )
    with pytest.raises(control_grid.ControlGridError, match="CPU slot"):
        control_grid.acquire_claim(
            tmp_path,
            plan,
            next(control_grid.iter_jobs(plan)),
            cpu_slot=forged,
        )


def test_job_identity_cannot_escape_the_planned_dataset_fold_grid() -> None:
    plan = _tiny_plan()
    with pytest.raises(control_grid.ControlGridError, match="outside"):
        control_grid._validate_job_identity(
            plan,
            control_grid.Job(
                dataset="tiny",
                control="control.riemann",
                subject=1,
                fold=99,
            ),
        )


def test_live_owner_is_not_deleted_when_start_marker_probe_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control_grid.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(
        control_grid, "_process_start_marker", lambda unused_pid: None
    )
    assert control_grid._owner_is_live(
        {
            "host": control_grid.socket.gethostname(),
            "pid": os.getpid(),
            "start_marker": None,
        }
    )


def test_foreign_cpu_slot_owner_is_never_quarantined(tmp_path: Path) -> None:
    plan = _tiny_plan()
    paths: list[Path] = []
    for index in range(plan["execution_config"]["cpu_slot_count"]):
        path = control_grid._cpu_slot_path(tmp_path, index)
        control_grid._write_json_exclusive(
            path,
            {
                "owner": {
                    "host": "other-lab-host.example",
                    "pid": 4321,
                    "start_marker": "foreign",
                }
            },
        )
        paths.append(path)
    with pytest.raises(control_grid.CPUUnavailable):
        control_grid.acquire_cpu_slot(tmp_path, plan)
    assert all(path.exists() for path in paths)
    assert not (tmp_path / "quarantine").exists()


def test_score_blind_validator_rejects_unknown_outcome_alias_after_rehash(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    control_grid.write_or_validate_plan(tmp_path, plan)
    job = next(control_grid.iter_jobs(plan))
    output = control_grid.commit_job_output(
        tmp_path,
        plan,
        _claim(tmp_path, plan, job),
        metadata=_tiny_metadata(plan, job),
        test_rows=np.arange(4, 8, dtype=np.int64),
        probabilities=np.full((4, 2), 0.5, dtype=np.float64),
    )
    record_path = output / "record.json"
    completion_path = output / "completion.json"
    record = control_grid.strict_load(record_path)
    record["metadata"]["hidden_outcome_vector"] = [0, 1, 0, 1]
    record_path.chmod(0o600)
    record_path.write_bytes(control_grid._canonical_bytes(record) + b"\n")
    record_path.chmod(0o444)
    completion = control_grid.strict_load(completion_path)
    completion["files"]["record.json"] = control_grid._sha256_file(record_path)
    completion_path.chmod(0o600)
    completion_path.write_bytes(
        control_grid._canonical_bytes(completion) + b"\n"
    )
    completion_path.chmod(0o444)
    with pytest.raises(control_grid.ControlGridError, match="metadata fields"):
        control_grid.validate_completion(tmp_path, plan, job)


def test_audit_reports_stray_record_tree_and_residual_partial(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    control_grid.write_or_validate_plan(tmp_path, plan)
    stray = tmp_path / "records" / "rogue" / "unfinished.txt"
    stray.parent.mkdir(parents=True)
    stray.write_text("not a record", encoding="utf-8")
    partial = tmp_path / "partial" / "abandoned" / "record.json"
    partial.parent.mkdir(parents=True)
    partial.write_text("{}", encoding="utf-8")
    audit = control_grid.audit_grid(tmp_path, plan)
    assert audit["complete"] is False
    assert audit["unexpected_record_paths"]
    assert audit["residual_partial_paths"]


def test_audit_output_cannot_overwrite_immutable_run_files(
    tmp_path: Path,
) -> None:
    assert control_grid._validated_audit_output_path(
        tmp_path, tmp_path / "audit.json"
    ) == tmp_path / "audit.json"
    with pytest.raises(control_grid.ControlGridError, match="audit.json"):
        control_grid._validated_audit_output_path(
            tmp_path, tmp_path / "plan.json"
        )
    outside = tmp_path.parent / "standalone-control-audit.json"
    assert control_grid._validated_audit_output_path(tmp_path, outside) == outside


def test_tangent_anchor_transitive_source_is_frozen() -> None:
    assert "deepnet/spd.py" in control_grid.SOURCE_FILES


def test_uv_project_contract_requires_direct_and_locked_control_stack(
    tmp_path: Path,
) -> None:
    dependencies = sorted(control_grid.REQUIRED_DIRECT_DEPENDENCIES)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "test-control-grid"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        + "".join(f'  "{name}",\n' for name in dependencies)
        + "]\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text(
        "version = 1\nrevision = 1\n"
        + "".join(f'[[package]]\nname = "{name}"\nversion = "1.0"\n' for name in dependencies),
        encoding="utf-8",
    )
    control_grid._validate_uv_project_contract(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "test-control-grid"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        + "".join(
            f'  "{name}",\n' for name in dependencies if name != "torch"
        )
        + "]\n",
        encoding="utf-8",
    )
    with pytest.raises(control_grid.ControlGridError, match="missing_direct"):
        control_grid._validate_uv_project_contract(tmp_path)


def test_selection_is_train_validation_only_and_final_test_is_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _tiny_plan()
    job = next(control_grid.iter_jobs(plan))
    instances: list[object] = []

    class SpyEstimator:
        def __init__(self) -> None:
            self.classes_ = np.asarray([0, 1], dtype=np.int64)
            self.fit_lengths: list[int] = []
            self.predict_lengths: list[int] = []
            self.coef_ = np.zeros((1, 2), dtype=np.float64)
            self.intercept_ = np.zeros(1, dtype=np.float64)
            instances.append(self)

        def fit(self, x: np.ndarray, y: np.ndarray) -> SpyEstimator:
            self.fit_lengths.append(len(x))
            assert len(x) == len(y)
            return self

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            self.predict_lengths.append(len(x))
            probability = np.full((len(x), 2), 0.5, dtype=np.float64)
            probability[:, 0] += np.linspace(0.1, -0.1, len(x))
            probability[:, 1] = 1.0 - probability[:, 0]
            return probability

    monkeypatch.setattr(
        control_grid,
        "_new_estimator",
        lambda unused_control, unused_parameters: SpyEstimator(),
    )
    monkeypatch.setattr(
        control_grid,
        "_estimator_state_sha256",
        lambda unused_estimator: "e" * 64,
    )
    train = np.zeros((4, 1, 2, 2), dtype=np.float64)
    validation = np.ones((2, 1, 2, 2), dtype=np.float64)
    source = np.zeros((6, 1, 2, 2), dtype=np.float64)
    test = np.full((3, 1, 2, 2), 2.0, dtype=np.float64)
    prepared = control_grid.PreparedFeatures(
        train_epochs=train,
        validation_epochs=validation,
        source_epochs=source,
        test_epochs=test,
        train_covariances=train,
        validation_covariances=validation,
        source_covariances=source,
        test_covariances=test,
        train_outcomes=np.asarray([0, 1, 0, 1], dtype=np.int64),
        validation_outcomes=np.asarray([0, 1], dtype=np.int64),
        source_outcomes=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        test_rows=np.asarray([4, 5, 6], dtype=np.int64),
        selection_mean=np.zeros((1, 2, 1), dtype=np.float32),
        selection_std=np.ones((1, 2, 1), dtype=np.float32),
        refit_mean=np.zeros((1, 2, 1), dtype=np.float32),
        refit_std=np.ones((1, 2, 1), dtype=np.float32),
        channel_names=("C3", "C4"),
    )
    metadata, rows, probabilities = control_grid.select_refit_predict(
        job=job, plan=plan, prepared=prepared
    )
    assert len(instances) == 5  # four validation candidates plus one fresh refit
    assert all(instance.fit_lengths == [4] for instance in instances[:-1])
    assert all(instance.predict_lengths == [2] for instance in instances[:-1])
    assert instances[-1].fit_lengths == [6]
    assert instances[-1].predict_lengths == [3]
    assert metadata["fit"]["estimator_fit_calls"] == 5
    assert control_grid._recursive_forbidden_keys(metadata) == []
    assert np.array_equal(rows, prepared.test_rows)
    assert probabilities.shape == (3, 2)


def _synthetic_four_class_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260729)
    n_classes = 4
    trials_per_class = 8
    bands = 2
    channels = 6
    times = 96
    outcomes = np.repeat(np.arange(n_classes, dtype=np.int64), trials_per_class)
    epochs = rng.normal(
        scale=0.5,
        size=(len(outcomes), bands, channels, times),
    )
    phase = np.linspace(0.0, 8.0 * np.pi, times, endpoint=False)
    for row, outcome in enumerate(outcomes):
        epochs[row, :, outcome, :] += 2.0 * np.sin(phase + outcome * 0.4)
    centered = epochs - epochs.mean(axis=-1, keepdims=True)
    covariance = np.einsum(
        "nbct,nbdt->nbcd", centered, centered, optimize=True
    ) / float(times)
    covariance += 1e-3 * np.eye(channels)[None, None]
    return (
        np.asarray(epochs, dtype=np.float64),
        np.asarray(covariance, dtype=np.float64),
        outcomes,
    )


@pytest.mark.parametrize(
    "control",
    (
        "control.riemann",
        "control.tangent_anchor",
        "control.ea_fbcsp",
    ),
)
def test_actual_registered_estimator_supports_four_classes(
    control: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if (
        control == "control.tangent_anchor"
        and importlib.util.find_spec("torch") is None
    ):
        pytest.skip("local test venv lacks Torch; workstation UV venv exercises this")
    monkeypatch.setenv("MNE_DONTWRITE_HOME", "true")
    epochs, covariances, outcomes = _synthetic_four_class_features()
    parameters = {
        "control.riemann": {"c": 1.0},
        "control.tangent_anchor": {"C": 1.0},
        "control.ea_fbcsp": {"n_components": 2},
    }[control]
    estimator = control_grid._new_estimator(control, parameters)
    features = epochs if control == "control.ea_fbcsp" else covariances
    control_grid._fit_estimator(control, estimator, features, outcomes)
    probabilities = control_grid._valid_probabilities(
        estimator, features[:7], n_classes=4
    )
    assert probabilities.shape == (7, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_worker_supervisor_stops_only_its_owned_siblings_on_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _tiny_plan()
    processes: list[FakeProcess] = []

    class FakeProcess:
        def __init__(self, returncode: int | None) -> None:
            self.returncode = returncode
            self.terminated = False
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0 if self.returncode is None else self.returncode

    launch_codes: list[int | None] = [None, 1]

    def fake_popen(*unused_args: object, **unused_kwargs: object) -> FakeProcess:
        process = FakeProcess(launch_codes[len(processes)])
        processes.append(process)
        return process

    monkeypatch.setattr(control_grid.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(control_grid.time, "sleep", lambda unused: None)
    status = control_grid._spawn_workers(
        run_root=tmp_path,
        cache_root=tmp_path / "cache",
        plan=plan,
        workers=2,
    )
    assert status == 1
    assert processes[0].terminated is True
    assert processes[0].killed is False


def test_source_contains_no_gpu_probe_or_package_install() -> None:
    source = Path(control_grid.__file__).read_text(encoding="utf-8")
    assert "nvidia-smi" not in source
    assert "pip install" not in source
    assert "uv add" not in source
    assert '"CUDA_VISIBLE_DEVICES"] = ""' in source
