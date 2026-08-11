from __future__ import annotations

import contextlib
import copy
import io
import os
import signal
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ieee_mi import hemiq_v2_analysis as analysis
from ieee_mi import hemiq_v2_grid as grid


GPU_UUID = "GPU-576ef18e-b48f-c787-e91d-f93658b27b8e"
OTHER_GPU_UUID = "GPU-0756db6e-5c52-7883-dbc5-94a73e38196b"
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
TRAIN_ROWS = np.arange(10, dtype=np.int64)
VALIDATION_ROWS = np.arange(10, 14, dtype=np.int64)
SOURCE_ROWS = np.arange(14, dtype=np.int64)
TEST_ROWS = np.arange(14, 20, dtype=np.int64)


def _manifest(dtype: str, shape: list[int], token: str = HEX_A) -> dict:
    return {"dtype": dtype, "shape": shape, "sha256": token}


def _preprocessing_manifest() -> dict:
    return {
        "raw_mean": _manifest("<f4", [1, 3, 1]),
        "raw_std": _manifest("<f4", [1, 3, 1]),
        "teacher": {
            "log_reference": _manifest("<f8", [4, 3, 3]),
            "logistic_coef": _manifest("<f8", [1, 24]),
            "logistic_intercept": _manifest("<f8", [1]),
            "scaler_mean": _manifest("<f8", [24]),
            "scaler_scale": _manifest("<f8", [24]),
        },
    }


def _valid_plan() -> dict:
    cache_identity = {}
    split_identity = {}
    spec = grid.dataset_spec(grid.DATASET)
    dataset_identity = {
        "confirmation_subjects": list(spec.confirmation_subjects),
        "development_subjects": list(spec.development_subjects),
        "events": list(spec.events),
        "key": spec.key,
        "moabb_class": spec.moabb_class,
        "n_classes": spec.n_classes,
        "protocol": spec.protocol,
        "subjects": list(spec.subjects),
    }
    for subject in grid.SUBJECTS:
        key = f"{grid.DATASET}:s{subject:03d}"
        cache_identity[key] = {
            "archive": {
                "relative_path": (
                    f"ieee-mi-cache-v2/harmonized/{grid.DATASET}/"
                    f"subject_{subject:03d}.npz"
                ),
                "sha256": HEX_A,
                "size_bytes": 123,
            },
            "arrays": {
                "channel_names": _manifest("<U2", [3]),
                "positions": _manifest("<f4", [20, 3]),
                "runs": _manifest("<U2", [20]),
                "sessions": _manifest("<U6", [20]),
                "x": _manifest("<f4", [20, 3, 320]),
                "y": _manifest("<i8", [20]),
            },
            "embedded_identity": {
                "array_sha256": HEX_B,
                "channels": ["C3", "Cz", "C4"],
                "coordinates": grid.coordinate_contract_for_dataset(
                    grid.DATASET,
                    grid.MONTAGE_PROFILE,
                    ("C3", "Cz", "C4"),
                ),
                "dataset": copy.deepcopy(dataset_identity),
                "montage_profile": "harmonized",
                "preprocessing": grid.preprocessing_for_dataset(
                    grid.DATASET
                ),
                "shape": [20, 3, 320],
                "subject": subject,
            },
            "montage_interpretation": (
                "supplied bipolar derivations; not point-electrode geometry"
            ),
        }
        split_identity[f"{key}:f00"] = {
            "cache_array_sha256": HEX_B,
            "dataset": grid.DATASET,
            "fold": 0,
            "partitions": {
                "source": grid._rows_manifest(SOURCE_ROWS),
                "test": grid._rows_manifest(TEST_ROWS),
                "train": grid._rows_manifest(TRAIN_ROWS),
                "validation": grid._rows_manifest(VALIDATION_ROWS),
            },
            "subject": subject,
            "trial_count": 20,
        }
    source_identity = {
        name: {"sha256": HEX_A, "size_bytes": 1}
        for name in grid.SOURCE_FILES
    }
    uv_identity = {
        "active_venv": "/tmp/formal/.venv",
        "direct_dependencies": copy.deepcopy(grid.DIRECT_DEPENDENCIES),
        "lock": {
            "relative_path": "uv.lock",
            "sha256": HEX_A,
            "size_bytes": 1,
        },
        "project": {
            "relative_path": "pyproject.toml",
            "sha256": HEX_B,
            "size_bytes": 1,
        },
        "python": {
            "executable": "/tmp/formal/.venv/bin/python",
            "implementation": "CPython",
            "version": "3.12.13",
        },
        "python_version_file": {
            "relative_path": ".python-version",
            "sha256": HEX_C,
            "size_bytes": 1,
        },
        "uv_version": "uv 0.11.8",
    }
    gpu = {
        "driver_version": "550.127.05",
        "index": 0,
        "memory_total_mib": 24564,
        "name": "NVIDIA RTX A5000",
        "uuid": GPU_UUID,
    }
    environment_identity = {
        "cuda": {
            "cublas_workspace_config": ":4096:8",
            "cudnn_version": 90100,
            "torch_cuda_version": "12.4",
        },
        "hardware": {
            "architecture": "x86_64",
            "gpus": [gpu],
            "nvidia_driver_version": "550.127.05",
            "system": "Linux",
        },
        "libraries": {
            "numpy": "2.4.4",
            "scipy": "1.18.0",
            "torch": "2.6.0+cu124",
        },
        "parallelism": {
            "distributed_execution": "forbidden",
            "maximum_project_gpu_workers": 3,
        },
    }
    config_identity = {
        "config_file": grid.CONFIG_RELATIVE_PATH,
        "config_file_bytes": 1054,
        "config_file_sha256": (
            "0474b53b06c1f4bc56659f0cc75ff3f908aac67306652714f533c83289af128f"
        ),
        "immutable_settings_sha256": HEX_C,
        "parent_frozen_manifest": {
            "path": "deepnet/results/hemi_q/hemi_q_final_frozen_manifest.json",
            "sha256": (
                "2bbe28d4415a74cb007328579cf8153585ceaaa28219246aecd0622398987401"
            ),
            "size_bytes": 6688,
        },
        "runtime_overrides": ["device", "seed"],
        "schema": grid.CONFIG_SCHEMA,
    }
    return grid.assemble_plan(
        cache_identity=cache_identity,
        split_identity=split_identity,
        source_identity=source_identity,
        uv_identity=uv_identity,
        environment_identity=environment_identity,
        config_identity=config_identity,
        view_contract=grid.hemiq_view_contract(),
    )


def _record(plan: dict, job: grid.Job, rows: np.ndarray, probability: np.ndarray) -> dict:
    gpu = copy.deepcopy(plan["environment_identity"]["hardware"]["gpus"][0])
    source = {
        "raw": _manifest("<f4", [14, 3, 320]),
        "covariance": _manifest("<f4", [14, 4, 3, 3]),
    }
    train = {
        "raw": _manifest("<f4", [10, 3, 320]),
        "covariance": _manifest("<f4", [10, 4, 3, 3]),
    }
    test = {
        "raw": _manifest("<f4", [6, 3, 320]),
        "covariance": _manifest("<f4", [6, 4, 3, 3]),
    }
    return {
        "cache": {
            "archive_sha256": HEX_A,
            "cache_array_sha256": HEX_B,
        },
        "evidence_scope": plan["evidence_scope"],
        "features": {
            "source": source,
            "test": test,
            "train": train,
            "validation_raw": _manifest("<f4", [4, 3, 320]),
        },
        "job": job.as_dict(),
        "model": {
            "config_file_sha256": plan["configuration_identity"][
                "config_file_sha256"
            ],
            "initial_state_sha256": HEX_A,
            "parameter_count": 11_354,
            "stable_id": grid.MODEL_ID,
            "trainable_parameter_count": 11_354,
        },
        "plan_sha256": plan["plan_sha256"],
        "prediction": {
            "call_count": 1,
            "class_order": [0, 1],
            "probabilities": grid._array_manifest(probability),
            "rows": grid._rows_manifest(rows),
            "threshold_logit": 0.0,
        },
        "refit": {
            "epochs_run": 0,
            "initial_state_sha256": HEX_A,
            "mode": "fixed_refit",
            "model_state_sha256": HEX_A,
            "preprocessing_manifest": _preprocessing_manifest(),
            "teacher_status": "used",
        },
        "runtime": {
            "cuda_device_count": 1,
            "device": "cuda:0",
            "gpu": gpu,
            "torch_cuda_version": "12.4",
        },
        "schema": grid.RECORD_SCHEMA,
        "selection": {
            "epochs_run": 0,
            "initial_state_sha256": HEX_A,
            "selected_epoch_count": 0,
            "teacher_status": "used",
        },
        "split": copy.deepcopy(
            plan["split_identity"][
                f"{grid.DATASET}:s{job.subject:03d}:f00"
            ]
        ),
        "timing_seconds": {
            "features": 0.1,
            "prediction": 0.1,
            "refit": 0.1,
            "selection": 0.1,
            "total": 0.4,
        },
        "view_contract_sha256": grid._sha256_bytes(
            grid._canonical_bytes(plan["view_contract"])
        ),
    }


@contextlib.contextmanager
def _fake_guard(_lease):
    yield {"fake": "lease-receipt"}


def _valid_preflight_value(plan: dict, run_root: Path) -> dict:
    return {
        "checks": [
            {"check": name, "status": "pass"}
            for name in (
                "config_identity",
                "cuda_available",
                "deterministic_seeded_reset",
                "deterministic_forward_backward_optimizer_replay",
                "deterministic_probability_replay",
                "exact_reflection_anti_equivariance",
                "float32_spd_covariances",
                "supplied_bipolar_order",
                "zero_epoch_preserved",
            )
        ],
        "created_at": "2026-07-29T00:00:00+00:00",
        "disk_guard": {
            "free_bytes": 100 * 1024**3,
            "minimum_free_bytes": 50 * 1024**3,
            "path": str(run_root.absolute()),
            "status": "pass",
        },
        "gpu_lease_receipt": {"fake": "lease-receipt"},
        "gpu_uuid": GPU_UUID,
        "plan_sha256": plan["plan_sha256"],
        "runtime": {
            "cuda_device_count": 1,
            "device": "cuda:0",
            "gpu": copy.deepcopy(
                plan["environment_identity"]["hardware"]["gpus"][0]
            ),
            "torch_cuda_version": "12.4",
        },
        "schema": grid.PREFLIGHT_SCHEMA,
        "status": "pass",
    }


def _dummy_audit(plan: dict) -> dict:
    count = 1 + grid.EXPECTED_JOBS * len(grid.RESULT_FILENAMES) + 1
    roster = [
        {
            "relative_path": f"dummy/leaf-{index:03d}",
            "sha256": f"{index:064x}",
            "size_bytes": index + 1,
        }
        for index in range(count)
    ]
    return {
        "audited_at": "2026-07-29T00:00:00+00:00",
        "completed_jobs": grid.EXPECTED_JOBS,
        "evidence_scope": plan["evidence_scope"],
        "plan_sha256": plan["plan_sha256"],
        "preflight_gpu_uuids": [GPU_UUID],
        "roster": roster,
        "roster_sha256": grid._sha256_bytes(grid._canonical_bytes(roster)),
        "schema": grid.AUDIT_SCHEMA,
        "status": "pass",
    }


def _valid_analysis_value(plan: dict, audit: dict) -> dict:
    metric_values = {name: 0.0 for name in analysis.METRIC_NAMES}
    seed_rows = [
        {
            "dataset": grid.DATASET,
            "fold": job.fold,
            "n_trials": len(TEST_ROWS),
            "seed": job.seed,
            "subject": job.subject,
            **metric_values,
        }
        for job in grid.expected_jobs()
    ]
    subject_rows = [
        {
            "dataset": grid.DATASET,
            "n_seeds": len(grid.SEEDS),
            "n_trials_per_seed": len(TEST_ROWS),
            "subject": subject,
            **metric_values,
        }
        for subject in grid.SUBJECTS
    ]
    return {
        "audit_roster_sha256": audit["roster_sha256"],
        "dataset_summary": {
            "aggregation": analysis.AGGREGATION_DESCRIPTION,
            "dataset": grid.DATASET,
            "n_jobs": grid.EXPECTED_JOBS,
            "n_seeds": len(grid.SEEDS),
            "n_subjects": len(grid.SUBJECTS),
            **metric_values,
        },
        "evidence_scope": plan["evidence_scope"],
        "metric_contract": {
            "class_order": [0, 1],
            "ece_bins": 15,
            "metrics": list(analysis.METRIC_NAMES),
            "prediction_threshold": 0.5,
        },
        "model_id": grid.MODEL_ID,
        "plan_sha256": plan["plan_sha256"],
        "schema": analysis.ANALYSIS_SCHEMA,
        "subject_metrics": subject_rows,
        "subject_seed_metrics": seed_rows,
        "track": grid.TRACK,
    }


def _patch_lightweight_analysis(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan: dict,
    audit: dict,
) -> None:
    @contextlib.contextmanager
    def fake_fence(**kwargs):
        yield object()

    monkeypatch.setattr(grid, "analysis_fence", fake_fence)
    monkeypatch.setattr(grid, "assert_analysis_fence", lambda *a, **k: None)
    monkeypatch.setattr(
        grid, "audit_grid", lambda **k: copy.deepcopy(audit)
    )
    monkeypatch.setattr(
        analysis,
        "_verify_analysis_execution_identity",
        lambda **k: None,
    )
    monkeypatch.setattr(
        analysis,
        "_compute_after_audit",
        lambda **k: _valid_analysis_value(plan, audit),
    )


def _run_sigkill_child(boundary: str, run_root: Path) -> None:
    script = r"""
import contextlib
import copy
import os
import signal
import sys
from pathlib import Path

from ieee_mi import hemiq_v2_grid as grid
from ieee_mi.tests import test_hemiq_v2_grid as support

boundary = sys.argv[1]
run_root = Path(sys.argv[2])
plan = support._valid_plan()
job = grid.expected_jobs()[0]

if boundary == "preflight":
    preflight = support._valid_preflight_value(plan, run_root)
    grid._assert_lease_bound = lambda *a, **k: None
    grid.verify_static_identity = lambda *a, **k: None
    grid._require_single_gpu_runtime = (
        lambda **k: copy.deepcopy(preflight["runtime"])
    )
    grid._synthetic_cuda_preflight = (
        lambda **k: copy.deepcopy(preflight["checks"])
    )
    grid._hard_disk_guard = lambda *a, **k: preflight["disk_guard"]
    grid.project_gpu_leases.guard_gpu_lease = support._fake_guard
    grid.project_gpu_leases.validate_gpu_lease_receipt = (
        lambda *a, **k: None
    )
elif boundary == "failure":
    grid.project_gpu_leases.guard_gpu_lease = support._fake_guard
elif boundary == "claim":
    grid._assert_lease_bound = lambda *a, **k: None
    grid.validate_preflight = lambda **k: {}
    grid.project_gpu_leases.guard_gpu_lease = support._fake_guard
    grid.project_gpu_leases.validate_gpu_lease_receipt = (
        lambda *a, **k: None
    )

def kill_at_rename(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)

grid._rename_noreplace_at = kill_at_rename
if boundary == "plan":
    grid.write_or_validate_plan(run_root, plan)
elif boundary == "preflight":
    grid.run_preflight(
        plan=plan,
        project_root=run_root.parent,
        cache_root=run_root.parent,
        run_root=run_root,
        gpu_uuid=support.GPU_UUID,
        device="cuda:0",
        lease=object(),
    )
elif boundary == "failure":
    grid._record_failure(
        plan=plan,
        project_root=run_root.parent,
        run_root=run_root,
        job=job,
        gpu_uuid=support.GPU_UUID,
        lease=object(),
        error=RuntimeError("synthetic failure"),
    )
elif boundary == "claim":
    grid.acquire_claim(
        plan=plan,
        project_root=run_root.parent,
        run_root=run_root,
        job=job,
        gpu_uuid=support.GPU_UUID,
        lease=object(),
    )
else:
    raise AssertionError(boundary)
raise AssertionError("SIGKILL boundary was not reached")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, boundary, str(run_root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == -signal.SIGKILL, (
        completed.stdout,
        completed.stderr,
    )


def _rewrite_result_snapshot(
    directory: Path,
    payloads: dict[str, bytes],
) -> None:
    os.chmod(directory, 0o700)
    for name in sorted(grid.RESULT_FILENAMES):
        path = directory / name
        os.chmod(path, 0o600)
        path.write_bytes(payloads[name])
        os.chmod(path, 0o400)
    os.chmod(directory, 0o555)


def _prepare_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict, Path, grid.Job, object, grid.Claim]:
    plan = _valid_plan()
    run_root = tmp_path / "run"
    grid.write_or_validate_plan(run_root, plan)
    preflight = grid._preflight_path(run_root, GPU_UUID)
    preflight.write_bytes(
        grid._canonical_bytes(_valid_preflight_value(plan, run_root))
    )
    os.chmod(preflight, 0o400)
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "validate_preflight", lambda **k: {})
    monkeypatch.setattr(
        grid,
        "_verify_worker_publication_identity",
        lambda **k: None,
    )
    monkeypatch.setattr(
        grid.project_gpu_leases, "guard_gpu_lease", _fake_guard
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )
    lease = object()
    job = grid.expected_jobs()[0]
    claim = grid.acquire_claim(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
    )
    assert claim is not None
    return plan, run_root, job, lease, claim


def _release_claim(
    claim: grid.Claim,
    *,
    plan: dict,
    run_root: Path,
    lease: object,
    project_root: Path,
) -> None:
    grid.release_claim(
        claim,
        plan=plan,
        project_root=project_root,
        run_root=run_root,
        gpu_uuid=GPU_UUID,
        lease=lease,
    )


def _complete_fake_formal_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict, Path]:
    plan = _valid_plan()
    run_root = tmp_path / "formal-run"
    grid.write_or_validate_plan(run_root, plan)
    preflight = grid._preflight_path(run_root, GPU_UUID)
    preflight.write_bytes(
        grid._canonical_bytes(_valid_preflight_value(plan, run_root))
    )
    os.chmod(preflight, 0o400)
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "validate_preflight", lambda **k: {})
    monkeypatch.setattr(
        grid,
        "_verify_worker_publication_identity",
        lambda **k: None,
    )
    monkeypatch.setattr(
        grid.project_gpu_leases, "guard_gpu_lease", _fake_guard
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )
    lease = object()
    probability = np.full((len(TEST_ROWS), 2), 0.5, dtype=np.float64)
    for job in grid.expected_jobs():
        claim = grid.acquire_claim(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=lease,
        )
        assert claim is not None
        grid.commit_job_output(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=lease,
            claim=claim,
            record=_record(plan, job, TEST_ROWS, probability),
            test_rows=TEST_ROWS,
            probabilities=probability,
        )
        _release_claim(
            claim,
            plan=plan,
            run_root=run_root,
            lease=lease,
            project_root=tmp_path,
        )
    return plan, run_root


def test_formal_cardinality_and_order_are_exact() -> None:
    plan = _valid_plan()
    assert plan["n_jobs"] == 45
    assert len(plan["jobs"]) == 45
    assert plan["jobs"][0] == grid.Job(1, 0, 7).as_dict()
    assert plan["jobs"][-1] == grid.Job(9, 0, 47).as_dict()
    assert len({grid._job_from_mapping(value).job_id for value in plan["jobs"]}) == 45


@pytest.mark.parametrize(
    "mutator",
    (
        lambda plan: plan.__setitem__("n_jobs", True),
        lambda plan: plan["dataset_contract"].__setitem__("folds", [False]),
        lambda plan: plan["dataset_contract"]["preprocessing"].__setitem__(
            "epoch_demean", 1
        ),
        lambda plan: plan.__setitem__("seeds", [7, 17, 27, 37, 47, 57]),
        lambda plan: plan["execution"].__setitem__(
            "maximum_concurrent_gpu_workers", 4
        ),
        lambda plan: plan["execution"].__setitem__(
            "minimum_free_gib", 50
        ),
        lambda plan: plan["dataset_contract"].__setitem__("folds", [0, 1]),
        lambda plan: plan["cache_identity"][
            "bnci2014_004:s001"
        ]["embedded_identity"].__setitem__("channels", ["C4", "Cz", "C3"]),
    ),
)
def test_plan_rejects_cardinality_type_scope_and_channel_drift(mutator) -> None:
    plan = _valid_plan()
    mutator(plan)
    plan["plan_sha256"] = grid.plan_sha256(plan)
    with pytest.raises(grid.HemiQGridError):
        grid.validate_plan(plan)


def test_plan_rejects_source_closure_addition() -> None:
    plan = _valid_plan()
    plan["source_identity"]["unexpected.py"] = {
        "sha256": HEX_A,
        "size_bytes": 1,
    }
    plan["plan_sha256"] = grid.plan_sha256(plan)
    with pytest.raises(grid.HemiQGridError, match="source closure"):
        grid.validate_plan(plan)


def test_job_id_rejects_nonformal_subject_seed_and_fold() -> None:
    for value in (
        "bnci2014_004-s010-f00-seed0000000007",
        "bnci2014_004-s001-f01-seed0000000007",
        "bnci2014_004-s001-f00-seed0000000008",
    ):
        with pytest.raises(grid.HemiQGridError):
            grid.parse_job_id(value)


def test_npz_requires_exact_raw_and_numpy_member_order() -> None:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        test_rows=np.arange(2, dtype=np.int64),
        probabilities=np.full((2, 2), 0.5, dtype=np.float64),
    )
    loaded = grid._load_npz_exact(
        buffer.getvalue(),
        expected_keys=("test_rows", "probabilities"),
        source="valid",
    )
    assert list(loaded) == ["test_rows", "probabilities"]
    with pytest.raises(grid.HemiQGridError, match="raw ZIP"):
        grid._load_npz_exact(
            buffer.getvalue(),
            expected_keys=("probabilities", "test_rows"),
            source="reordered",
        )


def test_npz_rejects_duplicate_raw_zip_member() -> None:
    array = io.BytesIO()
    np.save(array, np.arange(2, dtype=np.int64), allow_pickle=False)
    probability = io.BytesIO()
    np.save(
        probability,
        np.full((2, 2), 0.5, dtype=np.float64),
        allow_pickle=False,
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("test_rows.npy", array.getvalue())
        handle.writestr("probabilities.npy", probability.getvalue())
        with pytest.warns(UserWarning, match="Duplicate name"):
            handle.writestr("test_rows.npy", array.getvalue())
    with pytest.raises(grid.HemiQGridError, match="raw ZIP"):
        grid._load_npz_exact(
            archive.getvalue(),
            expected_keys=("test_rows", "probabilities"),
            source="duplicate",
        )


def test_cache_npz_requires_exact_seven_member_roster() -> None:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        x=np.zeros((1, 3, 320), dtype=np.float32),
        y=np.zeros(1, dtype=np.int64),
        positions=np.zeros((3, 3), dtype=np.float32),
        channel_names=np.asarray(["C3", "Cz", "C4"]),
        sessions=np.asarray(["0train"]),
        runs=np.asarray(["0"]),
        identity=np.asarray("{}"),
    )
    loaded = grid._load_npz_exact(
        buffer.getvalue(),
        expected_keys=(
            "x",
            "y",
            "positions",
            "channel_names",
            "sessions",
            "runs",
            "identity",
        ),
        source="cache",
    )
    assert len(loaded) == 7


def test_unique_reader_rejects_symlink_hardlink_and_symlink_ancestor(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"immutable")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(source)
    hardlink = tmp_path / "hardlink"
    hardlink.hardlink_to(source)
    with pytest.raises((OSError, grid.HemiQGridError)):
        grid._read_unique_regular_bytes(symlink)
    with pytest.raises(grid.HemiQGridError, match="unique"):
        grid._read_unique_regular_bytes(hardlink)
    real = tmp_path / "real"
    real.mkdir()
    (real / "leaf").write_bytes(b"value")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises((OSError, grid.HemiQGridError)):
        grid._read_unique_regular_bytes(alias / "leaf")


def test_plan_publication_is_immutable_and_power_cut_cleans_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "run"
    grid.write_or_validate_plan(run_root, plan)
    changed = copy.deepcopy(plan)
    changed["evidence_scope"] += " changed"
    changed["plan_sha256"] = grid.plan_sha256(changed)
    with pytest.raises(grid.HemiQGridError, match="differs"):
        grid.write_or_validate_plan(run_root, changed)
    other = tmp_path / "other"
    original = grid._rename_noreplace_at

    def fail_rename(*args, **kwargs):
        raise OSError("simulated power cut")

    monkeypatch.setattr(grid, "_rename_noreplace_at", fail_rename)
    with pytest.raises(OSError, match="power cut"):
        grid.write_or_validate_plan(other, plan)
    assert not (other / "plan.json").exists()
    assert not any(
        name.startswith(".plan.json.stage-")
        for name in os.listdir(other)
    )
    monkeypatch.setattr(grid, "_rename_noreplace_at", original)


def test_sigkill_plan_stage_is_anchored_quarantined_on_restart(
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "sigkill-plan"
    _run_sigkill_child("plan", run_root)
    assert any(
        name.startswith(".plan.json.stage-")
        for name in os.listdir(run_root)
    )
    assert grid.write_or_validate_plan(run_root, plan) == plan
    assert not any(
        name.startswith(".plan.json.stage-")
        for name in os.listdir(run_root)
    )
    assert len(
        list((run_root / "quarantine/abandoned_plan_stage").iterdir())
    ) == 1


def test_sigkill_preflight_stage_is_quarantined_before_restart_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "sigkill-preflight"
    grid.write_or_validate_plan(run_root, plan)
    preflight = _valid_preflight_value(plan, run_root)
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "verify_static_identity", lambda *a, **k: None)
    monkeypatch.setattr(
        grid,
        "_require_single_gpu_runtime",
        lambda **k: copy.deepcopy(preflight["runtime"]),
    )
    monkeypatch.setattr(
        grid,
        "_synthetic_cuda_preflight",
        lambda **k: copy.deepcopy(preflight["checks"]),
    )
    monkeypatch.setattr(
        grid, "_hard_disk_guard", lambda *a, **k: preflight["disk_guard"]
    )
    monkeypatch.setattr(
        grid.project_gpu_leases, "guard_gpu_lease", _fake_guard
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )

    _run_sigkill_child("preflight", run_root)
    assert any(
        name.startswith(f".{GPU_UUID}.json.stage-")
        for name in os.listdir(run_root / "preflight")
    )
    assert grid.run_preflight(
        plan=plan,
        project_root=tmp_path,
        cache_root=tmp_path,
        run_root=run_root,
        gpu_uuid=GPU_UUID,
        device="cuda:0",
        lease=object(),
    )["status"] == "pass"
    assert len(
        list(
            (
                run_root
                / "quarantine/abandoned_preflight_stage"
            ).iterdir()
        )
    ) == 1


def test_sigkill_failure_stage_is_quarantined_before_restart_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "sigkill-failure"
    grid.write_or_validate_plan(run_root, plan)
    job = grid.expected_jobs()[0]
    monkeypatch.setattr(
        grid.project_gpu_leases, "guard_gpu_lease", _fake_guard
    )

    _run_sigkill_child("failure", run_root)
    assert any(
        name.startswith(f".{job.job_id}-")
        for name in os.listdir(run_root / "failures")
    )
    receipt = grid._record_failure(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=object(),
        error=RuntimeError("synthetic failure"),
    )
    assert receipt.exists()
    assert len(
        list(
            (
                run_root
                / "quarantine/abandoned_failure_stage"
            ).iterdir()
        )
    ) == 1


def test_sigkill_claim_stage_is_quarantined_before_restart_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "sigkill-claim"
    grid.write_or_validate_plan(run_root, plan)
    grid._preflight_path(run_root, GPU_UUID).write_bytes(b"{}")
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "validate_preflight", lambda **k: {})
    monkeypatch.setattr(
        grid.project_gpu_leases, "guard_gpu_lease", _fake_guard
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )
    job = grid.expected_jobs()[0]

    _run_sigkill_child("claim", run_root)
    assert any(
        name.startswith(f".{job.job_id}.")
        for name in os.listdir(run_root / "claims")
    )
    claim = grid.acquire_claim(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=object(),
    )
    assert claim is not None
    assert len(
        list(
            (
                run_root
                / "quarantine/abandoned_claim_stage"
            ).iterdir()
        )
    ) == 1
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=object(),
        project_root=tmp_path,
    )


def test_plan_loader_rejects_writable_plan(tmp_path: Path) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "run"
    grid.write_or_validate_plan(run_root, plan)
    os.chmod(run_root / "plan.json", 0o600)
    with pytest.raises(grid.HemiQGridError, match="writable"):
        grid.load_plan(run_root)


def test_cache_identity_rejects_duplicate_zip_before_shared_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = grid._cache_path(tmp_path, 1)
    path.parent.mkdir(parents=True)
    arrays = {
        "x": np.zeros((1, 3, 320), dtype=np.float32),
        "y": np.zeros(1, dtype=np.int64),
        "positions": np.zeros((3, 3), dtype=np.float32),
        "channel_names": np.asarray(["C3", "Cz", "C4"]),
        "sessions": np.asarray(["0train"]),
        "runs": np.asarray(["0"]),
        "identity": np.asarray("{}"),
    }
    encoded = {}
    for name, value in arrays.items():
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        encoded[name] = buffer.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        for name in arrays:
            archive.writestr(f"{name}.npy", encoded[name])
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("x.npy", encoded["x"])

    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("shared loader should not see ambiguous ZIP")

    monkeypatch.setattr(grid.data_module, "load_subject_cache", forbidden)
    with pytest.raises(grid.HemiQGridError, match="raw ZIP"):
        grid._cache_identity(tmp_path, 1)
    assert not called


def test_preflight_post_guard_loss_quarantines_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "run"
    grid.write_or_validate_plan(run_root, plan)
    runtime = {
        "cuda_device_count": 1,
        "device": "cuda:0",
        "gpu": copy.deepcopy(
            plan["environment_identity"]["hardware"]["gpus"][0]
        ),
        "torch_cuda_version": "12.4",
    }
    checks = [
        {"check": name, "status": "pass"}
        for name in (
            "config_identity",
            "cuda_available",
            "deterministic_seeded_reset",
            "deterministic_forward_backward_optimizer_replay",
            "deterministic_probability_replay",
            "exact_reflection_anti_equivariance",
            "float32_spd_covariances",
            "supplied_bipolar_order",
            "zero_epoch_preserved",
        )
    ]
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "verify_static_identity", lambda *a, **k: None)
    monkeypatch.setattr(
        grid, "_require_single_gpu_runtime", lambda **k: runtime
    )
    monkeypatch.setattr(
        grid, "_synthetic_cuda_preflight", lambda **k: checks
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )

    @contextlib.contextmanager
    def losing_guard(_lease):
        yield {"fake": "receipt"}
        raise RuntimeError("preflight lease loss")

    monkeypatch.setattr(
        grid.project_gpu_leases, "guard_gpu_lease", losing_guard
    )
    with pytest.raises(RuntimeError, match="preflight lease loss"):
        grid.run_preflight(
            plan=plan,
            project_root=tmp_path,
            cache_root=tmp_path,
            run_root=run_root,
            gpu_uuid=GPU_UUID,
            device="cuda:0",
            lease=object(),
        )
    assert not grid._preflight_path(run_root, GPU_UUID).exists()
    assert list(
        (run_root / "quarantine/preflight_authority_loss").iterdir()
    )


def test_preflight_post_guard_replacement_survives_exact_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "preflight-replacement"
    grid.write_or_validate_plan(run_root, plan)
    value = _valid_preflight_value(plan, run_root)
    path = grid._preflight_path(run_root, GPU_UUID)
    displaced = path.with_name(f".{path.name}.owned-displaced")
    replacement_payload = grid._canonical_bytes(
        {"authority": "replacement-preflight"}
    )
    replacement_identity: tuple[int, int] | None = None
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "verify_static_identity", lambda *a, **k: None)
    monkeypatch.setattr(
        grid,
        "_require_single_gpu_runtime",
        lambda **k: copy.deepcopy(value["runtime"]),
    )
    monkeypatch.setattr(
        grid,
        "_synthetic_cuda_preflight",
        lambda **k: copy.deepcopy(value["checks"]),
    )
    monkeypatch.setattr(
        grid,
        "_hard_disk_guard",
        lambda *a, **k: copy.deepcopy(value["disk_guard"]),
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )

    @contextlib.contextmanager
    def replace_after_yield(_lease):
        nonlocal replacement_identity
        yield {"fake": "receipt"}
        os.rename(path, displaced)
        path.write_bytes(replacement_payload)
        os.chmod(path, 0o400)
        status = path.stat()
        replacement_identity = (int(status.st_dev), int(status.st_ino))
        raise RuntimeError("post-yield preflight replacement")

    monkeypatch.setattr(
        grid.project_gpu_leases,
        "guard_gpu_lease",
        replace_after_yield,
    )
    with pytest.raises(
        grid.HemiQGridError,
        match="expected inode",
    ):
        grid.run_preflight(
            plan=plan,
            project_root=tmp_path,
            cache_root=tmp_path,
            run_root=run_root,
            gpu_uuid=GPU_UUID,
            device="cuda:0",
            lease=object(),
        )
    assert path.read_bytes() == replacement_payload
    assert (path.stat().st_dev, path.stat().st_ino) == replacement_identity
    assert displaced.exists()
    assert not (
        run_root / "quarantine/preflight_authority_loss"
    ).exists()


def test_claim_post_guard_loss_is_quarantined_and_reclaimable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "run"
    grid.write_or_validate_plan(run_root, plan)
    grid._preflight_path(run_root, GPU_UUID).write_bytes(b"{}")
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "validate_preflight", lambda **k: {})
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )

    @contextlib.contextmanager
    def losing_guard(_lease):
        yield {"fake": "receipt"}
        raise RuntimeError("claim lease loss")

    monkeypatch.setattr(
        grid.project_gpu_leases, "guard_gpu_lease", losing_guard
    )
    job = grid.expected_jobs()[0]
    with pytest.raises(RuntimeError, match="claim lease loss"):
        grid.acquire_claim(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=object(),
        )
    assert not grid._claim_path(run_root, job).exists()
    assert list((run_root / "quarantine/claim_authority_loss").iterdir())
    monkeypatch.setattr(
        grid.project_gpu_leases, "guard_gpu_lease", _fake_guard
    )
    lease = object()
    claim = grid.acquire_claim(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
    )
    assert claim is not None
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )


def test_live_claim_is_exclusive_and_release_removes_exact_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    with pytest.raises(grid.ClaimUnavailable):
        grid.acquire_claim(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=lease,
        )
    inode = claim.st_ino
    assert grid._anchored_lstat(claim.path).st_ino == inode
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )
    assert not claim.path.exists()


def test_stale_unlocked_claim_is_quarantined_before_reclaim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, first = _prepare_claim(monkeypatch, tmp_path)
    fcntl = __import__("fcntl")
    fcntl.flock(first.descriptor, fcntl.LOCK_UN)
    os.close(first.descriptor)
    second = grid.acquire_claim(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
    )
    assert second is not None
    assert second.st_ino != first.st_ino
    assert len(list((run_root / "quarantine/stale_claim").iterdir())) == 1
    _release_claim(
        second,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )


def test_completed_result_preserves_live_claim_then_recovers_exact_unlocked_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    rows = TEST_ROWS.copy()
    probability = np.full((len(rows), 2), 0.5, dtype=np.float64)
    original_unlink = grid._unlink_claim_path
    monkeypatch.setattr(grid, "_unlink_claim_path", lambda value: None)
    destination = grid.commit_job_output(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
        claim=claim,
        record=_record(plan, job, rows, probability),
        test_rows=rows,
        probabilities=probability,
    )
    assert destination.exists()
    assert claim.path.exists()
    with pytest.raises(grid.ClaimUnavailable, match="live claim"):
        grid.acquire_claim(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=lease,
        )

    fcntl = __import__("fcntl")
    fcntl.flock(claim.descriptor, fcntl.LOCK_UN)
    os.close(claim.descriptor)
    monkeypatch.setattr(grid, "_unlink_claim_path", original_unlink)
    assert (
        grid.acquire_claim(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=lease,
        )
        is None
    )
    assert not claim.path.exists()
    quarantined = list(
        (run_root / "quarantine/completed_claim").iterdir()
    )
    assert len(quarantined) == 1
    assert quarantined[0].stat().st_ino == claim.st_ino


def test_analysis_fence_blocks_new_claim_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "run"
    grid.write_or_validate_plan(run_root, plan)
    (grid._preflight_path(run_root, GPU_UUID)).write_bytes(b"{}")
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "validate_preflight", lambda **k: {})
    monkeypatch.setattr(
        grid.project_gpu_leases, "guard_gpu_lease", _fake_guard
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )
    with grid.analysis_fence(run_root=run_root, plan=plan):
        with pytest.raises(grid.AnalysisActive):
            grid.acquire_claim(
                plan=plan,
                project_root=tmp_path,
                run_root=run_root,
                job=grid.expected_jobs()[0],
                gpu_uuid=GPU_UUID,
                lease=object(),
            )


def test_record_rejects_outcome_alias_and_bool_epoch() -> None:
    plan = _valid_plan()
    job = grid.expected_jobs()[0]
    rows = TEST_ROWS.copy()
    probability = np.full((6, 2), 0.5, dtype=np.float64)
    record = _record(plan, job, rows, probability)
    record["features"]["test"]["labels"] = HEX_A
    with pytest.raises(grid.HemiQGridError, match="outcome aliases"):
        grid._validate_record(
            record,
            plan=plan,
            job=job,
            test_rows=rows,
            probabilities=probability,
        )
    record = _record(plan, job, rows, probability)
    record["selection"]["selected_epoch_count"] = False
    with pytest.raises(grid.HemiQGridError, match="reset/refit"):
        grid._validate_record(
            record,
            plan=plan,
            job=job,
            test_rows=rows,
            probabilities=probability,
        )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value["prediction"].__setitem__("call_count", True),
        lambda value: value["prediction"].__setitem__(
            "class_order", [False, True]
        ),
        lambda value: value["prediction"].__setitem__(
            "threshold_logit", False
        ),
        lambda value: value["runtime"].__setitem__(
            "cuda_device_count", True
        ),
        lambda value: value["runtime"]["gpu"].__setitem__("index", False),
        lambda value: value["runtime"]["gpu"].__setitem__(
            "memory_total_mib", 1
        ),
        lambda value: value["runtime"]["gpu"].__setitem__(
            "driver_version", "forged"
        ),
        lambda value: value["runtime"]["gpu"].__setitem__(
            "name", "forged"
        ),
        lambda value: value["timing_seconds"].__setitem__("total", 0),
    ),
)
def test_record_rejects_bool_and_forged_runtime_gpu_rows(mutator) -> None:
    plan = _valid_plan()
    job = grid.expected_jobs()[0]
    probability = np.full((len(TEST_ROWS), 2), 0.5, dtype=np.float64)
    record = _record(plan, job, TEST_ROWS, probability)
    mutator(record)
    with pytest.raises(grid.HemiQGridError):
        grid._validate_record(
            record,
            plan=plan,
            job=job,
            test_rows=TEST_ROWS,
            probabilities=probability,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("cuda_device_count", True),
        ("index", False),
        ("index", 1),
        ("memory_total_mib", 1),
        ("driver_version", "forged"),
        ("name", "forged"),
    ),
)
def test_preflight_rejects_bool_and_forged_runtime_gpu_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "preflight-types"
    grid.write_or_validate_plan(run_root, plan)
    preflight = _valid_preflight_value(plan, run_root)
    if field == "cuda_device_count":
        preflight["runtime"][field] = value
    else:
        preflight["runtime"]["gpu"][field] = value
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )
    with pytest.raises(grid.HemiQGridError):
        grid._validate_preflight_value(
            preflight,
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            gpu_uuid=GPU_UUID,
        )


def test_commit_fsyncs_and_publishes_read_only_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    rows = TEST_ROWS.copy()
    probability = np.full((6, 2), 0.5, dtype=np.float64)
    original_rename = grid._rename_noreplace_at
    same_parent_seen = False

    def inspect_rename(source_parent, source_name, destination_parent, destination_name):
        nonlocal same_parent_seen
        if destination_name == job.job_id:
            source = os.fstat(source_parent)
            target = os.fstat(destination_parent)
            assert (source.st_dev, source.st_ino) == (
                target.st_dev,
                target.st_ino,
            )
            same_parent_seen = True
        return original_rename(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
        )

    monkeypatch.setattr(grid, "_rename_noreplace_at", inspect_rename)
    destination = grid.commit_job_output(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
        claim=claim,
        record=_record(plan, job, rows, probability),
        test_rows=rows,
        probabilities=probability,
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert same_parent_seen
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o400
        for path in destination.iterdir()
    )
    validated = grid.validate_completion(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
    )
    assert np.array_equal(validated["test_rows"], rows)
    assert {
        entry["relative_path"]
        for entry in validated["artifact_roster"]
    } == {
        f"records/{job.job_id}/{name}"
        for name in grid.RESULT_FILENAMES
    }
    assert not claim.path.exists()
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "nan",
        "inf",
        "wrong_count",
        "out_of_range",
        "bad_sum",
        "duplicate_rows",
        "wrong_probability_shape",
        "wrong_probability_dtype",
        "wrong_row_dtype",
    ),
)
def test_completion_rejects_coherently_resealed_prediction_corruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    corruption: str,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    rows = TEST_ROWS.copy()
    probability = np.full((len(rows), 2), 0.5, dtype=np.float64)
    destination = grid.commit_job_output(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
        claim=claim,
        record=_record(plan, job, rows, probability),
        test_rows=rows,
        probabilities=probability,
    )
    forged_rows = rows.copy()
    forged_probability = probability.copy()
    if corruption == "nan":
        forged_probability[0, 0] = np.nan
    elif corruption == "inf":
        forged_probability[0, 0] = np.inf
    elif corruption == "wrong_count":
        forged_rows = forged_rows[:-1]
        forged_probability = forged_probability[:-1]
    elif corruption == "out_of_range":
        forged_probability[0] = (-1.0, 2.0)
    elif corruption == "bad_sum":
        forged_probability[0] = (0.2, 0.2)
    elif corruption == "duplicate_rows":
        forged_rows[1] = forged_rows[0]
    elif corruption == "wrong_probability_shape":
        forged_probability = forged_probability[:, :1]
    elif corruption == "wrong_probability_dtype":
        forged_probability = forged_probability.astype(np.float32)
    elif corruption == "wrong_row_dtype":
        forged_rows = forged_rows.astype(np.int32)
    else:
        raise AssertionError(corruption)

    prediction_buffer = io.BytesIO()
    np.savez(
        prediction_buffer,
        test_rows=forged_rows,
        probabilities=forged_probability,
    )
    prediction_payload = prediction_buffer.getvalue()
    record = grid._strict_json_file(destination / "record.json")
    record["prediction"]["rows"] = grid._array_manifest(forged_rows)
    record["prediction"]["probabilities"] = grid._array_manifest(
        forged_probability
    )
    record_payload = grid._canonical_bytes(record)
    completion = grid._strict_json_file(destination / "completion.json")
    completion["artifacts"] = {
        "predictions.npz": {
            "sha256": grid._sha256_bytes(prediction_payload),
            "size_bytes": len(prediction_payload),
        },
        "record.json": {
            "sha256": grid._sha256_bytes(record_payload),
            "size_bytes": len(record_payload),
        },
    }
    completion["seal_sha256"] = grid._sha256_bytes(
        grid._canonical_bytes(completion["artifacts"])
    )
    _rewrite_result_snapshot(
        destination,
        {
            "completion.json": grid._canonical_bytes(completion),
            "predictions.npz": prediction_payload,
            "record.json": record_payload,
        },
    )
    with pytest.raises(grid.HemiQGridError):
        grid.validate_completion(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
        )
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )


def test_completion_rejects_writable_final_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    rows = TEST_ROWS.copy()
    probability = np.full((6, 2), 0.5, dtype=np.float64)
    destination = grid.commit_job_output(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
        claim=claim,
        record=_record(plan, job, rows, probability),
        test_rows=rows,
        probabilities=probability,
    )
    os.chmod(destination, 0o755)
    with pytest.raises(grid.HemiQGridError, match="writable or replaced"):
        grid.validate_completion(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
        )
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )


def test_completion_rejects_replaced_final_directory_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    rows = TEST_ROWS.copy()
    probability = np.full((6, 2), 0.5, dtype=np.float64)
    destination = grid.commit_job_output(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
        claim=claim,
        record=_record(plan, job, rows, probability),
        test_rows=rows,
        probabilities=probability,
    )
    original = destination.with_name(destination.name + ".original")
    destination.rename(original)
    destination.mkdir(mode=0o700)
    for leaf in original.iterdir():
        shutil.copyfile(leaf, destination / leaf.name)
        os.chmod(destination / leaf.name, 0o400)
    os.chmod(destination, 0o555)
    with pytest.raises(grid.HemiQGridError, match="completion identity"):
        grid.validate_completion(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
        )
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )


def test_power_cut_before_result_rename_quarantines_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    rows = TEST_ROWS.copy()
    probability = np.full((6, 2), 0.5, dtype=np.float64)
    original = grid._rename_noreplace_at

    def fail_result_rename(src_fd, src_name, dst_fd, dst_name):
        if dst_name == job.job_id:
            raise OSError("simulated power cut")
        return original(src_fd, src_name, dst_fd, dst_name)

    monkeypatch.setattr(grid, "_rename_noreplace_at", fail_result_rename)
    with pytest.raises(OSError, match="power cut"):
        grid.commit_job_output(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=lease,
            claim=claim,
            record=_record(plan, job, rows, probability),
            test_rows=rows,
            probabilities=probability,
        )
    assert not grid._record_directory(run_root, job).exists()
    assert list((run_root / "quarantine/failed_stage").iterdir())
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )


def test_post_guard_lease_loss_quarantines_published_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    rows = TEST_ROWS.copy()
    probability = np.full((6, 2), 0.5, dtype=np.float64)

    @contextlib.contextmanager
    def losing_guard(_lease):
        yield {"fake": "lease-receipt"}
        raise RuntimeError("lease lost after publication")

    monkeypatch.setattr(
        grid.project_gpu_leases, "guard_gpu_lease", losing_guard
    )
    with pytest.raises(RuntimeError, match="lease lost"):
        grid.commit_job_output(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=lease,
            claim=claim,
            record=_record(plan, job, rows, probability),
            test_rows=rows,
            probabilities=probability,
        )
    assert not grid._record_directory(run_root, job).exists()
    assert list(
        (run_root / "quarantine/post_commit_authority_loss").iterdir()
    )
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )


def test_claim_post_guard_replacement_survives_exact_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "claim-replacement"
    grid.write_or_validate_plan(run_root, plan)
    grid._preflight_path(run_root, GPU_UUID).write_bytes(b"{}")
    job = grid.expected_jobs()[0]
    path = grid._claim_path(run_root, job)
    displaced = path.with_name(f".{path.name}.owned-displaced")
    replacement_payload = grid._canonical_bytes(
        {"authority": "replacement-claim"}
    )
    replacement_identity: tuple[int, int] | None = None
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "validate_preflight", lambda **k: {})
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )

    @contextlib.contextmanager
    def replace_after_yield(_lease):
        nonlocal replacement_identity
        yield {"fake": "receipt"}
        os.rename(path, displaced)
        path.write_bytes(replacement_payload)
        os.chmod(path, 0o400)
        status = path.stat()
        replacement_identity = (int(status.st_dev), int(status.st_ino))
        raise RuntimeError("post-yield claim replacement")

    monkeypatch.setattr(
        grid.project_gpu_leases,
        "guard_gpu_lease",
        replace_after_yield,
    )
    with pytest.raises(
        grid.HemiQGridError,
        match="expected inode",
    ):
        grid.acquire_claim(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=object(),
        )
    assert path.read_bytes() == replacement_payload
    assert (path.stat().st_dev, path.stat().st_ino) == replacement_identity
    assert displaced.exists()
    assert not (run_root / "quarantine/claim_authority_loss").exists()


@pytest.mark.parametrize("fault", ("move_then_raise", "parent_fsync"))
def test_claim_publication_failure_hides_exact_canonical_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / f"claim-publication-{fault}"
    grid.write_or_validate_plan(run_root, plan)
    grid._preflight_path(run_root, GPU_UUID).write_bytes(b"{}")
    job = grid.expected_jobs()[0]
    path = grid._claim_path(run_root, job)
    original_rename = grid._rename_noreplace_at
    original_fsync = os.fsync
    tripped = False
    published_identity: tuple[int, int] | None = None
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "validate_preflight", lambda **k: {})
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "guard_gpu_lease",
        _fake_guard,
    )

    def move_then_raise(
        source_parent: int,
        source_name: str,
        destination_parent: int,
        destination_name: str,
    ) -> None:
        nonlocal tripped, published_identity
        original_rename(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
        )
        if (
            fault == "move_then_raise"
            and destination_name == path.name
            and source_name.endswith(".stage")
            and not tripped
        ):
            observed = os.stat(
                destination_name,
                dir_fd=destination_parent,
                follow_symlinks=False,
            )
            published_identity = (
                int(observed.st_dev),
                int(observed.st_ino),
            )
            tripped = True
            raise OSError("injected claim move-then-raise")

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal tripped, published_identity
        if fault == "parent_fsync" and path.exists() and not tripped:
            observed = path.stat()
            published_identity = (
                int(observed.st_dev),
                int(observed.st_ino),
            )
            tripped = True
            raise OSError("injected claim parent-fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(grid, "_rename_noreplace_at", move_then_raise)
    monkeypatch.setattr(os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match=fault.replace("_", "[- ]")):
        grid.acquire_claim(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=object(),
        )
    assert tripped
    assert published_identity is not None
    assert not path.exists()
    assert not any(
        candidate.name.startswith(f".{job.job_id}.")
        and candidate.name.endswith(".stage")
        for candidate in (run_root / "claims").iterdir()
    )
    quarantined = list(
        (run_root / "quarantine/claim_authority_loss").iterdir()
    )
    assert len(quarantined) == 1
    assert (
        quarantined[0].stat().st_dev,
        quarantined[0].stat().st_ino,
    ) == published_identity

    monkeypatch.setattr(grid, "_rename_noreplace_at", original_rename)
    monkeypatch.setattr(os, "fsync", original_fsync)
    claim = grid.acquire_claim(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=object(),
    )
    assert claim is not None
    grid.release_claim(
        claim,
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        gpu_uuid=GPU_UUID,
        lease=object(),
    )


@pytest.mark.parametrize("fault", ("move_then_raise", "parent_fsync"))
def test_claim_publication_failure_preserves_replacement_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / f"claim-publication-replacement-{fault}"
    grid.write_or_validate_plan(run_root, plan)
    grid._preflight_path(run_root, GPU_UUID).write_bytes(b"{}")
    job = grid.expected_jobs()[0]
    path = grid._claim_path(run_root, job)
    displaced = path.with_name(f".{path.name}.owned-displaced")
    replacement_payload = grid._canonical_bytes(
        {"authority": f"replacement-{fault}"}
    )
    original_rename = grid._rename_noreplace_at
    original_fsync = os.fsync
    tripped = False
    owned_identity: tuple[int, int] | None = None
    replacement_identity: tuple[int, int] | None = None
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "validate_preflight", lambda **k: {})
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "guard_gpu_lease",
        _fake_guard,
    )

    def install_replacement() -> None:
        nonlocal owned_identity, replacement_identity
        owned = path.stat()
        owned_identity = (int(owned.st_dev), int(owned.st_ino))
        os.rename(path, displaced)
        path.write_bytes(replacement_payload)
        os.chmod(path, 0o400)
        replacement = path.stat()
        replacement_identity = (
            int(replacement.st_dev),
            int(replacement.st_ino),
        )

    def move_replace_then_raise(
        source_parent: int,
        source_name: str,
        destination_parent: int,
        destination_name: str,
    ) -> None:
        nonlocal tripped
        original_rename(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
        )
        if (
            fault == "move_then_raise"
            and destination_name == path.name
            and source_name.endswith(".stage")
            and not tripped
        ):
            install_replacement()
            tripped = True
            raise OSError("injected claim move-then-raise replacement")

    def replace_before_parent_fsync(descriptor: int) -> None:
        nonlocal tripped
        if fault == "parent_fsync" and path.exists() and not tripped:
            install_replacement()
            tripped = True
            raise OSError("injected claim parent-fsync replacement")
        original_fsync(descriptor)

    monkeypatch.setattr(grid, "_rename_noreplace_at", move_replace_then_raise)
    monkeypatch.setattr(os, "fsync", replace_before_parent_fsync)
    with pytest.raises(OSError, match=fault.replace("_", "[- ]")):
        grid.acquire_claim(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=object(),
        )
    assert tripped
    assert owned_identity is not None
    assert replacement_identity is not None
    assert path.read_bytes() == replacement_payload
    assert (path.stat().st_dev, path.stat().st_ino) == replacement_identity
    assert (displaced.stat().st_dev, displaced.stat().st_ino) == owned_identity
    assert not (run_root / "quarantine/claim_authority_loss").exists()


def test_failure_post_guard_replacement_survives_exact_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "failure-replacement"
    grid.write_or_validate_plan(run_root, plan)
    job = grid.expected_jobs()[0]
    replacement_payload = grid._canonical_bytes(
        {"authority": "replacement-failure"}
    )
    displaced: Path | None = None
    path: Path | None = None
    replacement_identity: tuple[int, int] | None = None

    @contextlib.contextmanager
    def replace_after_yield(_lease):
        nonlocal displaced, path, replacement_identity
        yield {"fake": "receipt"}
        visible = sorted(
            candidate
            for candidate in (run_root / "failures").iterdir()
            if candidate.name.endswith(".json")
            and not candidate.name.startswith(".")
        )
        assert len(visible) == 1
        path = visible[0]
        displaced = path.with_name(f".{path.name}.owned-displaced")
        os.rename(path, displaced)
        path.write_bytes(replacement_payload)
        os.chmod(path, 0o400)
        status = path.stat()
        replacement_identity = (int(status.st_dev), int(status.st_ino))
        raise RuntimeError("post-yield failure replacement")

    monkeypatch.setattr(
        grid.project_gpu_leases,
        "guard_gpu_lease",
        replace_after_yield,
    )
    with pytest.raises(
        grid.HemiQGridError,
        match="expected inode",
    ):
        grid._record_failure(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=object(),
            error=RuntimeError("synthetic"),
        )
    assert path is not None and displaced is not None
    assert path.read_bytes() == replacement_payload
    assert (path.stat().st_dev, path.stat().st_ino) == replacement_identity
    assert displaced.exists()
    assert not (run_root / "quarantine/failure_authority_loss").exists()


def test_execute_job_preserves_zero_epoch_reset_source_only_and_predict_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _valid_plan()
    job = grid.expected_jobs()[0]
    x = np.zeros((20, 3, 320), dtype=np.float32)
    y = np.arange(20, dtype=np.int64) % 2
    cache = {
        "x": x,
        "y": y,
        "channel_names": np.asarray(["C3", "Cz", "C4"]),
        "identity": {"array_sha256": HEX_B},
    }
    train = np.arange(10, dtype=np.int64)
    validation = np.arange(10, 14, dtype=np.int64)
    test = np.arange(14, 20, dtype=np.int64)
    monkeypatch.setattr(
        grid,
        "_require_single_gpu_runtime",
        lambda **k: {
            "cuda_device_count": 1,
            "device": "cuda:0",
            "gpu": copy.deepcopy(
                plan["environment_identity"]["hardware"]["gpus"][0]
            ),
            "torch_cuda_version": "12.4",
        },
    )
    monkeypatch.setattr(
        grid,
        "_verify_loaded_cache",
        lambda *a, **k: (cache, (train, validation, test)),
    )

    def views(raw, *, channel_names):
        assert tuple(channel_names) == ("C3", "Cz", "C4")
        return {
            "raw": np.ascontiguousarray(raw, dtype=np.float32),
            "covariance": np.tile(
                np.eye(3, dtype=np.float32),
                (len(raw), 4, 1, 1),
            ),
        }

    monkeypatch.setattr(grid, "derive_hemiq_views", views)
    config = SimpleNamespace(epochs=240)
    identity = copy.deepcopy(plan["configuration_identity"])
    identity["runtime_overrides"] = {"device": "cuda:0", "seed": job.seed}
    monkeypatch.setattr(
        grid, "load_frozen_config", lambda *a, **k: (config, identity)
    )
    instances = []

    class FakeClassifier:
        predict_calls = 0

        def __init__(self, _config):
            instances.append(self)

        def fit_selection(
            self,
            raw_train,
            covariance_train,
            labels_train,
            raw_validation,
            labels_validation,
        ):
            assert len(raw_train) == len(labels_train) == 10
            assert len(covariance_train) == 10
            assert len(raw_validation) == len(labels_validation) == 4
            self.selected_epoch_count_ = 0
            self.epochs_run_ = 0
            self.initial_model_state_sha256_ = HEX_A
            self.teacher_was_used_ = True
            return self

        def fit_fixed_epochs(self, raw, covariance, labels, *, epochs):
            assert len(raw) == len(covariance) == len(labels) == 14
            assert epochs == 0
            self.epochs_run_ = 0
            self.initial_model_state_sha256_ = HEX_A
            self.model_state_sha256_ = HEX_A
            self.parameter_count_ = 11_354
            self.trainable_parameter_count_ = 11_354
            self.preprocessing_manifest_ = _preprocessing_manifest()
            self.teacher_was_used_ = True
            self.mode_ = "fixed_refit"
            return self

        def predict_proba(self, raw):
            type(self).predict_calls += 1
            assert len(raw) == 6
            return np.full((6, 2), 0.5, dtype=np.float64)

    record, rows, probability = grid.execute_job(
        plan=plan,
        project_root=Path("/unused"),
        cache_root=Path("/unused"),
        job=job,
        gpu_uuid=GPU_UUID,
        classifier_type=FakeClassifier,
    )
    assert len(instances) == 2
    assert FakeClassifier.predict_calls == 1
    assert record["selection"]["selected_epoch_count"] == 0
    assert record["refit"]["epochs_run"] == 0
    assert np.array_equal(rows, test)
    assert probability.shape == (6, 2)


def test_compute_analysis_rejects_bad_audit_before_opening_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def forbidden(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("labels were opened")

    monkeypatch.setattr(grid, "_verify_loaded_cache", forbidden)
    with pytest.raises((analysis.HemiQAnalysisError, grid.HemiQGridError)):
        analysis._compute_after_audit(
            plan=_valid_plan(),
            project_root=Path("/unused"),
            cache_root=Path("/unused"),
            run_root=Path("/unused"),
            audit={"schema": "forged"},
        )
    assert not opened


def test_classification_metrics_are_exact_and_reject_bool_bins() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probability = np.asarray(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
        dtype=np.float64,
    )
    metrics = analysis.classification_metrics(labels, probability)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    with pytest.raises(analysis.HemiQAnalysisError):
        analysis.classification_metrics(labels, probability, ece_bins=True)


def test_audit_and_analysis_reject_bool_numeric_aliases() -> None:
    plan = _valid_plan()
    audit = _dummy_audit(plan)
    audit["roster"][0]["size_bytes"] = True
    audit["roster_sha256"] = grid._sha256_bytes(
        grid._canonical_bytes(audit["roster"])
    )
    with pytest.raises(analysis.HemiQAnalysisError):
        analysis._validate_audit(audit, plan=plan)

    audit = _dummy_audit(plan)
    value = _valid_analysis_value(plan, audit)
    value["metric_contract"]["class_order"] = [False, True]
    with pytest.raises(analysis.HemiQAnalysisError):
        analysis._validate_analysis(value, plan=plan, audit=audit)

    value = _valid_analysis_value(plan, audit)
    value["subject_seed_metrics"][0]["accuracy"] = 0
    with pytest.raises(analysis.HemiQAnalysisError):
        analysis._validate_analysis(value, plan=plan, audit=audit)


def test_analyzer_persists_audit_before_metric_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "run"
    grid.write_or_validate_plan(run_root, plan)
    audit = _dummy_audit(plan)
    monkeypatch.setattr(grid, "audit_grid", lambda **k: copy.deepcopy(audit))
    monkeypatch.setattr(
        analysis,
        "_verify_analysis_execution_identity",
        lambda **k: None,
    )

    @contextlib.contextmanager
    def fake_fence(**kwargs):
        yield object()

    monkeypatch.setattr(grid, "analysis_fence", fake_fence)

    class MarkerSeen(RuntimeError):
        pass

    def stop_after_check(**kwargs):
        stages = list(tmp_path.glob(".analysis.stage-*"))
        assert len(stages) == 1
        marker = stages[0] / "audit.json"
        assert marker.exists()
        assert grid._strict_json_file(marker) == audit
        raise MarkerSeen

    monkeypatch.setattr(analysis, "_compute_after_audit", stop_after_check)
    with pytest.raises(MarkerSeen):
        analysis.analyze_and_publish(
            project_root=tmp_path,
            cache_root=tmp_path,
            run_root=run_root,
            destination=tmp_path / "analysis",
        )


@pytest.mark.parametrize(
    "fault",
    (
        "parent_fsync",
        "publication_verify",
        "final_fence",
        "final_audit",
        "fence_release",
    ),
)
def test_every_post_rename_failure_hides_exact_published_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / f"post-rename-{fault}"
    grid.write_or_validate_plan(run_root, plan)
    audit = _dummy_audit(plan)
    _patch_lightweight_analysis(
        monkeypatch,
        plan=plan,
        audit=audit,
    )
    destination = tmp_path / f"publication-{fault}"
    captured: dict[str, int] = {}
    original_stage = analysis._stage_analysis_payloads

    def capture_stage(**kwargs):
        manifest, st_dev, st_ino = original_stage(**kwargs)
        captured.update(st_dev=st_dev, st_ino=st_ino)
        return manifest, st_dev, st_ino

    monkeypatch.setattr(analysis, "_stage_analysis_payloads", capture_stage)

    class PostRenameFault(RuntimeError):
        pass

    if fault == "parent_fsync":
        original_fsync = os.fsync
        tripped = False

        def fail_after_rename(descriptor):
            nonlocal tripped
            if destination.exists() and not tripped:
                tripped = True
                raise PostRenameFault("parent fsync")
            return original_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", fail_after_rename)
    elif fault == "publication_verify":
        monkeypatch.setattr(
            analysis,
            "_verify_publication",
            lambda *a, **k: (_ for _ in ()).throw(
                PostRenameFault("publication verify")
            ),
        )
    elif fault == "final_fence":
        monkeypatch.setattr(
            grid,
            "assert_analysis_fence",
            lambda *a, **k: (
                (_ for _ in ()).throw(PostRenameFault("final fence"))
                if destination.exists()
                else None
            ),
        )
    elif fault == "final_audit":
        monkeypatch.setattr(
            grid,
            "audit_grid",
            lambda **k: (
                (_ for _ in ()).throw(PostRenameFault("final audit"))
                if destination.exists()
                else copy.deepcopy(audit)
            ),
        )
    elif fault == "fence_release":
        @contextlib.contextmanager
        def failing_release(**kwargs):
            yield object()
            raise PostRenameFault("fence release")

        monkeypatch.setattr(grid, "analysis_fence", failing_release)
    else:
        raise AssertionError(fault)

    with pytest.raises(PostRenameFault):
        analysis.analyze_and_publish(
            project_root=tmp_path,
            cache_root=tmp_path,
            run_root=run_root,
            destination=destination,
        )
    assert not destination.exists()
    hidden = list(
        tmp_path.glob(f".{destination.name}.failed-*")
    )
    assert len(hidden) == 1
    status = hidden[0].stat()
    assert (status.st_dev, status.st_ino) == (
        captured["st_dev"],
        captured["st_ino"],
    )


def test_analysis_move_then_raise_hides_the_moved_stage_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "analysis-move-then-raise"
    grid.write_or_validate_plan(run_root, plan)
    audit = _dummy_audit(plan)
    _patch_lightweight_analysis(
        monkeypatch,
        plan=plan,
        audit=audit,
    )
    destination = tmp_path / "publication-move-then-raise"
    original_rename = grid._rename_noreplace_at
    moved_identity: tuple[int, int] | None = None
    tripped = False

    def move_then_raise(
        source_parent: int,
        source_name: str,
        destination_parent: int,
        destination_name: str,
    ) -> None:
        nonlocal moved_identity, tripped
        original_rename(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
        )
        if destination_name == destination.name and not tripped:
            status = os.stat(
                destination_name,
                dir_fd=destination_parent,
                follow_symlinks=False,
            )
            moved_identity = (int(status.st_dev), int(status.st_ino))
            tripped = True
            raise OSError("injected analysis move-then-raise")

    monkeypatch.setattr(grid, "_rename_noreplace_at", move_then_raise)
    with pytest.raises(OSError, match="move-then-raise"):
        analysis.analyze_and_publish(
            project_root=tmp_path,
            cache_root=tmp_path,
            run_root=run_root,
            destination=destination,
        )
    assert tripped and moved_identity is not None
    assert not destination.exists()
    hidden = list(tmp_path.glob(f".{destination.name}.failed-*"))
    assert len(hidden) == 1
    assert (
        int(hidden[0].stat().st_dev),
        int(hidden[0].stat().st_ino),
    ) == moved_identity


@pytest.mark.parametrize("substitution", ("a_to_b", "a_to_b_to_a"))
def test_analysis_publication_rejects_coherent_directory_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    substitution: str,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / f"analysis-substitution-{substitution}"
    grid.write_or_validate_plan(run_root, plan)
    audit = _dummy_audit(plan)
    _patch_lightweight_analysis(
        monkeypatch,
        plan=plan,
        audit=audit,
    )
    destination = tmp_path / f"publication-{substitution}"
    replacement = tmp_path / f".{destination.name}.coherent-b"
    displaced = tmp_path / f".{destination.name}.displaced-a"
    identities: dict[str, tuple[int, int]] = {}
    original_open_snapshot = analysis._open_publication_snapshot
    original_read = grid._read_descriptor_bytes
    swapped = False

    def open_with_coherent_replacement(path, **kwargs):
        shutil.copytree(path, replacement, copy_function=shutil.copy2)
        snapshot = original_open_snapshot(path, **kwargs)
        identities["a"] = (
            int(os.fstat(snapshot.directory_descriptor).st_dev),
            int(os.fstat(snapshot.directory_descriptor).st_ino),
        )
        replacement_status = replacement.stat()
        identities["b"] = (
            int(replacement_status.st_dev),
            int(replacement_status.st_ino),
        )
        return snapshot

    def substitute_during_member_read(
        descriptor: int,
        *,
        source: str,
    ) -> bytes:
        nonlocal swapped
        if (
            not swapped
            and source == str(destination / "analysis.json")
        ):
            os.rename(destination, displaced)
            os.rename(replacement, destination)
            if substitution == "a_to_b_to_a":
                os.rename(destination, replacement)
                os.rename(displaced, destination)
            swapped = True
        return original_read(descriptor, source=source)

    monkeypatch.setattr(
        analysis,
        "_open_publication_snapshot",
        open_with_coherent_replacement,
    )
    monkeypatch.setattr(
        grid,
        "_read_descriptor_bytes",
        substitute_during_member_read,
    )
    with pytest.raises(analysis.HemiQAnalysisError):
        analysis.analyze_and_publish(
            project_root=tmp_path,
            cache_root=tmp_path,
            run_root=run_root,
            destination=destination,
        )
    assert swapped
    if substitution == "a_to_b":
        assert destination.exists()
        assert (
            int(destination.stat().st_dev),
            int(destination.stat().st_ino),
        ) == identities["b"]
        assert displaced.exists()
        assert (
            int(displaced.stat().st_dev),
            int(displaced.stat().st_ino),
        ) == identities["a"]
    else:
        assert not destination.exists()
        assert replacement.exists()
        assert (
            int(replacement.stat().st_dev),
            int(replacement.stat().st_ino),
        ) == identities["b"]
        hidden = list(
            tmp_path.glob(f".{destination.name}.failed-*")
        )
        assert len(hidden) == 1
        assert (
            int(hidden[0].stat().st_dev),
            int(hidden[0].stat().st_ino),
        ) == identities["a"]
    assert all(
        (
            int(path.stat().st_dev),
            int(path.stat().st_ino),
        )
        != identities["b"]
        for path in tmp_path.glob(f".{destination.name}.failed-*")
    )


@pytest.mark.parametrize(
    "drift_point",
    ("before_labels", "before_rename"),
)
def test_analyzer_source_uv_drift_aborts_at_both_fenced_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    drift_point: str,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / f"drift-{drift_point}"
    grid.write_or_validate_plan(run_root, plan)
    audit = _dummy_audit(plan)
    _patch_lightweight_analysis(
        monkeypatch,
        plan=plan,
        audit=audit,
    )
    calls = 0
    metric_phase = False

    def verify(**kwargs):
        nonlocal calls
        calls += 1
        if drift_point == "before_labels" or calls == 2:
            raise grid.HemiQGridError(f"{drift_point} identity drift")

    def compute(**kwargs):
        nonlocal metric_phase
        metric_phase = True
        return _valid_analysis_value(plan, audit)

    monkeypatch.setattr(
        analysis, "_verify_analysis_execution_identity", verify
    )
    monkeypatch.setattr(analysis, "_compute_after_audit", compute)
    destination = tmp_path / f"drift-output-{drift_point}"
    with pytest.raises(grid.HemiQGridError, match="identity drift"):
        analysis.analyze_and_publish(
            project_root=tmp_path,
            cache_root=tmp_path,
            run_root=run_root,
            destination=destination,
        )
    assert not destination.exists()
    assert metric_phase is (drift_point == "before_rename")
    assert calls == (1 if drift_point == "before_labels" else 2)


def test_analysis_execution_identity_rebinds_source_uv_and_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    calls: list[str] = []
    monkeypatch.setattr(
        grid,
        "verify_source_identity",
        lambda *a, **k: calls.append("source"),
    )
    monkeypatch.setattr(
        grid,
        "verify_uv_identity",
        lambda *a, **k: calls.append("uv"),
    )

    def config(project_root):
        calls.append("config")
        return copy.deepcopy(plan["configuration_identity"])

    monkeypatch.setattr(grid, "_config_identity", config)
    analysis._verify_analysis_execution_identity(
        plan=plan,
        project_root=tmp_path,
    )
    assert calls == ["source", "uv", "config"]


def test_exact_45_job_audit_is_label_free_and_analysis_publication_is_sealed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root = _complete_fake_formal_grid(monkeypatch, tmp_path)
    opened = False

    def forbidden(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("audit opened cache labels")

    monkeypatch.setattr(grid, "_verify_loaded_cache", forbidden)
    with grid.analysis_fence(run_root=run_root, plan=plan) as fence:
        audit = grid.audit_grid(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            fence=fence,
        )
    assert not opened
    assert audit["completed_jobs"] == 45
    assert len(audit["roster"]) == 1 + 45 * 3 + 1

    labels = np.arange(20, dtype=np.int64) % 2
    sessions = np.asarray(
        ["0train"] * 5
        + ["1train"] * 5
        + ["2train"] * 4
        + ["3test"] * 3
        + ["4test"] * 3
    )
    cache = {"y": labels, "sessions": sessions}

    def fake_loaded(*args, **kwargs):
        return (
            cache,
            (TRAIN_ROWS, VALIDATION_ROWS, TEST_ROWS),
        )

    monkeypatch.setattr(grid, "_verify_loaded_cache", fake_loaded)
    monkeypatch.setattr(
        analysis,
        "_verify_analysis_execution_identity",
        lambda **k: None,
    )
    destination = tmp_path / "analysis-publication"
    result = analysis.analyze_and_publish(
        project_root=tmp_path,
        cache_root=tmp_path,
        run_root=run_root,
        destination=destination,
    )
    assert result["n_jobs"] == 45
    assert result["status"] == "published"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        (*analysis.OUTPUT_FILENAMES, "manifest.json")
    )
    published = grid._strict_json_file(destination / "analysis.json")
    assert len(published["subject_seed_metrics"]) == 45
    assert len(published["subject_metrics"]) == 9
    assert published["dataset_summary"]["n_subjects"] == 9


def test_metric_phase_rejects_transient_a_to_b_to_a_result_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root = _complete_fake_formal_grid(monkeypatch, tmp_path)
    with grid.analysis_fence(run_root=run_root, plan=plan) as fence:
        audit = grid.audit_grid(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            fence=fence,
        )
    analysis._validate_audit(audit, plan=plan)
    job = grid.expected_jobs()[0]
    directory = grid._record_directory(run_root, job)
    original_payloads = {
        name: (directory / name).read_bytes()
        for name in grid.RESULT_FILENAMES
    }
    changed_probability = np.tile(
        np.asarray([[0.75, 0.25]], dtype=np.float64),
        (len(TEST_ROWS), 1),
    )
    changed_record = _record(
        plan, job, TEST_ROWS, changed_probability
    )
    changed_prediction = grid._prediction_npz_bytes(
        TEST_ROWS, changed_probability
    )
    changed_completion = grid._strict_json_bytes(
        original_payloads["completion.json"],
        source="original completion",
    )
    changed_artifacts = {
        "predictions.npz": {
            "sha256": grid._sha256_bytes(changed_prediction),
            "size_bytes": len(changed_prediction),
        },
        "record.json": {
            "sha256": grid._sha256_bytes(
                grid._canonical_bytes(changed_record)
            ),
            "size_bytes": len(grid._canonical_bytes(changed_record)),
        },
    }
    changed_completion["artifacts"] = changed_artifacts
    changed_completion["seal_sha256"] = grid._sha256_bytes(
        grid._canonical_bytes(changed_artifacts)
    )
    changed_payloads = {
        "completion.json": grid._canonical_bytes(changed_completion),
        "predictions.npz": changed_prediction,
        "record.json": grid._canonical_bytes(changed_record),
    }

    labels = np.arange(20, dtype=np.int64) % 2
    sessions = np.asarray(
        ["0train"] * 5
        + ["1train"] * 5
        + ["2train"] * 4
        + ["3test"] * 3
        + ["4test"] * 3
    )
    monkeypatch.setattr(
        grid,
        "_verify_loaded_cache",
        lambda *a, **k: (
            {"y": labels, "sessions": sessions},
            (TRAIN_ROWS, VALIDATION_ROWS, TEST_ROWS),
        ),
    )
    original_validate = grid.validate_completion
    swapped = False

    def transient_validate(**kwargs):
        nonlocal swapped
        if kwargs["job"] == job and not swapped:
            _rewrite_result_snapshot(directory, changed_payloads)
            try:
                validated = original_validate(**kwargs)
            finally:
                _rewrite_result_snapshot(directory, original_payloads)
            swapped = True
            return validated
        return original_validate(**kwargs)

    monkeypatch.setattr(grid, "validate_completion", transient_validate)
    with pytest.raises(
        analysis.HemiQAnalysisError,
        match="metric snapshot differs",
    ):
        analysis._compute_after_audit(
            plan=plan,
            project_root=tmp_path,
            cache_root=tmp_path,
            run_root=run_root,
            audit=audit,
        )
    assert swapped
    assert {
        name: (directory / name).read_bytes()
        for name in grid.RESULT_FILENAMES
    } == original_payloads


def test_atomic_plan_parent_fsync_failure_hides_canonical_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "atomic-plan-fsync"
    path = run_root / "plan.json"
    original_fsync = os.fsync
    tripped = False

    def fail_first_parent_fsync(descriptor: int) -> None:
        nonlocal tripped
        if path.exists() and not tripped:
            tripped = True
            raise OSError("injected plan parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_parent_fsync)
    with pytest.raises(OSError, match="plan parent fsync"):
        grid.write_or_validate_plan(run_root, plan)
    assert tripped
    assert not path.exists()
    assert not any(
        name.startswith(".plan.json.stage-")
        for name in os.listdir(run_root)
    )

    monkeypatch.setattr(os, "fsync", original_fsync)
    assert grid.write_or_validate_plan(run_root, plan) == plan
    assert stat.S_IMODE(path.stat().st_mode) == 0o400


def test_atomic_plan_move_then_raise_hides_exact_canonical_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "atomic-plan-move-raise"
    path = run_root / "plan.json"
    original_rename = grid._rename_noreplace_at
    tripped = False

    def move_then_raise(
        source_parent: int,
        source_name: str,
        destination_parent: int,
        destination_name: str,
    ) -> None:
        nonlocal tripped
        original_rename(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
        )
        if destination_name == "plan.json" and not tripped:
            tripped = True
            raise OSError("injected plan move-then-raise")

    monkeypatch.setattr(grid, "_rename_noreplace_at", move_then_raise)
    with pytest.raises(OSError, match="plan move-then-raise"):
        grid.write_or_validate_plan(run_root, plan)
    assert tripped
    assert not path.exists()
    assert not any(
        name.startswith(".plan.json.stage-")
        for name in os.listdir(run_root)
    )
    monkeypatch.setattr(grid, "_rename_noreplace_at", original_rename)
    assert grid.write_or_validate_plan(run_root, plan) == plan


def test_atomic_preflight_and_failure_parent_fsync_failures_are_retryable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "atomic-receipts"
    grid.write_or_validate_plan(run_root, plan)
    preflight = _valid_preflight_value(plan, run_root)
    monkeypatch.setattr(grid, "_assert_lease_bound", lambda *a, **k: None)
    monkeypatch.setattr(grid, "verify_static_identity", lambda *a, **k: None)
    monkeypatch.setattr(
        grid,
        "_require_single_gpu_runtime",
        lambda **k: copy.deepcopy(preflight["runtime"]),
    )
    monkeypatch.setattr(
        grid,
        "_synthetic_cuda_preflight",
        lambda **k: copy.deepcopy(preflight["checks"]),
    )
    monkeypatch.setattr(
        grid,
        "_hard_disk_guard",
        lambda *a, **k: copy.deepcopy(preflight["disk_guard"]),
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "guard_gpu_lease",
        _fake_guard,
    )
    monkeypatch.setattr(
        grid.project_gpu_leases,
        "validate_gpu_lease_receipt",
        lambda *a, **k: None,
    )
    original_fsync = os.fsync
    preflight_path = grid._preflight_path(run_root, GPU_UUID)
    tripped = False

    def fail_preflight_parent_fsync(descriptor: int) -> None:
        nonlocal tripped
        if preflight_path.exists() and not tripped:
            tripped = True
            raise OSError("injected preflight parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_preflight_parent_fsync)
    with pytest.raises(OSError, match="preflight parent fsync"):
        grid.run_preflight(
            plan=plan,
            project_root=tmp_path,
            cache_root=tmp_path,
            run_root=run_root,
            gpu_uuid=GPU_UUID,
            device="cuda:0",
            lease=object(),
        )
    assert tripped
    assert not preflight_path.exists()
    assert not any(
        name.startswith(f".{GPU_UUID}.json.stage-")
        for name in os.listdir(run_root / "preflight")
    )
    monkeypatch.setattr(os, "fsync", original_fsync)
    assert grid.run_preflight(
        plan=plan,
        project_root=tmp_path,
        cache_root=tmp_path,
        run_root=run_root,
        gpu_uuid=GPU_UUID,
        device="cuda:0",
        lease=object(),
    )["status"] == "pass"
    assert stat.S_IMODE(preflight_path.stat().st_mode) == 0o400

    tripped = False

    def fail_failure_parent_fsync(descriptor: int) -> None:
        nonlocal tripped
        visible = [
            name
            for name in os.listdir(run_root / "failures")
            if name.endswith(".json") and not name.startswith(".")
        ]
        if visible and not tripped:
            tripped = True
            raise OSError("injected failure parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_failure_parent_fsync)
    with pytest.raises(OSError, match="failure parent fsync"):
        grid._record_failure(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=grid.expected_jobs()[0],
            gpu_uuid=GPU_UUID,
            lease=object(),
            error=RuntimeError("synthetic"),
        )
    assert tripped
    assert not [
        name
        for name in os.listdir(run_root / "failures")
        if name.endswith(".json") and not name.startswith(".")
    ]
    monkeypatch.setattr(os, "fsync", original_fsync)
    receipt = grid._record_failure(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=grid.expected_jobs()[0],
        gpu_uuid=GPU_UUID,
        lease=object(),
        error=RuntimeError("synthetic retry"),
    )
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o400


def test_live_claim_binds_and_protects_matching_result_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    stage = (
        run_root
        / "records"
        / f"{job.job_id}.{claim.nonce}.stage"
    )
    stage.mkdir(mode=0o700)
    stage_inode = stage.stat().st_ino

    with pytest.raises(grid.ClaimUnavailable, match="live claim"):
        grid.acquire_claim(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=lease,
        )
    assert stage.exists()
    assert stage.stat().st_ino == stage_inode
    assert not list((run_root / "quarantine").glob("abandoned_stage/*"))

    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )
    recovered = grid.recover_job_partials(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
    )
    assert len(recovered) == 1
    assert recovered[0].stat().st_ino == stage_inode
    assert not stage.exists()


def test_analysis_fence_rejects_replaced_run_root_namespace(
    tmp_path: Path,
) -> None:
    plan = _valid_plan()
    run_root = tmp_path / "fenced-root"
    grid.write_or_validate_plan(run_root, plan)
    original_root = tmp_path / "fenced-root-original"

    with pytest.raises(grid.HemiQGridError, match="run-root namespace"):
        with grid.analysis_fence(run_root=run_root, plan=plan) as fence:
            run_root.rename(original_root)
            grid.write_or_validate_plan(run_root, plan)
            replacement_lock = (
                run_root / ".authority" / "coordination.lock"
            ).stat()
            assert (
                replacement_lock.st_dev,
                replacement_lock.st_ino,
            ) != (fence.st_dev, fence.st_ino)
            grid.assert_analysis_fence(fence, plan=plan)
    assert not list(
        (original_root / ".authority").glob("analysis-*.json")
    )


def test_quarantine_hides_same_parent_before_permission_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "quarantine-order"
    grid._initialize_run_root(run_root)
    source = run_root / "records" / "canonical-result"
    source.mkdir(mode=0o700)
    (source / "leaf").write_bytes(b"forensic")
    os.chmod(source / "leaf", 0o400)
    os.chmod(source, 0o555)
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    original_rename = grid._rename_noreplace_at
    original_fchmod = os.fchmod
    renames: list[tuple[int, str, int, str]] = []
    permission_recovery_seen = False

    def capture_rename(
        source_parent: int,
        source_name: str,
        destination_parent: int,
        destination_name: str,
    ) -> None:
        renames.append(
            (
                source_parent,
                source_name,
                destination_parent,
                destination_name,
            )
        )
        original_rename(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
        )

    def inspect_fchmod(descriptor: int, mode: int) -> None:
        nonlocal permission_recovery_seen
        if mode == 0o700:
            permission_recovery_seen = True
            assert not source.exists()
            assert any(
                path.name.startswith(
                    f".{source.name}.quarantine-"
                )
                for path in source.parent.iterdir()
            )
        original_fchmod(descriptor, mode)

    monkeypatch.setattr(grid, "_rename_noreplace_at", capture_rename)
    monkeypatch.setattr(os, "fchmod", inspect_fchmod)
    quarantined = grid._quarantine_leaf(
        run_root,
        source,
        category="ordering_test",
        expected_identity=source_identity,
    )
    assert permission_recovery_seen
    assert not source.exists()
    assert quarantined.exists()
    assert (quarantined.stat().st_dev, quarantined.stat().st_ino) == (
        source_identity
    )
    assert stat.S_IMODE(quarantined.stat().st_mode) == 0o555
    assert renames[0][1] == source.name
    assert renames[0][0] == renames[0][2]
    assert renames[0][3].startswith(
        f".{source.name}.quarantine-"
    )


def test_quarantine_destination_failure_still_hides_canonical_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "quarantine-destination-fault"
    grid._initialize_run_root(run_root)
    source = run_root / "records" / "canonical"
    source.write_bytes(b"authority")
    os.chmod(source, 0o400)
    identity = (source.stat().st_dev, source.stat().st_ino)
    monkeypatch.setattr(
        grid,
        "_quarantine_path",
        lambda *a, **k: (_ for _ in ()).throw(
            OSError("injected quarantine destination failure")
        ),
    )
    with pytest.raises(OSError, match="destination failure"):
        grid._quarantine_leaf(
            run_root,
            source,
            category="destination_fault",
            expected_identity=identity,
        )
    assert not source.exists()
    hidden = list(
        (run_root / "records").glob(
            f".{source.name}.quarantine-*"
        )
    )
    assert len(hidden) == 1
    assert (hidden[0].stat().st_dev, hidden[0].stat().st_ino) == identity


@pytest.mark.parametrize(
    "fault",
    ("parent_fsync", "published_inspection", "move_then_raise"),
)
def test_every_result_rename_failure_hides_exact_canonical_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    rows = TEST_ROWS.copy()
    probability = np.full((len(rows), 2), 0.5, dtype=np.float64)
    destination = grid._record_directory(run_root, job)
    original_rename = grid._rename_noreplace_at
    original_stat = os.stat
    original_fsync = os.fsync
    after_result_rename = False
    tripped = False
    published_identity: tuple[int, int] | None = None

    def inspect_rename(
        source_parent: int,
        source_name: str,
        destination_parent: int,
        destination_name: str,
    ) -> None:
        nonlocal after_result_rename, published_identity
        if destination_name == job.job_id:
            source_status = original_stat(
                source_name,
                dir_fd=source_parent,
                follow_symlinks=False,
            )
            published_identity = (
                int(source_status.st_dev),
                int(source_status.st_ino),
            )
            original_rename(
                source_parent,
                source_name,
                destination_parent,
                destination_name,
            )
            after_result_rename = True
            if fault == "move_then_raise":
                raise RuntimeError("injected move-then-raise")
            return
        original_rename(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
        )

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal tripped
        if (
            fault == "parent_fsync"
            and after_result_rename
            and not tripped
        ):
            tripped = True
            raise OSError("injected result parent fsync")
        original_fsync(descriptor)

    def fail_published_inspection(
        path,
        *args,
        **kwargs,
    ):
        nonlocal tripped
        if (
            fault == "published_inspection"
            and after_result_rename
            and path == job.job_id
            and kwargs.get("dir_fd") is not None
            and not tripped
        ):
            tripped = True
            raise OSError("injected published inspection")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(grid, "_rename_noreplace_at", inspect_rename)
    monkeypatch.setattr(os, "fsync", fail_parent_fsync)
    monkeypatch.setattr(os, "stat", fail_published_inspection)
    with pytest.raises(
        (OSError, RuntimeError),
        match="injected",
    ):
        grid.commit_job_output(
            plan=plan,
            project_root=tmp_path,
            run_root=run_root,
            job=job,
            gpu_uuid=GPU_UUID,
            lease=lease,
            claim=claim,
            record=_record(plan, job, rows, probability),
            test_rows=rows,
            probabilities=probability,
        )
    assert after_result_rename
    assert published_identity is not None
    assert not destination.exists()
    quarantined = list(
        (run_root / "quarantine/post_commit_authority_loss").iterdir()
    )
    assert len(quarantined) == 1
    assert (
        quarantined[0].stat().st_dev,
        quarantined[0].stat().st_ino,
    ) == published_identity
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )


def test_worker_rebinds_source_uv_and_config_before_and_after_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_rebind = grid._verify_worker_publication_identity
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        grid,
        "verify_source_identity",
        lambda *a, **k: calls.append("source"),
    )
    monkeypatch.setattr(
        grid,
        "verify_uv_identity",
        lambda *a, **k: calls.append("uv"),
    )

    def config(project_root: Path) -> dict:
        calls.append("config")
        return copy.deepcopy(plan["configuration_identity"])

    monkeypatch.setattr(grid, "_config_identity", config)
    monkeypatch.setattr(
        grid,
        "_verify_worker_publication_identity",
        real_rebind,
    )
    rows = TEST_ROWS.copy()
    probability = np.full((len(rows), 2), 0.5, dtype=np.float64)
    destination = grid.commit_job_output(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
        claim=claim,
        record=_record(plan, job, rows, probability),
        test_rows=rows,
        probabilities=probability,
    )
    assert destination.exists()
    assert calls == [
        "source",
        "uv",
        "config",
        "source",
        "uv",
        "config",
    ]
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )


def test_completion_snapshot_opens_full_package_and_rejects_live_b_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    rows = TEST_ROWS.copy()
    probability = np.full((len(rows), 2), 0.5, dtype=np.float64)
    destination = grid.commit_job_output(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
        claim=claim,
        record=_record(plan, job, rows, probability),
        test_rows=rows,
        probabilities=probability,
    )
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )
    records = destination.parent
    replacement = records / f".{job.job_id}.package-b"
    displaced = records / f".{job.job_id}.package-a-held"
    shutil.copytree(destination, replacement)
    for leaf in replacement.iterdir():
        os.chmod(leaf, 0o400)
    os.chmod(replacement, 0o555)

    original_open = os.open
    original_read = grid._read_descriptor_bytes
    opened_members: set[str] = set()
    swapped = False

    def capture_open(path, flags, *args, **kwargs):
        if path in grid.RESULT_FILENAMES and kwargs.get("dir_fd") is not None:
            opened_members.add(str(path))
        return original_open(path, flags, *args, **kwargs)

    def swap_package_before_first_read(
        descriptor: int,
        *,
        source: str,
    ) -> bytes:
        nonlocal swapped
        if not swapped and source.endswith("completion.json"):
            assert opened_members == set(grid.RESULT_FILENAMES)
            os.rename(destination, displaced)
            os.rename(replacement, destination)
            swapped = True
        return original_read(descriptor, source=source)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(
        grid,
        "_read_descriptor_bytes",
        swap_package_before_first_read,
    )
    try:
        with pytest.raises(
            grid.HemiQGridError,
            match="package|detached|changed",
        ):
            grid.validate_completion(
                plan=plan,
                project_root=tmp_path,
                run_root=run_root,
                job=job,
            )
        assert swapped
    finally:
        if destination.exists() and displaced.exists():
            os.rename(destination, replacement)
            os.rename(displaced, destination)


def test_worker_completion_snapshot_tolerates_unrelated_parallel_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, run_root, job, lease, claim = _prepare_claim(monkeypatch, tmp_path)
    rows = TEST_ROWS.copy()
    probability = np.full((len(rows), 2), 0.5, dtype=np.float64)
    destination = grid.commit_job_output(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
        gpu_uuid=GPU_UUID,
        lease=lease,
        claim=claim,
        record=_record(plan, job, rows, probability),
        test_rows=rows,
        probabilities=probability,
    )
    _release_claim(
        claim,
        plan=plan,
        run_root=run_root,
        lease=lease,
        project_root=tmp_path,
    )
    unrelated = destination.parent / "other-worker-result"
    original_read = grid._read_descriptor_bytes
    injected = False

    def publish_unrelated_record(
        descriptor: int,
        *,
        source: str,
    ) -> bytes:
        nonlocal injected
        if not injected and source.endswith("completion.json"):
            unrelated.mkdir(mode=0o555)
            injected = True
        return original_read(descriptor, source=source)

    monkeypatch.setattr(
        grid,
        "_read_descriptor_bytes",
        publish_unrelated_record,
    )
    validated = grid.validate_completion(
        plan=plan,
        project_root=tmp_path,
        run_root=run_root,
        job=job,
    )
    assert injected
    assert validated["completion"]["job"] == job.as_dict()


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value["subject_seed_metrics"][0].__setitem__(
            "accuracy", 2.0
        ),
        lambda value: value["subject_metrics"][0].__setitem__(
            "balanced_accuracy", -7.0
        ),
        lambda value: value["dataset_summary"].__setitem__(
            "macro_f1", 99.0
        ),
        lambda value: value["subject_seed_metrics"][0].__setitem__(
            "n_trials", 999
        ),
        lambda value: value["subject_seed_metrics"][0].__setitem__(
            "accuracy", 0.5
        ),
        lambda value: value["dataset_summary"].__setitem__(
            "accuracy", 0.25
        ),
        lambda value: value["dataset_summary"].__setitem__(
            "aggregation", "unweighted trial mean"
        ),
    ),
)
def test_persisted_analysis_rejects_ranges_trials_and_aggregate_drift(
    mutator,
) -> None:
    plan = _valid_plan()
    audit = _dummy_audit(plan)
    value = _valid_analysis_value(plan, audit)
    mutator(value)
    with pytest.raises(analysis.HemiQAnalysisError):
        analysis._validate_analysis(value, plan=plan, audit=audit)
