"""Prespecified v3 validation screen for three CHSD-conditioned variants.

This scratch runner adds exactly 51 validation jobs to the completed reference
screen: three fixed conditioning strengths crossed with the same 17 opened
development subject splits and seed 7.  It never reruns a reference model.

The immutable plan is bound to the corrected reference run by its exact plan
SHA-256.  Each of three workers owns one model variant and exactly 17 jobs.
Workers must be launched from a UV-managed virtual environment with one full,
plan-bound NVIDIA GPU UUID in ``CUDA_VISIBLE_DEVICES``.  Before every new job,
the runner rejects foreign compute PIDs and a filesystem with less than 50 GiB
free.  Only training and validation rows are materialized, and records contain
aggregate validation metrics rather than outcomes or predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# NumPy/BLAS pools can be initialized during imports.  Cap them before NumPy,
# PyTorch, or the existing screen helpers are imported.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "4"

import numpy as np

from .config import (
    CHANNEL_SCALING,
    DEFAULT_MONTAGE_PROFILE,
    PREPROCESSING,
    dataset_spec,
    preprocessing_for_dataset,
)
from .development_screen import (
    CURATED_SPLITS,
    EVALUATION_SCOPE,
    GPU_UUID_RE,
    HEX_64_RE,
    METRIC_NAMES,
    MINIMUM_FREE_GIB,
    REQUIRED_CUBLAS_WORKSPACE_CONFIG,
    RECORD_SCHEMA as REFERENCE_RECORD_SCHEMA,
    PLAN_SCHEMA as REFERENCE_PLAN_SCHEMA,
    ScreenJob,
    _array_sha256,
    _canonical_cuda_uuid,
    _file_sha256,
    _json_safe,
    _model_identity,
    _payload_sha256,
    _read_cache_plan_identity,
    _rows_sha256,
    _state_sha256,
    _strict_json_load,
    _subject_key,
    _split_key,
    _worker_preflight,
    assert_validation_scope,
    cooperative_gpu_guard,
    development_rows,
    disk_guard,
    environment_identity as reference_environment_identity,
    load_plan as load_reference_plan,
)


PLAN_SCHEMA = "ieee-mi-chsd-conditioned-development-screen-plan-v3"
RECORD_SCHEMA = "ieee-mi-chsd-conditioned-validation-record-v3"
STATUS_SCHEMA = "ieee-mi-chsd-conditioned-screen-status-v3"
SEED = 7
WORKER_COUNT = 3
CPU_THREADS = 4
EXPECTED_JOB_COUNT = 51
EXPECTED_JOBS_PER_WORKER = 17
EXPECTED_REFERENCE_PLAN_SHA256 = (
    "57298f7c18b8841e741d72194cb601faf3ac71cb81b1209b1c3964ddf43c0fc6"
)

CONDITIONED_MODELS: tuple[str, ...] = (
    "chsdnet_conditioned_005",
    "chsdnet_conditioned_010",
    "chsdnet_conditioned_020",
)
REFERENCE_MODELS: tuple[str, ...] = (
    "cardinal_fbc_micro_extended",
    "tcformer",
    "fbcnet",
)
MODEL_FACTORY = "ieee_mi.chsd_conditioned:make_chsd_conditioned_model"
MODEL_MAXIMUM_MODULATION: Mapping[str, float] = {
    "chsdnet_conditioned_005": 0.05,
    "chsdnet_conditioned_010": 0.10,
    "chsdnet_conditioned_020": 0.20,
}

# This must remain byte-for-byte equivalent to the established screen recipe.
TRAIN_CONFIG: Mapping[str, Any] = {
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 8e-4,
    "weight_decay": 5e-4,
    "patience": 35,
    "min_delta": 1e-4,
    "label_smoothing": 0.05,
    "segment_probability": 0.5,
    "segment_count": 8,
    "time_shift_samples": 8,
    "noise_std": 0.01,
    "reflection_weight": 0.0,
    "reflection_probability": 0.0,
    "gradient_clip": 5.0,
    "seed": SEED,
    "device": "cuda:0",
}

# Complete in-repository runtime dependency closure for these three variants.
# Keeping this explicit makes an accidental dependency addition fail review
# rather than silently escaping the immutable source fingerprint.
SOURCE_FILES: tuple[str, ...] = (
    "ieee_mi/__init__.py",
    "ieee_mi/conditioned_screen.py",
    "ieee_mi/development_screen.py",
    "ieee_mi/chsd_conditioned.py",
    "ieee_mi/chsd.py",
    "ieee_mi/baselines.py",
    "ieee_mi/models.py",
    "ieee_mi/training.py",
    "ieee_mi/data.py",
    "ieee_mi/config.py",
)

FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "labels",
        "samples",
        "predictions",
        "probabilities",
        "logits",
        "history",
        "checkpoint",
        "checkpoints",
    }
)
FORBIDDEN_SCOPE_TOKENS = frozenset({"test", "heldout"})


class ConditionedScreenError(RuntimeError):
    """Raised when the v3 screen contract is violated."""


def conditioned_jobs() -> tuple[ScreenJob, ...]:
    """Return the fixed variant-major 51-job order."""

    jobs = tuple(
        ScreenJob(dataset=dataset, model=model, subject=subject)
        for model in CONDITIONED_MODELS
        for dataset, subjects in CURATED_SPLITS
        for subject in subjects
    )
    if len(jobs) != EXPECTED_JOB_COUNT:
        raise AssertionError("conditioned screen cardinality changed")
    return jobs


def jobs_for_worker(
    jobs: Sequence[ScreenJob],
    worker_index: int,
    *,
    worker_count: int = WORKER_COUNT,
) -> tuple[ScreenJob, ...]:
    """Return the one-variant static partition for ``worker_index``."""

    if worker_count != WORKER_COUNT:
        raise ValueError(f"worker_count is fixed at {WORKER_COUNT}")
    if not 0 <= worker_index < WORKER_COUNT:
        raise ValueError(f"worker_index must lie in [0, {WORKER_COUNT - 1}]")
    expected_model = CONDITIONED_MODELS[worker_index]
    partition = tuple(job for job in jobs if job.model == expected_model)
    if len(partition) != EXPECTED_JOBS_PER_WORKER:
        raise ValueError("worker partition does not contain exactly 17 jobs")
    return partition


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(values))).hexdigest()


def _record_path(run_root: Path, job: ScreenJob) -> Path:
    return (
        run_root
        / "records"
        / job.dataset
        / job.model
        / f"subject_{job.subject:03d}"
        / f"fold_{job.fold:02d}"
        / f"seed_{job.seed:03d}.json"
    )


def source_identity(project_root: Path | None = None) -> dict[str, str]:
    """Hash the complete, explicit in-repository runtime dependency closure."""

    root = Path(__file__).resolve().parents[1] if project_root is None else project_root
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = _file_sha256(path)
    return result


def conditioned_environment_identity() -> dict[str, Any]:
    """Fingerprint UV, its virtualenv, packages, CUDA, and the NVIDIA driver."""

    # The existing preflight provides the strict UV marker check.  Invoke it
    # with no CUDA side effects by reproducing only its virtualenv portion.
    if sys.prefix == sys.base_prefix:
        raise ConditionedScreenError("a UV-managed virtual environment is required")
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    try:
        configuration_text = configuration.read_text(encoding="utf-8")
    except OSError as error:
        raise ConditionedScreenError(
            f"cannot read UV environment marker {configuration}: {error}"
        ) from error
    if not any(
        line.strip().lower().startswith("uv =")
        for line in configuration_text.splitlines()
    ):
        raise ConditionedScreenError(
            f"{sys.prefix} is not marked as a UV-managed virtual environment"
        )
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise ConditionedScreenError("UV is not available on PATH")
    try:
        uv_result = subprocess.run(
            [uv_executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.SubprocessError as error:
        raise ConditionedScreenError(f"cannot fingerprint UV: {error}") from error
    identity = dict(reference_environment_identity())
    identity.update(
        {
            "uv_executable": str(Path(uv_executable).resolve()),
            "uv_version": uv_result.stdout.strip(),
            "virtualenv_prefix": str(Path(sys.prefix).resolve()),
            "pyvenv_cfg_sha256": hashlib.sha256(
                configuration_text.encode("utf-8")
            ).hexdigest(),
        }
    )
    return identity


def _expected_subject_keys() -> set[str]:
    return {
        _subject_key(dataset, subject)
        for dataset, subjects in CURATED_SPLITS
        for subject in subjects
    }


def _expected_split_keys() -> set[str]:
    return {
        _split_key(dataset, subject)
        for dataset, subjects in CURATED_SPLITS
        for subject in subjects
    }


def _reference_job_ids() -> tuple[str, ...]:
    return tuple(
        ScreenJob(dataset=dataset, model=model, subject=subject).job_id
        for model in REFERENCE_MODELS
        for dataset, subjects in CURATED_SPLITS
        for subject in subjects
    )


def _validate_gpu_uuids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(result) != WORKER_COUNT:
        raise ValueError("exactly three worker GPU UUIDs are required")
    if len(set(result)) != WORKER_COUNT:
        raise ValueError("worker GPU UUIDs must be distinct")
    if any(not GPU_UUID_RE.fullmatch(value) for value in result):
        raise ValueError("every worker GPU UUID must be one full NVIDIA UUID")
    return result


def _validate_identity_maps(
    cache_identity: Mapping[str, Mapping[str, Any]],
    split_identity: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(cache_identity) != _expected_subject_keys():
        raise ValueError("cache identities differ from the 17 fixed subjects")
    if set(split_identity) != _expected_split_keys():
        raise ValueError("split identities differ from the 17 fixed splits")
    for key, value in cache_identity.items():
        if set(value) != {
            "relative_path",
            "file_size_bytes",
            "file_sha256",
            "array_sha256",
        }:
            raise ValueError(f"cache identity {key} has unexpected fields")
        if int(value["file_size_bytes"]) <= 0:
            raise ValueError(f"cache identity {key} has invalid size")
        for name in ("file_sha256", "array_sha256"):
            digest = value[name]
            if not isinstance(digest, str) or not HEX_64_RE.fullmatch(digest):
                raise ValueError(f"cache identity {key}/{name} is invalid")
    for key, value in split_identity.items():
        if set(value) != {"train", "validation"}:
            raise ValueError(f"split identity {key} exceeds validation scope")
        for partition in ("train", "validation"):
            item = value[partition]
            if (
                not isinstance(item, Mapping)
                or set(item) != {"count", "rows_sha256"}
                or int(item["count"]) <= 0
                or not isinstance(item["rows_sha256"], str)
                or not HEX_64_RE.fullmatch(item["rows_sha256"])
            ):
                raise ValueError(f"split identity {key}/{partition} is invalid")


def _validate_environment_identity(identity: Mapping[str, Any]) -> None:
    required = {
        "python",
        "executable",
        "platform",
        "packages",
        "torch_cuda_version",
        "torch_cudnn_version",
        "nvidia_driver_versions",
        "required_cublas_workspace_config",
        "cpu_threads_per_worker",
        "minimum_free_gib",
        "uv_executable",
        "uv_version",
        "virtualenv_prefix",
        "pyvenv_cfg_sha256",
    }
    if not required.issubset(identity):
        raise ValueError("environment identity lacks UV/runtime/package fields")
    for name in (
        "python",
        "executable",
        "platform",
        "torch_cuda_version",
        "uv_executable",
        "uv_version",
        "virtualenv_prefix",
    ):
        if not isinstance(identity[name], str) or not identity[name]:
            raise ValueError(f"environment identity {name} is invalid")
    if not str(identity["uv_version"]).lower().startswith("uv "):
        raise ValueError("environment identity does not report a UV version")
    if (
        not isinstance(identity["pyvenv_cfg_sha256"], str)
        or not HEX_64_RE.fullmatch(identity["pyvenv_cfg_sha256"])
    ):
        raise ValueError("environment pyvenv.cfg identity is invalid")
    packages = identity["packages"]
    if not isinstance(packages, Mapping) or not packages:
        raise ValueError("environment package fingerprint is empty")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in packages.items()
    ):
        raise ValueError("environment package fingerprint is invalid")
    drivers = identity["nvidia_driver_versions"]
    if (
        not isinstance(drivers, list)
        or not drivers
        or any(not isinstance(value, str) or not value for value in drivers)
    ):
        raise ValueError("environment NVIDIA driver fingerprint is invalid")
    if (
        identity["required_cublas_workspace_config"]
        != REQUIRED_CUBLAS_WORKSPACE_CONFIG
        or int(identity["cpu_threads_per_worker"]) != CPU_THREADS
        or float(identity["minimum_free_gib"]) != MINIMUM_FREE_GIB
    ):
        raise ValueError("environment safety/runtime constants changed")


def _validate_reference_binding(binding: Mapping[str, Any]) -> None:
    expected_keys = {
        "run_root",
        "plan_schema",
        "record_schema",
        "plan_sha256",
        "evaluation_scope",
        "models",
        "selected_record_count",
        "selected_job_ids_sha256",
    }
    if set(binding) != expected_keys:
        raise ValueError("reference binding fields differ from the fixed schema")
    if binding["plan_schema"] != REFERENCE_PLAN_SCHEMA:
        raise ValueError("unsupported reference plan schema")
    if binding["record_schema"] != REFERENCE_RECORD_SCHEMA:
        raise ValueError("unsupported reference record schema")
    if binding["plan_sha256"] != EXPECTED_REFERENCE_PLAN_SHA256:
        raise ValueError("reference plan SHA-256 differs from the prespecified run")
    if binding["evaluation_scope"] != EVALUATION_SCOPE:
        raise ValueError("reference evidence scope differs")
    if tuple(binding["models"]) != REFERENCE_MODELS:
        raise ValueError("reference model list differs")
    if int(binding["selected_record_count"]) != len(_reference_job_ids()):
        raise ValueError("reference record count differs")
    if binding["selected_job_ids_sha256"] != _sequence_sha256(
        _reference_job_ids()
    ):
        raise ValueError("reference job identity digest differs")
    if not isinstance(binding["run_root"], str) or not binding["run_root"]:
        raise ValueError("reference run root is invalid")


def assemble_manifest(
    *,
    cache_root: str,
    cache_identity: Mapping[str, Mapping[str, Any]],
    split_identity: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    environment_identity: Mapping[str, Any],
    worker_gpu_uuids: Sequence[str],
    reference_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the exact immutable v3 screen manifest."""

    _validate_identity_maps(cache_identity, split_identity)
    gpu_uuids = _validate_gpu_uuids(worker_gpu_uuids)
    _validate_reference_binding(reference_binding)
    if set(source_hashes) != set(SOURCE_FILES):
        raise ValueError("source identity does not cover the fixed dependency closure")
    if any(
        not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
        for value in source_hashes.values()
    ):
        raise ValueError("source identity contains an invalid SHA-256")
    if not isinstance(environment_identity, Mapping):
        raise ValueError("environment identity is not an object")
    _validate_environment_identity(environment_identity)

    jobs = conditioned_jobs()
    worker_by_model = {
        model: worker_index
        for worker_index, model in enumerate(CONDITIONED_MODELS)
    }
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "purpose": "prespecified_conditioned_development_validation_screen",
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "seed": SEED,
        "worker_count": WORKER_COUNT,
        "jobs_per_worker": EXPECTED_JOBS_PER_WORKER,
        "cache_root": str(cache_root),
        "curated_splits": [
            {"dataset": dataset, "subjects": list(subjects), "fold": 0}
            for dataset, subjects in CURATED_SPLITS
        ],
        "models": [
            {
                "name": model,
                "factory": MODEL_FACTORY,
                "maximum_modulation": MODEL_MAXIMUM_MODULATION[model],
                "expected_source_statistics_hook": False,
            }
            for model in CONDITIONED_MODELS
        ],
        "train_config": dict(TRAIN_CONFIG),
        "channel_scaling": dict(CHANNEL_SCALING),
        "cache_identity": {
            key: dict(value) for key, value in sorted(cache_identity.items())
        },
        "split_identity": {
            key: dict(value) for key, value in sorted(split_identity.items())
        },
        "source_identity": dict(sorted(source_hashes.items())),
        "environment_identity": dict(environment_identity),
        "reference_binding": dict(reference_binding),
        "worker_bindings": [
            {
                "worker_index": index,
                "model": CONDITIONED_MODELS[index],
                "gpu_uuid": gpu_uuids[index],
                "job_count": EXPECTED_JOBS_PER_WORKER,
            }
            for index in range(WORKER_COUNT)
        ],
        "jobs": [
            {
                **job.identity(),
                "job_id": job.job_id,
                "worker_index": worker_by_model[job.model],
            }
            for job in jobs
        ],
        "n_jobs": len(jobs),
        "artifact_contract": {
            "metrics": "validation_only",
            "raw_outcomes_persisted": False,
            "raw_trials_persisted": False,
            "model_state_artifacts_persisted": False,
            "epoch_traces_persisted": False,
            "atomic_unit": "one_json_per_job",
        },
    }
    assert_validation_scope(payload)
    payload["plan_sha256"] = _payload_sha256(payload, "plan_sha256")
    validate_manifest_payload(payload)
    return payload


def _reference_binding(
    reference_root: Path,
    reference_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_root": str(reference_root.resolve()),
        "plan_schema": reference_plan["schema"],
        "record_schema": REFERENCE_RECORD_SCHEMA,
        "plan_sha256": reference_plan["plan_sha256"],
        "evaluation_scope": reference_plan["evaluation_scope"],
        "models": list(REFERENCE_MODELS),
        "selected_record_count": len(_reference_job_ids()),
        "selected_job_ids_sha256": _sequence_sha256(_reference_job_ids()),
    }


def _validate_reference_jobs(reference_plan: Mapping[str, Any]) -> None:
    items = reference_plan.get("jobs")
    if not isinstance(items, list):
        raise ConditionedScreenError("reference plan jobs are not a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ConditionedScreenError("reference plan job is not an object")
        job = ScreenJob.from_mapping(item)
        if item.get("job_id") != job.job_id or job.job_id in by_id:
            raise ConditionedScreenError("reference plan job identity is invalid")
        by_id[job.job_id] = item
    missing = set(_reference_job_ids()) - set(by_id)
    if missing:
        raise ConditionedScreenError(
            f"reference plan lacks {len(missing)} required comparison jobs"
        )


def build_manifest(
    *,
    cache_root: Path,
    reference_run_root: Path,
    worker_gpu_uuids: Sequence[str],
) -> dict[str, Any]:
    """Bind cache/splits and source/runtime identities to the reference run."""

    cache_root = cache_root.resolve()
    reference_run_root = reference_run_root.resolve()
    reference_plan = load_reference_plan(reference_run_root)
    if reference_plan["plan_sha256"] != EXPECTED_REFERENCE_PLAN_SHA256:
        raise ConditionedScreenError(
            "reference run is not the prespecified corrected generation"
        )
    _validate_reference_jobs(reference_plan)

    cache_identities: dict[str, dict[str, Any]] = {}
    split_identities: dict[str, dict[str, Any]] = {}
    for dataset, subjects in CURATED_SPLITS:
        for subject in subjects:
            cache, split = _read_cache_plan_identity(
                cache_root, dataset, subject
            )
            subject_key = _subject_key(dataset, subject)
            split_key = _split_key(dataset, subject)
            if cache != reference_plan["cache_identity"][subject_key]:
                raise ConditionedScreenError(
                    f"cache identity differs from reference for {subject_key}"
                )
            if split != reference_plan["split_identity"][split_key]:
                raise ConditionedScreenError(
                    f"split identity differs from reference for {split_key}"
                )
            cache_identities[subject_key] = cache
            split_identities[split_key] = split

    # Planning is allowed only for three distinct GPUs that are idle at this
    # instant.  Workers repeat this guard before every unfinished job.
    gpu_uuids = _validate_gpu_uuids(worker_gpu_uuids)
    for gpu_uuid in gpu_uuids:
        cooperative_gpu_guard(gpu_uuid)

    return assemble_manifest(
        cache_root=str(cache_root),
        cache_identity=cache_identities,
        split_identity=split_identities,
        source_hashes=source_identity(),
        environment_identity=conditioned_environment_identity(),
        worker_gpu_uuids=gpu_uuids,
        reference_binding=_reference_binding(
            reference_run_root, reference_plan
        ),
    )


def validate_manifest_payload(plan: Mapping[str, Any]) -> None:
    """Fail closed unless ``plan`` is the exact v3 immutable manifest."""

    expected_top = {
        "schema",
        "purpose",
        "evaluation_scope",
        "confirmation_evidence",
        "seed",
        "worker_count",
        "jobs_per_worker",
        "cache_root",
        "curated_splits",
        "models",
        "train_config",
        "channel_scaling",
        "cache_identity",
        "split_identity",
        "source_identity",
        "environment_identity",
        "reference_binding",
        "worker_bindings",
        "jobs",
        "n_jobs",
        "artifact_contract",
        "plan_sha256",
    }
    if set(plan) != expected_top:
        raise ConditionedScreenError("conditioned plan fields differ from v3")
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["purpose"]
        != "prespecified_conditioned_development_validation_screen"
        or plan["evaluation_scope"] != EVALUATION_SCOPE
        or plan["confirmation_evidence"] is not False
        or int(plan["seed"]) != SEED
        or int(plan["worker_count"]) != WORKER_COUNT
        or int(plan["jobs_per_worker"]) != EXPECTED_JOBS_PER_WORKER
        or int(plan["n_jobs"]) != EXPECTED_JOB_COUNT
    ):
        raise ConditionedScreenError("conditioned plan fixed identity changed")
    if plan["train_config"] != dict(TRAIN_CONFIG):
        raise ConditionedScreenError("conditioned plan training recipe changed")
    if plan["channel_scaling"] != dict(CHANNEL_SCALING):
        raise ConditionedScreenError("conditioned plan scaler contract changed")
    _validate_identity_maps(plan["cache_identity"], plan["split_identity"])
    _validate_environment_identity(plan["environment_identity"])
    _validate_reference_binding(plan["reference_binding"])
    if set(plan["source_identity"]) != set(SOURCE_FILES):
        raise ConditionedScreenError("conditioned source closure changed")
    if plan["plan_sha256"] != _payload_sha256(plan, "plan_sha256"):
        raise ConditionedScreenError("conditioned plan SHA-256 mismatch")

    model_rows = plan["models"]
    expected_model_rows = [
        {
            "name": model,
            "factory": MODEL_FACTORY,
            "maximum_modulation": MODEL_MAXIMUM_MODULATION[model],
            "expected_source_statistics_hook": False,
        }
        for model in CONDITIONED_MODELS
    ]
    if model_rows != expected_model_rows:
        raise ConditionedScreenError("conditioned model definitions changed")

    bindings = plan["worker_bindings"]
    if not isinstance(bindings, list) or len(bindings) != WORKER_COUNT:
        raise ConditionedScreenError("conditioned worker bindings changed")
    gpu_uuids = _validate_gpu_uuids(
        [str(item["gpu_uuid"]) for item in bindings]
    )
    for index, item in enumerate(bindings):
        if item != {
            "worker_index": index,
            "model": CONDITIONED_MODELS[index],
            "gpu_uuid": gpu_uuids[index],
            "job_count": EXPECTED_JOBS_PER_WORKER,
        }:
            raise ConditionedScreenError("conditioned worker binding changed")

    items = plan["jobs"]
    expected_jobs = conditioned_jobs()
    if not isinstance(items, list) or len(items) != len(expected_jobs):
        raise ConditionedScreenError("conditioned plan job count changed")
    for expected, item in zip(expected_jobs, items, strict=True):
        expected_worker = CONDITIONED_MODELS.index(expected.model)
        if not isinstance(item, Mapping) or item != {
            **expected.identity(),
            "job_id": expected.job_id,
            "worker_index": expected_worker,
        }:
            raise ConditionedScreenError("conditioned plan job order changed")
    assert_validation_scope(plan)


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> bool:
    """Create JSON atomically without replacing an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def write_or_validate_plan(run_root: Path, plan: Mapping[str, Any]) -> bool:
    validate_manifest_payload(plan)
    path = run_root / "plan.json"
    if _atomic_create_json(path, plan):
        return True
    existing = _strict_json_load(path)
    validate_manifest_payload(existing)
    if existing != dict(plan):
        raise ConditionedScreenError(
            "run root contains a different immutable conditioned plan"
        )
    return False


def load_plan(run_root: Path) -> dict[str, Any]:
    plan = _strict_json_load(run_root / "plan.json")
    validate_manifest_payload(plan)
    return plan


def _assert_no_excluded_evidence(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject fields capable of persisting excluded rows or raw outcomes."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            tokens = frozenset(token for token in normalized.split("_") if token)
            if normalized in FORBIDDEN_ARTIFACT_KEYS or (
                tokens & FORBIDDEN_SCOPE_TOKENS
            ):
                location = ".".join((*path, str(key)))
                raise ValueError(
                    f"forbidden evidence field at {location}: {normalized!r}"
                )
            _assert_no_excluded_evidence(child, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_excluded_evidence(child, (*path, str(index)))


def validate_record_payload(
    payload: Mapping[str, Any],
    *,
    job: ScreenJob,
    plan_sha256: str,
) -> None:
    """Validate one aggregate-only conditioned validation record."""

    expected_keys = {
        "schema",
        "created_at",
        "plan_sha256",
        "job_id",
        "job",
        "evaluation_scope",
        "confirmation_evidence",
        "cache_identity",
        "source_identity",
        "split",
        "scaler",
        "model",
        "fit",
        "validation_metrics",
        "timing",
        "resource_preflight",
    }
    if set(payload) != expected_keys:
        raise ValueError("conditioned record fields differ from v3")
    if payload["schema"] != RECORD_SCHEMA:
        raise ValueError("conditioned record schema mismatch")
    if payload["plan_sha256"] != plan_sha256:
        raise ValueError("conditioned record plan hash mismatch")
    if payload["job_id"] != job.job_id or payload["job"] != job.identity():
        raise ValueError("conditioned record job identity mismatch")
    if (
        payload["evaluation_scope"] != EVALUATION_SCOPE
        or payload["confirmation_evidence"] is not False
    ):
        raise ValueError("conditioned record evidence scope changed")
    split = payload["split"]
    if not isinstance(split, Mapping) or set(split) != {"train", "validation"}:
        raise ValueError("conditioned record split exceeds validation scope")
    metrics = payload["validation_metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != METRIC_NAMES:
        raise ValueError("conditioned validation metric set changed")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"conditioned validation metric {name} is invalid")
    model = payload["model"]
    if not isinstance(model, Mapping):
        raise ValueError("conditioned model identity is missing")
    hook = model.get("source_statistics")
    if not isinstance(hook, Mapping) or set(hook) != {
        "available",
        "ran",
        "fit_scope",
        "outcome_aware",
    }:
        raise ValueError("conditioned source-statistics identity is invalid")
    if (
        not isinstance(hook["available"], bool)
        or not isinstance(hook["ran"], bool)
        or hook["ran"] is not hook["available"]
        or hook["fit_scope"] != "training_rows_only"
        or hook["outcome_aware"] is not False
    ):
        raise ValueError("conditioned source-statistics semantics changed")
    _assert_no_excluded_evidence(payload)


def _validate_record_bindings(
    payload: Mapping[str, Any],
    *,
    job: ScreenJob,
    plan: Mapping[str, Any],
) -> None:
    subject_key = _subject_key(job.dataset, job.subject)
    split_key = _split_key(job.dataset, job.subject, job.fold)
    expected_cache = {
        name: plan["cache_identity"][subject_key][name]
        for name in ("array_sha256", "file_sha256")
    }
    if payload["cache_identity"] != expected_cache:
        raise ValueError("conditioned record cache identity differs from plan")
    if payload["source_identity"] != plan["source_identity"]:
        raise ValueError("conditioned record source identity differs from plan")
    if payload["split"] != plan["split_identity"][split_key]:
        raise ValueError("conditioned record split identity differs from plan")
    if payload["fit"].get("config") != plan["train_config"]:
        raise ValueError("conditioned record training recipe differs from plan")
    scaler = payload.get("scaler")
    if (
        not isinstance(scaler, Mapping)
        or set(scaler)
        != {"fit_scope", "mean_sha256", "std_sha256", "contract"}
        or scaler["fit_scope"] != "training_only"
        or scaler["contract"] != plan["channel_scaling"]
        or any(
            not isinstance(scaler[name], str)
            or not HEX_64_RE.fullmatch(scaler[name])
            for name in ("mean_sha256", "std_sha256")
        )
    ):
        raise ValueError("conditioned record scaler is not source-only")
    identity = payload["model"].get("identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("requested_name") != job.model
    ):
        raise ValueError("conditioned record requested model differs")
    expected_hook = next(
        item["expected_source_statistics_hook"]
        for item in plan["models"]
        if item["name"] == job.model
    )
    hook = payload["model"]["source_statistics"]
    if hook["available"] is not expected_hook or hook["ran"] is not expected_hook:
        raise ValueError("conditioned source-statistics hook differs from plan")
    worker_index = CONDITIONED_MODELS.index(job.model)
    worker_binding = plan["worker_bindings"][worker_index]
    resource = payload.get("resource_preflight")
    gpu = resource.get("gpu") if isinstance(resource, Mapping) else None
    disk = resource.get("disk") if isinstance(resource, Mapping) else None
    if (
        not isinstance(gpu, Mapping)
        or gpu.get("gpu_uuid") != worker_binding["gpu_uuid"]
    ):
        raise ValueError("conditioned record GPU differs from worker binding")
    if "foreign_compute_processes" in gpu and gpu[
        "foreign_compute_processes"
    ] != []:
        raise ValueError("conditioned record reports a foreign GPU process")
    if (
        not isinstance(disk, Mapping)
        or float(disk.get("free_gib", -1.0)) < MINIMUM_FREE_GIB
    ):
        raise ValueError("conditioned record disk preflight is below 50 GiB")
    if (
        "minimum_free_gib" in disk
        and float(disk["minimum_free_gib"]) != MINIMUM_FREE_GIB
    ):
        raise ValueError("conditioned record disk low-water mark changed")
    bounded = ("accuracy", "balanced_accuracy", "roc_auc")
    if any(
        not 0.0 <= float(payload["validation_metrics"][name]) <= 1.0
        for name in bounded
    ):
        raise ValueError("conditioned record contains an out-of-range metric")
    if not -1.0 <= float(payload["validation_metrics"]["cohen_kappa"]) <= 1.0:
        raise ValueError("conditioned record contains invalid Cohen kappa")
    if float(payload["validation_metrics"]["negative_log_likelihood"]) < 0.0:
        raise ValueError("conditioned record contains negative log loss")
    for name in (
        "construction_seconds",
        "fit_seconds",
        "evaluation_seconds",
        "total_seconds",
    ):
        value = payload["timing"].get(name)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"conditioned timing {name} is invalid")


def run_job_atomically(
    *,
    record_path: Path,
    job: ScreenJob,
    plan_sha256: str,
    executor: Callable[[ScreenJob], Mapping[str, Any]],
) -> bool:
    """Run one missing job, or validate and skip a completed atomic record."""

    if record_path.exists():
        existing = _strict_json_load(record_path)
        validate_record_payload(
            existing, job=job, plan_sha256=plan_sha256
        )
        return False
    payload = dict(executor(job))
    validate_record_payload(payload, job=job, plan_sha256=plan_sha256)
    if _atomic_create_json(record_path, payload):
        return True
    raced = _strict_json_load(record_path)
    validate_record_payload(raced, job=job, plan_sha256=plan_sha256)
    return False


_VERIFIED_CACHE_BINDINGS: dict[Path, tuple[int, str]] = {}


def _load_validation_data(
    job: ScreenJob,
    plan: Mapping[str, Any],
    cache_root: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    str,
]:
    """Materialize only the plan-bound training and validation rows."""

    subject_key = _subject_key(job.dataset, job.subject)
    split_key = _split_key(job.dataset, job.subject, job.fold)
    cache_contract = plan["cache_identity"][subject_key]
    split_contract = plan["split_identity"][split_key]
    path = cache_root / str(cache_contract["relative_path"])
    expected_binding = (
        int(cache_contract["file_size_bytes"]),
        str(cache_contract["file_sha256"]),
    )
    observed_binding = _VERIFIED_CACHE_BINDINGS.get(path)
    if observed_binding is None:
        if path.stat().st_size != expected_binding[0]:
            raise ConditionedScreenError(f"cache size changed after plan: {path}")
        if _file_sha256(path) != expected_binding[1]:
            raise ConditionedScreenError(f"cache hash changed after plan: {path}")
        _VERIFIED_CACHE_BINDINGS[path] = expected_binding
    elif observed_binding != expected_binding:
        raise ConditionedScreenError("cache path reused with a different identity")

    with np.load(path, allow_pickle=False) as archive:
        sessions = archive["sessions"].astype(str)
        runs = archive["runs"].astype(str)
        train_rows, validation_rows = development_rows(
            job.dataset, job.subject, sessions, runs
        )
        if _rows_sha256(train_rows) != split_contract["train"]["rows_sha256"]:
            raise ConditionedScreenError("training rows differ from plan")
        if (
            _rows_sha256(validation_rows)
            != split_contract["validation"]["rows_sha256"]
        ):
            raise ConditionedScreenError("validation rows differ from plan")
        x_train = np.asarray(archive["x"][train_rows], dtype=np.float32).copy()
        y_train = np.asarray(archive["y"][train_rows], dtype=np.int64).copy()
        x_validation = np.asarray(
            archive["x"][validation_rows], dtype=np.float32
        ).copy()
        y_validation = np.asarray(
            archive["y"][validation_rows], dtype=np.int64
        ).copy()
        positions = np.asarray(archive["positions"], dtype=np.float32).copy()
        channel_names = tuple(
            str(value) for value in archive["channel_names"].tolist()
        )
        identity = json.loads(str(archive["identity"].item()))

    if identity.get("array_sha256") != cache_contract["array_sha256"]:
        raise ConditionedScreenError("cache array identity differs from plan")
    expected_classes = set(range(dataset_spec(job.dataset).n_classes))
    if set(y_train.tolist()) != expected_classes:
        raise ConditionedScreenError("training rows do not contain every class")
    if set(y_validation.tolist()) != expected_classes:
        raise ConditionedScreenError("validation rows do not contain every class")
    if x_train.ndim != 3 or x_validation.ndim != 3:
        raise ConditionedScreenError("EEG arrays must be three-dimensional")
    if x_train.shape[1:] != x_validation.shape[1:]:
        raise ConditionedScreenError("training and validation shapes differ")
    if positions.shape != (x_train.shape[1], 3):
        raise ConditionedScreenError("coordinate shape differs from EEG channels")
    if len(channel_names) != x_train.shape[1]:
        raise ConditionedScreenError("channel metadata differs from EEG channels")
    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        positions,
        channel_names,
        str(identity["array_sha256"]),
    )


def _make_conditioned_model(
    job: ScreenJob,
    *,
    n_channels: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
    channel_names: tuple[str, ...],
    positions: np.ndarray,
) -> Any:
    import torch

    from .chsd_conditioned import make_chsd_conditioned_model

    if job.model not in CONDITIONED_MODELS:
        raise ConditionedScreenError(f"unexpected conditioned model {job.model}")
    return make_chsd_conditioned_model(
        requested_model=job.model,
        n_channels=n_channels,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
        channel_names=channel_names,
        channel_positions=torch.as_tensor(positions, dtype=torch.float32),
    )


def execute_validation_job(
    *,
    job: ScreenJob,
    plan: Mapping[str, Any],
    cache_root: Path,
    resource_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit on source rows and score only the fixed validation rows."""

    import torch

    from .data import apply_channel_scaler, fit_channel_scaler
    from .training import (
        TrainConfig,
        classification_metrics,
        configure_determinism,
        fit_model,
        predict_probabilities,
    )

    started = time.perf_counter()
    (
        x_train_raw,
        y_train,
        x_validation_raw,
        y_validation,
        positions,
        channel_names,
        cache_array_sha256,
    ) = _load_validation_data(job, plan, cache_root)

    # The channel scaler is fit from source rows only.
    scaler_mean, scaler_std = fit_channel_scaler(
        x_train_raw, channel_names
    )
    x_train = apply_channel_scaler(
        x_train_raw, scaler_mean, scaler_std
    )
    x_validation = apply_channel_scaler(
        x_validation_raw, scaler_mean, scaler_std
    )
    configure_determinism(job.seed)

    construction_started = time.perf_counter()
    preprocessing = preprocessing_for_dataset(job.dataset)
    model = _make_conditioned_model(
        job,
        n_channels=x_train.shape[1],
        n_outputs=dataset_spec(job.dataset).n_classes,
        n_times=x_train.shape[2],
        sfreq=float(preprocessing["sfreq_hz"]),
        channel_names=channel_names,
        positions=positions,
    )
    construction_seconds = time.perf_counter() - construction_started
    constructed_state_sha256 = _state_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_identity = _model_identity(model, job.model)

    # Optional source-statistics initialization is outcome-blind by contract.
    source_hook = getattr(model, "fit_source_statistics", None)
    source_hook_available = callable(source_hook)
    source_hook_ran = False
    if source_hook_available:
        device = torch.device("cuda:0")
        model = model.to(device)
        source_x = torch.as_tensor(
            x_train, dtype=torch.float32, device=device
        )
        source_positions = torch.as_tensor(
            positions, dtype=torch.float32, device=device
        )
        source_hook(source_x, source_positions)
        source_hook_ran = True
        del source_x, source_positions
    initial_state_sha256 = _state_sha256(model)

    config = TrainConfig(**dict(plan["train_config"]))
    fit = fit_model(
        model,
        x_train,
        y_train,
        x_validation,
        y_validation,
        positions,
        config=config,
        mirror_index=None,
    )
    evaluation_started = time.perf_counter()
    probabilities = predict_probabilities(
        fit["model"],
        x_validation,
        positions,
        device="cuda:0",
    )
    metrics = classification_metrics(y_validation, probabilities)
    evaluation_seconds = time.perf_counter() - evaluation_started
    selected_state_sha256 = _state_sha256(fit["model"])
    total_seconds = time.perf_counter() - started

    record = {
        "schema": RECORD_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan["plan_sha256"],
        "job_id": job.job_id,
        "job": job.identity(),
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "cache_identity": {
            "array_sha256": cache_array_sha256,
            "file_sha256": plan["cache_identity"][
                _subject_key(job.dataset, job.subject)
            ]["file_sha256"],
        },
        "source_identity": dict(plan["source_identity"]),
        "split": {
            "train": dict(
                plan["split_identity"][
                    _split_key(job.dataset, job.subject, job.fold)
                ]["train"]
            ),
            "validation": dict(
                plan["split_identity"][
                    _split_key(job.dataset, job.subject, job.fold)
                ]["validation"]
            ),
        },
        "scaler": {
            "fit_scope": "training_only",
            "mean_sha256": _array_sha256(scaler_mean),
            "std_sha256": _array_sha256(scaler_std),
            "contract": dict(CHANNEL_SCALING),
        },
        "model": {
            "identity": _json_safe(model_identity),
            "parameter_count": int(parameter_count),
            "constructed_state_sha256": constructed_state_sha256,
            "initial_state_sha256": initial_state_sha256,
            "selected_state_sha256": selected_state_sha256,
            "source_statistics": {
                "available": source_hook_available,
                "ran": source_hook_ran,
                "fit_scope": "training_rows_only",
                "outcome_aware": False,
            },
        },
        "fit": {
            "best_epoch": int(fit["best_epoch"]),
            "epochs_run": int(fit["epochs_run"]),
            "best_validation_loss": float(fit["best_validation_loss"]),
            "early_stopping": True,
            "config": dict(fit["config"]),
        },
        "validation_metrics": {
            name: float(metrics[name]) for name in sorted(METRIC_NAMES)
        },
        "timing": {
            "construction_seconds": construction_seconds,
            "fit_seconds": float(fit["fit_seconds"]),
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": total_seconds,
        },
        "resource_preflight": dict(resource_preflight),
    }
    validate_record_payload(
        record, job=job, plan_sha256=str(plan["plan_sha256"])
    )
    _validate_record_bindings(record, job=job, plan=plan)
    del probabilities, fit, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _worker_binding(
    plan: Mapping[str, Any], worker_index: int
) -> Mapping[str, Any]:
    bindings = [
        item
        for item in plan["worker_bindings"]
        if int(item["worker_index"]) == worker_index
    ]
    if len(bindings) != 1:
        raise ConditionedScreenError("worker binding is absent or duplicated")
    return bindings[0]


def _verify_runtime(plan: Mapping[str, Any]) -> None:
    if source_identity() != plan["source_identity"]:
        raise ConditionedScreenError("source differs from immutable v3 plan")
    if conditioned_environment_identity() != plan["environment_identity"]:
        raise ConditionedScreenError(
            "UV/runtime/package/driver identity differs from v3 plan"
        )


def run_worker(
    *,
    run_root: Path,
    cache_root: Path,
    reference_run_root: Path,
    worker_index: int,
    gpu_uuid: str,
) -> int:
    """Execute one plan-bound, one-variant static GPU partition."""

    plan = load_plan(run_root)
    binding = _worker_binding(plan, worker_index)
    if gpu_uuid != binding["gpu_uuid"]:
        raise ConditionedScreenError("worker GPU UUID differs from plan binding")
    if str(cache_root.resolve()) != str(Path(plan["cache_root"]).resolve()):
        raise ConditionedScreenError("worker cache root differs from plan")
    if str(reference_run_root.resolve()) != str(
        Path(plan["reference_binding"]["run_root"]).resolve()
    ):
        raise ConditionedScreenError("worker reference root differs from plan")
    reference_plan = load_reference_plan(reference_run_root)
    if (
        reference_plan["plan_sha256"]
        != plan["reference_binding"]["plan_sha256"]
    ):
        raise ConditionedScreenError("reference plan binding changed")

    _worker_preflight(gpu_uuid)
    _verify_runtime(plan)
    jobs = tuple(
        ScreenJob.from_mapping(item)
        for item in plan["jobs"]
        if int(item["worker_index"]) == worker_index
    )
    expected = jobs_for_worker(conditioned_jobs(), worker_index)
    if jobs != expected or {job.model for job in jobs} != {binding["model"]}:
        raise ConditionedScreenError("worker partition differs from v3 plan")

    completed = 0
    resumed = 0
    for job in jobs:
        path = _record_path(run_root, job)
        if path.exists():
            existing = _strict_json_load(path)
            validate_record_payload(
                existing,
                job=job,
                plan_sha256=str(plan["plan_sha256"]),
            )
            _validate_record_bindings(existing, job=job, plan=plan)
            resumed += 1
            continue

        def executor(current: ScreenJob) -> Mapping[str, Any]:
            resource = {
                "gpu": cooperative_gpu_guard(gpu_uuid),
                "disk": disk_guard(run_root),
            }
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"worker={worker_index} start {current.job_id}",
                flush=True,
            )
            return execute_validation_job(
                job=current,
                plan=plan,
                cache_root=cache_root,
                resource_preflight=resource,
            )

        if run_job_atomically(
            record_path=path,
            job=job,
            plan_sha256=str(plan["plan_sha256"]),
            executor=executor,
        ):
            completed += 1
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"worker={worker_index} complete {job.job_id}",
                flush=True,
            )
        else:
            resumed += 1
    print(
        f"worker={worker_index} completed={completed} resumed={resumed}",
        flush=True,
    )
    return 0


def screen_status(run_root: Path) -> dict[str, Any]:
    plan = load_plan(run_root)
    missing: list[str] = []
    corrupt: dict[str, str] = {}
    complete = 0
    workers = {
        str(index): {"expected": EXPECTED_JOBS_PER_WORKER, "complete": 0}
        for index in range(WORKER_COUNT)
    }
    for item in plan["jobs"]:
        job = ScreenJob.from_mapping(item)
        worker_index = int(item["worker_index"])
        path = _record_path(run_root, job)
        if not path.is_file():
            missing.append(job.job_id)
            continue
        try:
            payload = _strict_json_load(path)
            validate_record_payload(
                payload,
                job=job,
                plan_sha256=str(plan["plan_sha256"]),
            )
            _validate_record_bindings(payload, job=job, plan=plan)
        except Exception as error:
            corrupt[job.job_id] = f"{type(error).__name__}: {error}"
        else:
            complete += 1
            workers[str(worker_index)]["complete"] += 1
    return {
        "schema": STATUS_SCHEMA,
        "evaluation_scope": EVALUATION_SCOPE,
        "plan_sha256": plan["plan_sha256"],
        "reference_plan_sha256": plan["reference_binding"]["plan_sha256"],
        "expected": EXPECTED_JOB_COUNT,
        "complete": complete,
        "missing": len(missing),
        "corrupt": len(corrupt),
        "workers": workers,
        "missing_job_ids": missing,
        "corrupt_records": corrupt,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="freeze the 51-job v3 manifest")
    plan.add_argument("--run-root", type=Path, required=True)
    plan.add_argument("--cache-root", type=Path, required=True)
    plan.add_argument("--reference-run-root", type=Path, required=True)
    plan.add_argument(
        "--gpu-uuid",
        action="append",
        required=True,
        help="full GPU UUID; repeat exactly three times in worker order",
    )

    status = subparsers.add_parser("status", help="audit v3 records")
    status.add_argument("--run-root", type=Path, required=True)

    worker = subparsers.add_parser(
        "worker", help="run one plan-bound variant/GPU partition"
    )
    worker.add_argument("--run-root", type=Path, required=True)
    worker.add_argument("--cache-root", type=Path, required=True)
    worker.add_argument("--reference-run-root", type=Path, required=True)
    worker.add_argument(
        "--worker-index",
        type=int,
        choices=range(WORKER_COUNT),
        required=True,
    )
    worker.add_argument("--gpu-uuid", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        manifest = build_manifest(
            cache_root=args.cache_root,
            reference_run_root=args.reference_run_root,
            worker_gpu_uuids=args.gpu_uuid,
        )
        created = write_or_validate_plan(args.run_root.resolve(), manifest)
        print(
            json.dumps(
                {
                    "created": created,
                    "plan_sha256": manifest["plan_sha256"],
                    "reference_plan_sha256": manifest["reference_binding"][
                        "plan_sha256"
                    ],
                    "n_jobs": manifest["n_jobs"],
                    "worker_count": manifest["worker_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "status":
        print(
            json.dumps(
                screen_status(args.run_root.resolve()),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "worker":
        return run_worker(
            run_root=args.run_root.resolve(),
            cache_root=args.cache_root.resolve(),
            reference_run_root=args.reference_run_root.resolve(),
            worker_index=args.worker_index,
            gpu_uuid=args.gpu_uuid,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONDITIONED_MODELS",
    "EXPECTED_JOB_COUNT",
    "EXPECTED_JOBS_PER_WORKER",
    "EXPECTED_REFERENCE_PLAN_SHA256",
    "PLAN_SCHEMA",
    "RECORD_SCHEMA",
    "REFERENCE_MODELS",
    "SEED",
    "SOURCE_FILES",
    "WORKER_COUNT",
    "assemble_manifest",
    "conditioned_jobs",
    "jobs_for_worker",
    "load_plan",
    "run_job_atomically",
    "screen_status",
    "validate_manifest_payload",
    "validate_record_payload",
]
