"""Scratch-only, resumable validation screen for eight EEG-MI candidates.

The screen is development evidence, not confirmation evidence.  It optimizes
on fixed training rows, selects checkpoints on fixed validation rows, and
persists validation metrics only.  Excluded evaluation rows are never
materialized by this module, passed to a model, scored, or represented in an
artifact.

Four independent workers receive an immutable modulo partition of the
136-job manifest.  ``CUDA_VISIBLE_DEVICES`` must be set externally to one full
GPU UUID; every worker addresses it as ``cuda:0``.  Before each unfinished job
the worker verifies that exact UUID with ``nvidia-smi``, rejects every compute
PID other than itself, and enforces a 50 GiB filesystem low-water mark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# NumPy may initialize its BLAS pool at import time, so cap shared-workstation
# thread pools before importing it.  The worker also applies the equivalent
# PyTorch limits before model construction.
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
    LOCAL_EXP4_SUBJECT_RUNS,
    PREPROCESSING,
    dataset_spec,
    preprocessing_for_dataset,
)
from .data import PHYSIONET_MI_RUNS


PLAN_SCHEMA = "ieee-mi-opened-development-screen-plan-v1"
RECORD_SCHEMA = "ieee-mi-opened-development-validation-record-v1"
EVALUATION_SCOPE = "opened_development_validation_only"
SEED = 7
WORKER_COUNT = 4
CPU_THREADS = 4
MINIMUM_FREE_GIB = 50.0
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
ENVIRONMENT_PACKAGES: tuple[str, ...] = (
    "braindecode",
    "mne",
    "moabb",
    "numpy",
    "pandas",
    "pyriemann",
    "scikit-learn",
    "scipy",
    "torch",
)

CURATED_SPLITS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("local_exp4", (1, 4, 7)),
    ("bnci2014_001", (1, 5, 9)),
    ("bnci2014_004", (1, 5, 9)),
    ("cho2017", (1, 18, 35, 52)),
    ("physionet_mi", (1, 18, 36, 54)),
)

CANDIDATE_MODELS: tuple[str, ...] = (
    "chsdnet",
    "chsdnet_hybrid",
    "chsdnet_direct",
    "chsdnet_joint",
    "cardinal_fbc_micro_extended",
    "tcformer",
    "fbcnet",
    "eegnet",
)

MODEL_FACTORIES: Mapping[str, str] = {
    "chsdnet": "ieee_mi.chsd:make_chsd_model",
    "chsdnet_hybrid": "ieee_mi.chsd_hybrid:make_chsd_hybrid_model",
    "chsdnet_direct": "ieee_mi.chsd_direct:make_chsd_direct_model",
    "chsdnet_joint": "ieee_mi.chsd_joint:make_chsd_joint_model",
    "cardinal_fbc_micro_extended": "ieee_mi.baselines:make_model",
    "tcformer": "ieee_mi.baselines:make_model",
    "fbcnet": "ieee_mi.baselines:make_model",
    "eegnet": "ieee_mi.baselines:make_model",
}

# This is the established common recipe, fixed here rather than exposed as
# search-space CLI arguments.
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

SOURCE_FILES: tuple[str, ...] = (
    "ieee_mi/development_screen.py",
    "ieee_mi/chsd.py",
    "ieee_mi/chsd_hybrid.py",
    "ieee_mi/chsd_direct.py",
    "ieee_mi/chsd_joint.py",
    "ieee_mi/baselines.py",
    "ieee_mi/models.py",
    "ieee_mi/training.py",
    "ieee_mi/data.py",
    "ieee_mi/config.py",
)

METRIC_NAMES = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "cohen_kappa",
        "negative_log_likelihood",
        "roc_auc",
    }
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


class DevelopmentScreenError(RuntimeError):
    """Base error for a development-screen contract violation."""


class ResourceUnavailable(DevelopmentScreenError):
    """Raised when the cooperative GPU or disk guard refuses a new job."""


@dataclass(frozen=True, order=True)
class ScreenJob:
    dataset: str
    model: str
    subject: int
    fold: int = 0
    seed: int = SEED

    @property
    def job_id(self) -> str:
        return (
            f"{self.dataset}__{self.model}__s{self.subject:03d}"
            f"__f{self.fold:02d}__seed{self.seed:03d}"
        )

    def identity(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "subject": self.subject,
            "fold": self.fold,
            "seed": self.seed,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScreenJob":
        return cls(
            dataset=str(value["dataset"]),
            model=str(value["model"]),
            subject=int(value["subject"]),
            fold=int(value["fold"]),
            seed=int(value["seed"]),
        )


@dataclass(frozen=True)
class ValidationData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    positions: np.ndarray
    channel_names: tuple[str, ...]
    cache_array_sha256: str


def curated_jobs() -> tuple[ScreenJob, ...]:
    """Return the immutable 136-job manifest order."""

    return tuple(
        ScreenJob(dataset=dataset, model=model, subject=subject)
        for dataset, subjects in CURATED_SPLITS
        for subject in subjects
        for model in CANDIDATE_MODELS
    )


def jobs_for_worker(
    jobs: Sequence[ScreenJob],
    worker_index: int,
    *,
    worker_count: int = WORKER_COUNT,
) -> tuple[ScreenJob, ...]:
    """Return one deterministic, non-overlapping modulo partition."""

    if worker_count != WORKER_COUNT:
        raise ValueError(f"worker_count is fixed at {WORKER_COUNT}")
    if not 0 <= worker_index < worker_count:
        raise ValueError(f"worker_index must lie in [0, {worker_count - 1}]")
    return tuple(jobs[worker_index::worker_count])


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, Any], hash_key: str) -> str:
    materialized = dict(value)
    materialized.pop(hash_key, None)
    return hashlib.sha256(_canonical_json_bytes(materialized)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _rows_sha256(rows: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(rows, dtype=np.int64).tobytes()
    ).hexdigest()


def _subject_key(dataset: str, subject: int) -> str:
    return f"{dataset}:s{subject:03d}"


def _split_key(dataset: str, subject: int, fold: int = 0) -> str:
    return f"{dataset}:s{subject:03d}:f{fold:02d}"


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


def development_rows(
    dataset: str,
    subject: int,
    sessions: np.ndarray,
    runs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive only training and validation rows from non-outcome metadata.

    Cho2017 is the one single-run dataset.  Its cache contract contains exactly
    200 class-contiguous trials (100 per class).  Fold zero reserves block zero
    of each 100-row sequence outside this screen, uses block one for validation,
    and uses blocks two through four for optimization.  The partition is thus
    derived from row order without reading the excluded outcomes.
    """

    sessions = np.asarray(sessions).astype(str)
    runs = np.asarray(runs).astype(str)
    if sessions.ndim != 1 or runs.ndim != 1 or len(sessions) != len(runs):
        raise ValueError("sessions and runs must be equal-length vectors")
    rows = np.arange(len(sessions), dtype=np.int64)

    if dataset == "local_exp4":
        expected = tuple(str(value) for value in LOCAL_EXP4_SUBJECT_RUNS[subject])
        train = rows[np.isin(runs, expected[:2])]
        validation = rows[runs == expected[2]]
    elif dataset == "bnci2014_001":
        source = rows[sessions == "0train"]
        source_runs = sorted(set(runs[source].tolist()))
        if len(source_runs) < 2:
            raise RuntimeError("BNCI2014-001 source session has too few runs")
        validation_run = source_runs[-1]
        train = source[runs[source] != validation_run]
        validation = source[runs[source] == validation_run]
    elif dataset == "bnci2014_004":
        train = rows[np.isin(sessions, ("0train", "1train"))]
        validation = rows[sessions == "2train"]
    elif dataset == "cho2017":
        if len(rows) != 200:
            raise RuntimeError("Cho2017 screen requires the fixed 200-trial cache")
        class_size = 100
        block_size = class_size // 5
        validation = np.concatenate(
            (
                np.arange(block_size, 2 * block_size, dtype=np.int64),
                np.arange(
                    class_size + block_size,
                    class_size + 2 * block_size,
                    dtype=np.int64,
                ),
            )
        )
        train = np.concatenate(
            (
                np.arange(2 * block_size, class_size, dtype=np.int64),
                np.arange(
                    class_size + 2 * block_size,
                    2 * class_size,
                    dtype=np.int64,
                ),
            )
        )
    elif dataset == "physionet_mi":
        # Fold zero: source run 12 optimizes, source run 8 selects.
        train = rows[runs == PHYSIONET_MI_RUNS[2]]
        validation = rows[runs == PHYSIONET_MI_RUNS[1]]
    else:
        raise ValueError(f"dataset {dataset!r} is outside the curated screen")

    train = np.sort(np.asarray(train, dtype=np.int64))
    validation = np.sort(np.asarray(validation, dtype=np.int64))
    if not len(train) or not len(validation):
        raise RuntimeError(f"{dataset} S{subject} has an empty screen partition")
    if np.intersect1d(train, validation).size:
        raise RuntimeError("training and validation rows overlap")
    if train[0] < 0 or validation[0] < 0:
        raise RuntimeError("screen rows must be nonnegative")
    if train[-1] >= len(rows) or validation[-1] >= len(rows):
        raise RuntimeError("screen rows exceed the cache")
    return train, validation


def _cache_path(cache_root: Path, dataset: str, subject: int) -> Path:
    return (
        cache_root
        / str(PREPROCESSING["schema"])
        / DEFAULT_MONTAGE_PROFILE
        / dataset
        / f"subject_{subject:03d}.npz"
    )


def _read_cache_plan_identity(
    cache_root: Path,
    dataset: str,
    subject: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _cache_path(cache_root, dataset, subject)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"sessions", "runs", "identity"}
        if not required.issubset(archive.files):
            raise RuntimeError(f"cache metadata fields are missing from {path}")
        sessions = archive["sessions"].astype(str)
        runs = archive["runs"].astype(str)
        identity = json.loads(str(archive["identity"].item()))
    train, validation = development_rows(dataset, subject, sessions, runs)
    array_digest = identity.get("array_sha256")
    if not isinstance(array_digest, str) or not HEX_64_RE.fullmatch(array_digest):
        raise RuntimeError(f"cache {path} has no valid array identity")
    cache_identity = {
        "relative_path": str(path.relative_to(cache_root)),
        "file_size_bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
        "array_sha256": array_digest,
    }
    split_identity = {
        "train": {
            "count": len(train),
            "rows_sha256": _rows_sha256(train),
        },
        "validation": {
            "count": len(validation),
            "rows_sha256": _rows_sha256(validation),
        },
    }
    return cache_identity, split_identity


def source_identity(project_root: Path | None = None) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1] if project_root is None else project_root
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = _file_sha256(path)
    return result


def assemble_manifest(
    *,
    cache_root: str,
    cache_identity: Mapping[str, Mapping[str, Any]],
    split_identity: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    environment_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble and validate the immutable development-screen manifest."""

    if set(cache_identity) != _expected_subject_keys():
        raise ValueError("cache identity keys differ from curated subjects")
    if set(split_identity) != _expected_split_keys():
        raise ValueError("split identity keys differ from curated subject-folds")
    for key, value in cache_identity.items():
        if not all(
            isinstance(value.get(name), str)
            and bool(HEX_64_RE.fullmatch(str(value[name])))
            for name in ("file_sha256", "array_sha256")
        ):
            raise ValueError(f"cache identity {key} has invalid hashes")
    for key, value in split_identity.items():
        if set(value) != {"train", "validation"}:
            raise ValueError(f"split identity {key} exceeds validation scope")
        for partition in ("train", "validation"):
            item = value[partition]
            if (
                not isinstance(item, Mapping)
                or int(item.get("count", 0)) <= 0
                or not isinstance(item.get("rows_sha256"), str)
                or not HEX_64_RE.fullmatch(str(item["rows_sha256"]))
            ):
                raise ValueError(f"split identity {key}/{partition} is invalid")
    if not source_hashes or any(
        not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
        for value in source_hashes.values()
    ):
        raise ValueError("source hashes must be a nonempty SHA-256 mapping")

    jobs = curated_jobs()
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "purpose": "scratch_candidate_validation_screen",
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "seed": SEED,
        "worker_count": WORKER_COUNT,
        "cache_root": str(cache_root),
        "curated_splits": [
            {"dataset": dataset, "subjects": list(subjects), "fold": 0}
            for dataset, subjects in CURATED_SPLITS
        ],
        "models": [
            {"name": name, "factory": MODEL_FACTORIES[name]}
            for name in CANDIDATE_MODELS
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
        "jobs": [
            {
                **job.identity(),
                "job_id": job.job_id,
                "worker_index": index % WORKER_COUNT,
            }
            for index, job in enumerate(jobs)
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
    return payload


def build_manifest(cache_root: Path) -> dict[str, Any]:
    cache_root = cache_root.resolve()
    cache_identities: dict[str, dict[str, Any]] = {}
    split_identities: dict[str, dict[str, Any]] = {}
    for dataset, subjects in CURATED_SPLITS:
        for subject in subjects:
            cache, split = _read_cache_plan_identity(
                cache_root, dataset, subject
            )
            cache_identities[_subject_key(dataset, subject)] = cache
            split_identities[_split_key(dataset, subject)] = split
    return assemble_manifest(
        cache_root=str(cache_root),
        cache_identity=cache_identities,
        split_identity=split_identities,
        source_hashes=source_identity(),
        environment_identity=environment_identity(),
    )


def environment_identity() -> dict[str, Any]:
    """Fingerprint the isolated runtime without including worker GPU choice."""

    import torch

    try:
        driver_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise DevelopmentScreenError(
            f"cannot fingerprint the NVIDIA driver: {error}"
        ) from error
    drivers = sorted(
        {line.strip() for line in driver_result.stdout.splitlines() if line.strip()}
    )
    if not drivers:
        raise DevelopmentScreenError("NVIDIA driver fingerprint is empty")
    packages: dict[str, str] = {}
    for name in ENVIRONMENT_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise DevelopmentScreenError(
                f"required UV-environment package is missing: {name}"
            ) from error
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": sys.platform,
        "packages": packages,
        "torch_cuda_version": str(torch.version.cuda),
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_versions": drivers,
        "required_cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        "cpu_threads_per_worker": CPU_THREADS,
        "minimum_free_gib": MINIMUM_FREE_GIB,
    }


def _strict_json_load(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> bool:
    """Atomically create JSON without replacing an existing completed record."""

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
    path = run_root / "plan.json"
    if _atomic_create_json(path, plan):
        return True
    existing = _strict_json_load(path)
    if existing != dict(plan):
        raise DevelopmentScreenError(
            "run root contains a different immutable screen plan"
        )
    return False


def load_plan(run_root: Path) -> dict[str, Any]:
    plan = _strict_json_load(run_root / "plan.json")
    if plan.get("schema") != PLAN_SCHEMA:
        raise DevelopmentScreenError("unexpected development-screen plan schema")
    expected = _payload_sha256(plan, "plan_sha256")
    if plan.get("plan_sha256") != expected:
        raise DevelopmentScreenError("development-screen plan hash mismatch")
    if int(plan.get("n_jobs", -1)) != len(curated_jobs()):
        raise DevelopmentScreenError(
            "development-screen plan has the wrong fixed job count"
        )
    assert_validation_scope(plan)
    return plan


def assert_validation_scope(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject artifact keys that could carry excluded-partition evidence."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            tokens = frozenset(token for token in normalized.split("_") if token)
            forbidden_tokens = tokens & FORBIDDEN_SCOPE_TOKENS
            if normalized in FORBIDDEN_ARTIFACT_KEYS or forbidden_tokens:
                location = ".".join((*path, str(key)))
                raise ValueError(
                    f"forbidden non-validation artifact key at {location}: "
                    f"{normalized!r}"
                )
            assert_validation_scope(child, (*path, str(key)))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_validation_scope(child, (*path, str(index)))


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


def validate_record_payload(
    payload: Mapping[str, Any],
    *,
    job: ScreenJob,
    plan_sha256: str,
) -> None:
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
        raise ValueError("validation record fields differ from the fixed schema")
    if payload.get("schema") != RECORD_SCHEMA:
        raise ValueError("validation record schema mismatch")
    if payload.get("plan_sha256") != plan_sha256:
        raise ValueError("validation record plan hash mismatch")
    if payload.get("job_id") != job.job_id or payload.get("job") != job.identity():
        raise ValueError("validation record job identity mismatch")
    if (
        payload.get("evaluation_scope") != EVALUATION_SCOPE
        or payload.get("confirmation_evidence") is not False
    ):
        raise ValueError("validation record overstates its evidence scope")
    split = payload.get("split")
    if not isinstance(split, Mapping) or set(split) != {"train", "validation"}:
        raise ValueError("validation record split exceeds the permitted scope")
    metrics = payload.get("validation_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != METRIC_NAMES:
        raise ValueError("validation metric set differs from the fixed schema")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"validation metric {name} is not finite")
    if "history" in payload.get("fit", {}) or "checkpoint" in payload.get("fit", {}):
        raise ValueError("long training artifacts are forbidden")
    model_identity = payload.get("model")
    if not isinstance(model_identity, Mapping):
        raise ValueError("validation record model identity is missing")
    source_statistics = model_identity.get("source_statistics")
    if not isinstance(source_statistics, Mapping) or set(source_statistics) != {
        "available",
        "ran",
        "fit_scope",
        "outcome_aware",
    }:
        raise ValueError("source-statistics hook identity is invalid")
    expected_hook = job.model == "chsdnet_direct"
    if (
        source_statistics.get("available") is not expected_hook
        or source_statistics.get("ran") is not expected_hook
        or source_statistics.get("fit_scope") != "training_rows_only"
        or source_statistics.get("outcome_aware") is not False
    ):
        raise ValueError("source-statistics hook execution differs from model contract")
    assert_validation_scope(payload)


def run_job_atomically(
    *,
    record_path: Path,
    job: ScreenJob,
    plan_sha256: str,
    executor: Callable[[ScreenJob], Mapping[str, Any]],
) -> bool:
    """Run once, or skip an existing valid atomic record on resume."""

    if record_path.exists():
        validate_record_payload(
            _strict_json_load(record_path),
            job=job,
            plan_sha256=plan_sha256,
        )
        return False
    payload = dict(executor(job))
    validate_record_payload(payload, job=job, plan_sha256=plan_sha256)
    if _atomic_create_json(record_path, payload):
        return True
    validate_record_payload(
        _strict_json_load(record_path),
        job=job,
        plan_sha256=plan_sha256,
    )
    return False


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _model_identity(model: Any, requested_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested_name": requested_name,
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "uses_positions": bool(getattr(model, "uses_positions", False)),
    }
    config = getattr(model, "config", None)
    if config is not None:
        result["config"] = _json_safe(config)
    provenance = getattr(type(model), "_ieee_provenance", None)
    if provenance is not None:
        result["third_party_provenance"] = _json_safe(provenance)
    return result


def _make_model(
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

    position_tensor = torch.as_tensor(positions, dtype=torch.float32)
    common = {
        "requested_model": job.model,
        "n_channels": n_channels,
        "n_outputs": n_outputs,
        "n_times": n_times,
        "sfreq": sfreq,
        "channel_names": channel_names,
        "channel_positions": position_tensor,
    }
    if job.model == "chsdnet":
        from .chsd import make_chsd_model

        return make_chsd_model(**common)
    if job.model == "chsdnet_hybrid":
        from .chsd_hybrid import make_chsd_hybrid_model

        return make_chsd_hybrid_model(**common)
    if job.model == "chsdnet_direct":
        from .chsd_direct import make_chsd_direct_model

        return make_chsd_direct_model(**common)
    if job.model == "chsdnet_joint":
        from .chsd_joint import make_chsd_joint_model

        return make_chsd_joint_model(**common)
    from .baselines import make_model

    return make_model(
        job.model,
        n_channels=n_channels,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
        channel_names=channel_names,
        channel_positions=position_tensor,
    )


_VERIFIED_CACHE_FILES: set[Path] = set()


def _load_validation_data(
    job: ScreenJob,
    plan: Mapping[str, Any],
    cache_root: Path,
) -> ValidationData:
    """Load and expose only fixed training/validation cache rows."""

    subject_key = _subject_key(job.dataset, job.subject)
    split_key = _split_key(job.dataset, job.subject, job.fold)
    cache_contract = plan["cache_identity"][subject_key]
    split_contract = plan["split_identity"][split_key]
    path = cache_root / str(cache_contract["relative_path"])
    if path not in _VERIFIED_CACHE_FILES:
        if path.stat().st_size != int(cache_contract["file_size_bytes"]):
            raise DevelopmentScreenError(f"cache size changed after plan: {path}")
        if _file_sha256(path) != cache_contract["file_sha256"]:
            raise DevelopmentScreenError(f"cache file hash changed after plan: {path}")
        _VERIFIED_CACHE_FILES.add(path)

    with np.load(path, allow_pickle=False) as archive:
        sessions = archive["sessions"].astype(str)
        runs = archive["runs"].astype(str)
        train_rows, validation_rows = development_rows(
            job.dataset, job.subject, sessions, runs
        )
        if _rows_sha256(train_rows) != split_contract["train"]["rows_sha256"]:
            raise DevelopmentScreenError("training rows differ from the plan")
        if (
            _rows_sha256(validation_rows)
            != split_contract["validation"]["rows_sha256"]
        ):
            raise DevelopmentScreenError("validation rows differ from the plan")
        # NPZ compression may internally decompress a complete member, but this
        # module indexes and exposes only the two explicitly permitted row sets.
        x_train = np.asarray(archive["x"][train_rows], dtype=np.float32).copy()
        x_validation = np.asarray(
            archive["x"][validation_rows], dtype=np.float32
        ).copy()
        y_train = np.asarray(archive["y"][train_rows], dtype=np.int64).copy()
        y_validation = np.asarray(
            archive["y"][validation_rows], dtype=np.int64
        ).copy()
        positions = np.asarray(archive["positions"], dtype=np.float32).copy()
        channel_names = tuple(
            str(value) for value in archive["channel_names"].tolist()
        )
        identity = json.loads(str(archive["identity"].item()))

    if identity.get("array_sha256") != cache_contract["array_sha256"]:
        raise DevelopmentScreenError("cache array identity differs from the plan")
    n_classes = dataset_spec(job.dataset).n_classes
    expected_classes = set(range(n_classes))
    if set(y_train.tolist()) != expected_classes:
        raise DevelopmentScreenError("training rows do not contain every class")
    if set(y_validation.tolist()) != expected_classes:
        raise DevelopmentScreenError("validation rows do not contain every class")
    if x_train.ndim != 3 or x_validation.ndim != 3:
        raise DevelopmentScreenError("screen EEG arrays must be three-dimensional")
    if x_train.shape[1:] != x_validation.shape[1:]:
        raise DevelopmentScreenError("training/validation EEG shapes differ")
    if positions.shape != (x_train.shape[1], 3):
        raise DevelopmentScreenError("coordinate shape differs from EEG channels")
    if len(channel_names) != x_train.shape[1]:
        raise DevelopmentScreenError("channel metadata differs from EEG channels")
    return ValidationData(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        positions=positions,
        channel_names=channel_names,
        cache_array_sha256=str(identity["array_sha256"]),
    )


def execute_validation_job(
    *,
    job: ScreenJob,
    plan: Mapping[str, Any],
    cache_root: Path,
    resource_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit on training rows and compute metrics on validation rows only."""

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
    data = _load_validation_data(job, plan, cache_root)
    scaler_mean, scaler_std = fit_channel_scaler(
        data.x_train, data.channel_names
    )
    x_train = apply_channel_scaler(
        data.x_train, scaler_mean, scaler_std
    )
    x_validation = apply_channel_scaler(
        data.x_validation, scaler_mean, scaler_std
    )
    configure_determinism(job.seed)
    construction_started = time.perf_counter()
    preprocessing = preprocessing_for_dataset(job.dataset)
    model = _make_model(
        job,
        n_channels=x_train.shape[1],
        n_outputs=dataset_spec(job.dataset).n_classes,
        n_times=x_train.shape[2],
        sfreq=float(preprocessing["sfreq_hz"]),
        channel_names=data.channel_names,
        positions=data.positions,
    )
    construction_seconds = time.perf_counter() - construction_started
    constructed_state_sha256 = _state_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    identity = _model_identity(model, job.model)
    source_statistics_hook = getattr(model, "fit_source_statistics", None)
    source_statistics_available = callable(source_statistics_hook)
    source_statistics_ran = False
    if source_statistics_available:
        device = torch.device("cuda:0")
        model = model.to(device)
        source_x = torch.as_tensor(
            x_train, dtype=torch.float32, device=device
        )
        source_positions = torch.as_tensor(
            data.positions, dtype=torch.float32, device=device
        )
        # This optional initialization receives no outcomes.
        source_statistics_hook(source_x, source_positions)
        source_statistics_ran = True
        del source_x, source_positions
    initial_state_sha256 = _state_sha256(model)
    config = TrainConfig(**dict(plan["train_config"]))
    fit = fit_model(
        model,
        x_train,
        data.y_train,
        x_validation,
        data.y_validation,
        data.positions,
        config=config,
        mirror_index=None,
    )
    evaluation_started = time.perf_counter()
    probabilities = predict_probabilities(
        fit["model"],
        x_validation,
        data.positions,
        device="cuda:0",
    )
    metrics = classification_metrics(data.y_validation, probabilities)
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
            "array_sha256": data.cache_array_sha256,
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
            "identity": identity,
            "parameter_count": int(parameter_count),
            "constructed_state_sha256": constructed_state_sha256,
            "initial_state_sha256": initial_state_sha256,
            "selected_state_sha256": selected_state_sha256,
            "source_statistics": {
                "available": source_statistics_available,
                "ran": source_statistics_ran,
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
        record,
        job=job,
        plan_sha256=str(plan["plan_sha256"]),
    )
    del probabilities, fit, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _parse_csv_row(line: str) -> list[str]:
    return [value.strip() for value in line.split(",")]


def cooperative_gpu_guard(gpu_uuid: str) -> dict[str, Any]:
    """Reject every compute process on ``gpu_uuid`` except this worker PID."""

    if not GPU_UUID_RE.fullmatch(gpu_uuid):
        raise ValueError("gpu_uuid must be one full NVIDIA GPU UUID")
    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu_uuid,
                "--query-gpu=uuid,pci.bus_id,name,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        rows = [
            _parse_csv_row(line)
            for line in gpu_result.stdout.splitlines()
            if line.strip()
        ]
        if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0] != gpu_uuid:
            raise ValueError("GPU query did not resolve the exact requested UUID")
        process_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as error:
        raise ResourceUnavailable(f"nvidia-smi guard failed closed: {error}") from error

    own_pid = os.getpid()
    own_memory_mib = 0.0
    foreign: list[dict[str, Any]] = []
    for line in process_result.stdout.splitlines():
        if not line.strip():
            continue
        fields = _parse_csv_row(line)
        if len(fields) != 4 or fields[1] != gpu_uuid:
            continue
        pid = int(fields[0])
        try:
            memory = float(fields[3])
        except ValueError:
            memory = 0.0
        if pid == own_pid:
            own_memory_mib += memory
        else:
            foreign.append(
                {
                    "pid": pid,
                    "process_name": fields[2],
                    "used_gpu_memory_mib": memory,
                }
            )
    if foreign:
        raise ResourceUnavailable(
            f"GPU {gpu_uuid} has foreign compute PIDs "
            f"{[item['pid'] for item in foreign]}"
        )
    return {
        "gpu_uuid": gpu_uuid,
        "pci_bus_id": rows[0][1],
        "name": rows[0][2],
        "utilization_percent": float(rows[0][3]),
        "memory_used_mib": float(rows[0][4]),
        "own_pid": own_pid,
        "own_compute_memory_mib": own_memory_mib,
        "foreign_compute_processes": [],
    }


def disk_guard(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = os.statvfs(resolved)
    free_bytes = int(stat.f_bavail) * int(stat.f_frsize)
    free_gib = free_bytes / float(1024**3)
    if free_gib < MINIMUM_FREE_GIB:
        raise ResourceUnavailable(
            f"{free_gib:.2f} GiB free is below the fixed "
            f"{MINIMUM_FREE_GIB:.0f} GiB low-water mark"
        )
    return {
        "path": str(resolved),
        "free_bytes": free_bytes,
        "free_gib": free_gib,
        "minimum_free_gib": MINIMUM_FREE_GIB,
    }


def _require_uv_venv() -> None:
    if sys.prefix == sys.base_prefix:
        raise DevelopmentScreenError("an activated uv-managed virtualenv is required")
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    try:
        text = configuration.read_text(encoding="utf-8").lower()
    except OSError as error:
        raise DevelopmentScreenError(f"cannot read {configuration}: {error}") from error
    if not any(line.strip().startswith("uv =") for line in text.splitlines()):
        raise DevelopmentScreenError(
            f"{sys.prefix} is a virtualenv but is not marked as uv-managed"
        )


def _configure_cpu_threads() -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = str(CPU_THREADS)
    import torch

    torch.set_num_threads(CPU_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _canonical_cuda_uuid(value: Any) -> str:
    """Normalize PyTorch's string or ``torch._C._CUuuid`` representation."""

    if value is None:
        raise DevelopmentScreenError("PyTorch did not expose the visible GPU UUID")
    observed = str(value).strip()
    if not observed:
        raise DevelopmentScreenError("PyTorch exposed an empty visible GPU UUID")
    if not observed.startswith("GPU-"):
        observed = f"GPU-{observed}"
    if not GPU_UUID_RE.fullmatch(observed):
        raise DevelopmentScreenError(
            f"PyTorch exposed an invalid visible GPU UUID: {observed!r}"
        )
    return observed


def _worker_preflight(gpu_uuid: str) -> None:
    _require_uv_venv()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise DevelopmentScreenError(
            f"CUBLAS_WORKSPACE_CONFIG must be "
            f"{REQUIRED_CUBLAS_WORKSPACE_CONFIG!r} before Python starts"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise DevelopmentScreenError(
            "CUDA_VISIBLE_DEVICES must externally contain the exact worker GPU UUID"
        )
    _configure_cpu_threads()
    import torch

    if torch.cuda.device_count() != 1:
        raise DevelopmentScreenError("worker requires exactly one visible CUDA GPU")
    observed = _canonical_cuda_uuid(
        getattr(torch.cuda.get_device_properties(0), "uuid", None)
    )
    if observed != gpu_uuid:
        raise DevelopmentScreenError(
            f"cuda:0 resolved to {observed}, expected {gpu_uuid}"
        )


def _verify_source_identity(plan: Mapping[str, Any]) -> None:
    observed = source_identity()
    if observed != plan["source_identity"]:
        raise DevelopmentScreenError("source files differ from the immutable plan")


def _verify_environment_identity(plan: Mapping[str, Any]) -> None:
    observed = environment_identity()
    if observed != plan["environment_identity"]:
        raise DevelopmentScreenError(
            "UV environment or NVIDIA driver differs from the immutable plan"
        )


def run_worker(
    *,
    run_root: Path,
    cache_root: Path,
    worker_index: int,
    gpu_uuid: str,
) -> int:
    _worker_preflight(gpu_uuid)
    plan = load_plan(run_root)
    if str(cache_root.resolve()) != str(Path(plan["cache_root"]).resolve()):
        raise DevelopmentScreenError("worker cache root differs from the plan")
    _verify_source_identity(plan)
    _verify_environment_identity(plan)
    jobs = tuple(
        ScreenJob.from_mapping(item)
        for item in plan["jobs"]
        if int(item["worker_index"]) == worker_index
    )
    expected = jobs_for_worker(curated_jobs(), worker_index)
    if jobs != expected:
        raise DevelopmentScreenError("static worker partition differs from the plan")

    completed = 0
    skipped = 0
    for job in jobs:
        path = _record_path(run_root, job)
        if path.exists():
            validate_record_payload(
                _strict_json_load(path),
                job=job,
                plan_sha256=str(plan["plan_sha256"]),
            )
            skipped += 1
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
            skipped += 1
    print(
        f"worker={worker_index} completed={completed} resumed={skipped}",
        flush=True,
    )
    return 0


def screen_status(run_root: Path) -> dict[str, Any]:
    plan = load_plan(run_root)
    complete: list[str] = []
    missing: list[str] = []
    corrupt: dict[str, str] = {}
    worker_counts = {
        str(index): {"expected": 0, "complete": 0}
        for index in range(WORKER_COUNT)
    }
    for item in plan["jobs"]:
        job = ScreenJob.from_mapping(item)
        worker = str(int(item["worker_index"]))
        worker_counts[worker]["expected"] += 1
        path = _record_path(run_root, job)
        if not path.exists():
            missing.append(job.job_id)
            continue
        try:
            validate_record_payload(
                _strict_json_load(path),
                job=job,
                plan_sha256=str(plan["plan_sha256"]),
            )
        except Exception as error:
            corrupt[job.job_id] = str(error)
        else:
            complete.append(job.job_id)
            worker_counts[worker]["complete"] += 1
    return {
        "schema": "ieee-mi-opened-development-screen-status-v1",
        "evaluation_scope": EVALUATION_SCOPE,
        "plan_sha256": plan["plan_sha256"],
        "expected": len(plan["jobs"]),
        "complete": len(complete),
        "missing": len(missing),
        "corrupt": len(corrupt),
        "workers": worker_counts,
        "missing_job_ids": missing,
        "corrupt_records": corrupt,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="freeze the 136-job screen manifest")
    plan.add_argument("--run-root", type=Path, required=True)
    plan.add_argument("--cache-root", type=Path, required=True)

    status = subparsers.add_parser("status", help="audit atomic screen records")
    status.add_argument("--run-root", type=Path, required=True)

    worker = subparsers.add_parser("worker", help="run one static GPU partition")
    worker.add_argument("--run-root", type=Path, required=True)
    worker.add_argument("--cache-root", type=Path, required=True)
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
    _require_uv_venv()
    if args.command == "plan":
        manifest = build_manifest(args.cache_root)
        created = write_or_validate_plan(args.run_root.resolve(), manifest)
        print(
            json.dumps(
                {
                    "created": created,
                    "plan_sha256": manifest["plan_sha256"],
                    "n_jobs": manifest["n_jobs"],
                    "worker_count": manifest["worker_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "status":
        print(json.dumps(screen_status(args.run_root.resolve()), indent=2, sort_keys=True))
        return 0
    if args.command == "worker":
        return run_worker(
            run_root=args.run_root.resolve(),
            cache_root=args.cache_root.resolve(),
            worker_index=args.worker_index,
            gpu_uuid=args.gpu_uuid,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_MODELS",
    "CURATED_SPLITS",
    "EVALUATION_SCOPE",
    "PLAN_SCHEMA",
    "RECORD_SCHEMA",
    "SEED",
    "ScreenJob",
    "WORKER_COUNT",
    "assemble_manifest",
    "assert_validation_scope",
    "curated_jobs",
    "development_rows",
    "jobs_for_worker",
    "run_job_atomically",
    "screen_status",
    "validate_record_payload",
]
