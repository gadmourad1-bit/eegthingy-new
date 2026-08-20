"""Disjoint-subject robustness amendment for the killed CHSD v3 family.

This module defines, but does not automatically launch, a transparent
post-v3 exploratory amendment.  It evaluates one fixed candidate and three
fixed references on the 115 opened-development subjects that were absent from
both the v2 and v3 screens.  Every subject uses fold 0 and seed 7, yielding
exactly 460 aggregate validation-only jobs.

The immutable manifest binds the completed v2 plan, the completed v3 plan, the
v3 analysis artifact, its historical ``kill_conditioned_family`` decision,
the UV environment, the in-repository source closure, the pinned external
TCFormer checkout, every cache file, every train/validation split, and four
physical GPU UUIDs.  Workers never materialize or persist an excluded
partition.  Before each new job they fail closed on source/environment drift,
foreign GPU compute PIDs, or less than 50 GiB free.
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
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# NumPy and BLAS pools can initialize during imports.  Fix their shared-machine
# limits before importing NumPy, PyTorch, or any existing runner.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "4"

import numpy as np

from .conditioned_screen import (
    _validate_environment_identity,
    conditioned_environment_identity,
)
from .config import (
    CHANNEL_SCALING,
    DEFAULT_MONTAGE_PROFILE,
    LOCAL_EXP4_SUBJECT_RUNS,
    PREPROCESSING,
    dataset_spec,
    preprocessing_for_dataset,
)
from .data import PHYSIONET_MI_RUNS
from .development_screen import (
    GPU_UUID_RE,
    HEX_64_RE,
    METRIC_NAMES,
    MINIMUM_FREE_GIB,
    ScreenJob,
    _array_sha256,
    _file_sha256,
    _json_safe,
    _model_identity,
    _payload_sha256,
    _rows_sha256,
    _split_key,
    _state_sha256,
    _strict_json_load,
    _subject_key,
    _worker_preflight,
    assert_validation_scope,
    cooperative_gpu_guard,
    disk_guard,
)
from .development_screen import (
    load_plan as load_v2_plan,
)

PLAN_SCHEMA = "eeg-mi-chsd-disjoint-robustness-plan-v1"
RECORD_SCHEMA = "eeg-mi-chsd-disjoint-robustness-record-v1"
STATUS_SCHEMA = "eeg-mi-chsd-disjoint-robustness-status-v1"
EVALUATION_SCOPE = "opened_development_disjoint_subject_validation_only"
AMENDMENT_LABEL = "transparent_post_v3_exploratory_amendment"
SEED = 7
FOLD = 0
WORKER_COUNT = 4
CPU_THREADS = 4
EXPECTED_SUBJECT_COUNT = 115
EXPECTED_JOB_COUNT = 460
EXPECTED_JOBS_PER_WORKER = 115
GATE_TOLERANCE = 1e-12

EXPECTED_V2_PLAN_SHA256 = (
    "57298f7c18b8841e741d72194cb601faf3ac71cb81b1209b1c3964ddf43c0fc6"
)
EXPECTED_V3_PLAN_SHA256 = (
    "1359ea1913d76d624272f8194dfb361e5d919892783003e1f06e89d179f0e784"
)
EXPECTED_V3_ANALYSIS_ARTIFACT_SHA256 = (
    "69d616c2e2fe93a9b7f4464c31d8e3e8be048be39cb505eaea89023372135f0b"
)
EXPECTED_V2_PLAN_SCHEMA = "eeg-mi-opened-development-screen-plan-v1"
EXPECTED_V3_PLAN_SCHEMA = "eeg-mi-chsd-conditioned-development-screen-plan-v3"
EXPECTED_V3_ANALYSIS_SCHEMA = "eeg-mi-chsd-conditioned-screen-analysis-v3"
HISTORICAL_V3_DECISION = "kill_conditioned_family"
TCFORMER_EXPECTED_COMMIT = "74c89b7ab8c64e4eb51e0f748dd87dd4c94e68c5"

ROBUSTNESS_COHORTS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("local_exp4", (3, 5, 6, 8, 10)),
    ("bnci2014_001", (2, 3, 4, 6, 7, 8)),
    ("bnci2014_004", (2, 3, 4, 6, 7, 8)),
    (
        "cho2017",
        tuple(subject for subject in range(1, 53) if subject not in {1, 18, 35, 52}),
    ),
    (
        "physionet_mi",
        tuple(subject for subject in range(1, 55) if subject not in {1, 18, 36, 54}),
    ),
)
PRIOR_SCREEN_SUBJECTS: Mapping[str, tuple[int, ...]] = {
    "local_exp4": (1, 4, 7),
    "bnci2014_001": (1, 5, 9),
    "bnci2014_004": (1, 5, 9),
    "cho2017": (1, 18, 35, 52),
    "physionet_mi": (1, 18, 36, 54),
}

CANDIDATE_MODEL = "chsdnet_conditioned_005"
REFERENCE_MODELS: tuple[str, ...] = (
    "cardinal_fbc_micro_extended",
    "fbcnet",
    "tcformer",
)
MODELS: tuple[str, ...] = (CANDIDATE_MODEL, *REFERENCE_MODELS)
MODEL_FACTORIES: Mapping[str, str] = {
    CANDIDATE_MODEL: "benchmark.chsd_conditioned:make_chsd_conditioned_model",
    "cardinal_fbc_micro_extended": "benchmark.baselines:make_model",
    "fbcnet": "benchmark.baselines:make_model",
    "tcformer": "benchmark.baselines:make_model",
}

# Byte-for-byte equivalent values to the v3 common-recipe screen.  Keeping the
# values local makes this amendment independently reviewable and immutable.
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

# Frozen before any robustness result exists.  The analyzer imports this exact
# object and refuses a manifest with a different rule.
DECISION_RULE: Mapping[str, Any] = {
    "candidate": CANDIDATE_MODEL,
    "references": list(REFERENCE_MODELS),
    "primary_metric": "equal_dataset_balanced_accuracy",
    "maximum_gap_to_strongest_reference": 0.01,
    "must_strictly_exceed": "tcformer",
    "minimum_nonnegative_datasets_per_reference": 3,
    "maximum_dataset_deficit_per_reference": 0.05,
    "minimum_paired_subject_win_rate_per_reference": 0.45,
    "floating_point_tolerance": GATE_TOLERANCE,
    "pass_requires_every_condition": True,
    "failure_action": "stop_chsd_promotion",
}

# Complete in-repository execution closure, including imported helper modules.
SOURCE_FILES: tuple[str, ...] = (
    "src/benchmark/__init__.py",
    "src/benchmark/robustness_screen.py",
    "src/benchmark/robustness_analysis.py",
    "src/benchmark/conditioned_screen.py",
    "src/benchmark/development_screen.py",
    "src/benchmark/chsd_conditioned.py",
    "src/benchmark/chsd.py",
    "src/benchmark/baselines.py",
    "src/benchmark/models.py",
    "src/benchmark/training.py",
    "src/benchmark/data.py",
    "src/benchmark/config.py",
)
TCFORMER_RUNTIME_FILES: tuple[str, ...] = (
    "models/tcformer.py",
    "models/modules.py",
    "models/channel_group_attention.py",
    "utils/weight_initialization.py",
)

FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "labels",
        "targets",
        "ground_truth",
        "true_labels",
        "validation_labels",
        "y_true",
        "y_pred",
        "outcomes",
        "raw_outcomes",
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


class RobustnessScreenError(RuntimeError):
    """Raised when the robustness-amendment contract is violated."""


def robustness_environment_identity() -> dict[str, Any]:
    """Bind the complete installed UV environment without modifying it."""

    identity = dict(conditioned_environment_identity())
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RobustnessScreenError("UV is not available on PATH")
    try:
        result = subprocess.run(
            [uv_executable, "pip", "freeze", "--python", sys.executable],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RobustnessScreenError(
            f"cannot fingerprint the complete UV environment: {error}"
        ) from error
    freeze = tuple(
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    normalized = tuple(sorted(freeze, key=str.casefold))
    if not normalized or len(normalized) != len(set(normalized)):
        raise RobustnessScreenError(
            "UV package freeze is empty or contains duplicate entries"
        )
    if not any(
        re.match(r"(?i)^einops(?:==| @ )", requirement)
        for requirement in normalized
    ):
        raise RobustnessScreenError(
            "complete UV freeze omits TCFormer's required einops dependency"
        )
    identity.update(
        {
            "uv_pip_freeze": list(normalized),
            "uv_pip_freeze_sha256": hashlib.sha256(
                "\n".join(normalized).encode("utf-8")
            ).hexdigest(),
        }
    )
    return identity


def _validate_robustness_environment_identity(
    identity: Mapping[str, Any],
) -> None:
    _validate_environment_identity(identity)
    inherited = {
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
    if set(identity) != inherited | {
        "uv_pip_freeze",
        "uv_pip_freeze_sha256",
    }:
        raise ValueError("robustness environment identity fields changed")
    freeze = identity["uv_pip_freeze"]
    if (
        not isinstance(freeze, list)
        or not freeze
        or any(not isinstance(item, str) or not item for item in freeze)
        or freeze != sorted(freeze, key=str.casefold)
        or len(freeze) != len(set(freeze))
    ):
        raise ValueError("robustness UV package freeze is invalid")
    digest = hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest()
    if identity["uv_pip_freeze_sha256"] != digest:
        raise ValueError("robustness UV package freeze digest differs")
    if not any(
        re.match(r"(?i)^einops(?:==| @ )", requirement)
        for requirement in freeze
    ):
        raise ValueError("robustness UV freeze omits einops")


def _subject_count() -> int:
    return sum(len(subjects) for _, subjects in ROBUSTNESS_COHORTS)


def robustness_jobs() -> tuple[ScreenJob, ...]:
    """Return the fixed model-major 460-job order."""

    jobs = tuple(
        ScreenJob(
            dataset=dataset,
            model=model,
            subject=subject,
            fold=FOLD,
            seed=SEED,
        )
        for model in MODELS
        for dataset, subjects in ROBUSTNESS_COHORTS
        for subject in subjects
    )
    if _subject_count() != EXPECTED_SUBJECT_COUNT:
        raise AssertionError("robustness subject cardinality changed")
    if len(jobs) != EXPECTED_JOB_COUNT:
        raise AssertionError("robustness job cardinality changed")
    return jobs


def jobs_for_worker(
    jobs: Sequence[ScreenJob],
    worker_index: int,
    *,
    worker_count: int = WORKER_COUNT,
) -> tuple[ScreenJob, ...]:
    """Return the one-model static partition for one physical GPU worker."""

    if worker_count != WORKER_COUNT:
        raise ValueError(f"worker_count is fixed at {WORKER_COUNT}")
    if not 0 <= worker_index < WORKER_COUNT:
        raise ValueError(f"worker_index must lie in [0, {WORKER_COUNT - 1}]")
    expected_model = MODELS[worker_index]
    partition = tuple(job for job in jobs if job.model == expected_model)
    if len(partition) != EXPECTED_JOBS_PER_WORKER:
        raise ValueError("worker partition does not contain exactly 115 jobs")
    return partition


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


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


def _cache_path(cache_root: Path, dataset: str, subject: int) -> Path:
    return (
        cache_root
        / str(PREPROCESSING["schema"])
        / DEFAULT_MONTAGE_PROFILE
        / dataset
        / f"subject_{subject:03d}.npz"
    )


def _expected_subject_keys() -> set[str]:
    return {
        _subject_key(dataset, subject)
        for dataset, subjects in ROBUSTNESS_COHORTS
        for subject in subjects
    }


def _expected_split_keys() -> set[str]:
    return {
        _split_key(dataset, subject, FOLD)
        for dataset, subjects in ROBUSTNESS_COHORTS
        for subject in subjects
    }


def _validate_cohorts() -> None:
    if _subject_count() != EXPECTED_SUBJECT_COUNT:
        raise ValueError("robustness cohort does not contain exactly 115 subjects")
    expected_datasets = (
        "local_exp4",
        "bnci2014_001",
        "bnci2014_004",
        "cho2017",
        "physionet_mi",
    )
    if tuple(dataset for dataset, _ in ROBUSTNESS_COHORTS) != expected_datasets:
        raise ValueError("robustness dataset order changed")
    for dataset, subjects in ROBUSTNESS_COHORTS:
        if not subjects or len(subjects) != len(set(subjects)):
            raise ValueError(f"{dataset} robustness subjects are empty or duplicated")
        if set(subjects) & set(PRIOR_SCREEN_SUBJECTS[dataset]):
            raise ValueError(f"{dataset} robustness subjects overlap v2/v3")
        if set(subjects) | set(PRIOR_SCREEN_SUBJECTS[dataset]) != set(
            dataset_spec(dataset).development_subjects
        ):
            raise ValueError(
                f"{dataset} robustness and prior-screen subjects do not exactly "
                "partition the opened development cohort"
            )


def robustness_rows(
    dataset: str,
    subject: int,
    sessions: np.ndarray,
    runs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive fold-0 training and validation rows from metadata only.

    The Cho cache is class-contiguous by contract.  Its two equal-length class
    sequences are split into five contiguous blocks without reading labels.
    This supports both the usual 200-trial cache and the 240-trial caches for
    S7, S9, and S46 while matching the formal fold-0 split.
    """

    subjects = dict(ROBUSTNESS_COHORTS).get(dataset)
    if subjects is None or subject not in subjects:
        raise ValueError(f"{dataset} S{subject} is outside the robustness cohort")
    sessions = np.asarray(sessions).astype(str)
    runs = np.asarray(runs).astype(str)
    if sessions.ndim != 1 or runs.ndim != 1 or len(sessions) != len(runs):
        raise ValueError("sessions and runs must be equal-length vectors")
    rows = np.arange(len(sessions), dtype=np.int64)

    if dataset == "local_exp4":
        expected = tuple(str(value) for value in LOCAL_EXP4_SUBJECT_RUNS[subject])
        if set(runs.tolist()) != set(expected):
            raise RobustnessScreenError(
                f"local_exp4 S{subject} run identity differs from the cache contract"
            )
        train = rows[np.isin(runs, expected[:2])]
        validation = rows[runs == expected[2]]
    elif dataset == "bnci2014_001":
        source = rows[sessions == "0train"]
        source_runs = sorted(set(runs[source].tolist()))
        if len(source_runs) < 2:
            raise RobustnessScreenError("BNCI2014-001 source session has too few runs")
        validation_run = source_runs[-1]
        train = source[runs[source] != validation_run]
        validation = source[runs[source] == validation_run]
    elif dataset == "bnci2014_004":
        train = rows[np.isin(sessions, ("0train", "1train"))]
        validation = rows[sessions == "2train"]
    elif dataset == "cho2017":
        if len(rows) not in {200, 240} or len(rows) % 2:
            raise RobustnessScreenError(
                "Cho2017 robustness cache must contain 200 or 240 trials"
            )
        class_size = len(rows) // 2
        if class_size % 5:
            raise RobustnessScreenError(
                "Cho2017 class sequence is not divisible into five blocks"
            )
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
        if set(runs.tolist()) != set(PHYSIONET_MI_RUNS):
            raise RobustnessScreenError(
                "PhysioNet robustness cache must contain imagery runs 4/8/12"
            )
        train = rows[runs == PHYSIONET_MI_RUNS[2]]
        validation = rows[runs == PHYSIONET_MI_RUNS[1]]
    else:  # pragma: no cover - guarded by the fixed cohort map
        raise ValueError(dataset)

    train = np.sort(np.asarray(train, dtype=np.int64))
    validation = np.sort(np.asarray(validation, dtype=np.int64))
    if not len(train) or not len(validation):
        raise RobustnessScreenError(f"{dataset} S{subject} has an empty partition")
    if np.intersect1d(train, validation).size:
        raise RobustnessScreenError("training and validation rows overlap")
    if train[0] < 0 or validation[0] < 0:
        raise RobustnessScreenError("partition rows must be nonnegative")
    if train[-1] >= len(rows) or validation[-1] >= len(rows):
        raise RobustnessScreenError("partition rows exceed the cache")
    return train, validation


def _read_cache_plan_identity(
    cache_root: Path,
    dataset: str,
    subject: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read metadata only and freeze one v2 harmonized cache/split binding."""

    path = _cache_path(cache_root, dataset, subject)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_relative = (
        Path(str(PREPROCESSING["schema"]))
        / DEFAULT_MONTAGE_PROFILE
        / dataset
        / f"subject_{subject:03d}.npz"
    )
    if path.relative_to(cache_root) != expected_relative:
        raise RobustnessScreenError("cache path is outside the v2 harmonized layout")
    with np.load(path, allow_pickle=False) as archive:
        required = {"sessions", "runs", "identity"}
        if not required.issubset(archive.files):
            raise RobustnessScreenError(f"cache metadata is incomplete: {path}")
        sessions = archive["sessions"].astype(str)
        runs = archive["runs"].astype(str)
        identity = json.loads(str(archive["identity"].item()))

    dataset_identity = identity.get("dataset")
    if (
        not isinstance(dataset_identity, Mapping)
        or dataset_identity.get("key") != dataset
        or int(identity.get("subject", -1)) != subject
        or identity.get("montage_profile") != DEFAULT_MONTAGE_PROFILE
        or identity.get("preprocessing") != preprocessing_for_dataset(dataset)
    ):
        raise RobustnessScreenError(
            f"cache identity does not match {dataset} S{subject}"
        )
    shape = identity.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or int(shape[0]) != len(sessions)
        or len(sessions) != len(runs)
    ):
        raise RobustnessScreenError("cache metadata row count is invalid")
    array_digest = identity.get("array_sha256")
    if not isinstance(array_digest, str) or not HEX_64_RE.fullmatch(array_digest):
        raise RobustnessScreenError("cache array SHA-256 is invalid")
    train, validation = robustness_rows(dataset, subject, sessions, runs)
    return (
        {
            "relative_path": str(expected_relative),
            "file_size_bytes": path.stat().st_size,
            "file_sha256": _file_sha256(path),
            "array_sha256": array_digest,
        },
        {
            "train": {
                "count": len(train),
                "rows_sha256": _rows_sha256(train),
            },
            "validation": {
                "count": len(validation),
                "rows_sha256": _rows_sha256(validation),
            },
        },
    )


def _read_selected_npy_rows(
    npz_path: Path,
    member: str,
    rows: np.ndarray,
    *,
    expected_dtype: np.dtype[Any],
    expected_ndim: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Decode only selected C-order rows from one compressed NPY member.

    Indexing ``numpy.load(npz)["x"]`` materializes the complete member before
    slicing.  This reader parses the NPY header, seeks over excluded payloads
    in the ZIP stream, and allocates arrays only for the permitted rows.
    """

    selected = np.asarray(rows, dtype=np.int64)
    if selected.ndim != 1 or not len(selected):
        raise RobustnessScreenError("selected NPY rows must be a nonempty vector")
    if len(np.unique(selected)) != len(selected):
        raise RobustnessScreenError("selected NPY rows must be unique")
    order = np.argsort(selected)
    sorted_rows = selected[order]
    try:
        with zipfile.ZipFile(npz_path, "r") as archive:
            with archive.open(f"{member}.npy", "r") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_1_0(stream)
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_2_0(stream)
                    )
                elif version == (3, 0):
                    shape, fortran_order, dtype = (
                        np.lib.format._read_array_header(stream, version)  # type: ignore[attr-defined]  # noqa: SLF001
                    )
                else:
                    raise RobustnessScreenError(
                        f"unsupported NPY version {version} for {member}"
                    )
                shape = tuple(int(value) for value in shape)
                dtype = np.dtype(dtype)
                if (
                    len(shape) != expected_ndim
                    or fortran_order
                    or dtype != np.dtype(expected_dtype)
                    or dtype.hasobject
                    or int(sorted_rows[0]) < 0
                    or shape[0] <= int(sorted_rows[-1])
                ):
                    raise RobustnessScreenError(
                        f"{member} NPY layout differs from the cache contract"
                    )
                row_shape = shape[1:]
                row_elements = math.prod(row_shape) if row_shape else 1
                row_bytes = row_elements * dtype.itemsize
                sorted_result = np.empty(
                    (len(sorted_rows), *row_shape),
                    dtype=dtype,
                )
                cursor = 0
                for output_index, row in enumerate(sorted_rows.tolist()):
                    skip = (row - cursor) * row_bytes
                    if skip:
                        stream.seek(skip, os.SEEK_CUR)
                    payload = stream.read(row_bytes)
                    if len(payload) != row_bytes:
                        raise RobustnessScreenError(
                            f"{member} NPY payload is truncated"
                        )
                    sorted_result[output_index] = np.frombuffer(
                        payload,
                        dtype=dtype,
                        count=row_elements,
                    ).reshape(row_shape)
                    cursor = row + 1
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise RobustnessScreenError(
            f"cannot stream selected {member} rows from {npz_path}: {error}"
        ) from error
    result = np.empty_like(sorted_result)
    result[order] = sorted_result
    return result, shape


def source_identity(project_root: Path | None = None) -> dict[str, str]:
    """Hash the explicit in-repository runtime dependency closure."""

    root = Path(__file__).resolve().parents[2] if project_root is None else project_root
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = _file_sha256(path)
    return result


def tcformer_source_identity(root: Path | None = None) -> dict[str, Any]:
    """Bind the exact clean official TCFormer checkout used by the worker."""

    checkout = (
        Path(
            os.environ.get(
                "EEG_MI_TCFORMER_ROOT",
                str(Path.cwd() / "third_party" / "TCFormer"),
            )
        )
        if root is None
        else Path(root)
    ).resolve()
    for relative in TCFORMER_RUNTIME_FILES:
        if not (checkout / relative).is_file():
            raise FileNotFoundError(checkout / relative)
    try:
        commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RobustnessScreenError(
            f"cannot verify the official TCFormer checkout: {error}"
        ) from error
    if commit != TCFORMER_EXPECTED_COMMIT:
        raise RobustnessScreenError(
            f"TCFormer is at {commit}, expected {TCFORMER_EXPECTED_COMMIT}"
        )
    if dirty:
        raise RobustnessScreenError("TCFormer has tracked modifications")
    return {
        "repository": str(checkout),
        "commit": commit,
        "tracked_dirty": False,
        "runtime_file_sha256": {
            relative: _file_sha256(checkout / relative)
            for relative in TCFORMER_RUNTIME_FILES
        },
    }


def _validate_external_source_identity(identity: Mapping[str, Any]) -> None:
    if set(identity) != {
        "repository",
        "commit",
        "tracked_dirty",
        "runtime_file_sha256",
    }:
        raise ValueError("TCFormer source identity fields changed")
    if (
        not isinstance(identity["repository"], str)
        or not identity["repository"]
        or identity["commit"] != TCFORMER_EXPECTED_COMMIT
        or identity["tracked_dirty"] is not False
    ):
        raise ValueError("TCFormer repository identity is invalid")
    files = identity["runtime_file_sha256"]
    if not isinstance(files, Mapping) or set(files) != set(TCFORMER_RUNTIME_FILES):
        raise ValueError("TCFormer runtime source closure changed")
    if any(
        not isinstance(digest, str) or not HEX_64_RE.fullmatch(digest)
        for digest in files.values()
    ):
        raise ValueError("TCFormer runtime source hash is invalid")


def _validate_gpu_uuids(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(result) != WORKER_COUNT:
        raise ValueError("exactly four physical GPU UUIDs are required")
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
        raise ValueError("cache identities differ from the fixed 115 subjects")
    if set(split_identity) != _expected_split_keys():
        raise ValueError("split identities differ from the fixed 115 subjects")
    expected_relative_paths = {
        _subject_key(dataset, subject): (
            Path(str(PREPROCESSING["schema"]))
            / DEFAULT_MONTAGE_PROFILE
            / dataset
            / f"subject_{subject:03d}.npz"
        )
        for dataset, subjects in ROBUSTNESS_COHORTS
        for subject in subjects
    }
    for key, value in cache_identity.items():
        if set(value) != {
            "relative_path",
            "file_size_bytes",
            "file_sha256",
            "array_sha256",
        }:
            raise ValueError(f"cache identity {key} has unexpected fields")
        relative = Path(str(value["relative_path"]))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative != expected_relative_paths[key]
        ):
            raise ValueError(
                f"cache identity {key} is not its exact v2/harmonized path"
            )
        if int(value["file_size_bytes"]) <= 0:
            raise ValueError(f"cache identity {key} has an invalid size")
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


def _lineage_binding(
    *,
    reference_v2_run_root: Path,
    v3_run_root: Path,
    v3_analysis_path: Path,
) -> dict[str, Any]:
    """Verify and freeze the historical v2/v3 artifacts without rewriting them."""

    reference_root = reference_v2_run_root.resolve()
    conditioned_root = v3_run_root.resolve()
    analysis_path = v3_analysis_path.resolve()
    v2_plan = load_v2_plan(reference_root)
    if v2_plan["plan_sha256"] != EXPECTED_V2_PLAN_SHA256:
        raise RobustnessScreenError("v2 reference plan SHA-256 differs")

    from .conditioned_screen import load_plan as load_v3_plan

    v3_plan = load_v3_plan(conditioned_root)
    if (
        v3_plan["schema"] != EXPECTED_V3_PLAN_SCHEMA
        or v3_plan["plan_sha256"] != EXPECTED_V3_PLAN_SHA256
    ):
        raise RobustnessScreenError("completed v3 plan identity differs")
    if (
        v3_plan.get("reference_binding", {}).get("plan_sha256")
        != EXPECTED_V2_PLAN_SHA256
    ):
        raise RobustnessScreenError("v3 plan is not bound to the expected v2 plan")
    if not analysis_path.is_file():
        raise FileNotFoundError(analysis_path)
    artifact_digest = _file_sha256(analysis_path)
    if artifact_digest != EXPECTED_V3_ANALYSIS_ARTIFACT_SHA256:
        raise RobustnessScreenError("v3 analysis artifact SHA-256 differs")
    if analysis_path.is_symlink():
        raise RobustnessScreenError("v3 analysis artifact must not be a symlink")
    analysis = _strict_json_load(analysis_path)
    selection = analysis.get("selection")
    if (
        analysis.get("schema") != EXPECTED_V3_ANALYSIS_SCHEMA
        or analysis.get("conditioned_plan_sha256") != EXPECTED_V3_PLAN_SHA256
        or analysis.get("reference_plan_sha256") != EXPECTED_V2_PLAN_SHA256
        or not isinstance(selection, Mapping)
        or selection.get("decision") != HISTORICAL_V3_DECISION
        or selection.get("selected_model") is not None
    ):
        raise RobustnessScreenError(
            "v3 analysis does not preserve the historical kill decision"
        )
    return {
        "amendment_label": AMENDMENT_LABEL,
        "prior_decision_is_rewritten": False,
        "reference_v2": {
            "run_root": str(reference_root),
            "plan_schema": v2_plan["schema"],
            "plan_sha256": EXPECTED_V2_PLAN_SHA256,
        },
        "conditioned_v3": {
            "run_root": str(conditioned_root),
            "plan_schema": EXPECTED_V3_PLAN_SCHEMA,
            "plan_sha256": EXPECTED_V3_PLAN_SHA256,
        },
        "conditioned_v3_analysis": {
            "artifact_path": str(analysis_path),
            "analysis_schema": EXPECTED_V3_ANALYSIS_SCHEMA,
            "artifact_sha256": EXPECTED_V3_ANALYSIS_ARTIFACT_SHA256,
            "decision": HISTORICAL_V3_DECISION,
            "selected_model": None,
        },
    }


def _validate_lineage_binding(lineage: Mapping[str, Any]) -> None:
    if set(lineage) != {
        "amendment_label",
        "prior_decision_is_rewritten",
        "reference_v2",
        "conditioned_v3",
        "conditioned_v3_analysis",
    }:
        raise ValueError("robustness lineage fields changed")
    if (
        lineage["amendment_label"] != AMENDMENT_LABEL
        or lineage["prior_decision_is_rewritten"] is not False
    ):
        raise ValueError("robustness amendment label/history semantics changed")
    v2 = lineage["reference_v2"]
    v3 = lineage["conditioned_v3"]
    analysis = lineage["conditioned_v3_analysis"]
    if (
        not isinstance(v2, Mapping)
        or set(v2) != {"run_root", "plan_schema", "plan_sha256"}
        or v2["plan_schema"] != EXPECTED_V2_PLAN_SCHEMA
        or v2["plan_sha256"] != EXPECTED_V2_PLAN_SHA256
        or not isinstance(v2["run_root"], str)
        or not v2["run_root"]
    ):
        raise ValueError("v2 lineage binding is invalid")
    if (
        not isinstance(v3, Mapping)
        or set(v3) != {"run_root", "plan_schema", "plan_sha256"}
        or v3["plan_schema"] != EXPECTED_V3_PLAN_SCHEMA
        or v3["plan_sha256"] != EXPECTED_V3_PLAN_SHA256
        or not isinstance(v3["run_root"], str)
        or not v3["run_root"]
    ):
        raise ValueError("v3 lineage binding is invalid")
    if (
        not isinstance(analysis, Mapping)
        or set(analysis)
        != {
            "artifact_path",
            "analysis_schema",
            "artifact_sha256",
            "decision",
            "selected_model",
        }
        or analysis["analysis_schema"] != EXPECTED_V3_ANALYSIS_SCHEMA
        or analysis["artifact_sha256"] != EXPECTED_V3_ANALYSIS_ARTIFACT_SHA256
        or analysis["decision"] != HISTORICAL_V3_DECISION
        or analysis["selected_model"] is not None
        or not isinstance(analysis["artifact_path"], str)
        or not analysis["artifact_path"]
    ):
        raise ValueError("v3 analysis lineage binding is invalid")


def assemble_manifest(
    *,
    cache_root: str,
    cache_identity: Mapping[str, Mapping[str, Any]],
    split_identity: Mapping[str, Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    environment_identity: Mapping[str, Any],
    tcformer_identity: Mapping[str, Any],
    worker_gpu_uuids: Sequence[str],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the exact immutable 460-job exploratory amendment."""

    _validate_cohorts()
    _validate_identity_maps(cache_identity, split_identity)
    _validate_lineage_binding(lineage)
    gpu_uuids = _validate_gpu_uuids(worker_gpu_uuids)
    if set(source_hashes) != set(SOURCE_FILES):
        raise ValueError("source identity does not cover the fixed closure")
    if any(
        not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
        for value in source_hashes.values()
    ):
        raise ValueError("source identity contains an invalid SHA-256")
    _validate_robustness_environment_identity(environment_identity)
    _validate_external_source_identity(tcformer_identity)

    jobs = robustness_jobs()
    worker_by_model = {model: index for index, model in enumerate(MODELS)}
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "purpose": "disjoint_subject_robustness_exploratory_amendment",
        "amendment_label": AMENDMENT_LABEL,
        "evaluation_scope": EVALUATION_SCOPE,
        "confirmation_evidence": False,
        "fold": FOLD,
        "seed": SEED,
        "worker_count": WORKER_COUNT,
        "jobs_per_worker": EXPECTED_JOBS_PER_WORKER,
        "cache_root": str(cache_root),
        "robustness_cohorts": [
            {"dataset": dataset, "subjects": list(subjects), "fold": FOLD}
            for dataset, subjects in ROBUSTNESS_COHORTS
        ],
        "prior_screen_subjects": {
            dataset: list(subjects)
            for dataset, subjects in PRIOR_SCREEN_SUBJECTS.items()
        },
        "models": [
            {
                "name": model,
                "role": "candidate" if model == CANDIDATE_MODEL else "reference",
                "factory": MODEL_FACTORIES[model],
                "expected_source_statistics_hook": False,
            }
            for model in MODELS
        ],
        "train_config": dict(TRAIN_CONFIG),
        "channel_scaling": dict(CHANNEL_SCALING),
        "decision_rule": dict(DECISION_RULE),
        "cache_identity": {
            key: dict(value) for key, value in sorted(cache_identity.items())
        },
        "split_identity": {
            key: dict(value) for key, value in sorted(split_identity.items())
        },
        "source_identity": dict(sorted(source_hashes.items())),
        "environment_identity": dict(environment_identity),
        "external_source_identity": {
            "tcformer": {
                **dict(tcformer_identity),
                "runtime_file_sha256": dict(tcformer_identity["runtime_file_sha256"]),
            }
        },
        "lineage": {
            **dict(lineage),
            "reference_v2": dict(lineage["reference_v2"]),
            "conditioned_v3": dict(lineage["conditioned_v3"]),
            "conditioned_v3_analysis": dict(lineage["conditioned_v3_analysis"]),
        },
        "worker_bindings": [
            {
                "worker_index": index,
                "model": MODELS[index],
                "gpu_uuid": gpu_uuids[index],
                "job_count": EXPECTED_JOBS_PER_WORKER,
                "cpu_threads": CPU_THREADS,
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
        "n_subjects": EXPECTED_SUBJECT_COUNT,
        "n_jobs": EXPECTED_JOB_COUNT,
        "artifact_contract": {
            "metrics": "aggregate_validation_only",
            "raw_outcomes_persisted": False,
            "raw_trials_persisted": False,
            "excluded_rows_persisted": False,
            "model_state_artifacts_persisted": False,
            "epoch_traces_persisted": False,
            "atomic_unit": "one_json_per_subject_model",
        },
    }
    assert_validation_scope(payload)
    payload["plan_sha256"] = _payload_sha256(payload, "plan_sha256")
    validate_manifest_payload(payload)
    return payload


def build_manifest(
    *,
    cache_root: Path,
    reference_v2_run_root: Path,
    v3_run_root: Path,
    v3_analysis_path: Path,
    worker_gpu_uuids: Sequence[str],
) -> dict[str, Any]:
    """Read metadata/provenance and bind the complete robustness manifest."""

    cache_root = cache_root.resolve()
    cache_identities: dict[str, dict[str, Any]] = {}
    split_identities: dict[str, dict[str, Any]] = {}
    for dataset, subjects in ROBUSTNESS_COHORTS:
        for subject in subjects:
            cache, split = _read_cache_plan_identity(cache_root, dataset, subject)
            cache_identities[_subject_key(dataset, subject)] = cache
            split_identities[_split_key(dataset, subject, FOLD)] = split

    gpu_uuids = _validate_gpu_uuids(worker_gpu_uuids)
    for gpu_uuid in gpu_uuids:
        cooperative_gpu_guard(gpu_uuid)
    disk_guard(cache_root)
    return assemble_manifest(
        cache_root=str(cache_root),
        cache_identity=cache_identities,
        split_identity=split_identities,
        source_hashes=source_identity(),
        environment_identity=robustness_environment_identity(),
        tcformer_identity=tcformer_source_identity(),
        worker_gpu_uuids=gpu_uuids,
        lineage=_lineage_binding(
            reference_v2_run_root=reference_v2_run_root,
            v3_run_root=v3_run_root,
            v3_analysis_path=v3_analysis_path,
        ),
    )


def validate_manifest_payload(plan: Mapping[str, Any]) -> None:
    """Fail closed unless ``plan`` is the exact frozen amendment."""

    expected_top = {
        "schema",
        "purpose",
        "amendment_label",
        "evaluation_scope",
        "confirmation_evidence",
        "fold",
        "seed",
        "worker_count",
        "jobs_per_worker",
        "cache_root",
        "robustness_cohorts",
        "prior_screen_subjects",
        "models",
        "train_config",
        "channel_scaling",
        "decision_rule",
        "cache_identity",
        "split_identity",
        "source_identity",
        "environment_identity",
        "external_source_identity",
        "lineage",
        "worker_bindings",
        "jobs",
        "n_subjects",
        "n_jobs",
        "artifact_contract",
        "plan_sha256",
    }
    if set(plan) != expected_top:
        raise RobustnessScreenError("robustness plan fields differ from v1")
    if (
        plan["schema"] != PLAN_SCHEMA
        or plan["purpose"] != "disjoint_subject_robustness_exploratory_amendment"
        or plan["amendment_label"] != AMENDMENT_LABEL
        or plan["evaluation_scope"] != EVALUATION_SCOPE
        or plan["confirmation_evidence"] is not False
        or int(plan["fold"]) != FOLD
        or int(plan["seed"]) != SEED
        or int(plan["worker_count"]) != WORKER_COUNT
        or int(plan["jobs_per_worker"]) != EXPECTED_JOBS_PER_WORKER
        or int(plan["n_subjects"]) != EXPECTED_SUBJECT_COUNT
        or int(plan["n_jobs"]) != EXPECTED_JOB_COUNT
    ):
        raise RobustnessScreenError("robustness fixed plan identity changed")
    if plan["train_config"] != dict(TRAIN_CONFIG):
        raise RobustnessScreenError("robustness training recipe changed")
    if plan["channel_scaling"] != dict(CHANNEL_SCALING):
        raise RobustnessScreenError("robustness scaling contract changed")
    if plan["decision_rule"] != dict(DECISION_RULE):
        raise RobustnessScreenError("robustness decision rule changed")
    expected_cohorts = [
        {"dataset": dataset, "subjects": list(subjects), "fold": FOLD}
        for dataset, subjects in ROBUSTNESS_COHORTS
    ]
    if plan["robustness_cohorts"] != expected_cohorts:
        raise RobustnessScreenError("robustness cohort changed")
    if plan["prior_screen_subjects"] != {
        dataset: list(subjects) for dataset, subjects in PRIOR_SCREEN_SUBJECTS.items()
    }:
        raise RobustnessScreenError("prior-screen exclusions changed")
    _validate_cohorts()
    _validate_identity_maps(plan["cache_identity"], plan["split_identity"])
    _validate_robustness_environment_identity(plan["environment_identity"])
    if set(plan["source_identity"]) != set(SOURCE_FILES):
        raise RobustnessScreenError("robustness source closure changed")
    if any(
        not isinstance(value, str) or not HEX_64_RE.fullmatch(value)
        for value in plan["source_identity"].values()
    ):
        raise RobustnessScreenError("robustness source hash is invalid")
    external = plan["external_source_identity"]
    if not isinstance(external, Mapping) or set(external) != {"tcformer"}:
        raise RobustnessScreenError("external source set changed")
    _validate_external_source_identity(external["tcformer"])
    _validate_lineage_binding(plan["lineage"])
    if plan["plan_sha256"] != _payload_sha256(plan, "plan_sha256"):
        raise RobustnessScreenError("robustness plan SHA-256 mismatch")

    expected_models = [
        {
            "name": model,
            "role": "candidate" if model == CANDIDATE_MODEL else "reference",
            "factory": MODEL_FACTORIES[model],
            "expected_source_statistics_hook": False,
        }
        for model in MODELS
    ]
    if plan["models"] != expected_models:
        raise RobustnessScreenError("robustness model roster changed")
    gpu_uuids = _validate_gpu_uuids(
        [str(item["gpu_uuid"]) for item in plan["worker_bindings"]]
    )
    for index, item in enumerate(plan["worker_bindings"]):
        if item != {
            "worker_index": index,
            "model": MODELS[index],
            "gpu_uuid": gpu_uuids[index],
            "job_count": EXPECTED_JOBS_PER_WORKER,
            "cpu_threads": CPU_THREADS,
        }:
            raise RobustnessScreenError("robustness worker binding changed")
    expected_jobs = robustness_jobs()
    if not isinstance(plan["jobs"], list) or len(plan["jobs"]) != len(expected_jobs):
        raise RobustnessScreenError("robustness plan job count changed")
    for expected, item in zip(expected_jobs, plan["jobs"], strict=True):
        if item != {
            **expected.identity(),
            "job_id": expected.job_id,
            "worker_index": MODELS.index(expected.model),
        }:
            raise RobustnessScreenError("robustness plan job order changed")
    expected_artifact_contract = {
        "metrics": "aggregate_validation_only",
        "raw_outcomes_persisted": False,
        "raw_trials_persisted": False,
        "excluded_rows_persisted": False,
        "model_state_artifacts_persisted": False,
        "epoch_traces_persisted": False,
        "atomic_unit": "one_json_per_subject_model",
    }
    if plan["artifact_contract"] != expected_artifact_contract:
        raise RobustnessScreenError("robustness artifact contract changed")
    assert_validation_scope(plan)


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> bool:
    """Create one JSON artifact atomically without replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
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


def _strict_regular_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RobustnessScreenError(f"JSON artifact is not a regular file: {path}")
    return _strict_json_load(path)


def write_or_validate_plan(run_root: Path, plan: Mapping[str, Any]) -> bool:
    validate_manifest_payload(plan)
    path = run_root / "plan.json"
    if _atomic_create_json(path, plan):
        return True
    existing = _strict_regular_json(path)
    validate_manifest_payload(existing)
    if existing != dict(plan):
        raise RobustnessScreenError(
            "run root contains a different immutable robustness plan"
        )
    return False


def load_plan(run_root: Path) -> dict[str, Any]:
    plan = _strict_regular_json(run_root / "plan.json")
    validate_manifest_payload(plan)
    return plan


def _assert_no_excluded_evidence(value: Any, path: tuple[str, ...] = ()) -> None:
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
    """Validate one aggregate-only robustness record."""

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
        "environment_identity_sha256",
        "split",
        "scaler",
        "model",
        "fit",
        "validation_metrics",
        "timing",
        "resource_preflight",
    }
    if set(payload) != expected_keys:
        raise ValueError("robustness record fields differ from v1")
    if payload["schema"] != RECORD_SCHEMA:
        raise ValueError("robustness record schema mismatch")
    if payload["plan_sha256"] != plan_sha256:
        raise ValueError("robustness record plan hash mismatch")
    if payload["job_id"] != job.job_id or payload["job"] != job.identity():
        raise ValueError("robustness record job identity mismatch")
    if (
        payload["evaluation_scope"] != EVALUATION_SCOPE
        or payload["confirmation_evidence"] is not False
    ):
        raise ValueError("robustness record evidence scope changed")
    if not isinstance(payload["created_at"], str) or not payload["created_at"]:
        raise ValueError("robustness record timestamp is invalid")
    split = payload["split"]
    if not isinstance(split, Mapping) or set(split) != {"train", "validation"}:
        raise ValueError("robustness record split exceeds validation scope")
    metrics = payload["validation_metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != METRIC_NAMES:
        raise ValueError("robustness validation metric set changed")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"robustness validation metric {name} is invalid")
    model = payload["model"]
    if (
        not isinstance(model, Mapping)
        or set(model)
        != {
            "identity",
            "parameter_count",
            "constructed_state_sha256",
            "initial_state_sha256",
            "selected_state_sha256",
            "source_statistics",
        }
        or int(model["parameter_count"]) <= 0
        or any(
            not isinstance(model[name], str)
            or not HEX_64_RE.fullmatch(model[name])
            for name in (
                "constructed_state_sha256",
                "initial_state_sha256",
                "selected_state_sha256",
            )
        )
        or model["constructed_state_sha256"] != model["initial_state_sha256"]
    ):
        raise TypeError("robustness model identity is missing")
    identity = model["identity"]
    if (
        not isinstance(identity, Mapping)
        or not {"requested_name", "class", "uses_positions"}.issubset(identity)
        or not set(identity).issubset(
            {
                "requested_name",
                "class",
                "uses_positions",
                "config",
                "third_party_provenance",
            }
        )
    ):
        raise ValueError("robustness model metadata fields are invalid")
    hook = model.get("source_statistics")
    if not isinstance(hook, Mapping) or set(hook) != {
        "available",
        "ran",
        "fit_scope",
        "outcome_aware",
    }:
        raise ValueError("robustness source-statistics identity is invalid")
    if (
        hook["available"] is not False
        or hook["ran"] is not False
        or hook["fit_scope"] != "training_rows_only"
        or hook["outcome_aware"] is not False
    ):
        raise ValueError("robustness source-statistics semantics changed")
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
        raise ValueError("robustness record cache identity differs from plan")
    if payload["source_identity"] != plan["source_identity"]:
        raise ValueError("robustness record source identity differs from plan")
    if payload["environment_identity_sha256"] != _mapping_sha256(
        plan["environment_identity"]
    ):
        raise ValueError("robustness record environment identity differs from plan")
    if payload["split"] != plan["split_identity"][split_key]:
        raise ValueError("robustness record split identity differs from plan")
    if payload["fit"].get("config") != plan["train_config"]:
        raise ValueError("robustness record training recipe differs from plan")
    scaler = payload.get("scaler")
    if (
        not isinstance(scaler, Mapping)
        or set(scaler) != {"fit_scope", "mean_sha256", "std_sha256", "contract"}
        or scaler["fit_scope"] != "training_only"
        or scaler["contract"] != plan["channel_scaling"]
        or any(
            not isinstance(scaler[name], str) or not HEX_64_RE.fullmatch(scaler[name])
            for name in ("mean_sha256", "std_sha256")
        )
    ):
        raise ValueError("robustness record scaler is not source-only")
    identity = payload["model"].get("identity")
    if not isinstance(identity, Mapping) or identity.get("requested_name") != job.model:
        raise ValueError("robustness record requested model differs")
    provenance = identity.get("third_party_provenance")
    if job.model == "tcformer":
        expected_tcformer = plan["external_source_identity"]["tcformer"]
        if provenance != {
            "repository": expected_tcformer["repository"],
            "commit": expected_tcformer["commit"],
            "tracked_dirty": False,
        }:
            raise ValueError("robustness TCFormer record provenance differs from plan")
    elif provenance is not None:
        raise ValueError(
            "robustness non-TCFormer record has unexpected third-party provenance"
        )
    worker_index = MODELS.index(job.model)
    worker_binding = plan["worker_bindings"][worker_index]
    resource = payload.get("resource_preflight")
    if not isinstance(resource, Mapping) or set(resource) != {"gpu", "disk"}:
        raise ValueError("robustness resource preflight fields changed")
    gpu = resource.get("gpu") if isinstance(resource, Mapping) else None
    disk = resource.get("disk") if isinstance(resource, Mapping) else None
    if (
        not isinstance(gpu, Mapping)
        or set(gpu)
        != {
            "gpu_uuid",
            "pci_bus_id",
            "name",
            "utilization_percent",
            "memory_used_mib",
            "own_pid",
            "own_compute_memory_mib",
            "foreign_compute_processes",
        }
        or gpu.get("gpu_uuid") != worker_binding["gpu_uuid"]
        or not isinstance(gpu.get("pci_bus_id"), str)
        or not gpu["pci_bus_id"]
        or not isinstance(gpu.get("name"), str)
        or not gpu["name"]
        or not 0.0 <= float(gpu.get("utilization_percent", -1.0)) <= 100.0
        or float(gpu.get("memory_used_mib", -1.0)) < 0.0
        or int(gpu.get("own_pid", 0)) <= 0
        or float(gpu.get("own_compute_memory_mib", -1.0)) < 0.0
    ):
        raise ValueError("robustness record GPU differs from worker binding")
    if gpu.get("foreign_compute_processes") != []:
        raise ValueError("robustness record reports a foreign GPU process")
    if (
        not isinstance(disk, Mapping)
        or set(disk)
        != {
            "path",
            "free_bytes",
            "free_gib",
            "minimum_free_gib",
        }
        or not isinstance(disk.get("path"), str)
        or not disk["path"]
        or int(disk.get("free_bytes", -1)) <= 0
        or float(disk.get("free_gib", -1.0)) < MINIMUM_FREE_GIB
        or float(disk.get("minimum_free_gib", -1.0)) != MINIMUM_FREE_GIB
    ):
        raise ValueError("robustness record disk preflight violates 50 GiB guard")
    for name in ("accuracy", "balanced_accuracy", "roc_auc"):
        if not 0.0 <= float(payload["validation_metrics"][name]) <= 1.0:
            raise ValueError(f"robustness record {name} is out of range")
    if not -1.0 <= float(payload["validation_metrics"]["cohen_kappa"]) <= 1.0:
        raise ValueError("robustness record Cohen kappa is out of range")
    if float(payload["validation_metrics"]["negative_log_likelihood"]) < 0.0:
        raise ValueError("robustness record log loss is negative")
    fit = payload["fit"]
    if (
        not isinstance(fit, Mapping)
        or set(fit)
        != {
            "best_epoch",
            "epochs_run",
            "best_validation_loss",
            "early_stopping",
            "config",
        }
        or fit["early_stopping"] is not True
        or not 0 <= int(fit["best_epoch"]) < int(fit["epochs_run"])
        or not 1 <= int(fit["epochs_run"]) <= int(plan["train_config"]["epochs"])
        or not math.isfinite(float(fit["best_validation_loss"]))
        or float(fit["best_validation_loss"]) < 0.0
    ):
        raise ValueError("robustness fit identity is invalid")
    if not isinstance(payload["timing"], Mapping) or set(payload["timing"]) != {
        "construction_seconds",
        "fit_seconds",
        "evaluation_seconds",
        "total_seconds",
    }:
        raise ValueError("robustness timing fields changed")
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
            raise ValueError(f"robustness timing {name} is invalid")


def run_job_atomically(
    *,
    record_path: Path,
    job: ScreenJob,
    plan: Mapping[str, Any],
    executor: Callable[[ScreenJob], Mapping[str, Any]],
) -> bool:
    """Run one missing job, or fail closed on a corrupt completed artifact."""

    if record_path.exists():
        existing = _strict_regular_json(record_path)
        validate_record_payload(existing, job=job, plan_sha256=str(plan["plan_sha256"]))
        _validate_record_bindings(existing, job=job, plan=plan)
        return False
    payload = dict(executor(job))
    validate_record_payload(payload, job=job, plan_sha256=str(plan["plan_sha256"]))
    _validate_record_bindings(payload, job=job, plan=plan)
    if _atomic_create_json(record_path, payload):
        return True
    raced = _strict_regular_json(record_path)
    validate_record_payload(raced, job=job, plan_sha256=str(plan["plan_sha256"]))
    _validate_record_bindings(raced, job=job, plan=plan)
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
            raise RobustnessScreenError(f"cache size changed after plan: {path}")
        if _file_sha256(path) != expected_binding[1]:
            raise RobustnessScreenError(f"cache hash changed after plan: {path}")
        _VERIFIED_CACHE_BINDINGS[path] = expected_binding
    elif observed_binding != expected_binding:
        raise RobustnessScreenError("cache path reused with a different identity")

    with np.load(path, allow_pickle=False) as archive:
        sessions = archive["sessions"].astype(str)
        runs = archive["runs"].astype(str)
        train_rows, validation_rows = robustness_rows(
            job.dataset, job.subject, sessions, runs
        )
        if _rows_sha256(train_rows) != split_contract["train"]["rows_sha256"]:
            raise RobustnessScreenError("training rows differ from plan")
        if _rows_sha256(validation_rows) != split_contract["validation"]["rows_sha256"]:
            raise RobustnessScreenError("validation rows differ from plan")
        positions = np.asarray(archive["positions"], dtype=np.float32).copy()
        channel_names = tuple(str(value) for value in archive["channel_names"].tolist())
        identity = json.loads(str(archive["identity"].item()))

    selected_rows = np.concatenate((train_rows, validation_rows))
    selected_x, x_shape = _read_selected_npy_rows(
        path,
        "x",
        selected_rows,
        expected_dtype=np.dtype(np.float32),
        expected_ndim=3,
    )
    selected_y, y_shape = _read_selected_npy_rows(
        path,
        "y",
        selected_rows,
        expected_dtype=np.dtype(np.int64),
        expected_ndim=1,
    )
    if x_shape[0] != len(sessions) or y_shape != (len(sessions),):
        raise RobustnessScreenError("streamed EEG/label row count is invalid")
    train_count = len(train_rows)
    x_train = selected_x[:train_count]
    x_validation = selected_x[train_count:]
    y_train = selected_y[:train_count]
    y_validation = selected_y[train_count:]

    if identity.get("array_sha256") != cache_contract["array_sha256"]:
        raise RobustnessScreenError("cache array identity differs from plan")
    expected_classes = set(range(dataset_spec(job.dataset).n_classes))
    if set(y_train.tolist()) != expected_classes:
        raise RobustnessScreenError("training rows do not contain every class")
    if set(y_validation.tolist()) != expected_classes:
        raise RobustnessScreenError("validation rows do not contain every class")
    if x_train.ndim != 3 or x_validation.ndim != 3:
        raise RobustnessScreenError("EEG arrays must be three-dimensional")
    if x_train.shape[1:] != x_validation.shape[1:]:
        raise RobustnessScreenError("training and validation shapes differ")
    if positions.shape != (x_train.shape[1], 3):
        raise RobustnessScreenError("coordinate shape differs from EEG channels")
    if len(channel_names) != x_train.shape[1]:
        raise RobustnessScreenError("channel metadata differs from EEG channels")
    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        positions,
        channel_names,
        str(identity["array_sha256"]),
    )


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
    if job.model == CANDIDATE_MODEL:
        from .chsd_conditioned import make_chsd_conditioned_model

        return make_chsd_conditioned_model(
            requested_model=job.model,
            n_channels=n_channels,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            channel_names=channel_names,
            channel_positions=position_tensor,
        )
    if job.model not in REFERENCE_MODELS:
        raise RobustnessScreenError(f"unexpected model {job.model}")
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
    scaler_mean, scaler_std = fit_channel_scaler(x_train_raw, channel_names)
    x_train = apply_channel_scaler(x_train_raw, scaler_mean, scaler_std)
    x_validation = apply_channel_scaler(x_validation_raw, scaler_mean, scaler_std)
    configure_determinism(job.seed)

    construction_started = time.perf_counter()
    preprocessing = preprocessing_for_dataset(job.dataset)
    model = _make_model(
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
    source_hook = getattr(model, "fit_source_statistics", None)
    if callable(source_hook):
        raise RobustnessScreenError(
            f"{job.model} unexpectedly exposes source-statistics initialization"
        )
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
        "created_at": datetime.now(UTC).isoformat(),
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
        "environment_identity_sha256": _mapping_sha256(plan["environment_identity"]),
        "split": {
            "train": dict(
                plan["split_identity"][_split_key(job.dataset, job.subject, job.fold)][
                    "train"
                ]
            ),
            "validation": dict(
                plan["split_identity"][_split_key(job.dataset, job.subject, job.fold)][
                    "validation"
                ]
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
                "available": False,
                "ran": False,
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
    validate_record_payload(record, job=job, plan_sha256=str(plan["plan_sha256"]))
    _validate_record_bindings(record, job=job, plan=plan)
    del probabilities, fit, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _worker_binding(plan: Mapping[str, Any], worker_index: int) -> Mapping[str, Any]:
    bindings = [
        item
        for item in plan["worker_bindings"]
        if int(item["worker_index"]) == worker_index
    ]
    if len(bindings) != 1:
        raise RobustnessScreenError("worker binding is absent or duplicated")
    return bindings[0]


def _verify_runtime(plan: Mapping[str, Any]) -> None:
    if source_identity() != plan["source_identity"]:
        raise RobustnessScreenError("source differs from immutable plan")
    if robustness_environment_identity() != plan["environment_identity"]:
        raise RobustnessScreenError("UV/runtime identity differs from plan")
    if tcformer_source_identity() != plan["external_source_identity"]["tcformer"]:
        raise RobustnessScreenError("TCFormer source differs from plan")


def run_worker(
    *,
    run_root: Path,
    cache_root: Path,
    worker_index: int,
    gpu_uuid: str,
) -> int:
    """Execute one plan-bound, one-model static physical-GPU partition."""

    plan = load_plan(run_root)
    binding = _worker_binding(plan, worker_index)
    if gpu_uuid != binding["gpu_uuid"]:
        raise RobustnessScreenError("worker GPU UUID differs from plan binding")
    if str(cache_root.resolve()) != str(Path(plan["cache_root"]).resolve()):
        raise RobustnessScreenError("worker cache root differs from plan")
    _worker_preflight(gpu_uuid)
    _verify_runtime(plan)
    jobs = tuple(
        ScreenJob.from_mapping(item)
        for item in plan["jobs"]
        if int(item["worker_index"]) == worker_index
    )
    expected = jobs_for_worker(robustness_jobs(), worker_index)
    if jobs != expected or {job.model for job in jobs} != {binding["model"]}:
        raise RobustnessScreenError("worker partition differs from plan")

    completed = 0
    resumed = 0
    for job in jobs:
        path = _record_path(run_root, job)
        if path.exists():
            existing = _strict_regular_json(path)
            validate_record_payload(
                existing,
                job=job,
                plan_sha256=str(plan["plan_sha256"]),
            )
            _validate_record_bindings(existing, job=job, plan=plan)
            resumed += 1
            continue

        def executor(current: ScreenJob) -> Mapping[str, Any]:
            # Repeat provenance and cooperative resource checks immediately
            # before every new unit.  Existing valid records are never rerun.
            _verify_runtime(plan)
            resource = {
                "gpu": cooperative_gpu_guard(gpu_uuid),
                "disk": disk_guard(run_root),
            }
            print(
                f"[{datetime.now(UTC).isoformat()}] "
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
            plan=plan,
            executor=executor,
        ):
            completed += 1
            print(
                f"[{datetime.now(UTC).isoformat()}] "
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
            payload = _strict_regular_json(path)
            validate_record_payload(
                payload,
                job=job,
                plan_sha256=str(plan["plan_sha256"]),
            )
            _validate_record_bindings(payload, job=job, plan=plan)
        except Exception as error:  # noqa: BLE001 - status must report all corruption
            corrupt[job.job_id] = f"{type(error).__name__}: {error}"
        else:
            complete += 1
            workers[str(worker_index)]["complete"] += 1
    return {
        "schema": STATUS_SCHEMA,
        "amendment_label": AMENDMENT_LABEL,
        "evaluation_scope": EVALUATION_SCOPE,
        "plan_sha256": plan["plan_sha256"],
        "v2_plan_sha256": plan["lineage"]["reference_v2"]["plan_sha256"],
        "v3_plan_sha256": plan["lineage"]["conditioned_v3"]["plan_sha256"],
        "historical_v3_decision": plan["lineage"]["conditioned_v3_analysis"][
            "decision"
        ],
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

    plan = subparsers.add_parser("plan", help="freeze the 460-job amendment")
    plan.add_argument("--run-root", type=Path, required=True)
    plan.add_argument("--cache-root", type=Path, required=True)
    plan.add_argument("--reference-v2-run-root", type=Path, required=True)
    plan.add_argument("--v3-run-root", type=Path, required=True)
    plan.add_argument("--v3-analysis-path", type=Path, required=True)
    plan.add_argument(
        "--gpu-uuid",
        action="append",
        required=True,
        help="full physical GPU UUID; repeat exactly four times in model order",
    )

    status = subparsers.add_parser("status", help="audit robustness records")
    status.add_argument("--run-root", type=Path, required=True)

    worker = subparsers.add_parser(
        "worker", help="run one plan-bound model/GPU partition"
    )
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
    if args.command == "plan":
        manifest = build_manifest(
            cache_root=args.cache_root,
            reference_v2_run_root=args.reference_v2_run_root,
            v3_run_root=args.v3_run_root,
            v3_analysis_path=args.v3_analysis_path,
            worker_gpu_uuids=args.gpu_uuid,
        )
        created = write_or_validate_plan(args.run_root.resolve(), manifest)
        print(
            json.dumps(
                {
                    "created": created,
                    "plan_sha256": manifest["plan_sha256"],
                    "n_subjects": manifest["n_subjects"],
                    "n_jobs": manifest["n_jobs"],
                    "worker_count": manifest["worker_count"],
                    "historical_v3_decision": manifest["lineage"][
                        "conditioned_v3_analysis"
                    ]["decision"],
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
            worker_index=args.worker_index,
            gpu_uuid=args.gpu_uuid,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AMENDMENT_LABEL",
    "CANDIDATE_MODEL",
    "DECISION_RULE",
    "EXPECTED_JOBS_PER_WORKER",
    "EXPECTED_JOB_COUNT",
    "EXPECTED_SUBJECT_COUNT",
    "EXPECTED_V2_PLAN_SHA256",
    "EXPECTED_V3_ANALYSIS_ARTIFACT_SHA256",
    "EXPECTED_V3_PLAN_SHA256",
    "HISTORICAL_V3_DECISION",
    "MODELS",
    "PLAN_SCHEMA",
    "RECORD_SCHEMA",
    "REFERENCE_MODELS",
    "ROBUSTNESS_COHORTS",
    "SEED",
    "SOURCE_FILES",
    "WORKER_COUNT",
    "assemble_manifest",
    "jobs_for_worker",
    "load_plan",
    "robustness_jobs",
    "robustness_rows",
    "run_job_atomically",
    "screen_status",
    "validate_manifest_payload",
    "validate_record_payload",
]
