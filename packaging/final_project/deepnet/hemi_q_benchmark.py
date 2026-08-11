"""Locked BNCI2014-004 benchmark for the single-output HemiQ-Field model.

The official five-session chronology is preserved for every participant:
sessions ``0train`` and ``1train`` fit the model, session ``2train`` selects
one checkpoint by binary cross-entropy, and sessions ``3test``/``4test`` are
prediction-only outer tests.  There is one HemiQ logit and no validation-time
choice among heads, routes, mixtures, augmentations, or post-hoc classifiers.

Development access is restricted to subjects 1--4.  Confirmation subjects
5--9 additionally require a validated, source-pinned development artifact, an
exact future output path, an explicit token, and a permanent study-wide receipt
claimed with ``O_EXCL`` before the confirmation loader can be called.  A failed
confirmation may resume only the exact claimed output and environment; the
receipt is never removed and cannot authorize an alternative run.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from .config import PROJECT_ROOT
from .external_bnci2014_004 import (
    BNCI_CHANNELS,
    BNCI_SFREQ,
    CONFIRMATION_SUBJECTS,
    DEVELOPMENT_SUBJECTS,
    EPOCH_SAMPLES,
    FIT_SESSIONS,
    PROTOCOL_ID,
    TEST_SESSIONS,
    VALIDATION_SESSION,
    load_bnci2014_004_subject,
    protocol_metadata,
)
from .hemi_q_field_net import HemiQFieldClassifier, HemiQFieldConfig


DATASET = "BNCI2014-004"
MODEL_NAME = "HemiQ-Field"
SCHEMA_VERSION = 1
FROZEN_MANIFEST_SCHEMA_VERSION = 1
CONFIRMATION_RECEIPT_SCHEMA_VERSION = 1
MODES = ("bnci004-dev", "bnci004-confirm")
CONFIRMATION_TOKEN = "CONFIRM-BNCI2014-004-S5-9-HEMIQ-ONCE"
CONFIRMATION_STUDY_ID = "eegthingy-hemiq-bnci2014-004-s5-9-final-v1"
CONFIRMATION_RECEIPT_PATH = (
    PROJECT_ROOT
    / "deepnet"
    / "results"
    / "hemi_q"
    / ".confirmation-receipts"
    / f"{CONFIRMATION_STUDY_ID}.json"
)

EXPECTED_SPLITS: dict[str, dict[str, Any]] = {
    "train": {"counts": [240, 260, 280, 300, 320], "sessions": list(FIT_SESSIONS)},
    "validation": {"counts": [120, 140, 160], "sessions": [VALIDATION_SESSION]},
    "test": {"counts": [240, 260, 280, 300, 320], "sessions": list(TEST_SESSIONS)},
}
ALLOWED_SESSION_TRIAL_COUNTS = (120, 140, 160)

MODEL_CONTRACT: dict[str, Any] = {
    "name": MODEL_NAME,
    "implementation": "deepnet.hemi_q_field_net.HemiQFieldClassifier",
    "input": "8-30 Hz provided-bipolar C3/Cz/C4 epochs",
    "channels": list(BNCI_CHANNELS),
    "reference": "provided_bipolar",
    "sampling_frequency_hz": float(BNCI_SFREQ),
    "fit_sessions": list(FIT_SESSIONS),
    "validation_sessions": [VALIDATION_SESSION],
    "test_sessions": list(TEST_SESSIONS),
    "output": "sole_hemi_q_logit",
    "checkpoint_selection_metric": "validation_binary_cross_entropy",
    "checkpoint_selection_direction": "minimize",
    "candidate_selection": False,
    "test_time_augmentation": False,
    "unlabeled_test_calibration": False,
}

# This list deliberately includes the complete numeric provenance graph, not
# only the two files named on the command line.  Keep it synchronized with the
# imports in hemi_q_field_net.py and external_bnci2014_004.py.
_SOURCE_FILES = (
    "deepnet/__init__.py",
    "deepnet/hemi_q_benchmark.py",
    "deepnet/hemi_q_field_net.py",
    "deepnet/external_bnci2014_004.py",
    "deepnet/config.py",
    "deepnet/data.py",
    "deepnet/augment.py",
    "deepnet/spd.py",
    "deepnet/cameo_net.py",
    "deepnet/tangent_anchor.py",
)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip()

    commit = run("git", "rev-parse", "HEAD") or None
    status = run("git", "status", "--porcelain") if commit is not None else ""
    return {
        "available": commit is not None,
        "commit": commit,
        "dirty": bool(status) if commit is not None else None,
        "status": status.splitlines() if commit is not None else None,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    packages = {
        name: _package_version(name)
        for name in (
            "numpy",
            "scipy",
            "scikit-learn",
            "torch",
            "braindecode",
            "mne",
            "moabb",
        )
    }
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }


def _configure_torch_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _run_environment() -> dict[str, Any]:
    environment = _environment()
    environment["determinism"] = {
        "algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": getattr(
            torch, "is_deterministic_algorithms_warn_only_enabled", lambda: None
        )(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    return environment


def _source_manifest() -> dict[str, str]:
    return {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in _SOURCE_FILES
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_config(config: HemiQFieldConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), allow_nan=False))


def _config_from_mapping(values: Mapping[str, Any]) -> HemiQFieldConfig:
    if not isinstance(values, Mapping):
        raise ValueError("HemiQ configuration must be an object")
    try:
        config = HemiQFieldConfig(**dict(values))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid HemiQ configuration") from error
    if _json_config(config) != dict(values):
        raise ValueError("HemiQ configuration is not canonically encoded")
    return config


def _sanitized_command() -> list[str]:
    command = list(sys.argv)
    for index, value in enumerate(command):
        if value.startswith("--confirmation-token="):
            command[index] = "--confirmation-token=<redacted>"
        elif value == "--confirmation-token" and index + 1 < len(command):
            command[index + 1] = "<redacted>"
    return command


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _data_manifest(
    data: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, str]]:
    return {
        split: {name: _array_sha256(values) for name, values in arrays.items()}
        for split, arrays in data.items()
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}-",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _exclusive_json_create(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


class _OutputLock:
    def __init__(self, output: Path) -> None:
        self.path = output.with_name(output.name + ".lock")
        self.stream: Any | None = None

    def __enter__(self) -> "_OutputLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError(f"another process holds the result lock: {self.path}") from error
        return self

    def __exit__(self, *args: object) -> None:
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


def _parse_ints(value: str) -> list[int]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            low, high = token.split("-", 1)
            result.extend(range(int(low), int(high) + 1))
        else:
            result.append(int(token))
    return result


def _validate_contract(mode: str, subjects: Sequence[int], seeds: Sequence[int]) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    allowed = set(
        DEVELOPMENT_SUBJECTS if mode == "bnci004-dev" else CONFIRMATION_SUBJECTS
    )
    if not subjects or not set(subjects).issubset(allowed):
        raise ValueError(
            f"{mode} subjects must stay inside the locked partition "
            f"{min(allowed)}--{max(allowed)}"
        )
    if len(set(subjects)) != len(subjects):
        raise ValueError("subjects must be unique")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")


def _validate_model_config(config: HemiQFieldConfig) -> None:
    if config.n_times != EPOCH_SAMPLES or not math.isclose(
        config.sfreq, BNCI_SFREQ, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError(
            "HemiQ n_times/sfreq must match the locked BNCI004 preprocessing"
        )
    if config.deterministic is not True:
        raise ValueError("the locked HemiQ study requires strict determinism")


def _validate_loaded_data(data: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    if set(data) != set(EXPECTED_SPLITS):
        raise ValueError("loader must return exactly train/validation/test splits")
    required = {"broadband", "covariances", "labels", "session_ids"}
    for split, expected in EXPECTED_SPLITS.items():
        arrays = data[split]
        if set(arrays) != required:
            raise ValueError(f"{split} must contain exactly {sorted(required)}")
        count = int(len(arrays["labels"]))
        if count not in expected["counts"]:
            raise ValueError(
                f"{split} has {count} trials; expected one of {expected['counts']}"
            )
        broadband = np.asarray(arrays["broadband"])
        covariances = np.asarray(arrays["covariances"])
        labels = np.asarray(arrays["labels"])
        session_ids = np.asarray(arrays["session_ids"]).astype(str)
        if broadband.shape != (count, len(BNCI_CHANNELS), EPOCH_SAMPLES):
            raise ValueError(f"{split} broadband shape violates the fixed protocol")
        if covariances.shape != (count, 4, len(BNCI_CHANNELS), len(BNCI_CHANNELS)):
            raise ValueError(f"{split} covariance shape violates the fixed protocol")
        if labels.shape != (count,) or session_ids.shape != (count,):
            raise ValueError(f"{split} labels/session ids violate the fixed protocol")
        if not np.all(np.isfinite(broadband)) or not np.all(np.isfinite(covariances)):
            raise ValueError(f"{split} contains non-finite signal values")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError(f"{split} must contain both binary classes")
        expected_sessions = list(expected["sessions"])
        if sorted(set(session_ids.tolist())) != sorted(expected_sessions):
            raise ValueError(f"{split} session ids violate the fixed chronology")
        for session in expected_sessions:
            rows = session_ids == session
            session_count = int(rows.sum())
            if session_count not in ALLOWED_SESSION_TRIAL_COUNTS:
                raise ValueError(
                    f"{split} session {session} has {session_count} trials; expected "
                    f"one of {list(ALLOWED_SESSION_TRIAL_COUNTS)}"
                )
            values, counts = np.unique(labels[rows], return_counts=True)
            if values.tolist() != [0, 1] or counts.tolist() != [
                session_count // 2,
                session_count // 2,
            ]:
                raise ValueError(f"{split} session {session} is not class balanced")
    split_sessions = [
        set(np.asarray(data[split]["session_ids"]).astype(str).tolist())
        for split in ("train", "validation", "test")
    ]
    if any(
        split_sessions[first] & split_sessions[second]
        for first in range(3)
        for second in range(first + 1, 3)
    ):
        raise ValueError("fit, selection, and test session sets must be disjoint")


def _validate_probabilities(probabilities: np.ndarray, n_expected: int) -> np.ndarray:
    result = np.asarray(probabilities, dtype=np.float64)
    if result.shape != (n_expected, 2):
        raise ValueError(f"expected probabilities shaped {(n_expected, 2)}, got {result.shape}")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("class probabilities do not sum to one")
    return result


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = _validate_probabilities(probabilities, len(labels))
    predicted = probabilities.argmax(axis=1)
    positive = np.clip(probabilities[:, 1], 1e-12, 1.0 - 1e-12)
    binary_cross_entropy = -np.mean(
        labels * np.log(positive) + (1 - labels) * np.log(1.0 - positive)
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])),
        "binary_cross_entropy": float(binary_cross_entropy),
    }


def _session_metrics(
    labels: np.ndarray, probabilities: np.ndarray, session_ids: np.ndarray
) -> dict[str, dict[str, Any]]:
    sessions = np.asarray(session_ids).astype(str)
    return {
        session: {
            "n_trials": int(np.sum(sessions == session)),
            **_metrics(labels[sessions == session], probabilities[sessions == session]),
        }
        for session in TEST_SESSIONS
    }


def _prediction_trace(
    labels: np.ndarray, probabilities: np.ndarray, session_ids: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "window_index": int(index),
            "session_id": str(session),
            "label": int(label),
            "prediction": int(np.argmax(probability)),
            "probability_left": float(probability[0]),
            "probability_right": float(probability[1]),
        }
        for index, (label, probability, session) in enumerate(
            zip(labels, probabilities, session_ids, strict=True)
        )
    ]


def _json_diagnostics(value: Any, *, name: str) -> Any:
    """Round-trip diagnostics now so NaNs or opaque tensors cannot reach JSON."""

    try:
        encoded = json.dumps(value, allow_nan=False)
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON data") from error
    if not isinstance(result, dict) or not result:
        raise ValueError(f"{name} must be a non-empty object")
    return result


def _fit_predict(
    data: Mapping[str, Mapping[str, np.ndarray]],
    *,
    config: HemiQFieldConfig,
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit one HemiQ model; the outer test split is prediction-only."""

    _validate_loaded_data(data)
    training = data["train"]
    validation = data["validation"]
    test = data["test"]
    run_config = replace(config, seed=int(seed), device=str(device))
    started = time.perf_counter()
    classifier = HemiQFieldClassifier(run_config).fit(
        training["broadband"],
        training["covariances"],
        training["labels"],
        validation["broadband"],
        None,
        validation["labels"],
        channels=BNCI_CHANNELS,
    )
    probabilities = _validate_probabilities(
        classifier.predict_proba(test["broadband"]), len(test["labels"])
    )
    parity = _json_diagnostics(
        {
            "definition": "max_abs_logit_x_plus_logit_reflection_x",
            "validation_max_abs_logit_error": float(
                classifier.max_equivariance_error(validation["broadband"])
            ),
            "test_max_abs_logit_error": float(
                classifier.max_equivariance_error(test["broadband"])
            ),
        },
        name="parity diagnostics",
    )
    filters = _json_diagnostics(
        classifier.filter_diagnostics(), name="filter diagnostics"
    )
    fit = {
        "output": "sole_hemi_q_logit",
        "candidate_selection": False,
        "checkpoint_selection": {
            "metric": "validation_binary_cross_entropy",
            "direction": "minimize",
            "best_epoch": int(classifier.best_epoch_),
            "best_value": float(classifier.best_validation_loss_),
        },
        "epochs_run": int(classifier.epochs_run_),
        "parameter_count": int(classifier.param_count_),
        "trainable_parameter_count": int(classifier.trainable_param_count_),
        "teacher_parameter_count": int(classifier.teacher_param_count_),
        "teacher_was_used": bool(classifier.teacher_was_used_),
        "train_seconds": float(classifier.train_seconds_),
        "wall_seconds": float(time.perf_counter() - started),
        "n_train": len(training["labels"]),
        "n_validation": len(validation["labels"]),
        "n_test": len(test["labels"]),
        "train_session_ids": list(FIT_SESSIONS),
        "validation_session_ids": [VALIDATION_SESSION],
        "test_session_ids": list(TEST_SESSIONS),
        "channels": list(BNCI_CHANNELS),
        "reference": "provided_bipolar",
        "source_only_preprocessing": True,
        "unlabeled_test_calibration": False,
        "session_trial_counts": {
            session: int(
                np.sum(
                    np.asarray(data[split]["session_ids"]).astype(str) == session
                )
            )
            for split in ("train", "validation", "test")
            for session in EXPECTED_SPLITS[split]["sessions"]
        },
    }
    return probabilities, fit, parity, filters


def _expected_record_keys(subjects: Sequence[int], seeds: Sequence[int]) -> list[list[int]]:
    return [[int(subject), int(seed)] for subject in subjects for seed in seeds]


def _actual_record_keys(payload: Mapping[str, Any]) -> list[list[int]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("result records must be a list")
    identities: list[list[int]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("result records must be objects")
        try:
            identities.append([int(record["subject"]), int(record["seed"])])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("result record identity is invalid") from error
    if len(identities) != len({tuple(identity) for identity in identities}):
        raise ValueError("result contains duplicate record identities")
    return identities


def _update_completion(payload: dict[str, Any], *, status: str) -> None:
    expected = _expected_record_keys(payload["subjects"], payload["seeds"])
    actual = _actual_record_keys(payload)
    if any(identity not in expected for identity in actual):
        raise ValueError("result contains a record outside its frozen contract")
    complete = actual == expected
    if status == "complete" and not complete:
        raise RuntimeError("cannot mark an incomplete result complete")
    payload["completion"] = {
        "status": status,
        "expected_record_keys": expected,
        "actual_record_keys": actual,
        "expected_records": len(expected),
        "actual_records": len(actual),
        "complete": complete and status == "complete",
    }


def _experiment_contract(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: HemiQFieldConfig,
    device: str,
    frozen_manifest_sha256: str | None,
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "protocol_id": PROTOCOL_ID,
        "model": MODEL_NAME,
        "mode": mode,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "config": _json_config(config),
        "device": device,
        "model_contract": MODEL_CONTRACT,
        "frozen_manifest_sha256": frozen_manifest_sha256,
    }


def _new_payload(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: HemiQFieldConfig,
    device: str,
    environment: Mapping[str, Any],
    frozen_manifest: Mapping[str, Any] | None,
    frozen_manifest_path: Path | None,
    frozen_manifest_sha256: str | None,
) -> dict[str, Any]:
    contract = _experiment_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        config=config,
        device=device,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    source = _source_manifest()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": None,
        **contract,
        "contract_sha256": _json_sha256(contract),
        "confirmation_access": mode == "bnci004-confirm",
        "confirmation": (
            {
                "token_verified": True,
                "frozen_manifest_path": str(Path(frozen_manifest_path).resolve()),
                "frozen_manifest_sha256": frozen_manifest_sha256,
                "development_artifact": frozen_manifest["development_artifact"],
                "confirmation_contract": frozen_manifest["confirmation_contract"],
            }
            if frozen_manifest is not None and frozen_manifest_path is not None
            else None
        ),
        "command": _sanitized_command(),
        "repository": _git_state(),
        "source_sha256": source,
        "source_manifest_sha256": _json_sha256(source),
        "environment": dict(environment),
        "environment_sha256": _json_sha256(environment),
        "records": [],
        "summary": {},
    }
    _update_completion(payload, status="running")
    return payload


def _load_or_initialize(
    output: Path,
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: HemiQFieldConfig,
    device: str,
    resume: bool,
    allow_claimed_resume_without_output: bool,
    environment: Mapping[str, Any],
    frozen_manifest: Mapping[str, Any] | None,
    frozen_manifest_path: Path | None,
    frozen_manifest_sha256: str | None,
) -> dict[str, Any]:
    expected = _experiment_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        config=config,
        device=device,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    if not output.exists():
        if resume and not allow_claimed_resume_without_output:
            raise FileNotFoundError(f"--resume requires an existing result: {output}")
        return _new_payload(
            mode=mode,
            subjects=subjects,
            seeds=seeds,
            config=config,
            device=device,
            environment=environment,
            frozen_manifest=frozen_manifest,
            frozen_manifest_path=frozen_manifest_path,
            frozen_manifest_sha256=frozen_manifest_sha256,
        )
    if not resume:
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    payload_bytes = output.read_bytes()
    payload = json.loads(payload_bytes)
    if {key: payload.get(key) for key in expected} != expected:
        raise ValueError("cannot resume an output with a different experiment contract")
    if payload.get("contract_sha256") != _json_sha256(expected):
        raise ValueError("cannot resume an output with a corrupt contract digest")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("cannot resume an unsupported result schema")
    source = _source_manifest()
    if payload.get("source_sha256") != source or payload.get(
        "source_manifest_sha256"
    ) != _json_sha256(source):
        raise ValueError("cannot resume after benchmark source files have changed")
    if payload.get("environment") != dict(environment) or payload.get(
        "environment_sha256"
    ) != _json_sha256(environment):
        raise ValueError("cannot resume in a different execution environment")
    _update_completion(payload, status="running")
    return payload


def _summarize(payload: Mapping[str, Any]) -> dict[str, Any]:
    by_subject: dict[int, list[float]] = {}
    for record in payload["records"]:
        by_subject.setdefault(int(record["subject"]), []).append(
            float(record["metrics"]["balanced_accuracy"])
        )
    subject_values = np.asarray(
        [np.mean(values) for values in by_subject.values()], dtype=np.float64
    )
    records = list(payload["records"])
    return {
        "balanced_accuracy_mean": float(subject_values.mean()) if len(subject_values) else None,
        "balanced_accuracy_std": (
            float(subject_values.std(ddof=1)) if len(subject_values) > 1 else 0.0
        ),
        "n_participants": len(subject_values),
        "n_records": len(records),
        "parameter_count": (
            int(np.median([row["fit"]["parameter_count"] for row in records]))
            if records
            else None
        ),
        "trainable_parameter_count": (
            int(np.median([row["fit"]["trainable_parameter_count"] for row in records]))
            if records
            else None
        ),
        "best_epoch_median": (
            float(np.median([row["fit"]["checkpoint_selection"]["best_epoch"] for row in records]))
            if records
            else None
        ),
        "train_seconds_mean": (
            float(np.mean([row["fit"]["train_seconds"] for row in records]))
            if records
            else None
        ),
    }


def _validate_hash_tree(value: Any, *, name: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty digest object")
    for key, child in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} has a non-string key")
        if isinstance(child, dict):
            _validate_hash_tree(child, name=f"{name}.{key}")
        elif not (
            isinstance(child, str)
            and len(child) == 64
            and all(character in "0123456789abcdef" for character in child)
        ):
            raise ValueError(f"{name}.{key} is not a SHA-256 digest")


def _validate_finite_json_tree(value: Any, *, name: str) -> None:
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{name} must not be empty")
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} has a non-string key")
            _validate_finite_json_tree(child, name=f"{name}.{key}")
    elif isinstance(value, list):
        if not value:
            raise ValueError(f"{name} must not be empty")
        for index, child in enumerate(value):
            _validate_finite_json_tree(child, name=f"{name}[{index}]")
    elif isinstance(value, bool) or value is None or isinstance(value, str):
        return
    elif isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} contains a non-finite number")
    else:
        raise ValueError(f"{name} contains unsupported data")


def _validate_record(record: Mapping[str, Any], *, subject: int, seed: int) -> None:
    if (
        record.get("model") != MODEL_NAME
        or record.get("subject") != subject
        or record.get("seed") != seed
        or record.get("deterministic") is not True
    ):
        raise ValueError("development artifact record identity is invalid")
    if record.get("protocol") != protocol_metadata(subject):
        raise ValueError(f"development S{subject} protocol metadata is invalid")
    _validate_hash_tree(record.get("data_sha256"), name=f"S{subject}.data_sha256")

    predictions = record.get("predictions")
    if not isinstance(predictions, list) or len(predictions) not in EXPECTED_SPLITS[
        "test"
    ]["counts"]:
        raise ValueError(f"development S{subject} predictions are incomplete")
    labels: list[int] = []
    probabilities: list[list[float]] = []
    sessions: list[str] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict) or prediction.get("window_index") != index:
            raise ValueError(f"development S{subject} prediction order is invalid")
        try:
            label = int(prediction["label"])
            predicted = int(prediction["prediction"])
            probability = [
                float(prediction["probability_left"]),
                float(prediction["probability_right"]),
            ]
            session = str(prediction["session_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"development S{subject} prediction payload is invalid") from error
        if label not in (0, 1) or predicted not in (0, 1):
            raise ValueError(f"development S{subject} has a non-binary label/prediction")
        if predicted != int(np.argmax(probability)):
            raise ValueError(f"development S{subject} stored prediction is inconsistent")
        labels.append(label)
        probabilities.append(probability)
        sessions.append(session)
    labels_array = np.asarray(labels, dtype=np.int64)
    probability_array = _validate_probabilities(
        np.asarray(probabilities), len(predictions)
    )
    sessions_array = np.asarray(sessions)
    observed_test_counts = {
        session: sessions.count(session) for session in TEST_SESSIONS
    }
    if set(sessions) != set(TEST_SESSIONS) or any(
        count not in ALLOWED_SESSION_TRIAL_COUNTS
        for count in observed_test_counts.values()
    ):
        raise ValueError(f"development S{subject} test sessions are invalid")
    recomputed = _metrics(labels_array, probability_array)
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"development S{subject} metrics are missing")
    for name, expected in recomputed.items():
        try:
            observed = float(metrics[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"development S{subject} metric {name} is invalid") from error
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"development S{subject} metric {name} does not recompute")
    if record.get("per_session_metrics") != _session_metrics(
        labels_array, probability_array, sessions_array
    ):
        raise ValueError(f"development S{subject} per-session metrics do not recompute")

    fit = record.get("fit")
    expected_fit = {
        "output": "sole_hemi_q_logit",
        "candidate_selection": False,
        "train_session_ids": list(FIT_SESSIONS),
        "validation_session_ids": [VALIDATION_SESSION],
        "test_session_ids": list(TEST_SESSIONS),
        "channels": list(BNCI_CHANNELS),
        "reference": "provided_bipolar",
        "source_only_preprocessing": True,
        "unlabeled_test_calibration": False,
    }
    if not isinstance(fit, dict) or any(fit.get(key) != value for key, value in expected_fit.items()):
        raise ValueError(f"development S{subject} fit contract is invalid")
    for split, key in (
        ("train", "n_train"),
        ("validation", "n_validation"),
        ("test", "n_test"),
    ):
        if fit.get(key) not in EXPECTED_SPLITS[split]["counts"]:
            raise ValueError(f"development S{subject} {key} is invalid")
    if fit["n_test"] != len(predictions):
        raise ValueError(f"development S{subject} test count is inconsistent")
    session_trial_counts = fit.get("session_trial_counts")
    if not isinstance(session_trial_counts, dict) or set(session_trial_counts) != set(
        FIT_SESSIONS + (VALIDATION_SESSION,) + TEST_SESSIONS
    ):
        raise ValueError(f"development S{subject} session counts are invalid")
    if any(
        not isinstance(value, int) or value not in ALLOWED_SESSION_TRIAL_COUNTS
        for value in session_trial_counts.values()
    ):
        raise ValueError(f"development S{subject} session counts are invalid")
    if fit["n_train"] != sum(session_trial_counts[name] for name in FIT_SESSIONS):
        raise ValueError(f"development S{subject} train count is inconsistent")
    if fit["n_validation"] != session_trial_counts[VALIDATION_SESSION]:
        raise ValueError(f"development S{subject} validation count is inconsistent")
    if fit["n_test"] != sum(session_trial_counts[name] for name in TEST_SESSIONS):
        raise ValueError(f"development S{subject} test count is inconsistent")
    if any(
        session_trial_counts[name] != observed_test_counts[name]
        for name in TEST_SESSIONS
    ):
        raise ValueError(f"development S{subject} stored test session counts differ")
    selection = fit.get("checkpoint_selection")
    if not isinstance(selection, dict) or selection.get("metric") != (
        "validation_binary_cross_entropy"
    ) or selection.get("direction") != "minimize":
        raise ValueError(f"development S{subject} checkpoint selection is invalid")
    for key in (
        "parameter_count",
        "trainable_parameter_count",
        "teacher_parameter_count",
        "epochs_run",
    ):
        if not isinstance(fit.get(key), int) or int(fit[key]) <= 0:
            raise ValueError(f"development S{subject} {key} is invalid")
    if fit["trainable_parameter_count"] > fit["parameter_count"]:
        raise ValueError(f"development S{subject} parameter counts are inconsistent")
    if not isinstance(selection.get("best_epoch"), int) or not (
        -1 <= selection["best_epoch"] < fit["epochs_run"]
    ):
        raise ValueError(f"development S{subject} selected epoch is invalid")
    if not math.isfinite(float(selection.get("best_value", float("nan")))) or float(
        selection["best_value"]
    ) < 0.0:
        raise ValueError(f"development S{subject} validation BCE is invalid")
    for key in ("train_seconds", "wall_seconds"):
        if not math.isfinite(float(fit.get(key, float("nan")))) or float(fit[key]) < 0.0:
            raise ValueError(f"development S{subject} {key} is invalid")
    if fit.get("teacher_was_used") is not True:
        raise ValueError(f"development S{subject} teacher-use record is invalid")
    parity = record.get("parity_diagnostics")
    filters = record.get("filter_diagnostics")
    if not isinstance(parity, dict) or set(parity) != {
        "definition",
        "validation_max_abs_logit_error",
        "test_max_abs_logit_error",
    }:
        raise ValueError(f"development S{subject} parity diagnostics are invalid")
    _validate_finite_json_tree(parity, name=f"S{subject}.parity_diagnostics")
    if not isinstance(filters, dict):
        raise ValueError(f"development S{subject} filter diagnostics are invalid")
    _validate_finite_json_tree(filters, name=f"S{subject}.filter_diagnostics")


def _validate_development_payload(
    payload: Mapping[str, Any], *, require_current_source: bool = False
) -> None:
    if not isinstance(payload.get("config"), dict):
        raise ValueError("development artifact lacks a HemiQ configuration")
    config = _config_from_mapping(payload["config"])
    expected_top = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "protocol_id": PROTOCOL_ID,
        "model": MODEL_NAME,
        "mode": "bnci004-dev",
        "confirmation_access": False,
        "subjects": list(DEVELOPMENT_SUBJECTS),
        "seeds": [7],
        "model_contract": MODEL_CONTRACT,
    }
    if any(payload.get(key) != value for key, value in expected_top.items()):
        raise ValueError("development artifact does not match the frozen HemiQ contract")
    device = payload.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError("development artifact device is invalid")
    if config.device != device or config.seed != 7:
        raise ValueError("development artifact config device/seed is inconsistent")
    contract = _experiment_contract(
        mode="bnci004-dev",
        subjects=DEVELOPMENT_SUBJECTS,
        seeds=[7],
        config=config,
        device=device,
        frozen_manifest_sha256=None,
    )
    if payload.get("contract_sha256") != _json_sha256(contract):
        raise ValueError("development artifact contract digest is invalid")
    source = payload.get("source_sha256")
    _validate_hash_tree(source, name="source_sha256")
    if payload.get("source_manifest_sha256") != _json_sha256(source):
        raise ValueError("development artifact source-manifest digest is invalid")
    if require_current_source and source != _source_manifest():
        raise ValueError("development artifact source differs from current source")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValueError("development artifact environment is missing")
    _validate_finite_json_tree(environment, name="environment")
    if payload.get("environment_sha256") != _json_sha256(environment):
        raise ValueError("development artifact environment digest is invalid")

    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(DEVELOPMENT_SUBJECTS):
        raise ValueError("development artifact has the wrong record count")
    data_digests: dict[int, Any] = {}
    for record, subject in zip(records, DEVELOPMENT_SUBJECTS, strict=True):
        if not isinstance(record, dict):
            raise ValueError("development artifact record must be an object")
        _validate_record(record, subject=subject, seed=7)
        data_digests[subject] = record["data_sha256"]
    if payload.get("summary") != _summarize(payload):
        raise ValueError("development artifact summary does not recompute")
    expected_keys = _expected_record_keys(DEVELOPMENT_SUBJECTS, [7])
    completion = payload.get("completion")
    expected_completion = {
        "status": "complete",
        "expected_record_keys": expected_keys,
        "actual_record_keys": expected_keys,
        "expected_records": len(expected_keys),
        "actual_records": len(expected_keys),
        "complete": True,
    }
    if not isinstance(completion, dict) or any(
        completion.get(key) != value for key, value in expected_completion.items()
    ):
        raise ValueError("development artifact is not complete")


def _confirmation_manifest_contract(
    output: Path,
    *,
    device: str,
    config: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "study_id": CONFIRMATION_STUDY_ID,
        "dataset": DATASET,
        "protocol_id": PROTOCOL_ID,
        "model": MODEL_NAME,
        "subjects": list(CONFIRMATION_SUBJECTS),
        "seeds": [7],
        "config": dict(config),
        "device": device,
        "environment": dict(environment),
        "environment_sha256": _json_sha256(environment),
        "output_path": str(output.expanduser().resolve()),
        "receipt_path": str(Path(CONFIRMATION_RECEIPT_PATH).expanduser().resolve()),
    }


def write_frozen_manifest(
    development_artifact: Path,
    output: Path,
    *,
    confirmation_output: Path,
    device: str,
) -> dict[str, Any]:
    """Freeze one validated S1--4 result and one exact future S5--9 run."""

    development_artifact = Path(development_artifact)
    output = Path(output)
    confirmation_output = Path(confirmation_output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output}")
    if Path(CONFIRMATION_RECEIPT_PATH).exists():
        raise FileExistsError("permanent HemiQ confirmation receipt exists; cannot refreeze")
    if not development_artifact.is_file():
        raise FileNotFoundError(development_artifact)
    development_bytes = development_artifact.read_bytes()
    payload = json.loads(development_bytes)
    _validate_development_payload(payload, require_current_source=True)
    if device != payload["device"]:
        raise ValueError("confirmation device must match the frozen development device")
    contract = _confirmation_manifest_contract(
        confirmation_output,
        device=device,
        config=payload["config"],
        environment=payload["environment"],
    )
    reserved = {
        output.expanduser().resolve(),
        development_artifact.expanduser().resolve(),
        Path(contract["receipt_path"]),
    }
    if Path(contract["output_path"]) in reserved:
        raise ValueError("confirmation output must be distinct from frozen artifacts")
    if Path(contract["output_path"]).exists():
        raise FileExistsError("pinned confirmation output already exists")
    manifest = {
        "schema_version": FROZEN_MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "study_id": CONFIRMATION_STUDY_ID,
        "source_sha256": _source_manifest(),
        "model_contract": MODEL_CONTRACT,
        "confirmation_contract": contract,
        "development_artifact": {
            "path": str(development_artifact.resolve()),
            "sha256": hashlib.sha256(development_bytes).hexdigest(),
            "summary_sha256": _json_sha256(payload["summary"]),
            "source_sha256": payload["source_sha256"],
            "source_manifest_sha256": payload["source_manifest_sha256"],
            "environment_sha256": payload["environment_sha256"],
            "contract_sha256": payload["contract_sha256"],
            "mode": "bnci004-dev",
            "subjects": list(DEVELOPMENT_SUBJECTS),
            "seeds": [7],
            "config": payload["config"],
            "expected_records": len(DEVELOPMENT_SUBJECTS),
        },
    }
    _exclusive_json_create(output, manifest)
    return manifest


def _read_and_validate_frozen_manifest(path: Path | None) -> tuple[dict[str, Any], str]:
    """Read each frozen file once and validate only those captured bytes."""

    if path is None:
        raise ValueError("bnci004-confirm requires --frozen-manifest")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen manifest does not exist: {path}")
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest.get("schema_version") != FROZEN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported frozen-manifest schema")
    if manifest.get("study_id") != CONFIRMATION_STUDY_ID:
        raise ValueError("frozen manifest names a different study")
    current_source = _source_manifest()
    if manifest.get("source_sha256") != current_source:
        raise ValueError("confirmation source differs from the frozen manifest")
    if manifest.get("model_contract") != MODEL_CONTRACT:
        raise ValueError("confirmation model contract differs from the manifest")
    contract = manifest.get("confirmation_contract")
    expected = {
        "study_id": CONFIRMATION_STUDY_ID,
        "dataset": DATASET,
        "protocol_id": PROTOCOL_ID,
        "model": MODEL_NAME,
        "subjects": list(CONFIRMATION_SUBJECTS),
        "seeds": [7],
        "receipt_path": str(Path(CONFIRMATION_RECEIPT_PATH).expanduser().resolve()),
    }
    if not isinstance(contract, dict) or any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen manifest confirmation contract is invalid")
    if (
        not isinstance(contract.get("device"), str)
        or not contract["device"]
        or not isinstance(contract.get("output_path"), str)
        or not Path(contract["output_path"]).is_absolute()
    ):
        raise ValueError("frozen manifest device/output contract is invalid")
    _config_from_mapping(contract.get("config"))
    environment = contract.get("environment")
    if not isinstance(environment, dict) or not environment or contract.get(
        "environment_sha256"
    ) != _json_sha256(environment):
        raise ValueError("frozen manifest environment contract is invalid")
    _validate_finite_json_tree(environment, name="confirmation.environment")

    development = manifest.get("development_artifact")
    if not isinstance(development, dict) or not isinstance(development.get("path"), str):
        raise ValueError("frozen manifest lacks a development artifact")
    artifact_path = Path(development["path"]).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = path.parent / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"frozen development artifact is missing: {artifact_path}")
    development_bytes = artifact_path.read_bytes()
    if development.get("sha256") != hashlib.sha256(development_bytes).hexdigest():
        raise ValueError("frozen development artifact hash no longer matches")
    payload = json.loads(development_bytes)
    _validate_development_payload(payload)
    expected_development = {
        "summary_sha256": _json_sha256(payload["summary"]),
        "source_sha256": payload["source_sha256"],
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "environment_sha256": payload["environment_sha256"],
        "contract_sha256": payload["contract_sha256"],
        "mode": "bnci004-dev",
        "subjects": list(DEVELOPMENT_SUBJECTS),
        "seeds": [7],
        "config": payload["config"],
        "expected_records": len(DEVELOPMENT_SUBJECTS),
    }
    if any(development.get(key) != value for key, value in expected_development.items()):
        raise ValueError("frozen development metadata is invalid")
    if payload["source_sha256"] != current_source:
        raise ValueError("frozen development source differs from confirmation source")
    if contract["config"] != payload["config"] or contract["environment"] != payload[
        "environment"
    ]:
        raise ValueError("frozen confirmation config/environment differs from development")
    return manifest, manifest_sha256


def _validate_confirmation_contract(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: HemiQFieldConfig,
    device: str,
    confirmation_token: str | None,
    frozen_manifest: Path | None,
) -> tuple[dict[str, Any], str] | None:
    if mode != "bnci004-confirm":
        if confirmation_token is not None or frozen_manifest is not None:
            raise ValueError("confirmation token/manifest may only be used with bnci004-confirm")
        return None
    if list(subjects) != list(CONFIRMATION_SUBJECTS):
        raise ValueError("bnci004-confirm requires the exact ordered subjects 5--9")
    if list(seeds) != [7]:
        raise ValueError("bnci004-confirm requires the sole frozen seed 7")
    if confirmation_token != CONFIRMATION_TOKEN:
        raise ValueError("bnci004-confirm requires the exact explicit confirmation token")
    manifest, digest = _read_and_validate_frozen_manifest(frozen_manifest)
    contract = manifest["confirmation_contract"]
    if contract["device"] != device:
        raise ValueError("confirmation device differs from the frozen manifest")
    if contract["config"] != _json_config(config):
        raise ValueError("confirmation configuration differs from the frozen manifest")
    return manifest, digest


def _receipt_payload(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    contract = manifest["confirmation_contract"]
    immutable = {
        "schema_version": CONFIRMATION_RECEIPT_SCHEMA_VERSION,
        "study_id": CONFIRMATION_STUDY_ID,
        "manifest_sha256": manifest_sha256,
        "development_artifact_sha256": manifest["development_artifact"]["sha256"],
        "source_sha256": manifest["source_sha256"],
        "model_contract": manifest["model_contract"],
        "subjects": list(CONFIRMATION_SUBJECTS),
        "seeds": [7],
        "config": contract["config"],
        "device": contract["device"],
        "environment": dict(environment),
        "environment_sha256": _json_sha256(environment),
        "output_path": contract["output_path"],
        "receipt_path": contract["receipt_path"],
    }
    return {
        **immutable,
        "contract_sha256": _json_sha256(immutable),
        "state": "claimed",
        "created_at": _utc_now(),
        "updated_at": None,
    }


def _claim_confirmation_receipt(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    environment: Mapping[str, Any],
    output: Path,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    contract = manifest["confirmation_contract"]
    if dict(environment) != contract["environment"]:
        raise ValueError("confirmation environment differs from the frozen manifest")
    if output.expanduser().resolve() != Path(contract["output_path"]):
        raise ValueError("confirmation output differs from the frozen exact path")
    receipt_path = Path(contract["receipt_path"])
    expected = _receipt_payload(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        environment=environment,
    )
    immutable_keys = tuple(
        key for key in expected if key not in {"state", "created_at", "updated_at"}
    )
    if receipt_path.exists():
        if not resume:
            raise FileExistsError("permanent HemiQ confirmation receipt exists; refusing re-access")
        receipt_bytes = receipt_path.read_bytes()
        observed = json.loads(receipt_bytes)
        if any(observed.get(key) != expected[key] for key in immutable_keys):
            raise ValueError("confirmation receipt does not match the exact contract/environment")
        state = observed.get("state")
        if state == "complete":
            if not output.is_file() or observed.get("output_sha256") != _file_sha256(output):
                raise ValueError("completed confirmation output no longer matches its receipt")
            raise FileExistsError("HemiQ confirmation is already complete")
        if state not in {"claimed", "running", "interrupted"}:
            raise ValueError("confirmation receipt has an invalid state")
        if state != "claimed" and not output.is_file():
            raise FileNotFoundError("confirmation resume requires the interrupted result artifact")
        return receipt_path, observed
    if resume:
        raise FileNotFoundError("confirmation resume requires the permanent study receipt")
    if output.exists():
        raise FileExistsError("refusing to replace the pinned confirmation result")
    _exclusive_json_create(receipt_path, expected)
    return receipt_path, expected


def _update_confirmation_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    state: str,
    output: Path,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("permanent HemiQ confirmation receipt disappeared")
    updated = dict(receipt)
    updated["state"] = state
    updated["updated_at"] = _utc_now()
    if error is not None:
        updated["last_error"] = {"type": type(error).__name__, "message": str(error)}
    elif state == "complete":
        updated.pop("last_error", None)
        updated["output_sha256"] = _file_sha256(output)
    _atomic_json(path, updated)
    return updated


def run(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: HemiQFieldConfig,
    device: str,
    output: Path,
    resume: bool = False,
    frozen_manifest: Path | None = None,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Run an allowed cohort; every confirmation guard precedes data access."""

    subjects = [int(value) for value in subjects]
    seeds = [int(value) for value in seeds]
    output = Path(output)
    _configure_torch_determinism()
    _validate_contract(mode, subjects, seeds)
    # Device and a sole seed live both in the dataclass used by the estimator
    # and in explicit contract fields.  Normalize them once so provenance
    # always describes the configuration that was actually fit.
    config = replace(config, device=str(device))
    if len(seeds) == 1:
        config = replace(config, seed=int(seeds[0]))
    _validate_model_config(config)
    validated_manifest = _validate_confirmation_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        config=config,
        device=device,
        confirmation_token=confirmation_token,
        frozen_manifest=frozen_manifest,
    )
    environment = _run_environment()
    manifest: Mapping[str, Any] | None = None
    manifest_sha256: str | None = None
    receipt_path: Path | None = None
    receipt: dict[str, Any] | None = None
    claimed_resume_without_output = False
    if validated_manifest is not None:
        manifest, manifest_sha256 = validated_manifest
        receipt_path, receipt = _claim_confirmation_receipt(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            environment=environment,
            output=output,
            resume=resume,
        )
        claimed_resume_without_output = (
            resume and receipt.get("state") == "claimed" and not output.exists()
        )

    with _OutputLock(output):
        payload = _load_or_initialize(
            output,
            mode=mode,
            subjects=subjects,
            seeds=seeds,
            config=config,
            device=device,
            resume=resume,
            allow_claimed_resume_without_output=claimed_resume_without_output,
            environment=environment,
            frozen_manifest=manifest,
            frozen_manifest_path=frozen_manifest,
            frozen_manifest_sha256=manifest_sha256,
        )
        if not output.exists():
            _exclusive_json_create(output, payload)
        else:
            payload["updated_at"] = _utc_now()
            _atomic_json(output, payload)
        if receipt_path is not None and receipt is not None:
            receipt = _update_confirmation_receipt(
                receipt_path, receipt, state="running", output=output
            )
        try:
            completed = {
                (int(record["subject"]), int(record["seed"]))
                for record in payload["records"]
            }
            for subject in subjects:
                pending = [seed for seed in seeds if (subject, seed) not in completed]
                if not pending:
                    continue
                data = load_bnci2014_004_subject(
                    subject,
                    use_cache=True,
                    allow_confirmation=mode == "bnci004-confirm",
                )
                _validate_loaded_data(data)
                data_sha256 = _data_manifest(data)
                for seed in pending:
                    probabilities, fit, parity, filters = _fit_predict(
                        data, config=config, seed=seed, device=device
                    )
                    truth = np.asarray(data["test"]["labels"], dtype=np.int64)
                    session_ids = np.asarray(data["test"]["session_ids"])
                    record = {
                        "model": MODEL_NAME,
                        "subject": subject,
                        "seed": seed,
                        "deterministic": True,
                        "protocol": protocol_metadata(subject),
                        "data_sha256": data_sha256,
                        "metrics": _metrics(truth, probabilities),
                        "per_session_metrics": _session_metrics(
                            truth, probabilities, session_ids
                        ),
                        "fit": fit,
                        "parity_diagnostics": parity,
                        "filter_diagnostics": filters,
                        "predictions": _prediction_trace(
                            truth, probabilities, session_ids
                        ),
                    }
                    payload["records"].append(record)
                    payload["updated_at"] = _utc_now()
                    payload["summary"] = _summarize(payload)
                    _update_completion(payload, status="running")
                    _atomic_json(output, payload)
                    print(
                        f"{MODEL_NAME:12s} {mode} S{subject} seed={seed}: "
                        f"bacc={100.0 * record['metrics']['balanced_accuracy']:.2f}% "
                        f"({fit['wall_seconds']:.1f}s)",
                        flush=True,
                    )
        except BaseException as error:
            payload["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "at": _utc_now(),
            }
            payload["updated_at"] = _utc_now()
            _update_completion(payload, status="interrupted")
            _atomic_json(output, payload)
            if receipt_path is not None and receipt is not None:
                _update_confirmation_receipt(
                    receipt_path, receipt, state="interrupted", output=output, error=error
                )
            raise
        payload.pop("failure", None)
        payload["updated_at"] = _utc_now()
        payload["summary"] = _summarize(payload)
        _update_completion(payload, status="complete")
        _atomic_json(output, payload)
        if receipt_path is not None and receipt is not None:
            _update_confirmation_receipt(
                receipt_path, receipt, state="complete", output=output
            )
        return payload


def _load_config(path: Path | None) -> HemiQFieldConfig:
    if path is None:
        return HemiQFieldConfig()
    payload_bytes = Path(path).read_bytes()
    return _config_from_mapping(json.loads(payload_bytes))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--subjects")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--confirmation-token")
    parser.add_argument("--write-frozen-manifest", type=Path)
    parser.add_argument("--confirmation-output", type=Path)
    args = parser.parse_args(argv)

    defaults = DEVELOPMENT_SUBJECTS if args.mode == "bnci004-dev" else CONFIRMATION_SUBJECTS
    subjects = list(defaults) if args.subjects is None else _parse_ints(args.subjects)
    seeds = _parse_ints(args.seeds)
    config = _load_config(args.config_json)
    if args.write_frozen_manifest is not None:
        if args.mode != "bnci004-dev":
            raise ValueError("--write-frozen-manifest is only valid with bnci004-dev")
        if subjects != list(DEVELOPMENT_SUBJECTS) or seeds != [7]:
            raise ValueError("a frozen manifest requires exact dev subjects 1--4 and seed 7")
        if args.confirmation_output is None:
            raise ValueError("--write-frozen-manifest requires --confirmation-output")
        if args.write_frozen_manifest.resolve() == args.output.resolve():
            raise ValueError("result and manifest outputs must differ")
    elif args.confirmation_output is not None:
        raise ValueError("--confirmation-output requires --write-frozen-manifest")
    payload = run(
        mode=args.mode,
        subjects=subjects,
        seeds=seeds,
        config=config,
        device=args.device,
        output=args.output,
        resume=args.resume,
        frozen_manifest=args.frozen_manifest,
        confirmation_token=args.confirmation_token,
    )
    if args.write_frozen_manifest is not None:
        write_frozen_manifest(
            args.output,
            args.write_frozen_manifest,
            confirmation_output=args.confirmation_output,
            device=args.device,
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIRMATION_RECEIPT_PATH",
    "CONFIRMATION_STUDY_ID",
    "CONFIRMATION_TOKEN",
    "EXPECTED_SPLITS",
    "MODEL_CONTRACT",
    "MODES",
    "run",
    "write_frozen_manifest",
]
