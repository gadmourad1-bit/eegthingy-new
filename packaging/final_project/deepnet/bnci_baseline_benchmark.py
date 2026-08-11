"""Fixed, source-only baselines for the locked BNCI2014-001 protocol.

This runner is the matched reference for :mod:`deepnet.parity_benchmark`.  It
uses exactly the partitions returned by :func:`load_bnci2014_subject`:

* official ``T`` runs 0--4 fit every estimator;
* official ``T`` run 5 is used only for neural checkpoint selection;
* official ``E`` is scored once and is never used for fitting, normalization,
  hyperparameter selection, or unlabeled target calibration.

The defaults are the strongest fixed references established by the project's
earlier local/Cho2017 work: ShallowConvNet with 0.5 left/right *training*
augmentation, non-transductive Riemannian tangent logistic regression, and the
convex log-Euclidean tangent anchor.  ``bnci-confirm`` must be named explicitly;
the default development mode can access only subjects 1--4.

Example development run::

    python -m deepnet.bnci_baseline_benchmark \
        --mode bnci-dev --device cuda \
        --output deepnet/results/parity/bnci_fixed_baselines_dev.json

Do not invoke ``bnci-confirm`` until the candidate architecture and this
baseline contract have been frozen.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from .config import PROJECT_ROOT, SFREQ
from .external_bnci2014 import (
    BNCI_CHANNELS,
    CONFIRMATION_SUBJECTS,
    DEVELOPMENT_SUBJECTS,
    PROTOCOL_ID,
    load_bnci2014_subject,
    protocol_metadata,
)


SCHEMA_VERSION = 1
FROZEN_MANIFEST_SCHEMA_VERSION = 1
CONFIRMATION_RECEIPT_SCHEMA_VERSION = 1
MODES = ("bnci-dev", "bnci-confirm")
BASELINES = ("shallow_swap", "riemann", "tangent_anchor")
DETERMINISTIC_BASELINES = frozenset(("riemann", "tangent_anchor"))
CONFIRMATION_TOKEN = "CONFIRM-BNCI2014-001-S5-9-BASELINES-ONCE"
CONFIRMATION_STUDY_ID = "eegthingy-fixed-baselines-bnci2014-001-s5-9-final-v1"
CONFIRMATION_RECEIPT_PATH = (
    PROJECT_ROOT
    / "deepnet"
    / "results"
    / "parity"
    / ".confirmation-receipts"
    / f"{CONFIRMATION_STUDY_ID}.json"
)

# Explicit values prevent a future change to a wrapper default from silently
# changing the frozen comparison.
BASELINE_CONTRACTS: dict[str, dict[str, Any]] = {
    "shallow_swap": {
        "label": "ShallowConvNet + left/right training augmentation",
        "implementation": "deepnet.dnn_baselines.TorchEEGClassifier[shallow]",
        "input": "broadband",
        "fit_split": "train",
        "validation_role": "early_stopping_only",
        "source_only_standardization": True,
        "left_right_training_probability": 0.5,
        "test_time_augmentation": False,
        "unlabeled_test_calibration": False,
        "epochs": 200,
        "patience": 40,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
    },
    "riemann": {
        "label": "Riemannian tangent space + logistic regression",
        "implementation": "deepnet.baselines.RiemannianTangentLogistic",
        "input": "covariances",
        "fit_split": "train",
        "validation_role": "unused_fixed_estimator",
        "source_only_alignment": True,
        "unlabeled_test_calibration": False,
        "C": 1.0,
        "max_iter": 3000,
    },
    "tangent_anchor": {
        "label": "Frozen log-Euclidean tangent anchor",
        "implementation": "deepnet.tangent_anchor.TangentAnchorClassifier",
        "input": "covariances",
        "fit_split": "train",
        "validation_role": "unused_fixed_estimator",
        "source_only_reference_and_scaler": True,
        "unlabeled_test_calibration": False,
        "C": 1.0,
        "max_iter": 2000,
    },
}


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
        for name in ("numpy", "scikit-learn", "torch", "braindecode", "pyriemann", "mne", "moabb")
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


_SOURCE_FILES = (
    "deepnet/__init__.py",
    "deepnet/bnci_baseline_benchmark.py",
    "deepnet/config.py",
    "deepnet/external_bnci2014.py",
    "deepnet/data.py",
    "deepnet/dnn_baselines.py",
    "deepnet/baselines.py",
    "deepnet/tangent_anchor.py",
    "deepnet/augment.py",
    "deepnet/spd.py",
)


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


def _sanitized_command() -> list[str]:
    command = list(sys.argv)
    for index, value in enumerate(command):
        if value.startswith("--confirmation-token="):
            command[index] = "--confirmation-token=<redacted>"
            continue
        if index == len(command) - 1:
            continue
        if value == "--confirmation-token":
            command[index + 1] = "<redacted>"
    return command


def _configure_torch_determinism() -> None:
    """Configure strict repeatability before a confirmation CUDA context exists."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


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


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _data_manifest(data: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, str]]:
    return {
        split: {name: _array_sha256(values) for name, values in arrays.items()}
        for split, arrays in data.items()
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _exclusive_json_create(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a JSON file with ``O_EXCL`` so competing claims cannot both win."""

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
    """Non-stale process lock for one exact artifact path."""

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
            raise RuntimeError(
                f"another process holds the result lock: {self.path}"
            ) from error
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


def _validate_contract(
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    baselines: Sequence[str],
) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    allowed = set(DEVELOPMENT_SUBJECTS if mode == "bnci-dev" else CONFIRMATION_SUBJECTS)
    if not subjects or not set(subjects).issubset(allowed):
        raise ValueError(
            f"{mode} subjects must stay inside the locked partition "
            f"{min(allowed)}--{max(allowed)}"
        )
    if len(set(subjects)) != len(subjects):
        raise ValueError("subjects must be unique")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    unknown = set(baselines) - set(BASELINES)
    if not baselines or unknown:
        raise ValueError(f"unknown baselines {sorted(unknown)}; choose from {BASELINES}")
    if len(set(baselines)) != len(baselines):
        raise ValueError("baselines must be unique")


def _seeds_for_baseline(baseline: str, seeds: Sequence[int]) -> Sequence[int]:
    return seeds[:1] if baseline in DETERMINISTIC_BASELINES else seeds


def _validate_probabilities(probabilities: np.ndarray, n_expected: int) -> np.ndarray:
    result = np.asarray(probabilities, dtype=np.float64)
    if result.shape != (n_expected, 2):
        raise ValueError(
            f"expected probabilities with shape {(n_expected, 2)}, got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("class probabilities do not sum to one")
    return result


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])),
    }


def _prediction_trace(
    labels: np.ndarray, probabilities: np.ndarray, run_ids: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "window_index": int(index),
            "run_id": str(run_id),
            "label": int(label),
            "probability_left": float(probability[0]),
            "probability_right": float(probability[1]),
        }
        for index, (label, probability, run_id) in enumerate(
            zip(labels, probabilities, run_ids, strict=True)
        )
    ]


def _fit_predict(
    baseline: str,
    data: Mapping[str, Mapping[str, np.ndarray]],
    *,
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one fixed baseline without exposing any test data to ``fit``."""

    training = data["train"]
    validation = data["validation"]
    test = data["test"]
    started = time.perf_counter()

    if baseline == "shallow_swap":
        from .dnn_baselines import TorchEEGClassifier

        raw_train = training["broadband"]
        classifier = TorchEEGClassifier(
            "shallow",
            n_times=raw_train.shape[-1],
            sfreq=SFREQ,
            n_epochs=200,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=64,
            patience=40,
            seed=seed,
            device=device,
            lr_swap_prob=0.5,
            channels=BNCI_CHANNELS,
        )
        classifier.fit(
            raw_train,
            training["labels"],
            validation["broadband"],
            validation["labels"],
        )
        probabilities = classifier.predict_proba(test["broadband"])
        detail = {
            "parameter_count": int(classifier.param_count_),
            "epochs_run": int(classifier.epochs_run_),
            "fit_seconds": float(classifier.fit_seconds_),
            "validation_role": "early_stopping_only",
        }
    elif baseline == "riemann":
        from .baselines import RiemannianTangentLogistic

        classifier = RiemannianTangentLogistic(c=1.0, max_iter=3000).fit(
            training["covariances"], training["labels"]
        )
        # Deliberately no classifier.calibrate(test["covariances"]): this is
        # the non-transductive, source-only reference.
        probabilities = classifier.predict_proba(test["covariances"])
        detail = {
            "parameter_count": 0,
            "validation_role": "unused_fixed_estimator",
            "fit_seconds": float(time.perf_counter() - started),
        }
    elif baseline == "tangent_anchor":
        from .tangent_anchor import TangentAnchorClassifier

        classifier = TangentAnchorClassifier(C=1.0, max_iter=2000).fit(
            training["covariances"], training["labels"]
        )
        probabilities = classifier.predict_proba(test["covariances"])
        detail = {
            "parameter_count": int(classifier.param_count_),
            "validation_role": "unused_fixed_estimator",
            "fit_seconds": float(time.perf_counter() - started),
        }
    else:  # pragma: no cover - guarded by _validate_contract
        raise ValueError(baseline)

    detail.update(
        {
            "n_train": int(len(training["labels"])),
            "n_validation": int(len(validation["labels"])),
            "n_test": int(len(test["labels"])),
            "train_run_ids": sorted(set(training["run_ids"].astype(str).tolist())),
            "validation_run_ids": sorted(set(validation["run_ids"].astype(str).tolist())),
            "test_run_ids": sorted(set(test["run_ids"].astype(str).tolist())),
            "source_only_preprocessing": True,
            "unlabeled_test_calibration": False,
            "wall_seconds": float(time.perf_counter() - started),
        }
    )
    return _validate_probabilities(probabilities, len(test["labels"])), detail


def _expected_record_keys(
    subjects: Sequence[int], seeds: Sequence[int], baselines: Sequence[str]
) -> list[list[Any]]:
    return [
        [baseline, int(subject), int(seed)]
        for subject in subjects
        for baseline in baselines
        for seed in _seeds_for_baseline(baseline, seeds)
    ]


def _actual_record_keys(payload: Mapping[str, Any]) -> list[list[Any]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("result records must be a list")
    result: list[list[Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("result records must be objects")
        try:
            result.append(
                [
                    str(record["baseline"]),
                    int(record["subject"]),
                    int(record["seed"]),
                ]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("result record identity is invalid") from error
    if len(result) != len({tuple(key) for key in result}):
        raise ValueError("result contains duplicate record identities")
    return result


def _update_completion(payload: dict[str, Any], *, status: str) -> None:
    expected = _expected_record_keys(
        payload["subjects"], payload["seeds"], payload["baselines"]
    )
    actual = _actual_record_keys(payload)
    if any(key not in expected for key in actual):
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
    baselines: Sequence[str],
    device: str,
    frozen_manifest_sha256: str | None,
) -> dict[str, Any]:
    return {
        "dataset": "BNCI2014-001",
        "protocol_id": PROTOCOL_ID,
        "mode": mode,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "baselines": list(baselines),
        "device": device,
        "baseline_contracts": {
            name: BASELINE_CONTRACTS[name] for name in baselines
        },
        "frozen_manifest_sha256": frozen_manifest_sha256,
    }


def _new_payload(
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    baselines: Sequence[str],
    device: str,
    *,
    environment: Mapping[str, Any] | None = None,
    frozen_manifest: Mapping[str, Any] | None = None,
    frozen_manifest_path: Path | None = None,
    frozen_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    contract = _experiment_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        baselines=baselines,
        device=device,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": None,
        **contract,
        "contract_sha256": _json_sha256(contract),
        "confirmation_access": mode == "bnci-confirm",
        "confirmation": (
            {
                "token_verified": True,
                "frozen_manifest_path": str(frozen_manifest_path.resolve()),
                "frozen_manifest_sha256": frozen_manifest_sha256,
                "development_artifact": frozen_manifest["development_artifact"],
                "confirmation_contract": frozen_manifest["confirmation_contract"],
            }
            if frozen_manifest is not None and frozen_manifest_path is not None
            else None
        ),
        "command": _sanitized_command(),
        "repository": _git_state(),
        "source_sha256": _source_manifest(),
        "environment": dict(environment or _run_environment()),
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
    baselines: Sequence[str],
    device: str,
    resume: bool,
    environment: Mapping[str, Any],
    frozen_manifest: Mapping[str, Any] | None,
    frozen_manifest_path: Path | None,
    frozen_manifest_sha256: str | None,
) -> dict[str, Any]:
    expected_contract = _experiment_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        baselines=baselines,
        device=device,
        frozen_manifest_sha256=frozen_manifest_sha256,
    )
    if not output.exists():
        if resume and frozen_manifest is None:
            raise FileNotFoundError(f"--resume requires an existing result: {output}")
        return _new_payload(
            mode,
            subjects,
            seeds,
            baselines,
            device,
            environment=environment,
            frozen_manifest=frozen_manifest,
            frozen_manifest_path=frozen_manifest_path,
            frozen_manifest_sha256=frozen_manifest_sha256,
        )
    if not resume:
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    observed_contract = {key: payload.get(key) for key in expected_contract}
    if observed_contract != expected_contract:
        raise ValueError("cannot resume an output with a different experiment contract")
    if payload.get("contract_sha256") != _json_sha256(expected_contract):
        raise ValueError("cannot resume an output with a corrupt contract digest")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("cannot resume an unsupported result schema")
    if payload.get("source_sha256") != _source_manifest():
        raise ValueError("cannot resume after benchmark source files have changed")
    if payload.get("environment") != dict(environment):
        raise ValueError("cannot resume in a different execution environment")
    _update_completion(payload, status="running")
    return payload


def _summarize(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for baseline in payload["baselines"]:
        rows = [row for row in payload["records"] if row["baseline"] == baseline]
        by_subject: dict[int, list[float]] = {}
        for row in rows:
            by_subject.setdefault(int(row["subject"]), []).append(
                float(row["metrics"]["balanced_accuracy"])
            )
        values = np.asarray(
            [np.mean(subject_rows) for subject_rows in by_subject.values()], dtype=np.float64
        )
        summary[baseline] = {
            "balanced_accuracy_mean": float(values.mean()) if len(values) else None,
            "balanced_accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_participants": int(len(values)),
            "n_records": int(len(rows)),
            "parameter_count": int(np.median([row["fit"]["parameter_count"] for row in rows]))
            if rows
            else None,
            "fit_seconds_mean": float(np.mean([row["fit"]["fit_seconds"] for row in rows]))
            if rows
            else None,
        }
    return summary


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


def _validate_development_record(
    record: Mapping[str, Any], *, baseline: str, subject: int
) -> None:
    if (
        record.get("baseline") != baseline
        or record.get("subject") != subject
        or record.get("seed") != 7
        or record.get("deterministic")
        is not (baseline in DETERMINISTIC_BASELINES)
    ):
        raise ValueError("development artifact record identity is invalid")
    if record.get("protocol") != protocol_metadata(subject):
        raise ValueError(f"development S{subject} protocol metadata is invalid")
    _validate_hash_tree(record.get("data_sha256"), name=f"S{subject}.data_sha256")

    predictions = record.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 144:
        raise ValueError(f"development S{subject} predictions are incomplete")
    labels: list[int] = []
    probabilities: list[list[float]] = []
    run_ids: list[str] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict) or prediction.get("window_index") != index:
            raise ValueError(f"development S{subject} prediction order is invalid")
        try:
            label = int(prediction["label"])
            probability = [
                float(prediction["probability_left"]),
                float(prediction["probability_right"]),
            ]
            run_id = str(prediction["run_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"development S{subject} prediction payload is invalid"
            ) from error
        if label not in (0, 1):
            raise ValueError(f"development S{subject} has a non-binary label")
        labels.append(label)
        probabilities.append(probability)
        run_ids.append(run_id)
    probability_array = _validate_probabilities(
        np.asarray(probabilities, dtype=np.float64), len(predictions)
    )
    if sorted(set(run_ids)) != ["0", "1", "2", "3", "4", "5"]:
        raise ValueError(f"development S{subject} test runs are invalid")
    recomputed = _metrics(np.asarray(labels, dtype=np.int64), probability_array)
    observed_metrics = record.get("metrics")
    if not isinstance(observed_metrics, dict):
        raise ValueError(f"development S{subject} metrics are missing")
    for key, expected_value in recomputed.items():
        try:
            observed_value = float(observed_metrics[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"development S{subject} metric {key} is invalid"
            ) from error
        if not math.isclose(
            observed_value, expected_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"development S{subject} metric {key} does not recompute")

    fit = record.get("fit")
    expected_fit = {
        "n_train": 120,
        "n_validation": 24,
        "n_test": 144,
        "train_run_ids": ["0", "1", "2", "3", "4"],
        "validation_run_ids": ["5"],
        "test_run_ids": ["0", "1", "2", "3", "4", "5"],
        "source_only_preprocessing": True,
        "unlabeled_test_calibration": False,
        "validation_role": BASELINE_CONTRACTS[baseline]["validation_role"],
    }
    if not isinstance(fit, dict) or any(
        fit.get(key) != value for key, value in expected_fit.items()
    ):
        raise ValueError(f"development S{subject} fit contract is invalid")


def _validate_development_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("development artifact schema is unsupported")
    expected_top_level = {
        "dataset": "BNCI2014-001",
        "protocol_id": PROTOCOL_ID,
        "mode": "bnci-dev",
        "confirmation_access": False,
        "subjects": list(DEVELOPMENT_SUBJECTS),
        "seeds": [7],
        "baselines": list(BASELINES),
        "baseline_contracts": BASELINE_CONTRACTS,
    }
    if any(payload.get(key) != value for key, value in expected_top_level.items()):
        raise ValueError("development artifact does not match the frozen baseline contract")
    _validate_hash_tree(payload.get("source_sha256"), name="source_sha256")
    records = payload.get("records")
    expected_identities = [
        (baseline, subject, 7)
        for subject in DEVELOPMENT_SUBJECTS
        for baseline in BASELINES
    ]
    if not isinstance(records, list) or len(records) != len(expected_identities):
        raise ValueError("development artifact has the wrong record count")
    observed_identities: list[tuple[str, int, int]] = []
    subject_data_digests: dict[int, Any] = {}
    for record, (baseline, subject, seed) in zip(
        records, expected_identities, strict=True
    ):
        if not isinstance(record, dict):
            raise ValueError("development artifact record must be an object")
        identity = (
            str(record.get("baseline")),
            int(record.get("subject", -1)),
            int(record.get("seed", -1)),
        )
        observed_identities.append(identity)
        if identity != (baseline, subject, seed):
            raise ValueError("development artifact record order is invalid")
        _validate_development_record(record, baseline=baseline, subject=subject)
        previous = subject_data_digests.setdefault(subject, record["data_sha256"])
        if previous != record["data_sha256"]:
            raise ValueError(f"development S{subject} baselines used different data")
    if len(set(observed_identities)) != len(observed_identities):
        raise ValueError("development artifact has duplicate record identities")
    if payload.get("summary") != _summarize(payload):
        raise ValueError("development artifact summary does not recompute")


def _confirmation_manifest_contract(output: Path, *, device: str) -> dict[str, Any]:
    return {
        "study_id": CONFIRMATION_STUDY_ID,
        "dataset": "BNCI2014-001",
        "protocol_id": PROTOCOL_ID,
        "subjects": list(CONFIRMATION_SUBJECTS),
        "seeds": [7],
        "baselines": list(BASELINES),
        "device": device,
        "output_path": str(output.expanduser().resolve()),
        "receipt_path": str(Path(CONFIRMATION_RECEIPT_PATH).expanduser().resolve()),
    }


def _resolve_development_artifact(manifest_path: Path, value: str) -> Path:
    artifact = Path(value).expanduser()
    if artifact.is_absolute():
        return artifact.resolve()
    return (manifest_path.parent / artifact).resolve()


def write_frozen_manifest(
    development_artifact: Path,
    output: Path,
    *,
    confirmation_output: Path,
    device: str,
) -> dict[str, Any]:
    """Freeze fixed baselines and one exact future S5--9 confirmation output."""

    development_artifact = Path(development_artifact)
    output = Path(output)
    confirmation_output = Path(confirmation_output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output}")
    if Path(CONFIRMATION_RECEIPT_PATH).exists():
        raise FileExistsError(
            "permanent baseline confirmation receipt already exists; cannot refreeze"
        )
    if not development_artifact.is_file():
        raise FileNotFoundError(development_artifact)
    development_bytes = development_artifact.read_bytes()
    payload = json.loads(development_bytes)
    _validate_development_payload(payload)
    current_source = _source_manifest()
    contract = _confirmation_manifest_contract(confirmation_output, device=device)
    reserved_paths = {
        output.expanduser().resolve(),
        development_artifact.expanduser().resolve(),
        Path(contract["receipt_path"]),
    }
    if Path(contract["output_path"]) in reserved_paths:
        raise ValueError("confirmation output must be distinct from frozen artifacts")
    if Path(contract["output_path"]).exists():
        raise FileExistsError("pinned confirmation output already exists")
    manifest = {
        "schema_version": FROZEN_MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "study_id": CONFIRMATION_STUDY_ID,
        "source_sha256": current_source,
        "baseline_contracts": BASELINE_CONTRACTS,
        "confirmation_contract": contract,
        "development_artifact": {
            "path": str(development_artifact.resolve()),
            "sha256": hashlib.sha256(development_bytes).hexdigest(),
            "summary_sha256": _json_sha256(payload["summary"]),
            "source_sha256": payload["source_sha256"],
            "mode": "bnci-dev",
            "subjects": list(DEVELOPMENT_SUBJECTS),
            "seeds": [7],
            "baselines": list(BASELINES),
            "expected_records": len(DEVELOPMENT_SUBJECTS) * len(BASELINES),
        },
    }
    _exclusive_json_create(output, manifest)
    return manifest


def _read_and_validate_frozen_manifest(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None:
        raise ValueError("bnci-confirm requires --frozen-manifest")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen manifest does not exist: {path}")
    manifest_bytes = path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != FROZEN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported frozen-manifest schema")
    if manifest.get("study_id") != CONFIRMATION_STUDY_ID:
        raise ValueError("frozen manifest names a different study")
    if manifest.get("source_sha256") != _source_manifest():
        raise ValueError("confirmation source differs from the frozen manifest")
    if manifest.get("baseline_contracts") != BASELINE_CONTRACTS:
        raise ValueError("confirmation baseline contracts differ from the manifest")
    contract = manifest.get("confirmation_contract")
    expected_receipt = str(Path(CONFIRMATION_RECEIPT_PATH).expanduser().resolve())
    if not isinstance(contract, dict) or any(
        contract.get(key) != value
        for key, value in {
            "study_id": CONFIRMATION_STUDY_ID,
            "dataset": "BNCI2014-001",
            "protocol_id": PROTOCOL_ID,
            "subjects": list(CONFIRMATION_SUBJECTS),
            "seeds": [7],
            "baselines": list(BASELINES),
            "receipt_path": expected_receipt,
        }.items()
    ):
        raise ValueError("frozen manifest confirmation contract is invalid")
    if (
        not isinstance(contract.get("device"), str)
        or not contract["device"]
        or not isinstance(contract.get("output_path"), str)
        or not Path(contract["output_path"]).is_absolute()
    ):
        raise ValueError("frozen manifest device/output contract is invalid")

    development = manifest.get("development_artifact")
    if not isinstance(development, dict) or not isinstance(
        development.get("path"), str
    ):
        raise ValueError("frozen manifest lacks a development artifact")
    artifact_path = _resolve_development_artifact(path, development["path"])
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
        "mode": "bnci-dev",
        "subjects": list(DEVELOPMENT_SUBJECTS),
        "seeds": [7],
        "baselines": list(BASELINES),
        "expected_records": len(DEVELOPMENT_SUBJECTS) * len(BASELINES),
    }
    if any(development.get(key) != value for key, value in expected_development.items()):
        raise ValueError("frozen development metadata is invalid")
    return manifest, manifest_sha256


def _validate_confirmation_contract(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    baselines: Sequence[str],
    device: str,
    confirmation_token: str | None,
    frozen_manifest: Path | None,
) -> tuple[dict[str, Any], str] | None:
    if mode != "bnci-confirm":
        if confirmation_token is not None or frozen_manifest is not None:
            raise ValueError(
                "confirmation token/manifest may only be used with bnci-confirm"
            )
        return None
    if list(subjects) != list(CONFIRMATION_SUBJECTS):
        raise ValueError("bnci-confirm requires the exact ordered subjects 5--9")
    if list(seeds) != [7]:
        raise ValueError("bnci-confirm requires the sole frozen seed 7")
    if list(baselines) != list(BASELINES):
        raise ValueError("bnci-confirm requires all fixed baselines in frozen order")
    if confirmation_token != CONFIRMATION_TOKEN:
        raise ValueError("bnci-confirm requires the exact explicit confirmation token")
    manifest, manifest_sha256 = _read_and_validate_frozen_manifest(frozen_manifest)
    if manifest["confirmation_contract"]["device"] != device:
        raise ValueError("confirmation device differs from the frozen manifest")
    return manifest, manifest_sha256


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
        "baseline_contracts": manifest["baseline_contracts"],
        "subjects": list(CONFIRMATION_SUBJECTS),
        "seeds": [7],
        "baselines": list(BASELINES),
        "device": contract["device"],
        "environment": dict(environment),
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
    pinned_output = Path(contract["output_path"])
    if output.expanduser().resolve() != pinned_output:
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
            raise FileExistsError(
                "permanent baseline confirmation receipt already exists; refusing re-access"
            )
        observed = json.loads(receipt_path.read_text(encoding="utf-8"))
        if any(observed.get(key) != expected[key] for key in immutable_keys):
            raise ValueError(
                "confirmation receipt does not match the exact frozen contract/environment"
            )
        state = observed.get("state")
        if state == "complete":
            if not output.is_file() or observed.get("output_sha256") != _file_sha256(output):
                raise ValueError("completed confirmation output no longer matches its receipt")
            raise FileExistsError(
                "baseline confirmation is already complete; refusing reevaluation"
            )
        if state not in {"claimed", "running", "interrupted"}:
            raise ValueError("confirmation receipt has an invalid state")
        if state != "claimed" and not output.is_file():
            raise FileNotFoundError(
                "confirmation resume requires the pinned interrupted result artifact"
            )
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
        raise RuntimeError("permanent baseline confirmation receipt disappeared")
    updated = dict(receipt)
    updated["state"] = state
    updated["updated_at"] = _utc_now()
    if error is not None:
        updated["last_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
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
    baselines: Sequence[str],
    device: str,
    output: Path,
    resume: bool = False,
    frozen_manifest: Path | None = None,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Run an allowed BNCI cohort; partition checks precede every data access."""

    subjects = [int(value) for value in subjects]
    seeds = [int(value) for value in seeds]
    baselines = [str(value) for value in baselines]
    output = Path(output)
    if mode == "bnci-confirm":
        _configure_torch_determinism()
    _validate_contract(mode, subjects, seeds, baselines)
    validated_manifest = _validate_confirmation_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        baselines=baselines,
        device=device,
        confirmation_token=confirmation_token,
        frozen_manifest=frozen_manifest,
    )
    environment = _run_environment()
    manifest: Mapping[str, Any] | None = None
    manifest_sha256: str | None = None
    receipt_path: Path | None = None
    receipt: dict[str, Any] | None = None
    if validated_manifest is not None:
        manifest, manifest_sha256 = validated_manifest
        receipt_path, receipt = _claim_confirmation_receipt(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            environment=environment,
            output=output,
            resume=resume,
        )

    with _OutputLock(output):
        payload = _load_or_initialize(
            output,
            mode=mode,
            subjects=subjects,
            seeds=seeds,
            baselines=baselines,
            device=device,
            resume=resume,
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
                (str(row["baseline"]), int(row["subject"]), int(row["seed"]))
                for row in payload["records"]
            }
            for subject in subjects:
                pending = [
                    (baseline, seed)
                    for baseline in baselines
                    for seed in _seeds_for_baseline(baseline, seeds)
                    if (baseline, int(subject), int(seed)) not in completed
                ]
                if not pending:
                    continue
                data = load_bnci2014_subject(subject)
                data_manifest = _data_manifest(data)
                for baseline, seed in pending:
                    probabilities, detail = _fit_predict(
                        baseline, data, seed=int(seed), device=device
                    )
                    truth = np.asarray(data["test"]["labels"])
                    record = {
                        "baseline": baseline,
                        "subject": int(subject),
                        "seed": int(seed),
                        "deterministic": baseline in DETERMINISTIC_BASELINES,
                        "protocol": protocol_metadata(subject),
                        "data_sha256": data_manifest,
                        "metrics": _metrics(truth, probabilities),
                        "fit": detail,
                        "predictions": _prediction_trace(
                            truth, probabilities, data["test"]["run_ids"]
                        ),
                    }
                    payload["records"].append(record)
                    payload["updated_at"] = _utc_now()
                    payload["summary"] = _summarize(payload)
                    _update_completion(payload, status="running")
                    _atomic_json(output, payload)
                    print(
                        f"{baseline:15s} {mode} S{subject} seed={seed}: "
                        f"bacc={100.0 * record['metrics']['balanced_accuracy']:.2f}% "
                        f"({detail['wall_seconds']:.1f}s)",
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
                    receipt_path,
                    receipt,
                    state="interrupted",
                    output=output,
                    error=error,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--baselines", default=",".join(BASELINES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--confirmation-token")
    parser.add_argument("--write-frozen-manifest", type=Path)
    parser.add_argument(
        "--confirmation-output",
        type=Path,
        help="exact future S5--9 baseline result path pinned into a new manifest",
    )
    args = parser.parse_args(argv)

    default_subjects = (
        DEVELOPMENT_SUBJECTS if args.mode == "bnci-dev" else CONFIRMATION_SUBJECTS
    )
    subjects = list(default_subjects) if args.subjects is None else _parse_ints(args.subjects)
    seeds = _parse_ints(args.seeds)
    baselines = [name.strip() for name in args.baselines.split(",") if name.strip()]
    if args.write_frozen_manifest is not None:
        if args.mode != "bnci-dev":
            raise ValueError("--write-frozen-manifest is only valid with bnci-dev")
        if (
            subjects != list(DEVELOPMENT_SUBJECTS)
            or seeds != [7]
            or baselines != list(BASELINES)
        ):
            raise ValueError(
                "a frozen baseline manifest requires exact BNCI development "
                "subjects 1--4, sole seed 7, and all baselines in frozen order"
            )
        if args.confirmation_output is None:
            raise ValueError("--write-frozen-manifest requires --confirmation-output")
        if args.write_frozen_manifest.resolve() == args.output.resolve():
            raise ValueError("result and frozen-manifest outputs must differ")
        if args.write_frozen_manifest.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen manifest: {args.write_frozen_manifest}"
            )
    elif args.confirmation_output is not None:
        raise ValueError("--confirmation-output requires --write-frozen-manifest")
    payload = run(
        mode=args.mode,
        subjects=subjects,
        seeds=seeds,
        baselines=baselines,
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
    "BASELINES",
    "BASELINE_CONTRACTS",
    "CONFIRMATION_RECEIPT_PATH",
    "CONFIRMATION_STUDY_ID",
    "CONFIRMATION_TOKEN",
    "MODES",
    "run",
    "write_frozen_manifest",
]
