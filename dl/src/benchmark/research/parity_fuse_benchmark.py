"""Leakage-controlled benchmark runner for :mod:`benchmark.research.parity_fuse_net`.

The runner keeps architecture development and the one-shot BNCI2014-001
confirmation cohort separate.  Development may use the first three local
recordings, Cho2017 subjects 1--26, and BNCI2014-001 subjects 1--4.  The local
loader is called with recordings 1--3 only, so recording 4 is not merely masked
after loading.  BNCI confirmation is restricted to subjects 5--9, seed 7, and
an exact configuration/source snapshot frozen from a completed four-subject
BNCI development artifact.

Confirmation requires all of the following before its loader can be called:

* the exact ordered subject cohort 5,6,7,8,9 and the sole seed 7;
* the explicit :data:`CONFIRMATION_TOKEN`;
* ``--frozen-manifest`` whose configuration and source hashes match this run;
* an unchanged, completed BNCI development artifact whose file hash is pinned
  by that manifest; and
* an exclusively locked output path.

Results are created without overwriting, checkpointed atomically after every
record, and may be resumed only under the identical experiment contract and
source snapshot.  Each record contains ordered per-window provenance,
probabilities, anchor and residual outputs, data hashes, exact signed-action
and gate-invariance errors, package versions, parameter counts, and explicit
expected/actual completion state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .cameo_benchmark import _git_state, _stratified_val
from .config import DEFAULT_DATA_CONFIG, PROJECT_ROOT, SUBJECT_RUNS, VALID_SUBJECTS
from .data import SessionKey, load_sessions
from .external_bnci2014 import (
    BNCI_CHANNELS,
    CONFIRMATION_SUBJECTS,
    DEVELOPMENT_SUBJECTS,
    PROTOCOL_ID,
    load_bnci2014_subject,
    protocol_metadata,
)
from .external_cho2017 import load_cho_subject
from .parity_fuse_net import ParityFuseClassifier, ParityFuseConfig, ParityFuseOutput


SCHEMA_VERSION = 1
FROZEN_MANIFEST_SCHEMA_VERSION = 1
ARCHITECTURE = "PARITY-Fuse"
MODES = ("local-dev", "cho-dev", "bnci-dev", "bnci-confirm")
CONFIRMATION_TOKEN = "CONFIRM-BNCI2014-001-S5-9-ONCE"
PROTOCOLS = {
    "local-dev": "local_rec1-fit_rec2-select_rec3-development_rec4-not-loaded",
    "cho-dev": "cho2017_s1-26_inner-select_outer-5fold-development",
    "bnci-dev": "bnci2014-001_s1-4_T-runs0-4-fit_T-run5-select_E-development",
    "bnci-confirm": "bnci2014-001_s5-9_T-runs0-4-fit_T-run5-select_E-confirmation",
}

_SOURCE_FILES = (
    "src/benchmark/research/parity_fuse_benchmark.py",
    "src/benchmark/research/parity_fuse_net.py",
    "src/benchmark/research/cameo_net.py",
    "src/benchmark/shared/augment.py",
    "src/benchmark/research/config.py",
    "src/benchmark/research/data.py",
    "src/benchmark/research/external_cho2017.py",
    "src/benchmark/research/external_bnci2014.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _array_manifest(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "sha256": _array_sha256(array),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _data_manifest(
    splits: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        str(split): {
            str(name): _array_manifest(np.asarray(values))
            for name, values in arrays.items()
        }
        for split, arrays in splits.items()
    }


def _source_manifest() -> dict[str, str]:
    return {
        name: _file_sha256(PROJECT_ROOT / name)
        for name in _SOURCE_FILES
    }


def _package_versions() -> dict[str, str]:
    """Return every installed distribution, not only hand-picked dependencies."""

    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if raw_name:
            versions[raw_name.lower().replace("_", "-")] = distribution.version
    return dict(sorted(versions.items()))


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "packages": _package_versions(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _json_config(config: ParityFuseConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), allow_nan=False))


def _sanitized_command() -> list[str]:
    command = list(sys.argv)
    for index, token in enumerate(command[:-1]):
        if token == "--confirmation-token":
            command[index + 1] = "<explicit-confirmation-token-provided>"
    return command


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace an already-owned result path."""

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
    """Atomically create JSON while refusing to replace an existing path."""

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
        # Hard-link publication is atomic and fails if ``path`` already exists.
        os.link(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


class _OutputLock:
    """Small exclusive lock whose acquisition always precedes output/data access."""

    def __init__(self, output: Path) -> None:
        self.path = output.with_name(output.name + ".lock")
        self.fd: int | None = None

    def __enter__(self) -> "_OutputLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"result lock already exists; refusing concurrent access: {self.path}"
            ) from error
        lock_payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": _utc_now(),
        }
        os.write(self.fd, _canonical_json(lock_payload) + b"\n")
        os.fsync(self.fd)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:  # pragma: no cover - external removal
            pass


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


def _validate_common_contract(
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("subjects must be non-empty and unique")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if folds < 2:
        raise ValueError("folds must be at least two")

    if mode == "local-dev":
        allowed = set(VALID_SUBJECTS)
    elif mode == "cho-dev":
        allowed = set(range(1, 27))
    elif mode == "bnci-dev":
        allowed = set(DEVELOPMENT_SUBJECTS)
    else:
        allowed = set(CONFIRMATION_SUBJECTS)
    if not set(subjects).issubset(allowed):
        raise ValueError(
            f"{mode} subjects must stay inside the locked partition "
            f"{min(allowed)}--{max(allowed)}"
        )


def _expected_record_keys(
    subjects: Sequence[int], seeds: Sequence[int]
) -> list[dict[str, int]]:
    return [
        {"subject": int(subject), "seed": int(seed)}
        for subject in subjects
        for seed in seeds
    ]


def _update_completion(
    payload: dict[str, Any], *, status: str | None = None
) -> None:
    actual = [
        {"subject": int(row["subject"]), "seed": int(row["seed"])}
        for row in payload["records"]
    ]
    expected = payload["completion"]["expected_record_keys"]
    if len({(x["subject"], x["seed"]) for x in actual}) != len(actual):
        raise ValueError("result contains duplicate subject/seed records")
    if not {(x["subject"], x["seed"]) for x in actual}.issubset(
        {(x["subject"], x["seed"]) for x in expected}
    ):
        raise ValueError("result contains records outside its experiment contract")
    payload["completion"].update(
        {
            "actual_record_keys": actual,
            "actual_records": len(actual),
            "complete": len(actual) == len(expected),
        }
    )
    if status is not None:
        payload["completion"]["status"] = status


def _experiment_contract(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
    config: ParityFuseConfig,
    frozen_manifest_sha256: str | None,
) -> dict[str, Any]:
    return {
        "architecture_name": ARCHITECTURE,
        "mode": mode,
        "protocol": PROTOCOLS[mode],
        "subjects": list(subjects),
        "seeds": list(seeds),
        "folds": int(folds) if mode == "cho-dev" else None,
        "config": _json_config(config),
        "frozen_manifest_sha256": frozen_manifest_sha256,
    }


def _new_payload(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
    config: ParityFuseConfig,
    frozen_manifest: Mapping[str, Any] | None,
    frozen_manifest_path: Path | None,
) -> dict[str, Any]:
    source = _source_manifest()
    manifest_hash = (
        _file_sha256(frozen_manifest_path)
        if frozen_manifest_path is not None
        else None
    )
    contract = _experiment_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        folds=folds,
        config=config,
        frozen_manifest_sha256=manifest_hash,
    )
    expected = _expected_record_keys(subjects, seeds)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": None,
        "architecture": {
            "name": ARCHITECTURE,
            "implementation": "benchmark.research.parity_fuse_net.ParityFuseNet",
            "trainable_parameter_contract_lt": 30_000,
        },
        **contract,
        "contract_sha256": _json_sha256(contract),
        "confirmation_access": mode == "bnci-confirm",
        "confirmation": (
            {
                "token_verified": True,
                "frozen_manifest_path": str(frozen_manifest_path.resolve()),
                "frozen_manifest_sha256": manifest_hash,
                "development_artifact": frozen_manifest["development_artifact"],
            }
            if frozen_manifest is not None and frozen_manifest_path is not None
            else None
        ),
        "command": _sanitized_command(),
        "repository": _git_state(),
        "source_sha256": source,
        "environment": _environment(),
        "records": [],
        "summary": {},
        "completion": {
            "status": "running",
            "expected_record_keys": expected,
            "actual_record_keys": [],
            "expected_records": len(expected),
            "actual_records": 0,
            "complete": False,
        },
    }


def _load_or_initialize(
    output: Path,
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
    config: ParityFuseConfig,
    resume: bool,
    frozen_manifest: Mapping[str, Any] | None,
    frozen_manifest_path: Path | None,
) -> dict[str, Any]:
    manifest_hash = (
        _file_sha256(frozen_manifest_path)
        if frozen_manifest_path is not None
        else None
    )
    expected_contract = _experiment_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        folds=folds,
        config=config,
        frozen_manifest_sha256=manifest_hash,
    )
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing result: {output}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        observed_contract = {
            key: payload.get(key) for key in expected_contract
        }
        if observed_contract != expected_contract:
            raise ValueError("cannot resume an output with a different experiment contract")
        if payload.get("contract_sha256") != _json_sha256(expected_contract):
            raise ValueError("cannot resume an output with a corrupt contract digest")
        if payload.get("source_sha256") != _source_manifest():
            raise ValueError("cannot resume after benchmark source files have changed")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("cannot resume an unsupported result schema")
        _update_completion(payload, status="running")
        return payload
    if resume:
        raise FileNotFoundError(f"--resume requires an existing result: {output}")
    return _new_payload(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        folds=folds,
        config=config,
        frozen_manifest=frozen_manifest,
        frozen_manifest_path=frozen_manifest_path,
    )


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), 2):
        raise ValueError("probability shape does not match labels")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities contain non-finite values")
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])),
    }


def _probabilities_from_logit(logit: np.ndarray) -> np.ndarray:
    positive = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
    return np.column_stack((1.0 - positive, positive))


def _gate_summary(weights: np.ndarray) -> dict[str, Any]:
    if weights.ndim != 3 or weights.shape[1] != 2:
        raise ValueError("fusion weights must have shape (N, 2, fusion_dim)")
    clipped = np.clip(weights, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    result: dict[str, Any] = {
        "shape": list(weights.shape),
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
    }
    for index, name in enumerate(("raw", "tangent")):
        values = weights[:, index]
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def _prediction_trace(
    labels: np.ndarray,
    probabilities: np.ndarray,
    anchor_probabilities: np.ndarray,
    output: ParityFuseOutput,
    provenance: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    labels = np.asarray(labels)
    anchor_logit = output.anchor_logit.detach().cpu().numpy()
    residual_logit = output.residual_logit.detach().cpu().numpy()
    gate = output.fusion_weights.detach().cpu().numpy()
    if len(provenance) != len(labels):
        raise ValueError("provenance length does not match predictions")
    rows: list[dict[str, Any]] = []
    for index in range(len(labels)):
        rows.append(
            {
                "window_index": int(index),
                "provenance": dict(provenance[index]),
                "label": int(labels[index]),
                "probability_left": float(probabilities[index, 0]),
                "probability_right": float(probabilities[index, 1]),
                "anchor_probability_left": float(anchor_probabilities[index, 0]),
                "anchor_probability_right": float(anchor_probabilities[index, 1]),
                "anchor_logit": float(anchor_logit[index]),
                "residual_logit": float(residual_logit[index]),
                "mean_raw_gate": float(gate[index, 0].mean()),
                "mean_tangent_gate": float(gate[index, 1].mean()),
            }
        )
    return rows


def _fit_score(
    *,
    raw_train: np.ndarray,
    cov_train: np.ndarray,
    y_train: np.ndarray,
    raw_validation: np.ndarray,
    cov_validation: np.ndarray,
    y_validation: np.ndarray,
    raw_test: np.ndarray,
    cov_test: np.ndarray,
    y_test: np.ndarray,
    channels: Sequence[str],
    config: ParityFuseConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, ParityFuseOutput]:
    classifier = ParityFuseClassifier(config).fit(
        raw_train,
        cov_train,
        y_train,
        raw_validation,
        cov_validation,
        y_validation,
        channels=channels,
    )
    output = classifier._predict_output(raw_test, cov_test)
    reflected = classifier._predict_output(raw_test, cov_test, swapped=True)
    logit = output.logit.detach().cpu().numpy()
    probabilities = _probabilities_from_logit(logit)
    anchor_probabilities = _probabilities_from_logit(
        output.anchor_logit.detach().cpu().numpy()
    )
    residual = output.residual_logit.detach().cpu().numpy()
    anchor_logit = output.anchor_logit.detach().cpu().numpy()
    gate = output.fusion_weights.detach().cpu().numpy()
    reflected_gate = reflected.fusion_weights.detach().cpu().numpy()
    detail = {
        # ``-1`` is meaningful: validation retained the exact convex anchor.
        "best_epoch": int(classifier.best_epoch_),
        "epochs_run": int(classifier.epochs_run_),
        "best_validation_loss": float(classifier.best_validation_loss_),
        "best_validation_balanced_accuracy": float(
            classifier.best_validation_balanced_accuracy_
        ),
        "parameter_count": int(classifier.param_count_),
        "trainable_parameter_count": int(classifier.trainable_param_count_),
        "train_seconds": float(classifier.train_seconds_),
        "effective_seed": int(config.seed),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_validation)),
        "n_test": int(len(y_test)),
        "equivariance": {
            "max_abs_end_to_end_logit_sum": float(
                torch.max(torch.abs(output.logit + reflected.logit)).cpu()
            ),
            "max_abs_anchor_logit_sum": float(
                torch.max(
                    torch.abs(output.anchor_logit + reflected.anchor_logit)
                ).cpu()
            ),
            "max_abs_residual_logit_sum": float(
                torch.max(
                    torch.abs(output.residual_logit + reflected.residual_logit)
                ).cpu()
            ),
            "max_abs_gate_difference": float(
                np.max(np.abs(gate - reflected_gate))
            ),
        },
        "anchor": {
            "metrics": _metrics(y_test, anchor_probabilities),
            "abs_logit_mean": float(np.mean(np.abs(anchor_logit))),
            "logit_rms": float(np.sqrt(np.mean(np.square(anchor_logit)))),
        },
        "residual": {
            "mean": float(residual.mean()),
            "std": float(residual.std()),
            "abs_mean": float(np.mean(np.abs(residual))),
            "rms": float(np.sqrt(np.mean(np.square(residual)))),
            "max_abs": float(np.max(np.abs(residual))),
        },
        "gate": _gate_summary(gate),
        "source_only_preprocessing": True,
        "unlabeled_test_calibration": False,
    }
    return detail, probabilities, anchor_probabilities, output


def _summarize(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = list(payload["records"])
    by_subject: dict[int, list[float]] = {}
    anchor_by_subject: dict[int, list[float]] = {}
    fits: list[Mapping[str, Any]] = []
    for row in records:
        subject = int(row["subject"])
        by_subject.setdefault(subject, []).append(
            float(row["metrics"]["balanced_accuracy"])
        )
        anchor_by_subject.setdefault(subject, []).append(
            float(row["anchor_metrics"]["balanced_accuracy"])
        )
        fits.extend(row["fits"])
    values = np.asarray(
        [np.mean(by_subject[key]) for key in sorted(by_subject)], dtype=np.float64
    )
    anchor_values = np.asarray(
        [np.mean(anchor_by_subject[key]) for key in sorted(anchor_by_subject)],
        dtype=np.float64,
    )
    parameter_counts = sorted({int(fit["parameter_count"]) for fit in fits})
    trainable_counts = sorted(
        {int(fit["trainable_parameter_count"]) for fit in fits}
    )
    best_epochs = [int(fit["best_epoch"]) for fit in fits]
    end_to_end_errors = [
        float(fit["equivariance"]["max_abs_end_to_end_logit_sum"])
        for fit in fits
    ]
    gate_errors = [
        float(fit["equivariance"]["max_abs_gate_difference"])
        for fit in fits
    ]
    residual_abs = [float(fit["residual"]["abs_mean"]) for fit in fits]
    raw_gate = [float(fit["gate"]["raw"]["mean"]) for fit in fits]
    return {
        "balanced_accuracy_mean": float(values.mean()) if len(values) else None,
        "balanced_accuracy_std": (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ),
        "anchor_balanced_accuracy_mean": (
            float(anchor_values.mean()) if len(anchor_values) else None
        ),
        "mean_delta_vs_anchor": (
            float((values - anchor_values).mean()) if len(values) else None
        ),
        "n_participants": int(len(values)),
        "n_records": int(len(records)),
        "n_fits": int(len(fits)),
        "parameter_counts": parameter_counts,
        "trainable_parameter_counts": trainable_counts,
        "best_epoch_counts": {
            str(epoch): best_epochs.count(epoch) for epoch in sorted(set(best_epochs))
        },
        "anchor_checkpoint_count": best_epochs.count(-1),
        "max_end_to_end_equivariance_error": (
            max(end_to_end_errors) if end_to_end_errors else None
        ),
        "max_gate_invariance_error": max(gate_errors) if gate_errors else None,
        "mean_residual_abs_logit": (
            float(np.mean(residual_abs)) if residual_abs else None
        ),
        "mean_raw_gate": float(np.mean(raw_gate)) if raw_gate else None,
    }


def _save_payload(output: Path, payload: dict[str, Any], *, status: str) -> None:
    payload["updated_at"] = _utc_now()
    payload["summary"] = _summarize(payload)
    _update_completion(payload, status=status)
    if status == "complete" and not payload["completion"]["complete"]:
        raise RuntimeError("refusing to mark an incomplete result complete")
    _atomic_json(output, payload)


def _completed_keys(payload: Mapping[str, Any]) -> set[tuple[int, int]]:
    return {
        (int(row["subject"]), int(row["seed"]))
        for row in payload["records"]
    }


def _split_manifest(
    *,
    raw: np.ndarray,
    covariances: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    metadata: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    result = {
        "broadband": raw[indices],
        "covariances": covariances[indices],
        "labels": labels[indices],
    }
    if metadata is not None:
        result.update({name: values[indices] for name, values in metadata.items()})
    return result


def _local_provenance(data: Any, indices: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "dataset": "local",
            "subject": int(data.subject_ids[row]),
            "session_id": str(data.session_ids[row]),
            "run_id": int(data.run_ids[row]),
            "trial_id": int(data.trial_ids[row]),
            "source_row": int(row),
        }
        for row in indices
    ]


def _cho_provenance(test: np.ndarray, fold: int) -> list[dict[str, Any]]:
    return [
        {
            "dataset": "Cho2017",
            "fold": int(fold),
            "source_trial_index": int(source_index),
            "fold_window_index": int(position),
        }
        for position, source_index in enumerate(test)
    ]


def _bnci_provenance(run_ids: np.ndarray) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    result: list[dict[str, Any]] = []
    for source_index, raw_run in enumerate(np.asarray(run_ids).astype(str)):
        ordinal = counters.get(raw_run, 0)
        counters[raw_run] = ordinal + 1
        result.append(
            {
                "dataset": "BNCI2014-001",
                "session": "1test",
                "run_id": raw_run,
                "trial_index_within_run": int(ordinal),
                "source_trial_index": int(source_index),
            }
        )
    return result


def _record(
    *,
    subject: int,
    seed: int,
    metrics: Mapping[str, float],
    anchor_metrics: Mapping[str, float],
    fits: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    data_sha256: Mapping[str, Any],
    protocol: Mapping[str, Any] | str,
) -> dict[str, Any]:
    return {
        "subject": int(subject),
        "seed": int(seed),
        "protocol": protocol,
        "data_sha256": data_sha256,
        "metrics": dict(metrics),
        "anchor_metrics": dict(anchor_metrics),
        "delta_balanced_accuracy_vs_anchor": float(
            metrics["balanced_accuracy"] - anchor_metrics["balanced_accuracy"]
        ),
        "fits": [dict(fit) for fit in fits],
        "predictions": [dict(row) for row in predictions],
    }


def _run_local(
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: ParityFuseConfig,
    output: Path,
    payload: dict[str, Any],
) -> None:
    completed = _completed_keys(payload)
    if all(
        (int(subject), int(seed)) in completed
        for subject in subjects
        for seed in seeds
    ):
        return
    # Physical isolation: recording 4 is never passed to ``load_sessions``.
    keys = [
        SessionKey(subject, run)
        for subject in subjects
        for run in SUBJECT_RUNS[subject][:3]
    ]
    data = load_sessions(keys, DEFAULT_DATA_CONFIG)
    task = data.labels >= 0
    for subject in subjects:
        runs = list(SUBJECT_RUNS[subject][:3])

        def rows(run: int) -> np.ndarray:
            return np.flatnonzero(
                (data.subject_ids == subject) & (data.run_ids == run) & task
            )

        train, validation, test = rows(runs[0]), rows(runs[1]), rows(runs[2])
        metadata = {
            "subject_ids": data.subject_ids,
            "session_ids": data.session_ids,
            "run_ids": data.run_ids,
            "trial_ids": data.trial_ids,
        }
        manifest = _data_manifest(
            {
                "train": _split_manifest(
                    raw=data.broadband_epochs,
                    covariances=data.covariances,
                    labels=data.labels,
                    indices=train,
                    metadata=metadata,
                ),
                "validation": _split_manifest(
                    raw=data.broadband_epochs,
                    covariances=data.covariances,
                    labels=data.labels,
                    indices=validation,
                    metadata=metadata,
                ),
                "test": _split_manifest(
                    raw=data.broadband_epochs,
                    covariances=data.covariances,
                    labels=data.labels,
                    indices=test,
                    metadata=metadata,
                ),
            }
        )
        for seed in seeds:
            if (int(subject), int(seed)) in completed:
                continue
            detail, probability, anchor_probability, model_output = _fit_score(
                raw_train=data.broadband_epochs[train],
                cov_train=data.covariances[train],
                y_train=data.labels[train],
                raw_validation=data.broadband_epochs[validation],
                cov_validation=data.covariances[validation],
                y_validation=data.labels[validation],
                raw_test=data.broadband_epochs[test],
                cov_test=data.covariances[test],
                y_test=data.labels[test],
                channels=DEFAULT_DATA_CONFIG.channels,
                config=replace(config, seed=int(seed)),
            )
            truth = data.labels[test]
            row = _record(
                subject=subject,
                seed=seed,
                metrics=_metrics(truth, probability),
                anchor_metrics=_metrics(truth, anchor_probability),
                fits=[detail],
                predictions=_prediction_trace(
                    truth,
                    probability,
                    anchor_probability,
                    model_output,
                    _local_provenance(data, test),
                ),
                data_sha256=manifest,
                protocol=PROTOCOLS["local-dev"],
            )
            payload["records"].append(row)
            _save_payload(output, payload, status="running")
            print(
                f"{ARCHITECTURE} local-dev S{subject} seed={seed}: "
                f"bacc={100.0 * row['metrics']['balanced_accuracy']:.2f}%",
                flush=True,
            )


def _run_cho(
    subjects: Sequence[int],
    seeds: Sequence[int],
    folds: int,
    config: ParityFuseConfig,
    output: Path,
    payload: dict[str, Any],
) -> None:
    completed = _completed_keys(payload)
    for subject in subjects:
        pending = [seed for seed in seeds if (int(subject), int(seed)) not in completed]
        if not pending:
            continue
        data = load_cho_subject(subject)
        raw = np.asarray(data["broadband"])
        cov = np.asarray(data["covariances"])
        labels = np.asarray(data["labels"])
        manifest = _data_manifest(
            {"all_trials_before_fixed_outer_split": {key: np.asarray(value) for key, value in data.items()}}
        )
        splits = list(
            StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=0
            ).split(np.zeros(len(labels)), labels)
        )
        for seed in pending:
            all_truth: list[np.ndarray] = []
            all_probability: list[np.ndarray] = []
            all_anchor_probability: list[np.ndarray] = []
            fits: list[dict[str, Any]] = []
            predictions: list[dict[str, Any]] = []
            trace_offset = 0
            for fold, (outer_train, test) in enumerate(splits):
                train_rel, validation_rel = _stratified_val(
                    labels[outer_train], int(seed) + fold
                )
                train = outer_train[train_rel]
                validation = outer_train[validation_rel]
                detail, probability, anchor_probability, model_output = _fit_score(
                    raw_train=raw[train],
                    cov_train=cov[train],
                    y_train=labels[train],
                    raw_validation=raw[validation],
                    cov_validation=cov[validation],
                    y_validation=labels[validation],
                    raw_test=raw[test],
                    cov_test=cov[test],
                    y_test=labels[test],
                    channels=DEFAULT_DATA_CONFIG.channels,
                    config=replace(config, seed=int(seed) + fold),
                )
                detail.update(
                    {
                        "fold": int(fold),
                        "outer_train_indices_sha256": _array_sha256(outer_train),
                        "train_indices_sha256": _array_sha256(train),
                        "validation_indices_sha256": _array_sha256(validation),
                        "test_indices_sha256": _array_sha256(test),
                    }
                )
                fold_trace = _prediction_trace(
                    labels[test],
                    probability,
                    anchor_probability,
                    model_output,
                    _cho_provenance(test, fold),
                )
                for row in fold_trace:
                    row["window_index"] += trace_offset
                trace_offset += len(fold_trace)
                predictions.extend(fold_trace)
                fits.append(detail)
                all_truth.append(labels[test])
                all_probability.append(probability)
                all_anchor_probability.append(anchor_probability)
            truth = np.concatenate(all_truth)
            probability = np.concatenate(all_probability)
            anchor_probability = np.concatenate(all_anchor_probability)
            row = _record(
                subject=subject,
                seed=seed,
                metrics=_metrics(truth, probability),
                anchor_metrics=_metrics(truth, anchor_probability),
                fits=fits,
                predictions=predictions,
                data_sha256=manifest,
                protocol={
                    "protocol_id": PROTOCOLS["cho-dev"],
                    "folds": int(folds),
                    "outer_split_random_state": 0,
                    "inner_validation_fraction": 0.2,
                },
            )
            payload["records"].append(row)
            _save_payload(output, payload, status="running")
            print(
                f"{ARCHITECTURE} cho-dev S{subject} seed={seed}: "
                f"bacc={100.0 * row['metrics']['balanced_accuracy']:.2f}%",
                flush=True,
            )


def _run_bnci(
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: ParityFuseConfig,
    output: Path,
    payload: dict[str, Any],
) -> None:
    completed = _completed_keys(payload)
    for subject in subjects:
        pending = [seed for seed in seeds if (int(subject), int(seed)) not in completed]
        if not pending:
            continue
        data = load_bnci2014_subject(subject)
        manifest = _data_manifest(data)
        train = data["train"]
        validation = data["validation"]
        test = data["test"]
        for seed in pending:
            detail, probability, anchor_probability, model_output = _fit_score(
                raw_train=train["broadband"],
                cov_train=train["covariances"],
                y_train=train["labels"],
                raw_validation=validation["broadband"],
                cov_validation=validation["covariances"],
                y_validation=validation["labels"],
                raw_test=test["broadband"],
                cov_test=test["covariances"],
                y_test=test["labels"],
                channels=BNCI_CHANNELS,
                config=replace(config, seed=int(seed)),
            )
            detail.update(
                {
                    "train_run_ids": sorted(set(train["run_ids"].astype(str).tolist())),
                    "validation_run_ids": sorted(
                        set(validation["run_ids"].astype(str).tolist())
                    ),
                    "test_run_ids": sorted(set(test["run_ids"].astype(str).tolist())),
                }
            )
            truth = test["labels"]
            row = _record(
                subject=subject,
                seed=seed,
                metrics=_metrics(truth, probability),
                anchor_metrics=_metrics(truth, anchor_probability),
                fits=[detail],
                predictions=_prediction_trace(
                    truth,
                    probability,
                    anchor_probability,
                    model_output,
                    _bnci_provenance(test["run_ids"]),
                ),
                data_sha256=manifest,
                protocol=protocol_metadata(subject),
            )
            payload["records"].append(row)
            _save_payload(output, payload, status="running")
            print(
                f"{ARCHITECTURE} {mode} S{subject} seed={seed}: "
                f"bacc={100.0 * row['metrics']['balanced_accuracy']:.2f}%",
                flush=True,
            )


def _resolve_development_artifact(
    manifest_path: Path, reference: str
) -> Path:
    path = Path(reference).expanduser()
    return path if path.is_absolute() else manifest_path.parent / path


def _validate_frozen_manifest(
    path: Path | None, config: ParityFuseConfig
) -> dict[str, Any]:
    if path is None:
        raise ValueError("bnci-confirm requires --frozen-manifest")
    if not path.is_file():
        raise FileNotFoundError(f"frozen manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FROZEN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported frozen-manifest schema")
    if manifest.get("architecture") != ARCHITECTURE:
        raise ValueError("frozen manifest names a different architecture")
    if manifest.get("config") != _json_config(config):
        raise ValueError("confirmation config differs from the frozen manifest")
    current_source = _source_manifest()
    if manifest.get("source_sha256") != current_source:
        raise ValueError("confirmation source differs from the frozen manifest")
    development = manifest.get("development_artifact")
    if not isinstance(development, dict):
        raise ValueError("frozen manifest lacks a development_artifact object")
    if not isinstance(development.get("path"), str):
        raise ValueError("frozen manifest lacks the development artifact path")
    artifact_path = _resolve_development_artifact(path, development["path"])
    if not artifact_path.is_file():
        raise FileNotFoundError(f"frozen development artifact is missing: {artifact_path}")
    if development.get("sha256") != _file_sha256(artifact_path):
        raise ValueError("frozen development artifact hash no longer matches")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected_dev_contract = {
        "architecture": ARCHITECTURE,
        "mode": "bnci-dev",
        "subjects": list(DEVELOPMENT_SUBJECTS),
        "seeds": [7],
        "config": _json_config(config),
        "source_sha256": current_source,
    }
    observed_dev_contract = {
        "architecture": payload.get("architecture", {}).get("name"),
        "mode": payload.get("mode"),
        "subjects": payload.get("subjects"),
        "seeds": payload.get("seeds"),
        "config": payload.get("config"),
        "source_sha256": payload.get("source_sha256"),
    }
    if observed_dev_contract != expected_dev_contract:
        raise ValueError("development artifact does not match the frozen contract")
    completion = payload.get("completion", {})
    if not completion.get("complete") or completion.get("status") != "complete":
        raise ValueError("development artifact is not complete")
    if (
        completion.get("expected_records") != len(DEVELOPMENT_SUBJECTS)
        or completion.get("actual_records") != len(DEVELOPMENT_SUBJECTS)
        or len(payload.get("records", [])) != len(DEVELOPMENT_SUBJECTS)
    ):
        raise ValueError("development artifact has the wrong expected record count")
    if development.get("summary_sha256") != _json_sha256(payload.get("summary", {})):
        raise ValueError("development summary hash no longer matches the manifest")
    return manifest


def write_frozen_manifest(
    development_artifact: Path,
    output: Path,
    *,
    config: ParityFuseConfig,
) -> dict[str, Any]:
    """Freeze an exact completed S1--4/seed-7 BNCI development artifact."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output}")
    if not development_artifact.is_file():
        raise FileNotFoundError(development_artifact)
    payload = json.loads(development_artifact.read_text(encoding="utf-8"))
    current_source = _source_manifest()
    expected = {
        "architecture": ARCHITECTURE,
        "mode": "bnci-dev",
        "subjects": list(DEVELOPMENT_SUBJECTS),
        "seeds": [7],
        "config": _json_config(config),
        "source_sha256": current_source,
    }
    observed = {
        "architecture": payload.get("architecture", {}).get("name"),
        "mode": payload.get("mode"),
        "subjects": payload.get("subjects"),
        "seeds": payload.get("seeds"),
        "config": payload.get("config"),
        "source_sha256": payload.get("source_sha256"),
    }
    if observed != expected:
        raise ValueError("cannot freeze a development artifact with a different contract")
    completion = payload.get("completion", {})
    if not completion.get("complete") or completion.get("status") != "complete":
        raise ValueError("cannot freeze an incomplete development artifact")
    if (
        completion.get("expected_records") != len(DEVELOPMENT_SUBJECTS)
        or completion.get("actual_records") != len(DEVELOPMENT_SUBJECTS)
        or len(payload.get("records", [])) != len(DEVELOPMENT_SUBJECTS)
    ):
        raise ValueError("cannot freeze a development artifact with missing records")
    manifest = {
        "schema_version": FROZEN_MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "architecture": ARCHITECTURE,
        "config": _json_config(config),
        "source_sha256": current_source,
        "development_artifact": {
            "path": str(development_artifact.resolve()),
            "sha256": _file_sha256(development_artifact),
            "summary_sha256": _json_sha256(payload.get("summary", {})),
            "mode": "bnci-dev",
            "subjects": list(DEVELOPMENT_SUBJECTS),
            "seeds": [7],
            "expected_records": len(DEVELOPMENT_SUBJECTS),
        },
    }
    _exclusive_json_create(output, manifest)
    return manifest


def _validate_confirmation_contract(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: ParityFuseConfig,
    confirmation_token: str | None,
    frozen_manifest: Path | None,
) -> dict[str, Any] | None:
    if mode != "bnci-confirm":
        if confirmation_token is not None or frozen_manifest is not None:
            raise ValueError(
                "confirmation token/manifest may only be used with bnci-confirm"
            )
        return None
    if list(subjects) != list(CONFIRMATION_SUBJECTS):
        raise ValueError("bnci-confirm requires the exact ordered subjects 5--9")
    if list(seeds) != [7] or config.seed != 7:
        raise ValueError("bnci-confirm requires the sole frozen seed 7")
    if confirmation_token != CONFIRMATION_TOKEN:
        raise ValueError(
            "bnci-confirm requires the exact explicit confirmation token"
        )
    # This verifies config, source, and the pinned development file before the
    # caller can acquire or invoke the confirmation data loader.
    return _validate_frozen_manifest(frozen_manifest, config)


def run(
    *,
    mode: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    config: ParityFuseConfig,
    output: Path,
    folds: int = 5,
    resume: bool = False,
    frozen_manifest: Path | None = None,
    confirmation_token: str | None = None,
) -> dict[str, Any]:
    """Run one locked cohort with all guards applied before data access."""

    subjects = [int(value) for value in subjects]
    seeds = [int(value) for value in seeds]
    output = Path(output)
    _validate_common_contract(mode, subjects, seeds, folds)
    manifest = _validate_confirmation_contract(
        mode=mode,
        subjects=subjects,
        seeds=seeds,
        config=config,
        confirmation_token=confirmation_token,
        frozen_manifest=frozen_manifest,
    )

    # The lock and initial exclusive result publication both happen before any
    # dataset loader.  A competing invocation therefore cannot duplicate a
    # one-shot confirmation run against the same output contract.
    with _OutputLock(output):
        payload = _load_or_initialize(
            output,
            mode=mode,
            subjects=subjects,
            seeds=seeds,
            folds=folds,
            config=config,
            resume=resume,
            frozen_manifest=manifest,
            frozen_manifest_path=frozen_manifest,
        )
        if not output.exists():
            _exclusive_json_create(output, payload)
        else:
            _save_payload(output, payload, status="running")
        try:
            if mode == "local-dev":
                _run_local(subjects, seeds, config, output, payload)
            elif mode == "cho-dev":
                _run_cho(subjects, seeds, folds, config, output, payload)
            else:
                _run_bnci(mode, subjects, seeds, config, output, payload)
        except BaseException as error:
            payload["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "at": _utc_now(),
            }
            _save_payload(output, payload, status="interrupted")
            raise
        payload.pop("failure", None)
        _save_payload(output, payload, status="complete")
        return payload


def _build_config(args: argparse.Namespace) -> ParityFuseConfig:
    return ParityFuseConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        temporal_filters=args.temporal_filters,
        temporal_kernel=args.temporal_kernel,
        dynamics_channels=args.dynamics_channels,
        dynamics_kernel=args.dynamics_kernel,
        raw_dim=args.raw_dim,
        tangent_hidden=args.tangent_hidden,
        tangent_band_dim=args.tangent_band_dim,
        tangent_dim=args.tangent_dim,
        fusion_dim=args.fusion_dim,
        gate_hidden=args.gate_hidden,
        residual_penalty=args.residual_penalty,
        gradient_clip=args.gradient_clip,
        seed=7,
        device=args.device,
        deterministic=not args.allow_nondeterministic,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--confirmation-token")
    parser.add_argument("--write-frozen-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=ParityFuseConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=ParityFuseConfig.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=ParityFuseConfig.learning_rate
    )
    parser.add_argument("--weight-decay", type=float, default=ParityFuseConfig.weight_decay)
    parser.add_argument("--patience", type=int, default=ParityFuseConfig.patience)
    parser.add_argument("--min-delta", type=float, default=ParityFuseConfig.min_delta)
    parser.add_argument(
        "--temporal-filters", type=int, default=ParityFuseConfig.temporal_filters
    )
    parser.add_argument(
        "--temporal-kernel", type=int, default=ParityFuseConfig.temporal_kernel
    )
    parser.add_argument(
        "--dynamics-channels", type=int, default=ParityFuseConfig.dynamics_channels
    )
    parser.add_argument(
        "--dynamics-kernel", type=int, default=ParityFuseConfig.dynamics_kernel
    )
    parser.add_argument("--raw-dim", type=int, default=ParityFuseConfig.raw_dim)
    parser.add_argument(
        "--tangent-hidden", type=int, default=ParityFuseConfig.tangent_hidden
    )
    parser.add_argument(
        "--tangent-band-dim", type=int, default=ParityFuseConfig.tangent_band_dim
    )
    parser.add_argument("--tangent-dim", type=int, default=ParityFuseConfig.tangent_dim)
    parser.add_argument("--fusion-dim", type=int, default=ParityFuseConfig.fusion_dim)
    parser.add_argument("--gate-hidden", type=int, default=ParityFuseConfig.gate_hidden)
    parser.add_argument(
        "--residual-penalty", type=float, default=ParityFuseConfig.residual_penalty
    )
    parser.add_argument(
        "--gradient-clip", type=float, default=ParityFuseConfig.gradient_clip
    )
    parser.add_argument("--allow-nondeterministic", action="store_true")
    args = parser.parse_args(argv)

    defaults: dict[str, Sequence[int]] = {
        "local-dev": VALID_SUBJECTS,
        "cho-dev": tuple(range(1, 27)),
        "bnci-dev": DEVELOPMENT_SUBJECTS,
        "bnci-confirm": CONFIRMATION_SUBJECTS,
    }
    subjects = (
        list(defaults[args.mode])
        if args.subjects is None
        else _parse_ints(args.subjects)
    )
    seeds = _parse_ints(args.seeds)
    config = _build_config(args)
    if args.write_frozen_manifest is not None:
        if args.mode != "bnci-dev":
            raise ValueError("--write-frozen-manifest is only valid with bnci-dev")
        if subjects != list(DEVELOPMENT_SUBJECTS) or seeds != [7]:
            raise ValueError(
                "a frozen manifest requires the exact BNCI development cohort "
                "1--4 and sole seed 7"
            )
        if args.write_frozen_manifest.resolve() == args.output.resolve():
            raise ValueError("result and frozen-manifest outputs must be different files")
        if args.write_frozen_manifest.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen manifest: {args.write_frozen_manifest}"
            )
    payload = run(
        mode=args.mode,
        subjects=subjects,
        seeds=seeds,
        config=config,
        output=args.output,
        folds=args.folds,
        resume=args.resume,
        frozen_manifest=args.frozen_manifest,
        confirmation_token=args.confirmation_token,
    )
    if args.write_frozen_manifest is not None:
        write_frozen_manifest(
            args.output,
            args.write_frozen_manifest,
            config=config,
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURE",
    "CONFIRMATION_TOKEN",
    "MODES",
    "run",
    "write_frozen_manifest",
]
