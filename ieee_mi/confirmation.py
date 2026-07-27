"""Manifest-gated, resumable, prediction-before-scoring confirmation primitives.

Public mutation entry points fail closed. Internal callbacks execute only
after the separately guarded data/model layer validates an opaque capability
bound to the freeze token, one resolved run root, and the requested phase.
This module does not itself open an EEG cache or construct a model.
"""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from .freeze import (
    PRIMARY_STATISTIC_NAME,
    _atomic_write_once,
    _file_sha256,
    _is_sha256,
    _json_clone,
    _json_sha256,
    _read_json_object,
    _validate_canonical_json_file,
    validate_freeze_manifest,
)

CONFIRMATION_STARTED_SCHEMA = "ieee-mi-confirmation-started-v1"
CONFIRMATION_PREDICTION_SCHEMA = "ieee-mi-confirmation-prediction-v1"
CONFIRMATION_COMPLETED_SCHEMA = "ieee-mi-confirmation-predictions-complete-v1"
CONFIRMATION_SCORING_STARTED_SCHEMA = "ieee-mi-confirmation-scoring-started-v1"
CONFIRMATION_SCORE_SCHEMA = "ieee-mi-confirmation-score-v1"

_RECEIPT_FIELDS = {
    CONFIRMATION_STARTED_SCHEMA: frozenset(
        {
            "schema",
            "mode",
            "phase",
            "confirmation_evidence",
            "manifest_sha256",
            "manifest_file_sha256",
            "expected_record_count",
            "expected_records_sha256",
            "started_at",
            "receipt_sha256",
        }
    ),
    CONFIRMATION_COMPLETED_SCHEMA: frozenset(
        {
            "schema",
            "mode",
            "phase",
            "confirmation_evidence",
            "scoring_performed",
            "manifest_sha256",
            "started_receipt_sha256",
            "record_count",
            "expected_records_sha256",
            "prediction_inventory",
            "prediction_inventory_sha256",
            "completed_at",
            "receipt_sha256",
        }
    ),
    CONFIRMATION_SCORING_STARTED_SCHEMA: frozenset(
        {
            "schema",
            "mode",
            "phase",
            "confirmation_evidence",
            "manifest_sha256",
            "completed_receipt_sha256",
            "prediction_inventory_sha256",
            "started_at",
            "receipt_sha256",
        }
    ),
    CONFIRMATION_SCORE_SCHEMA: frozenset(
        {
            "schema",
            "mode",
            "phase",
            "confirmation_evidence",
            "manifest_sha256",
            "started_receipt_sha256",
            "completed_receipt_sha256",
            "scoring_started_receipt_sha256",
            "prediction_inventory_sha256",
            "analysis_plan",
            "primary_statistic_value",
            "primary_statistic_details",
            "success",
            "scored_at",
            "receipt_sha256",
        }
    ),
}
_RECEIPT_TIMESTAMP_FIELD = {
    CONFIRMATION_STARTED_SCHEMA: "started_at",
    CONFIRMATION_COMPLETED_SCHEMA: "completed_at",
    CONFIRMATION_SCORING_STARTED_SCHEMA: "started_at",
    CONFIRMATION_SCORE_SCHEMA: "scored_at",
}

STARTED_RECEIPT = "started.json"
COMPLETED_RECEIPT = "completed.json"
SCORING_STARTED_RECEIPT = "scoring-started.json"
SCORE_RECEIPT = "score.json"
PREDICTION_DIRECTORY = "predictions"
RUN_LOCK = ".confirmation.lock"

_TRUSTED_INTEGRATION_ERROR = (
    "confirmation execution is disabled until the trusted data/model backend "
    "binds the freeze token to source-derived split rows, cache bytes, model "
    "construction, and the sealed-label provider"
)

_PREDICTION_OUTPUT_FIELDS = frozenset(
    {"test_rows", "probabilities", "sessions", "runs", "provenance"}
)
_PREDICTION_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "mode",
        "phase",
        "confirmation_evidence",
        "contains_test_labels",
        "contains_metrics",
        "manifest_sha256",
        "record_id",
        "record",
        "test_rows",
        "probabilities",
        "sessions",
        "runs",
        "provenance",
        "test_rows_sha256",
        "prediction_sha256",
    }
)
_PRIMARY_STATISTIC_DETAILS_FIELDS = frozenset(
    {
        "schema",
        "candidate_model",
        "comparator_model",
        "preprocessing_profile",
        "dataset_mean_paired_deltas",
        "subject_paired_deltas",
        "n_datasets",
        "n_subjects",
        "seeds",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "cache_array_sha256",
        "model_state_sha256",
        "model_contract_sha256",
        "architecture_sha256",
        "training_config_sha256",
        "pretrained_checkpoint_name",
        "pretrained_checkpoint_sha256",
        "environment",
    }
)
_FORBIDDEN_PREDICTION_KEYS = frozenset(
    {
        "label",
        "labels",
        "test_label",
        "test_labels",
        "ground_truth",
        "metric",
        "metrics",
        "score",
        "scores",
        "accuracy",
        "balanced_accuracy",
        "cohen_kappa",
        "roc_auc",
    }
)
_FORBIDDEN_SCORE_OUTPUT_KEYS = frozenset(
    {
        "label",
        "labels",
        "test_label",
        "test_labels",
        "ground_truth",
        "probability",
        "probabilities",
        "test_rows",
    }
)


PredictionCallback = Callable[
    ["ConfirmationRecord", Mapping[str, Any]],
    Mapping[str, Any],
]
LabelCallback = Callable[["ConfirmationSplit", tuple[int, ...]], Sequence[int]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_canonical_json(path: Path) -> dict[str, Any]:
    value = _read_json_object(path)
    _validate_canonical_json_file(path, value)
    return value


def _require_trusted_backend_capability(
    capability: object,
    *,
    manifest_path: str | Path,
    project_root: str | Path,
    run_root: str | Path,
    operation: str,
) -> Path:
    """Delegate capability verification to the separately guarded data layer.

    The data/model integration must provide
    ``data.validate_confirmation_execution_capability``. On ``initialize`` it
    must atomically claim exactly one resolved run root for the freeze token;
    every later operation must match that claim. Until the function exists and
    positively validates an opaque backend-owned capability, every internal
    mutation hook fails closed. Synthetic tests patch this verifier below the
    public gate; no production capability is minted in this module.
    """

    from . import data as data_module

    validator = getattr(
        data_module,
        "validate_confirmation_execution_capability",
        None,
    )
    if not callable(validator):
        raise PermissionError(_TRUSTED_INTEGRATION_ERROR)
    raw_run_root = Path(run_root)
    if raw_run_root.is_symlink():
        raise PermissionError("confirmation run root must not be a symlink")
    resolved_run_root = raw_run_root.resolve(strict=False)
    validated = validator(
        capability=capability,
        manifest_path=Path(manifest_path),
        project_root=Path(project_root),
        run_root=resolved_run_root,
        operation=operation,
    )
    if validated is not True:
        raise PermissionError(_TRUSTED_INTEGRATION_ERROR)
    return resolved_run_root


@dataclass(frozen=True, order=True)
class ConfirmationSplit:
    """Model-independent identity of one frozen prediction-only test split."""

    dataset: str
    subject: int
    fold: int


@dataclass(frozen=True, order=True)
class ConfirmationRecord:
    """One frozen dataset/profile/subject/fold/model/seed prediction key."""

    dataset: str
    preprocessing_profile: str
    subject: int
    fold: int
    role: str
    model: str
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "preprocessing_profile": self.preprocessing_profile,
            "subject": self.subject,
            "fold": self.fold,
            "role": self.role,
            "model": self.model,
            "seed": self.seed,
        }

    @property
    def record_id(self) -> str:
        return "record-" + _json_sha256(self.as_dict())

    @property
    def split(self) -> ConfirmationSplit:
        return ConfirmationSplit(
            dataset=self.dataset,
            subject=self.subject,
            fold=self.fold,
        )


@dataclass(frozen=True)
class ScoringRecord:
    """Ephemeral labels joined to one immutable prediction in memory only."""

    record: ConfirmationRecord
    test_rows: tuple[int, ...]
    probabilities: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]


def expected_confirmation_records(
    manifest: Mapping[str, Any],
) -> tuple[ConfirmationRecord, ...]:
    """Expand the exact Cartesian record set frozen in ``manifest``."""

    design = manifest.get("study_design")
    if not isinstance(design, Mapping):
        raise ValueError("freeze manifest has no study design")
    candidates = design.get("models")
    comparators = design.get("comparators")
    seeds = design.get("seeds")
    datasets = design.get("datasets")
    if not all(isinstance(value, list) for value in (candidates, comparators, seeds)):
        raise ValueError("freeze manifest model/seed dimensions are invalid")
    if not isinstance(datasets, Mapping):
        raise ValueError("freeze manifest dataset dimension is invalid")
    models = tuple((str(name), "model") for name in candidates) + tuple(
        (str(name), "comparator") for name in comparators
    )
    records: list[ConfirmationRecord] = []
    for dataset, raw_plan in sorted(datasets.items()):
        if not isinstance(raw_plan, Mapping):
            raise ValueError(f"freeze manifest {dataset} plan is invalid")
        subjects = raw_plan.get("subjects")
        folds = raw_plan.get("folds")
        profiles = raw_plan.get("preprocessing_profiles")
        if not isinstance(subjects, list) or not isinstance(folds, list):
            raise ValueError(f"freeze manifest {dataset} dimensions are invalid")
        if not isinstance(profiles, Mapping):
            raise ValueError(f"freeze manifest {dataset} profiles are invalid")
        for profile, subject, fold, (model, role), seed in product(
            sorted(profiles),
            subjects,
            folds,
            models,
            seeds,
        ):
            records.append(
                ConfirmationRecord(
                    dataset=str(dataset),
                    preprocessing_profile=str(profile),
                    subject=int(subject),
                    fold=int(fold),
                    role=role,
                    model=model,
                    seed=int(seed),
                )
            )
    result = tuple(sorted(records))
    if not result or len({record.record_id for record in result}) != len(result):
        raise ValueError("freeze manifest does not define unique confirmation records")
    return result


def _signed_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_clone(payload)
    result["receipt_sha256"] = _json_sha256(result)
    return result


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    schema: str,
    manifest_sha256: str,
) -> None:
    expected_fields = _RECEIPT_FIELDS[schema]
    if set(receipt) != expected_fields:
        missing = sorted(expected_fields - set(receipt))
        extra = sorted(set(receipt) - expected_fields)
        raise ValueError(
            f"receipt fields differ for {schema}; missing={missing}, extra={extra}"
        )
    if receipt.get("schema") != schema:
        raise ValueError(f"unsupported receipt schema; expected {schema}")
    if receipt.get("mode") != "confirmation":
        raise ValueError("receipt is not a confirmation artifact")
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise PermissionError("receipt belongs to a different freeze manifest")
    timestamp_name = _RECEIPT_TIMESTAMP_FIELD[schema]
    if not isinstance(receipt.get(timestamp_name), str) or not receipt[timestamp_name]:
        raise ValueError(f"receipt has no valid {timestamp_name}")
    digest = receipt.get("receipt_sha256")
    if not _is_sha256(digest):
        raise ValueError("receipt has no valid hash")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if _json_sha256(unsigned) != digest:
        raise ValueError("receipt hash does not match its contents")


def _expected_records_sha256(records: Sequence[ConfirmationRecord]) -> str:
    return _json_sha256([record.as_dict() for record in records])


def _initialize_confirmation_run_for_trusted_backend(
    manifest_path: str | Path,
    run_root: str | Path,
    *,
    capability: object,
    project_root: str | Path,
) -> dict[str, Any]:
    """Atomically start the one run claimed by the trusted backend."""

    root = _require_trusted_backend_capability(
        capability,
        manifest_path=manifest_path,
        project_root=project_root,
        run_root=run_root,
        operation="initialize",
    )
    manifest = validate_freeze_manifest(manifest_path, project_root=project_root)
    token = str(manifest["manifest_sha256"])
    records = expected_confirmation_records(manifest)
    root.parent.mkdir(parents=True, exist_ok=True)
    receipt = _signed_receipt(
        {
            "schema": CONFIRMATION_STARTED_SCHEMA,
            "mode": "confirmation",
            "phase": "prediction_started",
            "confirmation_evidence": False,
            "manifest_sha256": token,
            "manifest_file_sha256": _file_sha256(Path(manifest_path)),
            "expected_record_count": len(records),
            "expected_records_sha256": _expected_records_sha256(records),
            "started_at": _utc_now(),
        }
    )
    initialization_lock = root.parent / f".{root.name}.initialization.lock"
    lock_descriptor = os.open(initialization_lock, os.O_CREAT | os.O_RDWR, 0o600)
    staging: Path | None = None
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if os.path.lexists(root):
            raise FileExistsError(root)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{root.name}.staging-",
                dir=root.parent,
            )
        )
        (staging / PREDICTION_DIRECTORY).mkdir()
        _atomic_write_once(staging / STARTED_RECEIPT, receipt)
        directory_descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.rename(staging, root)
        staging = None
        parent_descriptor = os.open(root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
    return receipt


def initialize_confirmation_run(
    manifest_path: str | Path,
    run_root: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Refuse confirmation-run creation until trusted integration exists."""

    del manifest_path, run_root, project_root
    raise PermissionError(_TRUSTED_INTEGRATION_ERROR)


@contextmanager
def _exclusive_run_lock(root: Path) -> Iterator[None]:
    descriptor = os.open(root / RUN_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_run_context(
    manifest_path: str | Path,
    run_root: str | Path,
    *,
    project_root: str | Path,
) -> tuple[dict[str, Any], tuple[ConfirmationRecord, ...], dict[str, Any]]:
    manifest = validate_freeze_manifest(manifest_path, project_root=project_root)
    records = expected_confirmation_records(manifest)
    root = Path(run_root)
    receipt = _read_canonical_json(root / STARTED_RECEIPT)
    token = str(manifest["manifest_sha256"])
    _validate_receipt(
        receipt,
        schema=CONFIRMATION_STARTED_SCHEMA,
        manifest_sha256=token,
    )
    if receipt.get("phase") != "prediction_started":
        raise ValueError("started receipt has the wrong phase")
    if receipt.get("confirmation_evidence") is not False:
        raise ValueError("started receipt incorrectly claims confirmation evidence")
    if receipt.get("manifest_file_sha256") != _file_sha256(Path(manifest_path)):
        raise PermissionError("freeze manifest file bytes changed after run start")
    if receipt.get("expected_record_count") != len(records):
        raise ValueError("started receipt has the wrong record count")
    if receipt.get("expected_records_sha256") != _expected_records_sha256(records):
        raise ValueError("started receipt has the wrong Cartesian record identity")
    return manifest, records, receipt


def prediction_path(run_root: str | Path, record: ConfirmationRecord) -> Path:
    return Path(run_root) / PREDICTION_DIRECTORY / f"{record.record_id}.json"


def _contains_forbidden_key(value: Any, forbidden: frozenset[str]) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().lower() in forbidden:
                return True
            if _contains_forbidden_key(child, forbidden):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(child, forbidden) for child in value)
    return False


def _dataset_classes(manifest: Mapping[str, Any], dataset: str) -> int:
    try:
        return int(
            manifest["study_design"]["datasets"][dataset]["dataset_spec"]["n_classes"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"freeze manifest has no class count for {dataset}") from error


def _normalize_prediction_output(
    output: Mapping[str, Any],
    *,
    record: ConfirmationRecord,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        raise ValueError("prediction callback must return an object")
    if _contains_forbidden_key(output, _FORBIDDEN_PREDICTION_KEYS):
        raise PermissionError("prediction output contains labels, metrics, or scores")
    if set(output) != _PREDICTION_OUTPUT_FIELDS:
        missing = sorted(_PREDICTION_OUTPUT_FIELDS - set(output))
        extra = sorted(set(output) - _PREDICTION_OUTPUT_FIELDS)
        raise ValueError(
            f"prediction callback fields differ; missing={missing}, extra={extra}"
        )
    raw_rows = output["test_rows"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("prediction test_rows must be a nonempty array")
    if any(isinstance(row, bool) or not isinstance(row, int) for row in raw_rows):
        raise ValueError("prediction test_rows must contain integers")
    rows = tuple(int(row) for row in raw_rows)
    if any(row < 0 for row in rows) or len(set(rows)) != len(rows):
        raise ValueError("prediction test_rows must be unique and nonnegative")
    n_classes = _dataset_classes(manifest, record.dataset)
    try:
        probabilities = np.asarray(output["probabilities"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("prediction probabilities are invalid") from error
    if probabilities.shape != (len(rows), n_classes):
        raise ValueError(
            f"prediction probabilities must have shape {(len(rows), n_classes)}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("prediction probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("prediction probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError("prediction probability rows must sum to one")
    sessions = output["sessions"]
    runs = output["runs"]
    if not isinstance(sessions, list) or not isinstance(runs, list):
        raise ValueError("prediction sessions/runs must be arrays")
    if len(sessions) != len(rows) or len(runs) != len(rows):
        raise ValueError("prediction rows/sessions/runs differ in length")
    normalized_sessions = [str(value) for value in sessions]
    normalized_runs = [str(value) for value in runs]
    if any(not value for value in normalized_sessions + normalized_runs):
        raise ValueError("prediction sessions/runs must be nonempty")
    provenance = output["provenance"]
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("prediction provenance must be a nonempty object")
    if set(provenance) != _PROVENANCE_FIELDS:
        missing = sorted(_PROVENANCE_FIELDS - set(provenance))
        extra = sorted(set(provenance) - _PROVENANCE_FIELDS)
        raise ValueError(
            f"prediction provenance fields differ; missing={missing}, extra={extra}"
        )
    for name in ("cache_array_sha256", "model_state_sha256"):
        if not _is_sha256(provenance.get(name)):
            raise ValueError(f"prediction provenance has no valid {name}")
    if (
        not isinstance(provenance.get("environment"), dict)
        or not provenance["environment"]
    ):
        raise ValueError("prediction provenance has no environment identity")
    try:
        model_contract = manifest["model_contracts"][record.model]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"freeze manifest has no model contract for {record.model}"
        ) from error
    if model_contract.get("role") != record.role:
        raise PermissionError("prediction role differs from the frozen model contract")
    expected_provenance = {
        "model_contract_sha256": model_contract.get("model_contract_sha256"),
        "architecture_sha256": model_contract.get("architecture_sha256"),
        "training_config_sha256": model_contract.get("training_config_sha256"),
        "pretrained_checkpoint_name": model_contract.get("pretrained_checkpoint_name"),
        "pretrained_checkpoint_sha256": model_contract.get(
            "pretrained_checkpoint_sha256"
        ),
    }
    for name, expected in expected_provenance.items():
        if provenance.get(name) != expected:
            raise PermissionError(
                f"prediction {name} differs from the frozen model contract"
            )
    return {
        "test_rows": list(rows),
        "probabilities": probabilities.tolist(),
        "sessions": normalized_sessions,
        "runs": normalized_runs,
        "provenance": _json_clone(provenance),
    }


def _prediction_artifact(
    record: ConfirmationRecord,
    output: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _normalize_prediction_output(output, record=record, manifest=manifest)
    row_bytes = np.asarray(normalized["test_rows"], dtype=np.int64).tobytes()
    artifact: dict[str, Any] = {
        "schema": CONFIRMATION_PREDICTION_SCHEMA,
        "mode": "confirmation",
        "phase": "prediction_only",
        "confirmation_evidence": False,
        "contains_test_labels": False,
        "contains_metrics": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "record_id": record.record_id,
        "record": record.as_dict(),
        **normalized,
        "test_rows_sha256": hashlib.sha256(row_bytes).hexdigest(),
    }
    artifact["prediction_sha256"] = _json_sha256(artifact)
    return artifact


def _validate_prediction_artifact(
    artifact: Mapping[str, Any],
    *,
    record: ConfirmationRecord,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if set(artifact) != _PREDICTION_ARTIFACT_FIELDS:
        missing = sorted(_PREDICTION_ARTIFACT_FIELDS - set(artifact))
        extra = sorted(set(artifact) - _PREDICTION_ARTIFACT_FIELDS)
        raise ValueError(
            f"{record.record_id} prediction fields differ; "
            f"missing={missing}, extra={extra}"
        )
    if artifact.get("schema") != CONFIRMATION_PREDICTION_SCHEMA:
        raise ValueError(f"{record.record_id} has an unsupported prediction schema")
    if (
        artifact.get("mode") != "confirmation"
        or artifact.get("phase") != "prediction_only"
    ):
        raise ValueError(f"{record.record_id} is not a prediction-only artifact")
    if artifact.get("confirmation_evidence") is not False:
        raise ValueError(f"{record.record_id} incorrectly claims confirmation evidence")
    if (
        artifact.get("contains_test_labels") is not False
        or artifact.get("contains_metrics") is not False
    ):
        raise PermissionError(f"{record.record_id} claims labels or metrics")
    if artifact.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise PermissionError(f"{record.record_id} belongs to another freeze")
    if (
        artifact.get("record_id") != record.record_id
        or artifact.get("record") != record.as_dict()
    ):
        raise ValueError(f"{record.record_id} has the wrong record identity")
    digest = artifact.get("prediction_sha256")
    if not _is_sha256(digest):
        raise ValueError(f"{record.record_id} has no valid prediction hash")
    unsigned = dict(artifact)
    unsigned.pop("prediction_sha256", None)
    if _json_sha256(unsigned) != digest:
        raise ValueError(f"{record.record_id} prediction hash does not match")
    callback_output = {name: artifact.get(name) for name in _PREDICTION_OUTPUT_FIELDS}
    normalized = _normalize_prediction_output(
        callback_output,
        record=record,
        manifest=manifest,
    )
    row_bytes = np.asarray(normalized["test_rows"], dtype=np.int64).tobytes()
    if hashlib.sha256(row_bytes).hexdigest() != artifact.get("test_rows_sha256"):
        raise ValueError(f"{record.record_id} test-row hash does not match")
    return dict(artifact)


def _assert_prediction_phase_open(root: Path) -> None:
    if (root / SCORE_RECEIPT).exists() or (root / SCORING_STARTED_RECEIPT).exists():
        raise PermissionError("prediction phase is closed because scoring has started")
    if (root / COMPLETED_RECEIPT).exists():
        raise PermissionError("prediction phase is already complete")


def _write_prediction_unlocked(
    root: Path,
    record: ConfirmationRecord,
    output: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    target = prediction_path(root, record)
    if target.exists():
        raise FileExistsError(
            f"completed prediction must never be rerun: {record.record_id}"
        )
    artifact = _prediction_artifact(record, output, manifest)
    _atomic_write_once(target, artifact)
    return artifact


def _record_confirmation_prediction_for_trusted_backend(
    manifest_path: str | Path,
    run_root: str | Path,
    record: ConfirmationRecord,
    output: Mapping[str, Any],
    *,
    capability: object,
    project_root: str | Path,
) -> dict[str, Any]:
    """Internal write hook for the future manifest-gated trusted backend."""

    root = _require_trusted_backend_capability(
        capability,
        manifest_path=manifest_path,
        project_root=project_root,
        run_root=run_root,
        operation="record_prediction",
    )
    with _exclusive_run_lock(root):
        manifest, records, _ = _load_run_context(
            manifest_path,
            root,
            project_root=project_root,
        )
        if record not in set(records):
            raise PermissionError(
                "prediction record is outside the frozen Cartesian design"
            )
        _assert_prediction_phase_open(root)
        return _write_prediction_unlocked(root, record, output, manifest)


def record_confirmation_prediction(
    manifest_path: str | Path,
    run_root: str | Path,
    record: ConfirmationRecord,
    output: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Refuse caller-supplied confirmation outputs until trusted integration."""

    del manifest_path, run_root, record, output, project_root
    raise PermissionError(_TRUSTED_INTEGRATION_ERROR)


def _collect_complete_predictions(
    root: Path,
    records: Sequence[ConfirmationRecord],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    expected = {prediction_path(root, record): record for record in records}
    allowed_root_entries = {
        STARTED_RECEIPT,
        COMPLETED_RECEIPT,
        SCORING_STARTED_RECEIPT,
        SCORE_RECEIPT,
        PREDICTION_DIRECTORY,
        RUN_LOCK,
    }
    extra_root_entries = sorted(
        path.name for path in root.iterdir() if path.name not in allowed_root_entries
    )
    if extra_root_entries:
        raise ValueError(
            f"confirmation run contains unexpected files: {extra_root_entries}"
        )
    prediction_root = root / PREDICTION_DIRECTORY
    if not prediction_root.is_dir():
        raise ValueError("confirmation prediction directory is unavailable")
    actual = set(prediction_root.iterdir())
    if actual != set(expected):
        missing = sorted(path.name for path in set(expected) - actual)
        extra = sorted(path.name for path in actual - set(expected))
        raise ValueError(
            f"prediction files are not the complete Cartesian design; "
            f"missing={missing}, extra={extra}"
        )
    artifacts: dict[str, dict[str, Any]] = {}
    paired_data: dict[tuple[str, int, int], dict[str, Any]] = {}
    subject_cache: dict[tuple[str, str, int], str] = {}
    fold_rows: dict[tuple[str, str, int], dict[int, set[int]]] = {}
    environment_identity: dict[str, Any] | None = None
    for path, record in expected.items():
        artifact = _validate_prediction_artifact(
            _read_canonical_json(path),
            record=record,
            manifest=manifest,
        )
        paired_key = (
            record.dataset,
            record.subject,
            record.fold,
        )
        paired_contract = {
            "test_rows": artifact["test_rows"],
            "test_rows_sha256": artifact["test_rows_sha256"],
            "sessions": artifact["sessions"],
            "runs": artifact["runs"],
        }
        previous = paired_data.setdefault(paired_key, paired_contract)
        if paired_contract != previous:
            raise ValueError(f"paired prediction data differ for {paired_key}")
        cache_key = (
            record.dataset,
            record.preprocessing_profile,
            record.subject,
        )
        cache_hash = str(artifact["provenance"]["cache_array_sha256"])
        previous_cache = subject_cache.setdefault(cache_key, cache_hash)
        if cache_hash != previous_cache:
            raise ValueError(f"cache identity differs across folds for {cache_key}")
        current_fold_rows = fold_rows.setdefault(cache_key, {})
        current_fold_rows.setdefault(record.fold, set(artifact["test_rows"]))
        current_environment = artifact["provenance"]["environment"]
        if environment_identity is None:
            environment_identity = current_environment
        elif current_environment != environment_identity:
            raise ValueError("prediction records use different environment identities")
        artifacts[record.record_id] = artifact
    for cache_key, rows_by_fold in fold_rows.items():
        seen_rows: set[int] = set()
        for fold, rows in sorted(rows_by_fold.items()):
            overlap = seen_rows.intersection(rows)
            if overlap:
                raise ValueError(
                    f"test rows overlap across folds for {cache_key}, fold {fold}: {overlap}"
                )
            seen_rows.update(rows)
    return artifacts


def _prediction_inventory(
    root: Path,
    records: Sequence[ConfirmationRecord],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    return {
        record.record_id: {
            "prediction_sha256": str(artifacts[record.record_id]["prediction_sha256"]),
            "file_sha256": _file_sha256(prediction_path(root, record)),
        }
        for record in records
    }


def _completed_receipt(
    root: Path,
    records: Sequence[ConfirmationRecord],
    manifest: Mapping[str, Any],
    started: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    inventory = _prediction_inventory(root, records, artifacts)
    return _signed_receipt(
        {
            "schema": CONFIRMATION_COMPLETED_SCHEMA,
            "mode": "confirmation",
            "phase": "predictions_complete",
            "confirmation_evidence": False,
            "scoring_performed": False,
            "manifest_sha256": manifest["manifest_sha256"],
            "started_receipt_sha256": started["receipt_sha256"],
            "record_count": len(records),
            "expected_records_sha256": _expected_records_sha256(records),
            "prediction_inventory": inventory,
            "prediction_inventory_sha256": _json_sha256(inventory),
            "completed_at": _utc_now(),
        }
    )


def _validate_completed_receipt(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    records: Sequence[ConfirmationRecord],
    manifest: Mapping[str, Any],
    started: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> None:
    token = str(manifest["manifest_sha256"])
    _validate_receipt(
        receipt,
        schema=CONFIRMATION_COMPLETED_SCHEMA,
        manifest_sha256=token,
    )
    if receipt.get("phase") != "predictions_complete":
        raise ValueError("prediction completion receipt has the wrong phase")
    if (
        receipt.get("confirmation_evidence") is not False
        or receipt.get("scoring_performed") is not False
    ):
        raise ValueError("prediction completion receipt overstates its evidence")
    if receipt.get("started_receipt_sha256") != started.get("receipt_sha256"):
        raise ValueError("completion receipt references the wrong start")
    if receipt.get("record_count") != len(records):
        raise ValueError("completion receipt has the wrong record count")
    if receipt.get("expected_records_sha256") != _expected_records_sha256(records):
        raise ValueError("completion receipt has the wrong Cartesian identity")
    inventory = _prediction_inventory(root, records, artifacts)
    if receipt.get("prediction_inventory") != inventory:
        raise ValueError(
            "completion receipt prediction inventory differs from current bytes"
        )
    if receipt.get("prediction_inventory_sha256") != _json_sha256(inventory):
        raise ValueError("completion receipt prediction inventory hash is invalid")


def _run_confirmation_predictions_for_trusted_backend(
    manifest_path: str | Path,
    run_root: str | Path,
    callback: PredictionCallback,
    *,
    capability: object,
    project_root: str | Path,
) -> dict[str, Any]:
    """Internal execution hook for the future manifest-gated trusted backend."""

    root = _require_trusted_backend_capability(
        capability,
        manifest_path=manifest_path,
        project_root=project_root,
        run_root=run_root,
        operation="run_predictions",
    )
    with _exclusive_run_lock(root):
        manifest, records, started = _load_run_context(
            manifest_path,
            root,
            project_root=project_root,
        )
        if (root / SCORE_RECEIPT).exists() or (root / SCORING_STARTED_RECEIPT).exists():
            raise PermissionError("predictions cannot run after scoring starts")
        if (root / COMPLETED_RECEIPT).exists():
            artifacts = _collect_complete_predictions(root, records, manifest)
            completed = _read_canonical_json(root / COMPLETED_RECEIPT)
            _validate_completed_receipt(
                completed,
                root=root,
                records=records,
                manifest=manifest,
                started=started,
                artifacts=artifacts,
            )
            return completed
        for record in records:
            path = prediction_path(root, record)
            if path.exists():
                _validate_prediction_artifact(
                    _read_canonical_json(path),
                    record=record,
                    manifest=manifest,
                )
                continue
            # Callbacks receive a disposable copy so they cannot mutate the
            # internally validated freeze used to sign artifacts or receipts.
            output = callback(record, _json_clone(manifest))
            _write_prediction_unlocked(root, record, output, manifest)
        current_manifest = validate_freeze_manifest(
            manifest_path,
            project_root=project_root,
        )
        if current_manifest["manifest_sha256"] != manifest["manifest_sha256"]:
            raise PermissionError("freeze manifest changed during prediction")
        artifacts = _collect_complete_predictions(root, records, manifest)
        receipt = _completed_receipt(
            root,
            records,
            manifest,
            started,
            artifacts,
        )
        _atomic_write_once(root / COMPLETED_RECEIPT, receipt)
        return receipt


def run_confirmation_predictions(
    manifest_path: str | Path,
    run_root: str | Path,
    callback: PredictionCallback,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Refuse arbitrary prediction callbacks until trusted integration exists."""

    del manifest_path, run_root, callback, project_root
    raise PermissionError(_TRUSTED_INTEGRATION_ERROR)


def _validate_prediction_phase_unlocked(
    manifest_path: str | Path,
    run_root: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Validate every prediction and its immutable completion receipt."""

    manifest, records, started = _load_run_context(
        manifest_path,
        run_root,
        project_root=project_root,
    )
    root = Path(run_root)
    artifacts = _collect_complete_predictions(root, records, manifest)
    completed = _read_canonical_json(root / COMPLETED_RECEIPT)
    _validate_completed_receipt(
        completed,
        root=root,
        records=records,
        manifest=manifest,
        started=started,
        artifacts=artifacts,
    )
    return completed


def validate_prediction_phase(
    manifest_path: str | Path,
    run_root: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Validate every prediction and receipt under the run lock."""

    root = Path(run_root)
    with _exclusive_run_lock(root):
        return _validate_prediction_phase_unlocked(
            manifest_path,
            root,
            project_root=project_root,
        )


def _evaluate_success(rule: Mapping[str, Any], value: float) -> bool:
    threshold = float(rule["threshold"])
    operator = rule["operator"]
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    raise ValueError(f"unsupported frozen success operator {operator!r}")


def _balanced_accuracy(
    labels: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    *,
    n_classes: int,
) -> float:
    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.shape != (len(label_array), n_classes):
        raise ValueError("scoring probabilities have an invalid shape")
    if set(label_array.tolist()) != set(range(n_classes)):
        raise ValueError("every class must appear in each subject/seed scoring unit")
    predictions = probability_array.argmax(axis=1)
    recalls = [
        float(np.mean(predictions[label_array == label] == label))
        for label in range(n_classes)
    ]
    return float(np.mean(recalls))


def _compute_frozen_primary_statistic(
    scoring_records: Sequence[ScoringRecord],
    manifest: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    statistic = manifest["analysis_plan"]["primary_statistic"]
    if statistic.get("name") != PRIMARY_STATISTIC_NAME:
        raise ValueError("freeze requests an unsupported primary statistic")
    candidate = str(statistic["candidate_model"])
    comparator = str(statistic["comparator_model"])
    profile = str(statistic["preprocessing_profile"])
    selected = [
        item
        for item in scoring_records
        if item.record.preprocessing_profile == profile
        and item.record.model in {candidate, comparator}
    ]
    grouped: dict[tuple[str, int, int, str], list[ScoringRecord]] = {}
    for item in selected:
        expected_role = "model" if item.record.model == candidate else "comparator"
        if item.record.role != expected_role:
            raise ValueError("primary model role differs from the frozen statistic")
        key = (
            item.record.dataset,
            item.record.subject,
            item.record.seed,
            item.record.model,
        )
        grouped.setdefault(key, []).append(item)

    design = manifest["study_design"]
    seeds = tuple(int(value) for value in design["seeds"])
    subject_seed_scores: dict[tuple[str, int, int, str], float] = {}
    for dataset, plan in design["datasets"].items():
        n_classes = int(plan["dataset_spec"]["n_classes"])
        expected_folds = set(int(value) for value in plan["folds"])
        for subject, seed, model in product(
            plan["subjects"],
            seeds,
            (candidate, comparator),
        ):
            key = (dataset, int(subject), seed, model)
            fold_records = grouped.get(key)
            if not fold_records:
                raise ValueError(f"primary statistic is missing records for {key}")
            observed_folds = {item.record.fold for item in fold_records}
            if observed_folds != expected_folds or len(fold_records) != len(
                expected_folds
            ):
                raise ValueError(f"primary statistic has incomplete folds for {key}")
            rows_seen: set[int] = set()
            labels: list[int] = []
            probabilities: list[tuple[float, ...]] = []
            for item in sorted(fold_records, key=lambda value: value.record.fold):
                overlap = rows_seen.intersection(item.test_rows)
                if overlap:
                    raise ValueError(
                        f"primary statistic folds overlap for {key}: {overlap}"
                    )
                rows_seen.update(item.test_rows)
                labels.extend(item.labels)
                probabilities.extend(item.probabilities)
            subject_seed_scores[key] = _balanced_accuracy(
                labels,
                probabilities,
                n_classes=n_classes,
            )

    subject_deltas: dict[str, dict[str, float]] = {}
    dataset_deltas: dict[str, float] = {}
    for dataset, plan in design["datasets"].items():
        current_subjects: dict[str, float] = {}
        for subject in plan["subjects"]:
            candidate_score = float(
                np.mean(
                    [
                        subject_seed_scores[(dataset, int(subject), seed, candidate)]
                        for seed in seeds
                    ]
                )
            )
            comparator_score = float(
                np.mean(
                    [
                        subject_seed_scores[(dataset, int(subject), seed, comparator)]
                        for seed in seeds
                    ]
                )
            )
            current_subjects[str(subject)] = candidate_score - comparator_score
        subject_deltas[dataset] = current_subjects
        dataset_deltas[dataset] = float(np.mean(list(current_subjects.values())))
    primary_value = float(np.mean(list(dataset_deltas.values())))
    details = {
        "schema": "ieee-mi-primary-statistic-details-v1",
        "candidate_model": candidate,
        "comparator_model": comparator,
        "preprocessing_profile": profile,
        "dataset_mean_paired_deltas": dataset_deltas,
        "subject_paired_deltas": subject_deltas,
        "n_datasets": len(dataset_deltas),
        "n_subjects": sum(len(values) for values in subject_deltas.values()),
        "seeds": list(seeds),
    }
    return primary_value, details


def _validate_primary_statistic_details(
    details: Any,
    *,
    manifest: Mapping[str, Any],
    primary_value: float,
) -> None:
    """Verify every label-free aggregation identity in the score receipt."""

    if (
        not isinstance(details, dict)
        or set(details) != _PRIMARY_STATISTIC_DETAILS_FIELDS
        or details.get("schema") != "ieee-mi-primary-statistic-details-v1"
        or _contains_forbidden_key(details, _FORBIDDEN_SCORE_OUTPUT_KEYS)
    ):
        raise ValueError("score receipt primary details are invalid")
    statistic = manifest["analysis_plan"]["primary_statistic"]
    expected_identity = {
        "candidate_model": statistic["candidate_model"],
        "comparator_model": statistic["comparator_model"],
        "preprocessing_profile": statistic["preprocessing_profile"],
        "seeds": manifest["study_design"]["seeds"],
    }
    for name, expected in expected_identity.items():
        if details.get(name) != expected:
            raise ValueError(f"score receipt primary details have the wrong {name}")

    design_datasets = manifest["study_design"]["datasets"]
    dataset_deltas = details.get("dataset_mean_paired_deltas")
    subject_deltas = details.get("subject_paired_deltas")
    if not isinstance(dataset_deltas, dict) or set(dataset_deltas) != set(
        design_datasets
    ):
        raise ValueError("score receipt dataset deltas differ from the freeze")
    if not isinstance(subject_deltas, dict) or set(subject_deltas) != set(
        design_datasets
    ):
        raise ValueError("score receipt subject deltas differ from the freeze")

    def finite_number(name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"score receipt {name} is not numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"score receipt {name} is not finite")
        return result

    expected_subject_count = 0
    recomputed_dataset_deltas: list[float] = []
    for dataset, plan in design_datasets.items():
        current_subjects = subject_deltas[dataset]
        expected_subjects = {str(int(subject)) for subject in plan["subjects"]}
        if not isinstance(current_subjects, dict) or set(current_subjects) != (
            expected_subjects
        ):
            raise ValueError(f"score receipt subject identities differ for {dataset}")
        subject_values = [
            finite_number(f"{dataset} subject delta", value)
            for value in current_subjects.values()
        ]
        expected_subject_count += len(subject_values)
        recomputed = float(np.mean(subject_values))
        recorded = finite_number(
            f"{dataset} dataset delta",
            dataset_deltas[dataset],
        )
        if not math.isclose(recorded, recomputed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"score receipt dataset delta is inconsistent for {dataset}"
            )
        recomputed_dataset_deltas.append(recomputed)
    if details.get("n_datasets") != len(design_datasets):
        raise ValueError("score receipt dataset count differs from the freeze")
    if details.get("n_subjects") != expected_subject_count:
        raise ValueError("score receipt subject count differs from the freeze")
    recomputed_primary = float(np.mean(recomputed_dataset_deltas))
    if not math.isclose(
        float(primary_value),
        recomputed_primary,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("score receipt primary value is inconsistent with its details")


def _score_confirmation_once_for_trusted_backend(
    manifest_path: str | Path,
    run_root: str | Path,
    label_callback: LabelCallback,
    *,
    capability: object,
    project_root: str | Path,
) -> dict[str, Any]:
    """Internal one-shot scoring hook for the trusted sealed-label backend.

    A scoring-start receipt is published before any label callback.  If the
    scorer fails, the receipt remains and automatic retries are forbidden;
    that fail-closed run requires an explicit external protocol audit.
    """

    root = _require_trusted_backend_capability(
        capability,
        manifest_path=manifest_path,
        project_root=project_root,
        run_root=run_root,
        operation="score",
    )
    with _exclusive_run_lock(root):
        manifest, records, started = _load_run_context(
            manifest_path,
            root,
            project_root=project_root,
        )
        if (root / SCORE_RECEIPT).exists():
            raise FileExistsError("confirmation score receipt already exists")
        if (root / SCORING_STARTED_RECEIPT).exists():
            raise PermissionError("confirmation scoring has already been attempted")
        artifacts = _collect_complete_predictions(root, records, manifest)
        completed = _read_canonical_json(root / COMPLETED_RECEIPT)
        _validate_completed_receipt(
            completed,
            root=root,
            records=records,
            manifest=manifest,
            started=started,
            artifacts=artifacts,
        )
        scoring_started = _signed_receipt(
            {
                "schema": CONFIRMATION_SCORING_STARTED_SCHEMA,
                "mode": "confirmation",
                "phase": "scoring_started",
                "confirmation_evidence": False,
                "manifest_sha256": manifest["manifest_sha256"],
                "completed_receipt_sha256": completed["receipt_sha256"],
                "prediction_inventory_sha256": completed["prediction_inventory_sha256"],
                "started_at": _utc_now(),
            }
        )
        _atomic_write_once(root / SCORING_STARTED_RECEIPT, scoring_started)

        labels_by_split: dict[tuple[str, int, int], tuple[int, ...]] = {}
        for record in records:
            split_key = (
                record.dataset,
                record.subject,
                record.fold,
            )
            if split_key in labels_by_split:
                continue
            artifact = artifacts[record.record_id]
            rows = tuple(int(value) for value in artifact["test_rows"])
            raw_labels = label_callback(record.split, rows)
            if isinstance(raw_labels, (str, bytes)):
                raise ValueError("label callback must return a sequence")
            try:
                label_values = list(raw_labels)
            except TypeError as error:
                raise ValueError("label callback must return a sequence") from error
            if any(
                isinstance(label, bool) or not isinstance(label, (int, np.integer))
                for label in label_values
            ):
                raise ValueError("label callback must return integer labels")
            labels = tuple(int(label) for label in label_values)
            if len(labels) != len(rows):
                raise ValueError("label callback returned the wrong row count")
            n_classes = _dataset_classes(manifest, record.dataset)
            if any(label < 0 or label >= n_classes for label in labels):
                raise ValueError(
                    "label callback returned labels outside the class range"
                )
            labels_by_split[split_key] = labels

        scoring_records: list[ScoringRecord] = []
        for record in records:
            artifact = artifacts[record.record_id]
            split_key = (
                record.dataset,
                record.subject,
                record.fold,
            )
            scoring_records.append(
                ScoringRecord(
                    record=record,
                    test_rows=tuple(int(value) for value in artifact["test_rows"]),
                    probabilities=tuple(
                        tuple(float(value) for value in row)
                        for row in artifact["probabilities"]
                    ),
                    labels=labels_by_split[split_key],
                )
            )
        primary_value, primary_details = _compute_frozen_primary_statistic(
            scoring_records,
            manifest,
        )
        _validate_primary_statistic_details(
            primary_details,
            manifest=manifest,
            primary_value=primary_value,
        )
        current_manifest = validate_freeze_manifest(
            manifest_path,
            project_root=project_root,
        )
        if current_manifest["manifest_sha256"] != manifest["manifest_sha256"]:
            raise PermissionError("freeze manifest changed during scoring")
        analysis_plan = manifest["analysis_plan"]
        success = _evaluate_success(
            analysis_plan["success_rule"],
            primary_value,
        )
        receipt = _signed_receipt(
            {
                "schema": CONFIRMATION_SCORE_SCHEMA,
                "mode": "confirmation",
                "phase": "scored",
                "confirmation_evidence": True,
                "manifest_sha256": manifest["manifest_sha256"],
                "started_receipt_sha256": started["receipt_sha256"],
                "completed_receipt_sha256": completed["receipt_sha256"],
                "scoring_started_receipt_sha256": scoring_started["receipt_sha256"],
                "prediction_inventory_sha256": completed["prediction_inventory_sha256"],
                "analysis_plan": analysis_plan,
                "primary_statistic_value": primary_value,
                "primary_statistic_details": primary_details,
                "success": success,
                "scored_at": _utc_now(),
            }
        )
        _atomic_write_once(root / SCORE_RECEIPT, receipt)
        return receipt


def score_confirmation_once(
    manifest_path: str | Path,
    run_root: str | Path,
    label_callback: LabelCallback,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Refuse arbitrary label callbacks until trusted integration exists."""

    del manifest_path, run_root, label_callback, project_root
    raise PermissionError(_TRUSTED_INTEGRATION_ERROR)


def validate_score_receipt(
    manifest_path: str | Path,
    run_root: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Validate the one-shot score against the frozen prediction inventory."""

    root = Path(run_root)
    with _exclusive_run_lock(root):
        manifest, _, started = _load_run_context(
            manifest_path,
            root,
            project_root=project_root,
        )
        completed = _validate_prediction_phase_unlocked(
            manifest_path,
            root,
            project_root=project_root,
        )
        scoring_started = _read_canonical_json(root / SCORING_STARTED_RECEIPT)
        _validate_receipt(
            scoring_started,
            schema=CONFIRMATION_SCORING_STARTED_SCHEMA,
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
        if scoring_started.get("phase") != "scoring_started":
            raise ValueError("scoring-started receipt has the wrong phase")
        if scoring_started.get("confirmation_evidence") is not False:
            raise ValueError("scoring-started receipt incorrectly claims evidence")
        if scoring_started.get("completed_receipt_sha256") != completed.get(
            "receipt_sha256"
        ):
            raise ValueError("scoring-started receipt references the wrong completion")
        if scoring_started.get("prediction_inventory_sha256") != completed.get(
            "prediction_inventory_sha256"
        ):
            raise ValueError("scoring-started receipt references the wrong predictions")
        score = _read_canonical_json(root / SCORE_RECEIPT)
        _validate_receipt(
            score,
            schema=CONFIRMATION_SCORE_SCHEMA,
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
        if score.get("phase") != "scored":
            raise ValueError("score receipt has the wrong phase")
        if score.get("confirmation_evidence") is not True:
            raise ValueError("score receipt does not identify confirmation evidence")
        if score.get("started_receipt_sha256") != started.get("receipt_sha256"):
            raise ValueError("score receipt references the wrong run start")
        if score.get("completed_receipt_sha256") != completed.get("receipt_sha256"):
            raise ValueError("score receipt references the wrong prediction completion")
        if score.get("scoring_started_receipt_sha256") != scoring_started.get(
            "receipt_sha256"
        ):
            raise ValueError("score receipt references the wrong scoring start")
        if score.get("prediction_inventory_sha256") != completed.get(
            "prediction_inventory_sha256"
        ):
            raise ValueError("score receipt references the wrong predictions")
        if score.get("analysis_plan") != manifest.get("analysis_plan"):
            raise ValueError("score receipt analysis plan differs from the freeze")
        value = score.get("primary_statistic_value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score receipt primary value is invalid")
        if not math.isfinite(float(value)):
            raise ValueError("score receipt primary value must be finite")
        details = score.get("primary_statistic_details")
        _validate_primary_statistic_details(
            details,
            manifest=manifest,
            primary_value=float(value),
        )
        expected_success = _evaluate_success(
            manifest["analysis_plan"]["success_rule"],
            float(value),
        )
        if score.get("success") is not expected_success:
            raise ValueError(
                "score receipt success decision differs from the frozen rule"
            )
        return score
