"""Development-only validation and descriptive summaries for transfer records.

This module reads only an explicit Cartesian grid from a caller-supplied
directory.  Every grid key is authorized with :func:`validate_target_record`
before the directory is touched, so the audit surface cannot be used to read
sealed confirmation subjects.  It validates the immutable two-file record
format emitted by :mod:`ieee_mi.native_transfer` and computes descriptive
balanced accuracies only; it exposes no significance test or confirmatory
claim surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .native_transfer import (
    ANALYSIS_POLICY,
    ARTIFACT_SCHEMA,
    EVIDENCE_SCOPE,
    FROZEN_DEVELOPMENT_SEEDS,
    PREDICTION_SCHEMA,
    PREDICTIONS_FILENAME,
    PROVENANCE_FILENAME,
    TRANSFER_CONDITIONS,
    validate_target_record,
)


AUDIT_SCHEMA = "ieee-mi-native-transfer-development-audit-v1"
SUMMARY_SCHEMA = "ieee-mi-native-transfer-development-descriptive-summary-v1"
SUMMARY_ANALYSIS_POLICY = "development_only_descriptive_no_inference"
DEFAULT_RECORD_NAME_TEMPLATE = "s{subject:03d}_{condition}"

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


class TransferArtifactError(RuntimeError):
    """A present native-transfer artifact violates its write-time contract."""


@dataclass(frozen=True)
class TransferAuditContract:
    """Writer identity used by the shared fail-closed artifact auditor."""

    audit_schema: str
    summary_schema: str
    summary_analysis_policy: str
    artifact_schema: str
    prediction_schema: str
    predictions_filename: str
    provenance_filename: str
    evidence_scope: str
    analysis_policy: str
    conditions: tuple[str, ...]
    validator: Callable[[str, int, int, str, int], tuple[str, int, int, str, int]]
    checkpoint_schema: str | None = None
    checkpoint_model_identity: str | None = None
    required_source_files: tuple[str, ...] = ()
    artifact_semantics_validator: Callable[..., Mapping[str, str]] | None = None
    grid_semantics_validator: Callable[..., None] | None = None


LEGACY_AUDIT_CONTRACT = TransferAuditContract(
    audit_schema=AUDIT_SCHEMA,
    summary_schema=SUMMARY_SCHEMA,
    summary_analysis_policy=SUMMARY_ANALYSIS_POLICY,
    artifact_schema=ARTIFACT_SCHEMA,
    prediction_schema=PREDICTION_SCHEMA,
    predictions_filename=PREDICTIONS_FILENAME,
    provenance_filename=PROVENANCE_FILENAME,
    evidence_scope=EVIDENCE_SCOPE,
    analysis_policy=ANALYSIS_POLICY,
    conditions=TRANSFER_CONDITIONS,
    validator=validate_target_record,
)


@dataclass(frozen=True, order=True)
class TransferRecordKey:
    """Canonical identity of one authorized development transfer record."""

    dataset: str
    subject: int
    fold: int
    seed: int
    condition: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "subject": self.subject,
            "fold": self.fold,
            "seed": self.seed,
            "condition": self.condition,
        }


@dataclass(frozen=True)
class ValidatedTransferArtifact:
    """One fully loaded record whose file and array digests were verified."""

    key: TransferRecordKey
    path: Path
    predictions_file_sha256: str
    provenance_file_sha256: str
    checkpoint_file_sha256: str
    cache_file_sha256: str
    source_code_hashes: tuple[tuple[str, str], ...]
    test_rows: IntArray
    test_labels: IntArray
    predicted_labels: IntArray
    probabilities: FloatArray
    sessions: NDArray[np.str_]
    runs: NDArray[np.str_]
    semantic_identity: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TransferDirectoryAudit:
    """Validated present records and absent records for one explicit grid."""

    root: Path
    name_template: str
    expected_keys: tuple[TransferRecordKey, ...]
    records: tuple[ValidatedTransferArtifact, ...]
    missing_keys: tuple[TransferRecordKey, ...]
    contract: TransferAuditContract = LEGACY_AUDIT_CONTRACT

    @property
    def complete(self) -> bool:
        return not self.missing_keys


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TransferArtifactError(f"{name} must be a JSON object")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TransferArtifactError(f"{name} must be an integer")
    return value


def _digest(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise TransferArtifactError(
            f"{name} must be a lowercase 64-character SHA-256 digest"
        )
    return str(value)


def _load_json_object(path: Path) -> Mapping[str, object]:
    def reject_constant(token: str) -> None:
        raise TransferArtifactError(
            f"{path} contains non-finite JSON numeric constant {token}"
        )

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TransferArtifactError(f"could not read valid JSON from {path}") from error
    return _mapping(value, name=str(path))


def _canonical_key(
    dataset: str,
    subject: int,
    fold: int,
    seed: int,
    condition: str,
    *,
    contract: TransferAuditContract = LEGACY_AUDIT_CONTRACT,
) -> TransferRecordKey:
    key, subject_id, fold_id, condition_key, seed_id = contract.validator(
        dataset, subject, fold, condition, seed
    )
    return TransferRecordKey(
        dataset=key,
        subject=subject_id,
        fold=fold_id,
        seed=seed_id,
        condition=condition_key,
    )


def development_transfer_grid(
    *,
    dataset: str,
    subjects: Sequence[int],
    folds: Sequence[int],
    seeds: Sequence[int],
    conditions: Sequence[str] | None = None,
    contract: TransferAuditContract = LEGACY_AUDIT_CONTRACT,
) -> tuple[TransferRecordKey, ...]:
    """Authorize and return a unique, sorted development-only Cartesian grid."""

    selected_conditions: Sequence[str] = (
        contract.conditions if conditions is None else conditions
    )
    dimensions: tuple[tuple[str, Sequence[object]], ...] = (
        ("subjects", subjects),
        ("folds", folds),
        ("seeds", seeds),
        ("conditions", selected_conditions),
    )
    for name, values in dimensions:
        if not values:
            raise ValueError(f"{name} must not be empty")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must not contain duplicates")

    result = tuple(
        sorted(
            _canonical_key(
                dataset,
                subject,
                fold,
                seed,
                condition,
                contract=contract,
            )
            for subject in subjects
            for fold in folds
            for seed in seeds
            for condition in selected_conditions
        )
    )
    if len(result) != len(set(result)):
        raise ValueError("development transfer grid contains duplicate records")
    return result


def record_directory_name(
    key: TransferRecordKey,
    *,
    name_template: str = DEFAULT_RECORD_NAME_TEMPLATE,
) -> str:
    """Render one immediate child directory name for an authorized record."""

    try:
        name = name_template.format(**key.as_dict())
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("invalid record name template") from error
    candidate = Path(name)
    if (
        not name
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or name in {".", ".."}
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    ):
        raise ValueError("record name template must render one relative directory name")
    return name


def _require_exact_record_files(
    path: Path,
    *,
    contract: TransferAuditContract = LEGACY_AUDIT_CONTRACT,
) -> tuple[Path, Path]:
    if path.is_symlink() or not path.is_dir():
        raise TransferArtifactError(f"record path must be a real directory: {path}")
    names = {entry.name for entry in path.iterdir()}
    expected = {contract.predictions_filename, contract.provenance_filename}
    if names != expected:
        raise TransferArtifactError(
            f"record directory {path} must contain exactly {sorted(expected)}; "
            f"observed={sorted(names)}"
        )
    predictions = path / contract.predictions_filename
    provenance = path / contract.provenance_filename
    for artifact in (predictions, provenance):
        if artifact.is_symlink() or not artifact.is_file():
            raise TransferArtifactError(f"record artifact must be a real file: {artifact}")
    return predictions, provenance


def _scalar(archive: Mapping[str, np.ndarray], name: str) -> object:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise TransferArtifactError(f"prediction field {name} must be scalar")
    return value.item()


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


def validate_transfer_artifact(
    path: str | Path,
    *,
    expected_key: TransferRecordKey,
    contract: TransferAuditContract = LEGACY_AUDIT_CONTRACT,
) -> ValidatedTransferArtifact:
    """Validate one present development record, including all embedded hashes."""

    # Authorization deliberately precedes every path operation.
    key = _canonical_key(**expected_key.as_dict(), contract=contract)
    record_path = Path(path)
    predictions_path, provenance_path = _require_exact_record_files(
        record_path, contract=contract
    )
    provenance = _load_json_object(provenance_path)

    if provenance.get("schema") != contract.artifact_schema:
        raise TransferArtifactError(f"stale provenance schema in {provenance_path}")
    if provenance.get("mode") != "development":
        raise TransferArtifactError("native-transfer artifact is not development mode")
    if provenance.get("confirmation_access") is not False:
        raise TransferArtifactError("native-transfer artifact reports confirmation access")
    if provenance.get("evidence_scope") != contract.evidence_scope:
        raise TransferArtifactError("native-transfer evidence scope differs from writer")
    if provenance.get("analysis_policy") != contract.analysis_policy:
        raise TransferArtifactError("native-transfer analysis policy differs from writer")

    record = _mapping(provenance.get("record"), name="provenance.record")
    observed_key = _canonical_key(
        str(record.get("dataset", "")),
        _integer(record.get("subject"), name="record.subject"),
        _integer(record.get("fold"), name="record.fold"),
        _integer(record.get("seed"), name="record.seed"),
        str(record.get("condition", "")),
        contract=contract,
    )
    if observed_key != key:
        raise TransferArtifactError(
            f"record identity differs from expected grid key: "
            f"expected={key.as_dict()}, observed={observed_key.as_dict()}"
        )

    prediction_record = _mapping(
        provenance.get("predictions"), name="provenance.predictions"
    )
    if prediction_record.get("schema") != contract.prediction_schema:
        raise TransferArtifactError("provenance prediction schema is stale")
    if prediction_record.get("filename") != contract.predictions_filename:
        raise TransferArtifactError("provenance prediction filename is invalid")
    observed_file_hash = _sha256_file(predictions_path)
    expected_file_hash = _digest(
        prediction_record.get("file_sha256"), name="predictions.file_sha256"
    )
    if observed_file_hash != expected_file_hash:
        raise TransferArtifactError(f"prediction file SHA-256 mismatch in {record_path}")
    expected_size = _integer(
        prediction_record.get("size_bytes"), name="predictions.size_bytes"
    )
    if expected_size <= 0 or predictions_path.stat().st_size != expected_size:
        raise TransferArtifactError(f"prediction file size mismatch in {record_path}")

    exact_fields = {
        "schema",
        "dataset",
        "subject",
        "fold",
        "condition",
        "test_rows",
        "test_labels",
        "predicted_labels",
        "probabilities",
        "sessions",
        "runs",
    }
    try:
        with np.load(predictions_path, allow_pickle=False) as archive:
            if set(archive.files) != exact_fields:
                raise TransferArtifactError(
                    "prediction archive fields differ from the writer contract"
                )
            if str(_scalar(archive, "schema")) != contract.prediction_schema:
                raise TransferArtifactError("prediction archive schema is stale")
            scalar_identity = {
                "dataset": str(_scalar(archive, "dataset")),
                "subject": int(_scalar(archive, "subject")),
                "fold": int(_scalar(archive, "fold")),
                "condition": str(_scalar(archive, "condition")),
            }
            expected_identity = {
                "dataset": key.dataset,
                "subject": key.subject,
                "fold": key.fold,
                "condition": key.condition,
            }
            if scalar_identity != expected_identity:
                raise TransferArtifactError("prediction archive identity differs from grid")
            test_rows = np.asarray(archive["test_rows"])
            test_labels = np.asarray(archive["test_labels"])
            predicted_labels = np.asarray(archive["predicted_labels"])
            probabilities = np.asarray(archive["probabilities"])
            sessions = np.asarray(archive["sessions"])
            runs = np.asarray(archive["runs"])
    except TransferArtifactError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise TransferArtifactError(
            f"could not load a safe NumPy archive from {predictions_path}"
        ) from error

    if test_rows.dtype != np.dtype(np.int64) or test_rows.ndim != 1:
        raise TransferArtifactError("test_rows must be a one-dimensional int64 array")
    row_count = len(test_rows)
    if row_count == 0 or len(np.unique(test_rows)) != row_count:
        raise TransferArtifactError("test_rows must be nonempty and unique")
    if np.any(test_rows < 0):
        raise TransferArtifactError("test_rows contains a negative index")
    for name, values in (
        ("test_labels", test_labels),
        ("predicted_labels", predicted_labels),
    ):
        if values.dtype != np.dtype(np.int64) or values.shape != (row_count,):
            raise TransferArtifactError(f"{name} must have shape ({row_count},) and int64 dtype")
        if not set(values.tolist()) <= {0, 1}:
            raise TransferArtifactError(f"{name} contains a non-binary class label")
    if probabilities.dtype != np.dtype(np.float64) or probabilities.shape != (row_count, 2):
        raise TransferArtifactError(
            f"probabilities must have shape ({row_count}, 2) and float64 dtype"
        )
    if not np.isfinite(probabilities).all():
        raise TransferArtifactError("probabilities contains non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise TransferArtifactError("probabilities lies outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise TransferArtifactError("probability rows do not sum to one")
    if not np.array_equal(predicted_labels, probabilities.argmax(axis=1)):
        raise TransferArtifactError("predicted_labels differs from probability argmax")
    for name, values in (("sessions", sessions), ("runs", runs)):
        if values.shape != (row_count,) or values.dtype.kind not in {"U", "S"}:
            raise TransferArtifactError(
                f"{name} must have shape ({row_count},) and string dtype"
            )

    if _integer(prediction_record.get("row_count"), name="predictions.row_count") != row_count:
        raise TransferArtifactError("provenance prediction row count mismatch")
    arrays = {
        "test_rows": test_rows,
        "test_labels": test_labels,
        "probabilities": probabilities,
        "predicted_labels": predicted_labels,
    }
    for name, values in arrays.items():
        expected_hash = _digest(
            prediction_record.get(f"{name}_sha256"),
            name=f"predictions.{name}_sha256",
        )
        if _array_sha256(values) != expected_hash:
            raise TransferArtifactError(f"provenance {name} SHA-256 mismatch")

    target = _mapping(provenance.get("target"), name="provenance.target")
    if target.get("role") != "development_target_transfer":
        raise TransferArtifactError("target role is not development transfer")
    if target.get("montage_profile") != "native":
        raise TransferArtifactError("target montage profile is not native")
    cache_hash = _digest(target.get("cache_file_sha256"), name="target.cache_file_sha256")
    _digest(target.get("raw_array_sha256"), name="target.raw_array_sha256")
    _digest(target.get("positions_sha256"), name="target.positions_sha256")
    split = _mapping(target.get("split"), name="target.split")
    test_split = _mapping(split.get("test"), name="target.split.test")
    if _integer(test_split.get("count"), name="target.split.test.count") != row_count:
        raise TransferArtifactError("target test split count differs from predictions")
    for field, values in (("rows_sha256", test_rows), ("labels_sha256", test_labels)):
        if _digest(test_split.get(field), name=f"target.split.test.{field}") != _array_sha256(values):
            raise TransferArtifactError(f"target test split {field} mismatch")

    checkpoint = _mapping(
        provenance.get("source_checkpoint"), name="provenance.source_checkpoint"
    )
    if (
        contract.checkpoint_schema is not None
        and checkpoint.get("schema") != contract.checkpoint_schema
    ):
        raise TransferArtifactError("source checkpoint schema differs from family")
    if contract.checkpoint_model_identity is not None:
        checkpoint_model = _mapping(
            checkpoint.get("model"), name="source_checkpoint.model"
        )
        if checkpoint_model.get("identity") != contract.checkpoint_model_identity:
            raise TransferArtifactError("source checkpoint model differs from family")
    checkpoint_hash = _digest(
        checkpoint.get("file_sha256"), name="source_checkpoint.file_sha256"
    )
    _digest(checkpoint.get("state_sha256"), name="source_checkpoint.state_sha256")
    if _integer(checkpoint.get("size_bytes"), name="source_checkpoint.size_bytes") <= 0:
        raise TransferArtifactError("source checkpoint size must be positive")

    condition_record = _mapping(provenance.get("condition"), name="provenance.condition")
    if condition_record.get("requested") != key.condition:
        raise TransferArtifactError("condition provenance differs from record identity")
    protocol = _mapping(provenance.get("protocol"), name="provenance.protocol")
    test_protocol = _mapping(protocol.get("test"), name="provenance.protocol.test")
    if test_protocol.get("aggregate_metrics_emitted") is not False:
        raise TransferArtifactError("writer artifact unexpectedly contains aggregate metrics")
    if test_protocol.get("inferential_statistics_emitted") is not False:
        raise TransferArtifactError("writer artifact unexpectedly contains inference")

    source_code = _mapping(provenance.get("source_code"), name="provenance.source_code")
    if source_code.get("hash_algorithm") != "sha256":
        raise TransferArtifactError("source-code hash algorithm is not SHA-256")
    files = _mapping(source_code.get("files"), name="provenance.source_code.files")
    if not files:
        raise TransferArtifactError("source-code hash manifest is empty")
    missing_source_files = sorted(set(contract.required_source_files) - set(files))
    if missing_source_files:
        raise TransferArtifactError(
            f"source-code manifest misses family files {missing_source_files}"
        )
    source_hashes = tuple(
        sorted(
            (str(name), _digest(value, name=f"source_code.files[{name!r}]"))
            for name, value in files.items()
        )
    )
    semantic_identity: tuple[tuple[str, str], ...] = ()
    if contract.artifact_semantics_validator is not None:
        semantics = contract.artifact_semantics_validator(provenance, key)
        if not isinstance(semantics, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in semantics.items()
        ):
            raise TransferArtifactError(
                "family artifact semantic validator returned an invalid identity"
            )
        semantic_identity = tuple(sorted(semantics.items()))

    return ValidatedTransferArtifact(
        key=key,
        path=record_path,
        predictions_file_sha256=observed_file_hash,
        provenance_file_sha256=_sha256_file(provenance_path),
        checkpoint_file_sha256=checkpoint_hash,
        cache_file_sha256=cache_hash,
        source_code_hashes=source_hashes,
        test_rows=_readonly(test_rows),
        test_labels=_readonly(test_labels),
        predicted_labels=_readonly(predicted_labels),
        probabilities=_readonly(probabilities),
        sessions=_readonly(sessions.astype(str, copy=False)),
        runs=_readonly(runs.astype(str, copy=False)),
        semantic_identity=semantic_identity,
    )


def _validate_cross_record_provenance(
    records: Sequence[ValidatedTransferArtifact],
    *,
    contract: TransferAuditContract = LEGACY_AUDIT_CONTRACT,
) -> None:
    if not records:
        return
    checkpoint_hashes = {record.checkpoint_file_sha256 for record in records}
    if len(checkpoint_hashes) != 1:
        raise TransferArtifactError("grid records do not share one source checkpoint")
    source_manifests = {record.source_code_hashes for record in records}
    if len(source_manifests) != 1:
        raise TransferArtifactError("grid records do not share one source-code manifest")

    caches: dict[tuple[str, int], str] = {}
    splits: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        subject_key = (record.key.dataset, record.key.subject)
        previous_cache = caches.setdefault(subject_key, record.cache_file_sha256)
        if previous_cache != record.cache_file_sha256:
            raise TransferArtifactError(f"target cache changed within subject {subject_key}")
        split_key = (record.key.dataset, record.key.subject, record.key.fold)
        previous_split = splits.setdefault(
            split_key, (record.test_rows, record.test_labels)
        )
        if not np.array_equal(previous_split[0], record.test_rows) or not np.array_equal(
            previous_split[1], record.test_labels
        ):
            raise TransferArtifactError(
                f"test split changed across seed/condition records for {split_key}"
            )
    if contract.grid_semantics_validator is not None:
        contract.grid_semantics_validator(records)


def audit_transfer_directory(
    root: str | Path,
    *,
    dataset: str,
    subjects: Sequence[int],
    folds: Sequence[int],
    seeds: Sequence[int],
    conditions: Sequence[str] | None = None,
    name_template: str = DEFAULT_RECORD_NAME_TEMPLATE,
    contract: TransferAuditContract = LEGACY_AUDIT_CONTRACT,
) -> TransferDirectoryAudit:
    """Validate present records and report missing keys for an explicit grid."""

    # Complete authorization and collision checks happen before touching root.
    expected = development_transfer_grid(
        dataset=dataset,
        subjects=subjects,
        folds=folds,
        seeds=seeds,
        conditions=conditions,
        contract=contract,
    )
    names: dict[str, TransferRecordKey] = {}
    for key in expected:
        name = record_directory_name(key, name_template=name_template)
        if name in names:
            raise ValueError(
                "record name template collides for distinct grid keys; "
                f"{names[name].as_dict()} and {key.as_dict()} both render {name!r}"
            )
        names[name] = key

    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise FileNotFoundError(f"audit root must be a real directory: {root_path}")
    records: list[ValidatedTransferArtifact] = []
    missing: list[TransferRecordKey] = []
    for name, key in names.items():
        path = root_path / name
        try:
            path.lstat()
        except FileNotFoundError:
            missing.append(key)
            continue
        records.append(
            validate_transfer_artifact(path, expected_key=key, contract=contract)
        )
    records.sort(key=lambda record: record.key)
    missing.sort()
    _validate_cross_record_provenance(records, contract=contract)
    return TransferDirectoryAudit(
        root=root_path,
        name_template=name_template,
        expected_keys=expected,
        records=tuple(records),
        missing_keys=tuple(missing),
        contract=contract,
    )


def audit_manifest(audit: TransferDirectoryAudit) -> dict[str, object]:
    """Return a JSON-safe validation manifest without computing any metric."""

    return {
        "schema": audit.contract.audit_schema,
        "mode": "development",
        "confirmation_access": False,
        "analysis_policy": audit.contract.summary_analysis_policy,
        "root": str(audit.root),
        "name_template": audit.name_template,
        "complete": audit.complete,
        "expected_record_count": len(audit.expected_keys),
        "validated_record_count": len(audit.records),
        "missing_record_count": len(audit.missing_keys),
        "missing_records": [key.as_dict() for key in audit.missing_keys],
        "validated_records": [
            {
                **record.key.as_dict(),
                "path": str(record.path),
                "predictions_file_sha256": record.predictions_file_sha256,
                "provenance_file_sha256": record.provenance_file_sha256,
            }
            for record in audit.records
        ],
    }


def _balanced_accuracy(labels: IntArray, predictions: IntArray) -> float:
    recalls: list[float] = []
    for label in (0, 1):
        rows = labels == label
        if not np.any(rows):
            raise TransferArtifactError(
                "balanced accuracy requires both binary classes in each subject/seed unit"
            )
        recalls.append(float(np.mean(predictions[rows] == label)))
    return float(np.mean(recalls))


def descriptive_transfer_summary(
    audit: TransferDirectoryAudit,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Summarize subject balanced accuracy without inferential statistics.

    Outer-fold predictions are concatenated within each subject/seed/condition,
    then seed-level balanced accuracies are averaged within subject.  Dataset
    condition means weight each observed subject equally.
    """

    if require_complete and not audit.complete:
        raise RuntimeError(
            f"refusing to summarize an incomplete grid with "
            f"{len(audit.missing_keys)} missing records"
        )
    if not audit.records:
        raise RuntimeError("cannot summarize a grid with no validated records")

    grouped: dict[
        tuple[str, int, int], list[ValidatedTransferArtifact]
    ] = {}
    for record in audit.records:
        grouped.setdefault(
            (record.key.condition, record.key.subject, record.key.seed), []
        ).append(record)

    subject_seed_scores: dict[tuple[str, int, int], float] = {}
    for key, fold_records in grouped.items():
        rows_seen: set[int] = set()
        labels: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        for record in sorted(fold_records, key=lambda item: item.key.fold):
            overlap = rows_seen.intersection(record.test_rows.tolist())
            if overlap:
                raise TransferArtifactError(
                    f"outer test folds overlap for condition/subject/seed {key}: "
                    f"{sorted(overlap)}"
                )
            rows_seen.update(record.test_rows.tolist())
            labels.append(record.test_labels)
            predictions.append(record.predicted_labels)
        subject_seed_scores[key] = _balanced_accuracy(
            np.concatenate(labels).astype(np.int64, copy=False),
            np.concatenate(predictions).astype(np.int64, copy=False),
        )

    by_condition_subject: dict[str, dict[int, list[tuple[int, float]]]] = {}
    for (condition, subject, seed), score in subject_seed_scores.items():
        by_condition_subject.setdefault(condition, {}).setdefault(subject, []).append(
            (seed, score)
        )

    conditions: dict[str, object] = {}
    for condition, subject_scores in sorted(by_condition_subject.items()):
        per_subject = {
            subject: float(np.mean([score for _, score in seed_scores]))
            for subject, seed_scores in sorted(subject_scores.items())
        }
        values = np.asarray(list(per_subject.values()), dtype=np.float64)
        condition_records = [
            record for record in audit.records if record.key.condition == condition
        ]
        conditions[condition] = {
            "n_records_observed": len(condition_records),
            "n_subject_seed_units_observed": sum(
                len(seed_scores) for seed_scores in subject_scores.values()
            ),
            "n_subjects_observed": len(per_subject),
            "equal_subject_mean_balanced_accuracy": float(values.mean()),
            "subject_balanced_accuracy_sample_std": (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ),
            "subject_balanced_accuracy": {
                str(subject): score for subject, score in per_subject.items()
            },
            "subject_seed_balanced_accuracy": {
                f"{subject}:{seed}": score
                for subject, seed_scores in sorted(subject_scores.items())
                for seed, score in sorted(seed_scores)
            },
        }

    return {
        "schema": audit.contract.summary_schema,
        "mode": "development",
        "confirmation_access": False,
        "confirmatory_evidence": False,
        "analysis_policy": audit.contract.summary_analysis_policy,
        "inferential_statistics": False,
        "complete_grid": audit.complete,
        "expected_record_count": len(audit.expected_keys),
        "validated_record_count": len(audit.records),
        "missing_records": [key.as_dict() for key in audit.missing_keys],
        "aggregation": {
            "folds": "concatenate held-out rows within subject/seed/condition",
            "seeds": "arithmetic mean within subject/condition",
            "subjects": "equal-subject arithmetic mean within condition",
            "metric": "binary balanced accuracy",
        },
        "conditions": conditions,
    }


def _parse_integer_spec(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValueError(f"descending integer range is invalid: {token}")
            result.extend(range(start, stop + 1))
        else:
            result.append(int(token))
    return tuple(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subjects", required=True, help="comma/range list, e.g. 16-52")
    parser.add_argument("--folds", default="0", help="comma/range list")
    parser.add_argument("--seeds", default="7", help="comma/range list")
    parser.add_argument("--conditions", default=",".join(TRANSFER_CONDITIONS))
    parser.add_argument("--record-name-template", default=DEFAULT_RECORD_NAME_TEMPLATE)
    parser.add_argument(
        "--allow-incomplete-summary",
        action="store_true",
        help="emit explicitly marked partial descriptive metrics when records are missing",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    arguments = build_parser().parse_args(argv)
    audit = audit_transfer_directory(
        arguments.root,
        dataset=arguments.dataset,
        subjects=_parse_integer_spec(arguments.subjects),
        folds=_parse_integer_spec(arguments.folds),
        seeds=_parse_integer_spec(arguments.seeds),
        conditions=tuple(
            token.strip() for token in arguments.conditions.split(",") if token.strip()
        ),
        name_template=arguments.record_name_template,
    )
    result = audit_manifest(audit)
    if audit.complete or arguments.allow_incomplete_summary:
        result["descriptive_summary"] = descriptive_transfer_summary(
            audit, require_complete=not arguments.allow_incomplete_summary
        )
    return result


def main() -> None:
    print(json.dumps(run(), sort_keys=True, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
