from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark import native_transfer as transfer
from benchmark import native_transfer_audit as audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_record(
    root: Path,
    key: audit.TransferRecordKey,
    *,
    labels: tuple[int, ...],
    predictions: tuple[int, ...],
    rows: tuple[int, ...] | None = None,
) -> Path:
    path = root / audit.record_directory_name(key)
    path.mkdir()
    labels_array = np.asarray(labels, dtype=np.int64)
    predicted_array = np.asarray(predictions, dtype=np.int64)
    if rows is None:
        rows = tuple(range(len(labels)))
    rows_array = np.asarray(rows, dtype=np.int64)
    probabilities = np.full((len(labels), 2), 0.1, dtype=np.float64)
    probabilities[np.arange(len(labels)), predicted_array] = 0.9
    sessions = np.asarray(["session"] * len(labels))
    runs = np.asarray(["run"] * len(labels))
    predictions_path = path / transfer.PREDICTIONS_FILENAME
    np.savez_compressed(
        predictions_path,
        schema=np.asarray(transfer.PREDICTION_SCHEMA),
        dataset=np.asarray(key.dataset),
        subject=np.asarray(key.subject, dtype=np.int64),
        fold=np.asarray(key.fold, dtype=np.int64),
        condition=np.asarray(key.condition),
        test_rows=rows_array,
        test_labels=labels_array,
        predicted_labels=predicted_array,
        probabilities=probabilities,
        sessions=sessions,
        runs=runs,
    )
    digest = hashlib.sha256(b"locked-checkpoint").hexdigest()
    cache_digest = hashlib.sha256(f"cache-{key.subject}".encode()).hexdigest()
    provenance = {
        "schema": transfer.ARTIFACT_SCHEMA,
        "mode": "development",
        "confirmation_access": False,
        "evidence_scope": transfer.EVIDENCE_SCOPE,
        "analysis_policy": transfer.ANALYSIS_POLICY,
        "record": key.as_dict(),
        "target": {
            "role": "development_target_transfer",
            "montage_profile": "native",
            "cache_file_sha256": cache_digest,
            "raw_array_sha256": hashlib.sha256(b"raw").hexdigest(),
            "positions_sha256": hashlib.sha256(b"positions").hexdigest(),
            "split": {
                "test": {
                    "count": len(labels),
                    "rows_sha256": transfer._array_sha256(rows_array),
                    "labels_sha256": transfer._array_sha256(labels_array),
                }
            },
        },
        "source_checkpoint": {
            "file_sha256": digest,
            "state_sha256": hashlib.sha256(b"state").hexdigest(),
            "size_bytes": 123,
        },
        "condition": {"requested": key.condition},
        "protocol": {
            "test": {
                "aggregate_metrics_emitted": False,
                "inferential_statistics_emitted": False,
            }
        },
        "predictions": {
            "schema": transfer.PREDICTION_SCHEMA,
            "filename": transfer.PREDICTIONS_FILENAME,
            "file_sha256": _sha256(predictions_path),
            "size_bytes": predictions_path.stat().st_size,
            "row_count": len(labels),
            "test_rows_sha256": transfer._array_sha256(rows_array),
            "test_labels_sha256": transfer._array_sha256(labels_array),
            "predicted_labels_sha256": transfer._array_sha256(predicted_array),
            "probabilities_sha256": transfer._array_sha256(probabilities),
        },
        "source_code": {
            "hash_algorithm": "sha256",
            "files": {"src/benchmark/native_transfer.py": hashlib.sha256(b"source").hexdigest()},
        },
    }
    (path / transfer.PROVENANCE_FILENAME).write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    return path


def _key(subject: int, condition: str) -> audit.TransferRecordKey:
    return audit.TransferRecordKey(
        dataset="cho2017",
        subject=subject,
        fold=0,
        seed=7,
        condition=condition,
    )


def test_complete_grid_validates_hashes_and_produces_descriptive_subject_summary(
    tmp_path: Path,
) -> None:
    conditions = (
        transfer.PRETRAINED_CARDINAL_FBC,
        transfer.SCRATCH_CARDINAL_FBC,
    )
    # Subject 16: pretrained BA 1.0, scratch BA 0.5.
    _write_record(tmp_path, _key(16, conditions[0]), labels=(0, 0, 1, 1), predictions=(0, 0, 1, 1))
    _write_record(tmp_path, _key(16, conditions[1]), labels=(0, 0, 1, 1), predictions=(0, 1, 1, 0))
    # Subject 17: pretrained BA 0.5, scratch BA 0.0.
    _write_record(tmp_path, _key(17, conditions[0]), labels=(0, 0, 1, 1), predictions=(0, 1, 1, 0))
    _write_record(tmp_path, _key(17, conditions[1]), labels=(0, 0, 1, 1), predictions=(1, 1, 0, 0))

    result = audit.audit_transfer_directory(
        tmp_path,
        dataset="cho2017",
        subjects=(16, 17),
        folds=(0,),
        seeds=(7,),
        conditions=conditions,
    )
    assert result.complete
    assert len(result.records) == 4
    assert not result.missing_keys
    assert all(len(record.predictions_file_sha256) == 64 for record in result.records)
    assert all(len(record.provenance_file_sha256) == 64 for record in result.records)

    summary = audit.descriptive_transfer_summary(result)
    assert summary["complete_grid"] is True
    assert summary["inferential_statistics"] is False
    pretrained = summary["conditions"][conditions[0]]
    scratch = summary["conditions"][conditions[1]]
    assert pretrained["subject_balanced_accuracy"] == {"16": 1.0, "17": 0.5}
    assert pretrained["equal_subject_mean_balanced_accuracy"] == 0.75
    assert scratch["subject_balanced_accuracy"] == {"16": 0.5, "17": 0.0}
    assert scratch["equal_subject_mean_balanced_accuracy"] == 0.25


def test_missing_records_are_reported_and_complete_summary_fails_closed(
    tmp_path: Path,
) -> None:
    condition = transfer.PRETRAINED_CARDINAL_FBC
    _write_record(
        tmp_path,
        _key(16, condition),
        labels=(0, 1),
        predictions=(0, 1),
    )
    result = audit.audit_transfer_directory(
        tmp_path,
        dataset="cho2017",
        subjects=(16, 17),
        folds=(0,),
        seeds=(7,),
        conditions=(condition,),
    )
    assert not result.complete
    assert result.missing_keys == (_key(17, condition),)
    manifest = audit.audit_manifest(result)
    assert manifest["validated_record_count"] == 1
    assert manifest["missing_record_count"] == 1
    with pytest.raises(RuntimeError, match="incomplete grid"):
        audit.descriptive_transfer_summary(result)
    partial = audit.descriptive_transfer_summary(result, require_complete=False)
    assert partial["complete_grid"] is False


def test_prediction_byte_corruption_is_rejected(tmp_path: Path) -> None:
    key = _key(16, transfer.PRETRAINED_CARDINAL_FBC)
    path = _write_record(
        tmp_path,
        key,
        labels=(0, 1),
        predictions=(0, 1),
    )
    with (path / transfer.PREDICTIONS_FILENAME).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(audit.TransferArtifactError, match="SHA-256 mismatch"):
        audit.validate_transfer_artifact(path, expected_key=key)


def test_nonfinite_probabilities_are_rejected_after_consistent_rehash(
    tmp_path: Path,
) -> None:
    key = _key(16, transfer.PRETRAINED_CARDINAL_FBC)
    path = _write_record(
        tmp_path,
        key,
        labels=(0, 1),
        predictions=(0, 1),
    )
    prediction_path = path / transfer.PREDICTIONS_FILENAME
    with np.load(prediction_path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["probabilities"] = values["probabilities"].copy()
    values["probabilities"][0, 0] = np.nan
    np.savez_compressed(prediction_path, **values)
    provenance_path = path / transfer.PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["predictions"]["file_sha256"] = _sha256(prediction_path)
    provenance["predictions"]["size_bytes"] = prediction_path.stat().st_size
    provenance["predictions"]["probabilities_sha256"] = transfer._array_sha256(
        values["probabilities"]
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(audit.TransferArtifactError, match="non-finite"):
        audit.validate_transfer_artifact(path, expected_key=key)


def test_sealed_grid_rejects_before_root_access(tmp_path: Path) -> None:
    nonexistent_root = tmp_path / "must-not-be-probed"
    with pytest.raises(PermissionError, match="outside|sealed"):
        audit.audit_transfer_directory(
            nonexistent_root,
            dataset="physionet_mi",
            subjects=(55,),
            folds=(0,),
            seeds=(7,),
            conditions=(transfer.PRETRAINED_CARDINAL_FBC,),
        )


def test_default_name_template_rejects_fold_or_seed_collisions_before_io(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="collides"):
        audit.audit_transfer_directory(
            tmp_path / "does-not-exist",
            dataset="cho2017",
            subjects=(16,),
            folds=(0, 1),
            seeds=(7,),
            conditions=(transfer.PRETRAINED_CARDINAL_FBC,),
        )
