from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ieee_mi.confirmation as confirmation_module
import ieee_mi.data as data_module
import ieee_mi.freeze as freeze_module
from ieee_mi.config import channels_for_dataset, dataset_spec
from ieee_mi.confirmation import (
    COMPLETED_RECEIPT,
    CONFIRMATION_COMPLETED_SCHEMA,
    CONFIRMATION_PREDICTION_SCHEMA,
    CONFIRMATION_SCORE_SCHEMA,
    CONFIRMATION_SCORING_STARTED_SCHEMA,
    CONFIRMATION_STARTED_SCHEMA,
    PREDICTION_DIRECTORY,
    SCORE_RECEIPT,
    SCORING_STARTED_RECEIPT,
    ConfirmationRecord,
    _initialize_confirmation_run_for_trusted_backend as _trusted_initialize_run,
    _record_confirmation_prediction_for_trusted_backend as _trusted_record_prediction,
    _run_confirmation_predictions_for_trusted_backend as _trusted_run_predictions,
    _score_confirmation_once_for_trusted_backend as _trusted_score_once,
    expected_confirmation_records,
    prediction_path,
    validate_prediction_phase,
    validate_score_receipt,
)
from ieee_mi.freeze import create_freeze_manifest

_SYNTHETIC_CAPABILITY = object()


def record_confirmation_prediction(*args, **kwargs):
    return _trusted_record_prediction(
        *args,
        capability=_SYNTHETIC_CAPABILITY,
        **kwargs,
    )


def initialize_confirmation_run(*args, **kwargs):
    return _trusted_initialize_run(
        *args,
        capability=_SYNTHETIC_CAPABILITY,
        **kwargs,
    )


def run_confirmation_predictions(*args, **kwargs):
    return _trusted_run_predictions(
        *args,
        capability=_SYNTHETIC_CAPABILITY,
        **kwargs,
    )


def score_confirmation_once(*args, **kwargs):
    return _trusted_score_once(
        *args,
        capability=_SYNTHETIC_CAPABILITY,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _synthetic_protocol_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        freeze_module,
        "registered_confirmation_folds",
        lambda dataset: (0,),
    )

    def validate_capability(
        capability: object,
        *,
        manifest_path: Path,
        project_root: Path,
        run_root: Path,
        operation: str,
    ) -> bool:
        assert capability is _SYNTHETIC_CAPABILITY
        assert isinstance(manifest_path, Path)
        assert isinstance(project_root, Path)
        assert isinstance(run_root, Path)
        assert run_root.is_absolute()
        assert operation in {
            "initialize",
            "record_prediction",
            "run_predictions",
            "score",
        }
        return True

    monkeypatch.setattr(
        data_module,
        "validate_confirmation_execution_capability",
        validate_capability,
        raising=False,
    )


def _synthetic_project(root: Path) -> Path:
    project = root / "synthetic-project"
    package = project / "ieee_mi"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""synthetic"""\n', encoding="utf-8")
    (package / "runner.py").write_text(
        "FROZEN_RUNNER = 'v1'\n",
        encoding="utf-8",
    )
    (package / "requirements-cu128.txt").write_text(
        "numpy==2.4.4\ntorch==2.12.1\n",
        encoding="utf-8",
    )
    checkpoints = project / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "pretrained.pt").write_bytes(b"synthetic-checkpoint-v1")
    return project


def _specification() -> dict[str, object]:
    return {
        "architecture": {
            "synthetic_cardinal": {
                "class": "SyntheticCardinal",
                "width": 32,
                "pretrained_checkpoint": "native_pretraining",
            },
            "fbmsnet": {
                "class": "FBMSNet",
                "n_bands": 9,
                "pretrained_checkpoint": None,
            },
        },
        "training_config": {
            "synthetic_cardinal": {"epochs": 200, "batch_size": 64},
            "fbmsnet": {"epochs": 200, "batch_size": 64},
        },
        "pretrained_checkpoints": {
            "native_pretraining": "checkpoints/pretrained.pt",
        },
        "models": ["synthetic_cardinal"],
        "comparators": ["fbmsnet"],
        "seeds": [7, 17, 27, 37, 47],
        "datasets": {
            "zhou2016": {
                "subjects": list(dataset_spec("zhou2016").confirmation_subjects),
                "folds": [0],
                "preprocessing_profiles": [{"name": "harmonized"}],
            }
        },
        "primary_hypothesis": "SyntheticCardinal exceeds FBMSNet.",
        "primary_statistic": {
            "name": "equal_dataset_mean_subject_balanced_accuracy_delta",
            "candidate_model": "synthetic_cardinal",
            "comparator_model": "fbmsnet",
            "preprocessing_profile": "harmonized",
            "metric": "balanced_accuracy",
            "fold_aggregation": "concatenate_nonoverlapping_test_predictions",
            "seed_aggregation": "mean",
            "subject_aggregation": "mean",
            "dataset_aggregation": "equal_weight",
        },
        "success_rule": {"operator": ">", "threshold": 0.0},
    }


def _frozen_run(root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    project = _synthetic_project(root)
    manifest_path = root / "freeze.json"
    manifest = create_freeze_manifest(
        _specification(),
        manifest_path,
        project_root=project,
    )
    run_root = root / "confirmation-run"
    return project, manifest_path, run_root, manifest


def _frozen_two_profile_run(
    root: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    project = _synthetic_project(root)
    specification = _specification()
    native_channels = channels_for_dataset("zhou2016", "harmonized")
    assert native_channels is not None
    specification["datasets"]["zhou2016"]["preprocessing_profiles"].append(
        {"name": "native", "channels": list(native_channels)}
    )
    manifest_path = root / "freeze.json"
    manifest = create_freeze_manifest(
        specification,
        manifest_path,
        project_root=project,
    )
    return project, manifest_path, root / "confirmation-run", manifest


def _prediction_output(
    record: ConfirmationRecord,
    manifest: dict[str, object],
) -> dict[str, object]:
    if record.role == "model":
        probabilities = [[0.9, 0.1], [0.1, 0.9]]
    else:
        probabilities = [[0.6, 0.4], [0.6, 0.4]]
    cache_hash = hashlib.sha256(
        f"{record.dataset}:{record.preprocessing_profile}:{record.subject}".encode()
    ).hexdigest()
    contract = manifest["model_contracts"][record.model]
    return {
        "test_rows": [10, 11],
        "probabilities": probabilities,
        "sessions": ["test", "test"],
        "runs": ["0", "0"],
        "provenance": {
            "cache_array_sha256": cache_hash,
            "model_state_sha256": hashlib.sha256(record.record_id.encode()).hexdigest(),
            "model_contract_sha256": contract["model_contract_sha256"],
            "architecture_sha256": contract["architecture_sha256"],
            "training_config_sha256": contract["training_config_sha256"],
            "pretrained_checkpoint_name": contract["pretrained_checkpoint_name"],
            "pretrained_checkpoint_sha256": contract["pretrained_checkpoint_sha256"],
            "environment": {"python": "synthetic", "device": "cpu"},
        },
    }


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = {str(key) for key in value}
        for child in value.values():
            result.update(_all_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_all_keys(child))
        return result
    return set()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def test_started_receipt_binds_exact_cartesian_manifest_and_refuses_reuse(
    tmp_path: Path,
) -> None:
    project, manifest_path, run_root, manifest = _frozen_run(tmp_path)
    records = expected_confirmation_records(manifest)
    assert len(records) == 4 * 1 * 1 * 2 * 5
    assert len({record.record_id for record in records}) == len(records)
    receipt = initialize_confirmation_run(
        manifest_path,
        run_root,
        project_root=project,
    )
    assert receipt["schema"] == CONFIRMATION_STARTED_SCHEMA
    assert receipt["manifest_sha256"] == manifest["manifest_sha256"]
    assert receipt["expected_record_count"] == len(records)
    assert len(receipt["expected_records_sha256"]) == 64
    assert (run_root / PREDICTION_DIRECTORY).is_dir()
    with pytest.raises(FileExistsError):
        initialize_confirmation_run(
            manifest_path,
            run_root,
            project_root=project,
        )


def test_public_execution_apis_fail_closed_without_trusted_backend(
    tmp_path: Path,
) -> None:
    project, manifest_path, run_root, manifest = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    record = expected_confirmation_records(manifest)[0]
    calls = {"prediction": 0, "labels": 0}

    def prediction(record: ConfirmationRecord, frozen: dict[str, object]):
        calls["prediction"] += 1
        return _prediction_output(record, frozen)

    def labels(split, rows):
        calls["labels"] += 1
        return [0, 1]

    with pytest.raises(PermissionError, match="trusted data/model backend"):
        confirmation_module.initialize_confirmation_run(
            manifest_path,
            tmp_path / "untrusted-second-run",
            project_root=project,
        )
    with pytest.raises(PermissionError, match="trusted data/model backend"):
        confirmation_module.run_confirmation_predictions(
            manifest_path,
            run_root,
            prediction,
            project_root=project,
        )
    with pytest.raises(PermissionError, match="trusted data/model backend"):
        confirmation_module.record_confirmation_prediction(
            manifest_path,
            run_root,
            record,
            _prediction_output(record, manifest),
            project_root=project,
        )
    with pytest.raises(PermissionError, match="trusted data/model backend"):
        confirmation_module.score_confirmation_once(
            manifest_path,
            run_root,
            labels,
            project_root=project,
        )
    assert calls == {"prediction": 0, "labels": 0}
    assert list((run_root / PREDICTION_DIRECTORY).iterdir()) == []
    assert not (run_root / SCORING_STARTED_RECEIPT).exists()


def test_internal_execution_hook_requires_backend_owned_capability(
    tmp_path: Path,
) -> None:
    project, manifest_path, run_root, _ = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    calls = {"prediction": 0}

    def prediction(record: ConfirmationRecord, frozen: dict[str, object]):
        calls["prediction"] += 1
        return _prediction_output(record, frozen)

    patched_validator = data_module.validate_confirmation_execution_capability
    data_module.validate_confirmation_execution_capability = lambda **kwargs: False
    try:
        with pytest.raises(PermissionError, match="trusted data/model backend"):
            _trusted_run_predictions(
                manifest_path,
                run_root,
                prediction,
                capability=object(),
                project_root=project,
            )
    finally:
        data_module.validate_confirmation_execution_capability = patched_validator
    assert calls == {"prediction": 0}
    assert list((run_root / PREDICTION_DIRECTORY).iterdir()) == []


def test_initialization_does_not_replace_a_broken_run_symlink(tmp_path: Path) -> None:
    project, manifest_path, run_root, _ = _frozen_run(tmp_path)
    run_root.symlink_to(tmp_path / "missing-run-target", target_is_directory=True)
    with pytest.raises(PermissionError, match="must not be a symlink"):
        initialize_confirmation_run(
            manifest_path,
            run_root,
            project_root=project,
        )
    assert run_root.is_symlink()


def test_prediction_runner_resumes_only_missing_records_and_never_reruns_completed(
    tmp_path: Path,
) -> None:
    project, manifest_path, run_root, manifest = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    first_attempt: list[str] = []

    def interrupted(record: ConfirmationRecord, frozen: dict[str, object]):
        first_attempt.append(record.record_id)
        if len(first_attempt) == 4:
            raise RuntimeError("synthetic interruption")
        return _prediction_output(record, frozen)

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_confirmation_predictions(
            manifest_path,
            run_root,
            interrupted,
            project_root=project,
        )
    assert len(list((run_root / PREDICTION_DIRECTORY).glob("*.json"))) == 3
    completed_before_failure = set(first_attempt[:3])
    resumed: list[str] = []

    def resume(record: ConfirmationRecord, frozen: dict[str, object]):
        resumed.append(record.record_id)
        return _prediction_output(record, frozen)

    completed = run_confirmation_predictions(
        manifest_path,
        run_root,
        resume,
        project_root=project,
    )
    assert completed["schema"] == CONFIRMATION_COMPLETED_SCHEMA
    assert completed["record_count"] == len(expected_confirmation_records(manifest))
    assert completed_before_failure.isdisjoint(resumed)
    assert not (run_root / SCORE_RECEIPT).exists()

    rerun_calls: list[str] = []

    def forbidden_rerun(record: ConfirmationRecord, frozen: dict[str, object]):
        rerun_calls.append(record.record_id)
        raise AssertionError("completed callback must not run")

    assert (
        run_confirmation_predictions(
            manifest_path,
            run_root,
            forbidden_rerun,
            project_root=project,
        )
        == completed
    )
    assert rerun_calls == []


def test_prediction_artifacts_reject_labels_metrics_and_overwrite(
    tmp_path: Path,
) -> None:
    project, manifest_path, run_root, manifest = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    record = expected_confirmation_records(manifest)[0]
    output = _prediction_output(record, manifest)
    with_labels = dict(output)
    with_labels["test_labels"] = [0, 1]
    with pytest.raises(PermissionError, match="labels, metrics, or scores"):
        record_confirmation_prediction(
            manifest_path,
            run_root,
            record,
            with_labels,
            project_root=project,
        )
    nested_metrics = _prediction_output(record, manifest)
    nested_metrics["provenance"]["metrics"] = {"balanced_accuracy": 1.0}
    with pytest.raises(PermissionError, match="labels, metrics, or scores"):
        record_confirmation_prediction(
            manifest_path,
            run_root,
            record,
            nested_metrics,
            project_root=project,
        )

    artifact = record_confirmation_prediction(
        manifest_path,
        run_root,
        record,
        output,
        project_root=project,
    )
    assert artifact["schema"] == CONFIRMATION_PREDICTION_SCHEMA
    assert artifact["contains_test_labels"] is False
    assert artifact["contains_metrics"] is False
    forbidden = {"label", "labels", "test_labels", "metrics", "score"}
    assert not (_all_keys(artifact) & forbidden)
    assert prediction_path(run_root, record).exists()
    with pytest.raises(FileExistsError, match="must never be rerun"):
        record_confirmation_prediction(
            manifest_path,
            run_root,
            record,
            output,
            project_root=project,
        )


def test_prediction_provenance_must_match_exact_frozen_model_contract(
    tmp_path: Path,
) -> None:
    project, manifest_path, run_root, manifest = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    record = expected_confirmation_records(manifest)[0]

    wrong_recipe = _prediction_output(record, manifest)
    wrong_recipe["provenance"]["architecture_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="frozen model contract"):
        record_confirmation_prediction(
            manifest_path,
            run_root,
            record,
            wrong_recipe,
            project_root=project,
        )

    ambiguous = _prediction_output(record, manifest)
    ambiguous["provenance"]["runtime_override"] = {"width": 999}
    with pytest.raises(ValueError, match="provenance fields differ"):
        record_confirmation_prediction(
            manifest_path,
            run_root,
            record,
            ambiguous,
            project_root=project,
        )


def test_completion_requires_paired_cartesian_predictions(tmp_path: Path) -> None:
    project, manifest_path, run_root, _ = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)

    def mismatched(record: ConfirmationRecord, manifest: dict[str, object]):
        output = _prediction_output(record, manifest)
        if record.role == "model":
            output["test_rows"] = [12, 13]
        return output

    with pytest.raises(ValueError, match="paired prediction data differ"):
        run_confirmation_predictions(
            manifest_path,
            run_root,
            mismatched,
            project_root=project,
        )
    assert not (run_root / COMPLETED_RECEIPT).exists()


def test_profiles_share_exact_trial_identity_and_one_label_read_per_split(
    tmp_path: Path,
) -> None:
    mismatch_root = tmp_path / "mismatch"
    project, manifest_path, run_root, _ = _frozen_two_profile_run(mismatch_root)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)

    def mismatched_profile(record: ConfirmationRecord, manifest: dict[str, object]):
        output = _prediction_output(record, manifest)
        if record.preprocessing_profile == "native":
            output["test_rows"] = [12, 13]
        return output

    with pytest.raises(ValueError, match="paired prediction data differ"):
        run_confirmation_predictions(
            manifest_path,
            run_root,
            mismatched_profile,
            project_root=project,
        )

    paired_root = tmp_path / "paired"
    project, manifest_path, run_root, _ = _frozen_two_profile_run(paired_root)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    run_confirmation_predictions(
        manifest_path,
        run_root,
        _prediction_output,
        project_root=project,
    )
    label_calls: list[tuple[str, int, int]] = []

    def labels(split, rows):
        label_calls.append((split.dataset, split.subject, split.fold))
        return [0, 1]

    score_confirmation_once(
        manifest_path,
        run_root,
        labels,
        project_root=project,
    )
    assert len(label_calls) == 4
    assert len(set(label_calls)) == 4


def test_scoring_requires_complete_predictions_and_does_not_open_labels_early(
    tmp_path: Path,
) -> None:
    project, manifest_path, run_root, _ = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    calls = {"labels": 0}

    def labels(record: ConfirmationRecord, rows: tuple[int, ...]):
        calls["labels"] += 1
        return [0, 1]

    with pytest.raises(ValueError, match="complete Cartesian design"):
        score_confirmation_once(
            manifest_path,
            run_root,
            labels,
            project_root=project,
        )
    assert calls == {"labels": 0}
    assert not (run_root / SCORING_STARTED_RECEIPT).exists()


def test_one_shot_scoring_joins_labels_after_completion_and_seals_result(
    tmp_path: Path,
) -> None:
    project, manifest_path, run_root, manifest = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    run_confirmation_predictions(
        manifest_path,
        run_root,
        _prediction_output,
        project_root=project,
    )
    label_calls: list[tuple[str, int, int]] = []

    def labels(record: ConfirmationRecord, rows: tuple[int, ...]):
        label_calls.append(
            (
                record.dataset,
                record.subject,
                record.fold,
            )
        )
        assert rows == (10, 11)
        return [0, 1]

    receipt = score_confirmation_once(
        manifest_path,
        run_root,
        labels,
        project_root=project,
    )
    assert receipt["schema"] == CONFIRMATION_SCORE_SCHEMA
    assert receipt["confirmation_evidence"] is True
    assert receipt["manifest_sha256"] == manifest["manifest_sha256"]
    assert receipt["primary_statistic_value"] == 0.5
    assert receipt["primary_statistic_details"]["n_subjects"] == 4
    assert receipt["success"] is True
    assert len(label_calls) == 4
    assert len(set(label_calls)) == 4
    assert not (
        _all_keys(receipt) & {"labels", "test_labels", "probabilities", "test_rows"}
    )
    assert (
        validate_score_receipt(
            manifest_path,
            run_root,
            project_root=project,
        )
        == receipt
    )
    with pytest.raises(FileExistsError, match="already exists"):
        score_confirmation_once(
            manifest_path,
            run_root,
            labels,
            project_root=project,
        )
    with pytest.raises(PermissionError, match="after scoring starts"):
        run_confirmation_predictions(
            manifest_path,
            run_root,
            _prediction_output,
            project_root=project,
        )


def test_scoring_failure_is_fail_closed_and_cannot_be_retried(tmp_path: Path) -> None:
    project, manifest_path, run_root, _ = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    run_confirmation_predictions(
        manifest_path,
        run_root,
        _prediction_output,
        project_root=project,
    )

    def failing_labels(record: ConfirmationRecord, rows: tuple[int, ...]):
        raise RuntimeError("synthetic label-provider failure")

    with pytest.raises(RuntimeError, match="synthetic label-provider failure"):
        score_confirmation_once(
            manifest_path,
            run_root,
            failing_labels,
            project_root=project,
        )
    scoring_started = json.loads(
        (run_root / SCORING_STARTED_RECEIPT).read_text(encoding="utf-8")
    )
    assert scoring_started["schema"] == CONFIRMATION_SCORING_STARTED_SCHEMA
    assert not (run_root / SCORE_RECEIPT).exists()
    with pytest.raises(PermissionError, match="already been attempted"):
        score_confirmation_once(
            manifest_path,
            run_root,
            failing_labels,
            project_root=project,
        )


def test_score_validation_rejects_rehashed_inconsistent_aggregations(
    tmp_path: Path,
) -> None:
    project, manifest_path, run_root, _ = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    run_confirmation_predictions(
        manifest_path,
        run_root,
        _prediction_output,
        project_root=project,
    )
    score_confirmation_once(
        manifest_path,
        run_root,
        lambda split, rows: [0, 1],
        project_root=project,
    )
    target = run_root / SCORE_RECEIPT
    score = json.loads(target.read_text(encoding="utf-8"))
    subject_deltas = score["primary_statistic_details"]["subject_paired_deltas"]
    first_dataset = next(iter(subject_deltas))
    first_subject = next(iter(subject_deltas[first_dataset]))
    subject_deltas[first_dataset][first_subject] += 0.125
    unsigned = dict(score)
    unsigned.pop("receipt_sha256")
    score["receipt_sha256"] = hashlib.sha256(
        _canonical_bytes(unsigned).rstrip(b"\n")
    ).hexdigest()
    target.write_bytes(_canonical_bytes(score))
    with pytest.raises(ValueError, match="dataset delta is inconsistent"):
        validate_score_receipt(
            manifest_path,
            run_root,
            project_root=project,
        )


@pytest.mark.parametrize("mutation", ("tamper", "extra"))
def test_completed_inventory_detects_tampered_or_extra_prediction_files(
    tmp_path: Path,
    mutation: str,
) -> None:
    project, manifest_path, run_root, manifest = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    run_confirmation_predictions(
        manifest_path,
        run_root,
        _prediction_output,
        project_root=project,
    )
    if mutation == "tamper":
        first = expected_confirmation_records(manifest)[0]
        path = prediction_path(run_root, first)
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["probabilities"][0] = [0.5, 0.5]
        path.write_text(
            json.dumps(
                artifact,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        message = "prediction hash does not match"
    else:
        (run_root / PREDICTION_DIRECTORY / "unexpected.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        message = "complete Cartesian design"
    with pytest.raises(ValueError, match=message):
        validate_prediction_phase(
            manifest_path,
            run_root,
            project_root=project,
        )


@pytest.mark.parametrize("artifact_kind", ("prediction", "receipt"))
def test_formal_artifact_schemas_reject_rehashed_extra_fields(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    project, manifest_path, run_root, manifest = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    run_confirmation_predictions(
        manifest_path,
        run_root,
        _prediction_output,
        project_root=project,
    )
    if artifact_kind == "prediction":
        target = prediction_path(run_root, expected_confirmation_records(manifest)[0])
        artifact = json.loads(target.read_text(encoding="utf-8"))
        artifact["unexpected_debug"] = True
        unsigned = dict(artifact)
        unsigned.pop("prediction_sha256")
        artifact["prediction_sha256"] = hashlib.sha256(
            _canonical_bytes(unsigned).rstrip(b"\n")
        ).hexdigest()
        message = "prediction fields differ"
    else:
        target = run_root / COMPLETED_RECEIPT
        artifact = json.loads(target.read_text(encoding="utf-8"))
        artifact["unexpected_debug"] = True
        unsigned = dict(artifact)
        unsigned.pop("receipt_sha256")
        artifact["receipt_sha256"] = hashlib.sha256(
            _canonical_bytes(unsigned).rstrip(b"\n")
        ).hexdigest()
        message = "receipt fields differ"
    target.write_bytes(_canonical_bytes(artifact))
    with pytest.raises(ValueError, match=message):
        validate_prediction_phase(
            manifest_path,
            run_root,
            project_root=project,
        )


def test_current_source_bytes_are_revalidated_before_callbacks(tmp_path: Path) -> None:
    project, manifest_path, run_root, _ = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)
    (project / "ieee_mi" / "runner.py").write_text(
        "FROZEN_RUNNER = 'changed'\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def callback(record: ConfirmationRecord, manifest: dict[str, object]):
        calls.append(record.record_id)
        return _prediction_output(record, manifest)

    with pytest.raises(ValueError, match="source/requirements bytes differ"):
        run_confirmation_predictions(
            manifest_path,
            run_root,
            callback,
            project_root=project,
        )
    assert calls == []


def test_callbacks_cannot_mutate_the_validated_manifest_used_for_receipts(
    tmp_path: Path,
) -> None:
    project, manifest_path, run_root, manifest = _frozen_run(tmp_path)
    initialize_confirmation_run(manifest_path, run_root, project_root=project)

    def prediction(record: ConfirmationRecord, callback_manifest: dict[str, object]):
        output = _prediction_output(record, callback_manifest)
        callback_manifest["manifest_sha256"] = "f" * 64
        callback_manifest["analysis_plan"]["success_rule"]["threshold"] = 100.0
        return output

    completed = run_confirmation_predictions(
        manifest_path,
        run_root,
        prediction,
        project_root=project,
    )
    assert completed["manifest_sha256"] == manifest["manifest_sha256"]

    def labels(split, rows):
        return [0, 1]

    receipt = score_confirmation_once(
        manifest_path,
        run_root,
        labels,
        project_root=project,
    )
    assert receipt["manifest_sha256"] == manifest["manifest_sha256"]
    assert receipt["analysis_plan"]["success_rule"]["threshold"] == 0.0
    assert receipt["success"] is True
