from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import benchmark.freeze as freeze_module
from benchmark.config import dataset_spec
from benchmark.freeze import (
    FREEZE_MANIFEST_SCHEMA,
    create_freeze_manifest,
    main,
    manifest_token,
    validate_freeze_manifest,
)


@pytest.fixture(autouse=True)
def _synthetic_fold_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        freeze_module,
        "registered_confirmation_folds",
        lambda dataset: (0,),
    )


def _synthetic_project(root: Path) -> Path:
    project = root / "synthetic-project"
    package = project / "src" / "benchmark"
    tests = project / "tests" / "core"
    package.mkdir(parents=True)
    tests.mkdir(parents=True)
    (package / "__init__.py").write_text('"""synthetic"""\n', encoding="utf-8")
    (package / "model.py").write_text("MODEL = 'frozen'\n", encoding="utf-8")
    (tests / "test_model.py").write_text("def test_model(): pass\n", encoding="utf-8")
    (package / "requirements-cu128.txt").write_text(
        "numpy==2.4.4\ntorch==2.12.1\n",
        encoding="utf-8",
    )
    checkpoints = project / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "pretrained.pt").write_bytes(b"synthetic-checkpoint-v1")
    return project


def _freeze_specification() -> dict[str, object]:
    return {
        "architecture": {
            "synthetic_cardinal": {
                "module": "benchmark.models",
                "class": "SyntheticCardinalNet",
                "kwargs": {"width": 32, "spatial_rank": 8},
                "pretrained_checkpoint": "native_pretraining",
            },
            "fbmsnet": {
                "module": "benchmark.baselines",
                "class": "FBMSNet",
                "kwargs": {"n_bands": 9},
                "pretrained_checkpoint": None,
            },
        },
        "training_config": {
            "synthetic_cardinal": {
                "epochs": 200,
                "batch_size": 64,
                "optimizer": "AdamW",
                "selection": "source_validation_cross_entropy",
            },
            "fbmsnet": {
                "epochs": 200,
                "batch_size": 64,
                "optimizer": "AdamW",
                "selection": "source_validation_cross_entropy",
            },
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
        "primary_hypothesis": (
            "SyntheticCardinal has higher subject-level balanced accuracy than FBMSNet."
        ),
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


def _create(root: Path) -> tuple[Path, Path, dict[str, object]]:
    project = _synthetic_project(root)
    manifest_path = root / "freeze.json"
    manifest = create_freeze_manifest(
        _freeze_specification(),
        manifest_path,
        project_root=project,
    )
    return project, manifest_path, manifest


def test_freeze_manifest_hashes_complete_source_checkpoint_and_protocol(
    tmp_path: Path,
) -> None:
    project, manifest_path, manifest = _create(tmp_path)

    assert manifest["schema"] == FREEZE_MANIFEST_SCHEMA
    assert manifest["immutable"] is True
    assert len(manifest["source_inventory"]["python_sources"]) == 3
    assert len(manifest["source_inventory"]["requirements"]) == 1
    assert len(manifest["source_inventory"]["inventory_sha256"]) == 64
    checkpoint = manifest["pretrained_checkpoints"]["native_pretraining"]
    assert checkpoint["path_kind"] == "project_relative"
    assert checkpoint["path"] == "checkpoints/pretrained.pt"
    assert len(checkpoint["sha256"]) == 64
    assert manifest["study_design"]["models"] == ["synthetic_cardinal"]
    assert manifest["study_design"]["comparators"] == ["fbmsnet"]
    assert manifest["study_design"]["seeds"] == [7, 17, 27, 37, 47]
    candidate_contract = manifest["model_contracts"]["synthetic_cardinal"]
    comparator_contract = manifest["model_contracts"]["fbmsnet"]
    assert candidate_contract["role"] == "model"
    assert candidate_contract["pretrained_checkpoint_name"] == "native_pretraining"
    assert candidate_contract["pretrained_checkpoint_sha256"] == checkpoint["sha256"]
    assert comparator_contract["role"] == "comparator"
    assert comparator_contract["pretrained_checkpoint_name"] is None
    assert comparator_contract["pretrained_checkpoint_sha256"] is None
    assert len(candidate_contract["model_contract_sha256"]) == 64
    profile = manifest["study_design"]["datasets"]["zhou2016"][
        "preprocessing_profiles"
    ]["harmonized"]
    assert profile["dataset"] == "zhou2016"
    assert profile["montage_profile"] == "harmonized"
    assert profile["preprocessing"]["schema"] == "eeg-mi-cache-v2"
    assert profile["channel_scaling"]["schema"] == "eeg-mi-channel-scaling-v1"
    assert manifest_token(manifest) == manifest["manifest_sha256"]
    assert manifest_token(manifest_path) == manifest["manifest_sha256"]
    assert (
        validate_freeze_manifest(
            manifest_path,
            project_root=project,
        )
        == manifest
    )


def test_freeze_creation_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    project, manifest_path, original = _create(tmp_path)
    before = manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        create_freeze_manifest(
            _freeze_specification(),
            manifest_path,
            project_root=project,
        )
    assert manifest_path.read_bytes() == before
    assert json.loads(before)["manifest_sha256"] == original["manifest_sha256"]


@pytest.mark.parametrize("mutation", ("source", "added_source", "checkpoint"))
def test_freeze_validation_detects_every_referenced_byte_change(
    tmp_path: Path,
    mutation: str,
) -> None:
    project, manifest_path, _ = _create(tmp_path)
    if mutation == "source":
        (project / "src" / "benchmark" / "model.py").write_text(
            "MODEL = 'changed'\n",
            encoding="utf-8",
        )
        message = "source/requirements bytes differ"
    elif mutation == "added_source":
        (project / "src" / "benchmark" / "new_source.py").write_text(
            "NEW = True\n",
            encoding="utf-8",
        )
        message = "source/requirements bytes differ"
    else:
        (project / "checkpoints" / "pretrained.pt").write_bytes(
            b"synthetic-checkpoint-v2"
        )
        message = "checkpoint.*bytes changed"
    with pytest.raises(ValueError, match=message):
        validate_freeze_manifest(manifest_path, project_root=project)


def test_freeze_validation_rejects_manifest_content_tampering(tmp_path: Path) -> None:
    project, manifest_path, manifest = _create(tmp_path)
    manifest["analysis_plan"]["primary_hypothesis"] = "Changed after freeze"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash token does not match"):
        validate_freeze_manifest(manifest_path, project_root=project)


def test_freeze_validation_rejects_byte_only_manifest_rewrites(tmp_path: Path) -> None:
    project, manifest_path, _ = _create(tmp_path)
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen canonical encoding"):
        validate_freeze_manifest(manifest_path, project_root=project)
    with pytest.raises(ValueError, match="frozen canonical encoding"):
        manifest_token(manifest_path)


def test_freeze_parser_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    project, manifest_path, _ = _create(tmp_path)
    content = manifest_path.read_text(encoding="utf-8")
    content = content.replace(
        '"immutable":true',
        '"immutable":true,"immutable":true',
        1,
    )
    manifest_path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read JSON artifact"):
        validate_freeze_manifest(manifest_path, project_root=project)


def test_checkpoint_symlinks_are_rejected_at_creation_and_revalidation(
    tmp_path: Path,
) -> None:
    creation_root = tmp_path / "creation"
    project = _synthetic_project(creation_root)
    linked = project / "checkpoints" / "linked.pt"
    linked.symlink_to(project / "checkpoints" / "pretrained.pt")
    specification = _freeze_specification()
    specification["pretrained_checkpoints"][
        "native_pretraining"
    ] = "checkpoints/linked.pt"
    with pytest.raises(ValueError, match="must not be a symlink"):
        create_freeze_manifest(
            specification,
            creation_root / "freeze.json",
            project_root=project,
        )

    replacement_root = tmp_path / "replacement"
    project, manifest_path, _ = _create(replacement_root)
    checkpoint = project / "checkpoints" / "pretrained.pt"
    replacement = project / "checkpoints" / "replacement.pt"
    replacement.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    checkpoint.symlink_to(replacement)
    with pytest.raises(ValueError, match="must not be a symlink"):
        validate_freeze_manifest(manifest_path, project_root=project)


def test_scratch_only_models_freeze_with_an_empty_checkpoint_inventory(
    tmp_path: Path,
) -> None:
    project = _synthetic_project(tmp_path)
    specification = _freeze_specification()
    specification["pretrained_checkpoints"] = {}
    specification["architecture"]["synthetic_cardinal"]["pretrained_checkpoint"] = None
    manifest = create_freeze_manifest(
        specification,
        tmp_path / "scratch-freeze.json",
        project_root=project,
    )
    assert manifest["pretrained_checkpoints"] == {}
    assert (
        manifest["model_contracts"]["synthetic_cardinal"][
            "pretrained_checkpoint_sha256"
        ]
        is None
    )
    assert (
        manifest["model_contracts"]["fbmsnet"]["pretrained_checkpoint_sha256"] is None
    )


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    (
        ("few_seeds", ValueError, "at least 5"),
        ("overlap", ValueError, "models and comparators overlap"),
        ("partial_subjects", PermissionError, "registered cohort"),
        ("invalid_folds", PermissionError, "source-registered contract"),
        ("development_dataset", PermissionError, "no sealed confirmation cohort"),
        ("missing_statistic", ValueError, "primary_statistic fields differ"),
        ("invalid_success", ValueError, "success_rule.operator"),
        ("missing_model_recipe", ValueError, "exactly every model/comparator"),
        ("unknown_checkpoint", ValueError, "unknown checkpoint"),
        ("unassigned_checkpoint", ValueError, "unreferenced"),
    ),
)
def test_freeze_rejects_incomplete_or_ambiguous_scientific_contracts(
    tmp_path: Path,
    mutation: str,
    error: type[Exception],
    message: str,
) -> None:
    project = _synthetic_project(tmp_path)
    specification = _freeze_specification()
    if mutation == "few_seeds":
        specification["seeds"] = [7, 17, 27, 37]
    elif mutation == "overlap":
        specification["comparators"] = ["synthetic_cardinal"]
    elif mutation == "partial_subjects":
        specification["datasets"]["zhou2016"]["subjects"] = [1, 2, 3]
    elif mutation == "invalid_folds":
        specification["datasets"]["zhou2016"]["folds"] = [999]
    elif mutation == "development_dataset":
        specification["datasets"] = {
            "bnci2014_001": {
                "subjects": [],
                "folds": [0],
                "preprocessing_profiles": [{"name": "harmonized"}],
            }
        }
    elif mutation == "missing_statistic":
        specification["primary_statistic"] = {"unit": "proportion"}
    elif mutation == "invalid_success":
        specification["success_rule"] = {"operator": "approximately", "threshold": 0}
    elif mutation == "missing_model_recipe":
        specification["architecture"].pop("fbmsnet")
    elif mutation == "unknown_checkpoint":
        specification["architecture"]["synthetic_cardinal"][
            "pretrained_checkpoint"
        ] = "not_frozen"
    else:
        specification["architecture"]["synthetic_cardinal"][
            "pretrained_checkpoint"
        ] = None
    with pytest.raises(error, match=message):
        create_freeze_manifest(
            specification,
            tmp_path / "invalid.json",
            project_root=project,
        )


def test_freeze_cli_creates_and_validates_without_rewriting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _synthetic_project(tmp_path)
    specification_path = tmp_path / "specification.json"
    specification_path.write_text(
        json.dumps(_freeze_specification()),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "cli-freeze.json"
    main(
        [
            "create",
            "--spec",
            str(specification_path),
            "--output",
            str(manifest_path),
            "--project-root",
            str(project),
        ]
    )
    created_token = capsys.readouterr().out.strip()
    assert created_token == manifest_token(manifest_path)
    before = manifest_path.read_bytes()
    main(
        [
            "validate",
            str(manifest_path),
            "--project-root",
            str(project),
        ]
    )
    assert capsys.readouterr().out.strip() == created_token
    assert manifest_path.read_bytes() == before
