"""Immutable preregistration manifest for one-shot confirmation experiments.

The manifest freezes executable bytes and the complete scientific decision
contract before any confirmation callback is allowed to run.  This module has
no data-loading entry point and therefore cannot open a sealed cohort itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    CHANNEL_SCALING,
    channels_for_dataset,
    coordinate_contract_for_dataset,
    dataset_spec,
    preprocessing_for_dataset,
    validate_montage_profile,
)

FREEZE_MANIFEST_SCHEMA = "ieee-mi-confirmation-freeze-v1"
PREPROCESSING_PROFILE_SCHEMA = "ieee-mi-preprocessing-profile-v1"
MINIMUM_CONFIRMATION_SEEDS = 5
PRIMARY_STATISTIC_NAME = "equal_dataset_mean_subject_balanced_accuracy_delta"

_SPEC_FIELDS = frozenset(
    {
        "architecture",
        "training_config",
        "pretrained_checkpoints",
        "models",
        "comparators",
        "seeds",
        "datasets",
        "primary_hypothesis",
        "primary_statistic",
        "success_rule",
    }
)
_SUCCESS_OPERATORS = frozenset({">", ">=", "<", "<="})
_SUCCESS_RULE_FIELDS = frozenset({"operator", "threshold"})
_PRIMARY_STATISTIC_FIELDS = frozenset(
    {
        "name",
        "candidate_model",
        "comparator_model",
        "preprocessing_profile",
        "metric",
        "fold_aggregation",
        "seed_aggregation",
        "subject_aggregation",
        "dataset_aggregation",
    }
)
_MODEL_CHECKPOINT_FIELD = "pretrained_checkpoint"
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "created_at",
        "mode",
        "immutable",
        "source_inventory",
        "architecture",
        "architecture_sha256",
        "training_config",
        "training_config_sha256",
        "pretrained_checkpoints",
        "checkpoint_set_sha256",
        "model_contracts",
        "model_contracts_sha256",
        "study_design",
        "study_design_sha256",
        "analysis_plan",
        "analysis_plan_sha256",
        "manifest_sha256",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("freeze values must be finite JSON values") from error
    return serialized.encode("utf-8")


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_bytes(value))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _atomic_write_once(path: Path, value: Mapping[str, Any]) -> None:
    """Publish canonical JSON atomically without an overwrite race."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    payload = _canonical_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".partial",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(target) from None
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact {path} is not an object")
    return value


def _validate_canonical_json_file(path: Path, value: Mapping[str, Any]) -> None:
    """Reject byte-level rewrites even when they parse to the same JSON value."""

    try:
        current = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read JSON artifact {path}") from error
    expected = _canonical_bytes(value) + b"\n"
    if current != expected:
        raise ValueError(
            f"JSON artifact {path} is not in its frozen canonical encoding"
        )


def _unique_strings(name: str, values: Any, *, minimum: int = 1) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a JSON array")
    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"{name} must contain strings")
    result = tuple(value.strip() for value in values)
    if len(result) < minimum or any(not value for value in result):
        raise ValueError(f"{name} must contain at least {minimum} nonempty value(s)")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _unique_integers(name: str, values: Any, *, minimum: int = 1) -> tuple[int, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a JSON array")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{name} must contain integers")
    result = tuple(int(value) for value in values)
    if len(result) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} value(s)")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _nonempty_object(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty JSON object")
    return _json_clone(value)


def _validate_success_rule(value: Any) -> dict[str, Any]:
    rule = _nonempty_object("success_rule", value)
    if set(rule) != _SUCCESS_RULE_FIELDS:
        missing = sorted(_SUCCESS_RULE_FIELDS - set(rule))
        extra = sorted(set(rule) - _SUCCESS_RULE_FIELDS)
        raise ValueError(
            f"success_rule fields differ; missing={missing}, extra={extra}"
        )
    if rule.get("operator") not in _SUCCESS_OPERATORS:
        raise ValueError(
            f"success_rule.operator must be one of {sorted(_SUCCESS_OPERATORS)}"
        )
    threshold = rule.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("success_rule.threshold must be numeric")
    if not math.isfinite(float(threshold)):
        raise ValueError("success_rule.threshold must be finite")
    return rule


def registered_confirmation_folds(dataset: str) -> tuple[int, ...]:
    """Read the independent fold registry owned by the data/config layer.

    No fold count is inferred here. A confirmation freeze fails closed until
    ``config.CONFIRMATION_FOLDS`` explicitly registers the dataset.
    """

    from . import config as config_module

    registry = getattr(config_module, "CONFIRMATION_FOLDS", None)
    if not isinstance(registry, Mapping) or dataset not in registry:
        raise PermissionError(
            f"{dataset} has no source-registered confirmation fold contract"
        )
    raw_folds = registry[dataset]
    if not isinstance(raw_folds, (tuple, list)):
        raise ValueError(f"{dataset} confirmation fold contract is invalid")
    if any(isinstance(fold, bool) or not isinstance(fold, int) for fold in raw_folds):
        raise ValueError(f"{dataset} confirmation folds must be integers")
    folds = tuple(int(fold) for fold in raw_folds)
    if not folds or len(set(folds)) != len(folds) or any(fold < 0 for fold in folds):
        raise ValueError(f"{dataset} confirmation fold contract is invalid")
    return folds


def _normalize_primary_statistic(
    value: Any,
    study_design: Mapping[str, Any],
) -> dict[str, Any]:
    statistic = _nonempty_object("primary_statistic", value)
    if set(statistic) != _PRIMARY_STATISTIC_FIELDS:
        missing = sorted(_PRIMARY_STATISTIC_FIELDS - set(statistic))
        extra = sorted(set(statistic) - _PRIMARY_STATISTIC_FIELDS)
        raise ValueError(
            f"primary_statistic fields differ; missing={missing}, extra={extra}"
        )
    expected_constants = {
        "name": PRIMARY_STATISTIC_NAME,
        "metric": "balanced_accuracy",
        "fold_aggregation": "concatenate_nonoverlapping_test_predictions",
        "seed_aggregation": "mean",
        "subject_aggregation": "mean",
        "dataset_aggregation": "equal_weight",
    }
    for name, expected in expected_constants.items():
        if statistic.get(name) != expected:
            raise ValueError(f"primary_statistic.{name} must be {expected!r}")
    candidate = statistic.get("candidate_model")
    comparator = statistic.get("comparator_model")
    if candidate not in study_design["models"]:
        raise ValueError("primary_statistic.candidate_model is not a frozen model")
    if comparator not in study_design["comparators"]:
        raise ValueError(
            "primary_statistic.comparator_model is not a frozen comparator"
        )
    profile = statistic.get("preprocessing_profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("primary_statistic.preprocessing_profile must be nonempty")
    for dataset, plan in study_design["datasets"].items():
        if profile not in plan["preprocessing_profiles"]:
            raise ValueError(
                f"primary statistic profile {profile!r} is absent from {dataset}"
            )
    return statistic


def current_preprocessing_profile(
    dataset: str,
    montage_profile: str,
    *,
    channel_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Materialize the complete current preprocessing contract for freezing."""

    profile = validate_montage_profile(montage_profile)
    configured_channels = channels_for_dataset(dataset, profile)
    if configured_channels is None:
        if channel_names is None:
            raise ValueError(
                f"{dataset}/{profile} requires its exact native ordered channels"
            )
        ordered_channels = tuple(str(name) for name in channel_names)
    else:
        ordered_channels = tuple(configured_channels)
        if channel_names is not None and tuple(channel_names) != ordered_channels:
            raise ValueError(
                f"{dataset}/{profile} supplied channels differ from configuration"
            )
    if not ordered_channels or len(set(ordered_channels)) != len(ordered_channels):
        raise ValueError("preprocessing profile channels must be nonempty and unique")
    return _json_clone(
        {
            "schema": PREPROCESSING_PROFILE_SCHEMA,
            "dataset": dataset,
            "montage_profile": profile,
            "preprocessing": preprocessing_for_dataset(dataset),
            "channel_scaling": CHANNEL_SCALING,
            "ordered_channels": list(ordered_channels),
            "coordinate_contract": coordinate_contract_for_dataset(
                dataset,
                profile,
                ordered_channels,
            ),
        }
    )


def _normalize_profiles(dataset: str, value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"datasets.{dataset}.preprocessing_profiles must be a nonempty array"
        )
    result: dict[str, dict[str, Any]] = {}
    for raw_profile in value:
        if not isinstance(raw_profile, dict):
            raise ValueError(f"{dataset} preprocessing profile is not an object")
        extra = set(raw_profile) - {"name", "channels"}
        if extra:
            raise ValueError(
                f"{dataset} preprocessing profile has unknown fields {sorted(extra)}"
            )
        name = str(raw_profile.get("name", "")).strip().lower()
        if not name:
            raise ValueError(f"{dataset} preprocessing profile has no name")
        if name in result:
            raise ValueError(f"{dataset} repeats preprocessing profile {name}")
        channels = raw_profile.get("channels")
        if channels is not None and not isinstance(channels, list):
            raise ValueError(f"{dataset}/{name} channels must be an array")
        result[name] = current_preprocessing_profile(
            dataset,
            name,
            channel_names=channels,
        )
    return result


def _normalize_datasets(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("datasets must be a nonempty object")
    result: dict[str, dict[str, Any]] = {}
    for dataset, raw_plan in sorted(value.items()):
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("dataset names must be nonempty strings")
        spec = dataset_spec(dataset)
        if not spec.confirmation_subjects:
            raise PermissionError(f"{dataset} has no sealed confirmation cohort")
        if not isinstance(raw_plan, dict):
            raise ValueError(f"datasets.{dataset} must be an object")
        required = {"subjects", "folds", "preprocessing_profiles"}
        if set(raw_plan) != required:
            raise ValueError(
                f"datasets.{dataset} fields must be exactly {sorted(required)}"
            )
        subjects = _unique_integers(
            f"datasets.{dataset}.subjects", raw_plan["subjects"]
        )
        expected_subjects = tuple(spec.confirmation_subjects)
        if set(subjects) != set(expected_subjects):
            missing = sorted(set(expected_subjects) - set(subjects))
            extra = sorted(set(subjects) - set(expected_subjects))
            raise PermissionError(
                f"{dataset} confirmation subjects differ from the registered cohort; "
                f"missing={missing}, extra={extra}"
            )
        folds = _unique_integers(f"datasets.{dataset}.folds", raw_plan["folds"])
        expected_folds = registered_confirmation_folds(dataset)
        if set(folds) != set(expected_folds):
            missing = sorted(set(expected_folds) - set(folds))
            extra = sorted(set(folds) - set(expected_folds))
            raise PermissionError(
                f"{dataset} folds differ from the source-registered contract; "
                f"missing={missing}, extra={extra}"
            )
        profiles = _normalize_profiles(dataset, raw_plan["preprocessing_profiles"])
        result[dataset] = {
            "dataset_spec": _json_clone(asdict(spec)),
            "subjects": list(subjects),
            "folds": list(folds),
            "preprocessing_profiles": profiles,
        }
    return result


def _normalize_study_design(specification: Mapping[str, Any]) -> dict[str, Any]:
    models = _unique_strings("models", specification["models"])
    comparators = _unique_strings("comparators", specification["comparators"])
    overlap = sorted(set(models) & set(comparators))
    if overlap:
        raise ValueError(f"models and comparators overlap: {overlap}")
    seeds = _unique_integers(
        "seeds",
        specification["seeds"],
        minimum=MINIMUM_CONFIRMATION_SEEDS,
    )
    return {
        "models": list(models),
        "comparators": list(comparators),
        "seeds": list(seeds),
        "datasets": _normalize_datasets(specification["datasets"]),
    }


def _normalize_model_recipes(
    architecture: Any,
    training_config: Any,
    study_design: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Require one explicit architecture and training recipe per frozen model."""

    if not isinstance(study_design, Mapping):
        raise ValueError("study design must be an object before freezing model recipes")
    expected_models = _unique_strings("models", study_design.get("models")) + (
        _unique_strings("comparators", study_design.get("comparators"))
    )
    expected = set(expected_models)
    raw_architecture = _nonempty_object("architecture", architecture)
    raw_training = _nonempty_object("training_config", training_config)
    for name, recipes in (
        ("architecture", raw_architecture),
        ("training_config", raw_training),
    ):
        if set(recipes) != expected:
            missing = sorted(expected - set(recipes))
            extra = sorted(set(recipes) - expected)
            raise ValueError(
                f"{name} must define exactly every model/comparator; "
                f"missing={missing}, extra={extra}"
            )

    architectures: dict[str, dict[str, Any]] = {}
    training_configs: dict[str, dict[str, Any]] = {}
    for model in expected_models:
        recipe = _nonempty_object(
            f"architecture.{model}",
            raw_architecture[model],
        )
        if _MODEL_CHECKPOINT_FIELD not in recipe:
            raise ValueError(
                f"architecture.{model} must explicitly declare "
                f"{_MODEL_CHECKPOINT_FIELD!r} as a checkpoint name or null"
            )
        checkpoint_name = recipe[_MODEL_CHECKPOINT_FIELD]
        if checkpoint_name is not None and (
            not isinstance(checkpoint_name, str) or not checkpoint_name.strip()
        ):
            raise ValueError(
                f"architecture.{model}.{_MODEL_CHECKPOINT_FIELD} must be a "
                "nonempty string or null"
            )
        architectures[model] = recipe
        training_configs[model] = _nonempty_object(
            f"training_config.{model}",
            raw_training[model],
        )
    return architectures, training_configs


def _derive_model_contracts(
    architecture: Mapping[str, Mapping[str, Any]],
    training_config: Mapping[str, Mapping[str, Any]],
    study_design: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bind each model role and recipe to its exact frozen checkpoint bytes."""

    candidate_models = tuple(str(name) for name in study_design["models"])
    comparator_models = tuple(str(name) for name in study_design["comparators"])
    contracts: dict[str, dict[str, Any]] = {}
    referenced_checkpoints: set[str] = set()
    for model, role in (
        *((name, "model") for name in candidate_models),
        *((name, "comparator") for name in comparator_models),
    ):
        architecture_recipe = architecture[model]
        training_recipe = training_config[model]
        checkpoint_name = architecture_recipe[_MODEL_CHECKPOINT_FIELD]
        checkpoint_hash: str | None = None
        if checkpoint_name is not None:
            if checkpoint_name not in checkpoints:
                raise ValueError(
                    f"architecture.{model} references unknown checkpoint "
                    f"{checkpoint_name!r}"
                )
            checkpoint_entry = checkpoints[checkpoint_name]
            if not isinstance(checkpoint_entry, Mapping):
                raise ValueError(
                    f"checkpoint {checkpoint_name!r} has an invalid frozen entry"
                )
            checkpoint_hash = checkpoint_entry.get("sha256")
            if not _is_sha256(checkpoint_hash):
                raise ValueError(
                    f"checkpoint {checkpoint_name!r} has no valid frozen hash"
                )
            referenced_checkpoints.add(checkpoint_name)
        contract: dict[str, Any] = {
            "role": role,
            "architecture_sha256": _json_sha256(architecture_recipe),
            "training_config_sha256": _json_sha256(training_recipe),
            "pretrained_checkpoint_name": checkpoint_name,
            "pretrained_checkpoint_sha256": checkpoint_hash,
        }
        contract["model_contract_sha256"] = _json_sha256(contract)
        contracts[model] = contract
    unreferenced = sorted(set(checkpoints) - referenced_checkpoints)
    if unreferenced:
        raise ValueError(
            "pretrained checkpoints must be assigned to a frozen model; "
            f"unreferenced={unreferenced}"
        )
    return contracts


def _source_entries(project_root: Path) -> dict[str, Any]:
    package_root = project_root / "ieee_mi"
    if not package_root.is_dir():
        raise FileNotFoundError(
            f"IEEE-MI package directory does not exist: {package_root}"
        )
    python_paths = sorted(package_root.rglob("*.py"))
    requirement_paths = sorted(package_root.rglob("requirements*.txt"))
    if not python_paths:
        raise ValueError(f"no IEEE-MI Python source files found below {package_root}")
    if not requirement_paths:
        raise ValueError(f"no IEEE-MI requirements file found below {package_root}")

    def entries(paths: Sequence[Path]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source inventory requires regular files: {path}")
            relative = path.relative_to(project_root).as_posix()
            result.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
        return result

    inventory = {
        "python_sources": entries(python_paths),
        "requirements": entries(requirement_paths),
    }
    inventory["inventory_sha256"] = _json_sha256(inventory)
    return inventory


def _checkpoint_path_entry(path: Path, project_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(project_root)
    except ValueError:
        return {"path_kind": "absolute", "path": str(resolved)}
    return {"path_kind": "project_relative", "path": relative.as_posix()}


def _resolve_checkpoint_entry(entry: Mapping[str, Any], project_root: Path) -> Path:
    kind = entry.get("path_kind")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("checkpoint entry has no path")
    if kind == "project_relative":
        unresolved = project_root / raw_path
        if unresolved.is_symlink():
            raise ValueError("frozen checkpoint path must not be a symlink")
        candidate = unresolved.resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as error:
            raise ValueError("checkpoint relative path escapes project root") from error
        return candidate
    if kind == "absolute" and Path(raw_path).is_absolute():
        candidate = Path(raw_path)
        if candidate.is_symlink():
            raise ValueError("frozen checkpoint path must not be a symlink")
        return candidate
    raise ValueError("checkpoint entry has an invalid path kind")


def _checkpoint_entries(value: Any, project_root: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("pretrained_checkpoints must be an object")
    result: dict[str, dict[str, Any]] = {}
    seen_paths: set[Path] = set()
    for name, raw_path in sorted(value.items()):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("checkpoint names must be nonempty strings")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"checkpoint {name} path must be a nonempty string")
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        if path.is_symlink():
            raise ValueError(f"checkpoint path must not be a symlink: {path}")
        path = path.resolve()
        if path in seen_paths:
            raise ValueError(f"checkpoint file is repeated: {path}")
        seen_paths.add(path)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"checkpoint is not a regular file: {path}")
        result[name] = {
            **_checkpoint_path_entry(path, project_root),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return result


def build_freeze_manifest(
    specification: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Build, but do not write, one complete confirmation freeze manifest."""

    if not isinstance(specification, Mapping):
        raise ValueError("freeze specification must be an object")
    if set(specification) != _SPEC_FIELDS:
        missing = sorted(_SPEC_FIELDS - set(specification))
        extra = sorted(set(specification) - _SPEC_FIELDS)
        raise ValueError(
            f"freeze specification fields differ; missing={missing}, extra={extra}"
        )
    root = Path(project_root).resolve()
    study_design = _normalize_study_design(specification)
    architecture, training_config = _normalize_model_recipes(
        specification["architecture"],
        specification["training_config"],
        study_design,
    )
    hypothesis = str(specification["primary_hypothesis"]).strip()
    if not hypothesis:
        raise ValueError("primary_hypothesis must be a nonempty string")
    statistic = _normalize_primary_statistic(
        specification["primary_statistic"],
        study_design,
    )
    success_rule = _validate_success_rule(specification["success_rule"])
    analysis_plan = {
        "primary_hypothesis": hypothesis,
        "primary_statistic": statistic,
        "success_rule": success_rule,
    }
    checkpoints = _checkpoint_entries(
        specification["pretrained_checkpoints"],
        root,
    )
    model_contracts = _derive_model_contracts(
        architecture,
        training_config,
        study_design,
        checkpoints,
    )
    payload: dict[str, Any] = {
        "schema": FREEZE_MANIFEST_SCHEMA,
        "created_at": _utc_now(),
        "mode": "confirmation_freeze",
        "immutable": True,
        "source_inventory": _source_entries(root),
        "architecture": architecture,
        "architecture_sha256": _json_sha256(architecture),
        "training_config": training_config,
        "training_config_sha256": _json_sha256(training_config),
        "pretrained_checkpoints": checkpoints,
        "checkpoint_set_sha256": _json_sha256(checkpoints),
        "model_contracts": model_contracts,
        "model_contracts_sha256": _json_sha256(model_contracts),
        "study_design": study_design,
        "study_design_sha256": _json_sha256(study_design),
        "analysis_plan": analysis_plan,
        "analysis_plan_sha256": _json_sha256(analysis_plan),
    }
    payload["manifest_sha256"] = _json_sha256(payload)
    return payload


def create_freeze_manifest(
    specification: Mapping[str, Any],
    output: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Build and atomically publish a manifest, refusing every overwrite."""

    manifest = build_freeze_manifest(specification, project_root=project_root)
    _atomic_write_once(Path(output), manifest)
    # Fail closed if any source or checkpoint changed while the inventory was
    # being assembled. The immutable artifact remains available for audit.
    return validate_freeze_manifest(output, project_root=project_root)


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != _MANIFEST_FIELDS:
        missing = sorted(_MANIFEST_FIELDS - set(manifest))
        extra = sorted(set(manifest) - _MANIFEST_FIELDS)
        raise ValueError(
            f"freeze manifest fields differ; missing={missing}, extra={extra}"
        )
    if manifest.get("schema") != FREEZE_MANIFEST_SCHEMA:
        raise ValueError("unsupported freeze manifest schema")
    if (
        manifest.get("mode") != "confirmation_freeze"
        or manifest.get("immutable") is not True
    ):
        raise ValueError("artifact is not an immutable confirmation freeze")
    if not isinstance(manifest.get("created_at"), str) or not manifest["created_at"]:
        raise ValueError("freeze manifest creation timestamp is invalid")
    token = manifest.get("manifest_sha256")
    if not _is_sha256(token):
        raise ValueError("freeze manifest has no valid hash token")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if _json_sha256(unsigned) != token:
        raise ValueError("freeze manifest hash token does not match its contents")
    hashed_fields = (
        ("architecture", "architecture_sha256"),
        ("training_config", "training_config_sha256"),
        ("pretrained_checkpoints", "checkpoint_set_sha256"),
        ("model_contracts", "model_contracts_sha256"),
        ("study_design", "study_design_sha256"),
        ("analysis_plan", "analysis_plan_sha256"),
    )
    for value_name, hash_name in hashed_fields:
        if _json_sha256(manifest.get(value_name)) != manifest.get(hash_name):
            raise ValueError(f"freeze manifest {value_name} hash is invalid")
    inventory = manifest.get("source_inventory")
    if not isinstance(inventory, dict) or set(inventory) != {
        "python_sources",
        "requirements",
        "inventory_sha256",
    }:
        raise ValueError("freeze manifest has no source inventory")
    inventory_payload = dict(inventory)
    inventory_hash = inventory_payload.pop("inventory_sha256", None)
    if (
        not _is_sha256(inventory_hash)
        or _json_sha256(inventory_payload) != inventory_hash
    ):
        raise ValueError("freeze manifest source inventory hash is invalid")

    design = manifest.get("study_design")
    if not isinstance(design, dict):
        raise ValueError("freeze manifest study design is invalid")
    architecture, training_config = _normalize_model_recipes(
        manifest.get("architecture"),
        manifest.get("training_config"),
        design,
    )
    checkpoints = manifest.get("pretrained_checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("freeze manifest checkpoint inventory is invalid")
    expected_contracts = _derive_model_contracts(
        architecture,
        training_config,
        design,
        checkpoints,
    )
    if manifest.get("model_contracts") != expected_contracts:
        raise ValueError("freeze manifest model contracts differ from frozen recipes")


def _validate_study_design(design: Any) -> None:
    if not isinstance(design, dict) or set(design) != {
        "models",
        "comparators",
        "seeds",
        "datasets",
    }:
        raise ValueError("freeze manifest study design is invalid")
    models = _unique_strings("models", design.get("models"))
    comparators = _unique_strings("comparators", design.get("comparators"))
    if set(models) & set(comparators):
        raise ValueError("freeze manifest model roles overlap")
    _unique_integers(
        "seeds",
        design.get("seeds"),
        minimum=MINIMUM_CONFIRMATION_SEEDS,
    )
    datasets = design.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("freeze manifest has no datasets")
    for dataset, plan in datasets.items():
        spec = dataset_spec(dataset)
        if not isinstance(plan, dict) or set(plan) != {
            "dataset_spec",
            "subjects",
            "folds",
            "preprocessing_profiles",
        }:
            raise ValueError(f"freeze manifest {dataset} plan is invalid")
        expected_spec = _json_clone(asdict(spec))
        if plan.get("dataset_spec") != expected_spec:
            raise ValueError(f"freeze manifest {dataset} dataset contract is stale")
        subjects = _unique_integers(f"{dataset}.subjects", plan.get("subjects"))
        if set(subjects) != set(spec.confirmation_subjects):
            raise PermissionError(
                f"freeze manifest {dataset} confirmation cohort is stale"
            )
        folds = _unique_integers(f"{dataset}.folds", plan.get("folds"))
        if set(folds) != set(registered_confirmation_folds(dataset)):
            raise PermissionError(f"freeze manifest {dataset} fold contract is stale")
        profiles = plan.get("preprocessing_profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError(f"freeze manifest {dataset} has no preprocessing profiles")
        for name, profile in profiles.items():
            if not isinstance(profile, dict):
                raise ValueError(f"freeze manifest {dataset}/{name} profile is invalid")
            channels = profile.get("ordered_channels")
            if not isinstance(channels, list):
                raise ValueError(f"freeze manifest {dataset}/{name} has no channels")
            current = current_preprocessing_profile(
                dataset,
                name,
                channel_names=channels,
            )
            if profile != current:
                raise ValueError(
                    f"freeze manifest {dataset}/{name} preprocessing profile is stale"
                )


def validate_freeze_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Validate manifest integrity and every currently referenced byte."""

    manifest_path = Path(path)
    manifest = _read_json_object(manifest_path)
    _validate_manifest_structure(manifest)
    _validate_canonical_json_file(manifest_path, manifest)
    root = Path(project_root).resolve()
    current_inventory = _source_entries(root)
    if manifest.get("source_inventory") != current_inventory:
        raise ValueError(
            "current IEEE-MI source/requirements bytes differ from the freeze"
        )

    checkpoints = manifest.get("pretrained_checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("freeze manifest checkpoint inventory is invalid")
    for name, entry in checkpoints.items():
        if not isinstance(entry, dict) or set(entry) != {
            "path_kind",
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError(f"freeze checkpoint {name} entry is invalid")
        path_value = _resolve_checkpoint_entry(entry, root)
        if path_value.is_symlink() or not path_value.is_file():
            raise FileNotFoundError(f"frozen checkpoint is unavailable: {path_value}")
        if path_value.stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"frozen checkpoint {name} size changed")
        if (
            not _is_sha256(entry.get("sha256"))
            or _file_sha256(path_value) != entry["sha256"]
        ):
            raise ValueError(f"frozen checkpoint {name} bytes changed")

    _validate_study_design(manifest.get("study_design"))
    analysis_plan = manifest.get("analysis_plan")
    if not isinstance(analysis_plan, dict) or set(analysis_plan) != {
        "primary_hypothesis",
        "primary_statistic",
        "success_rule",
    }:
        raise ValueError("freeze manifest analysis plan is invalid")
    if not str(analysis_plan.get("primary_hypothesis", "")).strip():
        raise ValueError("freeze manifest primary hypothesis is invalid")
    statistic = _normalize_primary_statistic(
        analysis_plan.get("primary_statistic"),
        manifest["study_design"],
    )
    if statistic != analysis_plan.get("primary_statistic"):
        raise ValueError("freeze manifest primary statistic is not canonical")
    _validate_success_rule(analysis_plan.get("success_rule"))
    return manifest


def manifest_token(manifest_or_path: Mapping[str, Any] | str | Path) -> str:
    """Return the immutable token after validating the artifact's own bytes."""

    if isinstance(manifest_or_path, Mapping):
        manifest = dict(manifest_or_path)
    else:
        manifest_path = Path(manifest_or_path)
        manifest = _read_json_object(manifest_path)
    _validate_manifest_structure(manifest)
    if not isinstance(manifest_or_path, Mapping):
        _validate_canonical_json_file(manifest_path, manifest)
    return str(manifest["manifest_sha256"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create", help="create an immutable freeze")
    create_parser.add_argument("--spec", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    create_parser.add_argument("--project-root", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate", help="validate a freeze")
    validate_parser.add_argument("manifest", type=Path)
    validate_parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "create":
        specification = _read_json_object(args.spec)
        manifest = create_freeze_manifest(
            specification,
            args.output,
            project_root=args.project_root,
        )
    else:
        manifest = validate_freeze_manifest(
            args.manifest,
            project_root=args.project_root,
        )
    print(manifest["manifest_sha256"])


if __name__ == "__main__":
    main()
