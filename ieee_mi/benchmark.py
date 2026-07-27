"""Development benchmark CLI for proposed and rerun neural models."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import make_model
from .config import CHANNEL_SCALING, dataset_spec, preprocessing_for_dataset
from .data import (
    apply_channel_scaler,
    build_subject_cache,
    fit_channel_scaler,
    load_subject_cache,
    reflection_index,
    split_indices,
)
from .models import CardinalFieldConfig, ScopeConfig, parameter_count
from .training import (
    TrainConfig,
    classification_metrics,
    configure_determinism,
    fit_model,
    predict_probabilities,
    refit_model,
)


DEVELOPMENT_ARTIFACT_SCHEMA = "ieee-mi-development-v4"
LEGACY_SCREEN_ARTIFACT_SCHEMA = "ieee-mi-development-v3"
ARTIFACT_MODES = ("screen", "formal")
SCREEN_EVIDENCE_SCOPE = "exploratory_development_screen_descriptive_only"
FORMAL_EVIDENCE_SCOPE = "prespecified_formal_development_not_confirmation"
SCREEN_ANALYSIS_POLICY = "descriptive_only"
FORMAL_ANALYSIS_POLICY = "prespecified_development_inference_not_confirmation"

# This seed set is part of the formal-development contract rather than a CLI
# convenience.  Changing it creates a new protocol revision.  Exploratory
# screens remain free to use any nonempty set of unique seeds.
FORMAL_DEVELOPMENT_SEEDS: tuple[int, ...] = (7, 17, 27, 37, 47)


def development_folds(dataset: str) -> tuple[int, ...]:
    """Return every registered outer fold for one development dataset."""

    dataset_spec(dataset)
    if dataset == "cho2017":
        return tuple(range(5))
    if dataset == "physionet_mi":
        return tuple(range(3))
    return (0,)


def formal_development_contract(dataset: str) -> dict[str, Any]:
    """Return the complete, frozen dimensions of a formal development run."""

    spec = dataset_spec(dataset)
    if not spec.development_subjects:
        raise ValueError(f"{dataset} has no registered development cohort")
    return {
        "stage": "development",
        "dataset": dataset,
        "artifact_mode": "formal",
        "confirmation_evidence": False,
        "subjects": list(spec.development_subjects),
        "folds": list(development_folds(dataset)),
        "seeds": list(FORMAL_DEVELOPMENT_SEEDS),
        "seed_policy": "exact_frozen_set",
    }


def validate_development_design(
    dataset: str,
    *,
    artifact_mode: str,
    subjects: tuple[int, ...],
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    """Reject incomplete dimensions only when a run claims formal status."""

    if artifact_mode not in ARTIFACT_MODES:
        raise ValueError(f"artifact_mode must be one of {ARTIFACT_MODES}")
    if artifact_mode == "screen":
        return
    contract = formal_development_contract(dataset)
    dimensions = {
        "subjects": (set(subjects), set(contract["subjects"])),
        "folds": (set(folds), set(contract["folds"])),
        "seeds": (set(seeds), set(contract["seeds"])),
    }
    for name, (observed, expected) in dimensions.items():
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"formal development {name} differ from the frozen contract; "
                f"missing={missing}, extra={extra}"
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    """Hash every persistent tensor in a model checkpoint deterministically."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    import importlib.metadata

    import braindecode
    import mne
    import moabb
    import sklearn

    distributions = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    distribution_text = "\n".join(distributions).encode("utf-8")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "braindecode": braindecode.__version__,
        "mne": mne.__version__,
        "moabb": moabb.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "installed_distributions": distributions,
        "installed_distributions_sha256": hashlib.sha256(
            distribution_text
        ).hexdigest(),
    }


def _architecture_identity(model: torch.nn.Module, requested_name: str) -> dict[str, Any]:
    """Serialize the effective model identity without relying on ``repr``."""

    target = getattr(model, "base", model)
    result: dict[str, Any] = {
        "requested_name": requested_name,
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "wrapped_class": f"{type(target).__module__}.{type(target).__qualname__}",
    }
    config = getattr(model, "config", None)
    if config is not None:
        if is_dataclass(config):
            result["config"] = asdict(config)
        elif isinstance(config, dict):
            result["config"] = dict(config)
    scalar_attributes: dict[str, Any] = {}
    for name, value in vars(target).items():
        if name.startswith("_"):
            continue
        if isinstance(value, (str, bool, int, float)) or value is None:
            scalar_attributes[name] = value
        elif isinstance(value, tuple) and all(
            isinstance(item, (str, bool, int, float)) for item in value
        ):
            scalar_attributes[name] = list(value)
    result["scalar_attributes"] = scalar_attributes
    provenance = getattr(type(target), "_ieee_provenance", None)
    if provenance is not None:
        result["third_party_provenance"] = provenance
    return result


def _scope_config(variant: str) -> ScopeConfig:
    if variant in {"default", "soft", "aug"}:
        return ScopeConfig()
    if variant == "fir":
        return ScopeConfig(adaptive_fir=True)
    if variant == "fir_soft":
        return ScopeConfig(adaptive_fir=True)
    if variant == "compact":
        return ScopeConfig(
            n_filters=12,
            n_sources=6,
            width=32,
            mixer_blocks=1,
            dropout=0.2,
        )
    if variant == "wide":
        return ScopeConfig(
            n_filters=20,
            n_sources=10,
            width=64,
            mixer_blocks=2,
            dropout=0.3,
        )
    raise ValueError(f"unknown SCOPE variant {variant!r}")


def _cardinal_config(variant: str) -> CardinalFieldConfig:
    if variant in {"default", "soft", "aug"}:
        return CardinalFieldConfig()
    if variant == "9x32":
        return CardinalFieldConfig(n_filters=9, n_sources=32)
    if variant == "16x24":
        return CardinalFieldConfig(n_filters=16, n_sources=24)
    if variant == "16x48":
        return CardinalFieldConfig(n_filters=16, n_sources=48)
    if variant == "seg8":
        return CardinalFieldConfig(segments=8)
    if variant == "drop":
        return CardinalFieldConfig(dropout=0.25)
    if variant == "fir":
        return CardinalFieldConfig(adaptive_fir=True)
    if variant == "fir_soft":
        return CardinalFieldConfig(adaptive_fir=True)
    if variant == "joint":
        return CardinalFieldConfig(frequency_cardinal=True)
    if variant == "joint_drop":
        return CardinalFieldConfig(frequency_cardinal=True, dropout=0.25)
    if variant == "contrast":
        return CardinalFieldConfig(contrast_bins=8, dropout=0.25)
    if variant == "joint_contrast":
        return CardinalFieldConfig(
            frequency_cardinal=True,
            contrast_bins=8,
            dropout=0.25,
        )
    if variant == "9x32_joint":
        return CardinalFieldConfig(
            n_filters=9,
            n_sources=32,
            frequency_cardinal=True,
            dropout=0.25,
        )
    if variant == "extended_drop":
        return CardinalFieldConfig(extended_atlas=True, dropout=0.25)
    if variant == "residual_drop":
        return CardinalFieldConfig(spectral_residual=True, dropout=0.25)
    if variant == "extended_residual_drop":
        return CardinalFieldConfig(
            extended_atlas=True,
            spectral_residual=True,
            dropout=0.25,
        )
    if variant == "windows_drop":
        return CardinalFieldConfig(learned_windows=True, dropout=0.25)
    if variant == "residual_windows_drop":
        return CardinalFieldConfig(
            spectral_residual=True,
            learned_windows=True,
            dropout=0.25,
        )
    if variant == "extended_windows_drop":
        return CardinalFieldConfig(
            extended_atlas=True,
            learned_windows=True,
            dropout=0.25,
        )
    raise ValueError(f"unknown cardinal-field variant {variant!r}")


def _model_definition(name: str) -> tuple[str, str]:
    key = name.lower().replace("-", "_")
    if key in {
        "cardinal_mix",
        "cardinal_mix_drop",
        "cardinal_fbms",
        "cardinal_fbms_extended",
        "cardinal_fbc",
        "cardinal_fbc_extended",
        "cardinal_fbc_corr",
        "cardinal_fbc_corr_extended",
        "cardinal_fbc_compactdyn",
        "cardinal_fbc_compactdyn_extended",
        "cardinal_fbc_compactdyn_scale010",
        "cardinal_fbc_compactdyn_scale010_extended",
        "cardinal_fbc_compactdyn_scale025",
        "cardinal_fbc_compactdyn_scale025_extended",
        "cardinal_fbc_micro",
        "cardinal_fbc_micro_extended",
        "cardinal_fbc_physical",
        "cardinal_fbc_physical_extended",
    }:
        return key, "baseline"
    if key in {
        "cardinal_dynamics",
        "cardinal_dynamics_compact",
        "cardinal_dynamics_extended",
        "cardinal_dynamics_sinc",
        "cardinal_dynamics_sinc_residual",
        "cardinal_dynamics_sinc_extended",
    }:
        return key, "baseline"
    if key.startswith("free_cardinal"):
        suffix = key.removeprefix("free_cardinal").removeprefix("_") or "default"
        return "free_cardinal", suffix
    if key.startswith("cardinal_"):
        return "cardinal", key.removeprefix("cardinal_")
    if key == "cardinal":
        return "cardinal", "default"
    if key.startswith("scope_"):
        return "scope", key.removeprefix("scope_")
    if key == "scope":
        return "scope", "default"
    if key.startswith("free_scope"):
        suffix = key.removeprefix("free_scope").removeprefix("_") or "default"
        return "free_scope", suffix
    return key, "baseline"


def _train_config_for_model(base: TrainConfig, requested_name: str) -> TrainConfig:
    _, variant = _model_definition(requested_name)
    if variant in {"soft", "fir_soft"}:
        return replace(base, reflection_weight=0.05, reflection_probability=0.25)
    if variant == "aug":
        return replace(base, reflection_weight=0.0, reflection_probability=0.5)
    return base


def run_one(
    *,
    dataset: str,
    subject: int,
    fold: int,
    requested_model: str,
    seed: int,
    cache_root: Path,
    train_config: TrainConfig,
) -> dict[str, Any]:
    spec = dataset_spec(dataset)
    preprocessing = preprocessing_for_dataset(dataset)
    data = load_subject_cache(dataset, subject, cache_root=cache_root)
    train_rows, validation_rows, test_rows = split_indices(
        dataset,
        data["y"],
        data["sessions"],
        data["runs"],
        fold=fold,
        subject=subject,
    )
    channel_names = tuple(str(value) for value in data["channel_names"].tolist())
    selection_mean, selection_std = fit_channel_scaler(
        data["x"][train_rows], channel_names
    )
    x_train = apply_channel_scaler(
        data["x"][train_rows], selection_mean, selection_std
    )
    x_validation = apply_channel_scaler(
        data["x"][validation_rows], selection_mean, selection_std
    )

    # Phase B is still source-only: after the epoch count is selected on the
    # validation partition, scaler statistics may use the complete source
    # partition.  The prediction-only test rows remain untouched.
    source_rows = np.sort(np.concatenate((train_rows, validation_rows)))
    refit_mean, refit_std = fit_channel_scaler(
        data["x"][source_rows], channel_names
    )
    x_source = apply_channel_scaler(data["x"][source_rows], refit_mean, refit_std)
    x_test = apply_channel_scaler(data["x"][test_rows], refit_mean, refit_std)
    model_name, variant = _model_definition(requested_model)
    scope_config = (
        _scope_config(variant)
        if model_name in {"scope", "free_scope"}
        else ScopeConfig()
    )
    cardinal_config = (
        _cardinal_config(variant)
        if model_name in {"cardinal", "free_cardinal"}
        else CardinalFieldConfig()
    )
    # Model parameters are initialized in make_model, so the requested seed
    # must be installed before construction. Seeding only inside fit_model
    # makes initialization depend on which model happened to run previously.
    configure_determinism(seed)
    model = make_model(
        model_name,
        n_channels=x_train.shape[1],
        n_outputs=spec.n_classes,
        n_times=x_train.shape[2],
        sfreq=float(preprocessing["sfreq_hz"]),
        scope_config=scope_config,
        cardinal_config=cardinal_config,
        channel_names=channel_names,
        channel_positions=torch.as_tensor(data["positions"], dtype=torch.float32),
    )
    initial_state = copy.deepcopy(model.state_dict())
    current_train_config = replace(
        _train_config_for_model(train_config, requested_model), seed=seed
    )
    mirror = reflection_index(channel_names)
    fit = fit_model(
        model,
        x_train,
        data["y"][train_rows],
        x_validation,
        data["y"][validation_rows],
        data["positions"],
        config=current_train_config,
        mirror_index=mirror,
    )
    validation_probability = predict_probabilities(
        fit["model"],
        x_validation,
        data["positions"],
        device=current_train_config.device,
    )
    selection_state_sha256 = _model_state_sha256(fit["model"])
    # Discard the selected checkpoint, restore the seeded initialization, and
    # use the validation-selected duration to refit on every source row.
    fit["model"].load_state_dict(initial_state)
    refit = refit_model(
        fit["model"],
        x_source,
        data["y"][source_rows],
        data["positions"],
        epochs=int(fit["best_epoch"]) + 1,
        config=current_train_config,
        mirror_index=mirror,
    )
    test_probability = predict_probabilities(
        refit["model"],
        x_test,
        data["positions"],
        device=current_train_config.device,
    )
    frequencies = None
    if hasattr(refit["model"], "filter_bank") and hasattr(
        refit["model"].filter_bank, "frequencies_hz"
    ):
        frequencies = (
            refit["model"].filter_bank.frequencies_hz().detach().cpu().tolist()
        )
    return {
        "dataset": dataset,
        "subject": subject,
        "fold": fold,
        "model": requested_model,
        "seed": seed,
        "n_classes": spec.n_classes,
        "channels": list(channel_names),
        "split": {
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "refit_source_count": len(source_rows),
            "test_count": len(test_rows),
            "train_runs": sorted(set(data["runs"][train_rows].tolist())),
            "validation_runs": sorted(
                set(data["runs"][validation_rows].tolist())
            ),
            "test_runs": sorted(set(data["runs"][test_rows].tolist())),
            "train_rows_sha256": hashlib.sha256(train_rows.tobytes()).hexdigest(),
            "validation_rows_sha256": hashlib.sha256(validation_rows.tobytes()).hexdigest(),
            "test_rows_sha256": hashlib.sha256(test_rows.tobytes()).hexdigest(),
        },
        "scaler": {
            "selection_train": {
                "mean": selection_mean.reshape(-1).tolist(),
                "std": selection_std.reshape(-1).tolist(),
            },
            "refit_source": {
                "mean": refit_mean.reshape(-1).tolist(),
                "std": refit_std.reshape(-1).tolist(),
            },
        },
        "fit": {
            "best_epoch": fit["best_epoch"],
            "selection_epochs_run": fit["epochs_run"],
            "best_validation_loss": fit["best_validation_loss"],
            "selection_fit_seconds": fit["fit_seconds"],
            "selection_history": fit["history"],
            "selection_state_sha256": selection_state_sha256,
            "refit_epochs_run": refit["epochs_run"],
            "refit_fit_seconds": refit["fit_seconds"],
            "refit_history": refit["history"],
            "refit_state_sha256": _model_state_sha256(refit["model"]),
            "scheduler_horizon_epochs": current_train_config.epochs,
            "parameter_count": parameter_count(refit["model"]),
            "architecture": _architecture_identity(
                refit["model"], requested_model
            ),
            "train_config": fit["config"],
            "scope_config": (
                asdict(scope_config)
                if model_name in {"scope", "free_scope"}
                else None
            ),
            "cardinal_config": (
                asdict(cardinal_config)
                if model_name in {"cardinal", "free_cardinal"}
                else None
            ),
            "learned_frequencies_hz": frequencies,
        },
        "validation": classification_metrics(
            data["y"][validation_rows], validation_probability
        ),
        "test": {
            "metrics": classification_metrics(data["y"][test_rows], test_probability),
            "rows": test_rows.tolist(),
            "labels": data["y"][test_rows].tolist(),
            "probabilities": test_probability.tolist(),
            "sessions": data["sessions"][test_rows].tolist(),
            "runs": data["runs"][test_rows].tolist(),
        },
        "cache_identity": data["identity"],
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty benchmark")
    by_model: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    for record in records:
        model = str(record["model"])
        key = (int(record["subject"]), int(record["fold"]), int(record["seed"]))
        model_records = by_model.setdefault(model, {})
        if key in model_records:
            raise RuntimeError(f"duplicate benchmark key for {model}: {key}")
        model_records[key] = record
    reference_keys = next(iter(by_model.values())).keys()
    for model, model_records in by_model.items():
        if model_records.keys() != reference_keys:
            missing = sorted(set(reference_keys) - set(model_records))
            extra = sorted(set(model_records) - set(reference_keys))
            raise RuntimeError(
                f"incomplete Cartesian benchmark for {model}; "
                f"missing={missing}, extra={extra}"
            )

    result: dict[str, Any] = {}
    for model, model_records in by_model.items():
        subject_seed: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for (subject, _, seed), record in model_records.items():
            subject_seed.setdefault((subject, seed), []).append(record)
        per_subject_seed: dict[tuple[int, int], float] = {}
        for key, fold_records in subject_seed.items():
            seen_rows: set[int] = set()
            labels: list[int] = []
            probabilities: list[list[float]] = []
            for record in sorted(fold_records, key=lambda value: int(value["fold"])):
                rows = [int(value) for value in record["test"]["rows"]]
                overlap = seen_rows.intersection(rows)
                if overlap:
                    raise RuntimeError(
                        f"outer test folds overlap for {model}, subject/seed {key}"
                    )
                seen_rows.update(rows)
                labels.extend(int(value) for value in record["test"]["labels"])
                probabilities.extend(record["test"]["probabilities"])
            metrics = classification_metrics(
                np.asarray(labels, dtype=np.int64),
                np.asarray(probabilities, dtype=np.float64),
            )
            per_subject_seed[key] = float(metrics["balanced_accuracy"])

        subject_values: dict[int, list[float]] = {}
        for (subject, _), value in per_subject_seed.items():
            subject_values.setdefault(subject, []).append(value)
        seed_averaged = {
            subject: float(np.mean(values))
            for subject, values in sorted(subject_values.items())
        }
        values = np.asarray(list(seed_averaged.values()), dtype=np.float64)
        result[model] = {
            "balanced_accuracy_mean": float(values.mean()),
            "balanced_accuracy_std": (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ),
            "n_subjects": len(values),
            "n_records": len(model_records),
            "subject_balanced_accuracy": {
                str(subject): value for subject, value in seed_averaged.items()
            },
        }
    return result


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(token) for token in value.split(",") if token.strip())


def _require_unique(name: str, values: tuple[Any, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--models", default="scope,free_scope,shallow,eegnet")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--folds", default="0")
    parser.add_argument(
        "--artifact-mode",
        choices=ARTIFACT_MODES,
        default="screen",
        help=(
            "screen permits partial exploratory dimensions and is descriptive-only; "
            "formal requires the complete frozen development contract"
        ),
    )
    parser.add_argument("--cache-root", default="data_cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--build-cache", action="store_true")
    args = parser.parse_args(argv)

    spec = dataset_spec(args.dataset)
    subjects = _parse_ints(args.subjects)
    _require_unique("subjects", subjects)
    if not subjects or any(subject not in spec.development_subjects for subject in subjects):
        raise PermissionError("development CLI may access only registered development subjects")
    models = tuple(token.strip() for token in args.models.split(",") if token.strip())
    if not models:
        raise ValueError("at least one model is required")
    normalized_models = tuple(token.lower().replace("-", "_") for token in models)
    _require_unique("models", normalized_models)
    seeds = _parse_ints(args.seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    _require_unique("seeds", seeds)
    folds = _parse_ints(args.folds)
    _require_unique("folds", folds)
    allowed_folds = set(development_folds(args.dataset))
    if not folds or any(fold not in allowed_folds for fold in folds):
        choices = ",".join(str(fold) for fold in sorted(allowed_folds))
        raise ValueError(f"{args.dataset} folds must be selected from {choices}")
    validate_development_design(
        args.dataset,
        artifact_mode=args.artifact_mode,
        subjects=subjects,
        folds=folds,
        seeds=seeds,
    )
    cache_root = Path(args.cache_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.build_cache:
        for subject in subjects:
            build_subject_cache(
                args.dataset,
                subject,
                cache_root=cache_root,
                stage="development",
            )

    train_config = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
    )
    records: list[dict[str, Any]] = []
    for subject in subjects:
        for fold in folds:
            for requested_model in models:
                for seed in seeds:
                    print(
                        f"[{_utc_now()}] {args.dataset} S{subject} fold={fold} "
                        f"model={requested_model} seed={seed}",
                        flush=True,
                    )
                    record = run_one(
                        dataset=args.dataset,
                        subject=subject,
                        fold=fold,
                        requested_model=requested_model,
                        seed=seed,
                        cache_root=cache_root,
                        train_config=train_config,
                    )
                    records.append(record)
                    seconds = (
                        record["fit"]["selection_fit_seconds"]
                        + record["fit"]["refit_fit_seconds"]
                    )
                    print(
                        f"  BA={record['test']['metrics']['balanced_accuracy']:.4f} "
                        f"best_epoch={record['fit']['best_epoch']} "
                        f"seconds={seconds:.1f}",
                        flush=True,
                    )
    artifact = {
        "schema": DEVELOPMENT_ARTIFACT_SCHEMA,
        "created_at": _utc_now(),
        "mode": "development",
        "artifact_mode": args.artifact_mode,
        "evidence_scope": (
            SCREEN_EVIDENCE_SCOPE
            if args.artifact_mode == "screen"
            else FORMAL_EVIDENCE_SCOPE
        ),
        "confirmation_evidence": False,
        "analysis_policy": (
            SCREEN_ANALYSIS_POLICY
            if args.artifact_mode == "screen"
            else FORMAL_ANALYSIS_POLICY
        ),
        "formal_protocol": (
            formal_development_contract(args.dataset)
            if args.artifact_mode == "formal"
            else None
        ),
        "dataset": args.dataset,
        "dataset_spec": asdict(spec),
        "subjects": list(subjects),
        "models": list(models),
        "seeds": list(seeds),
        "folds": list(folds),
        "preprocessing": preprocessing_for_dataset(args.dataset),
        "channel_scaling": CHANNEL_SCALING,
        "train_config": asdict(train_config),
        "environment": _environment(),
        "source_sha256": {
            str(path): _sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("analysis.py"),
                Path(__file__).with_name("models.py"),
                Path(__file__).with_name("training.py"),
                Path(__file__).with_name("data.py"),
                Path(__file__).with_name("config.py"),
                Path(__file__).with_name("baselines.py"),
                Path(__file__).with_name("requirements-cu128.txt"),
            )
        },
        "records": records,
        "summary": _summary(records),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
