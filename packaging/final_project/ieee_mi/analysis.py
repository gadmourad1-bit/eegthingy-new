"""Protocol-aware aggregation for exploratory and formal development artifacts.

Exploratory screens are intentionally descriptive-only.  Formal artifacts may
receive prespecified inferential summaries, but remain development evidence and
must never be described as confirmation evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from .benchmark import (
    DEVELOPMENT_ARTIFACT_SCHEMA,
    FORMAL_ANALYSIS_POLICY,
    FORMAL_EVIDENCE_SCOPE,
    LEGACY_SCREEN_ARTIFACT_SCHEMA,
    SCREEN_ANALYSIS_POLICY,
    SCREEN_EVIDENCE_SCOPE,
    _summary,
    formal_development_contract,
    validate_development_design,
)
from .config import CHANNEL_SCALING, dataset_spec, preprocessing_for_dataset


@dataclass(frozen=True)
class LoadedArtifacts:
    records_by_dataset: dict[str, list[dict[str, Any]]]
    artifact_mode: str
    contains_legacy_screen: bool


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _percentile_interval(samples: np.ndarray) -> list[float]:
    low, high = np.quantile(samples, (0.025, 0.975))
    return [float(low), float(high)]


def _bootstrap_mean(
    values: np.ndarray,
    *,
    generator: np.random.Generator,
    repetitions: int,
) -> np.ndarray:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap values must be a nonempty vector")
    indices = generator.integers(0, len(values), size=(repetitions, len(values)))
    return values[indices].mean(axis=1)


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm family-wise correction with monotonic adjusted p-values."""

    ordered = sorted(p_values, key=p_values.get)  # type: ignore[arg-type]
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _unique_declared(artifact: dict[str, Any], name: str, path: Path) -> tuple[Any, ...]:
    values = artifact.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} has no declared {name}")
    result = tuple(values)
    try:
        unique_count = len(set(result))
    except TypeError as error:
        raise ValueError(f"{path} has non-scalar declared {name}") from error
    if unique_count != len(result):
        raise ValueError(f"{path} declares duplicate {name}")
    return result


def _normalized_record_train_config(record: dict[str, Any], path: Path) -> dict[str, Any]:
    try:
        config = copy.deepcopy(record["fit"]["train_config"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} record has no effective train configuration") from error
    if not isinstance(config, dict):
        raise ValueError(f"{path} record train configuration is not an object")
    if "seed" not in config:
        raise ValueError(f"{path} record train configuration has no seed")
    seed = int(record["seed"])
    if int(config["seed"]) != seed:
        raise ValueError(f"{path} record seed differs from its train configuration")
    config.pop("seed", None)
    return config


def _validate_record(
    record: dict[str, Any],
    *,
    path: Path,
    dataset: str,
    declared_subjects: tuple[int, ...],
    declared_models: tuple[str, ...],
    declared_folds: tuple[int, ...],
    declared_seeds: tuple[int, ...],
    artifact_dataset_spec: dict[str, Any],
    artifact_preprocessing: dict[str, Any],
) -> tuple[str, int, int, int]:
    try:
        record_dataset = str(record["dataset"])
        model = str(record["model"])
        subject = int(record["subject"])
        fold = int(record["fold"])
        seed = int(record["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} contains an invalid record identity") from error
    if record_dataset != dataset:
        raise ValueError(f"{path} record dataset differs from its artifact dataset")
    if subject not in declared_subjects or model not in declared_models:
        raise ValueError(f"{path} record identity is outside its declared dimensions")
    if fold not in declared_folds or seed not in declared_seeds:
        raise ValueError(f"{path} record identity is outside its declared dimensions")

    try:
        split = record["split"]
        test = record["test"]
        rows = [int(value) for value in test["rows"]]
        labels = [int(value) for value in test["labels"]]
        probabilities = np.asarray(test["probabilities"], dtype=np.float64)
        sessions = [str(value) for value in test["sessions"]]
        runs = [str(value) for value in test["runs"]]
        test_count = int(split["test_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} record has an invalid test contract") from error
    if not (
        len(rows)
        == len(labels)
        == len(sessions)
        == len(runs)
        == len(probabilities)
        == test_count
    ):
        raise ValueError(f"{path} record test arrays differ in length")
    if any(row < 0 for row in rows) or len(set(rows)) != test_count:
        raise ValueError(f"{path} record test rows are invalid or duplicated")
    n_classes = int(record.get("n_classes", -1))
    if probabilities.shape != (test_count, n_classes):
        raise ValueError(f"{path} record probabilities have an invalid shape")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"{path} record probabilities are invalid")
    if np.any(probabilities.sum(axis=1) <= 0.0):
        raise ValueError(f"{path} record probability rows have zero mass")
    if any(label < 0 or label >= n_classes for label in labels):
        raise ValueError(f"{path} record labels are outside its class range")
    row_array = np.asarray(rows, dtype=np.int64)
    expected_row_hash = hashlib.sha256(row_array.tobytes()).hexdigest()
    if split.get("test_rows_sha256") != expected_row_hash:
        raise ValueError(f"{path} record test-row hash is invalid")
    for hash_name in ("train_rows_sha256", "validation_rows_sha256"):
        if not _is_sha256(split.get(hash_name)):
            raise ValueError(f"{path} record {hash_name} is invalid")

    cache_identity = record.get("cache_identity")
    if not isinstance(cache_identity, dict):
        raise ValueError(f"{path} record has no cache identity")
    array_sha256 = cache_identity.get("array_sha256")
    if not _is_sha256(array_sha256):
        raise ValueError(f"{path} record cache array hash is invalid")
    if int(cache_identity.get("subject", -1)) != subject:
        raise ValueError(f"{path} record cache subject is invalid")
    cache_dataset = cache_identity.get("dataset", {})
    if cache_dataset != artifact_dataset_spec:
        raise ValueError(f"{path} record cache dataset is invalid")
    if cache_identity.get("preprocessing") != artifact_preprocessing:
        raise ValueError(f"{path} record cache preprocessing is invalid")
    if cache_identity.get("channels") != record.get("channels"):
        raise ValueError(f"{path} record channels differ from its cache identity")
    channels = record.get("channels")
    scaler = record.get("scaler")
    if not isinstance(channels, list) or not channels or not isinstance(scaler, dict):
        raise ValueError(f"{path} record has an invalid channel/scaler contract")
    for scaler_name in ("selection_train", "refit_source"):
        current_scaler = scaler.get(scaler_name)
        if not isinstance(current_scaler, dict):
            raise ValueError(f"{path} record has no {scaler_name} scaler")
        try:
            means = np.asarray(current_scaler["mean"], dtype=np.float64)
            standard_deviations = np.asarray(current_scaler["std"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path} record has an invalid {scaler_name} scaler") from error
        if means.shape != (len(channels),) or standard_deviations.shape != (
            len(channels),
        ):
            raise ValueError(f"{path} record {scaler_name} scaler has an invalid shape")
        if not np.all(np.isfinite(means)) or not np.all(
            np.isfinite(standard_deviations)
        ):
            raise ValueError(f"{path} record {scaler_name} scaler is non-finite")
        if np.any(standard_deviations <= 0.0):
            raise ValueError(f"{path} record {scaler_name} scaler has nonpositive scale")
    _normalized_record_train_config(record, path)
    return model, subject, fold, seed


def _record_data_contract(record: dict[str, Any]) -> dict[str, Any]:
    """Return every paired, non-model-specific data-selection field."""

    test = record["test"]
    return {
        "n_classes": record["n_classes"],
        "channels": record["channels"],
        "cache_identity": record["cache_identity"],
        "split": record["split"],
        "scaler": record["scaler"],
        "test_rows": test["rows"],
        "test_labels": test["labels"],
        "test_sessions": test["sessions"],
        "test_runs": test["runs"],
    }


def _load_artifacts(paths: list[Path]) -> LoadedArtifacts:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    code_identity: dict[str, str] | None = None
    environment_identity: dict[str, Any] | None = None
    channel_scaling_identity: dict[str, Any] | None = None
    train_config_identity: dict[str, Any] | None = None
    dataset_metadata: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    record_train_configs: dict[str, dict[str, Any]] = {}
    subject_cache_identities: dict[tuple[str, int], dict[str, Any]] = {}
    paired_data: dict[tuple[str, int, int], dict[str, Any]] = {}
    seen: set[tuple[str, str, int, int, int]] = set()
    observed_modes: set[str] = set()
    contains_legacy_screen = False

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        schema = artifact.get("schema")
        if schema == LEGACY_SCREEN_ARTIFACT_SCHEMA:
            artifact_mode = "screen"
            contains_legacy_screen = True
        elif schema == DEVELOPMENT_ARTIFACT_SCHEMA:
            artifact_mode = str(artifact.get("artifact_mode", ""))
            if artifact_mode not in {"screen", "formal"}:
                raise ValueError(f"{path} has no valid artifact mode")
            if artifact.get("confirmation_evidence") is not False:
                raise ValueError(f"{path} incorrectly claims confirmation evidence")
            expected_policy = (
                SCREEN_ANALYSIS_POLICY
                if artifact_mode == "screen"
                else FORMAL_ANALYSIS_POLICY
            )
            if artifact.get("analysis_policy") != expected_policy:
                raise ValueError(f"{path} has an invalid analysis policy")
            expected_scope = (
                SCREEN_EVIDENCE_SCOPE
                if artifact_mode == "screen"
                else FORMAL_EVIDENCE_SCOPE
            )
            if artifact.get("evidence_scope") != expected_scope:
                raise ValueError(f"{path} has an invalid evidence scope")
        else:
            raise ValueError(f"{path} is not a supported development artifact")
        if artifact.get("mode") != "development":
            raise ValueError(f"{path} is not a development artifact")
        observed_modes.add(artifact_mode)
        if len(observed_modes) > 1:
            raise ValueError("screen and formal artifacts must not be mixed")

        dataset = str(artifact.get("dataset", ""))
        try:
            spec = dataset_spec(dataset)
        except ValueError as error:
            raise ValueError(f"{path} has an unknown dataset") from error
        declared_subjects = tuple(
            int(value) for value in _unique_declared(artifact, "subjects", path)
        )
        declared_models = tuple(
            str(value) for value in _unique_declared(artifact, "models", path)
        )
        declared_folds = tuple(
            int(value) for value in _unique_declared(artifact, "folds", path)
        )
        declared_seeds = tuple(
            int(value) for value in _unique_declared(artifact, "seeds", path)
        )
        if any(subject not in spec.development_subjects for subject in declared_subjects):
            raise ValueError(f"{path} declares a non-development subject")
        validate_development_design(
            dataset,
            artifact_mode=artifact_mode,
            subjects=declared_subjects,
            folds=declared_folds,
            seeds=declared_seeds,
        )
        if artifact_mode == "formal":
            if artifact.get("formal_protocol") != formal_development_contract(dataset):
                raise ValueError(f"{path} formal protocol differs from the frozen contract")
        elif schema == DEVELOPMENT_ARTIFACT_SCHEMA and artifact.get("formal_protocol") is not None:
            raise ValueError(f"{path} screen artifact must not claim a formal protocol")

        current_dataset_spec = artifact.get("dataset_spec")
        current_preprocessing = artifact.get("preprocessing")
        if not isinstance(current_dataset_spec, dict) or not isinstance(
            current_preprocessing, dict
        ):
            raise ValueError(f"{path} has invalid dataset metadata")
        if schema == DEVELOPMENT_ARTIFACT_SCHEMA:
            # JSON turns DatasetSpec tuple fields into arrays. Compare against
            # the same wire representation that benchmark artifacts contain.
            expected_dataset_spec = json.loads(json.dumps(asdict(spec)))
            if current_dataset_spec != expected_dataset_spec:
                raise ValueError(f"{path} dataset specification is stale")
            if current_preprocessing != preprocessing_for_dataset(dataset):
                raise ValueError(f"{path} preprocessing contract is stale")
        metadata = (current_dataset_spec, current_preprocessing)
        previous_metadata = dataset_metadata.setdefault(dataset, metadata)
        if metadata != previous_metadata:
            raise ValueError(f"{dataset} artifacts use different dataset/preprocessing contracts")

        scaling = artifact.get("channel_scaling")
        if not isinstance(scaling, dict):
            raise ValueError(f"{path} has no channel-scaling contract")
        if schema == DEVELOPMENT_ARTIFACT_SCHEMA and scaling != CHANNEL_SCALING:
            raise ValueError(f"{path} channel-scaling contract is stale")
        if channel_scaling_identity is None:
            channel_scaling_identity = scaling
        elif scaling != channel_scaling_identity:
            raise ValueError("artifacts use different channel-scaling contracts")

        top_train_config = artifact.get("train_config")
        environment = artifact.get("environment")
        current_identity = artifact.get("source_sha256")
        if not isinstance(top_train_config, dict) or not top_train_config:
            raise ValueError(f"{path} has no base train configuration")
        if not isinstance(environment, dict) or not environment:
            raise ValueError(f"{path} has no environment identity")
        if not isinstance(current_identity, dict):
            raise ValueError(f"{path} has no source-code identity")
        if not current_identity or any(
            not isinstance(name, str) or not name or not _is_sha256(digest)
            for name, digest in current_identity.items()
        ):
            raise ValueError(f"{path} has an invalid source-code identity")
        if train_config_identity is None:
            train_config_identity = top_train_config
        elif top_train_config != train_config_identity:
            raise ValueError("artifacts use different base train configurations")
        if environment_identity is None:
            environment_identity = environment
        elif environment != environment_identity:
            raise ValueError("artifacts were produced in different environments")
        if code_identity is None:
            code_identity = current_identity
        elif current_identity != code_identity:
            raise ValueError("artifacts were produced by different source revisions")

        records = artifact.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path} contains no records")
        artifact_keys: set[tuple[str, int, int, int]] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{path} contains a non-object record")
            model, subject, fold, seed = _validate_record(
                record,
                path=path,
                dataset=dataset,
                declared_subjects=declared_subjects,
                declared_models=declared_models,
                declared_folds=declared_folds,
                declared_seeds=declared_seeds,
                artifact_dataset_spec=current_dataset_spec,
                artifact_preprocessing=current_preprocessing,
            )
            artifact_key = (model, subject, fold, seed)
            if artifact_key in artifact_keys:
                raise ValueError(f"{path} contains a duplicate record: {artifact_key}")
            artifact_keys.add(artifact_key)
            global_key = (dataset, *artifact_key)
            if global_key in seen:
                raise ValueError(f"duplicate record across artifacts: {global_key}")
            seen.add(global_key)

            effective_config = _normalized_record_train_config(record, path)
            previous_config = record_train_configs.setdefault(model, effective_config)
            if effective_config != previous_config:
                raise ValueError(
                    f"{model} records use different effective train configurations"
                )
            data_key = (dataset, subject, fold)
            contract = _record_data_contract(record)
            previous_contract = paired_data.setdefault(data_key, contract)
            if contract != previous_contract:
                raise ValueError(
                    f"paired data contract differs for {dataset} S{subject} fold {fold}"
                )
            cache_key = (dataset, subject)
            current_cache = record["cache_identity"]
            previous_cache = subject_cache_identities.setdefault(cache_key, current_cache)
            if current_cache != previous_cache:
                raise ValueError(
                    f"cache identity differs for {dataset} S{subject} across paired records"
                )
            by_dataset[dataset].append(record)

        expected_artifact_keys = set(
            product(declared_models, declared_subjects, declared_folds, declared_seeds)
        )
        if artifact_keys != expected_artifact_keys:
            missing = sorted(expected_artifact_keys - artifact_keys)
            extra = sorted(artifact_keys - expected_artifact_keys)
            raise ValueError(
                f"{path} record identities are not the declared Cartesian product; "
                f"missing={missing}, extra={extra}"
            )

    if not by_dataset:
        raise ValueError("no benchmark artifacts were supplied")

    # Also require the union of any artifact shards to be Cartesian.  This
    # prevents, for example, S1/seed7 and S2/seed17 from masquerading as a
    # complete two-subject/two-seed experiment.
    for dataset, records in by_dataset.items():
        by_model: dict[str, set[tuple[int, int, int]]] = defaultdict(set)
        for record in records:
            by_model[str(record["model"])].add(
                (int(record["subject"]), int(record["fold"]), int(record["seed"]))
            )
        reference_keys = next(iter(by_model.values()))
        if any(keys != reference_keys for keys in by_model.values()):
            raise ValueError(f"{dataset} does not have paired subject/fold/seed identities")
        subjects = tuple(sorted({key[0] for key in reference_keys}))
        folds = tuple(sorted({key[1] for key in reference_keys}))
        seeds = tuple(sorted({key[2] for key in reference_keys}))
        expected_keys = set(product(subjects, folds, seeds))
        if reference_keys != expected_keys:
            raise ValueError(f"{dataset} subject/fold/seed identities are not Cartesian")
        validate_development_design(
            dataset,
            artifact_mode=next(iter(observed_modes)),
            subjects=subjects,
            folds=folds,
            seeds=seeds,
        )

    return LoadedArtifacts(
        records_by_dataset=dict(by_dataset),
        artifact_mode=next(iter(observed_modes)),
        contains_legacy_screen=contains_legacy_screen,
    )


def analyze(
    paths: list[Path],
    *,
    reference: str,
    bootstrap_repetitions: int = 20_000,
    random_seed: int = 20260719,
) -> dict[str, Any]:
    loaded = _load_artifacts(paths)
    formal = loaded.artifact_mode == "formal"
    if formal and bootstrap_repetitions <= 0:
        raise ValueError("formal analysis requires positive bootstrap repetitions")
    summaries = {
        dataset: _summary(records)
        for dataset, records in loaded.records_by_dataset.items()
    }
    model_sets = [set(summary) for summary in summaries.values()]
    if any(models != model_sets[0] for models in model_sets[1:]):
        raise ValueError("every dataset must contain the same model set")
    models = sorted(model_sets[0])
    if reference not in models:
        raise ValueError(f"reference {reference!r} is absent from the artifacts")

    generator = np.random.default_rng(random_seed)
    dataset_results: dict[str, Any] = {}
    dataset_bootstrap: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    raw_p_values: dict[str, float] = {}
    paired_results: dict[str, Any] = {}
    for dataset, summary in sorted(summaries.items()):
        dataset_results[dataset] = {}
        reference_values = {
            int(subject): float(value)
            for subject, value in summary[reference][
                "subject_balanced_accuracy"
            ].items()
        }
        for model in models:
            subject_values = {
                int(subject): float(value)
                for subject, value in summary[model][
                    "subject_balanced_accuracy"
                ].items()
            }
            if subject_values.keys() != reference_values.keys():
                raise ValueError(f"subject sets differ for {dataset}/{model}")
            values = np.asarray(
                [subject_values[subject] for subject in sorted(subject_values)],
                dtype=np.float64,
            )
            model_result: dict[str, Any] = {
                "mean": float(values.mean()),
                "standard_deviation": (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                ),
                "n_subjects": len(values),
                "subject_balanced_accuracy": {
                    str(subject): subject_values[subject]
                    for subject in sorted(subject_values)
                },
            }
            if formal:
                boot = _bootstrap_mean(
                    values,
                    generator=generator,
                    repetitions=bootstrap_repetitions,
                )
                dataset_bootstrap[dataset][model] = boot
                model_result["bootstrap_95_ci"] = _percentile_interval(boot)
            dataset_results[dataset][model] = model_result
            if model == reference:
                continue
            ordered_subjects = sorted(reference_values)
            paired = np.asarray(
                [
                    subject_values[subject] - reference_values[subject]
                    for subject in ordered_subjects
                ],
                dtype=np.float64,
            )
            comparison_key = f"{dataset}:{model}-vs-{reference}"
            paired_result: dict[str, Any] = {
                "mean_paired_delta": float(paired.mean()),
                "median_paired_delta": float(np.median(paired)),
                "wins_ties_losses": [
                    int(np.sum(paired > 0.0)),
                    int(np.sum(paired == 0.0)),
                    int(np.sum(paired < 0.0)),
                ],
                "n_subjects": len(paired),
                "analysis": (
                    "descriptive_only"
                    if not formal
                    else "prespecified_formal_development_not_confirmation"
                ),
            }
            if formal:
                if np.allclose(paired, 0.0):
                    p_value = 1.0
                else:
                    p_value = float(
                        wilcoxon(
                            paired,
                            alternative="two-sided",
                            zero_method="pratt",
                            method="auto",
                        ).pvalue
                    )
                raw_p_values[comparison_key] = p_value
                paired_result["wilcoxon_p_raw"] = p_value
            paired_results[comparison_key] = paired_result

    if formal:
        adjusted = _holm_adjust(raw_p_values)
        for key, value in adjusted.items():
            paired_results[key]["wilcoxon_p_holm"] = value

    equal_dataset: dict[str, Any] = {}
    for model in models:
        point = float(
            np.mean(
                [dataset_results[dataset][model]["mean"] for dataset in summaries]
            )
        )
        model_result = {
            "mean": point,
            "n_datasets": len(summaries),
        }
        if formal:
            boot = np.stack(
                [dataset_bootstrap[dataset][model] for dataset in sorted(summaries)],
                axis=1,
            ).mean(axis=1)
            model_result["bootstrap_95_ci"] = _percentile_interval(boot)
        equal_dataset[model] = model_result
    return {
        "schema": "ieee-mi-subject-analysis-v2",
        "artifact_mode": loaded.artifact_mode,
        "analysis_policy": (
            "descriptive_screen_no_inferential_statistics"
            if not formal
            else "prespecified_formal_development_inference_not_confirmation"
        ),
        "evidence_scope": "development_only_not_confirmation",
        "confirmation_evidence": False,
        "inferential_statistics": formal,
        "contains_legacy_screen_artifact": loaded.contains_legacy_screen,
        "reference": reference,
        "bootstrap_repetitions": bootstrap_repetitions if formal else 0,
        "random_seed": random_seed if formal else None,
        "datasets": dataset_results,
        "equal_dataset_macro": equal_dataset,
        "paired_comparisons": paired_results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=20260719)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = analyze(
        args.artifacts,
        reference=args.reference,
        bootstrap_repetitions=args.bootstrap_repetitions,
        random_seed=args.random_seed,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
