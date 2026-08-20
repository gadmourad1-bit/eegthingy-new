"""Post-outcome, read-only verification of the CardinalFBMS full-grid gate.

This is deliberately not part of the frozen decision gate.  It was written
after the gate artifact existed to verify that artifact from the raw OOF NumPy
archives without importing or calling the gate or full-grid audit modules.

The verifier enforces the exact grid, rehashes every prediction/provenance
file, reconstructs every subject/seed balanced accuracy after concatenating
disjoint outer folds, reconciles all nested and top-level means, and reproduces
the two prespecified paired-subject bootstraps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


DATASET_SPECS: dict[str, dict[str, tuple[int, ...]]] = {
    "cho2017": {
        "subjects": tuple(range(16, 53)),
        "folds": tuple(range(5)),
    },
    "physionet_mi": {
        "subjects": tuple(range(1, 55)),
        "folds": tuple(range(3)),
    },
}
DATASETS = tuple(DATASET_SPECS)
SEEDS = (7, 17, 27, 37, 47)
CONDITIONS = (
    "pretrained_cardinal_fbms",
    "pretrained_indexed_fbmsnet_spherical_spline",
    "scratch_cardinal_fbms_canonical_seeded",
    "scratch_cardinal_fbms_native_projected",
    "scratch_fbmsnet_native",
)
CANDIDATE = "pretrained_cardinal_fbms"
SCRATCH = "scratch_cardinal_fbms_canonical_seeded"
PRIMARY = "pretrained_indexed_fbmsnet_spherical_spline"
REFERENCE_CONDITIONS = (
    "scratch_cardinal_fbms_native_projected",
    "scratch_fbmsnet_native",
    PRIMARY,
)
HISTORICAL_FLOORS = {
    "cho2017": 0.6708333333,
    "physionet_mi": 0.5911044974,
}
PREDICTION_FIELDS = {
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
BOOTSTRAP_REPETITIONS = 200_000
BOOTSTRAP_BATCH_SIZE = 4096
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
FLOAT_ABS_TOLERANCE = 1e-15


class VerificationError(RuntimeError):
    """The raw artifacts do not reproduce the published gate artifact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise VerificationError(f"{name} must be a mapping")
    return value


def _file_sha256(path: Path) -> str:
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


def _same_float(observed: object, expected: object) -> bool:
    try:
        left = float(observed)
        right = float(expected)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(left)
        and np.isfinite(right)
        and abs(left - right) <= FLOAT_ABS_TOLERANCE
    )


def _require_float(observed: object, expected: object, name: str) -> None:
    if not _same_float(observed, expected):
        raise VerificationError(
            f"{name} differs: observed={observed!r}, expected={expected!r}"
        )


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls: list[float] = []
    for label in (0, 1):
        rows = labels == label
        _require(bool(np.any(rows)), "balanced accuracy unit lacks one binary class")
        recalls.append(float(np.mean(predictions[rows] == label)))
    return float(np.mean(recalls))


def _record_name(subject: int, fold: int, seed: int, condition: str) -> str:
    return f"s{subject:03d}_f{fold}_seed{seed}_{condition}"


def _expected_names(dataset: str) -> set[str]:
    spec = DATASET_SPECS[dataset]
    return {
        _record_name(subject, fold, seed, condition)
        for subject in spec["subjects"]
        for fold in spec["folds"]
        for seed in SEEDS
        for condition in CONDITIONS
    }


def _load_json(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot load {name}: {path}") from error
    return _mapping(value, name)


def _gate_dataset_summary(
    gate: Mapping[str, object], dataset: str
) -> Mapping[str, object]:
    inputs = _mapping(gate.get("inputs"), "gate.inputs")
    summary = _mapping(
        inputs.get("recomputed_full_grid_summary"),
        "gate.inputs.recomputed_full_grid_summary",
    )
    datasets = _mapping(summary.get("datasets"), "gate summary datasets")
    return _mapping(datasets.get(dataset), f"gate summary {dataset}")


def _manifest_records(
    gate: Mapping[str, object], dataset: str
) -> dict[tuple[int, int, int, str], Mapping[str, object]]:
    inputs = _mapping(gate.get("inputs"), "gate.inputs")
    manifest = _mapping(
        inputs.get("freshly_revalidated_full_grid_audit"),
        "gate full-grid manifest",
    )
    datasets = _mapping(manifest.get("datasets"), "gate manifest datasets")
    dataset_manifest = _mapping(datasets.get(dataset), f"manifest {dataset}")
    records = dataset_manifest.get("validated_records")
    _require(isinstance(records, list), f"manifest {dataset} records must be a list")
    result: dict[tuple[int, int, int, str], Mapping[str, object]] = {}
    for value in records:
        record = _mapping(value, f"manifest {dataset} record")
        key = (
            int(record.get("subject", -1)),
            int(record.get("fold", -1)),
            int(record.get("seed", -1)),
            str(record.get("condition", "")),
        )
        _require(key not in result, f"duplicate manifest key {dataset}:{key}")
        result[key] = record
    return result


def _validate_prediction_archive(
    *,
    record_path: Path,
    dataset: str,
    subject: int,
    fold: int,
    seed: int,
    condition: str,
    manifest_record: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require(record_path.is_dir() and not record_path.is_symlink(), f"bad record {record_path}")
    names = {entry.name for entry in record_path.iterdir()}
    _require(
        names == {"predictions.npz", "provenance.json"},
        f"unexpected record files in {record_path}: {sorted(names)}",
    )
    predictions_path = record_path / "predictions.npz"
    provenance_path = record_path / "provenance.json"
    for path in (predictions_path, provenance_path):
        _require(path.is_file() and not path.is_symlink(), f"bad artifact file {path}")

    prediction_file_hash = _file_sha256(predictions_path)
    provenance_file_hash = _file_sha256(provenance_path)
    _require(
        prediction_file_hash == manifest_record.get("predictions_file_sha256"),
        f"manifest prediction hash mismatch in {record_path}",
    )
    _require(
        provenance_file_hash == manifest_record.get("provenance_file_sha256"),
        f"manifest provenance hash mismatch in {record_path}",
    )

    provenance = _load_json(provenance_path, "record provenance")
    identity = _mapping(provenance.get("record"), "provenance.record")
    expected_identity = {
        "dataset": dataset,
        "subject": subject,
        "fold": fold,
        "seed": seed,
        "condition": condition,
    }
    observed_identity = {
        "dataset": identity.get("dataset"),
        "subject": identity.get("subject"),
        "fold": identity.get("fold"),
        "seed": identity.get("seed"),
        "condition": identity.get("condition"),
    }
    _require(observed_identity == expected_identity, f"record identity mismatch in {record_path}")
    prediction_record = _mapping(provenance.get("predictions"), "provenance.predictions")
    _require(
        prediction_record.get("file_sha256") == prediction_file_hash,
        f"provenance prediction file hash mismatch in {record_path}",
    )

    try:
        with np.load(predictions_path, allow_pickle=False) as archive:
            _require(set(archive.files) == PREDICTION_FIELDS, f"NPZ fields differ in {record_path}")
            scalar_identity = {
                "dataset": str(np.asarray(archive["dataset"]).item()),
                "subject": int(np.asarray(archive["subject"]).item()),
                "fold": int(np.asarray(archive["fold"]).item()),
                "condition": str(np.asarray(archive["condition"]).item()),
            }
            _require(
                scalar_identity
                == {
                    "dataset": dataset,
                    "subject": subject,
                    "fold": fold,
                    "condition": condition,
                },
                f"NPZ identity mismatch in {record_path}",
            )
            _require(
                str(np.asarray(archive["schema"]).item())
                == prediction_record.get("schema"),
                f"NPZ schema mismatch in {record_path}",
            )
            rows = np.asarray(archive["test_rows"])
            labels = np.asarray(archive["test_labels"])
            predicted = np.asarray(archive["predicted_labels"])
            probabilities = np.asarray(archive["probabilities"])
    except VerificationError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise VerificationError(f"cannot safely load {predictions_path}") from error

    row_count = len(rows)
    _require(rows.dtype == np.dtype(np.int64) and rows.shape == (row_count,), f"bad rows {record_path}")
    _require(row_count > 0 and len(np.unique(rows)) == row_count, f"duplicate/empty rows {record_path}")
    _require(bool(np.all(rows >= 0)), f"negative rows {record_path}")
    for name, values in (("test_labels", labels), ("predicted_labels", predicted)):
        _require(
            values.dtype == np.dtype(np.int64) and values.shape == (row_count,),
            f"bad {name} in {record_path}",
        )
        _require(set(values.tolist()) <= {0, 1}, f"nonbinary {name} in {record_path}")
    _require(
        probabilities.dtype == np.dtype(np.float64)
        and probabilities.shape == (row_count, 2)
        and bool(np.isfinite(probabilities).all()),
        f"bad probabilities in {record_path}",
    )
    _require(
        bool(np.all((probabilities >= 0.0) & (probabilities <= 1.0))),
        f"out-of-range probabilities in {record_path}",
    )
    _require(
        bool(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)),
        f"probabilities do not sum to one in {record_path}",
    )
    _require(
        bool(np.array_equal(predicted, probabilities.argmax(axis=1))),
        f"prediction/argmax mismatch in {record_path}",
    )
    for name, values in (
        ("test_rows", rows),
        ("test_labels", labels),
        ("predicted_labels", predicted),
        ("probabilities", probabilities),
    ):
        _require(
            prediction_record.get(f"{name}_sha256") == _array_sha256(values),
            f"array hash mismatch for {name} in {record_path}",
        )
    return rows.copy(), labels.copy(), predicted.copy()


def _verify_dataset(
    *, root: Path, dataset: str, gate: Mapping[str, object]
) -> tuple[dict[str, dict[int, float]], dict[str, dict[tuple[int, int], float]]]:
    _require(root.is_dir() and not root.is_symlink(), f"bad dataset root {root}")
    expected_names = _expected_names(dataset)
    actual_names = {entry.name for entry in root.iterdir()}
    _require(actual_names == expected_names, f"top-level grid mismatch for {dataset}")

    manifest = _manifest_records(gate, dataset)
    spec = DATASET_SPECS[dataset]
    expected_record_count = (
        len(spec["subjects"]) * len(spec["folds"]) * len(SEEDS) * len(CONDITIONS)
    )
    _require(len(manifest) == expected_record_count, f"manifest count mismatch for {dataset}")
    dataset_summary = _gate_dataset_summary(gate, dataset)
    gate_conditions = _mapping(dataset_summary.get("conditions"), f"{dataset} gate conditions")
    _require(set(gate_conditions) == set(CONDITIONS), f"condition mismatch for {dataset}")

    fold_truth: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    subject_seed_scores: dict[str, dict[tuple[int, int], float]] = {
        condition: {} for condition in CONDITIONS
    }
    subject_scores: dict[str, dict[int, float]] = {condition: {} for condition in CONDITIONS}

    for condition in CONDITIONS:
        condition_summary = _mapping(gate_conditions[condition], f"{dataset}.{condition}")
        gate_seed_scores = _mapping(
            condition_summary.get("subject_seed_balanced_accuracy"),
            f"{dataset}.{condition}.subject_seed_scores",
        )
        gate_subject_scores = _mapping(
            condition_summary.get("subject_balanced_accuracy"),
            f"{dataset}.{condition}.subject_scores",
        )
        for subject in spec["subjects"]:
            per_seed: list[float] = []
            for seed in SEEDS:
                rows_parts: list[np.ndarray] = []
                label_parts: list[np.ndarray] = []
                prediction_parts: list[np.ndarray] = []
                for fold in spec["folds"]:
                    key = (subject, fold, seed, condition)
                    _require(key in manifest, f"missing manifest record {dataset}:{key}")
                    path = root / _record_name(subject, fold, seed, condition)
                    rows, labels, predicted = _validate_prediction_archive(
                        record_path=path,
                        dataset=dataset,
                        subject=subject,
                        fold=fold,
                        seed=seed,
                        condition=condition,
                        manifest_record=manifest[key],
                    )
                    truth_key = (subject, fold)
                    if truth_key not in fold_truth:
                        fold_truth[truth_key] = (rows, labels)
                    else:
                        expected_rows, expected_labels = fold_truth[truth_key]
                        _require(
                            bool(np.array_equal(rows, expected_rows))
                            and bool(np.array_equal(labels, expected_labels)),
                            f"test identity changed for {dataset} subject/fold {truth_key}",
                        )
                    rows_parts.append(rows)
                    label_parts.append(labels)
                    prediction_parts.append(predicted)

                all_rows = np.concatenate(rows_parts)
                _require(
                    len(np.unique(all_rows)) == len(all_rows),
                    f"fold overlap for {dataset} S{subject} seed {seed} {condition}",
                )
                _require(
                    bool(np.array_equal(np.sort(all_rows), np.arange(len(all_rows)))),
                    f"fold union is not exact row coverage for {dataset} S{subject}",
                )
                score = _balanced_accuracy(
                    np.concatenate(label_parts), np.concatenate(prediction_parts)
                )
                subject_seed_scores[condition][(subject, seed)] = score
                _require_float(
                    score,
                    gate_seed_scores.get(f"{subject}:{seed}"),
                    f"{dataset}.{condition}.S{subject}.seed{seed}",
                )
                per_seed.append(score)
            subject_score = float(np.mean(np.asarray(per_seed, dtype=np.float64)))
            subject_scores[condition][subject] = subject_score
            _require_float(
                subject_score,
                gate_subject_scores.get(str(subject)),
                f"{dataset}.{condition}.S{subject}",
            )

        dataset_mean = float(
            np.mean(
                np.asarray(
                    [subject_scores[condition][subject] for subject in spec["subjects"]],
                    dtype=np.float64,
                )
            )
        )
        _require_float(
            dataset_mean,
            condition_summary.get("equal_subject_mean_balanced_accuracy"),
            f"{dataset}.{condition}.nested_mean",
        )
        inputs = _mapping(gate.get("inputs"), "gate.inputs")
        full_summary = _mapping(
            inputs.get("recomputed_full_grid_summary"), "gate full summary"
        )
        top_means = _mapping(
            full_summary.get("dataset_condition_mean_balanced_accuracy"),
            "gate dataset means",
        )
        dataset_means = _mapping(top_means.get(dataset), f"gate means {dataset}")
        _require_float(
            dataset_mean,
            dataset_means.get(condition),
            f"{dataset}.{condition}.summary_top_mean",
        )
        gate_dataset_results = _mapping(gate.get("dataset_results"), "gate.dataset_results")
        gate_dataset_result = _mapping(
            gate_dataset_results.get(dataset), f"gate.dataset_results.{dataset}"
        )
        result_means = _mapping(
            gate_dataset_result.get("condition_mean_balanced_accuracy"),
            f"gate.dataset_results.{dataset}.means",
        )
        _require_float(
            dataset_mean,
            result_means.get(condition),
            f"{dataset}.{condition}.gate_top_mean",
        )
    return subject_scores, subject_seed_scores


def _bootstrap(
    *,
    subject_scores: Mapping[str, Mapping[str, Mapping[int, float]]],
    comparator: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    differences: dict[str, np.ndarray] = {}
    for dataset in DATASETS:
        subjects = DATASET_SPECS[dataset]["subjects"]
        candidate = np.asarray(
            [subject_scores[dataset][CANDIDATE][subject] for subject in subjects],
            dtype=np.float64,
        )
        control = np.asarray(
            [subject_scores[dataset][comparator][subject] for subject in subjects],
            dtype=np.float64,
        )
        differences[dataset] = candidate - control

    distributions = {
        dataset: np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
        for dataset in DATASETS
    }
    distributions["equal_dataset_macro"] = np.empty(
        BOOTSTRAP_REPETITIONS, dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    offset = 0
    while offset < BOOTSTRAP_REPETITIONS:
        count = min(BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_REPETITIONS - offset)
        dataset_means: list[np.ndarray] = []
        for dataset in DATASETS:
            diff = differences[dataset]
            indices = generator.integers(
                0, len(diff), size=(count, len(diff)), dtype=np.int64
            )
            values = diff[indices].mean(axis=1)
            distributions[dataset][offset : offset + count] = values
            dataset_means.append(values)
        distributions["equal_dataset_macro"][offset : offset + count] = np.mean(
            np.stack(dataset_means, axis=0), axis=0
        )
        offset += count

    alpha = 1.0 - BOOTSTRAP_CONFIDENCE_LEVEL
    intervals: dict[str, dict[str, float]] = {}
    for name, values in distributions.items():
        intervals[name] = {
            "one_sided_95_lower": float(np.quantile(values, alpha, method="linear")),
            "two_sided_95_lower": float(
                np.quantile(values, alpha / 2.0, method="linear")
            ),
            "two_sided_95_upper": float(
                np.quantile(values, 1.0 - alpha / 2.0, method="linear")
            ),
        }
    return differences, intervals


def _verify_bootstrap(
    *,
    gate: Mapping[str, object],
    subject_scores: Mapping[str, Mapping[str, Mapping[int, float]]],
    input_name: str,
    comparator: str,
    seed: int,
) -> dict[str, object]:
    inputs = _mapping(gate.get("inputs"), "gate.inputs")
    reported = _mapping(inputs.get(input_name), f"gate.inputs.{input_name}")
    _require(reported.get("repetitions") == BOOTSTRAP_REPETITIONS, "bootstrap repetitions differ")
    _require(reported.get("seed") == seed, "bootstrap seed differs")
    _require(reported.get("candidate_condition") == CANDIDATE, "bootstrap candidate differs")
    _require(reported.get("comparator_condition") == comparator, "bootstrap comparator differs")
    differences, intervals = _bootstrap(
        subject_scores=subject_scores, comparator=comparator, seed=seed
    )
    reported_datasets = _mapping(reported.get("datasets"), "bootstrap datasets")
    for dataset in DATASETS:
        result = _mapping(reported_datasets.get(dataset), f"bootstrap {dataset}")
        subjects = DATASET_SPECS[dataset]["subjects"]
        candidate = np.asarray(
            [subject_scores[dataset][CANDIDATE][subject] for subject in subjects],
            dtype=np.float64,
        )
        control = np.asarray(
            [subject_scores[dataset][comparator][subject] for subject in subjects],
            dtype=np.float64,
        )
        _require(
            result.get("candidate_subject_scores_sha256") == _array_sha256(candidate),
            f"candidate score hash mismatch for {dataset}/{comparator}",
        )
        _require(
            result.get("comparator_subject_scores_sha256") == _array_sha256(control),
            f"comparator score hash mismatch for {dataset}/{comparator}",
        )
        _require(
            result.get("paired_subject_differences_sha256")
            == _array_sha256(differences[dataset]),
            f"difference hash mismatch for {dataset}/{comparator}",
        )
        _require_float(
            differences[dataset].mean(),
            result.get("observed_mean_balanced_accuracy_delta"),
            f"observed bootstrap delta {dataset}/{comparator}",
        )
        for field, value in intervals[dataset].items():
            _require_float(value, result.get(field), f"bootstrap {dataset}/{comparator}.{field}")
    reported_macro = _mapping(reported.get("equal_dataset_macro"), "bootstrap macro")
    observed_macro = float(np.mean([differences[dataset].mean() for dataset in DATASETS]))
    _require_float(
        observed_macro,
        reported_macro.get("observed_mean_balanced_accuracy_delta"),
        f"observed macro delta/{comparator}",
    )
    for field, value in intervals["equal_dataset_macro"].items():
        _require_float(value, reported_macro.get(field), f"bootstrap macro/{comparator}.{field}")
    return {
        "comparator": comparator,
        "seed": seed,
        "observed_macro_delta": observed_macro,
        "intervals": intervals,
    }


def verify(
    *, cho_root: Path, physionet_root: Path, gate_path: Path
) -> dict[str, object]:
    gate = _load_json(gate_path, "gate JSON")
    _require(gate.get("overall_pass") is True, "gate artifact does not report PASS")
    _require(gate.get("confirmation_access") is False, "gate reports confirmation access")
    roots = {"cho2017": cho_root, "physionet_mi": physionet_root}
    subject_scores: dict[str, dict[str, dict[int, float]]] = {}
    record_count = 0
    for dataset in DATASETS:
        scores, _ = _verify_dataset(root=roots[dataset], dataset=dataset, gate=gate)
        subject_scores[dataset] = scores
        spec = DATASET_SPECS[dataset]
        record_count += len(spec["subjects"]) * len(spec["folds"]) * len(SEEDS) * len(CONDITIONS)
    _require(record_count == 8675, "independent record count differs from 8,675")

    raw_means = {
        dataset: {
            condition: float(
                np.mean(
                    np.asarray(
                        [
                            subject_scores[dataset][condition][subject]
                            for subject in DATASET_SPECS[dataset]["subjects"]
                        ],
                        dtype=np.float64,
                    )
                )
            )
            for condition in CONDITIONS
        }
        for dataset in DATASETS
    }
    macros = {
        condition: float(np.mean([raw_means[d][condition] for d in DATASETS]))
        for condition in CONDITIONS
    }
    gate_macro = _mapping(gate.get("equal_dataset_macro"), "gate macro")
    gate_macro_means = _mapping(
        gate_macro.get("condition_mean_balanced_accuracy"), "gate macro means"
    )
    for condition in CONDITIONS:
        _require_float(macros[condition], gate_macro_means.get(condition), f"macro.{condition}")

    scratch_bootstrap = _verify_bootstrap(
        gate=gate,
        subject_scores=subject_scores,
        input_name="internally_computed_paired_subject_bootstrap",
        comparator=SCRATCH,
        seed=20_260_720,
    )
    primary_bootstrap = _verify_bootstrap(
        gate=gate,
        subject_scores=subject_scores,
        input_name="internally_computed_primary_checkpoint_matched_paired_subject_bootstrap",
        comparator=PRIMARY,
        seed=20_260_721,
    )

    scratch_delta = macros[CANDIDATE] - macros[SCRATCH]
    scratch_by_dataset = {
        dataset: raw_means[dataset][CANDIDATE] - raw_means[dataset][SCRATCH]
        for dataset in DATASETS
    }
    envelope_by_dataset = {
        dataset: max(
            [raw_means[dataset][condition] for condition in REFERENCE_CONDITIONS]
            + [HISTORICAL_FLOORS[dataset]]
        )
        for dataset in DATASETS
    }
    envelope_deltas = {
        dataset: raw_means[dataset][CANDIDATE] - envelope_by_dataset[dataset]
        for dataset in DATASETS
    }
    envelope_macro_delta = float(np.mean(list(envelope_deltas.values())))
    scratch_intervals = scratch_bootstrap["intervals"]
    primary_intervals = primary_bootstrap["intervals"]
    checks = {
        "scratch_macro_at_least_1pp": scratch_delta >= 0.01,
        "scratch_positive_each_dataset": all(value > 0.0 for value in scratch_by_dataset.values()),
        "scratch_bootstrap_lower_positive_each_dataset_and_macro": all(
            scratch_intervals[name]["one_sided_95_lower"] > 0.0
            for name in (*DATASETS, "equal_dataset_macro")
        ),
        "primary_macro_bootstrap_lower_positive": (
            primary_intervals["equal_dataset_macro"]["one_sided_95_lower"] > 0.0
        ),
        "reference_envelope_macro_at_least_0_5pp": envelope_macro_delta >= 0.005,
        "worst_reference_envelope_delta_at_least_minus_1pp": (
            min(envelope_deltas.values()) >= -0.01
        ),
    }
    _require(all(checks.values()), "independently recomputed gate does not pass")
    _require(gate.get("overall_pass") is all(checks.values()), "overall gate decision differs")

    inputs = _mapping(gate.get("inputs"), "gate.inputs")
    manifest = _mapping(
        inputs.get("freshly_revalidated_full_grid_audit"), "gate manifest"
    )
    canonical_manifest = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    manifest_hash = hashlib.sha256(canonical_manifest).hexdigest()
    _require(
        manifest_hash == inputs.get("freshly_revalidated_full_grid_audit_sha256"),
        "embedded manifest SHA-256 differs",
    )
    return {
        "schema": "eeg-mi-cardinal-fbms-post-outcome-independent-verification-v1",
        "status": "PASS",
        "post_outcome_verification": True,
        "not_the_frozen_gate": True,
        "gate_file_sha256": _file_sha256(gate_path),
        "embedded_manifest_sha256": manifest_hash,
        "validated_record_count": record_count,
        "dataset_condition_mean_balanced_accuracy": raw_means,
        "equal_dataset_condition_mean_balanced_accuracy": macros,
        "independent_gate_checks": checks,
        "scratch_bootstrap": scratch_bootstrap,
        "primary_bootstrap": primary_bootstrap,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cho-root", required=True, type=Path)
    parser.add_argument("--physionet-root", required=True, type=Path)
    parser.add_argument("--gate-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    result = verify(
        cho_root=arguments.cho_root.expanduser().resolve(),
        physionet_root=arguments.physionet_root.expanduser().resolve(),
        gate_path=arguments.gate_json.expanduser().resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
