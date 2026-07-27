"""Fair, non-transductive geometric controls for the local Exp4 tournament.

This runner is deliberately separate from :mod:`ieee_mi.local_model_tournament`.
The neural tournament compares raw-trial neural architectures, whereas the three
controls here consume a fixed, model-side filter bank derived from the same frozen
``ieee-mi-cache-v2`` broadband trials.

For every participant the protocol is:

* recordings 1--2 fit each candidate estimator;
* recording 3 selects one predeclared hyperparameter by negative log likelihood;
* recordings 1--3 refit every learned scaler/reference/filter/classifier state;
* recording 4 is passed to ``predict_proba`` exactly once and is never calibrated.

The estimators are deterministic.  The formal five-seed identities are retained so
the artifact has the same Cartesian subject/seed contract as the neural tournament,
but every seed record is an explicitly declared deterministic replication of one
fit.  This must not be interpreted as five independent optimization runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal
from sklearn.metrics import log_loss

from . import benchmark
from .config import CHANNEL_SCALING, preprocessing_for_dataset
from .data import (
    apply_channel_scaler,
    fit_channel_scaler,
    load_subject_cache,
    split_indices,
)
from .training import classification_metrics


SCHEMA = "ieee-mi-local-geometric-controls-v1"
DATASET = "local_exp4"
SUBJECTS: tuple[int, ...] = (1, 3, 4, 5, 6, 7, 8, 10)
SEEDS: tuple[int, ...] = benchmark.FORMAL_DEVELOPMENT_SEEDS
MODELS: tuple[str, ...] = ("riemann", "tangent_anchor", "ea_fbcsp")

# These bands preserve the historical geometric-control hypothesis while making
# the transform an explicit model-side operation on the current v2 cache.  They
# are not represented as the legacy 125 Hz/minimum-phase preprocessing pipeline.
FILTER_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (8.0, 12.0),
    (11.0, 15.0),
    (14.0, 20.0),
    (20.0, 30.0),
)
FIR_TAPS = 65
FIR_WINDOW = "hamming"
COVARIANCE_SHRINKAGE = 1e-3

HYPERPARAMETER_GRID: dict[str, tuple[dict[str, float | int], ...]] = {
    "riemann": tuple({"c": value} for value in (0.01, 0.1, 1.0, 10.0)),
    "tangent_anchor": tuple(
        {"C": value} for value in (0.01, 0.1, 1.0, 10.0)
    ),
    "ea_fbcsp": tuple(
        {"n_components": value} for value in (2, 4, 6)
    ),
}

SOURCE_FILES: tuple[str, ...] = (
    "ieee_mi/local_geometric_controls.py",
    "ieee_mi/config.py",
    "ieee_mi/data.py",
    "ieee_mi/training.py",
    "deepnet/baselines.py",
    "deepnet/tangent_anchor.py",
)


@dataclass(frozen=True)
class FeaturePartition:
    """One split represented both as filter-bank epochs and SPD covariances."""

    epochs: np.ndarray
    covariances: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class PreparedSubject:
    """Selection and final-refit features for one local participant."""

    train: FeaturePartition
    validation: FeaturePartition
    source: FeaturePartition
    test: FeaturePartition
    train_rows: np.ndarray
    validation_rows: np.ndarray
    source_rows: np.ndarray
    test_rows: np.ndarray
    selection_mean: np.ndarray
    selection_std: np.ndarray
    refit_mean: np.ndarray
    refit_std: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def strict_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _fir_coefficients(sfreq: float) -> np.ndarray:
    return np.stack(
        [
            signal.firwin(
                FIR_TAPS,
                (low, high),
                pass_zero=False,
                fs=sfreq,
                window=FIR_WINDOW,
                scale=True,
            )
            for low, high in FILTER_BANDS_HZ
        ],
        axis=0,
    ).astype(np.float64, copy=False)


def fixed_filter_bank(x: np.ndarray, *, sfreq: float = 128.0) -> np.ndarray:
    """Apply the frozen symmetric FIR bank with reflection boundary padding.

    The cache already contains continuously filtered 4--40 Hz, CAR-referenced,
    demeaned trials.  This is a deterministic model-side transform; it estimates
    no state from any split and can therefore be applied independently to R4.
    """

    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] <= FIR_TAPS:
        raise ValueError("x must have shape (trials, channels, time) and exceed FIR length")
    if not np.all(np.isfinite(values)):
        raise ValueError("x contains non-finite values")
    half = FIR_TAPS // 2
    padded = np.pad(values, ((0, 0), (0, 0), (half, half)), mode="reflect")
    bands: list[np.ndarray] = []
    for coefficients in _fir_coefficients(sfreq):
        filtered = signal.fftconvolve(
            padded,
            coefficients.reshape(1, 1, -1),
            mode="valid",
            axes=-1,
        )
        filtered -= filtered.mean(axis=-1, keepdims=True)
        bands.append(filtered)
    result = np.stack(bands, axis=1)
    if result.shape != (len(values), len(FILTER_BANDS_HZ), *values.shape[1:]):
        raise RuntimeError("fixed filter bank returned an unexpected shape")
    return np.ascontiguousarray(result, dtype=np.float32)


def spd_covariances(epochs: np.ndarray) -> np.ndarray:
    """Return trace-shrunk per-band spatial covariances in stable float64 math."""

    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] < 2:
        raise ValueError("epochs must have shape (trials, bands, channels, time)")
    values = values - values.mean(axis=-1, keepdims=True)
    covariances = np.einsum(
        "nbct,nbdt->nbcd", values, values, optimize=True
    ) / float(values.shape[-1])
    channels = covariances.shape[-1]
    trace_scale = np.trace(covariances, axis1=-2, axis2=-1) / float(channels)
    identity = np.eye(channels, dtype=np.float64)
    covariances = (
        (1.0 - COVARIANCE_SHRINKAGE) * covariances
        + COVARIANCE_SHRINKAGE
        * trace_scale[..., None, None]
        * identity
    )
    covariances = 0.5 * (covariances + covariances.swapaxes(-1, -2))
    floor = np.maximum(
        np.abs(trace_scale) * np.finfo(np.float64).eps * 8.0,
        np.finfo(np.float64).tiny * 8.0,
    )
    minimum = np.linalg.eigvalsh(covariances)[..., 0]
    covariances += np.maximum(0.0, floor - minimum)[..., None, None] * identity
    if not np.all(np.isfinite(covariances)) or np.min(
        np.linalg.eigvalsh(covariances)
    ) <= 0.0:
        raise RuntimeError("covariance construction failed to produce SPD matrices")
    return np.ascontiguousarray(covariances, dtype=np.float64)


def _partition(x: np.ndarray, y: np.ndarray, *, sfreq: float) -> FeaturePartition:
    epochs = fixed_filter_bank(x, sfreq=sfreq)
    return FeaturePartition(
        epochs=epochs,
        covariances=spd_covariances(epochs),
        labels=np.asarray(y, dtype=np.int64),
    )


def prepare_subject(data: Mapping[str, Any], subject: int) -> PreparedSubject:
    """Prepare split-specific features without fitting anything on R4."""

    train_rows, validation_rows, test_rows = split_indices(
        DATASET,
        np.asarray(data["y"]),
        np.asarray(data["sessions"]),
        np.asarray(data["runs"]),
        fold=0,
        subject=subject,
    )
    source_rows = np.sort(np.concatenate((train_rows, validation_rows)))
    channel_names = tuple(str(value) for value in np.asarray(data["channel_names"]).tolist())
    selection_mean, selection_std = fit_channel_scaler(
        np.asarray(data["x"])[train_rows], channel_names
    )
    refit_mean, refit_std = fit_channel_scaler(
        np.asarray(data["x"])[source_rows], channel_names
    )
    sfreq = float(preprocessing_for_dataset(DATASET)["sfreq_hz"])
    return PreparedSubject(
        train=_partition(
            apply_channel_scaler(
                np.asarray(data["x"])[train_rows], selection_mean, selection_std
            ),
            np.asarray(data["y"])[train_rows],
            sfreq=sfreq,
        ),
        validation=_partition(
            apply_channel_scaler(
                np.asarray(data["x"])[validation_rows], selection_mean, selection_std
            ),
            np.asarray(data["y"])[validation_rows],
            sfreq=sfreq,
        ),
        source=_partition(
            apply_channel_scaler(
                np.asarray(data["x"])[source_rows], refit_mean, refit_std
            ),
            np.asarray(data["y"])[source_rows],
            sfreq=sfreq,
        ),
        test=_partition(
            apply_channel_scaler(
                np.asarray(data["x"])[test_rows], refit_mean, refit_std
            ),
            np.asarray(data["y"])[test_rows],
            sfreq=sfreq,
        ),
        train_rows=train_rows,
        validation_rows=validation_rows,
        source_rows=source_rows,
        test_rows=test_rows,
        selection_mean=selection_mean,
        selection_std=selection_std,
        refit_mean=refit_mean,
        refit_std=refit_std,
    )


def _new_estimator(model: str, parameters: Mapping[str, float | int]) -> Any:
    if model == "riemann":
        from deepnet.baselines import RiemannianTangentLogistic

        return RiemannianTangentLogistic(
            c=float(parameters["c"]), max_iter=3000
        )
    if model == "tangent_anchor":
        from deepnet.tangent_anchor import TangentAnchorClassifier

        return TangentAnchorClassifier(
            C=float(parameters["C"]), max_iter=3000
        )
    if model == "ea_fbcsp":
        from deepnet.baselines import EAFilterBankCSP

        return EAFilterBankCSP(n_components=int(parameters["n_components"]))
    raise ValueError(f"unknown geometric control {model!r}")


def _model_features(model: str, partition: FeaturePartition) -> np.ndarray:
    return partition.epochs if model == "ea_fbcsp" else partition.covariances


def _valid_probabilities(probabilities: Any, count: int) -> np.ndarray:
    result = np.asarray(probabilities, dtype=np.float64)
    if (
        result.shape != (count, 2)
        or not np.all(np.isfinite(result))
        or np.any(result < 0.0)
        or np.any(result > 1.0)
        or not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise RuntimeError("estimator returned invalid binary probabilities")
    return result


def _estimator_state_sha256(estimator: Any) -> str:
    # The hash is provenance, not an interchange format.  Package versions and
    # source hashes in the artifact make the pickle protocol/environment explicit.
    return _sha256_bytes(pickle.dumps(estimator, protocol=5))


def _fit_estimator(
    model: str, estimator: Any, features: np.ndarray, labels: np.ndarray
) -> Any:
    if model == "ea_fbcsp" and type(estimator).__module__.startswith("deepnet."):
        # MNE's CSP rank/covariance messages are useful interactively but would
        # otherwise dominate the resumable experiment log (four bands per fit).
        import mne

        with mne.use_log_level("ERROR"):
            return estimator.fit(features, labels)
    return estimator.fit(features, labels)


def select_and_refit(
    model: str, prepared: PreparedSubject
) -> dict[str, Any]:
    """Select on R3, refit on R1--3, and predict R4 without calibration."""

    if model not in MODELS:
        raise ValueError(f"unknown geometric control {model!r}")
    train_x = _model_features(model, prepared.train)
    validation_x = _model_features(model, prepared.validation)
    source_x = _model_features(model, prepared.source)
    test_x = _model_features(model, prepared.test)

    trace: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_parameters: dict[str, float | int] | None = None
    best_estimator: Any | None = None
    best_validation_probability: np.ndarray | None = None
    selection_started = time.perf_counter()
    for parameters in HYPERPARAMETER_GRID[model]:
        estimator = _new_estimator(model, parameters)
        _fit_estimator(model, estimator, train_x, prepared.train.labels)
        probability = _valid_probabilities(
            estimator.predict_proba(validation_x), len(prepared.validation.labels)
        )
        loss = float(
            log_loss(prepared.validation.labels, probability, labels=(0, 1))
        )
        trace.append({"parameters": dict(parameters), "validation_nll": loss})
        if loss < best_loss - 1e-12:
            best_loss = loss
            best_parameters = dict(parameters)
            best_estimator = estimator
            best_validation_probability = probability
    if best_parameters is None or best_estimator is None or best_validation_probability is None:
        raise RuntimeError("geometric-control selection produced no estimator")
    selection_seconds = time.perf_counter() - selection_started

    refit_started = time.perf_counter()
    final_estimator = _new_estimator(model, best_parameters)
    _fit_estimator(model, final_estimator, source_x, prepared.source.labels)
    # Deliberately no ``calibrate`` call here.  R4 is prediction-only.
    test_probability = _valid_probabilities(
        final_estimator.predict_proba(test_x), len(prepared.test.labels)
    )
    refit_seconds = time.perf_counter() - refit_started
    return {
        "selected_parameters": best_parameters,
        "selection_trace": trace,
        "selection_validation_nll": best_loss,
        "selection_validation_metrics": classification_metrics(
            prepared.validation.labels, best_validation_probability
        ),
        "selection_state_sha256": _estimator_state_sha256(best_estimator),
        "refit_state_sha256": _estimator_state_sha256(final_estimator),
        "selection_fit_seconds": selection_seconds,
        "refit_fit_seconds": refit_seconds,
        "test_probability": test_probability,
    }


def _feature_contract() -> dict[str, Any]:
    sfreq = float(preprocessing_for_dataset(DATASET)["sfreq_hz"])
    coefficients = _fir_coefficients(sfreq)
    return {
        "schema": "ieee-mi-local-fixed-geometric-features-v1",
        "input": "ieee-mi-cache-v2 broadband trials after split-specific channel scaling",
        "sfreq_hz": sfreq,
        "filter_bands_hz": [list(value) for value in FILTER_BANDS_HZ],
        "fir_taps": FIR_TAPS,
        "fir_window": FIR_WINDOW,
        "fir_design": "scipy.signal.firwin bandpass scale=True",
        "application": "single symmetric convolution",
        "boundary": "reflection padding by floor(fir_taps/2), then valid convolution",
        "post_filter_epoch_demean": True,
        "coefficient_sha256": _array_sha256(coefficients),
        "covariance": {
            "demean": True,
            "normalization": "divide by n_times",
            "shrinkage": COVARIANCE_SHRINKAGE,
            "target": "trace(C)/n_channels * identity",
            "computation_dtype": "float64",
        },
    }


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: _file_sha256(root / name) for name in SOURCE_FILES}


def _cache_manifest(cache_root: Path) -> dict[str, str]:
    return {
        str(subject): str(
            load_subject_cache(DATASET, subject, cache_root=cache_root)["identity"][
                "array_sha256"
            ]
        )
        for subject in SUBJECTS
    }


def _contract(cache_root: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dataset": DATASET,
        "evidence_scope": "formal_development_not_confirmation",
        "confirmation_evidence": False,
        "subjects": list(SUBJECTS),
        "folds": [0],
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "seed_policy": (
            "formal seed identities retained; estimators are deterministic and "
            "one canonical fit is replicated exactly across seed records"
        ),
        "canonical_fit_seed": SEEDS[0],
        "split_protocol": "R1-2 selection fit; R3 selection; R1-3 refit; R4 prediction-only",
        "test_calibration": False,
        "preprocessing": preprocessing_for_dataset(DATASET),
        "channel_scaling": CHANNEL_SCALING,
        "feature_contract": _feature_contract(),
        "hyperparameter_grid": {
            model: [dict(value) for value in values]
            for model, values in HYPERPARAMETER_GRID.items()
        },
        "primary_metric": (
            "mean subject balanced accuracy after averaging deterministic seed identities"
        ),
        "cache_array_sha256": _cache_manifest(cache_root),
        "source_sha256": _source_manifest(),
        "environment": benchmark._environment(),
    }


def _split_detail(prepared: PreparedSubject, data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "train_count": len(prepared.train_rows),
        "validation_count": len(prepared.validation_rows),
        "refit_source_count": len(prepared.source_rows),
        "test_count": len(prepared.test_rows),
        "train_runs": sorted(set(np.asarray(data["runs"])[prepared.train_rows].tolist())),
        "validation_runs": sorted(
            set(np.asarray(data["runs"])[prepared.validation_rows].tolist())
        ),
        "test_runs": sorted(set(np.asarray(data["runs"])[prepared.test_rows].tolist())),
        "train_rows_sha256": _array_sha256(prepared.train_rows),
        "validation_rows_sha256": _array_sha256(prepared.validation_rows),
        "test_rows_sha256": _array_sha256(prepared.test_rows),
    }


def _feature_hashes(prepared: PreparedSubject) -> dict[str, dict[str, str]]:
    return {
        name: {
            "epochs_sha256": _array_sha256(partition.epochs),
            "covariances_sha256": _array_sha256(partition.covariances),
        }
        for name, partition in (
            ("selection_train", prepared.train),
            ("selection_validation", prepared.validation),
            ("refit_source", prepared.source),
            ("prediction_test", prepared.test),
        )
    }


def _records_for_result(
    *,
    model: str,
    subject: int,
    result: Mapping[str, Any],
    prepared: PreparedSubject,
    data: Mapping[str, Any],
) -> list[dict[str, Any]]:
    probability = np.asarray(result["test_probability"], dtype=np.float64)
    metrics = classification_metrics(prepared.test.labels, probability)
    common = {
        "dataset": DATASET,
        "subject": subject,
        "fold": 0,
        "model": model,
        "deterministic_estimator": True,
        "canonical_fit_seed": SEEDS[0],
        "split": _split_detail(prepared, data),
        "scaler": {
            "selection_train": {
                "mean": prepared.selection_mean.reshape(-1).tolist(),
                "std": prepared.selection_std.reshape(-1).tolist(),
            },
            "refit_source": {
                "mean": prepared.refit_mean.reshape(-1).tolist(),
                "std": prepared.refit_std.reshape(-1).tolist(),
            },
        },
        "feature_sha256": _feature_hashes(prepared),
        "fit": {
            key: result[key]
            for key in (
                "selected_parameters",
                "selection_trace",
                "selection_validation_nll",
                "selection_validation_metrics",
                "selection_state_sha256",
                "refit_state_sha256",
                "selection_fit_seconds",
                "refit_fit_seconds",
            )
        },
        "test": {
            "metrics": metrics,
            "rows": prepared.test_rows.tolist(),
            "labels": prepared.test.labels.tolist(),
            "probabilities": probability.tolist(),
            "sessions": np.asarray(data["sessions"])[prepared.test_rows].tolist(),
            "runs": np.asarray(data["runs"])[prepared.test_rows].tolist(),
        },
        "cache_identity": data["identity"],
    }
    return [
        {
            **common,
            "seed": seed,
            "seed_role": (
                "canonical_deterministic_fit"
                if seed == SEEDS[0]
                else "exact_deterministic_replication"
            ),
        }
        for seed in SEEDS
    ]


def _metric_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _metric_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _metric_values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12))
    return left == right


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in MODELS:
        model_rows = [record for record in records if record["model"] == model]
        by_subject: dict[int, list[float]] = {}
        for record in model_rows:
            by_subject.setdefault(int(record["subject"]), []).append(
                float(record["test"]["metrics"]["balanced_accuracy"])
            )
        subject_values = {
            subject: float(np.mean(values))
            for subject, values in sorted(by_subject.items())
        }
        values = np.asarray(list(subject_values.values()), dtype=np.float64)
        result[model] = {
            "balanced_accuracy_mean": float(values.mean()) if len(values) else None,
            "balanced_accuracy_std": (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ),
            "n_subjects": len(values),
            "n_records": len(model_rows),
            "subject_balanced_accuracy": {
                str(subject): value for subject, value in subject_values.items()
            },
        }
    return result


def validate_payload(
    payload: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    cache_root: Path,
    require_complete: bool = False,
) -> None:
    for key, expected in contract.items():
        if payload.get(key) != expected:
            raise ValueError(f"artifact differs from requested contract at {key!r}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("artifact records must be a list")
    expected_keys = {
        (model, subject, seed)
        for model in MODELS
        for subject in SUBJECTS
        for seed in SEEDS
    }
    seen: set[tuple[str, int, int]] = set()
    bundles: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    caches: dict[int, Mapping[str, Any]] = {}
    for record in records:
        key = (str(record["model"]), int(record["subject"]), int(record["seed"]))
        if key not in expected_keys or key in seen:
            raise ValueError(f"invalid or duplicate record key {key}")
        seen.add(key)
        model, subject, _ = key
        bundles.setdefault((model, subject), []).append(record)
        data = caches.setdefault(
            subject, load_subject_cache(DATASET, subject, cache_root=cache_root)
        )
        train_rows, validation_rows, test_rows = split_indices(
            DATASET,
            data["y"],
            data["sessions"],
            data["runs"],
            subject=subject,
        )
        source_rows = np.sort(np.concatenate((train_rows, validation_rows)))
        expected_split = {
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "refit_source_count": len(source_rows),
            "test_count": len(test_rows),
            "train_runs": sorted(set(data["runs"][train_rows].tolist())),
            "validation_runs": sorted(
                set(data["runs"][validation_rows].tolist())
            ),
            "test_runs": sorted(set(data["runs"][test_rows].tolist())),
            "train_rows_sha256": _array_sha256(train_rows),
            "validation_rows_sha256": _array_sha256(validation_rows),
            "test_rows_sha256": _array_sha256(test_rows),
        }
        if record.get("split") != expected_split:
            raise ValueError(f"record {key} has a stale split contract")
        test = record["test"]
        if test.get("rows") != test_rows.tolist() or test.get("labels") != data["y"][
            test_rows
        ].tolist():
            raise ValueError(f"record {key} has stale R4 rows or labels")
        probability = _valid_probabilities(test.get("probabilities"), len(test_rows))
        recomputed = classification_metrics(data["y"][test_rows], probability)
        if not _metric_values_equal(test.get("metrics"), recomputed):
            raise ValueError(f"record {key} has stale test metrics")
        if record.get("cache_identity") != data["identity"]:
            raise ValueError(f"record {key} has stale cache identity")
        if record.get("deterministic_estimator") is not True:
            raise ValueError(f"record {key} is not marked deterministic")
        selected = record.get("fit", {}).get("selected_parameters")
        if selected not in HYPERPARAMETER_GRID[model]:
            raise ValueError(f"record {key} selected parameters outside the grid")
        trace = record.get("fit", {}).get("selection_trace")
        if not isinstance(trace, list) or len(trace) != len(HYPERPARAMETER_GRID[model]):
            raise ValueError(f"record {key} has an incomplete selection trace")
        for state_key in ("selection_state_sha256", "refit_state_sha256"):
            value = record.get("fit", {}).get(state_key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"record {key} has invalid {state_key}")

    for bundle_key, rows in bundles.items():
        if {int(row["seed"]) for row in rows} != set(SEEDS):
            raise ValueError(f"partial deterministic seed bundle {bundle_key}")
        canonical = sorted(rows, key=lambda row: int(row["seed"]))[0]
        invariant_fields = (
            "split",
            "scaler",
            "feature_sha256",
            "fit",
            "test",
            "cache_identity",
        )
        for row in rows[1:]:
            for field in invariant_fields:
                if not _metric_values_equal(row[field], canonical[field]):
                    raise ValueError(
                        f"deterministic seed bundle {bundle_key} differs at {field}"
                    )
    if require_complete and seen != expected_keys:
        missing = sorted(expected_keys - seen)
        raise ValueError(f"artifact is incomplete; missing={missing}")
    recomputed_summary = _summary(records)
    if not _metric_values_equal(payload.get("summary"), recomputed_summary):
        raise ValueError("artifact summary is stale")


def _load_or_create(
    output: Path, *, contract: Mapping[str, Any], cache_root: Path, resume: bool
) -> dict[str, Any]:
    if output.exists():
        if not resume:
            raise FileExistsError(f"{output} exists; pass --resume to validate and continue")
        payload = strict_load(output)
        validate_payload(payload, contract=contract, cache_root=cache_root)
        return payload
    return {
        **contract,
        "created_at": _utc_now(),
        "updated_at": None,
        "records": [],
        "summary": _summary([]),
    }


def run(*, cache_root: Path, output: Path, resume: bool) -> dict[str, Any]:
    cache_root = cache_root.resolve()
    output = output.resolve()
    contract = _contract(cache_root)
    payload = _load_or_create(
        output, contract=contract, cache_root=cache_root, resume=resume
    )
    complete = {
        (str(record["model"]), int(record["subject"]))
        for record in payload["records"]
    }
    for subject in SUBJECTS:
        pending = [model for model in MODELS if (model, subject) not in complete]
        if not pending:
            continue
        data = load_subject_cache(DATASET, subject, cache_root=cache_root)
        prepared = prepare_subject(data, subject)
        for model in pending:
            started = time.perf_counter()
            result = select_and_refit(model, prepared)
            new_records = _records_for_result(
                model=model,
                subject=subject,
                result=result,
                prepared=prepared,
                data=data,
            )
            payload["records"].extend(new_records)
            payload["records"].sort(
                key=lambda row: (
                    MODELS.index(str(row["model"])),
                    SUBJECTS.index(int(row["subject"])),
                    SEEDS.index(int(row["seed"])),
                )
            )
            payload["summary"] = _summary(payload["records"])
            payload["updated_at"] = _utc_now()
            _atomic_json(output, payload)
            score = new_records[0]["test"]["metrics"]["balanced_accuracy"]
            print(
                f"{model:14s} S{subject:<2d} R4 BA={100*float(score):.2f}% "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
    validate_payload(
        payload,
        contract=contract,
        cache_root=cache_root,
        require_complete=True,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    payload = run(
        cache_root=args.cache_root,
        output=args.output,
        resume=bool(args.resume),
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DATASET",
    "FILTER_BANDS_HZ",
    "MODELS",
    "SEEDS",
    "SUBJECTS",
    "FeaturePartition",
    "PreparedSubject",
    "fixed_filter_bank",
    "prepare_subject",
    "run",
    "select_and_refit",
    "spd_covariances",
    "strict_load",
    "validate_payload",
]
