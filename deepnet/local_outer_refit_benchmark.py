"""Exact local outer-row rerun for the four legacy deepnet procedures.

This runner is intentionally separate from both the historical ``deepnet``
benchmarks and the ``ieee_mi`` neural tournament.  It consumes the frozen
``local_exp4`` v2 cache and enforces, per participant and seed:

* Phase A: recordings 1--2 fit, recording 3 selects checkpoint duration and
  any predeclared route/candidate;
* Phase B: reset, refit source-only preprocessing on recordings 1--3, and fit
  for exactly the selected duration; and
* prediction-only: recording 4 is passed to the refitted estimator only after
  Phase B has completed.

The legacy methods require a covariance view in addition to broadband EEG.  A
fixed, label-free four-band Butterworth transform is derived independently per
cached trial and fully serialized in the artifact contract.  It never pools
statistics across trials or partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import torch
from numpy.typing import NDArray
from scipy import signal
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
)

from ieee_mi.config import preprocessing_for_dataset
from ieee_mi.data import load_subject_cache, split_indices

from .cameo_net import CAMEOClassifier, CAMEOConfig
from .config import PROJECT_ROOT
from .data import make_spd_covariances
from .orbit_transport_net import OrbitTransportClassifier, OrbitTransportConfig
from .parity_fuse_net import ParityFuseClassifier, ParityFuseConfig
from .parity_net import HemiParityClassifier, ParityConfig


SCHEMA_VERSION = "deepnet-local-outer-refit-v1"
DATASET = "local_exp4"
FORMAL_SUBJECTS: tuple[int, ...] = (1, 3, 4, 5, 6, 7, 8, 10)
FORMAL_SEEDS: tuple[int, ...] = (7, 17, 27, 37, 47)
MODEL_NAMES: tuple[str, ...] = (
    "cameo",
    "hemiparity",
    "parity_fuse",
    "orbit_v3",
)
SOURCE_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (8.0, 12.0),
    (11.0, 15.0),
    (14.0, 20.0),
    (20.0, 30.0),
)
FILTER_ORDER = 4

PROCEDURE_CONFIG_SCHEMA = "deepnet-local-procedure-config-v1"
PROCEDURE_CONFIG_VERSION = 1


@dataclass(frozen=True)
class _ProcedureConfigSpec:
    relative_path: str
    size_bytes: int
    sha256: str
    stable_id: str
    config_class: type[Any]
    architecture_fields: tuple[str, ...]
    training_fields: tuple[str, ...]


# These configuration-only files contain no scores, predictions, labels,
# cohorts, or other result material. Seed and execution device are runtime
# overrides and are deliberately absent from the frozen files.
_PROCEDURE_CONFIGS: dict[str, _ProcedureConfigSpec] = {
    "cameo": _ProcedureConfigSpec(
        relative_path="configs/local_procedures/cameo_v1.json",
        size_bytes=830,
        sha256="23d07dcfc03bac8ac1e0cb076621850d5738df5a7d31fa7ea89ba42b3dfb2336",
        stable_id="architecture.cameo",
        config_class=CAMEOConfig,
        architecture_fields=(
            "dropout",
            "dynamics_channels",
            "dynamics_kernel",
            "pool_kernel",
            "pool_stride",
            "temporal_filters",
            "temporal_kernel",
        ),
        training_fields=(
            "auxiliary_weight",
            "batch_size",
            "epochs",
            "gradient_clip",
            "learning_rate",
            "min_delta",
            "mirror_penalty",
            "mixture_names",
            "patience",
            "rho_grid",
            "swap_probability",
            "weight_decay",
        ),
    ),
    "hemiparity": _ProcedureConfigSpec(
        relative_path="configs/local_procedures/hemiparity_v1.json",
        size_bytes=570,
        sha256="0978f53bbc603378052d4ea97355641fe454d250f2745593ead239df4eb463cc",
        stable_id="architecture.hemiparity",
        config_class=ParityConfig,
        architecture_fields=(
            "dropout",
            "dynamics_channels",
            "raw_rank",
            "tangent_rank",
            "temporal_filters",
        ),
        training_fields=(
            "auxiliary_weight",
            "batch_size",
            "deterministic",
            "epochs",
            "gate_balance_weight",
            "gradient_clip",
            "learning_rate",
            "min_delta",
            "patience",
            "tangent_learning_rate_scale",
            "weight_decay",
        ),
    ),
    "parity_fuse": _ProcedureConfigSpec(
        relative_path="configs/local_procedures/parity_fuse_v1.json",
        size_bytes=629,
        sha256="c5a2e84163b49f2da7c47c359418aa790f7a9d6490c7e0bc48c9aa8be3f9c6fb",
        stable_id="architecture.parity_fuse",
        config_class=ParityFuseConfig,
        architecture_fields=(
            "dynamics_channels",
            "dynamics_kernel",
            "fusion_dim",
            "gate_hidden",
            "raw_dim",
            "tangent_band_dim",
            "tangent_dim",
            "tangent_hidden",
            "temporal_filters",
            "temporal_kernel",
        ),
        training_fields=(
            "batch_size",
            "deterministic",
            "epochs",
            "gradient_clip",
            "learning_rate",
            "min_delta",
            "patience",
            "residual_penalty",
            "weight_decay",
        ),
    ),
    "orbit_v3": _ProcedureConfigSpec(
        relative_path="configs/local_procedures/orbit_v3.json",
        size_bytes=849,
        sha256="95edd8a374892e4f28a8feb4e3dfe4ffb32e415b4c3cca2d1ea4a6f89a72b456",
        stable_id="architecture.orbit_v3",
        config_class=OrbitTransportConfig,
        architecture_fields=(
            "dynamics_channels",
            "dynamics_kernel",
            "gate_hidden",
            "normalization",
            "orientation_hidden",
            "orientation_rank",
            "pool_kernel",
            "pool_stride",
            "temporal_filters",
            "temporal_kernel",
        ),
        training_fields=(
            "batch_size",
            "candidate_names",
            "deterministic",
            "epochs",
            "gradient_clip",
            "learning_rate",
            "min_delta",
            "orientation_penalty",
            "patience",
            "transport_auxiliary_weight",
            "view_auxiliary_weight",
            "weight_decay",
        ),
    ),
}

_SOURCE_FILES: tuple[str, ...] = (
    "deepnet/local_outer_refit_benchmark.py",
    "deepnet/config.py",
    "deepnet/cameo_net.py",
    "deepnet/parity_net.py",
    "deepnet/parity_fuse_net.py",
    "deepnet/orbit_transport_net.py",
    "deepnet/data.py",
    "deepnet/augment.py",
    "deepnet/spd.py",
    "ieee_mi/config.py",
    "ieee_mi/data.py",
    *(spec.relative_path for spec in _PROCEDURE_CONFIGS.values()),
)

# Only these fitted, data-dependent initialization entries may differ between
# Phase A and Phase B.  Every random/trainable remainder must hash identically.
_RESET_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "cameo": (),
    "hemiparity": ("tangent_base.weight",),
    "parity_fuse": ("tangent_anchor_weight",),
    "orbit_v3": (),
}
_CONFIG_IDENTITY_KEYS = frozenset(
    {
        "schema",
        "version",
        "model",
        "config_file",
        "config_file_bytes",
        "config_file_sha256",
        "settings_sha256",
        "config",
        "runtime_overrides",
    }
)
_IMMUTABLE_CONFIG_IDENTITY_KEYS = _CONFIG_IDENTITY_KEYS - {
    "runtime_overrides"
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


_OUTCOME_CONFIG_KEY_TOKENS = frozenset(
    {
        "accuracy",
        "auc",
        "confusion",
        "f1",
        "history",
        "histories",
        "kappa",
        "label",
        "labels",
        "loss",
        "losses",
        "metric",
        "metrics",
        "outcome",
        "outcomes",
        "participant",
        "participants",
        "performance",
        "precision",
        "prediction",
        "predictions",
        "probabilities",
        "probability",
        "recall",
        "record",
        "records",
        "result",
        "results",
        "roc",
        "score",
        "scores",
        "sensitivity",
        "specificity",
        "subject",
        "subjects",
        "summary",
        "summaries",
        "target",
        "targets",
    }
)
_FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "best_epoch",
        "best_validation_loss",
        "cohort",
        "command",
        "dataset",
        "device",
        "environment",
        "execution_environment",
        "fit_seconds",
        "fold",
        "folds",
        "repository",
        "seed",
        "seeds",
        "selected_epoch",
        "selected_epochs",
        "split",
        "splits",
        "status",
        "test_loss",
        "validation_loss",
    }
)
_ALLOWED_CONFIG_KEYS_WITH_OUTCOME_TOKENS = frozenset(
    {
        # This is a prespecified augmentation probability, not a prediction.
        "swap_probability",
    }
)


def _normalized_config_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _reject_outcome_like_config_keys(
    value: Any, *, path: str = "config"
) -> None:
    """Reject result, cohort, runtime, or outcome fields at any nesting depth."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _normalized_config_key(raw_key)
            tokens = frozenset(token for token in key.split("_") if token)
            if (
                key in _FORBIDDEN_CONFIG_KEYS
                or (
                    key not in _ALLOWED_CONFIG_KEYS_WITH_OUTCOME_TOKENS
                    and tokens.intersection(_OUTCOME_CONFIG_KEY_TOKENS)
                )
            ):
                raise ValueError(
                    f"procedure configuration contains forbidden outcome/runtime "
                    f"key {path}.{raw_key}"
                )
            _reject_outcome_like_config_keys(
                child, path=f"{path}.{raw_key}"
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_outcome_like_config_keys(
                child, path=f"{path}[{index}]"
            )


def _validate_setting_value(value: Any, *, path: str) -> None:
    if isinstance(value, bool) or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"{path} must be finite")
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_setting_value(child, path=f"{path}[{index}]")
        return
    raise ValueError(
        f"{path} must be a JSON scalar or list of JSON scalars"
    )


def _validated_config_payload(
    model_name: str,
    spec: _ProcedureConfigSpec,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _reject_outcome_like_config_keys(payload)
    expected_top = {
        "schema",
        "version",
        "model",
        "architecture",
        "training",
    }
    if set(payload) != expected_top:
        raise ValueError(
            "procedure configuration top-level fields differ from the "
            f"config-only schema: missing={sorted(expected_top - set(payload))}, "
            f"extra={sorted(set(payload) - expected_top)}"
        )
    if (
        payload["schema"] != PROCEDURE_CONFIG_SCHEMA
        or type(payload["version"]) is not int
        or payload["version"] != PROCEDURE_CONFIG_VERSION
        or payload["model"] != spec.stable_id
    ):
        raise ValueError(
            f"procedure configuration identity is invalid for {model_name}"
        )
    architecture = payload["architecture"]
    training = payload["training"]
    if not isinstance(architecture, dict) or not isinstance(training, dict):
        raise ValueError(
            "procedure architecture and training settings must be JSON objects"
        )
    expected_architecture = set(spec.architecture_fields)
    expected_training = set(spec.training_fields)
    if set(architecture) != expected_architecture:
        raise ValueError(
            f"{model_name} architecture fields differ from the frozen schema"
        )
    if set(training) != expected_training:
        raise ValueError(
            f"{model_name} training fields differ from the frozen schema"
        )
    if expected_architecture.intersection(expected_training):
        raise RuntimeError(
            f"{model_name} frozen architecture/training schemas overlap"
        )

    runtime_fields = {"seed", "device"}
    dataclass_names = {
        field.name for field in dataclass_fields(spec.config_class)
    }
    expected_dataclass_names = (
        expected_architecture | expected_training | runtime_fields
    )
    if dataclass_names != expected_dataclass_names:
        raise RuntimeError(
            f"{model_name} configuration dataclass drifted from its frozen "
            "config-only schema"
        )

    values = {**architecture, **training}
    for key, value in values.items():
        _validate_setting_value(value, path=f"config.{key}")
    return json.loads(_canonical_json(values))


def _array_sha256(values: NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _array_manifest(values: NDArray[Any]) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": _array_sha256(array),
    }


def _state_sha256(
    state: Mapping[str, torch.Tensor], *, exclude: Sequence[str] = ()
) -> str:
    excluded = set(exclude)
    unknown = excluded - set(state)
    if unknown:
        raise ValueError(f"state exclusion names do not exist: {sorted(unknown)}")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name in excluded:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}-",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                _json_safe(payload),
                temporary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _source_manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"source-manifest file is missing: {path}")
        result[relative] = _sha256_file(path)
    return result


def _validate_config_source_binding(
    config_identity: Mapping[str, Any],
    source_manifest: Mapping[str, str],
) -> None:
    """Require the loaded config file to be the same file in source identity."""

    config_file = config_identity.get("config_file")
    config_sha256 = config_identity.get("config_file_sha256")
    if (
        not isinstance(config_file, str)
        or not isinstance(config_sha256, str)
        or source_manifest.get(config_file) != config_sha256
    ):
        raise RuntimeError(
            "frozen configuration hash differs from the source manifest"
        )


def _git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            arguments,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip()

    commit = run("git", "rev-parse", "HEAD") or None
    status = run("git", "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "available": commit is not None,
        "commit": commit,
        "dirty": bool(status) if commit is not None else None,
        "status_line_count": len(status.splitlines()) if status else 0,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _environment() -> dict[str, Any]:
    distributions = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "installed_distributions": distributions,
        "installed_distributions_sha256": hashlib.sha256(
            "\n".join(distributions).encode("utf-8")
        ).hexdigest(),
    }


def _load_frozen_config(
    model_name: str, *, seed: int, device: str
) -> tuple[Any, dict[str, Any]]:
    if model_name not in _PROCEDURE_CONFIGS:
        raise ValueError(f"unknown local procedure configuration: {model_name}")
    spec = _PROCEDURE_CONFIGS[model_name]
    path = PROJECT_ROOT / spec.relative_path
    payload = _strict_json_load(path)
    frozen_values = _validated_config_payload(model_name, spec, payload)
    observed_size = path.stat().st_size
    observed_sha256 = _sha256_file(path)
    if observed_size != spec.size_bytes or observed_sha256 != spec.sha256:
        raise RuntimeError(
            f"frozen procedure configuration changed for {model_name}: "
            f"expected {spec.size_bytes} bytes/{spec.sha256}, observed "
            f"{observed_size} bytes/{observed_sha256}"
        )
    runtime_values = {
        **frozen_values,
        "seed": int(seed),
        "device": str(device),
    }
    config = spec.config_class(**runtime_values)
    if _json_safe(asdict(config)) != _json_safe(runtime_values):
        raise RuntimeError(
            f"{model_name} runtime configuration does not round-trip exactly"
        )
    return config, {
        "schema": payload["schema"],
        "version": payload["version"],
        "model": payload["model"],
        "config_file": spec.relative_path,
        "config_file_bytes": observed_size,
        "config_file_sha256": observed_sha256,
        "settings_sha256": _json_sha256(frozen_values),
        "config": frozen_values,
        "runtime_overrides": {"seed": int(seed), "device": str(device)},
    }


def _filter_designs(sfreq: float) -> tuple[NDArray[np.float64], ...]:
    if not math.isclose(float(sfreq), 128.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"local covariance view requires 128 Hz, got {sfreq}")
    return tuple(
        signal.butter(
            FILTER_ORDER,
            (low, high),
            btype="bandpass",
            fs=float(sfreq),
            output="sos",
        ).astype(np.float64, copy=False)
        for low, high in SOURCE_BANDS_HZ
    )


def covariance_view_contract(sfreq: float = 128.0) -> dict[str, Any]:
    designs = _filter_designs(sfreq)
    return {
        "input": "unaltered ieee_mi local_exp4 v2 cached x",
        "fit_scope": (
            "independent deterministic transform per trial; "
            "no labels or cross-trial statistics"
        ),
        "bands_hz": [list(band) for band in SOURCE_BANDS_HZ],
        "filter": {
            "family": "scipy.signal.butter",
            "order": FILTER_ORDER,
            "representation": "second-order sections",
            "application": "scipy.signal.sosfiltfilt along each cached trial's time axis",
            "phase": "zero",
            "padtype": "odd",
            "padlen": "scipy default",
            "sfreq_hz": float(sfreq),
            "sos": [design.tolist() for design in designs],
            "sos_sha256": [_array_sha256(design) for design in designs],
        },
        "covariance": {
            "implementation": "deepnet.data.make_spd_covariances",
            "demean": True,
            "shrinkage": 1e-3,
            "shrinkage_method": "fixed",
            "dtype": "float32",
        },
    }


def derive_covariance_view(
    raw_epochs: NDArray[np.floating], *, sfreq: float = 128.0
) -> NDArray[np.float32]:
    """Return the frozen four-band SPD view without fitting any statistic."""

    raw = np.asarray(raw_epochs, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[-1] != 256:
        raise ValueError("local raw cache must have shape (trials, channels, 256)")
    if not np.all(np.isfinite(raw)):
        raise ValueError("local raw cache contains non-finite samples")
    band_epochs = []
    for design in _filter_designs(sfreq):
        filtered = signal.sosfiltfilt(
            design,
            raw.astype(np.float64, copy=False),
            axis=-1,
            padtype="odd",
            padlen=None,
        )
        band_epochs.append(filtered.astype(np.float32, copy=False))
    filter_bank = np.ascontiguousarray(np.stack(band_epochs, axis=1))
    covariance = make_spd_covariances(
        filter_bank,
        shrinkage=1e-3,
        method="fixed",
        demean=True,
        dtype=np.float32,
    )
    return np.ascontiguousarray(covariance, dtype=np.float32)


def _split_contract(
    data: Mapping[str, Any], *, subject: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
    train, validation, test = split_indices(
        DATASET,
        data["y"],
        data["sessions"],
        data["runs"],
        fold=0,
        subject=int(subject),
    )
    train = np.asarray(train, dtype=np.int64)
    validation = np.asarray(validation, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    source = np.sort(np.concatenate((train, validation)))
    if (len(train), len(validation), len(test), len(source)) != (120, 60, 60, 180):
        raise RuntimeError("local outer split counts changed")
    if set(train) & set(validation) or set(source) & set(test):
        raise RuntimeError("local outer split rows overlap")
    if len(np.unique(np.concatenate((source, test)))) != 240:
        raise RuntimeError("local outer split does not partition all 240 rows")
    for name, rows in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        counts = np.bincount(np.asarray(data["y"])[rows], minlength=2).tolist()
        expected = [60, 60] if name == "train" else [30, 30]
        if counts != expected:
            raise RuntimeError(f"{name} class counts changed: {counts}")
    contract = {
        "phase_a_train": {
            "count": len(train),
            "runs": sorted(set(np.asarray(data["runs"])[train].astype(str).tolist())),
            "rows_sha256": _array_sha256(train),
        },
        "phase_a_validation": {
            "count": len(validation),
            "runs": sorted(
                set(np.asarray(data["runs"])[validation].astype(str).tolist())
            ),
            "rows_sha256": _array_sha256(validation),
        },
        "phase_b_source": {
            "count": len(source),
            "runs": sorted(set(np.asarray(data["runs"])[source].astype(str).tolist())),
            "rows_sha256": _array_sha256(source),
        },
        "prediction_only_test": {
            "count": len(test),
            "runs": sorted(set(np.asarray(data["runs"])[test].astype(str).tolist())),
            "rows_sha256": _array_sha256(test),
        },
    }
    return train, validation, test, contract


def _metrics(labels: NDArray[np.integer], probabilities: NDArray[np.floating]) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.shape != (len(y), 2) or not np.all(np.isfinite(probability)):
        raise ValueError("binary probability matrix is malformed")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("binary probabilities lie outside [0, 1]")
    row_sums = probability.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("binary probabilities do not sum to one")
    # Normalize harmless float32 sigmoid roundoff before sklearn's strict
    # probability checks without changing the predicted class.
    probability = probability / row_sums
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "cohen_kappa": float(cohen_kappa_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability[:, 1])),
        "cross_entropy": float(log_loss(y, probability, labels=[0, 1])),
    }


def _fitted_preprocessing_manifest(classifier: Any) -> dict[str, Any]:
    anchor = classifier.anchor_
    arrays = {
        "raw_mean": np.asarray(classifier.raw_mean_),
        "raw_std": np.asarray(classifier.raw_std_),
        "anchor_log_reference": anchor.log_reference_.detach().cpu().numpy(),
        "anchor_scaler_mean": np.asarray(anchor.scaler_.mean_),
        "anchor_scaler_scale": np.asarray(anchor.scaler_.scale_),
        "anchor_logistic_coef": np.asarray(anchor.model_.coef_),
        "anchor_logistic_intercept": np.asarray(anchor.model_.intercept_),
    }
    return {name: _array_manifest(value) for name, value in arrays.items()}


def _selection_fit(
    model_name: str,
    config: Any,
    *,
    raw_train: NDArray[np.float32],
    covariance_train: NDArray[np.float32],
    labels_train: NDArray[np.int64],
    raw_validation: NDArray[np.float32],
    covariance_validation: NDArray[np.float32],
    labels_validation: NDArray[np.int64],
    channels: tuple[str, ...],
) -> tuple[Any, dict[str, Any]]:
    if model_name == "cameo":
        classifier = CAMEOClassifier(config).fit(
            raw_train,
            covariance_train,
            labels_train,
            raw_validation,
            covariance_validation,
            labels_validation,
            channels=channels,
        )
        decision = {
            "selected_mixture": classifier.selected_mixture_,
            "selected_rho": float(classifier.selected_rho_),
        }
    elif model_name == "hemiparity":
        classifier = HemiParityClassifier(config).fit(
            raw_train,
            covariance_train,
            labels_train,
            raw_validation,
            covariance_validation,
            labels_validation,
            channels=channels,
        )
        decision = {}
    elif model_name == "parity_fuse":
        classifier = ParityFuseClassifier(config).fit(
            raw_train,
            covariance_train,
            labels_train,
            raw_validation,
            covariance_validation,
            labels_validation,
            channels=channels,
        )
        decision = {}
    elif model_name == "orbit_v3":
        classifier = OrbitTransportClassifier(config).fit(
            raw_train,
            covariance_train,
            labels_train,
            raw_validation,
            covariance_validation,
            labels_validation,
            channels=channels,
        )
        decision = {"selected_candidate": classifier.selected_candidate_}
    else:  # pragma: no cover - CLI and contract validation reject this
        raise ValueError(model_name)

    best_epoch = int(classifier.best_epoch_)
    selected_epochs = best_epoch + 1
    if model_name == "parity_fuse":
        if not 0 <= selected_epochs <= config.epochs:
            raise RuntimeError("PARITY-Fuse selected an invalid refit duration")
    elif not 1 <= selected_epochs <= config.epochs:
        raise RuntimeError(f"{model_name} selected an invalid refit duration")
    validation_probability = classifier.predict_proba(
        raw_validation, covariance_validation
    )
    detail = {
        "best_epoch_zero_based": best_epoch,
        "selected_epoch_count": selected_epochs,
        "epochs_run": int(classifier.epochs_run_),
        "best_validation_loss": float(classifier.best_validation_loss_),
        "best_validation_balanced_accuracy": (
            float(classifier.best_validation_balanced_accuracy_)
            if hasattr(classifier, "best_validation_balanced_accuracy_")
            else None
        ),
        "selection_decision": decision,
        "history": classifier.history_,
        "fit_seconds": float(classifier.train_seconds_),
        "validation_metrics": _metrics(labels_validation, validation_probability),
        "validation_probabilities": validation_probability.tolist(),
        "fitted_preprocessing": _fitted_preprocessing_manifest(classifier),
    }
    if hasattr(classifier, "selection_trace_"):
        detail["selection_trace"] = classifier.selection_trace_
    return classifier, detail


def _refit(
    model_name: str,
    config: Any,
    selection: Any,
    selection_detail: Mapping[str, Any],
    *,
    raw_source: NDArray[np.float32],
    covariance_source: NDArray[np.float32],
    labels_source: NDArray[np.int64],
    channels: tuple[str, ...],
) -> tuple[Any, dict[str, Any]]:
    epochs = int(selection_detail["selected_epoch_count"])
    decision = dict(selection_detail["selection_decision"])
    if model_name == "cameo":
        classifier = CAMEOClassifier(config).fit_fixed_epochs(
            raw_source,
            covariance_source,
            labels_source,
            epochs=epochs,
            selected_mixture=str(decision["selected_mixture"]),
            selected_rho=float(decision["selected_rho"]),
            channels=channels,
        )
    elif model_name == "hemiparity":
        classifier = HemiParityClassifier(config).fit_fixed_epochs(
            raw_source,
            covariance_source,
            labels_source,
            epochs=epochs,
            channels=channels,
        )
    elif model_name == "parity_fuse":
        classifier = ParityFuseClassifier(config).fit_fixed_epochs(
            raw_source,
            covariance_source,
            labels_source,
            epochs=epochs,
            channels=channels,
        )
    elif model_name == "orbit_v3":
        classifier = OrbitTransportClassifier(config).fit_fixed_epochs(
            raw_source,
            covariance_source,
            labels_source,
            epochs=epochs,
            selected_candidate=str(decision["selected_candidate"]),
            channels=channels,
        )
    else:  # pragma: no cover
        raise ValueError(model_name)

    if int(classifier.epochs_run_) != epochs or len(classifier.history_) != epochs:
        raise RuntimeError("fixed refit did not run the validation-selected duration")
    exclusions = _RESET_EXCLUSIONS[model_name]
    selection_full = _state_sha256(selection.initial_model_state_)
    refit_full = _state_sha256(classifier.initial_model_state_)
    selection_reset = _state_sha256(
        selection.initial_model_state_, exclude=exclusions
    )
    refit_reset = _state_sha256(
        classifier.initial_model_state_, exclude=exclusions
    )
    if selection_reset != refit_reset:
        raise RuntimeError(
            f"{model_name} selection/refit random initializations differ"
        )
    detail = {
        "epoch_count": epochs,
        "scheduler_horizon_epochs": int(config.epochs),
        "history": classifier.history_,
        "fit_seconds": float(classifier.train_seconds_),
        "selection_start_full_sha256": selection_full,
        "refit_start_full_sha256": refit_full,
        "reset_hash_exclusions": list(exclusions),
        "selection_start_reset_sha256": selection_reset,
        "refit_start_reset_sha256": refit_reset,
        "reset_verified": True,
        "data_dependent_initialization_changed": selection_full != refit_full,
        "fitted_preprocessing": _fitted_preprocessing_manifest(classifier),
        "model_state_sha256": _state_sha256(classifier.model_.state_dict()),
        "parameter_count": int(classifier.param_count_),
        "trainable_parameter_count": int(
            getattr(classifier, "trainable_param_count_", classifier.param_count_)
        ),
        "determinism_state": {
            "algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "warn_only": bool(
                getattr(
                    torch,
                    "is_deterministic_algorithms_warn_only_enabled",
                    lambda: False,
                )()
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
    }
    return classifier, detail


def _prediction_trace(
    data: Mapping[str, Any],
    rows: NDArray[np.int64],
    probabilities: NDArray[np.float64],
) -> list[dict[str, Any]]:
    labels = np.asarray(data["y"], dtype=np.int64)[rows]
    sessions = np.asarray(data["sessions"]).astype(str)[rows]
    runs = np.asarray(data["runs"]).astype(str)[rows]
    return [
        {
            "row": int(row),
            "session": str(session),
            "run": str(run),
            "label": int(label),
            "probability_left": float(probability[0]),
            "probability_right": float(probability[1]),
        }
        for row, session, run, label, probability in zip(
            rows, sessions, runs, labels, probabilities, strict=True
        )
    ]


def _diagnostics(
    classifier: Any,
    raw_test: NDArray[np.float32],
    covariance_test: NDArray[np.float32],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if hasattr(classifier, "max_equivariance_error"):
        result["max_equivariance_error"] = float(
            classifier.max_equivariance_error(raw_test, covariance_test)
        )
    if hasattr(classifier, "max_gate_invariance_error"):
        result["max_gate_invariance_error"] = float(
            classifier.max_gate_invariance_error(raw_test, covariance_test)
        )
    return result


def run_one(
    *,
    model_name: str,
    subject: int,
    seed: int,
    device: str,
    data: Mapping[str, Any],
    covariance: NDArray[np.float32],
    split: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]],
    split_contract: Mapping[str, Any],
    config_identity: Mapping[str, Any],
) -> dict[str, Any]:
    train, validation, test = split
    source = np.sort(np.concatenate((train, validation)))
    raw = np.ascontiguousarray(data["x"], dtype=np.float32)
    labels = np.asarray(data["y"], dtype=np.int64)
    channels = tuple(str(value) for value in np.asarray(data["channel_names"]).tolist())
    if raw.shape != (240, 15, 256) or covariance.shape != (240, 4, 15, 15):
        raise RuntimeError("local v2 raw/covariance shapes changed")

    config, runtime_config_identity = _load_frozen_config(
        model_name, seed=seed, device=device
    )
    if (
        set(config_identity) != _CONFIG_IDENTITY_KEYS
        or any(
            runtime_config_identity[key] != config_identity[key]
            for key in _IMMUTABLE_CONFIG_IDENTITY_KEYS
        )
    ):
        raise RuntimeError("frozen configuration identity changed during the run")

    selection, selection_detail = _selection_fit(
        model_name,
        config,
        raw_train=raw[train],
        covariance_train=covariance[train],
        labels_train=labels[train],
        raw_validation=raw[validation],
        covariance_validation=covariance[validation],
        labels_validation=labels[validation],
        channels=channels,
    )
    refitted, refit_detail = _refit(
        model_name,
        config,
        selection,
        selection_detail,
        raw_source=raw[source],
        covariance_source=covariance[source],
        labels_source=labels[source],
        channels=channels,
    )

    # Prediction-only rows first enter an estimator API here, after Phase B.
    test_probability = np.asarray(
        refitted.predict_proba(raw[test], covariance[test]), dtype=np.float64
    )
    test_labels = labels[test]
    return {
        "model": model_name,
        "subject": int(subject),
        "seed": int(seed),
        "runtime_config": asdict(config),
        "split": dict(split_contract),
        "phase_a": selection_detail,
        "phase_b": refit_detail,
        "prediction_only_test": {
            "metrics": _metrics(test_labels, test_probability),
            "predictions": _prediction_trace(data, test, test_probability),
            "diagnostics": _diagnostics(
                refitted, raw[test], covariance[test]
            ),
            "labels_observed_after_refit": True,
        },
    }


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_subject: dict[int, list[float]] = {}
    for row in records:
        by_subject.setdefault(int(row["subject"]), []).append(
            float(row["prediction_only_test"]["metrics"]["balanced_accuracy"])
        )
    subject_values = {
        subject: float(np.mean(values))
        for subject, values in sorted(by_subject.items())
    }
    values = np.asarray(list(subject_values.values()), dtype=np.float64)
    return {
        "balanced_accuracy_mean": float(values.mean()) if len(values) else None,
        "balanced_accuracy_std": (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ),
        "subject_balanced_accuracy": {
            str(subject): value for subject, value in subject_values.items()
        },
        "n_participants": len(subject_values),
        "n_records": len(records),
        "aggregation": "mean seeds within participant, then equal-weight participant mean",
    }


def _contract(
    *,
    model_name: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    device: str,
    config_identity: Mapping[str, Any],
    source_manifest: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    preprocessing = preprocessing_for_dataset(DATASET)
    formal = tuple(subjects) == FORMAL_SUBJECTS and tuple(seeds) == FORMAL_SEEDS
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "model": model_name,
        "subjects": list(subjects),
        "seeds": list(seeds),
        "device": str(device),
        "formal_complete_dimensions": formal,
        "evidence_scope": (
            "post_selection_local_development_all_four_recordings_previously_opened"
            if formal
            else "exploratory_incomplete_screen"
        ),
        "outer_protocol": {
            "phase_a": "recordings 1-2 fit; recording 3 selects duration and route only",
            "phase_b": (
                "fresh reset; source preprocessing refit on recordings 1-3; "
                "fixed selected duration"
            ),
            "test": "recording 4 prediction only after phase B",
            "fold": 0,
        },
        "preprocessing": preprocessing,
        "covariance_view": covariance_view_contract(
            float(preprocessing["sfreq_hz"])
        ),
        "frozen_model_config": dict(config_identity),
        "source_sha256": dict(source_manifest),
        # Environment identity is contractual so a resumed artifact cannot
        # silently mix records from different numerical stacks or GPUs.
        "execution_environment": dict(environment),
        "execution_environment_sha256": _json_sha256(environment),
    }


def _load_or_initialize(
    output: Path,
    *,
    contract: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    contract_hash = _json_sha256(contract)
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        payload = _strict_json_load(output)
        if payload.get("contract_sha256") != contract_hash:
            raise ValueError("cannot resume output under a different experiment contract")
        if payload.get("contract") != _json_safe(contract):
            raise ValueError("resume contract hash matched but contract content differs")
        _validate_resumable_payload(payload, contract=contract)
        return payload
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": None,
        "status": "running",
        "contract": _json_safe(contract),
        "contract_sha256": contract_hash,
        "command": list(sys.argv),
        "repository": _git_state(),
        "environment": dict(
            contract.get("execution_environment", _environment())
        ),
        "records": [],
        "summary": {},
        "completion": {
            "expected_records": len(contract["subjects"]) * len(contract["seeds"]),
            "actual_records": 0,
            "complete": False,
        },
    }


def _validate_resumable_payload(
    payload: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> None:
    """Reject corrupt or internally inconsistent records before resume."""

    expected = {
        (int(subject), int(seed))
        for subject in contract["subjects"]
        for seed in contract["seeds"]
    }
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("resumed artifact records are not a list")
    seen: set[tuple[int, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("resumed artifact contains a non-object record")
        key = (int(record.get("subject", -1)), int(record.get("seed", -1)))
        if key not in expected or key in seen:
            raise ValueError(f"resumed artifact has invalid/duplicate record {key}")
        seen.add(key)
        if record.get("model") != contract["model"]:
            raise ValueError(f"resumed record {key} has the wrong model")
        split = record.get("split", {})
        counts = (
            split.get("phase_a_train", {}).get("count"),
            split.get("phase_a_validation", {}).get("count"),
            split.get("phase_b_source", {}).get("count"),
            split.get("prediction_only_test", {}).get("count"),
        )
        if counts != (120, 60, 180, 60):
            raise ValueError(f"resumed record {key} has invalid split counts")
        reset = record.get("phase_b", {})
        if reset.get("reset_verified") is not True:
            raise ValueError(f"resumed record {key} has no verified reset")
        selected_epochs = int(
            record.get("phase_a", {}).get("selected_epoch_count", -1)
        )
        if int(reset.get("epoch_count", -2)) != selected_epochs:
            raise ValueError(f"resumed record {key} has inconsistent refit duration")
        prediction = record.get("prediction_only_test", {})
        trace = prediction.get("predictions")
        if not isinstance(trace, list) or len(trace) != 60:
            raise ValueError(f"resumed record {key} must contain 60 test predictions")
        rows: list[int] = []
        labels: list[int] = []
        probabilities: list[list[float]] = []
        for row in trace:
            rows.append(int(row["row"]))
            labels.append(int(row["label"]))
            probabilities.append(
                [float(row["probability_left"]), float(row["probability_right"])]
            )
        if len(set(rows)) != 60 or np.bincount(labels, minlength=2).tolist() != [30, 30]:
            raise ValueError(f"resumed record {key} has invalid test rows/labels")
        recomputed = _metrics(
            np.asarray(labels, dtype=np.int64),
            np.asarray(probabilities, dtype=np.float64),
        )
        if _json_sha256(recomputed) != _json_sha256(prediction.get("metrics")):
            raise ValueError(f"resumed record {key} has stale test metrics")
        for digest_name in ("cache_identity_sha256",):
            digest = record.get(digest_name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"resumed record {key} has invalid {digest_name}")

    completion = payload.get("completion", {})
    if int(completion.get("expected_records", -1)) != len(expected):
        raise ValueError("resumed artifact has the wrong expected record count")
    if int(completion.get("actual_records", -1)) != len(records):
        raise ValueError("resumed artifact has a stale actual record count")
    is_complete = len(seen) == len(expected)
    if bool(completion.get("complete")) != is_complete:
        raise ValueError("resumed artifact has a stale completion flag")
    expected_status = "complete" if is_complete else "running"
    if payload.get("status") != expected_status:
        raise ValueError("resumed artifact has a stale status")
    if _json_sha256(payload.get("summary")) != _json_sha256(_summarize(records)):
        raise ValueError("resumed artifact has a stale summary")


def _save_payload(path: Path, payload: dict[str, Any], *, complete: bool) -> None:
    payload["updated_at"] = _utc_now()
    payload["summary"] = _summarize(payload["records"])
    expected = int(payload["completion"]["expected_records"])
    actual = len(payload["records"])
    if len({(int(row["subject"]), int(row["seed"])) for row in payload["records"]}) != actual:
        raise RuntimeError("artifact contains duplicate participant/seed records")
    payload["completion"].update(
        {"actual_records": actual, "complete": actual == expected}
    )
    if complete and actual != expected:
        raise RuntimeError("refusing to finalize an incomplete artifact")
    payload["status"] = "complete" if complete else "running"
    _atomic_json(path, payload)


def run_benchmark(
    *,
    model_name: str,
    subjects: Sequence[int],
    seeds: Sequence[int],
    cache_root: Path,
    device: str,
    output: Path,
    resume: bool = True,
) -> dict[str, Any]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    subjects = tuple(int(value) for value in subjects)
    seeds = tuple(int(value) for value in seeds)
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("subjects must be nonempty and unique")
    if not set(subjects).issubset(set(FORMAL_SUBJECTS)):
        raise ValueError("subjects contain a participant outside local_exp4")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty and unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)

    _base_config, config_identity = _load_frozen_config(
        model_name, seed=seeds[0], device=device
    )
    del _base_config
    source_manifest = _source_manifest()
    _validate_config_source_binding(config_identity, source_manifest)
    environment = _environment()
    contract = _contract(
        model_name=model_name,
        subjects=subjects,
        seeds=seeds,
        device=device,
        config_identity=config_identity,
        source_manifest=source_manifest,
        environment=environment,
    )
    payload = _load_or_initialize(output, contract=contract, resume=resume)
    completed = {
        (int(row["subject"]), int(row["seed"])) for row in payload["records"]
    }
    preprocessing = preprocessing_for_dataset(DATASET)
    sfreq = float(preprocessing["sfreq_hz"])

    for subject in subjects:
        data = load_subject_cache(DATASET, subject, cache_root=cache_root)
        raw = np.asarray(data["x"], dtype=np.float32)
        covariance = derive_covariance_view(raw, sfreq=sfreq)
        train, validation, test, split_contract = _split_contract(
            data, subject=subject
        )
        cache_identity = _json_safe(data["identity"])
        cache_identity_sha256 = _json_sha256(cache_identity)
        covariance_manifest = _array_manifest(covariance)

        for previous in payload["records"]:
            if int(previous["subject"]) != subject:
                continue
            if previous.get("cache_identity_sha256") != cache_identity_sha256:
                raise RuntimeError("current cache differs from a resumed record")
            if previous.get("covariance_view") != covariance_manifest:
                raise RuntimeError("derived covariance view differs from resumed record")

        for seed in seeds:
            if (subject, seed) in completed:
                continue
            record = run_one(
                model_name=model_name,
                subject=subject,
                seed=seed,
                device=device,
                data=data,
                covariance=covariance,
                split=(train, validation, test),
                split_contract=split_contract,
                config_identity=config_identity,
            )
            record["cache_identity"] = cache_identity
            record["cache_identity_sha256"] = cache_identity_sha256
            record["raw_cache_array"] = _array_manifest(raw)
            record["covariance_view"] = covariance_manifest
            payload["records"].append(record)
            completed.add((subject, seed))
            _save_payload(output, payload, complete=False)
            score = record["prediction_only_test"]["metrics"]["balanced_accuracy"]
            print(
                f"{model_name} local_exp4 S{subject} seed={seed}: "
                f"bacc={100.0 * score:.2f}%",
                flush=True,
            )

    _save_payload(output, payload, complete=True)
    return payload


def _parse_ints(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            low, high = token.split("-", 1)
            result.extend(range(int(low), int(high) + 1))
        else:
            result.append(int(token))
    return tuple(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--subjects", default=",".join(map(str, FORMAL_SUBJECTS)))
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_SEEDS)))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    payload = run_benchmark(
        model_name=args.model,
        subjects=_parse_ints(args.subjects),
        seeds=_parse_ints(args.seeds),
        cache_root=args.cache_root,
        device=args.device,
        output=args.output,
        resume=not args.no_resume,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DATASET",
    "FORMAL_SEEDS",
    "FORMAL_SUBJECTS",
    "MODEL_NAMES",
    "PROCEDURE_CONFIG_SCHEMA",
    "PROCEDURE_CONFIG_VERSION",
    "SCHEMA_VERSION",
    "covariance_view_contract",
    "derive_covariance_view",
    "run_benchmark",
    "run_one",
]
