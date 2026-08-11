"""Outcome-free harmonized-v2 adapter for the frozen HemiQ-Field architecture.

This module does not read HemiQ's historical result artifacts at runtime.  Its
configuration is copied into a score-free file before execution and changes
only the sampling contract (125 Hz / 251 samples to 128 Hz / 320 samples) plus
the per-job device and seed.  The historical confirmation remains a separate,
read-only result.

The adapter exposes two explicit fitting phases:

* :meth:`HemiQHarmonizedV2Classifier.fit_selection` selects a duration by
  validation binary cross-entropy, including the zero-epoch initial state.
* :meth:`HemiQHarmonizedV2Classifier.fit_fixed_epochs` reconstructs the same
  seeded neural initialization, refits preprocessing on train+validation, and
  optimizes for exactly the selected duration.

The tangent classifier is a source-only training teacher.  It is never stored
as an inference path and its coefficient always follows the original 240-epoch
decay horizon, even when the refit duration is shorter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from deepnet.cameo_net import FrozenTangentAnchor, _mirror_covariances
from deepnet.data import make_spd_covariances
from deepnet.hemi_q_field_net import (
    HEMI_Q_CHANNELS,
    HemiQFieldConfig,
    HemiQFieldNet,
)


CONFIG_SCHEMA = "ieee-mi-hemiq-harmonized-v2-config-v1"
VIEW_SCHEMA = "ieee-mi-hemiq-harmonized-v2-views-v1"
CONFIG_RELATIVE_PATH = "configs/hemiq/hemiq_field_harmonized_v2.json"
CONFIG_SIZE_BYTES = 1054
CONFIG_SHA256 = "0474b53b06c1f4bc56659f0cc75ff3f908aac67306652714f533c83289af128f"
PARENT_MANIFEST_RELATIVE_PATH = (
    "deepnet/results/hemi_q/hemi_q_final_frozen_manifest.json"
)
PARENT_MANIFEST_SIZE_BYTES = 6688
PARENT_MANIFEST_SHA256 = (
    "2bbe28d4415a74cb007328579cf8153585ceaaa28219246aecd0622398987401"
)
FILTER_ORDER = 4
BROADBAND_HZ: tuple[float, float] = (8.0, 30.0)
TEACHER_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (8.0, 12.0),
    (11.0, 15.0),
    (14.0, 20.0),
    (20.0, 30.0),
)
EXPECTED_SFREQ = 128.0
EXPECTED_N_TIMES = 320

_PARENT_FROZEN_CONFIG: Mapping[str, Any] = {
    "bandwidth_high": 8.0,
    "bandwidth_low": 1.5,
    "batch_size": 64,
    "deterministic": True,
    "device": "cuda",
    "epochs": 240,
    "frequency_high": 30.0,
    "frequency_low": 8.0,
    "gradient_clip": 5.0,
    "kernel_size": 63,
    "learning_rate": 0.0007,
    "local_stride": 10,
    "local_window": 25,
    "min_delta": 0.0001,
    "n_filters": 12,
    "n_times": 251,
    "patience": 40,
    "seed": 7,
    "sfreq": 125.0,
    "spectral_epsilon": 1e-5,
    "teacher_fraction": 0.5,
    "teacher_regularization": 1.0,
    "teacher_weight": 0.25,
    "temporal_stride": 2,
    "weight_decay": 0.0005,
    "width": 32,
}


class HemiQV2Error(RuntimeError):
    """Raised when the frozen harmonized-v2 contract is violated."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: NDArray[Any]) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_bytes(list(array.shape)))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _array_manifest(value: NDArray[Any]) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": _array_sha256(array),
    }


def _read_unique_regular_bytes(path: Path) -> bytes:
    """Read one immutable regular file without following aliases or links."""

    absolute = Path(os.path.abspath(path))
    if absolute.name in {"", ".", ".."}:
        raise HemiQV2Error(f"configuration path has no regular-file leaf: {absolute}")

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_descriptor = os.open(absolute.anchor, directory_flags)
    except OSError as error:
        raise HemiQV2Error(
            f"configuration root cannot be opened safely: {absolute.anchor}"
        ) from error
    try:
        walked = Path(absolute.anchor)
        for component in absolute.parts[1:-1]:
            walked /= component
            try:
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                raise HemiQV2Error(
                    f"configuration ancestor is unsafe: {walked}"
                ) from error
            child_stat = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child_descriptor)
                raise HemiQV2Error(
                    f"configuration ancestor is unsafe: {walked}"
                )
            os.close(parent_descriptor)
            parent_descriptor = child_descriptor

        try:
            path_stat = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise HemiQV2Error(
                f"configuration cannot be inspected safely: {absolute}"
            ) from error
        if not (
            path_stat.st_nlink == 1
            and path_stat.st_size > 0
            and stat.S_ISREG(path_stat.st_mode)
        ):
            raise HemiQV2Error(f"configuration is not a unique regular file: {absolute}")

        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            descriptor = os.open(
                absolute.name,
                file_flags,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise HemiQV2Error(
                f"configuration cannot be opened without following a link: {absolute}"
            ) from error
        before = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            before.st_nlink != 1
            or before.st_size <= 0
            or not stat.S_ISREG(before.st_mode)
            or any(
                getattr(before, name) != getattr(path_stat, name)
                for name in stable_fields
            )
        ):
            os.close(descriptor)
            raise HemiQV2Error(
                f"configuration changed while it was being opened: {absolute}"
            )
        try:
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                try:
                    chunk = os.read(descriptor, min(1024 * 1024, remaining))
                except InterruptedError:
                    continue
                if not chunk:
                    raise HemiQV2Error(
                        f"configuration was truncated while reading: {absolute}"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            try:
                leaf = os.stat(
                    absolute.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise HemiQV2Error(
                    f"configuration changed while reading: {absolute}"
                ) from error
            if any(
                getattr(before, name) != getattr(observed, name)
                for observed in (after, leaf)
                for name in stable_fields
            ):
                raise HemiQV2Error(
                    f"configuration changed while reading: {absolute}"
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


_ARCHITECTURE_FIELDS = frozenset(
    {
        "bandwidth_high",
        "bandwidth_low",
        "frequency_high",
        "frequency_low",
        "kernel_size",
        "local_stride",
        "local_window",
        "n_filters",
        "spectral_epsilon",
        "temporal_stride",
        "width",
    }
)
_TRAINING_FIELDS = frozenset(
    {
        "batch_size",
        "deterministic",
        "epochs",
        "gradient_clip",
        "learning_rate",
        "min_delta",
        "patience",
        "teacher_fraction",
        "teacher_regularization",
        "teacher_weight",
        "weight_decay",
    }
)


def load_frozen_config(
    project_root: Path,
    *,
    seed: int,
    device: str,
) -> tuple[HemiQFieldConfig, dict[str, Any]]:
    """Load and exactly validate the score-free HemiQ-v2 configuration."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    root = project_root.absolute()
    config_path = root / CONFIG_RELATIVE_PATH
    payload = _read_unique_regular_bytes(config_path)
    if len(payload) != CONFIG_SIZE_BYTES or _sha256_bytes(payload) != CONFIG_SHA256:
        raise HemiQV2Error("HemiQ-v2 configuration file identity differs")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HemiQV2Error("HemiQ-v2 configuration is not valid JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "architecture",
        "input_adaptation",
        "model",
        "parent_frozen_manifest",
        "runtime_overrides",
        "schema",
        "training",
        "version",
    }:
        raise HemiQV2Error("HemiQ-v2 configuration schema is not exact")
    architecture = document["architecture"]
    training = document["training"]
    adaptation = document["input_adaptation"]
    parent = document["parent_frozen_manifest"]
    if (
        document["schema"] != CONFIG_SCHEMA
        or document["version"] != 1
        or document["model"] != "architecture.hemi_q_field"
        or document["runtime_overrides"] != ["device", "seed"]
        or not isinstance(architecture, dict)
        or set(architecture) != _ARCHITECTURE_FIELDS
        or not isinstance(training, dict)
        or set(training) != _TRAINING_FIELDS
        or adaptation != {"n_times": EXPECTED_N_TIMES, "sfreq": EXPECTED_SFREQ}
        or not isinstance(parent, dict)
        or set(parent) != {"path", "sha256", "size_bytes"}
        or parent["path"] != PARENT_MANIFEST_RELATIVE_PATH
        or parent["sha256"] != PARENT_MANIFEST_SHA256
        or parent["size_bytes"] != PARENT_MANIFEST_SIZE_BYTES
    ):
        raise HemiQV2Error("HemiQ-v2 configuration contract drifted")

    settings = {
        **architecture,
        **training,
        **adaptation,
        "seed": seed,
        "device": device,
    }
    config = HemiQFieldConfig(**settings)
    changed = {
        key
        for key in set(_PARENT_FROZEN_CONFIG) | set(asdict(config))
        if _PARENT_FROZEN_CONFIG.get(key) != asdict(config).get(key)
    }
    allowed_changes = {"device", "n_times", "seed", "sfreq"}
    required_adaptation = {"n_times", "sfreq"}
    if (
        not required_adaptation.issubset(changed)
        or not changed.issubset(allowed_changes)
    ):
        raise HemiQV2Error(
            f"HemiQ-v2 changed fields outside its declared adaptation: {sorted(changed)}"
        )
    identity = {
        "schema": CONFIG_SCHEMA,
        "config_file": CONFIG_RELATIVE_PATH,
        "config_file_bytes": len(payload),
        "config_file_sha256": _sha256_bytes(payload),
        "parent_frozen_manifest": copy.deepcopy(parent),
        "immutable_settings_sha256": _sha256_bytes(
            _canonical_bytes(
                {
                    "architecture": architecture,
                    "input_adaptation": adaptation,
                    "training": training,
                }
            )
        ),
        "runtime_overrides": {"device": device, "seed": seed},
    }
    return config, identity


def _filter_design(band: tuple[float, float], sfreq: float) -> np.ndarray:
    from scipy import signal

    return signal.butter(
        FILTER_ORDER,
        band,
        btype="bandpass",
        fs=sfreq,
        output="sos",
    ).astype(np.float64, copy=False)


def hemiq_view_contract(sfreq: float = EXPECTED_SFREQ) -> dict[str, Any]:
    """Return the immutable, label-free view derivation contract."""

    if not math.isclose(sfreq, EXPECTED_SFREQ, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("HemiQ-v2 requires exactly 128 Hz")
    broadband = _filter_design(BROADBAND_HZ, sfreq)
    teachers = tuple(_filter_design(band, sfreq) for band in TEACHER_BANDS_HZ)
    return {
        "schema": VIEW_SCHEMA,
        "input": "unaltered harmonized ieee-mi-cache-v2 BNCI2014-004 trial",
        "channels": list(HEMI_Q_CHANNELS),
        "broadband_hz": list(BROADBAND_HZ),
        "teacher_bands_hz": [list(value) for value in TEACHER_BANDS_HZ],
        "filter": {
            "family": "scipy.signal.butter",
            "order": FILTER_ORDER,
            "representation": "second-order sections",
            "application": "scipy.signal.sosfiltfilt on each trial time axis",
            "phase": "zero",
            "padtype": "odd",
            "padlen": "scipy default",
            "sfreq_hz": sfreq,
            "broadband_sos_sha256": _array_sha256(broadband),
            "teacher_sos_sha256": [_array_sha256(value) for value in teachers],
            "post_filter_trial_demean": True,
        },
        "covariance": {
            "implementation": "deepnet.data.make_spd_covariances",
            "demean": True,
            "shrinkage": 1e-3,
            "shrinkage_method": "fixed",
            "dtype": "float32",
            "positivity_check": "strict after float32 conversion",
        },
    }


def _filter_and_demean(raw: np.ndarray, design: np.ndarray) -> np.ndarray:
    from scipy import signal

    filtered = signal.sosfiltfilt(
        design,
        raw.astype(np.float64, copy=False),
        axis=-1,
        padtype="odd",
        padlen=None,
    )
    filtered -= filtered.mean(axis=-1, keepdims=True)
    output = np.ascontiguousarray(filtered, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise HemiQV2Error("HemiQ-v2 filter produced a non-finite sample")
    return output


def _verify_float32_spd(covariances: np.ndarray) -> None:
    if (
        covariances.dtype != np.float32
        or covariances.ndim != 4
        or covariances.shape[1:] != (len(TEACHER_BANDS_HZ), 3, 3)
        or not np.all(np.isfinite(covariances))
        or not np.array_equal(covariances, np.swapaxes(covariances, -1, -2))
    ):
        raise HemiQV2Error("HemiQ-v2 teacher covariance geometry is invalid")
    eigenvalues = np.linalg.eigvalsh(covariances.astype(np.float64))
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
        raise HemiQV2Error("HemiQ-v2 teacher covariance is not positive after float32")


def derive_hemiq_views(
    raw_epochs: NDArray[np.floating],
    *,
    channel_names: Sequence[str],
    sfreq: float = EXPECTED_SFREQ,
) -> dict[str, np.ndarray]:
    """Derive the broadband neural view and four-band source-teacher SPD view."""

    raw = np.ascontiguousarray(raw_epochs, dtype=np.float32)
    channels = tuple(str(value) for value in channel_names)
    if channels != HEMI_Q_CHANNELS:
        raise ValueError("HemiQ-v2 requires exact supplied-bipolar C3,Cz,C4 order")
    if (
        raw.ndim != 3
        or raw.shape[1:] != (3, EXPECTED_N_TIMES)
        or not np.all(np.isfinite(raw))
    ):
        raise ValueError("HemiQ-v2 raw input must have shape (trials, 3, 320)")
    if not math.isclose(sfreq, EXPECTED_SFREQ, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("HemiQ-v2 requires exactly 128 Hz")

    broadband = _filter_and_demean(raw, _filter_design(BROADBAND_HZ, sfreq))
    filtered_bands = np.stack(
        [
            _filter_and_demean(raw, _filter_design(band, sfreq))
            for band in TEACHER_BANDS_HZ
        ],
        axis=1,
    )
    filtered_bands = np.ascontiguousarray(filtered_bands, dtype=np.float32)
    covariance = np.ascontiguousarray(
        make_spd_covariances(
            filtered_bands,
            shrinkage=1e-3,
            method="fixed",
            demean=True,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    _verify_float32_spd(covariance)
    return {"raw": broadband, "covariance": covariance}


def _binary_loss(logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logit, labels.to(dtype=logit.dtype)
    )


def _validate_partition(
    raw: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.ascontiguousarray(raw, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (3, EXPECTED_N_TIMES):
        raise ValueError(f"{name} raw input has the wrong geometry")
    if y.shape != (len(values),) or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError(f"{name} labels must be a binary vector with both classes")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} raw input contains non-finite values")
    return values, y


def _validate_covariances(
    value: NDArray[np.floating],
    *,
    n_rows: int,
) -> np.ndarray:
    covariance = np.ascontiguousarray(value, dtype=np.float32)
    if covariance.shape != (n_rows, len(TEACHER_BANDS_HZ), 3, 3):
        raise ValueError("teacher covariances have the wrong geometry")
    _verify_float32_spd(covariance)
    return covariance


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise HemiQV2Error("CUBLAS_WORKSPACE_CONFIG violates deterministic execution")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class HemiQHarmonizedV2Classifier:
    """Explicit selection/reset-refit HemiQ-v2 estimator."""

    def __init__(self, config: HemiQFieldConfig) -> None:
        if (
            config.n_times != EXPECTED_N_TIMES
            or not math.isclose(config.sfreq, EXPECTED_SFREQ, rel_tol=0.0, abs_tol=0.0)
        ):
            raise ValueError("classifier requires the harmonized-v2 sampling contract")
        self.config = config

    def _fit_raw_scaler(self, source: np.ndarray) -> None:
        hemispheric = source[:, (0, 2), :].reshape(-1).astype(np.float64)
        centre = source[:, 1, :].reshape(-1).astype(np.float64)
        self.raw_mean_ = np.asarray(
            [[[hemispheric.mean()], [centre.mean()], [hemispheric.mean()]]],
            dtype=np.float32,
        )
        self.raw_std_ = np.asarray(
            [[[hemispheric.std() + 1e-6], [centre.std() + 1e-6], [hemispheric.std() + 1e-6]]],
            dtype=np.float32,
        )

    def _prepare_raw(self, raw: NDArray[np.floating]) -> np.ndarray:
        values = np.ascontiguousarray(raw, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (3, EXPECTED_N_TIMES):
            raise ValueError("raw input has the wrong HemiQ-v2 geometry")
        return np.ascontiguousarray(
            (values - self.raw_mean_) / self.raw_std_,
            dtype=np.float32,
        )

    def _teacher_logits(
        self,
        covariance: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray | None:
        config = self.config
        if config.teacher_weight == 0.0:
            self.teacher_was_used_ = False
            self.teacher_parameter_count_ = 0
            self.teacher_preprocessing_ = {}
            return None
        mirror = np.asarray((2, 1, 0), dtype=np.int64)
        teacher = FrozenTangentAnchor(
            regularization=config.teacher_regularization
        ).fit(covariance, labels, mirror)
        reflected = _mirror_covariances(covariance, mirror)
        logits = 0.5 * (
            teacher.signed_logit(covariance) - teacher.signed_logit(reflected)
        )
        self.teacher_was_used_ = True
        self.teacher_parameter_count_ = int(
            teacher.model_.coef_.size + teacher.model_.intercept_.size
        )
        self.teacher_preprocessing_ = {
            "log_reference": _array_manifest(
                teacher.log_reference_.detach().cpu().numpy()
            ),
            "scaler_mean": _array_manifest(np.asarray(teacher.scaler_.mean_)),
            "scaler_scale": _array_manifest(np.asarray(teacher.scaler_.scale_)),
            "logistic_coef": _array_manifest(np.asarray(teacher.model_.coef_)),
            "logistic_intercept": _array_manifest(
                np.asarray(teacher.model_.intercept_)
            ),
        }
        return np.ascontiguousarray(logits, dtype=np.float32)

    def _initialize(
        self,
        raw_source: NDArray[np.floating],
        covariance_source: NDArray[np.floating],
        labels_source: NDArray[np.integer],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.optim.Optimizer, torch.Generator]:
        raw, labels = _validate_partition(raw_source, labels_source, name="source")
        covariance = _validate_covariances(covariance_source, n_rows=len(raw))
        _configure_determinism(self.config.seed)
        self.device_ = torch.device(self.config.device)
        if self.device_.type == "cuda" and not torch.cuda.is_available():
            raise HemiQV2Error("CUDA was requested but is unavailable")
        self._fit_raw_scaler(raw)
        prepared = self._prepare_raw(raw)
        teacher_logits = self._teacher_logits(covariance, labels)
        self.model_ = HemiQFieldNet(
            n_times=self.config.n_times,
            sfreq=self.config.sfreq,
            n_filters=self.config.n_filters,
            kernel_size=self.config.kernel_size,
            temporal_stride=self.config.temporal_stride,
            local_window=self.config.local_window,
            local_stride=self.config.local_stride,
            width=self.config.width,
            frequency_low=self.config.frequency_low,
            frequency_high=self.config.frequency_high,
            bandwidth_low=self.config.bandwidth_low,
            bandwidth_high=self.config.bandwidth_high,
            spectral_epsilon=self.config.spectral_epsilon,
        ).to(self.device_)
        self.initial_model_state_ = copy.deepcopy(self.model_.state_dict())
        self.initial_model_state_sha256_ = _state_sha256(self.initial_model_state_)
        self.parameter_count_ = int(
            sum(parameter.numel() for parameter in self.model_.parameters())
        )
        self.trainable_parameter_count_ = int(
            sum(
                parameter.numel()
                for parameter in self.model_.parameters()
                if parameter.requires_grad
            )
        )
        if self.trainable_parameter_count_ >= 50_000:
            raise HemiQV2Error("HemiQ-v2 exceeded the frozen parameter budget")
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = torch.Generator(device=self.device_).manual_seed(
            self.config.seed + 1
        )
        raw_tensor = torch.from_numpy(prepared).to(self.device_)
        labels_tensor = torch.from_numpy(labels).to(self.device_)
        teacher_tensor = (
            None
            if teacher_logits is None
            else torch.from_numpy(teacher_logits).to(self.device_)
        )
        self.preprocessing_manifest_ = {
            "raw_mean": _array_manifest(self.raw_mean_),
            "raw_std": _array_manifest(self.raw_std_),
            "teacher": copy.deepcopy(self.teacher_preprocessing_),
        }
        return raw_tensor, labels_tensor, teacher_tensor, optimizer, generator

    def _train_epoch(
        self,
        *,
        epoch: int,
        raw: torch.Tensor,
        labels: torch.Tensor,
        teacher: torch.Tensor | None,
        optimizer: torch.optim.Optimizer,
        generator: torch.Generator,
    ) -> tuple[float, float]:
        self.model_.train()
        order = torch.randperm(len(labels), device=self.device_, generator=generator)
        decay_horizon = max(
            1,
            int(math.ceil(self.config.epochs * self.config.teacher_fraction)),
        )
        coefficient = self.config.teacher_weight * max(
            0.0, 1.0 - epoch / decay_horizon
        )
        total = 0.0
        for start in range(0, len(order), self.config.batch_size):
            rows = order[start : start + self.config.batch_size]
            logit = self.model_(raw[rows])
            loss = _binary_loss(logit, labels[rows])
            if teacher is not None and coefficient > 0.0:
                loss = loss + coefficient * _binary_loss(
                    logit, torch.sigmoid(teacher[rows])
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model_.parameters(), self.config.gradient_clip
            )
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        return total / len(labels), float(coefficient)

    def fit_selection(
        self,
        raw_train: NDArray[np.floating],
        covariance_train: NDArray[np.floating],
        labels_train: NDArray[np.integer],
        raw_validation: NDArray[np.floating],
        labels_validation: NDArray[np.integer],
    ) -> "HemiQHarmonizedV2Classifier":
        """Select a duration on validation BCE without reading validation SPD."""

        validation_raw, validation_labels = _validate_partition(
            raw_validation, labels_validation, name="validation"
        )
        started = time.perf_counter()
        raw, labels, teacher, optimizer, generator = self._initialize(
            raw_train, covariance_train, labels_train
        )
        prepared_validation = torch.from_numpy(
            self._prepare_raw(validation_raw)
        ).to(self.device_)
        validation_y = torch.from_numpy(validation_labels).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            initial = self.model_(prepared_validation)
            best_loss = float(_binary_loss(initial, validation_y))
        best_state = copy.deepcopy(self.model_.state_dict())
        best_epoch = -1
        stale = 0
        self.history_: list[dict[str, Any]] = [
            {
                "epoch": -1,
                "train_loss": None,
                "validation_loss": best_loss,
                "teacher_coefficient": 0.0,
            }
        ]
        for epoch in range(self.config.epochs):
            train_loss, coefficient = self._train_epoch(
                epoch=epoch,
                raw=raw,
                labels=labels,
                teacher=teacher,
                optimizer=optimizer,
                generator=generator,
            )
            self.model_.eval()
            with torch.no_grad():
                validation_loss = float(
                    _binary_loss(self.model_(prepared_validation), validation_y)
                )
            self.history_.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "teacher_coefficient": coefficient,
                }
            )
            if validation_loss < best_loss - self.config.min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.config.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_epoch_ = best_epoch
        self.selected_epoch_count_ = best_epoch + 1
        self.epochs_run_ = len(self.history_) - 1
        self.best_validation_loss_ = best_loss
        self.train_seconds_ = time.perf_counter() - started
        self.mode_ = "selection"
        self.model_state_sha256_ = _state_sha256(self.model_.state_dict())
        return self

    def fit_fixed_epochs(
        self,
        raw_source: NDArray[np.floating],
        covariance_source: NDArray[np.floating],
        labels_source: NDArray[np.integer],
        *,
        epochs: int,
    ) -> "HemiQHarmonizedV2Classifier":
        """Reset and fit source data for exactly the selected duration."""

        if isinstance(epochs, bool) or not isinstance(epochs, int):
            raise ValueError("epochs must be an integer")
        if not 0 <= epochs <= self.config.epochs:
            raise ValueError("epochs lies outside the frozen HemiQ horizon")
        started = time.perf_counter()
        raw, labels, teacher, optimizer, generator = self._initialize(
            raw_source, covariance_source, labels_source
        )
        self.history_ = []
        for epoch in range(epochs):
            train_loss, coefficient = self._train_epoch(
                epoch=epoch,
                raw=raw,
                labels=labels,
                teacher=teacher,
                optimizer=optimizer,
                generator=generator,
            )
            self.history_.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "teacher_coefficient": coefficient,
                }
            )
        self.model_.eval()
        self.epochs_run_ = epochs
        self.train_seconds_ = time.perf_counter() - started
        self.mode_ = "fixed_refit"
        self.model_state_sha256_ = _state_sha256(self.model_.state_dict())
        return self

    def decision_function(self, raw: NDArray[np.floating]) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before prediction")
        prepared = self._prepare_raw(raw)
        chunks: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(prepared), 256):
                chunks.append(
                    self.model_(
                        torch.from_numpy(prepared[start : start + 256]).to(
                            self.device_
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
        return (
            np.concatenate(chunks).astype(np.float32, copy=False)
            if chunks
            else np.empty(0, dtype=np.float32)
        )

    def predict_proba(self, raw: NDArray[np.floating]) -> np.ndarray:
        logit = self.decision_function(raw)
        positive = 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
        return np.column_stack((1.0 - positive, positive))


__all__ = [
    "BROADBAND_HZ",
    "CONFIG_RELATIVE_PATH",
    "HemiQHarmonizedV2Classifier",
    "HemiQV2Error",
    "TEACHER_BANDS_HZ",
    "derive_hemiq_views",
    "hemiq_view_contract",
    "load_frozen_config",
]
