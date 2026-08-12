"""Causal, label-free online recentering for streaming EEG inference.

The model weights stay frozen.  This module adapts only two small pieces of
deployment state:

* a per-band covariance reference in the log-Euclidean domain; and
* a scalar decision boundary in log-odds space.

Both adapters follow a predict-then-update contract.  The observation at time
``t`` is transformed and classified using state ``t``; only after that result
has been formed may the observation update state ``t + 1``.  No epochs, labels,
gradients, optimizer state, or replay buffer are retained, so memory use is
constant in stream length.

NumPy is deliberate here.  Online adaptation is outside autograd and can run
in the embedded inference process.  A Torch model can consume the returned
aligned block with ``torch.as_tensor(result.aligned_covariances, ...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


STATE_VERSION = 1


class NotCalibratedError(RuntimeError):
    """Raised when streaming inference is attempted before calibration."""


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + np.swapaxes(matrix, -1, -2))


def _reconstruct(eigenvectors: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    result = (eigenvectors * eigenvalues[..., np.newaxis, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    return _symmetrize(result)


def _project_spd(
    matrices: np.ndarray, eig_floor: float, *, strict: bool = False
) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("covariance matrices must be square")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("covariance matrices must contain only finite values")

    values, vectors = np.linalg.eigh(_symmetrize(matrices))
    if strict and np.any(values < eig_floor):
        minimum = float(np.min(values))
        raise ValueError(
            f"covariance is not safely SPD: minimum eigenvalue {minimum:.3g} "
            f"is below eig_floor={eig_floor:.3g}"
        )
    return _reconstruct(vectors, np.maximum(values, eig_floor))


def _spd_log(matrices: np.ndarray, eig_floor: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(_symmetrize(matrices))
    values = np.maximum(values, eig_floor)
    return _reconstruct(vectors, np.log(values))


def _symmetric_exp(matrices: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_symmetrize(matrices))
    return _reconstruct(vectors, np.exp(values))


def _finite_scalar(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar")
    scalar = float(array.reshape(-1)[0])
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class LogEuclideanCovarianceRecenter:
    """Per-band causal covariance alignment with a robust online reference.

    Calibration receives an unlabeled block shaped ``(windows, bands, C, C)``.
    For each band it stores only

    ``M_b = mean_i(log(C_i,b))``.

    The corresponding reference is ``R_b = exp(M_b)`` and a live covariance is
    aligned by the congruence ``R_b^-1/2 C R_b^-1/2``.  After the aligned value
    has been returned, the reference can take a slow EMA step in matrix-log
    space.  A Frobenius-norm clip bounds each per-band innovation, making an
    artifact a bounded update instead of a new reference.

    ``alignment_enabled`` and ``adaptation_enabled`` independently provide the
    useful ablations: raw covariance, frozen calibration reference, or fully
    online recentering.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.001,
        robust_clip: float | None = 1.0,
        eig_floor: float = 1e-12,
        alignment_enabled: bool = True,
        adaptation_enabled: bool = True,
        strict_spd: bool = False,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if robust_clip is not None and robust_clip <= 0.0:
            raise ValueError("robust_clip must be positive or None")
        if eig_floor <= 0.0:
            raise ValueError("eig_floor must be positive")

        self.alpha = float(alpha)
        self.robust_clip = None if robust_clip is None else float(robust_clip)
        self.eig_floor = float(eig_floor)
        self.alignment_enabled = bool(alignment_enabled)
        self.adaptation_enabled = bool(adaptation_enabled)
        self.strict_spd = bool(strict_spd)

        self._reference_log: np.ndarray | None = None
        self._seed_reference_log: np.ndarray | None = None
        self.calibration_windows = 0
        self.update_count = 0

    @property
    def is_calibrated(self) -> bool:
        return self._reference_log is not None

    @property
    def n_bands(self) -> int | None:
        return None if self._reference_log is None else int(self._reference_log.shape[0])

    @property
    def n_channels(self) -> int | None:
        return None if self._reference_log is None else int(self._reference_log.shape[-1])

    @property
    def reference_log(self) -> np.ndarray:
        self._require_calibrated()
        return self._reference_log.copy()  # type: ignore[union-attr]

    @property
    def seed_reference_log(self) -> np.ndarray:
        self._require_calibrated()
        return self._seed_reference_log.copy()  # type: ignore[union-attr]

    @property
    def reference_covariances(self) -> np.ndarray:
        self._require_calibrated()
        return _symmetric_exp(self._reference_log)  # type: ignore[arg-type]

    def _require_calibrated(self) -> None:
        if not self.is_calibrated:
            raise NotCalibratedError(
                "covariance recentering requires an unlabeled calibration block"
            )

    def _calibration_array(self, covariances: Any) -> np.ndarray:
        array = np.asarray(covariances, dtype=np.float64)
        if array.ndim == 2:
            array = array[np.newaxis, np.newaxis, ...]
        elif array.ndim == 3:
            # An unambiguous and useful convention for the one-band case.
            array = array[:, np.newaxis, ...]
        elif array.ndim != 4:
            raise ValueError(
                "calibration covariances must have shape (N, B, C, C); "
                "(N, C, C) is accepted for one band"
            )
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("calibration block must contain at least one window and band")
        return _project_spd(array, self.eig_floor, strict=self.strict_spd)

    def _stream_array(self, covariances: Any) -> tuple[np.ndarray, str]:
        self._require_calibrated()
        array = np.asarray(covariances, dtype=np.float64)
        if array.ndim == 2:
            array = array[np.newaxis, np.newaxis, ...]
            shape_kind = "matrix"
        elif array.ndim == 3:
            if self.n_bands == 1 and array.shape[0] != 1:
                # Batch transform for a single-band adapter.
                array = array[:, np.newaxis, ...]
                shape_kind = "single_band_batch"
            else:
                array = array[np.newaxis, ...]
                shape_kind = "bands"
        elif array.ndim == 4:
            shape_kind = "batch"
        else:
            raise ValueError(
                "covariances must have shape (C,C), (B,C,C), or (N,B,C,C)"
            )

        expected = (self.n_bands, self.n_channels, self.n_channels)
        if tuple(array.shape[1:]) != expected:
            raise ValueError(
                f"expected each window to have shape {expected}, got {array.shape[1:]}"
            )
        return _project_spd(array, self.eig_floor, strict=self.strict_spd), shape_kind

    @staticmethod
    def _restore_shape(array: np.ndarray, shape_kind: str) -> np.ndarray:
        if shape_kind == "matrix":
            return array[0, 0]
        if shape_kind == "bands":
            return array[0]
        if shape_kind == "single_band_batch":
            return array[:, 0]
        return array

    def calibrate(self, covariances: Any) -> "LogEuclideanCovarianceRecenter":
        """Seed all band references from an unlabeled calibration block."""

        block = self._calibration_array(covariances)
        reference_log = np.mean(_spd_log(block, self.eig_floor), axis=0)
        self._reference_log = _symmetrize(reference_log)
        self._seed_reference_log = self._reference_log.copy()
        self.calibration_windows = int(block.shape[0])
        self.update_count = 0
        return self

    # ``seed`` mirrors the boundary adapter and is convenient in deployment code.
    seed = calibrate

    def transform(self, covariances: Any) -> np.ndarray:
        """Align one or more windows using the current reference, without updating it."""

        block, shape_kind = self._stream_array(covariances)
        if not self.alignment_enabled:
            return self._restore_shape(block.copy(), shape_kind)

        whitener = _symmetric_exp(-0.5 * self._reference_log)  # type: ignore[operator]
        aligned = whitener[np.newaxis, ...] @ block @ whitener[np.newaxis, ...]
        # Congruence preserves SPD analytically; this projection removes only
        # floating-point asymmetry/roundoff at very small eigenvalues.
        aligned = _project_spd(aligned, self.eig_floor, strict=False)
        return self._restore_shape(aligned, shape_kind)

    align = transform

    def update(self, covariances: Any, *, weight: float = 1.0) -> bool:
        """Advance the reference after a prediction has consumed this window.

        ``weight`` can be supplied by a label-free rest detector.  Zero skips the
        update.  Only one streaming window is accepted, which makes accidental
        future-window leakage difficult.
        """

        self._require_calibrated()
        weight = _finite_scalar(weight, "weight")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        if not self.adaptation_enabled or self.alpha == 0.0 or weight == 0.0:
            return False

        block, _ = self._stream_array(covariances)
        if block.shape[0] != 1:
            raise ValueError("online update accepts exactly one window")
        observation_log = _spd_log(block[0], self.eig_floor)
        residual = observation_log - self._reference_log  # type: ignore[operator]

        robust_weight = np.ones((residual.shape[0], 1, 1), dtype=np.float64)
        if self.robust_clip is not None:
            # Normalization by sqrt(C) makes the clip comparable across montages.
            norm = np.linalg.norm(residual, axis=(-2, -1)) / math.sqrt(residual.shape[-1])
            nonzero = norm > self.robust_clip
            robust_weight[nonzero, 0, 0] = self.robust_clip / norm[nonzero]

        step = self.alpha * weight * robust_weight * residual
        self._reference_log = _symmetrize(self._reference_log + step)  # type: ignore[operator]
        self.update_count += 1
        return bool(np.any(step != 0.0))

    def step(
        self,
        covariances: Any,
        *,
        update: bool = True,
        update_weight: float = 1.0,
    ) -> np.ndarray:
        """Return alignment under ``R_t``, then optionally construct ``R_t+1``."""

        aligned = self.transform(covariances)
        if update:
            self.update(covariances, weight=update_weight)
        return aligned

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "version": STATE_VERSION,
            "config": {
                "alpha": self.alpha,
                "robust_clip": self.robust_clip,
                "eig_floor": self.eig_floor,
                "alignment_enabled": self.alignment_enabled,
                "adaptation_enabled": self.adaptation_enabled,
                "strict_spd": self.strict_spd,
            },
            "reference_log": (
                None if self._reference_log is None else self._reference_log.copy()
            ),
            "seed_reference_log": (
                None
                if self._seed_reference_log is None
                else self._seed_reference_log.copy()
            ),
            "calibration_windows": self.calibration_windows,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "LogEuclideanCovarianceRecenter":
        if int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError("unsupported covariance recenter state version")
        config = dict(state["config"])
        replacement = type(self)(**config)
        reference = state.get("reference_log")
        seed_reference = state.get("seed_reference_log")
        if (reference is None) != (seed_reference is None):
            raise ValueError("reference and seed reference must be present together")
        if reference is not None:
            reference_array = np.asarray(reference, dtype=np.float64)
            seed_array = np.asarray(seed_reference, dtype=np.float64)
            if reference_array.ndim != 3 or reference_array.shape[-1] != reference_array.shape[-2]:
                raise ValueError("invalid serialized covariance reference shape")
            if seed_array.shape != reference_array.shape:
                raise ValueError("serialized seed/reference shapes differ")
            if not np.all(np.isfinite(reference_array)) or not np.all(np.isfinite(seed_array)):
                raise ValueError("serialized covariance state must be finite")
            replacement._reference_log = _symmetrize(reference_array.copy())
            replacement._seed_reference_log = _symmetrize(seed_array.copy())
        replacement.calibration_windows = int(state.get("calibration_windows", 0))
        replacement.update_count = int(state.get("update_count", 0))
        self.__dict__.update(replacement.__dict__)
        return self

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any]
    ) -> "LogEuclideanCovarianceRecenter":
        return cls().load_state_dict(state)


@dataclass(frozen=True)
class BoundaryDecision:
    """One causal boundary decision and the state transition it triggered."""

    prediction: int
    margin: float
    confidence: float
    committed: bool
    center_before: float
    center_after: float
    rest_like: bool
    updated: bool


class BoundaryRecenter:
    """Causal, label-free decision-boundary recentering in log-odds space.

    The center is seeded by the median calibration log-odds.  Each live score is
    first evaluated against ``center_t``.  A slow, clamped EMA update is then
    permitted only for a low-confidence/rest-like window.  If an independent
    rest detector supplies ``rest_probability``, its probability must also pass
    ``external_rest_threshold`` and scales the update.  This uses no test label.

    Confidence is the binary probability implied by the recentered log-odds,
    ``sigmoid(abs(margin) / temperature)``.  ``temperature`` supports a fixed
    calibration temperature learned on training/validation data without making
    any online gradient update.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.01,
        clamp: float = 2.0,
        rest_confidence: float = 0.65,
        commit_threshold: float = 0.85,
        temperature: float = 1.0,
        external_rest_threshold: float = 0.5,
        centering_enabled: bool = True,
        adaptation_enabled: bool = True,
        rest_gating_enabled: bool = True,
        abstention_enabled: bool = True,
        weight_by_rest_probability: bool = True,
        class1_label: int = 1,
        class2_label: int = 2,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if clamp < 0.0:
            raise ValueError("clamp must be non-negative")
        if not 0.5 < rest_confidence < 1.0:
            raise ValueError("rest_confidence must be in (0.5, 1)")
        if not 0.5 <= commit_threshold <= 1.0:
            raise ValueError("commit_threshold must be in [0.5, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= external_rest_threshold <= 1.0:
            raise ValueError("external_rest_threshold must be in [0, 1]")
        if class1_label == class2_label:
            raise ValueError("class labels must differ")

        self.alpha = float(alpha)
        self.clamp = float(clamp)
        self.rest_confidence = float(rest_confidence)
        self.commit_threshold = float(commit_threshold)
        self.temperature = float(temperature)
        self.external_rest_threshold = float(external_rest_threshold)
        self.centering_enabled = bool(centering_enabled)
        self.adaptation_enabled = bool(adaptation_enabled)
        self.rest_gating_enabled = bool(rest_gating_enabled)
        self.abstention_enabled = bool(abstention_enabled)
        self.weight_by_rest_probability = bool(weight_by_rest_probability)
        self.class1_label = int(class1_label)
        self.class2_label = int(class2_label)

        self.center = 0.0
        self.seed_center = 0.0
        self.calibration_windows = 0
        self.update_count = 0

    @property
    def is_calibrated(self) -> bool:
        return self.calibration_windows > 0

    @property
    def rest_margin(self) -> float:
        # Confidence uses margin / temperature, so invert that same mapping.
        odds = self.rest_confidence / (1.0 - self.rest_confidence)
        return self.temperature * math.log(odds)

    def calibrate(self, scores: Sequence[float] | np.ndarray) -> "BoundaryRecenter":
        """Seed ``center_0`` from the median finite unlabeled calibration score."""

        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("calibration scores must contain at least one finite value")
        self.center = self.seed_center = float(np.median(values))
        self.calibration_windows = int(values.size)
        self.update_count = 0
        return self

    seed = calibrate

    def _confidence(self, margin: float) -> float:
        # abs(margin) >= 0, so exp receives a non-positive value and cannot overflow.
        return 1.0 / (1.0 + math.exp(-abs(margin) / self.temperature))

    def step(
        self,
        score: float,
        *,
        rest_probability: float | None = None,
        update: bool = True,
    ) -> BoundaryDecision:
        """Predict with ``center_t`` and only then update to ``center_t+1``."""

        score = _finite_scalar(score, "score")
        if rest_probability is not None:
            rest_probability = _finite_scalar(rest_probability, "rest_probability")
            if not 0.0 <= rest_probability <= 1.0:
                raise ValueError("rest_probability must be in [0, 1]")

        center_before = self.center
        margin = score - center_before if self.centering_enabled else score
        confidence = self._confidence(margin)
        prediction = self.class2_label if margin > 0.0 else self.class1_label
        committed = (not self.abstention_enabled) or confidence >= self.commit_threshold
        rest_like = confidence < self.rest_confidence

        external_gate = (
            rest_probability is None
            or rest_probability >= self.external_rest_threshold
        )
        internal_gate = (not self.rest_gating_enabled) or rest_like
        should_update = (
            update
            and self.centering_enabled
            and self.adaptation_enabled
            and self.alpha > 0.0
            and internal_gate
            and external_gate
        )

        updated = False
        if should_update:
            probability_weight = (
                rest_probability
                if rest_probability is not None and self.weight_by_rest_probability
                else 1.0
            )
            candidate = center_before + self.alpha * probability_weight * (
                score - center_before
            )
            low = self.seed_center - self.clamp
            high = self.seed_center + self.clamp
            self.center = float(np.clip(candidate, low, high))
            updated = self.center != center_before
            if updated:
                self.update_count += 1

        return BoundaryDecision(
            prediction=prediction,
            margin=margin,
            confidence=confidence,
            committed=bool(committed),
            center_before=center_before,
            center_after=self.center,
            rest_like=bool(rest_like),
            updated=updated,
        )

    def predict(self, score: float) -> BoundaryDecision:
        """Evaluate a score without changing online state."""

        return self.step(score, update=False)

    def update(
        self, score: float, *, rest_probability: float | None = None
    ) -> float:
        """Compatibility helper returning the *pre-update* recentered margin."""

        return self.step(score, rest_probability=rest_probability, update=True).margin

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "version": STATE_VERSION,
            "config": {
                "alpha": self.alpha,
                "clamp": self.clamp,
                "rest_confidence": self.rest_confidence,
                "commit_threshold": self.commit_threshold,
                "temperature": self.temperature,
                "external_rest_threshold": self.external_rest_threshold,
                "centering_enabled": self.centering_enabled,
                "adaptation_enabled": self.adaptation_enabled,
                "rest_gating_enabled": self.rest_gating_enabled,
                "abstention_enabled": self.abstention_enabled,
                "weight_by_rest_probability": self.weight_by_rest_probability,
                "class1_label": self.class1_label,
                "class2_label": self.class2_label,
            },
            "center": self.center,
            "seed_center": self.seed_center,
            "calibration_windows": self.calibration_windows,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "BoundaryRecenter":
        if int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError("unsupported boundary recenter state version")
        replacement = type(self)(**dict(state["config"]))
        replacement.center = _finite_scalar(state["center"], "center")
        replacement.seed_center = _finite_scalar(state["seed_center"], "seed_center")
        replacement.calibration_windows = int(state.get("calibration_windows", 0))
        replacement.update_count = int(state.get("update_count", 0))
        low = replacement.seed_center - replacement.clamp
        high = replacement.seed_center + replacement.clamp
        if not low <= replacement.center <= high:
            raise ValueError("serialized center lies outside its clamp")
        self.__dict__.update(replacement.__dict__)
        return self

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "BoundaryRecenter":
        return cls().load_state_dict(state)


@dataclass(frozen=True)
class OnlineAdaptationResult:
    """Output of one dual-level causal inference step."""

    aligned_covariances: np.ndarray
    prediction: int
    margin: float
    confidence: float
    committed: bool
    covariance_updated: bool
    boundary_updated: bool
    center_before: float
    center_after: float


class DualLevelOnlineAdapter:
    """Coordinate covariance and boundary recentering without look-ahead.

    Order for a live window is fixed:

    1. align with covariance reference ``R_t``;
    2. obtain a log-odds score from that aligned block;
    3. form the prediction with boundary center ``c_t``;
    4. update boundary and covariance state for ``t + 1``.

    ``scorer`` may return a scalar log-odds or two logits/probability-like scores;
    for two values the coordinator uses ``value[1] - value[0]``.  Passing an
    already-computed scalar ``score`` is also supported, but the caller is then
    responsible for having computed it from the returned alignment convention.
    """

    def __init__(
        self,
        covariance: LogEuclideanCovarianceRecenter | None = None,
        boundary: BoundaryRecenter | None = None,
        *,
        covariance_updates_rest_only: bool = False,
    ) -> None:
        self.covariance = covariance or LogEuclideanCovarianceRecenter()
        self.boundary = boundary or BoundaryRecenter()
        self.covariance_updates_rest_only = bool(covariance_updates_rest_only)

    @property
    def is_calibrated(self) -> bool:
        return self.covariance.is_calibrated and self.boundary.is_calibrated

    @staticmethod
    def _scores_from_output(output: Any) -> np.ndarray:
        values = np.asarray(output, dtype=np.float64)
        if values.ndim == 0:
            return values.reshape(1)
        if values.shape[-1:] == (2,):
            return values[..., 1] - values[..., 0]
        return values.reshape(-1)

    @classmethod
    def _single_score(cls, output: Any) -> float:
        scores = cls._scores_from_output(output).reshape(-1)
        if scores.size != 1:
            raise ValueError("scorer must return one scalar score or one pair of logits")
        return _finite_scalar(scores[0], "score")

    def calibrate(
        self,
        covariances: Any,
        scores: Sequence[float] | np.ndarray | None = None,
        *,
        scorer: Callable[[np.ndarray], Any] | None = None,
    ) -> "DualLevelOnlineAdapter":
        """Seed both levels from the same unlabeled calibration block."""

        if (scores is None) == (scorer is None):
            raise ValueError("provide exactly one of scores or scorer")
        self.covariance.calibrate(covariances)
        if scorer is not None:
            aligned = self.covariance.transform(covariances)
            scores = self._scores_from_output(scorer(aligned))
        self.boundary.calibrate(scores)  # type: ignore[arg-type]
        return self

    def step(
        self,
        covariances: Any,
        score: float | None = None,
        *,
        scorer: Callable[[np.ndarray], Any] | None = None,
        rest_probability: float | None = None,
        update: bool = True,
    ) -> OnlineAdaptationResult:
        """Run one strictly causal dual-level inference step."""

        if not self.is_calibrated:
            raise NotCalibratedError("both online adaptation levels must be calibrated")
        if (score is None) == (scorer is None):
            raise ValueError("provide exactly one of score or scorer")

        # Everything returned to the caller is formed before either reference
        # observes this window.
        aligned = self.covariance.transform(covariances)
        live_score = (
            self._single_score(scorer(aligned))  # type: ignore[misc]
            if scorer is not None
            else _finite_scalar(score, "score")
        )
        decision = self.boundary.step(
            live_score, rest_probability=rest_probability, update=update
        )

        covariance_updated = False
        if update:
            covariance_weight = 1.0
            if self.covariance_updates_rest_only:
                covariance_weight = 1.0 if decision.rest_like else 0.0
                if rest_probability is not None:
                    probability = _finite_scalar(rest_probability, "rest_probability")
                    if probability < self.boundary.external_rest_threshold:
                        covariance_weight = 0.0
                    elif self.boundary.weight_by_rest_probability:
                        covariance_weight *= probability
            covariance_updated = self.covariance.update(
                covariances, weight=covariance_weight
            )

        return OnlineAdaptationResult(
            aligned_covariances=np.asarray(aligned).copy(),
            prediction=decision.prediction,
            margin=decision.margin,
            confidence=decision.confidence,
            committed=decision.committed,
            covariance_updated=covariance_updated,
            boundary_updated=decision.updated,
            center_before=decision.center_before,
            center_after=decision.center_after,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "version": STATE_VERSION,
            "config": {
                "covariance_updates_rest_only": self.covariance_updates_rest_only,
            },
            "covariance": self.covariance.state_dict(),
            "boundary": self.boundary.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> "DualLevelOnlineAdapter":
        if int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError("unsupported dual-level adapter state version")
        covariance = LogEuclideanCovarianceRecenter.from_state_dict(state["covariance"])
        boundary = BoundaryRecenter.from_state_dict(state["boundary"])
        replacement = type(self)(
            covariance,
            boundary,
            **dict(state.get("config", {})),
        )
        self.__dict__.update(replacement.__dict__)
        return self

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "DualLevelOnlineAdapter":
        return cls().load_state_dict(state)

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize all constant-size online state to portable JSON."""

        return json.dumps(_jsonable(self.state_dict()), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "DualLevelOnlineAdapter":
        return cls.from_state_dict(json.loads(payload))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DualLevelOnlineAdapter":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# Readable aliases for callers that prefer shorter or coordinator-oriented names.
CovarianceRecenter = LogEuclideanCovarianceRecenter
OnlineRecenterCoordinator = DualLevelOnlineAdapter


__all__ = [
    "BoundaryDecision",
    "BoundaryRecenter",
    "CovarianceRecenter",
    "DualLevelOnlineAdapter",
    "LogEuclideanCovarianceRecenter",
    "NotCalibratedError",
    "OnlineAdaptationResult",
    "OnlineRecenterCoordinator",
]
