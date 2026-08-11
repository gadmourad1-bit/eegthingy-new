"""Frozen-weight streaming inference for exported GeoAdaptNet checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .engine import load_model_checkpoint, resolve_device
from .model import GeoAdaptNet
from .recenter import (
    BoundaryRecenter,
    DualLevelOnlineAdapter,
    LogEuclideanCovarianceRecenter,
)


@dataclass(frozen=True)
class DeploymentDecision:
    prediction: int
    probability_left: float
    probability_right: float
    intent_probability: float | None
    rest_probability: float | None
    confidence: float
    committed: bool
    margin: float
    covariance_updated: bool
    boundary_updated: bool


class GeoAdaptDecoder:
    """Checkpoint-backed causal decoder operating on filter-bank covariances.

    Preprocessing is intentionally outside this class: each input must already be
    a `(bands, channels, channels)` shrinkage covariance block constructed with the
    checkpoint's data contract.  Calibration and stream steps never use a label.
    """

    def __init__(
        self,
        model: GeoAdaptNet,
        metadata: dict[str, Any],
        *,
        device: str = "auto",
        commit_confidence: float | None = None,
        intent_threshold: float = 0.5,
        preprocessing_contract: dict[str, Any] | None = None,
    ) -> None:
        if not 0.0 <= intent_threshold <= 1.0:
            raise ValueError("intent_threshold must be in [0, 1]")
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.metadata = dict(metadata)
        self.data_contract = dict(metadata.get("data_contract", {}))
        preprocessing = dict(self.data_contract.get("preprocessing", {}))
        if preprocessing:
            channels = tuple(preprocessing.get("channels", ()))
            bands = tuple(tuple(value) for value in preprocessing.get("bands", ()))
            if len(channels) != self.model.channels or len(bands) != self.model.bands:
                raise ValueError("checkpoint model and preprocessing contract dimensions differ")
        if preprocessing_contract is not None:
            self.validate_preprocessing_contract(preprocessing_contract)
        benchmark = dict(
            metadata.get("benchmark_config", metadata.get("loso_config", {}))
        )
        threshold = (
            float(commit_confidence)
            if commit_confidence is not None
            else float(benchmark.get("commit_confidence", 0.85))
        )
        self.temperature = float(metadata.get("temperature", 1.0))
        self.target_median_blend = float(metadata.get("target_median_blend", 1.0))
        self.intent_threshold = float(intent_threshold)
        self.adapter = DualLevelOnlineAdapter(
            LogEuclideanCovarianceRecenter(
                alpha=0.001,
                robust_clip=1.0,
                eig_floor=1e-15,
                alignment_enabled=True,
                adaptation_enabled=True,
            ),
            BoundaryRecenter(
                # Anti-drift deployment settings: a mean-reversion leak plus a tighter
                # clamp keep the boundary from leaning toward a class over a session,
                # while still tracking genuine neutral-point drift.
                alpha=0.008,
                leak=0.012,
                clamp=1.0,
                rest_confidence=0.60,
                commit_threshold=threshold,
                temperature=self.temperature,
                class1_label=0,
                class2_label=1,
            ),
            covariance_updates_rest_only=True,
        )
        self._zero_reference = np.zeros(
            (self.model.bands, self.model.channels, self.model.channels), dtype=np.float32
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        commit_confidence: float | None = None,
        intent_threshold: float = 0.5,
        preprocessing_contract: dict[str, Any] | None = None,
    ) -> "GeoAdaptDecoder":
        resolved = resolve_device(device)
        model, metadata = load_model_checkpoint(path, map_location=resolved)
        return cls(
            model,
            metadata,
            device=str(resolved),
            commit_confidence=commit_confidence,
            intent_threshold=intent_threshold,
            preprocessing_contract=preprocessing_contract,
        )

    def validate_preprocessing_contract(self, contract: dict[str, Any]) -> None:
        """Hard-fail if a live preprocessing contract differs from the checkpoint."""

        if not self.data_contract:
            raise ValueError("checkpoint does not contain a preprocessing/data contract")
        expected = self.data_contract.get("preprocessing_fingerprint")
        observed = dict(contract).get("preprocessing_fingerprint")
        if not expected or observed != expected:
            raise ValueError("live preprocessing contract does not match checkpoint")

    def _validate(self, covariances: Any, *, calibration: bool) -> np.ndarray:
        values = np.asarray(covariances, dtype=np.float64)
        expected = (self.model.bands, self.model.channels, self.model.channels)
        if calibration:
            if values.ndim != 4 or tuple(values.shape[1:]) != expected:
                raise ValueError(f"calibration covariances must have shape (N, {expected})")
            if len(values) < 2:
                raise ValueError("at least two unlabeled calibration windows are required")
        elif values.ndim != 3 or tuple(values.shape) != expected:
            raise ValueError(f"stream covariance must have shape {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError("covariances contain NaN or infinity")
        return values

    @torch.inference_mode()
    def _network(self, aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        values = np.asarray(aligned, dtype=np.float32)
        if values.ndim == 3:
            values = values[np.newaxis]
        covariances = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(
            self._zero_reference, dtype=torch.float32, device=self.device
        )
        output = self.model(
            covariances,
            log_reference=reference,
            update_reference=False,
        )
        scores = output.log_odds.detach().cpu().numpy()
        intent = (
            None
            if output.intent_logit is None
            else torch.sigmoid(output.intent_logit).detach().cpu().numpy()
        )
        return scores, intent

    def calibrate(self, covariances: Any) -> "GeoAdaptDecoder":
        """Seed covariance and boundary state from an unlabeled chronological block."""

        values = self._validate(covariances, calibration=True)
        self.adapter.covariance.calibrate(values)
        aligned = self.adapter.covariance.transform(values)
        scores, _ = self._network(aligned)
        self.adapter.boundary.calibrate(scores)
        self.adapter.boundary.center *= self.target_median_blend
        self.adapter.boundary.seed_center = self.adapter.boundary.center
        return self

    def step(self, covariance: Any, *, update: bool = True) -> DeploymentDecision:
        """Predict with current state and only then permit a causal state update."""

        values = self._validate(covariance, calibration=False)
        aligned = self.adapter.covariance.transform(values)
        scores, intent = self._network(aligned)
        score = float(scores[0])
        intent_probability = None if intent is None else float(intent[0])
        rest_probability = (
            None if intent_probability is None else 1.0 - intent_probability
        )
        result = self.adapter.step(
            values,
            score=score,
            rest_probability=rest_probability,
            update=update,
        )
        probability_right = 1.0 / (
            1.0 + np.exp(-np.clip(result.margin / self.temperature, -40.0, 40.0))
        )
        intent_permits = (
            intent_probability is None or intent_probability >= self.intent_threshold
        )
        return DeploymentDecision(
            prediction=result.prediction,
            probability_left=float(1.0 - probability_right),
            probability_right=float(probability_right),
            intent_probability=intent_probability,
            rest_probability=rest_probability,
            confidence=result.confidence,
            committed=bool(result.committed and intent_permits),
            margin=result.margin,
            covariance_updated=result.covariance_updated,
            boundary_updated=result.boundary_updated,
        )

    def adapter_state_json(self) -> str:
        return self.adapter.to_json(indent=2)


__all__ = ["DeploymentDecision", "GeoAdaptDecoder"]
