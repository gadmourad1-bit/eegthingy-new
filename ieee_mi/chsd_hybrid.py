"""Experimental function-preserving CHSD residual model.

This module is intentionally separate from the formal CHSDNet implementation
and benchmark registry.  It combines the existing
``cardinal_fbc_micro_extended`` model with a complete CHSDNet branch whose
bounded residual gate starts at zero. At construction, the wrapper therefore
computes exactly the native backbone function while retaining a direct
gradient into the gate.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor, nn

from . import baselines
from .chsd import CHSDConfig, CHSDNet
from .models import CardinalFBCMicroDynamicsNet


HYBRID_MODEL_NAME = "chsdnet_hybrid"
NATIVE_MODEL_NAME = "cardinal_fbc_micro_extended"


def _native_config(model: nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if not isinstance(config, dict):
        raise TypeError("native backbone must expose dictionary metadata in config")
    required = {
        "extended_atlas": True,
        "continuation_kind": "gabor_fir",
        "dynamic_fbc_floor_input_channels": True,
        "dynamic_fbc_floor_input_times": True,
        "zero_initialized_continuation_head": True,
        "single_expanded_head": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(
                f"native backbone metadata {key!r} is {config.get(key)!r}; "
                f"expected {expected!r}"
            )
    return dict(config)


class CHSDResidualNet(nn.Module):
    """Native CardinalFBC micro dynamics plus a zero-started CHSD correction."""

    uses_positions = True

    def __init__(
        self,
        native_backbone: nn.Module,
        chsd_branch: CHSDNet,
        *,
        requested_name: str = HYBRID_MODEL_NAME,
        native_model_name: str = NATIVE_MODEL_NAME,
        maximum_residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if requested_name != HYBRID_MODEL_NAME:
            raise ValueError(f"unsupported requested model {requested_name!r}")
        if native_model_name != NATIVE_MODEL_NAME:
            raise ValueError(f"unsupported native model {native_model_name!r}")
        if not isinstance(native_backbone, CardinalFBCMicroDynamicsNet):
            raise TypeError(
                "native_backbone must be CardinalFBCMicroDynamicsNet from "
                "cardinal_fbc_micro_extended"
            )
        if not bool(getattr(native_backbone, "uses_positions", False)):
            raise ValueError("native backbone must consume electrode positions")
        if not bool(getattr(chsd_branch, "uses_positions", False)):
            raise ValueError("CHSD branch must consume electrode positions")

        native_metadata = _native_config(native_backbone)
        native_head = getattr(native_backbone, "final_layer", None)
        if not isinstance(native_head, nn.Linear):
            raise TypeError("native backbone final_layer must be linear")
        if not isinstance(chsd_branch.classifier, nn.Sequential):
            raise TypeError("CHSD classifier must be an nn.Sequential")
        residual_head = chsd_branch.classifier[-1]
        if not isinstance(residual_head, nn.Linear):
            raise TypeError("the final CHSD classifier module must be linear")
        if residual_head.out_features != native_head.out_features:
            raise ValueError("native and CHSD branches must have equal output counts")
        if not 0.0 < maximum_residual_scale <= 1.0:
            raise ValueError("maximum_residual_scale must lie in (0, 1]")

        self.native_backbone = native_backbone
        self.chsd_branch = chsd_branch
        self.requested_name = requested_name
        self.native_model_name = native_model_name
        self.n_outputs = int(residual_head.out_features)
        self.maximum_residual_scale = float(maximum_residual_scale)
        self.raw_residual_gate = nn.Parameter(torch.tensor(0.0))
        # This stable dictionary is captured by benchmark._architecture_identity.
        # Copies prevent the wrapper from mutating either child configuration.
        self.config = {
            "architecture": HYBRID_MODEL_NAME,
            "native_model": NATIVE_MODEL_NAME,
            "native_class": (
                f"{type(native_backbone).__module__}."
                f"{type(native_backbone).__qualname__}"
            ),
            "native_config": native_metadata,
            "residual_model": "chsdnet",
            "residual_class": (
                f"{type(chsd_branch).__module__}.{type(chsd_branch).__qualname__}"
            ),
            "residual_config": asdict(chsd_branch.config),
            "combination": "native_logits_plus_chsd_logits",
            "maximum_residual_scale": self.maximum_residual_scale,
            "zero_initialized_residual_gate": True,
            "bounded_residual_gate": "maximum_scale_times_tanh",
        }

    @property
    def residual_head(self) -> nn.Linear:
        head = self.chsd_branch.classifier[-1]
        if not isinstance(head, nn.Linear):  # defensive against later mutation
            raise TypeError("the final CHSD classifier module is no longer linear")
        return head

    def residual_gate(self) -> Tensor:
        return self.maximum_residual_scale * torch.tanh(self.raw_residual_gate)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        if channel_mask is not None:
            raise ValueError(
                "CHSDResidualNet does not yet define masked virtual-sensor "
                "projection; pass unpadded trials without channel_mask"
            )
        native_logits = self.native_backbone(x, positions)
        residual_logits = self.chsd_branch(x, positions)
        if native_logits.shape != residual_logits.shape:
            raise RuntimeError("native and CHSD branches returned different shapes")
        return native_logits + self.residual_gate() * residual_logits


def _validate_factory_metadata(
    *,
    requested_model: str,
    n_channels: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
    channel_names: tuple[str, ...],
    channel_positions: Tensor,
) -> Tensor:
    if requested_model != HYBRID_MODEL_NAME:
        raise ValueError(f"unsupported requested model {requested_model!r}")
    if n_channels <= 0 or n_times <= 1:
        raise ValueError("n_channels and n_times must describe a nonempty trial")
    if n_outputs not in (2, 4):
        raise ValueError("CHSDResidualNet supports exactly two or four outputs")
    if float(sfreq) != 128.0:
        raise ValueError("CHSDResidualNet is locked to 128 Hz")
    if len(channel_names) != n_channels:
        raise ValueError("channel_names length must equal n_channels")
    if any(not isinstance(name, str) or not name.strip() for name in channel_names):
        raise ValueError("channel_names must contain nonempty strings")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("channel_names must be unique")
    positions = torch.as_tensor(channel_positions, dtype=torch.float32)
    if positions.shape != (n_channels, 3):
        raise ValueError("channel_positions must have shape (n_channels, 3)")
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("channel_positions must be finite")
    norms = torch.linalg.vector_norm(positions, dim=-1)
    if not bool((norms > 1e-6).all()):
        raise ValueError("channel_positions must contain nonzero vectors")
    return positions


def make_chsd_hybrid_model(
    *,
    requested_model: str,
    n_channels: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
    channel_names: tuple[str, ...],
    channel_positions: Tensor,
) -> CHSDResidualNet:
    """Construct the experimental hybrid through the full-grid factory contract."""

    positions = _validate_factory_metadata(
        requested_model=requested_model,
        n_channels=n_channels,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
        channel_names=channel_names,
        channel_positions=channel_positions,
    )
    native = baselines.make_model(
        NATIVE_MODEL_NAME,
        n_channels=n_channels,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=float(sfreq),
        channel_names=channel_names,
        channel_positions=positions,
    )
    # Validate the factory result before constructing or mutating the residual.
    _native_config(native)
    if not isinstance(native, CardinalFBCMicroDynamicsNet):
        raise TypeError(
            "baseline factory did not return CardinalFBCMicroDynamicsNet for "
            f"{NATIVE_MODEL_NAME!r}"
        )
    chsd = CHSDNet(
        n_outputs,
        config=CHSDConfig(
            sfreq=float(sfreq),
            n_virtual_sensors=n_channels,
            use_coordinate_projection=False,
            matrix_log_terms=64,
        ),
    )
    return CHSDResidualNet(
        native,
        chsd,
        requested_name=requested_model,
        native_model_name=NATIVE_MODEL_NAME,
    )


__all__ = [
    "CHSDResidualNet",
    "HYBRID_MODEL_NAME",
    "NATIVE_MODEL_NAME",
    "make_chsd_hybrid_model",
]
