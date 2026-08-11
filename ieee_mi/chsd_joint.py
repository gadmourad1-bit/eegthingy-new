"""Function-preserving joint-head fusion of CardinalFBC and native CHSD.

The strong coordinate-continuous filter-bank backbone keeps its exact
initial mapping. Native-montage CHSD summary coordinates are appended to the
same constrained linear classifier as zero columns, so their contribution is
learned only when supported by source/validation evidence.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor, nn

from . import baselines
from .chsd import CHSDConfig, CHSDNet
from .models import (
    CardinalFBCMicroDynamicsNet,
    ExpandedMaxNormLinear,
)


JOINT_MODEL_NAME = "chsdnet_joint"
NATIVE_MODEL_NAME = "cardinal_fbc_micro_extended"


class CHSDJointNet(nn.Module):
    """One zero-extended head over native FBC and CHSD summary features."""

    uses_positions = True

    def __init__(
        self,
        native_backbone: CardinalFBCMicroDynamicsNet,
        chsd_encoder: CHSDNet,
    ) -> None:
        super().__init__()
        if not isinstance(native_backbone, CardinalFBCMicroDynamicsNet):
            raise TypeError("native_backbone must be CardinalFBCMicroDynamicsNet")
        if not bool(native_backbone.config.get("extended_atlas")):
            raise ValueError("native backbone must use the extended atlas")
        if not isinstance(chsd_encoder.classifier, nn.Sequential):
            raise TypeError("CHSD encoder must still expose its classifier")
        first = chsd_encoder.classifier[0]
        if not isinstance(first, nn.LayerNorm) or len(first.normalized_shape) != 1:
            raise TypeError("cannot infer the CHSD summary dimension")
        self.chsd_feature_count = int(first.normalized_shape[0])
        self.native_feature_count = int(native_backbone.final_layer.in_features)
        native_backbone.final_layer = ExpandedMaxNormLinear.extend(
            native_backbone.final_layer,
            self.chsd_feature_count,
        )
        # CHSDNet.forward now returns its pre-classifier summary. Removing the
        # unused private head avoids optimizing parameters outside the model
        # mapping; its summary-producing representation is unchanged.
        chsd_encoder.classifier = nn.Identity()
        self.native_backbone = native_backbone
        self.chsd_encoder = chsd_encoder
        self.n_outputs = int(native_backbone.final_layer.out_features)
        self.config: dict[str, Any] = {
            "architecture": JOINT_MODEL_NAME,
            "native_model": NATIVE_MODEL_NAME,
            "native_config": dict(native_backbone.config),
            "chsd_config": asdict(chsd_encoder.config),
            "native_feature_count": self.native_feature_count,
            "chsd_feature_count": self.chsd_feature_count,
            "single_zero_extended_constrained_head": True,
        }

    def encode_native(self, x: Tensor, positions: Tensor) -> Tensor:
        floor = self.native_backbone.encode_floor(x, positions)
        continuation = self.native_backbone.encode_continuation(x, positions)
        values = torch.cat((floor, continuation), dim=1)
        if values.shape[1] != self.native_feature_count:
            raise RuntimeError("native feature count changed")
        return values

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        native = self.encode_native(x, positions)
        chsd = self.chsd_encoder(x, positions)
        if chsd.ndim != 2 or chsd.shape[1] != self.chsd_feature_count:
            raise RuntimeError("CHSD encoder returned an invalid summary")
        return self.native_backbone.final_layer(torch.cat((native, chsd), dim=1))


def make_chsd_joint_model(
    *,
    requested_model: str,
    n_channels: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
    channel_names: tuple[str, ...],
    channel_positions: Tensor,
) -> CHSDJointNet:
    """Construct joint-head CHSD through the grid factory contract."""

    if requested_model != JOINT_MODEL_NAME:
        raise ValueError(f"unsupported requested model {requested_model!r}")
    if n_channels <= 0 or n_times <= 1 or n_outputs not in (2, 4):
        raise ValueError("invalid channel/time/output dimensions")
    if len(channel_names) != n_channels or len(set(channel_names)) != n_channels:
        raise ValueError("channel_names must be unique and match n_channels")
    positions = torch.as_tensor(channel_positions, dtype=torch.float32)
    if positions.shape != (n_channels, 3) or not bool(torch.isfinite(positions).all()):
        raise ValueError("channel_positions must be finite with shape (n_channels, 3)")
    native = baselines.make_model(
        NATIVE_MODEL_NAME,
        n_channels=n_channels,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=float(sfreq),
        channel_names=channel_names,
        channel_positions=positions,
    )
    if not isinstance(native, CardinalFBCMicroDynamicsNet):
        raise TypeError("baseline factory returned the wrong native model type")
    chsd = CHSDNet(
        n_outputs,
        config=CHSDConfig(
            sfreq=float(sfreq),
            n_virtual_sensors=n_channels,
            use_coordinate_projection=False,
            matrix_log_terms=64,
        ),
    )
    return CHSDJointNet(native, chsd)


__all__ = [
    "CHSDJointNet",
    "JOINT_MODEL_NAME",
    "NATIVE_MODEL_NAME",
    "make_chsd_joint_model",
]
