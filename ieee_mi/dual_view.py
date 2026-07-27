"""Bounded dual-view continuation for native-montage CardinalFBC.

The module in this file is deliberately isolated from the registered benchmark
and confirmation runners.  It is a research candidate, not a selected model.
It combines two deterministic evaluations of one shared CardinalFBC backbone:

``direct``
    The learned cardinal scalp field is sampled at the native electrodes.

``canonical``
    A geometry-only spherical-spline matrix transports raw native voltages to
    the fixed inducing atlas.  That view receives its own train-partition-only
    channel standardization before entering this module, then the same learned
    field is evaluated at the inducing electrodes.

The separate input tensors are necessary for an exact comparison: channelwise
standardization and clipping do not commute with spherical-spline transport.
The two views share every trainable encoder and classifier parameter.  A tiny
gate observes bounded summaries of their FBC feature disagreement and mixes
their logits.  Its final layer is initialized to zero, so the initial model is
exactly the internal shared-backbone 50/50 logit fusion control.  That identity
does not equate it to a post-hoc fusion of two models fine-tuned independently.
The existing native-transfer spline implementation owns and validates the
geometry-only preprocessing; this module accepts its two scaled outputs
without fitting anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .models import CardinalFBCNet


@dataclass(frozen=True)
class CardinalSplineDualViewOutput:
    """Outputs needed by the fused and auxiliary train-only objectives."""

    fused_logits: Tensor
    direct_logits: Tensor
    canonical_logits: Tensor
    canonical_weight: Tensor
    disagreement_descriptor: Tensor


@dataclass(frozen=True)
class CardinalSplineDualViewLoss:
    """Scalar components of the predeclared dual-view objective."""

    total: Tensor
    fused_cross_entropy: Tensor
    auxiliary_view_cross_entropy: Tensor
    preference_cross_entropy: Tensor


class CardinalSplineDualViewNet(nn.Module):
    """Shared CardinalFBC with native and spline-canonical spatial views.

    ``canonical_x`` must have the anchor channel order stored by the Cardinal
    field.  It is formed by spline-transporting raw voltages *before* fitting
    and applying the canonical view's source-partition-only channel scaler.
    The primary 21-anchor model adds only 233 parameters to a binary CardinalFBC
    backbone when ``gate_hidden=8``:

    ``Linear(27, 8) + GELU + Linear(8, 1)``.

    The gate descriptor contains, for each of the nine FBC bands, the signed
    mean, mean absolute value, and root-mean-square of the canonical-minus-
    direct log-variance feature difference over sources and temporal views.
    The descriptor is detached before the gate.  This prevents the preference
    loss from teaching the shared encoder to manufacture view disagreement;
    the fused and per-view classification losses still train the encoder.
    """

    uses_positions = True
    uses_canonical_view = True

    def __init__(
        self,
        backbone: CardinalFBCNet,
        *,
        gate_hidden: int = 8,
        descriptor_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, CardinalFBCNet):
            raise TypeError("backbone must be a CardinalFBCNet")
        if gate_hidden <= 0:
            raise ValueError("gate_hidden must be positive")
        if not descriptor_scale > 0.0:
            raise ValueError("descriptor_scale must be positive")
        expected_floor_features = (
            backbone.n_bands * backbone.n_sources * backbone.stride_factor
        )
        if int(backbone.final_layer.in_features) != expected_floor_features:
            raise ValueError(
                "backbone must have the unextended CardinalFBC feature head"
            )

        self.backbone = backbone
        self.gate_hidden = int(gate_hidden)
        self.descriptor_scale = float(descriptor_scale)
        descriptor_features = 3 * backbone.n_bands
        self.preference_gate = nn.Sequential(
            nn.Linear(descriptor_features, self.gate_hidden),
            nn.GELU(),
            nn.Linear(self.gate_hidden, 1),
        )
        # Exactly recover fixed equal-logit fusion at initialization.  The
        # nonzero derivative of sigmoid at zero lets the selector learn on the
        # first optimization step.
        final_gate = self.preference_gate[-1]
        assert isinstance(final_gate, nn.Linear)
        nn.init.zeros_(final_gate.weight)
        nn.init.zeros_(final_gate.bias)
        self.config = {
            "candidate_status": "bounded_development_prototype",
            "backbone": "shared_cardinal_fbc",
            "views": (
                "native_cardinal_field",
                "spherical_spline_to_anchor_indexed_field",
            ),
            "temporal_filter_evaluations": 2,
            "gate_hidden": self.gate_hidden,
            "gate_descriptor_features": descriptor_features,
            "gate_descriptor_scale": self.descriptor_scale,
            "gate_descriptor_stop_gradient": True,
            "gate_initial_canonical_weight": 0.5,
            "spline_transport_outside_trainable_graph": True,
            "separate_train_partition_scaler_per_view": True,
            "shared_batch_norm_across_views": True,
            "shared_classifier_across_views": True,
        }

    @property
    def added_parameter_count(self) -> int:
        """Number of trainable parameters beyond the shared backbone."""

        return sum(parameter.numel() for parameter in self.preference_gate.parameters())

    def _validate_inputs(
        self,
        native_x: Tensor,
        canonical_x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None,
    ) -> None:
        if native_x.ndim != 3:
            raise ValueError(
                "native_x must have shape (batch, native_channels, time)"
            )
        expected_canonical = (
            native_x.shape[0],
            self.backbone.spatial_field.anchors.shape[0],
            native_x.shape[2],
        )
        if canonical_x.shape != expected_canonical:
            raise ValueError(
                f"canonical_x has shape {tuple(canonical_x.shape)}; "
                f"expected {expected_canonical}"
            )
        if positions.shape != (native_x.shape[1], 3):
            raise ValueError("positions do not match the native channel axis")
        if channel_mask is not None and channel_mask.shape != native_x.shape[:2]:
            raise ValueError("channel_mask must have shape (batch, channels)")
        if native_x.device != canonical_x.device:
            raise ValueError("native and canonical views must be on the same device")
        if native_x.dtype != canonical_x.dtype:
            raise ValueError("native and canonical views must have the same dtype")
        if not bool(torch.isfinite(native_x).all()) or not bool(
            torch.isfinite(canonical_x).all()
        ):
            raise ValueError("dual-view inputs contain non-finite values")

    def _paired_sources(
        self,
        native_x: Tensor,
        canonical_x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        self._validate_inputs(native_x, canonical_x, positions, channel_mask)
        native_filtered = self.backbone._apply_shared_spectral_filter(native_x)
        canonical_filtered = self.backbone._apply_shared_spectral_filter(canonical_x)
        if (
            native_filtered.ndim != 4
            or canonical_filtered.ndim != 4
            or native_filtered.shape[1] != self.backbone.n_bands
            or canonical_filtered.shape[1] != self.backbone.n_bands
        ):
            raise RuntimeError(
                "spectral filter returned an unexpected dual-view shape"
            )
        direct = self.backbone.spatial_field(
            native_filtered.permute(0, 2, 1, 3), positions, channel_mask
        )
        anchor_positions = self.backbone.spatial_field.anchors.to(
            device=canonical_x.device, dtype=positions.dtype
        )
        canonical = self.backbone.spatial_field(
            canonical_filtered.permute(0, 2, 1, 3), anchor_positions
        )
        bias = self.backbone.spatial_bias[None, :, :, None].permute(0, 2, 1, 3)
        return direct + bias, canonical + bias

    def _paired_floor_features(
        self,
        direct_sources: Tensor,
        canonical_sources: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if direct_sources.shape != canonical_sources.shape:
            raise RuntimeError("dual-view source tensors have different shapes")
        batch, sources, bands, n_times = direct_sources.shape
        if bands != self.backbone.n_bands or sources != self.backbone.n_sources:
            raise RuntimeError("dual-view source tensors have stale dimensions")

        def flatten_sources(values: Tensor) -> Tensor:
            return values.permute(0, 2, 1, 3).reshape(
                batch, bands * sources, 1, n_times
            )

        # One joint BN call makes the running-stat update invariant to whether
        # the direct or canonical view is encoded first.
        paired = torch.cat(
            (flatten_sources(direct_sources), flatten_sources(canonical_sources)),
            dim=0,
        )
        paired = self.backbone.activation(self.backbone.batch_norm(paired))
        paired = self.backbone.padding_layer(paired)
        n_times_padded = n_times + (-n_times % self.backbone.stride_factor)
        if paired.shape[-1] != n_times_padded:
            raise RuntimeError("FBC padding layer returned an unexpected time length")
        paired = paired.reshape(
            2 * batch,
            bands * sources,
            self.backbone.stride_factor,
            n_times_padded // self.backbone.stride_factor,
        )
        paired = self.backbone.flatten_layer(
            self.backbone.temporal_layer(paired)
        )
        return paired[:batch], paired[batch:]

    def _disagreement_descriptor(
        self,
        direct_features: Tensor,
        canonical_features: Tensor,
    ) -> Tensor:
        batch = direct_features.shape[0]
        expected = (
            self.backbone.n_bands
            * self.backbone.n_sources
            * self.backbone.stride_factor
        )
        if direct_features.shape != (batch, expected):
            raise RuntimeError("direct FBC features have an unexpected shape")
        if canonical_features.shape != direct_features.shape:
            raise RuntimeError("canonical FBC features have an unexpected shape")
        difference = (canonical_features - direct_features).reshape(
            batch,
            self.backbone.n_bands,
            self.backbone.n_sources,
            self.backbone.stride_factor,
        )
        dimensions = (2, 3)
        signed_mean = difference.mean(dim=dimensions)
        absolute_mean = difference.abs().mean(dim=dimensions)
        root_mean_square = torch.sqrt(
            difference.square().mean(dim=dimensions).clamp_min(1e-12)
        )
        descriptor = torch.cat(
            (signed_mean, absolute_mean, root_mean_square), dim=1
        )
        return torch.tanh(descriptor / self.descriptor_scale)

    def forward_views(
        self,
        native_x: Tensor,
        canonical_x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> CardinalSplineDualViewOutput:
        """Return fused logits and auditable view-specific intermediates."""

        direct_sources, canonical_sources = self._paired_sources(
            native_x, canonical_x, positions, channel_mask
        )
        direct_features, canonical_features = self._paired_floor_features(
            direct_sources, canonical_sources
        )
        direct_logits = self.backbone.final_layer(direct_features)
        canonical_logits = self.backbone.final_layer(canonical_features)
        descriptor = self._disagreement_descriptor(
            direct_features, canonical_features
        )
        canonical_weight = torch.sigmoid(
            self.preference_gate(descriptor.detach())
        )
        fused_logits = direct_logits + canonical_weight * (
            canonical_logits - direct_logits
        )
        return CardinalSplineDualViewOutput(
            fused_logits=fused_logits,
            direct_logits=direct_logits,
            canonical_logits=canonical_logits,
            canonical_weight=canonical_weight,
            disagreement_descriptor=descriptor,
        )

    def forward(
        self,
        native_x: Tensor,
        canonical_x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        return self.forward_views(
            native_x, canonical_x, positions, channel_mask
        ).fused_logits


def cardinal_spline_dual_view_loss(
    output: CardinalSplineDualViewOutput,
    labels: Tensor,
    *,
    auxiliary_view_weight: float = 0.25,
    preference_weight: float = 0.05,
    preference_temperature: float = 0.25,
    label_smoothing: float = 0.05,
) -> CardinalSplineDualViewLoss:
    """Compute the one fixed training objective proposed for the candidate.

    The soft preference target is one when the canonical view assigns lower
    train-row negative log-likelihood than the direct view.  View losses are
    detached when constructing that target, so the selector cannot improve its
    target by degrading either branch.  This function is for train rows only;
    model selection must use fused validation cross-entropy and test labels
    must never be passed here.
    """

    if auxiliary_view_weight < 0.0 or preference_weight < 0.0:
        raise ValueError("loss weights must be nonnegative")
    if preference_temperature <= 0.0:
        raise ValueError("preference_temperature must be positive")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must lie in [0, 1)")
    if labels.ndim != 1 or labels.shape[0] != output.fused_logits.shape[0]:
        raise ValueError("labels must have shape (batch,)")
    fused = nn.functional.cross_entropy(
        output.fused_logits, labels, label_smoothing=label_smoothing
    )
    direct_rows = nn.functional.cross_entropy(
        output.direct_logits,
        labels,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    canonical_rows = nn.functional.cross_entropy(
        output.canonical_logits,
        labels,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    auxiliary = 0.5 * (direct_rows.mean() + canonical_rows.mean())
    soft_preference = torch.sigmoid(
        (direct_rows.detach() - canonical_rows.detach())
        / preference_temperature
    )
    preference = nn.functional.binary_cross_entropy(
        output.canonical_weight.squeeze(1), soft_preference
    )
    total = (
        fused
        + auxiliary_view_weight * auxiliary
        + preference_weight * preference
    )
    return CardinalSplineDualViewLoss(
        total=total,
        fused_cross_entropy=fused,
        auxiliary_view_cross_entropy=auxiliary,
        preference_cross_entropy=preference,
    )
