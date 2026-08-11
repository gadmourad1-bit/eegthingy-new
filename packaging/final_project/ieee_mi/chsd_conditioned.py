"""Surface-conditioned CardinalFBC with an exact native initialization.

This scratch architecture uses the native-montage Complex-Hermitian
Surface-Differential (CHSD) representation to condition the strong
``cardinal_fbc_micro_extended`` model.  It does not append CHSD coordinates
to the classifier and it does not add residual logits.  Instead, normalized
CHSD hidden summaries generate a small trial-specific multiplicative
modulation of the existing FBC log-variance floor:

``conditioned_floor = floor * (1 + m * tanh(conditioner(CHSD)))``.

The modulation is shared over FBC spatial sources but remains distinct for
each spectral band and temporal variance view.  The conditioner's final
linear layer is exactly zero at construction.  Consequently the complete
model, including its native continuation and original constrained head, is
exactly the native CardinalFBC function at initialization.  The final
conditioner layer receives a first-step gradient while all upstream CHSD
parameters initially receive a mathematically zero gradient.

The four HSD fields remain explicit in the conditioner in the fixed order
state, temporal difference, frequency difference, and mixed difference.
Every field is normalized before it enters the conditioning representation.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, Final

import torch
from torch import Tensor, nn

from . import baselines
from .chsd import CHSDConfig, CHSDNet
from .models import CardinalFBCMicroDynamicsNet


NATIVE_MODEL_NAME: Final = "cardinal_fbc_micro_extended"
CONDITIONED_MODEL_VARIANTS: Final = MappingProxyType(
    {
        "chsdnet_conditioned_005": 0.05,
        "chsdnet_conditioned_010": 0.10,
        "chsdnet_conditioned_020": 0.20,
    }
)
HSD_FIELD_NAMES: Final = (
    "state",
    "temporal_difference",
    "frequency_difference",
    "mixed_time_frequency_difference",
)


def _ordered_linear_interpolation_matrix(
    source_count: int,
    target_count: int,
) -> Tensor:
    """Return fixed convex weights for ordered endpoint-aligned resampling."""

    if source_count <= 0 or target_count <= 0:
        raise ValueError("interpolation dimensions must be positive")
    matrix = torch.zeros(target_count, source_count, dtype=torch.float32)
    if source_count == 1:
        matrix[:, 0] = 1.0
        return matrix
    if target_count == 1:
        matrix[0, 0] = 1.0
        return matrix
    for target_index in range(target_count):
        coordinate = target_index * (source_count - 1) / (target_count - 1)
        lower = math.floor(coordinate)
        upper = min(lower + 1, source_count - 1)
        fraction = coordinate - lower
        matrix[target_index, lower] = 1.0 - fraction
        matrix[target_index, upper] += fraction
    return matrix


def _validated_native_config(
    model: CardinalFBCMicroDynamicsNet,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if not isinstance(config, dict):
        raise TypeError("native backbone must expose dictionary metadata")
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


class CHSDSurfaceConditionedCardinalFBC(nn.Module):
    """Bounded CHSD conditioning of native FBC band-by-view features."""

    uses_positions = True

    def __init__(
        self,
        native_backbone: CardinalFBCMicroDynamicsNet,
        chsd_encoder: CHSDNet,
        *,
        requested_model: str,
        maximum_modulation: float,
        context_width: int = 16,
        conditioner_width: int = 32,
    ) -> None:
        super().__init__()
        expected_modulation = CONDITIONED_MODEL_VARIANTS.get(requested_model)
        if expected_modulation is None:
            raise ValueError(f"unsupported requested model {requested_model!r}")
        if not math.isclose(
            float(maximum_modulation),
            expected_modulation,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "maximum_modulation does not match the requested model variant"
            )
        if not isinstance(native_backbone, CardinalFBCMicroDynamicsNet):
            raise TypeError(
                "native_backbone must be CardinalFBCMicroDynamicsNet from "
                f"{NATIVE_MODEL_NAME}"
            )
        if not isinstance(chsd_encoder, CHSDNet):
            raise TypeError("chsd_encoder must be CHSDNet")
        if chsd_encoder.config.use_coordinate_projection:
            raise ValueError("the CHSD encoder must retain the native montage")
        if context_width <= 0 or conditioner_width <= 0:
            raise ValueError("conditioner widths must be positive")

        native_config = _validated_native_config(native_backbone)
        native_head = native_backbone.final_layer
        if not isinstance(native_head, nn.Linear):
            raise TypeError("native backbone must retain its linear final head")
        expected_native_features = (
            native_backbone.floor_feature_count
            + native_backbone.continuation_feature_count
        )
        if native_head.in_features != expected_native_features:
            raise ValueError("native head no longer matches floor plus continuation")
        expected_floor_features = (
            native_backbone.n_bands
            * native_backbone.n_sources
            * native_backbone.stride_factor
        )
        if native_backbone.floor_feature_count != expected_floor_features:
            raise ValueError("native FBC floor no longer has band/source/view layout")

        classifier = chsd_encoder.classifier
        if not isinstance(classifier, nn.Sequential) or not classifier:
            raise TypeError("CHSD encoder must expose its original classifier")
        summary_norm = classifier[0]
        if not isinstance(summary_norm, nn.LayerNorm):
            raise TypeError("cannot infer the normalized CHSD summary width")
        if len(summary_norm.normalized_shape) != 1:
            raise TypeError("CHSD summary must be one-dimensional per trial")
        chsd_summary_width = int(summary_norm.normalized_shape[0])
        if len(chsd_encoder.field_norms) != len(HSD_FIELD_NAMES):
            raise ValueError("CHSD encoder must expose exactly four HSD fields")
        if len(chsd_encoder.field_projections) != len(HSD_FIELD_NAMES):
            raise ValueError("CHSD encoder field projection count changed")

        # The task classifier is not part of the conditioning encoder.  Remove
        # it so the wrapper has no trainable parameters outside its forward map.
        chsd_encoder.classifier = nn.Identity()
        self.native_backbone = native_backbone
        self.chsd_encoder = chsd_encoder
        self.requested_model = str(requested_model)
        self.maximum_modulation = float(maximum_modulation)
        self.context_width = int(context_width)
        self.conditioner_width = int(conditioner_width)
        self.n_outputs = int(native_head.out_features)
        self.n_bands = int(native_backbone.n_bands)
        self.n_chsd_bands = len(chsd_encoder.config.bands)
        self.n_sources = int(native_backbone.n_sources)
        self.n_temporal_views = int(native_backbone.stride_factor)
        self.floor_feature_count = int(native_backbone.floor_feature_count)
        self.continuation_feature_count = int(
            native_backbone.continuation_feature_count
        )

        hsd_width = int(chsd_encoder.config.hsd_width)
        explicit_field_width = len(HSD_FIELD_NAMES) * hsd_width
        self.global_summary_norm = nn.LayerNorm(chsd_summary_width)
        self.global_context = nn.Linear(chsd_summary_width, self.context_width)
        self.explicit_fields_norm = nn.LayerNorm(explicit_field_width)
        self.conditioner_input_norm = nn.LayerNorm(
            explicit_field_width + self.context_width
        )
        self.conditioner_hidden = nn.Linear(
            explicit_field_width + self.context_width,
            self.conditioner_width,
        )
        self.conditioner_hidden_norm = nn.LayerNorm(self.conditioner_width)
        self.conditioner_head = nn.Linear(
            self.conditioner_width,
            self.n_temporal_views,
        )
        self.register_buffer(
            "band_interpolation_matrix",
            _ordered_linear_interpolation_matrix(
                self.n_chsd_bands,
                self.n_bands,
            ),
        )
        with torch.no_grad():
            self.conditioner_head.weight.zero_()
            self.conditioner_head.bias.zero_()

        self.config: dict[str, Any] = {
            "architecture": self.requested_model,
            "architecture_family": "chsd_surface_conditioned_cardinal_fbc",
            "native_model": NATIVE_MODEL_NAME,
            "native_class": (
                f"{type(native_backbone).__module__}."
                f"{type(native_backbone).__qualname__}"
            ),
            "native_config": native_config,
            "chsd_model": "chsdnet_native_montage_encoder",
            "chsd_class": (
                f"{type(chsd_encoder).__module__}."
                f"{type(chsd_encoder).__qualname__}"
            ),
            "chsd_config": asdict(chsd_encoder.config),
            "conditioner_fields": HSD_FIELD_NAMES,
            "conditioner_field_order_is_explicit": True,
            "conditioner_input_normalization": (
                "per_field_layer_norm_then_pooled_hidden_layer_norm"
            ),
            "conditioner_band_interpolation": {
                "method": "fixed_piecewise_linear_matrix",
                "endpoint_aligned": True,
                "source_count": self.n_chsd_bands,
                "target_count": self.n_bands,
                "matrix_buffer": "band_interpolation_matrix",
                "source_band_centers_hz": tuple(
                    (low + high) / 2.0
                    for low, high in chsd_encoder.config.bands
                ),
                # Each target coordinate is stated in fractional source-band
                # index units. This makes the exact deterministic mapping
                # auditable without pretending the wider HSD bands are the
                # same filters as the nine native FBC bands.
                "target_source_index_coordinates": tuple(
                    index * (self.n_chsd_bands - 1) / (self.n_bands - 1)
                    for index in range(self.n_bands)
                ),
            },
            "conditioned_feature_target": (
                "fbc_log_variance_band_by_temporal_view_shared_over_sources"
            ),
            "modulation_formula": "floor_times_one_plus_max_tanh_conditioner",
            "maximum_modulation": self.maximum_modulation,
            "zero_initialized_conditioner_head": True,
            "native_continuation_unchanged": True,
            "native_constrained_head_unchanged": True,
            "raw_chsd_concatenation": False,
            "residual_logit_branch": False,
            "n_bands": self.n_bands,
            "n_sources": self.n_sources,
            "n_temporal_views": self.n_temporal_views,
            "context_width": self.context_width,
            "conditioner_width": self.conditioner_width,
        }

    def _explicit_field_summary(
        self,
        features: dict[str, Tensor],
    ) -> Tensor:
        """Return normalized per-band summaries retaining all four fields."""

        fields = features["fields"]
        field_reliability = features["field_reliability"]
        if fields.ndim != 5 or fields.shape[-2] != len(HSD_FIELD_NAMES):
            raise RuntimeError("CHSD fields have an unexpected shape")
        if field_reliability.shape != fields.shape[:-2] + (
            len(HSD_FIELD_NAMES),
        ):
            raise RuntimeError("CHSD field reliability has an unexpected shape")

        summaries: list[Tensor] = []
        for field_index, (normalizer, projection) in enumerate(
            zip(
                self.chsd_encoder.field_norms,
                self.chsd_encoder.field_projections,
                strict=True,
            )
        ):
            # Raw HSD coordinates never enter the conditioner: each named
            # field is normalized before its learned hidden projection.
            field = normalizer(fields[..., field_index, :])
            hidden = nn.functional.gelu(projection(field))
            weights = field_reliability[..., field_index, None].to(hidden)
            numerator = (hidden * weights).sum(dim=1)
            denominator = weights.sum(dim=1).clamp_min(1e-7)
            summaries.append(numerator / denominator)

        # (B, HSD bands, state|dt|df|mixed hidden coordinates).
        explicit = torch.cat(summaries, dim=-1)
        explicit = self.explicit_fields_norm(explicit)
        if explicit.shape[1] != self.n_chsd_bands:
            raise RuntimeError("CHSD field summary band count changed")
        if self.n_chsd_bands == self.n_bands:
            return explicit
        # Preserve frequency order while mapping the six HSD summaries to the
        # nine native FBC band locations. A fixed convex matrix replaces
        # ``F.interpolate`` because the latter has no deterministic CUDA
        # backward for linear mode. CUDA GEMM is deterministic under the
        # benchmark's required CUBLAS_WORKSPACE_CONFIG.
        matrix = self.band_interpolation_matrix.to(explicit)
        return torch.matmul(
            explicit.transpose(1, 2),
            matrix.transpose(0, 1),
        ).transpose(1, 2)

    def conditioner_logits(self, x: Tensor, positions: Tensor) -> Tensor:
        """Return unbounded zero-started gates with shape ``(B, bands, views)``."""

        features = self.chsd_encoder.forward_features(x, positions)
        explicit = self._explicit_field_summary(features)
        global_summary = self.global_summary_norm(features["summary"])
        global_context = nn.functional.gelu(self.global_context(global_summary))
        global_context = global_context[:, None, :].expand(
            -1, self.n_bands, -1
        )
        hidden = torch.cat((explicit, global_context), dim=-1)
        hidden = self.conditioner_input_norm(hidden)
        hidden = nn.functional.gelu(self.conditioner_hidden(hidden))
        hidden = self.conditioner_hidden_norm(hidden)
        logits = self.conditioner_head(hidden)
        expected = (x.shape[0], self.n_bands, self.n_temporal_views)
        if logits.shape != expected:
            raise RuntimeError(
                f"conditioner returned shape {tuple(logits.shape)}; "
                f"expected {expected}"
            )
        return logits

    def modulation(self, x: Tensor, positions: Tensor) -> Tensor:
        """Return bounded fractional modulation in ``[-maximum, maximum]``."""

        return self.maximum_modulation * torch.tanh(
            self.conditioner_logits(x, positions)
        )

    def condition_floor(self, floor: Tensor, modulation: Tensor) -> Tensor:
        if floor.ndim != 2 or floor.shape[1] != self.floor_feature_count:
            raise ValueError("floor has an unexpected feature shape")
        expected_modulation = (
            floor.shape[0],
            self.n_bands,
            self.n_temporal_views,
        )
        if modulation.shape != expected_modulation:
            raise ValueError("modulation has an unexpected shape")
        structured = floor.reshape(
            floor.shape[0],
            self.n_bands,
            self.n_sources,
            self.n_temporal_views,
        )
        conditioned = structured * (1.0 + modulation[:, :, None, :])
        return conditioned.flatten(start_dim=1)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        channel_mask: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, time)")
        if channel_mask is not None:
            raise ValueError(
                "native-montage CHSD conditioning requires unpadded trials"
            )
        floor = self.native_backbone.encode_floor(x, positions)
        continuation = self.native_backbone.encode_continuation(x, positions)
        modulation = self.modulation(x, positions)
        conditioned_floor = self.condition_floor(floor, modulation)
        native_features = torch.cat((conditioned_floor, continuation), dim=1)
        return self.native_backbone.final_layer(native_features)


def make_chsd_conditioned_model(
    *,
    requested_model: str,
    n_channels: int,
    n_outputs: int,
    n_times: int,
    sfreq: float,
    channel_names: tuple[str, ...],
    channel_positions: Tensor,
) -> CHSDSurfaceConditionedCardinalFBC:
    """Construct one prespecified scratch conditioning-strength variant."""

    maximum_modulation = CONDITIONED_MODEL_VARIANTS.get(requested_model)
    if maximum_modulation is None:
        raise ValueError(f"unsupported requested model {requested_model!r}")
    if n_channels < 2 or n_times <= 1 or n_outputs not in (2, 4):
        raise ValueError("invalid channel/time/output dimensions")
    if len(channel_names) != n_channels or len(set(channel_names)) != n_channels:
        raise ValueError("channel_names must be unique and match n_channels")
    positions = torch.as_tensor(channel_positions, dtype=torch.float32)
    if positions.shape != (n_channels, 3):
        raise ValueError(
            "channel_positions must have shape (n_channels, 3)"
        )
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("channel_positions must be finite")

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
    chsd_encoder = CHSDNet(
        n_outputs,
        config=CHSDConfig(
            sfreq=float(sfreq),
            n_virtual_sensors=n_channels,
            use_coordinate_projection=False,
            matrix_log_terms=64,
        ),
    )
    return CHSDSurfaceConditionedCardinalFBC(
        native,
        chsd_encoder,
        requested_model=requested_model,
        maximum_modulation=maximum_modulation,
    )


__all__ = [
    "CONDITIONED_MODEL_VARIANTS",
    "HSD_FIELD_NAMES",
    "NATIVE_MODEL_NAME",
    "CHSDSurfaceConditionedCardinalFBC",
    "make_chsd_conditioned_model",
]
