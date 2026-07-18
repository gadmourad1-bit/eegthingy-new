"""Compact geometric network for filter-bank motor-imagery covariances."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from .spd import BiMap, LogEig, ReEig, SPDMomentumBatchNorm, matrix_log, upper_vectorize


@dataclass
class GeoAdaptOutput:
    """Structured network output consumed by training and online engines."""

    logits: Tensor
    features: Tensor
    anchor_logits: Tensor
    residual_logits: Tensor
    tangent_features: Tensor
    band_features: Tensor
    band_attention: Tensor
    residual_gate: Tensor
    intent_logit: Tensor | None = None
    log_reference: Tensor | None = None

    @property
    def log_odds(self) -> Tensor:
        """Signed class-2 versus class-1 log-odds used by online recentering."""

        return self.logits[..., 1] - self.logits[..., 0]

    @property
    def aux_logits(self) -> Tensor | None:
        """Compatibility alias for the separate intent/rest auxiliary logit."""

        return self.intent_logit

    def __getitem__(self, key: str) -> Tensor | None:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return (field.name for field in fields(self))

    def keys(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self))

    def as_dict(self) -> dict[str, Tensor | None]:
        return {key: getattr(self, key) for key in self.keys()}


class GeoAdaptNet(nn.Module):
    """Anchor-preserving SPD residual network.

    The anchor is a linear classifier over the *full* four-band tangent space,
    a strong and deliberately hard-to-break low-data baseline.  A learned
    bandwise ``BiMap -> ReEig -> LogEig`` path contributes only an additive
    residual.  Its sigmoid gate starts near zero, so optimization begins at the
    tangent classifier and cannot silently replace it with an untrained deep
    path.

    The main output always has two MI classes.  If ``auxiliary_intent=True``, a
    separate scalar intent-vs-rest logit is exposed in :class:`GeoAdaptOutput`;
    it never turns the main task into an artificial three-class problem.
    """

    def __init__(
        self,
        bands: int = 4,
        channels: int = 15,
        reduced_dim: int = 8,
        num_classes: int = 2,
        band_width: int = 24,
        fusion_width: int = 32,
        dropout: float = 0.25,
        eps: float = 1e-5,
        reference_momentum: float = 0.05,
        residual_gate_init: float = 0.02,
        auxiliary_intent: bool = False,
        *,
        n_bands: int | None = None,
        n_channels: int | None = None,
        d: int | None = None,
        with_intent_head: bool | None = None,
    ) -> None:
        super().__init__()
        # A few explicit aliases keep experiment configs readable without
        # accepting arbitrary/misspelled keyword arguments.
        bands = bands if n_bands is None else n_bands
        channels = channels if n_channels is None else n_channels
        reduced_dim = reduced_dim if d is None else d
        auxiliary_intent = auxiliary_intent if with_intent_head is None else with_intent_head
        if num_classes != 2:
            raise ValueError("GeoAdaptNet's main motor-imagery task is binary (num_classes=2)")
        if not 0 < reduced_dim <= channels:
            raise ValueError("reduced_dim must be in [1, channels]")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < residual_gate_init < 1.0:
            raise ValueError("residual_gate_init must be strictly between 0 and 1")

        self.bands = int(bands)
        self.channels = int(channels)
        self.reduced_dim = int(reduced_dim)
        self.num_classes = 2
        self.eps = float(eps)
        self.auxiliary_intent = bool(auxiliary_intent)

        tangent_per_band = channels * (channels + 1) // 2
        reduced_per_band = reduced_dim * (reduced_dim + 1) // 2
        self.tangent_dim = bands * tangent_per_band

        self.reference = SPDMomentumBatchNorm(
            bands,
            channels,
            momentum=reference_momentum,
            eps=eps,
            use_batch_stats=True,
        )

        # Strong full-resolution tangent-space anchor (no dimensionality
        # bottleneck and no nonlinear feature extractor).
        # Tangent coordinates differ greatly in variance (especially diagonal
        # versus off-diagonal terms).  Running standardization gives the linear
        # anchor the same essential conditioning as a classical
        # StandardScaler+logistic-regression pipeline without using target data.
        self.anchor_norm = nn.BatchNorm1d(
            self.tangent_dim, momentum=0.05, affine=False, track_running_stats=True
        )
        self.anchor_head = nn.Linear(self.tangent_dim, 2)

        self.bimaps = nn.ModuleList([BiMap(channels, reduced_dim, eps) for _ in range(bands)])
        self.reeig = ReEig(eps)
        self.logeig = LogEig(eps, vectorize=True)
        self.band_norms = nn.ModuleList([nn.LayerNorm(reduced_per_band) for _ in range(bands)])
        self.band_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(reduced_per_band, band_width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(bands)
            ]
        )
        self.band_embeddings = nn.Parameter(torch.empty(bands, band_width))
        self.attention_score = nn.Linear(band_width, 1, bias=False)

        fusion_input = band_width * (bands + 1)  # flattened bands + attention-weighted summary
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, fusion_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_width),
        )
        self.residual_head = nn.Linear(fusion_width, 2)
        self.intent_head = nn.Linear(fusion_width, 1) if auxiliary_intent else None

        gate_logit = torch.logit(torch.tensor(float(residual_gate_init)))
        self.residual_gate_logit = nn.Parameter(gate_logit)
        self.reset_parameters()

        if self.parameter_count >= 50_000:
            raise RuntimeError(
                f"GeoAdaptNet must stay below 50k parameters, got {self.parameter_count}"
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def residual_gate(self) -> Tensor:
        return self.residual_gate_logit.sigmoid()

    @property
    def spd_bn(self) -> SPDMomentumBatchNorm:
        """Compatibility name for the covariance reference layer."""

        return self.reference

    def reset_parameters(self) -> None:
        # A neutral linear classifier is substantially more stable than random
        # high-dimensional logits in the small-data regime.
        nn.init.normal_(self.anchor_head.weight, std=1e-3)
        nn.init.zeros_(self.anchor_head.bias)
        nn.init.normal_(self.band_embeddings, std=0.02)
        nn.init.xavier_uniform_(self.attention_score.weight)
        for encoder in self.band_encoders:
            nn.init.xavier_uniform_(encoder[0].weight)
            nn.init.zeros_(encoder[0].bias)
        nn.init.xavier_uniform_(self.fusion[1].weight)
        nn.init.zeros_(self.fusion[1].bias)
        # A tiny residual starts the model essentially at the tangent anchor,
        # while still allowing gradients into the complete residual branch.
        nn.init.normal_(self.residual_head.weight, std=1e-3)
        nn.init.zeros_(self.residual_head.bias)
        if self.intent_head is not None:
            nn.init.xavier_uniform_(self.intent_head.weight)
            nn.init.zeros_(self.intent_head.bias)

    @torch.no_grad()
    def retract_(self) -> "GeoAdaptNet":
        """Retract every BiMap parameter to its semi-orthogonal manifold."""

        for layer in self.bimaps:
            layer.retract_()
        return self

    def reset_running_stats(self) -> None:
        self.reference.reset_running_stats()

    @torch.no_grad()
    def set_log_reference(self, log_reference: Tensor) -> None:
        self.reference.set_reference(log_reference, is_log=True)

    def _validate_input(self, covariances: Tensor) -> None:
        expected = (self.bands, self.channels, self.channels)
        if covariances.ndim != 4 or covariances.shape[1:] != expected:
            raise ValueError(
                f"expected covariance input (N, {expected[0]}, {expected[1]}, "
                f"{expected[2]}), got {tuple(covariances.shape)}"
            )
        if not covariances.is_floating_point():
            raise TypeError("covariances must be a floating-point tensor")

    def forward(
        self,
        covariances: Tensor,
        log_reference: Tensor | Any | None = None,
        *,
        update_reference: bool | None = None,
    ) -> GeoAdaptOutput:
        self._validate_input(covariances)
        if log_reference is not None:
            log_reference = torch.as_tensor(
                log_reference,
                dtype=covariances.dtype,
                device=covariances.device,
            )

        aligned, used_reference = self.reference(
            covariances,
            log_reference,
            update_stats=update_reference,
            return_reference=True,
        )

        # Full tangent anchor branch.
        tangent_by_band = upper_vectorize(matrix_log(aligned, self.eps))
        tangent_features = tangent_by_band.flatten(start_dim=1)
        if self.training and tangent_features.shape[0] == 1:
            # BatchNorm cannot estimate a variance from a single item; this case
            # occurs in deployment-oriented gradient checks and tiny final batches.
            normalized_tangent = nn.functional.batch_norm(
                tangent_features,
                self.anchor_norm.running_mean,
                self.anchor_norm.running_var,
                training=False,
                momentum=0.0,
                eps=self.anchor_norm.eps,
            )
        else:
            normalized_tangent = self.anchor_norm(tangent_features)
        anchor_logits = self.anchor_head(normalized_tangent)

        # Learned, lower-rank geometric residual branch.
        encoded_bands: list[Tensor] = []
        for band in range(self.bands):
            reduced = self.bimaps[band](aligned[:, band])
            reduced = self.reeig(reduced)
            tangent = self.logeig(reduced)
            encoded = self.band_encoders[band](self.band_norms[band](tangent))
            encoded_bands.append(encoded)
        band_features = torch.stack(encoded_bands, dim=1)

        attention_input = band_features + self.band_embeddings.unsqueeze(0)
        band_attention = (
            self.attention_score(torch.tanh(attention_input)).squeeze(-1).softmax(dim=1)
        )
        attended = torch.sum(band_attention.unsqueeze(-1) * band_features, dim=1)
        fusion_input = torch.cat((band_features.flatten(start_dim=1), attended), dim=-1)
        features = self.fusion(fusion_input)
        residual_logits = self.residual_head(features)
        gate = self.residual_gate
        logits = anchor_logits + gate * residual_logits

        intent_logit = None if self.intent_head is None else self.intent_head(features).squeeze(-1)
        return GeoAdaptOutput(
            logits=logits,
            features=features,
            anchor_logits=anchor_logits,
            residual_logits=residual_logits,
            tangent_features=tangent_features,
            band_features=band_features,
            band_attention=band_attention,
            residual_gate=gate,
            intent_logit=intent_logit,
            log_reference=used_reference,
        )


__all__ = ["GeoAdaptNet", "GeoAdaptOutput"]
