"""GeoAdaptNet-FB: a learnable-filterbank Riemannian tangent network.

The fixed-band GeoAdaptNet (and, as measured, even classical Riemann-TS+LR) cannot
beat ShallowConvNet once a subject has enough data, because its 4 filter bands are
fixed while ShallowConvNet *learns* subject-specific temporal filters.  This variant
prepends a small, differentiable **Sinc bandpass filterbank** (SincNet, Ravanelli &
Bengio 2018 -- 2 parameters per band) to the *existing* SPD/tangent pipeline: learn K
bandpass filters, form one full 15x15 spatial covariance per band, then run the same
log-Euclidean recenter -> matrix-log -> tangent -> linear anchor.

It stays a geometric net, NOT a re-skinned ShallowConvNet: it keeps the full channel
covariance (all pairs, no learned spatial reduction), the matrix logarithm (couples
channels, not an elementwise square/log), and the SPD recenter-to-reference that
ShallowConvNet has no analogue for.  It sits in the TSMNet / Tensor-CSPNet family of
learnable-filter Riemannian nets and must be reported as a *distinct* architecture,
not under the fixed-band GeoAdaptNet's identity.  The bank is warm-started at the
proven fixed bands so it starts at that baseline and can only refine.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor, nn

from .config import BANDS, SFREQ
from .spd import BiMap, LogEig, ReEig, matrix_log, symmetrize, upper_vectorize


class SincFilterBank(nn.Module):
    """K learnable band-pass filters (SincNet parameterisation) applied per channel.

    Each band has two learnable parameters -- a low cut-off and a (positive)
    bandwidth -- so the strong band-pass inductive bias resists overfitting on
    small motor-imagery sets.  Filters are shared across EEG channels.
    """

    def __init__(
        self,
        n_bands: int,
        sfreq: float,
        kernel_size: int = 65,
        init_bands: Sequence[tuple[float, float]] | None = None,
        min_hz: float = 2.0,
        min_bw: float = 2.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # odd length keeps the filter zero-phase / centered
        self.n_bands = int(n_bands)
        self.sfreq = float(sfreq)
        self.kernel_size = int(kernel_size)
        self.min_hz = float(min_hz)
        self.min_bw = float(min_bw)

        if init_bands is None:
            nyquist = sfreq / 2.0
            edges = np.linspace(4.0, min(40.0, nyquist - 1.0), n_bands + 1)
            init_bands = [(edges[i], edges[i + 1]) for i in range(n_bands)]
        lows = np.array([lo for lo, _ in init_bands], dtype=np.float32)
        highs = np.array([hi for _, hi in init_bands], dtype=np.float32)
        self.low_hz_ = nn.Parameter(torch.from_numpy(lows))
        self.band_hz_ = nn.Parameter(torch.from_numpy(np.maximum(highs - lows, min_bw)))

        half = (self.kernel_size - 1) // 2
        n = torch.arange(-half, half + 1, dtype=torch.float32) / sfreq
        self.register_buffer("time_", n.view(1, -1))  # (1, kernel_size)
        window = 0.54 - 0.46 * torch.cos(
            2 * math.pi * torch.arange(self.kernel_size) / (self.kernel_size - 1)
        )
        self.register_buffer("window_", window.view(1, -1))

    def _band_pass(self) -> Tensor:
        low = self.min_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_bw + torch.abs(self.band_hz_), max=self.sfreq / 2.0)
        t = self.time_  # (1, K_t)
        # A band-pass filter is the difference of two low-pass sinc filters.
        low_pass_high = 2 * high.view(-1, 1) * _sinc(2 * high.view(-1, 1) * t * math.pi)
        low_pass_low = 2 * low.view(-1, 1) * _sinc(2 * low.view(-1, 1) * t * math.pi)
        band = (low_pass_high - low_pass_low) * self.window_
        band = band / (band.norm(dim=1, keepdim=True) + 1e-8)
        return band.unsqueeze(1)  # (n_bands, 1, kernel_size)

    def forward(self, epochs: Tensor) -> Tensor:
        # epochs (N, C, T) -> (N, n_bands, C, T)
        n, channels, samples = epochs.shape
        filters = self._band_pass().to(epochs.dtype)
        flat = epochs.reshape(n * channels, 1, samples)
        filtered = nn.functional.conv1d(flat, filters, padding=(self.kernel_size - 1) // 2)
        filtered = filtered.reshape(n, channels, self.n_bands, samples)
        return filtered.permute(0, 2, 1, 3).contiguous()


def _sinc(x: Tensor) -> Tensor:
    # Use a safe denominator so the x=0 branch never produces a 0/0 that would
    # back-propagate a NaN gradient through torch.where (the center filter tap is
    # exactly t=0, so this fires on every forward).
    x_safe = torch.where(x.abs() < 1e-7, torch.ones_like(x), x)
    return torch.where(x.abs() < 1e-7, torch.ones_like(x), torch.sin(x) / x_safe)


def _band_covariances(signals: Tensor, shrinkage: float = 0.05) -> Tensor:
    """Differentiable per-band spatial covariances (N, K, C, C), SPD-regularized.

    A learnable narrow band can drive a covariance near rank-1, which makes
    ``eigh`` fail to converge on CUDA.  Each matrix is normalized to unit trace
    and then shrunk toward the isotropic ``I/C``: the eigenvalues then lie in
    ``[shrinkage/C, 1]``, hard-bounding the condition number so the SPD backbone
    is always numerically stable regardless of where the filters move.
    """

    centered = signals - signals.mean(dim=-1, keepdim=True)
    samples = centered.shape[-1]
    cov = symmetrize(centered @ centered.transpose(-1, -2) / samples)
    channels = cov.shape[-1]
    trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    cov = cov / trace.clamp_min(torch.finfo(cov.dtype).tiny)
    identity = torch.eye(channels, dtype=cov.dtype, device=cov.device) / channels
    cov = (1.0 - shrinkage) * cov + shrinkage * identity
    # A tiny deterministic diagonal ramp separates otherwise-clustered eigenvalues
    # of near-isotropic bands, which CUDA's eigh (syevd) cannot converge on.
    ramp = torch.linspace(0.0, 0.02, channels, device=cov.device, dtype=cov.dtype)
    return cov + torch.diag(ramp)


class FilterBankSPDNet(nn.Module):
    """Learnable Sinc filterbank -> per-band covariance -> tangent anchor -> 2 logits."""

    def __init__(
        self,
        channels: int = 15,
        n_bands: int = 4,
        sfreq: float = SFREQ,
        kernel_size: int = 65,
        init_bands: Sequence[tuple[float, float]] | None = None,
        eps: float = 1e-5,
        dropout: float = 0.25,
        reduced_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.n_bands = int(n_bands)
        self.eps = float(eps)
        self.filterbank = SincFilterBank(n_bands, sfreq, kernel_size, init_bands)
        # Optional learned spatial reduction: a per-band BiMap projects each 15x15
        # covariance onto a discriminative reduced_dim subspace (end-to-end CSP),
        # trained jointly with the filterbank -- the learned-spatial mechanism the
        # full-covariance tangent path lacks.
        self.reduced_dim = channels if reduced_dim is None else int(reduced_dim)
        if self.reduced_dim < channels:
            self.bimaps = nn.ModuleList(
                [BiMap(channels, self.reduced_dim, eps) for _ in range(n_bands)]
            )
            self.reeig = ReEig(eps)
            self.logeig = LogEig(eps, vectorize=True)
            tangent_dim = n_bands * self.reduced_dim * (self.reduced_dim + 1) // 2
        else:
            self.bimaps = None
            tangent_dim = n_bands * channels * (channels + 1) // 2
        self.anchor_norm = nn.BatchNorm1d(tangent_dim, momentum=0.05, affine=False)
        self.dropout = nn.Dropout(dropout)
        self.anchor_head = nn.Linear(tangent_dim, 2)
        nn.init.normal_(self.anchor_head.weight, std=1e-3)
        nn.init.zeros_(self.anchor_head.bias)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def retract_(self) -> "FilterBankSPDNet":
        if self.bimaps is not None:
            for layer in self.bimaps:
                layer.retract_()
        return self

    def forward(self, epochs: Tensor) -> Tensor:
        signals = self.filterbank(epochs)
        covariances = _band_covariances(signals)
        if self.bimaps is not None:
            # Learned spatial reduction per band, then log-Euclidean tangent.
            bands = []
            for k in range(self.n_bands):
                reduced = self.reeig(self.bimaps[k](covariances[:, k]))
                bands.append(self.logeig(reduced))
            tangent = torch.cat(bands, dim=1)
        else:
            # Full-covariance log-Euclidean tangent at the identity.
            tangent = upper_vectorize(matrix_log(covariances, self.eps)).flatten(start_dim=1)
        if self.training and tangent.shape[0] == 1:
            tangent = nn.functional.batch_norm(
                tangent, self.anchor_norm.running_mean, self.anchor_norm.running_var,
                training=False, momentum=0.0, eps=self.anchor_norm.eps,
            )
        else:
            tangent = self.anchor_norm(tangent)
        return self.anchor_head(self.dropout(tangent))


class FilterBankSPDClassifier:
    """Fit/predict wrapper (broadband epochs in) matching the benchmark interface."""

    def __init__(
        self,
        n_bands: int = 4,
        sfreq: float = SFREQ,
        *,
        n_epochs: int = 250,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        batch_size: int = 64,
        patience: int = 40,
        seed: int = 7,
        device: str = "cuda",
        warm_start_bands: bool = True,
        lr_swap_index: Sequence[int] | None = None,
        lr_swap_prob: float = 0.0,
        reduced_dim: int | None = None,
    ) -> None:
        self.reduced_dim = reduced_dim
        self.n_bands = int(n_bands)
        self.sfreq = float(sfreq)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.seed = int(seed)
        self.device = device
        self.warm_start_bands = bool(warm_start_bands)
        self.lr_swap_index = list(lr_swap_index) if lr_swap_index is not None else None
        self.lr_swap_prob = float(lr_swap_prob)

    def _resolve_device(self) -> torch.device:
        if self.device in (None, "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(self, x_tr, y_tr, x_val, y_val) -> "FilterBankSPDClassifier":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._resolve_device()
        self.device_ = device
        channels = x_tr.shape[1]
        init = list(BANDS) if (self.warm_start_bands and self.n_bands == len(BANDS)) else None
        model = FilterBankSPDNet(
            channels, self.n_bands, self.sfreq, init_bands=init, reduced_dim=self.reduced_dim
        ).to(device)
        self.param_count_ = model.parameter_count

        # Per-channel standardization (train stats only) keeps the learnable-band
        # covariances well-conditioned; raw EEG has ~1e-10 min eigenvalues that
        # otherwise break eigh once the filters move during training.
        x_tr = np.asarray(x_tr, dtype=np.float32)
        self.mean_ = x_tr.mean(axis=(0, 2), keepdims=True)
        self.std_ = x_tr.std(axis=(0, 2), keepdims=True) + 1e-6
        x_tr = (x_tr - self.mean_) / self.std_
        x_val = (np.asarray(x_val, dtype=np.float32) - self.mean_) / self.std_

        x_train = torch.as_tensor(np.asarray(x_tr, dtype=np.float32), device=device)
        y_train = torch.as_tensor(np.asarray(y_tr), dtype=torch.long, device=device)
        x_valid = torch.as_tensor(np.asarray(x_val, dtype=np.float32), device=device)
        y_valid = torch.as_tensor(np.asarray(y_val), dtype=torch.long, device=device)
        swap = (
            torch.as_tensor(self.lr_swap_index, dtype=torch.long, device=device)
            if self.lr_swap_index is not None and self.lr_swap_prob > 0
            else None
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        generator = torch.Generator(device=device).manual_seed(self.seed)
        best_loss, best_state, stale = float("inf"), None, 0
        started = time.time()
        for epoch in range(self.n_epochs):
            model.train()
            order = torch.randperm(len(x_train), device=device, generator=generator)
            for start in range(0, len(x_train), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb, yb = x_train[idx], y_train[idx]
                if swap is not None:
                    do = torch.rand(len(xb), device=device, generator=generator) < self.lr_swap_prob
                    if torch.any(do):
                        xb = xb.clone(); xb[do] = xb[do][:, swap, :]
                        yb = yb.clone(); yb[do] = 1 - yb[do]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(x_valid), y_valid))
            if val_loss < best_loss - 1e-4:
                best_loss, stale = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.fit_seconds_ = time.time() - started
        return self

    def predict_proba(self, x) -> np.ndarray:
        self.model_.eval()
        x = (np.asarray(x, dtype=np.float32) - self.mean_) / self.std_
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device_)
        out = []
        with torch.no_grad():
            for start in range(0, len(xt), 256):
                out.append(torch.softmax(self.model_(xt[start : start + 256]), dim=1).cpu())
        return torch.cat(out).numpy()


__all__ = ["FilterBankSPDClassifier", "FilterBankSPDNet", "SincFilterBank"]
