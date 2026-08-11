"""Differentiable building blocks for symmetric positive-definite matrices.

The spectral functions in this module use a divided-difference backward rather
than differentiating eigenvectors directly.  This is important for covariance
matrices: repeated (or nearly repeated) eigenvalues are common, and the usual
``eigh`` eigenvector backward is undefined at an eigenvalue collision.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn


def symmetrize(matrix: Tensor) -> Tensor:
    """Return the symmetric part of a batch of square matrices."""

    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"expected (..., d, d) square matrices, got {tuple(matrix.shape)}")
    return 0.5 * (matrix + matrix.transpose(-1, -2))


class _SymmetricSpectral(torch.autograd.Function):
    """Stable first-order autograd for a scalar symmetric spectral function."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object,
        matrix: Tensor,
        operation: str,
        lower: float | None,
        upper: float | None,
    ) -> Tensor:
        matrix = symmetrize(matrix)
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

        if operation == "clip":
            values = eigenvalues.clamp(min=lower, max=upper)
            derivatives = torch.ones_like(eigenvalues)
            if lower is not None:
                derivatives = derivatives * (eigenvalues > lower)
            if upper is not None:
                derivatives = derivatives * (eigenvalues < upper)
        elif operation == "log":
            if lower is None:
                raise RuntimeError("matrix logarithm requires a positive lower bound")
            safe = eigenvalues.clamp_min(lower)
            values = safe.log()
            derivatives = torch.where(eigenvalues > lower, safe.reciprocal(), 0.0)
        elif operation == "exp":
            safe = eigenvalues.clamp(min=lower, max=upper)
            values = safe.exp()
            derivatives = values
            if lower is not None:
                derivatives = derivatives * (eigenvalues > lower)
            if upper is not None:
                derivatives = derivatives * (eigenvalues < upper)
        elif operation == "sqrt":
            if lower is None:
                raise RuntimeError("matrix square root requires a positive lower bound")
            safe = eigenvalues.clamp_min(lower)
            values = safe.sqrt()
            derivatives = torch.where(eigenvalues > lower, 0.5 / values, 0.0)
        elif operation == "invsqrt":
            if lower is None:
                raise RuntimeError("inverse square root requires a positive lower bound")
            safe = eigenvalues.clamp_min(lower)
            values = safe.rsqrt()
            derivatives = torch.where(eigenvalues > lower, -0.5 * values / safe, 0.0)
        else:  # pragma: no cover - guarded by public wrappers
            raise ValueError(f"unknown spectral operation: {operation}")

        # Reconstruct by scaling columns.  Symmetrization removes round-off
        # asymmetry without changing the represented spectral function.
        result = symmetrize((eigenvectors * values.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2))
        ctx.save_for_backward(  # type: ignore[attr-defined]
            eigenvalues, eigenvectors, values, derivatives
        )
        return result

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: object, grad_output: Tensor
    ) -> tuple[Tensor, None, None, None]:
        eigenvalues, eigenvectors, values, derivatives = (  # type: ignore[attr-defined]
            ctx.saved_tensors
        )
        grad_output = symmetrize(grad_output)
        grad_eigenbasis = eigenvectors.transpose(-1, -2) @ grad_output @ eigenvectors

        li = eigenvalues.unsqueeze(-1)
        lj = eigenvalues.unsqueeze(-2)
        fi = values.unsqueeze(-1)
        fj = values.unsqueeze(-2)
        denominator = li - lj

        # The continuous limit of (f(li)-f(lj))/(li-lj) is f'(l).  Averaging
        # the endpoint derivatives is accurate and bounded for close pairs,
        # including exact multiplicities.
        eps = torch.finfo(eigenvalues.dtype).eps
        scale = torch.maximum(torch.maximum(li.abs(), lj.abs()), torch.ones_like(denominator))
        close = denominator.abs() <= (32.0 * eps * scale)
        divided = (fi - fj) / torch.where(close, torch.ones_like(denominator), denominator)
        limit = 0.5 * (derivatives.unsqueeze(-1) + derivatives.unsqueeze(-2))
        loewner = torch.where(close, limit, divided)
        loewner.diagonal(dim1=-2, dim2=-1).copy_(derivatives)

        grad_matrix = eigenvectors @ (loewner * grad_eigenbasis) @ eigenvectors.transpose(-1, -2)
        return symmetrize(grad_matrix), None, None, None


def _spectral(
    matrix: Tensor,
    operation: str,
    lower: float | None = None,
    upper: float | None = None,
) -> Tensor:
    # CPU eigh does not implement half/bfloat16.  Keeping spectral algebra in
    # fp32 under autocast also materially improves covariance stability.
    original_dtype = matrix.dtype
    work = matrix.float() if original_dtype in (torch.float16, torch.bfloat16) else matrix
    result = _SymmetricSpectral.apply(work, operation, lower, upper)
    return result.to(original_dtype) if result.dtype != original_dtype else result


def _matrix_scale(matrix: Tensor) -> Tensor:
    """Return a positive, homogeneous scale for each matrix in a batch.

    EEG covariances are represented in V^2 (typically around 1e-10), whereas
    aligned covariances are dimensionless and around one.  A relative spectral
    floor must work for both without requiring a data-unit convention.
    """

    matrix = symmetrize(matrix)
    scale = torch.diagonal(matrix, dim1=-2, dim2=-1).abs().mean(dim=-1)
    # The fallback only matters for a zero/trace-free non-SPD input.
    fallback = torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1)) / math.sqrt(
        matrix.shape[-1]
    )
    tiny = torch.finfo(matrix.dtype).tiny * 16.0
    return torch.where(scale > tiny, scale, fallback.clamp_min(tiny))


def eig_clip(matrix: Tensor, eps: float = 1e-5, max_eig: float | None = None) -> Tensor:
    """Rectify eigenvalues using bounds relative to each matrix's mean diagonal.

    ``eps`` is dimensionless: the physical floor is ``eps * trace(C)/d`` for
    an SPD matrix.  This prevents a conventional ``1e-5`` numerical epsilon
    from overwhelming EEG covariance eigenvalues measured in V^2.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    if max_eig is not None and max_eig <= eps:
        raise ValueError("max_eig must be greater than eps")
    scale = _matrix_scale(matrix)
    normalized = matrix / scale[..., None, None]
    clipped = _spectral(normalized, "clip", float(eps), max_eig)
    return clipped * scale[..., None, None]


def matrix_log(matrix: Tensor, eps: float = 1e-5) -> Tensor:
    """Principal SPD logarithm with a scale-relative eigenvalue floor."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    scale = _matrix_scale(matrix)
    normalized = matrix / scale[..., None, None]
    logged = _spectral(normalized, "log", float(eps))
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    return logged + scale.log()[..., None, None] * identity


def matrix_exp(matrix: Tensor, min_log: float = -60.0, max_log: float = 60.0) -> Tensor:
    """Matrix exponential of a symmetric matrix with overflow-safe bounds."""

    if min_log >= max_log:
        raise ValueError("min_log must be smaller than max_log")
    return _spectral(matrix, "exp", float(min_log), float(max_log))


def matrix_sqrt(matrix: Tensor, eps: float = 1e-5) -> Tensor:
    """Positive-definite square root."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    scale = _matrix_scale(matrix)
    normalized = matrix / scale[..., None, None]
    return _spectral(normalized, "sqrt", float(eps)) * scale.sqrt()[..., None, None]


def matrix_invsqrt(matrix: Tensor, eps: float = 1e-5) -> Tensor:
    """Positive-definite inverse square root."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    scale = _matrix_scale(matrix)
    normalized = matrix / scale[..., None, None]
    return _spectral(normalized, "invsqrt", float(eps)) / scale.sqrt()[..., None, None]


# Short aliases are convenient in equations and preserve compatibility with
# several SPD-network implementations.
logeig = matrix_log
expeig = matrix_exp
invsqrteig = matrix_invsqrt


def upper_vectorize(matrix: Tensor) -> Tensor:
    """Vectorize the upper triangle, scaling off-diagonals to preserve Frobenius norm."""

    matrix = symmetrize(matrix)
    d = matrix.shape[-1]
    row, col = torch.triu_indices(d, d, device=matrix.device)
    vector = matrix[..., row, col]
    scale = torch.where(row == col, 1.0, math.sqrt(2.0)).to(dtype=matrix.dtype)
    return vector * scale


def upper_unvectorize(vector: Tensor, matrix_dim: int | None = None) -> Tensor:
    """Inverse of :func:`upper_vectorize`."""

    size = vector.shape[-1]
    if matrix_dim is None:
        matrix_dim = int((math.sqrt(8 * size + 1) - 1) / 2)
    if matrix_dim * (matrix_dim + 1) // 2 != size:
        raise ValueError(f"length {size} is not triangular for matrix_dim={matrix_dim}")
    row, col = torch.triu_indices(matrix_dim, matrix_dim, device=vector.device)
    scale = torch.where(row == col, 1.0, math.sqrt(2.0)).to(dtype=vector.dtype)
    values = vector / scale
    output = vector.new_zeros(*vector.shape[:-1], matrix_dim, matrix_dim)
    output[..., row, col] = values
    output[..., col, row] = values
    return output


# Common name used by tangent-space libraries.
vech = upper_vectorize
unvech = upper_unvectorize


def log_euclidean_mean(
    matrices: Tensor,
    dim: int | Sequence[int] = 0,
    *,
    eps: float = 1e-5,
    return_log: bool = False,
) -> Tensor:
    """Log-Euclidean mean over one or more non-matrix dimensions."""

    log_mean = matrix_log(matrices, eps=eps).mean(dim=dim)
    return log_mean if return_log else matrix_exp(log_mean)


def _broadcast_log_reference(covariances: Tensor, log_reference: Tensor) -> Tensor:
    """Make shared, bandwise, or sample-wise references broadcastable."""

    d = covariances.shape[-1]
    if log_reference.shape[-2:] != (d, d):
        raise ValueError(
            f"reference matrices must end in {(d, d)}, got {tuple(log_reference.shape)}"
        )
    if log_reference.ndim == 2:
        return log_reference.reshape((1,) * (covariances.ndim - 2) + (d, d))
    if log_reference.ndim == covariances.ndim:
        return log_reference
    if covariances.ndim == 4 and log_reference.ndim == 3:
        n, bands = covariances.shape[:2]
        # A length-B reference is interpreted as bandwise.  For the ambiguous
        # N == B case, sample-wise callers can use shape (N, 1, d, d).
        if log_reference.shape[0] == bands:
            return log_reference.unsqueeze(0)
        if log_reference.shape[0] == n:
            return log_reference.unsqueeze(1)
    if covariances.ndim == 3 and log_reference.ndim == 3:
        return log_reference
    raise ValueError(
        "log_reference must be shared (d,d), bandwise (B,d,d), sample-wise "
        "(N,d,d)/(N,1,d,d), or fully specified (N,B,d,d)"
    )


def log_euclidean_recenter(
    covariances: Tensor,
    log_reference: Tensor,
    *,
    eps: float = 1e-5,
) -> Tensor:
    """Whiten covariances by a reference maintained in log coordinates.

    If ``L = log(R)``, the congruence ``exp(-L/2) C exp(-L/2)`` maps the
    reference ``R`` to identity while preserving positive definiteness.
    ``log_reference`` may be shared, bandwise, sample-wise, or fully batched.
    """

    covariances = eig_clip(covariances, eps=eps)
    reference = symmetrize(_broadcast_log_reference(covariances, log_reference))
    whitener = matrix_exp(-0.5 * reference)
    return eig_clip(whitener @ covariances @ whitener, eps=eps)


class BiMap(nn.Module):
    """Bilinear SPD map ``W C W.T`` with semi-orthogonal rows.

    An unconstrained tall parameter is mapped through reduced QR on every
    forward pass, so ``W W.T = I``.  :meth:`retract_` can periodically move the
    raw parameter back to the Stiefel manifold without changing the represented
    map, making the layer compatible with retraction-based optimization.
    """

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        if not 0 < out_features <= in_features:
            raise ValueError("BiMap requires 0 < out_features <= in_features")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.eps = float(eps)
        self.raw_weight = nn.Parameter(torch.empty(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.raw_weight)
            q, _ = torch.linalg.qr(self.raw_weight, mode="reduced")
            self.raw_weight.copy_(q)

    @property
    def weight(self) -> Tensor:
        q, r = torch.linalg.qr(self.raw_weight, mode="reduced")
        # Fix QR's arbitrary column signs.  At initialization diag(r) is +1;
        # zero is mapped to +1 to keep the projection well defined.
        diagonal = torch.diagonal(r)
        signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
        return (q * signs.unsqueeze(0)).transpose(-1, -2)

    @torch.no_grad()
    def retract_(self) -> "BiMap":
        self.raw_weight.copy_(self.weight.transpose(-1, -2))
        return self

    def forward(self, covariance: Tensor) -> Tensor:
        if covariance.shape[-2:] != (self.in_features, self.in_features):
            raise ValueError(
                f"expected (..., {self.in_features}, {self.in_features}), "
                f"got {tuple(covariance.shape)}"
            )
        weight = self.weight
        mapped = weight @ symmetrize(covariance) @ weight.transpose(-1, -2)
        return eig_clip(mapped, eps=self.eps)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, eps={self.eps}"


class ReEig(nn.Module):
    """Eigenvalue rectification layer."""

    def __init__(self, eps: float = 1e-5, max_eig: float | None = None) -> None:
        super().__init__()
        self.eps = float(eps)
        self.max_eig = max_eig

    def forward(self, matrix: Tensor) -> Tensor:
        return eig_clip(matrix, self.eps, self.max_eig)


class LogEig(nn.Module):
    """SPD matrix logarithm, optionally returned as a norm-preserving vector."""

    def __init__(self, eps: float = 1e-5, vectorize: bool = False) -> None:
        super().__init__()
        self.eps = float(eps)
        self.vectorize = bool(vectorize)

    def forward(self, matrix: Tensor) -> Tensor:
        logged = matrix_log(matrix, self.eps)
        return upper_vectorize(logged) if self.vectorize else logged


class SPDMomentumBatchNorm(nn.Module):
    """Log-Euclidean running-reference recentering for bandwise SPD inputs.

    Running statistics live in the flat vector space of symmetric matrix logs.
    Training uses the current batch reference when there is more than one
    sample; batch-size-one falls back to the running reference so information is
    not collapsed to identity.  At evaluation time stats are frozen unless
    ``update_stats=True`` is explicitly requested for label-free adaptation.
    """

    def __init__(
        self,
        num_bands: int,
        matrix_dim: int,
        momentum: float = 0.05,
        eps: float = 1e-5,
        *,
        use_batch_stats: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < momentum <= 1.0:
            raise ValueError("momentum must be in (0, 1]")
        self.num_bands = int(num_bands)
        self.matrix_dim = int(matrix_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.use_batch_stats = bool(use_batch_stats)
        self.register_buffer(
            "running_log_reference", torch.zeros(num_bands, matrix_dim, matrix_dim)
        )
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def reset_running_stats(self) -> None:
        self.running_log_reference.zero_()
        self.num_batches_tracked.zero_()

    @torch.no_grad()
    def set_reference(self, reference: Tensor, *, is_log: bool = False) -> None:
        value = symmetrize(reference if is_log else matrix_log(reference, self.eps))
        if value.shape == (self.matrix_dim, self.matrix_dim):
            value = value.expand(self.num_bands, -1, -1)
        if value.shape != self.running_log_reference.shape:
            raise ValueError(
                f"expected {(self.num_bands, self.matrix_dim, self.matrix_dim)}, "
                f"got {tuple(value.shape)}"
            )
        self.running_log_reference.copy_(value.to(self.running_log_reference))
        self.num_batches_tracked.fill_(1)

    @torch.no_grad()
    def _update(self, batch_log_reference: Tensor) -> None:
        batch_log_reference = symmetrize(batch_log_reference.detach()).to(
            self.running_log_reference
        )
        if self.num_batches_tracked.item() == 0:
            self.running_log_reference.copy_(batch_log_reference)
        else:
            self.running_log_reference.lerp_(batch_log_reference, self.momentum)
        self.num_batches_tracked.add_(1)

    def forward(
        self,
        covariances: Tensor,
        log_reference: Tensor | None = None,
        *,
        update_stats: bool | None = None,
        return_reference: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if covariances.ndim != 4 or covariances.shape[1:] != (
            self.num_bands,
            self.matrix_dim,
            self.matrix_dim,
        ):
            raise ValueError(
                f"expected (N, {self.num_bands}, {self.matrix_dim}, {self.matrix_dim}), "
                f"got {tuple(covariances.shape)}"
            )

        should_update = self.training if update_stats is None else bool(update_stats)
        batch_reference: Tensor | None = None
        if log_reference is None and (should_update or (self.training and self.use_batch_stats)):
            batch_reference = matrix_log(covariances, self.eps).mean(dim=0)
        if should_update and batch_reference is not None:
            self._update(batch_reference)

        if log_reference is not None:
            used_reference = log_reference
        elif self.training and self.use_batch_stats and covariances.shape[0] > 1:
            assert batch_reference is not None
            used_reference = batch_reference
        else:
            used_reference = self.running_log_reference.to(covariances)

        recentered = log_euclidean_recenter(covariances, used_reference, eps=self.eps)
        return (recentered, used_reference) if return_reference else recentered


# Explicit and legacy-friendly names for the same layer.
LogEuclideanRecenter = SPDMomentumBatchNorm
SPDLogEuclideanBatchNorm = SPDMomentumBatchNorm
SPDBatchNorm = SPDMomentumBatchNorm


__all__ = [
    "BiMap",
    "LogEig",
    "LogEuclideanRecenter",
    "ReEig",
    "SPDBatchNorm",
    "SPDLogEuclideanBatchNorm",
    "SPDMomentumBatchNorm",
    "eig_clip",
    "expeig",
    "invsqrteig",
    "log_euclidean_mean",
    "log_euclidean_recenter",
    "logeig",
    "matrix_exp",
    "matrix_invsqrt",
    "matrix_log",
    "matrix_sqrt",
    "symmetrize",
    "upper_unvectorize",
    "upper_vectorize",
    "unvech",
    "vech",
]
