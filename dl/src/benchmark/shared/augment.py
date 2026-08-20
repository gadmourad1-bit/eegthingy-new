"""Fit-side, SPD-safe covariance augmentation for GeoAdaptNet training.

Only the training (inner rec1-2) loader is ever augmented; the selection and
outer-test loaders stay raw.  Two class-consistent, manifold-correct operations:

* **Left/right electrode swap + label flip** -- motor imagery is laterally
  organised (C3 vs C4 etc.), so mirroring the sensor montage turns a left-hand
  trial into a valid right-hand trial.  On a covariance this is the symmetric
  permutation ``P C P^T``; the log-Euclidean reference (stored in log space)
  permutes the same way because ``log(P C P^T) = P log(C) P^T``.
* **Log-Euclidean same-class mixup** -- interpolate two same-label covariances on
  the manifold (mean of matrix logs), a Manifold-Mixup variant that keeps the
  label hard and the result SPD.

Both are standard, cited regularizers (mixup: Zhang 2018; manifold mixup: Verma
2019; MI mirror augmentation is common); they exist here to curb the 480-d tangent
anchor's overfitting on ~950 training epochs, not as a novelty claim.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .spd import matrix_exp, matrix_log


def left_right_swap_index(channels: Sequence[str]) -> list[int]:
    """Permutation that swaps 10-20 left/right sensor pairs (odd<->even number)."""

    name_to_index = {name: index for index, name in enumerate(channels)}
    permutation = list(range(len(channels)))
    for index, name in enumerate(channels):
        if not name or not name[-1].isdigit():
            continue  # midline (Cz, Pz, Fz) stays put
        digit = int(name[-1])
        partner_digit = digit + 1 if digit % 2 == 1 else digit - 1
        partner = f"{name[:-1]}{partner_digit}"
        if partner in name_to_index:
            permutation[index] = name_to_index[partner]
    return permutation


def _swap_channels(matrices: Tensor, swap_index: Tensor) -> Tensor:
    """Apply ``P M P^T`` for a channel permutation to a (..., C, C) batch."""

    swapped = matrices.index_select(-2, swap_index)
    return swapped.index_select(-1, swap_index)


def augment_batch(
    covariances: Tensor,
    labels: Tensor,
    references: Tensor | None,
    *,
    swap_index: Tensor,
    lr_swap_prob: float,
    mixup_alpha: float,
    eps: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Return an augmented copy of one training batch (covariances/labels/refs)."""

    device = covariances.device
    cov = covariances
    lab = labels
    ref = references if (references is not None and references.numel() > 0) else None

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, device=device, generator=generator)

    # --- left/right electrode swap + label flip (task rows only) ---
    if lr_swap_prob > 0.0:
        swap_index = swap_index.to(device)
        task = lab >= 0
        do_swap = (rand(len(cov)) < lr_swap_prob) & task
        if torch.any(do_swap):
            rows = torch.nonzero(do_swap, as_tuple=False).squeeze(-1)
            cov = cov.clone()
            cov[rows] = _swap_channels(cov[rows], swap_index)
            lab = lab.clone()
            lab[rows] = 1 - lab[rows]
            if ref is not None:
                ref = ref.clone()
                ref[rows] = _swap_channels(ref[rows], swap_index)

    # --- log-Euclidean same-class mixup (partners share the label) ---
    if mixup_alpha > 0.0:
        order = torch.argsort(lab + rand(len(lab)))  # groups equal labels together
        partner = torch.empty_like(order)
        partner[order] = order.roll(1)  # each row pairs with a same-label neighbour
        same = lab[partner] == lab
        beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
        lam = beta.sample((len(cov),)).to(device)
        lam = torch.where(same, lam, torch.ones_like(lam)).view(-1, *([1] * (cov.ndim - 1)))
        cov_log = matrix_log(cov, eps)
        cov = matrix_exp(lam * cov_log + (1.0 - lam) * cov_log[partner])
        if ref is not None:
            ref = lam * ref + (1.0 - lam) * ref[partner]

    return cov, lab, ref


__all__ = ["augment_batch", "left_right_swap_index"]
