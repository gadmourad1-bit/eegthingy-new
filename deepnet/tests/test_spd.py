from __future__ import annotations

import torch

from deepnet.spd import (
    BiMap,
    SPDMomentumBatchNorm,
    eig_clip,
    log_euclidean_mean,
    log_euclidean_recenter,
    matrix_exp,
    matrix_invsqrt,
    matrix_log,
    symmetrize,
    upper_unvectorize,
    upper_vectorize,
)


def _spd(*shape: int, dim: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    factor = torch.randn(*shape, dim, dim, generator=generator, dtype=dtype)
    eye = torch.eye(dim, dtype=dtype)
    return factor @ factor.transpose(-1, -2) + 0.4 * eye


def test_spectral_roundtrip_and_invsqrt_invariants() -> None:
    matrix = _spd(5, dim=6)
    reconstructed = matrix_exp(matrix_log(matrix, eps=1e-10))
    torch.testing.assert_close(reconstructed, matrix, rtol=2e-8, atol=2e-8)

    inverse_root = matrix_invsqrt(matrix, eps=1e-10)
    identity = inverse_root @ matrix @ inverse_root
    torch.testing.assert_close(
        identity, torch.eye(6, dtype=identity.dtype).expand_as(identity), rtol=2e-8, atol=2e-8
    )
    assert torch.all(torch.linalg.eigvalsh(reconstructed) > 0)


def test_spectral_ops_preserve_real_eeg_voltage_scale() -> None:
    # MNE stores EEG in volts, so real covariance entries are around 1e-10 V^2.
    # A fixed absolute 1e-5 floor would collapse all of these matrices.
    matrix = _spd(4, dim=6) * 1e-10
    logged = matrix_log(matrix)
    reconstructed = matrix_exp(logged)
    torch.testing.assert_close(reconstructed, matrix, rtol=2e-7, atol=1e-16)
    assert torch.linalg.vector_norm(logged[0] - logged[1]) > 0.1
    clipped = eig_clip(matrix)
    assert float(clipped.abs().max()) < 1e-7
    assert torch.all(torch.linalg.eigvalsh(clipped) > 0)


def test_spectral_backward_is_finite_at_repeated_eigenvalues() -> None:
    # Repeated eigenvalues make the raw eigh eigenvector backward singular.
    matrix = torch.eye(5, dtype=torch.float64, requires_grad=True)
    transformed = matrix_log(matrix) + matrix_invsqrt(matrix) + eig_clip(matrix)
    transformed.square().sum().backward()
    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()
    assert torch.count_nonzero(matrix.grad) > 0


def test_upper_vectorization_is_invertible_and_norm_preserving() -> None:
    matrix = symmetrize(torch.randn(4, 7, 7, dtype=torch.float64))
    vector = upper_vectorize(matrix)
    restored = upper_unvectorize(vector)
    torch.testing.assert_close(restored, matrix)
    torch.testing.assert_close(
        torch.linalg.vector_norm(vector, dim=-1),
        torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1)),
    )


def test_log_euclidean_mean_and_recenter_map_reference_to_identity() -> None:
    samples = _spd(8, 3, dim=5)
    log_reference = log_euclidean_mean(samples, dim=0, return_log=True, eps=1e-10)
    reference = matrix_exp(log_reference)
    centered = log_euclidean_recenter(reference.unsqueeze(0), log_reference, eps=1e-10)
    torch.testing.assert_close(
        centered,
        torch.eye(5, dtype=centered.dtype).expand_as(centered),
        rtol=2e-8,
        atol=2e-8,
    )

    # Fully per-sample references use the same congruence and remain SPD.
    centered_samples = log_euclidean_recenter(samples, matrix_log(samples, 1e-10), eps=1e-10)
    torch.testing.assert_close(
        centered_samples,
        torch.eye(5, dtype=samples.dtype).expand_as(samples),
        rtol=3e-8,
        atol=3e-8,
    )


def test_bimap_is_semiorthogonal_spd_and_differentiable() -> None:
    layer = BiMap(7, 4, eps=1e-8).double()
    covariance = _spd(3, dim=7).requires_grad_()
    output = layer(covariance)
    identity = torch.eye(4, dtype=layer.weight.dtype)
    torch.testing.assert_close(layer.weight @ layer.weight.T, identity, rtol=1e-10, atol=1e-10)
    assert torch.all(torch.linalg.eigvalsh(output) > 0)
    output.logdet().sum().backward()
    assert covariance.grad is not None and torch.isfinite(covariance.grad).all()
    assert layer.raw_weight.grad is not None and torch.isfinite(layer.raw_weight.grad).all()

    before = output.detach()
    layer.retract_()
    torch.testing.assert_close(layer(covariance.detach()), before, rtol=1e-8, atol=1e-8)


def test_momentum_reference_updates_and_batch_one_does_not_collapse() -> None:
    layer = SPDMomentumBatchNorm(3, 5, momentum=0.2, eps=1e-8).double()
    covariance = _spd(6, 3, dim=5)
    centered = layer(covariance)
    assert centered.shape == covariance.shape
    assert layer.num_batches_tracked.item() == 1
    assert torch.linalg.vector_norm(layer.running_log_reference) > 0

    layer.train()
    singleton = layer(covariance[:1])
    # Batch-one must use the running reference, not its own covariance (which
    # would map every possible input to identity and erase the label signal).
    assert not torch.allclose(
        singleton, torch.eye(5, dtype=singleton.dtype).expand_as(singleton)
    )

    layer.eval()
    count = layer.num_batches_tracked.item()
    first = layer(covariance[:2])
    second = layer(covariance[:2])
    torch.testing.assert_close(first, second)
    assert layer.num_batches_tracked.item() == count
