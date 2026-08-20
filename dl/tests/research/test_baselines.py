import numpy as np

from benchmark.research.baselines import EAFilterBankCSP, RiemannianTangentLogistic


def _synthetic_epochs(seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], 12)
    groups = np.tile(np.repeat([0, 1], 6), 2)
    epochs = rng.normal(size=(24, 2, 4, 40))
    epochs[labels == 1, :, 0] *= 2.0
    return epochs, labels, groups


def test_ea_fbcsp_baseline_fits_and_calibrates() -> None:
    epochs, labels, groups = _synthetic_epochs()
    baseline = EAFilterBankCSP(n_components=2).fit(epochs, labels, groups)
    baseline.calibrate(epochs[:4])
    probabilities = baseline.predict_proba(epochs[:3])
    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_riemannian_baseline_fits_covariances() -> None:
    epochs, labels, groups = _synthetic_epochs()
    covariances = np.einsum("nbct,nbdt->nbcd", epochs, epochs) / epochs.shape[-1]
    covariances += np.eye(4)[None, None] * 1e-3
    baseline = RiemannianTangentLogistic().fit(covariances, labels, groups)
    baseline.calibrate(covariances[:4])
    probabilities = baseline.predict_proba(covariances[:3])
    assert probabilities.shape == (3, 2)
