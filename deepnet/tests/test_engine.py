import numpy as np
import pytest
import torch

import deepnet.engine as engine
from deepnet.engine import (
    CovarianceDataset,
    TrainConfig,
    load_model_checkpoint,
    resolve_device,
    train_fixed_epochs,
)
from deepnet.model import GeoAdaptNet


def _identity_covariances(n: int = 3) -> np.ndarray:
    return np.broadcast_to(np.eye(15, dtype=np.float32), (n, 4, 15, 15)).copy()


def test_dataset_marks_rest_examples_as_no_intent() -> None:
    dataset = CovarianceDataset(_identity_covariances(), np.array([0, 1, -1]))
    assert dataset.intent_targets.tolist() == [1.0, 1.0, 0.0]


def test_dataset_expands_shared_reference() -> None:
    reference = np.zeros((4, 15, 15), dtype=np.float32)
    dataset = CovarianceDataset(
        _identity_covariances(), np.array([0, 1, 0]), log_references=reference
    )
    assert dataset.references.shape == (3, 4, 15, 15)


def test_auto_device_avoids_mps_without_required_spectral_kernel(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(engine, "_mps_supports_spectral_ops", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_auto_device_can_use_mps_when_required_spectral_kernel_exists(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(engine, "_mps_supports_spectral_ops", lambda: True)
    assert resolve_device("auto").type == "mps"


def test_explicit_mps_has_clear_error_without_required_spectral_kernel(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(engine, "_mps_supports_spectral_ops", lambda: False)
    with pytest.raises(RuntimeError, match="PYTORCH_ENABLE_MPS_FALLBACK"):
        resolve_device("mps")


def test_explicit_unavailable_cuda_has_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device("cuda")


def test_safe_checkpoint_roundtrip(tmp_path) -> None:
    model = GeoAdaptNet(auxiliary_intent=True)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model": "GeoAdaptNet",
            "model_config": {"auxiliary_intent": True},
            "state_dict": model.state_dict(),
            "subject": 1,
        },
        path,
    )
    loaded, metadata = load_model_checkpoint(path)
    assert loaded.auxiliary_intent
    assert metadata["subject"] == 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])


def test_fixed_refit_keeps_selection_scheduler_horizon() -> None:
    covariance = np.broadcast_to(
        np.eye(2, dtype=np.float32), (4, 1, 2, 2)
    ).copy()
    data = CovarianceDataset(covariance, np.asarray([0, 1, 0, 1]))
    learning_rate = 3e-4
    result = train_fixed_epochs(
        GeoAdaptNet(
            bands=1,
            channels=2,
            reduced_dim=1,
            band_width=4,
            fusion_width=4,
            dropout=0.0,
        ),
        data,
        epochs=2,
        config=TrainConfig(
            epochs=10,
            batch_size=4,
            learning_rate=learning_rate,
            device="cpu",
        ),
    )
    # A wrongly compressed two-epoch cosine would already halve the LR here.
    assert result.history[0]["learning_rate"] > 0.9 * learning_rate
