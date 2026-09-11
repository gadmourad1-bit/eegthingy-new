import os
import sys

import numpy as np
import torch

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "classifier"),
                os.path.join(os.path.dirname(__file__), "..")]

import mirepnet


def test_interpolation_preserves_observed_channels_and_is_convex():
    matrix = mirepnet.interpolation_matrix()
    assert matrix.shape == (45, 15)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    assert np.all(matrix >= 0.0)
    for source_index, channel in enumerate(mirepnet.SOURCE_CHANNELS):
        target_index = mirepnet.CHANNELS.index(channel)
        expected = np.zeros(15)
        expected[source_index] = 1.0
        np.testing.assert_array_equal(matrix[target_index], expected)


def test_ea_maps_mean_covariance_to_identity():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 15, 1000))
    x[:, 1] += 0.7 * x[:, 0]
    aligned = mirepnet._align(x, mirepnet._whitener(x))
    aligned -= aligned.mean(axis=2, keepdims=True)
    cov = np.einsum("nct,ndt->ncd", aligned, aligned) / (aligned.shape[2] - 1)
    np.testing.assert_allclose(cov.mean(axis=0), np.eye(15), atol=1e-5)


def test_architecture_matches_official_converted_state():
    model = mirepnet.MIRepNetModel(n_outputs=2)
    state = mirepnet.pretrained_state()
    state = {key: value for key, value in state.items() if not key.startswith("final_layer.")}
    incompatible = model.load_state_dict(state, strict=False)
    assert set(incompatible.missing_keys) == {"final_layer.weight", "final_layer.bias"}
    assert incompatible.unexpected_keys == []
    assert model(torch.zeros(2, 45, 1000)).shape == (2, 2)


def test_task_repeat_live_window_matches_input_contract():
    decoder = mirepnet.MIRepNetDecoder(window_mode="task-repeat")
    rng = np.random.default_rng(12)
    prepared = decoder.prepare_window(rng.normal(size=(15, 750)), sfreq=250.0)
    assert prepared.shape == (1, 15, 1000)
    np.testing.assert_allclose(prepared[0, :, :500], prepared[0, :, 500:])


def test_fast_feature_adapter_balances_completed_protocol_block():
    decoder = mirepnet.MIRepNetDecoder()
    decoder.set_reference = lambda X: decoder
    decoder.extract_features = lambda X: np.asarray(X)[:, 0, :]
    X_cal = np.array([[[-3.0, -2.0]], [[-2.0, -3.0]],
                      [[2.0, 3.0]], [[3.0, 2.0]]])
    y_cal = np.array([1, 1, 2, 2])
    X_test = np.array([[[-4.0, -1.0]], [[-1.0, -4.0]],
                       [[1.0, 4.0]], [[4.0, 1.0]]])
    decoder.fit_feature_adapter(X_cal, y_cal)
    prediction = decoder.predict_feature_adapter(X_test, balanced_protocol=True)
    assert prediction.shape == (4,)
    assert np.count_nonzero(prediction == 1) == 2
    assert np.count_nonzero(prediction == 2) == 2
