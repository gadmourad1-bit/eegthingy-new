from __future__ import annotations

import mne
import numpy as np
import torch

from benchmark.baselines import make_model
from benchmark.config import (
    BNCI004_3_CHANNELS,
    CANONICAL_21_CHANNELS,
    LOCAL_EXP4_15_CHANNELS,
)
from benchmark.models import (
    CANONICAL_21_POSITIONS,
    EXTENDED_31_POSITIONS,
    CardinalFBMSNet,
)


ADDED_10_CHANNELS = (
    "T5",
    "T6",
    "F7",
    "F8",
    "F3",
    "F4",
    "T3",
    "T4",
    "P3",
    "P4",
)


def _unit_mne_head_positions(channel_names: tuple[str, ...]) -> np.ndarray:
    info = mne.create_info(channel_names, sfreq=128.0, ch_types="eeg")
    info.set_montage("standard_1005")
    coordinates = np.stack([channel["loc"][:3] for channel in info["chs"]])
    return coordinates / np.linalg.norm(coordinates, axis=1, keepdims=True)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_extended_atlas_is_normalized_mne_head_geometry_in_declared_order() -> None:
    channel_names = CANONICAL_21_CHANNELS + ADDED_10_CHANNELS
    expected = _unit_mne_head_positions(channel_names)
    declared = np.asarray(EXTENDED_31_POSITIONS, dtype=np.float64)

    assert channel_names[:21] == CANONICAL_21_CHANNELS
    np.testing.assert_allclose(
        declared[:21],
        np.asarray(CANONICAL_21_POSITIONS, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(declared, expected, rtol=0.0, atol=7e-8)


def test_paired_cardinal_fbc_matches_indexed_fbc_across_montages_and_lengths() -> None:
    canonical_positions = torch.tensor(CANONICAL_21_POSITIONS)
    three_channel_names = ("C3", "Cz", "C4")
    three_channel_indices = tuple(
        CANONICAL_21_CHANNELS.index(name) for name in three_channel_names
    )
    three_channel_positions = canonical_positions[list(three_channel_indices)]
    montages = (
        (CANONICAL_21_CHANNELS, canonical_positions, 4),
        (three_channel_names, three_channel_positions, 2),
    )

    for n_times in (256, 320):
        for channel_names, positions, n_outputs in montages:
            for training in (True, False):
                torch.manual_seed(20260719)
                indexed = make_model(
                    "fbcnet",
                    n_channels=len(channel_names),
                    n_outputs=n_outputs,
                    n_times=n_times,
                    sfreq=128.0,
                    channel_names=channel_names,
                    channel_positions=positions,
                ).train(training)
                torch.manual_seed(20260719)
                continuous = make_model(
                    "cardinal_fbc",
                    n_channels=len(channel_names),
                    n_outputs=n_outputs,
                    n_times=n_times,
                    sfreq=128.0,
                    channel_names=channel_names,
                    channel_positions=positions,
                ).train(training)
                values = torch.randn(4, len(channel_names), n_times)

                with torch.inference_mode():
                    indexed_output = indexed(values)
                    continuous_output = continuous(values, positions)

                torch.testing.assert_close(
                    continuous_output,
                    indexed_output,
                    rtol=5e-6,
                    atol=5e-6,
                )


def test_cardinal_fbc_has_exact_fbc_parameter_count_on_canonical_montage() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    for n_times in (256, 320):
        torch.manual_seed(17)
        indexed = make_model(
            "fbcnet",
            n_channels=21,
            n_outputs=4,
            n_times=n_times,
            sfreq=128.0,
            channel_names=CANONICAL_21_CHANNELS,
            channel_positions=positions,
        )
        torch.manual_seed(17)
        continuous = make_model(
            "cardinal_fbc",
            n_channels=21,
            n_outputs=4,
            n_times=n_times,
            sfreq=128.0,
            channel_names=CANONICAL_21_CHANNELS,
            channel_positions=positions,
        )

        assert _parameter_count(continuous) == _parameter_count(indexed)


def test_paired_cardinal_fbms_matches_indexed_across_montages_and_lengths() -> None:
    canonical_positions = torch.tensor(CANONICAL_21_POSITIONS)
    three_channel_indices = tuple(
        CANONICAL_21_CHANNELS.index(name) for name in BNCI004_3_CHANNELS
    )
    montages = (
        (CANONICAL_21_CHANNELS, canonical_positions, 4),
        (
            LOCAL_EXP4_15_CHANNELS,
            torch.tensor(
                _unit_mne_head_positions(LOCAL_EXP4_15_CHANNELS),
                dtype=torch.float32,
            ),
            2,
        ),
        (
            BNCI004_3_CHANNELS,
            canonical_positions[list(three_channel_indices)],
            2,
        ),
    )
    for n_times in (256, 320):
        for channel_names, positions, n_outputs in montages:
            torch.manual_seed(20260720)
            indexed = make_model(
                "fbmsnet",
                n_channels=len(channel_names),
                n_outputs=n_outputs,
                n_times=n_times,
                sfreq=128.0,
                channel_names=channel_names,
                channel_positions=positions,
            ).train()
            torch.manual_seed(20260720)
            continuous = make_model(
                "cardinal_fbms",
                n_channels=len(channel_names),
                n_outputs=n_outputs,
                n_times=n_times,
                sfreq=128.0,
                channel_names=channel_names,
                channel_positions=positions,
            ).train()
            assert isinstance(continuous, CardinalFBMSNet)
            values = torch.randn(3, len(channel_names), n_times)
            with torch.inference_mode():
                indexed_output = indexed(values)
                continuous_output = continuous(values, positions)
            torch.testing.assert_close(
                continuous_output,
                indexed_output,
                rtol=1e-5,
                atol=1e-5,
            )


def test_cardinal_fbms_has_parameter_parity_on_canonical_montage() -> None:
    positions = torch.tensor(CANONICAL_21_POSITIONS)
    torch.manual_seed(49)
    indexed = make_model(
        "fbmsnet",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    )
    torch.manual_seed(49)
    continuous = make_model(
        "cardinal_fbms",
        n_channels=21,
        n_outputs=4,
        n_times=320,
        sfreq=128.0,
        channel_names=CANONICAL_21_CHANNELS,
        channel_positions=positions,
    )
    assert _parameter_count(continuous) == _parameter_count(indexed)
