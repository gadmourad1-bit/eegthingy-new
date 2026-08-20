#!/usr/bin/env python3
"""Deterministic CUDA smoke test for the HemiQ harmonized-v2 model adapter.

This script trains no formal benchmark record and opens no dataset label.  It
uses a synthetic balanced signal solely to verify one exact full-architecture
forward/backward/optimizer step, repeatability, reset identity, and reflection
anti-equivariance on the selected CUDA device.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def _synthetic_trials() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260729)
    labels = np.arange(8, dtype=np.int64) % 2
    time = np.arange(320, dtype=np.float32) / 128.0
    carrier = np.sin(2.0 * np.pi * 14.0 * time)
    raw = rng.normal(scale=0.03, size=(8, 3, 320)).astype(np.float32)
    for row, label in enumerate(labels):
        raw[row, 1] += carrier
        raw[row, 0 if label else 2] += carrier
    return raw, labels


def main() -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," in visible:
        raise RuntimeError("expose exactly one physical GPU UUID")

    import torch

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from benchmark.hemiq_v2_model import (
        HemiQHarmonizedV2Classifier,
        derive_hemiq_views,
        load_frozen_config,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the smoke test requires exactly one CUDA-visible device")
    target = torch.device("cuda:0")
    torch.cuda.set_device(target)
    torch.empty(0, device=target)
    config, config_identity = load_frozen_config(
        project_root,
        seed=47,
        device="cuda:0",
    )
    raw, labels = _synthetic_trials()
    views = derive_hemiq_views(raw, channel_names=("C3", "Cz", "C4"))

    torch.cuda.reset_peak_memory_stats(target)
    first = HemiQHarmonizedV2Classifier(config).fit_fixed_epochs(
        views["raw"],
        views["covariance"],
        labels,
        epochs=1,
    )
    first_probability = first.predict_proba(views["raw"])
    prepared = torch.from_numpy(first._prepare_raw(views["raw"])).to(target)
    first.model_.eval()
    with torch.no_grad():
        direct = first.model_(prepared)
        reflected = first.model_(prepared[:, (2, 1, 0), :])
    if not torch.equal(direct, -reflected):
        raise RuntimeError("HemiQ reflection action is not bit-exact on CUDA")

    second = HemiQHarmonizedV2Classifier(config).fit_fixed_epochs(
        views["raw"],
        views["covariance"],
        labels,
        epochs=1,
    )
    second_probability = second.predict_proba(views["raw"])
    if (
        first.initial_model_state_sha256_
        != second.initial_model_state_sha256_
        or first.model_state_sha256_ != second.model_state_sha256_
        or first.history_ != second.history_
        or not np.array_equal(first_probability, second_probability)
    ):
        raise RuntimeError("HemiQ CUDA optimizer replay is not bitwise repeatable")
    properties = torch.cuda.get_device_properties(target)
    result = {
        "schema": "eeg-mi-hemiq-v2-cuda-smoke-v1",
        "formal_result": False,
        "passed": True,
        "cuda_visible_devices": visible,
        "gpu_name": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "")),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "config_file_sha256": config_identity["config_file_sha256"],
        "initial_model_state_sha256": first.initial_model_state_sha256_,
        "one_step_model_state_sha256": first.model_state_sha256_,
        "trainable_parameter_count": first.trainable_parameter_count_,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(target)),
        "max_reflection_error": float(torch.max(torch.abs(direct + reflected)).cpu()),
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
