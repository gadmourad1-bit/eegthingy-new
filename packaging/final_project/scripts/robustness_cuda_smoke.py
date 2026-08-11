"""Read-only CUDA determinism smoke for the robustness-screen model roster.

This utility does not train, score, or write artifacts.  It constructs every
fixed robustness model twice with the same seed, runs one synthetic
forward/backward pass, and requires bitwise-identical logits and gradients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "4"

import numpy as np
import torch
from torch.nn import functional as F

from ieee_mi.config import dataset_spec, preprocessing_for_dataset
from ieee_mi.development_screen import ScreenJob
from ieee_mi.robustness_screen import MODELS, _make_model
from ieee_mi.training import _forward, configure_determinism


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _one_pass(
    *,
    model_name: str,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    positions: np.ndarray,
    channel_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    configure_determinism(7)
    job = ScreenJob(
        dataset="local_exp4",
        model=model_name,
        subject=3,
        fold=0,
        seed=7,
    )
    preprocessing = preprocessing_for_dataset(job.dataset)
    model = _make_model(
        job,
        n_channels=x_cpu.shape[1],
        n_outputs=dataset_spec(job.dataset).n_classes,
        n_times=x_cpu.shape[2],
        sfreq=float(preprocessing["sfreq_hz"]),
        channel_names=channel_names,
        positions=positions,
    ).to("cuda:0")
    model.train()
    x = x_cpu.to("cuda:0")
    y = y_cpu.to("cuda:0")
    position_tensor = torch.as_tensor(
        positions,
        dtype=torch.float32,
        device="cuda:0",
    )
    logits = _forward(model, x, position_tensor)
    if logits.shape != (len(x_cpu), dataset_spec(job.dataset).n_classes):
        raise RuntimeError(
            f"{model_name} returned unexpected logits {tuple(logits.shape)}"
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"{model_name} returned non-finite logits")
    F.cross_entropy(logits, y).backward()
    gradients = [
        parameter.grad.detach().flatten()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError(f"{model_name} produced no gradients")
    gradient = torch.cat(gradients)
    if not torch.isfinite(gradient).all():
        raise RuntimeError(f"{model_name} produced non-finite gradients")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logits_cpu = logits.detach().cpu()
    gradient_cpu = gradient.cpu()
    del logits, gradient, gradients, x, y, position_tensor, model
    torch.cuda.empty_cache()
    return logits_cpu, gradient_cpu, parameter_count


def run(cache_path: Path) -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the smoke requires exactly one visible CUDA GPU")
    with np.load(cache_path, allow_pickle=False) as archive:
        identity = json.loads(str(archive["identity"].item()))
        positions = np.asarray(archive["positions"], dtype=np.float32).copy()
        channel_names = tuple(str(value) for value in archive["channel_names"].tolist())
    shape = tuple(int(value) for value in identity["shape"])
    if len(shape) != 3 or positions.shape != (shape[1], 3):
        raise RuntimeError("cache metadata shape is inconsistent")

    configure_determinism(7)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260729)
    x_cpu = torch.randn(
        (4, shape[1], shape[2]),
        generator=generator,
        dtype=torch.float32,
    )
    y_cpu = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    results: dict[str, Any] = {}
    for model_name in MODELS:
        first_logits, first_gradient, parameter_count = _one_pass(
            model_name=model_name,
            x_cpu=x_cpu,
            y_cpu=y_cpu,
            positions=positions,
            channel_names=channel_names,
        )
        second_logits, second_gradient, repeated_count = _one_pass(
            model_name=model_name,
            x_cpu=x_cpu,
            y_cpu=y_cpu,
            positions=positions,
            channel_names=channel_names,
        )
        if parameter_count != repeated_count:
            raise RuntimeError(f"{model_name} parameter count changed")
        if not torch.equal(first_logits, second_logits):
            maximum = float((first_logits - second_logits).abs().max())
            raise RuntimeError(
                f"{model_name} logits are not bitwise deterministic: {maximum}"
            )
        if not torch.equal(first_gradient, second_gradient):
            maximum = float((first_gradient - second_gradient).abs().max())
            raise RuntimeError(
                f"{model_name} gradients are not bitwise deterministic: {maximum}"
            )
        results[model_name] = {
            "parameter_count": parameter_count,
            "logits_sha256": _tensor_sha256(first_logits),
            "gradient_sha256": _tensor_sha256(first_gradient),
            "finite": True,
            "bitwise_repeat": True,
        }
    return {
        "device": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "models": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.cache_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
