"""Pose trace: compact NPZ recording of the robot pose over time.

Sampled every ``trace_period`` seconds of simulation time (default 50 ms),
this is enough to reconstruct / replay a run later. Arrays stored:

    t      (N,)  simulation time of each sample [s]
    x      (N,)  robot x [m]
    y      (N,)  robot y [m]
    yaw    (N,)  robot heading [rad]
    pitch  (N,)  robot pitch [rad]  (0 for this planar base; kept for 3D use)

Plus scalar metadata needed to rebuild the exact maze for replay:
    seed, n_turns, cell, wall_thickness, wall_height, min_gap, max_gap,
    trace_period, forward_speed, turn_speed.

The filename mirrors the CSV report: tiago_maze_<timestamp>_seed<seed>.npz
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

POSE_KEYS = ("t", "x", "y", "yaw", "pitch")


def trace_filename(seed, when: datetime | None = None) -> str:
    when = when or datetime.now()
    return f"tiago_maze_{when.strftime('%Y%m%d_%H%M%S')}_seed{seed}.npz"


def write_trace(
    out_dir,
    samples: dict,
    meta: dict,
    when: datetime | None = None,
) -> Path:
    """Write the pose trace. ``samples`` maps each POSE_KEYS name to a list."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / trace_filename(meta.get("seed"), when)

    arrays = {k: np.asarray(samples.get(k, []), dtype=np.float64) for k in POSE_KEYS}
    # Store metadata as 0-d arrays alongside the pose arrays.
    for k, v in meta.items():
        arrays[f"meta_{k}"] = np.asarray(v)

    np.savez_compressed(path, **arrays)
    return path


def load_trace(path):
    """Return (samples, meta) where samples has POSE_KEYS arrays and meta is a
    dict of the ``meta_*`` scalars decoded back to Python values."""
    data = np.load(path, allow_pickle=False)
    samples = {k: data[k] for k in POSE_KEYS if k in data}
    meta: dict = {}
    for key in data.files:
        if key.startswith("meta_"):
            arr = data[key]
            meta[key[len("meta_"):]] = arr.item() if arr.ndim == 0 else arr
    return samples, meta
