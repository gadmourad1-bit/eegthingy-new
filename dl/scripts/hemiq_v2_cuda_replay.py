"""Run the score-free HemiQ-v2 synthetic CUDA replay under a project GPU lease."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark import hemiq_v2_grid as grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()

    project_root = grid._absolute(arguments.project_root)
    run_root = grid._safe_mkdir(arguments.run_root)
    replay_identity = {
        "config_file_sha256": grid._config_identity(project_root)[
            "config_file_sha256"
        ],
        "purpose": "synthetic_only_no_benchmark_data_or_score",
        "schema": "eeg-mi-hemiq-v2-synthetic-cuda-replay-v1",
        "source_sha256": grid.collect_source_identity(project_root)[
            "src/benchmark/hemiq_v2_grid.py"
        ]["sha256"],
    }
    plan_sha256 = hashlib.sha256(
        grid._canonical_bytes(replay_identity)
    ).hexdigest()
    lease = grid.project_gpu_leases.acquire_gpu_lease(
        project_root=project_root,
        run_root=run_root,
        plan_sha256=plan_sha256,
        gpu_uuid=arguments.gpu_uuid,
        track_scope="hemiq-v2-synthetic",
    )
    try:
        grid.project_gpu_leases.assert_gpu_lease(lease)
        checks = grid._synthetic_cuda_preflight(
            project_root=project_root,
            device=arguments.device,
        )
        grid.project_gpu_leases.assert_gpu_lease(lease)
    finally:
        grid.project_gpu_leases.release_gpu_lease(lease)
    print(
        json.dumps(
            {
                "checks": checks,
                "gpu_uuid": arguments.gpu_uuid,
                "identity": replay_identity,
                "status": "pass_no_benchmark_data_or_score",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
