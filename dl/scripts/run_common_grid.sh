#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_common.sh"
use_environment .venv

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 CACHE_ROOT RUN_ROOT [GPU_LIST]" >&2
    echo "Example GPU_LIST: 0,1,2" >&2
    exit 2
fi

CACHE_ROOT="$(canonical_existing_directory "CACHE_ROOT" "$1")"
RUN_ROOT="$(canonical_existing_directory "RUN_ROOT" "$2")"
GPU_LIST="${3:-0,1,2}"

require_outside_path "CACHE_ROOT" "$CACHE_ROOT" "$PROJECT_ROOT"
require_outside_path "RUN_ROOT" "$RUN_ROOT" "$PROJECT_ROOT"
require_disjoint_paths "CACHE_ROOT" "$CACHE_ROOT" "RUN_ROOT" "$RUN_ROOT"

run_frozen python -m benchmark.full_grid run \
    --cache-root "$CACHE_ROOT" \
    --run-root "$RUN_ROOT" \
    --gpus "$GPU_LIST" \
    --recover-stale
