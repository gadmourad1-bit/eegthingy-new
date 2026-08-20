#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_common.sh"
use_environment .venv

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 RUN_ROOT" >&2
    exit 2
fi

RUN_ROOT="$(canonical_existing_directory "RUN_ROOT" "$1")"
require_outside_path "RUN_ROOT" "$RUN_ROOT" "$PROJECT_ROOT"

run_frozen python -m benchmark.full_grid status --run-root "$RUN_ROOT"
