#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_common.sh"
use_environment .venv

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 RUN_ROOT CACHE_ROOT NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi

RUN_ROOT="$(canonical_existing_directory "RUN_ROOT" "$1")"
CACHE_ROOT="$(canonical_existing_directory "CACHE_ROOT" "$2")"
OUTPUT_DIRECTORY="$(canonical_new_file "analysis output directory" "$3")"

require_outside_path "RUN_ROOT" "$RUN_ROOT" "$PROJECT_ROOT"
require_outside_path "CACHE_ROOT" "$CACHE_ROOT" "$PROJECT_ROOT"
require_disjoint_paths "RUN_ROOT" "$RUN_ROOT" "CACHE_ROOT" "$CACHE_ROOT"
require_outside_path "analysis output directory" "$OUTPUT_DIRECTORY" "$PROJECT_ROOT"
require_outside_path "analysis output directory" "$OUTPUT_DIRECTORY" "$RUN_ROOT"
require_outside_path "analysis output directory" "$OUTPUT_DIRECTORY" "$CACHE_ROOT"

run_frozen python -m benchmark.full_grid_analysis \
    --run-root "$RUN_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --output-dir "$OUTPUT_DIRECTORY"
