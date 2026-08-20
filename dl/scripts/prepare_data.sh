#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_common.sh"
use_environment .venv

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 CACHE_ROOT [LOCAL_EXP4_DATA_ROOT]" >&2
    exit 2
fi

CACHE_ROOT="$(canonical_directory_target "CACHE_ROOT" "$1")"
require_outside_path "CACHE_ROOT" "$CACHE_ROOT" "$PROJECT_ROOT"
mkdir -p -- "$CACHE_ROOT"
CACHE_ROOT="$(canonical_existing_directory "CACHE_ROOT" "$CACHE_ROOT")"
LOCAL_ROOT="${2:-}"

build_cache() {
    local dataset="$1"
    local subjects="$2"
    shift 2
    run_frozen python -m benchmark.data \
        --dataset "$dataset" \
        --subjects "$subjects" \
        --cache-root "$CACHE_ROOT" \
        --montage-profile harmonized \
        "$@"
}

build_cache bnci2014_001 1,2,3,4,5,6,7,8,9
build_cache bnci2014_004 1,2,3,4,5,6,7,8,9
build_cache cho2017 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52
build_cache physionet_mi 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54

if [[ -n "$LOCAL_ROOT" ]]; then
    LOCAL_ROOT="$(canonical_existing_directory "LOCAL_EXP4_DATA_ROOT" "$LOCAL_ROOT")"
    require_disjoint_paths "CACHE_ROOT" "$CACHE_ROOT" \
        "LOCAL_EXP4_DATA_ROOT" "$LOCAL_ROOT"
    build_cache local_exp4 1,3,4,5,6,7,8,10 --local-data-root "$LOCAL_ROOT"
else
    echo "LOCAL_EXP4_DATA_ROOT was not supplied; the private local dataset was not cached." >&2
fi
