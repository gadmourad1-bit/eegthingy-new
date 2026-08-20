#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_common.sh"
use_environment .venv

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 RUN_ROOT [AUDIT_JSON]" >&2
    exit 2
fi

RUN_ROOT="$(canonical_existing_directory "RUN_ROOT" "$1")"
require_outside_path "RUN_ROOT" "$RUN_ROOT" "$PROJECT_ROOT"
if [[ $# -eq 2 ]]; then
    AUDIT_JSON="$(canonical_new_file "AUDIT_JSON" "$2")"
    require_outside_path "AUDIT_JSON" "$AUDIT_JSON" "$PROJECT_ROOT"
    require_outside_path "AUDIT_JSON" "$AUDIT_JSON" "$RUN_ROOT"
    run_frozen python -m benchmark.full_grid audit \
        --run-root "$RUN_ROOT" \
        --output "$AUDIT_JSON"
else
    run_frozen python -m benchmark.full_grid audit --run-root "$RUN_ROOT"
fi
