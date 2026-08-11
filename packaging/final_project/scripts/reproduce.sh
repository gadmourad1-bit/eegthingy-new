#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    setup) exec "$SCRIPT_DIR/setup_uv.sh" "$@" ;;
    test) exec "$SCRIPT_DIR/run_tests.sh" "$@" ;;
    prepare-data) exec "$SCRIPT_DIR/prepare_data.sh" "$@" ;;
    preflight) exec "$SCRIPT_DIR/preflight_common_grid.sh" "$@" ;;
    run|resume) exec "$SCRIPT_DIR/run_common_grid.sh" "$@" ;;
    status) exec "$SCRIPT_DIR/status_common_grid.sh" "$@" ;;
    audit) exec "$SCRIPT_DIR/audit_common_grid.sh" "$@" ;;
    analyze) exec "$SCRIPT_DIR/analyze_common_grid.sh" "$@" ;;
    reports)
        source "$SCRIPT_DIR/_common.sh"
        use_environment .venv-docs
        run_frozen --group docs python "$SCRIPT_DIR/build_reports.py" "$@"
        ;;
    verify)
        source "$SCRIPT_DIR/_common.sh"
        use_environment .venv
        run_frozen python "$SCRIPT_DIR/verify_release.py" "$@"
        ;;
    help|-h|--help)
        cat <<'EOF'
Usage: scripts/reproduce.sh COMMAND [ARGS...]

Commands:
  setup [runtime|test|docs|all]
  test [pytest arguments]
  prepare-data CACHE_ROOT [LOCAL_EXP4_DATA_ROOT]
  preflight CACHE_ROOT RUN_ROOT [GPU_INDEX]
  run CACHE_ROOT RUN_ROOT [GPU_LIST]
  resume CACHE_ROOT RUN_ROOT [GPU_LIST]
  status RUN_ROOT
  audit RUN_ROOT [AUDIT_JSON]
  analyze RUN_ROOT CACHE_ROOT NEW_OUTPUT_DIRECTORY
  reports [--output-dir DIRECTORY] [--validate-only]
  verify

The 96,320-job grid is intentionally separate from unit tests. Both run and
resume use the same audited, stale-claim-recovering runner.
EOF
        ;;
    *)
        echo "Unknown command: $command_name" >&2
        exit 2
        ;;
esac
