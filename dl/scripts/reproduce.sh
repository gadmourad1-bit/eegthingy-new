#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

select_stdlib_python() {
    local ambient_python=""
    local candidate=""
    ambient_python="$(command -v python3 2>/dev/null || true)"
    for candidate in \
        "$ambient_python" \
        "$SCRIPT_DIR/../.venv/bin/python" \
        "$SCRIPT_DIR/../.venv-test/bin/python" \
        "$SCRIPT_DIR/../.venv-docs/bin/python"
    do
        [[ -n "$candidate" && -x "$candidate" ]] || continue
        if "$candidate" -I -B -S -c \
            'import sys, tomllib; raise SystemExit(sys.version_info < (3, 11))' \
            >/dev/null 2>&1
        then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "A Python 3.11+ interpreter is required for bundle verification." >&2
    echo "Install it or run scripts/reproduce.sh setup runtime first." >&2
    return 1
}

case "$command_name" in
    setup) exec "$SCRIPT_DIR/setup_uv.sh" "$@" ;;
    test) exec "$SCRIPT_DIR/run_tests.sh" "$@" ;;
    prepare-data) exec "$SCRIPT_DIR/prepare_data.sh" "$@" ;;
    preflight) exec "$SCRIPT_DIR/preflight_common_grid.sh" "$@" ;;
    run|resume) exec "$SCRIPT_DIR/run_common_grid.sh" "$@" ;;
    status) exec "$SCRIPT_DIR/status_common_grid.sh" "$@" ;;
    audit) exec "$SCRIPT_DIR/audit_common_grid.sh" "$@" ;;
    analyze) exec "$SCRIPT_DIR/analyze_common_grid.sh" "$@" ;;
    reviewer)
        if [[ "${1:-}" == "catalog" ]]; then
            shift
            export PYTHONDONTWRITEBYTECODE=1
            verifier_python="$(select_stdlib_python)" || exit 2
            exec "$verifier_python" -I -B -S \
                "$SCRIPT_DIR/generate_reviewer_score_cells.py" "$@"
        fi
        source "$SCRIPT_DIR/_common.sh"
        use_environment .venv
        # The formal runner records and verifies UV from PATH.  UV_BIN may be
        # the only discovery route in non-interactive SSH sessions, so expose
        # that already-validated executable to both the parent and GPU worker.
        export PATH="$(dirname -- "$UV_EXECUTABLE"):${PATH:-}"
        run_frozen python -m benchmark.reviewer_replay "$@"
        ;;
    reports)
        source "$SCRIPT_DIR/_common.sh"
        use_environment .venv-docs
        run_frozen --group docs python "$SCRIPT_DIR/build_reports.py" "$@"
        ;;
    identity)
        if [[ $# -ne 0 ]]; then
            echo "Usage: $0 identity" >&2
            exit 2
        fi
        export PYTHONDONTWRITEBYTECODE=1
        verifier_python="$(select_stdlib_python)" || exit 2
        exec "$verifier_python" -I -B -S "$SCRIPT_DIR/reproducibility_contract.py"
        ;;
    verify)
        if [[ $# -ne 0 ]]; then
            echo "Usage: $0 verify" >&2
            exit 2
        fi
        export PYTHONDONTWRITEBYTECODE=1
        verifier_python="$(select_stdlib_python)" || exit 2
        exec "$verifier_python" -I -B -S "$SCRIPT_DIR/verify_release.py"
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
  reviewer {catalog|list|estimate|run|status|compare} [arguments]
  reports [--output-dir DIRECTORY] [--validate-only]
  identity
  verify

Reviewer replay examples:
  scripts/reproduce.sh reviewer catalog --check
  scripts/reproduce.sh reviewer list
  scripts/reproduce.sh reviewer estimate --scope model --model MODEL
  scripts/reproduce.sh reviewer estimate --scope dataset --model MODEL --dataset DATASET
  scripts/reproduce.sh reviewer run --scope job --model MODEL --dataset DATASET \
    --subject SUBJECT --fold FOLD --seed SEED --cache-root CACHE_ROOT \
    --run-root RUN_ROOT --gpu GPU
  scripts/reproduce.sh reviewer status --run-root RUN_ROOT
  scripts/reproduce.sh reviewer compare --run-root RUN_ROOT --cache-root CACHE_ROOT

Reviewer scopes are explicit and bounded: model, dataset, subject, or job. The
reviewer command never launches the 96,320-job full grid.

The 96,320-job grid is intentionally separate from unit tests. Both run and
resume use the same audited, stale-claim-recovering runner.
EOF
        ;;
    *)
        echo "Unknown command: $command_name" >&2
        exit 2
        ;;
esac
