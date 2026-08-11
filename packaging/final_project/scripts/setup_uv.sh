#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_common.sh"

mode="${1:-all}"
if [[ $# -gt 1 ]]; then
    fail "Usage: $0 [runtime|test|docs|all]"
fi

sync_environment() {
    local environment="$1"
    local environment_path="$PROJECT_ROOT/$environment"
    shift
    case "$environment" in
        .venv|.venv-test|.venv-docs) ;;
        *) fail "unsupported project environment name: $environment" ;;
    esac
    if [[ -L "$environment_path" \
          || ( -e "$environment_path" && ! -d "$environment_path" ) ]]; then
        fail "Refusing unsafe project environment path: $environment_path"
    fi
    echo "Synchronizing $environment from the frozen lockfile"
    UV_PROJECT_ENVIRONMENT="$environment_path" \
        PYTHONNOUSERSITE=1 \
        "$UV_EXECUTABLE" sync \
            --project "$PROJECT_ROOT" \
            --frozen \
            --no-default-groups \
            "$@"
    if [[ -L "$environment_path" || ! -x "$environment_path/bin/python" \
          || -L "$environment_path/pyvenv.cfg" \
          || ! -f "$environment_path/pyvenv.cfg" ]]; then
        fail "uv did not create a valid project-private environment: $environment_path"
    fi
    if ! grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*false[[:space:]]*$' \
        "$environment_path/pyvenv.cfg"; then
        fail "Environment is not isolated from system site packages: $environment_path"
    fi
    "$UV_EXECUTABLE" pip check --python "$environment_path/bin/python"
}

"$UV_EXECUTABLE" lock --check --project "$PROJECT_ROOT"

case "$mode" in
    runtime)
        sync_environment .venv
        ;;
    test)
        sync_environment .venv-test --group test
        ;;
    docs)
        sync_environment .venv-docs --group docs
        ;;
    all)
        sync_environment .venv
        sync_environment .venv-test --group test
        sync_environment .venv-docs --group docs
        ;;
    *)
        fail "Usage: $0 [runtime|test|docs|all]"
        ;;
esac

echo "UV environments are project-private; no system Python packages were changed."
