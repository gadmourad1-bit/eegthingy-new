#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
EXPECTED_UV_VERSION="0.12.0"
EXPECTED_CUBLAS_WORKSPACE_CONFIG=":4096:8"
EXPECTED_TCFORMER_ROOT="$PROJECT_ROOT/third_party/TCFormer"

fail() {
    echo "$*" >&2
    exit 2
}

if [[ -n "${UV_BIN:-}" ]]; then
    if [[ "$UV_BIN" != /* ]]; then
        fail "UV_BIN must be an absolute executable path."
    fi
    UV_EXECUTABLE="$UV_BIN"
else
    UV_EXECUTABLE="$(command -v uv || true)"
fi

if [[ -z "$UV_EXECUTABLE" || ! -x "$UV_EXECUTABLE" ]]; then
    fail "uv was not found. Install uv or set UV_BIN to its absolute path."
fi

UV_VERSION_OUTPUT="$("$UV_EXECUTABLE" --version)" || fail "unable to execute uv"
if [[ "$UV_VERSION_OUTPUT" != "uv $EXPECTED_UV_VERSION" \
      && "$UV_VERSION_OUTPUT" != "uv $EXPECTED_UV_VERSION "* ]]; then
    fail "Expected uv $EXPECTED_UV_VERSION, observed: $UV_VERSION_OUTPUT"
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONPATH="$PROJECT_ROOT/src"

if [[ -n "${CUBLAS_WORKSPACE_CONFIG:-}" \
      && "$CUBLAS_WORKSPACE_CONFIG" != "$EXPECTED_CUBLAS_WORKSPACE_CONFIG" ]]; then
    fail "CUBLAS_WORKSPACE_CONFIG must equal $EXPECTED_CUBLAS_WORKSPACE_CONFIG."
fi
export CUBLAS_WORKSPACE_CONFIG="$EXPECTED_CUBLAS_WORKSPACE_CONFIG"

if [[ -n "${EEG_MI_TCFORMER_ROOT:-}" \
      && "$EEG_MI_TCFORMER_ROOT" != "$EXPECTED_TCFORMER_ROOT" ]]; then
    fail "EEG_MI_TCFORMER_ROOT must use the vendored release snapshot."
fi
export EEG_MI_TCFORMER_ROOT="$EXPECTED_TCFORMER_ROOT"

use_environment() {
    local name="$1"
    local expected="$PROJECT_ROOT/$name"
    case "$name" in
        .venv|.venv-test|.venv-docs) ;;
        *) fail "unsupported project environment name: $name" ;;
    esac
    if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" \
          && "$UV_PROJECT_ENVIRONMENT" != "$expected" ]]; then
        fail "UV_PROJECT_ENVIRONMENT must equal $expected for this command."
    fi
    if [[ -L "$expected" || ! -d "$expected" \
          || ! -x "$expected/bin/python" || -L "$expected/pyvenv.cfg" \
          || ! -f "$expected/pyvenv.cfg" ]]; then
        fail "Project environment is missing or unsafe: $expected. Run scripts/setup_uv.sh first."
    fi
    if ! grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*false[[:space:]]*$' \
        "$expected/pyvenv.cfg"; then
        fail "Project environment enables or does not declare isolated site packages: $expected"
    fi
    export UV_PROJECT_ENVIRONMENT="$expected"
}

run_frozen() {
    if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
        fail "use_environment must be called before run_frozen"
    fi
    "$UV_EXECUTABLE" run \
        --project "$PROJECT_ROOT" \
        --offline \
        --frozen \
        --no-sync \
        --no-default-groups \
        "$@"
}

canonical_existing_directory() {
    local label="$1"
    local value="$2"
    if [[ -L "$value" || ! -d "$value" ]]; then
        fail "$label must be an existing real directory: $value"
    fi
    (cd -- "$value" && pwd -P)
}

canonical_directory_target() {
    local label="$1"
    local value="$2"
    local parent
    local leaf
    if [[ -e "$value" || -L "$value" ]]; then
        canonical_existing_directory "$label" "$value"
        return
    fi
    parent="$(dirname -- "$value")"
    leaf="$(basename -- "$value")"
    if [[ "$leaf" == "." || "$leaf" == ".." || -z "$leaf" ]]; then
        fail "$label has an invalid final path component: $value"
    fi
    parent="$(canonical_existing_directory "$label parent" "$parent")"
    printf '%s/%s\n' "$parent" "$leaf"
}

canonical_new_file() {
    local label="$1"
    local value="$2"
    local parent
    local leaf
    if [[ -e "$value" || -L "$value" ]]; then
        fail "$label must not already exist: $value"
    fi
    parent="$(dirname -- "$value")"
    leaf="$(basename -- "$value")"
    if [[ "$leaf" == "." || "$leaf" == ".." || -z "$leaf" ]]; then
        fail "$label has an invalid final path component: $value"
    fi
    parent="$(canonical_existing_directory "$label parent" "$parent")"
    printf '%s/%s\n' "$parent" "$leaf"
}

require_outside_path() {
    local label="$1"
    local candidate="$2"
    local protected="$3"
    case "$candidate" in
        "$protected"|"$protected"/*)
            fail "$label must be outside $protected"
            ;;
    esac
}

require_disjoint_paths() {
    local first_label="$1"
    local first="$2"
    local second_label="$3"
    local second="$4"
    case "$first" in
        "$second"|"$second"/*)
            fail "$first_label must not equal or be inside $second_label"
            ;;
    esac
    case "$second" in
        "$first"|"$first"/*)
            fail "$second_label must not equal or be inside $first_label"
            ;;
    esac
}

cd "$PROJECT_ROOT"
