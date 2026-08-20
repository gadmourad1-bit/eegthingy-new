#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPOSITORY_PYTHON="$ROOT_DIR/../.venv/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "$REPOSITORY_PYTHON" ]]; then
    PYTHON_BIN="$REPOSITORY_PYTHON"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
TECTONIC_BIN="${TECTONIC_BIN:-/private/tmp/tectonic-0.16.9/tectonic}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1787184000}"
export TZ="${TZ:-UTC}"
export PYTHONDONTWRITEBYTECODE=1
MAIN_TEX="$ROOT_DIR/main.tex"
ASSET_BUILDER="$ROOT_DIR/scripts/build_assets.py"
LOGO_CONVERTER="$ROOT_DIR/scripts/convert_template_logo.py"
CHECKER="$ROOT_DIR/scripts/check_manuscript.py"
OUTPUT_DIR="$ROOT_DIR/output"
OUTPUT_PDF="$OUTPUT_DIR/TBME_Internal_Working_Draft.pdf"

for required in "$MAIN_TEX" "$ASSET_BUILDER" "$LOGO_CONVERTER" "$CHECKER"; do
    if [[ ! -f "$required" ]]; then
        echo "error: required file is missing: $required" >&2
        exit 2
    fi
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: Python executable not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -x "$TECTONIC_BIN" ]]; then
    echo "error: Tectonic executable is unavailable: $TECTONIC_BIN" >&2
    exit 2
fi

(
    cd "$ROOT_DIR"
"$PYTHON_BIN" "$ASSET_BUILDER"
"$PYTHON_BIN" "$LOGO_CONVERTER"
)

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tbme-manuscript.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT
mkdir -p "$OUTPUT_DIR"

(
    cd "$ROOT_DIR"
    "$TECTONIC_BIN" --keep-logs --outdir "$BUILD_DIR" "$MAIN_TEX"
)

BUILT_PDF="$BUILD_DIR/main.pdf"
if [[ ! -f "$BUILT_PDF" ]]; then
    echo "error: Tectonic did not produce $BUILT_PDF" >&2
    exit 2
fi

install -m 0644 "$BUILT_PDF" "$OUTPUT_PDF"
"$PYTHON_BIN" "$CHECKER" --tex "$MAIN_TEX" --pdf "$OUTPUT_PDF"

echo "built: $OUTPUT_PDF"
