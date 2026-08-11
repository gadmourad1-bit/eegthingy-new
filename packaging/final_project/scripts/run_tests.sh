#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_common.sh"
use_environment .venv-test

run_frozen --group test python -m pytest -q -p no:cacheprovider \
  --import-mode=importlib "$@"
