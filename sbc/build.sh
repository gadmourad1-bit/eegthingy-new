#!/usr/bin/env bash
# Build the standalone Linux/arm64 SBC binary on an Apple-Silicon Mac (via Docker).
# Produces sbc/lean-online-linux-arm64 — copy it + model.npz to the SBC.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, so config.py is in the build context

echo "==> building Linux/arm64 image (compiles brainflow from source; first run is slow)…"
docker build --platform linux/arm64 -t eeg-sbc-build -f sbc/Dockerfile .

echo "==> extracting the binary…"
cid=$(docker create --platform linux/arm64 eeg-sbc-build)
docker cp "$cid:/app/dist/lean-online" ./sbc/lean-online-linux-arm64
docker rm "$cid" >/dev/null

chmod +x ./sbc/lean-online-linux-arm64
echo ""
echo "built: sbc/lean-online-linux-arm64  ($(du -h ./sbc/lean-online-linux-arm64 | cut -f1))"
echo "copy this + sbc/model.npz to the SBC, then:  ./lean-online-linux-arm64"
