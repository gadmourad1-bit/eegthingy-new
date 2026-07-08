# Builds a standalone Linux/arm64 binary of the lean SBC decoder.
# Build context is the repo root (see build.sh) so config.py is available.
FROM --platform=linux/arm64 python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# BrainFlow ships no ARM64 wheel — build the native libs from source (aarch64),
# then install its Python package (which bundles the freshly built .so files).
RUN git clone --depth 1 https://github.com/brainflow-dev/brainflow.git /brainflow \
 && cd /brainflow/tools && python3 build.py \
 && pip install --no-cache-dir /brainflow/python_package

# Runtime deps (lean — no mne/scikit-learn/matplotlib) + the bundler.
RUN pip install --no-cache-dir numpy scipy websockets pyserial pyinstaller

WORKDIR /app
COPY sbc/lean_online.py sbc/ws.py sbc/smoother.py config.py /app/

# One self-contained executable at /app/dist/lean-online.
# --collect-all serial: pyserial is imported lazily (auto-detect), so force-bundle it.
RUN pyinstaller --onefile --name lean-online \
        --collect-all brainflow --collect-all serial \
        --paths /app /app/lean_online.py
