#!/usr/bin/env bash
# Pre-build the native extensions for THIS machine: CUDA arch from the GPU, -mtune from the CPU.
#   scripts/build_extensions.sh [model-dir]        (MODEL=... also works)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
MODEL=${MODEL:-/media/deepak/E/qwen/model}

"$PY" "$ROOT/hw_profile.py" --model="$MODEL" --write "$ROOT/hw.env"
source "$ROOT/hw.env"
echo "[lean] building extensions: arch=$TORCH_CUDA_ARCH_LIST cpu-tune=$QWEN38_CPUTUNE"
MAX_JOBS=${MAX_JOBS:-4} bash "$ROOT/qwen38-flash-offload/scripts/install.sh" "$ROOT/qwen38-flash-offload"
echo "[lean] extensions built under qwen38-flash-offload/build/"
