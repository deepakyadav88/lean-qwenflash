#!/usr/bin/env bash
# Download the 3.05bpw checkpoint onto a local SSD, then pack the routed experts into an O_DIRECT store.
#   scripts/download_model.sh /media/deepak/E/qwen            download (~85 GB) + pack (~43 GiB)
#   scripts/download_model.sh /media/deepak/E/qwen --pack     pack only (checkpoint already present)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
WORK=${1:?usage: scripts/download_model.sh <ssd-dir> [--pack|--download-only]}
MODE=${2:-all}

FREE=$(df -Pk "$WORK" | awk 'NR==2{printf "%.1f", $4/1048576}')
echo "[lean] $WORK has ${FREE} GiB free; checkpoint ~85 GB, packed store ~43 GiB"

if [ "$MODE" != "--pack" ]; then
  echo "[lean] downloading turboderp/Qwen3.8-Flash-Next-exl3 @ 3.05bpw_h5_ng5 into $WORK/model"
  QWEN_WORK_ROOT="$WORK" QWEN_DOWNLOAD_JOIN_CGROUP=auto QWEN_DOWNLOAD_WORKERS=${QWEN_DOWNLOAD_WORKERS:-24} \
    "$PY" "$ROOT/download_http.py"
fi

STORE_DIR=${STORE_DIR:-$WORK/store}
mkdir -p "$STORE_DIR"
echo "[lean] packing routed experts into $STORE_DIR"
"$PY" "$ROOT/qwen38-flash-offload/scripts/pack_qwen38_store.py" pack \
  --model "$WORK/model" --out-dir "$STORE_DIR" --threads ${QWEN_PACK_THREADS:-8} --mtp ${QWEN_PACK_MTP:-1}
"$PY" "$ROOT/qwen38-flash-offload/scripts/pack_qwen38_store.py" verify \
  --threads ${QWEN_PACK_THREADS:-8} --model "$WORK/model" --out-dir "$STORE_DIR"
echo "[lean] store ready: $STORE_DIR/qwen38_flash_exl3_3.05bpw_experts.bin"
