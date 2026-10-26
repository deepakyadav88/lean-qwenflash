#!/usr/bin/env bash
# Download a Qwen3.8-Flash-Next EXL3 branch onto a local SSD, then pack the routed experts into an O_DIRECT store.
#   scripts/download_model.sh /media/deepak/E/qwen                          # 3.05bpw_h5_ng5 (default)
#   scripts/download_model.sh /media/deepak/E/qwen-205 --branch=2.05bpw_h4_ng4
#   scripts/download_model.sh /media/deepak/E/qwen --pack                  # pack only (checkpoint already present)
#   scripts/download_model.sh /media/deepak/E/qwen --download-only         # no pack
# The store name follows the branch (qwen38_flash_exl3_<bits>bpw_experts) unless --store-name=... is given.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
WORK=""
BRANCH=${QWEN_MODEL_BRANCH:-3.05bpw_h5_ng5}
MODE=all
STORE_NAME=""

for a in "$@"; do
  case "$a" in
    --pack) MODE=pack ;;
    --download-only) MODE=download ;;
    --branch=*) BRANCH=${a#--branch=} ;;
    --store-name=*) STORE_NAME=${a#--store-name=} ;;
    *) [ -z "$WORK" ] && WORK=$a || { echo "[lean] unexpected argument: $a" >&2; exit 2; } ;;
  esac
done
[ -n "$WORK" ] || { echo "usage: scripts/download_model.sh <ssd-dir> [--branch=<branch>|--pack|--download-only]" >&2; exit 2; }

BITS=${BRANCH%%_*}                       # "2.05bpw_h4_ng4" -> "2.05bpw"
STORE_NAME=${STORE_NAME:-qwen38_flash_exl3_${BITS}_experts}
STORE_DIR=${STORE_DIR:-$WORK/store}

FREE=$(df -Pk "$WORK" | awk 'NR==2{printf "%.1f", $4/1048576}')
echo "[lean] $WORK has ${FREE} GiB free; checkpoint + ngram table ~63-85 GB, packed store ~29-43 GiB"

if [ "$MODE" != "pack" ]; then
  echo "[lean] downloading turboderp/Qwen3.8-Flash-Next-exl3 @ $BRANCH into $WORK/model"
  QWEN_WORK_ROOT="$WORK" QWEN_MODEL_BRANCH="$BRANCH" QWEN_DOWNLOAD_JOIN_CGROUP=auto \
    QWEN_DOWNLOAD_WORKERS=${QWEN_DOWNLOAD_WORKERS:-24} "$PY" "$ROOT/download_http.py"
fi

if [ "$MODE" != "download" ]; then
  mkdir -p "$STORE_DIR"
  echo "[lean] packing routed experts into $STORE_DIR/$STORE_NAME.bin"
  "$PY" "$ROOT/qwen38-flash-offload/scripts/pack_qwen38_store.py" pack \
    --model "$WORK/model" --out-dir "$STORE_DIR" --name "$STORE_NAME" --threads ${QWEN_PACK_THREADS:-8} --mtp ${QWEN_PACK_MTP:-1}
  "$PY" "$ROOT/qwen38-flash-offload/scripts/pack_qwen38_store.py" verify \
    --threads ${QWEN_PACK_THREADS:-8} --model "$WORK/model" --out-dir "$STORE_DIR" --name "$STORE_NAME"
  "$PY" "$ROOT/qwen38-flash-offload/scripts/pack_qwen38_store.py" cmp --sample 300 \
    --model "$WORK/model" --out-dir "$STORE_DIR" --name "$STORE_NAME"
  echo "[lean] store ready: $STORE_DIR/$STORE_NAME.bin"
fi

echo "[lean] budgets for this quant: scripts/hw_preflight.py --model=$WORK/model --store=$STORE_DIR/$STORE_NAME.bin --store-dir=$STORE_DIR --write=$WORK/hw.env"
