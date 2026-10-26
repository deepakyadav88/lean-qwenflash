#!/usr/bin/env bash
# Serve Qwen3.8 Flash Next on a consumer PC (RTX 3060 12 GB / 32 GB RAM class).
#   scripts/run_local.sh                      # 32k context, derived budgets, host RAM guard in proc mode
#   scripts/run_local.sh --context 65536      # bigger KV cache (costs VRAM: fewer expert slots)
#   scripts/run_local.sh --scope              # run inside a systemd memory cap (cgroup mode) instead of proc mode
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
MODEL=${MODEL:-/media/deepak/E/qwen/model}
STORE=${STORE:-/media/deepak/E/qwen/store/qwen38_flash_exl3_3.05bpw_experts.bin}
CONTEXT=${CONTEXT:-32768}
CHUNK=${CHUNK:-4096}
PORT=${PORT:-30000}
RAM_GB=${RAM_GB:-}

ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --scope) SCOPE=1; shift ;;
    --context) CONTEXT=$2; ARGS+=(--cache "$2"); shift 2 ;;
    --chunk) CHUNK=$2; ARGS+=(--chunk "$2"); shift 2 ;;
    --ram-gb) RAM_GB=$2; ARGS+=(--ram-gb "$2"); shift 2 ;;
    --port) PORT=$2; ARGS+=(--port "$2"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

[ -f "$MODEL/config.json" ] || { echo "[lean] no checkpoint at $MODEL - run scripts/download_model.sh first" >&2; exit 1; }
[ -f "$STORE" ] || { echo "[lean] no expert store at $STORE - run scripts/download_model.sh --pack first" >&2; exit 1; }

if [ ! -f "$ROOT/hw.env" ]; then
  "$PY" "$ROOT/scripts/hw_preflight.py" --model="$MODEL" --store="$STORE" --store-dir="$(dirname "$STORE")" \
    --context="$CONTEXT" --chunk="$CHUNK" --measure --write "$ROOT/hw.env"
fi
source "$ROOT/hw.env"

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${USED:-0}" -gt 1500 ]; then
  echo "[lean] WARNING: ${USED} MiB of VRAM is already in use (desktop session or another serving process)." >&2
  echo "[lean]          Stop it, or the engine will have no room for the expert cache." >&2
fi
AVAIL=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
echo "[lean] MemAvailable ${AVAIL} GiB, VRAM used ${USED} MiB, expert tier ${QWEN38_NV_RAM_GB} GiB, VRAM cap ${QWEN_VRAM_LIMIT_BYTES} bytes"

CMD=("$PY" "$ROOT/qwen_server.py" --model "$MODEL" --store "$STORE" --port "$PORT" "${ARGS[@]}")
if [ "${SCOPE:-0}" = "1" ]; then
  CAP=${QWEN_RAM_LIMIT_BYTES:-$((24 * 1024 * 1024 * 1024))}
  # A systemd scope gives the same hard cap + no swap the container mode used;
  # memory_guard.py resolves the scope's own cgroup from /proc/self/cgroup when QWEN_RAM_GUARD=cgroup.
  export QWEN_RAM_GUARD=cgroup
  UNIT=qwen-lean-$$
  systemd-run --scope --unit="$UNIT" -p MemoryMax="$CAP" -p MemorySwapMax=0 -p TasksMax=512 -- "${CMD[@]}"
else
  export QWEN_RAM_GUARD=${QWEN_RAM_GUARD:-proc}
  exec "${CMD[@]}"
fi
