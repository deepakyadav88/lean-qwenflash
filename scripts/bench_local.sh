#!/usr/bin/env bash
# Sweep context lengths on one machine: restart the server at each KV size, run bench/sweep.py against it,
# and collect every run into bench_out/. Context is a restart knob (KV bytes = 6,144 B/token at 4-bit).
#
#   scripts/bench_local.sh                                  # 16k 32k 65k 131k, prefill 8k/32k/64k, decode C1 C2
#   scripts/bench_local.sh --contexts "32768 65536 262144"  # bigger contexts
#   scripts/bench_local.sh --conc "1 2 4" --reps 3 --dec-reps 2
#   scripts/bench_local.sh --config-label "ram20 prefetch-on"   # what you are labelling this sweep
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
MODEL=${MODEL:-/media/deepak/E/qwen/model}
STORE=${STORE:-/media/deepak/E/qwen/store/qwen38_flash_exl3_3.05bpw_experts.bin}
CONTEXTS="16384 32768 65536 131072"
PREFILL="8192 32768 65536"
CONC="1 2"
REPS=2
DEC_REPS=2
PORT=${PORT:-30000}
CARD=${CARD:-rtx3060}
LABEL=${LABEL:-"hw-derived profile"}
OUT=${OUT:-$ROOT/bench_out}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1200}

while [ $# -gt 0 ]; do
  case "$1" in
    --contexts) CONTEXTS=$2; shift 2 ;;
    --prefill) PREFILL=$2; shift 2 ;;
    --conc) CONC=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --dec-reps) DEC_REPS=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --card) CARD=$2; shift 2 ;;
    --config-label) LABEL=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    *) shift ;;
  esac
done

[ -f "$MODEL/config.json" ] || { echo "[bench] no checkpoint at $MODEL" >&2; exit 1; }
[ -f "$STORE" ] || { echo "[bench] no expert store at $STORE" >&2; exit 1; }
mkdir -p "$OUT"

# Same-session comparison only: arms measured hours apart drift (docs/how-it-works.md learned this the hard way).
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
AVAIL=$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)
if [ "${USED:-0}" -gt 1500 ]; then
  echo "[bench] ${USED} MiB of VRAM is in use by something else; stop it (this GPU needs ~11 GiB)." >&2; exit 1
fi
if ! ulimit -l unlimited 2>/dev/null; then echo "[bench] memlock is $(ulimit -l) KiB; run with ulimit -l unlimited" >&2; exit 1; fi

[ -f "$ROOT/hw.env" ] || "$PY" "$ROOT/scripts/hw_preflight.py" --model="$MODEL" --store="$STORE" \
  --store-dir="$(dirname "$STORE")" --write "$ROOT/hw.env" >/dev/null
source "$ROOT/hw.env"
echo "[bench] MemAvailable ${AVAIL} GiB, tier ${QWEN38_NV_RAM_GB} GiB, VRAM cap ${QWEN_VRAM_LIMIT_BYTES}, lane ${QWEN38_NV_CPU} cpus ${QWEN38_NV_CPU_CPUS}"

stop_server() {
  if [ -n "${SRV_PID:-}" ]; then kill -TERM "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; fi
  SRV_PID=""
  # the port may linger while CUDA tears down
  for _ in $(seq 1 30); do
    curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$PORT/health" || return 0
    sleep 2
  done
  fuser -k -n tcp "$PORT" 2>/dev/null || true
  sleep 3
}

for CTX in $CONTEXTS; do
  echo "=== context $CTX ==="
  "$PY" "$ROOT/qwen_server.py" --model "$MODEL" --store "$STORE" --cache "$CTX" --port "$PORT" \
      > "$OUT/server_${CTX}.log" 2>&1 &
  SRV_PID=$!
  ready=0
  for _ in $(seq 1 "$((LOAD_TIMEOUT / 2))"); do
    if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q .; then ready=1; break; fi
    sleep 2
  done
  [ "$ready" = 1 ] || { echo "[bench] server did not come up at context $CTX in ${LOAD_TIMEOUT}s; see $OUT/server_${CTX}.log" >&2; stop_server; continue; }

  "$PY" "$ROOT/qwen38-flash-offload/bench/sweep.py" --url "http://127.0.0.1:$PORT" \
    --card "$CARD" --config "$LABEL ctx=$CTX ram=$QWEN38_NV_RAM_GB vram=$QWEN_VRAM_LIMIT_BYTES lane=$QWEN38_NV_CPU" \
    --template qwen --prefill $PREFILL --conc $CONC --reps "$REPS" --dec-reps "$DEC_REPS" \
    --no-early-exit --out "$OUT/sweep_${CTX}.json" || echo "[bench] sweep at $CTX exited non-zero (see $OUT/sweep_${CTX}.json)"

  curl -s --max-time 3 "http://127.0.0.1:$PORT/memory" > "$OUT/memory_${CTX}.json" || true
  curl -s --max-time 3 "http://127.0.0.1:$PORT/stats" > "$OUT/stats_${CTX}.json" || true
  stop_server
  sleep 5
done

"$PY" - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
rows = []
for f in sorted(os.listdir(out)):
    if not f.startswith("sweep_") or not f.endswith(".json"):
        continue
    d = json.load(open(os.path.join(out, f)))
    mem = {}
    m = os.path.join(out, f.replace("sweep_", "memory_"))
    if os.path.exists(m):
        mem = json.load(open(m))
    pre = {k: v.get("median") for k, v in (d.get("prefill") or {}).items() if isinstance(v, dict) and "median" in v}
    dec = {k: v.get("aggregate") for k, v in (d.get("decode") or {}).items()}
    rows.append((d.get("ctx_len"), pre, dec, mem, d.get("status")))
print(f"{'ctx':>8}  {'prefill tok/s':>30}  {'decode agg tok/s':>26}  {'ram peak':>10} {'vram peak':>10}")
for ctx, pre, dec, mem, status in rows:
    p = " ".join(f"{k}:{v}" for k, v in sorted(pre.items(), key=lambda x: int(x[0])))
    g = " ".join(f"{k}:{v}" for k, v in sorted(dec.items()))
    rp = f"{mem.get('ram_sampled_peak', 0) / 2**30:.1f} GiB" if mem else "-"
    vp = f"{mem.get('total_vram_peak', 0) / 2**30:.1f} GiB" if mem else "-"
    print(f"{ctx!s:>8}  {p:>30}  {g:>26}  {rp:>10} {vp:>10}  {status}")
PY
