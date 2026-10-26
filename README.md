# Lean-QwenFlash

**Qwen3.8-Flash-Next at full 262k context on 8 GB RAM and 16 GB VRAM—20 tok/s generation and 1,600 tok/s prefill.**

![Qwen3.8-Flash-Next inference showcase](media/showcase.gif)

*Inference run on an RTX 4090, capped at 16 GB VRAM and 8 GB system RAM.*

Direct-from-SSD expert offloading for **Qwen3.8-Flash-Next** (125B MoE, 512 experts).

Most setups for this model require 48–64 GB of system RAM just to hold the expert weights in memory. This runtime streams uncached experts directly from NVMe storage over PCIe into a bounded GPU cache via `O_DIRECT`, while streaming the 51B N-gram embedding table from disk. All expert computation runs on the GPU.

Built on [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and adapted from the [glm53-flash-offload](https://github.com/sybil-solutions/glm53-flash-offload) engine.

---

## Measured Performance

Benchmarked using `turboderp/Qwen3.8-Flash-Next-exl3` (`3.05bpw_h5_ng5`) with 4-bit KV cache:

| Hardware | Context | System RAM Used | VRAM Used | Prefill | Decode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 4090** | 8k | ~8 GB | <16 GB | 1,080 tok/s | 28–33 tok/s (prose)<br>19–22 tok/s (code) |
| **RTX 4090** | 100k | ~8 GB | 15.6 GB | 1,536 tok/s | 20.8 tok/s |
| **RTX 4090** | 262k | ~8 GB | 15.6 GB | 1,627 tok/s | 20.0 tok/s |
| **RTX 3080 Laptop (16GB)** | 100k (8 GB cache) | 19.0 GB | 15.5 GB | 647 tok/s | 17.6 tok/s |
| **RTX 3080 Laptop (16GB)** | 100k (16 GB cache) | 23.9 GB | 15.5 GB | 693 tok/s | 20.9 tok/s |

---

## How It Works

Qwen3.8-Flash-Next activates only 10 of its 512 experts per layer per token (~2% of weights).

* **GPU (16 GB VRAM):** Dense attention mixers, DeltaNet state, routers, shared experts, output head, 4-bit KV cache, and an adaptive GPU expert cache.
* **Host RAM:** Configurable cache (4–16 GB) holding the most recently activated experts.
* **NVMe Storage:** The full ~47 GB packed expert store and 32 GB N-gram table sit on SSD. Cache misses are read directly via asynchronous ring buffers and staged ahead of the compute stream.
* **GPU-Only Math:** All expert computation stays on the GPU. Misses stream over PCIe rather than falling back to CPU execution, avoiding host DRAM bandwidth limits.
* **Prefix Caching:** Recurrent state checkpoints are saved alongside the KV cache so multi-turn agent requests start decoding immediately (~0.8s–1.4s TTFT on cached follow-ups).

---

## Quickstart

### 1. Requirements
* Linux (x86_64)
* NVIDIA GPU with at least 16 GB VRAM
* Fast NVMe SSD (PCIe 3.0 or 4.0, ~3+ GB/s read throughput)
* CUDA 12.4+ and Python 3.10–3.12

### 2. Install

```bash
git clone https://github.com/MaxDevv/lean-qwenflash.git
cd lean-qwenflash

python3 -m venv .venv
source .venv/bin/activate
pip install torch fastapi uvicorn safetensors jinja2
pip install -e exllamav3
```

Compile the native offload extensions:
```bash
python compile_local.py
```

### 3. Download & Build Store

Download the 3.05bpw checkpoint and N-gram table (~85 GB total). It writes `$QWEN_WORK_ROOT/model` and is
resumable (byte-range parts + per-file chunk state):
```bash
QWEN_WORK_ROOT=./ python download_http.py
```

Pack the routed experts into a direct-I/O binary store (~43.5 GiB):
```bash
python qwen38-flash-offload/scripts/pack_qwen38_store.py pack \
  --model ./model --out-dir ./store
```

On a consumer PC (12 GB VRAM, 32 GB RAM, one NVMe) the sizes above are derived from the machine instead of
copied from the measured server. `scripts/download_model.sh <ssd-dir>` downloads and packs, and
`docs/tuning-rtx3060.md` explains every lever:

```bash
scripts/download_model.sh /media/deepak/E/qwen
scripts/build_extensions.sh
scripts/run_local.sh
```

---

## Step-by-step on this machine (RTX 3060 12 GB, Ryzen 5 3600, one NVMe)

Every budget below is derived from the host by `hw_preflight.py` and written to `hw.env`, which `run_local.sh` and
`bench_local.sh` source. Commands assume the repo root and the NVMe mount `/media/deepak/E`. The checkpoint and the
store on this machine are downloaded and verified; **the serving and benchmark steps have not been run here yet**, so
the throughput figures are derived from bandwidth, not measured.

**0. Free the GPU and the RAM.** The engine needs ~11 GiB of VRAM and ~24 GiB of RAM, so nothing else can hold them.
```bash
pgrep -af strata                       # e.g. strata-host: ~10.6 GiB VRAM, ~27 GiB RAM
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # must read ~700 MiB (desktop only)
ulimit -l unlimited                    # memlock is 4 GiB here; the RAM tier is cudaHostRegister-ed
```

**1. Environment.**
```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install numpy safetensors huggingface_hub requests jinja2 fastapi uvicorn psutil
.venv/bin/pip install torch==2.11.0 triton==3.6.0 ninja
source .venv/bin/activate        # later steps write plain `python`; use .venv/bin/python if you skip this
```

**2. Checkpoint to NVMe (~85 GB for 3.05bpw, resumable, sha256-checked against the HF LFS oid).**
```bash
scripts/download_model.sh /media/deepak/E/qwen                     # download + pack + verify (branch 3.05bpw_h5_ng5)
scripts/download_model.sh /media/deepak/E/qwen --download-only     # just the checkpoint
# re-run the same command after an interrupted download; look for: VERIFY_DONE ... bad: []
```

**3. Expert store to NVMe (25,088 records, 46.76 GB, O_DIRECT).**
```bash
.venv/bin/python qwen38-flash-offload/scripts/pack_qwen38_store.py pack \
  --model /media/deepak/E/qwen/model --out-dir /media/deepak/E/qwen/store --threads 6
.venv/bin/python qwen38-flash-offload/scripts/pack_qwen38_store.py verify \
  --model /media/deepak/E/qwen/model --out-dir /media/deepak/E/qwen/store   # VERIFY OK
.venv/bin/python qwen38-flash-offload/scripts/pack_qwen38_store.py cmp --sample 300 \
  --model /media/deepak/E/qwen/model --out-dir /media/deepak/E/qwen/store   # CMP OK
```

**4. Build the extensions for this silicon (sm_86 + Zen2).**
```bash
TORCH_CUDA_ARCH_LIST=8.6 QWEN38_CPUTUNE=znver2 scripts/build_extensions.sh
```

**5. Derive the budgets (writes `hw.env`).**
```bash
scripts/hw_preflight.py --model=/media/deepak/E/qwen/model \
  --store=/media/deepak/E/qwen/store/qwen38_flash_exl3_3.05bpw_experts.bin \
  --store-dir=/media/deepak/E/qwen/store --measure
# PASS: VRAM budget 10.70 GiB -> 2835 slots; RAM tier 20 GiB -> 11650 slots; residency 58%; lane cpus 2-5
```

**6. Serve with the best params for this host.**
```bash
scripts/run_local.sh --scope            # preferred: systemd cap with MemorySwapMax=0, budget from MemAvailable
```

The same thing spelled out, if you want to run the server directly:

```bash
MODEL=/media/deepak/E/qwen/model
STORE=/media/deepak/E/qwen/store/qwen38_flash_exl3_3.05bpw_experts.bin
export QWEN_VRAM_LIMIT_BYTES=11492737024 QWEN_RAM_LIMIT_BYTES=25769803776 QWEN38_NV_RAM_GB=20
export QWEN38_NV_THREADS=12 QWEN38_NV_PIECE_KB=256 QWEN38_NV_PREFETCH=1 QWEN38_NV_PF_RING=256 QWEN38_NV_PF_QD=16
export QWEN38_NV_VRING=16 QWEN38_NV_EXCL=1 QWEN38_NV_MAXCPU=8 QWEN38_NV_NOADMIT=1
export QWEN38_NV_CPU=1 QWEN38_NV_CPU_THREADS=4 QWEN38_NV_CPU_CPUS=2-5 QWEN38_NV_READER_CPUS=6-11
export QWEN38_MAIN_CPUS=0 QWEN38_NV_CTL_CPU=1 QWEN38_RCS_GB=0.25
export QWEN38_NV_POL=0.17,0.009,0.145,0.145,0.183,0.044,0.153,0.12,0.745,0.865
export QWEN38_LA=1 QWEN38_LA_BTTRIM=1 QWEN38_NV_PUBFAST=1 QWEN38_NV_PFSIDE=1 QWEN38_NV_CPU_KERN=1
export TORCH_CUDA_ARCH_LIST=8.6 QWEN38_CPUTUNE=znver2 MAX_JOBS=4
python qwen_server.py --model "$MODEL" --store "$STORE" \
  --cache 32768 --cache-quant 4 --chunk 4096 --port 30000
```

Expect `QWEN_SERVER_LOADED ... ready in 60-120 s`, decode ~5-10 tok/s at concurrency 1, prefill ~150-400 tok/s.

**7. One-shot sanity check.**
```bash
python infer.py --model "$MODEL" --store "$STORE" --cache 32768 --cache-quant 4 --chunk 4096 \
  --max-tokens 256 --prompt "Write an implementation of merge sort in C."
```

**8. Benchmark.** Context is a restart knob, so sweep it with the harness (it restarts the server per KV size):
```bash
scripts/bench_local.sh                                        # 16k 32k 65k 131k
scripts/bench_local.sh --contexts "32768 65536 262144"       # pay for context
scripts/bench_local.sh --conc "1 2 4" --reps 3 --config-label "ram20 prefetch-on"
# results: bench_out/sweep_<ctx>.json, memory_<ctx>.json, stats_<ctx>.json, server_<ctx>.log
```

While a server is up, read the engine directly:
```bash
curl -s localhost:30000/stats | python -m json.tool        # hit rates, tier mix, reads/sec, write-backs
curl -s localhost:30000/memory | python -m json.tool       # ram/vram current + peak vs limit
curl -s localhost:30000/nv_trace | python -m json.tool     # NVMe batch latency p50/p99
curl -s localhost:30000/server_info | python -m json.tool  # context, chunk, KV dtype, every QWEN38_* value
```

**9. Tune one lever at a time, in the same session** (arms measured hours apart drift):
```bash
curl -s "localhost:30000/la?on=0"    # decode lookahead off   # back on: /la?on=1
QWEN38_NV_RAM_GB=12 scripts/run_local.sh          # smaller host tier
QWEN38_EC_MAX_SLOTS=2000 scripts/run_local.sh     # cap VRAM slots, leave headroom for the desktop
QWEN38_NV_PIECE_KB=128 scripts/run_local.sh       # read piece size (record is 1.86 MB)
QWEN38_NV_THREADS=8 scripts/run_local.sh          # reader pool on one NVMe
QWEN38_NV_CPU=0 scripts/run_local.sh              # AVX2 lane off (bit-exact decode)
QWEN38_NV_CPU_THREADS=6 QWEN38_NV_CPU_CPUS=2-7 scripts/run_local.sh
python benchmark.py --model "$MODEL" --store "$STORE" --cache 32768 --ram-gb 20 \
  --prefill 8192 32768 --prefetch-compare --cpu-compare --out bench_out/levers.json
```

**10. Check quality after any lane change** (prefill is bit-exact; decode with the CPU lane is not):
```bash
python qwen38-flash-offload/bench/decode_kl_nv.py            # top-1 agreement, mean KL (reference: 0.986 / 0.0047)
python qwen38-flash-offload/bench/stream_rate.py --url http://127.0.0.1:30000
```

Full lever table and measurement rules: [docs/benchmarking.md](docs/benchmarking.md).

## Running a different quant

The pipeline reads the checkpoint itself: record size, slot size, expert count and the resident weight budget come
from the safetensors headers, so `pack`, `verify`, `cmp`, `hw_preflight` and the tier engine work for any branch of
`turboderp/Qwen3.8-Flash-Next-exl3`. **One thing does not:** the hand-written AVX2 lane kernels
(`kernels/cpu_avx2/ft_core.h`, `kernels/nv2/ft_n135.h`) decode a **3-bit MUL1 trellis** (`BITS = 3`). A quant whose
expert trellis is not 3-bit runs GPU-only, and `hw_preflight` says so in a WARN.

| branch | expert trellis | CPU lane | checkpoint | store | VRAM slots | 20 GiB tier | experts resident |
|---|---|---|---|---|---|---|---|
| `3.05bpw_h5_ng5` | 3-bit | works | 85.2 GB | 43.5 GiB | 2,835 | 11,650 | 58% |
| `3.18bpw_h6_ng8` | 3-bit | works | 106.5 GB | ~50.7 GiB | ~2,280 | ~9,986 | ~49% |
| `2.68bpw_h6_ng8` | 3-bit | works | 98.4 GB | 43.5 GiB | 2,835 | 11,650 | 58% |
| `2.05bpw_h4_ng4` | 2-bit | **off** | 62.8 GB | 29.2 GiB | 4,403 | 17,476 | **87%** |
| `1.66bpw_h4_ng3` | 2-bit | **off** | 50.2 GB | 29.2 GiB | ~4,670 | ~17,476 | ~88% |

Checkpoint totals include the n-gram embedding table, which is the part that grows with the branch's `ng` setting
(51.8 GB on `ng8`, 32.6 GB on `ng5`, 26.2 GB on `ng4`, 19.8 GB on `ng3`). Slot and tier counts are approximations;
`scripts/hw_preflight.py` computes the real ones from the checkpoint headers and the memory actually available.

Measured geometry for `2.05bpw_h4_ng4` on this machine (strata stopped, desktop 0.7 GiB, tier 20 GiB, 32k context):
record 1,248,256 B, slot 1,228,800 B, VRAM budget 10.75 GiB, resident weights 4.09 GiB, KV 0.19 GiB, cache arena
5.04 GiB = 4,403 slots, tier 17,476 slots, **21,879 of 25,088 experts resident (87%)**, lane off, POL
`0.17,0.006,0.097,0.145,0.122,0.03,0.102,0.12,0.5,0.62`.

The POL strings above are derived from PCIe 12.8 GB/s and an assumed 2.5 GB/s NVMe. `hw_preflight.py --measure`
replaces them with measured values (it probes O_DIRECT read throughput, pinned host->device DMA, and aggregate DRAM
read at the lane's thread count) — run it with the GPU and disk idle, since a busy `strata-host` drags the NVMe
number down to ~0.2 GB/s.

```bash
# 1. make room: /media/deepak/E has 11 GiB free; a 2.05 checkpoint + store needs ~94 GB
rm -rf /media/deepak/E/qwen/model          # 80 GB; keep the store if you want to keep serving 3.05

# 2. download + pack + verify the branch
scripts/download_model.sh /media/deepak/E/qwen-205 --branch=2.05bpw_h4_ng4
# store lands at /media/deepak/E/qwen-205/store/qwen38_flash_exl3_2.05bpw_experts.bin

# 3. derive budgets for this quant (writes hw.env; expect the "2-bit trellis" WARN and lane OFF)
scripts/hw_preflight.py --model=/media/deepak/E/qwen-205/model \
  --store=/media/deepak/E/qwen-205/store/qwen38_flash_exl3_2.05bpw_experts.bin \
  --store-dir=/media/deepak/E/qwen-205/store --write=/media/deepak/E/qwen-205/hw.env --measure

# 4. serve (run_local.sh reads MODEL/STORE from the environment)
MODEL=/media/deepak/E/qwen-205/model \
STORE=/media/deepak/E/qwen-205/store/qwen38_flash_exl3_2.05bpw_experts.bin \
scripts/run_local.sh --scope

# 5. benchmark the same way
scripts/bench_local.sh --contexts "32768 65536" --config-label "2.05bpw lane-off"
```

What changes in practice: fewer NVMe reads and less PCIe traffic per miss (1.17 MiB slots instead of 1.76 MiB, 87%
resident instead of 58%), but the CPU lane is gone, so cold picks go over PCIe instead of being computed on six cores.
On this box the lane bought ~85% of decode speed on the measured host and the NVMe is the bottleneck here, so 2.05
should be the faster configuration; it is also a much more aggressive quant, and the repo's reference panel
(`reference/qwen3.8-flash-next-exl3-ref-panel.json`) was scored against 3.05, so re-check quality with
`bench/decode_kl_nv.py` and your own prompts before you trust it.

---

## Serving

Start the OpenAI-compatible server on port `30000`:

```bash
python qwen_server.py \
  --model ./model \
  --store ./store/qwen.bin \
  --ram-gb 8 \
  --cache 262144 \
  --chunk 24576 \
  --port 30000
```

### Agent Configuration
Point tools like Pi, OpenCode, or Claude Code to:
* **Base URL:** `http://127.0.0.1:30000/v1`
* **Model:** `qwen3.8-flash-next`
* **Context Window:** `262144`

Supports streaming, reasoning effort control (`off`, `low`, `medium`, `xhigh`), and OpenAI function calling.

### CLI Test
To run a one-shot generation from the terminal:

```bash
python infer.py \
  --model ./model \
  --store ./store/qwen.bin \
  --ram-gb 4 \
  --prompt "Write an implementation of merge sort in C."
```

### Benchmarking
`--cache`, `--cache-quant`, `--chunk` and the `QWEN38_*` env values are the knobs. Sweep context lengths with
`scripts/bench_local.sh` (restarts the server per KV size and runs `bench/sweep.py`), and read `/stats`, `/memory`,
`/nv_trace`, `/server_info` and `GET /la?on=0|1` while it runs. The full lever table and the measurement rules are in
[docs/benchmarking.md](docs/benchmarking.md); the per-machine budgets are derived in
[docs/tuning-rtx3060.md](docs/tuning-rtx3060.md).

---

## To Do

If I get some free time—or someone wants to contribute or put an agent on it:

- **Lower VRAM:** Adapt and benchmark on 12 GB, and potentially 8 GB GPUs. I expect adapting the existing cache and staging budgets to be relatively straightforward; neither configuration has been tested yet.
- **Higher VRAM:** Benchmark 24 GB and 32 GB GPUs to see how prefill and decode scale as more experts fit in VRAM.

---

## Lineage

* [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp (quantization format & inference runtime).
* [glm53-flash-offload](https://github.com/sybil-solutions/glm53-flash-offload) by sybil-solutions (NVMe staging & memory tiering architecture).
* [Qwen3.8-Flash-Next](https://github.com/QwenLM) by Alibaba Cloud Qwen team.
