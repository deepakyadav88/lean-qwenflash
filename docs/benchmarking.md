# Benchmarking and playing with the levers

Everything the engine exposes, and how to measure it on this machine. The measured numbers in
`qwen38-flash-offload/docs/` came from `bench/sweep.py`; use the same protocol so your numbers are comparable.

## Rules that make a measurement mean something

* **Same session.** Two arms run hours apart drift (the campaign's "prefetch off +27-35%" was drift; the real
  number was +13%). Flip the lever and re-measure inside one server session, or restart and run both arms back to
  back.
* **Warm up first.** One 512-token request, not recorded. `bench/sweep.py` does this.
* **Prefill must be fresh tokens.** `sweep.py` sends unique random-token prompts so the prefix cache never fires;
  reusing a prompt measures the cache, not prefill.
* **Decode runs to natural EOS**, uncapped, and aggregate = all completion tokens / (last token − first token).
  Per-stream = (n−1)/(t_last − t_first).
* **Take the median of ≥2 reps**, and record the min/max.
* **Record the config with the result.** `/server_info` returns the context, chunk, KV dtype and every `QWEN38_*`
  env value the server started with — put it in the same JSON as the numbers.

## Context length sweep

`--cache` is a restart knob: 4-bit KV costs 6,144 B/token (12 full-attention layers, 2 KV heads, head_dim 256), so
each 32k of context is ~0.19 GiB of VRAM, which is ~110 expert slots you no longer have.

```bash
scripts/bench_local.sh                                   # 16k 32k 65k 131k
scripts/bench_local.sh --contexts "32768 65536 262144"   # pay for context
scripts/bench_local.sh --conc "1 2 4" --reps 3 --dec-reps 2
scripts/bench_local.sh --config-label "ram20 prefetch-on"
```

It refuses to run if VRAM is occupied or memlock is capped, starts the server per context, waits for `/health`,
runs `bench/sweep.py --template qwen`, then dumps `/memory` and `/stats` before killing it. Output lands in
`bench_out/` (`sweep_<ctx>.json`, `memory_<ctx>.json`, `stats_<ctx>.json`, `server_<ctx>.log`) and prints one
comparison table at the end.

What to read off the table: prefill tok/s at 8k/32k/64k, decode aggregate at C1/C2, `C1@32k`, and the RAM/VRAM
peaks. Context should buy you prefill and cost you decode — the crossover is the context you want.

## The levers, and what measures each one

| lever | default here | what it changes | measure with |
|---|---|---|---|
| `--cache` | 32768 | KV tokens → VRAM expert slots | `scripts/bench_local.sh` context sweep |
| `--cache-quant` | 4 | KV bytes per token (2 = half, 8 = double) | context sweep at the same `--cache` |
| `--chunk` | 4096 | prefill chunk; bigger = faster prefill, higher activation peak | `benchmark.py --prefill 8192 32768` at two chunks |
| `QWEN38_NV_RAM_GB` | 20 | host expert tier size | decode C1 vs tier 12/20/24 |
| `QWEN38_EC_MAX_SLOTS` | unset | explicit VRAM slot cap (leave headroom for the desktop) | `/stats` hit rates + decode C1 |
| `QWEN38_EC_RESERVE_GB` | 1.2 | VRAM left free after the cache takes the rest | long-prefill OOM vs decode speed |
| `QWEN38_EC_ELASTIC_GB` | 3 | cache lent back for prefill activations, re-taken for decode | prefill tok/s vs decode tok/s |
| `QWEN38_EC_STAGE_GB` | 0.7 | prefill staging buffers (2, copy overlapped with compute) | `benchmark.py --stage-min-compare 256 512 1024` |
| `QWEN38_EC_STAGE_MIN` | 512 | pick count above which a forward is treated as prefill | same flag |
| `QWEN38_EC_COPY_GRID` / `QWEN38_NV_COPY_GRID` | 16 / 24 | blocks in the admission copy kernel | `benchmark.py --copy-grid-compare 8 16 24 32` |
| `QWEN38_NV_PREFETCH` | 1 | layer-ahead read of predicted misses | `benchmark.py --prefetch-compare` |
| `QWEN38_NV_PFSIDE` | 1 | the prefetch guess runs on a side stream | `--prefetch-compare` with it on/off |
| `QWEN38_NV_THREADS` | 12 | NVMe reader pool | `/stats` reads/sec + decode C1 |
| `QWEN38_NV_PIECE_KB` | 256 | read piece size (record is 1.86 MB) | decode C1 at 64/128/256/512 |
| `QWEN38_NV_PF_RING` / `QWEN38_NV_PF_QD` | 256 / 16 | prefill FIFO ring slots and queue depth | prefill tok/s at 64k |
| `QWEN38_NV_VRING` | 16 | VRAM victim ring drained by the DMA copy engine | `/nv_trace` write-back drops |
| `QWEN38_NV_CPU` | 1 | AVX2 lane for cold RAM picks | `benchmark.py --cpu-compare` (lane off/on, paired) |
| `QWEN38_NV_CPU_THREADS` / `QWEN38_NV_CPU_CPUS` | 4 / 2-5 | lane threads and cores | decode C1 at 2/4/6 threads |
| `QWEN38_NV_MAXCPU` | 8 | experts per CPU job | decode C1 |
| `QWEN38_NV_NOADMIT` | 1 | cold picks read zero-copy from RAM instead of entering VRAM | decode C1, `/stats` lane mix |
| `QWEN38_NV_POL` | see `hw.env` | lane planner cost model (ms): g0,thit,tzc,ca,cb,ctok,push,nvlat,nvdeep,nvone | re-measure with `scripts/hw_preflight.py --measure` |
| `QWEN38_LA` | 1 | decode lookahead — **runtime toggle**: `GET /la?on=0|1` | A/B in one session, no restart |
| `QWEN38_LA_BTTRIM` | 1 | block-table width = needed pages | with LA |
| `QWEN38_NV_PUBFAST` | 1 | `nv_pub` warp-ballot compaction | decode C1 |
| `QWEN38_NV_CPU_KERN` | 1 | N135 CPU-lane forward (~+6% C1, not bit-exact) | decode C1 + `decode_kl_nv.py` |
| `QWEN38_MAX_RQ_TOKENS` | unset here | page-allocation round so concurrent jobs decode together | C2/C4 aggregate |
| `QWEN38_TRITON_PIN` | unset | fixed KDA autotune picks (unset = autotune, which changes rounding) | `bench/score_ref_panel.py` |
| `QWEN38_RCS_GB` | 0.25 | host recurrent-state cache | multi-turn TTFT at long context |
| `QWEN_PREBUILT_EXT` | unset | reuse `build/*.so` instead of compiling | startup time only |

## Runtime endpoints (no restart needed)

| endpoint | gives you |
|---|---|
| `/stats` | cache hit rates, lane mix (VRAM/RAM/NVMe/CPU), reads, write-backs, last request prefill metrics |
| `/memory` | `ram_current`, `ram_sampled_peak`, `ram_limit`, `total_vram_current`, `total_vram_peak`, `vram_limit` |
| `/nv_trace` | per-tier latency histograms (NVMe batch p50/p99) |
| `/nv_verify` | byte-compares sampled VRAM and RAM slots against fresh store reads |
| `/la?on=0|1` | decode lookahead on/off — cheapest A/B you can run |
| `/server_info` | context, chunk, KV dtype, and every `QWEN38_*` env the server started with |

## Quality, whenever you turn the CPU lane on

Prefill is bit-exact; decode with the CPU lane is not. Check it after you change lane settings:

```bash
python qwen38-flash-offload/bench/decode_kl_nv.py   # paired CPU-lane off/on: top-1 agreement and mean KL
python qwen38-flash-offload/bench/stream_rate.py --url http://127.0.0.1:30000   # PASS = even delivery, no end burst
```

Reference from the measured host: mean KL 0.0047, top-1 agreement 0.986. If your KL goes much above that, a lane
setting is wrong, not just slow.

## Things that will skew numbers on this box

* `strata-host` (the local LLM for this session) takes ~10.6 GiB VRAM and ~27 GiB RAM. Benchmarks run next to it
  measure a starved engine.
* Swap is enabled: a swapped-out pinned expert shows up as a huge NVMe wait in `/nv_trace`. Use
  `scripts/run_local.sh --scope` for a cap with `MemorySwapMax=0`.
* memlock is 4 GiB; without `ulimit -l unlimited` the RAM tier allocation fails or degrades.
* The GPU idles between runs; the first request after a long pause is slower. Warm up, then measure.
* One NVMe drive, PCIe gen3 x16 — expect NVMe waits to dominate decode, so `QWEN38_NV_PREFETCH` and
  `QWEN38_NV_PIECE_KB` matter more here than they did on the RAID0 host.
