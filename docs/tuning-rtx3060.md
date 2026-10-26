# Tuning for a consumer PC — RTX 3060 12 GB, 32 GB RAM, Ryzen 5 3600, one NVMe

The upstream configuration was measured on a server: RTX 3090 24 GB, PCIe 4.0 x16 (~25 GB/s host→device),
4-drive NVMe RAID0 (~26 GB/s), 24-core EPYC on 8-channel DDR4 (~136 GB/s for the CPU lane), 55 GiB RAM cap.
Every number below is recomputed for this machine by `hw_profile.py`; nothing is copied from that box.

## What makes the engine work at all

Qwen3.8 Flash Next is 48 MoE layers x 512 routed experts (+1 MTP layer) = 25,088 experts of 1.76 MiB packed.
A token activates 10 experts per layer = 480 picks, ~300 unique. Only ~15% of them can be resident here, so the
engine's job is to answer, per pick, *where the bytes are* and *who computes them*:

| tier | holds | how it is filled |
|---|---|---|
| VRAM expert cache | the hottest experts, as fixed 1.76 MiB slots in one arena | CLOCK admission; a kernel dedups picks, marks hits, picks victims and repoints exllamav3's per-expert pointer tables |
| pinned host RAM | experts **not** in VRAM (exclusive tier, so nothing is duplicated) | `cudaHostRegister`-ed arena; the GPU reads any slot directly over PCIe (zero-copy) |
| NVMe store | every expert, one 4 KiB-aligned record each | `O_DIRECT` reads split into pieces over a reader pool, staged ahead of the compute stream |
| AVX2 CPU lane | the coldest RAM-resident picks during decode | computed in place on host cores; the GPU masks them out and adds the partial |

The layer-ahead prefetch runs layer *l+1*'s router while layer *l* computes and reads predicted NVMe-only experts
early. A victim ring parks evicted VRAM bytes device-to-device, then a DMA copy engine writes them to RAM off the
critical path. Prefill never uses the CPU lane and stays bit-exact; decode with the CPU lane is not bit-exact
(KL ~0.005 upstream).

## Derived budgets on this machine

`python hw_profile.py --model=/media/deepak/E/qwen/model --context=32768 --chunk=4096 --desktop-gb=0.75 --ram-gb=20`

| quantity | value | note |
|---|---|---|
| GPU | RTX 3060, sm_86, PCIe **gen3 x16** (~12.8 GB/s encode) | upstream was gen4 x16 at ~25 GB/s |
| VRAM budget | 10.70 GiB (12 GiB − 0.75 GiB desktop − 0.55 GiB headroom) | `QWEN_VRAM_LIMIT_BYTES` |
| resident weights | 4.21 GiB | dense attention, KDA state, routers, shared experts, lm_head, expert `suh/svh`; the 32 GB n-gram table streams from disk (`-embd`) and the vision tower is not loaded |
| KV cache | 0.19 GiB @ 32k, 4-bit | 6,144 B/token (12 full-attention layers, 2 KV heads, head_dim 256); 262k context would cost 1.6 GiB |
| VRAM expert cache | 4.87 GiB = **2,835 slots** of 1.76 MiB | `QWEN38_EC_RESERVE_GB=1.2`, `QWEN38_EC_STAGE_GB=0.7`, `QWEN38_EC_ELASTIC_GB=3` |
| host RAM tier | 20.0 GiB = **11,650 slots** | `QWEN38_NV_RAM_GB`, plus 0.25 GiB recurrent cache, 0.44 GiB prefill ring, 1.5 GiB margin, 3 GiB OS headroom |
| resident experts | 14,485 of 25,088 (**58%**) | upstream held ~6,100 of 12,096 (50%) on a 24 GB card |
| CPU lane | 4 threads on cores 2-5, readers on the SMT siblings 6-11, controller on cpu1, main on cpu0 | upstream: 22 lane threads, 14 readers |
| ISA tune | `-mtune=znver2` (family 23 = Zen2) | the shipped build used `-mtune=znver3` |
| CUDA arch | `TORCH_CUDA_ARCH_LIST=8.6` | the shipped default was `8.9`, which is wrong for this card |
| store | 43.5 GiB on the NVMe at `/media/deepak/E/qwen/store` | `O_DIRECT` verified on ext4 |

Lane cost model (`QWEN38_NV_POL`) rescaled by slot size, PCIe width, NVMe throughput and DRAM bandwidth:
`0.17,0.004,0.074,0.145,0.183,0.044,0.078,0.12,0.745,0.865` (ms) — a 1.76 MiB slot is 5x cheaper to move than the
9.44 MiB slot the server tuned against, but a single NVMe drive is ~10x slower than the measured RAID0.

## Levers, in order of how much they matter here

1. **`QWEN38_NV_RAM_GB`** — the RAM tier is the only tier that can be grown on this box. 20 GiB of pinned experts
   is what a 32 GB machine can give while the desktop stays usable.
2. **`--context`** — every 32k of context costs ~0.19 GiB of VRAM, which is ~110 expert slots. 32k is the default;
   65k or 262k is available but trades cache hits for context.
3. **`QWEN38_NV_PREFETCH=1`** — worth ~13% decode on the measured host, and it matters *more* here because NVMe
   misses cost ~0.75 ms instead of ~0.4 ms.
4. **`QWEN38_NV_PIECE_KB=256`** — records are 1.76 MiB, so 256 KiB pieces keep the queue deep on one drive
   instead of the server's 2,304 KiB pieces.
5. **CPU lane threads (`QWEN38_NV_CPU_THREADS=4`) and `QWEN38_NV_MAXCPU=8`** — 6 cores on dual-channel DDR4 is a
   different machine from 22 threads on 8-channel DDR4. Oversubscribing the lane makes the GPU wait for it.
6. **`QWEN38_NV_NOADMIT=1`** — cold picks are read zero-copy from RAM instead of being admitted into VRAM, which
   keeps the small cache for hot experts.
7. **`QWEN_VRAM_LIMIT_BYTES` + torch's per-process fraction** — the guard now caps at the card, not at 16 GB, and
   the fraction is computed from what the process already reserves.
8. **`QWEN38_CPUTUNE` / `TORCH_CUDA_ARCH_LIST`** — build for sm_86 and znver2, not sm_89/znver3.
9. **Storage placement** — the store must be on the NVMe (`/media/deepak/E`), not the HDDs; `O_DIRECT` on ext4 is
   probed by `scripts/hw_preflight.py`.
10. **Decode scheduling levers** the profile turns on: `QWEN38_LA=1` (launch the next decode step before the host
    reads the sampled token — bit-exact), `QWEN38_LA_BTTRIM=1`, `QWEN38_NV_PUBFAST=1`, `QWEN38_NV_PFSIDE=1`
    (prefetch guess on a side stream), `QWEN38_NV_CPU_KERN=1` (the N135 CPU-lane forward, ~+6% at C1, not
    bit-exact). Any of them can be set to 0 in `hw.env`.

## Honest expectations

This box is roughly a quarter of the measured server on PCIe, ~10x on storage, and ~4x on CPU lane bandwidth.
Expect decode in the **5-10 tok/s** range and prefill in the **150-400 tok/s** range at 32k context, rather than
the 20 tok/s / 1,600 tok/s the README reports on a 4090. The engine will run; the tiering is what makes it fit,
and the bandwidth limits what it achieves. Run `bench/sweep.py` after the first start and re-measure
`QWEN_NVME_GBPS` / `QWEN_PCIE_GBPS` with `scripts/hw_preflight.py --measure` to tighten the planner constants.

## Setup on this machine

```bash
scripts/download_model.sh /media/deepak/E/qwen          # checkpoint (~85 GB) + packed store (~43.5 GiB)
scripts/build_extensions.sh                             # sm_86 CUDA, -mtune=znver2
scripts/hw_preflight.py --model=/media/deepak/E/qwen/model \
  --store=/media/deepak/E/qwen/store/qwen38_flash_exl3_3.05bpw_experts.bin \
  --store-dir=/media/deepak/E/qwen/store --measure --write hw.env
scripts/run_local.sh                                    # OpenAI-compatible server on :30000
```

## Host conditions that will bite you

* **VRAM is contested.** The desktop session holds ~0.7 GiB; anything else serving a model on this GPU (a local
  LLM, a diffusion process) leaves the expert cache no room. Stop it before starting the server.
* **RAM is contested.** The pinned tier is allocated up front; `MemAvailable` must cover tier + recurrent cache +
  headroom. `memory_guard.py` exits 87 if the budget is exceeded (proc mode measures `total - MemAvailable`).
* **memlock** is 4 GiB here; `ulimit -l unlimited` before starting (the tier is `cudaHostRegister`-ed).
* **swap is enabled** (4 GiB). Swapped-out pinned experts stall the GPU lane; run with `--scope` (a systemd cap
  with `MemorySwapMax=0`) if you want the same guarantee the container gave.
* **PCIe is gen3 x16** on this board, so admissions cost ~0.08 ms per slot rather than ~0.04 ms.
