"""Hardware profile for the Qwen3.8 Flash Next offload engine.

The upstream configuration was measured on a server: RTX 3090 24 GB, PCIe 4.0 x16 (~25 GB/s host->device),
4-drive NVMe RAID0 (~26 GB/s), 24-core EPYC on 8-channel DDR4 (~136 GB/s for the CPU lane), 55 GiB RAM cap.
Every budget in this file is derived from the machine it runs on instead of copied from that box:

  VRAM      : NVML total minus what the desktop already holds, minus safety headroom -> expert cache slots
  host RAM  : MemAvailable (or the cgroup cap) minus recurrent cache, prefill ring, margins -> RAM expert tier
  CPU       : physical cores for the AVX2 lane, SMT siblings for the NVMe readers, one CPU for the controller
  ISA       : -mtune=znverN / -march=znverN for the CPU-lane build, TORCH_CUDA_ARCH_LIST for the CUDA build
  lane plan : nv2's cost model (QWEN38_NV_POL) rescaled by slot bytes, PCIe width, NVMe throughput and DRAM bandwidth

Nothing here loads a model. `python hw_profile.py` prints the derived numbers; `--write hw.env` emits shell
assignments for scripts/run_local.sh; scripts/hw_preflight.py measures the storage and the PCIe link first.
"""
import ctypes as C
import json
import os
import re
import struct
import sys
from pathlib import Path

GiB = 1024 ** 3
MB = 1024 ** 2

# Measured host constants from qwen38-flash-offload/docs/reference.md (G067 / N119 S3b2).
REF = {
    "slot_bytes": 9_437_184,          # one VRAM expert slot on the 42x288 checkpoint the docs measured
    "pcie_gbps": 25.0,                # DMA host->device on PCIe 4.0 x16
    "nvme_gbps": 26.0,                # 4-drive RAID0, O_DIRECT
    "nvme_latency_ms": 1.2,           # mean record read latency under load
    "cpu_gbps": 136.0,                # CPU lane aggregate DRAM read at 22 threads
    "cpu_threads": 22,
    "pol": {"g0": 0.17, "thit": 0.023, "tzc": 0.38, "ca": 0.145, "cb": 0.165,
            "ctok": 0.040, "push": 0.40, "nvlat": 0.12, "nvdeep": 0.395, "nvone": 0.75},
}


class Nvml:
    """ctypes NVML, so the profile needs no extra dependency."""

    class Info(C.Structure):
        _fields_ = [("total", C.c_ulonglong), ("free", C.c_ulonglong), ("used", C.c_ulonglong)]

    def __init__(self, index=0):
        self.lib = C.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError("NVML init failed")
        self.handle = C.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(index, C.byref(self.handle)) != 0:
            raise RuntimeError("NVML handle failed")

    def memory(self):
        info = self.Info()
        assert self.lib.nvmlDeviceGetMemoryInfo(self.handle, C.byref(info)) == 0
        return info.total, info.used, info.free

    def name(self):
        buf = C.create_string_buffer(128)
        assert self.lib.nvmlDeviceGetName(self.handle, buf, 128) == 0
        return buf.value.decode()

    def capability(self):
        maj, min_ = C.c_uint(), C.c_uint()
        if self.lib.nvmlDeviceGetCudaComputeCapability(self.handle, C.byref(maj), C.byref(min_)) != 0:
            return None
        return (maj.value, min_.value)


def pci_link():
    """PCIe gen x width of the GPU slot, from sysfs (fall back to Gen3 x16, the safe assumption)."""
    try:
        for d in Path("/sys/bus/pci/devices").iterdir():
            cls = (d / "class").read_text().strip()
            if not cls.startswith("0x03"):
                continue
            speed = (d / "current_link_speed").read_text().strip()
            width = int(re.search(r"x(\d+)", (d / "current_link_width").read_text()).group(1))
            gbps = float(re.search(r"([\d.]+) GT/s", speed).group(1))
            gen = 4 if gbps > 8 else 3 if gbps > 4 else 2
            # encode: 16 GT/s per lane for Gen4, 8 for Gen3; usable DMA is ~80% of encode
            return {"gen": gen, "width": width, "gbps": gbps * width * 0.8}
    except Exception:
        pass
    return {"gen": 3, "width": 16, "gbps": REF["pcie_gbps"]}


def cpu_layout():
    """Logical CPUs, physical core groups, ISA flags and the matching -mtune target."""
    affinity = sorted(os.sched_getaffinity(0))
    cores = {}
    for c in affinity:
        try:
            sib = (Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")).read_text().strip()
            rep = int(sib.split(",")[0].split("-")[0])
        except Exception:
            rep = c
        cores.setdefault(rep, []).append(c)
    flags = set()
    try:
        line = [l for l in open("/proc/cpuinfo") if "flags" in l][0]
        flags = {f for f in line.split(":", 1)[1].split()}
    except Exception:
        pass
    vendor, family, model = "", 0, 0
    try:
        txt = open("/proc/cpuinfo").read()
        vendor = re.search(r"vendor_id\s*:\s*(\S+)", txt).group(1)
        family = int(re.search(r"cpu family\s*:\s*(\d+)", txt).group(1))
        model = int(re.search(r"model\s*:\s*(\d+)", txt).group(1))
    except Exception:
        pass
    tune = "-mtune=generic"
    if vendor == "AuthenticAMD":
        znver = 1 if family == 21 else 2 if family == 23 else 3 if family in (25, 26) else 4 if family == 27 else 5
        tune = f"-mtune=znver{znver}"
    elif vendor == "GenuineIntel":
        tune = "-mtune=haswell" if family >= 6 else "-mtune=generic"
    lane_ok = {"avx2", "fma", "f16c"} <= flags
    # Dual-channel DDR4 on a 6-core Zen2 desktop streams ~35 GB/s for the CPU lane; scale with channel count
    chans = 2
    return {"affinity": affinity, "cores": cores, "ncores": len(cores), "flags": flags,
            "lane_ok": lane_ok, "tune": tune, "vendor": vendor, "family": family, "model": model,
            "dram_gbps": min(REF["cpu_gbps"], 12.0 * chans * (1 if family < 25 else 1.4))}


def host_memory():
    total = int(os.sysconf("SC_PAGE_SIZE")) * os.sysconf("SC_PHYS_PAGES")
    avail = None
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) * 1024
    cg = os.environ.get("QWEN_GLOBAL_MEMORY_CGROUP")
    cap = None
    if cg:
        p = Path(cg) / "memory.max"
        if p.exists():
            v = p.read_text().strip()
            if v != "max":
                cap = int(v)
    return {"total": total, "available": avail or total, "cap": cap, "cgroup": cg}


def checkpoint_geometry(model_dir):
    """Bytes per expert slot / record and the resident (non-streamed) weight budget, read from the checkpoint."""
    model_dir = Path(model_dir)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(idx.values()))
    sizes = {}
    gate_shape = down_shape = None
    for fn in shards:
        p = model_dir / fn
        if not p.exists():                      # a resumed download leaves .http-part files behind
            p = model_dir / (fn + ".http-part")
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            sizes[k] = v["data_offsets"][1] - v["data_offsets"][0]
            if gate_shape is None and k.endswith(".gate_proj.trellis"):
                gate_shape = v["shape"]
            if down_shape is None and k.endswith(".down_proj.trellis"):
                down_shape = v["shape"]
    trellis = sum(v for k, v in sizes.items() if ".mlp.experts." in k and k.endswith(".trellis"))
    side = sum(v for k, v in sizes.items() if ".mlp.experts." in k and
               (k.endswith(".suh") or k.endswith(".svh") or k.endswith(".mul1")))
    vision = sum(v for k, v in sizes.items() if k.startswith("model.visual"))
    mtp = sum(v for k, v in sizes.items() if k.startswith("mtp."))
    other = sum(sizes.values()) - trellis - side - vision - mtp
    n_trellis = sum(1 for k in sizes if ".mlp.experts." in k and k.endswith(".trellis"))
    n_exp = n_trellis // 3                      # every expert carries gate/up/down trellises
    trunk = {re.match(r"model\.language_model\.layers\.(\d+)", k).group(1)
             for k in sizes if ".mlp.experts." in k and k.startswith("model.language_model")}
    mtp_l = {re.match(r"mtp\.layers\.(\d+)", k).group(1)
             for k in sizes if ".mlp.experts." in k and k.startswith("mtp.")}
    layers = len(trunk) + len(mtp_l)
    experts_per_layer = n_exp // max(layers, 1)
    experts_trunk = len(trunk) * experts_per_layer
    per_matrix = trellis // max(n_trellis, 1)
    al = lambda x: (x + 255) // 256 * 256
    slot = al(per_matrix) * 3                   # gate | up @ off_u | down @ off_d, each 256-aligned
    record = (slot + side // max(n_exp, 1) + 4095) // 4096 * 4096
    disk = 0
    for f in model_dir.iterdir():
        if f.is_file() and (f.name.endswith(".safetensors") or f.name.endswith(".safetensors.http-part")):
            disk += f.stat().st_size
    cfg = {}
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    qc = cfg.get("quantization_config") or {}
    # The hand-written lane kernels (kernels/cpu_avx2/ft_core.h, kernels/nv2/ft_n135.h) decode a 3-bit MUL1 trellis.
    # The trellis shape says the expert bit width directly: the last dim is the tile in fp16 units,
    # 16 * bits (kernels/cpu_avx2/ft_core.h TILE_BYTES = 16 * BITS * 2). 3.05bpw -> 48 -> 3 bits.
    bits_from_shape = gate_shape[-1] // 16 if gate_shape else 0
    if bits_from_shape not in (1, 2, 3, 4, 5, 6, 8):
        bits_from_shape = int(round(qc.get("bits", 3.05))) if qc else 3
    expert_bits = bits_from_shape
    return {"total_bytes": sum(sizes.values()), "disk_bytes": disk, "trellis_bytes": trellis, "side_bytes": side,
            "vision_bytes": vision, "mtp_bytes": mtp, "dense_bytes": other,
            "experts": n_exp, "experts_trunk": experts_trunk, "experts_per_layer": experts_per_layer,
            "layers": layers, "mtp_layers": len(mtp_l), "slot_bytes": slot, "record_bytes": record,
            "quant_bits": qc.get("bits"), "head_bits": qc.get("head_bits"), "codebook": qc.get("codebook"),
            "expert_bits": expert_bits, "gate_trellis_shape": gate_shape}


def kv_bytes_per_token(model_dir, cache_quant=4):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    t = cfg.get("text_config") or cfg
    L = t.get("num_hidden_layers", 48)
    interval = t.get("full_attention_interval", 4)
    full = (L + interval - 1) // interval
    heads = t.get("num_key_value_heads", 2)
    hd = t.get("head_dim", 256) or t.get("hidden_size", 2560)
    kv = full * heads * hd * 2 * cache_quant / 8
    # sparse-indexer KV (indexer_budget x indexer_head_dim x indexer_kv_heads) is small and constant
    idx = t.get("indexer_budget", 0) * t.get("indexer_head_dim", 0) * t.get("indexer_kv_heads", 1) * 2 * cache_quant / 8
    # recurrent KDA state per sequence: linear layers x value heads x key dim x value dim, fp32
    lin = L - full
    rec = lin * t.get("linear_num_value_heads", 0) * t.get("linear_key_head_dim", 0) * t.get("linear_value_head_dim", 0) * 4
    return {"kv_per_token": int(kv), "indexer_bytes": int(idx), "recurrent_state_bytes": int(rec),
            "full_layers": full, "linear_layers": lin, "max_ctx": t.get("max_position_embeddings", 262144)}


def derive(model_dir, store=None, context=32768, chunk=4096, cache_quant=4, mtp=False,
           nvme_gbps=None, pcie_gbps=None, ram_gb=None, desktop_gb=None, measured=None):
    nvml = Nvml()
    total_vram, used_vram, _ = nvml.memory()
    cap = nvml.capability()
    geo = checkpoint_geometry(model_dir)
    ctx = kv_bytes_per_token(model_dir, cache_quant)
    cpus = cpu_layout()
    host = host_memory()
    link = pci_link()

    # --- VRAM: what the desktop already holds is not ours. CUDA context + allocator fragmentation headroom.
    # desktop_gb overrides the NVML reading (e.g. after stopping another GPU process, leaving only Xorg).
    # Otherwise assume the desktop floor and warn about whatever else is holding the card.
    desktop_floor = int(desktop_gb * GiB) if desktop_gb else min(used_vram, int(1.25 * GiB))
    vram_safety = int(0.55 * GiB)
    vram_limit = int(total_vram - desktop_floor - vram_safety)
    contested = max(0, used_vram - desktop_floor)
    weights = geo["dense_bytes"] + geo["side_bytes"] + (geo["mtp_bytes"] if mtp else 0)   # vision + ngram table are not loaded
    kv = ctx["kv_per_token"] * context + ctx["indexer_bytes"]
    activations = int(chunk * 2560 * 4 * 6)          # ~6 tensors of hidden-size activations per chunk, fp32
    reserve = int(1.2 * GiB)                         # staging buffers, pointer tables, recurrent state on device
    cache_bytes = vram_limit - weights - kv - activations - reserve
    slots = max(0, cache_bytes // geo["slot_bytes"])

    # --- host RAM: the pinned expert tier, sized from what is actually available right now
    rcs = 0.25 * GiB
    pf_ring = 256 * geo["slot_bytes"]
    margin = 1.5 * GiB                               # nv2's QWEN38_NV_MARGIN_GB
    host_reserve = 3.0 * GiB                         # desktop, browser, kernel, page cache
    avail = host["cap"] if host["cap"] else host["available"]
    tier_gb = ram_gb if ram_gb else max(2.0, (avail - rcs - pf_ring - margin - host_reserve) / GiB)

    # --- nv2 lane cost model, rescaled from the measured server
    slot_ratio = geo["slot_bytes"] / REF["slot_bytes"]
    pcie = pcie_gbps or link["gbps"]
    nvme = nvme_gbps or 2.5
    cpu_bw = cpus["dram_gbps"]
    p = dict(REF["pol"])
    p["thit"] = round(REF["pol"]["thit"] * slot_ratio * REF["pcie_gbps"] / pcie, 3)
    p["tzc"] = round(REF["pol"]["tzc"] * slot_ratio * REF["pcie_gbps"] / pcie, 3)
    p["push"] = round(REF["pol"]["push"] * slot_ratio * REF["pcie_gbps"] / pcie, 3)
    p["cb"] = round(REF["pol"]["cb"] * slot_ratio * REF["cpu_gbps"] / cpu_bw, 3)
    p["ctok"] = round(REF["pol"]["ctok"] * slot_ratio * REF["cpu_gbps"] / cpu_bw, 3)
    p["nvdeep"] = round(geo["record_bytes"] / (nvme * 1e9) * 1000, 3)
    p["nvone"] = round(p["nvdeep"] + REF["pol"]["nvlat"], 3)
    p["nvlat"] = round(max(0.10, REF["pol"]["nvlat"] * REF["nvme_latency_ms"] / 1.2), 3)
    if measured:
        p.update({k: v for k, v in measured.items() if k in p})

    # --- CPU pinning: physical cores for the lane, SMT siblings for the readers
    cores = [cpus["cores"][k] for k in sorted(cpus["cores"])]
    affinity = cpus["affinity"]
    # The lane kernels are hardcoded to a 3-bit MUL1 trellis, so a 2-bit or 4-bit quant cannot use them.
    lane_supported = cpus["lane_ok"] and geo["expert_bits"] == 3
    lane_threads = len(cores) - 2 if lane_supported and len(cores) > 3 else (1 if lane_supported else 0)
    lane_cpus = [cores[i][0] for i in range(2, min(len(cores), 2 + lane_threads))] or affinity[:lane_threads]
    lane_threads = len(lane_cpus)
    spare = [c for c in affinity if c not in lane_cpus and c not in (0, 1)]
    reader_cpus = spare or affinity[2:]
    def spec(cs):
        runs, out = [], []
        for c in sorted(set(cs)):
            if runs and c == runs[-1][1] + 1:
                runs[-1][1] = c
            else:
                runs.append([c, c])
        return ",".join(f"{a}-{b}" if b > a else str(a) for a, b in runs)

    env = {
        "TORCH_CUDA_ARCH_LIST": f"{cap[0]}.{cap[1]}" if cap else "8.6",
        "QWEN_VRAM_LIMIT_BYTES": str(vram_limit),
        "QWEN_RAM_LIMIT_BYTES": str(int(tier_gb * GiB + rcs + margin + host_reserve)),
        "QWEN38_NV_RAM_GB": f"{tier_gb:.1f}",
        "QWEN38_NV_MARGIN_GB": f"{margin / GiB:.1f}",
        "QWEN38_RCS_GB": f"{rcs / GiB:.2f}",
        "QWEN38_NV_THREADS": str(min(16, len(reader_cpus) * 2)),
        "QWEN38_NV_READER_CPUS": spec(reader_cpus),
        "QWEN38_NV_CTL_CPU": "1",
        "QWEN38_MAIN_CPUS": "0",
        "QWEN38_NV_CPU_CPUS": spec(lane_cpus),
        "QWEN38_NV_CPU_THREADS": str(lane_threads),
        "QWEN38_NV_MAXCPU": str(max(4, lane_threads * 2)),
        "QWEN38_NV_PIECE_KB": str(max(64, min(256, geo["record_bytes"] // 4096 // 4))),
        "QWEN38_NV_PF_RING": str(256),
        "QWEN38_NV_PF_QD": "16",
        "QWEN38_NV_VRING": "16",
        "QWEN38_NV_PREFETCH": "1",
        "QWEN38_NV_EXCL": "1",
        "QWEN38_NV_CPU": "1" if lane_supported else "0",
        "QWEN38_NV_NOADMIT": "1",
        # HOM-272 / N136 decode levers, measured on the same code path (the CPU kernel needs the CPU lane).
        # Each is a scheduling change; CPU_KERN is not bit-exact (~3e-4 relative per expert).
        "QWEN38_LA": "1" if lane_supported else "0",
        "QWEN38_LA_BTTRIM": "1" if lane_supported else "0",
        "QWEN38_NV_PUBFAST": "1",
        "QWEN38_NV_PFSIDE": "1",
        "QWEN38_NV_CPU_KERN": "1" if lane_supported else "0",
        "QWEN38_NV_POL": ",".join(str(p[k]) for k in
                                  ["g0", "thit", "tzc", "ca", "cb", "ctok", "push", "nvlat", "nvdeep", "nvone"]),
        "QWEN38_EC_RESERVE_GB": f"{reserve / GiB:.1f}",
        "QWEN38_EC_STAGE_GB": "0.7",
        "QWEN38_EC_ELASTIC_GB": "3",
        "QWEN38_EC_COPY_GRID": "16",
        "QWEN38_NV_COPY_GRID": "24",
        "QWEN38_CPUTUNE": cpus["tune"],
        "MAX_JOBS": "4",
    }

    warnings = []
    if contested > 0:
        warnings.append(
            f"{contested / GiB:.2f} GiB of VRAM is held by processes other than the desktop session (a local LLM, a "
            "diffusion run). Stop them before serving, or pass --desktop-gb=<what Xorg alone uses>.")
    if vram_limit <= weights + kv + activations + reserve:
        warnings.append(
            f"VRAM budget {vram_limit / GiB:.2f} GiB is below the {weights / GiB:.2f} GiB of resident weights plus KV and "
            f"activations: reduce --context/--chunk, or free VRAM.")
    if ram_gb is None and host["available"] < tier_gb + rcs + margin + host_reserve:
        warnings.append(
            f"MemAvailable {host['available'] / GiB:.1f} GiB cannot hold a {tier_gb:.1f} GiB pinned expert tier plus "
            f"{rcs / GiB:.1f} GiB recurrent cache and {host_reserve / GiB:.1f} GiB of OS headroom. Free memory or pass --ram-gb.")
    if not cpus["lane_ok"]:
        warnings.append("CPU lacks AVX2+FMA+F16C: the CPU lane is off (QWEN38_NV_CPU=0), decode falls back to the GPU-only exact path.")
    elif geo["expert_bits"] != 3:
        warnings.append(
            f"expert weights are {geo['quant_bits']} bpw ({geo['expert_bits']}-bit trellis, codebook "
            f"{geo['codebook']}): the AVX2 lane kernels are hardcoded 3-bit (kernels/cpu_avx2/ft_core.h BITS=3), so "
            f"QWEN38_NV_CPU / NV_CPU_KERN / LA are off and decode runs GPU-only over PCIe.")

    return {
        "warnings": warnings,
        "gpu": {"name": nvml.name(), "vram_total": total_vram, "vram_in_use_by_others": used_vram,
                "vram_desktop_floor": desktop_floor, "vram_contested": contested,
                "capability": cap, "pci": link, "vram_limit": vram_limit},
        "host": {"ram_total": host["total"], "ram_available": host["available"], "cgroup": host["cgroup"],
                 "cap": host["cap"], "tier_gb": tier_gb, "rcs_gb": rcs / GiB, "pf_ring_gb": pf_ring / GiB,
                 "margin_gb": margin / GiB, "host_reserve_gb": host_reserve / GiB},
        "cpu": {"ncores": cpus["ncores"], "logical": cpus["affinity"], "lane_ok": cpus["lane_ok"],
                "tune": cpus["tune"], "dram_gbps": cpu_bw, "lane_cpus": lane_cpus, "reader_cpus": reader_cpus},
        "model": {"checkpoint_bytes": geo["total_bytes"], "experts": geo["experts"], "layers": geo["layers"],
                  "quant_bits": geo["quant_bits"], "head_bits": geo["head_bits"], "expert_bits": geo["expert_bits"],
                  "slot_bytes": geo["slot_bytes"], "record_bytes": geo["record_bytes"],
                  "store_bytes": geo["record_bytes"] * geo["experts"],
                  "resident_weights_bytes": weights, "kv_per_token": ctx["kv_per_token"],
                  "context": context, "chunk": chunk, "kv_bytes": kv,
                  "recurrent_state_bytes": ctx["recurrent_state_bytes"], "cache_bytes": cache_bytes,
                  "vram_slots": slots,
                  "resident_experts": slots + int(tier_gb * GiB // geo["slot_bytes"]),
                  "resident_fraction": (slots + int(tier_gb * GiB // geo["slot_bytes"])) / max(geo["experts"], 1)},
        "storage": {"nvme_gbps": nvme, "pcie_gbps": pcie, "store": store},
        "pol": p,
        "env": env,
    }


def write_env(profile, path):
    lines = ["# derived by hw_profile.py - edit freely, every value is an override"]
    for k, v in sorted(profile["env"].items()):
        lines.append(f"export {k}={v}")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    args = sys.argv[1:]
    def opt(name, default=None, cast=str):
        for i, a in enumerate(args):
            if a.startswith(name + "="):
                return cast(a[len(name) + 1:])
        return default
    model = opt("--model", "model")
    store = opt("--store", "store/qwen38_flash_exl3_3.05bpw_experts.bin")
    ctx = opt("--context", 32768, int)
    chunk = opt("--chunk", 4096, int)
    ram = opt("--ram-gb", None, float)
    desktop = opt("--desktop-gb", None, float)
    prof = derive(model, store, ctx, chunk, ram_gb=ram, desktop_gb=desktop)
    if "--json" in args:
        print(json.dumps(prof, indent=2))
        return
    m, g, h, c = prof["model"], prof["gpu"], prof["host"], prof["cpu"]
    for w in prof["warnings"]:
        print(f"WARN   {w}")
    print(f"GPU    {g['name']}  {g['vram_total'] / GiB:.1f} GiB, {g['vram_in_use_by_others'] / GiB:.2f} GiB already used "
          f"(sm_{g['capability'][0]}{g['capability'][1] if g['capability'] else ''}, PCIe gen{g['pci']['gen']} x{g['pci']['width']})")
    print(f"VRAM   budget {g['vram_limit'] / GiB:.2f} GiB -> resident weights {m['resident_weights_bytes'] / GiB:.2f} GiB, "
          f"KV {m['kv_bytes'] / GiB:.2f} GiB @ {m['context']} ctx, cache arena {m['cache_bytes'] / GiB:.2f} GiB "
          f"= {m['vram_slots']} slots of {m['slot_bytes'] / MB:.2f} MiB")
    print(f"RAM    {h['ram_available'] / GiB:.1f} GiB available -> expert tier {h['tier_gb']:.1f} GiB "
          f"({int(h['tier_gb'] * GiB // m['slot_bytes'])} slots), rcs {h['rcs_gb']:.2f}, ring {h['pf_ring_gb']:.2f}, "
          f"margin {h['margin_gb']:.1f}, OS reserve {h['host_reserve_gb']:.1f}")
    print(f"Quant  {m['quant_bits']} bpw experts ({m['expert_bits']}-bit trellis), head_bits {m['head_bits']}, "
          f"slot {m['slot_bytes'] / MB:.2f} MiB, record {m['record_bytes'] / MB:.2f} MiB")
    print(f"Tier   {m['resident_experts']} of {m['experts']} experts resident ({m['resident_fraction']:.0%}); "
          f"store {m['store_bytes'] / GiB:.1f} GiB on NVMe")
    print(f"CPU    {c['ncores']} physical cores, AVX2 lane "
          f"{'on' if prof['env']['QWEN38_NV_CPU'] == '1' else 'OFF'}, tune {c['tune']}, "
          f"lane cpus {c['lane_cpus']}, reader cpus {c['reader_cpus']}")
    print(f"POL    {prof['env']['QWEN38_NV_POL']}")
    if "--write" in args:
        p = args[args.index("--write") + 1]
        write_env(prof, p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
