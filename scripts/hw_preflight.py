#!/usr/bin/env python3
"""Preflight + measurement for running lean-qwenflash on a consumer PC.

Checks the things the engine actually needs, measures the two numbers its lane planner is built on
(NVMe O_DIRECT throughput and PCIe host->device bandwidth), and writes hw.env with the tuned values.

  python3 scripts/hw_preflight.py --model /media/deepak/E/qwen/model --store-dir /media/deepak/E/qwen/store
  python3 scripts/hw_preflight.py ... --write hw.env --measure
"""
import json, mmap, os, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hw_profile

GiB = 1024 ** 3
BLOCK = 4096


def odirect_probe(path):
    """O_DIRECT write+read round trip on a directory (ext4/xfs pass; tmpfs and NFS do not)."""
    buf = mmap.mmap(-1, BLOCK)
    probe = path / f".odirect-probe-{os.getpid()}"
    try:
        fd = os.open(probe, os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o600)
        try:
            if os.pwritev(fd, [buf], 0) != BLOCK:
                raise OSError("short O_DIRECT write")
            if os.preadv(fd, [buf], 0) != BLOCK:
                raise OSError("short O_DIRECT read")
        finally:
            os.close(fd); probe.unlink(missing_ok=True)
        return True
    except OSError as e:
        print(f"FAIL O_DIRECT at {path}: {e.strerror or e} (the expert store needs a local ext4/xfs NVMe filesystem)")
        return False


def read_throughput(path, gib=2.0, threads=8, piece_kb=1024, record_bytes=None):
    """O_DIRECT sequential read throughput of a file, in GB/s, plus single-record latency."""
    import concurrent.futures as cf
    size = path.stat().st_size
    if size < 16 * 1024 ** 2:
        print(f"SKIP read throughput: {path} is too small")
        return None, None
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    piece = piece_kb * 1024
    take = min(gib * GiB, size)
    n = int(take // piece)
    t0 = time.monotonic()
    local = threading.local()

    def job(i):
        off = (i * piece) % size
        off = off // BLOCK * BLOCK
        if not hasattr(local, "buf"):
            local.buf = mmap.mmap(-1, piece)
        mv = memoryview(local.buf)
        got = 0
        while got < piece:
            r = os.preadv(fd, [mv[got:piece]], off + got)
            if r == 0:
                break
            got += r
        return got

    with cf.ThreadPoolExecutor(threads) as ex:
        bytes_read = sum(ex.map(job, range(n)))
    dt = time.monotonic() - t0
    os.close(fd)
    gbps = bytes_read / dt / 1e9
    # one record on its own, queue depth 1: the latency the planner calls nvone
    rec = record_bytes or 1_859_072
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, (rec + BLOCK - 1) // BLOCK * BLOCK)
    lat = None
    try:
        t1 = time.monotonic(); os.preadv(fd, [buf], 0); lat = (time.monotonic() - t1) * 1000
    finally:
        os.close(fd)
    print(f"NVMe   {gbps:.2f} GB/s O_DIRECT ({threads} threads, {piece_kb} KiB pieces), single read {lat:.2f} ms")
    return gbps, lat


def pcie_bandwidth(mb=256):
    """Pinned host -> device DMA bandwidth. Needs torch; does not load the model."""
    try:
        import torch
    except ImportError:
        print("SKIP PCIe bandwidth: torch not importable")
        return None
    if not torch.cuda.is_available():
        print("SKIP PCIe bandwidth: no CUDA device")
        return None
    for try_mb in (mb, 32):
        n = try_mb * 1024 ** 2
        try:
            src = torch.frombuffer(bytearray(n), dtype=torch.uint8).pin_memory()
            dst = torch.empty(n, dtype=torch.uint8, device="cuda")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"SKIP PCIe bandwidth ({try_mb} MiB): {str(e).splitlines()[0][:90]}")
            torch.cuda.empty_cache()
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        gbps = n / (time.perf_counter() - t0) / 1e9
        print(f"PCIe   {gbps:.2f} GB/s pinned host->device ({try_mb} MiB)")
        del src, dst
        torch.cuda.empty_cache()
        return gbps
    print("SKIP PCIe bandwidth: GPU too full to allocate a probe buffer")
    return None


def store_record_bytes(store, store_dir):
    """record_bytes from the store manifest (<name>.json next to <name>.bin), if it exists."""
    if not store:
        return None
    p = Path(store).with_suffix(".json")
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text())["record_bytes"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None


def cpu_bandwidth(workers=None, mb=512):
    """Aggregate DRAM read rate at the CPU lane's thread count (numpy sums over slices; np.sum releases the GIL).
    A proxy for the lane kernel's bandwidth, not its throughput -- override with --cpu-gbps= if you measured it."""
    try:
        import numpy as np
        import concurrent.futures as cf
    except ImportError:
        print("SKIP CPU bandwidth: numpy not importable")
        return None
    cpus = hw_profile.cpu_layout()
    n_threads = workers or max(1, cpus["ncores"] - 2)
    try:
        arr = np.random.default_rng(0).random(mb * 1024 ** 2 // 4, dtype=np.float32)
    except (MemoryError, ValueError):
        print("SKIP CPU bandwidth: probe buffer too large")
        return None
    step = len(arr) // n_threads
    nbytes = len(arr) * 4
    slices = [arr[i * step:(i + 1) * step] for i in range(n_threads)]
    base = arr                      # slices are views; keep the base alive while timing
    del arr
    def s(a):
        return float(np.sum(a))
    np.sum(slices[0])               # warm up
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(n_threads) as ex:
        list(ex.map(s, slices))
    dt = time.perf_counter() - t0
    gbps = nbytes / dt / 1e9
    print(f"CPU    {gbps:.1f} GB/s aggregate DRAM read at {n_threads} threads")
    return gbps


def main():
    args = sys.argv[1:]
    def opt(name, default=None, cast=str):
        for i, a in enumerate(args):
            if a.startswith(name + "="):
                return cast(a[len(name) + 1:])
        return default
    measure = "--measure" in args
    model = Path(opt("--model", "model"))
    store_dir = Path(opt("--store-dir", str(model.parent / "store")))
    store = opt("--store", None)
    context = opt("--context", 32768, int)
    chunk = opt("--chunk", 4096, int)
    desktop = opt("--desktop-gb", None, float)
    ram = opt("--ram-gb", None, float)
    cpu_gbps = opt("--cpu-gbps", None, float)

    ok = True
    print(f"Model  {model}")
    if not (model / "config.json").exists():
        print(f"FAIL no checkpoint at {model} (run download_http.py first)")
        ok = False
    parts = sorted(model.glob("*.http-part"))
    if parts:
        print(f"WARN   download not finished: {len(parts)} part file(s) remain, e.g. {parts[0].name}")

    nvme_gbps = lat = pcie = None
    if measure:
        target = Path(store) if store else None
        if target is None or not target.exists():
            target = store_dir / ".probe.bin"
            fd = os.open(target, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
            os.posix_fallocate(fd, 0, 2 * GiB)
            os.close(fd)
        nvme_gbps, lat = read_throughput(target, record_bytes=store_record_bytes(store, store_dir))
        pcie = pcie_bandwidth()
        cpu_gbps = cpu_gbps or cpu_bandwidth()
        if target.name == ".probe.bin":
            target.unlink(missing_ok=True)

    if not odirect_probe(store_dir):
        ok = False

    measured = {'nvone': lat} if lat else None
    prof = hw_profile.derive(str(model), str(store) if store else None, context=context, chunk=chunk,
                             ram_gb=ram, desktop_gb=desktop, nvme_gbps=nvme_gbps, pcie_gbps=pcie,
                             cpu_gbps=cpu_gbps, measured=measured)
    if measure and prof["gpu"]["vram_contested"] > 1 * GiB:
        print("NOTE   another process holds the GPU and probably the disk; these NVMe/PCIe numbers are contended. "
              "Re-run --measure after stopping it.")
    for w in prof["warnings"]:
        print(f"WARN   {w}")
    m = prof["model"]
    free_model = os.statvfs(model).f_bavail * os.statvfs(model).f_frsize
    free_store = os.statvfs(store_dir).f_bavail * os.statvfs(store_dir).f_frsize
    print(f"Disk   checkpoint {m['disk_bytes'] / GiB:.1f} GiB on disk ({m['checkpoint_bytes'] / GiB:.1f} GiB of tensors) "
          f"({free_model / GiB:.1f} GiB free at {model})")
    print(f"Disk   store {m['store_bytes'] / GiB:.1f} GiB ({free_store / GiB:.1f} GiB free at {store_dir})")
    existing = Path(store) if store else None
    if existing and existing.exists() and existing.stat().st_size >= m["store_bytes"]:
        print(f"PASS   store already on disk: {existing.name} ({existing.stat().st_size / GiB:.1f} GiB)")
    elif m["store_bytes"] > free_store * 0.9:
        print(f"FAIL store needs {m['store_bytes'] / GiB:.1f} GiB, only {free_store / GiB:.1f} GiB free at {store_dir}")
        ok = False
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    ml = 'unlimited' if soft == resource.RLIM_INFINITY else f'{soft // 1024} KiB'
    print(f"Memlock {ml} ({'ok' if ml == 'unlimited' else 'raise it with ulimit -l unlimited: the RAM tier is cudaHostRegister-ed'})")
    swap = [l for l in open("/proc/meminfo") if l.startswith("SwapTotal")][0].split()[1]
    if int(swap) > 0:
        print(f"WARN   swap is enabled ({int(swap) // 1024} MiB): swapped-out pinned experts stall the GPU lane")

    print(f"GPU    {prof['gpu']['name']} budget {prof['gpu']['vram_limit'] / GiB:.2f} GiB -> "
          f"{m['vram_slots']} VRAM slots; RAM tier {prof['host']['tier_gb']:.1f} GiB; "
          f"{m['resident_experts']}/{m['experts']} experts resident ({m['resident_fraction']:.0%})")
    print(f"CPU    {prof['cpu']['ncores']} cores, lane {'on' if prof['cpu']['lane_ok'] else 'off'}, {prof['cpu']['tune']}")
    print(f"POL    {prof['env']['QWEN38_NV_POL']}")

    if "--write" in args:
        path = args[args.index("--write") + 1]
        hw_profile.write_env(prof, path)
        print(f"wrote {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
