"""Track total device memory (including CUDA allocations outside Torch) via NVML, and host RAM usage.

HostMemory has two scopes:
  cgroup  - a capped systemd/docker scope (memory.max + memory.swap.max=0). What the campaign ran under.
  proc    - a plain desktop machine with no scope: usage is total - MemAvailable, so the desktop session,
            page cache and pinned expert arena all count. Set QWEN_RAM_LIMIT_BYTES to the budget you want.
"""
import ctypes as C
import threading
import time
import os
from pathlib import Path


def _mem_available():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return None


def _mem_total():
    return int(os.sysconf('SC_PAGE_SIZE')) * os.sysconf('SC_PHYS_PAGES')


def _own_cgroup():
    """The cgroup v2 path this process lives in (set by systemd-run --scope / docker)."""
    try:
        for line in open('/proc/self/cgroup'):
            ctrl, path = line.split(':', 1)
            if ctrl == '0':
                return Path('/sys/fs/cgroup') / path.strip().lstrip('/')
    except OSError:
        pass
    return None


class HostMemory:
    """Sample the entire capped scope, including loop backing-file cache."""

    def __init__(self, path=None, limit=None):
        limit = int(os.environ.get('QWEN_RAM_LIMIT_BYTES', '16000000000')) if limit is None else limit
        self.limit = limit; self.peak = 0; self.running = True
        if path is None and os.environ.get('QWEN_RAM_GUARD') == 'cgroup':
            path = _own_cgroup()
        self.path = Path(path) if path else None
        self.mode = 'proc'
        if self.path:
            cap = (self.path / 'memory.max').read_text().strip()
            assert cap != 'max' and int(cap) <= limit, f'Uncapped global RAM scope: {self.path}'
            assert (self.path / 'memory.swap.max').read_text().strip() == '0'
            self.mode = 'cgroup'
        elif os.environ.get('QWEN_RAM_GUARD', 'proc') == 'off':
            self.mode = 'off'
        self.total = _mem_total()
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def sample(self):
        if self.mode == 'cgroup':
            used = int((self.path / 'memory.current').read_text())
        elif self.mode == 'proc':
            avail = _mem_available()
            used = self.total - avail if avail is not None else 0
        else:
            used = 0
        self.peak = max(self.peak, used)
        return used

    def _run(self):
        while self.running:
            if self.mode != 'off' and self.sample() > self.limit:
                print(f'RAM LIMIT EXCEEDED: {self.peak} > {self.limit} ({self.mode})', flush=True)
                os._exit(87)
            time.sleep(0.02)

    def close(self):
        self.running = False; self.thread.join(); self.sample()


class DeviceMemory:
    class Info(C.Structure):
        _fields_ = [('total', C.c_ulonglong), ('free', C.c_ulonglong), ('used', C.c_ulonglong)]

    def __init__(self, limit=None, index=0):
        self.limit = int(os.environ.get('QWEN_VRAM_LIMIT_BYTES', '16000000000')) if limit is None else limit
        self.lib = C.CDLL('libnvidia-ml.so.1')
        assert self.lib.nvmlInit_v2() == 0
        self.handle = C.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(index, C.byref(self.handle)) == 0
        total = self.Info()
        assert self.lib.nvmlDeviceGetMemoryInfo(self.handle, C.byref(total)) == 0
        self.guard = os.environ.get('QWEN_VRAM_GUARD', 'on') != 'off'
        # A desktop session already holds VRAM; the guard budget is the whole card, not what is free now.
        self.limit = min(self.limit, total.total)
        self.peak = 0; self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def sample(self):
        info = self.Info()
        assert self.lib.nvmlDeviceGetMemoryInfo(self.handle, C.byref(info)) == 0
        self.peak = max(self.peak, info.used)
        return info.used

    def _run(self):
        while self.running:
            if self.guard and self.sample() > self.limit:
                print(f'VRAM LIMIT EXCEEDED: {self.peak} > {self.limit}', flush=True)
                import sys
                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized():
                    print(f'TORCH MEMORY AT EXIT: allocated={torch.cuda.memory_allocated()} '
                          f'reserved={torch.cuda.memory_reserved()}', flush=True)
                os._exit(86)
            time.sleep(0.02)

    def close(self):
        self.running = False; self.thread.join(); self.sample()
