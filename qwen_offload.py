"""Qwen3.8 EXL3 adapter for the GLM disk/RAM/VRAM expert engine."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Defaults measured on the campaign server (RTX 3090 24 GB, PCIe 4.0 x16, 4-drive NVMe RAID0, 24-core EPYC, 55 GiB cap).
# On a smaller PC these are wrong; hw_profile.derive() recomputes them from the actual GPU, RAM, CPU and storage.
SERVER_DEFAULTS = {
    'QWEN38_NV': '2', 'QWEN38_EC': '1',
    'QWEN38_NV_RAM_GB': '6', 'QWEN38_NV_CPU': '0', 'QWEN38_RCS_GB': '0.25',
    'QWEN38_NV_EXCL': '1', 'QWEN38_NV_PREFETCH': '0',
    'QWEN38_NV_THREADS': '12', 'QWEN38_NV_READER_CPUS': '2-11',
    'QWEN38_NV_CTL_CPU': '1', 'QWEN38_NV_CPU_CPUS': '2-7',
    'QWEN38_NV_PIECE_KB': '256', 'QWEN38_NV_PF_RING': '512',
    'QWEN38_NV_PF_QD': '24', 'QWEN38_NV_VRING': '24',
    'QWEN38_EC_STAGE_GB': '1.1', 'QWEN38_EC_STAGE_MIN': '512',
    'QWEN38_EC_ELASTIC_GB': '6', 'QWEN38_EC_RESERVE_GB': '1.7',
    'QWEN38_NV_COPY_GRID': '24', 'QWEN38_EC_COPY_GRID': '24',
    'EXL3_NGRAM_STREAM': '1', 'QWEN_VRAM_LIMIT_BYTES': '16000000000',
    'MAX_JOBS': '4', 'TORCH_CUDA_ARCH_LIST': '8.9',
}


def profile_for(model_dir, store=None, ram_gb=None, cache_tokens=32768, chunk=4096, cache_quant=4, mtp=False):
    """Derive the hardware budgets, or None if the host/checkpoint cannot be read yet."""
    try:
        sys.path.insert(0, str(ROOT))
        import hw_profile
        desktop = os.environ.get('QWEN_DESKTOP_GB')
        return hw_profile.derive(str(model_dir), store, context=cache_tokens, chunk=chunk, cache_quant=cache_quant,
                                 mtp=mtp, ram_gb=ram_gb, desktop_gb=float(desktop) if desktop else None,
                                 nvme_gbps=float(os.environ['QWEN_NVME_GBPS']) if os.environ.get('QWEN_NVME_GBPS') else None,
                                 pcie_gbps=float(os.environ['QWEN_PCIE_GBPS']) if os.environ.get('QWEN_PCIE_GBPS') else None,
                                 measured=json_load_policy(os.environ.get('QWEN_NV_MEASURED_POL')))
    except Exception as e:
        print(f'hw_profile unavailable ({e}); using the measured-server defaults', flush=True)
        return None


def json_load_policy(spec):
    if not spec:
        return None
    import json
    try:
        return {k: float(v) for k, v in json.loads(spec).items()}
    except Exception:
        return None


def apply_profile(profile, ram_gb=None):
    """Set the engine's environment from a profile, keeping anything the caller already set."""
    env = dict(SERVER_DEFAULTS)
    if profile:
        env.update(profile['env'])
        env['QWEN38_EC_STAGE_MIN'] = '512'
    if ram_gb is not None:
        env['QWEN38_NV_RAM_GB'] = f'{ram_gb:.1f}'
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


def load(model_dir, store, chunk=2048, cache_tokens=16384, ram_gb=None, cpu=False,
         mtp=0, embed_disk=False, cache_quant=None, profile=None):
    import torch
    model_dir = Path(model_dir).resolve(); store = Path(store).resolve()
    sys.path.insert(0, str(ROOT / 'qwen38-flash-offload' / 'qwen38'))
    if profile is None:
        profile = profile_for(model_dir, store, ram_gb, cache_tokens, chunk,
                              int(cache_quant or 0), bool(mtp))
    env = apply_profile(profile, ram_gb)
    os.environ['QWEN38_NV_STORE'] = str(store)      # authoritative: the store the caller asked for
    if cpu:
        os.environ['QWEN38_NV_CPU'] = '1'
    torch.set_num_threads(1)
    torch.cuda.init()
    _, total = torch.cuda.mem_get_info()
    # Leave room for the CUDA context, non-PyTorch allocations and allocator fragmentation.
    # The cap is cumulative for the process, so add back what this process already reserves.
    limit = int(os.environ['QWEN_VRAM_LIMIT_BYTES'])
    reserved = torch.cuda.memory_reserved()
    torch.cuda.set_per_process_memory_fraction(min(1.0, (reserved + limit) / total))
    if os.environ.get('QWEN_EXL3_SOURCE') == '1':
        sys.path.insert(0, str(ROOT / 'exllamav3'))
    from exllamav3 import model_init
    import argparse, nv_tier, expert_cache, nv2
    if os.environ.get('QWEN_PREBUILT_EXT') == '1':
        # Explicit opt-in for the recovered binaries built with this same Torch/CUDA ABI.
        import importlib.util
        for module, directory, name in ((nv_tier, 'nv', 'qwen38_nv'),
                                        (expert_cache, 'ec', 'qwen38_ec'), (nv2, 'nv2', 'qwen38_nv2')):
            path = ROOT / 'qwen38-flash-offload' / 'build' / directory / (name + '.so')
            spec = importlib.util.spec_from_file_location(name, path)
            native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
            module._EXT = native
    if mtp: os.environ['QWEN_INCLUDE_MTP'] = '1'
    nv_tier.install(); expert_cache.install()
    ap = argparse.ArgumentParser(); model_init.add_args(ap, cache=True, add_draft_model_args=bool(mtp))
    argv = ['-m', str(model_dir), '-cs', str(cache_tokens), '-rcs', os.environ['QWEN38_RCS_GB'],
            '-chunk_size', str(chunk), '-ambs', '1', '-nw']
    if mtp: argv += ['-mtp', '-ndt', str(mtp)]
    if embed_disk: argv += ['-embd']
    if cache_quant is not None: argv += ['-cq', str(cache_quant)]
    args = ap.parse_args(argv)
    loaded = model_init.init(args)
    if mtp:
        model, config, cache, tok, draft_model, _draft_config, draft_cache = loaded
        mods = expert_cache.find_moe(model.modules) + expert_cache.find_moe(draft_model.modules)
        pools = expert_cache.attach_modules(mods)
        expert_cache._hook_prefill(type(model)); expert_cache._hook_prefill(type(draft_model))
    else:
        model, config, cache, tok = loaded; draft_model = draft_cache = None
        pools = expert_cache.attach(model)
    assert config.arch_string == 'Qwen4ExpForConditionalGeneration'
    nv2.finish(model)
    from exllamav3.modules.ngram_embedding import NGramEmbedding
    def walk(mods):
        for m in mods:
            yield m
            yield from walk(getattr(m, 'modules', []) or [])
    tables = [m for m in walk(model.modules) if isinstance(m, NGramEmbedding)]
    assert tables and all(m.mode == 'trellis_disk' for m in tables), 'All n-gram tables must stream from disk'
    print('NGRAM_MODES', [(m.key, m.mode, m.num_rows) for m in tables], flush=True)
    slots = getattr(pools[0], 'S', 0) if pools else 0
    print('VRAM_SLOTS', slots, 'slot_bytes', getattr(pools[0], 'rec', 0) if pools else 0,
          'RAM_TIER_GiB', os.environ['QWEN38_NV_RAM_GB'], 'VRAM_LIMIT_BYTES', limit, flush=True)
    return model, config, cache, tok, pools, tables, draft_model, draft_cache
