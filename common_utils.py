import os
import platform
import subprocess
from pathlib import Path
from functools import lru_cache
from typing import Dict, MutableMapping, Optional

def get_available_memory_bytes() -> int:
    """Return available memory in bytes when possible."""
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size) * int(avail_pages)
    except Exception:
        return 0


@lru_cache(maxsize=1)
def get_total_memory_bytes() -> int:
    """Return total system memory in bytes when possible."""
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().total)
    except Exception:
        pass

    # macOS fallback where sysconf is often less reliable for total memory.
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out)
        except Exception:
            pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size) * int(total_pages)
    except Exception:
        return 0


def _is_unset(env: MutableMapping[str, str], key: str) -> bool:
    value = env.get(key)
    return value is None or str(value).strip() == ""


def _default_embed_batch(total_gb: float, prefer_mps: bool) -> int:
    if prefer_mps:
        if total_gb <= 16:
            return 4
        if total_gb <= 24:
            return 8
        return 12
    if total_gb <= 16:
        return 8
    if total_gb <= 24:
        return 16
    return 24


def _default_chunk_batch(total_gb: float) -> int:
    if total_gb <= 16:
        return 50
    if total_gb <= 24:
        return 80
    return 100


def _default_training_sample_max(total_gb: float) -> int:
    if total_gb <= 16:
        return 12000
    if total_gb <= 24:
        return 18000
    return 25000


def _default_shard_cap_gb(total_gb: float) -> int:
    if total_gb <= 0:
        return 6
    # Keep shard resident memory conservative on unified-memory machines.
    return max(2, min(12, int(total_gb * 0.35)))


def _default_index_workers(total_gb: float) -> int:
    if total_gb <= 16:
        return 1
    if total_gb <= 32:
        return 2
    return 3


def _default_shard_search_workers(total_gb: float) -> int:
    if total_gb <= 16:
        return 1
    if total_gb <= 32:
        return 2
    return 3


def _default_min_free_ram_gb(total_gb: float) -> float:
    if total_gb <= 0:
        return 1.5
    return round(max(1.0, min(4.0, total_gb * 0.10)), 1)


def _default_torch_threads(total_gb: float) -> int:
    if total_gb <= 16:
        return 2
    if total_gb <= 32:
        return 4
    return 6


def _default_torch_interop_threads(total_gb: float) -> int:
    if total_gb <= 16:
        return 1
    if total_gb <= 32:
        return 2
    return 3


def apply_adaptive_runtime_defaults(
    env: Optional[MutableMapping[str, str]] = None,
) -> Dict[str, str]:
    """Apply conservative memory-aware defaults when callers did not set overrides.

    The goal is to prevent machine lockups on constrained systems (for example
    Apple Silicon laptops with unified memory) while still preserving throughput.
    """
    target_env = env if env is not None else os.environ
    applied: Dict[str, str] = {}

    total_bytes = get_total_memory_bytes()
    total_gb = (float(total_bytes) / (1024 ** 3)) if total_bytes > 0 else 0.0
    requested_device = str(target_env.get("CODE_SEARCH_DEVICE", "auto") or "auto").lower()
    is_apple_silicon = (
        platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"}
    )
    prefer_mps = is_apple_silicon and requested_device in {"auto", "mps"}

    def set_default(key: str, value: str) -> None:
        if _is_unset(target_env, key):
            target_env[key] = value
            applied[key] = value

    set_default("CODE_SEARCH_EMBED_BATCH_SIZE", str(_default_embed_batch(total_gb, prefer_mps)))
    set_default("CODE_SEARCH_CHUNK_BATCH_SIZE", str(_default_chunk_batch(total_gb)))
    set_default("CODE_SEARCH_TRAIN_SAMPLE_MAX", str(_default_training_sample_max(total_gb)))
    set_default("CODE_SEARCH_SHARD_MEMORY_CAP_GB", str(_default_shard_cap_gb(total_gb)))
    set_default("CODE_SEARCH_INDEX_WORKERS", str(_default_index_workers(total_gb)))
    set_default("CODE_SEARCH_SHARD_SEARCH_WORKERS", str(_default_shard_search_workers(total_gb)))
    set_default("CODE_SEARCH_MIN_FREE_RAM_GB", str(_default_min_free_ram_gb(total_gb)))
    set_default("CODE_SEARCH_TORCH_NUM_THREADS", str(_default_torch_threads(total_gb)))
    set_default("CODE_SEARCH_TORCH_INTEROP_THREADS", str(_default_torch_interop_threads(total_gb)))

    if prefer_mps:
        # PyTorch docs: >1.0 allows allocating beyond recommended working set.
        # Keep a lower ceiling by default on memory-constrained Apple Silicon.
        high = "0.95" if total_gb <= 16 else "1.0"
        low = "0.85" if total_gb <= 16 else "0.9"
        set_default("PYTORCH_MPS_HIGH_WATERMARK_RATIO", high)
        set_default("PYTORCH_MPS_LOW_WATERMARK_RATIO", low)
        set_default("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    return applied

@lru_cache(maxsize=1)
def get_storage_dir() -> Path:
    """Get or create base storage directory. Cached for performance."""
    # Default to the project root directory where this tool is installed
    default_path = Path(__file__).parent.resolve()
    storage_path = (
        os.getenv('CODE_SEARCH_STORAGE')
        or os.getenv('CODE_SEARCH_DATA_DIR')
        or str(default_path)
    )
    storage_dir = Path(storage_path)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir
