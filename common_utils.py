import os
from pathlib import Path
from functools import lru_cache

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
