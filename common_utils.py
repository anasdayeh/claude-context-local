"""Common utilities shared across modules."""

import os
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_storage_dir() -> Path:
    """Get or create base storage directory. Cached for performance."""
    storage_path = os.getenv('CODE_SEARCH_STORAGE', str(Path.home() / '.claude_code_search'))
    storage_dir = Path(storage_path)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir

