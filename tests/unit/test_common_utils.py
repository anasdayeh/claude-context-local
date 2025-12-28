import os
from pathlib import Path

import pytest

from common_utils import get_storage_dir


def test_storage_dir_uses_data_dir_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CODE_SEARCH_STORAGE", raising=False)
    get_storage_dir.cache_clear()
    assert get_storage_dir() == Path(tmp_path / "data")
