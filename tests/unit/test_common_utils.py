import os
from pathlib import Path

import pytest

from common_utils import apply_adaptive_runtime_defaults, get_storage_dir


def test_storage_dir_uses_data_dir_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CODE_SEARCH_STORAGE", raising=False)
    get_storage_dir.cache_clear()
    assert get_storage_dir() == Path(tmp_path / "data")


def test_adaptive_defaults_for_apple_silicon(monkeypatch):
    env = {"CODE_SEARCH_DEVICE": "auto"}
    monkeypatch.setattr("common_utils.get_total_memory_bytes", lambda: 16 * 1024 ** 3)
    monkeypatch.setattr("common_utils.platform.system", lambda: "Darwin")
    monkeypatch.setattr("common_utils.platform.machine", lambda: "arm64")

    applied = apply_adaptive_runtime_defaults(env)

    assert applied["CODE_SEARCH_EMBED_BATCH_SIZE"] == "4"
    assert applied["CODE_SEARCH_INDEX_WORKERS"] == "1"
    assert applied["CODE_SEARCH_SHARD_SEARCH_WORKERS"] == "1"
    assert applied["CODE_SEARCH_SHARD_MEMORY_CAP_GB"] == "5"
    assert applied["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.95"
    assert applied["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.85"
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_adaptive_defaults_do_not_override_explicit_values(monkeypatch):
    env = {
        "CODE_SEARCH_DEVICE": "mps",
        "CODE_SEARCH_EMBED_BATCH_SIZE": "32",
        "CODE_SEARCH_SHARD_MEMORY_CAP_GB": "9",
    }
    monkeypatch.setattr("common_utils.get_total_memory_bytes", lambda: 16 * 1024 ** 3)
    monkeypatch.setattr("common_utils.platform.system", lambda: "Darwin")
    monkeypatch.setattr("common_utils.platform.machine", lambda: "arm64")

    applied = apply_adaptive_runtime_defaults(env)

    assert "CODE_SEARCH_EMBED_BATCH_SIZE" not in applied
    assert env["CODE_SEARCH_EMBED_BATCH_SIZE"] == "32"
    assert "CODE_SEARCH_SHARD_MEMORY_CAP_GB" not in applied
    assert env["CODE_SEARCH_SHARD_MEMORY_CAP_GB"] == "9"
