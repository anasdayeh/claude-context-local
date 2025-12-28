import logging
from pathlib import Path

import pytest

from search.incremental_indexer import IncrementalIndexer


class DummyIndex:
    def clear_index(self):
        pass


class DummyEmbedder:
    pass


class DummyChunker:
    pass


def test_warns_on_low_disk(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("CODE_SEARCH_DISK_WARN_GB", "999")

    def fake_usage(_):
        class Usage:
            total = 100
            used = 99
            free = 1
        return Usage()

    monkeypatch.setattr("search.incremental_indexer.shutil.disk_usage", fake_usage)

    idx = IncrementalIndexer(DummyIndex(), DummyEmbedder(), DummyChunker(), str(tmp_path))
    with caplog.at_level(logging.WARNING):
        idx._warn_if_low_disk()

    assert "Low disk space" in caplog.text


def test_warns_on_large_file(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("CODE_SEARCH_LARGE_FILE_MB", "0")
    big_file = tmp_path / "big.txt"
    big_file.write_text("x" * 10)

    idx = IncrementalIndexer(DummyIndex(), DummyEmbedder(), DummyChunker(), str(tmp_path))
    with caplog.at_level(logging.WARNING):
        idx._warn_if_large_file(str(big_file))

    assert "Large file" in caplog.text
