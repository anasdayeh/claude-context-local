# MCP Stability Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce MCP indexing failures and memory/disk crashes with adaptive batching, safer I/O, and clearer warnings while keeping coverage high.

**Architecture:** Add adaptive embedding backoff + device fallback inside the embedder, introduce warn-only disk/large-file checks in the incremental indexer, improve file decoding robustness in tree-sitter, and restore missing preload hook. Ensure storage dir honors CODE_SEARCH_DATA_DIR alias and document new warn-only settings.

**Tech Stack:** Python 3, pytest, sentence-transformers, FAISS, FastMCP.

---

### Task 1: Honor CODE_SEARCH_DATA_DIR alias for storage

**Files:**
- Create: `tests/unit/test_common_utils.py`
- Modify: `common_utils.py`

**Step 1: Write the failing test**

```python
import os
from pathlib import Path

import pytest

from common_utils import get_storage_dir


def test_storage_dir_uses_data_dir_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CODE_SEARCH_STORAGE", raising=False)
    get_storage_dir.cache_clear()
    assert get_storage_dir() == Path(tmp_path / "data")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_common_utils.py -v`
Expected: FAIL because `CODE_SEARCH_DATA_DIR` is ignored.

**Step 3: Write minimal implementation**

```python
storage_path = os.getenv("CODE_SEARCH_STORAGE") or os.getenv("CODE_SEARCH_DATA_DIR") or str(default_path)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_common_utils.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit/test_common_utils.py common_utils.py
git commit -m "test: cover CODE_SEARCH_DATA_DIR alias"
```

### Task 2: Restore model preload hook safely

**Files:**
- Modify: `tests/unit/test_mcp_server.py`
- Modify: `mcp_server/code_search_server.py`

**Step 1: Write the failing test**

```python
import mcp_server.code_search_server as css


def test_preload_hook_exists_and_calls_embed_query(monkeypatch):
    server = css.CodeSearchServer()
    called = {"ok": False}

    class DummyEmbedder:
        def embed_query(self, text):
            called["ok"] = True
            return [0.0]

    server.embedder = DummyEmbedder()
    assert hasattr(server, "_maybe_start_model_preload")
    server._maybe_start_model_preload()
    assert called["ok"] is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_mcp_server.py -v`
Expected: FAIL because `_maybe_start_model_preload` is missing.

**Step 3: Write minimal implementation**

```python
    def _maybe_start_model_preload(self) -> None:
        try:
            self.embedder.embed_query("warmup")
        except Exception:
            pass
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_mcp_server.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit/test_mcp_server.py mcp_server/code_search_server.py
git commit -m "fix: restore model preload hook"
```

### Task 3: Read non-UTF-8 files without hard failure in tree-sitter

**Files:**
- Modify: `tests/unit/test_tree_sitter_main.py`
- Modify: `chunking/tree_sitter.py`

**Step 1: Write the failing test**

```python
import logging

import pytest


def test_non_utf8_file_does_not_log_error(self, caplog):
    import chunking.tree_sitter as tsf

    if 'python' not in tsf.AVAILABLE_LANGUAGES:
        pytest.skip("tree-sitter-python not installed")

    file_path = Path(self.temp_dir) / 'bad.py'
    file_path.write_bytes(b"def ok():\n    return 'x'\n\xff")

    with caplog.at_level(logging.ERROR):
        _ = self.chunker.chunk_file(str(file_path))

    assert "Failed to read file" not in caplog.text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_tree_sitter_main.py -v`
Expected: FAIL because the read error is logged.

**Step 3: Write minimal implementation**

```python
with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_tree_sitter_main.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit/test_tree_sitter_main.py chunking/tree_sitter.py
git commit -m "fix: tolerate non-utf8 in tree-sitter reads"
```

### Task 4: Add adaptive embedding backoff for OOM

**Files:**
- Create: `tests/unit/test_embedder_adaptive.py`
- Modify: `embeddings/embedder.py`

**Step 1: Write the failing test**

```python
import logging
import numpy as np

from chunking.code_chunk import CodeChunk
from embeddings.embedder import CodeEmbedder


def _make_chunks(n):
    chunks = []
    for i in range(n):
        chunks.append(
            CodeChunk(
                content=f"def f{i}(): pass",
                chunk_type="function",
                start_line=1,
                end_line=1,
                file_path=f"/tmp/f{i}.py",
                relative_path=f"f{i}.py",
                folder_structure=[],
                name=f"f{i}",
            )
        )
    return chunks


def test_embedder_backoff_on_oom(monkeypatch):
    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder.model_name = "test"
    embedder._logger = logging.getLogger("test")

    def fake_encode(texts):
        if len(texts) > 1:
            raise RuntimeError("MPS backend out of memory")
        return np.ones((len(texts), 3), dtype=np.float32)

    embedder._encode_documents = fake_encode
    embedder._clear_device_cache = lambda: None

    results = embedder.embed_chunks(_make_chunks(3), batch_size=4)
    assert len(results) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_embedder_adaptive.py -v`
Expected: FAIL because batch OOM is not retried.

**Step 3: Write minimal implementation**

```python
# in embed_chunks
current_batch = min(batch_size, total - i)
while True:
    try:
        batch_embeddings = self._encode_documents(batch_texts)
        break
    except Exception as e:
        if not self._is_oom_error(e) or current_batch == 1:
            raise
        self._clear_device_cache()
        current_batch = max(1, current_batch // 2)
```

Add helper methods:
- `_is_oom_error(self, exc: Exception) -> bool`
- `_clear_device_cache(self) -> None`

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_embedder_adaptive.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit/test_embedder_adaptive.py embeddings/embedder.py
git commit -m "feat: add adaptive batch backoff on embedding OOM"
```

### Task 5: Warn on low disk and large files (warn-only)

**Files:**
- Create: `tests/unit/test_incremental_indexer_warnings.py`
- Modify: `search/incremental_indexer.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_incremental_indexer_warnings.py -v`
Expected: FAIL because warnings are not implemented.

**Step 3: Write minimal implementation**

```python
import shutil

# in __init__
self.disk_warn_gb = self._read_env_int("CODE_SEARCH_DISK_WARN_GB", 5)
self.large_file_mb = self._read_env_int("CODE_SEARCH_LARGE_FILE_MB", 20)

# helper methods

def _warn_if_low_disk(self):
    usage = shutil.disk_usage(self.snapshot_manager.storage_dir)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < self.disk_warn_gb:
        logger.warning("Low disk space: %.2f GB free at %s", free_gb, self.snapshot_manager.storage_dir)


def _warn_if_large_file(self, file_path: str):
    size_mb = os.path.getsize(file_path) / (1024 ** 2)
    if size_mb >= self.large_file_mb:
        logger.warning("Large file %.2f MB: %s", size_mb, file_path)
```

Call `_warn_if_low_disk()` near the start of `_full_index`, and `_warn_if_large_file()` inside `_iter_chunks` before chunking each file.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_incremental_indexer_warnings.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/unit/test_incremental_indexer_warnings.py search/incremental_indexer.py
git commit -m "feat: warn on low disk and large files"
```

### Task 6: Document new warn-only behavior and env vars

**Files:**
- Modify: `README.md`
- Modify: `CODEX.md`
- Modify: `CHANGELOG.md`

**Step 1: Update docs**

Add entries for:
- `CODE_SEARCH_DATA_DIR` (alias for `CODE_SEARCH_STORAGE`)
- `CODE_SEARCH_DISK_WARN_GB`
- `CODE_SEARCH_LARGE_FILE_MB`
- Note that warnings do not skip files by default

**Step 2: Skip tests (docs-only change)**

No test run required.

**Step 3: Commit**

```bash
git add README.md CODEX.md CHANGELOG.md
git commit -m "docs: describe new warn-only safety knobs"
```
