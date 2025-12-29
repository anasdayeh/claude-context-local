# Hybrid Search Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship hybrid retrieval (FTS5 BM25 + FAISS) that works on sharded indexes by default, silently falls back to semantic search, and auto-builds FTS in the background.

**Architecture:** Add per-shard FTS5 tables inside `metadata.db`, keep them updated on add/remove, and fuse dense + sparse via RRF at query time. `search_mode="auto"` uses hybrid when FTS is ready; otherwise semantic with a background FTS build.

**Tech Stack:** Python 3.12, SQLite FTS5 (`sqlite3`), FAISS, threading, existing MCP server/search layers.

---

### Task 1: FTS helpers in CodeIndexManager

**Files:**
- Modify: `search/indexer.py:32-799`
- Test: `tests/unit/test_fts_indexer.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_fts_indexer.py
import sqlite3
from pathlib import Path
from search.indexer import CodeIndexManager
from embeddings.embedder import EmbeddingResult
import numpy as np


def test_fts_create_and_search(tmp_path: Path):
    mgr = CodeIndexManager(str(tmp_path))
    # Create a minimal embedding result
    emb = np.ones(4, dtype=np.float32)
    result = EmbeddingResult(
        chunk_id="abc",
        embedding=emb,
        metadata={"relative_path": "src/foo.py", "content": "def create_remote_pcap():\n    pass"},
    )

    # Should upsert into FTS and be searchable
    mgr.add_embeddings([result])
    hits = mgr.fts_search("create_remote_pcap", k=5)
    assert any(cid == "abc" for cid, _score in hits)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_fts_indexer.py::test_fts_create_and_search -v`
Expected: FAIL with `AttributeError: 'CodeIndexManager' object has no attribute 'fts_search'`.

**Step 3: Write minimal implementation**

```python
# search/indexer.py (new helpers)
import sqlite3
import threading

class CodeIndexManager:
    def __init__(...):
        ...
        self._fts_lock = threading.Lock()
        self._fts_built = False
        self._fts_building = False

    def _fts_connect(self):
        conn = sqlite3.connect(str(self.metadata_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=1000;")
        return conn

    def _ensure_fts_table(self, conn):
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "chunk_id UNINDEXED, path, content, tokenize='unicode61 tokenchars \"_.\"'"
            ");"
        )

    def fts_search(self, query: str, k: int = 5):
        q = normalize_fts_query(query)
        if not q:
            return []
        try:
            with self._fts_connect() as conn:
                self._ensure_fts_table(conn)
                rows = conn.execute(
                    "SELECT chunk_id, bm25(chunks_fts) AS score "
                    "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                    (q, k),
                ).fetchall()
            return [(row[0], float(row[1])) for row in rows]
        except Exception:
            return []
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_fts_indexer.py::test_fts_create_and_search -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add search/indexer.py tests/unit/test_fts_indexer.py
git commit -m "feat: add FTS helpers to indexer"
```

---

### Task 2: Wire FTS upsert/delete into add/remove

**Files:**
- Modify: `search/indexer.py:225-299`
- Modify: `search/indexer.py:751-799`
- Test: `tests/unit/test_fts_indexer.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_fts_indexer.py

def test_fts_delete_by_path(tmp_path: Path):
    mgr = CodeIndexManager(str(tmp_path))
    emb = np.ones(4, dtype=np.float32)
    result = EmbeddingResult(
        chunk_id="xyz",
        embedding=emb,
        metadata={"relative_path": "src/bar.py", "content": "class Foo: pass"},
    )
    mgr.add_embeddings([result])
    assert mgr.fts_search("Foo", k=5)

    mgr.remove_file_chunks("src/bar.py")
    assert not mgr.fts_search("Foo", k=5)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_fts_indexer.py::test_fts_delete_by_path -v`
Expected: FAIL because FTS rows remain.

**Step 3: Write minimal implementation**

```python
# search/indexer.py (inside add_embeddings loop)
path = result.metadata.get("relative_path") or result.metadata.get("file_path")
content = result.metadata.get("content")
if path and content:
    self.fts_upsert(result.chunk_id, path, content)

# search/indexer.py (inside remove_file_chunks)
self.fts_delete_by_path(norm_target)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_fts_indexer.py::test_fts_delete_by_path -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add search/indexer.py tests/unit/test_fts_indexer.py
git commit -m "feat: sync FTS on add/remove"
```

---

### Task 3: Add RRF + query normalization utility

**Files:**
- Create: `search/hybrid.py`
- Test: `tests/unit/test_rrf_fusion.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_rrf_fusion.py
from search.hybrid import rrf_fuse, normalize_fts_query


def test_rrf_fuse_prefers_high_ranked_items():
    dense = ["a", "b", "c"]
    sparse = ["c", "a", "d"]
    results = rrf_fuse(dense, sparse, rrf_k=60, top_k=3)
    assert results[0][0] in {"a", "c"}
    assert len(results) == 3


def test_normalize_fts_query_strips_symbols():
    q = normalize_fts_query("foo.bar()? baz")
    assert q == "foo.bar OR baz"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_rrf_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.hybrid'`.

**Step 3: Write minimal implementation**

```python
# search/hybrid.py
import re
from collections import defaultdict


def normalize_fts_query(text: str) -> str:
    tokens = re.findall(r"[\w.]+", text.lower())
    return " OR ".join(tokens)


def rrf_fuse(dense_ids, sparse_ids, rrf_k=60, top_k=5):
    scores = defaultdict(float)
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] += 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        scores[cid] += 1.0 / (rrf_k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_rrf_fusion.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add search/hybrid.py tests/unit/test_rrf_fusion.py
git commit -m "feat: add RRF and FTS query normalization"
```

---

### Task 4: Sharded hybrid search + background FTS build

**Files:**
- Modify: `search/sharded_index_manager.py:28-252`
- Modify: `search/searcher.py:105-386`
- Test: `tests/integration/test_hybrid_sharded_search.py`

**Step 1: Write the failing test**

```python
# tests/integration/test_hybrid_sharded_search.py
import numpy as np
from search.sharded_index_manager import ShardedIndexManager
from embeddings.embedder import EmbeddingResult


def test_hybrid_search_across_shards(tmp_path):
    mgr = ShardedIndexManager(str(tmp_path))
    # Create two shards by forcing add_embeddings twice
    emb = np.ones(4, dtype=np.float32)
    mgr.add_embeddings([EmbeddingResult(chunk_id="c1", embedding=emb, metadata={"relative_path": "a.py", "content": "alpha token"})])
    mgr._create_new_shard()  # force rollover for test
    mgr.add_embeddings([EmbeddingResult(chunk_id="c2", embedding=emb, metadata={"relative_path": "b.py", "content": "beta token"})])

    # Hybrid search should see BM25 hits from both shards
    results = mgr.search_hybrid("beta", np.ones(4, dtype=np.float32), k=5)
    assert any(cid == "c2" for cid, _score, _meta in results)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_hybrid_sharded_search.py::test_hybrid_search_across_shards -v`
Expected: FAIL with `AttributeError: 'ShardedIndexManager' object has no attribute 'search_hybrid'`.

**Step 3: Write minimal implementation**

```python
# search/sharded_index_manager.py
from search.hybrid import rrf_fuse

class ShardedIndexManager:
    def search_hybrid(self, query_text, query_embedding, k=5, filters=None):
        dense = self.search(query_embedding, k=k, filters=filters)
        dense_ids = [cid for cid, _sim, _meta in dense]

        sparse_ids = []
        for shard in self._manifest.shards:
            manager = self._get_manager(shard["id"], enforce_budget=False)
            for cid, _score in manager.fts_search(query_text, k=k):
                sparse_ids.append(cid)

        fused = rrf_fuse(dense_ids, sparse_ids, rrf_k=60, top_k=k)
        # Map to metadata tuples
        results = []
        for cid, score in fused:
            meta = self.get_chunk_by_id(cid)
            if meta:
                results.append((cid, score, meta))
        return results
```

Add a background build trigger in `search/searcher.py` when hybrid is requested but FTS is missing.

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_hybrid_sharded_search.py::test_hybrid_search_across_shards -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add search/sharded_index_manager.py search/searcher.py tests/integration/test_hybrid_sharded_search.py
git commit -m "feat: add sharded hybrid search"
```

---

### Task 5: Auto mode wiring + fallback build trigger

**Files:**
- Modify: `mcp_server/code_search_server.py:324-386`
- Modify: `search/searcher.py:105-173`
- Test: `tests/unit/test_search_mode_auto.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_search_mode_auto.py
from search.searcher import IntelligentSearcher


def test_auto_mode_uses_hybrid_when_available(monkeypatch, mock_storage_dir):
    # stub fts availability
    monkeypatch.setattr(IntelligentSearcher, "_fts_available", True)
    # Expect search_mode auto to route to hybrid
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_search_mode_auto.py -v`
Expected: FAIL because routing is not implemented.

**Step 3: Write minimal implementation**

```python
# search/searcher.py
if search_mode == "auto" and self._fts_available:
    return self._hybrid_search(...)
if search_mode == "hybrid" and not self._fts_available:
    self._trigger_fts_build()
    return self._semantic_search(...)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_search_mode_auto.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add search/searcher.py mcp_server/code_search_server.py tests/unit/test_search_mode_auto.py
git commit -m "feat: route auto/hybrid search modes"
```

---

### Task 6: Documentation updates

**Files:**
- Modify: `README.md:169-194`
- Modify: `CODEX.md:93-118`

**Step 1: Write doc changes**

Add new env flags:
- `CODE_SEARCH_HYBRID`
- `CODE_SEARCH_HYBRID_RRF_K`
- `CODE_SEARCH_HYBRID_DENSE_K`
- `CODE_SEARCH_HYBRID_SPARSE_K`
- `CODE_SEARCH_HYBRID_AUTOBUILD`

**Step 2: Commit**

```bash
git add README.md CODEX.md
git commit -m "docs: document hybrid search flags"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2025-12-29-hybrid-search-implementation-plan.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
