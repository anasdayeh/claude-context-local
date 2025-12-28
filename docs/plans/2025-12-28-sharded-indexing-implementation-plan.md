# Sharded Indexing Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add sharded FAISS indexes with memory‑bounded concurrent search, provide an offline indexing CLI that uses the MCP pipeline, and enforce env reuse to avoid redundant downloads.

**Architecture:** Introduce `ShardedIndexManager` with a shard manifest and LRU‑managed loaded shards. Indexing writes into the active shard and rolls over automatically based on memory estimates. Search queries all loaded shards in parallel and merges results. Add `scripts/index_repo.py` that calls `CodeSearchServer.index_directory` so manual indexing produces identical MCP artifacts. Wire `CODE_SEARCH_DEVICE` to `CodeEmbedder` in `CodeSearchServer`.

**Tech Stack:** Python, FAISS, ThreadPoolExecutor, existing `CodeIndexManager`/`IncrementalIndexer`/MCP server.

---

### Task 1: Add shard manifest model + tests

**Files:**
- Create: `search/shard_manifest.py`
- Test: `tests/unit/test_shard_manifest.py`

**Step 1: Write the failing test**
```python
def test_manifest_round_trip(tmp_path):
    from search.shard_manifest import ShardManifest

    manifest = ShardManifest(
        version=1,
        project_path="/tmp/project",
        embedding_dimension=768,
        index_type="flat",
        shard_count=1,
        shards=[{"id": "shard_000", "path": "shards/shard_000", "vector_count": 10}],
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = ShardManifest.load(path)
    assert loaded.project_path == manifest.project_path
    assert loaded.shards[0]["id"] == "shard_000"
```

**Step 2: Run test to verify it fails**
Run: `uv run python -m pytest tests/unit/test_shard_manifest.py -v`
Expected: FAIL (module or class not found)

**Step 3: Write minimal implementation**
- Implement `ShardManifest` dataclass with `save()` and `load()`.
- Ensure ASCII‑safe JSON with stable keys.

**Step 4: Run test to verify it passes**
Run: `uv run python -m pytest tests/unit/test_shard_manifest.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add search/shard_manifest.py tests/unit/test_shard_manifest.py
git commit -m "feat: add shard manifest model"
```

---

### Task 2: Introduce ShardedIndexManager (create, rollover, add/remove)

**Files:**
- Create: `search/sharded_index_manager.py`
- Modify: `search/indexer.py` (reuse `CodeIndexManager` unchanged)
- Test: `tests/unit/test_sharded_index_manager.py`

**Step 1: Write failing tests**
```python
def test_rollover_creates_new_shard(tmp_path, monkeypatch):
    from search.sharded_index_manager import ShardedIndexManager

    mgr = ShardedIndexManager(str(tmp_path))
    # Force small shard target so rollover triggers quickly
    mgr._target_shard_bytes = 1
    shard1 = mgr.active_shard_id
    mgr._maybe_rollover(current_shard_bytes=2)
    assert mgr.active_shard_id != shard1
```

**Step 2: Run test to verify it fails**
Run: `uv run python -m pytest tests/unit/test_sharded_index_manager.py -v`
Expected: FAIL

**Step 3: Implement minimal manager**
- Implement:
  - `__init__`, `active_shard_id`, `active_manager()`
  - `_maybe_rollover()` creates new shard directory + manifest update
  - `add_embeddings()` forwards to active shard `CodeIndexManager`
  - `remove_file_chunks()` scans shards and removes where present

**Step 4: Run test to verify it passes**
Run: `uv run python -m pytest tests/unit/test_sharded_index_manager.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add search/sharded_index_manager.py tests/unit/test_sharded_index_manager.py
-git commit -m "feat: add sharded index manager with rollover"
```

---

### Task 3: Memory budgeter + LRU shard cache

**Files:**
- Modify: `search/sharded_index_manager.py`
- Modify: `common_utils.py`
- Test: `tests/unit/test_sharded_index_manager.py`

**Step 1: Write failing tests**
```python
def test_lru_eviction_respects_budget(tmp_path, monkeypatch):
    from search.sharded_index_manager import ShardedIndexManager

    mgr = ShardedIndexManager(str(tmp_path))
    mgr._max_bytes = 10
    # fake load two shards with size 8 each
    mgr._loaded_shards = {"s1": 8, "s2": 8}
    mgr._lru = ["s1", "s2"]
    mgr._enforce_budget()
    assert sum(mgr._loaded_shards.values()) <= 10
```

**Step 2: Run test to verify it fails**
Run: `uv run python -m pytest tests/unit/test_sharded_index_manager.py -v`
Expected: FAIL

**Step 3: Implement memory utilities**
- Add `get_available_memory_bytes()` in `common_utils.py` using:
  - `psutil` if installed; otherwise fallback to `os.sysconf` when available.
- In `ShardedIndexManager`, compute budget:
  - `min(CODE_SEARCH_SHARD_MEMORY_CAP_GB, available * 0.75)`
- Implement `_enforce_budget()` LRU eviction.

**Step 4: Run test to verify it passes**
Run: `uv run python -m pytest tests/unit/test_sharded_index_manager.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add search/sharded_index_manager.py common_utils.py tests/unit/test_sharded_index_manager.py
-git commit -m "feat: add memory budgeter and LRU eviction"
```

---

### Task 4: Sharded search + merge

**Files:**
- Modify: `search/sharded_index_manager.py`
- Modify: `search/searcher.py`
- Test: `tests/unit/test_sharded_search.py`

**Step 1: Write failing tests**
```python
def test_merge_top_k():
    from search.sharded_index_manager import merge_top_k
    a = [("id1", 0.9, {}), ("id2", 0.5, {})]
    b = [("id3", 0.8, {}), ("id4", 0.4, {})]
    merged = merge_top_k([a, b], k=3)
    assert [m[0] for m in merged] == ["id1", "id3", "id2"]
```

**Step 2: Run test to verify it fails**
Run: `uv run python -m pytest tests/unit/test_sharded_search.py -v`
Expected: FAIL

**Step 3: Implement search/merge**
- Add `search()` in `ShardedIndexManager`:
  - Ensure loaded shards fit budget
  - Run shard searches concurrently (ThreadPoolExecutor)
  - Merge results via `merge_top_k`
- Update `IntelligentSearcher` to detect sharded manager and call its `search()`.

**Step 4: Run test to verify it passes**
Run: `uv run python -m pytest tests/unit/test_sharded_search.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add search/sharded_index_manager.py search/searcher.py tests/unit/test_sharded_search.py
-git commit -m "feat: add sharded search and result merge"
```

---

### Task 5: Wire sharded indexing into MCP pipeline

**Files:**
- Modify: `mcp_server/code_search_server.py`
- Modify: `search/incremental_indexer.py`
- Test: `tests/integration/test_sharded_indexing.py`

**Step 1: Write failing integration test**
```python
def test_sharded_indexing_and_search(tmp_path):
    from mcp_server.code_search_server import CodeSearchServer
    server = CodeSearchServer()
    res = server.index_directory(str(tmp_path), project_name="tmp", incremental=False)
    assert res.get("success") is True
```

**Step 2: Run test to verify it fails**
Run: `uv run python -m pytest tests/integration/test_sharded_indexing.py -v`
Expected: FAIL (sharded manager not used)

**Step 3: Implement sharded wiring**
- Add env flag: `CODE_SEARCH_SHARDED_INDEX` (default on for large repos)
- If enabled, use `ShardedIndexManager` inside `CodeSearchServer.index_directory`.
- Ensure `IncrementalIndexer` can accept a sharded manager via a unified interface.

**Step 4: Run test to verify it passes**
Run: `uv run python -m pytest tests/integration/test_sharded_indexing.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add mcp_server/code_search_server.py search/incremental_indexer.py tests/integration/test_sharded_indexing.py
-git commit -m "feat: integrate sharded indexing into MCP"
```

---

### Task 6: Offline CLI using MCP pipeline

**Files:**
- Create: `scripts/index_repo.py`
- Modify: `README.md`
- Modify: `CODEX.md`

**Step 1: Write failing test**
```python
def test_index_repo_cli_help():
    import subprocess, sys
    result = subprocess.run([sys.executable, "scripts/index_repo.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
```

**Step 2: Run test to verify it fails**
Run: `uv run python -m pytest tests/unit/test_index_repo_cli.py -v`
Expected: FAIL (script missing)

**Step 3: Implement CLI**
- CLI args: path, --project-name, --storage-dir, --incremental, --sharded, --verbose, --log-file, --background
- Uses `CodeSearchServer.index_directory` with same pipeline as MCP.

**Step 4: Run test to verify it passes**
Run: `uv run python -m pytest tests/unit/test_index_repo_cli.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add scripts/index_repo.py README.md CODEX.md tests/unit/test_index_repo_cli.py
-git commit -m "feat: add offline indexer CLI"
```

---

### Task 7: Wire `CODE_SEARCH_DEVICE` + docs

**Files:**
- Modify: `mcp_server/code_search_server.py`
- Modify: `README.md`
- Modify: `CODEX.md`

**Step 1: Write failing test**
```python
def test_code_search_device_env(monkeypatch):
    from mcp_server.code_search_server import CodeSearchServer
    monkeypatch.setenv("CODE_SEARCH_DEVICE", "cpu")
    server = CodeSearchServer()
    assert server.embedder.model_name
```

**Step 2: Run test to verify it fails**
Run: `uv run python -m pytest tests/unit/test_code_search_device.py -v`
Expected: FAIL (device not wired)

**Step 3: Implement device wiring**
- Pass `device=os.getenv("CODE_SEARCH_DEVICE", "auto")` into `CodeEmbedder`.

**Step 4: Run test to verify it passes**
Run: `uv run python -m pytest tests/unit/test_code_search_device.py -v`
Expected: PASS

**Step 5: Commit**
```bash
git add mcp_server/code_search_server.py README.md CODEX.md tests/unit/test_code_search_device.py
-git commit -m "feat: wire device selection"
```

---

### Task 8: Full verification

**Step 1:** `uv run python tests/run_tests.py --integration --verbose`

**Step 2:** `uv run python tests/run_tests.py --unit --verbose`
