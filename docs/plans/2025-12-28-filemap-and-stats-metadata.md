# File Map Deletes + Stats/Training Metadata Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a secondary file_path→chunk_id map to speed deletes, expose index metadata in stats.json, and persist a capped training sample for future IVF experiments without changing current index behavior.

**Architecture:** Extend `CodeIndexManager` with a `file_map_db` (SqliteDict) and update it on add/remove. Use it to accelerate `remove_file_chunks` with a fallback scan. Extend stats reporting to include FAISS index metadata. Add a lightweight reservoir sample store for embeddings + metadata for future training.

**Tech Stack:** Python, FAISS, SqliteDict, NumPy.

---

### Task 1: Add file_map_db storage + fast delete path

**Files:**
- Modify: `search/indexer.py`
- Test: `tests/unit/test_file_map_deletes.py`

**Step 1: Write the failing test**

```python
import numpy as np


def test_remove_file_chunks_uses_file_map(tmp_path):
    from search.indexer import CodeIndexManager
    from embeddings.embedder import EmbeddingResult
    from chunking.code_chunk import CodeChunk

    index_dir = tmp_path / "index"
    index = CodeIndexManager(str(index_dir))

    chunk = CodeChunk(
        name="f",
        chunk_type="function",
        content="def f(): pass",
        start_line=1,
        end_line=1,
        relative_path="a.py",
        file_path=str(tmp_path / "a.py"),
        tags=[],
        parent_name=None,
        docstring=None,
        folder_structure=[],
    )

    emb = EmbeddingResult(chunk=chunk, embedding=np.ones(4, dtype=np.float32))
    index.add_embeddings([emb])

    # Expect: removal succeeds without metadata scan when file_map_db has entry.
    removed = index.remove_file_chunks("a.py")
    assert removed == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_file_map_deletes.py::test_remove_file_chunks_uses_file_map -v`
Expected: FAIL (file_map not implemented)

**Step 3: Write minimal implementation**

- Add `self.file_map_path` and `self._file_map_db` to `CodeIndexManager`.
- Add `file_map_db` property via `_open_sqlitedict`.
- In `add_embeddings`, update file_map entries:
  - Normalize file path key (`relative_path` preferred).
  - Remove IDs from old path if entry exists in metadata for that ID.
  - Append IDs to `file_map_db[path_key]` (store list of ints).
- In `remove_file_chunks`, check file_map first. If found, remove those IDs without scanning metadata.
- Fallback to existing scan if file_map missing.
- In `clear_index`, delete file_map db + wal/shm.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_file_map_deletes.py::test_remove_file_chunks_uses_file_map -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_file_map_deletes.py search/indexer.py
git commit -m "feat: add file map for fast deletes"
```

---

### Task 2: Expose FAISS index metadata in stats.json

**Files:**
- Modify: `search/indexer.py`
- Test: `tests/unit/test_index_stats_metadata.py`

**Step 1: Write the failing test**

```python
import numpy as np


def test_stats_include_index_metadata(tmp_path):
    from search.indexer import CodeIndexManager
    from embeddings.embedder import EmbeddingResult
    from chunking.code_chunk import CodeChunk

    index_dir = tmp_path / "index"
    index = CodeIndexManager(str(index_dir))

    chunk = CodeChunk(
        name="f",
        chunk_type="function",
        content="def f(): pass",
        start_line=1,
        end_line=1,
        relative_path="a.py",
        file_path=str(tmp_path / "a.py"),
        tags=[],
        parent_name=None,
        docstring=None,
        folder_structure=[],
    )

    emb = EmbeddingResult(chunk=chunk, embedding=np.ones(4, dtype=np.float32))
    index.add_embeddings([emb])

    stats = index.get_stats()
    assert "index_type" in stats
    assert "metric" in stats
    assert "embedding_dim" in stats
    assert "trained" in stats
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_index_stats_metadata.py::test_stats_include_index_metadata -v`
Expected: FAIL (missing fields)

**Step 3: Write minimal implementation**

- Add helper `_get_index_metadata()` in `CodeIndexManager`:
  - Determine base index type (unwrap IDMap2 if needed).
  - Extract metric (IP vs L2), embedding dimension, trained flag.
  - For IVF, expose nlist + nprobe if present.
- Include these fields in `_update_stats` and `get_stats` fallback.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_index_stats_metadata.py::test_stats_include_index_metadata -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_index_stats_metadata.py search/indexer.py
git commit -m "feat: expose faiss metadata in stats"
```

---

### Task 3: Add training sample storage (reservoir sampling)

**Files:**
- Create: `search/training_sample.py`
- Modify: `search/indexer.py`
- Test: `tests/unit/test_training_sample.py`

**Step 1: Write the failing test**

```python
import numpy as np


def test_training_sample_capped(tmp_path):
    from search.training_sample import TrainingSampleStore

    store = TrainingSampleStore(tmp_path, max_vectors=10)
    for i in range(25):
        vec = np.ones(4, dtype=np.float32) * i
        store.add(vec, {"path": f"f{i}.py"})

    store.save()

    data = store.load()
    assert data["vectors"].shape[0] == 10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_training_sample.py::test_training_sample_capped -v`
Expected: FAIL (store missing)

**Step 3: Write minimal implementation**

- Implement `TrainingSampleStore` with reservoir sampling:
  - `max_vectors` from env (`CODE_SEARCH_TRAIN_SAMPLE_MAX=25000`).
  - Keep count and replace randomly after cap reached.
  - Store vectors in `training_sample.npy` and metadata in `training_sample_meta.json`.
- In `add_embeddings`, append to the training sample store (only for `embedding_results`).

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_training_sample.py::test_training_sample_capped -v`
Expected: PASS

**Step 5: Commit**

```bash
git add search/training_sample.py tests/unit/test_training_sample.py search/indexer.py
git commit -m "feat: store capped training samples"
```

---

### Task 4: Integration verification

**Files:**
- Run tests

**Step 1: Run integration suite**

Run: `uv run python tests/run_tests.py --integration --verbose`
Expected: PASS

**Step 2: Commit (if needed)**

```bash
git add docs/plans/2025-12-28-filemap-and-stats-metadata.md

git commit -m "docs: add file map + stats + training sample plan"
```
