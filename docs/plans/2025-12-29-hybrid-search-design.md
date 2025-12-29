# Hybrid Search (FTS5 + FAISS) Design

**Goal:** Add hybrid retrieval (BM25 + dense vectors) that works for sharded indexes on day one, stays fully local, and silently falls back to semantic search while auto-building FTS in the background.

---

## 1) Summary

- Add per-shard SQLite FTS5 table inside existing `metadata.db`.
- Query BM25 per shard and fuse with dense FAISS results via RRF.
- If FTS is missing or unavailable, return semantic results and kick off an async FTS build so hybrid works on the next try.
- Default behavior: `search_mode="auto"` uses hybrid when FTS is available; otherwise semantic.

---

## 2) Storage Layout (Per Shard)

```
projects/<project_hash>/index/
  shards/
    shard_000/
      code.index
      metadata.db   <-- contains SqliteDict tables + chunks_fts (FTS5)
      id_map.db
      stats.json
```

FTS5 lives inside `metadata.db` to keep lifecycle aligned with shard data.

---

## 3) FTS Schema

```
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
USING fts5(
  chunk_id UNINDEXED,
  path,
  content,
  tokenize = 'unicode61 tokenchars "_."'
);
```

- `chunk_id` is retrievable but not indexed.
- `tokenchars` preserves `_` and `.` for code identifiers.

---

## 4) Indexing Flow

### Add embeddings (incremental)
- When `CodeIndexManager.add_embeddings()` stores metadata, also upsert FTS rows:
  - `DELETE FROM chunks_fts WHERE chunk_id = ?` (optional)
  - `INSERT INTO chunks_fts(chunk_id, path, content) VALUES (?, ?, ?)`

### Remove file chunks (incremental)
- When `CodeIndexManager.remove_file_chunks()` removes a file, also:
  - `DELETE FROM chunks_fts WHERE path = ?`

### Lazy FTS build (first-use)
- On hybrid search, if FTS is missing:
  - Start a background build for each shard:
    - Create `chunks_fts` table if missing.
    - Scan `metadata_db` and bulk insert `(chunk_id, path, content)`.
    - Mark `fts_built=true` when complete.

---

## 5) Search Flow (Hybrid)

### Auto mode
- If `CODE_SEARCH_HYBRID=0`: always semantic
- Else if `search_mode="hybrid"`: use hybrid if FTS available, otherwise fallback semantic and start build
- Else if `search_mode="auto"` and FTS available: hybrid
- Else: semantic

### Hybrid search
1. Dense search via FAISS (per shard)
2. Sparse BM25 search via FTS5 (per shard)
3. Aggregate sparse results across shards
4. Fuse dense + sparse with RRF
5. Apply existing filters and result enrichment

RRF formula:
```
score += 1 / (rrf_k + rank)
```

---

## 6) Background Build Strategy (Threaded)

Use a lightweight thread per shard (best-effort):
- Low surface area
- No new job types
- Avoids blocking search

Guard with in-memory `fts_building` to prevent duplicate builds. If the process restarts, builds are idempotent.

---

## 7) Error Handling

- All FTS operations are best-effort and must never fail the request.
- If FTS errors:
  - Log once per shard
  - Fallback to semantic
  - Leave `fts_built=false` so a later request can retry

---

## 8) Config Knobs

- `CODE_SEARCH_HYBRID=0|1` (default on for auto)
- `CODE_SEARCH_HYBRID_RRF_K` (default 60)
- `CODE_SEARCH_HYBRID_DENSE_K` (default 50)
- `CODE_SEARCH_HYBRID_SPARSE_K` (default 50)
- `CODE_SEARCH_HYBRID_AUTOBUILD=0|1` (default on)

---

## 9) Testing Plan

**Unit**
- Create/query FTS table in temp metadata DB
- Upsert/delete paths and verify counts
- RRF fusion ordering
- Query normalization (non-word chars)

**Integration**
- Sharded hybrid: two shards, confirm BM25 matches across shards
- Fallback path: FTS missing -> semantic results + build triggered

---

## 10) Risks / Mitigations

- **SQLite locking:** use short-lived sqlite3 connections with `busy_timeout`.
- **Query syntax errors:** normalize/sanitize FTS query strings.
- **Score double-boosting:** dampen post-ranking boosts when hybrid is used.

---

## 11) Implementation Touchpoints

- `search/indexer.py`: add FTS helpers and hook into add/remove
- `search/sharded_index_manager.py`: add hybrid search + per-shard FTS search
- `search/searcher.py`: route `auto` -> hybrid when available
- `mcp_server/code_search_server.py`: default `search_mode="auto"` to hybrid if available
- `README.md` / `CODEX.md`: document new env flags

---

## 12) Decision

Use **threaded background FTS builds** (not the index job system) to keep hybrid fallback fast and low-intrusion.
