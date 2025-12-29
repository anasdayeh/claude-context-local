# Future-Proof Index & Search Reliability Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the indexing/search pipeline fully observable, metadata-consistent, and prepared for eventual 100% FTS coverage while preserving backward-compatible MCP interfaces.

**Architecture:** Layer changes around the existing chunk→index→search stack: (1) upgrade metadata/stat maintenance in `CodeIndexManager`/`ShardedIndexManager`, (2) enrich chunking/embedding to record usable `content` for every chunk type, (3) add diagnostics/observability around FTS/hybrid search, (4) expose state through the new `check_cs_rag_fts.sh` script and MCP tools so operators can validate health without reindexing.

**Tech Stack:** Python 3.13, FAISS (IndexFlat/IP + sharded manager), sqlite3 + SqliteDict, tree-sitter chunker, MCP server tools, custom bash helper scripts.

---

### Task 1: Manifest + stats consistency
**Files:**
- Modify: `search/sharded_index_manager.py:200-360`
- Modify: `search/indexer.py:500-760`
- Test: `tests/unit/test_sharded_manifest_repair.py`, `tests/unit/test_sharded_stats_sanity_warning.py`

1. Read the manifest/stats lifecycle (search existing tests) and write failing assertions that they pick up shard stats: e.g., after inserting mock stats with non-zero embedding_dim/storage_size show `manifest.json` reflects real embedding_dim/project_path and `stats.json` records storage_size every rebuild.
2. Update `get_stats`/`_update_stats` to always refresh storage_size (already done) and ensure `_compute_root_stats` caches manifest metadata (embedding_dim/index_type/project_path). Add test verifying manifest rewritten with values from shard stats.
3. Run `pytest tests/unit/test_sharded_*` to confirm no regressions and add new test verifying storage_size present when stats file prepopulated.
4. After tests pass, rerun `uv run python scripts/index_repo.py ... --repair` to refresh actual manifest/stats, then rerun `check_cs_rag_fts.sh` to show updated manifest metadata.

### Task 2: Guarantee chunk content for every chunk type (FTS coverage foundation)
**Files:**
- Modify: `chunking/multi_language_chunker.py` (ensure content snippet always produced even for tree-sitter nodes)
- Modify: `chunking/base_chunker.py` + per-language chunkers to normalize `content` and fallback to source range text when docstrings empty
- Modify: `search/indexer.py` around `fts_entries` collection to optionally synthesize `content` (e.g., join tokens or fallback to file segment) so FTS gets something per chunk.
- Test: `tests/unit/test_chunk_content_presence.py` (new) – ensure every chunk_type yields non-empty `metadata.content` `content_preview`.

1. Inspect existing chunk harvesting to understand when `content` is empty; add instrumentation to log chunk_type/content lengths, then define normalized function (maybe `ChunkMetadata.ensure_content()`).
2. Update chunkers so every chunk record saves `content` even if just `name` + path; for text/chunk fallback, derive substring. Add tests using tree-sitter to assert `content` exists for representative chunk types (classes, methods, text blocks) by mocking chunker output.
3. Extend `fts_entries` to call this helper before `fts_upsert`. Validate by mocking chunk with trimmed content and ensuring DB insert occurs (maybe using sqlite in memory).
4. Run targeted tests and, if possible, rerun `scripts/check_cs_rag_fts.sh` to check `chunks_fts_rows` equals `stats.total_chunks` once implemented; if not yet, document expectation and metrics to monitor.

### Task 3: Observability + diagnostics coverage
**Files:**
- Modify: `mcp_server/mcp_tools.py` — enhance existing indexing/logging meta (meta/k fields) to include `manifest`, `stats`, `FTS coverage`; add another tool to query FTS stats directly.
- Modify: `scripts/check_cs_rag_fts.sh` — already outputs multi-format data; extend to log manifest/stats even when running w/out JSON; include summary of `FTS coverage ratio = total_chunks_fts_rows / stats.total_chunks`.
- Test: `tests/unit/test_mcp_tools_search_code.py` (add assertions that meta has new fields); add tests for a new MCP tool returning FTS metrics.

1. Use MCP tool search to confirm `search_code` currently returns new metadata; add additional meta entries such as `manifest_version`, `manifest_index_bytes`, `stats_storage_size`, `fts_coverage_pct`. Update `_base_meta` helper to attach these fields via `_augment_dict_response` without breaking clients.
2. Create new MCP tool `query_fts_status(project_path)` that safely reads manifest/stats and returns coverage metrics and shard sizes; add tests verifying the tool handles missing stats gracefully.
3. Update `scripts/check_cs_rag_fts.sh` to print `FTS coverage ratio` and to show manifest/stats JSON summaries even outside `--json` (maybe trimmed). Log the command output location for future towers.
4. Run targeted tests (`pytest tests/unit/test_mcp_tools_search_code.py`, new test file) to ensure new fields are generated.

### Task 4: Chunk/embedding logging & monitoring
**Files:**
- Modify: `embeddings/embedder.py` (adding banch metrics) or chunking to log event sizes/latency for large files.
- Modify: `search/indexer.py` to emit structured logs on reindex cycles (chunks processed, FTS inserted count) using existing logging framework.
- Test: `tests/unit/test_indexer_logging.py` (mock logger to capture messages containing chunk counts/FTS updates). Possibly integrate with new helper script to surface these metrics.

1. Identify failure modes where chunk content missing or FTS insert fails; add logs to `fts_upsert` to warn when content is missing, and to `build_fts_from_metadata` to report counts.
2. Introduce helper `LOGGER.debug` statements in `indexer.add_embeddings` summarizing chunk counts and `fts_entries` length; ensure log includes `project_id` and `shard_id`. Add unit test that configures `logging` to capture message and asserts string patterns.
3. Document new log fields in README/ plan for easy grepping by operators; mention log file location from `index_repo` script.

### Task 5: Query-time validations and fallback logic
**Files:**
- Modify: `search/searcher.py` to log `fts_ready` status per query, to ensure fallback to hybrid search when FTS not ready or coverage low.
- Modify: `search/hybrid.py` (if needed) to expose metrics such as FTS candidate counts or RRF weights.
- Test: `tests/integration/test_hybrid_sharded_search.py` to assert requests that rely on FTS degrade gracefully when coverage incomplete.

1. In `searcher._ensure_fts_async`, log warning if `fts_ready()` returns False even after background build; include `chunks_fts_rows` from stats for context. Add measurement hooking to record `fts_ready` latencies (maybe store in `hybrid_metrics` object for future dashboards).
2. Extend hybrid search to record how many sparse candidates came from BTC vs FTS and log that ratio. Add tests verifying hybrid search returns reasonable scoring when FTS coverage is low.
3. Update `scripts/check_cs_rag_fts.sh` to read `stats.top_tags` plus maybe RRF stats from hybrid module (if stored) and surface them under `shards` output.

### Task 6: Documentation + rollout
**Files:**
- Modify: `docs/plans/2025-12-29-future-proof-index-plan.md` (this file) with plan summary (done).
- Modify: `README.md`/`CODEX.md` to mention new `check_cs_rag_fts.sh` script, new MCP metadata fields, and how to interpret FTS coverage.
- Test: N/A.

1. After code changes are complete, update docs to describe new `--json` output, new MCP meta fields, and how to verify manifest/stats health.
2. Provide in README the recommended command flow to run the script and interpret FTS coverage ratio.

---

Plan saved as `docs/plans/2025-12-29-future-proof-index-plan.md`. Execution options:
1. **Subagent-driven (this session)** – I use superpowers:subagent-driven-development per task.
2. **Parallel executing-plans session** – spin a new session with superpowers:executing-plans.
Which approach should I use?