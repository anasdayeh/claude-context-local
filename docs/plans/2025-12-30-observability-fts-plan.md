# Observability & Diagnostics Enhancements Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fill out MCP observability, structured logging, query-time diagnostics, and helper tooling so every FTS/manifest/coverage signal is surfaced reliably without breaking existing interfaces.

**Architecture:** Build on the chunk→index→search pipeline by (1) surfacing richer metadata in MCP tools including a dedicated `fts_status` endpoint, (2) instrumenting the `CodeIndexManager` + FTS lifecycle for chunk counts/coverage with structured logs, (3) letting `IntelligentSearcher` record FTS readiness/coverage and hybrid candidate counts, and (4) updating the helper script/docs so operators can verify a project end-to-end without guessing.

**Tech Stack:** Python 3.13, sqlite3/SqliteDict, FAISS (flat + sharded), MCP FastMCP tools, bash helper scripts, built-in logging, pytest.

---

### Task 1: Rich MCP metadata + `fts_status` tool
**Files:**
- Modify: `mcp_server/mcp_tools.py` (meta helpers + `search_code` output)
- Create: `tests/unit/test_mcp_tools_fts_status.py`
- Modify: `tests/unit/test_mcp_tools_search_code.py`

**Step 1: Extend tests**
- Update `tests/unit/test_mcp_tools_search_code.py` to assert `search_code` meta carries the new fields (`manifest_project_path`, `manifest_index_bytes`, `fts_status`, `fts_coverage_pct`, `index_last_indexed_source`).
- Create a failing test for the new `fts_status` tool that covers both stats-present and stats-missing scenarios (mocking file presence via temporary directories).

**Step 2: Augment meta helpers**
- Centralize manifest/stats reads so `_base_meta` can already include manifest path, shard count, and fallback timestamps (per plan). Add a helper that returns `fts_status` payload summarizing `chunks_fts_rows`, `total_chunks`, `coverage_pct`, `last_indexed`, and `last_indexed_source`.

**Step 3: Enhance `search_code` meta**
- After coercing results, call the helper to attach `meta['fts_status']`, `meta['stats_storage_size']`, `meta['manifest_project_path']`, `meta['manifest_index_bytes']`, and default `meta['active_project_id']`, ensuring `result_value` and meta follow the desired response shape (ok/result/meta/error structure).

**Step 4: Implement `fts_status` MCP tool**
- Expose a new tool that reads manifest/stats/FTS row counts for a given `project_path` (defaulting to active project) and returns the structured payload described in Step 2 (including warnings list and active project info). Reuse the helper so stats stay in sync.

**Step 5: Run tests**
- Execute `pytest tests/unit/test_mcp_tools_search_code.py tests/unit/test_mcp_tools_fts_status.py` and ensure the new tool is registered (`list_tools` should surface it). Confirm updates appear in MCP docs if needed.

### Task 2: Structured chunk/FTS logging
**Files:**
- Modify: `search/indexer.py` (log statements around `add_embeddings`, `fts_upsert`, `build_fts_from_metadata`, `_update_stats`)
- Create: `tests/unit/test_indexer_logging.py`

**Step 1: Identify key log points**
- Instrument `add_embeddings` to log `chunk_count`, chunk_type distribution (maybe top few), and `fts_candidate_count` before FTS insert.
- In `fts_upsert`/`build_fts_from_metadata`, log warnings when missing content and emit structured info (project_id, chunk path) for inserted counts.

**Step 2: Capture coverage metrics**
- After `_update_stats` runs, log the updated `fts_rows` vs `total_chunks` ratio and any sanity warnings (include hints if coverage < 50%).

**Step 3: Add logging test**
- Create `tests/unit/test_indexer_logging.py` that injects a `logging` capture handler, runs `add_embeddings` with deterministic `EmbeddingResult`s, and asserts the handler saw messages containing `fts_candidates=` and `chunks_added=`.

**Step 4: Run `pytest tests/unit/test_indexer_logging.py`** to exercise logs and ensure instrumentation doesn't raise exceptions.

### Task 3: Query-time readiness+hybrid metrics
**Files:**
- Modify: `search/searcher.py` (FTS readiness/logging), optionally `search/hybrid.py` (return candidate counts)
- Modify: `tests/integration/test_hybrid_sharded_search.py`

**Step 1: Log readiness**
- In `_fts_ready`/`_ensure_fts_async`, log the current `fts_row_count`, `stats.total_chunks`, `fts_coverage_pct`, and whether `_fts_building` was triggered. Include warnings when coverage stays below 30% after autoscalers.

**Step 2: Hybrid metrics**
- When running `_hybrid_search`, count how many sparse IDs came from FTS and how many dense from FAISS, log the ratio along with `rrf_k`. Propagate these counters into context info so tests can assert them.

**Step 3: Add integration test hooks**
- Extend `tests/integration/test_hybrid_sharded_search.py` (or create a new unit test) to mock low-FTS coverage, call `searcher.search(...)`, and assert the appropriate warning log and fallback (the test can use `logging` capture to confirm the message includes the computed ratio).

**Step 4: Verify logs**
- Run `pytest tests/integration/test_hybrid_sharded_search.py` (and any new unit test) ensuring hybrid fallback still returns results even when FTS coverage=0.

### Task 4: Script & docs updates for operators
**Files:**
- Modify: `scripts/check_cs_rag_fts.sh`
- Modify: `README.md`, `CODEX.md`, `AGENTS.md` (if needed) to describe new helper output and metadata fields

**Step 1: Enhance script output**
- Remove reliance on `--json/--verbose` flags (default to both). Always print human-readable summaries even when JSON suppressed. Print manifest story, stats summary, `FTS coverage ratio = x%` (using `total_chunks` and `total_chunks_fts_rows`).
- Append the computed `fts_status` payload from Task 1 so the script can be used as a quick check.

**Step 2: Document the workflow**
- Update `README.md`/`CODEX.md` to mention the new `check_cs_rag_fts.sh` (full command) and `fts_status` tool, noting the meta fields and coverage ratio. Mention the new structured logs (chunk counts, warnings) operators can grep for.

**Step 3: Verify output & docs**
- Run the script manually (without reindexing) to ensure it prints the manifest/stats/coverage and matches the new `fts_status` shape. Add a short doc snippet describing the recommended command and the meaning of the coverage metric.

### Task 5 (per user request): Wrap up verification
**Files:** N/A (cross-check outputs)

**Step 1:** After implementing the tasks, rerun the helper script and note the coverage ratio along with the `fts_status` tool response.

**Step 2:** Confirm MCP `list_tools` shows `fts_status` and the structured `search_code` responses still follow the {ok/meta/result/error} shape.

**Step 3:** Provide a short summary referencing the tests run (pytest modules, script run) and any outstanding follow-ups before declaring the work complete.
