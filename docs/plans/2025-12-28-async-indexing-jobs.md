# Async Indexing Jobs Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent large-repo indexing from hitting Codex’s ~1800s tool-call deadline by running indexing as a background job that can be polled, while keeping a single unified per-repo index.

**Architecture:** Add an in-process index job manager to `CodeSearchServer` that runs `IncrementalIndexer` in a dedicated background executor, records job state/progress, and exposes MCP tools to start/poll/cancel jobs. Keep `index_directory` synchronous for small repos, but auto-switch to background for large repos.

**Tech Stack:** Python, `concurrent.futures.ThreadPoolExecutor`, `threading.Event`, existing `IncrementalIndexer`, `FastMCP` tools.

---

### Task 1: Add background index job manager

**Files:**
- Modify: `mcp_server/code_search_server.py`
- Create (optional): `mcp_server/index_jobs.py`
- Modify (optional): `search/incremental_indexer.py`

**Step 1: Add job state model**
- Define a lightweight `IndexJob` record: `job_id`, `project_path`, `status`, `created_at`, `started_at`, `finished_at`, `last_message`, `events` (bounded), `result`, `error`, `cancel_event`.

**Step 2: Add job manager methods to server**
- `start_index_job(...) -> dict` (returns quickly, creates/returns existing active job for same path)
- `get_index_job_status(job_id=..., project_path=...) -> dict`
- `cancel_index_job(job_id) -> dict`

**Step 3: Run actual indexing in a dedicated executor**
- Use a per-server executor (max_workers=1 by default) to avoid contending with MCP tool execution.
- Ensure indexing updates `stats.json` with `status="indexing"` early, and `status="ready"` on completion (already exists).

**Step 4: Add cooperative cancellation**
- Thread-safe `cancel_event` checked periodically in `IncrementalIndexer` loops; abort gracefully with a clear error string and a final `save_index()` checkpoint.

---

### Task 2: Expose MCP tools for async indexing

**Files:**
- Modify: `mcp_server/mcp_tools.py`

**Step 1: Add tools**
- `start_index_directory(...)` → returns `job_id` and initial status
- `get_index_job_status(job_id=None, project_path=None)` → returns status + latest progress events
- `cancel_index_job(job_id)` → requests cancellation

**Step 2: Make `index_directory` auto-switch for large repos**
- Add a fast heuristic pre-scan (early exit) to detect “large repo”.
- If large: call `start_index_job(...)` and return immediately with `job_id` + guidance to poll.
- If small: keep current synchronous behavior (preserves tests and existing UX).

---

### Task 3: Update docs and skill guidance

**Files:**
- Modify: `README.md`
- Modify: `CODEX.md`
- Modify: `/Users/anasdayeh/.codex/skills/mcp-code-search/SKILL.md`

**Step 1: Document recommended workflow**
- For large repos: `start_index_directory` → poll `get_index_job_status` → `search_code`.
- Mention env toggles for forcing async/sync and tuning batch sizes.

---

### Task 4: Verify

**Files:**
- Test: `tests/` (existing suite)

**Step 1: Run integration suite**
- Run: `uv run python tests/run_tests.py --integration --verbose`
- Expected: PASS

**Step 2: Run unit suite (if integration passes quickly)**
- Run: `uv run python tests/run_tests.py --unit --verbose`
- Expected: PASS

