# MCP Semantic Search Bugfixes Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix two MCP-layer bugs discovered during end-to-end pressure testing: the broken `search://stats` resource and stale in-memory search behavior after `clear_index`.

**Architecture:** Make the smallest possible changes for backwards compatibility. Provide a compatibility attribute (`current_project_path`) on `CodeSearchServer` for resource handlers, and ensure `clear_index` resets in-memory state when clearing the active project.

**Tech Stack:** Python, MCP server (`mcp_server/*`), indexing/search layer (`search/*`).

### Task 1: Reproduce failures via MCP calls (no pytest)

**Files:**
- None

**Step 1: Reproduce `search://stats` failure**
- Read `search://stats` before fix.
- Expected: error containing `current_project_path` missing.

**Step 2: Reproduce `clear_index` stale in-memory search**
- Use `index_test_project()` to create a tiny index and auto-switch.
- Call `search_code("hello world")` (no `project_path`) and confirm it returns results.
- Call `clear_index()`.
- Call `search_code("hello world")` again (no `project_path`).
- Expected (desired): error like `No project selected`.
- Expected (current, buggy): returns results due to stale in-memory searcher.

### Task 2: Fix `search://stats` resource crash

**Files:**
- Modify: `mcp_server/code_search_server.py`

**Step 1: Add backwards-compatible `current_project_path` attribute/property**
- Map to existing `_current_project`.

**Step 2: Verify via MCP**
- Re-read `search://stats`.
- Expected: JSON stats payload (not an attribute error).

### Task 3: Fix stale in-memory search after `clear_index`

**Files:**
- Modify: `mcp_server/code_search_server.py`

**Step 1: Reset server in-memory state when clearing active project**
- If `target_path == self._current_project`: set `self._searcher = None`, `self._index_manager = None`, `self._current_project = None`.

**Step 2: Verify via MCP**
- Repeat Task 1, Step 2.
- Expected: post-clear `search_code` returns an error and does not return stale results.

### Task 4: Final verification sweep via MCP

**Files:**
- None

**Step 1: Smoke**
- `switch_project` to a known indexed project.
- `get_index_status`.
- `search_code` with and without filters.

