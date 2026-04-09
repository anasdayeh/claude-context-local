# MCP Surface Cleanup Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate duplicated MCP tools into a single clean, production-ready tool surface (no `*_v2`), with accurate docs and fully functional parameters.

**Architecture:** Treat the MCP layer (`mcp_server/mcp_tools.py`) as the canonical API. Remove legacy/duplicate tool registrations, normalize responses to `{ok, meta, result, error}` across all tools, and expose full MCP surface discovery via `list_tools` (tools + resources + resource templates + prompts). Ensure search metadata and `auto_reindex` are real behaviors.

**Tech Stack:** Python, FastMCP (`mcp.server.fastmcp`), pytest.

### Task 1: Lock desired tool surface with tests

**Files:**
- Modify: `tests/unit/test_mcp_tools_list_tools.py`
- Modify: `tests/unit/test_mcp_tools_fts_status.py`
- Modify: `tests/unit/test_mcp_tools_search_code.py`
- Create: `tests/unit/test_mcp_tools_surface.py`

**Step 1: Write failing tests**
- Assert there are no registered `*_v2` tools.
- Assert canonical tools return `{ok, meta, result}` and do not emit raw lists.
- Assert `list_tools` includes:
  - `tools`
  - `resources`
  - `resource_templates`
  - `prompts`

**Step 2: Run tests to verify they fail**
- Run: `python -m pytest tests/unit/test_mcp_tools_surface.py -q`

### Task 2: Consolidate tool names and response shapes

**Files:**
- Modify: `mcp_server/mcp_tools.py`

**Step 1: Remove duplicate tool registrations**
- Remove `*_v2` variants and rename their behavior into the canonical tool name:
  - `list_projects` becomes the structured behavior (old v2).
  - `find_similar_code` becomes the structured behavior (old v2).
  - `get_index_status` accepts optional `project_path`.
  - `clear_index` accepts optional `project_path`.

**Step 2: Normalize all tools to a consistent response contract**
- Ensure all tools return dicts with:
  - `ok` boolean
  - `meta` dict (active project info)
  - `result` (tool-specific primary payload)
  - `error` and `error_info` when `ok=false`

### Task 3: Make `search_code` parameters real and meta accurate

**Files:**
- Modify: `mcp_server/mcp_tools.py`
- Modify: `mcp_server/code_search_server.py`
- Modify: `search/searcher.py`

**Step 1: Implement `auto_reindex`/`max_age_minutes`**
- When `auto_reindex=true`, detect staleness via `stats.json` timestamp.
- If stale, start a background job and include `meta.reindex_started=true` + job id.

**Step 2: Report `search_mode_used` accurately**
- Update `search/searcher.py` to record the mode actually used.
- Plumb that into `search_code` meta.

### Task 4: Update tool docs and surface discovery

**Files:**
- Modify: `mcp_server/strings.yaml`
- Modify: `README.md`
- Modify: `CODEX.md`

**Step 1: Make docs match real parameters**
- Replace `file_pattern` vs `file_patterns` mismatches.
- Add missing tool descriptions: `fts_status`, `get_chunk`, `get_stats`, etc.

**Step 2: Make `list_tools` comprehensive**
- Include resource templates (e.g. `codesearch://projects/{project_id}`).
- Include prompts.

### Task 5: Verify

**Files:**
- None

**Step 1: Run MCP-related tests**
- Run: `python -m pytest tests/unit/test_mcp_tools_*.py -q`

**Step 2: Run full unit suite**
- Run: `python tests/run_tests.py --unit`

