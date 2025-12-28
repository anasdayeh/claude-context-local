# Resume-From-Checkpoint Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add resume‑from‑checkpoint to full indexing so interrupted runs pick up where they left off and only reindex changed files.

**Architecture:** Store resume state in `projects/<id>/index/resume.json`. During full indexing, load the resume state if present; skip completed + unchanged files, reindex changed files, remove deleted files, and checkpoint progress. Incremental indexing remains snapshot‑based.

**Tech Stack:** Python, existing Merkle DAG, `IncrementalIndexer`, `SnapshotManager`.

---

### Task 1: Add resume state helpers

**Files:**
- Create: `search/resume_state.py`
- Modify: `search/incremental_indexer.py`

**Step 1: Define resume state model + I/O**

Create `ResumeState` with fields:
- `project_path`, `project_id`, `status`, `last_updated`, `files_total`, `files_completed`
- `hashes` dict (`relative_path -> hash`)
- `completed` set

Implement:
- `load_resume_state(index_dir) -> ResumeState | None`
- `save_resume_state(index_dir, state)`
- `clear_resume_state(index_dir)`

**Step 2: Wire resume paths**

Resume file should live at:
`<project_dir>/index/resume.json`

---

### Task 2: Resume-aware full indexing

**Files:**
- Modify: `search/incremental_indexer.py`

**Step 1: Add resume options**

Add params to `incremental_index(...)`:
- `resume: bool = True`
- `resume_state_path` derived from index storage dir

**Step 2: Compute pending/changed/removed**

In `_full_index(...)`:
- Build Merkle DAG
- Load resume state if `resume=True` and status is `in_progress`
- Compute:
  - `pending = all_supported_files - completed`
  - `changed = completed where dag_hash != resume_hash`
  - `removed = completed - all_supported_files`
- Remove chunks for `removed`
- Reindex `changed`

**Step 3: Checkpoint resume state**

At each checkpoint:
- `save_index()`
- `save_resume_state()` with new completed files + hashes

**Step 4: Completion**

On success:
- Save snapshot (existing behavior)
- `resume.status = ready` or delete resume file

On cancel/failure:
- `resume.status = canceled/failed`

---

### Task 3: CLI + env toggles

**Files:**
- Modify: `scripts/index_repo.py`
- Modify: `README.md`
- Modify: `CODEX.md`

**Step 1: CLI**

Add `--no-resume` to disable resume.

**Step 2: Env**

Add `CODE_SEARCH_RESUME=0` opt‑out.

---

### Task 4: Tests

**Files:**
- Create: `tests/unit/test_resume_state.py`
- Create: `tests/integration/test_resume_full_index.py`

**Step 1: Unit test resume state I/O**

Validate:
- load/save/clear
- status transitions
- hash map + completed set

**Step 2: Integration test resume**

Scenario:
- Start full index, simulate interrupt after N files
- Resume and ensure it only processes remaining/changed files

---

### Task 5: Verification

**Step 1: Run unit tests**

Run: `uv run python tests/run_tests.py --unit --verbose`
Expected: PASS

**Step 2: Run integration tests**

Run: `uv run python tests/run_tests.py --integration --verbose`
Expected: PASS

