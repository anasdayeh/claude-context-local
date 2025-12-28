# Resume-From-Checkpoint Indexing Design

**Goal:** Allow large full-index runs to stop and resume without redoing already-processed files, while still reindexing files that changed since the last partial run.

## High-Level Strategy

Use a **project-local resume state** stored alongside the index in `projects/<id>/index/` to track which files were fully indexed during a full run. On restart, compare current file hashes (Merkle DAG) against the resume state to skip unchanged files and reindex only:

1) Files not yet processed  
2) Files whose content hash changed  
3) Files removed since the last run (for cleanup)

This keeps resume state tightly coupled to the index and avoids cross-project mixups.

## Storage Locations

- **Resume state:** `projects/<project_id>/index/resume.json`
- **Index stats:** `projects/<project_id>/index/stats.json` (existing)
- **Merkle snapshot:** unchanged, still saved on successful full index completion

## Resume State Format (Exact)

```json
{
  "version": 1,
  "project_path": "/abs/path/to/repo",
  "project_id": "md5-of-path",
  "status": "in_progress|ready|canceled|failed",
  "last_updated": "2025-12-28T12:34:56Z",
  "mode": "full",
  "files_total": 12345,
  "files_completed": 6789,
  "hashes": {
    "src/foo.py": "sha256...",
    "src/bar.ts": "sha256..."
  },
  "completed": [
    "src/foo.py",
    "src/bar.ts"
  ]
}
```

**Notes**
- `completed` can be a list (simpler) but may grow large; implement as a dict in memory and serialize to list to keep JSON simple.
- `hashes` maps **relative path → hash** from the Merkle DAG at time of processing.
- `status` is updated at checkpoints and on graceful cancel.

## In-Progress Detection

Resume is enabled by default whenever:
- `resume.json` exists AND `status == "in_progress"` AND `project_path` matches the target repo path.

Opt‑out via:
- Env: `CODE_SEARCH_RESUME=0`
- CLI: `--no-resume` on `scripts/index_repo.py`

## Full Indexing With Resume

### Start
1) Build Merkle DAG (current repo).
2) Load `resume.json` (if present and allowed).
3) Compute **pending set**:
   - `pending = all_supported_files - completed`
   - `changed = {f in completed | dag_hash[f] != resume_hash[f]}`
   - `pending += changed`
   - `removed = completed - all_supported_files` (for deletion cleanup)
4) If `removed` exists, call `remove_file_chunks` for each.
5) Begin chunking+embedding only for `pending` files.

### Checkpointing
Every N chunks or M seconds:
- `save_index()`
- Update `resume.json` with newly completed files + hashes
- `files_completed` and `last_updated`

### Completion
On successful full index:
- Save Merkle snapshot (existing behavior)
- Set `resume.json.status = "ready"` or remove it

## Incremental Indexing (After Full Index)

Incremental indexing still uses Merkle snapshots.
- If a **full index** completes successfully, snapshots are saved and `resume.json` marked ready.
- For **future runs**, incremental path uses change detection from snapshot (current behavior), not resume.json.

If a full run was **interrupted**, resume.json is used to finish the full run. This avoids mixing “partial snapshot” states with incremental logic.

## Handling Changed Files (Amended Scripts)

During resume:
- If file hash changes since resume state, it is reindexed.
- Old chunks for that file are removed before re-adding embeddings.

During incremental:
- Existing change detection handles modified/added/removed files.

## Safety Guarantees

- Resume only affects **full index** restarts.
- Incremental indexing continues to use snapshots (no behavioral change).
- Resume state is per-project and co-located with index to avoid mismatched history.

## Observability

Add log lines:
- “Resuming full index: X completed / Y total”
- “Skipping N unchanged files”
- “Reindexing M changed files”
- “Removing K deleted files”

