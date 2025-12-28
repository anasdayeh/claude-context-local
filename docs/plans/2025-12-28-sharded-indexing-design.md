# Sharded Indexing + Offline CLI + Env Reuse Design

**Goal:** Keep semantic search fast on large repos while preventing memory blowups on 16 GB Macs, reduce redundant dependency downloads, and provide a reliable offline indexing path that produces the same artifacts as MCP.

---

## 1) Sharded Indexing (Unified, RAM‑bounded, concurrent)

**Core idea:** Store multiple FAISS shards per project and search them in parallel while respecting a dynamic memory budget (hard cap 13 GB). Shards are fully automatic; the system decides when to roll over to a new shard based on estimated resident memory.

**Storage layout**
```
projects/<project_hash>/
  index/
    shards/
      shard_000/
        code.index
        metadata.db
        id_map.db
        stats.json
      shard_001/...
    manifest.json
```
`manifest.json` tracks shard boundaries (chunk_id ranges), vector counts, index file sizes, embedding dim, and last build info.

**Indexing**
- `ShardedIndexManager` manages shard creation and selection.
- When current shard’s estimated RAM crosses a fraction of the budget (e.g., 15–20%), roll to a new shard.
- `IncrementalIndexer` writes new embeddings into the active shard; deletions update any shard containing the chunk_id.

**Search**
- Load as many shards as fit into the live memory budget (dynamic: `min(13 GB, available_mem*0.75)`).
- Search loaded shards concurrently; merge top‑k globally by score.
- Optional: use FAISS `IndexShards` (threaded) when shards are loaded, otherwise explicit parallel search + merge.

**Memory budgeter**
- Estimates per‑shard memory using index size + metadata size + vector count × dim × 4 bytes.
- Enforces a hard cap and LRU evicts shards if loading exceeds budget.

---

## 2) Environment & Storage Efficiency

**Principle:** One canonical MCP environment, reused by Codex and manual runs.

- Always run indexing via `uv run` from `~/.local/share/claude-context-local` to reuse the same `.venv`.
- `uv` cache is shared and thread‑safe; avoid creating multiple repo clones that each generate their own `.venv`.
- Add a warning if MCP is run from a non‑canonical clone to prevent duplicate downloads.

---

## 3) Offline Indexing CLI (Same Pipeline as MCP)

Provide a first‑class script that uses the **exact same pipeline** as the MCP:

```
uv run --directory ~/.local/share/claude-context-local \
  python scripts/index_repo.py /path/to/repo \
  --project-name MyRepo --incremental --auto-shards --verbose
```

**Requirements**
- Uses `CodeSearchServer` / `IncrementalIndexer` so artifacts are identical to MCP.
- Supports background mode, logging to a file, and resume/incremental reindex.
- Never blocks Codex; manual indexing can run overnight.

---

## Apple Silicon / macOS Considerations

- Prefer MPS when available; fall back to CPU for unsupported ops.
- Keep batch sizes adaptive (already implemented) and ensure a safe memory ceiling.
- Avoid experimental free‑threaded Python for now due to separate ABI/wheels.

---

## Risks / Mitigations

- **Score comparability across shards:** keep shard training consistent; avoid heavy compression until we add shared training.
- **Shard count explosion:** enforce auto‑sizing + min/max shard count heuristics.
- **Metadata consistency:** shard manifest is the single source of truth.

---

## Test Plan

- Unit: sharded index manager (create/rollover/load/evict), manifest integrity, merge correctness.
- Integration: index large synthetic repo with sharding on, verify search returns stable results.
- Regression: existing MCP indexing/search tests must still pass.
