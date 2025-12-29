#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="/Users/anasdayeh/Downloads/CS-RAG"
BASE_DIR="/Users/anasdayeh/.local/share/claude-context-local"
STORAGE_ROOT="${CODE_SEARCH_STORAGE:-$HOME/code-search-storage}"

export CODE_SEARCH_STORAGE="$STORAGE_ROOT"

VERBOSE=1
JSON=1
for arg in "$@"; do
  case "$arg" in
    --verbose) VERBOSE=1 ;;
    --json) JSON=1 ;;
    --no-verbose) VERBOSE=0 ;;
    --no-json) JSON=0 ;;
    --help|-h)
      echo "Usage: $(basename "$0") [--verbose] [--json]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $(basename "$0") [--verbose|--no-verbose] [--json|--no-json]"
      exit 2
      ;;
  esac
done

proj_id=$(uv run --directory "$BASE_DIR" python - <<'PY'
from mcp_server.code_search_server import CodeSearchServer
print(CodeSearchServer().get_project_storage_dir("/Users/anasdayeh/Downloads/CS-RAG").name)
PY
)

non_sharded_db="$CODE_SEARCH_STORAGE/projects/$proj_id/index/metadata.db"
shards_root="$CODE_SEARCH_STORAGE/projects/$proj_id/index/shards"

db_paths=""
if [[ -d "$shards_root" ]]; then
  for p in "$shards_root"/shard_*/metadata.db; do
    if [[ -f "$p" ]]; then
      db_paths="${db_paths}${p}"$'\n'
    fi
  done
fi

if [[ -z "$db_paths" ]]; then
  if [[ -f "$non_sharded_db" ]]; then
    db_paths="$non_sharded_db"$'\n'
  else
    echo "No metadata.db found."
    echo "Tried:"
    echo "  $non_sharded_db"
    echo "  $shards_root/shard_*/metadata.db"
    exit 2
  fi
fi

DB_PATHS="$db_paths" VERBOSE="$VERBOSE" JSON="$JSON" REPO_PATH="$REPO_PATH" PROJECT_ID="$proj_id" \
uv run --directory "$BASE_DIR" python - <<'PY'
import json
import os
import sqlite3
from pathlib import Path

db_paths = [p for p in os.environ["DB_PATHS"].splitlines() if p.strip()]
verbose = os.environ.get("VERBOSE") == "1"
json_out = os.environ.get("JSON") == "1"
repo_path = os.environ.get("REPO_PATH")
project_id = os.environ.get("PROJECT_ID")

def stats_for_db(db_path: str) -> dict:
    info = {"db_path": db_path}
    p = Path(db_path)
    info["exists"] = p.exists()
    if p.exists():
        info["size_bytes"] = p.stat().st_size
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        info["chunks_fts_table"] = bool(row)
        if row:
            info["chunks_fts_rows"] = conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        else:
            info["chunks_fts_rows"] = 0
    except Exception as exc:
        info["error"] = str(exc)
    return info

def maybe_read_json(path: Path) -> dict:
    try:
        if path.exists():
            import json as _json
            return _json.loads(path.read_text())
    except Exception:
        return {}
    return {}

def size_bytes(path: Path) -> int | None:
    try:
        if path.exists():
            return path.stat().st_size
    except Exception:
        return None
    return None

results = [stats_for_db(p) for p in db_paths]

project_root = Path(db_paths[0]).parents[3] if db_paths else None
manifest_path = project_root / "index" / "manifest.json" if project_root else None
stats_path = project_root / "index" / "stats.json" if project_root else None

manifest = maybe_read_json(manifest_path) if manifest_path else {}
stats = maybe_read_json(stats_path) if stats_path else {}

shard_infos = []
for db in db_paths:
    p = Path(db)
    shard_dir = p.parent
    shard_stats = maybe_read_json(shard_dir / "stats.json")
    shard_infos.append(
        {
            "shard_dir": str(shard_dir),
            "stats": shard_stats,
            "code_index_bytes": size_bytes(shard_dir / "code.index"),
            "metadata_db_bytes": size_bytes(shard_dir / "metadata.db"),
            "file_map_db_bytes": size_bytes(shard_dir / "file_map.db"),
            "id_map_db_bytes": size_bytes(shard_dir / "id_map.db"),
            "training_sample_bytes": size_bytes(shard_dir / "training_sample.npy"),
        }
    )

payload = {
    "repo_path": repo_path,
    "project_id": project_id,
    "databases": results,
    "total_chunks_fts_rows": sum(r.get("chunks_fts_rows", 0) for r in results),
    "manifest_path": str(manifest_path) if manifest_path else None,
    "manifest": manifest,
    "stats_path": str(stats_path) if stats_path else None,
    "stats": stats,
    "shards": shard_infos,
}

if json_out:
    print(json.dumps(payload, indent=2))

print(f"Repo: {repo_path}")
print(f"Project id: {project_id}")
if manifest_path:
    print(f"Manifest: {manifest_path}")
if stats_path:
    print(f"Stats: {stats_path}")
for db in results:
    print(f"DB: {db['db_path']}")
    print(f"  exists: {db.get('exists')}")
    if "size_bytes" in db:
        print(f"  size_bytes: {db['size_bytes']}")
    if "error" in db:
        print(f"  error: {db['error']}")
    else:
        print(f"  chunks_fts_table: {db.get('chunks_fts_table')}")
        print(f"  chunks_fts_rows: {db.get('chunks_fts_rows')}")
    if verbose:
        pass
print(f"Total chunks_fts rows: {payload['total_chunks_fts_rows']}")

if verbose:
    for shard in shard_infos:
        print(f"Shard: {shard['shard_dir']}")
        if shard["stats"]:
            print(f"  shard stats keys: {sorted(list(shard['stats'].keys()))}")
        print(f"  code.index bytes: {shard['code_index_bytes']}")
        print(f"  metadata.db bytes: {shard['metadata_db_bytes']}")
        print(f"  file_map.db bytes: {shard['file_map_db_bytes']}")
        print(f"  id_map.db bytes: {shard['id_map_db_bytes']}")
        print(f"  training_sample.npy bytes: {shard['training_sample_bytes']}")
PY
