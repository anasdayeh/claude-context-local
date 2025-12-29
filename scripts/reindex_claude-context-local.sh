#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="/Users/anasdayeh/.local/share/claude-context-local"
BASE_DIR="/Users/anasdayeh/.local/share/claude-context-local"
STORAGE_ROOT="${CODE_SEARCH_STORAGE:-$HOME/code-search-storage}"

export CODE_SEARCH_STORAGE="$STORAGE_ROOT"
export CODE_SEARCH_DATA_DIR="${CODE_SEARCH_DATA_DIR:-$CODE_SEARCH_STORAGE}"
export CODE_SEARCH_DEVICE="${CODE_SEARCH_DEVICE:-auto}"
export CODE_SEARCH_PRELOAD_MODEL="${CODE_SEARCH_PRELOAD_MODEL:-0}"
export CODE_SEARCH_LOG_LEVEL="${CODE_SEARCH_LOG_LEVEL:-INFO}"
export CODE_SEARCH_LOG_FILE="${CODE_SEARCH_LOG_FILE:-}"
export CODE_SEARCH_CHUNK_BATCH_SIZE="${CODE_SEARCH_CHUNK_BATCH_SIZE:-100}"
export CODE_SEARCH_BATCH_SIZE="${CODE_SEARCH_BATCH_SIZE:-}"
export CODE_SEARCH_EMBED_BATCH_SIZE="${CODE_SEARCH_EMBED_BATCH_SIZE:-}"
export CODE_SEARCH_INCLUDE_CONTEXT="${CODE_SEARCH_INCLUDE_CONTEXT:-1}"
export CODE_SEARCH_DISK_WARN_GB="${CODE_SEARCH_DISK_WARN_GB:-}"
export CODE_SEARCH_LARGE_FILE_MB="${CODE_SEARCH_LARGE_FILE_MB:-}"
export CODE_SEARCH_PROGRESS_EVERY_FILES="${CODE_SEARCH_PROGRESS_EVERY_FILES:-50}"
export CODE_SEARCH_RESUME="${CODE_SEARCH_RESUME:-0}"
export CODE_SEARCH_ASYNC_INDEX="${CODE_SEARCH_ASYNC_INDEX:-}"
export CODE_SEARCH_SYNC_INDEX="${CODE_SEARCH_SYNC_INDEX:-}"
export CODE_SEARCH_ASYNC_FILE_THRESHOLD="${CODE_SEARCH_ASYNC_FILE_THRESHOLD:-}"
export CODE_SEARCH_ASYNC_SCAN_SECONDS="${CODE_SEARCH_ASYNC_SCAN_SECONDS:-}"
export CODE_SEARCH_INDEX_WORKERS="${CODE_SEARCH_INDEX_WORKERS:-}"
export CODE_SEARCH_JOB_EVENT_BUFFER="${CODE_SEARCH_JOB_EVENT_BUFFER:-}"
export CODE_SEARCH_SHARDED_INDEX="${CODE_SEARCH_SHARDED_INDEX:-1}"
export CODE_SEARCH_SHARD_TARGET_BYTES="${CODE_SEARCH_SHARD_TARGET_BYTES:-}"
export CODE_SEARCH_SHARD_MEMORY_CAP_GB="${CODE_SEARCH_SHARD_MEMORY_CAP_GB:-}"
export CODE_SEARCH_TRAIN_SAMPLE_MAX="${CODE_SEARCH_TRAIN_SAMPLE_MAX:-}"
export CODE_SEARCH_TORCH_BEFORE_FAISS="${CODE_SEARCH_TORCH_BEFORE_FAISS:-}"
export CODE_SEARCH_IGNORE_DIRS="${CODE_SEARCH_IGNORE_DIRS:-}"
export CODE_SEARCH_HYBRID="${CODE_SEARCH_HYBRID:-1}"
export CODE_SEARCH_HYBRID_RRF_K="${CODE_SEARCH_HYBRID_RRF_K:-}"
export CODE_SEARCH_HYBRID_DENSE_K="${CODE_SEARCH_HYBRID_DENSE_K:-}"
export CODE_SEARCH_HYBRID_SPARSE_K="${CODE_SEARCH_HYBRID_SPARSE_K:-}"
export CODE_SEARCH_HYBRID_AUTOBUILD="${CODE_SEARCH_HYBRID_AUTOBUILD:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-}"

proj_id=$(uv run --directory "$BASE_DIR" python - <<'PY'
from mcp_server.code_search_server import CodeSearchServer
print(CodeSearchServer().get_project_storage_dir("/Users/anasdayeh/.local/share/claude-context-local").name)
PY
)

rm -rf "$CODE_SEARCH_STORAGE/projects/$proj_id"

nohup uv run --directory "$BASE_DIR" \
  python "$BASE_DIR/scripts/index_repo.py" \
  "$REPO_PATH" \
  --project-name claude-context-local \
  --sharded \
  --background \
  --log-file "$HOME/claude-context-local-index.log" \
  > "$HOME/claude-context-local-index.out" 2>&1 &

echo "Started background reindex for: $REPO_PATH"
echo "Log: $HOME/claude-context-local-index.log"
echo "Output: $HOME/claude-context-local-index.out"

echo "FTS check command (run after job completes):"
echo "/Users/anasdayeh/.local/share/claude-context-local/scripts/check_claude_context_local_fts.sh"

echo ""
echo "Project id: $proj_id"
echo "Expected metadata db (non-sharded): $CODE_SEARCH_STORAGE/projects/$proj_id/index/metadata.db"
echo "Expected metadata db (sharded): $CODE_SEARCH_STORAGE/projects/$proj_id/index/shards/shard_000/metadata.db"
echo "Expected stats.json: $CODE_SEARCH_STORAGE/projects/$proj_id/index/stats.json"
