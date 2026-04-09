#!/usr/bin/env bash
set -euo pipefail

# This helper focuses on refreshing the local repo index when the working tree has edits.
# It mimics the environment setup from the other reindex scripts, reports changed files,
# and runs the MCP incremental index path without deleting existing index artifacts.

# Override with CS_RAG_REPO_PATH to target a specific checkout.
REPO_PATH="${CS_RAG_REPO_PATH:-}"
if [[ -z "$REPO_PATH" ]]; then
  for candidate in \
    "/Users/anasdayeh/Downloads/cs-rag" \
    "/Users/anasdayeh/Downloads/CS-RAG" \
    "$HOME/Downloads/cs-rag" \
    "$HOME/Downloads/CS-RAG"
  do
    if [[ -d "$candidate" ]]; then
      REPO_PATH="$candidate"
      break
    fi
  done
fi
if [[ -z "$REPO_PATH" ]]; then
  echo "Could not find cs-rag repo. Set CS_RAG_REPO_PATH=/absolute/path/to/cs-rag and retry." >&2
  exit 2
fi
PROJECT_NAME="CS-RAG"
BASE_DIR="/Users/anasdayeh/.local/share/claude-context-local"
STORAGE_ROOT="${CODE_SEARCH_STORAGE:-$HOME/.claude_code_search}"

export CODE_SEARCH_STORAGE="$STORAGE_ROOT"
export CODE_SEARCH_DATA_DIR="${CODE_SEARCH_DATA_DIR:-$CODE_SEARCH_STORAGE}"
export CODE_SEARCH_DEVICE="${CODE_SEARCH_DEVICE:-auto}"
export CODE_SEARCH_EMBED_BACKEND="${CODE_SEARCH_EMBED_BACKEND:-torch}"
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
export CODE_SEARCH_CHECKPOINT_CHUNKS="${CODE_SEARCH_CHECKPOINT_CHUNKS:-1500}"
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
export CODE_SEARCH_TORCH_NUM_THREADS="${CODE_SEARCH_TORCH_NUM_THREADS:-}"
export CODE_SEARCH_TORCH_INTEROP_THREADS="${CODE_SEARCH_TORCH_INTEROP_THREADS:-}"
export CODE_SEARCH_TORCH_BEFORE_FAISS="${CODE_SEARCH_TORCH_BEFORE_FAISS:-}"
export CODE_SEARCH_IGNORE_DIRS="${CODE_SEARCH_IGNORE_DIRS:-}"
export CODE_SEARCH_HYBRID="${CODE_SEARCH_HYBRID:-1}"
export CODE_SEARCH_HYBRID_RRF_K="${CODE_SEARCH_HYBRID_RRF_K:-}"
export CODE_SEARCH_HYBRID_DENSE_K="${CODE_SEARCH_HYBRID_DENSE_K:-}"
export CODE_SEARCH_HYBRID_SPARSE_K="${CODE_SEARCH_HYBRID_SPARSE_K:-}"
export CODE_SEARCH_HYBRID_AUTOBUILD="${CODE_SEARCH_HYBRID_AUTOBUILD:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-}"

changed_files=()
git_is_repo=0
if git -C "$REPO_PATH" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_is_repo=1
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    path="${line:3}"
    if [[ "$path" == *"->"* ]]; then
      path="${path##*-> }"
    fi
    changed_files+=("$path")
  done < <(git -C "$REPO_PATH" status --short)
fi

if [[ "$git_is_repo" -eq 1 ]] && [[ ${#changed_files[@]} -eq 0 ]]; then
  echo "No working-tree changes detected in $REPO_PATH; nothing to refresh."
  exit 0
fi

if [[ ${#changed_files[@]} -gt 0 ]]; then
  echo "Repo path: $REPO_PATH"
  echo "Detected ${#changed_files[@]} modified files (flagged via git status):"
  for file in "${changed_files[@]}"; do
    echo "  $file"
  done
else
  echo "Repo path: $REPO_PATH"
  echo "Working tree is not a git repository; running a full incremental index pass."
fi

proj_id=$(uv run --directory "$BASE_DIR" python - <<'PY'
from mcp_server.code_search_server import CodeSearchServer
print(CodeSearchServer().get_project_storage_dir("$REPO_PATH").name)
PY
)

LOG_FILE="$HOME/cs-rag-incremental-index.log"
uv run --directory "$BASE_DIR" \
  python "$BASE_DIR/scripts/index_repo.py" \
  "$REPO_PATH" \
  --project-name "$PROJECT_NAME" \
  --sharded \
  --incremental \
  --log-file "$LOG_FILE"

echo "Reindex finished for: $REPO_PATH"
echo "Log: $LOG_FILE"

echo "Project id: $proj_id"
echo "Expected metadata db (non-sharded): $CODE_SEARCH_STORAGE/projects/$proj_id/index/metadata.db"
echo "Expected metadata db (sharded): $CODE_SEARCH_STORAGE/projects/$proj_id/index/shards/shard_000/metadata.db"
echo "Expected stats.json: $CODE_SEARCH_STORAGE/projects/$proj_id/index/stats.json"
