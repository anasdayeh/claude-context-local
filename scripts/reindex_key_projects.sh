#!/usr/bin/env bash
set -euo pipefail

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

PROJECT_PATHS=(
  "/Users/anasdayeh/Downloads/CS-RAG"
  "/Users/anasdayeh/.local/share/claude-context-local"
  "/Users/anasdayeh/Library/Mobile Documents/com~apple~CloudDocs/Developer projects/instagram-follower-tracker-v9-audited"
  "/Users/anasdayeh/czkawka-opt/czkawka"
  "/Users/anasdayeh/Library/Mobile Documents/com~apple~CloudDocs/Downloads/doccrawl_project"
  "/Users/anasdayeh/Downloads/ADS_Website"
  "/Users/anasdayeh/Library/Mobile Documents/com~apple~CloudDocs/Downloads/pippa---ai-practice-assistant (3)"
)

PROJECT_NAMES=(
  "CS-RAG"
  "claude-context-local"
  "instagram-follower-tracker-v9-audited"
  "czkawka"
  "doccrawl_project"
  "ADS_Website"
  "pippa-ai-practice-assistant"
)

if [[ "${#PROJECT_PATHS[@]}" -ne "${#PROJECT_NAMES[@]}" ]]; then
  echo "PROJECT_PATHS and PROJECT_NAMES length mismatch"
  exit 2
fi

projects_dir="$CODE_SEARCH_STORAGE/projects"
if [[ -d "$projects_dir" ]]; then
  echo "Clearing all existing project indexes under: $projects_dir"
  rm -rf "$projects_dir"/*
else
  mkdir -p "$projects_dir"
fi

echo "Starting sequential reindex (foreground)"

to_safe_name() {
  printf "%s" "$1" | tr -cs '[:alnum:]' '_' | sed 's/^_\+//;s/_\+$//'
}

for i in "${!PROJECT_PATHS[@]}"; do
  repo_path="${PROJECT_PATHS[$i]}"
  project_name="${PROJECT_NAMES[$i]}"

  if [[ ! -d "$repo_path" ]]; then
    echo "Skipping missing path: $repo_path"
    continue
  fi

  safe_name=$(to_safe_name "$project_name")

  proj_id=$(uv run --directory "$BASE_DIR" python - <<PY
from mcp_server.code_search_server import CodeSearchServer
print(CodeSearchServer().get_project_storage_dir("$repo_path").name)
PY
  )

  echo "Reindexing $project_name ($proj_id): $repo_path"
  uv run --directory "$BASE_DIR" \
    python "$BASE_DIR/scripts/index_repo.py" \
    "$repo_path" \
    --project-name "$project_name" \
    --sharded \
    --log-file "$HOME/${safe_name}-index.log"

  echo "Finished: $project_name"
  echo "Log: $HOME/${safe_name}-index.log"
  echo ""
done

echo "All requested projects completed."
