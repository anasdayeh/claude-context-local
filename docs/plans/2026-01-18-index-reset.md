# Index Reset & Reindex Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reset all indexed projects safely and reindex key repos sequentially without overwhelming system resources.

**Architecture:** Add a clear-all script that removes only project index artifacts under the configured storage root, then add a sequential reindex script that loops through a curated list of project paths and runs the existing MCP indexer synchronously. Update existing single-project reindex scripts to align with the shared storage root and non-background behavior.

**Tech Stack:** Bash scripts, Python (existing `scripts/index_repo.py`), Codex MCP storage layout.

---

### Task 1: Add a safe clear-all script for project indexes

**Files:**
- Create: `scripts/clear_all_projects.sh`

**Step 1: Write the failing test**

```bash
# No automated tests exist for scripts. We'll use `bash -n` as a syntax check.
```

**Step 2: Run test to verify it fails**

Run: `bash -n scripts/clear_all_projects.sh`
Expected: FAIL with "No such file or directory"

**Step 3: Write minimal implementation**

```bash
#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${CODE_SEARCH_STORAGE:-$HOME/.claude_code_search}"
PROJECTS_DIR="$STORAGE_ROOT/projects"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h)
      echo "Usage: $(basename "$0") [--dry-run]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $(basename "$0") [--dry-run]"
      exit 2
      ;;
  esac
done

if [[ ! -d "$PROJECTS_DIR" ]]; then
  echo "No projects directory found: $PROJECTS_DIR"
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run. Would remove:" 
  find "$PROJECTS_DIR" -mindepth 1 -maxdepth 1 -print
  exit 0
fi

echo "Removing all project indexes under: $PROJECTS_DIR"
rm -rf "$PROJECTS_DIR"/*

# Recreate the directory so the app can write immediately
mkdir -p "$PROJECTS_DIR"

echo "Done. Models/cache preserved under: $STORAGE_ROOT"
```

**Step 4: Run test to verify it passes**

Run: `bash -n scripts/clear_all_projects.sh`
Expected: exit 0 with no output

**Step 5: Commit**

```bash
git add scripts/clear_all_projects.sh
git commit -m "feat: add safe clear-all projects script"
```

### Task 2: Add a sequential reindex script for key projects

**Files:**
- Create: `scripts/reindex_key_projects.sh`

**Step 1: Write the failing test**

```bash
# No automated tests exist for scripts. We'll use `bash -n` as a syntax check.
```

**Step 2: Run test to verify it fails**

Run: `bash -n scripts/reindex_key_projects.sh`
Expected: FAIL with "No such file or directory"

**Step 3: Write minimal implementation**

```bash
#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/Users/anasdayeh/.local/share/claude-context-local"
STORAGE_ROOT="${CODE_SEARCH_STORAGE:-$HOME/.claude_code_search}"

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

for i in "${!PROJECT_PATHS[@]}"; do
  repo_path="${PROJECT_PATHS[$i]}"
  project_name="${PROJECT_NAMES[$i]}"

  if [[ ! -d "$repo_path" ]]; then
    echo "Skipping missing path: $repo_path"
    continue
  fi

  proj_id=$(uv run --directory "$BASE_DIR" python - <<PY
from mcp_server.code_search_server import CodeSearchServer
print(CodeSearchServer().get_project_storage_dir("$repo_path").name)
PY
  )

  echo "Clearing prior index for $project_name ($proj_id)"
  rm -rf "$CODE_SEARCH_STORAGE/projects/$proj_id"

  echo "Reindexing $project_name: $repo_path"
  uv run --directory "$BASE_DIR" \
    python "$BASE_DIR/scripts/index_repo.py" \
    "$repo_path" \
    --project-name "$project_name" \
    --sharded \
    --log-file "$HOME/${project_name}-index.log"

  echo "Finished: $project_name"
  echo "Log: $HOME/${project_name}-index.log"
  echo ""
done

echo "All requested projects completed."
```

**Step 4: Run test to verify it passes**

Run: `bash -n scripts/reindex_key_projects.sh`
Expected: exit 0 with no output

**Step 5: Commit**

```bash
git add scripts/reindex_key_projects.sh
git commit -m "feat: add sequential reindex script for key projects"
```

### Task 3: Update existing single-project reindex scripts to align defaults

**Files:**
- Modify: `scripts/reindex_cs_rag.sh`
- Modify: `scripts/reindex_claude-context-local.sh`

**Step 1: Write the failing test**

```bash
# No automated tests exist for scripts. We'll use `bash -n` as a syntax check.
```

**Step 2: Run test to verify it fails**

Run: `bash -n scripts/reindex_cs_rag.sh scripts/reindex_claude-context-local.sh`
Expected: FAIL only if syntax errors exist (currently should pass)

**Step 3: Write minimal implementation**

```bash
# Update STORAGE_ROOT defaults to ~/.claude_code_search
# Remove background nohup usage so it runs sequentially in foreground
# Keep the per-project cleanup logic intact
```

**Step 4: Run test to verify it passes**

Run: `bash -n scripts/reindex_cs_rag.sh scripts/reindex_claude-context-local.sh`
Expected: exit 0 with no output

**Step 5: Commit**

```bash
git add scripts/reindex_cs_rag.sh scripts/reindex_claude-context-local.sh
git commit -m "chore: align reindex scripts with storage defaults and foreground runs"
```

### Task 4: Document the safe deletion locations and usage

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

```bash
# No automated tests exist for README updates.
```

**Step 2: Run test to verify it fails**

Run: `rg -n "clear_all_projects" README.md`
Expected: no matches

**Step 3: Write minimal implementation**

```markdown
Add a short section describing that deleting `$CODE_SEARCH_STORAGE/projects/*` resets indexes while preserving models/cache, and point to `scripts/clear_all_projects.sh` and `scripts/reindex_key_projects.sh`.
```

**Step 4: Run test to verify it passes**

Run: `rg -n "clear_all_projects" README.md`
Expected: matches new section

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add reset/reindex guidance"
```
