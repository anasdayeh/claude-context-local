# CLAUDE.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

Codex Embedding Search is an intelligent code search system that uses Google's EmbeddingGemma model and AST-based chunking to provide semantic search capabilities for Python codebases, integrated with Codex via MCP (Model Context Protocol).

## Key Commands

### Development Setup

```bash
# Install dependencies
uv sync

# Install in development mode
uv sync --dev
```

### Testing

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --unit          # Unit tests only
python tests/run_tests.py --integration   # Integration tests only
python tests/run_tests.py --chunking      # Chunking tests only
python tests/run_tests.py --embeddings    # Embedding tests only
python tests/run_tests.py --search        # Search tests only
python tests/run_tests.py --mcp           # MCP server tests only

# Run tests with coverage
python tests/run_tests.py --coverage

# Run tests with verbose output
python tests/run_tests.py --verbose

# Run specific test files or patterns
python tests/run_tests.py unit/test_chunking.py

# Alternative: Direct pytest usage
python -m pytest                          # All tests
python -m pytest -m "unit"               # Unit tests only
python -m pytest -m "not slow"           # Skip slow tests
python -m pytest tests/unit/test_chunking.py -v  # Single test file
```

### Indexing & Usage

```bash
# Index a Python codebase
./scripts/index_codebase.py /path/to/project

# Index a repo using the MCP pipeline (recommended for large repos)
uv run --directory ~/.local/share/claude-context-local \
  python scripts/index_repo.py /path/to/repo \
  --project-name MyRepo \
  --sharded \
  --log-file ~/code_search_index.log

# Index with custom storage location
./scripts/index_codebase.py /path/to/project --storage-dir /custom/location

# Clear existing index and reindex
./scripts/index_codebase.py /path/to/project --clear

# Enable verbose logging
./scripts/index_codebase.py /path/to/project --verbose
```

### MCP Server

```bash
# Run MCP server directly
uv run python mcp_server/server.py --transport stdio

# Add to Codex (global)
codex mcp add claude_context_local --scope user -- uv run --directory ~/.local/share/claude-context-local python mcp_server/server.py --transport stdio

# Add to Codex (project-specific)
codex mcp add claude_context_local -- uv run --directory ~/.local/share/claude-context-local python mcp_server/server.py --transport stdio
```

### MCP Environment Options

- `CODE_SEARCH_STORAGE`: Base directory for indexes and model cache.
- `CODE_SEARCH_DATA_DIR`: Alias for `CODE_SEARCH_STORAGE`.
- `CODE_SEARCH_DEVICE`: `cpu` | `mps` | `cuda` (auto if unset).
- `CODE_SEARCH_PRELOAD_MODEL`: `1` to preload on startup (off by default).
- `CODE_SEARCH_CHUNK_BATCH_SIZE`: chunk batch size for indexing.
- `CODE_SEARCH_EMBED_BATCH_SIZE`: embedding batch size for model encode.
- `CODE_SEARCH_INCLUDE_CONTEXT`: include same-file context in search results.
- `CODE_SEARCH_DISK_WARN_GB`: warn if free disk below this threshold (warn-only).
- `CODE_SEARCH_LARGE_FILE_MB`: warn on large files above this size (warn-only).
- `CODE_SEARCH_RESUME`: resume interrupted full indexing runs from checkpoint (default on; set `0` to disable).
- `CODE_SEARCH_ASYNC_INDEX`: force `index_directory` to run as a background job (recommended for large repos).
- `CODE_SEARCH_SYNC_INDEX`: force `index_directory` to block until completion (not recommended for large repos due to Codex tool-call timeouts).
- `CODE_SEARCH_ASYNC_FILE_THRESHOLD`: file-count heuristic used to auto background-index (default ~2500).
- `CODE_SEARCH_ASYNC_SCAN_SECONDS`: max seconds to spend estimating repo size before defaulting to background indexing (default ~2).
- `CODE_SEARCH_INDEX_WORKERS`: background indexing workers (default 1).
- `CODE_SEARCH_JOB_EVENT_BUFFER`: stored job progress events (default 200).
- `CODE_SEARCH_SHARDED_INDEX`: enable sharded FAISS indexes.
- `CODE_SEARCH_SHARD_TARGET_BYTES`: target shard size before rollover (default ~512MB).
- `CODE_SEARCH_SHARD_MEMORY_CAP_GB`: max RAM budget for loaded shards (default 13).

## Architecture

The codebase is organized into distinct modules with clear separation of concerns:

### Core Components

- **`chunking/`**: AST-based code parsing and chunking
  - `python_ast_chunker.py`: Breaks Python code into semantically meaningful chunks (functions, classes, modules)
  - `multi_language_chunker.py`: Tree-sitter based chunking for JavaScript, TypeScript, Go, Java, Rust, and Svelte
  - Preserves context and relationships between code elements
- **`embeddings/`**: Embedding generation using EmbeddingGemma
  - `embedder.py`: Handles model loading, caching, and batch embedding generation
  - Uses `google/embeddinggemma-300m` model with 768-dimensional embeddings
- **`search/`**: FAISS-based search and indexing
  - `indexer.py`: Manages FAISS indices, metadata storage (SQLite), and index persistence
  - `searcher.py`: Intelligent search with filtering, context-aware results, and similarity search
- **`mcp_server/`**: Codex integration via MCP

  - `server.py`: FastMCP server exposing search tools to Codex
  - `code_search_mcp.py`: legacy shim used by tests
  - Provides `search_code`, `index_directory`, `start_index_directory`, `get_index_job_status`, `cancel_index_job`, `find_similar_code`, etc.

- **`merkle/`**: Incremental indexing support
  - `merkle_dag.py`: Merkle tree implementation for efficient change detection
  - `change_detector.py`: Detects file additions, modifications, and deletions
  - `snapshot_manager.py`: Manages snapshots for incremental indexing
- **`search/incremental_indexer.py`**: Orchestrates incremental indexing using Merkle tree change detection

### Storage Structure

Data is stored in `~/.claude_code_search/` (configurable via `CODE_SEARCH_STORAGE`):

```
~/.claude_code_search/
├── models/          # Downloaded EmbeddingGemma models
├── projects/        # Project-specific data
│   └── {project_name}_{hash}/
│       ├── project_info.json  # Project metadata
│       ├── index/             # FAISS indices and metadata
│       │   ├── code.index     # Vector index
│       │   ├── metadata.db    # Chunk metadata (SQLite)
│       │   └── stats.json     # Index statistics
│       └── snapshots/         # Merkle tree snapshots for incremental indexing
```

### Chunking Strategy

The system uses AST parsing to create semantically meaningful chunks:

- Complete functions with docstrings and decorators
- Full classes with methods as separate chunks
- Module-level code blocks and constants
- Rich metadata: file paths, semantic tags, complexity scores, relationships

## Testing Strategy

Tests are organized by component with pytest markers:

- `unit`: Fast, isolated unit tests
- `integration`: End-to-end workflow tests
- `chunking`: AST chunking functionality
- `embeddings`: Model loading and embedding generation
- `search`: Indexing and search functionality
- `mcp`: MCP server integration
- `slow`: Time-intensive tests (excluded by default)

## Development Notes

### Key Dependencies

- `sentence-transformers`: EmbeddingGemma model loading and inference
- `faiss-cpu`: Efficient vector similarity search
- `fastmcp`: MCP server implementation for Codex integration
- `sqlitedict`: Persistent metadata storage
- `tree-sitter` & `tree-sitter-languages`: Multi-language parsing support
- `click`: Command-line interface utilities
- `pytest`: Testing framework with async support

### Performance Considerations

- Model size: ~300MB (EmbeddingGemma-300m)
- Embedding dimension: 768 (FAISS Flat index for small datasets, IVF for large)
- Batch processing: Configurable batch sizes for memory management
- Local processing: All embeddings computed locally, no API calls
- Incremental indexing: Only reprocesses changed files using Merkle tree snapshots

### Environment Variables

- `CODE_SEARCH_STORAGE`: Custom storage directory (default: `~/.claude_code_search`)
- `CODE_SEARCH_DATA_DIR`: Alias for `CODE_SEARCH_STORAGE`
- `CODE_SEARCH_CHUNK_BATCH_SIZE`: Chunk batch size for indexing (default: 256)
- `CODE_SEARCH_EMBED_BATCH_SIZE`: Embed batch size for model inference (falls back to `CODE_SEARCH_BATCH_SIZE`)
- `CODE_SEARCH_TORCH_BEFORE_FAISS`: Force torch import before FAISS on startup (`true`/`false`)
- `CODE_SEARCH_DISK_WARN_GB`: Warn if free disk below this threshold (warn-only)
- `CODE_SEARCH_LARGE_FILE_MB`: Warn on files larger than this size (warn-only)

## Common Tasks

### Adding New Chunk Types

1. Extend `python_ast_chunker.py` to handle new AST node types
2. Update metadata extraction in chunk creation
3. Add corresponding tests in `tests/unit/test_chunking.py`

### Modifying Search Behavior

1. Update `searcher.py` for new filtering/ranking logic
2. Modify MCP server tools in `server.py` if new parameters needed
3. Add integration tests in `tests/integration/test_full_flow.py`

### Testing Changes

Always run the full test suite before commits:

```bash
python tests/run_tests.py --coverage
```

For quick iteration during development:

```bash
python tests/run_tests.py --unit --verbose -x
```

### Multi-Language Support

The system now supports chunking and indexing multiple languages:

- Python (AST-based chunking)
- JavaScript/TypeScript (tree-sitter)
- JSX/TSX (React components)
- Go, Java, Rust (tree-sitter)
- Svelte components
- Markdown (section-based chunking by headers)
