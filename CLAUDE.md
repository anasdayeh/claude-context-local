# Code Search MCP Server

## Project Overview
Semantic code search MCP server using tree-sitter chunking, sentence-transformer embeddings, and FAISS/sharded indexes.

## Architecture
- `mcp_server/code_search_server.py` — Core server: indexing, search, project management
- `mcp_server/mcp_tools.py` — FastMCP tool/resource registration (the MCP surface)
- `mcp_server/server.py` — Entrypoint (has `os.execv` guard, only fires when `__name__ == "__main__"`)
- `chunking/` — Tree-sitter + text chunking (multi-language)
- `embeddings/` — Embedding models (SentenceTransformer, factory-based registry)
- `search/` — FAISS indexing, incremental indexer, sharded index manager
- `merkle/` — Merkle DAG for change tracking
- `common_utils.py` — Shared constants (ignore patterns), memory utils, adaptive defaults

## Key Patterns
- `AVAILABLE_MODELS` in `embeddings/embedding_models_register.py` maps model names to either classes or factory functions. Both must accept `(model_name, device=..., cache_dir=...)`.
- `DEFAULT_IGNORED_DIRS` lives in `common_utils.py` as a `frozenset` and is imported by `multi_language_chunker.py` and `merkle_dag.py`.
- `CodeSearchServer` creates a `ThreadPoolExecutor` for background indexing. Call `server.shutdown()` in tests or register atexit (done automatically in `__init__`).
- Local imports for `ShardedIndexManager` and `IncrementalIndexer` — these are heavy; imported only when needed. Every call site must have its own local import.
- `search_code()` always returns `Dict[str, Any]` with a `results` key (list) — removed the `as_dict` parameter.

## Running Tests
```bash
python -m pytest tests/unit/ -v --tb=short
```
- `conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` to prevent libomp SIGABRT on macOS.
- `conftest.py` patches `AVAILABLE_MODELS` with `EmbeddingModelMock` for integration tests.
- `conftest.py` uses `atexit._clear()` in `pytest_sessionfinish` to prevent sentence-transformers/torch atexit hangs.
- Tests that create `CodeSearchServer` should monkeypatch `CodeEmbedder` to avoid real model loading.
- Async tests use `@pytest.mark.anyio` (not `asyncio`) — `pytest-anyio` is available; `pytest-asyncio` may not be.
- `pytest.ini` includes `-p no:xonsh` to prevent the xonsh plugin from spawning nested pytest sessions.

## Gotchas
- `repair_index()` and `_maybe_auto_repair_sharded_index()` use local imports for `ShardedIndexManager` — if adding new callers, remember the local import.
- The `_gemma_factory` in `embedding_models_register.py` ignores its `model_name` argument and hardcodes `google/embeddinggemma-300m` — this is intentional.
- `server.py` has an `os.execv` at module level guarded by `__name__ == "__main__"` — importing the module in tests does NOT trigger re-exec. If this guard is removed, tests will spawn nested pytest sessions.
- `sentence_transformers` import registers atexit handlers that block interpreter shutdown — `conftest.py` clears them.
- FastMCP API: use `mcp._tool_manager._tools` to get tool objects; `mcp.get_tools()` returns name strings only.
- The `_check_indexing_disk_space()` guard is called in `index_directory()` and `start_index_job()` (the callers), NOT in `_index_directory_impl()` (the shared implementation).

## Env Vars
- `CODE_SEARCH_MIN_FREE_DISK_GB` — Minimum free disk for indexing (default: 5.0 GB)
- `CODE_SEARCH_RUNTIME_SELFTEST` — Enable embedder warmup selftest on init
- `CODE_SEARCH_DEVICE` — Embedding device (auto/cpu/mps/cuda)
- `CODE_SEARCH_IGNORE_DIRS` — Extra comma-separated dirs to ignore
- `KMP_DUPLICATE_LIB_OK` — Set TRUE on macOS to prevent libomp crash
