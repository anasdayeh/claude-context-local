# Changelog

## 2026-07-10

### Fixed
- Made MPS→CPU embedding fallback observable through requested/actual device and structured fallback events; benchmark artifacts no longer claim MPS timing after CPU fallback.
- Enforced Gate-B finiteness, dimension, normalization and numerical-determinism checks.
- Persisted the real reranker candidate depth independently from hit@k scoring.
- Replaced existence-only benchmark resume with schema-v2 input/source fingerprints and atomic writes.
- Made the benchmark orchestrator exclusive and nonzero on required-stage failures.
- Added stable-memory handoff gates between Metal models and process-group-scoped shutdown.
- Rejected report inputs with different corpus identities/query ordering and made reports atomic.
- Upgraded to Sentence Transformers 5.6 native generative CrossEncoder support for the official Qwen3 reranker.
- Added compatibility with the current `tree_sitter_xml.language_xml()` binding API.

### Added
- Shared production embedding-input formatting between indexing and corpus dumps.
- Dataset admission checks that reject missing or overly broad expected-file labels.
- MRR, nDCG, paired exact tests and deterministic bootstrap confidence intervals.
- Stable MCP search-quality metadata for semantic, CPU-degraded semantic and FTS-degraded responses.

## 2026-04-16

### Fixed
- Fixed `NameError` in `CodeSearchServer.repair_index()`: `ShardedIndexManager` was used without a local import (would crash at runtime when repairing sharded indexes).
- Fixed `ThreadPoolExecutor` in `CodeSearchServer` never being shut down: added `shutdown()` method and `atexit.register()` to prevent process hangs.
- Fixed macOS libomp SIGABRT on test import: `conftest.py` now sets `KMP_DUPLICATE_LIB_OK=TRUE` early.
- Fixed test suite hanging indefinitely: `server.py` `os.execv` now guarded by `__name__ == "__main__"` (was firing on import, spawning nested pytest); `conftest.py` clears atexit handlers from sentence-transformers/torch.
- Fixed `search_code()` return type inconsistency: removed `as_dict` parameter; all paths now return `Dict[str, Any]` with a `results` key. Error responses include `results: []`.
- Fixed double disk-space check: removed redundant check from `_index_directory_impl` (callers already check before lock/job creation).
- Fixed async test markers: switched `@pytest.mark.asyncio` → `@pytest.mark.anyio` in 4 test files (pytest-asyncio not available in system Python).
- Fixed `test_mcp_tool_descriptions.py`: adapted to FastMCP API change (`list_tools()` → `_tool_manager._tools`).
- Added `-p no:xonsh` to `pytest.ini` to prevent xonsh plugin interference.
- Removed stale `reset_global_state` fixture that referenced nonexistent `server_module._embedder` attributes.

### Added
- Disk-space safety guard (`_check_indexing_disk_space`) on all indexing entry points; configurable via `CODE_SEARCH_MIN_FREE_DISK_GB` env var (default: 5 GB).
- New unit tests for disk-space refusal in both synchronous and background indexing paths.

## 2026-04-09

### Removed
- Deleted `mcp_server/code_search_server.py.bak` (dead backup, 149 lines, no references).
- Deleted `scripts/index_codebase.py` (legacy standalone indexer, superseded by `scripts/index_repo.py`).
- Deleted `embeddings/gemma.py` (`GemmaEmbeddingModel` was a trivial subclass); replaced with factory function in `embedding_models_register.py`.
- Deleted `mcp_server/code_search_mcp.py` (legacy test wrapper); refactored `tests/unit/test_mcp_tool_descriptions.py` to use the production `FastMCP` + `register_tools()` path.

### Fixed
- Fixed `AVAILIABLE_MODELS` → `AVAILABLE_MODELS` typo propagated across 5 files (10 sites).
- Fixed `get_availiable_language` → `get_available_language` typo in 2 files (3 sites).

### Refactored
- Eliminated `TextChunk` intermediate dataclass in `chunking/text_chunker.py`; `CodeChunk` is now constructed directly in a single pass.
- Unified duplicate ignore-directory lists: `DEFAULT_IGNORED_DIRS` and `get_ignore_patterns()` now live in `common_utils.py` as the single source of truth, imported by both `chunking/multi_language_chunker.py` and `merkle/merkle_dag.py`. `CODE_SEARCH_IGNORE_DIRS` env var now applies to both chunking and merkle.
- Added `TestLanguageDefinitionSync` tests to verify `available_languages.py` and `LANGUAGE_MAP` stay in sync.
- Removed orphaned `# fallback logic improved` comment from `multi_language_chunker.py`.

## 2025-12-19
- Added ONNX backend support with optional int8 quantization for embeddings, plus prompt registry validation and safe float32 outputs.
- Switched FAISS indexing to ID-mapped vectors with real deletions and SQLite-backed ID mapping; removed pickle usage.
- Streamed incremental indexing batches to reduce memory pressure on large projects.
- Expanded tree-sitter language support (HTML/CSS/JSON/Astro/YAML/TOML/XML/GraphQL) and added text fallback chunker.
- Refactored MCP server entrypoint to FastMCP decorator style with resource update notifications.
- Added legacy index search compatibility and SQLite recovery for disk I/O errors, plus improved auto-index detection for non-Python repos.

## 2025-12-20
- Added embedding runtime fallbacks (ONNX → torch, MPS → CPU) and environment-driven device/batch-size overrides to stabilize large reindex runs.
- Disabled context expansion in search_code unless explicitly enabled via CODE_SEARCH_INCLUDE_CONTEXT, and made context lookups best-effort to avoid search hangs.
- Switched FastMCP startup to lifespan hook, reduced logging verbosity, and forced logs to stderr to keep stdio transport stable.
- Replaced stdout prints in snapshot manager with logging and disabled HF progress/telemetry to prevent stdio protocol pollution.
- Added optional log file routing and configured log handlers to avoid stdout pollution and reduce stderr volume.
- Suppressed third-party warnings/verbosity to minimize stderr output and prevent Codex stdio transport closures.
- Replaced FAISS reconstruct-based context expansion with same-file neighbor context to avoid segfaults when CODE_SEARCH_INCLUDE_CONTEXT=1.
- Ensured PyTorch initializes before FAISS (import ordering) to avoid OpenMP runtime crashes on Apple Silicon.
- Disabled model preload by default (CODE_SEARCH_PRELOAD_MODEL=0) to prevent blocking the stdio initialize handshake.
- Consolidated MCP tool registration and strings loading into shared helpers to prevent entrypoint drift.
- Moved MCP tool execution to background threads for long operations to keep stdio responsive.
- Added index access locks to prevent concurrent FAISS mutation and search conflicts.
- Added explicit chunk vs embedding batch size configuration knobs.
- Updated docs with Codex MCP usage and new environment variables.
- Made search_code auto-reindex opt-in by default to avoid hidden indexing during search.

## [Unreleased] - 2025-12-23

### Fixed
- **embeddings/embedding_model.py**: Hardened MPS device detection to avoid `NoneType` backend errors when `torch.backends.mps` is absent or partially initialized.
- **embeddings/sentence_transformer.py**: Made cleanup cache clears resilient to interpreter shutdown (guarded CUDA/MPS checks).
- **embeddings/embedder.py**: Made device cache clears resilient to interpreter shutdown (guarded CUDA/MPS checks).
- **search/incremental_indexer.py**: Added env-driven batch sizing and unified glob matching to make `file_patterns` more reliable for large index runs.
- **merkle/merkle_dag.py**: Added CODE_SEARCH_IGNORE_DIRS support to skip heavy build/vendor folders globally during indexing.
- **mcp_server/mcp_tools.py**: Improved progress token extraction robustness and added `related_request_id` support for progress notifications.
- **mcp_server/mcp_tools.py**: Corrected resource update URI scheme and added notifications for project list updates.
- **mcp_server/code_search_server.py**: Fixed `include_context` logic in `search_code` to honor the tool parameter.
- **mcp_server/code_search_server.py**: Improved "project indexed" check to use `stats.json` instead of a hardcoded index filename.
- **mcp_server/server.py**: Normalized transport handling and improved security defaults for HTTP transport.
- **mcp_server/code_search_mcp.py**: Fixed conditional inheritance of `FastMCP` to prevent `ImportError` in partial environments.
- **mcp_server/strings.yaml**: Updated `index_directory` documentation to match the tool schema (`file_patterns`).
- **common_utils.py**: Added CODE_SEARCH_DATA_DIR alias for storage configuration.
- **mcp_server/code_search_server.py**: Restored safe model preload hook to avoid startup crashes when preloading is enabled.
- **chunking/tree_sitter.py**: Tolerates non-UTF-8 source files by using replacement decoding.
- **embeddings/embedder.py**: Added adaptive batch backoff on OOM with cache clears and CPU fallback.
- **search/incremental_indexer.py**: Warns on low disk and large files without stopping indexing.
- **mcp_server/mcp_tools.py**: Surfaced adaptive embedding backoff messages via MCP progress notifications.

### Security
- Restricted default bind host for HTTP transport to `127.0.0.1` in `server.py`.

- **search/indexer.py**: Implemented robust glob matching in `_apply_filters` using `fnmatch` and unanchored path support.
- **search/incremental_indexer.py**: Added full support for `file_patterns` filtering during both incremental and full indexing phases.
- **mcp_server/code_search_server.py**: Updated to correctly propagate `file_patterns` from the MCP tool to the indexing engine.

### Chunking Logic Improvements
- **Robustness**: Updated `available_languages.py` and `base_chunker.py` to handle missing `tree-sitter` dependencies gracefully without crashing.
- **Fallback**: Improved `multi_language_chunker.py` to fallback to text chunking for ANY file type if Tree-sitter parsing is unavailable or fails.

### Embeddings Logic Improvements
- **Robust Wrapper**: `SentenceTransformerModel` now supports `encode_query`/`encode_document` and properly resolves devices (honoring "cpu" and "cuda:N").
- **Safety**: `CodeEmbedder` uses fallback-enabled wrapper methods and clamps text budgets to prevent negative indexing errors.
- **Usability**: Added model registry aliases (e.g. `embeddinggemma-300m`) and improved clean-up logic to release cached models.

### Fixed
- **Real E2E Verification**: Successfully verified full system flow with real `torch`, `sentence-transformers`, and `google/embeddinggemma-300m` model.
- **Search Key Alignment**: Fixed mismatches between searcher results and MCP tool expectations (using `res.to_search_tool_dict()`).
- **Tool Signature Alignment**: Fixed mismatch between `mcp_tools.py` and `code_search_server.py` signatures for `search_code`.
- **Project Management**: Implemented missing server methods (`find_similar_code`, `clear_index`, `get_index_status`, `index_test_project`).

### Fixed
- **MacOS ARM Support**: Upgraded project to Python 3.13 to resolve broken `torch` wheels on Python 3.12 (missing `libtorch_cpu.dylib`).
- **Dependency Isolation**: Fully migrated to `uv` for environment management, resolving PEP 668 "externally-managed-environment" errors.
- **Merkle Snapshot Manager**: Fixed regression where `load_latest_snapshot` was missing after refactor.
- **Installer**: Updated `scripts/install.sh` to enforce Python 3.13 and include `uv` best practices.
