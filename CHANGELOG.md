# Changelog

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
- **mcp_server/mcp_tools.py**: Improved progress token extraction robustness and added `related_request_id` support for progress notifications.
- **mcp_server/mcp_tools.py**: Corrected resource update URI scheme and added notifications for project list updates.
- **mcp_server/code_search_server.py**: Fixed `include_context` logic in `search_code` to honor the tool parameter.
- **mcp_server/code_search_server.py**: Improved "project indexed" check to use `stats.json` instead of a hardcoded index filename.
- **mcp_server/server.py**: Normalized transport handling and improved security defaults for HTTP transport.
- **mcp_server/code_search_mcp.py**: Fixed conditional inheritance of `FastMCP` to prevent `ImportError` in partial environments.
- **mcp_server/strings.yaml**: Updated `index_directory` documentation to match the tool schema (`file_patterns`).

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
