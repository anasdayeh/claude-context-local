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
