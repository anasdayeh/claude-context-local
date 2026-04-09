# 2026-02-17 MCP Performance + Dependency Audit (In Progress)

## Scope
- Trace indexing/search execution paths and identify CPU/RAM pressure points on Apple Silicon (M1, 16 GB).
- Audit direct dependencies, collect latest + last five releases, and map actionable upgrades.
- Keep index incremental (no destructive resets) and reduce context/output bloat for coding agents.

## Live Log
- 2026-02-17T10:29:50.120505+00:00: Refreshed dependency dataset at `docs/plans/audit-data/dependency-release-audit-2026-02-17.json` after transitive upgrade pins.
- Added dedicated transitive changelog pass: `docs/plans/2026-02-17-transitive-changelog-pass.md`.
- Identified high-cost hot paths:
  - `search/searcher.py`: same-file context was rebuilding via repeated full metadata scans per result.
  - `search/indexer.py`: FTS upserts were one-row-per-transaction during embedding adds.
  - `search/incremental_indexer.py`: checkpoint frequency was too aggressive for large repos.
  - `embeddings/embedder.py`: `content_preview` duplicated full content, bloating metadata/context.

## Code Changes Applied in This Pass
- `embeddings/embedder.py`: truncate stored `content_preview` (env: `CODE_SEARCH_CONTENT_PREVIEW_CHARS`, default `320`).
- `search/searcher.py`: build per-query file context cache once and reuse it across results.
- `search/indexer.py`: add batched `fts_upsert_many()` and use it in `add_embeddings()`.
- `search/incremental_indexer.py`: make checkpoint cadence configurable with `CODE_SEARCH_CHECKPOINT_CHUNKS`; safer default (`max(chunk_batch*20,1000)`).
- `embeddings/sentence_transformer.py`: add runtime backend control (`CODE_SEARCH_EMBED_BACKEND`), ONNX-on-MPS safe device coercion, and torch thread caps.
- `common_utils.py`: adaptive defaults now include torch thread controls for constrained devices.
- `scripts/reindex_*.sh`: expose new backend/thread env vars across operational scripts.
- `pyproject.toml`: raised stable dependency floors and added direct transitive pins for ONNX stack stability.

## Direct Dependency Release Snapshot
Source data: `docs/plans/audit-data/dependency-release-audit-2026-02-17.json`

| Package | Constraint | Latest on PyPI | Latest compatible | Last 5 compatible releases (version @ date) |
|---|---|---|---|---|
| `click` | `click>=8.3.1` | `8.3.1` | `8.3.1` | 8.3.1 @ 2025-11-15 |
| `faiss-cpu` | `faiss-cpu>=1.13.2` | `1.13.2` | `1.13.2` | 1.13.2 @ 2025-12-24 |
| `fastmcp` | `fastmcp>=2.14.5,<3.0.0` | `2.14.5` | `2.14.5` | 2.14.5 @ 2026-02-03 |
| `huggingface-hub` | `huggingface-hub>=0.36.2,<1.0` | `1.4.1` | `0.36.2` | 0.36.2 @ 2026-02-06 |
| `pyyaml` | `pyyaml>=6.0.3` | `6.0.3` | `6.0.3` | 6.0.3 @ 2025-09-29 |
| `mcp` | `mcp>=1.26.0` | `1.26.0` | `1.26.0` | 1.26.0 @ 2026-01-24 |
| `pytest` | `pytest>=8.4.2` | `9.0.2` | `9.0.2` | 9.0.2 @ 2025-12-06, 9.0.1 @ 2025-11-12, 9.0.0 @ 2025-11-08, 8.4.2 @ 2025-09-04 |
| `pytest-asyncio` | `pytest-asyncio>=1.1.0` | `1.3.0` | `1.3.0` | 1.3.0 @ 2025-11-10, 1.2.0 @ 2025-09-12, 1.1.1 @ 2025-09-12, 1.1.0 @ 2025-07-16 |
| `pytest-cov` | `pytest-cov>=6.2.1` | `7.0.0` | `7.0.0` | 7.0.0 @ 2025-09-09, 6.3.0 @ 2025-09-06, 6.2.1 @ 2025-06-12 |
| `pytest-mock` | `pytest-mock>=3.14.1` | `3.15.1` | `3.15.1` | 3.15.1 @ 2025-09-16, 3.15.0 @ 2025-09-04, 3.14.1 @ 2025-05-26 |
| `rich` | `rich>=14.3.2` | `14.3.2` | `14.3.2` | 14.3.2 @ 2026-02-01 |
| `sentence-transformers` | `sentence-transformers[onnx]>=5.2.2` | `5.2.2` | `5.2.2` | 5.2.2 @ 2026-01-27 |
| `transformers` | `transformers>=4.57.6,<4.58.0` | `5.2.0` | `4.57.6` | 4.57.6 @ 2026-01-16 |
| `optimum-onnx` | `optimum-onnx[onnxruntime]>=0.1.0` | `0.1.0` | `0.1.0` | 0.1.0 @ 2025-12-23 |
| `onnxruntime` | `onnxruntime>=1.24.1` | `1.24.1` | `1.24.1` | 1.24.1 @ 2026-02-05 |
| `sqlitedict` | `sqlitedict>=2.1.0` | `2.1.0` | `2.1.0` | 2.1.0 @ 2022-12-03 |
| `psutil` | `psutil>=7.2.2` | `7.2.2` | `7.2.2` | 7.2.2 @ 2026-01-28 |
| `tree-sitter` | `tree-sitter>=0.25.2` | `0.25.2` | `0.25.2` | 0.25.2 @ 2025-09-25 |
| `tree-sitter-c` | `tree-sitter-c>=0.24.1` | `0.24.1` | `0.24.1` | 0.24.1 @ 2025-05-24 |
| `tree-sitter-c-sharp` | `tree-sitter-c-sharp>=0.23.1` | `0.23.1` | `0.23.1` | 0.23.1 @ 2024-11-11 |
| `tree-sitter-cpp` | `tree-sitter-cpp>=0.23.4` | `0.23.4` | `0.23.4` | 0.23.4 @ 2024-11-11 |
| `tree-sitter-go` | `tree-sitter-go>=0.25.0` | `0.25.0` | `0.25.0` | 0.25.0 @ 2025-08-29 |
| `tree-sitter-java` | `tree-sitter-java>=0.23.5` | `0.23.5` | `0.23.5` | 0.23.5 @ 2024-12-21 |
| `tree-sitter-javascript` | `tree-sitter-javascript>=0.25.0` | `0.25.0` | `0.25.0` | 0.25.0 @ 2025-09-01 |
| `tree-sitter-html` | `tree-sitter-html>=0.23.2` | `0.23.2` | `0.23.2` | 0.23.2 @ 2024-11-11 |
| `tree-sitter-css` | `tree-sitter-css>=0.25.0` | `0.25.0` | `0.25.0` | 0.25.0 @ 2025-09-28 |
| `tree-sitter-json` | `tree-sitter-json>=0.24.8` | `0.24.8` | `0.24.8` | 0.24.8 @ 2024-11-11 |
| `tree-sitter-language-pack` | `tree-sitter-language-pack>=0.13.0` | `0.13.0` | `0.13.0` | 0.13.0 @ 2025-11-26 |
| `tree-sitter-markdown` | `tree-sitter-markdown>=0.5.1` | `0.5.1` | `0.5.1` | 0.5.1 @ 2025-09-16 |
| `tree-sitter-yaml` | `tree-sitter-yaml>=0.7.2` | `0.7.2` | `0.7.2` | 0.7.2 @ 2025-10-07 |
| `tree-sitter-toml` | `tree-sitter-toml>=0.7.0` | `0.7.0` | `0.7.0` | 0.7.0 @ 2024-12-03 |
| `tree-sitter-xml` | `tree-sitter-xml>=0.7.0` | `0.7.0` | `0.7.0` | 0.7.0 @ 2024-11-13 |
| `tree-sitter-graphql` | `tree-sitter-graphql>=0.1.0` | `0.1.0` | `0.1.0` | 0.1.0 @ 2025-06-11 |
| `tree-sitter-python` | `tree-sitter-python>=0.25.0` | `0.25.0` | `0.25.0` | 0.25.0 @ 2025-09-11 |
| `tree-sitter-rust` | `tree-sitter-rust>=0.24.0` | `0.24.0` | `0.24.0` | 0.24.0 @ 2025-04-01 |
| `tree-sitter-svelte` | `tree-sitter-svelte>=1.0.2` | `1.0.2` | `1.0.2` | 1.0.2 @ 2024-09-08 |
| `tree-sitter-typescript` | `tree-sitter-typescript>=0.23.2` | `0.23.2` | `0.23.2` | 0.23.2 @ 2024-11-11 |

## Key Upgrade + Adoption Opportunities (Prioritized)
1. **Context/search cost reduction**: keep context cache + default `CODE_SEARCH_INCLUDE_CONTEXT=0` in local profiles for large repos.
2. **Storage + token hygiene**: keep preview truncation; optionally add hard response-size guards in `mcp_server/mcp_tools.py` for very large result sets.
3. **Backend flexibility**: use `CODE_SEARCH_EMBED_BACKEND=onnx` for low-memory profiles and keep torch fallback path intact.
4. **Checkpoint I/O throttling**: tune `CODE_SEARCH_CHECKPOINT_CHUNKS` by repo size (e.g., 1500-4000) to reduce sync spikes.
5. **Transformers v5 migration track**: hold stable v4.57.x now; plan explicit v5 migration once sentence-transformers + optimum stack is validated end-to-end.

## Critical Compatibility Finding
- `sentence-transformers[onnx]` currently resolves through `optimum-onnx` + `transformers` ranges that still require `huggingface-hub<1.0`; upgrading hub to `>=1.0` made the environment unsatisfiable. Keep hub on `0.36.x` until upstream ranges move.

## Next Pass Backlog
- Add optional memory-pressure governor in indexing loop (pause/sleep when free RAM drops below threshold instead of only shrinking embed batches).
- Convert root stats updates to incremental counters to avoid full metadata scans on each save.
- Add benchmark harness for M1 profiles (index throughput, peak RSS, search latency, token payload size).
- Validate dependency upgrades in a clean venv (`uv sync`) and run full unit suite.
