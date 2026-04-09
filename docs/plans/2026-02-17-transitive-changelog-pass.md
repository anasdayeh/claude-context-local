# 2026-02-17 Transitive Dependency Changelog Pass

## Scope
- Packages: `transformers`, `optimum-onnx`, `onnxruntime`, `torch`.
- Goal: map the last ~5 releases per package to concrete adaptations for this MCP code-search server on Apple Silicon (M1, 16 GB).

## Current Resolved Versions (after this pass)
- `transformers==4.57.6`
- `optimum-onnx==0.1.0`
- `onnxruntime==1.24.1`
- `torch==2.9.1`

## Changelog-by-Changelog Assessment

### transformers
Current migration boundary: v4.57.6 -> v5.x

1) `v5.2.0` (2026-02-16)  
Source: <https://github.com/huggingface/transformers/releases/tag/v5.2.0>
- Change class: major feature release on top of v5 API surface.
- Adaptation: **defer** for this repo until explicit v5 migration window (we do not need v5-only features for embedding search).

2) `v5.1.0` (2026-02-05)  
Source: <https://github.com/huggingface/transformers/releases/tag/v5.1.0>
- Change class: major-line incremental release.
- Adaptation: **defer** with v5 line.

3) `v5.0.0` (2026-01-26)  
Source: <https://github.com/huggingface/transformers/releases/tag/v5.0.0>  
Migration guide: <https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md>
- Change class: breaking API changes.
- Adaptation: **hold at v4** and pin `<4.58.0` because current `sentence-transformers[onnx] + optimum-onnx` route is stable there.

4) `v4.57.6` (2026-01-16)  
Source: <https://github.com/huggingface/transformers/releases/tag/v4.57.6>
- Change class: patch fixes in late v4 series.
- Adaptation: **adopted** via direct constraint pin to `>=4.57.6,<4.58.0`.

5) `v4.57.5` (2026-01-13)  
Source: <https://github.com/huggingface/transformers/releases/tag/v4.57.5>
- Change class: patch line predecessor.
- Adaptation: superseded by v4.57.6 selection.

Decision: lock stable late-v4 line now, plan explicit v5 migration later.

---

### optimum-onnx

1) `v0.1.0` (2025-12-23)  
Source: <https://github.com/huggingface/optimum-onnx/releases/tag/v0.1.0>
- Notable: transformers 4.56/4.57 support.
- Adaptation: **adopted** (direct dependency floor `optimum-onnx[onnxruntime]>=0.1.0`).

2) `v0.0.3` (2025-12-23)  
Source: <https://github.com/huggingface/optimum-onnx/releases/tag/v0.0.3>
- Notable: torch.onnx export fix for older torch.
- Adaptation: superseded by v0.1.0.

3) `v0.0.2` (2025-10-16)  
Source: <https://github.com/huggingface/optimum-onnx/releases/tag/v0.0.2>
- Notable: torch 2.9 compatibility patch.
- Adaptation: retained implicitly via upgrade path.

4) `v0.0.1` (2025-10-09)  
Source: <https://github.com/huggingface/optimum-onnx/releases/tag/v0.0.1>
- Initial baseline.
- Adaptation: superseded.

Decision: move from `0.0.3` to `0.1.0` to unlock modern v4 transformers compatibility.

---

### onnxruntime

1) `v1.24.1` (2026-02-06)  
Source: <https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1>
- Notable: platform support updates (py3.10 wheels dropped, py3.14 support, macOS x86_64 binaries removed).
- Adaptation: **adopted** (`onnxruntime>=1.24.1`) because repo requires Python >=3.13 and target machine is Apple Silicon.

2) `v1.23.2` (2025-10-25)  
Source: <https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2>
- Patch baseline.
- Adaptation: superseded by v1.24.1.

3) `v1.23.1` (2025-10-08)  
Source: <https://github.com/microsoft/onnxruntime/releases/tag/v1.23.1>
- Includes CPU-side fixes/new Python APIs.
- Adaptation: covered by newer release.

4) `v1.23.0` (2025-09-26)  
Source: <https://github.com/microsoft/onnxruntime/releases/tag/v1.23.0>
- EP plugin infrastructure additions.
- Adaptation: no direct code change needed for current CPU EP usage.

5) `v1.22.2` (2025-08-13)  
Source: <https://github.com/microsoft/onnxruntime/releases/tag/v1.22.2>
- Notes mention client-build defaults and thread-spinning context.
- Adaptation: reinforced thread/governor focus in this repo.

Additional ORT tuning references:
- ORT FAQ (single-thread control): <https://github.com/microsoft/onnxruntime/blob/main/docs/FAQ.md>
- Session options/config entries: <https://onnxruntime.ai/docs/performance/tune-performance/threading.html>

Decision: upgrade ORT and keep CPU provider default for M1 stability profile.

---

### torch

1) `v2.10.0` (2026-01-21)  
Source: <https://github.com/pytorch/pytorch/releases/tag/v2.10.0>
- Major/minor update with broader compiler/runtime changes.
- Adaptation: **defer for now** (no pressing blocker in current embedding path).

2) `v2.9.1` (2025-11-12)  
Source: <https://github.com/pytorch/pytorch/releases/tag/v2.9.1>
- Bugfix release (regression fixes).
- Adaptation: **current chosen baseline**.

3) `v2.9.0` (2025-10-15)  
Source: <https://github.com/pytorch/pytorch/releases/tag/v2.9.0>
- Feature release; superseded by 2.9.1 bugfix line.

4) `v2.8.0` (2025-08-06)  
Source: <https://github.com/pytorch/pytorch/releases/tag/v2.8.0>
- Older minor line.

5) `v2.7.1` (2025-06-04)  
Source: <https://github.com/pytorch/pytorch/releases/tag/v2.7.1>
- Older bugfix line.

MPS/threads references:
- MPS notes: <https://github.com/pytorch/pytorch/blob/main/docs/source/notes/mps.rst>
- Multiprocessing/thread oversubscription notes: <https://github.com/pytorch/pytorch/blob/main/docs/source/notes/multiprocessing.rst>

Decision: stay on 2.9.1 and control runtime thread pressure in-app.

## Code Adaptations Implemented
- Added explicit transitive dependency floors in `pyproject.toml`:
  - `transformers>=4.57.6,<4.58.0`
  - `optimum-onnx[onnxruntime]>=0.1.0`
  - `onnxruntime>=1.24.1`
- Added runtime backend control `CODE_SEARCH_EMBED_BACKEND` (`torch` or `onnx`) in `embeddings/sentence_transformer.py`.
- Added safe device coercion for ONNX backend on Apple Silicon (`mps` -> `cpu`) in `embeddings/sentence_transformer.py`.
- Added torch thread caps support in runtime (`CODE_SEARCH_TORCH_NUM_THREADS`, `CODE_SEARCH_TORCH_INTEROP_THREADS`) and adaptive defaults in `common_utils.py`.
- Updated scripts/docs to expose these controls and M1 low-memory profile.

## Rollout and Verification Matrix
1) Resolve/install
- `uv sync`
- Verify: `uv run python -c "import importlib.metadata as m;print(m.version('transformers'), m.version('optimum-onnx'), m.version('onnxruntime'), m.version('torch'))"`

2) Unit verification
- `uv run pytest tests/unit/test_sentence_transformer_runtime_controls.py -q`
- `uv run pytest tests/unit/test_common_utils.py tests/unit/test_embedder_adaptive.py -q`

3) Runtime sanity on M1 profile
- Export low-memory profile env vars (README section).
- Run incremental index on a medium repo; watch RSS/log stability.

4) Performance follow-up (pending)
- Benchmark `torch` vs `onnx` backend for indexing throughput and memory peak on identical file sets.
