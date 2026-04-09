# Configurable ONNX Embedding Backend + Dependency Upgrades Implementation Plan

> **For codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a configurable SentenceTransformers backend (torch vs onnx) with optional ONNX export and dynamic int8 quantization, plus Hugging Face Hub v1 download improvements and dependency hygiene upgrades, with measurable benchmarks and rollback. Benchmarks must run in a fixed mode (same corpus + same query set) for comparability.

**Architecture:** Add env-driven backend selection inside `SentenceTransformerModel`, expose export and quantization controls, and implement a benchmark harness that measures embedding throughput, indexing time, and search latency. Keep CLI and MCP on the same path. Upgrade dependency pins and add logging for cold start model load time.

**Tech Stack:** Python 3.13+, SentenceTransformers, ONNX Runtime, FAISS, FastMCP, tree-sitter, pytest, uv

---

## Workstream 1: Configurable embeddings backend + ONNX fast path

### Task 1: Add benchmark harness skeleton

**Files:**
- Create: `scripts/bench_mcp_perf.py`
- Create: `tests/unit/test_bench_harness_schema.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
# tests/unit/test_bench_harness_schema.py
import json
import subprocess
import sys
from pathlib import Path

def test_bench_harness_schema(tmp_path):
    out = tmp_path / "bench.json"
    cmd = [sys.executable, "scripts/bench_mcp_perf.py", "--dry-run", f"--out={out}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    data = json.loads(out.read_text())
    assert "embedding_throughput_chunks_per_sec" in data
    assert "indexing_time_seconds" in data
    assert "search_latency_ms" in data
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_bench_harness_schema.py -v`
Expected: FAIL (file not found or missing fields)

**Step 3: Write minimal implementation**

```python
# scripts/bench_mcp_perf.py
import argparse, json, os, time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = {
        "embedding_throughput_chunks_per_sec": None,
        "indexing_time_seconds": None,
        "search_latency_ms": {"p50": None, "p95": None},
        "meta": {"dry_run": bool(args.dry_run)},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_bench_harness_schema.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/bench_mcp_perf.py tests/unit/test_bench_harness_schema.py README.md
git commit -m "feat: add benchmark harness skeleton"
```

### Task 2: Implement full benchmark harness (fixed benchmark mode)

**Files:**
- Modify: `scripts/bench_mcp_perf.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
# tests/unit/test_bench_harness_schema.py
# Extend to assert numeric values in non-dry-run mode using a tiny sample
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_bench_harness_schema.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scripts/bench_mcp_perf.py
# Add real measurements for:
# - embedding throughput (chunks/sec)
# - indexing time (seconds)
# - search latency (p50/p95)
# - cosine similarity sanity check if quantized enabled
# - top-k overlap check vs torch baseline
# Fixed benchmark mode:
# - Same corpus path every run: /Users/anasdayeh/Downloads/ADS_Website
# - Same query list every run (hard-coded list in script)
# - Same top-k for latency and overlap
# Write JSON to CODE_SEARCH_STORAGE/logs and print a table
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_bench_harness_schema.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/bench_mcp_perf.py tests/unit/test_bench_harness_schema.py README.md
git commit -m "feat: add performance benchmark harness"
```

### Task 3: Baseline benchmark (torch backend)

**Files:**
- Modify: `docs/plans/2026-02-03-onnx-embeddings-backend.md`

**Step 1: Run baseline benchmark**

Run:
```bash
CODE_SEARCH_EMBED_BACKEND=torch \
CODE_SEARCH_DEVICE=cpu \
uv run python scripts/bench_mcp_perf.py \
  --repo /Users/anasdayeh/Downloads/ADS_Website \
  --out "$CODE_SEARCH_STORAGE/logs/bench_baseline_torch.json"
```

Expected: JSON created and table printed

**Step 2: Record baseline numbers in plan**

Update this plan file with baseline values from JSON.

**Step 3: Commit**

```bash
git add docs/plans/2026-02-03-onnx-embeddings-backend.md
git commit -m "docs: record baseline perf numbers"
```

### Task 4: Add backend flags and ONNX export/quantization controls

**Files:**
- Modify: `embeddings/sentence_transformer.py`
- Modify: `embeddings/embedding_models_register.py`
- Modify: `embeddings/embedder.py`
- Modify: `README.md`
- Modify: `CODEX.md`
- Test: `tests/unit/test_sentence_transformer_backend.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_sentence_transformer_backend.py
import os
from unittest import mock
from embeddings.sentence_transformer import SentenceTransformerModel

@mock.patch("embeddings.sentence_transformer.SentenceTransformer")
def test_backend_env_is_used(mock_st):
    os.environ["CODE_SEARCH_EMBED_BACKEND"] = "onnx"
    model = SentenceTransformerModel("all-MiniLM-L6-v2", device="cpu")
    _ = model.model
    args, kwargs = mock_st.call_args
    assert kwargs.get("backend") == "onnx"
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_sentence_transformer_backend.py -v`
Expected: FAIL (backend not set)

**Step 3: Write minimal implementation**

```python
# embeddings/sentence_transformer.py
self.backend = os.getenv("CODE_SEARCH_EMBED_BACKEND", "torch").lower()
self._onnx_export = os.getenv("CODE_SEARCH_EMBED_ONNX_EXPORT", "0").lower() in {"1","true","yes"}
self._onnx_quantize = os.getenv("CODE_SEARCH_EMBED_ONNX_QUANTIZE", "0").lower() in {"1","true","yes"}

# In _load_model()
return SentenceTransformer(
    model_source,
    cache_folder=self.cache_dir,
    device=self._device,
    trust_remote_code=self.trust_remote_code,
    backend=self.backend,
    model_kwargs={
        "export": self._onnx_export,
        "provider": os.getenv("CODE_SEARCH_EMBED_ONNX_PROVIDER"),
        "file_name": os.getenv("CODE_SEARCH_EMBED_ONNX_FILE"),
    } if self.backend == "onnx" else None,
)

# If CODE_SEARCH_EMBED_ONNX_QUANTIZE=1
# call export_dynamic_quantized_onnx_model(model, quantization_config=...)
# guard with a check for existing quantized file
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_sentence_transformer_backend.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add embeddings/sentence_transformer.py embeddings/embedding_models_register.py embeddings/embedder.py README.md CODEX.md tests/unit/test_sentence_transformer_backend.py
git commit -m "feat: make embedding backend configurable with ONNX export/quant"
```

### Task 5: Quantization sanity check

**Files:**
- Modify: `scripts/bench_mcp_perf.py`
- Test: `tests/unit/test_bench_harness_schema.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_bench_harness_schema.py
# assert JSON includes "quant_similarity" when quantization enabled
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_bench_harness_schema.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# scripts/bench_mcp_perf.py
# If CODE_SEARCH_EMBED_ONNX_QUANTIZE=1, compute cosine similarity
# between torch and quantized ONNX embeddings on a fixed sample.
# Store summary stats: mean, p5, p50, p95.
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_bench_harness_schema.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/bench_mcp_perf.py tests/unit/test_bench_harness_schema.py
git commit -m "feat: add quantization similarity sanity check"
```

### Task 6: After benchmarks (ONNX, ONNX+quant)

**Files:**
- Modify: `docs/plans/2026-02-03-onnx-embeddings-backend.md`

**Step 1: Run ONNX backend benchmark**

Run:
```bash
CODE_SEARCH_EMBED_BACKEND=onnx \
CODE_SEARCH_EMBED_ONNX_EXPORT=1 \
CODE_SEARCH_DEVICE=cpu \
uv run python scripts/bench_mcp_perf.py \
  --repo /Users/anasdayeh/Downloads/ADS_Website \
  --out "$CODE_SEARCH_STORAGE/logs/bench_after_onnx.json"
```

**Step 2: Run ONNX+quant benchmark**

Run:
```bash
CODE_SEARCH_EMBED_BACKEND=onnx \
CODE_SEARCH_EMBED_ONNX_EXPORT=1 \
CODE_SEARCH_EMBED_ONNX_QUANTIZE=1 \
CODE_SEARCH_DEVICE=cpu \
uv run python scripts/bench_mcp_perf.py \
  --repo /Users/anasdayeh/Downloads/ADS_Website \
  --out "$CODE_SEARCH_STORAGE/logs/bench_after_onnx_quant.json"
```

**Step 3: Record after numbers in plan**

Update this plan file with after values from JSON.

**Step 4: Commit**

```bash
git add docs/plans/2026-02-03-onnx-embeddings-backend.md
git commit -m "docs: record ONNX perf numbers"
```

### Workstream 1 Risks and mitigations

- Risk: ONNX export fails for specific models.
- Mitigation: fallback to torch backend if export fails; log error and continue.

- Risk: Quantized outputs diverge too much.
- Mitigation: enforce similarity sanity check thresholds; disable quantization on failure.

- Risk: ONNX/int8 is a CPU optimization path and may not improve end-to-end latency for this workload.
- Mitigation: only claim performance differences after fixed-mode benchmarks; decide per-measurement.

### Workstream 1 Tests

- `uv run python -m pytest tests/unit/test_sentence_transformer_backend.py -v`
- `uv run python -m pytest tests/unit/test_bench_harness_schema.py -v`
- `CODE_SEARCH_EMBED_BACKEND=onnx CODE_SEARCH_EMBED_ONNX_EXPORT=1 uv run python scripts/bench_mcp_perf.py --repo /path/to/repo --out "$CODE_SEARCH_STORAGE/logs/bench_smoke.json"`

### Workstream 1 Rollback

- Set `CODE_SEARCH_EMBED_BACKEND=torch` and disable ONNX flags.
- Revert changes to `embeddings/sentence_transformer.py` and benchmark script.
- Delete ONNX artifacts under model cache if needed.

### Workstream 1 Benchmarks

- Baseline: `bench_baseline_torch.json`
- After ONNX: `bench_after_onnx.json`
- After ONNX+quant: `bench_after_onnx_quant.json`

Quality acceptance criteria (must pass):
- Cosine similarity sanity check on fixed sample set (mean and p50 above threshold; set in bench script)
- Top-k overlap check vs torch baseline on fixed query set (overlap >= threshold)

Note: ONNX/int8 is a CPU optimization path and must be benchmarked against the actual workload; do not assume gains.

Populate this table after measurements:

```
Metric | Baseline (torch) | ONNX | ONNX+quant
embedding_throughput_chunks_per_sec | TBD | TBD | TBD
indexing_time_seconds | TBD | TBD | TBD
search_latency_ms_p50 | TBD | TBD | TBD
search_latency_ms_p95 | TBD | TBD | TBD
quant_similarity_mean | n/a | n/a | TBD
```

---

## Workstream 2: Hugging Face Hub v1 download improvements

### Task 1: Verify versions and upgrade huggingface_hub if needed

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Step 1: Confirm current version**

Run: `python - <<'PY'\nimport tomllib, pathlib\nlock=tomllib.loads(pathlib.Path('uv.lock').read_text())\nprint([p for p in lock.get('package',[]) if p.get('name')=='huggingface-hub'])\nPY`

**Step 2: Update dependency**

Update `pyproject.toml` to `huggingface-hub>=1.0` if acceptable.

**Step 3: Update lockfile**

Run: `uv lock`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: upgrade huggingface_hub to v1"
```

### Task 2: Add HF_XET_HIGH_PERFORMANCE and cold start logging

**Files:**
- Modify: `mcp_server/server.py`
- Modify: `embeddings/sentence_transformer.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
# tests/unit/test_model_load_timing.py
# Assert log contains "model_load_seconds" when model is loaded
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_model_load_timing.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# mcp_server/server.py
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

# embeddings/sentence_transformer.py
start = time.time()
model = SentenceTransformer(...)
self._logger.info("model_load_seconds=%.2f cached=%s", time.time()-start, self._is_model_cached())
```

**Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_model_load_timing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/server.py embeddings/sentence_transformer.py README.md tests/unit/test_model_load_timing.py
git commit -m "feat: enable HF Xet high performance and log cold start"
```

### Workstream 2 Risks and mitigations

- Risk: `huggingface_hub` v1 changes break older code.
- Mitigation: run MCP indexing smoke test and revert if failure.

### Workstream 2 Tests

- `uv run python -m pytest tests/unit/test_model_load_timing.py -v`
- `uv run python scripts/index_repo.py /path/to/repo`

### Workstream 2 Rollback

- Revert `pyproject.toml` and `uv.lock` to prior versions.
- Remove `HF_XET_HIGH_PERFORMANCE` default if it causes issues.

---

## Workstream 3: Stability and hygiene upgrades

### Task 1: FastMCP pinning

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Step 1: Update dependency**

Set `fastmcp>=2.14.4,<3`.

**Step 2: Update lockfile**

Run: `uv lock`

**Step 3: Run MCP tool tests**

Run:
- `uv run python -m pytest tests/unit/test_mcp_tools_search_code.py -v`
- `uv run python -m pytest tests/unit/test_mcp_tool_descriptions.py -v`

**Step 4: Run real MCP indexing**

Run:
```bash
uv run python scripts/index_repo.py /path/to/medium/repo
```

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: pin fastmcp 2.14.4"
```

### Task 2: tree-sitter bump

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Step 1: Update dependency**

Set `tree-sitter>=0.25.4`.

**Step 2: Update lockfile**

Run: `uv lock`

**Step 3: Run chunking regression check**

Run:
```bash
CODE_SEARCH_DEVICE=cpu uv run python scripts/index_repo.py /path/to/medium/repo
```

**Step 4: Inspect stats**

Check `CODE_SEARCH_STORAGE/projects/<id>/index/stats.json` for chunk counts.

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump tree-sitter to 0.25.4"
```

### Workstream 3 Risks and mitigations

- Risk: FastMCP minor version breaks.
- Mitigation: pin `<3`, run MCP tool tests before release.

- Risk: tree-sitter parsing changes alter chunk counts.
- Mitigation: run regression on a known repo and compare counts.

### Workstream 3 Tests

- `uv run python -m pytest tests/unit/test_mcp_tools_search_code.py -v`
- `uv run python -m pytest tests/unit/test_mcp_tool_descriptions.py -v`
- `uv run python scripts/index_repo.py /path/to/medium/repo`

### Workstream 3 Rollback

- Revert `pyproject.toml` and `uv.lock` to previous versions.
- Reindex if needed.

---

## Notes

- Do not attempt Transformers v5 upgrade in this plan.
- Always record benchmarks before and after changes. No performance claims without numbers.
