# MCP Benchmark and Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Harden the existing code-search MCP and model bake-off so fallbacks remain useful, telemetry is truthful, reranking is valid, and model-selection reports are reproducible and statistically defensible.

**Architecture:** Extend the current embedding wrapper with structured execution telemetry; make the shared benchmark core own gate enforcement, candidate depth, fingerprints, and atomic artifacts; and make the existing Python orchestrator the single locked owner of model subprocesses. Keep production and benchmark policies distinct while sharing normalized telemetry and validation helpers.

**Tech Stack:** Python 3.13+, pytest, Sentence Transformers 5.6, Transformers 4.57, PyTorch 2.9-2.13/MPS, NumPy, FAISS, PyYAML, stdlib `fcntl` and `hashlib`.

## Global Constraints

- Preserve the existing MCP tool names and response compatibility.
- Production search falls back in this order: requested semantic device, same model on CPU, then FTS.
- Never substitute a different embedding model for an existing vector index.
- Never claim accelerator timing when any part of the arm ran on CPU.
- Do not log private query or document contents.
- Do not interrupt the pre-existing live benchmark while developing in the isolated worktree.
- Use official Qwen and Sentence Transformers sources rather than a community-converted reranker.

---

### Task 1: Structured embedding execution telemetry

**Files:**
- Modify: `embeddings/sentence_transformer.py`
- Modify: `embeddings/embedding_model.py`
- Test: `tests/unit/test_sentence_transformer_fallback.py`

**Interfaces:**
- Produces: `get_execution_info() -> dict[str, Any]` with requested/actual device and normalized fallback events.
- Preserves: `encode`, `encode_query`, `encode_document`, and `get_model_info` public behavior.

- [x] Write tests that force an MPS encode failure and assert CPU retry, actual-device reporting, and one `mps_to_cpu` event.
- [x] Run the focused tests and verify the new assertions fail against the current wrapper.
- [x] Implement structured state updates without changing vector values or swallowing the second failure.
- [x] Run the focused tests and the existing embedder unit tests.

### Task 2: Enforce benchmark gates and persist real candidate depth

**Files:**
- Modify: `scripts/_arm_core.py`
- Modify: `scripts/bench_arm_torch.py`
- Modify: `scripts/bench_arm_mlx.py`
- Modify: `scripts/bench_arm_gguf.py`
- Test: `tests/unit/test_benchmark_core.py`

**Interfaces:**
- Produces: `validate_gate_b(gate: dict) -> None`.
- Extends: `evaluate(..., k: int, candidate_k: int | None = None)` while keeping metric cutoffs at `k`.

- [x] Write failing tests for nondeterminism, dimension mismatch, non-finite vectors, and `candidate_k > k`.
- [x] Verify each regression test fails for the intended missing behavior.
- [x] Add strict validation, separate candidate and metric cutoffs, and actual-device/degradation metadata from the model wrapper.
- [x] Run focused tests and all existing benchmark tests.

### Task 3: Fingerprinted and atomic benchmark artifacts

**Files:**
- Create: `scripts/bench_artifacts.py`
- Modify: `scripts/bench_common.py`
- Modify: `scripts/_arm_core.py`
- Modify: `scripts/run_bakeoff.py`
- Test: `tests/unit/test_benchmark_artifacts.py`

**Interfaces:**
- Produces: `build_run_fingerprint(...) -> dict`, `artifact_is_reusable(path, expected) -> bool`, and `atomic_write_json(path, payload) -> None`.

- [x] Write failing tests for matching reuse, changed-query invalidation, corrupt output, incomplete output, and atomic replacement.
- [x] Run focused tests and verify failure reasons.
- [x] Implement SHA-256 file/config/code fingerprints and schema/completion markers.
- [x] Replace existence-only skipping and direct JSON writes with validated reuse and atomic writes.
- [x] Run artifact, orchestrator, and report tests.

### Task 4: Valid official Qwen reranking

**Files:**
- Modify: `pyproject.toml`
- Modify: `benchmarks/arms.yaml`
- Modify: `scripts/rerank_arm.py`
- Test: `tests/unit/test_rerank_arm.py`
- Test: `tests/integration/test_qwen_reranker_smoke.py`

**Interfaces:**
- Consumes: base artifacts containing at least `reranker.top_n` candidates.
- Produces: reranked artifact with `candidate_count`, actual device, fallback status, and base-run fingerprint.

- [x] Write a failing unit test proving a top-20 candidate list can promote an item initially below rank five.
- [x] Write a cache-aware integration smoke test that verifies the official checkpoint ranks an obviously relevant passage above an irrelevant one and has no random classification head warning.
- [x] Upgrade Sentence Transformers to `>=5.6.0,<5.7.0`, refresh the lock, and retain `Qwen/Qwen3-Reranker-0.6B`.
- [x] Use native `CrossEncoder.rank`/`predict` support with bounded batch size and max length.
- [x] Run unit and integration smoke tests; skip only when the model is genuinely unavailable offline.

### Task 5: Honest orchestration and exclusive execution

**Files:**
- Modify: `scripts/run_bakeoff.py`
- Test: `tests/unit/test_run_bakeoff.py`

**Interfaces:**
- Produces: nonzero exit status for required-stage failure and an exclusive advisory lock around the run.

- [x] Write failing tests for an arm failure, reranker failure, report failure, and lock contention.
- [x] Verify the current orchestrator incorrectly returns success.
- [x] Add `fcntl.flock`, run IDs, stage-state persistence, subprocess result checks, and final status aggregation.
- [x] Ensure interruption cleanup targets only owned subprocess process groups.
- [x] Run orchestrator tests and a dry-run acceptance command.

### Task 6: Corpus validation and statistical reporting

**Files:**
- Create: `scripts/bench_dataset.py`
- Modify: `scripts/bench_model_ab.py`
- Modify: `scripts/bench_blind.py`
- Test: `tests/unit/test_benchmark_dataset.py`
- Test: `tests/unit/test_benchmark_statistics.py`

**Interfaces:**
- Produces: `validate_query_labels(chunks, queries, ...)`, paired exact-test results, bootstrap intervals, MRR, and nDCG.

- [x] Write failing tests rejecting labels absent from the sample and labels matching an excessive fraction of paths/chunks.
- [x] Write failing tests for MRR/nDCG and an inconclusive paired comparison.
- [x] Implement deterministic validation and statistics without adding heavy dependencies.
- [x] Update reports to show uncertainty, degradation, and Pareto recommendations rather than unconditional winners.
- [x] Run dataset, statistics, blind, and render tests.

### Task 7: MCP response propagation and acceptance tests

**Files:**
- Modify: `mcp_server/code_search_server.py`
- Modify: `mcp_server/mcp_tools.py`
- Modify: `search/searcher.py`
- Test: `tests/unit/test_search_fallback_metadata.py`
- Test: `tests/integration/test_mcp_fallback_acceptance.py`

**Interfaces:**
- Produces stable response fields `search_mode_requested`, `search_mode_used`, `quality_state`, `requested_device`, `actual_device`, `fallback_events`, and `semantic_available`.

- [x] Write failing response-contract tests for normal semantic, CPU-degraded semantic, and FTS-degraded results.
- [x] Trace the current result construction path and add metadata at one canonical boundary.
- [x] Preserve existing `results` and `meta` compatibility while adding the new fields.
- [x] Run MCP unit/integration tests and a real stdio handshake/search acceptance harness.

### Task 8: Documentation and verification

**Files:**
- Modify: `benchmarks/README_bakeoff.md`
- Modify: `CODEX.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Documents the artifact schema, fallback states, lock behavior, reranker requirements, and interpretation rules.

- [x] Document how to run, resume, invalidate, and interpret the hardened benchmark.
- [x] Run `python -m pytest tests/unit/ -v --tb=short`.
- [x] Run the relevant integration tests and a benchmark dry run.
- [x] Run `git diff --check` and inspect the complete diff.
- [ ] Re-run regression tests after integration into the main checkout.
