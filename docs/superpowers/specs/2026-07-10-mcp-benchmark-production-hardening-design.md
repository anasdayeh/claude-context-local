# MCP Benchmark and Production Hardening Design

## Goal

Make the local code-search MCP trustworthy for Codex and Claude by preserving useful fallback behavior while ensuring benchmark comparisons, runtime metadata, and model-selection conclusions reflect what actually ran.

## Scope

This is a hardening of the existing architecture, not a rewrite. It covers:

- production embedding fallback and response telemetry;
- benchmark execution, artifact validity, reranking, and failure semantics;
- corpus and label validation;
- statistical reporting for small paired retrieval evaluations;
- durable single-owner orchestration suitable for a 16 GB Apple Silicon Mac;
- current official Sentence Transformers support for the Qwen3 reranker.

The existing live extra-repository run is not modified. Its outputs are considered provisional because they were produced by the pre-hardening harness.

## Architecture

### Embedding execution and fallback

`SentenceTransformerModel` remains the model wrapper. It records requested and actual backend/device plus normalized fallback events. An MPS failure retries the same operation on CPU using the same model. Callers can therefore distinguish an accelerator result from a CPU fallback without parsing log text.

Indexing and benchmark code must not report the requested device after the wrapper has fallen back. Benchmark artifacts produced after any fallback are marked degraded. Performance comparisons exclude degraded accelerator runs, while retrieval-quality data remains available with its provenance.

Production search follows this quality-preserving order:

1. semantic query embedding on the configured device;
2. the same embedding model on CPU if accelerator inference fails;
3. FTS only if semantic execution or the semantic index is unavailable.

No fallback substitutes a different embedding model for an existing vector index.

### Benchmark contract

Every arm produces an atomic JSON artifact containing:

- schema version;
- corpus, query, configuration, and code fingerprints;
- model ID and resolved revision where available;
- runtime/library versions;
- requested and actual device;
- fallback/degradation events;
- gate results;
- retrieval candidates to at least the configured reranking depth.

An existing artifact is reusable only when it is complete and its fingerprint matches the requested run. A stale, corrupt, failed, or incompatible artifact is rerun.

Gate B is executable policy rather than descriptive telemetry. Non-finite vectors, wrong dimensions, inability to normalize, or failed determinism cause the arm to fail. The orchestrator returns nonzero when required arms, reranking, or report generation fail.

### Reranking

The retriever persists `candidate_k = max(report_k, reranker.top_n)`. Metrics remain calculated at the report cutoffs. The reranker consumes the real top-N candidate set and emits top-k results.

The project upgrades from Sentence Transformers 5.2.3 to a current compatible release with native generative CrossEncoder support and uses the official `Qwen/Qwen3-Reranker-0.6B` checkpoint. A model-load smoke test rejects randomly initialized classification heads and verifies score ordering on a tiny relevance fixture.

### Corpus and evaluation quality

Corpus preparation uses segment-based ignore rules shared with production chunking. Sampling is deterministic and emits a manifest containing the seed, source hash, filters, counts, and selected-file coverage.

Expected-file labels are validated against the sampled corpus. Broad labels that match too many files or chunks are rejected. Reports include hit@1, recall@5, recall@10, MRR, nDCG, per-language breakdowns, paired discordance, exact paired tests, and bootstrap confidence intervals. A model is called a winner only when both practical and statistical thresholds are met; otherwise the report presents a Pareto set.

### Orchestration

A single Python orchestrator owns the full queue. An advisory file lock prevents concurrent model runs. It fails closed if the lock cannot be acquired; it never starts a second GPU workload after an arbitrary timeout. Run IDs and per-run logs replace synchronization by searching for an old `DRIVER DONE` string.

Subprocesses run sequentially, receive an explicit environment, and are terminated as a process group when the orchestrator is interrupted. Status is persisted after every stage so interrupted runs resume from fingerprint-valid artifacts.

## Testing

Tests cover:

- MPS-to-CPU fallback metadata and actual-device reporting;
- strict Gate B failures;
- candidate depth independent from metric cutoff;
- reranking a genuine top-N set;
- fingerprint matching, stale detection, and atomic writes;
- nonzero orchestration exit status on any required failure;
- exclusive lock behavior;
- expected-file coverage and broad-label rejection;
- paired statistics and no-winner behavior on inconclusive samples;
- official Qwen reranker smoke inference when the model cache/runtime is available;
- end-to-end MCP responses for semantic, CPU-degraded semantic, and FTS-degraded search.

## Safety and privacy

Private document contents and queries are not copied into the repository or logs. Artifacts containing Pre-Work content remain on the external drive. Logs contain hashes, counts, sanitized failures, and permitted paths only.

