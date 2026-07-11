# Gate-C embedding bake-off — how it works & how to run

Quality-first comparison of embedding models for code search on a **frozen BRIDGE
corpus**, judged **blind** by an agent. The default chunk dump uses the same
2,048-character production formatter as `CodeEmbedder`, so every arm receives the
same bounded input. Use `dump_chunks.py --input-mode raw` only for an explicitly
separate native-context experiment.

## One command (run at go-ahead)

```bash
cd /Volumes/Anas_2TB/AI/code-search/repos/claude-context-local
uv run python scripts/run_bakeoff.py --dry-run     # show the plan, run nothing
uv run python scripts/run_bakeoff.py               # run all pending arms + rerank + blind
```

Runs each arm **sequentially** under an exclusive advisory lock (never two model
jobs from this harness at once), light→heavy, **skipping only fingerprint-valid
complete artifacts**. Corpus, queries, configuration, runner code, and arm settings
are fingerprinted; stale or legacy JSON is rerun. Between Metal workloads the
orchestrator waits for stable unified-memory headroom instead of immediately loading
the next model. Flags: `--only <labels>`,
`--skip <labels>`, `--force`,
`--no-rerank`, `--no-blind`. For an unattended/overnight run, launch it as a
background job and read `benchmarks/arms/arm_*.log`.

## The arms (edit `benchmarks/arms.yaml` — no code changes to add/tune)

| label | runtime | model | notes |
|---|---|---|---|
| gemma | torch | google/embeddinggemma-300m | baseline / current default (768-d) |
| qwen_mlx | mlx | Qwen3-Embedding-4B-4bit-DWQ | fast/light 4-bit (2560-d) |
| bge_code | torch | BAAI/bge-code-v1 | code specialist, last-token pooling |
| nomic_code_gguf | gguf | nomic-embed-code Q6_K | code specialist (7B) via llama.cpp |
| qwen_bf16 | torch | Qwen3-Embedding-4B | full-precision (~8 GB, heaviest) |
| reranker | torch | Qwen3-Reranker-0.6B | official generative CrossEncoder; reranks the real top-N |

Each arm carries its **card-verified** query instruction / pooling in `arms.yaml`.
`${var}` interpolates paths; `BENCH_PATH_<KEY>` env overrides any path.

## Scripts

- `dump_chunks.py` — canonical production-formatted chunk set via the server's
  MultiLanguageChunker and shared embedding formatter.
- `bench_artifacts.py` — schema-v2 fingerprints, reuse admission, atomic JSON writes.
- `bench_dataset.py` — label admission, MRR/nDCG, paired exact tests and bootstrap CIs.
- `bench_arm_torch.py` — torch arms (gemma/qwen_bf16/bge). `--arm <label>`.
- `bench_arm_mlx.py` — MLX arm (runs in `.venv-mlx`).
- `bench_arm_gguf.py` — GGUF arm (runs in `.venv-gguf`).
- `rerank_arm.py` — reranks a base arm's candidates (full chunk text, not excerpt).
- `_arm_core.py` — shared FAISS + hit-scoring + summary (torch-free; numpy+faiss only).
- `bench_common.py` — config loader (arms.yaml), logging, RAM sampler, loaders.
- `bench_blind.py` — `make` a label-stripped blind report + key; `score` a verdict.
- `bench_model_ab.py` — `render` N-way side-by-side; also the older index-pipeline `run`.
- `run_bakeoff.py` — orchestrator (the one command).

## Venvs (isolated on purpose)

- main `.venv` — torch server + torch arms (transformers 4.57.6 pinned).
- `.venv-mlx` — MLX stack; **transformers must be <5** (mlx_lm/mlx_embeddings crash on 5.x).
- `.venv-gguf` — llama-cpp-python (Metal build) for GGUF embeddings.

## Gotchas already solved (don't rediscover)

- **faiss must import AFTER the torch model** on macOS/MPS, or model load segfaults
  (libomp double-init, exit 139). `_arm_core` imports faiss lazily; torch arms load the
  model first.
- **backend=torch** for the torch arms — the wrapper's default `onnx` backend triggers a
  fragile onnx export that segfaults on gemma.
- **MLX transformers pin <5** — the whole mlx stack calls the pre-5.x AutoTokenizer API.
- **nomic GGUF pooling** — llama-cpp-python doesn't auto-read it; runner sets
  `pooling_type=LAST` explicitly (nomic-embed-code uses last-token, not mean).
- Every run records **RAM telemetry** so speed is never confused with contention.
- Gate B is enforced. Non-finite, wrong-dimension, non-normalizable, or materially
  nondeterministic output fails the arm.
- Torch arms report the **actual** device. An MPS→CPU fallback is retained for
  quality analysis but marked degraded and excluded from accelerator-speed claims.
- `run_bakeoff.py` returns nonzero when an arm, reranker, or report stage fails.
- Interrupt handling targets only the subprocess group owned by the harness; it never
  uses application-wide `pkill` patterns.
- Arm JSON and generated reports are written through atomic replacement, and renderers
  reject runs with different corpus fingerprints or query ordering.
- Sentence Transformers 5.6+ is required so the official Qwen causal-LM reranker is
  scored through its native `LogitScore` adapter rather than a random classifier head.

## Interpreting results

- Primary agent metrics: recall@5/10, MRR, nDCG and reranker lift. Hit@1 remains
  useful but is not the sole selection criterion.
- Reports include paired discordance, exact p-values, and bootstrap confidence
  intervals. Small inconclusive differences are not called winners.
- Query labels must exist in the sampled corpus and must not match an excessive
  fraction of files/chunks. Broad labels such as `Data`, `CV`, or `Month` are rejected.
- `quality_state=semantic_degraded` means the same model completed through a device
  fallback. `fts_degraded` means semantic retrieval was unavailable and FTS answered.

## Status (2026-07-10)

All artifacts produced by the pre-schema-v2 harness are provisional. In particular,
the earlier BGE arm silently fell back to CPU and the earlier reranker received only
five candidates despite requesting twenty. Regenerate the production-formatted corpus
and rerun the hardened harness before choosing a long-term default.
