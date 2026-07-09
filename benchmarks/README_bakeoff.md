# Gate-C embedding bake-off — how it works & how to run

Quality-first comparison of embedding models for code search on a **frozen BRIDGE
corpus**, judged **blind** by an agent. Every arm embeds the *identical* canonical
chunk set, so the embedding model is the only variable.

## One command (run at go-ahead)

```bash
cd /Volumes/Anas_2TB/AI/code-search/repos/claude-context-local
uv run python scripts/run_bakeoff.py --dry-run     # show the plan, run nothing
uv run python scripts/run_bakeoff.py               # run all pending arms + rerank + blind
```

Runs each arm **sequentially** (never two models resident at once — 16 GB-safe),
light→heavy, **skipping** any arm whose `benchmarks/arms/arm_<label>.json` already
exists (resumable). Flags: `--only <labels>`, `--skip <labels>`, `--force`,
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
| reranker | torch | Qwen3-Reranker-0.6B | rescages gemma's top-N |

Each arm carries its **card-verified** query instruction / pooling in `arms.yaml`.
`${var}` interpolates paths; `BENCH_PATH_<KEY>` env overrides any path.

## Scripts

- `dump_chunks.py` — one canonical chunk set (`chunk_dump.json`, 1845 chunks) via the
  server's MultiLanguageChunker, shared by all arms.
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

## Status (2026-07-09)

- Full BRIDGE Gemma **daily-driver index**: complete (730 files, 7615 chunks).
- **Gemma arm** on the corpus: done — hit@1 0.481, hit@5 0.778 (ts is weakest: 0.27).
- All other arms + reranker: **built & verified model-free; pending the go-ahead run.**
- Blind judge / render / config / orchestrator: verified on synthetic data.
