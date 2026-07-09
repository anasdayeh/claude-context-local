#!/usr/bin/env python3
"""GGUF arm runner (isolated .venv-gguf) for the Gate-C A/B.

Embeds the canonical chunk dump + labeled queries with a llama.cpp GGUF embedding
model (default nomic-embed-code Q6_K) and emits the standard run JSON. Sets the
pooling type EXPLICITLY (llama-cpp-python does not read nomic's pooling from the
GGUF — verified caveat), applies the card-correct query prefix, and truncates
over-long texts to fit the context window. Runs in .venv-gguf, never the torch venv.

Usage (from repo root):
  HF_HOME=/Volumes/Anas_Repos/AppState/caches/huggingface \
  /Volumes/Anas_2TB/AI/code-search/.venv-gguf/bin/python scripts/bench_arm_gguf.py \
     --arm nomic_code_gguf [--limit 50]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="nomic_code_gguf")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap #chunks for a smoke run")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-threads", type=int, default=6)
    ap.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = offload all to Metal")
    ap.add_argument("--max-chars", type=int, default=8000, help="truncate texts to fit n_ctx")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_common as bc
    import _arm_core as core

    cfg = bc.load_config(a.config)
    arm = bc.get_arm(cfg, a.arm)
    if arm.get("runtime") != "gguf":
        raise SystemExit(f"arm '{a.arm}' runtime={arm.get('runtime')!r}, expected gguf")
    paths = cfg["paths"]
    defaults = cfg.get("defaults", {})
    os.environ.setdefault("HF_HOME", paths["hf_home"])
    log = bc.setup_logging(f"arm_{a.arm}", out_dir=paths["out_dir"], level=defaults.get("log_level"))

    from huggingface_hub import hf_hub_download
    gguf_path = hf_hub_download(arm["model_id"], arm["gguf_file"])
    log.info("gguf: %s", gguf_path)

    import llama_cpp
    from llama_cpp import Llama
    pooling = (arm.get("pooling") or "last").lower()
    ptype = {
        "none": getattr(llama_cpp, "LLAMA_POOLING_TYPE_NONE", 0),
        "mean": getattr(llama_cpp, "LLAMA_POOLING_TYPE_MEAN", 1),
        "cls": getattr(llama_cpp, "LLAMA_POOLING_TYPE_CLS", 2),
        "last": getattr(llama_cpp, "LLAMA_POOLING_TYPE_LAST", 3),
    }.get(pooling, getattr(llama_cpp, "LLAMA_POOLING_TYPE_LAST", 3))

    t0 = time.perf_counter()
    llm = Llama(
        model_path=gguf_path, embedding=True, n_ctx=a.n_ctx,
        n_gpu_layers=a.n_gpu_layers, n_threads=a.n_threads,
        pooling_type=ptype, verbose=False,
    )
    load_s = time.perf_counter() - t0
    log.info("loaded llama.cpp in %.1fs (pooling=%s, n_ctx=%d, gpu_layers=%d)",
             load_s, pooling, a.n_ctx, a.n_gpu_layers)

    chunks = bc.load_chunks(paths["chunk_dump"])
    if a.limit:
        chunks = chunks[:a.limit]
    queries = bc.load_jsonl(paths["queries"])
    qi = arm.get("query_instruction") or ""
    di = arm.get("doc_instruction") or ""

    def embed(texts, prefix):
        out = []
        for i, t in enumerate(texts):
            s = (prefix or "") + ((t or "")[:a.max_chars])
            e = llm.embed(s)
            # llm.embed returns a flat list[float] when pooling is enabled
            out.append(np.asarray(e, dtype=np.float32).reshape(-1))
            if i and i % 200 == 0:
                log.info("  embedded %d/%d", i, len(texts))
        return np.vstack(out)

    with bc.RamSampler() as ram:
        te = time.perf_counter()
        doc_vecs = embed([c.get("text") or "" for c in chunks], di)
        q_vecs = embed([r["query"] for r in queries], qi)
        dv2 = embed([c.get("text") or "" for c in chunks[:8]], di)  # determinism probe
        embed_s = time.perf_counter() - te

    det = bool(np.allclose(doc_vecs[:8], dv2, atol=1e-3))
    gate = core.gate_b_checks(doc_vecs, expected_dim=arm.get("expected_dim"))
    gate["deterministic"] = det
    log.info("GATE B: %s", gate)
    if not gate["all_finite"]:
        log.error("NON-FINITE embeddings — check pooling_type / prefixes for %s", a.arm)

    payload = core.evaluate(
        label=a.arm, model_key=arm["model_id"], model_name=arm["gguf_file"],
        backend="gguf(llama.cpp)", device="metal", dim=int(doc_vecs.shape[1]),
        chunks=chunks, doc_vecs=doc_vecs, queries=queries, query_vecs=q_vecs,
        k=defaults.get("k", 5), embed_seconds=embed_s,
        telemetry={**ram.stat(), "load_seconds": round(load_s, 1), "gate_b": gate},
        out_path=str(Path(paths["out_dir"]) / f"arm_{a.arm}.json"),
    )
    s = payload["summary"]
    log.info("[%s] hit@1=%s hit@5=%s dim=%s embed=%.0fs ram_min=%s%%",
             a.arm, s["hit_at_1_rate"], s["hit_at_k_rate"], s["embedding_dimension"],
             embed_s, ram.stat()["ram_free_pct_min"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
