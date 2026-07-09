#!/usr/bin/env python3
"""MLX arm runner for the Gate-C A/B (runs in the isolated .venv-mlx, NOT the
torch server venv). Embeds the canonical chunk dump + labeled queries with an
mlx_embeddings model (default Qwen3-Embedding-4B-4bit-DWQ), runs Gate-B smoke
checks, and emits the standard run JSON that bench_blind.py / bench_model_ab.py
render consume.

Usage (from repo root):
  HF_HOME=/Volumes/Anas_Repos/AppState/caches/huggingface \
  /Volumes/Anas_2TB/AI/code-search/.venv-mlx/bin/python scripts/bench_arm_mlx.py \
     --chunks benchmarks/chunk_dump.json \
     --queries benchmarks/bridge_quality_queries.jsonl \
     --out benchmarks/arm_qwen_mlx.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Qwen3-Embedding uses a query-side instruction; documents get no prefix.
QWEN_QUERY_INSTRUCTION = (
    "Instruct: Given a code search query, retrieve relevant code and "
    "documentation passages that answer the query\nQuery:"
)


def _load_jsonl(p):
    rows = []
    for line in Path(p).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


class _Ram:
    def __init__(self):
        self.s = []
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        import psutil
        while not self._stop.is_set():
            try:
                self.s.append(100.0 - psutil.virtual_memory().percent)
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        try:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        except Exception:
            pass
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    def stat(self):
        return {
            "ram_free_pct_min": round(min(self.s), 1) if self.s else None,
            "ram_free_pct_mean": round(sum(self.s) / len(self.s), 1) if self.s else None,
        }


def embed_texts(model, tok, texts, batch=8, log_every=200):
    import mlx_embeddings as me
    import mlx.core as mx
    out = []
    for i in range(0, len(texts), batch):
        res = me.generate(model, tok, texts=texts[i:i + batch])
        emb = res.text_embeds.astype(mx.float32)  # DWQ emits bf16; numpy can't read a bf16 buffer
        mx.eval(emb)
        out.append(np.array(emb))
        if i and i % log_every == 0:
            print(f"    embedded {i}/{len(texts)}", flush=True)
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
    ap.add_argument("--label", default="qwen_mlx")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap #chunks for a smoke run")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/Volumes/Anas_Repos/AppState/caches/huggingface")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _arm_core as core

    chunks = json.loads(Path(a.chunks).read_text())
    if a.limit:
        chunks = chunks[:a.limit]
    queries = _load_jsonl(a.queries)

    import mlx_embeddings as me
    t0 = time.perf_counter()
    model, tok = me.load(a.model)
    load_s = time.perf_counter() - t0
    print(f"loaded {a.model} in {load_s:.1f}s; embedding {len(chunks)} chunks + {len(queries)} queries", flush=True)

    doc_texts = [c.get("text") or "" for c in chunks]
    q_texts = [QWEN_QUERY_INSTRUCTION + r["query"] for r in queries]

    with _Ram() as ram:
        te = time.perf_counter()
        doc_vecs = embed_texts(model, tok, doc_texts, batch=a.batch)
        dv2 = embed_texts(model, tok, doc_texts[:8], batch=a.batch)  # determinism probe
        q_vecs = embed_texts(model, tok, q_texts, batch=a.batch)
        embed_s = time.perf_counter() - te

    det = bool(np.allclose(doc_vecs[:8], dv2, atol=1e-4))
    gate = core.gate_b_checks(doc_vecs, expected_dim=2560)
    gate["deterministic"] = det
    print("GATE B:", json.dumps(gate), flush=True)

    payload = core.evaluate(
        label=a.label, model_key=a.model, model_name=a.model, backend="mlx",
        device="mps/metal", dim=int(doc_vecs.shape[1]),
        chunks=chunks, doc_vecs=doc_vecs, queries=queries, query_vecs=q_vecs,
        k=a.k, embed_seconds=embed_s,
        telemetry={**ram.stat(), "load_seconds": round(load_s, 1), "gate_b": gate},
        out_path=a.out,
    )
    s = payload["summary"]
    print(f"[{a.label}] hit@1={s['hit_at_1_rate']} hit@{a.k}={s['hit_at_k_rate']} "
          f"dim={s['embedding_dimension']} embed={embed_s:.0f}s "
          f"ram_min={ram.stat()['ram_free_pct_min']}% -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
