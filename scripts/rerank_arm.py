#!/usr/bin/env python3
"""Reranker arm (main venv) for the Gate-C A/B.

Takes a base arm's run JSON (e.g. arm_gemma.json), and for each query rescents its
top-N FAISS candidates with a Qwen3-Reranker cross-encoder, then re-evaluates
hit@1/hit@k on the reranked order. Emits a run JSON labelled '<base>_rerank'. Tests
whether the *embedder* or the *missing rerank* is the retrieval ceiling.

Reranking uses the FULL chunk text (looked up from the canonical chunk dump), not
the truncated excerpt stored in the base JSON. No faiss is imported here (we reorder
existing candidates), so there's no libomp/torch conflict.

Usage:
  uv run python scripts/rerank_arm.py                 # base = reranker.base_arm in arms.yaml
  uv run python scripts/rerank_arm.py --base-arm gemma --top-n 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import bench_common as bc  # noqa: E402
import _arm_core as core  # noqa: E402


def rerank_order(scores, k):
    """Pure: indices of the top-k candidates by descending score."""
    order = list(np.argsort(-np.asarray(scores, dtype=float)))
    return order[:k]


def build_fulltext_index(chunk_dump_path):
    """Map (path, 'start-end') -> full chunk text, to enrich base candidates."""
    idx = {}
    for c in json.loads(Path(chunk_dump_path).read_text()):
        key = f"{c.get('path')}|{c.get('start_line')}-{c.get('end_line')}"
        idx[key] = c.get("text") or ""
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-arm", default=None, help="default: reranker.base_arm in arms.yaml")
    ap.add_argument("--base-json", default=None, help="explicit base run JSON path")
    ap.add_argument("--top-n", type=int, default=None, help="candidates to rescore")
    ap.add_argument("--device", default=os.environ.get("BENCH_DEVICE", "mps"))
    a = ap.parse_args()

    cfg = bc.load_config(a.config)
    paths = cfg["paths"]
    defaults = cfg.get("defaults", {})
    rr = cfg.get("reranker", {}) or {}
    base_label = a.base_arm or rr.get("base_arm", "gemma")
    top_n = a.top_n or rr.get("top_n", 20)
    k = defaults.get("k", 5)
    label = f"{base_label}_rerank"
    os.environ.setdefault("HF_HOME", paths["hf_home"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    log = bc.setup_logging(f"arm_{label}", out_dir=paths["out_dir"], level=defaults.get("log_level"))

    base_json = a.base_json or str(Path(paths["out_dir"]) / f"arm_{base_label}.json")
    if not Path(base_json).exists():
        raise SystemExit(f"base run JSON not found: {base_json} — run the '{base_label}' arm first")
    base = json.loads(Path(base_json).read_text())
    fulltext = build_fulltext_index(paths["chunk_dump"])
    log.info("rerank base=%s top_n=%d model=%s (%d fulltext chunks)",
             base_label, top_n, rr.get("model_id"), len(fulltext))

    from sentence_transformers import CrossEncoder
    t0 = time.perf_counter()
    # NOTE: if the base Qwen3-Reranker fails to load as a CrossEncoder, switch
    # arms.yaml reranker.model_id to tomaarsen/Qwen3-Reranker-0.6B-seq-cls.
    ce = CrossEncoder(rr["model_id"], trust_remote_code=True, device=a.device)
    load_s = time.perf_counter() - t0
    log.info("loaded reranker in %.1fs", load_s)

    def text_for(hit):
        key = f"{hit.get('path')}|{hit.get('lines')}"
        return fulltext.get(key) or hit.get("excerpt", "")

    per_query = []
    with bc.RamSampler() as ram:
        te = time.perf_counter()
        for pq in base["per_query"]:
            q = pq["query"]
            cands = pq.get("results", [])[:top_n]
            expected = pq.get("expected_files", [])
            if cands:
                pairs = [(q, text_for(h)[:4000]) for h in cands]
                scores = ce.predict(pairs, show_progress_bar=False)
                order = rerank_order(scores, k)
                reranked = [{**cands[i], "rank": r + 1, "rerank_score": round(float(scores[i]), 4)}
                            for r, i in enumerate(order)]
            else:
                reranked = []
            rpaths = [h["path"] for h in reranked]
            per_query.append({
                "query": q,
                "expected_files": expected,
                "notes": pq.get("notes", ""),
                "lang": pq.get("lang", "?"),
                "hit_at_1": core._hit(rpaths[:1], expected),
                "hit_at_k": core._hit(rpaths, expected),
                "top1_path": rpaths[0] if rpaths else None,
                "latency_ms": pq.get("latency_ms", 0),
                "results": reranked,
            })
        rerank_s = time.perf_counter() - te

    base_dim = base["summary"].get("embedding_dimension")
    payload = core.summarize(
        label=label, model_key=rr.get("model_id"), model_name=rr.get("model_id"),
        backend=f"{base_label}+reranker", device=a.device, dim=base_dim,
        chunks_count=base["summary"].get("chunks_added"), per_query=per_query, k=k,
        embed_seconds=rerank_s,
        telemetry={**ram.stat(), "load_seconds": round(load_s, 1),
                   "base_arm": base_label, "top_n": top_n},
        out_path=str(Path(paths["out_dir"]) / f"arm_{label}.json"),
    )
    s = payload["summary"]
    log.info("[%s] hit@1=%s hit@5=%s (base hit@1=%s) rerank=%.0fs ram_min=%s%%",
             label, s["hit_at_1_rate"], s["hit_at_k_rate"],
             base["summary"].get("hit_at_1_rate"), rerank_s, ram.stat()["ram_free_pct_min"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
