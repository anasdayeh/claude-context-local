#!/usr/bin/env python3
"""Reranker arm (main venv) for the Gate-C A/B.

Takes a base arm's run JSON (e.g. arm_gemma.json), and for each query rescents its
top-N FAISS candidates with a Qwen3-Reranker cross-encoder, then re-evaluates
hit@1/hit@k on the reranked order. Emits a run JSON labelled '<base>_rerank'. Tests
whether the *embedder* or the *missing rerank* is the retrieval ceiling.

Reranking uses the canonical production-formatted chunk text from the dump, not the
short report excerpt stored in the base JSON. No faiss is imported here (we reorder
existing candidates), so there's no libomp/torch conflict.

Usage:
  uv run python scripts/rerank_arm.py                 # base = reranker.base_arm in arms.yaml
  uv run python scripts/rerank_arm.py --base-arm gemma --top-n 20
"""
from __future__ import annotations

import argparse
import gc
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
import bench_artifacts  # noqa: E402
import _arm_core as core  # noqa: E402


def rerank_order(scores, k):
    """Pure: indices of the top-k candidates by descending score."""
    order = list(np.argsort(-np.asarray(scores, dtype=float)))
    return order[:k]


def rerank_candidates(candidates, scores, k):
    """Return the top-k candidates with stable reranker score provenance."""
    if len(candidates) != len(scores):
        raise ValueError("candidate and score counts differ")
    order = rerank_order(scores, k)
    return [
        {
            **candidates[index],
            "rank": rank + 1,
            "rerank_score": round(float(scores[index]), 4),
        }
        for rank, index in enumerate(order)
    ]


def validate_base_candidate_depth(base, top_n):
    """Reject legacy base runs that saved fewer candidates than reranking requires."""
    corpus_size = int((base.get("summary") or {}).get("chunks_added") or top_n)
    required = min(int(top_n), corpus_size)
    for row in base.get("per_query") or []:
        actual = len(row.get("results") or [])
        if actual < required:
            raise ValueError(
                f"reranker requires {required} candidates per query; "
                f"query {row.get('query')!r} has {actual}. Rerun the base arm."
            )


def build_fulltext_index(chunk_dump_path):
    """Map (path, 'start-end') -> full chunk text, to enrich base candidates."""
    idx = {}
    for c in json.loads(Path(chunk_dump_path).read_text()):
        key = f"{c.get('path')}|{c.get('start_line')}-{c.get('end_line')}"
        idx[key] = c.get("text") or ""
    return idx


def load_cross_encoder(model_id, device, max_length):
    """Load the official generative Qwen reranker through ST's native adapter."""
    import torch
    from sentence_transformers import CrossEncoder

    dtype = torch.bfloat16 if device == "mps" else torch.float32
    return CrossEncoder(
        model_id,
        trust_remote_code=True,
        device=device,
        max_length=max_length,
        model_kwargs={"dtype": dtype, "attn_implementation": "eager"},
    )


def _device_error(exc):
    message = str(exc).lower()
    return "mps" in message or "out of memory" in message or "oom" in message


def score_base(base, cross_encoder, fulltext, top_n, k, batch_size):
    def text_for(hit):
        key = f"{hit.get('path')}|{hit.get('lines')}"
        return fulltext.get(key) or hit.get("excerpt", "")

    rows = []
    for pq in base["per_query"]:
        query = pq["query"]
        candidates = pq.get("results", [])[:top_n]
        expected = pq.get("expected_files", [])
        if candidates:
            pairs = [(query, text_for(hit)) for hit in candidates]
            scores = cross_encoder.predict(
                pairs,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            reranked = rerank_candidates(candidates, scores, top_n)
        else:
            reranked = []
        paths = [hit["path"] for hit in reranked]
        rows.append({
            "query": query,
            "expected_files": expected,
            "notes": pq.get("notes", ""),
            "lang": pq.get("lang", "?"),
            "hit_at_1": core._hit(paths[:1], expected),
            "hit_at_k": core._hit(paths[:k], expected),
            "hit_at_10": core._hit(paths[:10], expected),
            "top1_path": paths[0] if paths else None,
            "latency_ms": pq.get("latency_ms", 0),
            "results": reranked,
        })
    return rows


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
    validate_base_candidate_depth(base, top_n)
    fulltext = build_fulltext_index(paths["chunk_dump"])
    log.info("rerank base=%s top_n=%d model=%s (%d fulltext chunks)",
             base_label, top_n, rr.get("model_id"), len(fulltext))

    t0 = time.perf_counter()
    actual_device = a.device
    fallback_events = []
    ce = load_cross_encoder(rr["model_id"], actual_device, int(rr.get("max_length", 2048)))
    load_s = time.perf_counter() - t0
    log.info("loaded reranker in %.1fs", load_s)

    with bc.RamSampler() as ram:
        te = time.perf_counter()
        try:
            per_query = score_base(base, ce, fulltext, top_n, k, int(rr.get("batch", 1)))
        except Exception as exc:
            if a.device != "mps" or not _device_error(exc):
                raise
            log.warning("reranker MPS failure; restarting the complete rerank on CPU: %s", exc)
            del ce
            gc.collect()
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass
            actual_device = "cpu"
            fallback_events.append({"from": "mps", "to": "cpu", "reason": "mps_error"})
            ce = load_cross_encoder(rr["model_id"], actual_device, int(rr.get("max_length", 2048)))
            per_query = score_base(base, ce, fulltext, top_n, k, int(rr.get("batch", 1)))
        rerank_s = time.perf_counter() - te

    pseudo_arm = dict(rr)
    pseudo_arm.update({
        "label": label,
        "base_fingerprint": (base.get("artifact") or {}).get("fingerprint"),
    })
    fingerprint = bench_artifacts.build_run_fingerprint(
        chunk_path=paths["chunk_dump"],
        query_path=paths["queries"],
        config_path=cfg["_config_path"],
        arm=pseudo_arm,
        source_paths=[Path(__file__), PROJECT_ROOT / "scripts" / "_arm_core.py"],
    )

    base_dim = base["summary"].get("embedding_dimension")
    payload = core.summarize(
        label=label, model_key=rr.get("model_id"), model_name=rr.get("model_id"),
        backend=f"{base_label}+reranker", device=actual_device, dim=base_dim,
        chunks_count=base["summary"].get("chunks_added"), per_query=per_query, k=k,
        candidate_k=top_n, embed_seconds=rerank_s,
        telemetry={**ram.stat(), "load_seconds": round(load_s, 1),
                   "base_arm": base_label, "top_n": top_n,
                   "requested_device": a.device, "actual_device": actual_device,
                   "fallback_events": fallback_events,
                   "degraded": bool(fallback_events)},
        fingerprint=fingerprint,
        out_path=str(Path(paths["out_dir"]) / f"arm_{label}.json"),
    )
    s = payload["summary"]
    log.info("[%s] hit@1=%s hit@5=%s (base hit@1=%s) rerank=%.0fs ram_min=%s%%",
             label, s["hit_at_1_rate"], s["hit_at_k_rate"],
             base["summary"].get("hit_at_1_rate"), rerank_s, ram.stat()["ram_free_pct_min"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
