"""Shared, torch-free core for A/B arm runners.

Given document + query vectors over a canonical chunk set, build a cosine
(IndexFlatIP over L2-normalised vectors) index, run each labeled query, and emit
the standard run JSON that bench_model_ab.py `render` and bench_blind.py consume.
Depends ONLY on numpy + faiss, so the SAME evaluation code runs in the torch
server venv AND the isolated MLX venv — the embedder is the only thing that varies
between arms.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# faiss is imported LAZILY inside the functions below — never at module load.
# Reason (verified): on macOS/MPS, if faiss's OpenMP runtime initialises BEFORE a
# torch model is loaded, loading that model segfaults (libomp double-init, exit
# 139). Loading the torch model + encoding FIRST, then importing faiss, is clean
# (exit 0). The torch arm runner imports this module at the top, so faiss must not
# come in until evaluate()/normalize() run — after the model is already resident.


def _hit(paths, expected):
    return any(any(exp in (p or "") for p in paths) for exp in expected)


def normalize(mat):
    import faiss  # lazy — see module header note on libomp/MPS segfault
    mat = np.ascontiguousarray(np.asarray(mat, dtype=np.float32))
    faiss.normalize_L2(mat)
    return mat


def gate_b_checks(doc_vecs, expected_dim=None):
    """Gate-B smoke results: finite (no NaN/Inf), dimension, unit-norm-ability."""
    v = np.asarray(doc_vecs, dtype=np.float32)
    dim = int(v.shape[1]) if v.ndim == 2 else None
    norms = np.linalg.norm(v, axis=1) if v.ndim == 2 else np.array([])
    normed = normalize(v.copy()) if v.ndim == 2 else v
    unit = bool(np.allclose(np.linalg.norm(normed, axis=1), 1.0, atol=1e-3)) if v.ndim == 2 else False
    return {
        "all_finite": bool(np.isfinite(v).all()),
        "dim": dim,
        "dim_ok": (expected_dim is None or dim == expected_dim),
        "unit_norm_after_normalize": unit,
        "raw_norm_min": float(norms.min()) if norms.size else None,
        "raw_norm_max": float(norms.max()) if norms.size else None,
    }


def determinism_check(reference, repeat, min_cosine=0.9999):
    """Compare repeated embeddings without demanding bitwise accelerator output."""
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(repeat, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2 or not a.size:
        return {"deterministic": False, "min_cosine": None, "max_abs_diff": None}
    an = normalize(a.copy())
    bn = normalize(b.copy())
    cosine = np.sum(an * bn, axis=1)
    minimum = float(np.min(cosine))
    return {
        "deterministic": bool(np.isfinite(cosine).all() and minimum >= min_cosine),
        "min_cosine": minimum,
        "max_abs_diff": float(np.max(np.abs(a - b))),
    }


def validate_gate_b(gate):
    """Raise when a Gate-B invariant fails instead of treating it as telemetry."""
    failures = []
    if not gate.get("all_finite"):
        failures.append("finite embeddings")
    if not gate.get("dim_ok"):
        failures.append("embedding dimension")
    if not gate.get("unit_norm_after_normalize"):
        failures.append("unit-normal vectors")
    if not gate.get("deterministic"):
        failures.append("deterministic embeddings")
    if failures:
        raise ValueError("Gate B failed: " + ", ".join(failures))


def summarize(*, label, model_key, model_name, backend, device, dim, chunks_count,
              per_query, k=5, index_seconds=None, embed_seconds=None,
              telemetry=None, out_path=None, candidate_k=None, fingerprint=None):
    """Build the standard run payload {summary, per_query} from an already-scored
    per_query list (each entry has hit_at_1/hit_at_k/lang). Shared by evaluate()
    (vector arms) and rerank_arm.py (which reorders existing candidates)."""
    n = len(per_query) or 1
    langs = sorted({p.get("lang", "?") for p in per_query})
    per_lang = {}
    for lg in langs:
        rows = [p for p in per_query if p.get("lang", "?") == lg]
        m = len(rows) or 1
        per_lang[lg] = {
            "n": len(rows),
            "hit_at_1_rate": round(sum(p["hit_at_1"] for p in rows) / m, 3),
            "hit_at_k_rate": round(sum(p["hit_at_k"] for p in rows) / m, 3),
            "recall_at_10_rate": round(sum(p.get("hit_at_10", p["hit_at_k"]) for p in rows) / m, 3),
        }
    summary = {
        "label": label, "model_key": model_key, "model_name": model_name,
        "embedding_dimension": dim, "backend": backend, "device": device,
        "repo": "chunk_dump", "k": k, "candidate_k": candidate_k or k,
        "index_seconds": round(index_seconds, 2) if index_seconds else None,
        "embed_seconds": round(embed_seconds, 2) if embed_seconds else None,
        "chunks_added": chunks_count,
        "hit_at_1_rate": round(sum(p["hit_at_1"] for p in per_query) / n, 3),
        "hit_at_k_rate": round(sum(p["hit_at_k"] for p in per_query) / n, 3),
        "recall_at_10_rate": round(
            sum(p.get("hit_at_10", p["hit_at_k"]) for p in per_query) / n, 3
        ),
        "mean_latency_ms": round(sum(p.get("latency_ms", 0) for p in per_query) / n, 1),
        "query_count": len(per_query),
        "per_lang": per_lang,
        "telemetry": telemetry or {},
    }
    from bench_dataset import ranking_metrics
    summary.update(ranking_metrics(per_query, k=k))
    payload = {
        "artifact": {
            "schema_version": 2,
            "status": "complete",
            "fingerprint": fingerprint,
        },
        "summary": summary,
        "per_query": per_query,
    }
    if out_path:
        from bench_artifacts import atomic_write_json
        atomic_write_json(out_path, payload)
    return payload


def evaluate(*, label, model_key, model_name, backend, device, dim,
             chunks, doc_vecs, queries, query_vecs, k=5,
             index_seconds=None, embed_seconds=None, telemetry=None,
             out_path=None, candidate_k=None, fingerprint=None):
    """chunks aligned to doc_vecs rows; queries aligned to query_vecs rows."""
    import faiss  # lazy — see module header note on libomp/MPS segfault
    doc = normalize(doc_vecs)
    qv = normalize(query_vecs)
    index = faiss.IndexFlatIP(doc.shape[1])
    index.add(doc)

    candidate_k = max(k, int(candidate_k or k))
    per_query = []
    for qi, row in enumerate(queries):
        t = time.perf_counter()
        D, I = index.search(qv[qi:qi + 1], candidate_k)
        latency_ms = (time.perf_counter() - t) * 1000.0
        hits = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
            if idx < 0:
                continue
            c = chunks[idx]
            content = c.get("text") or ""
            excerpt = " ".join(content.split())[:220]
            hits.append({
                "rank": rank,
                "path": c.get("path"),
                "score": round(float(score), 4),
                "lines": f"{c.get('start_line')}-{c.get('end_line')}",
                "name": c.get("name"),
                "excerpt": excerpt,
            })
        paths = [h["path"] for h in hits]
        expected = row.get("expected_files", [])
        per_query.append({
            "query": row["query"],
            "expected_files": expected,
            "notes": row.get("notes", ""),
            "lang": row.get("lang", "?"),
            "hit_at_1": _hit(paths[:1], expected),
            "hit_at_k": _hit(paths[:k], expected),
            "hit_at_10": _hit(paths[:10], expected),
            "top1_path": paths[0] if paths else None,
            "latency_ms": round(latency_ms, 1),
            "results": hits,
        })

    return summarize(
        label=label, model_key=model_key, model_name=model_name, backend=backend,
        device=device, dim=int(doc.shape[1]), chunks_count=len(chunks),
        per_query=per_query, k=k, index_seconds=index_seconds,
        embed_seconds=embed_seconds, telemetry=telemetry, out_path=out_path,
        candidate_k=candidate_k, fingerprint=fingerprint,
    )
