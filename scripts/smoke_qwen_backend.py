#!/usr/bin/env python3
"""Gate B smoke test for a new embedding backend (default: Qwen3-Embedding-4B).

Validates a backend on a small sample BEFORE any expensive full re-index:
  - no NaN/Inf in document or query embeddings
  - embedding dimension is exactly the expected value (Qwen = 2560)
  - vectors are unit-normalised after faiss.normalize_L2 (the real index pipeline)
  - re-encoding identical text is deterministic
  - records load/embed latency and peak RSS (the 16GB-M1 memory concern)

Run: uv run python scripts/smoke_qwen_backend.py [model_key]
     SMOKE_EXPECTED_DIM=2560 (default)
Exit 0 = PASS, 1 = FAIL.
"""
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make the repo importable when run as `python scripts/smoke_qwen_backend.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen3-embedding-4b"
    expected_dim = int(os.environ.get("SMOKE_EXPECTED_DIM", "2560"))
    max_docs = int(os.environ.get("SMOKE_MAX_DOCS", "80"))
    doc_chars = int(os.environ.get("SMOKE_DOC_CHARS", "2000"))
    try:
        import psutil
        proc = psutil.Process()
    except Exception:
        proc = None

    from embeddings.embedding_models_register import AVAILABLE_MODELS
    if model_key not in AVAILABLE_MODELS:
        print(f"FAIL: model '{model_key}' not in AVAILABLE_MODELS")
        return 1

    print(f"[smoke] loading {model_key} (this loads the full model into memory) ...")
    t0 = time.time()
    model = AVAILABLE_MODELS[model_key]()
    _ = model.model  # force the lazy load now so timing is honest
    t_load = time.time() - t0

    # Sample real documents from the repo's own source.
    docs = []
    for f in sorted(glob.glob("**/*.py", recursive=True)):
        if "/.venv" in f or f.startswith(".venv") or "/tests/" in f:
            continue
        try:
            docs.append(open(f, encoding="utf-8").read()[:doc_chars])
        except Exception:
            pass
        if len(docs) >= max_docs:
            break

    queries = [
        "Where is the KPI export generated?",
        "Which file defines the validation rule for dockless records?",
        "Where is this exception raised?",
        "Which tests cover this API route?",
        "Which config controls this integration?",
        "Find the CV paragraph where I describe data analysis in local government.",
        "Find the document where I mention social-services digital transformation.",
        "Find prior wording about transport, BRIDGE, KPIs, automation, or governance.",
    ]

    print(f"[smoke] embedding {len(docs)} docs + {len(queries)} queries ...")
    t1 = time.time()
    doc_emb = np.asarray(model.encode_document(docs), dtype=np.float32)
    q_emb = np.asarray(model.encode_query(queries), dtype=np.float32)
    t_embed = time.time() - t1
    q_emb2 = np.asarray(model.encode_query(queries), dtype=np.float32)  # determinism probe

    checks = []

    def check(name, ok):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    check("no NaN/Inf in document embeddings", bool(np.isfinite(doc_emb).all()))
    check("no NaN/Inf in query embeddings", bool(np.isfinite(q_emb).all()))
    check(f"document dim == {expected_dim}", doc_emb.shape[1] == expected_dim)
    check(f"query dim == {expected_dim}", q_emb.shape[1] == expected_dim)

    # Unit-norm check mirrors the real indexer (faiss.normalize_L2 before add/search).
    try:
        import faiss
        dcopy = doc_emb.copy()
        faiss.normalize_L2(dcopy)
        norms = np.linalg.norm(dcopy, axis=1)
        check("unit-norm after faiss.normalize_L2", bool(np.allclose(norms, 1.0, atol=1e-3)))
    except Exception as e:
        check(f"faiss normalize available ({e})", False)

    check("query re-encode is deterministic", bool(np.allclose(q_emb, q_emb2, atol=1e-4)))

    n = len(docs) + len(queries)
    rss = (proc.memory_info().rss / 1e9) if proc else float("nan")
    print(f"[smoke] load={t_load:.1f}s  embed={t_embed:.1f}s  "
          f"throughput={n / t_embed:.1f} texts/s  peak_rss~{rss:.1f} GB")

    ok = all(checks)
    print("SMOKE RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
