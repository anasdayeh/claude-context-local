#!/usr/bin/env python3
"""Torch arm runner (main server venv) for the Gate-C A/B.

Embeds the canonical chunk dump + labeled queries via a registered model
(encode_document for chunks, encode_query for queries — so each model's own
query/doc convention is honoured) and emits the standard run JSON. Uses the SAME
_arm_core FAISS + scoring as the MLX / GGUF arms, so the embedder is the only
variable. All model config comes from arms.yaml.

Usage:
  uv run python scripts/bench_arm_torch.py --arm gemma
  uv run python scripts/bench_arm_torch.py --arm bge_code --limit 120   # smoke
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm label in arms.yaml (runtime must be torch)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap #chunks for a smoke run")
    ap.add_argument("--device", default=os.environ.get("BENCH_DEVICE", "mps"))
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--backend", default=os.environ.get("BENCH_BACKEND", "torch"),
                    help="torch|onnx — default torch (onnx export is fragile / can crash on gemma)")
    ap.add_argument("--out", default=None, help="override output json path")
    a = ap.parse_args()

    cfg = bc.load_config(a.config)
    arm = bc.get_arm(cfg, a.arm)
    if arm.get("runtime") != "torch":
        raise SystemExit(f"arm '{a.arm}' has runtime={arm.get('runtime')!r}, expected torch")
    paths = cfg["paths"]
    defaults = cfg.get("defaults", {})
    out_dir = paths["out_dir"]
    log = bc.setup_logging(f"arm_{a.arm}", out_dir=out_dir, level=defaults.get("log_level"))
    os.environ.setdefault("HF_HOME", paths["hf_home"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    chunks = bc.load_chunks(paths["chunk_dump"])
    if a.limit:
        chunks = chunks[:a.limit]
    queries = bc.load_jsonl(paths["queries"])
    dataset_validation = bc.validate_dataset(chunks, queries)
    k = defaults.get("k", 5)
    batch = a.batch or arm.get("batch") or defaults.get("batch", 8)
    reranker = cfg.get("reranker", {}) or {}
    candidate_k = max(k, 10, int(reranker.get("top_n", k)) if reranker.get("enabled") else k)
    log.info("arm=%s model=%s chunks=%d queries=%d device=%s batch=%d",
             a.arm, arm["model_id"], len(chunks), len(queries), a.device, batch)

    from embeddings.embedding_models_register import AVAILABLE_MODELS
    key = arm.get("model_key") or arm["model_id"]
    if key not in AVAILABLE_MODELS:
        raise SystemExit(f"model_key {key!r} not in AVAILABLE_MODELS {list(AVAILABLE_MODELS)}")

    t0 = time.perf_counter()
    model = AVAILABLE_MODELS[key](device=a.device, backend=a.backend)
    load_s = time.perf_counter() - t0
    log.info("loaded %s (backend=%s) in %.1fs (dim=%s)", key, a.backend, load_s,
             getattr(model, "get_embedding_dimension", lambda: "?")())

    doc_texts = [c.get("text") or "" for c in chunks]
    q_texts = [r["query"] for r in queries]
    enc = {"batch_size": batch, "show_progress_bar": False}

    with bc.RamSampler() as ram:
        te = time.perf_counter()
        doc_vecs = np.asarray(model.encode_document(doc_texts, **enc), dtype=np.float32)
        q_vecs = np.asarray(model.encode_query(q_texts, **enc), dtype=np.float32)
        dv2 = np.asarray(model.encode_document(doc_texts[:8], **enc), dtype=np.float32)  # determinism
        embed_s = time.perf_counter() - te

    gate = core.gate_b_checks(doc_vecs, expected_dim=arm.get("expected_dim"))
    gate.update(core.determinism_check(doc_vecs[:8], dv2))
    log.info("GATE B: %s", gate)
    core.validate_gate_b(gate)

    execution = (
        model.get_execution_info()
        if hasattr(model, "get_execution_info")
        else {
            "requested_device": a.device,
            "actual_device": a.device,
            "fallback_events": [],
            "degraded": False,
        }
    )

    out_path = a.out or str(Path(out_dir) / f"arm_{a.arm}.json")
    payload = core.evaluate(
        label=a.arm, model_key=key, model_name=arm["model_id"], backend="torch",
        device=execution["actual_device"], dim=int(doc_vecs.shape[1]),
        chunks=chunks, doc_vecs=doc_vecs, queries=queries, query_vecs=q_vecs,
        k=k, candidate_k=candidate_k, embed_seconds=embed_s,
        telemetry={**ram.stat(), "load_seconds": round(load_s, 1), "gate_b": gate,
                   "execution": execution, "dataset_validation": dataset_validation},
        fingerprint=bc.run_fingerprint(cfg, arm),
        out_path=out_path,
    )
    s = payload["summary"]
    log.info("[%s] hit@1=%s hit@%d=%s dim=%s embed=%.0fs ram_min=%s%% -> %s",
             a.arm, s["hit_at_1_rate"], k, s["hit_at_k_rate"], s["embedding_dimension"],
             embed_s, ram.stat()["ram_free_pct_min"], out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
