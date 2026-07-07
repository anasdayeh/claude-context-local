#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from common_utils import get_storage_dir
from chunking.multi_language_chunker import MultiLanguageChunker
from embeddings.embedder import CodeEmbedder
from mcp_server.code_search_server import CodeSearchServer
from search.searcher import IntelligentSearcher

FIXED_REPO = Path("/Users/anasdayeh/Downloads/ADS_Website")
DEFAULT_QUERIES = [
    "hero animation",
    "site header",
    "navigation menu",
    "scroll behavior",
    "contact form",
    "footer layout",
    "css variables",
    "mobile nav",
]
DEFAULT_TOP_K = 10
DEFAULT_EMBED_SAMPLE_MAX = 200


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def _apply_test_mocks_if_requested() -> None:
    if not (_truthy_env("PYTEST_USE_MOCKS") or _truthy_env("CODE_SEARCH_BENCH_USE_MOCKS")):
        return

    class EmbeddingModelMock:
        def __init__(self, model_name=None, cache_dir=None, device="cpu", **kwargs):
            self.cache_dir = cache_dir
            self.device = device
            self.model_name = model_name or "google/embeddinggemma-300m"
            self.rng = np.random.RandomState(42)

        def encode(self, texts, **kwargs):
            return self.rng.randn(len(texts), 768).astype(np.float32)

        def get_embedding_dimension(self):
            return 768

        def get_model_info(self):
            return {
                "model_name": self.model_name,
                "embedding_dimension": 768,
                "max_seq_length": 512,
                "device": self.device,
                "status": "loaded",
            }

        def cleanup(self):
            return None

    from embeddings import embedding_models_register

    embedding_models_register.AVAILABLE_MODELS["google/embeddinggemma-300m"] = EmbeddingModelMock


def _iter_supported_files(repo_path: Path, chunker: MultiLanguageChunker) -> Iterable[Path]:
    for root, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [
            d for d in dirnames if d not in MultiLanguageChunker.DEFAULT_IGNORED_DIRS
        ]
        dirnames.sort()
        filenames.sort()
        for fname in filenames:
            path = Path(root) / fname
            if chunker.is_supported(str(path)):
                yield path


def _collect_chunks(repo_path: Path, max_chunks: int) -> List[Any]:
    chunker = MultiLanguageChunker(str(repo_path))
    chunks: List[Any] = []
    for path in _iter_supported_files(repo_path, chunker):
        chunks.extend(chunker.chunk_file(str(path)))
        if max_chunks and len(chunks) >= max_chunks:
            break
    return chunks[:max_chunks] if max_chunks else chunks


def _measure_embedding_throughput(
    embedder: CodeEmbedder,
    chunks: List[Any],
    batch_size: int,
) -> float:
    if not chunks:
        return 0.0

    warmup = chunks[:1]
    if warmup:
        embedder.embed_chunks(warmup, batch_size=1)

    start = time.perf_counter()
    embedder.embed_chunks(chunks, batch_size=batch_size)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return 0.0
    return float(len(chunks) / elapsed)


def _measure_search_latency_ms(
    searcher: IntelligentSearcher,
    queries: List[str],
    k: int,
) -> List[float]:
    latencies: List[float] = []
    for query in queries:
        start = time.perf_counter()
        _ = searcher.search(query, k=k, search_mode="semantic")
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(float(elapsed))
    return latencies


def _percentiles(values: List[float], pct_list: List[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {str(p): None for p in pct_list}
    arr = np.asarray(values, dtype=np.float64)
    return {str(p): float(np.percentile(arr, p)) for p in pct_list}


def _topk_overlap(
    searcher_a: IntelligentSearcher,
    searcher_b: IntelligentSearcher,
    queries: List[str],
    k: int,
) -> List[float]:
    overlaps: List[float] = []
    for query in queries:
        results_a = searcher_a.search(query, k=k, search_mode="semantic")
        results_b = searcher_b.search(query, k=k, search_mode="semantic")
        ids_a = [res.chunk_id for res in results_a]
        ids_b = [res.chunk_id for res in results_b]
        denom = min(k, len(ids_a), len(ids_b))
        if denom <= 0:
            overlaps.append(0.0)
            continue
        overlap = len(set(ids_a[:denom]) & set(ids_b[:denom])) / float(denom)
        overlaps.append(float(overlap))
    return overlaps


def _encode_documents(embedder: CodeEmbedder, texts: List[str]) -> np.ndarray:
    if hasattr(embedder, "_encode_documents"):
        return embedder._encode_documents(texts)  # type: ignore[attr-defined]
    return np.vstack([embedder.embed_document(text) for text in texts])


def _quant_similarity(
    current_embedder: CodeEmbedder,
    torch_embedder: CodeEmbedder,
    samples: List[str],
) -> Dict[str, Optional[float]]:
    if not samples:
        return {"mean": None, "p5": None, "p50": None, "p95": None}

    current = _encode_documents(current_embedder, samples)
    baseline = _encode_documents(torch_embedder, samples)

    current_norm = current / np.linalg.norm(current, axis=1, keepdims=True)
    baseline_norm = baseline / np.linalg.norm(baseline, axis=1, keepdims=True)
    similarities = np.sum(current_norm * baseline_norm, axis=1)

    stats = {
        "mean": float(np.mean(similarities)),
        "p5": float(np.percentile(similarities, 5)),
        "p50": float(np.percentile(similarities, 50)),
        "p95": float(np.percentile(similarities, 95)),
    }
    return stats


def _resolve_log_path(out_path: Path, storage_dir: Path) -> Path:
    logs_dir = storage_dir / "logs"
    try:
        out_path_resolved = out_path.resolve()
        if logs_dir.resolve() in out_path_resolved.parents:
            return out_path
    except Exception:
        pass
    return logs_dir / out_path.name


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_table(payload: Dict[str, Any]) -> None:
    rows = [
        ("embedding_throughput_chunks_per_sec", payload.get("embedding_throughput_chunks_per_sec")),
        ("indexing_time_seconds", payload.get("indexing_time_seconds")),
        ("search_latency_ms_p50", payload.get("search_latency_ms", {}).get("p50")),
        ("search_latency_ms_p95", payload.get("search_latency_ms", {}).get("p95")),
    ]

    overlap = payload.get("topk_overlap")
    if isinstance(overlap, dict):
        rows.append(("topk_overlap_mean", overlap.get("mean")))
        rows.append(("topk_overlap_p50", overlap.get("p50")))
        rows.append(("topk_overlap_p95", overlap.get("p95")))

    quant = payload.get("quant_similarity")
    if isinstance(quant, dict):
        rows.append(("quant_similarity_mean", quant.get("mean")))
        rows.append(("quant_similarity_p50", quant.get("p50")))
        rows.append(("quant_similarity_p95", quant.get("p95")))

    width = max(len(name) for name, _ in rows)
    print(f"{'Metric'.ljust(width)}  Value")
    print(f"{'-' * width}  {'-' * 10}")
    for name, value in rows:
        print(f"{name.ljust(width)}  {_format_value(value)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", help="Override repo path (default: fixed corpus)")
    args = parser.parse_args()

    repo_path = Path(args.repo).expanduser() if args.repo else FIXED_REPO
    if not repo_path.exists() or not repo_path.is_dir():
        print(f"Invalid repo path: {repo_path}", file=sys.stderr)
        return 2

    _apply_test_mocks_if_requested()

    storage_dir = get_storage_dir()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "embedding_throughput_chunks_per_sec": None,
        "indexing_time_seconds": None,
        "search_latency_ms": {"p50": None, "p95": None},
        "meta": {
            "dry_run": bool(args.dry_run),
            "repo_path": str(repo_path),
            "query_count": len(DEFAULT_QUERIES),
            "top_k": DEFAULT_TOP_K,
        },
    }

    if args.dry_run:
        out_path.write_text(json.dumps(payload, indent=2))
        log_path = _resolve_log_path(out_path, storage_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path != out_path:
            log_path.write_text(json.dumps(payload, indent=2))
        _print_table(payload)
        return 0

    embed_sample_max = int(os.getenv("CODE_SEARCH_BENCH_EMBED_SAMPLE_MAX", DEFAULT_EMBED_SAMPLE_MAX))
    embed_batch_size = int(
        os.getenv("CODE_SEARCH_EMBED_BATCH_SIZE")
        or os.getenv("CODE_SEARCH_BATCH_SIZE")
        or 32
    )
    if embed_batch_size <= 0:
        embed_batch_size = 32

    server = CodeSearchServer()
    sample_chunks = _collect_chunks(repo_path, embed_sample_max)
    throughput = _measure_embedding_throughput(server.embedder, sample_chunks, embed_batch_size)

    index_start = time.perf_counter()
    index_result = server.index_directory(
        str(repo_path),
        project_name=repo_path.name,
        incremental=False,
    )
    indexing_time = time.perf_counter() - index_start
    if not index_result.get("success"):
        print(f"Indexing failed: {index_result}", file=sys.stderr)
        return 1

    server.switch_project(str(repo_path))
    if server._searcher is None or server._index_manager is None:
        print("Searcher not available after indexing", file=sys.stderr)
        return 1

    searcher = server._searcher
    _ = searcher.search(DEFAULT_QUERIES[0], k=DEFAULT_TOP_K, search_mode="semantic")

    latencies_ms = _measure_search_latency_ms(searcher, DEFAULT_QUERIES, DEFAULT_TOP_K)
    latency_stats = _percentiles(latencies_ms, [50, 95])

    baseline_embedder = CodeEmbedder(
        cache_dir=str(storage_dir / "models"),
        device=os.getenv("CODE_SEARCH_DEVICE", "auto"),
    )
    baseline_searcher = IntelligentSearcher(server._index_manager, baseline_embedder)
    overlap_values = _topk_overlap(searcher, baseline_searcher, DEFAULT_QUERIES, DEFAULT_TOP_K)
    overlap_stats = _percentiles(overlap_values, [50, 95])
    overlap_payload = {
        "mean": float(np.mean(overlap_values)) if overlap_values else None,
        "p50": overlap_stats.get("50"),
        "p95": overlap_stats.get("95"),
    }

    quant_enabled = _truthy_env("CODE_SEARCH_EMBED_ONNX_QUANTIZE") or _truthy_env("ST_ONNX_QUANTIZE")
    quant_payload = None
    if quant_enabled:
        quant_payload = _quant_similarity(server.embedder, baseline_embedder, DEFAULT_QUERIES)

    payload.update(
        {
            "embedding_throughput_chunks_per_sec": throughput,
            "indexing_time_seconds": float(indexing_time),
            "search_latency_ms": {
                "p50": latency_stats.get("50"),
                "p95": latency_stats.get("95"),
            },
            "topk_overlap": overlap_payload,
            "quant_similarity": quant_payload,
        }
    )
    payload["meta"].update(
        {
            "embed_sample_chunks": len(sample_chunks),
            "embed_batch_size": embed_batch_size,
            "model_info": server.embedder.get_model_info(),
            "quantization_enabled": quant_enabled,
        }
    )

    out_path.write_text(json.dumps(payload, indent=2))
    log_path = _resolve_log_path(out_path, storage_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path != out_path:
        log_path.write_text(json.dumps(payload, indent=2))

    _print_table(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
