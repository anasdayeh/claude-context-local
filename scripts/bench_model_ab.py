#!/usr/bin/env python3
"""Cross-model retrieval-quality A/B (Phase 4 / Gate C).

Unlike scripts/bench_mcp_perf.py (latency + top-k overlap against the *same*
embedder), this compares retrieval QUALITY across two different embedding
backends on a labeled query set, so an agent can judge which model surfaces
the chunk actually needed.

Two modes:

  run    Index one repo under one model+store, run each labeled query in
         pure-semantic mode, and record ranked hits + latency to a JSON.
         Model selection is process-global (CODE_SEARCH_EMBED_MODEL), so run
         once per model in a separate process.

  render Merge two `run` JSONs into a side-by-side report.md + a machine
         report.json (per-query hit@1/hit@k + latency, plus a summary table).

Example:
  uv run python scripts/bench_model_ab.py run \
      --model-key google/embeddinggemma-300m --label gemma \
      --store ~/.claude_code_search --repo /path/to/fixture \
      --queries benchmarks/quality_fixture_queries.jsonl --out /tmp/gemma.json

  CODE_SEARCH_EMBED_MODEL=qwen3-embedding-4b uv run python scripts/bench_model_ab.py run \
      --model-key qwen3-embedding-4b --label qwen \
      --store /Volumes/Anas_2TB/AI/code-search/stores/qwen3-4b-test --repo /path/to/fixture \
      --queries benchmarks/quality_fixture_queries.jsonl --out /tmp/qwen.json

  uv run python scripts/bench_model_ab.py render \
      --inputs /tmp/gemma.json /tmp/qwen.json \
      --out-md report.md --out-json report.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load_queries(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _hit(paths: list[str], expected: list[str]) -> bool:
    """True if any expected substring matches any returned path."""
    return any(any(exp in p for p in paths) for exp in expected)


class _RamSampler:
    """Sample system free-RAM % on a background thread so each run records the
    memory pressure it actually experienced. The whole point is to never again
    confuse 'this model is slow' with 'the machine was starved by other apps'."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.samples: list[float] = []

    def _loop(self):
        import psutil
        while not self._stop.is_set():
            try:
                self.samples.append(100.0 - psutil.virtual_memory().percent)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    @property
    def min_free(self) -> float | None:
        return round(min(self.samples), 1) if self.samples else None

    @property
    def mean_free(self) -> float | None:
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None


def run(args: argparse.Namespace) -> int:
    # Model + store selection is read at server construction / import time,
    # so set the environment BEFORE importing any server module.
    os.environ["CODE_SEARCH_EMBED_MODEL"] = args.model_key
    os.environ["CODE_SEARCH_STORAGE"] = os.path.expanduser(args.store)

    repo = Path(args.repo).expanduser()
    if not repo.is_dir():
        print(f"repo not found: {repo}", file=sys.stderr)
        return 2
    queries = _load_queries(Path(args.queries).expanduser())
    from scripts.bench_dataset import validate_query_labels
    from scripts import bench_artifacts
    corpus_paths = [
        {"path": relative}
        for relative, _path in bench_artifacts.selected_tree_files(repo, args.file_patterns)
    ]
    validate_query_labels(corpus_paths, queries, max_chunk_fraction=1.0)

    from mcp_server.code_search_server import CodeSearchServer

    server = CodeSearchServer()

    t0 = time.perf_counter()
    with _RamSampler() as ram:
        index_result = server.index_directory(
            str(repo), project_name=repo.name, incremental=False,
            file_patterns=args.file_patterns or None,
        )
    index_secs = time.perf_counter() - t0
    if not index_result.get("success"):
        print(f"indexing failed: {index_result}", file=sys.stderr)
        return 1

    server.switch_project(str(repo))
    if server._searcher is None:
        print("searcher unavailable after indexing", file=sys.stderr)
        return 1
    searcher = server._searcher

    model_info = server.embedder.get_model_info()
    # warmup (exclude first-call load cost from per-query latency)
    candidate_k = max(args.k, 10)
    searcher.search(queries[0]["query"], k=candidate_k, search_mode="semantic")

    per_query = []
    for row in queries:
        q = row["query"]
        expected = row.get("expected_files", [])
        t = time.perf_counter()
        results = searcher.search(q, k=candidate_k, search_mode="semantic")
        latency_ms = (time.perf_counter() - t) * 1000.0
        hits = []
        for rank, r in enumerate(results, 1):
            content = (getattr(r, "content", "") or getattr(r, "content_preview", "") or "")
            excerpt = " ".join(content.split())[:220]
            hits.append({
                "rank": rank,
                "path": r.relative_path,
                "score": round(float(r.similarity_score), 4),
                "lines": f"{r.start_line}-{r.end_line}",
                "name": getattr(r, "name", None),
                "excerpt": excerpt,
            })
        paths = [h["path"] for h in hits]
        per_query.append({
            "query": q,
            "expected_files": expected,
            "notes": row.get("notes", ""),
            "lang": row.get("lang", "?"),
            "hit_at_1": _hit(paths[:1], expected),
            "hit_at_k": _hit(paths[:args.k], expected),
            "hit_at_10": _hit(paths[:10], expected),
            "top1_path": paths[0] if paths else None,
            "latency_ms": round(latency_ms, 1),
            "results": hits,
        })

    n = len(per_query)
    langs = sorted({p["lang"] for p in per_query})
    per_lang = {}
    for lg in langs:
        rows_lg = [p for p in per_query if p["lang"] == lg]
        m = len(rows_lg)
        per_lang[lg] = {
            "n": m,
            "hit_at_1_rate": round(sum(p["hit_at_1"] for p in rows_lg) / m, 3) if m else None,
            "hit_at_k_rate": round(sum(p["hit_at_k"] for p in rows_lg) / m, 3) if m else None,
            "recall_at_10_rate": round(sum(p["hit_at_10"] for p in rows_lg) / m, 3) if m else None,
        }
    import psutil as _ps
    summary = {
        "label": args.label,
        "model_key": args.model_key,
        "model_name": model_info.get("model_name"),
        "embedding_dimension": model_info.get("embedding_dimension"),
        "backend": model_info.get("backend"),
        "device": model_info.get("device"),
        "store": os.environ["CODE_SEARCH_STORAGE"],
        "repo": str(repo),
        "k": args.k,
        "candidate_k": candidate_k,
        "index_seconds": round(index_secs, 2),
        "chunks_added": index_result.get("chunks_added"),
        "hit_at_1_rate": round(sum(p["hit_at_1"] for p in per_query) / n, 3) if n else None,
        "hit_at_k_rate": round(sum(p["hit_at_k"] for p in per_query) / n, 3) if n else None,
        "recall_at_10_rate": round(sum(p["hit_at_10"] for p in per_query) / n, 3) if n else None,
        "mean_latency_ms": round(sum(p["latency_ms"] for p in per_query) / n, 1) if n else None,
        "query_count": n,
        "per_lang": per_lang,
        "ram_free_pct_min_during_index": ram.min_free,
        "ram_free_pct_mean_during_index": ram.mean_free,
        "system_total_gb": round(_ps.virtual_memory().total / 1e9, 1),
    }
    from scripts.bench_dataset import ranking_metrics
    summary.update(ranking_metrics(per_query, k=args.k))
    fingerprint = bench_artifacts.build_direct_run_fingerprint(
        repo_path=repo,
        query_path=Path(args.queries).expanduser(),
        arm={"label": args.label, "model_key": args.model_key, "k": args.k,
             "file_patterns": args.file_patterns},
        patterns=args.file_patterns,
        source_paths=[Path(__file__)],
    )
    payload = {
        "artifact": {
            "schema_version": bench_artifacts.SCHEMA_VERSION,
            "status": "complete",
            "fingerprint": fingerprint,
        },
        "summary": summary,
        "per_query": per_query,
    }
    out = Path(args.out).expanduser()
    bench_artifacts.atomic_write_json(out, payload)
    print(f"[{args.label}] hit@1={summary['hit_at_1_rate']} hit@{args.k}={summary['hit_at_k_rate']} "
          f"mean_latency={summary['mean_latency_ms']}ms dim={summary['embedding_dimension']} "
          f"index={summary['index_seconds']}s -> {out}")
    return 0


def render(args: argparse.Namespace) -> int:
    from scripts.bench_dataset import paired_comparison, pareto_frontier, validate_comparable_runs
    from scripts.bench_artifacts import atomic_write_json, atomic_write_text

    runs = [json.loads(Path(p).expanduser().read_text()) for p in args.inputs]
    validate_comparable_runs(runs)
    labels = [r["summary"]["label"] for r in runs]

    # machine report
    report_json = {
        "summaries": [r["summary"] for r in runs],
        "queries": [],
        "paired_comparisons": [],
        "pareto_frontier": pareto_frontier([r["summary"] for r in runs]),
    }
    # align per-query by index (same queries.jsonl order across runs)
    n = min(len(r["per_query"]) for r in runs)
    for i in range(n):
        q = runs[0]["per_query"][i]["query"]
        entry = {"query": q, "expected_files": runs[0]["per_query"][i]["expected_files"], "models": {}}
        for r in runs:
            pq = r["per_query"][i]
            entry["models"][r["summary"]["label"]] = {
                "hit_at_1": pq["hit_at_1"], "hit_at_k": pq["hit_at_k"],
                "hit_at_10": pq.get("hit_at_10"),
                "top1_path": pq["top1_path"], "latency_ms": pq["latency_ms"],
                "results": pq["results"],
            }
        report_json["queries"].append(entry)
    for left, right in itertools.combinations(runs, 2):
        metrics = ["hit_at_1", "hit_at_k"]
        if all(all(metric in row for row in run["per_query"][:n]) for run in runs for metric in ["hit_at_10"]):
            metrics.append("hit_at_10")
        for metric in metrics:
            comparison = paired_comparison(
                [row[metric] for row in left["per_query"][:n]],
                [row[metric] for row in right["per_query"][:n]],
                seed=0,
            )
            report_json["paired_comparisons"].append({
                "left": left["summary"]["label"],
                "right": right["summary"]["label"],
                "metric": metric,
                **comparison,
            })
    atomic_write_json(Path(args.out_json).expanduser(), report_json)

    # human/agent-readable markdown
    L = []
    L.append("# Embedding-model A/B — retrieval quality\n")
    L.append("Pure-semantic (vector-only) search on a labeled fixture. "
             "`hit@1` = expected file is the top result; `hit@k` = expected file anywhere in top-k.\n")
    L.append("## Summary\n")
    cols = ["metric"] + labels
    L.append("| " + " | ".join(cols) + " |")
    L.append("| " + " | ".join(["---"] * len(cols)) + " |")
    def row(metric, key):
        return "| " + " | ".join([metric] + [str(r["summary"].get(key)) for r in runs]) + " |"
    L.append(row("model_name", "model_name"))
    L.append(row("embedding_dimension", "embedding_dimension"))
    L.append(row("backend", "backend"))
    L.append(row("hit@1 rate", "hit_at_1_rate"))
    L.append(row("hit@k rate", "hit_at_k_rate"))
    L.append(row("recall@10 rate", "recall_at_10_rate"))
    L.append(row("MRR", "mrr"))
    L.append(row("nDCG@k", "ndcg_at_k"))
    L.append(row("mean latency (ms)", "mean_latency_ms"))
    L.append(row("index seconds", "index_seconds"))
    L.append(row("chunks", "chunks_added"))
    L.append("")
    L.append("## Descriptive Pareto frontier\n")
    L.append(
        "Non-dominated quality/runtime tradeoffs: **"
        + ", ".join(report_json["pareto_frontier"])
        + "**. This is descriptive; paired uncertainty still controls winner claims.\n"
    )
    L.append("## Paired uncertainty\n")
    L.append("A model is not declared better when the paired comparison is inconclusive.\n")
    L.append("| left | right | metric | difference | 95% bootstrap CI | exact p | conclusion |")
    L.append("| --- | --- | --- | ---: | --- | ---: | --- |")
    for comparison in report_json["paired_comparisons"]:
        L.append(
            f"| {comparison['left']} | {comparison['right']} | {comparison['metric']} | "
            f"{comparison['difference']} | {comparison['bootstrap_95_ci']} | "
            f"{comparison['exact_p']} | {comparison['conclusion']} |"
        )
    L.append("")
    L.append("## Per-query, side by side\n")
    for i in range(n):
        q = runs[0]["per_query"][i]
        L.append(f"### Q{i+1}: {q['query']}")
        L.append(f"*expected:* `{', '.join(q['expected_files'])}`\n")
        for r in runs:
            pq = r["per_query"][i]
            flag = "✅" if pq["hit_at_1"] else ("🔸" if pq["hit_at_k"] else "❌")
            L.append(f"**{r['summary']['label']}** {flag} "
                     f"(hit@1={pq['hit_at_1']}, hit@k={pq['hit_at_k']}, {pq['latency_ms']}ms)")
            for h in pq["results"][:5]:
                L.append(f"- `{h['rank']}` `{h['path']}` (score={h['score']}, {h['name']}) — {h['excerpt'][:120]}")
            L.append("")
    atomic_write_text(Path(args.out_md).expanduser(), "\n".join(L))
    print(f"rendered {args.out_md} and {args.out_json} for models: {', '.join(labels)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="index + query one model, write JSON")
    r.add_argument("--model-key", required=True, help="AVAILABLE_MODELS key / CODE_SEARCH_EMBED_MODEL")
    r.add_argument("--label", required=True, help="short label for reports (e.g. gemma, qwen)")
    r.add_argument("--store", required=True, help="CODE_SEARCH_STORAGE root for this model")
    r.add_argument("--repo", required=True, help="repo to index + query")
    r.add_argument("--queries", required=True, help="labeled queries.jsonl")
    r.add_argument("--out", required=True, help="output JSON path")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--file-patterns", nargs="+", default=None,
                   help="glob(s) limiting indexed files, e.g. '*.py' (code-only slice)")
    r.set_defaults(func=run)

    rd = sub.add_parser("render", help="merge two run JSONs into report.md/json")
    rd.add_argument("--inputs", nargs="+", required=True, help="two run JSONs")
    rd.add_argument("--out-md", required=True)
    rd.add_argument("--out-json", required=True)
    rd.set_defaults(func=render)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
