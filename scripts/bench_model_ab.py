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
import json
import os
import sys
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

    from mcp_server.code_search_server import CodeSearchServer

    server = CodeSearchServer()

    t0 = time.perf_counter()
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
    searcher.search(queries[0]["query"], k=args.k, search_mode="semantic")

    per_query = []
    for row in queries:
        q = row["query"]
        expected = row.get("expected_files", [])
        t = time.perf_counter()
        results = searcher.search(q, k=args.k, search_mode="semantic")
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
            "hit_at_1": _hit(paths[:1], expected),
            "hit_at_k": _hit(paths, expected),
            "top1_path": paths[0] if paths else None,
            "latency_ms": round(latency_ms, 1),
            "results": hits,
        })

    n = len(per_query)
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
        "index_seconds": round(index_secs, 2),
        "chunks_added": index_result.get("chunks_added"),
        "hit_at_1_rate": round(sum(p["hit_at_1"] for p in per_query) / n, 3) if n else None,
        "hit_at_k_rate": round(sum(p["hit_at_k"] for p in per_query) / n, 3) if n else None,
        "mean_latency_ms": round(sum(p["latency_ms"] for p in per_query) / n, 1) if n else None,
        "query_count": n,
    }
    payload = {"summary": summary, "per_query": per_query}
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[{args.label}] hit@1={summary['hit_at_1_rate']} hit@{args.k}={summary['hit_at_k_rate']} "
          f"mean_latency={summary['mean_latency_ms']}ms dim={summary['embedding_dimension']} "
          f"index={summary['index_seconds']}s -> {out}")
    return 0


def render(args: argparse.Namespace) -> int:
    runs = [json.loads(Path(p).expanduser().read_text()) for p in args.inputs]
    labels = [r["summary"]["label"] for r in runs]

    # machine report
    report_json = {
        "summaries": [r["summary"] for r in runs],
        "queries": [],
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
                "top1_path": pq["top1_path"], "latency_ms": pq["latency_ms"],
                "results": pq["results"],
            }
        report_json["queries"].append(entry)
    Path(args.out_json).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).expanduser().write_text(json.dumps(report_json, indent=2))

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
    L.append(row("mean latency (ms)", "mean_latency_ms"))
    L.append(row("index seconds", "index_seconds"))
    L.append(row("chunks", "chunks_added"))
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
    Path(args.out_md).expanduser().write_text("\n".join(L))
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
