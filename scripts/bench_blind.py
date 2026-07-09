#!/usr/bin/env python3
"""Blind N-way retrieval-quality judging (Gate C).

Takes the per-query run JSONs produced by bench_model_ab.py `run` (one per arm),
strips model identity, and deterministically permutes arm order PER QUERY (hash
seed, so the un-blinding key is reproducible), then emits:

  blind_report.md   the agent judges THIS — lettered result sets (A/B/C...) with
                    content excerpts, NO model names, so quality is scored on the
                    chunks themselves, not on which model produced them.
  blind_key.json    query -> {letter: arm_label} to un-blind after judging, plus
                    each query's expected_files for the automated hit check.

Usage:
  bench_blind.py make  --inputs a.json b.json c.json --out-dir benchmarks/blind --seed bridge
  bench_blind.py score --key benchmarks/blind/blind_key.json \
                       --verdict benchmarks/blind/verdict.json --out benchmarks/blind/scored.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def _det_shuffle(n: int, seed_str: str) -> list[int]:
    """Deterministic Fisher-Yates permutation of range(n) driven by a sha256
    stream of the seed. Reproducible across runs/machines (no RNG state), so the
    blind key always matches the report."""
    order = list(range(n))
    h = hashlib.sha256(seed_str.encode()).digest()
    stream = list(h)
    si = 0
    for i in range(n - 1, 0, -1):
        if si >= len(stream):
            h = hashlib.sha256(h).digest()
            stream = list(h)
            si = 0
        j = stream[si] % (i + 1)
        si += 1
        order[i], order[j] = order[j], order[i]
    return order


def make(args: argparse.Namespace) -> int:
    runs = [json.loads(Path(p).expanduser().read_text()) for p in args.inputs]
    labels = [r["summary"]["label"] for r in runs]
    n = min(len(r["per_query"]) for r in runs)
    letters = _letters(len(runs))
    key = {"arms": labels, "queries": []}
    L = [
        "# BLIND retrieval comparison — judge which result set best answers each query\n",
        f"{len(runs)} anonymised search backends (Set {', '.join(letters)}), {n} queries. "
        "You are the downstream engineer who will use these results to write or modify code.\n",
        "For EACH query, for EACH set, score how useful the returned chunks are for actually "
        "doing the task:\n"
        "  3 = the exact right file/chunk is the top result\n"
        "  2 = right file present but not first, or first result is adjacent-correct\n"
        "  1 = only related-but-wrong files\n"
        "  0 = useless\n"
        "Then name the winning set. You do NOT know which model is which — judge only the "
        "content. Record your answer as JSON: "
        '{"Q1": {"A": 3, "B": 1, "winner": "A"}, ...}\n',
    ]
    for i in range(n):
        base = runs[0]["per_query"][i]
        order = _det_shuffle(len(runs), f"{args.seed}:{i}")
        L.append(f"\n## Q{i + 1}: {base['query']}")
        qk = {
            "query": base["query"],
            "expected_files": base.get("expected_files", []),
            "lang": base.get("lang", "?"),
            "map": {},
        }
        for letter, ridx in zip(letters, order):
            qk["map"][letter] = runs[ridx]["summary"]["label"]
            pq = runs[ridx]["per_query"][i]
            L.append(f"\n**Set {letter}:**")
            res = pq.get("results", [])[:5]
            if not res:
                L.append("- (no results)")
            for h in res:
                nm = h.get("name")
                L.append(
                    f"- `{h['path']}` (L{h.get('lines', '?')}"
                    + (f", {nm}" if nm else "")
                    + f") — {h.get('excerpt', '')[:170]}"
                )
        key["queries"].append(qk)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "blind_report.md").write_text("\n".join(L))
    (out_dir / "blind_key.json").write_text(json.dumps(key, indent=2))
    print(f"wrote {out_dir}/blind_report.md + blind_key.json  ({len(runs)} arms, {n} queries)")
    return 0


def score(args: argparse.Namespace) -> int:
    key = json.loads(Path(args.key).expanduser().read_text())
    verdict = json.loads(Path(args.verdict).expanduser().read_text())
    arms = key["arms"]
    totals: dict[str, float] = {a: 0.0 for a in arms}
    wins = {a: 0 for a in arms}
    rows = []
    for i, qk in enumerate(key["queries"], 1):
        qv = verdict.get(f"Q{i}", {})
        mapping = qk["map"]
        row = {"q": qk["query"], "lang": qk.get("lang"), "scores": {}, "winner": None}
        for letter, arm in mapping.items():
            s = qv.get(letter)
            if isinstance(s, (int, float)):
                totals[arm] += s
                row["scores"][arm] = s
        w = qv.get("winner")
        if w in mapping:
            wins[mapping[w]] += 1
            row["winner"] = mapping[w]
        rows.append(row)
    n = len(key["queries"]) or 1
    L = [
        "# Un-blinded scored verdict\n",
        f"{len(key['queries'])} queries, {len(arms)} arms. Max 3/query.\n",
        "## Totals\n",
        "| arm | mean score | wins |",
        "| --- | --- | --- |",
    ]
    for a in sorted(arms, key=lambda a: -totals[a]):
        L.append(f"| {a} | {totals[a] / n:.2f} | {wins[a]} |")
    L.append("\n## Per query\n")
    for i, row in enumerate(rows, 1):
        sc = ", ".join(f"{a}={row['scores'].get(a, '?')}" for a in arms)
        L.append(f"- Q{i} ({row['lang']}) winner=**{row.get('winner', '?')}** — {sc} — {row['q'][:80]}")
    Path(args.out).expanduser().write_text("\n".join(L))
    print(f"scored -> {args.out}")
    for a in sorted(arms, key=lambda a: -totals[a]):
        print(f"  {a}: mean={totals[a] / n:.2f} wins={wins[a]}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="build blind report + key from run JSONs")
    m.add_argument("--inputs", nargs="+", required=True)
    m.add_argument("--out-dir", required=True)
    m.add_argument("--seed", default="bridge")
    m.set_defaults(func=make)
    s = sub.add_parser("score", help="un-blind a verdict JSON into a scored report")
    s.add_argument("--key", required=True)
    s.add_argument("--verdict", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=score)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
