"""Dataset admission checks and paired retrieval statistics."""
from __future__ import annotations

import math
import random


def validate_comparable_runs(runs):
    """Reject reports that would compare different corpora or query ordering."""
    if len(runs) < 2:
        raise ValueError("comparison requires at least two runs")
    reference_queries = [row.get("query") for row in runs[0].get("per_query") or []]
    if not reference_queries:
        raise ValueError("comparison run contains no queries")
    reference_corpus = (
        ((runs[0].get("artifact") or {}).get("fingerprint") or {}).get("corpus_sha256")
    )
    if not reference_corpus:
        raise ValueError("comparison run has no corpus fingerprint")
    for run in runs[1:]:
        queries = [row.get("query") for row in run.get("per_query") or []]
        if queries != reference_queries:
            raise ValueError("comparison query order or membership differs between runs")
        corpus = (((run.get("artifact") or {}).get("fingerprint") or {}).get("corpus_sha256"))
        if corpus != reference_corpus:
            raise ValueError("comparison corpus fingerprint differs between runs")
    return {"valid": True, "runs": len(runs), "queries": len(reference_queries)}


def pareto_frontier(summaries):
    """Return arms not descriptively dominated on retrieval quality and runtime."""
    quality_keys = ("hit_at_1_rate", "hit_at_k_rate", "recall_at_10_rate", "mrr")

    def vector(summary):
        quality = [float(summary[key]) for key in quality_keys if summary.get(key) is not None]
        speed = summary.get("embed_seconds")
        if speed is None:
            speed = summary.get("index_seconds")
        if speed is None:
            speed = summary.get("mean_latency_ms")
        return quality, None if speed is None else float(speed)

    frontier = []
    for index, candidate in enumerate(summaries):
        candidate_quality, candidate_speed = vector(candidate)
        dominated = False
        for other_index, other in enumerate(summaries):
            if index == other_index:
                continue
            other_quality, other_speed = vector(other)
            if len(other_quality) != len(candidate_quality) or not candidate_quality:
                continue
            quality_no_worse = all(a >= b for a, b in zip(other_quality, candidate_quality))
            speed_no_worse = (
                True if candidate_speed is None or other_speed is None else other_speed <= candidate_speed
            )
            strictly_better = any(a > b for a, b in zip(other_quality, candidate_quality)) or (
                candidate_speed is not None and other_speed is not None and other_speed < candidate_speed
            )
            if quality_no_worse and speed_no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate.get("label") or f"arm_{index}"))
    return frontier


def _matches(path: str, expected: list[str]) -> bool:
    return any(label in (path or "") for label in expected)


def validate_query_labels(chunks, queries, *, max_path_fraction=0.2, max_chunk_fraction=0.2):
    paths = [str(chunk.get("path") or "") for chunk in chunks]
    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise ValueError("sample corpus contains no paths")
    for index, row in enumerate(queries, 1):
        labels = list(row.get("expected_files") or [])
        if not labels:
            raise ValueError(f"query {index} has no expected_files labels")
        for label in labels:
            path_matches = sum(label in path for path in unique_paths)
            chunk_matches = sum(label in path for path in paths)
            if not path_matches:
                raise ValueError(
                    f"query {index} expected label {label!r} matches no sampled path"
                )
            path_fraction = path_matches / len(unique_paths)
            chunk_fraction = chunk_matches / max(1, len(paths))
            exact_path = label in unique_paths
            if not exact_path and (path_fraction > max_path_fraction or chunk_fraction > max_chunk_fraction):
                raise ValueError(
                    f"query {index} expected label {label!r} is too broad "
                    f"(paths={path_fraction:.1%}, chunks={chunk_fraction:.1%})"
                )
    return {"valid": True, "queries": len(queries), "unique_paths": len(unique_paths)}


def ranking_metrics(rows, *, k=5):
    reciprocal_ranks = []
    ndcgs = []
    for row in rows:
        expected = list(row.get("expected_files") or [])
        rank = None
        for position, hit in enumerate((row.get("results") or [])[:k], 1):
            if _matches(str(hit.get("path") or ""), expected):
                rank = position
                break
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
        ndcgs.append(0.0 if rank is None else 1.0 / math.log2(rank + 1))
    count = len(rows) or 1
    return {
        "mrr": round(sum(reciprocal_ranks) / count, 3),
        "ndcg_at_k": round(sum(ndcgs) / count, 3),
    }


def _exact_p(left_only, right_only):
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def paired_comparison(left, right, *, seed=0, samples=4000, practical_delta=0.05):
    if len(left) != len(right) or not left:
        raise ValueError("paired comparisons require equal non-empty inputs")
    left_values = [int(bool(value)) for value in left]
    right_values = [int(bool(value)) for value in right]
    differences = [a - b for a, b in zip(left_values, right_values)]
    difference = sum(differences) / len(differences)
    left_only = sum(a == 1 and b == 0 for a, b in zip(left_values, right_values))
    right_only = sum(a == 0 and b == 1 for a, b in zip(left_values, right_values))
    exact_p = _exact_p(left_only, right_only)

    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        boot.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    boot.sort()
    low = boot[int(0.025 * (samples - 1))]
    high = boot[int(0.975 * (samples - 1))]
    conclusive = exact_p < 0.05 and abs(difference) >= practical_delta and (low > 0 or high < 0)
    conclusion = "left" if conclusive and difference > 0 else "right" if conclusive else "inconclusive"
    return {
        "difference": round(difference, 3),
        "left_only": left_only,
        "right_only": right_only,
        "exact_p": round(exact_p, 4),
        "bootstrap_95_ci": [round(low, 3), round(high, 3)],
        "conclusion": conclusion,
    }
