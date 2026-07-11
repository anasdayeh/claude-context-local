from scripts.bench_dataset import paired_comparison, pareto_frontier, ranking_metrics


def test_ranking_metrics_compute_mrr_and_ndcg():
    rows = [
        {"expected_files": ["right.py"], "results": [{"path": "wrong.py"}, {"path": "right.py"}]},
        {"expected_files": ["other.py"], "results": [{"path": "other.py"}]},
    ]

    metrics = ranking_metrics(rows, k=5)

    assert metrics["mrr"] == 0.75
    assert 0.8 < metrics["ndcg_at_k"] < 0.82


def test_small_paired_difference_is_reported_inconclusive():
    left = [True] * 13 + [False] * 14
    right = [True] * 12 + [False] * 15

    result = paired_comparison(left, right, seed=7)

    assert result["difference"] == 0.037
    assert result["conclusion"] == "inconclusive"
    assert result["exact_p"] > 0.05


def test_pareto_frontier_keeps_quality_speed_tradeoffs_and_drops_dominated_arm():
    summaries = [
        {"label": "fast", "hit_at_1_rate": 0.5, "hit_at_k_rate": 0.7, "embed_seconds": 10},
        {"label": "quality", "hit_at_1_rate": 0.6, "hit_at_k_rate": 0.8, "embed_seconds": 100},
        {"label": "dominated", "hit_at_1_rate": 0.4, "hit_at_k_rate": 0.6, "embed_seconds": 20},
    ]

    assert pareto_frontier(summaries) == ["fast", "quality"]
