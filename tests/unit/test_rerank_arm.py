import pytest

from scripts.rerank_arm import rerank_candidates, validate_base_candidate_depth


def test_reranker_can_promote_candidate_below_original_top_five():
    candidates = [{"path": f"f{i}.py", "rank": i + 1} for i in range(20)]
    scores = [0.0] * 20
    scores[12] = 9.0

    reranked = rerank_candidates(candidates, scores, k=5)

    assert reranked[0]["path"] == "f12.py"
    assert reranked[0]["rank"] == 1
    assert reranked[0]["rerank_score"] == 9.0


def test_reranker_rejects_base_artifact_without_requested_candidate_depth():
    base = {"per_query": [{"query": "q", "results": [{"path": f"f{i}.py"} for i in range(5)]}]}

    with pytest.raises(ValueError, match="requires 20 candidates"):
        validate_base_candidate_depth(base, top_n=20)

