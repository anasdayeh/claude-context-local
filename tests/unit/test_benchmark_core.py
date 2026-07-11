import numpy as np
import pytest

from scripts import _arm_core as core


@pytest.mark.parametrize(
    "gate,match",
    [
        ({"all_finite": False, "dim_ok": True, "unit_norm_after_normalize": True, "deterministic": True}, "finite"),
        ({"all_finite": True, "dim_ok": False, "unit_norm_after_normalize": True, "deterministic": True}, "dimension"),
        ({"all_finite": True, "dim_ok": True, "unit_norm_after_normalize": False, "deterministic": True}, "unit-normal"),
        ({"all_finite": True, "dim_ok": True, "unit_norm_after_normalize": True, "deterministic": False}, "determin"),
    ],
)
def test_validate_gate_b_rejects_failed_invariant(gate, match):
    with pytest.raises(ValueError, match=match):
        core.validate_gate_b(gate)


def test_evaluate_persists_candidate_depth_beyond_metric_cutoff(tmp_path):
    chunks = [
        {"path": f"f{i}.py", "text": f"chunk {i}", "start_line": 1, "end_line": 1, "name": f"f{i}"}
        for i in range(3)
    ]
    doc_vecs = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.6, 0.4]], dtype=np.float32)
    query_vecs = np.asarray([[1.0, 0.0]], dtype=np.float32)
    queries = [{"query": "find it", "expected_files": ["f0.py"], "lang": "py"}]

    payload = core.evaluate(
        label="test",
        model_key="test",
        model_name="test",
        backend="test",
        device="cpu",
        dim=2,
        chunks=chunks,
        doc_vecs=doc_vecs,
        queries=queries,
        query_vecs=query_vecs,
        k=1,
        candidate_k=3,
        out_path=tmp_path / "run.json",
    )

    row = payload["per_query"][0]
    assert len(row["results"]) == 3
    assert row["hit_at_k"] is True
    assert row["hit_at_10"] is True
    assert payload["summary"]["k"] == 1
    assert payload["summary"]["candidate_k"] == 3
    assert payload["summary"]["recall_at_10_rate"] == 1.0


def test_determinism_check_accepts_numerically_equivalent_vectors():
    reference = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    repeat = np.asarray([[1.0002, 0.00001], [0.00001, 0.9998]], dtype=np.float32)

    result = core.determinism_check(reference, repeat)

    assert result["deterministic"] is True
    assert result["min_cosine"] > 0.9999


def test_determinism_check_rejects_materially_different_vectors():
    reference = np.asarray([[1.0, 0.0]], dtype=np.float32)
    repeat = np.asarray([[0.0, 1.0]], dtype=np.float32)

    result = core.determinism_check(reference, repeat)

    assert result["deterministic"] is False
