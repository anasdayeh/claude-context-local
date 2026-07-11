import pytest

from scripts.bench_dataset import validate_comparable_runs, validate_query_labels


def test_rejects_expected_label_absent_from_corpus():
    chunks = [{"path": "src/auth.py"}]
    queries = [{"query": "auth", "expected_files": ["missing.py"]}]

    with pytest.raises(ValueError, match="matches no sampled path"):
        validate_query_labels(chunks, queries)


def test_rejects_label_that_matches_excessive_path_fraction():
    chunks = [{"path": f"Data/file_{i}.txt"} for i in range(20)] + [{"path": "src/auth.py"}]
    queries = [{"query": "auth", "expected_files": ["Data"]}]

    with pytest.raises(ValueError, match="too broad"):
        validate_query_labels(chunks, queries, max_path_fraction=0.2)


def test_accepts_precise_label_present_in_sample():
    chunks = [{"path": "src/auth.py"}, {"path": "src/users.py"}]
    queries = [{"query": "auth", "expected_files": ["src/auth.py"]}]

    report = validate_query_labels(chunks, queries)

    assert report["valid"] is True
    assert report["queries"] == 1


def test_comparison_rejects_different_query_order_or_corpus():
    base = {
        "artifact": {"fingerprint": {"corpus_sha256": "corpus-a"}},
        "per_query": [{"query": "first"}, {"query": "second"}],
    }
    reordered = {
        "artifact": {"fingerprint": {"corpus_sha256": "corpus-a"}},
        "per_query": [{"query": "second"}, {"query": "first"}],
    }
    other_corpus = {
        "artifact": {"fingerprint": {"corpus_sha256": "corpus-b"}},
        "per_query": [{"query": "first"}, {"query": "second"}],
    }

    with pytest.raises(ValueError, match="query order"):
        validate_comparable_runs([base, reordered])
    with pytest.raises(ValueError, match="corpus fingerprint"):
        validate_comparable_runs([base, other_corpus])
