from search.hybrid import rrf_fuse, normalize_fts_query


def test_rrf_fuse_prefers_high_ranked_items():
    dense = ["a", "b", "c"]
    sparse = ["c", "a", "d"]
    results = rrf_fuse(dense, sparse, rrf_k=60, top_k=3)
    assert results[0][0] in {"a", "c"}
    assert len(results) == 3


def test_normalize_fts_query_strips_symbols():
    q = normalize_fts_query("foo.bar()? baz")
    assert q == "foo OR bar OR baz"
