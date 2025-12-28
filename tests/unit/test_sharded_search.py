def test_merge_top_k():
    from search.sharded_index_manager import merge_top_k

    a = [("id1", 0.9, {}), ("id2", 0.5, {})]
    b = [("id3", 0.8, {}), ("id4", 0.4, {})]
    merged = merge_top_k([a, b], k=3)
    assert [m[0] for m in merged] == ["id1", "id3", "id2"]
