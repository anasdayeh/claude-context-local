import json


def test_sharded_stats_warn_when_metadata_without_vectors(tmp_path):
    from search.sharded_index_manager import ShardedIndexManager

    manager = ShardedIndexManager(str(tmp_path))
    shard = manager.active_manager()

    shard.metadata_db["1"] = {
        "chunk_id": "abc",
        "metadata": {
            "relative_path": "a.py",
            "file_path": str(tmp_path / "a.py"),
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 2,
            "content_preview": "def f(): pass",
        },
    }
    shard.metadata_db.commit()

    stats = manager.get_stats()
    assert stats.get("total_chunks") == 0
    assert "sanity_warning" in stats
    assert "sanity_suggestion" in stats
