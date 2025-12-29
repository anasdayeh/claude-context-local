import json


def test_clear_index_recreates_active_shard(tmp_path):
    from search.sharded_index_manager import ShardedIndexManager

    manager = ShardedIndexManager(str(tmp_path))
    assert manager._manifest.shard_count >= 1

    manager.clear_index()

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["shard_count"] == 1
    shard_id = manifest["shards"][0]["id"]
    assert (tmp_path / "shards" / shard_id).exists()


def test_clear_index_resets_root_stats(tmp_path):
    from search.sharded_index_manager import ShardedIndexManager

    manager = ShardedIndexManager(str(tmp_path))

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "total_chunks": 123,
                "files_indexed": 45,
                "chunk_types": {"code": 123},
                "top_tags": {"python": 123},
                "storage_size": 999,
                "shard_count": 9,
            },
            indent=2,
        )
    )
    assert manager.get_stats()["total_chunks"] == 123

    manager.clear_index()

    stats = manager.get_stats()
    assert stats["total_chunks"] == 0
    assert stats["files_indexed"] == 0
    assert json.loads(stats_path.read_text())["total_chunks"] == 0
