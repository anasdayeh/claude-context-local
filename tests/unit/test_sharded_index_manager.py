from pathlib import Path


def test_rollover_creates_new_shard(tmp_path):
    from search.sharded_index_manager import ShardedIndexManager

    mgr = ShardedIndexManager(str(tmp_path))
    mgr._target_shard_bytes = 1
    shard1 = mgr.active_shard_id
    mgr._maybe_rollover(current_shard_bytes=2)
    assert mgr.active_shard_id != shard1
    assert (Path(tmp_path) / "shards" / mgr.active_shard_id).exists()


def test_lru_eviction_respects_budget(tmp_path):
    from search.sharded_index_manager import ShardedIndexManager

    mgr = ShardedIndexManager(str(tmp_path))
    mgr._max_bytes = 10
    mgr._loaded_shards = {"s1": 8, "s2": 8}
    mgr._lru = ["s1", "s2"]
    mgr._enforce_budget()
    assert sum(mgr._loaded_shards.values()) <= 10
