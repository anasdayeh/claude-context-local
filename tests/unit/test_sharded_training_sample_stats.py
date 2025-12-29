import json


def test_sharded_stats_aggregate_training_sample(tmp_path):
    from search.sharded_index_manager import ShardedIndexManager

    mgr = ShardedIndexManager(str(tmp_path))
    # create a second shard
    mgr._create_new_shard()

    shard_dirs = [tmp_path / "shards" / "shard_000", tmp_path / "shards" / "shard_001"]
    counts = [5, 7]
    totals = [10, 14]

    for shard_dir, count, total in zip(shard_dirs, counts, totals):
        stats_path = shard_dir / "training_sample_stats.json"
        stats_path.write_text(
            json.dumps(
                {
                    "count": count,
                    "total_seen": total,
                    "max_vectors": 25000,
                }
            )
        )

    stats = mgr.get_stats()
    assert stats.get("training_sample_count") == sum(counts)
    assert stats.get("training_sample_total_seen") == sum(totals)
