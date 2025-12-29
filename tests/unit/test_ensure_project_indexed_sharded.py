import json


def test_ensure_project_indexed_accepts_sharded(tmp_path, monkeypatch):
    from mcp_server.code_search_server import CodeSearchServer
    from common_utils import get_storage_dir

    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    get_storage_dir.cache_clear()
    server = CodeSearchServer()

    project_path = tmp_path / "repo"
    project_path.mkdir()

    project_dir = server.get_project_storage_dir(str(project_path))
    index_dir = project_dir / "index"
    shards_dir = index_dir / "shards"
    shard_dir = shards_dir / "shard_000"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Simulate a shard index file
    (shard_dir / "code.index").write_text("shard")

    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project_path": str(project_path),
                "embedding_dimension": 768,
                "index_type": "flat",
                "shard_count": 1,
                "shards": [
                    {
                        "id": "shard_000",
                        "path": "shards/shard_000",
                        "vector_count": 10,
                        "index_bytes": 1,
                        "metadata_bytes": 1,
                    }
                ],
            }
        )
    )

    stats_path = index_dir / "stats.json"
    stats_path.write_text(json.dumps({"total_chunks": 10}))

    assert server.ensure_project_indexed(str(project_path)) is True
