import json


def test_manifest_round_trip(tmp_path):
    from search.shard_manifest import ShardManifest

    manifest = ShardManifest(
        version=1,
        project_path="/tmp/project",
        embedding_dimension=768,
        index_type="flat",
        shard_count=1,
        shards=[{"id": "shard_000", "path": "shards/shard_000", "vector_count": 10}],
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)

    loaded = ShardManifest.load(path)
    assert loaded.project_path == manifest.project_path
    assert loaded.shards[0]["id"] == "shard_000"


def test_manifest_json_is_stable(tmp_path):
    from search.shard_manifest import ShardManifest

    manifest = ShardManifest(
        version=1,
        project_path="/tmp/project",
        embedding_dimension=768,
        index_type="flat",
        shard_count=1,
        shards=[],
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)

    data = json.loads(path.read_text())
    assert data["version"] == 1
    assert data["project_path"] == "/tmp/project"
