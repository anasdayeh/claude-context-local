import json


def test_switch_project_auto_repairs_manifest(tmp_path, monkeypatch, caplog):
    from mcp_server.code_search_server import CodeSearchServer

    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    server = CodeSearchServer()

    project_path = tmp_path / "repo"
    project_path.mkdir()

    project_dir = server.get_project_storage_dir(str(project_path))
    index_dir = project_dir / "index"
    shards_dir = index_dir / "shards"
    shard_dir = shards_dir / "shard_000"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Corrupt/empty manifest
    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project_path": "",
                "embedding_dimension": 0,
                "index_type": "flat",
                "shard_count": 0,
                "shards": [],
            }
        )
    )

    # Stats to let ensure_project_indexed short-circuit
    stats_path = index_dir / "stats.json"
    stats_path.write_text(json.dumps({"total_chunks": 0}))

    with caplog.at_level("WARNING"):
        result = server.switch_project(str(project_path))

    assert result.get("success") is True
    repaired_manifest = json.loads(manifest_path.read_text())
    assert repaired_manifest.get("shard_count", 0) >= 1
    assert any("auto-repair" in rec.message.lower() for rec in caplog.records)
