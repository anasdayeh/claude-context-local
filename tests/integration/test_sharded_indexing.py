from pathlib import Path


def test_sharded_indexing_creates_manifest(tmp_path, monkeypatch):
    from mcp_server.code_search_server import CodeSearchServer

    monkeypatch.setenv("CODE_SEARCH_SHARDED_INDEX", "1")

    server = CodeSearchServer()
    result = server.index_directory(str(tmp_path), project_name="tmp", incremental=False)
    assert result.get("success") is True

    project_dir = server.get_project_storage_dir(str(tmp_path))
    manifest_path = project_dir / "index" / "manifest.json"
    assert manifest_path.exists()

    shards_dir = project_dir / "index" / "shards"
    assert shards_dir.exists()
