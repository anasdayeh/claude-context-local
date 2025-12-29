import logging


def test_server_logs_sanity_warning(tmp_path, monkeypatch, caplog):
    from mcp_server.code_search_server import CodeSearchServer
    from search.indexer import CodeIndexManager

    monkeypatch.setenv("CODE_SEARCH_STORAGE", str(tmp_path))
    server = CodeSearchServer()

    project_path = str(tmp_path / "project")
    index_dir = server.get_project_storage_dir(project_path) / "index"
    manager = CodeIndexManager(str(index_dir))

    manager.metadata_db["1"] = {
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
    manager.metadata_db.commit()

    with caplog.at_level(logging.WARNING):
        stats = server.get_stats(project_path=project_path)

    assert "sanity_warning" in stats
    assert any("sanity" in record.message.lower() for record in caplog.records)
