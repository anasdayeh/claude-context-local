def test_code_search_device_env(monkeypatch):
    from mcp_server.code_search_server import CodeSearchServer

    monkeypatch.setenv("CODE_SEARCH_DEVICE", "cpu")
    server = CodeSearchServer()
    assert getattr(server.embedder._model, "device", None) == "cpu"
