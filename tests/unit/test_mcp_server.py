"""Unit tests for MCP server functionality."""

import pytest


class TestMCPServerImport:
    """Test that MCP server can be imported."""

    def test_mcp_server_can_import(self):
        """Test that MCP server module can be imported without errors."""
        try:
            import mcp_server.code_search_server
            import mcp_server.code_search_mcp 
            assert True  # If we get here, import succeeded
        except ImportError as e:
            pytest.fail(f"Failed to import MCP server: {e}")


def test_preload_hook_exists_and_calls_embed_query(monkeypatch):
    import mcp_server.code_search_server as css

    server = css.CodeSearchServer()
    called = {"ok": False}

    class DummyEmbedder:
        def warmup(self, text):
            called["ok"] = True
            return True

        def health_status(self):
            return {"status": "ready", "backend": "torch", "device": "cpu", "error": None}

    server.embedder = DummyEmbedder()
    assert hasattr(server, "_maybe_start_model_preload")
    server._maybe_start_model_preload()
    assert called["ok"] is True


def test_runtime_selftest_records_embedder_failure(monkeypatch):
    import mcp_server.code_search_server as css

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            self._status = {
                "status": "failed",
                "backend": "torch",
                "device": "cpu",
                "error": "boom",
                "model_name": "dummy",
            }

        def warmup(self, probe: str = "healthcheck") -> bool:
            return False

        def health_status(self):
            return dict(self._status)

        def get_model_info(self):
            return dict(self._status)

        def cleanup(self):
            pass

    monkeypatch.setenv("CODE_SEARCH_RUNTIME_SELFTEST", "1")
    monkeypatch.setattr(css, "CodeEmbedder", DummyEmbedder)

    server = css.CodeSearchServer()
    status = server.get_embedder_status()

    assert status["status"] == "failed"
    assert status["error"] == "boom"


# Note: Most MCP server functionality is tested in integration tests
# where the actual decorators and FastMCP framework are working properly.
# Unit tests here would just be testing mocks, not real functionality.


def test_main_stdio_run_disables_fastmcp_banner(monkeypatch):
    import mcp_server.server as server_mod

    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(server_mod.mcp, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["server.py", "--transport", "stdio"],
    )

    server_mod.main()

    assert calls == [{"transport": "stdio", "show_banner": False}]


def test_main_http_run_disables_fastmcp_banner(monkeypatch):
    import mcp_server.server as server_mod

    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(server_mod.mcp, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["server.py", "--transport", "http", "--host", "127.0.0.1", "--port", "8011"],
    )

    server_mod.main()

    assert calls == [
        {
            "transport": "streamable-http",
            "host": "127.0.0.1",
            "port": 8011,
            "show_banner": False,
        }
    ]
