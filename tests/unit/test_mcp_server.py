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
        def embed_query(self, text):
            called["ok"] = True
            return [0.0]

    server.embedder = DummyEmbedder()
    assert hasattr(server, "_maybe_start_model_preload")
    server._maybe_start_model_preload()
    assert called["ok"] is True


# Note: Most MCP server functionality is tested in integration tests
# where the actual decorators and FastMCP framework are working properly.
# Unit tests here would just be testing mocks, not real functionality.
