"""search_code, get_index_status, and get_stats must surface the active embedding
model in their response `meta`, so a Gemma result is never mistaken for a Qwen one
mid-session (and vice-versa) during the Phase-4 A/B.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


class FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, description=None):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco

    def resource(self, name):
        def deco(fn):
            return fn
        return deco

    def prompt(self, name=None):
        def deco(fn):
            return fn
        return deco


class DummyServer:
    def __init__(self):
        self._current_project = "/tmp/proj"

    def current_project_path(self):
        return self._current_project

    def get_project_storage_dir(self, path):
        return Path("/tmp/storage/abc123")

    def get_embedder_status(self):
        return {
            "status": "loaded",
            "backend": "torch",
            "error": None,
            "model_name": "google/embeddinggemma-300m",
        }

    def search_code(self, *args, **kw):
        return {"results": []}

    def get_index_status(self, project_path=None):
        return {"status": "ready", "files_indexed": 3}

    def get_stats(self, project_path=None):
        return {"total_chunks": 8, "embedding_dim": 768}


def _register():
    from mcp_server.mcp_tools import register_tools
    mcp = FakeMCP()
    register_tools(mcp, DummyServer(), strings={}, executor=ThreadPoolExecutor(max_workers=1))
    return mcp


@pytest.mark.anyio
async def test_search_code_meta_has_embedding_model():
    mcp = _register()
    result = await mcp.tools["search_code"](query="x")
    assert result["meta"]["embedding_model"] == "google/embeddinggemma-300m"


@pytest.mark.anyio
async def test_get_index_status_meta_has_embedding_model():
    mcp = _register()
    result = await mcp.tools["get_index_status"]()
    assert result["meta"]["embedding_model"] == "google/embeddinggemma-300m"


@pytest.mark.anyio
async def test_get_stats_meta_has_embedding_model():
    mcp = _register()
    result = await mcp.tools["get_stats"]()
    assert result["meta"]["embedding_model"] == "google/embeddinggemma-300m"
