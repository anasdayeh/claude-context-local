import asyncio
import pytest

from concurrent.futures import ThreadPoolExecutor

import mcp_server.mcp_tools as mcp_tools
from mcp_server.mcp_tools import register_tools


class FakeMCP:
    def __init__(self):
        self.tools = {}
        self.resources = {}

    def tool(self, description=None):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator

    def resource(self, name):
        def decorator(fn):
            self.resources[name] = fn
            return fn
        return decorator

    def prompt(self, name=None):
        def decorator(fn):
            return fn
        return decorator


class DummyServer:
    def __init__(self):
        self.as_dict = None

    def search_code(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "auto",
        file_pattern: str = None,
        chunk_type: str = None,
        include_context: bool = True,
        auto_reindex: bool = False,
        max_age_minutes: float = 5,
        project_path: str = None,
        as_dict: bool = True,
    ):
        self.as_dict = as_dict
        return {"results": []}


@pytest.mark.asyncio
async def test_mcp_tool_search_code_returns_dict():
    mcp = FakeMCP()
    server = DummyServer()
    executor = ThreadPoolExecutor(max_workers=1)

    register_tools(mcp, server, strings={}, executor=executor)

    search_fn = mcp.tools["search_code"]
    result = await search_fn(query="test")

    assert server.as_dict is True
    assert isinstance(result, dict)
    assert "results" in result


@pytest.mark.asyncio
async def test_search_code_meta_includes_fts_status(monkeypatch):
    mcp = FakeMCP()
    server = DummyServer()
    executor = ThreadPoolExecutor(max_workers=1)

    stub_payload = {
        "project_path": "/tmp/proj",
        "manifest_path": "/tmp/proj/index/manifest.json",
        "manifest": {"project_path": "/tmp/proj"},
        "stats_path": "/tmp/proj/index/stats.json",
        "coverage_pct": 62.5,
        "fts_rows": 5,
        "total_chunks": 8,
        "manifest_index_bytes": 1024,
        "warnings": ["test warning"],
        "last_indexed": "2025-01-01T00:00:00",
        "last_indexed_source": "stats_json",
    }
    monkeypatch.setattr(mcp_tools, "_build_fts_status_payload", lambda *_: stub_payload)

    register_tools(mcp, server, strings={}, executor=executor)

    search_fn = mcp.tools["search_code"]
    result = await search_fn(query="meta-test")
    meta = result.get("meta", {})
    assert meta.get("fts_status") is stub_payload
    assert meta.get("fts_coverage_pct") == stub_payload["coverage_pct"]
    assert meta.get("fts_rows") == stub_payload["fts_rows"]
    assert meta.get("total_chunks") == stub_payload["total_chunks"]
    assert meta.get("manifest_project_path") == stub_payload["manifest"]["project_path"]
    assert meta.get("manifest_index_bytes") == stub_payload["manifest_index_bytes"]
    assert meta.get("manifest_path") == stub_payload["manifest_path"]
