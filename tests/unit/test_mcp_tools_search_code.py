import asyncio
import pytest

from concurrent.futures import ThreadPoolExecutor

import mcp_server.mcp_tools as mcp_tools
from mcp_server.mcp_tools import register_tools


class FakeMCP:
    def __init__(self):
        self.tools = {}
        self.resources = {}
        self.resource_templates = {}
        self.prompts = {}

    def tool(self, description=None):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator

    def resource(self, name):
        def decorator(fn):
            if "{" in name and "}" in name:
                self.resource_templates[name] = fn
            else:
                self.resources[name] = fn
            return fn
        return decorator

    def prompt(self, name=None):
        def decorator(fn):
            self.prompts[name or fn.__name__] = fn
            return fn
        return decorator

    async def list_tools(self):
        return [type("Tool", (), {"name": name, "description": ""})() for name in self.tools]

    async def list_resources(self):
        return [type("Resource", (), {"uri": uri})() for uri in self.resources]

    async def list_resource_templates(self):
        return [type("ResourceTemplate", (), {"uri_template": uri})() for uri in self.resource_templates]

    async def list_prompts(self):
        return [type("Prompt", (), {"name": name, "description": ""})() for name in self.prompts]


class DummyServer:
    def __init__(self):
        self._current_project = "/tmp/project"

    def current_project_path(self):
        return self._current_project

    def get_project_storage_dir(self, path):
        from pathlib import Path
        return Path("/tmp/storage/abc123")

    def search_code(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "auto",
        file_patterns=None,
        file_pattern: str = None,
        chunk_type: str = None,
        include_context: bool = True,
        auto_reindex: bool = False,
        max_age_minutes: float = 5,
        project_path: str = None,
    ):
        return {
            "results": [],
            "semantic_available": False,
            "fallback_mode": "fts",
            "error_code": "embedder_init_failed",
        }

    def get_embedder_status(self):
        return {
            "status": "failed",
            "backend": "torch",
            "error": "embedder bootstrap failed",
        }


@pytest.mark.anyio
async def test_mcp_tool_search_code_returns_dict():
    mcp = FakeMCP()
    server = DummyServer()
    executor = ThreadPoolExecutor(max_workers=1)

    register_tools(mcp, server, strings={}, executor=executor)

    search_fn = mcp.tools["search_code"]
    result = await search_fn(query="test")

    assert isinstance(result, dict)
    assert "results" in result


@pytest.mark.anyio
async def test_search_code_meta_includes_fts_scalars_not_full_payload(monkeypatch):
    """Test that FTS scalar fields are included in meta, but not the full fts_status payload."""
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

    # Scalar fields should be present
    assert meta.get("fts_coverage_pct") == stub_payload["coverage_pct"]
    assert meta.get("fts_rows") == stub_payload["fts_rows"]
    assert meta.get("total_chunks") == stub_payload["total_chunks"]
    assert meta.get("manifest_project_path") == stub_payload["manifest"]["project_path"]
    assert meta.get("manifest_index_bytes") == stub_payload["manifest_index_bytes"]
    assert meta.get("manifest_path") == stub_payload["manifest_path"]
    assert meta.get("embedder_status") == "failed"
    assert meta.get("embedder_backend") == "torch"
    assert meta.get("embedder_failure_summary") == "embedder bootstrap failed"

    # Full fts_status payload should NOT be included (context window bloat fix)
    assert meta.get("fts_status") is None
    # Full manifest should not be in meta
    assert meta.get("manifest") is None


@pytest.mark.anyio
async def test_search_code_preserves_structured_embedder_failure_fields():
    mcp = FakeMCP()
    server = DummyServer()
    executor = ThreadPoolExecutor(max_workers=1)

    register_tools(mcp, server, strings={}, executor=executor)

    search_fn = mcp.tools["search_code"]
    result = await search_fn(query="failure-shape")

    assert result.get("error_code") == "embedder_init_failed"
    assert result.get("semantic_available") is False
    assert result.get("fallback_mode") == "fts"
