import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

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


class StorageServer:
    def __init__(self, storage_dir: Path):
        self._storage_dir = storage_dir
        self._current_project = str(storage_dir)

    def current_project_path(self):
        return self._current_project

    def get_project_storage_dir(self, project_path: str | None = None):
        if project_path is None or project_path == self._current_project:
            return self._storage_dir
        raise FileNotFoundError("unknown project")


@pytest.mark.asyncio
async def test_fts_status_reports_coverage(tmp_path):
    project_root = tmp_path / "proj"
    index_dir = project_root / "index"
    shard_dir = index_dir / "shards" / "shard_000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "shards": [{"index_bytes": 2048}]})
    )
    stats = {
        "total_chunks": 10,
        "fts_rows": 5,
        "storage_size": 4096,
        "last_indexed": "2025-01-01T00:00:00",
    }
    (index_dir / "stats.json").write_text(json.dumps(stats))
    (shard_dir / "code.index").write_text("shard")
    (shard_dir / "metadata.db").write_text("meta")
    (shard_dir / "stats.json").write_text(
        json.dumps({"total_chunks": 5, "fts_rows": 5})
    )

    server = StorageServer(project_root)
    mcp = FakeMCP()
    executor = ThreadPoolExecutor(max_workers=1)

    register_tools(mcp, server, strings={}, executor=executor)
    fts_fn = mcp.tools["fts_status"]
    result = await fts_fn(project_path=str(project_root))

    assert result.get("ok") is True
    payload = result.get("result") or result.get("payload") or {}
    assert payload.get("coverage_pct") == 50.0
    assert payload.get("fts_rows") == 5
    assert payload.get("total_chunks") == 10
    assert payload.get("manifest")["version"] == 1
    assert payload.get("shards")
    shard = payload["shards"][0]
    assert shard["coverage_pct"] == 100.0
