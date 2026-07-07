from __future__ import annotations

from pathlib import Path

import pytest

import mcp_server.mcp_tools as mcp_tools


ROOT = Path(__file__).resolve().parents[2]


def test_production_code_no_longer_uses_legacy_fastmcp_import_path():
    production_files = [
        ROOT / "mcp_server" / "server.py",
        ROOT / "mcp_server" / "mcp_tools.py",
    ]

    for path in production_files:
        content = path.read_text()
        assert "from mcp.server.fastmcp import" not in content, path.name
        assert "import mcp.server.fastmcp" not in content, path.name


@pytest.mark.anyio
async def test_send_progress_prefers_context_report_progress():
    calls: list[dict] = []

    class DummyContext:
        request_id = "req-1"

        async def report_progress(self, *, progress=None, total=None, message=None):
            calls.append(
                {
                    "progress": progress,
                    "total": total,
                    "message": message,
                }
            )

    await mcp_tools._send_progress(DummyContext(), "indexing started", progress=1, total=10)

    assert calls == [
        {
            "progress": 1,
            "total": 10,
            "message": "indexing started",
        }
    ]
