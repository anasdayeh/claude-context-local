from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stdio_search_code_survives_repeated_calls(tmp_path):
    log_file = tmp_path / "mcp-server-repeated.log"
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "mcp_server" / "server.py"), "--transport", "stdio"],
        env={
            **os.environ,
            "CODE_SEARCH_LOG_FILE": str(log_file),
            "CODE_SEARCH_LOG_LEVEL": "INFO",
            "CODE_SEARCH_RUNTIME_SELFTEST": "1",
            "CODE_SEARCH_DEVICE": "cpu",
        },
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("index_test_project", {})

            for i in range(20):
                result = await session.call_tool(
                    "search_code",
                    {
                        "query": f"hello world {i}",
                        "k": 2,
                        "search_mode": "semantic",
                        "include_context": False,
                    },
                )
                assert result is not None

            status = await session.call_tool("get_index_status", {})
            assert status is not None
