"""Process-level regression tests for MCP stdio transport stability."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stdio_search_code_keeps_transport_alive(tmp_path):
    log_file = tmp_path / "mcp-server.log"
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "mcp_server" / "server.py"), "--transport", "stdio"],
        env={
            **os.environ,
            "CODE_SEARCH_LOG_FILE": str(log_file),
            "CODE_SEARCH_LOG_LEVEL": "INFO",
            "CODE_SEARCH_RUNTIME_SELFTEST": "1",
        },
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            index_result = await session.call_tool("index_test_project", {})
            assert index_result is not None

            search_result = await session.call_tool(
                "search_code",
                {
                    "query": "hello world",
                    "k": 3,
                    "search_mode": "semantic",
                    "include_context": False,
                },
            )
            assert search_result is not None

            # A second tool call verifies the transport stayed open after search_code.
            projects_result = await session.call_tool("list_projects", {})
            assert projects_result is not None


@pytest.mark.integration
def test_supported_import_order_sentence_transformer_before_faiss():
    hf_home = Path(os.environ.get("HF_HOME", "")).expanduser()
    configured_cache = hf_home / "hub" if str(hf_home) not in {"", "."} else ROOT / "models"
    code = f"""
from pathlib import Path
from sentence_transformers import SentenceTransformer

cache_root = Path({str(configured_cache)!r})
model = SentenceTransformer(
    'google/embeddinggemma-300m',
    cache_folder=str(cache_root),
    trust_remote_code=True,
    device='cpu',
)
import faiss
print(model.get_sentence_embedding_dimension())
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONFAULTHANDLER": "1"},
        timeout=180,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "768" in result.stdout
