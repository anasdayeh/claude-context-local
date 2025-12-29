import asyncio
import pytest

from concurrent.futures import ThreadPoolExecutor

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
