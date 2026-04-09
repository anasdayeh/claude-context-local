from concurrent.futures import ThreadPoolExecutor

import pytest

from mcp_server.mcp_tools import register_tools


class _FakeTool:
    def __init__(self, name: str, fn, description: str | None):
        self.name = name
        self.fn = fn
        self.description = description or ""


class _FakeToolManager:
    def __init__(self):
        self._tools = {}


class _FakeResourceManager:
    def __init__(self):
        self._resources = {}


class _FakePrompt:
    def __init__(self, fn, description: str | None = None):
        self.fn = fn
        self.description = description or ""


class _FakePromptManager:
    def __init__(self):
        self._prompts = {}


class FakeMCP:
    def __init__(self):
        self.tools = {}
        self.resources = {}
        self.prompts = {}
        self._tool_manager = _FakeToolManager()
        self._resource_manager = _FakeResourceManager()
        self._prompt_manager = _FakePromptManager()

    def tool(self, description=None):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            self._tool_manager._tools[fn.__name__] = _FakeTool(fn.__name__, fn, description)
            return fn

        return decorator

    def resource(self, name):
        def decorator(fn):
            self.resources[name] = fn
            # Mimic FastMCP internal registration keyed by resource URI/pattern.
            self._resource_manager._resources[name] = fn
            return fn

        return decorator

    def prompt(self, name=None):
        def decorator(fn):
            prompt_name = name or fn.__name__
            self.prompts[prompt_name] = fn
            self._prompt_manager._prompts[prompt_name] = _FakePrompt(fn)
            return fn

        return decorator

    async def list_tools(self):
        return list(self._tool_manager._tools.values())

    async def list_resources(self):
        return [type("Resource", (), {"uri": uri})() for uri in self.resources if "{" not in uri]

    async def list_resource_templates(self):
        return [type("ResourceTemplate", (), {"uri_template": uri})() for uri in self.resources if "{" in uri]

    async def list_prompts(self):
        return [type("Prompt", (), {"name": name, "description": prompt.description})() for name, prompt in self._prompt_manager._prompts.items()]


class DummyServer:
    def __init__(self):
        self._current_project = "/tmp/project"

    def current_project_path(self):
        return self._current_project

    def get_project_storage_dir(self, project_path: str):
        raise FileNotFoundError("not used")

    def list_projects(self, as_dict: bool = True):
        projects = [{"project_id": "p1", "project_path": "/tmp/project"}]
        return {"count": 1, "projects": projects} if as_dict else projects

    def find_similar_code(self, chunk_id: str, k: int = 5):
        return [{"chunk_id": "c2", "score": 0.5}]

    def get_embedder_status(self):
        return {"status": "ready", "backend": "torch", "error": None}


@pytest.mark.asyncio
async def test_tool_surface_has_no_v2_names():
    mcp = FakeMCP()
    server = DummyServer()
    executor = ThreadPoolExecutor(max_workers=1)

    register_tools(mcp, server, strings={}, executor=executor)

    tool_names = set(mcp.tools.keys())
    assert not any(name.endswith("_v2") for name in tool_names)


@pytest.mark.asyncio
async def test_list_tools_includes_templates_and_prompts():
    mcp = FakeMCP()
    server = DummyServer()
    executor = ThreadPoolExecutor(max_workers=1)

    register_tools(mcp, server, strings={}, executor=executor)
    list_tools = mcp.tools["list_tools"]
    payload = await list_tools()

    assert payload.get("ok") is True
    assert "tools" in payload
    assert "resources" in payload
    assert "resource_templates" in payload
    assert "prompts" in payload
    assert payload.get("embedder_status") == "ready"
    assert payload.get("embedder_backend") == "torch"

    template_uris = {t.get("uri") for t in (payload.get("resource_templates") or [])}
    assert "codesearch://projects/{project_id}" in template_uris


@pytest.mark.asyncio
async def test_list_projects_returns_normalized_shape():
    mcp = FakeMCP()
    server = DummyServer()
    executor = ThreadPoolExecutor(max_workers=1)

    register_tools(mcp, server, strings={}, executor=executor)
    fn = mcp.tools["list_projects"]
    result = await fn()

    assert result.get("ok") is True
    assert isinstance(result.get("meta"), dict)
    assert isinstance(result.get("result"), list)


@pytest.mark.asyncio
async def test_find_similar_code_returns_normalized_shape():
    mcp = FakeMCP()
    server = DummyServer()
    executor = ThreadPoolExecutor(max_workers=1)

    register_tools(mcp, server, strings={}, executor=executor)
    fn = mcp.tools["find_similar_code"]
    result = await fn(chunk_id="c1")

    assert result.get("ok") is True
    assert isinstance(result.get("meta"), dict)
    assert isinstance(result.get("result"), list)
