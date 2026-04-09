"""Legacy MCP wrapper retained for test compatibility."""

import logging
from concurrent.futures import ThreadPoolExecutor

try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None

from mcp_server.mcp_tools import register_tools
from mcp_server.code_search_server import CodeSearchServer
from mcp_server.strings_loader import load_strings

logger = logging.getLogger(__name__)


class CodeSearchMCP:
    """Compatibility wrapper around the shared MCP tool registration."""

    def __init__(self, server: "CodeSearchServer"):
        self._app = FastMCP("Code Search") if FastMCP else None
        self.server = server
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-code-search")
        strings = load_strings()
        register_tools(self, server, strings, self._executor)

    def __getattr__(self, name):
        if self._app is None:
            raise AttributeError(name)
        return getattr(self._app, name)

    def tool(self, *args, **kwargs):
        if self._app is None:
            raise RuntimeError("FastMCP not installed")
        return self._app.tool(*args, **kwargs)

    def resource(self, *args, **kwargs):
        if self._app is None:
            raise RuntimeError("FastMCP not installed")
        return self._app.resource(*args, **kwargs)

    def prompt(self, *args, **kwargs):
        if self._app is None:
            raise RuntimeError("FastMCP not installed")
        return self._app.prompt(*args, **kwargs)

    def run(self, transport: str = "stdio", host: str = "localhost", port: int = 8000):
        """Run the MCP server with specified transport."""
        if self._app is None:
            logger.error("FastMCP not installed. Cannot run server.")
            return

        if transport == "http":
            transport = "streamable-http"

        if transport in ["sse", "streamable-http"]:
            logger.info(f"Starting HTTP server on {host}:{port}")
            return self._app.run(
                transport=transport,
                host=host,
                port=port,
                show_banner=False,
            )
        return self._app.run(transport=transport, show_banner=False)
