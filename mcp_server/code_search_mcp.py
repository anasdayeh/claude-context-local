"""Legacy MCP wrapper retained for test compatibility."""

import logging
from concurrent.futures import ThreadPoolExecutor

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None

from mcp_server.mcp_tools import register_tools
from mcp_server.code_search_server import CodeSearchServer
from mcp_server.strings_loader import load_strings

logger = logging.getLogger(__name__)


class CodeSearchMCP(FastMCP if FastMCP else object):
    """Compatibility wrapper around the shared MCP tool registration."""

    def __init__(self, server: "CodeSearchServer"):
        if FastMCP:
            super().__init__("Code Search")
        self.server = server
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-code-search")
        strings = load_strings()
        register_tools(self, server, strings, self._executor)

    def run(self, transport: str = "stdio", host: str = "localhost", port: int = 8000):
        """Run the MCP server with specified transport."""
        if not FastMCP:
            logger.error("FastMCP not installed. Cannot run server.")
            return

        if transport == "http":
            transport = "streamable-http"

        if transport in ["sse", "streamable-http"]:
            logger.info(f"Starting HTTP server on {host}:{port}")
        return super().run(transport=transport)
