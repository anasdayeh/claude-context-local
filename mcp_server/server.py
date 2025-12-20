"""FastMCP server for Claude Code integration - main entry point."""
import sys

# CRITICAL: Import FAISS before torch to avoid OpenMP runtime conflicts on Apple Silicon
# See: https://github.com/facebookresearch/faiss/issues/2913
import faiss  # noqa: F401
import os
import json
import logging
import warnings
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Ensure we run inside the project virtualenv when invoked with system python
PROJECT_ROOT = Path(__file__).parent.parent
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from mcp.server.fastmcp import FastMCP, Context

from mcp_server.code_search_server import CodeSearchServer

# Keep stdout clean for stdio transport; log to stderr and keep volume low.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

warnings.filterwarnings("ignore")
log_level = os.getenv("CODE_SEARCH_LOG_LEVEL", "WARNING").upper()
if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    log_level = "WARNING"

log_file = os.getenv("CODE_SEARCH_LOG_FILE")
handlers = None
if log_file:
    try:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(log_path)]
    except Exception:
        handlers = [logging.StreamHandler(sys.stderr)]
else:
    handlers = [logging.StreamHandler(sys.stderr)]

logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)
logging.getLogger("mcp").setLevel(getattr(logging, log_level))
logging.getLogger("fastmcp").setLevel(getattr(logging, log_level))


@asynccontextmanager
async def _lifespan(app: FastMCP):
    """FastMCP lifespan hook for startup/shutdown."""
    server._maybe_start_model_preload()
    try:
        yield
    finally:
        pass


mcp = FastMCP("Code Search", log_level=log_level, lifespan=_lifespan)
server = CodeSearchServer()


def _load_strings() -> dict:
    strings_file = Path(__file__).parent / "strings.yaml"
    with open(strings_file, "r") as f:
        data = yaml.safe_load(f)
        assert isinstance(data, dict), "Expected a dict"
        return {
            "tools": data.get("tools", {}),
            "help": data.get("help", ""),
        }


STRINGS = _load_strings()


async def _notify_stats_updated(ctx: Optional[Context]) -> None:
    if ctx is None:
        return
    try:
        await ctx.session.send_resource_updated("search://stats")
    except Exception as e:
        logger.warning(f"Failed to send resource updated notification: {e}")


@mcp.tool(description=STRINGS["tools"].get("search_code", "Search code"))
async def search_code(
    query: str,
    k: int = 5,
    search_mode: str = "auto",
    file_pattern: str = None,
    chunk_type: str = None,
    include_context: bool = True,
    auto_reindex: bool = True,
    max_age_minutes: float = 5,
    ctx: Optional[Context] = None,
) -> str:
    return server.search_code(
        query=query,
        k=k,
        search_mode=search_mode,
        file_pattern=file_pattern,
        chunk_type=chunk_type,
        include_context=include_context,
        auto_reindex=auto_reindex,
        max_age_minutes=max_age_minutes,
    )


@mcp.tool(description=STRINGS["tools"].get("index_directory", "Index a codebase"))
async def index_directory(
    directory_path: str,
    project_name: str = None,
    file_patterns: list[str] = None,
    incremental: bool = True,
    ctx: Optional[Context] = None,
) -> str:
    result = server.index_directory(
        directory_path=directory_path,
        project_name=project_name,
        file_patterns=file_patterns,
        incremental=incremental,
    )
    await _notify_stats_updated(ctx)
    return result


@mcp.tool(description=STRINGS["tools"].get("find_similar_code", "Find similar code"))
async def find_similar_code(
    chunk_id: str,
    k: int = 5,
    ctx: Optional[Context] = None,
) -> str:
    return server.find_similar_code(chunk_id=chunk_id, k=k)


@mcp.tool(description=STRINGS["tools"].get("get_index_status", "Get index status"))
async def get_index_status(ctx: Optional[Context] = None) -> str:
    return server.get_index_status()


@mcp.tool(description=STRINGS["tools"].get("list_projects", "List projects"))
async def list_projects(ctx: Optional[Context] = None) -> str:
    return server.list_projects()


@mcp.tool(description=STRINGS["tools"].get("switch_project", "Switch project"))
async def switch_project(project_path: str, ctx: Optional[Context] = None) -> str:
    result = server.switch_project(project_path=project_path)
    await _notify_stats_updated(ctx)
    return result


@mcp.tool(description=STRINGS["tools"].get("index_test_project", "Index test project"))
async def index_test_project(ctx: Optional[Context] = None) -> str:
    result = server.index_test_project()
    await _notify_stats_updated(ctx)
    return result


@mcp.tool(description=STRINGS["tools"].get("clear_index", "Clear index"))
async def clear_index(ctx: Optional[Context] = None) -> str:
    result = server.clear_index()
    await _notify_stats_updated(ctx)
    return result


@mcp.resource("search://stats")
def get_search_statistics() -> str:
    """Get detailed search index statistics."""
    try:
        index_manager = server.get_index_manager()
        stats = index_manager.get_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get statistics: {str(e)}"})


@mcp.prompt()
def search_help() -> str:
    """Get help on using code search tools."""
    return STRINGS["help"]


def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Search MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol to use (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transport (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )

    args = parser.parse_args()
    transport = args.transport
    if transport == "http":
        transport = "streamable-http"

    if transport in {"http", "streamable-http", "sse"}:
        logger.info(f"Starting HTTP server on {args.host}:{args.port} ({transport})")
        mcp.run(transport=transport, host=args.host, port=args.port)
    else:
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
