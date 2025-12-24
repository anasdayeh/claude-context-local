"""FastMCP server for Codex integration - main entry point."""
import sys
import os
import logging
import warnings
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Ensure we run inside the project virtualenv when invoked with system python
PROJECT_ROOT = Path(__file__).parent.parent
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Keep stdout clean for stdio transport; log to stderr and keep volume low.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TQDM_DISABLE", "1")

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

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from mcp_server.code_search_server import CodeSearchServer
from mcp_server.mcp_tools import register_tools
from mcp_server.strings_loader import load_strings

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-code-search")


@asynccontextmanager
async def _lifespan(app: FastMCP):
    """FastMCP lifespan hook for startup/shutdown."""
    if os.getenv("CODE_SEARCH_PRELOAD_MODEL", "").lower() in {"1", "true", "yes"}:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(_EXECUTOR, server._maybe_start_model_preload)
    try:
        yield
    finally:
        pass


mcp = FastMCP("Code Search", log_level=log_level, lifespan=_lifespan)
server = CodeSearchServer()
strings = load_strings()
register_tools(mcp, server, strings, _EXECUTOR)


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
        default="127.0.0.1",
        help="Host for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)",
    )

    args = parser.parse_args()
    
    # Normalize transport
    transport = args.transport.lower()
    if transport == "http":
        transport = "streamable-http"

    if transport in {"streamable-http", "sse"}:
        # Security Note: Binding to 0.0.0.0 exposes the server to the network.
        # Ensure proper network-level authorization if used outside localhost.
        if args.host == "0.0.0.0":
            logger.warning("Server is binding to 0.0.0.0. Ensure network access is protected.")
            
        logger.info(f"Starting HTTP server on {args.host}:{args.port} ({transport})")
        mcp.run(transport=transport, host=args.host, port=args.port)
    else:
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
