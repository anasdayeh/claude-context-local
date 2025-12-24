"""Shared MCP tool/resource registration."""

import asyncio
import logging
import json
import os
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor

from mcp.server.fastmcp import FastMCP, Context

from mcp_server.code_search_server import CodeSearchServer

logger = logging.getLogger(__name__)


def _extract_progress_token(ctx: Optional[Context]) -> Any | None:
    """Extract progress token from context metadata or attributes."""
    if ctx is None:
        return None
    
    # Try request_context first (often where meta lives)
    rc = getattr(ctx, "request_context", None)
    meta = getattr(rc, "meta", None) if rc else getattr(ctx, "meta", None)
    
    if isinstance(meta, dict):
        token = meta.get("progressToken") or meta.get("progress_token")
        if token:
            return token
            
    # Fallback to direct attributes
    for attr in ("progress_token", "progressToken"):
        token = getattr(rc, attr, None) if rc else getattr(ctx, attr, None)
        if token:
            return token
    return None


def _extract_request_id(ctx: Optional[Context]) -> Any | None:
    """Extract request ID for relating notifications to requests."""
    if ctx is None:
        return None
    rc = getattr(ctx, "request_context", None)
    return getattr(rc, "request_id", None) or getattr(ctx, "request_id", None)


async def _send_progress(ctx: Optional[Context], message: str, progress: Optional[int] = None, total: Optional[int] = None) -> None:
    if ctx is None:
        return
    session = getattr(ctx, "session", None)
    if session is None:
        return
    sender = getattr(session, "send_progress_notification", None)
    if sender is None:
        return
    
    token = _extract_progress_token(ctx)
    if token is None:
        return
        
    req_id = _extract_request_id(ctx)
    
    try:
        # Try newer signature first (supports related_request_id), fall back safely
        import inspect
        sig = inspect.signature(sender)
        if "related_request_id" in sig.parameters:
            await sender(
                progress_token=token, 
                progress=progress, 
                total=total, 
                message=message,
                related_request_id=req_id
            )
        else:
            await sender(progress_token=token, progress=progress, total=total, message=message)
    except Exception as exc:
        logger.debug("Progress notification failed: %s", exc)


def _coerce_result(result):
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return {"result": result}
    return result


def register_tools(mcp: FastMCP, server: CodeSearchServer, strings: dict, executor: ThreadPoolExecutor) -> None:
    """Register tools/resources/prompts on the given MCP instance."""

    async def _run(func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

    @mcp.tool(description=strings.get("tools", {}).get("search_code", "Search code"))
    async def search_code(
        query: str,
        k: int = 5,
        search_mode: str = "auto",
        file_pattern: str = None,
        chunk_type: str = None,
        include_context: bool = True,
        auto_reindex: bool = False,
        max_age_minutes: float = 5, project_path: str = None,
        ctx: Optional[Context] = None,
    ) -> dict:
        result = await _run(
            server.search_code,
            query,
            k,
            search_mode,
            file_pattern,
            chunk_type,
            include_context,
            auto_reindex,
            max_age_minutes, project_path,
        )
        return _coerce_result(result)

    @mcp.tool(description=strings.get("tools", {}).get("index_directory", "Index a codebase"))
    async def index_directory(
        directory_path: str,
        project_name: str = None,
        file_patterns: list[str] = None,
        incremental: bool = True,
        ctx: Optional[Context] = None,
    ) -> dict:
        await _send_progress(ctx, "indexing started", progress=0)
        result = await _run(
            server.index_directory,
            directory_path,
            project_name,
            file_patterns,
            incremental,
        )
        await _send_progress(ctx, "indexing completed", progress=100)
        
        try:
            # Resource updates for the specific project and the project list
            project_id = server.get_project_storage_dir(directory_path).name
            await ctx.session.send_resource_updated(f"codesearch://projects/{project_id}")
            await ctx.session.send_resource_updated("codesearch://projects/list")
            await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        return _coerce_result(result)

    @mcp.tool(description=strings.get("tools", {}).get("find_similar_code", "Find similar code"))
    async def find_similar_code(
        chunk_id: str,
        k: int = 5,
        ctx: Optional[Context] = None,
    ) -> dict:
        result = await _run(server.find_similar_code, chunk_id, k)
        return _coerce_result(result)

    @mcp.tool(description=strings.get("tools", {}).get("get_index_status", "Get index status"))
    async def get_index_status(ctx: Optional[Context] = None) -> dict:
        result = await _run(server.get_index_status)
        return _coerce_result(result)

    @mcp.tool(description=strings.get("tools", {}).get("list_projects", "List projects"))
    async def list_projects(ctx: Optional[Context] = None) -> dict:
        result = await _run(server.list_projects)
        return _coerce_result(result)

    @mcp.tool(description=strings.get("tools", {}).get("switch_project", "Switch project"))
    async def switch_project(project_path: str, ctx: Optional[Context] = None) -> dict:
        result = await _run(server.switch_project, project_path)
        try:
            await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        return _coerce_result(result)

    @mcp.tool(description=strings.get("tools", {}).get("index_test_project", "Index test project"))
    async def index_test_project(ctx: Optional[Context] = None) -> dict:
        await _send_progress(ctx, "indexing test project", progress=0)
        result = await _run(server.index_test_project)
        await _send_progress(ctx, "indexing completed", progress=100)
        try:
            await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        return _coerce_result(result)

    @mcp.tool(description=strings.get("tools", {}).get("clear_index", "Clear index"))
    async def clear_index(ctx: Optional[Context] = None) -> dict:
        result = await _run(server.clear_index)
        try:
            await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        return _coerce_result(result)

    @mcp.resource("search://stats")
    def get_search_statistics() -> str:
        try:
            project_path = server.current_project_path() or os.getcwd()
            project_dir = server.get_project_storage_dir(project_path)
            stats_path = project_dir / "index" / "stats.json"
            if stats_path.exists():
                return stats_path.read_text()
            return json.dumps({"error": "No stats available"})
        except Exception as e:
            return json.dumps({"error": f"Failed to get statistics: {str(e)}"})

    @mcp.prompt()
    def search_help() -> str:
        return strings.get("help", "")
