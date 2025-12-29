"""Shared MCP tool/resource registration."""

import asyncio
import logging
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor

from mcp.server.fastmcp import FastMCP, Context
from merkle.merkle_dag import MerkleDAG

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

    def _get_active_project_path() -> Optional[str]:
        getter = getattr(server, "current_project_path", None)
        try:
            if callable(getter):
                return getter()
        except Exception:
            pass
        return getattr(server, "_current_project", None)

    def _read_stats_json(project_path: str) -> dict:
        try:
            project_dir = server.get_project_storage_dir(project_path)
            stats_path = project_dir / "index" / "stats.json"
            if not stats_path.exists():
                return {}
            return json.loads(stats_path.read_text())
        except Exception:
            return {}

    def _infer_last_indexed(project_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """Return (last_indexed_iso, source). Best-effort and cheap."""
        if not project_path:
            return None, None

        stats = _read_stats_json(project_path)
        li = stats.get("last_indexed")
        if isinstance(li, str) and li.strip():
            return li, "stats_json"

        try:
            project_dir = server.get_project_storage_dir(project_path)
            index_dir = project_dir / "index"
            stats_path = index_dir / "stats.json"
            if stats_path.exists():
                ts = stats_path.stat().st_mtime
                return datetime.fromtimestamp(ts).isoformat(), "stats_mtime"

            # Fall back to vector index mtime.
            index_path = index_dir / "code.index"
            if index_path.exists():
                ts = index_path.stat().st_mtime
                return datetime.fromtimestamp(ts).isoformat(), "index_mtime"

            shards_root = index_dir / "shards"
            if shards_root.exists():
                newest = None
                for shard in shards_root.glob("shard_*"):
                    p = shard / "code.index"
                    if not p.exists():
                        continue
                    mtime = p.stat().st_mtime
                    newest = mtime if newest is None else max(newest, mtime)
                if newest is not None:
                    return datetime.fromtimestamp(newest).isoformat(), "sharded_index_mtime"
        except Exception:
            pass

        return None, None

    def _base_meta(*, did_auto_switch: bool | None = None, project_path_used: str | None = None) -> dict:
        active_path = _get_active_project_path()
        if project_path_used:
            active_path = project_path_used
        active_id = None
        if active_path:
            try:
                active_id = server.get_project_storage_dir(active_path).name
            except Exception:
                active_id = None
        meta = {
            "active_project_path": active_path,
            "active_project_id": active_id,
        }
        if did_auto_switch is not None:
            meta["did_auto_switch"] = bool(did_auto_switch)
        return meta

    def _error_info(*, code: str, message: str, suggestion: str | None = None, details: dict | None = None) -> dict:
        payload = {"code": code, "message": message}
        if suggestion:
            payload["suggestion"] = suggestion
        if details:
            payload["details"] = details
        return payload

    def _augment_dict_response(
        *,
        tool_name: str,
        response: dict,
        meta: dict | None = None,
        result_value: object | None = None,
    ) -> dict:
        # Strictly additive: do not remove or rename any existing fields.
        out = response

        if "ok" not in out:
            if "success" in out:
                out["ok"] = bool(out.get("success"))
            elif "error" in out:
                out["ok"] = False
            else:
                out["ok"] = True

        if meta:
            # Never overwrite an existing meta field if one is already present.
            existing = out.get("meta")
            if isinstance(existing, dict):
                merged = dict(existing)
                for k, v in meta.items():
                    merged.setdefault(k, v)
                out["meta"] = merged
            else:
                out["meta"] = meta

        if result_value is not None and "result" not in out:
            out["result"] = result_value

        # Add structured error info without changing existing `error`/`suggestion` fields.
        if out.get("ok") is False and "error_info" not in out:
            message = str(out.get("error") or "Unknown error")
            suggestion = out.get("suggestion")
            out["error_info"] = _error_info(
                code=f"{tool_name.upper()}_ERROR",
                message=message,
                suggestion=str(suggestion) if suggestion else None,
            )

        return out

    def _derive_job_progress(job: dict) -> dict:
        """Best-effort parse of progress events into structured fields."""
        last = str(job.get("last_message") or "")
        progress: dict = {
            "phase": "unknown",
            "percent_estimate": None,
            "files_completed": None,
            "files_total": None,
            "chunks_checkpointed": None,
        }

        status = str(job.get("status") or "")
        if status in {"queued", "running", "completed", "failed", "canceled"}:
            progress["phase"] = status

        m = re.search(r"progress:\s*(\d+)\s*/\s*(\d+)\s*files\s*\(([\d.]+)%\)", last)
        if m:
            progress["phase"] = "indexing"
            progress["files_completed"] = int(m.group(1))
            progress["files_total"] = int(m.group(2))
            try:
                progress["percent_estimate"] = float(m.group(3))
            except Exception:
                progress["percent_estimate"] = None

        m2 = re.search(r"checkpoint:\s*saved\s*(\d+)\s*chunks", last)
        if m2:
            progress["phase"] = "writing_index"
            progress["chunks_checkpointed"] = int(m2.group(1))

        return progress

    def _should_background_index(directory_path: str, file_patterns: Optional[list[str]]) -> bool:
        force_async = os.getenv("CODE_SEARCH_ASYNC_INDEX", "").lower() in {"1", "true", "yes"}
        if force_async:
            return True
        force_sync = os.getenv("CODE_SEARCH_SYNC_INDEX", "").lower() in {"1", "true", "yes"}
        if force_sync:
            return False

        threshold = int(os.getenv("CODE_SEARCH_ASYNC_FILE_THRESHOLD", "2500") or 2500)
        scan_seconds = float(os.getenv("CODE_SEARCH_ASYNC_SCAN_SECONDS", "2") or 2)

        try:
            root = Path(directory_path).resolve()
        except Exception:
            root = Path(directory_path)

        dag = MerkleDAG(str(root))
        counted = 0
        deadline = time.time() + max(0.25, scan_seconds)

        try:
            for current_root, dirs, files in os.walk(str(root)):
                # Respect the same ignore patterns as the MerkleDAG builder.
                dirs[:] = [d for d in dirs if not dag.should_ignore(Path(d))]

                for f in files:
                    if dag.should_ignore(Path(f)):
                        continue
                    counted += 1
                    if counted >= threshold:
                        return True

                if time.time() >= deadline:
                    # Fail-safe: if scanning itself is slow, treat as "large" and avoid blocking tool call.
                    return True
        except Exception:
            # If we can't scan reliably, prefer background indexing to keep tools responsive.
            return True

        return False

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
        before_project = _get_active_project_path()
        result = await _run(
            server.search_code,
            query,
            k,
            search_mode,
            file_pattern,
            chunk_type,
            include_context,
            auto_reindex,
            max_age_minutes, project_path, True,
        )
        coerced = _coerce_result(result)
        if not isinstance(coerced, dict):
            return coerced

        # Determine whether the server auto-switched projects
        after_project = _get_active_project_path()
        did_auto_switch = bool(after_project and before_project != after_project and not project_path)

        # Normalize filters for meta
        fp = None
        if file_pattern:
            fp = [file_pattern] if isinstance(file_pattern, str) else file_pattern
        filters_applied = {"file_pattern": fp, "chunk_type": chunk_type}

        # Determine effective context behavior
        env_include_context = os.getenv("CODE_SEARCH_INCLUDE_CONTEXT", "").lower() in {"1", "true", "yes"}
        include_context_effective = bool(include_context or env_include_context)

        stats = _read_stats_json(after_project) if after_project else {}
        index_last_indexed = stats.get("last_indexed")
        index_last_indexed_source = "stats_json" if index_last_indexed else None
        if not index_last_indexed:
            index_last_indexed, index_last_indexed_source = _infer_last_indexed(after_project)
        index_id = None
        if after_project:
            try:
                index_id = server.get_project_storage_dir(after_project).name
            except Exception:
                index_id = None

        results_list = coerced.get("results") or []
        meta = _base_meta(did_auto_switch=did_auto_switch, project_path_used=after_project)
        meta.update(
            {
                "query": query,
                "k_requested": int(k),
                "k_returned": len(results_list) if isinstance(results_list, list) else None,
                "search_mode_requested": search_mode,
                "search_mode_used": "semantic" if search_mode == "auto" else search_mode,
                "filters_applied": filters_applied,
                "project_path_used": after_project,
                "index_id": index_id,
                "index_last_indexed": index_last_indexed,
                "index_last_indexed_source": index_last_indexed_source,
                "include_context_requested": bool(include_context),
                "include_context_effective": include_context_effective,
                "context_depth": 1 if include_context_effective else 0,
            }
        )

        return _augment_dict_response(
            tool_name="search_code",
            response=coerced,
            meta=meta,
            result_value=results_list if "results" in coerced else None,
        )

    @mcp.tool(description=strings.get("tools", {}).get("index_directory", "Index a codebase"))
    async def index_directory(
        directory_path: str,
        project_name: str = None,
        file_patterns: list[str] = None,
        incremental: bool = True,
        ctx: Optional[Context] = None,
    ) -> dict:
        if _should_background_index(directory_path, file_patterns):
            await _send_progress(ctx, "indexing started (background job)", progress=0)
            result = await _run(
                server.start_index_job,
                directory_path,
                project_name,
                file_patterns,
                incremental,
            )
            try:
                if ctx is not None:
                    await ctx.session.send_resource_updated("codesearch://projects/list")
                    await ctx.session.send_resource_updated("search://stats")
            except Exception:
                pass
            coerced = _coerce_result(result)
            if isinstance(coerced, dict):
                return _augment_dict_response(
                    tool_name="index_directory",
                    response=coerced,
                    meta=_base_meta(),
                    result_value=coerced.get("job") if "job" in coerced else None,
                )
            return coerced

        await _send_progress(ctx, "indexing started", progress=0)
        loop = asyncio.get_running_loop()

        def progress_callback(message: str) -> None:
            if ctx is None:
                return
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    _send_progress(ctx, message, progress=None, total=None),
                    loop,
                )
                # Avoid warnings about never-awaited coroutines
                try:
                    fut.result(timeout=0)
                except Exception:
                    pass
            except Exception:
                pass

        result = await _run(
            server.index_directory,
            directory_path,
            project_name,
            file_patterns,
            incremental,
            progress_callback,
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
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            return _augment_dict_response(
                tool_name="index_directory",
                response=coerced,
                meta=_base_meta(),
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("start_index_directory", "Start indexing as a background job"))
    async def start_index_directory(
        directory_path: str,
        project_name: str = None,
        file_patterns: list[str] = None,
        incremental: bool = True,
        ctx: Optional[Context] = None,
    ) -> dict:
        result = await _run(
            server.start_index_job,
            directory_path,
            project_name,
            file_patterns,
            incremental,
        )
        try:
            if ctx is not None:
                await ctx.session.send_resource_updated("codesearch://projects/list")
                await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            return _augment_dict_response(
                tool_name="start_index_directory",
                response=coerced,
                meta=_base_meta(),
                result_value=coerced.get("job") if "job" in coerced else None,
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("get_index_job_status", "Get status for a background index job"))
    async def get_index_job_status(
        job_id: str = None,
        project_path: str = None,
        ctx: Optional[Context] = None,
    ) -> dict:
        result = await _run(server.get_index_job_status, job_id, project_path)
        coerced = _coerce_result(result)
        if not isinstance(coerced, dict):
            return coerced

        if "job" in coerced and isinstance(coerced.get("job"), dict):
            coerced["job"].setdefault("progress", _derive_job_progress(coerced["job"]))
        if "jobs" in coerced and isinstance(coerced.get("jobs"), list):
            for j in coerced["jobs"]:
                if isinstance(j, dict):
                    j.setdefault("progress", _derive_job_progress(j))

        return _augment_dict_response(
            tool_name="get_index_job_status",
            response=coerced,
            meta=_base_meta(),
            result_value=coerced.get("job") if "job" in coerced else coerced.get("jobs"),
        )

    @mcp.tool(description=strings.get("tools", {}).get("cancel_index_job", "Cancel a background index job"))
    async def cancel_index_job(job_id: str, ctx: Optional[Context] = None) -> dict:
        result = await _run(server.cancel_index_job, job_id)
        coerced = _coerce_result(result)
        if isinstance(coerced, dict) and "job" in coerced and isinstance(coerced.get("job"), dict):
            coerced["job"].setdefault("progress", _derive_job_progress(coerced["job"]))
        if isinstance(coerced, dict):
            return _augment_dict_response(
                tool_name="cancel_index_job",
                response=coerced,
                meta=_base_meta(),
                result_value=coerced.get("job") if isinstance(coerced.get("job"), dict) else None,
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("find_similar_code", "Find similar code"))
    async def find_similar_code(
        chunk_id: str,
        k: int = 5,
        ctx: Optional[Context] = None,
    ) -> dict:
        result = await _run(server.find_similar_code, chunk_id, k)
        return _coerce_result(result)

    @mcp.tool(description=strings.get("tools", {}).get("find_similar_code_v2", "Find similar code (v2)"))
    async def find_similar_code_v2(
        chunk_id: str,
        k: int = 5,
        ctx: Optional[Context] = None,
    ) -> dict:
        result = await _run(server.find_similar_code, chunk_id, k)
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            # Unexpected shape; just augment and return.
            return _augment_dict_response(tool_name="find_similar_code_v2", response=coerced, meta=_base_meta())
        items = coerced if isinstance(coerced, list) else []
        return _augment_dict_response(
            tool_name="find_similar_code_v2",
            response={"results": items},
            meta=_base_meta(),
            result_value=items,
        )

    @mcp.tool(description=strings.get("tools", {}).get("get_index_status", "Get index status"))
    async def get_index_status(ctx: Optional[Context] = None) -> dict:
        result = await _run(server.get_index_status)
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            return _augment_dict_response(
                tool_name="get_index_status",
                response=coerced,
                meta=_base_meta(),
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("get_index_status_v2", "Get index status (v2)"))
    async def get_index_status_v2(project_path: str = None, ctx: Optional[Context] = None) -> dict:
        result = await _run(server.get_index_status, project_path)
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            meta = _base_meta(project_path_used=project_path)
            return _augment_dict_response(
                tool_name="get_index_status_v2",
                response=coerced,
                meta=meta,
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("get_stats", "Get stats (without model info)"))
    async def get_stats(project_path: str = None, ctx: Optional[Context] = None) -> dict:
        result = await _run(server.get_stats, project_path)
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            meta = _base_meta(project_path_used=project_path)
            return _augment_dict_response(
                tool_name="get_stats",
                response=coerced,
                meta=meta,
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("list_projects", "List projects"))
    async def list_projects(ctx: Optional[Context] = None) -> dict:
        result = await _run(server.list_projects, False)
        return _coerce_result(result)

    @mcp.tool(description=strings.get("tools", {}).get("list_projects_v2", "List projects (v2)"))
    async def list_projects_v2(ctx: Optional[Context] = None) -> dict:
        result = await _run(server.list_projects, True)
        coerced = _coerce_result(result)
        if isinstance(coerced, dict) and "projects" in coerced:
            return _augment_dict_response(
                tool_name="list_projects_v2",
                response=coerced,
                meta=_base_meta(),
                result_value=coerced.get("projects"),
            )
        # Fallback: normalize unexpected shape
        projects = coerced if isinstance(coerced, list) else []
        return _augment_dict_response(
            tool_name="list_projects_v2",
            response={"count": len(projects), "projects": projects},
            meta=_base_meta(),
            result_value=projects,
        )

    @mcp.tool(description=strings.get("tools", {}).get("switch_project", "Switch project"))
    async def switch_project(project_path: str, ctx: Optional[Context] = None) -> dict:
        result = await _run(server.switch_project, project_path)
        try:
            await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            # switch_project never auto-switches; it is explicit
            meta = _base_meta(did_auto_switch=False, project_path_used=project_path if coerced.get("success") else None)
            return _augment_dict_response(
                tool_name="switch_project",
                response=coerced,
                meta=meta,
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("index_test_project", "Index test project"))
    async def index_test_project(ctx: Optional[Context] = None) -> dict:
        await _send_progress(ctx, "indexing test project", progress=0)
        result = await _run(server.index_test_project)
        await _send_progress(ctx, "indexing completed", progress=100)
        try:
            await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            return _augment_dict_response(
                tool_name="index_test_project",
                response=coerced,
                meta=_base_meta(),
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("clear_index", "Clear index"))
    async def clear_index(ctx: Optional[Context] = None) -> dict:
        result = await _run(server.clear_index)
        try:
            await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            return _augment_dict_response(
                tool_name="clear_index",
                response=coerced,
                meta=_base_meta(),
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("clear_index_v2", "Clear index (v2)"))
    async def clear_index_v2(project_path: str = None, ctx: Optional[Context] = None) -> dict:
        result = await _run(server.clear_index, project_path)
        try:
            await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            return _augment_dict_response(
                tool_name="clear_index_v2",
                response=coerced,
                meta=_base_meta(project_path_used=project_path),
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("repair_index", "Repair index manifests"))
    async def repair_index(project_path: str = None, ctx: Optional[Context] = None) -> dict:
        result = await _run(server.repair_index, project_path)
        try:
            if ctx is not None:
                await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            return _augment_dict_response(
                tool_name="repair_index",
                response=coerced,
                meta=_base_meta(),
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("get_chunk", "Get chunk by chunk_id"))
    async def get_chunk(
        chunk_id: str,
        include_content: bool = True,
        context_depth: int = 0,
        max_content_chars: int = 8000,
        max_context_items: int = 6,
        project_path: str = None,
        ctx: Optional[Context] = None,
    ) -> dict:
        result = await _run(
            server.get_chunk,
            chunk_id,
            include_content=include_content,
            context_depth=context_depth,
            max_content_chars=max_content_chars,
            max_context_items=max_context_items,
            project_path=project_path,
        )
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            meta = _base_meta(project_path_used=_get_active_project_path())
            meta.update(
                {
                    "chunk_id": chunk_id,
                    "include_content": bool(include_content),
                    "context_depth": int(context_depth),
                    "max_content_chars": int(max_content_chars),
                    "max_context_items": int(max_context_items),
                }
            )
            return _augment_dict_response(
                tool_name="get_chunk",
                response=coerced,
                meta=meta,
                result_value=coerced.get("chunk") if "chunk" in coerced else None,
            )
        return coerced

    @mcp.tool(description=strings.get("tools", {}).get("list_tools", "List all MCP tools and resources"))
    async def list_tools(ctx: Optional[Context] = None) -> dict:
        """Best-effort introspection so agents can discover available tools/resources."""
        tools = []
        resources = []
        try:
            tm = getattr(mcp, "_tool_manager", None)
            internal = getattr(tm, "_tools", None) if tm is not None else None
            if isinstance(internal, dict):
                for name, tool in internal.items():
                    desc = getattr(tool, "description", "") or ""
                    tools.append({"name": name, "description": desc})
        except Exception:
            tools = []

        try:
            rm = getattr(mcp, "_resource_manager", None)
            internal_r = getattr(rm, "_resources", None) if rm is not None else None
            if isinstance(internal_r, dict):
                for name in internal_r.keys():
                    resources.append({"uri": name})
        except Exception:
            resources = []

        payload = {
            "count": len(tools),
            "tools": sorted(tools, key=lambda x: x.get("name", "")),
            "resources": sorted(resources, key=lambda x: x.get("uri", "")),
        }
        return _augment_dict_response(
            tool_name="list_tools",
            response=payload,
            meta=_base_meta(),
            # Avoid circular references: `response` and `result` must not be the same dict.
            result_value=dict(payload),
        )

    @mcp.resource("search://stats")
    def get_search_statistics() -> str:
        """Get search engine statistics."""
        try:
            project_path = server.current_project_path() or os.getcwd()
            project_dir = server.get_project_storage_dir(project_path)
            stats_path = project_dir / "index" / "stats.json"
            if stats_path.exists():
                raw = stats_path.read_text()
                try:
                    stats = json.loads(raw)
                except Exception:
                    stats = {}

                # Strictly additive: preserve existing keys from stats.json.
                stats.setdefault("active_project_path", project_path)
                stats.setdefault("active_project_id", project_dir.name)

                index_dir = project_dir / "index"
                index_present = (index_dir / "code.index").exists()
                if not index_present:
                    shards_root = index_dir / "shards"
                    if shards_root.exists():
                        for shard in shards_root.glob("shard_*"):
                            if (shard / "code.index").exists():
                                index_present = True
                                break
                stats.setdefault("index_present", index_present)
                stats.setdefault("vector_count", stats.get("total_chunks"))

                warnings_list = stats.get("warnings")
                if not isinstance(warnings_list, list):
                    warnings_list = []

                try:
                    total_chunks = int(stats.get("total_chunks", 0) or 0)
                except Exception:
                    total_chunks = 0

                metadata_hint = bool(stats.get("files_indexed") or stats.get("chunk_types") or stats.get("top_tags"))
                if total_chunks == 0 and metadata_hint and not index_present:
                    warnings_list.append("metadata_present_but_index_missing")

                if warnings_list:
                    stats["warnings"] = warnings_list

                return json.dumps(stats)
            return json.dumps({"error": "No stats available"})
        except Exception as e:
            return json.dumps({"error": f"Failed to get statistics: {str(e)}"})

    @mcp.resource("codesearch://projects/list")
    def list_indexed_projects() -> str:
        """List all projects that have been indexed in this environment."""
        try:
            projects = server.list_projects(as_dict=False)
            return json.dumps(projects, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to list projects: {str(e)}"})

    @mcp.resource("codesearch://projects/{project_id}")
    def get_project_details(project_id: str) -> str:
        """Get detailed statistics and metadata for a specific indexed project."""
        try:
            # We need a way to find project path by ID, or just look in storage root
            projects_dir = server.storage_root / "projects"
            p_dir = projects_dir / project_id
            if p_dir.exists() and p_dir.is_dir():
                stats_path = p_dir / "index" / "stats.json"
                if stats_path.exists():
                    return stats_path.read_text()
            return json.dumps({"error": f"Project {project_id} not found"})
        except Exception as e:
            return json.dumps({"error": f"Failed to get project details: {str(e)}"})

    @mcp.prompt()
    def search_help() -> str:
        return strings.get("help", "")
