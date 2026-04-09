"""Shared MCP tool/resource registration."""

import json
import asyncio
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastmcp import Context, FastMCP
from merkle.merkle_dag import MerkleDAG

from mcp_server.code_search_server import CodeSearchServer

logger = logging.getLogger(__name__)


def _load_json_file(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _coverage_pct(total: int | None, fts_rows: int | None) -> float | None:
    try:
        total = int(total) if total is not None else 0
        fts_rows = int(fts_rows) if fts_rows is not None else 0
    except Exception:
        return None
    if total > 0:
        # Cap at 100% to prevent inflated metrics from cross-shard duplication
        coverage = min(round((fts_rows / total) * 100, 2), 100.0)
        return coverage
    if fts_rows > 0:
        return 100.0
    return 0.0


def _read_stats_json(server: CodeSearchServer, project_path: str) -> dict:
    try:
        project_dir = server.get_project_storage_dir(project_path)
        stats_path = project_dir / "index" / "stats.json"
        if not stats_path.exists():
            return {}
        return json.loads(stats_path.read_text())
    except Exception:
        return {}


def _read_manifest_json(server: CodeSearchServer, project_path: str) -> dict:
    try:
        project_dir = server.get_project_storage_dir(project_path)
        manifest_path = project_dir / "index" / "manifest.json"
        if not manifest_path.exists():
            return {}
        return json.loads(manifest_path.read_text())
    except Exception:
        return {}


def _infer_last_indexed(server: CodeSearchServer, project_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not project_path:
        return None, None

    stats = _read_stats_json(server, project_path)
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


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        s = str(value).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _shard_summaries(project_dir: Path) -> list[dict]:
    shards_root = project_dir / "index" / "shards"
    summaries: list[dict] = []
    if not shards_root.exists():
        return summaries
    for shard_dir in sorted(shards_root.glob("shard_*")):
        stats = _load_json_file(shard_dir / "stats.json")
        total_chunks = stats.get("total_chunks")
        fts_rows = stats.get("fts_rows")
        coverage = _coverage_pct(total_chunks, fts_rows)
        shard_warnings: list[str] = []
        if total_chunks and fts_rows == 0:
            shard_warnings.append("FTS rows missing in shard.")
        if coverage is not None and total_chunks and coverage < 40:
            shard_warnings.append("Shard FTS coverage is low.")
        summaries.append(
            {
                "shard_id": shard_dir.name,
                "shard_path": str(shard_dir),
                "code_index_bytes": (shard_dir / "code.index").stat().st_size
                if (shard_dir / "code.index").exists()
                else None,
                "metadata_db_bytes": (shard_dir / "metadata.db").stat().st_size
                if (shard_dir / "metadata.db").exists()
                else None,
                "fts_rows": fts_rows,
                "total_chunks": total_chunks,
                "coverage_pct": coverage,
                "stats": stats,
                "warnings": shard_warnings,
            }
        )
    return summaries


def _build_fts_status_payload(server: CodeSearchServer, project_path: Optional[str]) -> dict:
    payload: dict[str, Any] = {
        "project_path": project_path,
        "project_id": None,
        "manifest_path": None,
        "manifest": {},
        "manifest_index_bytes": 0,
        "stats_path": None,
        "stats": {},
        "total_chunks": None,
        "fts_rows": None,
        "coverage_pct": None,
        "warnings": [],
        "last_indexed": None,
        "last_indexed_source": None,
        "shards": [],
    }
    if not project_path:
        payload["warnings"].append("No project selected.")
        return payload
    try:
        project_dir = server.get_project_storage_dir(project_path)
    except Exception as exc:
        payload["warnings"].append(f"Failed to resolve project: {exc}")
        payload["error"] = str(exc)
        return payload

    manifest_path = project_dir / "index" / "manifest.json"
    stats_path = project_dir / "index" / "stats.json"
    manifest = _read_manifest_json(server, project_path) or {}
    stats = _read_stats_json(server, project_path) or {}
    total_chunks = stats.get("total_chunks")
    fts_rows = stats.get("fts_rows")
    coverage = _coverage_pct(total_chunks, fts_rows)
    warnings: list[str] = []
    if not stats:
        warnings.append("Stats file missing or unreadable.")
    if stats.get("sanity_warning"):
        warnings.append(stats["sanity_warning"])
    if total_chunks and fts_rows == 0:
        warnings.append("FTS rows missing despite indexed chunks.")
    if coverage is not None and total_chunks and coverage < 40:
        warnings.append("FTS coverage is below 40%.")

    manifest_shards = manifest.get("shards") or []
    manifest_index_bytes = sum(int(shard.get("index_bytes", 0)) for shard in manifest_shards)

    payload.update(
        {
            "project_id": project_dir.name,
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "manifest_index_bytes": manifest_index_bytes,
            "stats_path": str(stats_path),
            "stats": stats,
            "total_chunks": total_chunks,
            "fts_rows": fts_rows,
            "coverage_pct": coverage,
            "warnings": warnings,
            "last_indexed": stats.get("last_indexed"),
            "last_indexed_source": stats.get("last_indexed") and "stats_json",
            "shards": _shard_summaries(project_dir),
        }
    )
    if not payload["last_indexed"]:
        inferred, source = _infer_last_indexed(server, project_path)
        payload["last_indexed"] = inferred
        payload["last_indexed_source"] = source

    return payload


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

    reporter = getattr(ctx, "report_progress", None)
    if reporter is not None:
        try:
            await reporter(progress=progress, total=total, message=message)
            return
        except Exception as exc:
            logger.debug("Context.report_progress failed: %s", exc)

    session = getattr(ctx, "session", None)
    sender = getattr(session, "send_progress_notification", None) if session is not None else None
    token = _extract_progress_token(ctx)
    if sender is None or token is None:
        return

    req_id = _extract_request_id(ctx)

    try:
        import inspect

        sig = inspect.signature(sender)
        if "related_request_id" in sig.parameters:
            await sender(
                progress_token=token,
                progress=progress,
                total=total,
                message=message,
                related_request_id=req_id,
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

    async def _maybe_collect_registered_items(method_name: str) -> list:
        method = getattr(mcp, method_name, None)
        if method is None:
            return []
        try:
            value = method()
            if asyncio.iscoroutine(value):
                value = await value
            return list(value or [])
        except Exception:
            return []

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

        # For error responses, ensure flat structure with error and suggestion at top level.
        # No nested error_info - keep it simple for MCP protocol and agent parsing.
        if out.get("ok") is False:
            # Ensure error field exists with a meaningful message
            if "error" not in out:
                out["error"] = "Unknown error"
            # suggestion field stays at top level if present (no changes needed)

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
        file_patterns: list[str] = None,
        chunk_type: str = None,
        include_context: bool = True,
        auto_reindex: bool = False,
        max_age_minutes: float = 5,
        project_path: str = None,
        ctx: Optional[Context] = None,
    ) -> dict:
        before_project = _get_active_project_path()
        # Be lenient: some clients may send a single string despite the list schema.
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]
        result = await _run(
            server.search_code,
            query,
            k,
            search_mode,
            file_patterns,
            None,
            chunk_type,
            include_context,
            auto_reindex,
            max_age_minutes,
            project_path,
            True,
        )
        coerced = _coerce_result(result)
        if not isinstance(coerced, dict):
            return coerced

        # Determine whether the server auto-switched projects
        after_project = _get_active_project_path()
        did_auto_switch = bool(after_project and before_project != after_project and not project_path)

        # Normalize filters for meta
        fp = None
        if file_patterns:
            fp = list(file_patterns) if isinstance(file_patterns, list) else [str(file_patterns)]
        filters_applied = {"file_patterns": fp, "chunk_type": chunk_type}

        # Determine effective context behavior
        env_include_context = os.getenv("CODE_SEARCH_INCLUDE_CONTEXT", "").lower() in {"1", "true", "yes"}
        include_context_effective = bool(include_context or env_include_context)

        stats = _read_stats_json(server, after_project) if after_project else {}
        index_last_indexed = stats.get("last_indexed")
        index_last_indexed_source = "stats_json" if index_last_indexed else None
        if not index_last_indexed:
            index_last_indexed, index_last_indexed_source = _infer_last_indexed(server, after_project)
        index_id = None
        if after_project:
            try:
                index_id = server.get_project_storage_dir(after_project).name
            except Exception:
                index_id = None
        manifest = _read_manifest_json(server, after_project) if after_project else {}
        manifest_version = manifest.get("version")

        results_list = coerced.get("results") or []

        # Observability: report the actual mode used by the underlying searcher.
        mode_used = None
        embedder_status = {}
        try:
            searcher = getattr(server, "_searcher", None)
            mode_used = getattr(searcher, "last_search_mode_used", None) if searcher is not None else None
            embedder_status = server.get_embedder_status()
        except Exception:
            mode_used = None
            embedder_status = {}
        if not mode_used:
            mode_used = "semantic" if search_mode == "auto" else search_mode

        # Optional background reindexing when index is stale.
        reindex_meta: dict[str, Any] = {}
        if auto_reindex and after_project:
            try:
                dt = _parse_iso_datetime(index_last_indexed)
                now = datetime.now(tz=dt.tzinfo) if dt and dt.tzinfo else datetime.now()
                age_minutes = (now - dt).total_seconds() / 60 if dt else None
                threshold = float(max_age_minutes or 0)
                should_reindex = dt is None or (age_minutes is not None and age_minutes > threshold)

                # Only attempt to reindex real paths.
                if should_reindex and Path(after_project).exists():
                    job_result = await _run(server.start_index_job, after_project, None, None, True)
                    job_result = _coerce_result(job_result)
                    job = job_result.get("job") if isinstance(job_result, dict) else None
                    reindex_meta.update(
                        {
                            "reindex_started": bool(isinstance(job_result, dict) and job_result.get("success")),
                            "reindex_deduped": bool(isinstance(job_result, dict) and job_result.get("deduped")),
                            "reindex_job_id": job.get("job_id") if isinstance(job, dict) else None,
                            "index_age_minutes": round(age_minutes, 2) if isinstance(age_minutes, (int, float)) else None,
                            "reindex_threshold_minutes": threshold,
                        }
                    )
                else:
                    reindex_meta.update(
                        {
                            "reindex_started": False,
                            "index_age_minutes": round(age_minutes, 2) if isinstance(age_minutes, (int, float)) else None,
                            "reindex_threshold_minutes": threshold,
                            "reindex_skipped_reason": "fresh" if dt and not should_reindex else "nonexistent_project_path",
                        }
                    )
            except Exception as exc:
                reindex_meta.update({"reindex_started": False, "reindex_error": str(exc)})

        meta = _base_meta(did_auto_switch=did_auto_switch, project_path_used=after_project)
        meta.update(
            {
                "query": query,
                "k_requested": int(k),
                "k_returned": len(results_list) if isinstance(results_list, list) else None,
                "search_mode_requested": search_mode,
                "search_mode_used": mode_used,
                "filters_applied": filters_applied,
                "project_path_used": after_project,
                "index_id": index_id,
                "index_last_indexed": index_last_indexed,
                "index_last_indexed_source": index_last_indexed_source,
                "manifest_version": manifest_version,
                "include_context_requested": bool(include_context),
                "include_context_effective": include_context_effective,
                "context_depth": 1 if include_context_effective else 0,
                "stats_storage_size": stats.get("storage_size"),
                "embedder_status": embedder_status.get("status"),
                "embedder_backend": embedder_status.get("backend"),
                "embedder_failure_summary": embedder_status.get("error"),
            }
        )
        meta.update(reindex_meta)
        fts_info = _build_fts_status_payload(server, after_project)
        meta.setdefault("fts_coverage_pct", fts_info.get("coverage_pct"))
        meta.setdefault("fts_rows", fts_info.get("fts_rows"))
        meta.setdefault("total_chunks", fts_info.get("total_chunks"))
        manifest_info = fts_info.get("manifest") if isinstance(fts_info, dict) else {}
        if not isinstance(manifest_info, dict):
            manifest_info = {}
        meta.setdefault("manifest_project_path", manifest_info.get("project_path"))
        meta.setdefault("manifest_path", fts_info.get("manifest_path"))
        meta.setdefault("manifest_index_bytes", fts_info.get("manifest_index_bytes"))
        meta.setdefault("stats_path", fts_info.get("stats_path"))
        meta.setdefault("last_indexed", fts_info.get("last_indexed"))
        meta.setdefault("last_indexed_source", fts_info.get("last_indexed_source"))

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
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]
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
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]
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
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            # Unexpected shape; just augment and return.
            return _augment_dict_response(tool_name="find_similar_code", response=coerced, meta=_base_meta())
        items = coerced if isinstance(coerced, list) else []
        return _augment_dict_response(
            tool_name="find_similar_code",
            response={"results": items},
            meta=_base_meta(),
            result_value=items,
        )

    @mcp.tool(description=strings.get("tools", {}).get("get_index_status", "Get index status"))
    async def get_index_status(project_path: str = None, ctx: Optional[Context] = None) -> dict:
        result = await _run(server.get_index_status, project_path)
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            try:
                embedder_status = server.get_embedder_status()
            except Exception:
                embedder_status = {}
            return _augment_dict_response(
                tool_name="get_index_status",
                response=coerced,
                meta={
                    **_base_meta(project_path_used=project_path),
                    "embedder_status": embedder_status.get("status"),
                    "embedder_backend": embedder_status.get("backend"),
                    "embedder_failure_summary": embedder_status.get("error"),
                },
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

    @mcp.tool(description=strings.get("tools", {}).get("fts_status", "Show manifest, stats, and shard-level FTS coverage"))
    async def fts_status(
        project_path: str | None = None,
        ctx: Optional[Context] = None,
    ) -> dict:
        target_path = project_path or _get_active_project_path()
        payload = _build_fts_status_payload(server, target_path)
        return _augment_dict_response(
            tool_name="fts_status",
            response=payload,
            meta=_base_meta(project_path_used=target_path),
            result_value=payload,
        )

    @mcp.tool(description=strings.get("tools", {}).get("list_projects", "List projects"))
    async def list_projects(ctx: Optional[Context] = None) -> dict:
        result = await _run(server.list_projects, True)
        coerced = _coerce_result(result)
        if isinstance(coerced, dict) and "projects" in coerced:
            return _augment_dict_response(
                tool_name="list_projects",
                response=coerced,
                meta=_base_meta(),
                result_value=coerced.get("projects"),
            )
        # Fallback: normalize unexpected shape
        projects = coerced if isinstance(coerced, list) else []
        return _augment_dict_response(
            tool_name="list_projects",
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
    async def clear_index(project_path: str = None, ctx: Optional[Context] = None) -> dict:
        result = await _run(server.clear_index, project_path)
        try:
            await ctx.session.send_resource_updated("search://stats")
        except Exception:
            pass
        coerced = _coerce_result(result)
        if isinstance(coerced, dict):
            return _augment_dict_response(
                tool_name="clear_index",
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
        resource_templates = []
        prompts = []
        public_tools = await _maybe_collect_registered_items("list_tools")
        if public_tools:
            for tool in public_tools:
                tools.append(
                    {
                        "name": getattr(tool, "name", None),
                        "description": getattr(tool, "description", "") or "",
                    }
                )
        else:
            try:
                tm = getattr(mcp, "_tool_manager", None)
                internal = getattr(tm, "_tools", None) if tm is not None else None
                if isinstance(internal, dict):
                    for name, tool in internal.items():
                        desc = getattr(tool, "description", "") or ""
                        tools.append({"name": name, "description": desc})
            except Exception:
                tools = []

        public_resources = await _maybe_collect_registered_items("list_resources")
        if public_resources:
            for resource in public_resources:
                uri = getattr(resource, "uri", None)
                if uri:
                    resources.append({"uri": str(uri)})
        else:
            try:
                rm = getattr(mcp, "_resource_manager", None)
                internal_r = getattr(rm, "_resources", None) if rm is not None else None
                if isinstance(internal_r, dict):
                    for name in internal_r.keys():
                        if "{" in name and "}" in name:
                            resource_templates.append({"uri": name})
                        else:
                            resources.append({"uri": name})
            except Exception:
                resources = []
                resource_templates = []

        public_templates = await _maybe_collect_registered_items("list_resource_templates")
        if public_templates:
            for resource_template in public_templates:
                uri = getattr(resource_template, "uri_template", None) or getattr(resource_template, "uri", None)
                if uri:
                    resource_templates.append({"uri": str(uri)})

        public_prompts = await _maybe_collect_registered_items("list_prompts")
        if public_prompts:
            for prompt in public_prompts:
                prompts.append(
                    {
                        "name": getattr(prompt, "name", None),
                        "description": getattr(prompt, "description", "") or "",
                    }
                )
        else:
            try:
                pm = getattr(mcp, "_prompt_manager", None)
                internal_p = getattr(pm, "_prompts", None) if pm is not None else None
                if isinstance(internal_p, dict):
                    for name, prompt in internal_p.items():
                        desc = getattr(prompt, "description", "") or ""
                        prompts.append({"name": name, "description": desc})
            except Exception:
                prompts = []

        payload = {
            "count": len(tools),
            "tools": sorted(tools, key=lambda x: x.get("name", "")),
            "resources": sorted(resources, key=lambda x: x.get("uri", "")),
            "resource_templates": sorted(resource_templates, key=lambda x: x.get("uri", "")),
            "prompts": sorted(prompts, key=lambda x: x.get("name", "")),
        }
        try:
            embedder_status = server.get_embedder_status()
        except Exception:
            embedder_status = {}
        payload["embedder_status"] = embedder_status.get("status")
        payload["embedder_backend"] = embedder_status.get("backend")
        payload["embedder_failure_summary"] = embedder_status.get("error")
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
