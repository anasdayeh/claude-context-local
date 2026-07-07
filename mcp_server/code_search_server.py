"""Core logic for code search and indexing server."""

import atexit
import os
import logging
import json
import hashlib
import time
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional

from common_utils import get_storage_dir, apply_adaptive_runtime_defaults
from embeddings.embedder import CodeEmbedder
from chunking.multi_language_chunker import MultiLanguageChunker
from search.searcher import IntelligentSearcher
from mcp_server.index_jobs import IndexJobManager

logger = logging.getLogger(__name__)

_DEFAULT_EMBED_MODEL = "google/embeddinggemma-300m"


def _resolve_embed_model_name() -> str:
    """Active embedding model, selectable via CODE_SEARCH_EMBED_MODEL (default Gemma).

    Switching models is env-only (no code edit) — pair it with a distinct
    CODE_SEARCH_STORAGE root so the 768-dim and 2560-dim vector spaces never mix.
    """
    return os.getenv("CODE_SEARCH_EMBED_MODEL", "").strip() or _DEFAULT_EMBED_MODEL


class CodeSearchServer:
    """Main server class managing indexing and search operations."""

    _DEFAULT_MIN_FREE_DISK_GB = 5.0

    def __init__(self):
        applied_defaults = apply_adaptive_runtime_defaults()
        if applied_defaults:
            logger.info("Applied adaptive runtime defaults: %s", applied_defaults)

        self.import_strategy = os.getenv("CODE_SEARCH_IMPORT_STRATEGY", "embedder_first").strip().lower() or "embedder_first"
        self.runtime_selftest_enabled = os.getenv("CODE_SEARCH_RUNTIME_SELFTEST", "0").lower() not in {"0", "false", "no"}
        self.semantic_fallback_enabled = os.getenv(
            "CODE_SEARCH_SEARCH_DISABLE_SEMANTIC_ON_EMBEDDER_FAILURE",
            "1",
        ).lower() not in {"0", "false", "no"}
        self.storage_root = get_storage_dir()
        # Default embedder uses local models/ directory if configured in CodeSearchServer init
        device = os.getenv("CODE_SEARCH_DEVICE", "auto")
        self.embedder = CodeEmbedder(
            model_name=_resolve_embed_model_name(),
            cache_dir=str(self.storage_root / "models"),
            device=device,
        )
        self.chunker = MultiLanguageChunker()
        self._embedder_status: Dict[str, Any] = self.embedder.health_status()
        self._current_project = None
        self._index_manager = None
        self._searcher = None

        self._indexing_lock = threading.Lock()
        self._job_executor = ThreadPoolExecutor(
            max_workers=int(os.getenv("CODE_SEARCH_INDEX_WORKERS", "2") or 2),
            thread_name_prefix="code-search-index",
        )
        self._jobs = IndexJobManager(
            event_buffer_size=int(os.getenv("CODE_SEARCH_JOB_EVENT_BUFFER", "200") or 200)
        )
        atexit.register(self.shutdown)
        if self.runtime_selftest_enabled:
            self._run_runtime_selftest()

    def _min_free_disk_gb(self) -> float:
        raw = str(os.getenv("CODE_SEARCH_MIN_FREE_DISK_GB", "") or "").strip()
        if raw:
            try:
                value = float(raw)
                if value > 0:
                    return value
            except Exception:
                pass
        return self._DEFAULT_MIN_FREE_DISK_GB

    def _check_indexing_disk_space(self) -> Optional[str]:
        """Refuse indexing when the storage volume is below the safety floor."""
        min_free_gb = self._min_free_disk_gb()
        try:
            usage = shutil.disk_usage(str(self.storage_root))
        except Exception as exc:
            logger.warning("Could not determine disk space for %s: %s", self.storage_root, exc)
            return None

        free_gb = usage.free / (1024 ** 3)
        if free_gb < min_free_gb:
            message = (
                f"Insufficient disk space for indexing: {free_gb:.2f} GB free at "
                f"{self.storage_root} (minimum {min_free_gb:.2f} GB)"
            )
            logger.error(message)
            return message

        return None

    def shutdown(self) -> None:
        """Shut down background resources (thread pool, etc.)."""
        try:
            self._job_executor.shutdown(wait=False)
        except Exception:
            pass

    def get_project_storage_dir(self, directory_path: str) -> Path:
        """Get unique storage directory for a project path."""
        path_hash = hashlib.md5(str(Path(directory_path).resolve()).encode()).hexdigest()
        project_dir = self.storage_root / "projects" / path_hash
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    def ensure_project_indexed(self, directory_path: str, project_name: str = None) -> bool:
        """Check if a project is already indexed and usable."""
        project_dir = self.get_project_storage_dir(directory_path)
        index_dir = project_dir / "index"
        
        # Robust check: verify stats exists and has content
        stats_path = index_dir / "stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                    if stats.get("total_chunks", 0) > 0:
                        # Also check if the FAISS index exists
                        if (index_dir / "code.index").exists():
                             return True
                        # Accept sharded indexes (per-shard code.index files)
                        shards_root = index_dir / "shards"
                        if shards_root.exists():
                            manifest_path = index_dir / "manifest.json"
                            shard_dirs = []
                            if manifest_path.exists():
                                try:
                                    payload = json.loads(manifest_path.read_text())
                                    for shard in payload.get("shards", []) or []:
                                        shard_path = shard.get("path")
                                        if shard_path:
                                            shard_dirs.append(index_dir / shard_path)
                                        elif shard.get("id"):
                                            shard_dirs.append(shards_root / shard["id"])
                                except Exception:
                                    shard_dirs = []
                            if not shard_dirs:
                                shard_dirs = list(shards_root.glob("shard_*"))
                            for shard_dir in shard_dirs:
                                if (Path(shard_dir) / "code.index").exists():
                                    return True
            except Exception:
                pass
        return False

    def switch_project(self, project_path: str) -> Dict[str, Any]:
        """Switch current active project."""
        project_dir = self.get_project_storage_dir(project_path)
        index_dir = project_dir / "index"

        if not self.ensure_project_indexed(project_path):
            repaired = self._maybe_auto_repair_sharded_index(index_dir)
            if not repaired:
                return {
                    "error": f"Project not indexed: {project_path}",
                    "suggestion": f"Run index_directory('{project_path}') first"
                }

        self._index_manager = self._build_index_manager(index_dir)
        self._searcher = IntelligentSearcher(self._index_manager, self.embedder)
        self._current_project = project_path
        
        return {"success": True, "project": project_path}

    def current_project_path(self) -> Optional[str]:
        """Return the currently selected project path.

        This is a small compatibility shim for MCP resources/tools that expect
        a `current_project_path()` method on the server.
        """
        return self._current_project

    def _maybe_auto_repair_sharded_index(self, index_dir: Path) -> bool:
        """Auto-repair empty sharded manifest when shard folders exist."""
        shards_root = index_dir / "shards"
        if not shards_root.exists() or not any(shards_root.glob("shard_*")):
            return False

        manifest_path = index_dir / "manifest.json"
        needs_repair = True
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text())
                shard_count = int(payload.get("shard_count", 0))
                shards = payload.get("shards") or []
                needs_repair = shard_count == 0 or not shards
            except Exception:
                needs_repair = True

        if not needs_repair:
            return False

        try:
            from search.sharded_index_manager import ShardedIndexManager
            manager = ShardedIndexManager(str(index_dir))
            result = manager.repair_manifest_from_shards()
            if result.get("repaired"):
                logger.warning(
                    "Auto-repair: rebuilt shard manifest for %s (shards=%s)",
                    index_dir,
                    ",".join(result.get("shards", [])),
                )
                return True
        except Exception as exc:
            logger.warning("Auto-repair failed for %s: %s", index_dir, exc)

        return False

    def _index_directory_impl(
        self,
        directory_path: str,
        project_name: str = None,
        file_patterns: List[str] = None,
        incremental: bool = True,
        progress_callback=None,
        cancel_event=None,
        auto_switch: bool = True,
    ) -> dict:
        """Index a directory synchronously (implementation shared by tools/jobs).

        Callers (index_directory, start_index_job) must check disk space
        before invoking this method.
        """
        try:
            from search.incremental_indexer import IncrementalIndexer
            
            project_dir = self.get_project_storage_dir(directory_path)
            index_dir = project_dir / "index"
            
            if not project_name:
                project_name = Path(directory_path).name
                
            index_manager = self._build_index_manager(index_dir)
            # Use MultiLanguageChunker with root path for relative path calculation
            chunker = MultiLanguageChunker(root_path=directory_path)
            indexer = IncrementalIndexer(index_manager, self.embedder, chunker, str(project_dir))
            
            logger.info(f"Indexing {directory_path} (name={project_name}, patterns={file_patterns})")
            
            result = indexer.incremental_index(
                directory_path,
                project_name,
                file_patterns=file_patterns,
                force_full=not incremental,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            
            # Auto-switch to newly indexed project (synchronous tool UX)
            if auto_switch:
                self.switch_project(directory_path)
            
            response = {
                "success": result.success,
                "files_added": result.files_added,
                "files_modified": result.files_modified,
                "files_removed": result.files_removed,
                "chunks_added": result.chunks_added,
                "chunks_removed": result.chunks_removed,
                "time_taken": round(result.time_taken, 2),
                "project_id": project_dir.name,
            }
            if result.error:
                response["error"] = result.error
            return response
        except Exception as e:
            logger.error(f"Index failed: {e}")
            # Best-effort mark status as failed for observability in list_projects
            try:
                project_dir = self.get_project_storage_dir(directory_path)
                index_dir = project_dir / "index"
                from search.indexer import CodeIndexManager
                CodeIndexManager(str(index_dir)).save_index(extra_metadata={
                    "project_name": project_name or Path(directory_path).name,
                    "project_path": directory_path,
                    "status": "failed",
                    "last_indexed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "error": str(e),
                })
            except Exception:
                pass
            return {"success": False, "error": str(e)}

    def index_directory(
        self,
        directory_path: str,
        project_name: str = None,
        file_patterns: List[str] = None,
        incremental: bool = True,
        progress_callback=None,
    ) -> dict:
        """Implementation of index_directory tool (synchronous)."""
        disk_error = self._check_indexing_disk_space()
        if disk_error:
            return {"success": False, "error": disk_error}

        with self._indexing_lock:
            return self._index_directory_impl(
                directory_path=directory_path,
                project_name=project_name,
                file_patterns=file_patterns,
                incremental=incremental,
                progress_callback=progress_callback,
                cancel_event=None,
                auto_switch=True,
            )

    def start_index_job(
        self,
        directory_path: str,
        project_name: str = None,
        file_patterns: Optional[List[str]] = None,
        incremental: bool = True,
    ) -> Dict[str, Any]:
        """Start indexing in the background and return a job_id for polling."""
        resolved_path = str(Path(directory_path).resolve())
        name = project_name or Path(resolved_path).name

        existing = self._jobs.find_active_job_for_path(resolved_path)
        if existing is not None:
            return {"success": True, "job": existing.to_dict(), "deduped": True}

        disk_error = self._check_indexing_disk_space()
        if disk_error:
            return {"success": False, "error": disk_error}

        job = self._jobs.create_job(
            project_path=resolved_path,
            project_name=name,
            file_patterns=file_patterns,
            incremental=incremental,
        )
        job.add_event("queued")

        def _run() -> None:
            with self._indexing_lock:
                job.status = "running"
                job.started_at = time.time()
                job.add_event("indexing started")
                try:
                    result = self._index_directory_impl(
                        directory_path=resolved_path,
                        project_name=name,
                        file_patterns=file_patterns,
                        incremental=incremental,
                        progress_callback=job.add_event,
                        cancel_event=job.cancel_event,
                        auto_switch=False,
                    )

                    job.result = result
                    # Prefer explicit cancel_event; fall back to error string
                    if job.cancel_event.is_set() or str(result.get("error", "")).lower().startswith("indexing canceled"):
                        job.status = "canceled"
                        job.add_event("indexing canceled")
                    elif result.get("success"):
                        job.status = "completed"
                        job.add_event("indexing completed")
                    else:
                        job.status = "failed"
                        job.error = result.get("error") or "indexing failed"
                        job.add_event(f"indexing failed: {job.error}")
                except Exception as exc:
                    job.status = "failed"
                    job.error = str(exc)
                    job.add_event(f"indexing failed: {job.error}")
                finally:
                    job.finished_at = time.time()

        self._job_executor.submit(_run)
        return {"success": True, "job": job.to_dict()}

    def get_index_job_status(
        self, job_id: Optional[str] = None, project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get status for a background index job."""
        if job_id:
            job = self._jobs.get_job(job_id)
            if job is None:
                return {"success": False, "error": f"Unknown job_id: {job_id}"}
            return {"success": True, "job": job.to_dict()}

        if project_path:
            resolved = str(Path(project_path).resolve())
            job = self._jobs.find_active_job_for_path(resolved)
            if job is None:
                # Also check for latest completed job for that path
                for j in self._jobs.list_jobs():
                    if j.get("project_path") == resolved:
                        return {"success": True, "job": j}
                return {"success": False, "error": f"No job found for project_path: {resolved}"}
            return {"success": True, "job": job.to_dict()}

        return {"success": True, "jobs": self._jobs.list_jobs()}

    def cancel_index_job(self, job_id: str) -> Dict[str, Any]:
        """Request cooperative cancellation of an indexing job."""
        return self._jobs.cancel(job_id)

    def search_code(
        self,
        query: str,
        k: int = 5,
        search_mode: str = "auto",
        file_patterns: Optional[List[str]] = None,
        # Back-compat: older callers used a singular pattern string.
        file_pattern: str = None,
        chunk_type: str = None,
        include_context: bool = True,
        auto_reindex: bool = False,
        max_age_minutes: float = 5,
        project_path: str = None,
    ) -> Dict[str, Any]:
        """Implementation of search_code tool.

        Returns a dict with at least ``results`` (list) on success, or
        ``error`` (str) on failure.
        """
        def _error(msg: str, **extra: Any) -> Dict[str, Any]:
            return {"error": msg, "results": [], **extra}

        # Determine which project to use, with clear priority:
        # 1. Explicit project_path parameter
        # 2. Currently cached project (self._current_project)
        # 3. Auto-discover first available indexed project
        target_project = project_path or self._current_project

        # If we have a target project but searcher is not loaded (or stale),
        # explicitly switch to ensure state is properly initialized.
        if target_project and not self._searcher:
            switch_res = self.switch_project(target_project)
            if "error" in switch_res:
                if project_path:
                    return _error(switch_res["error"])
                self._current_project = None
                target_project = None
        elif project_path and project_path != self._current_project:
            switch_res = self.switch_project(project_path)
            if "error" in switch_res:
                return _error(switch_res["error"])

        # If still no searcher, try auto-discovery of any indexed project
        if not self._searcher:
            projects = self.list_projects(as_dict=False)
            if isinstance(projects, dict):
                projects = projects.get("projects", [])
            if projects:
                for proj in projects:
                    proj_path = proj.get("project_path")
                    if proj_path and proj_path != "unknown":
                        switch_res = self.switch_project(proj_path)
                        if "error" not in switch_res and self._searcher:
                            break

        if not self._searcher:
            return _error("No project selected. Provide project_path or run index_directory first.")

        try:
            embedder_status = self.get_embedder_status()
            env_include_context = os.getenv("CODE_SEARCH_INCLUDE_CONTEXT", "").lower() in {"1", "true", "yes"}
            context_depth = 1 if (include_context or env_include_context) else 0

            filters = {}
            if file_patterns is None and file_pattern:
                file_patterns = [file_pattern]
            if isinstance(file_patterns, str):
                file_patterns = [file_patterns]
            if file_patterns:
                filters["file_pattern"] = list(file_patterns)
            if chunk_type:
                filters['chunk_type'] = chunk_type

            if (
                embedder_status.get("status") == "failed"
                and search_mode == "semantic"
            ):
                return self._semantic_unavailable_response(fallback_mode="none")

            results = self._searcher.search(
                query,
                k=k,
                filters=filters,
                context_depth=context_depth,
                search_mode=search_mode
            )

            tool_results = [res.to_search_tool_dict() for res in results]
            response: Dict[str, Any] = {"results": tool_results}
            if isinstance(self._searcher, IntelligentSearcher):
                mode_used = getattr(self._searcher, "last_search_mode_used", None)
                if mode_used == "fts" and search_mode != "fts":
                    response.update(
                        {
                            "semantic_available": False,
                            "fallback_mode": "fts",
                            "error_code": "embedder_init_failed",
                            "error": self.get_embedder_status().get("error"),
                        }
                    )
            return response
        except Exception as e:
            self._refresh_embedder_status()
            logger.error(f"Search failed: {e}")
            if search_mode == "semantic" or not self.semantic_fallback_enabled:
                return self._semantic_unavailable_response(
                    error=str(e),
                    fallback_mode="none",
                )
            return _error(str(e))

    def find_similar_code(self, chunk_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find chunks functionally similar to a given chunk."""
        if not self._searcher:
            return [{"error": "No project selected."}]
        
        try:
            results = self._searcher.find_similar_to_chunk(chunk_id, k=k)
            return [res.to_similar_tool_dict() for res in results]
        except Exception as e:
            logger.error(f"Similar search failed: {e}")
            return [{"error": str(e)}]

    def get_stats(self, project_path: str = None) -> Dict[str, Any]:
        """Get indexing statistics for a project."""
        target_path = project_path or self._current_project
        if not target_path:
            return {"error": "No project selected."}
            
        project_dir = self.get_project_storage_dir(target_path)
        index_dir = project_dir / "index"
        
        # Basic project info
        stats = {
            "project_name": Path(target_path).name,
            "project_path": target_path,
            "project_id": project_dir.name,
            "storage_path": str(project_dir)
        }
        
        # Load detailed stats if available
        index_manager = self._build_index_manager(index_dir)
        index_stats = index_manager.get_stats()
        stats.update(index_stats)

        if "sanity_warning" in index_stats:
            logger.warning("Sanity check: %s", index_stats.get("sanity_warning"))
        
        return stats

    def get_index_status(self, project_path: str = None) -> Dict[str, Any]:
        """Alias for get_stats with additional model info."""
        stats = self.get_stats(project_path=project_path)
        if "error" in stats:
            return stats

        index_stats = dict(stats)
        if "files_indexed" not in index_stats:
            index_stats["files_indexed"] = self._count_indexed_files(stats.get("project_path"))

        embedder_status = self.get_embedder_status()

        return {
            "index_statistics": index_stats,
            "model_info": self.embedder.get_model_info(),
            "embedder_status": embedder_status.get("status"),
            "embedder_backend": embedder_status.get("backend"),
            "embedder_failure_summary": embedder_status.get("error"),
        }

    def repair_index(self, project_path: str = None) -> Dict[str, Any]:
        """Repair sharded index manifests if they are missing/empty."""
        target_path = project_path or self._current_project
        if not target_path:
            return {"error": "No project selected."}

        project_dir = self.get_project_storage_dir(target_path)
        index_dir = project_dir / "index"

        shards_root = index_dir / "shards"
        has_shards = shards_root.exists() and any(shards_root.glob("shard_*"))
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists() and not has_shards:
            return {"repaired": False, "reason": "no_manifest_or_shards"}

        if has_shards:
            from search.sharded_index_manager import ShardedIndexManager
            manager = ShardedIndexManager(str(index_dir))
            return manager.repair_manifest_from_shards()

        return {"repaired": False, "reason": "not_sharded"}

    def list_projects(self, as_dict: bool = True) -> Any:
        """List all indexed projects."""
        projects_dir = self.storage_root / "projects"
        if not projects_dir.exists():
            return {"count": 0, "projects": []} if as_dict else []
            
        projects = []
        for p_dir in projects_dir.iterdir():
            if not p_dir.is_dir():
                continue
            
            index_dir = p_dir / "index"
            stats_path = index_dir / "stats.json"
            if stats_path.exists():
                try:
                    with open(stats_path, "r") as f:
                        s = json.load(f)
                        projects.append({
                            "project_id": p_dir.name,
                            "project_hash": p_dir.name,
                            "project_name": s.get("project_name", p_dir.name),
                            "project_path": s.get("project_path", "unknown"),
                            "total_chunks": s.get("total_chunks", 0),
                            "last_indexed": s.get("last_indexed", "unknown"),
                            "status": s.get("status", "ready")
                        })
                except Exception:
                    # Minimum fallback if stats.json is corrupted
                    projects.append({
                        "project_id": p_dir.name,
                        "project_hash": p_dir.name,
                        "project_name": p_dir.name,
                        "project_path": "unknown",
                        "total_chunks": 0,
                        "status": "partial"
                    })
        projects = sorted(projects, key=lambda x: x.get("last_indexed", ""), reverse=True)
        if as_dict:
            return {"count": len(projects), "projects": projects}
        return projects

    def _count_indexed_files(self, project_path: Optional[str]) -> int:
        if not project_path:
            return 0
        project_dir = self.get_project_storage_dir(project_path)
        index_dir = project_dir / "index"
        try:
            index_manager = self._build_index_manager(index_dir)
            iterator = getattr(index_manager, "iter_all_chunks", None)
            if callable(iterator):
                file_paths = {meta.get("relative_path") or meta.get("file_path") for _, meta in iterator() if isinstance(meta, dict)}
                return len({p for p in file_paths if p})
            file_paths = set()
            for entry in index_manager.metadata_db.values():
                meta = entry.get("metadata") if isinstance(entry, dict) else None
                if not isinstance(meta, dict):
                    continue
                path = meta.get("relative_path") or meta.get("file_path")
                if path:
                    file_paths.add(path)
            return len(file_paths)
        except Exception:
            return 0

    def _build_index_manager(self, index_dir: Path):
        """Return a sharded manager when enabled or when manifest exists."""
        manifest_path = index_dir / "manifest.json"
        flag = os.getenv("CODE_SEARCH_SHARDED_INDEX", "").lower()
        if self.import_strategy == "embedder_first":
            self._refresh_embedder_status()
        from search.indexer import CodeIndexManager
        if flag in {"1", "true", "yes"} or (flag not in {"0", "false", "no"} and manifest_path.exists()):
            from search.sharded_index_manager import ShardedIndexManager
            return ShardedIndexManager(str(index_dir))
        return CodeIndexManager(str(index_dir))

    def _maybe_start_model_preload(self) -> None:
        """Preload the embedding model in background if requested."""
        try:
            self._warmup_embedder("warmup")
        except Exception:
            logger.warning("Background embedder preload failed", exc_info=True)

    def _run_runtime_selftest(self) -> None:
        self._warmup_embedder("healthcheck")

    def _warmup_embedder(self, probe: str) -> bool:
        ok = self.embedder.warmup(probe)
        self._refresh_embedder_status()
        if ok:
            logger.info("Embedder warmup succeeded (probe=%s)", probe)
        else:
            logger.warning("Embedder warmup failed (probe=%s): %s", probe, self._embedder_status.get("error"))
        return ok

    def _refresh_embedder_status(self) -> Dict[str, Any]:
        try:
            self._embedder_status = self.embedder.health_status()
        except Exception as exc:
            self._embedder_status = {
                "status": "failed",
                "backend": None,
                "device": None,
                "error": str(exc),
                "model_name": getattr(self.embedder, "model_name", None),
            }
        return dict(self._embedder_status)

    def get_embedder_status(self) -> Dict[str, Any]:
        return self._refresh_embedder_status()

    def _semantic_unavailable_response(
        self,
        *,
        error: Optional[str] = None,
        fallback_mode: str = "none",
    ) -> Dict[str, Any]:
        embedder_status = self.get_embedder_status()
        return {
            "error": error or embedder_status.get("error") or "Semantic search is unavailable.",
            "error_code": "embedder_init_failed",
            "semantic_available": False,
            "fallback_mode": fallback_mode,
            "results": [],
        }

    def clear_index(self, project_path: str = None) -> Dict[str, Any]:
        """Clear the search index for a project."""
        target_path = project_path or self._current_project
        if not target_path:
            return {"error": "No project selected."}
        
        project_dir = self.get_project_storage_dir(target_path)
        index_dir = project_dir / "index"
        
        try:
            is_current = target_path == self._current_project

            # If we are clearing the active project, clear the currently loaded
            # manager to ensure in-memory search state can't return stale results.
            if is_current and self._index_manager is not None:
                self._index_manager.clear_index()
            else:
                index_manager = self._build_index_manager(index_dir)
                index_manager.clear_index()

            if is_current:
                self._searcher = None
                self._index_manager = None
                self._current_project = None
            return {"success": True, "message": f"Index cleared for {target_path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_chunk(
        self,
        chunk_id: str,
        *,
        include_content: bool = True,
        context_depth: int = 0,
        max_content_chars: int = 8000,
        max_context_items: int = 6,
        project_path: str = None,
    ) -> Dict[str, Any]:
        """Get full chunk metadata/content by chunk_id.

        Backwards compatible additive tool: does not change existing tool outputs.
        """
        if project_path:
            switch_res = self.switch_project(project_path)
            if "error" in switch_res:
                return switch_res

        if not self._index_manager or not self._current_project:
            return {"error": "No project selected. Provide project_path or run index_directory first."}

        meta = self._index_manager.get_chunk_by_id(chunk_id)  # type: ignore[union-attr]
        if not meta:
            return {
                "error": f"Chunk not found: {chunk_id}",
                "suggestion": "Verify chunk_id and project selection; use search_code to discover chunk_ids.",
            }

        chunk_meta = dict(meta)
        content = chunk_meta.get("content")
        if not include_content and "content" in chunk_meta:
            chunk_meta.pop("content", None)
        elif include_content and isinstance(content, str) and max_content_chars > 0:
            if len(content) > max_content_chars:
                chunk_meta["content"] = content[: max(0, max_content_chars)] + "..."

        # Minimal consistent identifiers
        chunk_meta.setdefault("chunk_id", chunk_id)
        chunk_meta.setdefault("relative_path", chunk_meta.get("relative_path") or chunk_meta.get("file_path"))
        chunk_meta.setdefault("chunk_type", chunk_meta.get("chunk_type") or chunk_meta.get("kind"))

        context = []
        if context_depth and self._searcher:
            try:
                rel = chunk_meta.get("relative_path") or ""
                neighbors = self._searcher._get_file_neighbors(chunk_id, rel, window=max(0, int(context_depth)))  # type: ignore[attr-defined]
                if isinstance(neighbors, list):
                    context = neighbors[: max(0, int(max_context_items))]
            except Exception:
                context = []

        return {
            "chunk_id": chunk_id,
            "chunk": chunk_meta,
            "context": context,
        }

    def index_test_project(self) -> Dict[str, Any]:
        """Index a small test dataset for verification."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "sample"
            test_path.mkdir()
            (test_path / "main.py").write_text("def hello():\n    print('hello world')\n")
            (test_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
            
            return self.index_directory(str(test_path), project_name="test_project")
