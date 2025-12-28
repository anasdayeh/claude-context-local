"""Core logic for code search and indexing server."""

import os
import logging
import json
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional

from common_utils import get_storage_dir
from search.indexer import CodeIndexManager
from embeddings.embedder import CodeEmbedder
from chunking.multi_language_chunker import MultiLanguageChunker
from search.searcher import IntelligentSearcher
from mcp_server.index_jobs import IndexJobManager

logger = logging.getLogger(__name__)


class CodeSearchServer:
    """Main server class managing indexing and search operations."""

    def __init__(self):
        self.storage_root = get_storage_dir()
        # Default embedder uses local models/ directory if configured in CodeSearchServer init
        self.embedder = CodeEmbedder(cache_dir=str(self.storage_root / "models"))
        self.chunker = MultiLanguageChunker()
        self._current_project = None
        self._index_manager = None
        self._searcher = None

        self._indexing_lock = threading.Lock()
        self._job_executor = ThreadPoolExecutor(
            max_workers=int(os.getenv("CODE_SEARCH_INDEX_WORKERS", "1") or 1),
            thread_name_prefix="code-search-index",
        )
        self._jobs = IndexJobManager(
            event_buffer_size=int(os.getenv("CODE_SEARCH_JOB_EVENT_BUFFER", "200") or 200)
        )

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
            except Exception:
                pass
        return False

    def switch_project(self, project_path: str) -> Dict[str, Any]:
        """Switch current active project."""
        project_dir = self.get_project_storage_dir(project_path)
        index_dir = project_dir / "index"

        if not self.ensure_project_indexed(project_path):
            return {
                "error": f"Project not indexed: {project_path}",
                "suggestion": f"Run index_directory('{project_path}') first"
            }

        self._index_manager = CodeIndexManager(str(index_dir))
        self._searcher = IntelligentSearcher(self._index_manager, self.embedder)
        self._current_project = project_path
        
        return {"success": True, "project": project_path}

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
        """Index a directory synchronously (implementation shared by tools/jobs)."""
        try:
            from search.incremental_indexer import IncrementalIndexer
            
            project_dir = self.get_project_storage_dir(directory_path)
            index_dir = project_dir / "index"
            
            if not project_name:
                project_name = Path(directory_path).name
                
            index_manager = CodeIndexManager(str(index_dir))
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
        file_pattern: str = None,
        chunk_type: str = None,
        include_context: bool = True,
        auto_reindex: bool = False,
        max_age_minutes: float = 5,
        project_path: str = None, # Added to support both tool styles
        as_dict: bool = True,
    ) -> List[Dict[str, Any]]:
        """Implementation of search_code tool."""
        if project_path:
            switch_res = self.switch_project(project_path)
            if "error" in switch_res:
                # If project not switched, but we have a current one, we might continue
                # but it's safer to return the error
                return [switch_res]
        
        if not self._searcher:
            # Try to auto-switch to the last used project or any project
            if self._current_project:
                self.switch_project(self._current_project)
            else:
                projects = self.list_projects(as_dict=False)
                if isinstance(projects, dict):
                    projects = projects.get("projects", [])
                if projects:
                   self.switch_project(projects[0]["project_path"])
        
        if not self._searcher:
            return {"error": "No project selected. Provide project_path or run index_directory first."}

        try:
            # Respect both tool parameter and env var for backward compatibility/global override
            env_include_context = os.getenv("CODE_SEARCH_INCLUDE_CONTEXT", "").lower() in {"1", "true", "yes"}
            context_depth = 1 if (include_context or env_include_context) else 0
            
            filters = {}
            if file_pattern:
                # Support both single string and list if needed by internal searcher
                filters['file_pattern'] = [file_pattern] if isinstance(file_pattern, str) else file_pattern
            if chunk_type:
                filters['chunk_type'] = chunk_type
                
            results = self._searcher.search(
                query, 
                k=k, 
                filters=filters, 
                context_depth=context_depth,
                search_mode="semantic" if search_mode == "auto" else search_mode
            )
            
            # Map search results to the format expected by the MCP tool
            tool_results = [res.to_search_tool_dict() for res in results]
            if as_dict:
                return {"results": tool_results}
            return tool_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"error": str(e)}

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
        index_manager = CodeIndexManager(str(index_dir))
        index_stats = index_manager.get_stats()
        stats.update(index_stats)
        
        return stats

    def get_index_status(self) -> Dict[str, Any]:
        """Alias for get_stats with additional model info."""
        stats = self.get_stats()
        if "error" in stats:
            return stats

        index_stats = dict(stats)
        if "files_indexed" not in index_stats:
            index_stats["files_indexed"] = self._count_indexed_files(stats.get("project_path"))

        return {
            "index_statistics": index_stats,
            "model_info": self.embedder.get_model_info(),
        }

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
            index_manager = CodeIndexManager(str(index_dir))
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

    def _maybe_start_model_preload(self) -> None:
        """Preload the embedding model in background if requested."""
        try:
            self.embedder.embed_query("warmup")
        except Exception:
            pass

    def clear_index(self, project_path: str = None) -> Dict[str, Any]:
        """Clear the search index for a project."""
        target_path = project_path or self._current_project
        if not target_path:
            return {"error": "No project selected."}
        
        project_dir = self.get_project_storage_dir(target_path)
        index_dir = project_dir / "index"
        
        try:
            index_manager = CodeIndexManager(str(index_dir))
            index_manager.clear_index()
            return {"success": True, "message": f"Index cleared for {target_path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def index_test_project(self) -> Dict[str, Any]:
        """Index a small test dataset for verification."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "sample"
            test_path.mkdir()
            (test_path / "main.py").write_text("def hello():\n    print('hello world')\n")
            (test_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
            
            return self.index_directory(str(test_path), project_name="test_project")
