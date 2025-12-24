"""Core logic for code search and indexing server."""

import os
import logging
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from common_utils import get_storage_dir
from search.indexer import CodeIndexManager
from embeddings.embedder import CodeEmbedder
from chunking.multi_language_chunker import MultiLanguageChunker
from search.searcher import IntelligentSearcher

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

    def index_directory(
        self,
        directory_path: str,
        project_name: str = None,
        file_patterns: List[str] = None,
        incremental: bool = True,
        progress_callback=None,
    ) -> dict:
        """Implementation of index_directory tool."""
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
                progress_callback=progress_callback
            )
            
            # Auto-switch to newly indexed project
            self.switch_project(directory_path)
            
            return {
                "success": result.success,
                "files_added": result.files_added,
                "files_modified": result.files_modified,
                "files_removed": result.files_removed,
                "chunks_added": result.chunks_added,
                "chunks_removed": result.chunks_removed,
                "time_taken": round(result.time_taken, 2),
                "project_id": project_dir.name
            }
        except Exception as e:
            logger.error(f"Index failed: {e}")
            return {"success": False, "error": str(e)}

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
                projects = self.list_projects()
                if projects:
                   self.switch_project(projects[0]["project_path"])
        
        if not self._searcher:
            return [{"error": "No project selected. Provide project_path or run index_directory first."}]

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
            return [res.to_search_tool_dict() for res in results]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [{"error": str(e)}]

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
        if "error" not in stats:
            stats["model_info"] = self.embedder.get_model_info()
        return stats

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all indexed projects."""
        projects_dir = self.storage_root / "projects"
        if not projects_dir.exists():
            return []
            
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
                            "project_name": s.get("project_name", p_dir.name),
                            "project_path": s.get("project_path", "unknown"),
                            "total_chunks": s.get("total_chunks", 0),
                            "last_indexed": s.get("last_indexed", "unknown")
                        })
                except Exception:
                    pass
        return sorted(projects, key=lambda x: x.get("last_indexed", ""), reverse=True)

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

