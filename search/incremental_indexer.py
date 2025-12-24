"""Incremental indexing logic using Merkle DAGs and change detection."""

import logging
import time
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from merkle.merkle_dag import MerkleDAG
from merkle.snapshot_manager import SnapshotManager
from merkle.change_detector import FileChanges, ChangeDetector
from embeddings.embedder import CodeEmbedder
from search.indexer import CodeIndexManager
from chunking.multi_language_chunker import MultiLanguageChunker

logger = logging.getLogger(__name__)


@dataclass
class IncrementalIndexResult:
    """Results from an incremental indexing run."""
    files_added: int
    files_removed: int
    files_modified: int
    chunks_added: int
    chunks_removed: int
    time_taken: float
    success: bool
    error: Optional[str] = None


class IncrementalIndexer:
    """Orchestrates incremental indexing process."""

    def __init__(
        self,
        index_manager: CodeIndexManager,
        embedder: CodeEmbedder,
        chunker: MultiLanguageChunker,
        storage_dir: str
    ):
        self.indexer = index_manager
        self.embedder = embedder
        self.chunker = chunker
        self.snapshot_manager = SnapshotManager(storage_dir)
        self.chunk_batch_size = 100
        self.embed_batch_size = 32
        self._progress_callback = None

    def incremental_index(
        self,
        project_path: str,
        project_name: str,
        file_patterns: Optional[List[str]] = None,
        force_full: bool = False,
        progress_callback=None
    ) -> IncrementalIndexResult:
        """Perform incremental indexing of a project."""
        self._progress_callback = progress_callback
        start_time = time.time()
        
        try:
            if force_full:
                return self._full_index(project_path, project_name, start_time, file_patterns)

            # Load latest snapshot
            latest_dag = self.snapshot_manager.load_latest_snapshot(project_path)
            if latest_dag is None:
                logger.info("No existing snapshot found. Performing full index.")
                return self._full_index(project_path, project_name, start_time, file_patterns)

            # Build current DAG
            current_dag = MerkleDAG(project_path)
            current_dag.build()

            # Detect changes
            detector = ChangeDetector(self.snapshot_manager)
            changes = detector.get_changes(latest_dag, current_dag)
            
            # Apply file patterns if provided
            if file_patterns:
                changes = self._filter_changes(changes, file_patterns)

            if not changes.has_changes():
                logger.info("No changes detected since last index.")
                return IncrementalIndexResult(0, 0, 0, 0, 0, time.time() - start_time, True)

            logger.info(
                f"Incremental changes: Added: {len(changes.added)}, "
                f"Removed: {len(changes.removed)}, Modified: {len(changes.modified)}"
            )
            
            # Process changes
            chunks_removed = self._remove_old_chunks(changes, project_name)
            chunks_added = self._add_new_chunks(changes, project_path, project_name)
            
            # Update snapshot
            self.snapshot_manager.save_snapshot(current_dag, {
                'project_name': project_name,
                'incremental_update': True,
                'files_added': len(changes.added),
                'files_removed': len(changes.removed),
                'files_modified': len(changes.modified)
            })
            
            # Update index
            self.indexer.save_index()
            
            return IncrementalIndexResult(
                files_added=len(changes.added),
                files_removed=len(changes.removed),
                files_modified=len(changes.modified),
                chunks_added=chunks_added,
                chunks_removed=chunks_removed,
                time_taken=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Incremental indexing failed: {e}")
            return IncrementalIndexResult(
                0, 0, 0, 0, 0, time.time() - start_time, False, error=str(e)
            )
        finally:
            self._progress_callback = None

    def _filter_changes(self, changes: FileChanges, file_patterns: List[str]) -> FileChanges:
        """Filter changes based on robust glob patterns."""
        def matches(path: str) -> bool:
            norm_path = path.replace('\\', '/')
            for pattern in file_patterns:
                norm_pattern = pattern.replace('\\', '/')
                # 1. Direct match
                if fnmatch.fnmatch(norm_path, norm_pattern): return True
                # 2. Match as sub-path (unanchored)
                if not norm_pattern.startswith('/') and not norm_pattern.startswith('./'):
                    if fnmatch.fnmatch(norm_path, "*/" + norm_pattern): return True
                # 3. Match on filename
                if fnmatch.fnmatch(Path(norm_path).name, norm_pattern): return True
            return False

        return FileChanges(
            added=[f for f in changes.added if matches(f)],
            removed=[f for f in changes.removed if matches(f)],
            modified=[f for f in changes.modified if matches(f)]
        )

    def _full_index(
        self,
        project_path: str,
        project_name: str,
        start_time: float,
        file_patterns: Optional[List[str]] = None
    ) -> IncrementalIndexResult:
        """Perform full indexing of a project."""
        try:
            # Clear existing index
            self.indexer.clear_index()
            
            # Build DAG for all files
            dag = MerkleDAG(project_path)
            dag.build()
            all_files = dag.get_all_files()
            
            # Filter supported and patterned files
            supported_files = []
            for f in all_files:
                if not self.chunker.is_supported(f):
                    continue
                
                if file_patterns:
                    norm_path = f.replace('\\', '/')
                    match = False
                    for pattern in file_patterns:
                        norm_pattern = pattern.replace('\\', '/')
                        # 1. Direct match
                        if fnmatch.fnmatch(norm_path, norm_pattern): 
                            match = True; break
                        # 2. Match as sub-path (unanchored)
                        if not norm_pattern.startswith('/') and not norm_pattern.startswith('./'):
                            if fnmatch.fnmatch(norm_path, "*/" + norm_pattern):
                                match = True; break
                        # 3. Match on filename
                        if fnmatch.fnmatch(Path(norm_path).name, norm_pattern):
                            match = True; break
                    if not match:
                        continue
                
                supported_files.append(f)
            
            chunks_added = 0
            batch: List = []
            for chunk in self._iter_chunks(supported_files, project_path):
                batch.append(chunk)
                if len(batch) >= self.chunk_batch_size:
                    chunks_added += self._process_batch(batch, project_name)
                    batch = []

            if batch:
                chunks_added += self._process_batch(batch, project_name)
            
            # Save snapshot
            self.snapshot_manager.save_snapshot(dag, {
                'project_name': project_name,
                'incremental_update': False,
                'file_count': len(supported_files),
                'chunks_indexed': chunks_added
            })
            
            # Save index to disk
            self.indexer.save_index()
            
            return IncrementalIndexResult(
                files_added=len(supported_files),
                files_removed=0,
                files_modified=0,
                chunks_added=chunks_added,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Full indexing failed: {e}")
            return IncrementalIndexResult(
                0, 0, 0, 0, 0, time.time() - start_time, False, error=str(e)
            )

    def _iter_chunks(self, files: List[str], project_path: str):
        """Yield chunks for each supported file."""
        for file_path in files:
            full_path = (Path(project_path) / file_path).resolve()
            try:
                chunks = self.chunker.chunk_file(str(full_path))
                if not chunks:
                    continue
                for chunk in chunks:
                    yield chunk
            except Exception as e:
                logger.warning(f"Failed to chunk {file_path}: {e}")

    def _process_batch(self, chunks: List, project_name: str) -> int:
        """Process a batch of chunks: embed and index."""
        if not chunks:
            return 0

        embedding_results = self.embedder.embed_chunks(chunks, batch_size=self.embed_batch_size or 32)
        if not embedding_results:
            return 0

        self.indexer.add_embeddings(embedding_results, update_stats=False)
        return len(embedding_results)

    def _remove_old_chunks(self, changes: FileChanges, project_name: str) -> int:
        """Remove old chunks for modified/removed files."""
        chunks_removed = 0
        for file_path in changes.modified + changes.removed:
            chunks_removed += self.indexer.remove_file_chunks(file_path)
        return chunks_removed

    def _add_new_chunks(self, changes: FileChanges, project_path: str, project_name: str) -> int:
        """Add chunks for new/modified files."""
        files_to_process = changes.added + changes.modified
        if not files_to_process:
            return 0

        chunks_added = 0
        batch: List = []
        for chunk in self._iter_chunks(files_to_process, project_path):
            batch.append(chunk)
            if len(batch) >= self.chunk_batch_size:
                chunks_added += self._process_batch(batch, project_name)
                batch = []

        if batch:
            chunks_added += self._process_batch(batch, project_name)

        return chunks_added

    def auto_reindex_if_needed(
        self,
        project_path: str,
        project_name: Optional[str] = None,
        max_age_minutes: float = 5,
        file_patterns: Optional[List[str]] = None
    ) -> IncrementalIndexResult:
        """Automatically reindex if snapshot is too old."""
        if not project_name:
            project_name = Path(project_path).name

        if not self.needs_reindex(project_path, max_age_minutes=max_age_minutes):
            return IncrementalIndexResult(0, 0, 0, 0, 0, 0.0, True)

        logger.info(f"Auto-reindexing {project_name}")
        return self.incremental_index(project_path, project_name, file_patterns=file_patterns, force_full=False)

    def needs_reindex(self, project_path: str, max_age_minutes: float = 5) -> bool:
        """Check if a project needs reindexing based on snapshot age or changes."""
        stats = self.get_indexing_stats(project_path)
        if not stats:
            return True

        detector = ChangeDetector(self.snapshot_manager)
        if detector.quick_check(project_path):
            return True

        age_seconds = stats.get('snapshot_age', float('inf'))
        return age_seconds > max_age_minutes * 60

    def get_indexing_stats(self, project_path: str) -> Dict[str, Any]:
        """Get indexing statistics for a project."""
        metadata = self.snapshot_manager.load_metadata(project_path)
        if metadata is None:
            return None
        try:
            index_stats = self.indexer.get_stats()
            metadata["current_chunks"] = index_stats.get("total_chunks", 0)
        except Exception:
            metadata.setdefault("current_chunks", 0)
        snapshot_age = self.snapshot_manager.get_snapshot_age(project_path)
        if snapshot_age is not None:
            metadata["snapshot_age"] = snapshot_age
        return metadata
