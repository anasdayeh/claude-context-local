"""Incremental indexing logic using Merkle DAGs and change detection."""

import logging
import os
import time
import fnmatch
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from merkle.merkle_dag import MerkleDAG
from merkle.snapshot_manager import SnapshotManager
from merkle.change_detector import FileChanges, ChangeDetector
from embeddings.embedder import CodeEmbedder
from search.indexer import CodeIndexManager
from chunking.multi_language_chunker import MultiLanguageChunker
from search.resume_state import ResumeState, load_resume_state, save_resume_state, clear_resume_state

logger = logging.getLogger(__name__)

class IndexingCanceled(Exception):
    """Raised when an indexing run is cooperatively canceled."""


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
        index_manager: Optional[CodeIndexManager] = None,
        embedder: Optional[CodeEmbedder] = None,
        chunker: Optional[MultiLanguageChunker] = None,
        storage_dir: Optional[str] = None,
        indexer: Optional[CodeIndexManager] = None,
        snapshot_manager: Optional[SnapshotManager] = None,
    ):
        if index_manager is None and indexer is not None:
            index_manager = indexer
        if index_manager is None:
            raise ValueError("index_manager is required")

        self.indexer = index_manager
        self.embedder = embedder
        self.chunker = chunker

        if snapshot_manager is None:
            if storage_dir is None:
                try:
                    storage_dir = str(Path(self.indexer.storage_dir).parent)
                except Exception:
                    storage_dir = str(Path.cwd())
            snapshot_manager = SnapshotManager(storage_dir)
        self.snapshot_manager = snapshot_manager
        self.chunk_batch_size = self._read_env_int("CODE_SEARCH_CHUNK_BATCH_SIZE", 100)
        self.embed_batch_size = self._read_env_int(
            "CODE_SEARCH_EMBED_BATCH_SIZE",
            self._read_env_int("CODE_SEARCH_BATCH_SIZE", 32),
        )
        self.disk_warn_gb = self._read_env_int_allow_zero("CODE_SEARCH_DISK_WARN_GB", 5)
        self.large_file_mb = self._read_env_int_allow_zero("CODE_SEARCH_LARGE_FILE_MB", 20)
        self._progress_callback = None
        self._checkpoint_interval = self.chunk_batch_size * 5
        self._progress_every_files = self._read_env_int("CODE_SEARCH_PROGRESS_EVERY_FILES", 50)

    def _read_env_int(self, name: str, default: int) -> int:
        value = os.getenv(name)
        if not value:
            return default
        try:
            parsed = int(value)
            return parsed if parsed > 0 else default
        except ValueError:
            return default

    def _read_env_int_allow_zero(self, name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None or value == "":
            return default
        try:
            parsed = int(value)
            return parsed if parsed >= 0 else default
        except ValueError:
            return default

    def _pattern_matches(self, path: str, pattern_list) -> bool:
        norm_path = path.replace('\\', '/')
        patterns = pattern_list if isinstance(pattern_list, list) else [pattern_list]
        for pattern in patterns:
            norm_pattern = str(pattern).replace('\\', '/')
            # Direct match
            if fnmatch.fnmatch(norm_path, norm_pattern):
                return True
            # Match as sub-path for unanchored patterns
            if not norm_pattern.startswith('/') and not norm_pattern.startswith('./'):
                if fnmatch.fnmatch(norm_path, "*/" + norm_pattern):
                    return True
            # Match filename only
            if fnmatch.fnmatch(Path(norm_path).name, norm_pattern):
                return True
            # Match **/prefix patterns
            if norm_pattern.startswith("**/"):
                if fnmatch.fnmatch(norm_path, norm_pattern[3:]):
                    return True
        return False

    def _warn_if_low_disk(self) -> None:
        if self.disk_warn_gb is None:
            return
        try:
            usage = shutil.disk_usage(self.snapshot_manager.storage_dir)
        except Exception:
            return
        free_gb = usage.free / (1024 ** 3)
        if free_gb < self.disk_warn_gb:
            logger.warning(
                "Low disk space: %.2f GB free at %s",
                free_gb,
                self.snapshot_manager.storage_dir,
            )

    def _warn_if_large_file(self, file_path: str) -> None:
        if self.large_file_mb is None:
            return
        try:
            size_mb = os.path.getsize(file_path) / (1024 ** 2)
        except Exception:
            return
        if size_mb >= self.large_file_mb:
            logger.warning("Large file %.2f MB: %s", size_mb, file_path)

    def incremental_index(
        self,
        project_path: str,
        project_name: str,
        file_patterns: Optional[List[str]] = None,
        force_full: bool = False,
        progress_callback=None,
        cancel_event=None,
        resume: bool = True,
    ) -> IncrementalIndexResult:
        """Perform incremental indexing of a project."""
        self._progress_callback = progress_callback
        start_time = time.time()
        
        try:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                raise IndexingCanceled("Indexing canceled before start")

            if force_full:
                return self._full_index(project_path, project_name, start_time, file_patterns, cancel_event, resume)

            # Load latest snapshot
            latest_dag = self.snapshot_manager.load_latest_snapshot(project_path)
            if latest_dag is None:
                logger.info("No existing snapshot found. Performing full index.")
                return self._full_index(project_path, project_name, start_time, file_patterns, cancel_event, resume)

            # Build current DAG
            current_dag = MerkleDAG(project_path)
            current_dag.build()

            # Detect changes
            detector = ChangeDetector(self.snapshot_manager)
            changes = detector.detect_changes(latest_dag, current_dag)
            
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
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                raise IndexingCanceled("Indexing canceled")
            chunks_removed = self._remove_old_chunks(changes, project_name)
            chunks_added = self._add_new_chunks(changes, project_path, project_name, cancel_event)
            
            # Update snapshot
            self.snapshot_manager.save_snapshot(current_dag, {
                'project_name': project_name,
                'incremental_update': True,
                'files_added': len(changes.added),
                'files_removed': len(changes.removed),
                'files_modified': len(changes.modified)
            })
            
            # Update index
            self.indexer.save_index(extra_metadata={
                "project_name": project_name,
                "project_path": project_path,
                "status": "ready",
                "last_indexed": datetime.now().isoformat()
            })
            
            return IncrementalIndexResult(
                files_added=len(changes.added),
                files_removed=len(changes.removed),
                files_modified=len(changes.modified),
                chunks_added=chunks_added,
                chunks_removed=chunks_removed,
                time_taken=time.time() - start_time,
                success=True
            )
            
        except IndexingCanceled as e:
            logger.warning("%s", e)
            # Best-effort checkpoint before returning
            try:
                self.indexer.save_index()
            except Exception:
                pass
            return IncrementalIndexResult(
                0, 0, 0, 0, 0, time.time() - start_time, False, error=str(e)
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
            return self._pattern_matches(path, file_patterns)

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
        file_patterns: Optional[List[str]] = None,
        cancel_event=None,
        resume: bool = True,
    ) -> IncrementalIndexResult:
        """Perform full indexing of a project."""
        try:
            self._warn_if_low_disk()

            resume_enabled = resume and os.getenv("CODE_SEARCH_RESUME", "1").lower() not in {"0", "false", "no"}
            resume_state = None
            if resume_enabled:
                try:
                    resume_state = load_resume_state(Path(self.indexer.storage_dir))
                except Exception:
                    resume_state = None

            resume_active = (
                resume_state is not None
                and resume_state.status == "in_progress"
                and resume_state.project_path == project_path
            )

            # Clear existing index only when not resuming an in-progress run.
            if not resume_active:
                self.indexer.clear_index()
            
            # Save preliminary metadata so it's not "unknown" in list
            self.indexer.save_index(extra_metadata={
                "project_name": project_name,
                "project_path": project_path,
                "status": "indexing",
                "last_indexed": datetime.now().isoformat()
            })
            
            # Build DAG for all files
            dag = MerkleDAG(project_path)
            dag.build()
            all_files = dag.get_all_files()

            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                raise IndexingCanceled("Indexing canceled")
            
            # Filter supported and patterned files
            supported_files = []
            for f in all_files:
                if not self.chunker.is_supported(f):
                    continue
                
                if file_patterns:
                    if not self._pattern_matches(f, file_patterns):
                        continue
                
                supported_files.append(f)

            file_hashes = dag.get_file_hashes()
            files_to_process = supported_files
            removed_files: Set[str] = set()
            changed_files: Set[str] = set()

            if resume_enabled:
                if not resume_active:
                    resume_state = ResumeState(
                        project_path=project_path,
                        project_id=self.snapshot_manager.get_project_id(project_path),
                        status="in_progress",
                        files_total=len(supported_files),
                        files_completed=0,
                    )
                    save_resume_state(Path(self.indexer.storage_dir), resume_state)
                else:
                    resume_state.files_total = len(supported_files)

                if resume_active:
                    completed = set(resume_state.completed)
                    removed_files = completed - set(supported_files)
                    changed_files = {
                        f
                        for f in completed
                        if file_hashes.get(f) and resume_state.hashes.get(f) != file_hashes.get(f)
                    }
                    pending = [f for f in supported_files if f not in completed]
                    # Reindex changed files
                    pending.extend(sorted(changed_files))
                    files_to_process = list(dict.fromkeys(pending))

                    skipped_unchanged = max(0, len(completed) - len(changed_files) - len(removed_files))
                    logger.info(
                        "Resume active: completed=%d/%d skipped=%d changed=%d removed=%d pending=%d",
                        len(completed),
                        len(supported_files),
                        skipped_unchanged,
                        len(changed_files),
                        len(removed_files),
                        len(files_to_process),
                    )
                    if removed_files or changed_files:
                        for f in removed_files | changed_files:
                            resume_state.completed.discard(f)
                            resume_state.hashes.pop(f, None)
                        resume_state.files_completed = len(resume_state.completed)
                        save_resume_state(Path(self.indexer.storage_dir), resume_state)
                else:
                    logger.info(
                        "Resume initialized: completed=0/%d pending=%d",
                        len(supported_files),
                        len(files_to_process),
                    )

                if resume_active:
                    if removed_files:
                        for f in sorted(removed_files):
                            try:
                                self.indexer.remove_file_chunks(f)
                            except Exception:
                                pass
                    if changed_files:
                        for f in sorted(changed_files):
                            try:
                                self.indexer.remove_file_chunks(f)
                            except Exception:
                                pass

            chunks_added = 0
            self._chunks_processed_in_session = 0

            for file_path in files_to_process:
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    raise IndexingCanceled("Indexing canceled")
                full_path = (Path(project_path) / file_path).resolve()
                try:
                    self._warn_if_large_file(str(full_path))
                    chunks = self.chunker.chunk_file(str(full_path))
                except Exception as e:
                    logger.warning(f"Failed to chunk {file_path}: {e}")
                    chunks = []

                if not chunks:
                    if resume_state and resume_enabled:
                        resume_state.completed.add(file_path)
                        resume_state.hashes[file_path] = file_hashes.get(file_path, "")
                        resume_state.files_completed = len(resume_state.completed)
                    continue

                batch: List = []
                for chunk in chunks:
                    batch.append(chunk)
                    if len(batch) >= self.chunk_batch_size:
                        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                            raise IndexingCanceled("Indexing canceled")
                        processed = self._process_batch(batch, project_name)
                        chunks_added += processed
                        self._chunks_processed_in_session += processed
                        batch = []

                        if self._chunks_processed_in_session >= self._checkpoint_interval:
                            logger.info(f"Checkpoint: Saved {chunks_added} chunks so far...")
                            if self._progress_callback:
                                try:
                                    self._progress_callback(
                                        f"checkpoint: saved {chunks_added} chunks"
                                    )
                                except Exception:
                                    pass
                            self.indexer.save_index()
                            if resume_state and resume_enabled:
                                total = max(1, resume_state.files_total)
                                pct = (resume_state.files_completed / total) * 100.0
                                logger.info(
                                    "Progress: %d/%d files (%.1f%%)",
                                    resume_state.files_completed,
                                    resume_state.files_total,
                                    pct,
                                )
                            if resume_state and resume_enabled:
                                save_resume_state(Path(self.indexer.storage_dir), resume_state)
                            self._chunks_processed_in_session = 0

                if batch:
                    if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                        raise IndexingCanceled("Indexing canceled")
                    processed = self._process_batch(batch, project_name)
                    chunks_added += processed
                    self._chunks_processed_in_session += processed

                if resume_state and resume_enabled:
                    resume_state.completed.add(file_path)
                    resume_state.hashes[file_path] = file_hashes.get(file_path, "")
                    resume_state.files_completed = len(resume_state.completed)

                if (
                    self._progress_callback
                    and resume_state
                    and resume_enabled
                    and self._progress_every_files > 0
                    and resume_state.files_completed % self._progress_every_files == 0
                ):
                    total = max(1, resume_state.files_total)
                    pct = (resume_state.files_completed / total) * 100.0
                    try:
                        self._progress_callback(
                            f"progress: {resume_state.files_completed}/{resume_state.files_total} files ({pct:.1f}%)"
                        )
                    except Exception:
                        pass

                if self._chunks_processed_in_session >= self._checkpoint_interval:
                    logger.info(f"Checkpoint: Saved {chunks_added} chunks so far...")
                    if self._progress_callback:
                        try:
                            self._progress_callback(
                                f"checkpoint: saved {chunks_added} chunks"
                            )
                        except Exception:
                            pass
                    self.indexer.save_index()
                    if resume_state and resume_enabled:
                        total = max(1, resume_state.files_total)
                        pct = (resume_state.files_completed / total) * 100.0
                        logger.info(
                            "Progress: %d/%d files (%.1f%%)",
                            resume_state.files_completed,
                            resume_state.files_total,
                            pct,
                        )
                    if resume_state and resume_enabled:
                        save_resume_state(Path(self.indexer.storage_dir), resume_state)
                    self._chunks_processed_in_session = 0

            if resume_state and resume_enabled:
                resume_state.status = "ready"
                save_resume_state(Path(self.indexer.storage_dir), resume_state)
            
            # Save final snapshot
            self.snapshot_manager.save_snapshot(dag, {
                'project_name': project_name,
                'project_path': project_path,
                'incremental_update': False,
                'file_count': len(supported_files),
                'chunks_indexed': chunks_added,
                'status': 'ready'
            })
            
            # Final save
            self.indexer.save_index(extra_metadata={
                "project_name": project_name,
                "project_path": project_path,
                "status": "ready",
                "last_indexed": datetime.now().isoformat()
            })
            
            return IncrementalIndexResult(
                files_added=len(supported_files),
                files_removed=0,
                files_modified=0,
                chunks_added=chunks_added,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=True
            )
            
        except IndexingCanceled as e:
            logger.warning("%s", e)
            try:
                if resume_state and resume_enabled:
                    resume_state.status = "canceled"
                    save_resume_state(Path(self.indexer.storage_dir), resume_state)
                self.indexer.save_index(extra_metadata={
                    "project_name": project_name,
                    "project_path": project_path,
                    "status": "canceled",
                    "last_indexed": datetime.now().isoformat(),
                })
            except Exception:
                pass
            return IncrementalIndexResult(
                0, 0, 0, 0, 0, time.time() - start_time, False, error=str(e)
            )
        except Exception as e:
            logger.error(f"Full indexing failed: {e}")
            try:
                if resume_state and resume_enabled:
                    resume_state.status = "failed"
                    save_resume_state(Path(self.indexer.storage_dir), resume_state)
            except Exception:
                pass
            return IncrementalIndexResult(
                0, 0, 0, 0, 0, time.time() - start_time, False, error=str(e)
            )

    def _iter_chunks(self, files: List[str], project_path: str, cancel_event=None):
        """Yield chunks for each supported file."""
        for file_path in files:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                raise IndexingCanceled("Indexing canceled")
            full_path = (Path(project_path) / file_path).resolve()
            try:
                self._warn_if_large_file(str(full_path))
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

        prev_callback = getattr(self.embedder, "_progress_callback", None)
        if self._progress_callback:
            self.embedder._progress_callback = self._progress_callback
        try:
            embedding_results = self.embedder.embed_chunks(
                chunks,
                batch_size=self.embed_batch_size or 32,
            )
        finally:
            self.embedder._progress_callback = prev_callback
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

    def _add_new_chunks(
        self,
        changes: FileChanges,
        project_path: str,
        project_name: str,
        cancel_event=None,
    ) -> int:
        """Add chunks for new/modified files."""
        files_to_process = changes.added + changes.modified
        if not files_to_process:
            return 0

        chunks_added = 0
        batch: List = []
        self._chunks_processed_in_session = 0
        for chunk in self._iter_chunks(files_to_process, project_path, cancel_event):
            batch.append(chunk)
            if len(batch) >= self.chunk_batch_size:
                if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                    raise IndexingCanceled("Indexing canceled")
                processed = self._process_batch(batch, project_name)
                chunks_added += processed
                self._chunks_processed_in_session += processed
                batch = []
                
                if self._chunks_processed_in_session >= self._checkpoint_interval:
                    self.indexer.save_index()
                    self._chunks_processed_in_session = 0

        if batch:
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                raise IndexingCanceled("Indexing canceled")
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
