"""Vector index management with FAISS and metadata storage."""

import os
import json
import logging
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import numpy as np
import faiss
from sqlitedict import SqliteDict
from embeddings.embedder import EmbeddingResult
from chunking.code_chunk import CodeChunk


class CodeIndexManager:
    """Manages FAISS vector index and metadata storage for code chunks."""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_path = self.storage_dir / "code.index"
        self.metadata_path = self.storage_dir / "metadata.db"
        self.id_map_path = self.storage_dir / "id_map.db"
        self.stats_path = self.storage_dir / "stats.json"
        
        # Initialize components
        self._index = None
        self._metadata_db = None
        self._id_map_db = None
        self._logger = logging.getLogger(__name__)
        self._on_gpu = False
        self._legacy_index_map = None
        
    @property
    def index(self):
        """Lazy loading of FAISS index."""
        if self._index is None:
            self._load_index()
        return self._index
    
    @property
    def metadata_db(self):
        """Lazy loading of metadata database."""
        if self._metadata_db is None:
            self._metadata_db = self._open_sqlitedict(self.metadata_path)
        return self._metadata_db

    @property
    def id_map_db(self):
        """Lazy loading of chunk_id -> int_id map."""
        if self._id_map_db is None:
            self._id_map_db = self._open_sqlitedict(self.id_map_path)
        return self._id_map_db

    def _open_sqlitedict(self, path: Path) -> SqliteDict:
        """Open a SqliteDict with basic corruption recovery."""
        def _open():
            return SqliteDict(
                str(path),
                autocommit=False,
                journal_mode="WAL"
            )

        try:
            self._cleanup_orphaned_wal(path)
            return _open()
        except sqlite3.Error as e:
            message = str(e).lower()
            if "disk i/o" in message or "malformed" in message or "not a database" in message:
                self._logger.warning(f"SQLite error opening {path}: {e}. Backing up and recreating.")
                self._backup_sqlite_files(path)
                return _open()
            raise

    def _backup_sqlite_files(self, path: Path) -> None:
        """Move sqlite db and wal/shm to a timestamped backup."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        for suffix in ("", "-wal", "-shm"):
            file_path = Path(f"{path}{suffix}")
            if file_path.exists():
                backup = file_path.with_suffix(file_path.suffix + f".corrupt-{timestamp}")
                try:
                    file_path.rename(backup)
                except Exception:
                    try:
                        file_path.unlink()
                    except Exception:
                        pass

    def _cleanup_orphaned_wal(self, path: Path) -> None:
        """Remove orphaned WAL/SHM if base DB is missing."""
        if path.exists():
            return
        for suffix in ("-wal", "-shm"):
            orphan = Path(f"{path}{suffix}")
            if orphan.exists():
                try:
                    orphan.unlink()
                except Exception:
                    pass
    
    def _load_index(self):
        """Load existing FAISS index or create new one."""
        if self.index_path.exists():
            self._logger.info(f"Loading existing index from {self.index_path}")
            self._index = faiss.read_index(str(self.index_path))
            # If GPU support is available, optionally move to GPU for runtime speed
            self._maybe_move_index_to_gpu()

            if not isinstance(self._index, faiss.IndexIDMap2):
                self._logger.warning(
                    "Loaded legacy FAISS index without ID mapping. "
                    "Reindex is recommended for reliable deletions."
                )

            # Warn if legacy metadata format is detected
            try:
                for key in self.metadata_db.keys():
                    if not str(key).isdigit():
                        self._logger.warning(
                            "Legacy metadata format detected. "
                            "Please reindex to migrate to ID-mapped metadata."
                        )
                        self._build_legacy_index_map()
                        break
            except Exception:
                pass
            
        else:
            self._logger.info("Creating new index")
            # Create a new index - we'll initialize it when we get the first embedding
            self._index = None
            # id maps are stored in sqlite
    
    def create_index(self, embedding_dimension: int, index_type: str = "flat"):
        """Create a new FAISS index."""
        if index_type == "flat":
            # Simple flat index for exact search
            base_index = faiss.IndexFlatIP(embedding_dimension)  # Inner product (cosine similarity)
            self._index = faiss.IndexIDMap2(base_index)
        elif index_type == "ivf":
            # IVF index for faster approximate search on large datasets
            quantizer = faiss.IndexFlatIP(embedding_dimension)
            n_centroids = min(100, max(10, embedding_dimension // 8))  # Adaptive number of centroids
            base_index = faiss.IndexIVFFlat(quantizer, embedding_dimension, n_centroids)
            self._index = faiss.IndexIDMap2(base_index)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self._logger.info(f"Created {index_type} index with IDMap2, dimension {embedding_dimension}")
        self._maybe_move_index_to_gpu()
    
    def add_embeddings(self, embedding_results: List[EmbeddingResult]) -> None:
        """Add embeddings to the index and metadata to the database."""
        if not embedding_results:
            return

        if self._index is not None and not isinstance(self._index, faiss.IndexIDMap2):
            raise RuntimeError(
                "Cannot add embeddings to legacy FAISS index without ID mapping. "
                "Please reindex this project to migrate."
            )
        
        # Initialize index if needed
        if self._index is None:
            embedding_dim = embedding_results[0].embedding.shape[0]
            # Default to flat index for better recall - only use IVF for very large datasets
            index_type = "ivf" if len(embedding_results) > 10000 else "flat"
            self.create_index(embedding_dim, index_type)
        
        # Prepare embeddings and metadata
        embeddings = np.asarray([result.embedding for result in embedding_results], dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train IVF index if needed
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            self._logger.info("Training IVF index...")
            self._index.train(embeddings)
        
        # Generate stable integer IDs
        ids = np.array([self._get_or_create_int_id(r.chunk_id) for r in embedding_results], dtype=np.int64)

        # Remove existing IDs to prevent duplicates
        existing_ids = []
        for int_id in ids:
            if str(int(int_id)) in self.metadata_db:
                existing_ids.append(int(int_id))
        if existing_ids:
            try:
                self._index.remove_ids(np.asarray(existing_ids, dtype=np.int64))
            except Exception as e:
                self._logger.warning(f"Failed to remove existing IDs before re-add: {e}")

        # Add to FAISS index with explicit IDs
        self._index.add_with_ids(embeddings, ids)
        
        # Store metadata and update id map
        for i, result in enumerate(embedding_results):
            int_id = int(ids[i])
            self.metadata_db[str(int_id)] = {
                'chunk_id': result.chunk_id,
                'metadata': result.metadata
            }
        
        self._logger.info(f"Added {len(embedding_results)} embeddings to index")
        
        # Commit metadata in a single transaction for performance
        try:
            self.metadata_db.commit()
            self.id_map_db.commit()
        except Exception:
            # If commit is unavailable for some reason, continue without failing
            pass
        
        # Update statistics
        self._update_stats()

    def _gpu_is_available(self) -> bool:
        """Check if GPU FAISS support is available and GPUs are present."""
        try:
            if not hasattr(faiss, 'StandardGpuResources'):
                return False
            get_num_gpus = getattr(faiss, 'get_num_gpus', None)
            if get_num_gpus is None:
                return False
            return get_num_gpus() > 0
        except Exception:
            return False

    def _maybe_move_index_to_gpu(self) -> None:
        """Move the current index to GPU if supported. No-op if already on GPU or unsupported."""
        if self._index is None or self._on_gpu:
            return
        if not self._gpu_is_available():
            return
        try:
            # Move index to all GPUs for faster add/search
            self._index = faiss.index_cpu_to_all_gpus(self._index)
            self._on_gpu = True
            self._logger.info("FAISS index moved to GPU(s)")
        except Exception as e:
            self._logger.warning(f"Failed to move FAISS index to GPU, continuing on CPU: {e}")

    def _stable_int_id(self, chunk_id: str) -> int:
        """Create a stable 63-bit integer ID from a chunk_id."""
        digest = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        return value & 0x7FFFFFFFFFFFFFFF

    def _build_legacy_index_map(self) -> None:
        """Build index_id -> chunk_id map for legacy indexes."""
        try:
            legacy_map = {}
            for chunk_id, entry in self.metadata_db.items():
                index_id = entry.get("index_id")
                if isinstance(index_id, int):
                    legacy_map[index_id] = chunk_id
            self._legacy_index_map = legacy_map
            self._logger.info(f"Legacy index map built with {len(legacy_map)} entries")
        except Exception as e:
            self._logger.warning(f"Failed to build legacy index map: {e}")

    def _get_or_create_int_id(self, chunk_id: str) -> int:
        """Return a stable integer ID for a given chunk_id, handling collisions."""
        existing = self.id_map_db.get(chunk_id)
        if existing is not None:
            return int(existing)

        candidate = self._stable_int_id(chunk_id)
        salt = 0
        while True:
            existing_entry = self.metadata_db.get(str(int(candidate)))
            if not existing_entry or existing_entry.get("chunk_id") == chunk_id:
                break
            salt += 1
            candidate = self._stable_int_id(f"{chunk_id}#{salt}")

        self.id_map_db[chunk_id] = int(candidate)
        return int(candidate)

    def _lookup_int_id(self, chunk_id: str) -> Optional[int]:
        """Lookup an int ID for a chunk_id."""
        value = self.id_map_db.get(chunk_id)
        if value is None:
            return None
        return int(value)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar code chunks."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Index manager search called with k={k}, filters={filters}")
        
        # Use property to trigger lazy loading
        index = self.index
        if index is None or index.ntotal == 0:
            logger.warning(f"Index is empty or None. Index: {index}, ntotal: {index.ntotal if index else 'N/A'}")
            return []
        
        logger.info(f"Index has {index.ntotal} total vectors")
        
        # Normalize query embedding
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        search_k = min(k * 3, index.ntotal)  # Get more results for filtering
        similarities, indices = index.search(query_embedding, search_k)
        
        results = []
        for i, (similarity, int_id) in enumerate(zip(similarities[0], indices[0])):
            if int_id == -1:  # No more results
                break

            metadata_entry = None
            chunk_id = None

            if isinstance(self._index, faiss.IndexIDMap2):
                metadata_entry = self.metadata_db.get(str(int(int_id)))
                if metadata_entry is None:
                    continue
                metadata = metadata_entry['metadata']
                chunk_id = metadata_entry.get('chunk_id')
            else:
                if self._legacy_index_map is None:
                    self._build_legacy_index_map()
                if not self._legacy_index_map:
                    continue
                legacy_chunk_id = self._legacy_index_map.get(int(int_id))
                if not legacy_chunk_id:
                    continue
                metadata_entry = self.metadata_db.get(legacy_chunk_id)
                if metadata_entry is None:
                    continue
                metadata = metadata_entry['metadata']
                chunk_id = legacy_chunk_id
            
            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            if chunk_id:
                results.append((chunk_id, float(similarity), metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        for key, value in filters.items():
            if key == 'file_pattern':
                # Pattern matching for file paths
                if not any(pattern in metadata.get('relative_path', '') for pattern in value):
                    return False
            elif key == 'chunk_type':
                # Exact match for chunk type
                if metadata.get('chunk_type') != value:
                    return False
            elif key == 'tags':
                # Tag intersection
                chunk_tags = set(metadata.get('tags', []))
                required_tags = set(value if isinstance(value, list) else [value])
                if not required_tags.intersection(chunk_tags):
                    return False
            elif key == 'folder_structure':
                # Check if any of the required folders are in the path
                chunk_folders = set(metadata.get('folder_structure', []))
                required_folders = set(value if isinstance(value, list) else [value])
                if not required_folders.intersection(chunk_folders):
                    return False
            elif key in metadata:
                # Direct metadata comparison
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata by ID."""
        int_id = self._lookup_int_id(chunk_id)
        if int_id is None:
            # Legacy lookup
            metadata_entry = self.metadata_db.get(chunk_id)
            return metadata_entry['metadata'] if metadata_entry else None
        metadata_entry = self.metadata_db.get(str(int_id))
        return metadata_entry['metadata'] if metadata_entry else None
    
    def get_similar_chunks(self, chunk_id: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find chunks similar to a given chunk."""
        int_id = self._lookup_int_id(chunk_id)
        if int_id is None:
            # Legacy mapping: use index_id from metadata entry if present
            legacy_entry = self.metadata_db.get(chunk_id)
            if not legacy_entry:
                return []
            legacy_index_id = legacy_entry.get("index_id")
            if legacy_index_id is None:
                return []
            int_id = legacy_index_id

        if self._index is None or self._index.ntotal == 0:
            return []

        # Get the embedding for this chunk
        try:
            embedding = self._index.reconstruct(int_id)
        except Exception:
            return []
        
        # Search for similar chunks (excluding the original)
        results = self.search(embedding, k + 1)
        
        # Filter out the original chunk
        return [(cid, sim, meta) for cid, sim, meta in results if cid != chunk_id][:k]
    
    def remove_file_chunks(self, file_path: str, project_name: Optional[str] = None) -> int:
        """Remove all chunks from a specific file.
        
        Args:
            file_path: Path to the file (relative or absolute)
            project_name: Optional project name filter
            
        Returns:
            Number of chunks removed
        """
        ids_to_remove: List[int] = []
        
        # Find chunks to remove
        for int_id_str, metadata_entry in self.metadata_db.items():
            metadata = metadata_entry['metadata']
            
            # Check if this chunk belongs to the file
            chunk_file = metadata.get('file_path') or metadata.get('relative_path')
            if not chunk_file:
                continue
            
            # Check if paths match (handle both relative and absolute)
            if file_path in chunk_file or chunk_file in file_path:
                # Check project name if provided
                if project_name and metadata.get('project_name') != project_name:
                    continue
                ids_to_remove.append(int(int_id_str))
        
        if not ids_to_remove:
            return 0

        # Remove from FAISS index
        if self._index is not None:
            try:
                self._index.remove_ids(np.asarray(ids_to_remove, dtype=np.int64))
            except Exception as e:
                self._logger.warning(f"Failed to remove IDs from FAISS: {e}")

        # Remove chunks from metadata and id map
        for int_id in ids_to_remove:
            metadata_entry = self.metadata_db.get(str(int_id))
            if metadata_entry:
                chunk_id = metadata_entry.get("chunk_id")
                if chunk_id in self.id_map_db:
                    del self.id_map_db[chunk_id]
            if str(int_id) in self.metadata_db:
                del self.metadata_db[str(int_id)]
        
        self._logger.info(f"Removed {len(ids_to_remove)} chunks from {file_path}")
        
        # Commit removals in batch
        try:
            self.metadata_db.commit()
            self.id_map_db.commit()
        except Exception:
            pass
        return len(ids_to_remove)
    
    def save_index(self):
        """Save the FAISS index to disk."""
        if self._index is not None:
            try:
                # If on GPU, convert to CPU before saving
                index_to_write = self._index
                if self._on_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                    index_to_write = faiss.index_gpu_to_cpu(self._index)
                faiss.write_index(index_to_write, str(self.index_path))
                self._logger.info(f"Saved index to {self.index_path}")
            except Exception as e:
                self._logger.warning(f"Failed to save GPU index directly, attempting CPU fallback: {e}")
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self._index)
                    faiss.write_index(cpu_index, str(self.index_path))
                    self._logger.info(f"Saved index to {self.index_path} (CPU fallback)")
                except Exception as e2:
                    self._logger.error(f"Failed to save FAISS index: {e2}")
        
        self._update_stats()
    
    def _update_stats(self):
        """Update index statistics."""
        total_chunks = len(self.metadata_db)
        stats = {
            'total_chunks': total_chunks,
            'index_size': self._index.ntotal if self._index else 0,
            'embedding_dimension': self._index.d if self._index else 0,
            'index_type': type(self._index).__name__ if self._index else 'None'
        }
        
        # Add file and folder statistics
        file_counts = {}
        folder_counts = {}
        chunk_type_counts = {}
        tag_counts = {}
        
        for _, metadata_entry in self.metadata_db.items():
            metadata = metadata_entry['metadata']
            
            # Count by file
            file_path = metadata.get('relative_path', 'unknown')
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
            
            # Count by folder
            for folder in metadata.get('folder_structure', []):
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            # Count by chunk type
            chunk_type = metadata.get('chunk_type', 'unknown')
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            
            # Count by tags
            for tag in metadata.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        stats.update({
            'files_indexed': len(file_counts),
            'top_folders': dict(sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'chunk_types': chunk_type_counts,
            'top_tags': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        })
        
        # Save stats
        with open(self.stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        # Ensure index is loaded so ntotal and dimension are accurate
        _ = self.index
        if self.stats_path.exists():
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        if len(self.metadata_db) > 0:
            self._update_stats()
            if self.stats_path.exists():
                with open(self.stats_path, 'r') as f:
                    return json.load(f)
        return {
            'total_chunks': 0,
            'index_size': 0,
            'embedding_dimension': 0,
            'files_indexed': 0
        }
    
    def get_index_size(self) -> int:
        """Get the number of chunks in the index."""
        return len(self.metadata_db)
    
    def clear_index(self):
        """Clear the entire index and metadata."""
        # Close database connection
        if self._metadata_db is not None:
            self._metadata_db.close()
            self._metadata_db = None
        
        # Remove files
        legacy_chunk_ids = self.storage_dir / "chunk_ids.pkl"
        for file_path in [self.index_path, self.metadata_path, self.id_map_path, self.stats_path, legacy_chunk_ids]:
            if file_path.exists():
                file_path.unlink()
        
        # Reset in-memory state
        self._index = None
        if self._metadata_db is not None:
            self._metadata_db.close()
            self._metadata_db = None
        if self._id_map_db is not None:
            self._id_map_db.close()
            self._id_map_db = None
        
        self._logger.info("Index cleared")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._metadata_db is not None:
            self._metadata_db.close()
        if self._id_map_db is not None:
            self._id_map_db.close()
