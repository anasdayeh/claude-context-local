"""Main embedding logic for handling code and queries."""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import numpy as np

from chunking.code_chunk import CodeChunk
from embeddings.embedding_models_register import AVAILIABLE_MODELS

@dataclass
class EmbeddingResult:
    """Result of embedding generation for a chunk."""
    chunk: CodeChunk
    embedding: np.ndarray
    model_name: str
    tokens: int = 0
    @property
    def chunk_id(self) -> str:
        """Generate a stable unique ID for this chunk."""
        import hashlib
        # Combine path, name, lines, and type for uniqueness
        raw_id = f"{self.chunk.relative_path}:{self.chunk.name}:{self.chunk.start_line}:{self.chunk.chunk_type}"
        return hashlib.md5(raw_id.encode()).hexdigest()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Convert chunk into storage-ready metadata dictionary."""
        return {
            "name": self.chunk.name,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk.chunk_type,
            "start_line": self.chunk.start_line,
            "end_line": self.chunk.end_line,
            "relative_path": self.chunk.relative_path,
            "file_path": self.chunk.file_path,
            "parent_name": self.chunk.parent_name,
            "tags": self.chunk.tags,
            "content": self.chunk.content,
            "content_preview": self.chunk.content,
            "folder_structure": self.chunk.folder_structure,
            "model": self.model_name
        }

class CodeEmbedder:
    """Handles embedding generation for code chunks and search queries using semantic models."""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        device: str = "auto",
        cache_dir: Optional[str] = None
    ):
        """Initialize code embedder."""
        self._logger = logging.getLogger(__name__)

        # Normalize model name if using known aliases
        if model_name in AVAILIABLE_MODELS:
            model_class = AVAILIABLE_MODELS[model_name]
        else:
            model_class = AVAILIABLE_MODELS.get(model_name)
            if not model_class:
                for k, v in AVAILIABLE_MODELS.items():
                    if k.endswith(model_name) or model_name.endswith(k):
                        model_class = v
                        model_name = k
                        break
            
            if not model_class:
                available = sorted(AVAILIABLE_MODELS.keys())
                raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        self.model_name = model_name
        try:
            self._model = model_class(
                model_name,
                device=device,
                cache_dir=cache_dir
            )
        except Exception as e:
            msg = f"Failed to load model '{model_name}': {e}"
            self._logger.error(msg)
            raise RuntimeError(msg) from e

    @property
    def raw_model(self):
        """Access the underlying SentenceTransformer model for introspection."""
        return getattr(self._model, "model", self._model)

    def _resolve_prompt_name(self, is_query: bool) -> Optional[str]:
        """Resolve generic prompt names to model-specific ones via introspection."""
        prompts = getattr(self.raw_model, "prompts", {})
        if not prompts:
            return None
        if is_query:
            return "query" if "query" in prompts else None
        else:
            return "document" if "document" in prompts else None

    def _encode_documents(self, texts: List[str]) -> np.ndarray:
        """Encode documents using robust wrapper methods."""
        encode_kwargs = {"show_progress_bar": False}
        
        if hasattr(self._model, "encode_document"):
            embeddings = self._model.encode_document(texts, **encode_kwargs)
        else:
            prompt_name = self._resolve_prompt_name(is_query=False)
            embeddings = self._model.encode(
                texts,
                prompt_name=prompt_name,
                **encode_kwargs,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def _encode_queries(self, texts: List[str]) -> np.ndarray:
        """Encode queries using robust wrapper methods."""
        encode_kwargs = {"show_progress_bar": False}

        if hasattr(self._model, "encode_query"):
            embeddings = self._model.encode_query(texts, **encode_kwargs)
        else:
            prompt_name = self._resolve_prompt_name(is_query=True)
            embeddings = self._model.encode(
                texts,
                prompt_name=prompt_name,
                **encode_kwargs,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def create_embedding_content(self, chunk: CodeChunk, max_chars: int = 2048) -> str:
        """Create formatted content string for embedding."""
        parts = []
        name = chunk.name or "unknown"
        chunk_type = chunk.chunk_type or "unknown"
        
        parts.append(f"Name: {name}")
        parts.append(f"Type: {chunk_type}")
        
        if getattr(chunk, 'parent_name', None):
             parts.append(f"Context: {chunk.parent_name}")

        tags = chunk.tags or []
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")

        docstring = chunk.docstring or ""
        overhead = sum(len(p) + 1 for p in parts) + 10 
        
        remaining_budget = max_chars - overhead
        docstring_len = len(docstring)
        
        if remaining_budget <= 20:
             return f"Name: {name}\n{(chunk.content or '')[:max_chars//2]}"

        docstring_budget = min(docstring_len, int(remaining_budget * 0.3))
        if docstring_len < remaining_budget * 0.5:
             docstring_budget = docstring_len
             
        if docstring:
             parts.append(f"Docstring: {docstring[:docstring_budget]}")
             
        current_used = sum(len(p) + 1 for p in parts)
        code_budget = max(0, max_chars - current_used)
        
        content = chunk.content or ""
        header = "\n".join(parts)
        
        if not content:
            return header
            
        full_text = f"{header}\n{content}"
        
        if len(full_text) > max_chars:
            allowed_content_len = max(0, max_chars - len(header) - 1)
            return f"{header}\n{content[:allowed_content_len]}"
            
        return full_text

    def embed_chunks(self, chunks: List[CodeChunk], batch_size: int = 32) -> List[EmbeddingResult]:
        """Generate embeddings for code chunks in batches."""
        if not chunks:
            return []

        texts = [self.create_embedding_content(chunk) for chunk in chunks]
        results = []
        total = len(texts)
        
        self._logger.info(f"Generating embeddings for {total} chunks (batch_size={batch_size})")

        for i in range(0, total, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_chunks = chunks[i : i + batch_size]
            try:
                self._logger.info(f"Loop {i}: text_len={len(batch_texts)} chunk_len={len(batch_chunks)}")
                batch_embeddings = self._encode_documents(batch_texts)
                self._logger.info(f"Loop {i}: embed_len={len(batch_embeddings)}")
                
                # Zip embeddings with chunks to create result objects
                batch_results = []
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    batch_results.append(EmbeddingResult(
                        chunk=chunk,
                        embedding=embedding,
                        model_name=self.model_name
                    ))
                self._logger.info(f"Loop {i}: zipped_results={len(batch_results)}")
                results.extend(batch_results)
            except Exception as e:
                self._logger.error(f"Batch encoding failed at index {i}: {e}")
                raise

        self._logger.info(f"Embedding generation completed. Results: {len(results)}")
        return results

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode_queries([query])[0]

    def embed_document(self, text: str) -> np.ndarray:
        return self._encode_documents([text])[0]

    def get_model_info(self) -> Dict[str, Any]:
        return self._model.get_model_info()

    def cleanup(self):
        if hasattr(self, '_model'):
            self._model.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
