import numpy as np
import tempfile

from chunking.code_chunk import CodeChunk
from embeddings.embedder import EmbeddingResult
from search.indexer import CodeIndexManager


def make_dummy_chunk(tmp_path):
    file_path = tmp_path / "dummy.py"
    file_path.write_text("def foo():\n    pass\n")
    return CodeChunk(
        content="def foo():\n    pass\n",
        chunk_type="function",
        start_line=1,
        end_line=2,
        file_path=str(file_path),
        relative_path="dummy.py",
        folder_structure=[],
        name="foo"
    )


def make_embedding_result(chunk, metadata_override=None):
    embedding = np.zeros((1,), dtype=np.float32)
    return EmbeddingResult(
        chunk=chunk,
        embedding=embedding,
        metadata=metadata_override or {},
    )


def test_resolve_fts_content_priority(tmp_path):
    manager = CodeIndexManager(str(tmp_path / "storage"))
    chunk = make_dummy_chunk(tmp_path)
    metadata = {"content": "", "chunk_type": "function"}
    result = make_embedding_result(chunk, metadata_override=metadata)
    assert manager._resolve_fts_content(result).strip() == chunk.content.strip()


def test_resolve_fts_content_fallback(tmp_path):
    manager = CodeIndexManager(str(tmp_path / "storage"))
    chunk = make_dummy_chunk(tmp_path)
    chunk.content = ""
    metadata = {"content": "", "chunk_type": "", "name": "fallback"}
    result = make_embedding_result(chunk, metadata_override=metadata)
    assert manager._resolve_fts_content(result) == "fallback"
