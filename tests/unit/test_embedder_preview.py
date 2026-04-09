import numpy as np

from chunking.code_chunk import CodeChunk
from embeddings.embedder import EmbeddingResult


def _chunk_with_content(content: str) -> CodeChunk:
    return CodeChunk(
        content=content,
        chunk_type="function",
        start_line=1,
        end_line=5,
        file_path="/tmp/a.py",
        relative_path="a.py",
        folder_structure=[],
        name="f",
    )


def test_content_preview_truncated_by_default(monkeypatch):
    monkeypatch.delenv("CODE_SEARCH_CONTENT_PREVIEW_CHARS", raising=False)
    chunk = _chunk_with_content("x" * 400)
    result = EmbeddingResult(chunk=chunk, embedding=np.zeros(8, dtype=np.float32))
    meta = result.metadata
    assert len(meta["content"]) == 400
    assert len(meta["content_preview"]) == 323
    assert meta["content_preview"].endswith("...")


def test_content_preview_respects_env_override(monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_CONTENT_PREVIEW_CHARS", "64")
    chunk = _chunk_with_content("y" * 200)
    result = EmbeddingResult(chunk=chunk, embedding=np.zeros(8, dtype=np.float32))
    assert len(result.metadata["content_preview"]) == 67


def test_document_extra_metadata_is_merged():
    chunk = _chunk_with_content("document text")
    chunk.extra_metadata = {
        "document_type": "pdf",
        "page_number": 2,
        "ocr_used": False,
    }
    result = EmbeddingResult(chunk=chunk, embedding=np.zeros(8, dtype=np.float32))
    meta = result.metadata
    assert meta["document_type"] == "pdf"
    assert meta["page_number"] == 2
    assert meta["ocr_used"] is False
