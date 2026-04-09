from chunking.text_chunker import TextChunker


def test_chunk_text_preserves_document_metadata(tmp_path):
    chunker = TextChunker(root_path=str(tmp_path))
    file_path = tmp_path / "notes.pdf"
    file_path.write_text("placeholder")

    chunks = chunker.chunk_text(
        "Alpha line\nBeta line",
        str(file_path),
        chunk_type="text",
        name="notes.pdf#page:1",
        tags=["document", "pdf", "page"],
        extra_metadata={
            "document_type": "pdf",
            "block_kind": "page",
            "page_number": 1,
            "ocr_used": False,
            "source_format": ".pdf",
        },
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.name == "notes.pdf#page:1"
    assert chunk.tags == ["document", "pdf", "page"]
    assert chunk.extra_metadata["page_number"] == 1
    assert chunk.extra_metadata["document_type"] == "pdf"
