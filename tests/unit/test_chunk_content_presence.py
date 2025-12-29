import textwrap

import pytest

from chunking.multi_language_chunker import MultiLanguageChunker


def test_chunks_always_have_content(tmp_path):
    source = textwrap.dedent(
        """
        class Service:
            def process(self, value):
                pass

        def helper():
            return 42
        """
    )
    file_path = tmp_path / "example.py"
    file_path.write_text(source)
    chunker = MultiLanguageChunker(root_path=str(tmp_path))
    chunks = chunker.chunk_file(str(file_path))

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.content is not None
        assert chunk.content.strip(), f"Chunk {chunk.chunk_type} had empty content"
