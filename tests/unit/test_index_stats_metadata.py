import numpy as np


def test_stats_include_index_metadata(tmp_path):
    from search.indexer import CodeIndexManager
    from embeddings.embedder import EmbeddingResult
    from chunking.code_chunk import CodeChunk

    index_dir = tmp_path / "index"
    index = CodeIndexManager(str(index_dir))

    chunk = CodeChunk(
        content="def f():\n    pass\n",
        chunk_type="function",
        start_line=1,
        end_line=2,
        file_path=str(tmp_path / "a.py"),
        relative_path="a.py",
        folder_structure=[],
        name="f",
    )
    emb = EmbeddingResult(chunk=chunk, embedding=np.ones(4, dtype=np.float32))
    index.add_embeddings([emb])

    stats = index.get_stats()
    assert "index_type" in stats
    assert "metric" in stats
    assert "embedding_dim" in stats
    assert "trained" in stats
