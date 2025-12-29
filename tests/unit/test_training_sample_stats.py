import numpy as np


def test_stats_include_training_sample_counts(tmp_path):
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
    index.save_index()

    stats = index.get_stats()
    assert "training_sample_count" in stats
    assert "training_sample_total_seen" in stats
    assert stats["training_sample_count"] >= 1
    assert stats["training_sample_total_seen"] >= 1


def test_stats_merge_training_sample_when_stats_file_stale(tmp_path):
    import json
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
    index.save_index()

    # Simulate a stale stats file that predates training sample fields.
    index.stats_path.write_text(json.dumps({"total_chunks": 1, "files_indexed": 1}))

    stats = index.get_stats()
    assert stats["training_sample_count"] >= 1
    assert stats["training_sample_total_seen"] >= 1
