import numpy as np


def test_file_map_adds_and_removes_ids(tmp_path):
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

    int_id = index._lookup_int_id(emb.chunk_id)
    assert int_id is not None

    key = "a.py"
    stored_ids = index.file_map_db.get(key)
    assert stored_ids is not None
    assert int(int_id) in stored_ids

    removed = index.remove_file_chunks("a.py")
    assert removed == 1
    assert index.metadata_db.get(str(int_id)) is None
    assert index.file_map_db.get(key) in (None, [])
