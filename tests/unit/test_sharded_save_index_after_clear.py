import json
import numpy as np


def test_save_index_after_clear_writes_shard_index(tmp_path):
    from search.sharded_index_manager import ShardedIndexManager
    from embeddings.embedder import EmbeddingResult
    from chunking.code_chunk import CodeChunk

    manager = ShardedIndexManager(str(tmp_path))
    manager.clear_index()

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

    manager.add_embeddings([emb], update_stats=False)
    manager.save_index()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    shard_id = manifest["shards"][0]["id"]
    shard_index = tmp_path / "shards" / shard_id / "code.index"
    assert shard_index.exists()
