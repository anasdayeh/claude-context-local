import json
import numpy as np


def test_repair_manifest_from_shards(tmp_path):
    from search.sharded_index_manager import ShardedIndexManager
    from embeddings.embedder import EmbeddingResult
    from chunking.code_chunk import CodeChunk

    manager = ShardedIndexManager(str(tmp_path))

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
    manager.add_embeddings([emb], update_stats=True)
    manager.save_index()

    # Corrupt manifest to simulate empty shard list.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "project_path": "",
                "embedding_dimension": 0,
                "index_type": "flat",
                "shard_count": 0,
                "shards": [],
            }
        )
    )

    repaired = ShardedIndexManager(str(tmp_path))
    result = repaired.repair_manifest_from_shards()

    assert result.get("repaired") is True
    assert result.get("shard_count", 0) >= 1

    manifest = json.loads(manifest_path.read_text())
    assert manifest["shard_count"] >= 1
    assert manifest["shards"], "Manifest should include shard entries"
