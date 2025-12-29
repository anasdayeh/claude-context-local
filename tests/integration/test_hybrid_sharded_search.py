import numpy as np

from embeddings.embedder import EmbeddingResult
from search.sharded_index_manager import ShardedIndexManager


def test_hybrid_search_across_shards(tmp_path):
    mgr = ShardedIndexManager(str(tmp_path))
    emb = np.ones(4, dtype=np.float32)

    mgr.add_embeddings([
        EmbeddingResult(
            chunk_id="c1",
            embedding=emb,
            metadata={"relative_path": "a.py", "content": "alpha token"},
        )
    ])
    mgr._create_new_shard()
    mgr.add_embeddings([
        EmbeddingResult(
            chunk_id="c2",
            embedding=emb,
            metadata={"relative_path": "b.py", "content": "beta token"},
        )
    ])

    results = mgr.search_hybrid("beta", np.ones(4, dtype=np.float32), k=5)
    assert any(cid == "c2" for cid, _score, _meta in results)
