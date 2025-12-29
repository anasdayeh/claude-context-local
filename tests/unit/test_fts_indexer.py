import numpy as np
from pathlib import Path

from embeddings.embedder import EmbeddingResult
from search.indexer import CodeIndexManager


def test_fts_create_and_search(tmp_path: Path):
    mgr = CodeIndexManager(str(tmp_path))
    emb = np.ones(4, dtype=np.float32)
    result = EmbeddingResult(
        chunk_id="abc",
        embedding=emb,
        metadata={
            "relative_path": "src/foo.py",
            "content": "def create_remote_pcap():\n    pass",
        },
    )

    mgr.fts_upsert(
        result.chunk_id,
        result.metadata["relative_path"],
        result.metadata["content"],
    )
    hits = mgr.fts_search("create_remote_pcap", k=5)
    assert any(cid == "abc" for cid, _score in hits)
