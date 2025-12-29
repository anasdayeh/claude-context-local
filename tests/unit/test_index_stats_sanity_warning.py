import numpy as np


def test_index_stats_warn_when_metadata_without_vectors(tmp_path):
    from search.indexer import CodeIndexManager

    index_dir = tmp_path / "index"
    manager = CodeIndexManager(str(index_dir))

    # Inject metadata without creating any vectors.
    manager.metadata_db["1"] = {
        "chunk_id": "abc",
        "metadata": {
            "relative_path": "a.py",
            "file_path": str(tmp_path / "a.py"),
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 2,
            "content_preview": "def f(): pass",
        },
    }
    manager.metadata_db.commit()

    stats = manager.get_stats()
    assert stats.get("total_chunks") == 0
    assert "sanity_warning" in stats
    assert "sanity_suggestion" in stats
