from pathlib import Path


def test_progress_callback_rate_limited(tmp_path):
    import numpy as np
    from embeddings.embedder import EmbeddingResult
    from search.indexer import CodeIndexManager
    from search.incremental_indexer import IncrementalIndexer
    from chunking.multi_language_chunker import MultiLanguageChunker

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "a.py").write_text("def a():\n    return 1\n")
    (project_dir / "b.py").write_text("def b():\n    return 2\n")
    (project_dir / "c.py").write_text("def c():\n    return 3\n")

    storage_dir = tmp_path / "storage"
    index_dir = storage_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    class DummyEmbedder:
        def embed_chunks(self, chunks, batch_size=32):
            results = []
            for chunk in chunks:
                results.append(
                    EmbeddingResult(
                        chunk=chunk,
                        embedding=np.ones(4, dtype=np.float32),
                        model_name="dummy",
                    )
                )
            return results

    chunker = MultiLanguageChunker(root_path=str(project_dir))
    index_manager = CodeIndexManager(str(index_dir))
    indexer = IncrementalIndexer(index_manager, DummyEmbedder(), chunker, str(storage_dir))
    indexer._checkpoint_interval = 10_000
    indexer._progress_every_files = 2

    messages = []

    result = indexer.incremental_index(
        str(project_dir),
        "project",
        force_full=True,
        progress_callback=messages.append,
    )

    assert result.success
    progress_msgs = [msg for msg in messages if "progress" in msg.lower()]
    assert len(progress_msgs) >= 1
