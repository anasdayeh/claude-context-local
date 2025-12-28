import logging
from pathlib import Path


def test_resume_logging_and_progress(tmp_path, caplog):
    from merkle.merkle_dag import MerkleDAG
    from search.indexer import CodeIndexManager
    from search.incremental_indexer import IncrementalIndexer
    from search.resume_state import ResumeState, save_resume_state
    from chunking.multi_language_chunker import MultiLanguageChunker
    import numpy as np
    from embeddings.embedder import EmbeddingResult

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "a.py").write_text("def a():\n    return 1\n")
    (project_dir / "b.py").write_text("def b():\n    return 2\n")

    dag = MerkleDAG(str(project_dir))
    dag.build()
    hashes = dag.get_file_hashes()

    storage_dir = tmp_path / "storage"
    index_dir = storage_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    state = ResumeState(
        project_path=str(project_dir),
        project_id="test",
        status="in_progress",
        files_total=2,
        files_completed=1,
        hashes={"a.py": hashes.get("a.py", "")},
        completed={"a.py"},
    )
    save_resume_state(index_dir, state)

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

    class TestChunker(MultiLanguageChunker):
        def chunk_file(self, file_path: str):
            return super().chunk_file(file_path)

    index_manager = CodeIndexManager(str(index_dir))
    chunker = TestChunker(root_path=str(project_dir))
    indexer = IncrementalIndexer(index_manager, DummyEmbedder(), chunker, str(storage_dir))
    indexer._checkpoint_interval = 1

    caplog.set_level(logging.INFO, logger="search.incremental_indexer")

    result = indexer.incremental_index(
        str(project_dir),
        "project",
        force_full=True,
        resume=True,
    )

    assert result.success
    messages = "\n".join(rec.message for rec in caplog.records)
    assert "Resume active" in messages
    assert "Progress:" in messages
