from pathlib import Path


def test_resume_prune_saves_before_processing(tmp_path, monkeypatch):
    from merkle.merkle_dag import MerkleDAG
    from search.indexer import CodeIndexManager
    from search.incremental_indexer import IncrementalIndexer
    from search.resume_state import ResumeState, save_resume_state as real_save
    from chunking.multi_language_chunker import MultiLanguageChunker
    import numpy as np
    from embeddings.embedder import EmbeddingResult

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "a.py").write_text("def a():\n    return 1\n")

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
        files_total=1,
        files_completed=1,
        hashes={"missing.py": "old"},
        completed={"missing.py"},
    )
    real_save(index_dir, state)

    events = []

    def _save_wrapper(index_dir_path, resume_state):
        events.append("save")
        return real_save(index_dir_path, resume_state)

    monkeypatch.setattr("search.incremental_indexer.save_resume_state", _save_wrapper)

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

    class GuardedChunker(MultiLanguageChunker):
        def chunk_file(self, file_path: str):
            events.append("chunk")
            return super().chunk_file(file_path)

    index_manager = CodeIndexManager(str(index_dir))
    chunker = GuardedChunker(root_path=str(project_dir))
    indexer = IncrementalIndexer(index_manager, DummyEmbedder(), chunker, str(storage_dir))
    indexer._checkpoint_interval = 10_000

    result = indexer.incremental_index(
        str(project_dir),
        "project",
        force_full=True,
        resume=True,
    )

    assert result.success
    assert "chunk" in events
    assert events[0] == "save"
