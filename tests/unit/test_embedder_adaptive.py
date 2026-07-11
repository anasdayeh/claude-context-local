import logging
import numpy as np

from chunking.code_chunk import CodeChunk
from embeddings.embedder import CodeEmbedder


def _make_chunks(n):
    chunks = []
    for i in range(n):
        chunks.append(
            CodeChunk(
                content=f"def f{i}(): pass",
                chunk_type="function",
                start_line=1,
                end_line=1,
                file_path=f"/tmp/f{i}.py",
                relative_path=f"f{i}.py",
                folder_structure=[],
                name=f"f{i}",
            )
        )
    return chunks


def test_embedder_backoff_on_oom(monkeypatch):
    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder.model_name = "test"
    embedder._logger = logging.getLogger("test")

    def fake_encode(texts):
        if len(texts) > 1:
            raise RuntimeError("MPS backend out of memory")
        return np.ones((len(texts), 3), dtype=np.float32)

    embedder._encode_documents = fake_encode
    embedder._clear_device_cache = lambda: None

    results = embedder.embed_chunks(_make_chunks(3), batch_size=4)
    assert len(results) == 3


def test_progress_callback_receives_backoff_message(monkeypatch):
    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder.model_name = "test"
    embedder._logger = logging.getLogger("test")
    messages = []

    def fake_encode(texts):
        if len(texts) > 1:
            raise RuntimeError("out of memory")
        return np.ones((len(texts), 3), dtype=np.float32)

    embedder._encode_documents = fake_encode
    embedder._clear_device_cache = lambda: None
    embedder._progress_callback = messages.append

    embedder.embed_chunks(_make_chunks(2), batch_size=4)

    assert any("batch" in msg.lower() for msg in messages)


def test_embedder_reduces_batch_on_memory_pressure(monkeypatch):
    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder.model_name = "test"
    embedder._logger = logging.getLogger("test")
    embedder._clear_device_cache = lambda: None
    embedder._min_free_ram_bytes = 2 * 1024 ** 3
    messages = []
    embedder._progress_callback = messages.append

    def fake_available_memory():
        return 256 * 1024 ** 2

    monkeypatch.setattr("embeddings.embedder.get_available_memory_bytes", fake_available_memory)

    def fake_encode(texts):
        return np.ones((len(texts), 3), dtype=np.float32)

    embedder._encode_documents = fake_encode

    results = embedder.embed_chunks(_make_chunks(3), batch_size=4)

    assert len(results) == 3
    assert any("memory-pressure backoff" in msg.lower() for msg in messages)


def test_cpu_oom_does_not_loop_through_cpu_fallback_again():
    calls = []

    class CpuModel:
        _device = "cpu"
        _fallback_attempted = True

        def _reset_model(self):
            calls.append("reset")

    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder._model = CpuModel()

    assert embedder._force_cpu_fallback() is False
    assert calls == []


def test_cpu_fallback_requires_a_real_device_transition():
    calls = []

    class DeviceLessModel:
        def _reset_model(self):
            calls.append("reset")

    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder._model = DeviceLessModel()

    assert embedder._force_cpu_fallback() is False
    assert calls == []
