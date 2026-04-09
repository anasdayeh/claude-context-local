from embeddings.sentence_transformer import SentenceTransformerModel


def test_backend_can_be_selected_from_env(monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_EMBED_BACKEND", "onnx")
    model = SentenceTransformerModel("all-MiniLM-L6-v2", device="mps")
    assert model.backend == "onnx"
    assert model._effective_device_for_backend("onnx") == "cpu"


def test_torch_thread_limits_from_env(monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_EMBED_BACKEND", "torch")
    monkeypatch.setenv("CODE_SEARCH_TORCH_NUM_THREADS", "3")
    monkeypatch.setenv("CODE_SEARCH_TORCH_INTEROP_THREADS", "1")

    calls = {"num": None, "interop": None}

    monkeypatch.setattr("embeddings.sentence_transformer.torch.set_num_threads", lambda v: calls.__setitem__("num", v))
    monkeypatch.setattr(
        "embeddings.sentence_transformer.torch.set_num_interop_threads",
        lambda v: calls.__setitem__("interop", v),
    )

    model = SentenceTransformerModel("all-MiniLM-L6-v2", device="cpu")
    model._apply_torch_thread_limits()

    assert calls["num"] == 3
    assert calls["interop"] == 1
