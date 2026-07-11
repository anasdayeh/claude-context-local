import logging

import numpy as np

from embeddings.sentence_transformer import SentenceTransformerModel
from embeddings.embedder import CodeEmbedder


class _FailingModel:
    device = "mps"

    def encode(self, _texts, **_kwargs):
        raise RuntimeError("MPS backend out of memory")


class _CpuModel:
    device = "cpu"

    def encode(self, texts, **_kwargs):
        return np.ones((len(texts), 3), dtype=np.float32)


def test_mps_failure_records_cpu_fallback_and_actual_device():
    model = SentenceTransformerModel.__new__(SentenceTransformerModel)
    model.model_name = "test/model"
    model.cache_dir = None
    model.trust_remote_code = False
    model._extra_model_kwargs = {}
    model.query_instruction = None
    model.backend = "torch"
    model._model_loaded = True
    model._fallback_attempted = False
    model._load_error = None
    model._logger = logging.getLogger("test-fallback")
    model._device = "mps"
    model._requested_device = "mps"
    model._fallback_events = []
    model.__dict__["model"] = _FailingModel()
    model._load_model = lambda: _CpuModel()

    vectors = model.encode(["one", "two"])
    info = model.get_execution_info()

    assert vectors.shape == (2, 3)
    assert info["requested_device"] == "mps"
    assert info["actual_device"] == "cpu"
    assert info["degraded"] is True
    assert info["fallback_events"] == [
        {
            "from": "mps",
            "to": "cpu",
            "reason": "mps_oom",
        }
    ]


def test_cpu_retry_failure_is_not_swallowed():
    model = SentenceTransformerModel.__new__(SentenceTransformerModel)
    model.model_name = "test/model"
    model.cache_dir = None
    model.trust_remote_code = False
    model._extra_model_kwargs = {}
    model.query_instruction = None
    model.backend = "torch"
    model._model_loaded = True
    model._fallback_attempted = False
    model._load_error = None
    model._logger = logging.getLogger("test-fallback")
    model._device = "mps"
    model._requested_device = "mps"
    model._fallback_events = []
    model.__dict__["model"] = _FailingModel()
    model._load_model = lambda: _FailingModel()

    try:
        model.encode(["one"])
    except RuntimeError as exc:
        assert "out of memory" in str(exc).lower()
    else:
        raise AssertionError("CPU retry failure was swallowed")


def test_code_embedder_health_propagates_execution_provenance():
    class _Model:
        def get_model_info(self):
            return {
                "status": "loaded",
                "backend": "torch",
                "device": "cpu",
                "requested_device": "mps",
                "actual_device": "cpu",
                "fallback_events": [{"from": "mps", "to": "cpu", "reason": "mps_oom"}],
                "degraded": True,
            }

    embedder = CodeEmbedder.__new__(CodeEmbedder)
    embedder.model_name = "test/model"
    embedder._model = _Model()
    embedder._status = "ready"
    embedder._last_error = None

    status = embedder.health_status()

    assert status["requested_device"] == "mps"
    assert status["actual_device"] == "cpu"
    assert status["degraded"] is True
    assert status["fallback_events"][0]["reason"] == "mps_oom"


def test_onnx_load_fallback_is_recorded(monkeypatch):
    calls = []
    torch_model = object()

    def fake_sentence_transformer(*_args, **kwargs):
        calls.append(kwargs.get("backend", "torch"))
        if kwargs.get("backend") == "onnx":
            raise RuntimeError("onnx provider failed")
        return torch_model

    monkeypatch.setattr(
        "embeddings.sentence_transformer.SentenceTransformer", fake_sentence_transformer
    )
    model = SentenceTransformerModel.__new__(SentenceTransformerModel)
    model.model_name = "test/model"
    model.cache_dir = None
    model.trust_remote_code = False
    model._extra_model_kwargs = {}
    model._logger = logging.getLogger("test-onnx-fallback")
    model._device = "cpu"
    model._requested_device = "cpu"
    model._requested_backend = "onnx"
    model.backend = "onnx"
    model._model_loaded = False
    model._fallback_events = []
    model._is_model_cached = lambda: False
    model._apply_torch_thread_limits = lambda: None

    loaded = model._load_model()

    assert loaded is torch_model
    assert calls == ["onnx", "torch"]
    assert model.get_execution_info()["fallback_events"] == [
        {"from": "onnx", "to": "torch", "reason": "onnx_error"}
    ]
