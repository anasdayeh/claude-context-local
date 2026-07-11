import os
from pathlib import Path

import pytest

from scripts.rerank_arm import load_cross_encoder


@pytest.mark.integration
def test_official_qwen_reranker_orders_obvious_relevance():
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    cached = hf_home / "hub" / "models--Qwen--Qwen3-Reranker-0.6B"
    if not cached.exists():
        pytest.skip("official Qwen reranker is not cached")

    model = load_cross_encoder("Qwen/Qwen3-Reranker-0.6B", "cpu", 512)
    scores = model.predict(
        [
            ("Where is the Python authentication function?", "def authenticate_user(token): return verify(token)"),
            ("Where is the Python authentication function?", "The garden contains three red flowers."),
        ],
        batch_size=1,
        show_progress_bar=False,
    )

    assert float(scores[0]) > float(scores[1])

