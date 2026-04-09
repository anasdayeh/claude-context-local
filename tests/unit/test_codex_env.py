from pathlib import Path

def test_codex_env_contains_perf_flags():
    text = Path("/Users/anasdayeh/.codex/config.toml").read_text()
    assert "CODE_SEARCH_DEVICE" in text
    assert "CODE_SEARCH_INDEX_WORKERS" in text
