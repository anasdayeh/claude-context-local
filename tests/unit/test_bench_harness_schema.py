import json
import os
import subprocess
import sys
from pathlib import Path


def test_bench_harness_schema(tmp_path):
    out = tmp_path / "bench.json"
    cmd = [sys.executable, "scripts/bench_mcp_perf.py", "--dry-run", f"--out={out}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    data = json.loads(out.read_text())
    assert "embedding_throughput_chunks_per_sec" in data
    assert "indexing_time_seconds" in data
    assert "search_latency_ms" in data


def test_bench_harness_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "bench.json"
    cmd = [sys.executable, "scripts/bench_mcp_perf.py", "--dry-run", f"--out={out}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert out.exists()


def test_bench_harness_non_dry_run_numeric_values(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "util.py").write_text("class Greeter:\n    def hello(self):\n        return 'hi'\n")

    storage_dir = tmp_path / "storage"
    out = tmp_path / "bench.json"
    env = os.environ.copy()
    env["CODE_SEARCH_STORAGE"] = str(storage_dir)
    env["PYTEST_USE_MOCKS"] = "1"
    cmd = [
        sys.executable,
        "scripts/bench_mcp_perf.py",
        f"--out={out}",
        f"--repo={repo}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert isinstance(data["embedding_throughput_chunks_per_sec"], (int, float))
    assert isinstance(data["indexing_time_seconds"], (int, float))
    assert isinstance(data["search_latency_ms"]["p50"], (int, float))
    assert isinstance(data["search_latency_ms"]["p95"], (int, float))
