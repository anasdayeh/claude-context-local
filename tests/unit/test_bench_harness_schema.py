import json
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
