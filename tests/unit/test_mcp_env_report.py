import subprocess
import sys
from pathlib import Path


def test_env_report_runs():
    script = Path(__file__).parents[2] / "scripts" / "mcp_env_report.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "PYTHON" in result.stdout
