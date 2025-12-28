import subprocess
import sys
from pathlib import Path


def test_index_repo_cli_help():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "index_repo.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
