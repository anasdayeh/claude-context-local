"""Shared loader for MCP tool descriptions and help text."""

from pathlib import Path
import yaml


def load_strings() -> dict:
    """Load tool descriptions and help text from strings.yaml."""
    strings_file = Path(__file__).parent / "strings.yaml"
    with open(strings_file, "r") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("strings.yaml must parse to a dict")
        return {
            "tools": data.get("tools", {}),
            "help": data.get("help", ""),
        }
