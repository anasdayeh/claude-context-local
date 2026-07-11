"""Fingerprint and atomically persist benchmark artifacts."""
from __future__ import annotations

import hashlib
import fnmatch
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2
_IGNORED_TREE_PARTS = {".git", ".venv", "node_modules", "__pycache__"}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_tree_files(
    root: str | Path, patterns: Iterable[str] | None = None
) -> list[tuple[str, Path]]:
    """Return deterministic, noise-filtered files for direct comparisons."""
    root = Path(root).resolve()
    selected = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in _IGNORED_TREE_PARTS for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        if patterns and not any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern)
                                for pattern in patterns):
            continue
        selected.append((relative, path))
    return sorted(selected)


def hash_directory(root: str | Path, patterns: Iterable[str] | None = None) -> str:
    """Content-address a repository tree for direct index/search comparisons."""
    digest = hashlib.sha256()
    for relative, path in selected_tree_files(root, patterns=patterns):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_run_fingerprint(
    *,
    chunk_path: str | Path,
    query_path: str | Path,
    config_path: str | Path,
    arm: dict[str, Any],
    source_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Build a stable identity for every input that can change run semantics."""
    sources = {
        str(Path(path).name): _sha256_file(path)
        for path in sorted((Path(path) for path in source_paths), key=lambda p: str(p))
    }
    arm_json = json.dumps(arm, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": _sha256_file(chunk_path),
        "queries_sha256": _sha256_file(query_path),
        "config_sha256": _sha256_file(config_path),
        "arm_sha256": hashlib.sha256(arm_json.encode("utf-8")).hexdigest(),
        "sources": sources,
    }


def build_direct_run_fingerprint(
    *,
    repo_path: str | Path,
    query_path: str | Path,
    arm: dict[str, Any],
    patterns: Iterable[str] | None = None,
    source_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Fingerprint the direct index/search runner without a prebuilt chunk dump."""
    sources = {
        str(Path(path).name): _sha256_file(path)
        for path in sorted((Path(path) for path in source_paths), key=lambda p: str(p))
    }
    arm_json = json.dumps(arm, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": hash_directory(repo_path, patterns=patterns),
        "queries_sha256": _sha256_file(query_path),
        "config_sha256": None,
        "arm_sha256": hashlib.sha256(arm_json.encode("utf-8")).hexdigest(),
        "sources": sources,
    }


def artifact_is_reusable(
    path: str | Path, expected_fingerprint: dict[str, Any]
) -> tuple[bool, str]:
    path = Path(path)
    if not path.exists():
        return False, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False, "invalid_json"
    artifact = payload.get("artifact") if isinstance(payload, dict) else None
    if not isinstance(artifact, dict) or artifact.get("status") != "complete":
        return False, "incomplete"
    if artifact.get("schema_version") != SCHEMA_VERSION:
        return False, "schema_mismatch"
    if artifact.get("fingerprint") != expected_fingerprint:
        return False, "fingerprint_mismatch"
    if not isinstance(payload.get("summary"), dict) or not isinstance(payload.get("per_query"), list):
        return False, "incomplete"
    return True, "match"


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON durably and replace the destination in one filesystem operation."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, value: str) -> None:
    """Write a text report durably and replace its destination atomically."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
