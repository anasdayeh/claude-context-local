"""Resume state helpers for full-index checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResumeState:
    project_path: str
    project_id: str
    status: str
    files_total: int
    files_completed: int
    hashes: Dict[str, str] = field(default_factory=dict)
    completed: Set[str] = field(default_factory=set)
    last_updated: str = field(default_factory=_now_iso)
    version: int = 1
    mode: str = "full"

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "project_path": self.project_path,
            "project_id": self.project_id,
            "status": self.status,
            "last_updated": self.last_updated,
            "mode": self.mode,
            "files_total": self.files_total,
            "files_completed": self.files_completed,
            "hashes": dict(self.hashes),
            "completed": sorted(self.completed),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ResumeState":
        return cls(
            project_path=str(data.get("project_path", "")),
            project_id=str(data.get("project_id", "")),
            status=str(data.get("status", "in_progress")),
            last_updated=str(data.get("last_updated", _now_iso())),
            mode=str(data.get("mode", "full")),
            files_total=int(data.get("files_total", 0)),
            files_completed=int(data.get("files_completed", 0)),
            hashes=dict(data.get("hashes", {}) or {}),
            completed=set(data.get("completed", []) or []),
            version=int(data.get("version", 1)),
        )


def _resume_path(index_dir: Path) -> Path:
    return Path(index_dir) / "resume.json"


def load_resume_state(index_dir: Path) -> Optional[ResumeState]:
    path = _resume_path(index_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return ResumeState.from_dict(data)
    except Exception:
        return None


def save_resume_state(index_dir: Path, state: ResumeState) -> None:
    path = _resume_path(index_dir)
    state.last_updated = _now_iso()
    path.write_text(json.dumps(state.to_dict(), indent=2))


def clear_resume_state(index_dir: Path) -> None:
    path = _resume_path(index_dir)
    if path.exists():
        path.unlink()
