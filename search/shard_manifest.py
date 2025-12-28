"""Shard manifest persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ShardManifest:
    version: int
    project_path: str
    embedding_dimension: int
    index_type: str
    shard_count: int
    shards: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.shard_count != len(self.shards):
            self.shard_count = len(self.shards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "project_path": self.project_path,
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "shard_count": self.shard_count,
            "shards": self.shards,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "ShardManifest":
        payload = json.loads(Path(path).read_text())
        return cls(
            version=int(payload.get("version", 1)),
            project_path=payload.get("project_path", ""),
            embedding_dimension=int(payload.get("embedding_dimension", 0)),
            index_type=payload.get("index_type", "flat"),
            shard_count=int(payload.get("shard_count", 0)),
            shards=list(payload.get("shards", [])),
        )
