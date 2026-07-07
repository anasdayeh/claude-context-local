"""Shard manifest persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


class IndexModelMismatchError(RuntimeError):
    """Raised when an index is opened by an embedder it was not built with —
    a different embedding model or dimension means an incompatible vector space."""


@dataclass
class ShardManifest:
    version: int
    project_path: str
    embedding_dimension: int
    index_type: str
    shard_count: int
    shards: List[Dict[str, Any]] = field(default_factory=list)
    embedding_model: str = ""

    def __post_init__(self) -> None:
        if self.shard_count != len(self.shards):
            self.shard_count = len(self.shards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "project_path": self.project_path,
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_model,
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
            embedding_model=payload.get("embedding_model", ""),
        )

    def assert_compatible(self, embedding_model: str, embedding_dimension: int) -> None:
        """Raise IndexModelMismatchError if this index cannot be served by the given
        embedder. Dimension mismatch is the hard incompatibility; a model-name
        mismatch is caught too when both names are known. Unknown (falsy) values on
        either side are skipped, so legacy manifests (no model) and cheap name-only
        checks (pass dimension=0) both work.
        """
        if (self.embedding_dimension and embedding_dimension
                and self.embedding_dimension != embedding_dimension):
            raise IndexModelMismatchError(
                f"Index vector dimension {self.embedding_dimension} != active embedder "
                f"dimension {embedding_dimension} (index model="
                f"{self.embedding_model or 'unknown'}, embedder={embedding_model}). "
                f"Use a separate CODE_SEARCH_STORAGE root per model, or rebuild the index."
            )
        if (self.embedding_model and embedding_model
                and self.embedding_model != embedding_model):
            raise IndexModelMismatchError(
                f"Index was built by embedding model {self.embedding_model!r} but the active "
                f"embedder is {embedding_model!r}. Point CODE_SEARCH_STORAGE at that model's "
                f"store, or rebuild the index."
            )
