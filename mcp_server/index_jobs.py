"""Background indexing job management.

This module exists to avoid long-running MCP tool calls. Codex tool calls have a hard
deadline (often ~1800s). Indexing large repos must therefore run as a background job
that can be polled for status/progress.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional


@dataclass
class IndexJobEvent:
    ts: float
    message: str


@dataclass
class IndexJob:
    job_id: str
    project_path: str
    project_name: str
    file_patterns: Optional[list[str]]
    incremental: bool
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: str = "queued"  # queued|running|completed|failed|canceled
    last_message: str = ""
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    events: Deque[IndexJobEvent] = field(default_factory=lambda: deque(maxlen=200))
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def add_event(self, message: str) -> None:
        self.last_message = message
        self.events.append(IndexJobEvent(ts=time.time(), message=message))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "project_path": self.project_path,
            "project_name": self.project_name,
            "file_patterns": self.file_patterns,
            "incremental": self.incremental,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "last_message": self.last_message,
            "error": self.error,
            "result": self.result,
        }


class IndexJobManager:
    """In-process job registry. Thread-safe.

    The manager is intentionally small; it does not try to persist jobs across process
    restarts. It exists purely to keep MCP tool calls fast.
    """

    def __init__(self, event_buffer_size: int = 200) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, IndexJob] = {}
        self._event_buffer_size = max(10, int(event_buffer_size or 200))

    def create_job(
        self,
        *,
        project_path: str,
        project_name: str,
        file_patterns: Optional[list[str]],
        incremental: bool,
    ) -> IndexJob:
        job = IndexJob(
            job_id=uuid.uuid4().hex,
            project_path=project_path,
            project_name=project_name,
            file_patterns=file_patterns,
            incremental=incremental,
            events=deque(maxlen=self._event_buffer_size),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[IndexJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def find_active_job_for_path(self, project_path: str) -> Optional[IndexJob]:
        with self._lock:
            for job in self._jobs.values():
                if job.project_path == project_path and job.status in {"queued", "running"}:
                    return job
        return None

    def list_jobs(self) -> list[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        # Newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs]

    def cancel(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if job is None:
            return {"success": False, "error": f"Unknown job_id: {job_id}"}
        job.cancel_event.set()
        job.add_event("cancellation requested")
        return {"success": True, "job": job.to_dict()}

