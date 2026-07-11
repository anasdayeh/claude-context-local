#!/usr/bin/env python3
"""Orchestrate the full Gate-C embedding bake-off.

Runs every ENABLED arm from arms.yaml SEQUENTIALLY — never two models resident at
once (16 GB-safe) — each in its correct venv, ordered light->heavy, SKIPPING only an
arm whose complete schema-v2 fingerprint matches (resumable across restarts). Then runs the reranker
arm and builds the blind judge report + side-by-side render. Model loads happen only
inside the per-arm subprocesses; this orchestrator loads nothing heavy itself.

Usage:
  uv run python scripts/run_bakeoff.py --dry-run          # show plan, run nothing
  uv run python scripts/run_bakeoff.py                     # run all pending arms + rerank + blind
  uv run python scripts/run_bakeoff.py --only qwen_mlx bge_code
  uv run python scripts/run_bakeoff.py --force             # re-run even if output exists
  uv run python scripts/run_bakeoff.py --skip qwen_bf16    # defer the heaviest arm
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402
import bench_artifacts  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# Light -> heavy, so early results land fast and the risky 8GB arm runs last.
ORDER = ["gemma", "qwen_mlx", "bge_code", "nomic_code_gguf", "qwen_bf16"]


class BakeoffLock:
    """Non-blocking advisory lock preventing concurrent model workloads."""

    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.seek(0)
            owner = self._handle.read().strip() or "unknown owner"
            self._handle.close()
            self._handle = None
            raise RuntimeError(f"benchmark already running ({owner})") from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(json.dumps({"run_id": self.run_id, "pid": os.getpid()}))
        self._handle.flush()
        return self

    def __exit__(self, *_exc):
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def aggregate_exit_status(results: dict[str, str]) -> int:
    return 1 if any(str(value).startswith("FAIL") for value in results.values()) else 0


def wait_for_memory_handoff(
    *,
    min_available_bytes: int,
    stable_samples: int = 3,
    min_delay_seconds: float = 15,
    poll_seconds: float = 5,
    timeout_seconds: float = 180,
    read_available=None,
    sleep=time.sleep,
):
    """Wait for unified memory to settle before loading the next Metal model."""
    if read_available is None:
        import psutil
        read_available = lambda: int(psutil.virtual_memory().available)
    if min_delay_seconds > 0:
        sleep(min_delay_seconds)
    deadline = time.monotonic() + timeout_seconds
    stable = 0
    available = 0
    while time.monotonic() <= deadline:
        available = int(read_available())
        stable = stable + 1 if available >= min_available_bytes else 0
        if stable >= max(1, stable_samples):
            return {"stable": True, "available_gb": round(available / 1024**3, 2)}
        sleep(poll_seconds)
    return {"stable": False, "available_gb": round(available / 1024**3, 2)}


def run_command(cmd, *, cwd):
    """Run one owned stage and terminate only its process group on interruption."""
    process = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def runner_cmd(arm: dict, cfg: dict) -> list[str]:
    paths = cfg["paths"]
    rt = arm["runtime"]
    label = arm["label"]
    out = str(Path(paths["out_dir"]) / f"arm_{label}.json")
    if rt == "torch":
        return ["uv", "run", "python", "scripts/bench_arm_torch.py", "--arm", label]
    if rt == "mlx":
        defaults = cfg.get("defaults", {}) or {}
        reranker = cfg.get("reranker", {}) or {}
        k = int(defaults.get("k", 5))
        candidate_k = max(k, 10, int(reranker.get("top_n", k)) if reranker.get("enabled") else k)
        return [paths["mlx_venv"] + "/bin/python", "scripts/bench_arm_mlx.py",
                "--chunks", paths["chunk_dump"], "--queries", paths["queries"],
                "--out", out, "--model", arm["model_id"], "--label", label,
                "--candidate-k", str(candidate_k)]
    if rt == "gguf":
        return [paths["gguf_venv"] + "/bin/python", "scripts/bench_arm_gguf.py", "--arm", label]
    raise SystemExit(f"unknown runtime {rt!r} for arm {label}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--skip", nargs="+", default=None)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-blind", action="store_true")
    a = ap.parse_args()

    cfg = bc.load_config(a.config)
    paths = cfg["paths"]
    Path(paths["out_dir"]).mkdir(parents=True, exist_ok=True)

    arms = [x for x in cfg.get("arms", []) if x.get("enabled", True)]
    arms.sort(key=lambda x: ORDER.index(x["label"]) if x["label"] in ORDER else 99)
    if a.only:
        arms = [x for x in arms if x["label"] in a.only]
    if a.skip:
        arms = [x for x in arms if x["label"] not in a.skip]

    plan = []
    for arm in arms:
        out = Path(paths["out_dir"]) / f"arm_{arm['label']}.json"
        expected = bc.run_fingerprint(cfg, arm)
        reusable, reason = bench_artifacts.artifact_is_reusable(out, expected)
        done = reusable and not a.force
        plan.append((arm, out, done, "forced" if a.force else reason))

    print("=== BAKE-OFF PLAN (sequential; one model at a time) ===")
    for arm, out, done, reason in plan:
        print(f"  {arm['label']:16s} runtime={arm['runtime']:5s} "
              f"{'SKIP (valid)' if done else 'RUN':13s} -> {out.name} [{reason}]")
    rr = cfg.get("reranker", {}) or {}
    if rr.get("enabled") and not a.no_rerank:
        print(f"  {'reranker':16s} base={rr.get('base_arm')} model={rr.get('model_id')}")
    if not a.no_blind:
        print(f"  {'blind+render':16s} -> {paths['blind_dir']}/blind_report.md, report.md")
    if a.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    run_id = uuid.uuid4().hex
    try:
        lock = BakeoffLock(Path(paths["out_dir"]) / ".bakeoff.lock", run_id)
        lock.__enter__()
    except RuntimeError as exc:
        print(f"[lock] {exc}", file=sys.stderr)
        return 2

    results = {}
    state_path = Path(paths["out_dir"]) / "run_state.json"

    def record(stage, value):
        results[stage] = value
        bench_artifacts.atomic_write_json(
            state_path,
            {"run_id": run_id, "pid": os.getpid(), "results": results, "updated_at": time.time()},
        )

    try:
        resource_ready = True
        for position, (arm, out, done, reason) in enumerate(plan):
            if done:
                print(f"[skip] {arm['label']} (fingerprint match)")
                record(arm["label"], "reused")
                continue
            cmd = runner_cmd(arm, cfg)
            print(f"\n[run ] {arm['label']} :: {' '.join(cmd)}", flush=True)
            t = time.time()
            rc = run_command(cmd, cwd=str(REPO))
            record(arm["label"], "ok" if rc == 0 else f"FAIL(rc={rc})")
            print(f"[done] {arm['label']} -> {results[arm['label']]} in {time.time() - t:.0f}s", flush=True)

            future_model = any(not item[2] for item in plan[position + 1:])
            needs_handoff = future_model or (rr.get("enabled") and not a.no_rerank)
            if needs_handoff:
                defaults = cfg.get("defaults", {}) or {}
                handoff = wait_for_memory_handoff(
                    min_available_bytes=int(float(defaults.get("handoff_min_free_gb", 4)) * 1024**3),
                    stable_samples=int(defaults.get("handoff_stable_samples", 3)),
                    min_delay_seconds=float(defaults.get("handoff_min_delay_seconds", 15)),
                    poll_seconds=float(defaults.get("handoff_poll_seconds", 5)),
                    timeout_seconds=float(defaults.get("handoff_timeout_seconds", 180)),
                )
                print(f"[handoff] after {arm['label']}: {handoff}", flush=True)
                if not handoff["stable"]:
                    record(f"handoff_after_{arm['label']}", "FAIL(memory did not recover)")
                    resource_ready = False
                    break

        if rr.get("enabled") and not a.no_rerank and resource_ready:
            base = rr.get("base_arm", "gemma")
            try:
                base_arm = bc.get_arm(cfg, base)
            except KeyError:
                base_arm = None
            base_path = Path(paths["out_dir"]) / f"arm_{base}.json"
            base_valid = False
            if base_arm is not None:
                base_valid, _ = bench_artifacts.artifact_is_reusable(
                    base_path, bc.run_fingerprint(cfg, base_arm)
                )
            if base_valid:
                cmd = ["uv", "run", "python", "scripts/rerank_arm.py", "--base-arm", base]
                print(f"\n[run ] reranker :: {' '.join(cmd)}", flush=True)
                rc = run_command(cmd, cwd=str(REPO))
                record("reranker", "ok" if rc == 0 else f"FAIL(rc={rc})")
            else:
                record("reranker", f"FAIL(no valid arm_{base}.json)")
        elif rr.get("enabled") and not a.no_rerank and not resource_ready:
            record("reranker", "FAIL(resource handoff)")

        if not a.no_blind:
            jsons = []
            for arm in arms:
                path = Path(paths["out_dir"]) / f"arm_{arm['label']}.json"
                reusable, _ = bench_artifacts.artifact_is_reusable(
                    path, bc.run_fingerprint(cfg, arm)
                )
                if reusable:
                    jsons.append(str(path))
            rerank_path = Path(paths["out_dir"]) / f"arm_{rr.get('base_arm', 'gemma')}_rerank.json"
            if results.get("reranker") == "ok" and rerank_path.exists():
                jsons.append(str(rerank_path))
            jsons.sort()
            if len(jsons) >= 2:
                make_rc = run_command(
                    ["uv", "run", "python", "scripts/bench_blind.py", "make",
                     "--inputs", *jsons, "--out-dir", paths["blind_dir"], "--seed", "bridge"],
                    cwd=str(REPO),
                )
                render_rc = run_command(
                    ["uv", "run", "python", "scripts/bench_model_ab.py", "render",
                     "--inputs", *jsons, "--out-md", str(Path(paths["blind_dir"]) / "report.md"),
                     "--out-json", str(Path(paths["blind_dir"]) / "report.json")],
                    cwd=str(REPO),
                )
                value = f"{len(jsons)} arms" if make_rc == render_rc == 0 else f"FAIL(make={make_rc},render={render_rc})"
                record("blind+render", value)
            else:
                record("blind+render", f"FAIL(only {len(jsons)} arm json)")

        print("\n=== SUMMARY ===")
        for key, value in results.items():
            print(f"  {key:16s} {value}")
        print(f"\nblind report: {paths['blind_dir']}/blind_report.md")
        print(f"side-by-side: {paths['blind_dir']}/report.md")
        return aggregate_exit_status(results)
    finally:
        lock.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
