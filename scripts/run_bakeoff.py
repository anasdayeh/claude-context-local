#!/usr/bin/env python3
"""Orchestrate the full Gate-C embedding bake-off.

Runs every ENABLED arm from arms.yaml SEQUENTIALLY — never two models resident at
once (16 GB-safe) — each in its correct venv, ordered light->heavy, SKIPPING any arm
whose output JSON already exists (resumable across restarts). Then runs the reranker
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
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# Light -> heavy, so early results land fast and the risky 8GB arm runs last.
ORDER = ["gemma", "qwen_mlx", "bge_code", "nomic_code_gguf", "qwen_bf16"]


def runner_cmd(arm: dict, cfg: dict) -> list[str]:
    paths = cfg["paths"]
    rt = arm["runtime"]
    label = arm["label"]
    out = str(Path(paths["out_dir"]) / f"arm_{label}.json")
    if rt == "torch":
        return ["uv", "run", "python", "scripts/bench_arm_torch.py", "--arm", label]
    if rt == "mlx":
        return [paths["mlx_venv"] + "/bin/python", "scripts/bench_arm_mlx.py",
                "--chunks", paths["chunk_dump"], "--queries", paths["queries"],
                "--out", out, "--model", arm["model_id"], "--label", label]
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
        plan.append((arm, out, out.exists() and not a.force))

    print("=== BAKE-OFF PLAN (sequential; one model at a time) ===")
    for arm, out, done in plan:
        print(f"  {arm['label']:16s} runtime={arm['runtime']:5s} "
              f"{'SKIP (exists)' if done else 'RUN':13s} -> {out.name}")
    rr = cfg.get("reranker", {}) or {}
    if rr.get("enabled") and not a.no_rerank:
        print(f"  {'reranker':16s} base={rr.get('base_arm')} model={rr.get('model_id')}")
    if not a.no_blind:
        print(f"  {'blind+render':16s} -> {paths['blind_dir']}/blind_report.md, report.md")
    if a.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    results = {}
    for arm, out, done in plan:
        if done:
            print(f"[skip] {arm['label']} (output exists)")
            continue
        cmd = runner_cmd(arm, cfg)
        print(f"\n[run ] {arm['label']} :: {' '.join(cmd)}", flush=True)
        t = time.time()
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        results[arm["label"]] = "ok" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[done] {arm['label']} -> {results[arm['label']]} in {time.time() - t:.0f}s", flush=True)

    if rr.get("enabled") and not a.no_rerank:
        base = rr.get("base_arm", "gemma")
        if (Path(paths["out_dir"]) / f"arm_{base}.json").exists():
            cmd = ["uv", "run", "python", "scripts/rerank_arm.py", "--base-arm", base]
            print(f"\n[run ] reranker :: {' '.join(cmd)}", flush=True)
            rc = subprocess.run(cmd, cwd=str(REPO)).returncode
            results["reranker"] = "ok" if rc == 0 else f"FAIL(rc={rc})"
        else:
            results["reranker"] = f"SKIP (no arm_{base}.json)"

    if not a.no_blind:
        jsons = sorted(str(p) for p in Path(paths["out_dir"]).glob("arm_*.json"))
        if len(jsons) >= 2:
            subprocess.run(["uv", "run", "python", "scripts/bench_blind.py", "make",
                            "--inputs", *jsons, "--out-dir", paths["blind_dir"], "--seed", "bridge"],
                           cwd=str(REPO))
            subprocess.run(["uv", "run", "python", "scripts/bench_model_ab.py", "render",
                            "--inputs", *jsons, "--out-md", str(Path(paths["blind_dir"]) / "report.md"),
                            "--out-json", str(Path(paths["blind_dir"]) / "report.json")],
                           cwd=str(REPO))
            results["blind+render"] = f"{len(jsons)} arms"
        else:
            results["blind+render"] = f"SKIP (only {len(jsons)} arm json)"

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:16s} {v}")
    print(f"\nblind report: {paths['blind_dir']}/blind_report.md")
    print(f"side-by-side: {paths['blind_dir']}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
