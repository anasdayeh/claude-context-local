"""Shared, torch-free backbone for the A/B arm runners.

Provides: config loading (arms.yaml with ${var} interpolation + env overrides +
fallbacks), logging setup (console + per-run file), and query/chunk loaders.
Depends only on stdlib + pyyaml + numpy, so it imports cleanly in the torch
server venv AND the isolated MLX / GGUF venvs. Keeping ALL paths and per-arm
config here is what lets us try different models by editing arms.yaml alone.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

try:
    import yaml  # pyyaml
except Exception:  # pragma: no cover - guidance if a side venv lacks pyyaml
    yaml = None

_VAR = re.compile(r"\$\{([^}]+)\}")
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "benchmarks" / "arms.yaml"


def _interp(value, ctx):
    """Resolve ${name} against ctx then environment; leave unknown refs intact.
    Iterates so nested refs (a path built from another path) fully resolve."""
    if not isinstance(value, str):
        return value

    def repl(m: "re.Match[str]") -> str:
        key = m.group(1)
        if key in ctx:
            return str(ctx[key])
        return os.environ.get(key, m.group(0))

    prev = None
    while prev != value:
        prev = value
        value = _VAR.sub(repl, value)
    return value


def load_config(path=None) -> dict:
    """Load arms.yaml, resolve path interpolation, apply env overrides."""
    path = Path(path or os.environ.get("BENCH_CONFIG") or _DEFAULT_CONFIG)
    if yaml is None:
        raise RuntimeError(
            f"pyyaml not importable in this venv ({sys.executable}); "
            "install with: uv pip install --python <venv> pyyaml"
        )
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_config_path"] = str(path)

    # resolve paths against themselves, in insertion order
    paths: dict[str, str] = {}
    for k, v in (cfg.get("paths") or {}).items():
        paths[k] = _interp(v, paths)
    # env override: BENCH_PATH_<UPPER_KEY>
    for k in list(paths):
        env = os.environ.get(f"BENCH_PATH_{k.upper()}")
        if env:
            paths[k] = env
    cfg["paths"] = paths

    # interpolate any ${path} references inside arm / reranker entries
    for arm in cfg.get("arms", []) or []:
        for kk, vv in list(arm.items()):
            arm[kk] = _interp(vv, paths)
    if isinstance(cfg.get("reranker"), dict):
        for kk, vv in list(cfg["reranker"].items()):
            cfg["reranker"][kk] = _interp(vv, paths)
    return cfg


def get_arm(cfg: dict, label: str) -> dict:
    for a in cfg.get("arms", []) or []:
        if a.get("label") == label:
            return a
    have = [a.get("label") for a in cfg.get("arms", []) or []]
    raise KeyError(f"no arm '{label}' in config; available: {have}")


def enabled_arms(cfg: dict, runtime: str | None = None) -> list[dict]:
    out = []
    for a in cfg.get("arms", []) or []:
        if not a.get("enabled", True):
            continue
        if runtime and a.get("runtime") != runtime:
            continue
        out.append(a)
    return out


def setup_logging(name: str, out_dir=None, level=None) -> logging.Logger:
    level = (level or os.environ.get("BENCH_LOG_LEVEL", "INFO")).upper()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(out_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class RamSampler:
    """Sample system free-RAM % on a background thread so every run records the
    memory pressure it actually experienced — so 'slow model' is never confused
    with 'starved machine'. Torch-free; psutil imported lazily."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[float] = []
        self._stop = None
        self._t = None

    def _loop(self):
        import psutil
        while not self._stop.is_set():
            try:
                self.samples.append(100.0 - psutil.virtual_memory().percent)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        import threading
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        if self._stop:
            self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    def stat(self) -> dict:
        s = self.samples
        return {
            "ram_free_pct_min": round(min(s), 1) if s else None,
            "ram_free_pct_mean": round(sum(s) / len(s), 1) if s else None,
        }


def load_jsonl(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_chunks(path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
