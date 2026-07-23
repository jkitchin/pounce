"""Shared helpers for the solver-equivalence validation scripts.

Every problem script imports from here so the pounce/IPOPT plumbing,
JSON emission, and numeric-comparison helpers are written once.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

# Absolute paths into the MAIN checkout (this file lives in a git worktree;
# the .nl fixtures and the CHO model source are only in the main checkout).
MAIN = Path("/Users/jkitchin/projects/pounce")
IPOPT_BIN = "/opt/homebrew/bin/ipopt"
POUNCE_BIN = str(MAIN / "target/release/pounce")
VALIDATION_DIR = Path(__file__).resolve().parent


def rel_err(a: float, b: float) -> float:
    """Relative difference |a-b| / max(1, |b|)."""
    a = float(a)
    b = float(b)
    return abs(a - b) / max(1.0, abs(b))


def abs_err(a: float, b: float) -> float:
    return abs(float(a) - float(b))


def setup_pounce():
    """Import pyomo_pounce so SolverFactory('pounce') resolves to the
    wheel-bundled binary (a symlink onto the freshly built
    target/release/pounce) via the pyomo_pounce ASL plugin.

    This matters: with pyomo_pounce NOT imported, Pyomo's generic ASL
    fallback picks up whatever `pounce` is first on PATH -- here a stale
    ~/.local/bin/pounce from before the #271 dual-sign fix -- and reports
    sign-flipped duals. Importing pyomo_pounce pins the correct binary.
    Returns the resolved pounce executable path.
    """
    import pyomo_pounce  # noqa: F401
    from pyomo.opt import SolverFactory

    return SolverFactory("pounce").executable()


def pounce_version() -> str:
    import pounce

    return pounce.__version__


def ipopt_version() -> str:
    out = subprocess.run(
        [IPOPT_BIN, "--version"], capture_output=True, text=True
    ).stdout
    return out.strip().splitlines()[0] if out.strip() else "unknown"


def today() -> str:
    return subprocess.run(
        ["date", "+%Y-%m-%d"], capture_output=True, text=True
    ).stdout.strip()


def dump_result(name: str, payload: dict) -> None:
    """Write validation/results.<name>.json for run_all.py to gather."""
    path = VALIDATION_DIR / f"results.{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"[{name}] wrote {path}")


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return str(o)


def finite(x) -> float:
    return float(x)
