"""P3 -- hard nonconvex optimal-control problems (restoration-phase agreement).

corkscrw and clnlbeam are the reviewer's "messy, many-iteration" cases:
pounce reaches the optimum only by going THROUGH its restoration phase.
We run BOTH CLI binaries on the IDENTICAL .nl and check they land on the
same optimum -- objective to solver tolerance, matching "Optimal Solution
Found" status, and the primal/dual vectors -- and we count pounce's
restoration iterations to show the restoration machinery, not just the easy
static NLP path, converges to the same place IPOPT does.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from _common import (IPOPT_BIN, MAIN, POUNCE_BIN, abs_err, dump_result,
                     rel_err)

PROBLEMS = ["corkscrw", "clnlbeam"]


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def _parse_log(log):
    iters = status = None
    restoration = 0
    for line in log.splitlines():
        if "Number of Iterations" in line:
            m = re.search(r":\s*(\d+)", line)
            if m:
                iters = int(m.group(1))
        if line.startswith("EXIT"):
            status = line.replace("EXIT:", "").strip()
        # restoration iterations print the iter index with an 'r' suffix
        if re.match(r"^\s*\d+r\s", line):
            restoration += 1
    return iters, status, restoration


def _read_sol(path, n, m):
    """Return (duals[m], primals[n]) from an AMPL .sol file."""
    vals = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("objno"):
            break
        try:
            vals.append(float(s))
        except ValueError:
            pass
    return np.array(vals[-(n + m):-n]), np.array(vals[-n:])


def solve_problem(name):
    import pounce

    src = MAIN / "benchmarks/mittelmann/nl" / f"{name}.nl"
    tmp = Path(tempfile.mkdtemp(prefix=f"p3_{name}_"))
    nl_p = tmp / "p.nl"
    nl_i = tmp / "i.nl"
    shutil.copy(src, nl_p)
    shutil.copy(src, nl_i)

    log_p = _run([POUNCE_BIN, str(nl_p), "-AMPL",
                  "--sol-output", str(tmp / "p.sol"), "print_level=5"])
    log_i = _run([IPOPT_BIN, str(nl_i), "-AMPL"])
    it_p, st_p, restoration = _parse_log(log_p)
    it_i, st_i, _ = _parse_log(log_i)

    nl = pounce.read_nl(str(nl_p))
    dp, xp = _read_sol(tmp / "p.sol", nl.n, nl.m)
    di, xi = _read_sol(tmp / "i.sol", nl.n, nl.m)
    obj_p = float(nl.objective(xp))
    obj_i = float(nl.objective(xi))

    return {
        "problem": name,
        "n": int(nl.n), "m": int(nl.m),
        "pounce_iters": it_p, "pounce_status": st_p,
        "pounce_restoration_iters": restoration,
        "ipopt_iters": it_i, "ipopt_status": st_i,
        "pounce_objective": obj_p, "ipopt_objective": obj_i,
        "obj_absdiff": abs_err(obj_p, obj_i),
        "obj_relerr": rel_err(obj_p, obj_i),
        "primal_maxabs_diff": float(np.max(np.abs(xp - xi))),
        "primal_l2_relerr": float(
            np.linalg.norm(xp - xi) / max(1.0, np.linalg.norm(xi))),
        "dual_maxabs_diff": float(np.max(np.abs(dp - di))),
        "both_optimal": (st_p == st_i == "Optimal Solution Found."),
    }


def main():
    results = {p: solve_problem(p) for p in PROBLEMS}
    payload = {"problem": "hard_control", "cases": results}
    import json
    print(json.dumps(payload, indent=2))
    dump_result("p3_control", payload)


if __name__ == "__main__":
    main()
