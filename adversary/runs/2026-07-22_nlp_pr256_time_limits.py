#!/usr/bin/env python3
"""Adversarial probe of PR #256 (proactive factorization time-budget guard)
and PR #255 (feral_static_pivoting option).

Hypotheses:
  A  premature termination under near-natural-time limits
  B  time limit actually respected (overshoot ratio) on a slow problem
  C  status correctness when the limit / guard fires
  D  feral_static_pivoting is plumbed through and answer-neutral

Never modifies pounce source; only reads .nl fixtures from benchmarks/.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

REPO = "/Users/jkitchin/projects/pounce"
POUNCE = f"{REPO}/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"

SUITE = {
    # name: (nl path, expected regime)
    "hs071ish_sparseqp": f"{REPO}/benchmarks/large_scale/nl/sparseqp.nl",
    "bratu": f"{REPO}/benchmarks/large_scale/nl/bratu.nl",
    "poisson": f"{REPO}/benchmarks/large_scale/nl/poisson.nl",
    "optcontrol": f"{REPO}/benchmarks/large_scale/nl/optcontrol.nl",
    "clnlbeam": f"{REPO}/benchmarks/mittelmann/nl/clnlbeam.nl",
    "elec_400": f"{REPO}/benchmarks/mittelmann/nl/elec_400.nl",
    "bearing_400": f"{REPO}/benchmarks/mittelmann/nl/bearing_400.nl",
    "dirichlet120": f"{REPO}/benchmarks/mittelmann/nl/dirichlet120.nl",
    "cont5_2_4_l": f"{REPO}/benchmarks/mittelmann/nl/cont5_2_4_l.nl",
    "camshape_6400": f"{REPO}/benchmarks/mittelmann/nl/camshape_6400.nl",
    # NARX_CFy excluded from the default suite: >8 min baseline, unusable
    # as an oracle-checked fixture. Add by name on the command line.
    "NARX_CFy": f"{REPO}/benchmarks/mittelmann/nl/NARX_CFy.nl",
}
DEFAULT_SUITE = [k for k in SUITE if k != "NARX_CFy"]


def run_pounce(nl, opts=None, env_extra=None, timeout=900, binary=None):
    """Run pounce on nl with KEY=VALUE opts; return dict with wall, status, obj."""
    opts = opts or {}
    with tempfile.TemporaryDirectory() as td:
        jf = os.path.join(td, "report.json")
        cmd = [binary or POUNCE, nl, "--json-output", jf, "--no-sol"]
        cmd += [f"{k}={v}" for k, v in opts.items()]
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        t0 = time.perf_counter()
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=td
            )
            killed = False
        except subprocess.TimeoutExpired:
            return {"wall": time.perf_counter() - t0, "status": "HARNESS_TIMEOUT",
                    "obj": None, "rc": None, "killed": True, "iters": None}
        wall = time.perf_counter() - t0
        rep = None
        if os.path.exists(jf):
            try:
                rep = json.load(open(jf))
            except Exception:
                rep = None
    out = {"wall": wall, "rc": p.returncode, "killed": killed,
           "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-2000:]}
    if rep:
        out["status"] = _dig(rep, "status") or _dig(rep, "solver_status") or "?"
        out["obj"] = _dig(rep, "objective")
        out["iters"] = _dig(rep, "iteration_count")
        out["n_factors"] = _dig(rep, "n_factors")
        out["srn"] = _dig(rep, "solve_result_num")
        out["report"] = rep
    else:
        out["status"] = "NO_REPORT"
        out["obj"] = None
        out["iters"] = None
    return out


def _dig(d, key):
    """Find first occurrence of key anywhere in nested dict/list."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _dig(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _dig(v, key)
            if r is not None:
                return r
    return None


def run_ipopt(nl, timeout=900):
    with tempfile.TemporaryDirectory() as td:
        stub = os.path.join(td, "p")
        os.symlink(nl, stub + ".nl")
        t0 = time.perf_counter()
        p = subprocess.run([IPOPT, stub, "-AMPL"], capture_output=True,
                           text=True, timeout=timeout, cwd=td)
        wall = time.perf_counter() - t0
        obj = None
        status = "?"
        for line in p.stdout.splitlines():
            ls = line.strip()
            if ls.startswith("Objective...............:"):
                obj = float(ls.split()[-1])
            if "EXIT:" in ls:
                status = ls.split("EXIT:")[1].strip()
        return {"wall": wall, "obj": obj, "status": status}


def cmd_baseline():
    print(f"{'problem':>16} {'pounce_wall':>12} {'status':>26} {'obj':>18} "
          f"{'ipopt_wall':>11} {'ipopt_obj':>18}")
    res = {}
    for name in DEFAULT_SUITE:
        nl = SUITE[name]
        if not os.path.exists(nl):
            continue
        r = run_pounce(nl)
        try:
            i = run_ipopt(nl)
        except Exception as e:
            i = {"wall": float("nan"), "obj": None, "status": str(e)[:20]}
        res[name] = {"pounce": {k: r[k] for k in ("wall", "status", "obj", "iters")},
                     "ipopt": i}
        print(f"{name:>16} {r['wall']:12.3f} {str(r['status'])[:26]:>26} "
              f"{_f(r['obj']):>18} {i['wall']:11.3f} {_f(i['obj']):>18}")
    json.dump(res, open("baseline.json", "w"), indent=2)
    return res


def _f(x):
    return "None" if x is None else f"{x:.10g}"


def cmd_sweep(name, limits):
    """Sweep max_wall_time over `limits` for one problem."""
    nl = SUITE[name]
    print(f"== {name} : max_wall_time sweep ==")
    print(f"{'limit':>10} {'wall':>9} {'ratio':>8} {'status':>28} {'obj':>18} {'iters':>6}")
    rows = []
    for L in limits:
        r = run_pounce(nl, {"max_wall_time": L})
        ratio = r["wall"] / L if L else float("nan")
        print(f"{L:10.4g} {r['wall']:9.3f} {ratio:8.2f} {str(r['status'])[:28]:>28} "
              f"{_f(r['obj']):>18} {str(r['iters']):>6}")
        rows.append({"limit": L, **{k: r[k] for k in ("wall", "status", "obj", "iters", "rc")}})
    json.dump(rows, open(f"sweep_{name}.json", "w"), indent=2)
    return rows


def cmd_pivot(names):
    print(f"{'problem':>16} {'setting':>8} {'wall':>9} {'status':>26} {'obj':>20}")
    out = {}
    for name in names:
        nl = SUITE[name]
        row = {}
        for setting in ("unset", "yes", "no"):
            opts = {} if setting == "unset" else {"feral_static_pivoting": setting}
            r = run_pounce(nl, opts)
            row[setting] = {k: r[k] for k in ("wall", "status", "obj", "iters")}
            print(f"{name:>16} {setting:>8} {r['wall']:9.3f} "
                  f"{str(r['status'])[:26]:>26} {_f(r['obj']):>20}")
        out[name] = row
    json.dump(out, open("pivot.json", "w"), indent=2)
    return out


def cmd_pivot_env(names):
    """Same but via POUNCE_FERAL_STATIC_PIVOTING env var."""
    print(f"{'problem':>16} {'env':>8} {'wall':>9} {'status':>26} {'obj':>20} {'iters':>6}")
    for name in names:
        nl = SUITE[name]
        for setting in (None, "1", "0"):
            env = {} if setting is None else {"POUNCE_FERAL_STATIC_PIVOTING": setting}
            r = run_pounce(nl, {}, env_extra=env)
            print(f"{name:>16} {str(setting):>8} {r['wall']:9.3f} "
                  f"{str(r['status'])[:26]:>26} {_f(r['obj']):>20} {str(r['iters']):>6}")


PRE254 = ("/private/tmp/claude-501/-Users-jkitchin-projects-pounce/"
          "499bd6ed-f41c-4ce5-ae4a-06f80063e34a/scratchpad/pre254/target/release/pounce")


def cmd_ab(name, limits, reps=2):
    """A/B the overshoot: HEAD (with the #254 predictive guard) vs the
    parent commit ba29b53 (reactive checks only). Interleaved so drift in
    background load hits both arms equally."""
    nl = SUITE[name]
    print(f"== {name} : A/B overshoot, HEAD(guard) vs ba29b53(no guard) ==")
    print(f"{'limit':>8} {'rep':>4} {'build':>8} {'wall':>9} {'over':>8} "
          f"{'ratio':>7} {'status':>26} {'iters':>6}")
    rows = []
    for L in limits:
        for rep in range(reps):
            for tag, binp in (("HEAD", POUNCE), ("pre254", PRE254)):
                r = run_pounce(nl, {"max_wall_time": L}, binary=binp)
                over = r["wall"] - L
                print(f"{L:8.4g} {rep:4d} {tag:>8} {r['wall']:9.3f} {over:8.3f} "
                      f"{r['wall']/L:7.2f} {str(r['status'])[:26]:>26} "
                      f"{str(r['iters']):>6}")
                rows.append({"limit": L, "rep": rep, "build": tag,
                             **{k: r[k] for k in ("wall", "status", "obj", "iters")}})
    json.dump(rows, open(f"ab_{name}.json", "w"), indent=2)
    return rows


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if what == "baseline":
        cmd_baseline()
    elif what == "sweep":
        cmd_sweep(sys.argv[2], [float(x) for x in sys.argv[3:]])
    elif what == "pivot":
        cmd_pivot(sys.argv[2:])
    elif what == "pivotenv":
        cmd_pivot_env(sys.argv[2:])
    elif what == "ab":
        cmd_ab(sys.argv[2], [float(x) for x in sys.argv[3:]])
