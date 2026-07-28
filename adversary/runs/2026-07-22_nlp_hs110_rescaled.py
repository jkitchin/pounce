"""Adversary cross-check: Hock-Schittkowski 110, natural + deliberately rescaled.

Family: nlp   Class: bound-constrained, ill-conditioned / badly scaled
Source: W. Hock & K. Schittkowski, "Test Examples for Nonlinear Programming
        Codes", Lecture Notes in Economics and Mathematical Systems 187,
        Springer 1981, Problem 110 (p. 118).
        min sum_{i=1..10} [ (ln(x_i - 2))^2 + (ln(10 - x_i))^2 ] - (prod x_i)^0.2
        s.t. 2.001 <= x_i <= 9.999,  x0_i = 9
Known optimal: f* = -45.77846971, x*_i = 9.35025655 (all i)

The rescaled variant substitutes x_i = s_i * y_i with s_i spanning 1e-6..1e6.
The optimal OBJECTIVE VALUE is therefore mathematically IDENTICAL (-45.77846971)
and x* = s .* y* recovers the same point -- but the variables, their bounds, and
the Hessian entries now differ by 12 orders of magnitude.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pyomo.environ as pyo

KNOWN_OPTIMAL = -45.77846971
KNOWN_X = np.full(10, 9.35025655)

# scale factors spanning 1e-6 .. 1e6  (condition spread ~1e12)
S_MID = np.array([1e-3, 1e3, 1.0, 1e-6, 1e6, 1e2, 1e-2, 1e4, 1e-4, 1.0])
S_EXTREME = np.array([1e-9, 1e9, 1.0, 1e-6, 1e6, 1e3, 1e-3, 1e7, 1e-7, 1.0])
S = S_MID  # rebound per-run by build()

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"


def build(scale):
    s = np.ones(10) if scale is None else scale
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 9)
    m.y = pyo.Var(
        m.I,
        bounds=lambda mm, i: (2.001 / s[i], 9.999 / s[i]),
        initialize=lambda mm, i: 9.0 / s[i],
    )

    def obj(mm):
        t = sum(
            (pyo.log(s[i] * mm.y[i] - 2.0)) ** 2 + (pyo.log(10.0 - s[i] * mm.y[i])) ** 2
            for i in mm.I
        )
        p = 1.0
        for i in mm.I:
            p *= s[i] * mm.y[i]
        return t - p**0.2

    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)
    m._s = s
    return m


def xvals(m):
    return np.array([m._s[i] * pyo.value(m.y[i]) for i in range(10)])


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def run(tag, scale):
    print(f"\n########## {tag} ##########")
    print(f"    scale spread = {(scale.max()/scale.min() if scale is not None else 1.0):.1e}")
    out = {"tag": tag}

    # ---- pounce (via pyomo-pounce) ----
    m = build(scale)
    t0 = time.perf_counter()
    try:
        res = pyo.SolverFactory("pounce").solve(m, tee=False)
        tp = time.perf_counter() - t0
        st = str(res.solver.termination_condition)
        fp = pyo.value(m.obj)
        xp = xvals(m)
    except Exception as e:  # noqa: BLE001
        tp = time.perf_counter() - t0
        st, fp, xp = f"EXC:{e}", float("nan"), np.full(10, np.nan)
    print(f"pounce : status={st} obj={fp!r} t={tp:.4f}s")
    print(f"         x={np.array2string(xp, precision=6)}")
    out.update(pounce_status=st, pounce_obj=fp, pounce_t=tp, pounce_x=xp.tolist())

    # ---- oracle: Ipopt on the identical pyomo model ----
    mo = build(scale)
    t0 = time.perf_counter()
    try:
        reso = pyo.SolverFactory("ipopt", executable="/opt/homebrew/bin/ipopt").solve(
            mo, tee=False
        )
        to = time.perf_counter() - t0
        sto = str(reso.solver.termination_condition)
        fo = pyo.value(mo.obj)
        xo = xvals(mo)
    except Exception as e:  # noqa: BLE001
        to = time.perf_counter() - t0
        sto, fo, xo = f"EXC:{e}", float("nan"), np.full(10, np.nan)
    print(f"ipopt  : status={sto} obj={fo!r} t={to:.4f}s")
    print(f"         x={np.array2string(xo, precision=6)}")
    out.update(ipopt_status=sto, ipopt_obj=fo, ipopt_t=to, ipopt_x=xo.tolist())

    # ---- pounce verify on the .nl ----
    d = tempfile.mkdtemp(prefix="hs110_")
    mv = build(scale)
    nl = os.path.join(d, "m.nl")
    mv.write(nl, format="nl", io_options={"symbolic_solver_labels": False})
    sol = os.path.join(d, "m.sol")
    vrc, vout = None, ""
    try:
        t0 = time.perf_counter()
        cp = subprocess.run(
            [CLI, nl, sol], capture_output=True, text=True, timeout=60
        )
        t_cli = time.perf_counter() - t0
        cp2 = subprocess.run(
            [CLI, "verify", nl, sol], capture_output=True, text=True, timeout=60
        )
        vrc = cp2.returncode
        vout = (cp.stdout[-300:] + "\n---verify---\n" + cp2.stdout + cp2.stderr)[-1200:]
        out["cli_t"] = t_cli
    except Exception as e:  # noqa: BLE001
        vout = f"EXC:{e}"
    # bare-CLI ipopt timing on the same .nl for a fair wall-clock comparison
    try:
        t0 = time.perf_counter()
        subprocess.run(
            ["/opt/homebrew/bin/ipopt", nl, "-AMPL"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=d,
        )
        out["ipopt_cli_t"] = time.perf_counter() - t0
    except Exception:  # noqa: BLE001
        out["ipopt_cli_t"] = float("nan")
    print(f"pounce verify rc={vrc}\n{vout}")
    print(
        f"bare CLI wall: pounce={out.get('cli_t', float('nan')):.4f}s "
        f"ipopt={out.get('ipopt_cli_t', float('nan')):.4f}s"
    )
    out.update(verify_rc=vrc, verify_out=vout, workdir=d)

    # ---- errors ----
    for who, f, x in (("pounce", fp, xp), ("ipopt", fo, xo)):
        e_known = rel(f, KNOWN_OPTIMAL) if np.isfinite(f) else float("nan")
        e_x = (
            float(np.max(np.abs(x - KNOWN_X))) if np.all(np.isfinite(x)) else float("nan")
        )
        print(f"{who}: rel_err_vs_known={e_known:.3e}  x_inf_err_vs_known={e_x:.3e}")
        out[f"{who}_relerr"] = e_known
        out[f"{who}_xerr"] = e_x
    out["obj_err_vs_oracle"] = (
        rel(fp, fo) if np.isfinite(fp) and np.isfinite(fo) else float("nan")
    )
    print(f"obj_err_vs_oracle={out['obj_err_vs_oracle']:.3e}")
    shutil.rmtree(d, ignore_errors=True)
    return out


if __name__ == "__main__":
    runs = [
        run("HS110 natural", None),
        run("HS110 rescaled 1e-6..1e6", S_MID),
        run("HS110 rescaled 1e-9..1e9", S_EXTREME),
    ]

    print("\n================ SUMMARY ================")
    print(f"known optimal f* = {KNOWN_OPTIMAL}")
    ok = True
    for r in runs:
        good = (
            np.isfinite(r["pounce_obj"])
            and r["pounce_relerr"] < 1e-4
            and r["pounce_status"] in ("optimal", "TerminationCondition.optimal")
        )
        ok &= bool(good)
        print(
            f"{r['tag']:32s} pounce={r['pounce_obj']!r} ({r['pounce_status']}) "
            f"relerr={r['pounce_relerr']:.2e} t={r['pounce_t']:.3f}s | "
            f"ipopt={r['ipopt_obj']!r} ({r['ipopt_status']}) t={r['ipopt_t']:.3f}s"
        )
    with open(os.path.join(HERE, "_hs110_rescaled_results.json"), "w") as fh:
        json.dump(runs, fh, indent=2, default=str)
    print("VERDICT: PASS" if ok else "VERDICT: FAIL")
