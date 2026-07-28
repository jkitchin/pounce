"""Adversary cross-check: MPCC-style NLP with MFCQ/LICQ failure at x*.

Family: nlp   Class: degenerate / constraint-qualification failure,
                     non-unique optima, unbounded multiplier set

Problem (standard MPCC "complementarity relaxation" test; the canonical
degenerate NLP reformulation of a complementarity system -- see e.g.
Scheel & Scholtes, "Mathematical programs with complementarity constraints:
stationarity, optimality, and sensitivity", Math. of OR 25(1):1-22, 2000,
Sec. 2; and Fletcher & Leyffer, "Solving MPECs with NLP solvers",
Optim. Methods Softw. 19(1):15-40, 2004, eq. (2)-(3)):

    min  (x1 - 1)^2 + (x2 - 1)^2
    s.t. x1 >= 0
         x2 >= 0
         x1 * x2 <= 0          (complementarity)

KNOWN (analytic):
  * Two global minimizers, f* = 1 exactly:  x* = (1, 0)  and  x* = (0, 1).
    (Feasible set is the union of the two nonneg. coordinate half-axes;
     nearest point of {x1>=0, x2=0} to (1,1) is (1,0) with f = 0 + 1 = 1.)
  * At x* = (1,0) the active constraints are  -x2 <= 0  and  x1*x2 <= 0,
    with gradients  (0,-1)  and  (x2,x1) = (0,1)  -- PARALLEL and opposite.
    => LICQ fails; MFCQ fails (no d with both  -d2 > 0  and  -d2 < 0 ... i.e.
       no strictly feasible direction).  Every CQ of MFCQ strength fails.
  * KKT stationarity:  (2(x1-1), 2(x2-1)) + l1*(0,-1) + l2*(x2,x1) = 0
    at (1,0):  (0,-2) + l1*(0,-1) + l2*(0,1) = 0  =>  l2 - l1 = 2.
    => the multiplier set is the UNBOUNDED RAY {(t, t+2) : t >= 0}.
       KKT points exist, but the multipliers are NON-UNIQUE and UNBOUNDED.

Relaxed comparison family:  x1*x2 <= eps  with eps > 0 restores MFCQ; the
minimum-norm multiplier there stays bounded, but the *solution* moves.  We
sweep eps to show what a well-posed nearby problem looks like.

The question under test: does pounce claim `optimal` here, is the returned
point actually a global minimizer, and what does it report for the duals?
An unbounded/garbage dual is EXPECTED-ish (no unique multiplier exists);
a wrong primal point, or a claim of infeasibility, would be a real finding.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pyomo.environ as pyo

KNOWN_OPTIMAL = 1.0
KNOWN_X = [(1.0, 0.0), (0.0, 1.0)]

HERE = os.path.dirname(os.path.abspath(__file__))
CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"


def build(x0, eps=0.0):
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(0.0, None), initialize=x0[0])
    m.x2 = pyo.Var(bounds=(0.0, None), initialize=x0[1])
    m.obj = pyo.Objective(
        expr=(m.x1 - 1.0) ** 2 + (m.x2 - 1.0) ** 2, sense=pyo.minimize
    )
    m.comp = pyo.Constraint(expr=m.x1 * m.x2 <= eps)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def xy(m):
    return np.array([pyo.value(m.x1), pyo.value(m.x2)])


def dist_to_known(x):
    return min(float(np.linalg.norm(x - np.array(k), np.inf)) for k in KNOWN_X)


def get_dual(m):
    try:
        return float(m.dual[m.comp])
    except Exception:  # noqa: BLE001
        return float("nan")


def solve_with(name, m, tee=False):
    t0 = time.perf_counter()
    try:
        if name == "ipopt":
            res = pyo.SolverFactory("ipopt", executable=IPOPT).solve(m, tee=tee)
        else:
            res = pyo.SolverFactory("pounce").solve(m, tee=tee)
        t = time.perf_counter() - t0
        st = str(res.solver.termination_condition)
        return st, float(pyo.value(m.obj)), xy(m), get_dual(m), t
    except Exception as e:  # noqa: BLE001
        return f"EXC:{e}", float("nan"), np.array([np.nan, np.nan]), float("nan"), (
            time.perf_counter() - t0
        )


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def run_case(tag, x0, eps):
    print(f"\n########## {tag}   x0={x0}  eps={eps:g} ##########")
    out = {"tag": tag, "x0": list(x0), "eps": eps}

    for who in ("pounce", "ipopt"):
        m = build(x0, eps)
        st, f, x, d, t = solve_with(who, m)
        print(
            f"{who:7s}: status={st:28s} obj={f!r} "
            f"x=({x[0]:.10g}, {x[1]:.10g}) dual(comp)={d!r} t={t:.4f}s"
        )
        out[who] = dict(
            status=st, obj=f, x=x.tolist(), dual=d, t=t,
            dist=dist_to_known(x) if np.all(np.isfinite(x)) else float("nan"),
        )

    if eps == 0.0:
        for who in ("pounce", "ipopt"):
            r = out[who]
            if np.isfinite(r["obj"]):
                print(
                    f"    {who}: rel_err_vs_known_f*={rel(r['obj'], KNOWN_OPTIMAL):.3e} "
                    f"inf_dist_to_nearest_x*={r['dist']:.3e}"
                )
    return out


def run_verify(x0):
    """Solve the degenerate .nl with the bare CLI and re-check it with
    `pounce verify` (solver-independent KKT/feasibility oracle)."""
    d = tempfile.mkdtemp(prefix="mpcc_")
    m = build(x0, 0.0)
    nl = os.path.join(d, "m.nl")
    m.write(nl, format="nl", io_options={"symbolic_solver_labels": False})
    sol = os.path.join(d, "m.sol")
    info = {}
    try:
        t0 = time.perf_counter()
        cp = subprocess.run([CLI, nl, sol], capture_output=True, text=True, timeout=60)
        info["cli_t"] = time.perf_counter() - t0
        info["cli_rc"] = cp.returncode
        info["cli_out"] = (cp.stdout + cp.stderr)[-1500:]
        cv = subprocess.run(
            [CLI, "verify", nl, sol], capture_output=True, text=True, timeout=60
        )
        info["verify_rc"] = cv.returncode
        info["verify_out"] = (cv.stdout + cv.stderr)[-2500:]
    except Exception as e:  # noqa: BLE001
        info["verify_out"] = f"EXC:{e}"

    # independent oracle: verify the *published* optimum (1,0) as a claim.
    try:
        cv2 = subprocess.run(
            [CLI, "verify", nl, sol, "--help"], capture_output=True, text=True, timeout=20
        )
        info["verify_help"] = (cv2.stdout + cv2.stderr)[-1200:]
    except Exception:  # noqa: BLE001
        pass

    print("\n---------- bare CLI on degenerate .nl ----------")
    print(info.get("cli_out", ""))
    print(f"---------- pounce verify (rc={info.get('verify_rc')}) ----------")
    print(info.get("verify_out", ""))
    print(f"pounce CLI wall = {info.get('cli_t', float('nan')):.4f}s")
    shutil.rmtree(d, ignore_errors=True)
    return info


if __name__ == "__main__":
    runs = []
    # degenerate (eps = 0): MFCQ fails, multipliers form an unbounded ray
    runs.append(run_case("MPCC deg, symmetric start", (0.5, 0.5), 0.0))
    runs.append(run_case("MPCC deg, biased -> (1,0)", (0.9, 0.1), 0.0))
    runs.append(run_case("MPCC deg, biased -> (0,1)", (0.1, 0.9), 0.0))
    runs.append(run_case("MPCC deg, interior start", (1.0, 1.0), 0.0))
    # relaxed family: MFCQ restored, watch the multiplier blow up as eps -> 0
    for eps in (1e-1, 1e-3, 1e-5, 1e-7):
        runs.append(run_case(f"MPCC relaxed eps={eps:g}", (0.5, 0.5), eps))

    verify = run_verify((0.9, 0.1))

    print("\n================ SUMMARY ================")
    print(f"known f* = {KNOWN_OPTIMAL} at x* in {KNOWN_X} (both global)")
    ok = True
    for r in runs:
        p, i = r["pounce"], r["ipopt"]
        print(
            f"{r['tag']:28s} eps={r['eps']:8.1e} | "
            f"pounce {p['status'][:22]:22s} f={p['obj']:.8g} "
            f"x=({p['x'][0]:.6g},{p['x'][1]:.6g}) dual={p['dual']:.6g} "
            f"t={p['t']:.3f}s | ipopt {i['status'][:22]:22s} f={i['obj']:.8g} "
            f"x=({i['x'][0]:.6g},{i['x'][1]:.6g}) dual={i['dual']:.6g} t={i['t']:.3f}s"
        )
        if r["eps"] == 0.0:
            good = (
                np.isfinite(p["obj"])
                and rel(p["obj"], KNOWN_OPTIMAL) < 1e-4
                and p["dist"] < 1e-4
                and "optimal" in p["status"].lower()
            )
            ok &= bool(good)
    print(f"\npounce verify rc = {verify.get('verify_rc')}")
    with open(os.path.join(HERE, "_mpcc_unbounded_multipliers.json"), "w") as fh:
        json.dump({"runs": runs, "verify": verify}, fh, indent=2, default=str)
    print("VERDICT: PASS" if ok else "VERDICT: FAIL")
