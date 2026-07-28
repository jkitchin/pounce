"""Adversary: BLAST RADIUS of the .sol dual sign defect.

Family: sensitivity / duals    Class: multiplier sign convention
Source: Hillier & Lieberman, "Introduction to Operations Research",
        Wyndor Glass Co. LP (Sec. 3.1); shadow prices in Sec. 4.7 / 6.2.

Convention used throughout (the AMPL/Pyomo "marginal value" convention):
    dual_i  :=  d(objective as stated) / d(rhs_i)
For a MIN problem with  g_i(x) <= b_i  active, relaxing b_i can only lower
the objective, so dual_i <= 0.  For an active  >=  row, dual_i >= 0.
This is exactly what Pyomo's `model.dual[con]` reports from ipopt, and what
`.sol` files are read as.

Wyndor (written as a MIN):
    min  -3 x1 - 5 x2
    s.t. c1:      x1        <= 4
         c2:          2 x2  <= 12
         c3:  3 x1 + 2 x2   <= 18
         x >= 0
    x* = (2, 6), obj* = -36
    Analytic duals (d obj / d b): c1: 0, c2: -3/2, c3: -1
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"

ANALYTIC = {"c1": 0.0, "c2": -1.5, "c3": -1.0}
OBJ_STAR = -36.0
X_STAR = np.array([2.0, 6.0])

WORK = tempfile.mkdtemp(prefix="adv_duals_")
results = {}


def hdr(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 0. Analytic oracle #1: finite-difference re-solve using POUNCE ITSELF.
#    (so the conclusion does not depend on another solver)
# ---------------------------------------------------------------------------
def pounce_lp_obj(b):
    """min -3x1-5x2 s.t. x1<=b0, 2x2<=b1, 3x1+2x2<=b2, x>=0 via pounce solve_qp."""
    from pounce import solve_qp

    G = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([b[0], b[1], b[2], 0.0, 0.0])
    c = np.array([-3.0, -5.0])
    r = solve_qp(P=np.zeros((2, 2)), c=c, G=G, h=h)
    return float(r.obj), np.asarray(r.x), r


hdr("(0) ORACLE: finite-difference d(obj)/d(b) using POUNCE ITSELF")
b0 = np.array([4.0, 12.0, 18.0])
f0, x0, r0 = pounce_lp_obj(b0)
print(f"pounce nominal: obj={f0:.10f} x={x0}  (expect obj*={OBJ_STAR}, x*={X_STAR})")
fd = []
d = 1e-4
for i in range(3):
    bp = b0.copy()
    bp[i] += d
    bm = b0.copy()
    bm[i] -= d
    fd.append((pounce_lp_obj(bp)[0] - pounce_lp_obj(bm)[0]) / (2 * d))
fd = np.array(fd)
print(f"pounce FD d(obj)/d(b) = {fd}")
print(f"analytic              = {[ANALYTIC[k] for k in ('c1','c2','c3')]}")
results["fd_oracle"] = fd.tolist()
assert np.allclose(fd, [0.0, -1.5, -1.0], atol=1e-5), "FD oracle disagrees with analytic!"
print("=> ORACLE CONFIRMED: correct AMPL-convention duals are [0, -1.5, -1.0]")


# ---------------------------------------------------------------------------
# 1. Build the model in Pyomo, write .nl, solve with ipopt and pounce CLI.
# ---------------------------------------------------------------------------
import pyomo.environ as pyo


def build(eq_variant=False):
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(0, None), initialize=1.0)
    m.x2 = pyo.Var(bounds=(0, None), initialize=1.0)
    m.obj = pyo.Objective(expr=-3 * m.x1 - 5 * m.x2, sense=pyo.minimize)
    m.c1 = pyo.Constraint(expr=m.x1 <= 4)
    m.c2 = pyo.Constraint(expr=2 * m.x2 <= 12)
    if eq_variant:
        # replace c3 by the EQUALITY 3x1+2x2 == 18 (active at the same x*)
        m.c3 = pyo.Constraint(expr=3 * m.x1 + 2 * m.x2 == 18)
    else:
        m.c3 = pyo.Constraint(expr=3 * m.x1 + 2 * m.x2 <= 18)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def run_pyomo(solver, eq_variant=False):
    m = build(eq_variant)
    opt = pyo.SolverFactory(solver)
    t0 = time.perf_counter()
    res = opt.solve(m, tee=False)
    t = time.perf_counter() - t0
    duals = {c.name: float(m.dual[c]) for c in (m.c1, m.c2, m.c3)}
    return {
        "obj": float(pyo.value(m.obj)),
        "x": [float(pyo.value(m.x1)), float(pyo.value(m.x2))],
        "duals": duals,
        "t": t,
        "status": str(res.solver.termination_condition),
    }


hdr("(1) INEQUALITY LP: ipopt (AMPL oracle) vs pounce, via Pyomo .nl/.sol")
ip = run_pyomo("ipopt")
print(f"ipopt : obj={ip['obj']:.8f} x={ip['x']} duals={ip['duals']}")
try:
    po = run_pyomo("pounce")
    print(f"pounce: obj={po['obj']:.8f} x={po['x']} duals={po['duals']}")
except Exception as e:  # noqa: BLE001
    print(f"pounce via pyomo FAILED: {e}")
    po = None
print(f"analytic duals: {ANALYTIC}")
results["ineq_ipopt"] = ip
results["ineq_pounce"] = po


# ---------------------------------------------------------------------------
# 2. EQUALITY variant.
#    Convention: for an equality written  h(x) == b,  dual := d obj / d b,
#    identical definition to the inequality case.  Here relaxing 18 -> 18+e
#    changes obj by -1*e, so the correct dual is -1 for BOTH the <= and the
#    == form (the active <= and the == have the same shadow price).
# ---------------------------------------------------------------------------
hdr("(2) EQUALITY variant: 3x1+2x2 == 18 (same active point, shadow price -1)")
ip_eq = run_pyomo("ipopt", eq_variant=True)
print(f"ipopt : obj={ip_eq['obj']:.8f} x={ip_eq['x']} duals={ip_eq['duals']}")
try:
    po_eq = run_pyomo("pounce", eq_variant=True)
    print(f"pounce: obj={po_eq['obj']:.8f} x={po_eq['x']} duals={po_eq['duals']}")
except Exception as e:  # noqa: BLE001
    print(f"pounce eq via pyomo FAILED: {e}")
    po_eq = None
results["eq_ipopt"] = ip_eq
results["eq_pounce"] = po_eq

# FD oracle for the equality form, using pounce's own NLP path
hdr("(2b) FD oracle for the equality form, pounce Problem API")
from pounce import minimize


def eq_obj(rhs):
    from scipy.optimize import linprog  # only as a tie-break, not the oracle

    res = minimize(
        lambda x: -3 * x[0] - 5 * x[1],
        np.array([1.0, 1.0]),
        bounds=[(0, None), (0, None)],
        constraints=[
            {"type": "ineq", "fun": lambda x: 4 - x[0]},
            {"type": "ineq", "fun": lambda x: 12 - 2 * x[1]},
            {"type": "eq", "fun": lambda x: 3 * x[0] + 2 * x[1] - rhs},
        ],
    )
    return float(res.fun), res


f_eq0, r_eq0 = eq_obj(18.0)
fd_eq = (eq_obj(18.0 + 1e-4)[0] - eq_obj(18.0 - 1e-4)[0]) / 2e-4
print(f"pounce minimize eq nominal obj={f_eq0:.8f}")
print(f"pounce FD d(obj)/d(rhs_eq) = {fd_eq:.6f}   (analytic -1.0)")
results["fd_eq"] = fd_eq


# ---------------------------------------------------------------------------
# 3. PYTHON LIBRARY API: does solve_qp / minimize expose duals, and with
#    which sign?
# ---------------------------------------------------------------------------
hdr("(3) PYTHON LIBRARY API dual sign (solve_qp / minimize)")
print("solve_qp result attrs:", [a for a in dir(r0) if not a.startswith("_")])
for attr in ("y", "z", "lam", "lambda_", "duals", "dual"):
    if hasattr(r0, attr):
        print(f"  solve_qp r.{attr} = {np.asarray(getattr(r0, attr))}")
print("  (G rows order: x1<=4, 2x2<=12, 3x1+2x2<=18, -x1<=0, -x2<=0)")
print("minimize result attrs:", [a for a in dir(r_eq0) if not a.startswith("_")])
for attr in ("y", "z", "lam", "lambda_", "duals", "dual", "mult_g", "multipliers"):
    if hasattr(r_eq0, attr):
        v = getattr(r_eq0, attr)
        if v is not None:
            print(f"  minimize r.{attr} = {np.asarray(v)}")


# ---------------------------------------------------------------------------
# 4. Raw .sol inspection + `pounce verify` self-consistency.
# ---------------------------------------------------------------------------
hdr("(4) Raw .nl -> .sol: pounce vs ipopt byte-level duals, and `pounce verify`")


def write_nl(eq_variant=False):
    m = build(eq_variant)
    tag = "eq" if eq_variant else "ineq"
    nlpath = os.path.join(WORK, f"wyndor_{tag}.nl")
    m.write(nlpath, io_options={"symbolic_solver_labels": True})
    return nlpath


def read_sol_duals(path, m_expect):
    txt = open(path).read()
    lines = [ln.strip() for ln in txt.splitlines()]
    i = lines.index("Options")
    j = i + 1
    nopt = int(lines[j])
    j += 1 + nopt
    counts = [int(lines[j + k]) for k in range(4)]
    j += 4
    ndual = counts[0]
    duals = [float(lines[j + k]) for k in range(ndual)]
    prim = [float(lines[j + ndual + k]) for k in range(counts[2])]
    return duals, prim, txt


for eqv in (False, True):
    tag = "EQ " if eqv else "INEQ"
    nl = write_nl(eqv)
    for name, binpath, extra in (
        ("ipopt", IPOPT, []),
        ("pounce", POUNCE, []),
    ):
        base = nl[:-3]
        solf = base + ".sol"
        if os.path.exists(solf):
            os.remove(solf)
        cp = subprocess.run(
            [binpath, base, "-AMPL"] + extra,
            capture_output=True,
            text=True,
            cwd=WORK,
            timeout=30,
        )
        if not os.path.exists(solf):
            print(f"  [{tag}] {name}: no .sol written (rc={cp.returncode})")
            print("   ", cp.stdout[-300:], cp.stderr[-300:])
            continue
        duals, prim, _ = read_sol_duals(solf, 3)
        print(f"  [{tag}] {name:6s} .sol duals = {np.round(duals, 6)}  x={np.round(prim,6)}")
        results[f"sol_{tag.strip()}_{name}"] = duals
        shutil.copy(solf, os.path.join(WORK, f"{name}_{'eq' if eqv else 'ineq'}.sol"))

    # pounce verify on pounce's own .sol AND on ipopt's .sol
    for who in ("pounce", "ipopt"):
        s = os.path.join(WORK, f"{who}_{'eq' if eqv else 'ineq'}.sol")
        if not os.path.exists(s):
            continue
        cp = subprocess.run(
            [POUNCE, "verify", nl, s], capture_output=True, text=True, timeout=30
        )
        out = (cp.stdout + cp.stderr).strip().splitlines()
        tail = " | ".join(out[-4:])
        print(f"  [{tag}] verify({who}.sol) rc={cp.returncode}: {tail[:220]}")
        results[f"verify_{tag.strip()}_{who}"] = cp.returncode


# ---------------------------------------------------------------------------
# 5. sIPOPT sensitivity suffix path (sens_sol_state_1)
# ---------------------------------------------------------------------------
hdr("(5) sens_sol_state_1 suffix: primal or dual? does it carry the sign?")
nl = write_nl(False)
base = nl[:-3]
cp = subprocess.run(
    [POUNCE, base, "-AMPL", "sens_mode=1"],
    capture_output=True,
    text=True,
    cwd=WORK,
    timeout=30,
)
solf = base + ".sol"
txt = open(solf).read() if os.path.exists(solf) else ""
sfx = [ln for ln in txt.splitlines() if ln.startswith("suffix") or "sens" in ln]
print("suffix lines in .sol:", sfx if sfx else "(none)")
print("cli tail:", (cp.stdout + cp.stderr).strip().splitlines()[-3:])


print("\n\nRAW RESULTS:")
print(json.dumps(results, indent=2, default=str))
print(f"\nworkdir: {WORK}")
