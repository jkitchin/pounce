"""Follow-up probe for the provable-infeasibility battery.

Two questions the status-only battery cannot answer:

1. On the thin FEASIBLE controls POUNCE says SOLVED -- but is the returned point
   actually feasible and optimal?  A `solved` claim at an infeasible point is the
   mirror image of a false infeasibility claim and just as damaging.  Compared
   against the analytic optimum and Ipopt on the identical .nl.

2. At tangency gap = -1e-7 (disjoint by 1e-7, i.e. inside constr_viol_tol)
   POUNCE returned solve_result_num 500 (FAILURE band) where Ipopt returned 200.
   Dump the full solver output to classify that.
"""

import math
import os
import subprocess
import tempfile

import pyomo.environ as pyo

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
WORK = tempfile.mkdtemp(prefix="adv_probe_")

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "battery",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "2026-07-22_nlp_provable_infeasibility_battery.py"),
)
battery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(battery)

# Analytic optima, derived in the report.  These are GLOBAL values; POUNCE and
# Ipopt are both local solvers, so a local optimum on a disconnected or
# tolerance-dominated feasible set is correct behaviour, not a defect.  The
# binding criterion below is therefore pounce-vs-Ipopt agreement on the
# identical .nl, with the global value reported for context only.
d_lens = 1.0 - 5e-7
KNOWN = {
    # global min y over the lens
    "thin_lens": -math.sqrt(1.0 - d_lens**2),
    # GLOBAL min x is on the negative branch; the feasible set is two disjoint
    # intervals about +-1/sqrt(2) and x0=0.9 sits in the positive one, so both
    # solvers legitimately return the local optimum +sqrt(0.5-1e-4).
    "quartic_trough": +math.sqrt(0.5 - 1e-4),
    # binding at sin(x) = 0.5 - 1e-6, modulo constr_viol_tol on a constraint
    # whose value is squared (a 1e-4 slack in x reads as 1e-8 of violation)
    "curved_tube": math.asin(0.5 - 1e-6),
    "steep_sliver": None,          # ~ (1000-2)^2, solved for below
    "narrow_wedge": 0.0,
}
# tolerance for the informational known-value comparison
KNOWN_TOL = {"curved_tube": 1e-3, "narrow_wedge": 1e-3}


def solve(binary, model, tag):
    d = os.path.join(WORK, tag)
    os.makedirs(d, exist_ok=True)
    nl = os.path.join(d, "m.nl")
    smap_id = model.write(nl, io_options={"symbolic_solver_labels": False})
    p = subprocess.run([binary, nl, "-AMPL", "max_wall_time=8"],
                       capture_output=True, text=True, timeout=30, cwd=d)
    sol = os.path.join(d, "m.sol")
    obj = None
    txt = p.stdout + p.stderr
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("Objective") and ":" in s:
            try:
                obj = float(s.split()[-1])
            except ValueError:
                pass
    return obj, txt, sol, nl, smap_id


def load_back(model, nl, sol, smap_id):
    """Read the .sol back into the Pyomo model so we can evaluate constraints."""
    from pyomo.opt import ReaderFactory
    res = ReaderFactory("sol").__call__(sol)
    model.solutions.load_from(res)
    return res


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


print("=" * 78)
print("PROBE 1 -- are the SOLVED claims on thin feasible controls CORRECT?")
print("=" * 78)
print(f"{'case':<18} {'pounce_obj':>16} {'ipopt_obj':>16} {'known':>16} {'rel_vs_ipopt':>13}")

bad = []
for name in battery.FEASIBLE:
    build = battery.FEASIBLE[name]
    po, ptxt, psol, pnl, psid = solve(POUNCE, build(), f"p_{name}")
    io_, itxt, isol, inl, isid = solve(IPOPT, build(), f"i_{name}")
    kn = KNOWN.get(name)
    r = rel(po, io_) if (po is not None and io_ is not None) else float("nan")
    knstr = f"{kn:.10g}" if kn is not None else "-"
    print(f"{name:<18} {po!s:>16.16} {io_!s:>16.16} {knstr:>16} {r:>13.2e}")
    # binding criterion: pounce must agree with Ipopt on the identical .nl
    if not (r < 1e-5):
        bad.append((name, "disagrees with ipopt", po, io_))
    # informational: distance to the analytic value (local-vs-global caveat above)
    if kn is not None and po is not None and rel(po, kn) > KNOWN_TOL.get(name, 1e-4):
        print(f"    note: {name} differs from analytic {kn:.10g} by {rel(po, kn):.2e}")

print()
print("=" * 78)
print("PROBE 2 -- feasibility of the returned points (pounce verify)")
print("=" * 78)
for name in battery.FEASIBLE:
    m = battery.FEASIBLE[name]()
    d = os.path.join(WORK, f"v_{name}")
    os.makedirs(d, exist_ok=True)
    nl = os.path.join(d, "m.nl")
    m.write(nl, io_options={"symbolic_solver_labels": False})
    subprocess.run([POUNCE, nl, "-AMPL", "max_wall_time=8"],
                   capture_output=True, text=True, cwd=d)
    sol = os.path.join(d, "m.sol")
    v = subprocess.run([POUNCE, "verify", nl, sol],
                       capture_output=True, text=True, cwd=d)
    out = (v.stdout + v.stderr).strip().replace("\n", " | ")
    print(f"{name:<18} verify_exit={v.returncode}  {out[:110]}")
    if v.returncode != 0:
        bad.append((name, "verify", v.returncode))

print()
print("=" * 78)
print("PROBE 3 -- tangency gap = -1e-7, full pounce output (objno 500 band)")
print("=" * 78)
for gap in (-1e-6, -1e-7, -1e-8):
    dd = 1.0 - gap / 2.0
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-5, 5), initialize=0.0)
    m.y = pyo.Var(bounds=(-5, 5), initialize=0.0)
    m.obj = pyo.Objective(expr=m.y)
    m.c1 = pyo.Constraint(expr=(m.x + dd) ** 2 + m.y**2 <= 1.0)
    m.c2 = pyo.Constraint(expr=(m.x - dd) ** 2 + m.y**2 <= 1.0)
    tag = f"t{abs(gap):.0e}".replace("-", "")
    _, txt, sol, nl, _ = solve(POUNCE, m, f"p_gap_{tag}")
    res = None
    if os.path.exists(sol):
        for line in reversed(open(sol).read().strip().splitlines()):
            if line.startswith("objno"):
                res = int(line.split()[-1])
                break
    print(f"--- gap={gap:+.0e}  objno={res} ---")
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("EXIT") or "nfeasib" in s or "estoration" in s or s.startswith("Number of Iter"):
            print("   ", s[:100])
    print()

print("=" * 78)
if bad:
    print("PROBE FINDINGS:")
    for b in bad:
        print("  -", b)
    print("VERDICT: FAIL")
else:
    print("all SOLVED claims on thin feasible controls verified correct")
    print("VERDICT: PASS")
