"""Adversary probe: sensitivity steps against a warm re-solve oracle.
Family: sensitivity   Class: kink / degeneracy / released bound
Targets: PR #859 (gh#852 path re-holds a weakly active bound)
         PR #842 (degeneracy="release_all")
         PR #832 (gh#828 capped bound multipliers)
         PR #829/#830/#837 (corrector evaluated at the predicted point)

Oracle: an actual re-solve at the perturbed parameter, solved to a tolerance
two orders tighter than the base solve. This is the only check here that reads
a number the sensitivity layer did not produce.

Per CLAUDE.md the oracle owns steps ABOVE the barrier width and nothing below
it, so every delta swept here is >= 1e-3 while sqrt(mu) ~ 3e-5. Fixtures are
chosen to reach DIFFERENT branches: a decoupled kink (certifies WEAKLY_ACTIVE),
a coupled kink (lands in AMBIGUOUS, gh#763), and a strongly active bound that a
large perturbation drives negative (the release half of fix-relax).
"""
import numpy as np, warnings, itertools, sys
warnings.simplefilter("ignore")
import pyomo.environ as pyo
from pyomo_pounce import sens_solution, declare_sens_param

TOL_BASE, TOL_ORACLE = 1e-10, 1e-12

def solve(m, tol=TOL_BASE):
    pyo.SolverFactory("pounce").solve(m, options={"tol": tol})
    return m

# ---------------- fixtures: each reaches a different branch ----------------
def f_kink(p=0.0):
    """min (x-p)^2, x in [0,10]. At p=0 the bound is weakly active: a kink.
    d x/dp = 1 releasing (p>0), 0 holding (p<0)."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - m.p)**2)
    declare_sens_param(m.p); return solve(m)

def f_coupled(p=0.0, c=0.9):
    """Same kink, coupled to a neighbour so the ratio reduced/diagonal != 1
    and `classify` lands it in AMBIGUOUS rather than WEAKLY_ACTIVE (gh#763)."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.y = pyo.Var(bounds=(-10.0, 10.0), initialize=0.0)
    m.obj = pyo.Objective(expr=(m.x - m.p)**2 + 2*c*m.x*m.y + 2.0*m.y**2)
    declare_sens_param(m.p); return solve(m)

def f_release(p=0.0):
    """x pinned at its lower bound with a STRONGLY active multiplier that a
    large enough perturbation drives negative -> the bound must release."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p)**2 + 0.6*m.x)
    declare_sens_param(m.p); return solve(m)

def f_scaled(p=0.0, s=1e3):
    """The decoupled kink under a per-variable rescaling -- leg 1's dimension."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0*s), initialize=0.5*s)
    m.obj = pyo.Objective(expr=((m.x/s) - m.p)**2)
    declare_sens_param(m.p); return solve(m)

FIX = {"kink": f_kink, "coupled": f_coupled, "release": f_release, "scaled": f_scaled}
MODES = ("linear", "fix_relax", "path")
DEGEN = ("directional", "one_sided", "release_all")
# Per-fixture deltas. `release` needs delta > 0.3 to actually cross its
# release: a first draft swept 1e-3..1e-1 there, never left the bound, and
# reported a clean PASS on a branch it never reached.
DELTAS = {"kink": (1e-3, 1e-2, 1e-1), "coupled": (1e-3, 1e-2, 1e-1),
          "scaled": (1e-3, 1e-2, 1e-1), "release": (0.31, 0.6, 2.0)}

def oracle(builder, delta):
    """Re-solve at p+delta, two orders tighter."""
    m = builder(p=delta)
    solve(m, TOL_ORACLE)
    return pyo.value(m.x)

rows, anomalies = [], []
for fname, builder in FIX.items():
    base = builder()
    x0 = pyo.value(base.x)
    for delta, mode, deg in itertools.product(DELTAS[fname], MODES, DEGEN):
        m = builder()
        try:
            step = sens_solution(m, [(m.p, delta)], mode=mode, degeneracy=deg, clamp=False)
            xs = step[m.x]
        except Exception as e:
            rows.append((fname, delta, mode, deg, None, None, f"raised {type(e).__name__}"))
            anomalies.append((fname, delta, mode, deg, f"raised {type(e).__name__}: {str(e)[:70]}"))
            continue
        xo = oracle(builder, delta)
        err = abs(xs - xo)
        rel = err / max(1e-12, abs(delta))
        rows.append((fname, delta, mode, deg, xs, xo, f"{rel:.3e}"))

hdr = f"{'fixture':<9} {'delta':>7} {'mode':<10} {'degeneracy':<12} {'sens x':>13} {'oracle x':>13} {'err/delta':>10}"
print(hdr); print("-"*len(hdr))
for r in rows:
    xs = f"{r[4]:13.6e}" if r[4] is not None else f"{'n/a':>13}"
    xo = f"{r[5]:13.6e}" if r[5] is not None else f"{'n/a':>13}"
    print(f"{r[0]:<9} {r[1]:>7.0e} {r[2]:<10} {r[3]:<12} {xs} {xo} {r[6]:>10}")

# The documented contract, asserted per fixture at the largest delta:
print("\n-- documented contract checks --")
def best_mode(fname, delta, deg="directional"):
    out = {}
    for mode in MODES:
        m = FIX[fname]()
        try:
            out[mode] = abs(sens_solution(m, [(m.p, delta)], mode=mode,
                                          degeneracy=deg, clamp=False)[m.x]
                            - oracle(FIX[fname], delta)) / delta
        except Exception:
            out[mode] = float('nan')
    return out
for fname in FIX:
    e = best_mode(fname, DELTAS[fname][-1])
    win = min(e, key=lambda k: e[k])
    print(f"  {fname:<9} err/delta by mode: " +
          "  ".join(f"{k}={v:.2e}" for k, v in e.items()) + f"   -> best: {win}")
    if fname == "release" and e["linear"] < 1e-3:
        anomalies.append((fname, DELTAS[fname][-1], "linear", "-",
                          "VACUOUS: `linear` matched the oracle, so the release "
                          "branch was never crossed"))
    if fname == "release" and win == "linear":
        anomalies.append((fname, DELTAS[fname][-1], win, "-",
                          f"docs say only fix_relax reproduces a release; best was {win}"))

print(f"\nanomalies: {len(anomalies)}")
for a in anomalies: print("  ", a)
print("VERDICT: PASS" if not anomalies else f"VERDICT: REVIEW ({len(anomalies)})")
