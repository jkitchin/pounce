"""Adversary battery: UNBOUNDEDNESS detection edge cases, with BOUNDED controls.

Family: nlp   Class: unboundedness / status-reporting correctness
Targets: PR #250 / #253 (issue #248/#252) — the narrowed dual-divergence /
         DivergingIterates guard.

GOES BEYOND the prior runs on this axis
(2026-07-22_nlp_pr253_{decelerating_recession,missed_unboundedness}.py), which
tested only (a) linear recession rays and (b) sub-linear rays (-log, -sqrt,
-x^0.25). Neither tested the direction that a *too-eager* guard gets wrong:

    ==> PROBLEMS THAT LOOK UNBOUNDED FOR MANY ITERATIONS BUT ARE BOUNDED. <==

Three groups; the analytic truth of every instance is proved in the docstring
of its builder and re-checked numerically at the bottom of this file.

  GROUP A  genuinely UNBOUNDED, clean linear recession ray.
           Guard SHOULD fire. Miss => SOLVER_LIMITATION.

  GROUP B  genuinely UNBOUNDED but the objective descends VERY slowly along
           the ray (-x^0.1, -log(log x), -x/(1+log x)). Guard is expected to
           struggle; Ipopt is the control (if Ipopt also misses it, it is a
           shared gradient-based-termination blind spot, NOT a pounce finding).

  GROUP C  *** CRITICAL CONTROLS ***  BOUNDED, with a finite optimum very far
           from the start, reached only after the objective has plunged
           monotonically over many orders of magnitude while |x| grew
           geometrically. Every "unbounded" signal a divergence guard watches
           for (growing iterates + descending f + non-decelerating drops) is
           present, yet the true answer is a finite optimum. A pounce
           "unbounded"/DivergingIterates verdict here is a FALSE STATUS CLAIM
           on a provably bounded problem  =>  SOLVER_BUG.

Oracle: Ipopt 3.14.19 (/opt/homebrew/bin/ipopt) on the byte-identical .nl
files, PLUS the closed-form analytic optimum for every Group C instance, PLUS
an independent scipy 1-D verification of that optimum.
"""

import math
import os
import shutil
import subprocess
import time

import numpy as np
import pyomo.environ as pyo

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
OUT = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(OUT, "_unbounded_lookalike")
os.makedirs(TMP, exist_ok=True)


# ----------------------------------------------------------------------------
# GROUP A — genuinely unbounded, clean linear recession ray.
# ----------------------------------------------------------------------------
def a_linear_ray():
    """min -x1 - 2 x2  s.t.  x1 - x2 <= 1,  x >= 0.

    TRUTH: UNBOUNDED. d = (1,1) satisfies d1 - d2 = 0 <= 0 and d >= 0, so
    x(t) = (1,1) + t*(1,1) is feasible for all t >= 0 and f = -3 - 3t -> -inf.
    """
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(0, None), initialize=1.0)
    m.x2 = pyo.Var(bounds=(0, None), initialize=1.0)
    m.c = pyo.Constraint(expr=m.x1 - m.x2 <= 1.0)
    m.obj = pyo.Objective(expr=-m.x1 - 2 * m.x2, sense=pyo.minimize)
    return m


def a_nonlinear_ray():
    """min -x + 1/y  s.t.  y >= 1,  x <= 3y,  x >= 0.

    TRUTH: UNBOUNDED. Along x = 3t, y = t (t >= 1) all constraints hold and
    f = -3t + 1/t -> -inf.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=3.0)
    m.y = pyo.Var(bounds=(1, None), initialize=1.0)
    m.c = pyo.Constraint(expr=m.x <= 3 * m.y)
    m.obj = pyo.Objective(expr=-m.x + 1.0 / m.y, sense=pyo.minimize)
    return m


def a_quadratic_ray():
    """min -x^2 + y  s.t.  y >= 0, x >= 1, x <= 10 + y.

    TRUTH: UNBOUNDED. Take y = t, x = 10 + t: f = -(10+t)^2 + t -> -inf.
    Objective drop per step GROWS quadratically — the easiest possible case.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1, None), initialize=1.0)
    m.y = pyo.Var(bounds=(0, None), initialize=0.0)
    m.c = pyo.Constraint(expr=m.x <= 10.0 + m.y)
    m.obj = pyo.Objective(expr=-m.x**2 + m.y, sense=pyo.minimize)
    return m


# ----------------------------------------------------------------------------
# GROUP B — genuinely unbounded, VERY slow descent along the ray.
# ----------------------------------------------------------------------------
def b_x_pow_0p1():
    """min -x^0.1  s.t. x >= 1.  TRUTH: UNBOUNDED (x^0.1 -> inf, unbounded
    below, but needs x = 1e10 to reach f = -10)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1, None), initialize=2.0)
    m.c = pyo.Constraint(expr=m.x >= 1.0)
    m.obj = pyo.Objective(expr=-(m.x**0.1), sense=pyo.minimize)
    return m


def b_loglog():
    """min -log(log(x))  s.t. x >= 3.  TRUTH: UNBOUNDED; the slowest
    divergence in this battery (f = -3 needs x = e^(e^3) ~ 5e8)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(3, None), initialize=5.0)
    m.c = pyo.Constraint(expr=m.x >= 3.0)
    m.obj = pyo.Objective(expr=-pyo.log(pyo.log(m.x)), sense=pyo.minimize)
    return m


def b_x_over_log():
    """min -x/(1 + log(x))  s.t. x >= 1.  TRUTH: UNBOUNDED, super-log but
    sub-linear; per-step drops GROW, so #253's `keeping_up` premise holds."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1, None), initialize=2.0)
    m.c = pyo.Constraint(expr=m.x >= 1.0)
    m.obj = pyo.Objective(expr=-m.x / (1.0 + pyo.log(m.x)), sense=pyo.minimize)
    return m


# ----------------------------------------------------------------------------
# GROUP C — CRITICAL CONTROLS: BOUNDED, finite optimum far away.
# ----------------------------------------------------------------------------
def c_neglog_plus_linear(M):
    """min -log(x) + x/M  s.t. x >= 1.

    TRUTH: BOUNDED. f' = -1/x + 1/M = 0 => x* = M, f'' = 1/x^2 > 0, so x* = M
    is the unique global min, f* = 1 - log(M).  f -> +inf as x -> inf.
    LOOKS UNBOUNDED: for 1 <= x << M the objective is essentially -log(x), so
    it falls by a CONSTANT log(2) per doubling of x for log2(M) doublings.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1, None), initialize=2.0)
    m.c = pyo.Constraint(expr=m.x >= 1.0)
    m.obj = pyo.Objective(expr=-pyo.log(m.x) + m.x / M, sense=pyo.minimize)
    return m, M, 1.0 - math.log(M)


def c_negsqrt_plus_linear(M):
    """min -sqrt(x) + x/M  s.t. x >= 1.

    TRUTH: BOUNDED. f' = -0.5 x^-1/2 + 1/M = 0 => sqrt(x*) = M/2, x* = M^2/4,
    f* = -M/2 + M/4 = -M/4. f'' = 0.25 x^-3/2 > 0 => unique global min.
    LOOKS UNBOUNDED: while x << M^2/4 the drop per doubling of x GROWS like
    sqrt(x), so the `keeping_up` (non-decelerating) test is satisfied
    throughout the approach.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1, None), initialize=2.0)
    m.c = pyo.Constraint(expr=m.x >= 1.0)
    m.obj = pyo.Objective(expr=-pyo.sqrt(m.x) + m.x / M, sense=pyo.minimize)
    return m, M**2 / 4.0, -M / 4.0


def c_neglinear_plus_quadratic(M):
    """min -x + x^2/(2M)  s.t. x >= 0.

    TRUTH: BOUNDED. f' = -1 + x/M = 0 => x* = M, f'' = 1/M > 0, f* = -M/2.
    LOOKS UNBOUNDED: for x << M the objective is essentially -x, so the drop
    per doubling of x GROWS LINEARLY — the strongest possible
    'non-decelerating descent along a growing ray' signal a guard can see.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=1.0)
    m.c = pyo.Constraint(expr=m.x >= 0.0)
    m.obj = pyo.Objective(expr=-m.x + m.x**2 / (2.0 * M), sense=pyo.minimize)
    return m, M, -M / 2.0


def c_flat_then_wall(M):
    """min -x + (x/M)^8  s.t. x >= 0.

    TRUTH: BOUNDED. f' = -1 + 8 x^7 / M^8 = 0 => x*^7 = M^8/8, i.e.
    x* = M^(8/7) / 8^(1/7), f* = -x* + (x*/M)^8. x^8 dominates so f -> +inf;
    f'' = 56 x^6/M^8 > 0 => unique global min.
    LOOKS UNBOUNDED: the gradient is -1 + O((x/M)^7), i.e. numerically
    INDISTINGUISHABLE from `min -x` (a true ray) until x is within a factor of
    ~2 of x*. This is the adversarially hardest bounded control here.
    """
    xs = M ** (8.0 / 7.0) / 8.0 ** (1.0 / 7.0)
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=1.0)
    m.c = pyo.Constraint(expr=m.x >= 0.0)
    m.obj = pyo.Objective(expr=-m.x + (m.x / M) ** 8, sense=pyo.minimize)
    return m, xs, -xs + (xs / M) ** 8


def c_multivar_far(M, n=5):
    """min sum_i (-x_i + x_i^2/(2M)) s.t. sum_i x_i >= 1, x >= 0.

    TRUTH: BOUNDED, separable; x*_i = M for all i, f* = -n*M/2. The coupling
    constraint is inactive at the optimum. Multivariable version so |x|_inf
    growth is driven by n coordinates at once.
    """
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, bounds=(0, None), initialize=1.0)
    m.c = pyo.Constraint(expr=sum(m.x[i] for i in m.I) >= 1.0)
    m.obj = pyo.Objective(
        expr=sum(-m.x[i] + m.x[i] ** 2 / (2.0 * M) for i in m.I), sense=pyo.minimize
    )
    return m, M, -n * M / 2.0


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def write_nl(m, path):
    m.write(path, io_options={"symbolic_solver_labels": False})


def run_solver(binary, nl, extra):
    """Run one solver on its OWN copy of the .nl.

    Both binaries write `<stub>.sol`, so they MUST NOT share a stub or the
    second run silently reads the first run's .sol (this bit the first draft of
    this script and made every status read as `no-sol`).
    """
    tag = os.path.basename(binary)
    work = os.path.join(TMP, tag)
    os.makedirs(work, exist_ok=True)
    mine = os.path.join(work, os.path.basename(nl))
    shutil.copyfile(nl, mine)
    sol = mine[:-3] + ".sol"
    if os.path.exists(sol):
        os.remove(sol)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            [binary, mine, "-AMPL", *extra], capture_output=True, text=True, timeout=60
        )
        txt = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.perf_counter() - t0, None, None
    dt = time.perf_counter() - t0
    solres, obj = None, None
    if os.path.exists(sol):
        for ln in open(sol).read().splitlines():
            if ln.startswith("objno"):
                solres = int(ln.split()[-1])
    for ln in txt.splitlines():
        if ln.strip().startswith("Objective..............."):
            try:
                obj = float(ln.split()[-1])
            except ValueError:
                pass
    return txt, dt, solres, obj


def classify(solres):
    """AMPL solve_result_num bands: 0-99 solved, 100-199 solved?, 200-299
    infeasible, 300-399 UNBOUNDED, 400+ limit/failure."""
    if solres is None:
        return "no-sol"
    if 0 <= solres < 100:
        return "SOLVED"
    if 100 <= solres < 200:
        return "SOLVED?"
    if 200 <= solres < 300:
        return "INFEASIBLE"
    if 300 <= solres < 400:
        return "UNBOUNDED"
    return "LIMIT/FAIL"


def exit_line(txt):
    ls = [l.strip() for l in txt.splitlines() if l.startswith("EXIT")]
    return ls[-1] if ls else "?"


# ----------------------------------------------------------------------------
# Independent numeric proof of the analytic truths.
# ----------------------------------------------------------------------------
def verify_truths():
    from scipy.optimize import minimize_scalar

    print("=== independent verification of the analytic truths (scipy) ===")
    checks = [
        ("neglog+lin M=1e6", lambda x: -np.log(x) + x / 1e6, 1.0, 1e9, 1e6),
        ("negsqrt+lin M=1e3", lambda x: -np.sqrt(x) + x / 1e3, 1.0, 1e9, 1e3**2 / 4),
        ("neglin+quad M=1e6", lambda x: -x + x**2 / (2e6), 0.0, 1e9, 1e6),
        (
            "flat_then_wall M=1e4",
            lambda x: -x + (x / 1e4) ** 8,
            0.0,
            1e5,
            1e4 ** (8 / 7) / 8 ** (1 / 7),
        ),
    ]
    ok = True
    for name, f, lo, hi, xstar in checks:
        r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
        rel = abs(r.x - xstar) / xstar
        # also confirm f -> +inf (bounded below) by sampling far out
        far = f(np.array([1e10, 1e12, 1e14]))
        good = rel < 1e-3 and np.all(far > f(np.array([xstar])))
        ok &= bool(good)
        print(
            f"  {name:<24} scipy x*={r.x:.6e}  analytic x*={xstar:.6e}  "
            f"rel={rel:.2e}  f(far)>f(x*): {np.all(far > f(np.array([xstar])))}"
        )
    # Group A/B: show f actually diverges
    print("  ray checks (f -> -inf):")
    for name, f in [
        ("-x^0.1", lambda t: -(t**0.1)),
        ("-log(log x)", lambda t: -np.log(np.log(t))),
        ("-x/(1+log x)", lambda t: -t / (1 + np.log(t))),
    ]:
        vals = f(np.array([1e2, 1e6, 1e12, 1e30]))
        print(f"    {name:<16} f(1e2,1e6,1e12,1e30) = {np.array2string(vals, precision=3)}")
    print(f"  truths verified: {ok}\n")
    return ok


def main():
    truths_ok = verify_truths()

    # (label, group, model, truth, x_star, f_star)
    cases = []
    for nm, fn in [
        ("A1_linear_ray", a_linear_ray),
        ("A2_nonlinear_ray", a_nonlinear_ray),
        ("A3_quadratic_ray", a_quadratic_ray),
    ]:
        cases.append((nm, "A", fn(), "UNBOUNDED", None, None))
    for nm, fn in [
        ("B1_negx^0.1", b_x_pow_0p1),
        ("B2_neg_loglog", b_loglog),
        ("B3_neg_x_over_log", b_x_over_log),
    ]:
        cases.append((nm, "B", fn(), "UNBOUNDED", None, None))
    for M in (1e6, 1e9, 1e12):
        m, xs, fs = c_neglog_plus_linear(M)
        cases.append((f"C1_neglog+lin_M={M:.0e}", "C", m, "BOUNDED", xs, fs))
    for M in (1e3, 1e5, 1e6):
        m, xs, fs = c_negsqrt_plus_linear(M)
        cases.append((f"C2_negsqrt+lin_M={M:.0e}", "C", m, "BOUNDED", xs, fs))
    for M in (1e4, 1e8, 1e12):
        m, xs, fs = c_neglinear_plus_quadratic(M)
        cases.append((f"C3_neglin+quad_M={M:.0e}", "C", m, "BOUNDED", xs, fs))
    for M in (1e2, 1e4, 1e6):
        m, xs, fs = c_flat_then_wall(M)
        cases.append((f"C4_flat_wall_M={M:.0e}", "C", m, "BOUNDED", xs, fs))
    for M in (1e6, 1e10):
        m, xs, fs = c_multivar_far(M)
        cases.append((f"C5_multivar_M={M:.0e}", "C", m, "BOUNDED", xs, fs))

    rows = []
    for label, grp, m, truth, xstar, fstar in cases:
        nl = os.path.join(TMP, f"{label.replace('+','p').replace('=','_')}.nl")
        write_nl(m, nl)
        pt, tp, rp, op = run_solver(POUNCE, nl, [])
        # forced NLP route: the routing-transparency cross-check.
        nt, tn, rn, on = run_solver(POUNCE, nl, ["solver_selection=nlp"])
        it, ti, ri, oi = run_solver(IPOPT, nl, [])
        route = ""
        for ln in (pt or "").splitlines():
            if "Selected solver:" in ln:
                route = ln.split("Selected solver:")[1].split("[")[0].strip()
        rows.append(
            dict(
                label=label,
                grp=grp,
                truth=truth,
                fstar=fstar,
                pounce=classify(rp),
                p_obj=op,
                route=route,
                pnlp=classify(rn),
                n_obj=on,
                p_exit=exit_line(pt),
                t_p=tp,
                ipopt=classify(ri),
                i_obj=oi,
                i_exit=exit_line(it),
                t_i=ti,
            )
        )

    hdr = (f"{'case':<26} {'grp':<4} {'truth':<10} {'POUNCE(auto)':<13} {'p_obj':>13} "
           f"{'POUNCE(nlp)':<13} {'n_obj':>13} {'IPOPT':<11} {'i_obj':>13} {'f* (truth)':>13}")
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in rows:
        fmt = lambda v: f"{v:.5e}" if v is not None else "-"
        fs = f"{r['fstar']:.5e}" if r["fstar"] is not None else "-inf"
        print(
            f"{r['label']:<26} {r['grp']:<4} {r['truth']:<10} {r['pounce']:<13} "
            f"{fmt(r['p_obj']):>13} {r['pnlp']:<13} {fmt(r['n_obj']):>13} "
            f"{r['ipopt']:<11} {fmt(r['i_obj']):>13} {fs:>13}"
        )
    print()
    for r in rows:
        print(f"{r['label']:<26} pounce: {r['p_exit'][:70]}")
        print(f"{'':<26} ipopt : {r['i_exit'][:70]}")

    # ---- classification -----------------------------------------------------
    false_unbounded = [
        r for r in rows if r["truth"] == "BOUNDED" and r["pounce"] == "UNBOUNDED"
    ]
    wrong_obj = []
    for r in rows:
        if r["truth"] == "BOUNDED" and r["pounce"] in ("SOLVED", "SOLVED?"):
            if r["p_obj"] is None:
                continue
            rel = abs(r["p_obj"] - r["fstar"]) / max(1.0, abs(r["fstar"]))
            if rel > 1e-4:
                wrong_obj.append((r["label"], r["p_obj"], r["fstar"], rel))
    missed = [
        r
        for r in rows
        if r["truth"] == "UNBOUNDED"
        and r["pounce"] != "UNBOUNDED"
        and r["ipopt"] == "UNBOUNDED"
    ]
    shared_miss = [
        r
        for r in rows
        if r["truth"] == "UNBOUNDED"
        and r["pounce"] != "UNBOUNDED"
        and r["ipopt"] != "UNBOUNDED"
    ]

    routing_disagree = [
        r for r in rows
        if r["pounce"] != r["pnlp"] or (
            r["p_obj"] is not None and r["n_obj"] is not None
            and abs(r["p_obj"] - r["n_obj"]) > 1e-4 * max(1.0, abs(r["n_obj"]))
        )
    ]
    print()
    print("--- classification ---")
    print(f"auto-vs-forced-nlp DISAGREEMENT : {len(routing_disagree)} -> "
          f"{[(r['label'], r['pounce'], r['pnlp'], r['route']) for r in routing_disagree]}")
    print(f"truths independently verified : {truths_ok}")
    print(f"FALSE UNBOUNDED on bounded    : {len(false_unbounded)} -> {[r['label'] for r in false_unbounded]}")
    print(f"wrong objective on bounded    : {len(wrong_obj)} -> {wrong_obj}")
    print(f"missed unboundedness (Ipopt got it) : {len(missed)} -> {[r['label'] for r in missed]}")
    print(f"shared miss (Ipopt missed too)      : {len(shared_miss)} -> {[r['label'] for r in shared_miss]}")

    if false_unbounded:
        print(f"VERDICT: SOLVER_BUG ({len(false_unbounded)} false 'unbounded' on provably bounded problems)")
    elif wrong_obj:
        print(f"VERDICT: SOLVER_BUG (wrong optimum on bounded far-optimum problems: {wrong_obj})")
    elif missed:
        print(f"VERDICT: SOLVER_LIMITATION ({len(missed)} genuine unboundedness missed that Ipopt detected)")
    else:
        print("VERDICT: PASS")


if __name__ == "__main__":
    main()
