#!/usr/bin/env python
"""Adversarial test of PR #250 (ba29b53): the dual-divergence guard.

Hypothesis under test: the `dual_diverging_streak` guard (default 15) fires on
FEASIBLE, BOUNDED problems whose dual infeasibility legitimately grows for many
consecutive iterations, causing POUNCE to report DivergingIterates (AMPL
UNBOUNDED band, solve_result_num 300) where Ipopt finds a finite optimum.

Strategy: build small (<50 var) NLPs that are known-hard on the dual residual --
LICQ violations, MPEC complementarity, redundant active constraints, badly
scaled / ill-conditioned Hessians, deliberately awful initial multipliers --
write each to a .nl file, then solve the IDENTICAL .nl with
  * /opt/homebrew/bin/ipopt          (oracle)
  * target/release/pounce            (default options)
  * target/release/pounce dual_diverging_streak=0   (guard disabled)
  * target/release/pounce dual_diverging_streak=2   (guard hair-trigger)
and compare status + objective.

Never modifies pounce source.  Writes only under adversary/runs/pr250_work/.
"""

import json
import os
import re
import subprocess
import sys

import pyomo.environ as pyo

ROOT = "/Users/jkitchin/projects/pounce"
POUNCE = f"{ROOT}/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
WORK = f"{ROOT}/adversary/runs/pr250_work"

os.makedirs(WORK, exist_ok=True)


# --------------------------------------------------------------------------
# Problems.  Each returns (ConcreteModel, known_optimal_or_None, note)
# --------------------------------------------------------------------------
def hs13():
    """HS13.  min (x1-2)^2+x2^2  s.t. (1-x1)^3-x2>=0, x>=0.  f*=1 at (1,0).
    LICQ fails at the solution (constraint gradient vanishes)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], bounds=(0, None), initialize={1: -2.0, 2: -2.0})
    m.obj = pyo.Objective(expr=(m.x[1] - 2) ** 2 + m.x[2] ** 2)
    m.c = pyo.Constraint(expr=(1 - m.x[1]) ** 3 - m.x[2] >= 0)
    return m, 1.0, "LICQ violated at the solution"


def hs55():
    """HS55.  6 vars, 6 linear equalities, degenerate; f*=6.333333."""
    m = pyo.ConcreteModel()
    ub = {1: 1.0, 2: None, 3: None, 4: 1.0, 5: None, 6: None}
    init = {1: 1.0, 2: 2.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 2.0}
    m.x = pyo.Var(range(1, 7), initialize=init)
    for i in range(1, 7):
        m.x[i].setlb(0.0)
        m.x[i].setub(ub[i])
    x = m.x
    m.obj = pyo.Objective(
        expr=x[1] + 2 * x[2] + 4 * x[5] + pyo.exp(x[1] * x[4])
    )
    m.c1 = pyo.Constraint(expr=x[1] + 2 * x[2] + 5 * x[5] == 6)
    m.c2 = pyo.Constraint(expr=x[1] + x[2] + x[3] == 3)
    m.c3 = pyo.Constraint(expr=x[4] + x[5] + x[6] == 2)
    m.c4 = pyo.Constraint(expr=x[1] + x[4] == 1)
    m.c5 = pyo.Constraint(expr=x[2] + x[5] == 2)
    m.c6 = pyo.Constraint(expr=x[3] + x[6] == 2)
    return m, 6.333333333, "degenerate linear equalities, redundant rows"


def mpec_comp():
    """Small MPEC-style complementarity NLP (relaxed to an inequality).
    min (x-1)^2+(y-1)^2 s.t. x*y<=0, x,y>=0.  f*=1 at (1,0) or (0,1).
    MPCC: LICQ and MFCQ both fail at every feasible point of the
    complementarity set -> multipliers unbounded, inf_du prone to blow up."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=0.5)
    m.y = pyo.Var(bounds=(0, None), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - 1) ** 2 + (m.y - 1) ** 2)
    m.c = pyo.Constraint(expr=m.x * m.y <= 0)
    return m, 1.0, "MPCC: MFCQ fails everywhere on the feasible set"


def mpec_chain(n=8):
    """Chain of complementarities: n pairs, all MFCQ-violating."""
    m = pyo.ConcreteModel()
    I = range(n)
    m.x = pyo.Var(I, bounds=(0, None), initialize=0.7)
    m.y = pyo.Var(I, bounds=(0, None), initialize=0.7)
    m.obj = pyo.Objective(
        expr=sum((m.x[i] - 1 - 0.1 * i) ** 2 + (m.y[i] - 1) ** 2 for i in I)
    )
    m.c = pyo.Constraint(I, rule=lambda m, i: m.x[i] * m.y[i] <= 0)
    return m, None, "8 stacked complementarities"


def redundant_active(n=10):
    """n copies of the SAME active constraint -> multipliers non-unique,
    LICQ badly violated (rank 1 active Jacobian with n rows)."""
    m = pyo.ConcreteModel()
    I = range(n)
    m.x = pyo.Var([1, 2], initialize={1: 3.0, 2: -4.0})
    m.obj = pyo.Objective(expr=(m.x[1] - 5) ** 2 + (m.x[2] - 5) ** 2)
    m.c = pyo.Constraint(
        I, rule=lambda m, i: m.x[1] ** 2 + m.x[2] ** 2 <= 1.0
    )
    # f* = (r-5sqrt2)^2 ... = 2*(5 - 1/sqrt2)^2 with x=(1/sqrt2,1/sqrt2)
    fstar = 2 * (5 - 2 ** -0.5) ** 2
    return m, fstar, "10 identical active constraints (rank-deficient)"


def badscale_hess():
    """Hilbert-like ill-conditioned quadratic with a nonlinear equality and a
    huge objective scale.  Bounded, unique minimum."""
    n = 12
    m = pyo.ConcreteModel()
    I = range(n)
    m.x = pyo.Var(I, initialize=1.0, bounds=(-100, 100))
    m.obj = pyo.Objective(
        expr=1e6
        * sum(
            m.x[i] * m.x[j] / (i + j + 1.0) for i in I for j in I
        )
        + sum((10.0 ** (i - 6)) * m.x[i] for i in I)
    )
    m.c = pyo.Constraint(expr=sum(m.x[i] ** 2 for i in I) <= 50.0)
    return m, None, "Hilbert Hessian, 1e6 objective scale, 1e-6..1e6 gradient"


def deg_eq_ineq():
    """Equality + inequality that coincide at the solution (weakly active,
    zero multiplier), plus a nonconvex objective.  Strict complementarity
    fails -> the dual residual is slow and bumpy."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize={1: -1.5, 2: 2.5}, bounds=(-10, 10))
    m.obj = pyo.Objective(expr=m.x[1] ** 3 - m.x[2] ** 2 + 4 * m.x[1] * m.x[2])
    m.ce = pyo.Constraint(expr=m.x[1] ** 2 + m.x[2] ** 2 == 4.0)
    m.ci = pyo.Constraint(expr=m.x[1] ** 2 + m.x[2] ** 2 <= 4.0)
    return m, None, "coincident eq/ineq: weakly active, zero multiplier"


def hs112():
    """HS112 chemical equilibrium: 10 vars, 3 linear equalities, log terms.
    Notorious for large dual residuals from a poor start.  f* = -47.707579."""
    c = [
        -6.089, -17.164, -34.054, -5.914, -24.721,
        -14.986, -24.100, -10.708, -26.662, -22.179,
    ]
    m = pyo.ConcreteModel()
    I = range(1, 11)
    m.x = pyo.Var(I, bounds=(1e-6, None), initialize=0.1)
    s = sum(m.x[i] for i in I)
    m.obj = pyo.Objective(
        expr=sum(m.x[j] * (c[j - 1] + pyo.log(m.x[j] / s)) for j in I)
    )
    m.c1 = pyo.Constraint(
        expr=m.x[1] + 2 * m.x[2] + 2 * m.x[3] + m.x[6] + m.x[10] == 2
    )
    m.c2 = pyo.Constraint(
        expr=m.x[4] + 2 * m.x[5] + m.x[6] + m.x[7] == 1
    )
    m.c3 = pyo.Constraint(
        expr=m.x[3] + m.x[7] + m.x[8] + 2 * m.x[9] + m.x[10] == 1
    )
    return m, -47.707579, "HS112, log objective, poor start"


def unbounded_dual_ray():
    """Feasible bounded problem where the *initial* multipliers implied by the
    least-squares init are terrible: near-parallel active constraints with a
    tiny angle -> the multiplier estimates are ~1/eps and grow."""
    eps = 1e-8
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize={1: 10.0, 2: 10.0}, bounds=(-1e3, 1e3))
    m.obj = pyo.Objective(expr=m.x[1] + m.x[2])
    m.c1 = pyo.Constraint(expr=m.x[1] + m.x[2] >= 1.0)
    m.c2 = pyo.Constraint(expr=m.x[1] + (1 + eps) * m.x[2] >= 1.0)
    m.c3 = pyo.Constraint(expr=(1 + eps) * m.x[1] + m.x[2] >= 1.0)
    return m, 1.0, "near-parallel active constraints, 1e8 multiplier scale"


def hs013_warm_bad():
    """HS13 again but started far away with huge magnitudes -- stresses the
    early elevated-inf_du regime the guard counts in."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], bounds=(0, None), initialize={1: 1e4, 2: 1e4})
    m.obj = pyo.Objective(expr=(m.x[1] - 2) ** 2 + m.x[2] ** 2)
    m.c = pyo.Constraint(expr=(1 - m.x[1]) ** 3 - m.x[2] >= 0)
    return m, 1.0, "HS13 from x0=(1e4,1e4): inf_du starts ~1e12"


def steep_exp(K=200.0, n=1):
    """The ONLY family found that trips the guard at DEFAULT settings.
    min sum exp(x_i) s.t. x_i*y_i >= 1, 0 <= y_i <= 1/K  ==>  x_i >= K, so the
    objective gradient exp(x) ramps monotonically toward exp(200) ~ 1e86 while
    the iterates creep up the central path.  inf_du therefore grows for >=15
    consecutive iterations above 1e8 and the guard fires.  Bounded, feasible,
    finite optimum -- but NO solver (pounce with or without the guard, or
    Ipopt) converges on it, so the firing costs only extra restoration work."""
    m = pyo.ConcreteModel()
    I = range(n)
    m.x = pyo.Var(I, bounds=(0.0, 1000.0), initialize=1.0)
    m.y = pyo.Var(I, bounds=(0.0, 1.0 / K), initialize=1.0 / K)
    m.obj = pyo.Objective(expr=sum(pyo.exp(m.x[i]) for i in I))
    m.c = pyo.Constraint(I, rule=lambda m, i: m.x[i] * m.y[i] >= 1.0)
    return m, None, "steep exp ramp: 15+ consecutive growing inf_du (guard fires)"


PROBLEMS = [
    ("hs13", hs13),
    ("hs55", hs55),
    ("mpec_comp", mpec_comp),
    ("mpec_chain", mpec_chain),
    ("redundant_active", redundant_active),
    ("badscale_hess", badscale_hess),
    ("deg_eq_ineq", deg_eq_ineq),
    ("hs112", hs112),
    ("near_parallel", unbounded_dual_ray),
    ("hs13_bigstart", hs013_warm_bad),
    ("steep_exp_K200", steep_exp),
]


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------
def write_nl(m, name):
    stub = os.path.join(WORK, name)
    m.write(stub + ".nl", io_options={"symbolic_solver_labels": False})
    return stub


SOLVE_RESULT_RE = re.compile(r"solve_result_num[^0-9-]*(-?\d+)")


def run_solver(binary, stub, extra=()):
    """Run the solver on stub.nl.  Ipopt uses the AMPL stub convention."""
    if binary == IPOPT:
        cmd = [binary, stub, "-AMPL"] + list(extra)
    else:
        sol = stub + ".sol"
        if os.path.exists(sol):
            os.remove(sol)
        cmd = [binary, stub + ".nl", sol] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                       cwd=WORK)
    out = p.stdout + p.stderr
    return {"cmd": " ".join(cmd), "rc": p.returncode, "out": out}


ITER_RE = re.compile(
    r"^\s*(\d+)r?\s+([-\d.eE+]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s", re.M
)


def max_inf_du(out):
    """Largest inf_du (3rd column) seen in the iteration table."""
    best = 0.0
    n_grow = 0
    streak = 0
    prev = None
    for mm in ITER_RE.finditer(out):
        try:
            v = float(mm.group(4))
        except ValueError:
            continue
        best = max(best, v)
        if prev is not None and v > prev and v > 1e2:
            streak += 1
            n_grow = max(n_grow, streak)
        else:
            streak = 0
        prev = v
    return best, n_grow


def parse_ipopt(out):
    status = None
    obj = None
    m = re.search(r"EXIT: (.*)", out)
    if m:
        status = m.group(1).strip()
    m = re.search(r"Objective\.*:\s+\S+\s+(\S+)", out)
    if m:
        try:
            obj = float(m.group(1))
        except ValueError:
            pass
    it = re.search(r"Number of Iterations\.*:\s+(\d+)", out)
    return status, obj, (int(it.group(1)) if it else None)


def parse_pounce(out):
    """Scrape pounce's console output.  Note pounce auto-routes recognised
    convex QPs to its QP solver, which prints `POUNCE 0.9.0: <status>` and no
    `EXIT:` line -- handle both."""
    status = None
    obj = None
    it = None
    m = re.search(r"EXIT: (.*)", out)
    if m:
        status = m.group(1).strip()
    else:
        m = re.search(r"POUNCE [\d.]+: (.*)", out)
        if m:
            status = "[QP route] " + m.group(1).strip()
    m = re.search(r"Objective\.*:\s+\S+\s+(\S+)", out)
    if m:
        try:
            obj = float(m.group(1))
        except ValueError:
            pass
    mi = re.search(r"Number of Iterations\.*:\s+(\d+)", out)
    if mi:
        it = int(mi.group(1))
    return status, obj, it


def main():
    only = sys.argv[1:] or None
    results = []
    for name, fn in PROBLEMS:
        if only and name not in only:
            continue
        m, known, note = fn()
        stub = write_nl(m, name)

        ip = run_solver(IPOPT, stub, ["print_level=5"])
        ip_status, ip_obj, ip_it = parse_ipopt(ip["out"])

        rows = {"name": name, "note": note, "known": known,
                "ipopt": (ip_status, ip_obj, ip_it)}
        with open(f"{stub}.ipopt.log", "w") as fh:
            fh.write(ip["out"])

        for label, extra in [
            ("default", []),
            ("guard_off", ["dual_diverging_streak=0"]),
            ("guard_2", ["dual_diverging_streak=2"]),
            ("guard_1_noresto", ["dual_diverging_streak=1",
                                 "max_soft_resto_iters=0"]),
        ]:
            r = run_solver(POUNCE, stub, extra + ["print_level=5"])
            st, ob, itc = parse_pounce(r["out"])
            mx, gs = max_inf_du(r["out"])
            rows[label] = (st, ob, itc, f"max_inf_du={mx:.2e}",
                           f"max_growth_streak={gs}")
            with open(f"{stub}.{label}.log", "w") as fh:
                fh.write(r["out"])

        results.append(rows)
        print(json.dumps(rows, default=str))

    with open(os.path.join(WORK, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
