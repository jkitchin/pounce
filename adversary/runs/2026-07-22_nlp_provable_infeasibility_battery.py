"""Adversary: provably-infeasible NLP battery + feasible-but-thin controls.

Family: nlp   Class: infeasibility / status-reporting correctness

Every INFEASIBLE case below has a one-line analytic proof of emptiness that does
not depend on any solver (bounded trig range, AM-GM, triangle inequality, sum of
squares, log monotonicity, Motzkin positivity).  Requirement: POUNCE must report
infeasibility (AMPL solve_result_num in 200..299) and must NOT report `solved`.

Every FEASIBLE case has an explicitly exhibited feasible point, so a report of
infeasibility is a *false* infeasibility claim -- the worst status bug for a
branch-and-bound driver, which silently prunes a node containing the optimum.
The controls are all deliberately thin (lens of width 1e-6, quartic trough of
half-width 7e-5, a 1e-6-wide curved tube, a 1e-9 wedge) so that a too-eager
detector rejects them.

This goes beyond the prior runs on this axis
(2026-07-22_nlp_false_infeasible_*.py, _infeasibility_still_detected.py), which
covered contradictory cubics/equalities, disjoint circles, a bound
contradiction, and the HS13 constraint-scaling false positive.  New here:
non-polyhedral emptiness certificates (trig range, AM-GM, triangle inequality,
Motzkin), narrow-*margin* infeasibility (empty by only 1e-3, well above
constr_viol_tol), and a two-sided tangency sweep that locates the exact overlap
width at which the feasible/infeasible verdict flips.

Oracle: Ipopt 3.14 on the identical .nl file, plus the analytic proof.
"""

import math
import os
import shutil
import subprocess
import tempfile

import pyomo.environ as pyo

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"

# ---------------------------------------------------------------------------
# INFEASIBLE cases -- each with an analytic emptiness proof
# ---------------------------------------------------------------------------


def trig_range():
    """sin(x)+sin(y) >= 2.5 is empty: each term is <= 1, so the sum is <= 2."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-10, 10), initialize=1.0)
    m.y = pyo.Var(bounds=(-10, 10), initialize=1.0)
    m.obj = pyo.Objective(expr=m.x + m.y)
    m.c = pyo.Constraint(expr=pyo.sin(m.x) + pyo.sin(m.y) >= 2.5)
    return m


def amgm_exp():
    """exp(x)+exp(-x) >= 2 by AM-GM, so <= 1.5 with x+y==0 is empty."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-5, 5), initialize=0.3)
    m.y = pyo.Var(bounds=(-5, 5), initialize=-0.3)
    m.obj = pyo.Objective(expr=m.x**2)
    m.c = pyo.Constraint(expr=pyo.exp(m.x) + pyo.exp(m.y) <= 1.5)
    m.e = pyo.Constraint(expr=m.x + m.y == 0)
    return m


def circle_line():
    """x^2+y^2==1 caps x+y at sqrt(2)~1.414, so x+y==3 is empty."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-10, 10), initialize=0.6)
    m.y = pyo.Var(bounds=(-10, 10), initialize=0.8)
    m.obj = pyo.Objective(expr=m.x)
    m.c1 = pyo.Constraint(expr=m.x**2 + m.y**2 == 1.0)
    m.c2 = pyo.Constraint(expr=m.x + m.y == 3.0)
    return m


def triangle_inequality():
    """|a-c| <= |a-b|+|b-c| <= 2 contradicts |a-c| >= 5.  6 variables."""
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 1)
    m.a = pyo.Var(m.I, bounds=(-20, 20), initialize=0.0)
    m.b = pyo.Var(m.I, bounds=(-20, 20), initialize=0.5)
    m.c = pyo.Var(m.I, bounds=(-20, 20), initialize=1.0)
    m.obj = pyo.Objective(expr=sum(m.a[i] ** 2 for i in m.I))
    m.ab = pyo.Constraint(expr=sum((m.a[i] - m.b[i]) ** 2 for i in m.I) <= 1.0)
    m.bc = pyo.Constraint(expr=sum((m.b[i] - m.c[i]) ** 2 for i in m.I) <= 1.0)
    m.ac = pyo.Constraint(expr=sum((m.a[i] - m.c[i]) ** 2 for i in m.I) >= 25.0)
    return m


def log_domain():
    """log(x) >= 5 needs x >= e^5 ~ 148, contradicting x <= 2."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1e-6, 2.0), initialize=1.0)
    m.obj = pyo.Objective(expr=m.x)
    m.c = pyo.Constraint(expr=pyo.log(m.x) >= 5.0)
    return m


def quartic_margin():
    """(x^2-1/2)^2 >= 0 everywhere, so <= -1e-3 is empty by margin 1e-3.

    The margin is 10x the usual constr_viol_tol of 1e-4, so calling this
    `solved` is a genuine false claim, not a tolerance artifact.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-3, 3), initialize=0.9)
    m.obj = pyo.Objective(expr=m.x)
    m.c = pyo.Constraint(expr=(m.x**2 - 0.5) ** 2 <= -1e-3)
    return m


def mass_balance():
    """n1+n2==1 caps n1*n2 at 1/4 (AM-GM), so n1*n2==0.5 is empty."""
    m = pyo.ConcreteModel()
    m.n1 = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.n2 = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.obj = pyo.Objective(expr=m.n1)
    m.mass = pyo.Constraint(expr=m.n1 + m.n2 == 1.0)
    m.equil = pyo.Constraint(expr=m.n1 * m.n2 == 0.5)
    return m


def motzkin():
    """Motzkin: x^4y^2+x^2y^4-3x^2y^2+1 >= 0 (AM-GM), so <= -0.5 is empty.

    Positive but not a sum of squares -- no SOS/convex certificate exists.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-3, 3), initialize=1.0)
    m.y = pyo.Var(bounds=(-3, 3), initialize=1.0)
    m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
    m.c = pyo.Constraint(
        expr=m.x**4 * m.y**2 + m.x**2 * m.y**4 - 3 * m.x**2 * m.y**2 + 1 <= -0.5
    )
    return m


def norm_sandwich():
    """||z||^2 <= 1 and sum(z) >= 4 with n=4: Cauchy-Schwarz caps sum at 2."""
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 3)
    m.z = pyo.Var(m.I, bounds=(-5, 5), initialize=0.4)
    m.obj = pyo.Objective(expr=sum(m.z[i] for i in m.I))
    m.ball = pyo.Constraint(expr=sum(m.z[i] ** 2 for i in m.I) <= 1.0)
    m.sum = pyo.Constraint(expr=sum(m.z[i] for i in m.I) >= 4.0)
    return m


# ---------------------------------------------------------------------------
# FEASIBLE-BUT-THIN controls -- a feasible point is exhibited for each
# ---------------------------------------------------------------------------


def thin_lens():
    """Two unit disks at +-d, d = 1-5e-7: overlap lens of width 1e-6.

    Feasible point: origin, where both constraints read d^2 = 1-1e-6 < 1.
    """
    d = 1.0 - 5e-7
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-5, 5), initialize=0.0)
    m.y = pyo.Var(bounds=(-5, 5), initialize=0.0)
    m.obj = pyo.Objective(expr=m.y)
    m.c1 = pyo.Constraint(expr=(m.x + d) ** 2 + m.y**2 <= 1.0)
    m.c2 = pyo.Constraint(expr=(m.x - d) ** 2 + m.y**2 <= 1.0)
    return m


def quartic_trough():
    """(x^2-1/2)^2 <= 1e-8: two intervals of half-width ~7e-5 about +-1/sqrt2."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-3, 3), initialize=0.9)
    m.obj = pyo.Objective(expr=m.x)
    m.c = pyo.Constraint(expr=(m.x**2 - 0.5) ** 2 <= 1e-8)
    return m


def curved_tube():
    """(y-sin x)^2 <= 1e-12 with y >= 0.5: a 1e-6-wide tube along y=sin x.

    Feasible point: x = pi/2, y = 1.  Optimum x -> pi/6 = 0.5235987756.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10), initialize=math.pi / 2)
    m.y = pyo.Var(bounds=(0, 2), initialize=1.0)
    m.obj = pyo.Objective(expr=m.x)
    m.tube = pyo.Constraint(expr=(m.y - pyo.sin(m.x)) ** 2 <= 1e-12)
    m.floor = pyo.Constraint(expr=m.y >= 0.5)
    return m


def steep_sliver():
    """Badly scaled (Jacobian ~1e8) yet feasible: x=1000, y=1e-4 satisfies all."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1e4), initialize=5e3)
    m.y = pyo.Var(bounds=(0, 1e4), initialize=5e3)
    m.obj = pyo.Objective(expr=(m.x - 2.0) ** 2 + m.y)
    m.eq = pyo.Constraint(expr=m.x**3 + m.y**3 == 1e9)
    m.sliver = pyo.Constraint(expr=m.y <= 1e-6 * m.x)
    return m


def narrow_wedge():
    """x^2 <= y <= x^2+1e-9 on [-1,1]: a 1e-9 band, optimum y=0 at x=0."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-1, 1), initialize=0.7)
    m.y = pyo.Var(bounds=(-1, 2), initialize=0.49)
    m.obj = pyo.Objective(expr=m.y)
    m.lo = pyo.Constraint(expr=m.y >= m.x**2)
    m.hi = pyo.Constraint(expr=m.y <= m.x**2 + 1e-9)
    return m


INFEASIBLE = {
    "trig_range": trig_range,
    "amgm_exp": amgm_exp,
    "circle_line": circle_line,
    "triangle_inequality": triangle_inequality,
    "log_domain": log_domain,
    "quartic_margin": quartic_margin,
    "mass_balance": mass_balance,
    "motzkin": motzkin,
    "norm_sandwich": norm_sandwich,
}

FEASIBLE = {
    "thin_lens": thin_lens,
    "quartic_trough": quartic_trough,
    "curved_tube": curved_tube,
    "steep_sliver": steep_sliver,
    "narrow_wedge": narrow_wedge,
}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

WORK = tempfile.mkdtemp(prefix="adv_infeas_")


def band(res):
    """AMPL solve_result_num band."""
    if res is None:
        return "NONE"
    if res < 100:
        return "SOLVED"
    if res < 200:
        return "ACCEPTABLE"
    if res < 300:
        return "INFEASIBLE"
    if res < 400:
        return "UNBOUNDED"
    if res < 500:
        return "LIMIT"
    return "FAILURE"


def run(binary, model, tag, extra=()):
    d = os.path.join(WORK, tag)
    os.makedirs(d, exist_ok=True)
    nl = os.path.join(d, "m.nl")
    model.write(nl, io_options={"symbolic_solver_labels": False})
    sol = os.path.join(d, "m.sol")
    if os.path.exists(sol):
        os.remove(sol)
    try:
        p = subprocess.run(
            [binary, nl, "-AMPL", *extra],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=d,
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", ""
    res = None
    if os.path.exists(sol):
        for line in reversed(open(sol).read().strip().splitlines()):
            if line.startswith("objno"):
                res = int(line.split()[-1])
                break
    txt = p.stdout + p.stderr
    exit_line = ""
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("EXIT"):
            exit_line = s
    return res, band(res), exit_line


def main():
    findings = []
    print("=" * 78)
    print("PART 1 -- PROVABLY INFEASIBLE (pounce must NOT say SOLVED)")
    print("=" * 78)
    print(f"{'case':<24} {'pounce':<12} {'ipopt':<12} {'ok':<4} exit")
    for name, build in INFEASIBLE.items():
        pr, pb, pe = run(POUNCE, build(), f"p_{name}", ("max_wall_time=8",))
        ir, ib, _ = run(IPOPT, build(), f"i_{name}", ("max_wall_time=8",))
        ok = pb in ("INFEASIBLE",)
        if pb == "SOLVED":
            findings.append(f"FALSE OPTIMAL on infeasible '{name}' (objno={pr})")
        elif not ok:
            findings.append(f"non-infeasible status on infeasible '{name}': {pb}")
        print(f"{name:<24} {pb+f'({pr})':<12} {ib+f'({ir})':<12} {'y' if ok else 'N':<4} {pe[:40]}")

    print()
    print("=" * 78)
    print("PART 2 -- FEASIBLE BUT THIN (pounce must NOT say INFEASIBLE)")
    print("=" * 78)
    print(f"{'case':<24} {'pounce':<12} {'ipopt':<12} {'ok':<4} exit")
    for name, build in FEASIBLE.items():
        pr, pb, pe = run(POUNCE, build(), f"p_{name}", ("max_wall_time=8",))
        ir, ib, _ = run(IPOPT, build(), f"i_{name}", ("max_wall_time=8",))
        ok = pb not in ("INFEASIBLE", "UNBOUNDED")
        if pb == "INFEASIBLE":
            msg = f"FALSE INFEASIBLE on feasible '{name}' (objno={pr})"
            if ib == "INFEASIBLE":
                msg += " [ipopt agrees -- likely model/tolerance, not pounce]"
            findings.append(msg)
        elif not ok:
            findings.append(f"bad status on feasible '{name}': {pb}")
        print(f"{name:<24} {pb+f'({pr})':<12} {ib+f'({ir})':<12} {'y' if ok else 'N':<4} {pe[:40]}")

    print()
    print("=" * 78)
    print("PART 3 -- TANGENCY SWEEP: two unit disks at +-d")
    print("  gap = 2*(1-d) > 0 -> FEASIBLE (lens of that width)")
    print("  gap < 0          -> INFEASIBLE (disjoint by |gap|)")
    print("=" * 78)
    print(f"{'gap':<12} {'truth':<12} {'pounce':<12} {'ipopt':<12} ok")
    for mag in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        for sgn in (+1, -1):
            gap = sgn * mag
            d = 1.0 - gap / 2.0
            truth = "FEASIBLE" if gap > 0 else "INFEASIBLE"

            def mk(d=d):
                m = pyo.ConcreteModel()
                m.x = pyo.Var(bounds=(-5, 5), initialize=0.0)
                m.y = pyo.Var(bounds=(-5, 5), initialize=0.0)
                m.obj = pyo.Objective(expr=m.y)
                m.c1 = pyo.Constraint(expr=(m.x + d) ** 2 + m.y**2 <= 1.0)
                m.c2 = pyo.Constraint(expr=(m.x - d) ** 2 + m.y**2 <= 1.0)
                return m

            tag = f"g{gap:+.0e}".replace("+", "p").replace("-", "m").replace(".", "")
            pr, pb, _ = run(POUNCE, mk(), f"p_sw_{tag}", ("max_wall_time=8",))
            ir, ib, _ = run(IPOPT, mk(), f"i_sw_{tag}", ("max_wall_time=8",))
            if truth == "FEASIBLE":
                ok = pb not in ("INFEASIBLE", "UNBOUNDED")
            else:
                ok = pb == "INFEASIBLE"
            # gaps at/below 1e-6 are inside constr_viol_tol -- informational only
            informational = mag <= 1e-5
            mark = "y" if ok else ("~" if informational else "N")
            if not ok and not informational and pb == "INFEASIBLE":
                findings.append(f"FALSE INFEASIBLE on feasible tangency gap={gap:+.0e}")
            print(f"{gap:<+12.0e} {truth:<12} {pb+f'({pr})':<12} {ib+f'({ir})':<12} {mark}")

    print()
    print("=" * 78)
    if findings:
        print("FINDINGS:")
        for f in findings:
            print("  -", f)
        print("VERDICT: FAIL")
    else:
        print("no false status claims detected")
        print("VERDICT: PASS")
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
