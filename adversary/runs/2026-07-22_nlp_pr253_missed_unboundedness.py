"""Adversary cross-check: does the #248/#252 divergence guard MISS real unboundedness?

Family: nlp   Class: unboundedness detection (false negative / regression risk)
Targets: PR #249 (issue #248) and PR #253 (issue #252).

BACKGROUND. Those two PRs progressively narrowed when POUNCE will report
DivergingIterates (its unboundedness verdict). A step now counts toward the
firing streak only if ALL THREE hold:

    growing     : |x|_inf >= 2 * previous |x|_inf
    descending  : f dropped since the previous step
    keeping_up  : this drop >= 0.9 * previous drop      <-- added by #253

and the streak must reach 4 CONSECUTIVE qualifying steps. Critically, a step
that fails any condition sets the streak back to ZERO (not a decrement):

    if growing && descending && keeping_up { streak += 1 } else { streak = 0 }

ADVERSARIAL HYPOTHESIS. On a genuine recession ray the per-step drop ratio is
not smooth -- IPM step lengths are set by the fraction-to-the-boundary rule and
the filter line search, so they stutter. If the ratio dips below 0.9 even once
every few iterations, the streak can NEVER reach 4, and a genuinely unbounded
problem is silently reported as solved/limit instead of unbounded. The tighter
the ill-conditioning, the more the steps stutter.

This is the dangerous direction of error. A spurious UNBOUNDED (the bug #248/
#252/#257 fixed) is loud and a driver can retry. A MISSED unboundedness is
silent: a branch-and-bound driver fathoms the node on a finite "bound" that does
not exist, and returns a wrong global optimum.

DESIGN. A family of problems that are unbounded below along a LINEAR recession
ray -- so the premise of #253's rule ("a genuine ray's drop keeps up") is
satisfied in exact arithmetic and the guard SHOULD fire -- but with jit1-scale
ill-conditioning to make the realised steps stutter:

    min  sum_i c_i / x_i  -  sum_i d_i * x_i        s.t.  x_i >= lo_i,  x_i <= +inf

with c, d spanning many orders of magnitude. The -d_i*x_i tail drives f -> -inf
linearly. Sweeping the conditioning gives many independent samples.

Run at a low diverging_iterates_tol -- the setting a branch-and-bound driver
uses, and the exact configuration #248/#252/#253 were tuned for.

ORACLE: Ipopt 3.x on the identical .nl. Ipopt reporting Diverging_Iterates /
an unbounded-band status while POUNCE reports solved is a missed unboundedness.
"""

import os
import subprocess
import time

import pyomo.environ as pyo

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
OUT = os.path.dirname(os.path.abspath(__file__))
DIVTOL = "1e6"  # what a B&B driver sets to abort runaway nodes


def build(n, cond):
    """min sum c_i/x_i - sum d_i x_i, x_i >= lo_i, no upper bound.

    `cond` sets how many orders of magnitude c and d span.
    """
    m = pyo.ConcreteModel()
    idx = list(range(n))
    c = {i: 10.0 ** (((i % 7) - 3) * cond / 3.0) for i in idx}
    d = {i: 10.0 ** ((i % 8) * cond / 8.0) for i in idx}
    lo = {i: (1e-3 if i % 3 else 1e-6) for i in idx}
    m.I = pyo.Set(initialize=idx)
    m.x = pyo.Var(m.I, bounds=lambda mm, i: (lo[i], None), initialize=lambda mm, i: 1.0)
    m.obj = pyo.Objective(
        expr=sum(c[i] / m.x[i] for i in idx) - sum(d[i] * m.x[i] for i in idx),
        sense=pyo.minimize,
    )
    # One coupling constraint so the problem is not separable.
    m.c1 = pyo.Constraint(expr=sum(m.x[i] for i in idx) >= 1e-3)
    return m


def run(binary, nl, extra):
    sol = nl.replace(".nl", ".sol")
    if os.path.exists(sol):
        os.remove(sol)
    t0 = time.perf_counter()
    p = subprocess.run(
        [binary, nl, "-AMPL", *extra], capture_output=True, text=True, timeout=120
    )
    dt = time.perf_counter() - t0
    res = None
    if os.path.exists(sol):
        for line in reversed(open(sol).read().strip().splitlines()):
            if line.startswith("objno"):
                res = int(line.split()[-1])
                break
    exits = [l.strip() for l in (p.stdout + p.stderr).splitlines() if l.startswith("EXIT")]
    return res, dt, (exits[0] if exits else "")


def band(res):
    if res is None:
        return "none"
    if 0 <= res < 100:
        return "SOLVED"
    if 200 <= res < 300:
        return "INFEAS"
    if 300 <= res < 400:
        return "UNBOUNDED"
    return "LIMIT/FAIL"


def main():
    print(f"{'n':>4} {'cond':>5} | {'pounce':<12} {'ipopt':<12} | pounce EXIT")
    print("-" * 88)
    missed, agree, rows = [], 0, []
    for n in (5, 12, 25):
        for cond in (1, 3, 5, 7, 9):
            m = build(n, cond)
            nl = os.path.join(OUT, f"_pr253_ub_n{n}_c{cond}.nl")
            m.write(nl, io_options={"symbolic_solver_labels": False})
            r_p, t_p, e_p = run(POUNCE, nl, [f"diverging_iterates_tol={DIVTOL}"])
            r_i, t_i, e_i = run(IPOPT, nl, [f"diverging_iterates_tol={DIVTOL}"])
            bp, bi = band(r_p), band(r_i)
            rows.append((n, cond, bp, bi, e_p, e_i))
            print(f"{n:>4} {cond:>5} | {bp:<12} {bi:<12} | {e_p}")
            if bi == "UNBOUNDED" and bp != "UNBOUNDED":
                missed.append((n, cond, bp, bi, e_p, e_i))
            elif bi == bp:
                agree += 1
            os.remove(nl)

    print()
    print(f"agree: {agree}/{len(rows)}   missed-unboundedness: {len(missed)}")
    for n, cond, bp, bi, e_p, e_i in missed:
        print(f"  MISSED n={n} cond=1e{cond}: pounce={bp} ({e_p}) vs ipopt={bi} ({e_i})")
    print()
    print("VERDICT: FAIL (missed unboundedness)" if missed else "VERDICT: PASS")


if __name__ == "__main__":
    main()
