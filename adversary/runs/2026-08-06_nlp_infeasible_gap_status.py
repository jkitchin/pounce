"""Regression guard for gh#508: the exit status on an infeasible model, as a
function of the infeasibility gap.

Family: nlp   Class: equality-constrained, infeasible by construction
Target: the LocalInfeasibility / ErrorInStepComputation decision in
        `ipopt_alg.rs`, reached from the restoration-cycle exits.
Source: no published instance — a one-variable model whose infeasibility gap is
        an exact input parameter.
Status: found the defect (filed gh#508), fixed by 097a4719, now guards the fix.

# What this found, and what it now guards

gh#505 (David Bernal) reports a point that IPOPT accepts and POUNCE calls
locally infeasible, and lists gh#379 / gh#446 / gh#138 / gh#119 as the same
family: *a good or interpretable point returned under the wrong status*. That
family had been attacked several times from the "false infeasibility on a
feasible model" side, never from the other side — what POUNCE reports when the
model really **is** infeasible and the only question is which of two failure
statuses comes out.

The answer used to depend on `tol`, an option about KKT error, through a
`max(100·tol, 1e-4)` threshold that never consulted `constr_viol_tol` — the
option that declares what "violated" means. 20 of 60 cells returned
`solve_result_num = 500` (Pyomo `internalSolverError`) on a model that was
simply infeasible. Filed as gh#508; fixed by 097a4719, which asks the question
with `constr_viol_tol` instead.

The guard this file now enforces is the post-fix contract:

    a cell whose reported constraint violation is at least the declared
    `constr_viol_tol` must not come back with a 500+ status.

Below that threshold a 500 is deliberate — the violation is inside what the
user called feasible, so "converged to a point of local infeasibility" would be
the wrong claim and `Error in step computation` is the honest one. Those cells
are printed as `by design` and do not fail the run.

Sub-threshold, though, the two statuses are *not* handed out consistently: at
`tol=1e-8` a gap of 1e-9 returns 500 while 1e-8, 3e-8, 1e-7 and 3e-7 — all
equally inside `constr_viol_tol=1e-6` — return a clean 200. The two exits are
reached by different routes (the `ConvergenceStatus::LocallyInfeasible`
detector vs. the restoration-cycle exit) and only the second one consults
`constr_viol_tol`. That is a real inconsistency, but a smaller and different
one than gh#508, and this guard does not assert on it. See the report for why
it is left un-asserted rather than un-noticed.

# The model

    min (x - 5)^2   s.t.   x^2 + delta == 0

Infeasible for every delta > 0. `½‖c‖²` has a strict local minimum at `x = 0`
where the violation is exactly `delta` and `Jᵀc = 2x(x²+delta) = 0`, so the
feasibility-restoration phase converges there and no local move reduces the
violation. The honest verdict is "converged to a point of local infeasibility"
at every delta > 0, and `delta` is *exactly* the violation POUNCE reports —
which is what makes the threshold measurable to the digit.

# The oracle

IPOPT 3.14.19 on the identical `.nl` with identical options, classified off its
own `.sol` by the same `classify()` this file applies to POUNCE. It is not a
gold standard for the verdict — a solver may legitimately report a different
status on an infeasible model — and it is *not* clean here: on 7 of 60 cells
IPOPT's own `.sol` carries `solve_result_num = 501` (restoration failure),
which lands in the same 500+ band. So "the oracle never fails" is not a claim
this run supports, and neither the original finding nor this guard rests on it.
It is printed for contrast and never gates the exit code.

The load-bearing evidence was never the comparison. It was the shape of the
boundary, which is what section 2 measures: `constr_viol_tol` swept across four
orders of magnitude with `tol` held fixed. Before the fix the boundary did not
move at all; after it, it tracks `constr_viol_tol` exactly, which is the whole
point.

A feasible control family runs alongside, so a probe that reported "infeasible
everywhere" would be caught by its own controls rather than believed.

Usage:  python 2026-08-06_nlp_infeasible_gap_status.py
        POUNCE_BIN=... IPOPT_BIN=... python 2026-08-06_nlp_infeasible_gap_status.py
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[2]
POUNCE = os.environ.get("POUNCE_BIN", str(ROOT / "target" / "release" / "pounce"))
IPOPT = os.environ.get("IPOPT_BIN", "/opt/homebrew/bin/ipopt")

EXIT_RE = re.compile(r"EXIT: (.*)")
VIOL_RE = re.compile(r"Constraint violation\.*:\s+(\S+)\s+(\S+)")
ITER_RE = re.compile(r"Number of Iterations\.*:\s+(\d+)")

SHORT = {
    "Optimal Solution Found.": "optimal",
    "Solved To Acceptable Level.": "acceptable",
    "Feasible point for square problem found.": "square-feasible",
    "Converged to a point of local infeasibility. Problem may be infeasible.": "infeasible",
    "Error in step computation.": "ERROR-IN-STEP",
    "Maximum Number of Iterations Exceeded.": "max-iter",
    "Restoration Failed!": "resto-failed",
    "Search Direction is becoming Too Small.": "tiny-step",
}


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def infeasible_gap(delta):
    """min (x-5)^2 s.t. x^2 + delta == 0 — infeasible with gap exactly delta."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=2.0)
    m.obj = pyo.Objective(expr=(m.x - 5.0) ** 2)
    m.c = pyo.Constraint(expr=m.x**2 + delta == 0.0)
    return m


def feasible_control(n, k0, rho, h=0.25):
    """A feasible chain whose row coefficient grows geometrically.

    The control arm: the absolute residual on the late rows is floored by the
    row's own magnitude, which is gh#505's regime, and the correct answer is
    never `infeasible`. A probe whose infeasible arm fires everywhere would
    show up here.
    """
    m = pyo.ConcreteModel()
    idx = list(range(n))
    t = [1.0 + h * j + (0.4 if j % 3 == 0 else -0.2) for j in idx]
    m.x = pyo.Var(idx, initialize=lambda mm, j: 1.0 + h * j)
    m.obj = pyo.Objective(expr=sum((m.x[j] - t[j]) ** 4 for j in idx))

    def rule(mm, i):
        k = k0 * rho**i
        return k * mm.x[i + 1] - k * mm.x[i] == k * h

    m.c = pyo.Constraint(range(n - 1), rule=rule)
    return m


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def run(binary, nl, opts):
    """Solve and return (status, violation, iters, n_banners).

    `n_banners` matters: POUNCE's MC64 re-solve path runs the model a second
    time under a different scaling and prints a second `EXIT:` banner, and the
    two banners need not agree. The `.sol` carries only one verdict, so the
    banner a user reads on the console and the `solve_result_num` their
    modelling layer classifies on can differ. Reported rather than smoothed
    over; `solve_result_num` is the authoritative one (see `classify`).
    """
    stub = str(nl)[:-3]
    p = subprocess.run([binary, stub, "-AMPL"] + opts, capture_output=True,
                       text=True, timeout=600, cwd=str(nl.parent))
    out = p.stdout + p.stderr
    ex = EXIT_RE.findall(out)
    cv = VIOL_RE.findall(out)
    it = ITER_RE.findall(out)
    raw = ex[-1].strip() if ex else f"<no EXIT rc={p.returncode}>"
    return (SHORT.get(raw, raw[:28]),
            float(cv[-1][-1]) if cv else float("nan"),
            int(it[-1]) if it else -1,
            len(ex))


def solve_result_num(nl):
    """The `objno <k> <n>` line the .sol carries — what AMPL/Pyomo classify on."""
    sol = Path(str(nl)[:-3] + ".sol")
    if not sol.exists():
        return None
    m = re.findall(r"objno\s+\d+\s+(\d+)", sol.read_text(errors="replace"))
    return int(m[-1]) if m else None


def write(model, nl):
    model.write(str(nl), io_options={"symbolic_solver_labels": False})


def classify(srn, banner):
    """What a modelling layer sees. `solve_result_num` wins where it exists.

    AMPL bands: 0-99 solved, 100-199 solved-approximately, 200-299 infeasible,
    300-399 unbounded, 400-499 limit, 500+ failure. Pyomo maps 500+ to
    `internalSolverError`.
    """
    if srn is None:
        return banner
    if srn >= 500:
        return "FAILURE(%d)" % srn
    if 200 <= srn < 300:
        return "infeasible"
    if 400 <= srn < 500:
        return "limit"
    if srn < 200:
        return "solved"
    return "other(%d)" % srn


DELTAS = [1e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
          1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="adv-gap-"))
    failures = []
    by_design = []
    oracle_failures = []
    banner_splits = []

    # ---- 1. the gap sweep, at four values of `tol` ----------------------
    print("=" * 78)
    print("1. exit status vs infeasibility gap.  threshold should be "
          "constr_viol_tol=1e-6")
    print("=" * 78)
    for tol in ["1e-8", "1e-6", "1e-5", "1e-4"]:
        cvt = 1e-6
        opts = ["max_iter=3000", f"tol={tol}", "acceptable_tol=1e-12",
                f"constr_viol_tol={cvt:.0e}", "nlp_scaling_method=none"]
        print(f"\n  tol={tol}   constr_viol_tol={cvt:.0e}")
        print(f"  {'delta':>9}  {'pounce banner':<16} {'seen':>12} {'cv':>10} "
              f"{'it':>5}   {'ipopt banner':<16} {'seen':>12} {'cv':>10}")
        for d in DELTAS:
            nl = tmp / f"gap_{tol}_{d:.0e}.nl"
            write(infeasible_gap(d), nl)
            pe, pcv, pit, nban = run(POUNCE, nl, opts)
            srn = solve_result_num(nl)
            seen = classify(srn, pe)
            # The oracle is classified exactly the same way, off its own .sol.
            # Reading POUNCE's `objno` code against IPOPT's console banner would
            # be two different measurements dressed as one comparison.
            ie, icv, _, _ = run(IPOPT, nl, opts)
            iseen = classify(solve_result_num(nl), ie)
            mark = ""
            if seen.startswith("FAILURE"):
                # Post-gh#508 contract: a 500 is only wrong when the violation
                # POUNCE itself reports is at least what the user declared
                # feasible. Below that the model is infeasible by less than
                # `constr_viol_tol`, and refusing to certify local
                # infeasibility is the honest answer, not a defect.
                if pcv >= cvt:
                    mark = "  <-- solver-failure status on an infeasible model"
                    failures.append((tol, d, pe, srn, ie, iseen, pcv, cvt))
                else:
                    mark = f"  (by design: cv {pcv:.0e} < constr_viol_tol)"
                    by_design.append((tol, d, pcv, cvt))
            elif nban > 1 and pe != seen:
                # The MC64 re-solve prints a second banner. Two banners is
                # normal; the two DISAGREEING is the finding.
                mark = f"  <-- console says '{pe}', .sol says '{seen}'"
                banner_splits.append((tol, d, pe, seen))
            if iseen.startswith("FAILURE"):
                oracle_failures.append((tol, d, ie, iseen))
            print(f"  {d:>9.0e}  {pe:<16} {seen:>12} {pcv:>10.2e} "
                  f"{pit:>5}   {ie:<16} {iseen:>12} {icv:>10.2e}{mark}")

    # ---- 2. constr_viol_tol does not move the boundary ------------------
    print()
    print("=" * 78)
    print("2. the same three gaps under four constr_viol_tol values (tol fixed)")
    print("   the boundary is a statement about feasibility, so it must move")
    print("   here — and must move nowhere else.")
    print("=" * 78)
    print(f"  {'constr_viol_tol':>16}  " + "".join(f"{d:>16.0e}" for d in
                                                   (1e-5, 3e-5, 1e-4)))
    for cvts in ["1e-8", "1e-6", "1e-5", "1e-3"]:
        cvt = float(cvts)
        opts = ["max_iter=3000", "tol=1e-6", "acceptable_tol=1e-12",
                f"constr_viol_tol={cvts}", "nlp_scaling_method=none"]
        row = []
        for d in (1e-5, 3e-5, 1e-4):
            nl = tmp / f"cvt_{cvts}_{d:.0e}.nl"
            write(infeasible_gap(d), nl)
            banner, pcv, _, _ = run(POUNCE, nl, opts)
            srn = solve_result_num(nl)
            seen = classify(srn, banner)
            if seen.startswith("FAILURE"):
                # Same contract as section 1, applied per row's own cvt.
                if pcv >= cvt:
                    failures.append((f"cvt={cvts}", d, banner, srn, "-", "-",
                                     pcv, cvt))
                else:
                    by_design.append((f"cvt={cvts}", d, pcv, cvt))
            row.append(seen)
        print(f"  {cvts:>16}  " + "".join(f"{s:>16}" for s in row))

    # ---- 3. feasible control arm ---------------------------------------
    print()
    print("=" * 78)
    print("3. control: FEASIBLE models in the same residual regime")
    print("=" * 78)
    opts = ["max_iter=3000", "tol=1e-6", "acceptable_tol=1e-3",
            "constr_viol_tol=1e-6", "mu_strategy=adaptive",
            "bound_relax_factor=1e-8", "nlp_scaling_method=none"]
    print(f"  {'n':>4} {'K0':>8} {'rho':>5} {'Kmax':>9}  "
          f"{'pounce':<16} {'cv':>10}   {'ipopt':<16}")
    control_bad = []
    for n, k0, rho in [(20, 1e0, 1.0), (20, 1e6, 1.0), (20, 1e10, 1.0),
                       (20, 1e12, 1.0), (20, 1e13, 1.0), (20, 1e14, 1.0),
                       (36, 1e4, 2.0)]:
        nl = tmp / f"ctl_{n}_{k0:.0e}_{rho}.nl"
        write(feasible_control(n, k0, rho), nl)
        pe, pcv, _, _ = run(POUNCE, nl, opts)
        pe = classify(solve_result_num(nl), pe)
        ie, _, _, _ = run(IPOPT, nl, opts)
        bad = "  <-- FALSE INFEASIBLE" if pe == "infeasible" else ""
        if bad:
            control_bad.append((n, k0, rho))
        print(f"  {n:>4} {k0:>8.0e} {rho:>5} {k0 * rho ** (n - 2):>9.1e}  "
              f"{pe:<16} {pcv:>10.2e}   {ie:<16}{bad}")

    # ---- verdict --------------------------------------------------------
    print()
    print("=" * 78)
    if control_bad:
        print(f"CONTROL FAILURE: {len(control_bad)} feasible models "
              f"reported infeasible — {control_bad}")
    if failures:
        print(f"FAIL: {len(failures)} cells returned a solver-failure status "
              f"while violating constr_viol_tol (gh#508 regression).")
        for tol, d, pe, srn, ie, iseen, pcv, cvt in failures:
            print(f"  {tol} delta={d:.0e}: pounce={pe} "
                  f"(solve_result_num={srn}), cv={pcv:.2e} >= "
                  f"constr_viol_tol={cvt:.0e}; ipopt={ie} -> {iseen}")
    else:
        print("PASS: every cell violating constr_viol_tol returned an "
              "interpretable status (gh#508 fix holds).")
    if by_design:
        print(f"\n{len(by_design)} cells returned 500 with the violation "
              f"*inside* constr_viol_tol — deliberate, not counted:")
        for tol, d, pcv, cvt in by_design:
            print(f"  {tol} delta={d:.0e}: cv={pcv:.2e} < "
                  f"constr_viol_tol={cvt:.0e}")
    if oracle_failures:
        # Stated even though it weakens the headline: if the oracle also
        # returns 500+ on some cells, "IPOPT never fails here" is not a claim
        # this run supports, and the finding has to rest on the tol-driven
        # boundary instead.
        print(f"\nOracle failures ({len(oracle_failures)} cells where IPOPT's "
              f"own .sol carries a 500+ code):")
        for tol, d, ie, iseen in oracle_failures:
            print(f"  tol={tol} delta={d:.0e}: ipopt={ie} -> {iseen}")
    if banner_splits:
        print(f"\nAlso: {len(banner_splits)} cells printed more than one EXIT "
              f"banner, with the console text disagreeing with the .sol status:")
        for tol, d, pe, seen in banner_splits:
            print(f"  tol={tol} delta={d:.0e}: console '{pe}', .sol '{seen}'")
    print(f"files: {tmp}")
    return 1 if (failures or control_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
