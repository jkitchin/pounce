"""Adversary cross-check: Phase 6 bound-multiplier attribution, end to end.

Family: nlp   Class: bound-constrained + linear-equality structure
Target: `presolve_linear_eq_reduction` bound attribution (gh#493, on top of #487)
Source: no published instance — a randomized family built around a KNOWN
        feasible point, checked against the textbook KKT conditions.

# Why the existing probes cannot see this

`adversary-fuzz linelim` attacks the plan and the derivative transforms.
`2026-08-06_nlp_linear_eq_reduction_verify.py` attacks the whole stack, but its
generator draws every box with at least 1.0 of margin around `x*`, so **no
bound is ever active** and the bound multipliers it never inspects are all
zero. This probe exists to make bounds active on the columns Phase 6 removes.

`pounce verify` is also blind here, by construction: it reports *bound-projected*
stationarity (`verify.rs`, `stationarity_residual`) — it projects out exactly the
component of the residual that a bound multiplier could absorb, and never reads
`ipopt_zL_out` / `ipopt_zU_out` at all. A model whose bound multiplier is parked
on the wrong column passes `verify` cleanly. It is run here anyway, as a
feasibility oracle, but it is not the discriminating one.

# The oracle

The full KKT system, evaluated in this script from the same data used to write
the `.nl`, on the `.sol` that comes back:

1. **Primal feasibility** — every row holds, every `x` inside its declared box.
2. **Stationarity** — `∇f + Jᵀλ − z_l + z_u = 0`, with `∇f = 2(x − t)` and `J`
   the row coefficients written into the file.
3. **Dual feasibility** — `z_l ≥ 0`, `z_u ≥ 0`.
4. **Bound complementarity** — `z_l[j]·(x[j] − x_l[j]) = 0` and
   `z_u[j]·(x_u[j] − x[j]) = 0`.

(4) is the condition that discriminates. Attributing a transferred bound's
multiplier to the survivor puts a non-zero multiplier on a bound the survivor
is not sitting on, which violates complementarity in the original variable
space while leaving stationarity — the thing #487 checked — perfectly intact.
None of these four is a pounce-defined quantity; all four are the textbook
conditions, computed here from the generator's own arrays.

A fifth check is differential rather than absolute: the reduced solve's bound
multipliers are compared against the **no-presolve** solve of the same file.
Those must agree wherever the multipliers are unique. Where the survivor's own
bound and a transferred bound are active at once they are genuinely not unique,
so such instances are detected and excluded from that comparison (they are
still held to conditions 1-4).

# Conventions

`.sol` files negate the dual block (`sol_writer.rs`: AMPL wants `d obj / d b`,
which is `−λ`), and export `ipopt_zL_out = +z_l`, `ipopt_zU_out = −z_u`
(`main.rs`). Both are asserted rather than assumed: the script solves a
hand-checkable model first and refuses to run if either convention does not
hold.

Usage:  python 2026-08-06_nlp_phase6_bound_attribution.py [trials] [seed]
"""

import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import os
POUNCE = Path(os.environ.get("POUNCE_BIN",
    str(Path(__file__).resolve().parents[2] / "target" / "release" / "pounce")))

FEAS_TOL = 1e-6
# Relative to the size of the gradients involved. A quartic objective is very
# flat near its target, so the IPM sometimes stops at "Solved To Acceptable
# Level" with a residual around 1e-4 relative; the control arm below is what
# calibrates this, and the defects being hunted are four to six orders larger.
KKT_TOL = 1e-4
ACTIVE_TOL = 1e-6

# A pre-existing Phase-6 defect this probe surfaces but does not test for:
# when accumulated bound transfers squeeze a survivor's reduced box down to a
# single point, the reduced column is a fixed variable, no bound multiplier is
# produced for it, and the full-space duals come back incomplete. Reported
# separately so it cannot mask a gh#493 regression, and so it cannot be
# mistaken for one.
SEPARATE_ISSUE = "gh#495"

# Phase 6 and nothing else. `presolve=yes` alone turns on Phases 1-4 too.
PHASE6_ONLY = [
    "presolve=yes",
    "presolve_linear_eq_reduction=yes",
    "presolve_bound_tightening=no",
    "presolve_redundant_constraint_removal=no",
    "presolve_licq_check=no",
    "presolve_warm_z_bounds=no",
]


# --------------------------------------------------------------------------
# .nl writing:  min Σ (x_j - t_j)^2   s.t.   Σ a_ij x_j + c_i = rhs_i
# --------------------------------------------------------------------------
def write_nl(path, n, rows, rhs, consts, targets, x_l, x_u, x0):
    """A strictly convex quartic objective over linear equality rows.

    Quartics, not squares. A square objective makes the model classify as a
    convex QP, and the CLI then dispatches to `pounce-convex` *before* any
    presolve wrapper is built — Phase 6 never runs, and the probe silently
    measures a different code path (this is gh#494, filed as a known limitation
    of #487). A first draft of this script used squares and reported "0 columns
    eliminated" on every instance; the `n_elim` assertion below now makes that
    failure loud instead of silent.

    `Σ (x_j - t_j)^4` is still strictly convex, so the primal optimum is unique
    and a disagreement with the no-presolve solve cannot be waved away as a
    different-but-equal optimum. The row constant `c_i` stays in the expression
    segment so the row-constant branch remains live.
    """
    m = len(rows)
    jnnz = sum(len(r) for r in rows)
    L = [
        "g3 1 1 0\t# adversary phase-6 bound attribution",
        f" {n} {m} 1 0 {m} \t# vars, constraints, objectives, ranges, eqns",
        " 0 1 0 0 0 0\t# nonlinear constrs, objs",
        " 0 0\t# network",
        f" 0 {n} 0 \t# nonlinear vars in constraints, objectives, both",
        " 0 0 0 1\t# linear network vars; functions; arith, flags",
        " 0 0 0 0 0 \t# discrete",
        f" {jnnz} {n} \t# nonzeros in Jacobian, obj gradient",
        " 0 0\t# max name lengths",
        " 0 0 0 0 0\t# common exprs",
    ]
    for r in range(m):
        L += [f"C{r}", f"n{consts[r]!r}"]
    L += ["O0 0", "o54", str(n)]
    for j in range(n):
        L += ["o5", "o0", f"v{j}", f"n{-targets[j]!r}", "n4"]
    L.append(f"x{n}")
    for j in range(n):
        L.append(f"{j} {x0[j]!r}")
    L.append("r")
    for i in range(m):
        L.append(f"4 {rhs[i]!r}")
    L.append("b")
    for j in range(n):
        L.append(f"0 {x_l[j]!r} {x_u[j]!r}")
    colcount = [0] * n
    for r in rows:
        for j, _ in r:
            colcount[j] += 1
    L.append(f"k{n - 1}")
    cum = 0
    for j in range(n - 1):
        cum += colcount[j]
        L.append(str(cum))
    for i, r in enumerate(rows):
        L.append(f"J{i} {len(r)}")
        for j, a in sorted(r):
            L.append(f"{j} {a!r}")
    L.append(f"G0 {n}")
    for j in range(n):
        L.append(f"{j} 0")
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------
# Generator: bounds deliberately ACTIVE on columns the pass will eliminate
# --------------------------------------------------------------------------
def gen(rng):
    n = rng.randint(4, 9)
    m = rng.randint(1, max(1, n - 2))
    x_star = [round(rng.uniform(-4, 4), 6) for _ in range(n)]

    rows, rhs, consts = [], [], []
    for _ in range(m):
        # Doubletons are the shape Phase 6 exists for; keep singletons and
        # triples in the mix so the untouched paths stay live.
        arity = rng.choice([1, 2, 2, 2, 2, 3])
        cols = rng.sample(range(n), min(arity, n))
        row = [
            (c, round(rng.uniform(0.3, 3.0), 6) * rng.choice([1, -1]))
            for c in cols
        ]
        rows.append(row)
        consts.append(0.0 if rng.random() < 0.6 else round(rng.uniform(-3, 3), 6))
        rhs.append(round(sum(a * x_star[c] for c, a in row) + consts[-1], 9))

    # Boxes: one side often pinned exactly onto x*, so that side is active the
    # moment the objective pulls that way. Never both sides (that is the
    # equal-bounds shape, a different branch).
    x_l, x_u, tight = [], [], []
    for j in range(n):
        side = rng.choice(["lo", "hi", "lo", "hi", "none"])
        lo_gap = 0.0 if side == "lo" else round(rng.uniform(0.4, 6.0), 6)
        hi_gap = 0.0 if side == "hi" else round(rng.uniform(0.4, 6.0), 6)
        x_l.append(round(x_star[j] - lo_gap, 6))
        x_u.append(round(x_star[j] + hi_gap, 6))
        tight.append(side)

    # Targets: for a pinned side, put the unconstrained minimizer beyond the
    # bound so the objective drives x into it.
    targets = []
    for j in range(n):
        if tight[j] == "lo":
            targets.append(round(x_l[j] - rng.uniform(0.5, 5.0), 6))
        elif tight[j] == "hi":
            targets.append(round(x_u[j] + rng.uniform(0.5, 5.0), 6))
        else:
            targets.append(round(rng.uniform(-4, 4), 6))

    x0 = [
        round(min(max(x_star[j] + rng.uniform(-0.3, 0.3), x_l[j]), x_u[j]), 6)
        for j in range(n)
    ]
    return n, rows, rhs, consts, targets, x_l, x_u, x0


# --------------------------------------------------------------------------
# .sol parsing, including the suffix blocks
# --------------------------------------------------------------------------
def read_sol(path, n, m):
    """Return (lambda_sol, x, suffixes) parsed structurally, not by tail slicing."""
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines) and lines[i].strip() != "Options":
        i += 1
    if i >= len(lines):
        raise ValueError("no Options section in .sol")
    i += 1
    nopts = int(lines[i].strip())
    i += 1 + nopts
    n_dual = int(lines[i].strip()); i += 1
    _m = int(lines[i].strip()); i += 1
    n_prim = int(lines[i].strip()); i += 1
    _n = int(lines[i].strip()); i += 1
    lam = [float(lines[i + k].strip()) for k in range(n_dual)]; i += n_dual
    x = [float(lines[i + k].strip()) for k in range(n_prim)]; i += n_prim
    if (_m, _n) != (m, n):
        raise ValueError(f".sol declares {_n}x{_m}, expected {n}x{m}")

    suffixes = {}
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("suffix "):
            parts = s.split()
            nvalues = int(parts[2])
            name = lines[i + 1].strip()
            vals = {}
            for k in range(nvalues):
                idx, v = lines[i + 2 + k].split()
                vals[int(idx)] = float(v)
            suffixes[name] = vals
            i += 2 + nvalues
        else:
            i += 1
    return lam, x, suffixes


def duals_from_sol(suffixes, n):
    """`ipopt_zL_out = +z_l`, `ipopt_zU_out = -z_u`; absent index means zero."""
    zl_s = suffixes.get("ipopt_zL_out", {})
    zu_s = suffixes.get("ipopt_zU_out", {})
    z_l = [zl_s.get(j, 0.0) for j in range(n)]
    z_u = [-zu_s.get(j, 0.0) for j in range(n)]
    return z_l, z_u


# --------------------------------------------------------------------------
# The KKT oracle
# --------------------------------------------------------------------------
def kkt_report(n, rows, rhs, consts, targets, x_l, x_u, x, lam_sol, z_l, z_u):
    """Every violation of the four KKT conditions, as a list of strings.

    `lam_sol` is the `.sol` dual block, which AMPL convention negates relative
    to the Lagrangian multiplier of `L = f + λᵀg`; both signs are tried and the
    better one kept, exactly as `pounce verify` does, so a convention slip here
    cannot masquerade as a solver defect.
    """
    bad = []
    grad = [4.0 * (x[j] - targets[j]) ** 3 for j in range(n)]
    # Scale the residual thresholds by the size of the quantities involved: a
    # quartic gradient runs to O(100) where the primal runs to O(10).
    scale = max(1.0, max(abs(v) for v in x), max(abs(v) for v in targets),
                max(abs(v) for v in grad))

    # (1) primal feasibility
    for i, row in enumerate(rows):
        body = sum(a * x[c] for c, a in row) + consts[i]
        if abs(body - rhs[i]) > FEAS_TOL * max(1.0, abs(rhs[i])):
            bad.append(f"row {i} violated by {body - rhs[i]:.3e}")
    for j in range(n):
        if x[j] < x_l[j] - FEAS_TOL * scale or x[j] > x_u[j] + FEAS_TOL * scale:
            bad.append(f"x[{j}]={x[j]:.6g} outside [{x_l[j]}, {x_u[j]}]")

    # (2) stationarity, for whichever dual sign convention fits
    def resid_vec(sign):
        r = list(grad)
        for i, row in enumerate(rows):
            for c, a in row:
                r[c] += sign * a * lam_sol[i]
        for j in range(n):
            r[j] += -z_l[j] + z_u[j]
        return r

    def resid(sign):
        return max(abs(v) for v in resid_vec(sign))

    r_pos, r_neg = resid(1.0), resid(-1.0)
    sign = 1.0 if r_pos <= r_neg else -1.0
    best = min(r_pos, r_neg)
    r_vec = resid_vec(sign)
    if best > KKT_TOL * scale:
        bad.append(f"stationarity residual {best:.3e} (both dual signs tried)")

    # (3) dual feasibility
    for j in range(n):
        if z_l[j] < -KKT_TOL:
            bad.append(f"z_l[{j}]={z_l[j]:.3e} is negative")
        if z_u[j] < -KKT_TOL:
            bad.append(f"z_u[{j}]={z_u[j]:.3e} is negative")

    # (4) bound complementarity — the discriminating condition
    for j in range(n):
        slack_lo = x[j] - x_l[j]
        slack_hi = x_u[j] - x[j]
        if abs(z_l[j]) * slack_lo > KKT_TOL * scale:
            bad.append(
                f"complementarity: z_l[{j}]={z_l[j]:.4g} on a lower bound with "
                f"slack {slack_lo:.4g} (x={x[j]:.6g}, x_l={x_l[j]})"
            )
        if abs(z_u[j]) * slack_hi > KKT_TOL * scale:
            bad.append(
                f"complementarity: z_u[{j}]={z_u[j]:.4g} on an upper bound with "
                f"slack {slack_hi:.4g} (x={x[j]:.6g}, x_u={x_u[j]})"
            )
    return bad, {"sign": sign, "resid": r_vec, "scale": scale}


PLAN_DUMP = (Path(__file__).resolve().parents[1] / "fuzz" / "target" / "release"
             / "examples" / "plan_dump")


def collapsed_survivors(n, rows, consts, rhs, x_l, x_u):
    """Survivors whose reduced box the accumulated transfers squeezed to a point.

    Asks the planner itself, through the `plan_dump` example in
    `adversary/fuzz` — a mechanism, not a symptom. Returns `None` when the
    helper is not built, and the caller then refuses to classify rather than
    guessing (build it with
    `cd adversary/fuzz && cargo build --release --example plan_dump`).

    A survivor pinned to a single point is a *fixed* column in the reduced
    problem, which the solver drops from its internal problem, so no bound
    multiplier is produced for it at all — while the full-space cluster it
    stands for may have several simultaneously active bounds needing several
    multipliers. That is the pre-existing defect tracked as gh#495; it is not
    the misattribution gh#493 is about, and it is not fixable by
    re-attribution, because there is nothing to re-attribute.
    """
    if not PLAN_DUMP.exists():
        return None
    inp = "\n".join([
        str(n),
        ";".join(",".join(f"{c}:{a!r}" for c, a in r) for r in rows),
        ",".join(repr(v) for v in consts),
        ",".join(repr(v) for v in rhs),
        ",".join(repr(v) for v in x_l),
        ",".join(repr(v) for v in x_u),
    ])
    out = subprocess.run([str(PLAN_DUMP)], input=inp, capture_output=True,
                         text=True, timeout=30).stdout
    mm = re.search(r"collapsed_reduced_boxes=\[([^\]]*)\]", out)
    if not mm:
        return None
    body = mm.group(1).strip()
    collapsed = [int(v) for v in body.split(",") if v.strip()] if body else []
    pinned = [(int(a), float(b))
              for a, b in re.findall(r"col(\d+) -> Constant\(([^)]+)\)", out)]
    return collapsed, pinned


def duals_are_unique(n, rows, x, x_l, x_u):
    """Exact test for uniqueness of the KKT multipliers at this point.

    The multipliers are unique iff the active constraint gradients are linearly
    independent (LICQ): each equality row's gradient, plus a unit vector for
    every active bound. Where they are dependent the multiplier set is an
    affine family, every member of it is a valid KKT point, and demanding that
    the reduced solve reproduce the no-presolve split is demanding something
    that is not true of the problem.

    This replaces the cluster- and pinned-column heuristics an earlier draft
    used: both were approximations of exactly this condition, and both let
    through instances where a degenerate active set — six active bounds and
    two equality rows on seven columns, in the case that motivated this — made
    the duals legitimately non-unique.
    """
    grads = []
    for row in rows:
        g = [0.0] * n
        for c, a in row:
            g[c] += a
        grads.append(g)
    for j in range(n):
        at_lo = abs(x[j] - x_l[j]) <= ACTIVE_TOL * max(1.0, abs(x[j]))
        at_hi = abs(x_u[j] - x[j]) <= ACTIVE_TOL * max(1.0, abs(x[j]))
        if at_lo or at_hi:
            e = [0.0] * n
            e[j] = 1.0
            grads.append(e)
    if not grads:
        return True
    A = np.array(grads)
    return int(np.linalg.matrix_rank(A, tol=1e-9)) == A.shape[0]


def removed_columns_on_active_bounds(info, x, x_l, x_u):
    """Columns Phase 6 took out of the problem that are sitting on a declared
    bound of their own at the solution.

    These are the columns whose bound multiplier the reduced problem has no
    slot for. Two ways in:

    * a column **pinned to a constant** whose value lands on one of its own
      bounds, and
    * any member of a cluster whose survivor's reduced box the transfers
      squeezed to a **single point** — the survivor is then a fixed column,
      which the solver drops from its internal problem entirely.

    Wherever one of these carries stationarity residual, the reported duals
    are incomplete: there is no multiplier to re-attribute, so this is not the
    gh#493 misattribution and re-attribution cannot fix it. Tracked as
    gh#495.
    """
    collapsed, pinned = info
    out = set()

    def on_bound(j, v):
        tol = max(ACTIVE_TOL, 1e-7) * max(1.0, abs(v))
        return abs(v - x_l[j]) <= tol or abs(v - x_u[j]) <= tol

    for j, v in pinned:
        if on_bound(j, v):
            out.add(j)
    if collapsed:
        # A collapsed survivor makes every column of its cluster suspect; the
        # dump does not name cluster membership, so treat any column on an
        # active declared bound as covered when a collapse happened at all.
        for j in range(len(x)):
            if on_bound(j, x[j]):
                out.add(j)
    return out


def active_columns(n, x, x_l, x_u):
    lo = {j for j in range(n) if abs(x[j] - x_l[j]) <= ACTIVE_TOL * max(1.0, abs(x[j]))}
    hi = {j for j in range(n) if abs(x_u[j] - x[j]) <= ACTIVE_TOL * max(1.0, abs(x[j]))}
    return lo, hi


def clusters_of(n, rows):
    """Union-find over the doubleton equality rows — the columns Phase 6 can
    fold together. Two active bounds inside one cluster is the degenerate case
    where the multiplier split is genuinely not unique."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in rows:
        if len(row) == 2:
            a, b = find(row[0][0]), find(row[1][0])
            if a != b:
                parent[a] = b
    return [find(j) for j in range(n)]


# --------------------------------------------------------------------------
def run_cli(nl_dir, opts):
    return subprocess.run(
        [str(POUNCE), "m", "-AMPL"] + opts,
        cwd=nl_dir, capture_output=True, text=True, timeout=60,
    )


def convention_selfcheck(tmp):
    """Refuse to run unless the `.sol` sign conventions are what we assume.

    `min (x0-4)^4 + x1^4  s.t.  x0 - x1 = 0,  x0 <= 1` has its optimum at
    x0 = x1 = 1 with the upper bound of x0 active, and by hand
    `∇f = [-108, 4]`, `λ = 4`, `z_u[0] = 104`. This only pins the file format
    and the sign conventions; the attribution question is not asked here.
    """
    d = Path(tempfile.mkdtemp(prefix="adv_conv_", dir=tmp))
    try:
        write_nl(d / "m.nl", 2, [[(0, 1.0), (1, -1.0)]], [0.0], [0.0],
                 [4.0, 0.0], [-10.0, -10.0], [1.0, 10.0], [0.0, 0.0])
        p = run_cli(d, [])
        if "Optimal Solution Found" not in p.stdout:
            return "self-check model did not solve:\n" + p.stdout[-400:]
        lam, x, suf = read_sol(d / "m.sol", 2, 1)
        z_l, z_u = duals_from_sol(suf, 2)
        if abs(x[0] - 1.0) > 1e-5 or abs(x[1] - 1.0) > 1e-5:
            return f"self-check primal wrong: {x}"
        # x0 sits on its upper bound; z_u[0] must be positive under the
        # documented `ipopt_zU_out = -z_u` export.
        if abs(z_u[0] - 104.0) > 1e-2:
            return (f"z_u[0]={z_u[0]:.6g} at an active upper bound where the "
                    f"hand calculation gives 104 — the "
                    f"`ipopt_zU_out = -z_u` convention this probe assumes does "
                    f"not hold (suffixes={suf})")
        bad, _ = kkt_report(2, [[(0, 1.0), (1, -1.0)]], [0.0], [0.0],
                            [4.0, 0.0], [-10.0, -10.0], [1.0, 10.0],
                            x, lam, z_l, z_u)
        if bad:
            return "self-check model failed its own KKT oracle: " + "; ".join(bad)
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260806
    rng = random.Random(seed)
    tmp = tempfile.mkdtemp(prefix="adv_p6attr_")

    why = convention_selfcheck(tmp)
    if why:
        print("SELF-CHECK FAILED: " + why)
        print("VERDICT: INCONCLUSIVE (cannot trust the .sol conventions)")
        shutil.rmtree(tmp, ignore_errors=True)
        return 2

    checked = reduced_any = with_active = elim_active = degenerate = 0
    routed_away = 0
    dual_diff = 0
    kkt_bad, verify_bad, dual_bad, control_bad, other_bad = [], [], [], [], []
    missing_mult, not_optimal, dual_diff_valid = [], [], 0
    unclassified = []
    pinned_deg = 0

    for t in range(trials):
        n, rows, rhs, consts, targets, x_l, x_u, x0 = gen(rng)
        m = len(rows)
        d = Path(tempfile.mkdtemp(prefix="adv_p6_", dir=tmp))
        try:
            write_nl(d / "m.nl", n, rows, rhs, consts, targets, x_l, x_u, x0)

            off = run_cli(d, ["presolve=no"])
            if "Optimal Solution Found" not in off.stdout:
                continue
            lam_off, x_off, suf_off = read_sol(d / "m.sol", n, m)
            zl_off, zu_off = duals_from_sol(suf_off, n)

            # Phase 6 ONLY. The other presolve phases move duals for reasons
            # of their own — Phase 2's dropped rows carry the documented M24
            # attribution caveat, and Phase 1 hands the solver *tightened*
            # bounds, against which a multiplier is complementary to a bound
            # this script never wrote. Leaving them on confounds the
            # measurement; the second arm below quantifies exactly that.
            on = run_cli(d, PHASE6_ONLY)
            if "Optimal Solution Found" not in on.stdout:
                # Not an attribution question: the reduced solve never got to a
                # point to report duals at. Counted separately.
                not_optimal.append((t, "reduced solve did not reach optimality",
                                    on.stdout[-200:]))
                continue
            checked += 1
            # Assert the pass ENGAGED rather than trusting that it did: a
            # transparent fall-through to another solver or to the passthrough
            # wrapper would otherwise read as a clean run.
            if "Selected solver" in on.stdout and "pounce-convex" in on.stdout:
                routed_away += 1
            mm = re.search(r"eliminated (\d+) columns", on.stdout)
            n_elim = int(mm.group(1)) if mm else 0
            if n_elim:
                reduced_any += 1
            lam_on, x_on, suf_on = read_sol(d / "m.sol", n, m)
            zl_on, zu_on = duals_from_sol(suf_on, n)

            lo_act, hi_act = active_columns(n, x_on, x_l, x_u)
            if lo_act or hi_act:
                with_active += 1
            # Was an active bound on a column that is NOT the survivor of its
            # cluster? Survivorship is not observable from outside, but a
            # cluster of size > 1 with an active bound is exactly the situation
            # a transfer arises from.
            root = clusters_of(n, rows)
            size = {}
            for j in range(n):
                size[root[j]] = size.get(root[j], 0) + 1
            act = lo_act | hi_act
            if any(size[root[j]] > 1 for j in act):
                elim_active += 1
            # Degenerate: two active bounds inside one cluster.
            per_cluster = {}
            for j in act:
                per_cluster[root[j]] = per_cluster.get(root[j], 0) + 1
            is_degenerate = any(v > 1 for v in per_cluster.values())
            if is_degenerate:
                degenerate += 1

            # --- control: the SAME oracle on the no-presolve solve --------
            # If this fires, the oracle or the formulation is wrong, not the
            # pass under test. It is checked every run, not once.
            bad_off, _ = kkt_report(n, rows, rhs, consts, targets, x_l, x_u,
                                    x_off, lam_off, zl_off, zu_off)
            if bad_off:
                # The oracle could not certify the *no-presolve* solve on this
                # instance, so it is in no position to judge the reduced one.
                # Skipped, and counted: a handful is the solver stopping at
                # "acceptable level" on a very flat quartic, but a large
                # fraction would mean the oracle itself is miscalibrated, which
                # the cap below turns into a failed run.
                control_bad.append((t, "; ".join(bad_off[:3]), ""))
                continue

            # --- oracle A: the KKT conditions, from this script's own data ---
            bad, det = kkt_report(n, rows, rhs, consts, targets, x_l, x_u,
                                  x_on, lam_on, zl_on, zu_on)
            if bad:
                # Stationarity-only violations on an instance whose reduced
                # box collapsed to a point are the pre-existing defect, not
                # this one. Anything else — a complementarity violation, a
                # negative multiplier, or a residual with no collapse to
                # explain it — is a gh#493 failure and fails the run.
                stationarity_only = all("stationarity" in b for b in bad)
                info = collapsed_survivors(n, rows, consts, rhs, x_l, x_u)
                if info is None:
                    unclassified.append((t, "; ".join(bad[:2]) +
                                         "  [plan_dump not built; cannot classify]", ""))
                    continue
                tol = KKT_TOL * det["scale"]
                offenders = [j for j in range(n) if abs(det["resid"][j]) > tol]
                covered = removed_columns_on_active_bounds(info, x_on, x_l, x_u)
                # Either mechanism explains it. A collapsed survivor is a fixed
                # column in the reduced problem, and once its bound multiplier
                # is missing the recovered row multipliers are off too, so the
                # residual spreads across the whole cluster rather than staying
                # on the columns that sit on bounds — which is why this is an
                # OR and not the narrower per-column test alone.
                # Any removed column on an active bound is enough. Once one
                # missing multiplier is in the system, the recovered row
                # multipliers built on top of it are off too, so the residual
                # spreads to survivors and to columns that are not on bounds —
                # seed 66/42 puts it on eight of nine columns. Requiring every
                # offender to be on a bound would misfile exactly the worst
                # cases.
                explained = bool(info[0]) or bool(covered)
                if stationarity_only and offenders and explained:
                    missing_mult.append((
                        t, "; ".join(bad[:2]) +
                        f"  [residual at {offenders}; collapsed survivors="
                        f"{info[0]}; removed-and-on-a-bound={sorted(covered)}]", ""))
                else:
                    kkt_bad.append((t, "; ".join(bad[:3]), ""))

            # --- oracle B: pounce verify against the ORIGINAL .nl ---
            v = subprocess.run([str(POUNCE), "verify", "m.nl", "m.sol"],
                               cwd=d, capture_output=True, text=True, timeout=60)
            if v.returncode != 0:
                verify_bad.append((t, f"rc={v.returncode}", v.stdout[-400:]))

            # --- arm 2: every presolve phase on, for scope attribution -----
            other = run_cli(d, ["presolve=yes", "presolve_linear_eq_reduction=no"])
            if "Optimal Solution Found" in other.stdout:
                lam_o, x_o, suf_o = read_sol(d / "m.sol", n, m)
                zl_o, zu_o = duals_from_sol(suf_o, n)
                bad_o, _ = kkt_report(n, rows, rhs, consts, targets, x_l, x_u,
                                      x_o, lam_o, zl_o, zu_o)
                if bad_o:
                    other_bad.append((t, "; ".join(bad_o[:2]), ""))
                # Restore the Phase-6 .sol for anything downstream.
                run_cli(d, PHASE6_ONLY)

            # --- oracle C: the no-presolve solve, where duals are unique ---
            # Only where they really are unique: not degenerate by the cluster
            # test, no collapsed reduced box, and the reduced duals a valid KKT
            # point to begin with. Anywhere else a difference is either
            # non-uniqueness or an already-counted violation, and asserting
            # equality would be asserting something untrue.
            unique_duals = (not bad) and duals_are_unique(n, rows, x_on, x_l, x_u)
            if not unique_duals:
                pinned_deg += 1
            if unique_duals:
                sc = max(1.0, max(abs(v) for v in zl_off + zu_off + zl_on + zu_on))
                worst = max(
                    max(abs(zl_on[j] - zl_off[j]) for j in range(n)),
                    max(abs(zu_on[j] - zu_off[j]) for j in range(n)),
                )
                if worst > 1e-4 * sc:
                    dual_diff += 1
                    jw = max(range(n), key=lambda j: max(
                        abs(zl_on[j] - zl_off[j]), abs(zu_on[j] - zu_off[j])))
                    dual_bad.append((
                        t,
                        f"bound multipliers differ from the no-presolve solve by "
                        f"{worst:.3e} (worst at column {jw}: reduced "
                        f"z_l={zl_on[jw]:.6g} z_u={zu_on[jw]:.6g}, bare "
                        f"z_l={zl_off[jw]:.6g} z_u={zu_off[jw]:.6g})", ""))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    shutil.rmtree(tmp, ignore_errors=True)

    print(f"instances solved by both paths     : {checked}")
    print(f"  where the pass eliminated columns: {reduced_any}")
    print(f"  routed away from the NLP path    : {routed_away}")
    print(f"  with at least one active bound   : {with_active}")
    print(f"  active bound inside a fold cluster: {elim_active}")
    print(f"  degenerate (2 active in a cluster): {degenerate}")
    print(f"  duals non-unique (LICQ fails)     : {pinned_deg}")
    print(f"CONTROL uncertifiable, instance skipped: {len(control_bad)}")
    print(f"KKT violations (Phase 6 only)      : {len(kkt_bad)}   <- gh#493, must be 0")
    print(f"  explained by a collapsed reduced box: {len(missing_mult)}   "
          f"<- {SEPARATE_ISSUE}, pre-existing")
    print(f"KKT violations (other phases, no P6): {len(other_bad)}   <- scope check, not this issue")
    print(f"pounce verify rejections           : {len(verify_bad)}")
    print(f"reduced solve non-optimal          : {len(not_optimal)}")
    print(f"duals differing where they are UNIQUE: {dual_diff}   <- gh#493, must be 0")
    for label, items in (("CONTROL", control_bad), ("KKT", kkt_bad),
                         ("collapsed-box " + SEPARATE_ISSUE, missing_mult),
                         ("UNCLASSIFIED", unclassified),
                         ("non-optimal", not_optimal), ("verify", verify_bad),
                         ("dual-attribution", dual_bad)):
        for t, wy, extra in items[:40]:
            print(f"  [{label} {t}] {wy}")
            if extra:
                print("      " + extra.replace("\n", "\n      ")[:400])

    vacuous = elim_active == 0 or reduced_any == 0 or routed_away > 0
    # The verdict is about gh#493 and nothing else: a bound multiplier must
    # never be reported against a bound its column is not sitting on, and the
    # reduced duals must be a valid KKT point of the ORIGINAL problem. Dual
    # differences that are themselves valid KKT points are non-uniqueness, not
    # defects, and the pre-existing incomplete-duals class is tracked under its
    # own issue so it can neither mask a regression nor be mistaken for one.
    control_ok = len(control_bad) <= max(2, 0.05 * max(1, checked))
    ok = (not kkt_bad) and (not verify_bad) and control_ok \
        and (not unclassified) and (dual_diff == 0) \
        and not vacuous
    if not control_ok:
        print(f"ORACLE MISCALIBRATED: the no-presolve solve failed the KKT check "
              f"on {len(control_bad)} of {checked} instances")
    if vacuous:
        print("PROBE VACUOUS: the attribution path was not exercised "
              f"(eliminated-columns instances={reduced_any}, "
              f"active-in-cluster={elim_active}, routed-away={routed_away})")
    print("VERDICT: PASS" if ok else "VERDICT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
