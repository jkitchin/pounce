"""Adversary cross-check: the `.nl` reader's constant-row-body fold.

Family: nlp   Class: `.nl` parse boundary / linear rows with constant offsets
Target: `parse_nl_text`'s constant-row-body fold (gh#492),
        `crates/pounce-nl/src/nl_reader.rs`
Source: no published instance — a randomized family of LPs built around a
        KNOWN feasible point, with scipy's HiGHS as the external oracle.

The change under test rewrites the model at the parse boundary: when a row's
`C<i>` expression segment evaluates to a constant `c`, the reader subtracts
`c` from that row's bounds and replaces the body with zero. Every row in this
family carries such a constant, in five different spellings.

Four things can go wrong, and each has its own oracle here. None of them is
"pounce with the fold disabled" — that binary no longer exists.

1. **The shift lands on the wrong bound, or with the wrong sign.** Oracle:
   `scipy.optimize.linprog` (HiGHS) on the model as *this script* understands
   it, assembled from the generator's own data and never from the `.nl`.
2. **A bound is invented or lost.** Specifically the ±1e19 "absent" sentinels,
   which are directional (gh#401): shifting one turns "no bound" into a real
   one. Oracle: every row is re-evaluated here against its ORIGINAL declared
   bounds — `lo <= Σaⱼxⱼ + c <= hi`, constant included — plus a targeted
   sentinel battery at magnitudes where the shift is not absorbed by the
   sentinel's own ULP (2048 at 1e19).
3. **The duals move.** The fold's whole claim is that the body drops by `c`
   and the bound drops with it, so the residual — and therefore every
   multiplier — is untouched. Oracle: the same model written with the constant
   folded into the bound *by hand*, which reaches the reader with nothing to
   fold and so exercises none of the new code. Plus an independent
   stationarity check on the reported duals.
4. **The fold fires when it must not.** A non-finite constant, or one inside
   an imported-function call, must be left in the expression rather than
   pushed into a bound. Oracle: direct assertions on the resulting behavior.

Both `.nl` spellings of each instance are also passed through `pounce verify`.
That is a weaker oracle than usual here and is reported as such: `verify`
re-reads the `.nl` through the same reader, so it shares the fold and cannot
by itself catch a bound the fold moved wrongly. It is kept because it does
catch a point that is non-stationary or out of its box.

Usage: python 2026-08-06_nlp_row_constant_fold.py [trials] [seed]
"""

import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

POUNCE = Path(
    os.environ.get("POUNCE_BIN", "/home/user/pounce/target/release/pounce")
)

# `r`-segment bound kinds we generate. 3 (free) is included because a free row
# is the one shape where BOTH sentinels are live at once.
EQ, UPPER, LOWER, RANGE, FREE = 4, 1, 2, 0, 3


# ---------------------------------------------------------------------
# .nl writing
# ---------------------------------------------------------------------


def const_tokens(c, spelling):
    """`C<i>` token stream for the constant `c`, written five ways.

    The fold is by *evaluation*, not syntax, so every spelling must land on
    the same model. `literal` is what a writer usually emits; the rest are
    what the classifier's polynomial walk cannot lower, and so are the forms
    that used to force an LP onto the NLP route.
    """
    if spelling == "literal":
        return [f"n{c!r}"]
    if spelling == "add":  # (c - 0.25) + 0.25
        return ["o0", f"n{c - 0.25!r}", "n0.25"]
    if spelling == "sub":  # (c + 1.5) - 1.5
        return ["o1", f"n{c + 1.5!r}", "n1.5"]
    if spelling == "neg":  # -(-c)
        return ["o16", f"n{-c!r}"]
    if spelling == "sqrt":  # sqrt(c^2), sign restored by o16 when c < 0
        body = ["o39", f"n{c * c!r}"]
        return body if c >= 0 else ["o16"] + body
    raise ValueError(spelling)


def write_nl(path, n, obj_c, rows, kinds, los, his, consts, x_l, x_u, x0, spellings):
    """`min cᵀx  s.t.  lo_i <= Σ a_ij x_j + k_i <= hi_i,  x_l <= x <= x_u`.

    Every row's constant `k_i` goes in that row's expression segment. Pass
    `spellings=None` to fold the constants into the bounds here instead and
    emit an empty `C<i>` — the hand-folded reference model.
    """
    m = len(rows)
    jnnz = sum(len(r) for r in rows)
    folded = spellings is None
    L = [
        "g3 1 1 0\t# adversary row-constant fold",
        f" {n} {m} 1 0 0 \t# vars, constraints, objectives, ranges, eqns",
        f" {0 if folded else m} 0 0 0 0 0\t# nonlinear constrs, objs",
        " 0 0\t# network",
        f" {0 if folded else n} 0 0 \t# nonlinear vars in constraints, objectives, both",
        " 0 0 0 1\t# linear network vars; functions; arith, flags",
        " 0 0 0 0 0 \t# discrete",
        f" {jnnz} {n} \t# nonzeros in Jacobian, obj gradient",
        " 0 0\t# max name lengths",
        " 0 0 0 0 0\t# common exprs",
    ]
    for i in range(m):
        L.append(f"C{i}")
        L += ["n0"] if folded else const_tokens(consts[i], spellings[i])
    L += ["O0 0", "n0"]
    L.append(f"x{n}")
    for j in range(n):
        L.append(f"{j} {x0[j]!r}")
    L.append("r")
    for i in range(m):
        # The hand-folded model states the SAME row with the constant moved
        # onto whichever sides are actually bounded — the transformation the
        # reader is supposed to be performing.
        shift = consts[i] if folded else 0.0
        if kinds[i] == EQ:
            L.append(f"4 {los[i] - shift!r}")
        elif kinds[i] == UPPER:
            L.append(f"1 {his[i] - shift!r}")
        elif kinds[i] == LOWER:
            L.append(f"2 {los[i] - shift!r}")
        elif kinds[i] == FREE:
            L.append("3")
        else:
            L.append(f"0 {los[i] - shift!r} {his[i] - shift!r}")
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
        L.append(f"{j} {obj_c[j]!r}")
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------
# instance generation
# ---------------------------------------------------------------------


def gen(rng):
    n = rng.randint(4, 8)
    m = rng.randint(1, max(1, n - 1))
    x_star = [round(rng.uniform(-3, 3), 6) for _ in range(n)]
    x_l = [round(x_star[j] - rng.uniform(1.0, 5.0), 6) for j in range(n)]
    x_u = [round(x_star[j] + rng.uniform(1.0, 5.0), 6) for j in range(n)]
    obj_c = [round(rng.uniform(-2, 2), 6) for _ in range(n)]

    rows, kinds, los, his, consts, spellings = [], [], [], [], [], []
    for _ in range(m):
        cols = rng.sample(range(n), rng.randint(1, min(3, n)))
        row = [(c, round(rng.uniform(0.3, 3.0), 6) * rng.choice([1, -1])) for c in cols]
        # A zero constant on one row in five keeps the "nothing to fold" path
        # live inside the same generator.
        k = 0.0 if rng.random() < 0.2 else round(rng.uniform(-4, 4), 6)
        lin = sum(a * x_star[c] for c, a in row)
        body = lin + k  # the row's value at the known feasible point
        kind = rng.choice([EQ, UPPER, LOWER, RANGE, FREE])
        lo, hi = -math.inf, math.inf
        # Bounds are NOT rounded. Rounding them to 9 places was this probe's
        # own bug: the generator happily draws several singleton equality rows
        # on one column, and independently-rounded right-hand sides then pin
        # that column to values ~3e-10 apart. The system is genuinely
        # inconsistent at that point — HiGHS absorbs it inside its feasibility
        # tolerance, pounce-convex calls it primal infeasible, and neither is
        # wrong about a model the probe should not have written. Deriving the
        # bounds straight from `body` keeps every row exactly satisfied at x*.
        if kind == EQ:
            lo = hi = body
        elif kind == UPPER:
            hi = body + rng.uniform(0.0, 2.0)
        elif kind == LOWER:
            lo = body - rng.uniform(0.0, 2.0)
        elif kind == RANGE:
            lo = body - rng.uniform(0.1, 2.0)
            hi = body + rng.uniform(0.1, 2.0)
        rows.append(row)
        kinds.append(kind)
        los.append(lo)
        his.append(hi)
        consts.append(k)
        spellings.append(rng.choice(["literal", "add", "sub", "neg", "sqrt"]))
    return dict(
        n=n, m=m, obj_c=obj_c, rows=rows, kinds=kinds, los=los, his=his,
        consts=consts, x_l=x_l, x_u=x_u, x0=list(x_star), spellings=spellings,
    )


# ---------------------------------------------------------------------
# oracles
# ---------------------------------------------------------------------


def scipy_solve(inst):
    """The external oracle: HiGHS on the model as assembled here.

    The row constants are moved into the bounds *in this function*, from the
    generator's own numbers. Nothing here reads the `.nl`, so agreement is
    evidence about the reader and not about this script.
    """
    n, m = inst["n"], inst["m"]
    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    for i, r in enumerate(inst["rows"]):
        row = np.zeros(n)
        for j, a in r:
            row[j] += a
        lo = inst["los"][i] - inst["consts"][i]
        hi = inst["his"][i] - inst["consts"][i]
        if math.isfinite(lo) and math.isfinite(hi) and lo == hi:
            a_eq.append(row)
            b_eq.append(lo)
            continue
        if math.isfinite(hi):
            a_ub.append(row)
            b_ub.append(hi)
        if math.isfinite(lo):
            a_ub.append(-row)
            b_ub.append(-lo)
    return linprog(
        c=np.array(inst["obj_c"]),
        A_ub=np.array(a_ub) if a_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(a_eq) if a_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=list(zip(inst["x_l"], inst["x_u"])),
        method="highs",
    )


def rows_hold(inst, x, tol=1e-6):
    """Re-evaluate every row *with its constant* against its ORIGINAL bounds.

    This is the invented/lost-bound check. A fold that shifted the wrong side,
    or that touched an absent-bound sentinel, produces a point that satisfies
    the reader's idea of the row but not the row the file declares.
    """
    for i, r in enumerate(inst["rows"]):
        body = sum(a * x[c] for c, a in r) + inst["consts"][i]
        scale = max(1.0, abs(body))
        if math.isfinite(inst["los"][i]) and body < inst["los"][i] - tol * scale:
            return f"row {i} below its declared lower bound: {body} < {inst['los'][i]}"
        if math.isfinite(inst["his"][i]) and body > inst["his"][i] + tol * scale:
            return f"row {i} above its declared upper bound: {body} > {inst['his'][i]}"
    for j in range(inst["n"]):
        if not (inst["x_l"][j] - tol <= x[j] <= inst["x_u"][j] + tol):
            return f"x[{j}]={x[j]} outside [{inst['x_l'][j]}, {inst['x_u'][j]}]"
    return None


def stationarity(inst, x, lam, tol=1e-5):
    """`(c + Aᵀλ)_j ≈ 0` for every variable strictly inside its box.

    An interior column has no bound multiplier, so the row duals alone must
    cancel its objective gradient. Independent of the oracle LP and of the
    hand-folded model: it uses only the generator's `c`/`A` and the duals the
    `.sol` reports. The sign convention is fixed by trying both and demanding
    that one of them hold — AMPL's is `∇f + Jᵀλ`, but the point of the check
    is that the fold does not perturb it, not which sign it is.

    The interior margin is 1e-4, not the 1e-6 this probe first used. An IPM
    settles a *bound-active* column a small distance off that bound, and at
    1e-6 such a column read as interior — so the check demanded that the row
    duals alone cancel a gradient that a bound multiplier was in fact
    carrying, and reported a false positive on a solve that matched HiGHS to
    6e-10. Only columns clearly off both bounds are testable this way.
    """
    n = inst["n"]
    interior = [
        j for j in range(n)
        if inst["x_l"][j] + 1e-4 < x[j] < inst["x_u"][j] - 1e-4
    ]
    if not interior:
        return None
    for sign in (1.0, -1.0):
        worst = 0.0
        for j in interior:
            g = inst["obj_c"][j]
            for i, r in enumerate(inst["rows"]):
                for cidx, a in r:
                    if cidx == j:
                        g += sign * lam[i] * a
            worst = max(worst, abs(g))
        if worst <= tol:
            return None
    return f"stationarity fails at every sign for interior columns {interior}"


# ---------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------


def solve(d, name, opts=()):
    p = subprocess.run(
        [str(POUNCE), name, "-AMPL", *opts],
        cwd=d, capture_output=True, text=True, timeout=60,
    )
    return p


def read_sol(path, n, m):
    nums = []
    for line in path.read_text().splitlines():
        if line.startswith("objno"):
            break
        try:
            nums.append(float(line.strip()))
        except ValueError:
            pass
    tail = nums[-(n + m):] if (n + m) else []
    return tail[:m], tail[m:]


# ---------------------------------------------------------------------
# targeted batteries
# ---------------------------------------------------------------------


def sentinel_battery():
    """Constants large enough that shifting a sentinel would be visible.

    At everyday magnitudes an unguarded `g_l -= c` is absorbed by the
    sentinel's ULP and nothing happens. These sit at 1e17–1e18, where a
    shifted `-1e19` lands at `-9e18` — a real bound, and one that would cut
    off the true optimum of a row that is supposed to be one-sided.

    Model: `min -x0 - x1` s.t. `x0 + x1 + k <= k + 2` (upper only) or
    `-x0 - x1 + k >= k - 2` (lower only), `x ∈ [0,5]²`. Optimum is 2 either
    way; a lower bound invented on the first row would change nothing, so the
    row is also checked against the reader directly via the free-row case,
    where BOTH sentinels are live and any shift is a fabricated constraint.
    """
    cases = []
    for k in (1e17, -1e17, 1e18, -1e18, 3.0):
        # one-sided upper: the lower sentinel must stay absent
        cases.append(("upper", k))
        # one-sided lower: the upper sentinel must stay absent
        cases.append(("lower", k))
        # free row: both sentinels live
        cases.append(("free", k))
    return cases


def run_sentinel_case(kind, k, tmp):
    n, m = 2, 1
    rows = [[(0, 1.0), (1, 1.0)]]
    if kind == "upper":
        kinds, los, his = [UPPER], [-math.inf], [k + 2.0]
    elif kind == "lower":
        kinds, los, his = [LOWER], [k - 2.0], [math.inf]
    else:
        kinds, los, his = [FREE], [-math.inf], [math.inf]
    inst = dict(
        n=n, m=m, obj_c=[-1.0, -1.0], rows=rows, kinds=kinds, los=los, his=his,
        consts=[k], x_l=[0.0, 0.0], x_u=[5.0, 5.0], x0=[0.0, 0.0],
        spellings=["literal"],
    )
    d = Path(tempfile.mkdtemp(prefix="adv_sent_", dir=tmp))
    try:
        write_nl(d / "m.nl", n, inst["obj_c"], rows, kinds, los, his, inst["consts"],
                 inst["x_l"], inst["x_u"], inst["x0"], inst["spellings"])
        p = solve(d, "m")
        if "Optimal Solution Found" not in p.stdout:
            return f"{kind}/k={k}: not solved\n{p.stdout[-300:]}"
        _, x = read_sol(d / "m.sol", n, m)
        ref = scipy_solve(inst)
        if not ref.success:
            return f"{kind}/k={k}: oracle failed ({ref.message})"
        got = sum(c * xi for c, xi in zip(inst["obj_c"], x))
        if abs(got - ref.fun) > 1e-6 * max(1.0, abs(ref.fun)):
            return (f"{kind}/k={k}: objective {got:.10e} vs HiGHS {ref.fun:.10e} "
                    f"-- a shifted sentinel would do exactly this")
        return rows_hold(inst, x)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_decline_cases(tmp):
    """Bodies the fold must refuse: non-finite, and an imported-function call.

    Neither may be silently pushed into a bound. A NaN in `g_l`/`g_u` makes
    every presence test downstream unanswerable, and an external function is
    not a parse-time constant at all — it resolves to a shared library much
    later.
    """
    problems = []
    base = [
        "g3 1 1 0\t# adversary decline case",
        " 2 1 1 0 0 ",
        " 1 0 0 0 0 0",
        " 0 0",
        " 1 0 0 ",
        " 0 0 0 1",
        " 0 0 0 0 0 ",
        " 2 2 ",
        " 0 0",
        " 0 0 0 0 0",
    ]
    tail = [
        "O0 0", "n0",
        "r", "1 6.0",
        "b", "0 0 3", "0 0 3",
        "k1", "1",
        "J0 2", "0 1", "1 1",
        "G0 2", "0 -1", "1 -2",
    ]

    # log(-1) = NaN. The row is unsatisfiable; the one unacceptable outcome is
    # a confident "Optimal Solution Found" on a model whose bounds have been
    # replaced by NaN.
    d = Path(tempfile.mkdtemp(prefix="adv_nan_", dir=tmp))
    try:
        (d / "m.nl").write_text("\n".join(base + ["C0", "o43", "n-1"] + tail) + "\n")
        p = solve(d, "m")
        if "Optimal Solution Found" in p.stdout:
            _, x = read_sol(d / "m.sol", 2, 1)
            problems.append(
                f"a log(-1) row body solved to optimality at x={x}; the row is NaN"
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # An imported function with a constant argument. `myfunc(2)` looks like a
    # constant and is not one: it resolves to a shared library long after
    # parse. No library is loaded here, so the run must fail either way — the
    # question is *where*, and the two failures are distinguishable.
    #
    #   guard present: the call survives parse, and the run dies later at
    #     external-function resolution ("AMPLFUNC is not set"). That panic is
    #     pre-existing and unrelated to this fold.
    #   guard removed: the fold reaches `eval_expr` with a `Funcall`, which
    #     panics by construction rather than guessing a value.
    #
    # So "the run failed" is NOT the check — a check that only asserted
    # failure passes with the guard deleted, which is how this probe's first
    # draft let that mutation through. The check is that the failure did not
    # come from inside the fold's evaluator.
    d = Path(tempfile.mkdtemp(prefix="adv_fn_", dir=tmp))
    try:
        (d / "m.nl").write_text(
            "\n".join(base + ["F0 1 1 myfunc", "C0", "f0 1", "n2.0"] + tail) + "\n"
        )
        p = solve(d, "m")
        out = p.stdout + p.stderr
        if "Optimal Solution Found" in out:
            problems.append(
                "a model calling the unresolved imported function myfunc(2) "
                "solved to optimality; the call was folded away"
            )
        elif "eval_expr" in out:
            problems.append(
                "the fold evaluated an imported-function body: it reached "
                "eval_expr with a Funcall instead of declining"
            )
        elif "AMPLFUNC" not in out:
            problems.append(
                f"unexpected failure mode for the funcall body (rc={p.returncode}); "
                f"expected external-function resolution to be what fails: {out[-300:]}"
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return problems


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260806
    rng = random.Random(seed)
    tmp = tempfile.mkdtemp(prefix="adv_rcf_root_")

    bad = []
    checked = 0
    skipped_baseline = 0
    lp_routed = 0
    verify_ok = 0
    try:
        for t in range(trials):
            inst = gen(rng)
            n, m = inst["n"], inst["m"]
            ref = scipy_solve(inst)
            if not ref.success:
                continue  # unbounded / degenerate oracle: not this probe's business

            d = Path(tempfile.mkdtemp(prefix="adv_rcf_", dir=tmp))
            try:
                write_nl(d / "off.nl", n, inst["obj_c"], inst["rows"], inst["kinds"],
                         inst["los"], inst["his"], inst["consts"], inst["x_l"],
                         inst["x_u"], inst["x0"], inst["spellings"])
                write_nl(d / "fold.nl", n, inst["obj_c"], inst["rows"], inst["kinds"],
                         inst["los"], inst["his"], inst["consts"], inst["x_l"],
                         inst["x_u"], inst["x0"], None)

                p_off = solve(d, "off")
                p_fold = solve(d, "fold")
                if "Optimal Solution Found" not in p_fold.stdout:
                    # The hand-folded model exercises none of the fold's code,
                    # so a failure here is a pre-existing solver issue and not
                    # this probe's subject. Counted, never silently dropped:
                    # if this number is not small, the probe is measuring the
                    # solver's bad days rather than the reader.
                    #
                    # One such instance is understood: gh#496, a false
                    # primal-infeasible certificate when two equality rows are
                    # rank-deficient and their implied values differ by one
                    # ULP. Reproduced on origin/main with no `C`-segment
                    # content at all, so it predates this change.
                    skipped_baseline += 1
                    continue
                if "Optimal Solution Found" not in p_off.stdout:
                    bad.append((t, "row-constant model did not solve",
                                p_off.stdout[-300:]))
                    continue
                checked += 1
                if "Problem class: LP" in p_off.stdout:
                    lp_routed += 1

                lam_off, x_off = read_sol(d / "off.sol", n, m)
                lam_fold, x_fold = read_sol(d / "fold.sol", n, m)

                # --- oracle 1: HiGHS, from this script's own assembly ---
                obj_off = sum(c * xi for c, xi in zip(inst["obj_c"], x_off))
                if abs(obj_off - ref.fun) > 1e-6 * max(1.0, abs(ref.fun)):
                    bad.append((t, f"objective {obj_off:.10e} vs HiGHS {ref.fun:.10e}", ""))
                    continue

                # --- oracle 2: the rows as the FILE declares them ---
                why = rows_hold(inst, x_off)
                if why:
                    bad.append((t, f"reported point violates the original model: {why}", ""))
                    continue

                # --- oracle 3: the hand-folded model, primal AND dual ---
                dx = max(abs(a - b) for a, b in zip(x_off, x_fold)) if n else 0.0
                if dx > 1e-6:
                    bad.append((t, f"primal differs from the hand-folded model by {dx:.3e}", ""))
                    continue
                dl = max((abs(a - b) for a, b in zip(lam_off, lam_fold)), default=0.0)
                if dl > 1e-6:
                    bad.append((t, f"duals differ from the hand-folded model by {dl:.3e}", ""))
                    continue

                # --- oracle 4: stationarity of the reported duals ---
                why = stationarity(inst, x_off, lam_off)
                if why:
                    bad.append((t, why, ""))
                    continue

                # --- weak check: pounce verify (shares the reader; see docstring) ---
                v = subprocess.run([str(POUNCE), "verify", "off.nl", "off.sol"],
                                   cwd=d, capture_output=True, text=True, timeout=60)
                if v.returncode == 0:
                    verify_ok += 1
                else:
                    bad.append((t, f"pounce verify rejected the solve (rc={v.returncode})",
                                v.stdout[-400:]))
            finally:
                shutil.rmtree(d, ignore_errors=True)

        # --- targeted batteries ---
        sentinel_fail = []
        for kind, k in sentinel_battery():
            why = run_sentinel_case(kind, k, tmp)
            if why:
                sentinel_fail.append(why)
        decline_fail = run_decline_cases(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=== randomized LPs with constant row bodies ===")
    print(f"instances solved and checked : {checked}")
    print(f"  skipped, baseline unsolved : {skipped_baseline}  (gh#496, pre-existing)")
    print(f"  classified LP by pounce    : {lp_routed}")
    print(f"  accepted by pounce verify  : {verify_ok}")
    print(f"failures                     : {len(bad)}")
    for t, why, extra in bad[:8]:
        print(f"  [{t}] {why}")
        if extra:
            print("      " + extra.replace("\n", "\n      ")[:400])
    print("=== sentinel battery (|c| up to 1e18, one-sided and free rows) ===")
    print(f"cases                        : {len(sentinel_battery())}")
    print(f"failures                     : {len(sentinel_fail)}")
    for why in sentinel_fail[:8]:
        print(f"  {why}")
    print("=== decline battery (non-finite body, imported-function body) ===")
    print(f"failures                     : {len(decline_fail)}")
    for why in decline_fail:
        print(f"  {why}")

    total = len(bad) + len(sentinel_fail) + len(decline_fail)
    print("VERDICT: PASS" if total == 0 else f"VERDICT: FAIL ({total} failures)")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
