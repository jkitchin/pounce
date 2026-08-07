"""Adversary cross-check: convex-path doubleton aggregation, end to end.

Family: convex   Class: convex QP / LP with alias-linked columns
Target: the doubleton-equality aggregation in `pounce-convex`'s presolve (gh#494)
Source: no published instance — a randomized family built around a KNOWN
        feasible point, with `pounce verify` as the independent oracle.

`crates/pounce-convex/tests/presolve_aggregation.rs` attacks the reduction
through the library API, where the QP is handed over already extracted. This
one attacks the stack a user actually meets: a `.nl` file on disk, the CLI's
LP/QP classifier and extractor, the presolve, the IPM, the postsolve, and the
`.sol` file that comes back.

Four oracles, only one of which is "pounce with the option off":

1. **`pounce verify <nl> <sol>`** — an independent feasibility/KKT check of the
   claimed point against the ORIGINAL `.nl`, which knows nothing about the
   reduction. A point that violates a row the aggregation consumed is rejected
   here.
2. **Direct evaluation in this script** — every row is re-evaluated in Python
   from the same data used to write the `.nl`. That catches a `.sol` whose
   primal block is the right length but the wrong permutation, which `verify`
   alone cannot distinguish from a genuine solve.
3. **Bound-aware stationarity, in this script.** With `rⱼ = ∇fⱼ + s·(Aᵀλ)ⱼ`
   for the sign convention `s` the baseline solve establishes: `rⱼ = 0` at an
   interior column, `rⱼ ≥ 0` at an active lower bound, `rⱼ ≤ 0` at an active
   upper bound — the conditions a bound multiplier of the right sign has to
   absorb. This is the oracle that sees the reduction's hardest claim, which
   is where the bound force ends up when planning transfers a box onto a
   column that has no such bound of its own. `verify` alone does not: it is
   generous about columns sitting on a bound, exactly so that a missing
   suffix is not read as a wrong answer.
4. **`qp_presolve=no`** — the objective must agree with the un-reduced solve.
   Weakest of the four (both sides are POUNCE), but it is the one that
   notices a reduction that is self-consistently wrong.

Each instance is generated around a random `x*` used only to compute the
right-hand sides, so every model is feasible by construction and a "proved
infeasible" verdict is always a bug.

**What this run cannot reach, measured rather than assumed.** The CLI's QP
extractor lowers every variable bound to a row of `G`, so the `QpProblem` the
convex presolve receives from a `.nl` file has an *empty box*. Instrumenting
the branch shows the transferred-bound re-attribution in
`aggregate::postsolve` firing in 0 of 114 instances here, for that reason:
with no box, planning has nothing to transfer, and each bound's force lands
on an ordinary inequality multiplier the sweep already accounts for. That
recovery matters for callers who build a `QpProblem` with `lb`/`ub` directly
— `pounce-py`, the batch API, any embedder — and for the boxes the catalog's
own bound tightening creates, and it is covered at that level by
`crates/pounce-convex/tests/presolve_aggregation.rs` (a targeted fixture plus
a randomized probe that reaches the branch in 19 of 400 draws). What this run
does cover end to end is everything else: classification, extraction, the
primal substitution, the row rewrite, the consumed-row dual sweep, postsolve,
and the `.sol` that comes back.

Usage:  python3 <this file> [trials] [seed]
"""

import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

POUNCE = Path("/home/user/pounce/target/release/pounce")


def write_nl(path, n, rows, rhs, lin, x_l, x_u, x0, consts=None):
    """`min Σ xⱼ² + Σ linⱼ·xⱼ  s.t.  Σ aᵢⱼ xⱼ + cᵢ = rhsᵢ + cᵢ`, boxed.

    A separable convex quadratic objective and linear equality rows, which is
    what the classifier needs to route the model to `pounce-convex` rather
    than the NLP path.

    `consts[i]` is written into row `i`'s *expression* segment rather than
    folded into its bound, with the bound raised to match. Before gh#492
    that made the row read as nonlinear and the whole model as an NLP; the
    reader now folds it at parse, so such a row reaches this pass like any
    other. That composition is new, so the family exercises both spellings
    — the rows are the same constraints either way, and the reduction and
    the recovered duals must not be able to tell.
    """
    m = len(rows)
    consts = consts or [0.0] * m
    jnnz = sum(len(r) for r in rows)
    L = [
        "g3 1 1 0\t# adversary convex doubleton aggregation",
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
        L += ["o5", f"v{j}", "n2"]
    L.append(f"x{n}")
    for j in range(n):
        L.append(f"{j} {x0[j]!r}")
    L.append("r")
    for i in range(m):
        L.append(f"4 {rhs[i] + consts[i]!r}")
    L.append("b")
    for j in range(n):
        lo, hi = x_l[j], x_u[j]
        if lo is None and hi is None:
            L.append("3")
        elif hi is None:
            L.append(f"2 {lo!r}")
        elif lo is None:
            L.append(f"1 {hi!r}")
        else:
            L.append(f"0 {lo!r} {hi!r}")
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
        L.append(f"{j} {lin[j]!r}")
    path.write_text("\n".join(L) + "\n")


def gen(rng):
    """One instance: alias chains over disjoint clusters, plus wider rows."""
    clusters = rng.randint(2, 5)
    sizes = [rng.randint(1, 4) for _ in range(clusters)]
    n = sum(sizes)
    x_star = [round(rng.uniform(-3, 3), 6) for _ in range(n)]

    # Column index ranges per cluster.
    spans, base = [], 0
    for s in sizes:
        spans.append(list(range(base, base + s)))
        base += s

    rows, rhs = [], []

    # A "blocking" instance covers one column of *every* cluster with its
    # first wide row, so no alias row survives the disjoint-source rule and
    # none of them is tightened. See the note on row order below.
    blocking = clusters >= 3 and rng.random() < 0.5
    blocking_cols = [rng.choice(sp) for sp in spans] if blocking else []

    def wide_rows():
        """Multi-cluster rows, which the aggregation must rewrite, not consume.

        Kept below `clusters` in number so the system stays underdetermined
        and the objective, not the rows, picks the point.
        """
        out = []
        if blocking:
            out.append([(c, round(rng.uniform(0.5, 2.0), 6)) for c in blocking_cols])
        for _ in range(rng.randint(0, max(0, clusters - 2))):
            cols = [rng.choice(sp) for sp in rng.sample(spans, min(3, clusters))]
            if len(set(cols)) < len(cols):
                continue
            out.append([(c, round(rng.uniform(0.5, 2.0), 6)) for c in cols])
        return out

    def alias_rows():
        out = []
        for span in spans:
            for k in range(len(span) - 1):
                i, j = span[k], span[k + 1]
                a = round(rng.uniform(0.4, 2.5), 6) * rng.choice([1, -1])
                b = round(rng.uniform(0.4, 2.5), 6) * rng.choice([1, -1])
                out.append([(i, a), (j, b)])
        return out

    # Row *order* is load-bearing, so draw both orders. The catalog's
    # bound-tightening pass walks the equality rows in index order and takes
    # a row as a source only if its columns are untouched by an earlier one.
    # Put the wide rows first and they claim the alias rows' columns, so the
    # alias rows go untightened and the eliminated column's box reaches its
    # survivor only through the aggregation — which is the one arrangement
    # that makes the transferred-bound multiplier need re-attributing. Put
    # the alias rows first and tightening gets there on its own. Both
    # arrangements are realistic; only one exercises that recovery.
    wide, alias = wide_rows(), alias_rows()
    for row in (wide + alias) if blocking or rng.random() < 0.5 else (alias + wide):
        rows.append(row)
        rhs.append(round(sum(a * x_star[c] for c, a in row), 9))
    if not rows:
        return None

    # Linear objective term. Large enough, half the time, to push the
    # unconstrained optimum out of the box so a bound goes active.
    push = rng.random() < 0.5
    lin = [
        round(rng.uniform(-12, 12) if push else rng.uniform(-1, 1), 6) for _ in range(n)
    ]

    x_l, x_u = [], []
    for j in range(n):
        # A blocking row only claims its columns if it actually tightens a
        # bound, and it can only tighten from finite ones — so those columns
        # get two-sided boxes rather than the free draw.
        shape = 0 if j in blocking_cols else rng.randrange(4)
        lo = round(x_star[j] - rng.uniform(0.2, 4.0), 6)
        hi = round(x_star[j] + rng.uniform(0.2, 4.0), 6)
        if shape == 0:
            x_l.append(lo), x_u.append(hi)
        elif shape == 1:
            x_l.append(lo), x_u.append(None)
        elif shape == 2:
            x_l.append(None), x_u.append(hi)
        else:
            x_l.append(None), x_u.append(None)
    x0 = [round(x_star[j] + rng.uniform(-0.2, 0.2), 6) for j in range(n)]
    # Most rows carry a constant in their expression segment; some carry none,
    # so both the gh#492 fold and the plain path stay live in every run.
    consts = [
        0.0 if rng.random() < 0.4 else round(rng.uniform(-3, 3), 6)
        for _ in rows
    ]
    return n, rows, rhs, lin, x_l, x_u, x0, consts


def run(nl_dir, opts):
    cmd = [str(POUNCE), "m", "-AMPL"] + opts
    return subprocess.run(cmd, cwd=nl_dir, capture_output=True, text=True, timeout=120)


def read_sol(path, n, m):
    nums = []
    for line in path.read_text().splitlines():
        if line.startswith("objno"):
            break
        try:
            nums.append(float(line.strip()))
        except ValueError:
            pass
    tail = nums[-(n + m) :]
    return tail[:m], tail[m:]


def objective(x, lin):
    return sum(v * v + lin[j] * v for j, v in enumerate(x))


def stationarity_fault(x, lam, rows, lin, x_l, x_u, sign, tol=1e-5):
    """First column whose reduced cost no valid bound multiplier can absorb.

    `∇f + s·Aᵀλ − z_l + z_u = 0` with `z_l, z_u ≥ 0` and complementarity
    means the residual must vanish where the column is interior, be `≥ 0`
    where it sits on its lower bound, and `≤ 0` on its upper — with no
    reference to the reported bound multipliers at all, so a solver that
    omits them (or attributes them to the wrong column) cannot hide here.
    """
    n = len(x)
    resid = [2.0 * x[j] + lin[j] for j in range(n)]
    for i, row in enumerate(rows):
        for c, a in row:
            resid[c] += sign * a * lam[i]
    scale = 1.0 + max((abs(r) for r in resid), default=0.0)
    for j in range(n):
        lo = -float("inf") if x_l[j] is None else x_l[j]
        hi = float("inf") if x_u[j] is None else x_u[j]
        at_lo = abs(x[j] - lo) <= 1e-6
        at_hi = abs(x[j] - hi) <= 1e-6
        r = resid[j]
        if at_lo and at_hi:
            continue  # a pinned column absorbs anything
        if at_lo:
            if r < -tol * scale:
                return f"x[{j}] on its lower bound with reduced cost {r:.3e} < 0"
        elif at_hi:
            if r > tol * scale:
                return f"x[{j}] on its upper bound with reduced cost {r:.3e} > 0"
        elif abs(r) > tol * scale:
            return f"x[{j}] interior with non-zero reduced cost {r:.3e}"
    return None


def infer_sign(x, lam, rows, lin, x_l, x_u):
    """Which `.sol` dual sign convention the baseline solve is written in.

    Established from the *un-presolved* solve, then held against the
    presolved one — so the check never adapts itself to whatever the
    reduction happened to produce.
    """
    for s in (-1.0, 1.0):
        if stationarity_fault(x, lam, rows, lin, x_l, x_u, s) is None:
            return s
    return None


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260806
    rng = random.Random(seed)

    bad, checked, reduced_any, at_bound = [], 0, 0, 0
    for t in range(trials):
        inst = gen(rng)
        if inst is None:
            continue
        n, rows, rhs, lin, x_l, x_u, x0, consts = inst
        m = len(rows)
        d = Path(tempfile.mkdtemp(prefix="adv_agg_"))
        try:
            write_nl(d / "m.nl", n, rows, rhs, lin, x_l, x_u, x0, consts)

            off = run(d, ["qp_presolve=no"])
            if "Optimal Solution Found" not in off.stdout:
                continue  # baseline could not solve it; not this pass's business
            lam_off, x_off = read_sol(d / "m.sol", n, m)
            sign = infer_sign(x_off, lam_off, rows, lin, x_l, x_u)
            if sign is None:
                # The baseline itself is not cleanly stationary (a degenerate
                # draw); it cannot establish the convention, so this instance
                # has nothing to say about the reduction.
                continue

            on = run(d, [])
            checked += 1
            if "aggregated 0" not in on.stdout and "Presolve:" in on.stdout:
                reduced_any += 1
            if "Optimal Solution Found" not in on.stdout:
                bad.append((t, "presolved solve did not reach optimality", on.stdout[-400:]))
                continue
            lam_on, x_on = read_sol(d / "m.sol", n, m)

            # --- oracle 1: independent KKT / feasibility check vs the .nl ---
            v = subprocess.run(
                [str(POUNCE), "verify", "m.nl", "m.sol"],
                cwd=d, capture_output=True, text=True, timeout=120,
            )
            if v.returncode != 0 or "VERIFIED" not in v.stdout:
                bad.append((t, f"pounce verify rejected the presolved solve (rc={v.returncode})",
                            v.stdout[-700:] + v.stderr[-300:]))
                continue

            # --- oracle 2: re-evaluate every row here, from the source data ---
            if len(x_on) != n or len(lam_on) != m:
                bad.append((t, f".sol shape wrong: {len(x_on)} primals / {len(lam_on)} duals "
                               f"for an {n}x{m} model", ""))
                continue
            fail = None
            for i, row in enumerate(rows):
                body = sum(a * x_on[c] for c, a in row)
                if abs(body - rhs[i]) > 1e-6 * max(1.0, abs(rhs[i])):
                    fail = f"row {i} violated by {body - rhs[i]:.3e} in the reported point"
                    break
            if fail is None:
                for j in range(n):
                    lo = -float("inf") if x_l[j] is None else x_l[j]
                    hi = float("inf") if x_u[j] is None else x_u[j]
                    if not (lo - 1e-6 <= x_on[j] <= hi + 1e-6):
                        fail = f"x[{j}]={x_on[j]} outside its declared box [{lo}, {hi}]"
                        break
                    if abs(x_on[j] - lo) < 1e-6 or abs(x_on[j] - hi) < 1e-6:
                        at_bound += 1
            if fail is None:
                # --- oracle 3: bound-aware stationarity of the reported duals ---
                fault = stationarity_fault(x_on, lam_on, rows, lin, x_l, x_u, sign)
                if fault is not None:
                    fail = f"presolved duals are not stationary: {fault}"
            if fail is None:
                # --- oracle 4: agreement with the un-reduced solve ---
                f_on, f_off = objective(x_on, lin), objective(x_off, lin)
                if abs(f_on - f_off) > 1e-5 * max(1.0, abs(f_off)):
                    fail = f"objective disagrees with the un-presolved solve: {f_on:.8e} vs {f_off:.8e}"
            if fail is not None:
                bad.append((t, fail, ""))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    print(f"instances solved and checked : {checked}")
    print(f"  where the pass aggregated  : {reduced_any}")
    print(f"  with a bound active at x*  : {at_bound}")
    print(f"failures                     : {len(bad)}")
    for t, why, extra in bad[:8]:
        print(f"  [{t}] {why}")
        if extra:
            print("      " + extra.replace("\n", "\n      ")[:600])
    print("VERDICT: PASS" if not bad else f"VERDICT: FAIL ({len(bad)} failures)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
