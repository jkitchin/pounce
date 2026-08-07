"""Independent adjudicator for the elastic-certificate fuzz.

Checks the *generator*, not pounce. If it reports GENERATOR BUG, nothing
the probes say can be trusted until that is fixed.

Two layers, because they answer different questions:

1. **Direct verification** (authoritative). Re-checks each instance's
   constructive claim in numpy, independently of the Rust that built it:

   - `feasible` — evaluate `A·x_w` and the box at the supplied witness.
     Feasibility is then arithmetic, not opinion.
   - `infeasible-contradictory-equalities` — confirm two rows really are
     identical and really are pinned to different values. `aᵀx` is a
     function; it cannot take two values.
   - `infeasible-box-range` — confirm `max_box aᵀx < bl`. A linear
     function over a box attains its extremes at corners, so the maximum
     is exact.

2. **scipy.optimize.linprog / HiGHS** (second opinion). An LP solver that
   has never heard of pounce or of this generator. Feasibility of a QP's
   constraint set is a pure LP question — the objective plays no part —
   so this is an exact reformulation.

HiGHS is *not* authoritative here, and conflating the two layers was a
bug in an earlier version of this script: it mapped every nonzero exit
status to "infeasible", so a numerical exit (status 4) or an unbounded
one (status 3, which *implies* a feasible point exists) read as a
contradiction. On ill-conditioned instances — cond(A) ~ 1e10, equality
rows scaled 1e4 apart — HiGHS can also return a genuine INFEASIBLE where
the witness satisfies every row to 0.0 in float arithmetic. That is a
tolerance artifact in the oracle, not an error in the instance, and it is
reported as a disagreement to inspect rather than a generator bug.

Usage:  python 2026-08-05_qp-active-set_adjudicate.py <instances.jsonl> [seed ...]
"""

import json
import sys

import numpy as np
from scipy.optimize import linprog

INF = 1e19


def unpack(rec):
    n, m = rec["n"], rec["m"]
    A = np.array(rec["a"], dtype=float).reshape(m, n)
    return (
        n,
        m,
        A,
        np.array(rec["bl"], dtype=float),
        np.array(rec["bu"], dtype=float),
        np.array(rec["xl"], dtype=float),
        np.array(rec["xu"], dtype=float),
    )


def verify_construction(rec):
    """Authoritative check of the instance's own claim. -> (ok, detail)."""
    n, m, A, bl, bu, xl, xu = unpack(rec)

    if rec["truth"] == "Feasible":
        w = np.array(rec["witness"], dtype=float)
        if w.shape != (n,):
            return False, "witness has the wrong length"
        ax = A @ w
        lo_gap = np.where(bl > -INF, ax - bl, np.inf).min(initial=np.inf)
        hi_gap = np.where(bu < INF, bu - ax, np.inf).min(initial=np.inf)
        box_gap = min((w - xl).min(initial=np.inf), (xu - w).min(initial=np.inf))
        worst = min(lo_gap, hi_gap, box_gap)
        # The row bounds were *derived from* this witness, so it should
        # satisfy them to rounding, scaled by the row magnitudes.
        scale = max(1.0, float(np.abs(A).max()), float(np.abs(ax).max()))
        return bool(worst >= -1e-9 * scale), f"tightest witness gap = {worst:.3e}"

    if rec["kind"] == "infeasible-contradictory-equalities":
        for i in range(m):
            for j in range(i + 1, m):
                if not np.array_equal(A[i], A[j]):
                    continue
                if bl[i] != bu[i] or bl[j] != bu[j]:
                    continue
                if bl[i] != bl[j]:
                    return True, (
                        f"rows {i},{j} identical, pinned to "
                        f"{bl[i]:.6e} and {bl[j]:.6e}"
                    )
        return False, "no contradictory identical equality pair found"

    if rec["kind"] == "infeasible-box-range":
        if np.any(xl <= -INF) or np.any(xu >= INF):
            return False, "box-range proof needs a finite box"
        for i in range(m):
            if bl[i] <= -INF:
                continue
            hi = np.stack([A[i] * xl, A[i] * xu]).max(axis=0).sum()
            if hi < bl[i]:
                return True, f"row {i}: max over box {hi:.6e} < required {bl[i]:.6e}"
        return False, "no row is provably out of range over the box"

    return False, f"unknown kind {rec['kind']!r}"


def highs(rec):
    """Second opinion. -> ('FEASIBLE'|'INFEASIBLE'|'INCONCLUSIVE', res)."""
    n, m, A, bl, bu, xl, xu = unpack(rec)
    ub_rows, ub_rhs, eq_rows, eq_rhs = [], [], [], []
    for i in range(m):
        if bl[i] == bu[i]:
            eq_rows.append(A[i])
            eq_rhs.append(bl[i])
            continue
        if bu[i] < INF:
            ub_rows.append(A[i])
            ub_rhs.append(bu[i])
        if bl[i] > -INF:
            ub_rows.append(-A[i])
            ub_rhs.append(-bl[i])

    res = linprog(
        c=np.zeros(n),
        A_ub=np.array(ub_rows) if ub_rows else None,
        b_ub=np.array(ub_rhs) if ub_rows else None,
        A_eq=np.array(eq_rows) if eq_rows else None,
        b_eq=np.array(eq_rhs) if eq_rows else None,
        bounds=[
            (None if l <= -INF else l, None if u >= INF else u)
            for l, u in zip(xl, xu)
        ],
        method="highs",
    )
    # 0 optimal, 2 infeasible, 3 unbounded (which *implies* a feasible
    # point exists), 1 iteration limit, 4 numerical.
    verdict = {0: "FEASIBLE", 2: "INFEASIBLE", 3: "FEASIBLE"}.get(
        res.status, "INCONCLUSIVE"
    )
    return verdict, res


def main():
    path = sys.argv[1]
    wanted = {int(s) for s in sys.argv[2:]}

    bad, disagree, inconclusive, ok = [], [], 0, 0
    for line in open(path):
        rec = json.loads(line)
        if wanted and rec["seed"] not in wanted:
            continue

        verified, detail = verify_construction(rec)
        verdict, res = highs(rec)
        expected = "FEASIBLE" if rec["truth"] == "Feasible" else "INFEASIBLE"

        if not verified:
            bad.append((rec, detail))
        elif verdict == "INCONCLUSIVE":
            inconclusive += 1
        elif verdict != expected:
            disagree.append((rec, detail))
        else:
            ok += 1

        if wanted:
            print(f"--- seed={rec['seed']} kind={rec['kind']}")
            print(f"    constructive truth : {rec['truth']} ({rec['proof']})")
            print(
                f"    direct check       : "
                f"{'VERIFIED' if verified else 'FAILED'} — {detail}"
            )
            print(f"    scipy/HiGHS        : {verdict} (status={res.status})")

    for rec, detail in bad:
        print(f"GENERATOR BUG seed={rec['seed']} kind={rec['kind']}: {detail}")
    for rec, detail in disagree:
        A = np.array(rec["a"], float).reshape(rec["m"], rec["n"])
        print(
            f"ORACLE DISAGREEMENT (inspect, not fatal) seed={rec['seed']} "
            f"kind={rec['kind']}: direct check verified {rec['truth']} "
            f"({detail}); HiGHS says otherwise. cond(A)={np.linalg.cond(A):.2e}"
        )

    total = ok + len(bad) + len(disagree) + inconclusive
    print()
    print(f"instances adjudicated     : {total}")
    print(f"construction verified     : {total - len(bad)}")
    print(f"HiGHS agrees              : {ok}")
    print(f"HiGHS disagrees (inspect) : {len(disagree)}")
    print(f"HiGHS inconclusive        : {inconclusive}")
    print("VERDICT: " + ("GENERATOR SOUND" if not bad else "GENERATOR BUG"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
