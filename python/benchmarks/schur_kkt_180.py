"""Corpus + perf validation for the block-triangular / Schur KKT solve (pounce#180 item 2).

Companion to `dev-notes/research/issue-180-item2-schur-kkt-scope.md` § Phase 3.
**Not a CI benchmark** — a hands-on harness to (a) confirm the Schur path solves
a scalable corpus to the *same* optimum as the standard full-space solver
(correctness / inertia parity, at scale, across shapes), and (b) characterise
where its linear-algebra cost stands versus the monolithic factorization, using
the per-solve timing breakdown shipped in item 3 (`info["timing"]`).

Corpus: a convex, equality-constrained NLP family

    min  Σ_i 0.5 d_i (x_i − t_i)²       (separable, strictly convex ⇒ PD primal)
    s.t. A x = b                         (m linear equalities)

parameterised by the primal size `n` and the constraint count `m` (m ≪ n). With
no inequalities the augmented KKT is `[[D+δ, Aᵀ],[A, −δ_c]]` of dimension
`n + m`; choosing the constraint-dual block `[n, n+m)` as the Schur set leaves
the positive-definite primal block as the eliminated `A_FF` — the classic
range-space split, and exactly the regime where `n_schur = m ≪ n`.

Two Jacobian shapes are swept because they stress the monolithic factorization
differently:
  * `banded`  — each constraint couples a contiguous window (sparse, local).
  * `spread`  — each constraint couples entries strided across all of `x`
                (a diffuse coupling AMD cannot localise as well).

Run:  python python/benchmarks/schur_kkt_180.py [--sizes 500,2000,8000] [--m 16]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import pounce


def make_problem(n, m, shape, seed=0):
    """Build (target, A, b) for the convex equality-constrained QP."""
    rng = np.random.default_rng(seed)
    target = rng.uniform(-1.0, 1.0, size=n)
    d = rng.uniform(0.5, 2.0, size=n)  # diagonal Hessian (PD)

    rows, cols, vals = [], [], []
    width = max(4, n // (2 * m))
    for i in range(m):
        if shape == "banded":
            start = (i * (n - width)) // max(1, m - 1) if m > 1 else 0
            idx = np.arange(start, start + width)
        else:  # spread: strided across the whole vector
            idx = (i + np.arange(width) * m) % n
            idx = np.unique(idx)
        coef = rng.uniform(-1.0, 1.0, size=idx.size)
        rows.extend([i] * idx.size)
        cols.extend(idx.tolist())
        vals.extend(coef.tolist())
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    vals = np.asarray(vals, dtype=float)

    # Feasible RHS from a random point *inside* the [-0.5, 0.5] box, so the
    # equality system is feasible together with the bounds.
    x_feas = rng.uniform(-0.4, 0.4, size=n)
    A_dense_row = np.zeros(m)
    b = np.zeros(m)
    for i in range(m):
        sel = rows == i
        b[i] = np.dot(vals[sel], x_feas[cols[sel]])
    _ = A_dense_row

    return target, d, (rows, cols, vals), b


def build(n, m, target, d, jac, b):
    rows, cols, vals = jac

    class QP:
        def objective(self, x):
            dd = x - target
            return 0.5 * float(np.dot(d * dd, dd))

        def gradient(self, x):
            return d * (x - target)

        def constraints(self, x):
            out = np.zeros(m)
            np.add.at(out, rows, vals * x[cols])
            return out - b

        def jacobianstructure(self):
            return (rows, cols)

        def jacobian(self, x):
            return vals

        def hessianstructure(self):
            ar = np.arange(n, dtype=np.int64)
            return (ar, ar)

        def hessian(self, x, lagrange, obj_factor):
            # Linear constraints ⇒ no Lagrangian contribution.
            return obj_factor * d

    # Box bounds tighter than the unconstrained optimum (`target`) so a good
    # fraction are active at the solution — this forces the IPM to reduce the
    # barrier over many iterations, exercising the Schur factor + resolve + the
    # per-iteration inertia check repeatedly (not a trivial one-Newton-step QP).
    lb = [-0.5] * n
    ub = [0.5] * n
    prob = pounce.Problem(
        n=n, m=m, problem_obj=QP(), lb=lb, ub=ub, cl=[0.0] * m, cu=[0.0] * m
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    prob.add_option("max_iter", 300)
    return prob


def solve_once(prob, x0, schur_indices=None):
    if schur_indices is not None:
        prob.set_kkt_schur_block(schur_indices)
    t0 = time.perf_counter()
    x, info = prob.solve(x0=x0)
    wall = time.perf_counter() - t0
    t = info.get("timing", {})
    return {
        "x": x,
        "status": info["status_msg"],
        "iters": int(info["iter_count"]),
        "obj": float(info["obj_val"]),
        "wall": wall,
        "factor": float(t.get("linear_system_factorization", float("nan"))),
        "backsolve": float(t.get("linear_system_back_solve", float("nan"))),
        "la_total": float(t.get("linear_system_total", float("nan"))),
        "overall": float(t.get("overall_alg", float("nan"))),
    }


def run(sizes, m, shapes):
    print(f"\npounce#180 item 2 — Schur KKT corpus/perf validation (m={m} equality constraints)\n")
    hdr = (
        f"{'shape':7} {'n':>7} {'dim':>7} | {'status(std/schur)':>22} "
        f"{'iters':>11} {'max|Δx|':>10} {'Δobj':>10} | "
        f"{'factor_std':>11} {'factor_schur':>13} {'speedup':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    all_ok = True
    for shape in shapes:
        for n in sizes:
            target, d, jac, b = make_problem(n, m, shape)
            dim = n + m
            schur = list(range(n, n + m))  # constraint-dual block
            x0 = np.zeros(n)

            std = solve_once(build(n, m, target, d, jac, b), x0, schur_indices=None)
            sch = solve_once(build(n, m, target, d, jac, b), x0, schur_indices=schur)

            dx = float(np.max(np.abs(std["x"] - sch["x"]))) if std["x"].shape == sch["x"].shape else float("inf")
            dobj = abs(std["obj"] - sch["obj"])
            ok = (
                std["status"] == "Solve_Succeeded"
                and sch["status"] == "Solve_Succeeded"
                and dx < 1e-6
            )
            all_ok = all_ok and ok
            speed = (std["factor"] / sch["factor"]) if sch["factor"] > 0 else float("nan")
            flag = "" if ok else "  <-- MISMATCH"
            print(
                f"{shape:7} {n:>7} {dim:>7} | "
                f"{std['status'][:10]:>10}/{sch['status'][:10]:<10} "
                f"{std['iters']:>4}/{sch['iters']:<5} {dx:>10.2e} {dobj:>10.2e} | "
                f"{std['factor']:>11.4f} {sch['factor']:>13.4f} {speed:>7.2f}x{flag}"
            )
    print()
    print("ALL SOLVES MATCHED" if all_ok else "*** SOME SOLVES MISMATCHED ***")
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="500,2000,8000")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--shapes", default="banded,spread")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    shapes = args.shapes.split(",")
    ok = run(sizes, args.m, shapes)
    raise SystemExit(0 if ok else 1)
