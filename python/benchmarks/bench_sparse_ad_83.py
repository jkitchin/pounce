"""Benchmark dense vs colored/compressed forward AD (issue #83).

The `sparse=True` path computes the constraint Jacobian and the
Lagrangian Hessian with CPR-style colored AD — one JVP/HVP per color
(`k` colors) instead of materializing the full dense matrix and slicing
it. This script measures the per-evaluation cost the IPM actually pays
(the `jacobian(x)` / `hessian(x, λ, σ)` callbacks), dense vs sparse,
across a sweep of `n`, plus the end-to-end solve wall time.

Two problem families:
  * banded   — tridiagonal-coupled constraints `g_i = x_i x_{i+1} - 1`.
               Genuinely sparse: the Jacobian and Lagrangian Hessian are
               tridiagonal, so coloring collapses them to a few seeds
               (k ~ 2-3) regardless of n. This is where compression wins.
  * dense    — every constraint touches every variable (`g_i = <a_i, x>²`
               style). Coloring gives k = n (no structural orthogonality),
               so sparse is pure overhead — the control that bounds the
               worst-case loss.

For each (family, n) we report the color counts and the best-of-r time
for producing the Jacobian and the Hessian, and the dense/sparse ratio.
A correctness check (sparse values == dense values) runs first so a
timing win can't come from computing the wrong thing.

Run (from the repo root, with a Python that has pounce + jax installed):
    python python/benchmarks/bench_sparse_ad_83.py
    python python/benchmarks/bench_sparse_ad_83.py --json out.json

Representative results (CPU, jax 0.10, single run — ratios are the
stable part; absolute ms vary by machine):

    per-eval forward AD
    family     n     m | k_jac  jacD    jacS      x | k_hes  hesD    hesS      x
    banded    800   799 |    2   0.69    0.11   6.2x |    3   0.51   0.25    2.0x
    banded  2000  1999 |    2   2.56    0.14  18.4x |    3   1.28   0.24    5.4x
    banded  5000  4999 |    2 103.80    0.19 559.6x |    3  86.14   0.43  202.0x
    dense    250   125 |  250   0.55    0.45   1.2x |  250   1.37   1.08    1.3x

    end-to-end solve (banded)
       n   solveD   solveS  speedup
     800   22.17    17.62     1.3x
    2000  231.99    30.56     7.6x

Takeaways: on a genuinely sparse (banded) problem the color count is
constant in n (k=2 Jacobian, k=3 Hessian), so the per-eval AD cost stays
~flat while the dense path grows ~linearly — a 560x Jacobian / 200x
Hessian gap by n=5000, and a 7.6x faster *full solve* at n=2000. On a
dense problem (k=n, no orthogonality to exploit) sparse is within ~1x:
the compression bookkeeping is cheap, so it's a small, bounded loss when
it can't help — which is why it's opt-in rather than the default.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pounce.jax import from_jax
from pounce.jax._build import _JaxProblem, _color_columns


def timed(fn, r=5, warmup=2, budget_s=20.0):
    """Best-of-r milliseconds; bail to inf if a rep blows the budget."""
    for _ in range(warmup):
        t0 = time.perf_counter()
        fn()
        if time.perf_counter() - t0 > budget_s:
            return math.inf
    best = math.inf
    for _ in range(r):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt > budget_s:
            return math.inf
        best = min(best, dt)
    return best * 1e3


# ---- problem families: return (f, g, n, m) ----

def banded(n):
    def f(x):
        return jnp.sum(x ** 2) + jnp.sum(jnp.sin(x))

    def g(x):  # tridiagonal Jacobian, tridiagonal Lagrangian Hessian
        return x[:-1] * x[1:] - 1.0

    return f, g, n, n - 1


def dense_family(n):
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((max(1, n // 2), n)))

    def f(x):
        return jnp.sum(x ** 2)

    def g(x):  # each row is a full quadratic in x → dense Jac & Hess
        return (A @ x) ** 2 - 1.0

    return f, g, n, A.shape[0]


FAMILIES = {"banded": banded, "dense": dense_family}


def bench_one(family_fn, n, rng):
    f, g, nn, m = family_fn(n)
    dense = _JaxProblem(f, g, n=nn, m=m, sparse=False)
    sparse = _JaxProblem(f, g, n=nn, m=m, sparse=True, n_probes=3)

    x = jnp.asarray(rng.standard_normal(nn))
    lam = jnp.asarray(rng.standard_normal(m))

    # Correctness gate.
    jd, js = dense.jacobian(x), sparse.jacobian(x)
    hd, hs = dense.hessian(x, lam, 1.0), sparse.hessian(x, lam, 1.0)
    jac_ok = np.allclose(jd, js, rtol=1e-10, atol=1e-10)
    hess_ok = np.allclose(hd, hs, rtol=1e-10, atol=1e-10)

    jac_colors, k_jac = _color_columns(dense._jac_rows, dense._jac_cols, nn)
    fr = np.concatenate([dense._hess_rows, dense._hess_cols])
    fc = np.concatenate([dense._hess_cols, dense._hess_rows])
    _, k_hess = _color_columns(fr, fc, nn)

    # Warm the JITs, then time the per-eval callbacks.
    for p in (dense, sparse):
        p.jacobian(x); p.hessian(x, lam, 1.0)
    t_jd = timed(lambda: dense.jacobian(x))
    t_js = timed(lambda: sparse.jacobian(x))
    t_hd = timed(lambda: dense.hessian(x, lam, 1.0))
    t_hs = timed(lambda: sparse.hessian(x, lam, 1.0))

    return {
        "n": nn, "m": m,
        "jac_nnz": int(dense._jac_rows.size), "k_jac": int(k_jac),
        "hess_nnz": int(dense._hess_rows.size), "k_hess": int(k_hess),
        "jac_ok": bool(jac_ok), "hess_ok": bool(hess_ok),
        "t_jac_dense_ms": t_jd, "t_jac_sparse_ms": t_js,
        "t_hess_dense_ms": t_hd, "t_hess_sparse_ms": t_hs,
        "jac_speedup": (t_jd / t_js) if t_js else math.inf,
        "hess_speedup": (t_hd / t_hs) if t_hs else math.inf,
    }


def bench_solve(n):
    """End-to-end solve wall time, dense vs sparse, on the banded family."""
    f, g, nn, m = banded(n)
    out = {}
    for sparse in (False, True):
        prob = from_jax(
            f, g, n=nn, m=m,
            lb=-1e19 * np.ones(nn), ub=1e19 * np.ones(nn),
            cl=np.zeros(m), cu=np.zeros(m), sparse=sparse,
        )
        prob.add_option("print_level", 0)
        prob.add_option("tol", 1e-8)
        prob.add_option("sb", "yes")
        x0 = np.full(nn, 1.1)
        prob.solve(x0=x0)  # warm
        out["sparse" if sparse else "dense"] = timed(
            lambda: prob.solve(x0=x0), r=3, warmup=1,
        )
    return {"n": nn, "t_solve_dense_ms": out["dense"],
            "t_solve_sparse_ms": out["sparse"],
            "solve_speedup": (out["dense"] / out["sparse"])
            if out["sparse"] else math.inf}


def fmt(x):
    return "   inf" if x == math.inf else f"{x:7.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="write raw results to this path")
    ap.add_argument("--quick", action="store_true", help="smaller sweep")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    banded_ns = [50, 200, 800] if args.quick else [50, 200, 800, 2000, 5000]
    dense_ns = [20, 60] if args.quick else [20, 60, 120, 250]

    results = {"per_eval": [], "solve": []}

    print("\n=== per-eval forward AD: dense vs colored (issue #83) ===")
    header = (
        f"{'family':8} {'n':>5} {'m':>5} | "
        f"{'k_jac':>5} {'jacD':>7} {'jacS':>7} {'x':>5} | "
        f"{'k_hes':>5} {'hesD':>7} {'hesS':>7} {'x':>5}  ok"
    )
    print(header)
    print("-" * len(header))
    for fam_name, ns in (("banded", banded_ns), ("dense", dense_ns)):
        for n in ns:
            r = bench_one(FAMILIES[fam_name], n, rng)
            r["family"] = fam_name
            results["per_eval"].append(r)
            ok = "✓" if (r["jac_ok"] and r["hess_ok"]) else "✗"
            print(
                f"{fam_name:8} {r['n']:>5} {r['m']:>5} | "
                f"{r['k_jac']:>5} {fmt(r['t_jac_dense_ms'])} "
                f"{fmt(r['t_jac_sparse_ms'])} {r['jac_speedup']:>4.1f}x | "
                f"{r['k_hess']:>5} {fmt(r['t_hess_dense_ms'])} "
                f"{fmt(r['t_hess_sparse_ms'])} {r['hess_speedup']:>4.1f}x  {ok}"
            )

    print("\n=== end-to-end solve (banded family) ===")
    print(f"{'n':>5} {'solveD':>9} {'solveS':>9} {'speedup':>8}")
    print("-" * 34)
    for n in ([200, 800] if args.quick else [200, 800, 2000]):
        r = bench_solve(n)
        results["solve"].append(r)
        print(f"{r['n']:>5} {fmt(r['t_solve_dense_ms'])}  "
              f"{fmt(r['t_solve_sparse_ms'])}  {r['solve_speedup']:>6.1f}x")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
