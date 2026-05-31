"""Probe whether any realistic regime needs the deferred matrix-free
(MINRES/GMRES) bwd path in JaxProblem.

For three scaling profiles we sweep n and time:
  * factor_reuse=False  (dense JAX bwd: assembles (n+m)x(n+m) per block)
  * factor_reuse=True   (Rust sparse LDLT factor + kkt_solve_many)
under two AD shapes:
  * single grad           (1 cotangent — value_and_grad style)
  * full jacrev wrt p     (n cotangents — Jacobian extraction)

Three problem families:
  A. 1D Poisson-like banded tridiagonal constraints     (sparse, structured)
  B. 2D 5-point Laplacian on a sqrt(n) x sqrt(n) grid    (sparse, fill grows)
  C. Dense low-rank-coupled constraints                  (worst case for LDLT)

The deferred MINRES path is justified iff there's a region where:
  - dense crashes / falls off the cubic cliff, AND
  - sparse LDLT either runs out of memory or takes longer than a
    matvec-based iterative would.

Run (from the repo root, with a Python that has pounce + jax installed):
    python python/benchmarks/probe_minres_regime.py
"""
from __future__ import annotations
import math
import time
import warnings

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", message=".*factor_reuse=False.*")

from pounce.jax import JaxProblem


def timed(fn, r=3, warmup=1, budget_s=30.0):
    """Best of r reps, but bail out if any single rep exceeds budget."""
    for _ in range(warmup):
        t0 = time.perf_counter()
        fn()
        if time.perf_counter() - t0 > budget_s:
            return float("inf")
    best = math.inf
    for _ in range(r):
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        if dt > budget_s:
            return float("inf")
        best = min(best, dt)
    return best * 1e3


def build_jp_1d_poisson(n, factor_reuse):
    """Family A: tridiagonal banded constraints (1D Poisson-like)."""
    m = n - 2

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    def g(x, p):
        # -x[i-1] + 2 x[i] - x[i+1] = p[i] for i in 1..n-2
        return -x[:-2] + 2.0 * x[1:-1] - x[2:] - p[1:-1]

    return JaxProblem(
        f=f, g=g, n=n, m=m, p_example=np.zeros(n),
        lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
        cl=jnp.zeros(m), cu=jnp.zeros(m),
        options={"tol": 1e-9, "print_level": 0, "sb": "yes"},
        factor_reuse=factor_reuse,
    )


def build_jp_2d_laplace(k, factor_reuse):
    """Family B: 2D 5-point Laplacian on a k x k interior grid (n=k^2,
    m=(k-2)^2). Forces non-trivial fill-in on the sparse LDL^T factor."""
    n = k * k
    interior = max(0, k - 2)
    m = interior * interior

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    def g(x, p):
        X = x.reshape(k, k)
        P = p.reshape(k, k)
        # 5-point stencil on interior
        center = X[1:-1, 1:-1]
        up = X[:-2, 1:-1]
        down = X[2:, 1:-1]
        left = X[1:-1, :-2]
        right = X[1:-1, 2:]
        rhs_p = P[1:-1, 1:-1]
        return (4.0 * center - up - down - left - right - rhs_p).reshape(-1)

    return JaxProblem(
        f=f, g=g, n=n, m=m, p_example=np.zeros(n),
        lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
        cl=jnp.zeros(m), cu=jnp.zeros(m),
        options={"tol": 1e-9, "print_level": 0, "sb": "yes"},
        factor_reuse=factor_reuse,
    )


def measure(jp, label):
    """Return (t_grad_ms, t_jacrev_ms). inf if the path falls over."""
    n = jp._n
    p0 = jnp.asarray(np.random.default_rng(0).standard_normal(n))
    x0 = jnp.zeros(n)

    # warm fwd to populate any caches
    try:
        jp.solve(p0, x0).block_until_ready()
    except Exception as e:
        return float("inf"), float("inf"), f"fwd failed: {type(e).__name__}"

    # value_and_grad (single cotangent)
    @jax.jit
    def vag(p):
        return jax.value_and_grad(lambda q: jnp.sum(jp.solve(q, x0) ** 2))(p)

    try:
        vag(p0)[0].block_until_ready()
        t_g = timed(lambda: vag(p0)[0].block_until_ready(), r=3, warmup=1)
    except Exception as e:
        t_g = float("inf")

    # jacrev (n cotangents)
    @jax.jit
    def jac(p):
        return jax.jacrev(lambda q: jp.solve(q, x0))(p)

    try:
        # Skip jacrev when n is large enough that the (n,n) jacobian itself
        # is hundreds of MB — that's not a fair test of the kernel, it's a
        # test of allocator pressure.
        if n * n * 8 > 4 * 1024**3:  # > 4 GB output
            t_j = float("inf")
        else:
            jac(p0).block_until_ready()
            t_j = timed(lambda: jac(p0).block_until_ready(), r=2, warmup=1,
                        budget_s=60.0)
    except Exception as e:
        t_j = float("inf")

    return t_g, t_j, None


print("=" * 86)
print("Probe: where does each backward path live, where would MINRES help?")
print("=" * 86)

print("\n--- Family A: 1D Poisson-like (tridiagonal sparse) ---")
print(f"{'n':>6s}  {'m':>6s}  {'reuse grad':>11s}  {'reuse jac':>11s}  "
      f"{'dense grad':>11s}  {'dense jac':>11s}  notes")
for n in [50, 200, 500, 1500, 4000]:
    m = n - 2
    note = ""
    # factor_reuse=True
    try:
        jp_T = build_jp_1d_poisson(n, factor_reuse=True)
        t_gT, t_jT, errT = measure(jp_T, "1D-T")
    except Exception as e:
        t_gT = t_jT = float("inf")
        errT = f"build: {type(e).__name__}"
    # factor_reuse=False
    try:
        jp_F = build_jp_1d_poisson(n, factor_reuse=False)
        t_gF, t_jF, errF = measure(jp_F, "1D-F")
    except Exception as e:
        t_gF = t_jF = float("inf")
        errF = f"build: {type(e).__name__}"
    notes = []
    if errT: notes.append(f"reuse:{errT}")
    if errF: notes.append(f"dense:{errF}")
    if n * (n + m) ** 2 * 8 > 1e9:
        notes.append("dense KKT > 1GB/block")
    print(f"{n:>6d}  {m:>6d}  {t_gT:>9.1f}ms  {t_jT:>9.1f}ms  "
          f"{t_gF:>9.1f}ms  {t_jF:>9.1f}ms  {' '.join(notes)}")

print("\n--- Family B: 2D Laplacian (sparse with growing fill-in) ---")
print(f"{'k':>4s}  {'n':>6s}  {'m':>6s}  {'reuse grad':>11s}  "
      f"{'reuse jac':>11s}  {'dense grad':>11s}  {'dense jac':>11s}  notes")
for k in [8, 16, 24, 32, 48, 64]:
    n = k * k
    m = max(0, (k - 2)) ** 2
    note = ""
    try:
        jp_T = build_jp_2d_laplace(k, factor_reuse=True)
        t_gT, t_jT, errT = measure(jp_T, "2D-T")
    except Exception as e:
        t_gT = t_jT = float("inf")
        errT = f"build: {type(e).__name__}"
    try:
        jp_F = build_jp_2d_laplace(k, factor_reuse=False)
        t_gF, t_jF, errF = measure(jp_F, "2D-F")
    except Exception as e:
        t_gF = t_jF = float("inf")
        errF = f"build: {type(e).__name__}"
    notes = []
    if errT: notes.append(f"reuse:{errT}")
    if errF: notes.append(f"dense:{errF}")
    fill_pred = "fill ~ O(n^1.5)" if n > 256 else ""
    if fill_pred: notes.append(fill_pred)
    print(f"{k:>4d}  {n:>6d}  {m:>6d}  {t_gT:>9.1f}ms  {t_jT:>9.1f}ms  "
          f"{t_gF:>9.1f}ms  {t_jF:>9.1f}ms  {' '.join(notes)}")

print()
print("Reading the table:")
print("  - 'inf' means the path crashed or exceeded the per-rep budget.")
print("  - reuse grad / reuse jac  → factor_reuse=True via Rust LDL^T.")
print("  - dense grad / dense jac  → factor_reuse=False via JAX vmap'd solve.")
print("  - If reuse-jac column stays bounded as n grows but dense-jac diverges,")
print("    Rust LDL^T is handling the regime — no MINRES needed.")
print("  - If reuse-jac diverges too, that's the MINRES motivating regime.")
