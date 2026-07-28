"""Adversary cross-check: auto-routing must REFUSE a nonconvex (indefinite) box QP

Family: autoroute   Class: nonconvex quadratic objective + box bounds
Source: Classic indefinite box-QP / "BoxQP" test family, cf. Nocedal & Wright,
        *Numerical Optimization* 2e, Ch. 16.1 (an indefinite quadratic over a
        box has its minimizers on the boundary; the interior stationary point
        is a saddle).  Instance A is the canonical 2-D saddle
        Q = [[1,2],[2,1]] (eigenvalues 3, -1) on [-1,1]^2.

  A)  min 0.5*(x1^2 + x2^2) + 2*x1*x2   s.t.  -1 <= x <= 1
      Interior stationary point (0,0) is a SADDLE, f=0.
      Global minimum f* = -1 at (1,-1) and (-1,1) (check the 4 vertices:
      (1,1)->3, (-1,-1)->3, (1,-1)->-1, (-1,1)->-1; edges/interior give a
      saddle or larger value).
  B)  min 0.5*x1^2 - 0.5*mu*x2^2  s.t. -1 <= x <= 1,  mu = 1e-3
      (barely indefinite: min eigenvalue -1e-3).  f* = -mu/2 = -5e-4 at
      x1=0, x2=+-1.  A convexity test with a sloppy tolerance would call this
      a convex QP and return x=(0,0), f=0 -- a WRONG answer, not merely a
      slower one.

The routing contract under test: `solver_selection="auto"` must agree with the
forced `solver_selection="nlp"` answer, and must NOT hand either problem to the
convex LP/QP or SOCP fast path (both assume a PSD Hessian).
"""

import time
import warnings

import numpy as np
from pounce import minimize

def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def run_case(name, Q, mu_note, x0, bounds, known_f, known_xs):
    Q = np.asarray(Q, float)

    def f(x):
        return 0.5 * float(x @ Q @ x)

    def gf(x):
        return Q @ x

    def hf(x):
        return Q

    # skepticism first: finite-difference my gradient/Hessian
    e = 1e-6
    xt = np.array([0.37, -0.61])
    fd = np.array(
        [(f(xt + e * u) - f(xt - e * u)) / (2 * e) for u in np.eye(2)]
    )
    assert np.allclose(fd, gf(xt), atol=1e-7), (fd, gf(xt))

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wa:
        warnings.simplefilter("always")
        r_auto = minimize(f, x0, jac=gf, hess=hf, bounds=bounds, solver_selection="auto")
    t_auto = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_nlp = minimize(f, x0, jac=gf, hess=hf, bounds=bounds, solver_selection="nlp")
    t_nlp = time.perf_counter() - t0

    # oracle 1: brute force over a dense grid + local polish (nonconvex ->
    # cvxpy is not applicable; scipy is the local oracle)
    from scipy.optimize import minimize as smin

    gr = np.linspace(bounds[0][0], bounds[0][1], 401)
    XX, YY = np.meshgrid(gr, gr)
    FF = 0.5 * (Q[0, 0] * XX**2 + 2 * Q[0, 1] * XX * YY + Q[1, 1] * YY**2)
    k = np.unravel_index(np.argmin(FF), FF.shape)
    grid_f = float(FF[k])
    grid_x = np.array([XX[k], YY[k]])

    # oracle 2: scipy local solve from the SAME x0 (the fair comparison for a
    # local NLP solver)
    t0 = time.perf_counter()
    s = smin(f, x0, jac=gf, bounds=bounds, method="L-BFGS-B", tol=1e-12)
    t_sci = time.perf_counter() - t0

    routed = r_auto.info.get("solver") if hasattr(r_auto, "info") else None
    specialized = routed is not None

    print(f"--- case {name}  ({mu_note}) ---")
    print(f"known f*={known_f:.10e} at {known_xs}")
    print(
        f"auto : status={r_auto.status} obj={r_auto.fun:.10e} x={np.asarray(r_auto.x)} "
        f"nit={r_auto.nit} t={t_auto:.4f}s route={routed!r}"
    )
    print(f"       warnings={[str(x.message)[:60] for x in wa]}")
    print(
        f"nlp  : status={r_nlp.status} obj={r_nlp.fun:.10e} x={np.asarray(r_nlp.x)} t={t_nlp:.4f}s"
    )
    print(f"grid : obj={grid_f:.10e} x={grid_x}")
    print(f"scipy: obj={s.fun:.10e} x={s.x} t={t_sci:.4f}s")

    d_auto_nlp = abs(float(r_auto.fun) - float(r_nlp.fun))
    e_known = rel(float(r_auto.fun), known_f)
    e_scipy = rel(float(r_auto.fun), float(s.fun))
    print(
        f"auto_vs_nlp_gap={d_auto_nlp:.2e} auto_vs_known={e_known:.2e} "
        f"auto_vs_scipy={e_scipy:.2e} specialized_route={specialized}"
    )
    ok = (
        bool(r_auto.success)
        and d_auto_nlp < 1e-6
        and e_known < 1e-4
        and not specialized  # a PSD-assuming route on an indefinite Q is a bug
    )
    return ok, d_auto_nlp, e_known, specialized


okA, *_ = run_case(
    "A indefinite saddle",
    [[1.0, 2.0], [2.0, 1.0]],
    "eigs 3,-1",
    np.array([0.3, -0.6]),
    [(-1.0, 1.0), (-1.0, 1.0)],
    -1.0,
    "(1,-1)/(-1,1)",
)
okB, *_ = run_case(
    "B barely indefinite",
    [[1.0, 0.0], [0.0, -1e-3]],
    "eigs 1,-1e-3",
    np.array([0.2, 0.4]),
    [(-1.0, 1.0), (-1.0, 1.0)],
    -5e-4,
    "(0,+-1)",
)

print("VERDICT: PASS" if (okA and okB) else "VERDICT: FAIL")
