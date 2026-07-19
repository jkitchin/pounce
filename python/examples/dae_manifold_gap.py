"""How far off the constraint manifold does a singular-mass DAE drift between
accepted steps, and what do ``solve_ivp``'s two projection knobs do about it?

`pounce.ode.solve_ivp(mass=M)` with a singular ``M`` is an index-1
differential-algebraic equation ``M y' = f``: the zero rows of ``M`` are pure
algebraic constraints ``0 = f_alg(t, y)``. Two knobs keep the reported solution
on that manifold, one for each of two GitHub issues:

* ``consistent="project"`` (default, gh #215) projects an inconsistent
  *initial* condition onto ``0 = f`` before integrating. :func:`inconsistent_ic`
  demonstrates it.
* ``project_output=True`` (opt-in, gh #216) Newton-polishes each requested
  *output* point (``res.sol(t)`` and ``res.y`` at ``t_eval``) back onto the
  manifold. :func:`manifold_gap` measures the gap it closes.

Headline result: Radau IIA is stiffly accurate, so the constraint holds to
round-off at the solver's accepted steps; only the dense-output polynomial
*between* steps can drift. For a **linear** conservation law (``sum(x) = 1``,
atom / charge / site balance) the degree-3 interpolant reproduces the
constraint *exactly*, so ``project_output`` buys nothing — it matters only for a
**nonlinear** algebraic constraint, and only when the absolute interpolated
residual is large enough to care about.

Point :func:`manifold_gap` at your own DAE to decide whether it is worth the
flag before turning it on.

Run: ``python examples/dae_manifold_gap.py``
"""

from __future__ import annotations

import numpy as np

import pounce.ode as po


def _banner(title):
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


def _algebraic_rows(M, tol=1e-9):
    """Indices of the algebraic equations: the (near-)zero rows of ``M``."""
    M = np.asarray(M, dtype=float)
    row = np.max(np.abs(M), axis=1)
    return np.where(row <= tol * max(1.0, float(row.max())))[0]


def constraint_is_linear(f, M, t, y, *, rtol=1e-6, eps=1e-6):
    """Is every algebraic row of ``f`` affine in ``y``?

    A degree-3 collocation dense output reproduces an affine constraint
    ``c'y = d`` exactly between steps — it vanishes at the three stage nodes and
    the on-manifold step start, and a cubic with four roots is identically zero
    — so :func:`manifold_gap` will show no gap and ``project_output`` is a no-op.
    Detected by central-differencing the algebraic-row Jacobian at ``y`` and at a
    few deterministic perturbations of ``y`` and checking it is constant.
    """
    y = np.asarray(y, dtype=float)
    alg = _algebraic_rows(M)

    def jac_alg(z):
        n = z.size
        J = np.empty((alg.size, n))
        for j in range(n):
            dz = np.zeros(n)
            dz[j] = eps * (1.0 + abs(z[j]))
            fp = np.asarray(f(t, z + dz), float)[alg]
            fm = np.asarray(f(t, z - dz), float)[alg]
            J[:, j] = (fp - fm) / (2.0 * dz[j])
        return J

    J0 = jac_alg(y)
    scale = 1.0 + float(np.linalg.norm(y))
    for k in (1, 2):
        dy = scale * 0.1 * k * np.cos(np.arange(y.size) + k)
        if np.linalg.norm(jac_alg(y + dy) - J0) > rtol * (1.0 + np.linalg.norm(J0)):
            return False
    return True


def manifold_gap(name, f, M, t_span, y0, *, rtol=1e-8, atol=1e-10, npts=100,
                 log_grid=False):
    """Measure ``max|f_alg|`` at accepted steps vs. on a requested output grid.

    Prints one row: the residual the solver itself holds (at steps), the
    interpolated residual on ``npts`` requested points with the dense output as
    is, and again with ``project_output=True``. The **absolute** interpolated
    residual is the number that matters; the step/interp ratio only tells you
    whether interpolation is exact (linear constraint) or not.
    """
    alg = _algebraic_rows(M)

    def alg_resid(tq, Y):                   # max over the algebraic rows, per pt
        res = np.array([np.asarray(f(tq[q], Y[:, q]), float)[alg]
                        for q in range(Y.shape[1])])
        return np.max(np.abs(res), axis=1)

    linear = constraint_is_linear(f, M, t_span[0], y0)

    r = po.solve_ivp(f, t_span, y0, mass=M, rtol=rtol, atol=atol,
                     dense_output=True)
    at_steps = float(np.max(alg_resid(r.t, r.y)))

    if log_grid:
        te = np.unique(np.concatenate(
            [[t_span[0]], np.logspace(np.log10(max(t_span[0], 1e-6) or 1e-6),
                                      np.log10(t_span[1]), npts)]))
    else:
        te = np.linspace(t_span[0], t_span[1], npts)

    interp = float(np.max(alg_resid(te, r.sol(te))))

    rp = po.solve_ivp(f, t_span, y0, mass=M, rtol=rtol, atol=atol,
                      t_eval=te, project_output=True)
    projected = float(np.max(alg_resid(te, rp.y)))

    ratio = interp / at_steps if at_steps > 0 else float("inf")
    print(f"  {name:<34} {'linear' if linear else 'nonlinear':<10} "
          f"steps={at_steps:.1e}  interp={interp:.1e}  "
          f"projected={projected:.1e}  gap={ratio:.0f}x")
    return dict(at_steps=at_steps, interp=interp, projected=projected,
                linear=linear)


def inconsistent_ic():
    """gh #215: a rough algebraic guess in ``y0`` is projected onto the manifold
    by default, matching ``solve_dae`` — the mass path used to echo it back."""
    _banner("gh #215 — inconsistent initial condition is projected by default")

    # 0 = y1 - y0**2  =>  a consistent IC with y0 = 1 requires y1 = 1. Pass y1 = 5.
    f = lambda t, y: [-y[0], y[1] - y[0] ** 2]
    M = np.diag([1.0, 0.0])

    r = po.solve_ivp(f, (0.0, 2.0), [1.0, 5.0], mass=M, rtol=1e-8, atol=1e-10)
    print(f"  solve_ivp (project, default) y0 -> {r.y[:, 0]}  "
          f"|alg resid| = {abs(r.y[1, 0] - r.y[0, 0] ** 2):.1e}")

    ra = po.solve_ivp(f, (0.0, 2.0), [1.0, 5.0], mass=M, consistent="assume",
                      rtol=1e-8, atol=1e-10)
    print(f"  solve_ivp (assume, opt out)  y0 -> {ra.y[:, 0]}  "
          f"|alg resid| = {abs(ra.y[1, 0] - ra.y[0, 0] ** 2):.1e}")

    F = lambda t, y, yp: np.array([yp[0] + y[0], y[1] - y[0] ** 2])
    d = po.solve_dae(F, (0.0, 2.0), [1.0, 5.0], rtol=1e-8, atol=1e-10)
    print(f"  solve_dae (project)          y0 -> {d.y[:, 0]}  "
          f"|alg resid| = {abs(d.y[1, 0] - d.y[0, 0] ** 2):.1e}")


def output_gap_table():
    """gh #216: the interpolation gap, for a linear vs. nonlinear constraint."""
    _banner("gh #216 — constraint residual: at steps vs. interpolated vs. projected")

    # Robertson kinetics: sum(y) = 1 is a LINEAR conservation law.
    k1, k2, k3 = 0.04, 3e7, 1e4
    robertson = lambda t, y: [-k1 * y[0] + k3 * y[1] * y[2],
                              k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] ** 2,
                              y[0] + y[1] + y[2] - 1.0]
    manifold_gap("Robertson (sum = 1)", robertson, np.diag([1.0, 1.0, 0.0]),
                 (0.0, 1e4), [1.0, 0.0, 0.0], rtol=1e-6, atol=1e-8, log_grid=True)

    # 0 = y1 - y0**2 is NONLINEAR: interpolation is not exact.
    square = lambda t, y: [-y[0], y[1] - y[0] ** 2]
    manifold_gap("0 = y1 - y0**2", square, np.diag([1.0, 0.0]),
                 (0.0, 2.0), [1.0, 1.0])

    # Langmuir PSSH: a fast-equilibrated surface coverage. Bilinear site balance
    # 0 = k_ads * c * (1 - theta) - k_des * theta  is also nonlinear.
    k_ads, k_des = 1.0, 1e3
    langmuir = lambda t, y: [-k_ads * y[0] * (1.0 - y[1]) + k_des * y[1],
                             k_ads * y[0] * (1.0 - y[1]) - k_des * y[1]]
    manifold_gap("Langmuir PSSH (site balance)", langmuir, np.diag([1.0, 0.0]),
                 (0.0, 5.0), [1.0, 0.0])

    print("\n  The absolute interpolated residual is what matters, not the gap "
          "ratio.\n  A linear constraint is exact under interpolation; "
          "project_output only\n  helps a nonlinear constraint whose absolute "
          "residual is large enough to care.")


if __name__ == "__main__":
    inconsistent_ic()
    output_gap_table()
