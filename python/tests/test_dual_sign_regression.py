"""Dual-SIGN regression guard for the Python / pyomo dual surfaces (issue #294).

Why this test exists (the #271 post-mortem)
-------------------------------------------
A constraint-dual sign inversion (#271/#272, fixed in #287) flipped every
AMPL/Pyomo/GAMS marginal for an unknown span of releases and NO automated check
caught it. The benchmark suite compares objectives/status/iterations/wall-time
and never duals, so a flip that leaves primals and objectives exact is invisible
to it; and ``pounce verify`` evaluates KKT stationarity for both ``+λ`` and
``−λ`` and keeps the better residual, so it certifies either sign.

The key principle: **agreement between pounce's own surfaces is not a guard** — a
uniform flip satisfies it. Each assertion below therefore pins a dual against an
EXTERNAL reference (IPOPT, via pyomo) or an ANALYTIC value with an explicit
expected SIGN, so a future uniform flip fails loudly.

Analytic references
-------------------
* Wyndor Glass Co. LP (textbook): ``max 3x1+5x2 s.t. x1≤4, 2x2≤12, 3x1+2x2≤18``,
  optimum 36 at (2, 6), shadow prices (0, 1.5, 1). Written to pyomo both as a
  maximize and as the equivalent ``min −3x1−5x2``.
* Strictly convex equality QP: ``min x0²+x1² s.t. x0+x1=2``, optimum (1, 1);
  KKT ``2x + Aᵀy = 0`` gives the equality multiplier ``y = −2`` exactly, and
  the pyomo/AMPL marginal ``d obj / d b = +2``.

The sibling Rust test ``issue_294_dual_sign_regression.rs`` pins the ``.sol``
and JSON ``solution.lambda`` surfaces; ``test_gams_link.py`` pins the GAMS ``pi``
conversion for both senses.
"""

from __future__ import annotations

import numpy as np
import pytest

import pounce
from pounce.qp import solve_qp

# Textbook shadow prices for the Wyndor LP constraints (c1, c2, c3).
WYNDOR_SHADOW = np.array([0.0, 1.5, 1.0])


# ── pyomo model.dual — the surface #271 flipped, checked against IPOPT ────────


def _build_wyndor_pyomo(pyo, maximize):
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(0, None), initialize=0.0)
    m.x2 = pyo.Var(bounds=(0, None), initialize=0.0)
    if maximize:
        m.obj = pyo.Objective(expr=3 * m.x1 + 5 * m.x2, sense=pyo.maximize)
    else:
        m.obj = pyo.Objective(expr=-3 * m.x1 - 5 * m.x2, sense=pyo.minimize)
    m.c1 = pyo.Constraint(expr=m.x1 <= 4)
    m.c2 = pyo.Constraint(expr=2 * m.x2 <= 12)
    m.c3 = pyo.Constraint(expr=3 * m.x1 + 2 * m.x2 <= 18)
    return m


def _solve_pyomo_duals(pyo, solver_name, maximize):
    m = _build_wyndor_pyomo(pyo, maximize)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    pyo.SolverFactory(solver_name).solve(m)
    x = np.array([pyo.value(m.x1), pyo.value(m.x2)])
    duals = np.array([m.dual[c] for c in (m.c1, m.c2, m.c3)])
    return x, duals


@pytest.mark.parametrize(
    "maximize, expected",
    [
        # A maximize model's marginals are the natural shadow prices, +sign.
        (True, WYNDOR_SHADOW),
        # The equivalent minimize model's marginals are the exact negation.
        (False, -WYNDOR_SHADOW),
    ],
)
def test_pyomo_model_dual_sign(maximize, expected):
    """``SolverFactory('pounce')`` model.dual must carry the shadow price with
    the sign IPOPT/AMPL use: +1.5/+1 for the maximize model, −1.5/−1 for the
    equivalent minimize. The explicit sign is the guard — a uniform flip
    (the #271 defect) fails these exact-value asserts."""
    pyo = pytest.importorskip("pyomo.environ")
    pytest.importorskip("pyomo_pounce")  # registers 'pounce'
    import pyomo_pounce  # noqa: F401

    x, duals = _solve_pyomo_duals(pyo, "pounce", maximize)
    np.testing.assert_allclose(x, [2.0, 6.0], atol=1e-4)
    np.testing.assert_allclose(duals, expected, atol=1e-4)


@pytest.mark.parametrize("maximize", [True, False])
def test_pyomo_pounce_matches_ipopt_duals(maximize):
    """External-reference guard: pounce's pyomo duals must agree with IPOPT's
    on the SAME model, sign and value. This is the check the benchmark corpus
    lacks — agreement with an independent solver, not with pounce itself."""
    pyo = pytest.importorskip("pyomo.environ")
    pytest.importorskip("pyomo_pounce")
    import pyomo_pounce  # noqa: F401

    if not pyo.SolverFactory("ipopt").available(exception_flag=False):
        pytest.skip("ipopt not available for the external cross-check")

    _, ref = _solve_pyomo_duals(pyo, "ipopt", maximize)
    _, got = _solve_pyomo_duals(pyo, "pounce", maximize)
    np.testing.assert_allclose(got, ref, atol=1e-4)
    # And the sign matches the analytic shadow price, not just IPOPT.
    np.testing.assert_allclose(
        got, WYNDOR_SHADOW if maximize else -WYNDOR_SHADOW, atol=1e-4
    )


# ── pounce.minimize(...).info["mult_g"] — Lagrange convention = −marginal ─────


def test_minimize_mult_g_sign():
    """``info['mult_g']`` keeps the *Lagrange multiplier* convention (= −marginal).

    For ``min −3x1−5x2`` with the three ``Gx ≤ h`` rows, the KKT multipliers of
    ``L = f + Σ λᵢ (gᵢ − hᵢ)`` (λᵢ ≥ 0) are the analytic shadow prices
    ``[0, 1.5, 1]``. That is the opposite sign to the minimize-model AMPL
    marginal ``[0, −1.5, −1]``; assert against the Lagrange value, not the
    marginal."""
    from scipy.optimize import LinearConstraint

    G = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
    h = np.array([4.0, 12.0, 18.0])
    res = pounce.minimize(
        lambda x: -3.0 * x[0] - 5.0 * x[1],
        x0=[0.0, 0.0],
        jac=lambda x: np.array([-3.0, -5.0]),
        constraints=[LinearConstraint(G, -np.inf, h)],
        bounds=[(0, None), (0, None)],
    )
    np.testing.assert_allclose(res.x, [2.0, 6.0], atol=1e-4)
    mult_g = np.asarray(res.info["mult_g"], dtype=float)
    # Explicit signed analytic value: ≥ 0, NOT the negated marginal.
    np.testing.assert_allclose(mult_g, WYNDOR_SHADOW, atol=1e-4)
    assert mult_g[1] > 0 and mult_g[2] > 0, "mult_g must be the +λ Lagrange sign"


# ── solve_qp y / z — convex path ─────────────────────────────────────────────


def test_solve_qp_inequality_multipliers_sign():
    """``QpResult.z`` are the ≥ 0 inequality multipliers. For the Wyndor LP as
    ``min −3x1−5x2 s.t. Gx ≤ h`` they equal the analytic shadow prices
    ``[0, 1.5, 1]`` — a uniform flip would make the active ones negative."""
    G = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
    h = np.array([4.0, 12.0, 18.0])
    r = solve_qp(P=np.zeros((2, 2)), c=[-3.0, -5.0], G=G, h=h, lb=[0.0, 0.0])
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [2.0, 6.0], atol=1e-4)
    np.testing.assert_allclose(r.z, WYNDOR_SHADOW, atol=1e-4)
    assert r.z[1] > 0 and r.z[2] > 0, "active inequality multipliers must be ≥ 0"


def test_solve_qp_equality_multiplier_sign():
    """``QpResult.y`` is the equality multiplier. For ``min x0²+x1² s.t.
    x0+x1=2`` (P = 2I), KKT ``2x + Aᵀy = 0`` at x=(1,1) gives ``y = −2``
    exactly — pin the sign so a flip to +2 fails."""
    r = solve_qp(
        P=2.0 * np.eye(2),
        c=[0.0, 0.0],
        A=np.array([[1.0, 1.0]]),
        b=[2.0],
    )
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(r.y, [-2.0], atol=1e-5)
