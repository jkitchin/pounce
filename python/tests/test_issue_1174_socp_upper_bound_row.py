"""gh #1174: a convex quadratic written ``NonlinearConstraint(q, -inf, ub)``.

The QCQP router read every quadratic row as its ``g ≥ lo`` side only, so the
natural scipy spelling ``q(x) ≤ ub`` with ``q`` convex failed the concavity
test: ``socp`` raised and ``auto`` fell back to the NLP solver. The ``g ≤ hi``
side is now read with the mirrored sign. A two-sided quadratic row (a band
between two level sets, e.g. an annulus) stays rejected, as does a one-sided
row whose feasible set is nonconvex.
"""

import numpy as np
import pytest
from pounce import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint

C = np.array([1.0, -2.0, 0.5])
ORACLE = -np.sqrt(2.0) * np.linalg.norm(C)  # min c·x s.t. ‖x‖² ≤ 2

SPELLINGS = {
    "nlc_upper": NonlinearConstraint(lambda x: x @ x, -np.inf, 2.0),
    "nlc_upper_jac": NonlinearConstraint(
        lambda x: x @ x, -np.inf, 2.0, jac=lambda x: 2.0 * x[None, :]
    ),
    "nlc_lower": NonlinearConstraint(lambda x: 2.0 - x @ x, 0.0, np.inf),
    "dict_ineq": {"type": "ineq", "fun": lambda x: 2.0 - x @ x},
}


@pytest.mark.parametrize("selection", ["socp", "auto"])
@pytest.mark.parametrize("name", sorted(SPELLINGS))
def test_every_spelling_routes_to_socp(name, selection):
    res = minimize(
        lambda x: float(C @ x),
        np.zeros(3),
        constraints=[SPELLINGS[name]],
        solver_selection=selection,
    )
    assert res.info.get("solver") == "socp"
    assert res.success, res.message
    assert res.fun == pytest.approx(ORACLE, abs=1e-5)


def test_upper_bound_row_with_linear_constraint():
    # min c·x s.t. ‖x‖² ≤ 2, x₀ ≥ 0 (binding: the ball optimum has x₀ < 0).
    cons = [
        NonlinearConstraint(lambda x: x @ x, -np.inf, 2.0),
        LinearConstraint(np.array([[1.0, 0.0, 0.0]]), 0.0, np.inf),
    ]
    fun = lambda x: float(C @ x)
    conic = minimize(fun, np.zeros(3), constraints=cons, solver_selection="socp")
    nlp = minimize(fun, np.zeros(3), constraints=cons, solver_selection="nlp")
    assert conic.info.get("solver") == "socp"
    assert conic.success and nlp.success
    assert conic.fun == pytest.approx(-np.sqrt(2.0) * np.linalg.norm(C[1:]), abs=1e-5)
    assert conic.fun == pytest.approx(nlp.fun, abs=1e-5)


def test_offset_upper_bound_ball():
    # (x − 3)² + (y + 1)² ≤ 1, min x + y  ⇒  f* = 2 − √2.
    con = NonlinearConstraint(
        lambda x: (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2, -np.inf, 1.0
    )
    res = minimize(
        lambda x: x[0] + x[1],
        np.array([3.0, -1.0]),
        constraints=[con],
        solver_selection="socp",
    )
    assert res.info.get("solver") == "socp"
    assert res.fun == pytest.approx(2.0 - np.sqrt(2.0), abs=1e-5)


# The rejecting branches: each must stay off the conic path.
NONCONVEX = {
    # q ≤ ub with q concave: the outside of a ball.
    "concave_upper": NonlinearConstraint(lambda x: -(x @ x), -np.inf, -1.0),
    # q ≥ lo with q convex: also the outside of a ball.
    "convex_lower": NonlinearConstraint(lambda x: x @ x, 1.0, np.inf),
    # Annulus, both sides finite, either sign of Q.
    "annulus_neg": NonlinearConstraint(lambda x: -(x @ x), -4.0, -1.0),
    "annulus_pos": NonlinearConstraint(lambda x: x @ x, 1.0, 4.0),
}


@pytest.mark.parametrize("name", sorted(NONCONVEX))
def test_nonconvex_rows_are_rejected(name):
    with pytest.raises(ValueError, match="not detected as a convex QCQP"):
        minimize(
            lambda x: float(C @ x),
            np.full(3, 1.5),
            constraints=[NONCONVEX[name]],
            solver_selection="socp",
        )


@pytest.mark.parametrize("name", sorted(NONCONVEX))
def test_nonconvex_rows_do_not_auto_route_to_socp(name):
    res = minimize(
        lambda x: float(C @ x),
        np.full(3, 1.5),
        constraints=[NONCONVEX[name]],
        solver_selection="auto",
    )
    assert res.info.get("solver") != "socp"


def test_annulus_inner_side_is_not_dropped():
    # Before the fix the router read only ``g ≥ lo`` of a two-sided quadratic
    # row and dropped ``g ≤ hi``: this annulus routed to ``socp`` as the disc
    # ‖x‖² ≤ 4 and returned f = 0 with success. f* = 1, on the inner circle.
    con = NonlinearConstraint(lambda x: -(x @ x), -4.0, -1.0)
    res = minimize(
        lambda x: float(x @ x),
        np.full(2, 1.0),
        constraints=[con],
        solver_selection="auto",
    )
    assert res.info.get("solver") != "socp"
    assert res.success, res.message
    assert res.fun == pytest.approx(1.0, abs=1e-6)
