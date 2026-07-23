"""Polynomial global optimization via SOS (pounce.sos.sos_minimize).

Polynomials are dicts {exponent_tuple: coefficient}; the solver returns a
certified global lower bound and (when the moment matrix is flat) the global
minimizers extracted from the moment matrix.
"""

import numpy as np
import pytest

import pounce
from pounce.sos import SosResult, sos_minimize


def test_top_level_export():
    assert pounce.sos_minimize is sos_minimize
    assert pounce.SosResult is SosResult


def test_univariate_quartic_two_minimizers():
    # x⁴ − 2x² + 3 → min 2 at x = ±1.
    r = sos_minimize({(4,): 1.0, (2,): -2.0, (0,): 3.0})
    assert r.success
    assert abs(r.lower_bound - 2.0) < 1e-5
    assert r.is_exact and r.num_minimizers == 2
    roots = sorted(float(m[0]) for m in r.minimizers)
    assert abs(roots[0] + 1.0) < 1e-3 and abs(roots[1] - 1.0) < 1e-3


def test_facial_reduction_nonunique_minimizers():
    # (x²−1)² + y² → min 0 at (±1, 0). Non-unique optimum: the interior-point
    # solver's central moment matrix is rank-inflated, so flat truncation only
    # succeeds via the facial-reduction (trace-penalty) re-solve.
    p = {(4, 0): 1.0, (2, 0): -2.0, (0, 0): 1.0, (0, 2): 1.0}
    r = sos_minimize(p)
    assert r.success
    assert abs(r.lower_bound) < 1e-5
    assert r.is_exact and r.num_minimizers == 2
    xs = sorted(float(m[0]) for m in r.minimizers)
    assert abs(xs[0] + 1.0) < 1e-2 and abs(xs[1] - 1.0) < 1e-2
    assert all(abs(float(m[1])) < 1e-2 for m in r.minimizers)


def test_facial_reduction_four_minimizers_order_three():
    # (x²−1)² + (y²−1)² → four global minima (value 0) at (±1, ±1). Needs the
    # order-3 relaxation, a larger degenerate SDP that the solver now carries to
    # optimality (homogeneous self-dual embedding) so all four atoms come out.
    p = {
        (4, 0): 1.0,
        (2, 0): -2.0,
        (0, 4): 1.0,
        (0, 2): -2.0,
        (0, 0): 2.0,
    }
    r = sos_minimize(p, order=3)
    assert r.success
    assert abs(r.lower_bound) < 1e-5
    assert r.is_exact and r.num_minimizers == 4
    quads = {(float(m[0]) > 0, float(m[1]) > 0) for m in r.minimizers}
    assert len(quads) == 4, f"expected all four quadrants, got {r.minimizers}"
    for m in r.minimizers:
        assert abs(abs(float(m[0])) - 1.0) < 2e-2
        assert abs(abs(float(m[1])) - 1.0) < 2e-2


def test_unique_minimizer_2d():
    # (x−1)² + (y−2)² → min 0 at (1, 2).
    p = {(2, 0): 1.0, (1, 0): -2.0, (0, 2): 1.0, (0, 1): -4.0, (0, 0): 5.0}
    r = sos_minimize(p)
    assert r.success and r.is_exact
    assert r.num_minimizers == 1
    np.testing.assert_allclose(r.minimizers[0], [1.0, 2.0], atol=1e-3)
    assert abs(r.lower_bound) < 1e-5


def _boxed_rosenbrock(lo=-2.0, hi=2.0):
    # f = (1-x)^2 + 100(y - x^2)^2, unique global min (1,1), f* = 0.
    f = {(0, 0): 1.0, (1, 0): -2.0, (2, 0): 1.0, (0, 2): 100.0, (2, 1): -200.0, (4, 0): 100.0}
    box = []
    for j in range(2):
        e = lambda s: tuple(s if k == j else 0 for k in range(2))  # noqa: E731
        box.append({e(1): 1.0, (0, 0): -lo})
        box.append({e(1): -1.0, (0, 0): hi})
    return f, box


def _eval_poly(f, x):
    return sum(c * x[0] ** a * x[1] ** b for (a, b), c in f.items())


@pytest.mark.parametrize("order", [2, 3, 4])
def test_boxed_rosenbrock_never_certifies_a_non_minimizing_atom(order):
    # gh #281. On boxed Rosenbrock the moment relaxation is not flat at the true
    # measure: the first moments the SDP returns lie in the flat "banana" valley
    # (~(0.86, 0.74)), a point 0.26 from the true minimizer (1,1) whose objective
    # (~0.02) still reads close to the correct bound (~0). It used to come back as
    # is_exact=True, num_minimizers=1 -- a confidently wrong minimizer.
    #
    # The invariant that must always hold: is_exact => every reported minimizer
    # attains the lower bound. Either the tight, correct (1,1) with is_exact=True,
    # or the safe failure (is_exact=False, no minimizers) -- never a wrong point
    # claimed exact.
    f, box = _boxed_rosenbrock()
    r = sos_minimize(f, inequalities=box, n_vars=2, order=order)
    assert r.success, f"order {order}: status {r.status}"
    # The lower bound stays sound regardless.
    assert r.lower_bound <= 1e-4, f"order {order}: bound {r.lower_bound} exceeds 0"
    if r.is_exact:
        assert r.num_minimizers == len(r.minimizers)
        for m in r.minimizers:
            fx = _eval_poly(f, m)
            assert abs(fx - r.lower_bound) <= 1e-3, (
                f"order {order}: is_exact but minimizer {m} has f={fx:.3e}, "
                f"missing the bound {r.lower_bound:.3e}"
            )
            np.testing.assert_allclose(m, [1.0, 1.0], atol=1e-2)
    else:
        assert r.num_minimizers == 0
        assert len(r.minimizers) == 0


def test_constrained_box_nonconvex():
    # min −x  s.t.  1 − x² ≥ 0  (x ∈ [−1,1])  →  −1 at x = 1.
    r = sos_minimize({(1,): -1.0}, inequalities=[{(0,): 1.0, (2,): -1.0}])
    assert r.success
    assert abs(r.lower_bound + 1.0) < 1e-5


def test_equality_constraint():
    # min x² + y²  s.t.  x + y − 2 = 0  →  2 at (1,1).
    r = sos_minimize(
        {(2, 0): 1.0, (0, 2): 1.0},
        equalities=[{(1, 0): 1.0, (0, 1): 1.0, (0, 0): -2.0}],
    )
    assert r.success
    assert abs(r.lower_bound - 2.0) < 1e-5


def test_explicit_n_vars_and_order():
    # A constant in 2 vars: n_vars can't be inferred from a single (0,0) term
    # ambiguously, but order can be raised without changing the bound.
    r = sos_minimize({(0, 0): 5.0}, n_vars=2, order=2)
    assert r.success
    assert abs(r.lower_bound - 5.0) < 1e-6


def test_mismatched_exponent_length_raises():
    with pytest.raises(ValueError):
        sos_minimize({(2, 0): 1.0, (1,): -2.0})  # inconsistent tuple lengths


# --- constrained quartic hierarchy (gh #218) --------------------------------


def _lasserre_ex5():
    """Lasserre, SIAM J. Optim. 11(3):796-817 (2001), Example 5.

    min -x1 - x2 over two quartic constraints and the box [0,3]x[0,4].
    Known global minimum -5.50801 at (2.3295, 3.1783).
    """
    obj = {(1, 0): -1.0, (0, 1): -1.0}
    g = [
        {(4, 0): 2.0, (3, 0): -8.0, (2, 0): 8.0, (0, 0): 2.0, (0, 1): -1.0},
        {(4, 0): 4.0, (3, 0): -32.0, (2, 0): 88.0, (1, 0): -96.0, (0, 0): 36.0,
         (0, 1): -1.0},
        {(1, 0): 1.0},
        {(0, 0): 3.0, (1, 0): -1.0},
        {(0, 1): 1.0},
        {(0, 0): 4.0, (0, 1): -1.0},
    ]
    return obj, g


TRUE_MIN = -5.508013271595  # independently verified, 400-start SLSQP


def test_constrained_quartic_hierarchy_tightens_to_the_global_minimum():
    # The acceptance criterion from gh #218, verbatim: "a finite bound
    # <= -5.50801, tightening toward it as the order rises." Previously
    # order 2 returned the trivial box bound -7 and orders 3/4 returned nan.
    obj, g = _lasserre_ex5()
    bounds = []
    for order in (2, 3, 4):
        r = sos_minimize(obj, inequalities=g, order=order)
        assert r.success, f"order {order}: status {r.status}"
        assert np.isfinite(r.lower_bound), f"order {order}: bound is not finite"
        # Soundness, asserted strictly: the bound is certified, so it must
        # genuinely lie below the true minimum -- no tolerance slack.
        assert r.certified, f"order {order}: a boxed problem must certify"
        assert r.lower_bound <= TRUE_MIN, f"order {order}: {r.lower_bound}"
        bounds.append(r.lower_bound)
    # Monotone and genuinely tightening, not stuck at the trivial box bound.
    assert bounds[0] < bounds[1] < bounds[2], bounds
    assert abs(bounds[0] + 7.0) < 1e-4, "order 2 is the trivial box bound"
    assert abs(bounds[2] - TRUE_MIN) < 1e-4, f"order 4 should be exact: {bounds[2]}"
    assert bounds[0] <= TRUE_MIN and bounds[1] <= TRUE_MIN


def test_constrained_quartic_order_four_extracts_the_minimizer():
    # At order 4 the relaxation is exact, so the global minimizer comes back too.
    obj, g = _lasserre_ex5()
    r = sos_minimize(obj, inequalities=g, order=4)
    assert r.success and r.is_exact
    assert r.num_minimizers >= 1
    x = r.minimizers[0]
    np.testing.assert_allclose(x, [2.3295, 3.1783], atol=1e-3)


def test_reported_order_identifies_the_relaxation_that_produced_the_bound():
    # A bound from a coarser order is still valid, so sos_minimize falls back
    # rather than discarding one it already proved -- and `order` says which
    # relaxation the reported bound actually came from.
    obj, g = _lasserre_ex5()
    for requested in (2, 3, 4):
        r = sos_minimize(obj, inequalities=g, order=requested)
        assert r.order <= requested
        assert r.lower_bound <= TRUE_MIN
    # Below the minimum admissible order (the quartics force d >= 2), the
    # request is raised rather than rejected.
    r = sos_minimize(obj, inequalities=g, order=1)
    assert r.success and r.order == 2


# --- certification and solver controls (gh #218 caveats) --------------------


def test_certified_bound_is_below_the_true_minimum():
    # A converged SDP reports a value that is accurate but not necessarily a
    # lower *bound*: the raw order-4 value landed 2.2e-7 ABOVE the true
    # minimum. The certified value subtracts the SOS identity's measured miss,
    # so it is valid regardless of how the solve went.
    obj, g = _lasserre_ex5()
    r = sos_minimize(obj, inequalities=g, order=4)
    assert r.certified
    assert r.lower_bound <= TRUE_MIN, r.lower_bound
    # Valid, and not uselessly loose.
    assert r.lower_bound >= TRUE_MIN - 1e-4, r.lower_bound


def test_certification_withheld_on_an_unbounded_domain():
    # Certification needs the feasible set inside a box; without one the
    # residual cannot be bounded, so the flag must report the truth rather
    # than the bound silently pretending.
    r = sos_minimize({(4,): 1.0, (2,): -2.0, (0,): 3.0})
    assert r.success
    assert not r.certified


def test_square_box_idiom_is_certified():
    # min -x s.t. 1 - x**2 >= 0  =>  min = -1. The quadratic form is the
    # idiomatic way to write a box in an SOS model.
    r = sos_minimize({(1,): -1.0}, inequalities=[{(0,): 1.0, (2,): -1.0}])
    assert r.success and r.certified
    assert r.lower_bound <= -1.0 <= r.lower_bound + 1e-6


def test_tol_and_max_iter_are_accepted_and_never_cost_validity():
    # The escape hatch gh #218 asked for. A looser tol buys a weaker bound,
    # never an invalid one, because certification measures the actual miss.
    obj, g = _lasserre_ex5()
    gaps = []
    for tol in (1e-10, 1e-8, 1e-6):
        r = sos_minimize(obj, inequalities=g, order=4, tol=tol)
        assert r.success and r.certified, tol
        assert r.lower_bound <= TRUE_MIN, (tol, r.lower_bound)
        gaps.append(TRUE_MIN - r.lower_bound)
    assert gaps == sorted(gaps), f"looser tol should not tighten the bound: {gaps}"
    # max_iter is plumbed through and accepted.
    assert sos_minimize(obj, inequalities=g, order=2, max_iter=500).success


@pytest.mark.parametrize(
    "kwargs, msg",
    [({"tol": 0.0}, "tol"), ({"tol": -1e-8}, "tol"), ({"max_iter": 0}, "max_iter")],
)
def test_invalid_solver_controls_raise(kwargs, msg):
    obj, g = _lasserre_ex5()
    with pytest.raises(ValueError, match=msg):
        sos_minimize(obj, inequalities=g, order=2, **kwargs)
