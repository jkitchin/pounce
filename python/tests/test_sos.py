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
