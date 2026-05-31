"""SOCP solving from Python (pounce.qp.solve_socp)."""

import numpy as np

from pounce.qp import solve_socp


def test_min_norm_to_point():
    # min t s.t. (t, x0-2, x1+1) in SOC(3) -> t=0, x=(2,-1).
    r = solve_socp(c=[1.0, 0.0, 0.0], G=-np.eye(3), h=[0.0, -2.0, 1.0], cones=[("soc", 3)])
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [0.0, 2.0, -1.0], atol=1e-6)


def test_projection_onto_soc():
    # Euclidean projection of (1,2,0) onto the SOC: closed form (1.5,1.5,0).
    r = solve_socp(P=np.eye(3), c=[-1.0, -2.0, 0.0], G=-np.eye(3), h=[0, 0, 0], cones=[3])
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [1.5, 1.5, 0.0], atol=1e-5)


def test_mixed_orthant_and_soc():
    # max x0 + x1 s.t. x0 <= 1 (nonneg), |x1| <= 1 (soc) -> (1, 1).
    G = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, -1.0]])
    r = solve_socp(c=[-1.0, -1.0], G=G, h=[1.0, 1.0, 0.0], cones=[("nonneg", 1), ("soc", 2)])
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-5)


def test_int_shorthand_is_soc():
    r = solve_socp(c=[1.0, 0.0, 0.0], G=-np.eye(3), h=[0.0, -2.0, 1.0], cones=[3])
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [0.0, 2.0, -1.0], atol=1e-6)


def test_bad_cone_kind_raises():
    import pytest

    with pytest.raises(Exception):
        solve_socp(c=[1.0], G=-np.eye(1), h=[0.0], cones=[("banana", 1)])
