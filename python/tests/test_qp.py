"""Tests for the convex LP/QP solver bindings (pounce-convex via PyO3).

Cover one-shot solve, multiple-RHS, the build-once/solve-many
QpFactorization handle, batched solving, and status reporting
(infeasible / unbounded).
"""

import numpy as np

from pounce import _pounce as p


def _box_qp(c, lo=0.0, hi=1.0):
    """min ½·2·‖x‖² + cᵀx  s.t.  lo ≤ x ≤ hi  (P = 2I)."""
    n = len(c)
    return p.QpProblem(
        n=n,
        c=list(c),
        p_rows=list(range(n)),
        p_cols=list(range(n)),
        p_vals=[2.0] * n,
        lb=[lo] * n,
        ub=[hi] * n,
    )


def test_solve_qp_box_clamps_to_bounds():
    # unconstrained optimum at (1.5, 2.0); clamped to (1, 1).
    r = p.solve_qp(_box_qp([-3.0, -4.0]))
    assert r["status"] == "optimal"
    x = np.asarray(r["x"])
    assert abs(x[0] - 1.0) < 1e-6
    assert abs(x[1] - 1.0) < 1e-6
    # Upper-bound multipliers are active and positive.
    assert np.asarray(r["z_ub"])[0] > 0.5


def test_solve_qp_equality():
    # min x0²+x1² s.t. x0+x1 = 2  → (1, 1), equality dual reported.
    prob = p.QpProblem(
        n=2,
        c=[0.0, 0.0],
        p_rows=[0, 1],
        p_cols=[0, 1],
        p_vals=[2.0, 2.0],
        a_rows=[0, 0],
        a_cols=[0, 1],
        a_vals=[1.0, 1.0],
        b=[2.0],
    )
    r = p.solve_qp(prob)
    assert r["status"] == "optimal"
    x = np.asarray(r["x"])
    assert abs(x[0] - 1.0) < 1e-6 and abs(x[1] - 1.0) < 1e-6
    assert np.asarray(r["y"]).shape == (1,)


def test_solve_qp_multi_rhs_matches_individual():
    base = _box_qp([0.0, 0.0])
    cs = [[-1.0, -4.0], [-4.0, 1.0], [3.0, -2.0], [0.0, 0.0]]
    res = p.solve_qp_multi_rhs(base, cs)
    assert len(res) == len(cs)
    for c, r in zip(cs, res):
        single = p.solve_qp(_box_qp(c))
        assert r["status"] == "optimal"
        np.testing.assert_allclose(
            np.asarray(r["x"]), np.asarray(single["x"]), atol=1e-6
        )


def test_qp_factorization_build_once_solve_many():
    base = _box_qp([0.0, 0.0])
    handle = p.QpFactorization(base)
    for c in ([-1.0, -4.0], [-4.0, 1.0], [3.0, -2.0]):
        reused = handle.solve(_box_qp(c))
        one_shot = p.solve_qp(_box_qp(c))
        assert reused["status"] == "optimal"
        assert one_shot["status"] == "optimal"
        # Both are independent interior-point solves. When the optimum sits on
        # an active bound (e.g. c=[3,-2] → vertex (0,1)), the IPM only
        # approaches the boundary asymptotically, so the two runs stop at
        # slightly different distances from it (here ~1e-5, since they take a
        # different iteration count). They agree on the same optimum to the
        # solver's near-boundary primal slack, not to full KKT tolerance.
        np.testing.assert_allclose(
            np.asarray(reused["x"]), np.asarray(one_shot["x"]), atol=1e-4
        )


def test_qp_factorization_rejects_pattern_mismatch():
    handle = p.QpFactorization(_box_qp([0.0, 0.0]))  # n = 2
    bad = handle.solve(_box_qp([0.0, 0.0, 0.0]))  # n = 3
    assert bad["status"] == "numerical_failure"
    # A matching solve still works afterward.
    ok = handle.solve(_box_qp([-1.0, -1.0]))
    assert ok["status"] == "optimal"


def test_solve_qp_batch_order_and_status():
    probs = [_box_qp([-float(k), -1.0]) for k in range(6)]
    res = p.solve_qp_batch(probs)
    assert len(res) == 6
    assert all(r["status"] == "optimal" for r in res)


def test_solve_qp_batch_warm_start():
    # Per-instance warm starts: same solutions as cold, no iter regression.
    base_probs = [_box_qp([-float(k), -1.0]) for k in range(4)]
    base = p.solve_qp_batch(base_probs)
    pert_probs = [_box_qp([-float(k) - 0.1, -1.05]) for k in range(4)]
    cold = p.solve_qp_batch(pert_probs)
    warm = p.solve_qp_batch(pert_probs, warm_starts=base)
    assert len(warm) == 4
    for c, w in zip(cold, warm):
        assert w["status"] == "optimal"
        np.testing.assert_allclose(
            np.asarray(w["x"]), np.asarray(c["x"]), atol=1e-6
        )
        assert int(w["iters"]) <= int(c["iters"])


def test_solve_qp_detects_unbounded():
    # min −x0 with x0 ≥ 0, no upper bound  → unbounded below.
    prob = p.QpProblem(
        n=1,
        c=[-1.0],
        g_rows=[0],
        g_cols=[0],
        g_vals=[-1.0],  # −x0 ≤ 0  (x0 ≥ 0)
        h=[0.0],
    )
    r = p.solve_qp(prob)
    assert r["status"] == "dual_infeasible"


def test_solve_qp_warm_start_matches_cold():
    # Warm starting from a nearby solution must reach the same optimum and
    # not increase iterations.
    base = p.QpProblem(
        n=3,
        c=[-1.0, -2.0, -0.5],
        p_rows=[0, 1, 2],
        p_cols=[0, 1, 2],
        p_vals=[2.0, 2.0, 2.0],
        g_rows=[0, 0, 0],
        g_cols=[0, 1, 2],
        g_vals=[1.0, 1.0, 1.0],
        h=[1.0],
    )
    base_sol = p.solve_qp(base)
    pert = p.QpProblem(
        n=3,
        c=[-1.1, -1.9, -0.55],
        p_rows=[0, 1, 2],
        p_cols=[0, 1, 2],
        p_vals=[2.0, 2.0, 2.0],
        g_rows=[0, 0, 0],
        g_cols=[0, 1, 2],
        g_vals=[1.0, 1.0, 1.0],
        h=[1.05],
    )
    cold = p.solve_qp(pert)
    warm = p.solve_qp(pert, warm_start=base_sol)
    assert warm["status"] == "optimal"
    np.testing.assert_allclose(
        np.asarray(warm["x"]), np.asarray(cold["x"]), atol=1e-6
    )
    assert int(warm["iters"]) <= int(cold["iters"])


def test_qp_problem_validation():
    import pytest

    # c length must equal n.
    with pytest.raises(ValueError):
        p.QpProblem(n=2, c=[1.0])
    # P strict-upper entry rejected (lower triangle only).
    with pytest.raises(ValueError):
        p.QpProblem(n=2, c=[0.0, 0.0], p_rows=[0], p_cols=[1], p_vals=[1.0])
