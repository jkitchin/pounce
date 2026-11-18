"""gh#1181: ``QpResult.certificate`` on infeasible and unbounded verdicts.

Every certificate is checked here against the data the test passed in --
``P, c, A, b, G, h, lb, ub`` -- with numpy, sharing nothing with pounce, so a
certificate pounce wrongly marks ``verified`` fails the test.
"""

import numpy as np
import pytest

from pounce.qp import solve_qp, solve_socp

INF = np.inf


def galenet():
    """GALENET (netlib ``infeas``): node 5 receives at most 20 and must ship 30.

    Columns ``T14 T24 T25 T35 T46 T47 T57 T58``; rows: three supplies ``<= 20``,
    two transshipment balances ``= 0``, three demands ``>= 10, 20, 30``
    (written ``-row <= -rhs``).
    """
    A = np.zeros((2, 8))
    A[0, [0, 1, 4, 5]] = [1, 1, -1, -1]
    A[1, [2, 3, 6, 7]] = [1, 1, -1, -1]
    G = np.zeros((6, 8))
    G[0, 0] = 1
    G[1, [1, 2]] = 1
    G[2, 3] = 1
    G[3, 4] = -1
    G[4, [5, 6]] = -1
    G[5, 7] = -1
    h = np.array([20, 20, 20, -10, -20, -30], dtype=float)
    lb = np.zeros(8)
    ub = np.array([30, 20, 10, 10, 10, 2, 20, 30], dtype=float)
    return dict(c=np.zeros(8), A=A, b=np.zeros(2), G=G, h=h, lb=lb, ub=ub)


def check_farkas(m, cert, orthant=True):
    assert cert["kind"] == "primal_infeasible"
    assert cert["verified"], cert
    y, z, zl, zu = (np.asarray(cert[k]) for k in ("y", "z", "z_lb", "z_ub"))
    n = len(m["c"])
    lb = m.get("lb", np.full(n, -INF))
    ub = m.get("ub", np.full(n, INF))
    r = np.zeros(n)
    if "A" in m:
        r += m["A"].T @ y
    r += m["G"].T @ z - zl + zu
    assert np.max(np.abs(r)) <= 1e-9, r
    assert np.all(zl >= 0) and np.all(zu >= 0)
    assert np.all(zl[~np.isfinite(lb)] == 0) and np.all(zu[~np.isfinite(ub)] == 0)
    if orthant:
        assert np.all(z >= 0)
    value = m["h"] @ z
    if "b" in m:
        value += m["b"] @ y
    fin_l, fin_u = np.isfinite(lb), np.isfinite(ub)
    value += -(lb[fin_l] @ zl[fin_l]) + ub[fin_u] @ zu[fin_u]
    assert value < -1e-6, value
    assert abs(value - cert["value"]) <= 1e-9 * max(1.0, abs(value))


@pytest.mark.parametrize("presolve", [False, True])
def test_galenet_carries_a_checked_farkas_certificate(presolve):
    m = galenet()
    res = solve_qp(None, **m, qp_presolve=presolve)
    assert res.status == "primal_infeasible"
    check_farkas(m, res.certificate)
    # Presolve proves GALENET by itself; the ray then comes from re-solving
    # the unreduced problem.
    want = "unreduced_resolve" if presolve else "solve"
    assert res.certificate["source"] == want


def test_check_mode_never_resolves_and_no_mode_reports_nothing():
    m = galenet()
    # A presolve verdict has no ray of its own, and ``check`` will not
    # re-solve for one.
    assert solve_qp(None, **m, qp_presolve=True, qp_certificate="check").certificate is None
    res = solve_qp(None, **m, qp_certificate="check")
    check_farkas(m, res.certificate)
    assert solve_qp(None, **m, qp_certificate="no").certificate is None
    with pytest.raises(ValueError, match="qp_certificate"):
        solve_qp(None, **m, qp_certificate="maybe")


@pytest.mark.parametrize("method", ["ipm", "active-set"])
def test_unbounded_lp_carries_a_recession_direction(method):
    # min -x0 - 2 x1 + x2  s.t.  x0 - x1 = 0,  x0 + x1 - x2 >= 1,  x >= 0,
    # x2 <= 4: unbounded along d = (1, 1, 0).
    c = np.array([-1.0, -2.0, 1.0])
    A = np.array([[1.0, -1.0, 0.0]])
    G = np.array([[-1.0, -1.0, 1.0]])
    lb = np.zeros(3)
    ub = np.array([INF, INF, 4.0])
    res = solve_qp(None, c, A, [0.0], G, [-1.0], lb, ub, method=method)
    assert res.status == "dual_infeasible"
    cert = res.certificate
    assert cert["kind"] == "dual_infeasible" and cert["verified"], cert
    d = np.asarray(cert["d"])
    assert np.max(np.abs(d)) == pytest.approx(1.0)
    assert abs(A @ d)[0] <= 1e-9
    assert (G @ d)[0] <= 1e-9
    assert np.all(d >= -1e-9) and d[2] <= 1e-9
    assert c @ d < -1e-6


def test_conic_farkas_certificate_is_in_the_dual_cone():
    # x0 <= -1 beside (x0, x1) in the second-order cone, i.e. x0 >= |x1| >= 0.
    G = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([-1.0, 0.0, 0.0])
    res = solve_socp(None, np.zeros(2), G=G, h=h, cones=[("nonneg", 1), ("soc", 2)])
    assert res.status == "primal_infeasible"
    cert = res.certificate
    check_farkas(dict(c=np.zeros(2), G=G, h=h), cert, orthant=False)
    z = np.asarray(cert["z"])
    assert z[0] >= 0
    assert z[1] >= np.linalg.norm(z[2:]) - 1e-12  # SOC is self-dual


def test_feasible_sliver_has_no_certificate():
    # 1 - 1e-7 <= x0 + x1 <= 1 over x >= 0: feasible, barely.
    G = np.array([[1.0, 1.0], [-1.0, -1.0]])
    h = np.array([1.0, -(1.0 - 1e-7)])
    for presolve in (False, True):
        res = solve_qp(np.eye(2), [-1.0, 0.0], G=G, h=h, lb=[0, 0], qp_presolve=presolve)
        assert res.success, res.status
        assert res.certificate is None
