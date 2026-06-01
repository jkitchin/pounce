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


def test_exp_cone_geometric_program():
    # Geometric program  min x + 1/x = min_u e^u + e^{-u}  (optimum 2),
    # via two exponential cones: (u,1,t1)∈Kexp, (-u,1,t2)∈Kexp.
    G = np.zeros((6, 3))
    G[0, 0] = -1.0  # s0 = u
    G[2, 1] = -1.0  # s2 = t1
    G[3, 0] = 1.0  # s3 = -u
    G[5, 2] = -1.0  # s5 = t2
    r = solve_socp(
        c=[0.0, 1.0, 1.0],
        G=G,
        h=[0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        cones=[("exp", 3), ("exp", 3)],
    )
    assert r.status == "optimal"
    assert abs(r.obj - 2.0) < 1e-5
    assert abs(r.x[0]) < 1e-4  # u ~ 0


def test_exp_cone_log_sum_exp_mixed():
    # min t s.t. t >= log(e^0 + e^0) = log 2, via two exp cones plus an
    # orthant row (u1 + u2 <= 1) -- exercises a mixed exp + nonneg product.
    G = np.zeros((7, 3))
    G[0, 0] = 1.0  # s0 = -t
    G[2, 1] = -1.0  # s2 = u1
    G[3, 0] = 1.0  # s3 = -t
    G[5, 2] = -1.0  # s5 = u2
    G[6, 1] = 1.0
    G[6, 2] = 1.0  # s6 = 1 - u1 - u2
    r = solve_socp(
        c=[1.0, 0.0, 0.0],
        G=G,
        h=[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        cones=[("exp", 3), ("exp", 3), ("nonneg", 1)],
    )
    assert r.status == "optimal"
    assert abs(r.obj - np.log(2.0)) < 1e-5


def test_exp_cone_dim_must_be_three():
    import pytest

    with pytest.raises(Exception):
        solve_socp(c=[1.0, 0.0], G=-np.eye(2), h=[0.0, 0.0], cones=[("exp", 2)])


def test_soc_mixed_with_exp():
    # A SOC and an exp cone in one problem:
    #   min t + z  s.t.  (t, 3, 4) in SOC(3)  ->  t >= 5,
    #                    (1, 1, z) in K_exp   ->  z >= e.
    # Optimum t = 5, z = e.
    G = np.zeros((6, 2))
    G[0, 0] = -1.0  # SOC s0 = t
    G[5, 1] = -1.0  # exp s5 = z
    r = solve_socp(
        c=[1.0, 1.0],
        G=G,
        h=[0.0, 3.0, 4.0, 1.0, 1.0, 0.0],
        cones=[("soc", 3), ("exp", 3)],
    )
    assert r.status == "optimal"
    assert abs(r.x[0] - 5.0) < 1e-5
    assert abs(r.x[1] - np.e) < 1e-5


def test_power_cone_known_optimum():
    # max x s.t. (x, 2, 0.5) in K_alpha  ->  x = 2^alpha * 0.5^(1-alpha).
    import numpy as np

    G = -np.eye(3)
    for alpha in (0.5, 0.3, 0.75):
        r = solve_socp(
            c=[-1.0, 0.0, 0.0],
            A=[[0, 1, 0], [0, 0, 1]],
            b=[2.0, 0.5],
            G=G,
            h=[0.0, 0.0, 0.0],
            cones=[("pow", alpha)],
        )
        assert r.status == "optimal"
        want = 2.0**alpha * 0.5 ** (1.0 - alpha)
        assert abs(r.x[0] - want) < 1e-5


def test_power_cone_bad_alpha_raises():
    import numpy as np
    import pytest

    with pytest.raises(Exception):
        solve_socp(c=[-1.0, 0.0, 0.0], G=-np.eye(3), h=[0, 0, 0], cones=[("pow", 1.5)])


def test_psd_min_eigenvalue_diagonal():
    # max λ s.t. M − λI ⪰ 0  ⇒  λ = λ_min(M). M = diag(2, 5) → 2.
    # x = (λ); G's column is svec(I) = [1, 0, 1], h = svec(M) = [2, 0, 5].
    r = solve_socp(c=[-1.0], G=[[1.0], [0.0], [1.0]], h=[2.0, 0.0, 5.0],
                   cones=[("psd", 2)])
    assert r.status == "optimal"
    assert abs(r.x[0] - 2.0) < 1e-5
    assert abs(r.obj + 2.0) < 1e-5


def test_psd_min_eigenvalue_offdiagonal():
    # M = [[2,1],[1,2]] → λ_min = 1; svec(M) = [2, √2, 2] exercises the
    # off-diagonal of the dense W⊗ₛW scaling block.
    r = solve_socp(c=[-1.0], G=[[1.0], [0.0], [1.0]],
                   h=[2.0, 2.0 ** 0.5, 2.0], cones=[("psd", 2)])
    assert r.status == "optimal"
    assert abs(r.x[0] - 1.0) < 1e-5
    assert abs(r.obj + 1.0) < 1e-5


def test_psd_cannot_mix_with_exp():
    import numpy as np
    import pytest

    with pytest.raises(ValueError):
        solve_socp(c=[1.0, 0.0, 0.0, 0.0], G=-np.eye(4), h=[0.0] * 4,
                   cones=[("psd", 2), ("exp", 3)])
