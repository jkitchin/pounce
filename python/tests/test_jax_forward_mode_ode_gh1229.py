"""gh#1229: forward-mode autodiff through pounce.jax.odeint / daeint / solve_bvp.

The fixed-mesh ODE, DAE and Newton-BVP roots were ``custom_vjp`` functions, so
``jax.jvp`` / ``jax.jacfwd`` raised "can't apply forward-mode autodiff (jvp)
to a custom_vjp function". For parameter fitting (few parameters, many
residuals) forward mode is the cheap direction. They are now implicit-function
``custom_jvp`` roots, which must serve forward mode *and* — by JAX
transposition — reverse mode, and both under ``vmap``.

The references are independent of the rule under test: an analytic derivative
where one exists (collocation discretisation error only) and central finite
differences of the primal otherwise. Matching ``jacfwd`` against ``jacrev``
alone would not do, since both now come from the same rule.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pounce.jax as pj

T = jnp.linspace(0.0, 1.0, 41)


def _ode_y(th):
    # The issue's reproduction: decoupled decay, y_i(t) = exp(-th_i t).
    return pj.odeint(lambda t, y, p: -p * y, jnp.ones(2), T, th).y


def _analytic_jac(th):
    # d y_i(t_j) / d th_k = -t_j exp(-th_i t_j) delta_ik, shape (2, m, 2).
    th = np.asarray(th)
    t = np.asarray(T)
    J = np.zeros((2, t.size, 2))
    for i in range(2):
        J[i, :, i] = -t * np.exp(-th[i] * t)
    return J


def _fd_jac(f, x, h=1e-6):
    x = np.asarray(x, float)
    cols = []
    for k in range(x.size):
        e = np.zeros_like(x)
        e.flat[k] = h
        cols.append((np.asarray(f(jnp.asarray(x + e)))
                     - np.asarray(f(jnp.asarray(x - e)))) / (2 * h))
    return np.stack(cols, axis=-1)


TH = jnp.array([1.0, 2.0])


def test_issue_repro_jacfwd_matches_jacrev():
    g = lambda th: _ode_y(th)[:, -1]
    jf = np.asarray(jax.jacfwd(g)(TH))
    jr = np.asarray(jax.jacrev(g)(TH))
    np.testing.assert_allclose(jf, jr, atol=1e-12)
    np.testing.assert_allclose(jf, _analytic_jac(TH)[:, -1, :], rtol=1e-6)


@pytest.mark.parametrize("mode", ["jacfwd", "jacrev"])
def test_ode_full_trajectory_jacobian_analytic(mode):
    J = np.asarray(getattr(jax, mode)(_ode_y)(TH))
    np.testing.assert_allclose(J, _analytic_jac(TH), atol=1e-7)


def test_ode_jvp_matches_fd_coupled_y0_and_theta():
    # Coupled nonlinear system, tangent on both theta and y0 (not separable,
    # so the y0 columns and the cross terms are exercised).
    def f(t, y, p):
        return jnp.array([-p[0] * y[0] + p[1] * y[1] ** 2, -p[1] * y[1] * y[0]])

    def g(q):
        return pj.odeint(f, q[2:], T, q[:2]).y

    q = jnp.array([0.8, 0.5, 1.0, 0.7])
    v = jnp.array([0.3, -1.1, 0.4, 0.9])
    _, tan = jax.jvp(g, (q,), (v,))
    np.testing.assert_allclose(np.asarray(tan), _fd_jac(g, q) @ np.asarray(v),
                               atol=1e-7)
    np.testing.assert_allclose(np.asarray(jax.jacfwd(g)(q)),
                               np.asarray(jax.jacrev(g)(q)), atol=1e-10)


def test_ode_jacfwd_wrt_mesh_matches_jacrev():
    # t is differentiable too; forward and reverse must agree on it.
    g = lambda tt: pj.odeint(lambda t, y, p: -p * y, jnp.ones(1), tt,
                             jnp.array([1.5])).y[0]
    jf = np.asarray(jax.jacfwd(g)(T))
    np.testing.assert_allclose(jf, np.asarray(jax.jacrev(g)(T)), atol=1e-10)
    np.testing.assert_allclose(jf, _fd_jac(g, T, h=1e-7), atol=1e-6)


@pytest.mark.parametrize("ths", [[[1.0, 2.0], [0.3, 0.7], [2.5, 0.1]],
                                 [[2.5, 0.1], [0.3, 0.7], [1.0, 2.0]]],
                         ids=["forward", "reversed"])
def test_vmap_of_jacfwd_distinct_primals(ths):
    # Batched JVP over distinct parameters: every lane needs its own
    # factorisation (the gh#1049 shape, now on the forward-mode solve).
    ths = jnp.array(ths)
    g = lambda th: _ode_y(th)[:, -1]
    batch = np.asarray(jax.vmap(jax.jacfwd(g))(ths))
    for i, th in enumerate(ths):
        np.testing.assert_allclose(batch[i], _analytic_jac(th)[:, -1, :],
                                   rtol=1e-6)


def test_jit_jacfwd_and_gauss_newton_shape():
    # The motivating use: a least-squares Jacobian, jitted.
    data = np.asarray(_ode_y(jnp.array([1.2, 1.8])))

    def resid(th):
        return (_ode_y(th) - data).ravel()

    J = np.asarray(jax.jit(jax.jacfwd(resid))(TH))
    assert J.shape == (2 * T.shape[0], 2)
    np.testing.assert_allclose(J, _analytic_jac(TH).reshape(-1, 2), atol=1e-7)


def test_daeint_jacfwd_matches_fd_and_jacrev():
    # Semi-explicit index-1 DAE: y0' = -k y0 + y1, 0 = y1 - 0.5 y0.
    def F(t, y, yp, th):
        return jnp.array([yp[0] + th[0] * y[0] - y[1], y[1] - th[1] * y[0]])

    t = jnp.linspace(0.0, 1.0, 31)
    y0 = jnp.array([1.0, 0.5])

    def g(th):
        # tight Newton tol: a looser forward solve puts its own noise into
        # the finite-difference reference at h=1e-6.
        return pj.daeint(F, y0, t, th, tol=1e-13)[:, 1:].ravel()

    th = jnp.array([1.3, 0.5])
    jf = np.asarray(jax.jacfwd(g)(th))
    np.testing.assert_allclose(jf, np.asarray(jax.jacrev(g)(th)), atol=1e-10)
    np.testing.assert_allclose(jf, _fd_jac(g, th), atol=1e-6)
    ths = jnp.array([[1.3, 0.5], [0.4, 0.9]])
    batch = np.asarray(jax.vmap(jax.jacfwd(g))(ths))
    for i, thi in enumerate(ths):
        np.testing.assert_allclose(batch[i], _fd_jac(g, thi), atol=1e-6)


def test_solve_bvp_newton_jacfwd_matches_fd_and_jacrev():
    X = jnp.linspace(0.0, 1.0, 21)

    def g(k):
        sol = pj.solve_bvp(
            lambda x, y, th: jnp.vstack([y[1], -th[0] * y[0]]),
            lambda ya, yb, th: jnp.array([ya[0], yb[0] - 1.0]),
            X, jnp.vstack([X, jnp.ones_like(X)]), theta=k, tol=1e-12,
        )
        return sol.y[0]

    k = jnp.array([2.0])
    jf = np.asarray(jax.jacfwd(g)(k))
    np.testing.assert_allclose(jf, np.asarray(jax.jacrev(g)(k)), atol=1e-10)
    np.testing.assert_allclose(jf, _fd_jac(g, k), atol=1e-6)


def test_ode_hessian_through_ift_rule():
    # Second order now composes: the tangent rule's linear solve is a
    # custom_linear_solve, whose JVP JAX supplies. y(1) = exp(-k), so
    # d2y/dk2 = exp(-k).
    g = lambda k: pj.odeint(lambda t, y, p: -p * y, jnp.ones(1), T, k).y[0, -1]
    H = np.asarray(jax.hessian(g)(jnp.array([1.0])))
    np.testing.assert_allclose(H, [[np.exp(-1.0)]], rtol=1e-5)
