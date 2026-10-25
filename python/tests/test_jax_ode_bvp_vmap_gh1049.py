"""gh#1049: vmap of a pounce.jax ODE/BVP gradient over distinct parameters.

The IFT back-solve callback runs under ``vmap_method="broadcast_all"`` so
``jax.jacobian`` at one primal factors the collocation Jacobian once for
every cotangent. Its host body collapsed the batch to lane 0, which is only
right when every lane shares the primal: under an outer ``jax.vmap`` over
the parameter, every lane was back-solved against the first problem's
Jacobian. References are per-lane unbatched gradients (and the analytic
derivative), so a self-consistently wrong batched rule cannot pass.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pounce.jax as pj

T = jnp.linspace(0.0, 2.0, 21)
X = jnp.linspace(0.0, 1.0, 11)


def _ode_end(k):
    return pj.odeint(
        lambda t, y, th: -th[0] * y, jnp.array([1.0]), T, jnp.array([k])
    ).y[0, -1]


def _bvp_end(k):
    sol = pj.solve_bvp(
        lambda x, y, th: -th[0] * y,
        lambda ya, yb, th: jnp.array([ya[0] - 1.0]),
        X, jnp.ones((1, X.shape[0])), theta=jnp.array([k]),
    )
    return sol.y[0, -1]


def _separate(f, ks):
    return np.array([jax.grad(f)(k) for k in np.ravel(ks)]).reshape(np.shape(ks))


@pytest.mark.parametrize("f", [_ode_end, _bvp_end], ids=["ode", "bvp"])
@pytest.mark.parametrize("ks", [[0.2, 0.7, 1.3], [1.3, 0.7, 0.2]],
                         ids=["forward", "reversed"])
def test_vmap_grad_matches_separate_solves(f, ks):
    ks = jnp.array(ks)
    batch = np.array(jax.vmap(jax.grad(f))(ks))
    np.testing.assert_allclose(batch, _separate(f, ks), atol=1e-9, rtol=0)


def test_vmap_grad_ode_matches_analytic():
    ks = jnp.array([0.2, 0.7, 1.3])
    batch = np.array(jax.vmap(jax.grad(_ode_end))(ks))
    # Collocation discretization error only, not lane-0 contamination.
    np.testing.assert_allclose(batch, -2 * np.exp(-2 * np.asarray(ks)), rtol=1e-5)


def test_jit_vmap_grad_ode():
    ks = jnp.array([0.2, 0.7, 1.3])
    batch = np.array(jax.jit(jax.vmap(jax.grad(_ode_end)))(ks))
    np.testing.assert_allclose(batch, _separate(_ode_end, ks), atol=1e-9, rtol=0)


def test_nested_vmap_grad_ode():
    ks = jnp.array([[0.2, 0.7], [1.3, 0.4]])
    batch = np.array(jax.vmap(jax.vmap(jax.grad(_ode_end)))(ks))
    np.testing.assert_allclose(batch, _separate(_ode_end, ks), atol=1e-9, rtol=0)


def test_repeated_primal_lanes():
    ks = jnp.array([0.7, 0.7, 0.7])
    batch = np.array(jax.vmap(jax.grad(_ode_end))(ks))
    np.testing.assert_allclose(batch, _separate(_ode_end, ks), atol=1e-9, rtol=0)


def test_jacrev_at_fixed_primal_still_batches_cotangents():
    def traj(k):
        return pj.odeint(
            lambda t, y, th: -th[0] * y, jnp.array([1.0, 2.0]), T, jnp.array([k])
        ).y[:, -1]

    J = np.array(jax.jacrev(traj)(0.7))
    rows = np.array([jax.grad(lambda k, i=i: traj(k)[i])(0.7) for i in range(2)])
    np.testing.assert_allclose(J, rows, atol=1e-12, rtol=0)


def test_vmap_grad_ode_over_initial_state():
    # y0 rides in the same callback operand as theta; distinct y0 lanes
    # are distinct primals too.
    def end(y0, k):
        return pj.odeint(lambda t, y, th: -th[0] * y * y, jnp.array([y0]), T,
                         jnp.array([k])).y[0, -1]

    y0s = jnp.array([0.5, 1.0, 2.0])
    batch = np.array(jax.vmap(jax.grad(end), in_axes=(0, None))(y0s, 0.7))
    sep = np.array([jax.grad(end)(y0, 0.7) for y0 in y0s])
    np.testing.assert_allclose(batch, sep, atol=1e-9, rtol=0)


def test_vmap_grad_bvp_with_unknown_parameters():
    # k = 1 unknown parameter: the IFT block carries a non-state row.
    # y'' = -lam*y on [0, pi], y(0)=0, y'(0)=th, y(pi)=0 -> lam = 1;
    # the solution amplitude is th, so d y'(pi) / d th = -1 exactly.
    # (y is a fixed concrete guess: a traced guess is the separate
    # np.asarray(z0) limitation the issue reports.)
    x = jnp.linspace(0.0, np.pi, 41)
    y_guess = jnp.stack([jnp.sin(x), jnp.cos(x)])

    def end(th):
        sol = pj.solve_bvp(
            lambda x, y, p, th: jnp.stack([y[1], -p[0] * y[0]]),
            lambda ya, yb, p, th: jnp.array([ya[0], yb[0], ya[1] - th[0]]),
            x, y_guess, p=jnp.array([1.0]), theta=jnp.array([th]),
        )
        return sol.y[1, -1] + sol.p[0]

    ths = jnp.array([0.5, 1.0, 2.0])
    batch = np.array(jax.vmap(jax.grad(end))(ths))
    np.testing.assert_allclose(batch, _separate(end, ths), atol=1e-9, rtol=0)
    np.testing.assert_allclose(batch, -1.0, atol=1e-6)


def test_vmap_grad_bvp_second_order():
    def end(k):
        sol = pj.solve_bvp(
            lambda x, y, th: -th[0] * y,
            lambda ya, yb, th: jnp.array([ya[0] - 1.0]),
            X, jnp.ones((1, X.shape[0])), theta=jnp.array([k]),
            method="ipm", second_order=True,
        )
        return sol.y[0, -1]

    ks = jnp.array([0.2, 0.7, 1.3])
    batch = np.array(jax.vmap(jax.grad(end))(ks))
    np.testing.assert_allclose(batch, _separate(end, ks), atol=1e-9, rtol=0)


# -- the same defect class in JaxProblem's factor-reuse back-solve ---------

def test_kkt_backsolve_groups_lanes_by_solver_id():
    """``_kkt_backsolve_pure_callback`` must not read lane 0's solver id.

    ``JaxProblem.solve_with_warm(factor_reuse=True)`` routes its backward
    through this callback. Before its forward became vmappable that was the
    only thing keeping the lane-0 read from returning wrong gradients (a
    2.87 error once it was). Drive the callback directly with two distinct
    held factors, so the guard does not depend on any one public surface.
    """
    from pounce.jax._problem import (
        _kkt_backsolve_pure_callback,
        _pure_callback_warm_solve,
    )

    n, m = 2, 1
    jp = pj.JaxProblem(
        f=lambda x, p: (x[0] - p[0]) ** 2 + p[1] * (x[1] - 1.0) ** 2,
        g=lambda x, p: jnp.array([x[0] + x[1]]),
        n=n, m=m, p_example=jnp.array([1.0, 1.0]),
        cl=np.array([-np.inf]), cu=np.array([1.0]),
        options={"print_level": 0, "tol": 1e-10}, factor_reuse=True,
    )
    z = jnp.zeros
    sids = []
    for p in (jnp.array([2.0, 0.5]), jnp.array([-1.0, 4.0])):
        sids.append(_pure_callback_warm_solve(jp, p, jnp.ones(n), z(m), z(n), z(n))[4])
    sids = jnp.stack(sids)
    assert sids[0] != sids[1]
    vs = jnp.array([[1.0, -2.0], [0.5, 3.0]])

    def one(sid, v):
        return _kkt_backsolve_pure_callback(jp, sid, v, n, m)

    ux_b, ug_b = jax.vmap(one)(sids, vs)
    for i in range(2):
        ux, ug = one(sids[i], vs[i])
        np.testing.assert_allclose(ux_b[i], ux, atol=1e-12, rtol=0)
        np.testing.assert_allclose(ug_b[i], ug, atol=1e-12, rtol=0)
    # jacrev-shaped batch (one id, several cotangents) and nested batching.
    ux_j, _ = jax.vmap(one, in_axes=(None, 0))(sids[1], vs)
    np.testing.assert_allclose(ux_j[0], one(sids[1], vs[0])[0], atol=1e-12, rtol=0)
    ux_n, _ = jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(sids, vs)
    np.testing.assert_allclose(ux_n[1, 0], one(sids[1], vs[0])[0], atol=1e-12, rtol=0)


# -- solve_with_warm under vmap (now one host solve per lane) --------------

def _warm_model():
    n, m = 2, 1
    f = lambda x, p: (x[0] - p[0]) ** 2 + p[1] * (x[1] - 1.0) ** 2 + x[0] * x[1]
    g = lambda x, p: jnp.array([x[0] + x[1] - p[0] * p[1]])
    return f, g, n, m


@pytest.mark.parametrize("surface", ["pj", "jp-reuse", "jp-dense"])
def test_solve_with_warm_vmap_matches_separate(surface):
    f, g, n, m = _warm_model()
    cl, cu = np.array([-np.inf]), np.array([0.5])
    opts = {"print_level": 0, "tol": 1e-10}
    if surface == "pj":
        def sol(p):
            return pj.solve_with_warm(p, f=f, g=g, x0=jnp.ones(n), n=n, m=m,
                                      cl=cl, cu=cu, options=opts)[0]
    else:
        jp = pj.JaxProblem(f=f, g=g, n=n, m=m, p_example=jnp.array([1.0, 1.0]),
                           cl=cl, cu=cu, options=opts,
                           factor_reuse=(surface == "jp-reuse"))

        def sol(p):
            return jp.solve_with_warm(p, jnp.ones(n))[0]

    w = jnp.array([1.0, -2.0])
    s = lambda p: jnp.dot(sol(p), w)
    # Lanes on both sides of the constraint's activity, so the factors differ.
    ps = jnp.array([[2.0, 0.5], [-1.0, 4.0], [0.3, 0.2]])
    sep_x = np.stack([sol(p) for p in ps])
    sep_g = np.stack([jax.grad(s)(p) for p in ps])
    sep_j = np.stack([jax.jacrev(sol)(p) for p in ps])
    np.testing.assert_allclose(jax.vmap(sol)(ps), sep_x, atol=1e-12, rtol=0)
    np.testing.assert_allclose(jax.vmap(jax.grad(s))(ps), sep_g, atol=1e-12, rtol=0)
    np.testing.assert_allclose(jax.vmap(jax.jacrev(sol))(ps), sep_j, atol=1e-12, rtol=0)
    h = 1e-6
    fd = np.array([[(float(s(p + h * e)) - float(s(p - h * e))) / (2 * h)
                    for e in np.eye(2)] for p in ps])
    np.testing.assert_allclose(sep_g, fd, atol=1e-6)


# -- traced meshes / initial guesses (jit), mesh gradients ------------------

def _fd(F, a, h=1e-6):
    a = np.asarray(a, dtype=np.float64)
    out = np.zeros_like(a)
    for i in np.ndindex(a.shape):
        e = np.zeros_like(a)
        e[i] = h
        out[i] = (float(F(jnp.asarray(a + e))) - float(F(jnp.asarray(a - e)))) / (2 * h)
    return out


_X_NONUNIFORM = jnp.array(np.sort(np.r_[0.0, np.random.default_rng(0).uniform(0, 1, 13), 1.0]))


def _bvp_mesh_objective(x, method="newton"):
    sol = pj.solve_bvp(
        lambda x, y, th: jnp.stack([y[1], -th[0] * jnp.exp(y[0]) + th[1] * x]),
        lambda ya, yb, th: jnp.array([ya[0], yb[0] - 0.2]),
        x, jnp.zeros((2, x.shape[0])), theta=jnp.array([1.5, 0.3]), method=method,
    )
    return jnp.sum(jnp.sin(3 * sol.y[0]) * x) + sol.y[1, 3]


@pytest.mark.parametrize("method", ["newton", "ipm"])
def test_bvp_under_jit_with_mesh_and_guess_built_inside(method):
    def end(k):
        x = jnp.linspace(0.0, 1.0, 11)
        sol = pj.solve_bvp(lambda x, y, th: -th[0] * y,
                           lambda ya, yb, th: jnp.array([ya[0] - 1.0]),
                           x, jnp.ones((1, 11)), theta=jnp.array([k]), method=method)
        return sol.y[0, -1]

    np.testing.assert_allclose(jax.jit(end)(0.7), end(0.7), atol=1e-12, rtol=0)
    np.testing.assert_allclose(jax.jit(jax.grad(end))(0.7), jax.grad(end)(0.7),
                               atol=1e-12, rtol=0)
    ks = jnp.array([0.2, 0.7])
    np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(end)))(ks),
                               _separate(end, ks), atol=1e-9, rtol=0)


@pytest.mark.parametrize("method", ["newton", "ipm"])
def test_bvp_mesh_gradient_matches_finite_differences(method):
    F = lambda x: _bvp_mesh_objective(x, method)
    ref = _fd(F, _X_NONUNIFORM)
    np.testing.assert_allclose(jax.grad(F)(_X_NONUNIFORM), ref, rtol=0,
                               atol=1e-6 * np.max(np.abs(ref)))
    np.testing.assert_allclose(jax.jit(jax.grad(F))(_X_NONUNIFORM), ref, rtol=0,
                               atol=1e-6 * np.max(np.abs(ref)))


def test_bvp_vmap_grad_over_distinct_meshes():
    # Lanes differ only in the mesh (theta and the guess are shared), so the
    # back-solve grouping has to key on x as well.
    X = jnp.stack([_X_NONUNIFORM, _X_NONUNIFORM ** 1.3, jnp.sqrt(_X_NONUNIFORM)])
    batch = np.array(jax.vmap(jax.grad(_bvp_mesh_objective))(X))
    sep = np.stack([jax.grad(_bvp_mesh_objective)(x) for x in X])
    np.testing.assert_allclose(batch, sep, atol=1e-12, rtol=0)


def test_bvp_vmap_over_initial_guess():
    def end(scale):
        sol = pj.solve_bvp(lambda x, y, th: -th[0] * y,
                           lambda ya, yb, th: jnp.array([ya[0] - 1.0]),
                           X, scale * jnp.ones((1, X.shape[0])), theta=jnp.array([0.7]))
        return sol.y[0, -1]

    out = np.array(jax.vmap(end)(jnp.array([0.5, 2.0])))
    # Different starts converge to the same root within the Newton tol (1e-8).
    np.testing.assert_allclose(out, end(1.0), atol=1e-7, rtol=0)
    # The root does not depend on where Newton starts.
    assert float(jax.grad(end)(0.5)) == 0.0


def test_odeint_under_jit_with_mesh_built_inside():
    def end(k):
        t = jnp.linspace(0.0, 2.0, 21)
        return pj.odeint(lambda t, y, th: -th[0] * y, jnp.array([1.0]), t,
                         jnp.array([k])).y[0, -1]

    np.testing.assert_allclose(jax.jit(end)(0.7), _ode_end(0.7), atol=1e-12, rtol=0)
    np.testing.assert_allclose(jax.jit(jax.grad(end))(0.7), jax.grad(_ode_end)(0.7),
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(jax.jit(lambda k, t: pj.odeint(
        lambda t, y, th: -th[0] * y, jnp.array([1.0]), t, jnp.array([k])).y[0, -1])(0.7, T),
        _ode_end(0.7), atol=1e-12, rtol=0)


def test_odeint_mesh_gradient_and_vmap_over_meshes():
    t0 = jnp.array(np.sort(np.r_[0.0, np.random.default_rng(1).uniform(0, 3, 30), 3.0]))

    def F(t):
        y = pj.odeint(lambda t, y, p: jnp.array([p[0] * y[0] * (1 - y[0]) - 0.1 * y[1],
                                                 y[0] - y[1]]),
                      jnp.array([0.1, 0.0]), t, jnp.array([2.0])).y
        return jnp.sum(y * jnp.cos(t))

    ref = _fd(F, t0)
    np.testing.assert_allclose(jax.grad(F)(t0), ref, rtol=0, atol=1e-6 * np.max(np.abs(ref)))
    Ts = jnp.stack([t0, 0.9 * t0, t0 ** 1.1 / 3 ** 0.1])
    np.testing.assert_allclose(jax.vmap(jax.grad(F))(Ts),
                               np.stack([jax.grad(F)(t) for t in Ts]), atol=1e-12, rtol=0)


@pytest.mark.parametrize("second_order", [False, True])
def test_bvp_ipm_pytree_theta(second_order):
    xm = jnp.linspace(0.0, 1.0, 15)

    def F(k):
        sol = pj.solve_bvp(
            lambda x, y, th: jnp.stack([y[1], -th["a"] * jnp.exp(y[0]) + th["b"][0]]),
            lambda ya, yb, th: jnp.array([ya[0], yb[0] - 0.2]),
            xm, jnp.zeros((2, 15)), theta={"a": k[0], "b": k[1:]},
            method="ipm", second_order=second_order,
        )
        return jnp.sum(sol.y[0])

    k0 = jnp.array([1.5, 0.3])
    ref = _fd(F, k0)
    np.testing.assert_allclose(jax.grad(F)(k0), ref, rtol=0, atol=1e-6 * np.max(np.abs(ref)))
    if second_order:
        H = np.asarray(jax.hessian(F)(k0))
        Hfd = np.stack([_fd(lambda k, i=i: jax.grad(F)(k)[i], k0, 1e-4) for i in range(2)])
        np.testing.assert_allclose(H, Hfd, rtol=0, atol=1e-6 * np.max(np.abs(Hfd)))


@pytest.mark.parametrize("fn", ["solve", "solve_with_warm"])
def test_pytree_p_is_rejected_with_a_clear_error(fn):
    with pytest.raises(TypeError, match="single array or scalar"):
        getattr(pj, fn)((1.0, jnp.ones(2)), f=lambda x, p: jnp.sum(x ** 2),
                        x0=jnp.zeros(2), n=2)


# -- daeint: same traced-mesh failure -----------------------------------------

def _dae_objective(k, t):
    # index-1: y0' = -k*y0 + y1, 0 = y1 - y0**2 / 2
    F = lambda t, y, yp, th: jnp.array([yp[0] + th[0] * y[0] - y[1],
                                        y[1] - 0.5 * y[0] ** 2])
    Y = pj.daeint(F, jnp.array([1.0, 0.5]), t, jnp.array([k]))
    return jnp.sum(Y * jnp.cos(t))


_T_DAE = jnp.array(np.sort(np.r_[0.0, np.random.default_rng(2).uniform(0, 2, 25), 2.0]))


def test_daeint_under_jit_with_mesh_built_inside():
    t = jnp.linspace(0.0, 2.0, 21)
    np.testing.assert_allclose(
        jax.jit(lambda k: _dae_objective(k, jnp.linspace(0.0, 2.0, 21)))(0.7),
        _dae_objective(0.7, t), atol=1e-12, rtol=0)
    ks = jnp.array([0.3, 0.7, 1.5])
    batch = jax.jit(jax.vmap(jax.grad(_dae_objective), (0, None)))(ks, _T_DAE)
    sep = np.array([jax.grad(_dae_objective)(k, _T_DAE) for k in ks])
    np.testing.assert_allclose(batch, sep, atol=1e-12, rtol=0)


def test_daeint_mesh_gradient_matches_finite_differences():
    F = lambda t: _dae_objective(0.7, t)
    ref = _fd(F, _T_DAE)
    np.testing.assert_allclose(jax.grad(F)(_T_DAE), ref, rtol=0,
                               atol=1e-6 * np.max(np.abs(ref)))
