"""gh#1052: ``jax.vmap`` over ``JaxProblem.batched_solve`` and friends.

The stacked forwards (``batched_solve``, ``batched_solve_with_warm``,
``vmap_solve_parallel``) set no ``vmap_method`` and raised under an outer
``jax.vmap``. Enabling them makes each outer lane its own stacked solve
with its own held factor, so the stacked back-solve / JVP callbacks — which
pack a whole batch into one RHS — must group their rows by solver id rather
than read lane 0's (the gh#1049 defect class). References are per-lane
unbatched results and central finite differences, and the outer lanes sit
on different sides of the constraint's activity so their factors genuinely
differ: with identical factors a lane-0 read would stay green.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pounce.jax as pj

N, M = 2, 1
OPTS = {"print_level": 0, "tol": 1e-10}
X0 = jnp.ones(N)
W = jnp.array([1.0, -2.0])


def _f(x, p):
    return (x[0] - p[0]) ** 2 + p[1] * (x[1] - 1.0) ** 2 + x[0] * x[1]


def _g(x, p):
    return jnp.array([x[0] + x[1] - p[0] * p[1]])


def _jp(factor_reuse):
    return pj.JaxProblem(
        f=_f, g=_g, n=N, m=M, p_example=jnp.array([1.0, 1.0]),
        cl=np.array([-np.inf]), cu=np.array([0.5]), options=OPTS,
        factor_reuse=factor_reuse,
    )


# Outer batch of three inner (B=2) batches. Lane 0 is all-active, lane 1
# mixed, lane 2 all-inactive: no two outer lanes share an active set.
PS = jnp.array([
    [[2.0, 0.5], [2.5, 0.4]],
    [[-1.0, 4.0], [2.0, 0.5]],
    [[-1.0, 4.0], [0.3, 0.2]],
])


def _active(jp, P):
    x = np.asarray(jp.batched_solve(P, X0))
    return tuple(bool(v) for v in np.asarray(jax.vmap(_g)(x, P))[:, 0] > 0.5 - 1e-6)


def _surface(name, jp):
    if name == "batched":
        return lambda P: jp.batched_solve(P, X0)
    if name == "batched_warm":
        return lambda P: jp.batched_solve_with_warm(P, X0)[0]
    if name == "parallel":
        return lambda P: jp.vmap_solve_parallel(P, X0)
    if name == "parallel_serial":
        return lambda P: jp.vmap_solve_parallel(P, X0, workers=1)
    raise AssertionError(name)


SURFACES = ["batched", "batched_warm", "parallel", "parallel_serial"]


def test_outer_lanes_have_distinct_active_sets():
    jp = _jp(True)
    patterns = {_active(jp, P) for P in PS}
    assert len(patterns) == len(PS), patterns


@pytest.mark.parametrize("factor_reuse", [True, False], ids=["reuse", "dense"])
@pytest.mark.parametrize("surface", SURFACES)
def test_vmap_matches_separate_and_fd(surface, factor_reuse):
    jp = _jp(factor_reuse)
    sol = _surface(surface, jp)
    s = lambda P: jnp.sum(sol(P) @ W)

    sep_x = np.stack([sol(P) for P in PS])
    sep_g = np.stack([jax.grad(s)(P) for P in PS])
    sep_j = np.stack([jax.jacrev(sol)(P) for P in PS])

    np.testing.assert_allclose(jax.vmap(sol)(PS), sep_x, atol=1e-12, rtol=0)
    np.testing.assert_allclose(jax.vmap(jax.grad(s))(PS), sep_g, atol=1e-10, rtol=0)
    np.testing.assert_allclose(jax.vmap(jax.jacrev(sol))(PS), sep_j, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.grad(s)))(PS), sep_g, atol=1e-10, rtol=0,
    )

    h = 1e-6
    for P, gP in zip(PS, sep_g):
        fd = np.zeros(P.shape)
        for i in np.ndindex(P.shape):
            e = np.zeros(P.shape)
            e[i] = h
            fd[i] = (float(s(P + e)) - float(s(P - e))) / (2 * h)
        np.testing.assert_allclose(gP, fd, atol=1e-6)


def test_nested_outer_vmap_grad():
    jp = _jp(True)
    s = lambda P: jnp.sum(jp.batched_solve(P, X0) @ W)
    PP = jnp.stack([PS[:2], PS[1:]])  # (2, 2, B, 2)
    got = jax.vmap(jax.vmap(jax.grad(s)))(PP)
    ref = np.stack([[jax.grad(s)(P) for P in row] for row in PP])
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=0)


def test_outer_batch_beyond_registry_target_keeps_every_factor():
    """An outer batch of ``B_out`` stacked solves registers ``B_out``
    factors before any backward reads one. The soft eviction tier only
    trims consumed entries (gh#942), so a ``B_out`` above
    ``_solver_registry_target`` must still find every factor."""
    jp = _jp(True)
    jp._solver_registry_target = 2
    rng = np.random.default_rng(0)
    B_out = 12
    PSb = jnp.asarray(
        np.stack([PS[k % 3] for k in range(B_out)])
        + 0.05 * rng.standard_normal((B_out,) + PS.shape[1:])
    )
    s = lambda P: jnp.sum(jp.batched_solve(P, X0) @ W)
    got = jax.vmap(jax.grad(s))(PSb)
    ref = np.stack([jax.grad(s)(P) for P in PSb])
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=0)


def test_outer_batch_beyond_registry_capacity_fails_loudly():
    jp = _jp(True)
    jp._solver_registry_capacity = 2
    s = lambda P: jnp.sum(jp.batched_solve(P, X0) @ W)
    with pytest.raises(Exception, match="missing stacked Solver"):
        jax.vmap(jax.grad(s))(PS)


# -- the stacked callbacks, driven directly ---------------------------------


def _stacked_sids(jp):
    from pounce.jax._problem import _pure_callback_batched_solve

    sids = jnp.stack([_pure_callback_batched_solve(jp, 2, P, jnp.ones((2, N)))[4]
                      for P in PS[:2]])
    assert sids[0] != sids[1]
    return sids


def test_stacked_backsolve_groups_lanes_by_solver_id():
    """Two distinct stacked solver ids through the stacked back-solve, so
    the guard does not depend on one public surface (cf.
    ``test_kkt_backsolve_groups_lanes_by_solver_id``, gh#1049)."""
    from pounce.jax._problem import _kkt_backsolve_batched_pure_callback

    jp = _jp(True)
    sids = _stacked_sids(jp)
    cots = jnp.array([[[1.0, -2.0], [0.5, 3.0]], [[-0.7, 0.2], [2.0, 1.5]]])

    def one(sid, cot):
        return _kkt_backsolve_batched_pure_callback(jp, 2, sid, cot, N, M)

    ux_b, ug_b = jax.vmap(one)(sids, cots)
    for i in range(2):
        ux, ug = one(sids[i], cots[i])
        np.testing.assert_allclose(ux_b[i], ux, atol=1e-12, rtol=0)
        np.testing.assert_allclose(ug_b[i], ug, atol=1e-12, rtol=0)
    # The two factors really differ on these cotangents, so a lane-0 read
    # could not reproduce lane 1.
    assert not np.allclose(one(sids[0], cots[1])[0], ux_b[1])
    # jacrev-shaped batch (one id, several cotangents) and nested batching.
    ux_j, _ = jax.vmap(one, in_axes=(None, 0))(sids[1], cots)
    np.testing.assert_allclose(ux_j[0], one(sids[1], cots[0])[0], atol=1e-12, rtol=0)
    ux_n, _ = jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(sids, cots)
    np.testing.assert_allclose(ux_n[1, 0], one(sids[1], cots[0])[0], atol=1e-12, rtol=0)


def test_stacked_jvp_groups_lanes_by_anchor():
    """``batched_jvp_from_state``'s callback over two anchors' ids."""
    from pounce.jax._problem import _kkt_jvp_batched_pure_callback

    jp = _jp(True)
    with jp.anchor(PS[0], X0) as s0, jp.anchor(PS[1], X0) as s1:
        sids = jnp.array([s0.sid, s1.sid])
        rx = jnp.array([[[1.0, -2.0], [0.5, 3.0]], [[-0.7, 0.2], [2.0, 1.5]]])
        rg = jnp.array([[[0.3], [-1.0]], [[0.8], [0.1]]])

        def one(sid, rx_, rg_):
            return _kkt_jvp_batched_pure_callback(jp, 2, sid, rx_, rg_, N, M)

        wx_b, wg_b = jax.vmap(one)(sids, rx, rg)
        for i in range(2):
            wx, wg = one(sids[i], rx[i], rg[i])
            np.testing.assert_allclose(wx_b[i], wx, atol=1e-12, rtol=0)
            np.testing.assert_allclose(wg_b[i], wg, atol=1e-12, rtol=0)
        assert not np.allclose(one(sids[0], rx[1], rg[1])[0], wx_b[1])

        # And the public surface agrees with the callback per anchor.
        dp = jnp.array([[0.1, -0.2], [0.3, 0.05]])
        for st, P in ((s0, PS[0]), (s1, PS[1])):
            J = jax.jacrev(lambda P_: jp.batched_solve(P_, X0))(P)
            ref = jnp.stack([J[b, :, b, :] @ dp[b] for b in range(2)])
            np.testing.assert_allclose(
                jp.batched_jvp_from_state(st, dp), ref, atol=1e-8,
            )


def test_vmap_grad_with_equality_and_inequality_rows():
    """The y_c (equality) block's per-group de-interleave, under an outer vmap.

    Every fixture above has one inequality, so only the y_d block ever
    reached the grouped back-solve; an equality row runs through the y_c
    offsets, which a lane-0 or misaligned de-interleave would get wrong.
    """
    def f3(x, p):
        return ((x[0] - p[0]) ** 2 + p[1] * (x[1] - 1.0) ** 2
                + (x[2] - p[0] * p[1]) ** 2 + x[0] * x[1])

    def g3(x, p):
        return jnp.array([x[0] + x[1] + x[2] - p[1], x[0] - x[2]])

    jp = pj.JaxProblem(
        f=f3, g=g3, n=3, m=2, p_example=jnp.array([1.0, 1.0]),
        cl=np.array([-np.inf, 0.0]), cu=np.array([0.5, 0.0]), options=OPTS,
        factor_reuse=True,
    )
    x0, w3 = jnp.ones(3), jnp.array([1.0, -2.0, 0.7])

    def loss(P):
        return jnp.sum(jp.batched_solve(P, x0) @ w3)

    got = np.asarray(jax.vmap(jax.grad(loss))(PS))
    ref = np.stack([np.asarray(jax.grad(loss)(P)) for P in PS])
    np.testing.assert_allclose(got, ref, atol=1e-10, rtol=0)
    assert np.abs(ref[0] - ref[1]).max() > 1e-6, "lanes must differ"

    jac = jax.jacrev(lambda P: jp.batched_solve(P, x0))
    np.testing.assert_allclose(
        np.asarray(jax.vmap(jac)(PS)),
        np.stack([np.asarray(jac(P)) for P in PS]),
        atol=1e-10, rtol=0,
    )
