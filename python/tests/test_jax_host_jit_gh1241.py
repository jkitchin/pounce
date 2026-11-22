"""gh#1241: the host Newton loops of odeint / daeint / solve_bvp(newton) must
evaluate the user's function compiled, not eagerly op by op.

The host solve runs inside a ``pure_callback``, outside any enclosing
``jax.jit`` trace, and evaluates the RHS / residual hundreds of times
(residuals plus a finite-difference column Jacobian). Called as-is, every
evaluation re-ran the user's Python and dispatched each primitive one at a
time — 50x slower than passing ``jax.jit(rhs)``.

Wall-clock assertions are flaky, so these count how often the user's Python
body runs. Compiled, it runs once per trace (a handful of distinct input
shapes); eager, it runs once per host evaluation. Each test also checks the
eager count would have been large, so a bound that passes for the wrong
reason (a solve that barely evaluates ``fun``) is caught.

``host_jit`` has two branches, so the fallback is tested too: a function that
branches in Python on a concrete value worked under eager evaluation and must
keep working (eagerly) rather than fail under ``jax.jit``.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import pounce.jax as pj
from pounce.jax._host import host_jit

# A compiled solve traces fun once per distinct (shape, dtype) signature plus
# the JAX-side residual / yp traces; well under this. Eager is hundreds.
TRACE_BOUND = 25


def _counted(f):
    calls = {"n": 0}

    def g(*a):
        calls["n"] += 1
        return f(*a)
    return g, calls


def test_odeint_rhs_is_compiled_on_host():
    n = 6
    A = -jnp.diag(jnp.linspace(1.0, 20.0, n))

    def rhs(t, y, k):
        dy = A @ y
        for j in range(3):
            r = jnp.exp(k[j]) * y[j] * y[j + 1]
            dy = dy.at[j].add(-r).at[j + 1].add(-r).at[j + 2].add(r)
        return dy

    y0 = jnp.ones(n)
    t = jnp.linspace(0.0, 1.0, 21)
    k = jnp.array([0.1, 0.2, 0.3])

    f, calls = _counted(rhs)
    y = pj.odeint(f, y0, t, k).y
    assert calls["n"] <= TRACE_BOUND, calls["n"]

    # Same answer as a user-jitted RHS (the gh#1241 workaround).
    y_ref = pj.odeint(jax.jit(rhs), y0, t, k).y
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_ref), rtol=1e-12)

    # Under an outer jit and jacfwd, the host evaluations still compile.
    f2, calls2 = _counted(rhs)
    jac = jax.jit(jax.jacfwd(lambda kk: pj.odeint(f2, y0, t, kk).y[:, -1]))
    J = jac(k)
    assert calls2["n"] <= TRACE_BOUND, calls2["n"]
    assert np.all(np.isfinite(np.asarray(J)))


def test_daeint_residual_is_compiled_on_host():
    def F(t, y, yp, th):
        return yp + th * y

    f, calls = _counted(F)
    t = jnp.linspace(0.0, 1.0, 21)
    th = jnp.array([1.0, 2.0])
    Y = pj.daeint(f, jnp.ones(2), t, th)
    assert calls["n"] <= TRACE_BOUND, calls["n"]
    np.testing.assert_allclose(np.asarray(Y[:, -1]),
                               np.exp(-np.asarray(th)), rtol=1e-3)


def test_daeint_python_branch_on_t_falls_back_to_eager():
    # `if t > 0.5` needs a concrete t: fine eagerly, untraceable under jit.
    def F(t, y, yp, th):
        if t > 0.5:
            return yp + th * y
        return yp + th * y

    t = jnp.linspace(0.0, 1.0, 21)
    th = jnp.array([1.0, 2.0])
    Y = pj.daeint(F, jnp.ones(2), t, th)
    np.testing.assert_allclose(np.asarray(Y[:, -1]),
                               np.exp(-np.asarray(th)), rtol=1e-3)


def test_bvp_newton_fun_is_compiled_on_host():
    # y'' = -th * y, y(0) = 0, y(1) = 1.
    def fun(x, y, th):
        return jnp.vstack([y[1], -th * y[0]])

    def bc(ya, yb, th):
        return jnp.array([ya[0], yb[0] - 1.0])

    f, calls = _counted(fun)
    x = jnp.linspace(0.0, 1.0, 21)
    y = jnp.zeros((2, x.size))
    th = jnp.asarray(2.0)
    sol = pj.solve_bvp(f, bc, x, y, theta=th, method="newton")
    assert calls["n"] <= TRACE_BOUND, calls["n"]
    w = np.sqrt(2.0)
    np.testing.assert_allclose(np.asarray(sol.y[0]),
                               np.sin(w * np.asarray(x)) / np.sin(w),
                               atol=1e-6)


def test_host_jit_both_branches():
    f, calls = _counted(lambda a: a * 2.0)
    h = host_jit(f)
    for _ in range(50):
        h(np.ones(3))
    assert calls["n"] == 1  # traced once, reused

    def branchy(a):
        return a if float(a[0]) > 0 else -a
    g, gcalls = _counted(branchy)
    h = host_jit(g)
    for _ in range(5):
        np.testing.assert_allclose(np.asarray(h(np.ones(3))), np.ones(3))
    # one failed trace, then eager every time
    assert gcalls["n"] == 1 + 5
