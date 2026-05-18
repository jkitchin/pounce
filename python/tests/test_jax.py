"""Tests for the JAX integration. Skipped when JAX isn't installed."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp


def test_from_jax_hs071():
    from pounce.jax import from_jax

    def f(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def g(x):
        return jnp.stack([jnp.prod(x), jnp.dot(x, x)])

    prob = from_jax(
        f, g,
        n=4, m=2,
        lb=np.array([1.0] * 4), ub=np.array([5.0] * 4),
        cl=np.array([25.0, 40.0]), cu=np.array([2e19, 40.0]),
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(info["obj_val"], 17.0140172, rtol=1e-5)


def test_implicit_diff_parametric_qp():
    """Differentiate x*(p) for  min ||x - p||²   →   x*(p) = p,   dx*/dp = I.

    A trivial parametric problem where the analytic Jacobian is known
    in closed form (the identity). This exercises the custom_vjp end
    to end without needing scipy.
    """
    from pounce.jax import solve

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    def loss(p):
        x_star = solve(
            p, f=f, g=None, x0=jnp.zeros_like(p),
            n=p.size, m=0,
            options={"tol": 1e-10, "print_level": 0},
        )
        return jnp.sum(x_star ** 2)

    p = jnp.array([1.0, -2.0, 3.0])
    grad = jax.grad(loss)(p)
    # dL/dp = 2 x*(p) = 2 p.
    np.testing.assert_allclose(grad, 2.0 * p, atol=1e-4)
