"""Differentiable SOCP layer (pounce.jax.solve_socp).

Validates the cone-aware OptNet backward (arrow operators in the
complementarity row) against finite differences, for second-order and
mixed orthant+SOC cones.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from pounce.jax import solve_socp  # noqa: E402


def _fd(fn, x, eps=1e-6):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        g[i] = (float(fn(jnp.array(xp))) - float(fn(jnp.array(xm)))) / (2 * eps)
    return g


P3 = jnp.eye(3)
G3 = -jnp.eye(3)  # s = -G x = x ∈ SOC
H3 = jnp.zeros(3)


def test_grad_c_soc_projection():
    # min ½‖x‖² − cᵀx s.t. x ∈ SOC(3): projection-like, smooth in c.
    def loss(c):
        return jnp.sum(solve_socp(P=P3, c=c, G=G3, h=H3, cones=[("soc", 3)]) ** 2)

    c0 = jnp.array([-1.0, -2.0, 0.3])
    np.testing.assert_allclose(np.asarray(jax.grad(loss)(c0)), _fd(loss, c0), atol=1e-4)


def test_grad_h_soc():
    c0 = jnp.array([-1.0, -2.0, 0.3])

    def loss(h):
        return jnp.sum(solve_socp(P=P3, c=c0, G=G3, h=h, cones=[3]) ** 2)

    h0 = jnp.array([0.5, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(jax.grad(loss)(h0)), _fd(loss, h0), atol=1e-4)


def test_grad_c_and_b_soc_with_equality():
    A = jnp.array([[1.0, 0.0, 0.0]])

    def loss_c(c):
        return jnp.sum(
            solve_socp(P=P3, c=c, G=G3, h=H3, A=A, b=jnp.array([0.5]), cones=[3]) ** 2
        )

    def loss_b(b):
        c0 = jnp.array([0.0, -1.0, 0.0])
        return jnp.sum(solve_socp(P=P3, c=c0, G=G3, h=H3, A=A, b=b, cones=[3]) ** 2)

    c0 = jnp.array([0.0, -1.0, 0.0])
    b0 = jnp.array([0.5])
    np.testing.assert_allclose(np.asarray(jax.grad(loss_c)(c0)), _fd(loss_c, c0), atol=1e-4)
    np.testing.assert_allclose(np.asarray(jax.grad(loss_b)(b0)), _fd(loss_b, b0), atol=1e-4)


def test_grad_mixed_orthant_and_soc():
    # Composite cone: an orthant block and a second-order block. The
    # backward must use diag on the orthant rows and the arrow operator on
    # the SOC rows.
    G = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, -1.0]])
    h = jnp.array([1.0, 1.0, 0.0])

    def loss(c):
        return jnp.sum(
            solve_socp(P=jnp.eye(2), c=c, G=G, h=h, cones=[("nonneg", 1), ("soc", 2)]) ** 2
        )

    c0 = jnp.array([-0.5, -0.5])
    np.testing.assert_allclose(np.asarray(jax.grad(loss)(c0)), _fd(loss, c0), atol=1e-4)


def test_jacrev_of_soc_solution():
    # x*(c) for the projection is differentiable; jacrev is well-formed.
    c0 = jnp.array([-1.0, -2.0, 0.3])
    J = jax.jacrev(lambda c: solve_socp(P=P3, c=c, G=G3, h=H3, cones=[3]))(c0)
    assert J.shape == (3, 3)
