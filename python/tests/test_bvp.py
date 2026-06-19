"""Tests for the SciPy-compatible / differentiable BVP solver (pounce.bvp).

The NumPy path is validated against :func:`scipy.integrate.solve_bvp`; the
JAX and PyTorch differentiable paths are checked for gradient correctness
against finite differences (and skipped when the backend is absent).
"""

import numpy as np
import pytest

import pounce


# --------------------------------------------------------------------------
# NumPy SciPy-compatible path
# --------------------------------------------------------------------------

def test_solve_bvp_matches_scipy():
    """y'' = -|y|, y(0)=0, y(4)=-2 (SciPy docs example)."""
    scipy_integrate = pytest.importorskip("scipy.integrate")

    def fun(x, y):
        return np.vstack((y[1], -np.abs(y[0])))

    def bc(ya, yb):
        return np.array([ya[0], yb[0] + 2.0])

    x = np.linspace(0, 4, 41)
    y0 = np.zeros((2, x.size))
    y0[0] = 1.0

    res = pounce.solve_bvp(fun, bc, x, y0)
    ref = scipy_integrate.solve_bvp(fun, bc, x, y0)

    assert res.success
    # Collocation residual is satisfied essentially to machine precision.
    assert res.rms_residuals.max() < 1e-9
    xt = np.linspace(0, 4, 25)
    assert np.max(np.abs(res.sol(xt)[0] - ref.sol(xt)[0])) < 5e-3


def test_solve_bvp_unknown_parameter_eigenvalue():
    """y'' + k^2 y = 0, y(0)=y(1)=0, y'(0)=k recovers the first eigenvalue."""

    def fun(x, y, p):
        return np.vstack((y[1], -p[0] ** 2 * y[0]))

    def bc(ya, yb, p):
        return np.array([ya[0], yb[0], ya[1] - p[0]])

    x = np.linspace(0, 1, 31)
    y0 = np.zeros((2, x.size))
    y0[0] = np.sin(np.pi * x)
    y0[1] = np.pi * np.cos(np.pi * x)

    res = pounce.solve_bvp(fun, bc, x, y0, p=[3.0])
    assert res.success
    assert res.p is not None
    assert abs(res.p[0] - np.pi) < 1e-3


def test_solve_bvp_validates_inputs():
    def fun(x, y):
        return np.vstack((y[1], -y[0]))

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])

    x = np.linspace(0, 1, 11)
    y0 = np.zeros((2, x.size))

    # Non-increasing mesh.
    with pytest.raises(ValueError):
        pounce.solve_bvp(fun, bc, x[::-1], y0)

    # Wrong boundary-residual count.
    def bad_bc(ya, yb):
        return np.array([ya[0]])

    with pytest.raises(ValueError):
        pounce.solve_bvp(fun, bad_bc, x, y0)

    # Singular term not supported.
    with pytest.raises(NotImplementedError):
        pounce.solve_bvp(fun, bc, x, y0, S=np.eye(2))


# --------------------------------------------------------------------------
# JAX differentiable path
# --------------------------------------------------------------------------

def test_jax_solve_bvp_gradient_ode_param():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    import pounce.jax as pj

    def fun(x, y, theta):
        return jnp.vstack((y[1], theta * y[0]))

    def bc(ya, yb, theta):
        return jnp.array([ya[0] - 1.0, yb[0]])

    x = jnp.linspace(0, 1, 41)
    y0 = jnp.zeros((2, x.size)).at[0].set(1.0 - x)

    def loss(theta):
        return jnp.sum(pj.solve_bvp(fun, bc, x, y0, theta=theta).y[0] ** 2)

    th = 2.0
    g = float(jax.grad(loss)(th))
    fd = float((loss(th + 1e-4) - loss(th - 1e-4)) / 2e-4)
    assert abs(g - fd) / abs(fd) < 1e-6


def test_jax_solve_bvp_gradient_boundary_value():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    import pounce.jax as pj

    def fun(x, y, theta):
        return jnp.vstack((y[1], -y[0]))

    def bc(ya, yb, theta):
        return jnp.array([ya[0] - theta, yb[0]])

    x = jnp.linspace(0, 1, 41)
    y0 = jnp.zeros((2, x.size))

    def loss(theta):
        return jnp.sum(pj.solve_bvp(fun, bc, x, y0, theta=theta).y[0] ** 2)

    th = 0.7
    g = float(jax.grad(loss)(th))
    fd = float((loss(th + 1e-4) - loss(th - 1e-4)) / 2e-4)
    assert abs(g - fd) / abs(fd) < 1e-6


def test_jax_solve_bvp_unknown_parameter():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    import pounce.jax as pj

    def fun(x, y, p, theta):
        return jnp.vstack((y[1], -(p[0] ** 2) * y[0]))

    def bc(ya, yb, p, theta):
        return jnp.array([ya[0], yb[0], ya[1] - theta])

    x = jnp.linspace(0, 1, 31)
    y0 = jnp.zeros((2, x.size))
    y0 = y0.at[0].set(jnp.sin(jnp.pi * x)).at[1].set(jnp.pi * jnp.cos(jnp.pi * x))

    sol = pj.solve_bvp(fun, bc, x, y0, p=[3.0], theta=1.5)
    assert abs(float(sol.p[0]) - np.pi) < 1e-3


# --------------------------------------------------------------------------
# PyTorch differentiable path
# --------------------------------------------------------------------------

def test_torch_solve_bvp_gradient_ode_param():
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    import pounce.torch as pt

    def fun(x, y, theta):
        return torch.vstack((y[1], theta * y[0]))

    def bc(ya, yb, theta):
        return torch.stack([ya[0] - 1.0, yb[0]])

    x = torch.linspace(0, 1, 41, dtype=torch.float64)
    y0 = torch.zeros((2, x.shape[0]), dtype=torch.float64)
    y0[0] = 1.0 - x

    def loss_val(theta):
        return torch.sum(pt.solve_bvp(fun, bc, x, y0, theta=theta).y[0] ** 2)

    th = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    loss = loss_val(th)
    loss.backward()
    g = float(th.grad)

    with torch.no_grad():
        eps = 1e-4
        fp = float(loss_val(torch.tensor(2.0 + eps, dtype=torch.float64)))
        fm = float(loss_val(torch.tensor(2.0 - eps, dtype=torch.float64)))
    fd = (fp - fm) / (2 * eps)
    assert abs(g - fd) / abs(fd) < 1e-6
