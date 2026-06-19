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

    # Default is adaptive (like SciPy): both refine to meet tol.
    res = pounce.solve_bvp(fun, bc, x, y0, tol=1e-4)
    ref = scipy_integrate.solve_bvp(fun, bc, x, y0, tol=1e-4)

    assert res.success
    assert res.rms_residuals.max() < 1e-4  # refined until under tol
    xt = np.linspace(0, 4, 25)
    assert np.max(np.abs(res.sol(xt)[0] - ref.sol(xt)[0])) < 5e-3

    # adaptive=False solves the given mesh as-is — collocation residual to
    # ~machine precision on that fixed mesh.
    fixed = pounce.solve_bvp(fun, bc, x, y0, adaptive=False)
    assert fixed.success and fixed.x.size == x.size
    assert fixed.rms_residuals.max() < 1e-9


def test_solve_bvp_adaptive_matches_scipy():
    """adaptive=True reproduces SciPy's mesh refinement (same nodes & soln)."""
    scipy_integrate = pytest.importorskip("scipy.integrate")

    def fun(x, y):
        return np.vstack((y[1], -np.exp(y[0])))  # Bratu

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])

    x = np.linspace(0, 1, 5)
    y0 = np.zeros((2, x.size))
    ra = pounce.solve_bvp(fun, bc, x, y0, tol=1e-6, adaptive=True, max_nodes=2000)
    rs = scipy_integrate.solve_bvp(fun, bc, x, y0, tol=1e-6)
    assert ra.success
    # Same refinement decisions -> same final mesh size, and identical soln.
    assert ra.x.size == rs.x.size
    xt = np.linspace(0, 1, 200)
    assert np.max(np.abs(ra.sol(xt)[0] - rs.sol(xt)[0])) < 1e-10
    assert ra.rms_residuals.max() < 1e-6


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

    res = pounce.solve_bvp(fun, bc, x, y0, p=[3.0], tol=1e-6)
    assert res.success
    assert res.p is not None
    assert abs(res.p[0] - np.pi) < 1e-3


def test_solve_bvp_newton_and_ipm_agree():
    """The fast Newton path and the IPM feasibility path give the same y."""
    def fun(x, y):
        return np.vstack((y[1], -np.exp(y[0])))

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])

    x = np.linspace(0, 1, 51)
    y0 = np.zeros((2, x.size))
    # Compare the two forward solvers on the same fixed mesh.
    r_newton = pounce.solve_bvp(fun, bc, x, y0, method="newton", tol=1e-8,
                                adaptive=False)
    r_ipm = pounce.solve_bvp(fun, bc, x, y0, method="ipm", tol=1e-8,
                             adaptive=False)
    assert r_newton.success and r_ipm.success
    assert np.max(np.abs(r_newton.y - r_ipm.y)) < 1e-7


def test_solve_bvp_analytic_jac_matches_fd():
    """Supplying fun_jac / bc_jac gives the same solution as the FD path."""
    def fun(x, y):
        return np.vstack((y[1], -np.exp(y[0])))

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])

    def fun_jac(x, y):
        # df/dy with shape (n, n, m): row 0 -> [0, 1]; row 1 -> [-e^{y0}, 0].
        n, mm = 2, x.size
        J = np.zeros((n, n, mm))
        J[0, 1, :] = 1.0
        J[1, 0, :] = -np.exp(y[0])
        return J

    def bc_jac(ya, yb):
        dya = np.array([[1.0, 0.0], [0.0, 0.0]])
        dyb = np.array([[0.0, 0.0], [1.0, 0.0]])
        return dya, dyb

    x = np.linspace(0, 1, 41)
    y0 = np.zeros((2, x.size))

    res_fd = pounce.solve_bvp(fun, bc, x, y0)
    res_an = pounce.solve_bvp(fun, bc, x, y0, fun_jac=fun_jac, bc_jac=bc_jac)
    assert res_fd.success and res_an.success
    assert np.max(np.abs(res_fd.y - res_an.y)) < 1e-8


def test_solve_bvp_bc_tol_status():
    """An unmet bc_tol downgrades an otherwise-converged solve to status 3."""
    def fun(x, y):
        return np.vstack((y[1], -y[0]))

    def bc(ya, yb):
        return np.array([ya[0], yb[0] - 1.0])

    x = np.linspace(0, np.pi / 2, 21)
    y0 = np.zeros((2, x.size)); y0[0] = x / (np.pi / 2)
    base = pounce.solve_bvp(fun, bc, x, y0)
    achieved = float(np.max(np.abs(bc(base.y[:, 0], base.y[:, -1]))))
    assert base.status == 0
    if achieved > 0:  # ask for tighter than achieved -> status 3
        res = pounce.solve_bvp(fun, bc, x, y0, bc_tol=achieved / 2)
        assert res.status == 3
        assert not res.success


def test_solve_bvp_singular_does_not_raise():
    """A degenerate (resonant) BVP returns a result, never raises.

    y'' + π² y = 0, y(0)=y(1)=0 has a one-parameter family of solutions, so
    the collocation Jacobian is singular. SciPy returns a result bunch with
    success=False (status 2); pounce must not propagate the FERAL singular
    error as an exception.
    """
    def fun(x, y):
        return np.vstack((y[1], -(np.pi**2) * y[0]))

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])

    x = np.linspace(0, 1, 21)
    y0 = np.zeros((2, x.size)); y0[0] = np.sin(np.pi * x)  # nonzero -> must iterate
    res = pounce.solve_bvp(fun, bc, x, y0)  # must not raise
    assert isinstance(res, pounce.BVPResult)
    assert res.status in (0, 2, 4)


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
# Constrained / optimal-control BVP (pounce-unique)
# --------------------------------------------------------------------------

def test_constrained_bvp_optimal_control_bound():
    """Bounded optimal control: minimise ∫(y-1)² s.t. y''=0, y(0)=0.

    Unconstrained optimum is y = 1.5x (y(1)=1.5). With the active bound
    y <= 1.2 the optimum caps at y(1)=1.2 and the bound is respected
    everywhere.
    """
    def fun(x, y):
        return np.vstack((y[1], np.zeros_like(y[0])))

    def bc(ya, yb):
        return np.array([ya[0]])  # only y(0)=0 -> one degree of freedom

    x = np.linspace(0, 1, 41)
    y0 = np.zeros((2, x.size))
    y0[0] = x

    def obj(Y, p):
        return np.trapezoid((Y[0] - 1.0) ** 2, x)

    r = pounce.solve_bvp_constrained(fun, bc, x, y0, objective=obj)
    assert r.success
    assert abs(r.y[0, -1] - 1.5) < 1e-2

    rc = pounce.solve_bvp_constrained(
        fun, bc, x, y0, objective=obj,
        y_bounds=([-1e19, -1e19], [1.2, 1e19]),
    )
    assert rc.success
    assert rc.y[0].max() <= 1.2 + 1e-6
    assert abs(rc.y[0, -1] - 1.2) < 1e-4


def test_constrained_bvp_path_constraint():
    """A path constraint active on an under-determined system is satisfied."""
    def fun(x, y):
        return np.vstack((y[1], np.zeros_like(y[0])))

    def bc(ya, yb):
        return np.array([ya[0]])

    x = np.linspace(0, 1, 41)
    y0 = np.zeros((2, x.size))
    y0[0] = x

    def obj(Y, p):
        return np.trapezoid((Y[0] - 1.0) ** 2, x)

    # Enforce y <= 0.8 everywhere via a path constraint.
    def path(x, y):
        return np.vstack([y[0]])

    r = pounce.solve_bvp_constrained(
        fun, bc, x, y0, objective=obj, path=path, path_bounds=([-1e19], [0.8]),
    )
    assert r.success
    assert r.y[0].max() <= 0.8 + 1e-6


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


def test_jax_solve_bvp_second_order():
    """second_order=True unlocks jax.grad(jax.grad(...)) via custom_jvp."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    import pounce.jax as pj

    # Bratu: y'' + lambda e^y = 0, y(0)=y(1)=0.
    def fun(x, y, lam):
        return jnp.vstack((y[1], -lam * jnp.exp(y[0])))

    def bc(ya, yb, lam):
        return jnp.array([ya[0], yb[0]])

    x = jnp.linspace(0, 1, 51)
    y0 = jnp.zeros((2, x.size))

    def y_mid(lam):
        sol = pj.solve_bvp(
            fun, bc, x, y0, theta=lam, method="ipm", second_order=True
        )
        return sol.y[0, sol.y.shape[1] // 2]

    lam = 1.0
    g1 = float(jax.grad(y_mid)(lam))
    g2 = float(jax.grad(jax.grad(y_mid))(lam))

    h = 1e-5
    fd1 = float((y_mid(lam + h) - y_mid(lam - h)) / (2 * h))
    fd2 = float((jax.grad(y_mid)(lam + h) - jax.grad(y_mid)(lam - h)) / (2 * h))
    assert abs(g1 - fd1) / abs(fd1) < 1e-6
    assert abs(g2 - fd2) / abs(fd2) < 1e-5


def test_jax_solve_bvp_newton_ipm_grad_agree():
    """The feral-Newton VJP and the IPM implicit-diff VJP agree."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    import pounce.jax as pj

    def fun(x, y, theta):
        return jnp.vstack((y[1], -theta * jnp.exp(y[0])))

    def bc(ya, yb, theta):
        return jnp.array([ya[0], yb[0]])

    x = jnp.linspace(0, 1, 41)
    y0 = jnp.zeros((2, x.size))

    def loss(theta, method):
        sol = pj.solve_bvp(fun, bc, x, y0, theta=theta, method=method)
        return jnp.sum(sol.y[0] ** 2)

    g_newton = float(jax.grad(lambda t: loss(t, "newton"))(1.0))
    g_ipm = float(jax.grad(lambda t: loss(t, "ipm"))(1.0))
    assert abs(g_newton - g_ipm) / abs(g_ipm) < 1e-5


def test_jax_solve_bvp_full_jacobian_newton():
    """jax.jacobian (which vmaps the VJP) works on the Newton path."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    import pounce.jax as pj

    def fun(x, y, theta):
        return jnp.vstack((y[1], -theta * jnp.exp(y[0])))

    def bc(ya, yb, theta):
        return jnp.array([ya[0], yb[0]])

    x = jnp.linspace(0, 1, 21)
    y0 = jnp.zeros((2, x.size))

    def y_of_lambda(lam):
        return pj.solve_bvp(fun, bc, x, y0, theta=lam).y[0]

    J = np.asarray(jax.jacobian(y_of_lambda)(1.0))
    base = np.asarray(y_of_lambda(1.0))
    fd = (np.asarray(y_of_lambda(1.0 + 1e-5)) - base) / 1e-5
    assert np.max(np.abs(J - fd)) < 1e-5


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
