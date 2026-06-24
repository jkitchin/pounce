"""Fully-implicit index-1 DAE solver ``pounce.ode.solve_dae`` (F(t,y,y')=0)."""

import numpy as np
import pytest

import pounce
from pounce.ode import solve_ivp, solve_dae

# Robertson kinetics as an index-1 DAE: two differential equations plus the
# mass-conservation algebraic constraint.
_k1, _k2, _k3 = 0.04, 3.0e7, 1.0e4


def _f1(y):
    return -_k1 * y[0] + _k3 * y[1] * y[2]


def _f2(y):
    return _k1 * y[0] - _k3 * y[1] * y[2] - _k2 * y[1] ** 2


def _F(t, y, yp):
    return np.array([yp[0] - _f1(y), yp[1] - _f2(y), y[0] + y[1] + y[2] - 1.0])


def _F_jac(t, y, yp):
    Fy = np.array([[_k1, -_k3 * y[2], -_k3 * y[1]],
                   [-_k1, _k3 * y[2] + 2 * _k2 * y[1], _k3 * y[1]],
                   [1.0, 1.0, 1.0]])
    Fyp = np.diag([1.0, 1.0, 0.0])
    return Fy, Fyp


def _robertson_mass():
    def f(t, y):
        return np.array([_f1(y), _f2(y), y[0] + y[1] + y[2] - 1.0])
    return f, np.diag([1.0, 1.0, 0.0])


def test_dae_matches_mass_matrix_form_and_conserves():
    """The fully-implicit Robertson DAE matches pounce's validated mass-matrix
    solve to round-off, and holds the algebraic constraint to machine eps."""
    f_mass, M = _robertson_mass()
    tf, kw = 1e4, dict(rtol=1e-7, atol=1e-9)
    m = solve_ivp(f_mass, (0, tf), [1.0, 0, 0], mass=M, dense_output=True, **kw)
    d = solve_dae(_F, (0, tf), [1.0, 0, 0], yp0=[-_k1, _k1, 0.0],
                  consistent="assume", dense_output=True, **kw)
    assert d.success
    assert np.max(np.abs(d.y - m.sol(d.t))) < 1e-8          # match mass form
    assert np.max(np.abs(d.y.sum(axis=0) - 1.0)) < 1e-12    # conservation


def test_consistent_ic_projects_inconsistent_inputs():
    """A wrong algebraic component (sum != 1) and yp0=None are projected onto
    the constraint manifold before integrating (auto-detected algebraic var)."""
    tf, kw = 1e4, dict(rtol=1e-7, atol=1e-9)
    good = solve_dae(_F, (0, tf), [1.0, 0, 0], yp0=[-_k1, _k1, 0.0],
                     consistent="assume", **kw)
    bad = solve_dae(_F, (0, tf), [1.0, 0.0, 0.5], yp0=None,
                    consistent="project", **kw)        # sum = 1.5, no yp guess
    assert bad.success
    assert np.allclose(bad.y[:, -1], good.y[:, -1], rtol=1e-6)
    assert abs(bad.y[:, 0].sum() - 1.0) < 1e-12         # IC put it on manifold


def test_analytic_jac_matches_fd_and_is_cheaper():
    tf, kw = 1e4, dict(rtol=1e-7, atol=1e-9, yp0=[-_k1, _k1, 0.0],
                       consistent="assume")
    fd = solve_dae(_F, (0, tf), [1.0, 0, 0], **kw)
    an = solve_dae(_F, (0, tf), [1.0, 0, 0], jac=_F_jac, **kw)
    assert an.success and fd.success
    assert np.allclose(an.y[:, -1], fd.y[:, -1], rtol=1e-6)
    assert an.nfev < fd.nfev                            # no 2n FD-Jacobian evals


def test_top_level_alias_and_t_eval():
    te = np.array([1e-2, 1.0, 1e2, 1e4])
    r = pounce.solve_dae(_F, (0, 1e4), [1.0, 0, 0], yp0=[-_k1, _k1, 0.0],
                         consistent="assume", t_eval=te, rtol=1e-7, atol=1e-9)
    assert r.success
    assert np.array_equal(r.t, te)
    assert r.y.shape == (3, te.size)
    assert np.max(np.abs(r.y.sum(axis=0) - 1.0)) < 1e-10


# --- differentiable fixed-mesh DAE (jax / torch daeint) ----------------------

def _param_dae_jax():
    import jax.numpy as jnp
    def F(t, y, yp, th):                 # y0' + th*y0 - y1 = 0 ; y0+y1-1 = 0
        return jnp.array([yp[0] + th * y[0] - y[1], y[0] + y[1] - 1.0])
    return F, jnp.linspace(0.0, 2.0, 81), jnp.array([0.5, 0.5])


def test_jax_daeint_gradient_matches_fd():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pounce.jax import daeint
    F, t, y0 = _param_dae_jax()

    def loss(th):
        return daeint(F, y0, t, th)[0, -1] ** 2

    g = float(jax.grad(loss)(1.3))
    fd = (float(loss(1.3 + 1e-6)) - float(loss(1.3 - 1e-6))) / 2e-6
    assert abs(g - fd) <= 1e-5 * abs(fd)
    # constraint holds along the differentiable trajectory
    Y = np.asarray(daeint(F, y0, t, 1.3))
    assert np.max(np.abs(Y.sum(axis=0) - 1.0)) < 1e-10


def test_torch_daeint_gradient_matches_fd():
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    from pounce.torch import daeint

    def F(t, y, yp, th):
        return torch.stack([yp[0] + th * y[0] - y[1], y[0] + y[1] - 1.0])

    t = torch.linspace(0.0, 2.0, 81, dtype=torch.float64)
    y0 = torch.tensor([0.5, 0.5], dtype=torch.float64)

    def loss(th):
        return daeint(F, y0, t, th)[0, -1] ** 2

    th = torch.tensor(1.3, requires_grad=True)
    loss(th).backward()
    with torch.no_grad():
        fd = (float(loss(torch.tensor(1.3 + 1e-6))) -
              float(loss(torch.tensor(1.3 - 1e-6)))) / 2e-6
    assert abs(float(th.grad) - fd) <= 1e-5 * abs(fd)
