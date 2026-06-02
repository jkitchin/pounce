"""Tests for per-dimension (anisotropic / auto) scaling in find_minima."""

import os

os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

import pounce
from pounce._minima import _gauss_terms, _pole_terms, _resolve_lengths


def test_resolve_lengths_auto_scalar_vector():
    L = np.array([10.0, 0.1])
    # auto -> frac * range, per dimension
    np.testing.assert_allclose(
        _resolve_lengths("auto", L, True, frac=0.1, fallback=0.5), [1.0, 0.01])
    # no box -> fallback
    np.testing.assert_allclose(
        _resolve_lengths("auto", np.ones(2), False, frac=0.1, fallback=0.5),
        [0.5, 0.5])
    # scalar -> isotropic
    np.testing.assert_allclose(
        _resolve_lengths(0.3, L, True, frac=0.1, fallback=0.5), [0.3, 0.3])
    # vector -> as given
    np.testing.assert_allclose(
        _resolve_lengths([2.0, 4.0], L, True, frac=0.1, fallback=0.5), [2.0, 4.0])


@pytest.mark.parametrize("kernel", ["gauss", "pole"])
def test_anisotropic_kernel_derivatives(kernel):
    """Vector-width kernels have correct analytic grad/Hessian (vs FD)."""
    rng = np.random.default_rng(0)
    centers = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([-0.2, 0.1, 0.0, 0.5])]
    x = np.array([0.15, 0.25, 0.2, 0.35])      # near a center: kernel non-trivial
    widths = np.array([0.3, 1.0, 5.0, 100.0])
    if kernel == "gauss":
        fn = lambda z: _gauss_terms(z, centers, 2.0, widths)
    else:
        fn = lambda z: _pole_terms(z, centers, 1.5, 2.0, 1e-3, widths)

    v0, g0, H0 = fn(x)
    eps = 1e-6
    gfd = np.zeros(4)
    Hfd = np.zeros((4, 4))
    for i in range(4):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        gfd[i] = (fn(xp)[0] - fn(xm)[0]) / (2 * eps)
        Hfd[:, i] = (fn(xp)[1] - fn(xm)[1]) / (2 * eps)
    # Relative tolerance: pole Hessian entries can be large near a center.
    scale = max(1.0, np.max(np.abs(H0)))
    assert np.max(np.abs(g0 - gfd)) < 1e-6 * max(1.0, np.max(np.abs(g0)))
    assert np.max(np.abs(H0 - 0.5 * (Hfd + Hfd.T))) < 1e-5 * scale


def _stretched_camel(scale=1000.0):
    """Six-hump camel with the y-axis stretched by `scale` (ill-conditioned)."""
    def f(x, y):
        return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2

    def F(z):
        u, v = z
        return f(u, v / scale)

    def J(z):
        u, v = z
        y = v / scale
        return np.array([(8 - 8.4 * u**2 + 2 * u**4) * u + y,
                         (u + (-8 + 16 * y**2) * y) / scale])

    def H(z):
        u, v = z
        y = v / scale
        return np.array([[8 - 25.2 * u**2 + 10 * u**4, 1.0 / scale],
                         [1.0 / scale, (-8 + 48 * y**2) / scale**2]])

    bounds = [(-2.0, 2.0), (-1.5 * scale, 1.5 * scale)]
    return F, J, H, bounds


def test_auto_finds_all_on_ill_scaled_landscape():
    """With auto (per-dimension) widths, flooding handles a 1000x-stretched
    variable out of the box."""
    F, J, H, bounds = _stretched_camel(1000.0)
    r = pounce.find_minima(
        F, [0.5, 500.0], method="flooding", jac=J, hess=H, bounds=bounds,
        n_minima=6, max_solves=300, patience=80, dedup=1e-2, seed=0,
        options={"print_level": 0, "tol": 1e-9},
    )
    assert len(r) == 6
    assert r.status == "target_reached"


def test_scaled_dedup_distinguishes_small_scale_minima():
    """Two minima separated only in a small-range variable stay distinct
    under the default scaled metric, but a raw metric with a tolerance sized
    to the large variable would merge them."""
    # minima at (±0.1, 0); x-range 0.4, y-range 2000.
    def fun(z):
        x, y = z
        return (x * x - 0.01) ** 2 + (y / 1000.0) ** 2

    def jac(z):
        x, y = z
        return np.array([4 * x * (x * x - 0.01), 2 * y / 1e6])

    def hess(z):
        x, y = z
        return np.array([[4 * (3 * x * x - 0.01), 0.0], [0.0, 2 / 1e6]])

    bounds = [(-0.2, 0.2), (-1000.0, 1000.0)]
    kw = dict(method="multistart", jac=jac, hess=hess, bounds=bounds,
              n_minima=2, max_solves=80, patience=40, seed=0,
              options={"print_level": 0, "tol": 1e-10})

    # Default scaled metric: the two minima are far apart in scaled space.
    r = pounce.find_minima(fun, [0.05, 0.0], dedup=1e-2, **kw)
    assert len(r) == 2

    # Raw metric with a tolerance "reasonable" for the large variable merges
    # the two small-scale-separated minima into one.
    r_raw = pounce.find_minima(
        fun, [0.05, 0.0], dedup=0.5,
        distance=lambda a, b: float(np.linalg.norm(a - b)), **kw)
    assert len(r_raw) == 1
