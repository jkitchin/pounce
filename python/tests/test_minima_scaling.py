"""Tests for per-dimension (anisotropic / auto) scaling in find_minima."""

import os

os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

import pounce

# The unit tests of the width rule, the kernels' derivatives and the
# curvature-based amplitude moved to Rust with the search (gh#966):
# crates/pounce-minima/src/search.rs (`resolve_lengths_auto_scalar_vector`,
# `auto_amplitude_from_curvature`) and src/penalty_tnlp.rs
# (`kernel_gradients_match_finite_differences_of_the_value`,
# `kernel_hessians_match_finite_differences_of_the_gradient`). What stays here
# drives them end to end.


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


def test_auto_amplitude_handles_unknown_energy_scale():
    """Flooding finds all minima of a 1000x-scaled objective with no manual
    amplitude — the height is read from the Hessian."""
    s = 1000.0

    def fun(z):
        x, y = z
        return s * ((4 - 2.1*x**2 + x**4/3)*x**2 + x*y + (-4 + 4*y**2)*y**2)

    def jac(z):
        x, y = z
        return s * np.array([(8 - 8.4*x**2 + 2*x**4)*x + y, x + (-8 + 16*y**2)*y])

    def hess(z):
        x, y = z
        return s * np.array([[8 - 25.2*x**2 + 10*x**4, 1.0], [1.0, -8 + 48*y**2]])

    r = pounce.find_minima(
        fun, [0.5, 0.5], method="flooding", jac=jac, hess=hess,
        bounds=[(-2.0, 2.0), (-1.5, 1.5)], n_minima=6, max_solves=120,
        patience=40, dedup=1e-3, seed=0, options={"print_level": 0, "tol": 1e-9})
    assert len(r) == 6
    assert r.status == "target_reached"
