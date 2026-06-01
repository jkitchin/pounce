"""Spatial branch-and-bound global optimization (pounce.minimize_global)."""

import numpy as np
import pytest

import pounce
from pounce.global_opt import GlobalResult, eq, ge, minimize_global, var


def test_top_level_export():
    assert pounce.minimize_global is minimize_global
    assert pounce.GlobalResult is GlobalResult


def test_unconstrained_quartic_two_minima():
    # x⁴ − 3x² on [−2, 2]: global minimum −2.25 at x = ±√(3/2).
    f = var(0) ** 4 - 3.0 * var(0) ** 2
    r = minimize_global(f, lo=[-2.0], hi=[2.0])
    assert r.success
    assert abs(r.objective + 2.25) < 1e-3
    assert abs(abs(float(r.x[0])) - 1.2247449) < 1e-2
    assert r.lower_bound <= r.objective + 1e-6


def test_six_hump_camel():
    # Six local minima; global value ≈ −1.0316.
    x, y = var(0), var(1)
    f = (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2
    r = minimize_global(f, lo=[-2, -1.5], hi=[2, 1.5], abs_gap=1e-4, rel_gap=1e-4, max_nodes=200_000)
    assert r.success
    assert abs(r.objective + 1.031628) < 1e-2
    assert abs(float(r.x[0])) < 0.2 and abs(float(r.x[1])) > 0.5


def test_nonconvex_inequality():
    # min x + y  s.t.  x·y ≥ 4 on [1, 5]²  → 4 at (2, 2).
    r = minimize_global(
        var(0) + var(1),
        constraints=[ge(var(0) * var(1), 4.0)],
        lo=[1, 1],
        hi=[5, 5],
    )
    assert r.success
    assert abs(r.objective - 4.0) < 1e-2


def test_nonconvex_equality():
    # min x² + y²  s.t.  x·y = 1 on [0.1, 10]²  → 2 at (1, 1).
    r = minimize_global(
        var(0) ** 2 + var(1) ** 2,
        constraints=[eq(var(0) * var(1), 1.0)],
        lo=[0.1, 0.1],
        hi=[10, 10],
    )
    assert r.success
    assert abs(r.objective - 2.0) < 1e-2


def test_transcendental_atoms():
    # min eˣ − x on [−2, 2]: convex, optimum 1 at x = 0 (exercises exp + the
    # general factorable path, not just polynomials).
    r = minimize_global(var(0).exp() - var(0), lo=[-2.0], hi=[2.0])
    assert r.success
    assert abs(r.objective - 1.0) < 1e-3
    assert abs(float(r.x[0])) < 1e-2


def test_parallel_node_pool_matches():
    # The parallel pool is non-deterministic in node order but certifies the
    # same optimum as the serial driver.
    x, y = var(0), var(1)
    f = (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2
    kw = dict(lo=[-2, -1.5], hi=[2, 1.5], abs_gap=1e-4, rel_gap=1e-4, max_nodes=200_000)
    serial = minimize_global(f, threads=1, **kw)
    parallel = minimize_global(f, threads=4, **kw)
    assert serial.success and parallel.success
    assert abs(serial.objective - parallel.objective) < 1e-3


def test_infeasible():
    # x·y ≥ 100 is unreachable on [0, 1]².
    r = minimize_global(
        var(0) + var(1),
        constraints=[ge(var(0) * var(1), 100.0)],
        lo=[0, 0],
        hi=[1, 1],
    )
    assert r.status == "infeasible"


def test_negative_power_rejected():
    with pytest.raises(ValueError):
        var(0) ** -1
