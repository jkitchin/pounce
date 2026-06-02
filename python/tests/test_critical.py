"""Tests for find_critical_points and find_saddles."""

import os

os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

import pounce


# f = (x^2-1)^2 + (y^2-1)^2: 4 minima, 4 index-1 saddles, 1 maximum.
def fun(z):
    x, y = z
    return (x * x - 1) ** 2 + (y * y - 1) ** 2


def grad(z):
    x, y = z
    return np.array([4 * x * (x * x - 1), 4 * y * (y * y - 1)])


def hess(z):
    x, y = z
    return np.array([[4 * (3 * x * x - 1), 0.0], [0.0, 4 * (3 * y * y - 1)]])


BOUNDS = [(-1.5, 1.5), (-1.5, 1.5)]
OPTS = {"print_level": 0, "tol": 1e-10}


def test_route_a_enumerates_and_classifies():
    r = pounce.find_critical_points(
        fun, [0.3, 0.4], grad=grad, hess=hess, bounds=BOUNDS,
        method="deflation", n_points=12, max_solves=250, patience=50,
        dedup=1e-2, seed=0, options=OPTS,
    )
    assert len(r) == 9
    assert len(r.minima) == 4
    assert len(r.saddles) == 4
    assert len(r.maxima) == 1
    # Every reported point is genuinely stationary.
    for p in r.points:
        assert p.grad_norm <= 1e-6
    # Indices and energies line up with the analytic answer.
    assert all(p.f == pytest.approx(0.0, abs=1e-6) for p in r.minima)
    assert all(p.f == pytest.approx(1.0, abs=1e-6) for p in r.saddles)
    assert r.maxima[0].f == pytest.approx(2.0, abs=1e-6)


def test_route_b_finds_index1_saddles():
    s = pounce.find_saddles(
        fun, [0.3, 0.4], grad=grad, hess=hess, bounds=BOUNDS,
        index=1, n_saddles=4, max_solves=150, patience=60, dedup=1e-2, seed=0,
    )
    assert len(s) == 4
    assert s.status == "target_reached"
    for p in s.points:
        assert p.index == 1
        assert p.grad_norm <= 1e-6
        assert p.f == pytest.approx(1.0, abs=1e-6)
    # The four saddles are the expected (+-1,0),(0,+-1).
    locs = sorted(tuple(np.round(p.x, 3)) for p in s.points)
    expected = sorted([(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)])
    for got, exp in zip(locs, expected):
        assert got == pytest.approx(exp, abs=1e-2)


def test_kind_labels():
    r = pounce.find_critical_points(
        fun, [0.3, 0.4], grad=grad, hess=hess, bounds=BOUNDS,
        method="multistart", n_points=12, max_solves=250, patience=50,
        dedup=1e-2, seed=1, options=OPTS,
    )
    kinds = {p.kind for p in r.points}
    assert "minimum" in kinds
    assert "saddle(index=1)" in kinds
    assert "maximum" in kinds
