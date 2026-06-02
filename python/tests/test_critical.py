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


def test_muller_brown_reaction_barriers():
    """The Müller-Brown PES: 3 minima + 2 transition states, known energies.

    Mirrors python/examples/reaction_barrier.py.
    """
    A = np.array([-200.0, -100.0, -170.0, 15.0])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x0 = np.array([1.0, 0.0, -0.5, -1.0])
    y0 = np.array([0.0, 0.5, 1.5, 1.0])

    def V(z):
        x, y = z
        dx, dy = x - x0, y - y0
        return float(np.sum(A * np.exp(a * dx**2 + b * dx * dy + c * dy**2)))

    def grad(z):
        x, y = z
        dx, dy = x - x0, y - y0
        e = A * np.exp(a * dx**2 + b * dx * dy + c * dy**2)
        return np.array([np.sum(e * (2 * a * dx + b * dy)),
                         np.sum(e * (b * dx + 2 * c * dy))])

    def hess(z):
        x, y = z
        dx, dy = x - x0, y - y0
        e = A * np.exp(a * dx**2 + b * dx * dy + c * dy**2)
        px, py = 2 * a * dx + b * dy, b * dx + 2 * c * dy
        return np.array([[np.sum(e * (px * px + 2 * a)), np.sum(e * (px * py + b))],
                         [np.sum(e * (px * py + b)), np.sum(e * (py * py + 2 * c))]])

    bounds = [(-1.5, 1.2), (-0.5, 2.2)]

    states = pounce.find_minima(
        V, [-0.5, 1.4], method="flooding", jac=grad, hess=hess, bounds=bounds,
        n_minima=3, max_solves=120, patience=40, dedup=1e-2, seed=0,
        strategy_kw={"sigma": 0.4, "amplitude": 150.0},
        options={"print_level": 0, "tol": 1e-8},
    )
    assert len(states) == 3
    # Literature minima energies (Müller & Brown 1979).
    assert sorted(round(v, 1) for v in states.values) == [-146.7, -108.2, -80.8]

    ts = pounce.find_saddles(
        V, [0.0, 0.5], grad=grad, hess=hess, bounds=bounds, index=1,
        n_saddles=2, max_solves=120, patience=50, dedup=1e-2, seed=0,
        max_step=0.05, grad_tol=1e-5,
    )
    assert len(ts) == 2
    assert all(p.index == 1 for p in ts.points)
    # Literature transition-state energies.
    assert sorted(round(p.f, 1) for p in ts.points) == [-72.2, -40.7]


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
