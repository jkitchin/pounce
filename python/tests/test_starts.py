"""Tests for pounce.generate_starts / project_to_feasible / race_starts."""

import numpy as np
import pytest

import pounce


BOUNDS = [(-2.0, 2.0), (0.0, 4.0)]


def test_generate_starts_shape_and_box():
    starts = pounce.generate_starts(16, bounds=BOUNDS, seed=0)
    assert starts.shape == (16, 2)
    lo = np.array([-2.0, 0.0])
    hi = np.array([2.0, 4.0])
    assert np.all(starts >= lo) and np.all(starts <= hi)


def test_generate_starts_reproducible():
    a = pounce.generate_starts(5, bounds=BOUNDS, seed=42)
    b = pounce.generate_starts(5, bounds=BOUNDS, seed=42)
    np.testing.assert_array_equal(a, b)
    c = pounce.generate_starts(5, bounds=BOUNDS, seed=43)
    assert not np.array_equal(a, c)


def test_generate_starts_jitter_needs_x0():
    with pytest.raises(ValueError):
        pounce.generate_starts(3, strategy="jitter")
    starts = pounce.generate_starts(3, x0=[1.0, 1.0], strategy="jitter", seed=0)
    assert starts.shape == (3, 2)


def test_generate_starts_midpoint_first_point_deterministic():
    starts = pounce.generate_starts(4, bounds=BOUNDS, strategy="midpoint", seed=0)
    np.testing.assert_allclose(starts[0], [0.0, 2.0])


def test_generate_starts_unbounded_requires_anchor():
    with pytest.raises(ValueError):
        pounce.generate_starts(3, bounds=[(-2.0, 2.0), (None, None)])
    starts = pounce.generate_starts(
        3, bounds=[(-2.0, 2.0), (None, None)], x0=[0.0, 1.0], seed=0
    )
    assert starts.shape == (3, 2)
    assert np.all(np.abs(starts[:, 0]) <= 2.0)


def test_generate_starts_feeds_batch_and_find_minima_still_works():
    # find_minima imports the same sampler internals; smoke both paths.
    def hump(x):
        return float(np.sin(3 * x[0]) + 0.1 * x[0] ** 2)

    res = pounce.find_minima(hump, np.array([0.0]), method="multistart",
                             bounds=[(-3, 3)], n_minima=2, seed=1)
    assert len(res.minima) >= 1


class LinCon:
    """g(x) = [x0 + x1, x0 - x1], linear so projection is exact."""

    def constraints(self, x):
        return np.array([x[0] + x[1], x[0] - x[1]])

    def jacobianstructure(self):
        return (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))

    def jacobian(self, x):
        return np.array([1.0, 1.0, 1.0, -1.0])


def test_project_to_feasible_equality_and_inequality():
    # x0 violates the equality x0 + x1 = 1 and the inequality x0 - x1 <= 0.
    x0 = np.array([2.0, 2.0])
    x = pounce.project_to_feasible(
        LinCon(), x0, cl=[1.0, -2e19], cu=[1.0, 0.0],
        lb=[-5.0, -5.0], ub=[5.0, 5.0],
    )
    g = LinCon().constraints(x)
    assert g[0] == pytest.approx(1.0, abs=1e-6)
    assert g[1] <= 1e-6
    # Min-norm: stays as close to x0 as the constraints allow.
    assert np.linalg.norm(x - x0) <= np.linalg.norm(np.array([0.5, 0.5]) - x0) + 1e-6


def test_project_to_feasible_box_only():
    x = pounce.project_to_feasible(object(), [3.0, -7.0], lb=[0.0, 0.0], ub=[1.0, 1.0])
    np.testing.assert_allclose(x, [1.0, 0.0])


def test_project_to_feasible_inconsistent_raises():
    with pytest.raises(RuntimeError):
        # x0 + x1 == 1 AND x0 + x1 == 3 (same row twice via cl==cu rows).
        class TwoRows:
            def constraints(self, x):
                return np.array([x[0] + x[1], x[0] + x[1]])

            def jacobianstructure(self):
                return (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))

            def jacobian(self, x):
                return np.array([1.0, 1.0, 1.0, 1.0])

        pounce.project_to_feasible(
            TwoRows(), [0.0, 0.0], cl=[1.0, 3.0], cu=[1.0, 3.0]
        )


def test_race_starts_picks_the_better_basin():
    # Double well: minima near x = ±1 with f(-1) = 0 and f(+1) = 0.5.
    def f(x):
        return float((x[0] ** 2 - 1.0) ** 2 + 0.25 * (x[0] + 1.0))

    starts = np.array([[1.1], [-1.1]])
    best = pounce.race_starts(f, starts, iters=8, top=2)
    assert len(best) == 2
    # The winner must be the deeper (x = -1) basin.
    assert best[0].x[0] == pytest.approx(-1.0, abs=0.2)
    assert best[0].fun <= best[1].fun

    # The composable follow-up: continue the winner warm.
    ws = pounce.WarmStart.from_info(best[0].x, best[0].info)
    res = pounce.minimize(f, best[0].x, warm_start=ws)
    assert res.success
    assert res.x[0] == pytest.approx(-1.0290, abs=1e-2)
