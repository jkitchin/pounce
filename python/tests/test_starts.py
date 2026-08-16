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


def test_project_to_feasible_inconsistent_is_absorbed_by_elastics():
    """gh#605 behaviour change: an inconsistent linearization no longer
    raises. Before #605 the projection QP was infeasible and the helper
    raised RuntimeError, returning nothing usable. It now carries elastic
    variables, so the same input yields the least-violating point."""

    # x0 + x1 == 1 AND x0 + x1 == 3 (same row twice via cl==cu rows).
    class TwoRows:
        def constraints(self, x):
            return np.array([x[0] + x[1], x[0] + x[1]])

        def jacobianstructure(self):
            return (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))

        def jacobian(self, x):
            return np.array([1.0, 1.0, 1.0, 1.0])

    x = pounce.project_to_feasible(
        TwoRows(), [0.0, 0.0], cl=[1.0, 3.0], cu=[1.0, 3.0]
    )
    # The rows conflict by 2, so the best any point can do is split the
    # difference: x0 + x1 = 2 leaves a residual of 1 on each row.
    total = x[0] + x[1]
    assert 1.0 <= total <= 3.0
    g = TwoRows().constraints(x)
    before = np.linalg.norm([1.0 - 0.0, 3.0 - 0.0])
    after = np.linalg.norm([1.0 - g[0], 3.0 - g[1]])
    assert after < before


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


# --------------------------------------------------------------------------
# gh#605 — sparse elastic normal step + safeguarding.
# --------------------------------------------------------------------------
class Circle:
    """g(x) = x0^2 + x1^2 = 1. At (0.05, 0.05) the Jacobian is (0.1, 0.1):
    the min-norm linearized correction is ~7 units long and lands where the
    TRUE violation is ~50x worse than it started."""

    def constraints(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] ** 2 + x[1] ** 2])

    def jacobianstructure(self):
        return (np.array([0, 0]), np.array([0, 1]))

    def jacobian(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([2 * x[0], 2 * x[1]])


class DupRows:
    """Two identical rows: J is rank 1 with m = 2."""

    def constraints(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] + x[1], x[0] + x[1]])

    def jacobianstructure(self):
        return (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))

    def jacobian(self, x):
        return np.array([1.0, 1.0, 1.0, 1.0])


class Chain:
    """n vars, n-1 rows, 2 nonzeros per row: g_i = x_i + x_{i+1} + 0.1 x_i^2."""

    def __init__(self, n):
        self.n, self.m = n, n - 1
        r = np.repeat(np.arange(self.m), 2)
        c = np.empty(2 * self.m, dtype=int)
        c[0::2] = np.arange(self.m)
        c[1::2] = np.arange(1, self.m + 1)
        self._rows, self._cols = r, c

    def constraints(self, x):
        x = np.asarray(x, dtype=float)
        return x[:-1] + x[1:] + 0.1 * x[:-1] ** 2

    def jacobianstructure(self):
        return (self._rows, self._cols)

    def jacobian(self, x):
        x = np.asarray(x, dtype=float)
        v = np.empty(2 * self.m)
        v[0::2] = 1.0 + 0.2 * x[:-1]
        v[1::2] = 1.0
        return v


def _violation(prob, x, cl, cu):
    g = np.asarray(prob.constraints(x), dtype=float)
    return float(np.linalg.norm(np.maximum(np.maximum(cl - g, g - cu), 0.0)))


def test_poor_linearization_never_worsens_violation():
    """gh#605: the safeguard's contract. The linearized step at (0.05, 0.05)
    lands where the violation is ~49.5; the returned point must be better
    than the 0.995 it started with, not worse."""
    prob = Circle()
    x0 = np.array([0.05, 0.05])
    cl = cu = np.array([1.0])
    v0 = _violation(prob, x0, cl, cu)
    x, rep = pounce.project_to_feasible(
        prob, x0, cl=cl, cu=cu, lb=[-10.0, -10.0], ub=[10.0, 10.0],
        return_report=True,
    )
    vN = _violation(prob, x, cl, cu)
    assert vN <= v0, f"projection made the violation worse: {v0} -> {vN}"
    assert vN < 0.5 * v0
    # The full-length step had to be rejected to get there.
    assert rep.rejected_trials > 0
    assert rep.violation_final <= rep.violation_initial


def test_rank_deficient_jacobian_still_projects():
    """Duplicate rows with a consistent RHS: rank-1 J, m = 2."""
    prob = DupRows()
    cl = cu = np.array([1.0, 1.0])
    x = pounce.project_to_feasible(
        prob, [0.0, 0.0], cl=cl, cu=cu, lb=[-10.0, -10.0], ub=[10.0, 10.0]
    )
    assert _violation(prob, x, cl, cu) < 1e-6
    # Min-norm solution of x0 + x1 = 1 is (0.5, 0.5).
    np.testing.assert_allclose(x, [0.5, 0.5], atol=1e-5)


def test_inconsistent_rows_degrade_via_elastics():
    """gh#605: an inconsistent linearization is absorbed by the elastic
    variables and returns the least-violating point, instead of raising."""
    prob = DupRows()
    # x0 + x1 == 1 AND x0 + x1 == 3 cannot both hold.
    cl = cu = np.array([1.0, 3.0])
    x0 = np.array([0.0, 0.0])
    v0 = _violation(prob, x0, cl, cu)
    x, rep = pounce.project_to_feasible(
        prob, x0, cl=cl, cu=cu, lb=[-10.0, -10.0], ub=[10.0, 10.0],
        return_report=True,
    )
    vN = _violation(prob, x, cl, cu)
    assert vN < v0, f"elastic solve did not improve: {v0} -> {vN}"
    # It cannot reach zero -- the rows genuinely conflict by 2.
    assert vN > 1e-3
    assert rep.elastic_total > 0.0


def test_large_sparse_model_allocates_no_dense_blocks():
    """gh#605 acceptance: no O(n^2) identity, no dense m x n Jacobian.

    A dense `P = eye(n)` alone would be 8*n^2 bytes = 72 MB at n = 3000;
    a dense Jacobian another 72 MB. Cap well under that but well over
    what the sparse path legitimately needs.
    """
    import tracemalloc

    n = 3000
    prob = Chain(n)
    x0 = np.full(n, 0.3)
    cl = cu = np.ones(n - 1)
    lb, ub = np.full(n, -10.0), np.full(n, 10.0)
    # Warm-up so lazily-allocated solver workspace is not attributed to
    # the measured call.
    pounce.project_to_feasible(
        prob, x0, cl=cl, cu=cu, lb=lb, ub=ub, max_iter=1, max_trials=1
    )
    tracemalloc.start()
    x = pounce.project_to_feasible(
        prob, x0, cl=cl, cu=cu, lb=lb, ub=ub, max_iter=1, max_trials=1
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 24e6, f"peak allocation {peak / 1e6:.1f} MB suggests a dense block"
    assert _violation(prob, x, cl, cu) < _violation(prob, x0, cl, cu)


def test_projection_preserves_bound_interiority():
    """A positive `margin` keeps the result strictly inside the box."""
    prob = Chain(50)
    n = 50
    lb, ub = np.zeros(n), np.ones(n)
    x, rep = pounce.project_to_feasible(
        prob, np.full(n, 0.9), cl=np.ones(n - 1), cu=np.ones(n - 1),
        lb=lb, ub=ub, margin=1e-4, return_report=True,
    )
    assert np.all(x >= lb + 1e-4 - 1e-9), "left the lower bound margin"
    assert np.all(x <= ub - 1e-4 + 1e-9), "left the upper bound margin"


def test_report_diagnostics_are_populated():
    prob = Chain(20)
    x, rep = pounce.project_to_feasible(
        prob, np.full(20, 0.3), cl=np.ones(19), cu=np.ones(19),
        lb=np.full(20, -10.0), ub=np.full(20, 10.0), return_report=True,
    )
    assert isinstance(rep, pounce.ProjectionReport)
    assert rep.violation_initial > 0
    assert rep.violation_final < rep.violation_initial
    assert rep.step_norm > 0
    assert rep.iterations >= 1
    assert rep.n_constraint_evals >= 1
    assert rep.n_jacobian_evals >= 1
    assert rep.termination
    assert rep.accepted
