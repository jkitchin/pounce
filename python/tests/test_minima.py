"""Tests for pounce.find_minima (multiple-minima global search).

Run quietly: the unconstrained minimize facade logs a harmless
jacobian AttributeError per solve; RUST_LOG=off silences it.
"""

import os

os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

import pounce


# --- Test landscapes -------------------------------------------------------
def six_hump_camel():
    def fun(z):
        x, y = z
        return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2

    def jac(z):
        x, y = z
        return np.array([(8 - 8.4 * x**2 + 2 * x**4) * x + y,
                         x + (-8 + 16 * y**2) * y])

    def hess(z):
        x, y = z
        return np.array([[8 - 25.2 * x**2 + 10 * x**4, 1.0],
                         [1.0, -8 + 48 * y**2]])

    return fun, jac, hess, [(-2.0, 2.0), (-1.5, 1.5)]


def rastrigin():
    def fun(z):
        return 10 * len(z) + sum(zi * zi - 10 * np.cos(2 * np.pi * zi) for zi in z)

    def jac(z):
        return np.array([2 * zi + 20 * np.pi * np.sin(2 * np.pi * zi) for zi in z])

    def hess(z):
        return np.diag([2 + 40 * np.pi**2 * np.cos(2 * np.pi * zi) for zi in z])

    return fun, jac, hess, [(-5.12, 5.12)] * 2


REPULSION = ["deflation", "flooding", "tunneling"]
RESTART = ["multistart", "mlsl"]
HOPPING = ["basinhopping"]
ALL = REPULSION + RESTART + HOPPING

OPTS = {"print_level": 0, "tol": 1e-9}
GLOBAL_CAMEL = -1.031628


@pytest.mark.parametrize("method", ALL)
def test_finds_global_minimum(method):
    """Every method should at least locate a global minimum of the camel."""
    fun, jac, hess, bounds = six_hump_camel()
    r = pounce.find_minima(
        fun, [0.5, 0.5], method=method, jac=jac, hess=hess, bounds=bounds,
        n_minima=6, max_solves=160, patience=40, dedup=1e-3, seed=0, options=OPTS,
    )
    assert len(r) >= 1
    assert r.fun == pytest.approx(GLOBAL_CAMEL, abs=1e-3)
    assert r.status in ("target_reached", "converged", "budget_exhausted")
    # Reported minima are sorted ascending by objective.
    assert r.values == sorted(r.values)
    # Every reported point is a true minimum (PSD Hessian).
    for x in r.minima:
        H = hess(x)
        assert np.linalg.eigvalsh(0.5 * (H + H.T))[0] >= -1e-5


@pytest.mark.parametrize("method", ["deflation", "flooding", "multistart", "mlsl"])
def test_enumerates_all_camel_minima(method):
    """Enumeration-oriented methods recover all six camel minima."""
    fun, jac, hess, bounds = six_hump_camel()
    r = pounce.find_minima(
        fun, [0.5, 0.5], method=method, jac=jac, hess=hess, bounds=bounds,
        n_minima=6, max_solves=300, patience=80, dedup=1e-3, seed=0, options=OPTS,
    )
    assert len(r) == 6
    assert r.status == "target_reached"


def test_no_duplicates():
    fun, jac, hess, bounds = six_hump_camel()
    r = pounce.find_minima(
        fun, [0.5, 0.5], method="deflation", jac=jac, hess=hess, bounds=bounds,
        n_minima=6, dedup=1e-3, seed=0, options=OPTS,
    )
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            assert np.linalg.norm(r.minima[i] - r.minima[j]) > 1e-3


def test_patience_terminates_when_few_minima():
    """A bowl has one minimum; patience must stop and report 'converged'."""
    fun = lambda z: (z[0] - 1) ** 2 + (z[1] + 2) ** 2
    jac = lambda z: np.array([2 * (z[0] - 1), 2 * (z[1] + 2)])
    hess = lambda z: np.array([[2.0, 0.0], [0.0, 2.0]])
    r = pounce.find_minima(
        fun, [0.0, 0.0], method="multistart", jac=jac, hess=hess,
        bounds=[(-5, 5), (-5, 5)], n_minima=5, patience=6, max_solves=100,
        seed=0, options=OPTS,
    )
    assert len(r) == 1
    assert r.status == "converged"
    assert r.x == pytest.approx([1.0, -2.0], abs=1e-5)


def test_mlsl_terminates_and_respects_budget():
    """pounce#103: MLSL must not spin on solve-less rounds.

    In moderate dimension the clustering radius ``γ·√n·(ln N/N)^(1/n)``
    barely shrinks, so once the single minimum of this bowl is found almost
    every later sample is filtered out and no local solve fires. Before the
    fix, termination was solve-gated — neither ``max_solves`` nor
    ``patience`` advanced on those rounds — and the loop spun forever while
    the pool grew under an O(N²) scan. A sample budget
    (``max_solves`` × ``samples_per_round``) now bounds the loop directly,
    independent of whether any solve fires.
    """
    n = 8
    target = np.arange(1, n + 1) / (n + 1)  # interior optimum, well inside box
    fun = lambda z: float(np.sum((np.asarray(z, float) - target) ** 2))
    jac = lambda z: 2 * (np.asarray(z, float) - target)
    hess = lambda z: 2 * np.eye(n)
    bounds = [(-1.0, 1.0)] * n
    r = pounce.find_minima(
        fun, np.zeros(n), method="mlsl", jac=jac, hess=hess, bounds=bounds,
        n_minima=6, max_solves=20, patience=8,
        strategy_kw={"samples_per_round": 20},
        seed=0, options=OPTS,
    )
    # It returns at all (no hang) with a valid terminal status...
    assert r.status in ("converged", "budget_exhausted")
    # ...having located the single minimum...
    assert len(r) >= 1
    assert r.fun == pytest.approx(0.0, abs=1e-6)
    # ...and stayed within both budgets (solves, and the derived sample
    # ceiling max_solves * samples_per_round).
    assert r.n_solves <= 20


def test_budget_is_respected():
    fun, jac, hess, bounds = rastrigin()
    r = pounce.find_minima(
        fun, [0.3, 0.3], method="multistart", jac=jac, hess=hess, bounds=bounds,
        n_minima=999, max_solves=12, patience=999, seed=0, options=OPTS,
    )
    assert r.n_solves <= 12
    assert r.status == "budget_exhausted"


def test_callback_invoked_per_minimum():
    fun, jac, hess, bounds = six_hump_camel()
    seen = []
    pounce.find_minima(
        fun, [0.5, 0.5], method="deflation", jac=jac, hess=hess, bounds=bounds,
        n_minima=4, max_solves=80, dedup=1e-3, seed=0, options=OPTS,
        callback=lambda x, f: seen.append((x.copy(), f)),
    )
    assert len(seen) == 4


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        pounce.find_minima(lambda z: z[0] ** 2, [0.0], method="nope")


def test_rastrigin_global_via_hopping():
    fun, jac, hess, bounds = rastrigin()
    r = pounce.find_minima(
        fun, [0.3, 0.3], method="basinhopping", jac=jac, hess=hess, bounds=bounds,
        n_minima=8, max_solves=200, patience=60, dedup=1e-2, seed=1, options=OPTS,
    )
    assert r.fun == pytest.approx(0.0, abs=1e-4)


def test_find_minima_and_saddles_reject_wrong_length_bounds():
    """The sampling-based searches broadcast ``lo + (hi-lo)*u`` over the box, so
    a wrong-length ``bounds`` could silently sample every dimension from one
    variable's interval. Length is now validated up front (matching minimize)."""
    def fun(z):
        return float(z @ z)

    def jac(z):
        return 2.0 * np.asarray(z, float)

    # two variables, one bound pair
    with pytest.raises(ValueError, match="bounds has 1 entry but the problem has 2"):
        pounce.find_minima(fun, x0=np.zeros(2), jac=jac, bounds=[(-2, 2)],
                           options={"print_level": 0})

    def hess(z):
        return 2.0 * np.eye(2)

    with pytest.raises(ValueError, match="bounds has 1 entry but the problem has 2"):
        pounce.find_saddles(fun, x0=np.zeros(2), grad=jac, hess=hess,
                            bounds=[(-2, 2)])

    # correct length is accepted (smoke: returns a result, no exception)
    r = pounce.find_minima(fun, x0=np.zeros(2), jac=jac,
                           bounds=[(-2, 2), (-2, 2)], max_solves=5,
                           options={"print_level": 0})
    assert r is not None


def test_find_minima_rejects_nonsensical_budget():
    """``n_minima``/``patience``/``max_solves`` below 1 are nonsensical and used
    to silently no-op or return a wrong count; they now raise clear errors."""
    f = lambda v: float(v @ v)
    g = lambda v: 2.0 * np.asarray(v, float)
    box = [(-2, 2), (-2, 2)]
    opts = {"print_level": 0}
    with pytest.raises(ValueError, match="n_minima must be >= 1"):
        pounce.find_minima(f, np.zeros(2), jac=g, bounds=box, n_minima=0, options=opts)
    with pytest.raises(ValueError, match="patience must be >= 1"):
        pounce.find_minima(f, np.zeros(2), jac=g, bounds=box, patience=-1, options=opts)
    with pytest.raises(ValueError, match="max_solves must be >= 1"):
        pounce.find_minima(f, np.zeros(2), jac=g, bounds=box, max_solves=0, options=opts)


def test_result_repr_survives_seed_recorded_without_solve():
    """A seed ``x0`` accepted straight into the archive is recorded with a stub
    ``OptimizeResult`` carrying ``info={}``. scipy's shared result repr aligns
    keys via ``max(map(len, d.keys()))`` and so raises ``ValueError`` on that
    empty nested dict — the very first ``print(res)`` in a REPL used to crash
    (gh #340). Printing the result (and the stub itself) must now just work."""
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    res = pounce.find_minima(
        f, x0=[0.0, 0.0], bounds=[(-5, 5), (-5, 5)], n_minima=3,
        options={"print_level": 0},
    )

    # The seed sits at the global minimum, so it is recorded with the empty-info
    # stub; repr/str of the container and that stub must not raise.
    stub = res.results[0]
    assert stub.info == {}
    text = repr(stub)
    assert "info: {}" in text
    assert "recorded" in text
    # The MinimaResult dataclass repr recurses into results[0]; this is what the
    # issue's `print(res)` exercised.
    assert repr(res)
    assert str(res)
