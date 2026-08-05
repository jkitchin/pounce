"""retain_kkt() (covariance roadmap item 4): keep the KKT factor with
nothing declared, so covariance(m, wrt=...) and information(m, wrt=...)
work on any block without a declared default. The roadmap's truth
table: no declarations and no retain means no factor and the no-session
error; retain alone keeps the factor but covariance(m) has no default
and stays an error; retain plus declarations changes nothing."""
import gc
import weakref

import numpy as np
import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401
from pyomo_pounce import (
    covariance,
    declare_fitted,
    declare_residual,
    declare_sens_param,
    gradient,
    information,
    release_kkt,
    retain_kkt,
)

N = 25
SIGMA = 0.3


def linear_data():
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 4.0, N)
    y = 1.5 - 0.7 * x + SIGMA * rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])
    return x, y, X


def linear_model(x, y, declare=True):
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, len(x) - 1)
    m.a = pyo.Var(initialize=0.0)
    m.b = pyo.Var(initialize=0.0)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.r[i] == float(y[i]) - mm.a
        - mm.b * float(x[i]))
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    if declare:
        declare_fitted(m.a)
        declare_fitted(m.b)
        declare_residual(m.r)
    return m


def test_retain_only_serves_wrt_queries():
    # retain_kkt() with nothing declared: the factor is kept and any
    # wrt block works, matching a declared solve's answers exactly
    x, y, X = linear_data()
    m = linear_model(x, y, declare=False)
    retain_kkt(m)
    pyo.SolverFactory("pounce").solve(m)
    C = np.linalg.inv(X.T @ X)
    cov_a = covariance(m, sigma_sq=SIGMA**2, wrt=[m.a])
    assert cov_a[m.a] == pytest.approx(SIGMA**2 * C[0, 0], rel=1e-9)
    info_a = information(m, wrt=[m.a])
    assert info_a[m.a] == pytest.approx(2.0 / C[0, 0], rel=1e-9)
    band = covariance(m, sigma_sq=SIGMA**2, wrt=m.r)
    np.testing.assert_allclose(band.matrix, SIGMA**2 * (X @ C @ X.T),
                               rtol=1e-8, atol=1e-12)
    # and against a declared solve of the same model, entry for entry
    m2 = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m2)
    cov2 = covariance(m2, sigma_sq=SIGMA**2, wrt=[m2.a])
    assert cov_a[m.a] == pytest.approx(cov2[m2.a], rel=1e-12)


def test_retain_only_has_no_default_block():
    # the truth table's middle row: factor kept, but covariance(model)
    # without wrt= has no default to reduce onto
    x, y, _ = linear_data()
    m = linear_model(x, y, declare=False)
    retain_kkt(m)
    pyo.SolverFactory("pounce").solve(m)
    with pytest.raises(RuntimeError, match="no fitted parameters"):
        covariance(m, sigma_sq=SIGMA**2)
    with pytest.raises(RuntimeError, match="no fitted parameters"):
        information(m)


def test_no_declarations_no_retain_pays_nothing():
    # the table's first row: an undeclared solve takes the ordinary
    # path (no registry, no session) and the accessors say why
    x, y, _ = linear_data()
    m = linear_model(x, y, declare=False)
    pyo.SolverFactory("pounce").solve(m)
    assert "_pounce_sens" not in m.__dict__
    with pytest.raises(RuntimeError, match="retain_kkt"):
        covariance(m, sigma_sq=SIGMA**2, wrt=[m.a])
    with pytest.raises(RuntimeError, match="retain_kkt"):
        information(m, wrt=[m.a])


def test_retain_plus_declarations_changes_nothing():
    # the table's last row: retain is idempotent beside declarations
    x, y, _ = linear_data()
    m = linear_model(x, y)
    retain_kkt(m)
    pyo.SolverFactory("pounce").solve(m)
    m2 = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m2)
    cov = covariance(m)
    cov2 = covariance(m2)
    np.testing.assert_array_equal(cov.matrix, cov2.matrix)
    assert cov.sigma_sq == cov2.sigma_sq


def test_retain_survives_clone():
    # the registry is deepcopy-aware; the retain intent follows a clone
    x, y, _ = linear_data()
    m = linear_model(x, y, declare=False)
    retain_kkt(m)
    c = m.clone()
    pyo.SolverFactory("pounce").solve(c)
    info = information(c, wrt=[c.a])
    assert np.isfinite(info[c.a])


def test_retain_only_estimated_sigma_refuses():
    # with nothing declared fitted the degrees of freedom for a noise
    # ESTIMATE are unknown: silently dividing by n would bias every
    # variance low by n/(n-p), so both estimation routes raise and
    # point at sigma_sq= or declare_fitted()
    x, y, _ = linear_data()
    m = linear_model(x, y, declare=False)
    retain_kkt(m)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    with pytest.raises(ValueError, match="degrees of freedom"):
        covariance(m, wrt=[m.a])
    m2 = linear_model(x, y, declare=False)
    retain_kkt(m2)
    pyo.SolverFactory("pounce").solve(m2)
    with pytest.raises(ValueError, match="degrees of freedom"):
        covariance(m2, n_data=N, wrt=[m2.a])


def test_retain_only_sigma_still_required():
    # retain keeps the factor, not a noise model: covariance without
    # sigma_sq and without declared residuals must say what is missing
    x, y, _ = linear_data()
    m = linear_model(x, y, declare=False)
    retain_kkt(m)
    pyo.SolverFactory("pounce").solve(m)
    with pytest.raises(ValueError, match="noise variance is unknown"):
        covariance(m, wrt=[m.a])


def test_release_kkt_drops_the_factor():
    # release frees the model's hold; the accessors then raise their
    # no-session error, and a second release reports nothing to do
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    assert release_kkt(m) is True
    with pytest.raises(RuntimeError, match="no sensitivity session"):
        covariance(m, sigma_sq=SIGMA**2)
    assert release_kkt(m) is False


def test_release_kkt_with_nothing_held():
    # no registry at all: a clean False, no error
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - 1.0) ** 2)
    assert release_kkt(m) is False


def test_release_kkt_frees_the_session():
    # the point of the call: with no result outstanding, nothing else
    # holds the session, so the factor is collected at the release and
    # not merely hidden from the accessors
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    held = weakref.ref(m.__dict__["_pounce_sens"].session)
    assert held() is not None
    assert release_kkt(m) is True
    assert held() is None


def test_release_kkt_keeps_declarations_for_the_next_solve():
    # the declarations survive release: the next solve keeps its
    # factor again and covariance(m) still has its default block
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    release_kkt(m)
    pyo.SolverFactory("pounce").solve(m)
    C = np.linalg.inv(X.T @ X)
    assert covariance(m, sigma_sq=SIGMA**2)[m.a] == pytest.approx(
        SIGMA**2 * C[0, 0], rel=1e-9)


def test_release_kkt_keeps_the_retain_flag_for_the_next_solve():
    # same on the retain-only path: release does not undo retain_kkt()
    x, y, X = linear_data()
    m = linear_model(x, y, declare=False)
    retain_kkt(m)
    pyo.SolverFactory("pounce").solve(m)
    release_kkt(m)
    pyo.SolverFactory("pounce").solve(m)
    C = np.linalg.inv(X.T @ X)
    assert covariance(m, sigma_sq=SIGMA**2,
                      wrt=[m.a])[m.a] == pytest.approx(
        SIGMA**2 * C[0, 0], rel=1e-9)


def test_release_kkt_pending_conditioned_on_survives():
    # a result created before release holds its own reference through
    # the pending conditioned_on computation, so reading it afterwards
    # still works: release drops the model's hold, not the result's
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    cov = covariance(m, sigma_sq=SIGMA**2)
    assert release_kkt(m) is True
    assert cov.conditioned_on == ()
    C = np.linalg.inv(X.T @ X)
    assert cov[m.a] == pytest.approx(SIGMA**2 * C[0, 0], rel=1e-9)


def test_release_kkt_leaves_a_gradient_working():
    # the other result that holds the session: a Gradient reads the
    # factor on every lookup, so it too keeps working -- and keeps the
    # factor alive -- across a release
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=3.0, mutable=True)
    m.x = pyo.Var(initialize=0.0)
    m.con = pyo.Constraint(expr=m.x == 2.0 * m.p)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    g = gradient(wrt=m.p)
    held = weakref.ref(m.__dict__["_pounce_sens"].session)
    assert release_kkt(m) is True
    assert held() is not None                 # the Gradient still holds it
    assert g[m.x, m.p] == pytest.approx(2.0, rel=1e-9)
    del g
    gc.collect()
    assert held() is None
