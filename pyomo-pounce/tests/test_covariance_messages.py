"""What sens_covariance reports about its noise variance and its own
arguments: the residual groups by name, and the wording of the warnings and
errors.

The covariance itself is computed in pounce's Rust core. These pin what the
Python front end builds around it: which groups a variance is reported for,
which group an error names, and the exact text of messages callers match on.
"""
import warnings

import numpy as np
import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import (
    declare_sens_fitted,
    declare_sens_residual,
    sens_covariance,
    sens_information,
)


def grouped(n_hi=15, groups=("lo", "hi")):
    """y = a + b x fitted to two response groups, the second of `n_hi`
    points, each declared as its own residual group."""
    rng = np.random.default_rng(3)
    x1 = np.linspace(0.0, 2.0, 15)
    x2 = np.linspace(2.0, 4.0, n_hi)
    y1 = 1.0 + 0.5 * x1 + 0.1 * rng.standard_normal(15)
    y2 = 1.0 + 0.5 * x2 + 0.6 * rng.standard_normal(n_hi)
    m = pyo.ConcreteModel()
    m.I1 = pyo.RangeSet(0, 14)
    m.I2 = pyo.RangeSet(0, n_hi - 1)
    m.a = pyo.Var(initialize=0.0)
    m.b = pyo.Var(initialize=0.0)
    m.r1 = pyo.Var(m.I1, initialize=0.0)
    m.r2 = pyo.Var(m.I2, initialize=0.0)

    @m.Constraint(m.I1)
    def res1(mm, i):
        return mm.r1[i] == float(y1[i]) - mm.a - mm.b * float(x1[i])

    @m.Constraint(m.I2)
    def res2(mm, i):
        return mm.r2[i] == float(y2[i]) - mm.a - mm.b * float(x2[i])

    m.obj = pyo.Objective(
        expr=sum(m.r1[i] ** 2 for i in m.I1)
        + sum(m.r2[i] ** 2 for i in m.I2))
    declare_sens_fitted(m.a)
    declare_sens_fitted(m.b)
    declare_sens_residual(m.r1, group=groups[0])
    declare_sens_residual(m.r2, group=groups[1])
    pyo.SolverFactory("pounce").solve(m)
    X = np.column_stack([np.ones(15 + n_hi), np.concatenate([x1, x2])])
    return m, X


def test_a_scalar_sigma_sq_is_reported_for_every_group():
    """A scalar known variance applies to every declared group, and the
    result names each one with it. Equal variances pool, so the matrix is
    the homoscedastic sigma^2 inv(X^T X)."""
    m, X = grouped()
    cov = sens_covariance(m, sigma_sq=0.04)
    assert cov.sigma_sq == {"lo": 0.04, "hi": 0.04}
    np.testing.assert_allclose(cov.matrix, 0.04 * np.linalg.inv(X.T @ X),
                               rtol=1e-8)


def test_a_single_named_group_keeps_its_name():
    """One residual group declared with a name reports its variance under
    that name. Only an unnamed declaration reports a bare number."""
    m2 = pyo.ConcreteModel()
    m2.I = pyo.RangeSet(0, 9)
    x = np.linspace(0.0, 1.0, 10)
    y = 2.0 - x + 0.05 * np.random.default_rng(1).standard_normal(10)
    m2.a = pyo.Var(initialize=0.0)
    m2.b = pyo.Var(initialize=0.0)
    m2.r = pyo.Var(m2.I, initialize=0.0)

    @m2.Constraint(m2.I)
    def res(mm, i):
        return mm.r[i] == float(y[i]) - mm.a - mm.b * float(x[i])

    m2.obj = pyo.Objective(expr=sum(m2.r[i] ** 2 for i in m2.I))
    declare_sens_fitted(m2.a)
    declare_sens_fitted(m2.b)
    declare_sens_residual(m2.r, group="only")
    pyo.SolverFactory("pounce").solve(m2)
    cov = sens_covariance(m2)
    assert list(cov.sigma_sq) == ["only"]
    r = np.array([pyo.value(m2.r[i]) for i in m2.I])
    assert cov.sigma_sq["only"] == pytest.approx(float(r @ r) / 8, rel=1e-12)


def test_a_group_too_small_to_estimate_is_named():
    """The noise of a group with no more members than fitted parameters
    cannot be estimated, and the error names that group, not another."""
    m, _ = grouped(n_hi=2)
    with pytest.raises(ValueError, match="residual group 'hi' has 2 members"):
        sens_covariance(m)


def test_an_unknown_hessian_is_named_with_the_choices():
    m, _ = grouped()
    with pytest.raises(
            ValueError,
            match=r"^sens_covariance: hessian must be 'lagrangian' or "
                  r"'gauss-newton', got 'bogus'$"):
        sens_covariance(m, hessian="bogus")


def test_the_classification_warnings_keep_their_long_standing_names():
    """sens_covariance has always worded its bound-activity warnings as
    "covariance:", and sens_information as "sens_information:". Callers
    filter warnings on that text, so it stays as it was."""
    x = np.linspace(0.0, 4.0, 25)
    y = 1.5 - 0.7 * x + 0.3 * np.random.default_rng(42).standard_normal(25)
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 24)
    m.a = pyo.Var(bounds=(2.0, None), initialize=2.5)
    m.b = pyo.Var(initialize=0.0)
    m.r = pyo.Var(m.I, initialize=0.0)

    @m.Constraint(m.I)
    def res(mm, i):
        return mm.r[i] == float(y[i]) - mm.a - mm.b * float(x[i])

    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    declare_sens_fitted(m.a)
    declare_sens_fitted(m.b)
    declare_sens_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    held = "fitted parameter a is held by its bound"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sens_covariance(m)
        sens_information(m)
    msgs = [str(x.message) for x in w if held in str(x.message)]
    assert [s.split(":")[0] for s in msgs] == ["covariance", "sens_information"], msgs
