"""A `maximize` objective, across every quantity the in-process solve reports.

`pounce.read_nl` does not hand a maximization to the engine as written.  It
negates the objective callbacks and records what it did in `nl.minimize`, so
everything the engine reports *about the objective* -- `info['obj_val']`,
`nl.gradient()`, and the multipliers, which are stationarity coefficients of
the objective it minimized -- is stated against `-f`, while everything the
caller reads off the model (`pyo.value(obj)`, `m.dual[c]`) is stated against
the `f` they wrote.  `pounce.sensitivity.objective_sign` is the conversion, and
until it existed nothing applied it.

**Why this was invisible.**  The factor is `+1` on every minimization, and a
minimization is the only model the corpus had: `maximize` appeared zero times
in `test_sens.py` and zero times in
`test_issue_878_objective_total_derivative.py`.  A corpus that is uniform in
exactly the dimension a sign acts on cannot tell a right sign from no sign at
all -- the same shape as the trajectory blind spots in the root CLAUDE.md, one
level down.

The symptom that named it: `sens_jacobian(of=<Objective>)` on a maximize model
returned `-df/dp`, silently, with no status, warning, or residual to disagree
with.  `of=<Var>` was never affected -- POUNCE reaches the same stationary
point either way, so `dx*/dp` does not know which sense asked.  That asymmetry
is itself a test below: a defect that moves one accessor and not its neighbour
is the kind a "spot-check one number" review passes.

ORACLES.  Every expected value here is closed form, computed without a solver,
and each fixture carries its own derivation.  The paired-sense tests are
stronger still: they assert the two spellings of *one* model against each
other, so they hold whatever the arithmetic is.
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
import pytest

from pyomo_pounce import declare_sens_param, sens_jacobian
from pyomo_pounce.sens import _REG


def _solve(m, **opts):
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10, **opts})
    return m


# ── the fixture ──────────────────────────────────────────────────────
#
#   f(x; p) = -(x - p)^2 + 3 p
#
# is concave in x with its peak at x* = p for every p, so
#
#   x*(p) = p,   f*(p) = 3 p,   df*/dp = 3,   dx*/dp = 1
#
# exactly, for all p.  The minimize arm optimizes -f, whose optimum is
# the same point and whose value and derivative are the negation:
# obj* = -3p, d obj*/dp = -3.  No solver produced any of those four
# numbers.
#
# `p` is a mutable Param pinned to a Var by its own defining equality,
# which is the form `declare_sens_param` wants and the form that leaves
# the model unrewritten (so the test's own model is the model solved).

def build(p=2.0, sense=pyo.maximize):
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.pv = pyo.Var(initialize=p)
    m.pin = pyo.Constraint(expr=m.pv == m.p)
    m.x = pyo.Var(initialize=1.0)
    f = -(m.x - m.pv) ** 2 + 3 * m.pv
    m.obj = pyo.Objective(expr=f if sense == pyo.maximize else -f,
                          sense=sense)
    return m


def declared(p=2.0, sense=pyo.maximize):
    m = build(p, sense)
    declare_sens_param(m.p)
    return _solve(m)


# ── the reported symptom ─────────────────────────────────────────────

@pytest.mark.parametrize("sense,expect", [(pyo.maximize, 3.0),
                                          (pyo.minimize, -3.0)])
def test_the_total_derivative_is_stated_against_the_objective_as_written(
        sense, expect):
    """`sens_jacobian(of=<Objective>)` differentiates `pyo.value(obj)`.

    df*/dp = +3 on the maximize spelling and -3 on the minimize one, in
    closed form.  Before the sign existed the maximize arm returned -3:
    the right magnitude, the wrong sign, and nothing to notice it by.
    """
    m = declared(sense=sense)
    assert sens_jacobian(m.obj, wrt=m.p) == pytest.approx(expect, abs=1e-7)


def test_the_two_spellings_of_one_model_disagree_by_exactly_a_sign():
    """The paired form of the test above, which holds whatever the
    closed form is: `max f` and `min -f` are the same model, so their
    optima coincide and their objective derivatives negate."""
    hi = declared(sense=pyo.maximize)
    lo = declared(sense=pyo.minimize)
    assert pyo.value(hi.x) == pytest.approx(pyo.value(lo.x), abs=1e-7)
    assert (sens_jacobian(hi.obj, wrt=hi.p)
            == pytest.approx(-sens_jacobian(lo.obj, wrt=lo.p), abs=1e-7))


def test_the_variable_jacobian_does_not_know_which_sense_asked():
    """`of=<Var>` is sense-independent -- POUNCE reaches the same
    stationary point either way -- so dx*/dp = 1 on both arms.

    This is the neighbour the defect did *not* move, and it is here on
    purpose: a fix that reached the variable path too would be wrong,
    and reviewing one number in isolation would not have said so.
    """
    for sense in (pyo.maximize, pyo.minimize):
        m = declared(sense=sense)
        assert sens_jacobian(m.x, wrt=m.p) == pytest.approx(1.0, abs=1e-7)


# ── the same sign, one layer down ────────────────────────────────────

@pytest.mark.parametrize("sense,expect", [(pyo.maximize, 6.0),
                                          (pyo.minimize, -6.0)])
def test_the_sessions_base_objective_is_the_value_the_model_states(
        sense, expect):
    """`session.base_obj` is documented as "exactly what
    `pyo.value(objective)` returns an instant after the solve", and at
    p = 2 that is f* = 3p = 6 on the maximize arm.

    It is not cosmetic: `sens_covariance(n_data=)` reads `base_obj` as
    the residual sum of squares, and an SSR cannot be negative.
    """
    m = declared(sense=sense)
    session = m.__dict__[_REG].session
    assert session.base_obj == pytest.approx(expect, abs=1e-7)
    assert session.base_obj == pytest.approx(pyo.value(m.obj), abs=1e-7)


@pytest.mark.parametrize("sense,expect", [(pyo.maximize, 6.0),
                                          (pyo.minimize, -6.0)])
def test_the_results_object_reports_the_objective_in_the_models_sense(
        sense, expect):
    """`SolverResults.problem.{upper,lower}_bound` are the final
    objective value on both routes; the `.sol` route states it in the
    model's sense, so this one must too or the two disagree on the same
    model."""
    m = build(sense=sense)
    declare_sens_param(m.p)
    res = pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10})
    assert float(res.problem.upper_bound) == pytest.approx(expect, abs=1e-7)
    assert float(res.problem.lower_bound) == pytest.approx(expect, abs=1e-7)


def test_the_objective_gradient_is_the_gradient_of_the_stated_objective():
    """The core session's `objective_gradient()` is documented as
    `grad_x f`, and `total_objective_derivative` is its only consumer --
    so the sign belongs there, at cache-fill time, and not in
    `Jacobian._value` where only the Pyomo layer would get it.

    At x* = p the peak is flat: df/dx = -2(x - p) = 0, and df/dp
    collects the explicit +3 and the pin.  Only the `pv` slot carries a
    nonzero, and it carries +3 on the maximize arm.
    """
    grads = {}
    for sense in (pyo.maximize, pyo.minimize):
        m = declared(sense=sense)
        session = m.__dict__[_REG].session
        grads[sense] = np.asarray(session.objective_gradient(), dtype=float)
    assert np.allclose(grads[pyo.maximize], -grads[pyo.minimize], atol=1e-7)
    assert np.abs(grads[pyo.maximize]).max() == pytest.approx(3.0, abs=1e-6)


# ── the neighbour the sense fix could have broken ────────────────────────────
#
# `sens_solve` warns when the declared residuals do not reproduce the
# objective, because `sens_covariance` reads the objective as SSR to estimate
# the noise variance.  That check compares against the objective the solver
# MINIMIZED: "the objective is the plain sum of squares" is a claim about a
# quantity being driven down, and `maximize -SSR` is the same least-squares
# problem spelled the other way round.
#
# Making `base_obj` sense-correct moved that comparison out from under the
# check, which then warned on every maximize spelling of a perfectly ordinary
# fit -- a false alarm on the exact models the sense fix exists to serve.  It
# was found by asking what else reads `base_obj`, not by any test, which is
# why there is one now.

_X = np.linspace(0.0, 1.0, 12)
_Y = 2.0 + 3.0 * _X + np.array(
    [0.041, -0.012, 0.055, 0.010, -0.038, 0.023,
     -0.005, 0.031, -0.047, 0.019, 0.007, -0.026])


def _fit(sense, weighted=False):
    """A straight-line least-squares fit, spelled either way.

    `minimize SSR` and `maximize -SSR` are the same problem: same optimum,
    same residuals, same covariance.  With `weighted=True` the objective is
    no longer the plain sum of squares and the warning is correct.
    """
    from pyomo_pounce import declare_sens_fitted, declare_sens_residual

    m = pyo.ConcreteModel()
    m.a = pyo.Var(initialize=1.0)
    m.b = pyo.Var(initialize=1.0)
    m.I = pyo.RangeSet(0, _X.size - 1)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.r[i] == _Y[i] - (mm.a + mm.b * _X[i]))
    ssr = sum((3.0 if weighted else 1.0) * m.r[i] ** 2 for i in m.I)
    m.o = pyo.Objective(expr=-ssr if sense == pyo.maximize else ssr,
                        sense=sense)
    declare_sens_fitted(m.a, m.b)
    declare_sens_residual(m.r)
    return m


def test_a_maximize_spelling_of_a_least_squares_fit_does_not_warn():
    """Both spellings are the same fit, so neither warns and both give
    the same covariance.  The equality is the real assertion: it holds
    whatever the data are."""
    import warnings

    from pyomo_pounce import sens_covariance

    ses = {}
    for sense in (pyo.minimize, pyo.maximize):
        m = _fit(sense)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10})
        ssr_warnings = [w for w in caught
                        if "declared residuals give SSR" in str(w.message)]
        assert not ssr_warnings, (
            f"{sense} spelling warned: {ssr_warnings[0].message}")
        ses[sense] = np.sqrt(np.diag(sens_covariance(m).matrix))

    assert np.allclose(ses[pyo.minimize], ses[pyo.maximize], rtol=1e-9)


@pytest.mark.parametrize("sense", [pyo.minimize, pyo.maximize])
def test_the_ssr_check_still_fires_when_the_objective_is_not_the_ssr(sense):
    """The other branch.  A weighted objective is a real mismatch and
    must warn in BOTH senses -- a check that stops warning is as broken
    as one that warns spuriously, and only a fixture that reaches this
    branch says which one this is."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyo.SolverFactory("pounce").solve(_fit(sense, weighted=True),
                                          options={"tol": 1e-10})
    assert [w for w in caught if "declared residuals give SSR" in str(w.message)]
