"""User NLP scaling from the `scaling_factor` Suffix (gh #483).

A Pyomo user who tags the standard `scaling_factor` Suffix and sets
`nlp_scaling_method=user-scaling` — the workflow that works with Ipopt
through ASL — used to get no scaling at all and no message saying so:
pyomo-pounce contained no scaling code, so nothing ever populated the
solver's user-scaling channel and the option meant "none".

Three things are pinned here.

1. **The Suffix is read the way AMPL/Ipopt read it** (`test_read_*`):
   export-enabled only, containers expanded, inactive constraints and
   fixed variables skipped, untagged components unscaled.
2. **It reaches the solver on both paths** — the ASL/subprocess path
   through the writer's `.nl` suffix segments, and the in-process
   sensitivity path through `Problem.set_problem_scaling`.
3. **Nothing it cannot honor is dropped quietly**: a `user-scaling`
   request with no Suffix to apply warns.

The sensitivity accessors get their own axis. `natural_units_conj`
translates `df` / `dc` / `dd` — and, since gh #486 stage 3, the
per-variable factors — back to model units for any scaling method, so
user scaling *should* be invisible to every accessor. That is the kind
of claim worth proving rather than assuming, so each one is checked
against the same quantity computed with no scaling engaged. Variable
factors get the same treatment on their own axis (`solve_pair(x_scale=…)`):
stages 1 and 2 refused those queries, and what replaces the refusal is
not "it runs" but "it agrees with the unscaled answer".
"""
import warnings

import numpy as np
import pytest
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import (
    sens_covariance,
    declare_sens_fitted,
    declare_sens_residual,
    declare_sens_param,
    sens_solution,
    sens_jacobian,
    sens_information,
    sens_release_kkt,
    sens_retain_kkt,
)
from pyomo_pounce.scaling import (
    problem_scaling,
    read_scaling,
    user_scaling_requested,
)

USER_SCALING = {"nlp_scaling_method": "user-scaling"}


# ── the Suffix reader ────────────────────────────────────────────────────────

def tagged_model(**tags):
    """`min (x1-2)^4 + (x2-3)^2` s.t. `x1*x2 >= 1`, `x1 - x2 == 0.5`,
    with an export `scaling_factor` Suffix carrying `tags`."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize=1.0)
    m.o = pyo.Objective(expr=(m.x[1] - 2) ** 4 + (m.x[2] - 3) ** 2)
    m.c1 = pyo.Constraint(expr=m.x[1] * m.x[2] >= 1)
    m.c2 = pyo.Constraint(expr=m.x[1] - m.x[2] == 0.5)
    if tags:
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        for name, value in tags.items():
            key = {"o": m.o, "c1": m.c1, "c2": m.c2,
                   "x1": m.x[1], "x2": m.x[2]}[name]
            m.scaling_factor[key] = value
    return m


def test_read_no_suffix_is_none():
    # "the user supplied nothing" must stay distinguishable from "the
    # user asked for 1.0 everywhere": the first leaves the solver alone.
    assert read_scaling(tagged_model()) is None


def test_read_splits_objective_constraints_and_variables():
    m = tagged_model(o=100.0, c1=10.0, x1=3.0)
    obj, cons, variables = read_scaling(m)
    assert obj == 100.0
    assert cons == {m.c1: 10.0}
    assert variables == [(m.x[1], 3.0)]


def test_read_ignores_a_non_export_suffix():
    # A LOCAL Suffix never reaches a solver, so it must not look like
    # a scaling request here either.
    m = tagged_model()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.LOCAL)
    m.scaling_factor[m.o] = 100.0
    assert read_scaling(m) is None


def test_read_expands_a_container_entry():
    # `scaling_factor[m.c] = s` on an IndexedConstraint applies to every
    # member -- which is how the NL writer expands it, so both solve
    # paths must agree.
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], initialize=1.0)
    m.o = pyo.Objective(expr=m.x[1] ** 2 + m.x[2] ** 2)
    m.c = pyo.Constraint([1, 2], rule=lambda mm, i: mm.x[i] >= 0.5)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.c] = 7.0
    _, cons, _ = read_scaling(m)
    assert cons == {m.c[1]: 7.0, m.c[2]: 7.0}


def test_read_skips_inactive_constraints_and_fixed_variables():
    # Neither is a row/column of the problem the solver is handed, so a
    # factor on one is not a request pounce is dropping.
    m = tagged_model(o=100.0, c1=10.0, x1=3.0)
    m.c1.deactivate()
    m.x[1].fix(1.0)
    obj, cons, variables = read_scaling(m)
    assert obj == 100.0
    assert cons == {}
    assert variables == []


def test_read_treats_a_zero_factor_as_untagged():
    # AMPL's suffix default is 0 and 0 is not a usable scale factor.
    m = tagged_model(o=0.0, c1=0.0)
    obj, cons, _ = read_scaling(m)
    assert obj == 1.0
    assert cons == {}


def test_read_finds_a_block_local_suffix():
    m = tagged_model()
    m.b = pyo.Block()
    m.b.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.b.scaling_factor[m.c1] = 10.0
    _, cons, _ = read_scaling(m)
    assert cons == {m.c1: 10.0}


def test_user_scaling_requested_reads_the_option():
    assert user_scaling_requested(USER_SCALING)
    assert user_scaling_requested({"nlp_scaling_method": " User-Scaling "})
    assert not user_scaling_requested({"nlp_scaling_method": "gradient-based"})
    assert not user_scaling_requested({})
    assert not user_scaling_requested(None)


COLS = ["x[1]", "x[2]"]


def test_problem_scaling_builds_dense_row_vectors():
    # Untagged rows stay at 1.0, and a tagged constraint the solve does
    # not have a row for is skipped rather than mis-indexed.
    m = tagged_model(o=100.0, c1=10.0)
    obj, g, x, given = problem_scaling(m, ["c1", "c2"], {}, COLS)
    assert obj == 100.0
    assert g == [10.0, 1.0]
    assert given == [True, False], "c2 has no Suffix entry"
    assert x is None, "no variable entry means no change of variables"
    obj, g, x, given = problem_scaling(m, ["c2"], {}, COLS)
    assert g == [1.0]
    assert given == [False]


def test_problem_scaling_follows_the_surgery_alias():
    # The declared-parameter surgery renames replaced rows on the clone
    # that is actually solved; the Suffix is keyed by the original.
    m = tagged_model(c1=10.0)
    obj, g, _, given = problem_scaling(m, ["_pounce.c1", "c2"],
                                       {"c1": "_pounce.c1"}, COLS)
    assert g == [10.0, 1.0]
    assert given == [True, False]


def test_problem_scaling_builds_the_variable_vector():
    # gh #486 stage 3: variable entries are translated for the
    # in-process path too, dense and in `.col` order, so the column a
    # factor lands on is the one the Suffix named.
    m = tagged_model(o=100.0, x2=0.25)
    _, _, x, _ = problem_scaling(m, ["c1", "c2"], {}, COLS)
    assert x == [1.0, 0.25]


def test_problem_scaling_skips_a_variable_with_no_column():
    # A component the written model has no column for is not a column
    # pounce could rescale -- the same rule the rows follow, and the
    # alternative is a silently mis-indexed factor.
    m = tagged_model(x1=3.0)
    _, _, x, _ = problem_scaling(m, ["c1", "c2"], {}, ["x[2]"])
    assert x is None


# ── warnings ─────────────────────────────────────────────────────────────────

def test_solve_applies_a_variable_factor():
    # gh #486 stage 2 inverted this: the core applies variable factors
    # as a change of variables, so an ordinary solve runs instead of
    # raising, and reports the solution in the model's own units.
    s = pyo.SolverFactory("pounce")
    s.options.update(USER_SCALING)
    m = tagged_model(o=100.0, c1=10.0, x1=3.0)
    res = s.solve(m)
    assert res.solver.termination_condition == TerminationCondition.optimal

    plain = tagged_model()
    pyo.SolverFactory("pounce").solve(plain)
    for i in m.x:
        assert pyo.value(m.x[i]) == pytest.approx(
            pyo.value(plain.x[i]), abs=1e-6
        ), "the factor changed conditioning, not the answer"


def test_solve_warns_when_user_scaling_has_nothing_to_apply():
    s = pyo.SolverFactory("pounce")
    s.options.update(USER_SCALING)
    with pytest.warns(UserWarning, match="no export-enabled"):
        s.solve(tagged_model())


def test_a_variable_factor_is_inert_without_the_option():
    # A `scaling_factor` Suffix also drives Pyomo's own
    # `core.scale_model`; carrying one must not fail a default solve.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pyo.SolverFactory("pounce").solve(tagged_model(o=100.0, x1=3.0))


# ── the scaling reaches the solver, and the answer does not move ─────────────

def test_asl_path_solves_with_user_scaling():
    """The subprocess path: the writer emits the Suffix as `.nl` suffix
    segments and the solver reads them. Scaling changes conditioning,
    never the optimum, so the tagged solve must land where the untagged
    one does."""
    s = pyo.SolverFactory("pounce")
    s.options.update(USER_SCALING)
    scaled = tagged_model(o=100.0, c1=10.0)
    s.solve(scaled)
    plain = tagged_model()
    pyo.SolverFactory("pounce").solve(plain)
    assert pyo.value(scaled.x[1]) == pytest.approx(pyo.value(plain.x[1]),
                                                   abs=1e-6)
    assert pyo.value(scaled.x[2]) == pytest.approx(pyo.value(plain.x[2]),
                                                   abs=1e-6)
    assert pyo.value(scaled.o) == pytest.approx(pyo.value(plain.o), abs=1e-8)


# ── the in-process sensitivity path ─────────────────────────────────────────

N = 25
SIGMA = 0.3


def linear_data():
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 4.0, N)
    y = 1.5 - 0.7 * x + SIGMA * rng.standard_normal(N)
    return x, y, np.column_stack([np.ones(N), x])


def linear_model(x, y, scale=None, x_scale=None, dead=False, param=False):
    """`min sum r_i^2` with `r_i == y_i - a - b x_i`: the estimation
    fixture the covariance/information tests use, optionally tagged
    with a `scaling_factor` Suffix, optionally carrying an inert fixed
    variable, optionally with a declared sensitivity parameter.

    `scale` is the `(objective, constraint)` pair; `x_scale` is a
    `{'a': 4.0, 'b': 1e-2, 'r': 25.0}`-shaped mapping of per-variable
    factors, which needs `scale` set too (the Suffix is created there).
    """
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, len(x) - 1)
    m.a = pyo.Var(initialize=0.0)
    m.b = pyo.Var(initialize=0.0)
    m.r = pyo.Var(m.I, initialize=0.0)
    if param:
        m.shift = pyo.Param(initialize=0.0, mutable=True)
        m.res = pyo.Constraint(
            m.I, rule=lambda mm, i: mm.r[i] == float(y[i]) + mm.shift - mm.a
            - mm.b * float(x[i]))
    else:
        m.res = pyo.Constraint(
            m.I, rule=lambda mm, i: mm.r[i] == float(y[i]) - mm.a
            - mm.b * float(x[i]))
    if dead:
        # A fixed variable sorted before the fitted block: the
        # composition that shifts full-x rows against factor rows
        # (gh #450). Scaling must not disturb it.
        m.dead = pyo.Var(bounds=(2.0, 2.0), initialize=2.0)
        m.deadcon = pyo.Constraint(expr=m.dead * m.dead <= 1e6)
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    declare_sens_fitted(m.a)
    declare_sens_fitted(m.b)
    declare_sens_residual(m.r)
    if param:
        declare_sens_param(m.shift)
    if scale is not None:
        obj_s, con_s = scale
        m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.scaling_factor[m.obj] = obj_s
        for i in m.I:
            m.scaling_factor[m.res[i]] = con_s
        for name, factor in (x_scale or {}).items():
            comp = getattr(m, name)
            for vd in (comp.values() if comp.is_indexed() else [comp]):
                m.scaling_factor[vd] = factor
    return m


def solve_pair(x_scale=None, **kwargs):
    """The same estimation model solved twice — user scaling engaged,
    and no scaling at all — so every accessor can be compared against
    unscaled ground truth. `x_scale` adds a change of variables to the
    scaled arm (gh #486 stage 3); the unscaled arm never gets one, so
    every comparison below is scaled-against-plain, never
    scaled-against-scaled."""
    x, y, X = linear_data()
    scaled = linear_model(x, y, scale=(1e-3, 50.0), x_scale=x_scale, **kwargs)
    pyo.SolverFactory("pounce").solve(scaled, options=dict(USER_SCALING))
    plain = linear_model(x, y, **kwargs)
    pyo.SolverFactory("pounce").solve(plain)
    return scaled, plain, X


def session_scaling(m):
    return m.__dict__["_pounce_sens"].session.solver.nlp_scaling


def test_in_process_variable_factor_is_inert_without_the_option():
    # The in-process path reads the Suffix itself rather than leaving it
    # to the writer, so it has its own chance to raise where it should
    # not: a `core.scale_model`-style Suffix must not fail a plain solve.
    x, y, _ = linear_data()
    m = linear_model(x, y, scale=(1e-3, 50.0))
    m.scaling_factor[m.a] = 4.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pyo.SolverFactory("pounce").solve(m)


def test_in_process_path_installs_the_user_scaling():
    """Guard for the wiring itself: without it the engine reports the
    identity scaling and every comparison below passes vacuously."""
    scaled, plain, _ = solve_pair()
    assert float(session_scaling(scaled)["obj"]) == pytest.approx(1e-3)
    assert float(session_scaling(plain)["obj"]) == pytest.approx(1.0)


def test_estimates_are_unmoved_by_user_scaling():
    scaled, plain, X = solve_pair()
    for name in ("a", "b"):
        assert pyo.value(getattr(scaled, name)) == pytest.approx(
            pyo.value(getattr(plain, name)), rel=1e-7)


def test_covariance_matches_unscaled_ground_truth():
    scaled, plain, X = solve_pair()
    np.testing.assert_allclose(
        sens_covariance(scaled, sigma_sq=SIGMA**2).matrix,
        sens_covariance(plain, sigma_sq=SIGMA**2).matrix, rtol=1e-7)
    # and against the closed form, so both runs being wrong the same
    # way cannot pass
    np.testing.assert_allclose(
        sens_covariance(scaled, sigma_sq=SIGMA**2).matrix,
        SIGMA**2 * np.linalg.inv(X.T @ X), rtol=1e-6)


def test_information_matches_unscaled_ground_truth():
    scaled, plain, X = solve_pair()
    np.testing.assert_allclose(sens_information(scaled).matrix,
                               sens_information(plain).matrix, rtol=1e-7)
    np.testing.assert_allclose(sens_information(scaled).matrix,
                               2.0 * X.T @ X, rtol=1e-6)
    np.testing.assert_allclose(
        sens_information(scaled, hessian="gauss-newton").matrix,
        2.0 * X.T @ X, rtol=1e-6)


def test_of_blocks_match_unscaled_ground_truth():
    scaled, plain, X = solve_pair()
    C = SIGMA**2 * np.linalg.inv(X.T @ X)
    cov_a = sens_covariance(scaled, sigma_sq=SIGMA**2, of=[scaled.a])
    assert cov_a[scaled.a] == pytest.approx(C[0, 0], rel=1e-6)
    # the rank-deficient residual block: the prediction band
    H = X @ np.linalg.solve(X.T @ X, X.T)
    np.testing.assert_allclose(
        sens_covariance(scaled, sigma_sq=SIGMA**2, of=scaled.r).matrix,
        SIGMA**2 * H, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(
        sens_information(scaled, of=[scaled.a, scaled.b]).matrix,
        sens_information(plain, of=[plain.a, plain.b]).matrix, rtol=1e-7)


def test_classifier_statuses_match_unscaled():
    scaled, plain, _ = solve_pair()
    assert (sens_covariance(scaled, sigma_sq=SIGMA**2).conditioned_on
            == sens_covariance(plain, sigma_sq=SIGMA**2).conditioned_on)


def test_sens_jacobian_and_sens_solution_match_unscaled():
    scaled, plain, _ = solve_pair(param=True)
    for target in ("a", "b"):
        g_scaled = sens_jacobian(getattr(scaled, target), wrt=scaled.shift)
        g_plain = sens_jacobian(getattr(plain, target), wrt=plain.shift)
        assert g_scaled == pytest.approx(g_plain, abs=1e-6)
    est_scaled = sens_solution(scaled, [(scaled.shift, 0.25)])
    est_plain = sens_solution(plain, [(plain.shift, 0.25)])
    assert est_scaled[scaled.a] == pytest.approx(est_plain[plain.a], abs=1e-6)
    assert est_scaled[scaled.b] == pytest.approx(est_plain[plain.b], abs=1e-6)


def test_fixed_variable_composition_survives_user_scaling():
    scaled, plain, X = solve_pair(dead=True)
    np.testing.assert_allclose(sens_information(scaled).matrix,
                               2.0 * X.T @ X, rtol=1e-6)
    np.testing.assert_allclose(sens_information(scaled).matrix,
                               sens_information(plain).matrix, rtol=1e-7)


def test_retain_only_block_matches_unscaled():
    x, y, X = linear_data()

    def build(scaled):
        m = pyo.ConcreteModel()
        m.I = pyo.RangeSet(0, len(x) - 1)
        m.a = pyo.Var(initialize=0.0)
        m.b = pyo.Var(initialize=0.0)
        m.r = pyo.Var(m.I, initialize=0.0)
        m.res = pyo.Constraint(
            m.I, rule=lambda mm, i: mm.r[i] == float(y[i]) - mm.a
            - mm.b * float(x[i]))
        m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
        sens_retain_kkt(m)          # no declarations: the retain-only path
        if scaled:
            m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
            m.scaling_factor[m.obj] = 1e-3
            for i in m.I:
                m.scaling_factor[m.res[i]] = 50.0
            pyo.SolverFactory("pounce").solve(m, options=dict(USER_SCALING))
        else:
            pyo.SolverFactory("pounce").solve(m)
        return m

    scaled, plain = build(True), build(False)
    try:
        np.testing.assert_allclose(
            sens_information(scaled, of=[scaled.a, scaled.b]).matrix,
            sens_information(plain, of=[plain.a, plain.b]).matrix, rtol=1e-7)
        np.testing.assert_allclose(
            sens_information(scaled, of=[scaled.a, scaled.b]).matrix,
            2.0 * X.T @ X, rtol=1e-6)
    finally:
        sens_release_kkt(scaled)
        sens_release_kkt(plain)


# ── variable factors on the sensitivity path (gh #486 stage 3) ───────────────

#: A change of variables spanning six orders of magnitude, and not
#: uniform: `a` and `b` are the fitted parameters the covariance is
#: about, `r` the residual block. A uniform factor would cancel out of
#: several of the quantities below and prove less.
X_SCALE = {"a": 1e3, "b": 1e-3, "r": 20.0}


def test_the_in_process_path_installs_the_variable_factors():
    """Guard for the wiring, the same role
    `test_in_process_path_installs_the_user_scaling` plays for the row
    factors: without it every comparison below passes vacuously, on a
    solve that quietly applied no change of variables at all."""
    scaled, plain, _ = solve_pair(x_scale=X_SCALE)
    d = session_scaling(scaled)["x_scaling"]
    assert d is not None, "the solve applied no change of variables"
    assert session_scaling(plain)["x_scaling"] is None
    # By name, not by position: the writer chooses the column order,
    # and a factor landing on the wrong column is exactly the failure
    # this guard is here to catch.
    cols = scaled.__dict__["_pounce_sens"].session.var_names
    by_name = dict(zip(cols, np.asarray(d).tolist()))
    assert by_name["a"] == pytest.approx(1e3)
    assert by_name["b"] == pytest.approx(1e-3)
    assert by_name["r[0]"] == pytest.approx(20.0)


def test_estimates_are_unmoved_by_variable_scaling():
    scaled, plain, _ = solve_pair(x_scale=X_SCALE)
    for name in ("a", "b"):
        assert pyo.value(getattr(scaled, name)) == pytest.approx(
            pyo.value(getattr(plain, name)), rel=1e-6)


def test_covariance_matches_unscaled_ground_truth_under_variable_scaling():
    scaled, plain, X = solve_pair(x_scale=X_SCALE)
    np.testing.assert_allclose(
        sens_covariance(scaled, sigma_sq=SIGMA**2).matrix,
        sens_covariance(plain, sigma_sq=SIGMA**2).matrix, rtol=1e-6)
    # and against the closed form, so both runs being wrong the same
    # way cannot pass
    np.testing.assert_allclose(
        sens_covariance(scaled, sigma_sq=SIGMA**2).matrix,
        SIGMA**2 * np.linalg.inv(X.T @ X), rtol=1e-6)


def test_information_matches_unscaled_ground_truth_under_variable_scaling():
    scaled, plain, X = solve_pair(x_scale=X_SCALE)
    np.testing.assert_allclose(sens_information(scaled).matrix,
                               sens_information(plain).matrix, rtol=1e-6)
    np.testing.assert_allclose(sens_information(scaled).matrix,
                               2.0 * X.T @ X, rtol=1e-6)
    # The Gauss-Newton form is built from the residual Jacobian rather
    # than the KKT factor, so it carries the substitution down its own
    # path and needs asserting separately.
    np.testing.assert_allclose(
        sens_information(scaled, hessian="gauss-newton").matrix,
        2.0 * X.T @ X, rtol=1e-6)


def test_of_blocks_match_unscaled_under_variable_scaling():
    # The residual block is where a per-variable factor is most visible:
    # `r` carries a factor of its own, and the prediction band is a
    # rank-deficient block whose projection would not survive a missed
    # one.
    scaled, plain, X = solve_pair(x_scale=X_SCALE)
    C = SIGMA**2 * np.linalg.inv(X.T @ X)
    assert sens_covariance(scaled, sigma_sq=SIGMA**2,
                      of=[scaled.a])[scaled.a] == pytest.approx(C[0, 0],
                                                                 rel=1e-6)
    H = X @ np.linalg.solve(X.T @ X, X.T)
    np.testing.assert_allclose(
        sens_covariance(scaled, sigma_sq=SIGMA**2, of=scaled.r).matrix,
        SIGMA**2 * H, rtol=1e-5, atol=1e-10)


def test_classifier_statuses_match_unscaled_under_variable_scaling():
    scaled, plain, _ = solve_pair(x_scale=X_SCALE)
    assert (sens_covariance(scaled, sigma_sq=SIGMA**2).conditioned_on
            == sens_covariance(plain, sigma_sq=SIGMA**2).conditioned_on)


def test_sens_jacobian_and_sens_solution_match_unscaled_under_variable_scaling():
    scaled, plain, _ = solve_pair(x_scale=X_SCALE, param=True)
    for target in ("a", "b"):
        g_scaled = sens_jacobian(getattr(scaled, target), wrt=scaled.shift)
        g_plain = sens_jacobian(getattr(plain, target), wrt=plain.shift)
        assert g_scaled == pytest.approx(g_plain, abs=1e-6)
    est_scaled = sens_solution(scaled, [(scaled.shift, 0.25)])
    est_plain = sens_solution(plain, [(plain.shift, 0.25)])
    assert est_scaled[scaled.a] == pytest.approx(est_plain[plain.a], abs=1e-6)
    assert est_scaled[scaled.b] == pytest.approx(est_plain[plain.b], abs=1e-6)


def test_fixed_variable_composition_survives_variable_scaling():
    # A fixed variable is dropped from the solve's columns but not from
    # the Suffix's index space, so the factor vector and the factor's
    # `x` block are different lengths -- the composition gh #450 is
    # about, now with a change of variables on top.
    scaled, plain, X = solve_pair(x_scale=X_SCALE, dead=True)
    np.testing.assert_allclose(sens_information(scaled).matrix,
                               2.0 * X.T @ X, rtol=1e-6)
    np.testing.assert_allclose(sens_information(scaled).matrix,
                               sens_information(plain).matrix, rtol=1e-6)


# ── rows with no Suffix entry (nlp_scaling_untagged_constr) ──────────────────

def _untagged_rows_model():
    """`tagged` carries a factor and `untagged` does not. With `x0`
    scaled by 2, `untagged`'s largest Jacobian entry in the scaled
    columns is 1000 / 2 = 500, so the gradient-based rule gives it
    100 / 500 = 0.2. The declared parameter's pin row has no entry
    either, and its row max of 1 gives it 1."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=0.0, mutable=True)
    m.x0 = pyo.Var(initialize=1.0)
    m.x1 = pyo.Var(initialize=1.0)
    m.x2 = pyo.Var(initialize=1.0)
    m.obj = pyo.Objective(expr=m.x0**2 + m.x1**2 + m.x2**2)
    m.tagged = pyo.Constraint(expr=2 * m.x0 + m.x1 == m.p)
    m.untagged = pyo.Constraint(expr=1000 * m.x0 + m.x1 == 5)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.tagged] = 0.5
    m.scaling_factor[m.x0] = 2.0
    declare_sens_param(m.p)
    return m


@pytest.mark.parametrize("rule, want", [
    ("gradient-based", [0.2, 0.5, 1.0]),
    ("none", [0.5, 1.0, 1.0]),
])
def test_in_process_path_scales_the_rows_with_no_suffix_entry(rule, want):
    """The in-process solve marks the rows the Suffix does not name, as
    the `.nl` reader does, so the option scales the same rows on both
    paths. Without the marks every row would count as given and keep
    1.0, which is the `none` answer."""
    m = _untagged_rows_model()
    s = pyo.SolverFactory("pounce")
    s.options.update(USER_SCALING)
    s.options["nlp_scaling_untagged_constr"] = rule
    res = s.solve(m)
    assert res.solver.termination_condition == TerminationCondition.optimal
    c_scale = session_scaling(m)["c_scale"]
    got = sorted(np.asarray(c_scale).tolist()) if c_scale is not None else [1.0] * 3
    assert got == pytest.approx(want)
