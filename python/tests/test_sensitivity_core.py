"""`pounce.sensitivity` against models built with no modelling layer.

The point of these: every number here used to be reachable only through
`pyomo_pounce`, so the whole surface was untestable without Pyomo
installed and unusable from a `.nl` file, the CLI or CasADi. Nothing in
this file imports Pyomo, and `test_the_core_does_not_import_pyomo`
asserts that stays true.

The fixtures are small enough to have closed-form answers, so these are
checks against arithmetic done by hand rather than against a previous
run of the same code.
"""
import numpy as np
import pytest

import pounce
from pounce.sensitivity import (
    active_set_changes,
    covariance,
    information,
    solution,
    solution_report,
    solve_for_sensitivity,
)

P0 = 2.0


def parametric_model(fixed_variable=False, maximize=False):
    """min (x - p)^2 + 3 p^2  s.t.  x + y == 5,  p == P0.

    With `p` held at P0 the optimum is x = P0, y = 5 - P0, so
    dx/dp = 1, dy/dp = -1 and df/dp = 6 p -- the example from gh#878,
    where a chain-rule-only reading of df/dp returns 0 because every
    df/dx_i vanishes at the optimum.

    `fixed_variable` inserts an equal-bounds variable AHEAD of `p`, so
    the solve removes it and full-x stops agreeing with the factor's
    var-x from that column on (gh#450).

    `maximize` states the SAME model the other way round -- maximize the
    negation -- which reaches the same point and negates every objective
    quantity and nothing else. `build_nl_problem(minimize=False)` does
    to the callbacks exactly what `read_nl` does to a `maximize`
    objective: it negates them and records the fact in `nl.minimize`,
    so this is the sense conversion's fixture and not a special case of
    its own.
    """
    sgn = -1.0 if maximize else 1.0
    if fixed_variable:
        v = pounce.NlExpr.vars(4)                   # x, y, z, p
        return pounce.build_nl_problem(
            n=4,
            objective=sgn * ((v[0] - v[3]) ** 2 + 3.0 * v[3] ** 2
                             + v[2] ** 2),
            minimize=not maximize,
            constraints=[v[0] + v[1], v[3]],
            g_l=[5.0, P0], g_u=[5.0, P0],
            x_l=[-50.0, -50.0, 1.0, -100.0],
            x_u=[10.0, 50.0, 1.0, 100.0],
            x0=[0.0, 0.0, 1.0, P0],
            var_names=["x", "y", "z", "p"],
            con_names=["c1", "pin_p"],
        )
    v = pounce.NlExpr.vars(3)                       # x, y, p
    return pounce.build_nl_problem(
        n=3,
        objective=sgn * ((v[0] - v[2]) ** 2 + 3.0 * v[2] ** 2),
        minimize=not maximize,
        constraints=[v[0] + v[1], v[2]],
        g_l=[5.0, P0], g_u=[5.0, P0],
        x_l=[-50.0, -50.0, -100.0], x_u=[10.0, 50.0, 100.0],
        x0=[0.0, 0.0, P0],
        var_names=["x", "y", "p"], con_names=["c1", "pin_p"],
    )


def parametric_session(**kw):
    return solve_for_sensitivity(parametric_model(**kw), pins={"p": 1},
                                 options={"print_level": 0})


# ── the step ─────────────────────────────────────────────────────────────────

def test_the_step_matches_the_analytic_derivative():
    sess = parametric_session()
    np.testing.assert_allclose(sess.base_x, [P0, 5.0 - P0, P0], atol=1e-8)

    delta = 0.1
    x_new = solution(sess, [1], [delta])
    np.testing.assert_allclose(
        x_new, [P0 + delta, 5.0 - P0 - delta, P0 + delta], atol=1e-8)


def test_the_total_objective_derivative_carries_the_explicit_partial():
    """df/dp = 6p, not the 0 a chain-rule-only reading gives (gh#878)."""
    sess = parametric_session()
    df_dp = sess.total_objective_derivative(sess.column(1))
    assert df_dp == pytest.approx(6.0 * P0, abs=1e-7)
    # every df/dx_i IS zero at the optimum, which is what makes the
    # explicit partial the whole answer here
    step = sess.scatter_x(np.asarray(
        pounce.Solver.parametric_step(sess.solver, [1], [1.0])))
    assert sess.objective_gradient()[:2] @ step[:2] == pytest.approx(0.0,
                                                                    abs=1e-7)


@pytest.mark.parametrize("maximize,sgn", [(False, 1.0), (True, -1.0)])
def test_the_objective_quantities_are_stated_in_the_models_own_sense(
        maximize, sgn):
    """`objective_gradient`, `total_objective_derivative` and
    `base_obj` describe the objective the MODEL states, not the one the
    engine minimized.

    `build_nl_problem(minimize=False)` -- like `read_nl` on a `maximize`
    objective -- negates the callbacks before the engine sees them, so
    everything the engine reports about the objective is stated against
    `-f`. `objective_sign` is the conversion back, and until it existed
    nothing applied it: `df/dp` came back at the right magnitude and the
    wrong sign, silently, on every maximization.

    The factor is +1 on a minimization, which is the whole reason this
    sat undetected -- a corpus with no maximization in it cannot tell a
    right sign from no sign at all.

    Closed form on this model, both spellings: f* = sgn * 3 p^2 = 12 sgn
    and df*/dp = sgn * 6 p = 12 sgn at p = P0 = 2.
    """
    sess = solve_for_sensitivity(parametric_model(maximize=maximize),
                                 pins={"p": 1}, options={"print_level": 0})
    assert sess.base_obj == pytest.approx(sgn * 3.0 * P0 ** 2, abs=1e-7)
    assert sess.total_objective_derivative(sess.column(1)) == pytest.approx(
        sgn * 6.0 * P0, abs=1e-7)
    # the same point either way: a maximization does not move the
    # optimum, so dx/dp knows nothing about the sense
    np.testing.assert_allclose(sess.base_x, [P0, 5.0 - P0, P0], atol=1e-8)


def test_the_two_spellings_of_one_model_agree_up_to_the_sense():
    """The paired form, which holds whatever the closed form is: `min f`
    and `max -f` are one model, so their optima coincide exactly and
    every objective quantity negates."""
    lo = solve_for_sensitivity(parametric_model(), pins={"p": 1},
                               options={"print_level": 0})
    hi = solve_for_sensitivity(parametric_model(maximize=True),
                               pins={"p": 1}, options={"print_level": 0})
    np.testing.assert_allclose(lo.base_x, hi.base_x, atol=1e-8)
    assert lo.base_obj == pytest.approx(-hi.base_obj, abs=1e-7)
    np.testing.assert_allclose(lo.objective_gradient(),
                               -np.asarray(hi.objective_gradient()), atol=1e-7)
    assert lo.total_objective_derivative(lo.column(1)) == pytest.approx(
        -hi.total_objective_derivative(hi.column(1)), abs=1e-7)


def test_results_are_keyed_by_variable_name_with_no_modelling_layer():
    """`crossed` comes back keyed by `.col` name, not by a component."""
    sess = parametric_session()
    # p driven to 12 pushes x past its upper bound of 10
    rep = solution_report(sess, [1], [10.0], mode="linear")
    assert isinstance(rep.crossed, dict)
    assert list(rep.crossed) == ["x"], "x is the only variable that can bind"
    assert all(isinstance(k, str) for k in rep.crossed)


def test_active_set_changes_names_the_variable_that_moves():
    sess = parametric_session()
    changes = active_set_changes(sess, [1], [10.0])
    assert changes, "driving p to 12 must push x onto its upper bound"
    assert [c.var for c in changes] == ["x"]
    assert [(c.bound, c.action) for c in changes] == [("upper", "reaches")]


# ── the index spaces (gh#450) ────────────────────────────────────────────────

def test_a_fixed_variable_makes_full_x_and_var_x_diverge():
    """The precondition for the next test: without it the two spaces
    coincide and reading one as the other cannot be caught."""
    sess = parametric_session(fixed_variable=True)
    assert sess._primal_row_map() == [0, 1, None, 2], (
        "the solve should have removed z, shifting p's factor row")
    with pytest.raises(ValueError, match="fixed variable"):
        sess.primal_row(2, "test")


def test_the_step_is_right_across_a_removed_variable():
    """Reading a full-x index as a factor row returns a NEIGHBOURING
    variable's sensitivity -- plausible and wrong. p sits one column
    past the removed z, so that is exactly what this would show."""
    sess = parametric_session(fixed_variable=True)
    delta = 0.1
    x_new = solution(sess, [1], [delta])
    np.testing.assert_allclose(
        x_new, [P0 + delta, 5.0 - P0 - delta, 1.0, P0 + delta], atol=1e-8)
    assert sess.total_objective_derivative(
        sess.column(1)) == pytest.approx(6.0 * P0, abs=1e-7)


# ── the statistics ───────────────────────────────────────────────────────────

T = np.array([1.0, 2.0, 3.0])
Y = np.array([2.1, 3.9, 6.2])


def estimation_session(maximize=False, declare_residuals=True):
    """Fit y = a t to three points, residuals carried as variables.

    Ordinary linear least squares, so a-hat, the residual variance and
    var(a) all have closed forms to check against.

    `maximize` spells the same fit as `max -SSR`, which is what a
    maximum-likelihood formulation looks like written directly.
    `declare_residuals=False` withholds `res_rows`, which is the only
    way to reach `covariance`'s `n_data=` fallback: declared residuals
    take precedence over it.
    """
    sgn = -1.0 if maximize else 1.0
    v = pounce.NlExpr.vars(4)                        # a, r0, r1, r2
    nl = pounce.build_nl_problem(
        n=4,
        objective=sgn * pounce.NlExpr.sum([v[1] ** 2, v[2] ** 2, v[3] ** 2]),
        minimize=not maximize,
        constraints=[v[1 + i] - v[0] * float(T[i]) for i in range(3)],
        g_l=[-y for y in Y], g_u=[-y for y in Y],
        x_l=[-50.0] * 4, x_u=[50.0] * 4,
        x0=[1.0, 0.0, 0.0, 0.0],
        var_names=["a", "r[0]", "r[1]", "r[2]"],
        con_names=["res0", "res1", "res2"],
    )
    return solve_for_sensitivity(
        nl, fit_rows={"a": 0},
        res_rows={None: [1, 2, 3]} if declare_residuals else None,
        options={"print_level": 0})


def test_covariance_matches_the_least_squares_closed_form():
    sess = estimation_session()
    a_hat = float(T @ Y) / float(T @ T)
    assert sess.base_x[0] == pytest.approx(a_hat, abs=1e-9)

    resid = T * sess.base_x[0] - Y
    sigma_sq = float(resid @ resid) / (len(T) - 1)     # n - #fitted
    cov = covariance(sess)
    assert cov["a"] == pytest.approx(sigma_sq / float(T @ T), rel=1e-9)
    assert cov.std_err["a"] == pytest.approx(np.sqrt(cov["a"]), rel=1e-12)
    assert cov.sigma_sq == pytest.approx(sigma_sq, rel=1e-9)


@pytest.mark.parametrize("maximize", [False, True])
def test_the_n_data_noise_estimate_reads_the_objective_as_a_sum_of_squares(
        maximize):
    """The `n_data=` fallback takes SSR from the solve-time objective,
    and "the objective is a sum of squares" is a claim about the
    objective the solver MINIMIZED. `max -SSR` is the same fit spelled
    the other way round, so it must give the same answer.

    This is the neighbour the sense conversion could have broken, and
    did: making `base_obj` state the model's own sense moved this read
    out from under the assumption it depends on, and a maximize spelling
    divided a negative SSR by `n_data - n_fit` and returned NaN standard
    errors. Loud rather than silent, but wrong, and nothing in the
    corpus reached it -- `n_data=` and `maximize` had never met.

    Oracle: with n_data = 3 and one fitted parameter the fallback's
    SSR/(n - p) is exactly the declared-residual estimate, so the two
    routes to sigma^2 must agree to the last digit on the minimize arm,
    and both arms to each other.
    """
    declared = covariance(estimation_session())
    fallback = covariance(estimation_session(maximize=maximize,
                                             declare_residuals=False),
                          n_data=len(T))
    assert np.isfinite(fallback["a"])
    assert fallback.sigma_sq == pytest.approx(declared.sigma_sq, rel=1e-9)
    assert fallback["a"] == pytest.approx(declared["a"], rel=1e-9)


def test_information_is_the_hessian_the_covariance_inverts():
    """`pcov = 2 sigma^2 inv(H_S)`, so H_S is 2 sum(t^2) here."""
    sess = estimation_session()
    assert information(sess)["a"] == pytest.approx(2.0 * float(T @ T),
                                                   rel=1e-9)


def test_an_explicit_block_overrides_the_declared_one():
    sess = estimation_session()
    cov = covariance(sess, ["r[0]"], [1])
    assert cov.params == ["r[0]"]
    assert cov["r[0]"] > 0.0


# ── the caller's own name in diagnostics ─────────────────────────────────────

def test_who_names_the_caller_in_diagnostics():
    sess = parametric_session()
    with pytest.raises(ValueError, match=r"^solution: mode must be"):
        solution(sess, [1], [0.1], mode="nonsense")
    with pytest.raises(ValueError, match=r"^sens_solution: mode must be"):
        solution(sess, [1], [0.1], mode="nonsense", who="sens_solution")


def test_hints_name_the_callers_own_declarations():
    """The message points at how the CALLER declares residuals, because
    a message naming the wrong layer's spelling is worse than none."""
    sess = parametric_session()
    with pytest.raises(RuntimeError, match=r"fit_rows="):
        covariance(sess)
    with pytest.raises(RuntimeError, match=r"declare_sens_fitted\(\)"):
        covariance(sess, hints={"fitted": "declare_sens_fitted()",
                                "residual": "declare_sens_residual()",
                                "residual_group": "..."})


# ── the packaging claim itself ───────────────────────────────────────────────

def test_the_core_does_not_import_pyomo():
    """The reason this package exists. Run in a subprocess so an earlier
    test that imported Pyomo cannot mask a real import here."""
    import subprocess
    import sys

    code = ("import sys, pounce.sensitivity;"
            "assert 'pyomo' not in sys.modules, sorted("
            "m for m in sys.modules if m.startswith('pyomo'));"
            "print('clean')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    assert "clean" in out.stdout


# ── the reduced-curvature refinement in the report (gh#763, gh#804) ──────────
#
# `solution_report` reports an activity class per variable and per row, and
# until this it reported the *cheap* classifier's verdict raw. That classifier
# divides a variable's barrier diagonal by the Hessian's diagonal and a row's
# by the curvature along the row's own gradient, while the multiplier that
# produced it is generated by the REDUCED curvature. The ratio is therefore
# `reduced/diagonal`, which is 1 only when the coordinate is decoupled — so a
# genuine kink whose coordinate is coupled lands in "ambiguous" and stays
# there however tightly the problem is re-solved.
#
# `Solver.reduced_activity` has answered that question for a while and had
# zero callers outside its own tests. These pin that the report now spends it,
# on the ambiguous entries only, and says what it moved.


def _coupled_kink_model(rho):
    """min ½k² + c·k·y + ½y² − A·p·k  s.t.  p = 0,  0 ≤ k ≤ 10,  y free.

    At `p = 0` both the reduced gradient at `k = 0` and its multiplier
    vanish: a kink by construction, at every `rho`. `rho = 1 − c²` is the
    curvature reduced along `k` once `y` re-optimizes, so `rho = 1` is
    decoupled and smaller values are more strongly coupled. The cheap
    classifier's ratio is exactly `rho`, so the kink falls out of the
    `[1e-1, 1e1]` band — into "ambiguous" — once `rho < 1e-1`.

    Same shape as `test_activity.py`'s fixture of the same name, which owns
    the accessor; this file owns the report that calls it.
    """
    a, c = 1.10, np.sqrt(1.0 - rho)
    v = pounce.NlExpr.vars(3)                       # k, y, p
    k, y, pp = v[0], v[1], v[2]
    return pounce.build_nl_problem(
        n=3,
        objective=0.5 * k * k + c * k * y + 0.5 * y * y - a * pp * k,
        constraints=[pp],
        g_l=[0.0], g_u=[0.0],
        x_l=[0.0, -1e19, -1e19], x_u=[10.0, 1e19, 1e19],
        x0=[0.3, 0.0, 0.0],
        var_names=["k", "y", "p"], con_names=["pin_p"],
    )


def _coupled_session(rho):
    return solve_for_sensitivity(
        _coupled_kink_model(rho), pins={"p": 0},
        options={"print_level": 0, "tol": 1e-10})


@pytest.mark.parametrize("rho", [1e-2, 1e-3])
def test_the_report_refines_a_coupled_kink_the_cheap_rule_cannot_call(rho):
    """The gh#763 misreading, in the report a caller actually looks at.

    Same kink at every `rho`; only the coupling changes. The cheap rule
    reports "ambiguous", which is NOT "probably not a kink" — and a caller
    reading the report has no way to tell the difference. The refinement
    certifies it as weakly active, and `refined` says so happened.
    """
    sess = _coupled_session(rho)
    cheap = sess.solver.classify_activity()
    assert cheap["var_status"][0] == "ambiguous", (
        "the fixture must reach the branch this test is about; at this "
        "coupling the cheap ratio is rho and falls out of the band")

    rep = solution_report(sess, [0], [1e-3], mode="linear")
    assert rep.activity["k"] == "weakly_active", (
        "a coupled kink is a kink; reporting 'ambiguous' for it is the "
        "gh#763 inference")
    assert rep.refined["k"] == ("ambiguous", "weakly_active"), (
        "and the report must say the verdict was refined rather than "
        "leaving a caller to guess which rule produced it")


def test_the_refinement_is_off_when_asked_and_then_the_cheap_class_stands():
    """`refine_activity=False` is the opt-out, and it really opts out.

    Without this the parameter could be ignored and nothing would notice:
    the refined answer is the one every other assertion wants.
    """
    sess = _coupled_session(1e-3)
    rep = solution_report(sess, [0], [1e-3], mode="linear",
                          refine_activity=False)
    assert rep.activity["k"] == "ambiguous"
    assert rep.refined == {}


def test_the_refinement_leaves_an_unambiguous_verdict_alone():
    """It is spent on the ambiguous entries and nothing else.

    A decoupled kink (`rho = 1`) is already certified by the cheap rule, so
    the refinement must not run for it — `refined` empty is the observable
    form of "no back-solve was spent here".
    """
    sess = _coupled_session(1.0)
    cheap = sess.solver.classify_activity()
    assert cheap["var_status"][0] == "weakly_active", (
        "at rho = 1 the coordinate is decoupled and the cheap rule is exact")
    rep = solution_report(sess, [0], [1e-3], mode="linear")
    assert rep.activity["k"] == "weakly_active"
    assert rep.refined == {}, "nothing was ambiguous, so nothing was spent"


def test_the_refinement_is_spent_only_on_the_ambiguous_entries():
    """The cost contract, which no verdict assertion can see.

    Found by mutation: replacing the ambiguous-only index list with
    `range(n)` — refining every entry — left every other test in this file
    green, because the reduced verdict AGREES with the cheap one wherever
    the cheap one was already certain. So `refined` stays empty and nothing
    observes the extra work.

    That work is the reason CLAUDE.md specifies "on demand, over the
    ambiguous entries, never the default": the reduced normalizer is the
    reciprocal diagonal of an inverse, so classifying every bounded
    variable that way is `n` back-solves for a question almost none of them
    raise. This asserts the indices actually asked for, which is the only
    place that cost is visible.
    """
    sess = _coupled_session(1e-3)
    cheap = sess.solver.classify_activity()
    ambiguous = [i for i, st in enumerate(cheap["var_status"])
                 if st == "ambiguous"]
    assert ambiguous, "the fixture must have something to refine"
    assert len(ambiguous) < len(cheap["var_status"]), (
        "and something NOT to refine, or asking for everything and asking "
        "for the ambiguous ones would be the same call")

    asked = []
    real = sess.solver

    class _Spy:
        # The Rust-backed Solver forbids attribute assignment, so the spy
        # wraps it rather than patching it. Everything else forwards.
        def __getattr__(self, name):
            return getattr(real, name)

        def reduced_activity(self, idx):
            asked.append(list(idx))
            return real.reduced_activity(idx)

    sess.solver = _Spy()
    try:
        solution_report(sess, [0], [1e-3], mode="linear")
    finally:
        sess.solver = real

    assert asked == [ambiguous], (
        f"the refinement must ask for the ambiguous entries and no others; "
        f"asked for {asked}, ambiguous are {ambiguous}")


# ---------------------------------------------------------------------
# The refinement's ROW branch, and what it does to the ratio test.
#
# Round 5 of #889: `_refine_ambiguous` branches on variables and on rows,
# and every test above reaches only the variable one. The row branch's
# exposure is not the accessor (`test_activity.py` owns that) but the
# *wiring* — that `row_names`, `classify_activity`'s `row_status` and
# `reduced_row_activity`'s user-space row indices are the same space in
# the same order. Reading one as another returns a neighbouring row's
# answer: plausible, and wrong.
# ---------------------------------------------------------------------


def _coupled_row_model(rho):
    """`CoupledKinkRow` from `test_activity.py`, as an `NlExpr` model.

    The same kink as `_coupled_kink_model`, held by an inequality **row**
    (`2k >= 0`) instead of a bound on `k`. `classify_activity` divides a
    row's weight by the curvature along the row's own gradient — a real
    directional curvature, but not the *reduced* one — so its ratio is
    `rho` and the kink falls out of the band into "ambiguous" once
    `rho < 1e-1`. gh#804.
    """
    a, c = 1.10, np.sqrt(1.0 - rho)
    v = pounce.NlExpr.vars(3)                       # k, y, p
    k, y, pp = v[0], v[1], v[2]
    return pounce.build_nl_problem(
        n=3,
        objective=0.5 * k * k + c * k * y + 0.5 * y * y - a * pp * k,
        constraints=[pp, 2.0 * k],
        g_l=[0.0, 0.0], g_u=[0.0, 1e19],
        x_l=[-1e19] * 3, x_u=[1e19] * 3,
        x0=[0.3, 0.0, 0.0],
        var_names=["k", "y", "p"], con_names=["pin_p", "kink_row"],
    )


@pytest.mark.parametrize("rho", [1e-2, 1e-3])
def test_the_report_refines_a_coupled_ROW_kink_too(rho):
    """gh#804's misreading, through the report, on the row path.

    The variable branch above and this one are separate code paths in
    `_refine_ambiguous` reading separate index spaces. A row index handed
    to `reduced_row_activity` in the wrong space would come back with a
    real, plausible, wrong row's verdict — so this asserts the *name* that
    moved, not just that something did.
    """
    sess = solve_for_sensitivity(
        _coupled_row_model(rho), pins={"p": 0},
        options={"print_level": 0, "tol": 1e-10})

    cheap = sess.solver.classify_activity()
    assert cheap["row_status"][1] == "ambiguous", (
        "the fixture must reach the branch this test is about")
    assert cheap["row_ratio"][1] == pytest.approx(rho, rel=1e-3)

    rep = solution_report(sess, [0], [1e-3], mode="linear")
    assert rep.refined == {"kink_row": ("ambiguous", "weakly_active")}, (
        "the row branch must refine the coupled row kink, and must name "
        "the row it refined — a wrong index space names a neighbour")

    off = solution_report(sess, [0], [1e-3], mode="linear",
                          refine_activity=False)
    assert off.refined == {}, "and the opt-out must reach the row branch too"


def test_refinement_can_only_lengthen_alpha_never_shorten_it():
    """The ratio-test consequence of the default, and its direction.

    Refinement only ever rewrites `"ambiguous"` entries, and `_AT_BOUND`
    does not contain `"ambiguous"` — so it can only **add** coordinates to
    `on_bound`, never remove one. `_ratio_test` *excludes* what is on its
    bound, for the reason its own docstring gives: at an active bound the
    remaining gap is the slack the barrier leaves and the step component
    is the same size, so their quotient is meaningless and would become
    the minimum on any model carrying an active bound.

    So the refinement removes spurious candidates, and `alpha` can only
    get **longer**. Round 5 of #889 raised this property and read the
    direction as "can only shorten"; measured, it is the other way, and
    the exclusion is the point of the mechanism rather than a side effect.
    Pinned here because a default-on behaviour change whose consequence
    nothing asserts is one nobody can review.
    """
    from pounce.sensitivity._step import _AT_BOUND, _ratio_test

    assert "ambiguous" not in _AT_BOUND, (
        "refinement rewrites only ambiguous entries, so this is what makes "
        "the change one-directional")

    # A step that does NOT carry the coordinate past its bound, so the
    # 'always score a crossing' escape hatch does not apply and the
    # exclusion is what decides.
    base, step = np.array([0.0]), np.array([0.5])
    lo, hi, tol = np.array([-10.0]), np.array([1.0]), np.array([1e-9])
    a_amb, first_amb = _ratio_test(base, step, lo, hi, ["r"],
                                   on_bound=np.array([False]), tol=tol)
    a_ref, first_ref = _ratio_test(base, step, lo, hi, ["r"],
                                   on_bound=np.array([True]), tol=tol)

    assert (a_amb, first_amb) == (2.0, "r")
    assert (a_ref, first_ref) == (float("inf"), None)
    assert a_ref >= a_amb, "refining a coordinate onto its bound cannot shorten alpha"
