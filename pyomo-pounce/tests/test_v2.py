"""The `pyomo.contrib.solver` (v2) interface — gh #558.

Two things are under test here, and they pull in different directions.

1. **The v2 interface reaches POUNCE at all**, and carries `pyomo-pounce`'s
   extras onto the v2 lifecycle: the integer guard, the `scaling_factor`
   Suffix check, bundled-binary resolution, and the sensitivity route.
2. **It agrees with the legacy interface.** These are two genuinely
   different Pyomo code paths to one solver, so "both work" is not
   enough — a user moving between them must get the same numbers, with
   the same sign conventions on duals and reduced costs. Several tests
   below therefore solve the *same* model both ways and compare, which
   is also what pins the two ASL-compatibility behaviours from #553
   (quoted option values, and the per-model `.sol` `Options` echo) from
   the Python side: the v2 route exercises both on every solve and fails
   loudly if either regresses.
"""

import pytest

import pyomo_pounce  # noqa: F401  (registers both interfaces)

# The v2 interface is optional -- it needs Pyomo >= 6.10.1, while the
# package as a whole supports pyomo>=6.0 through the legacy plugin. On an
# older Pyomo there is nothing here to test, and that is not a failure.
#
# The import above is deliberately NOT wrapped in a try/skip. `import
# pyomo_pounce` raising is never an "old Pyomo" condition to skip past --
# it is the regression that takes SolverFactory('pounce') down with it
# (the v2 probe on `pyomo.contrib.solver.common` did exactly that on
# 6.9.2-6.10.0), and it has to fail loudly. `HAVE_V2_INTERFACE` is the
# only thing that may legitimately be False here. What actually
# exercises this guard is the below-floor CI leg in ci.yml, which
# installs a Pyomo older than the floor and asserts the import still
# works and the flag is False.
if not pyomo_pounce.HAVE_V2_INTERFACE:  # pragma: no cover
    pytest.skip(
        "the v2 interface needs Pyomo >= 6.10.1", allow_module_level=True)

from pyomo.contrib.solver.common.factory import (  # noqa: E402
    SolverFactory as SolverFactoryV2,
)
from pyomo.contrib.solver.common.results import (  # noqa: E402
    SolutionStatus,
    TerminationCondition,
)
from pyomo.environ import (  # noqa: E402
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    Param,
    SolverFactory,
    Suffix,
    Var,
    maximize,
    minimize,
    value,
)

from pyomo_pounce.v2 import LegacyPounceSolver, Pounce  # noqa: E402


@pytest.fixture(scope="module")
def v2():
    s = SolverFactoryV2("pounce")
    if not s.available():
        pytest.skip("pounce binary not found")
    # The duals/reduced costs compared below are only meaningful when the
    # variables carrying them survive to the `.nl`; Pyomo's linear presolve
    # eliminates them and (correctly) warns that the values may be wrong.
    # The legacy interface does no such presolve, so leaving it on would
    # make the cross-interface comparisons compare different problems.
    s.config.writer_config.linear_presolve = False
    return s


def _model(with_bounds=True, sense=minimize):
    """A small NLP with an active constraint and an active bound, so that
    duals and reduced costs are both non-trivial.

    `sense=maximize` states the same model the other way round --
    maximize the negation -- so it reaches the same point and negates
    every multiplier. It is the only shape that can see an
    objective-sense conversion, since the factor is +1 on a
    minimization.
    """
    m = ConcreteModel()
    m.p = Param(initialize=2.0, mutable=True)
    m.x = Var(initialize=0.5, bounds=(0.9, 10) if with_bounds else None)
    m.y = Var(initialize=0.5, bounds=(0.1, 10) if with_bounds else None)
    m.c = Constraint(expr=m.x + 2 * m.y == m.p)
    f = (m.x - 2) ** 2 + (m.y - 3) ** 2
    m.o = Objective(expr=-f if sense == maximize else f, sense=sense)
    m.dual = Suffix(direction=Suffix.IMPORT)
    m.rc = Suffix(direction=Suffix.IMPORT)
    return m


# ---------------------------------------------------------------------------
# registration: additive, and it must not disturb the legacy plugin
# ---------------------------------------------------------------------------


def test_v2_factory_returns_the_v2_class():
    assert isinstance(SolverFactoryV2("pounce"), Pounce)


def test_legacy_pounce_is_unchanged():
    """`SolverFactory('pounce')` is the documented route and stays on the
    legacy ASL plugin — registering the v2 interface must not silently
    move existing users onto a different code path."""
    from pyomo_pounce.pounce_solver import POUNCE

    assert isinstance(SolverFactory("pounce"), POUNCE)


def test_pounce_v2_registered_with_the_legacy_factory():
    """The v2 engine is also reachable through the legacy API, under the
    `pounce_v2` name — the same split Pyomo uses for `ipopt` / `ipopt_v2`."""
    assert isinstance(SolverFactory("pounce_v2"), LegacyPounceSolver)


# ---------------------------------------------------------------------------
# version / availability
# ---------------------------------------------------------------------------


def test_version_is_parsed(v2):
    """Pyomo's `Ipopt._get_version` requires a banner starting `ipopt`, so
    POUNCE reads as unavailable through it. The override must parse
    `pounce X.Y.Z`; without it `version()` is None and `available()` is
    NotFound. Solves still run — nothing gates on the version — so the
    damage is that every availability check lies, including the
    `if not solver.available(): skip` guard this suite's own fixture
    uses, which would silently skip the whole file."""
    ver = v2.version()
    assert isinstance(ver, tuple) and len(ver) >= 2
    assert all(isinstance(i, int) for i in ver)


def test_version_rejects_a_non_pounce_executable(tmp_path):
    """An executable that does not announce itself as `pounce` must not be
    driven as POUNCE — the same strictness Pyomo applies to Ipopt, and for
    the same reason: silently driving some other ASL binary produces
    plausible numbers from the wrong solver."""
    import stat

    fake = tmp_path / "notpounce"
    fake.write_text("#!/bin/sh\necho 'ipopt 3.14.20'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    s = Pounce(executable=str(fake))
    assert s.version() is None


# ---------------------------------------------------------------------------
# the extras, on the v2 lifecycle
# ---------------------------------------------------------------------------


def test_integer_guard_fires_on_the_v2_route(v2):
    """gh #341 on the v2 route. POUNCE has no branch-and-bound; without
    this guard the v2 interface would solve the continuous relaxation and
    report a fractional value as optimal — exactly the silent wrongness
    the legacy plugin already refuses."""
    m = _model()
    m.b = Var(domain=Binary, initialize=0)
    m.cb = Constraint(expr=m.x >= m.b)
    with pytest.raises(ValueError, match="no branch-and-bound"):
        v2.solve(m)


def test_integer_guard_ignores_fixed_discrete_vars(v2):
    """A *fixed* discrete variable is not a decision to relax, so it must
    not trip the guard."""
    m = _model()
    m.b = Var(domain=Binary, initialize=1)
    m.b.fix(1)
    v2.solve(m)  # must not raise


def test_scaling_suffix_request_without_a_suffix_warns(v2):
    """gh #483: asking for user scaling on a model that declares none used
    to be accepted silently and mean "none"."""
    m = _model()
    with pytest.warns(UserWarning):
        v2.solve(m, solver_options={"nlp_scaling_method": "user-scaling"})


def test_executable_defaults_to_the_bundled_binary():
    """gh #315: PATH can hand back a stale build reporting an identical
    version string, so the bundled binary wins when one is installed."""
    from pyomo_pounce.pounce_solver import _bundled_path

    bundled = _bundled_path()
    if bundled is None:
        pytest.skip("no wheel-bundled binary in this environment")
    assert Pounce().config.executable.path() == bundled


# ---------------------------------------------------------------------------
# the two interfaces must agree
# ---------------------------------------------------------------------------


def test_v2_matches_legacy_primals_and_duals(v2):
    """Same model, both interfaces, same answer — including the sign
    convention on duals and reduced costs.

    This is the check the interface table in `docs/src/pyomo.md` was
    making by hand, and it also exercises both #553 fixes end-to-end:
    the v2 route passes options as a single quoted `argv` entry and reads
    the `.sol` with the strict `Options` reader.
    """
    legacy_solver = SolverFactory("pounce")
    if not legacy_solver.available(exception_flag=False):
        pytest.skip("pounce binary not found")

    ml = _model()
    # The legacy ASL path does not synthesize an `rc` suffix -- combining
    # the two bound multipliers into one reduced cost is something the v2
    # solution loader does. So read the raw suffixes on this side and do
    # the combination by hand, which pins the sign convention explicitly
    # rather than comparing two numbers that happen to agree.
    ml.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    ml.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    legacy_solver.solve(ml)
    mv = _model()
    results = v2.solve(mv)

    assert results.solution_status is SolutionStatus.optimal
    assert results.termination_condition is (
        TerminationCondition.convergenceCriteriaSatisfied)

    assert value(mv.x) == pytest.approx(value(ml.x), rel=1e-8, abs=1e-9)
    assert value(mv.y) == pytest.approx(value(ml.y), rel=1e-8, abs=1e-9)
    assert results.incumbent_objective == pytest.approx(
        value(ml.o), rel=1e-8, abs=1e-9)

    assert mv.dual[mv.c] == pytest.approx(ml.dual[ml.c], rel=1e-6, abs=1e-8)

    # `x` is at its lower bound, so this is a live multiplier, not a zero
    # that would compare equal by accident.
    zl = ml.ipopt_zL_out.get(ml.x, 0.0)
    zu = ml.ipopt_zU_out.get(ml.x, 0.0)
    assert zl > 1e-3
    expected_rc = zu if abs(zu) > abs(zl) else zl
    assert mv.rc[mv.x] == pytest.approx(expected_rc, rel=1e-6, abs=1e-8)


def test_solver_options_reach_the_binary(v2):
    """Options must actually take effect, not merely be accepted. A
    one-iteration cap has to stop the solve."""
    m = _model()
    results = v2.solve(
        m, solver_options={"max_iter": 1},
        raise_exception_on_nonoptimal_result=False,
        load_solutions=False)
    assert results.termination_condition is TerminationCondition.iterationLimit


def test_instance_level_options_reach_the_solve():
    """gh #432 on the v2 route: options set on the solver instance, not
    just per call, must reach the solver. On the legacy side dropping
    these silently un-tuned every model that gained a declaration."""
    s = SolverFactoryV2("pounce")
    if not s.available():
        pytest.skip("pounce binary not found")
    s.config.solver_options["max_iter"] = 1
    results = s.solve(
        _model(),
        raise_exception_on_nonoptimal_result=False,
        load_solutions=False)
    assert results.termination_condition is TerminationCondition.iterationLimit


def test_instance_options_reach_the_sensitivity_route():
    """The same, through the in-process path — which reads the options
    itself rather than handing them to a subprocess, so it is a separate
    piece of plumbing that can break on its own."""
    pytest.importorskip("pounce")
    s = SolverFactoryV2("pounce")
    if not s.available():
        pytest.skip("pounce binary not found")
    s.config.solver_options["max_iter"] = 1
    results = s.solve(
        _sens_model(),
        raise_exception_on_nonoptimal_result=False,
        load_solutions=False)
    assert results.termination_condition is TerminationCondition.iterationLimit


def test_iteration_count_and_timing_are_parsed(v2):
    """POUNCE's log is Ipopt-format on the lines Pyomo's parser reads
    (`Number of Iterations....:` and `Total seconds in POUNCE =`), which
    is why that parser is inherited rather than reimplemented. If POUNCE's
    log format drifts, this is what catches it."""
    m = _model()
    results = v2.solve(m)
    assert results.extra_info.iteration_count is not None
    assert results.extra_info.iteration_count > 0


# ---------------------------------------------------------------------------
# the sensitivity route, translated onto the v2 contract
# ---------------------------------------------------------------------------


def _sens_model():
    from pyomo_pounce import declare_sens_param

    m = _model()
    declare_sens_param(m.p)
    return m


def test_sens_route_solves_and_keeps_the_session(v2):
    """A model carrying `declare_sens_param` must route through the
    in-process session on v2 too, so that `sens_jacobian()` works afterwards —
    the whole point of the sensitivity path. Pointing `ipopt_v2` at the
    binary silently skips it."""
    pytest.importorskip("pounce")
    from pyomo_pounce import sens_jacobian

    m = _sens_model()
    results = v2.solve(m)
    assert results.solution_status is SolutionStatus.optimal
    # d/dp of the solution: available only because the KKT factorization
    # was retained by the in-process solve. `x` sits on its lower bound
    # here, so the constraint `x + 2y = p` pins dy/dp to exactly 1/2 --
    # a value that is checkable by hand rather than merely reproducible.
    assert sens_jacobian(m.y, wrt=m.p) == pytest.approx(0.5, rel=1e-5)


def test_sens_route_matches_the_ordinary_route(v2):
    """The sensitivity route is a different solver path (evaluator
    callbacks, no `.nl` handed to a subprocess), so it has to be checked
    against the ordinary one rather than assumed equivalent — primals,
    objective, duals and reduced costs, with the sign conventions the
    loader translates by hand."""
    pytest.importorskip("pounce")

    plain = _model()
    plain_results = v2.solve(plain)

    sens = _sens_model()
    sens_results = v2.solve(sens)

    assert value(sens.x) == pytest.approx(value(plain.x), rel=1e-6, abs=1e-8)
    assert value(sens.y) == pytest.approx(value(plain.y), rel=1e-6, abs=1e-8)
    assert sens_results.incumbent_objective == pytest.approx(
        plain_results.incumbent_objective, rel=1e-6, abs=1e-8)
    assert sens.dual[sens.c] == pytest.approx(
        plain.dual[plain.c], rel=1e-5, abs=1e-7)
    assert sens.rc[sens.x] == pytest.approx(
        plain.rc[plain.x], rel=1e-5, abs=1e-7)


def test_sens_route_matches_the_ordinary_route_on_a_maximization(v2):
    """The same parity, on the one model class that can tell a sign
    convention from no sign convention at all.

    A multiplier is a stationarity coefficient of the objective it was
    generated against, and `pounce.read_nl` negates a `maximize`
    objective before the engine ever sees it -- so `info['mult_g']` and
    `mult_x_*` are stated against `-f` while `dual` and `rc` are stated
    against the `f` the model wrote. The loader converts with
    `capture['obj_sign']`.

    That factor is +1 on every minimization, so the test above passes
    whether or not it is applied: `_model()` is a minimization, and so
    is every other model in this file. This is the case that separates
    them.
    """
    pytest.importorskip("pounce")

    plain = _model(sense=maximize)
    v2.solve(plain)
    sens = _model(sense=maximize)
    from pyomo_pounce import declare_sens_param
    declare_sens_param(sens.p)
    v2.solve(sens)

    assert value(sens.x) == pytest.approx(value(plain.x), rel=1e-6, abs=1e-8)
    assert sens.dual[sens.c] == pytest.approx(
        plain.dual[plain.c], rel=1e-5, abs=1e-7)
    assert sens.rc[sens.x] == pytest.approx(
        plain.rc[plain.x], rel=1e-5, abs=1e-7)
    # ...and against the minimize spelling of the same model, which
    # holds whatever the arithmetic is: `max -f` and `min f` reach one
    # point and every marginal negates
    lo = _model()
    v2.solve(lo)
    assert sens.dual[sens.c] == pytest.approx(-lo.dual[lo.c], rel=1e-5,
                                              abs=1e-7)
    assert sens.rc[sens.x] == pytest.approx(-lo.rc[lo.x], rel=1e-5, abs=1e-7)


def test_sens_route_honours_load_solutions_false(v2):
    """The substantive difference between the legacy and v2 contracts: v2
    hands the solution back through a loader the caller may decline to
    load. The legacy sensitivity path writes values onto the model as a
    side effect of solving, so this had to be translated, not copied."""
    pytest.importorskip("pounce")

    m = _sens_model()
    before = value(m.x)
    results = v2.solve(m, load_solutions=False)
    assert value(m.x) == before
    vals = results.solution_loader.get_vars()
    assert vals[m.x] != pytest.approx(before)
    # ...and loading on demand still works
    results.solution_loader.load_vars()
    assert value(m.x) == pytest.approx(vals[m.x])


def test_routes_agree_on_status_at_the_iteration_limit(v2):
    """Both routes must report the same `solution_status` for a
    limit-stopped solve, with **default** `load_solutions`.

    This is the case the other two limit tests cannot see, because they
    pass `load_solutions=False`. `noSolution` is not loadable and
    `unknown` is, so a disagreement here is not cosmetic: it decides
    whether `solve()` raises `NoSolutionError` for the same model and
    options. The sens route mapped these to `noSolution` originally,
    which both diverged from the ordinary route and over-stated the
    outcome — POUNCE returns a usable iterate in both cases.
    """
    pytest.importorskip("pounce")

    opts = {"max_iter": 1}
    plain = _model()
    plain_results = v2.solve(
        plain, solver_options=opts,
        raise_exception_on_nonoptimal_result=False)
    sens = _sens_model()
    sens_results = v2.solve(
        sens, solver_options=opts,
        raise_exception_on_nonoptimal_result=False)

    assert sens_results.solution_status is plain_results.solution_status
    assert sens_results.solution_status is SolutionStatus.unknown
    # ...and both loaded the final iterate rather than raising
    assert value(plain.x) is not None
    assert value(sens.x) is not None


def test_limit_stopped_solve_reports_no_incumbent_objective(v2):
    """`incumbent_objective` is gated on {feasible, optimal} on both
    routes: the objective at a limit-stopped iterate is not the objective
    at a solution, and the ordinary route leaves it None."""
    pytest.importorskip("pounce")

    opts = {"max_iter": 1}
    plain_results = v2.solve(
        _model(), solver_options=opts,
        raise_exception_on_nonoptimal_result=False)
    sens_results = v2.solve(
        _sens_model(), solver_options=opts,
        raise_exception_on_nonoptimal_result=False)
    assert plain_results.incumbent_objective is None
    assert sens_results.incumbent_objective is None


def test_sens_loader_does_not_warn_about_surgery_components(v2, recwarn):
    """The unresolved-name check must stay quiet on a declared-sens-param
    model. The solve there runs on a surgery clone whose `.col`/`.row`
    legitimately name components that exist only on the clone
    (`_SENSITIVITY_TOOLBOX_DATA.p` and the pin row), so a plain count of
    resolved-vs-solved names would warn on every sensitivity model — the
    exact case this route exists for."""
    pytest.importorskip("pounce")

    m = _sens_model()
    results = v2.solve(m, load_solutions=False)
    recwarn.clear()
    results.solution_loader.get_vars()
    results.solution_loader.get_duals()
    results.solution_loader.get_reduced_costs()
    unresolved = [w for w in recwarn
                  if "could not be resolved" in str(w.message)]
    assert not unresolved, [str(w.message) for w in unresolved]


def test_sens_loader_does_warn_about_a_real_unresolved_name(v2):
    """...but it must still fire for a name the surgery cannot explain,
    or the test above is vacuous. A variable the solve knows about and
    the model does not is silently skipped by `load_vars`, leaving that
    variable stale with no diagnostic."""
    pytest.importorskip("pounce")

    m = _sens_model()
    results = v2.solve(m, load_solutions=False)
    loader = results.solution_loader
    # Splice in a name that is neither on the model nor under the
    # sensitivity-toolbox block.
    loader._capture["var_names"] = list(loader._capture["var_names"]) + [
        "not_a_component"]
    loader._capture["x"] = list(loader._capture["x"]) + [0.0]
    with pytest.warns(UserWarning, match="could not be resolved"):
        loader.get_vars()


def test_sens_route_reports_iteration_count(v2):
    pytest.importorskip("pounce")

    m = _sens_model()
    results = v2.solve(m)
    assert results.extra_info.iteration_count is not None
    assert results.extra_info.iteration_count > 0
