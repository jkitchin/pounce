"""`declare_sens_param` must not cost the caller their duals.

An ordinary `SolverFactory("pounce").solve(m)` writes an NL file, runs the
binary, and hands the `.sol` back to Pyomo's own solution loader, which fills
every active IMPORT suffix on the model.  A model carrying a declaration takes
a different route: `sens_solve` builds evaluator callbacks from
`pounce.read_nl` and reads the converged vectors straight out of the engine, so
the loader never runs.  That route used to load **primals only**.

The failure mode is the quiet kind.  Nothing about declaring a sensitivity
parameter suggests it should touch duals, and the two solves differ in no
visible way -- same call, same status, same objective, same primals.  The
suffix simply comes back empty and the first `m.dual[c]` is a `KeyError` on a
constraint that is plainly right there in the model.  It cost notebook 38 a
whole extra solve: section 10 takes the multiplier from a separate ordinary
solve and only `dlambda/dtheta` from the declared one.

THE CLOSED FORM, which is the same one three times.  All three suffixes are
the derivative of the objective with respect to relaxing something:

    dual[c]            = d obj / d (rhs of c)      -- the AMPL marginal
    ipopt_zL_out[v]    = d obj / d (lower bound of v)
    ipopt_zU_out[v]    = d obj / d (upper bound of v)

`ipopt_zL_out` is therefore positive and `ipopt_zU_out` negative at an active
bound of a minimization (gh #296), and `dual` is the marginal rather than the
internal `+lambda` (gh #271).  Reading them as one quantity is what makes it
obvious that all three carry the objective's *sense*: on a maximization each is
a derivative of `-f` unless something puts it back, which is the second half of
what these tests pin.  See `test_objective_sense.py` for the sense itself.

ORACLES.  The expected values below are worked out by hand from that
definition, on a separable model whose optimum is arithmetic.  The `.sol` route
runs alongside as an independent second opinion -- it is the code path that was
already right, on the same model, and it is where a value would have to agree
even if the hand derivation were wrong.
"""

from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
import pytest

from pyomo_pounce import declare_sens_param
from pyomo_pounce.sens import (
    _load_result_suffixes,
    _warm_start_from_suffixes,
)

OUT = ("dual", "ipopt_zL_out", "ipopt_zU_out")


# -- the fixture -----------------------------------------------------
#
# Four variables, one objective term each, so every optimum and every
# marginal is separable arithmetic:
#
#   min (x-10)^2 + (y-10)^2 + (z-10)^2 + (w-10)^2
#   s.t.  c:  x + y == b        (b = 6, a declared parameter)
#         z <= 3
#         w >= 12
#
#   x* = y* = 3     obj term 2*(3-10)^2;  d obj/db = 2(b/2 - 10) = -14
#   z* = 3   (ub)   d obj/d ub = 2(3-10)  = -14
#   w* = 12  (lb)   d obj/d lb = 2(12-10) = +4
#
# The maximize spelling optimizes the negation, so it reaches the same
# point and every marginal flips sign and nothing else.

MARGINALS = {"c": -14.0, "z_ub": -14.0, "w_lb": +4.0}


def build(b=6.0, sense=pyo.minimize, suffixes=OUT):
    m = pyo.ConcreteModel()
    m.b = pyo.Param(initialize=b, mutable=True)
    m.bv = pyo.Var(initialize=b)
    m.pin = pyo.Constraint(expr=m.bv == m.b)
    m.x = pyo.Var(initialize=0.0)
    m.y = pyo.Var(initialize=0.0)
    m.z = pyo.Var(bounds=(None, 3.0), initialize=0.0)
    m.w = pyo.Var(bounds=(12.0, None), initialize=12.0)
    m.c = pyo.Constraint(expr=m.x + m.y == m.bv)
    f = sum((v - 10) ** 2 for v in (m.x, m.y, m.z, m.w))
    m.obj = pyo.Objective(expr=-f if sense == pyo.maximize else f,
                          sense=sense)
    for n in suffixes:
        setattr(m, n, pyo.Suffix(direction=pyo.Suffix.IMPORT))
    return m


def _solve(m):
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10})
    return m


def declared(**kw):
    m = build(**kw)
    declare_sens_param(m.b)
    return _solve(m)


def grab(m):
    return {n: {k.name: float(v) for k, v in getattr(m, n).items()}
            for n in OUT}


# -- the reported symptom --------------------------------------------

def test_a_declared_model_still_reports_its_duals():
    """The bug as a user meets it: declare a parameter, solve, read the
    dual of a constraint that is right there in the model."""
    m = declared()
    assert len(m.dual), "declare_sens_param returned an empty dual Suffix"
    assert m.dual[m.c] == pytest.approx(MARGINALS["c"], abs=1e-6)


def test_the_optimum_is_where_the_hand_derivation_says():
    """Guard on the guard: if the fixture does not reach x=y=3, z=3,
    w=12 then the marginals below are being compared against the wrong
    model and every other test in this file is vacuous."""
    m = declared()
    for v, want in ((m.x, 3.0), (m.y, 3.0), (m.z, 3.0), (m.w, 12.0)):
        assert pyo.value(v) == pytest.approx(want, abs=1e-6)


# -- the values, against the hand derivation -------------------------

@pytest.mark.parametrize("sense", [pyo.minimize, pyo.maximize])
def test_every_suffix_is_the_marginal_of_the_objective_as_written(sense):
    """All three suffixes are `d obj / d (the thing relaxed)`, so on the
    maximize spelling all three are exactly the negation."""
    s = 1.0 if sense == pyo.minimize else -1.0
    m = declared(sense=sense)
    assert m.dual[m.c] == pytest.approx(s * MARGINALS["c"], abs=1e-6)
    assert m.ipopt_zU_out[m.z] == pytest.approx(s * MARGINALS["z_ub"],
                                                abs=1e-6)
    assert m.ipopt_zL_out[m.w] == pytest.approx(s * MARGINALS["w_lb"],
                                                abs=1e-6)


# -- the values, against the route that was already right ------------

@pytest.mark.parametrize("sense", [pyo.minimize, pyo.maximize])
def test_parity_with_the_sol_route_on_the_same_model(sense):
    """Every entry the `.sol` route reports, the in-process route
    reports too, with the same value.

    The reverse containment is deliberately not asserted: the `.sol`
    writer emits one entry per variable -- the combined reduced cost,
    routed to `zL` when positive and `zU` when negative -- so a bound
    whose multiplier lost that comparison is not reported at all.  The
    in-process route answers the question the suffix name asks instead,
    one entry per finite bound.  `_load_result_suffixes` documents the
    difference; the test below pins that it is the only one.
    """
    ref = grab(_solve(build(sense=sense)))
    got = grab(declared(sense=sense))
    for n in OUT:
        for key, val in ref[n].items():
            assert key in got[n], f"{n}[{key}] missing from the declared solve"
            assert got[n][key] == pytest.approx(val, abs=1e-6, rel=1e-6)


@pytest.mark.parametrize("sense", [pyo.minimize, pyo.maximize])
def test_the_only_extra_entries_are_bounds_the_sol_route_declines_to_report(
        sense):
    """The membership difference, stated as a bound rather than left
    open: an entry the `.sol` route omits is a bound whose multiplier is
    numerically zero -- an inactive bound -- never an active one."""
    ref = grab(_solve(build(sense=sense)))
    got = grab(declared(sense=sense))
    for n in ("ipopt_zL_out", "ipopt_zU_out"):
        for key, val in got[n].items():
            if key not in ref[n]:
                assert abs(val) < 1e-6, (
                    f"{n}[{key}] = {val:+.6g} is an active bound the .sol "
                    "route did not report; the routes disagree on more than "
                    "membership")


def test_a_model_that_declares_no_suffix_is_left_alone():
    """The load is driven by the model's own IMPORT suffixes, so a model
    that asked for nothing gets nothing -- and in particular does not
    acquire a `dual` component it never declared."""
    m = build(suffixes=())
    declare_sens_param(m.b)
    _solve(m)
    assert m.component("dual") is None


def test_a_stale_entry_does_not_survive_the_next_solve():
    """`Model.solutions.load_from` clears every active import suffix
    before loading, so a previous solve's answer cannot be left standing
    under a new solution.  This route does the same."""
    m = declared()
    m.dual[m.pin] = 1234.5
    m.b.value = 8.0
    _solve(m)
    # b = 8 -> x = y = 4, marginal 2(4 - 10) = -12
    assert m.dual[m.c] == pytest.approx(-12.0, abs=1e-6)
    assert m.dual[m.pin] != pytest.approx(1234.5)


def test_a_limit_stopped_solve_reports_the_iterates_multipliers():
    """The non-converged branch loads suffixes too, and it is a separate
    branch: the converged path builds a session and the failed one drops
    it, so the load had to be wired into both.

    The `.sol` route loads a limit-stopped solution's suffixes -- it has
    no idea the solve stopped early, it just parses what came back -- so
    this route must as well.  The alternative is worse than an empty
    suffix: the previous solve's duals left standing under a new set of
    primals, with nothing to say they do not belong together.
    """
    m = declared()
    assert m.dual[m.c] == pytest.approx(MARGINALS["c"], abs=1e-6)
    m.b.value = 8.0
    res = pyo.SolverFactory("pounce").solve(m, options={"max_iter": 1})
    assert str(res.solver.termination_condition) != "optimal", (
        "the fixture must actually stop at the limit")
    # the iterate's own multipliers, not the converged solve's
    assert m.dual[m.c] != pytest.approx(MARGINALS["c"], abs=1e-6)
    assert len(m.dual), "a limit-stopped solve reported no duals at all"


# -- the sign algebra alone, with no solver in the room --------------

class _FakeNl:
    """The two fields the sign path reads."""

    def __init__(self, n, m, minimize):
        self.n, self.m, self.minimize = n, m, minimize


@pytest.mark.parametrize("minimize", [True, False])
def test_writing_the_suffixes_and_reading_them_back_is_the_identity(minimize):
    """`_load_result_suffixes` and `_warm_start_from_suffixes` are
    inverses, and this pins that composition directly -- no solver, no
    fixture, so it fails for exactly one reason.

    A user who warm-starts a re-solve copies `ipopt_zL_out` into
    `ipopt_zL_in`; that copy is the only step below that is not library
    code.  Round-tripping through the suffixes must hand the engine back
    the multipliers it reported, on a maximization as much as on a
    minimization -- otherwise `warm_start_init_point=yes` seeds a
    certificate of the wrong sign, which is a worse starting point than
    the default it displaced.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0.0, 5.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-1.0, None), initialize=1.0)
    m.c = pyo.Constraint(expr=m.x + m.y == 2.0)
    m.obj = pyo.Objective(
        expr=m.x + m.y,
        sense=pyo.minimize if minimize else pyo.maximize)
    for n in OUT:
        setattr(m, n, pyo.Suffix(direction=pyo.Suffix.IMPORT))
    # EXPORT, so the load's clear_all_values() leaves them alone -- they
    # are what the caller sends *to* the solver, not what it reads back
    for n in ("ipopt_zL_in", "ipopt_zU_in"):
        setattr(m, n, pyo.Suffix(direction=pyo.Suffix.EXPORT))

    nl = _FakeNl(n=2, m=1, minimize=minimize)
    info = {"mult_g": np.array([2.5]),
            "mult_x_L": np.array([0.75, 0.5]),
            "mult_x_U": np.array([1.25, 0.0])}
    var_names, con_names = ["x", "y"], ["c"]

    _load_result_suffixes(m, info, nl, [m.x, m.y], con_names, {})
    # y has no upper bound, so it has no zU entry to report
    assert set(k.name for k in m.ipopt_zU_out) == {"x"}
    for src, dst in (("ipopt_zL_out", "ipopt_zL_in"),
                     ("ipopt_zU_out", "ipopt_zU_in")):
        for k, v in getattr(m, src).items():
            getattr(m, dst)[k] = v

    seed = _warm_start_from_suffixes(m, var_names, con_names, nl, {})
    assert seed["lagrange"] == pytest.approx(info["mult_g"])
    assert seed["zl"] == pytest.approx(info["mult_x_L"])
    # y's absent zU comes back unseeded (NaN), the session's marker for
    # "use the resolved default", not a fabricated zero
    assert seed["zu"][0] == pytest.approx(info["mult_x_U"][0])
    assert np.isnan(seed["zu"][1])


def test_the_sense_is_the_whole_difference_between_the_two_spellings():
    """The pair of the test above: the same engine vectors, read out
    under the two senses, differ by exactly a sign in every entry."""
    out = {}
    for minimize in (True, False):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0.0, 5.0), initialize=1.0)
        m.c = pyo.Constraint(expr=m.x == 2.0)
        m.obj = pyo.Objective(
            expr=m.x, sense=pyo.minimize if minimize else pyo.maximize)
        for n in OUT:
            setattr(m, n, pyo.Suffix(direction=pyo.Suffix.IMPORT))
        info = {"mult_g": np.array([2.5]),
                "mult_x_L": np.array([0.75]),
                "mult_x_U": np.array([1.25])}
        _load_result_suffixes(m, info, _FakeNl(1, 1, minimize), [m.x],
                              ["c"], {})
        out[minimize] = [m.dual[m.c], m.ipopt_zL_out[m.x],
                         m.ipopt_zU_out[m.x]]
    assert out[True] == pytest.approx([-x for x in out[False]])
    # and the minimize arm is the convention the docstrings state
    assert out[True] == pytest.approx([-2.5, 0.75, -1.25])
