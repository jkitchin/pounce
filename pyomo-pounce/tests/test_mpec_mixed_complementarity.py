"""Complementarity forms Pyomo's ``mpec.nl`` route loses on the way to ``.nl``.

Measured on Pyomo 6.10.0, before the fix, every model below was solved to
``optimal`` at a point that violates the complementarity it states:

* a two-sided ("mixed") pair, ``complements(inequality(lo, r, hi), f)`` in
  either order, was either dropped from the file outright (``n_cc = 0``: the
  writer skips the free row ``mpec.nl`` builds for it) or written with flag 0,
  which reads as the equation ``r == lo``;
* a one-sided pair on a bare variable that has a different bound of its own on
  the same side had the two bounds merged -- a tighter declared ``ub`` was
  discarded, a looser one replaced the pair's, and a tighter declared ``lb``
  became the complementarity's reference point.

``pyomo_pounce.mpec`` lowers those pairs itself, and checks the written file,
refusing the solve when a pair is still missing. Each expected point was worked
out by hand from the branches of the pair (stated next to the case), and every
returned point is also checked against the complementarity directly, so a
case cannot pass by landing on the unconstrained optimum.
"""
import pyomo.environ as pyo
import pyomo.version
import pytest
from pyomo.mpec import Complementarity, ComplementarityList, complements

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import mpec as pp_mpec

_PYOMO_VERSION = tuple(pyomo.version.version_info[:3])

TOL = 1e-5


def _model(xb=(None, None)):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=xb, initialize=0.5)
    m.y = pyo.Var(initialize=0.5)
    m.z = pyo.Var(initialize=0.5)
    return m


def _solve(m, **kw):
    res = pyo.SolverFactory("pounce").solve(m, **kw)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal, res
    return res


def _assert_mixed(lo, r, hi, f):
    """lo <= r <= hi, f >= 0 at lo, f <= 0 at hi, f == 0 strictly between."""
    r, f = pyo.value(r), pyo.value(f)
    assert lo - TOL <= r <= hi + TOL, (r, lo, hi)
    if abs(r - lo) <= TOL:
        assert f >= -TOL, (r, f)
    elif abs(r - hi) <= TOL:
        assert f <= TOL, (r, f)
    else:
        assert abs(f) <= TOL, (r, f)


def _assert_one_sided(a, b):
    """a >= 0, b >= 0, a * b == 0."""
    a, b = pyo.value(a), pyo.value(b)
    assert a >= -TOL and b >= -TOL and min(abs(a), abs(b)) <= TOL, (a, b)


# ---------------------------------------------------------------- mixed pairs

# (id, x bounds, objective target (x, y, z), pair builder, (lo, r, hi, f), want (x, y, z))
MIXED = [
    # min (x-.3)^2+(y-.5)^2: interior x forces x == y -> (.4, .4); at x = 0,
    # y <= 0 costs .34.  The original report's model.
    ("var-range-first", (0, 1), (0.3, 0.5, 0.5),
     lambda m: complements(pyo.inequality(0, m.x, 1), m.x - m.y),
     lambda m: (0, m.x, 1, m.x - m.y), (0.4, 0.4, 0.5)),
    ("var-range-second", (0, 1), (0.3, 0.5, 0.5),
     lambda m: complements(m.x - m.y, pyo.inequality(0, m.x, 1)),
     lambda m: (0, m.x, 1, m.x - m.y), (0.4, 0.4, 0.5)),
    # The range is the pair's alone: x has no declared bounds.
    ("var-range-undeclared", (None, None), (0.3, 0.5, 0.5),
     lambda m: complements(pyo.inequality(0, m.x, 1), m.x - m.y),
     lambda m: (0, m.x, 1, m.x - m.y), (0.4, 0.4, 0.5)),
    # Declared bounds looser than the range: still complementarity at 0 and 1.
    # min (x-2)^2+(y-.5)^2: at x = 1, x - y <= 0 -> y = 1.
    ("var-range-declared-looser", (-5, 5), (2.0, 0.5, 0.5),
     lambda m: complements(pyo.inequality(0, m.x, 1), m.x - m.y),
     lambda m: (0, m.x, 1, m.x - m.y), (1.0, 1.0, 0.5)),
    # Upper branch: (1, 1).  Lower branch: x = 0, y = -2 (x - y = 2 >= 0).
    ("upper-branch", (0, 1), (2.0, 0.5, 0.5),
     lambda m: complements(pyo.inequality(0, m.x, 1), m.x - m.y),
     lambda m: (0, m.x, 1, m.x - m.y), (1.0, 1.0, 0.5)),
    ("lower-branch", (0, 1), (-1.0, -2.0, 0.5),
     lambda m: complements(pyo.inequality(0, m.x, 1), m.x - m.y),
     lambda m: (0, m.x, 1, m.x - m.y), (0.0, -2.0, 0.5)),
    # Constant in the free side (the writer drops a complementarity row's
    # constant, so it must reach the file through the definition row):
    # interior x - y + .1 = 0 -> (.35, .45).
    ("free-side-constant", (0, 1), (0.3, 0.5, 0.5),
     lambda m: complements(pyo.inequality(0, m.x, 1), m.x - m.y + 0.1),
     lambda m: (0, m.x, 1, m.x - m.y + 0.1), (0.35, 0.45, 0.5)),
    # Expression range, bare variable on the free side (was written flag 0).
    # min (x-.3)^2+(y-.5)^2+(z-.1)^2: interior x - z forces y = 0 (cost .25);
    # x - z = 0 with y = .5 costs .02 -> (.2, .5, .2).
    ("expr-range-var-free", (None, None), (0.3, 0.5, 0.1),
     lambda m: complements(pyo.inequality(0, m.x - m.z, 1), m.y),
     lambda m: (0, m.x - m.z, 1, m.y), (0.2, 0.5, 0.2)),
    ("var-free-expr-range", (None, None), (0.3, 0.5, 0.1),
     lambda m: complements(m.y, pyo.inequality(0, m.x - m.z, 1)),
     lambda m: (0, m.x - m.z, 1, m.y), (0.2, 0.5, 0.2)),
    # Expression on both sides: interior forces x = y, then z = .1 -> x+z in (0,1).
    ("expr-range-expr-free", (None, None), (0.3, 0.5, 0.1),
     lambda m: complements(pyo.inequality(0, m.x + m.z, 1), m.x - m.y),
     lambda m: (0, m.x + m.z, 1, m.x - m.y), (0.4, 0.4, 0.1)),
    # Variables on both sides: x interior forces y = 0 (cost .25); x = 0
    # with y = .5 costs .09 -> (0, .5).
    ("var-range-var-free", (0, 1), (0.3, 0.5, 0.5),
     lambda m: complements(pyo.inequality(0, m.x, 1), m.y),
     lambda m: (0, m.x, 1, m.y), (0.0, 0.5, 0.5)),
    # The workaround of the original report: an auxiliary s == x - y.
    ("aux-free-variable", (0, 1), (0.3, 0.5, 0.5),
     lambda m: complements(pyo.inequality(0, m.x, 1), m.s),
     lambda m: (0, m.x, 1, m.s), (0.4, 0.4, 0.5)),
]


@pytest.mark.parametrize("case", MIXED, ids=[c[0] for c in MIXED])
def test_a_mixed_pair_is_solved_to_its_complementary_answer(case):
    _, xb, tgt, pair, check, want = case
    m = _model(xb)
    m.s = pyo.Var(initialize=0.0)
    m.s_def = pyo.Constraint(expr=m.s == m.x - m.y)
    m.o = pyo.Objective(expr=(m.x - tgt[0]) ** 2 + (m.y - tgt[1]) ** 2
                        + (m.z - tgt[2]) ** 2)
    m.c = Complementarity(expr=pair(m))
    _solve(m)
    _assert_mixed(*check(m))
    for v, w in zip((m.x, m.y, m.z), want):
        assert pyo.value(v) == pytest.approx(w, abs=1e-4), (v.name, w)


def test_the_written_file_carries_a_flag_3_row_for_a_mixed_pair(tmp_path):
    m = _model((0, 1))
    m.o = pyo.Objective(expr=(m.x - 0.3) ** 2 + (m.y - 0.5) ** 2)
    m.c = Complementarity(expr=complements(pyo.inequality(0, m.x, 1), m.x - m.y))
    opt = pyo.SolverFactory("pounce")
    _solve_keep = opt.solve(m, keepfiles=True)
    assert _solve_keep.solver.termination_condition == pyo.TerminationCondition.optimal
    nl = opt._problem_files[0]
    assert pp_mpec.count_written_pairs(nl) == (1, 0)
    rows = open(nl).read()
    assert "\n5 3 " in rows


def test_indexed_mixed_pairs_alongside_a_one_sided_list_on_a_sub_block():
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(3)
    m.x = pyo.Var(m.I, bounds=(0, 1), initialize=0.5)
    m.y = pyo.Var(m.I, initialize=0.5)
    tgt = {1: (0.3, 0.5), 2: (2.0, 0.5), 3: (-1.0, -2.0)}
    m.o = pyo.Objective(expr=sum((m.x[i] - tgt[i][0]) ** 2
                                 + (m.y[i] - tgt[i][1]) ** 2 for i in m.I))
    m.c = Complementarity(m.I, rule=lambda m, i: complements(
        pyo.inequality(0, m.x[i], 1), m.x[i] - m.y[i]))
    m.sub = pyo.Block()
    m.sub.z = pyo.Var(initialize=1.0)
    # 0 <= z ⟂ z - x[1] + .5 >= 0; with x[1] = .4 the row is > 0 at z = 0.
    m.sub.cl = ComplementarityList(
        rule=[complements(m.sub.z >= 0, m.sub.z - m.x[1] + 0.5 >= 0)])
    _solve(m)
    want = {1: (0.4, 0.4), 2: (1.0, 1.0), 3: (0.0, -2.0)}
    for i in m.I:
        _assert_mixed(0, m.x[i], 1, m.x[i] - m.y[i])
        assert pyo.value(m.x[i]) == pytest.approx(want[i][0], abs=1e-4)
        assert pyo.value(m.y[i]) == pytest.approx(want[i][1], abs=1e-4)
    _assert_one_sided(m.sub.z, m.sub.z - m.x[1] + 0.5)
    assert pyo.value(m.sub.z) == pytest.approx(0.0, abs=1e-4)


# ------------------------------------------- one-sided pairs: bound conflicts

ONE_SIDED_CONFLICT = [
    # x <= .2 declared, pair 1 - x >= 0 ⟂ x - y >= 0: 1 - x > 0 forces
    # x == y, so (.2, .2).  mpec.nl loosened x's ub to 1 and returned (.4, .4).
    ("declared-ub-tighter", (None, 0.2), (0.3, 0.5),
     lambda m: complements(m.x <= 1, m.x - m.y >= 0),
     lambda m: (1 - m.x, m.x - m.y), (0.2, 0.2)),
    # x <= 2 declared, pair 1 - x >= 0 ⟂ y - x + 1 >= 0: x is held to 1;
    # at x = 1, y >= 0 -> (1, .5).  mpec.nl kept ub 2 and returned (1.5, .5).
    ("declared-ub-looser", (None, 2.0), (1.5, 0.5),
     lambda m: complements(m.x <= 1, m.y - m.x + 1 >= 0),
     lambda m: (1 - m.x, m.y - m.x + 1), (1.0, 0.5)),
    # x >= .5 declared, pair x >= 0 ⟂ x - y >= 0: x > 0 forces x == y, so
    # (.5, .5).  mpec.nl made .5 the reference point and returned (.5, -1).
    ("declared-lb-tighter", (0.5, None), (-1.0, -1.0),
     lambda m: complements(m.x >= 0, m.x - m.y >= 0),
     lambda m: (m.x, m.x - m.y), (0.5, 0.5)),
]


@pytest.mark.parametrize("case", ONE_SIDED_CONFLICT,
                         ids=[c[0] for c in ONE_SIDED_CONFLICT])
def test_a_one_sided_pair_keeps_both_its_bound_and_the_declared_one(case):
    _, xb, tgt, pair, check, want = case
    m = _model(xb)
    m.o = pyo.Objective(expr=(m.x - tgt[0]) ** 2 + (m.y - tgt[1]) ** 2)
    m.c = Complementarity(expr=pair(m))
    _solve(m)
    _assert_one_sided(*check(m))
    lb, ub = xb
    assert lb is None or pyo.value(m.x) >= lb - TOL
    assert ub is None or pyo.value(m.x) <= ub + TOL
    assert (pyo.value(m.x), pyo.value(m.y)) == pytest.approx(want, abs=1e-4)
    assert m.x.bounds == xb  # and the declared bounds are untouched


# ---------------------------------- one-sided pairs that already worked: as-is

ONE_SIDED_OK = [
    ("var-expr", lambda m: complements(m.x >= 0, m.x - m.y >= 0)),
    ("expr-var", lambda m: complements(m.x - m.y >= 0, m.x >= 0)),
    ("expr-expr", lambda m: complements(m.x - m.y >= 0, m.x + m.z >= 0)),
    ("upper", lambda m: complements(m.x <= 1, m.x - m.y >= 0)),
    ("row-upper", lambda m: complements(m.x >= 0, m.x - m.y <= 0)),
]


@pytest.mark.parametrize("case", ONE_SIDED_OK, ids=[c[0] for c in ONE_SIDED_OK])
def test_a_one_sided_pair_mpec_nl_writes_correctly_is_left_to_it(case):
    """Not lowered by pyomo-pounce (so its `.nl` is what it always was), still
    solved to (.4, .4), and the bound mpec.nl writes onto x is taken back."""
    _, pair = case
    m = _model()
    m.o = pyo.Objective(expr=(m.x - 0.3) ** 2 + (m.y - 0.5) ** 2
                        + (m.z - 0.1) ** 2)
    m.c = Complementarity(expr=pair(m))
    low = pp_mpec.lower_complementarity(m)
    assert low.block is None and low.lowered == [] and low.expected_pairs == 1
    low.restore()
    opt = pyo.SolverFactory("pounce")
    res = opt.solve(m, keepfiles=True, symbolic_solver_labels=True)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert "_pounce_complementarity" not in open(opt._problem_files[0]).read()
    assert (pyo.value(m.x), pyo.value(m.y)) == pytest.approx((0.4, 0.4), abs=1e-4)
    assert m.x.bounds == (None, None)


def test_a_nonlinear_biactive_one_sided_pair_still_solves():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.y = pyo.Var(initialize=1.0)
    m.o = pyo.Objective(expr=(m.x + 1) ** 2 + (m.y + 1) ** 2)
    m.c = Complementarity(expr=complements(m.x >= 0, m.y - m.x**2 >= 0))
    _solve(m)
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-4)
    assert pyo.value(m.y) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------- no complementarity: untouched

def test_a_model_without_complementarity_is_not_touched():
    m = _model((0, 1))
    m.o = pyo.Objective(expr=(m.x - 0.3) ** 2 + (m.y - 0.5) ** 2)
    m.r = pyo.Constraint(expr=m.x + m.y >= 1)
    assert pp_mpec.lower_complementarity(m) is None
    before = sorted(c.name for c in m.component_objects())
    opt = pyo.SolverFactory("pounce")
    res = opt.solve(m, keepfiles=True)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    assert opt._cc_lowering is None
    assert pp_mpec.count_written_pairs(opt._problem_files[0]) == (0, 0)
    assert sorted(c.name for c in m.component_objects()) == before
    # min (x-.3)^2+(y-.5)^2 on x + y >= 1: (.4, .6).
    assert (pyo.value(m.x), pyo.value(m.y)) == pytest.approx((0.4, 0.6), abs=1e-5)


# ------------------------------------------------- the model is given back

def test_the_model_is_restored_after_the_solve_and_resolves_the_same():
    m = _model((0, 1))
    m.o = pyo.Objective(expr=(m.x - 0.3) ** 2 + (m.y - 0.5) ** 2)
    m.c = Complementarity(expr=complements(pyo.inequality(0, m.x, 1), m.x - m.y))
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    before = sorted(c.name for c in m.component_objects())
    for _ in range(2):
        _solve(m)
        assert (pyo.value(m.x), pyo.value(m.y)) == pytest.approx((0.4, 0.4), abs=1e-4)
        assert m.c.active
        assert sorted(c.name for c in m.component_objects()) == before
        assert m.x.bounds == (0, 1)
        # no dual left behind for an auxiliary row that no longer exists
        assert all(c.model() is m for c in m.dual)


def test_load_solutions_false_then_load_from():
    m = _model((0, 1))
    m.o = pyo.Objective(expr=(m.x - 0.3) ** 2 + (m.y - 0.5) ** 2)
    m.c = Complementarity(expr=complements(pyo.inequality(0, m.x, 1), m.x - m.y))
    res = pyo.SolverFactory("pounce").solve(m, load_solutions=False)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal
    m.solutions.load_from(res)
    assert (pyo.value(m.x), pyo.value(m.y)) == pytest.approx((0.4, 0.4), abs=1e-4)


# --------------------------------------------- fail closed when a pair is lost

@pytest.mark.parametrize("pair,why", [
    # mpec.nl drops this pair from the file entirely (n_cc = 0)...
    (lambda m: complements(pyo.inequality(0, m.x, 1), m.x - m.y), "carries 0"),
    # ...and writes this one with flag 0 (an equation).
    (lambda m: complements(pyo.inequality(0, m.x - m.z, 1), m.y), "flag 0"),
], ids=["dropped", "flag-0"])
def test_a_pair_lost_in_the_written_file_is_refused(monkeypatch, pair, why):
    """With the lowering disabled -- what a future Pyomo or a form this
    module does not recognise would look like -- the solve raises, naming
    the loss and what to do, instead of returning `optimal`."""
    real = pp_mpec._classify

    def as_mpec4(cdata):
        kind, sides = real(cdata)
        return ("mpec4", sides) if kind == "mixed" else (kind, sides)

    monkeypatch.setattr(pp_mpec, "_classify", as_mpec4)
    m = _model((0, 1))
    m.o = pyo.Objective(expr=(m.x - 0.3) ** 2 + (m.y - 0.5) ** 2)
    m.c = Complementarity(expr=pair(m))
    if why == "carries 0" and _PYOMO_VERSION >= (6, 10, 1):
        # Pyomo 6.10.1's mpec.nl no longer writes this form silently: it
        # fails while building the row ("does not have a proper value").
        # That is also fail-closed -- no `optimal` comes back -- so the
        # property under test still holds, just raised by Pyomo first.
        with pytest.raises(ValueError, match="proper value"):
            pyo.SolverFactory("pounce").solve(m)
        return
    with pytest.raises(pp_mpec.ComplementarityLoweringError, match=why) as ei:
        pyo.SolverFactory("pounce").solve(m)
    assert "pounce.Problem" in str(ei.value)
    assert m.c.active and m.x.bounds == (0, 1)
