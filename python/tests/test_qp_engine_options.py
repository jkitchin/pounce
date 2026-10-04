"""The CLI's convex-engine options on ``pounce.qp.solve_qp`` (gh#990).

``qp_presolve``, ``qp_reg``, ``qp_hsde``, ``qp_equilibrate`` and
``qp_crossover`` were CLI-only; the Python path documented the gap instead
of closing it, and a caller (discopt#1615) had to refuse them. These tests
pin three things: each option reaches the engine and changes something
observable; a bad value or an option the chosen engine has no counterpart for
is refused by name rather than dropped; and omitting them all is bitwise the
same solve as passing their defaults explicitly.
"""

import numpy as np
import pytest

from pounce import _pounce
from pounce.qp import _build, solve_qp

# min -x0 - x1 + x2   s.t.  x0 + x1 <= 1,  0 <= x0, x1 <= 10,  x2 fixed at 2.
# The optimal face {x0 + x1 = 1} is an edge, so the interior point lands in
# its middle and only a crossover returns a vertex; x2 is a fixed column, so
# presolve has something to remove.
LP = dict(
    c=np.array([-1.0, -1.0, 1.0]),
    G=np.array([[1.0, 1.0, 0.0]]),
    h=np.array([1.0]),
    lb=np.array([0.0, 0.0, 2.0]),
    ub=np.array([10.0, 10.0, 2.0]),
)

# A badly scaled QP, where equilibration changes the direct driver's path.
SKEWED = dict(
    P=np.diag([1e4, 1e-2, 1.0]),
    c=np.array([1e3, -1e-2, 3.0]),
    A=np.array([[1e3, 1e-3, 1.0]]),
    b=np.array([5.0]),
    G=np.array([[1.0, 1e2, 0.0]]),
    h=np.array([40.0]),
    lb=np.zeros(3),
    ub=np.full(3, 1e3),
)

DEFAULTS = dict(
    qp_presolve=False,
    qp_reg=1e-10,
    qp_hsde=True,
    qp_equilibrate=True,
    qp_crossover=False,
)


def _same(a, b):
    assert a.status == b.status
    assert a.iters == b.iters
    assert a.obj == b.obj
    for f in ("x", "y", "z", "z_lb", "z_ub"):
        assert getattr(a, f).tobytes() == getattr(b, f).tobytes(), f
    assert (a.tau, a.kappa) == (b.tau, b.kappa)
    assert a.iterates == b.iterates


def test_default_solve_reports_no_presolve_and_no_crossover():
    r = solve_qp(**LP)
    assert r.status == "optimal"
    assert r.presolve is None and r.crossover is None
    # The interior point sits strictly inside the optimal edge.
    assert 0.1 < r.x[0] < 0.9


@pytest.mark.parametrize("method", ["ipm", "active-set"])
@pytest.mark.parametrize("problem", [LP, SKEWED], ids=["lp", "skewed_qp"])
def test_omitting_the_options_is_bitwise_the_explicit_defaults(problem, method):
    kw = dict(collect_iterates=True, method=method)
    omitted = solve_qp(**problem, **kw)
    if method == "active-set":
        explicit = solve_qp(
            **problem, **kw, qp_presolve=False, qp_equilibrate=True
        )
    else:
        explicit = solve_qp(**problem, **kw, **DEFAULTS)
    _same(omitted, explicit)


def test_crossover_returns_a_vertex_and_reports_it():
    interior = solve_qp(**LP)
    r = solve_qp(**LP, qp_crossover=True)
    assert r.status == "optimal"
    assert r.crossover is not None and r.crossover["accepted"] is True
    assert r.crossover["kkt_error_after"] <= r.crossover["kkt_error_before"]
    # A vertex: each of x0, x1 is at a bound, not split across the edge.
    assert sorted(np.round(r.x[:2], 12)) == [0.0, 1.0]
    assert r.obj == pytest.approx(interior.obj, abs=1e-8)


def test_crossover_is_a_noop_on_a_qp():
    r = solve_qp(**SKEWED, qp_crossover=True)
    assert r.status == "optimal" and r.crossover is None


def test_presolve_reduces_and_postsolves():
    plain = solve_qp(**LP)
    r = solve_qp(**LP, qp_presolve="yes")
    assert r.status == "optimal"
    pre = r.presolve
    assert pre["outcome"] == "reduced"
    assert pre["orig_vars"] == 3 and pre["reduced_vars"] < 3
    # Postsolved back to the caller's variables, the fixed column restored.
    assert r.x.shape == (3,) and r.x[2] == 2.0
    assert r.obj == pytest.approx(plain.obj, abs=1e-7)


def test_presolve_proves_infeasibility_without_running_the_engine():
    kw = dict(
        c=np.array([1.0]),
        G=np.array([[1.0]]),
        h=np.array([-5.0]),
        lb=np.array([0.0]),
        ub=np.array([1.0]),
    )
    r = solve_qp(**kw, qp_presolve=True)
    assert r.status == "primal_infeasible" and r.iters == 0
    assert r.presolve["outcome"] == "infeasible" and r.presolve["reason"]
    # The engine alone agrees, which is what makes the shortcut sound here.
    assert solve_qp(**kw).status == "primal_infeasible"


def test_presolve_reaches_the_active_set_engine_too():
    r = solve_qp(**LP, method="active-set", qp_presolve=True)
    assert r.status == "optimal" and r.presolve["outcome"] == "reduced"


def test_hsde_off_runs_the_direct_driver():
    on = solve_qp(**LP)
    off = solve_qp(**LP, qp_hsde="no")
    assert on.tau is not None and on.kappa is not None
    assert off.tau is None and off.kappa is None
    assert off.status == "optimal"
    assert off.obj == pytest.approx(on.obj, abs=1e-7)


def test_hsde_off_still_certifies_infeasibility():
    r = solve_qp(
        c=np.array([1.0, 1.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([5.0]),
        lb=np.zeros(2),
        ub=np.ones(2),
        qp_hsde=False,
    )
    assert r.status == "primal_infeasible"


def test_equilibrate_changes_the_direct_drivers_path():
    on = solve_qp(**SKEWED, qp_hsde=False, collect_iterates=True)
    off = solve_qp(
        **SKEWED, qp_hsde=False, qp_equilibrate=False, collect_iterates=True
    )
    assert on.status == off.status == "optimal"
    assert on.x.tobytes() != off.x.tobytes()
    assert on.obj == pytest.approx(off.obj, rel=1e-8)


def test_reg_reaches_the_engine():
    base = solve_qp(**LP, collect_iterates=True)
    r = solve_qp(**LP, qp_reg=1e-6, collect_iterates=True)
    assert r.status == "optimal"
    assert r.x.tobytes() != base.x.tobytes()
    assert r.obj == pytest.approx(base.obj, abs=1e-6)


@pytest.mark.parametrize(
    "kw, needle",
    [
        (dict(qp_reg=-1.0), "qp_reg"),
        (dict(qp_reg=float("nan")), "qp_reg"),
        (dict(qp_reg=float("inf")), "qp_reg"),
        (dict(qp_reg=True), "qp_reg"),
        (dict(qp_reg="1e-8"), "qp_reg"),
        (dict(qp_hsde=1), "qp_hsde"),
        (dict(qp_presolve="true"), "qp_presolve"),
        (dict(qp_crossover="on"), "qp_crossover"),
        (dict(qp_equilibrate="YES"), "qp_equilibrate"),
    ],
)
def test_bad_values_are_refused_by_name(kw, needle):
    with pytest.raises(ValueError, match=needle):
        solve_qp(**LP, **kw)


@pytest.mark.parametrize(
    "kw", [dict(qp_reg=1e-8), dict(qp_hsde=True), dict(qp_crossover=False)]
)
def test_active_set_refuses_the_ipm_only_options(kw):
    (name,) = kw
    with pytest.raises(ValueError, match=f"`{name}`.*active-set"):
        solve_qp(**LP, method="active-set", **kw)


@pytest.mark.parametrize("name", ["qp_presolve", "qp_crossover"])
def test_warm_start_refuses_presolve_and_crossover(name):
    warm = solve_qp(**LP)
    with pytest.raises(ValueError, match=name):
        solve_qp(**LP, warm_start=warm, **{name: True})
    # Asking for what the warm path does anyway is fine.
    assert solve_qp(**LP, warm_start=warm, **{name: False}).status == "optimal"


def test_the_binding_validates_independently_of_the_wrapper():
    prob = _build(None, np.array([1.0]), None, None, None, None, [0.0], [1.0])
    with pytest.raises(ValueError, match="qp_reg"):
        _pounce.solve_qp(prob, qp_reg=-1.0)
    with pytest.raises(ValueError, match="qp_hsde"):
        _pounce.solve_qp(prob, method="active-set", qp_hsde=False)
