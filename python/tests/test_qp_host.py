"""Host-level convex QP surface (``pounce.qp`` + the top-level re-exports).

These cover the ergonomics that bring the QP path toward NLP parity:
top-level discoverability, the final KKT ``residuals`` and opt-in iterate
trace on :class:`~pounce.qp.QpResult`, the multiple-RHS host wrapper, and
the catchable error on a malformed cone partition.
"""

import numpy as np
import pytest

import pounce
from pounce.qp import (
    QpFactorization,
    QpResult,
    QpSensitivity,
    solve_qp,
    solve_qp_batch,
    solve_qp_multi_rhs,
    solve_socp,
)


def test_qp_is_reexported_at_top_level():
    # The QP entry points are reachable from ``pounce.*`` (like ``Problem``),
    # not only from ``pounce.qp.*``.
    for name in (
        "solve_qp",
        "solve_socp",
        "solve_qp_batch",
        "solve_qp_multi_rhs",
        "QpResult",
        "QpFactorization",
    ):
        assert hasattr(pounce, name), name
    assert pounce.solve_qp is solve_qp


def test_qp_module_star_import_has_no_dangling_names():
    # Every name advertised in ``__all__`` must actually exist (regression:
    # ``QpProblem`` was listed but never defined, breaking ``import *``).
    import pounce.qp as qp

    missing = [n for n in qp.__all__ if not hasattr(qp, n)]
    assert missing == []


def test_residuals_attached_and_kkt_error():
    # min x0²+x1² −3x0 −4x1  s.t.  0 ≤ x ≤ 1  → clamps to (1, 1).
    r = solve_qp(P=np.diag([2.0, 2.0]), c=[-3.0, -4.0], lb=[0, 0], ub=[1, 1])
    assert r.status == "optimal"
    assert isinstance(r, QpResult)
    assert set(r.residuals) == {
        "primal_infeasibility",
        "dual_infeasibility",
        "complementarity",
        "kkt_error",
    }
    assert r.kkt_error == r.residuals["kkt_error"]
    assert r.kkt_error < 1e-6


def test_iterate_trace_is_opt_in():
    kw = dict(P=np.diag([2.0, 2.0]), c=[-3.0, -4.0], lb=[0, 0], ub=[1, 1])
    assert solve_qp(**kw).iterates == []  # default: no trace
    traced = solve_qp(**kw, collect_iterates=True)
    # N interior-point iterations log N+1 records: one per iteration plus a
    # terminal record at the converged iterate (matching the NLP trace's
    # N+1 convention, so the trace always ends at the optimum).
    assert len(traced.iterates) == traced.iters + 1
    first = traced.iterates[0]
    assert set(first) == {
        "iter",
        "objective",
        "primal_infeasibility",
        "dual_infeasibility",
        "mu",
        "alpha_primal",
        "alpha_dual",
    }
    # The duality measure decreases over the run.
    assert traced.iterates[-1]["mu"] < traced.iterates[0]["mu"]


def test_conic_solve_has_no_orthant_residuals():
    # SOCP slack lives in a non-orthant cone: orthant residuals don't apply.
    r = solve_socp(
        c=[1.0, 0.0, 0.0], G=-np.eye(3), h=[0.0, -2.0, 1.0], cones=[("soc", 3)]
    )
    assert r.status == "optimal"
    assert r.residuals is None
    assert r.kkt_error is None


def test_solve_qp_multi_rhs_host_matches_individual():
    # Shared box structure, swept objective: each solve matches a one-off.
    cs = [[-3.0, -4.0], [1.0, 1.0], [-1.0, 2.0], [0.0, 0.0]]
    sweep = solve_qp_multi_rhs(P=np.diag([2.0, 2.0]), lb=[0, 0], ub=[1, 1], cs=cs)
    assert len(sweep) == len(cs)
    for c, r in zip(cs, sweep):
        one = solve_qp(P=np.diag([2.0, 2.0]), c=c, lb=[0, 0], ub=[1, 1])
        assert r.status == "optimal"
        np.testing.assert_allclose(r.x, one.x, atol=1e-6)
        assert r.residuals is not None  # multi-RHS still reports residuals


def test_solve_qp_multi_rhs_requires_cs():
    with pytest.raises(ValueError):
        solve_qp_multi_rhs(P=np.eye(2), cs=[])


def test_malformed_cone_partition_raises_valueerror():
    # An exp cone is always 3 rows; declaring it over a 2-row G is a usage
    # error and must raise a catchable ValueError (not panic across FFI).
    with pytest.raises(ValueError):
        solve_socp(c=[1.0, 0.0], G=-np.eye(2), h=[0.0, 0.0], cones=[("exp", 2)])


# --------------------------------------------------------------------------
# issue #112 — the indefinite-P guard must cover EVERY QP entry point, not
# only solve_qp. Pre-fix, an indefinite P fed to solve_qp_batch /
# solve_qp_multi_rhs / QpFactorization / QpSensitivity / solve_socp produced a
# silently-wrong status="optimal" (or a constructed handle) instead of an
# error. (Code review M31.)
# --------------------------------------------------------------------------

# Indefinite Hessian: eigenvalues +1, -1. Box bounds keep the convex IPM from
# diverging so, absent the guard, it would return a concrete status.
_P_INDEF = np.array([[1.0, 0.0], [0.0, -1.0]])
_C2 = np.zeros(2)
_LB = -np.ones(2)
_UB = np.ones(2)


def test_solve_qp_batch_rejects_indefinite_p():
    with pytest.raises(ValueError, match="positive semidefinite"):
        solve_qp_batch([{"P": _P_INDEF, "c": _C2, "lb": _LB, "ub": _UB}])


def test_solve_qp_multi_rhs_rejects_indefinite_p():
    with pytest.raises(ValueError, match="positive semidefinite"):
        solve_qp_multi_rhs(P=_P_INDEF, c=_C2, lb=_LB, ub=_UB, cs=[_C2])


def test_qp_factorization_rejects_indefinite_p():
    with pytest.raises(ValueError, match="positive semidefinite"):
        QpFactorization(P=_P_INDEF, c=_C2, lb=_LB, ub=_UB)


def test_qp_sensitivity_rejects_indefinite_p():
    with pytest.raises(ValueError, match="positive semidefinite"):
        QpSensitivity(P=_P_INDEF, c=_C2, A=[[1.0, 1.0]], b=[0.0])


def test_solve_socp_rejects_indefinite_p():
    with pytest.raises(ValueError, match="positive semidefinite"):
        solve_socp(P=_P_INDEF, c=_C2, G=-np.eye(2), h=np.ones(2), cones=[("nonneg", 2)])


def test_check_psd_false_bypasses_guard_everywhere():
    # check_psd=False must skip the guard on every entry point — the escape
    # hatch for a caller who knows P is PSD (or wants the nonconvex behavior)
    # and is avoiding the O(n^3) eigenvalue cost. None of these should raise.
    solve_qp_batch([{"P": _P_INDEF, "c": _C2, "lb": _LB, "ub": _UB}], check_psd=False)
    solve_qp_multi_rhs(P=_P_INDEF, c=_C2, lb=_LB, ub=_UB, cs=[_C2], check_psd=False)
    QpFactorization(P=_P_INDEF, c=_C2, lb=_LB, ub=_UB, check_psd=False)
    QpSensitivity(P=_P_INDEF, c=_C2, A=[[1.0, 1.0]], b=[0.0], check_psd=False)
    solve_socp(
        P=_P_INDEF,
        c=_C2,
        G=-np.eye(2),
        h=np.ones(2),
        cones=[("nonneg", 2)],
        check_psd=False,
    )


def test_psd_p_still_solves_on_all_entry_points():
    # A genuinely PSD P must pass the guard unscathed on every entry point.
    P = 2.0 * np.eye(2)  # PSD
    assert (
        solve_qp_batch([{"P": P, "c": _C2, "lb": _LB, "ub": _UB}])[0].status
        == "optimal"
    )
    assert (
        solve_qp_multi_rhs(P=P, c=_C2, lb=_LB, ub=_UB, cs=[_C2])[0].status == "optimal"
    )
    QpFactorization(P=P, c=_C2, lb=_LB, ub=_UB)  # constructs, no raise
    s = QpSensitivity(P=P, c=_C2, A=[[1.0, 1.0]], b=[2.0])
    np.testing.assert_allclose(s.x, [1.0, 1.0], atol=1e-6)
