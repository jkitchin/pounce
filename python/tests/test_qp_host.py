"""Host-level convex QP surface (``pounce.qp`` + the top-level re-exports).

These cover the ergonomics that bring the QP path toward NLP parity:
top-level discoverability, the final KKT ``residuals`` and opt-in iterate
trace on :class:`~pounce.qp.QpResult`, the multiple-RHS host wrapper, and
the catchable error on a malformed cone partition.
"""

import numpy as np
import pytest

import pounce
from pounce.qp import QpResult, solve_qp, solve_qp_multi_rhs, solve_socp


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
    r = solve_socp(c=[1.0, 0.0, 0.0], G=-np.eye(3), h=[0.0, -2.0, 1.0],
                   cones=[("soc", 3)])
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
