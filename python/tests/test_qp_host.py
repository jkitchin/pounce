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


def test_conic_solve_reports_cone_aware_residuals():
    # A SOCP slack lives in a non-orthant cone, so its residuals must be
    # measured against that cone. They used to be omitted entirely (the
    # orthant reading being meaningless); reporting them is what makes a
    # conic solve's convergence checkable at all (pounce#209).
    r = solve_socp(
        c=[1.0, 0.0, 0.0], G=-np.eye(3), h=[0.0, -2.0, 1.0], cones=[("soc", 3)]
    )
    assert r.status == "optimal"
    assert r.residuals is not None
    assert set(r.residuals) == {
        "primal_infeasibility",
        "dual_infeasibility",
        "complementarity",
        "kkt_error",
    }
    # Converged ⇒ every component is at tolerance. In particular the primal
    # residual measures cone *membership*: an orthant reading of this same
    # point is nonzero, since the SOC rows individually have `Gx > h`.
    assert r.kkt_error < 1e-6
    assert r.residuals["primal_infeasibility"] < 1e-6


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


# --- gh #279: duplicate COO entries must be summed by the PSD guard ------


def _dup_indefinite_coo():
    """P whose PSD-ness depends on the duplicate convention.

    Entries (1,0) appear twice at 1.5 each. Under the COO **sum** convention
    — which is what scipy documents and what the solver applies — the
    symmetric matrix is [[2, 3], [3, 2]], eigenvalues [-1, 5]: indefinite.
    Under last-duplicate-wins it would be [[2, 1.5], [1.5, 2]], eigenvalues
    [0.5, 3.5]: positive definite.
    """
    from scipy.sparse import coo_matrix

    return coo_matrix(
        ([2.0, 2.0, 1.5, 1.5], ([0, 1, 1, 1], [0, 1, 0, 0])), shape=(2, 2)
    )


def test_check_psd_sums_duplicate_coo_entries():
    """The guard must validate the matrix the solver actually solves.

    Before #279 ``_min_eig_lower_coo`` assigned rather than accumulated, so
    it validated a *different* matrix: this indefinite P passed the guard and
    ``solve_qp`` returned ``status="optimal"`` at a saddle point with
    objective ~0, while the true minimum over the box is -100.
    """
    P = _dup_indefinite_coo()
    with pytest.raises(ValueError, match="positive semidefinite"):
        solve_qp(
            P=P,
            c=np.zeros(2),
            lb=np.full(2, -10.0),
            ub=np.full(2, 10.0),
            check_psd=True,
        )


def test_check_psd_duplicate_coo_matches_dense_verdict():
    """Sparse-with-duplicates and its dense equivalent must agree.

    The dense form of the same matrix was always rejected correctly; the
    sparse form was not. Same mathematical input, same verdict.
    """
    P = _dup_indefinite_coo()
    dense = P.toarray()
    dense = np.tril(dense) + np.tril(dense, -1).T  # solver's lower-triangle read

    for form in (P, dense):
        with pytest.raises(ValueError, match="positive semidefinite"):
            solve_qp(
                P=form,
                c=np.zeros(2),
                lb=np.full(2, -10.0),
                ub=np.full(2, 10.0),
                check_psd=True,
            )


def test_check_psd_does_not_double_count_duplicate_diagonal():
    """Accumulating must not double the diagonal.

    The mirror write ``M[ci, ri]`` has to be skipped when ``ri == ci``, or a
    diagonal entry is counted twice — which would inflate the spectrum and
    could mask a genuine indefiniteness. Here two 1.0 entries on (0,0) sum to
    2.0, so the solve is ``min x0^2 - 2 x0 + 0.25 x1^2 - x1`` with optimum
    (1, 2); a doubled diagonal would move it to (0.5, 2).
    """
    from scipy.sparse import coo_matrix

    P = coo_matrix(([1.0, 1.0, 0.5], ([0, 0, 1], [0, 0, 1])), shape=(2, 2))
    r = solve_qp(
        P=P,
        c=np.array([-2.0, -1.0]),
        lb=np.full(2, -10.0),
        ub=np.full(2, 10.0),
        check_psd=True,
    )
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [1.0, 2.0], atol=1e-6)


# --- gh #275: sign-aware infinite bounds ---------------------------------


def test_solve_qp_rejects_bounds_no_finite_value_can_satisfy():
    """``lb = +inf`` / ``ub = -inf`` are constraints, not "absent" bounds.

    The solver's presence test (``lb > -BOUND_INF``, ``ub < BOUND_INF``) is
    sign-agnostic, so these were dropped as if unbounded and the solve
    returned ``status="optimal"`` at a point violating the stated bound by an
    infinite margin.
    """
    for lb, ub, pat in (
        ([np.inf], [np.inf], r"`lb\[0\]` is inf"),
        ([np.inf], [5.0], r"`lb\[0\]` is inf"),
        ([-np.inf], [-np.inf], r"`ub\[0\]` is -inf"),
        ([-5.0], [-np.inf], r"`ub\[0\]` is -inf"),
    ):
        with pytest.raises(ValueError, match=pat):
            solve_qp(
                P=np.eye(1), c=np.zeros(1), lb=np.array(lb), ub=np.array(ub)
            )


def test_solve_qp_still_accepts_legitimate_infinite_bounds():
    """±inf on the absent side is the documented one-sided encoding."""
    r = solve_qp(
        P=np.eye(1),
        c=np.array([-3.0]),
        lb=np.array([-np.inf]),
        ub=np.array([np.inf]),
    )
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [3.0], atol=1e-6)


def test_solve_qp_finite_reversed_bounds_still_report_infeasible():
    """Unchanged: a finite reversed box is a *status*, not an exception.

    Only the degenerate infinite spellings — which previously produced a
    silently wrong "optimal" — are rejected up front.
    """
    r = solve_qp(P=np.eye(1), c=np.zeros(1), lb=np.array([1.0]), ub=np.array([0.0]))
    assert r.status == "primal_infeasible"


def test_solve_qp_batch_inherits_the_bound_guard():
    with pytest.raises(ValueError, match=r"`lb\[0\]` is inf"):
        solve_qp_batch(
            [{"P": np.eye(1), "c": np.zeros(1), "lb": np.array([np.inf]),
              "ub": np.array([np.inf])}]
        )


# --- tol / max_iter option validation (gh #277) -----------------------------

# The convex entry points used to apply *no* validation to `tol` while every
# other pounce surface (NLP `minimize`, the CLI, `sos_minimize`) rejects a
# non-positive / non-finite tolerance with OPTION_INVALID. An unsatisfiable
# `tol` (<= 0, NaN, Inf) silently burned every iteration; a huge finite `tol`
# (1e300) short-circuited at the non-stationary starting iterate and returned a
# wrong point labeled "optimal". `max_iter=-5` leaked a raw PyO3 OverflowError.

# A small bound-constrained QP: min 1/2 x'x - 1'x, optimum x* = (1, 1).
_QP277 = dict(
    P=np.eye(2),
    c=np.array([-1.0, -1.0]),
    lb=np.array([-10.0, -10.0]),
    ub=np.array([10.0, 10.0]),
)
_BAD_TOLS = [0.0, -1.0, float("nan"), float("inf"), 1e300, 1.0]


@pytest.mark.parametrize("bad", _BAD_TOLS)
def test_solve_qp_rejects_bad_tol(bad):
    with pytest.raises(ValueError, match=r"`tol` must be a finite positive number"):
        solve_qp(**_QP277, tol=bad)


@pytest.mark.parametrize("bad", _BAD_TOLS)
def test_solve_socp_rejects_bad_tol(bad):
    # min t s.t. (t, x - x*) in SOC — a well-formed conic problem.
    with pytest.raises(ValueError, match=r"`tol` must be a finite positive number"):
        solve_socp(
            c=[1.0, 0.0, 0.0],
            G=-np.eye(3),
            h=[0.0, -2.0, 1.0],
            cones=[("soc", 3)],
            tol=bad,
        )


@pytest.mark.parametrize("bad", _BAD_TOLS)
def test_solve_qp_batch_rejects_bad_tol(bad):
    with pytest.raises(ValueError, match=r"`tol` must be a finite positive number"):
        solve_qp_batch([_QP277], tol=bad)


@pytest.mark.parametrize("bad", _BAD_TOLS)
def test_solve_qp_multi_rhs_rejects_bad_tol(bad):
    with pytest.raises(ValueError, match=r"`tol` must be a finite positive number"):
        solve_qp_multi_rhs(**_QP277, cs=[_QP277["c"]], tol=bad)


@pytest.mark.parametrize("bad", _BAD_TOLS)
def test_qp_factorization_rejects_bad_tol(bad):
    with pytest.raises(ValueError, match=r"`tol` must be a finite positive number"):
        QpFactorization(**_QP277, tol=bad)


@pytest.mark.parametrize("bad", _BAD_TOLS)
def test_qp_sensitivity_rejects_bad_tol(bad):
    with pytest.raises(ValueError, match=r"`tol` must be a finite positive number"):
        QpSensitivity(P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[2.0], tol=bad)


def test_huge_tol_never_returns_a_wrong_optimal():
    """The exact repro from gh #277: `tol=1e300` used to return
    `status="optimal"` at x=(0,0) (a non-stationary point, kkt_error=1.0)
    after 0 iterations. It must now be rejected outright, so no wrong
    "optimal" can escape."""
    with pytest.raises(ValueError, match=r"`tol` must be a finite positive number"):
        solve_qp(**_QP277, tol=1e300)


@pytest.mark.parametrize("bad", [-5, -1, 0, 2.5])
def test_solve_qp_rejects_bad_max_iter(bad):
    # A negative int previously leaked a raw PyO3 OverflowError from the usize
    # binding; every invalid value must now be a clear, named ValueError.
    with pytest.raises(ValueError, match=r"`max_iter` must be a positive integer"):
        solve_qp(**_QP277, max_iter=bad)


def test_all_convex_entry_points_reject_negative_max_iter():
    """The raw-OverflowError leak must be closed on *every* convex surface that
    accepts `max_iter`, not only `solve_qp`."""
    with pytest.raises(ValueError, match=r"`max_iter` must be a positive integer"):
        solve_socp(
            c=[1.0, 0.0, 0.0], G=-np.eye(3), h=[0.0, -2.0, 1.0],
            cones=[("soc", 3)], max_iter=-5,
        )
    with pytest.raises(ValueError, match=r"`max_iter` must be a positive integer"):
        solve_qp_batch([_QP277], max_iter=-5)
    with pytest.raises(ValueError, match=r"`max_iter` must be a positive integer"):
        solve_qp_multi_rhs(**_QP277, cs=[_QP277["c"]], max_iter=-5)
    with pytest.raises(ValueError, match=r"`max_iter` must be a positive integer"):
        QpFactorization(**_QP277, max_iter=-5)
    with pytest.raises(ValueError, match=r"`max_iter` must be a positive integer"):
        QpSensitivity(P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[2.0], max_iter=-5)


def test_valid_tol_and_max_iter_still_solve_to_optimum():
    """A legitimate tight `tol` and finite `max_iter` are untouched: the QP
    still solves to the known optimum x* = (1, 1)."""
    r = solve_qp(**_QP277, tol=1e-8, max_iter=100)
    assert r.status == "optimal"
    np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-6)
    assert r.kkt_error is not None and r.kkt_error <= 1e-6


def test_minimize_qp_route_rejects_bad_tol():
    """The convex facade (`minimize(solver_selection="qp-ipm")`) inherits the
    guard, so the `tol=1e300` mislabel no longer propagates to a
    `success=True, nit=0` OptimizeResult (gh #277)."""
    import pounce

    def f(x):
        return 0.5 * (x @ x) - x.sum()

    for bad in (0.0, 1e300, float("nan")):
        with pytest.raises(ValueError, match=r"`tol` must be a finite positive number"):
            pounce.minimize(
                f,
                x0=np.array([5.0, 5.0]),
                jac=lambda x: x - 1.0,
                solver_selection="qp-ipm",
                tol=bad,
            )
