"""Smoke + correctness tests for the cyipopt-shaped Problem API."""

import numpy as np
import pytest

import pounce


class HS071:
    def objective(self, x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def gradient(self, x):
        return np.array(
            [
                x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
                x[0] * x[3],
                x[0] * x[3] + 1.0,
                x[0] * (x[0] + x[1] + x[2]),
            ]
        )

    def constraints(self, x):
        return np.array([np.prod(x), np.dot(x, x)])

    def jacobianstructure(self):
        return (np.repeat([0, 1], 4), np.tile([0, 1, 2, 3], 2))

    def jacobian(self, x):
        return np.array(
            [
                x[1] * x[2] * x[3],
                x[0] * x[2] * x[3],
                x[0] * x[1] * x[3],
                x[0] * x[1] * x[2],
                2 * x[0],
                2 * x[1],
                2 * x[2],
                2 * x[3],
            ]
        )


def test_hs071_lbfgs():
    """L-BFGS path (no hessian methods on the user object)."""
    prob = pounce.Problem(
        n=4,
        m=2,
        problem_obj=HS071(),
        lb=[1.0] * 4,
        ub=[5.0] * 4,
        cl=[25.0, 40.0],
        cu=[2e19, 40.0],
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(info["obj_val"], 17.0140172, rtol=1e-5)
    np.testing.assert_allclose(x, [1.0, 4.7430, 3.8211, 1.3794], atol=1e-3)


def test_diff_handoff_masks_in_info():
    """The DiffHandoff active-set masks ride out in the info dict
    (dev-notes/diff-handoff-contract.md), computed once in the producer.

    HS071's known optimum x* ≈ (1, 4.743, 3.821, 1.379) has:
      * x0 pinned at its lower bound (1.0) → pinned_vars[0] is True;
        x1..x3 interior → not pinned;
      * constraint 0 (prod ≥ 25) binding and constraint 1 (sumsq = 40)
        an equality → both active.
    """
    prob = pounce.Problem(
        n=4,
        m=2,
        problem_obj=HS071(),
        lb=[1.0] * 4,
        ub=[5.0] * 4,
        cl=[25.0, 40.0],
        cu=[2e19, 40.0],
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"

    pinned = np.asarray(info["pinned_vars"])
    active_cons = np.asarray(info["active_constraints"])
    assert pinned.dtype == bool and pinned.shape == (4,)
    assert active_cons.dtype == bool and active_cons.shape == (2,)
    assert info["active_tol"] == 1e-6

    # x0 sits on its lower bound; x1..x3 are interior.
    assert bool(pinned[0]) is True
    assert not pinned[1:].any()
    # Both constraints active (binding inequality + equality).
    assert active_cons.all()

    # The masks are consistent with the raw multipliers they summarize.
    tol = info["active_tol"]
    zl = np.asarray(info["mult_x_L"])
    zu = np.asarray(info["mult_x_U"])
    np.testing.assert_array_equal(pinned, (zl > tol) | (zu > tol))


def test_problem_attributes():
    prob = pounce.Problem(
        n=2,
        m=0,
        problem_obj=type(
            "P",
            (),
            {
                "objective": staticmethod(lambda x: float(np.sum(x * x))),
                "gradient": staticmethod(lambda x: 2 * np.asarray(x, dtype=float)),
            },
        )(),
    )
    assert prob.n == 2
    assert prob.m == 0
    assert prob.has_hessian is False


def test_unconstrained_quadratic():
    """min ||x - target||² → x* = target."""
    target = np.array([1.0, 2.0, -3.0, 4.5])

    class Quad:
        def objective(self, x):
            d = x - target
            return float(d @ d)

        def gradient(self, x):
            return 2.0 * (x - target)

    prob = pounce.Problem(n=4, m=0, problem_obj=Quad())
    prob.add_option("tol", 1e-10)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.zeros(4))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, target, atol=1e-6)


# --------------------------------------------------------------------------
# issue #180 item 1 — caller-supplied KKT ordering (FERAL External).
# --------------------------------------------------------------------------
def _quad_problem(n, target):
    class Quad:
        def objective(self, x):
            d = x - target
            return float(d @ d)

        def gradient(self, x):
            return 2.0 * (x - target)

    prob = pounce.Problem(n=n, m=0, problem_obj=Quad())
    prob.add_option("tol", 1e-10)
    prob.add_option("print_level", 0)
    return prob


def test_external_ordering_matches_default_and_round_trips():
    """A caller-supplied permutation reaches the FERAL backend, solves to the
    same optimum as the default ordering, and round-trips through
    get_ordering / clear_ordering. For an unconstrained problem the augmented
    KKT system is n×n, so a length-n permutation is a valid bijection — and a
    valid ordering only changes fill/pivot order, never the solution."""
    target = np.array([1.0, 2.0, -3.0, 4.5])

    x_ref, _ = _quad_problem(4, target).solve(x0=np.zeros(4))

    prob = _quad_problem(4, target)
    assert prob.get_ordering() is None
    prob.set_ordering([3, 2, 1, 0])  # reversed — a valid 4×4 permutation
    np.testing.assert_array_equal(prob.get_ordering(), [3, 2, 1, 0])

    x, info = prob.solve(x0=np.zeros(4))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, x_ref, atol=1e-9)

    # Persistent config: a second solve still uses it; then it clears.
    x2, info2 = prob.solve(x0=np.ones(4))
    assert info2["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x2, x_ref, atol=1e-9)
    prob.clear_ordering()
    assert prob.get_ordering() is None


def test_external_ordering_negative_index_rejected():
    """Negative permutation entries are rejected at set time (0-based indices);
    the FERAL bijection check catches the rest at factorization."""
    prob = _quad_problem(3, np.zeros(3))
    with pytest.raises(ValueError):
        prob.set_ordering([0, -1, 2])


def test_external_ordering_wrong_length_fails_cleanly():
    """A permutation whose length ≠ the KKT dimension is rejected by FERAL as
    invalid input: the solve returns a non-success status rather than crashing
    or silently returning a wrong answer."""
    prob = _quad_problem(3, np.zeros(3))
    prob.set_ordering([0, 1, 2, 3, 4, 5])  # length 6 for a 3×3 KKT system
    _x, info = prob.solve(x0=np.ones(3))
    assert info["status_msg"] != "Solve_Succeeded"


# --------------------------------------------------------------------------
# issue #180 item 2 — Schur KKT partition (SchurAugSystemSolver).
# --------------------------------------------------------------------------
def _convex_eq_qp(target, A, b):
    """Convex equality-constrained QP: min ½‖x−target‖² s.t. A x = b. Exact
    Hessian (identity) so the solver takes the exact-Hessian path the Schur
    solver requires. All-equality ⇒ the KKT is `[[I+Σ, Aᵀ],[A, 0]]` of
    dimension `n + m`; the primal block is positive-definite, so the
    constraint-dual block `[n, n+m)` is a clean Schur set."""
    n = len(target)
    m = len(b)
    rows = np.repeat(np.arange(m), n)
    cols = np.tile(np.arange(n), m)
    Aflat = A.reshape(-1).astype(float)

    class QP:
        def objective(self, x):
            d = x - target
            return 0.5 * float(d @ d)

        def gradient(self, x):
            return x - target

        def constraints(self, x):
            return A @ x - b

        def jacobianstructure(self):
            return (rows, cols)

        def jacobian(self, x):
            return Aflat

        def hessianstructure(self):
            return (np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64))

        def hessian(self, x, lagrange, obj_factor):
            # Objective Hessian is I; constraints are linear ⇒ no contribution.
            return obj_factor * np.ones(n)

    prob = pounce.Problem(
        n=n,
        m=m,
        problem_obj=QP(),
        cl=list(b),
        cu=list(b),
    )
    prob.add_option("tol", 1e-9)
    prob.add_option("print_level", 0)
    return prob, n, m


def test_kkt_schur_block_matches_full_space_solve():
    """A Schur partition (the constraint-dual block) reaches the same optimum
    as the standard full-space solve, round-trips through the API, and the
    Schur path actually *engaged*.

    The last part used to be undetectable: the path falls back to the standard
    solver transparently, so a silently-disabled Schur path produced the same
    optimum. It once was disabled that way — the gate compared the *requested*
    linear solver against FERAL while the registry default "ma57" was recorded
    even on builds that substitute FERAL, so `set_kkt_schur_block()` was a
    no-op for every default user and this test still passed.
    `info["linear_solver"]["schur"]` is present only when the Schur backend
    factored, so it is now asserted directly.
    """
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    A = np.array([[1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]])
    b = np.array([3.0, 12.0])

    prob_ref, n, m = _convex_eq_qp(target, A, b)
    x_ref, info_ref = prob_ref.solve(x0=np.zeros(n))
    assert info_ref["status_msg"] == "Solve_Succeeded"

    prob, n, m = _convex_eq_qp(target, A, b)
    # KKT dim = n + m (no inequalities); the dual block is [n, n+m).
    schur = list(range(n, n + m))
    assert prob.get_kkt_schur_block() is None
    prob.set_kkt_schur_block(schur)
    np.testing.assert_array_equal(prob.get_kkt_schur_block(), schur)

    x, info = prob.solve(x0=np.zeros(n))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, x_ref, atol=1e-7)

    assert info_ref["linear_solver"]["schur"] is None
    sch = info["linear_solver"]["schur"]
    assert sch is not None, "Schur path fell back silently"
    assert (sch["n_eliminated"], sch["n_schur"]) == (n, m)
    assert sch["n_factors"] >= 1
    # Every Schur factorization factors the eliminated block once.
    assert info["linear_solver"]["n_factors"] >= sch["n_factors"]

    prob.clear_kkt_schur_block()
    assert prob.get_kkt_schur_block() is None


def test_kkt_schur_block_oversized_falls_back():
    """An oversized Schur block is rejected by the gate and the solve falls back
    to the full-space solver — still converging to the optimum."""
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    A = np.array([[1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]])
    b = np.array([3.0, 12.0])
    prob, n, m = _convex_eq_qp(target, A, b)
    # Almost the whole KKT (well past the 0.5 gate) → transparent fallback.
    prob.set_kkt_schur_block(list(range(n + m - 1)))
    x, info = prob.solve(x0=np.zeros(n))
    assert info["status_msg"] == "Solve_Succeeded"
    # The fallback is visible: no Schur breakdown, standard factorizations.
    assert info["linear_solver"]["schur"] is None
    assert info["linear_solver"]["n_factors"] >= 1

    prob_ref, _, _ = _convex_eq_qp(target, A, b)
    x_ref, _ = prob_ref.solve(x0=np.zeros(n))
    np.testing.assert_allclose(x, x_ref, atol=1e-7)


def test_info_linear_solver_reports_factorization_stats():
    """`info["linear_solver"]` carries the factorization counts plus the work
    and pivoting totals (structured-KKT Phase 0a), consistently with each
    other and keyed like the solve report's `linear_solver` object."""
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    A = np.array([[1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]])
    b = np.array([3.0, 12.0])
    prob, n, m = _convex_eq_qp(target, A, b)
    _x, info = prob.solve(x0=np.zeros(n))
    assert info["status_msg"] == "Solve_Succeeded"

    ls = info["linear_solver"]
    assert ls["solver_name"] == "feral"
    assert ls["n_factors"] >= 1
    assert ls["n_pattern_reuse"] + ls["n_pattern_changes"] == ls["n_factors"]
    # Fill of the final factor is bounded by the running maximum.
    assert ls["last_nnz_l"] / ls["last_nnz_a"] <= ls["max_fill_ratio"] + 1e-12
    # KKT inertia at a regular optimum: n positive, m negative, none zero.
    assert tuple(ls["last_inertia"]) == (n, m, 0)
    assert ls["total_factor_secs"] >= 0.0
    assert ls["total_factor_flops"] >= ls["last_factor_flops"] > 0.0
    assert ls["last_ordering"] in {"amd", "amf", "metis", "scotch", "kahip"}
    for k in ("total_delayed_cols", "total_two_by_two", "total_n_tiny"):
        assert isinstance(ls[k], int) and ls[k] >= 0
    assert ls["total_delayed_cols"] >= ls["last_delayed_cols"]
    assert ls["total_two_by_two"] >= ls["last_two_by_two"]
    assert ls["last_n_supernodes"] >= 1
    assert ls["last_max_front_rows"] >= 1
    assert ls["last_peak_bytes"] > 0
    assert ls["schur"] is None


def _infeasible_parabolas(ladder):
    """x0**2 - x1 + 1 = 0 and x1 + x0**2 = 0: their sum is 2*x0**2 + 1 = 0,
    so the problem is infeasible, the solve enters restoration, and it ends in
    ``Infeasible_Problem_Detected`` — a verdict the second-opinion ladder
    re-solves and, here, keeps.

    Nonlinear on purpose. On a linear infeasible pair every rung reproduces
    the original solve's factorization counts, so a summary taken from the
    last rejected rung is indistinguishable from the right one; here the last
    rung (``start_point_perturbation``) takes a different trajectory."""

    class P:
        def objective(self, x):
            return float(x @ x)

        def gradient(self, x):
            return 2.0 * x

        def constraints(self, x):
            return np.array([x[0] ** 2 - x[1] + 1.0, x[1] + x[0] ** 2])

        def jacobianstructure(self):
            return (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))

        def jacobian(self, x):
            return np.array([2.0 * x[0], -1.0, 2.0 * x[0], 1.0])

        def hessianstructure(self):
            return (np.array([0, 1, 1]), np.array([0, 0, 1]))

        def hessian(self, x, lagrange, obj_factor):
            return np.array([
                2.0 * obj_factor + 2.0 * lagrange[0] + 2.0 * lagrange[1],
                0.0,
                2.0 * obj_factor,
            ])

    prob = pounce.Problem(n=2, m=2, problem_obj=P(), cl=[0.0, 0.0], cu=[0.0, 0.0])
    prob.add_option("print_level", 0)
    if not ladder:
        # Every rung this verdict can open, or the "plain" solve is itself
        # a ladder run that ends on the same rung as the laddered one.
        for rung in ("feral_infeasibility_scaling_retry",
                     "infeasibility_mu_strategy_retry",
                     "infeasibility_perturbed_start_retry",
                     "feral_increase_quality_retry"):
            prob.add_option(rung, "no")
    _x, info = prob.solve(x0=np.array([0.5, 0.3]))
    return info


def test_linear_solver_summary_follows_the_kept_verdict():
    """Restoration factorizations are reported under ``restoration``, and when
    the ladder re-solves but keeps the original verdict, the summary is the
    original solve's — not the last rejected rung's. Every rung's solve resets
    and refills the summary, so before the ladder restored it, the verdict and
    statistics described one solve and ``linear_solver`` another."""
    plain = _infeasible_parabolas(ladder=False)
    laddered = _infeasible_parabolas(ladder=True)
    assert plain["status_msg"] == "Infeasible_Problem_Detected"
    assert laddered["status_msg"] == "Infeasible_Problem_Detected"
    assert plain["second_opinion"] is None, "the plain solve must not run the ladder"
    assert laddered["second_opinion"]["tried"], "the ladder must have run"
    assert laddered["second_opinion"]["promoted_by"] is None

    resto = plain["linear_solver"]["restoration"]
    assert resto is not None, "restoration factorizations were not reported"
    assert resto["n_factors"] >= plain["restoration_inner_iters"]
    assert resto["restoration"] is None

    a, b = plain["linear_solver"], laddered["linear_solver"]
    assert a["n_factors"] == b["n_factors"]
    assert a["restoration"]["n_factors"] == b["restoration"]["n_factors"]
    assert a["total_two_by_two"] == b["total_two_by_two"]


def test_kkt_schur_block_negative_index_rejected():
    prob, n, _m = _convex_eq_qp(
        np.zeros(3), np.array([[1.0, 1.0, 1.0]]), np.array([1.0])
    )
    with pytest.raises(ValueError):
        prob.set_kkt_schur_block([3, -1])


# --------------------------------------------------------------------------
# issue M32 — the `intermediate` callback return value must follow cyipopt
# truthiness. A falsy return (False, 0, 0.0, []) requests a stop; truthy
# continues. Pre-fix, the bridge used a strict `extract::<bool>().unwrap_or
# (true)`, so a valid falsy int `0` was coerced to "continue" and the user's
# stop was silently ignored. (Code review M32.)
# --------------------------------------------------------------------------


def _stopper(return_value):
    """A tiny well-conditioned problem (min (x-3)^2) whose `intermediate`
    returns ``return_value`` once at least one iteration has elapsed."""

    class P:
        def __init__(self):
            self.iters = []

        def objective(self, x):
            return float((x[0] - 3.0) ** 2)

        def gradient(self, x):
            return np.array([2.0 * (x[0] - 3.0)])

        def intermediate(self, **kw):
            self.iters.append(kw["iter_count"])
            return 1 if kw["iter_count"] < 1 else return_value

    obj = P()
    prob = pounce.Problem(
        n=1, m=0, problem_obj=obj, lb=[-10.0], ub=[10.0], cl=[], cu=[]
    )
    prob.add_option("print_level", 0)
    return obj, prob


@pytest.mark.parametrize("falsy", [0, False, 0.0, []])
def test_intermediate_falsy_return_stops(falsy):
    # Each cyipopt-falsy value must abort with User_Requested_Stop. Pre-fix,
    # the int/float/list values slipped through `extract::<bool>` and the
    # solve ran to Solve_Succeeded (only `False` stopped).
    obj, prob = _stopper(falsy)
    x, info = prob.solve(x0=np.array([-5.0]))
    assert info["status_msg"] == "User_Requested_Stop"
    # It really stopped early — never reached the optimum x* = 3.
    assert not np.isclose(x[0], 3.0, atol=1e-3)


@pytest.mark.parametrize("truthy", [1, True, 0.5, [0]])
def test_intermediate_truthy_return_continues(truthy):
    # The mirror image: a truthy return keeps iterating to convergence.
    obj, prob = _stopper(truthy)
    prob.add_option("tol", 1e-8)
    x, info = prob.solve(x0=np.array([-5.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x[0], 3.0, atol=1e-4)


def test_intermediate_no_return_continues():
    # A callback that returns None (the common "just observe" case) must NOT
    # be read as a stop.
    class P:
        def objective(self, x):
            return float((x[0] - 3.0) ** 2)

        def gradient(self, x):
            return np.array([2.0 * (x[0] - 3.0)])

        def intermediate(self, **kw):
            return None

    prob = pounce.Problem(
        n=1, m=0, problem_obj=P(), lb=[-10.0], ub=[10.0], cl=[], cu=[]
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([-5.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x[0], 3.0, atol=1e-4)


def test_intermediate_exception_aborts_with_user_stop():
    # A raising `intermediate` aborts the solve (User_Requested_Stop) rather
    # than crashing across the FFI boundary; post-fix it also logs a trace
    # line (verified manually — the log goes through the Rust subscriber).
    class P:
        def objective(self, x):
            return float((x[0] - 3.0) ** 2)

        def gradient(self, x):
            return np.array([2.0 * (x[0] - 3.0)])

        def intermediate(self, **kw):
            raise RuntimeError("boom from intermediate")

    prob = pounce.Problem(
        n=1, m=0, problem_obj=P(), lb=[-10.0], ub=[10.0], cl=[], cu=[]
    )
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([-5.0]))
    assert info["status_msg"] == "User_Requested_Stop"


def _noncontiguous(a):
    """A non-contiguous float64 view holding the values of ``a`` (a strided
    slice of a 2x-oversized buffer)."""
    a = np.asarray(a, dtype=float)
    buf = np.empty(a.size * 2, dtype=float)
    buf[::2] = a
    view = buf[::2]
    assert not view.flags["C_CONTIGUOUS"]
    return view


def test_noncontiguous_float64_arrays_are_copied_not_rejected():
    """L49: valid non-contiguous float64 arrays (strided views) must be
    copied, not rejected with "array is not contiguous". This exercises both
    decode paths — ``extract_f64_vec`` (bounds + ``x0``) and
    ``copy_pyarray_into`` (the gradient / constraints / Jacobian callback
    returns)."""

    class NonContigHS071(HS071):
        def gradient(self, x):
            return _noncontiguous(super().gradient(x))

        def constraints(self, x):
            return _noncontiguous(super().constraints(x))

        def jacobian(self, x):
            return _noncontiguous(super().jacobian(x))

    prob = pounce.Problem(
        n=4,
        m=2,
        problem_obj=NonContigHS071(),
        lb=_noncontiguous([1.0] * 4),
        ub=_noncontiguous([5.0] * 4),
        cl=_noncontiguous([25.0, 40.0]),
        cu=_noncontiguous([2e19, 40.0]),
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=_noncontiguous([1.0, 5.0, 5.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(info["obj_val"], 17.0140172, rtol=1e-5)
    np.testing.assert_allclose(x, [1.0, 4.7430, 3.8211, 1.3794], atol=1e-3)


def test_negative_obj_scaling_factor_maximizes():
    """obj_scaling_factor < 0 means maximize (upstream Ipopt semantics).

    Regression for the pounce#128 follow-up: the option was registered
    but never read, so the IPM minimized the unscaled objective and a
    concave maximization diverged (Diverging_Iterates) instead of
    converging to the maximizer.
    """

    class ConcaveBump:
        def objective(self, x):
            return -((x[0] - 1.0) ** 2)

        def gradient(self, x):
            return np.array([-2.0 * (x[0] - 1.0)])

    prob = pounce.Problem(n=1, m=0, problem_obj=ConcaveBump(),
                          lb=[-1e19], ub=[1e19])
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    prob.add_option("obj_scaling_factor", -1.0)
    x, info = prob.solve(x0=np.array([0.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, [1.0], atol=1e-6)
    # The reported objective is the user's (unscaled) value at the max.
    np.testing.assert_allclose(info["obj_val"], 0.0, atol=1e-8)


# --------------------------------------------------------------------------
# issue #276 — integer options outside the signed-32-bit Index range were
# silently *wrapped* by an `i as i32` cast, so e.g. `max_iter = 2**32 + 3`
# quietly ran 3 iterations instead of erroring. They must now raise, quoting
# the value the user passed and naming the option, matching the CLI / Pyomo.
# --------------------------------------------------------------------------
def _quad2():
    class Quad:
        def objective(self, x):
            return float(x @ x)

        def gradient(self, x):
            return 2.0 * x

    return pounce.Problem(n=2, m=0, problem_obj=Quad())


@pytest.mark.parametrize("bad", [2**31, 2**32 + 3, 2**32 + 1, 10**12, -(2**31) - 1])
def test_add_option_integer_out_of_range_rejected(bad):
    """Over-range integer options raise instead of silently truncating."""
    prob = _quad2()
    with pytest.raises(ValueError) as exc:
        prob.add_option("max_iter", bad)
    msg = str(exc.value)
    # Error names the option and quotes the *user's* value, not a wrapped one.
    assert "max_iter" in msg
    assert str(bad) in msg


def test_add_option_integer_out_of_range_via_minimize():
    """The high-level minimize() surface rejects the same value the CLI does."""
    with pytest.raises(ValueError) as exc:
        pounce.minimize(
            lambda x: float(x @ x),
            np.array([1.0, 1.0]),
            jac=lambda x: 2.0 * x,
            max_iter=2**32 + 3,
        )
    msg = str(exc.value)
    assert "max_iter" in msg
    assert str(2**32 + 3) in msg


@pytest.mark.parametrize("good", [2**31 - 1, -(2**31), 0, 5, 500])
def test_add_option_integer_in_range_accepted(good):
    """Legitimate in-range integers (incl. i32::MIN/MAX boundaries) still work."""
    prob = _quad2()
    # Must not raise; boundary values i32::MAX and i32::MIN are valid.
    prob.add_option("max_iter", good)


def test_add_option_large_but_in_range_solves():
    """A large-but-in-range max_iter still drives a real solve."""
    prob = _quad2()
    prob.add_option("max_iter", 2**31 - 1)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([1.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, [0.0, 0.0], atol=1e-6)


# --- gh#765: jacobianstructure is optional --------------------------------
#
# In the cyipopt interface POUNCE advertises, `jacobianstructure` is an
# *optional* callback: an object that omits it declares a dense (m, n)
# Jacobian, and its `jacobian(x)` returns all m*n entries row-major.
# `Problem.solve()` used to call the method unconditionally and die with a
# bare AttributeError, while `pounce.preflight` -- which implements the
# fallback in `_preflight.py` -- accepted the very same object.
#
# Each test below is run against *both* branches of the new feature
# detection: the structure-less object and an otherwise identical one that
# supplies the pattern explicitly. A green run on the fallback alone would
# say nothing about the path every existing model takes, and vice versa.


class HS071Dense:
    """HS071 with the (already dense) `jacobianstructure` omitted.

    Delegates to an `HS071` rather than subclassing it: an inherited
    `jacobianstructure` is still `hasattr`-visible, so a subclass that
    merely deletes the name from its own dict would keep taking the
    explicit-structure branch and test nothing.

    `jacobian` is unchanged -- the 8 entries it returns are exactly the
    dense (m, n) = (2, 4) block in row-major order, which is what the
    fallback pattern addresses.
    """

    def __init__(self):
        self._inner = HS071()

    def objective(self, x):
        return self._inner.objective(x)

    def gradient(self, x):
        return self._inner.gradient(x)

    def constraints(self, x):
        return self._inner.constraints(x)

    def jacobian(self, x):
        return self._inner.jacobian(x)


class SumToOne:
    """The issue's reproduction: min x·x s.t. x0 + x1 == 1, optimum
    x = [0.5, 0.5], obj = 0.5. No `jacobianstructure`."""

    def objective(self, x):
        return float(np.asarray(x, float) @ np.asarray(x, float))

    def gradient(self, x):
        return 2 * np.asarray(x, float)

    def constraints(self, x):
        return np.array([float(np.sum(x))])

    def jacobian(self, x):
        return np.ones(2)  # dense (m=1, n=2), row-major


class SumToOneWithStructure(SumToOne):
    def jacobianstructure(self):
        return (np.array([0, 0]), np.array([0, 1]))


def _sum_to_one_problem(obj):
    p = pounce.Problem(
        n=2, m=1, problem_obj=obj, lb=[-10.0] * 2, ub=[10.0] * 2, cl=[1.0], cu=[1.0]
    )
    p.add_option("print_level", 0)
    return p


@pytest.mark.parametrize("obj", [SumToOne(), SumToOneWithStructure()])
def test_missing_jacobianstructure_solves_as_dense(obj):
    """The filed repro, plus the explicit-structure control."""
    x, info = _sum_to_one_problem(obj).solve(x0=np.array([0.0, 0.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, [0.5, 0.5], atol=1e-7)
    np.testing.assert_allclose(info["obj_val"], 0.5, atol=1e-9)


def test_missing_jacobianstructure_matches_explicit_structure():
    """Same model, same answer, whichever branch resolves the sparsity.

    HS071 rather than the 2-variable repro: it has m = 2 and a genuinely
    dense 2x4 block, so a fallback that transposed the row-major order
    (`np.divmod(k, m)` instead of `np.divmod(k, n)`) would still solve the
    1-row model and fail here.
    """
    kw = dict(n=4, m=2, lb=[1.0] * 4, ub=[5.0] * 4, cl=[25.0, 40.0], cu=[2e19, 40.0])
    out = []
    for obj in (HS071Dense(), HS071()):
        p = pounce.Problem(problem_obj=obj, **kw)
        p.add_option("tol", 1e-8)
        p.add_option("print_level", 0)
        out.append(p.solve(x0=np.array([1.0, 5.0, 5.0, 1.0])))
    (x_dense, info_dense), (x_sparse, info_sparse) = out
    assert info_dense["status_msg"] == "Solve_Succeeded"
    # The published HS071 optimum, from the issue's third oracle.
    np.testing.assert_allclose(info_dense["obj_val"], 17.0140173, rtol=1e-8)
    np.testing.assert_allclose(x_dense, [1.0, 4.7430, 3.8211, 1.3794], atol=1e-3)
    # ... and bit-for-bit the same trajectory as the explicit pattern: the
    # fallback declares the same (rows, cols), so nothing downstream moves.
    np.testing.assert_array_equal(x_dense, x_sparse)
    assert info_dense["iter_count"] == info_sparse["iter_count"]
    assert info_dense["obj_val"] == info_sparse["obj_val"]


@pytest.mark.parametrize("obj", [SumToOne(), SumToOneWithStructure()])
def test_missing_jacobianstructure_solves_in_batch(obj):
    """`solve_nlp_batch` builds its `PyTnlpInit` through the same helper."""
    (x, info), = pounce.solve_nlp_batch([_sum_to_one_problem(obj)], [np.zeros(2)])
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, [0.5, 0.5], atol=1e-7)


@pytest.mark.parametrize("obj", [SumToOne(), SumToOneWithStructure()])
def test_preflight_and_solve_agree_on_the_same_object(obj):
    """The internal inconsistency the issue reports: `preflight` accepted
    an object `solve` rejected. Whatever preflight calls solvable, solve
    must solve."""
    report = pounce.preflight(
        obj, np.zeros(2), lb=[-10.0] * 2, ub=[10.0] * 2, cl=[1.0], cu=[1.0]
    )
    assert not report.fatal
    _, info = _sum_to_one_problem(obj).solve(x0=np.zeros(2))
    assert info["status_msg"] == "Solve_Succeeded"


def test_missing_jacobianstructure_unaffected_when_m_is_zero():
    """m = 0 skips the Jacobian block entirely; no dense pattern is
    synthesized and `jacobian` is never called."""

    class Unconstrained:
        def objective(self, x):
            return float(np.asarray(x, float) @ np.asarray(x, float))

        def gradient(self, x):
            return 2 * np.asarray(x, float)

    p = pounce.Problem(n=2, m=0, problem_obj=Unconstrained(), lb=[-10.0] * 2, ub=[10.0] * 2)
    p.add_option("print_level", 0)
    x, info = p.solve(x0=np.array([1.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, [0.0, 0.0], atol=1e-6)


def test_dense_jacobian_fallback_rejects_an_out_of_range_pattern():
    """The dense pattern is m*n entries, and `nele_jac` is a signed 32-bit
    count. A problem big enough to overflow it is not one that meant to be
    dense, so the message names the callback that declares the real
    pattern -- rather than wrapping to a negative or truncated nonzero
    count and mis-sizing every structure buffer downstream.

    No callback is ever invoked: the check runs while the sparsity is
    being resolved, so the bogus object below is never asked for values.
    """

    class Big:
        def objective(self, x):
            return 0.0

        def gradient(self, x):
            return np.zeros(len(x))

        def constraints(self, x):
            return np.zeros(100_000)

        def jacobian(self, x):
            raise AssertionError("must not be reached")

    n = m = 100_000  # m*n = 1e10 > i32::MAX
    p = pounce.Problem(
        n=n,
        m=m,
        problem_obj=Big(),
        lb=[-1.0] * n,
        ub=[1.0] * n,
        cl=[-1.0] * m,
        cu=[1.0] * m,
    )
    p.add_option("print_level", 0)
    with pytest.raises(ValueError, match="jacobianstructure"):
        p.solve(x0=np.zeros(n))
