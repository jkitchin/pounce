"""`Solver.classify_activity`: post-solve activity classification
(dev-notes/covariance-information-roadmap.md item 0, gh #362).

The scalar study model min ½x² − p·x with x ≥ 0 walks the three
regimes by moving the unconstrained minimizer: p > 0 puts it inside
(bound inactive), p < 0 outside (strongly active), p = 0 exactly on
the bound (weakly active: slack and multiplier both O(√μ), where no
fixed threshold on either alone can classify). The same geometry
written as an inequality row (x unbounded, row x ≥ 0) must classify
identically: that is the gh #362 shape, where the activity lives on a
row and never shows up in the bound multipliers.
"""

import numpy as np
import pytest

import pounce


class ScalarBound:
    """min ½x² − p·x with the bound on the variable: x ≥ 0."""

    def __init__(self, p):
        self.p = p

    def objective(self, x):
        return 0.5 * x[0] ** 2 - self.p * x[0]

    def gradient(self, x):
        return np.array([x[0] - self.p])

    def constraints(self, x):
        return np.array([])

    def jacobianstructure(self):
        empty = np.array([], dtype=np.int64)
        return empty, empty

    def jacobian(self, x):
        return np.array([])

    def hessianstructure(self):
        return np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)

    def hessian(self, x, lagrange, obj_factor):
        return np.array([obj_factor])


class ScalarRow:
    """min ½x² − p·x with the bound as an inequality row: g(x) = c·x ≥ 0,
    the variable itself unbounded. The coefficient c changes the row's
    units, not its geometry, so classification must not move with it."""

    def __init__(self, p, c=1.0):
        self.p = p
        self.c = c

    def objective(self, x):
        return 0.5 * x[0] ** 2 - self.p * x[0]

    def gradient(self, x):
        return np.array([x[0] - self.p])

    def constraints(self, x):
        return np.array([self.c * x[0]])

    def jacobianstructure(self):
        zero = np.array([0], dtype=np.int64)
        return zero, zero

    def jacobian(self, x):
        return np.array([self.c])

    def hessianstructure(self):
        return np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)

    def hessian(self, x, lagrange, obj_factor):
        return np.array([obj_factor])


class LinearBox:
    """min −x on 0 ≤ x ≤ 1: zero curvature everywhere, so activity is
    below the identification floor no matter how hard the bound binds."""

    def objective(self, x):
        return -x[0]

    def gradient(self, x):
        return np.array([-1.0])

    def constraints(self, x):
        return np.array([])

    def jacobianstructure(self):
        empty = np.array([], dtype=np.int64)
        return empty, empty

    def jacobian(self, x):
        return np.array([])

    def hessianstructure(self):
        empty = np.array([], dtype=np.int64)
        return empty, empty

    def hessian(self, x, lagrange, obj_factor):
        return np.array([])


def _options(p):
    p.add_option("tol", 1e-10)
    p.add_option("bound_relax_factor", 0.0)
    p.add_option("print_level", 0)
    p.add_option("sb", "yes")
    return p


def _solve_bound(p):
    prob = _options(pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(p),
        lb=[0.0], ub=[1e19], cl=[], cu=[],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    return solver.classify_activity()


def _solve_row(p, c=1.0):
    prob = _options(pounce.Problem(
        n=1, m=1, problem_obj=ScalarRow(p, c),
        lb=[-1e19], ub=[1e19], cl=[0.0], cu=[1e19],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    return solver.classify_activity()


@pytest.mark.parametrize("p, status", [
    (1.0, "inactive"),
    (-1.0, "strongly_active"),
    (0.0, "weakly_active"),
])
def test_variable_bound_regimes(p, status):
    rep = _solve_bound(p)
    assert rep["var_status"] == [status]
    assert rep["var_q_sign"][0] == 1
    assert not rep["var_off_central_path"][0]
    assert not rep["var_contaminated"][0]
    assert rep["mu"] < 1e-4


def test_variable_bound_ratio_scales():
    # on the central path z·s = μ with unit curvature: r ≈ μ inactive,
    # r ≈ 1/μ strongly active, r ≈ 1 weakly active
    rep_in = _solve_bound(1.0)
    mu = rep_in["mu"]
    assert rep_in["var_ratio"][0] == pytest.approx(mu, rel=10.0)
    assert _solve_bound(-1.0)["var_ratio"][0] > 1.0 / np.sqrt(mu)
    assert _solve_bound(0.0)["var_ratio"][0] == pytest.approx(1.0, rel=0.5)


@pytest.mark.parametrize("p, status", [
    (1.0, "inactive"),
    (-1.0, "strongly_active"),
    (0.0, "weakly_active"),
])
def test_row_agrees_with_bound(p, status):
    # gh #362: the same geometry moved onto an inequality row classifies
    # identically, and the now-unbounded variable reports as such
    rep = _solve_row(p)
    assert rep["row_status"] == [status]
    assert rep["row_q_sign"][0] == 1
    assert rep["var_status"] == ["unbounded"]
    assert np.isnan(rep["var_ratio"][0])


@pytest.mark.parametrize("c", [1.0, 100.0, 1000.0])
@pytest.mark.parametrize("p, status", [
    (1.0, "inactive"),
    (-1.0, "strongly_active"),
    (0.0, "weakly_active"),
])
def test_row_classification_is_scale_invariant(c, p, status):
    # d -> c*d sends Sigma -> Sigma/c^2 while the curvature along the
    # unit normal is unchanged; the ||grad||^4 normalization restores
    # the balance, so the status cannot move with the row's units
    # (second review's blocking finding)
    rep = _solve_row(p, c=c)
    assert rep["row_status"] == [status]


def test_row_scale_invariance_without_nlp_scaling():
    # gradient-based scaling (the default) caps the distortion the old
    # ratio suffered; with scaling off nothing does, so this is the
    # sharp version: the weakly active ratio must sit at 1 exactly
    prob = pounce.Problem(
        n=1, m=1, problem_obj=ScalarRow(0.0, 1000.0),
        lb=[-1e19], ub=[1e19], cl=[0.0], cu=[1e19],
    )
    prob.add_option("tol", 1e-10)
    prob.add_option("bound_relax_factor", 0.0)
    prob.add_option("nlp_scaling_method", "none")
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    rep = solver.classify_activity()
    assert rep["row_status"] == ["weakly_active"]
    assert rep["row_ratio"][0] == pytest.approx(1.0, rel=0.5)


def test_near_bound_inactive_flags_contamination():
    # lb at distance 0.01 from the optimum: genuinely inactive, but the
    # barrier contributes r = mu/s^2, about 1e4 times the O(mu) an
    # inactive bound should carry; the mu-relative rule flags it while
    # the far-bound cases above stay clean
    prob = _options(pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(1.0),
        lb=[0.99], ub=[1e19], cl=[], cu=[],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.995]))
    assert info["status_msg"] == "Solve_Succeeded"
    rep = solver.classify_activity()
    assert rep["var_status"] == ["inactive"]
    assert rep["var_contaminated"][0]
    assert not rep["var_off_central_path"][0]


def test_zero_curvature_is_unidentified():
    prob = _options(pounce.Problem(
        n=1, m=0, problem_obj=LinearBox(),
        lb=[0.0], ub=[1.0], cl=[], cu=[],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    rep = solver.classify_activity()
    assert rep["var_status"] == ["unidentified"]
    assert rep["var_q_sign"][0] == 0


def test_relaxed_bounds_are_refused():
    # bound_relax_factor defaults to 1e-8; the classifier's slacks and
    # complementarity products assume unperturbed bounds
    prob = pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(1.0),
        lb=[0.0], ub=[1e19], cl=[], cu=[],
    )
    prob.add_option("tol", 1e-10)
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    with pytest.raises(ValueError, match="bound_relax_factor"):
        solver.classify_activity()


def test_classify_before_solve_raises():
    prob = _options(pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(1.0),
        lb=[0.0], ub=[1e19], cl=[], cu=[],
    ))
    with pytest.raises(RuntimeError, match="no converged factor"):
        pounce.Solver(prob).classify_activity()


def test_loose_mu_reports_ambiguous():
    # the weakly active geometry solved only to μ = 1e-2: r ≈ 1 sits in
    # the fixed band, but at this μ the edges give it no margin, so the
    # classifier refuses rather than guesses
    prob = pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(0.0),
        lb=[0.0], ub=[1e19], cl=[], cu=[],
    )
    prob.add_option("mu_target", 1e-2)
    # the outer test measures unbarriered complementarity, which the
    # μ-floored point holds at ~1e-2, so both gates must sit above it
    prob.add_option("tol", 5e-2)
    prob.add_option("compl_inf_tol", 5e-2)
    prob.add_option("bound_relax_factor", 0.0)
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    rep = solver.classify_activity()
    assert rep["mu"] == pytest.approx(1e-2, rel=1e-6)
    assert rep["var_status"] == ["ambiguous"]
    assert rep["var_ratio"][0] == pytest.approx(1.0, rel=0.5)


class MixedModel:
    """Four variables, two constraints, every status in one report:

    min ½(x0-5)² + ½(x1+1)² + ½(x2-2)² + ½(x3+1)²
    s.t. x0 + x2 = 7        (equality row)
         x1 >= 0            (inequality row, pulled active by the objective)
         0 <= x0 <= 10      (two-sided bound, inactive: x0* = 5)
         x2 = 2             (fixed variable, removed internally)
         x3 >= 0            (bound, pulled active)
         x1 free
    """

    def objective(self, x):
        return 0.5 * ((x[0] - 5) ** 2 + (x[1] + 1) ** 2
                      + (x[2] - 2) ** 2 + (x[3] + 1) ** 2)

    def gradient(self, x):
        return np.array([x[0] - 5, x[1] + 1, x[2] - 2, x[3] + 1])

    def constraints(self, x):
        return np.array([x[0] + x[2], x[1]])

    def jacobianstructure(self):
        rows = np.array([0, 0, 1], dtype=np.int64)
        cols = np.array([0, 2, 1], dtype=np.int64)
        return rows, cols

    def jacobian(self, x):
        return np.array([1.0, 1.0, 1.0])

    def hessianstructure(self):
        idx = np.array([0, 1, 2, 3], dtype=np.int64)
        return idx, idx

    def hessian(self, x, lagrange, obj_factor):
        return obj_factor * np.ones(4)


def test_mixed_model_reports_in_user_space():
    # a fixed variable is removed from the internal solve
    # (make_parameter); the report must keep the user's indices, with
    # the fixed slot marked rather than everything after it shifted,
    # and rows indexed by the user's constraint order with the
    # equality as a placeholder
    prob = _options(pounce.Problem(
        n=4, m=2, problem_obj=MixedModel(),
        lb=[0.0, -1e19, 2.0, 0.0], ub=[10.0, 1e19, 2.0, 1e19],
        cl=[7.0, 0.0], cu=[7.0, 1e19],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([4.0, 0.5, 2.0, 0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    rep = solver.classify_activity()
    assert rep["var_status"] == [
        "inactive", "unbounded", "fixed", "strongly_active",
    ]
    assert rep["row_status"] == ["equality", "strongly_active"]
    assert len(rep["var_ratio"]) == 4 and len(rep["row_ratio"]) == 2
    assert np.isnan(rep["var_ratio"][1]) and np.isnan(rep["var_ratio"][2])
    assert np.isnan(rep["row_ratio"][0])
    assert rep["var_q_sign"][0] == 1 and rep["var_q_sign"][3] == 1
    assert rep["row_q_sign"][1] == 1
    assert not any(rep["var_contaminated"]) and not any(rep["row_contaminated"])


def test_sigma_is_natural_units():
    # the weak scalar row g = c*x >= 0 has natural geometric weight
    # Sigma*||grad||^2 = q = 1, so Sigma_nat = 1/c^2 exactly. At
    # c = 1000 the default gradient-based scaling engages a per-row
    # d_scale, and a scaled-space report would differ by df/dg^2;
    # asserting 1/c^2 under both scaling modes pins the natural-units
    # contract at the boundary.
    for scaling in ("gradient-based", "none"):
        c = 1000.0
        prob = pounce.Problem(
            n=1, m=1, problem_obj=ScalarRow(0.0, c),
            lb=[-1e19], ub=[1e19], cl=[0.0], cu=[1e19],
        )
        prob.add_option("tol", 1e-10)
        prob.add_option("bound_relax_factor", 0.0)
        prob.add_option("nlp_scaling_method", scaling)
        prob.add_option("print_level", 0)
        prob.add_option("sb", "yes")
        solver = pounce.Solver(prob)
        _, info = solver.solve(x0=np.array([0.5]))
        assert info["status_msg"] == "Solve_Succeeded"
        rep = solver.classify_activity()
        assert rep["row_status"] == ["weakly_active"], scaling
        assert rep["row_sigma"][0] == pytest.approx(1.0 / c**2, rel=0.5), scaling
        # row_normal is natural too: the user's coefficient, not dg*c
        np.testing.assert_allclose(solver.row_normal(0), [c], rtol=1e-9)


def test_sigma_is_reported_raw():
    # with unit curvature the ratio IS Σ/1, so the reported sigma must
    # reproduce it; Σ ≈ 1 at weak activity and ≈ μ when inactive
    rep = _solve_bound(0.0)
    assert rep["var_sigma"][0] == pytest.approx(rep["var_ratio"][0], rel=1e-12)
    assert rep["var_sigma"][0] == pytest.approx(1.0, rel=0.5)
    rep = _solve_bound(1.0)
    assert rep["var_sigma"][0] == pytest.approx(rep["mu"], rel=10.0)
    rep = _solve_row(0.0)
    assert rep["row_sigma"][0] == pytest.approx(rep["row_ratio"][0], rel=1e-12)
    assert rep["var_sigma"][0] == 0.0  # unbounded: nothing classified


def _mixed_solver():
    prob = _options(pounce.Problem(
        n=4, m=2, problem_obj=MixedModel(),
        lb=[0.0, -1e19, 2.0, 0.0], ub=[10.0, 1e19, 2.0, 1e19],
        cl=[7.0, 0.0], cu=[7.0, 1e19],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([4.0, 0.5, 2.0, 0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    return solver


def test_row_normal_in_user_space():
    solver = _mixed_solver()
    # g0 = x0 + x2 (equality; the fixed x2's column was removed from
    # the solve, so its entry reports 0), g1 = x1
    np.testing.assert_allclose(solver.row_normal(0), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(solver.row_normal(1), [0.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="out of range"):
        solver.row_normal(2)


def test_row_normal_before_solve_raises():
    prob = _options(pounce.Problem(
        n=1, m=1, problem_obj=ScalarRow(1.0),
        lb=[-1e19], ub=[1e19], cl=[0.0], cu=[1e19],
    ))
    with pytest.raises(RuntimeError, match="no converged factor"):
        pounce.Solver(prob).row_normal(0)


def test_hessian_vec_natural_units():
    # the scalar study model has H = 1 exactly (natural units); the
    # session's Hessian-vector product must return it, and reject a
    # wrong-length vector. Scaling does not engage on this fixture;
    # the df != 1 axis of the natural-units contract is pinned at the
    # pyomo level (test_information_exact_under_objective_scaling)
    prob = _options(pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(1.0),
        lb=[0.0], ub=[1e19], cl=[], cu=[],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(solver.hessian_vec([1.0]), [1.0], rtol=1e-12)
    with pytest.raises(ValueError, match="length"):
        solver.hessian_vec([1.0, 2.0])


def test_primal_rows_skips_the_fixed_column():
    # MixedModel fixes x2 (lb == ub == 2.0), so make_parameter removes
    # its column and every LATER user variable sits one row earlier in
    # the factor than its user index. That shift is the whole reason
    # this accessor exists: user-space indices (the .col file, the
    # activity report, row_normal) are not factor rows.
    solver = _mixed_solver()
    assert solver.primal_rows([0, 1, 2, 3]) == [0, 1, None, 2]
    # and the identity case, so a caller cannot conclude from one model
    # that user index == factor row in general
    assert solver.primal_rows([0]) == [0]
    with pytest.raises(ValueError, match="out of range"):
        solver.primal_rows([4])
    with pytest.raises(ValueError, match="out of range"):
        solver.primal_rows([-1])


def test_primal_rows_before_solve_raises():
    prob = _options(pounce.Problem(
        n=1, m=1, problem_obj=ScalarRow(1.0),
        lb=[-1e19], ub=[1e19], cl=[0.0], cu=[1e19],
    ))
    with pytest.raises(RuntimeError, match="no converged factor"):
        pounce.Solver(prob).primal_rows([0])


# ---------------------------------------------------------------
# `reduced_activity`: the class the diagonal normalizer cannot call
# ---------------------------------------------------------------


class CoupledKink:
    """min ½k² + c·k·y + ½y² − A·p·k  s.t. p = 0, 0 ≤ k ≤ 10, y free.

    At p = 0 the reduced gradient at k = 0 vanishes and so does the
    multiplier: a kink by construction, at every `rho`. `rho` is the
    curvature reduced along `k` after `y` re-optimizes, `1 − c²`, so
    `rho = 1` is decoupled and smaller values are more strongly
    coupled. `classify_activity` divides Σ by the Hessian DIAGONAL, so
    its ratio is exactly `rho` and the kink drops out of the
    [1e-1, 1e1] band once `rho < 1e-1` — gh #763.
    """

    A = 1.10

    def __init__(self, rho):
        self.rho = rho
        self.c = np.sqrt(1.0 - rho)

    def objective(self, x):
        k, y, p = x
        return 0.5 * k * k + self.c * k * y + 0.5 * y * y - self.A * p * k

    def gradient(self, x):
        k, y, p = x
        return np.array([k + self.c * y - self.A * p, self.c * k + y, -self.A * k])

    def constraints(self, x):
        return np.array([x[2]])

    def jacobianstructure(self):
        return np.array([0], dtype=np.int64), np.array([2], dtype=np.int64)

    def jacobian(self, x):
        return np.array([1.0])

    def hessianstructure(self):
        return (np.array([0, 1, 1, 2], dtype=np.int64),
                np.array([0, 0, 1, 0], dtype=np.int64))

    def hessian(self, x, lagrange, obj_factor):
        return obj_factor * np.array([1.0, self.c, 1.0, -self.A])


def _solve_coupled(rho):
    prob = _options(pounce.Problem(
        n=3, m=1, problem_obj=CoupledKink(rho),
        lb=[0.0, -1e19, -1e19], ub=[10.0, 1e19, 1e19],
        cl=[0.0], cu=[0.0],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.3, 0.0, 0.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    return solver


@pytest.mark.parametrize("rho, diagonal_class", [
    (1.0, "weakly_active"),
    (1e-1, "weakly_active"),
    (1e-2, "ambiguous"),
    (1e-3, "ambiguous"),
])
def test_reduced_activity_certifies_a_coupled_kink(rho, diagonal_class):
    # The issue's own table. All four rows are the SAME kink; only how
    # strongly `k` couples to `y` changes. The diagonal normalizer's
    # ratio is `reduced/diagonal`, so it tracks the coupling and the
    # bottom two fall out of the band — at any tolerance, since that
    # ratio does not move with μ. The reduced normalizer's ratio is 1
    # at every coupling.
    solver = _solve_coupled(rho)
    rep = solver.classify_activity()
    assert rep["var_ratio"][0] == pytest.approx(rho, rel=1e-3)
    assert rep["var_status"][0] == diagonal_class

    red = solver.reduced_activity([0])
    assert red["status"] == ["weakly_active"]
    assert red["ratio"][0] == pytest.approx(1.0, rel=1e-3)
    assert red["q_reduced"][0] == pytest.approx(rho, rel=1e-3)
    assert red["q_sign"][0] == 1
    assert red["var"][0] == 0
    assert red["sigma"][0] == pytest.approx(rep["var_sigma"][0], rel=1e-12)
    assert red["mu"] == pytest.approx(rep["mu"], rel=1e-12)


def test_reduced_activity_agrees_where_the_coordinate_is_decoupled():
    # The refinement must not move a verdict it has no reason to move.
    # MixedModel is diagonal, so every reduced curvature IS the
    # diagonal one and every class has to come back unchanged —
    # including the entries with no bound geometry at all.
    solver = _mixed_solver()
    rep = solver.classify_activity()
    n = len(rep["var_status"])
    red = solver.reduced_activity(list(range(n)))

    assert red["status"] == rep["var_status"]
    assert list(red["var"]) == list(range(n))
    for i in range(n):
        if np.isnan(rep["var_ratio"][i]):
            assert np.isnan(red["ratio"][i])
        else:
            assert red["ratio"][i] == pytest.approx(rep["var_ratio"][i], rel=1e-5)


def test_reduced_activity_rejects_indices_outside_the_users_variables():
    solver = _mixed_solver()
    with pytest.raises(ValueError, match="out of range"):
        solver.reduced_activity([4])
    with pytest.raises(ValueError, match="out of range"):
        solver.reduced_activity([-1])
    empty = solver.reduced_activity([])
    assert empty["status"] == [] and len(empty["ratio"]) == 0


def test_reduced_activity_before_solve_raises():
    prob = _options(pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(1.0),
        lb=[0.0], ub=[1e19], cl=[], cu=[],
    ))
    with pytest.raises(RuntimeError, match="no converged factor"):
        pounce.Solver(prob).reduced_activity([0])


def test_reduced_activity_refuses_a_relaxed_solve():
    # Same guard as classify_activity, for the same reason: relaxed
    # bounds shift the slacks Σ is read from.
    prob = pounce.Problem(
        n=1, m=0, problem_obj=ScalarBound(0.0),
        lb=[0.0], ub=[1e19], cl=[], cu=[],
    )
    prob.add_option("tol", 1e-10)
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    with pytest.raises(ValueError, match="bound_relax_factor"):
        solver.reduced_activity([0])


# ---------------------------------------------------------------
# `reduced_row_activity`: the same question for a constraint row
# ---------------------------------------------------------------


class CoupledKinkRow:
    """min ½k² + c·k·y + ½y² − A·p·k  s.t. p = 0, 2k ≥ 0, all x free.

    The row analogue of `CoupledKink`: the same kink, held by an
    inequality ROW instead of a bound on `k`. `classify_activity`
    divides a row's geometric weight by the curvature along the row's
    own gradient, which is a real directional curvature but still not
    a *reduced* one, so its ratio is `reduced/directional` — exactly
    `rho` here — and the kink drops out of the [1e-1, 1e1] band once
    `rho < 1e-1`. gh #804.

    The row's gradient is `2·e_k`, not a unit vector, so the geometric
    weight and the unit-normal curvature both have a factor to carry.
    The equality sits at g0, ahead of the inequality, so a full-g index
    read as an inequality position would answer about the wrong row.
    """

    A = 1.10
    K_SCALE = 2.0

    def __init__(self, rho):
        self.rho = rho
        self.c = np.sqrt(1.0 - rho)

    def objective(self, x):
        k, y, p = x
        return 0.5 * k * k + self.c * k * y + 0.5 * y * y - self.A * p * k

    def gradient(self, x):
        k, y, p = x
        return np.array([k + self.c * y - self.A * p, self.c * k + y, -self.A * k])

    def constraints(self, x):
        return np.array([x[2], self.K_SCALE * x[0]])

    def jacobianstructure(self):
        return np.array([0, 1], dtype=np.int64), np.array([2, 0], dtype=np.int64)

    def jacobian(self, x):
        return np.array([1.0, self.K_SCALE])

    def hessianstructure(self):
        return (np.array([0, 1, 1, 2], dtype=np.int64),
                np.array([0, 0, 1, 0], dtype=np.int64))

    def hessian(self, x, lagrange, obj_factor):
        return obj_factor * np.array([1.0, self.c, 1.0, -self.A])


def _solve_coupled_row(rho):
    prob = _options(pounce.Problem(
        n=3, m=2, problem_obj=CoupledKinkRow(rho),
        lb=[-1e19] * 3, ub=[1e19] * 3,
        cl=[0.0, 0.0], cu=[0.0, 1e19],
    ))
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.3, 0.0, 0.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    return solver


@pytest.mark.parametrize("rho, directional_class", [
    (1.0, "weakly_active"),
    (1e-1, "weakly_active"),
    (1e-2, "ambiguous"),
    (1e-3, "ambiguous"),
])
def test_reduced_row_activity_certifies_a_coupled_row_kink(rho, directional_class):
    # gh #763's table, on the row path. All four are the SAME row kink;
    # only how strongly the row's direction couples to `y` changes. The
    # directional normalizer's ratio tracks the coupling and the bottom
    # two fall out of the band — at any tolerance, since that ratio
    # does not move with μ. The reduced one is 1 at every coupling.
    solver = _solve_coupled_row(rho)
    rep = solver.classify_activity()
    assert rep["row_ratio"][1] == pytest.approx(rho, rel=1e-3)
    assert rep["row_status"][1] == directional_class

    red = solver.reduced_row_activity([1])
    assert red["status"] == ["weakly_active"]
    assert red["ratio"][0] == pytest.approx(1.0, rel=1e-3)
    assert red["q_reduced"][0] == pytest.approx(rho, rel=1e-3)
    assert red["q_sign"][0] == 1
    assert red["row"][0] == 1
    assert red["sigma"][0] == pytest.approx(rep["row_sigma"][1], rel=1e-12)
    assert red["mu"] == pytest.approx(rep["mu"], rel=1e-12)


def test_reduced_row_activity_agrees_where_the_row_is_decoupled():
    # The refinement must not move a verdict it has no reason to move.
    # MixedModel is diagonal and its rows point along coordinate
    # directions, so every reduced curvature IS the directional one and
    # every class has to come back unchanged — the equality row's
    # placeholder included.
    solver = _mixed_solver()
    rep = solver.classify_activity()
    m = len(rep["row_status"])
    red = solver.reduced_row_activity(list(range(m)))

    assert red["status"] == rep["row_status"]
    assert list(red["row"]) == list(range(m))
    for j in range(m):
        if np.isnan(rep["row_ratio"][j]):
            assert np.isnan(red["ratio"][j])
        else:
            assert red["ratio"][j] == pytest.approx(rep["row_ratio"][j], rel=1e-5)


def test_reduced_row_activity_rejects_indices_outside_the_users_constraints():
    solver = _mixed_solver()
    with pytest.raises(ValueError, match="out of range"):
        solver.reduced_row_activity([2])
    with pytest.raises(ValueError, match="out of range"):
        solver.reduced_row_activity([-1])
    empty = solver.reduced_row_activity([])
    assert empty["status"] == [] and len(empty["ratio"]) == 0


def test_reduced_row_activity_before_solve_raises():
    prob = _options(pounce.Problem(
        n=1, m=1, problem_obj=ScalarRow(1.0),
        lb=[-1e19], ub=[1e19], cl=[0.0], cu=[1e19],
    ))
    with pytest.raises(RuntimeError, match="no converged factor"):
        pounce.Solver(prob).reduced_row_activity([0])


def test_reduced_row_activity_refuses_a_relaxed_solve():
    # Same guard as classify_activity, for the same reason: relaxed
    # bounds shift the slacks Σ is read from.
    prob = pounce.Problem(
        n=1, m=1, problem_obj=ScalarRow(1.0),
        lb=[-1e19], ub=[1e19], cl=[0.0], cu=[1e19],
    )
    prob.add_option("tol", 1e-10)
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    with pytest.raises(ValueError, match="bound_relax_factor"):
        solver.reduced_row_activity([0])
