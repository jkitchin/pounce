"""`pounce.Solver` session API.

Confirms numerical equivalence with `Problem.solve` and
`Problem.solve_with_sens` on the same ParametricNLP fixture used by
`test_sensitivity.py`. Exercises `kkt_solve`, `parametric_step`, and
`reduced_hessian` against a held factor.
"""

import numpy as np
import pytest

import pounce


class ParametricNLP:
    """Same NLP as `test_sensitivity.ParametricNLP`; copied here so the
    two test modules are independently runnable."""

    def objective(self, x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    def gradient(self, x):
        return np.array([2 * x[0], 2 * x[1], 2 * x[2], 0.0, 0.0])

    def constraints(self, x):
        x1, x2, x3, eta1, eta2 = x
        return np.array([
            6 * x1 + 3 * x2 + 2 * x3 - eta1,
            eta2 * x1 + x2 - x3 - 1.0,
            eta1,
            eta2,
        ])

    def jacobianstructure(self):
        rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 3], dtype=np.int64)
        cols = np.array([0, 1, 2, 3, 0, 1, 2, 4, 3, 4], dtype=np.int64)
        return rows, cols

    def jacobian(self, x):
        return np.array([
            6.0, 3.0, 2.0, -1.0,
            x[4], 1.0, -1.0, x[0],
            1.0,
            1.0,
        ])

    def hessianstructure(self):
        rows = np.array([0, 1, 2, 4], dtype=np.int64)
        cols = np.array([0, 1, 2, 0], dtype=np.int64)
        return rows, cols

    def hessian(self, x, lagrange, obj_factor):
        return np.array([
            2.0 * obj_factor,
            2.0 * obj_factor,
            2.0 * obj_factor,
            lagrange[1],
        ])


def _make(eta1=5.0, eta2=1.0):
    p = pounce.Problem(
        n=5, m=4, problem_obj=ParametricNLP(),
        lb=[0.0, 0.0, 0.0, -1e19, -1e19],
        ub=[1e19, 1e19, 1e19, 1e19, 1e19],
        cl=[0.0, 0.0, eta1, eta2],
        cu=[0.0, 0.0, eta1, eta2],
    )
    p.add_option("tol", 1e-10)
    p.add_option("print_level", 0)
    p.add_option("sb", "yes")
    return p


X0 = np.array([0.15, 0.15, 0.0, 0.0, 0.0])


def test_solver_solve_matches_problem_solve():
    x_problem, info_problem = _make().solve(x0=X0)
    solver = pounce.Solver(_make())
    x_solver, info_solver = solver.solve(x0=X0)
    assert info_problem["status_msg"] == "Solve_Succeeded"
    assert info_solver["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x_solver, x_problem, atol=1e-12)
    assert info_solver["obj_val"] == pytest.approx(info_problem["obj_val"], abs=1e-12)
    assert solver.converged is True
    assert solver.kkt_dim is not None and solver.kkt_dim > 0


def test_solver_parametric_step_matches_problem_solve_with_sens():
    _, info_baseline = _make().solve_with_sens(
        x0=X0, pin_constraint_indices=[2, 3], deltas=[-0.5, 0.0],
    )
    dx_baseline = info_baseline["dx"]

    solver = pounce.Solver(_make())
    solver.solve(x0=X0)
    dx = solver.parametric_step([2, 3], [-0.5, 0.0])
    np.testing.assert_allclose(dx, dx_baseline, atol=1e-10)


def test_solver_reduced_hessian_matches_problem_solve_with_sens():
    _, info_baseline = _make().solve_with_sens(
        x0=X0, pin_constraint_indices=[2, 3], compute_reduced_hessian=True,
    )
    hr_baseline = info_baseline["reduced_hessian"]

    solver = pounce.Solver(_make())
    solver.solve(x0=X0)
    hr = solver.reduced_hessian([2, 3])
    np.testing.assert_allclose(hr, hr_baseline, atol=1e-10)


def test_solver_kkt_solve_zero_rhs_returns_zero():
    solver = pounce.Solver(_make())
    solver.solve(x0=X0)
    dim = solver.kkt_dim
    rhs = np.zeros(dim)
    lhs = solver.kkt_solve(rhs)
    np.testing.assert_allclose(lhs, np.zeros(dim), atol=1e-12)


def test_solver_methods_before_solve_error():
    solver = pounce.Solver(_make())
    assert solver.converged is False
    assert solver.kkt_dim is None
    with pytest.raises(RuntimeError):
        solver.kkt_solve(np.zeros(10))
    with pytest.raises(RuntimeError):
        solver.parametric_step([2, 3], [0.1, 0.0])
    with pytest.raises(RuntimeError):
        solver.reduced_hessian([2, 3])


def test_solver_pin_index_out_of_range_errors():
    solver = pounce.Solver(_make())
    solver.solve(x0=X0)
    with pytest.raises(ValueError):
        solver.parametric_step([4], [0.1])  # m=4, valid is [0,4)
    with pytest.raises(ValueError):
        solver.reduced_hessian([-1])
