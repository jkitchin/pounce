"""Second-order cones declared through the Python API: `Problem(...,
cones=[[r0, r1, ...], ...])` says `g_{r0} >= ||(g_{r1}, ...)||` over free
constraint rows, carried natively by the interior-point method (workstream C,
`dev-notes/research/nlp-soc-design.md`)."""
import numpy as np
import pytest

import pounce

INF = 2e19


class Disk:
    """min x0 + x1 over (1, x0, x1) in K. Answer x = -(1, 1)/sqrt(2): both
    tail entries negative, so a cone row read as `>= 0` cannot reach it."""

    def objective(self, x):
        return x[0] + x[1]

    def gradient(self, x):
        return np.array([1.0, 1.0])

    def constraints(self, x):
        return np.array([1.0, x[0], x[1]])

    def jacobianstructure(self):
        return np.array([1, 2]), np.array([0, 1])

    def jacobian(self, x):
        return np.array([1.0, 1.0])

    def hessianstructure(self):
        return np.array([], dtype=int), np.array([], dtype=int)

    def hessian(self, x, lagrange, obj_factor):
        return np.array([])


def disk(**kw):
    p = pounce.Problem(n=2, m=3, problem_obj=Disk(), lb=[-INF] * 2, ub=[INF] * 2,
                       cl=[-INF] * 3, cu=[INF] * 3, **kw)
    p.add_option("print_level", 0)
    return p


def test_a_declared_cone_reaches_its_negative_tail_optimum():
    x, info = disk(cones=[[0, 1, 2]]).solve(np.zeros(2))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, [-np.sqrt(0.5)] * 2, atol=1e-7)


def test_without_the_cone_the_model_is_unbounded():
    _, info = disk().solve(np.zeros(2))
    assert info["status_msg"] != "Solve_Succeeded"


def test_paths_that_do_not_carry_cones_refuse():
    p = disk(cones=[[0, 1, 2]])
    p.add_option("mu_strategy", "adaptive")
    _, info = p.solve(np.zeros(2))
    assert info["status_msg"] == "Invalid_Option"
    with pytest.raises(ValueError, match="second-order cones"):
        pounce.solve_nlp_batch([disk(cones=[[0, 1, 2]])], [np.zeros(2)])
