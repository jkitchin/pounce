"""gh#991: the active-set QP engine must report a convex QP that is
infeasible by a small exact margin as ``primal_infeasible``, not
``iteration_limit``.

``x0 + 2*x1 <= 2`` and ``x0 + 2*x1 >= 2 + eps`` have the closed-form Farkas
certificate ``y = (1, 1)``: ``y @ G = 0`` and ``y @ h = -eps < 0``. Before the
fix ``method="active-set"`` stopped after 3 iterations with the budget
untouched and said ``iteration_limit`` -- the same answer at
``max_iter=100000`` -- while ``method="ipm"`` certified the model infeasible.
"""

import numpy as np
import pytest

import pounce

P = np.array([[2.0, 1.0], [1.0, 2.0]])
C = np.array([-1.0, 2.0])
G = np.array([[1.0, 2.0], [-1.0, -2.0]])


def _h(eps):
    return np.array([2.0, -(2.0 + eps)])


@pytest.mark.parametrize("max_iter", [None, 100000])
@pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-6])
def test_active_set_certifies_the_sliver_infeasible(eps, max_iter):
    y = np.array([1.0, 1.0])
    assert np.allclose(y @ G, 0.0) and y @ _h(eps) < 0  # the oracle
    r = pounce.solve_qp(P, C, G=G, h=_h(eps), method="active-set", max_iter=max_iter)
    assert r.status == "primal_infeasible", (r.status, r.iters)
    ipm = pounce.solve_qp(P, C, G=G, h=_h(eps), method="ipm", max_iter=max_iter)
    assert ipm.status == "primal_infeasible"


def test_feasible_mirror_still_solves():
    # 2 - 1e-5 <= x0 + 2*x1 <= 2 is feasible: the fix must not call it infeasible.
    r = pounce.solve_qp(P, C, G=G, h=_h(-1e-5), method="active-set")
    assert r.status == "optimal"
