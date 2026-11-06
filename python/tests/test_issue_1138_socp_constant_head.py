"""gh#1138: an SOC whose head row has no x in it, with a negative constant,
is primal infeasible for every objective, not only for c = 0.

``s = h - G x = (-1, x1, x2)`` in the SOC is ``||x|| <= -1``. Before the
fix, ``c = 0`` was certified and any nonzero ``c`` ended
``numerical_failure``: the certificate's multiplier sits on a row with no
``G`` entry, so the Farkas margin never accepted it. The fix proves it before
the iteration, with the exact certificate ``z = (1, 0, 0)``.
"""

import numpy as np
import pytest

import pounce

G = np.vstack([np.zeros((1, 2)), -np.eye(2)])


@pytest.mark.parametrize("c", [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [-3.0, 2.0]])
def test_a_negative_constant_head_is_primal_infeasible(c):
    h = np.array([-1.0, 0.0, 0.0])
    r = pounce.solve_socp(c=np.array(c), G=G, h=h, cones=[("soc", 3)])
    assert r.status == "primal_infeasible"
    # An exact Farkas certificate: z in SOC*, G^T z = 0, h^T z < 0.
    z = np.asarray(r.z)
    assert z[0] >= np.linalg.norm(z[1:])
    assert np.array_equal(G.T @ z, np.zeros(2))
    assert h @ z < 0


def test_the_same_row_in_the_orthant_through_solve_qp():
    h = np.array([-1.0, 0.0, 0.0])
    r = pounce.solve_qp(c=np.array([1.0, 1.0]), G=G, h=h)
    assert r.status == "primal_infeasible"


def test_a_feasible_constant_head_is_still_solved():
    # ||x|| <= 1, min x1 + x2 = -sqrt(2).
    h = np.array([1.0, 0.0, 0.0])
    r = pounce.solve_socp(c=np.array([1.0, 1.0]), G=G, h=h, cones=[("soc", 3)])
    assert r.status == "optimal"
    assert r.obj == pytest.approx(-np.sqrt(2.0), abs=1e-6)
