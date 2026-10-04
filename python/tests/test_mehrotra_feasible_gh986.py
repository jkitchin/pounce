"""gh#986 item 5: ``mehrotra_algorithm=yes`` on a general NLP (Mittelmann's
clnlbeam) ended ``Restoration_Failed`` after three iterations *at a feasible
point* (``inf_pr`` ~ 1e-14).

Root cause: the probing oracle's first step picks sigma ~ 0 (mu ~ 1e-11) while
the iterate's complementarity is still ~5e-2, so the iterate-quality guard
(gh#58) fires with a ratio of ~2e9 and requested restoration -- which has
nothing to repair at a feasible point and fails on entry. At a feasible
iterate the guard now recentres (LOQO mu from the iterate's own
complementarity) and restoration stays what it was for an infeasible one.

Mehrotra remains unglobalised, so on this nonconvex model it may land on a
different local solution than the default algorithm; the contract pinned here
is the one the issue asked for: no restoration from a feasible iterate, and a
verdict that is a real solve.
"""

import numpy as np
import pytest

import pounce
from pounce import NlExpr as E


def clnlbeam(ni, alpha=350.0):
    h = 1.0 / ni
    N = ni + 1
    v = E.vars(3 * N)
    t, x, u = v[:N], v[N:2 * N], v[2 * N:]
    obj = E.sum([0.5 * h * (u[i + 1] ** 2 + u[i] ** 2)
                 + 0.5 * alpha * h * (E.cos(t[i + 1]) + E.cos(t[i])) for i in range(ni)])
    cons = []
    for i in range(ni):
        cons.append(x[i + 1] - x[i] - 0.5 * h * (E.sin(t[i + 1]) + E.sin(t[i])))
        cons.append(t[i + 1] - t[i] - 0.5 * h * (u[i + 1] + u[i]))
    for end in (0, ni):
        cons += [x[end] + 0.0, t[end] + 0.0]
    s0 = 0.05 * np.cos(np.arange(N) * h * np.pi)
    x0 = np.r_[s0, s0, np.zeros(N)]
    xl = np.r_[-np.ones(N), -0.05 * np.ones(N), -1e19 * np.ones(N)]
    xu = np.r_[np.ones(N), 0.05 * np.ones(N), 1e19 * np.ones(N)]
    return pounce.build_nl_problem(
        n=3 * N, objective=obj, constraints=cons, g_l=[0.0] * len(cons),
        g_u=[0.0] * len(cons), x0=x0, x_l=xl, x_u=xu,
    )


@pytest.mark.parametrize("ni", [1000, 2000])
def test_mehrotra_clnlbeam_is_not_sent_to_restoration_at_a_feasible_point(ni):
    (x, info), = pounce.solve_nlp_batch(
        [clnlbeam(ni)], options={"print_level": 0, "mehrotra_algorithm": "yes"},
        parallel=False,
    )
    assert info["status_msg"] != "Restoration_Failed"
    assert info["status_msg"] == "Solve_Succeeded"
    assert info["iter_count"] > 3
    assert info["final_constr_viol"] < 1e-6
    # a genuine stationary point of the beam problem (default solves reach
    # 344.876; unglobalised Mehrotra may stop at another local solution)
    assert 300.0 < info["obj_val"] < 360.0


def test_default_algorithm_unchanged_on_clnlbeam():
    (x, info), = pounce.solve_nlp_batch(
        [clnlbeam(200)], options={"print_level": 0}, parallel=False
    )
    assert info["status_msg"] == "Solve_Succeeded"
    assert info["obj_val"] == pytest.approx(344.88, abs=0.05)
