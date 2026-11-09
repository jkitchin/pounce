"""Declared complementarity through the Python API: `Problem(...,
complementarity=[(row, var), ...])` solves through the complementarity
driver (`pounce_restoration::mpcc_driver`) and reports which formulation
answered and the answer's stationarity class in `info["complementarity"]`.

A pair `(row, var)` means `x_var` and `g_row(x)` are complementary, each
measured from its one finite bound: `0 <= x_j - lb_j  ⟂  g_i(x) - cl_i >= 0`
(or the upper-bound orientation). The models are the ones the Rust driver
tests and the CLI tests use, so the three entry points answer the same
questions.
"""
from pathlib import Path

import numpy as np
import pytest

import pounce

FIXTURES = Path(__file__).resolve().parents[2] / "crates" / "pounce-cli" / "tests" / "fixtures_mpcc"
INF = 2e19


class QpecSmall:
    """min (x-1)^2 + (y1-1)^2 + y2^2,  0 <= x <= 2,
    0 <= y1 ⟂ 2 y1 - 1 - x >= 0,   0 <= y2 ⟂ x + 2 y2 - 1 >= 0.
    Answer (1, 1, 0), the second pair biactive there; S-stationary."""

    def objective(self, z):
        return (z[0] - 1) ** 2 + (z[1] - 1) ** 2 + z[2] ** 2

    def gradient(self, z):
        return np.array([2 * (z[0] - 1), 2 * (z[1] - 1), 2 * z[2]])

    def constraints(self, z):
        return np.array([2 * z[1] - 1 - z[0], z[0] + 2 * z[2] - 1])

    def jacobianstructure(self):
        return np.array([0, 0, 1, 1]), np.array([0, 1, 0, 2])

    def jacobian(self, z):
        return np.array([-1.0, 2.0, 1.0, 2.0])

    def hessianstructure(self):
        return np.array([0, 1, 2]), np.array([0, 1, 2])

    def hessian(self, z, lagrange, obj_factor):
        return obj_factor * np.array([2.0, 2.0, 2.0])


def qpec_problem(**kw):
    return pounce.Problem(
        n=3,
        m=2,
        problem_obj=QpecSmall(),
        lb=[0.0, 0.0, 0.0],
        ub=[2.0, INF, INF],
        cl=[0.0, 0.0],
        cu=[INF, INF],
        **kw,
    )


class Ralph1:
    """min 2x - y,  x >= 0,  0 <= y ⟂ y - x >= 0. Answer: the origin, which
    admits no S-stationary multiplier (M-stationary only)."""

    def objective(self, z):
        return 2 * z[0] - z[1]

    def gradient(self, z):
        return np.array([2.0, -1.0])

    def constraints(self, z):
        return np.array([z[1] - z[0]])

    def jacobianstructure(self):
        return np.array([0, 0]), np.array([0, 1])

    def jacobian(self, z):
        return np.array([-1.0, 1.0])

    def hessianstructure(self):
        return np.array([], dtype=int), np.array([], dtype=int)

    def hessian(self, z, lagrange, obj_factor):
        return np.array([])


@pytest.mark.parametrize("x0", [[0.0, 0.5, 0.5], [0.0, 0.0, 0.0], [2.0, 1.5, 0.0]])
def test_qpec_small_is_solved_and_certified(x0):
    prob = qpec_problem(complementarity=[(0, 1), (1, 2)])
    prob.add_option("print_level", 0)
    x, info = prob.solve(np.array(x0))
    np.testing.assert_allclose(x, [1.0, 1.0, 0.0], atol=1e-4)
    cc = info["complementarity"]
    assert cc["accepted_by"] is not None, cc
    assert cc["max_abs_product"] <= 1e-4
    assert cc["certificate"]["class"] == "S", cc
    assert cc["certificate"]["n_biactive"] == 1, cc
    assert "stationarity: S" in cc["summary"]


@pytest.mark.parametrize(
    "opts",
    [{}, {"complementarity_strategy": "exact"}, {"complementarity_lift": "yes"},
     {"hessian_approximation": "limited-memory"}],
)
def test_the_reported_multipliers_are_the_models(opts):
    """x in [0.25, 0.75]: the answer is (3/4, 7/8, 1/8), both pairs
    non-degenerate (y > 0, rows active), so the multipliers are unique. In
    Ipopt's convention (grad f + J^T lambda - z_L + z_U = 0): lambda =
    (1/8, -1/8), z_U[x] = 3/4, every other bound multiplier 0. They were
    reported as the lowered model's row/product split, around -1e8."""
    prob = pounce.Problem(
        n=3, m=2, problem_obj=QpecSmall(),
        lb=[0.25, 0.0, 0.0], ub=[0.75, INF, INF], cl=[0.0, 0.0], cu=[INF, INF],
        complementarity=[(0, 1), (1, 2)],
    )
    prob.add_option("print_level", 0)
    for k, v in opts.items():
        prob.add_option(k, v)
    x, info = prob.solve(np.array([0.5, 0.5, 0.5]))
    np.testing.assert_allclose(x, [0.75, 0.875, 0.125], atol=1e-6)
    np.testing.assert_allclose(info["mult_g"], [0.125, -0.125], atol=1e-5)
    np.testing.assert_allclose(info["mult_x_U"], [0.75, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(info["mult_x_L"], [0.0, 0.0, 0.0], atol=1e-5)


def test_a_converged_point_that_is_not_strongly_stationary_says_so():
    prob = pounce.Problem(
        n=2, m=1, problem_obj=Ralph1(), lb=[0.0, 0.0], ub=[INF, INF], cl=[0.0], cu=[INF],
        complementarity=[(0, 1)],
    )
    prob.add_option("print_level", 0)
    x, info = prob.solve(np.array([1.0, 2.0]))
    np.testing.assert_allclose(x, [0.0, 0.0], atol=1e-4)
    cc = info["complementarity"]
    assert cc["certificate"]["class"] == "W", cc
    # A W point is reported as a fallback, not as an answer.
    assert cc["accepted_by"] is None, cc
    assert cc["fallback"] is not None, cc
    # W but B-stationary: both branch LPs have optimum 0, so no branch NLP
    # runs and the origin stands.
    b = cc["b_stationarity"]
    assert (b["verdict"], b["n_biactive"], b["branches_solved"]) == ("B", 1, 2), cc
    assert cc["escapes"] == 0, cc
    # A strongly stationary answer is B by implication; no check runs.
    _, info = qpec_problem(complementarity=[(0, 1), (1, 2)]).solve(np.zeros(3))
    assert info["complementarity"]["b_stationarity"] is None


def test_without_pairs_nothing_changes():
    prob = qpec_problem()
    prob.add_option("print_level", 0)
    _, info = prob.solve(np.array([0.0, 0.5, 0.5]))
    assert info["complementarity"] is None
    assert prob.complementarity == []


class MixedPair:
    """`0 <= x <= 1 ⟂ g = y - x` with `y` and the row free (a mixed pair):
    `min (x - 2)² + y²` has its optimum at the upper bound, `(1, 0)` with
    `g = -1 <= 0` (f = 1)."""

    def objective(self, z):
        return (z[0] - 2.0) ** 2 + z[1] ** 2

    def gradient(self, z):
        return np.array([2.0 * (z[0] - 2.0), 2.0 * z[1]])

    def constraints(self, z):
        return np.array([z[1] - z[0]])

    def jacobianstructure(self):
        return np.array([0, 0]), np.array([0, 1])

    def jacobian(self, z):
        return np.array([-1.0, 1.0])

    def hessianstructure(self):
        return np.array([0, 1]), np.array([0, 1])

    def hessian(self, z, lagrange, obj_factor):
        return obj_factor * np.array([2.0, 2.0])


def test_a_pair_it_cannot_lower_raises_with_the_reason():
    # y has no finite bound and the row is free: neither one-sided nor mixed.
    prob = pounce.Problem(
        n=2, m=1, problem_obj=MixedPair(), lb=[0.0, -INF], ub=[1.0, INF],
        cl=[-INF], cu=[INF], complementarity=[(0, 1)],
    )
    prob.add_option("print_level", 0)
    with pytest.raises(ValueError, match="one-sided"):
        prob.solve(np.array([0.5, 0.5]))


def test_read_nl_exposes_pairs_and_the_batch_path_refuses_them():
    nl = pounce.read_nl(str(FIXTURES / "qpec_small.nl"))
    assert nl.complementarity == [(0, 1, 1), (1, 2, 1)]
    with pytest.raises(ValueError, match="complementarity"):
        pounce.solve_nlp_batch([nl], [nl.x0])
    # The extension refuses too: the wrapper's check is not the only guard.
    from pounce import _pounce

    with pytest.raises(ValueError, match="complementarity"):
        _pounce.solve_nlp_batch([nl])
    prob = qpec_problem(complementarity=[(0, 1)])
    with pytest.raises(ValueError, match="complementarity"):
        pounce.solve_nlp_batch([prob], [np.array([0.0, 0.5, 0.5])])


def test_the_intermediate_callback_is_called_and_can_stop_a_solve_with_pairs():
    """Review of #1103: the lowering wrapper used to drop the caller's
    `intermediate`, so it was never called and could not stop the solve."""
    calls = []

    class Stopping(QpecSmall):
        def intermediate(self, *args):
            calls.append(args[1])
            return len(calls) < 3

    prob = pounce.Problem(
        n=3, m=2, problem_obj=Stopping(), lb=[0.0, 0.0, 0.0], ub=[2.0, INF, INF],
        cl=[0.0, 0.0], cu=[INF, INF], complementarity=[(0, 1), (1, 2)],
    )
    prob.add_option("print_level", 0)
    _, info = prob.solve(np.array([0.0, 0.5, 0.5]))
    assert len(calls) == 3
    assert info["status"] == 5, info["status_msg"]  # User_Requested_Stop
    assert info["complementarity"]["n_solves"] == 1


def test_entry_points_that_would_drop_the_pairs_refuse_them():
    from pounce import _pounce

    prob = qpec_problem(complementarity=[(0, 1)])
    with pytest.raises(ValueError, match="complementarity"):
        pounce.Solver(prob).solve(np.array([0.0, 0.5, 0.5]))
    with pytest.raises(ValueError, match="complementarity"):
        _pounce.find_minima(prob, np.array([0.0, 0.5, 0.5]), {"method": "deflation"})


@pytest.mark.parametrize("strategy", ["both", "exact", "continuation"])
def test_complementarity_strategy_picks_the_formulations_that_run(strategy):
    prob = qpec_problem(complementarity=[(0, 1), (1, 2)])
    prob.add_option("print_level", 0)
    prob.add_option("complementarity_strategy", strategy)
    x, info = prob.solve(np.array([0.0, 0.5, 0.5]))
    np.testing.assert_allclose(x, [1.0, 1.0, 0.0], atol=1e-4)
    cc = info["complementarity"]
    assert cc["certificate"]["class"] == "S", cc
    # The l1 penalty may run under any strategy, as the fallback when no
    # formulation answers (from this start the exact product alone stalls at
    # an acceptable-level point the certificate rejects). A `branch polish`
    # is not a formulation either: it is the re-solve of the reported point
    # onto its branch, run under every strategy when a pair is left
    # complementary only to more than 10 tol -- here the biactive second pair
    # at (1, 1, 0). A B-escape (`branch NLP`) is still counted.
    ran = {s[0] for s in cc["solves"]} - {"l1 penalty", "branch polish"}
    expected = {
        "both": {"Scholtes continuation", "exact product"},
        "exact": {"exact product"},
        "continuation": {"Scholtes continuation"},
    }[strategy]
    assert ran == expected, cc["solves"]


def test_a_mixed_pair_is_solved():
    """A boxed variable paired with a free row is a mixed complementarity,
    lowered and solved -- not refused."""
    prob = pounce.Problem(
        n=2, m=1, problem_obj=MixedPair(), lb=[0.0, -INF], ub=[1.0, INF],
        cl=[-INF], cu=[INF], complementarity=[(0, 0)],
    )
    prob.add_option("print_level", 0)
    x, info = prob.solve(np.array([0.5, 0.5]))
    assert info["status_msg"] == "Solve_Succeeded"
    assert np.allclose(x, [1.0, 0.0], atol=1e-6)
    assert info["complementarity"]["certificate"]["class"] == "S"


def test_the_solve_report_carries_the_complementarity_block(tmp_path):
    """`solve(report_path=...)` writes the same `complementarity` block as the
    CLI's `--json-output`, through the same conversion, and it agrees with
    `info["complementarity"]`. `statistics.iteration_count` is the reported
    lowered solve's alone; `total_iterations` is what the run cost."""
    import json

    path = tmp_path / "report.json"
    x, info = qpec_problem(complementarity=[(0, 1), (1, 2)]).solve(
        np.zeros(3), report_path=str(path)
    )
    report = json.loads(path.read_text())
    block, cc = report["complementarity"], info["complementarity"]
    assert block["strategy"] == "both"
    assert block["reported"] == "answer"
    assert block["accepted_by"] == cc["accepted_by"]
    assert block["total_iterations"] == cc["total_iterations"]
    assert len(block["solves"]) == cc["n_solves"]
    assert block["certificate"]["class"] == cc["certificate"]["class"] == "S"
    assert block["summary"] == cc["summary"]
    assert block["total_iterations"] > report["statistics"]["iteration_count"]
    # How complementary the answer is, and what the branch re-solve did, the
    # same both ways. This QPEC's exact stage ends at min(G, H) = 5e-7, above
    # 10*tol, so the re-solve runs and is taken.
    assert block["natural_residual"] == cc["natural_residual"] <= 1e-7, cc
    assert block["polish"] == cc["polish"], cc
    assert cc["polish"]["taken"] is True, cc
    assert block["polish_threshold"] == cc["polish_threshold"] > 0.0, cc
    assert all(s["seconds"] >= 0.0 for s in block["solves"])


def test_the_solve_report_carries_the_b_stationarity_check(tmp_path):
    """The report's `b_stationarity` and `escapes` agree with `info`'s.

    ralph1's origin is the branch that reaches them: a W point, so the check
    runs (an S answer skips it, which is the other test's branch). Before
    these fields existed the report omitted the check entirely, so a CLI or
    `--json-output` reader could not see the verdict Python could."""
    import json

    path = tmp_path / "report.json"
    prob = pounce.Problem(
        n=2, m=1, problem_obj=Ralph1(), lb=[0.0, 0.0], ub=[INF, INF], cl=[0.0], cu=[INF],
        complementarity=[(0, 1)],
    )
    prob.add_option("print_level", 0)
    _, info = prob.solve(np.array([1.0, 2.0]), report_path=str(path))
    block, cc = json.loads(path.read_text())["complementarity"], info["complementarity"]
    assert cc["b_stationarity"] is not None, cc
    for key in ("verdict", "reason", "n_biactive", "branches_solved", "enumerated"):
        assert block["b_stationarity"][key] == cc["b_stationarity"][key], key
    assert block["escapes"] == cc["escapes"] == 0

    # An S answer: the check does not run, and the report omits it.
    path_s = tmp_path / "report_s.json"
    qpec_problem(complementarity=[(0, 1), (1, 2)]).solve(np.zeros(3), report_path=str(path_s))
    assert "b_stationarity" not in json.loads(path_s.read_text())["complementarity"]


def test_a_solve_report_without_pairs_has_no_complementarity_block(tmp_path):
    import json

    path = tmp_path / "report.json"
    qpec_problem().solve(np.zeros(3), report_path=str(path))
    assert "complementarity" not in json.loads(path.read_text())


class SmallRelu:
    """A ReLU network 4-10-10-1 (seed 3) as declared pairs: v = (x, y1, y2),
    row r_j = y_j - (W y_prev + b)_j >= 0 paired with y_j >= 0, minimising
    the output over x in [-1, 1]^4. The first layer's rows have 5 Jacobian
    entries and the second's 11, so `complementarity_lift=auto` (rows of 8
    or more) lifts the second layer only: all three settings reach a lifted
    branch or the unlifted one, and both in one model under `auto`."""

    def __init__(self):
        rng = np.random.default_rng(3)
        sizes = [4, 10, 10, 1]
        self.W = [rng.normal(0, np.sqrt(2 / m), (n, m)) for m, n in zip(sizes, sizes[1:])]
        self.b = [rng.normal(0, 0.1, n) for n in sizes[1:]]
        A = np.zeros((20, 24))
        a = np.zeros(20)
        for l, (src, dst) in enumerate([(slice(0, 4), 4), (slice(4, 14), 14)]):
            for j in range(10):
                A[10 * l + j, dst + j] = 1.0
                A[10 * l + j, src] -= self.W[l][j]
                a[10 * l + j] = -self.b[l][j]
        self.A, self.a = A, a
        self.jr, self.jc = np.nonzero(A)
        self.g = np.zeros(24)
        self.g[14:] = self.W[2][0]

    def forward(self, x):
        h = x
        for l in range(2):
            h = np.maximum(self.W[l] @ h + self.b[l], 0)
        return float(self.W[2][0] @ h + self.b[2][0])

    def objective(self, v):
        return float(self.g @ v + self.b[2][0])

    def gradient(self, v):
        return self.g.copy()

    def constraints(self, v):
        return self.A @ v + self.a

    def jacobianstructure(self):
        return self.jr, self.jc

    def jacobian(self, v):
        return self.A[self.jr, self.jc]

    def hessianstructure(self):
        return np.array([], int), np.array([], int)

    def hessian(self, v, lagrange, obj_factor):
        return np.array([])

    def start(self):
        v = np.zeros(24)
        h = np.zeros(4)
        for l in range(2):
            h = np.maximum(self.W[l] @ h + self.b[l], 0)
            v[4 + 10 * l : 14 + 10 * l] = h
        return v


@pytest.mark.parametrize("lift, n_lifted", [("auto", 10), ("yes", 20), ("no", 0)])
def test_lifting_a_pairs_row_reaches_the_same_answer(lift, n_lifted):
    """`complementarity_lift` changes the formulation, not the model: each
    setting reaches the same certified point, a genuine evaluation of the
    network (every unit at its ReLU), and reports how many pairs it lifted."""
    P = SmallRelu()
    p = pounce.Problem(
        n=24,
        m=20,
        problem_obj=P,
        lb=np.r_[-np.ones(4), np.zeros(20)],
        ub=np.r_[np.ones(4), np.full(20, INF)],
        cl=np.zeros(20),
        cu=np.full(20, INF),
        complementarity=[(j, 4 + j) for j in range(20)],
    )
    p.add_option("print_level", 0)
    p.add_option("complementarity_lift", lift)
    v, info = p.solve(P.start())
    cc = info["complementarity"]
    assert info["status"] == 0, info["status_msg"]
    assert cc["lifted"] == n_lifted
    assert cc["certificate"]["class"] == "S"
    assert cc["max_abs_product"] < 1e-6
    assert P.objective(v) == pytest.approx(-4.3531735, abs=1e-6)
    # The reported point is the network's own: the output at x matches.
    assert P.forward(v[:4]) == pytest.approx(P.objective(v), abs=1e-6)
    assert len(v) == 24 and len(info["mult_g"]) == 20
