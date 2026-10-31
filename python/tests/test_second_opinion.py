"""The second-opinion ladder, from Python.

The ladder used to live in `crates/pounce-cli/src/main.rs`, so a caller
embedding POUNCE did not get it — which was backwards. A caller reaching the
solver from a modelling layer is the one most likely to hand over an
uninitialized starting point, an uninitialized decision variable arrives as a
zero, and the origin is exactly where a squared slack or a homogeneous
quadratic loses rank. These tests pin the things that move when the
ladder becomes a library feature.
"""

import numpy as np
import pytest

import pounce


class NoRealSolution:
    """`x² + 1 = 0` over `x ∈ [-10, 10]`. Genuinely infeasible, and not
    provably so by presolve, so it reaches the ladder and survives it."""

    def objective(self, x):
        return float(x[0] ** 2)

    def gradient(self, x):
        return np.array([2.0 * x[0]])

    def constraints(self, x):
        return np.array([float(x[0] ** 2 + 1.0)])

    def jacobianstructure(self):
        return (np.array([0]), np.array([0]))

    def jacobian(self, x):
        return np.array([2.0 * x[0]])


class Quadratic:
    """`min (x - 3)²`, unconstrained apart from bounds. Converges."""

    def objective(self, x):
        return float((x[0] - 3.0) ** 2)

    def gradient(self, x):
        return np.array([2.0 * (x[0] - 3.0)])


LADDER_OFF = {
    "feral_infeasibility_scaling_retry": "no",
    "infeasibility_mu_strategy_retry": "no",
    "infeasibility_perturbed_start_retry": "no",
    # Rung 4 (gh#857). It is the conditional rung — see `_expected_rungs` —
    # but "the whole ladder off" has to name it regardless, or the one model
    # here that does escalate still pays for a solve with the ladder off.
    "feral_increase_quality_retry": "no",
    # Rung 5 opens only when the probe finds a saddle, which no model here
    # has, but "the whole ladder off" names it for the same reason.
    "infeasibility_saddle_escape_retry": "no",
}

#: The rungs the ladder always has, in order.
FIXED_RUNGS = [
    "feral_scaling=mc64",
    "mu_strategy=adaptive",
    "start_point_perturbation=1e-2",
]

#: Rung 4, which undoes a linear-solver quality escalation the base solve
#: made (gh#857).
QUALITY_RUNG = "feral_increase_quality=no"


def _solve(obj, n=1, m=1, x0=(0.5,), **options):
    kw = dict(n=n, m=m, problem_obj=obj, lb=[-10.0] * n, ub=[10.0] * n)
    if m:
        kw.update(cl=[0.0] * m, cu=[0.0] * m)
    prob = pounce.Problem(**kw)
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    for k, v in options.items():
        prob.add_option(k, v)
    return prob.solve(list(x0))


def _expected_rungs(*, without=()):
    """The rung list this platform should produce, measured rather than
    written down.

    Rung 4 opens only when the base solve escalated the linear solver's pivot
    quality at least once — undoing an escalation that never happened is a
    re-run of the solve that just failed. Whether a given model escalates is
    a property of the platform's arithmetic and not of the model: gh#857's
    own CLI fixture `deb7` escalates twice on macOS/aarch64 and zero times on
    linux/x86_64, and a test that writes a count down instead of measuring it
    is red on one of the two. `NoRealSolution` happens to escalate once on
    both, so the four-rung branch is the one taken here — but that is the
    measurement's answer, not this test's premise.

    The base solve is run with the ladder off, so its `quality_escalations`
    is the base solve's own count and not the last rung's (the counter resets
    per solve).
    """
    _, base = _solve(NoRealSolution(), **LADDER_OFF)
    rungs = list(FIXED_RUNGS)
    if base["quality_escalations"] >= 1:
        rungs.append(QUALITY_RUNG)
    return [r for r in rungs if r not in without]


def test_a_converged_solve_pays_nothing():
    """The common path. `second_opinion` is `None`, so no extra solve ran."""
    _, info = _solve(Quadratic(), m=0, x0=(0.0,))
    assert info["status_msg"] == "Solve_Succeeded"
    assert info["second_opinion"] is None


def test_an_infeasible_verdict_is_re_solved_every_way_before_it_ships():
    _, info = _solve(NoRealSolution())
    assert info["status_msg"] == "Infeasible_Problem_Detected"
    so = info["second_opinion"]
    assert so is not None, "the ladder did not run — this is the CLI-only bug"
    assert so["tried"] == _expected_rungs()
    # Nothing recovers a problem that has no solution, and the original
    # verdict is the one that ships.
    assert so["promoted_by"] is None
    # The narration the CLI prints to stderr is collected, not printed.
    assert any("keeping the original" in line for line in so["log"])


def test_each_rung_can_be_turned_off_independently():
    _, info = _solve(NoRealSolution(), infeasibility_mu_strategy_retry="no")
    assert info["second_opinion"]["tried"] == _expected_rungs(
        without=("mu_strategy=adaptive",)
    )
    # Rung 4 carries its own switch too, and this is the arm that pins it:
    # dropping it leaves exactly the three fixed rungs on every platform,
    # whether or not the base solve escalated.
    _, info = _solve(NoRealSolution(), feral_increase_quality_retry="no")
    assert info["second_opinion"]["tried"] == FIXED_RUNGS


def test_turning_the_whole_ladder_off_restores_upstream_behaviour():
    _, info = _solve(NoRealSolution(), **LADDER_OFF)
    assert info["status_msg"] == "Infeasible_Problem_Detected"
    assert info["second_opinion"] is None


def test_a_ladder_run_does_not_leak_its_options_into_the_next_solve():
    """Each rung writes `feral_scaling` / `mu_strategy` /
    `start_point_perturbation` into the live options list. The CLI could
    leave them there because the process was about to exit; a `Problem`
    solved twice cannot, or the second solve starts from rung 3's displaced
    point instead of the one the caller passed."""
    prob = pounce.Problem(
        n=1, m=1, problem_obj=NoRealSolution(), lb=[-10.0], ub=[10.0],
        cl=[0.0], cu=[0.0],
    )
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    x_first, info_first = prob.solve([0.5])
    assert info_first["second_opinion"] is not None  # the ladder did run
    x_second, info_second = prob.solve([0.5])
    # Same problem, same start, same options ⇒ same answer. If rung 3's
    # `start_point_perturbation=1e-2` survived the first call, the second
    # solve starts somewhere else and this fails.
    assert info_second["status_msg"] == info_first["status_msg"]
    assert info_second["iter_count"] == info_first["iter_count"]
    np.testing.assert_allclose(x_second, x_first, rtol=0, atol=0)


def test_the_ladder_stays_out_of_the_multi_start_path():
    """`solve_nlp_batch` deliberately does not ladder: a failed start is
    routine in a multi-start search, and up to three extra solves per failed
    start multiplies its cost for no benefit. It goes through
    `solve_problem_batch`, not `Problem.solve`, so it never reaches the
    driver."""
    prob = pounce.Problem(
        n=1, m=1, problem_obj=NoRealSolution(), lb=[-10.0], ub=[10.0],
        cl=[0.0], cu=[0.0],
    )
    res = pounce.solve_nlp_batch(
        [prob], x0s=[[0.5]], options={"print_level": 0, "sb": "yes"}
    )
    assert len(res) == 1
    _x, info = res[0]
    assert info["status_msg"] == "Infeasible_Problem_Detected"
    assert "second_opinion" not in info


def test_the_ladder_stays_out_of_the_sensitivity_path():
    """`solve_with_sens` deliberately does not ladder either, and says so by
    setting the key rather than omitting it.

    A sensitivity result is *about a particular solution*: the reduced
    Hessian and the parametric step are built from the factorization the
    solve ended on, and rung 3 displaces the very starting point the caller
    chose. Laddering here would silently answer a different question. The key
    is present and `None` — not absent — so `info` has the same shape as
    `solve`'s and a caller can read `info["second_opinion"]` on either.

    The mutation guard is the pair: on the *same* failing problem the plain
    `solve` below does open the ladder, so `None` here is the sensitivity
    path's own choice and not an artifact of the model.
    """
    prob = pounce.Problem(
        n=1, m=1, problem_obj=NoRealSolution(), lb=[-10.0], ub=[10.0],
        cl=[0.0], cu=[0.0],
    )
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    _x, info = prob.solve_with_sens(
        x0=[0.5], pin_constraint_indices=[0], deltas=[0.0]
    )
    assert info["status_msg"] == "Infeasible_Problem_Detected"
    assert "second_opinion" in info
    assert info["second_opinion"] is None

    _x2, info2 = _solve(NoRealSolution())
    assert info2["status_msg"] == "Infeasible_Problem_Detected"
    assert info2["second_opinion"] is not None


def test_the_solve_report_records_the_ladder(tmp_path):
    """`solve(report_path=...)` carries the `second_opinion` block the CLI's
    report does (gh #850). Without it a Python-written report showed a laddered
    solve as the base solve alone; a converged solve still carries none."""
    import json

    path = tmp_path / "report.json"
    prob = pounce.Problem(n=1, m=1, problem_obj=NoRealSolution(), lb=[-10.0], ub=[10.0], cl=[0.0], cu=[0.0])
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    _, info = prob.solve([0.5], report_path=str(path))
    block = json.loads(path.read_text())["second_opinion"]
    assert block["tried"] == info["second_opinion"]["tried"]
    assert block.get("promoted_by") == info["second_opinion"]["promoted_by"]
    assert block["base_status"] == "Infeasible_Problem_Detected"
    assert len(block["rung_iteration_counts"]) == len(block["tried"])
    assert block["total_iteration_count"] == block["base_iteration_count"] + sum(
        block["rung_iteration_counts"]
    )

    path = tmp_path / "converged.json"
    prob = pounce.Problem(n=1, m=0, problem_obj=Quadratic(), lb=[-10.0], ub=[10.0])
    prob.add_option("print_level", 0)
    prob.solve([0.0], report_path=str(path))
    assert "second_opinion" not in json.loads(path.read_text())
