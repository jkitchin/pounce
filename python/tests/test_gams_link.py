"""License-free tests for the pure-Python GAMS solver link.

These exercise the whole translate -> build -> solve path through an in-memory
fake :class:`~pounce.gams.gmo_translate.GmoView` (no GAMS / gamsapi needed), the
POUNCE-status -> GAMS-status mapping, the sign conventions, and the
``gamsconfig.yaml`` create/merge/replace logic.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from pounce.gams import (
    link,
    register,
)
from pounce.gams.gmo_translate import POUNCE_INF, problem_from_gmo


#: Every exit of the engine's `ApplicationReturnStatus`, spelled as
#: `upstream_name()` spells it -- which is what `pounce.Solver.solve` reports in
#: `info["status_msg"]`, and so what the link's tables are keyed by. Pinned here
#: rather than only derived from the Rust source so this still means something
#: from an installed wheel, where `crates/` is not shipped; `engine_statuses`
#: cross-checks the pin against the source in a checkout.
ENGINE_STATUSES = (
    "Solve_Succeeded",
    "Solved_To_Acceptable_Level",
    "Infeasible_Problem_Detected",
    "Search_Direction_Becomes_Too_Small",
    "Diverging_Iterates",
    "User_Requested_Stop",
    "Feasible_Point_Found",
    "Maximum_Iterations_Exceeded",
    "Restoration_Failed",
    "Error_In_Step_Computation",
    "Maximum_CpuTime_Exceeded",
    "Maximum_WallTime_Exceeded",
    "Not_Enough_Degrees_Of_Freedom",
    "Invalid_Problem_Definition",
    "Invalid_Option",
    "Invalid_Number_Detected",
    "Unrecoverable_Exception",
    "NonIpopt_Exception_Thrown",
    "Insufficient_Memory",
    "Internal_Error",
)

_RETURN_CODES_RS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "crates", "pounce-nlp", "src", "return_codes.rs")


@pytest.fixture(scope="session")
def engine_statuses():
    """`ENGINE_STATUSES`, checked against `return_codes.rs` when in a checkout.

    The Rust enum is the source of truth: `upstream_name()` is literally the
    string the link matches on. Adding a status there fails this fixture until
    the pin and both links are updated, which is the point -- an exit nobody
    listed is reported to GAMS as an internal error.
    """
    try:
        with open(_RETURN_CODES_RS, encoding="utf-8") as fh:
            src = fh.read()
    except OSError:
        return ENGINE_STATUSES
    start = src.find("fn upstream_name")
    if start < 0:
        return ENGINE_STATUSES
    # Arm bodies only; the file's own tests repeat the names in assertions.
    end = src.find("#[cfg(test)]", start)
    body = src[start:end if end > 0 else len(src)]
    names = re.findall(r'Self::\w+\s*=>\s*"([A-Za-z_]+)"', body)
    assert sorted(names) == sorted(ENGINE_STATUSES), (
        "ENGINE_STATUSES has drifted from ApplicationReturnStatus in "
        f"{_RETURN_CODES_RS}")
    return ENGINE_STATUSES


class HS071View:
    """In-memory fake of a GMO view for the classic HS071 NLP.

    minimize  x0*x3*(x0+x1+x2) + x2
    s.t.      x0*x1*x2*x3 >= 25
              x0^2+x1^2+x2^2+x3^2 == 40
              1 <= xi <= 5
    Known optimum: f* ~= 17.0140173 at (1, 4.743, 3.821, 1.379).

    ``maximize`` and ``with_hessian`` are constructor knobs so the same fake
    drives the minimize/L-BFGS path and the analytical-Hessian / sign-flip
    tests.
    """

    def __init__(self, maximize: bool = False, with_hessian: bool = False):
        self._max = maximize
        self._hess = with_hessian

    def name(self):
        return "hs071"

    def num_vars(self):
        return 4

    def num_cons(self):
        return 2

    def maximize(self):
        return self._max

    def has_hessian(self):
        return self._hess

    def var_lower(self):
        return [1.0, 1.0, 1.0, 1.0]

    def var_upper(self):
        return [5.0, 5.0, 5.0, 5.0]

    def var_init(self):
        return [1.0, 5.0, 5.0, 1.0]

    def con_lower(self):
        return [25.0, 40.0]

    def con_upper(self):
        return [POUNCE_INF, 40.0]

    def jac_structure(self):
        rows = [0, 0, 0, 0, 1, 1, 1, 1]
        cols = [0, 1, 2, 3, 0, 1, 2, 3]
        return rows, cols

    def hess_structure(self):
        # Dense lower triangle of the 4x4 Hessian (cyipopt HS071 layout).
        rows = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
        cols = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
        return rows, cols

    # --- evaluators (native minimize sense) ------------------------------
    def eval_obj(self, x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def eval_grad_obj(self, x):
        return [
            x[3] * (2 * x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ]

    def eval_cons(self, x):
        return [
            x[0] * x[1] * x[2] * x[3],
            x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2,
        ]

    def eval_jac(self, x):
        return [
            x[1] * x[2] * x[3], x[0] * x[2] * x[3], x[0] * x[1] * x[3], x[0] * x[1] * x[2],
            2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3],
        ]

    def hess_lag_value(self, x, lam, obj_weight, con_weight):
        # True Lagrangian Hessian (lower triangle), emulating GMO:
        #   obj_weight * d2f + con_weight * sum_i lam_i * d2c_i
        hf = [
            2 * x[3],                       # (0,0)
            x[3],                           # (1,0)
            0.0,                            # (1,1)
            x[3],                           # (2,0)
            0.0,                            # (2,1)
            0.0,                            # (2,2)
            2 * x[0] + x[1] + x[2],         # (3,0)
            x[0],                           # (3,1)
            x[0],                           # (3,2)
            0.0,                            # (3,3)
        ]
        hc0 = [
            0.0,
            x[2] * x[3],
            0.0,
            x[1] * x[3],
            x[0] * x[3],
            0.0,
            x[1] * x[2],
            x[0] * x[2],
            x[0] * x[1],
            0.0,
        ]
        hc1 = [2.0, 0, 0, 0, 0, 2.0, 0, 0, 0, 2.0]
        return [
            obj_weight * hf[k] + con_weight * (lam[0] * hc0[k] + lam[1] * hc1[k])
            for k in range(10)
        ]


HS071_OPT = 17.0140173

# Statuses that mean POUNCE converged to a usable local solution. Whether the
# IPM parks at "acceptable" vs full "optimal" is a solver-tuning detail (and is
# version-sensitive); the link's job is only to translate / solve / report.
_CONVERGED = {"Solve_Succeeded", "Solved_To_Acceptable_Level"}


# ── translate / solve ────────────────────────────────────────────────────────


def test_problem_from_gmo_dimensions_and_bounds():
    gp = problem_from_gmo(HS071View())
    assert gp.n == 4
    assert gp.m == 2
    assert gp.lb == [1.0] * 4
    assert gp.ub == [5.0] * 4
    assert gp.cl == [25.0, 40.0]
    assert gp.cu == [POUNCE_INF, 40.0]
    assert gp.obj_sign == 1.0
    np.testing.assert_allclose(gp.x0, [1.0, 5.0, 5.0, 1.0])


def test_no_hessian_object_omits_hessian_callbacks():
    gp = problem_from_gmo(HS071View(with_hessian=False))
    assert not hasattr(gp.problem_obj, "hessian")
    assert not hasattr(gp.problem_obj, "hessianstructure")
    assert hasattr(gp.problem_obj, "jacobian")


def test_hessian_object_exposes_hessian_callbacks():
    gp = problem_from_gmo(HS071View(with_hessian=True))
    assert hasattr(gp.problem_obj, "hessian")
    assert hasattr(gp.problem_obj, "hessianstructure")
    rows, cols = gp.problem_obj.hessianstructure()
    assert rows.dtype == np.int64 and cols.dtype == np.int64
    assert len(rows) == 10


def test_solve_view_lbfgs_reaches_hs071_optimum():
    """End-to-end translate -> build -> solve with L-BFGS (no analytical Hessian)."""
    _gp, x, info = link.solve_view(HS071View(with_hessian=False), options={"tol": 1e-8})
    assert info["status_msg"] in _CONVERGED
    assert info["obj_val"] == pytest.approx(HS071_OPT, abs=1e-4)
    np.testing.assert_allclose(x, [1.0, 4.743, 3.821, 1.379], atol=1e-2)


def test_solve_view_with_analytical_hessian_reaches_optimum():
    _gp, x, info = link.solve_view(HS071View(with_hessian=True), options={"tol": 1e-8})
    assert info["status_msg"] in _CONVERGED
    assert info["obj_val"] == pytest.approx(HS071_OPT, abs=1e-4)


def test_solve_view_writes_solve_report(tmp_path):
    """pounce#187: the pip link honors json_output/json_detail by writing a
    canonical pounce.solve-report/v1 JSON via the Rust writer, so the report is
    not a silent no-op on the pip route."""
    import json

    report = tmp_path / "hs071.report.json"
    _gp, x, info = link.solve_view(
        HS071View(with_hessian=True),
        options={"tol": 1e-8},
        report_path=str(report),
        report_detail="full",
    )
    assert info["status_msg"] in _CONVERGED
    assert report.exists(), "json_output must produce a report file on the pip route"

    doc = json.loads(report.read_text())
    assert doc["schema"] == "pounce.solve-report/v1"
    assert doc["solution"]["status"] == "SolveSucceeded"
    assert doc["problem"]["n_variables"] == 4
    assert doc["problem"]["n_constraints"] == 2
    # `full` detail carries the per-iteration trace the studio/MCP post-mortem
    # tools consume; `summary` omits it.
    assert doc["iterations"], "full detail should include the iteration history"
    assert doc["solution"]["objective"] == pytest.approx(HS071_OPT, abs=1e-4)

    # `summary` detail writes a valid report but drops the iteration history.
    summary = tmp_path / "hs071.summary.json"
    link.solve_view(
        HS071View(with_hessian=True),
        options={"tol": 1e-8},
        report_path=str(summary),
        report_detail="summary",
    )
    sdoc = json.loads(summary.read_text())
    assert sdoc["schema"] == "pounce.solve-report/v1"
    assert not sdoc.get("iterations")


def test_maximize_sign_flips_objective_and_gradient():
    gp_min = problem_from_gmo(HS071View(maximize=False))
    gp_max = problem_from_gmo(HS071View(maximize=True))
    x = np.array([1.0, 5.0, 5.0, 1.0])
    assert gp_max.obj_sign == -1.0
    assert gp_max.problem_obj.objective(x) == pytest.approx(-gp_min.problem_obj.objective(x))
    np.testing.assert_allclose(
        gp_max.problem_obj.gradient(x), -gp_min.problem_obj.gradient(x)
    )


def test_hessian_callback_applies_sign_and_conweight():
    """hessian() must call hess_lag_value with obj_sign*obj_factor and conweight=-1."""

    captured = {}

    class RecordingView(HS071View):
        def hess_lag_value(self, x, lam, obj_weight, con_weight):
            captured["obj_weight"] = obj_weight
            captured["con_weight"] = con_weight
            captured["lam"] = list(lam)
            return [0.0] * 10

    # minimize: obj_weight == obj_factor, conweight == -1, lambda passed through.
    gp = problem_from_gmo(RecordingView(maximize=False, with_hessian=True))
    gp.problem_obj.hessian(np.ones(4), np.array([2.0, 3.0]), 0.5)
    assert captured["obj_weight"] == pytest.approx(0.5)
    assert captured["con_weight"] == pytest.approx(-1.0)
    assert captured["lam"] == [2.0, 3.0]

    # maximize: obj_weight negated.
    gp = problem_from_gmo(RecordingView(maximize=True, with_hessian=True))
    gp.problem_obj.hessian(np.ones(4), np.array([1.0, 1.0]), 0.5)
    assert captured["obj_weight"] == pytest.approx(-0.5)
    assert captured["con_weight"] == pytest.approx(-1.0)


# ── status mapping ────────────────────────────────────────────────────────────


# A nine-row table checking `status_to_gams` against this module's own
# `SOLVESTAT_*` constants used to live here. It is gone rather than extended:
# comparing the table to the constants it is built from cannot fail, whatever
# the constants say, and three of them were wrong. What replaced it --
# `test_status_to_gams_matches_the_c_link` below -- covers all twenty exits
# against literal GAMS integers taken from the C link.


def test_status_to_gams_unknown_is_error():
    assert link.status_to_gams("Something_New") == (
        link.MODELSTAT_ERROR_NO_SOLUTION,
        link.SOLVESTAT_INTERNAL_ERR,
    )


# ── status mapping: agreement with the C link (gh #589) ───────────────────────
#
# The two links are supposed to report identically -- this one is a port of
# `map_status_to_gams()` / `pounce_status_has_solution()` in
# `gams/gams_pounce.c`. Nothing checked that, and three things had drifted: the
# `gmoSolveStat_*` constants below were wrong, four statuses used the wrong one
# of them, and three exits were missing from the table entirely. The tests here
# pin each of those against a source that cannot drift with us -- GAMS's own
# `gams.core.gmo` for the constants, the Rust enum for the status names.

#: `gmoSolveStat_*` / `gmoModelStat_*` spellings for the constants this module
#: defines, so gamsapi can be asked what they should be.
_GMO_CONSTANTS = {
    "MODELSTAT_OPTIMAL": "gmoModelStat_OptimalGlobal",
    "MODELSTAT_LOCALLY_OPTIMAL": "gmoModelStat_OptimalLocal",
    "MODELSTAT_UNBOUNDED": "gmoModelStat_Unbounded",
    "MODELSTAT_INFEASIBLE_LOCAL": "gmoModelStat_InfeasibleLocal",
    "MODELSTAT_INFEASIBLE_INTERMED": "gmoModelStat_InfeasibleIntermed",
    "MODELSTAT_FEASIBLE": "gmoModelStat_Feasible",
    "MODELSTAT_ERROR_NO_SOLUTION": "gmoModelStat_ErrorNoSolution",
    "MODELSTAT_NO_SOLUTION_RETURNED": "gmoModelStat_NoSolutionReturned",
    "SOLVESTAT_NORMAL": "gmoSolveStat_Normal",
    "SOLVESTAT_ITERATION": "gmoSolveStat_Iteration",
    "SOLVESTAT_RESOURCE": "gmoSolveStat_Resource",
    "SOLVESTAT_SOLVER": "gmoSolveStat_Solver",
    "SOLVESTAT_EVAL_ERR": "gmoSolveStat_EvalError",
    "SOLVESTAT_USER": "gmoSolveStat_User",
    "SOLVESTAT_SETUP_ERR": "gmoSolveStat_SetupErr",
    "SOLVESTAT_SOLVER_ERR": "gmoSolveStat_SolverErr",
    "SOLVESTAT_INTERNAL_ERR": "gmoSolveStat_InternalErr",
}


def test_status_constants_match_gamsapi():
    """Every status constant against GAMS's own definition.

    `SOLVESTAT_EVAL_ERR` was 11 (`gmoSolveStat_InternalErr`) and
    `SOLVESTAT_INTERNAL_ERR` was 12 (`gmoSolveStat_Skipped`), so this link told
    GAMS something different from what `gams_pounce.c` tells it, and something
    different from what it meant. A wrong integer here is invisible without
    GAMS in the loop, which is why it survived: the link ran, the value was in
    range, and the `.lst` just said the wrong thing.
    """
    gmo = pytest.importorskip(
        "gams.core.gmo",
        reason="needs gamsapi[core] (pure Python, no GAMS license) to read "
               "the authoritative constants")
    for ours, theirs in _GMO_CONSTANTS.items():
        assert getattr(link, ours) == getattr(gmo, theirs), (
            f"link.{ours} disagrees with GAMS's {theirs}")


def test_status_map_covers_every_engine_exit(engine_statuses):
    """An exit missing from the table takes the default silently and is
    reported to GAMS as an internal error — the gh #589 failure mode, here."""
    missing = [s for s in engine_statuses if s not in link._STATUS_MAP]
    assert not missing, f"{missing} fall to the `status_to_gams` default"


def test_has_solution_set_only_names_real_exits(engine_statuses):
    assert not (link._STATUS_HAS_SOLUTION - set(engine_statuses))


#: What `gams/gams_pounce.c` reports, as **literal** GAMS integers.
#:
#: Deliberately not written as `link.SOLVESTAT_*`. Every other assertion in
#: this file compares the table against the module's own constants, which
#: cannot catch a constant that is itself wrong -- and three of them were, so
#: the table "matched" while the two links disagreed. These numbers come from
#: the C link, which uses GAMS's `gmomcc.h` enumerators directly, so a
#: disagreement here is a real disagreement between the two links.
_C_LINK_STATUS = {
    "Solve_Succeeded": (2, 1),  # OptimalLocal, Normal
    "Solved_To_Acceptable_Level": (7, 1),  # Feasible, Normal
    "Feasible_Point_Found": (7, 1),
    "Infeasible_Problem_Detected": (5, 4),  # InfeasibleLocal, Solver
    "Search_Direction_Becomes_Too_Small": (7, 4),
    "Diverging_Iterates": (3, 4),  # Unbounded, Solver
    "User_Requested_Stop": (7, 8),  # Feasible, User
    "Maximum_Iterations_Exceeded": (7, 2),  # Feasible, Iteration
    "Restoration_Failed": (6, 4),  # InfeasibleIntermed, Solver
    "Error_In_Step_Computation": (7, 10),  # Feasible, SolverErr
    "Maximum_CpuTime_Exceeded": (7, 3),  # Feasible, Resource
    "Maximum_WallTime_Exceeded": (7, 3),
    "Not_Enough_Degrees_Of_Freedom": (13, 9),  # ErrorNoSolution, SetupErr
    "Invalid_Problem_Definition": (13, 9),
    "Invalid_Option": (13, 9),
    "Invalid_Number_Detected": (6, 5),  # InfeasibleIntermed, EvalError
    "Insufficient_Memory": (13, 10),  # ErrorNoSolution, SolverErr
    "Unrecoverable_Exception": (13, 11),  # ErrorNoSolution, InternalErr
    "NonIpopt_Exception_Thrown": (13, 11),
    "Internal_Error": (13, 11),
}

#: `pounce_status_has_solution()` in the C link, intersected with the finiteness
#: guard's "yes" case: the exits whose objective reaches GAMS.
_C_LINK_REPORTS_OBJECTIVE = frozenset({
    "Solve_Succeeded", "Solved_To_Acceptable_Level", "Feasible_Point_Found",
    "Infeasible_Problem_Detected", "Search_Direction_Becomes_Too_Small",
    "User_Requested_Stop", "Maximum_Iterations_Exceeded", "Restoration_Failed",
    "Error_In_Step_Computation", "Invalid_Number_Detected",
    "Maximum_CpuTime_Exceeded", "Maximum_WallTime_Exceeded",
})


@pytest.mark.parametrize("status_msg", sorted(_C_LINK_STATUS))
def test_status_to_gams_matches_the_c_link(status_msg):
    """The two links must report identically -- this one is a port of the
    other. Four statuses disagreed: this link said `gmoSolveStat_SolverErr`
    (10, "Solver Failure") where the C link says `gmoSolveStat_Solver` (4,
    "Terminated By Solver"), which is a verdict rather than a crash."""
    assert link.status_to_gams(status_msg) == _C_LINK_STATUS[status_msg]


@pytest.mark.parametrize("status_msg", sorted(_C_LINK_STATUS))
def test_objective_report_matches_the_c_link(status_msg):
    assert link.reports_objective(status_msg, 1.0) == (
        status_msg in _C_LINK_REPORTS_OBJECTIVE)


def test_restoration_failure_reports_its_objective():
    """gh #589 as a GAMS user sees it: `gmoSetSolution2` publishes the
    restoration failure's iterate as `x.l` unconditionally, so withholding the
    objective left the listing showing that point under an objective of 0."""
    assert link.reports_objective("Restoration_Failed", 17.014)


@pytest.mark.parametrize(
    "status_msg", ["Solve_Succeeded", "Restoration_Failed",
                   "Invalid_Number_Detected", "Maximum_Iterations_Exceeded"])
def test_a_non_finite_objective_is_never_reported(status_msg):
    """The finiteness half of the guard. POUNCE leaves `obj_val` at NaN when it
    refused the solve before evaluating anything, and `Invalid_Number_Detected`
    is by definition an exit where something went non-finite."""
    assert not link.reports_objective(status_msg, float("nan"))
    assert not link.reports_objective(status_msg, float("inf"))


@pytest.mark.parametrize(
    "status_msg", ["Invalid_Option", "Not_Enough_Degrees_Of_Freedom",
                   "Insufficient_Memory", "Diverging_Iterates",
                   "Something_New"])
def test_objective_withheld_where_the_c_link_withholds_it(status_msg):
    assert not link.reports_objective(status_msg, 1.0)


# ── option file parsing ───────────────────────────────────────────────────────


def test_parse_option_file(tmp_path):
    opt = tmp_path / "pounce.opt"
    opt.write_text(
        "* a comment\n"
        "# another comment\n"
        "\n"
        "max_iter 200\n"
        "tol 1e-9\n"
        "hessian_approximation limited-memory\n"
        "json_output /tmp/report.json\n"
        "json_detail summary\n"
    )
    pounce_opts, link_opts = link.parse_option_file(str(opt))
    assert pounce_opts["max_iter"] == 200
    assert isinstance(pounce_opts["max_iter"], int)
    assert pounce_opts["tol"] == pytest.approx(1e-9)
    assert pounce_opts["hessian_approximation"] == "limited-memory"
    assert link_opts["json_output"] == "/tmp/report.json"
    assert link_opts["json_detail"] == "summary"
    assert "json_output" not in pounce_opts


# ── argument resolution ───────────────────────────────────────────────────────


def test_parse_gams_args_finds_control_file(tmp_path):
    sysdir = tmp_path / "sys"
    sysdir.mkdir()
    (sysdir / "gmscmpun.txt").write_text("")
    cntr = tmp_path / "gamscntr.dat"
    cntr.write_text("")
    args = ["scrdir", "workdir", "prm.dat", str(cntr), str(sysdir), "pounce"]
    control_file, found_sysdir = link._parse_gams_args(args)
    assert control_file == str(cntr)
    assert found_sysdir == str(sysdir)


def test_parse_gams_args_single_arg(tmp_path):
    cntr = tmp_path / "gamscntr.dat"
    cntr.write_text("")
    control_file, found_sysdir = link._parse_gams_args([str(cntr)])
    assert control_file == str(cntr)
    assert found_sysdir is None


# ── gamsconfig.yaml render / merge ────────────────────────────────────────────


def test_render_gamsconfig_created():
    text, action = register.render_gamsconfig(None, "/opt/pounce-gams-link")
    assert action == "created"
    assert "solverConfig" in text
    assert "pounce" in text
    assert "/opt/pounce-gams-link" in text
    assert "NLP" in text


def test_gamsconfig_snippet_lists_only_continuous_types():
    snippet = register.gamsconfig_snippet()
    assert "NLP" in snippet and "DNLP" in snippet and "RMINLP" in snippet
    # POUNCE is continuous-only; no discrete/global types in the registered set.
    assert "MINLP" not in register.MODEL_TYPES
    assert "MIP" not in register.MODEL_TYPES
    assert set(register.MODEL_TYPES) == {"NLP", "DNLP", "RMINLP"}


def test_render_gamsconfig_merge_preserves_other_solver():
    yaml = pytest.importorskip("yaml")
    existing = yaml.safe_dump(
        {
            "solverConfig": [{"othersolver": {"scriptName": "/x/other", "modelTypes": ["LP"]}}],
            "someOtherKey": {"keep": True},
        }
    )
    text, action = register.render_gamsconfig(existing, "/opt/pounce-gams-link")
    assert action == "merged"
    data = yaml.safe_load(text)
    names = [list(item.keys())[0] for item in data["solverConfig"]]
    assert "othersolver" in names
    assert "pounce" in names
    assert data["someOtherKey"] == {"keep": True}


def test_render_gamsconfig_replace_existing_pounce():
    yaml = pytest.importorskip("yaml")
    existing = yaml.safe_dump(
        {"solverConfig": [{"pounce": {"scriptName": "/old/path", "modelTypes": ["NLP"]}}]}
    )
    text, action = register.render_gamsconfig(existing, "/new/pounce-gams-link")
    assert action == "replaced"
    data = yaml.safe_load(text)
    assert len(data["solverConfig"]) == 1
    assert data["solverConfig"][0]["pounce"]["scriptName"] == "/new/pounce-gams-link"


# ── write / unregister round trip ─────────────────────────────────────────────


def test_write_and_unregister_round_trip(tmp_path):
    pytest.importorskip("yaml")
    written = register.write_registration(tmp_path)
    assert written["action"] == "created"
    assert written["config"].exists()
    assert written["script"].exists()
    assert "pounce" in written["config"].read_text()

    result = register.unregister(tmp_path)
    assert result["removed"] is True
    assert not written["script"].exists()


def test_write_registration_merges_into_existing(tmp_path):
    yaml = pytest.importorskip("yaml")
    config = tmp_path / "gamsconfig.yaml"
    config.write_text(
        yaml.safe_dump(
            {"solverConfig": [{"othersolver": {"scriptName": "/x", "modelTypes": ["LP"]}}]}
        )
    )
    written = register.write_registration(tmp_path)
    assert written["action"] == "merged"
    data = yaml.safe_load(config.read_text())
    names = [list(item.keys())[0] for item in data["solverConfig"]]
    assert "othersolver" in names and "pounce" in names


def test_run_script_variants():
    posix = register.run_script(python_executable="/usr/bin/python3", windows=False)
    assert posix.startswith("#!/bin/sh")
    assert "pounce.gams.link" in posix
    win = register.run_script(python_executable="C:/py/python.exe", windows=True)
    assert "%*" in win
    assert "pounce.gams.link" in win


# --- gh #272: equation marginal sign convention -------------------------


def test_gams_pi_minimizing_negates_lambda():
    """For a minimizing model, pi = -lambda (the historical behavior)."""
    pi = link.gams_pi([1.5, -0.25, 0.0], obj_sign=1.0)
    assert pi == pytest.approx([-1.5, 0.25, 0.0])


def test_gams_pi_maximizing_preserves_lambda_sign():
    """For a maximizing model, pi = +lambda.

    Regression guard for gh #272: the link applied ``-lambda``
    unconditionally, so every equation marginal on a ``maximizing`` model
    came back inverted. Verified live against GAMS 53.2.0/CPLEX, which
    reports +2.25 / +0.25 on the test LP; POUNCE's internal multipliers
    there are +2.25 / +0.25, and the old ``pi = -lambda`` turned them into
    the -2.25 / -0.25 that GAMS displayed.
    """
    pi = link.gams_pi([2.25, 0.25], obj_sign=-1.0)
    assert pi == pytest.approx([2.25, 0.25])


def test_gams_pi_sign_flips_between_senses():
    """The two senses must be exact negations of one another."""
    lam = [3.0, -1.0, 0.5]
    assert link.gams_pi(lam, obj_sign=1.0) == pytest.approx(
        -link.gams_pi(lam, obj_sign=-1.0)
    )


def test_gams_pi_matches_analytic_shadow_price_maximizing():
    """End-to-end sign check against an analytic marginal.

    ``max 2x s.t. x <= 3`` has ``obj* = 6`` and ``d obj / d b = +2``.

    POUNCE minimizes ``-2x`` subject to ``x - 3 <= 0``, whose Lagrangian
    ``L = -2x + lambda (x - 3)`` gives stationarity ``-2 + lambda = 0``,
    i.e. ``lambda = +2``. With ``obj_sign = -1`` the GAMS marginal is
    ``-(-1) * 2 = +2`` -- the sign GAMS's own solvers report. The old
    unconditional negation returned ``-2``.
    """
    assert link.gams_pi([2.0], obj_sign=-1.0) == pytest.approx([2.0])


# --- gh #294: analytic dual-sign regression guard, BOTH senses ---------------


def test_gams_pi_wyndor_shadow_prices_both_senses():
    """Pin the GAMS marginal against the textbook Wyndor Glass shadow prices
    for BOTH a maximizing and a minimizing model (gh #294 hardening).

    The Wyndor LP ``max 3x1+5x2 s.t. x1<=4, 2x2<=12, 3x1+2x2<=18`` has optimum
    36 at (2, 6) with shadow prices ``(0, 1.5, 1)``. POUNCE's internal
    constraint multipliers there are ``lambda = [0, 1.5, 1]`` (the ``+lambda``
    Lagrange convention; verified live and against IPOPT).

    * Maximizing (``obj_sign = -1``): ``pi = -(-1)*lambda = +lambda`` = the
      shadow prices ``[0, 1.5, 1]`` GAMS's own solvers report.
    * Minimizing the equivalent ``min -3x1-5x2`` (``obj_sign = +1``):
      ``pi = -lambda`` = ``[0, -1.5, -1]``.

    A uniform sign flip (the #271/#272 defect) would negate both; the explicit
    signed expectations below fail loudly if that recurs.
    """
    lam = [0.0, 1.5, 1.0]  # POUNCE internal multipliers for the Wyndor optimum
    assert link.gams_pi(lam, obj_sign=-1.0) == pytest.approx([0.0, 1.5, 1.0])
    assert link.gams_pi(lam, obj_sign=1.0) == pytest.approx([0.0, -1.5, -1.0])


# ======================================================================
# Continuation over a GAMS path (pounce#608). The pip link builds an
# ordinary pounce.Problem from a GMO view, so the whole driver applies;
# these drive it on the same license-free fake as everything above.
# ======================================================================


class ParametricHS071View(HS071View):
    """HS071 with the equality right-hand side as the parameter.

    ``x0^2 + x1^2 + x2^2 + x3^2 == theta`` (40 in the original). A path
    in ``theta`` is a path in the equality RHS, which is what a
    sensitivity step's `deltas` argument means.
    """

    def __init__(self, theta, **kw):
        super().__init__(**kw)
        self.theta = float(theta)

    def con_lower(self):
        return [25.0, self.theta]

    def con_upper(self):
        return [POUNCE_INF, self.theta]


def _gams_thetas(k=6):
    return [np.array([40.0 + 0.5 * i]) for i in range(k)]


def test_gams_continuation_traces_a_path():
    from pounce.gams import continuation as gams_cont

    thetas = _gams_thetas()
    trace = gams_cont.trace(
        lambda th: ParametricHS071View(float(np.asarray(th).ravel()[0]),
                                       with_hessian=True),
        thetas,
        options={"tol": 1e-8, "print_level": 0, "sb": "yes"},
    )
    assert trace.n_steps == len(thetas)
    assert trace.status == "ok"
    assert all(st.status in (0, 1) for st in trace.steps)

    # Every traced point matches an independent cold solve at the same
    # parameter: a warm start may make the answer cheaper, never
    # different. (Same contract the Pyomo adapter's tests assert.)
    for th, st in zip(thetas, trace.steps):
        view = ParametricHS071View(float(th[0]), with_hessian=True)
        _gp, x, info = link.solve_view(
            view, options={"tol": 1e-8, "print_level": 0, "sb": "yes"})
        assert st.obj == pytest.approx(info["obj_val"], rel=1e-6)


def test_gams_continuation_warm_start_beats_cold_on_the_path():
    """The reason to route GAMS through the driver at all."""
    from pounce.gams import continuation as gams_cont

    thetas = _gams_thetas(8)
    opts = {"tol": 1e-8, "print_level": 0, "sb": "yes"}
    trace = gams_cont.trace(
        lambda th: ParametricHS071View(float(np.asarray(th).ravel()[0]),
                                       with_hessian=True),
        thetas, options=opts,
    )
    cold = 0
    for th in thetas:
        _gp, _x, info = link.solve_view(
            ParametricHS071View(float(th[0]), with_hessian=True),
            options=opts)
        cold += int(info["iter_count"])
    assert trace.total_iters < cold


def test_gams_continuation_reports_the_predictor_it_got():
    """`pins` is what enables the tangent; without it the driver runs
    the zero-order fallback, and says so rather than leaving it to be
    inferred from timings."""
    from pounce.gams import continuation as gams_cont

    def view_of(th):
        return ParametricHS071View(float(np.asarray(th).ravel()[0]),
                                   with_hessian=True)

    opts = {"tol": 1e-8, "print_level": 0, "sb": "yes"}
    zero = gams_cont.trace(view_of, _gams_thetas(4), options=opts)
    assert {st.predictor for st in zero.steps} == {"cold", "zero"}


def test_gams_continuation_rejects_a_view_factory_returning_none():
    from pounce.gams import continuation as gams_cont

    with pytest.raises(TypeError, match="must return the GmoView"):
        gams_cont.trace(lambda th: None, [np.array([40.0])])
