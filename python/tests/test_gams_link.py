"""License-free tests for the pure-Python GAMS solver link.

These exercise the whole translate -> build -> solve path through an in-memory
fake :class:`~pounce.gams.gmo_translate.GmoView` (no GAMS / gamsapi needed), the
POUNCE-status -> GAMS-status mapping, the sign conventions, and the
``gamsconfig.yaml`` create/merge/replace logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from pounce.gams import (
    link,
    register,
)
from pounce.gams.gmo_translate import POUNCE_INF, problem_from_gmo


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


@pytest.mark.parametrize(
    "status_msg, model_stat, solve_stat",
    [
        ("Solve_Succeeded", link.MODELSTAT_LOCALLY_OPTIMAL, link.SOLVESTAT_NORMAL),
        ("Solved_To_Acceptable_Level", link.MODELSTAT_FEASIBLE, link.SOLVESTAT_NORMAL),
        ("Feasible_Point_Found", link.MODELSTAT_FEASIBLE, link.SOLVESTAT_NORMAL),
        ("Infeasible_Problem_Detected", link.MODELSTAT_INFEASIBLE_LOCAL, link.SOLVESTAT_SOLVER_ERR),
        ("Diverging_Iterates", link.MODELSTAT_UNBOUNDED, link.SOLVESTAT_SOLVER_ERR),
        ("Maximum_Iterations_Exceeded", link.MODELSTAT_FEASIBLE, link.SOLVESTAT_ITERATION),
        ("Maximum_WallTime_Exceeded", link.MODELSTAT_FEASIBLE, link.SOLVESTAT_RESOURCE),
        ("Invalid_Option", link.MODELSTAT_ERROR_NO_SOLUTION, link.SOLVESTAT_SETUP_ERR),
        ("Internal_Error", link.MODELSTAT_ERROR_NO_SOLUTION, link.SOLVESTAT_INTERNAL_ERR),
    ],
)
def test_status_to_gams(status_msg, model_stat, solve_stat):
    assert link.status_to_gams(status_msg) == (model_stat, solve_stat)


def test_status_to_gams_unknown_is_error():
    assert link.status_to_gams("Something_New") == (
        link.MODELSTAT_ERROR_NO_SOLUTION,
        link.SOLVESTAT_INTERNAL_ERR,
    )


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
