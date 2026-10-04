"""gh#990 second pass: results the textbook had to scrape from stderr or the
banner are now programmatic.

* item 3  -- ``info["derivative_check"]`` and the report's
  ``statistics.derivative_check`` carry the derivative checker's verdict;
* item 7  -- ``linear_solver.requested`` names what was asked for beside
  ``solver_name`` (what ran);
* item 9  -- ``QpResult.tau`` / ``kappa`` / ``certificate_scale``;
* item 13 -- ``info["objective_scaling"]`` and ``statistics.objective_scaling``
  carry the objective-scaling decision.
"""

import json

import numpy as np
import pytest

import pounce


class _Rosen:
    def objective(self, x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def gradient(self, x):
        return np.array(
            [-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]
        )


class _WrongGradient(_Rosen):
    def gradient(self, x):
        g = super().gradient(x)
        g[1] += 0.5  # the textbook bug: a dropped term in one entry
        return g


def _problem(obj, **opts):
    p = pounce.Problem(
        n=2, m=0, problem_obj=obj, lb=[-5, -5], ub=[5, 5], cl=[], cu=[]
    )
    p.add_option("print_level", 0)
    p.add_option("hessian_approximation", "limited-memory")
    for k, v in opts.items():
        p.add_option(k, v)
    return p


def _report(p, tmp_path, **kw):
    path = tmp_path / "r.json"
    x, info = p.solve(
        np.array([-1.2, 1.0]), report_path=str(path), report_detail="full", **kw
    )
    return x, info, json.loads(path.read_text())


def test_derivative_check_is_in_info_and_report(tmp_path, capfd):
    p = _problem(_WrongGradient(), derivative_test="first-order", max_iter=0)
    _x, info, rep = _report(p, tmp_path)
    dc = info["derivative_check"]
    assert dc["mode"] == "first-order"
    assert dc["checked"] == 2 and dc["suspicious"] == 1 and not dc["clean"]
    assert dc["max_rel_error_gradient"] > 1e-3
    assert dc["max_rel_error_jacobian"] is None
    (flag,) = dc["flagged"]
    assert (flag["kind"], flag["col"]) == ("gradient", 1)
    assert flag["relative_error"] == pytest.approx(dc["max_rel_error_gradient"])
    # The report carries the same verdict, and stderr keeps its output.
    assert rep["statistics"]["derivative_check"]["suspicious"] == 1
    assert rep["statistics"]["derivative_check"]["flagged"][0]["col"] == 1
    assert "Derivative checker" in capfd.readouterr().err


def test_derivative_check_clean_and_absent(tmp_path):
    _x, info, rep = _report(_problem(_Rosen(), derivative_test="first-order", max_iter=0), tmp_path)
    assert info["derivative_check"]["clean"] is True
    assert info["derivative_check"]["flagged"] == []
    # No test requested: the key is present and None, the report omits it, and
    # a previous solve's verdict is not inherited.
    _x, info, rep = _report(_problem(_Rosen()), tmp_path)
    assert info["derivative_check"] is None
    assert "derivative_check" not in rep["statistics"]


def test_linear_solver_requested_beside_the_backend_that_ran(tmp_path):
    _x, info, rep = _report(_problem(_Rosen()), tmp_path)
    assert info["linear_solver"]["requested"] == "feral"
    assert rep["linear_solver"]["requested"] == "feral"
    # ma57 is not in the default build: FERAL runs, and the banner was the
    # only place that said so.
    _x, info, rep = _report(_problem(_Rosen(), linear_solver="ma57"), tmp_path)
    assert info["linear_solver"]["requested"] == "ma57"
    assert rep["linear_solver"]["requested"] == "ma57"
    if info["linear_solver"]["solver_name"] != "ma57":
        assert info["linear_solver"]["solver_name"] == "feral"


def test_objective_scaling_decision_is_recorded(tmp_path):
    _x, info, rep = _report(_problem(_Rosen()), tmp_path)
    os_ = info["objective_scaling"]
    assert os_["factor"] == pytest.approx(0.46, abs=0.05)
    assert os_["start_gradient_max"] == pytest.approx(215.6)
    assert os_["certificate_refused"] is False
    assert rep["statistics"]["objective_scaling"]["factor"] == pytest.approx(os_["factor"])


def test_refused_certificate_is_recorded_even_when_a_resolve_is_returned(tmp_path):
    # Issue item 13's model: an objective scaled by 6.4e4 from a far start. The
    # first pass refuses a certificate the scaling masked (an INFO line on
    # stderr); the answer returned is the audit's re-solve, so the decision has
    # to survive on the run, not on the returned attempt alone.
    f = lambda x: 6.4e4 * ((x[0] - 1) ** 2 + (x[1] - 2) ** 2) ** 2  # noqa: E731

    class Obj:
        def objective(self, x):
            return f(x)

        def gradient(self, x):
            r = (x[0] - 1) ** 2 + (x[1] - 2) ** 2
            return 6.4e4 * 4 * r * np.array([x[0] - 1, x[1] - 2])

    p = pounce.Problem(
        n=2, m=0, problem_obj=Obj(), lb=[-1e3, -1e3], ub=[1e3, 1e3], cl=[], cu=[]
    )
    p.add_option("print_level", 0)
    p.add_option("hessian_approximation", "limited-memory")
    x, info = p.solve(np.array([60.0, 70.0]))
    assert np.allclose(x, [1.0, 2.0], atol=1e-3)
    assert info["objective_scaling"]["certificate_refused"] is True


def test_qp_result_exposes_tau_kappa_and_certificate_scale():
    G = np.array([[1.0, 1.0], [-1.0, -1.0]])
    h = np.array([1.0, -3.0])  # x1 + x2 <= 1 and >= 3
    r = pounce.solve_qp(c=np.array([1.0, 2.0]), G=G, h=h, lb=np.zeros(2))
    assert r.status == "primal_infeasible"
    assert r.tau is not None and r.kappa is not None
    assert r.kappa > 0 and r.tau < 1e-6 * r.kappa  # the infeasibility ray
    scale = r.certificate_scale
    assert scale == pytest.approx(max(np.abs(r.z).max(), np.abs(r.z_lb).max()))
    # The documented use: divide by it for a unit-norm certificate.
    assert np.abs(r.z / scale).max() <= 1.0 + 1e-12

    ok = pounce.solve_qp(
        c=np.array([1.0, 2.0]), G=-np.eye(2), h=np.zeros(2), lb=np.zeros(2)
    )
    assert ok.status == "optimal"
    assert ok.certificate_scale is None
    assert ok.tau is None or ok.tau > 0
