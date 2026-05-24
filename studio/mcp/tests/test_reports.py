"""Smoke tests for the Rust-backed analysis helpers against bundled fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pounce_studio_mcp import reports as R
from pounce_studio_mcp import _native


FIXTURES = Path(__file__).parent.parent / "fixtures"
ROSENBROCK = FIXTURES / "rosenbrock.json"
STALLED = FIXTURES / "rosenbrock-stalled.json"


def test_native_module_exposes_constants():
    assert _native.SOLVE_REPORT_SCHEMA == "pounce.solve-report/v1"
    assert R.SCHEMA == _native.SOLVE_REPORT_SCHEMA


def test_load_report_returns_native_handle():
    r = R.load_report(ROSENBROCK)
    assert isinstance(r, _native.Report)


def test_load_report_rejects_unknown_schema(tmp_path: Path):
    bogus = tmp_path / "bogus.json"
    bogus.write_text(json.dumps({"schema": "other/v1"}))
    with pytest.raises(R.ReportError):
        R.load_report(bogus)


def test_load_report_rejects_missing_file(tmp_path: Path):
    with pytest.raises(R.ReportError):
        R.load_report(tmp_path / "missing.json")


def test_summarize_keys():
    s = R.summarize(R.load_report(ROSENBROCK))
    assert s["status"] == "SolveSucceeded"
    assert s["iteration_count"] >= 1
    assert s["iterations_captured"] >= 1
    assert s["restoration_calls"] == 0
    assert s["n_variables"] == 2


def test_convergence_trace_full():
    t = R.convergence_trace(R.load_report(ROSENBROCK))
    n = len(t["iter"])
    assert n >= 1
    for k, v in t.items():
        assert len(v) == n, f"column {k} length mismatch"


def test_get_iterate_in_range():
    row = R.get_iterate(R.load_report(ROSENBROCK), 0)
    # `AugmentedIterate` flattens IterRecord fields onto the top level
    # and appends log10_* derived fields alongside.
    assert row["iter"] == 0
    assert "log10_mu" in row
    assert "log10_inf_pr" in row


def test_get_iterate_out_of_range():
    with pytest.raises(R.ReportError):
        R.get_iterate(R.load_report(ROSENBROCK), 9999)


def test_diagnose_success_case():
    out = R.diagnose(R.load_report(ROSENBROCK))
    codes = {f["code"] for f in out["findings"]}
    assert "converged" in codes
    # clean convergence shouldn't trip the stall warning
    assert "convergence_stall" not in codes


def test_diagnose_stalled_case():
    out = R.diagnose(R.load_report(STALLED))
    codes = {f["code"] for f in out["findings"]}
    assert "max_iter_exceeded" in codes


def test_find_stalls_returns_list():
    stalls = R.find_stalls(R.load_report(ROSENBROCK))
    assert isinstance(stalls, list)


def test_restoration_windows_empty_on_clean_run():
    assert R.restoration_windows(R.load_report(ROSENBROCK)) == []


def test_compare_two_runs():
    a = R.load_report(ROSENBROCK)
    b = R.load_report(STALLED)
    cmp = R.compare([("ok", a), ("stalled", b)])
    assert cmp["n_runs"] == 2
    labels = [row["label"] for row in cmp["rows"]]
    assert labels == ["ok", "stalled"]


def test_compare_accepts_aliased_handle():
    # Regression: passing the same Report object twice used to panic
    # because compare_reports_json took PyRef<Report> and two PyRefs
    # to the same RefCell overlapped.
    a = R.load_report(ROSENBROCK)
    cmp = R.compare([("first", a), ("second", a)])
    assert cmp["n_runs"] == 2
    labels = [row["label"] for row in cmp["rows"]]
    assert labels == ["first", "second"]
    # Both rows must describe the same underlying run.
    assert cmp["rows"][0]["status"] == cmp["rows"][1]["status"]


def test_render_markdown():
    md = R.render_markdown(R.load_report(ROSENBROCK))
    assert "# Pounce solve report" in md
    assert "SolveSucceeded" in md


def test_iter_dump_parses_real_trace():
    iterdump = FIXTURES / "eq-quadratic.iterdump"
    d = _native.IterDump.from_path(str(iterdump))
    header = json.loads(d.header_json())
    assert header["format_version"] == 1
    assert header["name"] == "eq-quadratic"
    assert d.record_count() >= 1


def test_main_module_does_not_run_server_on_import():
    # Regression: __main__.py used to call main() at module top-level,
    # which would spawn the stdio MCP server on any `import` of the
    # package. Now guarded by `if __name__ == "__main__"`.
    import importlib

    mod = importlib.import_module("pounce_studio_mcp.__main__")
    # Just confirm import returned without hanging or raising; the
    # presence of `main` is verified by the fact that the module
    # imports it from `.server`.
    assert hasattr(mod, "main")
