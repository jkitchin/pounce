"""Smoke tests for the analysis helpers against bundled fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pounce_studio_mcp import reports as R


FIXTURES = Path(__file__).parent.parent / "fixtures"
ROSENBROCK = FIXTURES / "rosenbrock.json"
STALLED = FIXTURES / "rosenbrock-stalled.json"


def test_load_report_round_trip():
    r = R.load_report(ROSENBROCK)
    assert r["schema"] == R.SCHEMA
    assert r["solution"]["status"] == "SolveSucceeded"


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
    assert s["restoration"]["calls"] == 0
    assert s["problem"]["n_variables"] == 2


def test_convergence_trace_full():
    t = R.convergence_trace(R.load_report(ROSENBROCK))
    n = len(t["iter"])
    assert n >= 1
    # all columns same length
    for k, v in t.items():
        assert len(v) == n, f"column {k} length mismatch"


def test_convergence_trace_subset():
    t = R.convergence_trace(R.load_report(ROSENBROCK), columns=["iter", "mu"])
    assert set(t.keys()) == {"iter", "mu"}


def test_convergence_trace_rejects_unknown_column():
    with pytest.raises(R.ReportError):
        R.convergence_trace(R.load_report(ROSENBROCK), columns=["bogus"])


def test_get_iterate_in_range():
    row = R.get_iterate(R.load_report(ROSENBROCK), 0)
    assert row["iter"] == 0
    assert "log10_mu" in row


def test_get_iterate_out_of_range():
    with pytest.raises(R.ReportError):
        R.get_iterate(R.load_report(ROSENBROCK), 9999)


def test_diagnose_success_case():
    out = R.diagnose(R.load_report(ROSENBROCK))
    codes = {f["code"] for f in out["findings"]}
    assert "converged" in codes


def test_diagnose_stalled_case():
    out = R.diagnose(R.load_report(STALLED))
    codes = {f["code"] for f in out["findings"]}
    assert "max_iter_exceeded" in codes


def test_compare_two_runs():
    a = R.load_report(ROSENBROCK)
    b = R.load_report(STALLED)
    cmp = R.compare([("ok", a), ("stalled", b)])
    assert cmp["n_runs"] == 2
    labels = [row["label"] for row in cmp["rows"]]
    assert labels == ["ok", "stalled"]
