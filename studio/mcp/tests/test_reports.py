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
    header = json.loads(d.header())
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


# --- MCP tool wrapper tests ------------------------------------------
#
# FastMCP's @mcp.tool() decorator returns the original function so the
# wrapped function stays directly callable. We exercise each tool's
# Python path here without going through the stdio protocol.


def test_tool_load_solve_report():
    from pounce_studio_mcp.server import load_solve_report

    out = load_solve_report(str(ROSENBROCK))
    assert out["status"] == "SolveSucceeded"


def test_tool_convergence_trace_full_and_subset():
    from pounce_studio_mcp.server import convergence_trace

    full = convergence_trace(str(ROSENBROCK))
    assert set(full) >= {"iter", "objective", "inf_pr", "inf_du", "mu"}
    sub = convergence_trace(str(ROSENBROCK), columns=["iter", "mu"])
    assert set(sub) == {"iter", "mu"}


def test_tool_convergence_trace_rejects_unknown_column():
    from pounce_studio_mcp.server import convergence_trace

    with pytest.raises(ValueError):
        convergence_trace(str(ROSENBROCK), columns=["bogus"])


def test_tool_find_stalls_returns_count():
    from pounce_studio_mcp.server import find_stalls

    out = find_stalls(str(ROSENBROCK))
    assert "windows" in out and "count" in out
    assert out["count"] == len(out["windows"])


def test_tool_diagnose_includes_findings_key():
    from pounce_studio_mcp.server import diagnose

    out = diagnose(str(ROSENBROCK))
    assert "findings" in out and "n_findings" in out
    assert out["n_findings"] == len(out["findings"])


def test_tool_restoration_windows():
    from pounce_studio_mcp.server import restoration_windows

    out = restoration_windows(str(ROSENBROCK))
    assert out["windows"] == [] and out["count"] == 0


def test_tool_get_iterate():
    from pounce_studio_mcp.server import get_iterate

    out = get_iterate(str(ROSENBROCK), 0)
    assert out["iter"] == 0


def test_tool_compare_runs_label_mismatch_raises():
    from pounce_studio_mcp.server import compare_runs

    with pytest.raises(ValueError):
        compare_runs([str(ROSENBROCK), str(STALLED)], labels=["only-one"])


def test_tool_compare_runs():
    from pounce_studio_mcp.server import compare_runs

    out = compare_runs([str(ROSENBROCK), str(STALLED)], labels=["ok", "stalled"])
    assert out["n_runs"] == 2


def test_tool_linear_solver_summary_present_for_feral_run():
    from pounce_studio_mcp.server import linear_solver_summary

    out = linear_solver_summary(str(ROSENBROCK))
    assert out["available"] is True, out
    s = out["summary"]
    assert s["solver_name"] == "feral"
    assert s["n_factors"] >= 1
    # Pattern reuse + pattern changes should partition factors.
    assert s["n_pattern_reuse"] + s["n_pattern_changes"] == s["n_factors"]
    # Final inertia recorded as a 3-tuple.
    assert len(s["last_inertia"]) == 3


def test_tool_linear_solver_summary_absent_for_legacy_report(tmp_path):
    """Reports written before the field existed deserialize with `linear_solver = null`."""
    from pounce_studio_mcp.server import linear_solver_summary

    legacy = json.loads(ROSENBROCK.read_text())
    legacy.pop("linear_solver", None)
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy))
    out = linear_solver_summary(str(p))
    assert out == {"available": False, "summary": None}


# --- explain / citations ---------------------------------------------


def test_explain_known_column():
    from pounce_studio_mcp.server import explain

    out = explain("inf_pr")
    assert out["kind"] == "column" and out["term"] == "inf_pr"
    assert "definition" in out and "see_also" in out


def test_explain_known_finding():
    from pounce_studio_mcp.server import explain

    out = explain("mu_stuck")
    assert out["kind"] == "finding" and out["severity"] == "warning"


def test_explain_unknown_suggests():
    from pounce_studio_mcp.server import explain

    with pytest.raises(ValueError) as exc:
        explain("inf_p")  # near-miss for inf_pr
    assert "inf_pr" in str(exc.value)


def test_citations_no_args_lists_topics():
    from pounce_studio_mcp.server import citations

    out = citations()
    assert "topics" in out
    assert "restoration" in out["topics"]
    # Either we found the bib file or we didn't, but the call must work.
    assert "n_entries_loaded" in out


def test_citations_by_topic():
    from pounce_studio_mcp.server import citations

    out = citations(topic="restoration")
    assert out["topic"] == "restoration"
    assert any(e["key"] == "wachter2006" for e in out["entries"])


def test_citations_by_key():
    from pounce_studio_mcp.server import citations
    from pounce_studio_mcp import glossary as G

    # Skip if we couldn't locate the bib (e.g. installed-out-of-tree).
    if not G.load_bib():
        pytest.skip("references.bib not locatable")
    out = citations(key="wachter2006")
    assert out["key"] == "wachter2006"
    assert "title" in out


def test_citations_rejects_both_args():
    from pounce_studio_mcp.server import citations

    with pytest.raises(ValueError):
        citations(topic="restoration", key="wachter2006")


def test_citations_unknown_topic():
    from pounce_studio_mcp.server import citations

    with pytest.raises(ValueError):
        citations(topic="not-a-subsystem")


def test_findings_codes_match_rust_source():
    """Regression guard: every finding the Rust diagnose() can emit
    must have a glossary entry. Update glossary.FINDINGS when a new
    code is added on the Rust side."""
    from pounce_studio_mcp import glossary as G
    # Source of truth — keep in lockstep with analysis.rs.
    rust_codes = {
        "converged", "max_iter_exceeded", "restoration_used", "mu_stuck",
        "heavy_line_search", "hessian_regularized", "restoration_loop",
        "convergence_stall",
    }
    missing = rust_codes - set(G.FINDINGS)
    assert not missing, f"glossary missing finding codes: {missing}"


# --- analyze_problem / run_problem -----------------------------------


def _pounce_available() -> bool:
    from pounce_studio_mcp.server import _find_pounce_bin
    try:
        _find_pounce_bin()
        return True
    except FileNotFoundError:
        return False


PARAMETRIC_NL = (
    Path(__file__).parent.parent.parent.parent
    / "crates" / "pounce-cli" / "tests" / "fixtures" / "parametric.nl"
)


def test_analyze_builtin_rosenbrock():
    from pounce_studio_mcp.server import analyze_problem

    out = analyze_problem(builtin="rosenbrock")
    assert out["kind"] == "builtin"
    assert out["name"] == "rosenbrock"
    assert out["dimensions"]["n_variables"] == 2
    assert "unconstrained" in out["class"]
    assert isinstance(out["suggestions"], list)


def test_analyze_builtin_unknown_raises():
    from pounce_studio_mcp.server import analyze_problem

    with pytest.raises(ValueError):
        analyze_problem(builtin="not-a-real-problem")


def test_analyze_requires_exactly_one_input():
    from pounce_studio_mcp.server import analyze_problem

    with pytest.raises(ValueError):
        analyze_problem()
    with pytest.raises(ValueError):
        analyze_problem(builtin="rosenbrock", nl_file="x.nl")


def test_analyze_nl_file_dimensions():
    from pounce_studio_mcp.server import analyze_problem

    if not PARAMETRIC_NL.exists():
        pytest.skip(f"fixture not present at {PARAMETRIC_NL}")
    out = analyze_problem(nl_file=str(PARAMETRIC_NL))
    assert out["kind"] == "nl_file"
    dims = out["dimensions"]
    assert dims["format"] == "text"
    assert dims["n_variables"] == 5
    assert dims["n_constraints"] == 4
    assert dims["n_objectives"] == 1
    assert dims["n_nonlinear_objectives"] == 1
    assert "class" in out


def test_analyze_missing_nl_file():
    from pounce_studio_mcp.server import analyze_problem

    with pytest.raises(FileNotFoundError):
        analyze_problem(nl_file="/tmp/definitely-not-there.nl")


@pytest.mark.skipif(not _pounce_available(), reason="pounce binary not built")
def test_run_problem_rosenbrock_includes_analysis(tmp_path: Path):
    from pounce_studio_mcp.server import run_problem

    out_json = tmp_path / "rosenbrock.json"
    out = run_problem(builtin="rosenbrock", json_output=str(out_json))
    assert out["exit_code"] == 0
    assert out["report_path"] == str(out_json)
    assert out_json.exists()
    assert out["summary"]["status"] == "SolveSucceeded"
    assert out["analysis"]["kind"] == "builtin"
    assert out["analysis"]["name"] == "rosenbrock"


@pytest.mark.skipif(not _pounce_available(), reason="pounce binary not built")
def test_run_problem_forwards_options(tmp_path: Path):
    from pounce_studio_mcp.server import run_problem

    out_json = tmp_path / "rosenbrock-capped.json"
    out = run_problem(
        builtin="rosenbrock",
        json_output=str(out_json),
        options={"max_iter": "2"},
        analyze=False,
    )
    # max_iter=2 means we either converge in <=2 iters or trip max-iter.
    # Either way the option made it onto argv.
    assert "max_iter=2" in out["argv"]
    assert "analysis" not in out


def test_run_problem_requires_exactly_one_input():
    from pounce_studio_mcp.server import run_problem

    with pytest.raises(ValueError):
        run_problem()
    with pytest.raises(ValueError):
        run_problem(builtin="rosenbrock", nl_file="x.nl")


def test_run_problem_rejects_bad_json_detail():
    from pounce_studio_mcp.server import run_problem

    with pytest.raises(ValueError):
        run_problem(builtin="rosenbrock", json_detail="medium")


def test_memoization_returns_consistent_results():
    # The Rust side caches summarize/diagnose/etc.; call twice and
    # verify the results match (the cache must return the same JSON).
    r = R.load_report(ROSENBROCK)
    a = R.diagnose(r)
    b = R.diagnose(r)
    assert a == b
    a2 = R.summarize(r)
    b2 = R.summarize(r)
    assert a2 == b2
