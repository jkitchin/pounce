"""Tests for the GAMS-link MCP tools (analyze / parse / list / run)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pounce_studio_mcp.server import (
    _parse_gms_convert_header,
    _parse_gms_solve_directive,
    _parse_lst_solve_summary,
    analyze_gams_problem,
    list_gams_examples,
    parse_gams_listing,
    run_gams_problem,
)


FIXTURES = Path(__file__).parent.parent / "fixtures"
EX8_GMS = FIXTURES / "ex8_3_10_header.gms"
HANDWRITTEN_GMS = FIXTURES / "handwritten.gms"
EX8_LST = FIXTURES / "ex8_3_10.lst"
SPAWN_FAILED_LST = FIXTURES / "spawn_failed.lst"


# ---- analyze_gams_problem -------------------------------------------


def test_analyze_parses_convert_header_dims():
    r = analyze_gams_problem(str(EX8_GMS))
    d = r["dimensions"]
    assert d["n_variables_total"] == 142
    assert d["n_continuous_vars"] == 142
    assert d["n_binary_vars"] == 0
    assert d["n_integer_vars"] == 0
    assert d["n_equations_total"] == 109
    assert d["n_equality_eqs"] == 108
    assert d["n_le_eqs"] == 1
    assert d["nnz_total"] == 729
    assert d["nnz_nonlinear"] == 567


def test_analyze_parses_solve_directive():
    r = analyze_gams_problem(str(EX8_GMS))
    sd = r["solve_directive"]
    assert sd["model_name"] == "m"
    assert sd["model_type"] == "NLP"
    assert sd["direction"] == "minimizing"
    assert sd["objective_var"] == "objvar"
    assert r["class"] == "nonlinear program (continuous)"
    assert r["supported_by_pounce"] is True
    # adaptive mu suggestion is unconditional for NLP
    options = {s["option"] for s in r["suggestions"]}
    assert "mu_strategy" in options


def test_analyze_handwritten_gms_emits_warning_when_no_convert_header():
    r = analyze_gams_problem(str(HANDWRITTEN_GMS))
    assert r["dimensions"] == {}
    assert r["solve_directive"]["model_type"] == "NLP"
    assert r["solve_directive"]["direction"] == "maximizing"
    assert any("no `gams convert` header" in w for w in r["warnings"])


def test_analyze_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        analyze_gams_problem("/nonexistent/path/foo.gms")


# ---- helper-level: solve-directive regex ----------------------------


def test_solve_directive_minimizing():
    sd = _parse_gms_solve_directive("Solve m using NLP minimizing obj;")
    assert sd == {
        "model_name": "m", "model_type": "NLP",
        "direction": "minimizing", "objective_var": "obj",
    }


def test_solve_directive_maximizing_case_insensitive():
    sd = _parse_gms_solve_directive("solve foo USING dnlp MAXIMIZING z;")
    assert sd["model_type"] == "DNLP"
    assert sd["direction"] == "maximizing"
    assert sd["objective_var"] == "z"


def test_solve_directive_no_direction():
    sd = _parse_gms_solve_directive("Solve m using CNS;")
    assert sd["model_type"] == "CNS"
    assert sd["direction"] is None
    assert sd["objective_var"] is None


def test_solve_directive_not_present():
    assert _parse_gms_solve_directive("just some equations") == {}


# ---- helper-level: convert-header parse -----------------------------


def test_convert_header_returns_empty_on_blank():
    assert _parse_gms_convert_header("nothing here\n") == {}


def test_convert_header_partial_minlp_marks_discrete_vars():
    text = """\
*  Variable counts
*                 x       b       i
*     Total    cont  binary integer
*        10       6       2       2
"""
    dims = _parse_gms_convert_header(text)
    assert dims["n_variables_total"] == 10
    assert dims["n_continuous_vars"] == 6
    assert dims["n_binary_vars"] == 2
    assert dims["n_integer_vars"] == 2


# ---- parse_gams_listing ---------------------------------------------


def test_parse_listing_extracts_solve_summary():
    r = parse_gams_listing(str(EX8_LST))
    s = r["summary"]
    assert s["solver"] == "POUNCE"
    assert s["solver_status_code"] == 2
    assert s["solver_status"] == "Iteration Interrupt"
    assert s["model_status_code"] == 7
    assert s["model_status"] == "Feasible Solution"
    assert s["iteration_count"] == 2999
    assert isinstance(s["objective_value"], float)
    assert abs(s["objective_value"] - (-0.8594)) < 1e-3


def test_parse_listing_captures_embedded_solver_status_block():
    r = parse_gams_listing(str(EX8_LST))
    sf = r["summary"].get("solver_status_file", "")
    assert "POUNCE" in sf
    assert "Variables: 141" in sf  # from the embedded =C block
    assert "Number of Iterations" in sf


def test_parse_listing_spawn_failed_lst():
    r = parse_gams_listing(str(SPAWN_FAILED_LST))
    s = r["summary"]
    assert s["solver_status_code"] == 13
    assert s["model_status_code"] == 13
    assert s["objective_value"] == "NA"  # parsed as string when not numeric
    assert s.get("iteration_count") == "NA"
    # no embedded =C block in this fixture
    assert "solver_status_file" not in s


def test_parse_listing_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_gams_listing("/nonexistent/path/foo.lst")


# ---- helper-level: lst parser tolerance -----------------------------


def test_lst_parser_tolerates_partial_block():
    snippet = "**** SOLVER STATUS     1 Normal Completion\nOBJECTIVE VALUE 1.0\n"
    s = _parse_lst_solve_summary(snippet)
    assert s["solver_status_code"] == 1
    assert s["solver_status"] == "Normal Completion"
    # OBJECTIVE VALUE line lacks the ****, so it should NOT be parsed
    assert "objective_value" not in s


# ---- list_gams_examples ---------------------------------------------


def test_list_examples_summary_includes_known_suites():
    r = list_gams_examples()
    suites = {s["name"]: s["count"] for s in r["suites"]}
    # The four nlpbench suites are present (counts may be 0 if not synced)
    for name in ("globallib.gms", "mittelmann.gms", "princetonlib.gms",
                 "powerflow.gms", "examples", "smoke"):
        assert name in suites
    assert r["total"] == sum(s["count"] for s in r["suites"])


def test_list_examples_globallib_returns_paginated_files():
    # globallib.gms is committed into the repo; pagination should work.
    r = list_gams_examples(suite="globallib.gms", limit=5, offset=0)
    if r["count"] == 0:
        pytest.skip("globallib.gms suite not synced in this checkout")
    assert len(r["files"]) <= 5
    assert all(f.endswith(".gms") for f in r["files"])


def test_list_examples_unknown_suite_raises():
    with pytest.raises(ValueError, match="unknown suite"):
        list_gams_examples(suite="does-not-exist")


def test_list_examples_smoke_suite_returns_test_hs071():
    r = list_gams_examples(suite="smoke")
    assert r["count"] in (0, 1)
    if r["count"] == 1:
        assert r["files"][0].endswith("test_hs071.gms")


# ---- run_gams_problem (integration, gated on `gams` availability) ---


@pytest.mark.skipif(
    shutil.which("gams") is None
    and not Path("/Library/Frameworks/GAMS.framework/Versions/Current/Resources/gams").exists(),
    reason="`gams` not on PATH and not at the macOS framework default",
)
def test_run_gams_problem_solves_hs071(tmp_path: Path):
    """End-to-end: requires the instrumented GAMS link installed."""
    smoke = Path(__file__).parent.parent.parent.parent / "gams" / "test_hs071.gms"
    if not smoke.exists():
        pytest.skip(f"smoke .gms not found at {smoke}")

    r = run_gams_problem(
        str(smoke),
        options={"tol": "1e-8", "max_iter": "100", "mu_strategy": "adaptive"},
        workdir=str(tmp_path),
        timeout_seconds=60.0,
        analyze=False,
    )
    assert r["exit_code"] == 0
    assert r["lst_file"] is not None
    assert r["report_path"] is not None
    # hs071 should solve cleanly under POUNCE
    assert r["lst_summary"]["solver_status_code"] == 1
    assert r["lst_summary"]["model_status_code"] in (1, 2)  # Optimal / Locally Optimal
    rs = r["report_summary"]
    assert rs["status"] == "SolveSucceeded"
    assert rs["n_variables"] == 4
    assert rs["n_constraints"] == 2
    assert rs["iterations_captured"] >= 1
