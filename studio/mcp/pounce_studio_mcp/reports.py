"""Thin Python wrapper over the Rust `_native` analysis core.

The actual loading, parsing, and analysis logic lives in
`crates/pounce-studio-core/`; this module marshals across the FFI
(`crates/pounce-studio-pyo3/`) and adapts errors to the `ReportError`
exception the MCP server expects.

Why JSON-roundtrip on the FFI: keeps the Rust side serde-only, no
`pythonize` dep, and the data is small (a Full-detail solve report is a
few hundred KB at worst). The cost is one `serde_json::to_string` +
`json.loads` per call, which is negligible. Parameter-less results
(summarize, convergence_trace, restoration_windows, diagnose,
render_markdown) are memoised on the Rust side per `Report` handle, so
repeat MCP-tool calls within a session don't re-serialise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pounce_studio_mcp import _native


SCHEMA = _native.SOLVE_REPORT_SCHEMA


class ReportError(ValueError):
    """Raised when a file is not a recognised pounce solve report."""


def load_report(path: str | Path) -> _native.Report:
    """Load a `pounce.solve-report/v1` JSON file into a Rust-backed handle.

    The returned `Report` object owns the parsed document; analysis
    methods reuse it without re-parsing. Wrapped in `ReportError` for
    consistency with the previous API.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise ReportError(f"no such file: {p}")
    try:
        return _native.Report.from_path(str(p))
    except (ValueError, OSError) as e:
        raise ReportError(str(e)) from e


def load_report_bytes(data: bytes) -> _native.Report:
    """Parse a solve report directly from in-memory bytes."""
    try:
        return _native.Report.from_bytes(data)
    except ValueError as e:
        raise ReportError(str(e)) from e


def summarize(report: _native.Report) -> dict[str, Any]:
    return json.loads(report.summarize())


def convergence_trace(report: _native.Report) -> dict[str, list]:
    return json.loads(report.convergence_trace())


def get_iterate(report: _native.Report, k: int) -> dict[str, Any]:
    try:
        return json.loads(report.get_iterate(k))
    except ValueError as e:
        raise ReportError(str(e)) from e


def find_stalls(
    report: _native.Report,
    min_window: int | None = None,
    max_log10_progress: float | None = None,
) -> list[dict[str, Any]]:
    return json.loads(report.find_stalls(min_window, max_log10_progress))


def restoration_windows(report: _native.Report) -> list[dict[str, int]]:
    return json.loads(report.restoration_windows())


def diagnose(report: _native.Report) -> dict[str, Any]:
    findings = json.loads(report.diagnose())
    return {"findings": findings, "n_findings": len(findings)}


def compare(reports: list[tuple[str, _native.Report]]) -> dict[str, Any]:
    rows = json.loads(_native.compare_reports(reports))
    return {"rows": rows, "n_runs": len(rows)}


def render_markdown(report: _native.Report) -> str:
    """Render the same Markdown the `pounce-studio inspect` CLI emits."""
    return report.render_markdown()
