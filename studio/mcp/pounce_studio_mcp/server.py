"""MCP server exposing pounce solve reports as Claude-callable tools.

Spike scope: post-mortem analysis of `pounce.solve-report/v1` JSON files.
No state held between calls; every tool takes a file path. Live streaming
is out of scope here (Phase 3).
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from . import reports as R


mcp = FastMCP("pounce-studio")


@mcp.tool()
def load_solve_report(path: str) -> dict[str, Any]:
    """Load a pounce.solve-report/v1 JSON file and return a headline summary.

    Call this first to confirm the file is a valid pounce solve report and
    to get the high-level outcome (status, objective, iter count, KKT
    residuals, restoration counters, evaluation counts) before drilling in
    with the other tools.

    Args:
        path: Absolute or ~-expanded path to a JSON solve report.

    Returns:
        Dict with schema, problem dims, status, final objective and KKT
        residuals, restoration summary, eval counts, and how many
        iterations are captured (0 if the report was written at
        --json-detail summary).
    """
    return R.summarize(R.load_report(path))


@mcp.tool()
def convergence_trace(path: str, columns: list[str] | None = None) -> dict[str, Any]:
    """Return per-iteration trajectory as column-oriented arrays.

    Columns: iter, objective, inf_pr, inf_du, mu, d_norm, regularization,
    alpha_dual, alpha_primal, alpha_primal_char, ls_trials. Pass a subset
    in `columns` to keep responses small. The report must have been
    written at --json-detail full or this returns empty arrays.

    Args:
        path: Path to the solve report.
        columns: Subset of column names to return; None means all.
    """
    full = R.convergence_trace(R.load_report(path))
    if columns is None:
        return full
    unknown = [c for c in columns if c not in full]
    if unknown:
        return {
            "error": f"unknown trace column(s): {unknown}. valid: {list(full)}",
        }
    return {c: full[c] for c in columns}


@mcp.tool()
def get_iterate(path: str, k: int) -> dict[str, Any]:
    """Full per-iteration record for iter k (0-indexed).

    Includes the raw IterRecord fields plus derived log10 values for
    inf_pr, inf_du, mu. Use this when zooming in on a specific iteration
    flagged by find_stalls or diagnose.

    Args:
        path: Path to the solve report.
        k: Iteration index, 0-based.
    """
    return R.get_iterate(R.load_report(path), k)


@mcp.tool()
def find_stalls(
    path: str,
    min_window: int = 5,
    max_log10_progress: float = 0.3,
) -> dict[str, Any]:
    """Detect convergence-stall windows.

    A stall is `min_window` or more consecutive iterations whose
    log10(inf_pr) or log10(inf_du) moved by less than
    `max_log10_progress` total — the canonical "stuck" symptom.

    Args:
        path: Path to the solve report.
        min_window: Minimum consecutive iterations to count as a stall.
        max_log10_progress: Maximum log10-units of residual movement
            allowed within the window for it to count as a stall.

    Returns:
        Dict with `windows`: a list of {start_iter, end_iter, metric,
        delta_log10} entries.
    """
    report = R.load_report(path)
    stalls = R.find_stalls(report, min_window, max_log10_progress)
    return {"windows": stalls, "count": len(stalls)}


@mcp.tool()
def diagnose(path: str) -> dict[str, Any]:
    """Run common Ipopt-failure heuristics over a solve report.

    Detects: convergence success, max-iter exceeded, restoration entry,
    restoration loops, mu-stuck, line-search collapse, Hessian
    regularization, and convergence stalls. Each finding has a severity
    (info | warning | error) and a human message.

    Args:
        path: Path to the solve report.

    Returns:
        Dict with `findings`: list of {severity, code, message} and
        `n_findings`: total count.
    """
    return R.diagnose(R.load_report(path))


@mcp.tool()
def restoration_windows(path: str) -> dict[str, Any]:
    """Identify contiguous runs of iterations spent in restoration.

    Restoration iters are tagged with 'r' in the per-iter alpha-primal
    character. Returns each entry → exit window.

    Args:
        path: Path to the solve report.
    """
    report = R.load_report(path)
    windows = R.restoration_windows(report)
    return {"windows": windows, "count": len(windows)}


@mcp.tool()
def compare_runs(paths: list[str], labels: list[str] | None = None) -> dict[str, Any]:
    """Compare multiple solve reports side-by-side.

    Returns one row per report with status, iter count, final objective,
    final KKT error, restoration calls, and elapsed seconds — useful for
    A/B comparing option settings or solver builds.

    Args:
        paths: Paths to JSON solve reports.
        labels: Optional labels for each report (defaults to the path).
    """
    if labels is not None and len(labels) != len(paths):
        return {
            "error": (
                f"labels length ({len(labels)}) does not match paths length "
                f"({len(paths)})"
            )
        }
    use_labels = labels if labels is not None else paths
    return R.compare([
        (label, R.load_report(p)) for label, p in zip(use_labels, paths)
    ])


def main() -> None:
    """Entry point used by `python -m pounce_studio_mcp`."""
    mcp.run()


if __name__ == "__main__":
    main()
