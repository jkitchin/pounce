"""Loaders and analysis helpers for pounce solve reports.

Parses `pounce.solve-report/v1` JSON documents and computes derived series
(log-residual progress, stall detection, restoration windows) the MCP tools
expose to LLMs.

The functions here are intentionally pure — no I/O beyond the initial file
read, no caching — so they can also be reused by the desktop UI core later.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "pounce.solve-report/v1"


class ReportError(ValueError):
    """Raised when a JSON file is not a recognized pounce solve report."""


def load_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a solve report.

    Raises ReportError if the file does not declare the expected schema.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise ReportError(f"no such file: {p}")
    try:
        report = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ReportError(f"{p}: invalid JSON ({e})") from e
    schema = report.get("schema")
    if schema != SCHEMA:
        raise ReportError(
            f"{p}: unexpected schema {schema!r} (expected {SCHEMA!r})"
        )
    return report


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    """One-screen overview suitable for an LLM to read first.

    Pulls the headline numbers (status, objective, iter count, KKT
    residuals, restoration counters) without the full trajectory.
    """
    fair = report.get("fair_metadata", {})
    problem = report.get("problem", {})
    solution = report.get("solution", {})
    stats = report.get("statistics", {})
    iters = report.get("iterations", [])
    return {
        "schema": report.get("schema"),
        "result_id": fair.get("result_id"),
        "solver": fair.get("solver", {}).get("name"),
        "solver_version": fair.get("solver", {}).get("version"),
        "input": fair.get("input"),
        "elapsed_seconds": fair.get("elapsed_seconds"),
        "problem": {
            "n_variables": problem.get("n_variables"),
            "n_constraints": problem.get("n_constraints"),
            "minimize": problem.get("minimize"),
            "nnz_jac_g": problem.get("nnz_jac_g"),
            "nnz_h_lag": problem.get("nnz_h_lag"),
        },
        "status": solution.get("status"),
        "solve_result_num": solution.get("solve_result_num"),
        "final_objective": stats.get("final_objective"),
        "iteration_count": stats.get("iteration_count"),
        "final_kkt_error": stats.get("final_kkt_error"),
        "final_dual_inf": stats.get("final_dual_inf"),
        "final_constr_viol": stats.get("final_constr_viol"),
        "final_compl": stats.get("final_compl"),
        "restoration": {
            "calls": stats.get("restoration_calls", 0),
            "inner_iters": stats.get("restoration_inner_iters", 0),
            "outer_iters": stats.get("restoration_outer_iters", 0),
            "wall_secs": stats.get("restoration_wall_secs", 0.0),
        },
        "iterations_captured": len(iters),
        "evals": {
            "obj": stats.get("num_obj_evals"),
            "constr": stats.get("num_constr_evals"),
            "obj_grad": stats.get("num_obj_grad_evals"),
            "constr_jac": stats.get("num_constr_jac_evals"),
            "hess": stats.get("num_hess_evals"),
        },
    }


_TRACE_COLUMNS = (
    "iter",
    "objective",
    "inf_pr",
    "inf_du",
    "mu",
    "d_norm",
    "regularization",
    "alpha_dual",
    "alpha_primal",
    "alpha_primal_char",
    "ls_trials",
)


def convergence_trace(
    report: dict[str, Any],
    columns: Iterable[str] | None = None,
) -> dict[str, list[Any]]:
    """Return per-iteration trajectory as parallel columns.

    Column-oriented output is much more compact than a list of dicts when
    serialized for an LLM — N iterations × M columns rather than N copies
    of the keys.
    """
    iters = report.get("iterations", [])
    cols = tuple(columns) if columns is not None else _TRACE_COLUMNS
    unknown = [c for c in cols if c not in _TRACE_COLUMNS]
    if unknown:
        raise ReportError(
            f"unknown trace column(s): {unknown}. valid: {_TRACE_COLUMNS}"
        )
    out: dict[str, list[Any]] = {c: [] for c in cols}
    for row in iters:
        for c in cols:
            out[c].append(row.get(c))
    return out


def get_iterate(report: dict[str, Any], k: int) -> dict[str, Any]:
    """Full IterRecord for iteration k (0-indexed)."""
    iters = report.get("iterations", [])
    if not iters:
        raise ReportError(
            "report has no iteration history (rerun with --json-detail full)"
        )
    if k < 0 or k >= len(iters):
        raise ReportError(
            f"iter {k} out of range; report has {len(iters)} iterations (0..{len(iters) - 1})"
        )
    row = dict(iters[k])
    # Augment with derived fields the LLM finds useful.
    row["log10_inf_pr"] = _safe_log10(row.get("inf_pr"))
    row["log10_inf_du"] = _safe_log10(row.get("inf_du"))
    row["log10_mu"] = _safe_log10(row.get("mu"))
    return row


@dataclass
class Stall:
    """One stalled-progress window."""

    start_iter: int
    end_iter: int
    metric: str
    delta_log10: float


def find_stalls(
    report: dict[str, Any],
    min_window: int = 5,
    max_log10_progress: float = 0.3,
) -> list[dict[str, Any]]:
    """Detect windows where log10(inf_pr) or log10(inf_du) plateau.

    A "stall" is `min_window` or more consecutive iterations whose
    log10-residual moved by less than `max_log10_progress` total — i.e.
    the residual barely budged for several iters in a row. This is the
    canonical "Ipopt is stuck" symptom.
    """
    iters = report.get("iterations", [])
    out: list[Stall] = []
    for metric in ("inf_pr", "inf_du"):
        series = [_safe_log10(row.get(metric)) for row in iters]
        out.extend(_scan_stalls(series, metric, min_window, max_log10_progress))
    return [
        {
            "start_iter": s.start_iter,
            "end_iter": s.end_iter,
            "metric": s.metric,
            "delta_log10": round(s.delta_log10, 4),
        }
        for s in out
    ]


def restoration_windows(report: dict[str, Any]) -> list[dict[str, int]]:
    """Identify contiguous runs of iters tagged with the 'r' alpha-primal char.

    Restoration iterations show up in the per-iter output with a
    distinguishing single-character tag (`r` or `R`); contiguous runs of
    them correspond to one restoration entry → exit cycle.
    """
    iters = report.get("iterations", [])
    windows: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    for row in iters:
        char = (row.get("alpha_primal_char") or "").lower()
        if char == "r":
            if current is None:
                current = {"start_iter": row["iter"], "end_iter": row["iter"]}
            else:
                current["end_iter"] = row["iter"]
        else:
            if current is not None:
                windows.append(current)
                current = None
    if current is not None:
        windows.append(current)
    return windows


def diagnose(report: dict[str, Any]) -> dict[str, Any]:
    """Run common Ipopt-failure heuristics and return findings.

    Each finding has a `severity` (info | warning | error) and a human
    `message` the LLM can quote. The intent is to surface the well-known
    failure modes — mu stuck, line-search collapse, regularization
    growth, max-iter exceeded with no restoration progress — that
    experienced Ipopt users diagnose by eye.
    """
    findings: list[dict[str, Any]] = []
    stats = report.get("statistics", {})
    solution = report.get("solution", {})
    iters = report.get("iterations", [])
    status = solution.get("status", "")

    if status == "SolveSucceeded":
        findings.append({
            "severity": "info",
            "code": "converged",
            "message": (
                f"Solver converged in {stats.get('iteration_count')} iterations to "
                f"objective {stats.get('final_objective'):.6g}; KKT error "
                f"{stats.get('final_kkt_error'):.2e}."
            ),
        })
    elif status == "MaximumIterationsExceeded":
        findings.append({
            "severity": "error",
            "code": "max_iter_exceeded",
            "message": (
                "Hit max_iter without converging. KKT error at termination: "
                f"{stats.get('final_kkt_error'):.2e}. Consider raising max_iter, "
                "tightening initial guess, or relaxing tol."
            ),
        })

    if stats.get("restoration_calls", 0) > 0:
        findings.append({
            "severity": "warning",
            "code": "restoration_used",
            "message": (
                f"Restoration phase entered {stats['restoration_calls']} time(s); "
                f"{stats.get('restoration_outer_iters', 0)} outer iters spent in "
                f"restoration ({stats.get('restoration_wall_secs', 0.0):.3f}s). "
                "Indicates the line search couldn't make progress on the original problem."
            ),
        })

    # Mu-stuck detection: barrier parameter didn't decrease over a long window.
    if len(iters) >= 10:
        mu_series = [row.get("mu", 0.0) for row in iters]
        mu_first = max(mu_series[:3])
        mu_last = min(mu_series[-3:])
        if mu_first > 0 and mu_last > 0:
            log_drop = math.log10(mu_first) - math.log10(mu_last)
            if log_drop < 1.0:
                findings.append({
                    "severity": "warning",
                    "code": "mu_stuck",
                    "message": (
                        f"Barrier parameter μ dropped only {log_drop:.2f} orders of "
                        f"magnitude across {len(iters)} iterations "
                        f"(from {mu_first:.2e} to {mu_last:.2e}). "
                        "Try mu_strategy=adaptive or a smaller mu_init."
                    ),
                })

    # Line-search collapse: many backtracking trials in a row.
    heavy_ls = [row for row in iters if (row.get("ls_trials") or 0) >= 10]
    if heavy_ls:
        findings.append({
            "severity": "warning",
            "code": "heavy_line_search",
            "message": (
                f"{len(heavy_ls)} iteration(s) needed >=10 backtracking trials "
                f"(worst: iter {max(heavy_ls, key=lambda r: r['ls_trials'])['iter']} "
                f"with {max(r['ls_trials'] for r in heavy_ls)} trials). Search "
                "direction quality may be poor — check Hessian accuracy."
            ),
        })

    # Regularization growth.
    regs = [row.get("regularization") or 0.0 for row in iters]
    big_reg = [r for r in regs if r > 1e-4]
    if big_reg:
        findings.append({
            "severity": "info",
            "code": "hessian_regularized",
            "message": (
                f"Hessian regularization applied on {len(big_reg)} iteration(s) "
                f"(max δ_w = {max(big_reg):.2e}). The KKT system was indefinite; "
                "this is normal near saddle points but persistent regularization "
                "suggests a problematic Hessian."
            ),
        })

    # Restoration window count from per-iter tag.
    rwins = restoration_windows(report)
    if len(rwins) > 1:
        findings.append({
            "severity": "warning",
            "code": "restoration_loop",
            "message": (
                f"Restoration was entered {len(rwins)} separate times "
                f"(windows: {rwins[:3]}...). Repeated re-entry often means the "
                "problem is infeasible at the working point. Verify constraints."
            ),
        })

    # Only surface stalls in `diagnose` when they're concerning — i.e. the
    # solve did not cleanly succeed, OR a stall lasted long enough that
    # even a converging run shouldn't have hit it. Plain `find_stalls` is
    # still available to callers who want every window.
    stalls = find_stalls(report)
    if stalls:
        long_stall = max((s["end_iter"] - s["start_iter"] + 1) for s in stalls)
        if status != "SolveSucceeded" or long_stall >= 8:
            findings.append({
                "severity": "warning",
                "code": "convergence_stall",
                "message": (
                    f"Detected {len(stalls)} stall window(s) where log-residual "
                    f"barely moved (longest: {long_stall} iters; first: {stalls[0]}). "
                    "Either the problem is ill-conditioned, scaling is off, or "
                    "termination tolerance is too tight."
                ),
            })

    return {"findings": findings, "n_findings": len(findings)}


def compare(reports: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Side-by-side comparison of multiple solve reports.

    `reports` is a list of (label, parsed_report) pairs so callers can
    carry their own naming convention.
    """
    rows = []
    for label, r in reports:
        s = summarize(r)
        rows.append({
            "label": label,
            "status": s["status"],
            "iter_count": s["iteration_count"],
            "final_objective": s["final_objective"],
            "final_kkt_error": s["final_kkt_error"],
            "restoration_calls": s["restoration"]["calls"],
            "elapsed_seconds": s["elapsed_seconds"],
        })
    return {"rows": rows, "n_runs": len(rows)}


def _safe_log10(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v <= 0 or math.isnan(v) or math.isinf(v):
        return None
    return math.log10(v)


def _scan_stalls(
    series: list[float | None],
    metric: str,
    min_window: int,
    max_log10_progress: float,
) -> list[Stall]:
    out: list[Stall] = []
    i = 0
    n = len(series)
    while i < n:
        if series[i] is None:
            i += 1
            continue
        # Greedy: extend [i..j] while it stays a stall (delta <= threshold).
        j = i
        while j + 1 < n and series[j + 1] is not None:
            window = series[i : j + 2]
            if max(window) - min(window) > max_log10_progress:  # type: ignore[type-var]
                break
            j += 1
        if (j - i + 1) >= min_window:
            window = series[i : j + 1]
            out.append(Stall(i, j, metric, max(window) - min(window)))  # type: ignore[arg-type]
            i = j + 1
        else:
            i += 1
    return out
