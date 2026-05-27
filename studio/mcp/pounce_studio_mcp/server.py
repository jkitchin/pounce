"""MCP server exposing pounce solve reports as Claude-callable tools.

Spike scope: post-mortem analysis of `pounce.solve-report/v1` JSON files,
plus a `run_problem` batch tool that shells out to the `pounce` CLI to
produce a fresh report. No state held between calls; every tool takes a
file path. Live streaming is still out of scope here (Phase 3).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import glossary as G
from . import reports as R


mcp = FastMCP("pounce-studio")


# ---- run_problem / analyze_problem helpers --------------------------

# Built-in problems the CLI ships with (output of `pounce --list-problems`).
# Hardcoded so analyze_problem can answer without shelling out.
_BUILTINS: dict[str, dict[str, Any]] = {
    "quadratic": {
        "n_variables": 2, "n_constraints": 0,
        "class": "unconstrained quadratic",
        "notes": "Convex QP; trivial — single Newton step from any start.",
    },
    "rosenbrock": {
        "n_variables": 2, "n_constraints": 0,
        "class": "unconstrained nonlinear",
        "notes": "Classic non-convex banana valley; tests line search.",
    },
    "bounded-quadratic": {
        "n_variables": 2, "n_constraints": 0,
        "class": "bound-constrained quadratic",
        "notes": "Active-set quadratic; exercises bound multipliers.",
    },
    "eq-quadratic": {
        "n_variables": 3, "n_constraints": 1,
        "class": "equality-constrained quadratic",
        "notes": "QP with one linear equality; tests KKT factorisation.",
    },
    "circle": {
        "n_variables": 2, "n_constraints": 1,
        "class": "equality-constrained nonlinear",
        "notes": "Nonlinear equality; tests restoration entry.",
    },
}


def _find_pounce_bin() -> str:
    """Locate the pounce CLI binary.

    Search order: POUNCE_BIN env var → `target/release/pounce` walking
    up from this file → `pounce` on $PATH.
    """
    env = os.environ.get("POUNCE_BIN")
    if env:
        if not Path(env).exists():
            raise FileNotFoundError(f"POUNCE_BIN={env} does not exist")
        return env
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "target" / "release" / "pounce"
        if candidate.exists():
            return str(candidate)
        if parent == parent.parent:
            break
    which = shutil.which("pounce")
    if which:
        return which
    raise FileNotFoundError(
        "could not locate the pounce binary. Set POUNCE_BIN, build the "
        "repo with `make build`, or put `pounce` on $PATH."
    )


def _parse_nl_header(path: Path) -> dict[str, Any]:
    """Parse the textual `.nl` header (first ~10 lines).

    Returns a dict of dimensions plus warnings on anything we couldn't
    parse. Tolerant: partial parses still return what we got.
    """
    out: dict[str, Any] = {"format": "unknown", "warnings": []}
    try:
        with path.open("r", errors="replace") as fh:
            lines = [fh.readline().rstrip() for _ in range(10)]
    except OSError as e:
        return {"error": f"could not read .nl file: {e}"}

    if not lines or not lines[0]:
        return {"error": "empty .nl file"}
    out["format"] = "text" if lines[0].startswith("g") else (
        "binary" if lines[0].startswith("b") else "unknown"
    )
    if out["format"] == "binary":
        out["warnings"].append("binary .nl: header parse skipped")
        return out

    def ints(line: str) -> list[int]:
        return [int(t) for t in line.split() if t.lstrip("-").isdigit()]

    # Line 2: n_var n_con n_obj n_range n_eqn [n_lcon]
    try:
        l2 = ints(lines[1])
        if len(l2) >= 5:
            out["n_variables"] = l2[0]
            out["n_constraints"] = l2[1]
            out["n_objectives"] = l2[2]
            out["n_ranges"] = l2[3]
            out["n_equalities"] = l2[4]
    except (ValueError, IndexError):
        out["warnings"].append("could not parse dimensions line")

    # Line 3: nlc nlo  (nonlinear constraints, nonlinear objectives)
    try:
        l3 = ints(lines[2])
        if len(l3) >= 2:
            out["n_nonlinear_constraints"] = l3[0]
            out["n_nonlinear_objectives"] = l3[1]
    except (ValueError, IndexError):
        pass

    # Line 5: nlvc nlvo nlvb  (nonlinear var counts)
    try:
        l5 = ints(lines[4])
        if len(l5) >= 3:
            out["n_nonlinear_vars_in_cons"] = l5[0]
            out["n_nonlinear_vars_in_obj"] = l5[1]
            out["n_nonlinear_vars_in_both"] = l5[2]
    except (ValueError, IndexError):
        pass

    # Line 7 (some emitters) / line 8: nnz_jac nnz_grad
    for idx in (6, 7):
        try:
            li = ints(lines[idx])
            if len(li) == 2 and "nnz_jacobian" not in out:
                out["nnz_jacobian"] = li[0]
                out["nnz_objective_gradient"] = li[1]
                break
        except (ValueError, IndexError):
            pass

    return out


def _classify(dims: dict[str, Any]) -> str:
    n_con = dims.get("n_constraints", 0)
    nlc = dims.get("n_nonlinear_constraints", 0)
    nlo = dims.get("n_nonlinear_objectives", 0)
    n_eq = dims.get("n_equalities", 0)
    is_nl = (nlc > 0) or (nlo > 0)
    if n_con == 0:
        return "unconstrained " + ("nonlinear" if is_nl else "linear/quadratic")
    parts = []
    parts.append("nonlinear" if is_nl else "linear/quadratic")
    parts.append("equality-constrained" if n_eq == n_con else "general-constrained")
    return " ".join(parts)


def _suggest_options(dims: dict[str, Any]) -> list[dict[str, str]]:
    """Heuristic option suggestions. Each entry has `option`, `value`, `why`."""
    suggestions: list[dict[str, str]] = []
    n_var = dims.get("n_variables", 0)
    n_con = dims.get("n_constraints", 0)
    nlc = dims.get("n_nonlinear_constraints", 0)
    nlo = dims.get("n_nonlinear_objectives", 0)
    size = n_var + n_con

    if size > 5_000:
        suggestions.append({
            "option": "linear_solver",
            "value": "ma57",
            "why": (
                f"problem is medium/large ({n_var} vars, {n_con} cons); MA57 "
                "is much faster than FERAL at this scale. Requires the build "
                "to have been compiled with `--features ma57`."
            ),
        })
    if size > 1_000 and nlc == 0 and nlo == 0:
        suggestions.append({
            "option": "mu_strategy",
            "value": "adaptive",
            "why": "purely linear/quadratic — adaptive mu usually converges in fewer iters.",
        })
    if size > 10_000:
        suggestions.append({
            "option": "max_iter",
            "value": "1000",
            "why": "default 3000 is fine but raise tol expectations for large problems.",
        })
    if nlc > 0 and dims.get("n_equalities", 0) == n_con and n_con > 0:
        suggestions.append({
            "option": "bound_relax_factor",
            "value": "0",
            "why": (
                "all constraints equality + nonlinear: relaxing bounds can "
                "blur the feasible manifold; setting to 0 keeps it sharp."
            ),
        })
    return suggestions


def _heuristic_warnings(dims: dict[str, Any]) -> list[str]:
    warnings: list[str] = list(dims.get("warnings", []))
    n_var = dims.get("n_variables", 0)
    n_con = dims.get("n_constraints", 0)
    if n_var == 0:
        warnings.append("zero variables parsed — header read may have failed")
    if n_var + n_con > 50_000:
        warnings.append(
            f"very large problem ({n_var} vars, {n_con} cons); expect "
            "long solve times and consider running with `--dump` for diagnostics."
        )
    if dims.get("n_objectives", 1) == 0:
        warnings.append("no objective: this is a feasibility problem, not optimisation.")
    return warnings


@mcp.tool()
def analyze_problem(
    builtin: str | None = None,
    nl_file: str | None = None,
) -> dict[str, Any]:
    """Inspect a problem without solving it.

    Returns dimensions, problem class, heuristic warnings, and a list of
    suggested solver options the agent can choose to pass into
    `run_problem`. The suggestions are advisory — they are NOT applied
    automatically.

    Exactly one of `builtin` or `nl_file` must be specified.

    Args:
        builtin: Name of a built-in test problem (rosenbrock, quadratic,
            bounded-quadratic, eq-quadratic, circle).
        nl_file: Path to an AMPL .nl file. Only the textual header is
            inspected — no expression-tree walk.

    Returns:
        Dict with `kind` (builtin|nl_file), `dimensions`, `class`,
        `warnings`, `suggestions`, and (for builtins) `notes`.
    """
    if (builtin is None) == (nl_file is None):
        raise ValueError("specify exactly one of `builtin` or `nl_file`")

    if builtin is not None:
        meta = _BUILTINS.get(builtin)
        if meta is None:
            raise ValueError(
                f"unknown builtin {builtin!r}; valid: {sorted(_BUILTINS)}"
            )
        dims = {
            "n_variables": meta["n_variables"],
            "n_constraints": meta["n_constraints"],
        }
        return {
            "kind": "builtin",
            "name": builtin,
            "dimensions": dims,
            "class": meta["class"],
            "notes": meta["notes"],
            "warnings": _heuristic_warnings(dims),
            "suggestions": _suggest_options(dims),
        }

    nl = Path(nl_file).expanduser()
    if not nl.exists():
        raise FileNotFoundError(f"no such .nl file: {nl}")
    dims = _parse_nl_header(nl)
    if "error" in dims:
        return {"kind": "nl_file", "path": str(nl), "dimensions": dims}
    return {
        "kind": "nl_file",
        "path": str(nl),
        "dimensions": dims,
        "class": _classify(dims),
        "warnings": _heuristic_warnings(dims),
        "suggestions": _suggest_options(dims),
    }


@mcp.tool()
def run_problem(
    builtin: str | None = None,
    nl_file: str | None = None,
    json_output: str | None = None,
    json_detail: str = "full",
    options: dict[str, str] | None = None,
    options_file: str | None = None,
    extra_args: list[str] | None = None,
    timeout_seconds: float = 120.0,
    analyze: bool = True,
) -> dict[str, Any]:
    """Run a pounce solve and return the resulting report.

    Synchronously invokes the `pounce` CLI (located via POUNCE_BIN, the
    repo's `target/release/pounce`, or $PATH). The JSON report is parsed
    on return so the agent can immediately reason about the outcome.

    Exactly one of `builtin` or `nl_file` must be specified.

    Args:
        builtin: Name of a built-in test problem. Mutually exclusive
            with `nl_file`. See `analyze_problem` for the list.
        nl_file: Path to an AMPL .nl file. Mutually exclusive with `builtin`.
        json_output: Where to write the JSON solve report. If None, a
            temp file is created and its path returned.
        json_detail: "summary" or "full" (default "full"; the per-iter
            MCP tools need "full").
        options: OptionsList key=value pairs forwarded to the solver,
            e.g. {"max_iter": "500", "tol": "1e-10"}.
        options_file: Optional ipopt.opt-format file path.
        extra_args: Escape hatch for raw CLI flags not covered above.
        timeout_seconds: Kill the solve after this many seconds.
        analyze: When True (default), call `analyze_problem` first and
            embed the result under `analysis`. Suggestions are NOT
            auto-applied — the agent decides whether to re-run with them.

    Returns:
        Dict with `report_path`, `exit_code`, `elapsed_seconds`, `argv`,
        `stdout_tail`, `stderr_tail`, and (when the report file was
        written) `summary`. Includes `analysis` when analyze=True.
    """
    if (builtin is None) == (nl_file is None):
        raise ValueError("specify exactly one of `builtin` or `nl_file`")
    if json_detail not in ("summary", "full"):
        raise ValueError(
            f"json_detail must be 'summary' or 'full', got {json_detail!r}"
        )

    pre_analysis: dict[str, Any] | None = None
    if analyze:
        try:
            pre_analysis = analyze_problem(builtin=builtin, nl_file=nl_file)
        except (ValueError, FileNotFoundError) as e:
            pre_analysis = {"error": str(e)}

    binary = _find_pounce_bin()

    if json_output:
        out_path = Path(json_output).expanduser()
    else:
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix="pounce-run-")
        os.close(fd)
        out_path = Path(tmp)

    argv: list[str] = [binary]
    if nl_file:
        nl = Path(nl_file).expanduser()
        if not nl.exists():
            raise FileNotFoundError(f"no such .nl file: {nl}")
        argv.append(str(nl))
    else:
        argv.extend(["--problem", builtin])  # type: ignore[list-item]
    argv.extend(["--json-output", str(out_path), "--json-detail", json_detail])
    if options_file:
        argv.extend(["--options-file", str(Path(options_file).expanduser())])
    if extra_args:
        argv.extend(extra_args)
    if options:
        for k, v in options.items():
            argv.append(f"{k}={v}")

    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f"pounce did not finish within {timeout_seconds}s"
        ) from e
    elapsed = time.monotonic() - start

    def tail(s: str, n: int = 4000) -> str:
        return s if len(s) <= n else "...\n" + s[-n:]

    result: dict[str, Any] = {
        "report_path": str(out_path),
        "exit_code": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "argv": argv,
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }
    if pre_analysis is not None:
        result["analysis"] = pre_analysis
    if out_path.exists():
        try:
            result["summary"] = R.summarize(R.load_report(out_path))
        except R.ReportError as e:
            result["summary_error"] = str(e)
    return result


# ---- explain / citations --------------------------------------------


@mcp.tool()
def explain(term: str) -> dict[str, Any]:
    """Look up a per-iter column name or a diagnose finding code.

    Returns a glossary entry with definition, typical range, what
    abnormal values usually mean, and citation keys you can pass to
    `citations(key=...)`. On unknown terms, returns the closest
    matches so the agent can re-query.

    Args:
        term: Column name (e.g. "inf_pr", "mu", "alpha_primal_char")
            or finding code (e.g. "mu_stuck", "restoration_loop").
    """
    if term in G.COLUMNS:
        return {"kind": "column", "term": term, **G.COLUMNS[term]}
    if term in G.FINDINGS:
        return {"kind": "finding", "term": term, **G.FINDINGS[term]}
    pool = list(G.COLUMNS) + list(G.FINDINGS)
    suggestions = G.fuzzy_suggest(term, pool)
    raise ValueError(
        f"unknown term {term!r}. did you mean: {suggestions}? "
        f"all columns: {sorted(G.COLUMNS)}; all findings: {sorted(G.FINDINGS)}."
    )


@mcp.tool()
def citations(
    topic: str | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    """Return curated paper references.

    Three modes:
      - No args: list available topics and the keys under each.
      - `topic="restoration"`: return entries for that subsystem.
      - `key="wachter2006"`: return that single entry verbatim.

    Entries include `bibtex`-style fields (title, author, year, url)
    plus an `entry_type` (article, inproceedings, ...). Backed by
    `.crucible/references.bib` in the repo.

    Args:
        topic: One of the subsystem topics (see no-arg mode for the list).
        key: A specific bib key.
    """
    bib = G.load_bib()
    if key is not None and topic is not None:
        raise ValueError("specify at most one of `topic` or `key`")
    if key is not None:
        if key not in bib:
            raise ValueError(
                f"unknown citation key {key!r}. available: {sorted(bib)[:20]}..."
                if bib else
                f"unknown citation key {key!r} (bib file not found at runtime)."
            )
        return {"key": key, **bib[key]}
    if topic is not None:
        if topic not in G.TOPICS:
            raise ValueError(
                f"unknown topic {topic!r}. valid: {sorted(G.TOPICS)}"
            )
        keys = G.TOPICS[topic]
        return {
            "topic": topic,
            "entries": [
                {"key": k, **bib.get(k, {"missing": True})} for k in keys
            ],
        }
    return {
        "topics": {t: list(keys) for t, keys in sorted(G.TOPICS.items())},
        "n_entries_loaded": len(bib),
    }


# ---- post-mortem report tools ---------------------------------------


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
        raise ValueError(
            f"unknown trace column(s): {unknown}. valid: {list(full)}"
        )
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
        raise ValueError(
            f"labels length ({len(labels)}) does not match paths length "
            f"({len(paths)})"
        )
    use_labels = labels if labels is not None else paths
    return R.compare([
        (label, R.load_report(p)) for label, p in zip(use_labels, paths)
    ])


@mcp.tool()
def linear_solver_summary(path: str) -> dict[str, Any]:
    """Aggregate post-mortem from the symmetric linear solver.

    Populated when the workspace-default FERAL backend ran the solve;
    `null` (returned as `{"summary": null, "available": false}`) for
    HSL MA57 / custom backends and for older reports written before
    the field existed.

    Fields when present:
        solver_name        Backend identity ("feral").
        n_factors          Total factor() calls completed.
        n_pattern_reuse    Of those, how many reused the prior symbolic
                           factorisation (cheap path). Healthy IPM
                           workloads expect this to dominate.
        n_pattern_changes  How many required a fresh symbolic factorisation.
        max_fill_ratio     Largest nnz(L)/nnz(A) seen across factors;
                           values >> 10 on KKT systems indicate ordering trouble.
        min_abs_pivot      Smallest |pivot| seen; approaches the precision
                           floor when the matrix is near-singular.
        max_abs_pivot      Largest |pivot| seen.
        last_inertia       (positive, negative, zero) inertia of the final
                           factorisation. For a clean IPM at convergence
                           this should be (n, m, 0).
        last_nnz_a         nnz(A) at the final factor.
        last_nnz_l         nnz(L) at the final factor.

    Args:
        path: Path to the solve report.
    """
    summary = R.linear_solver_summary(R.load_report(path))
    return {"available": summary is not None, "summary": summary}


# ---- GAMS-link tools ------------------------------------------------

# Suites shipped under gams/nlpbench/instances/<suite>/, plus the
# stand-alone examples under gams/examples/ and the top-level smoke test.
_GAMS_SUITES = (
    "globallib.gms",
    "mittelmann.gms",
    "princetonlib.gms",
    "powerflow.gms",
)


def _find_repo_root() -> Path:
    """Walk up to find the pounce repo root (contains gams/ and Cargo.toml)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").exists() and (parent / "gams").is_dir():
            return parent
    raise FileNotFoundError("could not locate pounce repo root from MCP server")


def _find_gams_bin() -> str:
    """Locate the gams CLI. GAMS_BIN env wins, then $PATH, then macOS framework."""
    env = os.environ.get("GAMS_BIN")
    if env and Path(env).exists():
        return env
    which = shutil.which("gams")
    if which:
        return which
    mac = Path("/Library/Frameworks/GAMS.framework/Versions/Current/Resources/gams")
    if mac.exists():
        return str(mac)
    raise FileNotFoundError(
        "could not locate `gams`. Set GAMS_BIN or put gams on PATH."
    )


_GMS_HEADER_COUNTS = {
    "equations":   ("Equation counts",   ("Total", "E", "G", "L", "N", "X")),
    "variables":   ("Variable counts",   ("Total", "cont", "binary", "integer",
                                          "sos1", "sos2", "scont", "sint")),
    "nonzeros":    ("Nonzero counts",    ("Total", "const", "NL", "DLL")),
}


def _parse_gms_convert_header(text: str) -> dict[str, Any]:
    """Parse the comment block emitted by `gams convert`.

    Lines look like:
        *  Equation counts
        *     Total       E       G       L       N       X
        *       109     108       0       1       0       0
    Returns {} when the file wasn't produced by `convert`.
    """
    out: dict[str, Any] = {}
    lines = [l for l in text.splitlines() if l.startswith("*")]
    for key, (header, _cols) in _GMS_HEADER_COUNTS.items():
        for i, line in enumerate(lines):
            if header not in line:
                continue
            # Convert emits 1 or 2 column-header lines between the title and
            # the actual count line. Scan the next 4 lines for the first
            # one that's all numeric.
            nums: list[str] = []
            for j in range(i + 1, min(i + 5, len(lines))):
                cand = [t for t in lines[j].lstrip("*").split() if t.lstrip("-").isdigit()]
                if cand:
                    nums = cand
                    break
            if not nums:
                continue
            if key == "equations":
                out["n_equations_total"] = int(nums[0])
                if len(nums) >= 2: out["n_equality_eqs"] = int(nums[1])
                if len(nums) >= 3: out["n_ge_eqs"]       = int(nums[2])
                if len(nums) >= 4: out["n_le_eqs"]       = int(nums[3])
            elif key == "variables":
                out["n_variables_total"] = int(nums[0])
                if len(nums) >= 2: out["n_continuous_vars"] = int(nums[1])
                if len(nums) >= 3: out["n_binary_vars"]     = int(nums[2])
                if len(nums) >= 4: out["n_integer_vars"]    = int(nums[3])
            elif key == "nonzeros":
                out["nnz_total"] = int(nums[0])
                if len(nums) >= 2: out["nnz_constant"] = int(nums[1])
                if len(nums) >= 3: out["nnz_nonlinear"] = int(nums[2])
            break
    return out


def _parse_gms_solve_directive(text: str) -> dict[str, Any]:
    """Find the `Solve <model> using <TYPE> [minimizing|maximizing] <objvar>;` line."""
    import re
    pat = re.compile(
        r"^\s*Solve\s+(\w+)\s+using\s+(\w+)"
        r"(?:\s+(minimizing|maximizing)\s+(\w+))?",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return {}
    return {
        "model_name": m.group(1),
        "model_type": m.group(2).upper(),
        "direction": (m.group(3) or "").lower() or None,
        "objective_var": m.group(4),
    }


def _gms_classify(model_type: str | None, dims: dict[str, Any]) -> str:
    """Map GAMS model type + dims into a human description."""
    if not model_type:
        return "unknown"
    table = {
        "NLP":   "nonlinear program (continuous)",
        "DNLP":  "non-differentiable NLP",
        "RMINLP":"relaxed mixed-integer NLP",
        "MINLP": "mixed-integer NLP",
        "LP":    "linear program",
        "MIP":   "mixed-integer linear",
        "QCP":   "quadratically constrained program",
        "CNS":   "constrained nonlinear system",
    }
    base = table.get(model_type, f"{model_type} model")
    nnl = dims.get("nnz_nonlinear", 0)
    if model_type in ("NLP", "DNLP") and nnl == 0:
        base += " (linear in nonzero pattern — should solve trivially)"
    return base


def _suggest_gams_options(dims: dict[str, Any], model_type: str | None) -> list[dict[str, str]]:
    """pounce.opt key/value suggestions for a GAMS NLP."""
    suggestions: list[dict[str, str]] = []
    n_var = dims.get("n_variables_total", 0)
    n_eq  = dims.get("n_equations_total", 0)
    nnl   = dims.get("nnz_nonlinear", 0)
    size = n_var + n_eq

    if model_type in ("MINLP", "MIP"):
        suggestions.append({
            "option": "(none)",
            "value": "",
            "why": (
                f"model type is {model_type}; POUNCE handles only NLP/DNLP/RMINLP. "
                "Either relax the integrality (RMINLP) or pick a different solver."
            ),
        })
        return suggestions

    suggestions.append({
        "option": "mu_strategy",
        "value": "adaptive",
        "why": (
            "matches GAMS-IPOPT's effective default (optipopt.def). pounce's "
            "compile-time default is `monotone`, which stalls some hard NLPs."
        ),
    })
    if size > 5_000:
        suggestions.append({
            "option": "linear_solver",
            "value": "ma57",
            "why": (
                f"problem is medium/large ({n_var} vars, {n_eq} eqs); MA57 is much "
                "faster than FERAL at this scale. Requires --features ma57 build."
            ),
        })
    if nnl > 0 and nnl > 0.5 * dims.get("nnz_total", 1):
        suggestions.append({
            "option": "tol",
            "value": "1e-6",
            "why": (
                "heavily nonlinear pattern: tightening below 1e-6 often leads "
                "to dual stagnation on degenerate KKT systems."
            ),
        })
    return suggestions


def _parse_lst_solve_summary(text: str) -> dict[str, Any]:
    """Parse the `S O L V E   S U M M A R Y` block of a GAMS .lst file.

    Plus extracts the embedded `=C ...` solver-status block (where POUNCE
    writes its termination one-liner).
    """
    import re

    summary: dict[str, Any] = {}

    pat_model = re.compile(r"MODEL\s+(\S+).*?OBJECTIVE\s+(\S+)", re.IGNORECASE)
    pat_solver = re.compile(r"SOLVER\s+(\S+)\s+FROM LINE\s+(\d+)", re.IGNORECASE)
    pat_status = re.compile(r"\*\*\*\*\s+SOLVER STATUS\s+(\d+)\s+(.+)")
    pat_mstat  = re.compile(r"\*\*\*\*\s+MODEL STATUS\s+(\d+)\s+(.+)")
    pat_obj    = re.compile(r"\*\*\*\*\s+OBJECTIVE VALUE\s+(\S+)")
    pat_res    = re.compile(r"RESOURCE USAGE,\s*LIMIT\s+(\S+)\s+(\S+)")
    pat_it     = re.compile(r"ITERATION COUNT,\s*LIMIT\s+(\S+)\s+(\S+)")
    pat_eval   = re.compile(r"EVALUATION ERRORS\s+(\S+)\s+(\S+)")

    for line in text.splitlines():
        if m := pat_model.search(line):
            summary["model"] = m.group(1)
            summary["objective_var"] = m.group(2)
        elif m := pat_solver.search(line):
            summary["solver"] = m.group(1)
            summary["from_line"] = int(m.group(2))
        elif m := pat_status.search(line):
            summary["solver_status_code"] = int(m.group(1))
            summary["solver_status"] = m.group(2).strip()
        elif m := pat_mstat.search(line):
            summary["model_status_code"] = int(m.group(1))
            summary["model_status"] = m.group(2).strip()
        elif m := pat_obj.search(line):
            v = m.group(1)
            try:
                summary["objective_value"] = float(v)
            except ValueError:
                summary["objective_value"] = v  # e.g. "NA"
        elif m := pat_res.search(line):
            try: summary["resource_used_secs"] = float(m.group(1))
            except ValueError: summary["resource_used_secs"] = m.group(1)
            try: summary["resource_limit_secs"] = float(m.group(2))
            except ValueError: pass
        elif m := pat_it.search(line):
            try: summary["iteration_count"] = int(m.group(1))
            except ValueError: summary["iteration_count"] = m.group(1)
            try: summary["iteration_limit"] = int(m.group(2))
            except ValueError: pass
        elif m := pat_eval.search(line):
            try: summary["evaluation_errors"] = int(m.group(1))
            except ValueError: summary["evaluation_errors"] = m.group(1)

    # Extract the embedded solver-status block. Two formats appear in the wild:
    #   (a) `=C ...` lines wrapped by `SOLVER STATUS FILE LISTED BELOW/ABOVE`
    #       (some solvelink modes / older GAMS).
    #   (b) A plain block beginning with `--- POUNCE: ...` and ending just
    #       before the `---- EQU` / `---- VAR` solution tables (current
    #       in-process dylib path).
    lines = text.splitlines()
    solver_block_lines: list[str] = []
    # (a) =C wrapped form
    in_block = False
    for line in lines:
        if "SOLVER STATUS FILE LISTED BELOW" in line:
            in_block = True
            continue
        if "SOLVER STATUS FILE LISTED ABOVE" in line:
            in_block = False
            continue
        if in_block and line.startswith("=C"):
            solver_block_lines.append(line[2:].rstrip())
    # (b) `--- POUNCE:` form
    if not solver_block_lines:
        capturing = False
        for line in lines:
            if not capturing and line.startswith("--- POUNCE"):
                capturing = True
            if capturing:
                if line.startswith("---- ") or line.startswith("EXECUTION TIME"):
                    break
                solver_block_lines.append(line.rstrip())
    if solver_block_lines:
        summary["solver_status_file"] = "\n".join(solver_block_lines).rstrip()

    return summary


@mcp.tool()
def list_gams_examples(
    suite: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Enumerate the GAMS .gms instances bundled under `gams/`.

    Three sources are scanned: the four `nlpbench/instances/<suite>/` test
    suites (globallib, mittelmann, princetonlib, powerflow), the
    `gams/examples/` directory, and the top-level `gams/test_hs071.gms`
    smoke problem. Pass `suite=None` to get counts across all of them;
    pass a suite name to list files (offset/limit-paginated).

    Args:
        suite: One of "globallib.gms", "mittelmann.gms",
            "princetonlib.gms", "powerflow.gms", "examples", "smoke",
            or None for a high-level summary.
        limit: Max files to return when `suite` is set.
        offset: Skip this many files (for paging).

    Returns:
        When `suite` is None: `{"suites": [{name, count, root}, ...]}`.
        Otherwise: `{"suite", "root", "count", "limit", "offset", "files": [...]}`.
    """
    root = _find_repo_root()
    gams_dir = root / "gams"

    def _suite_root(name: str) -> Path:
        if name in _GAMS_SUITES:
            return gams_dir / "nlpbench" / "instances" / name
        if name == "examples":
            return gams_dir / "examples"
        if name == "smoke":
            return gams_dir
        raise ValueError(
            f"unknown suite {name!r}; valid: {list(_GAMS_SUITES) + ['examples', 'smoke']}"
        )

    if suite is None:
        suites = []
        for s in _GAMS_SUITES:
            d = gams_dir / "nlpbench" / "instances" / s
            n = sum(1 for _ in d.glob("*.gms")) if d.is_dir() else 0
            suites.append({"name": s, "count": n, "root": str(d)})
        ex = gams_dir / "examples"
        suites.append({
            "name": "examples",
            "count": sum(1 for _ in ex.glob("*.gms")) if ex.is_dir() else 0,
            "root": str(ex),
        })
        suites.append({
            "name": "smoke",
            "count": 1 if (gams_dir / "test_hs071.gms").exists() else 0,
            "root": str(gams_dir),
        })
        return {"suites": suites, "total": sum(s["count"] for s in suites)}

    sroot = _suite_root(suite)
    if suite == "smoke":
        f = gams_dir / "test_hs071.gms"
        files = [str(f)] if f.exists() else []
    else:
        if not sroot.is_dir():
            return {"suite": suite, "root": str(sroot), "count": 0, "files": []}
        files = sorted(str(p) for p in sroot.glob("*.gms"))

    return {
        "suite": suite,
        "root": str(sroot),
        "count": len(files),
        "limit": limit,
        "offset": offset,
        "files": files[offset:offset + limit],
    }


@mcp.tool()
def analyze_gams_problem(gms_file: str) -> dict[str, Any]:
    """Inspect a .gms file without solving it.

    Parses the comment-block header that `gams convert` emits (variable /
    equation / nonzero counts) and the `Solve ... using <TYPE>` line.
    Returns dimensions, model class, suggested pounce.opt entries, and
    heuristic warnings. For .gms files that were hand-written rather than
    convert-translated, the dimensions may be empty — pounce will still
    solve, but pass `analyze=False` to `run_gams_problem` in that case.

    Args:
        gms_file: Path to a .gms file.
    """
    p = Path(gms_file).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"no such .gms file: {p}")
    text = p.read_text(errors="replace")

    dims = _parse_gms_convert_header(text)
    solve = _parse_gms_solve_directive(text)
    model_type = solve.get("model_type")

    warnings: list[str] = []
    if not dims:
        warnings.append(
            "no `gams convert` header found — dimensions could not be parsed. "
            "POUNCE will still solve the model; the suggestion list is conservative."
        )
    if not solve:
        warnings.append("no `Solve` directive found in file — is this a complete model?")
    if model_type in ("MINLP", "MIP"):
        warnings.append(
            f"model type {model_type} is not supported by POUNCE "
            "(integer variables present)."
        )
    if dims.get("n_binary_vars", 0) or dims.get("n_integer_vars", 0):
        warnings.append("discrete variables present; POUNCE solves the continuous relaxation only.")

    return {
        "path": str(p),
        "dimensions": dims,
        "solve_directive": solve,
        "class": _gms_classify(model_type, dims),
        "supported_by_pounce": model_type in ("NLP", "DNLP", "RMINLP") if model_type else None,
        "suggestions": _suggest_gams_options(dims, model_type),
        "warnings": warnings,
    }


@mcp.tool()
def parse_gams_listing(lst_file: str) -> dict[str, Any]:
    """Parse the SOLVE SUMMARY block from a GAMS .lst file.

    Extracts model/solver identity, status codes (solver and model),
    objective value, resource/iteration usage, and the embedded `=C ...`
    solver status block (the per-solver one-liner GAMS echoes into the
    listing — POUNCE writes its termination message there).

    Args:
        lst_file: Path to a GAMS `.lst` listing.

    Returns:
        Dict with parsed `summary` fields. Missing fields are simply
        absent (the parser is tolerant). When the listing reports
        "Could not spawn solver", `solver_status_code` will be 13 and
        `solver_status_file` will be empty.
    """
    p = Path(lst_file).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"no such .lst file: {p}")
    return {"path": str(p), "summary": _parse_lst_solve_summary(p.read_text(errors="replace"))}


@mcp.tool()
def run_gams_problem(
    gms_file: str,
    options: dict[str, str] | None = None,
    json_detail: str = "full",
    workdir: str | None = None,
    extra_pounce_opt_lines: list[str] | None = None,
    timeout_seconds: float = 600.0,
    analyze: bool = True,
) -> dict[str, Any]:
    """Run a .gms problem through GAMS with POUNCE and capture the JSON report.

    Workflow:
        1. Copy the .gms into a working directory (tempdir if omitted).
        2. Write a `pounce.opt` containing user `options` plus the
           `json_output` / `json_detail` keys that route a
           pounce.solve-report/v1 JSON to disk.
        3. Invoke `gams <stem>.gms NLP=POUNCE optfile=1`.
        4. Parse the resulting `.lst` SOLVE SUMMARY block.
        5. If the JSON report was written, parse and summarise it too.

    The GAMS link must have been built and installed (see `gams/Makefile`)
    with the JSON-report instrumentation present in pounce.h.

    Args:
        gms_file: Path to the .gms file to solve.
        options: pounce.opt key/value pairs (e.g. {"tol": "1e-6",
            "max_iter": "5000", "mu_strategy": "adaptive"}).
        json_detail: "summary" or "full" (default "full"). Use "full"
            so the post-mortem MCP tools (diagnose, find_stalls, etc.)
            have iteration history to work on.
        workdir: Directory to run in (created if missing). When None, a
            tempdir is created and its path returned in `workdir`.
        extra_pounce_opt_lines: Raw pounce.opt lines to append verbatim
            (for options not in the simple key/value table).
        timeout_seconds: Kill the GAMS subprocess after this many seconds.
        analyze: When True (default), call `analyze_gams_problem` first
            and embed the result under `analysis`.

    Returns:
        Dict with `workdir`, `gms_file`, `lst_file`, `log_file`,
        `report_path`, `exit_code`, `elapsed_seconds`, `argv`, the
        parsed `lst_summary`, and (when the JSON report was written)
        a `report_summary` from `summarize`. Also includes `analysis`
        when analyze=True.
    """
    if json_detail not in ("summary", "full"):
        raise ValueError(
            f"json_detail must be 'summary' or 'full', got {json_detail!r}"
        )
    src = Path(gms_file).expanduser()
    if not src.exists():
        raise FileNotFoundError(f"no such .gms file: {src}")

    pre_analysis: dict[str, Any] | None = None
    if analyze:
        try:
            pre_analysis = analyze_gams_problem(str(src))
        except (ValueError, FileNotFoundError) as e:
            pre_analysis = {"error": str(e)}

    gams_bin = _find_gams_bin()

    if workdir is None:
        wd = Path(tempfile.mkdtemp(prefix=f"pounce-gams-{src.stem}-"))
        wd_created = True
    else:
        wd = Path(workdir).expanduser()
        wd.mkdir(parents=True, exist_ok=True)
        wd_created = False

    staged_gms = wd / src.name
    shutil.copy2(src, staged_gms)
    report_path = wd / f"{src.stem}.report.json"

    opt_lines = [
        "# pounce.opt generated by pounce-studio MCP run_gams_problem",
        "print_level 0",
    ]
    if options:
        for k, v in options.items():
            opt_lines.append(f"{k} {v}")
    if extra_pounce_opt_lines:
        opt_lines.extend(extra_pounce_opt_lines)
    opt_lines.append(f"json_output {report_path}")
    opt_lines.append(f"json_detail {json_detail}")
    (wd / "pounce.opt").write_text("\n".join(opt_lines) + "\n")

    argv = [gams_bin, str(staged_gms), "NLP=POUNCE", "optfile=1", "lo=2"]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_seconds, cwd=wd,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f"gams did not finish within {timeout_seconds}s"
        ) from e
    elapsed = time.monotonic() - start

    lst_file = wd / f"{src.stem}.lst"
    log_file = wd / f"{src.stem}.log"

    def tail(s: str, n: int = 4000) -> str:
        return s if len(s) <= n else "...\n" + s[-n:]

    result: dict[str, Any] = {
        "workdir": str(wd),
        "workdir_created": wd_created,
        "gms_file": str(staged_gms),
        "lst_file": str(lst_file) if lst_file.exists() else None,
        "log_file": str(log_file) if log_file.exists() else None,
        "report_path": str(report_path) if report_path.exists() else None,
        "exit_code": proc.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "argv": argv,
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }
    if pre_analysis is not None:
        result["analysis"] = pre_analysis
    if lst_file.exists():
        result["lst_summary"] = _parse_lst_solve_summary(
            lst_file.read_text(errors="replace")
        )
    if report_path.exists():
        try:
            result["report_summary"] = R.summarize(R.load_report(report_path))
        except R.ReportError as e:
            result["report_summary_error"] = str(e)
    return result


def main() -> None:
    """Entry point used by `python -m pounce_studio_mcp`."""
    mcp.run()


if __name__ == "__main__":
    main()
