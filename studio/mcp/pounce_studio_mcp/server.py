"""MCP server exposing pounce solve reports as Claude-callable tools.

Two families of tools:

  * **Post-mortem** (the majority) — stateless analysis of a finished
    `pounce.solve-report/v1` JSON file, plus `run_problem`/`run_gams_problem`
    batch tools that shell out to produce a fresh report. Every such tool
    takes a file path and holds no state between calls.

  * **Live debug sessions** (`debug_start` / `debug_command` / `debug_state`
    / `debug_sessions` / `debug_close`) — a stateful proxy over the CLI's
    `pounce --debug-json` protocol. `debug_start` spawns a long-lived solver
    child and parks it; `debug_command` steps it one command at a time. This
    lets an agent drive a *live*, steppable interior-point solve over MCP.
    `debug_session_guide` documents the underlying wire protocol (for callers
    who'd rather drive the CLI directly).
"""
from __future__ import annotations

import atexit
import itertools
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import glossary as G
from . import reports as R
from .verify_sig import check_signature


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


# ---- verify (independent solution check) ----------------------------


@mcp.tool()
def verify_solution(
    nl_file: str,
    sol_file: str,
    feas_tol: float = 1e-6,
    opt_tol: float = 1e-6,
    require_optimal: bool = False,
    expected_problem_sha256: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Independently verify that a `.sol` satisfies a `.nl`'s constraints.

    This is the trust anchor for agent workflows: it re-derives feasibility
    from the **canonical** problem rather than trusting the `.sol`'s status
    line or the agent's claim. Use it to confirm a solution before acting on
    it. The agent can *request* a check but cannot fake its result — the
    verdict comes from the `pounce verify` binary, and (when this server
    holds `POUNCE_VERIFY_KEY`) the receipt's HMAC-SHA256 signature is
    re-checked here.

    Args:
        nl_file: Path to the canonical AMPL `.nl` problem (the source of
            truth). Verification runs against THIS model's constraints and
            bounds, not whatever produced the `.sol`.
        sol_file: Path to the claimed AMPL `.sol` solution to check.
        feas_tol: Feasibility tolerance for constraints and bounds.
        opt_tol: Stationarity tolerance for the optimality check.
        require_optimal: If True, also reject a feasible-but-not-stationary
            point (needs duals in the `.sol`).
        expected_problem_sha256: If given, assert the canonical `.nl` hashes
            to this value. Use it to bind the check to a specific, pinned
            problem so a swapped/relaxed model is caught.
        timeout_seconds: Kill the check after this many seconds.

    Returns:
        Dict with `verified` (the bottom line), `verdict`, `exit_code`,
        `max_constraint_violation`, `max_bound_violation`, `problem_sha256`,
        `solution_sha256`, `signature_present`, `signature_valid`
        (True/False/None), `problem_matches_expected`, and `receipt` (the
        full parsed JSON). `verified` is only trustworthy when, for your use
        case, `signature_valid` is True (if you sign) and
        `problem_matches_expected` is True (if you pin a hash).
    """
    binary = _find_pounce_bin()
    nl = Path(nl_file).expanduser()
    sol = Path(sol_file).expanduser()
    if not nl.exists():
        raise FileNotFoundError(f"no such .nl file: {nl}")
    if not sol.exists():
        raise FileNotFoundError(f"no such .sol file: {sol}")

    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="pounce-verify-")
    os.close(fd)
    receipt_path = Path(tmp)

    argv: list[str] = [
        binary, "verify", str(nl), str(sol),
        "--feas-tol", repr(feas_tol),
        "--opt-tol", repr(opt_tol),
        "--json-output", str(receipt_path),
    ]
    if require_optimal:
        argv.append("--require-optimal")

    # The binary inherits this process's environment, so if the server holds
    # POUNCE_VERIFY_KEY the receipt is signed. NOTE: that key only stays out
    # of the agent's reach if this server runs in a SEPARATE trust boundary
    # (different user/container/host). If the agent has a shell on the same
    # user/host, it can read this process's environment and the signature is
    # not a real protection — fall back to the consumer recomputing `pounce
    # verify`. See docs/src/verify.md ("Out-of-process signing").
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f"pounce verify did not finish within {timeout_seconds}s"
        ) from e

    receipt: dict[str, Any] = {}
    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text())
        except json.JSONDecodeError:
            receipt = {}
        finally:
            receipt_path.unlink(missing_ok=True)

    key = os.environ.get("POUNCE_VERIFY_KEY") or ""
    signature_present = "signature" in receipt
    signature_valid: bool | None
    if not key:
        signature_valid = None  # not signing in this deployment
    elif not signature_present:
        signature_valid = False  # we expected a signature but got none
    else:
        signature_valid = check_signature(receipt, key)

    problem_sha256 = receipt.get("problem", {}).get("sha256")
    problem_matches_expected: bool | None
    if expected_problem_sha256 is None:
        problem_matches_expected = None
    else:
        problem_matches_expected = problem_sha256 == expected_problem_sha256

    feas = receipt.get("feasibility", {})
    return {
        "verified": bool(receipt.get("verified")) and proc.returncode == 0,
        "verdict": receipt.get("verdict"),
        "exit_code": proc.returncode,
        "max_constraint_violation": feas.get("max_constraint_violation"),
        "max_bound_violation": feas.get("max_bound_violation"),
        "worst_constraint": feas.get("worst_constraint"),
        "problem_sha256": problem_sha256,
        "solution_sha256": receipt.get("solution", {}).get("sha256"),
        "signature_present": signature_present,
        "signature_valid": signature_valid,
        "problem_matches_expected": problem_matches_expected,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "receipt": receipt,
    }


# ---- check-x0 (starting-point preflight) -----------------------------


@mcp.tool()
def check_x0(
    nl_file: str | None = None,
    builtin: str | None = None,
    x0_file: str | None = None,
    feas_tol: float = 1e-6,
    bound_push: float = 1e-2,
    bound_frac: float = 1e-2,
    max_list: int = 5,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Preflight a model's starting point before any solve.

    Evaluates the model once at its starting point (the `.nl` initial-guess
    segment, or `x0_file`) and reports what iteration 0 will see: NaN/inf
    evaluations (fatal — the solve would abort with
    `Invalid_Number_Detected`), bound violations of x0, how far the
    `bound_push` interior clamp will move the point, initial constraint
    violation per row, and derivative magnitude spread (the early signal
    for scaling trouble). Costs one evaluation of each callback: no
    factorization, no solve.

    Use this BEFORE `run_problem` when a model is new, when a previous
    solve died with `Invalid_Number_Detected`, or when a warm start did
    not help (on-bound components + the clamp preview explain that case).

    Args:
        nl_file: Path to the AMPL `.nl` problem. Exactly one of `nl_file`
            or `builtin` must be given.
        builtin: Name of a built-in problem (see `list_builtins`).
        x0_file: Optional whitespace-separated file of n values overriding
            the model's starting point (e.g. a candidate warm-start point).
        feas_tol: Violations above this are counted (default 1e-6).
        bound_push: `bound_push` used for the clamp preview (default 1e-2,
            the solver default).
        bound_frac: `bound_frac` used for the clamp preview (default 1e-2).
        max_list: Max offenders listed per category.
        timeout_seconds: Kill the check after this many seconds.

    Returns:
        Dict with `fatal` (the bottom line: True means a solve from this
        point would abort), `verdict` (CLEAN / WARNINGS / FATAL),
        `warnings` (human-readable findings), `exit_code` (0 clean or
        warnings, 21 fatal), and `report` (the full parsed
        `pounce.check-x0/v1` JSON: evaluation, bounds, interior_clamp,
        constraint_violation, derivative_scale sections).
    """
    return _check_x0_impl(
        nl_file=nl_file,
        builtin=builtin,
        x0_file=x0_file,
        feas_tol=feas_tol,
        bound_push=bound_push,
        bound_frac=bound_frac,
        max_list=max_list,
        timeout_seconds=timeout_seconds,
    )


def _check_x0_impl(
    nl_file: str | None,
    builtin: str | None,
    x0_file: str | None,
    feas_tol: float,
    bound_push: float,
    bound_frac: float,
    max_list: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    if (nl_file is None) == (builtin is None):
        raise ValueError("pass exactly one of nl_file or builtin")
    binary = _find_pounce_bin()

    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="pounce-check-x0-")
    os.close(fd)
    report_path = Path(tmp)

    argv: list[str] = [binary, "check-x0"]
    if nl_file is not None:
        nl = Path(nl_file).expanduser()
        if not nl.exists():
            raise FileNotFoundError(f"no such .nl file: {nl}")
        argv.append(str(nl))
    else:
        argv += ["--builtin", str(builtin)]
    if x0_file is not None:
        x0 = Path(x0_file).expanduser()
        if not x0.exists():
            raise FileNotFoundError(f"no such x0 file: {x0}")
        argv += ["--x0-file", str(x0)]
    argv += [
        "--feas-tol", repr(feas_tol),
        "--bound-push", repr(bound_push),
        "--bound-frac", repr(bound_frac),
        "--max-list", str(max_list),
        "--json-output", str(report_path),
    ]

    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f"pounce check-x0 did not finish within {timeout_seconds}s"
        ) from e

    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            report = {}
        finally:
            report_path.unlink(missing_ok=True)

    return {
        "fatal": bool(report.get("fatal", proc.returncode == 21)),
        "verdict": report.get("verdict"),
        "warnings": report.get("warnings", []),
        "exit_code": proc.returncode,
        "max_constraint_violation": report.get("constraint_violation", {}).get(
            "max_violation"
        ),
        "n_clamp_moved": report.get("interior_clamp", {}).get("n_moved"),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "report": report,
    }


# ---- suggest_initialization ------------------------------------------


@mcp.tool()
def suggest_initialization(
    nl_file: str | None = None,
    builtin: str | None = None,
    x0_file: str | None = None,
    prior_report: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Preflight a model's starting point and propose concrete fixes.

    Runs the `check_x0` preflight and translates its findings into an
    ordered list of deterministic, advisory suggestions: solver options
    to set, Python/Pyomo helper calls to make, and doc pointers. When a
    `prior_report` (a `pounce.solve-report/v1` JSON from a previous
    solve of this model) is supplied, its outcome sharpens the advice
    (e.g. restoration-heavy solves point at the starting point; a clean
    preflight with a failed solve points away from it).

    Suggestions are **advisory** — never auto-applied. Present them to
    the user in order and let them choose.

    Args:
        nl_file: Path to the AMPL `.nl` problem (or use `builtin`).
        builtin: Name of a built-in problem.
        x0_file: Optional candidate starting point to check instead of
            the model's own.
        prior_report: Optional path to a previous solve's JSON report.
        timeout_seconds: Kill the preflight after this many seconds.

    Returns:
        Dict with `verdict`, `fatal`, `suggestions` (list of
        `{kind, why, options?, python?, pyomo?, docs?}` in priority
        order), and `preflight` (the underlying check-x0 result).
    """
    checked = _check_x0_impl(
        nl_file=nl_file,
        builtin=builtin,
        x0_file=x0_file,
        feas_tol=1e-6,
        bound_push=1e-2,
        bound_frac=1e-2,
        max_list=5,
        timeout_seconds=timeout_seconds,
    )
    rep = checked.get("report", {})
    suggestions: list[dict[str, Any]] = []

    ev = rep.get("evaluation", {})
    nonfinite = (
        ev.get("grad_nonfinite_count", 0)
        + ev.get("constraints_nonfinite_count", 0)
        + ev.get("jacobian_nonfinite_count", 0)
        + (ev.get("hessian_nonfinite_count") or 0)
        + (0 if ev.get("objective_finite", True) else 1)
    )
    if checked.get("fatal"):
        suggestions.append({
            "kind": "fix-evaluation",
            "why": f"{nonfinite} non-finite evaluation(s) at the starting "
                   "point; a solve would abort with Invalid_Number_Detected. "
                   "The interior clamp only repairs bound violations, not "
                   "domain errors on free variables.",
            "python": "set an in-domain x0; pounce.preflight(problem_obj, x0, "
                      "...) reproduces this check on callback problems",
            "pyomo": "pyomo_pounce.initialize_missing_values(model) then "
                     "pyomo_pounce.preflight(model)",
            "docs": "initialization.md#diagnosing-a-bad-start",
        })

    if rep.get("x0", {}).get("all_zero"):
        suggestions.append({
            "kind": "provide-x0",
            "why": "the starting point is all zeros: the model supplies no "
                   "initial guess (or an explicitly zero one).",
            "python": "pounce.generate_starts(n, bounds=..., "
                      "strategy='midpoint') for a bounds-aware start",
            "pyomo": "set Var .value before solve, or "
                     "pyomo_pounce.block_initialize(model)",
            "docs": "initialization.md#where-the-starting-point-comes-from",
        })

    clamp = rep.get("interior_clamp", {})
    bounds_sec = rep.get("bounds", {})
    if bounds_sec.get("n_on_bounds", 0) > 0 and clamp.get("n_moved", 0) > 0:
        suggestions.append({
            "kind": "warm-start-recipe",
            "why": f"{bounds_sec['n_on_bounds']} component(s) sit exactly on a "
                   "bound and the interior clamp will move them (max move "
                   f"{clamp.get('max_move')}). If this x0 is a previous "
                   "solution, a plain re-solve discards it.",
            "options": {
                "warm_start_init_point": "yes",
                "mu_init": 1e-7,
                "warm_start_bound_push": 1e-9,
                "warm_start_bound_frac": 1e-9,
                "warm_start_slack_bound_push": 1e-9,
                "warm_start_slack_bound_frac": 1e-9,
                "warm_start_mult_bound_push": 1e-9,
            },
            "python": "ws = pounce.WarmStart.from_info(x, info); "
                      "prob.solve(warm_start=ws)",
            "docs": "initialization.md#warm-starting-the-interior-point-path",
        })

    max_viol = rep.get("constraint_violation", {}).get("max_violation") or 0.0
    if max_viol > 1e4:
        suggestions.append({
            "kind": "reduce-initial-infeasibility",
            "why": f"very large initial constraint violation ({max_viol:.3e}).",
            "options": {"least_square_init_primal": "yes"},
            "python": "x0 = pounce.project_to_feasible(problem_obj, x0, ...)",
            "docs": "initialization.md#what-the-solver-does-with-your-point-cold-start",
        })

    scale = rep.get("derivative_scale", {})
    for label in ("gradient", "jacobian"):
        s = scale.get(label, {})
        if (s.get("ratio") or 0.0) > 1e8 or (s.get("max_abs") or 0.0) > 1e8:
            suggestions.append({
                "kind": "scaling",
                "why": f"{label} magnitudes at x0 span a large range "
                       f"(max {s.get('max_abs')}, min nonzero "
                       f"{s.get('min_abs_nonzero')}); poor scaling mimics a "
                       "bad starting point.",
                "docs": "scaling.md",
            })
            break

    # Prior-solve context, when supplied.
    if prior_report is not None:
        try:
            r = R.load_report(Path(prior_report).expanduser())
            summary = R.summarize(r)
            findings = R.diagnose(r)
        except Exception as e:  # noqa: BLE001 - advisory only
            suggestions.append({
                "kind": "prior-report-unreadable",
                "why": f"could not read prior report: {e}",
            })
        else:
            status = str(summary.get("status", ""))
            codes = {f.get("code") for f in findings.get("findings", [])} if isinstance(
                findings, dict
            ) else set()
            if {"restoration_used", "restoration_loop"} & codes:
                suggestions.append({
                    "kind": "restoration-points-at-x0",
                    "why": f"the previous solve ({status}) spent time in "
                           "feasibility restoration — a classic symptom of a "
                           "poor starting point.",
                    "docs": "initialization.md",
                })
            elif status not in ("SolveSucceeded", "SolvedToAcceptableLevel") \
                    and not suggestions:
                suggestions.append({
                    "kind": "not-an-initialization-problem",
                    "why": f"the preflight is clean but the previous solve "
                           f"ended {status}: the starting point is probably "
                           "not the bottleneck.",
                    "docs": "troubleshooting.md",
                })

    if not suggestions:
        suggestions.append({
            "kind": "clean",
            "why": "the starting point evaluates cleanly with no red flags; "
                   "if the solve still struggles, look beyond initialization.",
            "docs": "troubleshooting.md",
        })

    return {
        "verdict": checked.get("verdict"),
        "fatal": checked.get("fatal"),
        "suggestions": suggestions,
        "preflight": checked,
    }


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

    Post-mortem over a finished report. To pause *live* the moment a
    stall develops, drive `pounce --debug-json` (`break on mu_stalled`);
    see `debug_session_guide`.

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

    This is a *post-mortem* over a finished report. To step through a
    solve live — pausing on these same conditions, inspecting and mutating
    the iterate — drive `pounce --debug-json`; call `debug_session_guide`
    for the protocol and a launch snippet.

    Args:
        path: Path to the solve report.

    Returns:
        Dict with `findings`: list of {severity, code, message} and
        `n_findings`: total count.
    """
    return R.diagnose(R.load_report(path))


@mcp.tool()
def debug_session_guide() -> dict[str, Any]:
    """How to drive POUNCE's *live* interactive debugger (`--debug-json`).

    The MCP tools here (diagnose, find_stalls, restoration_windows, …)
    analyze a *finished* solve report. For a live, steppable session —
    pause at each iteration, inspect/mutate the iterate and barrier
    parameter, set conditional/event breakpoints, rewind, re-solve — the
    `pounce` CLI speaks a self-describing newline-delimited JSON protocol.

    Two ways to use it from here:
      * **Through this server (recommended):** call `debug_start` to spawn
        and park a session, then `debug_command` to step it. The server
        manages the child process and the wire protocol for you.
      * **Driving the CLI yourself:** spawn `pounce --debug-json` with
        piped stdio and speak the protocol below directly.

    This tool needs no arguments; it returns the contract plus a
    ready-to-run launch snippet either way.

    Returns:
        Dict with `proxy_tools` (the in-server path), `launch`, `protocol`,
        `contract` (the step-by-step loop), `metrics` (the scalar field
        names carried by every event), and `docs` (path to the full spec).
    """
    return {
        "proxy_tools": {
            "start": "debug_start(builtin=… | nl_file=…) → {session_id, hello, pause}",
            "step": "debug_command(session_id, cmd=…, args=[…]) → {result, state, …}",
            "inspect": "debug_state(session_id); debug_sessions()",
            "close": "debug_close(session_id)",
            "note": "Prefer these over spawning the CLI yourself; the server "
            "owns the child process and the framing.",
        },
        "launch": "pounce <model.nl> --debug-json   "
        "# or: pounce --problem rosenbrock --debug-json",
        "transport": "Spawn the CLI with stdin and stdout piped. Read one "
        "JSON object per line from stdout; write one command object "
        "(or bare string) per line to stdin.",
        "protocol": "pounce-dbg/1",
        "contract": [
            "Read the first line: a `hello` event enumerating `commands`, "
            "`events`, `checkpoints`, `metrics`, `blocks`, and a "
            "`capabilities` map. Feature-detect off these lists, not the "
            "version string.",
            "Send commands as `{\"cmd\": \"...\", \"id\": N}` lines, e.g. "
            "`{\"cmd\": \"break if inf_pr<1e-6\", \"id\": 1}` then "
            "`{\"cmd\": \"continue\", \"id\": 2}`. The `id` is echoed back "
            "as `request_id` on the matching `result` event.",
            "Read `pause` / `progress` / `terminated` events. Each carries "
            "the scalar metrics under the names in `hello.metrics`, so you "
            "can index them directly.",
            "Finish with `{\"cmd\": \"continue\"}` to run to completion "
            "(then read `terminated`), or `{\"cmd\": \"quit\"}` to stop.",
        ],
        "metrics": [
            "iter",
            "mu",
            "objective",
            "inf_pr",
            "inf_du",
            "nlp_error",
            "complementarity",
        ],
        "docs": "docs/src/debugger.md",
    }


# ---- live debug session proxy (--debug-json) ------------------------
#
# Unlike every other tool here, the debug_* tools hold STATE between
# calls. `debug_start` spawns a long-lived `pounce --debug-json` child and
# parks it in `_SESSIONS` under an opaque id; `debug_command` writes one
# command to that child's stdin and reads its response events back off
# stdout, so an agent can step a *live* interior-point solve over MCP.
# `debug_close` (or process exit) reaps the child.
#
# Event model (verified against the CLI):
#   * Every command emits one `result` echoing the request `id`.
#   * "Flow" verbs (continue/step/run/resolve/sweep/multistart/quit/…)
#     then stream `progress` events and end at a `pause`; at solve end the
#     stop is a `pause` with checkpoint "terminated" followed by a
#     `terminated` event.
#   * Every other verb (print/info/break/set/diagnose/goto/…) emits only
#     the `result` and parks back at the prompt.

_SESSIONS: dict[str, "_DebugSession"] = {}
_SESSIONS_LOCK = threading.Lock()
_MAX_SESSIONS = 8

# Verbs that resume the solver and therefore stream events until the next
# stop. Keyed on the command's first whitespace token.
_FLOW_VERBS = frozenset({
    "continue", "c", "step", "s", "n", "stepi", "si", "run", "r",
    "detach", "resolve", "sweep", "multistart", "quit", "q", "exit",
})

_STARTUP_TIMEOUT = 30.0   # seconds to see hello + the first pause
_PAUSE_GRACE = 30.0       # after an in-band pause, seconds to reach a checkpoint


class _DebugSession:
    """One live `pounce --debug-json` child, driven one command at a time.

    A daemon thread drains the child's stdout into a queue (so we never
    fight Python's read-ahead buffering with select); `command` writes to
    stdin and pulls the response back off the queue under a per-session
    lock.
    """

    def __init__(self, sid: str, proc: subprocess.Popen, stderr_path: str,
                 stderr_fh: Any, argv: list[str], label: str) -> None:
        self.id = sid
        self.proc = proc
        self.stderr_path = stderr_path
        self._stderr_fh = stderr_fh
        self.argv = argv
        self.label = label
        self.hello: dict[str, Any] | None = None
        self.last_pause: dict[str, Any] | None = None
        self.terminated: dict[str, Any] | None = None
        self.finished = False  # parked at the `terminated` checkpoint
        self.n_commands = 0
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    # --- low-level io ---------------------------------------------------

    def _pump(self) -> None:
        """Drain stdout → queue, parsing each line as JSON. Daemon thread."""
        try:
            for raw in self.proc.stdout:  # type: ignore[union-attr]
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    self._q.put(("event", json.loads(raw)))
                except json.JSONDecodeError:
                    self._q.put(("noise", raw))
        finally:
            self._q.put(("eof", None))

    def _read(self, timeout: float) -> tuple[str, Any]:
        try:
            return self._q.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            return ("timeout", None)

    def _send(self, obj: dict[str, Any]) -> None:
        self.proc.stdin.write(json.dumps(obj) + "\n")  # type: ignore[union-attr]
        self.proc.stdin.flush()  # type: ignore[union-attr]

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stderr_tail(self, n: int = 2000) -> str:
        try:
            data = Path(self.stderr_path).read_text(errors="replace")
        except OSError:
            return ""
        return data if len(data) <= n else "...\n" + data[-n:]

    # --- handshake ------------------------------------------------------

    def read_startup(self, timeout: float) -> None:
        """Block until both `hello` and the initial `pause` arrive."""
        deadline = time.monotonic() + timeout
        while self.hello is None or self.last_pause is None:
            kind, ev = self._read(deadline - time.monotonic())
            if kind == "timeout":
                raise TimeoutError(
                    "pounce --debug-json did not emit hello+pause within "
                    f"{timeout}s. stderr tail:\n{self.stderr_tail()}"
                )
            if kind == "eof":
                raise RuntimeError(
                    "pounce --debug-json exited before the handshake "
                    f"(exit {self.proc.poll()}). stderr tail:\n{self.stderr_tail()}"
                )
            if kind != "event":
                continue
            e = ev.get("event")
            if e == "hello":
                self.hello = ev
            elif e == "pause":
                self.last_pause = ev
            elif e == "terminated":
                self.terminated = ev
                return

    # --- command --------------------------------------------------------

    def command(self, cmd: str, args: list[str] | None,
                timeout: float) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
        """Send one command; return (result, streamed_events, outcome)."""
        with self._lock:
            if self.terminated is not None:
                raise RuntimeError(
                    f"session {self.id} already terminated "
                    f"({self.terminated.get('status')}); start a new one."
                )
            if not self.alive():
                raise RuntimeError(
                    f"session {self.id} process is dead (exit "
                    f"{self.proc.poll()}). stderr tail:\n{self.stderr_tail()}"
                )

            rid = next(self._ids)
            verb = (cmd.split()[0] if cmd.strip() else "").lower()
            payload: dict[str, Any] = {"cmd": cmd, "id": rid}
            if args:
                payload["args"] = [str(a) for a in args]
            self._send(payload)
            self.n_commands += 1

            # 1) read up to the matching result; stash any stray events.
            result: dict[str, Any] | None = None
            pre: list[dict[str, Any]] = []
            r_deadline = time.monotonic() + timeout
            while result is None:
                kind, ev = self._read(r_deadline - time.monotonic())
                if kind == "timeout":
                    raise TimeoutError(
                        f"no `result` for command {cmd!r} within {timeout}s"
                    )
                if kind == "eof":
                    raise RuntimeError(
                        f"process exited awaiting result for {cmd!r}. "
                        f"stderr tail:\n{self.stderr_tail()}"
                    )
                if kind != "event":
                    continue
                if ev.get("event") == "result" and ev.get("request_id") == rid:
                    result = ev
                    break
                if ev.get("event") == "pause":
                    self.last_pause = ev
                elif ev.get("event") == "terminated":
                    self.terminated = ev
                pre.append(ev)

            # 2) flow verbs stream until the next stop; others park now.
            if verb in _FLOW_VERBS and self.terminated is None:
                tail, outcome = self._drain(timeout)
                return result, pre + tail, outcome
            outcome = "terminated" if self.terminated is not None else "parked"
            return result, pre, outcome

    def _drain(self, timeout: float) -> tuple[list[dict[str, Any]], str]:
        """Read streamed events until the next stop, with timeout recovery.

        If the solve outruns `timeout`, send an in-band `pause` so it stops
        at the next checkpoint rather than leaving the protocol mid-stream.
        """
        events: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        pause_sent = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not pause_sent:
                    try:
                        self._send({"cmd": "pause"})
                    except OSError:
                        return events, "eof"
                    pause_sent = True
                    deadline = time.monotonic() + _PAUSE_GRACE
                    continue
                return events, "stuck"
            kind, ev = self._read(remaining)
            if kind == "timeout":
                continue
            if kind == "eof":
                return events, "eof"
            if kind != "event":
                continue
            events.append(ev)
            e = ev.get("event")
            if e == "terminated":
                # The summary event: only fires once the solve is resumed
                # *past* the terminal checkpoint (or stdin closes). Process
                # is on its way out.
                self.terminated = ev
                self.finished = True
                return events, "terminated"
            if e == "pause":
                self.last_pause = ev
                if ev.get("checkpoint") == "terminated":
                    # The terminal checkpoint is a real stop: the solve is
                    # done (status is non-null) but parked here so the
                    # final/failing iterate stays inspectable. The summary
                    # `terminated` event only follows a resume past it.
                    self.finished = True
                    return events, "finished"
                return events, "interrupted" if pause_sent else "paused"

    # --- teardown -------------------------------------------------------

    def close(self) -> int | None:
        with self._lock:
            if self.alive():
                try:
                    self._send({"cmd": "quit"})
                except OSError:
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    stream.close()  # type: ignore[union-attr]
                except OSError:
                    pass
            try:
                self._stderr_fh.close()
            except OSError:
                pass
            return self.proc.poll()


def _spawn_session(argv: list[str], label: str) -> _DebugSession:
    sid = uuid.uuid4().hex[:12]
    fd, stderr_path = tempfile.mkstemp(suffix=".stderr", prefix="pounce-dbg-")
    os.close(fd)
    stderr_fh = open(stderr_path, "w")
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=stderr_fh, text=True, bufsize=1,
    )
    return _DebugSession(sid, proc, stderr_path, stderr_fh, argv, label)


def _get_session(session_id: str) -> _DebugSession:
    with _SESSIONS_LOCK:
        sess = _SESSIONS.get(session_id)
    if sess is None:
        with _SESSIONS_LOCK:
            active = sorted(_SESSIONS)
        raise ValueError(
            f"no such debug session {session_id!r}. Active: {active}. "
            "Start one with debug_start."
        )
    return sess


@atexit.register
def _reap_sessions() -> None:
    with _SESSIONS_LOCK:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
    for s in sessions:
        try:
            s.close()
        except Exception:
            pass


@mcp.tool()
def debug_start(
    builtin: str | None = None,
    nl_file: str | None = None,
    options: dict[str, str] | None = None,
    options_file: str | None = None,
    setup: list[str] | None = None,
    extra_args: list[str] | None = None,
    label: str | None = None,
    startup_timeout: float = 30.0,
) -> dict[str, Any]:
    """Start a LIVE, steppable solve and hold it open across calls.

    Spawns `pounce <model> --debug-json` and parks it at its first pause
    (iteration 0). Returns a `session_id` you pass to `debug_command` to
    step the solve, inspect/mutate the iterate, set breakpoints, re-solve,
    etc. — the full interactive debugger, driven over MCP. Close it with
    `debug_close` when done (sessions also reap at server shutdown).

    This is the *live* counterpart to the post-mortem tools (`diagnose`,
    `find_stalls`, …) which analyze a finished report. Feature-detect off
    the returned `hello.commands` / `hello.capabilities`, not the version.

    Exactly one of `builtin` or `nl_file` must be specified.

    Args:
        builtin: Built-in test problem (rosenbrock, quadratic,
            bounded-quadratic, eq-quadratic, circle).
        nl_file: Path to an AMPL .nl file. Mutually exclusive with builtin.
        options: OptionsList key=value pairs forwarded to the solver
            (e.g. {"mu_strategy": "adaptive"}).
        options_file: Optional ipopt.opt-format file path.
        setup: Debugger commands to run immediately after the first pause,
            e.g. ["break if inf_pr<1e-6", "stop-at kkt"]. Each is a
            non-resuming command; its result is returned under
            `setup_results`.
        extra_args: Escape hatch for raw CLI flags.
        label: Friendly name for this session (shown in debug_sessions).
        startup_timeout: Seconds to wait for the hello+pause handshake.

    Returns:
        Dict with `session_id`, `argv`, the self-describing `hello`
        handshake, the initial `pause` state, and `setup_results`.
    """
    if (builtin is None) == (nl_file is None):
        raise ValueError("specify exactly one of `builtin` or `nl_file`")

    with _SESSIONS_LOCK:
        live = sum(1 for s in _SESSIONS.values() if s.alive())
    if live >= _MAX_SESSIONS:
        raise RuntimeError(
            f"too many live debug sessions ({live}/{_MAX_SESSIONS}); "
            "close one with debug_close first."
        )

    binary = _find_pounce_bin()
    argv: list[str] = [binary]
    if nl_file:
        nl = Path(nl_file).expanduser()
        if not nl.exists():
            raise FileNotFoundError(f"no such .nl file: {nl}")
        argv.append(str(nl))
    else:
        argv.extend(["--problem", builtin])  # type: ignore[list-item]
    argv.append("--debug-json")
    if options_file:
        argv.extend(["--options-file", str(Path(options_file).expanduser())])
    if extra_args:
        argv.extend(extra_args)
    if options:
        for k, v in options.items():
            argv.append(f"{k}={v}")

    sess = _spawn_session(argv, label or (nl_file or builtin or "session"))
    try:
        sess.read_startup(startup_timeout)
    except (TimeoutError, RuntimeError):
        sess.close()
        raise

    with _SESSIONS_LOCK:
        _SESSIONS[sess.id] = sess

    setup_results: list[dict[str, Any]] = []
    if setup:
        for c in setup:
            res, _events, _outcome = sess.command(c, None, 30.0)
            setup_results.append({
                "command": c,
                "ok": res.get("ok"),
                "output": res.get("output"),
            })

    return {
        "session_id": sess.id,
        "argv": argv,
        "hello": sess.hello,
        "pause": sess.last_pause,
        "setup_results": setup_results or None,
        "note": (
            "Drive with debug_command(session_id, cmd=…). Feature-detect "
            "off hello.commands / hello.capabilities. Close with debug_close."
        ),
    }


@mcp.tool()
def debug_command(
    session_id: str,
    cmd: str,
    args: list[str] | None = None,
    timeout_seconds: float = 60.0,
    max_progress: int = 5,
) -> dict[str, Any]:
    """Send ONE command to a live debug session and read its response.

    `cmd` is any debugger verb from the session's `hello.commands` — a bare
    string (`"continue"`, `"break if inf_pr<1e-6"`, `"diagnose"`) or a verb
    plus `args` (`cmd="print", args=["x"]`). The full vocabulary is in the
    `hello` handshake from `debug_start`; see `debug_session_guide` for the
    protocol.

    Resuming verbs (continue/step/run/resolve/sweep/multistart/…) run the
    solver until the next stop and stream `progress` events along the way;
    this call blocks until that stop. If the solve outruns
    `timeout_seconds`, an in-band `pause` is sent so it halts cleanly at the
    next checkpoint (`outcome="interrupted"`) rather than corrupting the
    stream — raise the timeout or set a breakpoint to go further.

    Args:
        session_id: From debug_start.
        cmd: Debugger command verb or full command string.
        args: Optional argument tokens for the verb.
        timeout_seconds: Max wait for a resuming command to reach a stop.
        max_progress: How many trailing `progress` events to echo back
            (they are summarized to a count + this many tail samples).

    Outcomes:
        parked       a non-resuming command ran; back at the prompt.
        paused       a resuming command stopped at a breakpoint/checkpoint.
        interrupted  the timeout fired; the solve was paused mid-run.
        finished     reached the terminal checkpoint — the solve is done
                     (`state.status` holds the verdict) but parked so the
                     final iterate stays inspectable. Send `continue`/`quit`
                     or `debug_close` to release it.
        terminated   the summary `terminated` event fired; process exiting.
        stuck / eof  the solve wouldn't pause, or the process exited.

    Returns:
        Dict with `ok`, the raw `result` event, the `outcome`, a `finished`
        flag, the current `state` (latest pause snapshot, carrying the
        scalar metrics and — at the terminal checkpoint — `status`),
        `terminated` (the summary event once released), a `progress`
        summary, and any other streamed `events`.
    """
    sess = _get_session(session_id)
    result, events, outcome = sess.command(cmd, args, timeout_seconds)

    progress = [e for e in events if e.get("event") == "progress"]
    streamed = [e for e in events if e.get("event") != "progress"]

    pretty = cmd if not args else f"{cmd} {' '.join(str(a) for a in args)}"
    resp: dict[str, Any] = {
        "session_id": session_id,
        "command": pretty,
        "ok": result.get("ok"),
        "result": result,
        "outcome": outcome,
        "finished": sess.finished,
        "state": sess.last_pause,
        "terminated": sess.terminated,
    }
    if progress:
        resp["progress"] = {
            "count": len(progress),
            "last": progress[-max_progress:] if max_progress > 0 else [],
        }
    if streamed:
        resp["events"] = streamed
    return resp


@mcp.tool()
def debug_state(session_id: str) -> dict[str, Any]:
    """Cached state of a live debug session (no child I/O).

    Returns the last pause snapshot, whether the solve has terminated, and
    session metadata — cheap to call between `debug_command` steps.

    Args:
        session_id: From debug_start.
    """
    sess = _get_session(session_id)
    return {
        "session_id": session_id,
        "label": sess.label,
        "alive": sess.alive(),
        "finished": sess.finished,
        "n_commands": sess.n_commands,
        "state": sess.last_pause,
        "terminated": sess.terminated,
        "argv": sess.argv,
    }


@mcp.tool()
def debug_sessions() -> dict[str, Any]:
    """List all live debug sessions held by this server.

    Returns one row per session with its id, label, liveness, command
    count, and current iteration — plus the per-server session cap.
    """
    with _SESSIONS_LOCK:
        items = [
            {
                "session_id": s.id,
                "label": s.label,
                "alive": s.alive(),
                "terminated": bool(s.terminated),
                "n_commands": s.n_commands,
                "iter": (s.last_pause or {}).get("iter"),
            }
            for s in _SESSIONS.values()
        ]
    return {"sessions": items, "count": len(items), "max_sessions": _MAX_SESSIONS}


@mcp.tool()
def debug_close(session_id: str) -> dict[str, Any]:
    """Stop a live debug session and reap its process.

    Sends `quit` if the solve is still running, waits for the child to
    exit (killing it if it ignores quit), and drops the session from the
    registry.

    Args:
        session_id: From debug_start.
    """
    with _SESSIONS_LOCK:
        sess = _SESSIONS.pop(session_id, None)
    if sess is None:
        raise ValueError(f"no such session {session_id!r} (already closed?)")
    exit_code = sess.close()
    return {
        "session_id": session_id,
        "exit_code": exit_code,
        "terminated": sess.terminated,
        "final_status": (sess.terminated or sess.last_pause or {}).get("status"),
        "stderr_tail": sess.stderr_tail(),
    }


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
