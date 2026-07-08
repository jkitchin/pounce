"""Starting-point preflight: evaluate a problem once at x0, before any
solve, and report what iteration 0 will see.

This is the Python twin of the ``pounce check-x0`` CLI subcommand (see
``docs/src/initialization.md``): NaN/inf evaluations are fatal (a solve
would abort with ``Invalid_Number_Detected``), bound violations and
on-bound components will be moved by the interior clamp, very large
initial constraint violation usually means a wrong or missing starting
point, and wide derivative magnitude spread is the early signal for
scaling trouble.

The native :class:`pounce.Problem` does not expose its callbacks or
bounds, so ``preflight`` takes the same pieces the ``Problem``
constructor does::

    report = pounce.preflight(problem_obj, x0, lb=lb, ub=ub, cl=cl, cu=cu)
    if report.fatal:
        raise ValueError(str(report))
    x, info = pounce.Problem(n, m, problem_obj=problem_obj,
                             lb=lb, ub=ub, cl=cl, cu=cu).solve(x0=x0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

__all__ = ["preflight", "PreflightReport"]

# Bounds at or beyond this magnitude are treated as infinite, matching
# the solver's NLP_LOWER_BOUND_INF / NLP_UPPER_BOUND_INF sentinels.
_BOUND_INF = 1e19


@dataclass
class PreflightReport:
    """Result of :func:`preflight`. ``str(report)`` renders a readable
    text report; :meth:`to_dict` gives the JSON-ready form (the same
    shape as ``pounce check-x0 --json``, minus file provenance)."""

    n: int
    m: int
    objective: Optional[float]
    x0_all_zero: bool
    # non-finite scans: (index-or-(row,col), value) pairs, capped lists
    grad_nonfinite: List[Tuple[int, float]]
    grad_nonfinite_count: int
    g_nonfinite: List[Tuple[int, float]]
    g_nonfinite_count: int
    jac_nonfinite: List[Tuple[int, int, float]]
    jac_nonfinite_count: int
    hess_nonfinite_count: Optional[int]
    eval_errors: List[str]
    # x0 vs bounds
    bound_violations: List[Tuple[int, float, float, float, float]]  # (j, x, lo, hi, viol)
    n_bound_violations: int
    max_bound_violation: float
    n_on_bounds: int
    # interior-clamp preview
    clamp_moves: List[Tuple[int, float, float, float]]  # (j, from, to, dist)
    n_clamp_moved: int
    max_clamp_move: float
    # initial constraint violation
    con_violations: List[Tuple[int, float, float, float, float]]  # (i, g, lo, hi, viol)
    n_con_violations: int
    max_con_violation: float
    # derivative scale spread: (max_abs, min_abs_nonzero, ratio)
    grad_spread: Tuple[float, float, float]
    jac_spread: Tuple[float, float, float]
    # rollup
    warnings: List[str] = field(default_factory=list)
    fatal: bool = False
    verdict: str = "CLEAN"

    @property
    def ok(self) -> bool:
        """True when the model evaluates cleanly at x0 (warnings allowed)."""
        return not self.fatal

    def to_dict(self) -> dict:
        return {
            "schema": "pounce.check-x0/v1",
            "problem": {"n_vars": self.n, "n_cons": self.m},
            "x0": {"all_zero": self.x0_all_zero},
            "evaluation": {
                "objective": self.objective
                if self.objective is not None and np.isfinite(self.objective)
                else None,
                "objective_finite": self.objective is not None
                and bool(np.isfinite(self.objective)),
                "grad_nonfinite_count": self.grad_nonfinite_count,
                "constraints_nonfinite_count": self.g_nonfinite_count,
                "jacobian_nonfinite_count": self.jac_nonfinite_count,
                "hessian_nonfinite_count": self.hess_nonfinite_count,
                "errors": list(self.eval_errors),
            },
            "bounds": {
                "n_violations": self.n_bound_violations,
                "max_violation": self.max_bound_violation,
                "n_on_bounds": self.n_on_bounds,
                "worst": [
                    {"index": j, "value": x, "lower": lo, "upper": hi, "violation": v}
                    for (j, x, lo, hi, v) in self.bound_violations
                ],
            },
            "interior_clamp": {
                "n_moved": self.n_clamp_moved,
                "max_move": self.max_clamp_move,
                "worst": [
                    {"index": j, "from": a, "to": b, "distance": d}
                    for (j, a, b, d) in self.clamp_moves
                ],
            },
            "constraint_violation": {
                "n_violated": self.n_con_violations,
                "max_violation": self.max_con_violation,
                "worst": [
                    {"index": i, "value": g, "lower": lo, "upper": hi, "violation": v}
                    for (i, g, lo, hi, v) in self.con_violations
                ],
            },
            "derivative_scale": {
                "gradient": dict(
                    zip(("max_abs", "min_abs_nonzero", "ratio"), self.grad_spread)
                ),
                "jacobian": dict(
                    zip(("max_abs", "min_abs_nonzero", "ratio"), self.jac_spread)
                ),
            },
            "warnings": list(self.warnings),
            "fatal": self.fatal,
            "verdict": self.verdict,
        }

    def __str__(self) -> str:  # noqa: C901 - straightforward rendering
        lines = ["pounce preflight — starting-point check"]
        lines.append(f"  problem : {self.n} vars, {self.m} cons")
        lines.append(f"  x0      : {'all zeros' if self.x0_all_zero else 'supplied'}")
        obj = (
            f"{self.objective:.10e}"
            if self.objective is not None and np.isfinite(self.objective)
            else f"{self.objective}  <- NON-FINITE"
            if self.objective is not None
            else "EVALUATION FAILED"
        )
        lines.append(f"  objective at x0: {obj}")
        for label, count in (
            ("gradient", self.grad_nonfinite_count),
            ("constraints", self.g_nonfinite_count),
            ("Jacobian", self.jac_nonfinite_count),
        ):
            state = "finite" if count == 0 else f"{count} non-finite entries"
            lines.append(f"  {label:<11}: {state}")
        if self.hess_nonfinite_count is not None:
            state = (
                "finite (lambda=0)"
                if self.hess_nonfinite_count == 0
                else f"{self.hess_nonfinite_count} non-finite entries (lambda=0)"
            )
            lines.append(f"  Hessian    : {state}")
        for err in self.eval_errors:
            lines.append(f"  eval error : {err}")
        lines.append(
            f"  bounds     : {self.n_bound_violations} violated, "
            f"{self.n_on_bounds} on-bound; clamp moves {self.n_clamp_moved} "
            f"(max {self.max_clamp_move:.3e})"
        )
        lines.append(
            f"  constraints: {self.n_con_violations} violated at x0 "
            f"(max {self.max_con_violation:.3e})"
        )
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        lines.append(f"  VERDICT: {self.verdict}")
        return "\n".join(lines)


def _as_bounds(bound, n: int, default: float) -> np.ndarray:
    if bound is None:
        return np.full(n, default)
    b = np.asarray(bound, dtype=float).ravel()
    if b.size != n:
        raise ValueError(f"bound has length {b.size}, expected {n}")
    return b


def _finite_bound(b: float) -> bool:
    return bool(np.isfinite(b)) and -_BOUND_INF < b < _BOUND_INF


def _box_violation(v: float, lo: float, hi: float) -> float:
    if not np.isfinite(v):
        return float("inf")
    below = lo - v if _finite_bound(lo) else -float("inf")
    above = v - hi if _finite_bound(hi) else -float("inf")
    return max(below, above, 0.0)


def _clamp_to_interior(
    x: float, lo: float, hi: float, bound_push: float, bound_frac: float
) -> float:
    """The per-component interior clamp from the solver's
    ``DefaultIterateInitializer`` (see docs/src/initialization.md)."""
    flo, fhi = _finite_bound(lo), _finite_bound(hi)
    if flo and fhi:
        span = hi - lo
        p_l = min(bound_push * max(abs(lo), 1.0), bound_frac * span)
        p_u = min(bound_push * max(abs(hi), 1.0), bound_frac * span)
        return min(max(x, lo + p_l), hi - p_u)
    if flo:
        return max(x, lo + bound_push * max(abs(lo), 1.0))
    if fhi:
        return min(x, hi - bound_push * max(abs(hi), 1.0))
    return x


def _scale_spread(values: np.ndarray) -> Tuple[float, float, float]:
    a = np.abs(np.asarray(values, dtype=float).ravel())
    a = a[np.isfinite(a) & (a > 0.0)]
    if a.size == 0:
        return (0.0, 0.0, 0.0)
    mx, mn = float(a.max()), float(a.min())
    return (mx, mn, mx / mn)


def _nonfinite_indexed(values: np.ndarray, cap: int) -> Tuple[List[Tuple[int, float]], int]:
    v = np.asarray(values, dtype=float).ravel()
    bad = np.flatnonzero(~np.isfinite(v))
    return [(int(i), float(v[i])) for i in bad[:cap]], int(bad.size)


def preflight(
    problem_obj: Any,
    x0,
    *,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    feas_tol: float = 1e-6,
    bound_push: float = 1e-2,
    bound_frac: float = 1e-2,
    max_list: int = 5,
) -> PreflightReport:
    """Evaluate ``problem_obj`` once at ``x0`` and report what the solver's
    first iteration will see.

    Parameters mirror the :class:`pounce.Problem` constructor:
    ``problem_obj`` is a cyipopt-style object with ``objective(x)`` and
    optionally ``gradient(x)``, ``constraints(x)``, ``jacobian(x)`` (+
    ``jacobianstructure()``), ``hessian(x, lagrange, obj_factor)`` (+
    ``hessianstructure()``); ``lb``/``ub`` are variable bounds and
    ``cl``/``cu`` constraint bounds (``None`` = unbounded).

    Returns a :class:`PreflightReport`. ``report.fatal`` is True exactly
    when an evaluation produced NaN/inf or raised, i.e. when a solve from
    this ``x0`` would abort with ``Invalid_Number_Detected``. Bound and
    constraint violations are reported but are *not* fatal: the solver
    clamps bound violations and the interior-point method accepts an
    infeasible start.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    eval_errors: List[str] = []

    def _try(label, fn, *args):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 - report, don't crash
            eval_errors.append(f"{label} raised {type(e).__name__}: {e}")
            return None

    # --- constraint bounds first (they define m) ---
    g = None
    if hasattr(problem_obj, "constraints"):
        g = _try("constraints(x0)", problem_obj.constraints, x0)
    if g is not None:
        g = np.asarray(g, dtype=float).ravel()
        m = g.size
    elif cl is not None:
        m = np.asarray(cl, dtype=float).ravel().size
    else:
        m = 0
    x_l = _as_bounds(lb, n, -np.inf)
    x_u = _as_bounds(ub, n, np.inf)
    g_l = _as_bounds(cl, m, -np.inf)
    g_u = _as_bounds(cu, m, np.inf)

    # --- evaluations at x0 ---
    fval = _try("objective(x0)", problem_obj.objective, x0)
    objective = float(fval) if fval is not None else None

    grad = None
    if hasattr(problem_obj, "gradient"):
        grad = _try("gradient(x0)", problem_obj.gradient, x0)
    grad = np.asarray(grad, dtype=float).ravel() if grad is not None else np.zeros(0)
    grad_nonfinite, grad_nonfinite_count = _nonfinite_indexed(grad, max_list)

    g_arr = g if g is not None else np.zeros(0)
    g_nonfinite, g_nonfinite_count = _nonfinite_indexed(g_arr, max_list)

    # Jacobian: honor jacobianstructure() when present, else dense (m, n).
    jac_nonfinite: List[Tuple[int, int, float]] = []
    jac_nonfinite_count = 0
    jac_vals = np.zeros(0)
    if m > 0 and hasattr(problem_obj, "jacobian"):
        jv = _try("jacobian(x0)", problem_obj.jacobian, x0)
        if jv is not None:
            jac_vals = np.asarray(jv, dtype=float).ravel()
            if hasattr(problem_obj, "jacobianstructure"):
                rows, cols = problem_obj.jacobianstructure()
                rows = np.asarray(rows, dtype=int).ravel()
                cols = np.asarray(cols, dtype=int).ravel()
            else:
                rows, cols = np.divmod(np.arange(jac_vals.size), n)
            bad = np.flatnonzero(~np.isfinite(jac_vals))
            jac_nonfinite_count = int(bad.size)
            jac_nonfinite = [
                (int(rows[k]), int(cols[k]), float(jac_vals[k]))
                for k in bad[:max_list]
                if k < rows.size
            ]

    # Hessian of the Lagrangian at (x0, lambda=0, obj_factor=1).
    hess_nonfinite_count: Optional[int] = None
    if hasattr(problem_obj, "hessian"):
        hv = _try(
            "hessian(x0, 0, 1)", problem_obj.hessian, x0, np.zeros(m), 1.0
        )
        if hv is not None:
            hvals = np.asarray(hv, dtype=float).ravel()
            hess_nonfinite_count = int(np.count_nonzero(~np.isfinite(hvals)))

    # --- x0 vs bounds + clamp preview ---
    bound_violations: List[Tuple[int, float, float, float, float]] = []
    n_bound_violations = 0
    max_bound_violation = 0.0
    n_on_bounds = 0
    clamp_moves: List[Tuple[int, float, float, float]] = []
    n_clamp_moved = 0
    max_clamp_move = 0.0
    for j in range(n):
        viol = _box_violation(x0[j], x_l[j], x_u[j])
        if viol > feas_tol:
            n_bound_violations += 1
            if np.isfinite(viol):
                max_bound_violation = max(max_bound_violation, viol)
            bound_violations.append((j, float(x0[j]), float(x_l[j]), float(x_u[j]), viol))
        if np.isfinite(x0[j]):
            at_lo = _finite_bound(x_l[j]) and abs(x0[j] - x_l[j]) <= 1e-8 * (
                1.0 + abs(x_l[j])
            )
            at_hi = _finite_bound(x_u[j]) and abs(x_u[j] - x0[j]) <= 1e-8 * (
                1.0 + abs(x_u[j])
            )
            if at_lo or at_hi:
                n_on_bounds += 1
            to = _clamp_to_interior(x0[j], x_l[j], x_u[j], bound_push, bound_frac)
            d = abs(to - x0[j])
            if d > 0.0:
                n_clamp_moved += 1
                max_clamp_move = max(max_clamp_move, d)
                clamp_moves.append((j, float(x0[j]), float(to), float(d)))
    bound_violations.sort(key=lambda t: -t[4])
    del bound_violations[max_list:]
    clamp_moves.sort(key=lambda t: -t[3])
    del clamp_moves[max_list:]

    # --- initial constraint violation ---
    con_violations: List[Tuple[int, float, float, float, float]] = []
    n_con_violations = 0
    max_con_violation = 0.0
    for i in range(g_arr.size):
        viol = _box_violation(g_arr[i], g_l[i], g_u[i])
        if viol > feas_tol:
            n_con_violations += 1
            if np.isfinite(viol):
                max_con_violation = max(max_con_violation, viol)
            con_violations.append((i, float(g_arr[i]), float(g_l[i]), float(g_u[i]), viol))
    con_violations.sort(key=lambda t: -t[4])
    del con_violations[max_list:]

    grad_spread = _scale_spread(grad)
    jac_spread = _scale_spread(jac_vals)

    # --- warnings + verdict ---
    warnings: List[str] = []
    nonfinite_total = (
        grad_nonfinite_count
        + g_nonfinite_count
        + jac_nonfinite_count
        + (hess_nonfinite_count or 0)
        + int(objective is not None and not np.isfinite(objective))
    )
    fatal = bool(eval_errors) or nonfinite_total > 0 or objective is None
    if eval_errors:
        warnings.append(
            "an evaluation callback raised at the starting point; the solver "
            "cannot start from this x0"
        )
    if nonfinite_total > 0:
        warnings.append(
            f"{nonfinite_total} non-finite value(s) at the starting point; a "
            "solve would abort with Invalid_Number_Detected. Move x0 into the "
            "domain or add bounds that keep it there"
        )
    x0_all_zero = n > 0 and bool(np.all(x0 == 0.0))
    if x0_all_zero:
        warnings.append("the starting point is all zeros")
    if n_bound_violations > 0:
        warnings.append(
            f"x0 violates {n_bound_violations} variable bound(s) "
            f"(max {max_bound_violation:.3e}); the initializer will clamp them inside"
        )
    if n_on_bounds > 0:
        warnings.append(
            f"{n_on_bounds} component(s) of x0 sit exactly on a bound and will "
            f"be pushed into the interior (bound_push={bound_push:.1e}); if x0 "
            "is a previous solution, use the warm-start recipe "
            "(warm_start_init_point=yes with tightened warm_start_bound_push/_frac)"
        )
    if max_con_violation > 1e4:
        warnings.append(
            f"very large initial infeasibility (max constraint violation "
            f"{max_con_violation:.3e}); consider a better starting point or "
            "least_square_init_primal=yes"
        )
    for label, spread in (("gradient", grad_spread), ("Jacobian", jac_spread)):
        if spread[2] > 1e8 or spread[0] > 1e8:
            warnings.append(
                f"{label} magnitudes at x0 span a large range (max {spread[0]:.3e}, "
                f"min nonzero {spread[1]:.3e}); see the scaling reference page"
            )

    verdict = "FATAL" if fatal else ("WARNINGS" if warnings else "CLEAN")

    return PreflightReport(
        n=n,
        m=m,
        objective=objective,
        x0_all_zero=x0_all_zero,
        grad_nonfinite=grad_nonfinite,
        grad_nonfinite_count=grad_nonfinite_count,
        g_nonfinite=g_nonfinite,
        g_nonfinite_count=g_nonfinite_count,
        jac_nonfinite=jac_nonfinite,
        jac_nonfinite_count=jac_nonfinite_count,
        hess_nonfinite_count=hess_nonfinite_count,
        eval_errors=eval_errors,
        bound_violations=bound_violations,
        n_bound_violations=n_bound_violations,
        max_bound_violation=max_bound_violation,
        n_on_bounds=n_on_bounds,
        clamp_moves=clamp_moves,
        n_clamp_moved=n_clamp_moved,
        max_clamp_move=max_clamp_move,
        con_violations=con_violations,
        n_con_violations=n_con_violations,
        max_con_violation=max_con_violation,
        grad_spread=grad_spread,
        jac_spread=jac_spread,
        warnings=warnings,
        fatal=fatal,
        verdict=verdict,
    )
