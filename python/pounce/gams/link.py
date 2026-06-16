"""POUNCE as a GAMS solver: the control-file solver link.

When GAMS solves a model with ``option nlp = pounce;`` it launches this link
with the path to a *control file*, and expects the solution written back
through the GAMS Modeling Object (GMO).  This module:

1. boots the GAMS environment (GEV) and model (GMO) objects from that control
   file (:func:`solve_from_control_file`),
2. translates the GMO instance into a POUNCE problem object
   (:mod:`pounce.gams.gmo_translate`),
3. solves it with :class:`pounce.Problem`, and
4. writes the primal/dual solution and the GAMS model/solve status back.

The GAMS-library calls live entirely in :class:`_GmoAdapter` /
:func:`solve_from_control_file`; the rest (status mapping, option parsing, the
solve wrapper :func:`solve_view`) is plain Python and unit-tested without a GAMS
installation.

This is the pure-Python, pip-installable counterpart to the native C link in
``gams/gams_pounce.c``.  It uses GAMS's own ``gamsapi`` package (the ``[gams]``
extra) and redistributes nothing GAMS-owned; the bindings ``dlopen`` the user's
GAMS libraries, so their versions must match.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np

from .gmo_translate import POUNCE_INF, problem_from_gmo

if TYPE_CHECKING:
    from .gmo_translate import GmoProblem, GmoView

# ── GAMS status constants (gmomcc) ───────────────────────────────────────────
MODELSTAT_OPTIMAL = 1
MODELSTAT_LOCALLY_OPTIMAL = 2
MODELSTAT_UNBOUNDED = 3
MODELSTAT_INFEASIBLE_LOCAL = 5
MODELSTAT_INFEASIBLE_INTERMED = 6
MODELSTAT_FEASIBLE = 7  # intermediate non-optimal, feasible point available
MODELSTAT_ERROR_NO_SOLUTION = 13
MODELSTAT_NO_SOLUTION_RETURNED = 14

SOLVESTAT_NORMAL = 1
SOLVESTAT_ITERATION = 2
SOLVESTAT_RESOURCE = 3  # time / resource interrupt
SOLVESTAT_USER = 8
SOLVESTAT_SETUP_ERR = 9
SOLVESTAT_SOLVER_ERR = 10
SOLVESTAT_EVAL_ERR = 11
SOLVESTAT_INTERNAL_ERR = 12

# POUNCE (== Ipopt 3.14) ApplicationReturnStatus names -> (modelStat, solveStat).
# Ported verbatim from map_status_to_gams() in gams/gams_pounce.c so the two
# links report identically.  `status_msg` from POUNCE is the enum *name*.
_STATUS_MAP: dict[str, tuple[int, int]] = {
    "Solve_Succeeded": (MODELSTAT_LOCALLY_OPTIMAL, SOLVESTAT_NORMAL),
    "Solved_To_Acceptable_Level": (MODELSTAT_FEASIBLE, SOLVESTAT_NORMAL),
    "Feasible_Point_Found": (MODELSTAT_FEASIBLE, SOLVESTAT_NORMAL),
    "Infeasible_Problem_Detected": (MODELSTAT_INFEASIBLE_LOCAL, SOLVESTAT_SOLVER_ERR),
    "Search_Direction_Becomes_Too_Small": (MODELSTAT_FEASIBLE, SOLVESTAT_SOLVER_ERR),
    "Diverging_Iterates": (MODELSTAT_UNBOUNDED, SOLVESTAT_SOLVER_ERR),
    "User_Requested_Stop": (MODELSTAT_FEASIBLE, SOLVESTAT_USER),
    "Maximum_Iterations_Exceeded": (MODELSTAT_FEASIBLE, SOLVESTAT_ITERATION),
    "Restoration_Failed": (MODELSTAT_INFEASIBLE_INTERMED, SOLVESTAT_SOLVER_ERR),
    "Error_In_Step_Computation": (MODELSTAT_FEASIBLE, SOLVESTAT_SOLVER_ERR),
    "Maximum_CpuTime_Exceeded": (MODELSTAT_FEASIBLE, SOLVESTAT_RESOURCE),
    "Maximum_WallTime_Exceeded": (MODELSTAT_FEASIBLE, SOLVESTAT_RESOURCE),
    "Not_Enough_Degrees_Of_Freedom": (MODELSTAT_ERROR_NO_SOLUTION, SOLVESTAT_SETUP_ERR),
    "Invalid_Problem_Definition": (MODELSTAT_ERROR_NO_SOLUTION, SOLVESTAT_SETUP_ERR),
    "Invalid_Option": (MODELSTAT_ERROR_NO_SOLUTION, SOLVESTAT_SETUP_ERR),
    "Invalid_Number_Detected": (MODELSTAT_INFEASIBLE_INTERMED, SOLVESTAT_EVAL_ERR),
    "Internal_Error": (MODELSTAT_ERROR_NO_SOLUTION, SOLVESTAT_INTERNAL_ERR),
}

# Statuses that still leave a usable primal iterate in `x` (mirrors
# pounce_status_has_solution() in the C link).
_STATUS_HAS_SOLUTION = frozenset(
    {
        "Solve_Succeeded",
        "Solved_To_Acceptable_Level",
        "Feasible_Point_Found",
        "Infeasible_Problem_Detected",
        "Search_Direction_Becomes_Too_Small",
        "User_Requested_Stop",
        "Maximum_Iterations_Exceeded",
        "Error_In_Step_Computation",
        "Maximum_CpuTime_Exceeded",
        "Maximum_WallTime_Exceeded",
    }
)

# Option-file keys handled by the link itself rather than forwarded to POUNCE,
# matching the GAMS-link-specific keys in gams_pounce.c.
_LINK_OPTION_KEYS = frozenset({"sqp_state_file", "json_output", "json_detail"})


def is_available() -> bool:
    """Return ``True`` if the GAMS expert-level Python API (gamsapi) is present."""
    try:  # pragma: no cover - depends on optional gamsapi
        import gams.core.gev  # noqa: F401
        import gams.core.gmo  # noqa: F401

        return True
    except Exception:
        return False


def status_to_gams(status_msg: str) -> tuple[int, int]:
    """Map a POUNCE ``status_msg`` (ApplicationReturnStatus name) to GAMS.

    Returns ``(modelStat, solveStat)``.  Unknown statuses map to the generic
    error pair, matching the ``default`` arm of the C link's switch.
    """
    return _STATUS_MAP.get(
        status_msg, (MODELSTAT_ERROR_NO_SOLUTION, SOLVESTAT_INTERNAL_ERR)
    )


def parse_option_file(path: str) -> tuple[dict, dict]:
    """Parse a GAMS option file (``pounce.opt``) into POUNCE options.

    Returns ``(pounce_options, link_options)``.  ``pounce_options`` maps option
    names to int / float / str values (coerced the same way the C link does:
    int first, then float, then string); ``link_options`` holds the
    link-specific keys (``json_output`` etc.).  Lines beginning with ``*`` or
    ``#`` are comments; blank lines are skipped.
    """
    pounce_opts: dict = {}
    link_opts: dict = {}
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError:
        return pounce_opts, link_opts

    for raw in lines:
        line = raw.strip()
        if not line or line[0] in ("*", "#"):
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        key, val = parts[0], parts[1].strip()
        if key in _LINK_OPTION_KEYS:
            link_opts[key] = val
            continue
        pounce_opts[key] = _coerce(val)
    return pounce_opts, link_opts


def _coerce(val: str):
    """Coerce an option string to int, else float, else leave as a string."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def solve_view(
    view: "GmoView",
    *,
    options: dict | None = None,
    max_iter: int | None = None,
    max_wall_time: float | None = None,
):
    """Translate a GMO view, build a POUNCE problem, solve, return ``(prob, x, info)``.

    This is the GAMS-library-free core of the link: given anything implementing
    :class:`~pounce.gams.gmo_translate.GmoView` it produces a solved POUNCE
    problem, so it can be exercised in tests with an in-memory fake.

    ``options`` are POUNCE option name/value pairs (e.g. from a ``pounce.opt``
    file); ``max_iter`` / ``max_wall_time`` come from the GAMS environment
    (``gevIterLim`` / ``gevResLim``) and are applied as defaults the option file
    can still override.
    """
    import pounce

    gp: "GmoProblem" = problem_from_gmo(view)
    prob = pounce.Problem(
        n=gp.n,
        m=gp.m,
        problem_obj=gp.problem_obj,
        lb=gp.lb,
        ub=gp.ub,
        cl=gp.cl,
        cu=gp.cu,
    )

    # Defaults mirroring the C link, set before the option file so a user
    # pounce.opt can override them.
    if max_iter is not None:
        prob.add_option("max_iter", int(max_iter))
    if max_wall_time is not None:
        prob.add_option("max_wall_time", float(max_wall_time))
    # acceptable_iter=0 disables acceptable-level early termination, matching
    # the GAMS Ipopt link default (pounce#138).
    prob.add_option("acceptable_iter", 0)
    if not gp.has_hessian:
        prob.add_option("hessian_approximation", "limited-memory")

    for key, value in (options or {}).items():
        try:
            prob.add_option(key, value)
        except Exception as exc:  # unknown / invalid option: warn, keep going
            sys.stderr.write(f"pounce-gams: ignoring option '{key}': {exc}\n")

    x, info = prob.solve(x0=gp.x0)
    return gp, x, info


def solve_from_control_file(
    control_file: str, sysdir: str | None = None
) -> int:  # pragma: no cover - needs GAMS
    """Entry point GAMS invokes: solve the model described by ``control_file``.

    ``sysdir`` is the GAMS system directory (passed by GAMS); when omitted it is
    discovered from the control file so the GAMS shared libraries load from the
    right place via ``gmoCreateD`` / ``gevCreateD``.

    Returns a process exit code (0 on success).
    """
    try:
        import gams.core.gev as gev
        import gams.core.gmo as gmo
    except Exception as exc:  # gamsapi[core] not installed
        sys.stderr.write(
            "POUNCE GAMS link requires the GAMS expert-level Python API.\n"
            "Install it from your GAMS system: pip install gamsapi[core]\n"
            f"({exc})\n"
        )
        return 1

    if not (sysdir and os.path.isdir(sysdir)):
        sysdir = _gams_sysdir(control_file)

    gev_h = gev.new_gevHandle_tp()
    rc, msg = gev.gevCreateD(gev_h, sysdir, 256) if sysdir else gev.gevCreate(gev_h, 256)
    if not rc:
        sys.stderr.write(f"gevCreate failed: {msg}\n")
        return 1
    gev.gevInitEnvironmentLegacy(gev_h, control_file)

    gmo_h = gmo.new_gmoHandle_tp()
    rc, msg = gmo.gmoCreateD(gmo_h, sysdir, 256) if sysdir else gmo.gmoCreate(gmo_h, 256)
    if not rc:
        sys.stderr.write(f"gmoCreate failed: {msg}\n")
        return 1
    gmo.gmoRegisterEnvironment(gmo_h, gev.gevHandleToPtr(gev_h))
    gmo.gmoLoadDataLegacy(gmo_h)

    # Objective handled as a function (objective variable substituted out), with
    # 0-based column/row indexing -- matches gams_pounce.c.
    gmo.gmoObjStyleSet(gmo_h, gmo.gmoObjType_Fun)
    gmo.gmoObjReformSet(gmo_h, 1)
    gmo.gmoIndexBaseSet(gmo_h, 0)

    gev.gevLogStat(gev_h, "")
    gev.gevLogStat(gev_h, "--- POUNCE: A Rust Interior-Point Optimizer (Python link)")
    gev.gevLogStat(gev_h, "")

    # Resource/iteration limits from the GAMS environment.
    iterlim = int(gev.gevGetIntOpt(gev_h, gev.gevIterLim))
    reslim = float(gev.gevGetDblOpt(gev_h, gev.gevResLim))
    max_iter = iterlim if 0 <= iterlim < 2_000_000_000 else None
    max_wall = reslim if 0 < reslim < 1e10 else None

    # Option file (pounce.opt / .op2 ...).
    options: dict = {}
    if gmo.gmoOptFile(gmo_h) > 0:
        optname = gmo.gmoNameOptFile(gmo_h)
        if isinstance(optname, (list, tuple)):  # some bindings return [rc, name]
            optname = optname[-1]
        gev.gevLogStat(gev_h, f"  Reading option file {optname}")
        options, _link_opts = parse_option_file(str(optname))

    view = _GmoAdapter(gmo_h, gmo)
    _gp, x, info = solve_view(
        view, options=options, max_iter=max_iter, max_wall_time=max_wall
    )

    gev.gevLogStat(gev_h, f"POUNCE status: {info.get('status_msg')}")

    # GAMS requires the solver to open/finalize a status file via GEV; otherwise
    # it reports solveStat=13 regardless of the GMO solution we unload.
    gev.gevStatCon(gev_h)
    _write_solution(gmo_h, gmo, view, x, info)
    gev.gevStatEOF(gev_h)
    return 0


def _write_solution(gmo_h, gmo, view, x, info) -> None:  # pragma: no cover - needs GAMS
    """Write primal/dual solution and GAMS model/solve status back into GMO."""
    status_msg = str(info.get("status_msg", ""))
    obj_sign = -1.0 if view.maximize() else 1.0
    model_stat, solve_stat = status_to_gams(status_msg)
    gmo.gmoModelStatSet(gmo_h, model_stat)
    gmo.gmoSolveStatSet(gmo_h, solve_stat)

    if status_msg in _STATUS_HAS_SOLUTION and x is not None:
        # Objective in GAMS convention (undo our sign flip for max).
        obj_val = float(info.get("obj_val", 0.0))
        gmo.gmoSetHeadnTail(gmo_h, gmo.gmoHobjval, obj_sign * obj_val)

    iters = info.get("iter_count")
    if iters is not None:
        gmo.gmoSetHeadnTail(gmo_h, gmo.gmoHiterused, float(iters))

    m = int(view.num_cons())
    # Constraint multipliers: POUNCE lambda -> GAMS pi (negate).
    mult_g = info.get("mult_g")
    pi = None
    if m and mult_g is not None:
        pi = _to_double_array(gmo, -np.asarray(mult_g, dtype=float))

    if x is not None:
        gmo.gmoSetSolution2(gmo_h, _to_double_array(gmo, x), pi)

    # Variable marginals: z_L - z_U, negated for max (matches gams_pounce.c).
    mult_xl = info.get("mult_x_L")
    mult_xu = info.get("mult_x_U")
    if mult_xl is not None and mult_xu is not None:
        var_marg = np.asarray(mult_xl, dtype=float) - np.asarray(mult_xu, dtype=float)
        if obj_sign < 0.0:
            var_marg = -var_marg
        gmo.gmoSetVarM(gmo_h, _to_double_array(gmo, var_marg))

    gmo.gmoUnloadSolutionLegacy(gmo_h)


class _GmoAdapter:  # pragma: no cover - thin wrapper over gamsapi calls
    """Adapt a gamsapi GMO handle to the :class:`GmoView` protocol.

    Isolates the version-specific GMO calling convention.  Index base is set to
    0 by the caller, so column/row indices are 0-based.  GMO's vector getters
    fill preallocated SWIG ``intArray`` / ``doubleArray`` buffers; the dense
    evaluators (``gmoEvalGradObj`` / ``gmoEvalGrad``) likewise write into a
    preallocated dense buffer, from which we pull the structural nonzeros.

    The GMO call sequence and sign conventions are cross-checked against the
    working C link in ``gams/gams_pounce.c``.
    """

    def __init__(self, gmo_h, gmo):
        self._h = gmo_h
        self._gmo = gmo
        self._n = int(gmo.gmoN(gmo_h))
        self._m = int(gmo.gmoM(gmo_h))
        self._nz = int(gmo.gmoNZ(gmo_h))

        # Hessian-of-Lagrangian availability. gmoHessLoad(h, maxJacRatio,
        # do2dir_req, doHess_req) requests the structures; we ask for the
        # Lagrangian Hessian only (do2dir=0, doHess=1) and call it exactly ONCE
        # -- re-loading clears the structure that gmoHessLagStruct / -Value
        # then read.  Availability is gmoHessLagNz() > 0.
        gmo.gmoHessLoad(gmo_h, 0.0, 0, 1)
        self._nnz_hess = int(gmo.gmoHessLagNz(gmo_h))
        self._has_hess = self._nnz_hess > 0

        # Jacobian structure (CSR from GMO) -> 0-based COO, cached.
        self._jac_rows, self._jac_cols, self._row_has_nl, self._jac_lin = (
            self._load_jacobian_structure()
        )
        # Scratch dense gradient buffer reused across evaluations.
        self._grad_buf = gmo.doubleArray(self._n)

    # --- dimensions / sense ----------------------------------------------
    def name(self) -> str:
        try:
            return str(self._gmo.gmoNameModel(self._h))
        except Exception:
            return "gams_model"

    def num_vars(self) -> int:
        return self._n

    def num_cons(self) -> int:
        return self._m

    def maximize(self) -> bool:
        return bool(self._gmo.gmoSense(self._h) == self._gmo.gmoObj_Max)

    def has_hessian(self) -> bool:
        return self._has_hess

    # --- bounds / initial point ------------------------------------------
    def var_lower(self):
        return self._mapped_bounds(self._gmo.gmoGetVarLower, lower=True)

    def var_upper(self):
        return self._mapped_bounds(self._gmo.gmoGetVarUpper, lower=False)

    def var_init(self):
        buf = self._gmo.doubleArray(self._n)
        self._gmo.gmoGetVarL(self._h, buf)
        return [float(buf[j]) for j in range(self._n)]

    def con_lower(self):
        return [b[0] for b in self._con_bounds()]

    def con_upper(self):
        return [b[1] for b in self._con_bounds()]

    def _con_bounds(self):
        gmo = self._gmo
        out = []
        for i in range(self._m):
            etyp = gmo.gmoGetEquTypeOne(self._h, i)
            rhs = float(gmo.gmoGetRhsOne(self._h, i))
            if etyp == gmo.gmoequ_E:
                out.append((rhs, rhs))
            elif etyp == gmo.gmoequ_G:
                out.append((rhs, POUNCE_INF))
            elif etyp == gmo.gmoequ_L:
                out.append((-POUNCE_INF, rhs))
            else:  # gmoequ_N (free) or unsupported -> free row
                out.append((-POUNCE_INF, POUNCE_INF))
        return out

    def _mapped_bounds(self, getter, lower: bool):
        gmo = self._gmo
        buf = gmo.doubleArray(self._n)
        getter(self._h, buf)
        pinf = float(gmo.gmoPinf(self._h))
        minf = float(gmo.gmoMinf(self._h))
        out = []
        for j in range(self._n):
            v = float(buf[j])
            if lower and v <= minf:
                v = -POUNCE_INF
            elif (not lower) and v >= pinf:
                v = POUNCE_INF
            out.append(v)
        return out

    # --- structure -------------------------------------------------------
    def jac_structure(self):
        return self._jac_rows, self._jac_cols

    def hess_structure(self):
        gmo = self._gmo
        irow = gmo.intArray(self._nnz_hess)
        jcol = gmo.intArray(self._nnz_hess)
        gmo.gmoHessLagStruct(self._h, irow, jcol)
        rows = [int(irow[k]) for k in range(self._nnz_hess)]
        cols = [int(jcol[k]) for k in range(self._nnz_hess)]
        return rows, cols

    def _load_jacobian_structure(self):
        """Read the CSR Jacobian structure into 0-based COO arrays.

        Also caches, per nonzero, whether it is linear (constant coefficient)
        and the coefficient value, so :meth:`eval_jac` can copy linear-row
        values without an evaluator call -- matching the C link.
        """
        gmo, m, nz = self._gmo, self._m, self._nz
        if m == 0 or nz == 0:
            return [], [], [], []
        rowstart = gmo.intArray(m + 1)
        colidx = gmo.intArray(nz)
        values = gmo.doubleArray(nz)
        nlflag = gmo.intArray(nz)
        gmo.gmoGetMatrixRow(self._h, rowstart, colidx, values, nlflag)

        rows, cols, lin = [], [], []
        row_has_nl = [False] * m
        for i in range(m):
            for k in range(int(rowstart[i]), int(rowstart[i + 1])):
                rows.append(i)
                cols.append(int(colidx[k]))
                if int(nlflag[k]) != 0:
                    row_has_nl[i] = True
                    lin.append(None)  # filled by evaluator
                else:
                    lin.append(float(values[k]))
        return rows, cols, row_has_nl, lin

    # --- numerical evaluators (native sense) -----------------------------
    def eval_obj(self, x):
        ret = self._gmo.gmoEvalFuncObj(self._h, _to_double_array(self._gmo, x))
        # bindings return either fval or [rc, fval, numerr]; take the value.
        if isinstance(ret, (list, tuple)):
            return float(ret[1])
        return float(ret)

    def eval_grad_obj(self, x):
        gmo = self._gmo
        grad = gmo.doubleArray(self._n)
        gmo.gmoEvalGradObj(self._h, _to_double_array(gmo, x), grad)
        return [float(grad[j]) for j in range(self._n)]

    def eval_cons(self, x):
        gmo = self._gmo
        xa = _to_double_array(gmo, x)
        out = []
        for i in range(self._m):
            ret = gmo.gmoEvalFunc(self._h, i, xa)
            out.append(float(ret[1] if isinstance(ret, (list, tuple)) else ret))
        return out

    def eval_jac(self, x):
        gmo = self._gmo
        xa = _to_double_array(gmo, x)
        vals = [0.0] * len(self._jac_rows)
        # Walk per row, reusing the dense grad buffer for nonlinear rows.
        k = 0
        nrows = len(self._jac_rows)
        while k < nrows:
            i = self._jac_rows[k]
            # span of row i in the COO arrays (rows are contiguous & sorted)
            start = k
            while k < nrows and self._jac_rows[k] == i:
                k += 1
            end = k
            if not self._row_has_nl[i]:
                for t in range(start, end):
                    vals[t] = self._jac_lin[t]
                continue
            gmo.gmoEvalGrad(self._h, i, xa, self._grad_buf)
            for t in range(start, end):
                vals[t] = float(self._grad_buf[self._jac_cols[t]])
        return vals

    def hess_lag_value(self, x, lam, obj_weight, con_weight):
        gmo = self._gmo
        vals = gmo.doubleArray(self._nnz_hess)
        gmo.gmoHessLagValue(
            self._h,
            _to_double_array(gmo, x),
            _to_double_array(gmo, lam),
            vals,
            float(obj_weight),
            float(con_weight),
        )
        return [float(vals[k]) for k in range(self._nnz_hess)]


def _to_double_array(gmo, x):  # pragma: no cover - needs gamsapi
    """Copy a numpy/list vector into a fresh SWIG ``doubleArray``."""
    xa = np.asarray(x, dtype=float)
    buf = gmo.doubleArray(len(xa))
    for j in range(len(xa)):
        buf[j] = float(xa[j])
    return buf


def _gams_sysdir(control_file: str) -> str | None:
    """Best-effort discovery of the GAMS system directory from a control file."""
    try:
        with open(control_file) as fh:
            lines = [ln.strip() for ln in fh]
    except OSError:
        return None
    for ln in lines:
        for tok in ln.split():
            if os.path.basename(tok) in ("gmscmpun.txt", "gmscmpdef.txt") and os.path.isfile(tok):
                return os.path.dirname(tok)
    for ln in lines:
        if os.path.isdir(ln) and os.path.isfile(os.path.join(ln, "gmscmpun.txt")):
            return ln
    return None


def _parse_gams_args(args: list[str]) -> tuple[str | None, str | None]:
    """Resolve ``(control_file, sysdir)`` from a solver script's arguments.

    GAMS's classic script-solver interface passes six arguments -- ``<scrdir>
    <workdir> <prmfile> <cntrfile> <sysdir> <solvername>`` -- so we resolve the
    control file and system directory by *content* (the ``gamscntr*`` file vs.
    the directory containing ``gmscmpun.txt``).  This is robust to convention
    drift and also works when invoked directly with a single control-file path.
    """
    files = [tok for tok in args if os.path.isfile(tok)]
    control_file = next(
        (tok for tok in files if os.path.basename(tok).startswith("gamscntr")), None
    )
    if control_file is None:
        control_file = next((tok for tok in files if tok.endswith(".dat")), None)
    if control_file is None:
        control_file = files[0] if files else None

    sysdir = next(
        (
            tok
            for tok in args
            if os.path.isdir(tok) and os.path.isfile(os.path.join(tok, "gmscmpun.txt"))
        ),
        None,
    )
    return control_file, sysdir


def main(argv: list[str] | None = None) -> int:
    """Console entry point: the GAMS solver link.

    Accepts either a single control-file path (direct invocation) or the full
    GAMS script-solver argument list.
    """
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        sys.stderr.write("usage: pounce-gams-link <gams-control-file>\n")
        return 0 if args else 1

    control_file, sysdir = _parse_gams_args(args)
    if control_file is None:
        sys.stderr.write(f"pounce-gams-link: no control file found in arguments: {args}\n")
        return 1
    return solve_from_control_file(control_file, sysdir)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
