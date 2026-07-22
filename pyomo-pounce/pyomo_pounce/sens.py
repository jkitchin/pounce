"""Declared-parameter sensitivity for Pyomo models solved with POUNCE.

Declare which parameters matter when you build the model -- no perturbed
values required -- then solve normally. The converged KKT factorization is
kept, and every sensitivity is a cheap backsolve afterwards:

    import pyomo_pounce
    from pyomo_pounce import declare_sens_param, gradient, estimate

    m.p = pyo.Param(initialize=2.0, mutable=True)
    declare_sens_param(m.p)

    pyo.SolverFactory("pounce").solve(m)     # normal solve

    gradient(m.x, wrt=m.p)                   # dx*/dp (float)
    gradient(m.x, wrt=m.p2)                  # containers -> Gradient object
    gradient(m.c, wrt=m.p)                   # d(multiplier of c)/dp
    estimate(m, [(m.p, 2.5)])                # perturbed-solution estimate,
                                             # clamped to bounds, warns on clamp

Estimation models use the other two declarations: flag the FITTED
variables and the residual container, solve once, and ask for the
covariance with no further information:

    declare_fitted(m.A); declare_fitted(m.k)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)     # one ordinary solve
    covariance(m)                            # std errors, correlations,
                                             # identifiability diagnostics

Mechanics: declared Params become pinned variables on a clone
(pyomo.contrib.sensitivity_toolbox does the expression surgery), the clone
is written to .nl and evaluated in-process via pounce.read_nl, and the
pounce.Solver session's parametric_step answers gradient()/estimate()
queries from the stored factorization -- the sIPOPT computation, with no
suffixes and no upfront perturbation values.
"""
import codecs
import os
import shutil
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.sensitivity_toolbox.sens import SensitivityInterface
from pyomo.core.base.constraint import Constraint
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition

_REG = "_pounce_sens"


# ── declaration ───────────────────────────────────────────────────────────────

class _Registry:
    """Per-model registry of declared statistical roles. Deepcopy-aware so
    model.clone() (and the sensitivity surgery's own clone) works cleanly:
    declared components follow the clone through the memo, while the
    session -- which holds solver handles tied to one converged
    factorization -- is deliberately not copied (a clone has no solve of
    its own yet)."""

    def __init__(self):
        self.params = []          # pinned inputs: gradient()/estimate()
        self.fitted = []       # free fitted variables: covariance()
        self.residuals = []       # (container, group) pairs: sigma^2
        self.session = None

    def __deepcopy__(self, memo):
        import copy
        new = _Registry()
        memo[id(self)] = new
        new.params = [copy.deepcopy(p, memo) for p in self.params]
        new.fitted = [copy.deepcopy(p, memo) for p in self.fitted]
        new.residuals = [(copy.deepcopy(r, memo), g)
                         for r, g in self.residuals]
        return new


def _registry(model):
    return model.__dict__.setdefault(_REG, _Registry())


def declare_sens_param(*params):
    """Flag one or more mutable Params (or fixed Vars), scalar or indexed,
    as FIXED INPUTS for sensitivity: after a solve, gradient() and
    estimate() answer d(solution)/d(param) questions. No perturbed value
    is required, or accepted."""
    for param in params:
        _registry(param.model()).params.append(param)


def declare_fitted(*variables):
    """Flag one or more FREE Vars (scalar or indexed) as fitted
    parameters of a least-squares problem: after one ordinary solve,
    covariance() reports their asymptotic uncertainty. The variables stay
    free in the solve; do not fix them."""
    for var in variables:
        _registry(var.model()).fitted.append(var)


def declare_residual(*containers, group=None):
    """Flag one or more indexed Vars holding the fit residuals, one member
    per data point. covariance() derives the residual count and the SSR
    from them, so no data counts need to be passed. `group` is an
    arbitrary user string partitioning residuals into noise groups and
    applies to every container in the call: containers sharing a group
    (or all ungrouped containers together) pool into one estimated noise
    variance; distinct groups get their own, and the covariance switches
    to the heteroscedastic sandwich form."""
    for container in containers:
        _registry(container.model()).residuals.append((container, group))


def has_declarations(model):
    reg = getattr(model, "__dict__", {}).get(_REG)
    return bool(reg and (reg.params or reg.fitted or reg.residuals))


# ── the read_nl -> callback-Problem bridge ────────────────────────────────────

class _NlBridge:
    """cyipopt-style callback object backed by pounce.read_nl evaluators."""

    def __init__(self, nl):
        self._nl = nl

    def objective(self, x):
        return self._nl.objective(x)

    def gradient(self, x):
        return self._nl.gradient(x)

    def constraints(self, x):
        return self._nl.constraints(x)

    def jacobianstructure(self):
        return self._nl.jacobian_structure()

    def jacobian(self, x):
        return self._nl.jacobian(x)

    def hessianstructure(self):
        return self._nl.hessian_structure()

    def hessian(self, x, lam, obj_factor):
        return self._nl.hessian(x, lam, obj_factor)


# ── session ───────────────────────────────────────────────────────────────────

class _Session:
    def __init__(self, model, nl, solver, var_names, con_names, pins,
                 con_alias):
        self.model = model            # original model
        self.nl = nl
        self.solver = solver
        self.var_names = var_names    # .col order = x-vector order
        self.con_names = con_names    # .row order = g-vector order
        self.pins = pins              # ComponentMap: param data -> pin row
        self.con_alias = con_alias    # original con name -> clone row name
        self.base_x = None
        self._columns = {}            # pin row -> full KKT-space column

    def orig_var(self, name):
        return self.model.find_component(name)

    def column(self, pin_idx):
        """Full KKT-space derivative column for a unit perturbation."""
        if pin_idx not in self._columns:
            self._columns[pin_idx] = np.asarray(
                self.solver.parametric_step_full([pin_idx], [1.0]))
        return self._columns[pin_idx]

    def var_entry(self, name):
        return self.var_names.index(name)

    def mult_entry(self, con_name):
        # the sensitivity surgery replaces user constraints with copies on
        # its data block; translate the original name to the clone's row
        con_name = self.con_alias.get(con_name, con_name)
        g = self.con_names.index(con_name)
        row = self.solver.multiplier_rows([g])[0]
        if row is None:
            raise ValueError(
                f"{con_name}: multiplier sensitivities are only available "
                "for equality constraints")
        return row


def _iter_data(comp):
    if comp.is_indexed():
        for idx in comp:
            yield comp[idx]
    else:
        yield comp


# Engine status -> (termination condition, solver status), mirroring the
# semantics Pyomo's .sol reader gives the ordinary path via the AMPL
# exit-code ranges (optimal / infeasible / unbounded / limit / error).
_STATUS_RESULT = {
    "Solve_Succeeded":
        (TerminationCondition.optimal, SolverStatus.ok),
    "Solved_To_Acceptable_Level":
        (TerminationCondition.optimal, SolverStatus.warning),
    "Feasible_Point_Found":
        (TerminationCondition.feasible, SolverStatus.warning),
    "Infeasible_Problem_Detected":
        (TerminationCondition.infeasible, SolverStatus.warning),
    "Diverging_Iterates":
        (TerminationCondition.unbounded, SolverStatus.warning),
    "Maximum_Iterations_Exceeded":
        (TerminationCondition.maxIterations, SolverStatus.warning),
    "Maximum_CpuTime_Exceeded":
        (TerminationCondition.maxTimeLimit, SolverStatus.warning),
    "Maximum_WallTime_Exceeded":
        (TerminationCondition.maxTimeLimit, SolverStatus.warning),
    "User_Requested_Stop":
        (TerminationCondition.userInterrupt, SolverStatus.aborted),
}


def _stream_solve(solver, x0):
    """Run ``solver.solve(x0)`` with the engine's log streamed to sys.stdout.

    The engine (and ``pounce.print_banner``) write straight to the process
    stdout, fd 1, bypassing ``sys.stdout``: visible in a terminal, invisible
    in Jupyter and under ``contextlib.redirect_stdout``. When ``sys.stdout``
    already is fd 1 the log streams itself. Otherwise redirect fd 1 to a temp
    file, run the solve on a worker thread, and tail the file to
    ``sys.stdout`` so notebooks (and redirected streams) see the banner,
    problem statistics, iteration table, and summary live, not as one block
    at the end. ipykernel's OutStream coalesces on its own ~30-200 ms timer,
    so updates arrive in bursts.

    Returns ``(result, solve_secs)`` with ``solve_secs`` measured strictly
    around the solve, excluding banner/stream/decode overhead.
    """
    import pounce
    banner = getattr(pounce, "print_banner", lambda: None)

    def _timed():
        t0 = time.perf_counter()
        out = solver.solve(x0)
        return out, time.perf_counter() - t0

    try:
        live = sys.stdout.fileno() == 1
    except Exception:                                     # noqa: BLE001
        live = False
    if live:
        banner()
        return _timed()

    # Tail a regular temp file, never an os.pipe: a stalled pipe reader would
    # block the solver forever (its ~64 KB kernel buffer), whereas a file
    # never applies write backpressure. A separate read handle with its own
    # tracked offset keeps tailing from disturbing the engine's write position
    # (a dup'd fd would share the offset).
    # `saved` is the only resource acquired before the try; the temp file and
    # its reader open inside it, so a failure there still reaches the cleanup
    # (the finally guards each handle with a None check).
    saved = os.dup(1)
    fd_w = path = reader = None
    try:
        fd_w, path = tempfile.mkstemp(prefix="pounce_tee_")
        reader = open(path, "rb")
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        pos = 0
        stop = threading.Event()

        def _drain(final=False):
            nonlocal pos
            reader.seek(pos)
            chunk = reader.read()
            pos = reader.tell()
            text = dec.decode(chunk, final)
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()

        def _tail():
            # The solve runs on THIS (main) thread -- pounce.Solver is a pyo3
            # unsendable object and would panic if moved to a worker. It
            # releases the GIL during the solve, so it is the tailing that
            # lives on the worker: read new bytes as the engine writes them,
            # until the solve finishes and signals stop.
            while not stop.is_set():
                _drain()
                time.sleep(0.05)

        os.dup2(fd_w, 1)
        banner()
        tailer = threading.Thread(target=_tail, daemon=True)
        tailer.start()
        try:
            t0 = time.perf_counter()
            out = solver.solve(x0)
            solve_secs = time.perf_counter() - t0
        finally:
            # Stop the tailer and drain the tail even if the solve raised, so
            # its partial log still reaches the user before the error does.
            stop.set()
            tailer.join()
            _drain(final=True)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        if reader is not None:
            reader.close()
        if fd_w is not None:
            os.close(fd_w)
        if path is not None:
            try:
                os.remove(path)
            except OSError:
                pass
    return out, solve_secs


def sens_solve(model, tee=False, sens_params=None, fitted=None,
               residuals=None):
    """Solve `model` in-process with POUNCE and keep the KKT factorization
    for gradient()/estimate()/covariance(). Called automatically by
    SolverFactory('pounce').solve() when declarations are present; the
    keyword arguments are the explicit (call-time) form of the
    declarations and register the components exactly as the declare_*
    functions do. Returns a Pyomo SolverResults, like an ordinary solve."""
    import pounce

    reg = _registry(model)

    # Effective declarations for THIS solve: the persistent declared
    # components plus any explicit (call-time) ones. The explicit form is
    # deliberately solve-local -- it is NOT written back into reg -- so that
    # repeated solves of one model (the NMPC use case) do not accumulate
    # duplicate components and silently corrupt the covariance/pins.
    eff_params = list(reg.params) + list(sens_params or [])
    eff_fitted = list(reg.fitted) + list(fitted or [])
    eff_residuals = list(reg.residuals) + [
        item if isinstance(item, tuple) else (item, None)
        for item in (residuals or [])]

    if eff_params:
        # pinned inputs need the sensitivity-toolbox surgery (on a clone)
        si = SensitivityInterface(model, clone_model=True)
        si.setup_sensitivity(eff_params)
        clone = si.model_instance
    else:
        # estimation-only: nothing to pin, solve the model as written
        si = None
        clone = model

    # The .nl/.col/.row files exist only to hand the model to read_nl;
    # everything needed later (evaluators, bounds, names) lives in memory,
    # so the temp dir is removed as soon as they are parsed. Repeated
    # solves (the NMPC use case) must not accumulate temp dirs.
    tmp = tempfile.mkdtemp(prefix="pounce_sens_")
    try:
        nl_path = os.path.join(tmp, "model.nl")
        clone.write(nl_path, io_options={"symbolic_solver_labels": True})
        var_names = Path(nl_path[:-3] + ".col").read_text().splitlines()
        con_names = Path(nl_path[:-3] + ".row").read_text().splitlines()
        nl = pounce.read_nl(nl_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bridge = _NlBridge(nl)
    prob = pounce.Problem(nl.n, nl.m, bridge,
                          lb=nl.x_l, ub=nl.x_u, cl=nl.g_l, cu=nl.g_u)
    if not tee:
        # Pyomo convention is silence unless tee=True; print_level 0 makes
        # the engine emit nothing at all.
        prob.add_option("print_level", 0)
    solver = pounce.Solver(prob)
    if tee:
        # At the default print_level the engine emits its own banner (via
        # print_banner), problem statistics, iteration table, and summary;
        # _stream_solve tails them to sys.stdout live and times the solve
        # alone (excluding banner/stream overhead).
        (x, info), solve_secs = _stream_solve(solver, np.asarray(nl.x0))
    else:
        t_solve = time.perf_counter()
        x, info = solver.solve(np.asarray(nl.x0))
        solve_secs = time.perf_counter() - t_solve

    status_msg = str(info.get("status_msg", ""))
    tc, ss = _STATUS_RESULT.get(
        status_msg, (TerminationCondition.error, SolverStatus.error))

    # Return a Pyomo SolverResults indistinguishable from an ordinary
    # solve's: same fields (counts, time, Id/Error rc, emptied Solution
    # block), same message spelling, same exit-status mapping, same
    # noncommittal bounds/sense.
    def build_results():
        results = SolverResults()
        results.solver.name = "pounce (in-process sensitivity session)"
        results.solver.status = ss
        results.solver.termination_condition = tc
        # the binary's .sol message spells the status without underscores
        results.solver.message = (
            f"POUNCE {pounce.__version__}: {status_msg.replace('_', '')}")
        results.solver.id = 0
        results.solver.error_rc = 0
        # solve_secs is the solve alone (the tee stream/decode is excluded)
        results.solver.time = solve_secs
        results.problem.number_of_objectives = 1
        results.problem.number_of_constraints = int(nl.m)
        results.problem.number_of_variables = int(nl.n)
        # objective bounds, like the .sol path: both set to the final value
        obj_val = info.get("obj_val")
        if obj_val is not None:
            results.problem.upper_bound = float(obj_val)
            results.problem.lower_bound = float(obj_val)
        # the ordinary path's repr carries an emptied Solution block
        # (the parsed solution is loaded into the model, then cleared)
        results.solution.add()
        results.solution.clear()
        return results

    if not solver.converged:
        # Report the outcome through the results object (infeasible /
        # maxIterations / error) and load the final iterate, but drop
        # any session: a failed re-solve must not leave a prior
        # converged solve's factorization live, or
        # gradient()/estimate()/covariance() would silently answer from
        # the stale solve. With the session cleared they raise their
        # usual "no sensitivity session" error. Note the Feasible_Point_Found
        # asymmetry: the engine's on_converged callback fires only for
        # Solve_Succeeded / Solved_To_Acceptable_Level, so a feasible-point
        # solve reports termination_condition=feasible yet has converged=False
        # and lands here -- no KKT factorization is retained, so its session
        # is dropped even though the status is not a hard failure.
        reg.session = None
        for name, val in zip(var_names, np.asarray(x)):
            ov = model.find_component(name)
            if ov is not None:
                ov.set_value(float(val), skip_validation=True)
        return build_results()

    pins = ComponentMap()
    con_alias = {}
    if si is not None:
        block = clone.component(SensitivityInterface.get_default_block_name())
        for i, (var, clone_param, list_idx, comp_idx) in enumerate(
                block._sens_data_list):
            con = block.paramConst[i + 1]
            orig_comp = eff_params[list_idx]
            orig_data = (orig_comp if not orig_comp.is_indexed()
                         else orig_comp[comp_idx])
            pins[orig_data] = con_names.index(con.name)
        # original-name -> clone-row-name aliases for replaced constraints
        if getattr(block, "_has_replaced_expressions", False):
            for new_comp, old_comp in block._replaced_map.items():
                for nd, od in zip(_iter_data(new_comp),
                                  _iter_data(old_comp)):
                    con_alias[od.name] = nd.name

    session = _Session(model, nl, solver, var_names, con_names, pins,
                       con_alias)
    session.base_x = np.asarray(x)

    # fitted parameters: their rows in the primal vector
    session.fit_rows = ComponentMap()
    for comp in eff_fitted:
        for vd in _iter_data(comp):
            session.fit_rows[vd] = var_names.index(vd.name)

    # residual groups: member rows per group key (None = the common pool)
    session.res_rows = {}
    for container, group in eff_residuals:
        rows = [var_names.index(rd.name) for rd in _iter_data(container)]
        session.res_rows.setdefault(group, []).extend(rows)

    reg.session = session

    # load the solution back onto the ORIGINAL model's variables (when the
    # solve ran on a clone; in the estimation-only path clone IS model and
    # this simply refreshes the same variables)
    for name, val in zip(var_names, session.base_x):
        ov = model.find_component(name)
        if ov is not None:
            ov.set_value(float(val), skip_validation=True)

    # consistency check: declared residuals should reproduce the objective
    if session.res_rows:
        ssr = sum(float(session.base_x[r]) ** 2
                  for rows in session.res_rows.values() for r in rows)
        obj_val = info.get("obj_val")
        if obj_val is not None and abs(ssr - float(obj_val)) > 1e-6 * max(
                1.0, abs(float(obj_val))):
            warnings.warn(
                "sens_solve: the declared residuals give SSR = "
                f"{ssr:.6g} but the objective value is {float(obj_val):.6g}."
                " covariance() assumes the objective is the plain sum of "
                "squares of the declared residuals; extra terms (weights, "
                "regularization) will make the noise-variance estimate "
                "wrong.")

    return build_results()


# ── queries ───────────────────────────────────────────────────────────────────

def _session_for(component):
    reg = component.model().__dict__.get(_REG)
    if reg is None or reg.session is None:
        raise RuntimeError(
            "no sensitivity session: declare_sens_param() then solve with "
            "SolverFactory('pounce') first")
    return reg.session


def _param_pin(session, param_data):
    if param_data not in session.pins:
        raise ValueError(f"{param_data.name} was not declared with "
                         "declare_sens_param before the solve")
    return session.pins[param_data]


class Gradient:
    """Derivatives d(target*)/d(param) for one or more targets/parameters.
    Targets are variables (primal sensitivities) or equality constraints
    (multiplier sensitivities).

    Access with g[target_data, param_data] (either order); when one side is
    a single component, g[data] works. to_dataframe() gives the full
    Jacobian (rows = targets, columns = parameters)."""

    def __init__(self, session, targets, params):
        self._session = session
        self._targets = list(targets)
        self._params = list(params)
        self._tset = set(id(t) for t in self._targets)
        self._pset = set(id(p) for p in self._params)

    def _entry(self, td):
        if td.ctype is Constraint:
            return self._session.mult_entry(td.name)
        return self._session.var_entry(td.name)

    @staticmethod
    def _convention_sign(td):
        """Sign taking a raw sensitivity row into the convention of the
        quantity the user reads off the model.

        Variable targets need no conversion: `var_entry` rows are
        derivatives of primal values, which is what `m.x.value` holds.

        Constraint targets do. `mult_entry` rows come from
        `parametric_step_full`'s y_c block, i.e. derivatives of POUNCE's
        internal Lagrange multiplier, whereas `m.dual[con]` holds the AMPL
        *marginal* `d obj / d b = -lambda` (gh #271). So d(dual)/d(param)
        is the negation of the raw row -- without this, `gradient(m.con,
        wrt=m.p)` disagrees in sign with a finite difference of
        `m.dual[m.con]` taken across a re-solve.
        """
        return -1.0 if td.ctype is Constraint else 1.0

    def _value(self, td, pd):
        col = self._session.column(_param_pin(self._session, pd))
        return self._convention_sign(td) * float(col[self._entry(td)])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            td, pd = key
            if id(td) in self._pset and id(pd) in self._tset:
                td, pd = pd, td            # accept either order
            return self._value(td, pd)
        if id(key) in self._tset and len(self._params) == 1:
            return self._value(key, self._params[0])
        if id(key) in self._pset and len(self._targets) == 1:
            return self._value(self._targets[0], key)
        raise KeyError(
            f"{getattr(key, 'name', key)}: give g[target, param], or a "
            "single component when the other dimension has exactly one "
            "member")

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(
            [[self._value(td, p) for p in self._params]
             for td in self._targets],
            index=[td.name for td in self._targets],
            columns=[p.name for p in self._params])


def gradient(target=None, *, wrt):
    """d(target*)/d(wrt).

    target: a Var (primal sensitivity) or an equality Constraint (its
    multiplier's sensitivity); data object or container; omit for all
    model variables. wrt: a declared Param (data or container).

    Scalar target and scalar wrt -> float. Anything else -> a Gradient
    object: g[target, param], or g.to_dataframe() for the full Jacobian."""
    session = _session_for(wrt)
    params = list(_iter_data(wrt))
    if target is None:
        targets = [v for v in (session.orig_var(nm)
                               for nm in session.var_names) if v is not None]
    else:
        targets = list(_iter_data(target))
    if target is not None and not target.is_indexed() and len(params) == 1:
        return Gradient(session, targets, params)._value(
            targets[0], params[0])
    return Gradient(session, targets, params)


def estimate(model, perturb, clamp=True):
    """First-order estimate of the solution at perturbed parameter values.

    perturb: pairs of (declared Param, new value) -- a list of tuples or a
    ComponentMap (plain dicts don't work: Pyomo components are unhashable).
    Returns a ComponentMap {original var data: estimated value}. Values are
    clamped to variable bounds (with a warning) unless clamp=False.
    """
    reg = model.__dict__.get(_REG)
    session = reg.session if reg else None
    if session is None:
        raise RuntimeError(
            "no sensitivity session: declare_sens_param() then solve with "
            "SolverFactory('pounce') first")

    items = perturb.items() if hasattr(perturb, "items") else perturb
    pin_idx, deltas = [], []
    for comp, newval in items:
        for pd in _iter_data(comp):
            nv = newval[pd.index()] if comp.is_indexed() and hasattr(
                newval, "__getitem__") else newval
            pin_idx.append(_param_pin(session, pd))
            deltas.append(float(nv) - pyo.value(pd))

    dx = np.asarray(session.solver.parametric_step(pin_idx, deltas))
    x_new = session.base_x + dx

    lo, hi = np.asarray(session.nl.x_l), np.asarray(session.nl.x_u)
    if clamp:
        # scale-aware tolerance: 1e-9 relative to the variable's magnitude
        tol = 1e-9 * np.maximum(1.0, np.abs(x_new))
        clipped = (x_new < lo - tol) | (x_new > hi + tol)
        if clipped.any():
            names = [session.var_names[i] for i in np.where(clipped)[0]]
            warnings.warn(
                "estimate: linear step leaves the variable bounds for "
                f"{names}; values were clamped and the active set likely "
                "changed, so the estimate is unreliable there.")
        x_new = np.clip(x_new, lo, hi)

    out = ComponentMap()
    for name, val in zip(session.var_names, x_new):
        ov = model.find_component(name)
        if ov is not None:
            out[ov] = float(val)
    return out


# ── parameter covariance ──────────────────────────────────────────────────────

class _ParamKeyed:
    """Lookup from a declared Param's data object to its row index.
    Keyed by id() because Pyomo components are unhashable."""

    def __init__(self, params):
        self._params = list(params)
        self._pos = {id(p): i for i, p in enumerate(self._params)}

    def _loc(self, pd):
        i = self._pos.get(id(pd))
        if i is None:
            raise KeyError(f"{getattr(pd, 'name', pd)}: not one of the "
                           "covariance parameters")
        return i


class _ParamVector(_ParamKeyed):
    """Vector keyed by param data: v[m.k1] -> float."""

    def __init__(self, params, values):
        super().__init__(params)
        self.values = np.asarray(values, dtype=float)

    def __getitem__(self, pd):
        return float(self.values[self._loc(pd)])


class _ParamMatrix(_ParamKeyed):
    """Symmetric matrix keyed by param data: M[m.k1, m.k2] (either
    order) or M[m.k1] for a diagonal entry."""

    def __init__(self, params, matrix):
        super().__init__(params)
        self.matrix = np.asarray(matrix, dtype=float)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = (self._loc(k) for k in key)
        else:
            i = j = self._loc(key)
        return float(self.matrix[i, j])


class Covariance(_ParamMatrix):
    """Asymptotic parameter covariance, from covariance().

    Keyed by the fitted variables' data objects (the free `Var`s
    flagged with `declare_fitted`, not Pyomo `Param`s) in `params`
    (declaration) order: cov[m.k1, m.k2] (either order),
    cov[m.k1] for a variance, cov.std_err[m.k1],
    cov.correlation[m.k1, m.k2]. `matrix` is the dense numpy array
    ordered like `params`; `sigma_sq` is the residual variance that was
    used. eigen() supports identifiability diagnosis."""

    def __init__(self, params, matrix, sigma_sq):
        super().__init__(params, matrix)
        self.params = self._params
        self.sigma_sq = sigma_sq          # float, or {group: float}
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(np.diag(self.matrix))
            corr = self.matrix / np.outer(se, se)
        # entries whose scale is undefined (a projected bound-active
        # parameter has exactly zero variance) are reported as 0
        corr[~np.isfinite(corr)] = 0.0
        self.std_err = _ParamVector(self.params, se)
        self.correlation = _ParamMatrix(self.params, corr)

    def eigen(self):
        """(eigenvalues, eigenvectors) of the covariance matrix,
        eigenvalues ascending, eigenvectors[:, i] in `params` order.
        An eigenvalue much larger than the rest flags a poorly
        identified direction: its eigenvector gives the parameter
        combination the data cannot pin down."""
        return np.linalg.eigh(self.matrix)


def covariance(model, sigma_sq=None, n_data=None, hessian="lagrangian"):
    """Asymptotic covariance of the fitted parameters of a
    least-squares problem, from ONE ordinary solve.

    Workflow: declare the fitted variables with declare_fitted (they
    stay free), optionally declare the residual container(s) with
    declare_residual, solve with SolverFactory('pounce'), then call
    covariance(model) with no further information.

    ASSUMES the model objective is the plain sum of squared residuals.
    The parameter block of the inverse KKT matrix, obtained by one
    backsolve per parameter against the held factorization, equals the
    inverse reduced Hessian of the eliminated problem, inv(d2f*/dp2);
    for f = SSR the asymptotic covariance is then

        cov = 2 * sigma_sq * (K^-1)_pp

    The factor 2 belongs to the unscaled sum of squares; it is verified
    against the analytical linear-regression covariance
    sigma^2 * inv(X^T X) in tests/test_covariance.py.

    The noise variance sigma_sq comes from, in order of precedence:
    sigma_sq= (known measurement variance; scalar, or {group: value}
    when residual groups are declared); declared residuals (estimated
    per pooled or labeled group as SSR_g / (n_g - n_params)); or the
    n_data= fallback (count of data points, with SSR taken from the
    objective value on trust). With multiple labeled groups the
    heteroscedastic sandwich covariance is reported.

    hessian= selects the information matrix. "lagrangian" (the default)
    inverts the exact reduced Hessian of the Lagrangian from the held
    factorization: the observed-information form, the same object
    sIPOPT or k_aug would factor. "gauss-newton" rebuilds
    the expected-information form from the residual Jacobian, recovered
    from the same backsolves at no extra solve (requires declared
    residuals). They agree for linear models; for nonlinear fits
    Gauss-Newton drops the residual-curvature term, matches the scipy /
    ``pounce.curve_fit`` convention, and is structurally positive
    semidefinite, which makes it the safe choice when the covariance
    must stay PSD, e.g. feeding an arrival-cost update in moving
    horizon estimation.

    Returns a Covariance object keyed by the declared variables'
    data objects: cov[m.A, m.k], cov.std_err[m.A],
    cov.correlation[m.A, m.k], cov.matrix, cov.sigma_sq (float or
    per-group dict), cov.eigen().

    Same scale-and-invert-the-reduced-Hessian recipe as
    ``pounce.curve_fit``, with one difference for NONLINEAR models: this
    feeds the exact Lagrangian Hessian (via the .nl bridge) and so reports
    the OBSERVED-information covariance (the full reduced Hessian),
    whereas ``curve_fit`` factors the Gauss-Newton Hessian and reports
    ``2 sigma^2 (J^T J)^-1`` (the expected-information / scipy convention,
    always positive semidefinite). The two agree for linear models and in
    the small-residual limit and differ by O(residual x curvature)
    otherwise. Gauss-Newton cannot produce a negative variance; the full
    Hessian can go indefinite, which is what the negative-diagonal warning
    below signals -- pass hessian="gauss-newton" then, or whenever
    scipy-matching numbers are wanted. Use ``curve_fit`` for the
    callable-model-plus-data surface
    (starting point, robust losses, confidence intervals, prediction
    bands, active-bound projection); use this for a model already written
    in Pyomo. Like ``curve_fit``, a fitted parameter detected on its
    bound at the optimum is projected out: the covariance is computed in
    the remaining free directions (the covariance CONDITIONAL on the
    active bound) and the pinned parameter reports zero variance, with
    correlation entries involving it reported as 0. A warning still
    fires, because boundary asymptotics are nonstandard whichever number
    is reported. Only variable bounds ON the fitted parameters are
    detected; other active constraints involving them are not.
    """
    if hessian not in ("lagrangian", "gauss-newton"):
        raise ValueError(
            "covariance: hessian must be 'lagrangian' or 'gauss-newton', "
            f"got {hessian!r}")
    reg = model.__dict__.get(_REG)
    session = reg.session if reg else None
    if session is None:
        raise RuntimeError(
            "no sensitivity session: declare_fitted() (and optionally "
            "declare_residual()) then solve with SolverFactory('pounce') "
            "first")
    params = list(session.fit_rows.keys())
    n_params = len(params)
    if n_params == 0:
        raise RuntimeError(
            "covariance: no fitted parameters were declared; flag the "
            "fitted variables with declare_fitted() before the solve")

    # ── guardrails ────────────────────────────────────────────────────────
    pert = np.asarray(session.solver.kkt_perturbations)
    if pert.any():
        warnings.warn(
            "covariance: the held KKT factor carries inertia-correction "
            f"perturbations {pert.tolist()}, so the covariance is "
            "regularized rather than exact. Linearly dependent (structurally"
            " unidentifiable) parameters are the usual cause.")
    lo, hi = np.asarray(session.nl.x_l), np.asarray(session.nl.x_u)
    active = []
    for i, p in enumerate(params):
        r = session.fit_rows[p]
        xv = float(session.base_x[r])
        tol = 1e-6 * (1.0 + abs(xv))
        if xv - lo[r] < tol or hi[r] - xv < tol:
            active.append(i)
            warnings.warn(
                f"covariance: fitted parameter {p.name} sits on its "
                "bound at the optimum; its direction is projected out "
                "(zero variance, conditional on the active bound) and "
                "the boundary asymptotics are nonstandard.")

    # ── parameter block of the inverse KKT matrix ─────────────────────────
    dim = session.solver.kkt_dim
    rows = [session.fit_rows[p] for p in params]
    zcols = []
    for r in rows:
        e = np.zeros(dim)
        e[r] = 1.0
        zcols.append(np.asarray(session.solver.kkt_solve(e)))
    M = np.array([[zcols[j][rows[i]] for j in range(n_params)]
                  for i in range(n_params)])
    M = 0.5 * (M + M.T)

    # ── noise variance per group ──────────────────────────────────────────
    groups = dict(session.res_rows)
    if hessian == "gauss-newton" and not groups:
        raise ValueError(
            "covariance: hessian='gauss-newton' needs declared residuals "
            "(declare_residual()); the residual Jacobian is recovered from "
            "their rows. Without residual variables only the "
            "hessian='lagrangian' default is available.")
    if n_data is not None and (sigma_sq is not None or groups):
        warnings.warn(
            "covariance: n_data is ignored because a higher-precedence noise "
            "source was given (sigma_sq, or the declared residuals).")
    if sigma_sq is not None:
        if isinstance(sigma_sq, dict):
            named = [g for g in groups if g is not None]
            if not named:
                raise ValueError(
                    "covariance: sigma_sq was given as a per-group dict but "
                    "no named residual groups were declared; pass a scalar "
                    "sigma_sq, or declare grouped residuals with "
                    "declare_residual(..., group=...)")
            missing = [g for g in groups if g not in sigma_sq]
            if missing:
                raise ValueError(
                    "covariance: sigma_sq is missing an entry for residual "
                    f"group(s) {sorted(map(repr, missing))}")
            group_sigma = {g: float(sigma_sq[g]) for g in groups}
        else:
            group_sigma = {g: float(sigma_sq) for g in (groups or {None: []})}
    elif groups:
        group_sigma = {}
        for g, rws in groups.items():
            n_g = len(rws)
            if n_g <= n_params:
                raise ValueError(
                    f"covariance: residual group {g!r} has {n_g} members, "
                    f"not more than the {n_params} fitted parameters; "
                    "cannot estimate its noise variance")
            ssr_g = float(np.sum(session.base_x[rws] ** 2))
            group_sigma[g] = ssr_g / (n_g - n_params)
    elif n_data is not None:
        if n_data <= n_params:
            raise ValueError(
                f"covariance: n_data ({n_data}) must exceed the number of "
                f"fitted parameters ({n_params})")
        ssr = pyo.value(
            next(model.component_data_objects(pyo.Objective, active=True)))
        group_sigma = {None: ssr / (n_data - n_params)}
    else:
        raise ValueError(
            "covariance: the noise variance is unknown; declare the "
            "residual container(s) with declare_residual(), or pass "
            "sigma_sq= (known variance), or pass n_data= (data count, "
            "with the SSR taken from the objective value)")

    # ── assemble ──────────────────────────────────────────────────────────
    # Pooled covariance when there is one group or all group variances are
    # equal to relative tolerance; otherwise the heteroscedastic sandwich.
    sig_vals = list(group_sigma.values())
    homoscedastic = len(sig_vals) <= 1 or (
        max(sig_vals) - min(sig_vals)
        <= 1e-12 * max(abs(v) for v in sig_vals)
    )

    def minv():
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "covariance: the parameter block of the inverse KKT matrix "
                "is singular; the fitted parameters are linearly "
                "dependent (structurally unidentifiable)") from e

    def group_jacobians():
        # The Jacobian rows are recovered from the same backsolves: the
        # residual rows of the z-columns equal J * inv(d2f/dp2), so
        # J = Z_r * inv(M).
        Mi = minv()
        out = {}
        for g, rws in groups.items():
            Zr = np.array([[zcols[j][r] for j in range(n_params)]
                           for r in rws])
            out[g] = Zr @ Mi                  # d r_g / d p
        return out

    # Active-bound projection: the covariance is computed in the free
    # (off-bound) directions and embedded with zero rows/cols for the
    # pinned parameters, i.e. the covariance conditional on the active
    # set. Restricting the INFORMATION matrix to the free block and
    # inverting (not restricting the inverse) is the curve_fit
    # _projected_covariance construction.
    free = [i for i in range(n_params) if i not in active]

    def embed(cov_ff):
        if len(free) == n_params:
            return cov_ff
        full = np.zeros((n_params, n_params))
        if free:
            full[np.ix_(free, free)] = cov_ff
        return full

    if not free:
        cov = np.zeros((n_params, n_params))
    elif hessian == "gauss-newton":
        # Expected information: H_GN = 2 J^T J in place of the exact
        # reduced Hessian. Pooled: cov = 2 s^2 inv(H_GN) = s^2 inv(J^T J).
        # Grouped: cov = inv(J^T J) (sum_g s_g^2 Jg^T Jg) inv(J^T J).
        Js = {g: Jg[:, free] for g, Jg in group_jacobians().items()}
        G = sum(Jg.T @ Jg for Jg in Js.values())
        try:
            Ginv = np.linalg.inv(G)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "covariance: the Gauss-Newton matrix J^T J is singular; "
                "the fitted parameters are linearly dependent in the "
                "residual Jacobian") from e
        if homoscedastic:
            cov = embed(sig_vals[0] * Ginv)
        else:
            B = np.zeros((len(free), len(free)))
            for g, Jg in Js.items():
                B += group_sigma[g] * (Jg.T @ Jg)
            cov = embed(Ginv @ B @ Ginv)
    else:
        if len(free) == n_params:
            Mc = M
        else:
            try:
                Mc = np.linalg.inv(minv()[np.ix_(free, free)])
            except np.linalg.LinAlgError as e:
                raise RuntimeError(
                    "covariance: the reduced Hessian restricted to the "
                    "free (off-bound) parameters is singular; the "
                    "remaining fitted parameters are linearly dependent"
                ) from e
        if homoscedastic:
            s2 = sig_vals[0]
            cov = embed(2.0 * s2 * Mc)
        else:
            # heteroscedastic sandwich: cov = A^-1 B A^-1 with A = d2f/dp2
            # and B built from per-group residual Jacobians.
            B = np.zeros((len(free), len(free)))
            for g, Jg in group_jacobians().items():
                Jf = Jg[:, free]
                B += group_sigma[g] * (Jf.T @ Jf)
            # dtheta = -A^-1 * 2 J^T eps with A = d2f/dp2 = inv(M), so
            # cov = 4 M (sum_g sigma_g^2 Jg^T Jg) M; the single-group case
            # reduces to 2 sigma^2 M since J^T J = A/2.
            cov = embed(4.0 * Mc @ B @ Mc)

    cov = 0.5 * (cov + cov.T)
    if np.diag(cov).min() < 0:
        warnings.warn(
            "covariance: negative variance on the diagonal; the point is "
            "probably not a least-squares minimum.")
    sig_out = (next(iter(group_sigma.values()))
               if len(group_sigma) == 1 and None in group_sigma
               else group_sigma)
    return Covariance(params, cov, sig_out)
