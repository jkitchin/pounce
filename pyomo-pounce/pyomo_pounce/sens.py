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

Mechanics: declared Params become pinned variables on a clone
(pyomo.contrib.sensitivity_toolbox does the expression surgery), the clone
is written to .nl and evaluated in-process via pounce.read_nl, and the
pounce.Solver session's parametric_step answers gradient()/estimate()
queries from the stored factorization -- the sIPOPT computation, with no
suffixes and no upfront perturbation values.
"""
import os
import shutil
import tempfile
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
    """Per-model sensitivity registry. Deepcopy-aware so model.clone() (and
    the sensitivity surgery's own clone) works cleanly: declared parameters
    follow the clone through the memo, while the session -- which holds
    solver handles tied to one converged factorization -- is deliberately
    not copied (a clone has no solve of its own yet)."""

    def __init__(self):
        self.params = []
        self.session = None

    def __deepcopy__(self, memo):
        import copy
        new = _Registry()
        memo[id(self)] = new
        new.params = [copy.deepcopy(p, memo) for p in self.params]
        return new


def declare_sens_param(param):
    """Flag a mutable Param (or fixed Var), scalar or indexed, for
    sensitivity. No perturbed value is required, or accepted."""
    model = param.model()
    reg = model.__dict__.setdefault(_REG, _Registry())
    reg.params.append(param)


def has_declarations(model):
    reg = getattr(model, "__dict__", {}).get(_REG)
    return bool(reg and reg.params)


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


def sens_solve(model, tee=False):
    """Solve `model` in-process with POUNCE and keep the KKT factorization
    for gradient()/estimate(). Called automatically by
    SolverFactory('pounce').solve() when declarations are present.
    Returns a Pyomo SolverResults, like an ordinary solve."""
    import pounce

    reg = model.__dict__[_REG]
    si = SensitivityInterface(model, clone_model=True)
    si.setup_sensitivity(reg.params)
    clone = si.model_instance

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

    prob = pounce.Problem(nl.n, nl.m, _NlBridge(nl),
                          lb=nl.x_l, ub=nl.x_u, cl=nl.g_l, cu=nl.g_u)
    solver = pounce.Solver(prob)
    x, info = solver.solve(np.asarray(nl.x0))
    if tee:
        print(info.get("status_msg", info))
    if not solver.converged:
        raise RuntimeError(
            f"pounce did not converge: {info.get('status_msg')}")

    block = clone.component(SensitivityInterface.get_default_block_name())
    pins = ComponentMap()
    for i, (var, clone_param, list_idx, comp_idx) in enumerate(
            block._sens_data_list):
        con = block.paramConst[i + 1]
        orig_comp = reg.params[list_idx]
        orig_data = (orig_comp if not orig_comp.is_indexed()
                     else orig_comp[comp_idx])
        pins[orig_data] = con_names.index(con.name)

    # original-name -> clone-row-name aliases for the replaced constraints
    con_alias = {}
    if getattr(block, "_has_replaced_expressions", False):
        for new_comp, old_comp in block._replaced_map.items():
            for nd, od in zip(_iter_data(new_comp), _iter_data(old_comp)):
                con_alias[od.name] = nd.name

    session = _Session(model, nl, solver, var_names, con_names, pins,
                       con_alias)
    session.base_x = np.asarray(x)
    reg.session = session

    # load the solution back onto the ORIGINAL model's variables
    for name, val in zip(var_names, session.base_x):
        ov = model.find_component(name)
        if ov is not None:
            ov.set_value(val, skip_validation=True)

    # Return a Pyomo SolverResults so callers see the same shape as an
    # ordinary solve (res.solver.termination_condition etc).
    results = SolverResults()
    results.solver.name = "pounce (in-process sensitivity session)"
    results.solver.status = SolverStatus.ok
    results.solver.termination_condition = TerminationCondition.optimal
    results.solver.message = str(info.get("status_msg", ""))
    if info.get("obj_val") is not None:
        results.problem.upper_bound = float(info["obj_val"])
        results.problem.lower_bound = float(info["obj_val"])
    return results


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

    def _value(self, td, pd):
        col = self._session.column(_param_pin(self._session, pd))
        return float(col[self._entry(td)])

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
