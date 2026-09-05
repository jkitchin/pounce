"""Declared-parameter sensitivity for Pyomo models solved with POUNCE.

Declare which parameters matter when you build the model -- no perturbed
values required -- then solve normally. The converged KKT factorization is
kept, and every sensitivity is a cheap backsolve afterwards:

    import pyomo_pounce
    from pyomo_pounce import declare_sens_param, sens_jacobian, sens_solution

    m.p = pyo.Param(initialize=2.0, mutable=True)
    declare_sens_param(m.p)

    pyo.SolverFactory("pounce").solve(m)     # normal solve

    sens_jacobian(m.x, wrt=m.p)         # dx*/dp (float)
    sens_jacobian(m.x, wrt=m.p2)        # containers -> Jacobian object
    sens_jacobian(m.c, wrt=m.p)         # d(multiplier of c)/dp
    sens_solution(m, [(m.p, 2.5)])      # perturbed-solution estimate,
                                        # clamped to bounds, warns on clamp

Estimation models use the other two declarations: flag the FITTED
variables and the residual container, solve once, and ask for the
covariance with no further information:

    declare_sens_fitted(m.A); declare_sens_fitted(m.k)
    declare_sens_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)  # one ordinary solve
    sens_covariance(m)                    # std errors, correlations,
                                          # identifiability diagnostics

Mechanics: a declared Param should enter the model through one defining
equality, a single variable equal to the param, the shape a
parameterized initial condition already has. `declare_sens_param`
records that row once, at declaration, and every solve then writes the
model AS WRITTEN to .nl, evaluates it in-process via pounce.read_nl,
and the pounce.Solver session's parametric_step answers
sens_jacobian()/sens_solution() queries from the stored factorization by
shifting the defining rows' right-hand sides -- the sIPOPT computation,
with no suffixes, no upfront perturbation values, and no per-solve model
copy.

A declared Param without that form (folded into several expressions,
in the objective, in a Var's bound) is rewritten in place, once, at
declaration, with a warning: its occurrences are replaced by a new
variable held by a new defining equality on the `_pounce_sens_defs`
block, the affected constraints and objectives edited in place so
their names and activity are untouched. A bound holding such a Param
moves into a constraint over the substitute, so its sensitivity is
real rather than zero; the moved bound is dropped from the Var, so
m.x.ub reads None afterward and the NL carries the no-bound sentinel
for that row. A declared FIXED Var is unfixed and held by a defining
equality at its value, since the NL writer would otherwise substitute
it out as a constant with no row to perturb.

Call-time `sens_params` keep the older mechanics: their surgery
(pyomo.contrib.sensitivity_toolbox) runs on a clone built for that one
solve and thrown away.
"""
import codecs
import os
import shutil
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.sensitivity_toolbox.sens import SensitivityInterface
from pyomo.core.base.constraint import Constraint
from pyomo.core.expr import identify_mutable_parameters, identify_variables
from pyomo.core.expr.visitor import replace_expressions
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition

from pounce.sensitivity import (
    # re-exported unchanged for `from pyomo_pounce import Covariance` and
    # friends: the result types are the core's, and a Pyomo caller gets
    # the same objects a bare-NL one does
    ActiveSetChange,          # noqa: F401
    Covariance,               # noqa: F401
    Information,              # noqa: F401
    NlBridge as _NlBridge,
    SensSession,
    SolutionReport,           # noqa: F401
    active_set_changes as _core_active_set_changes,
    check_margins as _check_margins,
    covariance as _core_covariance,
    information as _core_information,
    objective_sign as _objective_sign,
    refuse_on_pdpert as _refuse_on_pdpert,
    row_index as _row_index,
    solution as _core_solution,
    solution_report as _core_solution_report,
    user_row_names as _user_row_names,
    weakly_active as _weakly_active,
)

from pyomo_pounce.scaling import (
    problem_scaling,
    user_scaling_requested,
    warn_if_no_suffix,
)

_REG = "_pounce_sens"
# the model block holding substituted variables and defining equalities
# the in-place rewrite creates for non-conforming declared params
_DEFS = "_pounce_sens_defs"


# ── declaration ───────────────────────────────────────────────────────────────

class _Registry:
    """Per-model registry of declared statistical roles. Deepcopy-aware so
    model.clone() (and the sensitivity surgery's own clone) works cleanly:
    declared components follow the clone through the memo, while the
    session -- which holds solver handles tied to one converged
    factorization -- is deliberately not copied (a clone has no solve of
    its own yet)."""

    def __init__(self):
        self.params = []      # pinned inputs: sens_jacobian/sens_solution
        self.fitted = []       # free fitted variables: sens_covariance()
        self.residuals = []       # (container, group) pairs: sigma^2
        self.retain = False       # keep the factor with no declaration
        self.session = None
        # (param data, defining ConstraintData, d(row)/d(param)) per
        # declared param, recorded at declaration. The solve pins these
        # rows as written. No clone and no surgery happen at solve time.
        self.pin_records = []
        # var name -> (lb, ub) numeric values at declaration, for
        # bounds the in-place rewrite moved into constraints
        self.moved_bounds = {}

    def __deepcopy__(self, memo):
        import copy
        new = _Registry()
        memo[id(self)] = new
        new.params = [copy.deepcopy(p, memo) for p in self.params]
        new.fitted = [copy.deepcopy(p, memo) for p in self.fitted]
        new.residuals = [(copy.deepcopy(r, memo), g)
                         for r, g in self.residuals]
        new.retain = self.retain
        new.pin_records = [
            (copy.deepcopy(p, memo), copy.deepcopy(c, memo), k)
            for p, c, k in self.pin_records]
        new.moved_bounds = dict(self.moved_bounds)
        return new


def _registry(model):
    return model.__dict__.setdefault(_REG, _Registry())


def declare_sens_param(*params):
    """Flag one or more mutable Params (or fixed Vars), scalar or indexed,
    as FIXED INPUTS for sensitivity: after a solve, sens_jacobian() and
    sens_solution() answer d(solution)/d(param) questions. No perturbed value
    is required, or accepted.

    A declared Param should enter the model through one defining
    equality: a single variable equal to the param, the way an initial
    condition pins a state. Such a model solves as written on every
    solve, and the defining equality is the row the sensitivity
    machinery pins. A declared Param that appears anywhere else (several
    constraints, the objective, a variable bound) is rewritten in
    place, once, at declaration: the param's occurrences are replaced
    by a variable pinned by a new defining equality, with a warning
    naming what changed. Declaring a fixed Var unfixes it and pins it
    where it stands. The inspection and any rewrite happen here, in
    this call. A solve never clones the model and never rewrites
    anything. Editing the model afterward so a declared Param leaks
    into new expressions is not detected and is unsupported."""
    by_model = {}
    for param in params:
        by_model.setdefault(id(param.model()), (param.model(), []))[1] \
            .append(param)
    # every component of the call is validated before anything is
    # recorded or rewritten, so a raising declaration leaves every
    # model exactly as it was
    for _model, comps in by_model.values():
        for comp in comps:
            for pd in _iter_data(comp):
                if pd.is_variable_type() and not pd.fixed:
                    raise ValueError(
                        f"declare_sens_param: {pd.name} is a Var that "
                        "is not fixed. Declare mutable Params or fixed "
                        "Vars.")
                if not (pd.is_variable_type() or pd.is_parameter_type()):
                    raise ValueError(
                        f"declare_sens_param: {pd.name} is neither a "
                        "Param nor a Var.")
    for model, comps in by_model.values():
        reg = _registry(model)
        _register_pins(model, reg, comps)
        reg.params.extend(comps)


def _linear_coefficient(expr, wrt):
    """d(expr)/d(wrt) when it is a plain number, else None.

    `wrt` may be a ParamData or a VarData. A param is substituted by a
    throwaway variable first, since the differentiator works with
    respect to variables. The derivative must be constant, no variables
    and no mutable params in it, so a nonlinear or param-scaled entry
    reads as None and the caller treats the row as non-conforming."""
    from pyomo.core.expr.calculus.derivatives import Modes, differentiate

    target = wrt
    if not wrt.is_variable_type():
        dummy = pyo.Var()
        dummy.construct()
        expr = replace_expressions(expr, {id(wrt): dummy})
        target = dummy
    try:
        d = differentiate(expr, wrt=target, mode=Modes.reverse_symbolic)
    except Exception:
        return None
    if isinstance(d, (int, float)):
        return float(d)
    if any(True for _ in identify_variables(d)):
        return None
    if any(True for _ in identify_mutable_parameters(d)):
        return None
    try:
        return float(pyo.value(d))
    except (ValueError, TypeError):
        return None


def _defining_row(pd, cons, other_declared_ids):
    """(ConstraintData, coefficient) when `cons` is one conforming
    defining equality for param data `pd`, else None. The row must be
    an equality over a single variable, linear in both the variable and
    the param, with no other declared param in it."""
    if len(cons) != 1:
        return None
    con = cons[0]
    if not con.equality:
        return None
    parts = [con.body]
    rhs = con.lower
    if rhs is not None and not isinstance(rhs, (int, float)):
        parts.append(rhs)
    for part in parts:
        for p in identify_mutable_parameters(part):
            if id(p) in other_declared_ids and p is not pd:
                return None
    variables = list(identify_variables(con.body))
    if len(variables) != 1:
        return None
    resid = con.body if rhs is None else con.body - rhs
    coeff = _linear_coefficient(resid, pd)
    if coeff is None or coeff == 0.0:
        return None
    vcoef = _linear_coefficient(resid, variables[0])
    if vcoef is None or vcoef == 0.0:
        return None
    return con, coeff


def _register_pins(model, reg, comps):
    """Record each newly declared param's defining equality, rewriting
    the model in place, once, for the params that have none. `comps`
    arrive validated: every data is a mutable Param or a fixed Var, so
    nothing below raises and a declaration either completes or leaves
    the model untouched."""
    datas = [(comp, list(_iter_data(comp))) for comp in comps]
    declared_ids = set()
    for comp in reg.params:
        for pd in _iter_data(comp):
            declared_ids.add(id(pd))
    for _comp, ds in datas:
        for pd in ds:
            declared_ids.add(id(pd))

    # every active constraint each new param data appears in, one walk
    param_ids = {id(pd): pd for _comp, ds in datas for pd in ds
                 if pd.is_parameter_type()}
    hits = {i: [] for i in param_ids}
    in_objective = set()
    in_bound = set()
    for con in model.component_data_objects(pyo.Constraint, active=True,
                                            descend_into=True):
        found = set()
        for part in (con.body, con.lower, con.upper):
            if part is None or isinstance(part, (int, float)):
                continue
            for p in identify_mutable_parameters(part):
                if id(p) in param_ids:
                    found.add(id(p))
        for i in found:
            hits[i].append(con)
    for obj in model.component_data_objects(pyo.Objective, active=True,
                                            descend_into=True):
        for p in identify_mutable_parameters(obj.expr):
            if id(p) in param_ids:
                in_objective.add(id(p))
    # a FIXED Var's bounds are never enforced, so a param sitting in
    # one is no reason to rewrite anything
    for v in model.component_data_objects(pyo.Var, active=True,
                                          descend_into=True):
        if v.fixed:
            continue
        for attr in ("_lb", "_ub"):
            expr = getattr(v, attr, None)
            if expr is None or isinstance(expr, (int, float)):
                continue
            for p in identify_mutable_parameters(expr):
                if id(p) in param_ids:
                    in_bound.add(id(p))

    conforming = []   # [(pd, con, coeff), ...]
    rewrite = []      # components needing the in-place rewrite
    loud = []         # the subset whose reason is worth a warning
    for comp, ds in datas:
        records = []
        noisy = False
        for pd in ds:
            if pd.is_variable_type():
                records = None
                noisy = True
                break
            if id(pd) in in_objective:
                records = None
                noisy = True
                break
            if id(pd) in in_bound:
                # A bound spelling always takes the rewrite: the move
                # to a constraint is how the perturbation reaches the
                # bound at all. It is the documented mechanics for the
                # documented `bounds=(0, m.p)` form, so it warns only
                # when the constraint side is ALSO non-conforming.
                records = None
                row = _defining_row(pd, hits.get(id(pd), []),
                                    declared_ids)
                if hits.get(id(pd)) and row is None:
                    noisy = True
                break
            found = _defining_row(pd, hits.get(id(pd), []), declared_ids)
            if found is None:
                records = None
                noisy = True
                break
            records.append((pd, found[0], found[1]))
        if records is None:
            rewrite.append(comp)
            if noisy:
                loud.append(comp)
        else:
            conforming.extend(records)

    reg.pin_records.extend(conforming)

    if not rewrite:
        return
    if loud:
        names = ", ".join(c.name for c in loud)
        warnings.warn(
            f"declare_sens_param: {names} do not enter the model "
            "through a single defining equality, so the model was "
            "rewritten in place: a folded Param's occurrences now read "
            "a substituted variable pinned by a new equality on the "
            "_pounce_sens_defs block, and a fixed Var is unfixed and "
            "pinned there at its value. To declare without rewriting, "
            "give the param one defining equality (v == p) and use v "
            "in the expressions.")
    blk = model.component(_DEFS)
    if blk is None:
        blk = pyo.Block()
        model.add_component(_DEFS, blk)
        blk.v = pyo.VarList()
        blk.pin = pyo.ConstraintList()
        blk.bound = pyo.ConstraintList()

    # One substituted variable and one defining equality per rewritten
    # param data. A declared FIXED Var needs no substitute: it is
    # unfixed and pinned where it stands, and the solve refreshes its
    # pin from the Var's current value so later set_value / fix calls
    # keep tracking.
    sub = {}
    for comp in rewrite:
        for pd in _iter_data(comp):
            if pd.is_variable_type():
                pd.unfix()
                con = blk.pin.add(pd == float(pyo.value(pd)))
                reg.pin_records.append((pd, con, -1.0))
                continue
            v = blk.v.add()
            v.set_value(float(pyo.value(pd)))
            sub[id(pd)] = v
            con = blk.pin.add(v == pd)
            # the defining row is `v - p == 0`, so d(row)/d(p) is -1
            reg.pin_records.append((pd, con, -1.0))
    if not sub:
        return

    # every occurrence outside the new defining rows reads the
    # substitute: constraints and objectives are edited in place, so
    # their names, activity and order are untouched
    touched = {}
    for i, cons in hits.items():
        if i in sub:
            for con in cons:
                touched[id(con)] = con
    for con in touched.values():
        con.set_value(replace_expressions(con.expr, sub))
    for obj in model.component_data_objects(pyo.Objective, active=True,
                                            descend_into=True):
        if any(id(p) in sub
               for p in identify_mutable_parameters(obj.expr)):
            obj.set_value(replace_expressions(obj.expr, sub))
    # A bound holding a rewritten param moves into a constraint over
    # the substitute, where the perturbation reaches it (gh#356). The
    # numeric values at declaration are recorded for sens_covariance()'s
    # activity test.
    for v in model.component_data_objects(pyo.Var, active=True,
                                          descend_into=True):
        if v.fixed:
            continue
        for attr in ("_lb", "_ub"):
            expr = getattr(v, attr, None)
            if expr is None or isinstance(expr, (int, float)):
                continue
            if not any(id(p) in sub
                       for p in identify_mutable_parameters(expr)):
                continue
            val = float(pyo.value(expr))
            moved = replace_expressions(expr, sub)
            lo, hi = reg.moved_bounds.get(v.name, (None, None))
            if attr == "_lb":
                v.setlb(None)
                blk.bound.add(moved <= v)
                reg.moved_bounds[v.name] = (val, hi)
            else:
                v.setub(None)
                blk.bound.add(v <= moved)
                reg.moved_bounds[v.name] = (lo, val)


def declare_sens_fitted(*variables):
    """Flag one or more FREE Vars (scalar or indexed) as fitted
    parameters of a least-squares problem: after one ordinary solve,
    sens_covariance() reports their asymptotic uncertainty. The variables stay
    free in the solve; do not fix them."""
    for var in variables:
        _registry(var.model()).fitted.append(var)


def declare_sens_residual(*containers, group=None):
    """Flag one or more indexed Vars holding the fit residuals, one member
    per data point. sens_covariance() derives the residual count and the SSR
    from them, so no data counts need to be passed. `group` is an
    arbitrary user string partitioning residuals into noise groups and
    applies to every container in the call: containers sharing a group
    (or all ungrouped containers together) pool into one estimated noise
    variance; distinct groups get their own, and the covariance switches
    to the heteroscedastic sandwich form."""
    for container in containers:
        _registry(container.model()).residuals.append((container, group))


def sens_retain_kkt(model):
    """Keep the KKT factorization after the next solve with NOTHING
    declared (covariance roadmap item 4). The factor the solve computes
    anyway is retained for post-solve queries, so `sens_covariance(model,
    of=block)` and `sens_information(model, of=block)` work on any block
    without a declared default: the MHE case, where the arrival state
    and the parameters are each queried by of= and neither is THE
    fitted set. `sens_covariance(model)` with no block stays an error (there
    is no default to reduce onto), and a solve without this call and
    without declarations pays nothing, exactly as before.

    The retention policy in one place: the factor is kept if anything
    is declared (declare_sens_param / declare_sens_fitted /
    declare_sens_residual), or if sens_retain_kkt() was called; and a
    Covariance/Information result whose lazy conditioned_on has not
    been read keeps the session alive through its pending computation
    until first access. sens_release_kkt(model) drops the held factor on
    demand.

    Like any declaration, this routes the solve through the
    in-process sensitivity path, whose solve() surface is not
    keyword-identical to the ordinary subprocess path (for example,
    load_solutions=False is not honored there): adding it to an
    existing script changes how the solve runs, not just what is
    kept."""
    _registry(model).retain = True


def sens_release_kkt(model):
    """Drop the held KKT factorization now, freeing its memory.

    The exit of the retention story, for the current factor only:
    declarations and a prior sens_retain_kkt() are untouched and apply to
    the NEXT solve, which keeps its factor again. After release the
    accessors raise their no-session error until another solve.

    Release drops the MODEL's hold, not a result's, and two kinds of
    result hold their own reference to the session: a Covariance or
    Information whose conditioned_on is still pending (until the
    attribute is read), and a Jacobian handed back by sens_jacobian()
    (which uses the session on every lookup). Such a result keeps
    working after the release, and keeps the factor in memory until
    it too is discarded.

    Returns True if a factorization was held and is now released,
    False if there was nothing to release."""
    reg = model.__dict__.get(_REG)
    if reg is None or reg.session is None:
        return False
    reg.session = None
    return True


def _reformulate_param_bounds(clone):
    """Move Var bounds that reference a declared Param into constraints.

    `SensitivityInterface` substitutes declared Params in constraint
    expressions only. A Param left in a bound is therefore written to the
    NL file as a constant at its pre-perturbation value, so the bound never
    moves and the reported sensitivity to that Param reads as exactly zero
    (jkitchin/pounce#356). Rewriting the bound as a constraint over the
    substituted Var puts it where the perturbation already reaches, which
    makes the answer exact in the bound rather than approximating it.

    Runs after `setup_sensitivity`, so the Param-to-Var map it needs is the
    one the surgery has already built. Returns {var name: (lb, ub)} of the
    numeric values the moved bounds had at the solve point, with None on the
    side that was not moved. sens_covariance() classifies these rows
    through the
    activity report: a moved bound is a single-coordinate constraint row
    and projects exactly as the original bound would, so no bound
    re-injection is needed anywhere.
    """
    block = clone.component(SensitivityInterface.get_default_block_name())
    if block is None:
        return {}
    sub = {id(param): var for var, param, _, _ in block._sens_data_list}
    if not sub:
        return {}

    moved = []
    # active=True so deactivated Blocks are skipped: stripping a bound there
    # and adding an active constraint would pull the Var into the NL as a
    # free column. Fixed Vars are skipped because their bounds are not
    # enforced -- Pyomo substitutes them out as constants -- so turning one
    # into a constraint would impose a restriction on the pinned Param that
    # the original model never had.
    for v in clone.component_data_objects(pyo.Var, active=True,
                                          descend_into=True):
        if v.fixed:
            continue
        for attr in ("_lb", "_ub"):
            expr = getattr(v, attr, None)
            if expr is None or isinstance(expr, (int, float)):
                continue
            if not any(id(p) in sub
                       for p in identify_mutable_parameters(expr)):
                continue
            # the numeric value the NL would have carried, kept so
            # sens_covariance() can still see where the bound was
            moved.append((v, attr, float(pyo.value(expr)),
                          replace_expressions(expr, sub)))

    if not moved:
        return {}

    recorded = {}
    block.boundConst = pyo.ConstraintList()
    for v, attr, val, expr in moved:
        lo, hi = recorded.get(v.name, (None, None))
        if attr == "_lb":
            v.setlb(None)
            block.boundConst.add(expr <= v)
            recorded[v.name] = (val, hi)
        else:
            v.setub(None)
            block.boundConst.add(v <= expr)
            recorded[v.name] = (lo, val)
    return recorded


def has_declarations(model):
    reg = getattr(model, "__dict__", {}).get(_REG)
    return bool(reg and (reg.params or reg.fitted or reg.residuals
                         or reg.retain))


# ── the read_nl -> callback-Problem bridge ────────────────────────────────────

# ── session ───────────────────────────────────────────────────────────────────

class SolutionMap(Mapping):
    """What `sens_solution()` returns: a read-only mapping {original var
    data: estimated value}, keyed by the component data objects
    themselves, by identity, the way `ComponentMap` keys them.

    The keys and the identity index are the session's, shared by every
    result, and each result carries only its own value vector, so
    constructing one costs nothing per variable where building a
    `ComponentMap` paid an insertion per variable per call. Item
    assignment is not supported: a result describes one estimate, and
    writing into it never changed anything downstream."""

    __slots__ = ("_keys", "_index_of", "_values")

    def __init__(self, keys, index_of, values):
        self._keys = keys
        self._index_of = index_of
        self._values = values

    def __getitem__(self, vd):
        try:
            return float(self._values[self._index_of[id(vd)]])
        except KeyError:
            raise KeyError(vd) from None

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, vd):
        return id(vd) in self._index_of

    def __eq__(self, other):
        # Mapping's default equality round-trips through dict(self),
        # which raises on unhashable component data, the very reason
        # ComponentMap exists. Compare through the identity index the
        # way ComponentMap compares.
        if self is other:
            return True
        if not isinstance(other, Mapping):
            return NotImplemented
        if len(self) != len(other):
            return False
        for k, v in other.items():
            i = self._index_of.get(id(k))
            if i is None or float(self._values[i]) != v:
                return False
        return True

    def __deepcopy__(self, memo):
        # The identity index is valid only for the objects in _keys, so
        # a copy rebuilds it from the copied keys at their original
        # columns. ComponentMap does the same through its autoslot
        # rehash hook.
        import copy
        cols = [self._index_of[id(k)] for k in self._keys]
        keys = copy.deepcopy(self._keys, memo)
        values = copy.deepcopy(self._values, memo)
        new = self.__class__(
            keys, {id(k): c for k, c in zip(keys, cols)}, values)
        memo[id(self)] = new
        return new

    def __repr__(self):
        return f"SolutionMap({len(self._keys)} variables)"


class _Session(SensSession):
    """`SensSession` in Pyomo's terms.

    The core session addresses everything by row and keys its results by
    whatever the caller handed it. This adds the modelling half: the
    model itself, the variable data behind each `.col` column, and the
    `ComponentMap` containers a Pyomo user expects back -- Pyomo
    components are unhashable, so a plain dict cannot hold them.
    """

    def __init__(self, model, nl, solver, var_names, con_names, pins,
                 con_alias, var_row=None, con_row=None):
        super().__init__(nl, solver, var_names, con_names, pins=pins,
                         con_alias=con_alias, var_row=var_row,
                         con_row=con_row)
        self.model = model            # original model
        # d(row)/d(param) and the param's solve-time value, per declared
        # param pinned through its own defining equality. Empty for the
        # clone-and-surgery paths, whose rows carry the param value as
        # the row's right-hand side. ComponentMap, not the base's dict,
        # because these are keyed by param data.
        self.pin_coefs = ComponentMap()
        self.pin_bases = ComponentMap()
        self.fit_rows = ComponentMap()
        # The original model's variable data, in .col order, captured
        # when the solve loads its solution back. A column the
        # declared-parameter surgery created has no model counterpart
        # and holds None. Resolving a name through find_component
        # parses it through pyomo's component-UID lexer, 0.87 s
        # accumulated inside one sens_solution() call on the 62k-variable
        # double column (N=25 Radau collocation), and every sens_solution() and
        # sens_jacobian(of=None) call needs the whole list, so the one
        # resolution the solve already performs is kept instead. The
        # references are the solve's own objects: a caller who deletes
        # and rebuilds model components after the solve invalidates
        # this session like any other of its caches.
        self.var_data = None
        self._solution_keys = None    # (keys, id -> column), on demand

    # ── the three hooks the core reports results through ─────────────

    @staticmethod
    def new_keymap():
        return ComponentMap()

    def var_key(self, full_idx):
        """The variable data behind `.col` column `full_idx`.

        None for a column the declared-parameter surgery created, which
        has no counterpart on the user's model: the core skips those
        rather than reporting a name the user never wrote.
        """
        return self.var_data[full_idx]

    def user_row_data(self):
        """The original model's constraint data per solve row, in .row
        order, resolved once per session. A pin constraint has no
        original counterpart and holds None."""
        if self._row_data is None:
            self._row_data = [self.model.find_component(nm)
                              for nm in _user_row_names(self)]
        return self._row_data

    # ── Pyomo-only ───────────────────────────────────────────────────

    def solution_keys(self):
        """The columns a solution map exposes: the captured variable
        data with the surgery-created columns dropped, plus the
        identity-to-column index a lookup uses. Built once per session;
        every `sens_solution()` result shares them. The keys list holds the
        strong references that keep the ids in the index unique: an id
        is only unique among live objects, so the index is valid
        exactly as long as the list that accompanies it."""
        if self._solution_keys is None:
            keys = []
            index_of = {}
            for i, vd in enumerate(self.var_data):
                if vd is not None:
                    index_of[id(vd)] = i
                    keys.append(vd)
            self._solution_keys = (keys, index_of)
        return self._solution_keys


def _iter_data(comp):
    if comp.is_indexed():
        for idx in comp:
            yield comp[idx]
    else:
        yield comp


# Engine status -> (termination condition, solver status), mirroring the
# semantics Pyomo's .sol reader gives the ordinary path via the AMPL
# exit-code ranges (optimal / infeasible / unbounded / limit / error).
#
# `Solved_To_Acceptable_Level` is `ok`, not `warning`: the solve is an
# accepted one, and the AMPL code POUNCE emits for it (1, Ipopt's own) puts
# the ordinary `.sol` route in the 0..99 band that Pyomo's reader loads as
# `ok`. Reporting `warning` here would make the sensitivity route disagree
# with both the `.sol` route and `v2._V2_STATUS`, and would make Pyomo log a
# load warning on a result IPOPT loads clean (gh #591). The reduced-accuracy
# distinction stays in the solver message, not the severity.
#
# The table is exhaustive over `ApplicationReturnStatus`
# (`crates/pounce-nlp/src/return_codes.rs`), whose `upstream_name()` is the
# `status_msg` read below; `test_issue_589_status_table_coverage.py` holds it
# and `v2._V2_STATUS` to the full enum. Eleven exits used to be missing,
# `Restoration_Failed` among them, and they fell to the `(error, error)`
# default. That default is a defensible severity, but it is less specific than
# the `.sol` route's answer for the same solve -- Pyomo's reader turns the
# AMPL 500 failure band into `internalSolverError` -- and on the v2 side the
# matching gap decided whether the solve raised at all (gh #589).
_STATUS_RESULT = {
    "Solve_Succeeded":
        (TerminationCondition.optimal, SolverStatus.ok),
    "Solved_To_Acceptable_Level":
        (TerminationCondition.optimal, SolverStatus.ok),
    # `ok`, not `warning`, for the reason `Solved_To_Acceptable_Level`
    # above is `ok`: POUNCE emits this status only for a square problem
    # (`resto_inner_solver.rs` gates it on `is_square_problem`), where the
    # objective is constant and a feasible point is the solution, and the
    # AMPL code it emits for it (2, Ipopt's own) puts the `.sol` route in
    # the 0..99 band both Pyomo readers load as a success. `warning` here
    # would make this route disagree with the `.sol` route and with
    # `v2._V2_STATUS` on the same solve (gh #815).
    #
    # The session asymmetry below is unchanged and deliberate: the engine's
    # `on_converged` callback still fires only for Solve_Succeeded /
    # Solved_To_Acceptable_Level, so this solve arrives with `converged =
    # False` and no retained factorization. That is not a reporting gap to
    # close by widening the callback gate -- a square feasible point is
    # reached through the restoration phase, whose factorization is not the
    # original problem's KKT matrix, so a session built from it would answer
    # sensitivity queries from the wrong system.
    "Feasible_Point_Found":
        (TerminationCondition.optimal, SolverStatus.ok),
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
    # AMPL's 400 "limit" band, like the two above. `warning` is the band's
    # severity, and this is the ONE row whose severity differs from the
    # `(error, error)` default it used to take -- a stalled solve is a limit
    # case, not a failure, which is why it sits in the limit band. The
    # termination condition deviates from what Pyomo's band-reading `.sol`
    # table gives (`maxIterations`): POUNCE names this exit exactly, and the
    # legacy enum has the member, so it does not have to borrow the
    # iteration-limit one. Same deliberate precision as `maxTimeLimit` above.
    "Search_Direction_Becomes_Too_Small":
        (TerminationCondition.minStepLength, SolverStatus.warning),
    # AMPL's 500 failure band. `internalSolverError` + `error` is what the
    # ordinary `.sol` route reports for every code in it, so these agree with
    # it rather than with the coarser `(error, error)` default they used to
    # take. The two definition errors deviate on the termination condition,
    # again for precision: `.sol` can only say `internalSolverError` for the
    # whole band, while an over-determined or malformed model is not a solver
    # failure and the legacy enum has `invalidProblem` for exactly that. The
    # severity -- which is what callers branch on -- stays `error` either way.
    "Restoration_Failed":
        (TerminationCondition.internalSolverError, SolverStatus.error),
    "Error_In_Step_Computation":
        (TerminationCondition.internalSolverError, SolverStatus.error),
    "Invalid_Number_Detected":
        (TerminationCondition.internalSolverError, SolverStatus.error),
    "Insufficient_Memory":
        (TerminationCondition.internalSolverError, SolverStatus.error),
    "Internal_Error":
        (TerminationCondition.internalSolverError, SolverStatus.error),
    "Invalid_Option":
        (TerminationCondition.internalSolverError, SolverStatus.error),
    "Not_Enough_Degrees_Of_Freedom":
        (TerminationCondition.invalidProblem, SolverStatus.error),
    "Invalid_Problem_Definition":
        (TerminationCondition.invalidProblem, SolverStatus.error),
    # ABI-parity members of the upstream enum that POUNCE never returns.
    # Listed so the table covers the enum, not just today's exits.
    "Unrecoverable_Exception":
        (TerminationCondition.internalSolverError, SolverStatus.error),
    "NonIpopt_Exception_Thrown":
        (TerminationCondition.internalSolverError, SolverStatus.error),
}


def _replaced_aliases(clone, si):
    """Original constraint name -> clone constraint name, for rows the
    declared-parameter surgery replaced. Built from the surgery block,
    so it is available before the solve; the session stores the same
    map afterwards."""
    if si is None:
        return {}
    block = clone.component(SensitivityInterface.get_default_block_name())
    if block is None or not getattr(block, "_has_replaced_expressions",
                                    False):
        return {}
    out = {}
    for new_comp, old_comp in block._replaced_map.items():
        for nd, od in zip(_iter_data(new_comp), _iter_data(old_comp)):
            out[od.name] = nd.name
    return out


def _warm_start_requested(options):
    """`warm_start_init_point` truthiness, accepting what
    `Problem.add_option` itself accepts: Python True maps to "yes"
    there, so it (and "y") must enter warm-start mode here too, or the
    Pyomo-natural spelling would warm-start with no seeds."""
    v = (options or {}).get("warm_start_init_point", "no")
    return v is True or str(v).strip().lower() in ("yes", "y")


def _warm_start_from_suffixes(model, var_names, con_names, nl, con_alias):
    """Initial multipliers for `warm_start_init_point=yes`, read from
    the model's `dual` (equality multipliers) and `ipopt_zL_in` /
    `ipopt_zU_in` (bound multipliers) suffixes, matched to the solve's
    rows by component name; a constraint replaced by the
    declared-parameter surgery is reached through its clone alias.

    Two sign conventions are crossed on the way in. `m.dual[c]` holds
    the AMPL marginal `d obj / d b = -lambda` (gh #271), while the
    session's `lagrange=` wants the internal `+lambda`, so dual entries
    negate. `ipopt_zU_in` follows Ipopt's convention, negative at an
    active upper bound (gh #296), while the session wants the internal
    non-negative `z_u`, so it negates too; `zL` is positive in both.

    A third crosses on a `maximize` model, and it multiplies all three.
    A suffix the user holds is stated against the objective they wrote;
    `read_nl` negates that objective before the engine ever sees it, so
    the multipliers the engine wants are the negation of the ones the
    suffix carries. `objective_sign` is that factor, +1 on every
    minimization -- which is why seeding a maximization used to hand the
    engine a certificate of the wrong sign, a worse starting point than
    the default it displaced.

    Entries the user did not supply are seeded NaN, the session's
    "unseeded" marker: the warm-start initializer substitutes its own
    resolved defaults (`bound_mult_init_val` for bound multipliers, 0
    for equality duals), so partial seeds never turn into the zero
    certificate an ASL-style dense array forces, and the defaults live
    in one place. An explicit zero is honored, then floored at
    `warm_start_mult_bound_push` exactly as a round-tripped inactive
    multiplier is.
    """
    sense = _objective_sign(nl)
    y = np.full(int(nl.m), np.nan)
    zl = np.full(int(nl.n), np.nan)
    zu = np.full(int(nl.n), np.nan)
    var_row = _row_index(var_names)
    con_row = _row_index(con_names)
    dual = model.component("dual")
    if isinstance(dual, pyo.Suffix):
        for cd, val in dual.items():
            r = con_row.get(con_alias.get(cd.name, cd.name))
            # con_names is trimmed to the constraint rows at the read
            # site, so a hit here is always a row of the dual vector
            if r is not None:
                # AMPL marginal -> the internal lambda of the objective
                # the engine minimizes
                y[r] = -sense * float(val)
    for sfx_name, arr, sign in (("ipopt_zL_in", zl, 1.0),
                                ("ipopt_zU_in", zu, -1.0)):
        sfx = model.component(sfx_name)
        if isinstance(sfx, pyo.Suffix):
            for vd, val in sfx.items():
                r = var_row.get(vd.name)
                if r is not None:
                    arr[r] = sense * sign * float(val)
    return {"lagrange": y, "zl": zl, "zu": zu}


#: The import suffixes this route fills, in the spelling Pyomo, AMPL and
#: Ipopt share. `rc` is deliberately absent: the `.sol` route does not
#: populate it either (measured), and a reduced cost is the combination
#: of the two bound multipliers, which `v2.get_reduced_costs` forms.
_RESULT_SUFFIXES = ("dual", "ipopt_zL_out", "ipopt_zU_out")


def _load_result_suffixes(model, info, nl, var_data, con_names, con_alias,
                          moved_bounds=None):
    """Fill the model's IMPORT suffixes from an in-process solve.

    Without this, `declare_sens_param` silently costs the caller their
    duals: the ordinary `.sol` route goes through Pyomo's own solution
    loader, which populates every active import suffix, while this route
    reads the engine's vectors directly and used to load primals only.
    A model that declared `m.dual = Suffix(IMPORT)` and then declared a
    sensitivity parameter got an *empty* suffix back and a `KeyError` on
    the first lookup -- no warning, and nothing about the declaration
    suggests it should touch duals.

    The three sign conventions are the ones `_warm_start_from_suffixes`
    crosses on the way in, run backwards, and they are pinned by
    `tests/test_result_suffixes.py` against the `.sol` route on the same
    models:

    * `dual` is the AMPL marginal ``d obj / d b = -lambda`` (gh #271)
      against the engine's internal ``+lambda`` in ``info['mult_g']``;
    * `ipopt_zU_out` is negative at an active upper bound (gh #296)
      against the engine's non-negative ``mult_x_U``; ``zL`` agrees in
      sign with ``mult_x_L``;
    * all three flip once more on a `maximize` model, because `read_nl`
      negated the objective before the engine saw it and a multiplier is
      a coefficient of the objective it was generated against.

    Two deliberate differences from the `.sol` route, neither of which
    changes a value:

    * **Membership.** The `.sol` writer emits one entry per variable --
      the combined reduced cost, routed to `zL` when positive and to
      `zU` when negative -- so a variable appears in exactly one of the
      two and a bound whose multiplier lost the comparison is not
      reported at all. Here every finite lower bound gets a `zL` entry
      and every finite upper bound a `zU` entry, which is the question
      the suffix name asks. Values agree wherever both routes report.
    * **Coverage.** A component the declared-parameter surgery created
      exists only on the clone and has no counterpart to key an entry
      by, so it is skipped, exactly as the primal load-back skips it.

    `moved_bounds` is the third, and it is a correction rather than a
    difference. The bound gate below reads `vd`, which is the MODEL's
    variable, while the multiplier beside it comes from the SOLVED
    problem -- and on a call-time `sens_params=` clone those two disagree
    about exactly one thing. `_reformulate_param_bounds` moves a bound
    that mentions a declared Param into a row over the substitute
    (gh#356), doing `setlb(None)` on the CLONE; the model keeps its
    bound, which still evaluates to a number. So the gate passes, the
    engine reports the zero it carries for a variable that has no such
    bound, and the caller reads `ipopt_zL_out[v] == 0.0` -- "this bound
    is inactive" -- for a bound whose marginal is alive on the row the
    surgery added. Skipping the moved side is what keeps the entry
    absent instead of fabricated. The declared route strips the bound
    from the model itself, so `vd.lb` is already None there and this is
    a no-op; the clone route is the one that needs it.

    Every active import suffix is cleared first, including ones left
    unfilled -- that is what `Model.solutions.load_from` does, and
    leaving a previous solve's entries standing under a new solution is
    the more dangerous of the two failure modes.
    """
    from pyomo.core.base.suffix import active_import_suffix_generator

    suffixes = dict(active_import_suffix_generator(model))
    if not suffixes:
        return
    for sfx in suffixes.values():
        sfx.clear_all_values()

    sense = _objective_sign(nl)

    lam = info.get("mult_g")
    dual = suffixes.get("dual")
    if dual is not None and lam is not None:
        con_row = _row_index(con_names)
        for cd in model.component_data_objects(Constraint, active=True,
                                               descend_into=True):
            # the model's constraint, reached in the solve under its
            # clone's name when the surgery replaced it -- the same
            # indirection the warm-start reader and v2's `_row_of` apply
            row = con_row.get(con_alias.get(cd.name, cd.name))
            if row is not None:
                dual[cd] = -sense * float(lam[row])

    zl, zu = info.get("mult_x_L"), info.get("mult_x_U")
    # `side` indexes the (lb, ub) pair `moved_bounds` records
    pairs = [(suffixes.get("ipopt_zL_out"), zl, 1.0, "lb", 0),
             (suffixes.get("ipopt_zU_out"), zu, -1.0, "ub", 1)]
    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
    if not pairs:
        return
    moved_bounds = moved_bounds or {}
    for row, vd in enumerate(var_data):
        if vd is None:
            continue
        moved = moved_bounds.get(vd.name)
        for sfx, vec, sign, bound, side in pairs:
            # the bound the SOLVE saw, not the one the model still
            # carries: a bound the surgery moved into a row is gone from
            # the solved problem, so the engine's zero is the absence of
            # a bound rather than an inactive one (see above)
            if moved is not None and moved[side] is not None:
                continue
            # an infinite bound has no multiplier to report; the engine
            # carries a zero there and reporting it would read as a
            # bound that exists and is inactive
            if getattr(vd, bound) is not None:
                sfx[vd] = sense * sign * float(vec[row])


def _stream_solve(solver, x0, **solve_kwargs):
    """Run ``solver.solve(x0, **solve_kwargs)`` with the engine's log streamed to sys.stdout.

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
        out = solver.solve(x0, **solve_kwargs)
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
            out = solver.solve(x0, **solve_kwargs)
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
               residuals=None, options=None, capture=None):
    """Solve `model` in-process with POUNCE and keep the KKT factorization
    for sens_jacobian()/sens_solution()/sens_covariance(). Called
    automatically by
    SolverFactory('pounce').solve() when declarations are present; the
    keyword arguments are the explicit (call-time) form of the
    declarations and register the components exactly as the declare_*
    functions do. `options` is a mapping of solver options applied to
    the in-process session exactly as the ordinary path would apply
    them; with `warm_start_init_point=yes` among them, the initial
    multipliers come from the model's `dual` / `ipopt_zL_in` /
    `ipopt_zU_in` suffixes (see `_warm_start_from_suffixes`). Returns a
    Pyomo SolverResults, like an ordinary solve.

    `capture`, when a mutable mapping is passed, is filled in with the
    raw outcome of the solve -- the primal iterate, the engine's `info`
    dict (multipliers included), the `.col`/`.row` name lists, the
    surgery alias map and the elapsed solve time. This path returns a
    *legacy* SolverResults because that is what the legacy interface
    needs; `pyomo_pounce.v2` builds a `pyomo.contrib.solver` `Results`
    and solution loader from the same solve, and needs the multipliers
    and row order to do it. Populated for a failed solve too, which is
    exactly when the session is dropped and there is nothing else left
    to read the outcome from. Ignored when None (the default), so the
    legacy path pays nothing for it."""
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

    if sens_params:
        # Call-time declarations are solve-local, so their surgery runs
        # on a clone built for this one call and thrown away, exactly
        # as every solve once did.
        si = SensitivityInterface(model, clone_model=True)
        si.setup_sensitivity(eff_params)
        clone = si.model_instance
        moved_bounds = _reformulate_param_bounds(clone)
    elif eff_params:
        # Declared params: the model already carries a defining
        # equality per param, as written or conformed at declaration,
        # so it solves as written. No clone, no surgery.
        si = None
        clone = model
        moved_bounds = dict(reg.moved_bounds)
        # A declared fixed Var has no live Param behind its pin, so the
        # pin is refreshed from the Var's current value here, every
        # solve: set_value and fix both keep tracking, and a Var the
        # caller re-fixed is unfixed again so the NL writer does not
        # fold it out from under its own pin row.
        for pdata, con, _k in reg.pin_records:
            if pdata.is_variable_type():
                if pdata.fixed:
                    pdata.unfix()
                con.set_value(pdata == float(pyo.value(pdata)))
    else:
        # estimation-only: nothing to pin, solve the model as written
        si = None
        clone = model
        moved_bounds = {}

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
        # The .row file lists the m constraint rows in nl order and
        # then appends the objective's name -- it is a row file, not a
        # constraint file. Trim to the constraint rows here so every
        # consumer of con_names sees rows only: the name->row index
        # feeds the pin map, mult_entry, and the warm-start suffix
        # reader alike, and an objective-keyed lookup would otherwise
        # return row m and index one past the end of the multiplier
        # vector. Surgery on a declared model aliases the objective
        # too, so the objective's name does reach these lookups.
        con_names = con_names[:int(nl.m)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bridge = _NlBridge(nl)
    prob = pounce.Problem(nl.n, nl.m, bridge,
                          lb=nl.x_l, ub=nl.x_u, cl=nl.g_l, cu=nl.g_u)
    if not tee:
        # Pyomo convention is silence unless tee=True; print_level 0 makes
        # the engine emit nothing at all.
        prob.add_option("print_level", 0)
    # sens_covariance() classifies bound activity off this solve's iterate
    # (Solver.classify_activity), which requires slacks measured against
    # the user's own bounds: bound relaxation (default 1e-8) would shift
    # every slack the classifier reads. Set BEFORE the user options so
    # an explicit bound_relax_factor still wins; sens_covariance() then
    # refuses with its clean error rather than classifying shifted
    # slacks.
    prob.add_option("bound_relax_factor", 0.0)
    # user options land after the defaults so an explicit print_level
    # (or anything else) wins
    for key, val in (options or {}).items():
        prob.add_option(key, val)
    con_alias = _replaced_aliases(clone, si)
    # gh #483: user scaling from the model's `scaling_factor` Suffix. The
    # ASL path gets this for free -- the writer emits the suffix as `.nl`
    # `S4`/`S5`/`S6` segments and the solver reads them -- but this path
    # hands pounce evaluator callbacks, with no `.nl` in between, so the
    # Suffix has to be translated into `set_problem_scaling` vectors here.
    # Read from `model` (not the surgery clone) and mapped through
    # `con_alias`, exactly as the warm-start suffixes are. Installing it
    # unconditionally is safe: `nlp_scaling_method` decides whether the
    # engine looks, so a tagged model solved without `user-scaling`
    # behaves as before.
    #
    # Variable entries go in alongside the row ones (gh #486 stage 3):
    # the core applies them as a change of variables and the
    # sensitivity accessors carry the factors back out, so this path
    # honors the same Suffix the ASL one does.
    if user_scaling_requested(options):
        warn_if_no_suffix(model)
    scaling = problem_scaling(model, con_names, con_alias, var_names)
    if scaling is not None:
        obj_scale, g_scale, x_scale = scaling
        prob.set_problem_scaling(obj_scale, x_scaling=x_scale,
                                 g_scaling=g_scale)
    warm = {}
    if _warm_start_requested(options):
        warm = _warm_start_from_suffixes(
            model, var_names, con_names, nl, con_alias)
    solver = pounce.Solver(prob)
    if tee:
        # At the default print_level the engine emits its own banner (via
        # print_banner), problem statistics, iteration table, and summary;
        # _stream_solve tails them to sys.stdout live and times the solve
        # alone (excluding banner/stream overhead).
        (x, info), solve_secs = _stream_solve(
            solver, np.asarray(nl.x0), **warm)
    else:
        t_solve = time.perf_counter()
        x, info = solver.solve(np.asarray(nl.x0), **warm)
        solve_secs = time.perf_counter() - t_solve

    status_msg = str(info.get("status_msg", ""))
    tc, ss = _STATUS_RESULT.get(
        status_msg, (TerminationCondition.error, SolverStatus.error))

    if capture is not None:
        # Everything the v2 interface needs to build its own Results and
        # solution loader, recorded before the non-converged early return
        # below (which drops the session) so a failed solve is reported
        # there as fully as a successful one.
        capture.update(
            x=np.asarray(x), info=info, status_msg=status_msg,
            var_names=var_names, con_names=con_names, con_alias=con_alias,
            n=int(nl.n), m=int(nl.m), solve_secs=solve_secs,
            # +1 / -1: the loader reports multipliers against the
            # objective the model states, and `read_nl` handed the
            # engine the negation of a maximization
            obj_sign=_objective_sign(nl))

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
        it = info.get("iter_count")
        if it is not None:
            stats = results.solver.statistics
            stats.black_box.number_of_iterations = int(it)
        results.problem.number_of_objectives = 1
        results.problem.number_of_constraints = int(nl.m)
        results.problem.number_of_variables = int(nl.n)
        # objective bounds, like the .sol path: both set to the final
        # value, in the sense the model states it (`read_nl` negated a
        # maximization before the engine reported `obj_val`)
        obj_val = info.get("obj_val")
        if obj_val is not None:
            val = _objective_sign(nl) * float(obj_val)
            results.problem.upper_bound = val
            results.problem.lower_bound = val
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
        # sens_jacobian()/sens_solution()/sens_covariance() would
        # silently answer from
        # the stale solve. With the session cleared they raise their
        # usual "no sensitivity session" error. Note the Feasible_Point_Found
        # asymmetry: the engine's on_converged callback fires only for
        # Solve_Succeeded / Solved_To_Acceptable_Level, so a square-problem
        # feasible point reports termination_condition=optimal, status=ok yet
        # has converged=False and lands here -- no KKT factorization is
        # retained, so its session is dropped even though the solve succeeded.
        # See `_STATUS_RESULT` for why widening the callback gate is the wrong
        # way to close that gap.
        reg.session = None
        failed_var_data = []
        for name, val in zip(var_names, np.asarray(x)):
            ov = model.find_component(name)
            failed_var_data.append(ov)
            if ov is not None:
                ov.set_value(float(val), skip_validation=True)
        # the final iterate's multipliers, on the same terms as the
        # primals beside them: the .sol route loads a non-converged
        # solution's suffixes too, and a caller reading `m.dual` after a
        # maxIterations exit should see the iterate, not the previous
        # solve's answer left standing
        _load_result_suffixes(model, info, nl, failed_var_data, con_names,
                              con_alias, moved_bounds)
        return build_results()

    # name -> row maps, built once here and handed to the session below.
    # Every pin / fitted / residual lookup that follows, and every later
    # query, would otherwise scan the whole name list (gh #365). Built
    # after the non-converged early return, which has no use for them.
    var_row = _row_index(var_names)
    con_row = _row_index(con_names)

    pins = ComponentMap()
    pin_coefs = ComponentMap()
    pin_bases = ComponentMap()
    if si is not None:
        block = clone.component(SensitivityInterface.get_default_block_name())
        for i, (var, clone_param, list_idx, comp_idx) in enumerate(
                block._sens_data_list):
            con = block.paramConst[i + 1]
            orig_comp = eff_params[list_idx]
            orig_data = (orig_comp if not orig_comp.is_indexed()
                         else orig_comp[comp_idx])
            pins[orig_data] = con_row[con.name]
    elif reg.pin_records:
        for pdata, con, coeff in reg.pin_records:
            row = con_row.get(con.name)
            if row is None:
                raise RuntimeError(
                    f"sens: the defining equality {con.name} recorded "
                    f"for declared param {pdata.name} is not among the "
                    "solved model's rows, so it was deactivated or "
                    "removed after declaration. Re-declare on the "
                    "current model.")
            pins[pdata] = row
            pin_coefs[pdata] = float(coeff)
            pin_bases[pdata] = float(pyo.value(pdata))
    # con_alias was built before the solve (the warm-start reader needs
    # it); the session stores the same map

    session = _Session(model, nl, solver, var_names, con_names, pins,
                       con_alias, var_row=var_row, con_row=con_row)
    session.base_x = np.asarray(x)
    # the engine always reports obj_val (NaN when it evaluated nothing),
    # and it is eval_f on this model's own bridge at the final iterate --
    # unscaled, in the model's objective units. `read_nl` negates a
    # `maximize` objective, so the sign is what makes this equal to
    # pyo.value(objective) an instant after the solve on BOTH senses.
    session.base_obj = session.obj_sign * float(
        info.get("obj_val", float("nan")))
    session.moved_bounds = moved_bounds
    session.pin_coefs = pin_coefs
    session.pin_bases = pin_bases

    # fitted parameters: their rows in the primal vector
    session.fit_rows = ComponentMap()
    for comp in eff_fitted:
        for vd in _iter_data(comp):
            session.fit_rows[vd] = var_row[vd.name]

    # residual groups: member rows per group key (None = the common pool)
    session.res_rows = {}
    for container, group in eff_residuals:
        rows = [var_row[rd.name] for rd in _iter_data(container)]
        session.res_rows.setdefault(group, []).extend(rows)

    # load the solution back onto the ORIGINAL model's variables (when the
    # solve ran on a clone; in the estimation-only path clone IS model and
    # this simply refreshes the same variables), and keep the resolved
    # objects: this is the one name resolution the session pays.
    # Registration comes after the capture, so a load loop that raises
    # leaves no session behind rather than one whose var_data is None.
    var_data = []
    for name, val in zip(var_names, session.base_x):
        ov = model.find_component(name)
        var_data.append(ov)
        if ov is not None:
            ov.set_value(float(val), skip_validation=True)
    session.var_data = var_data
    _load_result_suffixes(model, info, nl, var_data, con_names, con_alias,
                          moved_bounds)

    reg.session = session

    # consistency check: declared residuals should reproduce the objective.
    # Against the objective the solver MINIMIZED, not the one the model
    # states -- "the objective is the plain sum of squares" is a claim
    # about a quantity being driven down, and `maximize -SSR` is the same
    # least-squares problem spelled the other way. `session.base_obj` is in
    # the model's own sense (see `objective_sign`), so undo that here;
    # comparing a signed SSR against it would warn on every maximize
    # spelling of a fit that has nothing wrong with it.
    if session.res_rows:
        ssr = sum(float(session.base_x[r]) ** 2
                  for rows in session.res_rows.values() for r in rows)
        obj_val = session.obj_sign * session.base_obj
        if np.isfinite(obj_val) and abs(ssr - obj_val) > 1e-6 * max(
                1.0, abs(obj_val)):
            warnings.warn(
                "sens_solve: the declared residuals give SSR = "
                f"{ssr:.6g} but the objective value is {obj_val:.6g}"
                f"{' (minimized sense)' if session.obj_sign < 0 else ''}."
                " sens_covariance() assumes the objective is the plain sum of "
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
            "SolverFactory('pounce'), SolverFactory('pounce_v2') or the "
            "contrib SolverFactory('pounce') first")
    return reg.session


def _param_pin(session, param_data):
    if param_data not in session.pins:
        raise ValueError(f"{param_data.name} was not declared with "
                         "declare_sens_param before the solve")
    return session.pins[param_data]


class Jacobian:
    """Derivatives d(target*)/d(param) for one or more targets/parameters.
    Targets are variables (primal sensitivities), equality constraints
    (multiplier sensitivities), or the objective (the total derivative
    df/dp).

    Access with g[target_data, param_data] (either order); when one side is
    a single component, g[data] works. to_dataframe() gives the full
    Jacobian (rows = targets, columns = parameters)."""

    def __init__(self, session, targets, params):
        self._session = session
        self._targets = list(targets)
        self._params = list(params)
        self._tset = set(id(t) for t in self._targets)
        self._pset = set(id(p) for p in self._params)

    def _check_objective(self, td):
        """The objective target must be the one this session solved.

        Without this a second, deactivated objective left on the model --
        the ordinary way a script switches between formulations -- reads
        as a valid target and is answered with the gradient of the
        objective that *was* solved, under the name of the one that was
        not. Cheap to check, and the failure it prevents is silent.
        """
        active = [o for o in self._session.model.component_data_objects(
            pyo.Objective, active=True, descend_into=True)]
        if not any(o is td for o in active):
            raise ValueError(
                f"{td.name}: not the active objective of the solved model. "
                "sens_jacobian(of=<Objective>) differentiates the objective "
                "the solve minimized; " + (
                    f"this model's is {active[0].name}."
                    if len(active) == 1 else
                    f"this model has {len(active)} active objectives "
                    f"({[o.name for o in active]})."))

    def _entry(self, td):
        if td.ctype is Constraint:
            return self._session.mult_entry(td.name)
        # var_entry is full-x; the column being indexed is the factor's
        # KKT vector, so it needs the var-x row -- the same translation
        # mult_entry already does for the y_c block
        return self._session.primal_row(
            self._session.var_entry(td.name), f"sens_jacobian({td.name})")

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
        is the negation of the raw row -- without this, `sens_jacobian(m.con,
        wrt=m.p)` disagrees in sign with a finite difference of
        `m.dual[m.con]` taken across a re-solve.
        """
        return -1.0 if td.ctype is Constraint else 1.0

    def _value(self, td, pd):
        col = self._session.column(_param_pin(self._session, pd))
        if td.ctype is pyo.Objective:
            # Not a row of the factor at all: `df/dp` is a scalar contracted
            # over the whole step, so it has no `_entry` and no convention
            # sign to apply (gh#878).
            self._check_objective(td)
            return self._session.total_objective_derivative(col)
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




def sens_jacobian(of=None, *, wrt, max_pdpert=None):
    """d(of*)/d(wrt).

    of: what the derivative is about (the target rows) -- a Var
    (primal sensitivity), an equality Constraint (its multiplier's
    sensitivity), or the model's Objective (the TOTAL derivative
    df/dp, gh#878); data object or container; omit for all model
    variables. wrt: the differentiation variable, a declared Param
    (data or container).

    The objective target answers

        df/dp = df/dp|_x + sum_i (df/dx_i)(dx_i/dp)

    -- the quantity an outer loop, a design-of-experiments score or a
    "which parameter is my objective most exposed to" question wants.
    Both halves are included: a parameter that appears in the objective
    contributes its explicit partial as well as its effect through the
    solution. Only the active objective of the solved model is accepted.

    Scalar of and scalar wrt -> float. Anything else -> a Jacobian
    object: g[target, param], or g.to_dataframe() for the full
    Jacobian.

    At a degenerate base point the solution has two one-sided
    derivatives and this call has no direction to choose between them,
    so it returns the one-sided value the held factorization leans
    toward and warns. `sens_solution()` computes the directional derivative
    for the perturbation it is given.

    max_pdpert refuses rather than answering when the converged KKT
    factor carries an inertia correction larger than the value given,
    since every derivative here inverts that factor and a perturbed one
    answers for a nearby problem."""
    session = _session_for(wrt)
    _check_margins(None, max_pdpert, "sens_jacobian")
    _refuse_on_pdpert(session, max_pdpert, "sens_jacobian")
    weak = _weakly_active(session)
    if weak:
        warnings.warn(
            "sens_jacobian: the base point is degenerate: "
            f"{[f'{nm} ({side})' for nm, side in weak]} sit on a bound "
            "with a multiplier of the same order as the slack, so the "
            "solution has two one-sided derivatives there and this "
            "value is one side's. sens_solution() computes the directional "
            "derivative for the perturbation it is given.")
    params = list(_iter_data(wrt))
    if of is None:
        targets = [v for v in session.var_data if v is not None]
    else:
        targets = list(_iter_data(of))
    if of is not None and not of.is_indexed() and len(params) == 1:
        return Jacobian(session, targets, params)._value(
            targets[0], params[0])
    return Jacobian(session, targets, params)


def _perturbation_deltas(session, perturb):
    """Pin constraints and right-hand-side shifts for a perturbation.

    The shift is measured from the pin constraint's stored right-hand
    side, which holds the Param's value at the solve, not from the
    Param's current value on the model.
    """
    items = perturb.items() if hasattr(perturb, "items") else perturb
    pin_idx, deltas = [], []
    for comp, newval in items:
        for pd in _iter_data(comp):
            nv = newval[pd.index()] if comp.is_indexed() and hasattr(
                newval, "__getitem__") else newval
            pin = _param_pin(session, pd)
            pin_idx.append(pin)
            coeff = session.pin_coefs.get(pd)
            if coeff is None:
                # a surgery row `var == p`: the row's right-hand side IS
                # the param's solve-time value
                deltas.append(float(nv) - float(session.nl.g_l[pin]))
            else:
                # a defining equality as the user wrote it: moving the
                # param by dp moves the row by coeff * dp, so the
                # right-hand side shifts by the negative of that
                deltas.append(-coeff * (float(nv) - session.pin_bases[pd]))
    return pin_idx, deltas














# ── step diagnostics ──────────────────────────────────────────────────────────


















# ── parameter covariance ──────────────────────────────────────────────────────





























def _resolve_of(session, of, who):
    """Normalize of= into the block: an ordered list of variable data
    objects with their full-x (.col) rows. Accepted forms: None (the
    declared fitted block, exactly the prior behavior), a Var component
    (scalar or indexed: every member), an indexed slice (m.x[2, :]), a
    (Var, iterable) pair (var[t] for t in the iterable), a single
    VarData, or an iterable mixing any of these. Duplicates are an
    error: a repeated coordinate makes the block singular by
    construction."""
    if of is None:
        params = list(session.fit_rows.keys())
        if not params:
            raise RuntimeError(
                f"{who}: no fitted parameters were declared; flag the "
                "fitted variables with declare_sens_fitted() before the "
                "solve, or select a block explicitly with of=")
        return params, [session.fit_rows[p] for p in params]

    try:
        from pyomo.core.base.indexed_component_slice import (
            IndexedComponent_slice,
        )
    except ImportError:                    # pragma: no cover
        IndexedComponent_slice = ()

    def leaves(obj):
        if (isinstance(obj, tuple) and len(obj) == 2
                and hasattr(obj[0], "ctype")
                and not hasattr(obj[1], "ctype")):
            # a (Var, iterable) pair; a tuple of two Vars falls through
            # to the generic-iterable branch instead of the second Var
            # being consumed as an index set (review of gh #466)
            comp, idx = obj
            for t in idx:
                yield comp[t]
            return
        if isinstance(obj, IndexedComponent_slice):
            # a slice PROXIES attribute access to its members, so the
            # duck-typing below would misfire; iterate it directly
            for v in obj:
                yield v
            return
        if hasattr(obj, "ctype"):
            values = getattr(obj, "values", None)
            if callable(values):           # a component: every member
                for v in values():
                    yield v
            else:                          # a data object
                yield obj
            return
        if isinstance(obj, str):
            raise TypeError(
                f"{who}: of= takes variables, not names; got {obj!r}")
        try:
            it = iter(obj)                 # slice or plain iterable
        except TypeError:
            raise TypeError(
                f"{who}: of= element {obj!r} is not a Pyomo variable "
                "or an iterable of them") from None
        for el in it:
            yield from leaves(el)

    params, rows, seen = [], [], set()
    for v in leaves(of):
        name = getattr(v, "name", None)
        if name is None:
            raise TypeError(
                f"{who}: of= element {v!r} is not a Pyomo variable")
        try:
            r = session.var_entry(name)
        except ValueError as e:
            raise ValueError(f"{who}: of= member {e}") from None
        if id(v) in seen:
            raise ValueError(
                f"{who}: of= lists {name} twice; a repeated coordinate "
                "makes the block singular by construction")
        seen.add(id(v))
        params.append(v)
        rows.append(r)
    if not params:
        raise ValueError(f"{who}: of= resolved to an empty block")
    return params, rows
















# ── the Pyomo API over pounce.sensitivity ─────────────────────────────────────
#
# Everything below resolves Pyomo components to rows, calls the core, and
# keys the answer back by component. The numerics live in
# `pounce.sensitivity`, so a bare-NL or CasADi caller reaches the same
# code without Pyomo installed; `who=` keeps these function names in the
# messages a Pyomo user sees, and `_HINTS` keeps the declarations those
# messages point at spelled the Pyomo way.

#: How this layer spells the declarations the core's diagnostics name.
_HINTS = {
    "fitted": "declare_sens_fitted()",
    "residual": "declare_sens_residual()",
    "residual_group": "declare_sens_residual(..., group=...)",
}

_NO_SESSION_PARAM = (
    "no sensitivity session: declare_sens_param() then solve with "
    "SolverFactory('pounce'), SolverFactory('pounce_v2') or the "
    "contrib SolverFactory('pounce') first")

_NO_SESSION_FIT = (
    "no sensitivity session: declare_sens_fitted() (and optionally "
    "declare_sens_residual()), or sens_retain_kkt() for "
    "of= queries with nothing declared, then solve with "
    "SolverFactory('pounce'), SolverFactory('pounce_v2') or the "
    "contrib SolverFactory('pounce') first")


def _model_session(model, message):
    reg = model.__dict__.get(_REG)
    session = reg.session if reg else None
    if session is None:
        raise RuntimeError(message)
    return session


def sens_solution(model, perturb, clamp=True, mode="linear",
                  predictor_iter=16, degeneracy="directional",
                  degeneracy_iter=None, corrector_iter=0, bound_eps=None,
                  max_pdpert=None):
    """First-order estimate of the solution at perturbed parameter values.

    perturb: pairs of (declared Param, new value) -- a list of tuples or a
    ComponentMap (plain dicts don't work: Pyomo components are unhashable).
    Returns a read-only `SolutionMap` {original var data: estimated
    value}, keyed by the component data objects themselves. Values are
    clamped to variable bounds (with a warning) unless clamp=False.

    See `pounce.sensitivity.solution` for what every other argument does;
    this resolves the perturbation to pin rows and shifts and hands it
    the same session.
    """
    session = _model_session(model, _NO_SESSION_PARAM)
    pin_idx, deltas = _perturbation_deltas(session, perturb)
    x_new = _core_solution(
        session, pin_idx, deltas, clamp=clamp, mode=mode,
        predictor_iter=predictor_iter, degeneracy=degeneracy,
        degeneracy_iter=degeneracy_iter, corrector_iter=corrector_iter,
        bound_eps=bound_eps, max_pdpert=max_pdpert, who="sens_solution")
    keys, index_of = session.solution_keys()
    return SolutionMap(keys, index_of, x_new)


def sens_solution_report(model, perturb, max_iter=None,
                         degeneracy="directional", degeneracy_iter=None,
                         corrector_iter=0, mode="linear", predictor_iter=16,
                         bound_eps=None, max_pdpert=None,
                         refine_activity=True):
    """Report what `sens_solution()`'s step does about the bounds.

    Takes the same perturbation argument `sens_solution()` takes and
    returns a `SolutionReport`; see `pounce.sensitivity.solution_report`.
    `crossed` and `crossed_rows` come back as `ComponentMap`s keyed by
    the original model's data objects. **`activity`, `row_status` and
    `refined` do not** -- they stay keyed by solve-space names, and that
    asymmetry is deliberate rather than an oversight (raised in round 6
    of #889, which is where it got written down).

    The split is by what the field is: `crossed` and `crossed_rows`
    name *components the caller then does something with*, so handing
    back data objects saves a lookup. `activity` and `row_status` are a
    classification keyed the way the classifier keys it, and `refined`
    annotates `activity` -- it says which of those entries the reduced
    rule moved and from what. Remapping `refined` alone would key it
    differently from the field it explains, which is worse than the
    asymmetry it would remove.

    `refine_activity` re-classifies the entries the cheap classifier
    could not call, using the reduced curvature, and records what moved
    under `SolutionReport.refined`. On by default, because a coupled
    kink -- routine on a collocation model -- is "ambiguous" to the
    cheap rule at every tolerance, and reading that class as "probably
    not a kink" is what shipped gh#763. It costs one back-solve per
    ambiguous entry and nothing when there are none -- so the price is
    set by the ambiguous population rather than the model size. Measured
    in review of #889 on a 62k-variable Radau collocation column: 675
    ambiguous entries at ~29 ms each, 0.67 s -> 20.2 s. Passing
    `refine_activity=False` buys that back and leaves the cheap class
    standing, which on a collocation model means genuine kinks stay in
    "ambiguous"; `pounce.sensitivity`'s `solution_report` docstring
    carries the full trade-off.
    """
    session = _model_session(model, _NO_SESSION_PARAM)
    pin_idx, deltas = _perturbation_deltas(session, perturb)
    return _core_solution_report(
        session, pin_idx, deltas, max_iter=max_iter, degeneracy=degeneracy,
        degeneracy_iter=degeneracy_iter, corrector_iter=corrector_iter,
        mode=mode, predictor_iter=predictor_iter, bound_eps=bound_eps,
        max_pdpert=max_pdpert, refine_activity=refine_activity,
        who="sens_solution_report")


def sens_active_set_changes(model, perturb, predictor_iter=16,
                            degeneracy="directional", degeneracy_iter=None,
                            max_pdpert=None):
    """The active-set changes `sens_solution(mode="path")` applies, in order.

    Takes the same perturbation argument `sens_solution()` takes and
    returns a tuple of `ActiveSetChange`; see
    `pounce.sensitivity.active_set_changes`.
    """
    session = _model_session(model, _NO_SESSION_PARAM)
    pin_idx, deltas = _perturbation_deltas(session, perturb)
    return _core_active_set_changes(
        session, pin_idx, deltas, predictor_iter=predictor_iter,
        degeneracy=degeneracy, degeneracy_iter=degeneracy_iter,
        max_pdpert=max_pdpert, who="sens_active_set_changes")


def sens_covariance(model, sigma_sq=None, n_data=None, hessian="lagrangian",
                    of=None, max_pdpert=None):
    """Asymptotic covariance of the fitted parameters of a
    least-squares problem, from ONE ordinary solve.

    Workflow: declare the fitted variables with declare_sens_fitted (they
    stay free), optionally declare the residual container(s) with
    declare_sens_residual, solve with SolverFactory('pounce'), then call
    sens_covariance(model) with no further information.

    of= selects the block explicitly instead: a Var, a slice, a
    (Var, iterable) pair, a VarData, or an iterable mixing them. See
    `pounce.sensitivity.covariance` for the statistics and for what
    sigma_sq=, n_data=, hessian= and max_pdpert= do.
    """
    session = _model_session(model, _NO_SESSION_FIT)
    params, rows = (None, None) if of is None else _resolve_of(
        session, of, "sens_covariance")
    return _core_covariance(
        session, params, rows, sigma_sq=sigma_sq, n_data=n_data,
        hessian=hessian, max_pdpert=max_pdpert, who="sens_covariance",
        hints=_HINTS)


def sens_information(model, hessian="lagrangian", of=None, max_pdpert=None):
    """The information matrix of the fitted parameters: the reduced
    Hessian `sens_covariance()` inverts.

    Same block selection as `sens_covariance()`; see
    `pounce.sensitivity.information`.
    """
    session = _model_session(model, _NO_SESSION_FIT)
    params, rows = (None, None) if of is None else _resolve_of(
        session, of, "sens_information")
    return _core_information(
        session, params, rows, hessian=hessian, max_pdpert=max_pdpert,
        who="sens_information", hints=_HINTS)
