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

One deliberate divergence from pyomo.contrib.sensitivity_toolbox: a
declared Param appearing in a Var's BOUND is rewritten as a constraint
before the solve, so its sensitivity is real rather than zero. The
toolbox substitutes declared Params in constraint expressions only, and
leaves such a bound frozen at its pre-perturbation value. On the clone
that is solved the bound is dropped, so m.x.ub reads None there and the
NL carries the no-bound sentinel for that row.
"""
import codecs
import os
import shutil
import sys
import tempfile
import threading
import time
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.sensitivity_toolbox.sens import SensitivityInterface
from pyomo.core.base.constraint import Constraint
from pyomo.core.expr import identify_mutable_parameters
from pyomo.core.expr.visitor import replace_expressions
from pyomo.opt import SolverResults, SolverStatus, TerminationCondition

from pyomo_pounce.scaling import (
    problem_scaling,
    user_scaling_requested,
    warn_if_no_suffix,
)

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
        self.retain = False       # keep the factor with no declaration
        self.session = None

    def __deepcopy__(self, memo):
        import copy
        new = _Registry()
        memo[id(self)] = new
        new.params = [copy.deepcopy(p, memo) for p in self.params]
        new.fitted = [copy.deepcopy(p, memo) for p in self.fitted]
        new.residuals = [(copy.deepcopy(r, memo), g)
                         for r, g in self.residuals]
        new.retain = self.retain
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


def retain_kkt(model):
    """Keep the KKT factorization after the next solve with NOTHING
    declared (covariance roadmap item 4). The factor the solve computes
    anyway is retained for post-solve queries, so `covariance(model,
    wrt=block)` and `information(model, wrt=block)` work on any block
    without a declared default: the MHE case, where the arrival state
    and the parameters are each queried by wrt= and neither is THE
    fitted set. `covariance(model)` with no block stays an error (there
    is no default to reduce onto), and a solve without this call and
    without declarations pays nothing, exactly as before.

    The retention policy in one place: the factor is kept if anything
    is declared (declare_sens_param / declare_fitted /
    declare_residual), or if retain_kkt() was called; and a
    Covariance/Information result whose lazy conditioned_on has not
    been read keeps the session alive through its pending computation
    until first access. release_kkt(model) drops the held factor on
    demand.

    Like any declaration, this routes the solve through the
    in-process sensitivity path, whose solve() surface is not
    keyword-identical to the ordinary subprocess path (for example,
    load_solutions=False is not honored there): adding it to an
    existing script changes how the solve runs, not just what is
    kept."""
    _registry(model).retain = True


def release_kkt(model):
    """Drop the held KKT factorization now, freeing its memory.

    The exit of the retention story, for the current factor only:
    declarations and a prior retain_kkt() are untouched and apply to
    the NEXT solve, which keeps its factor again. After release the
    accessors raise their no-session error until another solve.

    Release drops the MODEL's hold, not a result's, and two kinds of
    result hold their own reference to the session: a Covariance or
    Information whose conditioned_on is still pending (until the
    attribute is read), and a Gradient handed back by gradient()
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
    side that was not moved. covariance() classifies these rows through the
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
            # covariance() can still see where the bound was
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

def _row_index(names):
    """{name: position} for a .col/.row name list.

    The NL writer emits unique symbolic labels, so first-wins and last-wins
    agree; enumerate order matches `list.index` either way.
    """
    return {nm: i for i, nm in enumerate(names)}


class _Session:
    def __init__(self, model, nl, solver, var_names, con_names, pins,
                 con_alias, var_row=None, con_row=None):
        self.model = model            # original model
        self.nl = nl
        self.solver = solver
        self.var_names = var_names    # .col order = x-vector order
        self.con_names = con_names    # .row order = g-vector order
        # Reverse maps for the two orders above. Every query resolves a
        # component name to its row, and a list scan makes that O(n) per
        # lookup -- quadratic for gradient(target=None).to_dataframe(),
        # which asks for every variable (gh #365). Built once here, or
        # reused from the caller when it has already built them.
        # `is None`, not truthiness: an unconstrained model's con_row is a
        # legitimately empty dict, which `or` would discard and rebuild.
        self._var_row = _row_index(var_names) if var_row is None else var_row
        self._con_row = _row_index(con_names) if con_row is None else con_row
        self.pins = pins              # ComponentMap: param data -> pin row
        self.con_alias = con_alias    # original con name -> clone row name
        self.base_x = None
        # Objective value at the solve. NaN, not None, is the "never
        # computed" sentinel: that is the convention the engine itself
        # uses for info["obj_val"] (pounce-py's problem.rs seeds
        # final_obj with NaN precisely because 0.0 is an ordinary
        # objective value and cannot signal it), so one isfinite check
        # covers both an unset session and a solve that evaluated
        # nothing.
        self.base_obj = float("nan")
        self.moved_bounds = {}        # var name -> (lb, ub) moved to rows
        self._columns = {}            # pin row -> full KKT-space column
        self._primal_rows = None      # full-x -> KKT row, lazily fetched

    def orig_var(self, name):
        return self.model.find_component(name)

    def _primal_row_map(self):
        if self._primal_rows is None:
            self._primal_rows = self.solver.primal_rows(
                list(range(len(self.var_names))))
        return self._primal_rows

    def primal_row(self, full_idx, what):
        """The KKT-factor row of a user-space (`.col`) variable index.

        `.col` order -- what `var_entry`, `fit_rows` and `res_rows` all
        hold -- is the user's FULL-x space. The factor's `x` block is
        the algorithm's var-x space, which drops every variable the
        solve removed as fixed (`lb == ub` under the default
        `fixed_variable_treatment=make_parameter`). The two spaces
        coincide exactly when the model has no fixed variable, which is
        every model in the test suite and most models anywhere, so
        indexing the factor with a full-x row is invisible until it is
        not: one fixed variable shifts every later column and the
        back-solve quietly returns a NEIGHBOURING variable's
        sensitivity, a plausible number with nothing wrong-looking
        about it. Route every factor index through here.
        """
        row = self._primal_row_map()[full_idx]
        if row is None:
            raise ValueError(
                f"{what}: {self.var_names[full_idx]} was removed from the "
                "solve as a fixed variable (its bounds are equal), so it "
                "has no row in the KKT factor and no sensitivity "
                "information. Give it distinct bounds to keep it in the "
                "solve.")
        return row

    def scatter_x(self, dx_var):
        """Full-x vector from an algorithm-space (var-x) one.

        `parametric_step` truncates its result to the factor's `x`
        block, so it is var-x while `base_x`, `nl.x_l/x_u` and
        `var_names` are all full-x. Variables the solve removed as
        fixed get 0: a fixed variable does not move.
        """
        out = np.zeros(len(self.var_names))
        for full_idx, row in enumerate(self._primal_row_map()):
            if row is not None:
                out[full_idx] = dx_var[row]
        return out

    def column(self, pin_idx):
        """Full KKT-space derivative column for a unit perturbation."""
        if pin_idx not in self._columns:
            self._columns[pin_idx] = np.asarray(
                self.solver.parametric_step_full([pin_idx], [1.0]))
        return self._columns[pin_idx]

    def var_entry(self, name):
        # ValueError, not the dict's KeyError: this used to be a list scan
        # and callers (and the message a user sees) expect ValueError.
        try:
            return self._var_row[name]
        except KeyError:
            raise ValueError(
                f"{name}: not a variable of the solved model") from None

    def mult_entry(self, con_name):
        # the sensitivity surgery replaces user constraints with copies on
        # its data block; translate the original name to the clone's row
        con_name = self.con_alias.get(con_name, con_name)
        try:
            g = self._con_row[con_name]
        except KeyError:
            raise ValueError(
                f"{con_name}: not a constraint of the solved model") from None
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

    Entries the user did not supply are seeded NaN, the session's
    "unseeded" marker: the warm-start initializer substitutes its own
    resolved defaults (`bound_mult_init_val` for bound multipliers, 0
    for equality duals), so partial seeds never turn into the zero
    certificate an ASL-style dense array forces, and the defaults live
    in one place. An explicit zero is honored, then floored at
    `warm_start_mult_bound_push` exactly as a round-tripped inactive
    multiplier is.
    """
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
                y[r] = -float(val)      # AMPL marginal -> internal lambda
    for sfx_name, arr, sign in (("ipopt_zL_in", zl, 1.0),
                                ("ipopt_zU_in", zu, -1.0)):
        sfx = model.component(sfx_name)
        if isinstance(sfx, pyo.Suffix):
            for vd, val in sfx.items():
                r = var_row.get(vd.name)
                if r is not None:
                    arr[r] = sign * float(val)
    return {"lagrange": y, "zl": zl, "zu": zu}


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
    for gradient()/estimate()/covariance(). Called automatically by
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

    if eff_params:
        # pinned inputs need the sensitivity-toolbox surgery (on a clone)
        si = SensitivityInterface(model, clone_model=True)
        si.setup_sensitivity(eff_params)
        clone = si.model_instance
        moved_bounds = _reformulate_param_bounds(clone)
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
    # covariance() classifies bound activity off this solve's iterate
    # (Solver.classify_activity), which requires slacks measured against
    # the user's own bounds: bound relaxation (default 1e-8) would shift
    # every slack the classifier reads. Set BEFORE the user options so
    # an explicit bound_relax_factor still wins; covariance() then
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
            n=int(nl.n), m=int(nl.m), solve_secs=solve_secs)

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

    # name -> row maps, built once here and handed to the session below.
    # Every pin / fitted / residual lookup that follows, and every later
    # query, would otherwise scan the whole name list (gh #365). Built
    # after the non-converged early return, which has no use for them.
    var_row = _row_index(var_names)
    con_row = _row_index(con_names)

    pins = ComponentMap()
    if si is not None:
        block = clone.component(SensitivityInterface.get_default_block_name())
        for i, (var, clone_param, list_idx, comp_idx) in enumerate(
                block._sens_data_list):
            con = block.paramConst[i + 1]
            orig_comp = eff_params[list_idx]
            orig_data = (orig_comp if not orig_comp.is_indexed()
                         else orig_comp[comp_idx])
            pins[orig_data] = con_row[con.name]
    # con_alias was built before the solve (the warm-start reader needs
    # it); the session stores the same map

    session = _Session(model, nl, solver, var_names, con_names, pins,
                       con_alias, var_row=var_row, con_row=con_row)
    session.base_x = np.asarray(x)
    # the engine always reports obj_val (NaN when it evaluated nothing),
    # and it is eval_f on this model's own bridge at the final iterate --
    # unscaled, in the model's objective units, i.e. exactly what
    # pyo.value(objective) returns an instant after the solve
    session.base_obj = float(info.get("obj_val", float("nan")))
    session.moved_bounds = moved_bounds

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
            "SolverFactory('pounce'), SolverFactory('pounce_v2') or the "
            "contrib SolverFactory('pounce') first")
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
        # var_entry is full-x; the column being indexed is the factor's
        # KKT vector, so it needs the var-x row -- the same translation
        # mult_entry already does for the y_c block
        return self._session.primal_row(
            self._session.var_entry(td.name), f"gradient({td.name})")

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


def _weakly_active(session):
    """The degenerate bounds at the held solve, cached per session.

    `(variable name, "lower" or "upper")` for every bound the
    classifier could call neither active nor inactive. Empty when the
    solve relaxed its bounds, since the classifier cannot read shifted
    slacks. Cached because the factor is fixed per solve and
    `gradient()` runs in tight loops.
    """
    cached = getattr(session, "_weakly_active_cache", None)
    if cached is None:
        if session.solver.bound_relax_factor:
            cached = []
        else:
            full_of = {row: full
                       for full, row in enumerate(session._primal_row_map())
                       if row is not None}
            cached = [
                (session.var_names[full_of[vr]],
                 "lower" if low else "upper")
                for vr, low in session.solver.weakly_active_bounds()
                if vr in full_of
            ]
        session._weakly_active_cache = cached
    return cached


def gradient(target=None, *, wrt, max_pdpert=None):
    """d(target*)/d(wrt).

    target: a Var (primal sensitivity) or an equality Constraint (its
    multiplier's sensitivity); data object or container; omit for all
    model variables. wrt: a declared Param (data or container).

    Scalar target and scalar wrt -> float. Anything else -> a Gradient
    object: g[target, param], or g.to_dataframe() for the full Jacobian.

    At a degenerate base point the solution has two one-sided
    derivatives and this call has no direction to choose between them,
    so it returns the one-sided value the held factorization leans
    toward and warns. `estimate()` computes the directional derivative
    for the perturbation it is given.

    max_pdpert refuses rather than answering when the converged KKT
    factor carries an inertia correction larger than the value given,
    since every derivative here inverts that factor and a perturbed one
    answers for a nearby problem."""
    session = _session_for(wrt)
    _check_margins(None, max_pdpert, "gradient")
    _refuse_on_pdpert(session, max_pdpert, "gradient")
    weak = _weakly_active(session)
    if weak:
        warnings.warn(
            "gradient: the base point is degenerate: "
            f"{[f'{nm} ({side})' for nm, side in weak]} sit on a bound "
            "with a multiplier of the same order as the slack, so the "
            "solution has two one-sided derivatives there and this "
            "value is one side's. estimate() computes the directional "
            "derivative for the perturbation it is given.")
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
            deltas.append(float(nv) - float(session.nl.g_l[pin]))
    return pin_idx, deltas


def _check_margins(bound_eps, max_pdpert, who):
    """Reject the values the CLI's option surface rejects.

    `sens_bound_eps` and `sens_max_pdpert` are both registered with
    `add_lower_bounded_number_option(..., 0.0, strict)`, so a value the
    CLI turns down is turned down here too. A `bound_eps` of zero
    reinstates the roundoff pinning the floor exists to prevent, a NaN
    makes every "outside a bound" comparison false so the refinement
    pins nothing and still reports settled, and a negative `max_pdpert`
    always refuses with a message that reads as though the factor were
    bad.
    """
    for name, value in (("bound_eps", bound_eps), ("max_pdpert", max_pdpert)):
        if value is None:
            continue
        if not float(value) > 0.0:
            raise ValueError(
                f"{who}: {name} must be a positive number, got {value!r}")


def _bound_margin(session, bound_eps):
    """How far outside a bound a coordinate has to end to count as
    having left it.

The refinement pins against this margin and `estimate()` clamps
    against it. `estimate_report()` measures `crossed` against the same
    number whenever the caller sets one, since deciding it twice is
    what let `refine_stop == "settled"` come back beside a coordinate
    reported 2.0 outside its bound.

    Unset it is how far outside the solve itself was willing to settle,
    floored so an unrelaxed solve does not pin on roundoff. The report
    keeps the floor there instead, because a coordinate the SOLVE left
    outside its bound is one of the things `crossed` exists to name,
    and the relaxation is exactly how far out it is.
    """
    if bound_eps is not None:
        return float(bound_eps)
    return max(abs(session.solver.bound_relax_factor or 0.0), 1e-9)


def _refuse_on_pdpert(session, max_pdpert, who):
    """Refuse when the converged factor carries more inertia correction
    than the caller will accept.

    A non-zero entry means the factorization every sensitivity output
    inverts is not this problem's KKT matrix but a nearby one's, and how
    nearby is what the entries measure. `EstimateReport.perturbations`
    reports them either way, and this is the caller choosing to stop
    rather than to read them.

    The comparison comes from `pounce_sensitivity::pdpert_verdict`,
    which `sens_max_pdpert` reads too, so the two surfaces cannot drift
    apart on what counts as too perturbed. The message is written here
    rather than taken from there, since that one names a CLI option and
    says the sensitivity was skipped, and neither is true of a call
    that raises.
    """
    if max_pdpert is None:
        return
    refuse, worst = session.solver.pdpert_verdict(float(max_pdpert))
    if refuse:
        pert = [abs(float(v)) for v in session.solver.kkt_perturbations]
        raise ValueError(
            f"{who}: the converged KKT factor carries a perturbation of "
            f"{worst:.3e} (dx={pert[0]:.3e}, ds={pert[1]:.3e}, "
            f"dc={pert[2]:.3e}, dd={pert[3]:.3e}), above the requested "
            f"max_pdpert={max_pdpert:.3e}. The factor the step would "
            f"invert is not this problem's KKT matrix. Raise max_pdpert "
            f"to accept it anyway.")


def _correct(session, pin_idx, deltas, step, corrector_iter):
    """Refine a step by Newton iterations on the barrier system.

    Returns the refined primal step and what the iterations did.

    The corrector starts from a point, so it needs the multipliers as
    well as the primal step, and only the plain parametric step reports
    both. Every mode refines the same underlying step, so the start is
    the mode's own primal block carrying the plain step's multipliers.
    The iterations correct the multipliers from there, which is what
    they are for.

    Two consequences worth naming. This spends a second solve even under
    mode="linear" with degeneracy="one_sided", where the primal block it
    then overwrites is the one it just computed. And under
    degeneracy="directional" the multipliers come from the one-sided
    step rather than the directional one that produced `step`, since
    there is no directional full step to ask.
    """
    full = np.asarray(session.solver.parametric_step_full(pin_idx, deltas))
    n_x = len(step)
    full[:n_x] = np.asarray(step)
    out, info = session.solver.correct_step(
        pin_idx, deltas, list(full), corrector_iter)
    return np.asarray(out)[:n_x], dict(info)


def estimate(model, perturb, clamp=True, mode="linear",
             predictor_iter=16, degeneracy="directional", degeneracy_iter=16,
             corrector_iter=0, bound_eps=None, max_pdpert=None):
    """First-order estimate of the solution at perturbed parameter values.

    perturb: pairs of (declared Param, new value) -- a list of tuples or a
    ComponentMap (plain dicts don't work: Pyomo components are unhashable).
    Returns a ComponentMap {original var data: estimated value}. Values are
    clamped to variable bounds (with a warning) unless clamp=False.

    mode selects what happens when the step leaves a variable bound.
    "linear" (the default) takes the step and clamps, which truncates the
    crossing variable and leaves every other one at its predictor value.
    "fix_relax" instead pins the crossing variable at its bound and
    re-solves, so the others move to stay consistent under the pin, which
    is what `estimate_report`'s step fraction says the step needs. It
    costs a dense solve and a backsolve per crossing and tracks a full
    re-solve far more closely. Where nothing crosses the two agree
    exactly.

    "path" applies the perturbation a little at a time, stopping at
    each fraction where the active set changes and continuing under
    the changed set, so a variable can reach a bound partway through
    the change and leave it again further along. Where the two settle
    the same active set it agrees with "fix_relax". At a change large
    enough that they disagree, "fix_relax" decides every change from
    a single step taken at the base point while "path" applies each
    change at the fraction where it happens. `active_set_changes()`
    returns the record of those changes.

    predictor_iter bounds that work. Under "fix_relax" it caps the
    passes, each of which pins every crossing it can see and costs a
    dense solve whose size grows with the number of pins. It is a
    safety limit there rather than a budget, since the loop ends when
    nothing is left outside a bound. Under "path" it caps the number of
    active-set changes applied, and past the cap the rest of the
    perturbation is taken in one step under the active set reached.

    degeneracy selects what happens when the base point itself sits at
    an active-set kink: a bound the classifier can call neither active
    nor inactive, on the bound with a multiplier of the same order as
    the slack. The solution has two one-sided derivatives there.
    "directional" (the default) decides the weakly active bounds by the
    directional-derivative QP for the perturbation's own direction, in
    every mode, at the cost of a few extra backsolves paid only at a
    kink. degeneracy_iter budgets those backsolves: the all-released
    solve and one basis column per engaged bound count against it, and
    a budget the engagement cannot fit falls back to the one-sided
    step with a warning naming the counts. "one_sided" takes the
    single-sided value today's thresholds produce, bit-identical to
    the release before this option existed.

    corrector_iter runs Newton iterations on the barrier system after
    the step, against the factorization the solve left behind, so each
    one costs a back-solve and no factorization. It aims at the barrier
    solution at the mu the solve finished on rather than at a re-solve,
    so the accuracy it reaches is bounded by that offset, and it stops
    as soon as an iteration fails to improve the residual. Where the
    perturbation needs a bound the base point held tightly to leave the
    active set, the held factorization cannot represent the change, the
    iterations make no progress, and a warning says so rather than
    letting the uncorrected step pass as corrected. Measured on a
    100-step Hicks-Ray CSTR displaced from its setpoint, a small step
    goes from 6.3e-6 to 5.5e-9 in nine back-solves, and a step needing a
    bound with sigma near 3e5 to leave stops after three with nothing
    gained. It applies under every mode, refining whatever step that
    mode produced.

    bound_eps sets how far outside a variable bound a step has to end
    to count as having left it, which decides what mode="fix_relax"
    pins, what `estimate_report()` reports in `crossed`, and what the
    clamp below acts on. It is absolute, as the refinement's own test
    is. Unset, it is how far outside the solve itself was willing to
    settle, floored so an unrelaxed solve does not pin on roundoff. A
    constraint row keeps its own floor, and a bound is released when the
    step drives its multiplier negative past the solve's own margin,
    whatever bound_eps is. Only mode="fix_relax" reads it, and passing
    it under another mode warns and changes nothing. It must be
    positive, as the CLI's sens_bound_eps is.

    max_pdpert refuses rather than answering when the converged KKT
    factor carries an inertia correction larger than the value given.
    Every sensitivity output inverts that factor, so a perturbed one
    answers for a nearby problem rather than this one.
    `estimate_report().perturbations` reports the same numbers for a
    caller who would rather read them than stop. It must be positive,
    as the CLI's sens_max_pdpert is, and the same argument is on
    gradient(), estimate(), estimate_report(), active_set_changes(),
    covariance() and information().

    clamp keeps its meaning in both modes: it clamps whatever is still
    outside a bound at the end. Under "fix_relax" the pins usually
    leave nothing to clamp, and when they do not, the warning says
    which of the refinement's stopping conditions was reached.

    The perturbation is measured from the SOLVE point (the pin
    constraint's stored right-hand side, which is the value the Param
    had at the solve), not from the Param's current value on the model.
    Writing a new value into the Param first (the receding-horizon
    pattern: solve at a prediction, record the measurement, then ask)
    does not change the answer.

    A bound written in terms of a declared Param is a constraint by the
    time the model is solved, so it is not clamped against here and no
    clamp warning is raised for it. That is deliberate: the bound moves
    with the perturbation, so the linear step already respects it to
    first order.
    """
    reg = model.__dict__.get(_REG)
    session = reg.session if reg else None
    if session is None:
        raise RuntimeError(
            "no sensitivity session: declare_sens_param() then solve with "
            "SolverFactory('pounce'), SolverFactory('pounce_v2') or the "
            "contrib SolverFactory('pounce') first")

    if mode not in ("linear", "fix_relax", "path"):
        raise ValueError(
            "estimate: mode must be 'linear', 'fix_relax' or 'path', got "
            f"{mode!r}")
    if degeneracy not in ("directional", "one_sided"):
        raise ValueError(
            "estimate: degeneracy must be 'directional' or 'one_sided', "
            f"got {degeneracy!r}")
    _check_margins(bound_eps, max_pdpert, "estimate")
    if bound_eps is not None and mode != "fix_relax":
        warnings.warn(
            "estimate: bound_eps is the margin the fix_relax refinement "
            f"pins against and mode={mode!r} runs no refinement, so it "
            "changes nothing here.")
    _refuse_on_pdpert(session, max_pdpert, "estimate")

    pin_idx, deltas = _perturbation_deltas(session, perturb)

    # parametric_step returns the factor's x block (var-x); base_x and
    # everything below it (nl.x_l/x_u, var_names) are full-x
    segments = []
    fell_back = False
    if degeneracy == "directional":
        # At a degenerate base point the weakly active bounds are
        # decided by the directional-derivative QP for this
        # perturbation's own direction; at a clean base point these are
        # the plain calls at no extra solve cost. A failed decision
        # (budget exhausted, no sign-consistent working set) falls back
        # to the one-sided step and says so.
        try:
            step, held_rows, _ = (
                session.solver.parametric_step_directional(
                    pin_idx, deltas, degeneracy_iter))
            if mode == "fix_relax":
                step, pinned, stop = (
                    session.solver.parametric_step_bounded_decided(
                        pin_idx, deltas, held_rows, predictor_iter,
                        bound_eps))
            elif mode == "path":
                step, segments = (
                    session.solver.parametric_step_path_decided(
                        pin_idx, deltas, held_rows, predictor_iter))
        except RuntimeError as e:
            if "directional derivative" not in str(e):
                raise
            warnings.warn(
                f"estimate: {e}. Falling back to the one-sided step, "
                "the degeneracy='one_sided' behavior.")
            fell_back = True
    if degeneracy == "one_sided" or fell_back:
        if mode == "fix_relax":
            step, pinned, stop = session.solver.parametric_step_bounded(
                pin_idx, deltas, predictor_iter, bound_eps)
        elif mode == "path":
            step, segments = session.solver.parametric_step_path(
                pin_idx, deltas, predictor_iter)
        else:
            step = session.solver.parametric_step(pin_idx, deltas)
    corrector = None
    if corrector_iter:
        step, corrector = _correct(
            session, pin_idx, deltas, step, corrector_iter)
        # A correction that works drives the residual down by orders;
        # one that cannot represent the active-set change the
        # perturbation needs shaves a few percent off and leaves the
        # estimate where it was. Halving is a low bar that separates
        # them cleanly, and saying nothing would let the second case
        # pass for the first.
        if corrector is not None and corrector["residual"] > 0.5 * corrector[
                "initial_residual"]:
            warnings.warn(
                "estimate: the corrector spent "
                f"{corrector['iterations']} back-solve(s) and moved the "
                f"residual from {corrector['initial_residual']:.2e} to "
                f"{corrector['residual']:.2e}, measured from the point the "
                "iterations start at rather than from the step handed in, "
                "so the estimate is close to "
                "the uncorrected step. That happens when the perturbation "
                "needs a bound the base point held tightly to leave the "
                "active set, which the held factorization cannot represent.")

    dx = session.scatter_x(np.asarray(step))
    x_new = session.base_x + dx

    lo, hi = np.asarray(session.nl.x_l), np.asarray(session.nl.x_u)
    if mode in ("fix_relax", "path"):
        # The refinement holds each pinned coordinate AT its bound and
        # lets the others move, so a coordinate can still be left
        # outside one, either because the pass budget ran out or because
        # the pins have exhausted the problem's degrees of freedom.
        # `clamp` decides what happens then, exactly as it does under
        # `linear`, and the warning says which of the two stopped it.
        #
        # One tolerance for "outside its bound", taken from the solve
        # rather than chosen here: it was willing to leave a converged
        # point `bound_relax_factor` outside, so anything within that is
        # on the bound. The refinement pins against the same number.
        # The comparison is absolute, as the refinement's own is, so the
        # clamp and the pins agree on a coordinate of any magnitude.
        eps = _bound_margin(
            session, bound_eps if mode == "fix_relax" else None)
        out = np.where((x_new < lo - eps) | (x_new > hi + eps))[0]
        if out.size:
            names = [session.var_names[i] for i in out]
            if mode == "fix_relax":
                did = f"pinned {len(pinned)} variable(s)"
                # The refinement says why it stopped rather than the
                # count being read as a proxy for it: a pass pins every
                # crossing it sees, so the number of pins says nothing
                # about whether predictor_iter bound the work (gh#732).
                why = {
                    "iteration_limit":
                        "the safety limit of %d pass(es) was reached, so "
                        "raising predictor_iter may finish it" % predictor_iter,
                    "degrees_of_freedom":
                        "holding them all would need more pins than the "
                        "problem has degrees of freedom, so no step does",
                    "worse_than_plain":
                        "the refinement ended further outside the bounds "
                        "than the plain step, which was returned instead",
                }.get(stop, "the refinement settled here")
            else:
                n_changes = len(segments)
                did = f"applied {n_changes} active-set change(s)"
                why = ("the limit of %d was reached, so raising "
                       "predictor_iter may finish it" % predictor_iter
                       if n_changes >= predictor_iter else
                       "the path settled the active set here")
            warnings.warn(
                f"estimate: {mode} {did} and "
                f"still leaves the bounds for {names}, because {why}."
                + (" The values were clamped, which breaks the constraints "
                   "the pins were solved against." if clamp else
                   " The values are returned unclamped."))
            if clamp:
                x_new = np.clip(x_new, lo, hi)
    elif clamp:
        # `linear` takes the step as the predictor gives it, so a
        # crossing shows up as a value outside its bound and clamping is
        # all this mode can do about it. The active set changed and the
        # step does not know, which is what `fix_relax` addresses.
        # The comparison is absolute, as the refinement's own is.
        eps = _bound_margin(session, None)
        clamped = (x_new < lo - eps) | (x_new > hi + eps)
        if clamped.any():
            names = [session.var_names[i] for i in np.where(clamped)[0]]
            warnings.warn(
                "estimate: linear step leaves the variable bounds for "
                f"{names}; values were clamped and the active set likely "
                "changed, so the estimate is unreliable there. mode="
                "'fix_relax' pins them and re-solves instead.")
        x_new = np.clip(x_new, lo, hi)

    out = ComponentMap()
    for name, val in zip(session.var_names, x_new):
        ov = model.find_component(name)
        if ov is not None:
            out[ov] = float(val)
    return out


# ── step diagnostics ──────────────────────────────────────────────────────────

class EstimateReport:
    """What the linear step behind `estimate()` does about the bounds.

    Attributes
    ----------
    alpha : float
        The fraction of the requested perturbation that can be taken
        before the first bound is reached, from the ratio test along
        the step. On a solve that kept its bounds, at least 1.0 means
        the full step stays inside every one of them. That does not
        follow when `bounds_relaxed` is true, since the solve can leave
        a coordinate outside a bound before the step starts. Infinity
        means no bound lies in the step direction.
    first : str or None
        Name of the variable or constraint reaching its bound at
        `alpha`, None when `alpha` is infinite.
    first_kind : str or None
        Either "variable" or "constraint", naming what `first` is.
    crossed : ComponentMap
        Variable data to the distance by which the full step leaves
        that variable's bound. These are the variables `estimate()`
        clamps. Measured at the predicted point against both bounds, so
        on a relaxed solve an entry can be a coordinate the SOLVE left
        outside its bound rather than one the step carried past. The
        step fraction looks only along the step direction, so the two
        answer different questions there and `alpha` can be at or above
        one while this is non-empty.
    crossed_rows : ComponentMap
        Constraint data to the same distance, for inequality
        constraints.

    Those two are keyed by model components, because every coordinate
    that crosses has one. The two classifications below are keyed by
    solve-space name instead, because they cover every coordinate of
    the solve, including the ones the declared-parameter surgery
    created, which have no counterpart on the model.
    violation : float
        Largest constraint violation at the predicted point, measured
        against the perturbed right-hand sides. This is the primal
        half of the residual. The dual half needs the multipliers at
        the perturbed point, so it is computed by the corrector, which
        holds them, rather than assembled here.
    mu : float
        Barrier parameter of the held factorization.
    activity, row_activity : dict
        Name to activity classification from the core classifier, over
        variables and over constraints. Both are empty when
        `bounds_relaxed` is true.
    perturbations : list
        The held factor's inertia-correction perturbations. Any
        non-zero entry means the factor is regularized, so the step is
        taken against a modified matrix and differs from the exact
        active-set answer by that much.
    corrector : dict or None
        What `corrector_iter` iterations did, None when none were run.
        Holds the back-solves spent under `iterations`, the residual
        before and after under `initial_residual` and `residual`
        (`initial_residual` is measured at the point the iterations
        start from, after the active-set decision and the clamp, not at
        the step handed in), and
        that residual split into `stationarity`, `feasibility` and
        `complementarity`. `released` counts the bounds the step took
        out of the active set and `pinned` the ones it brought in, with
        `active_set_changes` their total. `converged` says the loop
        stopped because an iteration failed to improve rather than
        because it ran out of budget. This is the dual half `violation`
        refers to: it needs the multipliers at the perturbed point,
        which the corrector holds.
    refine_stop : str or None
        Why the `mode="fix_relax"` refinement stopped, one of
        "settled", "iteration_limit", "degrees_of_freedom" or
        "worse_than_plain". None under the other two modes, which run
        no refinement. A pass pins every crossing it sees, so the
        number of pins says nothing about which limit was reached and
        this is the only thing that does. "worse_than_plain" means the
        step reported here is the unrefined one.
    bounds_relaxed : bool
        True when the solve ran with a non-zero `bound_relax_factor`,
        which lets a variable settle outside the bound the model
        declares. The classifier raises for such a solve, since relaxed
        bounds shift the slacks it reads, and the ratio test measures
        distances to bounds the solve did not hold to.

    The last three, with `mu`, are what separate this predictor from
    the exact value at the perturbed active set. A caller comparing the
    estimate against a re-solve reads them to tell which one explains
    the difference.
    """

    def __init__(self, alpha, first, first_kind, crossed, crossed_rows,
                 violation, mu, activity, row_activity, perturbations,
                 bounds_relaxed, corrector=None, refine_stop=None):
        self.alpha = alpha
        self.first = first
        self.first_kind = first_kind
        self.crossed = crossed
        self.crossed_rows = crossed_rows
        self.violation = violation
        self.mu = mu
        self.activity = activity
        self.row_activity = row_activity
        self.perturbations = perturbations
        self.bounds_relaxed = bounds_relaxed
        self.corrector = corrector
        self.refine_stop = refine_stop

    def __repr__(self):
        n = len(self.crossed) + len(self.crossed_rows)
        where = f", first {self.first_kind} {self.first}" if self.first else ""
        flags = "".join([
            ", regularized" if any(self.perturbations) else "",
            ", bounds relaxed" if self.bounds_relaxed else "",
        ])
        return (f"EstimateReport(alpha={self.alpha:.6g}{where}, "
                f"crossed={n}, violation={self.violation:.3e}, "
                f"mu={self.mu:.3e}{flags})")


def _row_step(session, dx):
    """Jacobian at the base point times the step, in row order."""
    rows, cols = session.nl.jacobian_structure()
    vals = np.asarray(session.nl.jacobian(session.base_x))
    return np.bincount(np.asarray(rows),
                       weights=vals * dx[np.asarray(cols)],
                       minlength=len(session.con_names))


#: No bound at all reaches here as the reader's sentinel rather than an
#: infinity: `read_nl` seeds every bound vector with +-1e19, so an
#: `isfinite` test passes on it and scores a crossing at 1e18 times the
#: perturbation. The magnitude test is the convention the rest of the
#: package already uses (`preflight`, `block_init`, gh #401 / #403).
_NO_BOUND = 1e19


def _ratio_test(base, step, lo, hi, names, live=None, on_bound=None,
                mu=float("nan"), tol=None):
    """Smallest fraction of `step` that reaches a bound, and where.

    Only coordinates strictly inside their bounds take part, because a
    coordinate already on a bound is not one the step can cross. That
    exclusion is what makes the answer mean anything: at an active
    bound the remaining gap is the small slack the barrier leaves, the
    step component is the same size, and their quotient can take any
    value. It would then become the minimum on any model carrying an
    active bound. Activity is reported through the classification
    instead.

    A coordinate the full step carries past a bound is always scored,
    whatever the exclusion would otherwise say. That case IS a crossing,
    at a fraction below one by construction, and the caller reads it in
    `crossed` off the same predicate and the same tolerance. Deciding it
    twice is what let the two disagree, reporting that 998 times the
    perturbation fits while naming a coordinate the single step leaves
    its bound by 0.25.

    So the exclusion below only ever applies to coordinates that do not
    cross, and cannot cost a crossing. Two things drive it. `on_bound`
    carries the classification, which is exact where the classifier
    commits, applied per SIDE rather than per coordinate, since a
    coordinate held at one bound can still be carried across its other
    one.

    The distance test covers the coordinates the classifier declines to
    rule on, where the label cannot answer the question because
    `ambiguous` spans both a coordinate near its bound with room left
    and one already on it. What separates those is the size of the
    remaining gap, which is O(mu) at a strongly active bound and
    O(sqrt(mu)) at a weakly active one, against O(1) room in the
    interior. So the threshold scales with sqrt(mu), measured four
    orders of magnitude clear of the interior case. It is capped
    because it is applied relative to the coordinate's own magnitude,
    and a loose `mu` at termination would otherwise widen it without
    limit. Without a classification there is no `mu` either, and it
    falls back to a fixed threshold.

    Coordinates whose step is below the roundoff of their own magnitude
    also take no part, since dividing by one puts the crossing at an
    enormous multiple of the perturbation, which is not a finding.
    """
    scale = np.maximum(1.0, np.abs(base))
    toward_hi = step > 0
    bound = np.where(toward_hi, hi, lo)
    distance = bound - base
    present = np.abs(bound) < _NO_BOUND
    moving = np.abs(step) > 1e-12 * scale

    # The same predicate, and the same tolerance, that fills `crossed`
    # or `crossed_rows`. The caller passes the tolerance it fills with,
    # and the default is the rows' own.
    reached = base + step
    if tol is None:
        tol = 1e-9 * np.maximum(1.0, np.abs(reached))
    crosses = present & moving & np.where(
        toward_hi, reached > bound + tol, reached < bound - tol)

    floor = 1e-9 if not np.isfinite(mu) else min(
        1e-2, max(1e-9, 10.0 * mu ** 0.5))
    interior = np.abs(distance) > floor * scale
    if on_bound is not None:
        # the nearer bound is the one the coordinate is held at, and it
        # is only that side the classification rules out
        at_hi = (hi - base) <= (base - lo)
        interior &= ~(on_bound & (toward_hi == at_hi))

    ok = present & moving & (crosses | interior)
    if live is not None:
        ok &= live
    if not ok.any():
        return float("inf"), None
    fraction = np.full(base.shape, np.inf)
    # a relaxed solve can settle outside a bound, which leaves no room
    # to move rather than a negative amount of it
    fraction[ok] = np.maximum(distance[ok] / step[ok], 0.0)
    i = int(np.argmin(fraction))
    return float(fraction[i]), names[i]


#: Classifications that put a coordinate ON its bound. Both have the
#: slack going to zero, so neither can be crossed by a step. Strongly
#: active leaves an O(mu) gap and weakly active an O(sqrt(mu)) one, and
#: neither gap is room the step can move through. `ambiguous` and
#: `unidentified` are absent on purpose: the first spans coordinates on
#: their bound AND coordinates near it with room left, and the second
#: says only that the curvature is below scale, so neither label
#: answers the question. `_ratio_test`'s distance test rules on those.
_AT_BOUND = ("strongly_active", "weakly_active")


def _on_bound(statuses):
    return np.array([s in _AT_BOUND for s in statuses])


def _user_row_names(session):
    """Row names as the user wrote them.

    The declared-parameter surgery moves every constraint onto its own
    block, so the solve's constraint names are clone names. `con_alias`
    records the move in the other direction. A constraint with no entry
    keeps the clone name, which is what the pin constraints have and
    what marks them as the solve's own.
    """
    back = {clone: orig for orig, clone in session.con_alias.items()}
    return [back.get(nm, nm) for nm in session.con_names]


def estimate_report(model, perturb, max_iter=None,
                    degeneracy="directional", degeneracy_iter=16,
                    corrector_iter=0, mode="linear", predictor_iter=16,
                    bound_eps=None, max_pdpert=None):
    """Report what `estimate()`'s step does about the bounds.

    degeneracy and degeneracy_iter match `estimate()`'s arguments of
    the same names, so the step measured here is the step `estimate()`
    takes for the same arguments, including the directional-derivative
    correction at a degenerate base point. degeneracy_iter budgets that
    correction's back-solves.

    max_iter is accepted for positional compatibility and does nothing.
    It used to budget the directional decision here, so passing it, in
    particular `max_iter=0` to force the one-sided fallback, changed the
    reported step. `degeneracy_iter` is that knob now. Passing it raises
    a DeprecationWarning rather than being ignored in silence, since the
    two readings differ and nothing else would say so.

    mode and predictor_iter match `estimate()`'s arguments of the same
    names, so the step measured here is the step `estimate()` takes for
    the same arguments. `violation` and `corrector` are properties of
    that step and move with the mode. Reporting the linear step's
    violation for a `fix_relax` estimate would describe a step the
    caller did not take.

    `alpha`, `first`, `crossed` and `crossed_rows` measure where the
    step leaves a bound. Under "fix_relax" and "path" it does not,
    since both stop at the bound by construction, so `alpha` is 1.0 and
    `crossed` is empty for every model. That is the correct answer for
    such a step rather than a missing one. A `bound_eps` wide enough to
    cover the crossing is the exception: the refinement then pins
    nothing, so the step is the linear one and reaches a bound at a
    fraction below one. The reason to run those
    modes is what "linear" reports at the same perturbation.

    `activity`, `row_activity` and `mu` come from the converged base
    point and `perturbations` and `bounds_relaxed` from the solve, so
    none of the five depends on the mode.

    bound_eps sets how far outside a variable bound a step has to end
    to count as having left it, which decides what mode="fix_relax"
    pins and what `crossed` and `alpha` measure against. It is absolute,
    as the refinement's own test is. Unset, it is how far outside the
    solve itself was willing to settle, floored so an unrelaxed solve
    does not pin on roundoff, and `crossed` keeps that floor so a
    coordinate the solve itself left outside a relaxed bound is still
    named. A constraint row keeps its own floor, and a bound is released
    when the step drives its multiplier negative past the solve's own
    margin, whatever bound_eps is. Only mode="fix_relax" reads it, and
    passing it under another mode warns. It must be positive, as the
    CLI's sens_bound_eps is.

    max_pdpert refuses rather than answering when the converged KKT
    factor carries an inertia correction larger than the value given.
    Every sensitivity output inverts that factor, so a perturbed one
    answers for a nearby problem rather than this one.
    `estimate_report().perturbations` reports the same numbers for a
    caller who would rather read them than stop. It must be positive,
    as the CLI's sens_max_pdpert is, and the same argument is on
    gradient(), estimate(), estimate_report(), active_set_changes(),
    covariance() and information().

    corrector_iter runs the same Newton iterations `estimate()` runs and
    reports what they did on the `corrector` attribute, without changing
    anything the rest of the report measures. Those describe the step
    handed to the corrector, which is what a caller comparing the two
    wants.

    Takes the same perturbation argument `estimate()` takes and returns
    an EstimateReport. Nothing about the estimate changes: this runs
    the same step and measures it.

    The ratio test divides each bounded coordinate's distance to the
    bound it moves toward by the size of its step component and keeps
    the smallest value, over variables and over inequality constraint
    constraints. Equality constraints do not take part, since the step
    holds them to first order and they admit no activity change. The
    constraints pinning the declared Params are excluded on the same
    grounds, because the perturbation moves their right-hand sides by
    construction.
    """
    reg = model.__dict__.get(_REG)
    session = reg.session if reg else None
    if session is None:
        raise RuntimeError(
            "no sensitivity session: declare_sens_param() then solve with "
            "SolverFactory('pounce'), SolverFactory('pounce_v2') or the "
            "contrib SolverFactory('pounce') first")

    if mode not in ("linear", "fix_relax", "path"):
        raise ValueError(
            "estimate_report: mode must be 'linear', 'fix_relax' or "
            f"'path', got {mode!r}")
    if degeneracy not in ("directional", "one_sided"):
        raise ValueError(
            "estimate_report: degeneracy must be 'directional' or "
            f"'one_sided', got {degeneracy!r}")
    _check_margins(bound_eps, max_pdpert, "estimate_report")
    if bound_eps is not None and mode != "fix_relax":
        warnings.warn(
            "estimate_report: bound_eps is the margin the fix_relax "
            f"refinement pins against and mode={mode!r} runs no "
            "refinement, so it changes nothing here.")
    _refuse_on_pdpert(session, max_pdpert, "estimate_report")
    if max_iter is not None:
        warnings.warn(
            "estimate_report: max_iter no longer does anything here and is "
            "ignored; it used to budget the directional decision, which "
            "degeneracy_iter budgets now. Pass degeneracy_iter instead.",
            DeprecationWarning, stacklevel=2)
    pin_idx, deltas = _perturbation_deltas(session, perturb)
    # the same dispatch `estimate()` runs, so the step measured here is
    # the step it takes for these arguments
    fell_back = False
    refine_stop = None
    if degeneracy == "directional":
        try:
            step, held_rows, _ = session.solver.parametric_step_directional(
                pin_idx, deltas, degeneracy_iter)
            if mode == "fix_relax":
                step, _, refine_stop = (
                    session.solver.parametric_step_bounded_decided(
                        pin_idx, deltas, held_rows, predictor_iter,
                        bound_eps))
            elif mode == "path":
                step, _ = session.solver.parametric_step_path_decided(
                    pin_idx, deltas, held_rows, predictor_iter)
        except RuntimeError as e:
            if "directional derivative" not in str(e):
                raise
            warnings.warn(
                f"estimate_report: {e}. Falling back to the one-sided "
                "step, the degeneracy='one_sided' behavior.")
            fell_back = True
    if degeneracy == "one_sided" or fell_back:
        if mode == "fix_relax":
            step, _, refine_stop = session.solver.parametric_step_bounded(
                pin_idx, deltas, predictor_iter, bound_eps)
        elif mode == "path":
            step, _ = session.solver.parametric_step_path(
                pin_idx, deltas, predictor_iter)
        else:
            step = session.solver.parametric_step(pin_idx, deltas)
    dx = session.scatter_x(np.asarray(step))
    base = np.asarray(session.base_x)
    x_new = base + dx

    lo, hi = np.asarray(session.nl.x_l), np.asarray(session.nl.x_u)
    g_l, g_u = np.asarray(session.nl.g_l), np.asarray(session.nl.g_u)

    # The margin the refinement pinned against, so `alpha` and `crossed`
    # answer with the number that decided the step. It is absolute, as
    # the refinement's own test is, so a coordinate of order 1e4 does not
    # get a tolerance the refinement never gave it. Unset keeps the
    # fixed floor rather than the solve's relaxation: `crossed` reports
    # a coordinate the SOLVE left outside its bound, and the relaxation
    # is exactly how far out that is, so taking it as the margin would
    # hide the case this report exists to name.
    eps = (1e-9 if bound_eps is None or mode != "fix_relax"
           else float(bound_eps))
    # The refinement pins variable bounds only, so a constraint row
    # keeps its floor whatever margin the caller set: a wide margin has
    # no say over a row the step carries past its limit. Its tolerance
    # is set beside the row ratio test below.

    row_names = _user_row_names(session)

    # The classifier raises for a solve that relaxed its bounds, since
    # relaxed bounds shift the slacks it reads. Ask the solve what it
    # ran under instead of provoking that and reading the message, and
    # report it as the fact it is. The rest of the report is still
    # measured, where a diagnostic that raised would give the caller
    # nothing exactly when they are trying to find out why the estimate
    # disagrees with a re-solve. `mu` comes from the classifier, so it
    # is unavailable on that path too.
    mu, activity, row_status = float("nan"), {}, {}
    var_on_bound = row_on_bound = None
    bounds_relaxed = bool(session.solver.bound_relax_factor)
    if not bounds_relaxed:
        act = session.solver.classify_activity()
        mu = float(act["mu"])
        activity = dict(zip(session.var_names, act["var_status"]))
        row_status = dict(zip(row_names, act["row_status"]))
        var_on_bound = _on_bound(act["var_status"])
        row_on_bound = _on_bound(act["row_status"])

    alpha, first = _ratio_test(base, dx, lo, hi, session.var_names, tol=eps,
                               on_bound=var_on_bound, mu=mu)
    first_kind = None if first is None else "variable"
    dg = _row_step(session, dx)
    g_base = np.asarray(session.nl.constraints(base))
    # an equality constraint is held to first order and cannot change
    # activity, and a pin constraint moves by construction
    live = g_l < g_u
    live[list(pin_idx)] = False
    g_pred = g_base + dg
    gtol = 1e-9 * np.maximum(1.0, np.abs(g_pred))
    a_row, f_row = _ratio_test(g_base, dg, g_l, g_u, row_names, live=live,
                               on_bound=row_on_bound, mu=mu, tol=gtol)
    if a_row < alpha:
        alpha, first, first_kind = a_row, f_row, "constraint"

    crossed = ComponentMap()
    for i in np.where((x_new < lo - eps) | (x_new > hi + eps))[0]:
        ov = model.find_component(session.var_names[i])
        if ov is not None:
            crossed[ov] = float(max(lo[i] - x_new[i], x_new[i] - hi[i]))
    crossed_rows = ComponentMap()
    out_of_bounds = (g_pred < g_l - gtol) | (g_pred > g_u + gtol)
    for j in np.where(live & out_of_bounds)[0]:
        oc = model.find_component(row_names[j])
        if oc is not None:
            crossed_rows[oc] = float(
                max(g_l[j] - g_pred[j], g_pred[j] - g_u[j]))

    # the perturbation IS the shift of the pin constraints' right-hand
    # sides, so the violation is measured against the shifted ones
    gl_p, gu_p = g_l.copy(), g_u.copy()
    for pin, d in zip(pin_idx, deltas):
        gl_p[pin] += d
        gu_p[pin] += d
    g_at = np.asarray(session.nl.constraints(x_new))
    violation = float(np.max(np.maximum.reduce(
        [gl_p - g_at, g_at - gu_p, np.zeros_like(g_at)])))

    corrector = None
    if corrector_iter:
        _, corrector = _correct(
            session, pin_idx, deltas, np.asarray(step), corrector_iter)

    return EstimateReport(
        alpha=alpha, first=first, first_kind=first_kind,
        crossed=crossed, crossed_rows=crossed_rows, violation=violation,
        mu=mu, activity=activity, row_activity=row_status,
        perturbations=np.asarray(session.solver.kkt_perturbations).tolist(),
        bounds_relaxed=bounds_relaxed, corrector=corrector,
        refine_stop=refine_stop,
    )


#: One active-set change along `estimate(mode="path")`'s path.
#: `fraction` is how far along the perturbation the change happens,
#: `var` is the variable (its solve-space name when the solve created
#: it without a model counterpart), `bound` is "lower" or "upper", and
#: `action` is "reaches" when the variable arrives at the bound and is
#: held there, "leaves" when it comes off it.
ActiveSetChange = namedtuple(
    "ActiveSetChange", ["fraction", "var", "bound", "action"])


def active_set_changes(model, perturb, predictor_iter=16,
                       degeneracy="directional", degeneracy_iter=16,
                       max_pdpert=None):
    """The active-set changes `estimate(mode="path")` applies, in order.

    Takes the same perturbation argument `estimate()` takes and returns
    a list of `ActiveSetChange` entries, one per change, in the order
    the path applies them. Nothing about the estimate changes: this
    runs the same path and returns its record.

    The record is what `mode="path"` produces that no other mode does.
    The first entry's fraction is how much of the measurement
    discrepancy the held solve's active set survives unchanged, and the
    list as a whole says which bounds the re-optimized solution enters
    and leaves between the predicted state and the measured one.

    A list of length `predictor_iter` means the cap stopped the path before
    the target, the same condition `estimate()` warns about.

    degeneracy and degeneracy_iter match `estimate()`'s arguments of the
    same names. Under "directional" (the default), a weakly active bound
    the perturbation releases appears in the record as a departure at
    the fraction where its multiplier reaches zero: essentially zero at
    an exact kink, and partway along the step when the held solve sits
    inside the ambiguous band, where the bound is genuinely active for
    the first stretch. degeneracy_iter budgets that decision's
    back-solves, and a budget it cannot fit falls back to the one-sided
    record with a warning, and predictor_iter above still caps the path
    itself.

    max_pdpert refuses rather than answering when the converged KKT
    factor carries an inertia correction larger than the value given.
    This runs the same predictor `estimate(mode="path")` runs and
    inverts the same factor, so it takes the same cap. There is no
    bound_eps here, since the path decides its changes from where a
    multiplier reaches zero and reads no margin.
    """
    reg = model.__dict__.get(_REG)
    session = reg.session if reg else None
    if session is None:
        raise RuntimeError(
            "no sensitivity session: declare_sens_param() then solve with "
            "SolverFactory('pounce'), SolverFactory('pounce_v2') or the "
            "contrib SolverFactory('pounce') first")

    if degeneracy not in ("directional", "one_sided"):
        raise ValueError(
            "active_set_changes: degeneracy must be 'directional' or "
            f"'one_sided', got {degeneracy!r}")
    _check_margins(None, max_pdpert, "active_set_changes")
    _refuse_on_pdpert(session, max_pdpert, "active_set_changes")
    pin_idx, deltas = _perturbation_deltas(session, perturb)
    if degeneracy == "directional":
        try:
            _, held_rows, _ = (
                session.solver.parametric_step_directional(
                    pin_idx, deltas, degeneracy_iter))
            _, segments = session.solver.parametric_step_path_decided(
                pin_idx, deltas, held_rows, predictor_iter)
        except RuntimeError as e:
            if "directional derivative" not in str(e):
                raise
            warnings.warn(
                f"active_set_changes: {e}. Falling back to the "
                "one-sided record, the degeneracy='one_sided' behavior.")
            _, segments = session.solver.parametric_step_path(
                pin_idx, deltas, predictor_iter)
    else:
        _, segments = session.solver.parametric_step_path(
            pin_idx, deltas, predictor_iter)

    # segments carry var-x rows (the factor's x block); var_names is
    # full-x, so invert the same map scatter_x applies
    full_of = {row: full
               for full, row in enumerate(session._primal_row_map())
               if row is not None}
    out = []
    for frac, var_row, lower, pinned in segments:
        name = session.var_names[full_of[var_row]]
        comp = model.find_component(name)
        out.append(ActiveSetChange(
            fraction=float(frac),
            var=comp if comp is not None else name,
            bound="lower" if lower else "upper",
            action="reaches" if pinned else "leaves",
        ))
    return out


# ── parameter covariance ──────────────────────────────────────────────────────

class _ParamKeyed:
    """Lookup from a declared Param's data object to its row index.
    Keyed by id() because Pyomo components are unhashable."""

    _who = "covariance"          # accessor name used in diagnostics

    def __init__(self, params):
        self._params = list(params)
        self._pos = {id(p): i for i, p in enumerate(self._params)}

    def _loc(self, pd):
        i = self._pos.get(id(pd))
        if i is None:
            raise KeyError(f"{getattr(pd, 'name', pd)}: not one of the "
                           f"{self._who} parameters")
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


def _pin_eigenvector_signs(vecs):
    """Fix the arbitrary sign LAPACK gives each eigenvector column, so
    the direction an eigen() call reports is reproducible.

    `v` and `-v` are equally valid eigenvectors, and which one `eigh`
    returns is a convention of the LAPACK build, not a property of the
    matrix: the same model and data can report opposite directions on
    two machines. The convention here is the largest-magnitude
    component positive, ties broken by the earliest `params` position
    (argmax takes the first maximum). It is scale-invariant and
    independent of parameter ordering except through that tie-break.

    This pins the sign only. A repeated eigenvalue leaves the basis
    WITHIN its eigenspace arbitrary — any rotation of those columns
    diagonalizes just as well — so the individual eigenvectors of a
    degenerate block are still not reproducible; only the subspace
    they span is."""
    vecs = np.asarray(vecs, dtype=float)
    if vecs.size == 0:
        return vecs
    lead = np.abs(vecs).argmax(axis=0)
    signs = np.sign(vecs[lead, np.arange(vecs.shape[1])])
    signs[signs == 0.0] = 1.0        # an all-zero column, if one ever
    return vecs * signs              # reaches here, is left alone


class Covariance(_ParamMatrix):
    """Asymptotic parameter covariance, from covariance().

    Keyed by the fitted variables' data objects (the free `Var`s
    flagged with `declare_fitted`, not Pyomo `Param`s) in `params`
    (declaration) order: cov[m.k1, m.k2] (either order),
    cov[m.k1] for a variance, cov.std_err[m.k1],
    cov.correlation[m.k1, m.k2]. `matrix` is the dense numpy array
    ordered like `params`; `sigma_sq` is the residual variance that was
    used. eigen() supports identifiability diagnosis."""

    def __init__(self, params, matrix, sigma_sq, conditioned_on=()):
        super().__init__(params, matrix)
        self.params = self._params
        self.sigma_sq = sigma_sq          # float, or {group: float}
        self._conditioned = conditioned_on
        with np.errstate(invalid="ignore", divide="ignore"):
            se = np.sqrt(np.diag(self.matrix))
            corr = self.matrix / np.outer(se, se)
        # entries whose scale is undefined (a projected bound-active
        # parameter has exactly zero variance) are reported as 0
        corr[~np.isfinite(corr)] = 0.0
        self.std_err = _ParamVector(self.params, se)
        self.correlation = _ParamMatrix(self.params, corr)

    @property
    def conditioned_on(self):
        """Strongly active variables OUTSIDE the block: the matrix is
        conditional on those bounds (roadmap item 3); empty when none.
        Computed on first access (one backsolve per near-bound
        candidate) and cached, so calls that never read it pay
        nothing. Until first access the pending computation keeps the
        sensitivity session, and so the held KKT factor, alive; read
        it (or discard the result) to release them."""
        if callable(self._conditioned):
            self._conditioned = self._conditioned()
        return self._conditioned

    def eigen(self):
        """(eigenvalues, eigenvectors) of the covariance matrix,
        eigenvalues ascending, eigenvectors[:, i] in `params` order.
        An eigenvalue much larger than the rest flags a poorly
        identified direction: its eigenvector gives the parameter
        combination the data cannot pin down.

        Each eigenvector's sign is pinned so the direction it names
        reproduces across machines: its largest-magnitude component
        is positive (see _pin_eigenvector_signs)."""
        w, v = np.linalg.eigh(self.matrix)
        return w, _pin_eigenvector_signs(v)


def _indefinite(block):
    """A symmetric block with a genuinely negative eigenvalue, at a
    tolerance relative to the block's own scale. At a strict local
    minimum with LICQ the reduced Hessian is PSD, so this fires only
    on non-minimum stationary points or regularized (inertia-
    corrected) convergence: exactly the finding worth returning
    rather than refusing."""
    if block.size == 0:
        return False
    ev_min = float(np.linalg.eigvalsh(block).min())
    scale = float(np.abs(block).max())
    return ev_min < -1e-10 * max(1.0, scale)


def _estimation_counts(session):
    """(#factor variables, #equality rows). The tangent recovery is
    the constrained tangent map only when n_var - n_eq equals the
    number of fitted parameters: the equalities then determine the
    non-fitted variables given the fitted block. Callers guard on
    this, because outside it T = Zx inv(M) is quietly not the tangent
    map and R would be silently wrong."""
    n_var = sum(1 for r in session._primal_row_map() if r is not None)
    g_l = np.asarray(session.nl.g_l)
    g_u = np.asarray(session.nl.g_u)
    return n_var, int(np.count_nonzero(g_l == g_u))


def _tangent_reduced_hessian(session, M, zcols, who="covariance"):
    """The reduced Hessian over the fitted block, by tangent recovery:
    the x-blocks of the K-inverse columns are T*M (each satisfies the
    equalities and has the fitted block as its own coordinates), so
    T = Zx inv(M) exactly, with the factor's barrier weight cancelling
    multiplicatively rather than by subtraction. Then R = T^T H T with
    the exact Lagrangian Hessian (covariance roadmap item 2).

    Machine-exact for equality and variable-bound activity (everything
    in W cancels multiplicatively, pinned variables included), verified
    against analytic ground truth where the subtraction route loses
    log10(Sigma/q) digits. A binding INEQUALITY row instead couples
    through its slack barrier with large-but-finite weight, tilting
    the recovered tangent along that row's normal: measured ~1e-6
    relative at practical mu, degrading as mu tightens (the pinned
    combination drives M toward singularity), still ~6 digits beyond
    the subtraction it replaces. Requires the square estimation
    structure of _estimation_counts(); callers guard."""
    # the factor's x block is var-x (a fixed variable's column is
    # removed under make_parameter, gh #450), so slice that block and
    # scatter each tangent back to full-x for hessian_vec, whose
    # contract is user-space with zeros on the removed columns
    n_var = sum(1 for r in session._primal_row_map() if r is not None)
    Zx = np.column_stack([z[:n_var] for z in zcols])
    T = np.column_stack(
        [session.scatter_x(col) for col in (Zx @ _minv(M, who)).T])
    HT = np.column_stack([
        np.asarray(session.solver.hessian_vec(T[:, j]))
        for j in range(T.shape[1])
    ])
    R = T.T @ HT
    return 0.5 * (R + R.T)


_WORDING = {
    # the default block IS the fitted set, and saying so is strictly
    # more informative there; an arbitrary wrt block gets the
    # block-relative nouns (review of gh #466)
    "fitted": {"member": "fitted parameter",
               "outside": "non-fitted variables",
               "reach": "the fitted parameters",
               "combo": "fitted combination",
               "scale": "the fitted block's"},
    "block": {"member": "block member",
              "outside": "variables outside the block",
              "reach": "the block",
              "combo": "block combination",
              "scale": "the block's"},
}


def _classify_fitted_block(session, params, rows, M, zcols,
                           who="covariance", wording="fitted"):
    """Membership and row handling shared by covariance() and
    information(): item 1's classification at the reduced fitted
    block, the value-correction bookkeeping, and the binding-row
    normals, with the warnings both accessors owe their callers.
    Returns a namespace consumed by each accessor's own assembly."""
    from types import SimpleNamespace

    W = _WORDING[wording]
    n_params = len(params)
    # ── membership from the barrier activity classification ──────────────
    # (covariance roadmap item 1). The classifier's per-coordinate rule
    # scales Sigma by the coordinate's own Lagrangian curvature, which is
    # zero for a fitted parameter in the residual-variable idiom (the
    # curvature lives on the residuals), so the same rule runs HERE on
    # the reduced fitted block, where the parameter's curvature actually
    # is: q = the reduced Hessian diagonal with the parameter's own
    # barrier term removed, Sigma retained by the solve. A weakly active
    # parameter is KEPT and warned rather than silently pinned; no slack
    # threshold can make that distinction, because slack and multiplier
    # are both O(sqrt(mu)) at weak activity.
    R_exact = None
    act = session.solver.classify_activity()
    mu = float(act["mu"])
    R_W = _minv(M, who)                # reduced Hessian off the factor, W-based
    # M (and so R_W) is natural-units by the kkt_solve contract
    # (pounce#128), and so are the report's sigmas and row_normal
    # (unscaled at the classifier boundary per the same contract), so
    # everything here composes without scale factors.
    sig_fit = np.array([float(act["var_sigma"][r]) for r in rows])
    q_red = np.abs(np.diag(R_W) - sig_fit)
    floor = np.sqrt(np.finfo(float).eps) * max(
        1.0, float(np.abs(np.diag(R_W)).max()))
    active = []
    for i, p in enumerate(params):
        st = act["var_status"][rows[i]]
        if st in ("unbounded", "fixed"):
            continue                       # no variable bound to classify
        ri = float(sig_fit[i]) / max(float(q_red[i]), floor)
        if q_red[i] < floor and ri <= 1e1:
            # curvature AND barrier weight both below scale: the bound
            # question does not arise, but the direction is poorly
            # identified (a dominant Sigma cancelling inside q_red
            # instead lands ri astronomically high and classifies
            # strongly active)
            status = "unidentified"
        else:
            status = _classify_ratio(ri, mu)
        if status == "strongly_active":
            active.append(i)
            warnings.warn(
                f"{who}: {W['member']} {p.name} is held by its "
                "bound at the optimum (strongly active); its direction is "
                "projected out (zero variance, conditional on the active "
                "bound) and the boundary asymptotics are nonstandard.")
        elif status == "weakly_active":
            warnings.warn(
                f"{who}: {W['member']} {p.name} sits exactly on "
                "its bound with a vanishing multiplier (weakly active). "
                "It is kept in the free block with finite variance; "
                "boundary asymptotics are nonstandard.")
        elif status == "ambiguous":
            warnings.warn(
                f"{who}: {W['member']} {p.name} has ambiguous "
                "bound activity at the solve's final barrier parameter; "
                "re-solve with a tighter tol to settle it. It is kept in "
                "the free block.")
        elif status == "unidentified":
            warnings.warn(
                f"{who}: {W['member']} {p.name} has curvature "
                "below the model's own noise scale (unidentified); its "
                "variance is large rather than small. It is kept in the "
                "free block.")

    # ── binding general rows on the fitted block ──────────────────────────
    # (item 1 row projection, jkitchin/pounce#362). A strongly active
    # inequality row whose normal touches the fitted parameters pins a
    # DIRECTION of the fitted block: no per-parameter disposition can
    # state that, so the free block is reduced on the null space of the
    # binding normals and pushed back, singular by the number of binding
    # rows. Rows classify at the reduced level exactly as the variable
    # bounds above: the row's barrier weight against the curvature along
    # its own normal. A bound moved onto a row by declared-parameter
    # reformulation (jkitchin/pounce#357) is the single-coordinate case
    # and reproduces the variable disposition exactly.
    R_corr = R_W - np.diag(sig_fit)
    # columns held constant by the declared-parameter pins: a row's
    # support there contributes nothing through elimination (the pin
    # variable cannot move), so it does not make the row "mixed". The
    # pin constraint's normal is e_{pin var}, so its support IS the
    # pin column.
    pin_cols = set()
    for _pr in session.pins.values():
        _pn = np.asarray(session.solver.row_normal(int(_pr)), dtype=float)
        pin_cols.update(int(i) for i in np.nonzero(_pn)[0])
    fit_cols = set(int(r) for r in rows)
    bind_normals = []                  # unit normals over the fitted block
    row_corrections = []               # (weight, unit normal), applied after
    for j, rst in enumerate(act["row_status"]):
        if rst in ("equality", "unbounded", "inactive"):
            # inactive rows carry O(mu) geometric weight (the invariant
            # form), the same order as every other accepted O(mu) term;
            # skipping them also avoids fetching every row normal on
            # wide models (an O(m*n) sweep). The bound and row
            # spellings of an INACTIVE limit therefore agree to O(mu)
            # rather than exactly, tested at that tolerance.
            continue
        a_full = np.asarray(session.solver.row_normal(j), dtype=float)
        a = a_full[rows]
        na = float(np.linalg.norm(a))
        # A row whose normal also touches NON-fitted variables pins a
        # combination that reaches the fitted block through the
        # eliminated variables, not along the restricted normal: e.g.
        # a + r_1 <= cap with r_1 = y_1 - a - b*x_1 actually pins a
        # b-direction, while the restricted normal reads e_a. The
        # restricted projection would delete the wrong direction, so a
        # mixed binding row is kept unprojected with an explicit
        # warning instead. The general treatment needs the row's
        # reduced normal through the elimination (roadmap item 2's
        # machinery).
        nf = float(np.linalg.norm(a_full))
        outside = [i for i in np.nonzero(a_full)[0]
                   if int(i) not in fit_cols and int(i) not in pin_cols]
        mixed = bool(outside) and (
            float(np.linalg.norm(a_full[outside])) > 1e-8 * max(1.0, nf))
        if na <= 1e-12 * max(1.0, nf):
            # entirely outside the fitted block: the extreme mixed case
            # (relative tolerance, matching the mixed test above)
            mixed = True
        cname = (session.con_names[j] if j < len(session.con_names)
                 else f"row {j}")
        if mixed:
            # the reduced-level rule is also unreliable here: the row's
            # barrier weight lands through elimination on a direction
            # the restricted normal cannot see, so re-classifying
            # against it manufactures a wrong ratio. Item 0's raw
            # classification (scale-invariant along the full normal) is
            # the honest status for a mixed row.
            if rst == "strongly_active":
                warnings.warn(
                    f"{who}: constraint {cname} is strongly active "
                    f"and involves {W['outside']}; the "
                    f"direction it pins reaches {W['reach']} through the "
                    "eliminated variables and cannot be represented by "
                    "a restricted normal, so it is NOT projected. Treat "
                    "the returned variances as not conditioned on this "
                    "constraint.")
            elif rst in ("weakly_active", "ambiguous", "unidentified"):
                warnings.warn(
                    f"{who}: constraint {cname} is {rst} and "
                    f"involves {W['outside']}; it is kept "
                    "unprojected and its barrier weight is not "
                    "corrected for (the restricted direction would be "
                    "the wrong one). Boundary asymptotics are "
                    "nonstandard.")
            continue
        a = a / na
        # the row's slack elimination contributes Sigma_j * (raw normal
        # outer product) to the reduced block; in the unit-normal basis
        # that coefficient is Sigma_j * ||raw normal||^2 (all natural
        # units: the report unscales at the classifier boundary)
        sig_row = float(act["row_sigma"][j]) * na * na
        q_w = float(a @ R_corr @ a)
        # |q|, matching the Rust classifier: indefinite curvature
        # classifies on magnitude, and indefiniteness still surfaces
        # through the negative-variance warning downstream
        q_row = abs(q_w - sig_row)
        ri = sig_row / max(q_row, floor)
        if q_row < floor and ri <= 1e1:
            status = "unidentified"
        else:
            status = _classify_ratio(ri, mu)
        combo = " + ".join(
            f"{a[k]:.3g}*{params[k].name}" for k in range(n_params)
            if abs(a[k]) > 1e-12)
        if status == "strongly_active":
            bind_normals.append(a)
            # conditional information along the pinned combination via
            # the tangent-recovered reduced Hessian (item 2 machinery;
            # lazy, only solves with a binding row pay the Hessian
            # products). Accurate to ~1e-6 at practical mu, the residue
            # being this row's own finite slack-barrier weight in the
            # recovery, where the factor subtraction lost ten digits.
            # Outside the square estimation structure the recovery is
            # undefined, so the subtraction value is kept there.
            if R_exact is None:
                n_var, n_eq = _estimation_counts(session)
                if n_var - n_eq == n_params:
                    R_exact = _tangent_reduced_hessian(
                        session, M, zcols, who)
            if R_exact is not None:
                s_a = float(a @ R_exact @ a)
            else:
                s_a = max(q_w - sig_row, 0.0)
            warnings.warn(
                f"{who}: constraint {cname} is strongly active and "
                f"pins the {W['combo']} {combo}; variance along it "
                "is projected to zero (conditional on the constraint). "
                f"Conditional information along the combination: {s_a:.6g}.")
        elif status == "weakly_active":
            warnings.warn(
                f"{who}: constraint {cname} is weakly active on the "
                f"{W['combo']} {combo} (multiplier and slack vanish "
                "together). It is kept unprojected with finite variance; "
                "boundary asymptotics are nonstandard.")
        elif status == "ambiguous":
            warnings.warn(
                f"{who}: constraint {cname} has ambiguous activity "
                f"on the {W['combo']} {combo} at the solve's final "
                "barrier parameter; re-solve with a tighter tol to "
                "settle it. It is kept unprojected.")
        elif status == "unidentified":
            warnings.warn(
                f"{who}: constraint {cname} has curvature below "
                f"{W['scale']} noise scale on {combo} "
                "(unidentified); it is kept unprojected and its "
                "variance is large rather than small.")
        if status in ("weakly_active", "ambiguous"):
            # collected, applied after the loop: every row classifies
            # against the same snapshot, so results do not depend on
            # the order rows happen to be visited in
            row_corrections.append((sig_row, a))
    for _w, _a in row_corrections:
        # the row analog of the variable value correction: remove a
        # kept row's own barrier weight from the reduced block
        R_corr = R_corr - _w * np.outer(_a, _a)
    ns = SimpleNamespace()
    ns.act = act
    ns.mu = mu
    ns.R_W = R_W
    ns.sig_fit = sig_fit
    ns.floor = floor
    ns.active = active
    ns.free = [i for i in range(n_params) if i not in active]
    ns.R_corr = R_corr
    ns.bind_normals = bind_normals
    ns.row_corrections = row_corrections
    ns.R_exact = R_exact
    return ns


class Information(_ParamMatrix):
    """Observed or expected information over the fitted block, from
    information(). Keyed like Covariance: info[m.k1, m.k2] (either
    order), info[m.k1] for a diagonal entry; `matrix` is the dense
    numpy array in `params` (declaration) order. Natural units, no
    sigma^2 anywhere: for the homoscedastic Lagrangian case,
    covariance() equals 2*sigma^2 * inv(information()) on the free
    block. eigen() supports identifiability diagnosis directly: a
    near-zero eigenvalue is a direction the data does not inform, and
    its eigenvector names the parameter combination."""

    _who = "information"

    def __init__(self, params, matrix, conditioned_on=()):
        super().__init__(params, matrix)
        self.params = self._params
        self._conditioned = conditioned_on

    @property
    def conditioned_on(self):
        """Strongly active variables OUTSIDE the block: the matrix is
        conditional on those bounds (roadmap item 3); empty when none.
        Computed on first access (one backsolve per near-bound
        candidate) and cached, so calls that never read it pay
        nothing. Until first access the pending computation keeps the
        sensitivity session, and so the held KKT factor, alive; read
        it (or discard the result) to release them."""
        if callable(self._conditioned):
            self._conditioned = self._conditioned()
        return self._conditioned

    def eigen(self):
        """(eigenvalues, eigenvectors) of the information matrix,
        eigenvalues ascending, eigenvectors[:, i] in `params` order.
        Small eigenvalues flag poorly informed directions; a negative
        one means the point is not a least-squares minimum along that
        direction (Lagrangian form only; Gauss-Newton is PSD by
        construction).

        Each eigenvector's sign is pinned so the direction it names
        reproduces across machines: its largest-magnitude component
        is positive (see _pin_eigenvector_signs)."""
        w, v = np.linalg.eigh(self.matrix)
        return w, _pin_eigenvector_signs(v)


def _classify_ratio(r, mu):
    """The activity rule of covariance roadmap item 0, applied at the
    reduced fitted block. Mirrors pounce_sensitivity::activity with one
    deliberate divergence: the Rust classifier maps q below the floor
    to `unidentified` unconditionally, while the caller here first
    checks the ratio, because at the reduced level a huge Sigma
    cancelling inside q_red would otherwise misfile a strongly active
    entry as unidentified. Exposing the rule from pounce-py so one
    implementation serves both is item-2 follow-up."""
    if mu > 1e-4:
        if r < 1e-1:
            return "inactive"
        if r > 1e1:
            return "strongly_active"
        return "ambiguous"
    if r < np.sqrt(mu):
        return "inactive"
    if r > 1.0 / np.sqrt(mu):
        return "strongly_active"
    if 1e-1 <= r <= 1e1:
        return "weakly_active"
    return "ambiguous"


def _free_nullspace(bind_normals, free):
    """Projection basis over the free fitted coordinates: the null
    space of the binding row normals restricted to `free`. A normal
    that vanished with the pinned coordinates is dropped (its content
    is already excluded by the free restriction)."""
    nf = len(free)
    rows_ = []
    for a in bind_normals:
        af = np.asarray(a)[free]
        nn = float(np.linalg.norm(af))
        if nn > 1e-12:
            rows_.append(af / nn)
    A = np.array(rows_) if rows_ else np.zeros((0, nf))
    return _nullspace(A)


def _nullspace(A):
    """Orthonormal basis of the null space of A's rows (columns of the
    returned matrix); the projection basis Z of item 1's row handling."""
    if A.shape[0] == 0:
        return np.eye(A.shape[1])
    _, sv, vh = np.linalg.svd(A, full_matrices=True)
    tol = max(A.shape) * np.finfo(float).eps * (sv[0] if sv.size else 1.0)
    rank = int(np.sum(sv > tol))
    return vh[rank:].T


def _resolve_wrt(session, wrt, who):
    """Normalize wrt= into the block: an ordered list of variable data
    objects with their full-x (.col) rows. Accepted forms: None (the
    declared fitted block, exactly the prior behavior), a Var component
    (scalar or indexed: every member), an indexed slice (m.x[2, :]), a
    (Var, iterable) pair (var[t] for t in the iterable), a single
    VarData, or an iterable mixing any of these. Duplicates are an
    error: a repeated coordinate makes the block singular by
    construction."""
    if wrt is None:
        params = list(session.fit_rows.keys())
        if not params:
            raise RuntimeError(
                f"{who}: no fitted parameters were declared; flag the "
                "fitted variables with declare_fitted() before the "
                "solve, or select a block explicitly with wrt=")
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
                f"{who}: wrt takes variables, not names; got {obj!r}")
        try:
            it = iter(obj)                 # slice or plain iterable
        except TypeError:
            raise TypeError(
                f"{who}: wrt element {obj!r} is not a Pyomo variable "
                "or an iterable of them") from None
        for el in it:
            yield from leaves(el)

    params, rows, seen = [], [], set()
    for v in leaves(wrt):
        name = getattr(v, "name", None)
        if name is None:
            raise TypeError(
                f"{who}: wrt element {v!r} is not a Pyomo variable")
        try:
            r = session.var_entry(name)
        except ValueError as e:
            raise ValueError(f"{who}: wrt member {e}") from None
        if id(v) in seen:
            raise ValueError(
                f"{who}: wrt lists {name} twice; a repeated coordinate "
                "makes the block singular by construction")
        seen.add(id(v))
        params.append(v)
        rows.append(r)
    if not params:
        raise ValueError(f"{who}: wrt resolved to an empty block")
    return params, rows


def _subblock_information(session, rows, dof, who):
    """Marginal information of a proper sub-block of the fitted set,
    by Schur complement of the EXACT tangent R over the fitted block:
    never inverts a covariance, so a pinned member costs no digits.
    Pinned fitted variables OUTSIDE the block are conditioned on
    (their rows and columns are dropped: they are pinned, not
    profiled); free ones are profiled out (the Schur step). The
    matrix identity behind it: the inverse of a submatrix of an
    inverse IS the Schur complement, so away from barrier
    contamination this equals inv(M_B) exactly; the point is the
    construction, not the value. Returns None when not applicable
    (the block is not inside a square fitted set, or the fitted level
    carries binding rows, whose projection does not compose simply
    with marginalization) and the caller falls back to the corrected
    reduction off the factor."""
    fit_params = list(session.fit_rows.keys())
    if len(fit_params) != dof:
        return None
    fit_rows_ = [session.fit_rows[fp] for fp in fit_params]
    pos = {r: i for i, r in enumerate(fit_rows_)}
    if any(r not in pos for r in rows):
        return None
    dim = session.solver.kkt_dim
    fkrows = [session.primal_row(r, f"{who}(fitted)") for r in fit_rows_]
    zcols_f = []
    for kr in fkrows:
        e = np.zeros(dim)
        e[kr] = 1.0
        zcols_f.append(np.asarray(session.solver.kkt_solve(e)))
    M_f = np.array([[zcols_f[j][fkrows[i]] for j in range(dof)]
                    for i in range(dof)])
    M_f = 0.5 * (M_f + M_f.T)
    with warnings.catch_warnings():
        # the fitted-level classification is scaffolding here; its
        # warnings belong to the block-level pass the caller runs
        warnings.simplefilter("ignore")
        cb_f = _classify_fitted_block(session, fit_params, fit_rows_,
                                      M_f, zcols_f,
                                      who=f"{who}(fitted)",
                                      wording="fitted")
    if cb_f.bind_normals:
        return None
    R_f = cb_f.R_exact
    if R_f is None:
        R_f = _tangent_reduced_hessian(session, M_f, zcols_f, who)
    keep = [pos[r] for r in rows]
    kept = set(keep)
    pinned_f = set(cb_f.active)
    o_free = [i for i in range(dof)
              if i not in kept and i not in pinned_f]
    sel = keep + o_free
    Rs = R_f[np.ix_(sel, sel)]
    nb = len(keep)
    if o_free:
        A_ = Rs[:nb, :nb]
        B_ = Rs[:nb, nb:]
        D_ = Rs[nb:, nb:]
        try:
            R_B = A_ - B_ @ np.linalg.solve(D_, B_.T)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"{who}: the free fitted coordinates outside the block "
                "are singular, so the exact Schur route is unavailable; "
                "falling back to the corrected reduction off the "
                "factor, which loses digits at tight mu on pinned "
                "members.")
            return None
    else:
        R_B = Rs[:nb, :nb]
    return 0.5 * (R_B + R_B.T)


def _conditioned_on(session, act, rows, who):
    """Strongly active variables outside the block. Their Sigma stays
    in the held factor and drives the coupling through them to zero as
    mu falls, so the block's numbers are the values conditional on
    those bounds, not the marginal over them. Returned with the matrix
    rather than warned: it is a property of the answer, not a defect.

    Identification is item 1's reduced-level rule applied to each
    candidate as a singleton block: one backsolve gives (K^-1)_ii, the
    effective reduced curvature is |1/(K^-1)_ii - Sigma_i| (a pinned
    variable in the residual idiom has zero RAW curvature, which is
    why the raw report calls it unidentified), and the shipped ratio
    edges make the call. Scale-invariant, same theory as the block
    members. Candidates pass a cheap Sigma > sqrt(mu) prefilter first,
    so only near-bound variables pay the backsolve; below the
    cancellation floor q_red is clamped to it, exactly as the
    block-level rule does."""
    inside = set(int(r) for r in rows)
    mu = float(act["mu"])
    pre = np.sqrt(mu) if mu > 0 else 0.0
    dim = session.solver.kkt_dim
    out = []
    for idx, st in enumerate(act["var_status"]):
        if idx in inside or st in ("unbounded", "fixed", "equality",
                                   "inactive"):
            continue
        sig = float(act["var_sigma"][idx])
        if st == "strongly_active":
            pinned = True
        elif sig <= pre:
            continue
        else:
            krow = session.primal_row(idx, f"{who} conditioned_on")
            e = np.zeros(dim)
            e[krow] = 1.0
            kii = float(np.asarray(session.solver.kkt_solve(e))[krow])
            if kii == 0.0:
                continue
            q_red = abs(1.0 / kii - sig)
            # clamp to the floor rather than refuse, exactly as the
            # block-level rule does: a huge Sigma cancelling inside
            # q_red would otherwise misfile a strongly active variable
            floor = np.sqrt(np.finfo(float).eps) * max(1.0, abs(1.0 / kii))
            pinned = (_classify_ratio(sig / max(q_red, floor), mu)
                      == "strongly_active")
        if pinned:
            v = session.orig_var(session.var_names[idx])
            out.append(v if v is not None else session.var_names[idx])
    return tuple(out)


def _rank_deficient(A):
    """True when the symmetric block `A` is rank-deficient, tested on
    its diagonally scaled form so the verdict is a statement about
    COLLINEARITY and not about units.

    `np.linalg.matrix_rank` thresholds the singular values at
    `sigma_max * n * eps`, which is relative to the largest singular
    value and so is not invariant under rescaling the coordinates
    against each other: a covariance block carries the SQUARE of any
    unit spread between its members, so two perfectly well-determined
    parameters ~1e8 apart in magnitude (a rate prefactor against a
    reaction order) push `cond(M)` past the default tolerance and read
    as dependent on unit spread alone. Scaling by `sqrt(|diag|)` first
    — the correlation form — makes the test the same kind of
    scale-invariant ratio as the block-membership rule (roadmap item 1)
    and the conditioned_on rule, rather than a threshold that tracks
    the user's choice of units. Second review of gh #466.

    A zero diagonal leaves its own row and column unscaled: there is no
    scale to divide by, and a genuinely zero row is rank-deficient
    either way."""
    n = A.shape[0]
    if n == 0:
        return False
    d = np.sqrt(np.abs(np.diag(A)))
    d = np.where(d > 0.0, d, 1.0)
    return int(np.linalg.matrix_rank(A / np.outer(d, d))) < n


class _SingularBlock(RuntimeError):
    """The requested block of the inverse KKT matrix is singular.
    A dedicated type so callers that rescue this case (the wrt
    dependent-block paths) do not do control flow on message text
    (review of gh #466)."""


def _minv(M, who="covariance"):
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError as e:
        raise _SingularBlock(
            f"{who}: the requested block of the inverse KKT matrix "
            "is singular; the block members are linearly "
            "dependent (structurally unidentifiable)") from e


def covariance(model, sigma_sq=None, n_data=None, hessian="lagrangian",
               wrt=None, max_pdpert=None):
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
    SOLVE-TIME objective value on trust -- writing into the model
    first, the receding-horizon pattern of estimate(), does not change
    the answer). With multiple labeled groups the heteroscedastic
    sandwich covariance is reported.

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

    wrt= selects the block (covariance roadmap item 3): any of the
    solve's variables, not only the declared fitted ones, given as a
    Var (scalar or indexed), an indexed slice (m.x[2, :]), a
    (Var, iterable) pair, data objects, or a list mixing these; None
    (default) is the declared fitted block, exactly the prior
    behavior. Each call re-reduces onto its own argument, so one solve
    serves as many blocks as are asked about, each getting that
    block's MARGINAL (everything else profiled out). Sigma estimation
    always divides by the fit's own degrees of freedom, a property of
    the solve, not the block. A rank-deficient block (more
    coordinates than the fit has degrees of freedom, e.g. a predicted
    trajectory: the prediction-band case) gets the homoscedastic
    Lagrangian marginal, with membership handling bypassed. Strongly
    active variables OUTSIDE the block come back on the result as
    .conditioned_on: the matrix is conditional on those bounds, not
    marginal over them.

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
    in Pyomo.

    Bound and constraint activity is classified from the solve's own
    barrier geometry (the ratio of each direction's barrier weight to
    its curvature; pounce's covariance roadmap item 1), not from a
    slack threshold. A STRONGLY ACTIVE bound pins its parameter: zero
    variance, correlation entries 0, conditional on the bound, with a
    warning. A WEAKLY ACTIVE bound (slack and multiplier vanish
    together) is KEPT: the parameter keeps its full finite variance,
    corrected for the barrier weight the held factor carries, and a
    warning notes the nonstandard boundary asymptotics. AMBIGUOUS
    (loosely converged) and UNIDENTIFIED (curvature below the model's
    own noise scale) parameters stay in the free block with warnings.
    A strongly active inequality CONSTRAINT involving fitted
    parameters pins a combination rather than a coordinate: the matrix
    is projected on the constraint's null space, going singular by one
    per binding row, and the surviving correlations say what the data
    still determines (for a + b <= cap binding, corr(a, b) = -1: only
    the difference is determined). The same limit written as a bound
    or as a row returns the same matrix (jkitchin/pounce#362).

    max_pdpert refuses rather than answering when the converged KKT
    factor carries an inertia correction larger than the value given.
    This inverts that factor, so a perturbed one answers for a nearby
    problem rather than this one. It warns either way, and the cap is
    the caller choosing to stop instead of to read the warning.
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
            "declare_residual()), or retain_kkt() for "
            "wrt= queries with nothing declared, then solve with "
            "SolverFactory('pounce'), SolverFactory('pounce_v2') or the "
            "contrib SolverFactory('pounce') first")
    # the block: the declared fitted parameters by default, or any
    # block of the solve's variables via wrt= (each call re-reduces
    # onto its own argument, so one solve serves as many blocks as are
    # asked about, each getting that block's marginal)
    params, rows = _resolve_wrt(session, wrt, "covariance")
    n_params = len(params)
    wording = "fitted" if wrt is None else "block"
    # sigma estimation divides by the FIT's degrees of freedom, a
    # property of the solve, not of the block being asked about
    n_fit = len(session.fit_rows)

    # ── guardrails ────────────────────────────────────────────────────────
    _check_margins(None, max_pdpert, "covariance")
    _refuse_on_pdpert(session, max_pdpert, "covariance")
    pert = np.asarray(session.solver.kkt_perturbations)
    if pert.any():
        warnings.warn(
            "covariance: the held KKT factor carries inertia-correction "
            f"perturbations {pert.tolist()}, so the covariance is "
            "regularized rather than exact. Linearly dependent (structurally"
            " unidentifiable) parameters are the usual cause.")
    # ── parameter block of the inverse KKT matrix ─────────────────────────
    # `rows` is full-x (the space of the activity report and of
    # row_normal, both read below); `krows` is the same variables as
    # factor rows. Keep the two apart -- they differ exactly when the
    # model has a fixed variable, and agree everywhere else.
    dim = session.solver.kkt_dim
    krows = [session.primal_row(r, f"covariance({p.name})")
             for r, p in zip(rows, params)]
    zcols = []
    for r in krows:
        e = np.zeros(dim)
        e[r] = 1.0
        zcols.append(np.asarray(session.solver.kkt_solve(e)))
    M = np.array([[zcols[j][krows[i]] for j in range(n_params)]
                  for i in range(n_params)])
    M = 0.5 * (M + M.T)

    # a rank-deficient EXPLICIT block (more coordinates than the fit
    # has degrees of freedom, e.g. a predicted trajectory) has a
    # perfectly well-defined marginal covariance, 2 sigma^2 M; what it
    # does not have is inv(M), which the membership classification
    # needs. Gate on the count: LAPACK does not reliably raise on a
    # structurally singular M (tiny nonzero pivots slip through and
    # give garbage, not an error).
    n_var, n_eq = _estimation_counts(session)
    deficient = None
    if wrt is not None:
        if n_params > n_var - n_eq:
            deficient = "count"
        elif _rank_deficient(M):
            # the count gate's own justification, one step to its
            # left: LAPACK does not reliably raise on a structurally
            # singular M, so a within-count dependent block needs its
            # own gate (fp-detectable dependence; anything softer is
            # caught by _SingularBlock below as a last resort).
            # Diagonally scaled, so a badly-scaled but well-determined
            # block is not refused for its units (second review)
            deficient = "dependent"
    if deficient is not None:
        cb = None
    else:
        try:
            cb = _classify_fitted_block(session, params, rows, M, zcols,
                                        wording=wording)
        except _SingularBlock:
            if wrt is None:
                raise
            deficient = "dependent"
            cb = None
    if cb is not None:
        sig_fit = cb.sig_fit
        floor = cb.floor
        R_corr = cb.R_corr
        bind_normals = cb.bind_normals
        row_corrections = cb.row_corrections

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
        if n_fit == 0:
            raise ValueError(
                "covariance: the noise variance must be estimated from "
                "the declared residuals, but no fitted parameters were "
                "declared, so the degrees of freedom for the estimate "
                "are unknown; pass sigma_sq= (known variance) or flag "
                "the fitted parameters with declare_fitted()")
        group_sigma = {}
        for g, rws in groups.items():
            n_g = len(rws)
            if n_g <= n_fit:
                raise ValueError(
                    f"covariance: residual group {g!r} has {n_g} members, "
                    f"not more than the {n_fit} fitted parameters; "
                    "cannot estimate its noise variance")
            ssr_g = float(np.sum(session.base_x[rws] ** 2))
            group_sigma[g] = ssr_g / (n_g - n_fit)
    elif n_data is not None:
        if n_fit == 0:
            raise ValueError(
                "covariance: n_data= estimates the noise variance, but "
                "no fitted parameters were declared, so the degrees of "
                "freedom for the estimate are unknown; pass sigma_sq= "
                "(known variance) or flag the fitted parameters with "
                "declare_fitted()")
        if n_data <= n_fit:
            raise ValueError(
                f"covariance: n_data ({n_data}) must exceed the number of "
                f"fitted parameters ({n_fit})")
        # the objective value AT THE SOLVE, not evaluated on the live
        # model: pyo.value(objective) reads the model's current variable
        # and Param values, so anything written after the solve (a
        # measurement, a warm start for the next horizon) would silently
        # rescale the covariance (gh #426)
        if not np.isfinite(session.base_obj):
            raise RuntimeError(
                "covariance: the solve reported no usable objective value "
                f"({session.base_obj}), so n_data= cannot estimate the "
                "noise variance. Pass sigma_sq= (known variance), or "
                "declare the residual container with declare_residual().")
        ssr = session.base_obj
        group_sigma = {None: ssr / (n_data - n_fit)}
    else:
        raise ValueError(
            "covariance: the noise variance is unknown; declare the "
            "residual container(s) with declare_residual(), or pass "
            "sigma_sq= (known variance), or pass n_data= (data count, "
            "with the SSR taken from the solve-time objective value)")

    # ── assemble ──────────────────────────────────────────────────────────
    # Pooled covariance when there is one group or all group variances are
    # equal to relative tolerance; otherwise the heteroscedastic sandwich.
    sig_vals = list(group_sigma.values())
    homoscedastic = len(sig_vals) <= 1 or (
        max(sig_vals) - min(sig_vals)
        <= 1e-12 * max(abs(v) for v in sig_vals)
    )

    def minv():
        return _minv(M)

    def group_jacobians():
        # The Jacobian rows are recovered from the same backsolves: the
        # residual rows of the z-columns equal J * inv(d2f/dp2), so
        # J = Z_r * inv(M).
        # No Sigma correction is needed on this path, by an exact
        # identity: the residual rows of the K-inverse columns are J
        # times the W-based parameter sensitivities, Z_r = J @ M, so
        # Z_r @ inv(M) = J exactly and the factor's barrier weight
        # cancels regardless of Sigma. The Lagrangian branch corrects
        # R_W because it USES the W-based reduced Hessian; Gauss-Newton
        # rebuilds from the exact J instead. Pinned empirically by the
        # GN counterpart of the weakly-active analytic test.
        Mi = minv()
        out = {}
        for g, rws in groups.items():
            # res_rows is full-x like fit_rows; zcols are factor rows
            Zr = np.array([[zcols[j][session.primal_row(r, "covariance")]
                            for j in range(n_params)]
                           for r in rws])
            out[g] = Zr @ Mi                  # d r_g / d p
        return out

    if cb is None:
        # marginal-only path for a rank-deficient explicit block. The
        # Gauss-Newton and heteroscedastic routes profile Jacobians
        # through inv(M), so only the homoscedastic Lagrangian marginal
        # exists here.
        reason = ("more coordinates than the fit has degrees of "
                  "freedom" if deficient == "count"
                  else "linearly dependent coordinates")
        if hessian == "gauss-newton" or not homoscedastic:
            raise RuntimeError(
                f"covariance: the wrt block is rank-deficient "
                f"({reason}), so the profiled Jacobians behind "
                "hessian='gauss-newton' and per-group noise are not "
                "defined; only the homoscedastic hessian='lagrangian' "
                "marginal is available for this block")
        act = session.solver.classify_activity()
        flagged = [p.name for i, p in enumerate(params)
                   if act["var_status"][rows[i]] not in
                   ("inactive", "unbounded", "fixed", "equality")]
        if flagged:
            warnings.warn(
                f"covariance: block members {flagged} have bound "
                f"activity, but the block is rank-deficient ({reason}), "
                "so the "
                "membership handling (pinned parameters, binding rows) "
                "is unavailable; the marginal is returned without it.")
        cov = 2.0 * sig_vals[0] * M
        cov = 0.5 * (cov + cov.T)
        sig_out = (next(iter(group_sigma.values()))
                   if len(group_sigma) == 1 and None in group_sigma
                   else group_sigma)
        return Covariance(
            params, cov, sig_out,
            lambda: _conditioned_on(session, act, rows, "covariance"))

    # Active-bound projection: the covariance is computed in the free
    # (off-bound) directions and embedded with zero rows/cols for the
    # pinned parameters, i.e. the covariance conditional on the active
    # set. Restricting the INFORMATION matrix to the free block and
    # inverting (not restricting the inverse) is the curve_fit
    # _projected_covariance construction.
    free = cb.free

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
        Zb = _free_nullspace(bind_normals, free)
        try:
            if Zb.shape[1] == 0:
                Ginv = np.zeros((len(free), len(free)))
            else:
                Ginv = Zb @ np.linalg.inv(Zb.T @ G @ Zb) @ Zb.T
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "covariance: the Gauss-Newton matrix J^T J is singular; "
                "the block members are linearly dependent in the "
                "residual Jacobian") from e
        if homoscedastic:
            cov = embed(sig_vals[0] * Ginv)
        else:
            B = np.zeros((len(free), len(free)))
            for g, Jg in Js.items():
                B += group_sigma[g] * (Jg.T @ Jg)
            cov = embed(Ginv @ B @ Ginv)
    else:
        if (not bind_normals and not row_corrections
                and len(free) == n_params
                and float(np.abs(sig_fit).max()) <= floor):
            # nothing active, nothing to correct above noise scale: M
            # is already the answer, and skipping inv(inv(M)) spares
            # the conditioning round-trip on the common all-free path
            Mc = M
        else:
            # R_corr is the reduced Hessian with the item-1 value
            # corrections applied: the fitted rows' own barrier
            # diagonal subtracted (a weakly active kept parameter
            # reports its true curvature q, not the factor's 2q) and
            # kept rows' barrier weight removed. Active rows and
            # binding normals never enter: coordinates are excluded by
            # the free restriction, directions are annihilated by the
            # projection basis Z (which also annihilates the binding
            # rows' huge barrier weight, exactly).
            Rff = R_corr[np.ix_(free, free)]
            Zb = _free_nullspace(bind_normals, free)
            try:
                if Zb.shape[1] == 0:
                    Mc = np.zeros((len(free), len(free)))
                else:
                    Mc = Zb @ np.linalg.inv(Zb.T @ Rff @ Zb) @ Zb.T
            except np.linalg.LinAlgError as e:
                raise RuntimeError(
                    "covariance: the reduced Hessian restricted to the "
                    "free (off-bound, off-constraint) members is "
                    "singular; the remaining free block members are "
                    "linearly dependent"
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
    return Covariance(
        params, cov, sig_out,
        lambda: _conditioned_on(session, cb.act, rows, "covariance"))


def information(model, hessian="lagrangian", wrt=None, max_pdpert=None):
    """The information matrix of the fitted parameters: the reduced
    Hessian over the declared block, the un-inverted sibling of
    covariance(), from the same single solve.

    Natural units and no sigma^2 anywhere (the core's convention;
    covariance() carries the 2*sigma^2 on top): for a homoscedastic
    Lagrangian fit, covariance() equals 2*sigma^2*inv(information())
    on the free block. hessian= selects the form exactly as in
    covariance(): "lagrangian" (default) is the observed information,
    built by tangent recovery against the held factorization (the
    K-inverse columns' x-blocks are T*M, so T = Zx*inv(M) exactly and
    R = T'HT with the exact Lagrangian Hessian: machine precision for
    equality and bound activity, no subtraction against the
    barrier-augmented factor; a binding inequality row leaves ~1e-6
    relative residue through its slack barrier). "gauss-newton" is the expected information 2*J'J, with J
    recovered over ALL fitted parameters and sliced afterwards, so the
    pinned rows exist to build their disposition from (requires
    declared residuals, as in covariance()).

    Membership and warnings follow covariance() exactly (item 1's
    table): a free parameter's row is the reduced Hessian; a strongly
    active parameter's block is S, the reduction onto the pinned set,
    NOT a zero row, because zero information is the opposite of what a
    pinned parameter carries; cross blocks between free and pinned are
    zero. Binding constraint rows project the free block on both sides
    (the pseudo-inverse of the projected covariance). An indefinite
    Lagrangian block is returned as computed with a warning naming
    Gauss-Newton as the PSD alternative: refusing would withhold the
    finding that the point is not a minimum or the model is
    over-parameterized.

    wrt= selects the block exactly as in covariance() (roadmap item
    3): the declared fitted block by default, or any block of the
    solve's variables. A block that parameterizes the constraint
    manifold (size equal to the fit's degrees of freedom, the square
    structure above) gets the exact tangent construction; a smaller
    block reduces off the held factor with the item-1 corrections
    (that route's documented precision); a rank-deficient block
    carries no information matrix and is refused toward covariance().
    Strongly active variables outside the block come back as
    .conditioned_on.

    Returns an Information object keyed by the declared variables'
    data objects: info[m.A, m.k], info.matrix, info.eigen().

    max_pdpert refuses rather than answering when the converged KKT
    factor carries an inertia correction larger than the value given.
    This inverts that factor, so a perturbed one answers for a nearby
    problem rather than this one. It warns either way, and the cap is
    the caller choosing to stop instead of to read the warning.
    """
    if hessian not in ("lagrangian", "gauss-newton"):
        raise ValueError(
            "information: hessian must be 'lagrangian' or 'gauss-newton', "
            f"got {hessian!r}")
    reg = model.__dict__.get(_REG)
    session = reg.session if reg else None
    if session is None:
        raise RuntimeError(
            "no sensitivity session: declare_fitted() (and optionally "
            "declare_residual()), or retain_kkt() for "
            "wrt= queries with nothing declared, then solve with "
            "SolverFactory('pounce'), SolverFactory('pounce_v2') or the "
            "contrib SolverFactory('pounce') first")
    params, rows = _resolve_wrt(session, wrt, "information")
    n_params = len(params)
    wording = "fitted" if wrt is None else "block"

    _check_margins(None, max_pdpert, "information")
    _refuse_on_pdpert(session, max_pdpert, "information")
    pert = np.asarray(session.solver.kkt_perturbations)
    if pert.any():
        warnings.warn(
            "information: the held KKT factor carries inertia-correction "
            f"perturbations {pert.tolist()}, so the information is "
            "regularized rather than exact; the isotropic delta_w lands on "
            "the free block and survives the projection.")

    # `rows` is full-x (the space _classify_fitted_block reads the
    # activity report and row_normal in); `krows` is the same variables
    # as factor rows (gh #450): they differ exactly when the model has
    # a fixed variable, and agree everywhere else
    dim = session.solver.kkt_dim
    krows = [session.primal_row(r, f"information({p.name})")
             for r, p in zip(rows, params)]
    zcols = []
    for r in krows:
        e = np.zeros(dim)
        e[r] = 1.0
        zcols.append(np.asarray(session.solver.kkt_solve(e)))
    M = np.array([[zcols[j][krows[i]] for j in range(n_params)]
                  for i in range(n_params)])
    M = 0.5 * (M + M.T)

    n_var, n_eq = _estimation_counts(session)
    def _refuse_rank(reason):
        raise RuntimeError(
            f"information: the wrt block is rank-deficient ({reason}), "
            "so it carries no information matrix; its covariance "
            "(covariance(model, wrt=...)) is the meaningful object "
            "for such a block")
    if wrt is not None:
        if n_params > n_var - n_eq:
            _refuse_rank("more coordinates than the fit has degrees "
                         "of freedom")
        if _rank_deficient(M):
            # the count gate one step to its left: LAPACK does not
            # reliably raise on a structurally singular M. Diagonally
            # scaled, so the refusal tracks collinearity and not the
            # user's units (second review)
            _refuse_rank("linearly dependent coordinates")
    try:
        cb = _classify_fitted_block(session, params, rows, M, zcols,
                                    who="information", wording=wording)
    except _SingularBlock:
        if wrt is None:
            raise
        _refuse_rank("linearly dependent coordinates")
    free, active = cb.free, cb.active

    if hessian == "gauss-newton":
        groups = dict(session.res_rows)
        if not groups:
            raise ValueError(
                "information: hessian='gauss-newton' needs declared "
                "residuals (declare_residual()); the residual Jacobian is "
                "recovered from their rows. Without residual variables "
                "only the hessian='lagrangian' default is available.")
        Mi = _minv(M, "information")
        R = np.zeros((n_params, n_params))
        for g, rws in groups.items():
            # slice LAST: J over the whole fitted block, so the pinned
            # rows exist for S below; res_rows is full-x like fit_rows,
            # zcols are factor rows (gh #450)
            Zr = np.array([[zcols[j][session.primal_row(r, "information")]
                            for j in range(n_params)]
                           for r in rws])
            Jg = Zr @ Mi
            R += 2.0 * (Jg.T @ Jg)
    else:
        R = cb.R_exact
        if R is None:
            if n_var - n_eq == n_params:
                R = _tangent_reduced_hessian(session, M, zcols,
                                             "information")
            elif wrt is not None:
                # a sub-block of the fitted set gets its marginal by
                # Schur complement of the exact tangent R over the
                # fitted block (never inverts a covariance, exact even
                # with pinned members); any other explicit block
                # reduces off the held factor with the item-1
                # corrections, which is benign for free coordinates
                # (no barrier term in the slice)
                R = _subblock_information(session, rows,
                                          n_var - n_eq, "information")
                if R is None:
                    R = cb.R_corr
            else:
                raise RuntimeError(
                    "information: hessian='lagrangian' requires the "
                    "square estimation structure (the equality "
                    "constraints determine the non-fitted variables "
                    "given the fitted block); this model has "
                    f"{n_var} factor variables, {n_eq} equalities and "
                    f"{n_params} fitted parameters, so "
                    f"{n_var} - {n_eq} != {n_params} and the tangent "
                    "recovery is not defined. hessian='gauss-newton' "
                    "does not need it.")

    info_mat = np.zeros((n_params, n_params))
    if free:
        Rff = R[np.ix_(free, free)]
        Zb = _free_nullspace(cb.bind_normals, free)
        if Zb.shape[1] == Rff.shape[0]:
            info_ff = Rff
        else:
            # projected on both sides: the pseudo-inverse of the
            # projected covariance (item 1's rule, identical in both
            # accessors)
            info_ff = Zb @ (Zb.T @ Rff @ Zb) @ Zb.T
        info_mat[np.ix_(free, free)] = info_ff
        if hessian == "lagrangian" and _indefinite(info_ff):
            warnings.warn(
                "information: the Lagrangian free block is indefinite "
                "(returned as computed): the point is not a "
                "least-squares minimum along some direction, or the "
                "model is over-parameterized. "
                "hessian='gauss-newton' is the PSD alternative.")
    if active:
        # the pinned set's disposition is S, the reduction onto the
        # pinned block, not a zero row: zero information is the
        # opposite of what a pinned parameter carries. Conditional on
        # the rest of the pinned set (item 1's caveat).
        if free:
            # rank-gate the free block before the solve: whether LAPACK
            # raises on a singular system is BLAS-dependent (the CI
            # wheel job's fresh numpy returned garbage where the local
            # build raised), the same non-determinism the wrt gates
            # exist for; the exception clause stays as the last resort.
            # Applies on the DEFAULT path too, not only under wrt=: a
            # numerically dependent free block now refuses where it
            # could previously return a large-but-meaningless S
            # (CHANGELOG "Changed"). Diagonally scaled like the wrt
            # gates, so an ill-scaled but well-determined free block
            # keeps its answer (second review)
            R_FF = R[np.ix_(free, free)]
            singular = _rank_deficient(R_FF)
            if not singular:
                try:
                    S = (R[np.ix_(active, active)]
                         - R[np.ix_(active, free)]
                         @ np.linalg.solve(R_FF,
                                           R[np.ix_(free, active)]))
                except np.linalg.LinAlgError:
                    singular = True
            if singular:
                raise RuntimeError(
                    "information: the free block is singular, so the "
                    "pinned parameters' conditional information S is not "
                    "defined; the free block members are linearly "
                    "dependent")
        else:
            S = R[np.ix_(active, active)]
        info_mat[np.ix_(active, active)] = S

    return Information(
        params, info_mat,
        lambda: _conditioned_on(session, cb.act, rows, "information"))
