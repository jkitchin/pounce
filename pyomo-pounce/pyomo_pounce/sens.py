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

    declare_estimated(m.A); declare_estimated(m.k)
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
    """Per-model registry of declared statistical roles. Deepcopy-aware so
    model.clone() (and the sensitivity surgery's own clone) works cleanly:
    declared components follow the clone through the memo, while the
    session -- which holds solver handles tied to one converged
    factorization -- is deliberately not copied (a clone has no solve of
    its own yet)."""

    def __init__(self):
        self.params = []          # pinned inputs: gradient()/estimate()
        self.estimated = []       # free fitted variables: covariance()
        self.residuals = []       # (container, group) pairs: sigma^2
        self.session = None

    def __deepcopy__(self, memo):
        import copy
        new = _Registry()
        memo[id(self)] = new
        new.params = [copy.deepcopy(p, memo) for p in self.params]
        new.estimated = [copy.deepcopy(p, memo) for p in self.estimated]
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


def declare_estimated(*variables):
    """Flag one or more FREE Vars (scalar or indexed) as estimated
    parameters of a least-squares problem: after one ordinary solve,
    covariance() reports their asymptotic uncertainty. The variables stay
    free in the solve; do not fix them."""
    for var in variables:
        _registry(var.model()).estimated.append(var)


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
    return bool(reg and (reg.params or reg.estimated or reg.residuals))


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


def sens_solve(model, tee=False, sens_params=None, estimated=None,
               residuals=None):
    """Solve `model` in-process with POUNCE and keep the KKT factorization
    for gradient()/estimate()/covariance(). Called automatically by
    SolverFactory('pounce').solve() when declarations are present; the
    keyword arguments are the explicit (call-time) form of the
    declarations and register the components exactly as the declare_*
    functions do. Returns a Pyomo SolverResults, like an ordinary solve."""
    import pounce

    # explicit form: register call-time components before proceeding
    for p in (sens_params or []):
        declare_sens_param(p)
    for v in (estimated or []):
        declare_estimated(v)
    for item in (residuals or []):
        if isinstance(item, tuple):
            declare_residual(*item)
        else:
            declare_residual(item)

    reg = model.__dict__[_REG]
    if reg.params:
        # pinned inputs need the sensitivity-toolbox surgery (on a clone)
        si = SensitivityInterface(model, clone_model=True)
        si.setup_sensitivity(reg.params)
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

    prob = pounce.Problem(nl.n, nl.m, _NlBridge(nl),
                          lb=nl.x_l, ub=nl.x_u, cl=nl.g_l, cu=nl.g_u)
    solver = pounce.Solver(prob)
    x, info = solver.solve(np.asarray(nl.x0))
    if tee:
        print(info.get("status_msg", info))
    if not solver.converged:
        raise RuntimeError(
            f"pounce did not converge: {info.get('status_msg')}")

    pins = ComponentMap()
    con_alias = {}
    if si is not None:
        block = clone.component(SensitivityInterface.get_default_block_name())
        for i, (var, clone_param, list_idx, comp_idx) in enumerate(
                block._sens_data_list):
            con = block.paramConst[i + 1]
            orig_comp = reg.params[list_idx]
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

    # estimated parameters: their rows in the primal vector
    session.est_rows = ComponentMap()
    for comp in reg.estimated:
        for vd in _iter_data(comp):
            session.est_rows[vd] = var_names.index(vd.name)

    # residual groups: member rows per group key (None = the common pool)
    session.res_rows = {}
    for container, group in reg.residuals:
        rows = [var_names.index(rd.name) for rd in _iter_data(container)]
        session.res_rows.setdefault(group, []).extend(rows)

    reg.session = session

    # load the solution back onto the ORIGINAL model's variables (when the
    # solve ran on a clone; in the estimation-only path clone IS model and
    # this simply refreshes the same variables)
    for name, val in zip(var_names, session.base_x):
        ov = model.find_component(name)
        if ov is not None:
            ov.set_value(val, skip_validation=True)

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

    Keyed by the estimated variables' data objects — the free `Var`s
    flagged with `declare_estimated`, not Pyomo `Param`s — in `params`
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
            self.std_err = _ParamVector(self.params, se)
            self.correlation = _ParamMatrix(
                self.params, self.matrix / np.outer(se, se))

    def eigen(self):
        """(eigenvalues, eigenvectors) of the covariance matrix,
        eigenvalues ascending, eigenvectors[:, i] in `params` order.
        An eigenvalue much larger than the rest flags a poorly
        identified direction: its eigenvector gives the parameter
        combination the data cannot pin down."""
        return np.linalg.eigh(self.matrix)


def covariance(model, sigma_sq=None, n_data=None):
    """Asymptotic covariance of the estimated parameters of a
    least-squares problem, from ONE ordinary solve.

    Workflow: declare the fitted variables with declare_estimated (they
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

    Returns a Covariance object keyed by the declared variables'
    data objects: cov[m.A, m.k], cov.std_err[m.A],
    cov.correlation[m.A, m.k], cov.matrix, cov.sigma_sq (float or
    per-group dict), cov.eigen().
    """
    reg = model.__dict__.get(_REG)
    session = reg.session if reg else None
    if session is None:
        raise RuntimeError(
            "no sensitivity session: declare_estimated() (and optionally "
            "declare_residual()) then solve with SolverFactory('pounce') "
            "first")
    params = list(session.est_rows.keys())
    n_params = len(params)
    if n_params == 0:
        raise RuntimeError(
            "covariance: no estimated parameters were declared; flag the "
            "fitted variables with declare_estimated() before the solve")

    # ── guardrails ────────────────────────────────────────────────────────
    pert = np.asarray(session.solver.kkt_perturbations)
    if pert.any():
        warnings.warn(
            "covariance: the held KKT factor carries inertia-correction "
            f"perturbations {pert.tolist()}, so the covariance is "
            "regularized rather than exact. Linearly dependent (structurally"
            " unidentifiable) parameters are the usual cause.")
    lo, hi = np.asarray(session.nl.x_l), np.asarray(session.nl.x_u)
    for p in params:
        r = session.est_rows[p]
        xv = float(session.base_x[r])
        tol = 1e-6 * (1.0 + abs(xv))
        if xv - lo[r] < tol or hi[r] - xv < tol:
            warnings.warn(
                f"covariance: estimated parameter {p.name} sits on its "
                "bound at the optimum; the asymptotic covariance is not "
                "valid for an active-bound parameter.")

    # ── parameter block of the inverse KKT matrix ─────────────────────────
    dim = session.solver.kkt_dim
    rows = [session.est_rows[p] for p in params]
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
    if sigma_sq is not None:
        if isinstance(sigma_sq, dict):
            group_sigma = {g: float(s) for g, s in sigma_sq.items()}
        else:
            group_sigma = {g: float(sigma_sq) for g in (groups or {None: []})}
    elif groups:
        group_sigma = {}
        for g, rws in groups.items():
            n_g = len(rws)
            if n_g <= n_params:
                raise ValueError(
                    f"covariance: residual group {g!r} has {n_g} members, "
                    f"not more than the {n_params} estimated parameters; "
                    "cannot estimate its noise variance")
            ssr_g = float(np.sum(session.base_x[rws] ** 2))
            group_sigma[g] = ssr_g / (n_g - n_params)
    elif n_data is not None:
        if n_data <= n_params:
            raise ValueError(
                f"covariance: n_data ({n_data}) must exceed the number of "
                f"estimated parameters ({n_params})")
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
    if homoscedastic:
        s2 = sig_vals[0]
        cov = 2.0 * s2 * M
    else:
        # heteroscedastic sandwich: cov = A^-1 B A^-1 with A = d2f/dp2 and
        # B built from per-group residual Jacobians. The Jacobian rows are
        # recovered from the same backsolves: the residual rows of the
        # z-columns equal J * inv(d2f/dp2), so J = Z_r * inv(M).
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "covariance: the parameter block of the inverse KKT matrix "
                "is singular; the estimated parameters are linearly "
                "dependent (structurally unidentifiable)") from e
        B = np.zeros((n_params, n_params))
        for g, rws in groups.items():
            Zr = np.array([[zcols[j][r] for j in range(n_params)]
                           for r in rws])
            Jg = Zr @ Minv                    # d r_g / d p
            B += group_sigma[g] * (Jg.T @ Jg)
        # dtheta = -A^-1 * 2 J^T eps with A = d2f/dp2 = inv(M), so
        # cov = 4 M (sum_g sigma_g^2 Jg^T Jg) M; the single-group case
        # reduces to 2 sigma^2 M since J^T J = A/2.
        cov = 4.0 * M @ B @ M

    cov = 0.5 * (cov + cov.T)
    if np.diag(cov).min() < 0:
        warnings.warn(
            "covariance: negative variance on the diagonal; the point is "
            "probably not a least-squares minimum.")
    sig_out = (next(iter(group_sigma.values()))
               if len(group_sigma) == 1 and None in group_sigma
               else group_sigma)
    return Covariance(params, cov, sig_out)
