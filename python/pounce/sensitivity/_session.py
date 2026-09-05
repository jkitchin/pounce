"""The held-factor session the sensitivity analyses run against.

A :class:`SensSession` is *a solved NL problem plus the KKT factorization
the solve left behind*, with enough row bookkeeping for the analyses in
this package to address the model in the caller's own terms. It owns no
modelling layer: rows are integers, labels are whatever the caller keyed
them with, and nothing here knows what a Pyomo component is.

`pyomo_pounce` subclasses this, adding the component-keyed containers and
the model handle its own API needs; every analysis in this package works
against either, because each reads only the attributes declared here.
"""
import numpy as np


def row_index(names):
    """{name: position} for a `.col` / `.row` name list.

    The NL writer emits unique symbolic labels, so first-wins and
    last-wins agree; enumerate order matches `list.index` either way.
    """
    return {nm: i for i, nm in enumerate(names)}


class SensSession:
    """A converged solve, its held KKT factor, and the rows that name things.

    Parameters
    ----------
    nl :
        The problem the solve ran on, as returned by :func:`pounce.read_nl`.
        Used for bounds (`x_l`, `x_u`, `g_l`, `g_u`) and for evaluating
        `constraints()` and `gradient()` at a perturbed point.
    solver : pounce.Solver
        A solver whose `solve()` has converged, so the factor is live.
    var_names, con_names : list of str
        `.col` and `.row` order — the user's FULL-x and full-g spaces.
    pins : mapping, optional
        Ordered mapping {key: pin row} for the declared parameters, one
        row each. Keys are opaque: this package uses them only to
        preserve order and to label results.
    pin_coefs, pin_bases : mapping, optional
        `d(row)/d(param)` and the param's solve-time value, keyed like
        `pins`, for parameters pinned through a defining equality the
        caller wrote rather than a plain `var == p` row.
    fit_rows : mapping, optional
        Ordered mapping {key: full-x column} for the fitted parameters of
        an estimation model.
    res_rows : dict, optional
        {group: [full-g rows]} for the residual rows of an estimation
        model. `group` may be None for the ungrouped case.
    con_alias : dict, optional
        Original constraint name -> the name the solved model carries,
        for callers whose solve ran on a rewritten copy.
    moved_bounds : dict, optional
        Variable name -> (lb, ub) that a bound held when the caller moved
        it into a row. NOT refreshed when the bound's parameter later
        moves: a reader needing the current bound evaluates the moved row.
    """

    def __init__(self, nl, solver, var_names, con_names, *, pins=None,
                 pin_coefs=None, pin_bases=None, fit_rows=None,
                 res_rows=None, con_alias=None, moved_bounds=None,
                 var_row=None, con_row=None):
        self.nl = nl
        self.solver = solver
        self.var_names = list(var_names)
        self.con_names = list(con_names)
        # Reverse maps for the two orders above. Every query resolves a
        # name to its row, and a list scan makes that O(n) per lookup --
        # quadratic for a whole-model Jacobian, which asks for every
        # variable (gh#365). Built once here, or reused from the caller
        # when it has already built them. `is None`, not truthiness: an
        # unconstrained model's con_row is a legitimately empty dict,
        # which `or` would discard and rebuild.
        self._var_row = row_index(self.var_names) if var_row is None else var_row
        self._con_row = row_index(self.con_names) if con_row is None else con_row
        self.pins = {} if pins is None else pins
        self.pin_coefs = {} if pin_coefs is None else pin_coefs
        self.pin_bases = {} if pin_bases is None else pin_bases
        self.fit_rows = {} if fit_rows is None else fit_rows
        self.res_rows = {} if res_rows is None else res_rows
        self.con_alias = {} if con_alias is None else con_alias
        self.moved_bounds = {} if moved_bounds is None else moved_bounds
        self.base_x = None
        # Objective value at the solve. NaN, not None, is the "never
        # computed" sentinel: that is the convention the engine itself
        # uses for info["obj_val"] (pounce-py seeds final_obj with NaN
        # precisely because 0.0 is an ordinary objective value and cannot
        # signal it), so one isfinite check covers both an unset session
        # and a solve that evaluated nothing.
        self.base_obj = float("nan")
        # Cached `grad_x f` at the solved point (gh#878). Evaluated at
        # most once per session; a total derivative over an indexed
        # parameter would otherwise re-evaluate it per column.
        self._obj_grad = None
        self._columns = {}            # pin row -> full KKT-space column
        self._primal_rows = None      # full-x -> KKT row, lazily fetched
        self._row_data = None         # user row name -> data, on demand
        self._weakly_active_cache = None

    # ── keys and containers ──────────────────────────────────────────
    #
    # The analyses report which variable or row something happened to.
    # A subclass that has richer objects than names -- Pyomo component
    # data, say -- overrides these three, and every result container in
    # this package is keyed with them instead. The names are the
    # default because they are what a bare NL solve has.

    def var_key(self, full_idx):
        """What results are keyed by for full-x column `full_idx`."""
        return self.var_names[full_idx]

    def row_key(self, full_row):
        """What results are keyed by for full-g row `full_row`."""
        return self.user_row_data()[full_row]

    @staticmethod
    def new_keymap():
        """An empty mapping that accepts whatever `var_key` returns.

        A plain dict here; `pyomo_pounce` returns a `ComponentMap`,
        because Pyomo components are unhashable.
        """
        return {}

    def user_row_data(self):
        """What each solve row is called, in `.row` order.

        The base session has only names. A subclass with real row
        objects returns those, and None for a row with no counterpart in
        the caller's model -- a pin row, say -- which the analyses skip.
        """
        if self._row_data is None:
            self._row_data = list(user_row_names(self))
        return self._row_data

    # ── index spaces ─────────────────────────────────────────────────

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
                "has no row in the KKT factor and no sensitivity. Give it "
                "distinct bounds to keep it in the solve.")
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

    # ── derived quantities ───────────────────────────────────────────

    def objective_gradient(self):
        """`grad_x f` at the solved point, in full-x (`.col`) order."""
        if self._obj_grad is None:
            if self.base_x is None:
                raise ValueError(
                    "objective gradient: the solve recorded no primal "
                    "point, so it cannot be evaluated")
            self._obj_grad = np.asarray(self.nl.gradient(self.base_x),
                                        dtype=float)
        return self._obj_grad

    def total_objective_derivative(self, col):
        """`df/dp` for the KKT derivative column `col` (gh#878).

        The chain rule is

            df/dp  =  df/dp|_x  +  sum_i (df/dx_i)(dx_i/dp)

        and both terms fall out of a single contraction here, because a
        declared sensitivity parameter is an ordinary coordinate of the
        solve: it has been rewritten into a variable pinned by a
        defining equality, so `objective_gradient()` carries `df/dp|_x`
        in `p`'s own slot and the step carries `dp/dp = 1` there.
        Contracting the two therefore picks up the explicit partial and
        the implicit sum in one product, with no separate `grad_p f`
        term to assemble and no second index convention to get wrong.

        Written as a dot product over **full-x**, not the factor's
        var-x: `scatter_x` is what reconciles them, and a model carrying
        one fixed variable is where the two diverge. Contracting a
        full-x gradient with a var-x step would silently pair `df/dx_i`
        with a NEIGHBOURING variable's sensitivity from the first fixed
        variable on -- gh#450 and gh#672 finding 1, the same defect
        twice. `sens_invariance_legs` leg 3 is the guard.
        """
        return float(self.objective_gradient() @ self.scatter_x(col))

    def column(self, pin_idx):
        """Full KKT-space derivative column for a unit perturbation."""
        if pin_idx not in self._columns:
            self._columns[pin_idx] = np.asarray(
                self.solver.parametric_step_full([pin_idx], [1.0]))
        return self._columns[pin_idx]

    def var_entry(self, name):
        """The full-x column of variable `name`."""
        # ValueError, not the dict's KeyError: this used to be a list
        # scan and callers (and the message a user sees) expect it.
        try:
            return self._var_row[name]
        except KeyError:
            raise ValueError(
                f"{name}: not a variable of the solved model") from None

    def mult_entry(self, con_name):
        """The KKT row of constraint `con_name`'s multiplier.

        Equality rows answer unconditionally. An **inequality** row
        (pounce#910) answers only where its multiplier is a
        differentiable function of the parameters, which is where the
        row is *strictly* complementary — active with a multiplier bounded away from
        zero. The three regimes and why only one of them has an answer
        to give:

        * **strictly active** (`s ≈ 0`, `λ > 0`) — the row behaves
          exactly like an equality over a neighbourhood of the solved
          point, and `dλ/dp` is the same KKT back-solve. Answered.
        * **inactive** (`s > 0`, `λ ≈ 0`) — `λ` is pinned at zero over
          a neighbourhood, so the derivative exists and is *zero*.
          Refused rather than returned as a row, because there is no
          KKT row to return: an inactive row's multiplier is not a
          free coordinate of the system being differentiated.
        * **weakly active / kink** (`s ≈ 0` *and* `λ ≈ 0`) — the two
          one-sided derivatives differ, so no two-sided `dλ/dp`
          exists. Refused.

        The gate is :meth:`Solver.reduced_row_activity`, not
        `classify_activity`. Neither of them can *admit* a kink: a
        row's reduced curvature is never larger than its directional
        one, so the reduced ratio is never smaller than the
        directional ratio, and `"strongly_active"` is the high-ratio
        tail of both. What the directional normalizer gets wrong is
        the **refusal reason**. Swept over a genuine row kink whose
        coordinate is coupled to the rest of the free space at
        strength `rho`, `classify_activity` reads it
        `"weakly_active"` at `rho = 1`, `"ambiguous"` by `1e-2`, and
        `"inactive"` by `1e-6`, while `reduced_row_activity` holds
        `"weakly_active"` throughout (pounce#804). `"inactive"` is
        the one class whose derivative a caller may legitimately
        treat as a structural zero, so gating on the directional
        ratio would tell the holder of a *tight* cap that its shadow
        price does not move -- a wrong answer wearing a refusal's
        clothes. Reading an activity class as a proxy for kink-ness
        is the inference that shipped pounce#756.
        """
        # a caller whose solve ran on a rewritten copy translates the
        # original name to the copy's row
        con_name = self.con_alias.get(con_name, con_name)
        try:
            g = self._con_row[con_name]
        except KeyError:
            raise ValueError(
                f"{con_name}: not a constraint of the solved model") from None
        row = self.solver.multiplier_rows([g])[0]
        if row is not None:
            return row
        row = self.solver.d_multiplier_rows([g])[0]
        if row is None:
            # Unreachable: the c/d split puts every row in exactly one
            # block, a row with no finite bound on either side included
            # (it lands in `d` with an empty bound list, and the gate
            # below then refuses it as inactive, which is the truth --
            # a free row's multiplier is zero always). Kept because the
            # two accessors are separately maintained and a future
            # split that grows a third case should say so here rather
            # than index the step with a None.
            raise ValueError(
                f"{con_name}: neither the equality nor the inequality "
                "multiplier block claims constraint row "
                f"{g}, which should be impossible; please report this.")
        self._require_strict_complementarity(con_name, g)
        return row

    def _require_strict_complementarity(self, con_name, g):
        """Raise unless inequality row `g` is strictly active.

        Split out of :meth:`mult_entry` so the regime message is one
        thing and the row lookup another; see that method for why the
        gate reads the *reduced* classification.
        """
        try:
            rep = self.solver.reduced_row_activity([g])
        except Exception as exc:  # BadOptions, a failed back-solve
            raise ValueError(
                f"{con_name}: multiplier sensitivities on an inequality "
                "need its activity classified, and that failed: "
                f"{exc}") from None
        status = rep["status"][0]
        if status == "strongly_active":
            return
        if status == "inactive":
            raise ValueError(
                f"{con_name}: the constraint is inactive at the solved "
                "point, so its multiplier is zero over a neighbourhood "
                "and every parameter derivative of it is structurally "
                "zero. There is no KKT row to differentiate.")
        if status in ("weakly_active", "ambiguous"):
            ratio = float(rep["ratio"][0])
            what = ("a kink" if status == "weakly_active"
                    else "possibly a kink")
            raise ValueError(
                f"{con_name}: the constraint is {what} at the solved point "
                f"(reduced_row_activity says {status!r}, ratio {ratio:.3g}) "
                "— active with a multiplier at the barrier floor. The two "
                "one-sided derivatives of its multiplier differ, so no "
                "two-sided dlambda/dp exists. Solver.parametric_step_full "
                "still reports a one-sided step for a chosen "
                "perturbation direction if that is the question.")
        raise ValueError(
            f"{con_name}: multiplier sensitivities on an inequality need "
            "the row strictly active, and reduced_row_activity reports "
            f"{status!r}.")


def user_row_names(session):
    """`con_names` in the caller's own naming, pin rows included."""
    back = {clone: orig for orig, clone in session.con_alias.items()}
    return [back.get(nm, nm) for nm in session.con_names]


class NlBridge:
    """cyipopt-style callback object backed by `read_nl` evaluators.

    `pounce.Problem` wants an object with the cyipopt callback names;
    an `NlProblem` from :func:`pounce.read_nl` or
    :func:`pounce.build_nl_problem` spells two of them differently.
    This is the adapter, and it is the same one `pyomo_pounce` uses.
    """

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


def solve_for_sensitivity(nl, x0=None, options=None, var_names=None,
                          con_names=None, **session_kwargs):
    """Solve `nl` and return a :class:`SensSession` over the held factor.

    The one-call route for a caller who has a `.nl` file (or a problem
    built with :func:`pounce.build_nl_problem`) and wants the
    sensitivity surface over it::

        nl = pounce.read_nl("model.nl")
        sess = solve_for_sensitivity(nl, pins={"p": 4})
        dx = solution(sess, [4], [0.05])

    Returns the session; the solved point is `sess.base_x` and the
    solver is `sess.solver`. Extra keyword arguments are passed to
    :class:`SensSession`, which is how the pin, fitted and residual rows
    are declared.

    `bound_relax_factor` is set to 0 before the caller's options,
    because the activity classification every statistic here leans on
    reads slacks against the user's own bounds and the default
    relaxation shifts every one of them. Pass it explicitly to override.
    """
    import pounce

    prob = pounce.Problem(nl.n, nl.m, NlBridge(nl), lb=nl.x_l, ub=nl.x_u,
                          cl=nl.g_l, cu=nl.g_u)
    prob.add_option("bound_relax_factor", 0.0)
    for key, val in (options or {}).items():
        prob.add_option(key, val)
    solver = pounce.Solver(prob)
    x, info = solver.solve(np.asarray(nl.x0 if x0 is None else x0))

    names = list(var_names if var_names is not None else nl.var_names)
    rows = list(con_names if con_names is not None else nl.con_names)
    session = SensSession(nl, solver, names, rows, **session_kwargs)
    session.base_x = np.asarray(x, dtype=float)
    session.base_obj = float(info.get("obj_val", float("nan")))
    return session
