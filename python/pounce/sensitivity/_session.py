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


def objective_sign(nl):
    """+1 if the model was written as a minimization, -1 for a maximization.

    `pounce.read_nl` does not hand a maximization to the engine as
    written: it negates the objective callbacks and records what it did
    in `nl.minimize`. So every objective quantity the engine reports --
    `info["obj_val"]`, `nl.gradient()`, and the multipliers, which are
    stationarity coefficients of the objective it minimized -- is in the
    MINIMIZED sense, while everything the caller reads off the model
    (`pyo.value(obj)`, `m.dual[c]`) is in the sense they wrote.

    This factor is the conversion, and it is the one thing standing
    between the two. It is +1 for every minimization, which is why its
    absence went unnoticed: a maximization is the only model that can
    tell the difference, and a sign that is right on every model in the
    corpus is indistinguishable from no sign at all.

    `getattr` with a default, because a session may be built over a
    problem that is not a `read_nl` result; those are minimizations by
    construction, since there is no other way to state them.
    """
    return 1.0 if getattr(nl, "minimize", True) else -1.0


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
        # +1 / -1; see `objective_sign`. Read once here rather than per
        # query so that everything crossing back to the caller's units
        # goes through one value.
        self.obj_sign = objective_sign(nl)
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
        """`grad_x f` at the solved point, in full-x (`.col`) order.

        `f` is the objective **as the model states it**, so this is
        `-grad_x` of what the engine minimized on a maximization; see
        `objective_sign`. Everything derived from it -- the total
        derivative below, and so `sens_jacobian(of=<Objective>)` --
        inherits that, which is what makes them agree with a finite
        difference of `pyo.value(obj)` across a re-solve.
        """
        if self._obj_grad is None:
            if self.base_x is None:
                raise ValueError(
                    "objective gradient: the solve recorded no primal "
                    "point, so it cannot be evaluated")
            self._obj_grad = self.obj_sign * np.asarray(
                self.nl.gradient(self.base_x), dtype=float)
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

        An equality is answered from the `y_c` block unconditionally:
        its multiplier is a differentiable function of the parameters
        wherever the solve is regular, and the row is the whole answer.

        An inequality is answered from `y_d`, but only where the row is
        **strictly active** (gh#910). That is the regime in which the
        row behaves as an equality over a neighbourhood of the solved
        point, so `dlambda/dp` is the same back-solve an equality gets.
        The other two regimes hold a number in `y_d` and no derivative:
        an inactive row's answer is a structural zero the KKT row does
        not carry, and a kink -- slack AND multiplier both vanishing --
        has two one-sided derivatives that differ, with the entry
        holding whichever side the factorization landed on. Both are
        refused, and the message names the regime, because they call
        for different things from the caller.

        The gate reads `reduced_row_activity`, not the `row_status` of
        `classify_activity`. The cheap classifier normalizes a row by
        the curvature along its own gradient rather than by the reduced
        curvature that generates its multiplier, so its ratio for a
        genuine kink is `reduced/directional` and slides with the
        coupling: at coupling `1e-6` a row that is a kink at every
        coupling reports "inactive" there (gh#804). "inactive" is the
        one verdict this method may pass through as "the shadow price
        does not move", so gating on it would answer a tight cap's
        kinked shadow price with a confident zero -- a wrong answer
        wearing a refusal's clothes. Reading an activity class as a
        proxy for kink-ness is the inference that shipped gh#756.
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
        return self._inequality_mult_entry(con_name, g)

    #: What the strictly-active gate says about each regime it turns
    #: away. Keyed by `reduced_row_activity` status; the value is the
    #: sentence after the refusal, and it names the regime rather than
    #: restating the rule, since the two regimes want different things.
    _INEQUALITY_REFUSAL = {
        "inactive": (
            "the row is inactive at the solved point (slack > 0, "
            "multiplier ~ 0), so its multiplier is zero over a "
            "neighbourhood of the parameters and d(dual)/dp is a "
            "structural zero -- there is no KKT row to differentiate"),
        "weakly_active": (
            "the row is weakly active at the solved point -- a kink, "
            "with slack AND multiplier both vanishing -- so the "
            "multiplier has two one-sided derivatives that differ and "
            "no two-sided value exists. Ask for a direction instead: "
            "`solution()` / `parametric_step_directional` report the "
            "one-sided step for a signed perturbation"),
        "ambiguous": (
            "the row's activity regime could not be resolved: its "
            "barrier-to-curvature ratio sits between the strictly "
            "active and the inactive thresholds, so whether its "
            "multiplier is differentiable here is undecided. A tighter "
            "solve (smaller `tol`) is what moves this ratio"),
        "unidentified": (
            "the row's reduced curvature is below the identification "
            "floor, so its activity regime cannot be measured"),
        "unbounded": (
            "the row has no finite bound, so it has no multiplier to "
            "differentiate"),
    }

    def _inequality_mult_entry(self, con_name, g):
        """`mult_entry`'s `y_d` half: the strictly-active gate."""
        row = self.solver.inequality_multiplier_rows([g])[0]
        if row is None:
            # neither block claims it, which the c/d split makes
            # impossible for a row of the solved model -- so this is a
            # map disagreement, not a user error
            raise ValueError(
                f"{con_name}: constraint row {g} is in neither the "
                "equality nor the inequality multiplier block")
        try:
            status = self.solver.reduced_row_activity([g])["status"][0]
        except Exception as exc:
            # the classifier's own refusals name the option they are
            # about -- a relaxed solve is the one a caller actually
            # meets -- and not the question that reached it, so the
            # constraint gets attached here
            raise ValueError(
                f"{con_name}: an inequality's multiplier sensitivity is "
                "gated on the row's activity regime, which could not be "
                f"classified: {exc}") from exc
        if status == "strongly_active":
            return row
        why = self._INEQUALITY_REFUSAL.get(
            status, f"the row classified as {status!r}")
        raise ValueError(
            f"{con_name}: multiplier sensitivities for an inequality "
            f"are available only where the row is strictly active, and "
            f"{why}.")


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
    # in the model's own sense, so that a maximization's `base_obj` is
    # the value it states rather than the negation the engine minimized
    session.base_obj = session.obj_sign * float(
        info.get("obj_val", float("nan")))
    return session
