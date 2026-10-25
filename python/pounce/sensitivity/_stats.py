"""Parameter covariance, the information matrix, and identifiability.

What one converged solve says about how well the data pinned the fitted
parameters down. :func:`covariance` is the asymptotic covariance of a
least-squares estimate; :func:`information` is the reduced Hessian it
inverts, which is the object that stays finite when the block is
rank-deficient and the covariance does not.

Both take a :class:`~pounce.sensitivity.SensSession` and a *block*: the
parameters the answer is about, as an ordered list of keys with their
full-x rows. Omit it and the session's declared `fit_rows` block is
used. Keys are opaque -- they order the result and label it -- so a
caller with richer objects than names keys the result by those.

The calculations are `pounce_sens_core::stats`, written against the
engine traits so they run over any interior-point engine. This module
resolves the block and the noise variance from the caller's keys, hands
the rows to that code, and turns what it reports into the results and
the warnings below.
"""
import warnings

import numpy as np

from .. import _pounce
from ._step import check_margins, pdpert_error

#: How the caller spells the declarations these diagnostics point at.
#: The core is told rather than assuming, because a bare-NL caller and
#: `pyomo_pounce` declare fitted parameters and residual rows very
#: differently, and a message naming the wrong declaration is worse
#: than one naming none.
DEFAULT_HINTS = {
    "fitted": "fit_rows=",
    "residual": "res_rows=",
    "residual_group": "res_rows={group: rows}",
}


def _label(key):
    """What to call a block member in a diagnostic.

    A caller keying the block by rich objects gets their `.name`; one
    keying it by the variable's own name gets the name back.
    """
    return getattr(key, "name", key)


def _resolve_block(session, params, rows, who, hints):
    """The block to answer about: the caller's, or the declared one.

    `params` and `rows` come in already resolved -- normalizing a
    modelling layer's way of naming a block is that layer's job, not
    this one's.
    """
    if params is not None:
        rows = list(rows)
        params = list(params)
        if len(params) != len(rows):
            raise ValueError(
                f"{who}: the block has {len(params)} keys but "
                f"{len(rows)} rows")
        if not params:
            raise ValueError(f"{who}: the block is empty")
        return params, rows
    params = list(session.fit_rows.keys())
    if not params:
        raise RuntimeError(
            f"{who}: no fitted parameters were declared; give the "
            f"session a {hints['fitted']} block before the solve, or "
            "pass one explicitly")
    return params, [session.fit_rows[p] for p in params]


class _ParamKeyed:
    """Lookup from a block member's key to its position.

    Keyed by id() rather than by value, so a caller whose keys are
    unhashable objects -- Pyomo component data, say -- can use them
    directly. The key list is held alongside, because an id is unique
    only among live objects."""

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
    """Fix the arbitrary sign an eigensolver gives each eigenvector
    column, so the direction an eigen() call reports is reproducible.

    `v` and `-v` are equally valid eigenvectors, and which one a solver
    returns is a convention of the build, not a property of the matrix:
    the same model and data can report opposite directions on two
    machines. The convention here is the largest-magnitude component
    positive, ties broken by the earliest `params` position. It is
    scale-invariant and independent of parameter ordering except
    through that tie-break.

    This pins the sign only. A repeated eigenvalue leaves the basis
    WITHIN its eigenspace arbitrary — any rotation of those columns
    diagonalizes just as well — so the individual eigenvectors of a
    degenerate block are still not reproducible; only the subspace
    they span is."""
    vecs = np.asarray(vecs, dtype=float)
    if vecs.size == 0:
        return vecs
    return np.asarray(_pounce._pin_eigenvector_signs(vecs))


def _eigen(matrix):
    w, v = _pounce._sym_eigen(np.asarray(matrix, dtype=float))
    return np.asarray(w, dtype=float), np.asarray(v, dtype=float)


class Covariance(_ParamMatrix):
    """Asymptotic parameter covariance, from covariance().

    Keyed by the block members' own keys -- the fitted variables, not
    the model's parameters -- in `params` order: cov[m.k1, m.k2] (either order),
    cov[m.k1] for a variance, cov.std_err[m.k1],
    cov.correlation[m.k1, m.k2]. `matrix` is the dense numpy array
    ordered like `params`; `sigma_sq` is the residual variance that was
    used. eigen() supports identifiability diagnosis."""

    def __init__(self, params, matrix, sigma_sq, conditioned_on=()):
        super().__init__(params, matrix)
        self.params = self._params
        self.sigma_sq = sigma_sq          # float, or {group: float}
        self._conditioned = conditioned_on
        # entries whose scale is undefined (a projected bound-active
        # parameter has exactly zero variance) are reported as 0
        se, corr = _pounce._correlation(self.matrix)
        self.std_err = _ParamVector(self.params, se)
        self.correlation = _ParamMatrix(self.params, corr)

    @property
    def conditioned_on(self):
        """Strongly active variables OUTSIDE the block: the matrix is
        conditional on those bounds (roadmap item 3); empty when none.
        The call that returns the matrix finds them, at one backsolve
        per near-bound candidate."""
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
        return _eigen(self.matrix)


def _indefinite(block):
    """A symmetric block with a genuinely negative eigenvalue, at a
    tolerance relative to the block's own scale. At a strict local
    minimum with LICQ the reduced Hessian is PSD, so this fires only
    on non-minimum stationary points or regularized (inertia-
    corrected) convergence: exactly the finding worth returning
    rather than refusing."""
    block = np.asarray(block, dtype=float)
    if block.size == 0:
        return False
    return bool(_pounce._indefinite(block))


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
        The call that returns the matrix finds them, at one backsolve
        per near-bound candidate."""
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
        return _eigen(self.matrix)


def _classify_ratio(r, mu):
    """The activity rule of covariance roadmap item 0, applied at the
    reduced fitted block. Mirrors the classifier with one deliberate
    divergence: the classifier maps q below the floor to
    `unidentified` unconditionally, while the caller here first checks
    the ratio, because at the reduced level a huge Sigma cancelling
    inside q_red would otherwise misfile a strongly active entry as
    unidentified. `pounce_sens_core::stats::classify_ratio`."""
    return _pounce._classify_ratio(float(r), float(mu))


def _rank_deficient(A):
    """True when the symmetric block `A` is rank-deficient, tested on
    its diagonally scaled form so the verdict is a statement about
    COLLINEARITY and not about units.

    The rank tolerance, `sigma_max * n * eps`, is relative to the
    largest singular value and so is not invariant under rescaling the
    coordinates against each other: a covariance block carries the
    SQUARE of any unit spread between its members, so two perfectly
    well-determined parameters ~1e8 apart in magnitude (a rate
    prefactor against a reaction order) push `cond(M)` past the
    tolerance and read as dependent on unit spread alone. Scaling by
    `sqrt(|diag|)` first — the correlation form — makes the test the
    same kind of scale-invariant ratio as the block-membership rule
    (roadmap item 1) and the conditioned_on rule, rather than a
    threshold that tracks the user's choice of units. Second review of
    gh #466.

    A zero diagonal leaves its own row and column unscaled: there is no
    scale to divide by, and a genuinely zero row is rank-deficient
    either way."""
    A = np.asarray(A, dtype=float)
    if A.shape[0] == 0:
        return False
    return bool(_pounce._rank_deficient(A))


class _SingularBlock(RuntimeError):
    """The requested block of the inverse KKT matrix is singular.
    A dedicated type so callers that rescue this case (the of=
    dependent-block paths) do not do control flow on message text
    (review of gh #466)."""


_WORDING = {
    # the default block IS the fitted set, and saying so is strictly
    # more informative there; an arbitrary of= block gets the
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


def _declarations(session):
    """The declared rows, in the order the core reads them."""
    pins = [int(r) for r in session.pins.values()]
    fit = [int(session.fit_rows[k]) for k in session.fit_rows]
    groups = dict(session.res_rows)
    res = [[int(r) for r in rws] for rws in groups.values()]
    return pins, fit, groups, res


def _removed_message(session, full_idx, at, who, params):
    kind = at[0]
    if kind == "member":
        what = f"{who}({_label(params[at[1]])})"
    elif kind == "fitted":
        what = f"{who}(fitted)"
    elif kind == "conditioned_on":
        what = f"{who} conditioned_on"
    else:
        what = who
    return (
        f"{what}: {session.var_names[full_idx]} was removed from the "
        "solve as a fixed variable (its bounds are equal), so it "
        "has no row in the KKT factor and no sensitivity. Give it "
        "distinct bounds to keep it in the solve.")


def _row_name(session, j):
    return (session.con_names[j] if j < len(session.con_names)
            else f"row {j}")


def _combo(combo, params):
    return " + ".join(f"{c:.3g}*{_label(params[k])}" for k, c in combo)


def _reason(code):
    return ("more coordinates than the fit has degrees of freedom"
            if code == "count" else "linearly dependent coordinates")


def _warn_findings(session, findings, who, params, wording, accessor):
    """Each finding of the core, as the warning this package has always
    raised for it, in the order the calculation met them."""
    W = _WORDING[wording]
    # covariance() never passed its name to the classification, whose
    # warnings have always begun "covariance:" whoever called it
    cw = "covariance" if accessor == "covariance" else who
    for f in findings:
        kind = f[0]
        if kind == "perturbed":
            if accessor == "covariance":
                tail = ("so the covariance is regularized rather than "
                        "exact. Linearly dependent (structurally"
                        " unidentifiable) parameters are the usual cause.")
            else:
                tail = ("so the information is regularized rather than "
                        "exact; the isotropic delta_w lands on the free "
                        "block and survives the projection.")
            warnings.warn(
                f"{who}: the held KKT factor carries inertia-correction "
                f"perturbations {list(f[1])}, " + tail)
        elif kind == "member_held":
            warnings.warn(
                f"{cw}: {W['member']} {_label(params[f[1]])} is held by "
                "its bound at the optimum (strongly active); its direction "
                "is projected out (zero variance, conditional on the active "
                "bound) and the boundary asymptotics are nonstandard.")
        elif kind == "member_weak":
            warnings.warn(
                f"{cw}: {W['member']} {_label(params[f[1]])} sits exactly "
                "on its bound with a vanishing multiplier (weakly active). "
                "It is kept in the free block with finite variance; "
                "boundary asymptotics are nonstandard.")
        elif kind == "member_ambiguous":
            warnings.warn(
                f"{cw}: {W['member']} {_label(params[f[1]])} has "
                "ambiguous bound activity at the solve's final barrier "
                "parameter; re-solve with a tighter tol to settle it. It is "
                "kept in the free block.")
        elif kind == "member_unidentified":
            warnings.warn(
                f"{cw}: {W['member']} {_label(params[f[1]])} has curvature "
                "below the model's own noise scale (unidentified); its "
                "variance is large rather than small. It is kept in the "
                "free block.")
        elif kind == "row_mixed_held":
            warnings.warn(
                f"{cw}: constraint {_row_name(session, f[1])} is strongly "
                f"active and involves {W['outside']}; the "
                f"direction it pins reaches {W['reach']} through the "
                "eliminated variables and cannot be represented by "
                "a restricted normal, so it is NOT projected. Treat "
                "the returned variances as not conditioned on this "
                "constraint.")
        elif kind == "row_mixed_kept":
            warnings.warn(
                f"{cw}: constraint {_row_name(session, f[1])} is {f[2]} "
                f"and involves {W['outside']}; it is kept "
                "unprojected and its barrier weight is not "
                "corrected for (the restricted direction would be "
                "the wrong one). Boundary asymptotics are "
                "nonstandard.")
        elif kind == "row_pins":
            warnings.warn(
                f"{cw}: constraint {_row_name(session, f[1])} is strongly "
                f"active and pins the {W['combo']} {_combo(f[2], params)}; "
                "variance along it is projected to zero (conditional on the "
                "constraint). Conditional information along the "
                f"combination: {f[3]:.6g}.")
        elif kind == "row_weak":
            warnings.warn(
                f"{cw}: constraint {_row_name(session, f[1])} is weakly "
                f"active on the {W['combo']} {_combo(f[2], params)} "
                "(multiplier and slack vanish together). It is kept "
                "unprojected with finite variance; boundary asymptotics "
                "are nonstandard.")
        elif kind == "row_ambiguous":
            warnings.warn(
                f"{cw}: constraint {_row_name(session, f[1])} has "
                f"ambiguous activity on the {W['combo']} "
                f"{_combo(f[2], params)} at the solve's final barrier "
                "parameter; re-solve with a tighter tol to settle it. It "
                "is kept unprojected.")
        elif kind == "row_unidentified":
            warnings.warn(
                f"{cw}: constraint {_row_name(session, f[1])} has "
                f"curvature below {W['scale']} noise scale on "
                f"{_combo(f[2], params)} (unidentified); it is kept "
                "unprojected and its variance is large rather than small.")
        elif kind == "n_data_ignored":
            warnings.warn(
                f"{who}: n_data is ignored because a "
                "higher-precedence noise "
                "source was given (sigma_sq, or the declared residuals).")
        elif kind == "deficient_members":
            flagged = [_label(params[i]) for i in f[1]]
            warnings.warn(
                f"{who}: block members {flagged} have bound "
                f"activity, but the block is rank-deficient "
                f"({_reason(f[2])}), so the "
                "membership handling (pinned parameters, binding rows) "
                "is unavailable; the marginal is returned without it.")
        elif kind == "negative_variance":
            warnings.warn(
                f"{who}: negative variance on the diagonal; the point is "
                "probably not a least-squares minimum.")
        elif kind == "indefinite":
            warnings.warn(
                f"{who}: the Lagrangian free block is indefinite "
                "(returned as computed): the point is not a "
                "least-squares minimum along some direction, or the "
                "model is over-parameterized. "
                "hessian='gauss-newton' is the PSD alternative.")
        elif kind == "subblock_fallback":
            warnings.warn(
                f"{who}: the free fitted coordinates outside the block "
                "are singular, so the exact Schur route is unavailable; "
                "falling back to the corrected reduction off the "
                "factor, which loses digits at tight mu on pinned "
                "members.")
        else:
            raise AssertionError(f"unknown finding {f!r}")


def _raise_error(session, err, who, params, hints, groups, n_data,
                 max_pdpert, singular_who=None):
    """The exception this package has always raised for each failure
    the core reports. `singular_who` names a singular block, which
    covariance() has always reported as "covariance" whoever called
    it."""
    kind = err[0]
    if kind == "raise":
        raise err[1]
    if kind == "pdpert":
        raise pdpert_error(who, err[1], err[2], max_pdpert)
    if kind == "not_positive":
        raise ValueError(
            f"{who}: {err[1]} must be a positive number, got {err[2]!r}")
    if kind == "singular_block":
        raise _SingularBlock(
            f"{singular_who or who}: the requested block of the inverse KKT matrix "
            "is singular; the block members are linearly "
            "dependent (structurally unidentifiable)")
    if kind == "removed":
        raise ValueError(_removed_message(session, err[1], err[2], who,
                                          params))
    if kind == "no_block":
        raise RuntimeError(
            f"{who}: no fitted parameters were declared; give the "
            f"session a {hints['fitted']} block before the solve, or "
            "pass one explicitly")
    if kind == "gn_needs_residuals":
        raise ValueError(
            f"{who}: hessian='gauss-newton' needs declared residuals "
            f"({hints['residual']}); the residual Jacobian is "
            "recovered from "
            "their rows. Without residual variables only the "
            "hessian='lagrangian' default is available.")
    if kind == "no_fit_for_residuals":
        raise ValueError(
            f"{who}: the noise variance must be estimated from "
            "the declared residuals, but no fitted parameters were "
            "declared, so the degrees of freedom for the estimate "
            "are unknown; pass sigma_sq= (known variance) or flag "
            f"the fitted parameters with {hints['fitted']}")
    if kind == "no_solve_point":
        raise RuntimeError(
            f"{who}: the noise variance must be estimated from the "
            "declared residuals, but the solve recorded no primal point "
            "to read them at")
    if kind == "group_too_small":
        g = list(groups)[err[1]]
        raise ValueError(
            f"{who}: residual group {g!r} has "
            f"{err[2]} members, "
            f"not more than the {err[3]} fitted parameters; "
            "cannot estimate its noise variance")
    if kind == "n_data_no_fit":
        raise ValueError(
            f"{who}: n_data= estimates the noise variance, but "
            "no fitted parameters were declared, so the degrees of "
            "freedom for the estimate are unknown; pass sigma_sq= "
            "(known variance) or flag the fitted parameters with "
            f"{hints['fitted']}")
    if kind == "n_data_too_small":
        raise ValueError(
            f"{who}: n_data ({n_data}) must exceed the "
            "number of "
            f"fitted parameters ({err[2]})")
    if kind == "no_objective_value":
        raise RuntimeError(
            f"{who}: the solve reported no usable "
            "objective value "
            f"({session.base_obj}), so n_data= cannot estimate the "
            "noise variance. Pass sigma_sq= (known variance), or "
            f"declare the residual container with {hints['residual']}.")
    if kind == "noise_unknown":
        raise ValueError(
            f"{who}: the noise variance is unknown; declare the "
            f"residual container(s) with {hints['residual']}, or pass "
            "sigma_sq= (known variance), or pass n_data= (data count, "
            "with the SSR taken from the solve-time objective value)")
    if kind == "deficient_needs_lagrangian":
        raise RuntimeError(
            f"{who}: the of= block is rank-deficient "
            f"({_reason(err[1])}), so the profiled Jacobians behind "
            "hessian='gauss-newton' and per-group noise are not "
            "defined; only the homoscedastic hessian='lagrangian' "
            "marginal is available for this block")
    if kind == "gn_singular":
        raise RuntimeError(
            f"{who}: the Gauss-Newton matrix J^T J is singular; "
            "the block members are linearly dependent in the "
            "residual Jacobian")
    if kind == "free_singular":
        raise RuntimeError(
            f"{who}: the reduced Hessian restricted to the "
            "free (off-bound, off-constraint) members is "
            "singular; the remaining free block members are "
            "linearly dependent")
    if kind == "information_deficient":
        raise RuntimeError(
            f"{who}: the of= block is rank-deficient ({_reason(err[1])}), "
            "so it carries no information matrix; its covariance "
            "(covariance(model, of=...)) is the meaningful object "
            "for such a block")
    if kind == "not_square":
        n_var, n_eq, n_params = err[1], err[2], err[3]
        raise RuntimeError(
            f"{who}: hessian='lagrangian' requires the "
            "square estimation structure (the equality "
            "constraints determine the non-fitted variables "
            "given the fitted block); this model has "
            f"{n_var} factor variables, {n_eq} equalities and "
            f"{n_params} fitted parameters, so "
            f"{n_var} - {n_eq} != {n_params} and the tangent "
            "recovery is not defined. hessian='gauss-newton' "
            "does not need it.")
    if kind == "pinned_undefined":
        raise RuntimeError(
            f"{who}: the free block is singular, so the "
            "pinned parameters' conditional information S is not "
            "defined; the free block members are linearly "
            "dependent")
    raise AssertionError(f"unknown error {err!r}")


def _conditioned_on(session, rows):
    """Strongly active variables outside the block, keyed like the
    block. Their Sigma stays in the held factor and drives the coupling
    through them to zero as mu falls, so the block's numbers are the
    values conditional on those bounds, not the marginal over them.
    Returned with the matrix rather than warned: it is a property of
    the answer, not a defect. `pounce_sens_core::stats::conditioned_on`
    decides which they are, and covariance and information return them
    as `conditioned_on`."""
    out = []
    for idx in rows:
        v = session.var_key(idx)
        out.append(v if v is not None else session.var_names[idx])
    return tuple(out)


def covariance(session, params=None, rows=None, sigma_sq=None,
               n_data=None, hessian="lagrangian", max_pdpert=None,
               who="covariance", hints=None):
    """Asymptotic covariance of the fitted parameters of a
    least-squares problem, from ONE ordinary solve.

    Workflow: give the session the fitted parameters' full-x columns as
    `fit_rows` (they stay free variables) and, optionally, the residual
    variables' columns as `res_rows`, solve once, then call this with no
    further information. Pass `params`/`rows` instead to ask about any
    other block of the solve's variables.

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
    first, the receding-horizon pattern of solution(), does not change
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

    of= selects the block (covariance roadmap item 3): any of the
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
    bands, active-bound projection); use this when the fit is a block
    inside a model you have already written and solved.

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
            f"{who}: hessian must be 'lagrangian' or 'gauss-newton', "
            f"got {hessian!r}")
    hints = DEFAULT_HINTS if hints is None else hints
    # None here means "the session's own declared block", which is what
    # the wording and the rank-deficiency advice below branch on
    of = params
    params, rows = _resolve_block(session, params, rows, who, hints)
    wording = "fitted" if of is None else "block"
    check_margins(None, max_pdpert, who)
    pins, fit, groups, res = _declarations(session)

    # the known noise variance, one value per declared group, or one
    # value when none were declared; the per-group form is checked
    # against the declared groups here, where the keys are
    known = None
    if sigma_sq is not None:
        if isinstance(sigma_sq, dict):
            named = [g for g in groups if g is not None]
            if not named:
                raise ValueError(
                    f"{who}: sigma_sq was given as a "
                    "per-group dict but "
                    "no named residual groups were declared; pass a scalar "
                    "sigma_sq, or declare grouped residuals with "
                    f"{hints['residual_group']}")
            missing = [g for g in groups if g not in sigma_sq]
            if missing:
                raise ValueError(
                    f"{who}: sigma_sq is missing an entry "
                    "for residual "
                    f"group(s) {sorted(map(repr, missing))}")
            known = [float(sigma_sq[g]) for g in groups]
        else:
            known = [float(sigma_sq)] * max(len(groups), 1)

    res_out = _pounce._sens_covariance(
        session.solver, pins, fit, res, session.base_x,
        float(session.base_obj), float(session.obj_sign),
        None if of is None else [int(r) for r in rows], known,
        None if n_data is None else float(n_data), hessian,
        None if max_pdpert is None else float(max_pdpert))
    _warn_findings(session, res_out["findings"], who, params, wording,
                   "covariance")
    if res_out["error"] is not None:
        _raise_error(session, res_out["error"], who, params, hints, groups,
                     n_data, max_pdpert, singular_who="covariance")
    keys = list(groups) if groups else [None]
    group_sigma = dict(zip(keys, res_out["sigma_sq"]))
    sig_out = (next(iter(group_sigma.values()))
               if len(group_sigma) == 1 and None in group_sigma
               else group_sigma)
    return Covariance(
        params, res_out["matrix"], sig_out,
        _conditioned_on(session, res_out["conditioned_on"]))


def information(session, params=None, rows=None, hessian="lagrangian",
                max_pdpert=None, who="information", hints=None):
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

    of= selects the block exactly as in covariance() (roadmap item
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
            f"{who}: hessian must be 'lagrangian' or "
            "'gauss-newton', "
            f"got {hessian!r}")
    hints = DEFAULT_HINTS if hints is None else hints
    # None here means "the session's own declared block", which is what
    # the wording and the rank-deficiency advice below branch on
    of = params
    params, rows = _resolve_block(session, params, rows, who, hints)
    wording = "fitted" if of is None else "block"
    check_margins(None, max_pdpert, who)
    pins, fit, groups, res = _declarations(session)
    res_out = _pounce._sens_information(
        session.solver, pins, fit, res,
        None if of is None else [int(r) for r in rows], hessian,
        None if max_pdpert is None else float(max_pdpert))
    _warn_findings(session, res_out["findings"], who, params, wording,
                   "information")
    if res_out["error"] is not None:
        _raise_error(session, res_out["error"], who, params, hints, groups,
                     None, max_pdpert)
    return Information(
        params, res_out["matrix"],
        _conditioned_on(session, res_out["conditioned_on"]))
