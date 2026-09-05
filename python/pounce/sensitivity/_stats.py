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
"""
import warnings

import numpy as np

from .._stats_util import nullspace as _nullspace
from ._step import check_margins, refuse_on_pdpert

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
                f"{who}: {W['member']} {_label(p)} is held by its "
                "bound at the optimum (strongly active); its direction is "
                "projected out (zero variance, conditional on the active "
                "bound) and the boundary asymptotics are nonstandard.")
        elif status == "weakly_active":
            warnings.warn(
                f"{who}: {W['member']} {_label(p)} sits exactly on "
                "its bound with a vanishing multiplier (weakly active). "
                "It is kept in the free block with finite variance; "
                "boundary asymptotics are nonstandard.")
        elif status == "ambiguous":
            warnings.warn(
                f"{who}: {W['member']} {_label(p)} has ambiguous "
                "bound activity at the solve's final barrier parameter; "
                "re-solve with a tighter tol to settle it. It is kept in "
                "the free block.")
        elif status == "unidentified":
            warnings.warn(
                f"{who}: {W['member']} {_label(p)} has curvature "
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
            f"{a[k]:.3g}*{_label(params[k])}" for k in range(n_params)
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
            v = session.var_key(idx)
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
    A dedicated type so callers that rescue this case (the of=
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
    # the block: the declared fitted parameters by default, or any
    # block of the solve's variables via of= (each call re-reduces
    # onto its own argument, so one solve serves as many blocks as are
    # asked about, each getting that block's marginal)
    hints = DEFAULT_HINTS if hints is None else hints
    # None here means "the session's own declared block", which is what
    # the wording and the rank-deficiency advice below branch on
    of = params
    params, rows = _resolve_block(session, params, rows, who, hints)
    n_params = len(params)
    wording = "fitted" if of is None else "block"
    # sigma estimation divides by the FIT's degrees of freedom, a
    # property of the solve, not of the block being asked about
    n_fit = len(session.fit_rows)

    # ── guardrails ────────────────────────────────────────────────────────
    check_margins(None, max_pdpert, who)
    refuse_on_pdpert(session, max_pdpert, who)
    pert = np.asarray(session.solver.kkt_perturbations)
    if pert.any():
        warnings.warn(
            f"{who}: the held KKT factor carries inertia-correction "
            f"perturbations {pert.tolist()}, so the covariance is "
            "regularized rather than exact. Linearly dependent (structurally"
            " unidentifiable) parameters are the usual cause.")
    # ── parameter block of the inverse KKT matrix ─────────────────────────
    # `rows` is full-x (the space of the activity report and of
    # row_normal, both read below); `krows` is the same variables as
    # factor rows. Keep the two apart -- they differ exactly when the
    # model has a fixed variable, and agree everywhere else.
    dim = session.solver.kkt_dim
    krows = [session.primal_row(r, f"{who}({_label(p)})")
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
    if of is not None:
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
            if of is None:
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
            f"{who}: hessian='gauss-newton' needs declared residuals "
            f"({hints['residual']}); the residual Jacobian is "
            "recovered from "
            "their rows. Without residual variables only the "
            "hessian='lagrangian' default is available.")
    if n_data is not None and (sigma_sq is not None or groups):
        warnings.warn(
            f"{who}: n_data is ignored because a "
            "higher-precedence noise "
            "source was given (sigma_sq, or the declared residuals).")
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
            group_sigma = {g: float(sigma_sq[g]) for g in groups}
        else:
            group_sigma = {g: float(sigma_sq) for g in (groups or {None: []})}
    elif groups:
        if n_fit == 0:
            raise ValueError(
                f"{who}: the noise variance must be estimated from "
                "the declared residuals, but no fitted parameters were "
                "declared, so the degrees of freedom for the estimate "
                "are unknown; pass sigma_sq= (known variance) or flag "
                f"the fitted parameters with {hints['fitted']}")
        group_sigma = {}
        for g, rws in groups.items():
            n_g = len(rws)
            if n_g <= n_fit:
                raise ValueError(
                    f"{who}: residual group {g!r} has "
                    f"{n_g} members, "
                    f"not more than the {n_fit} fitted parameters; "
                    "cannot estimate its noise variance")
            ssr_g = float(np.sum(session.base_x[rws] ** 2))
            group_sigma[g] = ssr_g / (n_g - n_fit)
    elif n_data is not None:
        if n_fit == 0:
            raise ValueError(
                f"{who}: n_data= estimates the noise variance, but "
                "no fitted parameters were declared, so the degrees of "
                "freedom for the estimate are unknown; pass sigma_sq= "
                "(known variance) or flag the fitted parameters with "
                f"{hints['fitted']}")
        if n_data <= n_fit:
            raise ValueError(
                f"{who}: n_data ({n_data}) must exceed the "
                "number of "
                f"fitted parameters ({n_fit})")
        # the objective value AT THE SOLVE, not re-evaluated: a
        # modelling layer that evaluates the objective reads its
        # CURRENT variable and parameter values, so anything written
        # after the solve (a measurement, a warm start for the next
        # horizon) would silently rescale the covariance (gh #426)
        if not np.isfinite(session.base_obj):
            raise RuntimeError(
                f"{who}: the solve reported no usable "
                "objective value "
                f"({session.base_obj}), so n_data= cannot estimate the "
                "noise variance. Pass sigma_sq= (known variance), or "
                f"declare the residual container with {hints['residual']}.")
        # in the MINIMIZED sense: `n_data=` reads the objective as a sum
        # of squares, and `maximize -SSR` is the same least-squares
        # problem spelled the other way round.  `base_obj` is in the
        # model's own sense (see `objective_sign`), so undo that here or
        # a maximize spelling divides a negative by `n_data - n_fit` and
        # every standard error comes back NaN.
        ssr = session.obj_sign * session.base_obj
        group_sigma = {None: ssr / (n_data - n_fit)}
    else:
        raise ValueError(
            f"{who}: the noise variance is unknown; declare the "
            f"residual container(s) with {hints['residual']}, or pass "
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
            Zr = np.array([[zcols[j][session.primal_row(r, who)]
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
                f"{who}: the of= block is rank-deficient "
                f"({reason}), so the profiled Jacobians behind "
                "hessian='gauss-newton' and per-group noise are not "
                "defined; only the homoscedastic hessian='lagrangian' "
                "marginal is available for this block")
        act = session.solver.classify_activity()
        flagged = [_label(p) for i, p in enumerate(params)
                   if act["var_status"][rows[i]] not in
                   ("inactive", "unbounded", "fixed", "equality")]
        if flagged:
            warnings.warn(
                f"{who}: block members {flagged} have bound "
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
            lambda: _conditioned_on(session, act, rows, who))

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
                f"{who}: the Gauss-Newton matrix J^T J is singular; "
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
                    f"{who}: the reduced Hessian restricted to the "
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
            f"{who}: negative variance on the diagonal; the point is "
            "probably not a least-squares minimum.")
    sig_out = (next(iter(group_sigma.values()))
               if len(group_sigma) == 1 and None in group_sigma
               else group_sigma)
    return Covariance(
        params, cov, sig_out,
        lambda: _conditioned_on(session, cb.act, rows, who))


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
    n_params = len(params)
    wording = "fitted" if of is None else "block"

    check_margins(None, max_pdpert, who)
    refuse_on_pdpert(session, max_pdpert, who)
    pert = np.asarray(session.solver.kkt_perturbations)
    if pert.any():
        warnings.warn(
            f"{who}: the held KKT factor carries inertia-correction "
            f"perturbations {pert.tolist()}, so the information is "
            "regularized rather than exact; the isotropic delta_w lands on "
            "the free block and survives the projection.")

    # `rows` is full-x (the space _classify_fitted_block reads the
    # activity report and row_normal in); `krows` is the same variables
    # as factor rows (gh #450): they differ exactly when the model has
    # a fixed variable, and agree everywhere else
    dim = session.solver.kkt_dim
    krows = [session.primal_row(r, f"{who}({_label(p)})")
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
            f"{who}: the of= block is rank-deficient ({reason}), "
            "so it carries no information matrix; its covariance "
            "(covariance(model, of=...)) is the meaningful object "
            "for such a block")
    if of is not None:
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
                                    who=who, wording=wording)
    except _SingularBlock:
        if of is None:
            raise
        _refuse_rank("linearly dependent coordinates")
    free, active = cb.free, cb.active

    if hessian == "gauss-newton":
        groups = dict(session.res_rows)
        if not groups:
            raise ValueError(
                f"{who}: hessian='gauss-newton' needs declared "
                f"residuals ({hints['residual']}); the residual "
                "Jacobian is "
                "recovered from their rows. Without residual variables "
                "only the hessian='lagrangian' default is available.")
        Mi = _minv(M, who)
        R = np.zeros((n_params, n_params))
        for g, rws in groups.items():
            # slice LAST: J over the whole fitted block, so the pinned
            # rows exist for S below; res_rows is full-x like fit_rows,
            # zcols are factor rows (gh #450)
            Zr = np.array([[zcols[j][session.primal_row(r, who)]
                            for j in range(n_params)]
                           for r in rws])
            Jg = Zr @ Mi
            R += 2.0 * (Jg.T @ Jg)
    else:
        R = cb.R_exact
        if R is None:
            if n_var - n_eq == n_params:
                R = _tangent_reduced_hessian(session, M, zcols,
                                             who)
            elif of is not None:
                # a sub-block of the fitted set gets its marginal by
                # Schur complement of the exact tangent R over the
                # fitted block (never inverts a covariance, exact even
                # with pinned members); any other explicit block
                # reduces off the held factor with the item-1
                # corrections, which is benign for free coordinates
                # (no barrier term in the slice)
                R = _subblock_information(session, rows,
                                          n_var - n_eq, who)
                if R is None:
                    R = cb.R_corr
            else:
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
                f"{who}: the Lagrangian free block is indefinite "
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
            # build raised), the same non-determinism the of= gates
            # exist for; the exception clause stays as the last resort.
            # Applies on the DEFAULT path too, not only under of=: a
            # numerically dependent free block now refuses where it
            # could previously return a large-but-meaningless S
            # (CHANGELOG "Changed"). Diagonally scaled like the of=
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
                    f"{who}: the free block is singular, so the "
                    "pinned parameters' conditional information S is not "
                    "defined; the free block members are linearly "
                    "dependent")
        else:
            S = R[np.ix_(active, active)]
        info_mat[np.ix_(active, active)] = S

    return Information(
        params, info_mat,
        lambda: _conditioned_on(session, cb.act, rows, who))