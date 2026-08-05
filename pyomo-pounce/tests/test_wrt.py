"""wrt= block selection (covariance roadmap item 3): both accessors
reduce onto any block of the solve's variables off the held factor,
post-solve. The declared fitted block is the default, so the no-wrt
behavior is untouched; each call re-reduces onto its own argument; a
rank-deficient block (more coordinates than the fit has degrees of
freedom, the prediction-band case) gets its marginal covariance and a
refusal from information(); strongly active variables outside the
block come back on the result as conditioned_on."""
import warnings

import numpy as np
import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401
from pyomo_pounce import (
    covariance,
    declare_fitted,
    declare_residual,
    information,
)
from pyomo_pounce.sens import _rank_deficient

N = 25
SIGMA = 0.3


def linear_data():
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 4.0, N)
    y = 1.5 - 0.7 * x + SIGMA * rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])
    return x, y, X


def linear_model(x, y):
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, len(x) - 1)
    m.a = pyo.Var(initialize=0.0)
    m.b = pyo.Var(initialize=0.0)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.r[i] == float(y[i]) - mm.a
        - mm.b * float(x[i]))
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r)
    return m


def solved():
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    return m, X


def test_wrt_default_is_the_fitted_block():
    # wrt=[the fitted block, in order] must be EXACTLY the no-wrt
    # answer, both accessors: same matrix, same keys
    m, X = solved()
    cov0 = covariance(m, sigma_sq=SIGMA**2)
    cov1 = covariance(m, sigma_sq=SIGMA**2, wrt=[m.a, m.b])
    np.testing.assert_array_equal(cov0.matrix, cov1.matrix)
    info0 = information(m)
    info1 = information(m, wrt=[m.a, m.b])
    np.testing.assert_array_equal(info0.matrix, info1.matrix)
    assert cov1[m.a, m.b] == cov0[m.a, m.b]
    assert cov0.conditioned_on == () and cov1.conditioned_on == ()


def test_wrt_subblock_is_the_marginal():
    # wrt=[m.a] is the marginal over b: element 00 of the full
    # covariance, and its information is the inverse of that marginal
    # (NOT the conditional element R_aa)
    m, X = solved()
    C = SIGMA**2 * np.linalg.inv(X.T @ X)
    cov_a = covariance(m, sigma_sq=SIGMA**2, wrt=[m.a])
    assert cov_a.matrix.shape == (1, 1)
    assert cov_a[m.a] == pytest.approx(C[0, 0], rel=1e-9)
    info_a = information(m, wrt=[m.a])
    assert info_a[m.a] == pytest.approx(
        2.0 / np.linalg.inv(X.T @ X)[0, 0], rel=1e-9)
    # the sibling identity holds per block
    assert cov_a[m.a] * info_a[m.a] == pytest.approx(
        2.0 * SIGMA**2, rel=1e-9)
    # and gauss-newton profiles to the same marginal on a linear model
    gn_a = covariance(m, sigma_sq=SIGMA**2, hessian="gauss-newton",
                      wrt=[m.a])
    assert gn_a[m.a] == pytest.approx(C[0, 0], rel=1e-9)


def test_wrt_rank_deficient_block_is_the_prediction_band():
    # the residual block has 25 coordinates against 2 degrees of
    # freedom: its marginal covariance is the hat-matrix prediction
    # band sigma^2 X (X'X)^-1 X', membership handling is bypassed,
    # information() refuses toward covariance()
    m, X = solved()
    H = X @ np.linalg.solve(X.T @ X, X.T)
    cov_r = covariance(m, sigma_sq=SIGMA**2, wrt=m.r)
    np.testing.assert_allclose(cov_r.matrix, SIGMA**2 * H,
                               rtol=1e-8, atol=1e-12)
    assert cov_r[m.r[0]] == pytest.approx(SIGMA**2 * H[0, 0], rel=1e-8)
    with pytest.raises(RuntimeError, match="rank-deficient"):
        information(m, wrt=m.r)
    with pytest.raises(RuntimeError, match="rank-deficient"):
        covariance(m, sigma_sq=SIGMA**2, hessian="gauss-newton", wrt=m.r)


def test_wrt_conditioned_on_reports_the_outside_active_set():
    # a pinned by its bound, block = [b]: the block's number is the
    # value conditional on that bound (sigma^2 / (X'X)_11, the
    # fixed-intercept variance), and a comes back on conditioned_on.
    # The default block CONTAINS a, so its conditioned_on stays empty:
    # inside-block activity is membership, not conditioning.
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    m = linear_model(x, y)
    m.a.setlb(float(beta[0]) + 0.4)      # binds, strongly active
    pyo.SolverFactory("pounce").solve(m)
    cov_b = covariance(m, sigma_sq=SIGMA**2, wrt=[m.b])
    assert cov_b.conditioned_on == (m.a,)
    assert cov_b[m.b] == pytest.approx(
        SIGMA**2 / (X.T @ X)[1, 1], rel=1e-6)
    info_b = information(m, wrt=[m.b])
    assert info_b.conditioned_on == (m.a,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov_full = covariance(m, sigma_sq=SIGMA**2)
    assert cov_full.conditioned_on == ()
    # the rank-deficient bypass reports the same conditioning
    band = covariance(m, sigma_sq=SIGMA**2, wrt=m.r)
    assert band.conditioned_on == (m.a,)
    # a wrt block that is ENTIRELY pinned: information is the Schur
    # onto it (S with nothing free), covariance is the zero row
    R = 2.0 * X.T @ X
    S_a = R[0, 0] - R[0, 1] ** 2 / R[1, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info_pin = information(m, wrt=[m.a])
        cov_pin = covariance(m, sigma_sq=SIGMA**2, wrt=[m.a])
    assert info_pin[m.a] == pytest.approx(S_a, rel=1e-9)
    assert cov_pin[m.a] == 0.0


def test_wrt_accepted_forms():
    # a whole IndexedVar, a slice, a (Var, iterable) pair, and a mixed
    # list all normalize to the same coordinates
    m, X = solved()
    whole = covariance(m, sigma_sq=SIGMA**2, wrt=m.r)
    sliced = covariance(m, sigma_sq=SIGMA**2, wrt=m.r[:])
    paired = covariance(m, sigma_sq=SIGMA**2, wrt=(m.r, range(N)))
    np.testing.assert_array_equal(whole.matrix, sliced.matrix)
    np.testing.assert_array_equal(whole.matrix, paired.matrix)
    mixed = covariance(m, sigma_sq=SIGMA**2, wrt=[m.a, m.r[0]])
    assert mixed.matrix.shape == (2, 2)
    assert mixed[m.a] == pytest.approx(
        SIGMA**2 * np.linalg.inv(X.T @ X)[0, 0], rel=1e-9)
    # a tuple of two Vars is a block of two, not a (Var, iterable)
    # pair that eats the second Var as an index set (gh #466 review,
    # blocking): must equal the list form exactly
    tup = covariance(m, sigma_sq=SIGMA**2, wrt=(m.a, m.b))
    lst = covariance(m, sigma_sq=SIGMA**2, wrt=[m.a, m.b])
    assert tup.matrix.shape == (2, 2)
    np.testing.assert_array_equal(tup.matrix, lst.matrix)


def test_wrt_derived_sigma_uses_the_fits_degrees_of_freedom():
    # sigma estimated from the declared residuals divides by n minus
    # the FITTED count (2), not the block size (1): the sub-block
    # marginal must equal the corresponding element of the default
    # answer exactly, both built from the same derived sigma
    m, X = solved()
    cov_full = covariance(m)
    cov_a = covariance(m, wrt=[m.a])
    assert cov_a[m.a] == cov_full[m.a]
    assert cov_a.sigma_sq == cov_full.sigma_sq


def test_wrt_conditioned_on_is_scale_invariant():
    # the pinned outside variable enters the model at scale 1e-3, so
    # its barrier weight Sigma is ~1e6 smaller than in the plain
    # fixture and sits BELOW the strong mu-edge in absolute terms: a
    # Sigma-threshold heuristic misses it, while the singleton
    # reduced-level rule is a ratio of two quantities that scale
    # together and still calls it
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, N - 1)
    m.A = pyo.Var(initialize=0.0)        # A = 1000 * intercept
    m.b = pyo.Var(initialize=0.0)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.r[i] == float(y[i])
        - 1e-3 * mm.A - mm.b * float(x[i]))
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    m.A.setlb(1e3 * (float(beta[0]) + 0.4))   # binds, strongly active
    declare_fitted(m.A)
    declare_fitted(m.b)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    cov_b = covariance(m, sigma_sq=SIGMA**2, wrt=[m.b])
    assert cov_b.conditioned_on == (m.A,)


def test_wrt_subblock_schur_is_exact_with_a_pinned_member():
    # quadratic fit (a, b, c), c pinned by its bound, block = [a, c]:
    # the marginal comes from the Schur route off the exact tangent R
    # over the fitted block (b, free outside, is profiled out; c,
    # pinned INSIDE, is kept and handled by membership), so every
    # entry is exact at 1e-9 where a corrected inv(M_B) would carry
    # the pinned member's barrier residue. Expected values follow the
    # stated rule directly from R = 2 X'X.
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 4.0, N)
    y = 1.5 - 0.7 * x + 0.2 * x**2 + SIGMA * rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x, x**2])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, N - 1)
    m.a = pyo.Var(initialize=0.0)
    m.b = pyo.Var(initialize=0.0)
    m.c = pyo.Var(initialize=0.0)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.r[i] == float(y[i]) - mm.a
        - mm.b * float(x[i]) - mm.c * float(x[i] ** 2))
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    m.c.setlb(float(beta[2]) + 0.3)      # binds, strongly active
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_fitted(m.c)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    R = 2.0 * X.T @ X                    # order (a, b, c)
    sel = [0, 2]                         # the block (a, c)
    R_B = (R[np.ix_(sel, sel)]
           - np.outer(R[sel, 1], R[1, sel]) / R[1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = information(m, wrt=[m.a, m.c])
    assert info[m.a] == pytest.approx(R_B[0, 0], rel=1e-9)
    S_c = R_B[1, 1] - R_B[0, 1] ** 2 / R_B[0, 0]
    assert info[m.c] == pytest.approx(S_c, rel=1e-9)
    assert info[m.a, m.c] == 0.0
    assert info.conditioned_on == ()     # c is inside the block
    # Gauss-Newton profiles through the K-inverse chain and must land
    # on the same marginal, pinned member included
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gn = information(m, hessian="gauss-newton", wrt=[m.a, m.c])
    assert gn[m.a] == pytest.approx(R_B[0, 0], rel=1e-9)
    assert gn[m.c] == pytest.approx(S_c, rel=1e-9)


def test_wrt_exact_under_objective_scaling():
    # gradient-based scaling engaged (df != 1, asserted): the wrt
    # marginal, the prediction band, the information marginal, and
    # conditioned_on must all be unchanged in the model's own units.
    # Every other fixture in this file runs at df = 1, the axis that
    # bit items 0 through 2
    scale = 400.0
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 4.0, N)
    y = scale * (1.5 - 0.7 * x)         + (scale * SIGMA) * rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])
    C = np.linalg.inv(X.T @ X)
    sig2 = (scale * SIGMA) ** 2
    m = linear_model(x, y)
    for i in m.I:
        m.r[i].set_value(400.0)
    pyo.SolverFactory("pounce").solve(m)
    session = m.__dict__["_pounce_sens"].session
    assert abs(float(session.solver.nlp_scaling["obj"]) - 1.0) > 1e-6
    assert covariance(m, sigma_sq=sig2, wrt=[m.a])[m.a] == pytest.approx(
        sig2 * C[0, 0], rel=1e-9)
    band = covariance(m, sigma_sq=sig2, wrt=m.r)
    np.testing.assert_allclose(band.matrix, sig2 * (X @ C @ X.T),
                               rtol=1e-8, atol=1e-12)
    assert information(m, wrt=[m.a])[m.a] == pytest.approx(
        2.0 / C[0, 0], rel=1e-9)
    # and the singleton conditioned_on call under the same scaling
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    m2 = linear_model(x, y)
    for i in m2.I:
        m2.r[i].set_value(400.0)
    m2.a.setlb(float(beta[0]) + 0.4 * scale)
    pyo.SolverFactory("pounce").solve(m2)
    assert covariance(m2, sigma_sq=sig2,
                      wrt=[m2.b]).conditioned_on == (m2.a,)


def test_wrt_with_a_fixed_variable():
    # one inert fixed variable ahead of the block in .col order (gh
    # #450): the three new factor-indexing paths (the marginal slice,
    # the Schur route's fitted-level columns, the band) must all be
    # unchanged
    x, y, X = linear_data()
    C = np.linalg.inv(X.T @ X)
    m = linear_model(x, y)
    m.dead = pyo.Var(bounds=(2.0, 2.0), initialize=2.0)
    m.deadcon = pyo.Constraint(expr=m.dead * m.dead <= 1e6)
    pyo.SolverFactory("pounce").solve(m)
    sess = m.__dict__["_pounce_sens"].session
    rows = sess.solver.primal_rows(list(range(len(sess.var_names))))
    assert rows.count(None) == 1, rows
    assert sess.var_names.index("dead") < sess.var_names.index("a")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert covariance(m, sigma_sq=SIGMA**2,
                          wrt=[m.a])[m.a] == pytest.approx(
            SIGMA**2 * C[0, 0], rel=1e-9)
        assert information(m, wrt=[m.a])[m.a] == pytest.approx(
            2.0 / C[0, 0], rel=1e-9)
        band = covariance(m, sigma_sq=SIGMA**2, wrt=m.r)
    np.testing.assert_allclose(band.matrix, SIGMA**2 * (X @ C @ X.T),
                               rtol=1e-8, atol=1e-12)


def test_wrt_grouped_band_refuses():
    # per-group noise profiles Jacobians through inv(M), which a
    # rank-deficient block does not have: the refusal must fire
    # rather than a silent wrong sandwich
    x, y, _ = linear_data()
    n1 = 12
    m = pyo.ConcreteModel()
    m.I1 = pyo.RangeSet(0, n1 - 1)
    m.I2 = pyo.RangeSet(n1, N - 1)
    m.a = pyo.Var(initialize=0.0)
    m.b = pyo.Var(initialize=0.0)
    m.r1 = pyo.Var(m.I1, initialize=0.0)
    m.r2 = pyo.Var(m.I2, initialize=0.0)
    m.res1 = pyo.Constraint(
        m.I1, rule=lambda mm, i: mm.r1[i] == float(y[i]) - mm.a
        - mm.b * float(x[i]))
    m.res2 = pyo.Constraint(
        m.I2, rule=lambda mm, i: mm.r2[i] == float(y[i]) - mm.a
        - mm.b * float(x[i]))
    m.obj = pyo.Objective(
        expr=sum(m.r1[i] ** 2 for i in m.I1)
        + sum(m.r2[i] ** 2 for i in m.I2))
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r1, group="lo")
    declare_residual(m.r2, group="hi")
    pyo.SolverFactory("pounce").solve(m)
    block = list(m.r1.values()) + list(m.r2.values())
    with pytest.raises(RuntimeError, match="rank-deficient"):
        covariance(m, wrt=block)


def test_wrt_binding_row_falls_back_smoothly():
    # a binding general row whose support leaves the block: the Schur
    # route declines (its projection does not compose simply with
    # marginalization), the corrected reduction runs instead, the
    # mixed-row warning fires, and the sibling identity still holds
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    m = linear_model(x, y)
    m.capcon = pyo.Constraint(
        expr=m.a + m.b <= float(beta[0] + beta[1]) - 0.5)
    pyo.SolverFactory("pounce").solve(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cov_a = covariance(m, sigma_sq=SIGMA**2, wrt=[m.a])
        info_a = information(m, wrt=[m.a])
    assert any("capcon" in str(wi.message) for wi in w)
    # with wrt= given, the shared diagnostics speak block-relative
    assert any("variables outside the block" in str(wi.message)
               for wi in w)
    assert np.isfinite(cov_a[m.a]) and np.isfinite(info_a[m.a])
    assert cov_a[m.a] * info_a[m.a] == pytest.approx(
        2.0 * SIGMA**2, rel=1e-6)


def test_rank_gate_is_scale_invariant():
    # the rank gates decide collinearity, not units. numpy's default
    # tolerance is relative to the largest singular value, and a
    # covariance block carries the SQUARE of any unit spread between
    # its members, so a well-determined block whose coordinates differ
    # by ~1e9 in magnitude reads as rank-deficient to the raw test and
    # would be refused for its units alone. Unit-level, because the
    # spread needed to trip it is past what a fixture can solve
    # cleanly -- the matrices are exactly what the gates see.
    C = np.array([[1.0, 0.5], [0.5, 1.0]])       # full rank, cond 3
    D = np.diag([1e9, 1.0])
    M = D @ C @ D                                # cond ~1e18, by units
    assert np.linalg.matrix_rank(M) < 2          # the raw test is fooled
    assert not _rank_deficient(M)                # the scaled one is not
    # and genuine dependence is still caught at any scale
    dep = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert _rank_deficient(D @ dep @ D)
    assert _rank_deficient(dep)
    # an indefinite block (the information side, where the diagonal
    # can be negative) and a zero diagonal are both handled
    indef = np.array([[-4.0, 1.0], [1.0, 9.0]])
    assert not _rank_deficient(indef)
    assert _rank_deficient(np.zeros((2, 2)))


def test_wrt_within_count_dependent_block():
    # two residual coordinates at a DUPLICATED data point are
    # bit-identical rows of the K-inverse, so the 2-coordinate block
    # is dependent while sitting below the count gate: the rank gate
    # must route covariance to the marginal bypass (the rank-1
    # marginal is legitimate) and information to the refusal, with
    # the dependence-specific message, not the count claim
    x, y, X = linear_data()
    x = x.copy()
    x[1] = x[0]                          # duplicate the design point
    y = y.copy()
    y[1] = y[0]
    X = np.column_stack([np.ones(N), x])
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    H = X @ np.linalg.solve(X.T @ X, X.T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = covariance(m, sigma_sq=SIGMA**2, wrt=[m.r[0], m.r[1]])
    np.testing.assert_allclose(
        cov.matrix,
        SIGMA**2 * H[np.ix_([0, 1], [0, 1])], rtol=1e-8)
    assert cov[m.r[0]] == pytest.approx(cov[m.r[0], m.r[1]], rel=1e-12)
    with pytest.raises(RuntimeError,
                       match="linearly dependent coordinates"):
        information(m, wrt=[m.r[0], m.r[1]])


def test_wrt_error_paths():
    m, X = solved()
    with pytest.raises(ValueError, match="twice"):
        covariance(m, sigma_sq=SIGMA**2, wrt=[m.a, m.a])
    with pytest.raises(TypeError, match="not names"):
        covariance(m, sigma_sq=SIGMA**2, wrt="a")
    with pytest.raises(TypeError, match="covariance: wrt element 5"):
        covariance(m, sigma_sq=SIGMA**2, wrt=5)
    with pytest.raises(ValueError, match="empty block"):
        covariance(m, sigma_sq=SIGMA**2, wrt=[])
    m2 = pyo.ConcreteModel()
    m2.q = pyo.Var(initialize=0.0)
    with pytest.raises(ValueError, match="not a variable of the solved"):
        covariance(m, sigma_sq=SIGMA**2, wrt=[m2.q])
    # a fixed (equal-bounds) variable has no factor row to reduce onto
    x, y, _ = linear_data()
    m3 = linear_model(x, y)
    m3.dead = pyo.Var(bounds=(2.0, 2.0), initialize=2.0)
    m3.deadcon = pyo.Constraint(expr=m3.dead * m3.dead <= 1e6)
    pyo.SolverFactory("pounce").solve(m3)
    with pytest.raises(ValueError, match="removed from the solve"):
        covariance(m3, sigma_sq=SIGMA**2, wrt=[m3.dead])
