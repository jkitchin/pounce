"""Differentiable convex-QP layer (pounce.jax.solve_qp / QpLayer).

Validates the OptNet implicit-differentiation backward against finite
differences for the linear/RHS parameters (c, b, h), and checks
jacrev / vmap / QpLayer compose.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from pounce.jax import QpLayer, solve_qp, solve_qp_batch  # noqa: E402


def _fd(fn, x, eps=1e-6):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        g[i] = (float(fn(jnp.array(xp))) - float(fn(jnp.array(xm)))) / (2 * eps)
    return g


def _fd_mat(fn, M, eps=1e-6):
    """Finite-difference gradient of a scalar ``fn`` over a dense matrix."""
    M = np.asarray(M, float)
    g = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            mp = M.copy()
            mp[i, j] += eps
            mm = M.copy()
            mm[i, j] -= eps
            g[i, j] = (float(fn(jnp.array(mp))) - float(fn(jnp.array(mm)))) / (2 * eps)
    return g


def _fd_mat_sym(fn, M, eps=1e-6):
    """Finite-difference gradient over a *symmetric* matrix: perturb the
    (i, j) and (j, i) entries together so the symmetry is preserved. The
    returned array matches the symmetrized analytic gradient."""
    M = np.asarray(M, float)
    g = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(i, M.shape[1]):
            mp = M.copy()
            mm = M.copy()
            mp[i, j] += eps
            mm[i, j] -= eps
            if i != j:
                mp[j, i] += eps
                mm[j, i] -= eps
            d = (float(fn(jnp.array(mp))) - float(fn(jnp.array(mm)))) / (2 * eps)
            # d is ∂/∂(symmetric pair); split across the two entries.
            if i == j:
                g[i, j] = d
            else:
                g[i, j] = d / 2
                g[j, i] = d / 2
    return g


P = jnp.array([[2.0, 0.0], [0.0, 2.0]])


def test_grad_c_interior():
    # Interior inequalities: gradient flows only through c.
    G = jnp.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([10.0, 0.0, 0.0])
    target = jnp.array([0.3, 0.4])

    def loss(c):
        return jnp.sum((solve_qp(P=P, c=c, G=G, h=h) - target) ** 2)

    c0 = jnp.array([-0.5, -0.7])
    g = jax.grad(loss)(c0)
    np.testing.assert_allclose(np.asarray(g), _fd(loss, c0), atol=1e-4)


def test_grad_h_active_inequality():
    # Active inequality x0+x1 ≤ h: gradient flows through h.
    G = jnp.array([[1.0, 1.0]])
    c0 = jnp.array([-4.0, -4.0])  # pulls past the constraint → active

    def loss(h):
        return jnp.sum(solve_qp(P=P, c=c0, G=G, h=h) ** 2)

    h0 = jnp.array([1.0])
    g = jax.grad(loss)(h0)
    np.testing.assert_allclose(np.asarray(g), _fd(loss, h0), atol=1e-4)


def test_grad_c_and_b_equality():
    A = jnp.array([[1.0, 1.0]])

    def loss_c(c):
        return jnp.sum(solve_qp(P=P, c=c, A=A, b=jnp.array([2.0])) ** 2)

    def loss_b(b):
        return jnp.sum(solve_qp(P=P, c=jnp.array([-1.0, -3.0]), A=A, b=b) ** 2)

    c0 = jnp.array([-1.0, -3.0])
    b0 = jnp.array([2.0])
    np.testing.assert_allclose(
        np.asarray(jax.grad(loss_c)(c0)), _fd(loss_c, c0), atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(jax.grad(loss_b)(b0)), _fd(loss_b, b0), atol=1e-4
    )


def test_jacrev_of_solution():
    # Jacobian of x*(c) w.r.t. c via jacrev should be well-formed.
    G = jnp.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = jnp.array([10.0, 0.0, 0.0])
    c0 = jnp.array([-0.5, -0.7])
    J = jax.jacrev(lambda c: solve_qp(P=P, c=c, G=G, h=h))(c0)
    assert J.shape == (2, 2)
    # For an interior solution of ½·2‖x‖²+cᵀx, x* = −c/2, so dx/dc = −½I.
    np.testing.assert_allclose(np.asarray(J), -0.5 * np.eye(2), atol=1e-5)


def test_qp_layer_and_vmap():
    # QpLayer captures fixed structure; vmap over a batch of objectives.
    G = jnp.array([[1.0, 1.0]])
    layer = QpLayer(P=P, G=G)
    cs = jnp.array([[-1.0, -1.0], [-4.0, -4.0], [0.5, 0.5]])
    hs = jnp.array([[1.0], [1.0], [1.0]])
    xs = jax.vmap(lambda c, h: layer(c, h=h))(cs, hs)
    assert xs.shape == (3, 2)
    # Each row matches a direct solve.
    for i in range(3):
        xi = solve_qp(P=P, c=cs[i], G=G, h=hs[i])
        np.testing.assert_allclose(np.asarray(xs[i]), np.asarray(xi), atol=1e-5)


# --- Matrix gradients (P, G, A) ---------------------------------------


# Matrix-perturbation finite differences amplify the solver's residual
# tolerance (≈ noise/eps), so tighten the IPM tolerance for these checks.
_TIGHT = dict(tol=1e-11, max_iter=200)


def test_grad_P_symmetric():
    # ∇P on an active-inequality QP, checked with symmetric perturbations.
    G = jnp.array([[1.0, 2.0]])
    h = jnp.array([1.0])
    c0 = jnp.array([-4.0, -1.0])
    target = jnp.array([0.2, 0.3])

    def loss(Pm):
        return jnp.sum((solve_qp(P=Pm, c=c0, G=G, h=h, **_TIGHT) - target) ** 2)

    P0 = jnp.array([[3.0, 0.5], [0.5, 2.0]])
    g = jax.grad(loss)(P0)
    np.testing.assert_allclose(np.asarray(g), _fd_mat_sym(loss, P0), atol=1e-4)


def test_grad_G_active_inequality():
    # ∇G with an active inequality: gradient flows through the constraint
    # matrix.
    h = jnp.array([1.0])
    c0 = jnp.array([-4.0, -4.0])

    def loss(Gm):
        return jnp.sum(solve_qp(P=P, c=c0, G=Gm, h=h, **_TIGHT) ** 2)

    G0 = jnp.array([[1.0, 1.0]])
    g = jax.grad(loss)(G0)
    np.testing.assert_allclose(np.asarray(g), _fd_mat(loss, G0), atol=1e-4)


def test_grad_A_equality():
    # ∇A with an equality constraint.
    b = jnp.array([1.0])
    c0 = jnp.array([-1.0, -3.0])

    def loss(Am):
        return jnp.sum(solve_qp(P=P, c=c0, A=Am, b=b, **_TIGHT) ** 2)

    A0 = jnp.array([[1.0, 2.0]])
    g = jax.grad(loss)(A0)
    np.testing.assert_allclose(np.asarray(g), _fd_mat(loss, A0), atol=1e-4)


# --- Parallel differentiable batch ------------------------------------


def test_solve_qp_batch_matches_single():
    G = jnp.array([[1.0, 1.0]])
    cs = jnp.array([[-1.0, -1.0], [-4.0, -4.0], [0.5, 0.5]])
    hs = jnp.array([[5.0], [1.0], [5.0]])
    xs = solve_qp_batch(P=P, c=cs, G=G, h=hs)
    assert xs.shape == (3, 2)
    for i in range(3):
        xi = solve_qp(P=P, c=cs[i], G=G, h=hs[i])
        np.testing.assert_allclose(np.asarray(xs[i]), np.asarray(xi), atol=1e-5)


def test_solve_qp_batch_grad_c_per_row():
    # Per-row gradient w.r.t. c matches summing each instance's grad.
    G = jnp.array([[1.0, 1.0]])
    hs = jnp.array([[5.0], [5.0]])  # inactive → interior, dx/dc = -½I

    def loss(cs):
        return jnp.sum(solve_qp_batch(P=P, c=cs, G=G, h=hs) ** 2)

    cs0 = jnp.array([[-0.5, -0.7], [0.3, -0.2]])
    g = jax.grad(loss)(cs0)
    # Interior: x = -c/2, loss row = ‖c/2‖², dloss/dc = c/2.
    np.testing.assert_allclose(np.asarray(g), np.asarray(cs0) / 2.0, atol=1e-5)


def test_warm_start_same_solution_and_grad():
    # A warm start must not change the solution or its gradient — only the
    # iteration count (which we can't see from JAX). Check x and ∇c match.
    G = jnp.array([[1.0, 1.0]])
    h = jnp.array([1.0])
    c0 = jnp.array([-4.0, -4.0])

    cold = solve_qp(P=P, c=c0, G=G, h=h)
    warm = solve_qp(P=P, c=c0, G=G, h=h, warm_start=cold)
    np.testing.assert_allclose(np.asarray(cold), np.asarray(warm), atol=1e-7)

    def loss(c, ws=None):
        return jnp.sum(solve_qp(P=P, c=c, G=G, h=h, warm_start=ws) ** 2)

    g_cold = jax.grad(lambda c: loss(c))(c0)
    # Warm start passed as a plain primal array; gradient must be identical.
    g_warm = jax.grad(lambda c: loss(c, ws=np.asarray(cold)))(c0)
    np.testing.assert_allclose(np.asarray(g_cold), np.asarray(g_warm), atol=1e-6)


def test_solve_qp_batch_warm_same_solution_and_grad():
    # Batch warm start: same xs and same ∇c as cold; only iterations differ.
    G = jnp.array([[1.0, 1.0]])
    cs = jnp.array([[-1.0, -1.0], [-4.0, -4.0], [0.5, 0.5]])
    hs = jnp.array([[5.0], [1.0], [5.0]])

    cold = solve_qp_batch(P=P, c=cs, G=G, h=hs)
    warm = solve_qp_batch(P=P, c=cs, G=G, h=hs, warm_start=cold)
    np.testing.assert_allclose(np.asarray(cold), np.asarray(warm), atol=1e-6)

    def loss(cs_, ws=None):
        return jnp.sum(solve_qp_batch(P=P, c=cs_, G=G, h=hs, warm_start=ws) ** 2)

    g_cold = jax.grad(lambda cs_: loss(cs_))(cs)
    g_warm = jax.grad(lambda cs_: loss(cs_, ws=np.asarray(cold)))(cs)
    np.testing.assert_allclose(np.asarray(g_cold), np.asarray(g_warm), atol=1e-6)


def test_solve_qp_batch_grad_shared_P_sums():
    # Gradient w.r.t. the shared P equals the sum of per-instance ∇P.
    cs = jnp.array([[-1.0, -2.0], [-3.0, 0.5]])

    def loss_batch(Pm):
        return jnp.sum(solve_qp_batch(P=Pm, c=cs) ** 2)

    def loss_single(Pm, c):
        return jnp.sum(solve_qp(P=Pm, c=c) ** 2)

    P0 = jnp.array([[3.0, 0.5], [0.5, 2.0]])
    g_batch = jax.grad(loss_batch)(P0)
    g_sum = sum(jax.grad(lambda Pm, c=c: loss_single(Pm, c))(P0) for c in cs)
    np.testing.assert_allclose(np.asarray(g_batch), np.asarray(g_sum), atol=1e-5)


def test_infeasible_forward_raises():
    """B3 regression: a non-optimal forward solve must raise, not return a
    silent garbage iterate (which would feed meaningless gradients into a
    downstream optimizer). Inconsistent equalities x0=1 and x0=2 are
    primal-infeasible."""
    P = jnp.array([[2.0]])
    c = jnp.array([0.0])
    A = jnp.array([[1.0], [1.0]])
    b = jnp.array([1.0, 2.0])
    with pytest.raises(RuntimeError, match="status"):
        solve_qp(P=P, c=c, A=A, b=b)


def test_infeasible_grad_raises():
    """The differentiation path must also surface the failure rather than
    differentiate through a non-KKT point."""
    P = jnp.array([[2.0]])
    A = jnp.array([[1.0], [1.0]])
    b = jnp.array([1.0, 2.0])

    def loss(c):
        return jnp.sum(solve_qp(P=P, c=c, A=A, b=b) ** 2)

    with pytest.raises(RuntimeError, match="status"):
        jax.grad(loss)(jnp.array([0.0]))


def test_indefinite_p_rejected_in_forward():
    # issue #112 (code review M31): the differentiable layer must not solve an
    # indefinite QP and feed a silently-wrong "optimal" into the backward pass.
    # The host-callback guard raises; jax wraps it, but the PSD message
    # survives in the error string.
    P = jnp.array([[1.0, 0.0], [0.0, -1.0]])  # eigenvalues +1, -1
    c = jnp.zeros(2)
    G = -jnp.eye(2)
    h = jnp.ones(2)
    with pytest.raises(Exception, match="semidefinite"):
        solve_qp(P=P, c=c, G=G, h=h)


def test_indefinite_p_rejected_in_batch_forward():
    # batched layer: ``c`` is (B, n) and ``h`` is (B, m); the shared P is
    # screened once before any instance solves.
    P = jnp.array([[1.0, 0.0], [0.0, -1.0]])
    c = jnp.zeros((2, 2))  # B=2, n=2
    G = -jnp.eye(2)
    h = jnp.ones((2, 2))  # B=2, m=2
    with pytest.raises(Exception, match="semidefinite"):
        solve_qp_batch(P=P, c=c, G=G, h=h)


# --------------------------------------------------------------------------
# M31 nit: the per-forward PSD guard must be skippable (check_psd=False, for
# a layer whose P is PSD by construction) and forceable above the auto cap
# (check_psd=True), with the same semantics as pounce.qp.solve_qp.
# --------------------------------------------------------------------------

_P_INDEF = jnp.array([[1.0, 0.0], [0.0, -1.0]])  # eigenvalues +1, -1
_BOX_LB = np.array([-1.0, -1.0])
_BOX_UB = np.array([1.0, 1.0])


def _assert_guard_skipped(fn):
    """Run ``fn`` and assert the PSD guard did not fire. The unguarded
    nonconvex solve may legitimately report a non-optimal status (raised by
    the layer's _check_status, possibly wrapped by jax); only a
    'semidefinite' error means the guard ran despite check_psd=False."""
    try:
        fn()
    except Exception as e:
        assert "semidefinite" not in str(e)


def test_check_psd_false_bypasses_indefinite_guard():
    # Box bounds keep the IPM from diverging (same fixture as the host
    # test_check_psd_false_bypasses_guard_everywhere).
    _assert_guard_skipped(
        lambda: solve_qp(
            P=_P_INDEF, c=jnp.zeros(2), lb=_BOX_LB, ub=_BOX_UB, check_psd=False
        )
    )


def test_check_psd_false_bypasses_batch_and_layer():
    _assert_guard_skipped(
        lambda: solve_qp_batch(
            P=_P_INDEF, c=jnp.zeros((2, 2)), lb=_BOX_LB, ub=_BOX_UB, check_psd=False
        )
    )
    layer = QpLayer(P=_P_INDEF, lb=_BOX_LB, ub=_BOX_UB, check_psd=False)
    _assert_guard_skipped(lambda: layer(jnp.zeros(2)))


def test_check_psd_true_forces_guard_above_auto_cap():
    # Above the dense-check ceiling (n > 1500) the guard used to be skipped
    # entirely at the default, which was gh #849; it now runs at every size by
    # an inertia count instead, and check_psd=True has always forced it. Both
    # settings must refuse. The guard runs before any solve, so the indefinite
    # problem never reaches the IPM and the test stays cheap.
    from pounce.qp import _PSD_CHECK_DENSE_MAX_N

    n = _PSD_CHECK_DENSE_MAX_N + 1
    diag = np.ones(n)
    diag[-1] = -1.0
    P_big = jnp.diag(jnp.asarray(diag))
    with pytest.raises(Exception, match="semidefinite"):
        solve_qp(P=P_big, c=jnp.zeros(n), check_psd=True)
    # gh #849: and at the default, which used to sail straight past.
    with pytest.raises(Exception, match="semidefinite"):
        solve_qp(P=P_big, c=jnp.zeros(n))


def test_check_psd_true_on_psd_p_solves_normally():
    P = jnp.array([[2.0, 0.0], [0.0, 2.0]])
    c = jnp.array([-4.0, -4.0])
    G = jnp.array([[1.0, 1.0]])
    h = jnp.array([0.5])
    x_default = solve_qp(P=P, c=c, G=G, h=h)
    x_forced = solve_qp(P=P, c=c, G=G, h=h, check_psd=True)
    np.testing.assert_allclose(np.asarray(x_forced), np.asarray(x_default), atol=1e-12)


# --------------------------------------------------------------------------
# jnp lb/ub under jax.jit (issue #795)
#
# `_expand_bounds` used to test each bound with ``float(ub[i])``. Under jit
# every jnp op is staged, so indexing a jnp bound produced a tracer and
# ``float()`` raised ConcretizationTypeError -- while the identical numpy
# array, indexed eagerly, worked. The bounds decide *which* constant rows
# are folded into G/h, so they are read on the host with one np.asarray.
#
# The inactive-bounds case alone cannot see the folded rows at all (drop
# every bound row and it still passes), so each case below either puts a
# bound on the solution or asserts the row count.
# --------------------------------------------------------------------------

_B_P = np.eye(2) * 2.0
_B_C = jnp.array([-1.0, 2.0])
_B_UNCONSTRAINED = np.array([0.5, -1.0])  # x* = -P^-1 c, bounds inactive


def _solve_bounded(lb, ub, c=_B_C):
    return solve_qp(P=_B_P, c=c, lb=lb, ub=ub)


# Built outside any trace, as the issue's repro does -- a jnp array
# constructed *inside* a jit is a tracer with no recoverable value (jax
# stages even a tracer-free op), which is the separate, still-unsupported
# case pinned by test_traced_bounds_raise_actionable_error below.
_B_LB_WIDE, _B_UB_WIDE = jnp.full(2, -10.0), jnp.full(2, 10.0)
_B_LB, _B_UB = jnp.array([-0.5, -0.5]), jnp.array([0.25, 0.25])
_B_BOUNDED = np.array([0.25, -0.5])  # both bounds bind


def test_jit_jnp_bounds_inactive_matches_closed_form():
    x = jax.jit(lambda cc: _solve_bounded(_B_LB_WIDE, _B_UB_WIDE, cc))(_B_C)
    np.testing.assert_allclose(np.asarray(x), _B_UNCONSTRAINED, atol=1e-8)


def test_jit_jnp_bounds_active_matches_numpy_bounds():
    # Both bounds bind, so a fix that quietly dropped the folded rows fails
    # here even though it would pass the inactive case above.
    x_jnp = jax.jit(lambda cc: _solve_bounded(_B_LB, _B_UB, cc))(_B_C)
    x_np = jax.jit(
        lambda cc: _solve_bounded(np.asarray(_B_LB), np.asarray(_B_UB), cc)
    )(_B_C)
    np.testing.assert_allclose(np.asarray(x_jnp), np.asarray(x_np), atol=1e-8)
    np.testing.assert_allclose(np.asarray(x_jnp), _B_BOUNDED, atol=1e-6)


@pytest.mark.parametrize(
    "lb, ub",
    [
        (jnp.array([-0.5, -0.5]), np.array([0.25, 0.25])),  # only lb is jnp
        (np.array([-0.5, -0.5]), jnp.array([0.25, 0.25])),  # only ub is jnp
        ([-0.5, -0.5], [0.25, 0.25]),  # python lists
    ],
)
def test_jit_mixed_bound_containers(lb, ub):
    # Either side alone triggered the defect.
    x = jax.jit(lambda cc: _solve_bounded(lb, ub, cc))(_B_C)
    np.testing.assert_allclose(np.asarray(x), [0.25, -0.5], atol=1e-6)


_B_LB_INF = jnp.array([-jnp.inf, -0.5])
_B_UB_INF = jnp.array([0.25, jnp.inf])


def test_jit_jnp_bounds_infinite_entries_fold_only_finite_rows():
    # Structural selection has to survive the host read: only x0's upper
    # bound and x1's lower bound are finite, and both bind.
    x = jax.jit(lambda cc: _solve_bounded(_B_LB_INF, _B_UB_INF, cc))(_B_C)
    np.testing.assert_allclose(np.asarray(x), [0.25, -0.5], atol=1e-6)

    from pounce.jax._qp import _expand_bounds

    G_full, h_full = _expand_bounds(
        jnp.zeros((0, 2)), jnp.zeros((0,)), _B_LB_INF, _B_UB_INF, 2
    )
    assert G_full.shape == (2, 2) and h_full.shape == (2,)
    np.testing.assert_allclose(np.asarray(G_full), [[1.0, 0.0], [0.0, -1.0]])
    np.testing.assert_allclose(np.asarray(h_full), [0.25, 0.5])


def test_jnp_bounds_eager_and_vmap_still_work():
    np.testing.assert_allclose(
        np.asarray(_solve_bounded(_B_LB, _B_UB)), _B_BOUNDED, atol=1e-6
    )
    cs = jnp.stack([_B_C, _B_C * 0.5])
    xs = jax.vmap(lambda cc: _solve_bounded(_B_LB, _B_UB, cc))(cs)
    assert xs.shape == (2, 2)
    np.testing.assert_allclose(np.asarray(xs[0]), _B_BOUNDED, atol=1e-6)


def test_jit_jnp_bounds_still_differentiates_c():
    g = jax.jit(jax.grad(lambda cc: _solve_bounded(_B_LB_WIDE, _B_UB_WIDE, cc).sum()))(
        _B_C
    )
    # x = -P^-1 c with the bounds inactive, so dx_sum/dc = -diag(1/2).
    np.testing.assert_allclose(np.asarray(g), [-0.5, -0.5], atol=1e-6)


def test_jit_jnp_bounds_batch_and_layer():
    cs = jnp.stack([_B_C, _B_C])
    xs = jax.jit(lambda cc: solve_qp_batch(P=_B_P, c=cc, lb=_B_LB, ub=_B_UB))(cs)
    np.testing.assert_allclose(np.asarray(xs[0]), _B_BOUNDED, atol=1e-6)
    np.testing.assert_allclose(np.asarray(xs[1]), _B_BOUNDED, atol=1e-6)

    layer = QpLayer(P=_B_P, lb=_B_LB, ub=_B_UB)
    x = jax.jit(layer)(_B_C)
    np.testing.assert_allclose(np.asarray(x), _B_BOUNDED, atol=1e-6)


def test_traced_bounds_raise_actionable_error():
    # A bound built *in* the trace cannot be read on the host at all -- the
    # row structure depends on it, and jax stages the construction whether
    # or not it depends on an argument. Point at hoisting and at the
    # documented G/h route rather than letting jax's bare conversion error
    # out. (This is #740's workaround territory, not #795's defect.)
    def f(scale):
        return solve_qp(P=_B_P, c=_B_C, lb=-scale * jnp.ones(2), ub=scale * jnp.ones(2))

    with pytest.raises(ValueError, match=r"`(lb|ub)` must be a concrete array"):
        jax.jit(f)(jnp.float64(0.25))

    def g(cc):
        # No dependence on the argument at all -- still staged, still caught.
        return solve_qp(P=_B_P, c=cc, ub=jnp.full(2, 0.25))

    with pytest.raises(ValueError, match=r"built outside the trace"):
        jax.jit(g)(_B_C)
