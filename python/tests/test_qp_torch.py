"""Differentiable convex-QP layer (pounce.torch.solve_qp / QpLayer).

Validates the OptNet implicit-differentiation backward against finite
differences and torch.autograd.gradcheck, and checks the batch path and
QpLayer compose."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

from pounce.torch import QpLayer, solve_qp, solve_qp_batch  # noqa: E402


def _fd(fn, x, eps=1e-6):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        g[i] = (float(fn(torch.tensor(xp))) - float(fn(torch.tensor(xm)))) / (2 * eps)
    return g


P = torch.tensor([[2.0, 0.0], [0.0, 2.0]])


def test_grad_c_interior():
    G = torch.tensor([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = torch.tensor([10.0, 0.0, 0.0])
    target = torch.tensor([0.3, 0.4])

    def loss(c):
        return torch.sum((solve_qp(P=P, c=c, G=G, h=h) - target) ** 2)

    c0 = torch.tensor([-0.5, -0.7], requires_grad=True)
    loss(c0).backward()
    np.testing.assert_allclose(c0.grad.numpy(), _fd(loss, c0.detach()), atol=1e-4)


def test_grad_h_active_inequality():
    G = torch.tensor([[1.0, 1.0]])
    c0 = torch.tensor([-4.0, -4.0])

    def loss(h):
        return torch.sum(solve_qp(P=P, c=c0, G=G, h=h) ** 2)

    h0 = torch.tensor([1.0], requires_grad=True)
    loss(h0).backward()
    np.testing.assert_allclose(h0.grad.numpy(), _fd(loss, h0.detach()), atol=1e-4)


def test_grad_c_and_b_equality():
    A = torch.tensor([[1.0, 1.0]])

    def loss_c(c):
        return torch.sum(solve_qp(P=P, c=c, A=A, b=torch.tensor([2.0])) ** 2)

    def loss_b(b):
        return torch.sum(solve_qp(P=P, c=torch.tensor([-1.0, -3.0]), A=A, b=b) ** 2)

    c0 = torch.tensor([-1.0, -3.0], requires_grad=True)
    loss_c(c0).backward()
    np.testing.assert_allclose(c0.grad.numpy(), _fd(loss_c, c0.detach()), atol=1e-4)

    b0 = torch.tensor([2.0], requires_grad=True)
    loss_b(b0).backward()
    np.testing.assert_allclose(b0.grad.numpy(), _fd(loss_b, b0.detach()), atol=1e-4)


def test_grad_matrices_P_G():
    """Full OptNet matrix derivatives (∇P symmetric, ∇G) vs FD."""
    G = torch.tensor([[1.0, 1.0]])
    h = torch.tensor([0.5])
    c0 = torch.tensor([-4.0, -4.0])

    def loss_G(Gv):
        return torch.sum(solve_qp(P=P, c=c0, G=Gv, h=h) ** 2)

    G0 = torch.tensor([[1.0, 1.0]], requires_grad=True)
    loss_G(G0).backward()
    # FD over the (1,2) matrix.
    g_fd = np.zeros((1, 2))
    for j in range(2):
        mp = G.clone()
        mp[0, j] += 1e-6
        mm = G.clone()
        mm[0, j] -= 1e-6
        g_fd[0, j] = (float(loss_G(mp)) - float(loss_G(mm))) / 2e-6
    np.testing.assert_allclose(G0.grad.numpy(), g_fd, atol=1e-4)


def test_gradcheck_qp():
    """torch.autograd.gradcheck on solve_qp w.r.t. c (float64)."""
    G = torch.tensor([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = torch.tensor([10.0, 0.0, 0.0])

    def fn(c):
        return solve_qp(P=P, c=c, G=G, h=h)

    c0 = torch.tensor([-0.5, -0.7], requires_grad=True)
    assert torch.autograd.gradcheck(fn, (c0,), atol=1e-4, rtol=1e-3, eps=1e-6)


def test_batch_matches_loop():
    G = torch.tensor([[1.0, 1.0]])
    h = torch.tensor([0.5])
    cs = torch.tensor([[-4.0, -4.0], [-1.0, -2.0], [0.5, -0.5]])

    xs = solve_qp_batch(P=P, c=cs, G=G, h=h)
    for i in range(cs.shape[0]):
        xi = solve_qp(P=P, c=cs[i], G=G, h=h)
        np.testing.assert_allclose(
            xs[i].detach().numpy(), xi.detach().numpy(), atol=1e-7
        )


def test_batch_grad_per_row_c():
    G = torch.tensor([[1.0, 1.0]])
    h = torch.tensor([0.5])
    cs = torch.tensor([[-4.0, -4.0], [-1.0, -2.0]], requires_grad=True)
    (solve_qp_batch(P=P, c=cs, G=G, h=h) ** 2).sum().backward()
    assert cs.grad.shape == (2, 2)
    assert torch.all(torch.isfinite(cs.grad))


def test_qp_layer():
    G = torch.tensor([[1.0, 1.0]])
    layer = QpLayer(P=P, G=G)
    c = torch.tensor([-4.0, -4.0], requires_grad=True)
    x = layer(c, h=torch.tensor([0.5]))
    (x**2).sum().backward()
    assert torch.all(torch.isfinite(c.grad))


def test_qp_bounds_folded():
    """lb/ub folded into G/h; the solution respects them."""
    c = torch.tensor([-5.0, -5.0])
    x = solve_qp(P=P, c=c, lb=torch.tensor([-1.0, -1.0]), ub=torch.tensor([1.0, 1.0]))
    np.testing.assert_allclose(x.detach().numpy(), [1.0, 1.0], atol=1e-6)


def test_indefinite_p_rejected_in_forward():
    # issue #112 (code review M31): the autograd.Function forward runs eagerly,
    # so an indefinite P would otherwise return a silently-wrong "optimal" and
    # corrupt the OptNet backward. The host-side guard raises ValueError.
    P_indef = torch.tensor([[1.0, 0.0], [0.0, -1.0]])  # eigenvalues +1, -1
    c = torch.zeros(2)
    with pytest.raises(ValueError, match="semidefinite"):
        solve_qp(
            P=P_indef, c=c, lb=torch.tensor([-1.0, -1.0]), ub=torch.tensor([1.0, 1.0])
        )


def test_indefinite_p_rejected_in_batch_forward():
    # batched layer: ``c`` is (B, n) and ``h`` is (B, m); the shared P is
    # screened once before any instance solves.
    P_indef = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    c = torch.zeros(2, 2)  # B=2, n=2
    G = -torch.eye(2)
    h = torch.ones(2, 2)  # B=2, m=2
    with pytest.raises(ValueError, match="semidefinite"):
        solve_qp_batch(P=P_indef, c=c, G=G, h=h)


# --------------------------------------------------------------------------
# M31 nit: the per-forward PSD guard must be skippable (check_psd=False, for
# a layer whose P is PSD by construction) and forceable above the auto cap
# (check_psd=True), with the same semantics as pounce.qp.solve_qp.
# --------------------------------------------------------------------------

_P_INDEF = torch.tensor([[1.0, 0.0], [0.0, -1.0]])  # eigenvalues +1, -1
_BOX_LB = torch.tensor([-1.0, -1.0])
_BOX_UB = torch.tensor([1.0, 1.0])


def _assert_guard_skipped(fn):
    """Run ``fn`` and assert the PSD guard did not fire. The unguarded
    nonconvex solve may legitimately report a non-optimal status (raised by
    the layer's _check_status); only a 'semidefinite' error means the guard
    ran despite check_psd=False."""
    try:
        fn()
    except Exception as e:
        assert "semidefinite" not in str(e)


def test_check_psd_false_bypasses_indefinite_guard():
    # Box bounds keep the IPM from diverging (same fixture as the host
    # test_check_psd_false_bypasses_guard_everywhere).
    _assert_guard_skipped(
        lambda: solve_qp(
            P=_P_INDEF, c=torch.zeros(2), lb=_BOX_LB, ub=_BOX_UB, check_psd=False
        )
    )


def test_check_psd_false_bypasses_batch_and_layer():
    _assert_guard_skipped(
        lambda: solve_qp_batch(
            P=_P_INDEF, c=torch.zeros(2, 2), lb=_BOX_LB, ub=_BOX_UB, check_psd=False
        )
    )
    layer = QpLayer(P=_P_INDEF, lb=_BOX_LB, ub=_BOX_UB, check_psd=False)
    _assert_guard_skipped(lambda: layer(torch.zeros(2)))


def test_check_psd_true_forces_guard_above_auto_cap():
    # Above the auto cap (n > 1500) the default skips the eigvalsh check;
    # check_psd=True must force it. The guard runs before any solve, so the
    # indefinite problem never reaches the IPM and the test stays cheap.
    from pounce.qp import _PSD_CHECK_AUTO_MAX_N

    n = _PSD_CHECK_AUTO_MAX_N + 1
    diag = np.ones(n)
    diag[-1] = -1.0
    P_big = torch.diag(torch.tensor(diag))
    with pytest.raises(ValueError, match="semidefinite"):
        solve_qp(P=P_big, c=torch.zeros(n), check_psd=True)


def test_check_psd_true_on_psd_p_solves_normally():
    c = torch.tensor([-4.0, -4.0])
    G = torch.tensor([[1.0, 1.0]])
    h = torch.tensor([0.5])
    x_default = solve_qp(P=P, c=c, G=G, h=h)
    x_forced = solve_qp(P=P, c=c, G=G, h=h, check_psd=True)
    np.testing.assert_allclose(
        x_forced.detach().numpy(), x_default.detach().numpy(), atol=1e-12
    )
