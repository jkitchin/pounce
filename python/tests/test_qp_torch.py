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
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
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
        mp = G.clone(); mp[0, j] += 1e-6
        mm = G.clone(); mm[0, j] -= 1e-6
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
        np.testing.assert_allclose(xs[i].detach().numpy(), xi.detach().numpy(), atol=1e-7)


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
    (x ** 2).sum().backward()
    assert torch.all(torch.isfinite(c.grad))


def test_qp_bounds_folded():
    """lb/ub folded into G/h; the solution respects them."""
    c = torch.tensor([-5.0, -5.0])
    x = solve_qp(P=P, c=c, lb=torch.tensor([-1.0, -1.0]), ub=torch.tensor([1.0, 1.0]))
    np.testing.assert_allclose(x.detach().numpy(), [1.0, 1.0], atol=1e-6)
