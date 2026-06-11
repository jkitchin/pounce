"""Differentiable SOCP layer (pounce.torch.solve_socp).

Validates the cone-aware OptNet backward (arrow operators in the
complementarity row) against finite differences for second-order and
mixed orthant+SOC cones."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

from pounce.torch import solve_socp  # noqa: E402


def _fd(fn, x, eps=1e-6):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (float(fn(torch.tensor(xp))) - float(fn(torch.tensor(xm)))) / (2 * eps)
    return g


P3 = torch.eye(3)
G3 = -torch.eye(3)
H3 = torch.zeros(3)


def test_grad_c_soc_projection():
    def loss(c):
        return torch.sum(solve_socp(P=P3, c=c, G=G3, h=H3, cones=[("soc", 3)]) ** 2)

    c0 = torch.tensor([-1.0, -2.0, 0.3], requires_grad=True)
    loss(c0).backward()
    np.testing.assert_allclose(c0.grad.numpy(), _fd(loss, c0.detach()), atol=1e-4)


def test_grad_h_soc():
    c0 = torch.tensor([-1.0, -2.0, 0.3])

    def loss(h):
        return torch.sum(solve_socp(P=P3, c=c0, G=G3, h=h, cones=[3]) ** 2)

    h0 = torch.tensor([0.5, 0.0, 0.0], requires_grad=True)
    loss(h0).backward()
    np.testing.assert_allclose(h0.grad.numpy(), _fd(loss, h0.detach()), atol=1e-4)


def test_grad_c_and_b_soc_with_equality():
    A = torch.tensor([[1.0, 0.0, 0.0]])

    def loss_c(c):
        return torch.sum(
            solve_socp(P=P3, c=c, G=G3, h=H3, A=A, b=torch.tensor([0.5]), cones=[3]) ** 2
        )

    def loss_b(b):
        c0 = torch.tensor([0.0, -1.0, 0.0])
        return torch.sum(solve_socp(P=P3, c=c0, G=G3, h=H3, A=A, b=b, cones=[3]) ** 2)

    c0 = torch.tensor([0.0, -1.0, 0.0], requires_grad=True)
    loss_c(c0).backward()
    np.testing.assert_allclose(c0.grad.numpy(), _fd(loss_c, c0.detach()), atol=1e-4)

    b0 = torch.tensor([0.5], requires_grad=True)
    loss_b(b0).backward()
    np.testing.assert_allclose(b0.grad.numpy(), _fd(loss_b, b0.detach()), atol=1e-4)


def test_grad_mixed_orthant_and_soc():
    G = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, -1.0]])
    h = torch.tensor([1.0, 1.0, 0.0])

    def loss(c):
        return torch.sum(
            solve_socp(P=torch.eye(2), c=c, G=G, h=h,
                       cones=[("nonneg", 1), ("soc", 2)]) ** 2
        )

    c0 = torch.tensor([-0.5, -0.5], requires_grad=True)
    loss(c0).backward()
    np.testing.assert_allclose(c0.grad.numpy(), _fd(loss, c0.detach()), atol=1e-4)
