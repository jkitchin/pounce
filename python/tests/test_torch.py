"""Tests for the PyTorch integration (pounce#109). Skipped when PyTorch
isn't installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)


# ----- from_torch build -----


def test_from_torch_hs071():
    from pounce.torch import from_torch

    def f(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def g(x):
        return torch.stack([torch.prod(x), torch.dot(x, x)])

    prob = from_torch(
        f, g, n=4, m=2,
        lb=np.array([1.0] * 4), ub=np.array([5.0] * 4),
        cl=np.array([25.0, 40.0]), cu=np.array([2e19, 40.0]),
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(info["obj_val"], 17.0140172, rtol=1e-5)


def _banded_problem(n):
    def f(x):
        return torch.sum(x ** 2) + torch.sum(torch.sin(x))

    def g(x):
        return x[:-1] * x[1:] - 1.0

    return f, g, n, n - 1


def test_from_torch_sparse_matches_dense_pounce_83():
    from pounce.torch._build import _TorchProblem

    def f_hs(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def g_hs(x):
        return torch.stack([torch.prod(x), torch.dot(x, x)])

    cases = [(f_hs, g_hs, 4, 2), _banded_problem(60)]
    for f, g, n, m in cases:
        dense = _TorchProblem(f, g, n=n, m=m, sparse=False)
        sparse = _TorchProblem(f, g, n=n, m=m, sparse=True, n_probes=3)

        for a, b in zip(dense.jacobianstructure(), sparse.jacobianstructure()):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(dense.hessianstructure(), sparse.hessianstructure()):
            np.testing.assert_array_equal(a, b)

        rng = np.random.default_rng(11)
        for _ in range(3):
            x = rng.standard_normal(n)
            lam = rng.standard_normal(m)
            np.testing.assert_allclose(
                dense.jacobian(x), sparse.jacobian(x), rtol=1e-12, atol=1e-12,
            )
            np.testing.assert_allclose(
                dense.hessian(x, lam, 0.7), sparse.hessian(x, lam, 0.7),
                rtol=1e-12, atol=1e-12,
            )


def test_from_torch_sparse_solves_match_dense_pounce_83():
    from pounce.torch import from_torch

    def f(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def g(x):
        return torch.stack([torch.prod(x), torch.dot(x, x)])

    results = {}
    for sparse in (False, True):
        prob = from_torch(
            f, g, n=4, m=2,
            lb=np.array([1.0] * 4), ub=np.array([5.0] * 4),
            cl=np.array([25.0, 40.0]), cu=np.array([2e19, 40.0]),
            sparse=sparse,
        )
        prob.add_option("tol", 1e-8)
        prob.add_option("print_level", 0)
        x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
        assert info["status_msg"] == "Solve_Succeeded"
        results[sparse] = (np.asarray(x), info["obj_val"])

    np.testing.assert_allclose(results[True][0], results[False][0], atol=1e-7)
    np.testing.assert_allclose(results[True][1], results[False][1], rtol=1e-9)


# ----- implicit diff (top-level solve) -----


def test_implicit_diff_parametric():
    """min ||x - p||² → x*(p) = p, dL/dp = 2p."""
    from pounce.torch import solve

    def f(x, p):
        d = x - p
        return torch.dot(d, d)

    p = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    x_star = solve(p, f=f, g=None, x0=torch.zeros(3), n=3, m=0,
                   options={"tol": 1e-10, "print_level": 0})
    (x_star ** 2).sum().backward()
    np.testing.assert_allclose(p.grad.numpy(), (2.0 * p.detach()).numpy(), atol=1e-4)


def _solve_box_projection(p, *, n=3, B=0.5):
    from pounce.torch import solve

    def f(x, p_):
        d = x - p_
        return torch.dot(d, d)

    def g(x, p_):  # noqa: ARG001
        return torch.stack([x[0]])

    return solve(
        p, f=f, g=g, x0=torch.zeros(n), n=n, m=1,
        lb=torch.full((n,), -1e19), ub=torch.full((n,), 1e19),
        cl=torch.tensor([-1e19]), cu=torch.tensor([B]),
        options={"tol": 1e-10, "print_level": 0},
    )


def _fd_jac(forward, p, eps=1e-6):
    p_np = np.asarray(p.detach(), dtype=np.float64)
    n_out = forward(torch.tensor(p_np)).numel()
    J = np.zeros((n_out, p_np.size))
    for j in range(p_np.size):
        e = np.zeros_like(p_np)
        e[j] = eps
        J[:, j] = (
            forward(torch.tensor(p_np + e)).detach().numpy()
            - forward(torch.tensor(p_np - e)).detach().numpy()
        ) / (2.0 * eps)
    return J


def test_implicit_diff_inactive_inequality_pounce_73():
    """Slack inequality must not pin the sensitivity: dx*/dp = I."""
    p = torch.tensor([-1.0, 2.0, -3.0])
    analytic = torch.autograd.functional.jacobian(_solve_box_projection, p)
    fd = _fd_jac(_solve_box_projection, p)
    np.testing.assert_allclose(analytic.numpy(), fd, atol=5e-6)
    np.testing.assert_allclose(analytic.numpy(), np.eye(3), atol=5e-6)


def test_implicit_diff_active_inequality_pounce_73():
    """When the inequality binds, dx*/dp must still match FD."""
    p = torch.tensor([2.0, 2.0, -3.0])
    analytic = torch.autograd.functional.jacobian(_solve_box_projection, p)
    fd = _fd_jac(_solve_box_projection, p)
    np.testing.assert_allclose(analytic.numpy(), fd, atol=5e-6)


def test_solve_gradcheck():
    """torch.autograd.gradcheck on the constrained solve (float64)."""
    from pounce.torch import solve

    def f(x, p):
        return torch.sum((x - p) ** 2) + 0.1 * torch.sum(x ** 4)

    def g(x, p):  # noqa: ARG001
        return torch.stack([x[0] + x[1] + x[2] - 1.0])

    def fn(p):
        return solve(
            p, f=f, g=g, x0=torch.ones(3), n=3, m=1,
            lb=torch.full((3,), -5.0), ub=torch.full((3,), 5.0),
            cl=torch.zeros(1), cu=torch.zeros(1),
            options={"tol": 1e-11, "print_level": 0, "sb": "yes"},
        )

    p = torch.tensor([-0.2, 0.5, 0.4], requires_grad=True)
    assert torch.autograd.gradcheck(fn, (p,), atol=1e-4, rtol=1e-3, eps=1e-6)


# ----- warm start -----


def test_solve_with_warm_pounce_74():
    from pounce.torch import solve_with_warm

    n, m, B = 3, 1, 0.5

    def f(x, p):
        d = x - p
        return torch.dot(d, d)

    def g(x, p):  # noqa: ARG001
        return torch.stack([x[0]])

    def forward(p, warm):
        return solve_with_warm(
            p, f=f, g=g, x0=torch.zeros(n), n=n, m=m,
            lb=torch.full((n,), -1e19), ub=torch.full((n,), 1e19),
            cl=torch.tensor([-1e19]), cu=torch.tensor([B]),
            options={"tol": 1e-10, "print_level": 0},
            warm_start=warm,
        )

    p0 = torch.tensor([2.0, 2.0, -3.0])
    x0_star, warm0 = forward(p0, warm=None)
    np.testing.assert_allclose(x0_star.detach().numpy(), [B, 2.0, -3.0], atol=1e-6)

    x1_star, (lam1, zL1, zU1) = forward(p0, warm=warm0)
    np.testing.assert_allclose(x1_star.detach().numpy(), x0_star.detach().numpy(), atol=1e-8)
    assert torch.all(torch.isfinite(lam1))

    p_var = torch.tensor([2.0, 2.0, -3.0], requires_grad=True)
    x_star, _ = forward(p_var, warm=warm0)
    (x_star ** 2).sum().backward()
    np.testing.assert_allclose(p_var.grad.numpy(), [0.0, 4.0, -6.0], atol=1e-6)


def test_solve_with_warm_threads_mu_pounce_86():
    from pounce.torch import solve_with_warm

    n, m, B = 3, 1, 0.5

    def f(x, p):
        d = x - p
        return torch.dot(d, d)

    def g(x, p):  # noqa: ARG001
        return torch.stack([x[0]])

    def forward(p, warm):
        return solve_with_warm(
            p, f=f, g=g, x0=torch.zeros(n), n=n, m=m,
            lb=torch.full((n,), -1e19), ub=torch.full((n,), 1e19),
            cl=torch.tensor([-1e19]), cu=torch.tensor([B]),
            options={"tol": 1e-10, "print_level": 0},
            warm_start=warm,
        )

    p0 = torch.tensor([2.0, 2.0, -3.0])
    _x, warm3 = forward(p0, warm=None)
    assert len(warm3) == 3

    x_ro, warm_ro = forward(p0, warm=(*warm3, None))
    assert len(warm_ro) == 4
    mu_out = float(warm_ro[3])
    assert np.isfinite(mu_out) and 0.0 < mu_out < 1e-6

    lam, zL, zU = warm3
    x_seed, warm_seed = forward(p0, warm=(lam, zL, zU, mu_out))
    assert len(warm_seed) == 4
    np.testing.assert_allclose(x_seed.detach().numpy(), x_ro.detach().numpy(), atol=1e-8)


# ----- batched -----


def test_vmap_solve_parallel_matches_serial_pounce_74():
    from pounce.torch import solve as serial_solve
    from pounce.torch import vmap_solve_parallel

    n, B = 3, 4

    def f(x, p):
        d = x - p
        return torch.dot(d, d)

    rng = np.random.default_rng(0)
    p_batch = torch.tensor(rng.standard_normal((B, n)))
    x0 = torch.zeros(n)

    x_parallel = vmap_solve_parallel(
        p_batch, f=f, g=None, x0=x0, n=n, m=0,
        options={"tol": 1e-10, "print_level": 0}, workers=4,
    )
    x_serial = torch.stack([
        serial_solve(p_batch[i], f=f, g=None, x0=x0, n=n, m=0,
                     options={"tol": 1e-10, "print_level": 0})
        for i in range(B)
    ])
    np.testing.assert_allclose(x_parallel.detach().numpy(), x_serial.detach().numpy(), atol=1e-7)
    np.testing.assert_allclose(x_parallel.detach().numpy(), p_batch.numpy(), atol=1e-7)

    pb = p_batch.clone().requires_grad_(True)
    xb = vmap_solve_parallel(
        pb, f=f, g=None, x0=x0, n=n, m=0,
        options={"tol": 1e-10, "print_level": 0}, workers=4,
    )
    (xb ** 2).sum().backward()
    np.testing.assert_allclose(pb.grad.numpy(), 2.0 * p_batch.numpy(), atol=1e-6)


# ----- TorchProblem (build-once) -----


def test_torch_problem_build_once_pounce_75():
    from pounce.torch import TorchProblem, solve as top_solve

    n, m, B = 3, 1, 0.5

    def f(x, p):
        d = x - p
        return torch.dot(d, d)

    def g(x, p):  # noqa: ARG001
        return torch.stack([x[0]])

    tp = TorchProblem(
        f=f, g=g, n=n, m=m, p_example=torch.zeros(n),
        lb=torch.full((n,), -1e19), ub=torch.full((n,), 1e19),
        cl=torch.tensor([-1e19]), cu=torch.tensor([B]),
        options={"tol": 1e-10, "print_level": 0},
    )
    for p in (torch.tensor([2.0, 2.0, -3.0]), torch.tensor([-1.0, 2.0, -3.0]),
              torch.tensor([0.3, -0.5, 0.7])):
        x_reuse = tp.solve(p, torch.zeros(n))
        x_ref = top_solve(
            p, f=f, g=g, x0=torch.zeros(n), n=n, m=m,
            lb=torch.full((n,), -1e19), ub=torch.full((n,), 1e19),
            cl=torch.tensor([-1e19]), cu=torch.tensor([B]),
            options={"tol": 1e-10, "print_level": 0},
        )
        np.testing.assert_allclose(x_reuse.detach().numpy(), x_ref.detach().numpy(), atol=1e-8)


def test_torch_problem_grad_pounce_75():
    from pounce.torch import TorchProblem

    n, m, B = 3, 1, 0.5

    def f(x, p):
        d = x - p
        return torch.dot(d, d)

    def g(x, p):  # noqa: ARG001
        return torch.stack([x[0]])

    tp = TorchProblem(
        f=f, g=g, n=n, m=m, p_example=torch.zeros(n),
        lb=torch.full((n,), -1e19), ub=torch.full((n,), 1e19),
        cl=torch.tensor([-1e19]), cu=torch.tensor([B]),
        options={"tol": 1e-10, "print_level": 0},
    )
    p0 = torch.tensor([-1.0, 2.0, -3.0])
    J = torch.autograd.functional.jacobian(lambda p: tp.solve(p, torch.zeros(n)), p0)
    np.testing.assert_allclose(J.numpy(), np.eye(n), atol=5e-6)


def _warm_problem():
    """min ½‖x−p‖² s.t. Σx = 1, −10 ≤ x ≤ 10 — x*(p) projects p onto the
    simplex hyperplane, so ∂x*/∂p = I − 11ᵀ/n and the equality multiplier
    is λ = (Σp − 1)/n."""
    n, m = 3, 1

    def f(x, p):
        d = x - p
        return 0.5 * torch.dot(d, d)

    def g(x, p):  # noqa: ARG001
        return torch.stack([torch.sum(x) - 1.0])

    from pounce.torch import TorchProblem

    tp = TorchProblem(
        f=f, g=g, n=n, m=m, p_example=torch.zeros(n),
        lb=torch.full((n,), -10.0), ub=torch.full((n,), 10.0),
        cl=torch.tensor([0.0]), cu=torch.tensor([0.0]),
        options={"tol": 1e-10, "print_level": 0},
    )
    return tp, n, m


def test_torch_problem_warm_single_solve_and_grad():
    """`TorchProblem.solve_with_warm` / `batched_solve_with_warm` must run
    the IPM exactly once per call (no redundant dual-recovery solve) and
    stay correctly differentiable w.r.t. p (duals threaded out, not
    differentiated)."""
    tp, n, m = _warm_problem()
    from pounce.torch import TorchProblem

    calls = {"single": 0, "batched": 0}
    orig_s = TorchProblem._host_solve_warm
    orig_b = TorchProblem._host_batched_solve_warm
    TorchProblem._host_solve_warm = lambda self, *a, **k: (
        calls.__setitem__("single", calls["single"] + 1) or orig_s(self, *a, **k)
    )
    TorchProblem._host_batched_solve_warm = lambda self, *a, **k: (
        calls.__setitem__("batched", calls["batched"] + 1) or orig_b(self, *a, **k)
    )
    try:
        x0 = torch.zeros(n)
        p = torch.tensor([0.2, 0.5, 0.9], requires_grad=True)
        x_star, (lam, zL, zU) = tp.solve_with_warm(p, x0)
        # x* lies on the equality hyperplane and the duals come back.
        np.testing.assert_allclose(float(x_star.sum()), 1.0, atol=1e-8)
        assert lam.shape == (m,) and zL.shape == (n,) and zU.shape == (n,)
        assert not lam.requires_grad  # duals are non-differentiable outputs
        (x_star ** 2).sum().backward()
        # ∂x*/∂p = I − 11ᵀ/n  ⇒  grad of Σx*² is 2 (I − 11ᵀ/n) x*.
        xs = x_star.detach()
        expected = 2.0 * (xs - xs.mean())
        np.testing.assert_allclose(p.grad.numpy(), expected.numpy(), atol=1e-6)
        assert calls["single"] == 1, f"expected one solve, got {calls['single']}"

        pb = torch.tensor([[0.2, 0.5, 0.9], [0.1, 0.1, 0.1]], requires_grad=True)
        xb, (lamb, zLb, zUb) = tp.batched_solve_with_warm(pb, x0)
        np.testing.assert_allclose(xb.sum(dim=1).detach().numpy(), [1.0, 1.0], atol=1e-8)
        assert not lamb.requires_grad
        (xb ** 2).sum().backward()
        assert torch.isfinite(pb.grad).all()
        assert calls["batched"] == 1, f"expected one batched solve, got {calls['batched']}"
    finally:
        TorchProblem._host_solve_warm = orig_s
        TorchProblem._host_batched_solve_warm = orig_b


def test_torch_problem_rejects_float32_param():
    """The differentiable TorchProblem entry points reject an explicit
    float32 parameter (precision-losing) the same way the top-level
    `pounce.torch.solve` does, while still accepting lists / NumPy /
    float64 tensors."""
    tp, n, _ = _warm_problem()
    x0 = torch.zeros(n)
    with pytest.raises(TypeError, match="float64"):
        tp.solve(torch.tensor([0.2, 0.5, 0.9], dtype=torch.float32), x0)
    with pytest.raises(TypeError, match="float64"):
        tp.vmap_solve(torch.tensor([[0.2, 0.5, 0.9]], dtype=torch.float32), x0)
    # Non-tensor and float64 inputs are accepted (coerced) without error.
    np.testing.assert_allclose(
        float(tp.solve([0.2, 0.5, 0.9], x0).sum()), 1.0, atol=1e-8,
    )
    np.testing.assert_allclose(
        float(tp.solve(np.array([0.2, 0.5, 0.9]), x0).sum()), 1.0, atol=1e-8,
    )


def test_factor_reuse_matches_dense_pounce_76():
    """The k_aug-style factor-reuse backward must agree with the dense
    KKT backward across equality, slack-inequality, and active-bound."""
    from pounce.torch import TorchProblem

    def f(x, p):
        return torch.sum((x - p) ** 2) + 0.1 * torch.sum(x ** 4)

    def g(x, p):  # noqa: ARG001
        return torch.stack([x[0] + x[1] + x[2] - 1.0, x[2]])

    kwargs = dict(
        f=f, g=g, n=3, m=2, p_example=torch.zeros(3),
        lb=torch.tensor([0.4, -10.0, -10.0]), ub=torch.full((3,), 10.0),
        cl=torch.tensor([0.0, -1e20]), cu=torch.tensor([0.0, 1e20]),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )
    tp_new = TorchProblem(**kwargs, factor_reuse=True)
    tp_old = TorchProblem(**kwargs, factor_reuse=False)

    p = torch.tensor([-0.2, 0.5, 0.4], requires_grad=True)

    def loss(tp, p):
        return (tp.solve(p, torch.ones(3)) ** 2).sum()

    pn = p.clone().detach().requires_grad_(True)
    loss(tp_new, pn).backward()
    po = p.clone().detach().requires_grad_(True)
    loss(tp_old, po).backward()
    np.testing.assert_allclose(pn.grad.numpy(), po.grad.numpy(), atol=1e-7)


def test_factor_reuse_jacobian_pounce_76():
    from pounce.torch import TorchProblem

    def f(x, p):
        return torch.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return torch.stack([x[0] - x[1]])

    tp = TorchProblem(
        f=f, g=g, n=2, m=1, p_example=torch.zeros(2),
        lb=torch.full((2,), -1e19), ub=torch.full((2,), 1e19),
        cl=torch.zeros(1), cu=torch.zeros(1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )
    p = torch.tensor([0.3, 0.7])
    J = torch.autograd.functional.jacobian(lambda p: tp.solve(p, torch.zeros(2)), p)
    np.testing.assert_allclose(J.numpy(), 0.5 * np.ones((2, 2)), atol=1e-6)


def test_batched_solve_matches_parallel_pounce_76():
    from pounce.torch import TorchProblem

    def f(x, p):
        return torch.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return torch.stack([x[0] + x[1] - 1.0])

    tp = TorchProblem(
        f=f, g=g, n=2, m=1, p_example=torch.zeros(2),
        lb=torch.full((2,), -10.0), ub=torch.full((2,), 10.0),
        cl=torch.zeros(1), cu=torch.zeros(1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )
    p_batch = torch.tensor([[0.3, 0.7], [0.5, 0.5], [-0.1, 0.4], [1.0, 2.0]])
    x_b = tp.batched_solve(p_batch, x0=torch.zeros(2))
    x_p = tp.vmap_solve_parallel(p_batch, x0=torch.zeros(2), workers=2)
    np.testing.assert_allclose(x_b.detach().numpy(), x_p.detach().numpy(), atol=1e-8)


def test_batched_solve_grad_factor_reuse_pounce_76():
    from pounce.torch import TorchProblem

    def f(x, p):
        return torch.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return torch.stack([x[0] + x[1] - 1.0])

    def build(reuse):
        return TorchProblem(
            f=f, g=g, n=2, m=1, p_example=torch.zeros(2),
            lb=torch.full((2,), -10.0), ub=torch.full((2,), 10.0),
            cl=torch.zeros(1), cu=torch.zeros(1),
            options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
            factor_reuse=reuse,
        )

    tp_reuse, tp_dense = build(True), build(False)
    p_batch = torch.tensor([[0.3, 0.7], [0.5, 0.5], [-0.1, 0.4], [0.8, -0.2]])

    pr = p_batch.clone().requires_grad_(True)
    (tp_reuse.batched_solve(pr, x0=torch.zeros(2)) ** 2).sum().backward()
    pd = p_batch.clone().requires_grad_(True)
    (tp_dense.batched_solve(pd, x0=torch.zeros(2)) ** 2).sum().backward()
    np.testing.assert_allclose(pr.grad.numpy(), pd.grad.numpy(), atol=1e-7)


# ----- sensitivity / anchor / path-following -----


def test_solve_with_jacobian_and_sensitivity():
    from pounce.torch import TorchProblem

    def f(x, p):
        return torch.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return torch.stack([x[0] + x[1] - 1.0])

    tp = TorchProblem(
        f=f, g=g, n=2, m=1, p_example=torch.zeros(2),
        lb=torch.full((2,), -10.0), ub=torch.full((2,), 10.0),
        cl=torch.zeros(1), cu=torch.zeros(1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )
    p = torch.tensor([0.3, 0.7])
    x_star, duals, J = tp.solve_with_jacobian(p, torch.zeros(2))
    # Project p onto x0+x1=1: dx*/dp = I - 0.5 * 11^T.
    expected = np.eye(2) - 0.5 * np.ones((2, 2))
    np.testing.assert_allclose(J.numpy(), expected, atol=1e-6)

    with tp.anchor(p, torch.zeros(2)) as state:
        # JVP against held factor matches J @ dp.
        dp = torch.tensor([1.0, -2.0])
        dx = tp.jvp_from_state(state, dp)
        np.testing.assert_allclose(dx.numpy(), (J @ dp).numpy(), atol=1e-7)
        # VJP against held factor matches J^T @ x_bar.
        xbar = torch.tensor([0.5, 1.5])
        dpar = tp.vjp_from_state(state, xbar)
        np.testing.assert_allclose(dpar.numpy(), (J.T @ xbar).numpy(), atol=1e-7)


def test_path_follower_linear_map():
    """Trace x*(θ) for min ||x - θ||² along θ(s) = s·[1,1]; x* = θ."""
    from pounce.torch import TorchProblem, PathFollower

    def f(x, p):
        return torch.sum((x - p) ** 2)

    tp = TorchProblem(
        f=f, g=None, n=2, m=0, p_example=torch.zeros(2),
        lb=torch.full((2,), -10.0), ub=torch.full((2,), 10.0),
        options={"tol": 1e-11, "print_level": 0, "sb": "yes"},
    )
    pf = PathFollower(tp, monitor_tol=1e-6)
    trace = pf.follow(lambda s: torch.tensor([s, s]), (0.0, 1.0), torch.zeros(2))
    assert trace.status == "ok"
    # Final point: x* = θ(1) = [1, 1].
    np.testing.assert_allclose(trace.x[-1], [1.0, 1.0], atol=1e-5)


def test_trace_arclength_fold():
    """Pseudo-arclength continuation traces past a fold. Stationarity of
    f(x, θ) = x⁴/4 − x²/2 − θ x gives R = x³ − x − θ = 0, an S-curve with
    folds at x = ±1/√3 (θ = ∓2/(3√3))."""
    from pounce.torch import TorchProblem, PathFollower

    def f(x, p):
        return 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2 - p[0] * x[0]

    tp = TorchProblem(
        f=f, g=None, n=1, m=0, p_example=torch.zeros(1),
        options={"tol": 1e-11, "print_level": 0, "sb": "yes"},
    )
    pf = PathFollower(tp)
    trace = pf.trace_arclength(
        x0=torch.tensor([-1.2]), theta0=-0.5, ds=0.05, n_steps=120, direction=1.0,
    )
    assert trace.status == "ok"
    # Two folds expected near θ = ±2/(3√3) ≈ ±0.385.
    assert len(trace.turning_points) >= 2
    np.testing.assert_allclose(
        sorted(abs(t) for t in trace.turning_points[:2]),
        [2.0 / (3 * np.sqrt(3))] * 2, atol=2e-2,
    )


def test_inverse_map_rhs_identity():
    """Identity output: dθ/ds = J^{-1} dy/ds; for x*=θ, J=I so dθ/ds=dy/ds."""
    from pounce.torch import TorchProblem, inverse_map_rhs

    def f(x, p):
        return torch.sum((x - p) ** 2)

    tp = TorchProblem(
        f=f, g=None, n=2, m=0, p_example=torch.zeros(2),
        lb=torch.full((2,), -10.0), ub=torch.full((2,), 10.0),
        options={"tol": 1e-11, "print_level": 0, "sb": "yes"},
    )
    rhs = inverse_map_rhs(tp, dy_ds=torch.tensor([1.0, -1.0]))
    out = rhs(0.0, torch.tensor([0.5, 0.5]))
    np.testing.assert_allclose(out.numpy(), [1.0, -1.0], atol=1e-6)
