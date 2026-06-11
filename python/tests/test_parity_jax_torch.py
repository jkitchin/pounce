"""JAX ↔ Torch parity (pounce#109).

The same numerical core (Rust IPM) under two autodiff frontends must
produce the same ``x*`` and the same ``dL/dp`` on shared fixtures. This is
the headline "one numerical backbone under any autodiff frontend" claim.
Skipped unless *both* JAX and PyTorch are installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_parity_unconstrained_solve_and_grad():
    """min ||x - p||² + 0.1||x||⁴ : x* and dL/dp agree across frontends."""
    from pounce.jax import solve as jsolve
    from pounce.torch import solve as tsolve

    def fj(x, p):
        return jnp.sum((x - p) ** 2) + 0.1 * jnp.sum(x ** 4)

    def ft(x, p):
        return torch.sum((x - p) ** 2) + 0.1 * torch.sum(x ** 4)

    p_np = np.array([0.5, -1.0, 2.0, 0.3])
    opts = {"tol": 1e-11, "print_level": 0, "sb": "yes"}

    xj = np.asarray(jsolve(jnp.asarray(p_np), f=fj, g=None, x0=jnp.zeros(4),
                           n=4, m=0, options=opts))
    xt = tsolve(torch.tensor(p_np), f=ft, g=None, x0=torch.zeros(4),
                n=4, m=0, options=opts).detach().numpy()
    np.testing.assert_allclose(xj, xt, atol=1e-8)

    gj = np.asarray(jax.grad(
        lambda p: jnp.sum(jsolve(p, f=fj, g=None, x0=jnp.zeros(4), n=4, m=0,
                                 options=opts) ** 2)
    )(jnp.asarray(p_np)))

    pt_ = torch.tensor(p_np, requires_grad=True)
    (tsolve(pt_, f=ft, g=None, x0=torch.zeros(4), n=4, m=0, options=opts) ** 2).sum().backward()
    np.testing.assert_allclose(gj, pt_.grad.numpy(), atol=1e-6)


def test_parity_constrained_solve_and_grad():
    """Equality + slack inequality + active bound: full active-set logic."""
    from pounce.jax import solve as jsolve
    from pounce.torch import solve as tsolve

    def fj(x, p):
        return jnp.sum((x - p) ** 2) + 0.1 * jnp.sum(x ** 4)

    def gj(x, p):  # noqa: ARG001
        return jnp.stack([x[0] + x[1] + x[2] - 1.0, x[2]])

    def ft(x, p):
        return torch.sum((x - p) ** 2) + 0.1 * torch.sum(x ** 4)

    def gt(x, p):  # noqa: ARG001
        return torch.stack([x[0] + x[1] + x[2] - 1.0, x[2]])

    p_np = np.array([-0.2, 0.5, 0.4])
    common = dict(
        n=3, m=2,
        lb=np.array([0.4, -10.0, -10.0]), ub=np.array([10.0, 10.0, 10.0]),
        cl=np.array([0.0, -1e20]), cu=np.array([0.0, 1e20]),
        options={"tol": 1e-11, "print_level": 0, "sb": "yes"},
    )

    xj = np.asarray(jsolve(
        jnp.asarray(p_np), f=fj, g=gj, x0=jnp.ones(3),
        lb=jnp.asarray(common["lb"]), ub=jnp.asarray(common["ub"]),
        cl=jnp.asarray(common["cl"]), cu=jnp.asarray(common["cu"]),
        n=3, m=2, options=common["options"],
    ))
    xt = tsolve(
        torch.tensor(p_np), f=ft, g=gt, x0=torch.ones(3),
        lb=torch.tensor(common["lb"]), ub=torch.tensor(common["ub"]),
        cl=torch.tensor(common["cl"]), cu=torch.tensor(common["cu"]),
        n=3, m=2, options=common["options"],
    ).detach().numpy()
    np.testing.assert_allclose(xj, xt, atol=1e-7)

    gj_grad = np.asarray(jax.grad(
        lambda p: jnp.sum(jsolve(
            p, f=fj, g=gj, x0=jnp.ones(3),
            lb=jnp.asarray(common["lb"]), ub=jnp.asarray(common["ub"]),
            cl=jnp.asarray(common["cl"]), cu=jnp.asarray(common["cu"]),
            n=3, m=2, options=common["options"],
        ) ** 2)
    )(jnp.asarray(p_np)))

    pt_ = torch.tensor(p_np, requires_grad=True)
    (tsolve(
        pt_, f=ft, g=gt, x0=torch.ones(3),
        lb=torch.tensor(common["lb"]), ub=torch.tensor(common["ub"]),
        cl=torch.tensor(common["cl"]), cu=torch.tensor(common["cu"]),
        n=3, m=2, options=common["options"],
    ) ** 2).sum().backward()
    np.testing.assert_allclose(gj_grad, pt_.grad.numpy(), atol=1e-6)


def test_parity_qp_grad():
    """OptNet QP gradient w.r.t. c agrees across frontends."""
    from pounce.jax import solve_qp as jqp
    from pounce.torch import solve_qp as tqp

    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    G = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([10.0, 0.0, 0.0])
    c_np = np.array([-0.5, -0.7])

    gj = np.asarray(jax.grad(
        lambda c: jnp.sum(jqp(P=jnp.asarray(P), c=c, G=jnp.asarray(G),
                              h=jnp.asarray(h)) ** 2)
    )(jnp.asarray(c_np)))

    ct = torch.tensor(c_np, requires_grad=True)
    (tqp(P=torch.tensor(P), c=ct, G=torch.tensor(G), h=torch.tensor(h)) ** 2).sum().backward()
    np.testing.assert_allclose(gj, ct.grad.numpy(), atol=1e-6)


def test_parity_socp_grad():
    """Cone-aware OptNet SOCP gradient agrees across frontends."""
    from pounce.jax import solve_socp as jsocp
    from pounce.torch import solve_socp as tsocp

    P = np.eye(3)
    G = -np.eye(3)
    h = np.zeros(3)
    c_np = np.array([-1.0, -2.0, 0.3])

    gj = np.asarray(jax.grad(
        lambda c: jnp.sum(jsocp(P=jnp.asarray(P), c=c, G=jnp.asarray(G),
                                h=jnp.asarray(h), cones=[("soc", 3)]) ** 2)
    )(jnp.asarray(c_np)))

    ct = torch.tensor(c_np, requires_grad=True)
    (tsocp(P=torch.tensor(P), c=ct, G=torch.tensor(G), h=torch.tensor(h),
           cones=[("soc", 3)]) ** 2).sum().backward()
    np.testing.assert_allclose(gj, ct.grad.numpy(), atol=1e-6)
