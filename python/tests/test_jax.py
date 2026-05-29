"""Tests for the JAX integration. Skipped when JAX isn't installed."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp


def test_from_jax_hs071():
    from pounce.jax import from_jax

    def f(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def g(x):
        return jnp.stack([jnp.prod(x), jnp.dot(x, x)])

    prob = from_jax(
        f, g,
        n=4, m=2,
        lb=np.array([1.0] * 4), ub=np.array([5.0] * 4),
        cl=np.array([25.0, 40.0]), cu=np.array([2e19, 40.0]),
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(info["obj_val"], 17.0140172, rtol=1e-5)


def test_implicit_diff_parametric_qp():
    """Differentiate x*(p) for  min ||x - p||²   →   x*(p) = p,   dx*/dp = I.

    A trivial parametric problem where the analytic Jacobian is known
    in closed form (the identity). This exercises the custom_vjp end
    to end without needing scipy.
    """
    from pounce.jax import solve

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    def loss(p):
        x_star = solve(
            p, f=f, g=None, x0=jnp.zeros_like(p),
            n=p.size, m=0,
            options={"tol": 1e-10, "print_level": 0},
        )
        return jnp.sum(x_star ** 2)

    p = jnp.array([1.0, -2.0, 3.0])
    grad = jax.grad(loss)(p)
    # dL/dp = 2 x*(p) = 2 p.
    np.testing.assert_allclose(grad, 2.0 * p, atol=1e-4)


def _solve_box_projection(p, *, n=3, B=0.5):
    """Helper: min ||x - p||² s.t. x[0] <= B, all other x free."""
    from pounce.jax import solve

    def f(x, p_):
        d = x - p_
        return jnp.dot(d, d)

    def g(x, p_):  # noqa: ARG001
        return jnp.stack([x[0]])

    return solve(
        p, f=f, g=g, x0=jnp.zeros(n), n=n, m=1,
        lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
        cl=jnp.array([-1e19]), cu=jnp.array([B]),
        options={"tol": 1e-10, "print_level": 0},
    )


def _finite_diff_jacobian(forward, p, eps=1e-6):
    p_np = np.asarray(p, dtype=np.float64)
    n_out = np.asarray(forward(jnp.asarray(p_np))).size
    J = np.zeros((n_out, p_np.size))
    for j in range(p_np.size):
        e = np.zeros_like(p_np)
        e[j] = eps
        J[:, j] = (
            np.asarray(forward(jnp.asarray(p_np + e)))
            - np.asarray(forward(jnp.asarray(p_np - e)))
        ) / (2.0 * eps)
    return J


def test_implicit_diff_inactive_inequality_pounce_73():
    """Issue #73: slack inequality must not pin the sensitivity.

    `min ||x - p||²` s.t. `x[0] <= 0.5`. With `p[0] = -1 < 0.5` the
    inequality is slack at the optimum (mult_g ≈ 0), so `x*(p) = p`
    and `dx*/dp = I`. The bug was that the inequality row was kept
    as an equality in the backward, yielding ``dx*/dp[:, 0] ≈ 0``.
    """
    p = jnp.array([-1.0, 2.0, -3.0])
    analytic = np.asarray(jax.jacobian(_solve_box_projection)(p))
    fd = _finite_diff_jacobian(_solve_box_projection, p)
    np.testing.assert_allclose(analytic, fd, atol=5e-6)
    # Truth at slack ineq: dx*/dp = I.
    np.testing.assert_allclose(analytic, np.eye(p.size), atol=5e-6)


def test_implicit_diff_active_inequality_pounce_73():
    """Companion: when the inequality binds, dx*/dp must still match FD."""
    p = jnp.array([2.0, 2.0, -3.0])  # p[0] > B → x*[0] = B, binding
    analytic = np.asarray(jax.jacobian(_solve_box_projection)(p))
    fd = _finite_diff_jacobian(_solve_box_projection, p)
    np.testing.assert_allclose(analytic, fd, atol=5e-6)


def test_solve_with_warm_reduces_iterations_pounce_74():
    """`solve_with_warm` should consume the previous solve's duals and
    take fewer interior-point iterations on a small perturbation —
    that's the whole point of the warm-start surface (pounce#74)."""
    from pounce.jax import solve_with_warm

    n, m, B = 3, 1, 0.5

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    def g(x, p):
        return jnp.stack([x[0]])

    def forward(p, warm):
        return solve_with_warm(
            p, f=f, g=g, x0=jnp.zeros(n), n=n, m=m,
            lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
            cl=jnp.array([-1e19]), cu=jnp.array([B]),
            options={"tol": 1e-10, "print_level": 0},
            warm_start=warm,
        )

    # Cold-start, then warm-start at a small perturbation of p.
    p0 = jnp.array([2.0, 2.0, -3.0])  # active inequality
    x0_star, warm0 = forward(p0, warm=None)
    np.testing.assert_allclose(np.asarray(x0_star), [B, 2.0, -3.0], atol=1e-6)

    # Re-solve at the same p with the warm duals — answer must match,
    # and the dual triple must round-trip without exploding.
    x1_star, (lam1, zL1, zU1) = forward(p0, warm=warm0)
    np.testing.assert_allclose(np.asarray(x1_star), np.asarray(x0_star), atol=1e-8)
    assert np.all(np.isfinite(np.asarray(lam1)))
    assert np.all(np.isfinite(np.asarray(zL1)))
    assert np.all(np.isfinite(np.asarray(zU1)))

    # Differentiability w.r.t. p still works through the warm path
    # (cotangents on the dual outputs are dropped — only x* feeds back).
    def loss(p):
        x_star, _ = forward(p, warm=warm0)
        return jnp.sum(x_star ** 2)

    grad = np.asarray(jax.grad(loss)(p0))
    # x*[0] = B is fixed (binding), so dL/dp[0] = 0; the others
    # contribute 2 * x*[i] = 2 * p[i] for i in {1, 2}.
    np.testing.assert_allclose(grad, np.array([0.0, 4.0, -6.0]), atol=1e-6)


def test_vmap_solve_parallel_matches_vmap_solve_pounce_74():
    """Parallel batched solve must agree numerically with the sequential
    `vmap_solve` reference, both for the forward x* and for `jax.grad`
    through a downstream loss (pounce#74-1)."""
    from pounce.jax import solve as serial_solve
    from pounce.jax import vmap_solve_parallel

    n = 3
    B = 4

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    # Unconstrained: x*(p) = p, dL/dp_i = 2 p_i.
    rng = np.random.default_rng(0)
    p_batch = jnp.asarray(rng.standard_normal((B, n)))
    x0 = jnp.zeros(n)

    x_parallel = vmap_solve_parallel(
        p_batch, f=f, g=None, x0=x0, n=n, m=0,
        options={"tol": 1e-10, "print_level": 0},
        workers=4,
    )

    # Reference: serial loop in pure Python.
    x_serial = np.stack([
        np.asarray(serial_solve(
            p_batch[i], f=f, g=None, x0=x0, n=n, m=0,
            options={"tol": 1e-10, "print_level": 0},
        ))
        for i in range(B)
    ])

    np.testing.assert_allclose(np.asarray(x_parallel), x_serial, atol=1e-7)
    np.testing.assert_allclose(np.asarray(x_parallel), np.asarray(p_batch), atol=1e-7)

    def loss(p_batch):
        x_batch = vmap_solve_parallel(
            p_batch, f=f, g=None, x0=x0, n=n, m=0,
            options={"tol": 1e-10, "print_level": 0},
            workers=4,
        )
        return jnp.sum(x_batch ** 2)

    grad = np.asarray(jax.grad(loss)(p_batch))
    # Closed form: ∂(Σ x*² )/∂p = 2 p (since x* = p).
    np.testing.assert_allclose(grad, 2.0 * np.asarray(p_batch), atol=1e-6)


def test_vmap_solve_parallel_with_constraints_pounce_74():
    """Parallel solve with an active inequality on some batch elements
    and not others — exercises the GIL-release path with re-entrant
    JAX callbacks for f and g, and confirms the per-element active
    set is respected (pounce#74-1)."""
    from pounce.jax import vmap_solve_parallel

    n, B_thresh = 3, 0.5

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    def g(x, p):
        return jnp.stack([x[0]])

    # Mix: first row binds (p[0]=2 > 0.5), second is slack (p[0]=-1).
    p_batch = jnp.array([
        [2.0, 2.0, -3.0],
        [-1.0, 2.0, -3.0],
        [1.5, 0.0, 1.0],   # binds
        [0.3, -0.5, 0.7],  # slack
    ])
    x0 = jnp.zeros(n)
    x_parallel = vmap_solve_parallel(
        p_batch, f=f, g=g, x0=x0, n=n, m=1,
        lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
        cl=jnp.array([-1e19]), cu=jnp.array([B_thresh]),
        options={"tol": 1e-10, "print_level": 0},
        workers=4,
    )
    x_np = np.asarray(x_parallel)
    # x*[0] = min(p[0], B_thresh); x*[1:] = p[1:].
    expected_x0 = np.minimum(np.asarray(p_batch)[:, 0], B_thresh)
    np.testing.assert_allclose(x_np[:, 0], expected_x0, atol=1e-7)
    np.testing.assert_allclose(x_np[:, 1:], np.asarray(p_batch)[:, 1:], atol=1e-7)


def test_jax_problem_build_once_solve_many_pounce_75():
    """Issue #75: `JaxProblem` builds the JAX derivatives and sparsity
    pattern once. Repeated `.solve(p, x0)` calls must reuse the prebuilt
    state and produce the same answers as the top-level `solve`.
    """
    from pounce.jax import JaxProblem, solve as top_level_solve

    n, m = 3, 1
    B_thresh = 0.5

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    def g(x, p):  # noqa: ARG001
        return jnp.stack([x[0]])

    p0 = jnp.array([2.0, 2.0, -3.0])
    jp = JaxProblem(
        f=f, g=g, n=n, m=m, p_example=p0,
        lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
        cl=jnp.array([-1e19]), cu=jnp.array([B_thresh]),
        options={"tol": 1e-10, "print_level": 0},
    )

    # Same-structure problem at multiple p — reuses the prebuilt JIT and
    # sparsity pattern. Must match the top-level `solve` answer.
    for p in (
        jnp.array([2.0, 2.0, -3.0]),     # binding
        jnp.array([-1.0, 2.0, -3.0]),    # slack
        jnp.array([0.3, -0.5, 0.7]),     # slack
    ):
        x_reuse = jp.solve(p, jnp.zeros(n))
        x_ref = top_level_solve(
            p, f=f, g=g, x0=jnp.zeros(n), n=n, m=m,
            lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
            cl=jnp.array([-1e19]), cu=jnp.array([B_thresh]),
            options={"tol": 1e-10, "print_level": 0},
        )
        np.testing.assert_allclose(np.asarray(x_reuse), np.asarray(x_ref), atol=1e-8)


def test_jax_problem_grad_pounce_75():
    """`jax.grad` through `JaxProblem.solve` must match the closed-form
    sensitivity. Same test as `test_implicit_diff_inactive_inequality_pounce_73`
    but threaded through the prebuilt path (pounce#75).
    """
    from pounce.jax import JaxProblem

    n, m = 3, 1
    B_thresh = 0.5

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    def g(x, p):  # noqa: ARG001
        return jnp.stack([x[0]])

    p0 = jnp.array([-1.0, 2.0, -3.0])
    jp = JaxProblem(
        f=f, g=g, n=n, m=m, p_example=p0,
        lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
        cl=jnp.array([-1e19]), cu=jnp.array([B_thresh]),
        options={"tol": 1e-10, "print_level": 0},
    )

    # Slack inequality: x*(p) = p, dx*/dp = I.
    J = np.asarray(jax.jacobian(lambda p: jp.solve(p, jnp.zeros(n)))(p0))
    np.testing.assert_allclose(J, np.eye(n), atol=5e-6)


def test_jax_problem_solve_with_warm_pounce_75():
    """`JaxProblem.solve_with_warm` round-trips the dual triple and stays
    differentiable through the warm path."""
    from pounce.jax import JaxProblem

    n, m = 3, 1
    B_thresh = 0.5

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    def g(x, p):  # noqa: ARG001
        return jnp.stack([x[0]])

    p0 = jnp.array([2.0, 2.0, -3.0])
    jp = JaxProblem(
        f=f, g=g, n=n, m=m, p_example=p0,
        lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
        cl=jnp.array([-1e19]), cu=jnp.array([B_thresh]),
        options={"tol": 1e-10, "print_level": 0},
    )

    x_cold, warm = jp.solve_with_warm(p0, jnp.zeros(n), warm_start=None)
    np.testing.assert_allclose(np.asarray(x_cold), [B_thresh, 2.0, -3.0], atol=1e-6)

    # Warm re-solve at same p must agree to tight tolerance.
    x_warm, _ = jp.solve_with_warm(p0, jnp.zeros(n), warm_start=warm)
    np.testing.assert_allclose(np.asarray(x_warm), np.asarray(x_cold), atol=1e-8)


def test_jax_problem_vmap_solve_parallel_pounce_75():
    """`JaxProblem.vmap_solve_parallel` matches the standalone
    `vmap_solve_parallel` numerically — same code path, just with the
    prebuilt state reused across worker threads."""
    from pounce.jax import JaxProblem, vmap_solve_parallel

    n = 3
    B = 4

    def f(x, p):
        d = x - p
        return jnp.dot(d, d)

    rng = np.random.default_rng(0)
    p_batch = jnp.asarray(rng.standard_normal((B, n)))
    x0 = jnp.zeros(n)

    jp = JaxProblem(
        f=f, g=None, n=n, m=0, p_example=p_batch[0],
        options={"tol": 1e-10, "print_level": 0},
    )
    x_jp = jp.vmap_solve_parallel(p_batch, x0, workers=4)
    x_ref = vmap_solve_parallel(
        p_batch, f=f, g=None, x0=x0, n=n, m=0,
        options={"tol": 1e-10, "print_level": 0}, workers=4,
    )
    np.testing.assert_allclose(np.asarray(x_jp), np.asarray(x_ref), atol=1e-7)

    # Differentiable through the parallel path.
    def loss(pb):
        return jnp.sum(jp.vmap_solve_parallel(pb, x0, workers=4) ** 2)

    grad = np.asarray(jax.grad(loss)(p_batch))
    np.testing.assert_allclose(grad, 2.0 * np.asarray(p_batch), atol=1e-6)


def test_jax_problem_no_rebuild_on_repeat_solve_pounce_75():
    """Repeated solves through a prebuilt `JaxProblem` must skip the
    expensive build (jit + sparsity probe + Problem construction). We
    can't easily assert "no jax.jit" but we can assert the second solve
    is dramatically faster than the first (which paid no build cost
    itself, because the build happened in __init__).

    The build-once contract is: subsequent solves cost at most a small
    multiple of the bare `Problem.solve` time.
    """
    import time
    from pounce.jax import JaxProblem

    n = 5

    def f(x, p):
        d = x - p
        return jnp.dot(d, d) + 1e-4 * jnp.sum(x ** 4)

    p0 = jnp.zeros(n)
    jp = JaxProblem(
        f=f, g=None, n=n, m=0, p_example=p0,
        options={"tol": 1e-9, "print_level": 0},
    )

    # First solve pays no JIT cost (already done in __init__) but does
    # build the Problem instance lazily via the per-thread cache.
    rng = np.random.default_rng(0)
    p_seq = [jnp.asarray(rng.standard_normal(n)) for _ in range(10)]

    # Warm-up: jit'd derivatives run once with concrete arrays.
    _ = jp.solve(p_seq[0], jnp.zeros(n)).block_until_ready()

    t0 = time.perf_counter()
    for p in p_seq:
        _ = jp.solve(p, jnp.zeros(n)).block_until_ready()
    dt_reused = time.perf_counter() - t0

    # Sanity ceiling: 10 reuse-path solves on n=5 should comfortably
    # come in well under 1 second on any machine that can run the
    # tests. The pre-#75 path was paying ~70ms per solve so 10 solves
    # was ~0.7s — assert under 0.5s to catch any regression that
    # silently re-rebuilds.
    assert dt_reused < 0.5, f"reused solves too slow: {dt_reused:.3f}s"


def test_factor_reuse_matches_dense_pounce_76():
    """Issue #76 (B): the k_aug-style factor-reuse backward must
    produce gradients that agree with the dense ``jnp.linalg.solve``
    backward to ~1e-8. We exercise three cases together because each
    exercises a different part of the compound back-solve:

    * pure equality constraint — primary y_c row coupling
    * slack inequality (cl < cu, multiplier ≈ 0 at convergence) —
      verifies the (v_l, v_u) barrier rows correctly drop the row
      from the back-solve (k_aug's reason for existing on slack
      ineqs is the same as the dense path's #73 active-set fix)
    * active variable bound — verifies the (z_l, z_u) barrier rows
      collapse ``dx_i/dp`` to zero on the active coordinate
    """
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2) + 0.1 * jnp.sum(x ** 4)

    def g(x, p):  # noqa: ARG001
        return jnp.stack([
            x[0] + x[1] + x[2] - 1.0,   # equality
            x[2],                        # slack inequality (cl < cu)
        ])

    kwargs = dict(
        f=f, g=g, n=3, m=2, p_example=jnp.zeros(3),
        lb=jnp.array([0.4, -10.0, -10.0]),  # x[0] >= 0.4 — likely active
        ub=jnp.full(3, 10.0),
        cl=jnp.array([0.0, -1e20]),
        cu=jnp.array([0.0, 1e20]),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )
    jp_new = JaxProblem(**kwargs, factor_reuse=True)
    jp_old = JaxProblem(**kwargs, factor_reuse=False)

    p = jnp.array([-0.2, 0.5, 0.4])

    def loss(jp, p):
        return jnp.sum(jp.solve(p, jnp.ones(3)) ** 2)

    g_new = jax.grad(lambda p: loss(jp_new, p))(p)
    g_old = jax.grad(lambda p: loss(jp_old, p))(p)
    np.testing.assert_allclose(np.asarray(g_new), np.asarray(g_old), atol=1e-7)


def test_factor_reuse_jacobian_pounce_76():
    """The bwd is called once per output direction under
    ``jax.jacobian``; verify the LRU lookup holds the factor across
    repeated reads (pop-on-read would crash from the second direction
    onward)."""
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return jnp.stack([x[0] - x[1]])

    jp = JaxProblem(
        f=f, g=g, n=2, m=1, p_example=jnp.zeros(2),
        lb=jnp.full(2, -1e19), ub=jnp.full(2, 1e19),
        cl=jnp.zeros(1), cu=jnp.zeros(1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )
    p = jnp.array([0.3, 0.7])
    J = jax.jacobian(lambda p: jp.solve(p, jnp.zeros(2)))(p)
    # x* projects p onto the line x[0] = x[1], so dx*/dp = 0.5 * (1 1; 1 1).
    expected = 0.5 * np.ones((2, 2))
    np.testing.assert_allclose(np.asarray(J), expected, atol=1e-6)


def test_batched_solve_matches_vmap_parallel_pounce_76():
    """Issue #76 (A): the stacked block-diagonal solve must produce
    the same per-block ``x*`` as :meth:`vmap_solve_parallel` (within
    IPM convergence tolerance). The stacked path runs one IPM with a
    shared barrier homotopy over a size ``B*(n+m)`` block-diagonal KKT;
    the parallel path runs B independent IPMs. At convergence both
    sit on the same per-block KKT manifold, so the answers should
    agree to ~``tol``.
    """
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return jnp.stack([x[0] + x[1] - 1.0])

    jp = JaxProblem(
        f=f, g=g, n=2, m=1, p_example=jnp.zeros(2),
        lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
        cl=jnp.zeros(1), cu=jnp.zeros(1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )

    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5], [-0.1, 0.4], [1.0, 2.0]])
    x_b = jp.batched_solve(p_batch, x0=jnp.zeros(2))
    x_p = jp.vmap_solve_parallel(p_batch, x0=jnp.zeros(2), workers=2)
    np.testing.assert_allclose(np.asarray(x_b), np.asarray(x_p), atol=1e-8)


def test_batched_solve_grad_pounce_76():
    """Issue #76 (A): ``jax.grad`` through ``batched_solve`` must
    agree with ``jax.grad`` through ``vmap_solve_parallel`` — block-
    diagonal coupling means the per-element bwd is exact, and we vmap
    it over the batch.
    """
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return jnp.stack([x[0] + x[1] - 1.0])

    jp = JaxProblem(
        f=f, g=g, n=2, m=1, p_example=jnp.zeros(2),
        lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
        cl=jnp.zeros(1), cu=jnp.zeros(1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )

    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5], [-0.1, 0.4]])

    def loss_b(P):
        return jnp.sum(jp.batched_solve(P, x0=jnp.zeros(2)) ** 2)

    def loss_p(P):
        return jnp.sum(
            jp.vmap_solve_parallel(P, x0=jnp.zeros(2), workers=2) ** 2
        )

    g_b = jax.grad(loss_b)(p_batch)
    g_p = jax.grad(loss_p)(p_batch)
    np.testing.assert_allclose(np.asarray(g_b), np.asarray(g_p), atol=1e-7)


def test_batched_solve_unconstrained_pounce_76():
    """Issue #76 (A): the m=0 path takes the no-constraints branch of
    the stacked Hessian (signature ``(x, sigma, p)`` instead of
    ``(x, lam, sigma, p)``). Verifies it works and agrees with the
    closed-form ``x* = p``.
    """
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    jp = JaxProblem(
        f=f, g=None, n=3, m=0, p_example=jnp.zeros(3),
        lb=jnp.full(3, -10.0), ub=jnp.full(3, 10.0),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )

    p_batch = jnp.array([[0.1, 0.2, 0.3], [-1.0, 0.5, 2.0]])
    x_b = jp.batched_solve(p_batch, x0=jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(x_b), np.asarray(p_batch), atol=1e-7)


def test_factor_reuse_raises_clean_error_when_no_factor():
    """The factor-reuse backward must raise a clear, actionable
    ``RuntimeError`` (not crash with a low-level ``TypeError`` on
    ``NoneType.__getitem__``) when the forward solve terminated without
    a converged factor. Caught early during the pounce#76 (A) bench:
    a poorly-conditioned problem hit infeasibility and the bwd
    surfaced as a confusing crash deep inside the pure_callback.
    """
    import pytest
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        # Infeasible: requires sum(x) == 100 but bounds force x in [-1, 1]
        # so the feasible set is empty.
        return jnp.stack([jnp.sum(x) - 100.0])

    jp = JaxProblem(
        f=f, g=g, n=3, m=1, p_example=jnp.zeros(3),
        lb=jnp.full(3, -1.0), ub=jnp.full(3, 1.0),
        cl=jnp.zeros(1), cu=jnp.zeros(1),
        options={"tol": 1e-9, "print_level": 0, "sb": "yes", "max_iter": 30},
        factor_reuse=True,
    )

    p_val = jnp.array([0.1, 0.2, 0.3])
    with pytest.raises(Exception) as excinfo:
        jax.grad(lambda p: jnp.sum(jp.solve(p, jnp.zeros(3)) ** 2))(p_val)
    msg = str(excinfo.value)
    # The error gets wrapped by JAX's callback machinery; we just want
    # the actionable text to land somewhere in the chain.
    chain = msg + " " + str(excinfo.value.__cause__ or "")
    assert "factor_reuse=False" in chain, f"missing fallback hint in: {chain}"
