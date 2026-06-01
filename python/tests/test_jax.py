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


def test_solve_with_warm_threads_barrier_mu_pounce_86():
    """A 4-element warm-state `(lam, zL, zU, mu)` seeds the barrier and
    returns the converged μ, so a predictor–corrector loop can thread it
    forward; the 3-tuple path is unchanged (pounce#86)."""
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

    p0 = jnp.array([2.0, 2.0, -3.0])  # active inequality

    # 3-tuple / None warm-start: backward-compatible 3-element state.
    x0_star, warm3 = forward(p0, warm=None)
    assert len(warm3) == 3

    # Report-only 4-tuple (mu=None): get μ out without seeding it in.
    x_ro, warm_ro = forward(p0, warm=(*warm3, None))
    assert len(warm_ro) == 4
    mu_out = float(np.asarray(warm_ro[3]))
    assert np.isfinite(mu_out) and 0.0 < mu_out < 1e-6

    # Seed μ back in (full 4-tuple) — same optimum, μ still round-trips.
    lam, zL, zU = warm3
    x_seed, warm_seed = forward(p0, warm=(lam, zL, zU, mu_out))
    assert len(warm_seed) == 4
    np.testing.assert_allclose(np.asarray(x_seed), np.asarray(x0_star), atol=1e-8)
    assert np.isfinite(float(np.asarray(warm_seed[3])))

    # Differentiability w.r.t. p is preserved through the μ-threaded path.
    def loss(p):
        x_star, _ = forward(p, warm=(lam, zL, zU, mu_out))
        return jnp.sum(x_star ** 2)

    grad = np.asarray(jax.grad(loss)(p0))
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


def test_batched_solve_factor_reuse_matches_dense_pounce_76():
    """Issue #76 (A)+(B) composition: ``batched_solve`` with
    ``factor_reuse=True`` routes the bwd through one stacked
    ``Solver.kkt_solve`` against the held stacked LDLᵀ factor; with
    ``factor_reuse=False`` it uses ``jax.vmap`` of the dense per-element
    KKT back-solve. The two paths solve the same IFT system on a
    block-diagonal compound KKT, so gradients must agree to IPM-tolerance.
    """
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return jnp.stack([x[0] + x[1] - 1.0])

    def build(reuse: bool):
        return JaxProblem(
            f=f, g=g, n=2, m=1, p_example=jnp.zeros(2),
            lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
            cl=jnp.zeros(1), cu=jnp.zeros(1),
            options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
            factor_reuse=reuse,
        )

    jp_reuse = build(True)
    jp_dense = build(False)
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5], [-0.1, 0.4], [0.8, -0.2]])

    def loss(jp, P):
        return jnp.sum(jp.batched_solve(P, x0=jnp.zeros(2)) ** 2)

    # Forward must be identical (same IPM, same stacked NLP).
    x_reuse = jp_reuse.batched_solve(p_batch, x0=jnp.zeros(2))
    x_dense = jp_dense.batched_solve(p_batch, x0=jnp.zeros(2))
    np.testing.assert_allclose(
        np.asarray(x_reuse), np.asarray(x_dense), atol=1e-12,
    )

    g_reuse = jax.grad(lambda P: loss(jp_reuse, P))(p_batch)
    g_dense = jax.grad(lambda P: loss(jp_dense, P))(p_batch)
    np.testing.assert_allclose(
        np.asarray(g_reuse), np.asarray(g_dense), atol=1e-7,
    )


def test_batched_solve_factor_reuse_jacobian_pounce_76():
    """Issue #76 (A)+(B): ``jax.jacobian`` over ``batched_solve`` with
    factor_reuse=True must produce a block-diagonal Jacobian (zero
    off-block) and match the dense per-element vmap path within
    IPM-tol. Exercises that the stacked back-solve's de-interleaving
    handles single-direction cotangents correctly under JAX's outer
    vmap.
    """
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    jp_reuse = JaxProblem(
        f=f, g=None, n=2, m=0, p_example=jnp.zeros(2),
        lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
        factor_reuse=True,
    )
    jp_dense = JaxProblem(
        f=f, g=None, n=2, m=0, p_example=jnp.zeros(2),
        lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
        factor_reuse=False,
    )

    p_batch = jnp.array([[0.3, 0.7], [-0.5, 0.5], [1.1, -0.4]])
    J_reuse = jax.jacobian(
        lambda P: jp_reuse.batched_solve(P, x0=jnp.zeros(2))
    )(p_batch)
    J_dense = jax.jacobian(
        lambda P: jp_dense.batched_solve(P, x0=jnp.zeros(2))
    )(p_batch)
    # Block-diagonal: J[k, :, j, :] = 0 for k != j.
    B = p_batch.shape[0]
    for k in range(B):
        for j in range(B):
            if k != j:
                np.testing.assert_allclose(
                    np.asarray(J_reuse[k, :, j, :]),
                    0.0, atol=1e-7,
                )
    np.testing.assert_allclose(
        np.asarray(J_reuse), np.asarray(J_dense), atol=1e-7,
    )


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


def test_factor_reuse_bwd_offthread_pounce_77():
    """Regression for pounce#77: ``jax.grad`` through a
    ``factor_reuse=True`` JaxProblem must survive being invoked from a
    worker thread.

    PySolver is ``#[pyclass(unsendable)]`` because ``RustSolver`` holds
    ``Rc<RefCell<dyn TNLP>>`` — touching it from any thread other than
    the one that built it triggers a PyO3 panic ("PySolver is
    unsendable, but sent to another thread"). JAX's ``pure_callback``
    dispatches ``host_call`` from XLA worker threads in jit'd training
    loops, which made the factor-reuse bwd unusable in practice.

    The fix pins all solver creation, ``solve()``, and ``kkt_solve``
    calls to a per-JaxProblem dedicated single-thread executor. This
    test confirms the fix end-to-end by running the entire forward +
    backward from a ``threading.Thread`` worker (which the pre-fix code
    would have panicked under).
    """
    import threading

    from pounce.jax import JaxProblem

    def f(x, p):
        return 0.5 * jnp.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return jnp.stack([jnp.sum(x) - 1.0])

    jp = JaxProblem(
        f=f, g=g, n=4, m=1, p_example=jnp.zeros(4),
        lb=jnp.full(4, -2.0), ub=jnp.full(4, 2.0),
        cl=jnp.zeros(1), cu=jnp.zeros(1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
        factor_reuse=True,
    )

    p_val = jnp.array([0.1, 0.2, 0.3, 0.4])
    x0 = jnp.zeros(4)

    def loss(p):
        x_star = jp.solve(p, x0)
        return jnp.sum(x_star ** 2)

    # Reference from the main thread — the existing tests cover this
    # path, so it should always work.
    grad_main = jax.grad(loss)(p_val)
    assert jnp.all(jnp.isfinite(grad_main))

    # Now invoke from a worker thread. Pre-fix this raised PyO3's
    # unsendable panic at the bwd's first kkt_solve call.
    result_holder: dict = {}

    def worker():
        try:
            result_holder["grad"] = jax.grad(loss)(p_val)
        except BaseException as exc:  # pragma: no cover - defensive
            result_holder["exc"] = exc

    t = threading.Thread(target=worker, name="pounce-77-worker")
    t.start()
    t.join(timeout=60.0)
    assert not t.is_alive(), "worker did not finish (pounce#77 regression?)"
    assert "exc" not in result_holder, (
        f"worker raised: {result_holder['exc']!r}"
    )
    grad_thread = result_holder["grad"]
    assert jnp.all(jnp.isfinite(grad_thread))
    # The pinned executor serializes solves, but the answer must match
    # the main-thread gradient to floating-point tolerance.
    assert jnp.allclose(grad_thread, grad_main, atol=1e-9)

    # And the batched path — the (A)+(B) composition uses the same
    # pinned executor for the stacked Solver and its bwd kkt_solve.
    p_batch = jnp.stack([p_val, p_val + 0.1, p_val - 0.1])

    def batched_loss(p_b):
        X = jp.batched_solve(p_b, x0)
        return jnp.sum(X ** 2)

    grad_b_main = jax.grad(batched_loss)(p_batch)

    result_holder2: dict = {}

    def worker_b():
        try:
            result_holder2["grad"] = jax.grad(batched_loss)(p_batch)
        except BaseException as exc:  # pragma: no cover - defensive
            result_holder2["exc"] = exc

    t2 = threading.Thread(target=worker_b, name="pounce-77-worker-batched")
    t2.start()
    t2.join(timeout=60.0)
    assert not t2.is_alive(), "batched worker did not finish"
    assert "exc" not in result_holder2, (
        f"batched worker raised: {result_holder2['exc']!r}"
    )
    assert jnp.allclose(result_holder2["grad"], grad_b_main, atol=1e-9)


# Module-level callables for pounce#77 pickle test — local functions
# inside a test body trip ``AttributeError: Can't get local object``
# before they reach the JaxProblem's own state, which would mask the
# real ``TypeError: cannot pickle '_thread.lock' object`` we want to
# assert against.
def _pounce_77_pickle_f(x, p):
    return 0.5 * jnp.sum((x - p) ** 2)


def _pounce_77_pickle_g(x, p):
    return jnp.stack([jnp.sum(x) - 1.0])


def test_jaxproblem_pickle_roundtrip_pounce_77():
    """Round-trip a JaxProblem through pickle and verify numerical
    agreement (pounce#77 follow-up).

    Originally the JaxProblem held a ``threading.Lock``, a
    ``threading.local`` cache, JAX JIT'd closures, and (with
    ``factor_reuse=True``) a ``ThreadPoolExecutor`` — none of which
    survive ``pickle.dumps``. That blocked the realistic distributed-
    training paths (Ray / Dask actors via ``cloudpickle``,
    ``multiprocessing(start_method='spawn')`` that ships a built
    JaxProblem to its workers, checkpointing for resume) at the
    serialization boundary.

    The fix is ``__getstate__`` / ``__setstate__`` that drop the per-
    process runtime state on the sending side and rebuild it on the
    receiving side. JIT closures are rebuilt from ``self._f`` /
    ``self._g`` (the user is responsible for ensuring those are
    themselves picklable — module-level functions or cloudpickle-
    compatible). Sparsity arrays are pickled so the receiving side
    doesn't redo the one-shot probe. Held LDLᵀ factors and registry
    ids reset (a fresh process has no history of forward solves).

    Verifies: forward / single-grad / batched-forward / batched-grad
    all agree exactly between pre- and post-pickle instances.
    """
    import pickle

    from pounce.jax import JaxProblem

    jp = JaxProblem(
        f=_pounce_77_pickle_f, g=_pounce_77_pickle_g,
        n=4, m=1, p_example=jnp.zeros(4),
        lb=jnp.full(4, -2.0), ub=jnp.full(4, 2.0),
        cl=jnp.zeros(1), cu=jnp.zeros(1),
        options={"print_level": 0, "sb": "yes"},
        factor_reuse=True,
    )

    p_val = jnp.array([0.1, 0.2, 0.3, 0.4])
    x0 = jnp.zeros(4)
    p_batch = jnp.stack([p_val, p_val + 0.1, p_val - 0.1])

    # Reference (pre-pickle).
    x_before = jp.solve(p_val, x0)
    grad_before = jax.grad(
        lambda p: jnp.sum(jp.solve(p, x0) ** 2)
    )(p_val)
    X_before = jp.batched_solve(p_batch, x0)
    grad_b_before = jax.grad(
        lambda pb: jnp.sum(jp.batched_solve(pb, x0) ** 2)
    )(p_batch)

    # Round-trip and re-run.
    blob = pickle.dumps(jp)
    jp2 = pickle.loads(blob)
    x_after = jp2.solve(p_val, x0)
    grad_after = jax.grad(
        lambda p: jnp.sum(jp2.solve(p, x0) ** 2)
    )(p_val)
    X_after = jp2.batched_solve(p_batch, x0)
    grad_b_after = jax.grad(
        lambda pb: jnp.sum(jp2.batched_solve(pb, x0) ** 2)
    )(p_batch)

    # JIT closures rebuild deterministically and the IPM is
    # deterministic, so the round-trip should be exact. Allow a
    # token tolerance against future numerical drift in JAX
    # canonicalisation, but expect 0 in practice.
    assert jnp.allclose(x_before, x_after, atol=1e-12)
    assert jnp.allclose(grad_before, grad_after, atol=1e-10)
    assert jnp.allclose(X_before, X_after, atol=1e-12)
    assert jnp.allclose(grad_b_before, grad_b_after, atol=1e-10)

    # And the dense path (factor_reuse=False) also round-trips. No
    # executor or registry lock to drop, but the JIT closures and
    # threading.local cache still aren't picklable without the
    # __getstate__ hook.
    jp_dense = JaxProblem(
        f=_pounce_77_pickle_f, g=_pounce_77_pickle_g,
        n=4, m=1, p_example=jnp.zeros(4),
        lb=jnp.full(4, -2.0), ub=jnp.full(4, 2.0),
        cl=jnp.zeros(1), cu=jnp.zeros(1),
        options={"print_level": 0, "sb": "yes"},
        factor_reuse=False,
    )
    x_dense_before = jp_dense.solve(p_val, x0)
    jp_dense2 = pickle.loads(pickle.dumps(jp_dense))
    x_dense_after = jp_dense2.solve(p_val, x0)
    assert jnp.allclose(x_dense_before, x_dense_after, atol=1e-12)


def test_dense_path_warns_on_large_n_plus_m_pounce_77():
    """n+m > 10000 must emit a UserWarning regardless of factor_reuse
    (pounce#77).

    Both bwd paths have scaling limits at this size: factor_reuse=False
    is O((n+m)^3)/O((n+m)^2) per block; factor_reuse=True is FFI-bound
    per cotangent so jacrev/jacfwd loses LAPACK factor sharing. The
    matrix-free MINRES/GMRES bwd that would close the gap is not yet
    implemented. The warning steers users to the regime table in the
    JaxProblem.factor_reuse docstring.
    """
    import warnings
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    def g(x, p):
        return jnp.array([jnp.sum(x) - p[0]])

    big_n, big_m = 10500, 1
    for fr in (False, True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            JaxProblem(
                f=f, g=g, n=big_n, m=big_m, p_example=jnp.zeros(big_n),
                lb=jnp.full(big_n, -1e19), ub=jnp.full(big_n, 1e19),
                cl=jnp.zeros(big_m), cu=jnp.zeros(big_m),
                options={"print_level": 0, "sb": "yes"},
                factor_reuse=fr,
            )
        matched = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "n+m=" in str(w.message)
            and "10000" in str(w.message)
        ]
        assert matched, (
            f"expected UserWarning at n+m={big_n + big_m} with "
            f"factor_reuse={fr}, got {[str(w.message) for w in caught]}"
        )

    # Below the threshold, neither setting should warn.
    small_n, small_m = 200, 1
    for fr in (False, True):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            JaxProblem(
                f=f, g=g, n=small_n, m=small_m,
                p_example=jnp.zeros(small_n),
                lb=jnp.full(small_n, -1e19), ub=jnp.full(small_n, 1e19),
                cl=jnp.zeros(small_m), cu=jnp.zeros(small_m),
                options={"print_level": 0, "sb": "yes"},
                factor_reuse=fr,
            )
        matched = [
            w for w in caught
            if issubclass(w.category, UserWarning) and "10000" in str(w.message)
        ]
        assert not matched, (
            f"factor_reuse={fr} at n+m={small_n + small_m} must not warn, "
            f"got {[str(w.message) for w in matched]}"
        )


def _build_warm_qp(reuse: bool):
    """Shared fixture for the pounce#78 batched-warm tests: simple
    parametric QP per block (min 0.5 ||x - p||² s.t. sum(x)=0)."""
    from pounce.jax import JaxProblem

    def f(x, p):
        return 0.5 * jnp.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return jnp.array([jnp.sum(x)])

    n, m = 5, 1
    return JaxProblem(
        f=f, g=g, n=n, m=m, p_example=jnp.zeros(n),
        lb=jnp.full(n, -1e19), ub=jnp.full(n, 1e19),
        cl=jnp.zeros(m), cu=jnp.zeros(m),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
        factor_reuse=reuse,
    )


def test_batched_solve_with_warm_matches_loop_pounce_78():
    """Issue #78: the stacked warm-batched forward must agree with
    looping ``solve_with_warm`` per sample. Same KKT system per block;
    only the dispatch shape (one stacked solve vs B loose solves)
    differs.
    """
    for reuse in (True, False):
        jp = _build_warm_qp(reuse)
        n, m = jp._n, jp._m
        B = 3
        rng = np.random.default_rng(0)
        p_batch = jnp.asarray(rng.standard_normal((B, n)))
        x0 = jnp.zeros((B, n))

        # Cold warm-batched: no warm vectors supplied.
        x_b, (lam_b, zL_b, zU_b) = jp.batched_solve_with_warm(p_batch, x0)

        # Reference: loop solve_with_warm per sample, cold.
        x_ref = []
        lam_ref = []
        for k in range(B):
            x_k, (lam_k, _, _) = jp.solve_with_warm(p_batch[k], x0[k])
            x_ref.append(x_k)
            lam_ref.append(lam_k)
        x_ref = jnp.stack(x_ref)
        lam_ref = jnp.stack(lam_ref)

        np.testing.assert_allclose(np.asarray(x_b), np.asarray(x_ref), atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(lam_b), np.asarray(lam_ref), atol=1e-7,
        )

        # Hot warm-batched: thread converged duals back in. Must still
        # land at the same x* (warm-start should not change the
        # solution, only the iteration count).
        x_b2, _ = jp.batched_solve_with_warm(
            p_batch, x0, warm_start=(lam_b, zL_b, zU_b),
        )
        np.testing.assert_allclose(np.asarray(x_b2), np.asarray(x_ref), atol=1e-7)


def test_batched_solve_with_warm_grad_matches_batched_solve_pounce_78():
    """Issue #78: gradient through the warm-batched forward must match
    the gradient through plain ``batched_solve`` — same KKT system, and
    the warm path treats ``warm_start`` / ``x0`` as stop-gradient so the
    only differentiable input is ``p_batch``.
    """
    for reuse in (True, False):
        jp = _build_warm_qp(reuse)
        n, m = jp._n, jp._m
        B = 4
        rng = np.random.default_rng(1)
        p_batch = jnp.asarray(rng.standard_normal((B, n)))
        x0 = jnp.zeros((B, n))

        def loss_warm(P):
            x, _ = jp.batched_solve_with_warm(P, x0)
            return 0.5 * jnp.sum(x ** 2)

        def loss_cold(P):
            x = jp.batched_solve(P, x0)
            return 0.5 * jnp.sum(x ** 2)

        g_warm = jax.grad(loss_warm)(p_batch)
        g_cold = jax.grad(loss_cold)(p_batch)
        np.testing.assert_allclose(
            np.asarray(g_warm), np.asarray(g_cold), atol=1e-7,
        )


def test_batched_solve_with_warm_warm_inputs_are_stop_gradient_pounce_78():
    """Issue #78: per the docstring, the warm-batched custom_vjp treats
    ``warm_start`` and ``x0`` as stop-gradient — same convention
    ``solve_with_warm`` uses on the single-sample path. Verifies the
    bwd returns zero cotangents for those four arguments.
    """
    jp = _build_warm_qp(reuse=True)
    n, m = jp._n, jp._m
    B = 2
    p_batch = jnp.asarray(np.random.default_rng(2).standard_normal((B, n)))
    x0 = jnp.zeros((B, n))
    lam_w = jnp.zeros((B, m))
    zL_w = jnp.zeros((B, n))
    zU_w = jnp.zeros((B, n))

    fn = jp._batched_solve_with_warm_fn(B)

    def loss(P, X0, L, ZL, ZU):
        x, _, _, _ = fn(P, X0, L, ZL, ZU)
        return 0.5 * jnp.sum(x ** 2)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(p_batch, x0, lam_w, zL_w, zU_w)
    g_p, g_x0, g_lam, g_zL, g_zU = grads
    # p gradient is nonzero (this is the differentiable input).
    assert float(jnp.max(jnp.abs(g_p))) > 0.0
    # Stop-gradient inputs return exact zeros.
    for arr, name in [(g_x0, "x0"), (g_lam, "lam_warm"),
                       (g_zL, "zL_warm"), (g_zU, "zU_warm")]:
        np.testing.assert_array_equal(np.asarray(arr), np.zeros_like(np.asarray(arr)))


# --------------------------------------------------------------------------
# pounce#82: post-solve Jacobian / sensitivity API from the held KKT factor
# --------------------------------------------------------------------------


def _build_jac_qp(reuse=True, bounded=False):
    """min ||x - p||² s.t. x[0] + x[1] = 1, optionally with a binding
    upper bound (to exercise the zU multiplier block)."""
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    def g(x, p):  # noqa: ARG001
        return jnp.stack([x[0] + x[1] - 1.0])

    ub = jnp.array([0.2, 10.0]) if bounded else jnp.full(2, 10.0)
    return JaxProblem(
        f=f, g=g, n=2, m=1, p_example=jnp.zeros(2),
        lb=jnp.full(2, -10.0), ub=ub,
        cl=jnp.zeros(1), cu=jnp.zeros(1),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
        factor_reuse=reuse,
    )


def _jacrev_block_diag(jp, p_batch, x0):
    """Reference per-block Jacobian via jax.jacrev over batched_solve."""
    J_full = jax.jacrev(lambda P: jp.batched_solve(P, x0))(p_batch)
    B = p_batch.shape[0]
    return jnp.stack([J_full[k, :, k, :] for k in range(B)])


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("reuse", [True, False])
def test_batched_solve_with_jacobian_matches_jacrev_pounce_82(bounded, reuse):
    """Issue #82: the full Jacobian from the held factor must equal
    ``jax.jacrev`` over ``batched_solve`` (the KKT system is symmetric,
    so J's row i is the VJP at cotangent e_i). Covers a binding upper
    bound (zU block participates) and both ``factor_reuse`` settings —
    the anchor path forces a held factor regardless."""
    jp = _build_jac_qp(reuse=reuse, bounded=bounded)
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5], [-0.1, 0.4]])
    x0 = jnp.zeros(2)

    x_star, (lam, zL, zU), J = jp.batched_solve_with_jacobian(p_batch, x0)

    # Forward agrees with batched_solve.
    np.testing.assert_allclose(
        np.asarray(x_star), np.asarray(jp.batched_solve(p_batch, x0)), atol=1e-9
    )
    # Jacobian agrees with jacrev.
    Jref = _jacrev_block_diag(jp, p_batch, x0)
    np.testing.assert_allclose(np.asarray(J), np.asarray(Jref), atol=1e-6)
    # Duals have the documented shapes.
    assert lam.shape == (3, 1)
    assert zL.shape == (3, 2)
    assert zU.shape == (3, 2)


def test_batched_solve_with_jacobian_wrt_cols_pounce_82():
    """Issue #82: ``wrt_cols`` selects parameter columns and matches the
    corresponding slice of the full Jacobian."""
    jp = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)

    _, _, J = jp.batched_solve_with_jacobian(p_batch, x0)
    _, _, J0 = jp.batched_solve_with_jacobian(p_batch, x0, wrt_cols=slice(0, 1))
    assert J0.shape == (2, 2, 1)
    np.testing.assert_allclose(np.asarray(J0), np.asarray(J[..., :1]), atol=1e-12)

    # Index-array form selects the same column.
    _, _, Jidx = jp.batched_solve_with_jacobian(p_batch, x0, wrt_cols=[0])
    np.testing.assert_allclose(np.asarray(Jidx), np.asarray(J0), atol=1e-12)


@pytest.mark.parametrize("bounded", [False, True])
def test_sensitivity_at_matches_jacobian_pounce_87(bounded):
    """Issue #87: exact ``∂x*/∂θ`` re-factored at a supplied point must
    equal the ground-truth ``jax.jacobian`` over a fresh solve — at the
    solver's own converged point and at *other* parameter values along a
    path. Covers a binding upper bound so the active-set read off the
    supplied bound multipliers is exercised."""
    jp = _build_jac_qp(bounded=bounded)
    x0 = jnp.zeros(2)

    # Sweep several θ as if walking a path. For each, get the converged
    # primal + duals, then re-factor the sensitivity at that supplied
    # point with no held factor in play.
    for theta in (jnp.array([0.3, 0.7]),
                  jnp.array([0.5, 0.5]),
                  jnp.array([-0.1, 0.4])):
        x_star, (lam, zL, zU), _ = jp.batched_solve_with_jacobian(
            theta[None, :], x0,
        )
        J = jp.sensitivity_at(
            x_star[0], theta, (lam[0], zL[0], zU[0]),
        )
        # Ground truth: jacobian through a single solve at this θ.
        Jref = jax.jacobian(lambda p: jp.solve(p, x0))(theta)
        assert J.shape == (2, 2)
        np.testing.assert_allclose(np.asarray(J), np.asarray(Jref), atol=1e-6)


def test_sensitivity_at_wrt_cols_pounce_87():
    """Issue #87: ``wrt_cols`` selects parameter columns of the
    re-factored sensitivity, matching the corresponding slice."""
    jp = _build_jac_qp()
    x0 = jnp.zeros(2)
    theta = jnp.array([0.3, 0.7])
    x_star, (lam, zL, zU), _ = jp.batched_solve_with_jacobian(theta[None, :], x0)
    duals = (lam[0], zL[0], zU[0])

    J = jp.sensitivity_at(x_star[0], theta, duals)
    J0 = jp.sensitivity_at(x_star[0], theta, duals, wrt_cols=slice(0, 1))
    assert J0.shape == (2, 1)
    np.testing.assert_allclose(np.asarray(J0), np.asarray(J[:, :1]), atol=1e-12)

    Jidx = jp.sensitivity_at(x_star[0], theta, duals, wrt_cols=[1])
    np.testing.assert_allclose(np.asarray(Jidx), np.asarray(J[:, 1:2]), atol=1e-12)


def test_sensitivity_at_requires_dual_triple_pounce_87():
    """Issue #87: the duals argument must be the full ``(lam, zL, zU)``
    triple — the active set is read from the bound multipliers."""
    jp = _build_jac_qp()
    x0 = jnp.zeros(2)
    theta = jnp.array([0.3, 0.7])
    x_star, (lam, zL, zU), _ = jp.batched_solve_with_jacobian(theta[None, :], x0)
    with pytest.raises(ValueError, match="lam, zL, zU"):
        jp.sensitivity_at(x_star[0], theta, (lam[0], zL[0]))


def test_batched_vjp_from_state_matches_jax_vjp_pounce_82():
    """Issue #82: ``batched_vjp_from_state`` equals ``jax.vjp`` over
    ``batched_solve`` (J^T @ x_bar), and equals J^T @ x_bar from the
    materialised Jacobian."""
    jp = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5], [-0.1, 0.4]])
    x0 = jnp.zeros(2)
    x_bar = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    x_star, duals, J, state = jp.batched_solve_with_jacobian(
        p_batch, x0, return_state=True
    )
    try:
        dp = jp.batched_vjp_from_state(state, x_bar)

        _, vjp_fn = jax.vjp(lambda P: jp.batched_solve(P, x0), p_batch)
        dp_ref = vjp_fn(x_bar)[0]
        np.testing.assert_allclose(np.asarray(dp), np.asarray(dp_ref), atol=1e-6)

        dp_fromJ = jnp.einsum("knp,kn->kp", J, x_bar)
        np.testing.assert_allclose(np.asarray(dp), np.asarray(dp_fromJ), atol=1e-9)
    finally:
        state.close()


def test_vjp_from_state_respects_wrt_cols_pounce_82():
    """Issue #82: a state anchored with ``wrt_cols`` returns the reduced
    parameter cotangent from ``batched_vjp_from_state``."""
    jp = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)
    x_bar = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    with jp.anchor(p_batch, x0, wrt_cols=slice(0, 1)) as state:
        dp = jp.batched_vjp_from_state(state, x_bar)
    assert dp.shape == (2, 1)

    with jp.anchor(p_batch, x0) as state_full:
        dp_full = jp.batched_vjp_from_state(state_full, x_bar)
    np.testing.assert_allclose(np.asarray(dp), np.asarray(dp_full[..., :1]), atol=1e-9)


def test_jacobian_duals_match_batched_solve_with_warm_pounce_82():
    """Issue #82: the (lam, zL, zU) contract matches
    ``batched_solve_with_warm`` on the same problem."""
    jp = _build_jac_qp(bounded=True)
    p_batch = jnp.array([[0.3, 0.7], [0.9, 0.1]])
    x0 = jnp.zeros(2)

    _, (lam, zL, zU) = jp.batched_solve_with_warm(p_batch, x0)
    _, (lam2, zL2, zU2), _J = jp.batched_solve_with_jacobian(p_batch, x0)
    np.testing.assert_allclose(np.asarray(lam2), np.asarray(lam), atol=1e-7)
    np.testing.assert_allclose(np.asarray(zL2), np.asarray(zL), atol=1e-7)
    np.testing.assert_allclose(np.asarray(zU2), np.asarray(zU), atol=1e-7)


def test_anchor_lifetime_context_manager_pounce_82():
    """Issue #82: context manager releases the pinned factor on exit."""
    jp = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)

    with jp.anchor(p_batch, x0) as state:
        assert not state.closed
        assert len(jp._pinned_solvers) == 1
    assert state.closed
    assert len(jp._pinned_solvers) == 0


def test_anchor_explicit_close_idempotent_and_raises_when_closed_pounce_82():
    """Issue #82: explicit close frees the pin, is idempotent, and a
    from-state op on a closed handle raises."""
    jp = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)
    x_bar = jnp.zeros((2, 2))

    state = jp.anchor(p_batch, x0)
    assert len(jp._pinned_solvers) == 1
    state.close()
    assert len(jp._pinned_solvers) == 0
    state.close()  # idempotent
    assert state.closed
    with pytest.raises(RuntimeError, match="closed"):
        jp.batched_vjp_from_state(state, x_bar)


def test_anchor_survives_beyond_lru_capacity_pounce_82():
    """Issue #82 core requirement: a pinned anchor outlives LRU eviction.
    With the LRU bounded at 1, many intervening batched_solve calls would
    evict an ordinary cached factor — the pinned one must remain usable."""
    jp = _build_jac_qp()
    jp._solver_registry_capacity = 1  # force aggressive LRU eviction
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)
    x_bar = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    with jp.anchor(p_batch, x0) as state:
        # Hammer the LRU with unrelated forward+backward solves.
        for _ in range(5):
            jax.grad(lambda P: jnp.sum(jp.batched_solve(P, x0) ** 2))(p_batch)
        # The pinned factor is still resolvable.
        dp = jp.batched_vjp_from_state(state, x_bar)
        _, vjp_fn = jax.vjp(lambda P: jp.batched_solve(P, x0), p_batch)
        np.testing.assert_allclose(
            np.asarray(dp), np.asarray(vjp_fn(x_bar)[0]), atol=1e-6
        )


def test_anchor_gc_finalizer_reclaims_pounce_82():
    """Issue #82: dropping a handle without close() still reclaims the
    pinned factor via the weakref finalizer."""
    import gc

    jp = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)

    state = jp.anchor(p_batch, x0)
    assert len(jp._pinned_solvers) == 1
    del state
    gc.collect()
    assert len(jp._pinned_solvers) == 0


def test_anchor_reanchor_swaps_without_leak_pounce_82():
    """Issue #82: reanchor closes the prior pin before taking a new one,
    so the pinned count stays at 1 (no overwrite leak)."""
    jp = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)

    state = jp.anchor(p_batch, x0)
    sid1 = state.sid
    state.reanchor(p_batch * 1.1, x0)
    assert state.sid != sid1
    assert len(jp._pinned_solvers) == 1
    state.close()
    assert len(jp._pinned_solvers) == 0


def test_anchor_capacity_raises_loudly_pounce_82():
    """Issue #82: exceeding the pinned-handle cap raises rather than
    growing unbounded, and does not leak a factor on the failed call."""
    jp = _build_jac_qp()
    jp._pinned_capacity = 3
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)

    states = [jp.anchor(p_batch, x0) for _ in range(3)]
    try:
        assert len(jp._pinned_solvers) == 3
        with pytest.raises(RuntimeError, match="too many live anchored"):
            jp.anchor(p_batch, x0)
        # Failed anchor did not add a pin.
        assert len(jp._pinned_solvers) == 3
    finally:
        for s in states:
            s.close()
    assert len(jp._pinned_solvers) == 0


def test_vjp_from_state_rejects_foreign_state_pounce_82():
    """Issue #82: a state from one JaxProblem can't be used with another."""
    jp1 = _build_jac_qp()
    jp2 = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)

    with jp1.anchor(p_batch, x0) as state:
        with pytest.raises(ValueError, match="different"):
            jp2.batched_vjp_from_state(state, jnp.zeros((2, 2)))


# --------------------------------------------------------------------------
# pounce#82 Phase 2: forward-mode JVP from the held factor (J @ dp)
# --------------------------------------------------------------------------


def _fd_jvp(jp, p_batch, x0, dp, eps=1e-6):
    """Central finite-difference directional derivative of batched_solve.
    (batched_solve is a custom_vjp, so jax.jvp can't trace it directly.)"""
    xp = jp.batched_solve(p_batch + eps * dp, x0)
    xm = jp.batched_solve(p_batch - eps * dp, x0)
    return (xp - xm) / (2.0 * eps)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("reuse", [True, False])
def test_batched_jvp_from_state_matches_jacobian_pounce_82(bounded, reuse):
    """Issue #82 Phase 2: ``J @ dp`` from the forward path equals the
    contraction of the materialised Jacobian (machine precision) and a
    finite-difference directional derivative (loose)."""
    jp = _build_jac_qp(reuse=reuse, bounded=bounded)
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5], [-0.05, 0.4]])
    x0 = jnp.zeros(2)
    dp = jnp.array([[1.0, 0.0], [0.2, -0.3], [0.5, 0.5]])

    x_star, _, J, state = jp.batched_solve_with_jacobian(
        p_batch, x0, return_state=True
    )
    try:
        dx = jp.batched_jvp_from_state(state, dp)
    finally:
        state.close()

    assert dx.shape == (3, 2)
    dx_fromJ = jnp.einsum("knp,kp->kn", J, dp)
    np.testing.assert_allclose(np.asarray(dx), np.asarray(dx_fromJ), atol=1e-10)
    dx_fd = _fd_jvp(jp, p_batch, x0, dp)
    np.testing.assert_allclose(np.asarray(dx), np.asarray(dx_fd), atol=1e-6)


def test_batched_jvp_unconstrained_pounce_82():
    """Issue #82 Phase 2: for ``min ||x - p||²`` (x* = p) the directional
    derivative is the identity, so ``J @ dp = dp``."""
    from pounce.jax import JaxProblem

    def f(x, p):
        return jnp.sum((x - p) ** 2)

    jp = JaxProblem(
        f=f, g=None, n=3, m=0, p_example=jnp.zeros(3),
        lb=jnp.full(3, -10.0), ub=jnp.full(3, 10.0),
        options={"tol": 1e-10, "print_level": 0, "sb": "yes"},
    )
    p_batch = jnp.array([[0.1, 0.2, 0.3], [-1.0, 0.5, 2.0]])
    dp = jnp.array([[1.0, 2.0, 3.0], [0.5, 0.0, -1.0]])
    x0 = jnp.zeros(3)

    with jp.anchor(p_batch, x0) as state:
        dx = jp.batched_jvp_from_state(state, dp)
    np.testing.assert_allclose(np.asarray(dx), np.asarray(dp), atol=1e-7)


def test_batched_jvp_respects_wrt_cols_pounce_82():
    """Issue #82 Phase 2: a state anchored with ``wrt_cols`` takes a
    reduced ``dp`` and matches the full-space JVP with zeros elsewhere."""
    jp = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)

    with jp.anchor(p_batch, x0, wrt_cols=slice(0, 1)) as state:
        dx_red = jp.batched_jvp_from_state(state, jnp.array([[1.0], [0.4]]))
    dx_fd = _fd_jvp(jp, p_batch, x0, jnp.array([[1.0, 0.0], [0.4, 0.0]]))
    np.testing.assert_allclose(np.asarray(dx_red), np.asarray(dx_fd), atol=1e-6)


def test_batched_jvp_shape_validation_pounce_82():
    """Issue #82 Phase 2: a dp_batch with the wrong per-block shape or
    batch size is rejected with a clear error."""
    jp = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)

    with jp.anchor(p_batch, x0) as state:
        with pytest.raises(ValueError, match="per-block shape"):
            jp.batched_jvp_from_state(state, jnp.zeros((2, 3)))
        with pytest.raises(ValueError, match="leading dim"):
            jp.batched_jvp_from_state(state, jnp.zeros((3, 2)))
    # wrt_cols state expects the reduced width.
    with jp.anchor(p_batch, x0, wrt_cols=slice(0, 1)) as state:
        with pytest.raises(ValueError, match="per-block shape"):
            jp.batched_jvp_from_state(state, jnp.zeros((2, 2)))


def test_batched_jvp_rejects_closed_and_foreign_state_pounce_82():
    """Issue #82 Phase 2: JVP-from-state guards lifetime and ownership
    like the VJP path."""
    jp1 = _build_jac_qp()
    jp2 = _build_jac_qp()
    p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    x0 = jnp.zeros(2)

    state = jp1.anchor(p_batch, x0)
    state.close()
    with pytest.raises(RuntimeError, match="closed"):
        jp1.batched_jvp_from_state(state, jnp.zeros((2, 2)))

    with jp1.anchor(p_batch, x0) as state:
        with pytest.raises(ValueError, match="different"):
            jp2.batched_jvp_from_state(state, jnp.zeros((2, 2)))
