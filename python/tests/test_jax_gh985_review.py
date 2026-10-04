"""gh#985 review fixes: the jaxpr sparsity analysis must be a superset of
what AD produces, including where AD does *not* differentiate the primal
(custom_jvp / custom_vjp) and where a primitive carries an optional
parameter (``reshape``'s ``dimensions``); its memory guard must fire before
allocating; batched solves report failures once, aggregated."""

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pounce.jax import JaxProblem, from_jax, vmap_solve, vmap_solve_parallel
from pounce.jax import _jaxpr_sparsity as J

from test_jax_sparsity_gh985 import BECK_KW, P_BECK, _beck_f, _beck_g  # noqa: E402


# --- custom derivatives: AD uses the rule, not the primal --------------------


@jax.custom_jvp
def _cubic_root(a):
    """y(a) with y^3 + y = a: primal by Newton under stop_gradient, the
    derivative by the implicit-function rule (the standard implicit-diff
    pattern).  The primal has *no* dependency on ``a`` as AD sees it."""
    a = jax.lax.stop_gradient(a)
    y = a
    for _ in range(30):
        y = y - (y**3 + y - a) / (3 * y**2 + 1)
    return y


@_cubic_root.defjvp
def _cubic_root_jvp(p, t):
    y = _cubic_root(p[0])
    return y, t[0] / (3 * y**2 + 1)


@jax.custom_jvp
def _ste(x):
    """Straight-through estimator: primal round (zero derivative), rule id."""
    return jnp.round(x)


_ste.defjvp(lambda p, t: (_ste(p[0]), t[0]))


@jax.custom_jvp
def _hidden(x):
    """Primal ignores x[3:], the rule's coefficient reads it."""
    return x[0] * jnp.ones(3)


_hidden.defjvp(lambda p, t: (_hidden(p[0]), t[0][:3] * p[0][3:] ** 2))


@jax.custom_vjp
def _ste_vjp(x):
    return jnp.round(x)


_ste_vjp.defvjp(lambda x: (_ste_vjp(x), None), lambda r, g: (g,))


@jax.custom_vjp
def _mix_vjp(x):
    return jnp.sin(x) * x[::-1]


def _mix_fwd(x):
    return _mix_vjp(x), x


def _mix_bwd(x, g):
    return (jax.vjp(lambda y: jnp.sin(y) * y[::-1], x)[1](g)[0],)


_mix_vjp.defvjp(_mix_fwd, _mix_bwd)


def test_implicit_diff_custom_jvp_solves_to_the_right_answer():
    """The reviewer's repro: reading the primal, the Jacobian pattern lost
    the (0, 0) entry, the solver never saw dy/da, and returned
    Solve_Succeeded at obj 2.25 instead of 0.6248."""
    f = lambda x: (x[1] - 1.5) ** 2 + 0.1 * x[0] ** 2 + x[2] ** 2  # noqa: E731
    g = lambda x: jnp.stack([_cubic_root(x[0]) - x[1]])  # noqa: E731
    jp = J.jacobian_pattern(g, 3, 1)
    assert jp is not None and (0, 0) in set(zip(*map(list, jp)))
    p = from_jax(f, g, n=3, m=1, cl=np.zeros(1), cu=np.zeros(1))
    assert p.problem_obj.pattern_source == {"jac": "jaxpr", "hess": "jaxpr"}
    p.add_option("print_level", 0)
    x, info = p.solve(x0=np.array([1.0, 1.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(info["obj_val"], 0.6248125409691748, rtol=1e-8)
    np.testing.assert_allclose(x[:2], [1.67512, 0.91331], atol=1e-4)


def test_straight_through_estimator_is_not_empty():
    jp = J.jacobian_pattern(lambda x: _ste(x) * 2.0, 4, 4)
    assert set(zip(*map(list, jp))) == {(i, i) for i in range(4)}


def test_custom_jvp_rule_is_read_precisely_not_densely():
    # relu is a custom_jvp in jax.nn; it must stay diagonal (no dense
    # fallback for every custom_jvp in the library).
    jp = J.jacobian_pattern(lambda x: jax.nn.relu(x) * x, 50, 50)
    assert len(jp[0]) == 50


def test_custom_vjp_is_bounded_soundly():
    jp = J.jacobian_pattern(lambda x: _ste_vjp(x) * 2.0, 4, 4)
    assert jp is not None
    S = np.zeros((4, 4), bool)
    S[jp] = True
    assert np.diag(S).all()


# --- reshape(dimensions=...) ------------------------------------------------


def _hess_superset(f, n, pts):
    hp = J.hessian_lower_pattern(jax.grad(f), (jnp.zeros(n),), n)
    assert hp is not None
    HS = np.zeros((n, n), bool)
    HS[hp] = True
    HS |= HS.T
    for x in pts:
        H = np.asarray(jax.hessian(f)(jnp.asarray(x)))
        assert not ((H != 0) & ~HS).any(), "Hessian pattern dropped a nonzero"
    return HS


def test_prod_axis_hessian_has_the_pairwise_blocks():
    """The gradient program of prod(..., axis=1) uses
    reshape(dimensions=perm); ignoring the permutation dropped H01, H23,
    H45."""
    f = lambda x: jnp.sum(jnp.prod(x.reshape(3, 2), axis=1) ** 2)  # noqa: E731
    HS = _hess_superset(f, 6, [np.arange(1.0, 7.0)])
    want = np.kron(np.eye(3, dtype=bool), np.ones((2, 2), bool))
    np.testing.assert_array_equal(HS, want)


def test_ravel_order_f_jacobian():
    g = lambda x: jnp.ravel(x.reshape(2, 3), order="F")[:2] ** 2  # noqa: E731
    jp = J.jacobian_pattern(g, 6, 2)
    # F-order ravel of [[x0 x1 x2], [x3 x4 x5]] starts x0, x3
    assert set(zip(*map(list, jp))) == {(0, 0), (1, 3)}


def test_lax_reshape_with_dimensions_matches_transpose_then_reshape():
    g = lambda x: jax.lax.reshape(  # noqa: E731
        x.reshape(2, 3, 2), (12,), dimensions=(2, 0, 1)
    ) * jnp.arange(1.0, 13.0)
    jp = J.jacobian_pattern(g, 12, 12)
    perm = np.transpose(np.arange(12).reshape(2, 3, 2), (2, 0, 1)).ravel()
    assert set(zip(*map(list, jp))) == {(i, int(perm[i])) for i in range(12)}


# --- extended superset fuzz -------------------------------------------------

n_ = 6
_B = np.random.default_rng(0).standard_normal((3, n_))

REVIEW_CASES = {
    "custom_jvp_implicit": lambda x: jnp.stack(
        [_cubic_root(x[0]) * x[1], _cubic_root(x[2] + x[3])]
    ),
    "custom_jvp_ste": lambda x: _ste(x) * x[::-1],
    "custom_jvp_hidden": lambda x: _hidden(x) * 2.0,
    "custom_jvp_in_jit": lambda x: jax.jit(lambda y: _ste(y) ** 2)(x),
    "custom_vjp_ste": lambda x: _ste_vjp(x) * x[::-1],
    "custom_vjp_mix": lambda x: _mix_vjp(x) * x[0],
    "custom_jvp_in_checkpoint": lambda x: jax.checkpoint(lambda y: _ste(y) * y)(x),
    "prod_axis1": lambda x: jnp.prod(x.reshape(3, 2), axis=1),
    "prod_axis0": lambda x: jnp.prod(x.reshape(2, 3), axis=0),
    "prod_3d": lambda x: jnp.prod(x.reshape(1, 2, 3), axis=(0, 2)),
    "ravel_F": lambda x: jnp.ravel(x.reshape(2, 3), order="F")[:4] ** 2,
    "reshape_F": lambda x: x.reshape(3, 2, order="F")[:, 0] * x[:3],
    "cumsum_rev": lambda x: jnp.cumsum(x[::-1] ** 2)[::-1],
    "cumprod": lambda x: jnp.cumprod(x),
    "cummax": lambda x: jax.lax.cummax(x * x[::-1]),
    "expand_squeeze": lambda x: jnp.squeeze(jnp.expand_dims(x, (0, 2)) ** 2)[1:4],
    "rev2d": lambda x: jnp.flip(x.reshape(2, 3), axis=1).ravel() * x,
    "argmax_weighted": lambda x: x * jax.nn.one_hot(jnp.argmax(x), n_),
    "convert": lambda x: (x.astype(jnp.float32) ** 2).astype(jnp.float64),
    "dyn_slice_const": lambda x: jax.lax.dynamic_slice(x, (1,), (4,)) * x[:4],
    "dyn_update": lambda x: jax.lax.dynamic_update_slice(x, x[:2] ** 2, (3,)),
    "transpose3d": lambda x: jnp.transpose(
        jnp.concatenate([x, x ** 2]).reshape(2, 3, 2), (2, 0, 1)
    ).ravel()[:6] * x,
    "batched_matmul": lambda x: jnp.einsum(
        "bij,bjk->bik", x.reshape(2, 3, 1), x.reshape(2, 1, 3)
    ).ravel(),
    "dot_const_rhs": lambda x: (x.reshape(2, 3) @ _B).ravel(),
    "scatter_add": lambda x: jnp.zeros(4).at[jnp.array([0, 0, 3])].add(x[:3] * x[3:]),
    "stop_gradient": lambda x: jax.lax.stop_gradient(x) * x + x[::-1],
    "softplus_silu": lambda x: jax.nn.softplus(x) * jax.nn.silu(x[::-1]),
}


@pytest.mark.parametrize("name", sorted(REVIEW_CASES))
def test_review_cases_superset_of_ad(name):
    fn = REVIEW_CASES[name]
    m = int(np.size(fn(jnp.ones(n_))))
    rng = np.random.default_rng(3)
    pts = [rng.standard_normal(n_) * 1.5 for _ in range(8)]
    pts += [rng.uniform(0.2, 2.0, n_) for _ in range(4)]

    jp = J.jacobian_pattern(lambda x: jnp.ravel(fn(x)), n_, m)
    if jp is not None:  # a fallback (None) is always sound
        S = np.zeros((m, n_), bool)
        S[jp] = True
        Jf = jax.jacrev(lambda x: jnp.ravel(fn(x)))
        for x in pts:
            Jx = np.asarray(Jf(jnp.asarray(x)))
            assert not ((Jx != 0) & np.isfinite(Jx) & ~S).any(), (
                "jaxpr Jacobian pattern dropped a nonzero"
            )

    w = jnp.arange(1, m + 1) ** 0.5
    h = lambda x: jnp.sum(jnp.ravel(fn(x)) * w) + jnp.sum(jnp.sin(jnp.ravel(fn(x))))  # noqa: E731
    hp = J.hessian_lower_pattern(jax.grad(h), (jnp.zeros(n_),), n_)
    if hp is not None:
        HS = np.zeros((n_, n_), bool)
        HS[hp] = True
        HS |= HS.T
        hess = jax.jacrev(jax.grad(h))  # reverse-over-reverse: custom_vjp ok
        for x in pts:
            Hx = np.asarray(hess(jnp.asarray(x)))
            assert not ((Hx != 0) & np.isfinite(Hx) & ~HS).any(), (
                "jaxpr Hessian pattern dropped a nonzero"
            )


def test_cases_that_used_to_be_unsound_are_now_bounded():
    # Not just "falls back": these are read off the jaxpr.
    for name in ("custom_jvp_implicit", "custom_jvp_ste", "prod_axis1", "ravel_F"):
        fn = REVIEW_CASES[name]
        m = int(np.size(fn(jnp.ones(n_))))
        assert J.jacobian_pattern(lambda x: jnp.ravel(fn(x)), n_, m) is not None, name


# --- memory guard -----------------------------------------------------------


def test_large_outer_product_falls_back_without_allocating(monkeypatch):
    """The dot_general / broadcast guards fire before the B*M*N index
    arrays are built: on an outer product past the budget the analysis
    returns None immediately instead of allocating ~1 GB first."""
    import tracemalloc

    n = 5000  # 2.5e7 output entries > the 2e7 budget
    for g in (
        lambda x: (x[:, None] @ x[None, :]).ravel(),  # dot_general
        lambda x: jnp.outer(x, x).ravel(),  # broadcast mul
    ):
        tracemalloc.start()
        try:
            assert J.jacobian_pattern(g, n, n * n) is None
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 64 * 2**20, f"peak {peak / 2**20:.0f} MB"


# --- batched failure reporting ----------------------------------------------


def _mixed_batch():
    bad = P_BECK.at[2].set(20_000.0)  # demand above the total capacity
    return jnp.stack([P_BECK, bad, P_BECK, bad, bad])


@pytest.mark.parametrize("which", ["vmap_solve", "vmap_solve_parallel"])
def test_batched_failures_are_one_aggregated_warning(which):
    fn = {"vmap_solve": vmap_solve, "vmap_solve_parallel": vmap_solve_parallel}[which]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")  # the filter that collapsed them
        jax.block_until_ready(fn(_mixed_batch(), **BECK_KW))
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(msgs) == 1, msgs
    assert "3 of 5 element(s) did not converge" in msgs[0]
    for i in (1, 3, 4):
        assert f" {i} (" in msgs[0]
    assert " 0 (" not in msgs[0] and " 2 (" not in msgs[0]


def test_parallel_raise_names_every_failing_element():
    with pytest.raises(Exception, match=r"3 of 5 element\(s\) did not converge"):
        jax.block_until_ready(
            vmap_solve_parallel(_mixed_batch(), on_failure="raise", **BECK_KW)
        )


@pytest.mark.parametrize("which", ["vmap_solve", "vmap_solve_parallel"])
def test_jaxproblem_batched_failures_are_aggregated(which):
    kw = dict(
        f=_beck_f, g=_beck_g, n=3, m=1, p_example=P_BECK,
        lb=np.zeros(3), ub=np.full(3, 5000.0), cl=np.zeros(1), cu=np.zeros(1),
        options={"print_level": 0, "max_iter": 50},
    )
    with JaxProblem(**kw) as jp:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            jax.block_until_ready(getattr(jp, which)(_mixed_batch(), jnp.full(3, 500.0)))
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(msgs) == 1, msgs
    assert "3 of 5 element(s) did not converge" in msgs[0]
