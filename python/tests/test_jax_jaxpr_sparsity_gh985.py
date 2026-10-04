"""gh#985 (second pass): structural sparsity is read off the jaxpr, so a
value-dependent zero can never drop an entry; unknown primitives fall back
to the union-of-probes detector; every pounce.jax entry point takes the
pattern / on_failure arguments."""

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pounce.jax import JaxProblem, from_jax, solve_with_warm, vmap_solve
from pounce.jax import _jaxpr_sparsity as J

from test_jax_sparsity_gh985 import (  # noqa: E402
    BECK_KW, CSTR_BOUNDS, P_BECK, _assert_cstr_structure, _beck_f, _beck_g,
    _cstr_f, _cstr_g,
)


def _pairs(rc):
    return set(zip(np.asarray(rc[0]).tolist(), np.asarray(rc[1]).tolist()))


def test_cstr_pattern_is_exact_and_comes_from_the_jaxpr():
    # No x0, no bounds: nothing for a probe to latch onto, and exp(-E/RT)
    # underflows to 0 at every standard-normal T.
    p = from_jax(_cstr_f, _cstr_g, n=4, m=2, cl=np.zeros(2), cu=np.zeros(2))
    assert p.problem_obj.pattern_source == {"jac": "jaxpr", "hess": "jaxpr"}
    _assert_cstr_structure(p)


def test_probe_only_mode_still_available():
    p = from_jax(_cstr_f, _cstr_g, n=4, m=2, pattern_detection="probe",
                 x0=np.array([1.0, 1.0, 10.0, 330.0]), **CSTR_BOUNDS)
    assert p.problem_obj.pattern_source == {"jac": "probe", "hess": "probe"}
    with pytest.raises(ValueError):
        from_jax(_cstr_f, _cstr_g, n=4, m=2, pattern_detection="nope")


def test_beckmann_true_structure_without_hints():
    pj = from_jax(
        lambda x: _beck_f(x, P_BECK), lambda x: _beck_g(x, P_BECK), n=3, m=1,
        cl=np.zeros(1), cu=np.zeros(1),
    )
    assert pj.problem_obj.pattern_source["hess"] == "jaxpr"
    assert _pairs(pj.problem_obj.hessianstructure()) == {(0, 0), (1, 1), (2, 2)}
    assert _pairs(pj.problem_obj.jacobianstructure()) == {(0, 0), (0, 1), (0, 2)}


def test_polynomial_second_derivative_vanishing_at_every_probe():
    # f''(x) = 6*(x - 1) vanishes at the single point x = 1; a model whose
    # coupling is  (x0 - 1) * x1  has H01 = 1 but H00 = 0 identically.
    f = lambda x: (x[0] - 1.0) ** 3 + (x[0] - 1.0) * x[1] ** 2
    p = from_jax(f, n=2, x0=np.ones(2))
    assert _pairs(p.problem_obj.hessianstructure()) == {(0, 0), (1, 0), (1, 1)}


n_ = 6
_A = np.zeros((n_, n_))
for _i in range(n_):
    _A[_i, _i] = 2.0
    if _i + 1 < n_:
        _A[_i, _i + 1] = -1.0
_IDX = np.array([3, 1, 1, 5, 0])
_B = np.random.default_rng(0).standard_normal((3, n_))

CASES = {
    "banded_matvec": lambda x: _A @ x,
    "banded_quad": lambda x: jnp.stack([jnp.sum(x * (_A @ x))]),
    "gather": lambda x: x[_IDX] ** 2,
    "gather_prod": lambda x: jnp.stack([jnp.prod(x[_IDX]), x[0]]),
    "cumsum": lambda x: jnp.cumsum(x ** 2),
    "diff": lambda x: jnp.diff(x) * x[:-1],
    "roll": lambda x: jnp.roll(x, 2) * x,
    "where": lambda x: jnp.where(x > 0, x ** 2, jnp.sin(x)),
    "abs": lambda x: jnp.abs(x)[1:4] * x[:3],
    "outer": lambda x: jnp.outer(x[:2], x[2:5]).ravel(),
    "sum_axis": lambda x: jnp.sum(x.reshape(2, 3) ** 2, axis=0),
    "pad_flip": lambda x: jnp.pad(jnp.flip(x), (1, 2))[::2] * 2,
    "tile": lambda x: jnp.tile(x[:2], 3) * jnp.repeat(x[2:4], 3),
    "dense": lambda x: _B @ x,
    "matmat": lambda x: (x.reshape(2, 3) @ _B[:, :2]).ravel() * x[:4],
    "max": lambda x: jnp.stack([jnp.max(x), jnp.min(x[:3])]),
    "softmax": lambda x: jax.nn.softmax(x)[:3],
    "logsumexp": lambda x: jnp.stack([jax.scipy.special.logsumexp(x)]),
    "relu": lambda x: jax.nn.relu(x) * x,
    "jit_inner": lambda x: jax.jit(lambda y: y[:3] * y[3:])(x),
    "dyn_slice": lambda x: jax.lax.dynamic_slice(x, (2,), (3,)) ** 2,
    "concat": lambda x: jnp.concatenate([x[:2] ** 2, x[4:] * x[:2], x[2:3]]),
    "einsum": lambda x: jnp.einsum("i,j->ij", x[:3], x[3:]).sum(1),
    "checkpoint": lambda x: jax.checkpoint(lambda y: jnp.sin(y[:3]) * y[3:])(x),
    "gelu": lambda x: jax.nn.gelu(x)[:2] * x[2:4],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_jaxpr_pattern_is_a_superset_of_ad_nonzeros(name):
    fn = CASES[name]
    m = int(np.size(fn(jnp.zeros(n_))))
    rng = np.random.default_rng(1)
    pts = [jnp.asarray(rng.standard_normal(n_) * 1.5) for _ in range(12)]

    jp = J.jacobian_pattern(fn, n_, m)
    assert jp is not None, "analysis should bound this model"
    S = np.zeros((m, n_), bool)
    S[jp] = True
    Jf = jax.jacfwd(lambda x: jnp.ravel(fn(x)))
    true = np.zeros_like(S)
    for x in pts:
        true |= np.abs(np.asarray(Jf(x))) > 0
    assert (S | ~true).all(), "jaxpr Jacobian pattern dropped a nonzero"

    w = jnp.arange(1, m + 1) ** 0.5
    h = lambda x: jnp.sum(jnp.ravel(fn(x)) * w)
    hp = J.hessian_lower_pattern(jax.grad(h), (jnp.zeros(n_),), n_)
    assert hp is not None
    HS = np.zeros((n_, n_), bool)
    HS[hp] = True
    assert (np.triu(HS, 1) == 0).all()
    Ht = np.zeros((n_, n_), bool)
    for x in pts:
        Ht |= np.abs(np.asarray(jax.hessian(h)(x))) > 0
    assert (HS | ~np.tril(Ht)).all(), "jaxpr Hessian pattern dropped a nonzero"


def test_banded_constant_matrix_stays_banded():
    jp = J.jacobian_pattern(CASES["banded_matvec"], n_, n_)
    assert len(jp[0]) == int((_A != 0).sum())


def test_unbounded_primitives_fall_back_to_probes():
    scan_g = lambda x: jax.lax.scan(lambda c, xi: (c + xi, c * xi), 0.0, x)[1]
    assert J.jacobian_pattern(scan_g, 4, 4) is None
    f = lambda x: jnp.sum(jnp.sort(x) ** 2)
    p = from_jax(f, n=4, x0=np.arange(4.0))
    assert p.problem_obj.pattern_source["hess"] == "probe"
    # the fallback is the hardened probe, so the diagonal is still found
    assert {(i, i) for i in range(4)} <= _pairs(p.problem_obj.hessianstructure())


def test_large_vectorised_model_is_fast_and_exact():
    n = 20_000
    g = lambda x: x[1:] - x[:-1] - 0.1 * jnp.sin(x[:-1])
    jp = J.jacobian_pattern(g, n, n - 1)
    assert len(jp[0]) == 2 * (n - 1)  # bidiagonal


def test_dependency_budget_falls_back(monkeypatch):
    monkeypatch.setattr(J, "_MAX_NNZ", 10)
    assert J.jacobian_pattern(lambda x: jnp.outer(x, x).ravel(), 6, 36) is None


# ---- the other entry points take the same arguments (gh#985) ----


def test_solve_with_warm_accepts_patterns_and_on_failure():
    kw = dict(BECK_KW)
    x, _ = solve_with_warm(
        P_BECK, jac_pattern=np.nonzero(np.ones((1, 3))),
        hess_pattern=np.tril_indices(3), **kw,
    )
    np.testing.assert_allclose(np.asarray(x).sum(), 2500.0, atol=1e-4)
    # default detection (jaxpr) converges without a warning too
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        x, _ = solve_with_warm(P_BECK, **kw)
    np.testing.assert_allclose(np.asarray(x).sum(), 2500.0, atol=1e-4)

    bad = dict(kw, options={"print_level": 0, "max_iter": 1})
    with pytest.warns(RuntimeWarning, match="did not converge"):
        solve_with_warm(P_BECK, **bad)
    with pytest.raises(Exception, match="did not converge"):
        jax.block_until_ready(solve_with_warm(P_BECK, on_failure="raise", **bad)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solve_with_warm(P_BECK, on_failure="ignore", **bad)
    with pytest.raises(ValueError):
        solve_with_warm(P_BECK, on_failure="bogus", **kw)


def test_vmap_solve_accepts_patterns():
    xb = np.asarray(vmap_solve(
        jnp.stack([P_BECK, P_BECK]), jac_pattern=np.nonzero(np.ones((1, 3))),
        hess_pattern=np.tril_indices(3), **BECK_KW,
    ))
    np.testing.assert_allclose(xb.sum(axis=1), [2500.0, 2500.0], atol=1e-4)


def _jp(**kw):
    return JaxProblem(
        f=_beck_f, g=_beck_g, n=3, m=1, p_example=P_BECK,
        lb=np.zeros(3), ub=np.full(3, 5000.0), cl=np.zeros(1), cu=np.zeros(1),
        options={"print_level": 0, "max_iter": 50}, **kw,
    )


def test_jaxproblem_uses_jaxpr_pattern_and_reports_failures():
    with _jp() as jp:
        assert jp.pattern_source == {"jac": "jaxpr", "hess": "jaxpr"}
        assert _pairs((jp._hess_rows, jp._hess_cols)) == {(0, 0), (1, 1), (2, 2)}
        x = jp.solve(P_BECK, jnp.full(3, 500.0))
        np.testing.assert_allclose(np.asarray(x).sum(), 2500.0, atol=1e-4)
    with _jp(pattern_detection="probe") as jp:
        assert jp.pattern_source == {"jac": "probe", "hess": "probe"}
    short = dict(
        f=_beck_f, g=_beck_g, n=3, m=1, p_example=P_BECK,
        lb=np.zeros(3), ub=np.full(3, 5000.0), cl=np.zeros(1), cu=np.zeros(1),
        options={"print_level": 0, "max_iter": 1},
    )
    with JaxProblem(**short) as jp:
        with pytest.warns(RuntimeWarning, match="did not converge"):
            jp.solve(P_BECK, jnp.full(3, 500.0))
    with JaxProblem(on_failure="raise", **short) as jp:
        with pytest.raises(Exception, match="did not converge"):
            jax.block_until_ready(jp.solve(P_BECK, jnp.full(3, 500.0)))
    with JaxProblem(on_failure="ignore", **short) as jp:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            jp.solve(P_BECK, jnp.full(3, 500.0))
    with pytest.raises(ValueError):
        _jp(on_failure="bogus")
