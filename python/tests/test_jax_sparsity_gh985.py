"""gh#985: sparsity probe must not drop structural nonzeros, and the
function API must expose patterns and surface non-converged solves."""

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pounce.jax import from_jax, solve as layer_solve, vmap_solve_parallel

cA0, k10, E1, k20, E2, R = 2.0, 1.0e6, 50_000.0, 5.0e3, 35_000.0, 8.314


def _cstr_g(x):
    cA, cB, tau, T = x[0], x[1], x[2], x[3]
    k1 = k10 * jnp.exp(-E1 / (R * T))
    k2 = k20 * jnp.exp(-E2 / (R * T))
    return jnp.stack([cA0 - cA - tau * k1 * cA, -cB + tau * (k1 * cA - k2 * cB)])


def _cstr_f(x):
    return -(10 * x[1] - 0.02 * x[2])


CSTR_BOUNDS = dict(
    lb=np.array([0, 0, 0.1, 300.0]), ub=np.array([2, 2, 60, 360.0]),
    cl=np.zeros(2), cu=np.zeros(2),
)
CSTR_X0 = np.array([1.0, 1.0, 10.0, 330.0])


def _assert_cstr_structure(p):
    # True structure (the issue's 8/10 are the dense supersets): J has 7
    # entries (row 0 does not touch cB), H has 6 lower entries.
    jr, jc = p.problem_obj.jacobianstructure()
    assert set(zip(jr.tolist(), jc.tolist())) == {
        (0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)}
    hr, hc = p.problem_obj.hessianstructure()
    assert set(zip(hr.tolist(), hc.tolist())) == {
        (2, 0), (3, 0), (2, 1), (3, 1), (3, 2), (3, 3)}


def test_cstr_exp_underflow_pattern_is_complete():
    p = from_jax(_cstr_f, _cstr_g, n=4, m=2, x0=CSTR_X0, **CSTR_BOUNDS)
    _assert_cstr_structure(p)


def test_cstr_pattern_complete_without_x0_via_box_probes():
    # No x0: the in-box probes alone must recover the structure.
    p = from_jax(_cstr_f, _cstr_g, n=4, m=2, **CSTR_BOUNDS)
    _assert_cstr_structure(p)


def test_cstr_solves_to_global_profit():
    p = from_jax(_cstr_f, _cstr_g, n=4, m=2, x0=CSTR_X0, **CSTR_BOUNDS)
    p.add_option("print_level", 0)
    _, info = p.solve(x0=CSTR_X0)
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(-info["obj_val"], 5.3524, atol=1e-3)


cap = 1000 * np.array([1.0, 1.5, 0.8])
t01 = 10.0


def _beck_f(x, p):
    t0 = jnp.stack([t01, p[0], p[1]])
    return jnp.sum(t0 * (x + 0.03 * x ** 5 / cap ** 4))


def _beck_g(x, p):
    return jnp.stack([jnp.sum(x) - p[2]])


P_BECK = jnp.array([12.0, 20.0, 2500.0])
BECK_KW = dict(
    f=_beck_f, g=_beck_g, x0=jnp.full(3, 500.0), n=3, m=1,
    lb=np.zeros(3), ub=np.full(3, 5000.0), cl=np.zeros(1), cu=np.zeros(1),
    options={"print_level": 0, "max_iter": 50},
)


def test_beckmann_hessian_pattern_is_complete():
    pj = from_jax(
        lambda x: _beck_f(x, P_BECK), lambda x: _beck_g(x, P_BECK),
        n=3, m=1, lb=np.zeros(3), ub=np.full(3, 5000.0),
        cl=np.zeros(1), cu=np.zeros(1), x0=np.full(3, 500.0),
    )
    assert len(pj.problem_obj.hessianstructure()[0]) == 3  # diagonal


def test_beckmann_function_api_converges_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        x = np.asarray(layer_solve(P_BECK, **BECK_KW))
    np.testing.assert_allclose(x.sum(), 2500.0, atol=1e-4)
    np.testing.assert_allclose(x, [1192.9, 1307.1, 0.0], atol=0.1)


def test_function_api_accepts_explicit_patterns():
    x = np.asarray(layer_solve(
        P_BECK, jac_pattern=np.nonzero(np.ones((1, 3))),
        hess_pattern=np.tril_indices(3), **BECK_KW,
    ))
    np.testing.assert_allclose(x.sum(), 2500.0, atol=1e-4)
    xb = np.asarray(vmap_solve_parallel(
        jnp.stack([P_BECK, P_BECK]), jac_pattern=np.nonzero(np.ones((1, 3))),
        hess_pattern=np.tril_indices(3), **BECK_KW,
    ))
    np.testing.assert_allclose(xb.sum(axis=1), [2500.0, 2500.0], atol=1e-4)


def _nonconvergent_kw():
    # A deliberately wrong (empty) Hessian pattern plus a tiny iteration
    # budget reproduces a non-converged forward solve.
    kw = dict(BECK_KW)
    kw["options"] = {"print_level": 0, "max_iter": 1}
    return kw


def test_nonconverged_forward_solve_warns_by_default():
    with pytest.warns(RuntimeWarning, match="did not converge"):
        x = layer_solve(P_BECK, **_nonconvergent_kw())
    assert np.asarray(x).shape == (3,)  # return type unchanged


def test_on_failure_raise_and_ignore():
    with pytest.raises(Exception, match="did not converge"):
        jax.block_until_ready(
            layer_solve(P_BECK, on_failure="raise", **_nonconvergent_kw())
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        layer_solve(P_BECK, on_failure="ignore", **_nonconvergent_kw())
    with pytest.raises(ValueError):
        layer_solve(P_BECK, on_failure="bogus", **BECK_KW)


def test_parallel_nonconverged_warns():
    with pytest.warns(RuntimeWarning, match="did not converge"):
        vmap_solve_parallel(
            jnp.stack([P_BECK, P_BECK]), workers=1, **_nonconvergent_kw()
        )
