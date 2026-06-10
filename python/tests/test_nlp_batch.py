"""Tests for ``pounce.solve_nlp_batch`` (pounce#126) — batched NLP
solving: parallel native path for ``NlProblem`` inputs, sequential
fallback for callback-based ``Problem`` inputs."""

from pathlib import Path

import numpy as np
import pytest

import pounce

# Hand-crafted 5-var / 4-con NLP fixture committed for the CLI tests
# (a transliteration of upstream sIPOPT's parametric_cpp example).
_PARAMETRIC = (
    Path(__file__).resolve().parents[2]
    / "crates" / "pounce-cli" / "tests" / "fixtures" / "parametric.nl"
)

pytestmark = pytest.mark.skipif(
    not _PARAMETRIC.exists(), reason="parametric.nl fixture missing"
)


def _load():
    return pounce.read_nl(str(_PARAMETRIC))


# ---------------------------------------------------------------------
# Native (NlProblem) path
# ---------------------------------------------------------------------


def test_empty_batch_returns_empty_list():
    assert pounce.solve_nlp_batch([]) == []


def test_single_element_batch():
    p = _load()
    results = pounce.solve_nlp_batch([p])
    assert len(results) == 1
    x, info = results[0]
    assert info["status_msg"] == "Solve_Succeeded"
    assert x.shape == (p.n,)
    assert np.all(np.isfinite(x))
    assert info["iter_count"] > 0


def test_batch_results_in_input_order_and_match_sequential():
    p = _load()
    # base, multi-start (shifted x0), and a bound-tightened sibling —
    # the branch-and-bound node shape.
    shifted = p.variant(x0=np.asarray(p.x0) + 0.1)
    base_x, _ = pounce.solve_nlp_batch([p])[0]
    xu = np.asarray(p.x_u).copy()
    xu[0] = base_x[0] - 0.05
    tightened = p.variant(x_u=xu)

    batch = [p, shifted, tightened]
    par = pounce.solve_nlp_batch(batch)
    seq = pounce.solve_nlp_batch(batch, parallel=False)
    assert len(par) == len(seq) == 3

    for i, ((xp, ip), (xs, _)) in enumerate(zip(par, seq)):
        assert ip["status_msg"] == "Solve_Succeeded", f"instance {i}"
        # Parallel and sequential agree on the same instance.
        np.testing.assert_allclose(xp, xs, rtol=0, atol=1e-12)

    # Instance 0 reproduces the standalone solve.
    np.testing.assert_array_equal(par[0][0], base_x)
    # Instance 1 (multi-start) reaches the same optimum.
    np.testing.assert_allclose(par[1][0], base_x, atol=1e-5)
    # Instance 2 honors its tightened bound (modulo the IPM's
    # bound_relax_factor slack) and differs from instance 0.
    assert par[2][0][0] <= xu[0] + 1e-7
    assert abs(par[2][0][0] - base_x[0]) > 1e-6


def test_x0s_override_per_instance():
    p = _load()
    x0 = np.asarray(p.x0)
    results = pounce.solve_nlp_batch([p, p], x0s=[None, x0 + 0.1])
    assert all(info["status_msg"] == "Solve_Succeeded" for _, info in results)
    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-5)


def test_x0s_length_mismatch_raises():
    p = _load()
    with pytest.raises(ValueError, match="starting points"):
        pounce.solve_nlp_batch([p, p], x0s=[np.asarray(p.x0)])


def test_infeasible_instance_does_not_poison_batch():
    p = _load()
    # Contradictory variable bounds: x_l > x_u on the first variable.
    xl = np.asarray(p.x_l).copy()
    xu = np.asarray(p.x_u).copy()
    xl[0], xu[0] = 2.0, 1.0
    bad = p.variant(x_l=xl, x_u=xu)
    results = pounce.solve_nlp_batch([p, bad, p])
    assert results[0][1]["status_msg"] == "Solve_Succeeded"
    assert results[2][1]["status_msg"] == "Solve_Succeeded"
    assert results[1][1]["status_msg"] != "Solve_Succeeded"
    np.testing.assert_array_equal(results[0][0], results[2][0])


def test_unknown_option_raises():
    p = _load()
    with pytest.raises(RuntimeError, match="no_such_option"):
        pounce.solve_nlp_batch([p], options={"no_such_option": 1})


def test_options_are_applied():
    p = _load()
    # An absurdly tight iteration cap must change the outcome.
    (x, info), = pounce.solve_nlp_batch([p], options={"max_iter": 1})
    assert info["status_msg"] == "Maximum_Iterations_Exceeded"
    assert info["iter_count"] <= 1


def test_info_dict_layout_matches_problem_solve():
    p = _load()
    (x, info), = pounce.solve_nlp_batch([p])
    for key in (
        "status", "status_msg", "obj_val", "g", "mult_g",
        "mult_x_L", "mult_x_U", "iter_count", "mu",
        "final_kkt_error", "final_dual_inf", "final_constr_viol",
        "final_compl",
    ):
        assert key in info, key
    assert info["g"].shape == (p.m,)
    assert info["mult_g"].shape == (p.m,)
    assert info["mult_x_L"].shape == (p.n,)
    assert info["mult_x_U"].shape == (p.n,)


def test_variant_validates_lengths():
    p = _load()
    with pytest.raises(ValueError, match="x0"):
        p.variant(x0=[1.0])


def test_native_warm_start_chain():
    """MPC-style chain: solve cold, re-solve a perturbed batch seeded
    from the previous results. Warm solves converge to the perturbed
    optimum without iterating more than the cold solves."""
    p = _load()
    batch = [p, p.variant(x0=np.asarray(p.x0) + 0.05)]
    cold = pounce.solve_nlp_batch(batch)

    xu = np.asarray(p.x_u).copy()
    xu[0] = cold[0][0][0] + 0.5  # loose bound: same optimum, new instance
    perturbed = [q.variant(x_u=xu) for q in batch]
    warm = pounce.solve_nlp_batch(perturbed, warms=cold)
    cold2 = pounce.solve_nlp_batch(perturbed)
    for i, ((xw, iw), (xc, ic)) in enumerate(zip(warm, cold2)):
        assert iw["status_msg"] == "Solve_Succeeded", f"instance {i}"
        np.testing.assert_allclose(xw, xc, atol=1e-6)
        assert iw["iter_count"] <= ic["iter_count"], (
            f"instance {i}: warm {iw['iter_count']} > cold {ic['iter_count']}"
        )


def test_share_structure_matches_default_within_tolerance():
    """The identical-sparsity backend pool returns the same solutions
    (within solver tolerance) as fresh-per-instance backends."""
    p = _load()
    rng = np.random.default_rng(1)
    batch = [p.variant(x0=np.asarray(p.x0) + rng.normal(0, 0.01, p.n))
             for _ in range(6)]
    fresh = pounce.solve_nlp_batch(batch)
    pooled = pounce.solve_nlp_batch(batch, share_structure=True)
    for i, ((xf, inf_f), (xp, inf_p)) in enumerate(zip(fresh, pooled)):
        assert inf_f["status_msg"] == "Solve_Succeeded", f"instance {i}"
        assert inf_p["status_msg"] == "Solve_Succeeded", f"instance {i}"
        np.testing.assert_allclose(xp, xf, atol=1e-6)


def test_warms_validation():
    p = _load()
    results = pounce.solve_nlp_batch([p])
    # Wrong length.
    with pytest.raises(ValueError, match="warms"):
        pounce.solve_nlp_batch([p, p], warms=results)
    # Missing dual keys.
    x, info = results[0]
    with pytest.raises(ValueError, match="mult_g"):
        pounce.solve_nlp_batch([p], warms=[(x, {"obj_val": 0.0})])


# ---------------------------------------------------------------------
# Callback (Problem) path — parallel with per-callback GIL (phase 2)
# ---------------------------------------------------------------------


class _HS071:
    def objective(self, x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def gradient(self, x):
        return np.array([
            x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ])

    def constraints(self, x):
        return np.array([np.prod(x), np.dot(x, x)])

    def jacobianstructure(self):
        return (np.repeat([0, 1], 4), np.tile([0, 1, 2, 3], 2))

    def jacobian(self, x):
        return np.array([
            x[1] * x[2] * x[3],
            x[0] * x[2] * x[3],
            x[0] * x[1] * x[3],
            x[0] * x[1] * x[2],
            2 * x[0],
            2 * x[1],
            2 * x[2],
            2 * x[3],
        ])


def _hs071_problem():
    return pounce.Problem(
        n=4, m=2, problem_obj=_HS071(),
        lb=[1.0] * 4, ub=[5.0] * 4,
        cl=[25.0, 40.0], cu=[2e19, 40.0],
    )


def test_problem_batch_parallel_matches_individual_solves():
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    k = 6
    batch = pounce.solve_nlp_batch(
        [_hs071_problem() for _ in range(k)],
        x0s=[x0] * k,
        options={"tol": 1e-8},
    )
    assert len(batch) == k

    ref = _hs071_problem()
    ref.add_option("tol", 1e-8)
    ref.add_option("print_level", 0)
    x_ref, info_ref = ref.solve(x0=x0)

    for i, (x, info) in enumerate(batch):
        assert info["status_msg"] == "Solve_Succeeded", f"instance {i}"
        np.testing.assert_allclose(info["obj_val"], 17.0140172, rtol=1e-5)
        np.testing.assert_array_equal(x, x_ref)
        assert info["iter_count"] == info_ref["iter_count"]
    # All instances identical → identical iterates regardless of which
    # worker ran them.
    np.testing.assert_array_equal(batch[0][0], batch[k - 1][0])


def test_problem_batch_parallel_equals_sequential():
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    probs = lambda: [_hs071_problem(), _hs071_problem()]  # noqa: E731
    par = pounce.solve_nlp_batch(probs(), x0s=[x0, x0], parallel=True)
    seq = pounce.solve_nlp_batch(probs(), x0s=[x0, x0], parallel=False)
    for (xp, ip), (xs, _) in zip(par, seq):
        assert ip["status_msg"] == "Solve_Succeeded"
        np.testing.assert_array_equal(xp, xs)


def test_problem_batch_honors_per_instance_options():
    """Each Problem's own add_option settings apply to its instance."""
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    capped = _hs071_problem()
    capped.add_option("max_iter", 1)
    results = pounce.solve_nlp_batch(
        [_hs071_problem(), capped], x0s=[x0, x0]
    )
    assert results[0][1]["status_msg"] == "Solve_Succeeded"
    assert results[1][1]["status_msg"] == "Maximum_Iterations_Exceeded"


def test_problem_batch_raising_callback_does_not_poison_batch():
    """A Python exception inside one instance's callback degrades to an
    eval failure for that instance only."""

    class Exploding(_HS071):
        def objective(self, x):
            raise RuntimeError("boom")

    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    bad = pounce.Problem(
        n=4, m=2, problem_obj=Exploding(),
        lb=[1.0] * 4, ub=[5.0] * 4,
        cl=[25.0, 40.0], cu=[2e19, 40.0],
    )
    results = pounce.solve_nlp_batch(
        [_hs071_problem(), bad, _hs071_problem()], x0s=[x0] * 3
    )
    assert results[0][1]["status_msg"] == "Solve_Succeeded"
    assert results[2][1]["status_msg"] == "Solve_Succeeded"
    assert results[1][1]["status_msg"] != "Solve_Succeeded"


def test_problem_batch_warm_start_chain():
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    probs = lambda: [_hs071_problem(), _hs071_problem()]  # noqa: E731
    cold = pounce.solve_nlp_batch(probs(), x0s=[x0, x0])
    warm = pounce.solve_nlp_batch(probs(), x0s=[x0, x0], warms=cold)
    for i, ((xw, iw), (xc, ic)) in enumerate(zip(warm, cold)):
        assert iw["status_msg"] == "Solve_Succeeded", f"instance {i}"
        np.testing.assert_allclose(xw, xc, atol=1e-6)
        assert iw["iter_count"] <= ic["iter_count"]


def test_problem_inputs_require_x0s():
    with pytest.raises(ValueError, match="x0s is required"):
        pounce.solve_nlp_batch([_hs071_problem()])


def test_mixed_inputs_raise():
    with pytest.raises(TypeError, match="mixed"):
        pounce.solve_nlp_batch(
            [_load(), _hs071_problem()],
            x0s=[None, np.ones(4)],
        )
