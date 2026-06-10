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


# ---------------------------------------------------------------------
# Callback (Problem) path — sequential fallback
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


def test_problem_inputs_solve_sequentially():
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    results = pounce.solve_nlp_batch(
        [_hs071_problem(), _hs071_problem()],
        x0s=[x0, x0],
        options={"print_level": 0, "tol": 1e-8},
    )
    assert len(results) == 2
    for x, info in results:
        assert info["status_msg"] == "Solve_Succeeded"
        np.testing.assert_allclose(info["obj_val"], 17.0140172, rtol=1e-5)
    np.testing.assert_array_equal(results[0][0], results[1][0])


def test_problem_inputs_require_x0s():
    with pytest.raises(ValueError, match="x0s is required"):
        pounce.solve_nlp_batch([_hs071_problem()])


def test_mixed_inputs_raise():
    with pytest.raises(TypeError, match="mixed"):
        pounce.solve_nlp_batch(
            [_load(), _hs071_problem()],
            x0s=[None, np.ones(4)],
        )
