"""`Problem.get_current_iterate()`, and the objective `intermediate` receives.

An `intermediate` callback reads the live iterate as cyipopt's
`get_current_iterate` gives it, and its `obj_value` is the problem's own
objective at that iterate whatever scaling the solve applies.
"""

import numpy as np
import pytest

import pounce


class HS071:
    def objective(self, x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def gradient(self, x):
        return np.array(
            [
                x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
                x[0] * x[3],
                x[0] * x[3] + 1.0,
                x[0] * (x[0] + x[1] + x[2]),
            ]
        )

    def constraints(self, x):
        return np.array([np.prod(x), np.dot(x, x)])

    def jacobianstructure(self):
        return (np.repeat([0, 1], 4), np.tile([0, 1, 2, 3], 2))

    def jacobian(self, x):
        return np.array(
            [
                x[1] * x[2] * x[3],
                x[0] * x[2] * x[3],
                x[0] * x[1] * x[3],
                x[0] * x[1] * x[2],
                2 * x[0],
                2 * x[1],
                2 * x[2],
                2 * x[3],
            ]
        )


class Recording(HS071):
    """HS071 whose `intermediate` records the statistics and both iterates."""

    def __init__(self):
        self.prob = None
        self.calls = []

    def intermediate(self, **kw):
        self.calls.append(
            (
                kw,
                self.prob.get_current_iterate(),
                self.prob.get_current_iterate(scaled=True),
            )
        )


def solve_hs071(**options):
    obj = Recording()
    prob = pounce.Problem(
        n=4,
        m=2,
        problem_obj=obj,
        lb=[1.0] * 4,
        ub=[5.0] * 4,
        cl=[25.0, 40.0],
        cu=[2e19, 40.0],
    )
    obj.prob = prob
    prob.add_option("print_level", 0)
    prob.add_option("tol", 1e-8)
    for name, value in options.items():
        prob.add_option(name, value)
    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded", info["status_msg"]
    return obj, prob, x, info


def test_each_call_reads_the_iterate_and_the_last_is_the_solution():
    obj, prob, x, info = solve_hs071()
    assert [kw["iter_count"] for kw, _, _ in obj.calls] == list(
        range(info["iter_count"] + 1)
    )
    for kw, it, _ in obj.calls:
        assert kw["alg_mod"] == 0
        assert set(it) == {"x", "mult_x_L", "mult_x_U", "g", "mult_g"}
    _, last, _ = obj.calls[-1]
    np.testing.assert_array_equal(last["x"], x)
    for key in ("g", "mult_g", "mult_x_L", "mult_x_U"):
        np.testing.assert_allclose(last[key], info[key], rtol=1e-12, atol=1e-14)


def test_there_is_no_iterate_outside_the_callback():
    _, prob, _, _ = solve_hs071()
    assert prob.get_current_iterate() is None


@pytest.mark.parametrize(
    "options",
    [{}, {"obj_scaling_factor": 3.5}, {"nlp_scaling_method": "none"}],
    ids=["default", "obj_scaling_factor", "no_scaling"],
)
def test_obj_value_is_the_objective_at_the_iterate(options):
    obj, _, _, info = solve_hs071(**options)
    for kw, it, _ in obj.calls:
        assert kw["obj_value"] == pytest.approx(HS071().objective(it["x"]), rel=1e-12)
    assert obj.calls[-1][0]["obj_value"] == pytest.approx(info["obj_val"], rel=1e-12)


def test_the_scaled_iterate_carries_the_objective_scale():
    # With the variables and rows unscaled, the scaled problem differs from
    # the user's only by the objective factor, which multiplies every
    # multiplier and leaves x and g alone.
    obj, _, _, _ = solve_hs071(nlp_scaling_method="none", obj_scaling_factor=3.5)
    _, it, scaled = obj.calls[-1]
    np.testing.assert_array_equal(scaled["x"], it["x"])
    np.testing.assert_array_equal(scaled["g"], it["g"])
    for key in ("mult_g", "mult_x_L", "mult_x_U"):
        np.testing.assert_allclose(scaled[key], 3.5 * it[key], rtol=1e-12)
