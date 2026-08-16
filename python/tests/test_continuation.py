"""Frontend-neutral predictor--corrector continuation (pounce#608).

Driven on the same upstream sIPOPT `ParametricTNLP` fixture that
`test_sensitivity.py` and `test_solver_session.py` use, so the tangent
predictor here is checked against a problem whose sensitivity answer is
already pinned to upstream's golden numbers.
"""

import numpy as np
import pytest

import pounce

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


class ParametricNLP:
    """Same NLP as `test_sensitivity.ParametricNLP`. Variables `x[3]`,
    `x[4]` are the parameters `eta1`, `eta2`, pinned by `g[2] = eta1`,
    `g[3] = eta2` -- the sIPOPT pin convention the driver's `pins=` takes."""

    def objective(self, x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    def gradient(self, x):
        return np.array([2 * x[0], 2 * x[1], 2 * x[2], 0.0, 0.0])

    def constraints(self, x):
        x1, x2, x3, eta1, eta2 = x
        return np.array([
            6 * x1 + 3 * x2 + 2 * x3 - eta1,
            eta2 * x1 + x2 - x3 - 1.0,
            eta1,
            eta2,
        ])

    def jacobianstructure(self):
        return (np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 3], dtype=np.int64),
                np.array([0, 1, 2, 3, 0, 1, 2, 4, 3, 4], dtype=np.int64))

    def jacobian(self, x):
        return np.array([6.0, 3.0, 2.0, -1.0,
                         x[4], 1.0, -1.0, x[0],
                         1.0, 1.0])

    def hessianstructure(self):
        return (np.array([0, 1, 2, 4], dtype=np.int64),
                np.array([0, 1, 2, 0], dtype=np.int64))

    def hessian(self, x, lagrange, obj_factor):
        return np.array([2.0 * obj_factor, 2.0 * obj_factor,
                         2.0 * obj_factor, lagrange[1]])


LB = np.array([0.0, 0.0, 0.0, -1e19, -1e19])
UB = np.full(5, 1e19)
X0 = np.array([0.15, 0.15, 0.0, 0.0, 0.0])
PINS = [2, 3]


def bounds_at(theta):
    theta = np.asarray(theta, float)
    rhs = np.array([0.0, 0.0, theta[0], theta[1]])
    return LB, UB, rhs, rhs


class Counter:
    """Counting wrapper with the `reset_counts()` / `counts()` protocol
    the driver's `counter=` argument expects."""

    def __init__(self, inner):
        self._inner = inner
        self.reset_counts()

    def reset_counts(self):
        self.n_obj = self.n_grad = self.n_cons = self.n_jac = self.n_hess = 0

    def counts(self):
        return dict(n_obj=self.n_obj, n_grad=self.n_grad, n_cons=self.n_cons,
                    n_jac=self.n_jac, n_hess=self.n_hess)

    def objective(self, x):
        self.n_obj += 1
        return self._inner.objective(x)

    def gradient(self, x):
        self.n_grad += 1
        return self._inner.gradient(x)

    def constraints(self, x):
        self.n_cons += 1
        return self._inner.constraints(x)

    def jacobian(self, x):
        self.n_jac += 1
        return self._inner.jacobian(x)

    def jacobianstructure(self):
        return self._inner.jacobianstructure()

    def hessian(self, x, lagrange, obj_factor):
        self.n_hess += 1
        return self._inner.hessian(x, lagrange, obj_factor)

    def hessianstructure(self):
        return self._inner.hessianstructure()


def make_update(obj):
    def update(theta):
        lb, ub, cl, cu = bounds_at(theta)
        p = pounce.Problem(n=5, m=4, problem_obj=obj,
                           lb=lb, ub=ub, cl=cl, cu=cu)
        p.add_option("tol", 1e-10)
        p.add_option("print_level", 0)
        p.add_option("sb", "yes")
        return p
    return update


def theta_path(k=8):
    """A short, smooth path in (eta1, eta2) from the fixture's nominal."""
    s = np.linspace(0.0, 1.0, k)
    return [np.array([5.0 - 0.5 * t, 1.0 + 0.2 * t]) for t in s]


def theta_of_s(s):
    return np.array([5.0 - 0.5 * s, 1.0 + 0.2 * s])


def exact_at(theta):
    x, _ = make_update(ParametricNLP())(theta).solve(x0=X0)
    return np.asarray(x, float)


# -- acceptance criterion 1: a sequence traces through `Problem` --------


def test_run_traces_a_parametric_sequence():
    """The whole point: a parametric NLP sequence goes through the
    generic `Problem` API without the caller rebuilding orchestration."""
    path = theta_path()
    trace = pounce.Continuation(make_update(ParametricNLP()),
                                pins=PINS, bounds=bounds_at).run(path, x0=X0)

    assert trace.status == "ok"
    assert trace.n_steps == len(path)
    assert trace.n_corrections == len(path)
    for th, x in zip(path, trace.x):
        assert np.allclose(x, exact_at(th), atol=1e-6)


def test_run_reports_every_counter_the_issue_asks_for():
    obj = Counter(ParametricNLP())
    trace = pounce.Continuation(make_update(obj), pins=PINS,
                                bounds=bounds_at).run(theta_path(), x0=X0,
                                                      counter=obj)
    assert trace.total_evals > 0
    assert trace.total_iters > 0
    assert trace.n_corrections == trace.n_steps
    assert trace.n_predictor_accepts == 0
    assert trace.n_rejections == 0
    assert trace.n_active_set_events >= 0
    assert "active-set events" in trace.report()
    assert all(st.evals > 0 for st in trace.steps)


# -- the tangent predictor, and the documented fallback ----------------


def test_tangent_predictor_is_used_when_pins_are_declared():
    drv = pounce.Continuation(make_update(ParametricNLP()), pins=PINS,
                              bounds=bounds_at)
    assert drv.has_tangent
    trace = drv.run(theta_path(), x0=X0)
    kinds = [st.predictor for st in trace.steps]
    assert kinds[0] == "cold"
    assert set(kinds[1:]) == {"tangent"}


def test_falls_back_to_zero_order_transfer_without_pins():
    """pounce#608: 'falls back to zero-order warm transfer when
    sensitivities are unavailable'."""
    drv = pounce.Continuation(make_update(ParametricNLP()), bounds=bounds_at)
    assert not drv.has_tangent
    trace = drv.run(theta_path(), x0=X0)
    assert trace.status == "ok"
    assert [st.predictor for st in trace.steps][1:] == ["zero"] * 7
    for th, x in zip(theta_path(), trace.x):
        assert np.allclose(x, exact_at(th), atol=1e-6)


def test_tangent_predictor_beats_zero_order_transfer_on_seed_accuracy():
    """The predictor's own claim, measured directly rather than through
    an iteration count: `x_prev + dx` must land nearer the next solution
    than `x_prev` does."""
    path = theta_path()
    obj = ParametricNLP()
    prob = make_update(obj)(path[0])
    solver = pounce.Solver(prob)
    x, _ = solver.solve(x0=X0)
    x = np.asarray(x, float)

    dtheta = path[1] - path[0]
    dx = np.asarray(solver.parametric_step(PINS, list(dtheta)), float)[:5]
    truth = exact_at(path[1])

    err_zero = np.max(np.abs(x - truth))
    err_tangent = np.max(np.abs(x + dx - truth))
    assert err_tangent < 0.2 * err_zero


def test_pin_count_must_match_theta():
    drv = pounce.Continuation(make_update(ParametricNLP()), pins=[2],
                              bounds=bounds_at)
    with pytest.raises(ValueError, match="one-to-one"):
        drv.run(theta_path(), x0=X0)


# -- acceptance criterion: predictor accepts, and what they cost -------


def test_follow_accepts_steps_on_the_predictor_alone():
    """`follow` is where continuation pays: a predicted point under
    `monitor_tol` is taken with no solve at all."""
    obj = ParametricNLP()
    mon = pounce.kkt_residual_monitor(obj, bounds_at)

    tight = pounce.Continuation(make_update(obj), pins=PINS, monitor=mon,
                                bounds=bounds_at, monitor_tol=1e-6)
    loose = pounce.Continuation(make_update(obj), pins=PINS, monitor=mon,
                                bounds=bounds_at, monitor_tol=1e-1)

    t_tight = tight.follow(theta_of_s, (0.0, 1.0), x0=X0)
    t_loose = loose.follow(theta_of_s, (0.0, 1.0), x0=X0)

    assert t_tight.n_predictor_accepts == 0
    assert t_loose.n_predictor_accepts > 0
    assert t_loose.n_corrections < t_tight.n_corrections
    assert t_loose.total_iters < t_tight.total_iters
    # ... and the accuracy that buys is the thing to report, not hide.
    assert np.max(np.abs(t_tight.x[-1] - exact_at(theta_of_s(1.0)))) < 1e-8
    assert np.max(np.abs(t_loose.x[-1] - exact_at(theta_of_s(1.0)))) < 1e-2


def test_follow_without_a_monitor_corrects_every_step():
    """No monitor means no way to know a predicted point is good, so the
    driver must solve rather than guess."""
    trace = pounce.Continuation(make_update(ParametricNLP()), pins=PINS,
                                bounds=bounds_at).follow(
        theta_of_s, (0.0, 1.0), x0=X0)
    assert trace.n_predictor_accepts == 0
    assert trace.n_corrections == trace.n_steps


def test_follow_rejects_a_reversed_span():
    drv = pounce.Continuation(make_update(ParametricNLP()), pins=PINS)
    with pytest.raises(ValueError, match="s1 > s0"):
        drv.follow(theta_of_s, (1.0, 0.0), x0=X0)


# -- the monitor -------------------------------------------------------


def test_kkt_residual_monitor_vanishes_at_a_solution():
    obj = ParametricNLP()
    theta = np.array([5.0, 1.0])
    prob = make_update(obj)(theta)
    x, info = prob.solve(x0=X0)
    r = pounce.kkt_residual_monitor(obj, bounds_at)(
        theta, x, info["mult_g"], info["mult_x_L"], info["mult_x_U"])
    assert r < 1e-6


def test_kkt_residual_monitor_sees_a_displaced_point():
    obj = ParametricNLP()
    theta = np.array([5.0, 1.0])
    prob = make_update(obj)(theta)
    x, info = prob.solve(x0=X0)
    mon = pounce.kkt_residual_monitor(obj, bounds_at)
    bad = np.asarray(x, float) + 0.1
    assert mon(theta, bad, info["mult_g"], info["mult_x_L"],
               info["mult_x_U"]) > 1e-3


# -- transfer / prolongation ------------------------------------------


def test_transfer_map_is_applied_between_steps():
    """pounce#608 asks for 'an explicit user transfer/prolongation map'.
    Protocol is `WarmStart.transfer`'s: mapper(ctx) -> dict."""
    seen = []

    def mapper(ctx):
        seen.append(np.asarray(ctx.source.x, float).copy())
        return {}          # identity, so the trace must be unchanged

    trace = pounce.Continuation(make_update(ParametricNLP()), pins=PINS,
                                bounds=bounds_at,
                                transfer=mapper).run(theta_path(), x0=X0)
    assert trace.status == "ok"
    assert len(seen) == len(theta_path()) - 1     # not on the anchor
    for th, x in zip(theta_path(), trace.x):
        assert np.allclose(x, exact_at(th), atol=1e-6)


def test_transfer_map_rejects_unknown_keys():
    def mapper(ctx):
        return {"nonsense": np.zeros(5)}

    drv = pounce.Continuation(make_update(ParametricNLP()), pins=PINS,
                              bounds=bounds_at, transfer=mapper)
    with pytest.raises(TypeError, match="unknown keys"):
        drv.run(theta_path(), x0=X0)


def test_transfer_map_rejects_a_wrong_length_array():
    # The arity check belongs to `WarmStart.transfer` (pounce#607), which the
    # driver delegates to; this asserts the rejection still reaches the caller
    # through `Continuation.run`, and matches that owner's wording rather than
    # the shim's.
    def mapper(ctx):
        return {"x": np.zeros(4)}

    drv = pounce.Continuation(make_update(ParametricNLP()), pins=PINS,
                              bounds=bounds_at, transfer=mapper)
    with pytest.raises(ValueError, match=r"the mapped 'x' has 4 entries"):
        drv.run(theta_path(), x0=X0)


# -- step controller (shared with pounce.jax.PathFollower, pounce#90) ---


def test_step_controller_grows_on_accept_and_shrinks_on_hard_correction():
    c = pounce.StepController(ds0=0.1, ds_min=1e-3, ds_max=0.4,
                              grow=2.0, shrink=0.5)
    assert c.accepted() == pytest.approx(0.2)
    assert c.accepted() == pytest.approx(0.4)
    assert c.accepted() == pytest.approx(0.4)      # clamped at ds_max
    assert c.corrected(iters=20) == pytest.approx(0.2)
    assert c.corrected(iters=1) == pytest.approx(0.4)
    assert c.corrected(iters=5) == pytest.approx(0.4)   # neither easy nor hard


def test_step_controller_reanchors_after_an_active_set_event():
    c = pounce.StepController(ds0=0.1, ds_min=1e-3, ds_max=0.4, shrink=0.5)
    c.accepted()
    c.accepted()
    assert c.ds == pytest.approx(0.225)
    assert c.corrected(iters=1, active_set_event=True) == pytest.approx(0.1)


def test_step_controller_signals_the_floor():
    c = pounce.StepController(ds0=0.01, ds_min=0.005, shrink=0.5)
    assert c.rejected() == pytest.approx(0.005)
    assert c.rejected() is None


def test_step_controller_is_what_the_jax_follower_uses():
    """The reuse claim in the module docstring, asserted rather than
    left as a comment: pounce#90's follower must import this class."""
    jax_path = pytest.importorskip("pounce.jax._path")
    assert jax_path.StepController is pounce.StepController


# -- misuse ------------------------------------------------------------


def test_update_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        pounce.Continuation(object())


def test_update_returning_none_is_an_error():
    drv = pounce.Continuation(lambda theta: None, pins=PINS)
    with pytest.raises(TypeError, match="must return the Problem"):
        drv.run(theta_path(), x0=X0)


def test_empty_path_is_an_empty_trace():
    trace = pounce.Continuation(make_update(ParametricNLP())).run([])
    assert trace.n_steps == 0
    assert trace.status == "ok"
