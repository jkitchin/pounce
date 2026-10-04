"""Tests for pounce.trf (trust-region filter, glass box/black box optimization).

Run quietly: the minimize facade logs a harmless jacobian AttributeError per
solve; RUST_LOG=off silences it.
"""

import os

os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

from pounce.trf import (
    CorrectedSurrogate,
    FilterPoint,
    QuadraticBasis,
    TRFilter,
    ZeroBasis,
    central_difference_design,
    forward_difference_design,
    quadratic_design,
    trf_minimize,
)


class AffineBasis:
    """An affine basis, kept here only to prove it is useless.

    ``pounce.trf`` deliberately ships no linear basis; see
    ``test_affine_basis_cancels_under_zoc_foc``.
    """

    def __init__(self, a, B, w_ref):
        self.a = np.atleast_1d(np.asarray(a, float))
        self.B = np.atleast_2d(np.asarray(B, float))
        self.w_ref = np.asarray(w_ref, float).ravel()

    def fit(self, W, Y):
        return self

    def predict(self, w):
        return self.a + self.B @ (np.asarray(w, float).ravel() - self.w_ref)

    def jacobian(self, w):
        return self.B


class ConstantBasis:
    """Records how many times it was fitted, so freezing can be observed."""

    def __init__(self, n_y, n_w, value=0.0):
        self.n_y, self.n_w = n_y, n_w
        self.value = value
        self.n_fits = 0

    def fit(self, W, Y):
        self.n_fits += 1
        return self

    def predict(self, w):
        return np.full(self.n_y, self.value)

    def jacobian(self, w):
        return np.zeros((self.n_y, self.n_w))


# --- Test problems ---------------------------------------------------------


def sine_problem():
    """min (z-1)^2 + x^2  s.t.  z = sin(x), with sin() as the black box.

    Stationary point satisfies 2(sin x - 1)cos x + 2x = 0.
    """
    return dict(
        fun=lambda v: (v[1] - 1.0) ** 2 + v[0] ** 2,
        x0=[0.5, 0.0],
        truth_model=lambda w: np.sin(w),
        w_index=[0],
        y_index=[1],
        jac=lambda v: np.array([2 * v[0], 2 * (v[1] - 1.0)]),
        truth_jac=lambda w: np.cos(w).reshape(1, 1),
    )


SINE_FUN = 0.520078332823


def cubic_problem():
    """Biegler (2024) Sec. 2: min x1^2 + x2^2  s.t.  x2 = x1^3 + x1^2 + 1.

    Global minimum is (0, 1) with f = 1. A surrogate loop that matches only
    *values* at the base point walks away from it to the local maximum (-1, 1),
    f = 2. This is the failure mode the whole method exists to prevent.
    """
    return dict(
        fun=lambda v: v[0] ** 2 + v[1] ** 2,
        truth_model=lambda w: np.array([w[0] ** 3 + w[0] ** 2 + 1.0]),
        w_index=[0],
        y_index=[1],
        jac=lambda v: np.array([2 * v[0], 2 * v[1]]),
        truth_jac=lambda w: np.array([[3 * w[0] ** 2 + 2 * w[0]]]),
    )


def eason_example1():
    """Eason & Biegler (2018), as shipped in pyomo.contrib.trustregion.

    Variable vector is [z0, z1, z2, x0, x1, y] where y is the external-function
    output. The black box is sin(x0 - x1); the degrees of freedom are z.
    """
    s2 = np.sqrt(2.0)

    def fun(v):
        return (
            (v[0] - 1) ** 2
            + (v[0] - v[1]) ** 2
            + (v[2] - 1) ** 2
            + (v[3] - 1) ** 4
            + (v[4] - 1) ** 6
        )

    def jac(v):
        g = np.zeros(6)
        g[0] = 2 * (v[0] - 1) + 2 * (v[0] - v[1])
        g[1] = -2 * (v[0] - v[1])
        g[2] = 2 * (v[2] - 1)
        g[3] = 4 * (v[3] - 1) ** 3
        g[4] = 6 * (v[4] - 1) ** 5
        return g

    def cons(v):
        return np.array(
            [v[3] * v[0] ** 2 + v[5] - 2 * s2, v[2] ** 4 * v[1] ** 2 + v[1] - (8 + s2)]
        )

    def cons_jac(v):
        J = np.zeros((2, 6))
        J[0, 0] = 2 * v[3] * v[0]
        J[0, 3] = v[0] ** 2
        J[0, 5] = 1.0
        J[1, 1] = 2 * v[2] ** 4 * v[1] + 1.0
        J[1, 2] = 4 * v[2] ** 3 * v[1] ** 2
        return J

    return dict(
        fun=fun,
        x0=[2.0, 2.0, 2.0, 2.0, 1.0, np.sin(1.0)],
        truth_model=lambda w: np.array([np.sin(w[0] - w[1])]),
        w_index=[3, 4],
        y_index=[5],
        jac=jac,
        truth_jac=lambda w: np.array(
            [[np.cos(w[0] - w[1]), -np.cos(w[0] - w[1])]]
        ),
        dof_index=[0, 1, 2],
        constraints=[{"type": "eq", "fun": cons, "jac": cons_jac}],
    )


# Reference values from pyomo.contrib.trustregion driven with both ipopt and
# pounce 0.9.0 as the subproblem solver; the two agreed to ~1e-16 on the
# objective. See dev-notes / the Phase 0 oracle for the full sweep.
ORACLE_EX1_FUN = 0.27704478876374156
ORACLE_EX1_Z = np.array([1.286931683273266, 1.4802348006326804, 1.3794547317539292])
ORACLE_EX1_X = np.array([1.3219485331078022, 0.628718497180208])


# --- Filter ----------------------------------------------------------------


class TestFilter:
    def test_empty_filter_accepts_anything(self):
        f = TRFilter()
        assert f.is_acceptable(FilterPoint(1e3, 1e3))

    def test_rejects_above_theta_max(self):
        f = TRFilter(theta_max=1.0)
        assert not f.is_acceptable(FilterPoint(2.0, -1e6))

    def test_current_iterate_is_tested_with_empty_filter(self):
        """gh#989: a trial raising theta tenfold over the current iterate."""
        f = TRFilter()
        cur = FilterPoint(1e-2, 5.0)
        worse = FilterPoint(1e-1, 5.0)
        assert f.is_acceptable(worse)  # legacy behaviour: empty filter
        assert not f.is_acceptable(worse, current=cur)
        assert f.is_acceptable(FilterPoint(1e-1, 4.0), current=cur)
        assert f.is_acceptable(FilterPoint(1e-3, 6.0), current=cur)

    def test_dominated_point_rejected(self):
        f = TRFilter(gamma_theta=0.0, gamma_f=0.0)
        f.add(FilterPoint(1.0, 1.0))
        # worse in both measures -> dominated
        assert not f.is_acceptable(FilterPoint(2.0, 2.0))
        # better in at least one -> acceptable under plain dominance
        assert f.is_acceptable(FilterPoint(0.5, 2.0))
        assert f.is_acceptable(FilterPoint(2.0, 0.5))

    def test_margins_are_stricter_than_plain_dominance(self):
        """A point that plain dominance accepts can fail the envelope test."""
        strict = TRFilter(gamma_theta=0.1, gamma_f=0.1)
        loose = TRFilter(gamma_theta=0.0, gamma_f=0.0)
        for f in (strict, loose):
            f.add(FilterPoint(1.0, 1.0))
        # theta improves by only 5%, objective is unchanged: passes plain
        # dominance (strictly better in theta) but not a 10% envelope.
        pt = FilterPoint(0.95, 1.0)
        assert loose.is_acceptable(pt)
        assert not strict.is_acceptable(pt)

    def test_add_prunes_dominated_entries(self):
        f = TRFilter(gamma_theta=0.0, gamma_f=0.0)
        # mutually non-dominated: each is better than the other in one measure
        f.add(FilterPoint(1.0, 3.0))
        f.add(FilterPoint(3.0, 1.0))
        assert len(f) == 2
        f.add(FilterPoint(0.5, 0.5))  # dominates both
        assert len(f) == 1
        assert next(iter(f)).theta == 0.5

    def test_add_skips_dominated_point(self):
        f = TRFilter(gamma_theta=0.0, gamma_f=0.0)
        f.add(FilterPoint(1.0, 1.0))
        assert f.add(FilterPoint(2.0, 2.0)) is False
        assert len(f) == 1

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_rejects_invalid_gammas(self, bad):
        with pytest.raises(ValueError):
            TRFilter(gamma_theta=bad)
        with pytest.raises(ValueError):
            TRFilter(gamma_f=bad)

    def test_rejects_nonpositive_theta_max(self):
        with pytest.raises(ValueError):
            TRFilter(theta_max=0.0)


# --- Sampling designs ------------------------------------------------------


class TestDesigns:
    def test_forward_difference_shape_and_base_point(self):
        w0 = np.array([1.0, 2.0, 3.0])
        D = forward_difference_design(w0, 0.5)
        assert D.shape == (4, 3)
        np.testing.assert_allclose(D[0], w0)
        np.testing.assert_allclose(D[1] - w0, [0.5, 0, 0])

    def test_central_difference_is_symmetric(self):
        w0 = np.array([1.0, 2.0])
        D = central_difference_design(w0, 0.25)
        assert D.shape == (5, 2)
        np.testing.assert_allclose(D[1] - w0, -(D[3] - w0))


# --- Surrogates ------------------------------------------------------------


class TestSurrogate:
    def test_affine_basis_cancels_under_zoc_foc(self):
        """An affine basis contributes *nothing*: only curvature survives.

        With ``rbar(w) = a + B(w - w_ref)`` the Jacobian is B everywhere, so the
        basis-dependent part of the corrected surrogate,
        ``rbar(w) - rbar(w_k) - B(w - w_k)``, is identically zero. This is why
        there is no ``LinearBasis`` -- it would burn n_w+1 truth-model calls per
        iteration to reproduce ``ZeroBasis`` exactly. Any new basis that is
        affine in ``w`` is dead weight for the same reason.
        """
        rng = np.random.default_rng(0)
        w_k = np.array([0.7, -0.2])
        d_k = np.array([0.93])
        Jd_k = np.array([[1.1, -0.8]])

        zero = CorrectedSurrogate(ZeroBasis(1, 2), w_k, d_k, Jd_k)
        affine = CorrectedSurrogate(
            AffineBasis(a=rng.normal(size=1), B=rng.normal(size=(1, 2)), w_ref=w_k),
            w_k,
            d_k,
            Jd_k,
        )
        for probe in rng.normal(size=(25, 2)) * 5.0:
            np.testing.assert_allclose(
                affine.predict(probe), zero.predict(probe), atol=1e-12
            )
            np.testing.assert_allclose(
                affine.jacobian(probe), zero.jacobian(probe), atol=1e-14
            )

    def test_quadratic_basis_recovers_a_quadratic_function(self):
        def truth(w):
            return np.array([w[0] ** 2 + 3 * w[0] * w[1] - w[1] ** 2 + 2 * w[0] + 1])

        W = quadratic_design(np.array([0.5, -0.25]), 0.2)
        Y = np.vstack([truth(w) for w in W])
        fit = QuadraticBasis().fit(W, Y)
        for probe in ([0.6, -0.3], [0.4, -0.1], [1.5, 2.0]):
            np.testing.assert_allclose(
                fit.predict(np.array(probe)), truth(np.array(probe)), atol=1e-8
            )

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_quadratic_design_has_exactly_enough_points(self, n):
        assert quadratic_design(np.zeros(n), 0.1).shape == ((n + 1) * (n + 2) // 2, n)

    def test_central_design_alone_cannot_identify_cross_terms(self):
        """Regression: a central design is rank-deficient for a full quadratic.

        It supplies 2n+1 points against the (n+1)(n+2)/2 a quadratic needs, so
        every cross term is unidentified. lstsq still returns the minimum-norm
        solution, which means the fit is silently wrong rather than an error --
        and at n=1 the two designs coincide, so a 1-D test cannot catch it.
        """

        def truth(w):
            return np.array([5.0 * w[0] * w[1]])  # pure cross term

        w0 = np.array([0.3, -0.2])
        probe = np.array([1.0, -1.0])

        good = QuadraticBasis().fit(
            quadratic_design(w0, 0.15),
            np.vstack([truth(w) for w in quadratic_design(w0, 0.15)]),
        )
        np.testing.assert_allclose(good.predict(probe), truth(probe), atol=1e-8)

        starved = QuadraticBasis().fit(
            central_difference_design(w0, 0.15),
            np.vstack([truth(w) for w in central_difference_design(w0, 0.15)]),
        )
        assert abs(starved.predict(probe)[0] - truth(probe)[0]) > 1.0

    def test_zoc_foc_identities_hold_exactly(self):
        """The whole point of the correction: value AND slope match at w_k."""
        w_k = np.array([0.3, -0.4])
        d_k = np.array([1.25])
        Jd_k = np.array([[2.0, -3.0]])
        # deliberately a terrible basis: constant 7, nowhere near the truth
        basis = ConstantBasis(n_y=1, n_w=2, value=7.0)
        r = CorrectedSurrogate(basis, w_k, d_k, Jd_k)
        np.testing.assert_allclose(r.predict(w_k), d_k, atol=1e-12)
        np.testing.assert_allclose(r.jacobian(w_k), Jd_k, atol=1e-12)

    def test_zoc_foc_identities_hold_for_zero_basis(self):
        w_k = np.array([1.0])
        d_k = np.array([2.0, 3.0])
        Jd_k = np.array([[0.5], [-0.5]])
        r = CorrectedSurrogate(ZeroBasis(2, 1), w_k, d_k, Jd_k)
        np.testing.assert_allclose(r.predict(w_k), d_k, atol=1e-12)
        np.testing.assert_allclose(r.jacobian(w_k), Jd_k, atol=1e-12)
        # with a zero basis the surrogate is exactly the linearization
        w = np.array([1.7])
        np.testing.assert_allclose(r.predict(w), d_k + Jd_k @ (w - w_k), atol=1e-12)


# --- The algorithm ---------------------------------------------------------


class TestTRFMinimize:
    def test_sine_problem_converges(self):
        res = trf_minimize(**sine_problem(), max_iterations=40)
        assert res.success
        assert res.fun == pytest.approx(SINE_FUN, abs=1e-8)
        # the surrogate constraint is satisfied against the TRUTH model
        assert res.x[1] == pytest.approx(np.sin(res.x[0]), abs=1e-8)
        # and the point is stationary for the real problem
        x = res.x[0]
        assert 2 * (np.sin(x) - 1) * np.cos(x) + 2 * x == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("basis", ["zero", "quadratic"])
    @pytest.mark.parametrize("analytic_jac", [True, False])
    def test_all_bases_and_jacobian_modes_agree(self, basis, analytic_jac):
        kw = sine_problem()
        if not analytic_jac:
            kw["truth_jac"] = None
        res = trf_minimize(**kw, basis=basis, max_iterations=40)
        assert res.success
        assert res.fun == pytest.approx(SINE_FUN, abs=1e-7)

    def test_does_not_walk_away_from_the_optimum(self):
        """Biegler (2024) Sec. 2 -- the motivating failure mode.

        Started exactly at the global optimum, a value-matching surrogate loop
        diverges to the local maximum at (-1, 1). TRF must stay.
        """
        res = trf_minimize(
            **cubic_problem(), x0=[0.0, 1.0], max_iterations=40
        )
        assert res.success
        np.testing.assert_allclose(res.x, [0.0, 1.0], atol=1e-7)
        assert res.fun == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize("start", [[0.6, 1.5], [-0.5, 0.8], [1.2, 3.0]])
    def test_finds_the_optimum_from_a_displaced_start(self, start):
        res = trf_minimize(**cubic_problem(), x0=start, max_iterations=120)
        assert res.success, res.message
        x1, x2 = res.x
        # feasible against the truth model
        assert x2 == pytest.approx(x1**3 + x1**2 + 1.0, abs=1e-7)
        # and genuinely first-order critical for the real problem. Checking
        # only feasibility here would pass at points with a gradient of order
        # one -- see the criticality discussion in _loop.py.
        true_grad = 2 * x1 + 2 * (x1**3 + x1**2 + 1) * (3 * x1**2 + 2 * x1)
        assert true_grad == pytest.approx(0.0, abs=1e-3)
        assert res.fun == pytest.approx(1.0, abs=1e-6)

    def test_matches_the_pyomo_oracle_on_eason_example1(self):
        res = trf_minimize(**eason_example1(), max_iterations=50)
        assert res.success
        assert res.fun == pytest.approx(ORACLE_EX1_FUN, abs=1e-9)
        np.testing.assert_allclose(res.x[:3], ORACLE_EX1_Z, atol=1e-5)
        np.testing.assert_allclose(res.x[3:5], ORACLE_EX1_X, atol=1e-5)

    def test_truth_model_calls_are_cached(self):
        """Repeated evaluation at an identical w must not re-run the black box."""
        calls = {"n": 0}

        def counting_truth(w):
            calls["n"] += 1
            return np.sin(w)

        kw = sine_problem()
        kw["truth_model"] = counting_truth
        res = trf_minimize(**kw, max_iterations=40)
        # n_truth_evals reports *cache misses*, which is what should be
        # comparable to published call counts; the raw callable must not have
        # been invoked more often than that.
        assert calls["n"] == res.n_truth_evals

    def test_history_is_recorded(self):
        res = trf_minimize(**sine_problem(), max_iterations=40)
        assert len(res.history) == res.nit
        assert {h.kind for h in res.history} <= {
            "f-step",
            "theta-step",
            "rejected",
            "incompatible",
            "subproblem-failed",
        }
        assert all(h.theta >= 0.0 for h in res.history)

    def test_infeasible_glass_box_constraint_does_not_crash(self):
        """A subproblem the solver cannot solve must not be silently accepted.

        Here the user constraints are contradictory, so every subproblem fails.
        The loop should shrink the trust region, give up cleanly, and report
        failure -- not feed the solver's abandoned iterate into the filter as
        though it were a step.
        """
        kw = sine_problem()
        res = trf_minimize(
            **kw,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda v: np.array([v[0] - 5.0, v[0] + 5.0]),
                    "jac": lambda v: np.array([[1.0, 0.0], [1.0, 0.0]]),
                }
            ],
            max_iterations=25,
        )
        assert not res.success
        assert np.all(np.isfinite(res.x))

    def test_bounds_are_respected(self):
        kw = sine_problem()
        res = trf_minimize(**kw, bounds=[(0.6, 2.0), (None, None)], max_iterations=40)
        # The interior-point solver relaxes bounds by `bound_relax_factor`
        # (Ipopt default 1e-8) before solving, so the converged point may sit
        # just outside by roughly that much. Anything tighter than ~1e-7 here
        # is testing the solver's bound-relaxation setting, not this module.
        assert res.x[0] >= 0.6 - 1e-7
        assert res.x[0] <= 2.0 + 1e-7

    def test_rejects_unknown_basis(self):
        with pytest.raises(ValueError, match="unknown basis"):
            trf_minimize(**sine_problem(), basis="not-a-basis")

    def test_linear_basis_is_rejected_with_an_explanation(self):
        """'linear' used to exist; the error must say why it does not."""
        with pytest.raises(ValueError, match="cancels identically"):
            trf_minimize(**sine_problem(), basis="linear")

    def test_rejects_unknown_option(self):
        with pytest.raises(TypeError, match="unknown option"):
            trf_minimize(**sine_problem(), definitely_not_an_option=1)


class TestFrozenBasis:
    """A pre-fitted basis must be reused, not silently re-fitted.

    Freezing is the pattern that pays on an expensive truth model: a basis
    fitted once from a designed dataset costs nothing per iteration, so each
    iteration needs only the base-point evaluation. ZOC/FOC re-anchors the
    fixed basis to the truth model at each new point, which is what makes this
    sound.
    """

    def test_user_supplied_basis_is_frozen_by_default(self):
        basis = ConstantBasis(n_y=1, n_w=1, value=0.0)
        res = trf_minimize(**sine_problem(), basis=basis, max_iterations=40)
        assert res.success
        assert basis.n_fits == 0, "a user-supplied basis must not be re-fitted"

    def test_refit_can_be_forced_on(self):
        basis = ConstantBasis(n_y=1, n_w=1, value=0.0)
        trf_minimize(
            **sine_problem(), basis=basis, refit_basis=True, max_iterations=40
        )
        assert basis.n_fits > 0

    def test_freezing_a_string_basis_removes_its_sampling_cost(self):
        """`quadratic` frozen after one fit should cost fewer truth calls."""
        refit = trf_minimize(**sine_problem(), basis="quadratic", max_iterations=60)
        # Pre-fit a quadratic once, then freeze it.
        w0 = np.array([0.5])
        design = quadratic_design(w0, 0.05)
        fitted = QuadraticBasis().fit(design, np.vstack([np.sin(w) for w in design]))
        frozen = trf_minimize(
            **sine_problem(), basis=fitted, max_iterations=60
        )
        assert refit.success and frozen.success
        assert frozen.fun == pytest.approx(refit.fun, abs=1e-6)
        assert frozen.n_truth_evals < refit.n_truth_evals

    def test_frozen_basis_still_converges_to_the_truth_model(self):
        """A frozen basis fitted somewhere else must not bias the answer.

        ZOC/FOC re-anchors it every iteration, so even a basis fitted far from
        the solution has to land on the same point.
        """
        w_far = np.array([3.0])
        design = quadratic_design(w_far, 0.2)
        stale = QuadraticBasis().fit(design, np.vstack([np.sin(w) for w in design]))
        res = trf_minimize(**sine_problem(), basis=stale, max_iterations=60)
        assert res.success
        assert res.fun == pytest.approx(SINE_FUN, abs=1e-6)
        assert res.x[1] == pytest.approx(np.sin(res.x[0]), abs=1e-7)


class TestStepNormOverDofBlock:
    """gh#989: the trust radius bounds |dw| only, so it is updated from |dw|."""

    def test_large_valued_output_does_not_inflate_radius(self):
        truth = lambda w: np.array([1000.0 * np.sin(w[0])])  # noqa: E731
        fun = lambda x: (x[1] - 700.0) ** 2 * 1e-3 + (x[0] - 3.0) ** 2  # noqa: E731
        jac = lambda x: np.array(  # noqa: E731
            [2 * (x[0] - 3.0), 2e-3 * (x[1] - 700.0)]
        )
        x0 = np.array([0.1, truth([0.1])[0]])
        res = trf_minimize(
            fun,
            x0,
            truth,
            w_index=[0],
            y_index=[1],
            jac=jac,
            bounds=[(-5, 5), (-2000, 2000)],
            basis="zero",
            trust_radius=0.5,
            max_iterations=30,
        )
        first = res.history[0]
        # The step is at most the radius in w; y moved by hundreds.
        assert first.step_norm <= 0.5 + 1e-8
        assert first.trust_radius < 10.0
