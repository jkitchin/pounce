"""gh#967: `bound_relax_factor` × a large objective coefficient.

`min C·x + y/C` over `x + y >= 1`, `x, y ∈ [0,1]` has optimum `x=0, y=1` and
value `1/C` for **every** `C`, so the exact answer is known at every scale.
On the NLP arm the returned `x[0]` sits ≈ `-1e-8` outside its declared bound
regardless of `C` — the `bound_relax_factor` signature — and `C` amplifies
that constant box violation into the objective. At `C = 1e12` the reported
objective is `-1.0e+04`: wrong by `1e4`, and negative, for an objective that
is non-negative everywhere on the declared box.

These tests pin the *semantics*, not a preference. `bound_relax_factor` is
load-bearing (a feasible-iterate log-barrier needs `x` strictly inside its
bounds) and every candidate default change was measured and rejected — see
`dev-notes/bound-relax-objective-amplification.md`, which records the
`honor_original_bounds` rows-for-box trade and the negative result on capping
the relaxation by objective sensitivity.

So what is pinned here is: the amplification is real and proportional, the
measurement that reveals it is reported, and each documented remedy works.
If any of these move, the note is stale and should be revisited with it.
"""

import os

os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

import pounce

SCALES = [1e2, 1e4, 1e6, 1e8, 1e10, 1e12]


def solve(C, **options):
    """`min C·x + y/C` s.t. `x + y >= 1`, `x, y ∈ [0,1]`. Optimum `1/C`."""
    c = np.array([C, 1.0 / C])
    return pounce.minimize(
        lambda z, c=c: float(c @ z),
        np.array([0.5, 0.5]),
        jac=lambda z, c=c: c,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        constraints=[
            {
                "type": "ineq",
                "fun": lambda z: np.array([z[0] + z[1] - 1.0]),
                "jac": lambda z: np.array([[1.0, 1.0]]),
            }
        ],
        options=options or None,
    )


def test_the_box_violation_is_constant_and_the_objective_error_tracks_it():
    """The defect is `bound_relax_factor`, not the objective scale.

    `x[0]` sits the same ~1e-8 outside its bound at every `C`; the objective
    error is that distance times `C`. Pinning the *ratio* is what says the
    cause is the widening — an error that grew for some other reason would
    not track `C` to within a factor of two.
    """
    for C in SCALES:
        r = solve(C)
        viol = -r.x[0]
        assert viol > 0.0, f"C={C:g}: expected x[0] below its lower bound 0"
        assert 5e-9 < viol < 2e-8, f"C={C:g}: box violation {viol:e} off signature"
        err = abs(r.fun - 1.0 / C)
        # err ≈ C · viol, allowing a factor of two for the second coordinate.
        assert 0.5 < err / (C * viol) < 2.0, (
            f"C={C:g}: objective error {err:e} does not track C·(box violation)"
        )


def test_large_coefficients_return_a_negative_objective_that_cannot_be_negative():
    """`C·x + y/C` with `x, y >= 0` is non-negative on the declared box.

    It is reported negative from `C = 1e6` up, under `Solve_Succeeded`. This
    is the reported defect; it is pinned so that a change which fixes it
    fails here loudly rather than silently altering an unasserted number.
    """
    negative_at = [C for C in SCALES if solve(C).fun < 0.0]
    assert negative_at == [1e6, 1e8, 1e10, 1e12], (
        f"the scales returning a negative objective moved: {negative_at}"
    )


def test_the_violation_is_reported_even_though_nothing_gates_on_it():
    """`final_declared_box_viol` is the measurement, and it is not a placeholder.

    A consumer that needs an unscaled guarantee reads this and
    `final_unscaled_kkt_error` rather than branching on the status — which is
    a statement about the *scaled* residuals of the *relaxed* model.
    """
    info = solve(1e12)["info"]
    assert info["final_declared_box_viol"] == pytest.approx(1e-8, rel=0.1)
    # The scaled test is what terminated the solve; unscaled is ~1e-5.
    assert info["final_kkt_error"] < 1e-12
    assert info["final_unscaled_kkt_error"] > 1e-6


@pytest.mark.parametrize("C", SCALES)
def test_honor_original_bounds_recovers_the_exact_optimum_here(C):
    """Opt-in remedy 1. Exact on this family at every scale.

    Not the default: measured over the fixture corpus it trades box
    feasibility for *row* feasibility, taking equality residuals from exactly
    zero to ~1e-8 on 10 of 97 fixtures. See the dev-note.
    """
    r = solve(C, honor_original_bounds="yes")
    assert r.x[0] == 0.0, "the returned point should sit inside the declared box"
    assert r.fun == pytest.approx(1.0 / C, rel=1e-6)


def test_the_convex_arm_already_solves_the_declared_model():
    """Opt-in remedy 2, and the reason this is an NLP-arm gap rather than a
    policy question: gh#760 made `bound_relax_factor` opt-in on the convex
    arm, so routing there returns a point inside the box.

    The Python frontend defaults `solver_selection` to `nlp`, so a caller has
    to ask for `auto` by name to get it.
    """
    nlp = solve(1e12, solver_selection="nlp")
    auto = solve(1e12, solver_selection="auto")
    assert nlp.fun < -1e3, "the NLP arm is expected to show the defect"
    assert auto.x[0] >= 0.0, "the convex arm should stay inside the declared box"
    assert abs(auto.fun) < 1e-6, f"convex arm objective {auto.fun:e} not near 0"
