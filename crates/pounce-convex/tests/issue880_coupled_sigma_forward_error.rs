//! Regression for gh #880: on a **coupled** (non-separable) ill-conditioned QP
//! the cost-normalized (`σ`) path still returned a materially wrong `x` under
//! `Optimal` after gh #875 was fixed, because both componentwise guards ask a
//! *per-row* question and a per-row question cannot see a rotated spectrum.
//!
//! This is the half `issue875_unconstrained_sigma_stationarity.rs` names in its
//! own "NOT evidence about" section, and this file is the coupled coverage that
//! paragraph asks for.
//!
//! # The gap
//!
//! `sigma_stationarity_is_genuine` and `sigma_complementarity_is_genuine` both
//! hold row `i`'s residual against `dᵢ`, the largest single term that built it
//! — a **directional** scale, not a reduced one. It is the same distinction
//! CLAUDE.md draws one crate over for the sensitivity classifier
//! (`reduced/diagonal`, gh #763) and for constraint rows
//! (`reduced/directional`, gh #804), and it fails here for the same reason.
//!
//! On a diagonal `P` row `i`'s residual is `eᵢ(xᵢ − tᵢ)` against a scale of
//! order `eᵢ|tᵢ|`, so the ratio *is* the relative error in `xᵢ`: exact, which
//! is why gh #875's fix worked. Rotate the same spectrum and the stiff mode
//! enters every row, every denominator collapses to one number of order
//! `e_max`, and an error of order 1 along the *soft* eigenvector produces a
//! ratio of order `e_min/e_max` — i.e. `1/cond`. The guard reads a converged
//! `7e-9` and accepts an answer that is 100% wrong.
//! `the_row_ratio_is_blind_on_a_rotated_spectrum_and_exact_on_a_diagonal_one`
//! is that claim as arithmetic, with no solver in the loop.
//!
//! # The fix, and why it is not a fourth ratio
//!
//! `x − x*` is not a per-row quantity, so no per-row denominator can bound it.
//! `sigma_forward_error_is_small` solves for it instead — `Δ` is the
//! **affine-scaling** Newton step, the step the iteration itself would take
//! toward `μ = 0` from the point it is about to certify:
//!
//! ```text
//!     (P + Gᵀ Σ_row G + Σ_bnd) Δ = −(Px + c)
//! ```
//!
//! where `Σ` is the barrier diagonal `zⱼ/sⱼ` implied by the multipliers and
//! slacks actually returned — the IPM's own Newton operator, reducing to
//! exactly `P` when nothing is active. The verdict is `‖Δ‖∞ / max(1, ‖x‖∞)`
//! against the same `cut` the other two arms already use, so nothing here is
//! a tuned threshold. It is a norm of a vector rather than a ratio of
//! coordinates, and therefore basis-free: rotating the instance rotates `Δ`
//! and leaves `‖Δ‖∞` where it was, up to the usual `∞`-vs-`2` factor.
//!
//! **The right-hand side is `−(Px + c)`, not the stationarity residual, and
//! that is not a simplification.** Eliminating `Δz` and `Δs` cancels the
//! multiplier term exactly, which is the point: the returned multipliers are
//! trusted to say how *stiff* each row is and nothing else. Using `r_stat`
//! instead — this arm's first draft — makes the estimate agree with any point
//! that is self-consistent in its own multipliers, however wrong in `x`, and
//! that is a shape the `σ` cascade actually produces: see
//! `a_bound_constrained_coupled_instance_is_solved`, where the un-normalized
//! re-solve came back 31% wrong with `‖r_stat‖ = 2e-6` and an `r_stat`-based
//! estimate read `9.9e-8` and accepted it.
//!
//! # Which direction is safe
//!
//! Under-solving `Δ` is safe. CG starts at `Δ = 0` and builds `‖Δₖ‖` upward,
//! so the iteration cap, a `pᵀMp ≤ 0` breakout, and an LP's `P = 0` all
//! *under*-estimate, accept, and leave the status quo.
//!
//! Under-estimating `Σ` is **not**. With this right-hand side an active bound
//! contributes a large `−(Px + c)` that only its own `Σ → ∞` holds down, so a
//! `Σ` read as zero declares an active bound free, inflates `‖Δ‖`, and rejects
//! a correct answer. `barrier_ratio` therefore declines (`None`, which
//! accepts) on any slack or ratio it cannot use, rather than substituting `0`;
//! its corners are unit-tested in `ipm.rs`'s `forward_error_operator_tests`,
//! because a converged interior point never reaches them and a defensive
//! branch with no test is a branch that has never run.
//!
//! # Measured
//!
//! Over the 72-instance unconstrained census gh #875 used (`cond` `1e2 ‥ 1e12`
//! × magnitude `1e-3 ‥ 1e3` × `n` ∈ {2, 5} × rotated or not), rebuilt under
//! gh #882 so its reference is the **exact** optimum of the realised problem —
//! `x* = −P⁻¹c` computed in `fractions.Fraction`, verified by an exactly-zero
//! residual on all 72 — rather than the `t` each instance was designed around.
//! That distinction is not pedantry: `c = −Pt` is rounded, so `t` sits up to
//! `2.8e-07` (`cond = 1e10`) and `4.3e-05` (`1e12`) away from the optimum the
//! solver is actually handed, which is the same order as the errors being
//! judged. Claimed-optimal-but-wrong (relative error `> 1e-6`) goes
//! **17/72 → 9/72**:
//!
//! | `cond` | wrong before | wrong after | worst rel. err before | after |
//! |---|---|---|---|---|
//! | `1e2`  | 0 | 0 | 6.405e-08 | 6.405e-08 |
//! | `1e4`  | 0 | 0 | 6.275e-08 | 6.275e-08 |
//! | `1e6`  | 2 | 0 | 2.287e-04 | 3.828e-09 |
//! | `1e8`  | 3 | 0 | 1.091e-02 | 1.405e-08 |
//! | `1e10` | 6 | 3 | 3.208e-01 | 2.026e-06 |
//! | `1e12` | 6 | 6 | 7.050e-01 | 4.908e-04 |
//!
//! **The nine survivors are real, and they begin exactly where double
//! precision ends.** `‖Δ‖` is accumulated in `f64`, so this guard cannot
//! resolve a relative error below `ε·cond`, and cannot reject what it cannot
//! see. A `1e-6` verdict is therefore meaningful only while `ε·cond < 1e-6`,
//! i.e. `cond < 1e-6/ε ≈ 4.5e9` — every conditioning below that boundary is
//! now clean, and every survivor is above it. Per instance, `err/(ε·cond)`:
//!
//! ```text
//!   k=49 0.58   k=51 0.51   k=57 0.91              (cond 1e10)
//!   k=61 0.15   k=63 0.03   k=65 2.21
//!   k=67 0.08   k=69 0.11   k=71 0.01              (cond 1e12)
//! ```
//!
//! — eight of the nine *below* the estimator's own arithmetic floor, one at
//! 2.2×. The boundary is soft rather than sharp: of the two instances this
//! change fixed at `cond ≤ 1e8`, one sat at 17.2× of its floor and the other
//! at 0.63×, so the arm can reject just under the line but not dependably. The
//! statement that survives is the weaker one — this guard works over the range
//! a double-precision forward-error test can act on at all, and closing the
//! residue needs a differently-conditioned estimator rather than a tighter cut.
//! That residue is gh #880, still open; this file does not claim it.
//!
//! The other signature is gone outright, and that one is not a matter of
//! resolution: every failing coupled row in gh #880's table reported `Optimal
//! Solution Found` in **one** iteration, which is what a certificate accepted
//! without being tested looks like. After this change the survivors take 5–13.
//! The arm fires everywhere; what is left is accuracy, not an unchecked claim.
//!
//! # What this file is NOT evidence about
//!
//! **`cond ≳ 4.5e9`.** Nine instances stay wrong, this file does not fix
//! them, and the section above says why a `f64` estimator cannot. One further
//! thing is true up there and is not addressed here either: `qp_hsde=no` — the
//! destination a `σ` reject routes to — was *itself* wrong at that
//! conditioning in gh #880's table (`1.39` and `1.04` relative), so rejecting
//! harder had nowhere correct to route to. The honest statement is that this
//! file demonstrates the fix through `cond = 1e8`, and above it demonstrates a
//! large improvement in magnitude with no regression.
//! `the_cond_1e12_floor_is_a_carve_out`
//! pins the improvement as a number so the carve-out cannot quietly widen.
//!
//! **Equality rows.** `sigma_forward_error_is_small` returns `true` outright
//! when `m_eq > 0`: `A` has no barrier diagonal, so the operator above is not
//! the right one and the honest form is the full saddle system.
//! `an_equality_constrained_instance_is_declined_not_broken` pins that this is
//! a decline, not a regression.
//!
//! **The fixture corpus.** `scripts/sweep-fixtures.sh` is a
//! no-collateral-damage check on this change and nothing more — per CLAUDE.md
//! exactly 1 of 79 fixtures and 0 of 138 Maros-Meszaros problems reach the `σ`
//! path at all, so an empty diff there is the expected result and carries no
//! information about whether this works.
//!
//! # Mutation table
//!
//! Each row is a defect reintroduced at `crates/pounce-convex/src/ipm.rs` and
//! the tests that go red. **Every row below was run**, one mutation at a time,
//! and the "caught by" column is the observed failure list, not a prediction —
//! four of these rows started out wrong.
//!
//! The split matters and is the reason `ipm.rs` carries a
//! `forward_error_operator_tests` module at all. Five of the eight mutations
//! leave *every* test in this file green: dropping either `Σ` block, dropping
//! the `‖x‖` scale, removing the `m_eq` early return, and letting the operator
//! accumulate across CG iterations all change the arm's **verdict** without
//! changing the **answer** the caller finally receives, because the `σ`
//! cascade's later candidates are also correct on the instances that reach it.
//! An arm that has gone inert is invisible from out here. So the four
//! structural pieces are pinned by calling `sigma_forward_error_is_small`
//! directly on hand-built points whose optimum is known in closed form; this
//! file pins that the arm, working, fixes gh #880.
//!
//! | Mutation | Caught by |
//! |---|---|
//! | drop the `sigma_forward_error_is_small` conjunct — i.e. gh #880 itself | **8 here**: `the_rotated_reported_instance_is_solved_at_the_default_tolerance`, `the_coupled_condition_number_sweep_is_correct_in_x`, `every_tolerance_is_correct_on_the_coupled_arm`, `an_inert_objective_rescaling_stays_inert_on_the_coupled_arm`, `a_bound_constrained_coupled_instance_is_solved`, `an_inequality_constrained_coupled_instance_is_solved`, `an_active_bound_is_stiff_not_free`, `the_cond_1e12_floor_is_a_carve_out` |
//! | right-hand side back to `r_stat` (trust the returned multipliers) | **3 here**: `a_bound_constrained_coupled_instance_is_solved`, `an_inequality_constrained_coupled_instance_is_solved`, `an_active_bound_is_stiff_not_free` — and none of the unconstrained ones, which is the point: the multipliers are only wrong where there are multipliers |
//! | invert the verdict (reject on small `‖Δ‖`) | the 8 above, **plus** 4 in `forward_error_operator_tests` |
//! | drop `Σ_bnd` from the operator | *nothing here.* `forward_error_operator_tests::an_active_bound_is_held_down_by_its_own_sigma` |
//! | drop `Gᵀ Σ_row G` from the operator | *nothing here.* `forward_error_operator_tests::an_active_row_is_held_down_by_its_own_sigma` |
//! | compare `‖Δ‖∞` against `cut` without the `‖x‖` scale | *nothing here.* `forward_error_operator_tests::the_verdict_is_relative_to_x` |
//! | remove the `m_eq > 0` early return | *nothing here.* `forward_error_operator_tests::an_equality_row_makes_the_arm_decline` (`an_equality_constrained_instance_is_declined_not_broken` stays green — the cascade repairs it either way) |
//! | let the operator accumulate (drop the zeroing of `y` in `apply`) | *nothing here.* `forward_error_operator_tests::the_operator_does_not_accumulate_across_cg_iterations` — it needs four coupled variables, and every fixture here has two |
//! | `barrier_ratio` returns `Some(0.0)` instead of `None` on an unusable slack | *nothing here.* `forward_error_operator_tests::{an_exactly_active_bound_declines_rather_than_reading_as_free, a_negative_slack_declines_too}` |
//!

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// A fixed plane rotation with **rational** entries, so the fixture is the
/// same bit pattern on every platform and the coupling is not an accident of a
/// PRNG: `cos θ = 3/5`, `sin θ = 4/5`.
const COS: f64 = 0.6;
const SIN: f64 = 0.8;

/// `P = Q diag(e) Qᵀ` for the rotation above, lower triangle. Fully coupled:
/// `P₀₁ = cs(e₀ − e₁)` is of order `e₁`, so the stiff mode is present in both
/// rows with a comparable magnitude and *neither* row's denominator reflects
/// the soft direction.
fn rotated_p(e: [f64; 2], k: f64) -> Vec<Triplet> {
    let (c2, s2, cs) = (COS * COS, SIN * SIN, COS * SIN);
    vec![
        Triplet::new(0, 0, k * (c2 * e[0] + s2 * e[1])),
        Triplet::new(1, 0, k * (cs * (e[0] - e[1]))),
        Triplet::new(1, 1, k * (s2 * e[0] + c2 * e[1])),
    ]
}

fn p_times(p: &[Triplet], v: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; v.len()];
    for t in p {
        let (i, j, x) = (t.row, t.col, t.val);
        y[i] += x * v[j];
        if i != j {
            y[j] += x * v[i];
        }
    }
    y
}

/// `min ½(x − t)ᵀ P (x − t)` with `P = k·Q diag(e) Qᵀ` and no constraints and
/// no bounds, written as `½xᵀPx + cᵀx` with `c = −Pt`. The exact minimiser is
/// `t` **by identity** — `P` is positive definite for any positive `e`, so the
/// stationarity condition `Px = Pt` has the unique solution `x = t`. No oracle
/// and no reference solver are in the loop.
fn coupled_qp(e: [f64; 2], t: &[f64], k: f64) -> (QpProblem, Vec<f64>) {
    let p_lower = rotated_p(e, k);
    let c = p_times(&p_lower, t).iter().map(|v| -v).collect();
    (
        QpProblem {
            n: 2,
            p_lower,
            c,
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        },
        t.to_vec(),
    )
}

/// The diagonal sibling, so the separable control is built the same way.
fn diagonal_qp(e: [f64; 2], t: &[f64], k: f64) -> (QpProblem, Vec<f64>) {
    let p_lower = vec![Triplet::new(0, 0, k * e[0]), Triplet::new(1, 1, k * e[1])];
    let c = p_times(&p_lower, t).iter().map(|v| -v).collect();
    (
        QpProblem {
            n: 2,
            p_lower,
            c,
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        },
        t.to_vec(),
    )
}

fn rel_x_err(x: &[f64], exact: &[f64]) -> f64 {
    let scale = exact.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    x.iter()
        .zip(exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
        / scale
}

/// The largest componentwise ratio `|rᵢ| / dᵢ` at a point, with `dᵢ` the
/// largest single term that built row `i` — i.e. exactly the quantity
/// `sigma_stationarity_is_genuine` tests. Written out here so the claim about
/// what that guard can and cannot see is checkable rather than asserted.
fn worst_row_ratio(p_lower: &[Triplet], c: &[f64], x: &[f64]) -> f64 {
    let n = x.len();
    let (mut r, mut d) = (vec![0.0; n], vec![0.0_f64; n]);
    for (i, ci) in c.iter().enumerate() {
        r[i] = *ci;
        d[i] = ci.abs();
    }
    for t in p_lower {
        let (i, j, v) = (t.row, t.col, t.val);
        r[i] += v * x[j];
        d[i] = d[i].max((v * x[j]).abs());
        if i != j {
            r[j] += v * x[i];
            d[j] = d[j].max((v * x[i]).abs());
        }
    }
    (0..n)
        .map(|i| if d[i] > 0.0 { r[i].abs() / d[i] } else { 0.0 })
        .fold(0.0_f64, f64::max)
}

/// The `σ` gate is `max(‖P‖∞, ‖c‖∞)·ε > tol`, so these targets are the
/// gh #875 instance's, unchanged, to keep the two files comparable.
const TGT: [f64; 2] = [3.0, 0.5];

// ---------------------------------------------------------------------------
// The thesis, as arithmetic. No solver.
// ---------------------------------------------------------------------------

/// **Why a third arm was needed at all.** Take the same spectrum twice — once
/// diagonal, once rotated — and displace `x` from the true optimum by a unit
/// step along the **soft** eigenvector, the displacement an ill-conditioned
/// solve actually makes. The componentwise ratio the existing guards measure
/// reads `O(1)` on the diagonal instance and `O(1/cond)` on the rotated one,
/// for the *same* error in `x`.
///
/// That single pair of numbers is the whole of gh #880: gh #875's fix is exact
/// on the left column and inert on the right, and no choice of threshold on a
/// per-row ratio can change that, because the quantity itself has stopped
/// carrying the information.
#[test]
fn the_row_ratio_is_blind_on_a_rotated_spectrum_and_exact_on_a_diagonal_one() {
    for cond in [1e8, 1e10, 1e12] {
        let e = [1.0, cond];

        // Diagonal: the soft eigenvector is e₀, so displace x₀ by 1.
        let (dprob, dexact) = diagonal_qp(e, &TGT, 1.0);
        let dx: Vec<f64> = vec![dexact[0] + 1.0, dexact[1]];
        let dratio = worst_row_ratio(&dprob.p_lower, &dprob.c, &dx);

        // Rotated: the soft eigenvector is Q's first column, (cos, sin).
        let (rprob, rexact) = coupled_qp(e, &TGT, 1.0);
        let rx: Vec<f64> = vec![rexact[0] + COS, rexact[1] + SIN];
        let rratio = worst_row_ratio(&rprob.p_lower, &rprob.c, &rx);

        // Both points are wrong by a unit step in x, by construction.
        assert!((rel_x_err(&dx, &dexact) - 1.0 / 3.0).abs() < 1e-12);
        assert!(rel_x_err(&rx, &rexact) > 0.1);

        assert!(
            dratio > 0.1,
            "cond {cond:.0e}: the diagonal instance's row ratio is \
             {dratio:.3e}; gh #875's fix depends on this being O(1)"
        );
        assert!(
            rratio < 1e-6,
            "cond {cond:.0e}: the rotated instance's row ratio is \
             {rratio:.3e} for the same unit error in x; if this is no longer \
             below the guard's cut, the premise of gh #880 has changed"
        );
        assert!(
            rratio < dratio * 1e-5,
            "cond {cond:.0e}: rotating collapsed the ratio only from \
             {dratio:.3e} to {rratio:.3e}"
        );
    }
}

// ---------------------------------------------------------------------------
// The reject branch: instances that were wrong and must now be right.
// ---------------------------------------------------------------------------

/// The headline. gh #875's instance, rotated: same spectrum, same targets,
/// same `σ` gate — and before this fix it returned `x` wrong by ~100% under
/// `Optimal`, bit-identical to the pre-#875 baseline because the arm #875
/// installed never fired here.
#[test]
fn the_rotated_reported_instance_is_solved_at_the_default_tolerance() {
    let (prob, exact) = coupled_qp([1.0, 1e10], &TGT, 1.0);
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal, "the instance is trivial");
    let err = rel_x_err(&sol.x, &exact);
    assert!(
        err < 1e-5,
        "x = {:?} off the closed form {exact:?} by {err:.3e} relative; the \
         componentwise ratio at that point is {:.3e}, three orders below the \
         guard's cut, which is why it was accepted",
        sol.x,
        worst_row_ratio(&prob.p_lower, &prob.c, &sol.x)
    );
}

/// The condition-number sweep on the coupled arm, matching the shape of
/// `the_condition_number_sweep_is_correct_in_x_not_just_in_the_objective` in
/// the gh #875 file. `1e12` is excluded here and pinned separately below,
/// because there the un-normalized destination is itself wrong.
#[test]
fn the_coupled_condition_number_sweep_is_correct_in_x() {
    for cond in [1e4, 1e6, 1e8, 1e10] {
        let (prob, exact) = coupled_qp([1.0, cond], &TGT, 1.0);
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "cond {cond:.0e}");
        let err = rel_x_err(&sol.x, &exact);
        assert!(
            err < 1e-5,
            "cond {cond:.0e}: x off by {err:.3e} relative (census worst on \
             the coupled arm was 3.208e-01 at 1e10)"
        );
    }
}

/// Rotating must not undo gh #875: the separable instance it fixed is still
/// fixed. A guard that repaired the coupled arm by rerouting the whole `σ`
/// path would pass everything above and quietly regress this.
#[test]
fn the_separable_instance_is_still_repaired() {
    for cond in [1e6, 1e8, 1e10, 1e12] {
        let (prob, exact) = diagonal_qp([1.0, cond], &TGT, 1.0);
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "cond {cond:.0e}");
        assert!(
            rel_x_err(&sol.x, &exact) < 1e-6,
            "cond {cond:.0e}: gh #875's own instance regressed to {:?}",
            sol.x
        );
    }
}

/// Tightening `tol` is not a workaround on this path and never was —
/// `hsde_cost_scale` reads `tol`, so tightening it *pulls* models onto the `σ`
/// path rather than off it. Both the "below the gate" and "above it" branches
/// have to be right, so the sweep spans the gate.
#[test]
fn every_tolerance_is_correct_on_the_coupled_arm() {
    for tol in [1e-4, 1e-6, 1e-8, 1e-10] {
        let (prob, exact) = coupled_qp([1.0, 1e10], &TGT, 1.0);
        let opts = QpOptions {
            tol,
            ..QpOptions::default()
        };
        let sol = solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(sol.status, QpStatus::Optimal, "tol {tol:.0e}");
        let err = rel_x_err(&sol.x, &exact);
        // `cut` is `min(100·tol, 1e-3)`, so a loose `tol` licenses a looser
        // answer; the bar tracks it rather than pretending otherwise.
        let bar = 1e-5_f64.max(1e3 * tol);
        assert!(
            err < bar,
            "tol {tol:.0e}: x off by {err:.3e} relative, bar {bar:.3e}"
        );
    }
}

/// Multiplying the objective by `k > 0` leaves the argmin unchanged **by
/// identity**. The defect did not scale with `k`, so this is a contradiction
/// with no oracle in it, and it sweeps `σ` from off (small `k`) to on by ten
/// decades.
#[test]
fn an_inert_objective_rescaling_stays_inert_on_the_coupled_arm() {
    let tol = QpOptions::default().tol;
    for k in [1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6] {
        let (prob, exact) = coupled_qp([1.0, 1e10], &TGT, k);
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "k = {k:.0e}");
        let err = rel_x_err(&sol.x, &exact);
        // The stopping test is absolute and the soft curvature is `k`, so no
        // correct solve pins the soft coordinate better than `tol/k`; see the
        // same argument in the gh #875 file.
        let bar = 1e-5_f64.max(10.0 * tol / k);
        assert!(
            err < bar,
            "k = {k:.0e}: x off by {err:.3e} relative, bar {bar:.3e}"
        );
    }
}

// ---------------------------------------------------------------------------
// The other two terms of the operator. Unconstrained fixtures exercise `P`
// alone, so each of `Σ_bnd` and `Gᵀ Σ_row G` needs its own instance.
// ---------------------------------------------------------------------------

/// The bar for the two constrained instances below. A reject routes to the
/// **un-normalized** solve, so the answer this file can demand is bounded by
/// what that path delivers, and that ceiling degrades with `cond`: measured
/// with `use_hsde = false` on these exact models it is `1.1e-11 / 2.3e-9 /
/// 1.2e-4` at `cond = 1e6 / 1e8 / 1e10`. `cond·1e-13` tracks that with an
/// order of headroom, and still separates the defect by three to four orders
/// at every point — before this fix these two returned `4.2e-1` and `7.4e-1`.
fn destination_bar(cond: f64) -> f64 {
    (cond * 1e-13).max(1e-9)
}

/// `Σ_bnd`. The same coupled objective inside a **box** the optimum does not
/// touch, so the barrier term is finite and nonzero and enters the operator.
///
/// This shape is why the right-hand side is `−(Px + c)` and not the
/// stationarity residual. At the wrong point the un-normalized re-solve
/// returned — `x = (2.064, −0.748)`, displaced 1.56 along the soft eigenvector
/// — the bound multipliers came back at `O(1)` on bounds with slack `~10`, and
/// they absorbed the whole gradient: the stationarity residual read `2e-6`
/// while `x` was 31% wrong. A forward-error estimate that trusts those
/// multipliers reads `9.9e-8` and accepts. Dropping them from the right-hand
/// side, which is the affine-scaling step, reads `1.1` and rejects.
#[test]
fn a_bound_constrained_coupled_instance_is_solved() {
    for cond in [1e6, 1e8, 1e10] {
        let (mut prob, exact) = coupled_qp([1.0, cond], &TGT, 1.0);
        prob.lb = vec![-10.0, -10.0];
        prob.ub = vec![10.0, 10.0];
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "cond {cond:.0e}");
        let (err, bar) = (rel_x_err(&sol.x, &exact), destination_bar(cond));
        assert!(
            err < bar,
            "cond {cond:.0e}: x = {:?} against the interior optimum {exact:?},              off by {err:.3e} against bar {bar:.3e} (this returned 4.161e-1              before the fix, at cond 1e10)",
            sol.x
        );
    }
}

/// `Gᵀ Σ_row G`. An inactive inequality row, again with the optimum strictly
/// inside, so `Σ_row` is finite and nonzero. Same multiplier story as the box
/// above: the pre-fix answer was `x = (1.339, −1.714)`, 74% wrong, carried by
/// a row multiplier of `1.98` against a slack of `20.4`.
#[test]
fn an_inequality_constrained_coupled_instance_is_solved() {
    for cond in [1e6, 1e8, 1e10] {
        let (mut prob, exact) = coupled_qp([1.0, cond], &TGT, 1.0);
        // x₀ + x₁ ≤ 20, slack ≈ 16.5 at the optimum.
        prob.g = vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)];
        prob.h = vec![20.0];
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "cond {cond:.0e}");
        let (err, bar) = (rel_x_err(&sol.x, &exact), destination_bar(cond));
        assert!(
            err < bar,
            "cond {cond:.0e}: x = {:?} against the interior optimum {exact:?}, \
             off by {err:.3e} against bar {bar:.3e} (this returned 7.381e-1 \
             before the fix, at cond 1e10)",
            sol.x
        );
    }
}

/// The **other** branch of the barrier diagonal, per CLAUDE.md's rule: an
/// **active** bound, where `Σ` is legitimately large and is the only thing
/// holding `‖Δ‖` down. With `x₀ ≤ 2` cutting off the unconstrained optimum
/// `x₀ = 3`, the minimiser is `x₀ = 2` and
/// `x₁ = t₁ − (P₁₀/P₁₁)(2 − t₀)`, in closed form and with no oracle.
///
/// At that point `−(Px + c)` — this arm's right-hand side — is *large*, of
/// order the multiplier, and it is `Σ_bnd → ∞` that makes `Δ` small. A `Σ`
/// that collapses to zero on a nearly-active bound therefore rejects a correct
/// answer, which is why [`barrier_ratio`] floors the slack rather than
/// dropping the term. The cost assertion is the one that bites: a spurious
/// reject here buys an un-normalized re-solve on every such model.
#[test]
fn an_active_bound_is_stiff_not_free() {
    for cond in [1e6, 1e8, 1e10] {
        let (mut prob, _) = coupled_qp([1.0, cond], &TGT, 1.0);
        prob.ub = vec![2.0, f64::INFINITY];
        let (p10, p11) = (COS * SIN * (1.0 - cond), SIN * SIN + COS * COS * cond);
        let exact = vec![2.0, TGT[1] - (p10 / p11) * (2.0 - TGT[0])];
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "cond {cond:.0e}");
        assert!(
            rel_x_err(&sol.x, &exact) < 1e-8,
            "cond {cond:.0e}: x = {:?} against the closed form {exact:?}",
            sol.x
        );
        assert!(
            sol.iters <= 25,
            "cond {cond:.0e} took {} iterations; a bound whose Σ reads zero \
             inflates ‖Δ‖ and buys an un-normalized re-solve every time",
            sol.iters
        );
    }
}

/// The `m_eq > 0` early return is a **decline**, not a repair: `A` carries no
/// barrier diagonal, so the operator this arm builds is not the right one for
/// an equality-constrained model and the honest form is the full saddle
/// system. What must hold is that declining leaves the model exactly where the
/// other two arms had it — this is the status quo, asserted so that removing
/// the early return (and silently applying the wrong operator) is caught.
#[test]
fn an_equality_constrained_instance_is_declined_not_broken() {
    let (mut prob, _) = coupled_qp([1.0, 1e10], &TGT, 1.0);
    // x₀ + x₁ = 3.5, which the unconstrained optimum (3, ½) already satisfies,
    // so the constrained minimiser is the unconstrained one.
    prob.a = vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)];
    prob.b = vec![TGT[0] + TGT[1]];
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    // The equality is what actually pins this model, and it is satisfied
    // regardless of the soft direction, so only feasibility is asserted — the
    // arm declines, and this file makes no claim about `x` here.
    let feas = (sol.x[0] + sol.x[1] - (TGT[0] + TGT[1])).abs();
    assert!(
        feas < 1e-6,
        "the equality row is violated by {feas:.3e}; the arm is supposed to \
         decline on m_eq > 0, not to perturb the solve"
    );
}

/// An LP has `P = 0`, so CG has no curvature to work with and the arm returns
/// `true` on the first breakout. That is the deliberate under-estimate: the
/// conjunct becomes inert and the model is decided by the two componentwise
/// arms exactly as before.
#[test]
fn an_lp_is_left_to_the_existing_arms() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 1e10],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(1, 1, -1.0)],
        h: vec![0.0, 0.0],
        lb: vec![0.0, 0.0],
        ub: vec![5.0, 5.0],
    };
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        rel_x_err(&sol.x, &[0.0, 0.0]) < 1e-6,
        "x = {:?} against the vertex (0, 0)",
        sol.x
    );
}

// ---------------------------------------------------------------------------
// The accept branch. A guard that rejected unconditionally would pass every
// test above and be a pure cost: each reject buys an un-normalized re-solve.
// ---------------------------------------------------------------------------

/// A well-conditioned coupled QP must not be newly rejected, and must not
/// newly cost iterations. The baseline is the same binary with the conjunct
/// removed, measured at the fix commit.
#[test]
fn a_well_conditioned_coupled_qp_is_not_newly_rejected() {
    for cond in [1.0, 1e1, 1e2, 1e3] {
        let (prob, exact) = coupled_qp([1.0, cond], &TGT, 1.0);
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal, "cond {cond:.0e}");
        assert!(rel_x_err(&sol.x, &exact) < 1e-8, "cond {cond:.0e}");
        assert!(
            sol.iters <= 20,
            "cond {cond:.0e} took {} iterations; a rejected certificate buys \
             an un-normalized re-solve, and a guard that rejects a \
             well-conditioned model pays that on every solve",
            sol.iters
        );
    }
}

/// `σ` engaged, coupled, and *already right*: `P = k·Q I Qᵀ = k·I` rotated is
/// perfectly conditioned but large enough to trip the `σ` gate. The arm must
/// keep this answer.
#[test]
fn sigma_is_engaged_but_a_conditioned_coupled_answer_survives_it() {
    let (prob, exact) = coupled_qp([1e9, 1e9], &TGT, 1.0);
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        rel_x_err(&sol.x, &exact) < 1e-8,
        "x = {:?} against {exact:?}",
        sol.x
    );
}

/// The verdict is `‖Δ‖∞ ≤ cut·max(1, ‖x‖∞)`, and the `‖x‖` factor is
/// load-bearing: without it a correct answer at large `x` is rejected purely
/// for being large. `t = (3e4, 5e3)` is the same well-conditioned model four
/// decades up.
#[test]
fn a_large_magnitude_coupled_answer_survives_sigma() {
    let t = [3.0e4, 5.0e3];
    let (prob, exact) = coupled_qp([1e6, 1e9], &t, 1.0);
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!(
        rel_x_err(&sol.x, &exact) < 1e-6,
        "x = {:?} against {exact:?}",
        sol.x
    );
    assert!(
        sol.iters <= 30,
        "took {} iterations; a spurious reject costs an un-normalized re-solve",
        sol.iters
    );
}

// ---------------------------------------------------------------------------
// The carve-out, pinned as a number.
// ---------------------------------------------------------------------------

/// **`cond = 1e12` is improved, not repaired, and this is where that is
/// written down.** `‖Δ‖` is accumulated in `f64`, so the estimator's own
/// arithmetic floor is `ε·cond ≈ 2e-4` here — larger than every error it is
/// being asked to reject — and independently, `qp_hsde=no`, where a reject
/// routes, is itself wrong at this conditioning, so rejecting harder has
/// nowhere correct to go. That is sub-problem 2 of gh #880.
///
/// The census worst case moves `7.050e-01 → 4.908e-04`, ~1400×. The bar below
/// is that measured number with headroom, not a target: it exists so the
/// carve-out cannot quietly widen back toward `O(1)` without a test going red,
/// and it is deliberately *far* looser than every other bar in this file.
#[test]
fn the_cond_1e12_floor_is_a_carve_out() {
    let (prob, exact) = coupled_qp([1.0, 1e12], &TGT, 1.0);
    let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(sol.status, QpStatus::Optimal);
    let err = rel_x_err(&sol.x, &exact);
    assert!(
        err < 1e-2,
        "cond 1e12: x off by {err:.3e} relative — the census worst case is \
         4.908e-04 after this fix and 7.050e-01 before it, so anything near \
         O(1) means the arm stopped firing here"
    );
}
