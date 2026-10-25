//! What "the same result" means between pounce's run and the outside
//! engine's.
//!
//! The two runs share the step they start from and the model they evaluate,
//! and differ in everything else: pounce reads its own iterate in the frame
//! it solved in and factors the augmented system, while the outside engine
//! fills in the slacks and row multipliers a report leaves out, works in
//! natural units, and factors the eight-block matrix with a dense LU. So
//! discrete outcomes have to be equal, and numbers have to agree to a
//! tolerance that covers roundoff in two different factorizations.

use pounce_common::types::Number;
use pounce_sens_core::activity::{
    ActivityReport, INACTIVE, ReducedActivityReport, ReducedRowActivityReport,
};
use pounce_sens_core::corrector::CorrectorReport;

/// Relative agreement required of a corrected point, entry by entry.
pub const POINT_RTOL: Number = 1e-9;

/// Relative agreement required of a residual the two engines measure in the
/// same frame, with [`RESIDUAL_ATOL`] below which both are noise.
pub const RESIDUAL_RTOL: Number = 1e-6;

/// The level under which a residual is roundoff in either engine.
pub const RESIDUAL_ATOL: Number = 1e-12;

/// Relative agreement required of a classification ratio or a barrier
/// diagonal, where both are finite.
///
/// A report gives an inequality row's bound multiplier only through the
/// stationarity of its slack, which the solve holds to its dual tolerance
/// in absolute terms, so the outside engine's multiplier for an inactive
/// row, of order `μ/s`, can differ from the engine's own by a few parts in
/// a million. The classification's bands are decades apart, so agreement
/// to `1e-4` is the same classification with room to spare.
pub const RATIO_RTOL: Number = 1e-4;

/// `a` and `b` agree to `rtol` relative to the larger, plus `atol`. Two NaNs
/// agree, and equal values agree under any tolerance.
pub fn close(a: Number, b: Number, rtol: Number, atol: Number) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    // Equal values agree under any tolerance, including an infinite one,
    // which a bound sitting exactly on the point produces and which would
    // otherwise meet a zero magnitude and make the test below NaN.
    if a == b {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    (a - b).abs() <= rtol * a.abs().max(b.abs()) + atol
}

/// Which frame pounce's residual is in, relative to the outside engine's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Frame {
    /// pounce solved unscaled, so both residuals are natural-units residuals
    /// and are compared.
    Same,
    /// pounce solved a scaled model and reports its residual in that frame,
    /// so the corrected point and the active-set decision are compared, and
    /// the residuals and the iteration count, which depend on the frame, are
    /// not.
    Scaled,
}

/// The corrector produced the same correction over both engines.
pub fn assert_same_correction(
    label: &str,
    pounce: &(Vec<Number>, CorrectorReport),
    outside: &(Vec<Number>, CorrectorReport),
    frame: Frame,
) {
    assert_same_correction_to(label, pounce, outside, frame, POINT_RTOL);
}

/// [`assert_same_correction`] with the corrected point compared to
/// `point_tol`, for a fixture whose conditioning a test documents.
pub fn assert_same_correction_to(
    label: &str,
    pounce: &(Vec<Number>, CorrectorReport),
    outside: &(Vec<Number>, CorrectorReport),
    frame: Frame,
    point_tol: Number,
) {
    let (pa, ra) = pounce;
    let (pb, rb) = outside;
    assert_eq!(pa.len(), pb.len(), "{label}: step lengths differ");
    for (k, (&a, &b)) in pa.iter().zip(pb).enumerate() {
        assert!(
            close(a, b, point_tol, point_tol),
            "{label}: corrected step entry {k} is {a:e} over pounce and {b:e} over the \
             outside engine\n  pounce  {ra:?}\n  outside {rb:?}",
        );
    }
    assert_eq!(
        (ra.released, ra.pinned),
        (rb.released, rb.pinned),
        "{label}: the active-set decision differs\n  pounce  {ra:?}\n  outside {rb:?}",
    );
    if frame == Frame::Same {
        assert_same_report(label, ra, rb);
    }
}

/// Two corrector reports measured in the same frame agree: the iterations
/// ended the same way, and the residuals agree to [`RESIDUAL_RTOL`] above
/// [`RESIDUAL_ATOL`].
fn assert_same_report(label: &str, ra: &CorrectorReport, rb: &CorrectorReport) {
    assert_same_iterations(label, ra, rb);
    for (name, a, b) in [
        ("residual", ra.residual, rb.residual),
        ("initial_residual", ra.initial_residual, rb.initial_residual),
        ("stationarity", ra.stationarity, rb.stationarity),
        ("feasibility", ra.feasibility, rb.feasibility),
        ("complementarity", ra.complementarity, rb.complementarity),
    ] {
        assert!(
            close(a, b, RESIDUAL_RTOL, RESIDUAL_ATOL),
            "{label}: {name} is {a:e} over pounce and {b:e} over the outside engine",
        );
    }
}

/// The iterations of two corrector reports measured in the same frame ended
/// the same way, where that is decided by more than roundoff.
pub fn assert_same_iterations(label: &str, ra: &CorrectorReport, rb: &CorrectorReport) {
    // The loop stops at the first iteration that fails to lower the
    // residual's sup-norm, measured in the engine's own frame. Over a
    // scaled solve that norm weighs the blocks differently, so once both
    // engines are at roundoff they stop on different iterations at the
    // same point, and only the same frame makes the count comparable.
    // Within the same frame it is still decided by roundoff once the
    // residual is there, so it is compared only above that level.
    if ra.residual.max(rb.residual) > RESIDUAL_ATOL {
        assert_eq!(
            (ra.iterations, ra.converged),
            (rb.iterations, rb.converged),
            "{label}: the iterations ended differently
  pounce  {ra:?}
  outside {rb:?}",
        );
    }
}

/// How far apart an inactive row's ratio may be over the two engines.
///
/// A report carries no bound multiplier for an inequality row. The engine
/// fills it in, from the stationarity of the row's slack for a one-sided
/// row and as the barrier's `μ/s` on the inactive side of a row bounded on
/// both sides. An inactive row's multiplier is of order `μ/s`, and the
/// solve holds that stationarity only to its absolute tolerance, so the
/// filled-in value is known to the factor of ten the classification's own
/// central-path test allows, and no better. What the ratio decides for an
/// inactive row is its status, its central-path flag and its contamination
/// flag, all compared exactly.
///
/// A variable's bound multipliers come in the report itself, so an inactive
/// variable is held to [`RATIO_RTOL`] like every other entry.
const INACTIVE_RTOL: Number = 0.9;

/// Whether a ratio is within `RATIO_RTOL` of an edge of the classification's
/// bands at `mu`: `1e-1` and `1e1` around the weakly active band, and
/// `√μ` and `1/√μ` at the inactive and strongly active ends.
fn on_an_edge(r: Number, mu: Number) -> bool {
    let rm = mu.sqrt();
    [1e-1, 1e1, rm, 1.0 / rm]
        .iter()
        .any(|&edge| edge.is_finite() && close(r, edge, RATIO_RTOL, 0.0))
}

/// Two statuses agree, or differ only because both engines' ratio sits on
/// the same band edge to within their agreement, where roundoff decides the
/// side. The decoupled and coupled row fixtures put a ratio exactly on
/// `1e-1` by construction.
fn same_status(sa: i8, sb: i8, ra: Number, rb: Number, mu: Number) -> bool {
    sa == sb || (close(ra, rb, RATIO_RTOL, 0.0) && on_an_edge(ra, mu) && on_an_edge(rb, mu))
}

fn assert_same_statuses(
    label: &str,
    what: &str,
    a: &[i8],
    b: &[i8],
    ra: &[Number],
    rb: &[Number],
    mu: Number,
) {
    assert_eq!(a.len(), b.len(), "{label}: {what} lengths differ");
    for k in 0..a.len() {
        assert!(
            same_status(a[k], b[k], ra[k], rb[k], mu),
            "{label}: {what}[{k}] is {} over pounce and {} over the outside engine, \
             with ratios {:e} and {:e}",
            a[k],
            b[k],
            ra[k],
            rb[k],
        );
    }
}

/// What the comparison of two classifications allows for, entry by entry,
/// beyond [`RATIO_RTOL`].
pub struct Allowance {
    /// How far a bound the solve moved can move each variable's ratio and
    /// diagonal, relative to the larger of the two. See
    /// `OutsideEngine::bound_move_tolerance`.
    pub var_move: Vec<Number>,
    /// [`Self::var_move`] for each row.
    pub row_move: Vec<Number>,
    /// The ceiling pounce puts on each variable's barrier diagonal, in the
    /// units the report gives it in, `INFINITY` where none applies.
    ///
    /// pounce factors the augmented system, where a bound's barrier diagonal
    /// `z/s` sits beside the constraint rows. It caps that diagonal at the
    /// largest value the factorization still represents next to them
    /// (gh#737) and reports the capped value. The outside engine factors the
    /// eight-block matrix, which holds `z` and `s` apart, so it has no cap
    /// and reports `z/s` itself. Where the outside diagonal is above the cap,
    /// pounce's diagonal and ratio are the outside ones times `cap / Σ`.
    pub var_cap: Vec<Number>,
    /// [`Self::var_cap`] for each row.
    pub row_cap: Vec<Number>,
}

impl Allowance {
    /// No allowance: no bound moved and no ceiling.
    pub fn none(n_vars: usize, n_rows: usize) -> Self {
        Self {
            var_move: vec![0.0; n_vars],
            row_move: vec![0.0; n_rows],
            var_cap: vec![Number::INFINITY; n_vars],
            row_cap: vec![Number::INFINITY; n_rows],
        }
    }
}

/// The factor pounce's ceiling leaves on an entry whose outside diagonal is
/// `sigma`: `cap / |sigma|` where the diagonal is above the cap, and `1`
/// where it is not.
fn ceiling_factor(sigma: Number, cap: Number) -> Number {
    if sigma.abs() > cap {
        cap / sigma.abs()
    } else {
        1.0
    }
}

/// The classification came out the same over both engines.
pub fn assert_same_activity(label: &str, a: &ActivityReport, b: &ActivityReport) {
    let (nv, nr) = (a.var_status.len(), a.row_status.len());
    assert_same_activity_within(label, a, b, &Allowance::none(nv, nr));
}

/// [`assert_same_activity`], with what [`Allowance`] allows for.
pub fn assert_same_activity_within(
    label: &str,
    a: &ActivityReport,
    b: &ActivityReport,
    allow: &Allowance,
) {
    let mu = a.mu;
    assert_same_statuses(
        label,
        "var_status",
        &a.var_status,
        &b.var_status,
        &a.var_ratio,
        &b.var_ratio,
        mu,
    );
    assert_same_statuses(
        label,
        "row_status",
        &a.row_status,
        &b.row_status,
        &a.row_ratio,
        &b.row_ratio,
        mu,
    );
    assert_eq!(
        a.var_q_sign, b.var_q_sign,
        "{label}: variable curvature signs differ"
    );
    assert_eq!(
        a.row_q_sign, b.row_q_sign,
        "{label}: row curvature signs differ"
    );
    assert_eq!(
        a.var_off_central_path, b.var_off_central_path,
        "{label}: variable central-path flags differ"
    );
    assert_eq!(
        a.row_off_central_path, b.row_off_central_path,
        "{label}: row central-path flags differ"
    );
    assert_eq!(
        a.var_contaminated, b.var_contaminated,
        "{label}: variable contamination differs"
    );
    assert_eq!(
        a.row_contaminated, b.row_contaminated,
        "{label}: row contamination differs"
    );
    for (name, row, xa, xb) in [
        ("var_ratio", false, &a.var_ratio, &b.var_ratio),
        ("var_sigma", false, &a.var_sigma, &b.var_sigma),
        ("row_ratio", true, &a.row_ratio, &b.row_ratio),
        ("row_sigma", true, &a.row_sigma, &b.row_sigma),
    ] {
        let (status, moved, cap, sigma) = if row {
            (&a.row_status, &allow.row_move, &allow.row_cap, &b.row_sigma)
        } else {
            (&a.var_status, &allow.var_move, &allow.var_cap, &b.var_sigma)
        };
        for (k, (&u, &v)) in xa.iter().zip(xb).enumerate() {
            let rtol = if row && status[k] == INACTIVE {
                INACTIVE_RTOL
            } else {
                RATIO_RTOL
            };
            let expected = v * ceiling_factor(sigma[k], cap[k]);
            assert!(
                close(u, expected, rtol + moved[k], 0.0),
                "{label}: {name}[{k}] is {u:e} over pounce and {v:e} over the outside engine, \
                 {expected:e} after pounce's ceiling",
            );
        }
    }
    assert!(
        close(a.mu, b.mu, RATIO_RTOL, 0.0),
        "{label}: mu {} vs {}",
        a.mu,
        b.mu
    );
}

/// The reduced curvature past which an equality determines the coordinate,
/// as `a_coordinate_an_equality_determines_has_no_direction_left_to_reduce_along`
/// states it. There `(K⁻¹)ᵢᵢ` is zero in exact arithmetic, so the curvature
/// is infinite, and an engine's factorization returns zero or a roundoff-sized
/// number of either sign. Past this both engines say the same thing, and the
/// value, its sign and the ratio it divides to zero are roundoff.
///
/// The curvature has units, so the cutoff is a gap between two measured
/// populations rather than a dimensionless constant. Over the 113 entries
/// the suite compares, the genuine curvatures reach `1.8e10` (on
/// `cross_engine_scaled`) and the determined ones start at `1.1e30` (the
/// `coupled` leg at row scale `1e-3`). The cutoff sits at their geometric
/// middle, ten orders from each. A fixture whose genuine curvature passes
/// `1e20` would be compared on its status alone; re-measure before adding
/// one in small units.
const DETERMINED: Number = 1e20;

/// One reduced report's entries against the other's.
#[allow(clippy::too_many_arguments)]
fn assert_same_reduced(
    label: &str,
    inactive_rtol: Number,
    mu: Number,
    status: (&[i8], &[i8]),
    q_sign: (&[i8], &[i8]),
    ratio: (&[Number], &[Number]),
    q_reduced: (&[Number], &[Number]),
    sigma: (&[Number], &[Number]),
) {
    assert_same_statuses(label, "status", status.0, status.1, ratio.0, ratio.1, mu);
    let determined = |q: Number| q.is_infinite() || q.abs() > DETERMINED;
    for k in 0..status.0.len() {
        let both = determined(q_reduced.0[k]) && determined(q_reduced.1[k]);
        let rtol = if status.0[k] == INACTIVE {
            inactive_rtol
        } else {
            RATIO_RTOL
        };
        if !both {
            assert_eq!(
                q_sign.0[k], q_sign.1[k],
                "{label}: reduced curvature sign [{k}] differs",
            );
        }
        for (name, xa, xb, skip) in [
            ("ratio", ratio.0, ratio.1, both),
            ("q_reduced", q_reduced.0, q_reduced.1, both),
            ("sigma", sigma.0, sigma.1, false),
        ] {
            let (u, v) = (xa[k], xb[k]);
            assert!(
                skip || close(u, v, rtol, 0.0),
                "{label}: {name}[{k}] is {u:e} over pounce and {v:e} over the outside engine",
            );
        }
    }
}

/// The reduced classification of the variables came out the same.
pub fn assert_same_reduced_activity(
    label: &str,
    a: &ReducedActivityReport,
    b: &ReducedActivityReport,
) {
    assert_eq!(a.var, b.var, "{label}: different variables answered");
    assert_same_reduced(
        label,
        RATIO_RTOL,
        a.mu,
        (&a.status, &b.status),
        (&a.q_sign, &b.q_sign),
        (&a.ratio, &b.ratio),
        (&a.q_reduced, &b.q_reduced),
        (&a.sigma, &b.sigma),
    );
}

/// The reduced classification of the rows came out the same.
pub fn assert_same_reduced_row_activity(
    label: &str,
    a: &ReducedRowActivityReport,
    b: &ReducedRowActivityReport,
) {
    assert_eq!(a.row, b.row, "{label}: different rows answered");
    assert_same_reduced(
        label,
        INACTIVE_RTOL,
        a.mu,
        (&a.status, &b.status),
        (&a.q_sign, &b.q_sign),
        (&a.ratio, &b.ratio),
        (&a.q_reduced, &b.q_reduced),
        (&a.sigma, &b.sigma),
    );
}
