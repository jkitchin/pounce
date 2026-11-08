//! What a frontend reports about a matrix-form convex solve: the final KKT
//! residual pair, the infeasibility certificate's scale, and the status in
//! the NLP side's [`ApplicationReturnStatus`] vocabulary.
//!
//! Each was written out by hand in more than one frontend — the residual
//! pair and certificate scale in `pounce-py`'s `solution_dict_hsde` and the
//! C API's `PounceQp*` handle, the status mapping in the CLI's
//! `qp_status_to_ars` and the C handle again — and moved here unchanged
//! (gh#1147 PR review) so that the CLI, Python and C call one definition.
//! Nothing here changes what any frontend reports.

use crate::cones::ConeSpec;
use crate::qp::{QpProblem, QpResiduals, QpSolution, QpStatus};
use pounce_nlp::return_codes::ApplicationReturnStatus;

/// The final KKT residuals a frontend reports, as `(reported, raw)`.
///
/// `raw` is [`QpSolution::kkt_residuals_conic`] against `cones` (with no
/// cones, the plain orthant residuals). `reported` is, on the plain QP/LP
/// path (no cones), [`QpSolution::kkt_residuals_above_floor`] — each residual
/// read above its own finite-precision floor and in the objective's units
/// (gh#984), the measurement the `Optimal` verdict is judged on, so a status
/// of optimal never sits beside `kkt_error > tol`; with cones it is `raw`.
pub fn reported_kkt_residuals(
    sol: &QpSolution,
    prob: &QpProblem,
    cones: &[ConeSpec],
) -> (QpResiduals, QpResiduals) {
    let raw = sol.kkt_residuals_conic(prob, cones);
    let r = if cones.is_empty() {
        sol.kkt_residuals_above_floor(prob)
    } else {
        raw
    };
    (r, raw)
}

/// The inf-norm of the certificate ray a solve returned: of `y`, `z`,
/// `z_lb`, `z_ub` when primal infeasible (the Farkas ray), of `x` when dual
/// infeasible (the recession ray), `None` on every other status. The factor
/// to divide by for a unit-norm certificate; the ray's own magnitude carries
/// the HSDE's `1/tau` and is meaningless.
pub fn certificate_scale(sol: &QpSolution) -> Option<f64> {
    let norm = |vs: &[&[f64]]| {
        vs.iter()
            .flat_map(|v| v.iter())
            .fold(0.0_f64, |m, x| m.max(x.abs()))
    };
    match sol.status {
        QpStatus::PrimalInfeasible => Some(norm(&[&sol.y, &sol.z, &sol.z_lb, &sol.z_ub])),
        QpStatus::DualInfeasible => Some(norm(&[&sol.x])),
        _ => None,
    }
}

/// Map the convex solver's status onto the NLP-side `ApplicationReturnStatus`
/// used by the JSON solve report, so QP and NLP reports share one status
/// vocabulary.
pub fn qp_status_to_ars(s: QpStatus) -> ApplicationReturnStatus {
    match s {
        QpStatus::Optimal => ApplicationReturnStatus::SolveSucceeded,
        // Reduced-accuracy solve (residual above `tol` but usable) — Ipopt's
        // "Solved To Acceptable Level" is the matching NLP-side status.
        QpStatus::OptimalInaccurate => ApplicationReturnStatus::SolvedToAcceptableLevel,
        QpStatus::PrimalInfeasible => ApplicationReturnStatus::InfeasibleProblemDetected,
        QpStatus::DualInfeasible => ApplicationReturnStatus::DivergingIterates, // unbounded
        QpStatus::IterationLimit => ApplicationReturnStatus::MaximumIterationsExceeded,
        QpStatus::TimeLimit => ApplicationReturnStatus::MaximumWallTimeExceeded,
        // gh#986: not `InternalError`, whose console rendering is "INTERNAL
        // ERROR: Unknown SolverReturn value." — a crash-shaped message for an
        // honest "no verified KKT point". `ErrorInStepComputation` is the
        // known status the NLP path uses for a linear-algebra breakdown.
        QpStatus::NumericalFailure => ApplicationReturnStatus::ErrorInStepComputation,
    }
}
