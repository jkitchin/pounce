//! Policy for solving a model with declared complementarity pairs: which
//! lowerings of [`pounce_nlp::mpcc::MpccTnlp`] to try, in what order, and
//! what counts as an answer.
//!
//! Workstream B of `dev-notes/research/calipso-contact-plan.md`. Like
//! [`crate::second_opinion`], this module holds the policy and runs no
//! solves; the driver that runs them is
//! `pounce_restoration::mpcc_driver`, so every entry point shares one
//! implementation.
//!
//! ## Why a ladder and not one route
//!
//! Measured on `benchmarks/calipso/` (seven complementarity models, POUNCE
//! and Ipopt), no single lowering wins: the exact product is fast and
//! solves most models from most starts but fails where a pair is biactive
//! at the answer; the Scholtes continuation with full warm starts and an
//! exact finish was the only route physical on 5/5 starts on bunny-hop and
//! the drifting car, at several times the iterations; the ℓ₁ penalty was the
//! only route that solved the quadruped on every seed. So the driver tries
//! them in [`STAGES`] order and stops at the first answer that passes
//! [`accepts`].
//!
//! ## What an answer is
//!
//! A converged status is not enough: a lowered solve "succeeds" on its own
//! reformulation, and the reformulation's tolerance is not the model's. A
//! stage is accepted only if it converged **and** every product `G·H` is
//! within `constr_viol_tol` — the bar the model's own rows are held to.

use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;

/// The stages, in the order the driver tries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpccStage {
    /// `G·H = 0` from the model's own start.
    Exact,
    /// `G·H <= τ` for τ down [`SCHOLTES_SCHEDULE`], each stage warm-started
    /// from the last, then one `G·H = 0` solve from the end of it
    /// (`G·H <= 0` where the equality would leave the model with more
    /// equalities than variables).
    Continuation,
    /// `G·H = 0` through the ℓ₁ exact-penalty barrier.
    Penalty,
}

impl MpccStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact product",
            Self::Continuation => "Scholtes continuation",
            Self::Penalty => "l1 penalty",
        }
    }
}

pub const STAGES: [MpccStage; 3] = [
    MpccStage::Exact,
    MpccStage::Continuation,
    MpccStage::Penalty,
];

/// The relaxation schedule, as in `benchmarks/mpcc/routes.py`: it starts at
/// 1 so a relaxation whose crossover lies in `[0.25, 1]` is seen, and ends
/// four orders below the default `constr_viol_tol`.
pub const SCHOLTES_SCHEDULE: [Number; 9] = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8];

/// The options a warm-started continuation stage sets, as
/// `pounce.WarmStart.options()` does: the seeds are honoured, the barrier
/// starts where the last stage stopped instead of re-walking from 0.1, and
/// the pushes are tightened so a point at a bound stays there. Each is an
/// upstream Ipopt option. The driver restores every one of them to the
/// caller's set-ness afterwards.
pub fn warm_start_assignments(final_mu: Number) -> Vec<String> {
    let mu = if final_mu.is_finite() && final_mu > 0.0 {
        final_mu.clamp(1e-9, 1e-1)
    } else {
        1e-6
    };
    let p = 1e-9;
    vec![
        "warm_start_init_point yes\n".to_string(),
        format!("mu_init {mu:e}\n"),
        format!("warm_start_bound_push {p:e}\n"),
        format!("warm_start_bound_frac {p:e}\n"),
        format!("warm_start_slack_bound_push {p:e}\n"),
        format!("warm_start_slack_bound_frac {p:e}\n"),
        format!("warm_start_mult_bound_push {p:e}\n"),
    ]
}

/// Every option any stage writes; the driver snapshots these before the
/// first stage and restores them after the last.
pub const TOUCHED_OPTIONS: [&str; 8] = [
    "warm_start_init_point",
    "mu_init",
    "warm_start_bound_push",
    "warm_start_bound_frac",
    "warm_start_slack_bound_push",
    "warm_start_slack_bound_frac",
    "warm_start_mult_bound_push",
    "l1_exact_penalty_barrier",
];

/// Statuses a stage can converge with.
pub fn converged(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded
            | ApplicationReturnStatus::SolvedToAcceptableLevel
            | ApplicationReturnStatus::FeasiblePointFound
    )
}

/// Is a stage's answer an answer to the *model*, not just to the lowering?
pub fn accepts(
    status: ApplicationReturnStatus,
    max_abs_product: Number,
    constr_viol_tol: Number,
) -> bool {
    converged(status) && max_abs_product.is_finite() && max_abs_product <= constr_viol_tol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_converged_lowering_with_a_large_product_is_not_an_answer() {
        let ok = ApplicationReturnStatus::SolveSucceeded;
        assert!(accepts(ok, 1e-9, 1e-4));
        assert!(!accepts(ok, 1e-3, 1e-4));
        assert!(!accepts(ok, Number::NAN, 1e-4));
        assert!(!accepts(
            ApplicationReturnStatus::InfeasibleProblemDetected,
            0.0,
            1e-4
        ));
        assert!(accepts(
            ApplicationReturnStatus::FeasiblePointFound,
            0.0,
            1e-4
        ));
    }

    #[test]
    fn warm_mu_is_clamped_and_never_zero() {
        let mu = |v| {
            warm_start_assignments(v)
                .into_iter()
                .find(|a| a.starts_with("mu_init"))
                .unwrap()
        };
        assert_eq!(mu(1e-12), "mu_init 1e-9\n");
        assert_eq!(mu(3.0), "mu_init 1e-1\n");
        assert_eq!(mu(0.0), "mu_init 1e-6\n");
        assert_eq!(mu(Number::NAN), "mu_init 1e-6\n");
    }

    #[test]
    fn every_assignment_is_restored() {
        for a in warm_start_assignments(1e-5) {
            let name = a.split_whitespace().next().unwrap();
            assert!(
                TOUCHED_OPTIONS.contains(&name),
                "{name} is written but not restored"
            );
        }
    }
}
