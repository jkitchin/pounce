//! Thierry-Biegler ℓ₁-exact penalty-barrier TNLP wrapper for POUNCE.
//!
//! Wraps a user [`TNLP`] so that the IPM solves the augmented problem
//!
//! ```text
//! min   f(x) + ρ · 1ᵀ(p + n)
//! s.t.  c_i(x) − p_i + n_i = g_i      for each equality row i
//!       c_i(x) ∈ [g_l_i, g_u_i]       for each inequality row i (unchanged)
//!       x_L ≤ x ≤ x_U,   p ≥ 0,   n ≥ 0
//! ```
//!
//! The augmented NLP automatically satisfies LICQ on `(p, n)`, which
//! is the property the method exploits to handle degenerate /
//! MPCC-like cases that the stock filter line search thrashes on.
//!
//! # Status
//!
//! Phase 1 — fixed `ρ` at construction time, default-off behind
//! `SolverOptions::l1_exact_penalty_barrier`. The wrapper performs
//! solution back-projection (truncate `x`, recompute `f(x*)`,
//! recompute `c(x*)`) inside [`L1PenaltyBarrierTnlp::finalize_solution`]
//! so the user sees original-space results even before the algorithm-
//! side wiring lands in Phase 2. Multiplier mapping refinement, BNW
//! dynamic ρ, and auto-fallback land in Phases 2 / 3 / 3.5.
//!
//! See [pounce#10] for the full plan.
//!
//! [pounce#10]: https://github.com/jkitchin/pounce/issues/10

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod options;
pub mod wrapper;

pub use options::L1PenaltyOptions;
pub use wrapper::L1PenaltyBarrierTnlp;
