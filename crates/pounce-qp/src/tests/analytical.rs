//! Analytical correctness ladder (§8.0 of the design note). Six
//! closed-form QPs with hand-computable answers; runtime budget
//! <50 ms total. Each catches a distinct class of bug at the
//! earliest possible point.
//!
//! Phase 5a scaffold: the ladder is enumerated here so the contract
//! is visible from day one. The actual `#[test]` functions land in
//! the commit that introduces the cold-start solver (KKT assembly +
//! single linsol call). At that point this module gains:
//!
//! 1. `unconstrained_identity_hessian` — `x* = −g`, one Newton step.
//!    Catches: KKT sign, gradient assembly.
//! 2. `equality_only_full_rank` — `[H Aᵀ; A 0]⁻¹ [−g; b]`. Catches:
//!    KKT block layout, multiplier sign convention.
//! 3. `box_constrained_diagonal_hessian` — `x*_i = clip(−gᵢ/hᵢ,
//!    xlᵢ, xuᵢ)` per coordinate. Catches: bound-multiplier sign,
//!    working-set add/drop.
//! 4. `redundant_equality` — strictly convex QP with one redundant
//!    constraint; redundant row stays inactive at optimum. Catches:
//!    degeneracy detection, EXPAND triggering.
//! 5. `infeasible_bounds` — `xl > xu` on one coord; elastic mode
//!    returns minimal-infeasibility point. Catches: §4.3 phase-1
//!    elastic detection.
//! 6. `indefinite_h_pd_reduced` — indefinite `H`, single equality,
//!    reduced Hessian PD. Catches: §4.5 inertia-control trigger.
