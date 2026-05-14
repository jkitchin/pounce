//! Option-table entries for the ℓ₁-exact penalty-barrier wrapper.
//!
//! These mirror ripopt's `SolverOptions::l1_*` fields. Phase 1 ships
//! only the entries used by the wrapper itself
//! ([`L1PenaltyOptions::enabled`] and [`L1PenaltyOptions::rho`]).
//! Phase 3 adds the BNW-update parameters; Phase 3.5 adds the
//! auto-fallback gate.

use pounce_common::types::Number;

/// User-facing options for the ℓ₁ penalty-barrier wrapper.
///
/// Defaults match upstream Ipopt option-table conventions: the
/// wrapper is **off by default**; the user must opt in via the
/// `l1_exact_penalty_barrier` option key (registered in
/// `pounce-algorithm`'s upstream-options table by Phase 1's wiring).
#[derive(Debug, Clone, Copy)]
pub struct L1PenaltyOptions {
    /// Master switch. When `false` (default), `IpoptApplication` does
    /// not wrap the user TNLP and the solve trajectory is identical to
    /// pre-l1 pounce. Mapped from option key
    /// `l1_exact_penalty_barrier`.
    pub enabled: bool,
    /// Initial (and, for Phase 1, only) value of the penalty weight
    /// `ρ` applied to `1ᵀ(p + n)` in the augmented objective. Phase 3
    /// will replace this with a Byrd-Nocedal-Waltz dynamic update.
    /// Mapped from option key `l1_penalty_init`. Default `1.0`
    /// matches ripopt 0.8.0.
    pub rho: Number,
}

impl Default for L1PenaltyOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            rho: 1.0,
        }
    }
}

impl L1PenaltyOptions {
    /// Convenience constructor for tests and one-off solver setups
    /// that wire the wrapper explicitly without going through the
    /// option-table parser.
    pub fn enabled_with_rho(rho: Number) -> Self {
        Self {
            enabled: true,
            rho,
        }
    }
}
