//! Diagnostics for the auxiliary-equality preprocessing pass (Phase 0).
//!
//! Populated across PRs 1, 8, and 9 of the auxiliary-presolve port
//! (issue #53). PR 9 expanded the struct with per-stage timings,
//! per-coupling-class counts, and a human-readable `Display` impl
//! so users can pipe `wrapped.auxiliary_diagnostics()` straight to
//! a log line.

use std::fmt;

use pounce_common::types::{Index, Number};

/// Reasons the orchestrator may decline to eliminate a candidate block.
///
/// PR 1 wires the enum so it can live in the diagnostics struct; the
/// populating logic lands with PR 5 (coupling classification) and PR 6
/// (block solve).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxiliaryRejectionReason {
    /// Block too large for the lightweight Newton solver and no IPM
    /// fallback installed (PR 11).
    BlockTooLarge,
    /// Block is coupled to inequality rows or the objective in a way
    /// the current coupling policy disallows.
    CouplingDisallowed,
    /// Newton diverged or hit `presolve_auxiliary_max_iter`.
    BlockSolveDiverged,
    /// Full-space KKT residual after the candidate reduction exceeded
    /// `presolve_auxiliary_tol`.
    ResidualCheckFailed,
}

/// Per-stage wall-time breakdown for one Phase-0 pass.
#[derive(Debug, Clone, Default, Copy)]
pub struct StageTimings {
    pub incidence_ms: u128,
    pub matching_ms: u128,
    pub dm_ms: u128,
    pub components_ms: u128,
    pub btf_ms: u128,
    pub block_solve_ms: u128,
    pub residual_check_ms: u128,
}

/// Count of candidate blocks broken down by
/// [`crate::coupling::AuxiliaryCouplingClass`].
#[derive(Debug, Clone, Default, Copy)]
pub struct ClassCounts {
    pub pure_equality: Index,
    pub objective_coupled: Index,
    pub inequality_coupled: Index,
    pub objective_and_inequality_coupled: Index,
}

/// Per-run summary of what the auxiliary-equality preprocessing pass
/// did. All counters are zeroed by [`Default::default`].
///
/// # Example
///
/// ```
/// use pounce_presolve::diagnostics::AuxiliaryPreprocessingDiagnostics;
///
/// let d = AuxiliaryPreprocessingDiagnostics::default();
/// assert_eq!(d.blocks_eliminated, 0);
/// assert_eq!(d.candidate_blocks, 0);
/// assert!(d.rejection_reasons.is_empty());
/// ```
#[derive(Debug, Clone, Default)]
pub struct AuxiliaryPreprocessingDiagnostics {
    /// Number of blocks the orchestrator successfully eliminated.
    pub blocks_eliminated: Index,
    /// Total candidate blocks examined (eliminated + rejected).
    pub candidate_blocks: Index,
    /// Variables fixed by the eliminated blocks (sum of block dims).
    pub vars_eliminated: Index,
    /// Equality rows dropped from the reduced problem.
    pub rows_eliminated: Index,
    /// Total wall time spent inside Phase 0, in milliseconds.
    pub total_time_ms: u128,
    /// Per-stage wall-time breakdown.
    pub stage_time_ms: StageTimings,
    /// Per-coupling-class candidate counts.
    pub class_counts: ClassCounts,
    /// Largest block-solve residual accepted under
    /// `presolve_auxiliary_tol`. `0.0` when nothing was eliminated.
    pub max_block_residual: Number,
    /// Dimension of the largest accepted block.
    pub max_accepted_block_dim: Index,
    /// One entry per rejected candidate, in encounter order.
    pub rejection_reasons: Vec<AuxiliaryRejectionReason>,
}

impl fmt::Display for AuxiliaryPreprocessingDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "auxiliary-preprocessing: {} of {} candidate block(s) eliminated, \
             fixing {} variable(s) and dropping {} row(s) in {} ms",
            self.blocks_eliminated,
            self.candidate_blocks,
            self.vars_eliminated,
            self.rows_eliminated,
            self.total_time_ms,
        )?;
        if self.blocks_eliminated > 0 {
            writeln!(
                f,
                "  max block dim: {}, max residual: {:.3e}",
                self.max_accepted_block_dim, self.max_block_residual
            )?;
        }
        let cc = &self.class_counts;
        if cc.pure_equality
            + cc.objective_coupled
            + cc.inequality_coupled
            + cc.objective_and_inequality_coupled
            > 0
        {
            writeln!(
                f,
                "  coupling: pure={}, obj={}, ineq={}, both={}",
                cc.pure_equality,
                cc.objective_coupled,
                cc.inequality_coupled,
                cc.objective_and_inequality_coupled,
            )?;
        }
        if !self.rejection_reasons.is_empty() {
            writeln!(f, "  rejections ({}):", self.rejection_reasons.len())?;
            // Tally by reason.
            let mut by_reason: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            for r in &self.rejection_reasons {
                let key = match r {
                    AuxiliaryRejectionReason::BlockTooLarge => "block-too-large",
                    AuxiliaryRejectionReason::CouplingDisallowed => "coupling-disallowed",
                    AuxiliaryRejectionReason::BlockSolveDiverged => "block-solve-diverged",
                    AuxiliaryRejectionReason::ResidualCheckFailed => "residual-check-failed",
                };
                *by_reason.entry(key).or_insert(0) += 1;
            }
            for (reason, count) in by_reason {
                writeln!(f, "    {reason}: {count}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_default_is_empty() {
        let d = AuxiliaryPreprocessingDiagnostics::default();
        assert_eq!(d.blocks_eliminated, 0);
        assert_eq!(d.candidate_blocks, 0);
        assert_eq!(d.vars_eliminated, 0);
        assert_eq!(d.rows_eliminated, 0);
        assert_eq!(d.total_time_ms, 0);
        assert_eq!(d.max_block_residual, 0.0);
        assert_eq!(d.max_accepted_block_dim, 0);
        assert_eq!(d.stage_time_ms.matching_ms, 0);
        assert_eq!(d.class_counts.pure_equality, 0);
        assert!(d.rejection_reasons.is_empty());
    }

    #[test]
    fn display_empty_diagnostics() {
        let d = AuxiliaryPreprocessingDiagnostics::default();
        let s = format!("{d}");
        assert!(s.contains("0 of 0 candidate block(s) eliminated"));
        // No rejections or class lines when everything is zero.
        assert!(!s.contains("rejections"));
        assert!(!s.contains("coupling"));
    }

    #[test]
    fn display_populated_diagnostics() {
        let mut d = AuxiliaryPreprocessingDiagnostics {
            blocks_eliminated: 2,
            candidate_blocks: 3,
            vars_eliminated: 4,
            rows_eliminated: 4,
            total_time_ms: 12,
            max_block_residual: 1.5e-13,
            max_accepted_block_dim: 2,
            ..Default::default()
        };
        d.class_counts.pure_equality = 2;
        d.class_counts.inequality_coupled = 1;
        d.rejection_reasons
            .push(AuxiliaryRejectionReason::CouplingDisallowed);
        let s = format!("{d}");
        assert!(s.contains("2 of 3 candidate block(s) eliminated"));
        assert!(s.contains("max block dim: 2"));
        assert!(s.contains("max residual: 1.500e-13"));
        assert!(s.contains("pure=2"));
        assert!(s.contains("ineq=1"));
        assert!(s.contains("coupling-disallowed: 1"));
    }
}
