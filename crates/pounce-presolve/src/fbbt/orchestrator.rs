//! FBBT outer loop: sweep all constraints to tighten variable bounds
//! to a fixed point (or `max_iter`).
//!
//! ```text
//! for iter in 0..max_iter:
//!     for each constraint i:
//!         tape = provider.constraint_expression(i)   # None ⇒ skip
//!         forward = forward_pass(tape, x_lo, x_hi)
//!         result  = reverse_pass(tape, &forward, [g_lo[i], g_hi[i]])
//!         if result.infeasible: report and bail
//!         for each Var(j) slot s in tape:
//!             new_bound = result.slots[s]
//!             tighten x_lo[j], x_hi[j] against new_bound
//!             if improvement > tol: mark progress
//!     if no progress this iter: break
//! ```
//!
//! Matches the Belotti, Cafieri, Lee, Liberti (2010) algorithm,
//! including the Gauss-Seidel-style sweep (each constraint sees the
//! freshly tightened bounds from earlier constraints in the same
//! iteration). Tolerance-based termination — FBBT does not converge
//! finitely in general.
//!
//! Issue [#62].
//!
//! [#62]: https://github.com/jkitchin/pounce/issues/62

use pounce_common::types::Number;
use pounce_nlp::expression_provider::{ExpressionProvider, FbbtOp};

use crate::fbbt::forward::forward_pass;
use crate::fbbt::interval::Interval;
use crate::fbbt::reverse::reverse_pass;

/// Knobs for [`run_fbbt`]. Defaults match the proposed `presolve_*`
/// option set in issue #62.
#[derive(Debug, Clone, Copy)]
pub struct FbbtConfig {
    /// Minimum bound improvement (in absolute units of the variable)
    /// to keep iterating. Per Belotti et al., FBBT must terminate by
    /// tolerance, not by convergence.
    pub tol: Number,
    /// Outer sweep cap.
    pub max_iter: usize,
    /// Cap on the number of constraints to examine per sweep. `0`
    /// means unlimited. Useful as a wall-time guard on very large
    /// problems where the first few constraints carry most of the
    /// tightening.
    pub max_constraints: usize,
}

impl Default for FbbtConfig {
    fn default() -> Self {
        Self {
            tol: 1.0e-6,
            max_iter: 10,
            max_constraints: 0,
        }
    }
}

/// What the orchestrator did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FbbtReport {
    /// Number of outer sweeps actually executed (≤ `cfg.max_iter`).
    pub iterations: usize,
    /// Total number of `(variable, bound)` tightening events across
    /// all sweeps and all constraints.
    pub bound_updates: usize,
    /// Index of the constraint that proved infeasibility, if any.
    /// When set, the variable bounds in the caller's arrays are
    /// undefined and must not be trusted.
    pub infeasibility_witness: Option<usize>,
    /// Sum of absolute bound improvements across all updates — for
    /// reporting, not part of the algorithm.
    pub total_tightening: Number,
}

/// Run FBBT against `provider` until quiescent or `cfg.max_iter`.
///
/// `x_lo` / `x_hi` are read AND written. They start as the user's
/// declared variable bounds and end as the FBBT-tightened bounds. On
/// detected infeasibility, the contents are left in their
/// partially-updated state and `report.infeasibility_witness` is
/// `Some(constraint_idx)`.
///
/// `g_lo` / `g_hi` are the constraint bounds, length `m`. Providers
/// that return `None` for a constraint index are skipped silently
/// (FBBT can't tighten without a structural expression).
pub fn run_fbbt(
    provider: &dyn ExpressionProvider,
    n_vars: usize,
    n_constraints: usize,
    x_lo: &mut [Number],
    x_hi: &mut [Number],
    g_lo: &[Number],
    g_hi: &[Number],
    cfg: &FbbtConfig,
) -> FbbtReport {
    let mut report = FbbtReport::default();

    assert_eq!(x_lo.len(), n_vars, "x_lo length");
    assert_eq!(x_hi.len(), n_vars, "x_hi length");
    assert_eq!(g_lo.len(), n_constraints, "g_lo length");
    assert_eq!(g_hi.len(), n_constraints, "g_hi length");

    let cap = if cfg.max_constraints == 0 {
        n_constraints
    } else {
        cfg.max_constraints.min(n_constraints)
    };

    for _iter in 0..cfg.max_iter {
        report.iterations += 1;
        let mut improved = false;

        for i in 0..cap {
            let Some(tape) = provider.constraint_expression(i) else {
                continue;
            };
            if tape.is_empty() {
                continue;
            }

            let forward = match forward_pass(&tape, x_lo, x_hi) {
                Ok(v) => v,
                Err(_) => continue, // Malformed tape or out-of-range — skip safely.
            };
            let bound = Interval::new(g_lo[i], g_hi[i]);
            let reverse = reverse_pass(&tape, &forward, bound);
            if reverse.infeasible {
                report.infeasibility_witness = Some(i);
                return report;
            }

            // Aggregate per-variable tightening: a variable can
            // appear in multiple `Var(j)` slots of the tape (when
            // the constraint references it without CSE sharing).
            // Each slot may carry a different reverse-propagated
            // interval; the variable's tightened interval is the
            // INTERSECTION of all those slot intervals.
            let mut tighten: Vec<Interval> = vec![Interval::ENTIRE; n_vars];
            for (slot_idx, op) in tape.ops.iter().enumerate() {
                if let FbbtOp::Var(j) = *op {
                    tighten[j] = tighten[j].intersect(reverse.slots[slot_idx]);
                }
            }

            // Apply.
            for j in 0..n_vars {
                let t = tighten[j];
                if t.is_empty() {
                    report.infeasibility_witness = Some(i);
                    return report;
                }
                if t.is_entire() {
                    continue;
                }
                let new_lo = x_lo[j].max(t.lo);
                let new_hi = x_hi[j].min(t.hi);
                if new_lo > new_hi {
                    report.infeasibility_witness = Some(i);
                    return report;
                }
                let delta_lo = (new_lo - x_lo[j]).max(0.0);
                let delta_hi = (x_hi[j] - new_hi).max(0.0);
                let delta = delta_lo.max(delta_hi);
                if delta > cfg.tol {
                    x_lo[j] = new_lo;
                    x_hi[j] = new_hi;
                    report.bound_updates += 1;
                    report.total_tightening += delta;
                    improved = true;
                } else if delta_lo > 0.0 || delta_hi > 0.0 {
                    // Tiny tightening below tol — apply but don't
                    // count as progress.
                    x_lo[j] = new_lo;
                    x_hi[j] = new_hi;
                }
            }
        }

        if !improved {
            break;
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_nlp::expression_provider::{FbbtOp, FbbtTape};

    /// Test helper: a provider that just returns a stored vec of
    /// tapes, one per constraint.
    struct StubProvider {
        tapes: Vec<Option<FbbtTape>>,
    }

    impl ExpressionProvider for StubProvider {
        fn constraint_expression(&self, i: usize) -> Option<FbbtTape> {
            self.tapes.get(i).and_then(|t| t.clone())
        }
    }

    /// `x² + y² = 1` with initial box `[-10, 10]²`. Each variable
    /// should be tightened to a subset of `[-1, 1]`.
    #[test]
    fn unit_circle_tightens_box() {
        let tape = FbbtTape {
            ops: vec![
                FbbtOp::Var(0),
                FbbtOp::PowInt(0, 2),
                FbbtOp::Var(1),
                FbbtOp::PowInt(2, 2),
                FbbtOp::Add(1, 3),
            ],
        };
        let provider = StubProvider {
            tapes: vec![Some(tape)],
        };
        let mut x_lo = vec![-10.0, -10.0];
        let mut x_hi = vec![10.0, 10.0];
        let r = run_fbbt(
            &provider,
            2,
            1,
            &mut x_lo,
            &mut x_hi,
            &[1.0],
            &[1.0],
            &FbbtConfig::default(),
        );
        assert!(r.infeasibility_witness.is_none());
        // Both variables must tighten (one update each, per-variable).
        assert!(r.bound_updates >= 2, "got {} updates", r.bound_updates);
        for (lo, hi) in x_lo.iter().zip(&x_hi) {
            assert!(*lo >= -1.0 - 1e-6, "lo = {lo}");
            assert!(*hi <= 1.0 + 1e-6, "hi = {hi}");
        }
    }

    /// `exp(x) ≤ 10` ⇒ `x ≤ ln 10 ≈ 2.302`.
    #[test]
    fn exp_upper_bound_tightens() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Exp(0)],
        };
        let provider = StubProvider {
            tapes: vec![Some(tape)],
        };
        let mut x_lo = vec![-10.0];
        let mut x_hi = vec![10.0];
        let r = run_fbbt(
            &provider,
            1,
            1,
            &mut x_lo,
            &mut x_hi,
            &[Number::NEG_INFINITY],
            &[10.0],
            &FbbtConfig::default(),
        );
        assert!(r.infeasibility_witness.is_none());
        // ln(10) ≈ 2.3026.
        assert!(x_hi[0] <= 2.31, "x_hi = {}", x_hi[0]);
        // Lower bound unaffected by an upper-only constraint.
        assert_eq!(x_lo[0], -10.0);
    }

    /// Cross-constraint iteration: constraint A tightens y, after
    /// which constraint B (which mentions y on its RHS) can tighten
    /// x further than a single pass would.
    ///
    /// * A: `y² ≤ 1` ⇒ `y ∈ [-1, 1]` (tightens y from [-10, 10]).
    /// * B: `x + y² = 0.5` ⇒ once y ∈ [-1, 1], y² ∈ [0, 1], so
    ///   `x = 0.5 - y² ∈ [-0.5, 0.5]`. Before A runs, B would only
    ///   tighten x to `[0.5 - 100, 0.5 - 0] = [-99.5, 0.5]`.
    #[test]
    fn coupled_constraints_iterate() {
        let tape_a = FbbtTape {
            ops: vec![FbbtOp::Var(1), FbbtOp::PowInt(0, 2)],
        };
        let tape_b = FbbtTape {
            ops: vec![
                FbbtOp::Var(0),
                FbbtOp::Var(1),
                FbbtOp::PowInt(1, 2),
                FbbtOp::Add(0, 2),
            ],
        };
        let provider = StubProvider {
            tapes: vec![Some(tape_a), Some(tape_b)],
        };
        let mut x_lo = vec![-10.0, -10.0];
        let mut x_hi = vec![10.0, 10.0];
        let r = run_fbbt(
            &provider,
            2,
            2,
            &mut x_lo,
            &mut x_hi,
            &[Number::NEG_INFINITY, 0.5],
            &[1.0, 0.5],
            &FbbtConfig::default(),
        );
        assert!(r.infeasibility_witness.is_none());
        // y was tightened to [-1, 1].
        assert!(x_lo[1] >= -1.0 - 1e-6, "y_lo = {}", x_lo[1]);
        assert!(x_hi[1] <= 1.0 + 1e-6, "y_hi = {}", x_hi[1]);
        // x was tightened to [-0.5, 0.5] — only achievable when the
        // first sweep gave y² ≤ 1 before constraint B fires.
        assert!(x_lo[0] >= -0.5 - 1e-6, "x_lo = {}", x_lo[0]);
        assert!(x_hi[0] <= 0.5 + 1e-6, "x_hi = {}", x_hi[0]);
    }

    /// FBBT should detect infeasibility: x ∈ [10, 20] but
    /// `x ∈ [1, 5]` from the constraint.
    #[test]
    fn detects_infeasibility() {
        let tape = FbbtTape {
            ops: vec![FbbtOp::Var(0)],
        };
        let provider = StubProvider {
            tapes: vec![Some(tape)],
        };
        let mut x_lo = vec![10.0];
        let mut x_hi = vec![20.0];
        let r = run_fbbt(
            &provider,
            1,
            1,
            &mut x_lo,
            &mut x_hi,
            &[1.0],
            &[5.0],
            &FbbtConfig::default(),
        );
        assert_eq!(r.infeasibility_witness, Some(0));
    }

    /// Constraint without expression (provider returns None) →
    /// no-op, no tightening, no infeasibility.
    #[test]
    fn missing_expression_is_silent_noop() {
        let provider = StubProvider { tapes: vec![None] };
        let mut x_lo = vec![-1.0];
        let mut x_hi = vec![1.0];
        let r = run_fbbt(
            &provider,
            1,
            1,
            &mut x_lo,
            &mut x_hi,
            &[-100.0],
            &[100.0],
            &FbbtConfig::default(),
        );
        assert!(r.infeasibility_witness.is_none());
        assert_eq!(r.bound_updates, 0);
        assert_eq!(x_lo, vec![-1.0]);
        assert_eq!(x_hi, vec![1.0]);
    }

    /// Max-iter cap: a fixed-point that needs many sweeps must stop
    /// at `cfg.max_iter`. We test by setting `max_iter = 1` and
    /// observing the bound is loose.
    #[test]
    fn max_iter_caps_iteration_count() {
        let tape_sum = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Var(1), FbbtOp::Add(0, 1)],
        };
        let tape_diff = FbbtTape {
            ops: vec![FbbtOp::Var(0), FbbtOp::Var(1), FbbtOp::Sub(0, 1)],
        };
        let provider = StubProvider {
            tapes: vec![Some(tape_sum), Some(tape_diff)],
        };
        let mut x_lo = vec![-10.0, -10.0];
        let mut x_hi = vec![10.0, 10.0];
        let cfg = FbbtConfig {
            tol: 1e-6,
            max_iter: 1,
            max_constraints: 0,
        };
        let r = run_fbbt(
            &provider,
            2,
            2,
            &mut x_lo,
            &mut x_hi,
            &[1.0, 0.0],
            &[1.0, 0.0],
            &cfg,
        );
        assert!(r.infeasibility_witness.is_none());
        assert_eq!(r.iterations, 1);
        // After one sweep the box should still be much wider than
        // 1e-3 (the converged width seen in the previous test).
        let width0 = x_hi[0] - x_lo[0];
        let width1 = x_hi[1] - x_lo[1];
        assert!(
            width0 > 1e-3 || width1 > 1e-3,
            "single sweep already converged unexpectedly"
        );
    }

    /// `max_constraints` caps the per-sweep workload.
    #[test]
    fn max_constraints_truncates_sweep() {
        let tape_a = FbbtTape {
            ops: vec![FbbtOp::Var(0)],
        };
        let tape_b = FbbtTape {
            ops: vec![FbbtOp::Var(1)],
        };
        let provider = StubProvider {
            tapes: vec![Some(tape_a), Some(tape_b)],
        };
        let mut x_lo = vec![-10.0, -10.0];
        let mut x_hi = vec![10.0, 10.0];
        let cfg = FbbtConfig {
            tol: 1e-6,
            max_iter: 5,
            max_constraints: 1, // skip constraint 1
        };
        let _ = run_fbbt(
            &provider,
            2,
            2,
            &mut x_lo,
            &mut x_hi,
            &[-1.0, -1.0],
            &[1.0, 1.0],
            &cfg,
        );
        // x_0 must have tightened, x_1 untouched.
        assert!(x_lo[0] >= -1.0 - 1e-12);
        assert!(x_hi[0] <= 1.0 + 1e-12);
        assert_eq!(x_lo[1], -10.0);
        assert_eq!(x_hi[1], 10.0);
    }

    /// Soundness fuzz on a quadratic: any feasible point of the
    /// original problem must still be feasible w.r.t. the
    /// FBBT-tightened bounds.
    #[test]
    fn fuzz_soundness_pointwise() {
        // y² + x = 5, with original bounds x ∈ [-10, 5], y ∈ [-3, 3].
        let tape = FbbtTape {
            ops: vec![
                FbbtOp::Var(1),
                FbbtOp::PowInt(0, 2),
                FbbtOp::Var(0),
                FbbtOp::Add(1, 2),
            ],
        };
        let provider = StubProvider {
            tapes: vec![Some(tape)],
        };
        let mut x_lo = vec![-10.0, -3.0];
        let mut x_hi = vec![5.0, 3.0];
        let _ = run_fbbt(
            &provider,
            2,
            1,
            &mut x_lo,
            &mut x_hi,
            &[5.0],
            &[5.0],
            &FbbtConfig::default(),
        );
        // For y values on a grid, x = 5 - y²; test that (x, y) lies
        // inside the tightened box.
        for k in -30..=30 {
            let y = k as Number / 10.0;
            if !(-3.0..=3.0).contains(&y) {
                continue;
            }
            let x = 5.0 - y * y;
            if !(-10.0..=5.0).contains(&x) {
                continue;
            }
            assert!(
                x_lo[0] - 1e-6 <= x && x <= x_hi[0] + 1e-6,
                "feasible x={x} dropped (bounds {} .. {})",
                x_lo[0],
                x_hi[0]
            );
            assert!(
                x_lo[1] - 1e-6 <= y && y <= x_hi[1] + 1e-6,
                "feasible y={y} dropped"
            );
        }
    }
}
