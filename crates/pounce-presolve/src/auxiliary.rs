//! Phase-0 orchestrator for auxiliary-equality preprocessing.
//!
//! PR 8 of the auxiliary-presolve port (issue #53). This module runs
//! the full pipeline assembled in PRs 2-7:
//!
//! ```text
//!   incidence  → matching  → DM  → components  → BTF
//!     → classify  → linear block solve  → residual check
//!     → reduction frame
//! ```
//!
//! Scope: **linear blocks only**. A row counts as linear when its
//! `Linearity` tag is `Linear`. Nonlinear blocks need repeated
//! `eval_g` / `eval_jac_g` calls during Newton iteration; those
//! arrive in PR 8b. Linear blocks reuse the pre-fetched Jacobian
//! from `PresolveTnlp::ensure_init` and cover the common case.
//!
//! Variables are clamped (`x_l[i] = x_u[i] = value`) rather than
//! removed from the IPM's problem. This keeps the existing
//! `PresolveTnlp` row-mask machinery unchanged.
//!
//! Tracking issue: <https://github.com/jkitchin/pounce/issues/53>.

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::Linearity;

use crate::block_solve::{BlockEquations, BlockSolveOptions, BlockSolver, DampedNewtonSolver};
use crate::btf::BlockTriangularForm;
use crate::components::SquareComponents;
use crate::coupling::{classify_block, objective_gradient_support, AuxiliaryCouplingClass};
use crate::diagnostics::{AuxiliaryPreprocessingDiagnostics, AuxiliaryRejectionReason};
use crate::dulmage_mendelsohn::DulmageMendelsohnPartition;
use crate::incidence::{EqualityIncidence, InequalityIncidence, ProbeView};
use crate::matching::hopcroft_karp;
use crate::options::{AuxiliaryCouplingPolicy, PresolveOptions};
use crate::reduction_frame::ReductionFrame;

/// All the problem data Phase 0 needs, gathered once by
/// `PresolveTnlp::ensure_init` and passed in.
pub struct Phase0Probe<'a> {
    pub n_vars: usize,
    pub n_rows: usize,
    pub jac_irow: &'a [Index],
    pub jac_jcol: &'a [Index],
    pub jac_values: &'a [Number],
    pub g_l: &'a [Number],
    pub g_u: &'a [Number],
    pub g_at_probe: &'a [Number],
    pub linearity: &'a [Linearity],
    pub one_based: bool,
    pub eq_tol: Number,
    pub x_probe: &'a [Number],
    pub grad_f: &'a [Number],
}

/// Output of one Phase-0 pass.
#[derive(Debug, Clone)]
pub struct Phase0Plan {
    pub diagnostics: AuxiliaryPreprocessingDiagnostics,
    /// `Some` iff at least one block was accepted; describes the
    /// composite elimination to apply.
    pub frame: Option<ReductionFrame>,
}

impl Default for Phase0Plan {
    fn default() -> Self {
        Self {
            diagnostics: AuxiliaryPreprocessingDiagnostics::default(),
            frame: None,
        }
    }
}

/// Run the Phase-0 pipeline and return the resulting plan.
///
/// When `opts.auxiliary` is `false` this is a true no-op. When `true`,
/// the orchestrator walks every candidate block, accepts only those
/// allowed by `opts.auxiliary_coupling`, and accumulates a single
/// composite `ReductionFrame`.
pub fn run_auxiliary_phase0(opts: &PresolveOptions, probe: &Phase0Probe<'_>) -> Phase0Plan {
    let mut diag = AuxiliaryPreprocessingDiagnostics::default();
    if !opts.auxiliary {
        return Phase0Plan {
            diagnostics: diag,
            frame: None,
        };
    }
    if matches!(opts.auxiliary_coupling, AuxiliaryCouplingPolicy::None) {
        // Diagnostics-only mode: run nothing.
        return Phase0Plan {
            diagnostics: diag,
            frame: None,
        };
    }

    let start = std::time::Instant::now();

    // -- 1. Build the structural graphs ------------------------------
    let pv = ProbeView {
        n_vars: probe.n_vars,
        m_rows: probe.n_rows,
        jac_irow: probe.jac_irow,
        jac_jcol: probe.jac_jcol,
        jac_values: Some(probe.jac_values),
        g_l: probe.g_l,
        g_u: probe.g_u,
        linearity: Some(probe.linearity),
        one_based: probe.one_based,
        eq_tol: probe.eq_tol,
    };
    let t_inc = std::time::Instant::now();
    let eq_inc = EqualityIncidence::from_probe(&pv);
    let ineq_inc = InequalityIncidence::from_probe(&pv);
    diag.stage_time_ms.incidence_ms = t_inc.elapsed().as_millis();

    let t_match = std::time::Instant::now();
    let matching = hopcroft_karp(&eq_inc);
    diag.stage_time_ms.matching_ms = t_match.elapsed().as_millis();

    let t_dm = std::time::Instant::now();
    let dm = DulmageMendelsohnPartition::from_matching(&eq_inc, &matching);
    diag.stage_time_ms.dm_ms = t_dm.elapsed().as_millis();

    let t_comp = std::time::Instant::now();
    let comps = SquareComponents::of_square_part(&eq_inc, &matching, &dm);
    diag.stage_time_ms.components_ms = t_comp.elapsed().as_millis();

    let obj_support = objective_gradient_support(probe.grad_f, 1e-12);

    // -- 2. Decide which dropped rows are linear --------------------
    // We need a fast lookup from inner row index → linearity. The
    // EqualityIncidence carries `eq_row_inner_idx[k]` but the dropped
    // row checks happen by inner index, so map directly.
    let is_linear_inner: Vec<bool> = probe
        .linearity
        .iter()
        .map(|l| matches!(l, Linearity::Linear))
        .collect();

    // -- 3. Walk components / BTF, classify, try to solve -----------
    let aggressive = matches!(opts.auxiliary_coupling, AuxiliaryCouplingPolicy::Aggressive);
    let mut accepted_fixed_vars: Vec<usize> = Vec::new();
    let mut accepted_fixed_values: Vec<Number> = Vec::new();
    let mut accepted_dropped_rows: Vec<usize> = Vec::new();
    let mut max_residual: Number = 0.0;
    let mut x_running: Vec<Number> = probe.x_probe.to_vec();

    for comp in &comps.components {
        let t_btf = std::time::Instant::now();
        let btf = BlockTriangularForm::of_component(&eq_inc, &matching, comp);
        diag.stage_time_ms.btf_ms += t_btf.elapsed().as_millis();
        for block in &btf.blocks {
            diag.candidate_blocks += 1;
            // -- 3a. Linearity gate --
            let all_linear = block
                .eq_rows
                .iter()
                .map(|&k| eq_inc.eq_row_inner_idx[k])
                .all(|r| is_linear_inner[r]);
            if !all_linear {
                diag.rejection_reasons
                    .push(AuxiliaryRejectionReason::BlockSolveDiverged);
                continue;
            }

            // -- 3b. Size gate --
            if block.eq_rows.len() > opts.auxiliary_max_block_dim as usize {
                diag.rejection_reasons
                    .push(AuxiliaryRejectionReason::BlockTooLarge);
                continue;
            }

            // -- 3c. Coupling gate --
            let class = classify_block(block, &ineq_inc, &obj_support);
            match class {
                AuxiliaryCouplingClass::PureEquality => diag.class_counts.pure_equality += 1,
                AuxiliaryCouplingClass::ObjectiveCoupled => {
                    diag.class_counts.objective_coupled += 1
                }
                AuxiliaryCouplingClass::InequalityCoupled => {
                    diag.class_counts.inequality_coupled += 1
                }
                AuxiliaryCouplingClass::ObjectiveAndInequalityCoupled => {
                    diag.class_counts.objective_and_inequality_coupled += 1
                }
            }
            let allowed = match class {
                AuxiliaryCouplingClass::PureEquality => true,
                AuxiliaryCouplingClass::ObjectiveCoupled => aggressive,
                _ => false,
            };
            if !allowed {
                diag.rejection_reasons
                    .push(AuxiliaryRejectionReason::CouplingDisallowed);
                continue;
            }

            // -- 3d. Build the linear block system --
            // Each dropped row r (which is linear) reads:
            //   g_r(x) = Σ_j J[r][j] x[j] + const_r
            // where const_r = g_r(x_probe) - Σ_j J[r][j] x_probe[j].
            // Setting g_r(x) = g_l[r] (= g_u[r] for an equality row):
            //   Σ_{j fixed} J[r][j] x_fixed[j] = g_l[r] - const_r
            //                                  - Σ_{j non-fixed} J[r][j] x_running[j]
            let k = block.eq_rows.len();
            let inner_rows: Vec<usize> = block
                .eq_rows
                .iter()
                .map(|&kk| eq_inc.eq_row_inner_idx[kk])
                .collect();
            let block_cols = &block.cols;

            let mut a_block = vec![0.0; k * k];
            let mut b_block = vec![0.0; k];

            for (ii, &r_inner) in inner_rows.iter().enumerate() {
                // J[r][·] from the pre-fetched Jacobian: walk the
                // triplets and pull entries whose row matches r_inner.
                // (This is O(nnz) per block; for tiny blocks fine.)
                // Also accumulate Σ_j J[r][j] x_probe[j] for the
                // constant computation.
                let mut sum_jx = 0.0;
                let nnz = probe.jac_irow.len();
                for kk in 0..nnz {
                    let i = if probe.one_based {
                        (probe.jac_irow[kk] as isize - 1) as usize
                    } else {
                        probe.jac_irow[kk] as usize
                    };
                    if i != r_inner {
                        continue;
                    }
                    let j = if probe.one_based {
                        (probe.jac_jcol[kk] as isize - 1) as usize
                    } else {
                        probe.jac_jcol[kk] as usize
                    };
                    let v = probe.jac_values[kk];
                    sum_jx += v * probe.x_probe[j];
                    if let Some(jj) = block_cols.iter().position(|&c| c == j) {
                        a_block[ii * k + jj] = v;
                    } else {
                        // Non-block variable, value comes from x_running.
                        // Subtract from RHS.
                        b_block[ii] -= v * x_running[j];
                    }
                }
                let const_r = probe.g_at_probe[r_inner] - sum_jx;
                let g_target = probe.g_l[r_inner];
                b_block[ii] += g_target - const_r;
            }

            // -- 3e. Solve via the lightweight Newton wrapper --
            struct LinearBlock {
                a: Vec<Number>,
                b: Vec<Number>,
                n: usize,
            }
            impl BlockEquations for LinearBlock {
                fn dim(&self) -> usize {
                    self.n
                }
                fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
                    for i in 0..self.n {
                        let mut s = -self.b[i];
                        for j in 0..self.n {
                            s += self.a[i * self.n + j] * x[j];
                        }
                        f[i] = s;
                    }
                    true
                }
                fn jacobian(&mut self, _x: &[Number], j: &mut [Number]) -> bool {
                    j.copy_from_slice(&self.a);
                    true
                }
            }
            let mut eqs = LinearBlock {
                a: a_block,
                b: b_block,
                n: k,
            };
            let bs_opts = BlockSolveOptions {
                tol: opts.auxiliary_tol,
                max_dim: opts.auxiliary_max_block_dim as usize,
                ..Default::default()
            };
            let x0 = vec![0.0; k];
            let t_solve = std::time::Instant::now();
            let solve_result = DampedNewtonSolver.solve(&x0, &mut eqs, &bs_opts);
            diag.stage_time_ms.block_solve_ms += t_solve.elapsed().as_millis();
            let out = match solve_result {
                Ok(o) => o,
                Err(crate::block_solve::BlockSolveError::Singular) => {
                    diag.rejection_reasons
                        .push(AuxiliaryRejectionReason::BlockSolveDiverged);
                    continue;
                }
                Err(_) => {
                    diag.rejection_reasons
                        .push(AuxiliaryRejectionReason::BlockSolveDiverged);
                    continue;
                }
            };

            // -- 3f. Full-space residual check at the candidate x --
            let t_resid = std::time::Instant::now();
            let mut candidate_x = x_running.clone();
            for (ii, &c) in block_cols.iter().enumerate() {
                candidate_x[c] = out.x[ii];
            }
            // Re-evaluate each dropped row's residual using the
            // linear model g_r(x) = J[r][:] · x + const_r and check
            // |g_r(x) - g_l[r]| < tol.
            let mut row_resid: Number = 0.0;
            for &r_inner in &inner_rows {
                let mut s = 0.0;
                let nnz = probe.jac_irow.len();
                let mut sum_jx = 0.0;
                for kk in 0..nnz {
                    let i = if probe.one_based {
                        (probe.jac_irow[kk] as isize - 1) as usize
                    } else {
                        probe.jac_irow[kk] as usize
                    };
                    if i != r_inner {
                        continue;
                    }
                    let j = if probe.one_based {
                        (probe.jac_jcol[kk] as isize - 1) as usize
                    } else {
                        probe.jac_jcol[kk] as usize
                    };
                    let v = probe.jac_values[kk];
                    s += v * candidate_x[j];
                    sum_jx += v * probe.x_probe[j];
                }
                let const_r = probe.g_at_probe[r_inner] - sum_jx;
                let residual = (s + const_r - probe.g_l[r_inner]).abs();
                row_resid = row_resid.max(residual);
            }
            diag.stage_time_ms.residual_check_ms += t_resid.elapsed().as_millis();
            if row_resid > opts.auxiliary_tol {
                diag.rejection_reasons
                    .push(AuxiliaryRejectionReason::ResidualCheckFailed);
                continue;
            }
            max_residual = max_residual.max(row_resid);

            // -- 3g. Accept --
            for (ii, &c) in block_cols.iter().enumerate() {
                accepted_fixed_vars.push(c);
                accepted_fixed_values.push(out.x[ii]);
                x_running[c] = out.x[ii];
            }
            for &r_inner in &inner_rows {
                accepted_dropped_rows.push(r_inner);
            }
            diag.blocks_eliminated += 1;
            if (k as Index) > diag.max_accepted_block_dim {
                diag.max_accepted_block_dim = k as Index;
            }
        }
    }

    diag.vars_eliminated = accepted_fixed_vars.len() as Index;
    diag.rows_eliminated = accepted_dropped_rows.len() as Index;
    diag.max_block_residual = max_residual;
    diag.total_time_ms = start.elapsed().as_millis();

    // -- 4. Build one composite frame --
    let frame = if accepted_fixed_vars.is_empty() {
        None
    } else {
        // Sort by variable index for the frame's contract.
        let mut order: Vec<usize> = (0..accepted_fixed_vars.len()).collect();
        order.sort_by_key(|&i| accepted_fixed_vars[i]);
        let fixed_vars_sorted: Vec<usize> = order.iter().map(|&i| accepted_fixed_vars[i]).collect();
        let fixed_values_sorted: Vec<Number> =
            order.iter().map(|&i| accepted_fixed_values[i]).collect();
        let mut dropped_rows_sorted = accepted_dropped_rows.clone();
        dropped_rows_sorted.sort_unstable();
        Some(ReductionFrame::new(
            probe.n_vars,
            probe.n_rows,
            fixed_vars_sorted,
            fixed_values_sorted,
            dropped_rows_sorted,
        ))
    };

    Phase0Plan {
        diagnostics: diag,
        frame,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_probe<'a>(
        n_vars: usize,
        n_rows: usize,
        jac_irow: &'a [Index],
        jac_jcol: &'a [Index],
        jac_values: &'a [Number],
        g_l: &'a [Number],
        g_u: &'a [Number],
        g_at_probe: &'a [Number],
        linearity: &'a [Linearity],
        x_probe: &'a [Number],
        grad_f: &'a [Number],
    ) -> Phase0Probe<'a> {
        Phase0Probe {
            n_vars,
            n_rows,
            jac_irow,
            jac_jcol,
            jac_values,
            g_l,
            g_u,
            g_at_probe,
            linearity,
            one_based: false,
            eq_tol: 1e-12,
            x_probe,
            grad_f,
        }
    }

    #[test]
    fn noop_when_disabled() {
        let probe = linear_probe(
            2,
            1,
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            &[1.0],
            &[1.0],
            &[0.0],
            &[Linearity::Linear],
            &[0.0, 0.0],
            &[0.0, 0.0],
        );
        let opts = PresolveOptions::defaults();
        let plan = run_auxiliary_phase0(&opts, &probe);
        assert_eq!(plan.diagnostics.blocks_eliminated, 0);
        assert!(plan.frame.is_none());
    }

    #[test]
    fn phase0_eliminates_linear_singleton() {
        // 1 var, 1 row, equality: x = 3.
        // x_probe = [0]. g(x_probe) = 0, so const = 0 - 1*0 = 0.
        // Linear model: g(x) = x → x_target = 3.
        // grad_f is zero at this variable → PureEquality, eligible
        // under the default `Safe` policy.
        let probe = linear_probe(
            1,
            1,
            &[0],
            &[0],
            &[1.0],
            &[3.0],
            &[3.0],
            &[0.0],
            &[Linearity::Linear],
            &[0.0],
            &[0.0],
        );
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let plan = run_auxiliary_phase0(&opts, &probe);
        assert_eq!(plan.diagnostics.blocks_eliminated, 1);
        assert_eq!(plan.diagnostics.vars_eliminated, 1);
        assert_eq!(plan.diagnostics.rows_eliminated, 1);
        let frame = plan.frame.expect("accepted frame");
        assert_eq!(frame.fixed_vars, vec![0]);
        assert!((frame.fixed_values[0] - 3.0).abs() < 1e-12);
        assert_eq!(frame.dropped_rows, vec![0]);
    }

    #[test]
    fn phase0_eliminates_linear_2x2_block() {
        // 2 vars, 2 rows:
        //   c0: x + y = 3
        //   c1: x - y = 1
        // → (x, y) = (2, 1).
        let probe = linear_probe(
            2,
            2,
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, -1.0],
            &[3.0, 1.0],
            &[3.0, 1.0],
            &[0.0, 0.0],
            &[Linearity::Linear, Linearity::Linear],
            &[0.0, 0.0],
            &[0.0, 0.0],
        );
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let plan = run_auxiliary_phase0(&opts, &probe);
        assert_eq!(plan.diagnostics.blocks_eliminated, 1);
        let frame = plan.frame.expect("accepted");
        assert_eq!(frame.fixed_vars, vec![0, 1]);
        assert!((frame.fixed_values[0] - 2.0).abs() < 1e-12);
        assert!((frame.fixed_values[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn phase0_rejects_inequality_coupled() {
        // 2 vars, 2 rows. Row 0 equality, row 1 inequality touching
        // var 1. Block on var 1 is InequalityCoupled → rejected.
        let probe = linear_probe(
            2,
            2,
            &[0, 1],
            &[1, 1],
            &[1.0, 1.0],
            &[5.0, -1e19],
            &[5.0, 10.0], // row 1 is inequality
            &[0.0, 0.0],
            &[Linearity::Linear, Linearity::Linear],
            &[0.0, 0.0],
            &[0.0, 0.0],
        );
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        opts.auxiliary_coupling = AuxiliaryCouplingPolicy::Safe;
        let plan = run_auxiliary_phase0(&opts, &probe);
        assert_eq!(plan.diagnostics.blocks_eliminated, 0);
        assert!(plan.frame.is_none());
        assert!(plan
            .diagnostics
            .rejection_reasons
            .iter()
            .any(|r| matches!(r, AuxiliaryRejectionReason::CouplingDisallowed)));
    }

    #[test]
    fn phase0_rejects_nonlinear_row() {
        // 1 var, 1 row marked as nonlinear → rejected (PR 8 is
        // linear-only; nonlinear lands in PR 8b).
        let probe = linear_probe(
            1,
            1,
            &[0],
            &[0],
            &[1.0],
            &[3.0],
            &[3.0],
            &[0.0],
            &[Linearity::NonLinear],
            &[0.0],
            &[0.0],
        );
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let plan = run_auxiliary_phase0(&opts, &probe);
        assert_eq!(plan.diagnostics.blocks_eliminated, 0);
        assert!(plan.frame.is_none());
    }

    /// Smoke test: when `auxiliary_coupling = None` (diagnostics
    /// only), the orchestrator never produces a frame even if every
    /// other gate would pass.
    #[test]
    fn phase0_none_policy_is_diagnostics_only() {
        let probe = linear_probe(
            1,
            1,
            &[0],
            &[0],
            &[1.0],
            &[3.0],
            &[3.0],
            &[0.0],
            &[Linearity::Linear],
            &[0.0],
            &[0.0],
        );
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        opts.auxiliary_coupling = AuxiliaryCouplingPolicy::None;
        let plan = run_auxiliary_phase0(&opts, &probe);
        assert!(plan.frame.is_none());
        assert_eq!(plan.diagnostics.blocks_eliminated, 0);
    }
}
