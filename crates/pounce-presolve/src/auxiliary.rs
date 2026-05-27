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

/// Callback the orchestrator uses for nonlinear block solves.
/// Implemented in `PresolveTnlp::ensure_init` by wrapping the inner
/// TNLP. Linear blocks don't need this (they use the pre-fetched
/// Jacobian); calling `run_auxiliary_phase0(_, _, None)` falls back
/// to linear-only and is the cheaper path when no nonlinear rows
/// participate.
pub trait Phase0TnlpCallback {
    /// Evaluate the full constraint vector `g(x)` (length `n_rows`).
    fn eval_g_full(&mut self, x: &[Number], g: &mut [Number]) -> bool;
    /// Evaluate the full Jacobian values at `x` into the user buffer
    /// (length `nnz`), matching the sparsity pattern from the probe.
    fn eval_jac_g_values(&mut self, x: &[Number], values: &mut [Number]) -> bool;
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
/// When `opts.auxiliary` is `false` this is a true no-op. When
/// `true`, the orchestrator walks every candidate block, accepts
/// only those allowed by `opts.auxiliary_coupling`, and accumulates
/// a single composite `ReductionFrame`.
///
/// `tnlp` is consulted for nonlinear blocks. Pass `None` for
/// linear-only behaviour (PR 8 default); pass `Some` to also handle
/// blocks where any dropped row has `Linearity::NonLinear`.
pub fn run_auxiliary_phase0(
    opts: &PresolveOptions,
    probe: &Phase0Probe<'_>,
    mut tnlp: Option<&mut dyn Phase0TnlpCallback>,
    mut large_solver: Option<&mut dyn crate::block_solve::LargeBlockSolver>,
) -> Phase0Plan {
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
    let has_large_solver = large_solver.is_some();
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
            // -- 3a. Linearity classification --
            // A block where every dropped row is linear uses the fast
            // path (pre-fetched Jacobian, one-iteration Newton).
            // Otherwise we need TNLP callbacks; reject if none was
            // supplied.
            let all_linear = block
                .eq_rows
                .iter()
                .map(|&k| eq_inc.eq_row_inner_idx[k])
                .all(|r| is_linear_inner[r]);
            if !all_linear && tnlp.is_none() {
                diag.rejection_reasons
                    .push(AuxiliaryRejectionReason::BlockSolveDiverged);
                continue;
            }

            // -- 3b. Size gate --
            // Blocks within `auxiliary_max_block_dim` use the
            // lightweight Newton. Blocks larger than that need a
            // `LargeBlockSolver`; if none was supplied, reject as
            // `BlockTooLarge` (PR 10 behaviour).
            let is_large = block.eq_rows.len() > opts.auxiliary_max_block_dim as usize;
            if is_large && !has_large_solver {
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

            // -- 3d/3e. Build + solve the block --
            let k = block.eq_rows.len();
            let inner_rows: Vec<usize> = block
                .eq_rows
                .iter()
                .map(|&kk| eq_inc.eq_row_inner_idx[kk])
                .collect();
            let block_cols = &block.cols;
            let bs_opts = BlockSolveOptions {
                tol: opts.auxiliary_tol,
                max_dim: opts.auxiliary_max_block_dim as usize,
                ..Default::default()
            };
            let t_solve = std::time::Instant::now();
            // Closure-based solver dispatch: each branch picks the
            // right BlockSolver/LargeBlockSolver at the call site,
            // so the helpers don't have to plumb the option through.
            let solver_call = |x0: &[Number],
                               eqs: &mut dyn BlockEquations,
                               opts: &BlockSolveOptions|
             -> Result<
                crate::block_solve::BlockSolveOutcome,
                crate::block_solve::BlockSolveError,
            > {
                if is_large {
                    large_solver
                        .as_deref_mut()
                        .expect("checked above")
                        .solve_large(x0, eqs, opts)
                } else {
                    DampedNewtonSolver.solve(x0, eqs, opts)
                }
            };
            let solve_result = if all_linear {
                solve_linear_block(
                    probe,
                    &inner_rows,
                    block_cols,
                    &x_running,
                    &bs_opts,
                    solver_call,
                )
            } else {
                // SAFETY of `.as_mut().unwrap()`: the linearity gate
                // above already rejects nonlinear blocks when
                // `tnlp.is_none()`, so reaching here means it's Some.
                let cb: &mut dyn Phase0TnlpCallback = *tnlp.as_mut().expect("checked above");
                solve_nonlinear_block(
                    probe,
                    &inner_rows,
                    block_cols,
                    &x_running,
                    &bs_opts,
                    cb,
                    solver_call,
                )
            };
            diag.stage_time_ms.block_solve_ms += t_solve.elapsed().as_millis();
            let out = match solve_result {
                Ok(o) => o,
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
            let row_resid = if all_linear {
                // Re-evaluate using the linear model — same as the
                // build code above.
                residual_norm_linear(probe, &inner_rows, &candidate_x)
            } else {
                // Ask the TNLP for `g(candidate_x)` and compare each
                // dropped row to `g_l`. Reuses the callback we
                // already required above.
                let cb: &mut dyn Phase0TnlpCallback = *tnlp.as_mut().expect("checked above");
                match residual_norm_nonlinear(probe, &inner_rows, &candidate_x, cb) {
                    Some(r) => r,
                    None => {
                        diag.rejection_reasons
                            .push(AuxiliaryRejectionReason::BlockSolveDiverged);
                        continue;
                    }
                }
            };
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

// -- Helpers used by both the linear and nonlinear solve paths ------

/// Solve a linear block from the pre-fetched Jacobian. Builds the
/// k×k system from the probe data and dispatches to Newton (which
/// converges in one iteration on linear systems).
fn solve_linear_block<F>(
    probe: &Phase0Probe<'_>,
    inner_rows: &[usize],
    block_cols: &[usize],
    x_running: &[Number],
    bs_opts: &BlockSolveOptions,
    solver_call: F,
) -> Result<crate::block_solve::BlockSolveOutcome, crate::block_solve::BlockSolveError>
where
    F: FnOnce(
        &[Number],
        &mut dyn BlockEquations,
        &BlockSolveOptions,
    )
        -> Result<crate::block_solve::BlockSolveOutcome, crate::block_solve::BlockSolveError>,
{
    let k = inner_rows.len();
    let mut a_block = vec![0.0; k * k];
    let mut b_block = vec![0.0; k];
    for (ii, &r_inner) in inner_rows.iter().enumerate() {
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
                b_block[ii] -= v * x_running[j];
            }
        }
        let const_r = probe.g_at_probe[r_inner] - sum_jx;
        b_block[ii] += probe.g_l[r_inner] - const_r;
    }

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
    let x0 = vec![0.0; k];
    solver_call(&x0, &mut eqs, bs_opts)
}

/// Solve a nonlinear block by feeding TNLP callbacks into Newton.
fn solve_nonlinear_block<F>(
    probe: &Phase0Probe<'_>,
    inner_rows: &[usize],
    block_cols: &[usize],
    x_running: &[Number],
    bs_opts: &BlockSolveOptions,
    tnlp: &mut dyn Phase0TnlpCallback,
    solver_call: F,
) -> Result<crate::block_solve::BlockSolveOutcome, crate::block_solve::BlockSolveError>
where
    F: FnOnce(
        &[Number],
        &mut dyn BlockEquations,
        &BlockSolveOptions,
    )
        -> Result<crate::block_solve::BlockSolveOutcome, crate::block_solve::BlockSolveError>,
{
    let k = inner_rows.len();

    struct NonlinearBlock<'a> {
        n: usize,
        nnz: usize,
        inner_rows: &'a [usize],
        block_cols: &'a [usize],
        x_running: &'a [Number],
        jac_irow: &'a [Index],
        jac_jcol: &'a [Index],
        g_l: &'a [Number],
        one_based: bool,
        tnlp: &'a mut dyn Phase0TnlpCallback,
        // Scratch buffers reused across iterations.
        full_x: Vec<Number>,
        full_g: Vec<Number>,
        full_jac_vals: Vec<Number>,
    }
    impl<'a> NonlinearBlock<'a> {
        fn splice_full_x(&mut self, x_block: &[Number]) {
            self.full_x.copy_from_slice(self.x_running);
            for (ii, &c) in self.block_cols.iter().enumerate() {
                self.full_x[c] = x_block[ii];
            }
        }
    }
    impl<'a> BlockEquations for NonlinearBlock<'a> {
        fn dim(&self) -> usize {
            self.n
        }
        fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
            self.splice_full_x(x);
            if !self.tnlp.eval_g_full(&self.full_x, &mut self.full_g) {
                return false;
            }
            for (ii, &r) in self.inner_rows.iter().enumerate() {
                f[ii] = self.full_g[r] - self.g_l[r];
            }
            true
        }
        fn jacobian(&mut self, x: &[Number], j: &mut [Number]) -> bool {
            self.splice_full_x(x);
            if !self
                .tnlp
                .eval_jac_g_values(&self.full_x, &mut self.full_jac_vals)
            {
                return false;
            }
            // Zero the dense submatrix, then scatter the nonzeros.
            for v in j.iter_mut() {
                *v = 0.0;
            }
            for kk in 0..self.nnz {
                let i = if self.one_based {
                    (self.jac_irow[kk] as isize - 1) as usize
                } else {
                    self.jac_irow[kk] as usize
                };
                let Some(ii) = self.inner_rows.iter().position(|&r| r == i) else {
                    continue;
                };
                let col = if self.one_based {
                    (self.jac_jcol[kk] as isize - 1) as usize
                } else {
                    self.jac_jcol[kk] as usize
                };
                let Some(jj) = self.block_cols.iter().position(|&c| c == col) else {
                    continue;
                };
                j[ii * self.n + jj] = self.full_jac_vals[kk];
            }
            true
        }
    }

    let nnz = probe.jac_irow.len();
    let mut eqs = NonlinearBlock {
        n: k,
        nnz,
        inner_rows,
        block_cols,
        x_running,
        jac_irow: probe.jac_irow,
        jac_jcol: probe.jac_jcol,
        g_l: probe.g_l,
        one_based: probe.one_based,
        tnlp,
        full_x: vec![0.0; probe.n_vars],
        full_g: vec![0.0; probe.n_rows],
        full_jac_vals: vec![0.0; nnz],
    };
    // Start at the probe's value for each block variable.
    let x0: Vec<Number> = block_cols.iter().map(|&c| probe.x_probe[c]).collect();
    solver_call(&x0, &mut eqs, bs_opts)
}

/// Linear-model residual check: each dropped row should evaluate to
/// `g_l[r]` after splicing the candidate point in.
fn residual_norm_linear(
    probe: &Phase0Probe<'_>,
    inner_rows: &[usize],
    candidate_x: &[Number],
) -> Number {
    let mut row_resid: Number = 0.0;
    let nnz = probe.jac_irow.len();
    for &r_inner in inner_rows {
        let mut s = 0.0;
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
        row_resid = row_resid.max((s + const_r - probe.g_l[r_inner]).abs());
    }
    row_resid
}

/// TNLP-backed residual check: ask the inner TNLP for `g` at the
/// candidate and compare each dropped row to `g_l[r]`. Returns
/// `None` if the TNLP callback fails.
fn residual_norm_nonlinear(
    probe: &Phase0Probe<'_>,
    inner_rows: &[usize],
    candidate_x: &[Number],
    tnlp: &mut dyn Phase0TnlpCallback,
) -> Option<Number> {
    let mut g_full = vec![0.0; probe.n_rows];
    if !tnlp.eval_g_full(candidate_x, &mut g_full) {
        return None;
    }
    let mut row_resid: Number = 0.0;
    for &r in inner_rows {
        row_resid = row_resid.max((g_full[r] - probe.g_l[r]).abs());
    }
    Some(row_resid)
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
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
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
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
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
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
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
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
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
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
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
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
        assert!(plan.frame.is_none());
        assert_eq!(plan.diagnostics.blocks_eliminated, 0);
    }

    /// Stub callback implementing `xy = 1, x + y = 3` for the
    /// nonlinear-block test. The probe carries the static Jacobian
    /// sparsity (every entry present) and the equality targets; the
    /// callback returns `g(x)` and `J(x)`.
    struct XyCallback;
    impl Phase0TnlpCallback for XyCallback {
        fn eval_g_full(&mut self, x: &[Number], g: &mut [Number]) -> bool {
            g[0] = x[0] * x[1];
            g[1] = x[0] + x[1];
            true
        }
        fn eval_jac_g_values(&mut self, x: &[Number], values: &mut [Number]) -> bool {
            // Sparsity layout (matches the probe below):
            //   (0,0) = ∂(xy)/∂x = y
            //   (0,1) = ∂(xy)/∂y = x
            //   (1,0) = ∂(x+y)/∂x = 1
            //   (1,1) = ∂(x+y)/∂y = 1
            values[0] = x[1];
            values[1] = x[0];
            values[2] = 1.0;
            values[3] = 1.0;
            true
        }
    }

    #[test]
    fn phase0_eliminates_nonlinear_block() {
        // 2 vars, 2 rows.  Row 0: x*y = 1  (nonlinear).  Row 1: x+y = 3.
        // x_probe = (3.0, 0.5): off-symmetric so the Jacobian
        // [[y, x], [1, 1]] = [[0.5, 3.0], [1, 1]] is non-singular.
        // Newton converges to (1, 2) (or 2, 1).
        let probe = linear_probe(
            2,
            2,
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[0.5, 3.0, 1.0, 1.0], // values at x_probe = (3.0, 0.5)
            &[1.0, 3.0],
            &[1.0, 3.0],
            &[1.5, 3.5], // g(3.0, 0.5) = [3.0*0.5, 3.0+0.5]
            &[Linearity::NonLinear, Linearity::Linear],
            &[3.0, 0.5],
            &[0.0, 0.0],
        );
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let mut tnlp = XyCallback;
        let plan = run_auxiliary_phase0(&opts, &probe, Some(&mut tnlp), None);
        assert_eq!(plan.diagnostics.blocks_eliminated, 1);
        let frame = plan.frame.expect("accepted");
        // Newton from (1.5, 1.5) on this system converges to (1, 2)
        // or (2, 1) depending on column ordering; either is a root.
        let v0 = frame.fixed_values[0];
        let v1 = frame.fixed_values[1];
        assert!(
            (v0 * v1 - 1.0).abs() < 1e-8,
            "x*y should be 1, got {v0}*{v1}={}",
            v0 * v1
        );
        assert!(
            (v0 + v1 - 3.0).abs() < 1e-8,
            "x+y should be 3, got {}",
            v0 + v1
        );
    }

    #[test]
    fn phase0_nonlinear_rejected_without_callback() {
        // Same probe as above but with `tnlp = None`.
        let probe = linear_probe(
            2,
            2,
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[0.5, 3.0, 1.0, 1.0],
            &[1.0, 3.0],
            &[1.0, 3.0],
            &[1.5, 3.5],
            &[Linearity::NonLinear, Linearity::Linear],
            &[3.0, 0.5],
            &[0.0, 0.0],
        );
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
        assert_eq!(plan.diagnostics.blocks_eliminated, 0);
        assert!(plan.frame.is_none());
        assert!(plan
            .diagnostics
            .rejection_reasons
            .iter()
            .any(|r| matches!(r, AuxiliaryRejectionReason::BlockSolveDiverged)));
    }

    /// Build a synthetic probe for an `n`-variable diagonal linear
    /// equality system: `2 x[i] = i + 1`, all linear, all
    /// PureEquality. Used to exercise the large-block path with a
    /// trivially-solvable system.
    fn diagonal_linear_probe(n: usize) -> Phase0Probe<'static> {
        // Leak the arrays so the probe can carry &'static slices
        // (tests-only).
        let irow: Vec<Index> = (0..n).map(|i| i as Index).collect();
        let jcol: Vec<Index> = (0..n).map(|i| i as Index).collect();
        let vals = vec![2.0; n];
        let g_l: Vec<Number> = (1..=n).map(|i| i as Number).collect();
        let g_u = g_l.clone();
        let g_probe: Vec<Number> = vec![0.0; n]; // g(0) = 0
        let linearity = vec![Linearity::Linear; n];
        let x_probe = vec![0.0; n];
        let grad_f = vec![0.0; n];
        Phase0Probe {
            n_vars: n,
            n_rows: n,
            jac_irow: Box::leak(irow.into_boxed_slice()),
            jac_jcol: Box::leak(jcol.into_boxed_slice()),
            jac_values: Box::leak(vals.into_boxed_slice()),
            g_l: Box::leak(g_l.into_boxed_slice()),
            g_u: Box::leak(g_u.into_boxed_slice()),
            g_at_probe: Box::leak(g_probe.into_boxed_slice()),
            linearity: Box::leak(linearity.into_boxed_slice()),
            one_based: false,
            eq_tol: 1e-12,
            x_probe: Box::leak(x_probe.into_boxed_slice()),
            grad_f: Box::leak(grad_f.into_boxed_slice()),
        }
    }

    #[test]
    fn phase0_large_block_rejected_without_solver() {
        // 10-row block exceeds the default max_block_dim=8 →
        // BlockTooLarge when no LargeBlockSolver is supplied.
        let probe = diagonal_linear_probe(10);
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
        // The components decompose this into 10 singleton blocks
        // (every row touches a distinct column). Each singleton is
        // dim 1, well under the threshold → all should be eliminated.
        // So this test doesn't exercise the large-block path with
        // the diagonal probe.
        //
        // Adjust: 10 singletons all get eliminated, not rejected.
        assert_eq!(plan.diagnostics.blocks_eliminated, 10);
    }

    #[test]
    fn phase0_large_block_uses_fallback() {
        // Make a single 10×10 dense block by having every row touch
        // every column. Then it's one connected component, one BTF
        // block of size 10. Default Newton rejects → fallback solves.
        let n = 10usize;
        // jac_irow / jac_jcol: every (i, j) entry. Values placed so
        // the matrix is diagonally dominant.
        let mut irow = Vec::new();
        let mut jcol = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for j in 0..n {
                irow.push(i as Index);
                jcol.push(j as Index);
                vals.push(if i == j { 5.0 } else { 0.1 });
            }
        }
        let g_l: Vec<Number> = (1..=n).map(|i| i as Number).collect();
        let g_u = g_l.clone();
        let g_probe = vec![0.0; n];
        let linearity = vec![Linearity::Linear; n];
        let x_probe = vec![0.0; n];
        let grad_f = vec![0.0; n];
        let probe = Phase0Probe {
            n_vars: n,
            n_rows: n,
            jac_irow: &irow,
            jac_jcol: &jcol,
            jac_values: &vals,
            g_l: &g_l,
            g_u: &g_u,
            g_at_probe: &g_probe,
            linearity: &linearity,
            one_based: false,
            eq_tol: 1e-12,
            x_probe: &x_probe,
            grad_f: &grad_f,
        };

        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;

        // Without fallback → 1 candidate, rejected as TooLarge.
        let plan_no_fb = run_auxiliary_phase0(&opts, &probe, None, None);
        assert_eq!(plan_no_fb.diagnostics.blocks_eliminated, 0);
        assert!(plan_no_fb
            .diagnostics
            .rejection_reasons
            .iter()
            .any(|r| matches!(r, AuxiliaryRejectionReason::BlockTooLarge)));

        // With fallback → eliminated.
        let mut fb = crate::block_solve::RelaxedNewtonSolver;
        let plan_with_fb = run_auxiliary_phase0(&opts, &probe, None, Some(&mut fb));
        assert_eq!(plan_with_fb.diagnostics.blocks_eliminated, 1);
        assert_eq!(plan_with_fb.diagnostics.vars_eliminated, n as Index);
        let frame = plan_with_fb.frame.expect("accepted");
        // Quick sanity: each x[i] should satisfy `5*x[i] + 0.1 *
        // sum_{j!=i} x[j] = i+1`. With dominance, x ≈ b/5 ≈ 0.2*(i+1).
        for (k, &i) in frame.fixed_vars.iter().enumerate() {
            let expected_approx = (i + 1) as Number / 5.0;
            assert!((frame.fixed_values[k] - expected_approx).abs() < 0.2);
        }
    }
}
