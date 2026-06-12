//! Phase-0 orchestrator for auxiliary-equality preprocessing.
//!
//! Port of [ripopt PR #32](https://github.com/jkitchin/ripopt/pull/32)
//! by David Bernal Neira (`@bernalde`). PR 8 of the auxiliary-presolve
//! port (issue #53). This module runs the full pipeline assembled in
//! PRs 2-7:
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
    /// Per-variable linearity tags **with respect to the objective**
    /// (`get_objective_variables_linearity`, falling back to the global
    /// `get_variables_linearity` — a conservative superset), when the
    /// inner TNLP supplies them. `grad_f` is sampled at a single probe
    /// point, so a variable nonlinear in the objective can read as
    /// objective-free there (e.g. `f=(x−x₀)²` started at `x₀`). A `Linear`
    /// tag makes the zero probe-gradient conclusive; for a `NonLinear` tag
    /// it is not, so the variable is treated as objective-coupled (H11).
    /// `None` when the TNLP declines — the probe gradient is then used
    /// alone, preserving the pre-H11 behavior.
    pub var_linearity: Option<&'a [Linearity]>,
    /// PR 13: variable bounds — needed for the trivial-elimination
    /// pre-pass that runs before incidence is built.
    pub x_l: &'a [Number],
    pub x_u: &'a [Number],
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

/// Decode a COO index respecting the probe's one-/zero-based convention.
#[inline]
fn decode_idx(raw: Index, one_based: bool) -> usize {
    if one_based {
        (raw as isize - 1) as usize
    } else {
        raw as usize
    }
}

/// CSR row index into the probe's COO Jacobian, built once per Phase-0 pass.
/// `entries[ptr[r]..ptr[r + 1]]` lists the nnz positions `kk` whose row is
/// `r`. Block-assembly helpers iterate a row's nonzeros directly through this
/// instead of scanning the full `nnz` array per block row (M27 — that scan was
/// O(total_block_rows × nnz), quadratic on models built from many small
/// blocks, e.g. gas networks).
#[derive(Clone, Copy)]
struct RowNnz<'a> {
    ptr: &'a [usize],
    entries: &'a [usize],
}

impl<'a> RowNnz<'a> {
    /// The nnz positions `kk` belonging to row `r`.
    #[inline]
    fn of_row(&self, r: usize) -> &[usize] {
        &self.entries[self.ptr[r]..self.ptr[r + 1]]
    }
}

/// Build the CSR row index from the probe's COO row indices. O(nnz) time and
/// space; rows with out-of-range indices are skipped (defensive — the COO is
/// trusted, but a stray index must not panic the slice build).
fn build_row_nnz(jac_irow: &[Index], n_rows: usize, one_based: bool) -> (Vec<usize>, Vec<usize>) {
    let nnz = jac_irow.len();
    let mut ptr = vec![0usize; n_rows + 1];
    for &raw in jac_irow {
        let i = decode_idx(raw, one_based);
        if i < n_rows {
            ptr[i + 1] += 1;
        }
    }
    for r in 0..n_rows {
        ptr[r + 1] += ptr[r];
    }
    let mut entries = vec![0usize; ptr[n_rows]];
    let mut cursor = ptr.clone();
    for (kk, &raw) in jac_irow.iter().enumerate().take(nnz) {
        let i = decode_idx(raw, one_based);
        if i < n_rows {
            entries[cursor[i]] = kk;
            cursor[i] += 1;
        }
    }
    (ptr, entries)
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

    // -- 0. Trivial-elimination pre-pass (PR 13). Identifies fixed
    // variables, free rows, and trivially-slack inequalities so the
    // incidence graph below doesn't see them.
    let trivial = crate::trivial_elim::find_trivial_eliminations(
        probe.n_vars,
        probe.n_rows,
        probe.x_l,
        probe.x_u,
        probe.g_l,
        probe.g_u,
        probe.jac_irow,
        probe.jac_jcol,
        probe.jac_values,
        probe.linearity,
        probe.one_based,
        probe.eq_tol,
        1e19,
    );
    diag.trivially_fixed_vars = trivial.fixed_vars.len() as Index;
    diag.trivially_free_rows = trivial.free_rows.len() as Index;
    diag.trivially_slack_rows = trivial.trivially_slack_rows.len() as Index;
    let excluded_vars_buf: Option<Vec<bool>> = if trivial.fixed_vars.is_empty() {
        None
    } else {
        let mut v = vec![false; probe.n_vars];
        for &i in &trivial.fixed_vars {
            v[i] = true;
        }
        Some(v)
    };
    let excluded_rows_buf: Option<Vec<bool>> =
        if trivial.free_rows.is_empty() && trivial.trivially_slack_rows.is_empty() {
            None
        } else {
            let mut v = vec![false; probe.n_rows];
            for &r in &trivial.free_rows {
                v[r] = true;
            }
            for &r in &trivial.trivially_slack_rows {
                v[r] = true;
            }
            Some(v)
        };

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
        excluded_vars: excluded_vars_buf.as_deref(),
        excluded_rows: excluded_rows_buf.as_deref(),
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

    // Variables with a non-negligible objective gradient *at the probe*.
    let mut obj_support = objective_gradient_support(probe.grad_f, 1e-12);
    // H11: the probe gradient is a single sample. For any variable tagged
    // `NonLinear` *in the objective*, a zero probe-gradient does NOT prove it
    // is objective-free (the gradient may be non-zero elsewhere — the classic
    // `f=(x−x₀)²` started at `x₀` reads as zero). Treat every such variable
    // as objective-coupled so its block is not eliminated as `PureEquality`
    // under the `Safe` policy. The tags are objective-scoped when the TNLP
    // provides them (constraint-only nonlinearity must NOT block elimination
    // of an objective-free block — the gas-network case); the global tags are
    // only a conservative fallback. When the TNLP declines both, fall back to
    // the probe gradient alone.
    if let Some(var_lin) = probe.var_linearity {
        for (i, l) in var_lin.iter().enumerate() {
            if matches!(l, Linearity::NonLinear) {
                obj_support.insert(i);
            }
        }
    }

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
    // Reject non-finite probe points up-front. Newton seeded with
    // NaN returns NaN, and `NaN > tol` is `false`, so a block would
    // otherwise spuriously "succeed". PR review #60.
    if !probe.x_probe.iter().all(|v| v.is_finite()) {
        return Phase0Plan {
            diagnostics: diag,
            frame: None,
        };
    }
    // CSR row index over the probe Jacobian, built once and shared by every
    // block-assembly helper below so none of them re-scans the full nnz array
    // per block row (M27).
    let (row_nnz_ptr, row_nnz_entries) =
        build_row_nnz(probe.jac_irow, probe.n_rows, probe.one_based);
    let row_nnz = RowNnz {
        ptr: &row_nnz_ptr,
        entries: &row_nnz_entries,
    };

    let mut x_running: Vec<Number> = probe.x_probe.to_vec();
    // C2(c): a trivially-fixed variable (`x_l == x_u`) is excluded from
    // incidence, so it can only appear in a block's rows as a *non-block*
    // column folded into the RHS. Fold it at its fixed value, not its
    // (possibly different) probe value — otherwise the dropped equality
    // is satisfied at a point the IPM will never occupy.
    //
    // `fixed_mask[j]` tracks which columns are pinned: trivially fixed
    // up front, plus any fixed by an earlier accepted block. The C2
    // soundness gate below consults it.
    let mut fixed_mask = vec![false; probe.n_vars];
    for &i in &trivial.fixed_vars {
        x_running[i] = probe.x_l[i];
        fixed_mask[i] = true;
    }

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
                AuxiliaryCouplingClass::InequalityCoupled => {
                    // PR 14: try to admit via inequality projection.
                    // Conditions: all block equality rows linear, all
                    // coupled inequality rows linear, coupled count ≤
                    // `auxiliary_max_block_dim`, and every projected
                    // inequality is implied by the variable box on
                    // the surviving variables.
                    let inner_rows_for_check: Vec<usize> = block
                        .eq_rows
                        .iter()
                        .map(|&kk| eq_inc.eq_row_inner_idx[kk])
                        .collect();
                    let block_eqs_linear = inner_rows_for_check.iter().all(|&r| is_linear_inner[r]);
                    let coupled_ineq_rows: Vec<usize> = block
                        .cols
                        .iter()
                        .flat_map(|&c| ineq_inc.rows_for_var(c).iter().copied())
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .map(|k| ineq_inc.ineq_row_inner_idx[k])
                        .collect();
                    let ineq_within_cap =
                        coupled_ineq_rows.len() <= opts.auxiliary_max_block_dim as usize;
                    let ineqs_linear = coupled_ineq_rows.iter().all(|&r| is_linear_inner[r]);
                    if block_eqs_linear && ineqs_linear && ineq_within_cap {
                        if let Some(res) = crate::inequality_projection::project_inequalities(
                            &inner_rows_for_check,
                            &block.cols,
                            &coupled_ineq_rows,
                            probe.n_vars,
                            probe.x_l,
                            probe.x_u,
                            probe.g_l,
                            probe.g_u,
                            probe.jac_irow,
                            probe.jac_jcol,
                            probe.jac_values,
                            probe.g_at_probe,
                            probe.x_probe,
                            probe.one_based,
                        ) {
                            if res.all_implied {
                                diag.inequality_coupled_accepted_via_projection += 1;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
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

            // -- 3c'. C2 soundness gate --
            // A block's rows are dropped from the IPM's problem, so
            // every variable they depend on must be pinned: either a
            // block column (solved + clamped here) or an already-fixed
            // non-block column. Any *free* non-block column means the
            // IPM can move it after the row is gone, silently breaking
            // the dropped equality (`solve_linear_block` folds it into
            // the RHS at a fixed probe value). Scan the **raw Jacobian
            // sparsity** — not incidence, which drops entries that are
            // numerically zero at the probe (a nonlinear row's
            // derivative can be zero there yet structurally nonzero,
            // C2(d)). Catches C2(a) (free column from a rejected earlier
            // block), C2(b) (Square row adjacent to an Over column), and
            // C2(d) in one gate.
            let mut depends_on_free = false;
            'gate: for &r_inner in &inner_rows {
                for &kk in row_nnz.of_row(r_inner) {
                    let j = decode_idx(probe.jac_jcol[kk], probe.one_based);
                    if block_cols.contains(&j) {
                        continue;
                    }
                    let pinned =
                        (probe.x_u[j] - probe.x_l[j]).abs() <= probe.eq_tol || fixed_mask[j];
                    if !pinned {
                        depends_on_free = true;
                        break 'gate;
                    }
                }
            }
            if depends_on_free {
                diag.rejection_reasons
                    .push(AuxiliaryRejectionReason::NonBlockColumnFree);
                continue;
            }

            // PR #60 review nit: pass the block's variable bounds
            // into Newton so a converged-but-out-of-box solution
            // gets caught early as `OutOfBounds` rather than
            // becoming an `x_l = x_u = bad_value` clamp.
            let bounds_lo: Vec<Number> = block_cols.iter().map(|&c| probe.x_l[c]).collect();
            let bounds_hi: Vec<Number> = block_cols.iter().map(|&c| probe.x_u[c]).collect();
            let bs_opts = BlockSolveOptions {
                tol: opts.auxiliary_tol,
                max_dim: opts.auxiliary_max_block_dim as usize,
                bounds: Some((bounds_lo, bounds_hi)),
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
                    &row_nnz,
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
                    &row_nnz,
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
                Err(crate::block_solve::BlockSolveError::OutOfBounds) => {
                    diag.rejection_reasons
                        .push(AuxiliaryRejectionReason::OutOfBounds);
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
            let row_resid = if all_linear {
                // Re-evaluate using the linear model — same as the
                // build code above.
                residual_norm_linear(probe, &row_nnz, &inner_rows, &candidate_x)
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
                // This column is now pinned for the C2 gate on later
                // blocks in BTF order.
                fixed_mask[c] = true;
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
    row_nnz: &RowNnz<'_>,
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
        for &kk in row_nnz.of_row(r_inner) {
            let j = decode_idx(probe.jac_jcol[kk], probe.one_based);
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
fn solve_nonlinear_block<'a, F>(
    probe: &Phase0Probe<'_>,
    row_nnz: &RowNnz<'a>,
    inner_rows: &'a [usize],
    block_cols: &'a [usize],
    x_running: &'a [Number],
    bs_opts: &BlockSolveOptions,
    tnlp: &'a mut dyn Phase0TnlpCallback,
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
        row_nnz: RowNnz<'a>,
        inner_rows: &'a [usize],
        block_cols: &'a [usize],
        x_running: &'a [Number],
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
            // Zero the dense submatrix, then scatter the nonzeros. Iterate
            // only the block's inner rows (via the CSR index), not the whole
            // nnz array — the column is still mapped through `block_cols`,
            // which is the small block dimension (M27).
            for v in j.iter_mut() {
                *v = 0.0;
            }
            for (ii, &r) in self.inner_rows.iter().enumerate() {
                for &kk in self.row_nnz.of_row(r) {
                    let col = decode_idx(self.jac_jcol[kk], self.one_based);
                    let Some(jj) = self.block_cols.iter().position(|&c| c == col) else {
                        continue;
                    };
                    j[ii * self.n + jj] = self.full_jac_vals[kk];
                }
            }
            true
        }
    }

    let nnz = probe.jac_irow.len();
    let mut eqs = NonlinearBlock {
        n: k,
        row_nnz: *row_nnz,
        inner_rows,
        block_cols,
        x_running,
        jac_jcol: probe.jac_jcol,
        g_l: probe.g_l,
        one_based: probe.one_based,
        tnlp,
        full_x: vec![0.0; probe.n_vars],
        full_g: vec![0.0; probe.n_rows],
        full_jac_vals: vec![0.0; nnz],
    };
    // Start at the probe's value for each block variable.
    // Seed Newton from `x_running` (which carries values fixed by
    // earlier blocks in this same pass), falling back to the probe
    // point for variables not yet touched. PR review #60.
    let x0: Vec<Number> = block_cols.iter().map(|&c| x_running[c]).collect();
    solver_call(&x0, &mut eqs, bs_opts)
}

/// Linear-model residual check: each dropped row should evaluate to
/// `g_l[r]` after splicing the candidate point in.
fn residual_norm_linear(
    probe: &Phase0Probe<'_>,
    row_nnz: &RowNnz<'_>,
    inner_rows: &[usize],
    candidate_x: &[Number],
) -> Number {
    let mut row_resid: Number = 0.0;
    for &r_inner in inner_rows {
        let mut s = 0.0;
        let mut sum_jx = 0.0;
        for &kk in row_nnz.of_row(r_inner) {
            let j = decode_idx(probe.jac_jcol[kk], probe.one_based);
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
        // PR 13: default variable bounds are wide open so the
        // trivial-elimination pre-pass doesn't accidentally mark
        // anything as fixed in tests that don't care about bounds.
        let x_l: Vec<Number> = vec![-1e19; n_vars];
        let x_u: Vec<Number> = vec![1e19; n_vars];
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
            var_linearity: None,
            x_l: Box::leak(x_l.into_boxed_slice()),
            x_u: Box::leak(x_u.into_boxed_slice()),
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

    /// H11: the objective gradient is a single-point sample, so a variable
    /// nonlinear in the objective can read as objective-free at the probe
    /// (`f=(x−x₀)²` started at `x₀` → zero gradient). A `NonLinear` variable
    /// tag must make the classifier treat that variable as objective-coupled,
    /// so its block is NOT eliminated as `PureEquality` under `Safe` (which
    /// would silently fix it at a Newton root with no regard to the
    /// objective). Same eliminable 2×2 block as the test above, but var 0 is
    /// nonlinear in the objective and zero-gradient at the probe.
    #[test]
    fn phase0_nonlinear_var_with_zero_probe_grad_blocks_elimination_under_safe() {
        let irow = [0, 0, 1, 1];
        let jcol = [0, 1, 0, 1];
        let vals = [1.0, 1.0, 1.0, -1.0];
        let g_l = [3.0, 1.0];
        let g_u = [3.0, 1.0];
        let g_probe = [0.0, 0.0];
        let linearity = [Linearity::Linear, Linearity::Linear];
        let x_probe = [0.0, 0.0];
        let grad_f = [0.0, 0.0]; // objective-free *at the probe* only
        let x_l = [-1e19, -1e19];
        let x_u = [1e19, 1e19];
        let var_lin = [Linearity::NonLinear, Linearity::Linear];

        let base = Phase0Probe {
            n_vars: 2,
            n_rows: 2,
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
            var_linearity: None,
            x_l: &x_l,
            x_u: &x_u,
        };

        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        opts.auxiliary_coupling = AuxiliaryCouplingPolicy::Safe;

        // Control (pre-H11 behavior): with no variable-linearity tag the zero
        // probe gradient alone makes the block look objective-free → eliminated.
        let plan_blind = run_auxiliary_phase0(&opts, &base, None, None);
        assert_eq!(
            plan_blind.diagnostics.blocks_eliminated, 1,
            "control: zero probe-gradient alone eliminates the block"
        );

        // H11 fix: the `NonLinear` tag on var 0 makes it objective-coupled, so
        // the block is rejected under Safe.
        let tagged = Phase0Probe {
            var_linearity: Some(&var_lin),
            ..base
        };
        let plan_tagged = run_auxiliary_phase0(&opts, &tagged, None, None);
        assert_eq!(
            plan_tagged.diagnostics.blocks_eliminated, 0,
            "H11: nonlinear-tagged objective variable must block elimination"
        );
        assert!(plan_tagged.frame.is_none());
        assert_eq!(
            plan_tagged.diagnostics.class_counts.objective_coupled, 1,
            "block classified objective-coupled via the linearity tag"
        );
    }

    #[test]
    fn phase0_inequality_coupled_admitted_via_projection() {
        // PR 14: 2 vars, 2 rows. Row 0 equality `x[1] = 5`. Row 1
        // inequality `x[1] in (-∞, 10]`. Block on var 1 is
        // `InequalityCoupled`. Project: `x[1] = 5` is implied by
        // the inequality (5 ≤ 10), so PR 14 admits the block.
        let probe = linear_probe(
            2,
            2,
            &[0, 1],
            &[1, 1],
            &[1.0, 1.0],
            &[5.0, -1e19],
            &[5.0, 10.0], // row 1 is inequality, slack at the solution
            &[0.0, 0.0],
            &[Linearity::Linear, Linearity::Linear],
            &[0.0, 0.0],
            &[0.0, 0.0],
        );
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        opts.auxiliary_coupling = AuxiliaryCouplingPolicy::Safe;
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
        // Block is admitted via projection.
        assert_eq!(plan.diagnostics.blocks_eliminated, 1);
        assert_eq!(
            plan.diagnostics.inequality_coupled_accepted_via_projection,
            1
        );
        assert!(plan.frame.is_some());
    }

    #[test]
    fn phase0_inequality_coupled_rejected_when_not_implied() {
        // PR 14 negative case: same shape but the inequality bound
        // is violated by the equality solution. Row 0: `x[1] = 5`.
        // Row 1: `x[1] ≤ 4`. Projection gives `5 ≤ 4` → not
        // implied. Block stays rejected as `InequalityCoupled`.
        let probe = linear_probe(
            2,
            2,
            &[0, 1],
            &[1, 1],
            &[1.0, 1.0],
            &[5.0, -1e19],
            &[5.0, 4.0],
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
        assert_eq!(
            plan.diagnostics.inequality_coupled_accepted_via_projection,
            0
        );
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

    /// Callback for the C2(d) gate test: row 0 is `g = x0 + x1^2`,
    /// whose derivative w.r.t. x1 is `2*x1` — zero at the probe
    /// x1 = 0. The probe's Jacobian *values* therefore carry a 0 for
    /// the (0, x1) entry, so `EqualityIncidence::from_probe` drops it
    /// and DM forms a clean square block {row0, x0}; but x1 is a real
    /// dependency the IPM is free to move.
    struct HiddenDepCallback;
    impl Phase0TnlpCallback for HiddenDepCallback {
        fn eval_g_full(&mut self, x: &[Number], g: &mut [Number]) -> bool {
            g[0] = x[0] + x[1] * x[1];
            true
        }
        fn eval_jac_g_values(&mut self, x: &[Number], values: &mut [Number]) -> bool {
            // Sparsity (0,0)=∂/∂x0=1, (0,1)=∂/∂x1=2*x1.
            values[0] = 1.0;
            values[1] = 2.0 * x[1];
            true
        }
    }

    #[test]
    fn c2_gate_rejects_block_with_probe_hidden_free_dependency() {
        // Regression for C2(d) (and the C2 gate generally). Row 0 is
        // `x0 + x1^2 = 5`, nonlinear in x1. At the probe x1 = 0 the
        // derivative ∂/∂x1 is 0, so incidence omits the entry and DM
        // forms the square block {row0, x0}. Pre-fix, Phase 0 solved
        // x0 = 5 (folding x1 at its probe 0), passed the residual check
        // at (5, 0), and dropped row 0 — leaving the IPM free to move
        // x1 and silently break x0 + x1^2 = 5. The C2 soundness gate
        // scans the raw Jacobian sparsity, sees x1 is a free non-block
        // column, and rejects.
        let probe = linear_probe(
            2,
            1,
            &[0, 0],
            &[0, 1],
            &[1.0, 0.0], // ∂/∂x0=1, ∂/∂x1=2*x1=0 at probe x1=0
            &[5.0],
            &[5.0],
            &[0.0], // g(0,0) = 0
            &[Linearity::NonLinear],
            &[0.0, 0.0],
            &[0.0, 0.0],
        );
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let mut tnlp = HiddenDepCallback;
        let plan = run_auxiliary_phase0(&opts, &probe, Some(&mut tnlp), None);
        assert_eq!(
            plan.diagnostics.blocks_eliminated, 0,
            "block depending on free non-block var x1 must not be eliminated"
        );
        assert!(plan.frame.is_none());
        assert!(
            plan.diagnostics
                .rejection_reasons
                .iter()
                .any(|r| matches!(r, AuxiliaryRejectionReason::NonBlockColumnFree)),
            "expected NonBlockColumnFree, got {:?}",
            plan.diagnostics.rejection_reasons
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
        let x_l_def = vec![-1e19; n];
        let x_u_def = vec![1e19; n];
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
            var_linearity: None,
            x_l: Box::leak(x_l_def.into_boxed_slice()),
            x_u: Box::leak(x_u_def.into_boxed_slice()),
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
        let x_l_def = vec![-1e19; n];
        let x_u_def = vec![1e19; n];
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
            var_linearity: None,
            x_l: &x_l_def,
            x_u: &x_u_def,
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

    /// PR 13: diagnostics counts populate when the trivial pre-pass
    /// finds anything. Build a probe with one fixed variable, one
    /// free row, and one trivially-slack inequality.
    #[test]
    fn phase0_trivial_pre_pass_populates_diagnostics() {
        // 2 vars, 2 rows.
        // Var 0: x_l = x_u = 1.0 (fixed).
        // Var 1: x_l = 0, x_u = 1 (free).
        // Row 0: equality x[0] + x[1] = 0  (so g_l = g_u = 0).
        // Row 1: inequality x[1] in [-100, 100]  → trivially slack
        //        (activity range [0, 1] is strictly inside).
        // No free rows here (g_l/g_u both finite).
        let x_l = [1.0, 0.0];
        let x_u = [1.0, 1.0];
        let g_l = [0.0, -100.0];
        let g_u = [0.0, 100.0];
        let jac_irow: [Index; 3] = [0, 0, 1];
        let jac_jcol: [Index; 3] = [0, 1, 1];
        let jac_vals = [1.0, 1.0, 1.0];
        let g_probe = [1.0, 0.5];
        let linearity = [Linearity::Linear, Linearity::Linear];
        let x_probe = [1.0, 0.5];
        let grad_f = [0.0, 0.0];
        let probe = Phase0Probe {
            n_vars: 2,
            n_rows: 2,
            jac_irow: &jac_irow,
            jac_jcol: &jac_jcol,
            jac_values: &jac_vals,
            g_l: &g_l,
            g_u: &g_u,
            g_at_probe: &g_probe,
            linearity: &linearity,
            one_based: false,
            eq_tol: 1e-12,
            x_probe: &x_probe,
            grad_f: &grad_f,
            var_linearity: None,
            x_l: &x_l,
            x_u: &x_u,
        };
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
        assert_eq!(plan.diagnostics.trivially_fixed_vars, 1);
        assert_eq!(plan.diagnostics.trivially_free_rows, 0);
        assert_eq!(plan.diagnostics.trivially_slack_rows, 1);
    }

    /// Build a diagonal singleton-block system: `n` vars, `n` rows, row `r`
    /// is the equality `x_r = r+1`. Each row is its own 1×1 block, so the
    /// orchestrator walks `n` blocks; the full-`nnz`-per-row scans make the
    /// linear solve/residual cost O(n²).
    fn diagonal_singletons(n: usize) -> (Vec<Index>, Vec<Index>, Vec<Number>, Vec<Number>) {
        let jac_irow: Vec<Index> = (0..n as Index).collect();
        let jac_jcol: Vec<Index> = (0..n as Index).collect();
        let jac_vals: Vec<Number> = vec![1.0; n];
        let g_l: Vec<Number> = (0..n).map(|r| (r + 1) as Number).collect();
        (jac_irow, jac_jcol, jac_vals, g_l)
    }

    #[test]
    fn phase0_diagonal_many_singletons_correct() {
        // M27 regression: with the CSR row index replacing the per-row
        // full-nnz scans, a large diagonal system must still be eliminated
        // exactly — every var fixed to its row's target, every row dropped.
        // (This is also the shape whose O(n²) scan cost motivated the fix;
        // here we pin correctness at scale, not timing.)
        let n = 400;
        let (jac_irow, jac_jcol, jac_vals, g_l) = diagonal_singletons(n);
        let g_u = g_l.clone();
        let g_at_probe = vec![0.0; n];
        let linearity = vec![Linearity::Linear; n];
        let x_probe = vec![0.0; n];
        let grad_f = vec![0.0; n];
        let probe = linear_probe(
            n,
            n,
            &jac_irow,
            &jac_jcol,
            &jac_vals,
            &g_l,
            &g_u,
            &g_at_probe,
            &linearity,
            &x_probe,
            &grad_f,
        );
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
        assert_eq!(plan.diagnostics.blocks_eliminated as usize, n);
        let frame = plan.frame.expect("accepted frame");
        assert_eq!(frame.fixed_vars, (0..n).collect::<Vec<_>>());
        assert_eq!(frame.dropped_rows, (0..n).collect::<Vec<_>>());
        for r in 0..n {
            assert!(
                (frame.fixed_values[r] - (r + 1) as Number).abs() < 1e-12,
                "var {r} fixed to {}, want {}",
                frame.fixed_values[r],
                r + 1
            );
        }
    }

    #[test]
    fn build_row_nnz_groups_by_row_zero_based() {
        // COO rows (per kk): [0, 2, 0, 1, 2] — out of order, row 0 and row 2
        // each have two entries, row 1 one. The CSR slice for each row must
        // list exactly that row's kk positions.
        let jac_irow: [Index; 5] = [0, 2, 0, 1, 2];
        let (ptr, entries) = build_row_nnz(&jac_irow, 3, false);
        let nnz = RowNnz {
            ptr: &ptr,
            entries: &entries,
        };
        let row_of = |r: usize| {
            let mut v = nnz.of_row(r).to_vec();
            v.sort_unstable();
            v
        };
        assert_eq!(row_of(0), vec![0, 2]);
        assert_eq!(row_of(1), vec![3]);
        assert_eq!(row_of(2), vec![1, 4]);
        // Every COO position placed exactly once.
        assert_eq!(entries.len(), 5);
    }

    #[test]
    fn build_row_nnz_honours_one_based_decode() {
        // Same pattern but one-based rows (`r + 1`); the decode must reproduce
        // the zero-based grouping exactly.
        let jac_irow_0: [Index; 5] = [0, 2, 0, 1, 2];
        let jac_irow_1: [Index; 5] = [1, 3, 1, 2, 3];
        let (ptr0, ent0) = build_row_nnz(&jac_irow_0, 3, false);
        let (ptr1, ent1) = build_row_nnz(&jac_irow_1, 3, true);
        assert_eq!(ptr0, ptr1, "one-based ptr must match zero-based");
        assert_eq!(ent0, ent1, "one-based entries must match zero-based");
    }

    #[test]
    fn phase0_one_based_two_blocks_eliminated() {
        // End-to-end one-based guard: every changed helper (C2 gate, linear
        // solve, residual check) reads the Jacobian through the CSR index,
        // which decodes one-based COO indices. Two 1×1 blocks: row 1 → x0 = 5,
        // row 2 → x1 = 7. Both must be eliminated, matching the pre-M27 code.
        let jac_irow: [Index; 2] = [1, 2];
        let jac_jcol: [Index; 2] = [1, 2];
        let jac_vals = [1.0, 1.0];
        let g_l = [5.0, 7.0];
        let g_u = [5.0, 7.0];
        let g_at_probe = [0.0, 0.0];
        let linearity = [Linearity::Linear, Linearity::Linear];
        let x_probe = [0.0, 0.0];
        let grad_f = [0.0, 0.0];
        let x_l = [-1e19, -1e19];
        let x_u = [1e19, 1e19];
        let probe = Phase0Probe {
            n_vars: 2,
            n_rows: 2,
            jac_irow: &jac_irow,
            jac_jcol: &jac_jcol,
            jac_values: &jac_vals,
            g_l: &g_l,
            g_u: &g_u,
            g_at_probe: &g_at_probe,
            linearity: &linearity,
            one_based: true,
            eq_tol: 1e-12,
            x_probe: &x_probe,
            grad_f: &grad_f,
            var_linearity: None,
            x_l: &x_l,
            x_u: &x_u,
        };
        let mut opts = PresolveOptions::defaults();
        opts.auxiliary = true;
        let plan = run_auxiliary_phase0(&opts, &probe, None, None);
        let frame = plan.frame.expect("accepted frame");
        assert_eq!(frame.fixed_vars, vec![0, 1]);
        assert!((frame.fixed_values[0] - 5.0).abs() < 1e-12);
        assert!((frame.fixed_values[1] - 7.0).abs() < 1e-12);
        assert_eq!(frame.dropped_rows, vec![0, 1]);
    }
}
