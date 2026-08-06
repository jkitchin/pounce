//! Phase 6 — the TNLP wrapper that applies a [`EliminationPlan`] (issue
//! #487).
//!
//! [`crate::linear_eq_plan`] decides *which* variables the linear equality
//! rows determine; this module re-presents the problem with those columns
//! and rows gone, and puts the solution back together afterwards.
//!
//! # The transform
//!
//! The plan is an affine map `x = A·y + c`, where `A` has exactly one
//! non-zero per row (each eliminated column is a multiple of a **single**
//! surviving column, or a constant). Everything else follows from the
//! chain rule:
//!
//! ```text
//!   f_red(y)  = f(A y + c)
//!   ∇f_red    = Aᵀ ∇f                     (scaled gather onto columns)
//!   g_red(y)  = g(A y + c) at kept rows
//!   J_red     = (J A) at kept rows        (scaled gather onto columns)
//!   ∇²L_red   = Aᵀ ∇²L A                  (scaled gather onto both axes)
//! ```
//!
//! Because `A` has one non-zero per row, none of these is a sparse
//! matrix product: each reduced non-zero is a fixed, precomputed sum of
//! scaled inner non-zeros, so an evaluation is one inner call plus a
//! linear pass over a gather list.
//!
//! # Postsolve
//!
//! `finalize_solution` lifts `x` back to full space and recovers a
//! multiplier for every consumed row. The recovery is a **triangular**
//! sweep, not a factorization, and that is worth spelling out because it
//! is the reason this pass scales.
//!
//! Write the dropped rows `D` and kept rows `K`. Full-space stationarity,
//! in the convention POUNCE hands to `finalize_solution` (Ipopt's: the
//! Lagrangian is `f + λᵀg`), is
//! `∇f + J_Kᵀλ_K + J_Dᵀλ_D − z_l + z_u = 0`. Restricted to the eliminated
//! columns — where the bound multipliers are known before the sweep starts,
//! either zero or whatever the attribution below placed there — that is a
//! square system `J_D[:,e]ᵀ λ_D = −(∇f_e + J_K[:,e]ᵀ λ_K − z_l,e + z_u,e)`
//! — which, taken all at once, is a sparse solve nobody wants inside
//! presolve.
//!
//! Taken one elimination at a time in **reverse** order it is triangular
//! instead. Step `t` consumed row `r_t` to determine variable `v_t`; every
//! other row of the problem-as-it-stood-then survives step `t`, so
//! stationarity at column `v_t` in that intermediate problem involves
//! exactly one unknown multiplier, `λ_{r_t}` — every other dropped row in
//! it is `r_{t+1}..r_k`, already recovered by the time the reverse sweep
//! reaches `t`. The pivot is the step's own `pivot` (non-zero by
//! construction), and the intermediate problem's gradient at `v_t` is the
//! sum of the full-space stationarity residual over `v_t`'s cluster,
//! weighted by each member's coefficient — which is just a subtree sum
//! over the elimination forest.
//!
//! Rows dropped as `0 = 0` are linearly dependent on the rest and get
//! `λ = 0`; that is a valid choice, not an approximation, because their
//! Jacobian row is annihilated by `A` and so contributes nothing to
//! stationarity in either space.
//!
//! # Bound multipliers, and where they belong
//!
//! An eliminated column's box does not disappear during planning: it is
//! *transferred* onto its survivor, so the reduced problem's box is an
//! intersection of boxes borrowed from the whole cluster. The IPM therefore
//! reports one `z_l` / `z_u` pair per surviving column, for a bound that
//! may well have been declared on a column that is no longer there.
//!
//! Handing that multiplier back verbatim to the survivor is a valid KKT
//! point but the wrong *attribution*: a no-presolve solve reports it on the
//! column that owns the bound. [`EliminationPlan::x_l_src`] /
//! [`EliminationPlan::x_u_src`] record which column each reduced bound came
//! from, and `attribute_bound_multiplier` routes the multiplier there
//! (issue #493).
//!
//! Two details make that routing exact rather than approximate. With
//! `x_src = α·x_kept + β`, the survivor's bound is the origin's bound
//! divided by `α`, so the multiplier — a derivative with respect to the
//! *other* variable — is multiplied by `1/|α|` for the chain rule to close.
//! And a negative `α` reverses the interval, so a **lower** bound on the
//! survivor is an **upper** bound of the eliminated column, and the
//! multiplier has to change sides with it. That is the same flip
//! `transfer_bounds` performs on the way in.
//!
//! With the bound multipliers placed first, the dual sweep below starts
//! from the full stationarity residual `∇f + Jᵀλ − z_l + z_u` rather than
//! from `∇f + Jᵀλ` alone, so the recovered row multipliers absorb the
//! difference and full-space stationarity closes either way.
//!
//! # What is still non-unique
//!
//! When the survivor's *own* bound and a transferred bound are active at
//! the same point, the reduced problem has one multiplier where the full
//! problem has two, and any split summing correctly is a valid KKT point.
//! The plan breaks that tie in favour of the incumbent (see
//! `transfer_bounds`'s strict comparisons), so the multiplier stays on the
//! survivor. Tests pin KKT validity there, not equality with a bare solve —
//! the posture `pounce-convex` already takes for forcing constraints.
//!
//! A column pinned to a constant by a singleton row `a·x = b` is the other
//! degenerate case: if `b/a` lands on that column's bound, the row
//! multiplier and the bound multiplier are interchangeable, and the sweep
//! puts the whole thing on the row. That too is a valid KKT point, and the
//! same class of caveat as the Phase-2 note in [`crate`]'s module docs
//! (issue M24).

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, InfeasibilityProof, IpoptCq, IpoptData, IterStats, Linearity, MetaData,
    NlpInfo, ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};

use crate::linear_eq_plan::{
    EliminationPlan, LinearEqElimReport, PlanConfig, PlanInput, VarRecovery, build_plan,
};
use crate::options::PresolveOptions;

/// A gather list: `out[k] = Σ scale · inner_values[src]` over the terms in
/// `terms[start[k]..start[k+1]]`.
#[derive(Debug, Default, Clone)]
struct Gather {
    start: Vec<usize>,
    terms: Vec<(usize, Number)>,
}

impl Gather {
    fn len(&self) -> usize {
        self.start.len().saturating_sub(1)
    }

    fn apply(&self, src: &[Number], out: &mut [Number]) {
        for (k, slot) in out.iter_mut().enumerate().take(self.len()) {
            let mut acc = 0.0;
            for &(i, s) in &self.terms[self.start[k]..self.start[k + 1]] {
                acc += s * src[i];
            }
            *slot = acc;
        }
    }
}

/// Build a [`Gather`] from unsorted `(slot, source, scale)` triples,
/// returning the distinct slots in ascending order alongside it.
fn gather_from(mut triples: Vec<((Index, Index), usize, Number)>) -> (Vec<(Index, Index)>, Gather) {
    triples.sort_by(|a, b| a.0.cmp(&b.0));
    let mut slots: Vec<(Index, Index)> = Vec::new();
    let mut g = Gather {
        start: vec![0],
        terms: Vec::with_capacity(triples.len()),
    };
    for (slot, src, scale) in triples {
        if slots.last() != Some(&slot) {
            slots.push(slot);
            g.start.push(g.terms.len());
        }
        g.terms.push((src, scale));
        let last = g.start.len() - 1;
        g.start[last] = g.terms.len();
    }
    (slots, g)
}

struct ElimState {
    info_inner: NlpInfo,
    info_outer: NlpInfo,
    plan: EliminationPlan,
    /// The plan removes nothing, so every method forwards verbatim. This
    /// is not just an optimisation: it is also the state the wrapper falls
    /// into when the option is off or the inner TNLP cannot supply what
    /// the transform needs, and forwarding is the only behaviour that is
    /// certainly right there.
    passthrough: bool,
    /// Reduced constraint bounds, aligned with `plan.rows_kept`.
    g_l_red: Vec<Number>,
    g_u_red: Vec<Number>,
    /// `col_of[i]` is where full column `i` lands: the reduced column and
    /// the scale, or `None` when the variable became a constant.
    col_of: Vec<Option<(usize, Number)>>,
    /// `∇f_red = Aᵀ ∇f`, as a gather over full-space gradient entries.
    grad_gather: Gather,
    jac_irow_outer: Vec<Index>,
    jac_jcol_outer: Vec<Index>,
    jac_gather: Gather,
    h_irow_outer: Vec<Index>,
    h_jcol_outer: Vec<Index>,
    h_gather: Gather,
    /// Full-space Jacobian structure, kept for the postsolve multiplier
    /// recovery (which reads `J` by row).
    jac_irow_inner: Vec<Index>,
    jac_jcol_inner: Vec<Index>,
    /// Reduced nonlinear-variable list, `None` until first asked for.
    nonlinear_vars: Option<Vec<Index>>,

    scratch_x: Vec<Number>,
    scratch_g: Vec<Number>,
    scratch_grad: Vec<Number>,
    scratch_jac: Vec<Number>,
    scratch_h: Vec<Number>,
    scratch_lambda: Vec<Number>,
}

/// Full-space solution quantities captured at `finalize_solution`.
#[derive(Debug, Clone, Default)]
pub struct FullSolution {
    pub x: Vec<Number>,
    pub lambda: Vec<Number>,
    pub z_l: Vec<Number>,
    pub z_u: Vec<Number>,
}

/// TNLP wrapper that re-presents its inner problem with the columns and
/// rows the linear equality system determines removed.
///
/// Sits **outside** [`crate::PresolveTnlp`] so the phases below it keep
/// operating on the shape they were written against; this layer is the
/// only one that changes the variable count.
pub struct LinearEqElimTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    opts: PresolveOptions,
    state: Option<ElimState>,
    finalized: Option<FullSolution>,
}

impl LinearEqElimTnlp {
    pub fn new(inner: Rc<RefCell<dyn TNLP>>, opts: PresolveOptions) -> Self {
        Self {
            inner,
            opts,
            state: None,
            finalized: None,
        }
    }

    /// What the pass achieved. Zeroed until init has run.
    pub fn report(&self) -> LinearEqElimReport {
        self.state
            .as_ref()
            .map(|s| s.plan.report)
            .unwrap_or_default()
    }

    /// Number of variables removed from the problem the solver sees.
    pub fn n_eliminated_vars(&self) -> usize {
        self.state
            .as_ref()
            .map(|s| s.plan.n_full - s.plan.n_reduced_vars())
            .unwrap_or(0)
    }

    /// Number of rows removed by this pass.
    pub fn n_eliminated_rows(&self) -> usize {
        self.state
            .as_ref()
            .map(|s| s.plan.m_full - s.plan.n_reduced_rows())
            .unwrap_or(0)
    }

    /// The full-space `(x, λ, z_l, z_u)` this wrapper handed down at the
    /// last `finalize_solution`. `None` until a solve finalizes.
    ///
    /// Frontends that write a solution file positionally against the
    /// original model — the CLI's `.sol` writer, the JSON report — must
    /// prefer this over anything captured outside the wrapper, which is in
    /// the reduced space and therefore the wrong length.
    pub fn finalized_full_solution(&self) -> Option<&FullSolution> {
        self.finalized.as_ref()
    }

    fn ensure_init(&mut self) -> Option<&ElimState> {
        if self.state.is_some() {
            return self.state.as_ref();
        }
        let info_inner = self.inner.borrow_mut().get_nlp_info()?;
        let n = info_inner.n.max(0) as usize;
        let m = info_inner.m.max(0) as usize;
        let nnz_jac = info_inner.nnz_jac_g.max(0) as usize;
        let nnz_h = info_inner.nnz_h_lag.max(0) as usize;
        let one_based = matches!(info_inner.index_style, IndexStyle::Fortran);

        let mut x_l = vec![0.0; n];
        let mut x_u = vec![0.0; n];
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        if !self.inner.borrow_mut().get_bounds_info(BoundsInfo {
            x_l: &mut x_l,
            x_u: &mut x_u,
            g_l: &mut g_l,
            g_u: &mut g_u,
        }) {
            return None;
        }

        let mut jac_irow = vec![0 as Index; nnz_jac];
        let mut jac_jcol = vec![0 as Index; nnz_jac];
        if nnz_jac > 0
            && !self.inner.borrow_mut().eval_jac_g(
                None,
                false,
                SparsityRequest::Structure {
                    irow: &mut jac_irow,
                    jcol: &mut jac_jcol,
                },
            )
        {
            return None;
        }

        // The Hessian structure is needed up front because `nnz_h_lag` is
        // part of `get_nlp_info`. A TNLP that declines it (quasi-Newton
        // bridges, callback shims) simply does not get the reduction — the
        // alternative would be publishing a variable count the Hessian
        // transform cannot honour.
        let mut h_irow = vec![0 as Index; nnz_h];
        let mut h_jcol = vec![0 as Index; nnz_h];
        if nnz_h > 0
            && !self.inner.borrow_mut().eval_h(
                None,
                false,
                1.0,
                None,
                false,
                SparsityRequest::Structure {
                    irow: &mut h_irow,
                    jcol: &mut h_jcol,
                },
            )
        {
            tracing::warn!(
                target: "pounce::presolve",
                "linear-equality variable elimination is off for this solve: the \
                 inner TNLP declined to publish its Hessian structure."
            );
            self.install_passthrough(info_inner, LinearEqElimReport::default());
            return self.state.as_ref();
        }

        // The option is checked here, not only where the wrapper is
        // stacked, so a caller that constructs it directly still honours
        // `presolve_linear_eq_reduction=no`.
        if !self.opts.linear_eq_reduction {
            self.install_passthrough(info_inner, LinearEqElimReport::default());
            return self.state.as_ref();
        }

        // Row linearity, and a probe point at which a linear row's Jacobian
        // values are its exact coefficients.
        let mut linearity = vec![Linearity::NonLinear; m];
        let have_linearity = m == 0
            || self
                .inner
                .borrow_mut()
                .get_constraints_linearity(&mut linearity);
        if !have_linearity {
            // Without linearity tags nothing is eligible; skip the probe work.
            self.install_passthrough(info_inner, LinearEqElimReport::default());
            return self.state.as_ref();
        }

        let mut x_probe = vec![0.0; n];
        let mut z_l_probe = vec![0.0; n];
        let mut z_u_probe = vec![0.0; n];
        let mut lambda_probe = vec![0.0; m];
        if !self.inner.borrow_mut().get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x_probe,
            init_z: false,
            z_l: &mut z_l_probe,
            z_u: &mut z_u_probe,
            init_lambda: false,
            lambda: &mut lambda_probe,
        }) {
            return None;
        }
        let mut jac_values = vec![0.0; nnz_jac];
        if nnz_jac > 0
            && !self.inner.borrow_mut().eval_jac_g(
                Some(&x_probe),
                true,
                SparsityRequest::Values {
                    values: &mut jac_values,
                },
            )
        {
            return None;
        }
        let mut g_probe = vec![0.0; m];
        if m > 0 && !self.inner.borrow_mut().eval_g(&x_probe, true, &mut g_probe) {
            return None;
        }

        let mut rows: Vec<Vec<(usize, Number)>> = vec![Vec::new(); m];
        for k in 0..nnz_jac {
            let (i, j) = decode(jac_irow[k], jac_jcol[k], one_based);
            if i < m && j < n {
                rows[i].push((j, jac_values[k]));
            }
        }
        // `g_r(x) = Σ a_j x_j + c` for a linear row, so read `c` off the
        // probe. Assuming `c = 0` would silently shift every row whose
        // model text keeps a constant on the left.
        let row_const: Vec<Number> = (0..m)
            .map(|r| {
                let lin: Number = rows[r].iter().map(|&(j, a)| a * x_probe[j]).sum();
                g_probe[r] - lin
            })
            .collect();
        let eligible: Vec<bool> = (0..m)
            .map(|r| matches!(linearity[r], Linearity::Linear))
            .collect();

        let cfg = PlanConfig {
            feas_tol: self.opts.certify_tol,
            max_passes: (self.opts.max_passes.max(1) as usize).max(20),
            ..PlanConfig::default()
        };
        let plan = build_plan(
            &PlanInput {
                n_vars: n,
                n_rows: m,
                rows: &rows,
                row_const: &row_const,
                g_l: &g_l,
                g_u: &g_u,
                eligible: &eligible,
                x_l: &x_l,
                x_u: &x_u,
            },
            &cfg,
        );
        if plan.report.infeasible {
            tracing::warn!(
                target: "pounce::presolve",
                "linear-equality elimination found the equality system contradictory; \
                 standing down and handing the model to the solver untouched."
            );
        }
        if plan.report.pass_cap_hit {
            tracing::info!(
                target: "pounce::presolve",
                passes = plan.report.passes,
                "linear-equality elimination hit its sweep cap with candidate rows \
                 still open; the reduction is valid but not maximal."
            );
        }

        self.install(
            info_inner, plan, g_l, g_u, jac_irow, jac_jcol, h_irow, h_jcol,
        );
        self.state.as_ref()
    }

    /// Install the do-nothing state: dimensions unchanged, every call
    /// forwarded.
    fn install_passthrough(&mut self, info_inner: NlpInfo, report: LinearEqElimReport) {
        let n = info_inner.n.max(0) as usize;
        let m = info_inner.m.max(0) as usize;
        let mut plan = EliminationPlan::identity(n, m, &vec![0.0; n], &vec![0.0; n]);
        plan.report = report;
        self.state = Some(ElimState {
            info_inner,
            info_outer: info_inner,
            plan,
            passthrough: true,
            g_l_red: Vec::new(),
            g_u_red: Vec::new(),
            col_of: Vec::new(),
            grad_gather: Gather::default(),
            jac_irow_outer: Vec::new(),
            jac_jcol_outer: Vec::new(),
            jac_gather: Gather::default(),
            h_irow_outer: Vec::new(),
            h_jcol_outer: Vec::new(),
            h_gather: Gather::default(),
            jac_irow_inner: Vec::new(),
            jac_jcol_inner: Vec::new(),
            nonlinear_vars: None,
            scratch_x: Vec::new(),
            scratch_g: Vec::new(),
            scratch_grad: Vec::new(),
            scratch_jac: Vec::new(),
            scratch_h: Vec::new(),
            scratch_lambda: Vec::new(),
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn install(
        &mut self,
        info_inner: NlpInfo,
        plan: EliminationPlan,
        g_l: Vec<Number>,
        g_u: Vec<Number>,
        jac_irow: Vec<Index>,
        jac_jcol: Vec<Index>,
        h_irow: Vec<Index>,
        h_jcol: Vec<Index>,
    ) {
        if plan.is_identity() {
            self.install_passthrough(info_inner, plan.report);
            return;
        }
        let n = plan.n_full;
        let m = plan.m_full;
        let one_based = matches!(info_inner.index_style, IndexStyle::Fortran);
        let bump = |v: usize| -> Index {
            if one_based {
                v as Index + 1
            } else {
                v as Index
            }
        };

        let col_of: Vec<Option<(usize, Number)>> = plan
            .recovery
            .iter()
            .map(|rec| match *rec {
                VarRecovery::Kept(j) => Some((j, 1.0)),
                VarRecovery::Constant(_) => None,
                VarRecovery::Affine { rep, coeff, .. } => match plan.recovery[rep] {
                    VarRecovery::Kept(j) => Some((j, coeff)),
                    // `build_plan` guarantees a representative survives; if
                    // that ever breaks, drop the column rather than index a
                    // slot that does not exist.
                    _ => None,
                },
            })
            .collect();

        // ∇f_red = Aᵀ ∇f.
        let mut grad_triples: Vec<((Index, Index), usize, Number)> = Vec::new();
        for (i, slot) in col_of.iter().enumerate() {
            if let Some((j, s)) = *slot {
                grad_triples.push(((j as Index, 0), i, s));
            }
        }
        let (_, grad_gather) = gather_from(grad_triples);

        // Row renumbering.
        let mut row_of = vec![usize::MAX; m];
        for (red, &full) in plan.rows_kept.iter().enumerate() {
            row_of[full] = red;
        }

        // J_red = (J A) restricted to kept rows.
        let mut jac_triples: Vec<((Index, Index), usize, Number)> = Vec::new();
        for k in 0..jac_irow.len() {
            let (i, j) = decode(jac_irow[k], jac_jcol[k], one_based);
            if i >= m || j >= n || !plan.row_kept[i] {
                continue;
            }
            if let Some((col, s)) = col_of[j] {
                jac_triples.push(((row_of[i] as Index, col as Index), k, s));
            }
        }
        let (jac_slots, jac_gather) = gather_from(jac_triples);
        let jac_irow_outer: Vec<Index> = jac_slots.iter().map(|s| bump(s.0 as usize)).collect();
        let jac_jcol_outer: Vec<Index> = jac_slots.iter().map(|s| bump(s.1 as usize)).collect();

        // ∇²L_red = Aᵀ ∇²L A. The inner matrix is stored as one triangle;
        // an off-diagonal entry (r, c) therefore stands for the symmetric
        // pair, and when both of its columns land on the *same* reduced
        // column the two orderings both hit the diagonal — hence the 2.
        let h_upper = h_irow.iter().zip(h_jcol.iter()).any(|(&r, &c)| r < c)
            && !h_irow.iter().zip(h_jcol.iter()).any(|(&r, &c)| r > c);
        let mut h_triples: Vec<((Index, Index), usize, Number)> = Vec::new();
        for k in 0..h_irow.len() {
            let (r, c) = decode(h_irow[k], h_jcol[k], one_based);
            if r >= n || c >= n {
                continue;
            }
            let (Some((pr, sr)), Some((pc, sc))) = (col_of[r], col_of[c]) else {
                continue;
            };
            let mut scale = sr * sc;
            if r != c && pr == pc {
                scale *= 2.0;
            }
            let (hi, lo) = if pr >= pc { (pr, pc) } else { (pc, pr) };
            let slot = if h_upper {
                (lo as Index, hi as Index)
            } else {
                (hi as Index, lo as Index)
            };
            h_triples.push((slot, k, scale));
        }
        let (h_slots, h_gather) = gather_from(h_triples);
        let h_irow_outer: Vec<Index> = h_slots.iter().map(|s| bump(s.0 as usize)).collect();
        let h_jcol_outer: Vec<Index> = h_slots.iter().map(|s| bump(s.1 as usize)).collect();

        let g_l_red: Vec<Number> = plan.rows_kept.iter().map(|&r| g_l[r]).collect();
        let g_u_red: Vec<Number> = plan.rows_kept.iter().map(|&r| g_u[r]).collect();

        let info_outer = NlpInfo {
            n: plan.n_reduced_vars() as Index,
            m: plan.n_reduced_rows() as Index,
            nnz_jac_g: jac_irow_outer.len() as Index,
            nnz_h_lag: h_irow_outer.len() as Index,
            index_style: info_inner.index_style,
        };

        self.state = Some(ElimState {
            info_inner,
            info_outer,
            plan,
            passthrough: false,
            g_l_red,
            g_u_red,
            col_of,
            grad_gather,
            jac_irow_outer,
            jac_jcol_outer,
            jac_gather,
            h_irow_outer,
            h_jcol_outer,
            h_gather,
            jac_irow_inner: jac_irow,
            jac_jcol_inner: jac_jcol,
            nonlinear_vars: None,
            scratch_x: vec![0.0; n],
            scratch_g: vec![0.0; m],
            scratch_grad: vec![0.0; n],
            scratch_jac: vec![0.0; info_inner.nnz_jac_g.max(0) as usize],
            scratch_h: vec![0.0; info_inner.nnz_h_lag.max(0) as usize],
            scratch_lambda: vec![0.0; m],
        });
    }
}

fn decode(irow: Index, jcol: Index, one_based: bool) -> (usize, usize) {
    if one_based {
        (
            (irow as isize - 1).max(0) as usize,
            (jcol as isize - 1).max(0) as usize,
        )
    } else {
        (irow.max(0) as usize, jcol.max(0) as usize)
    }
}

/// Hand one reduced bound multiplier to the full-space column whose own
/// declared bound produced that reduced bound (issue #493).
///
/// `src` is the plan's recorded provenance for the bound, `survivor` the
/// full-space index of the surviving column the IPM reported `z` against,
/// and `upper` says which side of the *reduced* box `z` belongs to.
///
/// With `x_src = α·x_survivor + β`, the reduced bound is the origin's bound
/// pulled back through that map, so the multiplier scales by `1/|α|`, and a
/// negative `α` sends it to the opposite side of the origin's box. That
/// keeps `Σ over the cluster of α_e·(−z_l[e] + z_u[e])` equal to what the
/// survivor alone used to contribute, which is exactly what full-space
/// stationarity at the survivor needs.
///
/// Anything the plan cannot vouch for — a provenance that is not an affine
/// image of this survivor, a non-finite or zero coefficient — leaves the
/// multiplier where the solver put it. A wrong attribution would be worse
/// than the old one.
fn attribute_bound_multiplier(
    plan: &EliminationPlan,
    src: usize,
    survivor: usize,
    z: Number,
    upper: bool,
    z_l_full: &mut [Number],
    z_u_full: &mut [Number],
) {
    let keep_here = |z_l_full: &mut [Number], z_u_full: &mut [Number]| {
        if upper {
            z_u_full[survivor] += z;
        } else {
            z_l_full[survivor] += z;
        }
    };
    if src == survivor || src >= plan.n_full || z == 0.0 {
        keep_here(z_l_full, z_u_full);
        return;
    }
    let VarRecovery::Affine { rep, coeff, .. } = plan.recovery[src] else {
        keep_here(z_l_full, z_u_full);
        return;
    };
    if rep != survivor || !coeff.is_finite() || coeff == 0.0 {
        keep_here(z_l_full, z_u_full);
        return;
    }
    // `upper XOR (α < 0)`: the side flips exactly when the map reverses the
    // interval.
    if upper != (coeff < 0.0) {
        z_u_full[src] += z / coeff.abs();
    } else {
        z_l_full[src] += z / coeff.abs();
    }
}

/// Recover a multiplier for every row the plan consumed.
///
/// See the module docs for why a reverse sweep over the elimination steps
/// is exact and triangular. `lambda_full` arrives with the kept rows filled
/// in and zeros everywhere else, and leaves with the consumed rows filled
/// in too. Rows dropped as `0 = 0` keep their zero.
///
/// Signs follow the convention `finalize_solution` uses, `∇f + Jᵀλ − z_l +
/// z_u = 0`, so `resid` accumulates `+Jᵀλ` and each pivot solve carries the
/// matching negation.
///
/// `z_l_full` / `z_u_full` are the full-space bound multipliers **after**
/// attribution (gh#493). They belong in the residual the sweep cancels: an
/// eliminated column that carries an attributed multiplier needs a
/// correspondingly smaller row multiplier for stationarity to close there.
fn recover_dropped_multipliers(
    plan: &EliminationPlan,
    grad_f: &[Number],
    z_l_full: &[Number],
    z_u_full: &[Number],
    jac_irow: &[Index],
    jac_jcol: &[Index],
    jac_values: &[Number],
    one_based: bool,
    lambda_full: &mut [Number],
) {
    if plan.steps.is_empty() {
        return;
    }
    let n = plan.n_full;
    let m = plan.m_full;

    // Row-wise view of J, so each reverse step can update the residual over
    // just the row it resolved.
    let mut row_entries: Vec<Vec<(usize, Number)>> = vec![Vec::new(); m];
    let mut resid = grad_f.to_vec();
    for (j, r) in resid.iter_mut().enumerate() {
        *r += z_u_full.get(j).copied().unwrap_or(0.0) - z_l_full.get(j).copied().unwrap_or(0.0);
    }
    for k in 0..jac_irow.len() {
        let (i, j) = decode(jac_irow[k], jac_jcol[k], one_based);
        if i >= m || j >= n {
            continue;
        }
        resid[j] += jac_values[k] * lambda_full[i];
        if !plan.row_kept[i] {
            row_entries[i].push((j, jac_values[k]));
        }
    }

    // Q[v] = Σ over v's subtree of (coefficient to v) · resid, built bottom
    // up. A node's children are always eliminated before it is, so walking
    // the steps forward completes every subtree in order.
    let mut q = resid.clone();
    for step in &plan.steps {
        if let Some((parent, alpha)) = plan.parent[step.var] {
            q[parent] += alpha * q[step.var];
        }
    }

    for step in plan.steps.iter().rev() {
        let lam = -q[step.var] / step.pivot;
        if !lam.is_finite() {
            continue;
        }
        lambda_full[step.row] = lam;
        // Fold the newly-known multiplier into the residual, and carry the
        // change up every affected ancestor chain so the remaining subtree
        // sums stay exact.
        for &(j, a) in &row_entries[step.row] {
            let delta = lam * a;
            resid[j] += delta;
            let mut cur = j;
            let mut coeff = delta;
            q[cur] += coeff;
            while let Some((parent, alpha)) = plan.parent[cur] {
                coeff *= alpha;
                q[parent] += coeff;
                cur = parent;
            }
        }
    }
}

// Every `.expect("inited")` below is guarded by the preceding
// `ensure_init`, which is the only way `state` becomes `Some`.
#[allow(clippy::expect_used)]
impl TNLP for LinearEqElimTnlp {
    fn is_presolve_wrapper(&self) -> bool {
        true
    }

    fn presolve_infeasibility_proof(&self) -> Option<InfeasibilityProof> {
        self.inner.borrow().presolve_infeasibility_proof()
    }

    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let s = self.ensure_init()?;
        Some(s.info_outer)
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let Some(s) = self.ensure_init() else {
            return false;
        };
        if s.passthrough {
            return self.inner.borrow_mut().get_bounds_info(b);
        }
        b.x_l.copy_from_slice(&s.plan.x_l_red);
        b.x_u.copy_from_slice(&s.plan.x_u_red);
        b.g_l.copy_from_slice(&s.g_l_red);
        b.g_u.copy_from_slice(&s.g_u_red);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let Some(s) = self.ensure_init() else {
            return false;
        };
        if s.passthrough {
            return self.inner.borrow_mut().get_starting_point(sp);
        }
        let (n_in, m_in) = {
            let s = self.state.as_ref().expect("inited");
            (s.plan.n_full, s.plan.m_full)
        };
        let mut x_full = vec![0.0; n_in];
        let mut z_l_full = vec![0.0; n_in];
        let mut z_u_full = vec![0.0; n_in];
        let mut lambda_full = vec![0.0; m_in];
        if !self.inner.borrow_mut().get_starting_point(StartingPoint {
            init_x: sp.init_x,
            x: &mut x_full,
            init_z: sp.init_z,
            z_l: &mut z_l_full,
            z_u: &mut z_u_full,
            init_lambda: sp.init_lambda,
            lambda: &mut lambda_full,
        }) {
            return false;
        }
        let s = self.state.as_ref().expect("inited");
        for (red, &full) in s.plan.vars_kept.iter().enumerate() {
            sp.x[red] = x_full[full];
            sp.z_l[red] = z_l_full[full];
            sp.z_u[red] = z_u_full[full];
        }
        for (red, &full) in s.plan.rows_kept.iter().enumerate() {
            sp.lambda[red] = lambda_full[full];
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        if self.ensure_init()?.passthrough {
            return self.inner.borrow_mut().eval_f(x, new_x);
        }
        let s = self.state.as_mut().expect("inited");
        s.plan.lift_x(x, &mut s.scratch_x);
        self.inner.borrow_mut().eval_f(&s.scratch_x, new_x)
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        match self.ensure_init() {
            None => return false,
            Some(s) if s.passthrough => {
                return self.inner.borrow_mut().eval_grad_f(x, new_x, grad_f);
            }
            Some(_) => {}
        }
        let s = self.state.as_mut().expect("inited");
        s.plan.lift_x(x, &mut s.scratch_x);
        if !self
            .inner
            .borrow_mut()
            .eval_grad_f(&s.scratch_x, new_x, &mut s.scratch_grad)
        {
            return false;
        }
        s.grad_gather.apply(&s.scratch_grad, grad_f);
        true
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        match self.ensure_init() {
            None => return false,
            Some(s) if s.passthrough => return self.inner.borrow_mut().eval_g(x, new_x, g),
            Some(_) => {}
        }
        let s = self.state.as_mut().expect("inited");
        s.plan.lift_x(x, &mut s.scratch_x);
        if !self
            .inner
            .borrow_mut()
            .eval_g(&s.scratch_x, new_x, &mut s.scratch_g)
        {
            return false;
        }
        for (red, &full) in s.plan.rows_kept.iter().enumerate() {
            g[red] = s.scratch_g[full];
        }
        true
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        match self.ensure_init() {
            None => return false,
            Some(s) if s.passthrough => {
                return self.inner.borrow_mut().eval_jac_g(x, new_x, mode);
            }
            Some(_) => {}
        }
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let s = self.state.as_ref().expect("inited");
                irow.copy_from_slice(&s.jac_irow_outer);
                jcol.copy_from_slice(&s.jac_jcol_outer);
                true
            }
            SparsityRequest::Values { values } => {
                let s = self.state.as_mut().expect("inited");
                let x_full = match x {
                    Some(xr) => {
                        s.plan.lift_x(xr, &mut s.scratch_x);
                        Some(&s.scratch_x[..])
                    }
                    None => None,
                };
                if !self.inner.borrow_mut().eval_jac_g(
                    x_full,
                    new_x,
                    SparsityRequest::Values {
                        values: &mut s.scratch_jac,
                    },
                ) {
                    return false;
                }
                s.jac_gather.apply(&s.scratch_jac, values);
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match self.ensure_init() {
            None => return false,
            Some(s) if s.passthrough => {
                return self
                    .inner
                    .borrow_mut()
                    .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode);
            }
            Some(_) => {}
        }
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let s = self.state.as_ref().expect("inited");
                irow.copy_from_slice(&s.h_irow_outer);
                jcol.copy_from_slice(&s.h_jcol_outer);
                true
            }
            SparsityRequest::Values { values } => {
                let s = self.state.as_mut().expect("inited");
                let x_full = match x {
                    Some(xr) => {
                        s.plan.lift_x(xr, &mut s.scratch_x);
                        Some(&s.scratch_x[..])
                    }
                    None => None,
                };
                // Consumed rows are satisfied identically by the
                // substitution, so they carry no curvature: λ = 0 there.
                let lam_full = match lambda {
                    Some(lam) => {
                        for v in s.scratch_lambda.iter_mut() {
                            *v = 0.0;
                        }
                        for (red, &full) in s.plan.rows_kept.iter().enumerate() {
                            s.scratch_lambda[full] = lam[red];
                        }
                        Some(&s.scratch_lambda[..])
                    }
                    None => None,
                };
                if !self.inner.borrow_mut().eval_h(
                    x_full,
                    new_x,
                    obj_factor,
                    lam_full,
                    new_lambda,
                    SparsityRequest::Values {
                        values: &mut s.scratch_h,
                    },
                ) {
                    return false;
                }
                s.h_gather.apply(&s.scratch_h, values);
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        if self.ensure_init().is_none_or(|s| s.passthrough) {
            self.inner
                .borrow_mut()
                .finalize_solution(sol, ip_data, ip_cq);
            return;
        }
        let (n_in, m_in, nnz_in, one_based) = {
            let s = self.state.as_ref().expect("inited");
            (
                s.plan.n_full,
                s.plan.m_full,
                s.info_inner.nnz_jac_g.max(0) as usize,
                matches!(s.info_inner.index_style, IndexStyle::Fortran),
            )
        };

        let mut x_full = vec![0.0; n_in];
        {
            let s = self.state.as_ref().expect("inited");
            s.plan.lift_x(sol.x, &mut x_full);
        }

        // Bound multipliers: each one routed to the column whose declared
        // bound the reduced bound came from, which is the survivor itself
        // unless a transfer moved that bound there (gh#493). This has to
        // happen before the dual sweep below, which cancels the residual
        // these leave behind.
        let mut z_l_full = vec![0.0; n_in];
        let mut z_u_full = vec![0.0; n_in];
        {
            let s = self.state.as_ref().expect("inited");
            for (red, &full) in s.plan.vars_kept.iter().enumerate() {
                if red < sol.z_l.len() {
                    attribute_bound_multiplier(
                        &s.plan,
                        s.plan.x_l_src.get(red).copied().unwrap_or(full),
                        full,
                        sol.z_l[red],
                        false,
                        &mut z_l_full,
                        &mut z_u_full,
                    );
                }
                if red < sol.z_u.len() {
                    attribute_bound_multiplier(
                        &s.plan,
                        s.plan.x_u_src.get(red).copied().unwrap_or(full),
                        full,
                        sol.z_u[red],
                        true,
                        &mut z_l_full,
                        &mut z_u_full,
                    );
                }
            }
        }

        // Constraint values: re-evaluate so the dropped rows carry their
        // real residual rather than a zero. If that fails, splice the
        // solver's own reduced `g` into the kept rows and leave the rest at
        // zero — never forward a partially-written scratch buffer.
        let mut g_full = vec![0.0; m_in];
        {
            let ok = m_in == 0 || self.inner.borrow_mut().eval_g(&x_full, true, &mut g_full);
            let s = self.state.as_ref().expect("inited");
            if !ok {
                for v in g_full.iter_mut() {
                    *v = 0.0;
                }
                for (red, &full) in s.plan.rows_kept.iter().enumerate() {
                    if red < sol.g.len() {
                        g_full[full] = sol.g[red];
                    }
                }
            }
        }

        let mut lambda_full = vec![0.0; m_in];
        {
            let s = self.state.as_ref().expect("inited");
            for (red, &full) in s.plan.rows_kept.iter().enumerate() {
                if red < sol.lambda.len() {
                    lambda_full[full] = sol.lambda[red];
                }
            }
        }

        let has_steps = !self.state.as_ref().expect("inited").plan.steps.is_empty();
        if has_steps {
            let mut grad_f = vec![0.0; n_in];
            let ok_grad = self
                .inner
                .borrow_mut()
                .eval_grad_f(&x_full, true, &mut grad_f);
            let mut jac_values = vec![0.0; nnz_in];
            let ok_jac = nnz_in == 0
                || self.inner.borrow_mut().eval_jac_g(
                    Some(&x_full),
                    false,
                    SparsityRequest::Values {
                        values: &mut jac_values,
                    },
                );
            if ok_grad && ok_jac {
                let s = self.state.as_ref().expect("inited");
                recover_dropped_multipliers(
                    &s.plan,
                    &grad_f,
                    &z_l_full,
                    &z_u_full,
                    &s.jac_irow_inner,
                    &s.jac_jcol_inner,
                    &jac_values,
                    one_based,
                    &mut lambda_full,
                );
            } else {
                tracing::warn!(
                    target: "pounce::presolve",
                    "could not re-evaluate the model at the solution; the rows \
                     consumed by linear-equality elimination are reported with a \
                     zero multiplier."
                );
            }
        }

        self.finalized = Some(FullSolution {
            x: x_full.clone(),
            lambda: lambda_full.clone(),
            z_l: z_l_full.clone(),
            z_u: z_u_full.clone(),
        });
        self.inner.borrow_mut().finalize_solution(
            Solution {
                status: sol.status,
                x: &x_full,
                z_l: &z_l_full,
                z_u: &z_u_full,
                g: &g_full,
                lambda: &lambda_full,
                obj_value: sol.obj_value,
            },
            ip_data,
            ip_cq,
        );
    }

    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        match self.ensure_init() {
            None => return false,
            Some(s) if s.passthrough => {
                return self.inner.borrow_mut().get_var_con_metadata(var, con);
            }
            Some(_) => {}
        }
        let mut inner_var = MetaData::default();
        let mut inner_con = MetaData::default();
        if !self
            .inner
            .borrow_mut()
            .get_var_con_metadata(&mut inner_var, &mut inner_con)
        {
            return false;
        }
        let s = self.state.as_ref().expect("inited");
        *var = project_metadata(&inner_var, &s.plan.vars_kept, s.plan.n_full);
        *con = project_metadata(&inner_con, &s.plan.rows_kept, s.plan.m_full);
        true
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        match self.ensure_init() {
            None => return false,
            Some(s) if s.passthrough => {
                return self.inner.borrow_mut().get_scaling_parameters(req);
            }
            Some(_) => {}
        }
        let (n_in, m_in) = {
            let s = self.state.as_ref().expect("inited");
            (s.plan.n_full, s.plan.m_full)
        };
        let mut inner_x = vec![1.0; n_in];
        let mut inner_g = vec![1.0; m_in];
        let mut obj_scaling = 1.0;
        let mut use_x = false;
        let mut use_g = false;
        if !self
            .inner
            .borrow_mut()
            .get_scaling_parameters(ScalingRequest {
                obj_scaling: &mut obj_scaling,
                use_x_scaling: &mut use_x,
                x_scaling: &mut inner_x,
                use_g_scaling: &mut use_g,
                g_scaling: &mut inner_g,
            })
        {
            return false;
        }
        *req.obj_scaling = obj_scaling;
        *req.use_x_scaling = use_x;
        *req.use_g_scaling = use_g;
        let s = self.state.as_ref().expect("inited");
        // A reduced column *is* its surviving variable, so it keeps that
        // variable's declared scale factor.
        for (red, &full) in s.plan.vars_kept.iter().enumerate() {
            if red < req.x_scaling.len() {
                req.x_scaling[red] = inner_x[full];
            }
        }
        for (red, &full) in s.plan.rows_kept.iter().enumerate() {
            if red < req.g_scaling.len() {
                req.g_scaling[red] = inner_g[full];
            }
        }
        true
    }

    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.project_var_linearity(types, false)
    }

    fn get_objective_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.project_var_linearity(types, true)
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        match self.ensure_init() {
            None => return false,
            Some(s) if s.passthrough => {
                return self.inner.borrow_mut().get_constraints_linearity(types);
            }
            Some(_) => {}
        }
        let m_in = self.state.as_ref().expect("inited").plan.m_full;
        let mut full = vec![Linearity::NonLinear; m_in];
        if !self.inner.borrow_mut().get_constraints_linearity(&mut full) {
            return false;
        }
        let s = self.state.as_ref().expect("inited");
        for (red, &r) in s.plan.rows_kept.iter().enumerate() {
            types[red] = full[r];
        }
        true
    }

    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        match self.reduced_nonlinear_vars() {
            Some(list) => list.len() as Index,
            None => -1,
        }
    }

    fn get_list_of_nonlinear_variables(&mut self, pos_nonlin_vars: &mut [Index]) -> bool {
        let Some(list) = self.reduced_nonlinear_vars() else {
            return false;
        };
        if pos_nonlin_vars.len() < list.len() {
            return false;
        }
        pos_nonlin_vars[..list.len()].copy_from_slice(&list);
        true
    }

    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) -> bool {
        self.inner
            .borrow_mut()
            .intermediate_callback(stats, ip_data, ip_cq)
    }

    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        if self.ensure_init().is_none_or(|s| s.passthrough) {
            self.inner.borrow_mut().finalize_metadata(var, con);
            return;
        }
        let s = self.state.as_ref().expect("inited");
        let var_full = expand_metadata(var, &s.plan.vars_kept, s.plan.n_full);
        let con_full = expand_metadata(con, &s.plan.rows_kept, s.plan.m_full);
        self.inner
            .borrow_mut()
            .finalize_metadata(&var_full, &con_full);
    }
}

#[allow(clippy::expect_used)]
impl LinearEqElimTnlp {
    /// A reduced column is non-linear when **any** full column folded onto
    /// it is: `y` enters through `α·y + β`, so one non-linear member is
    /// enough to make the objective non-linear in `y`.
    fn project_var_linearity(&mut self, types: &mut [Linearity], objective_scoped: bool) -> bool {
        match self.ensure_init() {
            None => return false,
            Some(s) if s.passthrough => {
                let mut inner = self.inner.borrow_mut();
                return if objective_scoped {
                    inner.get_objective_variables_linearity(types)
                } else {
                    inner.get_variables_linearity(types)
                };
            }
            Some(_) => {}
        }
        let n_in = self.state.as_ref().expect("inited").plan.n_full;
        let mut full = vec![Linearity::NonLinear; n_in];
        let ok = {
            let mut inner = self.inner.borrow_mut();
            if objective_scoped {
                inner.get_objective_variables_linearity(&mut full)
            } else {
                inner.get_variables_linearity(&mut full)
            }
        };
        if !ok {
            return false;
        }
        let s = self.state.as_ref().expect("inited");
        for t in types.iter_mut() {
            *t = Linearity::Linear;
        }
        for (i, slot) in s.col_of.iter().enumerate() {
            if let Some((red, _)) = *slot
                && matches!(full[i], Linearity::NonLinear)
                && red < types.len()
            {
                types[red] = Linearity::NonLinear;
            }
        }
        true
    }

    /// The inner nonlinear-variable list mapped onto reduced columns, or
    /// `None` when the inner TNLP declines to publish one.
    fn reduced_nonlinear_vars(&mut self) -> Option<Vec<Index>> {
        if self.ensure_init()?.passthrough {
            let count = self.inner.borrow_mut().get_number_of_nonlinear_variables();
            if count < 0 {
                return None;
            }
            let mut list = vec![0 as Index; count as usize];
            if !self
                .inner
                .borrow_mut()
                .get_list_of_nonlinear_variables(&mut list)
            {
                return None;
            }
            return Some(list);
        }
        if let Some(cached) = self.state.as_ref().expect("inited").nonlinear_vars.as_ref() {
            return Some(cached.clone());
        }
        let count = self.inner.borrow_mut().get_number_of_nonlinear_variables();
        if count < 0 {
            return None;
        }
        let mut inner_list = vec![0 as Index; count as usize];
        if !self
            .inner
            .borrow_mut()
            .get_list_of_nonlinear_variables(&mut inner_list)
        {
            return None;
        }
        let s = self.state.as_mut().expect("inited");
        let one_based = matches!(s.info_inner.index_style, IndexStyle::Fortran);
        let mut seen = vec![false; s.plan.n_reduced_vars()];
        let mut out: Vec<Index> = Vec::new();
        for &v in &inner_list {
            let full = if one_based {
                (v as isize - 1).max(0) as usize
            } else {
                v.max(0) as usize
            };
            if full >= s.col_of.len() {
                continue;
            }
            if let Some((red, _)) = s.col_of[full]
                && !seen[red]
            {
                seen[red] = true;
                out.push(if one_based {
                    red as Index + 1
                } else {
                    red as Index
                });
            }
        }
        out.sort_unstable();
        s.nonlinear_vars = Some(out.clone());
        Some(out)
    }
}

/// Subset every per-index vector of `meta` to `kept`.
///
/// Per-index-ness is inferred from the vector's length, exactly as
/// `PresolveTnlp` does for its row metadata: `MetaData` mirrors upstream's
/// untyped map and carries no shape information of its own.
fn project_metadata(meta: &MetaData, kept: &[usize], n_full: usize) -> MetaData {
    let mut out = MetaData::default();
    for (k, v) in &meta.strings {
        out.strings.insert(
            k.clone(),
            if v.len() == n_full {
                kept.iter().map(|&i| v[i].clone()).collect()
            } else {
                v.clone()
            },
        );
    }
    for (k, v) in &meta.integers {
        out.integers.insert(
            k.clone(),
            if v.len() == n_full {
                kept.iter().map(|&i| v[i]).collect()
            } else {
                v.clone()
            },
        );
    }
    for (k, v) in &meta.numerics {
        out.numerics.insert(
            k.clone(),
            if v.len() == n_full {
                kept.iter().map(|&i| v[i]).collect()
            } else {
                v.clone()
            },
        );
    }
    out
}

/// Inverse of [`project_metadata`]: scatter a reduced-length vector back
/// into full-length, leaving defaults at the removed indices.
fn expand_metadata(meta: &MetaData, kept: &[usize], n_full: usize) -> MetaData {
    let n_red = kept.len();
    let mut out = MetaData::default();
    for (k, v) in &meta.strings {
        if v.len() == n_red && n_red != n_full {
            let mut full = vec![String::new(); n_full];
            for (r, &i) in kept.iter().enumerate() {
                full[i] = v[r].clone();
            }
            out.strings.insert(k.clone(), full);
        } else {
            out.strings.insert(k.clone(), v.clone());
        }
    }
    for (k, v) in &meta.integers {
        if v.len() == n_red && n_red != n_full {
            let mut full = vec![0 as Index; n_full];
            for (r, &i) in kept.iter().enumerate() {
                full[i] = v[r];
            }
            out.integers.insert(k.clone(), full);
        } else {
            out.integers.insert(k.clone(), v.clone());
        }
    }
    for (k, v) in &meta.numerics {
        if v.len() == n_red && n_red != n_full {
            let mut full = vec![0.0; n_full];
            for (r, &i) in kept.iter().enumerate() {
                full[i] = v[r];
            }
            out.numerics.insert(k.clone(), full);
        } else {
            out.numerics.insert(k.clone(), v.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_eq_plan::ElimStep;

    fn plan_with(
        steps: Vec<ElimStep>,
        parent: Vec<Option<(usize, Number)>>,
        m: usize,
    ) -> EliminationPlan {
        let n = parent.len();
        let mut row_kept = vec![true; m];
        for s in &steps {
            row_kept[s.row] = false;
        }
        EliminationPlan {
            n_full: n,
            m_full: m,
            recovery: vec![VarRecovery::Kept(0); n],
            vars_kept: Vec::new(),
            rows_kept: (0..m).filter(|&r| row_kept[r]).collect(),
            row_kept,
            x_l_red: Vec::new(),
            x_u_red: Vec::new(),
            x_l_src: Vec::new(),
            x_u_src: Vec::new(),
            steps,
            parent,
            report: LinearEqElimReport::default(),
        }
    }

    /// `min 4·x  s.t.  x = 3` (row 0 consumed). Stationarity in the
    /// `∇f + Jᵀλ = 0` convention gives λ = −4.
    #[test]
    fn recovers_a_singleton_rows_multiplier() {
        let plan = plan_with(
            vec![ElimStep {
                row: 0,
                var: 0,
                pivot: 1.0,
            }],
            vec![None],
            1,
        );
        let mut lambda = vec![0.0];
        recover_dropped_multipliers(
            &plan,
            &[4.0],
            &[0.0],
            &[0.0],
            &[0],
            &[0],
            &[1.0],
            false,
            &mut lambda,
        );
        assert!((lambda[0] + 4.0).abs() < 1e-12, "{lambda:?}");
    }

    /// Two chained aggregations. Row 0: `x0 - x1 = 0` eliminates x0.
    /// Row 1: `x1 - x2 = 0` eliminates x1. Row 2 (kept) touches x2.
    /// f = 2·x0 + 3·x1 + 5·x2.
    ///
    /// Stationarity at x0: 2 + λ0 = 0            → λ0 = −2
    /// Stationarity at x1: 3 − λ0 + λ1 = 0        → λ1 = −5
    /// (row 2 does not touch x0 or x1)
    #[test]
    fn reverse_sweep_resolves_a_chain() {
        let plan = plan_with(
            vec![
                ElimStep {
                    row: 0,
                    var: 0,
                    pivot: 1.0,
                },
                ElimStep {
                    row: 1,
                    var: 1,
                    pivot: 1.0,
                },
            ],
            vec![Some((1, 1.0)), Some((2, 1.0)), None],
            3,
        );
        // J rows: r0 = [1, -1, 0], r1 = [0, 1, -1], r2 = [0, 0, 1].
        let irow = [0, 0, 1, 1, 2];
        let jcol = [0, 1, 1, 2, 2];
        let vals = [1.0, -1.0, 1.0, -1.0, 1.0];
        let mut lambda = vec![0.0, 0.0, 0.0];
        recover_dropped_multipliers(
            &plan,
            &[2.0, 3.0, 5.0],
            &[0.0; 3],
            &[0.0; 3],
            &irow,
            &jcol,
            &vals,
            false,
            &mut lambda,
        );
        assert!((lambda[0] + 2.0).abs() < 1e-12, "{lambda:?}");
        assert!((lambda[1] + 5.0).abs() < 1e-12, "{lambda:?}");
    }

    /// The recovery must reproduce full-space stationarity, which is the
    /// property that actually matters: `∇f + Jᵀλ = 0` at every eliminated
    /// column.
    #[test]
    fn recovered_multipliers_close_full_space_stationarity() {
        // x0 = 2·x1 + 1 (row 0), x1 = x2 (row 1), row 2 kept with λ = 0.5.
        // J: r0 = [1, -2, 0], r1 = [0, 1, -1], r2 = [0.3, 0.7, 1.1].
        let plan = plan_with(
            vec![
                ElimStep {
                    row: 0,
                    var: 0,
                    pivot: 1.0,
                },
                ElimStep {
                    row: 1,
                    var: 1,
                    pivot: 1.0,
                },
            ],
            vec![Some((1, 2.0)), Some((2, 1.0)), None],
            3,
        );
        let irow = [0, 0, 1, 1, 2, 2, 2];
        let jcol = [0, 1, 1, 2, 0, 1, 2];
        let vals = [1.0, -2.0, 1.0, -1.0, 0.3, 0.7, 1.1];
        let grad = [2.0, 3.0, 5.0];
        let mut lambda = vec![0.0, 0.0, 0.5];
        recover_dropped_multipliers(
            &plan,
            &grad,
            &[0.0; 3],
            &[0.0; 3],
            &irow,
            &jcol,
            &vals,
            false,
            &mut lambda,
        );
        for col in [0usize, 1] {
            let mut resid = grad[col];
            for k in 0..irow.len() {
                if jcol[k] as usize == col {
                    resid += vals[k] * lambda[irow[k] as usize];
                }
            }
            assert!(
                resid.abs() < 1e-10,
                "stationarity at eliminated column {col} = {resid}, λ = {lambda:?}"
            );
        }
    }

    /// `x0 = α·x1 + β` with x0's bound transferred onto x1: the multiplier
    /// goes back to x0, scaled by `1/|α|`, and changes sides when α < 0
    /// (gh#493).
    fn plan_with_transferred_bound(coeff: Number) -> EliminationPlan {
        let mut plan = plan_with(
            vec![ElimStep {
                row: 0,
                var: 0,
                pivot: 1.0,
            }],
            vec![Some((1, coeff)), None],
            1,
        );
        plan.recovery = vec![
            VarRecovery::Affine {
                rep: 1,
                coeff,
                offset: 0.0,
            },
            VarRecovery::Kept(0),
        ];
        plan.vars_kept = vec![1];
        // Both of x1's reduced bounds were born on x0.
        plan.x_l_src = vec![0];
        plan.x_u_src = vec![0];
        plan
    }

    #[test]
    fn a_positive_coefficient_keeps_the_multiplier_on_the_same_side() {
        let plan = plan_with_transferred_bound(2.0);
        let (mut z_l, mut z_u) = (vec![0.0; 2], vec![0.0; 2]);
        attribute_bound_multiplier(&plan, 0, 1, 12.0, true, &mut z_l, &mut z_u);
        assert_eq!(z_u, vec![6.0, 0.0], "z_u = 12/|α| on x0");
        assert_eq!(z_l, vec![0.0, 0.0]);
    }

    #[test]
    fn a_negative_coefficient_flips_the_side_the_multiplier_lands_on() {
        let plan = plan_with_transferred_bound(-2.0);
        let (mut z_l, mut z_u) = (vec![0.0; 2], vec![0.0; 2]);
        // The survivor's *lower* bound is x0's *upper* bound when α < 0.
        attribute_bound_multiplier(&plan, 0, 1, 12.0, false, &mut z_l, &mut z_u);
        assert_eq!(z_u, vec![6.0, 0.0]);
        assert_eq!(z_l, vec![0.0, 0.0]);
    }

    #[test]
    fn a_survivors_own_bound_keeps_its_multiplier() {
        let plan = plan_with_transferred_bound(2.0);
        let (mut z_l, mut z_u) = (vec![0.0; 2], vec![0.0; 2]);
        attribute_bound_multiplier(&plan, 1, 1, 7.0, false, &mut z_l, &mut z_u);
        assert_eq!(z_l, vec![0.0, 7.0]);
    }

    /// A provenance the plan cannot vouch for must not move the multiplier:
    /// a wrong attribution is worse than the old one.
    #[test]
    fn an_unvouched_provenance_leaves_the_multiplier_where_it_was() {
        let mut plan = plan_with_transferred_bound(2.0);
        plan.recovery[0] = VarRecovery::Constant(3.0);
        let (mut z_l, mut z_u) = (vec![0.0; 2], vec![0.0; 2]);
        attribute_bound_multiplier(&plan, 0, 1, 5.0, true, &mut z_l, &mut z_u);
        assert_eq!(z_u, vec![0.0, 5.0]);

        // Out of range, and pointing at a different survivor, likewise.
        let mut plan = plan_with_transferred_bound(2.0);
        plan.recovery[0] = VarRecovery::Affine {
            rep: 9,
            coeff: 2.0,
            offset: 0.0,
        };
        let (mut z_l, mut z_u) = (vec![0.0; 2], vec![0.0; 2]);
        attribute_bound_multiplier(&plan, 0, 1, 5.0, true, &mut z_l, &mut z_u);
        attribute_bound_multiplier(&plan, 99, 1, 4.0, true, &mut z_l, &mut z_u);
        assert_eq!(z_u, vec![0.0, 9.0]);
    }

    /// The sweep has to see the attributed multipliers, or the row
    /// multiplier it recovers will double-count them.
    ///
    /// `min 4·x0  s.t.  x0 − 2·x1 = 0`, row 0 consumed to eliminate x0.
    /// Stationarity at x0 is `∇f + λ·1 − z_l + z_u = 0`, so with nothing
    /// attributed there the sweep must return `λ = −4`, and with `z_u = 4`
    /// attributed to x0 it must return `λ = −8` instead.
    #[test]
    fn the_sweep_absorbs_an_attributed_bound_multiplier() {
        let plan = plan_with_transferred_bound(2.0);
        let irow = [0, 0];
        let jcol = [0, 1];
        let vals = [1.0, -2.0];

        let mut lambda = vec![0.0];
        recover_dropped_multipliers(
            &plan,
            &[4.0, 0.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &irow,
            &jcol,
            &vals,
            false,
            &mut lambda,
        );
        assert!((lambda[0] + 4.0).abs() < 1e-12, "{lambda:?}");

        let mut lambda = vec![0.0];
        recover_dropped_multipliers(
            &plan,
            &[4.0, 0.0],
            &[0.0, 0.0],
            &[4.0, 0.0],
            &irow,
            &jcol,
            &vals,
            false,
            &mut lambda,
        );
        assert!((lambda[0] + 8.0).abs() < 1e-12, "{lambda:?}");
    }

    #[test]
    fn one_based_structure_is_decoded() {
        assert_eq!(decode(1, 1, true), (0, 0));
        assert_eq!(decode(0, 0, false), (0, 0));
        assert_eq!(decode(3, 5, true), (2, 4));
    }

    #[test]
    fn gather_sums_scaled_sources() {
        let (slots, g) = gather_from(vec![((0, 0), 2, 1.0), ((1, 0), 0, 3.0), ((0, 0), 1, -2.0)]);
        assert_eq!(slots, vec![(0, 0), (1, 0)]);
        let mut out = vec![0.0; 2];
        g.apply(&[7.0, 5.0, 11.0], &mut out);
        assert_eq!(out, vec![11.0 - 10.0, 21.0]);
    }

    #[test]
    fn metadata_round_trips_through_the_reduction() {
        let mut meta = MetaData::default();
        meta.strings
            .insert("idx_names".into(), vec!["a".into(), "b".into(), "c".into()]);
        meta.numerics.insert("weights".into(), vec![1.0, 2.0, 3.0]);
        // A vector that is not per-index must pass through untouched.
        meta.integers.insert("scalarish".into(), vec![9]);
        let kept = [0usize, 2];
        let projected = project_metadata(&meta, &kept, 3);
        assert_eq!(projected.strings["idx_names"], vec!["a", "c"]);
        assert_eq!(projected.numerics["weights"], vec![1.0, 3.0]);
        assert_eq!(projected.integers["scalarish"], vec![9]);
        let expanded = expand_metadata(&projected, &kept, 3);
        assert_eq!(expanded.strings["idx_names"], vec!["a", "", "c"]);
        assert_eq!(expanded.numerics["weights"], vec![1.0, 0.0, 3.0]);
        assert_eq!(expanded.integers["scalarish"], vec![9]);
    }
}
