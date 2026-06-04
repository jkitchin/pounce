//! Interactive solver debugger — a "pdb for the interior-point loop".
//!
//! The main loop ([`crate::ipopt_alg::IpoptAlgorithm::optimize`]) fires
//! a [`DebugHook`] at well-defined checkpoints. A hook receives a
//! [`DebugCtx`] — a live, *mutable* view of the algorithm state — and
//! returns a [`DebugAction`] telling the loop whether to keep solving
//! or stop. This is the engine; the user-facing REPL / agent protocol
//! lives in the CLI (`pounce --debug`), which implements [`DebugHook`].
//!
//! Two design points make mutation safe:
//!
//!   * [`DebugCtx`] holds cheap `Rc` clones of the same `IpoptData` /
//!     `IpoptCq` handles the loop uses, so reads and writes go through
//!     the identical `RefCell` path — there is no shadow copy to drift.
//!   * Overwriting the iterate rebuilds a *fresh* [`IteratesVector`]
//!     (via `deep_copy().freeze()`), which mints a new vector tag. The
//!     CQ caches are tag-keyed (see `ipopt_cq.rs`), so a mutated iterate
//!     transparently invalidates every derived quantity — exactly as if
//!     the line search had produced the new point.
//!
//! Checkpoints fire at the iteration top, the sub-iteration phases
//! (`after_mu` / `after_search_dir` / `after_step`), around restoration
//! entry/exit, and at termination. The same hook is shared
//! (`Rc<RefCell<…>>`) with the restoration inner IPM, so one debugger
//! steps both the outer and inner solves.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::Number;
use pounce_linalg::{Matrix, Vector};
use pounce_nlp::ipopt_nlp::SplitNames;

/// Where in the main loop a checkpoint fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checkpoint {
    /// Top of an outer iteration — after the intermediate callback,
    /// before this iteration's Newton step is computed. The iterate,
    /// multipliers, and μ all reflect the *accepted* point from the
    /// previous iteration.
    IterStart,
    /// After the barrier parameter μ was updated for this iteration
    /// (before the search direction is computed).
    AfterBarrierUpdate,
    /// After the primal-dual Newton step was computed — the search
    /// direction `δ` (`data.delta`), the applied regularization, and the
    /// KKT factorization are available.
    AfterSearchDirection,
    /// After the line search chose a step length and the trial point was
    /// accepted — α (`info_alpha_*`) and the new iterate are in place.
    AfterStep,
    /// The line search *rejected* this iteration's step — it hit the tiny-step
    /// floor or exhausted its backtracks without an acceptable point, and the
    /// solver is about to fall into restoration. The search direction `δ` and
    /// the un-accepted current iterate are intact for inspection. The "why did
    /// the line search give up here?" stop, distinct from the restoration entry
    /// that follows.
    StepRejected,
    /// Just before the algorithm switches into the restoration phase —
    /// the iterate that tripped restoration is intact. The most-requested
    /// "why did this go to restoration?" stop.
    PreRestoration,
    /// Just after the restoration phase returns, so its effect on the
    /// iterate can be inspected.
    PostRestoration,
    /// The solve has finished (or is about to): fired once before
    /// `optimize` returns, at the final iterate, carrying the outcome
    /// via [`DebugCtx::status`]. Lets a debugger drop in for a
    /// post-mortem at the failing (or final) point. The [`DebugAction`]
    /// returned at this checkpoint is **ignored** — the solve is already
    /// over, so there is nothing left to resume or stop.
    Terminated,
}

impl Checkpoint {
    /// The stable wire/CLI protocol name for this checkpoint. These strings
    /// are intentionally **not** the variant identifiers (`AfterBarrierUpdate`
    /// → `"after_mu"`, `PreRestoration` → `"pre_restoration_entry"`) — they're
    /// the names the JSON protocol and `stop-at` use, so match on the variant,
    /// not the string. Locked by the `checkpoint_as_str_is_stable` test.
    pub fn as_str(self) -> &'static str {
        match self {
            Checkpoint::IterStart => "iter_start",
            Checkpoint::AfterBarrierUpdate => "after_mu",
            Checkpoint::AfterSearchDirection => "after_search_dir",
            Checkpoint::AfterStep => "after_step",
            Checkpoint::StepRejected => "step_rejected",
            Checkpoint::PreRestoration => "pre_restoration_entry",
            Checkpoint::PostRestoration => "post_restoration_exit",
            Checkpoint::Terminated => "terminated",
        }
    }

    /// Sub-iteration checkpoints (everything between `IterStart` and the
    /// next `IterStart`).
    pub fn is_sub_iteration(self) -> bool {
        matches!(
            self,
            Checkpoint::AfterBarrierUpdate
                | Checkpoint::AfterSearchDirection
                | Checkpoint::AfterStep
                | Checkpoint::StepRejected
                | Checkpoint::PreRestoration
                | Checkpoint::PostRestoration
        )
    }
}

/// What the algorithm should do after a [`DebugHook`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugAction {
    /// Keep solving.
    Resume,
    /// Stop the solve now. Surfaces to the caller as
    /// `SolverReturn::UserRequestedStop`.
    Stop,
}

/// The eight primal/dual blocks of an iterate, addressable by name.
pub const BLOCK_NAMES: [&str; 8] = ["x", "s", "y_c", "y_d", "z_l", "z_u", "v_l", "v_u"];

/// KKT-factorization report (see [`DebugCtx::kkt`]). The inertia of a
/// well-posed primal-dual system is `(n_pos = n, n_neg = m, n_zero = 0)`;
/// a mismatch (or nonzero regularization) is the classic signal that the
/// step is being stabilized.
#[derive(Clone, Debug)]
pub struct KktReport {
    /// The outer iteration this factorization was assembled at — may be the
    /// previous iteration when paused at `iter_start` (look-back).
    pub iter: i32,
    /// Augmented-system dimension (n + m).
    pub dim: i32,
    /// Negative eigenvalues reported (-1 if the backend has no inertia).
    pub n_neg: i32,
    /// Positive eigenvalues = `dim − n_neg` (-1 if unknown).
    pub n_pos: i32,
    /// Expected negatives = number of equality + inequality multipliers.
    pub expected_neg: i32,
    /// Whether the backend reports inertia.
    pub provides_inertia: bool,
    /// `true` when reported inertia matches the expected `(n, m, 0)`.
    pub inertia_correct: bool,
    /// Primal regularization δ_w applied to the (1,1) block.
    pub delta_w: Number,
    /// Dual regularization δ_c applied to the (3,3)/(4,4) blocks.
    pub delta_c: Number,
    /// Factorization status (debug string).
    pub status: String,
}

/// Which residual space a [`Residual`] entry comes from.
///
/// Primal entries are the per-constraint violations whose max-norm is
/// `inf_pr`; dual entries are the per-variable Lagrangian-gradient
/// components whose max-norm is `inf_du`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidKind {
    /// Equality constraint residual `c_i(x)`.
    Eq,
    /// Inequality residual `d_i(x) − s_i` (the IPM slack reformulation).
    Ineq,
    /// `x`-space stationarity component `(∇_x L)_i`.
    DualX,
    /// `s`-space stationarity component `(∇_s L)_i`.
    DualS,
}

impl ResidKind {
    /// Short label used in the debugger's `print residuals` output and
    /// the JSON `space` field. Stable — readers may match on it.
    pub fn tag(self) -> &'static str {
        match self {
            ResidKind::Eq => "c",
            ResidKind::Ineq => "d-s",
            ResidKind::DualX => "grad_x_L",
            ResidKind::DualS => "grad_s_L",
        }
    }

    /// `true` for the primal (constraint) spaces, `false` for the dual
    /// (stationarity) spaces.
    pub fn is_primal(self) -> bool {
        matches!(self, ResidKind::Eq | ResidKind::Ineq)
    }
}

/// One signed residual component at the current iterate: its space, its
/// index within that space, and its value. See
/// [`DebugCtx::constraint_residuals`] / [`DebugCtx::dual_residuals`].
#[derive(Clone, Copy, Debug)]
pub struct Residual {
    pub kind: ResidKind,
    pub index: usize,
    pub value: Number,
}

/// Live, mutable view of solver state handed to a [`DebugHook`].
///
/// Cheap to construct (two `Rc` clones); every accessor borrows the
/// shared `RefCell`s on demand.
pub struct DebugCtx {
    data: IpoptDataHandle,
    /// Always `Some` in production (the main loop has a live CQ). Left
    /// `None` only by the data-only unit-test constructor, in which case
    /// the CQ-derived scalar accessors report `NaN`.
    cq: Option<IpoptCqHandle>,
    cp: Checkpoint,
    /// Solve outcome, set only for the [`Checkpoint::Terminated`] fire.
    status: Option<String>,
    /// Convergence-tolerance changes the debugger asked to apply *in
    /// place* (so the next `step` honors them). The main loop drains
    /// these after the hook returns and writes them into the live
    /// convergence-check policy — see [`Self::take_live_tolerances`].
    pending_tol: Vec<(String, Number)>,
}

/// Solver options the debugger can apply in place at the next checkpoint:
/// the convergence-check tolerances [`crate::conv_check`]'s policy
/// re-reads each iteration. Anything not listed here is baked into a
/// strategy at build time and needs a `resolve` to take effect.
pub const LIVE_TOLERANCE_OPTS: &[&str] = &[
    "tol",
    "dual_inf_tol",
    "constr_viol_tol",
    "compl_inf_tol",
    "acceptable_tol",
    "acceptable_dual_inf_tol",
    "acceptable_constr_viol_tol",
    "acceptable_compl_inf_tol",
    "acceptable_obj_change_tol",
];

/// Whether `name` is a tolerance the debugger can hot-swap live (next
/// `step`), as opposed to a structural option that needs `resolve`.
pub fn is_live_tolerance(name: &str) -> bool {
    LIVE_TOLERANCE_OPTS.contains(&name)
}

/// A cheap, correct snapshot of the primal-dual state at one step.
///
/// Accepted iterates are immutable frozen [`IteratesVector`]s, so this is
/// just an `Rc` clone plus a few scalars. It captures the iterate, μ, τ,
/// and the iteration index — **not** strategy history (filter, adaptive-μ
/// oracle, quasi-Newton memory), so restoring and continuing is an
/// approximate "resume from here", not a bit-exact rewind.
#[derive(Clone)]
pub struct IterateSnapshot {
    pub iter: i32,
    pub mu: Number,
    pub tau: Number,
    curr: crate::iterates_vector::IteratesVector,
}

impl IterateSnapshot {
    pub fn iter(&self) -> i32 {
        self.iter
    }

    pub fn mu(&self) -> Number {
        self.mu
    }

    pub fn tau(&self) -> Number {
        self.tau
    }

    /// The captured full primal-dual iterate (algorithm space), for the
    /// debugger's `resolve` warm restart. Cloning is `Rc`-shallow.
    pub(crate) fn iterates(&self) -> &crate::iterates_vector::IteratesVector {
        &self.curr
    }

    /// Read a named block of the snapshotted iterate as a flat `f64` vec.
    pub fn block(&self, name: &str) -> Option<Vec<Number>> {
        let v = block_ref(&self.curr, name)?;
        Some(crate::ipopt_alg::flat_read_owned(v.as_ref()))
    }
}

impl DebugCtx {
    pub fn new(data: IpoptDataHandle, cq: IpoptCqHandle, cp: Checkpoint) -> Self {
        Self {
            data,
            cq: Some(cq),
            cp,
            status: None,
            pending_tol: Vec::new(),
        }
    }

    /// Stage a live convergence-tolerance change (e.g. `tol`,
    /// `acceptable_tol`). Accumulated across all commands at one pause and
    /// applied by the main loop after the hook returns, so the next
    /// iteration's convergence test uses the new value. No effect for
    /// names outside [`LIVE_TOLERANCE_OPTS`].
    pub fn set_live_tolerance(&mut self, name: &str, value: Number) {
        self.pending_tol.push((name.to_string(), value));
    }

    /// Drain the staged live-tolerance changes (main loop only).
    pub fn take_live_tolerances(&mut self) -> Vec<(String, Number)> {
        std::mem::take(&mut self.pending_tol)
    }

    /// Attach a solve-outcome string (used for the terminal checkpoint).
    pub fn with_status(mut self, status: String) -> Self {
        self.status = Some(status);
        self
    }

    /// Solve outcome, present only at [`Checkpoint::Terminated`].
    pub fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }

    /// Test-only constructor without a CQ. CQ-derived scalars are `NaN`.
    #[cfg(test)]
    fn new_data_only(data: IpoptDataHandle, cp: Checkpoint) -> Self {
        Self {
            data,
            cq: None,
            cp,
            status: None,
            pending_tol: Vec::new(),
        }
    }

    /// Capture the current primal-dual state for later [`Self::restore`].
    /// `None` before the iterate is set.
    pub fn snapshot(&self) -> Option<IterateSnapshot> {
        let d = self.data.borrow();
        let curr = d.curr.as_ref()?.clone();
        Some(IterateSnapshot {
            iter: d.iter_count,
            mu: d.curr_mu,
            tau: d.curr_tau,
            curr,
        })
    }

    /// Restore a previously captured snapshot: rewinds the iterate, μ, τ,
    /// and iteration index so the next `iterate()` resumes from that
    /// point. Strategy history is not rewound (see [`IterateSnapshot`]).
    pub fn restore(&mut self, snap: &IterateSnapshot) {
        let mut d = self.data.borrow_mut();
        d.set_curr(snap.curr.clone());
        d.curr_mu = snap.mu;
        d.curr_tau = snap.tau;
        d.iter_count = snap.iter;
    }

    fn cq_scalar(
        &self,
        f: impl FnOnce(&crate::ipopt_cq::IpoptCalculatedQuantities) -> Number,
    ) -> Number {
        match self.cq.as_ref() {
            Some(cq) => f(&cq.borrow()),
            None => Number::NAN,
        }
    }

    /// Which checkpoint we are paused at.
    pub fn checkpoint(&self) -> Checkpoint {
        self.cp
    }

    // ---- scalar reads --------------------------------------------------

    /// Current outer iteration counter.
    pub fn iter(&self) -> i32 {
        self.data.borrow().iter_count
    }

    /// Current barrier parameter μ.
    pub fn mu(&self) -> Number {
        self.data.borrow().curr_mu
    }

    /// Unscaled objective at the current iterate.
    pub fn objective(&self) -> Number {
        self.cq_scalar(|c| c.unscaled_curr_f())
    }

    /// Max-norm primal infeasibility.
    pub fn inf_pr(&self) -> Number {
        self.cq_scalar(|c| c.curr_primal_infeasibility_max())
    }

    /// Max-norm dual infeasibility.
    pub fn inf_du(&self) -> Number {
        self.cq_scalar(|c| c.curr_dual_infeasibility_max())
    }

    /// Scaled overall NLP error driving convergence.
    pub fn nlp_error(&self) -> Number {
        self.cq_scalar(|c| c.curr_nlp_error())
    }

    /// Average complementarity (mean slack·multiplier over all bounds) —
    /// the IPM's "distance from the central path" gauge; should track μ.
    pub fn complementarity(&self) -> Number {
        self.cq_scalar(|c| c.curr_avrg_compl())
    }

    /// Slacks to a bound category — `x_l` / `x_u` / `s_l` / `s_u` — i.e.
    /// the distance of each (lower/upper-bounded) variable or inequality
    /// slack from its bound. A small entry ⇒ that bound is near-active.
    pub fn bound_slack(&self, which: &str) -> Option<Vec<Number>> {
        let c = self.cq.as_ref()?.borrow();
        let v = match which {
            "x_l" => c.curr_slack_x_l(),
            "x_u" => c.curr_slack_x_u(),
            "s_l" => c.curr_slack_s_l(),
            "s_u" => c.curr_slack_s_u(),
            _ => return None,
        };
        Some(crate::ipopt_alg::flat_read_owned(v.as_ref()))
    }

    /// Full-length variable bounds in algorithm space — `(x_L, x_U)`, each
    /// of length `n`, with `-∞` / `+∞` in slots that have no lower / upper
    /// bound. Reconstructed from the NLP's *reduced* bound vectors
    /// (`x_l`/`x_u`, indexed over only the bounded variables) and their
    /// expansion matrices, so the result lines up with the `x` block and
    /// with a `set x` / `resolve` seed. `None` in the CQ-less test context
    /// or before the iterate exists.
    ///
    /// These are the bounds the *algorithm* sees — post-scaling and after
    /// any `bound_relax_factor` — which is exactly the space a box-sampled
    /// start must live in to be a valid seed.
    pub fn var_bounds(&self) -> Option<(Vec<Number>, Vec<Number>)> {
        let cq = self.cq.as_ref()?.borrow();
        let nlp = cq.nlp().borrow();
        let d = self.data.borrow();
        let x = &d.curr.as_ref()?.x; // full x-space template
        let lower = expand_bound(&*nlp.px_l(), nlp.x_l(), &**x, Number::NEG_INFINITY);
        let upper = expand_bound(&*nlp.px_u(), nlp.x_u(), &**x, Number::INFINITY);
        Some((lower, upper))
    }

    /// Per-constraint signed primal residuals at the current iterate,
    /// equality constraints ([`ResidKind::Eq`], `c_i(x)`) then inequality
    /// constraints ([`ResidKind::Ineq`], `d_i(x) − s_i`). These are the
    /// same quantities the studio iterate-dump emits as `slack`, and the
    /// largest `|value|` over the returned vector equals [`Self::inf_pr`].
    /// `None` only in the CQ-less test context.
    pub fn constraint_residuals(&self) -> Option<Vec<Residual>> {
        let cq = self.cq.as_ref()?.borrow();
        let c = crate::ipopt_alg::flat_read_owned(cq.curr_c().as_ref());
        let dms = crate::ipopt_alg::flat_read_owned(cq.curr_d_minus_s().as_ref());
        let mut out = Vec::with_capacity(c.len() + dms.len());
        out.extend(c.iter().enumerate().map(|(index, &value)| Residual {
            kind: ResidKind::Eq,
            index,
            value,
        }));
        out.extend(dms.iter().enumerate().map(|(index, &value)| Residual {
            kind: ResidKind::Ineq,
            index,
            value,
        }));
        Some(out)
    }

    /// Per-variable signed dual residuals (Lagrangian-gradient
    /// components) at the current iterate, `x`-space
    /// ([`ResidKind::DualX`], `(∇_x L)_i`) then `s`-space
    /// ([`ResidKind::DualS`], `(∇_s L)_i`). The largest `|value|` over
    /// the returned vector equals [`Self::inf_du`]. `None` only in the
    /// CQ-less test context.
    pub fn dual_residuals(&self) -> Option<Vec<Residual>> {
        let cq = self.cq.as_ref()?.borrow();
        let gx = crate::ipopt_alg::flat_read_owned(cq.curr_grad_lag_x().as_ref());
        let gs = crate::ipopt_alg::flat_read_owned(cq.curr_grad_lag_s().as_ref());
        let mut out = Vec::with_capacity(gx.len() + gs.len());
        out.extend(gx.iter().enumerate().map(|(index, &value)| Residual {
            kind: ResidKind::DualX,
            index,
            value,
        }));
        out.extend(gs.iter().enumerate().map(|(index, &value)| Residual {
            kind: ResidKind::DualS,
            index,
            value,
        }));
        Some(out)
    }

    /// Model names for the residual index spaces, projected into the
    /// solver's split space (free variables, equalities, inequalities), or
    /// `None` when the problem carries no names (or in the CQ-less test
    /// context). The REPL pairs these with [`Residual::kind`] /
    /// [`Residual::index`] to print `mass_balance` instead of `c[3]` —
    /// closing the model-vs-index reporting gap Lee et al. (2024,
    /// <https://doi.org/10.69997/sct.147875>) identify for equation-oriented
    /// model debugging.
    pub fn split_names(&self) -> Option<SplitNames> {
        let cq = self.cq.as_ref()?.borrow();
        let names = cq.nlp().borrow().split_space_names();
        names
    }

    /// Primal regularization δ_w **as recorded for this iteration's info**
    /// (`info_regu_x`) — the value reported in the iteration table's `lg(rg)`
    /// column, reset to 0 at the start of each iteration. This is distinct
    /// from [`Self::kkt`]'s `delta_w`, which reads the *live* perturbation
    /// (`perturbations.delta_x`) applied during the inertia-correction loop;
    /// the two can differ by timing at a given checkpoint.
    pub fn regularization(&self) -> Number {
        self.data.borrow().info_regu_x
    }

    /// Number of line-search trial points tried for the accepted step
    /// (1 ⇒ full step accepted first try).
    pub fn ls_count(&self) -> i32 {
        self.data.borrow().info_ls_count
    }

    /// Accepted primal / dual step lengths (α_pr, α_du).
    pub fn alpha(&self) -> (Number, Number) {
        let d = self.data.borrow();
        (d.info_alpha_primal, d.info_alpha_dual)
    }

    /// KKT-factorization report for the current iteration, if a search
    /// direction has been computed this iteration (i.e. at/after the
    /// `after_search_dir` checkpoint). Combines the captured inertia with
    /// the applied regularization and the *expected* inertia derived from
    /// the multiplier dimensions. `delta_w`/`delta_c` are the **live**
    /// primal/dual perturbations (`perturbations.delta_x/delta_c`) applied
    /// during inertia correction — distinct from [`Self::regularization`]'s
    /// recorded per-iteration info value (see its note).
    pub fn kkt(&self) -> Option<KktReport> {
        let d = self.data.borrow();
        let k = d.kkt_debug.as_ref()?;
        let curr = d.curr.as_ref();
        let expected_neg = curr.map(|c| c.y_c.dim() + c.y_d.dim()).unwrap_or(0);
        // n+ = dim − n− (assuming a non-singular KKT, n0 = 0).
        let n_pos = if k.n_neg >= 0 { k.dim - k.n_neg } else { -1 };
        let inertia_correct = k.provides_inertia && k.n_neg == expected_neg;
        Some(KktReport {
            iter: k.iter,
            dim: k.dim,
            n_neg: k.n_neg,
            n_pos,
            expected_neg,
            provides_inertia: k.provides_inertia,
            inertia_correct,
            delta_w: d.perturbations.delta_x,
            delta_c: d.perturbations.delta_c,
            status: k.status.clone(),
        })
    }

    /// The assembled KKT matrix triplets `(dim, irn, jcn, vals)` (1-based
    /// lower triangle) for `viz kkt`, if captured this iteration.
    pub fn kkt_matrix(&self) -> Option<(i32, Vec<i32>, Vec<i32>, Vec<Number>)> {
        self.data.borrow().kkt_debug.as_ref()?.matrix.clone()
    }

    /// The outer iteration the captured KKT system / factor came from —
    /// the previous iteration at an `iter_start` pause (look-back). For
    /// labeling `viz kkt` / `viz L` with the right iteration.
    pub fn kkt_captured_iter(&self) -> Option<i32> {
        Some(self.data.borrow().kkt_debug.as_ref()?.iter)
    }

    /// The `LDLᵀ` factor (`n`, `perm`, strict-lower `l_irn`/`l_jcn` and
    /// optional `l_vals`) for `viz L`, if captured this iteration (i.e.
    /// the debugger was stepping — see `DebugHook::wants_kkt_capture`).
    #[allow(clippy::type_complexity)]
    pub fn kkt_l_factor(
        &self,
    ) -> Option<(usize, Vec<usize>, Vec<i32>, Vec<i32>, Option<Vec<Number>>)> {
        let d = self.data.borrow();
        let f = d.kkt_debug.as_ref()?.l_factor.as_ref()?;
        Some((
            f.n,
            f.perm.clone(),
            f.l_irn.clone(),
            f.l_jcn.clone(),
            f.l_vals.clone(),
        ))
    }

    // ---- vector reads --------------------------------------------------

    /// Dimensions of every named block, in [`BLOCK_NAMES`] order.
    pub fn block_dims(&self) -> Vec<(&'static str, usize)> {
        let d = self.data.borrow();
        let Some(curr) = d.curr.as_ref() else {
            return BLOCK_NAMES.iter().map(|&n| (n, 0)).collect();
        };
        BLOCK_NAMES
            .iter()
            .map(|&n| (n, block_ref(curr, n).map(|v| v.dim() as usize).unwrap_or(0)))
            .collect()
    }

    /// Read a named block of the current iterate as a flat `f64` vec.
    /// Returns `None` for an unknown name or before the iterate is set.
    pub fn block(&self, name: &str) -> Option<Vec<Number>> {
        let d = self.data.borrow();
        let curr = d.curr.as_ref()?;
        let v = block_ref(curr, name)?;
        Some(crate::ipopt_alg::flat_read_owned(v.as_ref()))
    }

    /// Read a named block of the most recent search direction δ.
    pub fn delta_block(&self, name: &str) -> Option<Vec<Number>> {
        let d = self.data.borrow();
        let delta = d.delta.as_ref()?;
        let v = block_ref(delta, name)?;
        Some(crate::ipopt_alg::flat_read_owned(v.as_ref()))
    }

    // ---- mutation ------------------------------------------------------

    /// Overwrite the barrier parameter μ. Takes effect on the next
    /// `update_barrier_parameter` consult (the monotone updater treats
    /// it as the current value; adaptive updaters re-derive from it).
    pub fn set_mu(&mut self, mu: Number) -> Result<(), String> {
        if !mu.is_finite() || mu <= 0.0 {
            return Err(format!("mu must be finite and positive, got {mu}"));
        }
        self.data.borrow_mut().curr_mu = mu;
        Ok(())
    }

    /// Overwrite an entire named block of the current iterate.
    ///
    /// Rebuilds `curr` from a deep copy with a fresh vector tag, so all
    /// tag-keyed CQ caches invalidate and downstream quantities recompute
    /// from the new point.
    ///
    /// **No invariant enforcement.** Only the dimension is checked. Mutating
    /// the slacks (`s`) to ≤ 0, or the bound/inequality multipliers
    /// (`z_l`/`z_u`/`v_l`/`v_u`) to ≤ 0, or pushing `x` past a bound, breaks
    /// the interior-point feasibility invariant (`s > 0, z > 0`) — the next
    /// step's σ = z/s and fraction-to-the-boundary rule can then produce
    /// `NaN`/`Inf` or a non-descent direction rather than erroring here. The
    /// solver's strategy history (filter, adaptive-μ oracle, quasi-Newton
    /// memory) is **not** rewound to the mutated point either, so a resumed
    /// solve may behave inconsistently. This is a debugging tool: "wrong"
    /// states are allowed on purpose — it's on the caller to keep the point
    /// sane if they intend to continue the solve.
    pub fn set_block(&mut self, name: &str, vals: &[Number]) -> Result<(), String> {
        if !BLOCK_NAMES.contains(&name) {
            return Err(format!(
                "unknown block `{name}` (expected one of {BLOCK_NAMES:?})"
            ));
        }
        let mut d = self.data.borrow_mut();
        let curr = d.curr.as_ref().ok_or("no current iterate yet")?;
        let mut m = curr.deep_copy();
        let blk = block_ref_mut(&mut m, name).expect("name checked above");
        let dim = blk.dim() as usize;
        if vals.len() != dim {
            return Err(format!(
                "block `{name}` has dimension {dim}, got {} value(s)",
                vals.len()
            ));
        }
        crate::ipopt_alg::flat_write_into(blk.as_mut(), vals);
        let frozen = m.freeze();
        d.set_curr(frozen);
        Ok(())
    }

    /// Overwrite a single component of a named block.
    pub fn set_component(&mut self, name: &str, idx: usize, val: Number) -> Result<(), String> {
        let mut vals = self
            .block(name)
            .ok_or_else(|| format!("unknown block `{name}` or no iterate yet"))?;
        if idx >= vals.len() {
            return Err(format!(
                "index {idx} out of range for block `{name}` (dimension {})",
                vals.len()
            ));
        }
        vals[idx] = val;
        self.set_block(name, &vals)
    }
}

/// Expand a *reduced* bound vector (one entry per bounded variable) into a
/// full-length `Vec<Number>`, placing `absent` in slots that variable has
/// no such bound. `p` is the bound's expansion matrix (full × reduced),
/// `template` any full x-space vector (for the right dimension/space).
///
/// `P · 1` marks which full slots are bounded; `P · bound` scatters the
/// bound values into them. Anything the mask leaves untouched gets `absent`
/// (`±∞`).
fn expand_bound(
    p: &dyn Matrix,
    reduced: &dyn Vector,
    template: &dyn Vector,
    absent: Number,
) -> Vec<Number> {
    let mut ones = reduced.make_new();
    ones.set(1.0);
    let mut mask = template.make_new();
    p.mult_vector(1.0, &*ones, 0.0, &mut *mask);
    let mut vals = template.make_new();
    p.mult_vector(1.0, reduced, 0.0, &mut *vals);
    let mask = crate::ipopt_alg::flat_read_owned(&*mask);
    let vals = crate::ipopt_alg::flat_read_owned(&*vals);
    mask.iter()
        .zip(vals)
        .map(|(&m, v)| if m > 0.5 { v } else { absent })
        .collect()
}

/// Borrow a named block of an [`IteratesVector`].
fn block_ref<'a>(
    iv: &'a crate::iterates_vector::IteratesVector,
    name: &str,
) -> Option<&'a std::rc::Rc<dyn pounce_linalg::Vector>> {
    Some(match name {
        "x" => &iv.x,
        "s" => &iv.s,
        "y_c" => &iv.y_c,
        "y_d" => &iv.y_d,
        "z_l" => &iv.z_l,
        "z_u" => &iv.z_u,
        "v_l" => &iv.v_l,
        "v_u" => &iv.v_u,
        _ => return None,
    })
}

/// Borrow a named block of a mutable [`IteratesVectorMut`].
fn block_ref_mut<'a>(
    iv: &'a mut crate::iterates_vector::IteratesVectorMut,
    name: &str,
) -> Option<&'a mut Box<dyn pounce_linalg::Vector>> {
    Some(match name {
        "x" => &mut iv.x,
        "s" => &mut iv.s,
        "y_c" => &mut iv.y_c,
        "y_d" => &mut iv.y_d,
        "z_l" => &mut iv.z_l,
        "z_u" => &mut iv.z_u,
        "v_l" => &mut iv.v_l,
        "v_u" => &mut iv.v_u,
        _ => return None,
    })
}

/// A consumer that the main loop pauses at each checkpoint. The CLI's
/// REPL / agent driver is the production implementation.
pub trait DebugHook {
    /// Called at every [`Checkpoint`]. Inspect and/or mutate via `ctx`,
    /// then return whether to keep solving.
    fn at_checkpoint(&mut self, ctx: &mut DebugCtx) -> DebugAction;

    /// Whether the main loop should capture the (heavier) KKT matrix
    /// triplets and `LDLᵀ` factor into `kkt_debug` this iteration, so
    /// `viz kkt` / `viz L` can look back at the previous iteration's
    /// system. True while the debugger is stepping interactively; an
    /// implementation that has detached (running free) returns false so
    /// the O(nnz) assembly isn't paid every iteration. Defaults to true
    /// — the cheap inertia/status fields are captured regardless.
    fn wants_kkt_capture(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipopt_data::IpoptData;
    use crate::iterates_vector::IteratesVector;
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use pounce_linalg::Vector;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn iv(xvals: &[f64]) -> IteratesVector {
        let dense = |vals: &[f64]| {
            let mut v = DenseVectorSpace::new(vals.len() as i32).make_new_dense();
            v.set_values(vals);
            Rc::new(v) as Rc<dyn Vector>
        };
        let z = |n| dense(&vec![0.0; n]);
        IteratesVector::new(dense(xvals), z(1), z(1), z(1), z(2), z(2), z(1), z(1))
    }

    fn ctx_with(xvals: &[f64]) -> DebugCtx {
        let mut data = IpoptData::new();
        data.set_curr(iv(xvals));
        data.curr_mu = 0.1;
        let data = Rc::new(RefCell::new(data));
        DebugCtx::new_data_only(data, Checkpoint::IterStart)
    }

    #[test]
    fn reads_block_and_mu() {
        let ctx = ctx_with(&[1.0, 2.0]);
        assert_eq!(ctx.mu(), 0.1);
        assert_eq!(ctx.block("x"), Some(vec![1.0, 2.0]));
        assert_eq!(ctx.block("nope"), None);
    }

    #[test]
    fn set_component_rebuilds_iterate_with_fresh_tag() {
        let mut ctx = ctx_with(&[1.0, 2.0]);
        let before = ctx
            .data
            .borrow()
            .curr
            .as_ref()
            .unwrap()
            .x
            .as_tagged()
            .get_tag();
        ctx.set_component("x", 1, 9.0).unwrap();
        let after = ctx
            .data
            .borrow()
            .curr
            .as_ref()
            .unwrap()
            .x
            .as_tagged()
            .get_tag();
        assert_eq!(ctx.block("x"), Some(vec![1.0, 9.0]));
        assert_ne!(before, after, "mutating the iterate must mint a new tag");
    }

    #[test]
    fn set_block_dim_mismatch_is_rejected() {
        let mut ctx = ctx_with(&[1.0, 2.0]);
        assert!(ctx.set_block("x", &[1.0]).is_err());
        assert!(ctx.set_block("x", &[1.0, 2.0, 3.0]).is_err());
        assert!(ctx.set_block("x", &[3.0, 4.0]).is_ok());
        assert_eq!(ctx.block("x"), Some(vec![3.0, 4.0]));
    }

    #[test]
    fn block_names_all_resolve_in_block_ref() {
        // Locks `BLOCK_NAMES` to the `block_ref` / `block_ref_mut` match arms:
        // every name must resolve in both, or `set_block`'s
        // `expect("name checked above")` could panic on a name that's in the
        // array but missing from a match arm.
        let mut ctx = ctx_with(&[1.0, 2.0]);
        for name in BLOCK_NAMES {
            let cur = ctx
                .block(name)
                .unwrap_or_else(|| panic!("block_ref does not resolve `{name}`"));
            // Round-trips through `block_ref_mut` (dimension-correct values).
            ctx.set_block(name, &cur)
                .unwrap_or_else(|e| panic!("block_ref_mut does not resolve `{name}`: {e}"));
        }
    }

    #[test]
    fn residuals_are_none_without_cq() {
        // The data-only test ctx has no CQ, so both residual accessors
        // report `None` (mirrors the documented NaN-scalar contract).
        let ctx = ctx_with(&[1.0, 2.0]);
        assert!(ctx.constraint_residuals().is_none());
        assert!(ctx.dual_residuals().is_none());
    }

    #[test]
    fn resid_kind_tags_and_primal_classification_are_stable() {
        assert_eq!(ResidKind::Eq.tag(), "c");
        assert_eq!(ResidKind::Ineq.tag(), "d-s");
        assert_eq!(ResidKind::DualX.tag(), "grad_x_L");
        assert_eq!(ResidKind::DualS.tag(), "grad_s_L");
        assert!(ResidKind::Eq.is_primal());
        assert!(ResidKind::Ineq.is_primal());
        assert!(!ResidKind::DualX.is_primal());
        assert!(!ResidKind::DualS.is_primal());
    }

    #[test]
    fn checkpoint_as_str_is_stable() {
        // These strings are the wire/CLI protocol names — intentionally
        // distinct from the variant identifiers (e.g. `AfterBarrierUpdate` →
        // `"after_mu"`). Locked here so a rename is a deliberate,
        // protocol-breaking change rather than a silent one.
        assert_eq!(Checkpoint::IterStart.as_str(), "iter_start");
        assert_eq!(Checkpoint::AfterBarrierUpdate.as_str(), "after_mu");
        assert_eq!(
            Checkpoint::AfterSearchDirection.as_str(),
            "after_search_dir"
        );
        assert_eq!(Checkpoint::AfterStep.as_str(), "after_step");
        assert_eq!(Checkpoint::StepRejected.as_str(), "step_rejected");
        assert_eq!(Checkpoint::PreRestoration.as_str(), "pre_restoration_entry");
        assert_eq!(
            Checkpoint::PostRestoration.as_str(),
            "post_restoration_exit"
        );
        assert_eq!(Checkpoint::Terminated.as_str(), "terminated");
    }

    #[test]
    fn snapshot_then_restore_round_trips_iterate_and_mu() {
        let mut ctx = ctx_with(&[1.0, 2.0]);
        let snap = ctx.snapshot().expect("snapshot");
        assert_eq!(snap.iter(), 0);
        // Mutate away from the snapshot.
        ctx.set_component("x", 0, 99.0).unwrap();
        ctx.set_mu(0.5).unwrap();
        assert_eq!(ctx.block("x"), Some(vec![99.0, 2.0]));
        assert_eq!(ctx.mu(), 0.5);
        // Restore brings back the captured state.
        ctx.restore(&snap);
        assert_eq!(ctx.block("x"), Some(vec![1.0, 2.0]));
        assert_eq!(ctx.mu(), 0.1);
        assert_eq!(ctx.iter(), 0);
    }

    #[test]
    fn set_mu_rejects_nonpositive() {
        let mut ctx = ctx_with(&[1.0]);
        assert!(ctx.set_mu(-1.0).is_err());
        assert!(ctx.set_mu(0.0).is_err());
        assert!(ctx.set_mu(1e-3).is_ok());
        assert_eq!(ctx.mu(), 1e-3);
    }
}
