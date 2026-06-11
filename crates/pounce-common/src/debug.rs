//! Shared interior-point debugger abstraction.
//!
//! The interactive solver debugger (a "pdb for the interior-point loop")
//! is driven by a [`DebugHook`] that the solver fires at well-defined
//! [`Checkpoint`]s. The hook receives a `&mut dyn` [`DebugState`] — a
//! live, possibly-mutable view of the solver's per-iteration state — and
//! returns a [`DebugAction`] telling the loop whether to keep solving.
//!
//! These traits live in `pounce-common` so that *every* solver can be
//! debugged by the *same* REPL: the NLP filter-IPM (`pounce-algorithm`)
//! and the convex / conic IPM (`pounce-convex`) both implement
//! [`DebugState`] over their own state, and the CLI's `SolverDebugger`
//! implements [`DebugHook`] once against the trait.
//!
//! [`DebugState`] splits its surface in two:
//!
//!   * **Generic** accessors every interior-point method has — iteration
//!     index, μ, objective, primal/dual infeasibility, complementarity,
//!     step lengths, and named iterate / search-direction blocks — are
//!     required methods.
//!   * **Solver-specific** extras (the NLP error metric, bound-slack
//!     active-set view, KKT inertia / matrix / factor capture, line-search
//!     trial count, snapshot/restore, mutation) have default impls that
//!     report "unsupported", so a solver overrides only what it actually
//!     has. The REPL turns an unsupported result into a friendly message.

use crate::types::Number;
use std::any::Any;

/// Where in a solver's loop a checkpoint fired.
///
/// The variants cover the NLP filter-IPM's loop; other interior-point
/// solvers fire the subset that applies to them (e.g. the convex IPM uses
/// [`IterStart`](Checkpoint::IterStart),
/// [`AfterSearchDirection`](Checkpoint::AfterSearchDirection),
/// [`AfterStep`](Checkpoint::AfterStep), and
/// [`Terminated`](Checkpoint::Terminated); it has no restoration phase or
/// backtracking line search, so those variants simply never fire).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checkpoint {
    /// Top of an outer iteration — before this iteration's step is
    /// computed. The iterate, multipliers, and μ reflect the *accepted*
    /// point from the previous iteration.
    IterStart,
    /// After the barrier parameter μ was updated for this iteration
    /// (before the search direction is computed).
    AfterBarrierUpdate,
    /// After the primal-dual Newton step was computed — the search
    /// direction `δ`, the applied regularization, and the KKT
    /// factorization are available.
    AfterSearchDirection,
    /// After a step length was chosen and the trial point accepted — the
    /// step lengths α and the new iterate are in place.
    AfterStep,
    /// The line search *rejected* this iteration's step and the solver is
    /// about to fall into restoration (NLP filter-IPM only).
    StepRejected,
    /// Just before the algorithm switches into the restoration phase
    /// (NLP filter-IPM only).
    PreRestoration,
    /// Just after the restoration phase returns (NLP filter-IPM only).
    PostRestoration,
    /// The solve has finished: fired once before the solver returns, at
    /// the final iterate, carrying the outcome via [`DebugState::status`].
    /// The [`DebugAction`] returned here is **ignored** — the solve is
    /// already over.
    Terminated,
}

impl Checkpoint {
    /// The stable wire/CLI protocol name for this checkpoint. These strings
    /// are intentionally **not** the variant identifiers (`AfterBarrierUpdate`
    /// → `"after_mu"`, `PreRestoration` → `"pre_restoration_entry"`) — they're
    /// the names the JSON protocol and `stop-at` use, so match on the variant,
    /// not the string.
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

/// What the solver should do after a [`DebugHook`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DebugAction {
    /// Keep solving.
    Resume,
    /// Stop the solve now. Surfaces to the caller as a
    /// user-requested-stop outcome.
    Stop,
}

/// KKT-factorization report (see [`DebugState::kkt`]). The inertia of a
/// well-posed primal-dual system is `(n_pos = n, n_neg = m, n_zero = 0)`;
/// a mismatch (or nonzero regularization) is the classic signal that the
/// step is being stabilized.
#[derive(Clone, Debug)]
pub struct KktReport {
    /// The outer iteration this factorization was assembled at — may be the
    /// previous iteration when paused at `iter_start` (viz look-back).
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

/// Captured `LDLᵀ` factor for `viz L`:
/// `(n, perm, l_irn, l_jcn, l_vals)`.
pub type LFactor = (usize, Vec<usize>, Vec<i32>, Vec<i32>, Option<Vec<Number>>);

/// Assembled KKT matrix triplets for `viz kkt`:
/// `(dim, irn, jcn, vals)` (1-based lower triangle).
pub type KktTriplets = (i32, Vec<i32>, Vec<i32>, Vec<Number>);

/// Which residual space a [`Residual`] entry comes from.
///
/// Primal entries are the per-constraint violations whose max-norm is
/// `inf_pr`; dual entries are the per-variable Lagrangian-gradient
/// components whose max-norm is `inf_du`. (NLP-specific; the convex/conic
/// and global solvers do not expose per-component residuals.)
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
/// [`DebugState::constraint_residuals`] / [`DebugState::dual_residuals`].
#[derive(Clone, Copy, Debug)]
pub struct Residual {
    pub kind: ResidKind,
    pub index: usize,
    pub value: Number,
}

/// An opaque, readable snapshot of a solver's primal-dual state at one
/// iteration, returned by [`DebugState::snapshot`] and replayed by
/// [`DebugState::restore`].
///
/// The reader methods (`iter` / `mu` / `block`) let the REPL `diff` two
/// captured points generically; [`as_any`](IterSnapshot::as_any) lets the
/// originating solver downcast back to its concrete snapshot type to
/// restore it.
pub trait IterSnapshot: Any {
    /// Iteration index this snapshot was taken at.
    fn iter(&self) -> i32;
    /// Barrier parameter μ at the snapshot.
    fn mu(&self) -> Number;
    /// A named iterate block at the snapshot, if present.
    fn block(&self, name: &str) -> Option<Vec<Number>>;
    /// Downcast handle for the originating solver's `restore`.
    fn as_any(&self) -> &dyn Any;
}

/// A live view of solver state handed to a [`DebugHook`] at a checkpoint.
///
/// Required methods are the quantities every interior-point method has.
/// The remaining methods carry solver-specific capabilities and default
/// to "unsupported" (NaN / `None` / `-1` / `Err`), so a solver overrides
/// only the ones it can answer. `set_*` mutators likewise default to a
/// descriptive `Err` for solvers that don't support in-place edits.
pub trait DebugState {
    // ---- required: generic interior-point quantities -------------------

    /// Downcast escape hatch for **solver-specific** REPL commands whose
    /// payload can't live in this leaf crate (e.g. the NLP debugger's
    /// rank diagnosis, model-name resolution, or full primal-dual warm
    /// `resolve`). A solver that supports those returns `Some(self)` so the
    /// REPL can downcast to its concrete state; the default `None` makes the
    /// command report "not supported for this solver".
    fn as_any(&self) -> Option<&dyn Any> {
        None
    }

    /// Mutable form of [`as_any`](DebugState::as_any), for commands that
    /// mutate solver-specific state (e.g. live-tolerance hot-swap).
    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        None
    }

    /// Which checkpoint we are paused at.
    fn checkpoint(&self) -> Checkpoint;

    /// Current outer iteration counter.
    fn iter(&self) -> i32;

    /// Current barrier parameter μ.
    fn mu(&self) -> Number;

    /// Objective at the current iterate (in the user's original sense).
    fn objective(&self) -> Number;

    /// Max-norm primal infeasibility.
    fn inf_pr(&self) -> Number;

    /// Max-norm dual infeasibility.
    fn inf_du(&self) -> Number;

    /// Average complementarity — the IPM's "distance from the central
    /// path" gauge; should track μ.
    fn complementarity(&self) -> Number;

    /// Accepted primal / dual step lengths (α_pr, α_du). A solver with a
    /// single symmetric step (e.g. HSDE) reports it in both slots.
    fn alpha(&self) -> (Number, Number);

    /// Dimensions of every named iterate block, in display order.
    fn block_dims(&self) -> Vec<(&'static str, usize)>;

    /// Read a named block of the current iterate as a flat `f64` vec.
    /// `None` for an unknown name or before the iterate is set.
    fn block(&self, name: &str) -> Option<Vec<Number>>;

    /// Read a named block of the most recent search direction.
    fn delta_block(&self, name: &str) -> Option<Vec<Number>>;

    // ---- optional: solver-specific extras (default = unsupported) ------

    /// Solve outcome, present only at [`Checkpoint::Terminated`].
    fn status(&self) -> Option<&str> {
        None
    }

    /// A scalar convergence error driving termination (the NLP "nlp_error").
    /// `NaN` when the solver has no single such metric.
    fn nlp_error(&self) -> Number {
        Number::NAN
    }

    /// Slacks to a bound category (`x_l` / `x_u` / `s_l` / `s_u`) for the
    /// active-set view. `None` when the solver has no bound-slack notion.
    fn bound_slack(&self, _which: &str) -> Option<Vec<Number>> {
        None
    }

    /// Regularization applied to the KKT system this iteration. `NaN` when
    /// the solver does not expose one.
    fn regularization(&self) -> Number {
        Number::NAN
    }

    /// Number of line-search trial points for the accepted step. `-1` for
    /// solvers without a backtracking line search (e.g. the convex IPM,
    /// which takes a fraction-to-boundary step).
    fn ls_count(&self) -> i32 {
        -1
    }

    /// KKT-factorization inertia / regularization report, if available.
    fn kkt(&self) -> Option<KktReport> {
        None
    }

    /// Assembled KKT matrix triplets for `viz kkt`, if captured.
    fn kkt_matrix(&self) -> Option<KktTriplets> {
        None
    }

    /// The `LDLᵀ` factor for `viz L`, if captured.
    fn kkt_l_factor(&self) -> Option<LFactor> {
        None
    }

    /// The iteration the currently-captured KKT matrix / factor came from
    /// (may be the previous iteration when paused at `iter_start`, the viz
    /// look-back). `None` when nothing is captured or unsupported.
    fn kkt_captured_iter(&self) -> Option<i32> {
        None
    }

    /// Ask the solver to capture the `LDLᵀ` factor on later solves.
    /// Returns whether it is already available now.
    fn request_l_factor(&mut self) -> bool {
        false
    }

    /// Ask the solver to assemble the KKT triplets on later solves.
    /// Returns whether they are already available now.
    fn request_kkt_matrix(&mut self) -> bool {
        false
    }

    /// Overwrite the barrier parameter μ.
    fn set_mu(&mut self, _mu: Number) -> Result<(), String> {
        Err("this solver does not support setting mu".into())
    }

    /// Overwrite an entire named block of the current iterate.
    fn set_block(&mut self, _name: &str, _vals: &[Number]) -> Result<(), String> {
        Err("this solver does not support editing the iterate".into())
    }

    /// Overwrite a single component of a named block. Defaults to a
    /// read-modify-write through [`block`](DebugState::block) /
    /// [`set_block`](DebugState::set_block).
    fn set_component(&mut self, name: &str, idx: usize, val: Number) -> Result<(), String> {
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

    /// Capture the current primal-dual state for a later [`restore`].
    /// `None` when snapshots are unsupported or no iterate is set yet.
    ///
    /// [`restore`]: DebugState::restore
    fn snapshot(&self) -> Option<Box<dyn IterSnapshot>> {
        None
    }

    /// Restore a snapshot previously returned by [`snapshot`]. Returns
    /// whether the restore succeeded (false on unsupported, or a snapshot
    /// minted by a different solver).
    ///
    /// [`snapshot`]: DebugState::snapshot
    fn restore(&mut self, _snap: &dyn IterSnapshot) -> bool {
        false
    }

    /// Per-constraint signed primal residuals at the current iterate (the
    /// components whose max-norm is `inf_pr`), for the `print residuals`
    /// command. `None` when the solver does not expose per-component
    /// residuals (the convex/conic and global solvers).
    fn constraint_residuals(&self) -> Option<Vec<Residual>> {
        None
    }

    /// Per-variable signed dual (Lagrangian-gradient) residuals at the
    /// current iterate (the components whose max-norm is `inf_du`). `None`
    /// when unsupported.
    fn dual_residuals(&self) -> Option<Vec<Residual>> {
        None
    }
}

/// A consumer that a solver pauses at each [`Checkpoint`]. The CLI's
/// REPL / agent driver is the production implementation; the same hook
/// instance can drive any solver that exposes a [`DebugState`].
pub trait DebugHook {
    /// Called at every checkpoint. Inspect and/or mutate via `state`, then
    /// return whether to keep solving.
    fn at_checkpoint(&mut self, state: &mut dyn DebugState) -> DebugAction;

    /// Whether the solver should capture the (heavier) KKT matrix triplets
    /// and `LDLᵀ` factor this iteration, so `viz kkt` / `viz L` can look back
    /// at the previous iteration's system. True while stepping interactively;
    /// a detached (running-free) hook returns false so the O(nnz) assembly
    /// isn't paid every iteration. The cheap inertia/status fields are
    /// captured regardless.
    fn wants_kkt_capture(&self) -> bool {
        true
    }

    /// Arm the hook to pause at the next checkpoint. Used to debug a
    /// sub-solve **on demand** — an outer driver can re-arm this
    /// interior-point hook just before a particular solve, so the hook
    /// stays quiet otherwise but drops in for that one solve. Default:
    /// no-op (always-on hooks ignore it).
    fn arm(&mut self) {}
}
