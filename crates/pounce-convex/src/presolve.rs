//! Presolve for convex QP and LP (Phase 3.5).
//!
//! Reduces a [`QpProblem`] before the interior-point solve and maps the
//! reduced solution back to the original problem space, recovering both
//! the primal `x` and the duals `(y, z)`. The contract is correctness of
//! the recovered KKT point: a presolved-then-postsolved solve yields a
//! valid primal–dual solution of the *original* problem (see
//! `tests/presolve_roundtrip.rs` and `tests/presolve_reductions.rs`).
//!
//! This is the architectural seam the dev note calls the "missing
//! piece": a **transaction stack** of [`Reduction`]s, each carrying the
//! data needed to undo itself (primal *and* dual). Postsolve replays the
//! stack in reverse. The catalog is small but the postsolve is complete,
//! so richer reductions can be added without reworking the recovery path.
//!
//! Reductions implemented:
//! - **Empty rows** (equality / inequality with no nonzeros): a
//!   feasibility check, then drop. Their dual is zero. Detects trivial
//!   primal infeasibility (`0 = b≠0` or `0 ≤ h<0`). A row emptied by
//!   *substitution* is judged against the rounding error of the terms
//!   that cancelled, not against exact zero — otherwise two equalities
//!   that agree to all but the last bit read as contradictory (gh#496).
//! - **Fixed-variable elimination** from a singleton equality row
//!   (`a·x_k = b ⇒ x_k = b/a`): substitute `x_k` out of `P`, `c`, `A`,
//!   `G` (adjusting the objective constant and the row right-hand
//!   sides), and recover the fixing row's multiplier from stationarity
//!   at the postsolved point. The QP-aware reduction (the Hessian
//!   coupling moves into the linear term and the dual must be recovered
//!   consistently with `P`).
//! - **Empty/free-column elimination**: a variable absent from `P`, `A`,
//!   and `G` is free and unconstrained, so its only objective effect is
//!   `c_k x_k`. If `c_k = 0` the variable is irrelevant (set to 0, drop);
//!   if `c_k ≠ 0` the problem is unbounded below (detected as
//!   [`PresolveOutcome::Unbounded`]).
//! - **Parallel-row removal** (equality / inequality): rows that are
//!   **scalar multiples** of one another (after substitution) — exact
//!   duplicates being the unit-scale case — are redundant or expose
//!   infeasibility. Detection normalizes each row by a canonical pivot and
//!   uses rayon-parallel per-row hashing (PaPILO's hashing-based pairing),
//!   confirming candidates with a tolerance so a wrong merge is
//!   impossible (a quantization split only ever *misses* a pair).
//!   Parallel equalities with inconsistent (scaled) right-hand sides ⇒
//!   infeasible; parallel inequalities (positive multiples — same
//!   direction) keep the most restrictive row. Dual recovery stays
//!   trivial because the *kept* row is an original one in its own frame
//!   and every dropped row's multiplier is zero — a valid KKT point.
//! - **Free column singleton substitution**: an unbounded variable,
//!   absent from `P` and `G`, that appears in exactly one (multi-entry)
//!   equality row is substituted out via `x_col = (b_r − Σ_{j≠col} a_j
//!   x_j) / a_col`, eliminating both the variable *and* the row. The
//!   substitution shifts cost onto the surviving variables; the consumed
//!   row's multiplier is the unique value `y_r = −c_col / a_col`. This is
//!   a clean PaPILO reduction (uniquely determined dual), unlike forcing
//!   constraints / bound tightening.
//! - **Activity-bound reductions** (need the variable box): for each
//!   inequality `g·x ≤ h`, compute the activity range `[min, max]` over
//!   the box. If `max ≤ h` the row is always satisfied → **redundant**,
//!   drop it (dual 0). If `min > h` the row can never hold →
//!   **infeasible**. For each equality `a·x = b`, infeasible when `b`
//!   lies outside `[min, max]`.
//! - **Dominated columns**: a variable absent from `P` and the equalities
//!   that appears in inequalities `Gx ≤ h` with sign-definite coefficients
//!   matching its cost sign is optimal at a bound (pushing it there raises
//!   neither the objective nor any row's activity), so it is fixed and
//!   dropped. Its bound multiplier is its reduced cost `c_k + Σᵢ aᵢₖ zᵢ`,
//!   which the sign conditions make nonnegative — a valid dual by
//!   construction. (PaPILO's dominated-column reduction, restricted to the
//!   clean sign-guaranteed case.)
//! - **Forcing constraints**: when a row's activity range *touches* its
//!   right-hand side it can hold only at one vertex of the box, pinning
//!   every involved variable to a bound (inequality `g·x ≤ h` with
//!   `min = h`; equality `a·x = b` with `min = b` or `max = b`). The row
//!   is dropped and each variable fixed. The dual recovery — the reason
//!   this was the hard PaPILO postsolve — is exact: the forcing row's
//!   multiplier is the tightest value making every pinned variable's bound
//!   multiplier correctly signed (`max`/`min` over `−gradⱼ/coefⱼ`, clamped
//!   `≥ 0` for inequalities), and each pinned variable's bound multiplier
//!   is then its full reduced cost. The multiplier is generally *not
//!   unique* (it ranges over an interval), so postsolve emits a valid
//!   representative; correctness is checked as KKT validity, not dual
//!   equality (`tests/presolve_forcing.rs`). Forcing rows are required to
//!   have disjoint column sets so the recovery stays independent.
//!
//! # Relationship to PaPILO
//!
//! [PaPILO](https://github.com/scipopt/papilo) (Gleixner, Gottwald &
//! Hoen; the presolving library SCIP uses) is the reference architecture
//! for this module. It is C++ and Apache-2.0, so POUNCE does **not** wrap
//! it — that would break the pure-Rust guarantee — but ports its ideas:
//!
//! - the **transaction/reduction-stack** model with reversible postsolve
//!   (the [`Reduction`] enum + `stack` + [`Presolve::postsolve`]);
//! - **hashing-based pairing** for duplicate detection, parallelized
//!   (PaPILO uses Intel TBB; we use rayon).
//!
//! PaPILO is the catalog to mine for the next reductions — singleton /
//! doubleton rows, dominated columns, coefficient strengthening, probing
//! — and, importantly, for each one's *postsolve transform*, since the
//! dual recovery is the hard part.
//!
//! Implemented from that catalog so far: the transaction stack, fixed /
//! free / free-singleton columns, empty + duplicate rows, activity-based
//! redundancy/feasibility, and **forcing constraints** (above) — which
//! capture the dual-safe slice of activity/bound reasoning, since a
//! forcing row is exactly a model-changing bound deduction whose dual
//! re-attributes to the source row.
//!
//! - **Bound tightening** (domain propagation): each live row implies
//!   bounds on its variables (`a_k x_k ≤ h − amin_{−k}`, etc.); where one
//!   is strictly tighter than the declared box, the box is shrunk in the
//!   reduced problem (the variable is *kept*). The subtle dual — when a
//!   tightened bound is active at the optimum while the original bound is
//!   slack, its multiplier is not a real bound multiplier but belongs to
//!   the row that implied it — is handled in postsolve by **global bound
//!   recovery**: every row multiplier is recovered first (re-attributing
//!   each active tightened bound to its source row), then every variable's
//!   bound multipliers are read off the complete reduced cost by
//!   complementarity. To keep the re-attributions independent, tightening
//!   sources are restricted to column-disjoint rows untouched by other
//!   reductions (the same conservative rule as forcing). A single pass
//!   (not iterated to a fixpoint), validated by randomized KKT roundtrips
//!   (`tests/presolve_bound_tightening.rs`).
//!
//! The full deferred catalog — forcing constraints, parallel rows,
//! dominated columns, and bound tightening — is implemented, each with a
//! dual recovery proven correct (and KKT-validated in tests).
//!
//! [`presolve`] iterates the single-pass catalog ([`presolve_once`]) to a
//! **fixpoint**, so deductions cascade across rounds (a fixing exposes a
//! new singleton; a tightened bound makes a row forcing). Because each pass
//! is a correct solution-space transform, the iterate is their composition
//! and reuses every pass's proven dual recovery — no new dual math.
//!
//! This is also how the disjoint-source restriction on forcing / tightening
//! is *lifted*. Within one round, overlapping forcing / tightening sources
//! must stay column-disjoint so their dual re-attributions don't couple.
//! But the fixpoint resolves the overlap across rounds: a source claims its
//! columns only when it actually fires, so the round after it reaches its
//! own fixpoint it stops blocking its neighbours, which then fire — and the
//! *composed* postsolve recovers the shared variable's bound multiplier
//! with **both** rows' contributions present (each layer's global bound
//! recovery sees the inner layers' row multipliers mapped through). The
//! effect is a coupled re-attribution, achieved by composition rather than
//! a within-round coupled solve, and validated by randomized KKT roundtrips
//! over *overlapping* constraint chains
//! (`tests/presolve_bound_tightening.rs`).
//!
//! # What makes the fixpoint a fixpoint (gh #527)
//!
//! Resolving the overlap across rounds is also what makes the iteration able
//! to run away. Every reduction but one *consumes* what it acts on — a fixed
//! or substituted column and a dropped or aggregated row are gone from the
//! next round's problem — so those can fire at most `n + m` times in total
//! however many rounds there are. Bound tightening consumes nothing: the
//! column stays, the row stays, and only the box moves. Two equality rows
//! that mutually imply ever-tighter bounds on the same nonnegative variables
//! therefore converge *geometrically* toward a limit they never reach, and
//! every round finds a strictly tighter box, reports progress, and asks for
//! another one.
//!
//! Netlib `bore3d` does exactly that. Before #527 the loop exited on its
//! layer cap on every solve at every cap tried (32, 64, 200) and never once
//! on the fixpoint the module documents, which meant an arbitrary defensive
//! constant — not the algorithm — was choosing which of two different reduced
//! problems the solver was handed, with nothing anywhere to say so.
//!
//! Two things close that:
//!
//! - [`MAX_BOX_REFINEMENTS`] bounds how many times the iteration may refine
//!   the *same* box, so the one non-consuming reduction becomes finite like
//!   the rest and termination follows from the algorithm rather than from the
//!   cap. It is a count, never a magnitude, so no scale-dependent tolerance
//!   enters (which is what makes it safe where a minimum-improvement
//!   threshold is not: on this cascade the relative improvement is ~96% every
//!   round, so a relative test passes forever, and an absolute floor is
//!   exactly the sort of constant gh #523 came out of).
//! - [`FixpointExit`], recorded on [`PresolveStats::exit`], says which of the
//!   two exits happened. A reduction that came out of a truncated loop is no
//!   longer indistinguishable from one that came out of a fixpoint.
//!
//! With the budget in place and the cap lifted, `bore3d` reaches
//! `FixpointExit::Fixpoint` for the first time, in 238 layers.
//!
//! ## What that leaves, and why the cap still binds on `bore3d`
//!
//! 238 is not 32, and at the shipped cap that model still stops on the cap.
//! The reason is a *second*, independent mechanism, and it is worth naming so
//! the next person does not go looking for more cascade. Bound propagation is
//! **serialized**: a source row may claim a `(column, side)` only if no other
//! row already claimed it this pass, because that disjointness is what keeps
//! the dual re-attributions independent (see above). One round therefore
//! advances the propagation graph by roughly one edge per column — on
//! `bore3d`, about three tightenings per layer — and traversing it takes
//! hundreds of rounds no matter how well behaved each individual box is.
//! Variables are still receiving their *first* finite bound at layer 320.
//!
//! Measured, that extra depth buys nothing on this model: 32 layers and 330
//! layers reduce `bore3d` to the same 128 variables and 77 rows, with the
//! same 61 fixings, 17 forcing rows and 64 aggregations. Only the boxes
//! differ. So the cap is not currently costing a reduction here — but that it
//! *isn't* is now a measurement rather than an assumption, and the exit
//! reason says which case a given solve was in.

use crate::cones::ConeSpec;
use crate::qp::{BOUND_INF, QpProblem, QpSolution, QpStatus, Triplet};
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Outcome of presolve.
// `Reduced` carries the full reduced problem and is by far the common case;
// boxing it to shrink the two rare unit variants would just add an
// allocation + deref on the hot path and ripple through every caller's match.
#[allow(clippy::large_enum_variant)]
pub enum PresolveOutcome {
    /// Problem reduced; solve `reduced`, then call [`Presolve::postsolve`].
    Reduced(Presolve),
    /// Presolve proved the problem primal-infeasible (e.g. an empty row
    /// `0 = b` with `b ≠ 0`, contradictory fixed bounds, or duplicate
    /// equality rows with different right-hand sides). Carries the screen
    /// that fired and what it tripped on — see [`InfeasibleTrigger`].
    Infeasible(InfeasibleTrigger),
    /// Presolve proved the problem unbounded below (a free column with a
    /// nonzero objective coefficient).
    Unbounded,
}

/// Which screen concluded that the problem is primal-infeasible, and what it
/// tripped on.
///
/// A presolve infeasibility comes back in milliseconds with no iteration
/// behind it, so when it is wrong it is the most expensive failure the solver
/// has: a confident wrong answer with no trace. Carrying the trigger makes
/// the claim auditable — the CLI prints it, and a bug report names a row, a
/// column and the numbers that were compared instead of `iters=0` (gh #523).
#[derive(Debug, Clone, PartialEq)]
pub struct InfeasibleTrigger {
    /// The screen that fired, e.g. `"empty equality row"`.
    pub screen: &'static str,
    /// The row / column / bound it tripped on, with the compared values.
    pub detail: String,
}

impl std::fmt::Display for InfeasibleTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.screen, self.detail)
    }
}

/// Build a [`PresolveOutcome::Infeasible`] with its trigger recorded.
fn infeasible(screen: &'static str, detail: String) -> PresolveOutcome {
    PresolveOutcome::Infeasible(InfeasibleTrigger { screen, detail })
}

/// Why the fixpoint iteration stopped (gh #527).
///
/// Recorded because the two answers mean very different things and the loop
/// could not previously tell them apart. [`Self::Fixpoint`] is the documented
/// contract: no reduction in the catalog fires any more, and the reduced
/// problem is the one presolve actually converges to. [`Self::RoundCap`] is a
/// *truncation* — more reductions were still firing when the layer cap cut
/// them off, so the reduced problem handed to the solver is whatever the cap
/// happened to leave, and a different cap would have produced a different one.
///
/// That distinction went unseen through three releases on netlib `bore3d`,
/// where every solve exited on the cap at every cap and the constant was
/// silently choosing between two different reduced problems. Surfaced on
/// [`PresolveStats::exit`] so it is reportable rather than invisible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FixpointExit {
    /// A round found nothing left to reduce — the real fixpoint.
    #[default]
    Fixpoint,
    /// The `MAX_ROUNDS` layer cap stopped a loop that was still reducing.
    RoundCap,
}

impl std::fmt::Display for FixpointExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FixpointExit::Fixpoint => f.write_str("fixpoint"),
            FixpointExit::RoundCap => f.write_str("round cap"),
        }
    }
}

/// How many times the fixpoint may refine the *same* variable box before it
/// stops chasing that box (gh #527).
///
/// Bound propagation is the one reduction whose output can feed itself
/// forever. Two rows that mutually imply ever-tighter bounds on the same
/// nonnegative variables converge geometrically to a limit they never reach,
/// so every round finds a strictly tighter box, reports progress, and the
/// loop runs until something else stops it. On `bore3d` that something else
/// was `MAX_ROUNDS`, at every cap tried — the "fixpoint" was never reached
/// and the cap, not the algorithm, set how much reduction the solver got.
///
/// The budget is a **count, not a magnitude**: it never asks whether an
/// improvement is "big enough", so it introduces no scale-dependent constant
/// of the kind gh #523 came out of. A box that is genuinely being narrowed by
/// distinct deductions settles in a handful of refinements — the `bore3d`
/// cascade that triggered #523 needed seven — while a box being chased toward
/// an unreachable limit needs unboundedly many, and that is exactly what runs
/// out of budget.
///
/// Exhausting it costs at most a *tighter box*, never a wrong one: the bounds
/// already derived stay, and the variable is still handed to the solver with
/// every deduction made so far.
///
/// # Why twelve
///
/// Swept on `bore3d` with the layer cap lifted, so the budget is the only
/// thing ending the loop. `vars/rows` is the reduced problem it converges to;
/// unlimited refinement converges to `128/77` in 330 layers.
///
/// | budget | layers | vars/rows |
/// |---|---|---|
/// | 2 | — | 136/83 |
/// | 6 | — | 136/83 |
/// | **8** | 184 | **128/77** |
/// | 12 | 238 | 128/77 |
/// | 16 | 274 | 128/77 |
/// | 24 | 979 tightenings | 128/77 |
///
/// Eight is where the full reduction appears — the deepest box that unlocks a
/// structural reduction on this model is refined seven times before it does.
/// Twelve leaves a margin over that without buying anything past it, and the
/// cost of the margin is bounded: a box that has stopped yielding deductions
/// stops being refined on its own, long before the budget is reached.
const MAX_BOX_REFINEMENTS: u8 = 12;

/// Remaining [`MAX_BOX_REFINEMENTS`] per variable box side, carried across
/// fixpoint rounds.
///
/// Bound tightening is the only reduction that needs cross-round state. Every
/// other one consumes itself — a fixed column is gone, a dropped row is gone —
/// so a round that reruns the catalog cannot repeat it. Refining a box leaves
/// the column in place and the row live, so nothing but this stops the pair
/// from firing again next round.
struct BoxRefinements {
    ub: Vec<u8>,
    lb: Vec<u8>,
}

impl BoxRefinements {
    fn new(n: usize) -> Self {
        BoxRefinements {
            ub: vec![MAX_BOX_REFINEMENTS; n],
            lb: vec![MAX_BOX_REFINEMENTS; n],
        }
    }

    /// A budget that never runs out, for the single-pass callers ([`presolve_conic`])
    /// that do not iterate and so cannot cascade.
    fn unlimited(n: usize) -> Self {
        BoxRefinements {
            ub: vec![u8::MAX; n],
            lb: vec![u8::MAX; n],
        }
    }

    /// Columns this budget is sized for. A mismatch against the problem
    /// would make [`Self::allows`] answer `false` for the missing tail and
    /// silently switch bound tightening off — a constant deciding the
    /// reduction with nothing to say so, which is the failure class gh #527
    /// exists to remove. Asserted at each [`presolve_once`] entry.
    fn len(&self) -> usize {
        debug_assert_eq!(self.ub.len(), self.lb.len(), "budget sides diverged");
        self.ub.len()
    }

    /// May this pass still refine `col`'s upper (`is_upper`) or lower bound?
    fn allows(&self, col: usize, is_upper: bool) -> bool {
        let side = if is_upper { &self.ub } else { &self.lb };
        side.get(col).is_some_and(|&r| r > 0)
    }

    /// Charge `layer`'s bound tightenings against the budget. A layer records
    /// at most one [`Reduction::BoundTightening`] per `(column, side)`, so
    /// this is one decrement per box side actually refined.
    fn charge(&mut self, layer: &Presolve) {
        for r in &layer.stack {
            if let Reduction::BoundTightening { col, is_upper, .. } = *r {
                let side = if is_upper { &mut self.ub } else { &mut self.lb };
                if let Some(r) = side.get_mut(col) {
                    *r = r.saturating_sub(1);
                }
            }
        }
    }

    /// Renumber onto a layer's surviving columns (`kept_cols[new] = old`), so
    /// a box's remaining budget follows it across the eliminations that
    /// renumber the problem between rounds.
    fn remap(&mut self, kept_cols: &[usize]) {
        self.ub = kept_cols.iter().map(|&c| self.ub[c]).collect();
        self.lb = kept_cols.iter().map(|&c| self.lb[c]).collect();
    }
}

/// A reversible presolve transaction. Each variant stores exactly what
/// postsolve needs to reconstruct the eliminated primal and dual.
///
/// Dropped *rows* (empty rows, duplicate rows) need no stack entry: they
/// are simply absent from the kept-row maps, so postsolve leaves their
/// dual at the zero initialization, which is the correct multiplier.
enum Reduction {
    /// Variable `col` was fixed to `value` by the singleton equality row
    /// `eq_row` (coefficient `a_coef`). Postsolve restores `x[col] =
    /// value` and computes the row's multiplier from stationarity.
    FixedVar {
        col: usize,
        value: f64,
        eq_row: usize,
        a_coef: f64,
    },
    /// A column absent from `P`, `A`, `G` (linear-only) was fixed at
    /// `value` — its optimal box position given the sign of `c_col` —
    /// and dropped. Its reduced cost equals `c_col` (carried by the
    /// active variable bound).
    FreeColumnFixed { col: usize, value: f64 },
    /// A *free column singleton*: variable `col` is unbounded, absent
    /// from `P` and `G`, and appears in exactly one equality row
    /// `eq_row` (coefficient `a_coef`). It is substituted out via
    /// `x_col = (b_r − Σ_{j≠col} a_j x_j) / a_coef`, consuming the row.
    /// Postsolve recovers `x_col` from that expression and sets the
    /// consumed row's multiplier to the unique value `y_r = −c_col / a_coef`.
    FreeColSingleton {
        col: usize,
        eq_row: usize,
        a_coef: f64,
        /// `c_col`, used to recover `y_eq_row = −c_col / a_coef`.
        c_col: f64,
    },
    /// A **forcing constraint**: a row whose activity range touches its
    /// right-hand side, so the row can only hold at one vertex of the box,
    /// pinning every involved variable to a bound. The row is dropped and
    /// each variable fixed; postsolve recovers the row's multiplier and the
    /// pinned variables' bound multipliers (see [`Presolve::postsolve`]).
    ForcingRow {
        /// Original row index.
        row: usize,
        /// Equality row? (else inequality.)
        is_equality: bool,
        /// The forced-to vertex is the *max*-activity one (only possible
        /// for equalities); else the min-activity vertex.
        at_max: bool,
        /// Each pinned variable: `(col, coef, value, at_upper)`.
        cols: Vec<(usize, f64, f64, bool)>,
    },
    /// A **dominated column**: a variable absent from `P` and the
    /// equalities, appearing in inequalities `Gx ≤ h` with sign-definite
    /// coefficients that match the sign of its cost, so pushing it to one
    /// bound never hurts the objective *or* feasibility — it is optimal
    /// there. Fixed and dropped; its bound multiplier is its reduced cost,
    /// which the sign conditions make valid by construction (recovered in
    /// the global bound pass from where the variable lands).
    DominatedColumn { col: usize, value: f64 },
    /// A **tightened bound**: row `row` implies a bound on `col` strictly
    /// inside its declared box, so the box is shrunk in the reduced problem
    /// (the variable is *kept*, not removed). Postsolve handles the dual:
    /// if the tightened bound is active at the optimum while the original
    /// bound is slack, its multiplier is re-attributed to the source row
    /// (the multiplier on a non-real bound belongs to the constraint that
    /// implied it). See [`Presolve::postsolve`]'s global bound recovery.
    BoundTightening {
        col: usize,
        row: usize,
        is_equality: bool,
        /// Source-row coefficient `a_{row,col}`.
        coef: f64,
        /// Tightened the upper bound? (else lower.)
        is_upper: bool,
    },
    /// **Doubleton-equality aggregation** (gh #494): a batch of two-variable
    /// equality rows `a₁·x + a₂·y = b`, each folding one of its columns onto
    /// the other as `x = α·y + β` with no anchoring requirement, iterated to
    /// a fixed point so alias chains collapse. Planned by
    /// [`pounce_presolve::linear_eq_plan`] — shared with the general NLP
    /// path rather than restated — and applied to `P`, `c`, `A`, `G` by
    /// `crate::aggregate::reduce`.
    ///
    /// Unlike every other variant this one is a whole pass, not a single
    /// elimination: the plan's chains and its reverse-sweep dual recovery
    /// are only meaningful together. So an aggregation always forms a
    /// **layer of its own** in `Presolve::chain` whose stack is exactly
    /// this one entry, and `Presolve::postsolve_once` hands that layer
    /// straight to `crate::aggregate::postsolve`.
    Aggregate {
        plan: Box<pounce_presolve::linear_eq_plan::EliminationPlan>,
    },
}

/// Captured presolve state: the reduced problem plus the transaction
/// stack and the index maps needed to expand a reduced solution back to
/// the original space.
pub struct Presolve {
    /// The reduced problem to hand to the solver.
    pub reduced: QpProblem,
    /// Constant added to the objective by variable substitutions; the
    /// reduced objective plus this equals the original objective. For an
    /// iterated presolve this is the sum over `chain` — the constant between
    /// the *final* `reduced` problem and the user's, not one layer's share of
    /// it (gh #697).
    pub obj_offset: f64,
    /// Original problem dimensions.
    orig_n: usize,
    orig_m_eq: usize,
    orig_m_ineq: usize,
    /// `kept_cols[reduced_col] = orig_col`.
    kept_cols: Vec<usize>,
    /// `kept_eq[reduced_eq_row] = orig_eq_row`.
    kept_eq: Vec<usize>,
    /// `kept_ineq[reduced_ineq_row] = orig_ineq_row`.
    kept_ineq: Vec<usize>,
    /// Original problem data, retained for fixing-row dual recovery.
    orig: QpProblem,
    stack: Vec<Reduction>,
    /// For an *iterated* presolve, the ordered single-pass layers
    /// (`L0, L1, …`) whose composition this object represents; empty for a
    /// single pass. `reduced` is then the final layer's reduced problem and
    /// `postsolve` folds the layers in reverse. The single-pass fields
    /// above are unused in that case.
    chain: Vec<Presolve>,
    /// An infeasibility claim the full catalog raised that the confirming
    /// re-derivation would not reproduce, so it was discarded and this
    /// reduction returned instead (gh #523). `None` on the normal path. Read
    /// it with [`Presolve::discarded_infeasibility`].
    discarded_infeasibility: Option<InfeasibleTrigger>,
    /// Why the fixpoint iteration stopped (gh #527). Read it with
    /// [`PresolveStats::exit`]; `Fixpoint` for a single pass, which by
    /// definition did not run out of anything.
    exit: FixpointExit,
}

/// Coefficients are treated as nonzero unless exactly 0.0.
const ZERO_TOL: f64 = 0.0;
/// Slack allowed when checking a fixed value against its variable box.
const BOUND_FEAS_TOL: f64 = 1e-9;
/// Slack allowed in activity-bound comparisons (redundancy / feasibility).
const ACTIVITY_TOL: f64 = 1e-9;
/// How far a **forcing** reduction may be wrong about where it pins a
/// variable before the pin stops counting as a deduction (gh #523).
///
/// A forcing row's vertex is only ever recognized to within [`ACTIVITY_TOL`],
/// so a residual gap of that size survives the test — and the row can still
/// spend it. Spent on one variable it moves that variable `gap / |coefⱼ|` off
/// the bound the pin claims it must sit at, which for a small coefficient is
/// orders of magnitude more than the gap that licensed the pin. Matching
/// [`BOUND_FEAS_TOL`] keeps a pinned value inside the same window presolve
/// judges every other fixed value's box feasibility by. See
/// [`forcing_pin_is_tight`].
const FORCING_PIN_TOL: f64 = BOUND_FEAS_TOL;
/// Relative slack allowed when a row emptied by substitution is checked for
/// feasibility (`0 = rhs` / `0 ≤ rhs`). Scaled by the size of the terms that
/// cancelled, because that is where the residual's rounding error lives —
/// judging it against exact zero declares a redundant equality inconsistent
/// at one ULP (gh#496). Matches [`ACTIVITY_TOL`], the same feasibility
/// question asked of a row that kept its coefficients, and sits far above
/// accumulated `f64` cancellation yet far below any real conflict.
const EMPTY_ROW_TOL: f64 = 1e-9;
/// How close `x_i` must be to a box bound to count it *active* when
/// recovering bound multipliers. Looser than [`BOUND_FEAS_TOL`] because an
/// interior-point solve only drives a variable to within ~1e-8 of a bound,
/// not to machine zero; interior variables sit far further away.
///
/// Applied **relative to the bound's magnitude** — see [`at_bound`]. The
/// "~1e-8" above is a relative statement, and reading it as an absolute
/// window silently loses the multiplier of any bound bigger than about `1e4`.
pub(crate) const ACTIVE_BOUND_TOL: f64 = 1e-6;

/// Is `x` sitting *on* the bound `b`?
///
/// The window scales with the bound: an interior-point solve stops a relative
/// ~1e-8 short of a bound, which is an absolute `5e-3` when the bound is
/// `5e5`. Judged against a fixed `1e-6` such a variable reads as interior, and
/// every rule keyed on this — the global bound-multiplier recovery below, and
/// the tightened-bound re-attribution that reads its output — concludes the
/// bound is slack and reports its multiplier as **zero**. That is a wrong dual
/// on an ordinary model, not an edge case: `min x² − 4u·x` boxed at `x ≤ u`
/// lost its bound multiplier for every `u ≳ 1e4` while still reporting
/// `Optimal`, and it reaches `.sol` as `ipopt_zL_out`/`ipopt_zU_out` and as
/// the dual of any constraint row that became a bound.
///
/// Widening is safe: a genuinely interior variable sits far outside even the
/// scaled window, and both callers additionally require the reduced cost to
/// have the sign that bound could produce, so a misread would have to carry a
/// correctly-signed nonzero gradient to do any harm.
pub(crate) fn at_bound(x: f64, b: f64) -> bool {
    (x - b).abs() <= ACTIVE_BOUND_TOL * (1.0 + b.abs())
}

/// Group nonzero entries by row index: `out[row] = [(col, val), …]`.
pub(crate) fn group_by_row(triplets: &[Triplet], m: usize) -> Vec<Vec<(usize, f64)>> {
    let mut out = vec![Vec::new(); m];
    for t in triplets {
        if t.val != ZERO_TOL {
            out[t.row].push((t.col, t.val));
        }
    }
    out
}

/// Minimum and maximum of `Σ a_j x_j` over the variable box, given each
/// variable's effective lower/upper bound. An infinite contribution
/// makes the corresponding extreme `±∞`.
fn activity<L, U>(row: &[(usize, f64)], lb: &L, ub: &U) -> (f64, f64)
where
    L: Fn(usize) -> f64,
    U: Fn(usize) -> f64,
{
    let mut amin = 0.0;
    let mut amax = 0.0;
    for &(c, a) in row {
        let (lo, hi) = (lb(c), ub(c));
        if a > 0.0 {
            amin += a * lo; // a>0: min at lower bound
            amax += a * hi;
        } else {
            amin += a * hi; // a<0: min at upper bound
            amax += a * lo;
        }
    }
    (amin, amax)
}

/// Does a row whose activity range *touches* its right-hand side to within
/// [`ACTIVITY_TOL`] really **force** its variables onto the touched vertex?
///
/// `gap` is the leftover activity at the vertex (`b − amin` for a row pinned
/// to its min vertex, `amax − b` for the max vertex, `h − amin` for a `≤`
/// row); the forcing test accepted the row because `|gap| ≤ ACTIVITY_TOL`.
/// A *genuine* forcing row has `gap == 0`: the vertex is the only point of
/// the box the row admits, so pinning every variable there is a deduction.
/// A merely-near-zero gap is not. The row can still spend that activity, and
/// spending all of it on variable `j` moves it `gap / |coefⱼ|` off the bound
/// the pin claims it must occupy — capped by `j`'s own box width, since it
/// cannot leave its box either way.
///
/// With a small coefficient, or a box that bound tightening has already
/// narrowed toward rounding-noise width, that displacement runs orders of
/// magnitude past the gap that licensed it. The pin is then not a deduction
/// but a guess: it fixes variables at values the feasible set does not
/// require, and the next row those variables appear in reads as
/// contradictory — a **false primal infeasibility** on a feasible problem.
///
/// That is exactly gh #523 (netlib `bore3d` / Maros-Mészáros `QBORE3D`),
/// where a gap of `6.3e-10` on a row with coefficients `−1.14e-1` and
/// `−5.7e-3` pinned two variables `4.8e-9` and `1.5e-8` away from their true
/// values, and a later row emptied by those fixings was declared inconsistent
/// by `2.0e-8`.
///
/// So a forcing row must clear a stronger bar than "the gap is small": the
/// gap must be small enough that **every** pinned variable lands within
/// [`FORCING_PIN_TOL`] of where the pin says it is.
fn forcing_pin_is_tight<L, U>(row: &[(usize, f64)], gap: f64, lb: &L, ub: &U) -> bool
where
    L: Fn(usize) -> f64,
    U: Fn(usize) -> f64,
{
    // A gap the wrong side of zero means the vertex slightly *violates* the
    // right-hand side (the activity screen tolerates that much before calling
    // the row infeasible); there is no leftover activity to spend, so the
    // displacement it licenses is zero, not negative.
    let gap = gap.max(0.0);
    row.iter().all(|&(c, coef)| {
        // Coefficients reaching here are nonzero (`group_by_row` drops exact
        // zeros), so the division is safe.
        let displacement = (gap / coef.abs()).min(ub(c) - lb(c));
        displacement <= FORCING_PIN_TOL
    })
}

/// A single constraint row in the reduced column space, tagged with its
/// original row index. Used for duplicate detection and final assembly.
struct Row {
    /// `(reduced_col, value)` pairs, sorted by column, duplicates merged.
    coeffs: Vec<(usize, f64)>,
    rhs: f64,
    orig: usize,
    /// Magnitude of the largest term that went into `rhs` — the original
    /// `|b|` and every substituted `|aⱼ vⱼ|`. This is the scale the
    /// subtraction's rounding error lives on, so it is what an emptied
    /// row's residual must be judged against (see [`build_rows`]); a row
    /// that keeps coefficients never consults it.
    scale: f64,
}

/// Run presolve on `prob`, iterating the reduction passes to a **fixpoint**
/// so deductions cascade (a fixing exposes a new singleton, a tightened
/// bound makes a row forcing, …). Each pass is a correct solution-space
/// transform, so the iterate is the composition of the per-pass transforms
/// — postsolve folds them back in reverse — and inherits each pass's proven
/// dual recovery with no new dual math.
///
/// Each round is the single-pass catalog ([`presolve_once`]) followed by a
/// doubleton-equality aggregation ([`aggregate_once`], gh #494), which is
/// its own layer because its plan and dual recovery are only meaningful as
/// a unit. Both feed the same fixpoint: an aggregation can expose a
/// singleton the catalog then fixes, and a fixing can turn a three-term row
/// into a doubleton the next aggregation folds away.
///
/// # Infeasibility is re-derived before it is believed (gh #523)
///
/// A presolve infeasibility is the worst answer this solver can give when it
/// is wrong: `Infeasible_Problem_Detected` in five milliseconds, with no
/// iteration behind it, on a problem that has an optimum. And the catalog can
/// manufacture one. Two of its reductions — forcing constraints and dominated
/// columns — do not merely drop or rewrite constraints, they **fix a variable
/// at a value they chose from a tolerance judgment**. A fixing that is wrong
/// is substituted into every row the variable appears in, and the first row
/// that cannot absorb it reads as contradictory: a false infeasibility, in a
/// row nowhere near the reduction that caused it.
///
/// So an infeasibility verdict is not emitted on the strength of the pass
/// that raised it. It is **re-derived from the original problem with those
/// two reductions switched off** ([`Catalog::NoSpeculativeFixings`]), and
/// only a verdict the re-derivation reaches on its own is returned as
/// `Infeasible`. If the re-derivation instead reduces, the claim depended on
/// a fixing that was not forced, and the reduction it hands back is solved
/// normally — a misfiring reduction costs a few eliminations instead of the
/// answer. The discarded claim rides along on the returned handle
/// ([`Presolve::discarded_infeasibility`]) so the near-miss is reportable
/// rather than silent.
///
/// The re-derivation keeps everything else, including the screens that reason
/// through tolerances but only ever *report* — empty rows, activity ranges,
/// parallel rows, an emptied row's residual — so no infeasibility class the
/// catalog could detect before is lost.
///
/// # This entry point is orthant-only, and pointing it at a cone is a wrong
/// answer, not an error
///
/// Every inequality row here is taken to be an independent `gᵢx ≤ hᵢ`. A
/// problem carrying a second-order, exponential, power or PSD block must go
/// through [`presolve_conic`], which is handed the partition and protects it.
/// The failure mode is silent: on a conic problem this function returns a
/// perfectly well-formed reduced problem that encodes *different constraints*.
/// `crates/pounce-convex/tests/presolve_conic_quadratic_rows.rs` walks one —
/// a QCQP whose two quadratic rows share a linear part, which
/// `extract_socp_with_map` writes as byte-identical `G` rows in different
/// cones. The unprotected merge calls them duplicates, keeps one, and the
/// solve reports `Optimal` at an objective 67% off (gh #588 §7).
pub fn presolve(prob: &QpProblem) -> PresolveOutcome {
    match presolve_fixpoint(prob, Catalog::Full, DedupMemo::Enabled) {
        PresolveOutcome::Infeasible(trigger) => {
            match presolve_fixpoint(prob, Catalog::NoSpeculativeFixings, DedupMemo::Enabled) {
                // Confirmed without the speculative fixings: the problem
                // really is infeasible, and this is the trigger that shows it.
                confirmed @ PresolveOutcome::Infeasible(_) => confirmed,
                PresolveOutcome::Reduced(mut ps) => {
                    ps.discarded_infeasibility = Some(trigger);
                    PresolveOutcome::Reduced(ps)
                }
                unbounded => unbounded,
            }
        }
        other => other,
    }
}

/// Which reductions a presolve pass may apply.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Catalog {
    /// The whole catalog.
    Full,
    /// The catalog minus its **speculative fixings**: the two reductions that
    /// pick a *value* for a variable out of a tolerance judgment instead of
    /// deriving it — forcing constraints (pin every variable of a row whose
    /// activity range comes within [`ACTIVITY_TOL`] of touching its
    /// right-hand side) and dominated columns (fix a variable at a bound on
    /// sign conditions read off the rows the activity screen left live).
    ///
    /// Those two are singled out by *how* a wrong one fails. Every other
    /// reduction drops a constraint, narrows a box, or rewrites a row — being
    /// wrong loses a reduction. A wrong fixing instead substitutes a
    /// fabricated value into every row the variable appears in, and the first
    /// row that cannot absorb it reads as contradictory. That is the
    /// mechanism behind gh #523, and the one an infeasibility verdict is
    /// re-derived without; see [`presolve`].
    NoSpeculativeFixings,
}

impl Catalog {
    /// Whether the speculative fixings — forcing rows, dominated columns —
    /// may run.
    fn speculative_fixings(self) -> bool {
        self == Catalog::Full
    }
}

/// Whether the fixpoint may skip re-deriving a duplicate-row merge it has
/// already performed (gh #527).
///
/// [`DedupMemo::Disabled`] exists so a test can run the same problem both ways
/// and assert the reduced problems come out identical. The skip is the one
/// change in #527 that could alter a *reduction* rather than merely a box, and
/// its soundness was argued by reading; `dedup_memo_tests` makes it something
/// execution checks instead, so a reduction added later cannot quietly
/// invalidate the argument from elsewhere in the catalog.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DedupMemo {
    Enabled,
    /// Only ever constructed by `dedup_memo_tests` — that is the point of it,
    /// so a release build has nothing to warn about.
    #[cfg_attr(not(test), allow(dead_code))]
    Disabled,
}

/// The fixpoint iteration itself, over the reductions `catalog` permits.
///
/// # Why this terminates (gh #527)
///
/// Every reduction but one consumes what it acts on — a fixed or substituted
/// column leaves the problem, a dropped or aggregated row leaves the problem —
/// so those can fire at most `n + m` times in total across all rounds. Bound
/// tightening is the exception: it keeps the column and keeps the row, so the
/// same pair may fire every round forever, and on `bore3d` it did. A
/// [`BoxRefinements`] budget bounds that at `MAX_BOX_REFINEMENTS` per box
/// side, which caps bound-only rounds at `2n·MAX_BOX_REFINEMENTS`. With both
/// halves finite the loop reaches a genuine fixpoint on its own — `MAX_ROUNDS`
/// is a backstop against a bug, not the argument for why this stops.
///
/// It can still *bind* on a model whose propagation graph is deeper than the
/// cap, `bore3d` among them; what has changed is that the loop now says so
/// ([`FixpointExit`]) instead of leaving a truncated reduction and a fixpoint
/// looking identical.
fn presolve_fixpoint(prob: &QpProblem, catalog: Catalog, memo: DedupMemo) -> PresolveOutcome {
    // Cap layers defensively. A round can contribute two (the catalog pass
    // and an aggregation), so this is a bound on layers, not on rounds.
    //
    // Left at 32 deliberately (gh #527). It is no longer the termination
    // argument — that is `MAX_BOX_REFINEMENTS` — and it is no longer silent
    // when it binds, which is what made it dangerous. Raising it was measured
    // rather than assumed: with the structure-stable dedup skip below, 128
    // layers on `bore3d` cost ~16 ms against ~14 ms for 32 layers on `main`,
    // so it is affordable — but it reduced `bore3d` to exactly the same
    // problem (128 vars / 77 rows) that 32 layers reach, while multiplying
    // the chain's retained memory, since every layer holds a clone of its
    // input. A deeper search with no measured reduction to show for it and a
    // real memory cost is not a trade worth making blind.
    const MAX_ROUNDS: usize = 32;
    let mut chain: Vec<Presolve> = Vec::new();
    let mut current = prob.clone();
    // Per-box refinement allowance, carried across rounds and renumbered onto
    // each layer's surviving columns.
    let mut refinements = BoxRefinements::new(prob.n);
    // The passthrough layer from a first round that changed nothing, kept
    // so a no-op presolve still returns a usable handle.
    let mut passthrough: Option<Presolve> = None;
    // Whether the previous round left the coefficient structure alone (it
    // only narrowed boxes), which lets the next pass skip the duplicate-row
    // hashing. False on the first round: nothing has been deduped yet.
    let mut structure_stable = false;
    let exit;
    loop {
        let ps = match presolve_once(&current, &[], catalog, &refinements, structure_stable) {
            infeasible @ PresolveOutcome::Infeasible(_) => return infeasible,
            PresolveOutcome::Unbounded => return PresolveOutcome::Unbounded,
            PresolveOutcome::Reduced(ps) => ps,
        };
        let catalog_changed = ps.changed();
        // A round that only narrowed boxes leaves every row exactly as the
        // previous pass deduped it (gh #527).
        let bounds_only = ps.reduced.n == ps.orig_n
            && ps.reduced.m_eq() == ps.orig_m_eq
            && ps.reduced.m_ineq() == ps.orig_m_ineq
            && ps
                .stack
                .iter()
                .all(|r| matches!(r, Reduction::BoundTightening { .. }));
        if catalog_changed {
            refinements.charge(&ps);
            refinements.remap(&ps.kept_cols);
            current = ps.reduced.clone();
            chain.push(ps);
        } else if passthrough.is_none() {
            passthrough = Some(ps);
        }
        // Doubleton-equality aggregation (gh #494). It runs *after* the
        // catalog each round, so the cheap single-column reductions have
        // already shrunk the rows it plans over, and the round that follows
        // gets a crack at whatever the aggregation exposes.
        let aggregated = match aggregate_once(&current) {
            Some(ps) => {
                refinements.remap(&ps.kept_cols);
                current = ps.reduced.clone();
                chain.push(ps);
                true
            }
            None => false,
        };
        structure_stable = memo == DedupMemo::Enabled && bounds_only && !aggregated;
        if !catalog_changed && !aggregated {
            exit = FixpointExit::Fixpoint; // neither half found anything
            break;
        }
        if chain.len() >= MAX_ROUNDS {
            exit = FixpointExit::RoundCap;
            break;
        }
    }
    // A chain this short cannot have reached `MAX_ROUNDS`, so `exit` is
    // `Fixpoint` here and the single-pass layer already carries it.
    if chain.is_empty() {
        // Nothing to do at all; hand back the verbatim-forwarding layer.
        return PresolveOutcome::Reduced(
            passthrough.expect("a round that changed nothing yields a passthrough layer"),
        );
    }
    if chain.len() == 1 {
        let mut only = chain.pop().expect("chain has one layer");
        only.exit = exit;
        return PresolveOutcome::Reduced(only);
    }
    let reduced = chain.last().expect("chain non-empty").reduced.clone();
    // The constant each layer moved into its own objective, composed (gh
    // #697). Layer `Lₖ`'s objective is `Lₖ₊₁`'s plus `Lₖ₊₁.obj_offset`, so
    // telescoping the chain gives the constant between the *final* reduced
    // problem — the one this wrapper hands the solver — and the user's. Left
    // at `0.0`, the accessor reported "presolve moved no constant" for every
    // multi-layer reduction, which is the common case. No double-count in
    // `postsolve`: with a non-empty chain it folds the layers and never reads
    // this field.
    let obj_offset = chain.iter().map(|l| l.obj_offset).sum();
    PresolveOutcome::Reduced(Presolve {
        reduced,
        obj_offset,
        orig_n: prob.n,
        orig_m_eq: prob.m_eq(),
        orig_m_ineq: prob.m_ineq(),
        kept_cols: Vec::new(),
        kept_eq: Vec::new(),
        kept_ineq: Vec::new(),
        orig: prob.clone(),
        stack: Vec::new(),
        chain,
        discarded_infeasibility: None,
        exit,
    })
}

/// One doubleton-equality aggregation layer over `prob`, or `None` when
/// there is nothing to aggregate (gh #494).
///
/// Deliberately reached only from [`presolve`], never from
/// [`presolve_conic`]: a non-orthant cone block is a coupled, fixed-layout
/// set of rows over columns the substitution would rewrite, and
/// [`Presolve::reduced_cones`] reads the surviving partition off the kept
/// rows. The conic path therefore opts out of this reduction entirely, as
/// it already does of the fixpoint iteration.
fn aggregate_once(prob: &QpProblem) -> Option<Presolve> {
    let plan = crate::aggregate::plan(prob)?;
    let (reduced, obj_offset) = crate::aggregate::reduce(prob, &plan)?;
    Some(Presolve {
        reduced,
        obj_offset,
        orig_n: prob.n,
        orig_m_eq: prob.m_eq(),
        orig_m_ineq: prob.m_ineq(),
        kept_cols: plan.vars_kept.clone(),
        kept_eq: plan.rows_kept.clone(),
        kept_ineq: (0..prob.m_ineq()).collect(),
        orig: prob.clone(),
        stack: vec![Reduction::Aggregate {
            plan: Box::new(plan),
        }],
        chain: Vec::new(),
        discarded_infeasibility: None,
        exit: FixpointExit::Fixpoint,
    })
}

/// Cone-aware presolve for a problem whose inequality block is partitioned
/// by `cones`. Applies only the cone-safe reductions (equality singletons,
/// free columns / free-column singletons, fixed-variable substitution; and
/// the orthant `≤`-row reductions on the *nonnegative* blocks), leaving
/// every **non-orthant** cone row (second-order, exponential, power, PSD)
/// and the columns coupled to them untouched. A **single pass** (the
/// fixpoint loop is orthant-only), so the reduced cone partition is
/// recoverable from the kept rows — see [`Presolve::reduced_cones`].
///
/// `cones` must partition the **whole** inequality block. Rows past the end
/// of the partition are treated as orthant here — and then panic in
/// [`Presolve::reduced_cones`], which has no cone to attribute them to. The
/// assertion below names the contract at the call rather than leaving it to
/// an index panic three frames away; `run_convex_socp`, the only production
/// caller, satisfies it by construction (`extract_socp_with_map` emits the
/// nonnegative block and one cone per quadratic row, covering `G` exactly).
pub fn presolve_conic(prob: &QpProblem, cones: &[ConeSpec]) -> PresolveOutcome {
    debug_assert_eq!(
        cones.iter().map(ConeSpec::dim).sum::<usize>(),
        prob.m_ineq(),
        "the cone partition must cover every inequality row; an uncovered \
         tail is silently unprotected here and unattributable in reduced_cones"
    );
    // Protect the rows of every non-orthant cone. The orthant `≤`-row
    // reductions (empty-row infeasibility, activity-redundancy drop, forcing,
    // bound tightening, parallel/duplicate) are sound only for the nonnegative
    // orthant: a non-orthant cone row is coupled to its block, its `h<0` is
    // legal (e.g. `K_exp` contains points with a negative first coordinate),
    // and dropping any one row corrupts the block's fixed layout (3-row
    // exp/power, `svec` PSD) AND desyncs [`Presolve::reduced_cones`], which
    // assumes non-orthant blocks keep their full dimension. Marking only
    // `SecondOrder` (the old behavior) left exp/power/PSD rows exposed.
    let mut protected_row = vec![false; prob.m_ineq()];
    let mut row = 0;
    for spec in cones {
        let d = spec.dim();
        if !matches!(spec, ConeSpec::Nonneg(_)) {
            for r in row..row + d {
                if r < protected_row.len() {
                    protected_row[r] = true;
                }
            }
        }
        row += d;
    }
    // Same guard as [`presolve`] (gh #523): an infeasibility verdict is
    // re-derived without the speculative fixings before it is emitted, and
    // downgraded to that pass's reduction when it will not confirm.
    // A single pass cannot cascade its own bound tightening, so it needs no
    // refinement budget (gh #527); the conic path does not iterate.
    let refinements = BoxRefinements::unlimited(prob.n);
    match presolve_once(prob, &protected_row, Catalog::Full, &refinements, false) {
        PresolveOutcome::Infeasible(trigger) => {
            match presolve_once(
                prob,
                &protected_row,
                Catalog::NoSpeculativeFixings,
                &refinements,
                false,
            ) {
                confirmed @ PresolveOutcome::Infeasible(_) => confirmed,
                PresolveOutcome::Reduced(mut ps) => {
                    ps.discarded_infeasibility = Some(trigger);
                    PresolveOutcome::Reduced(ps)
                }
                unbounded => unbounded,
            }
        }
        other => other,
    }
}

/// A single presolve pass (the reduction catalog applied once). [`presolve`]
/// iterates this to a fixpoint.
///
/// `soc_row` (length `m_ineq`, or empty for the all-orthant QP path) marks
/// inequality rows that belong to a *non-orthant* cone (e.g. a second-order
/// cone). Such rows are coupled, so the `≤`-row reductions (empty-row,
/// activity, forcing, bound-tightening, parallel/duplicate) must not touch
/// them, and columns appearing in them are not eligible for the dominated-
/// column reduction. The cone-safe reductions (equality singletons, free
/// columns, free-column singletons, fixed-variable substitution) apply
/// regardless. Marked rows are never dropped, so the conic partition is
/// recoverable from the kept rows.
///
/// `catalog` selects which reductions run: [`Catalog::NoSpeculativeFixings`]
/// withholds the two that fix a variable at a tolerance-chosen value, leaving
/// the pass unable to manufacture a contradiction that way (gh #523).
///
/// `refinements` is the caller's per-box tightening allowance (gh #527). A box
/// side whose budget is spent is left alone by bound tightening — every other
/// reduction, this one's own already-derived bounds included, is unaffected.
///
/// `structure_stable` says the previous round left the coefficient structure
/// alone (it only narrowed boxes), which lets this pass skip re-deriving a
/// duplicate-row merge it already performed — see the memoization at the
/// `dedup_rows` calls. `false` is always safe; the iterated caller is the only
/// one that can ever pass `true`.
fn presolve_once(
    prob: &QpProblem,
    soc_row: &[bool],
    catalog: Catalog,
    refinements: &BoxRefinements,
    structure_stable: bool,
) -> PresolveOutcome {
    let n = prob.n;
    debug_assert_eq!(
        refinements.len(),
        n,
        "refinement budget must be sized for the problem it gates; a short \
         budget silently disables bound tightening on the missing columns"
    );
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    // gh #295: a *present* lower bound at `+∞` or upper bound at `−∞` admits no
    // finite point — the same primal-infeasible class as a finite reversed box
    // (`tlb[k] > tub[k]`, detected during bound tightening below). Catch it up
    // front so an impossible bound is never mistaken for an *absent* one (the
    // normal one-sided `±∞` encoding, which stays feasible). See
    // [`QpProblem::bounds_admit_no_point`].
    if let Some(k) = (0..n).find(|&i| prob.lb_of(i) >= BOUND_INF || prob.ub_of(i) <= -BOUND_INF) {
        debug_assert!(prob.bounds_admit_no_point());
        return infeasible(
            "impossible variable bound",
            format!(
                "column {k} has lb={:e}, ub={:e}",
                prob.lb_of(k),
                prob.ub_of(k)
            ),
        );
    }
    let is_soc_row = |i: usize| soc_row.get(i).copied().unwrap_or(false);
    // A column is conic-coupled if it appears in any SOC inequality row.
    let mut soc_col = vec![false; n];
    if !soc_row.is_empty() {
        for t in &prob.g {
            if is_soc_row(t.row) && t.val != ZERO_TOL {
                soc_col[t.col] = true;
            }
        }
    }

    let mut stack: Vec<Reduction> = Vec::new();

    // --- per-row / per-column nonzero structure ---
    let mut eq_nnz = vec![0usize; m_eq];
    let mut eq_single: Vec<Option<(usize, f64)>> = vec![None; m_eq];
    // Finer per-column appearance counts: total (`col_nnz`), and split
    // by where the variable appears, so we can recognize a free *column
    // singleton* (a variable in exactly one equality row, nowhere else).
    let mut col_nnz = vec![0usize; n];
    let mut a_col_count = vec![0usize; n];
    let mut g_col_count = vec![0usize; n];
    let mut p_col_present = vec![false; n];
    // For a column singleton: which equality row holds it, with coef.
    let mut col_eq_single: Vec<Option<(usize, f64)>> = vec![None; n];
    for t in &prob.a {
        if t.val != ZERO_TOL {
            eq_nnz[t.row] += 1;
            eq_single[t.row] = Some((t.col, t.val));
            col_nnz[t.col] += 1;
            a_col_count[t.col] += 1;
            col_eq_single[t.col] = Some((t.row, t.val));
        }
    }
    let mut ineq_nnz = vec![0usize; m_ineq];
    for t in &prob.g {
        if t.val != ZERO_TOL {
            ineq_nnz[t.row] += 1;
            col_nnz[t.col] += 1;
            g_col_count[t.col] += 1;
        }
    }
    for t in &prob.p_lower {
        if t.val != ZERO_TOL {
            col_nnz[t.row] += 1;
            p_col_present[t.row] = true;
            if t.row != t.col {
                col_nnz[t.col] += 1;
                p_col_present[t.col] = true;
            }
        }
    }

    // --- empty equality rows + singleton-equality fixings ---
    let mut fixed: Vec<Option<f64>> = vec![None; n];
    let mut eq_dropped = vec![false; m_eq];
    for row in 0..m_eq {
        match eq_nnz[row] {
            0 => {
                if prob.b[row] != 0.0 {
                    return infeasible(
                        "empty equality row",
                        format!("equality row {row} is `0 = {:e}`", prob.b[row]),
                    );
                }
                eq_dropped[row] = true;
            }
            1 => {
                let (col, a) = eq_single[row].expect("singleton has an entry");
                if fixed[col].is_none() {
                    let value = prob.b[row] / a;
                    // The fixed value must satisfy the variable's box.
                    if value < prob.lb_of(col) - BOUND_FEAS_TOL
                        || value > prob.ub_of(col) + BOUND_FEAS_TOL
                    {
                        return infeasible(
                            "singleton equality outside variable box",
                            format!(
                                "equality row {row} fixes column {col} to {value:e}, \
                                 outside its box [{:e}, {:e}]",
                                prob.lb_of(col),
                                prob.ub_of(col)
                            ),
                        );
                    }
                    fixed[col] = Some(value);
                    eq_dropped[row] = true;
                    stack.push(Reduction::FixedVar {
                        col,
                        value,
                        eq_row: row,
                        a_coef: a,
                    });
                }
            }
            _ => {}
        }
    }

    // --- free column singletons ---
    // A free variable (unbounded both ways), absent from P and G, that
    // appears in exactly one equality row whose row has ≥ 2 nonzeros, is
    // substituted out: `x_col = (b_r − Σ_{j≠col} a_j x_j) / a_col`. This
    // consumes both the variable and the row. The substitution shifts the
    // cost of the row's other variables (`c_adjust`) and a constant into
    // the objective offset; the consumed row's dual is the unique value
    // `−c_col / a_col`, recovered in postsolve.
    let mut substituted = vec![false; n];
    let mut c_adjust = vec![0.0; n];
    let mut subst_offset = 0.0;
    for col in 0..n {
        if fixed[col].is_some() || substituted[col] {
            continue;
        }
        let free = prob.lb_of(col) <= -BOUND_INF && prob.ub_of(col) >= BOUND_INF;
        let only_in_one_eq = a_col_count[col] == 1 && g_col_count[col] == 0 && !p_col_present[col];
        if !(free && only_in_one_eq) {
            continue;
        }
        let (row, a_col) = col_eq_single[col].expect("column singleton entry");
        // The row must still be live and non-trivial (≥ 2 vars: a plain
        // singleton row was already turned into a FixedVar above).
        if eq_dropped[row] || eq_nnz[row] < 2 {
            continue;
        }
        // Substitute: c_col·x_col = (c_col·b_r/a_col) − Σ_{j≠col}
        // (c_col·a_jr/a_col)·x_j.
        let c_col = prob.c[col];
        subst_offset += c_col * prob.b[row] / a_col;
        for t in &prob.a {
            if t.row == row && t.col != col && t.val != ZERO_TOL {
                c_adjust[t.col] -= c_col * t.val / a_col;
            }
        }
        substituted[col] = true;
        eq_dropped[row] = true;
        stack.push(Reduction::FreeColSingleton {
            col,
            eq_row: row,
            a_coef: a_col,
            c_col,
        });
    }

    // --- empty inequality rows ---
    // (SOC rows are coupled — an "empty" SOC row is part of a cone block and
    // must be kept; skip.)
    let mut ineq_dropped = vec![false; m_ineq];
    for row in 0..m_ineq {
        if !is_soc_row(row) && ineq_nnz[row] == 0 {
            if prob.h[row] < 0.0 {
                return infeasible(
                    "empty inequality row",
                    format!("inequality row {row} is `0 <= {:e}`", prob.h[row]),
                );
            }
            ineq_dropped[row] = true;
        }
    }

    // --- activity-bound reductions (need the variable box) ---
    // Effective bounds: a fixed variable contributes its exact value;
    // others contribute their declared box (±∞ when absent).
    let eff_lb = |c: usize| fixed[c].unwrap_or_else(|| prob.lb_of(c));
    let eff_ub = |c: usize| fixed[c].unwrap_or_else(|| prob.ub_of(c));

    // Group nonzeros by row once, reused for inequalities and equalities.
    let g_by_row = group_by_row(&prob.g, m_ineq);
    let a_by_row = group_by_row(&prob.a, m_eq);

    // Inequality `g·x ≤ h`:
    //   max-activity ≤ h  ⇒ redundant (always satisfied) → drop;
    //   min-activity > h   ⇒ infeasible.
    for row in 0..m_ineq {
        if ineq_dropped[row] || is_soc_row(row) {
            continue;
        }
        let (amin, amax) = activity(&g_by_row[row], &eff_lb, &eff_ub);
        if amin > prob.h[row] + ACTIVITY_TOL {
            return infeasible(
                "inequality activity above right-hand side",
                format!(
                    "inequality row {row}: min activity {amin:e} > h {:e}",
                    prob.h[row]
                ),
            );
        }
        if amax <= prob.h[row] + ACTIVITY_TOL {
            ineq_dropped[row] = true;
        }
    }

    // Equality `a·x = b`: feasible only if `b` lies in the activity
    // range `[min, max]`. Out of range ⇒ infeasible. (A redundant
    // equality whose range is the single point `b` is left in place; its
    // dual is genuine, unlike a dropped inequality's zero multiplier.)
    for row in 0..m_eq {
        if eq_dropped[row] {
            continue;
        }
        let (amin, amax) = activity(&a_by_row[row], &eff_lb, &eff_ub);
        if prob.b[row] < amin - ACTIVITY_TOL || prob.b[row] > amax + ACTIVITY_TOL {
            return infeasible(
                "equality right-hand side outside activity range",
                format!(
                    "equality row {row}: b {:e} outside [{amin:e}, {amax:e}]",
                    prob.b[row]
                ),
            );
        }
    }

    // --- forcing constraints ---
    // A row whose activity range touches its RHS can hold only at one
    // vertex of the box, pinning every involved variable to a bound:
    //   inequality g·x ≤ h with min-activity == h  ⇒ pin to the min vertex;
    //   equality   a·x = b with min-activity == b  ⇒ pin to the min vertex;
    //   equality   a·x = b with max-activity == b  ⇒ pin to the max vertex.
    // Each pinned variable becomes fixed (substituted out like any fixed
    // var); the row is dropped. Dual recovery (the reason this is subtle)
    // is handled in postsolve. We require each forcing row's columns to be
    // disjoint from every other forcing row's, so the multiplier recovery
    // stays independent (a conservative but always-correct restriction).
    let eff_lb_at = |fixed: &[Option<f64>], c: usize| fixed[c].unwrap_or_else(|| prob.lb_of(c));
    let eff_ub_at = |fixed: &[Option<f64>], c: usize| fixed[c].unwrap_or_else(|| prob.ub_of(c));
    let mut forced_touched = vec![false; n];

    // Pin the variables of one forcing row to `at_max` vertex (or the min
    // vertex when `at_max` is false), recording the reduction. Returns
    // false (skipped) if any column is already fixed/substituted/forced.
    // `row_entries` is the row's `(col, coef)` list, all coefficients nonzero.
    let try_force = |row_entries: &[(usize, f64)],
                     orig_row: usize,
                     is_equality: bool,
                     at_max: bool,
                     fixed: &mut [Option<f64>],
                     forced_touched: &mut [bool],
                     stack: &mut Vec<Reduction>|
     -> bool {
        // Every involved column must be free to fix and not shared with
        // another forcing row.
        for &(c, _) in row_entries {
            if fixed[c].is_some() || substituted[c] || forced_touched[c] {
                return false;
            }
        }
        let mut cols = Vec::with_capacity(row_entries.len());
        for &(c, coef) in row_entries {
            // Vertex bound: min-activity puts coef>0 at lb, coef<0 at
            // ub; max-activity is the mirror.
            let at_upper = if at_max { coef > 0.0 } else { coef < 0.0 };
            let value = if at_upper {
                prob.ub_of(c)
            } else {
                prob.lb_of(c)
            };
            // A forcing vertex requires finite bounds; guard anyway.
            if !value.is_finite() || value.abs() >= BOUND_INF {
                return false;
            }
            cols.push((c, coef, value, at_upper));
        }
        for &(c, _, value, _) in &cols {
            fixed[c] = Some(value);
            forced_touched[c] = true;
        }
        stack.push(Reduction::ForcingRow {
            row: orig_row,
            is_equality,
            at_max,
            cols,
        });
        true
    };

    // A forcing pin is a speculative fixing, so a confirming pass withholds it
    // (gh #523) — an empty range rather than a wrapper block, to keep the two
    // loops below at their original indentation.
    let (forcing_ineq, forcing_eq) = if catalog.speculative_fixings() {
        (m_ineq, m_eq)
    } else {
        (0, 0)
    };

    for row in 0..forcing_ineq {
        if ineq_dropped[row] || is_soc_row(row) || g_by_row[row].is_empty() {
            continue;
        }
        let (amin, _) = activity(&g_by_row[row], &|c| eff_lb_at(&fixed, c), &|c| {
            eff_ub_at(&fixed, c)
        });
        if amin.is_finite()
            && (prob.h[row] - amin).abs() <= ACTIVITY_TOL
            && forcing_pin_is_tight(
                &g_by_row[row],
                prob.h[row] - amin,
                &|c| eff_lb_at(&fixed, c),
                &|c| eff_ub_at(&fixed, c),
            )
            && try_force(
                &g_by_row[row],
                row,
                false,
                false,
                &mut fixed,
                &mut forced_touched,
                &mut stack,
            )
        {
            ineq_dropped[row] = true;
        }
    }

    for row in 0..forcing_eq {
        if eq_dropped[row] || a_by_row[row].len() < 2 {
            continue;
        }
        let (amin, amax) = activity(&a_by_row[row], &|c| eff_lb_at(&fixed, c), &|c| {
            eff_ub_at(&fixed, c)
        });
        let b = prob.b[row];
        // Both vertices are candidates; each must clear the pin-tightness bar
        // (gh #523) on its own gap before it may fix anything.
        let tight = |gap: f64| {
            forcing_pin_is_tight(&a_by_row[row], gap, &|c| eff_lb_at(&fixed, c), &|c| {
                eff_ub_at(&fixed, c)
            })
        };
        let at_max = if amin.is_finite() && (b - amin).abs() <= ACTIVITY_TOL && tight(b - amin) {
            Some(false)
        } else if amax.is_finite() && (amax - b).abs() <= ACTIVITY_TOL && tight(amax - b) {
            Some(true)
        } else {
            None
        };
        if let Some(at_max) = at_max {
            if try_force(
                &a_by_row[row],
                row,
                true,
                at_max,
                &mut fixed,
                &mut forced_touched,
                &mut stack,
            ) {
                eq_dropped[row] = true;
            }
        }
    }

    // --- dominated columns ---
    // A variable absent from P and the equalities, present only in
    // inequalities `Gx ≤ h`, whose live G-coefficients are sign-definite in
    // a way that matches its cost sign, is optimal at a bound: pushing it
    // there never raises the objective nor tightens a `≤` row, so an
    // optimal solution with it at that bound always exists. Fix and drop
    // it. Its bound multiplier is its reduced cost `c_k + Σᵢ aᵢₖ zᵢ`, which
    // the sign conditions (`aᵢₖ ≥ 0, c_k ≥ 0` for the lower bound; mirror
    // for the upper) make nonnegative — so the recovered dual is valid by
    // construction. This is PaPILO's dominated-column reduction, restricted
    // to the case with a clean, sign-guaranteed dual.
    //
    // It reads `ineq_dropped`, which the activity screen writes, so it can fix
    // a variable at a bound that a row dropped as redundant would have
    // forbidden: a speculative fixing, withheld by a confirming pass.
    if catalog.speculative_fixings() {
        // Per-column G-coefficient sign summary over *live* inequality rows.
        let mut g_all_nonneg = vec![true; n];
        let mut g_all_nonpos = vec![true; n];
        for t in &prob.g {
            if t.val == ZERO_TOL || ineq_dropped[t.row] {
                continue;
            }
            if t.val < 0.0 {
                g_all_nonneg[t.col] = false;
            } else if t.val > 0.0 {
                g_all_nonpos[t.col] = false;
            }
        }
        for col in 0..n {
            if fixed[col].is_some()
                || substituted[col]
                || p_col_present[col]
                || a_col_count[col] != 0
                || g_col_count[col] == 0
                || soc_col[col]
            {
                continue;
            }
            let c_k = prob.c[col];
            let lb = prob.lb_of(col);
            let ub = prob.ub_of(col);
            if g_all_nonneg[col] && c_k >= 0.0 && lb > -BOUND_INF {
                fixed[col] = Some(lb);
                stack.push(Reduction::DominatedColumn { col, value: lb });
            } else if g_all_nonpos[col] && c_k <= 0.0 && ub < BOUND_INF {
                fixed[col] = Some(ub);
                stack.push(Reduction::DominatedColumn { col, value: ub });
            }
        }
    }

    // --- bound tightening (domain propagation, single pass) ---
    // From each live row, derive implied bounds on its variables and shrink
    // the box where strictly tighter. The variable is *kept* (only its box
    // changes); the subtle dual — re-attributing an active tightened
    // bound's multiplier to the source row — is handled by postsolve's
    // global bound recovery. A single pass (not iterated to a fixpoint),
    // so it tightens but does not cascade into further reductions here.
    let mut tlb: Vec<f64> = (0..n).map(|c| prob.lb_of(c)).collect();
    let mut tub: Vec<f64> = (0..n).map(|c| prob.ub_of(c)).collect();
    for c in 0..n {
        if let Some(v) = fixed[c] {
            tlb[c] = v;
            tub[c] = v;
        }
    }
    // Source row (and its coef / kind) of each variable's tightened bound.
    let mut ub_src: Vec<Option<(usize, f64, bool)>> = vec![None; n];
    let mut lb_src: Vec<Option<(usize, f64, bool)>> = vec![None; n];

    // Re-attributing an active tightened bound's multiplier to its source
    // row is only *independent* when no two source rows can claim the same
    // bound; otherwise the re-attributions couple. So a row may serve as a
    // tightening source only if all its columns are kept (not
    // fixed/substituted) and none of the bounds it would claim has already
    // been taken — the same disjointness rule forcing uses.
    //
    // The claim is per **column** for a row with more than one column, and
    // per `(column, side)` for a **singleton** row. The split matters because
    // a variable bound written as a pair of rows — `x ≤ u` and `−x ≤ −l`,
    // which is how a model reaches here whenever its bounds were not carried
    // in the box — puts both rows on the same single column. Under a
    // whole-column rule the first row claimed the column and the second was
    // skipped, so only one side of that box was ever derived and a
    // *contradictory* pair (`u < l`) could not be seen at all. It reached the
    // solver as two rows whose infeasibility had to be certified numerically,
    // which the interior-point method managed at most widths but not around
    // `1e-8`, where it returned `NumericalFailure` at a `NaN` iterate.
    //
    // For a singleton the relaxation is sound, and only for a singleton:
    // postsolve credits the source row with `z_ub[col]` or `z_lb[col]`, at
    // most one of which complementarity allows to be nonzero, and the
    // credited row has no *other* column whose stationarity that credit could
    // disturb. A multi-column row does — its `Gᵀz` contribution lands in
    // every column it touches — so it keeps the conservative whole-column
    // claim. (Measured: relaxing it there breaks
    // `randomized_overlapping_tightening_roundtrip`, which chains two-column
    // rows and finds a postsolved point with a nonzero reduced cost at an
    // interior variable.)
    let reduction_touched: Vec<bool> = (0..n)
        .map(|c| fixed[c].is_some() || substituted[c])
        .collect();
    let mut bt_used_upper = vec![false; n];
    let mut bt_used_lower = vec![false; n];
    // Which `(column, side)` bounds a row claims. A singleton inequality
    // `a·x_c ≤ h` implies one bound, and which side is fixed by the sign of
    // `a` (`a > 0` bounds `x_c` above, `a < 0` below); everything else claims
    // both sides of every column it touches.
    let row_claims = |entries: &[(usize, f64)], is_eq: bool| -> Vec<(usize, bool)> {
        if !is_eq && entries.len() == 1 {
            let (c, a) = entries[0];
            return vec![(c, a > 0.0)];
        }
        entries
            .iter()
            .flat_map(|&(c, _)| [(c, true), (c, false)])
            .collect()
    };
    let row_is_clean =
        |entries: &[(usize, f64)], is_eq: bool, used_upper: &[bool], used_lower: &[bool]| {
            entries.iter().all(|&(c, _)| !reduction_touched[c])
                && row_claims(entries, is_eq)
                    .into_iter()
                    .all(|(c, up)| !if up { used_upper[c] } else { used_lower[c] })
        };

    // Tighten variable boxes from one row whose activity lies in `[lo, hi]`
    // (inequality `≤ h`: `lo = −∞, hi = h`; equality: `lo = hi = b`).
    // `Err((col, lb, ub))` ⇒ a detected empty domain (infeasible), naming the
    // column whose box crossed; `Ok(k)` ⇒ `k` bounds were tightened.
    let tighten_from_row = |entries: &[(usize, f64)],
                            lo: f64,
                            hi: f64,
                            row_idx: usize,
                            is_eq: bool,
                            tlb: &mut [f64],
                            tub: &mut [f64],
                            ub_src: &mut [Option<(usize, f64, bool)>],
                            lb_src: &mut [Option<(usize, f64, bool)>]|
     -> Result<usize, (usize, f64, f64)> {
        // Row activity as a *finite part plus a count of infinite terms*,
        // rather than the plain sum [`activity`] returns.
        //
        // The implied bound on column `k` needs the activity of the row
        // **without** `k`, and subtracting `k`'s own contribution from a total
        // is wrong the moment that contribution is the infinite one: `−∞ −
        // (−∞)` is `NaN`, `val` came out `NaN`, `val.is_finite()` was false,
        // and the tightening silently did nothing. That is not a rare corner —
        // it fired for *every* column of any row holding a variable unbounded
        // on the relevant side, including the singleton row `x ≤ u`, which is
        // nothing but a bound and should always imply one. Counting instead
        // makes the leave-one-out exact: drop `k`'s term from whichever
        // accumulator held it, and the rest is infinite only if some *other*
        // column is.
        let (mut min_finite, mut max_finite) = (0.0_f64, 0.0_f64);
        let (mut min_infs, mut max_infs) = (0usize, 0usize);
        let contrib = |k: usize, a: f64, tlb: &[f64], tub: &[f64]| -> (f64, f64) {
            if a == 0.0 {
                (0.0, 0.0)
            } else if a > 0.0 {
                (a * tlb[k], a * tub[k])
            } else {
                (a * tub[k], a * tlb[k])
            }
        };
        for &(c, a) in entries {
            let (cmin, cmax) = contrib(c, a, tlb, tub);
            if cmin.is_finite() {
                min_finite += cmin;
            } else {
                min_infs += 1;
            }
            if cmax.is_finite() {
                max_finite += cmax;
            } else {
                max_infs += 1;
            }
        }
        // Compute all implied bounds against the row-start state, then
        // apply (so within-row order doesn't matter).
        let mut updates: Vec<(usize, bool, f64, f64)> = Vec::new(); // (col,is_upper,val,coef)
        for &(k, a) in entries {
            if fixed[k].is_some() || a == 0.0 {
                continue;
            }
            let (contrib_min, contrib_max) = contrib(k, a, tlb, tub);
            // An infinite *min* contribution is `−∞` and an infinite *max* one
            // is `+∞`, so a leftover infinity has a known sign.
            let amin_mk = if min_infs > usize::from(!contrib_min.is_finite()) {
                f64::NEG_INFINITY
            } else {
                min_finite
                    - if contrib_min.is_finite() {
                        contrib_min
                    } else {
                        0.0
                    }
            };
            let amax_mk = if max_infs > usize::from(!contrib_max.is_finite()) {
                f64::INFINITY
            } else {
                max_finite
                    - if contrib_max.is_finite() {
                        contrib_max
                    } else {
                        0.0
                    }
            };
            if hi.is_finite() {
                let val = (hi - amin_mk) / a;
                if val.is_finite() {
                    if a > 0.0 {
                        if val < tub[k] - BOUND_FEAS_TOL {
                            updates.push((k, true, val, a));
                        }
                    } else if val > tlb[k] + BOUND_FEAS_TOL {
                        updates.push((k, false, val, a));
                    }
                }
            }
            if lo.is_finite() {
                let val = (lo - amax_mk) / a;
                if val.is_finite() {
                    if a > 0.0 {
                        if val > tlb[k] + BOUND_FEAS_TOL {
                            updates.push((k, false, val, a));
                        }
                    } else if val < tub[k] - BOUND_FEAS_TOL {
                        updates.push((k, true, val, a));
                    }
                }
            }
        }
        let mut tightened = 0usize;
        for (k, is_upper, val, a) in updates {
            // Out of refinements for this box side (gh #527): leave the bound
            // where the earlier rounds derived it. Skipping *before* the
            // update also leaves the row unclaimed when nothing else in it
            // tightened, so a spent box does not block its neighbours.
            if !refinements.allows(k, is_upper) {
                continue;
            }
            if is_upper {
                if val < tub[k] - BOUND_FEAS_TOL {
                    tub[k] = val;
                    ub_src[k] = Some((row_idx, a, is_eq));
                    tightened += 1;
                }
            } else if val > tlb[k] + BOUND_FEAS_TOL {
                tlb[k] = val;
                lb_src[k] = Some((row_idx, a, is_eq));
                tightened += 1;
            }
            if tlb[k] > tub[k] + BOUND_FEAS_TOL {
                return Err((k, tlb[k], tub[k]));
            }
        }
        Ok(tightened)
    };

    // A source row claims its columns (blocking overlapping sources, so the
    // re-attributions stay independent) only when it *actually* tightens —
    // a clean row that tightens nothing must not block its neighbours, or a
    // pair of overlapping rows where only one is useful would deadlock
    // across fixpoint rounds. With this, the fixpoint progressively fires
    // overlapping tightenings (each round the previous round's sources are
    // at their fixpoint and no longer claim columns).
    for row in 0..m_ineq {
        if ineq_dropped[row]
            || is_soc_row(row)
            || g_by_row[row].is_empty()
            || !row_is_clean(&g_by_row[row], false, &bt_used_upper, &bt_used_lower)
        {
            continue;
        }
        match tighten_from_row(
            &g_by_row[row],
            f64::NEG_INFINITY,
            prob.h[row],
            row,
            false,
            &mut tlb,
            &mut tub,
            &mut ub_src,
            &mut lb_src,
        ) {
            Err((c, lo, hi)) => {
                return infeasible(
                    "bound tightening emptied a variable box",
                    format!("inequality row {row} tightened column {c} to [{lo:e}, {hi:e}]"),
                );
            }
            Ok(0) => {}
            Ok(_) => {
                for (c, up) in row_claims(&g_by_row[row], false) {
                    if up {
                        bt_used_upper[c] = true;
                    } else {
                        bt_used_lower[c] = true;
                    }
                }
            }
        }
    }
    for row in 0..m_eq {
        if eq_dropped[row]
            || a_by_row[row].is_empty()
            || !row_is_clean(&a_by_row[row], true, &bt_used_upper, &bt_used_lower)
        {
            continue;
        }
        let b = prob.b[row];
        match tighten_from_row(
            &a_by_row[row],
            b,
            b,
            row,
            true,
            &mut tlb,
            &mut tub,
            &mut ub_src,
            &mut lb_src,
        ) {
            Err((c, lo, hi)) => {
                return infeasible(
                    "bound tightening emptied a variable box",
                    format!("equality row {row} tightened column {c} to [{lo:e}, {hi:e}]"),
                );
            }
            Ok(0) => {}
            Ok(_) => {
                for (c, up) in row_claims(&a_by_row[row], true) {
                    if up {
                        bt_used_upper[c] = true;
                    } else {
                        bt_used_lower[c] = true;
                    }
                }
            }
        }
    }

    // Record a reduction for each variable whose box was strictly tightened.
    for k in 0..n {
        if fixed[k].is_some() {
            continue;
        }
        if tub[k] < prob.ub_of(k) - BOUND_FEAS_TOL {
            if let Some((row, coef, is_eq)) = ub_src[k] {
                stack.push(Reduction::BoundTightening {
                    col: k,
                    row,
                    is_equality: is_eq,
                    coef,
                    is_upper: true,
                });
            }
        }
        if tlb[k] > prob.lb_of(k) + BOUND_FEAS_TOL {
            if let Some((row, coef, is_eq)) = lb_src[k] {
                stack.push(Reduction::BoundTightening {
                    col: k,
                    row,
                    is_equality: is_eq,
                    coef,
                    is_upper: false,
                });
            }
        }
    }

    // --- free / linear-only columns ---
    // A column absent from P, A, G contributes only `c_k x_k`, so its
    // optimum is at a bound dictated by the sign of c_k:
    //   c_k > 0 → minimize by pushing to lb  (unbounded if lb = −∞)
    //   c_k < 0 → push to ub                 (unbounded if ub = +∞)
    //   c_k = 0 → irrelevant; pin to lb if finite else ub if finite else 0
    let mut dropped_col = vec![false; n];
    for c in 0..n {
        if fixed[c].is_some() || substituted[c] {
            dropped_col[c] = true; // fixed / substituted columns are removed
            continue;
        }
        if col_nnz[c] == 0 {
            let (lb, ub) = (prob.lb_of(c), prob.ub_of(c));
            let value = if prob.c[c] > 0.0 {
                if lb <= -BOUND_INF {
                    return PresolveOutcome::Unbounded;
                }
                lb
            } else if prob.c[c] < 0.0 {
                if ub >= BOUND_INF {
                    return PresolveOutcome::Unbounded;
                }
                ub
            } else if lb > -BOUND_INF {
                lb
            } else if ub < BOUND_INF {
                ub
            } else {
                0.0
            };
            dropped_col[c] = true;
            stack.push(Reduction::FreeColumnFixed { col: c, value });
        }
    }

    // --- column map over surviving columns ---
    let mut kept_cols = Vec::new();
    let mut col_new = vec![usize::MAX; n];
    for c in 0..n {
        if !dropped_col[c] {
            col_new[c] = kept_cols.len();
            kept_cols.push(c);
        }
    }
    let fixval = |c: usize| fixed[c].unwrap_or(0.0);

    // --- objective: P, c, offset with fixed vars substituted ---
    // Surviving variables' linear cost is their original `c` plus any
    // cost shifted onto them by a free-column-singleton substitution.
    let mut new_c = vec![0.0; kept_cols.len()];
    for (newc, &oldc) in kept_cols.iter().enumerate() {
        new_c[newc] = prob.c[oldc] + c_adjust[oldc];
    }
    let mut offset = subst_offset;
    for (c, &fixed_c) in fixed.iter().enumerate() {
        if let Some(v) = fixed_c {
            offset += prob.c[c] * v;
        }
    }
    // Free/linear-only columns fixed to a bound contribute `c_k · value`.
    for r in &stack {
        if let Reduction::FreeColumnFixed { col, value } = r {
            offset += prob.c[*col] * value;
        }
    }
    let mut new_p: Vec<Triplet> = Vec::new();
    for t in &prob.p_lower {
        let (i, j, v) = (t.row, t.col, t.val);
        match (fixed[i].is_some(), fixed[j].is_some()) {
            (false, false) => new_p.push(Triplet::new(col_new[i], col_new[j], v)),
            (true, true) => {
                // both fixed → constant. Off-diagonal counts twice.
                if i == j {
                    offset += 0.5 * v * fixval(i) * fixval(j);
                } else {
                    offset += v * fixval(i) * fixval(j);
                }
            }
            (true, false) => new_c[col_new[j]] += v * fixval(i),
            (false, true) => new_c[col_new[i]] += v * fixval(j),
        }
    }

    // --- build reduced rows (after substitution), then dedup ---
    let eq_rows = match build_rows(
        &prob.a,
        m_eq,
        &eq_dropped,
        &prob.b,
        &fixed,
        &col_new,
        true,
        &[],
    ) {
        Ok(rows) => rows,
        Err(trigger) => return PresolveOutcome::Infeasible(trigger),
    };
    let ineq_rows = match build_rows(
        &prob.g,
        m_ineq,
        &ineq_dropped,
        &prob.h,
        &fixed,
        &col_new,
        false,
        soc_row,
    ) {
        Ok(rows) => rows,
        Err(trigger) => return PresolveOutcome::Infeasible(trigger),
    };

    // Duplicate/parallel-row merging reads *coefficients and right-hand
    // sides only* — never a variable box. So when the previous round left
    // the structure untouched (it only narrowed boxes) and this pass has
    // fixed nothing, substituted nothing and dropped no row, the rows here
    // are byte-for-byte the rows the previous pass already deduped, and
    // running it again cannot find a pair or a contradiction it did not find
    // then. Skipping it is a memoization, not a heuristic — and it is what
    // makes iterating to a real fixpoint affordable, since on a
    // bound-propagation cascade nearly every round is of exactly this shape
    // (gh #527; this hashing pass was ~70% of presolve's cost on `bore3d`).
    let rows_unchanged = structure_stable
        && stack
            .iter()
            .all(|r| matches!(r, Reduction::BoundTightening { .. }))
        && !eq_dropped.contains(&true)
        && !ineq_dropped.contains(&true);
    let eq_rows = if rows_unchanged {
        eq_rows
    } else {
        match dedup_rows(eq_rows, true, &[]) {
            Ok(rows) => rows,
            Err(trigger) => return PresolveOutcome::Infeasible(trigger),
        }
    };
    // SOC rows are coupled and must survive verbatim — exclude them from
    // parallel/duplicate merging.
    let ineq_rows = if rows_unchanged {
        ineq_rows
    } else {
        dedup_rows(ineq_rows, false, soc_row).expect("ineq dedup never infeasible")
    };

    // --- flatten surviving rows to triplets + kept-row maps ---
    let mut kept_eq = Vec::with_capacity(eq_rows.len());
    let mut new_a = Vec::new();
    let mut new_b = vec![0.0; eq_rows.len()];
    for (newr, row) in eq_rows.iter().enumerate() {
        kept_eq.push(row.orig);
        new_b[newr] = row.rhs;
        for &(c, v) in &row.coeffs {
            new_a.push(Triplet::new(newr, c, v));
        }
    }
    let mut kept_ineq = Vec::with_capacity(ineq_rows.len());
    let mut new_g = Vec::new();
    let mut new_h = vec![0.0; ineq_rows.len()];
    for (newr, row) in ineq_rows.iter().enumerate() {
        kept_ineq.push(row.orig);
        new_h[newr] = row.rhs;
        for &(c, v) in &row.coeffs {
            new_g.push(Triplet::new(newr, c, v));
        }
    }

    // Carry the kept columns' (possibly tightened) bounds into the reduced
    // problem. Emit bounds when the original had them or bound tightening
    // produced a finite bound on a kept variable; otherwise leave empty.
    let need_bounds = prob.has_bounds()
        || kept_cols
            .iter()
            .any(|&c| tlb[c] > -BOUND_INF || tub[c] < BOUND_INF);
    let (new_lb, new_ub) = if need_bounds {
        (
            kept_cols.iter().map(|&c| tlb[c]).collect(),
            kept_cols.iter().map(|&c| tub[c]).collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    let reduced = QpProblem {
        n: kept_cols.len(),
        p_lower: new_p,
        c: new_c,
        a: new_a,
        b: new_b,
        g: new_g,
        h: new_h,
        lb: new_lb,
        ub: new_ub,
    };

    PresolveOutcome::Reduced(Presolve {
        reduced,
        obj_offset: offset,
        orig_n: n,
        orig_m_eq: m_eq,
        orig_m_ineq: m_ineq,
        kept_cols,
        kept_eq,
        kept_ineq,
        orig: prob.clone(),
        stack,
        chain: Vec::new(),
        discarded_infeasibility: None,
        exit: FixpointExit::Fixpoint,
    })
}

/// Build per-row coefficient lists in the reduced column space,
/// substituting fixed variables into the right-hand side. Rows that
/// become empty after substitution trigger a feasibility check:
/// `0 = rhs` (equality) requires `rhs ≈ 0`; `0 ≤ rhs` (inequality)
/// requires `rhs ⪆ 0`. Returns `Err` with the trigger on detected
/// infeasibility.
///
/// The residual `rhs` of an emptied row is `b − Σ aⱼ vⱼ`, a *computed*
/// difference of terms of size up to [`Row::scale`], so it carries that
/// subtraction's rounding error — testing it against exact zero would call
/// a redundant-but-inexact row (two equalities implying the same value to
/// the last bit) infeasible. Compare against
/// [`EMPTY_ROW_TOL`] × the cancellation scale instead; see
/// `tests/issue496_ulp_inconsistent_equalities.rs`.
fn build_rows(
    triplets: &[Triplet],
    m: usize,
    dropped: &[bool],
    base_rhs: &[f64],
    fixed: &[Option<f64>],
    col_new: &[usize],
    is_equality: bool,
    protected: &[bool],
) -> Result<Vec<Row>, InfeasibleTrigger> {
    // A coupled cone row (second-order / exp / power / PSD) is kept verbatim
    // even when substitution empties its coefficients: an empty `G` row with
    // `s = h` is a legal cone slack (e.g. `h<0` is in `K_exp`), so it must
    // neither trip the orthant `0 ≤ rhs` feasibility check nor be dropped —
    // dropping one row of a fixed-layout block desyncs `reduced_cones`.
    let is_protected = |i: usize| protected.get(i).copied().unwrap_or(false);
    let mut acc: Vec<Option<Row>> = (0..m)
        .map(|r| {
            if dropped[r] {
                None
            } else {
                Some(Row {
                    coeffs: Vec::new(),
                    rhs: base_rhs[r],
                    orig: r,
                    scale: base_rhs[r].abs(),
                })
            }
        })
        .collect();

    for t in triplets {
        if dropped[t.row] || t.val == ZERO_TOL {
            continue;
        }
        let row = acc[t.row].as_mut().expect("non-dropped row");
        if let Some(v) = fixed[t.col] {
            let term = t.val * v;
            row.rhs -= term;
            row.scale = row.scale.max(term.abs());
        } else {
            row.coeffs.push((col_new[t.col], t.val));
        }
    }

    let mut out = Vec::new();
    for row in acc.into_iter().flatten() {
        let mut row = row;
        merge_sort_coeffs(&mut row.coeffs);
        if row.coeffs.is_empty() {
            // A protected cone row stays verbatim — `0·x ≤ h` is the cone
            // slack `s = h`, not an orthant feasibility check.
            if is_protected(row.orig) {
                out.push(row);
                continue;
            }
            // Row reduced to `0 (cmp) rhs`: a feasibility check, on a
            // residual that is only meaningful down to the rounding error
            // of the substitution that produced it.
            let tol = EMPTY_ROW_TOL * (1.0 + row.scale);
            let kind = if is_equality {
                "equality"
            } else {
                "inequality"
            };
            let violated = if is_equality {
                row.rhs.abs() > tol
            } else {
                row.rhs < -tol
            };
            if violated {
                return Err(InfeasibleTrigger {
                    screen: "row emptied by substitution is inconsistent",
                    detail: format!(
                        "{kind} row {}: residual {:e} exceeds tolerance {tol:e} \
                         (cancellation scale {:e})",
                        row.orig, row.rhs, row.scale
                    ),
                });
            }
            // Feasible empty row: drop it (no coefficients, no dual).
            continue;
        }
        out.push(row);
    }
    Ok(out)
}

/// Sort coefficients by column and merge any duplicate columns (a
/// variable appearing twice in one row). Drops entries that cancel to 0.
pub(crate) fn merge_sort_coeffs(coeffs: &mut Vec<(usize, f64)>) {
    coeffs.sort_by_key(|&(c, _)| c);
    let mut merged: Vec<(usize, f64)> = Vec::with_capacity(coeffs.len());
    for &(c, v) in coeffs.iter() {
        if let Some(last) = merged.last_mut() {
            if last.0 == c {
                last.1 += v;
                continue;
            }
        }
        merged.push((c, v));
    }
    merged.retain(|&(_, v)| v != 0.0);
    *coeffs = merged;
}

/// Relative tolerance for confirming two rows are scalar multiples.
const PARALLEL_TOL: f64 = 1e-9;

/// Canonical pivot used to normalize a row for *parallel* (scalar-
/// multiple) detection: its first coefficient (the rows' coeffs are
/// sorted by column). For inequalities we divide by the pivot's
/// **magnitude** so only *positive* multiples — same inequality direction
/// — normalize alike; for equalities we divide by the **signed** pivot so
/// `±` multiples (the same constraint either way) match.
fn pivot_divisor(row: &Row, is_equality: bool) -> f64 {
    // Empty (coupled cone) rows are never grouped/merged, so any nonzero
    // divisor is fine; guard the index to keep normalization panic-free.
    let p = row.coeffs.first().map_or(1.0, |&(_, v)| v);
    if is_equality { p } else { p.abs() }
}

/// Normalized coefficient values (parallel detection): `coeffs / divisor`.
fn normalized_coeffs(row: &Row, is_equality: bool) -> Vec<(usize, f64)> {
    let d = pivot_divisor(row, is_equality);
    row.coeffs.iter().map(|&(c, v)| (c, v / d)).collect()
}

/// Hash a normalized coefficient pattern. Values are quantized so exact
/// scalar multiples hash together; the hash is only a *filter* (a quantize
/// boundary can split a true pair into different buckets, which merely
/// misses a reduction — never a wrong merge, since membership is confirmed
/// by [`approx_parallel`]).
fn parallel_signature(norm: &[(usize, f64)]) -> u64 {
    let mut h = DefaultHasher::new();
    norm.len().hash(&mut h);
    for &(c, v) in norm {
        c.hash(&mut h);
        ((v / PARALLEL_TOL).round() as i64).hash(&mut h);
    }
    h.finish()
}

/// Confirm two normalized patterns are equal to `PARALLEL_TOL` (same
/// columns, matching values). Conservative: only true scalar multiples
/// pass, so a wrong merge is impossible.
fn approx_parallel(a: &[(usize, f64)], b: &[(usize, f64)]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(&(ca, va), &(cb, vb))| {
            ca == cb && (va - vb).abs() <= PARALLEL_TOL * (1.0 + va.abs().max(vb.abs()))
        })
}

/// Remove **parallel** rows (scalar multiples of one another), the
/// generalization of exact-duplicate removal (PaPILO's parallel-row
/// reduction). Normalized signatures are computed in parallel (rayon);
/// grouping and the per-group decision are serial and cheap.
///
/// Dual recovery stays trivial because we always keep an *original* row in
/// its own frame and set every dropped row's multiplier to 0 (the kept row
/// carries the constraint):
/// - equalities — all scalar multiples represent one constraint; their
///   *normalized* right-hand sides must agree, else the system is
///   infeasible. Keep the first; drop the rest.
/// - inequalities — positive multiples of one direction; keep the **most
///   restrictive** original row (smallest normalized rhs `h / |pivot|`)
///   and drop the looser ones, which it implies.
fn dedup_rows(
    rows: Vec<Row>,
    is_equality: bool,
    protected: &[bool],
) -> Result<Vec<Row>, InfeasibleTrigger> {
    if rows.len() < 2 {
        return Ok(rows);
    }
    // A row is protected (never merged) when its *original* index is marked
    // — used to keep coupled cone rows verbatim.
    let is_protected = |i: usize| protected.get(rows[i].orig).copied().unwrap_or(false);

    // Parallel: normalize + hash each row (PaPILO-style hashing-based
    // pairing, generalized to scalar multiples).
    let norms: Vec<Vec<(usize, f64)>> = rows
        .par_iter()
        .map(|r| normalized_coeffs(r, is_equality))
        .collect();
    let sigs: Vec<u64> = norms.par_iter().map(|n| parallel_signature(n)).collect();

    // Group row indices by signature (serial; small). Protected rows are
    // excluded from grouping, so they are never dropped and never drop
    // others.
    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &s) in sigs.iter().enumerate() {
        if !is_protected(i) {
            buckets.entry(s).or_default().push(i);
        }
    }

    // Normalized rhs of a row, for the tightness / consistency decisions.
    let norm_rhs = |i: usize| rows[i].rhs / pivot_divisor(&rows[i], is_equality);

    let mut keep = vec![true; rows.len()];
    for idxs in buckets.values() {
        if idxs.len() < 2 {
            continue;
        }
        // Within a signature bucket, partition into confirmed-parallel
        // groups (guards against quantization collisions).
        let mut handled = vec![false; idxs.len()];
        for a in 0..idxs.len() {
            if handled[a] {
                continue;
            }
            let mut group = vec![idxs[a]];
            handled[a] = true;
            for b in (a + 1)..idxs.len() {
                if !handled[b] && approx_parallel(&norms[idxs[a]], &norms[idxs[b]]) {
                    handled[b] = true;
                    group.push(idxs[b]);
                }
            }
            if group.len() < 2 {
                continue;
            }
            if is_equality {
                // Parallel equalities: normalized rhs must agree, else the
                // two scaled-identical constraints are contradictory.
                let r0 = norm_rhs(group[0]);
                for &g in &group[1..] {
                    if (norm_rhs(g) - r0).abs() > PARALLEL_TOL * (1.0 + r0.abs()) {
                        return Err(InfeasibleTrigger {
                            screen: "parallel equalities disagree",
                            detail: format!(
                                "equality rows {} and {} are scalar multiples with \
                                 normalized right-hand sides {r0:e} and {:e}",
                                rows[group[0]].orig,
                                rows[g].orig,
                                norm_rhs(g)
                            ),
                        });
                    }
                }
                for &g in &group[1..] {
                    keep[g] = false;
                }
            } else {
                // Parallel inequalities: keep the most restrictive original
                // row (smallest normalized rhs); it implies the rest.
                let tightest = *group
                    .iter()
                    .min_by(|&&p, &&q| norm_rhs(p).partial_cmp(&norm_rhs(q)).unwrap())
                    .unwrap();
                for &g in &group {
                    if g != tightest {
                        keep[g] = false;
                    }
                }
            }
        }
    }

    Ok(rows
        .into_iter()
        .zip(keep)
        .filter_map(|(r, k)| if k { Some(r) } else { None })
        .collect())
}

/// Summary of what presolve removed, for logging and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PresolveStats {
    /// Variables in the original problem.
    pub orig_vars: usize,
    /// Variables in the reduced problem.
    pub reduced_vars: usize,
    /// Equality + inequality rows in the original problem.
    pub orig_rows: usize,
    /// Equality + inequality rows in the reduced problem.
    pub reduced_rows: usize,
    /// Variables fixed by a singleton equality row.
    pub fixed_vars: usize,
    /// Free / linear-only columns pinned to a bound and dropped.
    pub free_cols_fixed: usize,
    /// Free column singletons substituted out (each also removes a row).
    pub free_col_singletons: usize,
    /// Forcing rows: each pins all its variables to a bound and is dropped.
    pub forcing_rows: usize,
    /// Dominated columns fixed to a bound and dropped.
    pub dominated_cols: usize,
    /// Variable bounds tightened by domain propagation.
    pub tightened_bounds: usize,
    /// Variables folded onto another by a two-variable equality row
    /// (doubleton aggregation). Each also consumes its row.
    pub aggregated_vars: usize,
    /// Layers in the reduction chain — how many passes the fixpoint took.
    /// One round contributes one or two (the catalog pass, and an
    /// aggregation when one fired). `1` whenever the handle *is* a single
    /// layer, which covers the conic path, a no-op presolve, and a fixpoint
    /// that converged in one round alike — those are the same object here.
    pub rounds: usize,
    /// Why the fixpoint iteration stopped (gh #527). [`FixpointExit::RoundCap`]
    /// means the reduction below is a **truncation**, not the fixpoint
    /// presolve documents — reductions were still firing when the layer cap
    /// stopped them, and a different cap would have handed the solver a
    /// different problem.
    pub exit: FixpointExit,
}

impl PresolveStats {
    /// Did presolve remove anything?
    pub fn reduced_anything(&self) -> bool {
        self.reduced_vars < self.orig_vars || self.reduced_rows < self.orig_rows
    }
}

impl Presolve {
    /// An infeasibility claim the full catalog raised that the confirming
    /// re-derivation would not reproduce, so [`presolve`] discarded it and
    /// returned this reduction instead (gh #523).
    ///
    /// `None` on every normal solve. When it is `Some`, presolve came within
    /// one speculative fixing of answering `Infeasible_Problem_Detected` on a
    /// problem it could not otherwise prove infeasible — worth surfacing,
    /// because the guard turns that into a few lost eliminations rather than
    /// a wrong answer, and nobody finds the underlying bug if the near-miss
    /// is silent.
    pub fn discarded_infeasibility(&self) -> Option<&InfeasibleTrigger> {
        self.discarded_infeasibility.as_ref()
    }

    /// The cone partition of the *reduced* inequality block, given the
    /// original `cones`. Walks the kept inequality rows (a cone-aware
    /// presolve never drops or reorders a second-order-cone block, so each
    /// cone's surviving rows stay contiguous) and run-length-encodes them by
    /// source cone. Orthant blocks may shrink (or vanish); SOC blocks keep
    /// their full dimension. Use after [`presolve_conic`] (a single pass).
    pub fn reduced_cones(&self, cones: &[ConeSpec]) -> Vec<ConeSpec> {
        // Original inequality row → cone index.
        let mut row_cone = vec![usize::MAX; self.orig_m_ineq];
        let mut r = 0;
        for (ci, spec) in cones.iter().enumerate() {
            for _ in 0..spec.dim() {
                if r < row_cone.len() {
                    row_cone[r] = ci;
                }
                r += 1;
            }
        }
        let mut out = Vec::new();
        let mut i = 0;
        while i < self.kept_ineq.len() {
            let ci = row_cone[self.kept_ineq[i]];
            let mut j = i;
            while j < self.kept_ineq.len() && row_cone[self.kept_ineq[j]] == ci {
                j += 1;
            }
            let count = j - i;
            out.push(match cones[ci] {
                ConeSpec::Nonneg(_) => ConeSpec::Nonneg(count),
                ConeSpec::SecondOrder(_) => ConeSpec::SecondOrder(count),
                // Non-symmetric cones are fixed at 3 rows and are not split or
                // merged by presolve.
                ConeSpec::Exponential => ConeSpec::Exponential,
                ConeSpec::Power(a) => ConeSpec::Power(a),
                // PSD blocks are structurally coupled (svec of a fixed n×n)
                // and likewise pass through unchanged.
                ConeSpec::Psd(n) => ConeSpec::Psd(n),
            });
            i = j;
        }
        out
    }

    /// Did this single pass change anything (a reduction, or a dropped
    /// row)? Used by [`presolve`] to detect the fixpoint.
    fn changed(&self) -> bool {
        !self.stack.is_empty()
            || self.reduced.n < self.orig_n
            || self.reduced.m_eq() + self.reduced.m_ineq() < self.orig_m_eq + self.orig_m_ineq
    }

    /// Reduction summary (sizes before/after and counts by reduction). For
    /// an iterated presolve, counts aggregate over the rounds.
    pub fn stats(&self) -> PresolveStats {
        if self.chain.is_empty() {
            return self.stats_once();
        }
        let mut s = PresolveStats {
            orig_vars: self.orig_n,
            reduced_vars: self.reduced.n,
            orig_rows: self.orig_m_eq + self.orig_m_ineq,
            reduced_rows: self.reduced.m_eq() + self.reduced.m_ineq(),
            exit: self.exit,
            rounds: self.chain.len(),
            ..Default::default()
        };
        for layer in &self.chain {
            let ls = layer.stats_once();
            s.fixed_vars += ls.fixed_vars;
            s.free_cols_fixed += ls.free_cols_fixed;
            s.free_col_singletons += ls.free_col_singletons;
            s.forcing_rows += ls.forcing_rows;
            s.dominated_cols += ls.dominated_cols;
            s.tightened_bounds += ls.tightened_bounds;
            s.aggregated_vars += ls.aggregated_vars;
        }
        s
    }

    fn stats_once(&self) -> PresolveStats {
        let mut s = PresolveStats {
            orig_vars: self.orig_n,
            reduced_vars: self.reduced.n,
            orig_rows: self.orig_m_eq + self.orig_m_ineq,
            reduced_rows: self.reduced.m_eq() + self.reduced.m_ineq(),
            exit: self.exit,
            // This handle is exactly one layer, however it was reached.
            rounds: 1,
            ..Default::default()
        };
        for r in &self.stack {
            match r {
                Reduction::FixedVar { .. } => s.fixed_vars += 1,
                Reduction::FreeColumnFixed { .. } => s.free_cols_fixed += 1,
                Reduction::FreeColSingleton { .. } => s.free_col_singletons += 1,
                Reduction::ForcingRow { .. } => s.forcing_rows += 1,
                Reduction::DominatedColumn { .. } => s.dominated_cols += 1,
                Reduction::BoundTightening { .. } => s.tightened_bounds += 1,
                // A whole pass in one entry: report what its plan achieved.
                Reduction::Aggregate { plan } => {
                    s.aggregated_vars += plan.report.n_aggregated_vars;
                    // A singleton row the aggregation's fixed point reached
                    // is the same reduction `FixedVar` performs, so it is
                    // counted in the same column.
                    s.fixed_vars += plan.report.n_constant_vars;
                }
            }
        }
        s
    }

    /// Expand a reduced-problem solution back to the original space,
    /// recovering primal `x` and duals `(y, z)`. For an iterated presolve,
    /// folds the per-round postsolves in reverse.
    pub fn postsolve(&self, red: &QpSolution) -> QpSolution {
        if self.chain.is_empty() {
            return self.postsolve_once(red);
        }
        let mut sol = red.clone();
        for layer in self.chain.iter().rev() {
            sol = layer.postsolve_once(&sol);
        }
        sol
    }

    /// Expand a single pass's reduced solution back to its original space.
    fn postsolve_once(&self, red: &QpSolution) -> QpSolution {
        // An aggregation layer is a whole pass in one entry, with its own
        // primal lift and dual sweep; it never shares a layer with the
        // single-elimination catalog.
        let mut out = if let [Reduction::Aggregate { plan }] = self.stack.as_slice() {
            crate::aggregate::postsolve(&self.orig, plan, red)
        } else {
            self.postsolve_catalog(red)
        };
        // [`QpIterate::objective`] is documented to be in the original
        // problem's coordinates, but the trace comes back from a solve of
        // the *reduced* problem — which differs by exactly the constant any
        // substitution moved into the objective. Put it back, per layer, so
        // the chain composes to the user's objective.
        if self.obj_offset != 0.0 {
            for it in &mut out.iterates {
                it.objective += self.obj_offset;
            }
        }
        out
    }

    /// [`Self::postsolve_once`] for a layer built by [`presolve_once`] —
    /// every reduction in the catalog except the aggregation, which has its
    /// own recovery.
    fn postsolve_catalog(&self, red: &QpSolution) -> QpSolution {
        let mut x = vec![0.0; self.orig_n];
        let mut y = vec![0.0; self.orig_m_eq];
        let mut z = vec![0.0; self.orig_m_ineq];

        // Primal: kept columns from the reduced solution.
        for (newc, &oldc) in self.kept_cols.iter().enumerate() {
            x[oldc] = red.x[newc];
        }
        // Duals: kept rows from the reduced solution. Dropped rows
        // (empty / duplicate) stay 0, which is their correct multiplier.
        for (newr, &oldr) in self.kept_eq.iter().enumerate() {
            y[oldr] = red.y[newr];
        }
        for (newr, &oldr) in self.kept_ineq.iter().enumerate() {
            z[oldr] = red.z[newr];
        }

        // Restore eliminated primals in two passes, ordered by dependency.
        //
        // A free-column-singleton recovers `x_col = (b_r − Σ_{j≠col} a_jr
        // x_j) / a_col`, so it *reads* the values of the other variables in
        // its consumed row. Those neighbours may themselves have been
        // eliminated by a **constant-valued** reduction (a fixed / free-fixed
        // / dominated / forced variable) earlier in the same pass — earlier,
        // hence *lower* on the stack. A plain reverse-LIFO replay would
        // restore the singleton (higher on the stack) before its constant
        // neighbour, reading a stale 0 for it and producing an infeasible
        // recovered point (the capri LP wrong-answer bug). The neighbours are
        // never themselves singletons (a free-column-singleton variable
        // appears in exactly one equality row — its own consumed row — so it
        // cannot appear in another singleton's row), so two passes suffice:
        //   1. all constant-valued primal restorations (any order — they
        //      depend on nothing); then
        //   2. the formula-based free-column-singletons, which now read fully
        //      restored neighbours.
        for r in self.stack.iter().rev() {
            match r {
                Reduction::FixedVar { col, value, .. } => x[*col] = *value,
                Reduction::FreeColumnFixed { col, value } => x[*col] = *value,
                Reduction::ForcingRow { cols, .. } => {
                    // Each forced variable sits at the stored bound value.
                    for &(col, _, value, _) in cols {
                        x[col] = value;
                    }
                }
                Reduction::DominatedColumn { col, value, .. } => x[*col] = *value,
                // Restored in the second pass (depends on its neighbours).
                Reduction::FreeColSingleton { .. } => {}
                // The variable is kept; only its box changed, so its primal
                // comes from the reduced solution (already mapped above).
                Reduction::BoundTightening { .. } => {}
                // Unreachable: an aggregation layer returned above, before
                // any of this. Left as a no-op rather than a panic so the
                // arm can never be the thing that fails a solve.
                Reduction::Aggregate { .. } => {}
            }
        }
        for r in &self.stack {
            if let Reduction::FreeColSingleton {
                col,
                eq_row,
                a_coef,
                ..
            } = r
            {
                // x_col = (b_r − Σ_{j≠col} a_jr x_j) / a_col.
                let mut acc = self.orig.b[*eq_row];
                for t in &self.orig.a {
                    if t.row == *eq_row && t.col != *col {
                        acc -= t.val * x[t.col];
                    }
                }
                x[*col] = acc / a_coef;
            }
        }

        // Free-column-singleton consumed-row multipliers have the unique
        // value y_r = −c_col / a_col (from stationarity of the eliminated
        // free variable, which has no P/G terms).
        for r in &self.stack {
            if let Reduction::FreeColSingleton {
                eq_row,
                a_coef,
                c_col,
                ..
            } = r
            {
                y[*eq_row] = -c_col / a_coef;
            }
        }

        // Recover each fixing row's multiplier from stationarity for its
        // variable: with all primals and other duals known,
        //   (Px)_k + c_k + (Aᵀy)_k + (Gᵀz)_k + a·y_fix = 0
        //   ⇒ y_fix = −[(Px)_k + c_k + (Aᵀy)_k + (Gᵀz)_k] / a.
        let n = self.orig_n;
        let mut grad = vec![0.0; n];
        grad[..n].copy_from_slice(&self.orig.c[..n]);
        self.orig.p_mul(&x, &mut grad);
        self.orig.at_mul(&y, &mut grad);
        self.orig.gt_mul(&z, &mut grad);
        for r in &self.stack {
            if let Reduction::FixedVar {
                col,
                eq_row,
                a_coef,
                ..
            } = r
            {
                y[*eq_row] = -grad[*col] / a_coef;
            }
        }

        // Forcing-row multipliers. `grad` (above, = grad0) is each pinned
        // variable's reduced cost *excluding* the forcing row (its
        // multiplier is still 0). The row multiplier is the tightest value
        // making every pinned variable's bound multiplier correctly signed:
        //   min-vertex  ⇒ mult = maxⱼ(−gradⱼ/coefⱼ)  (clamped ≥ 0 if ≤-row);
        //   max-vertex  ⇒ mult = minⱼ(−gradⱼ/coefⱼ)  (equalities only).
        // (The pinned variables' bound multipliers themselves come out of
        // the global recovery below.)
        for r in &self.stack {
            if let Reduction::ForcingRow {
                row,
                is_equality,
                at_max,
                cols,
            } = r
            {
                let mut mult = if *at_max {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                };
                for &(col, coef, _, _) in cols {
                    let t = -grad[col] / coef;
                    mult = if *at_max { mult.min(t) } else { mult.max(t) };
                }
                if !*is_equality {
                    mult = mult.max(0.0); // inequality multiplier ≥ 0
                }
                if !mult.is_finite() {
                    mult = 0.0;
                }
                if *is_equality {
                    y[*row] = mult;
                } else {
                    z[*row] = mult;
                }
            }
        }

        // Re-attribute active tightened-bound multipliers to their source
        // rows. A tightened bound that is active in the reduced solve while
        // the *original* bound is slack is not a real bound — its
        // multiplier belongs to the row that implied it. Because tightening
        // sources are column-disjoint, these moves are independent.
        let mut col_reduced = vec![usize::MAX; n];
        for (newc, &oldc) in self.kept_cols.iter().enumerate() {
            col_reduced[oldc] = newc;
        }
        for r in &self.stack {
            if let Reduction::BoundTightening {
                col,
                row,
                is_equality,
                coef,
                is_upper,
            } = r
            {
                let newc = col_reduced[*col];
                if newc == usize::MAX {
                    continue;
                }
                let delta = if *is_upper {
                    let m = red.z_ub.get(newc).copied().unwrap_or(0.0);
                    if m > 0.0 && x[*col] < self.orig.ub_of(*col) - BOUND_FEAS_TOL {
                        m / coef
                    } else {
                        0.0
                    }
                } else {
                    let m = red.z_lb.get(newc).copied().unwrap_or(0.0);
                    if m > 0.0 && x[*col] > self.orig.lb_of(*col) + BOUND_FEAS_TOL {
                        -m / coef
                    } else {
                        0.0
                    }
                };
                if *is_equality {
                    y[*row] += delta;
                } else {
                    z[*row] += delta;
                }
            }
        }

        // Global bound-multiplier recovery. With every row multiplier now in
        // place, recompute the full reduced cost and read off each
        // variable's bound multipliers by complementarity against its
        // *original* box: at the lower bound `z_lb = max(0, grad)`, at the
        // upper `z_ub = max(0, −grad)`, interior ⇒ both 0. This single rule
        // subsumes the per-reduction bound recovery (fixed, free-fixed,
        // forcing, dominated — each lands at a real bound or interior with
        // the right reduced cost) and correctly zeroes a tightened
        // variable's bound dual (it sits interior to its real box, the force
        // having moved to the source row above).
        let mut grad = vec![0.0; n];
        grad[..n].copy_from_slice(&self.orig.c[..n]);
        self.orig.p_mul(&x, &mut grad);
        self.orig.at_mul(&y, &mut grad);
        self.orig.gt_mul(&z, &mut grad);
        let mut z_lb = vec![0.0; n];
        let mut z_ub = vec![0.0; n];
        for i in 0..n {
            let lb = self.orig.lb_of(i);
            let ub = self.orig.ub_of(i);
            let at_lb = lb > -BOUND_INF && at_bound(x[i], lb);
            let at_ub = ub < BOUND_INF && at_bound(x[i], ub);
            if at_lb && grad[i] > 0.0 {
                z_lb[i] = grad[i];
            } else if at_ub && grad[i] < 0.0 {
                z_ub[i] = -grad[i];
            }
        }

        // Objective in the original problem.
        let mut px = vec![0.0; n];
        self.orig.p_mul(&x, &mut px);
        let mut obj = 0.0;
        for i in 0..n {
            obj += 0.5 * x[i] * px[i] + self.orig.c[i] * x[i];
        }

        QpSolution {
            status: red.status,
            x,
            y,
            z,
            z_lb,
            z_ub,
            obj,
            iters: red.iters,
            iterates: red.iterates.clone(),
        }
    }
}

/// Convenience: presolve, solve the reduced problem with `solve`, and
/// postsolve — returning a solution in the *original* problem space. On a
/// presolve-detected infeasibility / unboundedness, returns the matching
/// status without invoking the solver.
pub fn solve_with_presolve<S>(prob: &QpProblem, solve: S) -> QpSolution
where
    S: FnOnce(&QpProblem) -> QpSolution,
{
    let trivial = |status| QpSolution {
        status,
        x: vec![0.0; prob.n],
        y: vec![0.0; prob.m_eq()],
        z: vec![0.0; prob.m_ineq()],
        z_lb: vec![0.0; prob.n],
        z_ub: vec![0.0; prob.n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    };
    match presolve(prob) {
        PresolveOutcome::Infeasible(_) => trivial(QpStatus::PrimalInfeasible),
        PresolveOutcome::Unbounded => trivial(QpStatus::DualInfeasible),
        PresolveOutcome::Reduced(ps) => {
            let red = solve(&ps.reduced);
            ps.postsolve(&red)
        }
    }
}

#[cfg(test)]
mod dedup_memo_tests {
    //! gh #527 — the duplicate-row memoization must not change a reduction.
    //!
    //! Skipping `dedup_rows` on a structure-stable round is sound only because
    //! an all-`BoundTightening` stack implies no column was fixed or
    //! substituted, so the rows this pass builds are byte-for-byte the rows
    //! the previous pass already deduped. That argument depends on every
    //! column-removing reduction pushing a stack entry, and on the two
    //! `*_dropped` flags covering the row removals that push none — an
    //! exhaustiveness claim over the whole catalog that a reduction added
    //! later can break from anywhere. Running both ways and comparing is what
    //! turns it from a reading into a check.

    use super::*;

    /// Everything about a reduced problem that a wrongly-skipped dedup could
    /// disturb: its shape, its coefficients, and its boxes.
    fn fingerprint(p: &QpProblem) -> String {
        let trips = |t: &[Triplet]| {
            let mut v: Vec<(usize, usize, u64)> =
                t.iter().map(|t| (t.row, t.col, t.val.to_bits())).collect();
            v.sort_unstable();
            format!("{v:?}")
        };
        let floats = |f: &[f64]| format!("{:?}", f.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        format!(
            "n={} m_eq={} m_ineq={} P={} c={} A={} b={} G={} h={} lb={} ub={}",
            p.n,
            p.m_eq(),
            p.m_ineq(),
            trips(&p.p_lower),
            floats(&p.c),
            trips(&p.a),
            floats(&p.b),
            trips(&p.g),
            floats(&p.h),
            floats(&p.lb),
            floats(&p.ub),
        )
    }

    fn reduce(prob: &QpProblem, memo: DedupMemo) -> Option<(String, PresolveStats)> {
        match presolve_fixpoint(prob, Catalog::Full, memo) {
            PresolveOutcome::Reduced(ps) => Some((fingerprint(&ps.reduced), ps.stats())),
            _ => None,
        }
    }

    /// Compare the two builds on `prob`, naming `label` if they diverge.
    fn assert_same(label: &str, prob: &QpProblem) {
        let on = reduce(prob, DedupMemo::Enabled);
        let off = reduce(prob, DedupMemo::Disabled);
        match (on, off) {
            (Some((f_on, s_on)), Some((f_off, s_off))) => {
                assert_eq!(f_on, f_off, "{label}: reduced problems differ");
                // The stats too: a skipped merge that dropped a different row
                // could leave the same shape by coincidence.
                assert_eq!(
                    (s_on.fixed_vars, s_on.aggregated_vars, s_on.forcing_rows),
                    (s_off.fixed_vars, s_off.aggregated_vars, s_off.forcing_rows),
                    "{label}: reduction counts differ"
                );
            }
            (None, None) => {} // both proved infeasible / unbounded — agreed
            (a, b) => panic!(
                "{label}: outcomes differ ({:?})",
                (a.is_some(), b.is_some())
            ),
        }
    }

    /// A deterministic stand-in for a random generator, so a failure is
    /// reproducible from the seed alone.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() as usize) % n
        }
        fn coef(&mut self) -> f64 {
            // Small integers keep duplicate/parallel rows genuinely reachable
            // — the whole point is to give dedup something to find.
            [-2.0, -1.0, 1.0, 2.0, 3.0][self.below(5)]
        }
    }

    /// Random bounded LPs with **deliberately duplicated and scaled rows**, so
    /// the memoized pass has real merges to get wrong, and with boxes wide
    /// enough that bound propagation runs for several rounds and the skip
    /// actually engages.
    #[test]
    fn the_dedup_memoization_never_changes_the_reduction() {
        let mut rng = Lcg(0x527);
        for case in 0..200 {
            let n = 4 + rng.below(8);
            let m_eq = 1 + rng.below(4);
            let m_ineq = 2 + rng.below(5);
            let mut a = Vec::new();
            for r in 0..m_eq {
                for _ in 0..2 + rng.below(2) {
                    a.push(Triplet::new(r, rng.below(n), rng.coef()));
                }
            }
            let mut g = Vec::new();
            for r in 0..m_ineq {
                for _ in 0..2 + rng.below(2) {
                    g.push(Triplet::new(r, rng.below(n), rng.coef()));
                }
            }
            // Duplicate one inequality row, and add a positive multiple of
            // another — the two shapes `dedup_rows` merges.
            let dup_src = rng.below(m_ineq);
            let scale = [1.0, 2.0, 0.5][rng.below(3)];
            let extra: Vec<Triplet> = g
                .iter()
                .filter(|t| t.row == dup_src)
                .map(|t| Triplet::new(m_ineq, t.col, t.val * scale))
                .collect();
            g.extend(extra);
            let mut h: Vec<f64> = (0..m_ineq).map(|_| 1.0 + rng.below(20) as f64).collect();
            h.push(h[dup_src] * scale);

            let prob = QpProblem {
                n,
                p_lower: vec![],
                c: (0..n).map(|_| rng.coef()).collect(),
                a,
                b: (0..m_eq).map(|_| rng.below(5) as f64).collect(),
                g,
                h,
                lb: vec![0.0; n],
                ub: vec![1e5; n],
            };
            assert_same(&format!("case {case}"), &prob);
        }
    }

    /// The case that gives this test its teeth: a fixing whose **substitution
    /// creates a duplicate pair that did not exist before**, on a round after
    /// the first.
    ///
    /// `x₃ = 1` fixes in round 1, which turns `x₂ − x₃ = −1` into a singleton
    /// that fixes `x₂ = 0` in round 2. Substituting `x₂` out collapses
    /// `x₀ + x₁ + x₂ ≤ 10` onto `x₀ + x₁ ≤ 5`, and only then are the two rows
    /// parallel — dedup must run on that round and drop the looser one.
    ///
    /// This is the failure the gate's stack check is written to prevent. Worth
    /// recording what trying to *provoke* it showed, because it changes where
    /// the safety actually comes from: weakening the gate all the way to
    /// `rows_unchanged = structure_stable` does **not** make this diverge. The
    /// round trace says why — the aggregation collapses the `x₃ → x₂` chain in
    /// round 1, so the substitution and the merge both land in round 2, whose
    /// predecessor was a *structural* round and which therefore never had
    /// `structure_stable` set in the first place.
    ///
    /// That appears to be general: a column removal is always preceded by a
    /// round that changed the shape, so the `bounds_only` dimension check is
    /// what carries the argument and the stack/`*_dropped` conjuncts are belt
    /// and braces. They are kept — being conservative here costs one boolean —
    /// but this test is a *guard against future divergence*, not a
    /// demonstration of a caught bug, and no weakening tried so far makes it
    /// fail. See the PR thread on #530.
    #[test]
    fn a_substitution_that_creates_a_duplicate_still_gets_deduped() {
        let prob = QpProblem {
            n: 5,
            p_lower: vec![],
            // x₀/x₁ carry a negative cost so they are not dominated columns
            // (which would fix them at their lower bound and erase the rows).
            c: vec![-1.0, -1.0, 0.0, 0.0, 1.0],
            a: vec![
                Triplet::new(0, 3, 1.0),
                Triplet::new(1, 2, 1.0),
                Triplet::new(1, 3, -1.0),
            ],
            b: vec![1.0, -1.0],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(0, 2, 1.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(1, 1, 1.0),
                Triplet::new(2, 4, -1.0),
            ],
            h: vec![10.0, 5.0, -10.0],
            lb: vec![0.0, 0.0, 0.0, 0.0, f64::NEG_INFINITY],
            ub: vec![100.0, 100.0, 100.0, 100.0, f64::INFINITY],
        };
        assert_same("substitution-created duplicate", &prob);

        // And the merge really happens, or the comparison above is vacuous:
        // the looser of the two parallel rows must be gone.
        let PresolveOutcome::Reduced(ps) =
            presolve_fixpoint(&prob, Catalog::Full, DedupMemo::Enabled)
        else {
            panic!("feasible bounded problem");
        };
        let kept: Vec<f64> = ps.reduced.h.clone();
        assert!(
            !kept.contains(&10.0),
            "the looser parallel row survived — dedup did not run on the \
             substitution round; reduced h = {kept:?}"
        );
    }

    /// The shape the skip is actually built for: many consecutive rounds whose
    /// only change is a narrowed box, with duplicate rows present throughout.
    /// If the memoization is going to drop a merge, it is here.
    #[test]
    fn a_long_bounds_only_cascade_reduces_identically_either_way() {
        // x₀ − 0.999·x₁ ≤ 1 and its mirror: one refinement per round for as
        // many rounds as the budget allows. Rows 2 and 3 duplicate row 0
        // exactly and at a positive multiple.
        let prob = QpProblem {
            n: 3,
            p_lower: vec![],
            c: vec![0.0, 0.0, 1.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, -0.999),
                Triplet::new(1, 1, 1.0),
                Triplet::new(1, 0, -0.999),
                Triplet::new(2, 0, 1.0),
                Triplet::new(2, 1, -0.999),
                Triplet::new(3, 0, 2.0),
                Triplet::new(3, 1, -1.998),
                Triplet::new(4, 2, -1.0),
            ],
            h: vec![1.0, 1.0, 1.0, 2.0, -10.0],
            lb: vec![0.0, 0.0, f64::NEG_INFINITY],
            ub: vec![1e6, 1e6, f64::INFINITY],
        };
        assert_same("contracting pair with duplicate rows", &prob);

        // And the skip must really be engaging, or the test proves nothing.
        let PresolveOutcome::Reduced(ps) =
            presolve_fixpoint(&prob, Catalog::Full, DedupMemo::Enabled)
        else {
            panic!("feasible bounded problem");
        };
        assert!(
            ps.stats().rounds > 4,
            "expected several bounds-only rounds for the skip to apply to, \
             got {} layers",
            ps.stats().rounds
        );
    }
}
