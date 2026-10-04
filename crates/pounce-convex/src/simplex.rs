//! Bounded-variable revised primal simplex — the LP-crossover engine.
//!
//! # Why a simplex
//!
//! A pure interior-point method cannot certify a *degenerate* LP vertex to
//! `tol`: where strict complementarity fails the fraction-to-boundary step
//! collapses, `μ` freezes, and the primal residual plateaus above tolerance, so
//! the solve grinds to its iteration cap with the correct objective but no
//! convergence certificate (NETLIB GEN family — issue #133). Every production
//! LP solver pairs the IPM with a **crossover** that pivots the near-optimal
//! interior point to an exact optimal vertex basis (Andersen & Ye 1996; Megiddo
//! 1991).
//!
//! The previous crossover bridged to the symmetric LDLᵀ active-set QP engine
//! ([`pounce_qp`]), which cannot step through a highly degenerate constraint
//! vertex without an explicit, pivotable basis — it stalls and never-regresses.
//! This module is the architecturally-correct replacement: a self-contained
//! revised simplex on an **unsymmetric LU basis** ([`feral::SparseLu`], with
//! `ftran`/`btran` solves and an `O(bump)` column-replacement `update`), which
//! pivots one variable at a time and walks straight through degeneracy with
//! Bland's rule.
//!
//! # Standard form
//!
//! The convex LP `min cᵀx  s.t.  Ax = b, Gx ≤ h, lb ≤ x ≤ ub` is converted to
//! computational standard form with one **logical** (slack/artificial) variable
//! per row:
//!
//! ```text
//!   variables w = [ x (n structural) ; ℓ (m = m_eq + m_ineq logical) ]
//!   M w = rhs,   M = [ A_eq ; G  |  I_m ],   rhs = [ b ; h ]
//!   bounds:  structural j   → [lb_j, ub_j]
//!            eq-row logical  → [0, 0]      (artificial: feasible only at 0)
//!            ineq-row logical→ [0, +∞)     (slack s = h − Gx ≥ 0)
//!   cost = [ c | 0 ]
//! ```
//!
//! # Index spaces (the easy thing to get wrong)
//!
//! `feral`'s basis `B` is factored from the basis columns in **slot order**
//! (slot = position in the basis). Its solves bridge two spaces:
//! - `ftran(r)`: **row-space → slot-space** — solves `B z = r`; used for the
//!   basic values `x_B = B⁻¹(rhs − N x_N)` and the pivot column `α = B⁻¹ a_q`.
//! - `btran(c_B)`: **slot-space → row-space** — solves `Bᵀ π = c_B`; used for
//!   the simplex multipliers `π` (a row-space vector).
//!
//! # Dual recovery (sign convention)
//!
//! With `π` the row-space multiplier of `M w = rhs` and `rc_j = cost_j − π·a_j`
//! the reduced cost, mapping back to the convex KKT
//! `c + Aᵀy + Gᵀz − z_lb + z_ub = 0` (see [`crate::qp::QpSolution::kkt_residuals`]):
//! `y = −π_eq`, `z_i = −π_ineq[i] (≥ 0)`, and for a structural `j`,
//! `z_lb_j = max(0, rc_j)`, `z_ub_j = max(0, −rc_j)`.

use crate::ipm::QpOptions;
use crate::qp::{BOUND_INF, QpProblem, QpSolution};
use feral::{
    LuParams, LuPivoting, LuScaling, LuSingularAction, SparseColMatrix, SparseLu, SparseLuSymbolic,
};

/// Feasibility tolerance for bound satisfaction / degenerate-step detection.
const FEAS_TOL: f64 = 1e-9;
/// Pivot-magnitude floor: a basic variable whose rate along the entering
/// direction is below this does not move and cannot block.
const PIVOT_TOL: f64 = 1e-8;
/// Reduced-cost tolerance for entering-variable eligibility / optimality.
const DUAL_TOL: f64 = 1e-9;
/// Rebuild the LU from scratch after this many `update`s (caps update fill /
/// drift even when feral does not itself ask for a refactor).
const REFACTOR_INTERVAL: usize = 100;
/// Consecutive degenerate (≈ zero-length) pivots before latching Bland's rule.
const STALL_LIMIT: u32 = 50;
/// A warm-start structural within this distance of a bound is snapped onto it
/// (nonbasic); farther in is treated as a superbasic for the push to resolve.
const BOUND_SNAP: f64 = 1e-9;
/// Phase-1 iterations without a meaningful infeasibility decrease before
/// latching Bland's rule (guards against non-degenerate cycling/plateaus).
const NO_PROGRESS_LIMIT: u32 = 50;

/// The purified exact-vertex solution, in the convex problem's coordinates.
pub(crate) struct VertexSolution {
    /// Pivot census (gh#990 item 12).
    pub stats: SimplexStats,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub z_lb: Vec<f64>,
    pub z_ub: Vec<f64>,
    pub obj: f64,
}

/// What the crossover simplex did (gh#990 item 12): the superbasics it had to
/// resolve and the basis changes it spent, by stage.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SimplexStats {
    /// Structurals the interior iterate left strictly inside their bounds.
    pub superbasics: usize,
    /// Basis changes made resolving them (the push).
    pub pivots_push: usize,
    /// Basis changes cleaning the residual bound infeasibility (phase 1).
    pub pivots_phase1: usize,
    /// Basis changes optimizing to the vertex (phase 2).
    pub pivots_phase2: usize,
    /// Bound flips (a move to the opposite bound with no basis change).
    pub flips: usize,
}

/// Where a nonbasic variable currently sits.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NbStatus {
    AtLower,
    AtUpper,
    /// Free variable (both bounds infinite), parked at value 0.
    FreeZero,
    /// Held at an interior (non-bound) value from the warm start — a *superbasic*
    /// awaiting resolution by the push phase. Never appears once `push_superbasics`
    /// returns; pricing therefore never sees it.
    Superbasic,
}

/// Outcome of a ratio test.
enum Step {
    /// Pivot: entering enters at `slot`, the leaving basic goes to `to` after a
    /// step of length `theta`.
    Pivot {
        slot: usize,
        theta: f64,
        to: NbStatus,
    },
    /// Bound flip: entering moves `theta` to its opposite bound, no basis change.
    Flip { theta: f64 },
    /// The entering direction is unbounded — no blocker (should not occur on a
    /// problem with a finite optimum; the caller bails to never-regress).
    Unbounded,
}

/// Run the revised-simplex crossover. Returns the exact vertex on success, or
/// `None` on any breakdown (unbounded direction, factorization failure, or
/// iteration cap) — the caller then keeps the interior iterate (never-regress).
pub(crate) fn crossover_simplex(
    prob: &QpProblem,
    sol: &QpSolution,
    _opts: &QpOptions,
) -> Option<VertexSolution> {
    if crate::deadline::expired() {
        return None;
    }
    let mut s = Simplex::new(prob);
    s.warm_start(sol);
    s.factor_basis()?;
    if crate::deadline::expired() {
        return None;
    }
    s.recompute_basics()?; // slacks absorb the residual ⇒ feasible start
    s.push_superbasics()?; // drive interior structurals to bounds / into the basis
    s.run_phase1()?; // clean the residual bound infeasibility (e.g. eq artificials)
    s.run_phase2()?; // optimize to the vertex
    Some(s.extract(prob))
}

/// Where a variable sits at a basic feasible solution, in a form the
/// active-set engine can consume without knowing about simplex internals.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AtBound {
    /// Basic (or free at zero): not part of the active set.
    Free,
    Lower,
    Upper,
}

/// A basic feasible solution: a **vertex** of the feasible region, together
/// with the active set the basis induces.
///
/// Exactly `n` of the `n + m` variables are nonbasic, and each sits on one of
/// its bounds — so `struct_at` and `row_at` together name exactly `n` active
/// constraints in `x`-space, and they are linearly independent *by
/// construction* because the simplex basis matrix is nonsingular. That
/// independence is the property the active-set engine needs and cannot get from
/// tolerance-snapping a point (which at a degenerate vertex names more than `n`
/// binding rows and yields a singular KKT factorization).
pub(crate) struct FeasibleVertex {
    /// Feasible primal point (length `n`).
    pub x: Vec<f64>,
    /// Per-structural bound activity (length `n`).
    pub struct_at: Vec<AtBound>,
    /// Per-row activity (length `m = m_eq + m_ineq`, same row order as
    /// `[A_eq ; G]`). A row is active iff its logical is nonbasic.
    pub row_at: Vec<AtBound>,
}

/// Find a **feasible vertex** of `prob`'s linear constraints from a cold start,
/// running this engine's phase-1 only — phase-2 would optimize the *linear*
/// objective, which is the wrong objective for a QP (any feasible vertex is an
/// equally valid warm start, so the extra work buys nothing).
///
/// Exists because [`crate::active_set`] needs a feasible seed and cannot get
/// one from `pounce-qp`'s own cold start: that engine's l1-elastic phase-1 does
/// not terminate on the degenerate netlib-derived QPs in the Maros-Mészáros
/// set. `QAFIRO` (n = 32, among the smallest instances) runs past 500,000
/// active-set iterations without converging even with Bland's rule forced —
/// finite in exact arithmetic, but the elastic vertices are degenerate enough
/// that the floating-point tolerance gates keep flipping. This engine pivots on
/// an LU basis and walks straight through exactly that degeneracy, which is why
/// crossover already prefers it over the active-set bridge for the NETLIB GEN
/// vertices.
///
/// Returns `None` on any breakdown (factorization failure, iteration cap, or a
/// phase-1 that cannot reach feasibility). The caller must treat that as "no
/// seed available" and fall back, **not** as an infeasibility verdict: a
/// phase-1 that gives up is not a certificate.
pub(crate) fn simplex_feasible_vertex(prob: &QpProblem) -> Option<FeasibleVertex> {
    let mut s = Simplex::new(prob);
    // `warm_start` only reads `sol.x`, clamping each structural into its box and
    // snapping to a bound, so a zero vector is a valid *cold* seed: every
    // structural lands at a bound (or is pushed off by `push_superbasics`) and
    // the logicals start basic, as set up in `new`.
    let zero = QpSolution {
        status: crate::qp::QpStatus::Optimal,
        x: vec![0.0; prob.n],
        y: Vec::new(),
        z: Vec::new(),
        z_lb: Vec::new(),
        z_ub: Vec::new(),
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    };
    let dbg = std::env::var("POUNCE_SEED_DEBUG").is_ok();
    macro_rules! stage {
        ($e:expr, $name:literal) => {
            match $e {
                Some(v) => v,
                None => {
                    if dbg {
                        eprintln!("[seed] failed at {}", $name);
                    }
                    return None;
                }
            }
        };
    }
    s.warm_start(&zero);
    stage!(s.factor_basis(), "factor_basis");
    stage!(s.recompute_basics(), "recompute_basics");
    stage!(s.push_superbasics(), "push_superbasics");
    stage!(s.run_phase1(), "run_phase1");

    let x: Vec<f64> = (0..s.n).map(|j| s.xval[j]).collect();
    // `run_phase1` returning `Some` means it stopped, not that it succeeded —
    // confirm the point really is feasible before offering it as a seed.
    if !feasible_at(prob, &x) {
        if dbg {
            eprintln!("[seed] run_phase1 returned but point is INFEASIBLE");
        }
        return None;
    }

    let at = |v: usize| -> AtBound {
        if s.slot_of[v] != usize::MAX {
            return AtBound::Free; // basic
        }
        match s.nb[v] {
            NbStatus::AtLower => AtBound::Lower,
            NbStatus::AtUpper => AtBound::Upper,
            // A free variable parked at zero pins no constraint, and
            // `push_superbasics` has already cleared every superbasic.
            NbStatus::FreeZero | NbStatus::Superbasic => AtBound::Free,
        }
    };

    Some(FeasibleVertex {
        struct_at: (0..s.n).map(at).collect(),
        row_at: (0..s.m).map(|i| at(s.n + i)).collect(),
        x,
    })
}

/// Does `x` satisfy `Ax = b`, `Gx ≤ h`, and the variable box, to `FEAS_TOL`?
fn feasible_at(prob: &QpProblem, x: &[f64]) -> bool {
    let mut ax = vec![0.0; prob.m_eq()];
    for t in &prob.a {
        ax[t.row] += t.val * x[t.col];
    }
    for (i, &v) in ax.iter().enumerate() {
        if (v - prob.b[i]).abs() > FEAS_TOL {
            return false;
        }
    }

    let mut gx = vec![0.0; prob.m_ineq()];
    for t in &prob.g {
        gx[t.row] += t.val * x[t.col];
    }
    for (i, &v) in gx.iter().enumerate() {
        if v - prob.h[i] > FEAS_TOL {
            return false;
        }
    }

    for (i, &xi) in x.iter().enumerate() {
        if xi < lo_bound(prob.lb_of(i)) - FEAS_TOL || xi > hi_bound(prob.ub_of(i)) + FEAS_TOL {
            return false;
        }
    }
    true
}

/// Treat `|v| ≥ 1e20` as an infinite bound (mirrors [`QpProblem`] conventions).
fn lo_bound(v: f64) -> f64 {
    if v <= -BOUND_INF {
        f64::NEG_INFINITY
    } else {
        v
    }
}
fn hi_bound(v: f64) -> f64 {
    if v >= BOUND_INF { f64::INFINITY } else { v }
}

struct Simplex {
    /// Number of structural variables `n`.
    n: usize,
    /// Number of equality rows.
    m_eq: usize,
    /// Total constraint rows `m = m_eq + m_ineq` (= basis dimension).
    m: usize,
    /// Total variables `nv = n + m`.
    nv: usize,

    /// Sparse column of each variable in `M`, `(row, val)`, row-space.
    cols: Vec<Vec<(usize, f64)>>,
    /// `cost` for every variable (`c` for structural, 0 for logical).
    cost: Vec<f64>,
    /// Lower / upper bound of every variable (`±∞` allowed).
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// Right-hand side `[b ; h]`, row-space.
    rhs: Vec<f64>,

    /// `basis[slot]` = variable index occupying that basis slot.
    basis: Vec<usize>,
    /// `slot_of[v]` = slot if `v` is basic, else `usize::MAX`.
    slot_of: Vec<usize>,
    /// Nonbasic status (meaningful only where `slot_of[v] == MAX`).
    nb: Vec<NbStatus>,
    /// Current value of every variable.
    xval: Vec<f64>,

    /// LU factor of the current basis `B` (columns in slot order).
    lu: Option<SparseLu>,
    /// `update`s applied since the last full refactor.
    since_refactor: usize,

    /// Bland anti-cycling latch + degenerate-step counter.
    bland: bool,
    stall: u32,
    /// Pivot census and the stage (1 or 2) the pricing loop is in.
    stats: SimplexStats,
    phase: u8,
}

impl Simplex {
    fn new(prob: &QpProblem) -> Self {
        let n = prob.n;
        let m_eq = prob.m_eq();
        let m_ineq = prob.m_ineq();
        let m = m_eq + m_ineq;
        let nv = n + m;

        // Column of each variable. Structural columns are accumulated from the
        // A/G triplets (summing any duplicate entries); logicals are unit cols.
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nv];
        for t in &prob.a {
            cols[t.col].push((t.row, t.val));
        }
        for t in &prob.g {
            cols[t.col].push((m_eq + t.row, t.val));
        }
        for j in 0..n {
            consolidate(&mut cols[j]);
        }
        for i in 0..m {
            cols[n + i].push((i, 1.0));
        }

        let mut cost = vec![0.0; nv];
        cost[..n].copy_from_slice(&prob.c);

        let mut lo = vec![0.0; nv];
        let mut hi = vec![0.0; nv];
        for j in 0..n {
            lo[j] = lo_bound(prob.lb_of(j));
            hi[j] = hi_bound(prob.ub_of(j));
        }
        for i in 0..m_eq {
            // Equality artificial: fixed at zero.
            lo[n + i] = 0.0;
            hi[n + i] = 0.0;
        }
        for i in 0..m_ineq {
            // Inequality slack: s = h − Gx ≥ 0.
            lo[n + m_eq + i] = 0.0;
            hi[n + m_eq + i] = f64::INFINITY;
        }

        let mut rhs = vec![0.0; m];
        rhs[..m_eq].copy_from_slice(&prob.b);
        rhs[m_eq..].copy_from_slice(&prob.h);

        // Initial basis: all logicals (B = I).
        let basis: Vec<usize> = (n..nv).collect();
        let mut slot_of = vec![usize::MAX; nv];
        for (slot, &v) in basis.iter().enumerate() {
            slot_of[v] = slot;
        }

        Simplex {
            n,
            m_eq,
            m,
            nv,
            cols,
            cost,
            lo,
            hi,
            rhs,
            basis,
            slot_of,
            nb: vec![NbStatus::AtLower; nv],
            xval: vec![0.0; nv],
            lu: None,
            since_refactor: 0,
            bland: false,
            stall: 0,
            stats: SimplexStats::default(),
            phase: 0,
        }
    }

    /// Feasibility-preserving warm start. Every structural is nonbasic at its
    /// (bound-clamped) IPM value; one within `BOUND_SNAP` of a bound is snapped to
    /// that bound, otherwise it is held *interior* as a [`NbStatus::Superbasic`]
    /// for the push phase. The logicals stay basic (set in `new`), so after
    /// [`Self::recompute_basics`] the slacks absorb the row residuals — the
    /// starting point satisfies `M w = rhs` exactly and the only bound
    /// infeasibility is the IPM's own (tiny) residual, not the huge artificial
    /// gap a cold all-slack start would have.
    fn warm_start(&mut self, sol: &QpSolution) {
        for j in 0..self.n {
            let (lo, hi) = (self.lo[j], self.hi[j]);
            // Written as max-then-min rather than `clamp`, which *panics* on an
            // inverted interval (`min > max`). Callers are expected to have
            // screened an empty box out before reaching here — both convex
            // drivers do — but this is a seed builder, not a feasibility
            // verdict, and it must not be the thing that takes the process down
            // if one ever does not: the panic crossed the PyO3 boundary as an
            // uncatchable `PanicException` (gh #491). On a reversed box this
            // parks the variable at `hi`; the solve that follows is answering a
            // problem with no feasible point either way.
            let xj = sol.x.get(j).copied().unwrap_or(0.0).max(lo).min(hi);
            let (status, val) = if lo.is_finite() && (xj - lo).abs() <= BOUND_SNAP {
                (NbStatus::AtLower, lo)
            } else if hi.is_finite() && (hi - xj).abs() <= BOUND_SNAP {
                (NbStatus::AtUpper, hi)
            } else if !lo.is_finite() && !hi.is_finite() && xj == 0.0 {
                (NbStatus::FreeZero, 0.0)
            } else {
                // Strictly interior to its bounds: a superbasic to be pushed out.
                (NbStatus::Superbasic, xj)
            };
            self.nb[j] = status;
            self.xval[j] = val;
        }
    }

    /// Crossover **push** (basis identification). The warm start leaves a feasible
    /// point whose only non-vertex feature is a set of *superbasic* structurals
    /// held at interior values. This resolves each one — moving it to a bound (it
    /// stays nonbasic) or into the basis (a blocking logical leaves) — by a
    /// feasibility-preserving bounded ratio test, so the point stays feasible
    /// throughout. After it returns there are no superbasics: a valid basic
    /// feasible solution from which phase-1/phase-2 run.
    ///
    /// Each superbasic is resolved by exactly one ratio test, so the push is
    /// `O(#superbasic)` pivots with no risk of cycling.
    fn push_superbasics(&mut self) -> Option<()> {
        let dbg = std::env::var("POUNCE_SIMPLEX_DEBUG").is_ok();
        let supers: Vec<usize> = (0..self.n)
            .filter(|&j| self.slot_of[j] == usize::MAX && self.nb[j] == NbStatus::Superbasic)
            .collect();
        let n_super = supers.len();
        self.stats.superbasics = n_super;
        for j in supers {
            if self.slot_of[j] != usize::MAX || self.nb[j] != NbStatus::Superbasic {
                continue; // already resolved as a side effect of an earlier pivot
            }
            self.resolve_superbasic(j)?;
        }
        // gh#984: the pushes move the basics by rank-one updates, so the
        // recompute is the numerics, not the diagnostic -- it ran only under
        // `POUNCE_SIMPLEX_DEBUG`, which therefore changed the returned vertex.
        // Always do it; the switch only decides whether to *print*.
        self.recompute_basics()?;
        if dbg {
            eprintln!(
                "[simplex push] n={} m={} superbasics={} infeas_after_push={:.3e}",
                self.n,
                self.m,
                n_super,
                self.primal_infeasibility()
            );
        }
        Some(())
    }

    /// Resolve one superbasic structural `j`: move it toward its nearer finite
    /// bound (free variables toward the side that the largest pivot allows) until
    /// either it reaches that bound (resolved nonbasic) or a basic variable
    /// reaches a bound first (pivot `j` in, that basic leaves — resolved basic).
    /// The ratio test only allows steps that keep every basic within bounds, so
    /// feasibility is preserved.
    fn resolve_superbasic(&mut self, j: usize) -> Option<()> {
        let xj = self.xval[j];
        let lo = self.lo[j];
        let hi = self.hi[j];
        // Direction toward the nearer finite bound (ties / free → increase).
        let dist_lo = if lo.is_finite() {
            xj - lo
        } else {
            f64::INFINITY
        };
        let dist_hi = if hi.is_finite() {
            hi - xj
        } else {
            f64::INFINITY
        };
        let (t, own_bound, own_to) = if dist_hi <= dist_lo {
            (1.0, hi, NbStatus::AtUpper) // increase toward upper
        } else {
            (-1.0, lo, NbStatus::AtLower) // decrease toward lower
        };
        let own_range = if own_bound.is_finite() {
            (own_bound - xj).abs()
        } else {
            f64::INFINITY
        };

        let alpha = self.ftran_col(j)?;
        // Min-ratio over basics, largest-|rate| tie-break for stability.
        let mut best_theta = own_range;
        let mut best_slot: Option<usize> = None;
        let mut best_to = NbStatus::AtLower;
        let mut best_rate = 0.0;
        for slot in 0..self.m {
            let rate = alpha[slot] * t; // basic moves as x − rate·θ
            if rate.abs() <= PIVOT_TOL {
                continue;
            }
            let v = self.basis[slot];
            let x = self.xval[v];
            let (theta, to) = if rate > 0.0 {
                if self.lo[v].is_finite() {
                    ((x - self.lo[v]) / rate, NbStatus::AtLower)
                } else {
                    continue;
                }
            } else if self.hi[v].is_finite() {
                ((x - self.hi[v]) / rate, NbStatus::AtUpper)
            } else {
                continue;
            };
            let theta = theta.max(0.0);
            let better = theta < best_theta - FEAS_TOL
                || ((theta - best_theta).abs() <= FEAS_TOL && rate.abs() > best_rate);
            if better {
                best_theta = theta;
                best_slot = Some(slot);
                best_to = to;
                best_rate = rate.abs();
            }
        }

        let theta = best_theta;
        let delta = t * theta;
        // Move along the ray (incremental; resync happens on refactor / phase entry).
        for slot in 0..self.m {
            self.xval[self.basis[slot]] -= alpha[slot] * delta;
        }
        self.xval[j] += delta;

        match best_slot {
            // j reached its own bound first: it stays nonbasic, snapped there.
            None => {
                self.xval[j] = if own_bound.is_finite() {
                    own_bound
                } else {
                    self.xval[j]
                };
                self.nb[j] = own_to;
                Some(())
            }
            // A basic blocked first: pivot j into that slot.
            Some(slot) => {
                let leaving = self.basis[slot];
                self.xval[leaving] = match best_to {
                    NbStatus::AtLower => self.lo[leaving],
                    NbStatus::AtUpper => self.hi[leaving],
                    _ => self.xval[leaving],
                };
                self.nb[leaving] = best_to;
                self.slot_of[leaving] = usize::MAX;
                self.basis[slot] = j;
                self.slot_of[j] = slot;
                self.pivot_lu(slot, j)
            }
        }
    }

    /// Update the LU for replacing the basis column in `slot` with variable `var`,
    /// refactoring on any feral refusal or periodically to cap fill / drift.
    fn pivot_lu(&mut self, slot: usize, var: usize) -> Option<()> {
        self.stats.pivots_push += 1;
        self.since_refactor += 1;
        let mut entering_col = vec![0.0; self.m];
        for &(row, val) in &self.cols[var] {
            entering_col[row] += val;
        }
        let need_refactor = self.since_refactor >= REFACTOR_INTERVAL
            || self.lu.as_mut()?.update(slot, &entering_col).is_err();
        if need_refactor {
            self.factor_basis()?;
            self.recompute_basics()?;
        }
        Some(())
    }

    /// Build the basis matrix and factor it.
    fn factor_basis(&mut self) -> Option<()> {
        let basis_cols: Vec<Vec<(usize, f64)>> =
            self.basis.iter().map(|&v| self.cols[v].clone()).collect();
        let scm = SparseColMatrix::from_sparse_columns(self.m, &basis_cols).ok()?;
        let sym = SparseLuSymbolic::analyze(&scm).ok()?;
        let lu = SparseLu::factor(&scm, &sym, lu_params()).ok()?;
        self.lu = Some(lu);
        self.since_refactor = 0;
        Some(())
    }

    /// `x_B = B⁻¹ (rhs − N x_N)`, written back into `xval`.
    fn recompute_basics(&mut self) -> Option<()> {
        let mut r = self.rhs.clone();
        for v in 0..self.nv {
            if self.slot_of[v] != usize::MAX {
                continue; // basic
            }
            let xv = self.xval[v];
            if xv == 0.0 {
                continue;
            }
            for &(row, val) in &self.cols[v] {
                r[row] -= val * xv;
            }
        }
        self.lu.as_mut()?.ftran(&mut r).ok()?;
        for slot in 0..self.m {
            self.xval[self.basis[slot]] = r[slot];
        }
        Some(())
    }

    /// `α = B⁻¹ a_q` (slot-space), where `a_q` is the entering column.
    fn ftran_col(&mut self, var: usize) -> Option<Vec<f64>> {
        let mut col = vec![0.0; self.m];
        for &(row, val) in &self.cols[var] {
            col[row] += val;
        }
        self.lu.as_mut()?.ftran(&mut col).ok()?;
        Some(col)
    }

    /// `π = B⁻ᵀ c_B` (row-space) for the given basic-cost vector.
    fn btran(&mut self, cost_b: &[f64]) -> Option<Vec<f64>> {
        let mut pi = cost_b.to_vec();
        self.lu.as_mut()?.btran(&mut pi).ok()?;
        Some(pi)
    }

    /// Reduced cost `rc_v = cost_v − π·a_v` for the given objective costs.
    fn reduced_cost(&self, var: usize, pi: &[f64], cost: &[f64]) -> f64 {
        let mut rc = cost[var];
        for &(row, val) in &self.cols[var] {
            rc -= pi[row] * val;
        }
        rc
    }

    // ---- Phase 1: composite (minimize the sum of bound infeasibilities) ----

    /// Total bound infeasibility of the basic variables.
    fn primal_infeasibility(&self) -> f64 {
        let mut s = 0.0;
        for &v in &self.basis {
            let x = self.xval[v];
            if x < self.lo[v] - FEAS_TOL {
                s += self.lo[v] - x;
            } else if x > self.hi[v] + FEAS_TOL {
                s += x - self.hi[v];
            }
        }
        s
    }

    /// Phase-1 cost gradient of the *basic* variables: `−1` below lower, `+1`
    /// above upper, `0` feasible. (Nonbasic phase-1 cost is 0.)
    fn phase1_cost_b(&self) -> Vec<f64> {
        let mut cb = vec![0.0; self.m];
        for slot in 0..self.m {
            let v = self.basis[slot];
            let x = self.xval[v];
            cb[slot] = if x < self.lo[v] - FEAS_TOL {
                -1.0
            } else if x > self.hi[v] + FEAS_TOL {
                1.0
            } else {
                0.0
            };
        }
        cb
    }

    fn run_phase1(&mut self) -> Option<()> {
        self.phase = 1;
        if self.primal_infeasibility() <= FEAS_TOL {
            return Some(());
        }
        let max_iter = 20 * (self.m + self.nv) + 1000;
        self.bland = false;
        self.stall = 0;
        let dbg = std::env::var("POUNCE_SIMPLEX_DEBUG").is_ok();
        let zero_cost = vec![0.0; self.nv];
        // No-progress tracking: latch Bland's rule if the infeasibility fails to
        // drop meaningfully for NO_PROGRESS_LIMIT consecutive iterations (catches
        // non-degenerate plateaus the per-pivot `stall` counter misses).
        let mut best_infeas = self.primal_infeasibility();
        let mut no_progress: u32 = 0;
        for _it in 0..max_iter {
            if crate::deadline::expired() {
                return None;
            }
            if dbg && _it % 100 == 0 {
                eprintln!(
                    "[simplex p1] it={_it} infeas={:.3e} bland={}",
                    self.primal_infeasibility(),
                    self.bland
                );
            }
            let cb = self.phase1_cost_b();
            let pi = self.btran(&cb)?;
            // Price: entering must decrease the phase-1 objective.
            let entering = self.price(&pi, &zero_cost)?;
            let (q, t) = match entering {
                Some(e) => e,
                None => {
                    // No improving direction: feasible iff infeasibility is 0.
                    return if self.primal_infeasibility() <= 1e-7 {
                        Some(())
                    } else {
                        None // primal infeasible — bail to never-regress
                    };
                }
            };
            let alpha = self.ftran_col(q)?;
            let rc = self.reduced_cost(q, &pi, &zero_cost);
            let step = self.ratio_test_phase1(q, t, rc, &alpha);
            self.apply_step(q, t, &alpha, step)?;
            // Resync basic values exactly from the factor: the long step relies on
            // accurate `xval`/infeasibility, and incremental updates drift across
            // the many `update`s between refactors.
            self.recompute_basics()?;
            let infeas = self.primal_infeasibility();
            if infeas <= FEAS_TOL {
                return Some(());
            }
            if infeas < best_infeas - FEAS_TOL {
                best_infeas = infeas;
                no_progress = 0;
                // Release the latch once progress resumes.
                //
                // Bland's rule is an anti-cycling *escape*, not a pricing
                // strategy: it is finite but can take exponentially many steps,
                // so leaving it on permanently trades a stall for a crawl. It
                // was latched here and never cleared, and the cost is stark on
                // `QSCTAP1` — Dantzig pricing drove infeasibility 6.5e2 → 5.0e0
                // in 200 iterations, then the latch engaged and infeasibility
                // sat at 3.25e0 for the next 40,000 iterations until the cap
                // ran out. Phase-1 then returned `None`, the caller lost its
                // feasible seed, and the QP fell back to the non-terminating
                // l1-elastic path — which is why a large block of
                // Maros-Mészáros problems timed out rather than solving.
                //
                // Latch on stall, release on progress, is the standard
                // discipline: the anti-cycling guarantee is only needed while
                // actually stalled.
                self.bland = false;
            } else {
                no_progress += 1;
                if no_progress >= NO_PROGRESS_LIMIT {
                    self.bland = true;
                }
            }
        }
        None
    }

    fn run_phase2(&mut self) -> Option<()> {
        self.phase = 2;
        let max_iter = 20 * (self.m + self.nv) + 1000;
        self.bland = false;
        self.stall = 0;
        let cost = self.cost.clone();
        for _ in 0..max_iter {
            if crate::deadline::expired() {
                return None;
            }
            let cb: Vec<f64> = (0..self.m).map(|slot| cost[self.basis[slot]]).collect();
            let pi = self.btran(&cb)?;
            let entering = self.price(&pi, &cost)?;
            let (q, t) = match entering {
                Some(e) => e,
                None => return Some(()), // optimal
            };
            let alpha = self.ftran_col(q)?;
            let step = self.ratio_test_phase2(q, t, &alpha);
            match step {
                Step::Unbounded => return None,
                _ => self.apply_step(q, t, &alpha, step)?,
            }
        }
        None
    }

    /// Select an entering variable and its direction `t` (`+1` increase from a
    /// lower bound, `−1` decrease from an upper bound). Returns `None` when no
    /// nonbasic variable improves the objective (optimal for this phase). With
    /// the Bland latch, returns the lowest-index eligible variable.
    fn price(&self, pi: &[f64], cost: &[f64]) -> Option<Option<(usize, f64)>> {
        let mut best: Option<(usize, f64)> = None;
        let mut best_score = DUAL_TOL;
        for v in 0..self.nv {
            if self.slot_of[v] != usize::MAX {
                continue; // basic
            }
            let rc = self.reduced_cost(v, pi, cost);
            let t = match self.nb[v] {
                NbStatus::AtLower => {
                    if rc < -DUAL_TOL {
                        1.0
                    } else {
                        continue;
                    }
                }
                NbStatus::AtUpper => {
                    if rc > DUAL_TOL {
                        -1.0
                    } else {
                        continue;
                    }
                }
                NbStatus::FreeZero => {
                    if rc < -DUAL_TOL {
                        1.0
                    } else if rc > DUAL_TOL {
                        -1.0
                    } else {
                        continue;
                    }
                }
                // No superbasics survive the push, so pricing never sees one.
                NbStatus::Superbasic => continue,
            };
            if self.bland {
                // Lowest-index eligible — provably finite.
                return Some(Some((v, t)));
            }
            let score = rc.abs();
            if score > best_score {
                best_score = score;
                best = Some((v, t));
            }
        }
        Some(best)
    }

    /// Phase-2 ratio test: all basics feasible, objective linear ⇒ stop at the
    /// first breakpoint (standard min-ratio), or flip if the entering variable
    /// reaches its opposite bound first.
    fn ratio_test_phase2(&self, q: usize, t: f64, alpha: &[f64]) -> Step {
        // Entering's own range (bound flip distance).
        let flip = self.entering_range(q);
        let mut best_theta = flip;
        let mut best_slot: Option<usize> = None;
        let mut best_to = NbStatus::AtLower;
        let mut best_rate = 0.0;
        let mut unbounded = !flip.is_finite();
        for slot in 0..self.m {
            let rate = alpha[slot] * t; // x_B[slot] decreases at this rate
            if rate.abs() <= PIVOT_TOL {
                continue;
            }
            let v = self.basis[slot];
            let x = self.xval[v];
            // Bound the basic variable heads toward.
            let (theta, to) = if rate > 0.0 {
                // decreasing → lower bound
                if self.lo[v].is_finite() {
                    ((x - self.lo[v]) / rate, NbStatus::AtLower)
                } else {
                    continue;
                }
            } else {
                // increasing → upper bound
                if self.hi[v].is_finite() {
                    ((x - self.hi[v]) / rate, NbStatus::AtUpper)
                } else {
                    continue;
                }
            };
            let theta = theta.max(0.0);
            unbounded = false;
            let better = if self.bland {
                // Bland: among min-ratio ties, lowest variable index.
                theta < best_theta - FEAS_TOL
                    || ((theta - best_theta).abs() <= FEAS_TOL
                        && best_slot.is_some_and(|bs| v < self.basis[bs]))
                    || best_slot.is_none() && theta <= best_theta + FEAS_TOL
            } else {
                // Dantzig-ish: smallest ratio, ties broken by largest |rate|.
                theta < best_theta - FEAS_TOL
                    || ((theta - best_theta).abs() <= FEAS_TOL && rate.abs() > best_rate)
            };
            if better {
                best_theta = theta;
                best_slot = Some(slot);
                best_to = to;
                best_rate = rate.abs();
            }
        }
        if unbounded {
            return Step::Unbounded;
        }
        match best_slot {
            Some(slot) => Step::Pivot {
                slot,
                theta: best_theta,
                to: best_to,
            },
            None => Step::Flip { theta: best_theta },
        }
    }

    /// The bound crossings the basic variables meet **ahead** on the entering ray
    /// `x_B(θ) = x_B − α·t·θ`, as `(θ, slot, status-the-leaver-takes, |rate|)`.
    /// Shared by both phase-1 ratio tests so they agree on what a blocker is.
    ///
    /// Which bounds lie ahead is decided by where the basic sits *now*, using the
    /// same three-way test as [`Self::phase1_cost_b`] so that the breakpoint set
    /// and the slope `f'(0) = rc·t` stay consistent:
    ///
    /// * feasible → it can only leave through the single bound it is heading for;
    /// * below its lower bound → it crosses nothing unless the ray lifts it
    ///   (`rate < 0`), and then hits its lower bound (regaining feasibility — a
    ///   legitimate blocker, this is how an infeasibility gets *removed*) before
    ///   its upper bound;
    /// * above its upper bound → the mirror image.
    ///
    /// The `θ ≈ 0` crossings count as much as the far ones: a basic sitting *on*
    /// the bound the ray pushes it out through is a breakpoint at `θ = 0`. What is
    /// *not* a breakpoint is the bound a basic sits on and moves **away** from —
    /// that crossing is behind the ray (`θ ≤ 0` only because `x = bound`), and
    /// admitting it invents a zero-length blocker.
    fn phase1_breakpoints(&self, t: f64, alpha: &[f64]) -> Vec<(f64, usize, NbStatus, f64)> {
        let mut bps: Vec<(f64, usize, NbStatus, f64)> = Vec::new();
        for slot in 0..self.m {
            let rate = alpha[slot] * t; // basic value moves as x − rate·θ
            if rate.abs() <= PIVOT_TOL {
                continue;
            }
            let v = self.basis[slot];
            let x = self.xval[v];
            let arate = rate.abs();
            let (lo, hi) = (self.lo[v], self.hi[v]);
            if x < lo - FEAS_TOL {
                if rate < 0.0 {
                    bps.push(((x - lo) / rate, slot, NbStatus::AtLower, arate));
                    if hi.is_finite() {
                        bps.push(((x - hi) / rate, slot, NbStatus::AtUpper, arate));
                    }
                }
            } else if x > hi + FEAS_TOL {
                if rate > 0.0 {
                    bps.push(((x - hi) / rate, slot, NbStatus::AtUpper, arate));
                    if lo.is_finite() {
                        bps.push(((x - lo) / rate, slot, NbStatus::AtLower, arate));
                    }
                }
            } else if rate > 0.0 {
                // Feasible and decreasing: it can only leave through its lower bound.
                if lo.is_finite() {
                    bps.push((((x - lo) / rate).max(0.0), slot, NbStatus::AtLower, arate));
                }
            } else {
                // Feasible and increasing: it can only leave through its upper bound.
                if hi.is_finite() {
                    bps.push((((x - hi) / rate).max(0.0), slot, NbStatus::AtUpper, arate));
                }
            }
        }
        bps
    }

    /// Phase-1 long-step (Maros) ratio test. The piecewise-linear infeasibility
    /// objective along the entering ray has directional derivative `f'(0) = rc·t <
    /// 0`, and `f'` is convex/increasing: every breakpoint (a basic reaching a
    /// bound) adds `|rate|` to the slope — whether the basic goes
    /// infeasible→feasible or feasible→infeasible, the contribution is `+|rate|`
    /// (derivation in the module notes). We walk breakpoints in increasing `θ`,
    /// accumulate the slope, and stop at the first breakpoint where it turns
    /// nonnegative; that variable leaves. Passing earlier breakpoints — the whole
    /// point of the long step — is what drives the simplex *through* degenerate
    /// (`θ ≈ 0`) blockers that defeat a first-breakpoint short step on the GEN
    /// vertices.
    ///
    /// The slope bookkeeping is only valid if *every* breakpoint ahead is in the
    /// list, including the ones at `θ = 0`: see [`Self::phase1_breakpoints`]. Omit
    /// them and the accumulated slope is a systematic underestimate of `f'`, the
    /// walk sails past the breakpoint where `f'` really turns nonnegative, and the
    /// step lands beyond the ray's minimizer — i.e. phase-1 *increases* the
    /// infeasibility it exists to remove.
    fn ratio_test_phase1(&self, q: usize, t: f64, rc: f64, alpha: &[f64]) -> Step {
        // Bland's anti-cycling guarantee requires the leaving variable to be the
        // lowest-index one among the minimum-ratio ties — which the long step does
        // NOT honor (it stops by slope, not by min ratio). So once the Bland latch
        // is on, fall back to the Bland-correct first-breakpoint short step; it is
        // provably finite and the only thing that escapes the degenerate plateaus.
        if self.bland {
            return self.ratio_test_phase1_bland(q, t, alpha);
        }
        let mut bps = self.phase1_breakpoints(t, alpha);
        // The entering variable's own opposite bound caps the step (a flip).
        let flip = self.entering_range(q);

        bps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut slope = rc * t; // initial directional derivative (< 0)
        for &(theta, slot, to, arate) in &bps {
            if theta >= flip {
                break; // the flip happens first
            }
            slope += arate;
            if slope >= -DUAL_TOL {
                return Step::Pivot { slot, theta, to };
            }
        }
        if flip.is_finite() {
            Step::Flip { theta: flip }
        } else {
            // No breakpoint restored nonnegativity and the entering variable is
            // unbounded: the infeasibility objective would be unbounded below,
            // which cannot happen (it is bounded by 0). Treat as a breakdown.
            Step::Unbounded
        }
    }

    /// Bland-correct phase-1 ratio test: step to the *first* bound any basic
    /// reaches (minimum ratio over [`Self::phase1_breakpoints`]), breaking ties by
    /// lowest leaving-variable index. No basic crosses a bound during the step, so
    /// it is monotone, and the lowest-index tie-break makes the entering/leaving
    /// pair Bland-consistent — together that guarantees finite termination through
    /// degeneracy.
    ///
    /// It shares the breakpoint set with the long step deliberately: this test used
    /// to scan *both* bounds of every basic and accept any ratio `≥ −FEAS_TOL`,
    /// which admits the bound a basic already sits on and is moving **away** from
    /// (that ratio is `±0.0`, not negative). Every such phantom blocker pinned
    /// `θ = 0`, so the Bland escape could not move at all — `QSCSD6` sat at
    /// `infeas = 5.25` for its whole iteration budget and then failed.
    fn ratio_test_phase1_bland(&self, q: usize, t: f64, alpha: &[f64]) -> Step {
        let flip = self.entering_range(q);
        let mut best_theta = flip;
        let mut best_slot: Option<usize> = None;
        let mut best_to = NbStatus::AtLower;
        for (theta, slot, to, _) in self.phase1_breakpoints(t, alpha) {
            let v = self.basis[slot];
            let better = theta < best_theta - FEAS_TOL
                || ((theta - best_theta).abs() <= FEAS_TOL
                    && best_slot.is_none_or(|bs| v < self.basis[bs]));
            if better {
                best_theta = theta;
                best_slot = Some(slot);
                best_to = to;
            }
        }
        match best_slot {
            Some(slot) => Step::Pivot {
                slot,
                theta: best_theta,
                to: best_to,
            },
            None if flip.is_finite() => Step::Flip { theta: flip },
            None => Step::Unbounded,
        }
    }

    /// Distance the entering variable can travel to its opposite bound (`∞` if
    /// that bound is infinite).
    fn entering_range(&self, q: usize) -> f64 {
        if self.lo[q].is_finite() && self.hi[q].is_finite() {
            self.hi[q] - self.lo[q]
        } else {
            f64::INFINITY
        }
    }

    /// Apply a ratio-test outcome: move along the ray, update the basis and LU.
    fn apply_step(&mut self, q: usize, t: f64, alpha: &[f64], step: Step) -> Option<()> {
        match step {
            Step::Unbounded => None,
            Step::Flip { theta } => {
                let delta = t * theta;
                for slot in 0..self.m {
                    self.xval[self.basis[slot]] -= alpha[slot] * delta;
                }
                self.xval[q] += delta;
                self.stats.flips += 1;
                self.nb[q] = match self.nb[q] {
                    NbStatus::AtLower => NbStatus::AtUpper,
                    NbStatus::AtUpper => NbStatus::AtLower,
                    // Unreachable: a flip needs two finite bounds, and superbasics
                    // are gone before pricing runs.
                    NbStatus::FreeZero | NbStatus::Superbasic => self.nb[q],
                };
                Some(())
            }
            Step::Pivot { slot, theta, to } => {
                if self.phase == 1 {
                    self.stats.pivots_phase1 += 1;
                } else {
                    self.stats.pivots_phase2 += 1;
                }
                let delta = t * theta;
                for s in 0..self.m {
                    self.xval[self.basis[s]] -= alpha[s] * delta;
                }
                self.xval[q] += delta;

                let leaving = self.basis[slot];
                // Snap the leaving variable exactly onto the bound it hit.
                self.xval[leaving] = match to {
                    NbStatus::AtLower => self.lo[leaving],
                    NbStatus::AtUpper => self.hi[leaving],
                    NbStatus::FreeZero | NbStatus::Superbasic => 0.0,
                };
                self.nb[leaving] = to;
                self.slot_of[leaving] = usize::MAX;

                // Entering takes the slot.
                self.basis[slot] = q;
                self.slot_of[q] = slot;

                // Degeneracy / anti-cycling bookkeeping.
                if theta.abs() <= FEAS_TOL {
                    self.stall += 1;
                    if self.stall >= STALL_LIMIT {
                        self.bland = true;
                    }
                } else {
                    self.stall = 0;
                }

                // Update the LU factor (column replacement), refactoring on any
                // feral refusal or periodically to cap fill / drift.
                self.since_refactor += 1;
                let mut entering_col = vec![0.0; self.m];
                for &(row, val) in &self.cols[q] {
                    entering_col[row] += val;
                }
                let need_refactor = self.since_refactor >= REFACTOR_INTERVAL
                    || self.lu.as_mut()?.update(slot, &entering_col).is_err();
                if need_refactor {
                    self.factor_basis()?;
                    self.recompute_basics()?;
                }
                Some(())
            }
        }
    }

    /// Map the final basis to the convex problem's primal/dual solution.
    fn extract(&mut self, prob: &QpProblem) -> VertexSolution {
        let n = self.n;
        // gh#984: after the final pivot `x_B` carries every rank-one update
        // since the last refactor (measured: complementarity `2.8e-8` on a
        // 12-variable transportation LP). Recompute `x_B = B⁻¹(b − N x_N)` so
        // the vertex is exact. On a refusal keep the running values: they are
        // what this function returned before.
        let _ = self.recompute_basics();
        let x: Vec<f64> = (0..n).map(|j| self.xval[j]).collect();

        // π = B⁻ᵀ c_B (row-space) with the *real* objective.
        let cost = self.cost.clone();
        let cb: Vec<f64> = (0..self.m).map(|slot| cost[self.basis[slot]]).collect();
        let pi = self.btran(&cb).unwrap_or_else(|| vec![0.0; self.m]);

        // y = −π_eq ; z = −π_ineq (≥ 0).
        let y: Vec<f64> = (0..self.m_eq).map(|i| -pi[i]).collect();
        let z: Vec<f64> = (self.m_eq..self.m).map(|i| (-pi[i]).max(0.0)).collect();

        // Bound duals from structural reduced costs: rc = z_lb − z_ub.
        let mut z_lb = vec![0.0; n];
        let mut z_ub = vec![0.0; n];
        for j in 0..n {
            if self.slot_of[j] != usize::MAX {
                continue; // basic ⇒ rc ≈ 0
            }
            let rc = self.reduced_cost(j, &pi, &cost);
            z_lb[j] = rc.max(0.0);
            z_ub[j] = (-rc).max(0.0);
        }

        let obj = (0..n).map(|j| prob.c[j] * x[j]).sum();
        VertexSolution {
            stats: self.stats,
            x,
            y,
            z,
            z_lb,
            z_ub,
            obj,
        }
    }
}

/// Sum duplicate `(row, val)` entries in a column so the matvec and the LU see
/// a canonical column (NL extraction can in principle emit split entries).
fn consolidate(col: &mut Vec<(usize, f64)>) {
    if col.len() <= 1 {
        return;
    }
    col.sort_by_key(|&(r, _)| r);
    let mut out: Vec<(usize, f64)> = Vec::with_capacity(col.len());
    for &(r, v) in col.iter() {
        if let Some(last) = out.last_mut() {
            if last.0 == r {
                last.1 += v;
                continue;
            }
        }
        out.push((r, v));
    }
    out.retain(|&(_, v)| v != 0.0);
    *col = out;
}

/// LU parameters for the basis: perturb numerically-null pivots to a small
/// floor (degenerate, redundant-row bases are common in crossover) rather than
/// hard-failing, and leave scaling off (the basis columns are O(1)).
fn lu_params() -> LuParams {
    LuParams {
        on_singular: LuSingularAction::PerturbToEps { abs_floor: 1e-12 },
        scaling: LuScaling::None,
        // feral 0.16 (feral#171) changed `LuParams::default().pivoting` to
        // `Markowitz`, which picks its column order *during* the
        // factorization and so ignores the `SparseLuSymbolic` that
        // `factor_basis` computes and hands to `SparseLu::factor`. Pinning
        // `GilbertPeierls` keeps that symbolic load-bearing, which is what
        // every basis here was measured under. Markowitz is a large reported
        // fill win on exactly this workload (LP bases, geomean 2.77x -> 1.06x)
        // bought with growth (max|U|/max|B| up to 81.8), so switching is a
        // measured change with its own crossover evidence, not a bump.
        pivoting: LuPivoting::GilbertPeierls,
        ..LuParams::default()
    }
}
