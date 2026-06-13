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
use crate::qp::{QpProblem, QpSolution, BOUND_INF};
use feral::{LuParams, LuScaling, LuSingularAction, SparseColMatrix, SparseLu, SparseLuSymbolic};

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
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub z_lb: Vec<f64>,
    pub z_ub: Vec<f64>,
    pub obj: f64,
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
    let mut s = Simplex::new(prob);
    s.warm_start(sol);
    s.factor_basis()?;
    s.recompute_basics()?; // slacks absorb the residual ⇒ feasible start
    s.push_superbasics()?; // drive interior structurals to bounds / into the basis
    s.run_phase1()?; // clean the residual bound infeasibility (e.g. eq artificials)
    s.run_phase2()?; // optimize to the vertex
    Some(s.extract(prob))
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
    if v >= BOUND_INF {
        f64::INFINITY
    } else {
        v
    }
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
            let xj = sol.x.get(j).copied().unwrap_or(0.0).clamp(lo, hi);
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
        for j in supers {
            if self.slot_of[j] != usize::MAX || self.nb[j] != NbStatus::Superbasic {
                continue; // already resolved as a side effect of an earlier pivot
            }
            self.resolve_superbasic(j)?;
        }
        if dbg {
            self.recompute_basics();
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
        let max_iter = 20 * (self.m + self.nv) + 1000;
        self.bland = false;
        self.stall = 0;
        let cost = self.cost.clone();
        for _ in 0..max_iter {
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
    fn ratio_test_phase1(&self, q: usize, t: f64, rc: f64, alpha: &[f64]) -> Step {
        // Bland's anti-cycling guarantee requires the leaving variable to be the
        // lowest-index one among the minimum-ratio ties — which the long step does
        // NOT honor (it stops by slope, not by min ratio). So once the Bland latch
        // is on, fall back to the Bland-correct first-breakpoint short step; it is
        // provably finite and the only thing that escapes the degenerate plateaus.
        if self.bland {
            return self.ratio_test_phase1_bland(q, t, alpha);
        }
        // Breakpoints: (theta, slot, bound-status-on-leaving, |rate|).
        let mut bps: Vec<(f64, usize, NbStatus, f64)> = Vec::new();
        for slot in 0..self.m {
            let rate = alpha[slot] * t; // basic value moves as x − rate·θ
            if rate.abs() <= PIVOT_TOL {
                continue;
            }
            let v = self.basis[slot];
            let x = self.xval[v];
            let arate = rate.abs();
            // Every bound this basic reaches at θ > 0 is a breakpoint (a bound it
            // moves away from gives θ < 0 and is skipped).
            for &(bound, to) in &[
                (self.lo[v], NbStatus::AtLower),
                (self.hi[v], NbStatus::AtUpper),
            ] {
                if !bound.is_finite() {
                    continue;
                }
                let theta = (x - bound) / rate;
                if theta > FEAS_TOL {
                    bps.push((theta, slot, to, arate));
                }
            }
        }
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
    /// reaches (minimum ratio), breaking ties by lowest leaving-variable index.
    /// No basic crosses a bound during the step, so it is monotone, and the
    /// lowest-index tie-break makes the entering/leaving pair Bland-consistent —
    /// together that guarantees finite termination through degeneracy.
    fn ratio_test_phase1_bland(&self, q: usize, t: f64, alpha: &[f64]) -> Step {
        let flip = self.entering_range(q);
        let mut best_theta = flip;
        let mut best_slot: Option<usize> = None;
        let mut best_to = NbStatus::AtLower;
        for slot in 0..self.m {
            let rate = alpha[slot] * t;
            if rate.abs() <= PIVOT_TOL {
                continue;
            }
            let v = self.basis[slot];
            let x = self.xval[v];
            for &(bound, to) in &[
                (self.lo[v], NbStatus::AtLower),
                (self.hi[v], NbStatus::AtUpper),
            ] {
                if !bound.is_finite() {
                    continue;
                }
                let theta = (x - bound) / rate;
                if theta < -FEAS_TOL {
                    continue; // heading away from this bound
                }
                let theta = theta.max(0.0);
                let better = theta < best_theta - FEAS_TOL
                    || ((theta - best_theta).abs() <= FEAS_TOL
                        && best_slot.is_none_or(|bs| v < self.basis[bs]));
                if better {
                    best_theta = theta;
                    best_slot = Some(slot);
                    best_to = to;
                }
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
        ..LuParams::default()
    }
}
