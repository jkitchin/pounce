//! Bounded-variable two-phase revised primal simplex.
//!
//! Solves `min cᵀx  s.t.  A x = b,  l ≤ x ≤ u` (bounds may be ±∞). Inequalities
//! are expected to have already been converted to equalities with explicit
//! slack columns by the caller; this engine sees only the equality system.
//!
//! The method is the textbook *bounded-variable* revised simplex: every
//! nonbasic variable sits at one of its finite bounds (or at `0` if free), the
//! `m` basic variables take whatever values `B x_B = b − N x_N` forces, and a
//! pivot either flips a nonbasic variable to its opposite bound (no basis
//! change) or swaps an entering variable for a leaving one. Phase I drives a set
//! of artificial variables (an instant identity basis) to zero to find a feasible
//! vertex; Phase II then optimizes the real cost from that vertex.
//!
//! All basis algebra (`B⁻¹ A_q`, `c_Bᵀ B⁻¹`) goes through [`crate::basis::Basis`],
//! so Phase 6.2 can swap the dense engine for a sparse LU without touching the
//! pivoting logic here. Pricing is Dantzig (most-negative reduced cost) with a
//! Bland's-rule fallback that engages after a run of degenerate pivots, which
//! guarantees termination.

use crate::basis::{BasisEngine, FaerBasis, REFACTOR_INTERVAL};
use crate::{LpProblem, LpSolution, LpStatus, BOUND_INF};

const INF: f64 = f64::INFINITY;
/// Geometric-mean equilibration sweeps applied at construction. Each sweep is
/// `O(nnz)`; a handful collapses the row/column dynamic range enough to keep the
/// dense basis inverse well-conditioned. See [`equilibrate`].
const SCALE_SWEEPS: usize = 6;
/// In geometric-mean equilibration, an entry smaller than `row/col max ×
/// EQUILIBRATE_DROP` is treated as a structural zero. A relaxation can carry a
/// coefficient that has collapsed to numerical noise (e.g. a McCormick secant
/// slope that goes to ~1e-44 at a degenerate box edge); without this guard such
/// an entry drags the row/column geometric mean to zero and inflates the scale
/// by 1e10–1e20, which then distorts the reduced-cost tolerances enough to make
/// the simplex declare a *wrong* vertex optimal (observed on the quartic OBBT
/// child box `[-2, ~0]`).
const EQUILIBRATE_DROP: f64 = 1e-12;
/// A basis column entry smaller than this is treated as a structural zero in
/// the ratio test (can't be a pivot).
const PIV_TOL: f64 = 1e-9;
/// Reduced-cost slack: a nonbasic variable prices favourably only past this.
const DUAL_TOL: f64 = 1e-7;
/// Primal feasibility / degeneracy slack.
const FEAS_TOL: f64 = 1e-7;
/// Two ratios within this are a tie in the ratio test (pick by stability/Bland).
const RATIO_TIE: f64 = 1e-9;

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// Nonbasic, sitting at its (finite) lower bound.
    Lower,
    /// Nonbasic, sitting at its (finite) upper bound.
    Upper,
    /// In the basis; value determined by `B x_B = b − N x_N`.
    Basic,
    /// Nonbasic and free (both bounds infinite), sitting at `0`.
    Free,
}

/// What the ratio test decided for one iteration.
enum Step {
    /// The entering variable flips to its opposite bound; basis unchanged.
    Flip { theta: f64 },
    /// The entering variable replaces the basic variable in `row`, which leaves
    /// to `leave_state` (its hit bound).
    Pivot {
        theta: f64,
        row: usize,
        leave_state: State,
    },
    /// The objective decreases without bound along the entering direction.
    Unbounded,
}

struct Spx {
    /// Structural variables (the caller's `x`).
    n: usize,
    /// Equality rows.
    m: usize,
    /// Total variables = structural + `m` artificials.
    nn: usize,
    /// Sparse columns of the structural variables (`(row, value)`).
    cols: Vec<Vec<(usize, f64)>>,
    b: Vec<f64>,
    /// Bounds over all `nn` variables (artificials appended).
    lb: Vec<f64>,
    ub: Vec<f64>,
    /// Current-phase cost over all `nn` variables.
    cost: Vec<f64>,
    /// Engine's structural cost in *scaled* space, `c_j · col_scale_j` (the
    /// reported objective `Σ orig_c·x` then equals the original `Σ c·x_orig`).
    orig_c: Vec<f64>,
    /// Per-structural-variable column scale `s_j > 0`. The engine works in the
    /// scaled variable `y_j = x_j / s_j`; the caller's `x_j` is `s_j · y_j` and
    /// the caller's objective coefficient maps to `c_j · s_j`. Equilibration
    /// (geometric mean of row/column magnitudes) keeps the basis well-
    /// conditioned on badly-scaled LPs — without it the dense `B⁻¹` loses enough
    /// accuracy to declare a wrong vertex optimal. Length `n`.
    col_scale: Vec<f64>,
    /// Value of every variable (basic and nonbasic).
    x: Vec<f64>,
    state: Vec<State>,
    /// Variable basic in each row.
    basis_var: Vec<usize>,
    /// Fixed column orientation of each artificial (`±1` on its row).
    art_sign: Vec<f64>,
    basis: FaerBasis,
    /// `true` once Bland's rule is engaged to break out of cycling.
    bland: bool,
    /// `true` once a Phase I has left the basis primal-feasible. A warm
    /// objective-only re-solve (the OBBT inner loop) can then skip Phase I.
    feasible: bool,
    /// `true` once the last solve ended `Optimal`, so the current basis is
    /// dual-feasible for `orig_c`. A warm *bound-change* re-solve (Phase 6.3)
    /// can then dual-simplex from it; otherwise it must cold-start.
    last_optimal: bool,
    // Reused scratch.
    y: Vec<f64>,
    cb: Vec<f64>,
    /// Pivot row `ρ = e_rᵀ B⁻¹` for the dual ratio test (length `m`).
    rho: Vec<f64>,
    alpha: Vec<f64>,
    buf: Vec<(usize, f64)>,
}

/// Geometric-mean row/column equilibration of `prob.a`. Returns
/// `(row_scale, col_scale)` (lengths `m`, `n`, all `> 0`) such that the scaled
/// entry `r_i · A_ij · c_j` has a far smaller dynamic range than `A_ij`. Empty
/// rows/columns keep scale `1`. A few sweeps suffice; this is `O(nnz)` per
/// sweep. Equilibration is essential for the dense basis inverse: on a matrix
/// spanning ~1e8 in magnitude (common in McCormick relaxation LPs) an
/// un-equilibrated basis is ill-conditioned enough that `B⁻¹` yields a wrong
/// basic solution *and* matching wrong reduced costs, so the simplex declares a
/// non-optimal vertex optimal (observed on GLOBALLib `ex4_1_2`).
fn equilibrate(prob: &LpProblem, sweeps: usize) -> (Vec<f64>, Vec<f64>) {
    let (m, n) = (prob.m, prob.n);
    let mut r = vec![1.0_f64; m];
    let mut c = vec![1.0_f64; n];
    for _ in 0..sweeps {
        // Row pass. Two sub-passes: first the row maxima, then the row minima
        // over only the *significant* entries (≥ rmax · EQUILIBRATE_DROP), so a
        // collapsed near-zero coefficient cannot drag the geometric mean to zero
        // and blow up the scale.
        let mut rmax = vec![0.0_f64; m];
        for t in &prob.a {
            let a = (r[t.row] * t.val * c[t.col]).abs();
            rmax[t.row] = rmax[t.row].max(a);
        }
        let mut rmin = vec![INF; m];
        for t in &prob.a {
            let a = (r[t.row] * t.val * c[t.col]).abs();
            if a > rmax[t.row] * EQUILIBRATE_DROP {
                rmin[t.row] = rmin[t.row].min(a);
            }
        }
        for i in 0..m {
            if rmax[i] > 0.0 {
                r[i] /= (rmin[i] * rmax[i]).sqrt();
            }
        }
        // Column pass, same significant-entry rule.
        let mut cmax = vec![0.0_f64; n];
        for t in &prob.a {
            let a = (r[t.row] * t.val * c[t.col]).abs();
            cmax[t.col] = cmax[t.col].max(a);
        }
        let mut cmin = vec![INF; n];
        for t in &prob.a {
            let a = (r[t.row] * t.val * c[t.col]).abs();
            if a > cmax[t.col] * EQUILIBRATE_DROP {
                cmin[t.col] = cmin[t.col].min(a);
            }
        }
        for j in 0..n {
            if cmax[j] > 0.0 {
                c[j] /= (cmin[j] * cmax[j]).sqrt();
            }
        }
    }
    (r, c)
}

impl Spx {
    fn new(prob: &LpProblem) -> Spx {
        let n = prob.n;
        let m = prob.m;
        let nn = n + m;

        // Equilibrate the constraint matrix. The engine then works entirely in
        // the scaled variable `y_j = x_j / col_scale_j`: scaled column entry
        // `row_scale_i · A_ij · col_scale_j`, scaled rhs `row_scale_i · b_i`,
        // scaled bound `bound_j / col_scale_j`, scaled cost `c_j · col_scale_j`.
        // `result()` maps back: `x_j = col_scale_j · y_j`, and the scaled
        // objective value already equals the original `Σ c_j x_j`.
        let (row_scale, col_scale) = equilibrate(prob, SCALE_SWEEPS);

        // Structural columns from the triplet form, scaled.
        let mut cols = vec![Vec::new(); n];
        for t in &prob.a {
            if t.val != 0.0 {
                cols[t.col].push((t.row, row_scale[t.row] * t.val * col_scale[t.col]));
            }
        }

        // Normalize bounds: clamp the ±BOUND_INF sentinels to true infinities so
        // the ratio test's `is_infinite` checks fire; scale finite bounds by
        // `1/col_scale_j` into the engine's `y` space.
        let norm = |v: f64, hi: bool| -> f64 {
            if hi {
                if v >= BOUND_INF {
                    INF
                } else {
                    v
                }
            } else if v <= -BOUND_INF {
                -INF
            } else {
                v
            }
        };
        let mut lb = vec![0.0; nn];
        let mut ub = vec![0.0; nn];
        for j in 0..n {
            let l = norm(prob.lb[j], false);
            let u = norm(prob.ub[j], true);
            lb[j] = if l.is_finite() { l / col_scale[j] } else { l };
            ub[j] = if u.is_finite() { u / col_scale[j] } else { u };
        }

        // Scaled rhs `b'_i = row_scale_i · b_i`.
        let b_scaled: Vec<f64> = (0..m).map(|i| row_scale[i] * prob.b[i]).collect();

        let mut spx = Spx {
            n,
            m,
            nn,
            cols,
            b: b_scaled,
            lb,
            ub,
            cost: vec![0.0; nn],
            // Engine cost lives in scaled `y` space: `c_j · col_scale_j`.
            orig_c: (0..n).map(|j| prob.c[j] * col_scale[j]).collect(),
            col_scale,
            x: vec![0.0; nn],
            state: vec![State::Lower; nn],
            basis_var: vec![0usize; m],
            art_sign: vec![1.0; m],
            basis: FaerBasis::identity(m),
            bland: false,
            feasible: false,
            last_optimal: false,
            y: vec![0.0; m],
            cb: vec![0.0; m],
            rho: vec![0.0; m],
            alpha: vec![0.0; m],
            buf: Vec::new(),
        };
        // Place the structural variables at bounds and seed the artificial basis.
        spx.seed_artificial_basis();
        spx
    }

    /// Place every structural variable nonbasic at a finite bound (lower
    /// preferred, else upper, else free at `0`) and rebuild the all-artificial
    /// starting basis: the artificial in row `i` absorbs the residual
    /// `b − N x_N`, oriented (`art_sign[i] = ±1`) so its value is non-negative.
    /// This is the cold-start configuration — used at construction and again as
    /// the guaranteed-correct fallback if a warm dual re-solve fails to converge.
    fn seed_artificial_basis(&mut self) {
        for j in 0..self.n {
            if self.lb[j] > -INF {
                self.state[j] = State::Lower;
                self.x[j] = self.lb[j];
            } else if self.ub[j] < INF {
                self.state[j] = State::Upper;
                self.x[j] = self.ub[j];
            } else {
                self.state[j] = State::Free;
                self.x[j] = 0.0;
            }
        }

        let mut resid = self.b.clone();
        for j in 0..self.n {
            let xj = self.x[j];
            if xj != 0.0 {
                for &(i, v) in &self.cols[j] {
                    resid[i] -= v * xj;
                }
            }
        }
        for (i, &ri) in resid.iter().enumerate() {
            let s = if ri >= 0.0 { 1.0 } else { -1.0 };
            self.art_sign[i] = s;
            let a = self.n + i;
            self.lb[a] = 0.0;
            self.ub[a] = INF;
            self.x[a] = ri * s; // = |ri| ≥ 0
            self.state[a] = State::Basic;
            self.basis_var[i] = a;
        }
        self.bland = false;
        self.feasible = false;
    }

    /// Sparse column of variable `j` (structural column, or `±e_i` for an
    /// artificial) written into `self.buf`.
    fn col_into_buf(&mut self, j: usize) {
        self.buf.clear();
        if j < self.n {
            self.buf.extend_from_slice(&self.cols[j]);
        } else {
            self.buf.push((j - self.n, self.art_sign[j - self.n]));
        }
    }

    /// `vecᵀ A_j` — dot an `m`-vector against the (structural or artificial)
    /// column of variable `j`.
    fn col_dot(&self, j: usize, vec: &[f64]) -> f64 {
        if j < self.n {
            self.cols[j].iter().map(|&(i, v)| vec[i] * v).sum()
        } else {
            vec[j - self.n] * self.art_sign[j - self.n]
        }
    }

    /// `yᵀ A_j` for the current dual multipliers `y`.
    fn col_dot_y(&self, j: usize) -> f64 {
        self.col_dot(j, &self.y)
    }

    /// Rebuild `B⁻¹` from the current basis columns. Returns `false` if the
    /// basis is singular (numerical failure).
    fn refactor(&mut self) -> bool {
        // Materialize the basic columns, then hand references to the engine.
        let mut owned: Vec<Vec<(usize, f64)>> = Vec::with_capacity(self.m);
        for r in 0..self.m {
            let j = self.basis_var[r];
            if j < self.n {
                owned.push(self.cols[j].clone());
            } else {
                owned.push(vec![(j - self.n, self.art_sign[j - self.n])]);
            }
        }
        let refs: Vec<&[(usize, f64)]> = owned.iter().map(|c| c.as_slice()).collect();
        self.basis.refactor(&refs)
    }

    /// Recompute basic-variable values from scratch: `x_B = B⁻¹ (b − N x_N)`.
    /// Called after a refactor to flush accumulated round-off.
    fn recompute_basics(&mut self) {
        let mut rhs = self.b.clone();
        for j in 0..self.nn {
            if self.state[j] == State::Basic {
                continue;
            }
            let xj = self.x[j];
            if xj == 0.0 {
                continue;
            }
            if j < self.n {
                for &(i, v) in &self.cols[j] {
                    rhs[i] -= v * xj;
                }
            } else {
                let i = j - self.n;
                rhs[i] -= self.art_sign[i] * xj;
            }
        }
        self.buf.clear();
        for (i, &r) in rhs.iter().enumerate() {
            if r != 0.0 {
                self.buf.push((i, r));
            }
        }
        let mut out = vec![0.0; self.m];
        self.basis.ftran(&self.buf, &mut out);
        #[allow(clippy::needless_range_loop)] // `r` indexes basis_var and out together
        for r in 0..self.m {
            self.x[self.basis_var[r]] = out[r];
        }
    }

    /// Price the nonbasic variables and pick an entering one. Returns
    /// `(variable, direction)` where `direction` is `+1` to increase it and
    /// `-1` to decrease it, or `None` if the basis is optimal.
    fn price(&mut self) -> Option<(usize, f64)> {
        // y = c_Bᵀ B⁻¹.
        for r in 0..self.m {
            self.cb[r] = self.cost[self.basis_var[r]];
        }
        self.basis.btran(&self.cb, &mut self.y);

        let mut best: Option<(usize, f64)> = None;
        let mut best_score = DUAL_TOL;
        for j in 0..self.nn {
            let (dir, score) = match self.state[j] {
                State::Basic => continue,
                State::Lower => {
                    let d = self.cost[j] - self.col_dot_y(j);
                    if d < -DUAL_TOL {
                        (1.0, -d)
                    } else {
                        continue;
                    }
                }
                State::Upper => {
                    let d = self.cost[j] - self.col_dot_y(j);
                    if d > DUAL_TOL {
                        (-1.0, d)
                    } else {
                        continue;
                    }
                }
                State::Free => {
                    let d = self.cost[j] - self.col_dot_y(j);
                    if d.abs() > DUAL_TOL {
                        (if d > 0.0 { -1.0 } else { 1.0 }, d.abs())
                    } else {
                        continue;
                    }
                }
            };
            if self.bland {
                // First eligible (smallest index) — anti-cycling.
                return Some((j, dir));
            }
            if score > best_score {
                best_score = score;
                best = Some((j, dir));
            }
        }
        best
    }

    /// Bounded-variable ratio test for entering variable `t` moving in `dir`.
    /// `self.alpha` already holds `B⁻¹ A_t`.
    fn ratio_test(&self, t: usize, dir: f64) -> Step {
        let mut theta_row = INF;
        let mut leave_row = usize::MAX;
        let mut leave_state = State::Lower;
        let mut best_mag = 0.0;
        for r in 0..self.m {
            // x_B[r] changes by −g·θ as the entering variable steps by θ in its
            // improving direction.
            let g = self.alpha[r] * dir;
            let v = self.basis_var[r];
            let (ratio, st) = if g > PIV_TOL {
                // Basic value decreasing → limited by its lower bound.
                if self.lb[v] <= -INF {
                    continue;
                }
                (((self.x[v] - self.lb[v]).max(0.0)) / g, State::Lower)
            } else if g < -PIV_TOL {
                // Basic value increasing → limited by its upper bound.
                if self.ub[v] >= INF {
                    continue;
                }
                (((self.ub[v] - self.x[v]).max(0.0)) / (-g), State::Upper)
            } else {
                continue;
            };
            let mag = g.abs();
            let take = if ratio < theta_row - RATIO_TIE {
                true
            } else if ratio <= theta_row + RATIO_TIE && leave_row != usize::MAX {
                // Tie: Bland picks the smallest basic-variable index (anti-
                // cycling); otherwise prefer the larger pivot (stability).
                if self.bland {
                    v < self.basis_var[leave_row]
                } else {
                    mag > best_mag
                }
            } else {
                false
            };
            if take {
                theta_row = ratio;
                leave_row = r;
                leave_state = st;
                best_mag = mag;
            }
        }

        // The entering variable can also just flip to its opposite bound.
        let range = if self.lb[t] > -INF && self.ub[t] < INF {
            self.ub[t] - self.lb[t]
        } else {
            INF
        };

        if leave_row == usize::MAX {
            if range >= INF {
                return Step::Unbounded;
            }
            return Step::Flip { theta: range };
        }
        if range < theta_row - RATIO_TIE {
            Step::Flip { theta: range }
        } else {
            Step::Pivot {
                theta: theta_row,
                row: leave_row,
                leave_state,
            }
        }
    }

    /// Run primal simplex on the current cost to optimality. Returns the
    /// terminal status for this phase.
    fn run_phase(&mut self) -> LpStatus {
        let limit = 50 * self.nn + 2000;
        let mut degenerate_run = 0usize;
        let mut iters = 0usize;
        loop {
            if iters >= limit {
                return LpStatus::IterationLimit;
            }
            iters += 1;

            let (t, dir) = match self.price() {
                Some(p) => p,
                None => return LpStatus::Optimal,
            };

            self.col_into_buf(t);
            // Disjoint field borrows: basis (&), buf (&), alpha (&mut).
            self.basis.ftran(&self.buf, &mut self.alpha);

            match self.ratio_test(t, dir) {
                Step::Unbounded => return LpStatus::Unbounded,
                Step::Flip { theta } => {
                    let delta = dir * theta;
                    for r in 0..self.m {
                        self.x[self.basis_var[r]] -= self.alpha[r] * delta;
                    }
                    self.x[t] += delta;
                    self.state[t] = if dir > 0.0 {
                        State::Upper
                    } else {
                        State::Lower
                    };
                    if theta <= FEAS_TOL {
                        degenerate_run += 1;
                    } else {
                        degenerate_run = 0;
                        self.bland = false;
                    }
                }
                Step::Pivot {
                    theta,
                    row,
                    leave_state,
                } => {
                    let delta = dir * theta;
                    for r in 0..self.m {
                        self.x[self.basis_var[r]] -= self.alpha[r] * delta;
                    }
                    let leaving = self.basis_var[row];
                    self.x[leaving] = if leave_state == State::Lower {
                        self.lb[leaving]
                    } else {
                        self.ub[leaving]
                    };
                    self.state[leaving] = leave_state;
                    // Entering variable's new basic value.
                    self.x[t] += delta;
                    self.basis_var[row] = t;
                    self.state[t] = State::Basic;

                    self.basis.update(row, &self.alpha);

                    if theta <= FEAS_TOL {
                        degenerate_run += 1;
                    } else {
                        degenerate_run = 0;
                        self.bland = false;
                    }

                    if self.basis.updates_since_refactor() >= REFACTOR_INTERVAL {
                        if !self.refactor() {
                            return LpStatus::NumericalFailure;
                        }
                        self.recompute_basics();
                    }
                }
            }

            // Persistent degeneracy → switch to Bland's rule until a real step.
            if degenerate_run > self.nn + 10 {
                self.bland = true;
            }
        }
    }

    /// Drive Phase I to a feasible basis. Returns `Optimal` (feasible found),
    /// `Infeasible`, or a hard failure. On success, sets `self.feasible` and
    /// pins the artificials at `0` so Phase II can never reintroduce them.
    fn phase_one(&mut self) -> LpStatus {
        // The artificial in row `i` has column `art_sign[i]·e_i`, so the
        // starting basis is `diag(art_sign)`, not the identity. Build `B⁻¹` from
        // the real columns before pricing (a no-op when every sign is `+1`).
        if self.m > 0 {
            if !self.refactor() {
                return LpStatus::NumericalFailure;
            }
            self.recompute_basics();
        }

        // Phase I objective: minimize the sum of artificials.
        for j in 0..self.nn {
            self.cost[j] = if j >= self.n { 1.0 } else { 0.0 };
        }
        self.bland = false;
        let st = self.run_phase();
        if st == LpStatus::NumericalFailure || st == LpStatus::IterationLimit {
            return st;
        }
        // Sum of artificial values = Phase I objective.
        let infeas: f64 = (self.n..self.nn).map(|a| self.x[a]).sum();
        if infeas > FEAS_TOL.max(1e-7 * (1.0 + self.b.iter().map(|v| v.abs()).sum::<f64>())) {
            return LpStatus::Infeasible;
        }

        // Pin the artificials at 0 (a basic artificial then leaves on its first
        // eligible pivot, degenerately). After this the basis is a feasible
        // vertex of the real polytope, reusable for any objective.
        for a in self.n..self.nn {
            self.lb[a] = 0.0;
            self.ub[a] = 0.0;
            self.cost[a] = 0.0;
            if self.state[a] != State::Basic {
                self.state[a] = State::Lower;
                self.x[a] = 0.0;
            }
        }
        self.feasible = true;
        LpStatus::Optimal
    }

    /// Phase II: optimize the real cost from the current feasible basis. Loads
    /// `self.orig_c` into the structural cost slots first.
    fn phase_two(&mut self) -> LpStatus {
        for j in 0..self.n {
            self.cost[j] = self.orig_c[j];
        }
        self.bland = false;
        self.run_phase()
    }

    /// Full cold solve: Phase I then Phase II.
    fn cold_solve(&mut self) -> LpStatus {
        let st = self.phase_one();
        if st != LpStatus::Optimal {
            return st;
        }
        // A clean inverse before Phase II bounds round-off carried from Phase I.
        if self.m > 0 {
            if !self.refactor() {
                return LpStatus::NumericalFailure;
            }
            self.recompute_basics();
        }
        self.phase_two()
    }

    /// Warm objective-only re-solve: replace the structural cost with `c`
    /// (constraints and bounds unchanged) and re-optimize from the current
    /// basis. The basis is still primal-feasible — feasibility is independent of
    /// the objective — so Phase I is skipped and only Phase II runs. Falls back
    /// to a cold solve if no feasible basis is in hand yet. This is the OBBT
    /// inner loop: `min x_i` then `max x_i` over one polytope is exactly an
    /// objective flip.
    fn warm_objective(&mut self, c: &[f64]) -> LpStatus {
        // Caller's `c` is in original-`x` space; scale into the engine's `y`
        // space (`c_j · col_scale_j`) so the reported objective stays `Σ c·x`.
        for (oc, (&cj, &sj)) in self
            .orig_c
            .iter_mut()
            .zip(c.iter().zip(self.col_scale.iter()))
        {
            *oc = cj * sj;
        }
        if !self.feasible {
            return self.cold_solve();
        }
        self.phase_two()
    }

    /// Install new structural bounds (caller's `x` space) into the engine's
    /// scaled `y` space. Returns `false` if any box is crossed (`lb > ub`),
    /// which is immediate infeasibility. Artificial bounds are untouched.
    fn install_bounds(&mut self, lb: &[f64], ub: &[f64]) -> bool {
        for j in 0..self.n {
            let l = if lb[j] <= -BOUND_INF { -INF } else { lb[j] };
            let u = if ub[j] >= BOUND_INF { INF } else { ub[j] };
            // col_scale_j > 0, so dividing preserves order and the ±∞ sentinels.
            let sl = if l.is_finite() {
                l / self.col_scale[j]
            } else {
                l
            };
            let su = if u.is_finite() {
                u / self.col_scale[j]
            } else {
                u
            };
            if sl.is_finite() && su.is_finite() && sl > su + FEAS_TOL {
                return false;
            }
            self.lb[j] = sl;
            self.ub[j] = su;
        }
        true
    }

    /// Bounded-variable **dual** simplex from a dual-feasible (but possibly
    /// primal-infeasible) basis. Each iteration picks the most primal-infeasible
    /// basic variable to leave at its violated bound, then a dual ratio test
    /// picks the entering variable that restores its feasibility while keeping
    /// every reduced cost dual-feasible. Terminates at `Optimal` (primal feasible
    /// reached — and still dual feasible, hence optimal), `Infeasible` (no
    /// eligible entering variable: the dual is unbounded), or a hard failure.
    fn run_dual_phase(&mut self) -> LpStatus {
        let limit = 50 * self.nn + 2000;
        let mut iters = 0usize;
        loop {
            if iters >= limit {
                return LpStatus::IterationLimit;
            }
            iters += 1;

            // Leaving variable: the basic with the largest bound violation.
            // `need_inc` = the basic value is below its lower bound and must rise
            // (it will leave at its lower bound); otherwise it is above its upper
            // bound and must fall (leaves at its upper bound).
            let mut worst = FEAS_TOL;
            let mut row = usize::MAX;
            let mut need_inc = false;
            for r in 0..self.m {
                let v = self.basis_var[r];
                let xv = self.x[v];
                if self.lb[v] > -INF && xv < self.lb[v] - FEAS_TOL {
                    let viol = self.lb[v] - xv;
                    if viol > worst {
                        worst = viol;
                        row = r;
                        need_inc = true;
                    }
                } else if self.ub[v] < INF && xv > self.ub[v] + FEAS_TOL {
                    let viol = xv - self.ub[v];
                    if viol > worst {
                        worst = viol;
                        row = r;
                        need_inc = false;
                    }
                }
            }
            if row == usize::MAX {
                return LpStatus::Optimal;
            }

            // Dual multipliers y = c_Bᵀ B⁻¹ (for reduced costs) and the pivot row
            // ρ = e_rᵀ B⁻¹ (for αᵣⱼ = ρ·A_j). `cb` is reused as the unit RHS for
            // the second BTRAN after the first has produced `y`.
            for r in 0..self.m {
                self.cb[r] = self.cost[self.basis_var[r]];
            }
            self.basis.btran(&self.cb, &mut self.y);
            for k in 0..self.m {
                self.cb[k] = 0.0;
            }
            self.cb[row] = 1.0;
            self.basis.btran(&self.cb, &mut self.rho);

            // Dual ratio test. A nonbasic j is eligible only if entering it moves
            // the leaving variable in the required direction without violating j's
            // own move sense (lower-bound vars may only rise, upper only fall).
            // Among the eligible, pick the smallest |dⱼ| / |αᵣⱼ| (largest pivot
            // breaks ties, for stability).
            let mut best_q = usize::MAX;
            let mut best_ratio = INF;
            let mut best_piv = 0.0_f64;
            for j in 0..self.nn {
                if self.state[j] == State::Basic {
                    continue;
                }
                let arj = self.col_dot(j, &self.rho);
                if arj.abs() <= PIV_TOL {
                    continue;
                }
                let eligible = match self.state[j] {
                    State::Lower => {
                        if need_inc {
                            arj < 0.0
                        } else {
                            arj > 0.0
                        }
                    }
                    State::Upper => {
                        if need_inc {
                            arj > 0.0
                        } else {
                            arj < 0.0
                        }
                    }
                    State::Free => true,
                    State::Basic => continue,
                };
                if !eligible {
                    continue;
                }
                let dj = self.cost[j] - self.col_dot_y(j);
                let ratio = dj.abs() / arj.abs();
                let take = ratio < best_ratio - RATIO_TIE
                    || (ratio <= best_ratio + RATIO_TIE && arj.abs() > best_piv);
                if take {
                    best_ratio = ratio;
                    best_q = j;
                    best_piv = arj.abs();
                }
            }
            if best_q == usize::MAX {
                // No way to restore this row's feasibility: primal infeasible.
                return LpStatus::Infeasible;
            }

            // Pivot: entering column α = B⁻¹ A_q. Step the entering variable so
            // the leaving variable lands exactly on its violated bound.
            self.col_into_buf(best_q);
            self.basis.ftran(&self.buf, &mut self.alpha);
            if self.alpha[row].abs() <= PIV_TOL {
                return LpStatus::NumericalFailure;
            }
            let leaving = self.basis_var[row];
            let bound_v = if need_inc {
                self.lb[leaving]
            } else {
                self.ub[leaving]
            };
            let dxq = (self.x[leaving] - bound_v) / self.alpha[row];
            for r in 0..self.m {
                self.x[self.basis_var[r]] -= self.alpha[r] * dxq;
            }
            self.x[leaving] = bound_v;
            self.state[leaving] = if need_inc { State::Lower } else { State::Upper };
            self.x[best_q] += dxq;
            self.basis_var[row] = best_q;
            self.state[best_q] = State::Basic;

            self.basis.update(row, &self.alpha);

            if self.basis.updates_since_refactor() >= REFACTOR_INTERVAL {
                if !self.refactor() {
                    return LpStatus::NumericalFailure;
                }
                self.recompute_basics();
            }
        }
    }

    /// Warm re-solve after a structural **bound change** (same objective, same
    /// constraints). When a spatial-branch-and-bound child tightens variable
    /// bounds, the parent's optimal basis stays *dual* feasible — only primal
    /// feasibility can be lost — so a dual simplex re-optimizes in a few pivots.
    /// This is the parent→child lever, distinct from the per-objective flip
    /// [`warm_objective`] handles within one node.
    ///
    /// Falls back to a cold solve when there is no dual-feasible basis in hand
    /// (nothing solved yet), and again as a guaranteed-correct backstop if the
    /// dual phase fails to converge (iteration limit / numerical trouble).
    fn warm_bounds(&mut self, lb: &[f64], ub: &[f64]) -> LpStatus {
        if !self.install_bounds(lb, ub) {
            return LpStatus::Infeasible;
        }
        if !self.last_optimal {
            // No dual-feasible basis yet — cold start under the new bounds.
            self.seed_artificial_basis();
            return self.cold_solve();
        }

        // The basis is unchanged, so it stays dual feasible. Nonbasic variables
        // track their (possibly moved) bound; recomputing the basics from the new
        // nonbasic values is what can introduce primal infeasibility.
        for j in 0..self.n {
            match self.state[j] {
                State::Lower => {
                    if self.lb[j] > -INF {
                        self.x[j] = self.lb[j];
                    }
                }
                State::Upper => {
                    if self.ub[j] < INF {
                        self.x[j] = self.ub[j];
                    }
                }
                State::Free => {
                    // A previously-free variable that just gained a finite bound
                    // moves onto it (its reduced cost was 0, so either bound is
                    // dual feasible).
                    if self.lb[j] > -INF {
                        self.state[j] = State::Lower;
                        self.x[j] = self.lb[j];
                    } else if self.ub[j] < INF {
                        self.state[j] = State::Upper;
                        self.x[j] = self.ub[j];
                    }
                }
                State::Basic => {}
            }
        }
        self.recompute_basics();

        match self.run_dual_phase() {
            LpStatus::Optimal => LpStatus::Optimal,
            LpStatus::Infeasible => LpStatus::Infeasible,
            // Dual phase stalled or went numerically bad: rebuild from scratch.
            // A full cold solve under the (already-installed) new bounds is
            // always correct, just slower.
            _ => {
                self.seed_artificial_basis();
                self.cold_solve()
            }
        }
    }

    fn result(&self, status: LpStatus) -> LpSolution {
        // Map the engine's scaled `y` back to the caller's `x_j = col_scale_j·y_j`.
        // The objective is `Σ orig_c_j · y_j` with `orig_c` already in scaled
        // space (`c_j·col_scale_j`), so it equals the original `Σ c_j · x_j`.
        if status == LpStatus::Optimal {
            let x: Vec<f64> = (0..self.n).map(|j| self.col_scale[j] * self.x[j]).collect();
            let obj = (0..self.n).map(|j| self.orig_c[j] * self.x[j]).sum();
            LpSolution { status, x, obj }
        } else {
            LpSolution {
                status,
                x: (0..self.n).map(|j| self.col_scale[j] * self.x[j]).collect(),
                obj: f64::NAN,
            }
        }
    }
}

/// A reusable bounded-variable revised simplex over a fixed polytope. Built once
/// from an [`LpProblem`], then re-solved for new objectives (the OBBT inner
/// loop) with the basis warm-started across each call.
pub struct Simplex {
    spx: Spx,
}

impl Simplex {
    /// Build a solver over `prob`'s constraints and bounds. `prob.c` is the
    /// first objective; call [`Simplex::solve`] to optimize it.
    pub fn new(prob: &LpProblem) -> Simplex {
        debug_assert_eq!(prob.c.len(), prob.n);
        debug_assert_eq!(prob.lb.len(), prob.n);
        debug_assert_eq!(prob.ub.len(), prob.n);
        debug_assert_eq!(prob.b.len(), prob.m);
        Simplex {
            spx: Spx::new(prob),
        }
    }

    /// Cold-solve the current objective (Phase I + Phase II).
    pub fn solve(&mut self) -> LpSolution {
        let st = self.spx.cold_solve();
        self.spx.last_optimal = st == LpStatus::Optimal;
        self.spx.result(st)
    }

    /// Re-solve with a new objective `c` (same constraints and bounds),
    /// warm-starting from the basis left by the previous solve. `c` has length
    /// `n`.
    pub fn solve_objective(&mut self, c: &[f64]) -> LpSolution {
        let st = self.spx.warm_objective(c);
        self.spx.last_optimal = st == LpStatus::Optimal;
        self.spx.result(st)
    }

    /// Re-solve after changing the structural variable bounds to `lb`/`ub` (each
    /// length `n`), keeping the same objective and constraints and warm-starting
    /// from the previous optimal basis via the dual simplex. This is the
    /// parent→child lever in spatial branch-and-bound: a box that only tightens
    /// bounds leaves the parent basis dual-feasible, so re-optimization takes a
    /// handful of dual pivots instead of a cold Phase I/II. Falls back to a cold
    /// solve when no optimal basis is in hand or the dual phase cannot converge.
    pub fn solve_bounds(&mut self, lb: &[f64], ub: &[f64]) -> LpSolution {
        debug_assert_eq!(lb.len(), self.spx.n);
        debug_assert_eq!(ub.len(), self.spx.n);
        let st = self.spx.warm_bounds(lb, ub);
        self.spx.last_optimal = st == LpStatus::Optimal;
        self.spx.result(st)
    }
}

/// Solve `min cᵀx s.t. A x = b, l ≤ x ≤ u` with the bounded-variable revised
/// simplex (cold start).
pub(crate) fn solve(prob: &LpProblem) -> LpSolution {
    Simplex::new(prob).solve()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triplet;

    fn lp(
        n: usize,
        m: usize,
        c: &[f64],
        a: &[(usize, usize, f64)],
        b: &[f64],
        lb: &[f64],
        ub: &[f64],
    ) -> LpProblem {
        LpProblem {
            n,
            m,
            c: c.to_vec(),
            a: a.iter()
                .map(|&(r, col, v)| Triplet::new(r, col, v))
                .collect(),
            b: b.to_vec(),
            lb: lb.to_vec(),
            ub: ub.to_vec(),
        }
    }

    #[test]
    fn single_equality() {
        // min x s.t. x = 5, 0 ≤ x ≤ 10 → x = 5.
        let p = lp(1, 1, &[1.0], &[(0, 0, 1.0)], &[5.0], &[0.0], &[10.0]);
        let s = solve(&p);
        assert_eq!(s.status, LpStatus::Optimal);
        assert!((s.x[0] - 5.0).abs() < 1e-9, "{:?}", s.x);
        assert!((s.obj - 5.0).abs() < 1e-9);
    }

    #[test]
    fn bounds_only_minimize() {
        // min -x s.t. 0 ≤ x ≤ 4 (no equality rows) → x = 4, obj −4.
        let p = lp(1, 0, &[-1.0], &[], &[], &[0.0], &[4.0]);
        let s = solve(&p);
        assert_eq!(s.status, LpStatus::Optimal);
        assert!((s.x[0] - 4.0).abs() < 1e-9, "{:?}", s.x);
        assert!((s.obj + 4.0).abs() < 1e-9);
    }

    #[test]
    fn slacked_inequality() {
        // max x + 2y  s.t.  x + y ≤ 4,  0 ≤ x ≤ 3,  y ≥ 0
        // as min −x − 2y  s.t.  x + y + s = 4 (s ≥ 0).
        // Optimum: x=0, y=4, obj −8.
        let p = lp(
            3,
            1,
            &[-1.0, -2.0, 0.0],
            &[(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0)],
            &[4.0],
            &[0.0, 0.0, 0.0],
            &[3.0, BOUND_INF, BOUND_INF],
        );
        let s = solve(&p);
        assert_eq!(s.status, LpStatus::Optimal);
        assert!((s.obj + 8.0).abs() < 1e-7, "obj {} x {:?}", s.obj, s.x);
        assert!((s.x[1] - 4.0).abs() < 1e-6, "{:?}", s.x);
    }

    #[test]
    fn two_constraints() {
        // min −2x − 3y
        // s.t. x +  y + s1      = 4
        //      x + 3y      + s2 = 6
        //      x,y,s1,s2 ≥ 0
        // Optimum at x=3, y=1: obj = −9.
        let p = lp(
            4,
            2,
            &[-2.0, -3.0, 0.0, 0.0],
            &[
                (0, 0, 1.0),
                (0, 1, 1.0),
                (0, 2, 1.0),
                (1, 0, 1.0),
                (1, 1, 3.0),
                (1, 3, 1.0),
            ],
            &[4.0, 6.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[BOUND_INF, BOUND_INF, BOUND_INF, BOUND_INF],
        );
        let s = solve(&p);
        assert_eq!(s.status, LpStatus::Optimal);
        assert!((s.obj + 9.0).abs() < 1e-6, "obj {} x {:?}", s.obj, s.x);
        assert!(
            (s.x[0] - 3.0).abs() < 1e-6 && (s.x[1] - 1.0).abs() < 1e-6,
            "{:?}",
            s.x
        );
    }

    #[test]
    fn infeasible_box() {
        // x = 5 but 0 ≤ x ≤ 3 → infeasible.
        let p = lp(1, 1, &[1.0], &[(0, 0, 1.0)], &[5.0], &[0.0], &[3.0]);
        let s = solve(&p);
        assert_eq!(s.status, LpStatus::Infeasible);
    }

    #[test]
    fn unbounded_ray() {
        // min −x, x ≥ 0, no upper bound, no rows → unbounded.
        let p = lp(1, 0, &[-1.0], &[], &[], &[0.0], &[BOUND_INF]);
        let s = solve(&p);
        assert_eq!(s.status, LpStatus::Unbounded);
    }

    #[test]
    fn free_variable() {
        // min x + y s.t. x + y = 2, x free, 0 ≤ y ≤ 5.
        // x = 2 − y, obj = 2 regardless; any feasible point optimal.
        let p = lp(
            2,
            1,
            &[1.0, 1.0],
            &[(0, 0, 1.0), (0, 1, 1.0)],
            &[2.0],
            &[-BOUND_INF, 0.0],
            &[BOUND_INF, 5.0],
        );
        let s = solve(&p);
        assert_eq!(s.status, LpStatus::Optimal);
        assert!((s.obj - 2.0).abs() < 1e-7, "obj {}", s.obj);
        assert!((s.x[0] + s.x[1] - 2.0).abs() < 1e-7, "{:?}", s.x);
    }

    #[test]
    fn negative_rhs_orientation() {
        // min x s.t. −x = −5, 0 ≤ x ≤ 10 → x = 5. Exercises art_sign = −1.
        let p = lp(1, 1, &[1.0], &[(0, 0, -1.0)], &[-5.0], &[0.0], &[10.0]);
        let s = solve(&p);
        assert_eq!(s.status, LpStatus::Optimal);
        assert!((s.x[0] - 5.0).abs() < 1e-9, "{:?}", s.x);
    }

    #[test]
    fn objective_flip_min_then_max() {
        // The OBBT pattern: same polytope, flip the objective sign.
        // x + y = 10, 0 ≤ x ≤ 8, 0 ≤ y ≤ 8.
        // min x → x = 2 (y = 8); max x → x = 8 (y = 2).
        let mk = |c0: f64| {
            lp(
                2,
                1,
                &[c0, 0.0],
                &[(0, 0, 1.0), (0, 1, 1.0)],
                &[10.0],
                &[0.0, 0.0],
                &[8.0, 8.0],
            )
        };
        let smin = solve(&mk(1.0));
        assert_eq!(smin.status, LpStatus::Optimal);
        assert!((smin.x[0] - 2.0).abs() < 1e-7, "min {:?}", smin.x);
        let smax = solve(&mk(-1.0));
        assert_eq!(smax.status, LpStatus::Optimal);
        assert!((smax.x[0] - 8.0).abs() < 1e-7, "max {:?}", smax.x);
    }

    #[test]
    fn klee_minty_3d() {
        // The classic Klee–Minty cube — the textbook worst case for Dantzig
        // pricing (visits all 2ⁿ−1 vertices). Correctness check: the optimum is
        // at (0, 0, 10000) with value 10000.
        //   max 100 x1 + 10 x2 + x3
        //   s.t.  x1 ≤ 1
        //         20 x1 +   x2        ≤ 100
        //        200 x1 + 20 x2 + x3  ≤ 10000,  x ≥ 0
        // Slacks s1,s2,s3 ≥ 0 turn each ≤ into an equality. Minimize the negated
        // objective.
        let inf = BOUND_INF;
        let p = lp(
            6, // x1 x2 x3 s1 s2 s3
            3,
            &[-100.0, -10.0, -1.0, 0.0, 0.0, 0.0],
            &[
                (0, 0, 1.0),
                (0, 3, 1.0),
                (1, 0, 20.0),
                (1, 1, 1.0),
                (1, 4, 1.0),
                (2, 0, 200.0),
                (2, 1, 20.0),
                (2, 2, 1.0),
                (2, 5, 1.0),
            ],
            &[1.0, 100.0, 10000.0],
            &[0.0; 6],
            &[inf; 6],
        );
        let s = solve(&p);
        assert_eq!(s.status, LpStatus::Optimal);
        assert!((s.obj + 10000.0).abs() < 1e-4, "obj {}", s.obj);
        assert!((s.x[2] - 10000.0).abs() < 1e-4, "x3 {:?}", s.x);
    }

    #[test]
    fn warm_objective_flip_matches_cold() {
        // Same OBBT flip, but reuse the basis: solve min x, then warm-solve max x.
        // x + y = 10, 0 ≤ x,y ≤ 8.
        let p = lp(
            2,
            1,
            &[1.0, 0.0],
            &[(0, 0, 1.0), (0, 1, 1.0)],
            &[10.0],
            &[0.0, 0.0],
            &[8.0, 8.0],
        );
        let mut s = Simplex::new(&p);
        let smin = s.solve();
        assert_eq!(smin.status, LpStatus::Optimal);
        assert!((smin.x[0] - 2.0).abs() < 1e-7, "min {:?}", smin.x);
        // Warm-start the flipped objective from the min-basis.
        let smax = s.solve_objective(&[-1.0, 0.0]);
        assert_eq!(smax.status, LpStatus::Optimal);
        assert!((smax.x[0] - 8.0).abs() < 1e-7, "warm max {:?}", smax.x);
    }

    #[test]
    fn warm_sweep_matches_cold_each_variable() {
        // OBBT-style 2n sweep over a 3-var polytope: min/max each variable,
        // warm-started, must match independent cold solves to tolerance.
        //   x + y + z = 6,  0 ≤ x ≤ 5,  0 ≤ y ≤ 5,  0 ≤ z ≤ 5.
        let n = 3;
        let cols = [(0usize, 0usize, 1.0), (0, 1, 1.0), (0, 2, 1.0)];
        let mk_c = |i: usize, sign: f64| {
            let mut c = vec![0.0; n];
            c[i] = sign;
            c
        };
        let base = |c: &[f64]| lp(n, 1, c, &cols, &[6.0], &[0.0, 0.0, 0.0], &[5.0, 5.0, 5.0]);

        // One warm solver swept across all 2n objectives.
        let mut warm = Simplex::new(&base(&mk_c(0, 1.0)));
        let _ = warm.solve();
        for i in 0..n {
            for &sign in &[1.0_f64, -1.0] {
                let c = mk_c(i, sign);
                let cold = solve(&base(&c));
                let w = warm.solve_objective(&c);
                assert_eq!(cold.status, LpStatus::Optimal);
                assert_eq!(w.status, LpStatus::Optimal);
                assert!(
                    (cold.obj - w.obj).abs() < 1e-7,
                    "var {i} sign {sign}: cold {} warm {}",
                    cold.obj,
                    w.obj
                );
            }
        }
    }

    #[test]
    fn warm_bounds_dual_pivot_matches_cold() {
        // x + y = 10, 0 ≤ x,y ≤ 8. min x → x = 2 (y = 8, x is the basic var).
        // Tighten x's *lower* bound to 4: the basic x=2 now violates it, forcing
        // a genuine dual pivot. Warm re-solve must reach x = 4 (y = 6), matching a
        // cold solve of the tightened LP.
        let mk = |lb: [f64; 2], ub: [f64; 2]| {
            lp(
                2,
                1,
                &[1.0, 0.0],
                &[(0, 0, 1.0), (0, 1, 1.0)],
                &[10.0],
                &lb,
                &ub,
            )
        };
        let mut warm = Simplex::new(&mk([0.0, 0.0], [8.0, 8.0]));
        let s0 = warm.solve();
        assert_eq!(s0.status, LpStatus::Optimal);
        assert!((s0.x[0] - 2.0).abs() < 1e-7, "prime {:?}", s0.x);

        let w = warm.solve_bounds(&[4.0, 0.0], &[8.0, 8.0]);
        let cold = solve(&mk([4.0, 0.0], [8.0, 8.0]));
        assert_eq!(w.status, LpStatus::Optimal);
        assert_eq!(cold.status, LpStatus::Optimal);
        assert!((w.x[0] - 4.0).abs() < 1e-6, "warm x {:?}", w.x);
        assert!(
            (w.obj - cold.obj).abs() < 1e-7,
            "warm {} cold {}",
            w.obj,
            cold.obj
        );
    }

    #[test]
    fn warm_bounds_detects_infeasible() {
        // x + y = 10, but tightening both upper bounds to 3 makes x + y ≤ 6 < 10:
        // infeasible. The dual warm re-solve must report it (not a wrong optimum),
        // matching a cold solve.
        let mk = |ub: [f64; 2]| {
            lp(
                2,
                1,
                &[1.0, 0.0],
                &[(0, 0, 1.0), (0, 1, 1.0)],
                &[10.0],
                &[0.0, 0.0],
                &ub,
            )
        };
        let mut warm = Simplex::new(&mk([8.0, 8.0]));
        assert_eq!(warm.solve().status, LpStatus::Optimal);
        let w = warm.solve_bounds(&[0.0, 0.0], &[3.0, 3.0]);
        let cold = solve(&mk([3.0, 3.0]));
        assert_eq!(cold.status, LpStatus::Infeasible);
        assert_eq!(w.status, LpStatus::Infeasible, "warm should match cold");
    }

    #[test]
    fn warm_bounds_sweep_matches_cold() {
        // OBBT parent→child lever: a 3-variable polytope re-solved through a chain
        // of cumulative bound tightenings, each warm-started from the previous
        // optimal basis, must match an independent cold solve at every step.
        //   x + y + z = 6,  bounds tightened in sequence; min x + 2y + 3z.
        let c = [1.0, 2.0, 3.0];
        let cols = [(0usize, 0usize, 1.0), (0, 1, 1.0), (0, 2, 1.0)];
        let mk = |lb: [f64; 3], ub: [f64; 3]| lp(3, 1, &c, &cols, &[6.0], &lb, &ub);

        let lb0 = [0.0, 0.0, 0.0];
        let mut warm = Simplex::new(&mk(lb0, [5.0, 5.0, 5.0]));
        assert_eq!(warm.solve().status, LpStatus::Optimal);

        // Each entry tightens an upper bound; some hit the basic variable, and
        // the last drives the box infeasible (Σ ub = 5 < 6) — the warm path must
        // agree with cold on *both* outcomes, then recover on the next loosening.
        let ubs: [[f64; 3]; 5] = [
            [5.0, 0.5, 5.0],
            [4.0, 0.5, 5.0],
            [4.0, 0.5, 1.5],
            [3.0, 0.5, 1.5], // Σ = 5 < 6 ⇒ infeasible
            [4.0, 1.0, 2.0], // loosened again ⇒ feasible (cold-restarts the basis)
        ];
        for (k, ub) in ubs.iter().enumerate() {
            let w = warm.solve_bounds(&lb0, ub);
            let cold = solve(&mk(lb0, *ub));
            assert_eq!(w.status, cold.status, "step {k} status");
            if cold.status == LpStatus::Optimal {
                assert!(
                    (w.obj - cold.obj).abs() < 1e-6,
                    "step {k}: warm {} cold {}",
                    w.obj,
                    cold.obj
                );
            }
        }
    }

    #[test]
    fn warm_bounds_then_objective_flip() {
        // The two warm levers compose: a bound tightening (dual) followed by an
        // objective flip (primal) on the tightened polytope must still match cold.
        let mk = |c: &[f64], ub: [f64; 2]| {
            lp(
                2,
                1,
                c,
                &[(0, 0, 1.0), (0, 1, 1.0)],
                &[10.0],
                &[0.0, 0.0],
                &ub,
            )
        };
        let mut warm = Simplex::new(&mk(&[1.0, 0.0], [8.0, 8.0]));
        assert_eq!(warm.solve().status, LpStatus::Optimal);
        // Tighten, then flip min x → max x on the tightened box.
        let _ = warm.solve_bounds(&[0.0, 0.0], &[6.0, 6.0]);
        let w = warm.solve_objective(&[-1.0, 0.0]);
        let cold = solve(&mk(&[-1.0, 0.0], [6.0, 6.0]));
        assert_eq!(w.status, LpStatus::Optimal);
        assert_eq!(cold.status, LpStatus::Optimal);
        assert!(
            (w.obj - cold.obj).abs() < 1e-7,
            "warm {} cold {}",
            w.obj,
            cold.obj
        );
        // max x on x+y=10, x,y ≤ 6 ⇒ x = 6 (y = 4).
        assert!((w.x[0] - 6.0).abs() < 1e-6, "warm x {:?}", w.x);
    }
}
