//! gh#936 efficacy POC: do soft-mode hops find more distinct minima per NLP
//! solve than isotropic ones?
//!
//! `soft_mode_scaling.rs` answers the *extraction* question — soft modes cost
//! ~2k back-solves at any `n`. It cannot answer this one, because its fixture
//! is a convex quadratic with exactly one minimum. This program is the other
//! half: a genuinely multimodal, genuinely sparse, scalable fixture on which
//! the two move constructions can be scored against gh#936's own metric,
//!
//! > Metric: **distinct minima found per NLP solve**, not wall-clock and not
//! > best-objective. The hypothesis is specifically that isotropic hops waste
//! > solves on duplicates, so duplicate rate is the number that should move.
//!
//! # Two fixtures that did not work, and why they are worth recording
//!
//! **A Frenkel-Kontorova chain** (every coordinate in its own periodic well)
//! is degenerate for this metric: with exponentially many minima *every* hop
//! lands somewhere new. Measured duplicate rate `0.0%` for both strategies at
//! `n` = 200 / 2 000 / 20 000, ratio exactly `1.00x`. "Distinct minima per
//! solve" saturates at 1.000 and discriminates nothing. That is gh#936's own
//! trap one level up — a *metric* uniform in the dimension the change acts on
//! reports nothing however large the model.
//!
//! **Corrugating a collective coordinate** (windows of the chain) failed for a
//! deeper reason: the corrugation that creates a well also supplies curvature
//! at the bottom of it, so the barriered direction is *stiff*, and the soft
//! modes are the uncorrugated long-wavelength ones that lead nowhere. Measured:
//! soft-mode hops moved the collective coordinates by exactly `0.0000` at every
//! `n`, finding 1 minimum against isotropic's 27. That is gh#936's own caveat —
//! "soft modes … describe the basin we are leaving, not the barrier" — as a
//! number, and it is the **branch where the method is actively harmful**.
//!
//! # The fixture
//!
//! Both branches have to be reachable or neither result means anything, so
//! this fixture carries one switch:
//!
//! ```text
//! f(x) = ½ Σ_i c_i x_i²  +  ½ρ Σ_j (x_{j+1} − x_j)²  +  A Σ_{i ∈ C} (1 − cos(2πx_i/P))
//! ```
//!
//! `c_i` is `c_soft` on `M` designated coordinates and `c_stiff` on the rest,
//! `ρ` is a weak nearest-neighbour coupling that keeps the problem non-separable
//! and tridiagonal, and `C` — **the switch** — is the set of corrugated
//! coordinates:
//!
//! * `C` = the soft coordinates ⇒ **barriers lie along soft modes**, which is
//!   gh#936's premise holding.
//! * `C` = an equal number of stiff coordinates ⇒ **barriers lie along stiff
//!   modes**, the premise failing.
//!
//! Everything else is held fixed between the two, so the switch isolates the
//! premise rather than the fixture. Minima are indexed by which well each
//! corrugated coordinate sits in, so they are countable and duplicates are
//! possible — the failure of the first fixture, fixed.
//!
//! An isotropic hop of norm `S` puts `≈ S/√n` on any one coordinate, so it
//! must grow like `√n` to keep crossing barriers; a hop confined to the `M`
//! soft modes puts `≈ S/√M` there regardless of `n`. `S` is held **fixed
//! across the sweep** so this is what the `n` column measures.
//!
//! # Fairness
//!
//! Both strategies hop with the **same step norm** `‖Δx‖` from the **same
//! incumbent** under the **same seed stream**, so the only difference is the
//! *direction*. Equalizing `‖Δx‖` rather than objective rise is the
//! conservative choice: it is the comparison the status quo
//! (`_minima.py:551`, isotropic Gaussian times `step`) actually makes.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p pounce-sensitivity --example minima_hop_efficacy
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint,
};
use pounce_sensitivity::Solver;

const TWO_PI: Number = std::f64::consts::TAU;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct RuggedChain {
    n: usize,
    c_soft: Number,
    c_stiff: Number,
    rho: Number,
    amp: Number,
    period: Number,
    /// Indices carrying the small curvature `c_soft`.
    soft_idx: Vec<usize>,
    /// Indices carrying the corrugation — either `soft_idx` or an equal
    /// number of stiff ones. This is the switch.
    corr_idx: Vec<usize>,
    start: Vec<Number>,
    sol: Rc<RefCell<Vec<Number>>>,
    obj: Rc<Cell<Number>>,
}

impl RuggedChain {
    fn c_of(&self, i: usize) -> Number {
        if self.soft_idx.contains(&i) {
            self.c_soft
        } else {
            self.c_stiff
        }
    }
    fn is_corr(&self, i: usize) -> bool {
        self.corr_idx.contains(&i)
    }
}

impl TNLP for RuggedChain {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as Index,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: (2 * self.n - 1) as Index,
            index_style: IndexStyle::C,
        })
    }
    fn get_scaling_parameters(&mut self, _r: ScalingRequest<'_>) -> bool {
        false
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for i in 0..self.n {
            b.x_l[i] = -1.0e19;
            b.x_u[i] = 1.0e19;
        }
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.start);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        let mut f = 0.0;
        for i in 0..self.n {
            f += 0.5 * self.c_of(i) * x[i] * x[i];
            if self.is_corr(i) {
                f += self.amp * (1.0 - (TWO_PI * x[i] / self.period).cos());
            }
        }
        for j in 0..self.n - 1 {
            let d = x[j + 1] - x[j];
            f += 0.5 * self.rho * d * d;
        }
        Some(f)
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        for i in 0..self.n {
            let mut gi = self.c_of(i) * x[i];
            if self.is_corr(i) {
                gi += self.amp * TWO_PI / self.period * (TWO_PI * x[i] / self.period).sin();
            }
            if i > 0 {
                gi += self.rho * (x[i] - x[i - 1]);
            }
            if i + 1 < self.n {
                gi -= self.rho * (x[i + 1] - x[i]);
            }
            g[i] = gi;
        }
        true
    }
    fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _mode: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        obj: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for i in 0..self.n {
                    irow[k] = i as Index;
                    jcol[k] = i as Index;
                    k += 1;
                }
                for i in 0..self.n - 1 {
                    irow[k] = (i + 1) as Index;
                    jcol[k] = i as Index;
                    k += 1;
                }
            }
            SparsityRequest::Values { values } => {
                let x = match x {
                    Some(x) => x,
                    None => return false,
                };
                let w = TWO_PI / self.period;
                let mut k = 0;
                for i in 0..self.n {
                    let mut d = self.c_of(i);
                    if self.is_corr(i) {
                        d += self.amp * w * w * (w * x[i]).cos();
                    }
                    if i > 0 {
                        d += self.rho;
                    }
                    if i + 1 < self.n {
                        d += self.rho;
                    }
                    values[k] = obj * d;
                    k += 1;
                }
                for _ in 0..self.n - 1 {
                    values[k] = obj * (-self.rho);
                    k += 1;
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.sol.borrow_mut().clear();
        self.sol.borrow_mut().extend_from_slice(s.x);
        self.obj.set(s.obj_value);
    }
}

// ---------------------------------------------------------------------------
// Operator + randomized range finder (the recommended extraction)
// ---------------------------------------------------------------------------

fn dot(a: &[Number], b: &[Number]) -> Number {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[Number]) -> Number {
    dot(a, a).sqrt()
}

struct KktXOperator<'a> {
    solver: &'a Solver,
    n_x: usize,
    kkt_dim: usize,
    matvecs: Cell<usize>,
}

impl<'a> KktXOperator<'a> {
    fn new(solver: &'a Solver) -> Option<Self> {
        let kkt_dim = solver.kkt_dim()?;
        let n_x = solver.block_dims()?[0];
        Some(Self {
            solver,
            n_x,
            kkt_dim,
            matvecs: Cell::new(0),
        })
    }
    fn apply(&self, v: &[Number], out: &mut [Number]) -> bool {
        let mut rhs = vec![0.0; self.kkt_dim];
        let mut lhs = vec![0.0; self.kkt_dim];
        rhs[..self.n_x].copy_from_slice(&v[..self.n_x]);
        if self.solver.kkt_solve(&rhs, &mut lhs).is_err() {
            return false;
        }
        out[..self.n_x].copy_from_slice(&lhs[..self.n_x]);
        self.matvecs.set(self.matvecs.get() + 1);
        true
    }
}

fn orthonormalize(block: &mut Vec<Vec<Number>>) {
    let mut out: Vec<Vec<Number>> = Vec::with_capacity(block.len());
    for col in block.iter() {
        let mut v = col.clone();
        for _ in 0..2 {
            for b in out.iter() {
                let c = dot(&v, b);
                for (vi, bi) in v.iter_mut().zip(b) {
                    *vi -= c * bi;
                }
            }
        }
        let nr = norm(&v);
        if nr > 1e-13 {
            v.iter_mut().for_each(|a| *a /= nr);
            out.push(v);
        }
    }
    *block = out;
}

/// `Q = orth(A^q Ω)` then a `k×k` Rayleigh-Ritz, returning `(Ritz values,
/// Ritz vectors)`. `2k` back-solves, no iterative eigensolver.
fn soft_subspace(
    op: &KktXOperator,
    k: usize,
    rng: &mut u64,
) -> Option<(Vec<Number>, Vec<Vec<Number>>)> {
    let n = op.n_x;
    let mut nextf = || {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        ((*rng >> 11) as Number) / ((1u64 << 53) as Number) - 0.5
    };
    let mut block: Vec<Vec<Number>> = (0..k).map(|_| (0..n).map(|_| nextf()).collect()).collect();
    for _ in 0..2 {
        for col in block.iter_mut() {
            let mut y = vec![0.0; n];
            if !op.apply(col, &mut y) {
                return None;
            }
            *col = y;
        }
        orthonormalize(&mut block);
    }
    let kk = block.len();
    if kk == 0 {
        return None;
    }
    let aq: Vec<Vec<Number>> = block
        .iter()
        .map(|col| {
            let mut y = vec![0.0; n];
            op.apply(col, &mut y);
            y
        })
        .collect();
    let mut h = vec![0.0; kk * kk];
    for i in 0..kk {
        for j in 0..kk {
            h[j * kk + i] = dot(&block[i], &aq[j]);
        }
    }
    let mut vals = vec![0.0; kk];
    let mut vecs = vec![0.0; kk * kk];
    if !pounce_linalg::symmetric_eigen(&h, kk, &mut vals, &mut vecs) {
        return None;
    }
    let ritz: Vec<Vec<Number>> = (0..kk)
        .map(|c| {
            let mut v = vec![0.0; n];
            for (j, qj) in block.iter().enumerate() {
                let w = vecs[c * kk + j];
                for (vi, qv) in v.iter_mut().zip(qj) {
                    *vi += w * qv;
                }
            }
            v
        })
        .collect();
    Some((vals, ritz))
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

struct SolveOut {
    solver: Solver,
    x: Vec<Number>,
    /// Which well each corrugated coordinate landed in — this is what
    /// identifies the minimum.
    wells: Vec<i64>,
}

#[derive(Clone)]
struct Model {
    n: usize,
    c_soft: Number,
    c_stiff: Number,
    rho: Number,
    amp: Number,
    period: Number,
    soft_idx: Vec<usize>,
    corr_idx: Vec<usize>,
}

impl Model {
    /// `m` soft coordinates, evenly spread. `corrugate_soft` selects the
    /// branch: corrugate those same coordinates (gh#936's premise holds), or
    /// an equal number of stiff ones offset from them (premise fails).
    fn new(n: usize, m: usize, corrugate_soft: bool, amp: Number) -> Self {
        let stride = n / m;
        let soft_idx: Vec<usize> = (0..m).map(|k| k * stride).collect();
        let corr_idx: Vec<usize> = if corrugate_soft {
            soft_idx.clone()
        } else {
            (0..m).map(|k| k * stride + stride / 2).collect()
        };
        Self {
            n,
            c_soft: 1e-3,
            c_stiff: 1.0,
            rho: 1e-3,
            amp,
            period: 4.0,
            soft_idx,
            corr_idx,
        }
    }

    fn wells(&self, x: &[Number]) -> Vec<i64> {
        self.corr_idx
            .iter()
            .map(|&i| (x[i] / self.period).round() as i64)
            .collect()
    }

    fn solve_from(&self, x0: &[Number]) -> Option<SolveOut> {
        let mut app = IpoptApplication::new();
        let o = app.options_mut();
        o.set_integer_value("print_level", 0, true, false).ok()?;
        o.set_string_value("sb", "yes", true, false).ok()?;
        o.set_numeric_value("tol", 1e-8, true, false).ok()?;
        o.set_integer_value("max_iter", 300, true, false).ok()?;
        app.initialize().ok()?;

        let sol = Rc::new(RefCell::new(Vec::new()));
        let obj = Rc::new(Cell::new(0.0));
        let tnlp = RuggedChain {
            n: self.n,
            c_soft: self.c_soft,
            c_stiff: self.c_stiff,
            rho: self.rho,
            amp: self.amp,
            period: self.period,
            soft_idx: self.soft_idx.clone(),
            corr_idx: self.corr_idx.clone(),
            start: x0.to_vec(),
            sol: sol.clone(),
            obj: obj.clone(),
        };
        let dynamic: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
        let mut sv = Solver::new(app, dynamic);
        let st = sv.solve();
        if !matches!(
            st,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ) {
            return None;
        }
        let x = sol.borrow().clone();
        if x.len() != self.n {
            return None;
        }
        let wells = self.wells(&x);
        Some(SolveOut {
            solver: sv,
            x,
            wells,
        })
    }
}

#[derive(Default)]
struct Tally {
    solves: usize,
    distinct: usize,
    backsolves: usize,
    failed: usize,
    /// Mean number of corrugated coordinates whose well changed per hop.
    mean_well_changes: Number,
}

fn run_strategy(
    model: &Model,
    x_init: &[Number],
    budget: usize,
    step: Number,
    k: usize,
    use_soft_modes: bool,
    seed: u64,
) -> Tally {
    let n = model.n;
    let mut rng = seed;
    let mut nextf = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 11) as Number) / ((1u64 << 53) as Number) - 0.5
    };
    let mut t = Tally::default();
    let mut archive: Vec<Vec<i64>> = Vec::new();

    let first = match model.solve_from(x_init) {
        Some(s) => s,
        None => return t,
    };
    t.solves += 1;
    archive.push(first.wells.clone());
    t.distinct += 1;
    let mut cur = first;

    let mut dir_rng = seed ^ 0x5DEECE66D;
    let (mut wc_acc, mut wc_cnt) = (0.0, 0usize);
    for _ in 0..budget {
        let mut dir = vec![0.0; n];
        if use_soft_modes {
            let op = match KktXOperator::new(&cur.solver) {
                Some(o) => o,
                None => return t,
            };
            match soft_subspace(&op, k, &mut dir_rng) {
                Some((vals, vecs)) => {
                    for (c, col) in vecs.iter().enumerate() {
                        let w = vals[c].max(0.0).sqrt() * nextf();
                        for (di, cv) in dir.iter_mut().zip(col) {
                            *di += w * cv;
                        }
                    }
                    t.backsolves += op.matvecs.get();
                }
                None => dir.iter_mut().for_each(|d| *d = nextf()),
            }
        } else {
            dir.iter_mut().for_each(|d| *d = nextf());
        }
        let dn = norm(&dir);
        if dn <= 0.0 {
            continue;
        }
        // Same step NORM for both strategies, held fixed across n.
        let scale = step / dn;
        let trial: Vec<Number> = cur.x.iter().zip(&dir).map(|(a, d)| a + scale * d).collect();

        let before = cur.wells.clone();
        match model.solve_from(&trial) {
            Some(next) => {
                t.solves += 1;
                let changed = next
                    .wells
                    .iter()
                    .zip(&before)
                    .filter(|(a, b)| a != b)
                    .count();
                wc_acc += changed as Number;
                wc_cnt += 1;
                if !archive.contains(&next.wells) {
                    t.distinct += 1;
                    archive.push(next.wells.clone());
                }
                cur = next;
            }
            None => {
                t.solves += 1;
                t.failed += 1;
            }
        }
    }
    t.mean_well_changes = wc_acc / wc_cnt.max(1) as Number;
    t
}

/// A corrugation only creates a second well if it can overcome the quadratic:
/// `A·2π/P` must exceed `c·(P−1)` or the coordinate has a single minimum and
/// the whole comparison is vacuous. The first attempt at branch 2 missed this
/// and measured `1` minimum for both strategies at every `n` — not "the
/// premise fails" but "there is nothing to find". So the amplitude is sized
/// per branch, which also says something structural: a barrier on a stiff
/// direction has to be proportionally higher to exist at all.
fn wells_exist(c: Number, amp: Number, period: Number) -> bool {
    amp * TWO_PI / period > c * (period - 1.0)
}

fn sweep(
    corrugate_soft: bool,
    label: &str,
    budget: usize,
    step: Number,
    k: usize,
    m: usize,
    amp: Number,
) {
    println!("{label}");
    println!(
        "  {:>7}  {:<11} {:>7} {:>9} {:>9} {:>15} {:>12} {:>9}",
        "n",
        "strategy",
        "solves",
        "distinct",
        "dup rate",
        "distinct/solve",
        "wells/hop",
        "b-solves"
    );
    for &n in &[16usize, 100, 1_000, 10_000] {
        let model = Model::new(n, m, corrugate_soft, amp);
        let c_corr = if corrugate_soft {
            model.c_soft
        } else {
            model.c_stiff
        };
        assert!(
            wells_exist(c_corr, model.amp, model.period),
            "fixture is vacuous: corrugation amp {} cannot create a second well against curvature {c_corr}",
            model.amp
        );
        let x_init: Vec<Number> = (0..n).map(|i| 0.05 * (i as Number * 0.7).sin()).collect();
        let mut rows: Vec<(&str, Tally)> = Vec::new();
        for (name, soft) in [("isotropic", false), ("soft-mode", true)] {
            rows.push((
                name,
                run_strategy(&model, &x_init, budget, step, k, soft, 0xC0FFEE_1234_u64),
            ));
        }
        for (name, t) in &rows {
            println!(
                "  {:>7}  {:<11} {:>7} {:>9} {:>8.1}% {:>15.3} {:>12.2} {:>9}{}",
                n,
                name,
                t.solves,
                t.distinct,
                100.0 * (1.0 - t.distinct as Number / t.solves.max(1) as Number),
                t.distinct as Number / t.solves.max(1) as Number,
                t.mean_well_changes,
                t.backsolves,
                if t.failed > 0 {
                    format!("  ({} failed)", t.failed)
                } else {
                    String::new()
                }
            );
        }
        let r = |t: &Tally| t.distinct as Number / t.solves.max(1) as Number;
        println!(
            "  {:>7}  {:<11} soft-mode / isotropic = {:.2}x\n",
            "",
            "-> ratio",
            r(&rows[1].1) / r(&rows[0].1).max(1e-9)
        );
    }
}

fn main() {
    println!("gh#936 efficacy POC — distinct minima per NLP solve\n");
    let budget = 40usize;
    let step = 10.0;
    let k = 8usize;
    let m = 4usize;
    println!("  M = {m} soft coords (c 1e-3) among c_stiff 1.0, coupling 1e-3,");
    println!("  corrugation period 4, |dx| = {step} fixed across n, k = {k}\n");

    sweep(
        true,
        "=== BRANCH 1: barriers along SOFT modes (gh#936's premise HOLDS), amp 0.02 ===",
        budget,
        step,
        k,
        m,
        0.02,
    );
    sweep(
        false,
        "=== BRANCH 2: barriers along STIFF modes (premise FAILS), amp 2.5 ===",
        budget,
        step,
        k,
        m,
        2.5,
    );
}
