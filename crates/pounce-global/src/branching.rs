//! Branching-variable selection: widest, most-violation, and **reliability**.
//!
//! Reliability branching (Achterberg, Koch & Martin) is the SOTA rule. It
//! blends two estimators of "how much will the lower bound improve if I branch
//! on variable *i*":
//!
//! * **Pseudocosts** — cheap, learned from history: each time a real child node
//!   is solved, the observed lb gain per unit of domain change updates that
//!   variable/direction's running average. Free, but unreliable until a
//!   variable has been branched a few times.
//! * **Strong branching** — accurate, expensive: tentatively split the
//!   variable, solve both child relaxations, and read the *actual* gains.
//!
//! The rule uses strong branching on a variable until its pseudocost is
//! *reliable* (updated `ETA` times in each direction), then trusts the cheap
//! pseudocost. Early nodes pay for accuracy; once pseudocosts mature the cost
//! evaporates. A per-node strong-branching budget and a lookahead cap bound the
//! work.

use crate::problem::GlobalProblem;
use crate::relax::{branch_scores, build_relaxation, BranchTerm};
use pounce_convex::{solve_qp_ipm, QpOptions, QpStatus};
use pounce_linsol::SparseSymLinearSolverInterface;

/// Reliability threshold: branch a variable by strong branching until each
/// direction's pseudocost has this many observations.
const ETA: usize = 8;
/// Strong-branching evaluations allowed per node.
const MAX_SB: usize = 8;
/// Stop strong branching after this many consecutive non-improving candidates.
const LOOKAHEAD: usize = 8;
/// Floor in the product score so a zero gain in one direction doesn't annihilate
/// the score (Achterberg's `score = max(Δ⁻,ε)·max(Δ⁺,ε)`).
const SCORE_EPS: f64 = 1e-6;

/// Where to split `[lo, hi]`: the relaxation point when strictly interior, else
/// the midpoint. Shared by the driver and the strong-branching probe.
pub(crate) fn split_point(x: f64, lo: f64, hi: f64) -> f64 {
    let w = hi - lo;
    let margin = 1e-4 * w;
    if x.is_finite() && x > lo + margin && x < hi - margin {
        x
    } else {
        0.5 * (lo + hi)
    }
}

/// Down/up pseudocost statistics per variable: a running mean of lb-gain per
/// unit child-width.
pub(crate) struct PseudoCosts {
    down: Vec<(f64, usize)>,
    up: Vec<(f64, usize)>,
}

impl PseudoCosts {
    pub(crate) fn new(n: usize) -> Self {
        PseudoCosts {
            down: vec![(0.0, 0); n],
            up: vec![(0.0, 0); n],
        }
    }

    /// Record an observed `gain` over a child of width `frac` for variable `i`
    /// in the down (`down = true`) or up direction.
    pub(crate) fn update(&mut self, i: usize, down: bool, gain: f64, frac: f64) {
        let unit = (gain.max(0.0)) / frac.max(1e-12);
        let s = if down {
            &mut self.down[i]
        } else {
            &mut self.up[i]
        };
        s.0 += unit;
        s.1 += 1;
    }

    fn mean(stat: (f64, usize), fallback: f64) -> f64 {
        if stat.1 > 0 {
            stat.0 / stat.1 as f64
        } else {
            fallback
        }
    }

    /// Mean of all initialized pseudocosts (the standard cold-start default for
    /// an uninitialized variable); `1.0` if none seen yet.
    fn global_mean(&self) -> f64 {
        let mut sum = 0.0;
        let mut cnt = 0;
        for s in self.down.iter().chain(self.up.iter()) {
            if s.1 > 0 {
                sum += s.0 / s.1 as f64;
                cnt += 1;
            }
        }
        if cnt > 0 {
            sum / cnt as f64
        } else {
            1.0
        }
    }

    fn reliable(&self, i: usize) -> bool {
        self.down[i].1 >= ETA && self.up[i].1 >= ETA
    }
}

fn product_score(down_gain: f64, up_gain: f64) -> f64 {
    down_gain.max(SCORE_EPS) * up_gain.max(SCORE_EPS)
}

/// Quick lower bound of a child box for strong branching: the McCormick
/// relaxation LP. `+∞` signals an infeasible child (a maximal "gain").
fn quick_lb<F>(
    prob: &GlobalProblem,
    lo: &[f64],
    hi: &[f64],
    multilinear: bool,
    qp_opts: &QpOptions,
    make_backend: &mut F,
) -> Option<f64>
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let relax = build_relaxation(prob, lo, hi, multilinear);
    if relax.trivially_infeasible {
        return Some(f64::INFINITY);
    }
    let sol = solve_qp_ipm(&relax.qp, qp_opts, make_backend);
    match sol.status {
        QpStatus::Optimal => Some(sol.obj),
        QpStatus::PrimalInfeasible => Some(f64::INFINITY),
        _ => None,
    }
}

/// Select a branching variable by reliability branching. `node_lb` is the
/// current node's lower bound; `pc` is updated in place by any strong-branching
/// probes. Returns `None` when no variable is both branchable and violated (the
/// caller then falls back to the widest side).
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_reliability<F>(
    prob: &GlobalProblem,
    lo: &[f64],
    hi: &[f64],
    relax_pt: &[f64],
    terms: &[BranchTerm],
    sol_x: &[f64],
    node_lb: f64,
    pc: &mut PseudoCosts,
    box_tol: f64,
    multilinear: bool,
    qp_opts: &QpOptions,
    make_backend: &mut F,
) -> Option<usize>
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let n = lo.len();
    let scores = branch_scores(terms, sol_x, n);
    let mut cands: Vec<usize> = (0..n)
        .filter(|&i| hi[i] - lo[i] > box_tol && scores[i] > 1e-9)
        .collect();
    if cands.is_empty() {
        return None;
    }

    let frac = |i: usize| {
        let s = split_point(relax_pt[i], lo[i], hi[i]);
        (s, (s - lo[i]).max(1e-12), (hi[i] - s).max(1e-12))
    };
    let gmean = pc.global_mean();
    let pseudo_score = |pc: &PseudoCosts, i: usize, fd: f64, fu: f64| {
        product_score(
            PseudoCosts::mean(pc.down[i], gmean) * fd,
            PseudoCosts::mean(pc.up[i], gmean) * fu,
        )
    };

    // Process the most promising candidates first (by current pseudocost
    // prediction) so the strong-branching budget is spent where it matters.
    cands.sort_by(|&a, &b| {
        let (_, fda, fua) = frac(a);
        let (_, fdb, fub) = frac(b);
        pseudo_score(pc, b, fdb, fub)
            .partial_cmp(&pseudo_score(pc, a, fda, fua))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut best = (f64::NEG_INFINITY, cands[0]);
    let mut sb_used = 0;
    let mut lookahead = 0;
    for &i in &cands {
        let (s, fd, fu) = frac(i);
        let score = if pc.reliable(i) || sb_used >= MAX_SB || lookahead >= LOOKAHEAD {
            pseudo_score(pc, i, fd, fu)
        } else {
            // Strong branch: cheap lower bound of each child.
            let mut down_hi = hi.to_vec();
            down_hi[i] = s;
            let down_lb =
                quick_lb(prob, lo, &down_hi, multilinear, qp_opts, make_backend).unwrap_or(node_lb);
            let mut up_lo = lo.to_vec();
            up_lo[i] = s;
            let up_lb =
                quick_lb(prob, &up_lo, hi, multilinear, qp_opts, make_backend).unwrap_or(node_lb);
            let dg = (down_lb - node_lb).max(0.0);
            let ug = (up_lb - node_lb).max(0.0);
            pc.update(i, true, dg, fd);
            pc.update(i, false, ug, fu);
            sb_used += 1;
            product_score(dg, ug)
        };
        if score > best.0 {
            best = (score, i);
            lookahead = 0;
        } else {
            lookahead += 1;
        }
    }
    Some(best.1)
}
