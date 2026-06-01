//! Spatial branch-and-bound driver.
//!
//! Best-first search over boxes. Each node: tighten the box with FBBT and OBBT
//! (prune on infeasibility), build and solve the McCormick relaxation LP for a
//! lower bound (prune against the incumbent), probe for a feasible point to
//! improve the incumbent upper bound, then branch (see [`BranchRule`]). Because
//! the relaxation is exact in the limit of a zero-width box, the incumbent and
//! the frontier lower bound squeeze together and the search returns a globally
//! optimal point with a certified optimality gap.

use crate::expr::eval;
use crate::problem::{ConstraintProvider, GlobalProblem};
use crate::relax::build_relaxation;
use pounce_convex::{solve_qp_ipm, QpOptions, QpStatus};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_presolve::fbbt::{run_fbbt, FbbtConfig};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Which variable to branch on at a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchRule {
    /// The widest box side. Simplest; geometry-only.
    Widest,
    /// The variable with the largest relaxation violation (its nonconvexity
    /// drives the gap), falling back to widest.
    MostViolation,
    /// Reliability branching: pseudocosts learned from child solves, with
    /// strong branching until a variable's pseudocost is reliable. The MINLP
    /// SOTA rule; most useful on larger problems where variable choice
    /// dominates the node count.
    Reliability,
}

/// Termination status of a global solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalStatus {
    /// A globally optimal point was found and the optimality gap is within
    /// tolerance.
    Optimal,
    /// The feasible set was proven empty (no point satisfies the constraints
    /// over the box).
    Infeasible,
    /// The node budget was exhausted before the gap closed; `x` is the best
    /// point found and `[lower_bound, objective]` brackets the global optimum.
    NodeLimit,
}

/// Result of [`solve_global`].
#[derive(Debug, Clone)]
pub struct GlobalSolution {
    pub status: GlobalStatus,
    /// Best feasible point found (empty if none / infeasible).
    pub x: Vec<f64>,
    /// Objective at `x` — a valid global **upper** bound.
    pub objective: f64,
    /// Certified global **lower** bound. `objective − lower_bound` is the
    /// optimality gap.
    pub lower_bound: f64,
    /// Branch-and-bound nodes processed.
    pub nodes: usize,
}

impl GlobalSolution {
    /// Optimality gap `objective − lower_bound` (`+∞` if no incumbent yet).
    pub fn gap(&self) -> f64 {
        self.objective - self.lower_bound
    }
}

/// Tuning for the global solve.
#[derive(Debug, Clone)]
pub struct GlobalOptions {
    /// Absolute optimality-gap tolerance: stop when `ub − lb ≤ abs_gap`.
    pub abs_gap: f64,
    /// Relative optimality-gap tolerance: stop when `ub − lb ≤ rel_gap·|ub|`.
    pub rel_gap: f64,
    /// Constraint feasibility tolerance for accepting an incumbent point.
    pub feas_tol: f64,
    /// Stop branching a box once its widest side is `≤ box_tol`.
    pub box_tol: f64,
    /// Maximum branch-and-bound nodes.
    pub max_nodes: usize,
    /// Interior-point iteration cap for the per-node local NLP upper-bound
    /// solve. `0` disables local solves (upper bounds then come only from
    /// probing the relaxation point and box center).
    pub local_solve_iters: usize,
    /// Maximum cutting-plane ("sandwich") rounds per node: after the relaxation
    /// LP, add tangent cuts at the solution for loose convex/concave atoms and
    /// re-solve, tightening the lower bound without branching. `0` disables.
    pub sandwich_rounds: usize,
    /// Optimization-based bound-tightening passes per node (each pass is `2n` LP
    /// solves that minimize/maximize every variable over the relaxation, with an
    /// incumbent cutoff). The strongest box reducer, but costly — `0` disables.
    pub obbt_passes: usize,
    /// Number of αBB tangent-plane underestimator cuts added to the objective
    /// per node (sample points across the box). αBB convexifies the objective as
    /// a whole via an interval-Hessian spectral shift, complementing the
    /// factorable relaxation. `0` disables.
    pub alphabb_cuts: usize,
    /// Add level-1 RLT cuts (affine constraints × variable bound factors,
    /// linearized with shared product columns). Tightens bilinearly coupled
    /// problems; a no-op when there are no affine constraints.
    pub rlt: bool,
    /// Use the tighter multi-grouping relaxation of 3-way products (intersect
    /// all three bilinear groupings instead of one nested grouping).
    pub multilinear: bool,
    /// Which branching rule to use.
    pub branching: BranchRule,
    /// FBBT configuration for per-node bound tightening.
    pub fbbt: FbbtConfig,
}

impl Default for GlobalOptions {
    fn default() -> Self {
        GlobalOptions {
            abs_gap: 1e-6,
            rel_gap: 1e-6,
            feas_tol: 1e-6,
            box_tol: 1e-7,
            max_nodes: 5000,
            local_solve_iters: 50,
            sandwich_rounds: 4,
            obbt_passes: 2,
            alphabb_cuts: 1,
            rlt: true,
            multilinear: true,
            branching: BranchRule::MostViolation,
            fbbt: FbbtConfig::default(),
        }
    }
}

/// How a child node was branched from its parent — used to update pseudocosts
/// once the child is solved (observed lb gain over the parent).
#[derive(Clone, Copy)]
struct BranchInfo {
    var: usize,
    down: bool,
    frac: f64,
    parent_lb: f64,
}

/// A frontier node: a box and the (valid) lower bound inherited from its
/// parent's relaxation, used as the best-first priority.
struct Node {
    key: f64,
    lo: Vec<f64>,
    hi: Vec<f64>,
    branch: Option<BranchInfo>,
}

// BinaryHeap is a max-heap; invert so the smallest `key` is popped first.
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.key.total_cmp(&self.key)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Evaluate `prob`'s objective at `x` if `x` is feasible (constraints within
/// tolerance); `None` otherwise.
fn feasible_objective(prob: &GlobalProblem, x: &[f64], tol: f64) -> Option<f64> {
    for con in &prob.constraints {
        let v = eval(&con.tape, x);
        if !v.is_finite() || v < con.lo - tol || v > con.hi + tol {
            return None;
        }
    }
    let obj = eval(&prob.objective, x);
    obj.is_finite().then_some(obj)
}

/// Globally minimize `prob`. `make_backend` supplies a fresh sparse symmetric
/// linear solver for each relaxation LP (e.g. `FeralSolverInterface::new`).
pub fn solve_global<F>(
    prob: &GlobalProblem,
    opts: &GlobalOptions,
    mut make_backend: F,
) -> GlobalSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let n = prob.n_vars;
    let (g_lo, g_hi) = prob.constraint_bounds();
    let qp_opts = QpOptions::default();

    let mut heap: BinaryHeap<Node> = BinaryHeap::new();
    heap.push(Node {
        key: f64::NEG_INFINITY,
        lo: prob.x_lo.clone(),
        hi: prob.x_hi.clone(),
        branch: None,
    });
    let mut pseudo = crate::branching::PseudoCosts::new(n);

    let mut incumbent_x: Vec<f64> = Vec::new();
    let mut incumbent_ub = f64::INFINITY;
    let mut global_lb = f64::NEG_INFINITY;
    let mut nodes = 0usize;

    let gap_ok = |lb: f64, ub: f64| -> bool {
        let g = ub - lb;
        g <= opts.abs_gap || g <= opts.rel_gap * ub.abs().max(1.0)
    };

    while let Some(node) = heap.pop() {
        // Best-first: this node's key is the minimum over the whole frontier,
        // hence a valid global lower bound at this moment.
        if node.key.is_finite() {
            global_lb = node.key;
        }
        // Everything remaining has key ≥ node.key, so once that clears the
        // incumbent the global optimum is pinned to the incumbent.
        if incumbent_ub.is_finite() && gap_ok(node.key, incumbent_ub) {
            // The frontier minimum `node.key` already meets the incumbent, so
            // nothing unexplored can beat it. A valid global lower bound never
            // exceeds the achievable incumbent, so clamp.
            let lb = if node.key.is_finite() {
                node.key
            } else {
                global_lb
            };
            return GlobalSolution {
                status: GlobalStatus::Optimal,
                x: incumbent_x,
                objective: incumbent_ub,
                lower_bound: lb.min(incumbent_ub),
                nodes,
            };
        }
        if nodes >= opts.max_nodes {
            let status = if incumbent_ub.is_finite() && (incumbent_ub - global_lb) <= opts.abs_gap {
                GlobalStatus::Optimal
            } else {
                GlobalStatus::NodeLimit
            };
            return GlobalSolution {
                status,
                x: incumbent_x,
                objective: incumbent_ub,
                lower_bound: global_lb.min(incumbent_ub),
                nodes,
            };
        }
        nodes += 1;

        // 1. FBBT bound tightening (prune on infeasibility witness).
        let mut lo = node.lo.clone();
        let mut hi = node.hi.clone();
        if !prob.constraints.is_empty() {
            let provider = ConstraintProvider::new(&prob.constraints);
            let report = run_fbbt(
                &provider,
                n,
                prob.constraints.len(),
                &mut lo,
                &mut hi,
                &g_lo,
                &g_hi,
                &opts.fbbt,
            );
            if report.infeasibility_witness.is_some() {
                continue;
            }
        }
        if (0..n).any(|i| lo[i] > hi[i] + 1e-12) {
            continue; // empty box
        }

        // 1b. Optimization-based bound tightening (with the incumbent cutoff),
        // a stronger box reducer than FBBT — prune if it collapses the box.
        if opts.obbt_passes > 0 {
            let cutoff = incumbent_ub.is_finite().then_some(incumbent_ub);
            if !crate::obbt::tighten(
                prob,
                &mut lo,
                &mut hi,
                cutoff,
                opts.obbt_passes,
                &qp_opts,
                &mut make_backend,
            ) {
                continue;
            }
        }

        // 2. Relaxation lower bound, tightened by cutting-plane (sandwich)
        // rounds: re-solve with tangent cuts added at the LP point for loose
        // convex/concave atoms until the bound stops improving.
        let relax = build_relaxation(prob, &lo, &hi, opts.multilinear);
        if relax.trivially_infeasible {
            continue;
        }
        let mut qp = relax.qp;
        let atoms = relax.atoms;
        let branch_terms = relax.branch_terms;
        let (col_lo, col_hi) = (qp.lb.clone(), qp.ub.clone());
        // αBB tangent-plane underestimators of the objective as a whole,
        // complementing the factorable relaxation of its individual atoms.
        if opts.alphabb_cuts > 0 {
            if let Some(oc) = relax.obj_col {
                let cuts = crate::alphabb::objective_cuts(
                    &prob.objective,
                    &lo,
                    &hi,
                    oc,
                    opts.alphabb_cuts,
                );
                crate::relax::append_cuts(&mut qp, &cuts);
            }
        }
        // RLT cuts (affine constraints × bound factors): appends product columns
        // + McCormick + the linearized cuts. No-op without affine constraints.
        if opts.rlt {
            crate::rlt::augment(&mut qp, prob, &lo, &hi);
        }
        let sol = solve_qp_ipm(&qp, &qp_opts, &mut make_backend);
        let mut node_lb = match sol.status {
            QpStatus::Optimal => sol.obj,
            QpStatus::PrimalInfeasible => continue, // box is infeasible → prune
            // Dual-infeasible (unbounded relaxation) or numerical trouble: keep
            // the inherited bound and keep branching rather than prune wrongly.
            _ => node.key,
        };
        // Branch from and probe at the *original* relaxation point — it marks
        // where the relaxation is loosest; the cuts only sharpen the bound.
        let relax_pt: Vec<f64> = (0..n).map(|i| sol.x[i].clamp(lo[i], hi[i])).collect();
        if sol.status == QpStatus::Optimal {
            let mut cut_x = sol.x.clone();
            for _ in 0..opts.sandwich_rounds {
                let cuts = crate::relax::sandwich_cuts(&atoms, &col_lo, &col_hi, &cut_x, 1e-7);
                if cuts.is_empty() {
                    break;
                }
                crate::relax::append_cuts(&mut qp, &cuts);
                let s = solve_qp_ipm(&qp, &qp_opts, &mut make_backend);
                if s.status != QpStatus::Optimal || s.obj <= node_lb + 1e-9 {
                    break;
                }
                node_lb = s.obj;
                cut_x = s.x;
            }
        }
        // Learn: update the branched variable's pseudocost with the realized
        // lower-bound gain of this child over its parent.
        if let Some(bi) = node.branch {
            pseudo.update(bi.var, bi.down, node_lb - bi.parent_lb, bi.frac);
        }
        if incumbent_ub.is_finite() && gap_ok(node_lb, incumbent_ub) {
            continue;
        }

        // 3. Upper bound: probe the relaxation point and box center, and (when
        // enabled) polish the relaxation point with a local NLP solve over the
        // node box for a much sharper feasible incumbent.
        let center: Vec<f64> = (0..n).map(|i| 0.5 * (lo[i] + hi[i])).collect();
        let mut candidates = vec![relax_pt.clone(), center];
        if opts.local_solve_iters > 0 {
            if let Some(polished) =
                crate::nlp::local_solve(prob, &lo, &hi, &relax_pt, opts.local_solve_iters)
            {
                candidates.push(polished);
            }
        }
        for cand in &candidates {
            if let Some(val) = feasible_objective(prob, cand, opts.feas_tol) {
                if val < incumbent_ub {
                    incumbent_ub = val;
                    incumbent_x = cand.clone();
                }
            }
        }

        // 4. Leaf test, else branch. The leaf test is on the overall widest
        // side; the branching variable comes from the configured rule. Split at
        // the relaxation point — where the relaxation is loosest.
        let (widest_k, width) = widest(&lo, &hi);
        let lb_for_children = node_lb.max(node.key);
        if width <= opts.box_tol
            || (incumbent_ub.is_finite() && gap_ok(lb_for_children, incumbent_ub))
        {
            continue;
        }
        let k = match opts.branching {
            BranchRule::Widest => widest_k,
            BranchRule::MostViolation => {
                let scores = crate::relax::branch_scores(&branch_terms, &sol.x, n);
                (0..n)
                    .filter(|&i| hi[i] - lo[i] > opts.box_tol)
                    .max_by(|&i, &j| scores[i].partial_cmp(&scores[j]).unwrap_or(Ordering::Equal))
                    .filter(|&i| scores[i] > 1e-9)
                    .unwrap_or(widest_k)
            }
            BranchRule::Reliability => crate::branching::select_reliability(
                prob,
                &lo,
                &hi,
                &relax_pt,
                &branch_terms,
                &sol.x,
                node_lb,
                &mut pseudo,
                opts.box_tol,
                opts.multilinear,
                &qp_opts,
                &mut make_backend,
            )
            .unwrap_or(widest_k),
        };
        let split = crate::branching::split_point(relax_pt[k], lo[k], hi[k]);
        let f_down = (split - lo[k]).max(1e-12);
        let f_up = (hi[k] - split).max(1e-12);
        let mut left_hi = hi.clone();
        left_hi[k] = split;
        let mut right_lo = lo.clone();
        right_lo[k] = split;
        heap.push(Node {
            key: lb_for_children,
            lo: lo.clone(),
            hi: left_hi,
            branch: Some(BranchInfo {
                var: k,
                down: true,
                frac: f_down,
                parent_lb: node_lb,
            }),
        });
        heap.push(Node {
            key: lb_for_children,
            lo: right_lo,
            hi: hi.clone(),
            branch: Some(BranchInfo {
                var: k,
                down: false,
                frac: f_up,
                parent_lb: node_lb,
            }),
        });
    }

    // Frontier exhausted: every region was resolved by pruning or shrunk to a
    // leaf, so nothing unexplored can beat the incumbent — it is global.
    let _ = global_lb;
    if incumbent_ub.is_finite() {
        GlobalSolution {
            status: GlobalStatus::Optimal,
            x: incumbent_x,
            objective: incumbent_ub,
            lower_bound: incumbent_ub,
            nodes,
        }
    } else {
        GlobalSolution {
            status: GlobalStatus::Infeasible,
            x: Vec::new(),
            objective: f64::INFINITY,
            lower_bound: global_lb,
            nodes,
        }
    }
}

fn widest(lo: &[f64], hi: &[f64]) -> (usize, f64) {
    let mut k = 0;
    let mut w = f64::NEG_INFINITY;
    for i in 0..lo.len() {
        let wi = hi[i] - lo[i];
        if wi > w {
            w = wi;
            k = i;
        }
    }
    (k, w)
}
