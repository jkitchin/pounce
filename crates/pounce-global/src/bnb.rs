//! Spatial branch-and-bound driver.
//!
//! Best-first search over boxes. Each node: tighten the box with FBBT and OBBT
//! (prune on infeasibility), build and solve the McCormick relaxation LP for a
//! lower bound (prune against the incumbent), probe for a feasible point to
//! improve the incumbent upper bound, then branch (see [`BranchRule`]). Because
//! the relaxation is exact in the limit of a zero-width box, the incumbent and
//! the frontier lower bound squeeze together and the search returns a globally
//! optimal point with a certified optimality gap.

use crate::debug::{fire_tree, BnbDebugState};
use crate::expr::eval;
use crate::problem::{ConstraintProvider, GlobalProblem};
use crate::relax::build_relaxation;
use pounce_common::debug::{DebugAction, DebugHook, PruneReason, TreeCheckpoint, TreeDebugHook};
use pounce_convex::{solve_qp_ipm, solve_qp_ipm_debug, QpOptions, QpStatus};
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
    /// Peak number of open nodes held on the best-first frontier at once. The
    /// frontier is the dominant memory consumer; this is its high-water mark.
    pub peak_frontier: usize,
    /// Estimated peak frontier memory in bytes (`peak_frontier ×`
    /// [`estimate_node_bytes`]). The transient per-node relaxation LP is freed
    /// each node and not counted; this is the resident-set term that grows with
    /// the search.
    pub peak_memory_bytes: usize,
}

impl GlobalSolution {
    /// Optimality gap `objective − lower_bound` (`+∞` if no incumbent yet).
    pub fn gap(&self) -> f64 {
        self.objective - self.lower_bound
    }
}

/// Estimated heap bytes one frontier node occupies for an `n_vars`-variable
/// problem: the node struct plus its two owned length-`n` box vectors. Used to
/// project frontier memory before and after a solve.
pub fn estimate_node_bytes(n_vars: usize) -> usize {
    std::mem::size_of::<Node>() + 2 * n_vars * std::mem::size_of::<f64>()
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
    /// Run OBBT's `2n` per-node bound-tightening solves on a thread pool. The
    /// result is identical to the serial sweep (the solves are independent),
    /// only faster; pays off when `n` and the relaxation are large enough to
    /// amortize threading overhead. (Ignored when `threads > 1`, where whole
    /// nodes run in parallel instead and OBBT stays serial within a worker.)
    pub parallel: bool,
    /// Worker threads for the parallel **node pool**: `> 1` processes whole
    /// frontier nodes concurrently (coarse-grained, the bigger speedup) at the
    /// cost of determinism — node counts vary run to run, but the certified
    /// optimum and gap do not. `1` (default) is the deterministic serial driver
    /// and uses most-violation/reliability branching; the pool uses
    /// most-violation (pseudocosts are not shared across workers).
    pub threads: usize,
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
            parallel: false,
            threads: 1,
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
    /// Depth in the tree (root = 0); tracked for the tree debugger.
    depth: usize,
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

fn gap_ok(lb: f64, ub: f64, opts: &GlobalOptions) -> bool {
    let g = ub - lb;
    g <= opts.abs_gap || g <= opts.rel_gap * ub.abs().max(1.0)
}

/// The bounding result for one node: its lower bound, the (FBBT/OBBT-tightened)
/// box, the relaxation point and branch terms for branching, and a feasible
/// incumbent candidate found while probing. `None` from [`process_node`] means
/// the node was pruned (proven infeasible / dominated).
struct Bounded {
    node_lb: f64,
    lo: Vec<f64>,
    hi: Vec<f64>,
    relax_pt: Vec<f64>,
    branch_terms: Vec<crate::relax::BranchTerm>,
    sol_x: Vec<f64>,
    incumbent: Option<(Vec<f64>, f64)>,
}

/// All the per-node work that is independent of the shared frontier: FBBT +
/// OBBT bound tightening, the relaxation lower bound (with αBB / RLT cuts and
/// sandwich refinement), and upper-bound probing. Pure given the box and an
/// incumbent snapshot, so it runs unsynchronized in both the serial and the
/// parallel drivers. `obbt_parallel` is forced off in the parallel pool to
/// avoid nesting thread pools.
#[allow(clippy::too_many_arguments)]
fn process_node<F>(
    prob: &GlobalProblem,
    opts: &GlobalOptions,
    mut lo: Vec<f64>,
    mut hi: Vec<f64>,
    parent_key: f64,
    incumbent_ub: f64,
    g_lo: &[f64],
    g_hi: &[f64],
    qp_opts: &QpOptions,
    obbt_parallel: bool,
    make_backend: &F,
    mut subsolve_hook: Option<&mut dyn DebugHook>,
) -> Option<Bounded>
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    let n = prob.n_vars;
    let mut mk = || make_backend();

    // 1. FBBT bound tightening (prune on infeasibility witness).
    if !prob.constraints.is_empty() {
        let provider = ConstraintProvider::new(&prob.constraints);
        let report = run_fbbt(
            &provider,
            n,
            prob.constraints.len(),
            &mut lo,
            &mut hi,
            g_lo,
            g_hi,
            &opts.fbbt,
        );
        if report.infeasibility_witness.is_some() {
            return None;
        }
    }
    if (0..n).any(|i| lo[i] > hi[i] + 1e-12) {
        return None; // empty box
    }

    // 1b. Optimization-based bound tightening (with the incumbent cutoff).
    if opts.obbt_passes > 0 {
        let cutoff = incumbent_ub.is_finite().then_some(incumbent_ub);
        if !crate::obbt::tighten(
            prob,
            &mut lo,
            &mut hi,
            cutoff,
            opts.obbt_passes,
            obbt_parallel,
            qp_opts,
            make_backend,
        ) {
            return None;
        }
    }

    // 2. Relaxation lower bound + αBB / RLT cuts + sandwich refinement.
    let relax = build_relaxation(prob, &lo, &hi, opts.multilinear);
    if relax.trivially_infeasible {
        return None;
    }
    let mut qp = relax.qp;
    let atoms = relax.atoms;
    let branch_terms = relax.branch_terms;
    let (col_lo, col_hi) = (qp.lb.clone(), qp.ub.clone());
    if opts.alphabb_cuts > 0 {
        if let Some(oc) = relax.obj_col {
            let cuts =
                crate::alphabb::objective_cuts(&prob.objective, &lo, &hi, oc, opts.alphabb_cuts);
            crate::relax::append_cuts(&mut qp, &cuts);
        }
    }
    if opts.rlt {
        crate::rlt::augment(&mut qp, prob, &lo, &hi);
    }
    // The node's relaxation lower bound. When the tree debugger stepped into
    // this node, `subsolve_hook` is armed and we run it under the
    // interior-point debugger; `take` keeps the later sandwich re-solves
    // un-debugged.
    let sol = match subsolve_hook.take() {
        Some(h) => solve_qp_ipm_debug(&qp, qp_opts, h, &mut mk),
        None => solve_qp_ipm(&qp, qp_opts, &mut mk),
    };
    let mut node_lb = match sol.status {
        QpStatus::Optimal => sol.obj,
        QpStatus::PrimalInfeasible => return None,
        _ => parent_key, // numerical trouble: keep the inherited bound
    };
    // Branch from / probe at the original relaxation point (loosest there);
    // cuts only sharpen the bound.
    let relax_pt: Vec<f64> = (0..n).map(|i| sol.x[i].clamp(lo[i], hi[i])).collect();
    if sol.status == QpStatus::Optimal {
        let mut cut_x = sol.x.clone();
        for _ in 0..opts.sandwich_rounds {
            let cuts = crate::relax::sandwich_cuts(&atoms, &col_lo, &col_hi, &cut_x, 1e-7);
            if cuts.is_empty() {
                break;
            }
            crate::relax::append_cuts(&mut qp, &cuts);
            let s = solve_qp_ipm(&qp, qp_opts, &mut mk);
            if s.status != QpStatus::Optimal || s.obj <= node_lb + 1e-9 {
                break;
            }
            node_lb = s.obj;
            cut_x = s.x;
        }
    }

    // 3. Upper bound: probe the relaxation point, the box center, and (when
    // enabled) a local NLP polish; return the best feasible point found.
    let mut incumbent: Option<(Vec<f64>, f64)> = None;
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
            if incumbent.as_ref().is_none_or(|(_, o)| val < *o) {
                incumbent = Some((cand.clone(), val));
            }
        }
    }

    Some(Bounded {
        node_lb,
        lo,
        hi,
        relax_pt,
        branch_terms,
        sol_x: sol.x,
        incumbent,
    })
}

/// Most-violation branching variable for a bounded node (stateless — used by
/// the parallel pool, where pseudocosts are not shared). `None` ⇒ caller falls
/// back to the widest side.
fn select_most_violation(b: &Bounded, box_tol: f64, n: usize) -> Option<usize> {
    let scores = crate::relax::branch_scores(&b.branch_terms, &b.sol_x, n);
    (0..n)
        .filter(|&i| b.hi[i] - b.lo[i] > box_tol)
        .max_by(|&i, &j| scores[i].partial_cmp(&scores[j]).unwrap_or(Ordering::Equal))
        .filter(|&i| scores[i] > 1e-9)
}

/// Build the two child nodes from a branched bounded node.
fn children(b: &Bounded, k: usize, lb_for_children: f64, parent_depth: usize) -> [Node; 2] {
    let split = crate::branching::split_point(b.relax_pt[k], b.lo[k], b.hi[k]);
    let f_down = (split - b.lo[k]).max(1e-12);
    let f_up = (b.hi[k] - split).max(1e-12);
    let mut left_hi = b.hi.clone();
    left_hi[k] = split;
    let mut right_lo = b.lo.clone();
    right_lo[k] = split;
    let depth = parent_depth + 1;
    [
        Node {
            key: lb_for_children,
            lo: b.lo.clone(),
            hi: left_hi,
            branch: Some(BranchInfo {
                var: k,
                down: true,
                frac: f_down,
                parent_lb: b.node_lb,
            }),
            depth,
        },
        Node {
            key: lb_for_children,
            lo: right_lo,
            hi: b.hi.clone(),
            branch: Some(BranchInfo {
                var: k,
                down: false,
                frac: f_up,
                parent_lb: b.node_lb,
            }),
            depth,
        },
    ]
}

/// Globally minimize `prob`. `make_backend` supplies a fresh sparse symmetric
/// linear solver for each relaxation LP (e.g. `FeralSolverInterface::new`).
/// `opts.threads > 1` runs the [parallel node pool](solve_parallel); otherwise
/// the deterministic serial driver.
pub fn solve_global<F>(
    prob: &GlobalProblem,
    opts: &GlobalOptions,
    make_backend: F,
) -> GlobalSolution
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    if opts.threads > 1 {
        solve_parallel(prob, opts, &make_backend, opts.threads)
    } else {
        solve_serial(prob, opts, make_backend, None, None)
    }
}

/// Globally minimize `prob` with an interactive [`TreeDebugHook`] attached:
/// the hook is fired at each branch-and-bound checkpoint (node selection,
/// relaxation, incumbent, prune, branch, termination) so a debugger can step
/// the tree, inspect node boxes / global bounds / the gap, and break.
///
/// Always runs the **serial** driver (the parallel node pool is not
/// debuggable — concurrent nodes have no single well-defined "current node");
/// apart from the hook the result matches [`solve_global`] at `threads = 1`.
pub fn solve_global_debug<F>(
    prob: &GlobalProblem,
    opts: &GlobalOptions,
    hook: &mut dyn TreeDebugHook,
    make_backend: F,
) -> GlobalSolution
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    solve_serial(prob, opts, make_backend, Some(hook), None)
}

/// Like [`solve_global_debug`], but also threads an interior-point
/// [`DebugHook`] that the tree debugger arms on demand to **step into** a
/// node's relaxation solve. The `subsolve_hook` stays quiet until a
/// `request_subsolve_debug` (the REPL's "step into") arms it for one node.
pub fn solve_global_debug_into<F>(
    prob: &GlobalProblem,
    opts: &GlobalOptions,
    hook: &mut dyn TreeDebugHook,
    subsolve_hook: &mut dyn DebugHook,
    make_backend: F,
) -> GlobalSolution
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    solve_serial(prob, opts, make_backend, Some(hook), Some(subsolve_hook))
}

/// Deterministic best-first serial driver. `hook`, when present, is fired at
/// each tree checkpoint (node selection, relaxation, incumbent, prune, branch,
/// termination) and may stop the search.
fn solve_serial<F>(
    prob: &GlobalProblem,
    opts: &GlobalOptions,
    mut make_backend: F,
    mut hook: Option<&mut dyn TreeDebugHook>,
    mut subsolve_hook: Option<&mut dyn DebugHook>,
) -> GlobalSolution
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
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
        depth: 0,
    });
    let mut pseudo = crate::branching::PseudoCosts::new(n);
    let mut incumbent_x: Vec<f64> = Vec::new();
    let mut incumbent_ub = f64::INFINITY;
    let mut global_lb = f64::NEG_INFINITY;
    let mut nodes = 0usize;
    let node_bytes = estimate_node_bytes(n);
    let mut peak_frontier = heap.len();
    let mut node_id = 0u64;

    // Single exit: a `break 'search` records the outcome here, the natural
    // (frontier-exhausted) end fills it after the loop, then `Terminated`
    // fires once and the solution is built.
    let mut status = GlobalStatus::Infeasible;
    let mut final_lb = global_lb;
    let mut exited = false;

    // Fire a tree checkpoint with the current search state; evaluates to
    // `true` if the hook asked to stop. A no-op when no hook is attached.
    macro_rules! fire_cp {
        ($cp:expr, $lo:expr, $hi:expr, $node_lb:expr, $depth:expr, $bvar:expr, $prune:expr) => {{
            if hook.is_some() {
                let mut st = BnbDebugState {
                    cp: $cp,
                    node_id,
                    depth: $depth,
                    nodes: nodes as u64,
                    frontier_len: heap.len(),
                    lo: $lo,
                    hi: $hi,
                    node_lb: $node_lb,
                    global_lb,
                    incumbent: incumbent_ub.is_finite().then_some(incumbent_ub),
                    incumbent_x: (!incumbent_x.is_empty()).then_some(incumbent_x.as_slice()),
                    branch_var: $bvar,
                    prune_reason: $prune,
                    status: None,
                    arm: None,
                };
                matches!(fire_tree(&mut hook, &mut st), DebugAction::Stop)
            } else {
                false
            }
        }};
    }
    // Outcome to record when the debugger stops the search early.
    let stop_status = |ub: f64| {
        if ub.is_finite() {
            GlobalStatus::NodeLimit
        } else {
            GlobalStatus::Infeasible
        }
    };

    'search: while let Some(node) = heap.pop() {
        node_id += 1;
        if node.key.is_finite() {
            global_lb = node.key;
        }

        // NodeSelected is fired inline (not via `fire_cp!`) so the hook can
        // arm the next relaxation's interior-point sub-solve via `arm`.
        let mut arm_next = false;
        if hook.is_some() {
            let stop = {
                let mut st = BnbDebugState {
                    cp: TreeCheckpoint::NodeSelected,
                    node_id,
                    depth: node.depth,
                    nodes: nodes as u64,
                    frontier_len: heap.len(),
                    lo: &node.lo,
                    hi: &node.hi,
                    node_lb: f64::NAN,
                    global_lb,
                    incumbent: incumbent_ub.is_finite().then_some(incumbent_ub),
                    incumbent_x: (!incumbent_x.is_empty()).then_some(incumbent_x.as_slice()),
                    branch_var: None,
                    prune_reason: None,
                    status: None,
                    arm: Some(&mut arm_next),
                };
                matches!(fire_tree(&mut hook, &mut st), DebugAction::Stop)
            };
            if stop {
                status = stop_status(incumbent_ub);
                final_lb = global_lb.min(incumbent_ub);
                exited = true;
                break 'search;
            }
        }

        // Best-first: this node's key is the frontier minimum, so once it meets
        // the incumbent nothing unexplored can beat it.
        if incumbent_ub.is_finite() && gap_ok(node.key, incumbent_ub, opts) {
            let lb = if node.key.is_finite() {
                node.key
            } else {
                global_lb
            };
            status = GlobalStatus::Optimal;
            final_lb = lb.min(incumbent_ub);
            exited = true;
            break 'search;
        }
        if nodes >= opts.max_nodes {
            status = if incumbent_ub.is_finite() && (incumbent_ub - global_lb) <= opts.abs_gap {
                GlobalStatus::Optimal
            } else {
                GlobalStatus::NodeLimit
            };
            final_lb = global_lb.min(incumbent_ub);
            exited = true;
            break 'search;
        }
        nodes += 1;

        // If the user stepped into this node, arm the interior-point hook and
        // hand it to the relaxation solve; otherwise the relaxation runs
        // un-debugged.
        let node_subsolve: Option<&mut dyn DebugHook> = if arm_next {
            match subsolve_hook.as_deref_mut() {
                Some(h) => {
                    h.arm();
                    Some(h)
                }
                None => None,
            }
        } else {
            None
        };

        let Some(b) = process_node(
            prob,
            opts,
            node.lo.clone(),
            node.hi.clone(),
            node.key,
            incumbent_ub,
            &g_lo,
            &g_hi,
            &qp_opts,
            opts.parallel,
            &make_backend,
            node_subsolve,
        ) else {
            if fire_cp!(
                TreeCheckpoint::NodePruned,
                &node.lo,
                &node.hi,
                f64::NAN,
                node.depth,
                None,
                Some(PruneReason::Infeasible)
            ) {
                status = stop_status(incumbent_ub);
                final_lb = global_lb.min(incumbent_ub);
                exited = true;
                break 'search;
            }
            continue;
        };

        // Learn the branched variable's pseudocost from the realized gain.
        if let Some(bi) = node.branch {
            pseudo.update(bi.var, bi.down, b.node_lb - bi.parent_lb, bi.frac);
        }

        if fire_cp!(
            TreeCheckpoint::RelaxationSolved,
            &b.lo,
            &b.hi,
            b.node_lb,
            node.depth,
            None,
            None
        ) {
            status = stop_status(incumbent_ub);
            final_lb = global_lb.min(incumbent_ub);
            exited = true;
            break 'search;
        }

        if let Some((x, obj)) = &b.incumbent {
            if *obj < incumbent_ub {
                incumbent_ub = *obj;
                incumbent_x = x.clone();
                if fire_cp!(
                    TreeCheckpoint::IncumbentFound,
                    &b.lo,
                    &b.hi,
                    b.node_lb,
                    node.depth,
                    None,
                    None
                ) {
                    status = stop_status(incumbent_ub);
                    final_lb = global_lb.min(incumbent_ub);
                    exited = true;
                    break 'search;
                }
            }
        }
        if incumbent_ub.is_finite() && gap_ok(b.node_lb, incumbent_ub, opts) {
            if fire_cp!(
                TreeCheckpoint::NodePruned,
                &b.lo,
                &b.hi,
                b.node_lb,
                node.depth,
                None,
                Some(PruneReason::BoundDominated)
            ) {
                status = stop_status(incumbent_ub);
                final_lb = global_lb.min(incumbent_ub);
                exited = true;
                break 'search;
            }
            continue;
        }

        let (widest_k, width) = widest(&b.lo, &b.hi);
        let lb_for_children = b.node_lb.max(node.key);
        if width <= opts.box_tol
            || (incumbent_ub.is_finite() && gap_ok(lb_for_children, incumbent_ub, opts))
        {
            if fire_cp!(
                TreeCheckpoint::NodePruned,
                &b.lo,
                &b.hi,
                b.node_lb,
                node.depth,
                None,
                Some(PruneReason::Leaf)
            ) {
                status = stop_status(incumbent_ub);
                final_lb = global_lb.min(incumbent_ub);
                exited = true;
                break 'search;
            }
            continue;
        }
        let k = match opts.branching {
            BranchRule::Widest => widest_k,
            BranchRule::MostViolation => {
                select_most_violation(&b, opts.box_tol, n).unwrap_or(widest_k)
            }
            BranchRule::Reliability => crate::branching::select_reliability(
                prob,
                &b.lo,
                &b.hi,
                &b.relax_pt,
                &b.branch_terms,
                &b.sol_x,
                b.node_lb,
                &mut pseudo,
                opts.box_tol,
                opts.multilinear,
                &qp_opts,
                &mut make_backend,
            )
            .unwrap_or(widest_k),
        };
        if fire_cp!(
            TreeCheckpoint::Branched,
            &b.lo,
            &b.hi,
            b.node_lb,
            node.depth,
            Some(k),
            None
        ) {
            status = stop_status(incumbent_ub);
            final_lb = global_lb.min(incumbent_ub);
            exited = true;
            break 'search;
        }
        for child in children(&b, k, lb_for_children, node.depth) {
            heap.push(child);
        }
        peak_frontier = peak_frontier.max(heap.len());
    }

    // Natural exit (frontier exhausted): everything was pruned or hit a leaf.
    if !exited {
        if incumbent_ub.is_finite() {
            status = GlobalStatus::Optimal;
            final_lb = incumbent_ub;
        } else {
            status = GlobalStatus::Infeasible;
            final_lb = global_lb;
        }
    }

    // Post-mortem tree checkpoint (the returned action is ignored).
    if hook.is_some() {
        let status_str = format!("{status:?}");
        let empty: [f64; 0] = [];
        let mut st = BnbDebugState {
            cp: TreeCheckpoint::Terminated,
            node_id,
            depth: 0,
            nodes: nodes as u64,
            frontier_len: heap.len(),
            lo: &empty,
            hi: &empty,
            node_lb: f64::NAN,
            global_lb: final_lb,
            incumbent: incumbent_ub.is_finite().then_some(incumbent_ub),
            incumbent_x: (!incumbent_x.is_empty()).then_some(incumbent_x.as_slice()),
            branch_var: None,
            prune_reason: None,
            status: Some(&status_str),
            arm: None,
        };
        let _ = fire_tree(&mut hook, &mut st);
    }

    GlobalSolution {
        status,
        x: incumbent_x,
        objective: incumbent_ub,
        lower_bound: final_lb,
        nodes,
        peak_frontier,
        peak_memory_bytes: peak_frontier * node_bytes,
    }
}

/// Shared state for the parallel node pool.
struct Shared {
    heap: BinaryHeap<Node>,
    incumbent_ub: f64,
    incumbent_x: Vec<f64>,
    global_lb: f64,
    nodes: usize,
    active: usize,
    stop: bool,
    node_limit: bool,
    peak_frontier: usize,
}

/// Parallel best-first driver: a pool of `threads` workers pulls nodes from a
/// shared frontier, processes them unsynchronized ([`process_node`]), and under
/// the lock updates the incumbent and pushes children. **Non-deterministic** in
/// which nodes are explored and in what order (so node counts vary run to run),
/// but the certified optimum value and gap are correct: a node is processed
/// only if its inherited bound cannot already meet the incumbent, and children
/// inherit a valid lower bound, so the incumbent on exhaustion is global.
// The `lock().unwrap()`s propagate mutex poisoning, which can only arise if a
// worker panics — a bug we want surfaced, not swallowed.
#[allow(clippy::unwrap_used)]
fn solve_parallel<F>(
    prob: &GlobalProblem,
    opts: &GlobalOptions,
    make_backend: &F,
    threads: usize,
) -> GlobalSolution
where
    F: Fn() -> Box<dyn SparseSymLinearSolverInterface> + Sync,
{
    use std::sync::{Condvar, Mutex};

    let n = prob.n_vars;
    let (g_lo, g_hi) = prob.constraint_bounds();
    let qp_opts = QpOptions::default();

    let mut root_heap = BinaryHeap::new();
    root_heap.push(Node {
        key: f64::NEG_INFINITY,
        lo: prob.x_lo.clone(),
        hi: prob.x_hi.clone(),
        branch: None,
        depth: 0,
    });
    let shared = Mutex::new(Shared {
        heap: root_heap,
        incumbent_ub: f64::INFINITY,
        incumbent_x: Vec::new(),
        global_lb: f64::NEG_INFINITY,
        nodes: 0,
        active: 0,
        stop: false,
        node_limit: false,
        peak_frontier: 1,
    });
    let cv = Condvar::new();

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                // --- acquire a node (or terminate) ---
                let (node, inc) = {
                    let mut s = shared.lock().unwrap();
                    loop {
                        if s.stop {
                            return;
                        }
                        if s.nodes >= opts.max_nodes {
                            s.node_limit = true;
                            s.stop = true;
                            let peek = s.heap.peek().map(|x| x.key).unwrap_or(s.incumbent_ub);
                            s.global_lb = peek.min(s.incumbent_ub);
                            cv.notify_all();
                            return;
                        }
                        if let Some(node) = s.heap.pop() {
                            // Skip a node whose own bound already meets the
                            // incumbent — it (and its children) cannot improve.
                            if s.incumbent_ub.is_finite() && gap_ok(node.key, s.incumbent_ub, opts)
                            {
                                continue;
                            }
                            s.nodes += 1;
                            s.active += 1;
                            let inc = s.incumbent_ub;
                            break (node, inc);
                        } else if s.active == 0 {
                            // Frontier empty and nobody is producing children.
                            if s.incumbent_ub.is_finite() {
                                s.global_lb = s.incumbent_ub;
                            }
                            s.stop = true;
                            cv.notify_all();
                            return;
                        } else {
                            s = cv.wait(s).unwrap();
                        }
                    }
                };

                // --- process unsynchronized ---
                let outcome = process_node(
                    prob,
                    opts,
                    node.lo.clone(),
                    node.hi.clone(),
                    node.key,
                    inc,
                    &g_lo,
                    &g_hi,
                    &qp_opts,
                    false, // OBBT serial inside a worker — nesting oversubscribes
                    make_backend,
                    None, // the parallel pool is not debuggable
                );

                // --- commit under the lock ---
                {
                    let mut s = shared.lock().unwrap();
                    if let Some(b) = outcome {
                        if let Some((x, obj)) = &b.incumbent {
                            if *obj < s.incumbent_ub {
                                s.incumbent_ub = *obj;
                                s.incumbent_x = x.clone();
                            }
                        }
                        let ub = s.incumbent_ub;
                        let leaf = ub.is_finite() && gap_ok(b.node_lb, ub, opts);
                        if !leaf {
                            let (widest_k, width) = widest(&b.lo, &b.hi);
                            let lb_for_children = b.node_lb.max(node.key);
                            let dominated = ub.is_finite() && gap_ok(lb_for_children, ub, opts);
                            if width > opts.box_tol && !dominated {
                                let k =
                                    select_most_violation(&b, opts.box_tol, n).unwrap_or(widest_k);
                                for child in children(&b, k, lb_for_children, node.depth) {
                                    s.heap.push(child);
                                }
                                s.peak_frontier = s.peak_frontier.max(s.heap.len());
                            }
                        }
                    }
                    s.active -= 1;
                    cv.notify_all();
                }
            });
        }
    });

    let s = shared.into_inner().unwrap();
    let status = if s.node_limit {
        if s.incumbent_ub.is_finite() && (s.incumbent_ub - s.global_lb) <= opts.abs_gap {
            GlobalStatus::Optimal
        } else {
            GlobalStatus::NodeLimit
        }
    } else if s.incumbent_ub.is_finite() {
        GlobalStatus::Optimal
    } else {
        GlobalStatus::Infeasible
    };
    let lower_bound = if s.incumbent_ub.is_finite() {
        s.global_lb.min(s.incumbent_ub)
    } else {
        s.global_lb
    };
    GlobalSolution {
        status,
        x: s.incumbent_x,
        objective: s.incumbent_ub,
        lower_bound,
        nodes: s.nodes,
        peak_frontier: s.peak_frontier,
        peak_memory_bytes: s.peak_frontier * estimate_node_bytes(n),
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
