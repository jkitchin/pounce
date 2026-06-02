//! The spatial branch-and-bound solver honors an attached `TreeDebugHook`:
//! it fires the tree checkpoints, exposes node/global search state, and the
//! hook does not change the result.

use pounce_common::debug::{
    DebugAction, PruneReason, TreeCheckpoint, TreeDebugHook, TreeDebugState,
};
use pounce_feral::FeralSolverInterface;
use pounce_global::{
    expr::var, solve_global, solve_global_debug, GlobalOptions, GlobalProblem, GlobalStatus,
};
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Records what the tree debugger sees, and resumes.
#[derive(Default)]
struct Recorder {
    checkpoints: Vec<TreeCheckpoint>,
    nodes_selected: u64,
    saw_incumbent: bool,
    saw_branch_var: bool,
    saw_prune: Vec<PruneReason>,
    max_depth: usize,
    last_box_dims: usize,
    terminal_status: Option<String>,
    finite_gap_seen: bool,
}

impl TreeDebugHook for Recorder {
    fn at_node(&mut self, st: &mut dyn TreeDebugState) -> DebugAction {
        let cp = st.checkpoint();
        self.checkpoints.push(cp);
        self.max_depth = self.max_depth.max(st.depth());
        if cp == TreeCheckpoint::NodeSelected {
            self.nodes_selected += 1;
            let (lo, hi) = st.node_box();
            assert_eq!(lo.len(), hi.len());
            self.last_box_dims = lo.len();
        }
        if cp == TreeCheckpoint::IncumbentFound {
            self.saw_incumbent = true;
            assert!(st.incumbent().is_some(), "incumbent must be set");
            assert!(st.incumbent_point().is_some(), "incumbent point");
        }
        if cp == TreeCheckpoint::Branched {
            self.saw_branch_var = true;
            assert!(st.branch_var().is_some(), "branch var at Branched");
        }
        if cp == TreeCheckpoint::NodePruned {
            if let Some(r) = st.prune_reason() {
                self.saw_prune.push(r);
            }
        }
        if st.gap().is_finite() {
            self.finite_gap_seen = true;
        }
        if cp == TreeCheckpoint::Terminated {
            self.terminal_status = st.status().map(str::to_owned);
        }
        DebugAction::Resume
    }
}

/// f(x) = x⁴ − 3x² on [−2, 2]; global min −9/4 at x = ±√(3/2). Nonconvex, so
/// the search actually branches.
fn quartic() -> GlobalProblem {
    let f = var(0).powi(4) - 3.0 * var(0).powi(2);
    GlobalProblem::new(vec![-2.0], vec![2.0], &f)
}

#[test]
fn bnb_fires_tree_checkpoints_and_exposes_state() {
    let prob = quartic();
    let opts = GlobalOptions::default();
    let mut rec = Recorder::default();
    let sol = solve_global_debug(&prob, &opts, &mut rec, backend);

    // Same optimum as the non-debug solve.
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!((sol.objective + 2.25).abs() < 1e-3, "obj={}", sol.objective);

    // The headline checkpoints fired.
    let fired = |c| rec.checkpoints.contains(&c);
    assert!(fired(TreeCheckpoint::NodeSelected), "NodeSelected");
    assert!(fired(TreeCheckpoint::RelaxationSolved), "RelaxationSolved");
    assert!(fired(TreeCheckpoint::Branched), "Branched (should split)");
    assert!(fired(TreeCheckpoint::Terminated), "Terminated");

    // State surfaced: several nodes, a 1-D box, a branch variable, an
    // incumbent, a finite gap at some point, and the terminal status.
    assert!(rec.nodes_selected >= 2, "should explore multiple nodes");
    assert_eq!(rec.last_box_dims, 1, "1-variable problem");
    assert!(rec.saw_branch_var, "a branching decision");
    assert!(rec.saw_incumbent, "an incumbent improvement");
    assert!(rec.finite_gap_seen, "a finite gap once an incumbent exists");
    assert_eq!(rec.terminal_status.as_deref(), Some("Optimal"));
}

#[test]
fn attaching_a_tree_hook_does_not_change_the_result() {
    let prob = quartic();
    let opts = GlobalOptions::default();

    let plain = solve_global(&prob, &opts, backend);
    let mut rec = Recorder::default();
    let debugged = solve_global_debug(&prob, &opts, &mut rec, backend);

    assert_eq!(plain.status, debugged.status);
    assert_eq!(plain.nodes, debugged.nodes, "node count must match");
    assert!(
        (plain.objective - debugged.objective).abs() < 1e-12,
        "obj differs"
    );
    assert!(
        (plain.lower_bound - debugged.lower_bound).abs() < 1e-9,
        "bound differs"
    );
}

/// Step-into composability: a tree hook that requests `into` at the first
/// node causes that node's relaxation to run under the interior-point
/// sub-solve hook — proving the two debug surfaces compose.
#[test]
fn step_into_runs_the_relaxation_under_the_ip_hook() {
    use pounce_common::debug::{Checkpoint, DebugHook, DebugState};

    // Tree hook: arm the sub-solve at the very first node, then resume.
    struct StepInto {
        armed_once: bool,
    }
    impl TreeDebugHook for StepInto {
        fn at_node(&mut self, st: &mut dyn TreeDebugState) -> DebugAction {
            if !self.armed_once && st.checkpoint() == TreeCheckpoint::NodeSelected {
                st.request_subsolve_debug();
                self.armed_once = true;
            }
            DebugAction::Resume
        }
    }

    // Interior-point sub-hook: records that it was armed and saw checkpoints.
    #[derive(Default)]
    struct SubRecorder {
        arm_calls: usize,
        checkpoints: usize,
        saw_iter_start: bool,
    }
    impl DebugHook for SubRecorder {
        fn arm(&mut self) {
            self.arm_calls += 1;
        }
        fn at_checkpoint(&mut self, st: &mut dyn DebugState) -> DebugAction {
            self.checkpoints += 1;
            if st.checkpoint() == Checkpoint::IterStart {
                self.saw_iter_start = true;
            }
            DebugAction::Resume
        }
    }

    let prob = quartic();
    let opts = GlobalOptions::default();
    let mut tree = StepInto { armed_once: false };
    let mut sub = SubRecorder::default();
    let sol = pounce_global::solve_global_debug_into(&prob, &opts, &mut tree, &mut sub, backend);

    // The solve still reaches the optimum.
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    // The sub-solve hook was armed exactly once and the relaxation's IPM
    // iterations flowed through it (so a REPL would have dropped in).
    assert_eq!(sub.arm_calls, 1, "armed once for the stepped-into node");
    assert!(sub.checkpoints > 0, "relaxation ran under the IP hook");
    assert!(sub.saw_iter_start, "saw an IPM iteration checkpoint");
}

/// A hook that stops at the first node halts the search early.
#[test]
fn tree_stop_action_halts_the_search() {
    struct StopAtFirstNode {
        seen: u64,
    }
    impl TreeDebugHook for StopAtFirstNode {
        fn at_node(&mut self, st: &mut dyn TreeDebugState) -> DebugAction {
            if st.checkpoint() == TreeCheckpoint::Terminated {
                return DebugAction::Resume;
            }
            self.seen += 1;
            DebugAction::Stop
        }
    }
    let prob = quartic();
    let opts = GlobalOptions::default();
    let mut hook = StopAtFirstNode { seen: 0 };
    let sol = solve_global_debug(&prob, &opts, &mut hook, backend);
    // Stopped almost immediately — far fewer nodes than a full solve.
    assert!(
        sol.nodes <= 1,
        "should stop at the first node, got {}",
        sol.nodes
    );
    assert_eq!(
        hook.seen, 1,
        "exactly one pre-terminal checkpoint before stop"
    );
}
