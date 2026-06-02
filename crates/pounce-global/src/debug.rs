//! Debugger glue for the spatial branch-and-bound global optimizer.
//!
//! Branch-and-bound is a tree search, not an iteration loop, so it exposes
//! the [`TreeDebugState`] surface (node box, global bounds, gap, frontier,
//! prune reason, branching choice) rather than the interior-point
//! [`DebugState`]. [`BnbDebugState`] adapts the serial driver's live search
//! state to that surface at each tree checkpoint.

use pounce_common::debug::{
    DebugAction, PruneReason, TreeCheckpoint, TreeDebugHook, TreeDebugState,
};
use pounce_common::types::Number;

/// A read-only view of the branch-and-bound search at one [`TreeCheckpoint`].
///
/// Borrows the current node's box (`lo`/`hi`) and the incumbent point; cheap
/// to build and dropped before the search advances.
pub(crate) struct BnbDebugState<'a> {
    pub cp: TreeCheckpoint,
    pub node_id: u64,
    pub depth: usize,
    pub nodes: u64,
    pub frontier_len: usize,
    pub lo: &'a [f64],
    pub hi: &'a [f64],
    pub node_lb: f64,
    pub global_lb: f64,
    pub incumbent: Option<f64>,
    pub incumbent_x: Option<&'a [f64]>,
    pub branch_var: Option<usize>,
    pub prune_reason: Option<PruneReason>,
    pub status: Option<&'a str>,
    /// Flag the loop reads after `at_node` to decide whether to run this
    /// node's relaxation under the interior-point debugger. `request_subsolve_debug`
    /// sets it; only wired (and only meaningful) at `NodeSelected`.
    pub arm: Option<&'a mut bool>,
}

impl TreeDebugState for BnbDebugState<'_> {
    fn checkpoint(&self) -> TreeCheckpoint {
        self.cp
    }
    fn node_id(&self) -> u64 {
        self.node_id
    }
    fn depth(&self) -> usize {
        self.depth
    }
    fn nodes_processed(&self) -> u64 {
        self.nodes
    }
    fn frontier_len(&self) -> usize {
        self.frontier_len
    }
    fn node_box(&self) -> (Vec<Number>, Vec<Number>) {
        (self.lo.to_vec(), self.hi.to_vec())
    }
    fn node_lb(&self) -> Number {
        self.node_lb
    }
    fn global_lb(&self) -> Number {
        self.global_lb
    }
    fn incumbent(&self) -> Option<Number> {
        self.incumbent
    }
    fn branch_var(&self) -> Option<usize> {
        self.branch_var
    }
    fn prune_reason(&self) -> Option<PruneReason> {
        self.prune_reason
    }
    fn incumbent_point(&self) -> Option<Vec<Number>> {
        self.incumbent_x.map(|x| x.to_vec())
    }
    fn status(&self) -> Option<&str> {
        self.status
    }
    fn request_subsolve_debug(&mut self) {
        if let Some(f) = self.arm.as_deref_mut() {
            *f = true;
        }
    }
}

/// Fire a tree checkpoint at `state` if a hook is attached. A no-op (and
/// always [`DebugAction::Resume`]) when `hook` is `None`.
pub(crate) fn fire_tree(
    hook: &mut Option<&mut dyn TreeDebugHook>,
    state: &mut dyn TreeDebugState,
) -> DebugAction {
    match hook.as_mut() {
        Some(h) => h.at_node(state),
        None => DebugAction::Resume,
    }
}
