//! Tarjan SCC + topological order → block-triangular form on each
//! connected component of the square-matched part.
//!
//! PR 4 of the auxiliary-presolve port (issue #53). Given a square
//! [`crate::components::SquareComponent`] (with a perfect matching
//! restricted to it), build the **dependency DAG** on its matched
//! pairs and decompose it into strongly-connected components in
//! reverse topological order. Each SCC becomes one BTF block:
//!
//! - size-1 blocks: variable can be solved on its own once its
//!   prerequisites are known;
//! - size-N blocks (N>1): an irreducible cyclic dependency — those
//!   N rows have to be solved simultaneously as a small system.
//!
//! The blocks are returned in **elimination order** — `blocks[0]`
//! has no in-component dependencies, `blocks[k]` may use values
//! produced by `blocks[0..k]`.
//!
//! ripopt anchor: `src/auxiliary_preprocessing.rs:2473-2552`.

use crate::components::SquareComponent;
use crate::incidence::EqualityIncidence;
use crate::matching::BipartiteMatching;

/// One block of the block-triangular form. Rows and cols are
/// returned sorted ascending for determinism.
#[derive(Debug, Clone, Default)]
pub struct BlockTriangularBlock {
    pub eq_rows: Vec<usize>,
    pub cols: Vec<usize>,
}

/// Block-triangular decomposition of one [`SquareComponent`].
#[derive(Debug, Clone, Default)]
pub struct BlockTriangularForm {
    /// Blocks in elimination order: index 0 is solved first.
    pub blocks: Vec<BlockTriangularBlock>,
}

// Each `.expect("matched")` below is invariant-protected: rows
// inside a `SquareComponent` are guaranteed by PR 3 to be matched.
#[allow(clippy::expect_used)]
impl BlockTriangularForm {
    /// Decompose `component` into blocks via Tarjan SCC on the
    /// dependency DAG induced by `inc` and the matching `m`.
    ///
    /// # Example
    ///
    /// ```
    /// use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
    /// use pounce_presolve::matching::hopcroft_karp;
    /// use pounce_presolve::dulmage_mendelsohn::DulmageMendelsohnPartition;
    /// use pounce_presolve::components::SquareComponents;
    /// use pounce_presolve::btf::BlockTriangularForm;
    ///
    /// // 2x2 lower-triangular: row 0 needs col 0; row 1 needs
    /// // cols 0 and 1. Matching: 0↔0, 1↔1. Two singleton blocks.
    /// let p = ProbeView {
    ///     n_vars: 2,
    ///     m_rows: 2,
    ///     jac_irow: &[0, 1, 1],
    ///     jac_jcol: &[0, 0, 1],
    ///     jac_values: None,
    ///     g_l: &[0.0; 2],
    ///     g_u: &[0.0; 2],
    ///     linearity: None,
    ///     one_based: false,
    ///     eq_tol: 1e-12,
    /// };
    /// let inc = EqualityIncidence::from_probe(&p);
    /// let m = hopcroft_karp(&inc);
    /// let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
    /// let comps = SquareComponents::of_square_part(&inc, &m, &dm);
    /// let btf = BlockTriangularForm::of_component(&inc, &m, &comps.components[0]);
    /// assert_eq!(btf.blocks.len(), 2);
    /// assert_eq!(btf.blocks[0].eq_rows, vec![0]);
    /// assert_eq!(btf.blocks[1].eq_rows, vec![1]);
    /// ```
    pub fn of_component(
        inc: &EqualityIncidence,
        m: &BipartiteMatching,
        component: &SquareComponent,
    ) -> Self {
        let n = component.eq_rows.len();
        debug_assert_eq!(
            n,
            component.cols.len(),
            "square component shapes must match"
        );
        if n == 0 {
            return Self::default();
        }

        // node[i] = matched pair (component.eq_rows[i], matched_col).
        // We map col → block-node index.
        let mut col_to_node: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::with_capacity(n);
        for (i, &r) in component.eq_rows.iter().enumerate() {
            let c = m.row_to_var[r].expect("square row must be matched");
            col_to_node.insert(c, i);
        }

        // Build dependency adjacency: edge i → j when row at node i
        // touches a non-matched col owned by node j.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &r) in component.eq_rows.iter().enumerate() {
            let own_col = m.row_to_var[r].expect("matched");
            for &c in inc.neighbors(r) {
                if c == own_col {
                    continue;
                }
                if let Some(&j) = col_to_node.get(&c) {
                    if j != i {
                        adj[i].push(j);
                    }
                }
            }
            adj[i].sort_unstable();
            adj[i].dedup();
        }

        // Tarjan SCC: when our edges have "depends-on" semantics
        // (i → j means i needs j's value), Tarjan finishes leaves of
        // the dependency DAG first, so its natural emission order is
        // already the elimination order — sinks (no remaining deps)
        // come out first.
        let sccs = tarjan_scc(&adj);

        let blocks: Vec<BlockTriangularBlock> = sccs
            .into_iter()
            .map(|scc| {
                let mut eq_rows: Vec<usize> = scc.iter().map(|&i| component.eq_rows[i]).collect();
                let mut cols: Vec<usize> = scc
                    .iter()
                    .map(|&i| {
                        let r = component.eq_rows[i];
                        m.row_to_var[r].expect("matched")
                    })
                    .collect();
                eq_rows.sort_unstable();
                cols.sort_unstable();
                BlockTriangularBlock { eq_rows, cols }
            })
            .collect();
        debug_assert!(blocks.len() <= n);
        Self { blocks }
    }
}

/// Iterative Tarjan SCC. Returns SCCs as lists of node indices.
/// With edges read as "depends-on", Tarjan finishes leaves (no
/// outgoing deps) first, so the output is already in elimination
/// order.
#[allow(clippy::expect_used)]
fn tarjan_scc(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    const UNVISITED: usize = usize::MAX;
    let mut index = vec![UNVISITED; n];
    let mut lowlink = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut next_index: usize = 0;
    let mut sccs: Vec<Vec<usize>> = Vec::new();

    // Explicit DFS stack: each frame is (node, neighbor-iter-pos).
    for v0 in 0..n {
        if index[v0] != UNVISITED {
            continue;
        }
        let mut call_stack: Vec<(usize, usize)> = Vec::new();
        index[v0] = next_index;
        lowlink[v0] = next_index;
        next_index += 1;
        stack.push(v0);
        on_stack[v0] = true;
        call_stack.push((v0, 0));

        while let Some(&(v, ref_pos)) = call_stack.last() {
            let pos = ref_pos;
            if pos < adj[v].len() {
                let w = adj[v][pos];
                // Advance the iterator for this frame before
                // possibly recursing.
                call_stack.last_mut().expect("non-empty").1 = pos + 1;
                if index[w] == UNVISITED {
                    index[w] = next_index;
                    lowlink[w] = next_index;
                    next_index += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    call_stack.push((w, 0));
                } else if on_stack[w] {
                    lowlink[v] = lowlink[v].min(index[w]);
                }
            } else {
                // All neighbours processed: pop and update parent.
                if lowlink[v] == index[v] {
                    // v is a root of an SCC; pop the stack down to v.
                    let mut scc = Vec::new();
                    while let Some(w) = stack.pop() {
                        on_stack[w] = false;
                        scc.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(scc);
                }
                call_stack.pop();
                if let Some(&mut (parent, _)) = call_stack.last_mut() {
                    lowlink[parent] = lowlink[parent].min(lowlink[v]);
                }
            }
        }
    }

    sccs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::SquareComponents;
    use crate::dulmage_mendelsohn::DulmageMendelsohnPartition;
    use crate::matching::hopcroft_karp;

    fn eq_inc(n_vars: usize, n_rows: usize, edges: &[(usize, usize)]) -> EqualityIncidence {
        let mut per_row: Vec<Vec<usize>> = vec![Vec::new(); n_rows];
        for &(r, v) in edges {
            per_row[r].push(v);
        }
        let mut adj_ptr = Vec::with_capacity(n_rows + 1);
        let mut vars = Vec::new();
        adj_ptr.push(0);
        for row in per_row.iter_mut() {
            row.sort_unstable();
            row.dedup();
            vars.extend_from_slice(row);
            adj_ptr.push(vars.len());
        }
        EqualityIncidence {
            n_vars,
            eq_row_inner_idx: (0..n_rows).collect(),
            adj_ptr,
            vars,
        }
    }

    fn btf_of(n_vars: usize, n_rows: usize, edges: &[(usize, usize)]) -> Vec<BlockTriangularForm> {
        let inc = eq_inc(n_vars, n_rows, edges);
        let m = hopcroft_karp(&inc);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        let comps = SquareComponents::of_square_part(&inc, &m, &dm);
        comps
            .components
            .iter()
            .map(|c| BlockTriangularForm::of_component(&inc, &m, c))
            .collect()
    }

    #[test]
    fn btf_singleton_block() {
        // 1×1 — one diagonal entry, one trivial block.
        let btfs = btf_of(1, 1, &[(0, 0)]);
        assert_eq!(btfs.len(), 1);
        assert_eq!(btfs[0].blocks.len(), 1);
        assert_eq!(btfs[0].blocks[0].eq_rows, vec![0]);
        assert_eq!(btfs[0].blocks[0].cols, vec![0]);
    }

    #[test]
    fn btf_chain_lower_triangular() {
        // Row 0 ↔ col 0 (no off-diag).
        // Row 1 ↔ col 1, uses col 0.
        // Row 2 ↔ col 2, uses cols 0 and 1.
        // Matching: 0↔0, 1↔1, 2↔2. Three size-1 blocks in order
        // [0], [1], [2].
        let edges = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)];
        let btfs = btf_of(3, 3, &edges);
        assert_eq!(btfs.len(), 1);
        let btf = &btfs[0];
        assert_eq!(btf.blocks.len(), 3);
        assert_eq!(btf.blocks[0].eq_rows, vec![0]);
        assert_eq!(btf.blocks[1].eq_rows, vec![1]);
        assert_eq!(btf.blocks[2].eq_rows, vec![2]);
    }

    #[test]
    fn btf_full_cycle_single_block() {
        // Row 0 ↔ col 0, uses col 1.
        // Row 1 ↔ col 1, uses col 2.
        // Row 2 ↔ col 2, uses col 0.
        // Dependency graph forms a 3-cycle → one SCC of size 3.
        let edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)];
        let btfs = btf_of(3, 3, &edges);
        assert_eq!(btfs.len(), 1);
        let btf = &btfs[0];
        assert_eq!(btf.blocks.len(), 1);
        assert_eq!(btf.blocks[0].eq_rows, vec![0, 1, 2]);
        assert_eq!(btf.blocks[0].cols, vec![0, 1, 2]);
    }

    #[test]
    fn btf_two_subcycles_chained() {
        // Two 2-cycles, the second depending on the first:
        //   {0,1} form a 2-cycle on cols {0,1}.
        //   {2,3} form a 2-cycle on cols {2,3}.
        //   Row 2 also uses col 0 → dependency on first block.
        // Matching: 0↔0, 1↔1, 2↔2, 3↔3.
        let edges = [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 2),
            (2, 0), // bridge: pair {2,3} depends on pair {0,1}
        ];
        let btfs = btf_of(4, 4, &edges);
        assert_eq!(btfs.len(), 1);
        let btf = &btfs[0];
        assert_eq!(btf.blocks.len(), 2, "two size-2 SCCs");
        assert_eq!(btf.blocks[0].eq_rows, vec![0, 1]);
        assert_eq!(btf.blocks[1].eq_rows, vec![2, 3]);
    }

    #[test]
    fn btf_empty_component() {
        let inc = eq_inc(0, 0, &[]);
        let m = hopcroft_karp(&inc);
        let comp = SquareComponent {
            eq_rows: vec![],
            cols: vec![],
        };
        let btf = BlockTriangularForm::of_component(&inc, &m, &comp);
        assert!(btf.blocks.is_empty());
    }

    #[test]
    fn btf_elimination_order_respects_dependencies() {
        // 5×5 single component:
        //   row 0 ↔ col 0 (no off-diag)
        //   row 1 ↔ col 1, uses col 0
        //   row 2 ↔ col 2, uses col 1
        //   rows 3, 4 form a 2-cycle on cols 3, 4, and one of them
        //   bridges into col 2 so the whole thing is one component.
        let edges = [
            (0, 0),
            (1, 1),
            (1, 0),
            (2, 2),
            (2, 1),
            (3, 3),
            (3, 4),
            (3, 2), // bridge into the {0,1,2} chain
            (4, 4),
            (4, 3),
        ];
        let btfs = btf_of(5, 5, &edges);
        assert_eq!(btfs.len(), 1);
        let btf = &btfs[0];
        // Build a map from variable to the block index that owns it.
        let mut col_block = std::collections::HashMap::new();
        for (b_idx, block) in btf.blocks.iter().enumerate() {
            for &c in &block.cols {
                col_block.insert(c, b_idx);
            }
        }
        // For every block k and every row r in it, every column it
        // touches outside its own block must belong to a strictly
        // earlier block.
        let inc = eq_inc(5, 5, &edges);
        for (k, block) in btf.blocks.iter().enumerate() {
            for &r in &block.eq_rows {
                for &c in inc.neighbors(r) {
                    if block.cols.contains(&c) {
                        continue;
                    }
                    let owner = *col_block.get(&c).expect("col owned by some block");
                    assert!(owner < k, "block {k} uses col {c} from later block {owner}");
                }
            }
        }
        // The bridging row 3 puts the {3,4} 2-cycle strictly after
        // the [0],[1],[2] chain. Expected order: 3 singletons + one
        // size-2 cycle = 4 blocks.
        assert_eq!(btf.blocks.len(), 4);
        assert_eq!(btf.blocks[0].eq_rows, vec![0]);
        assert_eq!(btf.blocks[1].eq_rows, vec![1]);
        assert_eq!(btf.blocks[2].eq_rows, vec![2]);
        assert_eq!(btf.blocks[3].eq_rows, vec![3, 4]);
    }

    #[test]
    fn btf_self_loop_singleton() {
        // 1×1 where the only edge is (0, 0) — node 0 has no
        // outgoing edges in the dependency graph. Single block.
        let btfs = btf_of(1, 1, &[(0, 0)]);
        assert_eq!(btfs.len(), 1);
        assert_eq!(btfs[0].blocks.len(), 1);
    }

    #[test]
    fn btf_three_disjoint_singletons() {
        let edges = [(0, 0), (1, 1), (2, 2)];
        let btfs = btf_of(3, 3, &edges);
        // Three components, each one size-1 BTF block.
        assert_eq!(btfs.len(), 3);
        for btf in &btfs {
            assert_eq!(btf.blocks.len(), 1);
        }
    }
}
