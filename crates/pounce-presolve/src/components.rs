//! Weakly-connected component extraction on the square-matched part
//! of a Dulmage-Mendelsohn partition.
//!
//! PR 3 of the auxiliary-presolve port (issue #53). Each component
//! becomes one independent candidate block for elimination in PR 8.
//! ripopt anchor: `src/auxiliary_preprocessing.rs:2416-2469`.

use crate::dulmage_mendelsohn::{DMPart, DulmageMendelsohnPartition};
use crate::incidence::EqualityIncidence;
use crate::matching::BipartiteMatching;

/// One connected component of the square sub-graph.
///
/// # Invariant
///
/// Every row in `eq_rows` must be matched to some column in `cols`
/// under the bipartite matching that produced this component.
/// Downstream code (notably `crate::btf::BlockTriangularForm`)
/// relies on `m.row_to_var[r]` being `Some` for every `r ∈
/// eq_rows`. Construct only via `SquareComponents::of_square_part`,
/// which preserves this invariant by filtering on `DMPart::Square`.
/// Hand-constructing a `SquareComponent` with unmatched rows
/// violates the invariant and causes `BlockTriangularForm` to
/// panic.
#[derive(Debug, Clone, Default)]
pub struct SquareComponent {
    /// Equality-row indices (positions in `inc.eq_row_inner_idx`).
    /// See struct-level invariant: must be matched.
    pub eq_rows: Vec<usize>,
    /// Variable indices. Same length as `eq_rows`.
    pub cols: Vec<usize>,
}

/// Decomposition of the square sub-graph into connected components.
#[derive(Debug, Clone, Default)]
pub struct SquareComponents {
    /// One entry per component, sorted by smallest contained
    /// equality-row index for determinism.
    pub components: Vec<SquareComponent>,
}

/// Disjoint-set / union-find over `(rows ∪ cols)` with rows numbered
/// `0..n_rows` and cols offset by `n_rows`.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression.
        let mut cur = x;
        while self.parent[cur] != root {
            let nxt = self.parent[cur];
            self.parent[cur] = root;
            cur = nxt;
        }
        root
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

impl SquareComponents {
    /// Decompose the square sub-graph of `part` into connected
    /// components. Edges considered are exactly those of `inc` whose
    /// row AND column are both in `DMPart::Square`.
    ///
    /// # Example
    ///
    /// ```
    /// use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
    /// use pounce_presolve::matching::hopcroft_karp;
    /// use pounce_presolve::dulmage_mendelsohn::DulmageMendelsohnPartition;
    /// use pounce_presolve::components::SquareComponents;
    ///
    /// // 4×4 with a 2-block and two singletons.
    /// let p = ProbeView {
    ///     n_vars: 4,
    ///     m_rows: 4,
    ///     jac_irow: &[0, 0, 1, 1, 2, 3],
    ///     jac_jcol: &[0, 1, 0, 1, 2, 3],
    ///     jac_values: None,
    ///     g_l: &[0.0; 4],
    ///     g_u: &[0.0; 4],
    ///     linearity: None,
    ///     one_based: false,
    ///     eq_tol: 1e-12,
    ///     excluded_vars: None,
    ///     excluded_rows: None,
    /// };
    /// let inc = EqualityIncidence::from_probe(&p);
    /// let m = hopcroft_karp(&inc);
    /// let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
    /// let c = SquareComponents::of_square_part(&inc, &m, &dm);
    /// assert_eq!(c.components.len(), 3);
    /// ```
    pub fn of_square_part(
        inc: &EqualityIncidence,
        _m: &BipartiteMatching,
        part: &DulmageMendelsohnPartition,
    ) -> Self {
        let n_rows = inc.n_eq_rows();
        let n_vars = inc.n_vars;

        // Union-find IDs: rows are 0..n_rows, cols are n_rows..n_rows+n_vars.
        let mut uf = UnionFind::new(n_rows + n_vars);
        let col_off = n_rows;

        for r in 0..n_rows {
            if part.row_part[r] != DMPart::Square {
                continue;
            }
            for &v in inc.neighbors(r) {
                if part.col_part[v] != DMPart::Square {
                    continue;
                }
                uf.union(r, col_off + v);
            }
        }

        // Bucket members by component root.
        use std::collections::BTreeMap;
        let mut buckets: BTreeMap<usize, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
        for r in 0..n_rows {
            if part.row_part[r] != DMPart::Square {
                continue;
            }
            let root = uf.find(r);
            buckets.entry(root).or_default().0.push(r);
        }
        for v in 0..n_vars {
            if part.col_part[v] != DMPart::Square {
                continue;
            }
            let root = uf.find(col_off + v);
            buckets.entry(root).or_default().1.push(v);
        }

        // Sort components by the smallest contained equality-row
        // index for deterministic output.
        let mut comps: Vec<SquareComponent> = buckets
            .into_values()
            .map(|(mut rows, mut cols)| {
                rows.sort_unstable();
                cols.sort_unstable();
                SquareComponent {
                    eq_rows: rows,
                    cols,
                }
            })
            .filter(|c| !c.eq_rows.is_empty() || !c.cols.is_empty())
            .collect();
        comps.sort_by_key(|c| c.eq_rows.first().copied().unwrap_or(usize::MAX));

        Self { components: comps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn decompose(n_vars: usize, n_rows: usize, edges: &[(usize, usize)]) -> SquareComponents {
        let inc = eq_inc(n_vars, n_rows, edges);
        let m = hopcroft_karp(&inc);
        let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
        SquareComponents::of_square_part(&inc, &m, &dm)
    }

    #[test]
    fn components_empty_square() {
        let c = decompose(0, 0, &[]);
        assert!(c.components.is_empty());
    }

    #[test]
    fn components_disjoint_3x3() {
        let c = decompose(3, 3, &[(0, 0), (1, 1), (2, 2)]);
        assert_eq!(c.components.len(), 3);
        for (i, comp) in c.components.iter().enumerate() {
            assert_eq!(comp.eq_rows, vec![i]);
            assert_eq!(comp.cols, vec![i]);
        }
    }

    #[test]
    fn components_two_blocks_5x5() {
        // Block A: rows {0,1,2} share cols {0,1,2}.
        // Block B: rows {3,4} share cols {3,4}.
        let edges = [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 2),
            (3, 3),
            (3, 4),
            (4, 4),
        ];
        let c = decompose(5, 5, &edges);
        assert_eq!(c.components.len(), 2);
        assert_eq!(c.components[0].eq_rows, vec![0, 1, 2]);
        assert_eq!(c.components[0].cols, vec![0, 1, 2]);
        assert_eq!(c.components[1].eq_rows, vec![3, 4]);
        assert_eq!(c.components[1].cols, vec![3, 4]);
    }

    #[test]
    fn components_star_inside_square() {
        // Row 0 acts as a hub connecting cols 0, 1, 2. Rows 1, 2
        // pick up cols 1, 2 respectively to keep things square.
        let c = decompose(3, 3, &[(0, 0), (0, 1), (0, 2), (1, 1), (2, 2)]);
        assert_eq!(c.components.len(), 1);
        let only = &c.components[0];
        assert_eq!(only.eq_rows, vec![0, 1, 2]);
        assert_eq!(only.cols, vec![0, 1, 2]);
    }

    #[test]
    fn components_order_is_deterministic() {
        // Build two disjoint 2-blocks but order edges so the second
        // block's rows come first in the edge list.
        let c = decompose(
            4,
            4,
            &[
                (2, 2),
                (2, 3),
                (3, 2),
                (3, 3),
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ],
        );
        assert_eq!(c.components.len(), 2);
        // Lowest row index in component[0] must be smaller.
        assert_eq!(c.components[0].eq_rows.first(), Some(&0));
        assert_eq!(c.components[1].eq_rows.first(), Some(&2));
    }

    #[test]
    fn components_skip_overdetermined_and_underdetermined() {
        // 3 rows × 2 cols, fully connected → all over. Square is
        // empty, so zero components.
        let c = decompose(2, 3, &[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]);
        assert!(c.components.is_empty());
    }

    /// Independent reference: BFS-based connected components on the
    /// square sub-graph. Returns one (sorted_rows, sorted_cols) pair
    /// per component, components ordered by smallest row index.
    fn reference_components(
        inc: &EqualityIncidence,
        dm: &DulmageMendelsohnPartition,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        let n_rows = inc.n_eq_rows();
        let n_vars = inc.n_vars;

        // Reverse adjacency col → rows so BFS can step both ways.
        let mut col_to_rows: Vec<Vec<usize>> = vec![Vec::new(); n_vars];
        for r in 0..n_rows {
            for &v in inc.neighbors(r) {
                col_to_rows[v].push(r);
            }
        }

        let mut row_seen = vec![false; n_rows];
        let mut col_seen = vec![false; n_vars];
        let mut comps: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        for start in 0..n_rows {
            if row_seen[start] || dm.row_part[start] != DMPart::Square {
                continue;
            }
            let mut comp_rows = Vec::new();
            let mut comp_cols = Vec::new();
            let mut row_q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
            let mut col_q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
            row_seen[start] = true;
            row_q.push_back(start);
            comp_rows.push(start);
            while !row_q.is_empty() || !col_q.is_empty() {
                while let Some(r) = row_q.pop_front() {
                    for &v in inc.neighbors(r) {
                        if dm.col_part[v] != DMPart::Square || col_seen[v] {
                            continue;
                        }
                        col_seen[v] = true;
                        comp_cols.push(v);
                        col_q.push_back(v);
                    }
                }
                while let Some(v) = col_q.pop_front() {
                    for &r2 in &col_to_rows[v] {
                        if dm.row_part[r2] != DMPart::Square || row_seen[r2] {
                            continue;
                        }
                        row_seen[r2] = true;
                        comp_rows.push(r2);
                        row_q.push_back(r2);
                    }
                }
            }
            comp_rows.sort_unstable();
            comp_cols.sort_unstable();
            comps.push((comp_rows, comp_cols));
        }
        // Catch square cols that aren't reachable from any square
        // row (degenerate — they'd be isolated cols, which means the
        // square would be unbalanced; the DM invariant test catches
        // that elsewhere, but we still want a defined ordering).
        for v in 0..n_vars {
            if dm.col_part[v] == DMPart::Square && !col_seen[v] {
                comps.push((Vec::new(), vec![v]));
            }
        }
        comps.sort_by_key(|(rows, cols)| {
            rows.first()
                .copied()
                .unwrap_or_else(|| cols.first().copied().unwrap_or(usize::MAX))
        });
        comps
    }

    /// Fuzz against the BFS reference impl on 30 random small graphs.
    /// Asserts:
    ///   - same component count
    ///   - same (rows, cols) per component
    ///   - sums add up to |square_rows| / |square_cols|
    ///   - no edges cross component boundaries
    #[test]
    fn components_fuzz_invariants() {
        let mut state: u64 = 0xc0de_face_beef_b00b;
        let mut next = || -> u64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 32
        };

        for trial in 0..30 {
            let n_rows = 1 + (next() % 4) as usize;
            let n_vars = 1 + (next() % 4) as usize;
            let max_edges = (n_rows * n_vars).min(8);
            let n_edges = (next() % (max_edges as u64 + 1)) as usize;

            let mut edge_set = std::collections::BTreeSet::<(usize, usize)>::new();
            let mut draws = 0usize;
            while edge_set.len() < n_edges {
                let r = (next() % n_rows as u64) as usize;
                let v = (next() % n_vars as u64) as usize;
                edge_set.insert((r, v));
                draws += 1;
                assert!(draws < 10_000);
            }
            let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();

            let inc = eq_inc(n_vars, n_rows, &edges);
            let m = hopcroft_karp(&inc);
            let dm = DulmageMendelsohnPartition::from_matching(&inc, &m);
            let ours = SquareComponents::of_square_part(&inc, &m, &dm);
            let theirs = reference_components(&inc, &dm);

            // Compare against reference.
            assert_eq!(
                ours.components.len(),
                theirs.len(),
                "trial {trial}: component count differs (ours={}, ref={})",
                ours.components.len(),
                theirs.len()
            );
            for (i, (ours_c, theirs_c)) in ours.components.iter().zip(theirs.iter()).enumerate() {
                assert_eq!(
                    ours_c.eq_rows, theirs_c.0,
                    "trial {trial} comp {i}: rows differ"
                );
                assert_eq!(
                    ours_c.cols, theirs_c.1,
                    "trial {trial} comp {i}: cols differ"
                );
            }

            // Self-consistency: sum of sizes == |square|.
            let sum_rows: usize = ours.components.iter().map(|c| c.eq_rows.len()).sum();
            let sum_cols: usize = ours.components.iter().map(|c| c.cols.len()).sum();
            assert_eq!(sum_rows, dm.square_rows.len(), "trial {trial}");
            assert_eq!(sum_cols, dm.square_cols.len(), "trial {trial}");

            // Separation: no edge crosses component boundaries
            // (restricted to the square sub-graph).
            let mut col_to_comp: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for (i, c) in ours.components.iter().enumerate() {
                for &v in &c.cols {
                    col_to_comp.insert(v, i);
                }
            }
            for (i, c) in ours.components.iter().enumerate() {
                for &r in &c.eq_rows {
                    for &v in inc.neighbors(r) {
                        if dm.col_part[v] != DMPart::Square {
                            continue;
                        }
                        let owner = col_to_comp.get(&v).copied().unwrap_or(usize::MAX);
                        assert_eq!(
                            owner, i,
                            "trial {trial}: edge ({r},{v}) crosses comp {i}→{owner}"
                        );
                    }
                }
            }
        }
    }
}
