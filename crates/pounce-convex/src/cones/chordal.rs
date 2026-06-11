//! Chordal-graph analysis for sparse SDP decomposition (Phase H7 sparsity).
//!
//! The range-space chordal decomposition of a sparse PSD constraint
//! `smat(s) ⪰ 0` (with `s` supported on a pattern `E`) rewrites it as a sum
//! of clique-supported PSD blocks (Agler–Helton–McCullough–Rodman): for a
//! **chordal** `E` with maximal cliques `C₁…C_p`,
//!
//! ```text
//!   s ⪰ 0   ⟺   s = Σ_k Tᵀ_{C_k} S_k T_{C_k},   S_k ⪰ 0,
//! ```
//!
//! where `T_{C_k}` selects the rows/cols in clique `C_k`. This module does
//! the graph part: take the aggregate sparsity pattern, compute a **chordal
//! extension** by symbolic elimination (natural order + fill), and read off
//! the **maximal cliques** — the data the conic-program reformulation needs.
//!
//! The elimination is the textbook one (Vandenberghe & Andersen, *Chordal
//! Graphs and Semidefinite Optimization*, §4): eliminating vertex `v` makes
//! its still-present higher-ordered neighbors a clique (adding fill edges);
//! `clique(v) = {v} ∪ higher-neighbors(v)` in the filled graph, and the
//! maximal such sets are the maximal cliques of the chordal completion.

use std::collections::BTreeSet;

/// The chordal completion of a sparsity pattern: its maximal cliques (each a
/// sorted, ascending vertex list).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chordal {
    pub n: usize,
    /// Maximal cliques of the chordal completion, each sorted ascending.
    pub cliques: Vec<Vec<usize>>,
}

impl Chordal {
    /// Whether the completion is a single clique covering everything — i.e.
    /// the pattern is (effectively) dense, so decomposition buys nothing.
    pub fn is_single_block(&self) -> bool {
        self.cliques.len() == 1 && self.cliques[0].len() == self.n
    }
}

/// Compute the chordal completion (maximal cliques) of the undirected graph
/// on `0..n` with the given `edges` (off-diagonal pattern entries). The
/// natural elimination order `0,1,…,n−1` is used; for SDPs whose variables
/// are already laid out band-like this is a good order, and correctness does
/// not depend on it (any order yields a valid — if larger — chordal cover).
pub fn analyze(n: usize, edges: &[(usize, usize)]) -> Chordal {
    // Adjacency as sorted sets.
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for &(a, b) in edges {
        if a != b {
            adj[a].insert(b);
            adj[b].insert(a);
        }
    }

    // Symbolic elimination in natural order, accumulating fill. `clique(v)`
    // is `{v}` plus the neighbors of `v` that are eliminated later.
    let mut clique_sets: Vec<BTreeSet<usize>> = Vec::with_capacity(n);
    for v in 0..n {
        let higher: Vec<usize> = adj[v].iter().copied().filter(|&u| u > v).collect();
        // Make the higher neighbors a clique (fill edges).
        for i in 0..higher.len() {
            for j in (i + 1)..higher.len() {
                let (a, b) = (higher[i], higher[j]);
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }
        let mut c: BTreeSet<usize> = higher.into_iter().collect();
        c.insert(v);
        clique_sets.push(c);
    }

    // Keep only the maximal sets (drop any that is a subset of another).
    let mut maximal: Vec<Vec<usize>> = Vec::new();
    for (i, ci) in clique_sets.iter().enumerate() {
        let subsumed = clique_sets
            .iter()
            .enumerate()
            .any(|(j, cj)| j != i && ci.len() < cj.len() && ci.is_subset(cj));
        // Among equal-size duplicates keep the first occurrence only.
        let dup_earlier = clique_sets[..i].iter().any(|cj| cj == ci);
        if !subsumed && !dup_earlier {
            maximal.push(ci.iter().copied().collect());
        }
    }

    Chordal {
        n,
        cliques: maximal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut cliques: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        cliques.iter_mut().for_each(|c| c.sort_unstable());
        cliques.sort();
        cliques
    }

    #[test]
    fn path_graph_cliques_are_consecutive_pairs() {
        // 0–1–2–3 (already chordal): maximal cliques {0,1},{1,2},{2,3}.
        let c = analyze(4, &[(0, 1), (1, 2), (2, 3)]);
        assert!(!c.is_single_block());
        assert_eq!(sorted(c.cliques), vec![vec![0, 1], vec![1, 2], vec![2, 3]]);
    }

    #[test]
    fn two_disjoint_edges_give_two_cliques() {
        // 0–1 and 2–3: block-diagonal pattern → cliques {0,1},{2,3}.
        let c = analyze(4, &[(0, 1), (2, 3)]);
        assert_eq!(sorted(c.cliques), vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn dense_triangle_is_single_block() {
        // Fully connected 3-vertex graph → one clique {0,1,2}.
        let c = analyze(3, &[(0, 1), (0, 2), (1, 2)]);
        assert!(c.is_single_block());
        assert_eq!(sorted(c.cliques), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn cycle_gets_chordal_fill() {
        // 4-cycle 0–1–2–3–0 is NOT chordal; natural-order elimination fills
        // chord(s) so the completion's cliques cover it. Eliminating 0 (nbrs
        // 1,3) adds edge 1–3; the maximal cliques become {0,1,3} and {1,2,3}.
        let c = analyze(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let cl = sorted(c.cliques);
        // Every original edge must sit inside some clique.
        for &(a, b) in &[(0, 1), (1, 2), (2, 3), (3, 0)] {
            assert!(
                cl.iter().any(|c| c.contains(&a) && c.contains(&b)),
                "edge ({a},{b}) not covered by {cl:?}"
            );
        }
        // And it genuinely decomposed (no single 4-clique).
        assert!(cl.iter().all(|c| c.len() < 4));
    }

    #[test]
    fn isolated_vertices_are_singleton_cliques() {
        // No edges: each vertex is its own clique.
        let c = analyze(3, &[]);
        assert_eq!(sorted(c.cliques), vec![vec![0], vec![1], vec![2]]);
    }
}
