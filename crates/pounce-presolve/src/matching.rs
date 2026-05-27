//! Hopcroft-Karp bipartite matching on an [`crate::incidence::EqualityIncidence`].
//!
//! PR 2 of the auxiliary-presolve port (issue #53). ripopt anchor:
//! `src/auxiliary_preprocessing.rs:2280-2318`.
//!
//! Hopcroft-Karp finds a maximum bipartite matching in
//! `O(E · sqrt(V))` time by alternating BFS layering with DFS
//! augmentation along shortest augmenting paths.

use std::collections::VecDeque;

use crate::incidence::EqualityIncidence;

const NIL: usize = usize::MAX;
const INF: usize = usize::MAX;

/// Maximum-cardinality bipartite matching between equality rows
/// (left) and variables (right).
#[derive(Debug, Clone, Default)]
pub struct BipartiteMatching {
    /// `row_to_var[k] = Some(j)` when equality row `k` is matched to
    /// variable `j`, else `None`. Length = `n_eq_rows`.
    pub row_to_var: Vec<Option<usize>>,
    /// Inverse mapping; length = `n_vars`.
    pub var_to_row: Vec<Option<usize>>,
    /// Cardinality of the matching.
    pub size: usize,
}

/// Run Hopcroft-Karp on `inc` and return the maximum matching.
///
/// # Example
///
/// ```
/// use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
/// use pounce_presolve::matching::hopcroft_karp;
///
/// // 2 equality rows × 2 vars, each row touching one distinct var.
/// let p = ProbeView {
///     n_vars: 2,
///     m_rows: 2,
///     jac_irow: &[0, 1],
///     jac_jcol: &[0, 1],
///     jac_values: None,
///     g_l: &[0.0, 0.0],
///     g_u: &[0.0, 0.0],
///     linearity: None,
///     one_based: false,
///     eq_tol: 1e-12,
///     excluded_vars: None,
///     excluded_rows: None,
/// };
/// let inc = EqualityIncidence::from_probe(&p);
/// let m = hopcroft_karp(&inc);
/// assert_eq!(m.size, 2);
/// ```
pub fn hopcroft_karp(inc: &EqualityIncidence) -> BipartiteMatching {
    let n_rows = inc.n_eq_rows();
    let n_vars = inc.n_vars;

    // `pair_u[r]` is the var matched to row r, NIL when unmatched.
    let mut pair_u: Vec<usize> = vec![NIL; n_rows];
    let mut pair_v: Vec<usize> = vec![NIL; n_vars];
    // BFS distance from row r, when participating in the layered graph.
    let mut dist: Vec<usize> = vec![INF; n_rows];

    let mut size = 0;
    // Each successful BFS round augments at least one path, so the
    // outer loop terminates in O(min(n_rows, n_vars)) rounds. Cap
    // defensively at 2x to catch any subtle bug as a panic in tests
    // rather than as a runaway loop in production.
    let max_rounds = (n_rows.min(n_vars) + 1) * 2;
    let mut round = 0;
    while bfs(inc, &pair_u, &pair_v, &mut dist) {
        let mut augmented = false;
        for r in 0..n_rows {
            if pair_u[r] == NIL && dfs(inc, r, &mut pair_u, &mut pair_v, &mut dist) {
                size += 1;
                augmented = true;
            }
        }
        if !augmented {
            // BFS claimed a path exists but DFS found none — that
            // would mean an invariant break. Bail out instead of
            // looping forever.
            break;
        }
        round += 1;
        debug_assert!(round <= max_rounds, "Hopcroft-Karp round cap exceeded");
    }

    let row_to_var = pair_u
        .iter()
        .map(|&v| if v == NIL { None } else { Some(v) })
        .collect();
    let var_to_row = pair_v
        .iter()
        .map(|&r| if r == NIL { None } else { Some(r) })
        .collect();

    BipartiteMatching {
        row_to_var,
        var_to_row,
        size,
    }
}

/// BFS layering with the `dist[NIL]` sentinel. Records the
/// shortest distance to *any* unmatched column and prunes BFS
/// expansion past that layer, recovering the textbook O(√V)
/// phase-count bound (PR #60 review nit).
fn bfs(inc: &EqualityIncidence, pair_u: &[usize], pair_v: &[usize], dist: &mut [usize]) -> bool {
    let mut queue: VecDeque<usize> = VecDeque::new();
    for r in 0..inc.n_eq_rows() {
        if pair_u[r] == NIL {
            dist[r] = 0;
            queue.push_back(r);
        } else {
            dist[r] = INF;
        }
    }
    // `dist_nil` is the distance to the nearest unmatched column,
    // used both as the augmenting-path-found signal and as a BFS
    // pruning threshold.
    let mut dist_nil = INF;
    while let Some(r) = queue.pop_front() {
        if dist[r] >= dist_nil {
            continue;
        }
        for &v in inc.neighbors(r) {
            let next = pair_v[v];
            if next == NIL {
                // Reaching an unmatched column means an augmenting
                // path ends here; pin `dist_nil` to the shortest
                // such layer (and skip exploring further past it).
                if dist_nil == INF {
                    dist_nil = dist[r] + 1;
                }
            } else if dist[next] == INF {
                dist[next] = dist[r] + 1;
                queue.push_back(next);
            }
        }
    }
    dist_nil != INF
}

/// DFS along the layered graph, flipping any augmenting path it
/// finds. Returns `true` iff `r` participated in an augmentation.
fn dfs(
    inc: &EqualityIncidence,
    r: usize,
    pair_u: &mut [usize],
    pair_v: &mut [usize],
    dist: &mut [usize],
) -> bool {
    for &v in inc.neighbors(r) {
        let next = pair_v[v];
        let ok = if next == NIL {
            true
        } else if dist[next] == dist[r] + 1 {
            dfs(inc, next, pair_u, pair_v, dist)
        } else {
            false
        };
        if ok {
            pair_u[r] = v;
            pair_v[v] = r;
            return true;
        }
    }
    dist[r] = INF;
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::incidence::ProbeView;
    use pounce_common::types::{Index, Number};

    /// Build an `EqualityIncidence` directly from an edge list
    /// (skipping `ProbeView` so tests don't need to fabricate
    /// `(g_l, g_u)` slices). Every row is treated as an equality.
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

    #[test]
    fn square_full_match_3x3() {
        // Diagonal-ish graph: row k touches col k. Trivial matching
        // of size 3.
        let inc = eq_inc(3, 3, &[(0, 0), (1, 1), (2, 2)]);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 3);
        assert_eq!(m.row_to_var, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn under_determined_2x3() {
        // 2 rows, 3 vars. Row 0 → {0, 1}; row 1 → {1, 2}. Max match = 2.
        let inc = eq_inc(3, 2, &[(0, 0), (0, 1), (1, 1), (1, 2)]);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 2);
        assert!(m.row_to_var.iter().all(|v| v.is_some()));
    }

    #[test]
    fn over_determined_3x2() {
        // 3 rows, 2 vars. Rows 0 and 2 both want var 0 / 1.
        let inc = eq_inc(2, 3, &[(0, 0), (1, 1), (2, 0), (2, 1)]);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 2);
        assert_eq!(m.row_to_var.iter().filter(|v| v.is_some()).count(), 2);
    }

    #[test]
    fn disconnected_components() {
        // Two disjoint K1,1s: row 0 ↔ var 0, row 1 ↔ var 1.
        let inc = eq_inc(2, 2, &[(0, 0), (1, 1)]);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 2);
        assert_eq!(m.row_to_var, vec![Some(0), Some(1)]);
    }

    #[test]
    fn augmenting_path_required() {
        // Two rows, two vars; edges (0,0), (0,1), (1,0). Greedy
        // pairing row0→var0 leaves row1 stuck; Hopcroft-Karp must
        // augment to size 2.
        let inc = eq_inc(2, 2, &[(0, 0), (0, 1), (1, 0)]);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 2);
    }

    #[test]
    fn empty_graph_yields_empty_matching() {
        let inc = eq_inc(0, 0, &[]);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 0);
        assert!(m.row_to_var.is_empty());
        assert!(m.var_to_row.is_empty());
    }

    #[test]
    fn matches_built_via_probeview_too() {
        // Smoke test: building via ProbeView produces the same
        // matching as building the incidence directly.
        let irow: [Index; 3] = [0, 0, 1];
        let jcol: [Index; 3] = [0, 1, 0];
        let g: [Number; 2] = [0.0, 0.0];
        let p = ProbeView {
            n_vars: 2,
            m_rows: 2,
            jac_irow: &irow,
            jac_jcol: &jcol,
            jac_values: None,
            g_l: &g,
            g_u: &g,
            linearity: None,
            one_based: false,
            eq_tol: 1e-12,
            excluded_vars: None,
            excluded_rows: None,
        };
        let inc = EqualityIncidence::from_probe(&p);
        let m = hopcroft_karp(&inc);
        assert_eq!(m.size, 2);
    }

    // Brute-force max matching by enumerating edge subsets. Only used
    // in `konigs_theorem_random_check` where graphs are tiny.
    // Reuses scratch buffers to keep debug-mode allocations down.
    fn brute_force_max_matching(edges: &[(usize, usize)], n_rows: usize, n_vars: usize) -> usize {
        let e = edges.len();
        let mut row_used = vec![false; n_rows];
        let mut var_used = vec![false; n_vars];
        let mut best = 0;
        for mask in 0u32..(1u32 << e) {
            row_used.iter_mut().for_each(|v| *v = false);
            var_used.iter_mut().for_each(|v| *v = false);
            let mut count = 0;
            let mut valid = true;
            for k in 0..e {
                if (mask >> k) & 1 == 1 {
                    let (r, v) = edges[k];
                    if row_used[r] || var_used[v] {
                        valid = false;
                        break;
                    }
                    row_used[r] = true;
                    var_used[v] = true;
                    count += 1;
                }
            }
            if valid && count > best {
                best = count;
            }
        }
        best
    }

    /// Cross-check Hopcroft-Karp against brute force on small random
    /// bipartite graphs. König's theorem guarantees both algorithms
    /// agree on the cardinality (it's the size of a minimum vertex
    /// cover); discrepancies mean Hopcroft-Karp is wrong.
    ///
    /// Graphs capped at 4×4 with ≤ 8 edges so the 2^e brute-force
    /// enumeration stays fast in debug builds.
    #[test]
    fn konigs_theorem_random_check() {
        // Deterministic LCG to keep the test reproducible without
        // pulling in `rand` as a dev-dep. The low bits of a linear
        // congruential generator with odd multiplier and odd
        // increment cycle predictably (parity alternates every call),
        // so we always sample from the high 32 bits before taking a
        // modulus.
        let mut state: u64 = 0xdead_beef_cafe_f00d;
        let mut next = || -> u64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 32
        };

        for _ in 0..20 {
            let n_rows = 1 + (next() % 4) as usize; // 1..=4
            let n_vars = 1 + (next() % 4) as usize; // 1..=4
            let max_edges = (n_rows * n_vars).min(8); // 2^8 = 256
            let n_edges = (next() % (max_edges as u64 + 1)) as usize;

            // Bail out if we'd need too many distinct pairs from a
            // small space — defensive, the cap above already
            // guarantees `n_edges <= n_rows * n_vars`.
            let mut edges_set = std::collections::BTreeSet::<(usize, usize)>::new();
            let mut draws = 0usize;
            while edges_set.len() < n_edges {
                let r = (next() % n_rows as u64) as usize;
                let v = (next() % n_vars as u64) as usize;
                edges_set.insert((r, v));
                draws += 1;
                assert!(draws < 10_000, "edge draw loop is not making progress");
            }
            let edges: Vec<(usize, usize)> = edges_set.into_iter().collect();

            let inc = eq_inc(n_vars, n_rows, &edges);
            let hk = hopcroft_karp(&inc).size;
            let bf = brute_force_max_matching(&edges, n_rows, n_vars);
            assert_eq!(
                hk, bf,
                "Hopcroft-Karp disagrees with brute force on {edges:?} ({n_rows}x{n_vars})"
            );
        }
    }
}
