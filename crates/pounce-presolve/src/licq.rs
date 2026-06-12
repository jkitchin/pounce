//! Phase 3 — LICQ degeneracy detection.
//!
//! Equality constraints satisfy LICQ at `x*` iff the rows of `J_c(x*)`
//! are linearly independent. A failure is what the ℓ₁-exact
//! penalty-barrier wrapper (pounce#10) is designed to handle; running
//! the check *before* the IPM starts lets us preempt a guaranteed
//! restoration failure.
//!
//! This module ships a **structural** rank check — cheap, sufficient
//! to catch the common degeneracy patterns, and free of any
//! dependence on pounce-feral or numerical factorization:
//!
//! 1. `m_eq > n` ⇒ `Singular`.
//! 2. Any all-zero equality row ⇒ `Singular`.
//! 3. Bipartite matching between equality rows and the columns of
//!    `J_c` with nonzero entries; a matching of size `m_eq` is
//!    necessary for full structural rank.
//!
//! A numerical rank check (sparse QR or LDLᵀ-of-Gram with zero-pivot
//! counting) is a worthwhile follow-up — see the issue text. For
//! now `LicqVerdict::Full` means *the structural rank is full*, not
//! that the matrix is numerically full-rank.

use pounce_common::types::{Index, Number};

/// One equality row's nonzero column indices.
#[derive(Debug, Clone)]
pub struct EqRow {
    pub cols: Vec<Index>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicqVerdict {
    /// Full structural rank — bipartite matching exists.
    Full,
    /// At least one equality row is structurally zero (no nonzero
    /// Jacobian entries).
    EmptyRow(Index),
    /// `m_eq > n` ⇒ guaranteed singular.
    OverDetermined { m_eq: Index, n: Index },
    /// Bipartite matching could not match every row to a distinct
    /// column ⇒ structural rank < m_eq.
    StructuralRank(Index),
}

/// Run the structural LICQ check on a list of equality rows.
pub fn licq_check(rows: &[EqRow], n: Index) -> LicqVerdict {
    let m = rows.len() as Index;
    if m == 0 {
        return LicqVerdict::Full;
    }
    if m > n {
        return LicqVerdict::OverDetermined { m_eq: m, n };
    }
    for (i, r) in rows.iter().enumerate() {
        if r.cols.is_empty() {
            return LicqVerdict::EmptyRow(i as Index);
        }
    }
    let rank = bipartite_matching_rank(rows, n as usize);
    if rank == rows.len() {
        LicqVerdict::Full
    } else {
        LicqVerdict::StructuralRank(rank as Index)
    }
}

/// Maximum bipartite matching size between equality rows and columns.
///
/// Delegates to the crate's iterative Hopcroft-Karp primitive
/// ([`crate::matching::hopcroft_karp`]) rather than carrying a second,
/// weaker matcher. The previous local implementation allocated a
/// `vec![false; n]` *per row* (`O(m·n)` total) and augmented with
/// recursion whose depth grows with the augmenting-path length — so a
/// long alternating chain (e.g. discretized dynamics) could both burn
/// `O(m·n)` and overflow the stack. Hopcroft-Karp uses BFS layering to
/// prune searches and bounds its DFS depth to the layer distance
/// (`O(√V)`), and shares its battle-tested König-theorem cross-check.
fn bipartite_matching_rank(rows: &[EqRow], n: usize) -> usize {
    use crate::incidence::EqualityIncidence;
    use crate::matching::hopcroft_karp;

    // Pack the rows into the CSR bipartite incidence Hopcroft-Karp
    // consumes. Columns out of range (`≥ n`) are dropped — preserving
    // the previous guard — and each row's columns are sorted + deduped
    // exactly as `EqualityIncidence::from_probe` does, so a column never
    // contributes two parallel edges from one row.
    let mut adj_ptr: Vec<usize> = Vec::with_capacity(rows.len() + 1);
    let mut vars: Vec<usize> = Vec::new();
    let mut scratch: Vec<usize> = Vec::new();
    adj_ptr.push(0);
    for row in rows {
        scratch.clear();
        scratch.extend(row.cols.iter().filter_map(|&c| {
            let c = c as usize;
            (c < n).then_some(c)
        }));
        scratch.sort_unstable();
        scratch.dedup();
        vars.extend_from_slice(&scratch);
        adj_ptr.push(vars.len());
    }

    let inc = EqualityIncidence {
        n_vars: n,
        eq_row_inner_idx: (0..rows.len()).collect(),
        adj_ptr,
        vars,
    };
    hopcroft_karp(&inc).size
}

/// Construct an `EqRow` list from CSR-ish triples: for each equality
/// row (caller-identified), the columns of its Jacobian nonzeros.
/// Drops exact-zero entries; preserves duplicates as a single column.
pub fn eq_rows_from_triples(
    eq_row_indices: &[usize],
    triples: &[(Index, Index, Number)],
    inner_m: usize,
) -> Vec<EqRow> {
    use std::collections::BTreeSet;
    let mut by_row: Vec<BTreeSet<Index>> = vec![BTreeSet::new(); inner_m];
    for &(i, j, v) in triples {
        if v == 0.0 {
            continue;
        }
        let i = i as usize;
        if i < inner_m {
            by_row[i].insert(j);
        }
    }
    eq_row_indices
        .iter()
        .map(|&i| EqRow {
            cols: by_row[i].iter().copied().collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(cols: &[Index]) -> EqRow {
        EqRow {
            cols: cols.to_vec(),
        }
    }

    #[test]
    fn no_equality_rows_is_full_rank() {
        assert_eq!(licq_check(&[], 5), LicqVerdict::Full);
    }

    #[test]
    fn over_determined_caught() {
        let rows = vec![row(&[0]), row(&[0]), row(&[0])];
        assert!(matches!(
            licq_check(&rows, 2),
            LicqVerdict::OverDetermined { m_eq: 3, n: 2 }
        ));
    }

    #[test]
    fn empty_row_caught() {
        let rows = vec![row(&[0]), row(&[])];
        assert!(matches!(licq_check(&rows, 5), LicqVerdict::EmptyRow(1)));
    }

    #[test]
    fn duplicate_singletons_dropped_by_matching() {
        // Two rows that each only touch column 0: only one matches,
        // structural rank = 1.
        let rows = vec![row(&[0]), row(&[0])];
        assert!(matches!(
            licq_check(&rows, 5),
            LicqVerdict::StructuralRank(1)
        ));
    }

    #[test]
    fn distinct_singletons_full_rank() {
        let rows = vec![row(&[0]), row(&[1]), row(&[2])];
        assert_eq!(licq_check(&rows, 5), LicqVerdict::Full);
    }

    #[test]
    fn matching_via_augmenting_path() {
        // Row 0 touches cols {0,1}; row 1 touches col {0}.
        // Without augmenting, row 0 may grab 0; then row 1 stuck.
        // With augmenting, row 0 gets bumped to 1, row 1 takes 0.
        let rows = vec![row(&[0, 1]), row(&[0])];
        assert_eq!(licq_check(&rows, 2), LicqVerdict::Full);
    }

    /// Long alternating chain — `m` equality rows over `m − 1` columns
    /// as a staircase (`row 0: {0}`, `row i: {i−1, i}`, final row: just
    /// the last column). Structural rank is `m − 1`, so LICQ must report
    /// `StructuralRank(m − 1)`. The previous recursive matcher augmented
    /// the final row by recursing the full length of the chain (verified
    /// depth ≈ m), which overflows the stack for `m` in the tens of
    /// thousands — exactly the discretized-dynamics shape the review
    /// flags. Hopcroft-Karp's BFS layering bounds the search, so this
    /// completes on a normal stack. (Issue M29.)
    #[test]
    fn long_chain_does_not_overflow_stack() {
        let m = 50_000usize;
        let mut rows: Vec<EqRow> = Vec::with_capacity(m);
        rows.push(row(&[0]));
        for i in 1..(m - 1) {
            rows.push(row(&[(i - 1) as Index, i as Index]));
        }
        rows.push(row(&[(m - 2) as Index]));
        // The chain touches m-1 distinct columns (0..m-2); declaring an
        // extra phantom column (n = m) keeps `m <= n` so the over-
        // determined short-circuit does NOT fire and the matcher actually
        // runs. Max matching = m-1 ⇒ StructuralRank(m-1).
        assert_eq!(
            licq_check(&rows, m as Index),
            LicqVerdict::StructuralRank((m - 1) as Index)
        );
    }

    /// A long chain that IS full structural rank — `m` rows over `m`
    /// columns, same staircase plus a final row reaching a fresh column.
    /// The maximum matching is perfect, so the verdict is `Full`. Guards
    /// against the fix mistakenly capping the matching short on long
    /// augmenting paths.
    #[test]
    fn long_chain_full_rank() {
        let m = 20_000usize;
        let mut rows: Vec<EqRow> = Vec::with_capacity(m);
        rows.push(row(&[0]));
        for i in 1..m {
            rows.push(row(&[(i - 1) as Index, i as Index]));
        }
        assert_eq!(licq_check(&rows, m as Index), LicqVerdict::Full);
    }
}
