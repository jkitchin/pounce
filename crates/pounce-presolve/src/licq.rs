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

/// Maximum bipartite matching size between rows and columns. Plain
/// Hungarian-style augmenting paths over an adjacency list. For the
/// problem sizes we care about (equality blocks of LP/NLP models)
/// the simple algorithm is fast enough.
fn bipartite_matching_rank(rows: &[EqRow], n: usize) -> usize {
    let mut match_col: Vec<isize> = vec![-1; n];
    let mut count = 0;
    for (u, row) in rows.iter().enumerate() {
        let mut seen = vec![false; n];
        if try_augment(u, row, rows, &mut match_col, &mut seen) {
            count += 1;
        }
    }
    count
}

fn try_augment(
    u: usize,
    row: &EqRow,
    rows: &[EqRow],
    match_col: &mut [isize],
    seen: &mut [bool],
) -> bool {
    for &col in &row.cols {
        let c = col as usize;
        if c >= seen.len() || seen[c] {
            continue;
        }
        seen[c] = true;
        if match_col[c] < 0
            || try_augment(
                match_col[c] as usize,
                &rows[match_col[c] as usize],
                rows,
                match_col,
                seen,
            )
        {
            match_col[c] = u as isize;
            return true;
        }
    }
    false
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
}
