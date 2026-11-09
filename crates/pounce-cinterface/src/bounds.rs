//! In-place bound updates on an `IpoptProblem` handle (gh#1147, Stage 2), and
//! the indexed pair records they need.
//!
//! Design and the decisions behind it: `dev-notes/issue-1147-c-api-capabilities-plan.md`,
//! §6. The constraint that shapes this file is scale: an update touching `k`
//! entries must cost O(k + touched pairs), never O(n + m) or O(P). So the
//! declared pairs and the `AddPounceComplementsConstraint` ledger are kept
//! with var→entry and row→entry maps, and nothing here calls
//! [`pounce_nlp::mpcc::validate_pairs`] (which allocates `n + m` flags).

use std::collections::HashMap;
use std::ops::Deref;

use pounce_common::types::Number;
use pounce_nlp::mpcc::ComplementarityPair;

/// The declared complementarity pairs of a handle, 0-based, in declaration
/// order, with a var→position and a row→position map.
///
/// Every stored set passed the pair classifier against the handle's
/// effective bounds when it was stored, and every mutation of those bounds
/// re-classifies the pairs it touches, so the set stays valid. A variable
/// and a row each appear in at most one pair, which is what makes the maps
/// well defined.
///
/// Read access is through `Deref` to the slice. Mutation only through the
/// methods here, so the maps cannot drift from the list.
#[derive(Debug, Clone, Default)]
pub(crate) struct DeclaredPairs {
    list: Vec<ComplementarityPair>,
    by_var: HashMap<usize, usize>,
    by_row: HashMap<usize, usize>,
}

impl PartialEq for DeclaredPairs {
    fn eq(&self, other: &Self) -> bool {
        self.list == other.list
    }
}

impl Deref for DeclaredPairs {
    type Target = [ComplementarityPair];
    fn deref(&self) -> &[ComplementarityPair] {
        &self.list
    }
}

impl DeclaredPairs {
    /// Replace the set with `list`, which the caller has validated (no
    /// variable or row twice).
    pub(crate) fn replace(&mut self, list: Vec<ComplementarityPair>) {
        self.by_var = list.iter().enumerate().map(|(k, p)| (p.var, k)).collect();
        self.by_row = list.iter().enumerate().map(|(k, p)| (p.row, k)).collect();
        debug_assert_eq!(self.by_var.len(), list.len(), "a variable in two pairs");
        debug_assert_eq!(self.by_row.len(), list.len(), "a row in two pairs");
        self.list = list;
    }

    /// Append one validated pair whose variable and row are in no pair.
    pub(crate) fn push(&mut self, p: ComplementarityPair) {
        debug_assert!(!self.by_var.contains_key(&p.var) && !self.by_row.contains_key(&p.row));
        let k = self.list.len();
        self.by_var.insert(p.var, k);
        self.by_row.insert(p.row, k);
        self.list.push(p);
    }

    pub(crate) fn clear(&mut self) {
        self.list.clear();
        self.by_var.clear();
        self.by_row.clear();
    }

    /// Position of the pair holding variable `v` (0-based), if any.
    pub(crate) fn of_var(&self, v: usize) -> Option<usize> {
        self.by_var.get(&v).copied()
    }

    /// Position of the pair holding row `r` (0-based), if any.
    pub(crate) fn of_row(&self, r: usize) -> Option<usize> {
        self.by_row.get(&r).copied()
    }
}

/// Which row of the MOI `Complements` table an
/// [`crate::AddPounceComplementsConstraint`] call applied, read off the
/// variable's bounds with the `±1e19` convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComplementsRow {
    Mixed,
    Lower,
    Upper,
    RowFixed,
    VarFixed,
}

/// One accepted [`crate::AddPounceComplementsConstraint`] call: the row and
/// variable (0-based), the table row it applied, and the row bounds the
/// caller had given before the edit — the *user* layer for that row (§6.2
/// D2). The handle's `g_l`/`g_u` hold the *effective* bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ComplementsUse {
    pub(crate) row: usize,
    pub(crate) var: usize,
    pub(crate) table: ComplementsRow,
    pub(crate) user_gl: Number,
    pub(crate) user_gu: Number,
}

/// Every accepted `AddPounceComplementsConstraint` call on a handle, pair or
/// not, with var→entry and row→entry maps. Never cleared: the bound edits it
/// records are not undone by `ClearPounceComplementarityPairs` either.
#[derive(Debug, Clone, Default)]
pub(crate) struct ComplementsLedger {
    list: Vec<ComplementsUse>,
    by_var: HashMap<usize, usize>,
    by_row: HashMap<usize, usize>,
}

impl PartialEq for ComplementsLedger {
    fn eq(&self, other: &Self) -> bool {
        self.list == other.list
    }
}

impl Deref for ComplementsLedger {
    type Target = [ComplementsUse];
    fn deref(&self) -> &[ComplementsUse] {
        &self.list
    }
}

impl ComplementsLedger {
    /// Record an accepted call whose variable and row are in no entry.
    pub(crate) fn push(&mut self, u: ComplementsUse) {
        debug_assert!(!self.by_var.contains_key(&u.var) && !self.by_row.contains_key(&u.row));
        let k = self.list.len();
        self.by_var.insert(u.var, k);
        self.by_row.insert(u.row, k);
        self.list.push(u);
    }

    pub(crate) fn of_var(&self, v: usize) -> Option<usize> {
        self.by_var.get(&v).copied()
    }

    pub(crate) fn of_row(&self, r: usize) -> Option<usize> {
        self.by_row.get(&r).copied()
    }

    /// Overwrite the caller's bounds recorded for entry `k`.
    pub(crate) fn set_user_row_bounds(&mut self, k: usize, gl: Number, gu: Number) {
        self.list[k].user_gl = gl;
        self.list[k].user_gu = gu;
    }
}
