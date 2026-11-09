//! In-place bound updates on an `IpoptProblem` handle (gh#1147, Stage 2), and
//! the indexed pair records they need.
//!
//! Design and the decisions behind it: `dev-notes/issue-1147-c-api-capabilities-plan.md`,
//! §6. The constraint that shapes this file is scale: an update touching `k`
//! entries must cost O(k + touched pairs), never O(n + m) or O(P). So the
//! declared pairs and the `AddPounceComplementsConstraint` ledger are kept
//! with var→entry and row→entry maps, and nothing here calls
//! [`pounce_nlp::mpcc::validate_pairs`] (which allocates `n + m` flags).

use std::collections::{HashMap, HashSet};
use std::ffi::c_int;
use std::ops::Deref;

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, Number};
use pounce_nlp::mpcc::{ComplementarityPair, PairShape, classify_pair};

use crate::{
    Index, IpoptProblem, IpoptProblemInfo, bound_status_consistent, complements_row,
    cons_status_consistent, ffi_guard, zero_based,
};

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

// ---------------------------------------------------------------------------
// SetPounceBounds / GetPounceBounds (gh#1147 Stage 2, plan §6.2)
// ---------------------------------------------------------------------------

/// The update was applied (setter) or the bounds were written (getter).
pub const POUNCE_BOUNDS_OK: c_int = 0;
/// NULL handle.
pub const POUNCE_BOUNDS_ERR_NULL: c_int = -1;
/// A negative count, or a NULL index array with a count that is neither 0
/// nor the full size (`n` for variables, `m` for rows).
pub const POUNCE_BOUNDS_ERR_COUNT: c_int = -2;
/// An index outside `[0, n)` / `[0, m)`, read in the problem's index style.
pub const POUNCE_BOUNDS_ERR_OUT_OF_RANGE: c_int = -3;
/// An index given twice on one side of the batch (setter only).
pub const POUNCE_BOUNDS_ERR_DUPLICATE: c_int = -4;
/// A NaN bound.
pub const POUNCE_BOUNDS_ERR_NAN: c_int = -5;
/// A declared pair would leave its native shapes, or change which variable
/// bound and which row side it pairs (plan §6.2 D3).
pub const POUNCE_BOUNDS_ERR_PAIR_CHANGED: c_int = -6;
/// A variable used by `AddPounceComplementsConstraint` would move to another
/// row of the MOI table (lower-only, upper-only, boxed, fixed, free).
pub const POUNCE_BOUNDS_ERR_COMPLEMENTS_CLASS: c_int = -7;
/// A row used by `AddPounceComplementsConstraint` given a finite bound: MOI's
/// `F` is a free function (plan §6.2 D2).
pub const POUNCE_BOUNDS_ERR_COMPLEMENTS_ROW: c_int = -8;
/// The working set staged by `IpoptSetWarmStartWorkingSet` would claim
/// something the new bounds contradict.
pub const POUNCE_BOUNDS_ERR_WORKING_SET: c_int = -9;
/// `GetPounceBounds`: `view` is neither `POUNCE_BOUNDS_VIEW_USER` nor
/// `POUNCE_BOUNDS_VIEW_EFFECTIVE`.
pub const POUNCE_BOUNDS_ERR_VIEW: c_int = -10;
/// The handle is inconsistent (a declared pair that no longer classifies),
/// or a Rust panic was caught at the boundary. Nothing was changed.
pub const POUNCE_BOUNDS_ERR_INTERNAL: c_int = -11;
/// The bounds the caller gave (`CreateIpoptProblem`, `SetPounceBounds`).
pub const POUNCE_BOUNDS_VIEW_USER: c_int = 0;
/// The bounds the solve sees: the user's, with the edits
/// `AddPounceComplementsConstraint` made to its rows.
pub const POUNCE_BOUNDS_VIEW_EFFECTIVE: c_int = 1;

/// What a native pair pairs: which variable bound with which row side, or
/// the mixed form. The invariant a bound update must keep (plan §6.2 D3).
/// `OneSided` and `OneSidedPlusBound` with the same sides are the same
/// identity: the second only adds a plain limit on the variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairIdentity {
    Sides { var_lower: bool, row_lower: bool },
    Mixed,
}

fn identity(shape: PairShape) -> PairIdentity {
    match shape {
        PairShape::OneSided { g, h } | PairShape::OneSidedPlusBound { g, h } => {
            PairIdentity::Sides {
                var_lower: g.0 > 0.0,
                row_lower: h.0 > 0.0,
            }
        }
        PairShape::Mixed { .. } => PairIdentity::Mixed,
    }
}

/// A refusal: the code and the batch position to report (`-1`: none).
type Refusal = (c_int, Index);

/// One side of a batch read and checked entry by entry: per entry, in
/// order, range, duplicate, NaN. `cur(j)` gives entry `j`'s current user
/// bounds, used for a NULL value array. `base` is the side's offset in the
/// concatenated batch.
struct Side {
    /// `(index, lo, hi)` per entry, 0-based, in batch order.
    entries: Vec<(usize, Number, Number)>,
    /// index → position in `entries`.
    at: HashMap<usize, usize>,
}

#[allow(clippy::too_many_arguments)]
unsafe fn read_side(
    count: Index,
    idx: *const Index,
    lo: *const Number,
    hi: *const Number,
    len: Index,
    offset: Index,
    base: Index,
    cur: impl Fn(usize) -> (Number, Number),
) -> Result<Side, Refusal> {
    // More entries than the side has indices must repeat one, so it is
    // refused here, before anything is reserved: a garbage count (say an
    // uninitialised `nx`) would otherwise size the reservation below, and a
    // failed allocation aborts the process, past `ffi_guard`.
    if count < 0 || count > len || (idx.is_null() && count != 0 && count != len) {
        return Err((POUNCE_BOUNDS_ERR_COUNT, -1));
    }
    let k = count as usize;
    let mut side = Side {
        entries: Vec::with_capacity(k),
        at: HashMap::with_capacity(k),
    };
    for p in 0..k {
        let pos = base + p as Index;
        let j = if idx.is_null() {
            p
        } else {
            zero_based(unsafe { *idx.add(p) }, offset, len)
        };
        if j == usize::MAX {
            return Err((POUNCE_BOUNDS_ERR_OUT_OF_RANGE, pos));
        }
        if side.at.insert(j, p).is_some() {
            return Err((POUNCE_BOUNDS_ERR_DUPLICATE, pos));
        }
        let (cl, cu) = cur(j);
        let l = if lo.is_null() {
            cl
        } else {
            unsafe { *lo.add(p) }
        };
        let u = if hi.is_null() {
            cu
        } else {
            unsafe { *hi.add(p) }
        };
        if l.is_nan() || u.is_nan() {
            return Err((POUNCE_BOUNDS_ERR_NAN, pos));
        }
        side.entries.push((j, l, u));
    }
    Ok(side)
}

/// The user-layer bounds of row `r`: the ledger's copy for a row an
/// `AddPounceComplementsConstraint` call edited, else the handle's.
fn user_row(info: &IpoptProblemInfo, r: usize) -> (Number, Number) {
    match info.complements_used.of_row(r) {
        Some(e) => {
            let u = &info.complements_used[e];
            (u.user_gl, u.user_gu)
        }
        None => (info.g_l[r], info.g_u[r]),
    }
}

fn is_free(lo: Number, hi: Number) -> bool {
    lo <= NLP_LOWER_BOUND_INF && hi >= NLP_UPPER_BOUND_INF
}

/// Validate the whole batch against the bounds it would install, then
/// commit it, or change nothing. Plan §6.2.
#[allow(clippy::too_many_arguments)]
unsafe fn set_bounds(
    info: &mut IpoptProblemInfo,
    nx: Index,
    x_idx: *const Index,
    x_l: *const Number,
    x_u: *const Number,
    ng: Index,
    g_idx: *const Index,
    g_l: *const Number,
    g_u: *const Number,
) -> Result<(), Refusal> {
    let offset = if info.index_style == 1 { 1 } else { 0 };
    // Both counts before any entry, so a bad count is never reported as an
    // entry's fault.
    if nx < 0 || ng < 0 {
        return Err((POUNCE_BOUNDS_ERR_COUNT, -1));
    }
    let xs = unsafe {
        read_side(nx, x_idx, x_l, x_u, info.n, offset, 0, |v| {
            (info.x_l[v], info.x_u[v])
        })?
    };
    let gs = unsafe {
        read_side(ng, g_idx, g_l, g_u, info.m, offset, nx, |r| {
            user_row(info, r)
        })?
    };

    // The bounds after the update, as the solve would see them.
    let var_new = |v: usize| match xs.at.get(&v) {
        Some(&p) => (xs.entries[p].1, xs.entries[p].2),
        None => (info.x_l[v], info.x_u[v]),
    };
    let row_new_effective = |r: usize| {
        let user = match gs.at.get(&r) {
            Some(&p) => (gs.entries[p].1, gs.entries[p].2),
            None => user_row(info, r),
        };
        match info.complements_used.of_row(r) {
            Some(e) => {
                let (l, u, _) = info.complements_used[e].table.row_bounds(user.0, user.1);
                (l, u)
            }
            None => user,
        }
    };
    let mut pairs_checked: HashSet<usize> = HashSet::new();
    let mut check_pair = |q: usize, pos: Index| -> Result<(), Refusal> {
        if !pairs_checked.insert(q) {
            return Ok(());
        }
        let pair = info.complementarity[q];
        let (v, r) = (pair.var, pair.row);
        let Some(old) = classify_pair(info.x_l[v], info.x_u[v], info.g_l[r], info.g_u[r]) else {
            // A stored pair that no longer classifies: the handle was
            // already inconsistent; not the caller's entry.
            return Err((POUNCE_BOUNDS_ERR_INTERNAL, -1));
        };
        let (xl, xu) = var_new(v);
        let (gl, gu) = row_new_effective(r);
        match classify_pair(xl, xu, gl, gu) {
            Some(new) if identity(new) == identity(old) => Ok(()),
            _ => Err((POUNCE_BOUNDS_ERR_PAIR_CHANGED, pos)),
        }
    };

    // D2 then D3, entry by entry in batch order.
    for (p, &(v, l, u)) in xs.entries.iter().enumerate() {
        let pos = p as Index;
        if let Some(e) = info.complements_used.of_var(v)
            && complements_row(l, u) != Some(info.complements_used[e].table)
        {
            return Err((POUNCE_BOUNDS_ERR_COMPLEMENTS_CLASS, pos));
        }
        if let Some(q) = info.complementarity.of_var(v) {
            check_pair(q, pos)?;
        }
    }
    for (p, &(r, l, u)) in gs.entries.iter().enumerate() {
        let pos = nx + p as Index;
        if info.complements_used.of_row(r).is_some() && !is_free(l, u) {
            return Err((POUNCE_BOUNDS_ERR_COMPLEMENTS_ROW, pos));
        }
        if let Some(q) = info.complementarity.of_row(r) {
            check_pair(q, pos)?;
        }
    }

    // A staged working set's claims, at the touched indices only.
    if let Some(ws) = info.pending_working_set.as_ref() {
        for (p, &(v, l, u)) in xs.entries.iter().enumerate() {
            if !bound_status_consistent(ws.bounds[v], l, u) {
                return Err((POUNCE_BOUNDS_ERR_WORKING_SET, p as Index));
            }
        }
        for (p, &(r, _, _)) in gs.entries.iter().enumerate() {
            let (l, u) = row_new_effective(r);
            if !cons_status_consistent(ws.constraints[r], l, u) {
                return Err((POUNCE_BOUNDS_ERR_WORKING_SET, nx + p as Index));
            }
        }
    }

    // Commit. Nothing below can refuse.
    let effective_rows: Vec<(usize, Number, Number)> = gs
        .entries
        .iter()
        .map(|&(r, _, _)| {
            let (l, u) = row_new_effective(r);
            (r, l, u)
        })
        .collect();
    for &(v, l, u) in &xs.entries {
        info.x_l[v] = l;
        info.x_u[v] = u;
    }
    for (&(r, ul, uu), &(_, el, eu)) in gs.entries.iter().zip(&effective_rows) {
        if let Some(e) = info.complements_used.of_row(r) {
            info.complements_used.set_user_row_bounds(e, ul, uu);
        }
        info.g_l[r] = el;
        info.g_u[r] = eu;
    }
    Ok(())
}

/// Update variable and constraint bounds on an existing handle, in place
/// (gh#1147 Stage 2). See `pounce.h` and the plan, §6.2, for the contract.
///
/// `nx` entries `(x_idx[k], x_L[k], x_U[k])` and `ng` entries
/// `(g_idx[k], g_L[k], g_U[k])`, indices in the problem's index style. A
/// NULL index array with the full count (`n`, `m`) is dense; a NULL value
/// array leaves that side unchanged. All or nothing: the batch is checked
/// against the bounds it would install — NaN, duplicates, declared pairs
/// keeping their paired sides, `Complements` variables keeping their MOI
/// table row and rows staying free, a staged SQP working set staying true
/// — and committed only if it passes. `bad_index` (may be NULL) receives
/// the refused entry's position in the concatenated batch (`0..nx` for
/// variables, `nx..nx+ng` for rows), or `-1`.
///
/// Expected cost O(k + touched pairs) for k entries, whatever n, m and the
/// number of pairs.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL. Each non-NULL
/// array must hold its side's count of elements; `bad_index` must be NULL
/// or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SetPounceBounds(
    ipopt_problem: IpoptProblem,
    nx: Index,
    x_idx: *const Index,
    x_L: *const Number,
    x_U: *const Number,
    ng: Index,
    g_idx: *const Index,
    g_L: *const Number,
    g_U: *const Number,
    bad_index: *mut Index,
) -> c_int {
    let report = |k: Index| {
        if !bad_index.is_null() {
            unsafe { *bad_index = k };
        }
    };
    report(-1);
    ffi_guard(POUNCE_BOUNDS_ERR_INTERNAL, || unsafe {
        if ipopt_problem.is_null() {
            return POUNCE_BOUNDS_ERR_NULL;
        }
        match set_bounds(
            &mut *ipopt_problem,
            nx,
            x_idx,
            x_L,
            x_U,
            ng,
            g_idx,
            g_L,
            g_U,
        ) {
            Ok(()) => POUNCE_BOUNDS_OK,
            Err((code, pos)) => {
                report(pos);
                code
            }
        }
    })
}

/// Read variable and constraint bounds from a handle (gh#1147 Stage 2).
///
/// Same index arguments as [`SetPounceBounds`] (duplicates allowed); a NULL
/// output array is not written. `view` is [`POUNCE_BOUNDS_VIEW_USER`] (what
/// the caller gave) or [`POUNCE_BOUNDS_VIEW_EFFECTIVE`] (what the solve
/// sees: the user bounds with `AddPounceComplementsConstraint`'s row
/// edits). The two views differ only on rows that function edited. Every
/// index is checked before anything is written, so a refused call writes
/// nothing.
///
/// # Safety
///
/// As for [`SetPounceBounds`]; the output arrays must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetPounceBounds(
    ipopt_problem: IpoptProblem,
    nx: Index,
    x_idx: *const Index,
    x_L: *mut Number,
    x_U: *mut Number,
    ng: Index,
    g_idx: *const Index,
    g_L: *mut Number,
    g_U: *mut Number,
    view: c_int,
    bad_index: *mut Index,
) -> c_int {
    let report = |k: Index| {
        if !bad_index.is_null() {
            unsafe { *bad_index = k };
        }
    };
    report(-1);
    ffi_guard(POUNCE_BOUNDS_ERR_INTERNAL, || unsafe {
        if ipopt_problem.is_null() {
            return POUNCE_BOUNDS_ERR_NULL;
        }
        let info = &*ipopt_problem;
        if view != POUNCE_BOUNDS_VIEW_USER && view != POUNCE_BOUNDS_VIEW_EFFECTIVE {
            return POUNCE_BOUNDS_ERR_VIEW;
        }
        let offset = if info.index_style == 1 { 1 } else { 0 };
        let resolve = |count: Index, idx: *const Index, len: Index, base: Index| {
            if count < 0 || (idx.is_null() && count != 0 && count != len) {
                return Err((POUNCE_BOUNDS_ERR_COUNT, -1));
            }
            (0..count as usize)
                .map(|p| {
                    let j = if idx.is_null() {
                        p
                    } else {
                        zero_based(*idx.add(p), offset, len)
                    };
                    if j == usize::MAX {
                        Err((POUNCE_BOUNDS_ERR_OUT_OF_RANGE, base + p as Index))
                    } else {
                        Ok(j)
                    }
                })
                .collect::<Result<Vec<usize>, Refusal>>()
        };
        if nx < 0 || ng < 0 {
            return POUNCE_BOUNDS_ERR_COUNT;
        }
        let (vars, rows) = match resolve(nx, x_idx, info.n, 0)
            .and_then(|v| resolve(ng, g_idx, info.m, nx).map(|r| (v, r)))
        {
            Ok(vr) => vr,
            Err((code, pos)) => {
                report(pos);
                return code;
            }
        };
        for (p, &v) in vars.iter().enumerate() {
            if !x_L.is_null() {
                *x_L.add(p) = info.x_l[v];
            }
            if !x_U.is_null() {
                *x_U.add(p) = info.x_u[v];
            }
        }
        for (p, &r) in rows.iter().enumerate() {
            let (l, u) = if view == POUNCE_BOUNDS_VIEW_USER {
                user_row(info, r)
            } else {
                (info.g_l[r], info.g_u[r])
            };
            if !g_L.is_null() {
                *g_L.add(p) = l;
            }
            if !g_U.is_null() {
                *g_U.add(p) = u;
            }
        }
        POUNCE_BOUNDS_OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddPounceComplementarityPair, CreateIpoptProblem, FreeIpoptProblem};

    unsafe extern "C" fn f(
        _: Index,
        _: *const Number,
        _: crate::Bool,
        _: *mut Number,
        _: *mut std::ffi::c_void,
    ) -> crate::Bool {
        0
    }
    unsafe extern "C" fn g(
        _: Index,
        _: *const Number,
        _: crate::Bool,
        _: Index,
        _: *mut Number,
        _: *mut std::ffi::c_void,
    ) -> crate::Bool {
        0
    }
    #[allow(clippy::too_many_arguments)]
    unsafe extern "C" fn jac(
        _: Index,
        _: *const Number,
        _: crate::Bool,
        _: Index,
        _: Index,
        _: *mut Index,
        _: *mut Index,
        _: *mut Number,
        _: *mut std::ffi::c_void,
    ) -> crate::Bool {
        0
    }

    #[test]
    fn a_stored_pair_that_no_longer_classifies_is_an_internal_error() {
        // No public call can leave a declared pair unclassifiable (the
        // setter re-classifies every pair it touches), so plant it: var 0's
        // box collapsed under its mixed pair. An update touching that pair
        // must answer INTERNAL with no entry blamed, and change nothing.
        let (x_l, x_u) = ([0.0, 0.0], [2.0, 2.0]);
        let (g_l, g_u) = ([-2e19, -2e19], [2e19, 2e19]);
        unsafe {
            let p = CreateIpoptProblem(
                2,
                x_l.as_ptr(),
                x_u.as_ptr(),
                2,
                g_l.as_ptr(),
                g_u.as_ptr(),
                0,
                0,
                0,
                Some(f),
                Some(g),
                Some(f_grad),
                Some(jac),
                None,
            );
            assert!(!p.is_null());
            assert_eq!(AddPounceComplementarityPair(p, 0, 0), crate::POUNCE_CC_OK);
            (&mut (*p).x_u)[0] = 0.0;
            let before = ((&*p).x_l.clone(), (&*p).x_u.clone(), (&*p).g_l.clone());
            let (idx, lo, hi) = ([1], [-1.0], [3.0]);
            let mut bad = -42;
            let code = SetPounceBounds(
                p,
                1,
                idx.as_ptr(),
                lo.as_ptr(),
                hi.as_ptr(),
                1,
                [0].as_ptr(),
                [-2e19].as_ptr(),
                [2e19].as_ptr(),
                &mut bad,
            );
            assert_eq!((code, bad), (POUNCE_BOUNDS_ERR_INTERNAL, -1));
            assert_eq!(
                before,
                ((&*p).x_l.clone(), (&*p).x_u.clone(), (&*p).g_l.clone())
            );
            FreeIpoptProblem(p);
        }
    }

    unsafe extern "C" fn f_grad(
        _: Index,
        _: *const Number,
        _: crate::Bool,
        _: *mut Number,
        _: *mut std::ffi::c_void,
    ) -> crate::Bool {
        0
    }
}
