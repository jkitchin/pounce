//! The dependent-row probe's elimination, with a budget (gh#1206 B5).
//!
//! [`crate::determine_dependent_rows`] rank-reveals the augmented system
//! `M = [s·I Jᵀ; J 0]` with threshold-Markowitz LU. It called
//! `feral::SparseLu::factor_markowitz`, which has no way to be stopped: no
//! deadline, no cap on fill. gh#1206 reported one call on a 90 797-variable,
//! 88 800-row collocation NLP (`d ≈ 1.8·10⁵`) that ran for more than 2 400 s of
//! CPU and grew past 11.6 GB, well past the IPM→SQP handover's deadline, which
//! the QP solver checks only between its own steps.
//!
//! This module is that elimination with a budget threaded into it, and with
//! the two costs that made it quadratic taken out. It is **the same
//! elimination**, step for step: the pivot search, the singleton stacks, the
//! lazily built column heap (whose multiset is part of the pivot rule), the
//! tie-breaks, each column's element order, the rank-1 update and its
//! floating-point operation order are feral 0.24's
//! `lu::markowitz::markowitz_factor` (MIT, same author), so on every input it
//! finishes it chooses the same pivots with the same values and the probe flags
//! the same rows. The test `budgeted_elimination_matches_feral_markowitz`
//! holds it to `SparseLu` bit for bit. What differs:
//!
//! * **It can stop.** Between pivot steps it checks a wall-clock deadline
//!   ([`rank_probe_deadline`], set by the active-set QP from its
//!   `time_limit`), the size of the active submatrix (the only storage that
//!   grows) and a work count. Past any of them it stops, the probe returns
//!   `FatalError`, and the QP is told the rank test could not run, which it
//!   already handles (`checked_independent_active_subset`).
//! * **It keeps no factor.** The probe reads one number per pivot, the `U`
//!   diagonal; `factor_markowitz` also stored all of `L` and `U`, which on a
//!   filling elimination is most of the memory. Here they are dropped as they
//!   are produced.
//! * **"First live row/column" is a cursor, not a scan.** A perturbed singular
//!   pivot takes the lowest-numbered live row, found by `(0..m).find(..)` in
//!   feral, `O(m)` per dependent row. Rows and columns only ever die, so a
//!   forward cursor finds the same one in amortized `O(1)`.
//! * **Numerically empty columns are parked** out of the search heap instead
//!   of being popped, scanned and pushed back at every step ([`Parked`]).
//!   With `r` dependent rows that was `O(r)` per pivot step.
//! * **Long columns are indexed by row** ([`Active`]), so finding the pivot
//!   row's entry and updating it no longer scan the column.
//!
//! The last two are the scaling fix. Measured (release build) on the gh#1061
//! recurrence-Jacobian generator at 8 states, which has long aggregate rows,
//! scaled duplicates and bound rows: at 98 862 rows × 90 008 columns
//! (`d = 188 870`, 8 854 dependent rows, about the size of gh#1206's) the
//! `SparseLu` probe took 37–39 s and this one 0.58–0.61 s; at 191 859 × 180 008
//! (11 849 dependent), 144 s against 1.5 s, peak RSS 333 MB against 261 MB. Both
//! flagged the same rows. Of the 37 s, 31 s were the search re-scanning parked
//! columns; most of the rest was the two scans of long columns.
//!
//! ## Why not a symmetric LDLᵀ
//!
//! The issue suggests keeping the KKT's symmetry, e.g. FERAL's own multifrontal
//! LDLᵀ, which the IPM factors at this size in seconds. It does not answer the
//! question asked here. Bunch–Kaufman-style pivoting with 1×1 and 2×2 pivots
//! is not rank-revealing: a near-null direction of `M` can be spread over a
//! 2×2 block that pairs an `x`-row with a `J`-row, or carried into a delayed
//! pivot that is force-accepted at the root, so a small `D` entry does not
//! name a dependent `J`-row, and the pivot order is fixed by a fill-reducing
//! ordering chosen before any value is seen. feral's public API also reports
//! only the *count* of zero pivots (`SparseFactors::n_zero_pivots`), not
//! where they fell. The count alone would give the rank but not which rows to
//! drop, which is what the QP needs. Before gh#1061 the probe was a
//! static-order factorization (Gilbert–Peierls in an AMD order) and was wrong
//! on about two thirds of the dynamic-optimization corpus, over- and
//! under-dropping both; threshold-Markowitz, which chooses pivots from the
//! values, was the fix. A static-order symmetric factorization would go back
//! to that failure class. So the LU stays, with a budget, and a probe that
//! cannot finish says so instead of running unbounded.

use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

use feral::SparseColMatrix;

thread_local! {
    /// The deadline the dependent-row probe honours on this thread, if any.
    static PROBE_DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
    /// Why the last probe on this thread stopped early, if it did.
    static LAST_BAILOUT: Cell<Option<ProbeBailout>> = const { Cell::new(None) };
}

/// Restores the previous probe deadline when dropped. See
/// [`rank_probe_deadline`].
#[must_use = "the deadline is lifted when the guard is dropped"]
pub struct RankProbeDeadline {
    prev: Option<Instant>,
}

impl Drop for RankProbeDeadline {
    fn drop(&mut self) {
        let prev = self.prev;
        PROBE_DEADLINE.with(|d| d.set(prev));
    }
}

/// Make every dependent-row probe on this thread (the free
/// [`crate::determine_dependent_rows`] and `FeralSolverInterface`'s trait
/// method alike) stop at `deadline` until the returned guard is dropped.
///
/// A deadline is passed this way rather than as an argument because the probe
/// is reached through `SparseSymLinearSolverInterface::determine_dependent_rows`,
/// whose signature every backend shares. Scopes nest, and an inner one can only
/// tighten: the effective deadline is the earlier of the two. `None` leaves the
/// enclosing deadline in force.
pub fn rank_probe_deadline(deadline: Option<Instant>) -> RankProbeDeadline {
    let prev = PROBE_DEADLINE.with(Cell::get);
    let eff = match (prev, deadline) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    PROBE_DEADLINE.with(|d| d.set(eff));
    RankProbeDeadline { prev }
}

/// Which budget stopped a dependent-row probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeBailout {
    /// The wall-clock deadline passed.
    Deadline,
    /// The active submatrix outgrew [`DependentRowsBudget::max_active_bytes`].
    Fill,
    /// The elimination did more than [`DependentRowsBudget::max_work`] entry
    /// operations.
    Work,
}

/// Why the last dependent-row probe on this thread stopped before it had an
/// answer, or `None` if it finished (or failed for another reason). Reset at
/// the start of every probe.
pub fn last_probe_bailout() -> Option<ProbeBailout> {
    LAST_BAILOUT.with(Cell::get)
}

/// Limits on one dependent-row probe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DependentRowsBudget {
    /// Stop when this instant has passed (checked between pivot steps).
    pub deadline: Option<Instant>,
    /// Stop when the active submatrix's storage exceeds this many bytes.
    /// `None` uses [`default_max_active_bytes`] of the augmented system.
    pub max_active_bytes: Option<usize>,
    /// Stop after this many entry operations (column scans in the pivot
    /// search plus entries touched by the updates).
    pub max_work: u64,
}

/// Work budget of [`DependentRowsBudget::from_scope`]: entry operations
/// (column entries scanned by the search, lookups, entries walked by the
/// update), at a few nanoseconds each some tens of seconds. The probes measured
/// when the LU replaced Gilbert–Peierls (gh#1061, `brainpc2` and the
/// collocation captures) took 0.07–0.25 s, and the 188 870-dimension
/// recurrence system in the module doc 0.6 s. This is the guard for a caller
/// that set no deadline; the QP's deadline is the one that normally binds.
pub const DEFAULT_PROBE_MAX_WORK: u64 = 10_000_000_000;

/// Floor and ceiling of [`default_max_active_bytes`].
const MIN_ACTIVE_BYTES: usize = 256 << 20;
const MAX_ACTIVE_BYTES: usize = 2 << 30;
/// Fill factor of [`default_max_active_bytes`] over the input's storage.
const ACTIVE_FILL_FACTOR: usize = 64;

/// The active-submatrix cap a probe of a matrix with `nnz` stored entries gets
/// by default: 64 times the input's own storage, kept within [256 MiB, 2 GiB].
/// gh#1206's call passed 11.6 GB.
pub fn default_max_active_bytes(nnz: usize) -> usize {
    (ACTIVE_FILL_FACTOR
        .saturating_mul(nnz)
        .saturating_mul(ENTRY_BYTES))
    .clamp(MIN_ACTIVE_BYTES, MAX_ACTIVE_BYTES)
}

/// Bytes per stored active entry: `(row, value)` by column plus the column
/// index by row.
const ENTRY_BYTES: usize = std::mem::size_of::<(usize, f64)>() + std::mem::size_of::<usize>();

impl DependentRowsBudget {
    /// The budget a probe gets by default: this thread's
    /// [`rank_probe_deadline`], the default fill cap, and
    /// [`DEFAULT_PROBE_MAX_WORK`].
    pub fn from_scope() -> Self {
        Self {
            deadline: PROBE_DEADLINE.with(Cell::get),
            max_active_bytes: None,
            max_work: DEFAULT_PROBE_MAX_WORK,
        }
    }

    /// No limit at all; what the probe was before gh#1206.
    pub fn unlimited() -> Self {
        Self {
            deadline: None,
            max_active_bytes: Some(usize::MAX),
            max_work: u64::MAX,
        }
    }
}

/// Why the elimination produced no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProbeError {
    /// What `factor_markowitz` would have returned an `Err` for.
    Failed,
    /// A budget ran out first.
    OverBudget(ProbeBailout),
}

/// One pivot of the elimination: original row and the pivot's value (the `U`
/// diagonal at that position).
pub(crate) struct Pivots {
    pub perm: Vec<usize>,
    pub value: Vec<f64>,
}

/// Variable-length lists, one per column (or row), packed in one buffer.
/// feral's `lu::markowitz::Lists`, whose element order the pivot search's
/// tie-breaks depend on.
#[derive(Debug, Clone, Default)]
struct Lists<T> {
    start: Vec<usize>,
    len: Vec<usize>,
    cap: Vec<usize>,
    data: Vec<T>,
}

impl<T: Copy + Default> Lists<T> {
    fn with_capacities(caps: &[usize]) -> Self {
        let mut start = Vec::with_capacity(caps.len());
        let mut total = 0usize;
        for &c in caps {
            start.push(total);
            total += c;
        }
        Self {
            start,
            len: vec![0; caps.len()],
            cap: caps.to_vec(),
            data: vec![T::default(); total],
        }
    }

    #[inline]
    fn get(&self, j: usize) -> &[T] {
        let s = self.start[j];
        &self.data[s..s + self.len[j]]
    }

    #[inline]
    fn get_mut(&mut self, j: usize) -> &mut [T] {
        let s = self.start[j];
        let l = self.len[j];
        &mut self.data[s..s + l]
    }

    #[inline]
    fn push(&mut self, j: usize, x: T) {
        if self.len[j] == self.cap[j] {
            self.grow(j);
        }
        let at = self.start[j] + self.len[j];
        self.data[at] = x;
        self.len[j] += 1;
    }

    #[cold]
    fn grow(&mut self, j: usize) {
        let (s, l, c) = (self.start[j], self.len[j], self.cap[j]);
        let new_cap = (2 * c).max(4);
        if s + c == self.data.len() {
            self.data.resize(s + new_cap, T::default());
        } else {
            let ns = self.data.len();
            self.data.extend_from_within(s..s + l);
            self.data.resize(ns + new_cap, T::default());
            self.start[j] = ns;
        }
        self.cap[j] = new_cap;
    }

    #[inline]
    fn swap_remove(&mut self, j: usize, k: usize) {
        let s = self.start[j];
        let last = s + self.len[j] - 1;
        self.data.swap(s + k, last);
        self.len[j] -= 1;
    }

    #[inline]
    fn clear(&mut self, j: usize) {
        self.len[j] = 0;
    }
}

#[inline]
fn ckey(count: usize, col: usize) -> u64 {
    ((count as u64) << 32) | col as u64
}

#[inline]
fn ckey_count(key: u64) -> usize {
    (key >> 32) as usize
}

#[inline]
fn ckey_col(key: u64) -> usize {
    (key & 0xFFFF_FFFF) as usize
}

/// How many live rows have each count (feral's `RowCounts`).
#[derive(Debug, Clone, Default)]
struct RowCounts {
    hist: Vec<usize>,
    reg: Vec<usize>,
    lo: usize,
}

impl RowCounts {
    fn new(rcnt: &[usize]) -> Self {
        let mut hist = vec![0; rcnt.len() + 1];
        for &c in rcnt {
            hist[c] += 1;
        }
        Self {
            hist,
            reg: rcnt.to_vec(),
            lo: 0,
        }
    }

    fn set(&mut self, i: usize, c: usize) {
        self.hist[self.reg[i]] -= 1;
        if c >= self.hist.len() {
            self.hist.resize(c + 1, 0);
        }
        self.hist[c] += 1;
        self.reg[i] = c;
        self.lo = self.lo.min(c);
    }

    fn remove(&mut self, i: usize) {
        self.hist[self.reg[i]] -= 1;
    }

    fn min(&mut self) -> Option<usize> {
        while self.lo < self.hist.len() && self.hist[self.lo] == 0 {
            self.lo += 1;
        }
        (self.lo < self.hist.len()).then_some(self.lo)
    }
}

/// Multiplicative hash for row indices (the default SipHash costs more than
/// the scan it replaces on a column of a few dozen entries).
#[derive(Default, Clone, Copy)]
struct RowHasher(u64);

impl Hasher for RowHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_usize(&mut self, n: usize) {
        self.0 = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// Row → position within one column's list.
type RowMap = HashMap<usize, usize, BuildHasherDefault<RowHasher>>;

#[cfg(test)]
thread_local! {
    /// Test-only: rank-1 updates that took the indexed path, so a test can
    /// show it reached that branch.
    pub(crate) static INDEXED_UPDATES: Cell<u64> = const { Cell::new(0) };
    /// Test-only: searches that stopped on a parked entry
    /// (`Parked::stops_before`), and entries parked.
    pub(crate) static PARKED_STOPS: Cell<u64> = const { Cell::new(0) };
    pub(crate) static PARKED: Cell<u64> = const { Cell::new(0) };
}

/// Columns at least this long carry a [`RowMap`].
const INDEX_MIN_LEN: usize = 32;

/// The active submatrix by column, with a row → position index on the long
/// columns.
///
/// feral finds an entry `(i, j)` by scanning column `j`. A column of `J`ᵀ
/// for a row that touches many variables, or an `x`-column for a variable in
/// many rows (a free final time, a design parameter, a long aggregate), is
/// long, and every pivot on a row it meets scans it twice, to find the
/// pivot-row entry and again to update it: `O(len)` per pivot, quadratic over
/// the elimination. Measured on a recurrence Jacobian of 98 862 rows and
/// 90 008 columns with long aggregate rows (`d = 188 870`), 2.6·10⁹ of the
/// 3.4·10⁹ entry operations were those two scans. The index makes both a
/// lookup. It changes where an entry is found, never the list's element
/// order, which the pivot search's tie-breaks read: every removal is still a
/// `swap_remove` and happens in the order feral's walk performs it (see the
/// update).
struct Active {
    cols: Lists<(usize, f64)>,
    idx: Vec<Option<RowMap>>,
}

impl Active {
    fn index(&mut self, j: usize) {
        let mut map = RowMap::default();
        map.reserve(2 * self.cols.len[j]);
        for (k, &(i, _)) in self.cols.get(j).iter().enumerate() {
            map.insert(i, k);
        }
        self.idx[j] = Some(map);
    }

    /// Position of row `i` in column `j`.
    fn find(&self, j: usize, i: usize) -> Option<usize> {
        match &self.idx[j] {
            Some(map) => map.get(&i).copied(),
            None => self.cols.get(j).iter().position(|&(r, _)| r == i),
        }
    }

    fn push(&mut self, j: usize, x: (usize, f64)) {
        self.cols.push(j, x);
        let k = self.cols.len[j] - 1;
        match &mut self.idx[j] {
            Some(map) => {
                map.insert(x.0, k);
            }
            None if k + 1 >= INDEX_MIN_LEN => self.index(j),
            None => {}
        }
    }

    fn swap_remove(&mut self, j: usize, k: usize) {
        if let Some(map) = &mut self.idx[j] {
            let last = self.cols.len[j] - 1;
            map.remove(&self.cols.get(j)[k].0);
            if k != last {
                map.insert(self.cols.get(j)[last].0, k);
            }
        }
        self.cols.swap_remove(j, k);
    }

    fn clear(&mut self, j: usize) {
        self.cols.clear(j);
        self.idx[j] = None;
    }
}

/// Heap entries of numerically empty columns, held out of the heap.
///
/// A column whose every entry is at or below `ztol` is never a pivot
/// candidate: the search pops it, skips it and pushes it back, every step,
/// until an update changes it. With a dependent row its partner column in `M`
/// goes numerically empty early and, having a low count, sits at the top of
/// the heap, so each of the `r` dependent rows costs every remaining pivot
/// step a pop, a scan and a push. Measured on a recurrence Jacobian of
/// 98 862 rows and 90 008 columns (8 854 of them dependent, `d = 188 870`):
/// 31 of 36 s were spent in that search.
///
/// Parking such an entry instead changes nothing feral's search decides.
/// Popping and re-pushing a valid key leaves the heap's multiset as it was;
/// a skipped column takes no part in the pivot choice or the `max_search`
/// count; and the two things the parked entries still decide are kept:
///
/// * **the early stop.** feral checks `(c − 1)(min r − 1) > best` on each
///   valid entry it peeks, parked columns included, and the check precedes
///   the discarding of the stale entries behind them. So before each heap
///   entry is peeked, the parked keys passed since the previous one are
///   checked against the current best; the bound is monotone in `c`, so the
///   largest of them decides. See [`Parked::stops_before`].
/// * **the singular column.** When no pivot is found, `PerturbToEps` takes the
///   smallest numerically empty column the search saw; with no candidate
///   there is no stop, so it saw all of them, parked ones included.
///
/// An entry is valid only while its column is untouched, so a column leaves
/// the parked set (its keys going back into the heap, valid or stale exactly
/// as they would be there) before an update changes it, and is dropped when
/// it is eliminated.
struct Parked {
    keys: BTreeSet<u64>,
    copies: Vec<u32>,
    cols: BTreeSet<usize>,
}

impl Parked {
    fn new(m: usize) -> Self {
        Self {
            keys: BTreeSet::new(),
            copies: vec![0; m],
            cols: BTreeSet::new(),
        }
    }

    fn park(&mut self, key: u64) {
        let j = ckey_col(key);
        self.copies[j] += 1;
        self.keys.insert(key);
        self.cols.insert(j);
    }

    /// Take column `j` out, its keys going back into `heap` when `push_back`
    /// (an update is about to change it) or nowhere (it is being eliminated:
    /// its keys are stale for good, as feral's heap discards them). `key` is
    /// its key while parked, so this is called before `ccnt[j]` changes.
    fn unpark(&mut self, key: u64, heap: &mut BinaryHeap<Reverse<u64>>, push_back: bool) {
        let j = ckey_col(key);
        let n = std::mem::take(&mut self.copies[j]);
        if n == 0 {
            return;
        }
        self.keys.remove(&key);
        self.cols.remove(&j);
        if push_back {
            for _ in 0..n {
                heap.push(Reverse(key));
            }
        }
    }

    /// Whether feral's search, now at heap key `next` (or at the end of the
    /// heap, `u64::MAX`) having last been at `last`, would have stopped on a
    /// parked entry in between: `(c − 1)(min r − 1) > best` for the largest
    /// parked key in `(last, next)`.
    fn stops_before(&self, last: Option<u64>, next: u64, minr: usize, best_cost: usize) -> bool {
        // Keys come off the heap in order, and duplicates repeat one.
        if last.is_some_and(|l| l >= next) {
            return false;
        }
        let lo = last.map_or(std::ops::Bound::Unbounded, std::ops::Bound::Excluded);
        self.keys
            .range((lo, std::ops::Bound::Excluded(next)))
            .next_back()
            .is_some_and(|&k| ckey_count(k).saturating_sub(1) * minr.saturating_sub(1) > best_cost)
    }
}

/// Lowest-numbered `i ≥ *cursor` with `alive[i]`, advancing the cursor past
/// the dead prefix. Valid because entries only ever go from alive to dead.
fn first_alive(alive: &[bool], cursor: &mut usize) -> Option<usize> {
    while *cursor < alive.len() && !alive[*cursor] {
        *cursor += 1;
    }
    (*cursor < alive.len()).then_some(*cursor)
}

/// Steps between deadline checks when little work is being done per step.
const DEADLINE_CHECK_STEPS: usize = 256;
/// Work between deadline checks.
const DEADLINE_CHECK_WORK: u64 = 1 << 20;

/// Threshold-Markowitz elimination of `a` at `u_thresh` / `max_search` with
/// singular pivots perturbed to `abs_floor` — `SparseLu::factor_markowitz` at
/// `LuScaling::None` and `LuSingularAction::PerturbToEps { abs_floor }` —
/// returning only the pivot rows and values, within `budget`.
pub(crate) fn markowitz_pivots(
    a: &SparseColMatrix,
    u_thresh: f64,
    max_search: usize,
    zero_pivot_tol: f64,
    abs_floor: f64,
    budget: &DependentRowsBudget,
) -> Result<Pivots, ProbeError> {
    let m = a.m;
    if m > u32::MAX as usize {
        return Err(ProbeError::Failed);
    }
    let max_bytes = budget
        .max_active_bytes
        .unwrap_or_else(|| default_max_active_bytes(a.nnz()));
    let deadline = budget.deadline;
    let over_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    if over_deadline() {
        return Err(ProbeError::OverBudget(ProbeBailout::Deadline));
    }

    let mut a_max = 0.0_f64;
    for &v in a.values.iter() {
        a_max = a_max.max(v.abs());
    }
    let ztol = zero_pivot_tol * a_max;

    // ---- active submatrix ------------------------------------------------
    let mut ccnt = vec![0usize; m];
    let mut rcnt = vec![0usize; m];
    for (j, cj) in ccnt.iter_mut().enumerate() {
        for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
            if a.values[idx] != 0.0 {
                *cj += 1;
                rcnt[a.row_idx[idx]] += 1;
            }
        }
    }
    let mut cols = Active {
        cols: Lists::with_capacities(&ccnt),
        idx: vec![None; m],
    };
    let mut row_cols: Lists<usize> = Lists::with_capacities(&rcnt);
    for j in 0..m {
        for idx in a.col_ptr[j]..a.col_ptr[j + 1] {
            let (i, v) = (a.row_idx[idx], a.values[idx]);
            if v != 0.0 {
                cols.push(j, (i, v));
                row_cols.push(i, j);
            }
        }
    }
    let mut rm_mark = vec![false; m];
    let mut rm_pos: Vec<usize> = Vec::new();
    let mut alive_r = vec![true; m];
    let mut alive_c = vec![true; m];
    let (mut first_r, mut first_c) = (0usize, 0usize);

    let mut col_singletons: Vec<usize> = (0..m).filter(|&j| ccnt[j] == 1).collect();
    let mut row_singletons: Vec<usize> = (0..m).filter(|&i| rcnt[i] == 1).collect();

    // The heap's *multiset* is part of the pivot rule (feral's comment: a
    // search that examined each column once changed which rows pounce's probe
    // declared dependent and made QGFRDXPN cycle), so it is reproduced exactly:
    // logged during the singleton peel, heapified when the stacks first run dry.
    let mut heaps_built = false;
    let mut cheap_log: Vec<u64> = (0..m).map(|j| ckey(ccnt[j], j)).collect();
    let mut row_counts = RowCounts::new(&rcnt);
    let mut cheap: BinaryHeap<Reverse<u64>> = BinaryHeap::new();
    let mut side = Parked::new(m);

    let mut lscat = vec![0.0_f64; m];
    let mut lmark = vec![false; m];
    let mut seen_mark = vec![false; m];
    let mut seen_list: Vec<usize> = Vec::new();
    let mut row_seen = vec![false; m];
    let mut dirty_rows: Vec<usize> = Vec::new();
    let mut dirty_mark = vec![false; m];
    let mut lvec: Vec<(usize, f64)> = Vec::new();
    let mut uvec: Vec<(usize, f64)> = Vec::new();
    let mut upos: Vec<usize> = Vec::new();
    let mut scanned: Vec<usize> = Vec::new();

    let mut perm = Vec::with_capacity(m);
    let mut value = Vec::with_capacity(m);

    let mut work: u64 = a.nnz() as u64;
    let mut checked_at_work = work;
    let mut checked_at_step = 0usize;

    for step in 0..m {
        // ---- budget ------------------------------------------------------
        if work > budget.max_work {
            return Err(ProbeError::OverBudget(ProbeBailout::Work));
        }
        let bytes = cols.cols.data.len() * std::mem::size_of::<(usize, f64)>()
            + row_cols.data.len() * std::mem::size_of::<usize>();
        if bytes > max_bytes {
            return Err(ProbeError::OverBudget(ProbeBailout::Fill));
        }
        if deadline.is_some()
            && (step - checked_at_step >= DEADLINE_CHECK_STEPS
                || work - checked_at_work >= DEADLINE_CHECK_WORK)
        {
            if over_deadline() {
                return Err(ProbeError::OverBudget(ProbeBailout::Deadline));
            }
            checked_at_step = step;
            checked_at_work = work;
        }

        // ---- pivot search ------------------------------------------------
        let mut fast: Option<(usize, usize)> = None;
        while let Some(j) = col_singletons.pop() {
            if !alive_c[j] || ccnt[j] != 1 {
                continue;
            }
            let (i, v) = cols.cols.get(j)[0];
            if v.abs() > ztol {
                fast = Some((i, j));
                break;
            }
        }
        if fast.is_none() {
            while let Some(i) = row_singletons.pop() {
                if !alive_r[i] || rcnt[i] != 1 {
                    continue;
                }
                let mut found = None;
                for &j in row_cols.get(i).iter() {
                    work += 1;
                    if !alive_c[j] {
                        continue;
                    }
                    work += 1;
                    if let Some(k) = cols.find(j, i) {
                        found = Some((j, cols.cols.get(j)[k].1));
                        break;
                    }
                }
                if let Some((j, v)) = found {
                    let colmax = cols
                        .cols
                        .get(j)
                        .iter()
                        .fold(0.0_f64, |a, &(_, x)| a.max(x.abs()));
                    if v.abs() > ztol && v.abs() >= u_thresh * colmax {
                        fast = Some((i, j));
                        break;
                    }
                }
            }
        }

        if fast.is_none() && !heaps_built {
            let v: Vec<Reverse<u64>> = cheap_log
                .drain(..)
                .filter(|&k| alive_c[ckey_col(k)])
                .map(Reverse)
                .collect();
            work += v.len() as u64;
            cheap = BinaryHeap::from(v);
            heaps_built = true;
        }

        let minr = if fast.is_some() {
            1
        } else {
            row_counts.min().map_or(1, |c| c.max(1))
        };

        let mut best: Option<(usize, f64, usize, usize)> = None;
        let mut singular_col: Option<usize> = None;
        scanned.clear();
        let mut examined = 0usize;
        // The last heap key this search peeked: parked keys at or below it
        // have been passed (see `Parked::stops_before`).
        let mut last_key: Option<u64> = None;
        while fast.is_none() {
            let Some(&Reverse(key)) = cheap.peek() else {
                // The rest of the walk is parked entries: with a candidate in
                // hand it stops among them or runs out, either way with this
                // candidate; with none it passes all of them.
                if best.is_none()
                    && let Some(&j) = side.cols.first()
                {
                    singular_col = Some(singular_col.map_or(j, |p: usize| p.min(j)));
                }
                break;
            };
            if let Some((bc, _, _, _)) = best
                && side.stops_before(last_key, key, minr, bc)
            {
                #[cfg(test)]
                PARKED_STOPS.with(|c| c.set(c.get() + 1));
                break;
            }
            last_key = Some(key);
            let (c, j) = (ckey_count(key), ckey_col(key));
            if !alive_c[j] || ccnt[j] != c {
                cheap.pop();
                continue;
            }
            if let Some((bc, _, _, _)) = best
                && c.saturating_sub(1) * minr.saturating_sub(1) > bc
            {
                break;
            }
            cheap.pop();
            work += 1 + cols.cols.len[j] as u64;

            let colmax = cols
                .cols
                .get(j)
                .iter()
                .fold(0.0_f64, |acc, &(_, v)| acc.max(v.abs()));
            if colmax <= ztol {
                singular_col = Some(singular_col.map_or(j, |p: usize| p.min(j)));
                side.park(key);
                #[cfg(test)]
                PARKED.with(|c| c.set(c.get() + 1));
                continue;
            }
            scanned.push(j);
            let bar = u_thresh * colmax;
            for &(i, v) in cols.cols.get(j).iter() {
                if v.abs() < bar {
                    continue;
                }
                let cost = rcnt[i].saturating_sub(1) * c.saturating_sub(1);
                let better = match best {
                    None => true,
                    Some((bc, bv, _, _)) => cost < bc || (cost == bc && v.abs() > bv),
                };
                if better {
                    best = Some((cost, v.abs(), i, j));
                }
            }
            examined += 1;
            if let Some((0, _, _, _)) = best {
                break;
            }
            if examined >= max_search && best.is_some() {
                break;
            }
        }
        for &j in scanned.iter() {
            if alive_c[j] {
                cheap.push(Reverse(ckey(ccnt[j], j)));
            }
        }

        let (pi, pj) = match fast.or(best.map(|(_, _, i, j)| (i, j))) {
            Some((i, j)) => (i, j),
            None => {
                // Nothing acceptable anywhere: perturb, as `PerturbToEps` does.
                let col = singular_col
                    .or_else(|| first_alive(&alive_c, &mut first_c))
                    .unwrap_or(0);
                let Some(row) = first_alive(&alive_r, &mut first_r) else {
                    return Err(ProbeError::Failed);
                };
                let mag = abs_floor.max(f64::MIN_POSITIVE);
                // `col` is eliminated at this step: its parked keys go stale.
                side.unpark(ckey(ccnt[col], col), &mut cheap, false);
                match cols.find(col, row) {
                    Some(k) => {
                        let e = &mut cols.cols.get_mut(col)[k];
                        e.1 = if e.1 < 0.0 { -mag } else { mag };
                    }
                    None => {
                        cols.push(col, (row, mag));
                        row_cols.push(row, col);
                        rcnt[row] += 1;
                        ccnt[col] += 1;
                        cheap.push(Reverse(ckey(ccnt[col], col)));
                        row_counts.set(row, rcnt[row]);
                    }
                }
                (row, col)
            }
        };

        let pv = cols
            .find(pj, pi)
            .map(|k| cols.cols.get(pj)[k].1)
            .ok_or(ProbeError::Failed)?;

        // ---- gather the pivot column and pivot row -------------------------
        lvec.clear();
        let inv = 1.0 / pv;
        for &(i, v) in cols.cols.get(pj).iter() {
            if i != pi {
                lvec.push((i, v * inv));
            }
        }
        uvec.clear();
        upos.clear();
        for &j in row_cols.get(pi).iter() {
            if j == pj || !alive_c[j] || row_seen[j] {
                continue;
            }
            row_seen[j] = true;
            work += 1;
            if let Some(k) = cols.find(j, pi) {
                uvec.push((j, cols.cols.get(j)[k].1));
                upos.push(k);
            }
        }
        for &j in row_cols.get(pi).iter() {
            row_seen[j] = false;
        }

        alive_r[pi] = false;
        row_counts.remove(pi);
        alive_c[pj] = false;
        perm.push(pi);
        value.push(pv);

        dirty_rows.clear();
        for &(i, _) in lvec.iter() {
            rcnt[i] -= 1;
            if !dirty_mark[i] {
                dirty_mark[i] = true;
                dirty_rows.push(i);
            }
        }
        side.unpark(ckey(ccnt[pj], pj), &mut cheap, false);
        cols.clear(pj);
        row_cols.clear(pi);

        // ---- rank-1 update, column-oriented ------------------------------
        for &(i, lv) in lvec.iter() {
            lscat[i] = lv;
            lmark[i] = true;
        }
        for (&(j, uval), &pos) in uvec.iter().zip(upos.iter()) {
            // Column `j` is about to change: its parked keys rejoin the heap,
            // where feral's search has kept them all along.
            side.unpark(ckey(ccnt[j], j), &mut cheap, true);
            if lvec.is_empty() || uval == 0.0 {
                cols.swap_remove(j, pos);
            } else if cols.idx[j].is_some() {
                // The walk below, without walking: look up the pivot-column
                // rows, update them, then make the removals feral's walk
                // makes, in its order. That walk visits positions upward and,
                // on a removal at `k`, examines the entry `swap_remove` just
                // moved into `k` before going on; an entry only ever moves
                // from the end into the position being removed. So taking the
                // doomed positions in ascending order, and at each one
                // removing while the entry there is doomed, leaves the list as
                // the walk leaves it. Every `lvec` row is already in
                // `dirty_rows` (pushed above), so that order is unchanged too.
                work += 2 * lvec.len() as u64 + 1;
                #[cfg(test)]
                INDEXED_UPDATES.with(|c| c.set(c.get() + 1));
                rm_pos.clear();
                if let Some(k) = cols.find(j, pi) {
                    rm_mark[pi] = true;
                    rm_pos.push(k);
                }
                for &(i, _) in lvec.iter() {
                    let Some(k) = cols.find(j, i) else { continue };
                    let v = cols.cols.get(j)[k].1;
                    let nv = v - uval * lscat[i];
                    seen_mark[i] = true;
                    seen_list.push(i);
                    if nv == 0.0 {
                        rcnt[i] -= 1;
                        rm_mark[i] = true;
                        rm_pos.push(k);
                    } else {
                        cols.cols.get_mut(j)[k].1 = nv;
                    }
                }
                rm_pos.sort_unstable();
                for &p in rm_pos.iter() {
                    while p < cols.cols.len[j] {
                        let r = cols.cols.get(j)[p].0;
                        if !rm_mark[r] {
                            break;
                        }
                        rm_mark[r] = false;
                        cols.swap_remove(j, p);
                    }
                }
                for &(i, lv) in lvec.iter() {
                    if seen_mark[i] {
                        continue;
                    }
                    let nv = -uval * lv;
                    if nv != 0.0 {
                        cols.push(j, (i, nv));
                        rcnt[i] += 1;
                        row_cols.push(i, j);
                    }
                }
                for &i in seen_list.iter() {
                    seen_mark[i] = false;
                }
                seen_list.clear();
            } else {
                work += (cols.cols.len[j] + lvec.len()) as u64;
                let mut k = 0;
                while k < cols.cols.len[j] {
                    let (i, v) = cols.cols.get(j)[k];
                    if i == pi {
                        cols.swap_remove(j, k);
                        continue;
                    }
                    if uval != 0.0 && lmark[i] {
                        let nv = v - uval * lscat[i];
                        seen_mark[i] = true;
                        seen_list.push(i);
                        if nv == 0.0 {
                            rcnt[i] -= 1;
                            if !dirty_mark[i] {
                                dirty_mark[i] = true;
                                dirty_rows.push(i);
                            }
                            cols.swap_remove(j, k);
                            continue;
                        }
                        cols.cols.get_mut(j)[k].1 = nv;
                    }
                    k += 1;
                }
                if uval != 0.0 {
                    for &(i, lv) in lvec.iter() {
                        if seen_mark[i] {
                            continue;
                        }
                        let nv = -uval * lv;
                        if nv != 0.0 {
                            cols.push(j, (i, nv));
                            rcnt[i] += 1;
                            row_cols.push(i, j);
                            if !dirty_mark[i] {
                                dirty_mark[i] = true;
                                dirty_rows.push(i);
                            }
                        }
                    }
                }
                for &i in seen_list.iter() {
                    seen_mark[i] = false;
                }
                seen_list.clear();
            }
            ccnt[j] = cols.cols.len[j];
            if heaps_built {
                cheap.push(Reverse(ckey(ccnt[j], j)));
            } else {
                cheap_log.push(ckey(ccnt[j], j));
            }
            if ccnt[j] == 1 {
                col_singletons.push(j);
            }
        }
        for &(i, _) in lvec.iter() {
            lmark[i] = false;
        }

        for &i in dirty_rows.iter() {
            dirty_mark[i] = false;
            if alive_r[i] {
                if rcnt[i] == 1 {
                    row_singletons.push(i);
                }
                row_counts.set(i, rcnt[i]);
            }
        }
    }

    Ok(Pivots { perm, value })
}

/// Record why the probe stopped (or that it did not), for
/// [`last_probe_bailout`].
pub(crate) fn note_bailout(b: Option<ProbeBailout>) {
    LAST_BAILOUT.with(|c| c.set(b));
}
