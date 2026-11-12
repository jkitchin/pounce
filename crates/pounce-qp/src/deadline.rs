//! Solve-wide budgets for the active-set solver: a monotonic wall-clock
//! deadline ([`QpOptions::time_limit`]) and a cap on working-set changes
//! ([`QpOptions::max_working_set_changes`]).
//!
//! One budget per *top-level* solve: the nested stages a solve can reach —
//! homotopy, elastic phase-1 and its penalty passes, feasibility/recovery
//! solves, seeded retries — share the outer budget instead of each restarting
//! it. `max_iter`, by contrast, is a per-pass cap that every one of those
//! stages gets afresh.
//!
//! Running out of either budget is the same event to the solver: every check
//! that stops a solve on its deadline — inside the pivot loops and at the
//! decision points between passes, where a truncated pass must not be read as
//! a verdict — stops it on a spent change budget too. Only the reported
//! status differs ([`cut_off_status`]).
//!
//! [`QpOptions::time_limit`]: crate::QpOptions::time_limit
//! [`QpOptions::max_working_set_changes`]: crate::QpOptions::max_working_set_changes

use std::cell::Cell;
use std::time::{Duration, Instant};

use crate::error::QpStatus;

thread_local! {
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
    /// Working-set changes the current top-level solve may still make;
    /// `None` when it has no such cap (or there is no scope).
    static CHANGES_LEFT: Cell<Option<u32>> = const { Cell::new(None) };
    /// When the owning scope started. Used only to report how long a
    /// cancelled solve actually ran (`QpStats::time`), so a timeout is not
    /// recorded as an instantaneous solve.
    static SCOPE_START: Cell<Option<Instant>> = const { Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Test-only: when set, `expired` reports `true` once it has been asked
    /// this many more times, inside a scope or not. Lets a test cut a solve
    /// off at a chosen point instead of racing a wall clock.
    static EXPIRE_AFTER_CHECKS: Cell<Option<u64>> = const { Cell::new(None) };
    /// Test-only: whether a check has reported the `EXPIRE_AFTER_CHECKS`
    /// expiry since it was last armed.
    static EXPIRY_FIRED: Cell<bool> = const { Cell::new(false) };
}

/// Test-only: make the deadline expire on the `n`-th next check (`None`
/// clears it). See `EXPIRE_AFTER_CHECKS`.
#[cfg(test)]
pub(crate) fn expire_after_checks(n: Option<u64>) {
    EXPIRE_AFTER_CHECKS.with(|c| c.set(n));
    EXPIRY_FIRED.with(|c| c.set(false));
}

/// Test-only: whether any check has seen the expiry armed by
/// [`expire_after_checks`] since it was armed — that is, whether the solve
/// was actually cut off rather than finishing inside its check budget.
#[cfg(test)]
pub(crate) fn expiry_fired() -> bool {
    EXPIRY_FIRED.with(Cell::get)
}

pub(crate) struct Guard {
    owner: bool,
}

/// Open the solve-wide scope, unless one is already open — a nested entry
/// point shares the outer solve's budgets and its own `time_limit` and
/// `max_working_set_changes` are ignored.
pub(crate) fn enter(limit: Option<Duration>, max_working_set_changes: Option<u32>) -> Guard {
    let owner = SCOPE_START.with(|start| {
        if start.get().is_some() {
            false
        } else {
            let now = Instant::now();
            DEADLINE.with(|slot| slot.set(limit.and_then(|d| now.checked_add(d))));
            CHANGES_LEFT.with(|slot| slot.set(max_working_set_changes));
            start.set(Some(now));
            true
        }
    });
    Guard { owner }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.owner {
            DEADLINE.with(|slot| slot.set(None));
            CHANGES_LEFT.with(|slot| slot.set(None));
            SCOPE_START.with(|s| s.set(None));
        }
    }
}

#[inline]
pub(crate) fn expired() -> bool {
    #[cfg(test)]
    if let Some(k) = EXPIRE_AFTER_CHECKS.with(Cell::get) {
        if k == 0 {
            EXPIRY_FIRED.with(|c| c.set(true));
            return true;
        }
        EXPIRE_AFTER_CHECKS.with(|c| c.set(Some(k - 1)));
    }
    changes_spent() || time_passed()
}

/// The wall-clock deadline has passed.
fn time_passed() -> bool {
    DEADLINE.with(|slot| {
        slot.get()
            .is_some_and(|deadline| Instant::now() >= deadline)
    })
}

/// The working-set change budget is used up.
fn changes_spent() -> bool {
    CHANGES_LEFT.with(|slot| slot.get() == Some(0))
}

/// Charge one working-set change to the solve-wide budget. Every site that
/// counts a change in `QpStats::n_working_set_changes` calls this, so the
/// budget and the reported count measure the same thing.
#[inline]
pub(crate) fn charge_working_set_change() {
    CHANGES_LEFT.with(|slot| {
        if let Some(left) = slot.get() {
            slot.set(Some(left.saturating_sub(1)));
        }
    });
}

/// The status a solve cut off by [`expired`] reports: `MaxIter` when the
/// working-set change budget ran out (the solve-wide form of `max_iter`), and
/// `TimeLimit` when only the clock did.
pub(crate) fn cut_off_status() -> QpStatus {
    if changes_spent() && !time_passed() {
        QpStatus::MaxIter
    } else {
        QpStatus::TimeLimit
    }
}

/// A pass that ended with `status` was cut off by a solve-wide budget, so its
/// result is not a verdict: it reports `TimeLimit`, or either budget is now
/// spent (a change-budget cut reports `MaxIter`, which a pass also reports
/// when only its own per-pass `max_iter` ran out). Every decision point
/// between passes tests this, never the status alone.
pub(crate) fn cut_off(status: QpStatus) -> bool {
    status == QpStatus::TimeLimit || expired()
}

/// Wall-clock time spent inside the current top-level solve, or
/// [`Duration::ZERO`] when there is no active scope.
pub(crate) fn scope_elapsed() -> Duration {
    SCOPE_START.with(|s| s.get().map_or(Duration::ZERO, |start| start.elapsed()))
}
