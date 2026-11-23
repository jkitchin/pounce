//! Solve-wide monotonic wall-clock deadline support.

use std::cell::Cell;
use std::time::{Duration, Instant};

thread_local! {
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
}

pub(crate) fn with_deadline<T>(limit: Option<Duration>, f: impl FnOnce() -> T) -> T {
    DEADLINE.with(|slot| {
        if slot.get().is_some() {
            return f();
        }
        // The outermost scope is the start of a top-level convex solve: a
        // FERAL budget refusal latched by an earlier solve on this thread is
        // not this solve's (gh#1249). Nested scopes keep it, so a refusal
        // stays terminal for the solve that hit it.
        pounce_feral::clear_budget_refusal();
        let now = Instant::now();
        let deadline = limit.and_then(|d| now.checked_add(d));
        slot.set(deadline);
        struct Reset<'a>(&'a Cell<Option<Instant>>);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.set(None);
            }
        }
        let _reset = Reset(slot);
        // gh#1249: the same deadline stops FERAL's symbolic pre-analysis
        // from going on into the numeric factor once it has passed.
        let _feral = pounce_feral::factor_deadline(deadline);
        f()
    })
}

#[inline]
pub(crate) fn expired() -> bool {
    DEADLINE.with(|slot| {
        slot.get()
            .is_some_and(|deadline| Instant::now() >= deadline)
    })
}

pub(crate) fn remaining() -> Option<Duration> {
    DEADLINE.with(|slot| {
        slot.get()
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    })
}
