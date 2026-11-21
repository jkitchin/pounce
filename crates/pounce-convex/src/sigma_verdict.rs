//! Whether the `σ` cascade certified the answer it returned.
//!
//! gh #880. When cost normalization engages, [`crate::ipm::solve_qp_core`]
//! may try three drivers, ask `normalized_optimum_is_genuine` of each, reject
//! all three, and still return the best of them — which is the right point to
//! keep, but not a certified one. The caller has to be told, because a
//! `QpStatus::Optimal` on an uncertified point is exactly the wrong answer
//! that issue reports.
//!
//! Carried out-of-band rather than on [`crate::QpSolution`]: the struct has
//! 145 construction sites and no `Default`, so a field is a crate-wide edit
//! for one bit. This mirrors [`crate::deadline`], including its
//! outermost-frame-owns-the-slot discipline and RAII reset.
//!
//! The discipline that makes it correct is **clear-at-entry plus
//! record-at-the-one-exit**, and it is worth stating precisely because the
//! module's whole correctness argument is the discipline. [`clear`] runs at
//! the top of every [`crate::ipm::solve_qp_core`] attempt; there is exactly
//! one [`record`] call, on the branch that returns an uncertified pick. Every
//! other exit from the cascade is covered by the `clear` that already ran,
//! not by a `record` of its own — so do not "restore" missing `record` calls,
//! and do not remove the `clear` on the belief that the `record`s cover it.
//! Either edit breaks it in opposite directions.
//!
//! A retry that re-enters the solver clears first and records its own verdict,
//! so the slot describes the last attempt to run — never a stale `true` from
//! an attempt that was superseded.
//!
//! **What the slot does not cover.** It is recorded about the answer
//! `solve_qp_core` returns. [`crate::crossover::maybe_crossover`] can replace
//! that answer afterwards without re-entering the solver, so on a pure LP the
//! verdict describes the interior iterate rather than the purified vertex.
//! That is sound here because the verdict is the forward-error arm's, measured
//! against data the crossover does not change, and because crossover only ever
//! moves to an *exact* vertex — but a future consumer of this bit that is
//! sensitive to which of the two points it describes needs to re-record after
//! crossover rather than assume.

use std::cell::Cell;

thread_local! {
    /// `None` outside a tracked solve; `Some(true)` once an attempt has
    /// returned an answer the `σ` cascade declined to certify.
    static SLOT: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Run `f` as the outermost convex solve, reporting alongside its result
/// whether the `σ` cascade declined to certify the answer being returned.
///
/// Nested calls (the un-normalized re-solve and the direct-driver fallback
/// both re-enter the solver) see the slot already installed and leave
/// ownership to the outer frame, which is what makes the outer cascade's
/// verdict — recorded last, after every inner solve has finished — the one
/// that survives.
pub(crate) fn tracking<T>(f: impl FnOnce() -> T) -> (T, bool) {
    SLOT.with(|slot| {
        if slot.get().is_some() {
            return (f(), false);
        }
        slot.set(Some(false));
        struct Reset<'a>(&'a Cell<Option<bool>>);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.set(None);
            }
        }
        let _reset = Reset(slot);
        let out = f();
        (out, slot.get() == Some(true))
    })
}

/// Record this attempt's verdict, overwriting any earlier one.
///
/// **Debug-asserts that a frame is installed**, because the failure mode
/// without one is silent: `record` would no-op and an uncertified pick would
/// go back under a bare `Optimal`, which is gh #880 verbatim on that path.
/// The assertion turns "every entry that reaches the cascade installs
/// [`tracking`]" from a convention someone has to remember into an invariant
/// the test suite enforces — a new entry, or an existing one that starts
/// running the cascade because HSDE was enabled on its route, fails loudly
/// the first time a test drives it.
///
/// Only the cascade's own verdict goes through here. [`clear`] deliberately
/// does not, since it runs at the top of *every* `solve_qp_core` attempt,
/// including the many that are reached without a frame and never touch the
/// cascade at all.
pub(crate) fn record(uncertified: bool) {
    SLOT.with(|slot| {
        debug_assert!(
            slot.get().is_some(),
            "the σ cascade recorded a verdict with no frame installed: the \
             calling entry point needs `sigma_verdict::tracking` (gh #880)"
        );
        if slot.get().is_some() {
            slot.set(Some(uncertified));
        }
    });
}

/// Reset at the top of a solve attempt, so a retry cannot inherit a verdict
/// about an answer it is replacing.
///
/// Unlike [`record`] this is reached constantly without a frame — every
/// non-HSDE convex solve runs it — so it asserts nothing and simply no-ops.
pub(crate) fn clear() {
    SLOT.with(|slot| {
        if slot.get().is_some() {
            slot.set(Some(false));
        }
    });
}

thread_local! {
    /// Whether the last `σ`-demotion site to run on this thread relabelled
    /// its answer `Optimal` -> `OptimalInaccurate`. Written by every call of
    /// `ipm::demote_uncertified_sigma_optimum`. A nested entry's call runs
    /// before its outer frame's, so once an entry point returns the slot
    /// describes the answer it returned. Read and cleared by
    /// [`take_last_sigma_demotion`].
    static LAST_DEMOTED: Cell<bool> = const { Cell::new(false) };
}

/// Record whether a `σ`-demotion site relabelled its answer.
pub(crate) fn note_demotion(demoted: bool) {
    LAST_DEMOTED.with(|c| c.set(demoted));
}

/// Did the last convex IPM solve on this thread return `OptimalInaccurate`
/// because the `σ` cascade (gh #880; the gh #414 cost-normalization path)
/// declined to certify it, rather than for another reason? Clears the slot,
/// so call it once before a solve to drop a stale value, and once after.
///
/// For a caller that must tell the two kinds of reduced-accuracy answer
/// apart without a field on [`crate::QpSolution`]; this module's doc says why
/// it has none. The CLI's LP / convex-QP → NLP fallback reads it. It reports
/// an in-tol reduced-accuracy answer without the NLP re-solve, but not a
/// `σ`-uncertified one: that demotion is about forward error, which the
/// relative KKT reading the skip relies on does not see.
///
/// It describes the answer the entry point *returned*. A caller that runs
/// further solves (a postsolve retry, a certificate re-solve) must read it
/// right after the solve whose answer it keeps.
pub fn take_last_sigma_demotion() -> bool {
    LAST_DEMOTED.with(|c| c.replace(false))
}
