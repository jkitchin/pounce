//! gh #434 — the homotopy's ratio-test selection rule.
//!
//! The rule is two claims: the step stops at the **earliest** crossing, and
//! **every** crossing coincident with it fires. Both were false, and both
//! failure modes are silent — the path keeps running, having stepped over a
//! constraint, and the violation it leaves behind is absorbing. The primal
//! ratio test can only prevent a violation, never repair one, so a row crossed
//! once stays crossed and drifts further out for the rest of the path
//! (`QSHARE2B` row 7: `8e-2 -> 0.4 -> 7.5 -> 11 -> 22`).
//!
//! What made it worth testing here rather than through a solve: the rule is
//! pure arithmetic on two numbers, but reaching it through `trace_path` costs a
//! KKT factorization per step, and the instances that expose it are
//! Maros-Mészáros-sized.

use crate::homotopy::{Event, RatioTest};

/// The crossings that fire, in the order the tests offered them.
fn firing(r: &RatioTest) -> Vec<Event> {
    r.firing().collect()
}

/// The measured #434 case, in the arithmetic that produced it: `QSHARE2B` row
/// 132 crosses at `dt = 2.9e-16` while another row crosses at `1.1e-14`.
///
/// Both are far below the old `T_EPS = 1e-12` comparison margin, so the earlier
/// one could not displace the later one and the step overshot it. The margin is
/// gone; the earliest crossing has to win no matter how thin the margin.
#[test]
fn earliest_crossing_wins_below_the_old_t_eps_margin() {
    let mut r = RatioTest::new(0.5);
    r.admit(1.1e-14, Event::AddRowUpper(7));
    r.admit(2.9e-16, Event::AddRowLower(132));

    assert_eq!(
        firing(&r),
        vec![Event::AddRowLower(132)],
        "row 132 must win"
    );
    assert!(
        r.t_next <= 0.5 + 2.9e-16,
        "step must stop at the earlier crossing, not overshoot it: t_next = {}",
        r.t_next
    );
}

/// Order must not decide it: offering the earlier crossing first has to give
/// the same answer as offering it second.
#[test]
fn earliest_crossing_wins_regardless_of_offer_order() {
    let mut first = RatioTest::new(0.5);
    first.admit(2.9e-16, Event::AddRowLower(132));
    first.admit(1.1e-14, Event::AddRowUpper(7));

    let mut second = RatioTest::new(0.5);
    second.admit(1.1e-14, Event::AddRowUpper(7));
    second.admit(2.9e-16, Event::AddRowLower(132));

    assert_eq!(firing(&first), vec![Event::AddRowLower(132)]);
    assert_eq!(firing(&second), vec![Event::AddRowLower(132)]);
}

/// A degenerate vertex: several rows bind at the same parameter value. All of
/// them must enter the working set. Firing one and leaving the rest sitting
/// exactly on their bounds is what the next direction then pushes them across.
#[test]
fn every_coincident_crossing_fires() {
    let mut r = RatioTest::new(0.25);
    r.admit(0.125, Event::AddRowUpper(1));
    r.admit(0.125, Event::AddRowUpper(2));
    r.admit(0.125, Event::AddRowLower(3));
    // Not coincident — an order of magnitude later, and it must not fire.
    r.admit(0.5, Event::AddRowUpper(4));

    assert_eq!(
        firing(&r),
        vec![
            Event::AddRowUpper(1),
            Event::AddRowUpper(2),
            Event::AddRowLower(3)
        ]
    );
    assert!((r.t_next - 0.375).abs() < 1e-15, "t_next = {}", r.t_next);
}

/// The two ratio tests compete for the same step, so a vanishing multiplier
/// coincident with a binding row fires alongside it rather than instead of it.
#[test]
fn primal_and_dual_events_can_fire_together() {
    let mut r = RatioTest::new(0.0);
    r.admit(0.4, Event::AddRowUpper(9));
    r.admit(0.4, Event::DropRow(2));
    r.admit(0.4, Event::DropBound(5));

    assert_eq!(
        firing(&r),
        vec![
            Event::AddRowUpper(9),
            Event::DropRow(2),
            Event::DropBound(5)
        ]
    );
}

/// A crossing a hair *behind* the current `t` is a row already marginally past
/// its bound. It binds now, with a zero-length step, rather than being ignored
/// — and it displaces a genuine forward crossing, because it is earlier.
#[test]
fn marginally_negative_crossing_binds_at_the_current_t() {
    let mut r = RatioTest::new(0.75);
    r.admit(0.1, Event::AddRowUpper(1));
    r.admit(-1e-13, Event::AddRowLower(2));

    assert_eq!(firing(&r), vec![Event::AddRowLower(2)]);
    assert_eq!(r.t_next, 0.75, "the step must not move backwards");
}

/// A row further behind than that is already violated, and the ratio test has
/// no way to bring it back — admitting it would move `t` backwards. It is
/// dropped, and the report in `trace_path` is what surfaces it.
#[test]
fn unrecoverably_violated_crossing_is_not_an_event() {
    let mut r = RatioTest::new(0.75);
    r.admit(-1e-3, Event::AddRowLower(2));
    assert!(firing(&r).is_empty());
    assert_eq!(
        r.t_next, 1.0,
        "no event ⇒ the step runs to the end of the path"
    );
}

/// A crossing past the end of the path is not on this path.
#[test]
fn crossing_beyond_t_equals_one_is_not_an_event() {
    let mut r = RatioTest::new(0.9);
    r.admit(0.3, Event::AddRowUpper(1));
    assert!(firing(&r).is_empty());
    assert_eq!(r.t_next, 1.0);
}

/// `restart` has to clear the previous step's winners as well as its `t_next`;
/// a leaked winner would fire an event at a parameter value the path has
/// already passed.
#[test]
fn restart_clears_the_previous_step() {
    let mut r = RatioTest::new(0.1);
    r.admit(0.05, Event::AddRowUpper(1));
    assert_eq!(firing(&r).len(), 1);

    r.restart(0.2);
    assert!(firing(&r).is_empty());
    assert_eq!(r.t_next, 1.0);

    r.admit(0.3, Event::DropRow(4));
    assert_eq!(firing(&r), vec![Event::DropRow(4)]);
    assert!((r.t_next - 0.5).abs() < 1e-15, "t_next = {}", r.t_next);
}

// ---------------------------------------------------------------------------
// gh #602 — bound-adding events.
//
// Until #602 the `Event` enum had no way to say "an inactive variable bound
// became active": rows could be added and dropped and bounds could be
// *dropped*, but the primal ratio test looped over general rows only. So
// `x(t)` crossed inactive variable bounds with nothing capping the step, on the
// cold arm as much as the warm one, and `worst_path_violation` skipped the box
// so nothing reported it either.
//
// These pin the selection rule for the new events. The rule itself is shared
// with the rows — earliest wins, coincident set fires together — so what is
// worth testing here is that a bound crossing *competes on the same footing*
// as a row crossing rather than being ignored or always losing.
// ---------------------------------------------------------------------------

/// A bound crossing earlier than every row crossing has to win the step. This
/// is the case that previously could not exist: the step ran to the row and
/// walked the variable straight through its bound.
#[test]
fn an_earlier_bound_crossing_beats_a_later_row_crossing() {
    let mut r = RatioTest::new(0.0);
    r.admit(0.4, Event::AddRowUpper(3));
    r.admit(0.1, Event::AddBoundUpper(7));

    assert_eq!(
        firing(&r),
        vec![Event::AddBoundUpper(7)],
        "the bound binds first and must cap the step"
    );
    assert!(
        (r.t_next - 0.1).abs() < 1e-15,
        "t_next = {}, expected 0.1",
        r.t_next
    );
}

/// The converse, so the new events cannot simply pre-empt everything: a row
/// that binds first still wins.
#[test]
fn an_earlier_row_crossing_still_beats_a_later_bound_crossing() {
    let mut r = RatioTest::new(0.0);
    r.admit(0.6, Event::AddBoundLower(2));
    r.admit(0.25, Event::AddRowLower(1));

    assert_eq!(firing(&r), vec![Event::AddRowLower(1)]);
}

/// A bound and a row binding at the same `t` are a degenerate vertex, and both
/// have to enter the working set. Firing only one leaves the other sitting
/// exactly on a bound it is not in the working set for, which the next
/// direction then pushes it across — the #434 mechanism, now reachable through
/// the box as well as through the rows.
#[test]
fn a_coincident_bound_and_row_both_fire() {
    let mut r = RatioTest::new(0.25);
    r.admit(0.125, Event::AddRowUpper(4));
    r.admit(0.125, Event::AddBoundLower(9));

    let fired = firing(&r);
    assert_eq!(fired.len(), 2, "both must fire, got {fired:?}");
    assert!(fired.contains(&Event::AddRowUpper(4)), "{fired:?}");
    assert!(fired.contains(&Event::AddBoundLower(9)), "{fired:?}");
}

/// A variable already a hair past its bound is admitted and clamped to `t`, so
/// it binds now with a zero-length step rather than being skipped — the same
/// treatment `admit` gives a row, and the reason is the same: a crossing the
/// ratio test declines to cap is never repaired.
#[test]
fn a_bound_just_past_its_crossing_still_binds() {
    let mut r = RatioTest::new(0.3);
    r.admit(-1e-13, Event::AddBoundUpper(5));

    assert_eq!(firing(&r), vec![Event::AddBoundUpper(5)]);
    assert!(
        (r.t_next - 0.3).abs() < 1e-15,
        "must clamp to the current t, got {}",
        r.t_next
    );
}
