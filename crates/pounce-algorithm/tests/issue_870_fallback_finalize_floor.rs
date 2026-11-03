//! pounce#870 — the `mu_strategy_fallback` floor at the **`finalize_solution`
//! boundary**, which is a different sink from the CLI's report.
//!
//! `honor_original_bounds_finalize.rs` states the split this test depends on:
//! "The CLI reads its `.sol` primal from the `on_converged` hook, but every
//! other consumer — `pounce-py`'s `Problem.solve`, the C interface, any Rust
//! `TNLP` — reads it from `finalize_solution`. Those are two separate lifts of
//! the same iterate." A floor therefore has to cover both, and the two halves
//! of pounce#870's fix cover one each:
//!
//! * restoring `self.statistics` fixes the CLI's report —
//!   `pounce-cli/tests/issue_870_mu_fallback_solution_floor.rs` pins that;
//! * `FinalizeSnapshot::replay` fixes the TNLP payload — this file pins that.
//!
//! **Neither test sees the other's half.** Dropping `replay` leaves the whole
//! CLI suite green while `pounce.minimize` goes back to returning the losing
//! retry's point, which is exactly how the defect stayed invisible: the
//! statistics and the finalize payload are written by different code and only
//! one of them was ever asserted.
//!
//! The trigger here is deterministic rather than a stalled schedule.
//! `Maximum_Iterations_Exceeded` is an unconditional retry trigger, so a low
//! `max_iter` guarantees that attempt 1 caps, the flipped-schedule retry runs
//! and also caps — failing to promote — and the floor must put attempt 1's
//! payload back. No exotic model is needed, and nothing depends on a
//! particular numerical stall surviving future retuning.
//!
//! A losing retry leaks into consumer-visible state through THREE sinks. This
//! file owns two of them; `pounce-cli`'s owns the third.
//!
//! MUTATION TABLE — measured, both directions, against both files:
//!
//! | change                                | this file   | the pounce-cli file |
//! |---------------------------------------|-------------|---------------------|
//! | drop `floor.replay(&tnlp)`            | 2 of 4 fail | **green**           |
//! | drop the trace re-emit                | 1 of 4 fails| **green**           |
//! | drop the `SolutionCertificate` restore| **green**   | 3 of 5 fail         |
//!
//! The two greens in that table are the point of having two files. Retiring
//! either one leaves half of the fix unprotected, and the surviving file will
//! not tell you so.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const N: usize = 4;

/// Records every `finalize_solution` payload, in order.
#[derive(Default)]
struct Recording {
    payloads: Vec<(Vec<Number>, Number)>,
    /// Every `IterStats` the solver sent, in order — the trace a consumer
    /// such as the CasADi plugin accumulates for `stats()["iterations"]`.
    trace: Vec<IterStats>,
}

impl TNLP for Recording {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: N as Index,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: N as Index,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for i in 0..N {
            b.x_l[i] = -50.0;
            b.x_u[i] = 50.0;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for i in 0..N {
            sp.x[i] = 20.0 + i as Number;
        }
        true
    }

    // A badly scaled separable quartic: no equality rows, curvature spanning
    // six orders, so the two barrier schedules walk visibly different paths
    // and a low iteration cap lands them in different places.
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut f = 0.0;
        for i in 0..N {
            let s = 10f64.powi(3 * i as i32 - 3);
            f += s * (x[i] - 1.0).powi(4) + s * 0.5 * (x[i] - 1.0).powi(2);
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..N {
            let s = 10f64.powi(3 * i as i32 - 3);
            g[i] = s * 4.0 * (x[i] - 1.0).powi(3) + s * (x[i] - 1.0);
        }
        true
    }

    fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..N {
                    irow[i] = i as Index;
                    jcol[i] = i as Index;
                }
            }
            SparsityRequest::Values { values } => {
                let x = match x {
                    Some(x) => x,
                    None => return false,
                };
                for i in 0..N {
                    let s = 10f64.powi(3 * i as i32 - 3);
                    values[i] = obj_factor * (s * 12.0 * (x[i] - 1.0).powi(2) + s);
                }
            }
        }
        true
    }

    fn intermediate_callback(&mut self, s: IterStats, _d: &IpoptData, _q: &IpoptCq) -> bool {
        self.trace.push(s);
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.payloads.push((sol.x.to_vec(), sol.obj_value));
    }
}

struct Run {
    payloads: Vec<(Vec<Number>, Number)>,
    trace: Vec<IterStats>,
    stats: SolveStatistics,
}

fn run_full(max_iter: i32, fallback: Option<bool>) -> Run {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("max_iter", max_iter, true, false)
        .unwrap();
    if let Some(v) = fallback {
        app.options_mut()
            .set_string_value(
                "mu_strategy_fallback",
                if v { "yes" } else { "no" },
                true,
                false,
            )
            .unwrap();
    }
    app.initialize().unwrap();
    let concrete = Rc::new(RefCell::new(Recording::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let _ = app.optimize_tnlp(tnlp);
    let out = concrete.borrow().payloads.clone();
    assert!(!out.is_empty(), "finalize_solution never ran");
    Run {
        payloads: out,
        trace: concrete.borrow().trace.clone(),
        stats: app.statistics(),
    }
}

fn run(max_iter: i32, fallback: Option<bool>) -> Vec<(Vec<Number>, Number)> {
    run_full(max_iter, fallback).payloads
}

/// The retry must run and must actually displace the answer — otherwise every
/// assertion below would hold vacuously on a build with no floor at all.
#[test]
fn the_retry_really_displaces_the_first_attempts_payload() {
    let p = run(4, None);
    assert!(
        p.len() >= 2,
        "expected at least one retry payload; got {} finalize call(s), so the \
         mu_strategy_fallback retry did not fire and this file proves nothing",
        p.len()
    );
    let differ = p[0]
        .0
        .iter()
        .zip(&p[1].0)
        .any(|(a, b)| (a - b).abs() > 1e-9);
    assert!(
        differ,
        "attempt 1 and the retry produced the same point, so this fixture \
         cannot detect a missing floor: {:?} vs {:?}",
        p[0].0, p[1].0
    );
}

/// The defect: what the user's TNLP is left holding must be the payload of the
/// attempt whose status was returned.
#[test]
fn the_last_finalize_payload_is_the_first_attempts() {
    let p = run(4, None);
    let (first_x, first_obj) = &p[0];
    let (last_x, last_obj) = p.last().unwrap();
    assert!(
        first_x
            .iter()
            .zip(last_x)
            .all(|(a, b)| (a - b).abs() <= 1e-12),
        "the losing retry's point was left in the user's TNLP.\n  attempt 1: \
         {first_x:?}\n  left behind: {last_x:?}\nThe status is floored to \
         attempt 1's, so the point must be too (pounce#870)."
    );
    assert!(
        (first_obj - last_obj).abs() <= 1e-12 * first_obj.abs().max(1.0),
        "objective left behind ({last_obj:e}) is not attempt 1's ({first_obj:e})"
    );
}

/// And the floored answer is the one a build that never retries produces —
/// the floor restores, it does not merely make the two attempts agree.
#[test]
fn the_floored_answer_matches_a_run_with_no_retry_at_all() {
    let floored = run(4, None);
    let no_retry = run(4, Some(false));
    assert_eq!(
        no_retry.len(),
        1,
        "mu_strategy_fallback=no must not retry at all"
    );
    let (a, _) = floored.last().unwrap();
    let (b, _) = &no_retry[0];
    assert!(
        a.iter().zip(b).all(|(p, q)| (p - q).abs() <= 1e-12),
        "floored answer {a:?} differs from the no-retry answer {b:?}"
    );
}

/// The third sink. A consumer accumulates the per-iteration trace itself, so
/// POUNCE cannot rewind it when a retry loses — both attempts concatenate. If
/// the certificate is restored and the trace is left alone, the reported
/// numbers describe attempt 1 while the trace ends on the retry.
///
/// `casadi/test_parity.py` states the invariant this pins: "The final numbers
/// and the end of the trace are the same quantities, and must not come from two
/// different places." That check runs at `max_iter=3`, where the numbers are
/// O(1e-2) — on a converged solve both ends are ~1e-17 and any tolerance passes
/// for the wrong reason — so this test uses a cut-short solve too.
#[test]
fn the_trace_ends_on_the_iterate_the_statistics_describe() {
    let r = run_full(4, None);
    let last = r.trace.last().expect("the callback fired at least once");
    assert!(
        r.stats.final_constr_viol > 1e-4 || r.stats.final_dual_inf > 1e-4,
        "the solve must be cut short for this to discriminate; got \
         inf_pr={:e} inf_du={:e}",
        r.stats.final_constr_viol,
        r.stats.final_dual_inf
    );
    assert!(
        (last.inf_pr - r.stats.final_constr_viol).abs() < 1e-12
            && (last.inf_du - r.stats.final_dual_inf).abs() < 1e-12,
        "the trace ends on a different iterate than the statistics describe.\n  \
         trace[-1]: inf_pr={:e} inf_du={:e}\n  statistics: inf_pr={:e} inf_du={:e}\n\
         A losing retry left its own last row at the end of the trace while the \
         certificate was floored back to the winning attempt (pounce#870).",
        last.inf_pr,
        last.inf_du,
        r.stats.final_constr_viol,
        r.stats.final_dual_inf
    );
}

/// One application, two solves: the second must get the fallback the first
/// did. The retry used to restore `mu_strategy` as a caller-set value, so
/// after one fallback the table read as "the caller chose a strategy" and
/// `is_mu_strategy_fallback_enabled` switched the default-on retry off for
/// every later solve on the same application. The declared-complementarity
/// driver solves every stage on one application; measured on
/// `benchmarks/mpcc` `biactive_positive` (skew, far), a fallback in the
/// Scholtes finish left the exact stage's `Solved_To_Acceptable_Level` to the
/// gh#884 cold retry instead. The gh#884 retry's `perturb_always_cd` restore
/// had the same shape and is held to the same rule.
#[test]
fn a_reused_application_keeps_the_fallback_and_leaves_the_table_as_it_was() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("max_iter", 4, true, false)
        .unwrap();
    app.initialize().unwrap();
    for solve in 1..=2 {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Recording::default()));
        let _ = app.optimize_tnlp(tnlp);
        assert!(
            app.statistics().mu_strategy_fallback.is_some(),
            "solve {solve}: the mu_strategy_fallback retry did not run"
        );
        for tag in ["mu_strategy", "perturb_always_cd"] {
            assert!(
                !matches!(app.options().get_string_value(tag, ""), Ok((_, true))),
                "solve {solve}: `{tag}` reads as caller-set after the solve"
            );
        }
    }
}
