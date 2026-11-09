//! gh#1147 — `IpoptSolve` on a problem with declared complementarity pairs.
//!
//! With pairs declared, `IpoptSolve` hands the model to
//! `pounce_restoration::mpcc_driver::solve_mpcc`, as the CLI and Python do;
//! without them it takes the path it always took. Every branch the routing
//! takes has its own test (CLAUDE.md, "A test is only evidence about the
//! branch it reaches"):
//!
//! * pairs declared: the driver runs and its reported point comes back;
//! * no pairs, and pairs declared then cleared: the old path, pinned to the
//!   status, iteration count and objective measured on the commit before
//!   the routing existed;
//! * the refusals: a session (`IpoptCreateSolver`), a staged SQP working
//!   set, and `IpoptSolveWarmStart` with an input working set;
//! * the live inspectors during a lowered solve, and the stop request;
//! * each `complementarity_*` option reaching the driver, read off the
//!   total iterations the intermediate callback sees.
//!
//! The model is `QpecSmall`, whose answer `(1, 1, 0)` is known. Without its
//! pairs it is a plain NLP with the same minimizer (the unconstrained one is
//! feasible), so "the driver ran" is told apart by what the driver does — a
//! sequence of lowered solves — not by the answer.
//!
//! Two references. The **answer** is held against the CLI's on
//! `crates/pounce-cli/tests/fixtures_mpcc/qpec_small.nl` (`1e-6`, the CLI
//! test's tolerance), from the `.nl` file's own start `(0, 0.5, 0.5)`. The
//! **trajectory** — point to the bit, iteration counts — is held against
//! Python's `pounce.Problem(complementarity=)` on the same callbacks
//! (`python/tests/test_complementarity.py::QpecSmall`), not against the
//! CLI: from the same start the CLI's run diverges at its fourth lowered
//! solve (191 iterations in all against the callback frontends' 139). Both
//! callback frontends agree with each other exactly, and the CLI's `.nl`
//! model reports structure (linearity, derivative proofs) a callback model
//! cannot, which is where that difference lives; S1-I8 measures parity
//! over the declared-pair corpus.

#[path = "common/qpec.rs"]
mod qpec;

use pounce_cinterface::solver::{IpoptCreateSolver, IpoptFreeSolver};
use pounce_cinterface::*;
use qpec::{G_L, G_U, M, N, NELE_HESS, NELE_JAC, PAIRS, X_L, X_U};
use std::ffi::{CString, c_void};
use std::ptr;

const SOLVE_SUCCEEDED: Index = 0;
const USER_REQUESTED_STOP: Index = 5;
const INVALID_PROBLEM_DEFINITION: Index = -11;

/// What the callbacks saw, threaded through `user_data`.
#[derive(Default)]
struct Probe {
    /// The handle the inspectors are called on (set by the test).
    prob: IpoptProblem,
    eval_f_calls: usize,
    /// Every intermediate-callback fire, and those with `iter_count == 0`
    /// (one per lowered solve: each starts at iteration 0).
    fires: usize,
    starts: usize,
    /// Fires with `alg_mod != 0` (restoration).
    resto_fires: usize,
    /// `GetIpoptCurrentIterate` / `GetIpoptCurrentViolations` results, one
    /// per fire: `[x only, x and g, violations]`.
    inspected: Vec<[Bool; 3]>,
    /// Return `FALSE` from the callback on this fire (1-based); 0 = never.
    stop_on_fire: usize,
}

unsafe fn probe<'a>(u: *mut c_void) -> &'a mut Probe {
    unsafe { &mut *(u as *mut Probe) }
}

unsafe extern "C" fn counting_ev_f(
    n: Index,
    x: *const Number,
    new_x: Bool,
    obj: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        if !u.is_null() {
            probe(u).eval_f_calls += 1;
        }
        qpec::ev_f(n, x, new_x, obj, u)
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn watch(
    alg_mod: Index,
    iter_count: Index,
    _obj: Number,
    _inf_pr: Number,
    _inf_du: Number,
    _mu: Number,
    _d_norm: Number,
    _reg: Number,
    _alpha_du: Number,
    _alpha_pr: Number,
    _ls: Index,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let p = probe(u);
        p.fires += 1;
        if iter_count == 0 && alg_mod == 0 {
            p.starts += 1;
        }
        if alg_mod != 0 {
            p.resto_fires += 1;
        }
        if !p.prob.is_null() {
            p.inspected.push(inspect(p.prob));
        }
        if p.stop_on_fire != 0 && p.fires == p.stop_on_fire {
            return 0;
        }
        1
    }
}

/// `GetIpoptCurrentIterate` (x only; x and g) and `GetIpoptCurrentViolations`
/// on `prob`, in that order.
unsafe fn inspect(prob: IpoptProblem) -> [Bool; 3] {
    unsafe {
        let mut x = [0.0; N];
        let mut g = [0.0; M];
        let mut z = [0.0; N];
        let x_only = GetIpoptCurrentIterate(
            prob,
            0,
            N as Index,
            x.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            M as Index,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        let x_and_g = GetIpoptCurrentIterate(
            prob,
            0,
            N as Index,
            x.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            M as Index,
            g.as_mut_ptr(),
            ptr::null_mut(),
        );
        let viol = GetIpoptCurrentViolations(
            prob,
            0,
            N as Index,
            z.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            M as Index,
            g.as_mut_ptr(),
            ptr::null_mut(),
        );
        [x_only, x_and_g, viol]
    }
}

/// `QpecSmall` (C index style) with the counting objective, quiet.
fn create() -> IpoptProblem {
    let prob = unsafe {
        CreateIpoptProblem(
            N as Index,
            X_L.as_ptr(),
            X_U.as_ptr(),
            M as Index,
            G_L.as_ptr(),
            G_U.as_ptr(),
            NELE_JAC as Index,
            NELE_HESS as Index,
            0,
            Some(counting_ev_f),
            Some(qpec::ev_g),
            Some(qpec::ev_grad_f),
            Some(qpec::ev_jac_g::<0>),
            Some(qpec::ev_h::<0>),
        )
    };
    assert!(!prob.is_null());
    int_opt(prob, "print_level", 0);
    str_opt(prob, "sb", "yes");
    prob
}

fn with_pairs() -> IpoptProblem {
    let prob = create();
    for (r, v) in PAIRS {
        assert_eq!(
            unsafe { AddPounceComplementarityPair(prob, r, v) },
            POUNCE_OK
        );
    }
    prob
}

/// `CString::new`, panicking with the reason on an interior NUL.
fn cstr(s: &str) -> CString {
    CString::new(s).unwrap_or_else(|e| panic!("{s:?}: {e}"))
}

fn str_opt(prob: IpoptProblem, k: &str, v: &str) -> Bool {
    let (k, v) = (cstr(k), cstr(v));
    unsafe { AddIpoptStrOption(prob, k.as_ptr(), v.as_ptr()) }
}

fn int_opt(prob: IpoptProblem, k: &str, v: Index) {
    let k = cstr(k);
    assert_eq!(unsafe { AddIpoptIntOption(prob, k.as_ptr(), v) }, 1);
}

struct Solved {
    status: Index,
    x: [Number; N],
    g: [Number; M],
    obj: Number,
}

/// `IpoptSolve` from `x0 = 0` with every output buffer, `probe` as
/// `user_data`.
fn solve(prob: IpoptProblem, probe: &mut Probe) -> Solved {
    solve_from(prob, probe, START)
}

/// The start in `qpec_small.nl` (its `x` segment), and the one Python's
/// test uses first.
const START: [Number; N] = [0.0, 0.5, 0.5];

fn solve_from(prob: IpoptProblem, probe: &mut Probe, x0: [Number; N]) -> Solved {
    let mut x = x0;
    let mut g = [f64::NAN; M];
    let mut obj = f64::NAN;
    let mut mult_g = [0.0; M];
    let mut zl = [0.0; N];
    let mut zu = [0.0; N];
    let status = unsafe {
        IpoptSolve(
            prob,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            mult_g.as_mut_ptr(),
            zl.as_mut_ptr(),
            zu.as_mut_ptr(),
            probe as *mut Probe as *mut c_void,
        )
    };
    Solved { status, x, g, obj }
}

fn watched(prob: IpoptProblem) {
    assert_eq!(unsafe { SetIntermediateCallback(prob, Some(watch)) }, 1);
}

fn free(prob: IpoptProblem) {
    unsafe { FreeIpoptProblem(prob) }
}

fn assert_answer(s: &Solved) {
    for (got, want) in s.x.iter().zip([1.0, 1.0, 0.0]) {
        assert!((got - want).abs() < 1e-6, "x = {:?}", s.x);
    }
}

/// Python's reported point (`pounce.Problem(complementarity=[(0, 1), (1,
/// 2)])`, `print_level 0`, from [`START`]): the accepted continuation's.
/// Bit-equal here means the bridge received the point the driver reported
/// and no other stage's.
const PY_X: [Number; N] = [
    1.0000000061040115,
    1.0000000030553011,
    5.5115660435885625e-12,
];

#[test]
fn pairs_route_the_solve_through_the_driver() {
    let prob = with_pairs();
    watched(prob);
    let mut p = Probe::default();
    let s = solve(prob, &mut p);
    assert_eq!(s.status, SOLVE_SUCCEEDED);
    assert_answer(&s);
    assert_eq!(
        s.x.map(f64::to_bits),
        PY_X.map(f64::to_bits),
        "x = {:?}",
        s.x
    );
    // `g` and the objective are the user's, at the reported point
    // (Python's `info["g"]` and `info["obj_val"]`).
    assert_eq!(s.g, [6.590727963384779e-12, 6.115034700826527e-09]);
    assert_eq!(s.obj, 4.6593851948345574e-17);
    // The driver's lowered solves, each a fresh solve from iteration 0
    // (ten here: the continuation's rungs and finishes, then the exact
    // product's); a plain solve is one.
    assert_eq!(p.starts, 10, "lowered solves seen by the callback");
    assert_eq!(p.resto_fires, 0);
    // `GetIpoptIterCount` is the reported solve's count (Python's
    // `info["iter_count"]`), not the run's total (its
    // `complementarity["total_iterations"]`).
    let iters = unsafe { GetIpoptIterCount(prob) };
    assert_eq!(iters, 12);
    assert_eq!(p.fires - p.starts, 139, "total iterations, as Python's");
    // The report writes (its `complementarity` block is S1-I7's test).
    let file = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("cc_solve_report.json");
    let path = cstr(&file.to_string_lossy());
    let detail = cstr("full");
    assert_eq!(
        unsafe { IpoptWriteSolveReport(prob, path.as_ptr(), detail.as_ptr()) },
        1
    );
    let text = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file:?}: {e}"));
    let json: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{file:?}: {e}"));
    assert_eq!(json["statistics"]["iteration_count"].as_i64(), Some(12));
    // Pairs persist across solves on the handle, like `nonlinear_vars`.
    assert_eq!(unsafe { GetPounceComplementarityPairCount(prob) }, 2);
    let mut p2 = Probe::default();
    let s2 = solve(prob, &mut p2);
    assert_eq!(s2.status, SOLVE_SUCCEEDED);
    assert_eq!(p2.starts, 10);
    assert_eq!(s2.x, s.x);
    free(prob);
}

/// Status, iteration count and objective of the pair-less solve, measured
/// from `x0 = 0` on the commit before `IpoptSolve` routed pairs
/// (`5aa87107`): the no-pairs path must be exactly the path it was.
const OLD_PATH_STATUS: Index = SOLVE_SUCCEEDED;
const OLD_PATH_ITERS: Index = 15;
const OLD_PATH_OBJ: Number = 7.732176659546963e-9;
const OLD_PATH_X: [Number; N] = [0.9999919127767478, 1.000050108266574, 7.180484037000472e-5];

fn assert_old_path(prob: IpoptProblem) {
    watched(prob);
    let mut p = Probe::default();
    let s = solve_from(prob, &mut p, [0.0; N]);
    assert_eq!(s.status, OLD_PATH_STATUS);
    assert_eq!(unsafe { GetIpoptIterCount(prob) }, OLD_PATH_ITERS);
    assert_eq!(s.obj.to_bits(), OLD_PATH_OBJ.to_bits(), "obj = {:e}", s.obj);
    assert_eq!(
        s.x.map(f64::to_bits),
        OLD_PATH_X.map(f64::to_bits),
        "x = {:?}",
        s.x
    );
    // One solve, no driver; 17 objective evaluations, as before.
    assert_eq!(p.starts, 1);
    assert_eq!(p.fires as Index, OLD_PATH_ITERS + 1);
    assert_eq!(p.eval_f_calls, 17);
}

#[test]
fn no_pairs_takes_the_old_path() {
    let prob = create();
    assert_old_path(prob);
    free(prob);
}

#[test]
fn pairs_cleared_before_the_solve_take_the_old_path() {
    let prob = with_pairs();
    assert_eq!(unsafe { ClearPounceComplementarityPairs(prob) }, 1);
    assert_old_path(prob);
    free(prob);
}

// error-path: IpoptCreateSolver
#[test]
fn a_session_refuses_a_problem_with_pairs() {
    let mut prob = with_pairs();
    let solver = unsafe { IpoptCreateSolver(&mut prob) };
    assert!(solver.is_null(), "a session must not hold a lowered factor");
    // The caller still owns the handle, pairs intact, and it still solves.
    assert!(!prob.is_null());
    assert_eq!(unsafe { GetPounceComplementarityPairCount(prob) }, 2);
    let mut p = Probe::default();
    assert_eq!(solve(prob, &mut p).status, SOLVE_SUCCEEDED);
    // The other branch: with the pairs cleared the same handle makes a
    // session, which consumes it.
    assert_eq!(unsafe { ClearPounceComplementarityPairs(prob) }, 1);
    let solver = unsafe { IpoptCreateSolver(&mut prob) };
    assert!(!solver.is_null());
    assert!(prob.is_null());
    unsafe { IpoptFreeSolver(solver) };
}

/// An all-inactive working set for `QpecSmall`.
fn stage_working_set(prob: IpoptProblem) {
    let bounds = [0 as IpoptBoundStatus; N];
    let cons = [0 as IpoptConsStatus; M];
    assert_eq!(
        unsafe { IpoptSetWarmStartWorkingSet(prob, bounds.as_ptr(), cons.as_ptr()) },
        1
    );
}

#[test]
fn a_staged_working_set_with_pairs_is_refused_before_any_callback() {
    let prob = with_pairs();
    watched(prob);
    stage_working_set(prob);
    let mut p = Probe::default();
    let s = solve(prob, &mut p);
    assert_eq!(s.status, INVALID_PROBLEM_DEFINITION);
    assert_eq!(p.eval_f_calls, 0);
    assert_eq!(p.fires, 0);
    // Nothing was solved: no stats to report.
    assert_eq!(unsafe { GetIpoptIterCount(prob) }, 0);
    // The refusal consumed neither: it refuses again until one is cleared.
    let mut p = Probe::default();
    assert_eq!(solve(prob, &mut p).status, INVALID_PROBLEM_DEFINITION);
    assert_eq!(p.eval_f_calls, 0);
    // Clearing the working set lets the pairs solve.
    assert_eq!(unsafe { IpoptClearWarmStartWorkingSet(prob) }, 1);
    let mut p = Probe::default();
    let s = solve(prob, &mut p);
    assert_eq!(s.status, SOLVE_SUCCEEDED);
    assert_eq!(p.starts, 10);
    free(prob);
}

#[test]
fn a_staged_working_set_without_pairs_still_solves() {
    // The other branch of the refusal: the same working set, no pairs, on
    // the active-set SQP it is meant for.
    let prob = create();
    assert_eq!(str_opt(prob, "algorithm", "active-set-sqp"), 1);
    stage_working_set(prob);
    let mut p = Probe::default();
    let s = solve(prob, &mut p);
    assert_eq!(s.status, SOLVE_SUCCEEDED);
    assert_answer(&s);
    assert!(p.eval_f_calls > 0);
    free(prob);
}

// error-path: IpoptSolveWarmStart
#[test]
fn ipopt_solve_warm_start_with_pairs_is_refused() {
    let prob = with_pairs();
    let bounds = [0 as IpoptBoundStatus; N];
    let cons = [0 as IpoptConsStatus; M];
    let mut bounds_out = [-7 as IpoptBoundStatus; N];
    let mut cons_out = [-7 as IpoptConsStatus; M];
    let mut x = [0.0; N];
    let mut p = Probe::default();
    let status = unsafe {
        IpoptSolveWarmStart(
            prob,
            x.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            bounds.as_ptr(),
            cons.as_ptr(),
            bounds_out.as_mut_ptr(),
            cons_out.as_mut_ptr(),
            &mut p as *mut Probe as *mut c_void,
        )
    };
    assert_eq!(status, INVALID_PROBLEM_DEFINITION);
    assert_eq!(p.eval_f_calls, 0);
    // No working set was produced, so the output buffers are untouched.
    assert_eq!(bounds_out, [-7; N]);
    assert_eq!(cons_out, [-7; M]);
    free(prob);
}

// error-path: IpoptGetWorkingSet
#[test]
fn no_working_set_is_reported_after_a_driver_solve() {
    // Control: a pair-less active-set SQP solve reports one.
    let prob = create();
    watched(prob);
    assert_eq!(str_opt(prob, "algorithm", "active-set-sqp"), 1);
    let mut p = Probe::default();
    assert_eq!(solve(prob, &mut p).status, SOLVE_SUCCEEDED);
    let mut b = [-7 as IpoptBoundStatus; N];
    let mut c = [-7 as IpoptConsStatus; M];
    assert_eq!(
        unsafe { IpoptGetWorkingSet(prob, b.as_mut_ptr(), c.as_mut_ptr()) },
        1
    );
    // The same handle, now with pairs: the driver's lowered solves ran on a
    // model with more rows, and no set over the caller's rows exists.
    for (r, v) in PAIRS {
        assert_eq!(
            unsafe { AddPounceComplementarityPair(prob, r, v) },
            POUNCE_OK
        );
    }
    let mut p = Probe::default();
    let s = solve(prob, &mut p);
    assert!(p.starts > 1, "the driver ran");
    assert!(s.status == SOLVE_SUCCEEDED, "status {}", s.status);
    let mut b = [-7 as IpoptBoundStatus; N];
    let mut c = [-7 as IpoptConsStatus; M];
    assert_eq!(
        unsafe { IpoptGetWorkingSet(prob, b.as_mut_ptr(), c.as_mut_ptr()) },
        0
    );
    assert_eq!(b, [-7; N]);
    assert_eq!(c, [-7; M]);
    free(prob);
}

/// An active-set SQP solve without pairs on `prob`, which leaves a working
/// set on the handle (checked).
fn sqp_solve_leaving_a_working_set(prob: IpoptProblem) {
    assert_eq!(str_opt(prob, "algorithm", "active-set-sqp"), 1);
    let mut p = Probe::default();
    assert_eq!(solve(prob, &mut p).status, SOLVE_SUCCEEDED);
    let mut b = [-7 as IpoptBoundStatus; N];
    let mut c = [-7 as IpoptConsStatus; M];
    assert_eq!(
        unsafe { IpoptGetWorkingSet(prob, b.as_mut_ptr(), c.as_mut_ptr()) },
        1,
        "the control: an SQP solve leaves a set"
    );
    assert_ne!(b, [-7; N]);
}

fn declare_pairs(prob: IpoptProblem) {
    for (r, v) in PAIRS {
        assert_eq!(
            unsafe { AddPounceComplementarityPair(prob, r, v) },
            POUNCE_OK
        );
    }
}

// error-path: IpoptSolveWarmStart
/// PR review (gh#1147, thread 1): the refused warm start must not hand back
/// the working set an *earlier* SQP solve on the handle left behind. Before
/// the fix the refusal cleared `last_solve`, and `IpoptGetWorkingSet`'s
/// guard, which looked only at `last_solve.mpcc`, fell through to that set.
#[test]
fn a_refused_warm_start_does_not_copy_out_an_earlier_working_set() {
    let prob = create();
    sqp_solve_leaving_a_working_set(prob);
    declare_pairs(prob);
    let bounds = [0 as IpoptBoundStatus; N];
    let cons = [0 as IpoptConsStatus; M];
    let mut bounds_out = [-7 as IpoptBoundStatus; N];
    let mut cons_out = [-7 as IpoptConsStatus; M];
    let mut x = START;
    let mut p = Probe::default();
    let status = unsafe {
        IpoptSolveWarmStart(
            prob,
            x.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            bounds.as_ptr(),
            cons.as_ptr(),
            bounds_out.as_mut_ptr(),
            cons_out.as_mut_ptr(),
            &mut p as *mut Probe as *mut c_void,
        )
    };
    assert_eq!(status, INVALID_PROBLEM_DEFINITION);
    assert_eq!(p.eval_f_calls, 0);
    assert_eq!(bounds_out, [-7; N], "stale bound statuses written");
    assert_eq!(cons_out, [-7; M], "stale row statuses written");
    free(prob);
}

// error-path: IpoptGetWorkingSet
/// PR review (gh#1147, thread 1), one fixture per branch of the new guard:
/// pairs declared after an SQP solve (no solve since), and an `IpoptSolve`
/// refused without pairs (`last_solve` empty). Both used to return the
/// earlier solve's set.
#[test]
fn no_working_set_is_reported_once_pairs_are_declared_or_a_solve_is_refused() {
    // Branch 1: pairs declared, no solve since.
    let prob = create();
    sqp_solve_leaving_a_working_set(prob);
    declare_pairs(prob);
    let mut b = [-7 as IpoptBoundStatus; N];
    let mut c = [-7 as IpoptConsStatus; M];
    assert_eq!(
        unsafe { IpoptGetWorkingSet(prob, b.as_mut_ptr(), c.as_mut_ptr()) },
        0
    );
    assert_eq!((b, c), ([-7; N], [-7; M]));
    // Clearing the pairs makes the set readable again: the guard is on the
    // declaration, not on the set.
    assert_eq!(unsafe { ClearPounceComplementarityPairs(prob) }, 1);
    assert_eq!(
        unsafe { IpoptGetWorkingSet(prob, b.as_mut_ptr(), c.as_mut_ptr()) },
        1
    );
    free(prob);

    // Branch 2: no pairs, the next solve refused (NULL `x`), so
    // `last_solve` is empty.
    let prob = create();
    sqp_solve_leaving_a_working_set(prob);
    let status = unsafe {
        IpoptSolve(
            prob,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(status, INVALID_PROBLEM_DEFINITION);
    let mut b = [-7 as IpoptBoundStatus; N];
    let mut c = [-7 as IpoptConsStatus; M];
    assert_eq!(
        unsafe { IpoptGetWorkingSet(prob, b.as_mut_ptr(), c.as_mut_ptr()) },
        0
    );
    assert_eq!((b, c), ([-7; N], [-7; M]));
    free(prob);
}

/// State for [`nesting_watch`]: the outer problem `a` (with pairs) and the
/// inner pair-free problem `b` it solves from inside its own callback.
struct Nest {
    a: IpoptProblem,
    b: IpoptProblem,
    /// `inspect(a)` on the fire that starts the nested solve, before it.
    a_before: Option<[Bool; 3]>,
    /// `inspect(a)` on the same fire, after the nested solve returned.
    a_after: Option<[Bool; 3]>,
    /// What `b`'s own callback saw, and its status.
    b_probe: Probe,
    b_status: Option<Index>,
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn nesting_watch(
    _alg_mod: Index,
    _iter_count: Index,
    _obj: Number,
    _inf_pr: Number,
    _inf_du: Number,
    _mu: Number,
    _d_norm: Number,
    _reg: Number,
    _alpha_du: Number,
    _alpha_pr: Number,
    _ls: Index,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let nest = &mut *(u as *mut Nest);
        if nest.b_status.is_none() {
            nest.a_before = Some(inspect(nest.a));
            nest.b_probe.prob = nest.b;
            let s = solve(nest.b, &mut nest.b_probe);
            nest.b_status = Some(s.status);
            nest.a_after = Some(inspect(nest.a));
        }
        1
    }
}

/// PR review (gh#1147, thread 2): the lowered-solve flag is per solve, not
/// per thread. A pair-free problem solved from inside a lowered solve's
/// callback on the same thread has working inspectors (it used to read
/// FALSE on every fire), and the outer, lowered problem's stay FALSE before
/// and after the nested solve.
#[test]
fn a_pair_free_solve_nested_in_a_lowered_solve_has_working_inspectors() {
    let a = with_pairs();
    assert_eq!(
        unsafe { SetIntermediateCallback(a, Some(nesting_watch)) },
        1
    );
    let b = create();
    watched(b);
    let mut nest = Nest {
        a,
        b,
        a_before: None,
        a_after: None,
        b_probe: Probe::default(),
        b_status: None,
    };
    let mut x = START;
    let status = unsafe {
        IpoptSolve(
            a,
            x.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut nest as *mut Nest as *mut c_void,
        )
    };
    assert_eq!(status, SOLVE_SUCCEEDED);
    assert_eq!(nest.b_status, Some(SOLVE_SUCCEEDED));
    assert!(!nest.b_probe.inspected.is_empty());
    assert!(
        nest.b_probe.inspected.iter().all(|r| *r == [1, 1, 1]),
        "the nested pair-free solve's inspectors: {:?}",
        nest.b_probe.inspected
    );
    assert_eq!(nest.a_before, Some([0, 0, 0]));
    assert_eq!(nest.a_after, Some([0, 0, 0]), "the lowered flag came back");
    // And once both solves are over, the thread is back to "no solve".
    assert_eq!(unsafe { inspect(a) }, [0, 0, 0]);
    free(b);
    free(a);
}

#[test]
fn the_live_inspectors_report_no_data_during_a_driver_solve() {
    let prob = with_pairs();
    watched(prob);
    let mut p = Probe {
        prob,
        ..Probe::default()
    };
    assert_eq!(solve(prob, &mut p).status, SOLVE_SUCCEEDED);
    assert!(p.fires > 7);
    // Every fire, every lowered solve, every request shape: FALSE.
    assert_eq!(p.inspected.len(), p.fires);
    assert!(
        p.inspected.iter().all(|r| *r == [0, 0, 0]),
        "{:?}",
        p.inspected
    );
    free(prob);
}

#[test]
fn the_live_inspectors_report_data_on_a_plain_solve() {
    // The other branch: the same callback and requests, no pairs.
    let prob = create();
    watched(prob);
    let mut p = Probe {
        prob,
        ..Probe::default()
    };
    assert_eq!(solve(prob, &mut p).status, SOLVE_SUCCEEDED);
    assert!(!p.inspected.is_empty());
    assert!(
        p.inspected.iter().all(|r| *r == [1, 1, 1]),
        "{:?}",
        p.inspected
    );
    free(prob);
}

#[test]
fn a_stop_request_inside_a_lowered_solve_ends_the_run() {
    let prob = with_pairs();
    watched(prob);
    // Stop on the twentieth fire: inside the second lowered solve (the
    // first, the continuation's tau = 1 rung, takes 15 iterations, 16
    // fires). A plain solve has no second start to be inside.
    let mut p = Probe {
        stop_on_fire: 20,
        ..Probe::default()
    };
    let s = solve(prob, &mut p);
    assert_eq!(s.status, USER_REQUESTED_STOP);
    // The driver started no further solve after the stop.
    assert_eq!(p.fires, 20);
    assert_eq!(p.starts, 2);
    free(prob);
}

/// `(option, value, reported x, iterations the callback sees)`, each
/// against Python on the same callbacks from [`START`]: `x` is Python's
/// reported point to the bit, and the count is Python's
/// `complementarity["total_iterations"]` — except under `schedule`, where
/// the callback sees 145 against the driver's 142: three lowered solves
/// there run a gh#983 scale-audit re-solve inside one `optimize_tnlp`, and
/// the discarded candidates' iterations fire the callback but are not the
/// solve's. Each non-default row moves `x` or the count, so a C option the
/// driver never read fails its row; the default-valued rows are controls.
const PY_OPTION_ROWS: [(&str, &str, [Number; N], usize); 7] = [
    ("complementarity_strategy", "both", PY_X, 139),
    (
        "complementarity_strategy",
        "exact",
        [
            1.0000000000018847,
            1.0000000000009424,
            2.945323339678135e-12,
        ],
        91,
    ),
    // The continuation alone answers at the same point `both` keeps.
    ("complementarity_strategy", "continuation", PY_X, 75),
    ("complementarity_continuation", "coupled", PY_X, 139),
    (
        "complementarity_continuation",
        "schedule",
        [1.000000001593589, 1.0000000007967946, 0.0],
        145,
    ),
    ("complementarity_lift", "no", PY_X, 139),
    (
        "complementarity_lift",
        "yes",
        [1.0000004204091528, 1.0000002102045764, 0.0],
        166,
    ),
];

#[test]
fn the_complementarity_options_reach_the_driver() {
    for (k, v, x, total) in PY_OPTION_ROWS {
        let prob = with_pairs();
        watched(prob);
        assert_eq!(str_opt(prob, k, v), 1, "{k}={v}");
        let mut p = Probe::default();
        let s = solve(prob, &mut p);
        assert_eq!(s.status, SOLVE_SUCCEEDED, "{k}={v}");
        assert_answer(&s);
        assert_eq!(
            s.x.map(f64::to_bits),
            x.map(f64::to_bits),
            "{k}={v}: x = {:?}",
            s.x
        );
        assert_eq!(p.resto_fires, 0, "{k}={v}");
        assert_eq!(p.fires - p.starts, total, "{k}={v}");
        free(prob);
    }
}

#[test]
fn a_bad_complementarity_option_value_is_refused_at_set_time() {
    let prob = with_pairs();
    assert_eq!(str_opt(prob, "complementarity_strategy", "sometimes"), 0);
    assert_eq!(str_opt(prob, "complementarity_lift", "maybe"), 0);
    free(prob);
}

/// `SetIpoptProblemScaling` under pairs: the lowering wrapper hands the
/// bridge's factors to the caller's rows and variables and leaves the
/// product rows unscaled (`MpccTnlp::get_scaling_parameters`). Held against
/// Python's `set_problem_scaling` with the same factors and
/// `nlp_scaling_method user-scaling`: its point to the bit, 83 iterations
/// against 139 unscaled — so a scaling the driver's solves never saw would
/// fail here.
#[test]
fn user_scaling_reaches_the_lowered_solves() {
    let prob = with_pairs();
    watched(prob);
    assert_eq!(str_opt(prob, "nlp_scaling_method", "user-scaling"), 1);
    let xs = [1.0, 2.0, 0.5];
    let gs = [3.0, 0.25];
    assert_eq!(
        unsafe { SetIpoptProblemScaling(prob, 10.0, xs.as_ptr(), gs.as_ptr()) },
        1
    );
    let mut p = Probe::default();
    let s = solve(prob, &mut p);
    assert_eq!(s.status, SOLVE_SUCCEEDED);
    assert_eq!(
        s.x.map(f64::to_bits),
        [
            1.0000197552996812,
            1.0000098776510076,
            1.8151812631268459e-10
        ]
        .map(f64::to_bits),
        "x = {:?}",
        s.x
    );
    assert_eq!(p.fires - p.starts, 83);
    assert_eq!(unsafe { GetIpoptIterCount(prob) }, 10);
    free(prob);
}

/// L-BFGS solve of `prob` from [`START`], optionally with a nonlinear-variable
/// mask; returns `(x, iterations the callback saw)`.
fn lbfgs_solve(prob: IpoptProblem, mask: Option<&[Index]>) -> ([Number; N], usize) {
    watched(prob);
    assert_eq!(str_opt(prob, "hessian_approximation", "limited-memory"), 1);
    if let Some(mask) = mask {
        assert_eq!(
            unsafe { IpoptSetNonlinearVariables(prob, mask.len() as Index, mask.as_ptr()) },
            1
        );
    }
    let mut p = Probe::default();
    let s = solve(prob, &mut p);
    assert_eq!(s.status, SOLVE_SUCCEEDED);
    (s.x, p.fires - p.starts)
}

/// `IpoptSetNonlinearVariables` under pairs: the wrapper forwards the
/// caller's list and adds every variable a product touches (each pair's
/// variable and its row's columns). On `QpecSmall` those are all three, so
/// a mask of `{x}` is the full set and the L-BFGS run is the unmasked one,
/// to the bit.
#[test]
fn a_nonlinear_variable_mask_under_pairs_gains_the_product_variables() {
    let unmasked = lbfgs_solve(with_pairs(), None);
    let masked = lbfgs_solve(with_pairs(), Some(&[0]));
    assert_eq!(masked.0.map(f64::to_bits), unmasked.0.map(f64::to_bits));
    assert_eq!(masked.1, unmasked.1);
}

/// The other branch: without pairs the same mask is honoured as given, and
/// the L-BFGS run differs from the unmasked one.
#[test]
fn a_nonlinear_variable_mask_without_pairs_is_honoured_as_given() {
    let unmasked = lbfgs_solve(create(), None);
    let masked = lbfgs_solve(create(), Some(&[0]));
    assert_ne!(masked, unmasked);
}
