//! The convex IPM honors an attached `DebugHook`: it fires the shared
//! checkpoints, exposes the iterate through the `DebugState` surface, and
//! the attached hook does not change the solve result.

use pounce_common::debug::{Checkpoint, DebugAction, DebugHook, DebugState};
use pounce_convex::{solve_qp_ipm, solve_qp_ipm_debug, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// min ½(x0² + x1²) s.t. x0 + x1 ≥ 2  (i.e. −x0 − x1 ≤ −2). Optimum (1, 1),
/// f* = 1, the inequality active with z ≈ 1 — a nonempty cone, so the IPM
/// takes several predictor-corrector iterations.
fn active_ineq_qp() -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(0, 1, -1.0)],
        h: vec![-2.0],
        lb: vec![],
        ub: vec![],
    }
}

/// Records what the debugger sees at each checkpoint, and resumes.
#[derive(Default)]
struct Recorder {
    checkpoints: Vec<Checkpoint>,
    max_mu: f64,
    saw_nonempty_z: bool,
    saw_tau: bool,
    x_dim_at_iter_start: Option<usize>,
    terminal_status: Option<String>,
}

impl DebugHook for Recorder {
    fn at_checkpoint(&mut self, st: &mut dyn DebugState) -> DebugAction {
        self.checkpoints.push(st.checkpoint());
        self.max_mu = self.max_mu.max(st.mu());
        if let Some(z) = st.block("z") {
            if !z.is_empty() {
                self.saw_nonempty_z = true;
            }
        }
        if st.block("tau").is_some() {
            self.saw_tau = true;
        }
        if st.checkpoint() == Checkpoint::IterStart {
            self.x_dim_at_iter_start = st.block("x").map(|v| v.len());
        }
        if st.checkpoint() == Checkpoint::Terminated {
            self.terminal_status = st.status().map(str::to_owned);
        }
        DebugAction::Resume
    }
}

#[test]
fn convex_ipm_fires_checkpoints_and_exposes_state() {
    let prob = active_ineq_qp();
    let opts = QpOptions::default();
    let mut rec = Recorder::default();
    let sol = solve_qp_ipm_debug(&prob, &opts, &mut rec, backend);

    // The solve still reaches the known optimum.
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);

    // Every checkpoint kind fired at least once.
    let fired = |c| rec.checkpoints.contains(&c);
    assert!(fired(Checkpoint::IterStart), "no IterStart");
    assert!(
        fired(Checkpoint::AfterSearchDirection),
        "no AfterSearchDirection"
    );
    assert!(fired(Checkpoint::AfterStep), "no AfterStep");
    assert!(fired(Checkpoint::Terminated), "no Terminated");

    // State surfaced correctly: nonempty cone, μ moved, x has the right
    // dimension, and the terminal checkpoint carried the status.
    assert!(
        rec.saw_nonempty_z,
        "z block should be nonempty (one cone row)"
    );
    assert!(rec.max_mu > 0.0, "mu should be positive on a coned solve");
    assert_eq!(rec.x_dim_at_iter_start, Some(2), "x dim");
    assert_eq!(rec.terminal_status.as_deref(), Some("Optimal"));
}

#[test]
fn attaching_a_hook_does_not_change_the_result() {
    let prob = active_ineq_qp();
    let opts = QpOptions::default();

    let plain = solve_qp_ipm(&prob, &opts, backend);
    let mut rec = Recorder::default();
    let debugged = solve_qp_ipm_debug(&prob, &opts, &mut rec, backend);

    assert_eq!(plain.status, debugged.status);
    assert_eq!(plain.iters, debugged.iters, "iteration count must match");
    for (a, b) in plain.x.iter().zip(&debugged.x) {
        assert!((a - b).abs() < 1e-12, "x differs: {a} vs {b}");
    }
    assert!((plain.obj - debugged.obj).abs() < 1e-12, "obj differs");
}

/// The HSDE driver (`use_hsde`) is debuggable through the same entry: it
/// fires the checkpoints, exposes the homogenizing τ/κ as blocks, and the
/// hook does not change the recovered solution.
#[test]
fn hsde_driver_is_debuggable_and_exposes_tau_kappa() {
    let prob = active_ineq_qp();
    let opts = QpOptions {
        use_hsde: true,
        ..QpOptions::default()
    };

    let mut rec = Recorder::default();
    let sol = solve_qp_ipm_debug(&prob, &opts, &mut rec, backend);

    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-5, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-5, "x1={}", sol.x[1]);

    assert!(
        rec.checkpoints.contains(&Checkpoint::IterStart),
        "IterStart"
    );
    assert!(
        rec.checkpoints.contains(&Checkpoint::AfterStep),
        "AfterStep"
    );
    assert!(
        rec.checkpoints.contains(&Checkpoint::Terminated),
        "Terminated"
    );
    assert!(rec.saw_tau, "HSDE must expose the `tau` block");
    assert_eq!(rec.terminal_status.as_deref(), Some("Optimal"));

    // The attached hook leaves the HSDE result untouched.
    let plain = {
        let o = QpOptions {
            use_hsde: true,
            ..QpOptions::default()
        };
        solve_qp_ipm(&prob, &o, backend)
    };
    assert_eq!(plain.status, sol.status);
    for (a, b) in plain.x.iter().zip(&sol.x) {
        assert!((a - b).abs() < 1e-10, "x differs: {a} vs {b}");
    }
}

/// The non-symmetric (exponential/power) HSDE driver is debuggable too,
/// through `solve_conic_hsde_nonsym_debug`. Uses the exp-cone epigraph
/// `min z s.t. x=1, y=1, (x,y,z) ∈ K_exp` (optimum z = e).
#[test]
fn nonsym_exp_cone_driver_is_debuggable() {
    use pounce_convex::hsde_nonsym::{
        solve_conic_hsde_nonsym, solve_conic_hsde_nonsym_debug, NsBlock,
    };

    let e = std::f64::consts::E;
    let prob = QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![0.0, 0.0, 1.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        b: vec![1.0, 1.0],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(2, 2, -1.0),
        ],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let specs = [NsBlock::exp()];
    let opts = QpOptions::default();

    let mut rec = Recorder::default();
    let sol = solve_conic_hsde_nonsym_debug(&prob, &specs, &opts, &mut rec, backend);

    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[2] - e).abs() < 1e-5, "z={} vs e", sol.x[2]);

    assert!(
        rec.checkpoints.contains(&Checkpoint::IterStart),
        "IterStart"
    );
    assert!(
        rec.checkpoints.contains(&Checkpoint::AfterStep),
        "AfterStep"
    );
    assert!(
        rec.checkpoints.contains(&Checkpoint::Terminated),
        "Terminated"
    );
    assert!(rec.saw_tau, "nonsym HSDE must expose the `tau` block");
    assert_eq!(rec.terminal_status.as_deref(), Some("Optimal"));

    // The hook leaves the recovered solution untouched.
    let plain = solve_conic_hsde_nonsym(&prob, &specs, &opts, backend);
    assert_eq!(plain.status, sol.status);
    for (a, b) in plain.x.iter().zip(&sol.x) {
        assert!((a - b).abs() < 1e-9, "x differs: {a} vs {b}");
    }
}

/// The debugger can edit the iterate in place (`set`) and snapshot/restore
/// it (`goto`). `set mu` is rejected (μ is derived).
#[test]
fn convex_debugger_supports_set_and_rewind() {
    use std::cell::RefCell;

    // A hook that, at the first IterStart, snapshots the iterate, perturbs
    // `x`, confirms the edit took, then restores — all via the trait.
    #[derive(Default)]
    struct Mutator {
        snap: RefCell<Option<Box<dyn pounce_common::debug::IterSnapshot>>>,
        edited_x0: RefCell<Option<f64>>,
        restored_x0: RefCell<Option<f64>>,
        set_mu_err: RefCell<bool>,
        done: bool,
    }
    impl DebugHook for Mutator {
        fn at_checkpoint(&mut self, st: &mut dyn DebugState) -> DebugAction {
            if self.done || st.checkpoint() != Checkpoint::IterStart {
                return DebugAction::Resume;
            }
            self.done = true;
            // Snapshot, then edit x[0].
            *self.snap.borrow_mut() = st.snapshot();
            let mut x = st.block("x").unwrap();
            x[0] += 1.25;
            st.set_block("x", &x).expect("set_block x");
            *self.edited_x0.borrow_mut() = st.block("x").map(|v| v[0]);
            // μ is derived — editing it must be refused.
            *self.set_mu_err.borrow_mut() = st.set_mu(0.5).is_err();
            // Restore the snapshot and read x[0] back.
            let snap = self.snap.borrow_mut().take().unwrap();
            assert!(st.restore(snap.as_ref()), "restore should succeed");
            *self.restored_x0.borrow_mut() = st.block("x").map(|v| v[0]);
            DebugAction::Resume
        }
    }

    let prob = active_ineq_qp();
    let opts = QpOptions::default();
    let mut hook = Mutator::default();
    let sol = solve_qp_ipm_debug(&prob, &opts, &mut hook, backend);

    // The edit was observed, set_mu refused, and the restore undid the edit.
    assert_eq!(hook.edited_x0.into_inner(), Some(1.25), "edit visible");
    assert!(hook.set_mu_err.into_inner(), "set mu must be rejected");
    assert_eq!(
        hook.restored_x0.into_inner(),
        Some(0.0),
        "restore should bring x[0] back to the cold-start 0"
    );
    // The solve still converges (the edit+restore was a no-op net change).
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6 && (sol.x[1] - 1.0).abs() < 1e-6);
}

/// `solve_socp_ipm_debug` is the umbrella conic debug entry used by the
/// `pounce_cblib --debug` CLI path: exp/power cones route to the
/// non-symmetric driver, all others to the direct symmetric IPM. Here an
/// exp-cone epigraph (optimum z = e) exercises the routing.
#[test]
fn solve_socp_ipm_debug_routes_and_fires() {
    use pounce_convex::{solve_socp_ipm, solve_socp_ipm_debug, ConeSpec};

    let e = std::f64::consts::E;
    let prob = QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![0.0, 0.0, 1.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        b: vec![1.0, 1.0],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(2, 2, -1.0),
        ],
        h: vec![0.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let cones = [ConeSpec::Exponential];
    let opts = QpOptions::default();

    let mut rec = Recorder::default();
    let sol = solve_socp_ipm_debug(&prob, &cones, &opts, &mut rec, backend);

    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[2] - e).abs() < 1e-5, "z={} vs e", sol.x[2]);
    assert!(
        rec.checkpoints.contains(&Checkpoint::IterStart),
        "IterStart"
    );
    assert!(rec.saw_tau, "exp cone routes to HSDE → tau exposed");

    let plain = solve_socp_ipm(&prob, &cones, &opts, backend);
    assert_eq!(plain.status, sol.status);
    for (a, b) in plain.x.iter().zip(&sol.x) {
        assert!((a - b).abs() < 1e-9, "x differs: {a} vs {b}");
    }
}

/// A hook that requests `Stop` at the first checkpoint halts the solve
/// short of convergence (the debugger `quit` path).
#[test]
fn stop_action_halts_the_solve() {
    struct StopNow;
    impl DebugHook for StopNow {
        fn at_checkpoint(&mut self, _st: &mut dyn DebugState) -> DebugAction {
            DebugAction::Stop
        }
    }
    let prob = active_ineq_qp();
    let opts = QpOptions::default();
    let mut hook = StopNow;
    let sol = solve_qp_ipm_debug(&prob, &opts, &mut hook, backend);
    // Stopped at iteration 0 before convergence — not Optimal.
    assert_ne!(sol.status, QpStatus::Optimal);
}
