//! Performance smoke test: POUNCE + FERAL against POUNCE + RSLAB at three
//! scales.
//!
//! Not a benchmark of the two factorization kernels — `rslab_bench.rs` is that.
//! This measures **whole POUNCE solves**: the same NLP, driven end to end by
//! the interior-point loop, with only the linear-solver backend swapped
//! underneath through `IpoptApplication::set_linear_backend_factory`. It is
//! the number a user would feel.
//!
//! ```sh
//! cargo run -p pounce-rslab --release --example rslab_scale_smoke
//! RSLAB_SMOKE_REPS=5 cargo run -p pounce-rslab --release --example rslab_scale_smoke
//! ```
//!
//! # Reading it
//!
//! The wall-clock column is only meaningful when the two backends reach the
//! **same status in the same number of iterations**. A backend that stops
//! early is not faster, and the table says so rather than leaving the reader
//! to notice. Where the trajectories differ, `it` and `status` are the result
//! and the seconds are context.
//!
//! RSLAB appears twice because its exact mode cannot factor most POUNCE KKT
//! systems at all (no delayed pivoting — see
//! `dev-notes/rslab-backend-assessment.md`). `rslab-sp` is static pivoting,
//! the configuration in which it can.
//!
//! # The model
//!
//! A discretised optimal-control problem: minimise `∫ (x² + u²)` subject to
//! explicit-Euler dynamics `x_{k+1} = x_k + h(a·x_k + u_k)`. Linear equality
//! constraints, a separable quadratic objective, and a banded KKT whose (2,2)
//! block is structurally zero — the shape POUNCE actually factors, at a size
//! the caller sets. Scaling `T` scales `n` and `m` together, so the three
//! rows differ in size and not in character.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use pounce_algorithm::alg_builder::LinearSolverChoice;
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_rslab::{RslabConfig, RslabSolverInterface};

/// Discretised LQ optimal control over `T` steps.
///
/// Variables `x_0..x_T` then `u_0..u_{T-1}`, so `n = 2T + 1`; constraints are
/// the `T` dynamics rows plus the fixed initial state, so `m = T + 1`. The
/// Hessian is diagonal and the Jacobian has three entries per dynamics row.
struct OptControl {
    t: usize,
    h: Number,
    a: Number,
}

impl OptControl {
    fn n(&self) -> usize {
        2 * self.t + 1
    }
    fn m(&self) -> usize {
        self.t + 1
    }
    /// Index of `u_k` among the variables.
    fn u(&self, k: usize) -> usize {
        self.t + 1 + k
    }
}

impl TNLP for OptControl {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n() as Index,
            m: self.m() as Index,
            // 3 per dynamics row (x_k, x_{k+1}, u_k) + 1 for the initial state.
            nnz_jac_g: (3 * self.t + 1) as Index,
            // Diagonal Hessian.
            nnz_h_lag: self.n() as Index,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for v in b.x_l.iter_mut() {
            *v = -10.0;
        }
        for v in b.x_u.iter_mut() {
            *v = 10.0;
        }
        for v in b.g_l.iter_mut() {
            *v = 0.0;
        }
        for v in b.g_u.iter_mut() {
            *v = 0.0;
        }
        // The initial state is pinned to 1 by its own equality row.
        b.g_l[self.t] = 1.0;
        b.g_u[self.t] = 1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for (i, v) in sp.x.iter_mut().enumerate() {
            *v = 0.5 + 0.1 * ((i % 7) as Number);
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x.iter().map(|v| v * v).sum::<Number>())
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for (gi, xi) in g.iter_mut().zip(x) {
            *gi = 2.0 * xi;
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for k in 0..self.t {
            g[k] = x[k + 1] - x[k] - self.h * (self.a * x[k] + x[self.u(k)]);
        }
        g[self.t] = x[0];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut p = 0;
                for k in 0..self.t {
                    irow[p] = k as Index;
                    jcol[p] = k as Index;
                    p += 1;
                    irow[p] = k as Index;
                    jcol[p] = (k + 1) as Index;
                    p += 1;
                    irow[p] = k as Index;
                    jcol[p] = self.u(k) as Index;
                    p += 1;
                }
                irow[p] = self.t as Index;
                jcol[p] = 0;
            }
            SparsityRequest::Values { values } => {
                let mut p = 0;
                for _ in 0..self.t {
                    values[p] = -1.0 - self.h * self.a;
                    p += 1;
                    values[p] = 1.0;
                    p += 1;
                    values[p] = -self.h;
                    p += 1;
                }
                values[p] = 1.0;
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..self.n() {
                    irow[i] = i as Index;
                    jcol[i] = i as Index;
                }
            }
            SparsityRequest::Values { values } => {
                // The constraints are linear, so lambda contributes nothing.
                for v in values.iter_mut() {
                    *v = 2.0 * obj_factor;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

struct Run {
    status: ApplicationReturnStatus,
    iters: i32,
    objective: Number,
    constr_viol: Number,
    wall_s: f64,
    /// Phase totals over the whole solve, for the RSLAB arms. `None` for
    /// FERAL, which does not split.
    phases: Option<pounce_rslab::PhaseTimings>,
}

/// Sums [`pounce_rslab::PhaseTimings`] across every call of a whole solve.
///
/// Without this the smoke test can report that RSLAB is Nx slower and not say
/// whether that is RSLAB or the adapter wrapped around it — and two of the six
/// phases (`materialize`, and the CSC reference `solve` that replaces RSLAB's
/// supernodal sweep) exist only because the adapter reaches for the low-level
/// entry points. Attributing the gap is the difference between a finding and a
/// number.
#[derive(Default)]
struct PhaseAccum(std::sync::Mutex<pounce_rslab::PhaseTimings>);

impl PhaseAccum {
    fn add(&self, t: pounce_rslab::PhaseTimings) {
        let mut g = self.0.lock().unwrap_or_else(|e| e.into_inner());
        g.conversion_ms += t.conversion_ms;
        g.equilibration_ms += t.equilibration_ms;
        g.symbolic_ms += t.symbolic_ms;
        g.numeric_ms += t.numeric_ms;
        g.materialize_ms += t.materialize_ms;
        g.solve_ms += t.solve_ms;
    }
    fn snapshot(&self) -> pounce_rslab::PhaseTimings {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The RSLAB backend with a phase accumulator attached.
struct Timed {
    inner: RslabSolverInterface,
    accum: std::sync::Arc<PhaseAccum>,
}

impl SparseSymLinearSolverInterface for Timed {
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> pounce_linsol::ESymSolverStatus {
        self.inner.initialize_structure(dim, nonzeros, ia, ja)
    }
    fn values_array_mut(&mut self) -> &mut [Number] {
        self.inner.values_array_mut()
    }
    fn multi_solve(
        &mut self,
        new_matrix: bool,
        ia: &[Index],
        ja: &[Index],
        nrhs: Index,
        rhs_vals: &mut [Number],
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> pounce_linsol::ESymSolverStatus {
        let st = self.inner.multi_solve(
            new_matrix,
            ia,
            ja,
            nrhs,
            rhs_vals,
            check_neg_evals,
            number_of_neg_evals,
        );
        self.accum.add(self.inner.phase_timings());
        st
    }
    fn number_of_neg_evals(&self) -> Index {
        self.inner.number_of_neg_evals()
    }
    fn increase_quality(&mut self) -> bool {
        self.inner.increase_quality()
    }
    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }
    fn matrix_format(&self) -> pounce_linsol::EMatrixFormat {
        self.inner.matrix_format()
    }
}

/// Which backend one arm of the smoke test installs. A plain tag rather than a
/// closure because the factory must be `'static` and is rebuilt per solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Feral,
    Rslab,
    RslabStaticPivot,
}

impl Arm {
    fn tag(self) -> &'static str {
        match self {
            Arm::Feral => "feral",
            Arm::Rslab => "rslab",
            Arm::RslabStaticPivot => "rslab-sp",
        }
    }

    fn backend(
        self,
        accum: &std::sync::Arc<PhaseAccum>,
    ) -> Box<dyn SparseSymLinearSolverInterface> {
        let cfg = match self {
            Arm::Feral => return Box::new(pounce_feral::FeralSolverInterface::new()),
            Arm::Rslab => RslabConfig::default(),
            Arm::RslabStaticPivot => RslabConfig::static_pivoting(1e-12),
        };
        Box::new(Timed {
            inner: RslabSolverInterface::with_config(cfg),
            accum: std::sync::Arc::clone(accum),
        })
    }
}

fn run(t: usize, arm: Arm) -> Run {
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OptControl {
        t,
        h: 0.01,
        a: -0.5,
    }));
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.initialize_with_options_str("print_level 0\nmax_iter 300\n")
        .unwrap();
    let accum = std::sync::Arc::new(PhaseAccum::default());
    let accum_for_factory = std::sync::Arc::clone(&accum);
    app.set_linear_backend_factory(Box::new(
        move |_choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            arm.backend(&accum_for_factory)
        },
    ));
    let clock = Instant::now();
    let status = app.optimize_tnlp(tnlp);
    let wall_s = clock.elapsed().as_secs_f64();
    let s = app.statistics();
    Run {
        status,
        iters: s.iteration_count,
        objective: s.final_objective,
        constr_viol: s.final_constr_viol,
        wall_s,
        phases: (arm != Arm::Feral).then(|| accum.snapshot()),
    }
}

fn main() {
    let reps: usize = std::env::var("RSLAB_SMOKE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);

    // Three scales, an order of magnitude apart in the KKT dimension.
    let scales = [500usize, 5_000, 50_000];

    println!("# POUNCE end-to-end: FERAL vs RSLAB, three scales");
    println!("# LQ optimal control, T steps -> n = 2T+1 variables, m = T+1 constraints,");
    println!("# KKT dimension n + m = 3T + 2.");
    println!("# Wall clock is the best of {reps} whole solves.");
    println!("# Comparable ONLY where status and iteration count match; flagged otherwise.\n");
    println!(
        "{:>7} {:>9} {:<10} {:<24} {:>5} {:>15} {:>11} {:>9} {:>8}",
        "T", "KKT dim", "backend", "status", "it", "objective", "constr_viol", "wall_s", "vs feral"
    );

    for t in scales {
        let dim = 3 * t + 2;
        let mut baseline: Option<f64> = None;
        let mut baseline_shape: Option<(String, i32)> = None;

        for arm in [Arm::Feral, Arm::Rslab, Arm::RslabStaticPivot] {
            let mut best: Option<Run> = None;
            for _ in 0..reps {
                let r = run(t, arm);
                if best.as_ref().is_none_or(|b| r.wall_s < b.wall_s) {
                    best = Some(r);
                }
            }
            let r = best.expect("at least one rep");
            let shape = (format!("{:?}", r.status), r.iters);

            let note = match (&baseline, &baseline_shape) {
                (None, _) => {
                    baseline = Some(r.wall_s);
                    baseline_shape = Some(shape.clone());
                    "baseline".to_string()
                }
                (Some(b), Some(bs)) if *bs == shape => format!("{:.2}x", r.wall_s / b),
                _ => "(different trajectory)".to_string(),
            };

            println!(
                "{:>7} {:>9} {:<10} {:<24} {:>5} {:>15.7e} {:>11.2e} {:>9.3} {:>8}",
                t,
                dim,
                arm.tag(),
                shape.0,
                r.iters,
                r.objective,
                r.constr_viol,
                r.wall_s,
                note
            );
            if let Some(p) = r.phases {
                println!(
                    "{:>7} {:>9} {:<10} {:<24} phases ms: convert {:.1}  equil {:.1}  symbolic {:.1}  \
                     numeric {:.1}  materialize {:.1}  solve {:.1}  (sum {:.1} = {:.0}% of wall)",
                    "",
                    "",
                    "",
                    "",
                    p.conversion_ms,
                    p.equilibration_ms,
                    p.symbolic_ms,
                    p.numeric_ms,
                    p.materialize_ms,
                    p.solve_ms,
                    p.factor_total_ms() + p.solve_ms,
                    100.0 * (p.factor_total_ms() + p.solve_ms) / (r.wall_s * 1e3),
                );
            }
        }
        println!();
    }
}
