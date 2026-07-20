//! Adversarial fuzz over the gh #200 masked-certificate veto.
//!
//! The hand-written tests in `masked_certificate_veto.rs` check the cases the
//! fix was designed around, which is exactly what makes them weak evidence: a
//! fix can pass every test written by the person who wrote the fix and still be
//! wrong off that path. This file instead generates problems designed to *break*
//! the mechanism and asserts invariants that must hold for all of them.
//!
//! The generator sweeps the parameters that drive the pathology rather than
//! random noise: the exponent (how fast the gradient vanishes near the
//! minimum), the offset magnitude (how large the initial gradient is, hence how
//! extreme `obj_scale` becomes), conditioning spread, dimension, start point,
//! and convexity. Non-convex instances are included deliberately — the veto
//! makes a run travel further, and travelling further is how a solver finds a
//! *different* stationary point.
//!
//! The invariants are stated against the opt-out (`obj_scale_certificate_threshold
//! = 0`), which is the pre-fix behaviour, so every one of them is a statement of
//! the form "the fix did not make this worse".

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Deterministic xorshift64*, so any failure is reproducible from its seed.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> Number {
        (self.next_u64() >> 11) as Number / (1u64 << 53) as Number
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[(self.next_u64() % xs.len() as u64) as usize]
    }
}

/// `f(x) = Σᵢ cᵢ·(xᵢ − aᵢ)^p − wᵢ·xᵢ²`
///
/// `p` even and ≥ 4 gives a gradient that vanishes super-linearly at the
/// minimum, which combined with a large `a` is the scaling pathology. `w > 0`
/// adds a concave term, making the problem non-convex with several stationary
/// points — the case where "keep iterating" could plausibly land somewhere
/// worse rather than better.
#[derive(Clone)]
struct Spec {
    n: usize,
    p: i32,
    a: Vec<Number>,
    c: Vec<Number>,
    w: Vec<Number>,
    x0: Number,
    /// Dense rows of a linear constraint block `A x {=,<=} b`. Linear keeps the
    /// Lagrangian Hessian free of multiplier terms while still exercising the
    /// machinery the unconstrained fuzz cannot reach at all: constraint
    /// violation, equality/inequality multipliers, barrier complementarity,
    /// the filter line search, and restoration.
    arows: Vec<Vec<Number>>,
    brhs: Vec<Number>,
    /// `true` → equalities (`= b`), `false` → inequalities (`<= b`).
    eq: bool,
}

struct Problem(Spec, Rc<RefCell<Vec<Number>>>);

impl TNLP for Problem {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.0.n as i32,
            m: self.0.arows.len() as i32,
            nnz_jac_g: (self.0.arows.len() * self.0.n) as i32,
            nnz_h_lag: self.0.n as i32,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for v in b.x_l.iter_mut() {
            *v = -2.0e19;
        }
        for v in b.x_u.iter_mut() {
            *v = 2.0e19;
        }
        let s = &self.0;
        for (k, rhs) in s.brhs.iter().enumerate() {
            b.g_u[k] = *rhs;
            b.g_l[k] = if s.eq { *rhs } else { -2.0e19 };
        }
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for v in sp.x.iter_mut() {
            *v = self.0.x0;
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        let s = &self.0;
        Some(
            (0..s.n)
                .map(|i| s.c[i] * (x[i] - s.a[i]).powi(s.p) - s.w[i] * x[i] * x[i])
                .sum(),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        let s = &self.0;
        for i in 0..s.n {
            g[i] = s.c[i] * s.p as Number * (x[i] - s.a[i]).powi(s.p - 1) - 2.0 * s.w[i] * x[i];
        }
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        for (k, row) in self.0.arows.iter().enumerate() {
            g[k] = row.iter().zip(x).map(|(a, xi)| a * xi).sum();
        }
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        let s = &self.0;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut t = 0;
                for k in 0..s.arows.len() {
                    for j in 0..s.n {
                        irow[t] = k as i32;
                        jcol[t] = j as i32;
                        t += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let mut t = 0;
                for row in &s.arows {
                    for a in row {
                        values[t] = *a;
                        t += 1;
                    }
                }
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let s = &self.0;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..s.n {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                for i in 0..s.n {
                    let pp = s.p as Number;
                    values[i] = obj_factor
                        * (s.c[i] * pp * (pp - 1.0) * (x[i] - s.a[i]).powi(s.p - 2) - 2.0 * s.w[i]);
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.1.borrow_mut() = s.x.to_vec();
    }
}

struct Outcome {
    status: ApplicationReturnStatus,
    obj: Number,
    iters: usize,
}

fn run(spec: &Spec, threshold: Option<Number>, max_cpu: Option<Number>) -> Outcome {
    let mut app = IpoptApplication::new();
    if let Some(t) = threshold {
        app.options_mut()
            .set_numeric_value("obj_scale_certificate_threshold", t, true, false)
            .unwrap();
    }
    if let Some(t) = max_cpu {
        app.options_mut()
            .set_numeric_value("max_cpu_time", t, true, false)
            .unwrap();
    }
    // Keep the fuzz fast; the pathology shows up well inside this budget.
    app.options_mut()
        .set_integer_value("max_iter", 300, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Problem(
        spec.clone(),
        Rc::new(RefCell::new(Vec::new())),
    )));
    let status = app.optimize_tnlp(tnlp);
    let s = app.statistics();
    Outcome {
        status,
        obj: s.final_objective,
        iters: s.iteration_count as usize,
    }
}

fn run_capped(spec: &Spec, threshold: Option<Number>, max_iter: i32) -> Outcome {
    let mut app = IpoptApplication::new();
    if let Some(t) = threshold {
        app.options_mut()
            .set_numeric_value("obj_scale_certificate_threshold", t, true, false)
            .unwrap();
    }
    app.options_mut()
        .set_integer_value("max_iter", max_iter, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Problem(
        spec.clone(),
        Rc::new(RefCell::new(Vec::new())),
    )));
    let status = app.optimize_tnlp(tnlp);
    let s = app.statistics();
    Outcome {
        status,
        obj: s.final_objective,
        iters: s.iteration_count as usize,
    }
}

/// A terminating-at-a-real-point outcome. Deliberately includes
/// `SolvedToAcceptableLevel`: the veto blocks the acceptable-level branch too,
/// so a run that would have ended there is just as much at risk of being turned
/// into a bare failure. Matching only `SolveSucceeded` silently skipped that
/// entire population.
fn succeeded(s: ApplicationReturnStatus) -> bool {
    matches!(
        s,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

fn gen_spec(rng: &mut Rng) -> Spec {
    let n = rng.pick(&[2usize, 5, 20, 200]);
    let p = rng.pick(&[2i32, 4, 6, 8]);
    // Offset magnitude drives the initial gradient, hence obj_scale. The large
    // end pins it at the 1e-8 floor; the small end leaves scaling ordinary, so
    // the sweep covers both sides of the threshold.
    let amag = rng.pick(&[1.0, 10.0, 1e3, 1e5]);
    // Conditioning spread across coordinates.
    let cspread = rng.pick(&[1.0, 1e3, 1e6]);
    // Non-convexity: 0 for convex, positive adds a concave well.
    let wmag = rng.pick(&[0.0, 0.0, 1.0, 100.0]);
    let x0 = rng.pick(&[0.0, 2.0, -50.0]);
    // Constraint block: none, equalities, or inequalities. Kept small relative
    // to `n` so the problems stay solvable.
    let m = rng.pick(&[0usize, 0, 1, 3]).min(n.saturating_sub(1));
    let eq = rng.pick(&[true, false]);
    let a: Vec<Number> = (0..n).map(|_| (rng.unit() * 2.0 - 1.0) * amag).collect();
    let arows: Vec<Vec<Number>> = (0..m)
        .map(|_| (0..n).map(|_| rng.unit() * 2.0 - 1.0).collect())
        .collect();
    // Route each row near the unconstrained minimum so the feasible set is
    // non-empty and the constraints sometimes bind.
    let brhs = arows
        .iter()
        .map(|row| {
            let at_min: Number = row.iter().zip(&a).map(|(r, ai)| r * ai).sum();
            at_min + (rng.unit() * 2.0 - 1.0) * amag.sqrt()
        })
        .collect();
    Spec {
        n,
        p,
        a,
        c: (0..n).map(|_| 1.0 + rng.unit() * (cspread - 1.0)).collect(),
        w: (0..n).map(|_| rng.unit() * wmag).collect(),
        x0,
        arows,
        brhs,
        eq,
    }
}

/// The core guarantee, stated adversarially: enabling the veto must never turn
/// a successful solve into a failure, and must never return a worse point.
///
/// Both halves have teeth. The first is what the two-site fallback got wrong
/// (a run held back by the veto could exit through an unwired path); the second
/// is what the last-acceptable snapshot got wrong (the restored point could
/// drift away from the one that was refused).
#[test]
fn veto_never_degrades_status_or_objective() {
    let mut rng = Rng(0x5EED_2000);
    let (mut cases, mut improved, mut vetoed_paths) = (0, 0, 0);
    for case in 0..240 {
        let spec = gen_spec(&mut rng);
        let base = run(&spec, Some(0.0), None);
        let veto = run(&spec, None, None);
        cases += 1;

        if succeeded(base.status) {
            assert!(
                succeeded(veto.status),
                "case {case} (n={} p={} x0={}): baseline succeeded but veto gave {:?}",
                spec.n,
                spec.p,
                spec.x0,
                veto.status
            );
            // Minimization: never return a worse objective. The slack is
            // relative and tiny — this is meant to catch real regressions, not
            // last-bit noise.
            let slack = 1e-9 * base.obj.abs().max(1.0);
            assert!(
                veto.obj <= base.obj + slack,
                "case {case} (n={} p={} amag~{:.0e} w={}): veto objective {:.12e} is WORSE than \
                 baseline {:.12e}",
                spec.n,
                spec.p,
                spec.a.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
                spec.w.iter().fold(0.0_f64, |m, v| m.max(*v)),
                veto.obj,
                base.obj
            );
            if veto.obj < base.obj - slack {
                improved += 1;
            }
        }
        if veto.iters > base.iters {
            vetoed_paths += 1;
        }
    }
    // Guard the premise: if the generator stopped producing problems where the
    // veto actually engages, every assertion above would pass vacuously.
    assert!(
        vetoed_paths >= 10,
        "only {vetoed_paths}/{cases} cases engaged the veto — the fuzz is not exercising it"
    );
    eprintln!("fuzz: {cases} cases, veto engaged on {vetoed_paths}, improved {improved}");
}

/// The paths the original two-site fallback silently missed.
///
/// The veto spends extra iterations by design, so anything that bounds the run
/// can fire *because* of it — and before this was fixed only two of sixteen
/// termination sites restored the refused certificate. Here the run is cut off
/// at exactly the iteration count the baseline needed, which guarantees the
/// veto run cannot finish naturally and must exit through the cap instead.
///
/// A CPU-time budget is the other such bound and flows through the same
/// post-loop hook, but it is deliberately not fuzzed: a threshold tight enough
/// to cut the veto run but loose enough to spare the baseline is a race, and a
/// flaky test here would be worse than none. (An earlier revision of this file
/// did exactly that and "failed" only because the budget was so small the veto
/// never fired at all — the test was wrong, not the code.)
#[test]
fn an_exit_forced_before_the_veto_finishes_still_yields_the_refused_certificate() {
    let mut rng = Rng(0xC0DE_2000);
    let (mut checked, mut forced) = (0, 0);
    for case in 0..80 {
        let spec = gen_spec(&mut rng);
        let base = run(&spec, Some(0.0), None);
        if !succeeded(base.status) || base.iters == 0 {
            continue;
        }
        let veto_free = run(&spec, None, None);
        // Only interesting where the veto actually made the run longer; that is
        // precisely the population that a cap can now cut off.
        if veto_free.iters <= base.iters {
            continue;
        }
        forced += 1;

        // Cap at the baseline's own iteration count: the veto run provably
        // cannot converge within it.
        let capped = run_capped(&spec, None, base.iters as i32);
        checked += 1;
        assert!(
            !matches!(
                capped.status,
                ApplicationReturnStatus::MaximumIterationsExceeded
            ),
            "case {case}: a veto cut short at {} iters surfaced MaximumIterationsExceeded \
             where the baseline succeeded",
            base.iters
        );
        let slack = 1e-9 * base.obj.abs().max(1.0);
        assert!(
            capped.obj <= base.obj + slack,
            "case {case}: cut-short veto objective {:.12e} is worse than the refused \
             certificate {:.12e}",
            capped.obj,
            base.obj
        );
    }
    assert!(
        forced >= 10 && checked >= 10,
        "only {checked} cases exercised a forced exit — the fuzz is not reaching this path"
    );
    eprintln!("forced-exit fuzz: {checked} cases checked");
}

/// The opt-out must be inert: with the mechanism disabled, results must not
/// depend on it at all, and repeated runs must agree bit-for-bit.
#[test]
fn opt_out_is_inert_and_the_solver_stays_deterministic() {
    let mut rng = Rng(0xDEAD_2000);
    for case in 0..40 {
        let spec = gen_spec(&mut rng);
        let a = run(&spec, Some(0.0), None);
        let b = run(&spec, Some(0.0), None);
        assert_eq!(
            format!("{:?}", a.status),
            format!("{:?}", b.status),
            "case {case}: opt-out is non-deterministic"
        );
        assert!(
            (a.obj - b.obj).abs() <= 0.0 || a.obj.to_bits() == b.obj.to_bits(),
            "case {case}: opt-out objective differs between runs: {} vs {}",
            a.obj,
            b.obj
        );
        // And the veto run is itself reproducible.
        let c = run(&spec, None, None);
        let d = run(&spec, None, None);
        assert_eq!(
            format!("{:?}", c.status),
            format!("{:?}", d.status),
            "case {case}: veto run is non-deterministic"
        );
        assert!(
            c.obj.to_bits() == d.obj.to_bits(),
            "case {case}: veto objective differs between runs: {} vs {}",
            c.obj,
            d.obj
        );
    }
}

/// Evaluate the objective directly, independent of the solver's own bookkeeping.
fn eval_obj(spec: &Spec, x: &[Number]) -> Number {
    (0..spec.n)
        .map(|i| spec.c[i] * (x[i] - spec.a[i]).powi(spec.p) - spec.w[i] * x[i] * x[i])
        .sum()
}

/// The returned point must be the point the reported objective describes.
///
/// This is the failure mode a statistics-only test cannot see: the veto's
/// fallback rewrites the iterate *after* the solve loop has ended, so if the
/// restore and the reported statistics were drawn at different moments a caller
/// would receive an `x` that does not correspond to the objective it was handed
/// — a silent, and much nastier, kind of wrong answer than a bad status.
#[test]
fn the_returned_point_matches_the_reported_objective() {
    let mut rng = Rng(0xF00D_2000);
    let mut checked = 0;
    for case in 0..120 {
        let spec = gen_spec(&mut rng);
        let seen = Rc::new(RefCell::new(Vec::new()));
        let mut app = IpoptApplication::new();
        app.options_mut()
            .set_integer_value("max_iter", 300, true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("print_level", 0, true, false)
            .unwrap();
        app.initialize().unwrap();
        let tnlp: Rc<RefCell<dyn TNLP>> =
            Rc::new(RefCell::new(Problem(spec.clone(), Rc::clone(&seen))));
        let status = app.optimize_tnlp(tnlp);
        let reported = app.statistics().final_objective;
        let x = seen.borrow().clone();
        if x.len() != spec.n || !reported.is_finite() {
            continue;
        }
        checked += 1;
        let direct = eval_obj(&spec, &x);
        let scale = reported.abs().max(direct.abs()).max(1.0);
        assert!(
            (direct - reported).abs() <= 1e-6 * scale,
            "case {case} ({status:?}): returned x evaluates to {direct:.12e} but the reported \
             objective is {reported:.12e}"
        );
    }
    assert!(
        checked >= 60,
        "only {checked} cases produced a usable solution vector"
    );
    eprintln!("solution-consistency fuzz: {checked} cases checked");
}

/// Veto state must not leak between solves on a reused application object.
///
/// `vetoed_iterate` holds a full iterate snapshot. If it survived into a second
/// solve — a different problem, possibly a different dimension — restoring it
/// would at best return a stranger's answer and at worst corrupt the iterate.
/// Reusing one `IpoptApplication` across solves is ordinary in warm-start and
/// parametric workflows, and every other test here builds a fresh app, so
/// nothing else would catch this.
#[test]
fn veto_state_does_not_leak_across_solves_on_a_reused_application() {
    let mut rng = Rng(0xBEEF_2000);
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("max_iter", 300, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();

    for case in 0..40 {
        let spec = gen_spec(&mut rng);
        // Reference: same problem on a pristine application.
        let fresh = run(&spec, None, None);

        let seen = Rc::new(RefCell::new(Vec::new()));
        let tnlp: Rc<RefCell<dyn TNLP>> =
            Rc::new(RefCell::new(Problem(spec.clone(), Rc::clone(&seen))));
        let status = app.optimize_tnlp(tnlp);
        let obj = app.statistics().final_objective;

        assert_eq!(
            format!("{status:?}"),
            format!("{:?}", fresh.status),
            "case {case}: reused application gave a different status than a fresh one"
        );
        let scale = obj.abs().max(fresh.obj.abs()).max(1.0);
        assert!(
            (obj - fresh.obj).abs() <= 1e-9 * scale,
            "case {case}: reused application gave {obj:.12e}, fresh gave {:.12e} — state leaked",
            fresh.obj
        );
    }
}

/// `f = Σ c(x−a)⁴` with the provably inconsistent pair
/// `x₀²+x₁²−1 = 0`, `x₀²+x₁²−4 = 0`.
///
/// No point satisfies both, and LICQ fails everywhere (the two gradients are
/// identical), so the solver is driven into the restoration phase. The quartic
/// keeps the objective scale pinned at the 1e-8 floor so the veto engages on the
/// same run.
struct InconsistentPair {
    a: Number,
}

impl TNLP for InconsistentPair {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 4,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-2.0e19; 2]);
        b.x_u.copy_from_slice(&[2.0e19; 2]);
        b.g_l.copy_from_slice(&[0.0, 0.0]);
        b.g_u.copy_from_slice(&[0.0, 0.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[3.0, 3.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some((x[0] - self.a).powi(4) + (x[1] - self.a).powi(4))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 4.0 * (x[0] - self.a).powi(3);
        g[1] = 4.0 * (x[1] - self.a).powi(3);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        let r = x[0] * x[0] + x[1] * x[1];
        g[0] = r - 1.0;
        g[1] = r - 4.0;
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("no x");
                values[0] = 2.0 * x[0];
                values[1] = 2.0 * x[1];
                values[2] = 2.0 * x[0];
                values[3] = 2.0 * x[1];
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("no x");
                let lam = lambda.map(|l| l[0] + l[1]).unwrap_or(0.0);
                values[0] = obj_factor * 12.0 * (x[0] - self.a).powi(2) + 2.0 * lam;
                values[1] = obj_factor * 12.0 * (x[1] - self.a).powi(2) + 2.0 * lam;
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// The veto must stay correct across the restoration phase.
///
/// `vetoed_iterate` snapshots an *outer* iterate while restoration runs an inner
/// IPM over a different problem. Reading the code says the inner IPM owns its
/// own data and the outer `curr` keeps its dimensions — but two earlier
/// arguments of that exact shape turned out to be wrong, so this exercises it.
///
/// The earlier version of this test generated linear, consistent constraints and
/// never entered restoration at all: 0 of 200 cases. It passed its assertions
/// and proved nothing, which is why the count is asserted here.
#[test]
fn the_veto_survives_the_restoration_phase() {
    let mut entered = 0;
    for a in [1e3, 1e4, 1e5] {
        let solve = |threshold: Number| {
            let mut app = IpoptApplication::new();
            app.options_mut()
                .set_numeric_value("obj_scale_certificate_threshold", threshold, true, false)
                .unwrap();
            app.options_mut()
                .set_integer_value("max_iter", 300, true, false)
                .unwrap();
            app.options_mut()
                .set_integer_value("print_level", 0, true, false)
                .unwrap();
            app.initialize().unwrap();
            let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(InconsistentPair { a }));
            let st = app.optimize_tnlp(t);
            let s = app.statistics();
            (st, s.final_objective, s.restoration_calls)
        };
        let (bs, bo, br) = solve(0.0);
        let (vs, vo, vr) = solve(1e-4);
        eprintln!(
            "a={a:e}: baseline {bs:?} f={bo:.6e} resto={br} | veto {vs:?} f={vo:.6e} resto={vr}"
        );
        // `restoration_calls` stays 0 on this path — the phase is entered and
        // fails before the counter is bumped — so the status is the evidence
        // that restoration was actually involved.
        let restoration_involved = |st: ApplicationReturnStatus| {
            matches!(
                st,
                ApplicationReturnStatus::RestorationFailed
                    | ApplicationReturnStatus::InfeasibleProblemDetected
            )
        };
        if br > 0 || vr > 0 || restoration_involved(bs) || restoration_involved(vs) {
            entered += 1;
        }
        // An infeasible problem must not be certified as solved off a stale
        // snapshot — the restore is guarded on a finite objective only, so this
        // is the assertion that the guard is not the whole story.
        if !succeeded(bs) {
            assert!(
                !succeeded(vs),
                "a={a:e}: baseline correctly reported {bs:?} on an infeasible problem but the \
                 veto reported {vs:?}"
            );
        }
        assert!(
            vo.is_finite(),
            "a={a:e}: restoration + veto produced a non-finite objective"
        );
    }
    assert!(
        entered >= 3,
        "only {entered}/3 configurations reached restoration — the trigger no longer works"
    );
    eprintln!("restoration: {entered}/3 configurations entered restoration");
}

/// `f(x,y) = A(x−a)⁴ − K·√(1+y²)` — a masked objective whose *acceptable*-level
/// certificate the veto also blocks.
struct AcceptableOnly {
    a: Number,
    amp: Number,
    k: Number,
}

impl TNLP for AcceptableOnly {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-2.0e19; 2]);
        b.x_u.copy_from_slice(&[2.0e19; 2]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 1.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(self.amp * (x[0] - self.a).powi(4) - self.k * (1.0 + x[1] * x[1]).sqrt())
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 4.0 * self.amp * (x[0] - self.a).powi(3);
        g[1] = -self.k * x[1] / (1.0 + x[1] * x[1]).sqrt();
        true
    }
    fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("no x");
                let d = (1.0 + x[1] * x[1]).sqrt();
                values[0] = obj_factor * 12.0 * self.amp * (x[0] - self.a).powi(2);
                values[1] = obj_factor * (-self.k / (d * d * d));
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// The veto blocks acceptable-level termination as well as strict, but the
/// fallback that undoes a refusal is armed only when a *strict* certificate was
/// refused. A run whose best outcome was `Solved_To_Acceptable_Level` therefore
/// had its exit blocked with nothing to catch it, and surfaced a bare failure —
/// the veto turning a usable answer into an unusable one, which is exactly what
/// it promises never to do.
#[test]
fn a_blocked_acceptable_certificate_is_not_turned_into_a_failure() {
    for (a, amp, k) in [
        (1e5, 1.0, 10.0),
        (1e5, 1.0, 50.0),
        (1e3, 1.0, 10.0),
        (1e4, 1.0, 3.0),
    ] {
        let solve = |threshold: Number| {
            let mut app = IpoptApplication::new();
            app.options_mut()
                .set_numeric_value("obj_scale_certificate_threshold", threshold, true, false)
                .unwrap();
            app.options_mut()
                .set_integer_value("max_iter", 300, true, false)
                .unwrap();
            app.options_mut()
                .set_integer_value("print_level", 0, true, false)
                .unwrap();
            app.initialize().unwrap();
            let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(AcceptableOnly { a, amp, k }));
            let st = app.optimize_tnlp(t);
            (
                st,
                app.statistics().final_objective,
                app.statistics().iteration_count,
            )
        };
        let (bs, bo, bi) = solve(0.0);
        let (vs, vo, vi) = solve(1e-4);
        eprintln!(
            "a={a:e} k={k}: baseline {bs:?} f={bo:.6e} it={bi} | veto {vs:?} f={vo:.6e} it={vi}"
        );
        if !succeeded(bs) {
            continue;
        }
        assert!(
            succeeded(vs),
            "a={a:e} k={k}: baseline ended {bs:?} but the veto surfaced {vs:?}"
        );
    }
}

/// `g(x) = −Σ (xᵢ − aᵢ)⁴` — concave, maximum 0 at `x = a`.
struct ConcaveQuartic {
    a: Vec<Number>,
}

impl TNLP for ConcaveQuartic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.a.len() as i32,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: self.a.len() as i32,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for v in b.x_l.iter_mut() {
            *v = -2.0e19;
        }
        for v in b.x_u.iter_mut() {
            *v = 2.0e19;
        }
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for v in sp.x.iter_mut() {
            *v = 2.0;
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(
            -x.iter()
                .zip(&self.a)
                .map(|(xi, ai)| (xi - ai).powi(4))
                .sum::<Number>(),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        for (i, gi) in g.iter_mut().enumerate() {
            *gi = -4.0 * (x[i] - self.a[i]).powi(3);
        }
        true
    }
    fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..self.a.len() {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("no x");
                for (i, v) in values.iter_mut().enumerate() {
                    *v = obj_factor * (-12.0) * (x[i] - self.a[i]).powi(2);
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// The same flat quartic posed as a minimization and as the mathematically
/// identical maximization (`obj_scaling_factor = -1`, Ipopt's documented way to
/// maximize). Both have optimum 0 at `x = a`, and both are well-posed — the
/// concave objective is what makes the maximization bounded, so a failure here
/// cannot be blamed on the problem.
///
/// `obj_scaling_factor` is signed, and the unscaled residual accessors divide a
/// max-norm by it (`ipopt_cq.rs:783-806`). Under a negative factor those
/// "max-norms" come back negative, which makes `unscaled_err > acceptable_tol`
/// false and silently disables the veto. The same division feeds
/// `passes_component_tols`, so a negative value also sails under
/// `dual_inf_tol` / `compl_inf_tol` — meaning the unscaled residual gate added
/// for pounce#173 is defeated on maximization too, independently of this fix.
#[test]
fn the_veto_is_not_disabled_by_a_negative_objective_scaling_factor() {
    let a: Vec<Number> = (0..50).map(|i| 1e3 + i as Number).collect();
    let solve_min = |threshold: Number| {
        let mut app = IpoptApplication::new();
        app.options_mut()
            .set_numeric_value("obj_scale_certificate_threshold", threshold, true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("max_iter", 300, true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("print_level", 0, true, false)
            .unwrap();
        app.initialize().unwrap();
        let spec = Spec {
            n: 50,
            p: 4,
            a: a.clone(),
            c: vec![1.0; 50],
            w: vec![0.0; 50],
            x0: 2.0,
            arows: Vec::new(),
            brhs: Vec::new(),
            eq: true,
        };
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Problem(
            spec,
            Rc::new(RefCell::new(Vec::new())),
        )));
        let st = app.optimize_tnlp(t);
        let s = app.statistics();
        (st, s.final_objective, s.final_unscaled_kkt_error)
    };
    let solve_max = |threshold: Number| {
        let mut app = IpoptApplication::new();
        app.options_mut()
            .set_numeric_value("obj_scaling_factor", -1.0, true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("obj_scale_certificate_threshold", threshold, true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("max_iter", 300, true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("print_level", 0, true, false)
            .unwrap();
        app.initialize().unwrap();
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ConcaveQuartic { a: a.clone() }));
        let st = app.optimize_tnlp(t);
        let s = app.statistics();
        (st, s.final_objective, s.final_unscaled_kkt_error)
    };

    let (min_off, min_off_obj, _) = solve_min(0.0);
    let (min_on, min_on_obj, _) = solve_min(1e-4);
    let (max_off, max_off_obj, max_err) = solve_max(0.0);
    let (max_on, max_on_obj, _) = solve_max(1e-4);
    eprintln!(
        "min f -> 0: off {min_off:?} {min_off_obj:.6e} | on {min_on:?} {min_on_obj:.6e}\n\
         max g -> 0: off {max_off:?} {max_off_obj:.6e} | on {max_on:?} {max_on_obj:.6e}  \
         unscaled_err(off)={max_err:.3e}"
    );

    assert!(
        max_err >= 0.0,
        "unscaled KKT error came back NEGATIVE ({max_err:.3e}) under a negative objective \
         scaling factor — a max-norm cannot be negative, and the pounce#173 unscaled gate is \
         defeated by it, independently of this veto"
    );
    // Distance from the optimum (0) in each form.
    let min_gain = min_off_obj.abs() - min_on_obj.abs();
    let max_gain = max_off_obj.abs() - max_on_obj.abs();
    assert!(
        min_gain > 0.0,
        "premise: the veto should improve the minimization ({min_off_obj:.6e} -> {min_on_obj:.6e})"
    );
    assert!(
        max_gain > 0.5 * min_gain,
        "the veto moved the minimization {min_gain:.6e} closer to the optimum but the identical \
         maximization only {max_gain:.6e} — the mechanism is sign-dependent"
    );
}

/// Does the SQP path share the gh #200 bug?
///
/// The plan's §3 called for a relabel backstop so no exit path could report
/// success at a masked point; §8 dropped it when the design changed from
/// predicting a false stop to testing for one. The SQP path has no `ConvCheck`
/// and so cannot use the veto, but it *does* evaluate through `OrigIpoptNlp`,
/// which applies the objective scaling — so whether it is exposed is a question
/// about its convergence test, not something to assume either way.
///
/// This pins the answer. If SQP reports success on the masked quartic at a
/// point far from the minimum, it has the bug and needs its own remedy; if it
/// lands near the minimum, it does not, and the dropped backstop was
/// unnecessary rather than an omission.
#[test]
fn sqp_path_behaviour_on_a_masked_objective_is_pinned() {
    let a: Vec<Number> = (0..50).map(|i| 1e3 + i as Number).collect();
    let spec = Spec {
        n: 50,
        p: 4,
        a,
        c: vec![1.0; 50],
        w: vec![0.0; 50],
        x0: 2.0,
        arows: Vec::new(),
        brhs: Vec::new(),
        eq: true,
    };
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("algorithm", "active-set-sqp", true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("max_iter", 300, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Problem(
        spec,
        Rc::new(RefCell::new(Vec::new())),
    )));
    let status = app.optimize_tnlp(t);
    let obj = app.statistics().final_objective;
    eprintln!("SQP on the masked quartic: {status:?} obj={obj:.6e} (true minimum 0)");

    // The IPM path, unfixed, stops around 2.27 here and calls it optimal.
    // Assert the SQP path does not do the same thing silently.
    if succeeded(status) {
        assert!(
            obj < 1e-3,
            "SQP reported {status:?} at objective {obj:.6e} on a masked problem whose minimum \
             is 0 — it shares the gh #200 false-certificate bug and needs its own remedy"
        );
    }
}

/// The barrier parameter reported alongside a restored point must belong to
/// that point.
///
/// `curr_mu` lives on `IpoptData`, not in the `IteratesVector`, so restoring the
/// refused iterate does not rewind it — while `stats.final_mu` is read after the
/// restore. Left unhandled, a run that falls back reports the *continued* run's
/// barrier parameter next to the *refused* run's `x`. That pair is not
/// cosmetic: `final_mu` feeds a warm-started corrector's `mu_init` /
/// `warm_start_target_mu` and is exported to callers as `info["mu"]`, so a
/// warm-start chain would resume from a rewound point at a barrier parameter
/// belonging to a far more converged one.
#[test]
fn the_reported_barrier_parameter_belongs_to_the_returned_point() {
    for (a, k) in [(1e5, 10.0), (1e3, 10.0)] {
        let solve = |threshold: Number| {
            let mut app = IpoptApplication::new();
            app.options_mut()
                .set_numeric_value("obj_scale_certificate_threshold", threshold, true, false)
                .unwrap();
            app.options_mut()
                .set_integer_value("max_iter", 300, true, false)
                .unwrap();
            app.options_mut()
                .set_integer_value("print_level", 0, true, false)
                .unwrap();
            app.initialize().unwrap();
            let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(AcceptableOnly { a, amp: 1.0, k }));
            let st = app.optimize_tnlp(t);
            let s = app.statistics();
            (st, s.final_objective, s.final_mu)
        };
        let (bs, bo, bmu) = solve(0.0);
        let (vs, vo, vmu) = solve(1e-4);
        eprintln!(
            "a={a:e} k={k}: baseline {bs:?} f={bo:.6e} mu={bmu:.3e} | veto {vs:?} f={vo:.6e} mu={vmu:.3e}"
        );
        // The fallback restores the refused point, so the objective matches the
        // baseline exactly — and the barrier parameter must match too, since it
        // describes the same iterate.
        assert!(
            (vo - bo).abs() <= 1e-9 * bo.abs().max(1.0),
            "premise: the fallback should return the baseline point"
        );
        assert!(
            (vmu - bmu).abs() <= 1e-6 * bmu.abs().max(1e-300),
            "a={a:e}: returned the baseline point (f={vo:.6e}) but reported mu={vmu:.3e} \
             where the point's own barrier parameter is {bmu:.3e} — an (x, mu) pair that \
             never existed"
        );
    }
}

/// A well-conditioned problem the user has deliberately scaled down.
///
/// `obj_scaling_factor` is a *user* option. Someone who sets it to 1e-6 on a
/// perfectly ordinary problem is not in the pathological regime this fix exists
/// for — nothing is masked, the solve converges normally — but the veto's gate
/// is on the scale factor, so it arms anyway. Worse, the bar it then has to
/// clear is the *unscaled* error, which is `1/df` times the scaled one: with
/// `df = 1e-6`, lifting the veto needs a scaled error of 1e-12, four orders
/// tighter than `tol`. So the veto cannot lift, and every such solve pays the
/// full continuation budget for nothing.
///
/// This is the user-scaling interaction flagged in plan section 9.4. It is a
/// cost question, not a correctness one — the fallback still returns the
/// refused certificate — but a silent per-solve tax on a legitimate
/// configuration would be a poor trade.
struct WellConditioned;

impl TNLP for WellConditioned {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 10,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 10,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        // Active lower bounds: the barrier leaves a non-zero complementarity at
        // convergence, so the solve takes real iterations and its residual does
        // not collapse to exactly zero — without which the veto could never
        // engage and this test would pass vacuously.
        b.x_l.copy_from_slice(&[5.0; 10]);
        b.x_u.copy_from_slice(&[2.0e19; 10]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[8.0; 10]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(
            x.iter()
                .enumerate()
                .map(|(i, v)| (v - i as Number).powi(2))
                .sum(),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        for (i, gi) in g.iter_mut().enumerate() {
            *gi = 2.0 * (x[i] - i as Number);
        }
        true
    }
    fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..10 {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                for v in values.iter_mut() {
                    *v = obj_factor * 2.0;
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

#[test]
fn user_scaled_well_conditioned_problems_do_not_pay_a_veto_tax() {
    for df in [1e-5, 1e-6, 1e-8] {
        let solve = |threshold: Number| {
            let mut app = IpoptApplication::new();
            app.options_mut()
                .set_numeric_value("obj_scaling_factor", df, true, false)
                .unwrap();
            app.options_mut()
                .set_numeric_value("obj_scale_certificate_threshold", threshold, true, false)
                .unwrap();
            app.options_mut()
                .set_integer_value("max_iter", 300, true, false)
                .unwrap();
            app.options_mut()
                .set_integer_value("print_level", 0, true, false)
                .unwrap();
            app.initialize().unwrap();
            let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(WellConditioned));
            let st = app.optimize_tnlp(t);
            let s = app.statistics();
            (
                st,
                s.final_objective,
                s.iteration_count,
                s.final_unscaled_kkt_error,
            )
        };
        let (bs, bo, bi, berr) = solve(0.0);
        let (vs, vo, vi, _) = solve(1e-4);
        eprintln!(
            "obj_scaling_factor={df:e}: baseline {bs:?} f={bo:.6e} it={bi} unscaled_err={berr:.2e} | veto {vs:?} f={vo:.6e} it={vi}"
        );
        // Guard the premise: the veto can only engage where the unscaled error
        // is above acceptable_tol at the stopping point. If it is not, this
        // configuration never reaches the regime and the test proves nothing.
        assert!(
            bi > 3 && berr > 1e-6,
            "premise: df={df:e} did not reach the veto regime (it={bi}, unscaled_err={berr:.2e}) \
             — this test would pass vacuously"
        );
        assert_eq!(
            format!("{bs:?}"),
            format!("{vs:?}"),
            "df={df:e}: user scaling changed the status"
        );
        assert!(
            vi <= bi + 2,
            "df={df:e}: a deliberately user-scaled, well-conditioned solve took {vi} iterations \
             with the veto vs {bi} without — a per-solve tax on a legitimate configuration"
        );
    }
}

/// The veto must not suppress a retry the fallback drivers would have made.
///
/// `mu_strategy_fallback` and `l1_auto_fallback` re-run the solve when the first
/// attempt lands in `Solved_To_Acceptable_Level` or
/// `Maximum_Iterations_Exceeded`. The veto converts non-success verdicts, so on
/// paper it could turn a status that *would* have triggered a retry into one
/// that does not — silently costing the caller a recovery attempt that finds a
/// genuinely converged point.
///
/// The case analysis says it cannot: the fallback restores under `Success` only
/// when a *strict* certificate was refused, and a refused strict certificate is
/// exactly what the baseline would have returned as `Success` — so neither arm
/// retries. A refused acceptable-level termination restores as
/// `StopAtAcceptablePoint`, which still triggers the retry. But the same kind of
/// case analysis has been wrong twice in this work, so it is measured.
#[test]
fn the_veto_does_not_suppress_a_fallback_retry() {
    for (flag, a, k) in [
        ("mu_strategy_fallback", 1e5, 10.0),
        ("mu_strategy_fallback", 1e3, 10.0),
        ("l1_fallback_on_restoration_failure", 1e5, 10.0),
        ("l1_fallback_on_restoration_failure", 1e3, 10.0),
    ] {
        let solve = |threshold: Number| {
            let mut app = IpoptApplication::new();
            app.options_mut()
                .set_string_value(flag, "yes", true, false)
                .unwrap();
            app.options_mut()
                .set_numeric_value("obj_scale_certificate_threshold", threshold, true, false)
                .unwrap();
            app.options_mut()
                .set_integer_value("max_iter", 300, true, false)
                .unwrap();
            app.options_mut()
                .set_integer_value("print_level", 0, true, false)
                .unwrap();
            app.initialize().unwrap();
            let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(AcceptableOnly { a, amp: 1.0, k }));
            let st = app.optimize_tnlp(t);
            (st, app.statistics().final_objective)
        };
        let (bs, bo) = solve(0.0);
        let (vs, vo) = solve(1e-4);
        eprintln!("{flag} a={a:e}: baseline {bs:?} f={bo:.6e} | veto {vs:?} f={vo:.6e}");
        // The retry promotes only on Solve_Succeeded, so the observable
        // guarantee is the same one as everywhere else: never a worse status,
        // never a worse point.
        assert!(
            !(succeeded(bs) && !succeeded(vs)),
            "{flag} a={a:e}: baseline ended {bs:?} but the veto gave {vs:?} — a retry was \
             suppressed or a status lost"
        );
        assert!(
            vo <= bo + 1e-9 * bo.abs().max(1.0),
            "{flag} a={a:e}: veto objective {vo:.6e} worse than baseline {bo:.6e}"
        );
    }
}
