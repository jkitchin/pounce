//! Phase 6 acceptance (issue #495): every column whose declared bound is
//! active at the optimum must come back with a bound multiplier, so that
//! what postsolve reports is a KKT multiplier set for the *original*
//! problem.
//!
//! #493 moved a multiplier the solver *reported* to the column that owns
//! the bound. This is the case where the solver reported none: accumulated
//! bound transfers squeeze a survivor's reduced box down to a single point,
//! the solver treats that column as a fixed variable and drops it, and no
//! bound multiplier is produced at all — while the full-space cluster it
//! stands for is sitting on a bound that needs one. There is nothing to
//! re-attribute, so postsolve has to derive it.
//!
//! The oracle here is the textbook KKT system, evaluated from the fixture's
//! own analytic derivatives: primal feasibility, stationarity
//! `∇f + Jᵀλ − z_l + z_u = 0`, dual feasibility, and bound
//! complementarity. `pounce verify` cannot stand in for it — it reports
//! *bound-projected* stationarity, which projects out exactly the
//! component a bound multiplier carries.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_presolve::{LinearEqElimTnlp, PresolveOptions, PresolveTnlp};
use std::cell::RefCell;
use std::rc::Rc;

/// `min Σ (x_j − t_j)^p` over linear equality rows and a box.
///
/// Separable and polynomial, so every derivative the KKT oracle needs is
/// available in closed form on the test side rather than read back out of
/// the solver.
#[derive(Clone)]
struct Model {
    targets: Vec<Number>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    /// `Σ a_j x_j = b`, one entry per row.
    rows: Vec<(Vec<(usize, Number)>, Number)>,
    power: i32,
}

impl Model {
    fn n(&self) -> usize {
        self.targets.len()
    }

    fn grad(&self, x: &[Number]) -> Vec<Number> {
        let p = self.power;
        (0..self.n())
            .map(|j| p as Number * (x[j] - self.targets[j]).powi(p - 1))
            .collect()
    }

    /// `∇f + Jᵀλ − z_l + z_u`, in POUNCE's `finalize_solution` convention.
    fn stationarity(&self, c: &Captured) -> Vec<Number> {
        let mut r = self.grad(&c.x);
        for (i, (entries, _)) in self.rows.iter().enumerate() {
            for &(j, a) in entries {
                r[j] += a * c.lambda[i];
            }
        }
        for j in 0..self.n() {
            r[j] += -c.z_l[j] + c.z_u[j];
        }
        r
    }
}

#[derive(Debug, Clone, Default)]
struct Captured {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
}

struct Fixture {
    model: Model,
    captured: Option<Captured>,
}

impl TNLP for Fixture {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let nnz_jac: usize = self.model.rows.iter().map(|(e, _)| e.len()).sum();
        Some(NlpInfo {
            n: self.model.n() as i32,
            m: self.model.rows.len() as i32,
            nnz_jac_g: nnz_jac as i32,
            nnz_h_lag: self.model.n() as i32,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.model.x_l);
        b.x_u.copy_from_slice(&self.model.x_u);
        for (i, (_, rhs)) in self.model.rows.iter().enumerate() {
            b.g_l[i] = *rhs;
            b.g_u[i] = *rhs;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for (j, slot) in sp.x.iter_mut().enumerate() {
            // Inside the box, but nowhere near the answer.
            let lo = self.model.x_l[j].max(-1e3);
            let hi = self.model.x_u[j].min(1e3);
            *slot = 0.5 * (lo + hi);
        }
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        for t in types.iter_mut() {
            *t = Linearity::Linear;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(
            (0..self.model.n())
                .map(|j| (x[j] - self.model.targets[j]).powi(self.model.power))
                .sum(),
        )
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&self.model.grad(x));
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for (i, (entries, _)) in self.model.rows.iter().enumerate() {
            g[i] = entries.iter().map(|&(j, a)| a * x[j]).sum();
        }
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        let mut k = 0;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (i, (entries, _)) in self.model.rows.iter().enumerate() {
                    for &(j, _) in entries {
                        irow[k] = i as i32;
                        jcol[k] = j as i32;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                for (entries, _) in &self.model.rows {
                    for &(_, a) in entries {
                        values[k] = a;
                        k += 1;
                    }
                }
            }
        }
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
                for j in 0..self.model.n() {
                    irow[j] = j as i32;
                    jcol[j] = j as i32;
                }
            }
            SparsityRequest::Values { values } => {
                // The rows are linear, so λ contributes no curvature.
                let x = x.expect("values need a point");
                let p = self.model.power;
                for j in 0..self.model.n() {
                    values[j] = obj_factor
                        * (p * (p - 1)) as Number
                        * (x[j] - self.model.targets[j]).powi(p - 2);
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.captured = Some(Captured {
            x: sol.x.to_vec(),
            z_l: sol.z_l.to_vec(),
            z_u: sol.z_u.to_vec(),
            lambda: sol.lambda.to_vec(),
        });
    }
}

fn opts(linear_eq_reduction: bool) -> PresolveOptions {
    PresolveOptions {
        enabled: true,
        linear_eq_reduction,
        // Everything else off, so any dual that moves moved for the reason
        // under test.
        bound_tightening: false,
        redundant_constraint_removal: false,
        licq_check: false,
        warm_z_bounds: false,
        ..PresolveOptions::defaults()
    }
}

struct Outcome {
    captured: Captured,
    reduced_n: i32,
}

fn solve(model: &Model, linear_eq_reduction: bool) -> Outcome {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    // Unscaled, so the reported duals are directly comparable against the
    // fixture's own analytic gradient.
    app.options_mut()
        .set_string_value("nlp_scaling_method", "none", true, false)
        .unwrap();

    let concrete = Rc::new(RefCell::new(Fixture {
        model: model.clone(),
        captured: None,
    }));
    let presolve = Rc::new(RefCell::new(PresolveTnlp::new(
        Rc::clone(&concrete) as Rc<RefCell<dyn TNLP>>,
        opts(linear_eq_reduction),
    )));
    let elim = Rc::new(RefCell::new(LinearEqElimTnlp::new(
        Rc::clone(&presolve) as Rc<RefCell<dyn TNLP>>,
        opts(linear_eq_reduction),
    )));
    let info = elim.borrow_mut().get_nlp_info().expect("dims");
    let _ = app.optimize_tnlp(Rc::clone(&elim) as Rc<RefCell<dyn TNLP>>);

    Outcome {
        captured: concrete.borrow().captured.clone().expect("finalized"),
        reduced_n: info.n,
    }
}

/// The full KKT system, in the original variable space. Returns the list of
/// violations, so a failure names which condition broke rather than only
/// that one did.
fn kkt_violations(model: &Model, c: &Captured, tol: Number) -> Vec<String> {
    let mut out = Vec::new();
    for (i, (entries, rhs)) in model.rows.iter().enumerate() {
        let lhs: Number = entries.iter().map(|&(j, a)| a * c.x[j]).sum();
        if (lhs - rhs).abs() > tol {
            out.push(format!("row {i} violated: {lhs} != {rhs}"));
        }
    }
    for (j, r) in model.stationarity(c).iter().enumerate() {
        if r.abs() > tol {
            out.push(format!("stationarity at column {j} = {r}"));
        }
    }
    for j in 0..model.n() {
        if c.z_l[j] < -tol {
            out.push(format!("z_l[{j}] = {} is negative", c.z_l[j]));
        }
        if c.z_u[j] < -tol {
            out.push(format!("z_u[{j}] = {} is negative", c.z_u[j]));
        }
        if c.x[j] < model.x_l[j] - tol || c.x[j] > model.x_u[j] + tol {
            out.push(format!("x[{j}] = {} is outside its box", c.x[j]));
        }
        let slack_l = (c.x[j] - model.x_l[j]).abs();
        let slack_u = (c.x[j] - model.x_u[j]).abs();
        if c.z_l[j].abs() > tol && slack_l > 1e-5 {
            out.push(format!(
                "complementarity: z_l[{j}] = {} on a bound {} away",
                c.z_l[j], slack_l
            ));
        }
        if c.z_u[j].abs() > tol && slack_u > 1e-5 {
            out.push(format!(
                "complementarity: z_u[{j}] = {} on a bound {} away",
                c.z_u[j], slack_u
            ));
        }
    }
    out
}

fn assert_kkt(tag: &str, model: &Model, c: &Captured) {
    let v = kkt_violations(model, c, 1e-4);
    assert!(v.is_empty(), "{tag}: {v:#?}\nduals = {c:?}");
}

/// The issue's minimal case.
///
/// ```text
///   min (x0−4)⁴ + (x1−4)⁴   s.t.  x0 − x1 = 0,  x0 ∈ [1,5],  x1 ∈ [−5,1]
/// ```
///
/// The transfer squeezes `x1`'s reduced box to the single point `{1}`, so
/// the solver drops the column and reports no multiplier for it. The
/// optimum is `x = (1,1)` with `∇f = (−108, −108)`; `λ = 108`,
/// `z_u[1] = 216` closes stationarity.
#[test]
fn a_collapsed_reduced_box_still_reports_its_bound_multiplier() {
    let model = Model {
        targets: vec![4.0, 4.0],
        x_l: vec![1.0, -5.0],
        x_u: vec![5.0, 1.0],
        rows: vec![(vec![(0, 1.0), (1, -1.0)], 0.0)],
        power: 4,
    };
    let on = solve(&model, true);
    assert_eq!(on.reduced_n, 1, "one column should be gone");
    assert!((on.captured.x[0] - 1.0).abs() < 1e-6, "{:?}", on.captured.x);
    assert!((on.captured.x[1] - 1.0).abs() < 1e-6, "{:?}", on.captured.x);
    assert_kkt("reduction on", &model, &on.captured);

    // The multiplier the reduced solve never produced: 216 on x1's upper
    // bound, which is the one the collapsed box came to rest on.
    assert!(
        (on.captured.z_u[1] - 216.0).abs() < 1e-2,
        "z_u = {:?}, λ = {:?}",
        on.captured.z_u,
        on.captured.lambda
    );

    // And the bare solve agrees the point is what it is.
    let off = solve(&model, false);
    assert_kkt("reduction off", &model, &off.captured);
    for j in 0..2 {
        assert!(
            (on.captured.x[j] - off.captured.x[j]).abs() < 1e-5,
            "primal moved: {:?} vs {:?}",
            on.captured.x,
            off.captured.x
        );
    }
}

/// The same collapse with the transfer running the other way: `x1`'s box is
/// the wide one, so the point the cluster comes to rest on is the *lower*
/// end. The multiplier has to change sides with it.
#[test]
fn a_collapsed_box_resting_on_a_lower_bound_reports_a_lower_multiplier() {
    let model = Model {
        targets: vec![-4.0, -4.0],
        x_l: vec![-5.0, -1.0],
        x_u: vec![-1.0, 5.0],
        rows: vec![(vec![(0, 1.0), (1, -1.0)], 0.0)],
        power: 4,
    };
    let on = solve(&model, true);
    assert_eq!(on.reduced_n, 1);
    assert!((on.captured.x[0] + 1.0).abs() < 1e-6, "{:?}", on.captured.x);
    assert_kkt("reduction on", &model, &on.captured);
    assert!(
        on.captured.z_l[1] > 1e-3,
        "no multiplier on the active lower bound: {:?}",
        on.captured
    );
}

/// A negative substitution coefficient reverses the interval on the way
/// across, so the survivor's collapsed point sits on the *opposite* side of
/// the eliminated column's box from its own.
#[test]
fn a_negative_coefficient_does_not_lose_the_multiplier() {
    let model = Model {
        targets: vec![4.0, -4.0],
        x_l: vec![1.0, -1.0],
        x_u: vec![5.0, 5.0],
        // x0 + x1 = 0, i.e. x0 = −x1.
        rows: vec![(vec![(0, 1.0), (1, 1.0)], 0.0)],
        power: 4,
    };
    let on = solve(&model, true);
    assert_eq!(on.reduced_n, 1);
    assert!((on.captured.x[0] - 1.0).abs() < 1e-6, "{:?}", on.captured.x);
    assert!((on.captured.x[1] + 1.0).abs() < 1e-6, "{:?}", on.captured.x);
    assert_kkt("reduction on", &model, &on.captured);
}

/// The survivor is not always the column that can carry the multiplier.
///
/// ```text
///   min (x0−10)⁴ + (x1−10)⁴   s.t.  x0 − x1 = 0,  x0 ∈ [1,3],  x1 ∈ [3,5]
/// ```
///
/// `x0` folds onto `x1`, whose reduced box collapses to `{3}`. The
/// objective pulls upwards, so the multiplier the collapse hides is an
/// *upper*-bound one — and `x1 = 3` is sitting on its own **lower** bound,
/// nowhere near an upper one. The only column that can carry it is `x0`,
/// which is exactly the column whose box supplied the collapsing side.
/// Getting there means moving the multiplier across the cluster and
/// re-solving the consumed row for it.
#[test]
fn a_multiplier_the_survivor_cannot_carry_moves_to_the_column_that_can() {
    let model = Model {
        targets: vec![10.0, 10.0],
        x_l: vec![1.0, 3.0],
        x_u: vec![3.0, 5.0],
        rows: vec![(vec![(0, 1.0), (1, -1.0)], 0.0)],
        power: 4,
    };
    let on = solve(&model, true);
    assert_eq!(on.reduced_n, 1);
    assert!((on.captured.x[0] - 3.0).abs() < 1e-6, "{:?}", on.captured.x);
    assert_kkt("reduction on", &model, &on.captured);
    // ∇f = (−1372, −1372) at x = (3,3); λ = −1372 leaves 2744 for x0's
    // upper bound to carry.
    assert!(
        (on.captured.z_u[0] - 2744.0).abs() < 1e-1,
        "expected z_u[0] = 2744, got {:?}",
        on.captured
    );
    assert!(
        on.captured.z_l[1].abs() < 1e-4 && on.captured.z_u[1].abs() < 1e-4,
        "the survivor kept a multiplier it cannot own: {:?}",
        on.captured
    );

    // Flip the pull and the same collapse is carried by the survivor's own
    // lower bound instead, with no hand-off at all.
    let mirrored = Model {
        targets: vec![-10.0, -10.0],
        ..model.clone()
    };
    let on = solve(&mirrored, true);
    assert_kkt("mirrored", &mirrored, &on.captured);
    assert!(
        on.captured.z_l[1] > 1e-3,
        "expected the survivor to carry this one: {:?}",
        on.captured
    );
}

/// A chain, so the residual has to travel: `x0 = x1 = x2`, with the boxes
/// arranged so the intersection is a single point contributed by the
/// column at the far end of the chain from the survivor.
#[test]
fn a_chain_whose_intersection_is_a_point_still_closes_stationarity() {
    let model = Model {
        targets: vec![4.0, 4.0, 4.0],
        x_l: vec![1.0, -5.0, -5.0],
        x_u: vec![5.0, 1.0, 8.0],
        rows: vec![
            (vec![(0, 1.0), (1, -1.0)], 0.0),
            (vec![(1, 1.0), (2, -1.0)], 0.0),
        ],
        power: 4,
    };
    let on = solve(&model, true);
    assert_eq!(on.reduced_n, 1, "two columns should be gone");
    for j in 0..3 {
        assert!((on.captured.x[j] - 1.0).abs() < 1e-6, "{:?}", on.captured.x);
    }
    assert_kkt("reduction on", &model, &on.captured);
}

/// The boundary of the fix, pinned so it cannot drift.
///
/// ```text
///   min (x0−4)² + (x1−4)² + (x2−1)²   s.t.  2·x0 + 3·x1 = 6,  x1 ∈ [1,1]
/// ```
///
/// `x1` is a constant before the sweep starts, so the row is a singleton in
/// `x0` and pins it at `1.5`; stationarity at `x0` fixes `λ = 2.5`, and the
/// `3·λ` that lands on `x1` is carried by nothing. That is *not* a Phase 6
/// defect: the model declares `x1` fixed, so the solver drops it as a
/// parameter and reports no multiplier for it whether or not the reduction
/// runs. The bar here is parity with the bare solve, not a KKT point —
/// manufacturing a multiplier a no-presolve solve does not report would be
/// a divergence, and belongs to `fixed_variable_treatment` if anywhere.
#[test]
fn a_declared_fixed_column_reports_exactly_what_the_bare_solve_reports() {
    let model = Model {
        targets: vec![4.0, 4.0, 1.0],
        x_l: vec![-1e19, 1.0, -1e19],
        x_u: vec![1e19, 1.0, 1e19],
        rows: vec![(vec![(0, 2.0), (1, 3.0)], 6.0)],
        power: 2,
    };
    let on = solve(&model, true);
    let off = solve(&model, false);
    assert!((on.captured.x[0] - 1.5).abs() < 1e-6, "{:?}", on.captured.x);
    assert_eq!(
        kkt_violations(&model, &on.captured, 1e-4),
        kkt_violations(&model, &off.captured, 1e-4),
        "the reduction changed what a declared-fixed column reports:\n\
         on = {:?}\noff = {:?}",
        on.captured,
        off.captured
    );
    for j in 0..3 {
        assert!((on.captured.z_l[j] - off.captured.z_l[j]).abs() < 1e-6);
        assert!((on.captured.z_u[j] - off.captured.z_u[j]).abs() < 1e-6);
    }
}

/// Nothing above may cost the models that were already fine: where every
/// eliminated column is interior to its box, the duals must not move.
#[test]
fn an_interior_reduction_reports_exactly_what_it_did_before() {
    let model = Model {
        targets: vec![4.0, 4.0],
        x_l: vec![-100.0, -100.0],
        x_u: vec![100.0, 100.0],
        rows: vec![(vec![(0, 1.0), (1, -2.0)], 1.0)],
        power: 4,
    };
    let on = solve(&model, true);
    assert_eq!(on.reduced_n, 1);
    assert_kkt("reduction on", &model, &on.captured);
    for j in 0..2 {
        assert!(
            on.captured.z_l[j].abs() < 1e-6 && on.captured.z_u[j].abs() < 1e-6,
            "a bound multiplier appeared on an interior column: {:?}",
            on.captured
        );
    }
}

/// A sweep over clusters whose boxes touch the same point from every
/// combination of sides. The oracle is the whole KKT system, so this covers
/// the cases the named tests above single out and the ones between them —
/// in particular the ones where the collapsed point is contributed by a
/// column that is *not* the survivor, so the multiplier has to be handed
/// across the cluster.
///
/// Every column's box is built around a shared witness point, so the models
/// are feasible by construction and a stand-down is a bug in the generator
/// rather than a legitimate outcome to skip past.
#[test]
fn randomized_clusters_are_kkt_points_in_the_original_space() {
    // A deterministic LCG: the point is coverage, not entropy, and a test
    // that fails on Tuesdays is worse than one that fails never.
    let mut seed: u64 = 0x5DEE_CE66_D125_1F03;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f64) / ((1u64 << 31) as f64)
    };
    let mut collapsed = 0usize;
    for case in 0..60 {
        let n = 2 + (case % 4);
        // x_j = c_j · v, with the whole cluster passing through v = 1.
        let mut c = vec![1.0 as Number];
        let mut rows = Vec::new();
        for j in 0..n - 1 {
            let r = [1.0, -1.0, 2.0, -0.5][(next() * 4.0) as usize % 4];
            c.push(c[j] * r);
            rows.push((vec![(j + 1, 1.0), (j, -r)], 0.0));
        }
        // A margin of zero puts the witness point exactly on that bound,
        // which is what makes the intersection collapse.
        let mut margin = || {
            if next() < 0.45 {
                0.0
            } else {
                0.2 + 2.0 * next()
            }
        };
        let mut x_l = Vec::new();
        let mut x_u = Vec::new();
        for &cj in &c {
            let (mut lo, mut hi) = (margin(), margin());
            if lo == 0.0 && hi == 0.0 {
                // A both-sides-zero margin declares the column fixed, which
                // the solver drops as a parameter with or without the
                // reduction; that boundary has its own test.
                hi = 1.0;
            }
            x_l.push(cj - lo);
            x_u.push(cj + hi);
        }
        let targets: Vec<Number> = (0..n).map(|_| 8.0 * next() - 4.0).collect();
        let model = Model {
            targets,
            x_l,
            x_u,
            rows,
            power: 4,
        };
        let on = solve(&model, true);
        assert!(
            (on.reduced_n as usize) < model.n(),
            "case {case}: the reduction stood down on a feasible cluster"
        );
        if (on.captured.z_l.iter().chain(on.captured.z_u.iter())).any(|&z| z > 1e-6) {
            collapsed += 1;
        }
        let v = kkt_violations(&model, &on.captured, 1e-4);
        assert!(
            v.is_empty(),
            "case {case}: {v:#?}\nboxes = {:?} / {:?}, targets = {:?}\nrows = {:?}\nduals = {:?}",
            model.x_l,
            model.x_u,
            model.targets,
            model.rows,
            on.captured
        );
    }
    assert!(
        collapsed >= 15,
        "only {collapsed} cases came to rest on a bound; the generator has \
         stopped exercising what this test is for"
    );
}
