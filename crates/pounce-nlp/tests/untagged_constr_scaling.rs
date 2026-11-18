//! `nlp_scaling_untagged_constr=gradient-based` under `user-scaling`.
//!
//! The model is linear in its rows, so each row's Jacobian ∞-norm is a
//! constant the tests can state:
//!
//! * `c0 = 1000 x0 + x1 = 5`, an equality, row max 1000;
//! * `d0 = 2 x0 + 400 x1 <= 10`, row max 400;
//! * `d1 = x0 + x1 <= 100`, row max 1, below the cutoff.
//!
//! With the default cutoff of 100 the rule gives `c0` 0.1, `d0` 0.25 and
//! `d1` 1, the factors WaterTAP's `_constraint_autoscale_large_jac` gives
//! the same rows. Each test changes one input and checks the factors the
//! core installs.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::Number;
use pounce_nlp::orig_ipopt_nlp::{ScalingMethod, UntaggedConstrScaling};
use pounce_nlp::scaling_tnlp::wrap_with_scaling;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_nlp::tnlp_adapter::{DEFAULT_NLP_LOWER_BOUND_INF, DEFAULT_NLP_UPPER_BOUND_INF};
use pounce_nlp::{IpoptNlp, NoScaling, OrigIpoptNlp, TNLPAdapter};

/// What the fixture's `get_scaling_parameters` answers.
#[derive(Clone)]
struct Supplied {
    /// `false` makes the callback return `false`, as an `.nl` with no
    /// `scaling_factor` suffix does.
    any: bool,
    obj: Number,
    /// Per-row factor and whether it was given, in `g` order.
    g: [Number; 3],
    given: [bool; 3],
    /// Per-variable factors, applied by the `ScalingTnlp` wrapper.
    x: Option<[Number; 2]>,
}

impl Default for Supplied {
    fn default() -> Self {
        Self {
            any: true,
            obj: 1.0,
            g: [1.0; 3],
            given: [false; 3],
            x: None,
        }
    }
}

struct Fixture(Supplied);

impl TNLP for Fixture {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 3,
            nnz_jac_g: 6,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-10.0, -10.0]);
        b.x_u.copy_from_slice(&[10.0, 10.0]);
        b.g_l.copy_from_slice(&[
            5.0,
            DEFAULT_NLP_LOWER_BOUND_INF,
            DEFAULT_NLP_LOWER_BOUND_INF,
        ]);
        b.g_u.copy_from_slice(&[5.0, 10.0, 100.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            sp.x.copy_from_slice(&[1.0, 1.0]);
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1])
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 1000.0 * x[0] + x[1];
        g[1] = 2.0 * x[0] + 400.0 * x[1];
        g[2] = x[0] + x[1];
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
                irow.copy_from_slice(&[0, 0, 1, 1, 2, 2]);
                jcol.copy_from_slice(&[0, 1, 0, 1, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1000.0, 1.0, 2.0, 400.0, 1.0, 1.0]);
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[2.0 * obj_factor, 2.0 * obj_factor]);
            }
        }
        true
    }
    fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let s = &self.0;
        if !s.any {
            return false;
        }
        *req.obj_scaling = s.obj;
        *req.use_g_scaling = true;
        req.g_scaling.copy_from_slice(&s.g);
        req.g_scaling_given.copy_from_slice(&s.given);
        match s.x {
            Some(x) => {
                *req.use_x_scaling = true;
                req.x_scaling.copy_from_slice(&x);
            }
            None => *req.use_x_scaling = false,
        }
        true
    }
}

/// The knobs `determine_scaling_from_starting_point` takes, at the
/// registered defaults unless a test changes one.
#[derive(Clone, Copy)]
struct Knobs {
    rule: UntaggedConstrScaling,
    min_value: Number,
    constr_target: Number,
}

const DEFAULT: Knobs = Knobs {
    rule: UntaggedConstrScaling::GradientBased,
    min_value: 1e-8,
    constr_target: 0.0,
};

/// The objective factor and the `c` and `d` row factors the core installs,
/// with an absent vector written out as ones.
fn installed(s: Supplied, k: Knobs) -> (Number, Vec<Number>, Vec<Number>) {
    let wraps = s.x.is_some();
    let mut tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture(s)));
    if wraps {
        tnlp = wrap_with_scaling(
            tnlp,
            DEFAULT_NLP_LOWER_BOUND_INF,
            DEFAULT_NLP_UPPER_BOUND_INF,
        )
        .expect("valid factors")
        .expect("non-trivial factors wrap");
    }
    let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
    let mut nlp = OrigIpoptNlp::new(adapter, Rc::new(NoScaling)).unwrap();
    nlp.set_untagged_constr_scaling(k.rule);
    nlp.determine_scaling_from_starting_point(
        ScalingMethod::UserScaling,
        100.0,
        k.min_value,
        0.0,
        k.constr_target,
    );
    assert!(!nlp.user_x_scaling_rejected());
    (
        nlp.obj_scale_factor(),
        nlp.c_scale_vec().unwrap_or_else(|| vec![1.0]),
        nlp.d_scale_vec().unwrap_or_else(|| vec![1.0, 1.0]),
    )
}

fn assert_factors(label: &str, got: &[Number], want: &[Number]) {
    assert_eq!(got.len(), want.len(), "{label}: {got:?} against {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-15 * w.abs(),
            "{label}: {got:?} against {want:?}"
        );
    }
}

#[test]
fn a_row_with_a_user_factor_keeps_it_and_one_without_gets_the_rule() {
    let s = Supplied {
        g: [0.5, 1.0, 1.0],
        given: [true, false, false],
        ..Supplied::default()
    };
    let (_, c, d) = installed(s, DEFAULT);
    assert_factors("tagged c0", &c, &[0.5]);
    assert_factors("untagged d0 and d1", &d, &[100.0 / 400.0, 1.0]);
}

#[test]
fn the_default_installs_the_user_factors_unchanged() {
    let s = Supplied {
        g: [0.5, 1.0, 1.0],
        given: [true, false, false],
        ..Supplied::default()
    };
    let (obj, c, d) = installed(
        s,
        Knobs {
            rule: UntaggedConstrScaling::None,
            ..DEFAULT
        },
    );
    assert_eq!(obj, 1.0);
    assert_factors("c", &c, &[0.5]);
    assert_factors("d", &d, &[1.0, 1.0]);
}

#[test]
fn the_rule_reads_the_jacobian_after_the_variable_scaling() {
    // x1 scaled by 4 divides its column by 4: d0's row max falls from 400
    // to 100, at the cutoff and so unscaled, while c0's stays 1000.
    let s = Supplied {
        x: Some([1.0, 4.0]),
        ..Supplied::default()
    };
    let (_, c, d) = installed(s.clone(), DEFAULT);
    assert_factors("c0", &c, &[0.1]);
    assert_factors("d0 at the cutoff", &d, &[1.0, 1.0]);
    let (_, _, d_plain) = installed(Supplied { x: None, ..s }, DEFAULT);
    assert_factors("d0 without the variable factor", &d_plain, &[0.25, 1.0]);
}

#[test]
fn the_rule_is_floored_at_nlp_scaling_min_value() {
    let (_, c, d) = installed(
        Supplied::default(),
        Knobs {
            min_value: 0.2,
            ..DEFAULT
        },
    );
    assert_factors("c0 floored", &c, &[0.2]);
    assert_factors("d0 floored", &d, &[0.25, 1.0]);
}

#[test]
fn a_positive_constr_target_gradient_overrides_the_cutoff() {
    let (_, c, d) = installed(
        Supplied::default(),
        Knobs {
            constr_target: 10.0,
            ..DEFAULT
        },
    );
    assert_factors("c0", &c, &[10.0 / 1000.0]);
    assert_factors("d0 and d1", &d, &[10.0 / 400.0, 10.0]);
}

#[test]
fn a_model_with_no_scaling_at_all_gets_the_rule_on_every_row() {
    let s = Supplied {
        any: false,
        ..Supplied::default()
    };
    let (obj, c, d) = installed(s.clone(), DEFAULT);
    assert_eq!(obj, 1.0, "the objective is never scaled by the rule");
    assert_factors("c0", &c, &[0.1]);
    assert_factors("d0 and d1", &d, &[0.25, 1.0]);
    let (obj, c, d) = installed(
        s,
        Knobs {
            rule: UntaggedConstrScaling::None,
            ..DEFAULT
        },
    );
    assert_eq!(obj, 1.0);
    assert_factors("unscaled c", &c, &[1.0]);
    assert_factors("unscaled d", &d, &[1.0, 1.0]);
}

#[test]
fn the_objective_keeps_its_user_factor() {
    let s = Supplied {
        obj: 0.01,
        ..Supplied::default()
    };
    let (obj, _, _) = installed(s, DEFAULT);
    assert_eq!(obj, 0.01);
}

// ---------------------------------------------------------------------------
// The MPCC wrapper's product rows
// ---------------------------------------------------------------------------

use pounce_nlp::mpcc::{ComplementarityPair, Lowering, MpccTnlp};

/// What [`PairModel`]'s callback answers.
#[derive(Clone, Copy, Debug)]
enum PairAnswer {
    /// `false`, as an `.nl` with no `scaling_factor` suffix answers.
    Nothing,
    /// `true` with an objective factor and no row factors, as an `.nl`
    /// whose suffix names only the objective answers.
    ObjectiveOnly,
    /// `true` with row 0 given factor 1 and row 1 not given.
    Rows,
}

/// `min x² + y²` with `x >= 0 ⟂ y >= 0` written as row 0 (`x`) paired with
/// `y`, and an equality row 1, `1000 x + y = 5`, from `(500, 500)`. The
/// pair's product row `x·y` has Jacobian `(y, x) = (500, 500)` there, so
/// the gradient-based rule would give it 100 / 500 = 0.2. Row 1's would be
/// 0.1, and row 0's 1.
struct PairModel(PairAnswer);

impl TNLP for PairModel {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 3,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[DEFAULT_NLP_LOWER_BOUND_INF, 0.0]);
        b.x_u
            .copy_from_slice(&[DEFAULT_NLP_UPPER_BOUND_INF, DEFAULT_NLP_UPPER_BOUND_INF]);
        b.g_l.copy_from_slice(&[0.0, 5.0]);
        b.g_u.copy_from_slice(&[DEFAULT_NLP_UPPER_BOUND_INF, 5.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            sp.x.copy_from_slice(&[500.0, 500.0]);
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1])
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
        g[1] = 1000.0 * x[0] + x[1];
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
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1000.0, 1.0]);
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[2.0 * obj_factor, 2.0 * obj_factor]);
            }
        }
        true
    }
    fn finalize_solution(&mut self, _: Solution<'_>, _: &IpoptData, _: &IpoptCq) {}
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        match self.0 {
            PairAnswer::Nothing => false,
            PairAnswer::ObjectiveOnly => {
                *req.obj_scaling = 0.5;
                *req.use_g_scaling = false;
                true
            }
            PairAnswer::Rows => {
                *req.use_g_scaling = true;
                req.g_scaling.copy_from_slice(&[1.0, 1.0]);
                req.g_scaling_given.copy_from_slice(&[true, false]);
                true
            }
        }
    }
}

/// Every row factor the core installs on the lowered pair model, in no
/// particular order, with an absent block left out (it is all ones).
fn pair_model_factors(answer: PairAnswer, rule: UntaggedConstrScaling) -> Vec<Number> {
    let wrapped = MpccTnlp::new(
        Rc::new(RefCell::new(PairModel(answer))),
        vec![ComplementarityPair { row: 0, var: 1 }],
        Lowering::Ineq,
    )
    .expect("a valid pair");
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(wrapped));
    let adapter = Rc::new(RefCell::new(TNLPAdapter::new(tnlp).unwrap()));
    let mut nlp = OrigIpoptNlp::new(adapter, Rc::new(NoScaling)).unwrap();
    nlp.set_untagged_constr_scaling(rule);
    nlp.determine_scaling_from_starting_point(ScalingMethod::UserScaling, 100.0, 1e-8, 0.0, 0.0);
    let mut all = nlp.c_scale_vec().unwrap_or_default();
    all.extend(nlp.d_scale_vec().unwrap_or_default());
    all
}

/// The product row the wrapper adds keeps factor 1 under the rule,
/// whatever the inner model answered. The inner rows without a factor
/// still get it: 0.1 for row 1, and 1 for row 0, whose row max is 1.
#[test]
fn the_mpcc_product_row_keeps_factor_one_under_the_rule() {
    for answer in [
        PairAnswer::Nothing,
        PairAnswer::ObjectiveOnly,
        PairAnswer::Rows,
    ] {
        let all = pair_model_factors(answer, UntaggedConstrScaling::GradientBased);
        assert!(
            all.iter().all(|&s| s == 1.0 || (s - 0.1).abs() < 1e-15),
            "{answer:?}: the product row was rescaled: {all:?}"
        );
        assert_eq!(
            all.iter().filter(|&&s| (s - 0.1).abs() < 1e-15).count(),
            1,
            "{answer:?}: row 1 gets the rule: {all:?}"
        );
    }
}

/// Without the option the wrapper's answer installs no row factor, as
/// before it marked its rows.
#[test]
fn the_mpcc_wrapper_installs_no_row_factor_without_the_rule() {
    for answer in [
        PairAnswer::Nothing,
        PairAnswer::ObjectiveOnly,
        PairAnswer::Rows,
    ] {
        let all = pair_model_factors(answer, UntaggedConstrScaling::None);
        assert!(all.iter().all(|&s| s == 1.0), "{answer:?}: {all:?}");
    }
}
