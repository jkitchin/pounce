//! Phase 3 acceptance for pounce-presolve (#20):
//!
//! Build a tiny NLP with two *duplicate* linear equality rows so the
//! structural rank check has to fall back from `Full` to
//! `StructuralRank(1)`. The wrapper exposes the verdict via
//! `licq_verdict()` after `ensure_init` has run.

use pounce_common::types::Number;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_presolve::{LicqVerdict, PresolveOptions, PresolveTnlp};
use std::cell::RefCell;
use std::rc::Rc;

/// Two equality rows that both touch only column 0 ⇒ structural
/// matching size = 1 < m_eq = 2.
struct DuplicateEqs;

impl TNLP for DuplicateEqs {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-1.0, -1.0]);
        b.x_u.copy_from_slice(&[1.0, 1.0]);
        b.g_l.copy_from_slice(&[0.0, 0.0]);
        b.g_u.copy_from_slice(&[0.0, 0.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.5]);
        true
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types[0] = Linearity::Linear;
        types[1] = Linearity::Linear;
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + (x[1] - 0.3) * (x[1] - 0.3))
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * (x[1] - 0.3);
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
        g[1] = x[0];
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 0]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0]);
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
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
            }
        }
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Three distinct singleton equality rows: full structural rank.
struct DistinctEqs;

impl TNLP for DistinctEqs {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 3,
            nnz_jac_g: 3,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-1.0; 3]);
        b.x_u.copy_from_slice(&[1.0; 3]);
        b.g_l.copy_from_slice(&[0.0, 0.0, 0.0]);
        b.g_u.copy_from_slice(&[0.0, 0.0, 0.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0; 3]);
        true
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        for t in types.iter_mut() {
            *t = Linearity::Linear;
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x.iter().map(|v| v * v).sum())
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for (gi, &xi) in g.iter_mut().zip(x.iter()) {
            *gi = 2.0 * xi;
        }
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(x);
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
                irow.copy_from_slice(&[0, 1, 2]);
                jcol.copy_from_slice(&[0, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0, 1.0]);
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
                irow.copy_from_slice(&[0, 1, 2]);
                jcol.copy_from_slice(&[0, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                for v in values.iter_mut() {
                    *v = 2.0 * obj_factor;
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn build_and_init<T: TNLP + 'static>(inner: T, opts: PresolveOptions) -> PresolveTnlp {
    let inner_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(inner));
    let mut p = PresolveTnlp::new(inner_rc, opts);
    let _ = p.get_nlp_info();
    p
}

#[test]
fn phase3_duplicate_eqs_report_structural_rank_one() {
    // Phase 1+2 would otherwise pin x0=0 and drop both duplicate
    // rows as redundant; disable them so LICQ sees the original
    // structurally-deficient block.
    let p = build_and_init(
        DuplicateEqs,
        PresolveOptions {
            enabled: true,
            bound_tightening: false,
            redundant_constraint_removal: false,
            licq_check: true,
            ..PresolveOptions::defaults()
        },
    );
    let verdict = p.licq_verdict().cloned();
    assert!(
        matches!(verdict, Some(LicqVerdict::StructuralRank(1))),
        "expected StructuralRank(1), got {verdict:?}"
    );
}

#[test]
fn phase3_distinct_eqs_report_full_rank() {
    let p = build_and_init(
        DistinctEqs,
        PresolveOptions {
            enabled: true,
            ..PresolveOptions::defaults()
        },
    );
    let verdict = p.licq_verdict().cloned();
    assert!(
        matches!(verdict, Some(LicqVerdict::Full)),
        "expected Full, got {verdict:?}"
    );
}

#[test]
fn phase3_check_disabled_yields_no_verdict() {
    let p = build_and_init(
        DuplicateEqs,
        PresolveOptions {
            enabled: true,
            licq_check: false,
            ..PresolveOptions::defaults()
        },
    );
    assert!(p.licq_verdict().is_none(), "no verdict when check is off");
}
