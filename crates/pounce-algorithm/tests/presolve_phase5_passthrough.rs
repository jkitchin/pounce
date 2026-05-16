//! Phase 5 acceptance for pounce-presolve (#20):
//!
//! When presolve drops constraint rows, the user's metadata and
//! scaling for those rows must still survive the round-trip:
//!
//!   * `get_var_con_metadata` must publish per-(outer-row) values
//!     in kept-row order — not decline.
//!   * `get_scaling_parameters` must subset `g_scaling` to kept rows.
//!   * `finalize_metadata` must reinstate the full-sized vectors so
//!     the user's downstream consumers see inner-row indexing.

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, MetaData, NlpInfo, ScalingRequest,
    Solution, SparsityRequest, StartingPoint, TNLP,
};
use pounce_presolve::{PresolveOptions, PresolveTnlp};
use std::cell::RefCell;
use std::rc::Rc;

/// Same shape as the Phase 2 fixture, plus per-row metadata + scaling.
struct WithMetaScaling {
    final_con: RefCell<Option<MetaData>>,
}

impl WithMetaScaling {
    fn new() -> Self {
        Self {
            final_con: RefCell::new(None),
        }
    }
}

impl TNLP for WithMetaScaling {
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
        b.x_l.copy_from_slice(&[0.0, 0.0]);
        b.x_u.copy_from_slice(&[1.0, 1.0]);
        b.g_l.copy_from_slice(&[1.0, -10.0]);
        b.g_u.copy_from_slice(&[1.0, 10.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.5]);
        true
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types[0] = Linearity::Linear;
        types[1] = Linearity::Linear;
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 0.3).powi(2) + (x[1] - 0.3).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 0.3);
        g[1] = 2.0 * (x[1] - 0.3);
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        g[1] = x[0] - x[1];
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
                irow.copy_from_slice(&[0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0, 1.0, -1.0]);
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
    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        var.strings
            .insert("var_names".into(), vec!["a".into(), "b".into()]);
        con.strings.insert(
            "con_names".into(),
            vec!["budget".into(), "spread".into()],
        );
        con.integers.insert("priority".into(), vec![7 as Index, 3]);
        con.numerics.insert("weight".into(), vec![1.5, 0.5]);
        true
    }
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        *req.obj_scaling = 1.0;
        *req.use_x_scaling = false;
        *req.use_g_scaling = true;
        req.g_scaling.copy_from_slice(&[2.0, 4.0]);
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    fn finalize_metadata(&mut self, _var: &MetaData, con: &MetaData) {
        *self.final_con.borrow_mut() = Some(con.clone());
    }
}

fn build() -> (Rc<RefCell<WithMetaScaling>>, PresolveTnlp) {
    let concrete = Rc::new(RefCell::new(WithMetaScaling::new()));
    let dyn_inner: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let mut p = PresolveTnlp::new(
        dyn_inner,
        PresolveOptions {
            enabled: true,
            ..PresolveOptions::defaults()
        },
    );
    let _ = p.get_nlp_info();
    (concrete, p)
}

#[test]
fn phase5_con_metadata_projects_to_kept_row_only() {
    let (_inner, mut p) = build();
    let mut var = MetaData::default();
    let mut con = MetaData::default();
    assert!(p.get_var_con_metadata(&mut var, &mut con));
    assert_eq!(var.strings.get("var_names").unwrap().len(), 2);
    assert_eq!(con.strings.get("con_names").unwrap(), &vec!["budget".to_string()]);
    assert_eq!(con.integers.get("priority").unwrap(), &vec![7 as Index]);
    assert_eq!(con.numerics.get("weight").unwrap(), &vec![1.5]);
}

#[test]
fn phase5_g_scaling_subsets_to_kept_rows() {
    let (_inner, mut p) = build();
    let mut x_scaling = vec![0.0; 2];
    let mut g_scaling = vec![0.0; 1]; // m_out = 1 after row drop
    let mut obj = 0.0;
    let mut use_x = false;
    let mut use_g = false;
    let ok = p.get_scaling_parameters(ScalingRequest {
        obj_scaling: &mut obj,
        use_x_scaling: &mut use_x,
        x_scaling: &mut x_scaling,
        use_g_scaling: &mut use_g,
        g_scaling: &mut g_scaling,
    });
    assert!(ok);
    assert!(use_g);
    assert_eq!(g_scaling, vec![2.0], "kept row was index 0 with scale 2.0");
}

#[test]
fn phase5_finalize_metadata_expands_back_to_inner_rows() {
    let (inner, mut p) = build();
    // Caller sends outer (m_out=1) metadata; inner must receive m_in=2.
    let var = MetaData::default();
    let mut con = MetaData::default();
    con.strings.insert("post".into(), vec!["only-kept".into()]);
    con.numerics.insert("dual".into(), vec![42.0]);
    p.finalize_metadata(&var, &con);
    let inner_seen = inner
        .borrow()
        .final_con
        .borrow()
        .clone()
        .expect("inner finalize_metadata called");
    assert_eq!(
        inner_seen.strings.get("post").unwrap(),
        &vec!["only-kept".to_string(), String::new()]
    );
    assert_eq!(inner_seen.numerics.get("dual").unwrap(), &vec![42.0, 0.0]);
}
