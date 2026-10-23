//! Adapter from [`crate::ipopt_nlp::IpoptNlp`] (the rich IPM-
//! shaped NLP trait pounce-algorithm shares with the IPOPT
//! lineage) to [`crate::sqp::SqpProblemSpec`] (the minimal
//! evaluation surface the SQP outer loop binds against).
//!
//! Lets `SqpAlgorithm` consume any NLP that the existing IPM
//! `IpoptAlgorithm` consumes — same `.nl` files via the AMPL
//! frontend, same CUTEst harness, same Python bindings — without
//! duplicating the NLP layer.
//!
//! Conversions:
//! - Slice ↔ `DenseVector` for inputs/outputs (per-call allocation;
//!   the IPM does the same inside `IpoptCalculatedQuantities`).
//! - `eval_c` and `eval_d` combined into a single constraint
//!   vector (equalities first, inequalities after). The combined
//!   bounds set `bl = bu = 0` for equality rows, `bl = d_l[i]`,
//!   `bu = d_u[i]` for inequality rows.
//! - `eval_jac_c` and `eval_jac_d` combined into a single
//!   sparse-triplet Jacobian (inequality-row indices shifted by
//!   `m_c`).
//! - `eval_h(x, 1.0, λ[..m_c], λ[m_c..])` for the Lagrangian
//!   Hessian. The SQP multiplier vector `λ_g` is layout-
//!   compatible: first `m_c` entries are `y_c`, next `m_d` are
//!   `y_d`.

use crate::ipopt_nlp::IpoptNlp;
use crate::sqp::problem::SqpProblemSpec;
use crate::sqp::qp_assembly::Triplet;
use pounce_common::Number;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::expansion_matrix::ExpansionMatrix;
use pounce_linalg::triplet::{GenTMatrix, SymTMatrix};
use std::cell::RefCell;
use std::rc::Rc;

pub struct IpoptNlpAdapter {
    nlp: Rc<RefCell<dyn IpoptNlp>>,
    n: usize,
    m_c: usize,
    m_d: usize,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    d_l: Vec<Number>,
    d_u: Vec<Number>,
    x_init: Vec<Number>,
    x_space: Rc<DenseVectorSpace>,
    c_space: Rc<DenseVectorSpace>,
    d_space: Rc<DenseVectorSpace>,
}

impl IpoptNlpAdapter {
    /// Build the adapter from an IpoptNlp handle. Dimensions are
    /// queried directly from `Nlp::n()`, `Nlp::m_eq()`,
    /// `Nlp::m_ineq()`.
    pub fn new(nlp: Rc<RefCell<dyn IpoptNlp>>) -> Self {
        let (n, m_c, m_d) = {
            let b = nlp.borrow();
            (b.n() as usize, b.m_eq() as usize, b.m_ineq() as usize)
        };
        let x_space = DenseVectorSpace::new(n as i32);
        let c_space = DenseVectorSpace::new(m_c as i32);
        let d_space = DenseVectorSpace::new(m_d as i32);

        // Extract bounds and initial point from the NLP. IpoptNlp
        // exposes bounds in *compressed* form (length = number of
        // entries with a finite bound); SQP wants full-length
        // vectors (length n / m_d) with ±∞ for unbounded entries.
        // The expansion matrices `px_l`, `px_u`, `pd_l`, `pd_u`
        // own the small→large index map; use them to scatter.
        let (x_l, x_u, d_l, d_u, x_init) = {
            let mut n_borrow = nlp.borrow_mut();
            let x_l_small = vec_from_dyn(n_borrow.x_l());
            let x_u_small = vec_from_dyn(n_borrow.x_u());
            let d_l_small = if m_d > 0 {
                vec_from_dyn(n_borrow.d_l())
            } else {
                Vec::new()
            };
            let d_u_small = if m_d > 0 {
                vec_from_dyn(n_borrow.d_u())
            } else {
                Vec::new()
            };
            let px_l = n_borrow.px_l();
            let px_u = n_borrow.px_u();
            let pd_l = n_borrow.pd_l();
            let pd_u = n_borrow.pd_u();
            let x_l = scatter_bound(&*px_l, &x_l_small, n, Number::NEG_INFINITY);
            let x_u = scatter_bound(&*px_u, &x_u_small, n, Number::INFINITY);
            let d_l = if m_d > 0 {
                scatter_bound(&*pd_l, &d_l_small, m_d, Number::NEG_INFINITY)
            } else {
                Vec::new()
            };
            let d_u = if m_d > 0 {
                scatter_bound(&*pd_u, &d_u_small, m_d, Number::INFINITY)
            } else {
                Vec::new()
            };
            let mut x = x_space.make_new_dense();
            let _ = n_borrow.get_starting_x(&mut x);
            let x_init = x.expanded_values();
            (x_l, x_u, d_l, d_u, x_init)
        };

        Self {
            nlp,
            n,
            m_c,
            m_d,
            x_l,
            x_u,
            d_l,
            d_u,
            x_init,
            x_space,
            c_space,
            d_space,
        }
    }

    /// The same adapter, but reporting the bounds the user *declared* rather
    /// than the live ones the interior method widened by
    /// `bound_relax_factor`.
    ///
    /// For the interior iteration the widening is the point — it is what
    /// keeps a strictly-interior iterate from being pinned against a bound
    /// it must approach. For anything asking *where the solution sits
    /// relative to the model*, it inverts the answer: a point exactly on a
    /// declared bound is a full `1e-8` inside the relaxed one, so an
    /// activity test against the live bounds calls the binding constraint
    /// inactive, and a pivot against them stops short of it. That is the
    /// difference between crossover identifying the active set and
    /// crossover identifying nothing.
    ///
    /// Falls back to the live bounds for any block the NLP does not track
    /// (the trait accessors default to `None`).
    pub fn new_with_declared_bounds(nlp: Rc<RefCell<dyn IpoptNlp>>) -> Self {
        let mut me = Self::new(Rc::clone(&nlp));
        let (n, m_d) = (me.n, me.m_d);
        let b = nlp.borrow();
        if let Some((x_l_small, x_u_small)) = b.declared_x_bounds() {
            me.x_l = scatter_bound(&*b.px_l(), &x_l_small, n, Number::NEG_INFINITY);
            me.x_u = scatter_bound(&*b.px_u(), &x_u_small, n, Number::INFINITY);
        }
        if m_d > 0 {
            if let Some((d_l_small, d_u_small)) = b.declared_d_bounds() {
                me.d_l = scatter_bound(&*b.pd_l(), &d_l_small, m_d, Number::NEG_INFINITY);
                me.d_u = scatter_bound(&*b.pd_u(), &d_u_small, m_d, Number::INFINITY);
            }
        }
        drop(b);
        me
    }

    /// Constraint violation of the model **as the user declared it**, in
    /// user units, at `x`: the general rows (unscaled, against the declared
    /// bounds) and the variable box (`None` when the declared box was never
    /// snapshotted). The SQP's own residuals are in the scaled space, where
    /// gradient-based scaling can shrink a row's violation by orders of
    /// magnitude (gh#1054: a violation of `1` read `3.3e-7`), so these are
    /// what an acceptance decision about the user's problem has to read.
    pub fn declared_violation(&self, x: &[Number]) -> (Number, Option<Number>) {
        let x_dv = self.dv_from_slice(&self.x_space, x);
        let mut c = self.c_space.make_new_dense();
        let mut d = self.d_space.make_new_dense();
        {
            let mut nlp = self.nlp.borrow_mut();
            if self.m_c > 0 {
                nlp.eval_c(&x_dv, &mut c);
            }
            if self.m_d > 0 {
                nlp.eval_d(&x_dv, &mut d);
            }
        }
        let nlp = self.nlp.borrow();
        let rows = crate::ipopt_cq::unscaled_nlp_constraint_violation_max(&*nlp, &c, &d);
        (rows, nlp.declared_box_violation(&x_dv))
    }

    /// `|df|`, the objective scaling factor's magnitude (`1` when unset):
    /// the divisor that takes a stationarity or complementarity residual
    /// from the scaled space to user units, as
    /// `IpoptCalculatedQuantities::curr_unscaled_dual_infeasibility_max` does.
    pub fn obj_scale(&self) -> Number {
        let df = self.nlp.borrow().obj_scaling_factor().abs();
        if df == 0.0 { 1.0 } else { df }
    }

    fn dv_from_slice(&self, space: &Rc<DenseVectorSpace>, s: &[Number]) -> DenseVector {
        let mut dv = space.make_new_dense();
        dv.set_values(s);
        dv
    }
}

impl SqpProblemSpec for IpoptNlpAdapter {
    fn n(&self) -> usize {
        self.n
    }
    fn m(&self) -> usize {
        self.m_c + self.m_d
    }

    fn x_init(&self) -> Vec<Number> {
        self.x_init.clone()
    }

    fn variable_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        (self.x_l.clone(), self.x_u.clone())
    }

    fn constraint_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        let mut bl = vec![0.0; self.m_c];
        bl.extend_from_slice(&self.d_l);
        let mut bu = vec![0.0; self.m_c];
        bu.extend_from_slice(&self.d_u);
        (bl, bu)
    }

    fn eval_f(&mut self, x: &[Number]) -> Number {
        let x_dv = self.dv_from_slice(&self.x_space, x);
        let mut nlp = self.nlp.borrow_mut();
        nlp.eval_f(&x_dv)
    }

    fn eval_grad_f(&mut self, x: &[Number]) -> Vec<Number> {
        let x_dv = self.dv_from_slice(&self.x_space, x);
        let mut g = self.x_space.make_new_dense();
        {
            let mut nlp = self.nlp.borrow_mut();
            nlp.eval_grad_f(&x_dv, &mut g);
        }
        g.expanded_values()
    }

    fn eval_c(&mut self, x: &[Number]) -> Vec<Number> {
        let x_dv = self.dv_from_slice(&self.x_space, x);
        let mut combined = Vec::with_capacity(self.m_c + self.m_d);
        if self.m_c > 0 {
            let mut c_out = self.c_space.make_new_dense();
            {
                let mut nlp = self.nlp.borrow_mut();
                nlp.eval_c(&x_dv, &mut c_out);
            }
            combined.extend(c_out.expanded_values());
        }
        if self.m_d > 0 {
            let mut d_out = self.d_space.make_new_dense();
            {
                let mut nlp = self.nlp.borrow_mut();
                nlp.eval_d(&x_dv, &mut d_out);
            }
            combined.extend(d_out.expanded_values());
        }
        combined
    }

    fn eval_jac_c(&mut self, x: &[Number]) -> Triplet {
        let x_dv = self.dv_from_slice(&self.x_space, x);
        let mut irow = Vec::new();
        let mut jcol = Vec::new();
        let mut vals = Vec::new();

        if self.m_c > 0 {
            let jac_c = {
                let mut nlp = self.nlp.borrow_mut();
                nlp.eval_jac_c(&x_dv)
            };
            let t = gen_t_downcast(&*jac_c);
            irow.extend_from_slice(t.irows());
            jcol.extend_from_slice(t.jcols());
            vals.extend_from_slice(t.values());
        }

        if self.m_d > 0 {
            let jac_d = {
                let mut nlp = self.nlp.borrow_mut();
                nlp.eval_jac_d(&x_dv)
            };
            let t = gen_t_downcast(&*jac_d);
            let shift = self.m_c as pounce_common::Index;
            irow.extend(t.irows().iter().map(|&r| r + shift));
            jcol.extend_from_slice(t.jcols());
            vals.extend_from_slice(t.values());
        }

        Triplet {
            n_rows: self.m_c + self.m_d,
            n_cols: self.n,
            irow,
            jcol,
            vals,
        }
    }

    fn eval_hess_lag(&mut self, x: &[Number], lambda_g: &[Number]) -> Triplet {
        let x_dv = self.dv_from_slice(&self.x_space, x);
        let y_c_dv = self.dv_from_slice(&self.c_space, &lambda_g[..self.m_c]);
        let y_d_dv = self.dv_from_slice(&self.d_space, &lambda_g[self.m_c..]);

        let h = {
            let mut nlp = self.nlp.borrow_mut();
            nlp.eval_h(&x_dv, 1.0, &y_c_dv, &y_d_dv)
        };
        let t = sym_t_downcast(&*h);
        Triplet {
            n_rows: self.n,
            n_cols: self.n,
            irow: t.irows().to_vec(),
            jcol: t.jcols().to_vec(),
            vals: t.values().to_vec(),
        }
    }
}

fn vec_from_dyn(v: &dyn pounce_linalg::Vector) -> Vec<Number> {
    let dv = v
        .as_any()
        .downcast_ref::<DenseVector>()
        .expect("IpoptNlp bound accessors must return DenseVector");
    dv.expanded_values()
}

/// Scatter a compressed bound vector (length = number of finite bounds)
/// into the full-length bound vector (length `n_large`), filling
/// not-in-map entries with `fill`. Uses the `ExpansionMatrix`'s
/// small→large index map.
fn scatter_bound(
    expansion: &dyn pounce_linalg::Matrix,
    small: &[Number],
    n_large: usize,
    fill: Number,
) -> Vec<Number> {
    let em = expansion
        .as_any()
        .downcast_ref::<ExpansionMatrix>()
        .expect("px_l / px_u / pd_l / pd_u must be ExpansionMatrix");
    let exp_pos = em.expanded_pos_indices();
    debug_assert_eq!(small.len(), exp_pos.len());
    let mut out = vec![fill; n_large];
    for (i, &pos) in exp_pos.iter().enumerate() {
        out[pos as usize] = small[i];
    }
    out
}

fn gen_t_downcast(m: &dyn pounce_linalg::Matrix) -> &GenTMatrix {
    m.as_any()
        .downcast_ref::<GenTMatrix>()
        .expect("IpoptNlp::eval_jac_* must return GenTMatrix")
}

fn sym_t_downcast(m: &dyn pounce_linalg::matrix::SymMatrix) -> &SymTMatrix {
    m.as_any()
        .downcast_ref::<SymTMatrix>()
        .expect("IpoptNlp::eval_h must return SymTMatrix")
}

/// Gather the compressed indices of the finite lower / upper variable bounds
/// — the small→large maps `px_l` and `px_u` own — as plain index vectors.
///
/// The IPM carries `z_l` and `z_u` in those *compressed* spaces (one entry
/// per finite bound), while the SQP / `pounce-qp` side carries one packed
/// multiplier per variable. Anything translating between the two engines
/// needs the maps, and there is exactly one correct source for them; reading
/// them off the NLP here keeps that from being re-derived (wrongly) at each
/// call site.
fn bound_maps(nlp: &Rc<RefCell<dyn IpoptNlp>>) -> (Vec<usize>, Vec<usize>) {
    let b = nlp.borrow();
    let idx = |m: &dyn pounce_linalg::Matrix| -> Vec<usize> {
        m.as_any()
            .downcast_ref::<ExpansionMatrix>()
            .expect("px_l / px_u must be ExpansionMatrix")
            .expanded_pos_indices()
            .iter()
            .map(|&p| p as usize)
            .collect()
    };
    let lo = idx(&*b.px_l());
    let up = idx(&*b.px_u());
    (lo, up)
}

/// Pack the IPM's compressed bound multipliers into the SQP convention:
/// one entry per variable, `λ_x = z_l − z_u`.
///
/// Both engines write stationarity as `∇f + Jᵀλ_g − λ_x` once `λ_x` is
/// packed this way (the IPM's own form is `∇f + Jᵀλ − z_l + z_u = 0`), so
/// this is a repacking and not a sign convention change.
pub fn pack_bound_multipliers(
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    z_l: &[Number],
    z_u: &[Number],
) -> Vec<Number> {
    let n = nlp.borrow().n() as usize;
    let (lo_map, up_map) = bound_maps(nlp);
    let mut out = vec![0.0; n];
    for (i, &pos) in lo_map.iter().enumerate() {
        if let Some(&v) = z_l.get(i) {
            out[pos] += v;
        }
    }
    for (i, &pos) in up_map.iter().enumerate() {
        if let Some(&v) = z_u.get(i) {
            out[pos] -= v;
        }
    }
    out
}

/// Inverse of [`pack_bound_multipliers`]: split a packed per-variable `λ_x`
/// back into the IPM's compressed `(z_l, z_u)`.
///
/// The split is by sign — positive mass to the lower bound, negative to the
/// upper — which is the only choice consistent with the sign restrictions
/// `z_l ≥ 0`, `z_u ≥ 0` and with complementarity: away from a degenerate
/// fixed variable at most one of the two can be nonzero, so the packing loses
/// nothing to recover. A variable *fixed* by equal bounds is the one case
/// where `λ_x` genuinely does not determine the pair; sign choice is the
/// established convention there and the two are interchangeable in every
/// downstream identity, since only their difference is ever used.
///
/// Mass that lands on a variable with no bound on the corresponding side has
/// nowhere to go and is dropped — that can only happen if a caller hands in a
/// multiplier violating the sign restrictions, and silently keeping it would
/// corrupt the stationarity residual the user is shown.
pub fn split_bound_multipliers(
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    lambda_x: &[Number],
) -> (Vec<Number>, Vec<Number>) {
    let (lo_map, up_map) = bound_maps(nlp);
    let z_l = lo_map
        .iter()
        .map(|&pos| lambda_x.get(pos).copied().unwrap_or(0.0).max(0.0))
        .collect();
    let z_u = up_map
        .iter()
        .map(|&pos| (-lambda_x.get(pos).copied().unwrap_or(0.0)).max(0.0))
        .collect();
    (z_l, z_u)
}

/// Split the inequality multipliers `y_d` into the IPM's compressed slack
/// bound multipliers `(v_l, v_u)`.
///
/// Stationarity of the barrier problem with respect to the slacks is
/// `−y_d − v_l + v_u = 0`, i.e. `v_l − v_u = −y_d`; with `v_l, v_u ≥ 0` and
/// complementarity that determines the pair up to the same degenerate case
/// [`split_bound_multipliers`] documents. Needed when writing a point
/// computed on the active-set side back onto an IPM iterate, whose slack
/// duals would otherwise still describe the interior point.
pub fn split_slack_multipliers(
    nlp: &Rc<RefCell<dyn IpoptNlp>>,
    y_d: &[Number],
) -> (Vec<Number>, Vec<Number>) {
    let b = nlp.borrow();
    let idx = |m: &dyn pounce_linalg::Matrix| -> Vec<usize> {
        m.as_any()
            .downcast_ref::<ExpansionMatrix>()
            .expect("pd_l / pd_u must be ExpansionMatrix")
            .expanded_pos_indices()
            .iter()
            .map(|&p| p as usize)
            .collect()
    };
    let lo_map = idx(&*b.pd_l());
    let up_map = idx(&*b.pd_u());
    let v_l = lo_map
        .iter()
        .map(|&pos| (-y_d.get(pos).copied().unwrap_or(0.0)).max(0.0))
        .collect();
    let v_u = up_map
        .iter()
        .map(|&pos| y_d.get(pos).copied().unwrap_or(0.0).max(0.0))
        .collect();
    (v_l, v_u)
}
