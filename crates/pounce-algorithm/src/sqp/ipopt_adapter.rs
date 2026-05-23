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

        // Extract bounds and initial point from the NLP. Each
        // call borrows mutably so we sequence them.
        let (x_l, x_u, d_l, d_u, x_init) = {
            let mut n_borrow = nlp.borrow_mut();
            let x_l = vec_from_dyn(n_borrow.x_l());
            let x_u = vec_from_dyn(n_borrow.x_u());
            let d_l = if m_d > 0 {
                vec_from_dyn(n_borrow.d_l())
            } else {
                Vec::new()
            };
            let d_u = if m_d > 0 {
                vec_from_dyn(n_borrow.d_u())
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
