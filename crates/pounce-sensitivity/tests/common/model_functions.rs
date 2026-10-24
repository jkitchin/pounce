//! The model's own functions over a TNLP, for the step reports, which
//! measure a moved point with the model rather than with the engine.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::tnlp::{IndexStyle, SparsityRequest};
use pounce_sens_core::model::ModelFunctions;

use super::outside_engine::Bounds;

/// A TNLP's bounds, constraints, Jacobian and objective gradient.
pub struct TnlpModel {
    model: Rc<RefCell<dyn TNLP>>,
    bounds: Bounds,
    jac: Vec<(usize, usize)>,
}

impl TnlpModel {
    /// Over `model`, reading its declared bounds and Jacobian structure
    /// once.
    pub fn new(model: Rc<RefCell<dyn TNLP>>) -> Self {
        let (bounds, jac) = {
            let mut md = model.borrow_mut();
            let bounds = Bounds::declared(&mut *md);
            let info = md.get_nlp_info().expect("nlp info");
            let base = match info.index_style {
                IndexStyle::C => 0,
                IndexStyle::Fortran => 1,
            };
            let mut ir = vec![0 as Index; info.nnz_jac_g as usize];
            let mut jc = vec![0 as Index; info.nnz_jac_g as usize];
            assert!(md.eval_jac_g(
                None,
                false,
                SparsityRequest::Structure {
                    irow: &mut ir,
                    jcol: &mut jc,
                },
            ));
            let jac = ir
                .iter()
                .zip(&jc)
                .map(|(&r, &c)| ((r - base) as usize, (c - base) as usize))
                .collect();
            (bounds, jac)
        };
        Self { model, bounds, jac }
    }
}

impl ModelFunctions for TnlpModel {
    fn variable_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        (self.bounds.x_l.clone(), self.bounds.x_u.clone())
    }

    fn row_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        (self.bounds.g_l.clone(), self.bounds.g_u.clone())
    }

    fn constraints(&mut self, x: &[Number]) -> Result<Vec<Number>, String> {
        let mut g = vec![0.0; self.bounds.g_l.len()];
        if self.model.borrow_mut().eval_g(x, true, &mut g) {
            Ok(g)
        } else {
            Err("eval_g failed".into())
        }
    }

    fn jacobian(&mut self, x: &[Number]) -> Result<Vec<(usize, usize, Number)>, String> {
        let mut v = vec![0.0; self.jac.len()];
        if !self.model.borrow_mut().eval_jac_g(
            Some(x),
            true,
            SparsityRequest::Values { values: &mut v },
        ) {
            return Err("eval_jac_g failed".into());
        }
        Ok(self
            .jac
            .iter()
            .zip(v)
            .map(|(&(r, c), val)| (r, c, val))
            .collect())
    }

    fn objective_gradient(&mut self, x: &[Number]) -> Result<Vec<Number>, String> {
        let mut g = vec![0.0; x.len()];
        if self.model.borrow_mut().eval_grad_f(x, true, &mut g) {
            Ok(g)
        } else {
            Err("eval_grad_f failed".into())
        }
    }
}
