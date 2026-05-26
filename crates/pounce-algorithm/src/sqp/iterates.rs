//! SQP iterate state — `(x, λ_g, λ_x)` plus the working set
//! carried across QP subproblem solves for warm-starting (the
//! §5/§6 design-note contract).
//!
//! Distinct from [`crate::iterates_vector::IteratesVector`] (which
//! is IPM-shaped with slacks and barrier multipliers). The SQP
//! iterate is simpler: primal `x`, constraint multipliers
//! `λ_g`, packed bound multipliers `λ_x = z_l − z_u`, and the
//! `pounce_qp::WorkingSet` from the previous QP solve.

use pounce_common::Number;
use pounce_qp::WorkingSet;

#[derive(Debug, Clone)]
pub struct SqpIterates {
    pub x: Vec<Number>,
    pub lambda_g: Vec<Number>,
    pub lambda_x: Vec<Number>,
    pub working: Option<WorkingSet>,
}

impl SqpIterates {
    pub fn cold(n: usize, m: usize) -> Self {
        Self {
            x: vec![0.0; n],
            lambda_g: vec![0.0; m],
            lambda_x: vec![0.0; n],
            working: None,
        }
    }

    pub fn n(&self) -> usize {
        self.x.len()
    }

    pub fn m(&self) -> usize {
        self.lambda_g.len()
    }
}
