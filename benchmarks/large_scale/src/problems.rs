//! Five synthetic large-scale NLPs, each as a `TNLP` implementation.
//!
//! Ported from the original `problems.rs` (ripopt era). Each problem keeps
//! the same math, sparsity pattern, and Hessian layout as the source.

mod bratu;
mod chained_rosenbrock;
mod optimal_control;
mod poisson_control;
mod sparse_qp;

pub use bratu::BratuProblem;
pub use chained_rosenbrock::ChainedRosenbrock;
pub use optimal_control::OptimalControl;
pub use poisson_control::PoissonControl;
pub use sparse_qp::SparseQP;

use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::tnlp::Solution;

/// Final-solve state stashed by a `TNLP::finalize_solution` impl, so the
/// driver can read out the status/objective after `optimize_tnlp`.
#[derive(Debug, Default, Clone)]
pub struct FinalState {
    pub status: Option<SolverReturn>,
    pub obj: f64,
    pub x: Vec<f64>,
}

impl FinalState {
    pub fn new() -> Self {
        Self {
            status: None,
            obj: f64::NAN,
            x: Vec::new(),
        }
    }

    pub fn capture(&mut self, sol: Solution<'_>) {
        self.status = Some(sol.status);
        self.obj = sol.obj_value;
        self.x = sol.x.to_vec();
    }
}
