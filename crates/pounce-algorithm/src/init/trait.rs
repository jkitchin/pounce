//! `IterateInitializer` trait — port of `IpIterateInitializer.hpp`.

use crate::init::default::LeastSquareInitReport;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::kkt::aug_system_solver::AugSystemSolver;
use std::cell::RefCell;
use std::rc::Rc;

pub trait IterateInitializer {
    /// Populate `IpoptData::curr` with an initial iterate. Mirrors
    /// `IterateInitializer::SetInitialIterates`. The implementation
    /// can use `aug_solver` for least-square multiplier estimates;
    /// callers that don't need that may pass any solver — concrete
    /// initializers consult it only if their option settings require
    /// it.
    fn set_initial_iterates(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        aug_solver: &mut dyn AugSystemSolver,
    ) -> bool;

    /// Diagnostics from the safeguarded `least_square_init_primal`
    /// step (gh#605): initial/final nonlinear violation, accepted step
    /// norm, rejected-trial count, termination reason. `None` for
    /// initializers that do not run that step, or when it was not
    /// attempted on this solve.
    fn least_square_report(&self) -> Option<LeastSquareInitReport> {
        None
    }
}
