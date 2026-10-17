//! The seam between the search and the local solver.
//!
//! [`find_minima`](crate::find_minima) never sees a solver application: it
//! hands each start to a [`LocalSolve`] and reads back a [`LocalOutcome`].
//! [`IpoptLocalSolve`] is the implementation over a reused
//! [`IpoptApplication`], which is what the CLI uses; a frontend with its own
//! solve path (the Python bindings) implements the trait over that instead.

use crate::capture_tnlp::{CaptureTnlp, FinalSlot};
use pounce_algorithm::application::{ConvergedCallback, IpoptApplication};
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use std::cell::RefCell;
use std::rc::Rc;

/// Which Hessian a local solve should use. The search asks for
/// [`QuasiNewton`](Self::QuasiNewton) on the augmented penalty / tunnel
/// objectives, which decline an exact Hessian, and [`Exact`](Self::Exact) on
/// clean and polish solves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HessianMode {
    Exact,
    QuasiNewton,
}

/// The solver's own unscaled solve-quality numbers, read by the `kkt_tol`
/// veto (gh#983). Each is `NaN` when the solve path does not compute it,
/// which the veto treats as "no evidence" rather than as a rejection.
#[derive(Clone, Copy, Debug)]
pub struct SolveQuality {
    /// Unscaled dual infeasibility (stationarity residual) at the returned
    /// point: `SolveStatistics::final_unscaled_dual_inf`.
    pub unscaled_dual_inf: Number,
    /// Magnitude of the terms that residual is assembled from:
    /// `SolveStatistics::final_unscaled_dual_scale`.
    pub unscaled_dual_scale: Number,
    /// `max |∇f|` at the solve's starting point:
    /// `SolveStatistics::start_obj_grad_max`.
    pub start_obj_grad_max: Number,
}

impl Default for SolveQuality {
    fn default() -> Self {
        Self {
            unscaled_dual_inf: Number::NAN,
            unscaled_dual_scale: Number::NAN,
            start_obj_grad_max: Number::NAN,
        }
    }
}

/// One local solve's result.
#[derive(Clone, Debug)]
pub struct LocalOutcome {
    /// The solve converged (`Solve_Succeeded` or `Solved_To_Acceptable_Level`)
    /// and produced a point. The search applies its own acceptance tests on
    /// top of this; it is not a verdict that the point is a minimum.
    pub converged: bool,
    /// The returned primal, full length — a candidate only when `converged`;
    /// otherwise whatever point the solve ended at, or empty if none.
    pub x: Vec<Number>,
    /// Constraint duals in the user's `∇f + λ·∇g + z = 0` convention, length
    /// `m` (empty if the solve path cannot report them).
    pub lambda: Vec<Number>,
    pub quality: SolveQuality,
}

/// Solves one problem from the start it carries.
///
/// `problem` is already seeded with the start (`get_starting_point` returns
/// it) and, for the repulsion strategies, already augmented. Calls are
/// sequential and the search counts them against its own budget.
pub trait LocalSolve {
    fn solve(&mut self, problem: Rc<RefCell<dyn TNLP>>, mode: HessianMode) -> LocalOutcome;

    /// Whether the problem's `eval_h` is its real Hessian. A frontend that
    /// declares a Hessian structure only to give L-BFGS a workspace, and fills
    /// it with zeros, returns `false`: the search then treats the problem as
    /// having no Hessian — no saddle check, the fallback flooding amplitude,
    /// quasi-Newton penalty solves — as `pounce.find_minima` does without
    /// `hess=`. Default `true`.
    fn has_exact_hessian(&self) -> bool {
        true
    }
}

/// [`LocalSolve`] over one reused [`IpoptApplication`] (no rebuild per start).
///
/// Each solve sets `hessian_approximation` for its [`HessianMode`] — the IPM
/// rereads the option every solve — and reads the answer from the
/// `finalize_solution` payload, through a wrapper around the problem: the
/// reported `x` (after `honor_original_bounds`, when set) and the user-space
/// `lambda`, the same numbers a plain solve's `.sol` carries.
///
/// It used to read them from the application's on-converged hook instead,
/// which sees the algorithm's iterate: the un-projected `x`, so
/// `honor_original_bounds=yes` was ignored, and — when a later attempt lost
/// and an earlier answer was restored from the floor (the mu fallback,
/// pounce#870, on by default, or the gh#884 retry) — the *discarded*
/// attempt's point, while the status described the winner's. The floor
/// restore re-delivers the winner through `finalize_solution`, so the payload
/// is right in both cases.
///
/// Holding the application mutably for its lifetime also keeps the caller's
/// own on-converged callback, if any, out of the search's solves; it is put
/// back when this is dropped, and so are the options the solves wrote
/// (`hessian_approximation`, and `mu_strategy` below).
///
/// A quasi-Newton solve also pins `mu_strategy = monotone`, unless the caller
/// set `mu_strategy` themselves: the search, not the caller, chose L-BFGS for
/// that subproblem, and gh#746's rule (an unset `mu_strategy` under
/// limited-memory resolves to `adaptive`) would otherwise read a barrier
/// preference out of that choice — the inference `pounce-py`'s `prepare`
/// already refuses for the same reason. Exact solves get the option unset
/// again, so they see the registered default exactly as before. Measured on
/// six-hump camel tunneling over 20 seeds (`find_minima(hess=...)`, gh#966):
/// mean minima found 3.2 under the inferred `adaptive`, 4.5 under `monotone`
/// (the pure-Python search, whose tunnel solves were Hessian-less problems
/// and so ran `monotone`: 3.85).
pub struct IpoptLocalSolve<'a> {
    app: &'a mut IpoptApplication,
    previous_hook: Option<ConvergedCallback>,
    /// The caller left `mu_strategy` unset, so the quasi-Newton solves pin it.
    pin_mu: bool,
    /// The caller's own `hessian_approximation`, if they set one; every solve
    /// overwrites it, and `Drop` puts it back (or unsets it again).
    caller_hessian: Option<String>,
}

impl<'a> IpoptLocalSolve<'a> {
    pub fn new(app: &'a mut IpoptApplication) -> Self {
        let previous_hook = app.take_on_converged();
        let pin_mu = matches!(
            app.options().get_string_value("mu_strategy", ""),
            Ok((_, false))
        );
        let caller_hessian = match app.options().get_string_value("hessian_approximation", "") {
            Ok((v, true)) => Some(v),
            _ => None,
        };
        Self {
            app,
            previous_hook,
            pin_mu,
            caller_hessian,
        }
    }
}

impl LocalSolve for IpoptLocalSolve<'_> {
    fn solve(&mut self, problem: Rc<RefCell<dyn TNLP>>, mode: HessianMode) -> LocalOutcome {
        let line = match mode {
            HessianMode::Exact => "hessian_approximation exact\n",
            HessianMode::QuasiNewton => "hessian_approximation limited-memory\n",
        };
        let opts = self.app.options_mut();
        let _ = opts.read_from_str(line, true);
        if self.pin_mu {
            match mode {
                HessianMode::QuasiNewton => {
                    let _ = opts.set_string_value("mu_strategy", "monotone", true, true);
                }
                HessianMode::Exact => {
                    opts.unset_value("mu_strategy");
                }
            }
        }
        let slot: FinalSlot = Rc::new(RefCell::new(None));
        let wrapped: Rc<RefCell<dyn TNLP>> =
            Rc::new(RefCell::new(CaptureTnlp::new(problem, Rc::clone(&slot))));
        let status = self.app.optimize_tnlp(wrapped);
        let succeeded = matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        );
        let stats = self.app.statistics();
        let quality = SolveQuality {
            unscaled_dual_inf: stats.final_unscaled_dual_inf,
            unscaled_dual_scale: stats.final_unscaled_dual_scale,
            start_obj_grad_max: stats.start_obj_grad_max,
        };
        match slot.borrow_mut().take() {
            Some((x, lambda)) => LocalOutcome {
                converged: succeeded,
                x,
                lambda,
                quality,
            },
            None => LocalOutcome {
                converged: false,
                x: Vec::new(),
                lambda: Vec::new(),
                quality,
            },
        }
    }
}

impl Drop for IpoptLocalSolve<'_> {
    fn drop(&mut self) {
        // Leave the caller's options as they were: every solve wrote
        // `hessian_approximation`, and quasi-Newton ones `mu_strategy`.
        let opts = self.app.options_mut();
        if self.pin_mu {
            opts.unset_value("mu_strategy");
        }
        match self.caller_hessian.take() {
            Some(v) => {
                let _ = opts.set_string_value("hessian_approximation", &v, true, true);
            }
            None => {
                opts.unset_value("hessian_approximation");
            }
        }
        // Put back whatever the caller had registered.
        if let Some(cb) = self.previous_hook.take() {
            self.app.set_on_converged(cb);
        }
    }
}
