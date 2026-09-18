//! Ergonomic builder API over the [`TNLP`](crate::TNLP) trait.
//!
//! The raw `TNLP` interface is a faithful port of Ipopt's C++ `TNLP` (nine
//! methods, sparsity bookkeeping, an `Rc<RefCell<dyn TNLP>>` driver) — full
//! control, but heavy for a simple problem. This module offers the
//! argmin-style alternative requested in
//! [#168](https://github.com/jkitchin/pounce/issues/168): implement the small
//! [`Problem`] trait (only `objective` is required), then configure and solve
//! with the [`Nlp`] builder. Anything you don't implement is finite-
//! differenced (gradient / constraint Jacobian) or approximated (the Hessian
//! defaults to limited-memory L-BFGS), so a basic problem stays small while the
//! full `TNLP` trait remains available for everything this doesn't expose.
//!
//! ```
//! use pounce_rs::builder::{Problem, Nlp};
//!
//! // min (x0-1)^2 + (x1-2)^2  s.t.  x0 + x1 == 3,  0 <= xi <= 5
//! struct P;
//! impl Problem for P {
//!     fn objective(&self, x: &[f64]) -> f64 {
//!         (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
//!     }
//!     fn n_constraints(&self) -> usize { 1 }
//!     fn constraints(&self, x: &[f64], g: &mut [f64]) { g[0] = x[0] + x[1]; }
//! }
//!
//! let sol = Nlp::new(P)                       // variable count inferred below
//!     .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
//!     .constraint_bounds(&[3.0], &[3.0])      // equality: lower == upper
//!     .x0(&[2.0, 0.5])
//!     .option_num("tol", 1e-10)
//!     .solve();
//!
//! assert!(sol.success);
//! assert!((sol.x[0] - 1.0).abs() < 1e-5 && (sol.x[1] - 2.0).abs() < 1e-5);
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use crate::{
    ApplicationReturnStatus, BoundsInfo, IndexStyle, IpoptApplication, IpoptCq, IpoptData, NlpInfo,
    Solution as TnlpSolution, SolveStatistics, SparsityRequest, StartingPoint, TNLP,
};
use pounce_nlp::expression_provider::{ExpressionProvider, FbbtOp, FbbtTape};

const FD: f64 = 1.4901161193847656e-8; // sqrt(f64::EPSILON)
const INF: f64 = 2.0e19; // Ipopt's "infinity" bound sentinel

/// A nonlinear program. Implement `objective`; override the rest as needed.
///
/// `gradient` / `jacobian` return `false` (their default) to request a
/// finite-difference approximation. The Hessian is never required — the
/// builder uses a limited-memory (L-BFGS) approximation by default.
pub trait Problem {
    /// Objective `f(x)` to minimize.
    fn objective(&self, x: &[f64]) -> f64;

    /// Number of constraints `m` (default `0`, i.e. bound-constrained only).
    fn n_constraints(&self) -> usize {
        0
    }

    /// Constraint values `g(x)` into `out` (length `n_constraints`).
    fn constraints(&self, _x: &[f64], _out: &mut [f64]) {}

    /// Optional FBBT expression for constraint `index`.
    ///
    /// The tape must restate [`Self::constraints`] for that row, and every
    /// slot in it must influence the root's value — a slot the root does not
    /// depend on says nothing about the constraint, and gh #877 is what
    /// happens when bounds are propagated back out of one anyway. The reverse
    /// pass now skips non-influencing slots rather than trusting the caller,
    /// but keeping them out of the tape is cheaper.
    ///
    /// `try_solve` samples the two forms at the starting point and the box
    /// midpoint. That is a smoke test twice over: two points are not a proof
    /// of equivalence, and even at those two the comparison allows a relative
    /// mismatch of `sqrt(f64::EPSILON)` (~1.5e-8). Generate the callback and
    /// the tape from one source when you can.
    ///
    /// It is used only when both `presolve=yes` and `presolve_fbbt=yes`.
    fn constraint_expression(&self, _index: usize) -> Option<FbbtTape> {
        None
    }

    /// Objective gradient `∇f(x)` into `grad`; return `false` for finite
    /// differences.
    fn gradient(&self, _x: &[f64], _grad: &mut [f64]) -> bool {
        false
    }

    /// Dense constraint Jacobian (row-major, `n_constraints × n`) into `jac`;
    /// return `false` for finite differences.
    fn jacobian(&self, _x: &[f64], _jac: &mut [f64]) -> bool {
        false
    }
}

/// The outcome of [`Nlp::solve`].
///
/// The vector fields (`x`, `multipliers`, `g`, `z_l`, `z_u`) are filled
/// by the solver's `finalize_solution` callback. They stay empty
/// when the solve aborts before finalization, so check
/// `success`/`status` before indexing.
/// What the second-opinion ladder did, when it ran.
///
/// Present on [`Solution::second_opinion`] only when a failing solve
/// actually opened the ladder; `None` is the overwhelmingly common path
/// and means nothing extra was spent.
///
/// Without this a Rust embedder saw only a failing solve that took up to
/// four times as long as it used to, with nothing to attribute the time
/// to -- the ladder is on by default, so that is a real reporting gap
/// rather than a nicety. Mirrors Python's `info["second_opinion"]`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SecondOpinion {
    /// Rung labels actually run, in order.
    pub tried: Vec<&'static str>,
    /// The rung whose re-solve was promoted, if any. `None` here with a
    /// non-empty `tried` means every rung was tried and the original
    /// verdict stood -- which is evidence *for* the verdict.
    pub promoted_by: Option<&'static str>,
    /// The narration the CLI would have printed to stderr. Collected
    /// rather than printed: a library caller has not asked for a running
    /// commentary on its own stderr.
    pub log: Vec<String>,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Solution {
    /// Solver status; `success` is the convenient boolean.
    pub status: ApplicationReturnStatus,
    /// `true` for `SolveSucceeded` / `SolvedToAcceptableLevel`.
    pub success: bool,
    /// Optimal variables (length `n`).
    pub x: Vec<f64>,
    /// Objective at the solution.
    pub objective: f64,
    /// Constraint multipliers `λ` (length `n_constraints`).
    pub multipliers: Vec<f64>,
    /// Constraint values `g(x)` at the solution (length `n_constraints`).
    pub g: Vec<f64>,
    /// Lower-bound multipliers `z_L` (length `n`).
    pub z_l: Vec<f64>,
    /// Upper-bound multipliers `z_U` (length `n`).
    pub z_u: Vec<f64>,
    /// Per-solve statistics: wall time (`total_wallclock_time_secs`),
    /// `iteration_count`, evaluation counts, final scaled and unscaled
    /// infeasibilities, final barrier `final_mu`, restoration counters.
    /// `stats.iterations` holds the per-iteration trajectory and is
    ///  non-empty only when [`Nlp::capture_iterations`] was requested.
    pub stats: SolveStatistics,
    /// FBBT diagnostics when presolve FBBT ran.
    pub fbbt_report: Option<pounce_presolve::fbbt::FbbtReport>,
    /// What the second-opinion ladder did, or `None` if it never ran.
    pub second_opinion: Option<SecondOpinion>,
}

/// Why a solve could not be started.
///
/// Returned by [`Nlp::try_solve`]; [`Nlp::solve`] panics on the same
/// conditions. Every variant here is a configuration error detected
/// *before* the solver runs — a solve that runs and fails to converge is
/// not an error, it comes back as a [`Solution`] with `success == false`
/// and the reason in `status`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum NlpError {
    /// Neither [`var_bounds`](Nlp::var_bounds) nor [`x0`](Nlp::x0) was
    /// called, so the number of variables is unknown.
    UnknownVariableCount,
    /// An option passed to [`option_num`](Nlp::option_num) /
    /// [`option_int`](Nlp::option_int) / [`option_str`](Nlp::option_str)
    /// was rejected by the options registry: unknown name, wrong value
    /// type for that name, or a value outside the registered range or
    /// set of choices.
    InvalidOption {
        /// The option name as passed by the caller.
        tag: String,
        /// The rejected value, rendered as a string.
        value: String,
        /// The registry's explanation.
        reason: String,
    },
    /// `IpoptApplication::initialize` failed.
    Initialize(String),
    /// Presolve setup failed.
    Presolve(String),
    /// An FBBT tape was invalid for this problem.
    InvalidFbbtTape { constraint: usize, reason: String },
}

impl std::fmt::Display for NlpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownVariableCount => write!(
                f,
                "number of variables unknown — call .var_bounds(..) or .x0(..) \
                 to set it",
            ),
            Self::InvalidOption { tag, value, reason } => {
                write!(f, "option {tag}={value} rejected: {reason}")
            }
            Self::Initialize(msg) => write!(f, "IpoptApplication::initialize failed: {msg}"),
            Self::Presolve(msg) => write!(f, "presolve setup failed: {msg}"),
            Self::InvalidFbbtTape { constraint, reason } => {
                write!(f, "invalid FBBT tape for constraint {constraint}: {reason}")
            }
        }
    }
}

impl std::error::Error for NlpError {}

/// Builder: `Nlp::new(problem)` then `.var_bounds(..)` / `.x0(..)` (which fix
/// the number of variables) and `.solve()` / `.try_solve()`.
pub struct Nlp<P: Problem> {
    problem: P,
    n: Option<usize>, // inferred from var_bounds / x0 (must agree)
    x_l: Option<Vec<f64>>,
    x_u: Option<Vec<f64>>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    x0: Option<Vec<f64>>,
    num: Vec<(String, f64)>,
    int: Vec<(String, i32)>,
    string: Vec<(String, String)>,
    capture_iterations: bool,
}

impl<P: Problem + 'static> Nlp<P> {
    /// A new builder for `problem`. The number of variables is inferred from
    /// the first of [`var_bounds`](Self::var_bounds) / [`x0`](Self::x0) you set
    /// (they must agree); the number of constraints comes from
    /// `Problem::n_constraints`. Variable bounds default to `±∞`, constraint
    /// bounds to `0`, and `x0` to the origin.
    pub fn new(problem: P) -> Self {
        let m = problem.n_constraints();
        Nlp {
            problem,
            n: None,
            x_l: None,
            x_u: None,
            g_l: vec![0.0; m],
            g_u: vec![0.0; m],
            x0: None,
            num: Vec::new(),
            int: Vec::new(),
            string: Vec::new(),
            capture_iterations: false,
        }
    }

    // Record (and cross-check) the variable count implied by a length-`len`
    // argument.
    fn set_n(&mut self, len: usize, what: &str) {
        match self.n {
            Some(n) if n != len => panic!(
                "pounce_rs::Nlp: {what} has length {len}, but the problem was \
                 already sized to {n} variables",
            ),
            _ => self.n = Some(len),
        }
    }

    /// Variable bounds `x_l ≤ x ≤ x_u` (use `±2e19` for ∞). Fixes the number of
    /// variables.
    pub fn var_bounds(mut self, lo: &[f64], hi: &[f64]) -> Self {
        assert_eq!(lo.len(), hi.len(), "var_bounds: lo and hi differ in length");
        self.set_n(lo.len(), "var_bounds");
        self.x_l = Some(lo.to_vec());
        self.x_u = Some(hi.to_vec());
        self
    }

    /// Constraint bounds `g_l ≤ g(x) ≤ g_u` (`g_l == g_u` is an equality).
    pub fn constraint_bounds(mut self, lo: &[f64], hi: &[f64]) -> Self {
        self.g_l = lo.to_vec();
        self.g_u = hi.to_vec();
        self
    }

    /// Initial guess. Fixes the number of variables.
    pub fn x0(mut self, x0: &[f64]) -> Self {
        self.set_n(x0.len(), "x0");
        self.x0 = Some(x0.to_vec());
        self
    }

    /// A numeric solver option (e.g. `("tol", 1e-8)`).
    pub fn option_num(mut self, tag: &str, value: f64) -> Self {
        self.num.push((tag.to_string(), value));
        self
    }

    /// An integer solver option (e.g. `("max_iter", 500)`).
    pub fn option_int(mut self, tag: &str, value: i32) -> Self {
        self.int.push((tag.to_string(), value));
        self
    }

    /// A string solver option (e.g. `("mu_strategy", "adaptive")`).
    pub fn option_str(mut self, tag: &str, value: &str) -> Self {
        self.string.push((tag.to_string(), value.to_string()));
        self
    }

    /// Record the per-iteration trajectory: one
    /// [`IterRecord`](crate::IterRecord) per Newton iteration into
    /// [`Solution::stats`]`.iterations` (empty without this call).
    ///
    /// Interior-point solves only, since the per-iteration event
    /// is emitted by the IPM engine, so on the active-set SQP engine
    /// (`solver_selection=qp-active-set`/`algorithm=active-set-sqp`)
    /// `stats.iterations` stays empty even though
    /// `stats.iteration_count` still reports the iterations run.
    pub fn capture_iterations(mut self) -> Self {
        self.capture_iterations = true;
        self
    }

    /// Build the `TNLP` adapter and run the interior-point solver.
    ///
    /// # Panics
    /// If the solve could not be started: the number of variables was never
    /// fixed (no `var_bounds` or `x0`), or an option was rejected by the
    /// options registry. Use [`try_solve`](Self::try_solve) to handle those
    /// as a [`Result`] instead.
    pub fn solve(self) -> Solution {
        match self.try_solve() {
            Ok(sol) => sol,
            Err(e) => panic!("pounce_rs::Nlp: {e}"),
        }
    }

    /// Build the `TNLP` adapter and run the interior-point solver, reporting
    /// setup failures as an [`NlpError`] instead of panicking.
    ///
    /// Unlike [`solve`](Self::solve) this surfaces a rejected option — a
    /// misspelled name, a value out of the registered range, or a value
    /// that is not one of the registered choices — rather than letting it
    /// pass silently and solving with the default still in effect.
    ///
    /// A solve that *runs* and does not converge is not an error here: it
    /// returns `Ok` with `success == false` and the reason in `status`.
    ///
    /// ```
    /// use pounce_rs::builder::{Nlp, NlpError, Problem};
    ///
    /// struct P;
    /// impl Problem for P {
    ///     fn objective(&self, x: &[f64]) -> f64 { (x[0] - 1.0).powi(2) }
    /// }
    ///
    /// // "mu_stratgey" is a typo for "mu_strategy".
    /// let err = Nlp::new(P).x0(&[0.0]).option_str("mu_stratgey", "adaptive").try_solve();
    /// assert!(matches!(err, Err(NlpError::InvalidOption { .. })));
    /// ```
    ///
    /// # Errors
    /// [`NlpError::UnknownVariableCount`] if neither `var_bounds` nor `x0`
    /// was called, [`NlpError::InvalidOption`] if an option was rejected,
    /// [`NlpError::Initialize`] if the application failed to initialize,
    /// [`NlpError::Presolve`] if presolve setup failed, and
    /// [`NlpError::InvalidFbbtTape`] if an enabled FBBT tape is malformed or
    /// disagrees with sampled values from [`Problem::constraints`].
    pub fn try_solve(self) -> Result<Solution, NlpError> {
        let n = self.n.ok_or(NlpError::UnknownVariableCount)?;
        let m = self.problem.n_constraints();

        let mut app = IpoptApplication::new();
        app.initialize()
            .map_err(|e| NlpError::Initialize(e.message))?;
        // No analytic Hessian is required from `Problem`, so default to L-BFGS.
        let _ = app.options_mut().set_string_value(
            "hessian_approximation",
            "limited-memory",
            true,
            true,
        );
        // The active-set SQP engine (selected by `solver_selection=qp-active-set`
        // or `algorithm=active-set-sqp`) reads `sqp_hessian`, whose default
        // `exact` needs the analytic Hessian this builder never supplies. Default
        // it to limited-memory BFGS so the SQP route works Hessian-free too; a
        // user `.option_str("sqp_hessian", ...)` below overrides it.
        let _ = app
            .options_mut()
            .set_string_value("sqp_hessian", "lbfgs", true, true);
        // User options, in contrast to the two defaults above, are reported
        // rather than dropped: an unknown name or an out-of-range value that
        // is silently discarded leaves the default in effect and the solve
        // looks like it honoured the request (gh#649). `Ok(false)` is not a
        // failure — it means an earlier no-clobber setting won — so only the
        // `Err` arm is escalated.
        for (k, v) in &self.string {
            app.options_mut()
                .set_string_value(k, v, true, true)
                .map_err(|e| NlpError::InvalidOption {
                    tag: k.clone(),
                    value: v.clone(),
                    reason: e.message,
                })?;
        }
        for (k, v) in &self.num {
            app.options_mut()
                .set_numeric_value(k, *v, true, true)
                .map_err(|e| NlpError::InvalidOption {
                    tag: k.clone(),
                    value: v.to_string(),
                    reason: e.message,
                })?;
        }
        for (k, v) in &self.int {
            app.options_mut()
                .set_integer_value(k, *v, true, true)
                .map_err(|e| NlpError::InvalidOption {
                    tag: k.clone(),
                    value: v.to_string(),
                    reason: e.message,
                })?;
        }

        let presolve_opts = pounce_presolve::PresolveOptions::from_options_list(app.options())
            .map_err(|e| NlpError::Presolve(e.to_string()))?;
        let x_l = self.x_l.unwrap_or_else(|| vec![-INF; n]);
        let x_u = self.x_u.unwrap_or_else(|| vec![INF; n]);
        let x0 = self.x0.unwrap_or_else(|| vec![0.0; n]);
        let constraint_expressions = if presolve_opts.enabled && presolve_opts.fbbt {
            let mut tapes = Vec::with_capacity(m);
            for i in 0..m {
                let tape = self.problem.constraint_expression(i);
                if let Some(ref tape) = tape {
                    validate_fbbt_tape(i, tape, n)?;
                }
                tapes.push(tape);
            }
            validate_fbbt_tape_values(&self.problem, &tapes, &x0, &x_l, &x_u, m)?;
            tapes
        } else {
            Vec::new()
        };

        let adapter = Rc::new(RefCell::new(Adapter {
            problem: self.problem,
            n,
            m,
            x_l,
            x_u,
            g_l: self.g_l,
            g_u: self.g_u,
            x0,
            constraint_expressions,
            sol_x: Vec::new(),
            sol_obj: 0.0,
            sol_lambda: Vec::new(),
            sol_g: Vec::new(),
            sol_z_l: Vec::new(),
            sol_z_u: Vec::new(),
        }));

        let mut fbbt_handle = None;
        let tnlp: Rc<RefCell<dyn TNLP>> = if presolve_opts.enabled && presolve_opts.fbbt {
            let provider: Rc<RefCell<dyn ExpressionProvider>> = Rc::clone(&adapter) as _;
            let presolve = Rc::new(RefCell::new(
                pounce_presolve::PresolveTnlp::with_expression_provider(
                    Rc::clone(&adapter) as Rc<RefCell<dyn TNLP>>,
                    provider,
                    presolve_opts,
                ),
            ));
            fbbt_handle = Some(Rc::clone(&presolve));
            app.set_presolve_already_applied(true);
            presolve
        } else {
            Rc::clone(&adapter) as Rc<RefCell<dyn TNLP>>
        };

        let scope = self.capture_iterations.then(|| {
            app.enable_iter_history();
            crate::collector_scope()
        });
        // The derivative test must run against the *unpresolved* adapter, since
        // presolve may have eliminated variables the user's `gradient` /
        // `jacobian` still index. When no presolve wrapper was installed,
        // `tnlp` **is** `adapter` and the override is the same `Rc` — passing
        // it is a no-op (`application.rs` does
        // `derivative_test_tnlp.as_ref().unwrap_or(&tnlp)`), but it means the
        // builder never exercises the `None` arm. gh #877 F-7: pass the
        // override only when it actually differs, so the default path here is
        // the same call shape every other frontend makes.
        // The run-ending `EXIT:` / `POUNCE <version>:` verdict belongs to the
        // whole run, not to each attempt. Deferred from here through the
        // second-opinion ladder below, which releases it and prints it once
        // with the status that actually ships.
        app.defer_end_verdict();
        let status = match &fbbt_handle {
            Some(_) => {
                let derivative_test_tnlp = Rc::clone(&adapter) as Rc<RefCell<dyn TNLP>>;
                app.optimize_tnlp_with_derivative_test_tnlp(
                    Rc::clone(&tnlp),
                    Some(derivative_test_tnlp),
                )
            }
            None => app.optimize_tnlp(Rc::clone(&tnlp)),
        };
        let stats = app.statistics();
        // Second-opinion ladder, on by default here as in the CLI and the
        // Python / C frontends: an `Infeasible_Problem_Detected` or an
        // `Invalid_Number_Detected` is re-solved along up to three
        // deliberately different trajectories, and a re-solve is promoted only
        // if it converges. A converged solve pays nothing — the ladder reads
        // the status and returns without touching the application. The three
        // `*_retry` options turn individual rungs off; `.string("…_retry",
        // "no")` on this builder disables one.
        //
        // Narration is collected rather than printed: a library caller has
        // not asked for a running commentary on its own stderr. It comes back
        // on `Solution::second_opinion`, alongside which rungs ran and which
        // one was promoted -- otherwise a failing solve that quietly costs up
        // to four solves is unattributable from the outside.
        let mut ladder_log: Vec<String> = Vec::new();
        let ladder = pounce_restoration::second_opinion_driver::run_second_opinion_ladder(
            &mut app,
            tnlp,
            status,
            stats,
            &mut |line| ladder_log.push(line.to_string()),
        );
        let second_opinion = ladder.ran().then(|| SecondOpinion {
            tried: ladder.tried.clone(),
            promoted_by: ladder.promoted_by,
            log: ladder_log,
        });
        let status = ladder.status;
        // `ladder.statistics` is the shipped solve's — the original's when
        // nothing promotes, the promoted rung's when one does — so the
        // captured trajectory below belongs to the answer being returned. The
        // collector scope therefore has to stay open *across* the ladder: drop
        // it first and a promoted rung's re-solve captures nothing, leaving
        // `stats.iterations` empty under a `stats.iteration_count` that is not.
        let stats = ladder.statistics;
        drop(scope);
        let a = adapter.borrow();
        let fbbt_report = fbbt_handle.as_ref().and_then(|p| p.borrow().fbbt_report());
        Ok(Solution {
            status,
            success: matches!(
                status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
            ),
            x: a.sol_x.clone(),
            objective: a.sol_obj,
            multipliers: a.sol_lambda.clone(),
            g: a.sol_g.clone(),
            z_l: a.sol_z_l.clone(),
            z_u: a.sol_z_u.clone(),
            fbbt_report,
            stats,
            second_opinion,
        })
    }
}

/// Internal `TNLP` adapter: owns the user [`Problem`] and config, fills in
/// finite-difference gradient / Jacobian and a dense Jacobian sparsity.
struct Adapter<P: Problem> {
    problem: P,
    n: usize,
    m: usize,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    x0: Vec<f64>,
    constraint_expressions: Vec<Option<FbbtTape>>,
    sol_x: Vec<f64>,
    sol_obj: f64,
    sol_lambda: Vec<f64>,
    sol_g: Vec<f64>,
    sol_z_l: Vec<f64>,
    sol_z_u: Vec<f64>,
}

fn validate_fbbt_tape(
    constraint: usize,
    tape: &FbbtTape,
    n_variables: usize,
) -> Result<(), NlpError> {
    if let Some(slot) = tape.first_invalid_slot() {
        return Err(NlpError::InvalidFbbtTape {
            constraint,
            reason: format!("operand reference at slot {slot} is not backward"),
        });
    }
    for (slot, op) in tape.ops.iter().enumerate() {
        match *op {
            FbbtOp::Var(variable) if variable >= n_variables => {
                return Err(NlpError::InvalidFbbtTape {
                    constraint,
                    reason: format!("variable index {variable} is outside 0..{n_variables}"),
                });
            }
            FbbtOp::Const(value) if !value.is_finite() => {
                return Err(NlpError::InvalidFbbtTape {
                    constraint,
                    reason: format!("non-finite constant at slot {slot}: {value}"),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_fbbt_tape_values<P: Problem>(
    problem: &P,
    tapes: &[Option<FbbtTape>],
    x0: &[f64],
    x_l: &[f64],
    x_u: &[f64],
    n_constraints: usize,
) -> Result<(), NlpError> {
    let midpoint: Vec<_> = x_l
        .iter()
        .zip(x_u)
        .zip(x0)
        .map(|((&lo, &hi), &start)| {
            if lo > -INF && hi < INF {
                lo + 0.5 * (hi - lo)
            } else {
                start
            }
        })
        .collect();

    for (sample_name, point) in [("starting point", x0), ("box midpoint", &midpoint)] {
        let mut values = vec![0.0; n_constraints];
        problem.constraints(point, &mut values);
        for (constraint, tape) in tapes.iter().enumerate() {
            let Some(tape) = tape.as_ref().filter(|tape| !tape.is_empty()) else {
                continue;
            };
            let actual = values[constraint];
            if !actual.is_finite() {
                continue;
            }
            let slots = pounce_presolve::fbbt::forward_pass(tape, point, point).map_err(|e| {
                NlpError::InvalidFbbtTape {
                    constraint,
                    reason: format!("evaluation failed at {sample_name}: {e:?}"),
                }
            })?;
            let range = pounce_presolve::fbbt::forward_result(&slots);
            let scale = [actual, range.lo, range.hi]
                .into_iter()
                .filter(|value| value.is_finite())
                .fold(1.0_f64, |scale, value| scale.max(value.abs()));
            let tolerance = FD * scale;
            let matches = range.contains(actual)
                || (!range.is_empty()
                    && actual >= range.lo - tolerance
                    && actual <= range.hi + tolerance);
            if !matches {
                return Err(NlpError::InvalidFbbtTape {
                    constraint,
                    reason: format!(
                        "constraints() returned {actual:.16e} but the tape returned [{:.16e}, {:.16e}] at the {sample_name}; the tape must restate this constraint (checked at two points, to a relative tolerance of {FD:.3e} — agreement here is necessary, not sufficient)",
                        range.lo, range.hi
                    ),
                });
            }
        }
    }
    Ok(())
}

impl<P: Problem> ExpressionProvider for Adapter<P> {
    fn constraint_expression(&self, index: usize) -> Option<FbbtTape> {
        self.constraint_expressions
            .get(index)
            .and_then(Clone::clone)
    }
}

impl<P: Problem> TNLP for Adapter<P> {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as i32,
            m: self.m as i32,
            nnz_jac_g: (self.m * self.n) as i32, // dense Jacobian
            nnz_h_lag: 0,                        // L-BFGS: no analytic Hessian
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.x_l);
        b.x_u.copy_from_slice(&self.x_u);
        b.g_l.copy_from_slice(&self.g_l);
        b.g_u.copy_from_slice(&self.g_u);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }

    fn eval_f(&mut self, x: &[f64], _new_x: bool) -> Option<f64> {
        Some(self.problem.objective(x))
    }

    fn eval_grad_f(&mut self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        if self.problem.gradient(x, grad) {
            return true;
        }
        // forward-difference fallback
        let f0 = self.problem.objective(x);
        let mut xp = x.to_vec();
        for j in 0..self.n {
            let h = FD * x[j].abs().max(1.0);
            xp[j] = x[j] + h;
            grad[j] = (self.problem.objective(&xp) - f0) / h;
            xp[j] = x[j];
        }
        true
    }

    fn eval_g(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        self.problem.constraints(x, g);
        true
    }

    fn eval_jac_g(&mut self, x: Option<&[f64]>, _new_x: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for i in 0..self.m {
                    for j in 0..self.n {
                        irow[k] = i as i32;
                        jcol[k] = j as i32;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                #[allow(clippy::expect_used)]
                let x = x.expect("eval_jac_g(Values) without x");
                if self.problem.jacobian(x, values) {
                    return true;
                }
                // forward-difference fallback (dense)
                let mut g0 = vec![0.0; self.m];
                self.problem.constraints(x, &mut g0);
                let mut xp = x.to_vec();
                let mut gp = vec![0.0; self.m];
                for j in 0..self.n {
                    let h = FD * x[j].abs().max(1.0);
                    xp[j] = x[j] + h;
                    self.problem.constraints(&xp, &mut gp);
                    for i in 0..self.m {
                        values[i * self.n + j] = (gp[i] - g0[i]) / h;
                    }
                    xp[j] = x[j];
                }
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[f64]>,
        _new_x: bool,
        _obj_factor: f64,
        _lambda: Option<&[f64]>,
        _new_lambda: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        false // never called: the builder uses limited-memory (L-BFGS)
    }

    fn finalize_solution(&mut self, sol: TnlpSolution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.sol_x = sol.x.to_vec();
        self.sol_obj = sol.obj_value;
        self.sol_lambda = sol.lambda.to_vec();
        self.sol_g = sol.g.to_vec();
        self.sol_z_l = sol.z_l.to_vec();
        self.sol_z_u = sol.z_u.to_vec();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Quad; // min (x0-1)^2 + (x1-2)^2  s.t. x0 + x1 == 3
    impl Problem for Quad {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
        }
        fn n_constraints(&self) -> usize {
            1
        }
        fn constraints(&self, x: &[f64], g: &mut [f64]) {
            g[0] = x[0] + x[1];
        }
    }

    struct CircleConstraint;
    impl Problem for CircleConstraint {
        fn objective(&self, x: &[f64]) -> f64 {
            -x[0]
        }
        fn n_constraints(&self) -> usize {
            1
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) {
            out[0] = x[0] * x[0];
        }
        fn constraint_expression(&self, _index: usize) -> Option<FbbtTape> {
            Some(FbbtTape {
                ops: vec![FbbtOp::Var(0), FbbtOp::PowInt(0, 2)],
            })
        }
    }

    struct InvalidExpression(FbbtTape);
    impl Problem for InvalidExpression {
        fn objective(&self, _x: &[f64]) -> f64 {
            0.0
        }
        fn n_constraints(&self) -> usize {
            1
        }
        fn constraints(&self, _x: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
        fn constraint_expression(&self, _index: usize) -> Option<FbbtTape> {
            Some(self.0.clone())
        }
    }

    struct MismatchedExpression;
    impl Problem for MismatchedExpression {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] - 3.0).powi(2)
        }
        fn n_constraints(&self) -> usize {
            1
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) {
            out[0] = x[0];
        }
        fn constraint_expression(&self, _index: usize) -> Option<FbbtTape> {
            Some(FbbtTape {
                ops: vec![FbbtOp::Const(10.0), FbbtOp::Var(0), FbbtOp::Mul(0, 1)],
            })
        }
    }

    struct DerivativeTestCircle {
        gradient_points: Rc<RefCell<Vec<f64>>>,
    }

    struct PresolveProbe {
        jacobian_calls: Rc<RefCell<usize>>,
    }

    impl Problem for PresolveProbe {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] - 1.0).powi(2)
        }
        fn n_constraints(&self) -> usize {
            1
        }
        fn constraints(&self, x: &[f64], out: &mut [f64]) {
            out[0] = x[0];
        }
        fn jacobian(&self, _x: &[f64], jac: &mut [f64]) -> bool {
            *self.jacobian_calls.borrow_mut() += 1;
            jac[0] = 1.0;
            true
        }
    }

    fn run_presolve_probe(presolve: bool) -> (Solution, usize) {
        let jacobian_calls = Rc::new(RefCell::new(0));
        let mut nlp = Nlp::new(PresolveProbe {
            jacobian_calls: Rc::clone(&jacobian_calls),
        })
        .var_bounds(&[0.0], &[2.0])
        .constraint_bounds(&[0.5], &[INF])
        .x0(&[1.0])
        .option_int("max_iter", 0)
        .option_int("print_level", 0);
        if presolve {
            nlp = nlp.option_str("presolve", "yes");
        }
        let solution = nlp.solve();
        let calls = *jacobian_calls.borrow();
        (solution, calls)
    }

    impl Problem for DerivativeTestCircle {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] - 0.25).powi(2)
        }

        fn n_constraints(&self) -> usize {
            1
        }

        fn constraints(&self, x: &[f64], out: &mut [f64]) {
            out[0] = x[0] * x[0];
        }

        fn constraint_expression(&self, _index: usize) -> Option<FbbtTape> {
            Some(FbbtTape {
                ops: vec![FbbtOp::Var(0), FbbtOp::PowInt(0, 2)],
            })
        }

        fn gradient(&self, x: &[f64], grad: &mut [f64]) -> bool {
            self.gradient_points.borrow_mut().push(x[0]);
            grad[0] = 2.0 * (x[0] - 0.25);
            true
        }
    }

    #[test]
    fn fbbt_tightens_builder_bounds_and_reports() {
        let sol = Nlp::new(CircleConstraint)
            .var_bounds(&[-10.0], &[10.0])
            .constraint_bounds(&[-2.0e19], &[1.0])
            .option_str("presolve", "yes")
            .option_str("presolve_fbbt", "yes")
            .solve();
        assert!(sol.success, "status = {:?}", sol.status);
        let report = sol.fbbt_report.expect("FBBT report");
        assert!(report.bound_updates > 0);
        assert!(report.total_tightening > 0.0);
        assert!((sol.x[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn invalid_fbbt_tapes_are_rejected_before_solving() {
        let malformed = Nlp::new(InvalidExpression(FbbtTape {
            ops: vec![FbbtOp::Add(0, 0)],
        }))
        .x0(&[0.0])
        .option_str("presolve", "yes")
        .option_str("presolve_fbbt", "yes")
        .try_solve();
        assert!(matches!(
            malformed,
            Err(NlpError::InvalidFbbtTape { constraint: 0, .. })
        ));

        let out_of_range = Nlp::new(InvalidExpression(FbbtTape {
            ops: vec![FbbtOp::Var(1)],
        }))
        .x0(&[0.0])
        .option_str("presolve", "yes")
        .option_str("presolve_fbbt", "yes")
        .try_solve();
        assert!(matches!(
            out_of_range,
            Err(NlpError::InvalidFbbtTape { constraint: 0, .. })
        ));

        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let non_finite = Nlp::new(InvalidExpression(FbbtTape {
                ops: vec![FbbtOp::Const(value)],
            }))
            .x0(&[0.0])
            .option_str("presolve", "yes")
            .option_str("presolve_fbbt", "yes")
            .try_solve();
            match non_finite {
                Err(NlpError::InvalidFbbtTape { constraint, reason }) => {
                    assert_eq!(constraint, 0);
                    assert!(reason.contains("non-finite constant at slot 0"), "{reason}");
                }
                _ => panic!("non-finite FBBT constant was accepted"),
            }
        }
    }

    #[test]
    fn fbbt_tape_mismatch_is_rejected_before_solving() {
        let result = Nlp::new(MismatchedExpression)
            .var_bounds(&[0.0], &[10.0])
            .constraint_bounds(&[-INF], &[5.0])
            .x0(&[0.0])
            .option_str("presolve", "yes")
            .option_str("presolve_fbbt", "yes")
            .try_solve();

        match result {
            Err(NlpError::InvalidFbbtTape { constraint, reason }) => {
                assert_eq!(constraint, 0);
                assert!(reason.contains("box midpoint"), "{reason}");
                assert!(reason.contains("5.0000000000000000e0"), "{reason}");
            }
            _ => panic!("mismatched FBBT tape was accepted"),
        }
    }

    #[test]
    fn fbbt_report_is_absent_when_not_enabled() {
        let sol = Nlp::new(CircleConstraint)
            .var_bounds(&[-10.0], &[10.0])
            .constraint_bounds(&[-2.0e19], &[1.0])
            .solve();
        assert!(sol.success, "status = {:?}", sol.status);
        assert!(sol.fbbt_report.is_none());
    }

    #[test]
    fn application_presolve_runs_without_builder_fbbt() {
        let (without, calls_without) = run_presolve_probe(false);
        let (with, calls_with) = run_presolve_probe(true);

        assert!(without.fbbt_report.is_none());
        assert!(with.fbbt_report.is_none());
        assert!(
            calls_with > calls_without,
            "{calls_with} <= {calls_without}"
        );
    }

    #[test]
    fn fbbt_with_no_expression_tapes_is_a_noop() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .option_str("presolve", "yes")
            .option_str("presolve_fbbt", "yes")
            .solve();

        assert!(sol.success, "status = {:?}", sol.status);
        let report = sol.fbbt_report.expect("FBBT report");
        assert_eq!(report.bound_updates, 0);
        assert!(report.infeasibility_witness.is_none());
    }

    #[test]
    fn fbbt_switch_without_presolve_is_a_noop() {
        let sol = Nlp::new(CircleConstraint)
            .var_bounds(&[-10.0], &[10.0])
            .constraint_bounds(&[-INF], &[1.0])
            .option_str("presolve_fbbt", "yes")
            .solve();

        assert!(sol.success, "status = {:?}", sol.status);
        assert!(sol.fbbt_report.is_none());
    }

    #[test]
    fn fbbt_keeps_derivative_test_on_original_bounds() {
        let gradient_points = Rc::new(RefCell::new(Vec::new()));
        let sol = Nlp::new(DerivativeTestCircle {
            gradient_points: Rc::clone(&gradient_points),
        })
        .var_bounds(&[-10.0], &[10.0])
        .constraint_bounds(&[-2.0e19], &[1.0])
        .x0(&[5.0])
        .option_str("presolve", "yes")
        .option_str("presolve_fbbt", "yes")
        .option_str("derivative_test", "first-order")
        .option_int("print_level", 0)
        .solve();

        assert!(sol.success, "status = {:?}", sol.status);
        assert!(
            sol.fbbt_report
                .as_ref()
                .is_some_and(|report| report.bound_updates > 0)
        );
        assert!(
            gradient_points
                .borrow()
                .iter()
                .any(|&x| (x - 5.0).abs() < 1e-12),
            "derivative test did not use the original starting point"
        );
    }

    #[test]
    fn fbbt_infeasibility_is_reported() {
        let sol = Nlp::new(CircleConstraint)
            .var_bounds(&[-1.0], &[1.0])
            .constraint_bounds(&[2.0], &[2.0e19])
            .option_str("presolve", "yes")
            .option_str("presolve_fbbt", "yes")
            .solve();
        assert!(!sol.success);
        assert_eq!(
            sol.fbbt_report.and_then(|r| r.infeasibility_witness),
            Some(0)
        );
    }

    #[test]
    fn infers_n_from_bounds_and_solves() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0]) // n inferred = 2
            .constraint_bounds(&[3.0], &[3.0])
            .option_num("tol", 1e-10)
            .solve();
        assert!(sol.success);
        assert!((sol.x[0] - 1.0).abs() < 1e-5 && (sol.x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn infers_n_from_x0() {
        let sol = Nlp::new(Quad)
            .constraint_bounds(&[3.0], &[3.0])
            .x0(&[0.0, 0.0]) // n inferred = 2
            .solve();
        assert!(sol.success);
    }

    #[test]
    fn solve_populates_stats_and_duals() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .solve();
        assert!(sol.success);
        assert!(sol.stats.iteration_count > 0);
        assert!(sol.stats.total_wallclock_time_secs > 0.0);
        assert!(sol.stats.num_obj_evals > 0);
        assert!(sol.stats.final_constr_viol < 1e-6);
        assert_eq!(sol.g.len(), 1);
        assert!((sol.g[0] - 3.0).abs() < 1e-6, "g at solution: {:?}", sol.g);
        assert_eq!(sol.z_l.len(), 2);
        assert_eq!(sol.z_u.len(), 2);
        assert!(sol.stats.iterations.is_empty());
    }

    #[test]
    fn capture_iterations_fills_trajectory() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .capture_iterations()
            .solve();
        assert!(sol.success);
        let iters = &sol.stats.iterations;
        assert!(!iters.is_empty(), "no iteration records captured");
        assert_eq!(iters[0].iter, 0, "trajectory must start at iteration 0");
        assert!(
            iters.windows(2).all(|w| w[0].iter < w[1].iter),
            "iteration counter must be strictly increasing"
        );
    }

    #[test]
    fn capture_iterations_is_empty_on_sqp_engine() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .option_str("solver_selection", "qp-active-set")
            .capture_iterations()
            .solve();
        assert!(sol.success, "status = {:?}", sol.status);
        assert!(sol.stats.iteration_count > 0);
        assert!(sol.stats.iterations.is_empty());
    }

    #[test]
    fn qp_active_set_selection_solves() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .option_str("solver_selection", "qp-active-set")
            .solve();
        assert!(sol.success, "status = {:?}", sol.status);
        assert!((sol.x[0] - 1.0).abs() < 1e-4 && (sol.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn forced_convex_selection_fails_in_builder() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .option_str("solver_selection", "qp-ipm")
            .solve();
        assert!(
            !sol.success,
            "forced qp-ipm must not silently succeed via NLP"
        );
        assert_eq!(sol.status, ApplicationReturnStatus::InvalidOption);
    }

    /// `min (x - 1)^2`, with a singularity at exactly `x == 0` — the value a
    /// modelling layer hands over for a decision variable nobody
    /// initialised. The origin evaluates to NaN, so the first solve exits
    /// `Invalid_Number_Detected`; displacing the start clears the
    /// singularity and the re-solve converges. That is the failure rung 3
    /// exists for, and the only rung an invalid number opens.
    struct SingularAtOrigin;
    impl Problem for SingularAtOrigin {
        fn objective(&self, x: &[f64]) -> f64 {
            if x[0] == 0.0 {
                f64::NAN
            } else {
                (x[0] - 1.0).powi(2)
            }
        }
        fn gradient(&self, x: &[f64], grad: &mut [f64]) -> bool {
            grad[0] = if x[0] == 0.0 {
                f64::NAN
            } else {
                2.0 * (x[0] - 1.0)
            };
            true
        }
    }

    fn from_the_origin() -> Nlp<SingularAtOrigin> {
        Nlp::new(SingularAtOrigin)
            .x0(&[0.0])
            .option_int("print_level", 0)
    }

    /// The ladder runs from `pounce-rs` too, recovers this model, and says
    /// so. The narration goes nowhere on this surface unless the caller
    /// reads it, so `log` is the only place it exists — the ladder is on by
    /// default, and without this field an embedder sees a solve that
    /// silently took several times as long with nothing to attribute the
    /// time to.
    #[test]
    fn a_recovered_solve_reports_the_rung_that_recovered_it() {
        let sol = from_the_origin().solve();
        assert!(sol.success, "status = {:?}", sol.status);
        assert!((sol.x[0] - 1.0).abs() < 1e-5, "x = {:?}", sol.x);
        let so = sol
            .second_opinion
            .as_ref()
            .unwrap_or_else(|| panic!("no ladder; status = {:?}", sol.status));
        // An invalid number opens exactly the rung that moves the point:
        // re-running the same callbacks at the same point under a different
        // scaling or barrier strategy evaluates the same NaN again.
        assert_eq!(so.tried, ["start_point_perturbation=1e-2"]);
        assert_eq!(so.promoted_by, Some("start_point_perturbation=1e-2"));
        assert!(
            so.log
                .iter()
                .any(|l| l.contains("start_point_perturbation")),
            "the narration must be collected, not dropped: {:?}",
            so.log
        );
    }

    /// `None` is not "the ladder found nothing" — it is "the ladder did not
    /// run", which is the overwhelmingly common path. A caller that reads
    /// `promoted_by` without checking for `Some` first would otherwise read a
    /// converged solve as a rejected recovery.
    #[test]
    fn a_succeeding_solve_has_no_second_opinion() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .solve();
        assert!(sol.success);
        assert!(sol.second_opinion.is_none());
    }

    /// The opt-outs reach this surface as well — and this is the mutation
    /// guard for the test above: with the start rung off, the same model
    /// keeps its `Invalid_Number_Detected` and the field goes back to
    /// `None`, so `Some` there is evidence the ladder ran rather than an
    /// artifact of the status.
    #[test]
    fn turning_the_start_rung_off_gives_the_upstream_verdict_back() {
        let sol = from_the_origin()
            .option_str("infeasibility_perturbed_start_retry", "no")
            .solve();
        assert!(!sol.success, "status = {:?}", sol.status);
        assert_eq!(sol.status, ApplicationReturnStatus::InvalidNumberDetected);
        assert!(sol.second_opinion.is_none(), "{:?}", sol.second_opinion);
    }

    #[test]
    #[should_panic(expected = "already sized to 2")]
    fn mismatched_sizes_panic() {
        let _ = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .x0(&[0.0, 0.0, 0.0]) // length 3 != 2
            .solve();
    }

    #[test]
    #[should_panic(expected = "number of variables unknown")]
    fn missing_size_panics() {
        let _ = Nlp::new(Quad).constraint_bounds(&[3.0], &[3.0]).solve();
    }
}
