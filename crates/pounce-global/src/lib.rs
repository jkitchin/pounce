//! Deterministic **global** optimization for factorable nonconvex NLPs.
//!
//! Where [`pounce_convex`] finds the optimum of a *convex* problem and the NLP
//! filter-IPM finds a *local* optimum of a smooth one, this crate certifies a
//! **global** optimum of a nonconvex one by spatial branch-and-bound:
//!
//! * **Lower bounds** — a McCormick polyhedral [`relax`]ation of the problem
//!   over the current box is a linear program solved through
//!   [`pounce_convex`]; its optimum underestimates the true minimum, exactly in
//!   the zero-width-box limit.
//! * **Domain reduction** — [`pounce_presolve`]'s feasibility-based bound
//!   tightening (FBBT) shrinks each box and prunes infeasible ones.
//! * **Upper bounds** — probing feasible points (the relaxation solution, the
//!   box center) furnishes an incumbent.
//! * **Branching** — bisecting the widest variable refines the relaxation.
//!
//! The loop returns a globally optimal point with a certified optimality gap.
//!
//! # Example
//!
//! ```
//! use pounce_global::{expr::{var, con}, GlobalProblem, GlobalOptions, solve_global, GlobalStatus};
//! use pounce_feral::FeralSolverInterface;
//!
//! // The six-hump-camel-flavored 1-D toy 2x⁴ − x² has two global minima.
//! // Here a simpler nonconvex case: minimize x·sin-free quartic x⁴ − 3x² over
//! // [−2, 2], global minimum at x = ±√(3/2).
//! let f = var(0).powi(4) - 3.0 * var(0).powi(2);
//! let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);
//! let sol = solve_global(&prob, &GlobalOptions::default(), || Box::new(FeralSolverInterface::new()));
//! assert_eq!(sol.status, GlobalStatus::Optimal);
//! assert!((sol.objective - (-2.25)).abs() < 1e-4);
//! ```

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod ad;
mod alphabb;
pub mod bnb;
mod branching;
pub(crate) mod debug;
mod envelope;
pub mod expr;
mod nlp;
mod obbt;
pub mod problem;
mod relax;
mod rlt;

pub use bnb::{
    estimate_node_bytes, solve_global, solve_global_debug, solve_global_debug_into, BranchRule,
    GlobalOptions, GlobalSolution, GlobalStatus,
};
pub use expr::Expr;
pub use problem::{Constraint, GlobalProblem};
