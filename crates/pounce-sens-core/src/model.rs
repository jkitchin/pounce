//! What the statistics and the step reports read about the model and the
//! solve beyond the engine's converged iterate: the model's declared bounds
//! and its own functions, in the model's indices and units, the point and
//! objective value the solve returned, and the rows the caller declared as
//! parameters, fitted variables and residuals.
//!
//! The engine answers for the KKT system. These answer for the model the
//! caller wrote, which is where a perturbed point is measured: how far it
//! leaves a bound, and how far it violates a row.

use pounce_common::types::Number;

/// The model's declared bounds and its constraint functions, in the
/// model's own variable and row order (full-x and full-g) and units.
pub trait ModelFunctions {
    /// The variables' declared lower and upper bounds. An absent bound is
    /// whatever magnitude the model uses for one, `±1e19` for an `.nl`.
    fn variable_bounds(&self) -> (Vec<Number>, Vec<Number>);

    /// The rows' declared lower and upper bounds.
    fn row_bounds(&self) -> (Vec<Number>, Vec<Number>);

    /// The constraint values `g(x)` at a model point.
    fn constraints(&mut self, x: &[Number]) -> Result<Vec<Number>, String>;

    /// The constraint Jacobian at a model point, as `(row, column, value)`
    /// entries in the order the model reports them.
    fn jacobian(&mut self, x: &[Number]) -> Result<Vec<(usize, usize, Number)>, String>;

    /// The gradient of the objective at a model point, in the sense the
    /// engine minimized it.
    fn objective_gradient(&mut self, x: &[Number]) -> Result<Vec<Number>, String>;
}

/// The point and the objective value the solve returned to its caller.
#[derive(Debug, Clone, Copy)]
pub struct SolveRecord<'a> {
    /// The solved point over the model's variables, in its own units.
    pub x: &'a [Number],
    /// The objective value at the solve, in the sense the model states it.
    /// NaN when the solve reported none.
    pub obj: Number,
    /// `1` for a model written as a minimization and `-1` for a
    /// maximization: the factor between the objective the engine minimized
    /// and the one the model states.
    pub obj_sign: Number,
}

/// The rows the caller declared before the solve.
#[derive(Debug, Clone, Copy, Default)]
pub struct Declarations<'a> {
    /// The model row pinning each declared parameter, in declaration order.
    pub pins: &'a [usize],
    /// The model variable of each fitted parameter, in declaration order.
    pub fit_rows: &'a [usize],
    /// The model variables of each residual group, in declaration order.
    pub res_groups: &'a [Vec<usize>],
}
