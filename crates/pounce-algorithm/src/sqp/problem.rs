//! Minimal NLP-evaluation trait the SQP outer loop binds against.
//!
//! Distinct from [`crate::ipopt_nlp::IpoptNlp`] (the rich IPM-
//! shaped interface with slacks, bound vector spaces, and
//! IPM-specific initialization hooks). `SqpProblemSpec` is a thin
//! evaluation surface — just what `SqpAlgorithm::optimize` calls
//! per iteration.
//!
//! An adapter from `IpoptNlp` to `SqpProblemSpec` lands in a
//! later commit so the same benchmarks (CUTEst, `.nl` files,
//! `pounce-py`) can drive both algorithm paths via the
//! `AlgorithmChoice` dispatch in `alg_builder`.

use crate::sqp::qp_assembly::Triplet;
use pounce_common::Number;

pub trait SqpProblemSpec {
    fn n(&self) -> usize;
    fn m(&self) -> usize;

    fn x_init(&self) -> Vec<Number>;

    fn variable_bounds(&self) -> (Vec<Number>, Vec<Number>);
    fn constraint_bounds(&self) -> (Vec<Number>, Vec<Number>);

    fn eval_f(&mut self, x: &[Number]) -> Number;
    fn eval_grad_f(&mut self, x: &[Number]) -> Vec<Number>;

    /// `c(x)` — combined constraint values (length `m`). The
    /// constraint bounds from `constraint_bounds` apply directly:
    /// row `i` is a strict equality if `bl[i] == bu[i]`, an
    /// inequality otherwise.
    fn eval_c(&mut self, x: &[Number]) -> Vec<Number>;

    /// `∇c(x)` as a sparse `m × n` triplet (1-based indices).
    fn eval_jac_c(&mut self, x: &[Number]) -> Triplet;

    /// `∇²L(x, λ_g) = ∇²f(x) + Σ λ_g_i · ∇²c_i(x)` as a sparse
    /// symmetric `n × n` triplet.
    fn eval_hess_lag(&mut self, x: &[Number], lambda_g: &[Number]) -> Triplet;
}
