//! High-level symmetric linear-solver trait — port of
//! `IpSymLinearSolver.hpp`.
//!
//! This is the algorithm-side abstraction (it takes a `SymMatrix` /
//! `Vector` rather than raw triplet arrays). The concrete
//! [`crate::SparseSymLinearSolverInterface`] backend is wrapped by
//! `TSymLinearSolver` (Phase 5+, when `IpoptData` lands).
//!
//! Phase 4 only needs the trait surface so `pounce-hsl` can be
//! compiled and tested against it. The `multi_solve` entry point
//! lands in Phase 5 once the `SymMatrix`/`Vector` traits from Phase 2
//! are connected to `IpoptData`.

use pounce_common::types::Index;

pub trait SymLinearSolver {
    /// Most recent factorization's negative-eigenvalue count.
    fn number_of_neg_evals(&self) -> Index;

    /// Ask for a higher-quality next solve.
    fn increase_quality(&mut self) -> bool;

    /// Whether this solver reports inertia.
    fn provides_inertia(&self) -> bool;

    /// gh#981: whether the most recent `Singular` came from a pivot the
    /// backend found to be zero rather than from an untrusted inertia
    /// count. See `SparseSymLinearSolverInterface::singularity_certified`.
    fn singularity_certified(&self) -> bool {
        false
    }
}
