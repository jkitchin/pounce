//! Unit tests for the crate. Phase 5a delivers:
//!
//! * `api` — type-plumbing tests (construct, validate, default).
//!   These prove the public surface compiles and behaves as
//!   advertised, and serve as the integration anchor for
//!   downstream crates depending on `pounce-qp` while the solver
//!   internals are still under construction.
//! * `analytical` — the §8.0 analytical correctness ladder. Lands
//!   alongside the solver implementation in a later commit; the
//!   module shell sits empty here so the ladder's contract is
//!   visible in the tree from day one.

mod analytical;
mod api;
mod elastic_unit;
mod kkt_unit;
mod qps_unit;
mod refinement_unit;
mod scaling_unit;
mod schur_unit;
