//! POUNCE NLP-side glue.
//!
//! Port of Ipopt's `src/Interfaces/`. This crate provides:
//!
//! * The user-facing [`tnlp::TNLP`] trait for problem definition
//!   (port of `IpTNLP.{hpp,cpp}`).
//! * Public return-code enums [`return_codes::ApplicationReturnStatus`]
//!   and [`return_codes::AlgorithmMode`] (port of `IpReturnCodes_inc.h`)
//!   plus algorithm-side [`alg_types::SolverReturn`] (port of
//!   `IpAlgTypes.hpp`).
//! * Per-solve [`solve_statistics::SolveStatistics`] counters (port of
//!   `IpSolveStatistics.{hpp,cpp}`).
//! * `TNLPAdapter` and `OrigIpoptNlp`, the bound/constraint splitter
//!   chain feeding the algorithm-side IPM.
//! * [`diagnostics`]: model checks that need only a TNLP — starting-point
//!   preflight and solution verification — so an embedder gets what the
//!   `pounce` CLI's `check-x0` / `verify` subcommands report.
//! * Transparent TNLP decorators every frontend can stack:
//!   [`counting_tnlp::CountingTnlp`] (evaluation counts) and
//!   [`seeded_tnlp::SeededTnlp`] (primal warm start from a chosen
//!   iterate).
//!
//! The user-facing `IpoptApplication` lives in `pounce-algorithm`
//! (since `optimize_tnlp` orchestrates the algorithm). It is
//! re-exported as `pounce_algorithm::IpoptApplication`. The dependency
//! direction is `pounce-algorithm → pounce-nlp`; this crate must not
//! import `pounce-algorithm` types.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod alg_types;
pub mod constant_derivatives;
pub mod counting_tnlp;
pub mod derivative_test;
pub mod diagnostics;
pub mod expression_provider;
pub mod ipopt_nlp;
pub mod orig_ipopt_nlp;
pub mod quadratic;
pub mod return_codes;
pub mod scaling_tnlp;
pub mod seeded_tnlp;
pub mod solve_statistics;
pub mod tnlp;
pub mod tnlp_adapter;

pub use alg_types::SolverReturn;
pub use constant_derivatives::{
    ConstantDerivatives, DerivativeProof, DerivativeProofs, HintOutcome,
};
pub use counting_tnlp::CountingTnlp;
pub use expression_provider::{ExpressionProvider, FbbtOp, FbbtTape};
pub use ipopt_nlp::{IpoptNlp, Nlp};
pub use orig_ipopt_nlp::{ConstObjScaling, NlpScaling, NoScaling, OrigIpoptNlp};
pub use quadratic::QuadraticStructure;
pub use return_codes::{AlgorithmMode, ApplicationReturnStatus};
pub use seeded_tnlp::SeededTnlp;
pub use solve_statistics::SolveStatistics;
pub use tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo,
    ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};
pub use tnlp_adapter::{BoundClassification, TNLPAdapter};
