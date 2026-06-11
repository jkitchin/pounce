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
//!
//! The user-facing `IpoptApplication` lives in `pounce-algorithm`
//! (since `optimize_tnlp` orchestrates the algorithm). It is
//! re-exported as `pounce_algorithm::IpoptApplication`. The dependency
//! direction is `pounce-algorithm → pounce-nlp`; this crate must not
//! import `pounce-algorithm` types.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod alg_types;
pub mod expression_provider;
pub mod ipopt_nlp;
pub mod orig_ipopt_nlp;
pub mod return_codes;
pub mod solve_statistics;
pub mod tnlp;
pub mod tnlp_adapter;

pub use alg_types::SolverReturn;
pub use expression_provider::{ExpressionProvider, FbbtOp, FbbtTape};
pub use ipopt_nlp::{IpoptNlp, Nlp};
pub use orig_ipopt_nlp::{ConstObjScaling, NlpScaling, NoScaling, OrigIpoptNlp};
pub use return_codes::{AlgorithmMode, ApplicationReturnStatus};
pub use solve_statistics::SolveStatistics;
pub use tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, Linearity, MetaData, NlpInfo,
    ScalingRequest, Solution, SparsityRequest, StartingPoint, TNLP,
};
pub use tnlp_adapter::{BoundClassification, TNLPAdapter};
