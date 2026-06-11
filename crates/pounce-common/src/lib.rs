//! POUNCE common primitives.
//!
//! Port of Ipopt's `src/Common/`: scalar types, exceptions, journalist,
//! tagged-object change tracking, cached results, registered options
//! and options list, utility helpers, and timed-task accumulators.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod cached;
pub mod debug;
pub mod diagnostics;
pub mod exception;
pub mod journalist;
pub mod options_list;
pub mod reg_options;
pub mod style;
pub mod tagged;
pub mod timing;
pub mod types;
pub mod utils;

pub use cached::Cache;
pub use diagnostics::{DiagCategory, DiagnosticsConfig, DiagnosticsState, DumpFormat, IterSpec};
pub use exception::{ExceptionKind, SolverException};
pub use journalist::{
    FileJournal, Journal, JournalCategory, JournalLevel, Journalist, StringJournal,
};
pub use options_list::OptionsList;
pub use reg_options::{DefaultValue, OptionType, RegisteredOption, RegisteredOptions, StringEntry};
pub use tagged::{Tag, TaggedCell, TaggedObject};
pub use timing::{TimedTask, TimingStatistics};
pub use types::{Index, Number, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
