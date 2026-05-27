//! Feasibility-Based Bound Tightening (FBBT).
//!
//! Tracks pounce issue [#62]. FBBT walks each nonlinear constraint's
//! expression DAG to tighten variable bounds by interval propagation:
//!
//! 1. **Forward pass** — bottom-up, compute an outward-rounded
//!    interval over-approximation of each node's value given current
//!    variable bounds.
//! 2. **Reverse pass** — top-down, narrow each child's interval
//!    against the parent's `[lb, ub]` and sibling intervals.
//! 3. **Read-back** — leaves' tightened intervals become the new
//!    variable bounds, applied when strictly tighter by at least
//!    `fbbt_tol`.
//!
//! The orchestrator iterates until no bound improves or
//! `fbbt_max_iter` is hit (Belotti et al., 2010 — FBBT does not
//! converge finitely in general).
//!
//! # Phase status
//!
//! Landing in three commits per issue #62 review:
//!
//! 1. **Interval arithmetic** — `interval.rs`. _This file._
//! 2. Expression-provider trait + forward pass —
//!    `forward.rs`, plus a public accessor on `NlTnlp`.
//! 3. Reverse-propagation rules + orchestrator + options +
//!    integration tests — `reverse.rs`, `mod.rs` orchestrator.
//!
//! Until phase 3 lands the module is unused; the orchestrator and
//! presolve-option wiring follow.
//!
//! [#62]: https://github.com/jkitchin/pounce/issues/62

pub mod interval;
