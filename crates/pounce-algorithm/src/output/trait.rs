//! Iteration-output trait — port of `IpIterationOutput.hpp`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;

/// Strategy that emits one row of the iter-by-iter table. The default
/// `write_output` is a no-op so structural unit tests can drive the
/// algorithm without an output sink. Phase 7 ports `OrigIterationOutput`
/// with the full upstream column format.
pub trait IterationOutput {
    fn write_output(&mut self) {}

    /// Format the next iteration row into a fresh `String`, given the
    /// current data + CQ snapshots. Default returns an empty string.
    /// Mirrors `IpOrigIterationOutput::WriteOutput` minus the
    /// journalist write — callers route the returned line wherever
    /// they want (stdout, file, log buffer for tests).
    fn format_row(&mut self, _data: &IpoptDataHandle, _cq: &IpoptCqHandle) -> String {
        String::new()
    }

    /// The `(objective, inf_pr)` pair the row prints, for the structured
    /// per-iteration event, so the solve report carries the console's
    /// numbers rather than a neighbouring quantity (gh#979). `None` means
    /// this output prints no such columns, and the caller falls back to
    /// the algorithm's own `unscaled_curr_f` / internal infeasibility.
    fn printed_objective_inf_pr(
        &mut self,
        _data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
    ) -> Option<(f64, f64)> {
        None
    }

    /// Whether the rows this output prints belong to the restoration
    /// phase (the `r`-suffixed rows). Tags the structured event.
    fn is_restoration(&self) -> bool {
        false
    }
}
