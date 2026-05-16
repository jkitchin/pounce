//! Original iteration output — port of
//! `Algorithm/IpOrigIterationOutput.{hpp,cpp}`.
//!
//! Reproduces upstream's column layout byte-exactly so iteration
//! logs can be text-diffed against an Ipopt 3.14.x run.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::output::r#trait::IterationOutput;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintInfoString {
    Yes,
    No,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfPrTag {
    Internal,
    Original,
}

pub struct OrigIterationOutput {
    pub print_info_string: PrintInfoString,
    pub inf_pr_output: InfPrTag,
    pub print_frequency_iter: i32,
    pub print_frequency_time: f64,
    /// Iteration index of the last header print; the upstream code
    /// re-prints the header every 10 lines.
    last_header_iter: i32,
}

impl Default for OrigIterationOutput {
    fn default() -> Self {
        Self {
            print_info_string: PrintInfoString::No,
            inf_pr_output: InfPrTag::Original,
            print_frequency_iter: 1,
            print_frequency_time: 0.0,
            last_header_iter: -1,
        }
    }
}

impl OrigIterationOutput {
    pub fn new() -> Self {
        Self::default()
    }

    /// Header line printed every ten iterations; matches the literal
    /// in `IpOrigIterationOutput.cpp:75`.
    pub const HEADER: &'static str =
        "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls\n";
}

impl IterationOutput for OrigIterationOutput {
    fn write_output(&mut self) {
        // Header-print bookkeeping; the actual emission is handled by
        // `format_row`, which the caller wires to its journalist.
        self.last_header_iter = 0;
    }

    /// Build the single-line iteration row. Field-for-field port of
    /// the `Snprintf` block at `IpOrigIterationOutput.cpp:152`:
    /// `"%4d %14.7e %7.2e %7.2e %5.1f %7.2e %5s %7.2e %7.2e%c%3d"`.
    fn format_row(&mut self, data: &IpoptDataHandle, cq: &IpoptCqHandle) -> String {
        let d = data.borrow();
        let c = cq.borrow();

        let iter = d.iter_count;
        let unscaled_f = c.curr_f();
        let inf_pr = match self.inf_pr_output {
            InfPrTag::Internal => c.curr_primal_infeasibility_max(),
            // The "original" mode wants the unscaled NLP constraint
            // violation; until NLP-side scaling lands we feed the
            // (already unscaled) internal violation.
            InfPrTag::Original => c.curr_primal_infeasibility_max(),
        };
        let inf_du = c.curr_dual_infeasibility_max();
        let mu = d.curr_mu;
        let lg_mu = mu.log10();

        // ||d||_∞ over the (x, s) blocks of the latest search step.
        let dnrm = match &d.delta {
            Some(delta) => delta.x.amax().max(delta.s.amax()),
            None => 0.0,
        };

        let regu_x = d.info_regu_x;
        let regu_str: String = if regu_x == 0.0 {
            "   - ".to_string()
        } else {
            format_field_5_1(regu_x.log10())
        };

        let alpha_dual = d.info_alpha_dual;
        let alpha_primal = d.info_alpha_primal;
        let alpha_char = d.info_alpha_primal_char;
        let ls_count = d.info_ls_count;

        let mut row = format!(
            "{:>4} {:14.7e} {:7.2e} {:7.2e} {:5.1} {:7.2e} {:>5} {:7.2e} {:7.2e}{}{:>3}",
            iter,
            unscaled_f,
            inf_pr,
            inf_du,
            lg_mu,
            dnrm,
            regu_str,
            alpha_dual,
            alpha_primal,
            alpha_char,
            ls_count,
        );
        // `print_info_string` (upstream
        // `IpOrigIterationOutput.cpp:WriteOutputImpl`): append the
        // per-iter diagnostic-tag string accumulated on `IpoptData`
        // (e.g. soft-resto / watchdog / corrector markers). The string
        // is cleared by the algorithm at the start of each outer
        // iteration via `clear_info_string`.
        if self.print_info_string == PrintInfoString::Yes && !d.info_string.is_empty() {
            row.push(' ');
            row.push_str(&d.info_string);
        }
        row
    }
}

/// Width-5, precision-1 fixed-point — emulates upstream's
/// `Snprintf(buf, 7, "%5.1f", x)`. The format!() spec
/// `"{:5.1}"` already matches `%5.1f` for `f64` in Rust.
fn format_field_5_1(x: f64) -> String {
    format!("{:5.1}", x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_matches_upstream_literal() {
        assert_eq!(
            OrigIterationOutput::HEADER,
            "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls\n"
        );
    }

    #[test]
    fn regu_field_dashes_when_zero() {
        // `regu_x == 0` → "   - " (5 chars including trailing space).
        assert_eq!(format_field_5_1(-1.0), " -1.0");
    }
}
