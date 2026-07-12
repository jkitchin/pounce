//! Original iteration output — port of
//! `Algorithm/IpOrigIterationOutput.{hpp,cpp}`.
//!
//! Column layout follows upstream's literal `Snprintf` schema but
//! widens the `lg(mu)` / `lg(rg)` / e-format fields by one or two
//! characters so that:
//!
//! * the e-format columns no longer wiggle by one character when a
//!   value transitions between 1-digit and 2-digit exponent
//!   magnitudes (`1.44e-7` vs `8.83e-13`), since [`format_e`] always
//!   emits the C `%.Ne` form (signed, zero-padded 2-digit exponent);
//! * each header label right-aligns exactly to the right edge of its
//!   data column, instead of inheriting upstream's hand-rolled spacing
//!   that left several labels off by 1–2 characters.

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

    /// Header line printed every ten iterations. Each label is
    /// right-aligned to the right edge of its data column under the
    /// new widths (see module docs).
    pub const HEADER: &'static str =
        "iter      objective   inf_pr   inf_du lg(mu)    ||d|| lg(rg) alpha_du alpha_pr  ls\n";
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
        let unscaled_f = c.unscaled_curr_f();
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
            "     -".to_string()
        } else {
            format!("{:6.1}", regu_x.log10())
        };

        let alpha_dual = d.info_alpha_dual;
        let alpha_primal = d.info_alpha_primal;
        let alpha_char = d.info_alpha_primal_char;
        let ls_count = d.info_ls_count;

        let mut row = format!(
            "{:>4} {:>14} {:>8} {:>8} {:6.1} {:>8} {:>6} {:>8} {:>8}{}{:>3}",
            iter,
            format_e(unscaled_f, 7),
            format_e(inf_pr, 2),
            format_e(inf_du, 2),
            lg_mu,
            format_e(dnrm, 2),
            regu_str,
            format_e(alpha_dual, 2),
            format_e(alpha_primal, 2),
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

/// Format `x` in C printf `%.{precision}e` style — signed exponent,
/// zero-padded to at least two digits. E.g. `0.178` with precision 2
/// → `"1.78e-01"`, `1.0` → `"1.00e+00"`, `8.83e-13` → `"8.83e-13"`.
///
/// Rust's native `{:.Ne}` formatter emits the exponent with no sign
/// and no zero-pad (so `1e0`, `1.78e-1`, `8.83e-13` are 6 / 7 / 8
/// chars respectively), which causes the e-format columns in the
/// iteration log to wiggle as the exponent magnitude changes. This
/// helper normalises to the C `%e` form, which is always 8 chars for
/// 1-precision e-fields with 1- or 2-digit exponents.
pub(crate) fn format_e(x: f64, precision: usize) -> String {
    if !x.is_finite() {
        return format!("{}", x);
    }
    let s = format!("{:.*e}", precision, x);
    let (mantissa, exp) = match s.split_once('e') {
        Some(pair) => pair,
        None => return s,
    };
    let (sign, digits) = match exp.strip_prefix('-') {
        Some(rest) => ('-', rest),
        None => ('+', exp),
    };
    if digits.len() == 1 {
        format!("{}e{}0{}", mantissa, sign, digits)
    } else {
        format!("{}e{}{}", mantissa, sign, digits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_layout_right_aligns_each_label() {
        // Width budget: iter(4) sp obj(14) sp inf_pr(8) sp inf_du(8)
        // sp lg_mu(6) sp dnrm(8) sp regu(6) sp alpha_du(8) sp
        // alpha_pr(8) alpha_char(1) ls(3) = 82 chars.
        assert_eq!(OrigIterationOutput::HEADER.len(), 83); // 82 + \n
        // Spot-check the right-edges of a few labels.
        let h = OrigIterationOutput::HEADER.trim_end_matches('\n');
        assert!(h.ends_with("ls"), "h = {h:?}");
        assert_eq!(&h[10..19], "objective");
        assert_eq!(&h[22..28], "inf_pr");
        assert_eq!(&h[61..69], "alpha_du");
        assert_eq!(&h[70..78], "alpha_pr");
    }

    #[test]
    fn format_e_pads_short_exponents() {
        assert_eq!(format_e(0.0, 2), "0.00e+00");
        assert_eq!(format_e(1.0, 2), "1.00e+00");
        assert_eq!(format_e(0.178, 2), "1.78e-01");
        assert_eq!(format_e(8.83e-13, 2), "8.83e-13");
        assert_eq!(format_e(7.74, 2), "7.74e+00");
        // 2-digit exponent: no padding needed.
        assert_eq!(format_e(1.0e10, 2), "1.00e+10");
    }

    #[test]
    fn format_e_passes_through_non_finite() {
        assert_eq!(format_e(f64::NAN, 2), "NaN");
        assert_eq!(format_e(f64::INFINITY, 2), "inf");
    }
}
