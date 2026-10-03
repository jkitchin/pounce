//! Restoration-phase iteration output — port of
//! `Algorithm/IpRestoIterationOutput.{hpp,cpp}`.
//!
//! Differences from `OrigIterationOutput` (`pounce-algorithm/src/output/orig.rs`):
//!
//! * the iteration counter is followed by a literal `r` (not a space) so
//!   restoration rows are visually distinct in a unified log;
//! * the objective column reports the **original** NLP's unscaled
//!   trial objective at the resto-trial point (after copying the
//!   `orig_x` slice of the resto iterate into the orig trial), not the
//!   resto problem's own `f`;
//! * the `inf_pr` column reports either the resto problem's primal
//!   infeasibility (`InfPrTag::Internal`) or the original NLP's
//!   unscaled constraint violation at the trial (`InfPrTag::Original`);
//! * `inf_du`, `mu`, `‖d‖`, `lg(rg)`, `alpha_*`, and the LS count come
//!   from the **resto** `IpoptData`/CQ exactly as in the orig formatter.
//!
//! The plumbing that supplies the orig-NLP numbers depends on
//! `RestoIpoptNlp::orig_ip_nlp/orig_ip_cq` being wired up. Until then,
//! [`RestoIterationOutput::format_row_explicit`] takes the relevant
//! numbers as parameters so this module can be unit-tested standalone.

/// Mirrors `IpOrigIterationOutput.hpp:InfPrOutput` for the resto
/// formatter. The `inf_pr` column either reports the resto problem's
/// internal primal infeasibility or the original NLP's unscaled
/// constraint violation at the trial point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfPrTag {
    Internal,
    Original,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintInfoString {
    Yes,
    No,
}

pub struct RestoIterationOutput {
    pub print_info_string: PrintInfoString,
    pub inf_pr_output: InfPrTag,
    pub print_frequency_iter: i32,
    pub print_frequency_time: f64,
}

impl Default for RestoIterationOutput {
    fn default() -> Self {
        // Defaults mirror upstream `RegisterOptions` (inherited from
        // `OrigIterationOutput`).
        Self {
            print_info_string: PrintInfoString::No,
            inf_pr_output: InfPrTag::Original,
            print_frequency_iter: 1,
            print_frequency_time: 0.0,
        }
    }
}

impl RestoIterationOutput {
    pub fn new() -> Self {
        Self::default()
    }

    /// Header line — matches [`pounce_algorithm::output::OrigIterationOutput::HEADER`]
    /// so the resto-phase block reads as a continuation of the
    /// orig-phase log under the same column widths.
    pub const HEADER: &'static str =
        "iter      objective   inf_pr   inf_du lg(mu)    ||d|| lg(rg) alpha_du alpha_pr  ls\n";

    /// Build the single-line restoration iteration row, mirroring the
    /// `Snprintf` block at `IpRestoIterationOutput.cpp:156`:
    /// `"%4d r%14.7e %7.2e %7.2e %5.1f %7.2e %5s %7.2e %7.2e%c%3d"`.
    ///
    /// `iter` and `f` come from the resto-side `IpData` and the **orig**
    /// CQ respectively (orig `unscaled_trial_f` after the orig trial
    /// has been set to the resto-trial's `x_only`/`s_only` slices).
    /// `inf_pr` is whichever of the two infeasibilities the
    /// `inf_pr_output` setting selects (the caller has already chosen).
    /// `inf_du`, `mu`, `dnrm`, `regu_x`, `alpha_*`, `alpha_char`, and
    /// `ls_count` are all resto-side.
    #[allow(clippy::too_many_arguments)]
    pub fn format_row_explicit(
        &self,
        iter: i32,
        f: f64,
        inf_pr: f64,
        inf_du: f64,
        mu: f64,
        dnrm: f64,
        regu_x: f64,
        alpha_dual: f64,
        alpha_primal: f64,
        alpha_char: char,
        ls_count: i32,
    ) -> String {
        let lg_mu = mu.log10();
        let regu_str: String = if regu_x == 0.0 {
            "     -".to_string()
        } else {
            format!("{:6.1}", regu_x.log10())
        };
        format!(
            "{:>4}r{:>14} {:>8} {:>8} {:6.1} {:>8} {:>6} {:>8} {:>8}{}{:>3}",
            iter,
            format_e(f, 7),
            format_e(inf_pr, 2),
            format_e(inf_du, 2),
            lg_mu,
            format_e(dnrm, 2),
            regu_str,
            format_e(alpha_dual, 2),
            format_e(alpha_primal, 2),
            alpha_char,
            ls_count,
        )
    }
}

/// C-style `%.Ne` formatter — see
/// [`pounce_algorithm::output::orig::format_e`] for the rationale.
/// Duplicated here to avoid widening the orig-output API surface for
/// a single helper.
fn format_e(x: f64, precision: usize) -> String {
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

/// Adapter wiring [`RestoIterationOutput`] into the algorithm-side
/// [`pounce_algorithm::output::r#trait::IterationOutput`] trait.
///
/// Pulls every numeric column from the inner-IPM data / CQ exactly as
/// `OrigIterationOutput::format_row` does, then routes through
/// [`RestoIterationOutput::format_row_explicit`] so the row gets the
/// literal `r` suffix on the iter index.
///
/// When `orig_nlp` is wired ([`Self::with_orig_nlp`]), the `inf_pr`
/// and `objective` columns report the **original** NLP's unscaled
/// constraint violation and unscaled objective at `x_orig`, in the main
/// rows' units (see `eval_orig_at_inner_curr`, gh#981) — after upstream
/// `IpRestoIterationOutput.cpp:106-156`. Without it, both fall back
/// to the resto NLP's own values (the original v0.1 behavior).
pub struct RestoIterationOutputAdapter {
    pub inner: RestoIterationOutput,
    orig_nlp: Option<std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>>,
    /// `(iter, objective, inf_pr)` of the row `format_row` last printed,
    /// handed to [`Self::printed_objective_inf_pr`] so the structured
    /// event reports the printed numbers without evaluating the original
    /// NLP a second time (gh#979). Taken on read, so it is one-shot.
    last_printed: Option<(i32, f64, f64)>,
}

impl RestoIterationOutputAdapter {
    pub fn new() -> Self {
        Self {
            inner: RestoIterationOutput::new(),
            orig_nlp: None,
            last_printed: None,
        }
    }

    /// Wire the original NLP so `format_row` can report orig-NLP
    /// `inf_pr` / objective on resto rows. Mirrors upstream's
    /// `RestoIterationOutput::WriteOutput` reading `orig_ip_cq`.
    pub fn with_orig_nlp(
        mut self,
        orig: std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
    ) -> Self {
        self.orig_nlp = Some(orig);
        self
    }
}

impl Default for RestoIterationOutputAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl pounce_algorithm::output::r#trait::IterationOutput for RestoIterationOutputAdapter {
    fn format_row(
        &mut self,
        data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
        cq: &pounce_algorithm::ipopt_cq::IpoptCqHandle,
    ) -> String {
        let (unscaled_f, inf_pr) = self.objective_inf_pr(data, cq);
        let d = data.borrow();
        let c = cq.borrow();

        let iter = d.iter_count;
        self.last_printed = Some((iter, unscaled_f, inf_pr));
        let inf_du = c.curr_dual_infeasibility_max();
        let mu = d.curr_mu;

        let dnrm = match &d.delta {
            Some(delta) => delta.x.amax().max(delta.s.amax()),
            None => 0.0,
        };

        let regu_x = d.info_regu_x;
        let alpha_dual = d.info_alpha_dual;
        let alpha_primal = d.info_alpha_primal;
        let alpha_char = d.info_alpha_primal_char;
        let ls_count = d.info_ls_count;

        self.inner.format_row_explicit(
            iter,
            unscaled_f,
            inf_pr,
            inf_du,
            mu,
            dnrm,
            regu_x,
            alpha_dual,
            alpha_primal,
            alpha_char,
            ls_count,
        )
    }

    fn printed_objective_inf_pr(
        &mut self,
        data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
        cq: &pounce_algorithm::ipopt_cq::IpoptCqHandle,
    ) -> Option<(f64, f64)> {
        let iter = data.borrow().iter_count;
        match self.last_printed.take() {
            Some((it, f, pr)) if it == iter => Some((f, pr)),
            _ => Some(self.objective_inf_pr(data, cq)),
        }
    }

    fn is_restoration(&self) -> bool {
        true
    }
}

impl RestoIterationOutputAdapter {
    /// The objective and `inf_pr` columns of a restoration row. Resto-NLP
    /// fallbacks, overridden when `orig_nlp` is wired so the row reports
    /// orig-NLP `f` and `inf_pr` at the `(x_orig, s)` slice of the resto
    /// iterate (upstream `IpRestoIterationOutput.cpp:106-156`).
    fn objective_inf_pr(
        &self,
        data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
        cq: &pounce_algorithm::ipopt_cq::IpoptCqHandle,
    ) -> (f64, f64) {
        let d = data.borrow();
        let c = cq.borrow();
        if let (Some(orig_rc), Some(curr)) = (&self.orig_nlp, d.curr.as_ref()) {
            if let Some(vals) = eval_orig_at_inner_curr(curr, orig_rc, self.inner.inf_pr_output) {
                return vals;
            }
        }
        (c.curr_f(), c.curr_primal_infeasibility_max())
    }
}

/// Evaluate the original NLP's unscaled `f(x_orig)` and its constraint
/// violation at the resto iterate's `x_orig` slice (block 0 of the
/// compound `x`). Returns `None` if any of the downcasts / dimension
/// reads fail (in which case the caller falls back to resto-NLP values).
///
/// The violation follows the main rows' convention (gh#981), so a
/// restoration row and the main row beside it are in the same units:
///
/// * [`InfPrTag::Original`] (the default) — the user's constraints in
///   user units against their declared bounds, the quantity
///   `OrigIterationOutput` prints, computed by the same
///   [`pounce_algorithm::ipopt_cq::unscaled_nlp_constraint_violation_max`].
///   This used to be `max(‖c‖∞, ‖d − s‖∞)` on the *row-scaled* `c` and
///   `d` that `IpoptNlp::eval_c` / `eval_d` return: on a badly scaled
///   water main the main rows read 2.21e5 metres of head and the
///   restoration rows beside them 0.211.
/// * [`InfPrTag::Internal`] — that scaled slack-form residual,
///   `max(‖c(x_orig)‖∞, ‖d(x_orig) − s‖∞)` with the inner `s`.
fn eval_orig_at_inner_curr(
    curr: &pounce_algorithm::iterates_vector::IteratesVector,
    orig_rc: &std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
    inf_pr_output: InfPrTag,
) -> Option<(f64, f64)> {
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use pounce_linalg::{CompoundVector, Vector};

    let xc = curr.x.as_any().downcast_ref::<CompoundVector>()?;
    let x_orig = xc.comp(crate::resto_nlp::BLOCK_X);
    let s_inner = &*curr.s;

    let mut orig = orig_rc.borrow_mut();
    let m_eq = orig.m_eq();
    let m_ineq = orig.m_ineq();

    // c(x_orig), d(x_orig) — row-scaled, as the NLP evaluates them.
    let mut c_buf = DenseVectorSpace::new(m_eq).make_new_dense();
    if m_eq > 0 {
        orig.eval_c(x_orig, &mut c_buf);
    }
    let mut d_buf = DenseVectorSpace::new(m_ineq).make_new_dense();
    if m_ineq > 0 {
        orig.eval_d(x_orig, &mut d_buf);
    }

    let inf_pr = match inf_pr_output {
        InfPrTag::Original => {
            pounce_algorithm::ipopt_cq::unscaled_nlp_constraint_violation_max(
                &*orig, &c_buf, &d_buf,
            )
        }
        InfPrTag::Internal => {
            let c_amax = if m_eq > 0 { c_buf.amax() } else { 0.0 };
            let d_minus_s_amax = if m_ineq > 0 {
                d_buf.axpy(-1.0, s_inner);
                d_buf.amax()
            } else {
                0.0
            };
            c_amax.max(d_minus_s_amax)
        }
    };

    // `eval_f` is the scaled objective; unscale it the way
    // `IpoptCalculatedQuantities::unscaled_curr_f` does for main rows.
    let f_scaled = orig.eval_f(x_orig);
    let factor = orig.obj_scaling_factor();
    let f = if factor == 0.0 { f_scaled } else { f_scaled / factor };
    Some((f, inf_pr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_matches_orig_layout() {
        // Resto rows continue under the orig header — both must use the
        // same right-aligned column widths.
        assert_eq!(
            RestoIterationOutput::HEADER,
            "iter      objective   inf_pr   inf_du lg(mu)    ||d|| lg(rg) alpha_du alpha_pr  ls\n"
        );
    }

    #[test]
    fn row_has_r_suffix_after_iter_field() {
        let out = RestoIterationOutput::new();
        let row =
            out.format_row_explicit(7, 1.234567e+0, 1e-3, 1e-4, 0.1, 1e-2, 0.0, 1.0, 0.5, 'f', 2);
        // Iter is right-justified in width 4 then the literal 'r', so
        // `"   7r"` is the prefix.
        assert!(row.starts_with("   7r"), "row = {row:?}");
    }

    #[test]
    fn regu_field_dashes_when_zero() {
        let out = RestoIterationOutput::new();
        let row = out.format_row_explicit(0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, ' ', 0);
        // Width-6 right-aligned dash.
        assert!(row.contains("     -"), "row = {row:?}");
    }

    #[test]
    fn regu_field_logs_value_when_nonzero() {
        let out = RestoIterationOutput::new();
        let row = out.format_row_explicit(0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-3, 1.0, 1.0, ' ', 0);
        // log10(1e-3) = -3.0 → "  -3.0" (width 6).
        assert!(row.contains("  -3.0"), "row = {row:?}");
    }

    #[test]
    fn alpha_char_appears_immediately_before_ls_field() {
        let out = RestoIterationOutput::new();
        let row = out.format_row_explicit(0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 'h', 12);
        // alpha_pr (8) + alpha_char ('h') + ls (3 right-aligned).
        assert!(row.ends_with("h 12"), "row = {row:?}");
    }

    // The adapter's `IterationOutput::format_row` impl just plumbs
    // numeric fields from the inner data/CQ into
    // `RestoIterationOutput::format_row_explicit`, which the
    // `format_row_explicit` tests above already cover field-by-field.
    // End-to-end coverage of the adapter through a live inner IPM is
    // provided by the `restoration_triggers` integration test.
}
