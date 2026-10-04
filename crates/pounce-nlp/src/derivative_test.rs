//! `derivative_test` — check a TNLP's analytic derivatives against
//! finite differences before the solve. Port of upstream Ipopt's
//! `TNLPAdapter::CheckDerivatives` (`IpTNLPAdapter.cpp`).
//!
//! # Why this exists
//!
//! Every `derivative_test*` option was registered and none was ever
//! read, so `derivative_test=first-order` ran no test and printed
//! nothing (gh#483 follow-up). That is the worst shape an unimplemented
//! option can take: a *checker* that silently checks nothing reports
//! success by omission. A user with a hand-written `eval_grad_f` turns
//! it on, sees no complaints, and concludes the gradient is right.
//!
//! # What it does
//!
//! At the (bound-projected) starting point, each analytic derivative is
//! compared against a one-sided finite difference with a relative step
//! `derivative_test_perturbation · max(1, |xᵢ|)`. An entry is flagged
//! when
//!
//! ```text
//! |analytic − fd| > derivative_test_tol · max(1, |fd|)
//! ```
//!
//! * `first-order` — `eval_grad_f` and `eval_jac_g`.
//! * `second-order` — the above plus `eval_h`.
//! * `only-second-order` — `eval_h` alone.
//!
//! The Hessian is checked one multiplier block at a time: with
//! `obj_factor = 1, λ = 0` the analytic `eval_h` must match differences
//! of `eval_grad_f`; with `obj_factor = 0, λ = e_j` it must match
//! differences of row `j` of `eval_jac_g`.
//!
//! Two checks upstream does not make are included, because both catch a
//! real and otherwise-invisible class of bug:
//!
//! * A Jacobian or Hessian entry that the finite difference says is
//!   nonzero but the **sparsity structure omits**. A missing structural
//!   entry is not a wrong number, it is a derivative the solver can
//!   never see, and no value-by-value comparison finds it.
//! * A perturbation that would leave the variable's box is taken
//!   *downward* instead (with the difference negated), so a model whose
//!   functions are undefined outside their bounds — `sqrt`, `log`,
//!   `1/x` — is not evaluated out of its domain by the checker itself.
//!
//! The report is advisory: like upstream, a suspicious derivative is
//! printed but does not stop the solve. It is emitted on **stderr**, so
//! it survives `print_level=0` and never pollutes `--json-output`'s
//! stdout.

use crate::tnlp::{BoundsInfo, IndexStyle, SparsityRequest, StartingPoint, TNLP};
use pounce_common::types::{Index, Number};

/// `derivative_test` (default `none`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DerivativeTest {
    #[default]
    None,
    FirstOrder,
    SecondOrder,
    OnlySecondOrder,
}

impl DerivativeTest {
    /// Parse the registered option string; unknown values are `None`,
    /// matching the registry's own default.
    pub fn from_option(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "first-order" => Self::FirstOrder,
            "second-order" => Self::SecondOrder,
            "only-second-order" => Self::OnlySecondOrder,
            _ => Self::None,
        }
    }

    fn checks_first(self) -> bool {
        matches!(self, Self::FirstOrder | Self::SecondOrder)
    }

    fn checks_second(self) -> bool {
        matches!(self, Self::SecondOrder | Self::OnlySecondOrder)
    }
}

/// The registered `derivative_test*` knobs, resolved.
#[derive(Debug, Clone, Copy)]
pub struct DerivativeTestOptions {
    pub mode: DerivativeTest,
    /// `derivative_test_perturbation` (1e-8). Relative: the step on
    /// `xᵢ` is `perturbation · max(1, |xᵢ|)`.
    pub perturbation: Number,
    /// `derivative_test_tol` (1e-4). Relative deviation above which an
    /// entry is flagged.
    pub tol: Number,
    /// `derivative_test_first_index` (-2 = everything). For the
    /// first-order test this is the first **variable**; for the
    /// second-order test the first **constraint**, where `-1` is the
    /// objective's Hessian.
    pub first_index: Index,
    /// `derivative_test_print_all` — report every entry, not just the
    /// suspicious ones.
    pub print_all: bool,
}

impl Default for DerivativeTestOptions {
    fn default() -> Self {
        Self {
            mode: DerivativeTest::None,
            perturbation: 1e-8,
            tol: 1e-4,
            first_index: -2,
            print_all: false,
        }
    }
}

/// Cap on the flagged entries a report carries (gh#990 item 3). A model with a
/// wrong derivative flags thousands of entries; the count (`suspicious`) is
/// exact, the list is the first this many.
pub const MAX_FLAGGED_ENTRIES: usize = 200;

/// One flagged comparison, in a form a program can read (gh#990 item 3).
#[derive(Debug, Clone, PartialEq)]
pub struct DerivativeEntry {
    /// `gradient`, `jacobian`, `hessian`, `jacobian_structure` or
    /// `hessian_structure` (a finite difference is nonzero where the declared
    /// sparsity says the entry does not exist).
    pub kind: &'static str,
    /// Hessian block: `obj` or `g[<row>]`. Empty otherwise.
    pub block: String,
    /// Row index (constraint row, or Hessian row); `-1` for a gradient entry.
    pub row: i64,
    /// Column index (variable).
    pub col: i64,
    /// The model's derivative; `None` for a structural miss.
    pub analytic: Option<Number>,
    pub finite_difference: Number,
    /// `|analytic - fd| / max(1, |fd|)`, the quantity compared with the
    /// tolerance; `None` for a structural miss.
    pub relative_error: Option<Number>,
}

/// The machine-readable result of a derivative check, carried on the solve
/// statistics, `info["derivative_check"]` and the report's
/// `statistics.derivative_check`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DerivativeCheckSummary {
    /// `first-order`, `second-order` or `only-second-order`.
    pub mode: String,
    pub tolerance: Number,
    pub perturbation: Number,
    pub checked: usize,
    pub suspicious: usize,
    pub missing_structure: usize,
    pub evaluations: usize,
    /// `true` when nothing was flagged *and* something was compared.
    pub clean: bool,
    /// Largest relative error over every compared entry of each kind; `None`
    /// when no entry of that kind was compared.
    pub max_rel_error_gradient: Option<Number>,
    pub max_rel_error_jacobian: Option<Number>,
    pub max_rel_error_hessian: Option<Number>,
    /// The flagged entries (at most [`MAX_FLAGGED_ENTRIES`]).
    pub flagged: Vec<DerivativeEntry>,
}

/// Outcome of a run. `lines` is the formatted report, ready to print.
#[derive(Debug, Default)]
pub struct DerivativeTestReport {
    /// Machine-readable counterpart of `lines` (gh#990 item 3).
    pub summary: DerivativeCheckSummary,
    /// Entries compared against a finite difference.
    pub checked: usize,
    /// Entries whose deviation exceeded `derivative_test_tol`.
    pub suspicious: usize,
    /// Structurally-absent entries whose finite difference is nonzero.
    pub missing_structure: usize,
    /// TNLP evaluations the test cost, so "(slow!)" is quantified.
    pub evaluations: usize,
    pub lines: Vec<String>,
}

impl DerivativeTestReport {
    fn note_missing(&mut self, kind: &'static str, block: &str, row: i64, col: i64, fd: Number) {
        if self.summary.flagged.len() < MAX_FLAGGED_ENTRIES {
            self.summary.flagged.push(DerivativeEntry {
                kind,
                block: block.to_string(),
                row,
                col,
                analytic: None,
                finite_difference: fd,
                relative_error: None,
            });
        }
    }

    /// True when nothing looked wrong.
    pub fn clean(&self) -> bool {
        self.suspicious == 0 && self.missing_structure == 0
    }
}

/// One analytic-vs-finite-difference comparison.
struct Comparison {
    label: String,
    analytic: Number,
    fd: Number,
    kind: &'static str,
    block: String,
    row: i64,
    col: i64,
}

impl Comparison {
    fn deviation(&self) -> Number {
        (self.analytic - self.fd).abs()
    }

    /// Upstream's relative test: the tolerance scales with the finite
    /// difference, so a derivative of `1e6` is not flagged for an
    /// absolute error a derivative of `1e-6` would be.
    fn suspicious(&self, tol: Number) -> bool {
        self.deviation() > tol * self.fd.abs().max(1.0)
    }

    fn format(&self, flagged: bool) -> String {
        let rel = self.deviation() / self.fd.abs().max(1.0);
        format!(
            "{} {} = {:>23.16e}    ~ {:>23.16e}  [{:>10.3e}]",
            if flagged { "*" } else { " " },
            self.label,
            self.analytic,
            self.fd,
            rel
        )
    }
}

/// Working state: the TNLP's dimensions and the point being tested.
struct Fixture {
    n: usize,
    m: usize,
    x: Vec<Number>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    jac_rows: Vec<Index>,
    jac_cols: Vec<Index>,
    h_rows: Vec<Index>,
    h_cols: Vec<Index>,
}

impl Fixture {
    /// The perturbed point for variable `i`, and the signed step taken.
    ///
    /// The step is relative (`perturbation · max(1, |xᵢ|)`) and is taken
    /// downward when going up would leave the box — a `sqrt(x)` model
    /// must not be evaluated outside its domain by its own derivative
    /// checker. Returns `None` when the variable is fixed, which has no
    /// usable direction and no derivative worth checking.
    fn perturb(&self, i: usize, perturbation: Number) -> Option<(Vec<Number>, Number)> {
        let xi = self.x[i];
        let step = perturbation * xi.abs().max(1.0);
        let up = xi + step;
        let down = xi - step;
        let signed = if up <= self.x_u[i] {
            step
        } else if down >= self.x_l[i] {
            -step
        } else {
            return None;
        };
        let mut xp = self.x.clone();
        xp[i] = xi + signed;
        Some((xp, signed))
    }
}

/// Run the derivative test. `None` when the mode is `none` or the TNLP
/// declines to describe itself; otherwise a report to print.
pub fn run(tnlp: &mut dyn TNLP, opts: &DerivativeTestOptions) -> Option<DerivativeTestReport> {
    if matches!(opts.mode, DerivativeTest::None) {
        return None;
    }
    let info = tnlp.get_nlp_info()?;
    let (n, m) = (info.n as usize, info.m as usize);
    let style_offset: Index = match info.index_style {
        IndexStyle::C => 0,
        IndexStyle::Fortran => 1,
    };

    // Starting point, projected into the box: the solver's own start is
    // projected too, so testing anywhere else would check derivatives at
    // a point the solve never visits — and an unprojected start can sit
    // outside a function's domain.
    let (mut x_l, mut x_u) = (vec![0.0; n], vec![0.0; n]);
    let (mut g_l, mut g_u) = (vec![0.0; m], vec![0.0; m]);
    if !tnlp.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return None;
    }
    let mut x = vec![0.0; n];
    let (mut z_l, mut z_u, mut lambda) = (vec![0.0; n], vec![0.0; n], vec![0.0; m]);
    if !tnlp.get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: false,
        z_l: &mut z_l,
        z_u: &mut z_u,
        init_lambda: false,
        lambda: &mut lambda,
    }) {
        return None;
    }
    for i in 0..n {
        x[i] = x[i].clamp(x_l[i], x_u[i]);
    }

    let nnz_jac = info.nnz_jac_g as usize;
    let nnz_h = info.nnz_h_lag as usize;
    let mut jac_rows = vec![0 as Index; nnz_jac];
    let mut jac_cols = vec![0 as Index; nnz_jac];
    if nnz_jac > 0
        && !tnlp.eval_jac_g(
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut jac_rows,
                jcol: &mut jac_cols,
            },
        )
    {
        return None;
    }
    let mut h_rows = vec![0 as Index; nnz_h];
    let mut h_cols = vec![0 as Index; nnz_h];
    if nnz_h > 0
        && !tnlp.eval_h(
            None,
            true,
            1.0,
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut h_rows,
                jcol: &mut h_cols,
            },
        )
    {
        return None;
    }
    for v in jac_rows.iter_mut().chain(jac_cols.iter_mut()) {
        *v -= style_offset;
    }
    for v in h_rows.iter_mut().chain(h_cols.iter_mut()) {
        *v -= style_offset;
    }

    let fx = Fixture {
        n,
        m,
        x,
        x_l,
        x_u,
        jac_rows,
        jac_cols,
        h_rows,
        h_cols,
    };

    let mut report = DerivativeTestReport::default();
    report.lines.push(format!(
        "Derivative checker: {} at the starting point \
         (perturbation {:.1e}, tolerance {:.1e}).",
        match opts.mode {
            DerivativeTest::FirstOrder => "first derivatives",
            DerivativeTest::SecondOrder => "first and second derivatives",
            DerivativeTest::OnlySecondOrder => "second derivatives",
            DerivativeTest::None => unreachable!("handled above"),
        },
        opts.perturbation,
        opts.tol,
    ));

    if opts.mode.checks_first() {
        check_first_order(tnlp, &fx, opts, &mut report);
    }
    if opts.mode.checks_second() {
        check_second_order(tnlp, &fx, opts, &mut report);
    }

    let summary = if report.checked == 0 {
        // "No suspicious derivatives found" would be true and useless
        // here: nothing was compared. Every variable is fixed (or the
        // requested `derivative_test_first_index` is past the last one),
        // so there was no direction to difference along. Saying "clean"
        // would be this checker committing the defect it exists to
        // remove — silence reading as a pass.
        "No derivatives could be checked: every variable in range is fixed \
         (or derivative_test_first_index is past the last one), so there is \
         no direction to take a finite difference along."
            .to_string()
    } else if report.clean() {
        format!(
            "No suspicious derivatives found ({} entries checked, \
             {} evaluations).",
            report.checked, report.evaluations
        )
    } else {
        format!(
            "{} suspicious derivative(s) and {} missing sparsity entrie(s) \
             out of {} checked ({} evaluations). Entries marked `*` above \
             deviate by more than derivative_test_tol.",
            report.suspicious, report.missing_structure, report.checked, report.evaluations
        )
    };
    report.lines.push(summary);
    report.summary.mode = match opts.mode {
        DerivativeTest::FirstOrder => "first-order",
        DerivativeTest::SecondOrder => "second-order",
        DerivativeTest::OnlySecondOrder => "only-second-order",
        DerivativeTest::None => unreachable!("handled above"),
    }
    .to_string();
    report.summary.tolerance = opts.tol;
    report.summary.perturbation = opts.perturbation;
    report.summary.checked = report.checked;
    report.summary.suspicious = report.suspicious;
    report.summary.missing_structure = report.missing_structure;
    report.summary.evaluations = report.evaluations;
    report.summary.clean = report.checked > 0 && report.clean();
    Some(report)
}

/// `eval_grad_f` and `eval_jac_g` against forward differences of
/// `eval_f` and `eval_g`.
fn check_first_order(
    tnlp: &mut dyn TNLP,
    fx: &Fixture,
    opts: &DerivativeTestOptions,
    report: &mut DerivativeTestReport,
) {
    let (n, m) = (fx.n, fx.m);
    let Some(f0) = tnlp.eval_f(&fx.x, true) else {
        return;
    };
    let mut grad_f = vec![0.0; n];
    tnlp.eval_grad_f(&fx.x, false, &mut grad_f);
    let mut g0 = vec![0.0; m];
    if m > 0 {
        tnlp.eval_g(&fx.x, false, &mut g0);
    }
    let mut jac = vec![0.0; fx.jac_rows.len()];
    if !fx.jac_rows.is_empty() {
        tnlp.eval_jac_g(
            Some(&fx.x),
            false,
            SparsityRequest::Values { values: &mut jac },
        );
    }
    report.evaluations += 2 + usize::from(m > 0) * 2;

    // Sparse column lookup: (row, col) -> position in the triplet.
    let mut entry_at = std::collections::HashMap::new();
    for (k, (&r, &c)) in fx.jac_rows.iter().zip(fx.jac_cols.iter()).enumerate() {
        entry_at.insert((r as usize, c as usize), k);
    }

    // `derivative_test_first_index` selects the first *variable* here.
    let start = if opts.first_index < 0 {
        0
    } else {
        (opts.first_index as usize).min(n)
    };
    let mut g_pert = vec![0.0; m];
    for i in start..n {
        let Some((xp, step)) = fx.perturb(i, opts.perturbation) else {
            continue;
        };
        let Some(f_pert) = tnlp.eval_f(&xp, true) else {
            continue;
        };
        report.evaluations += 1;
        push(
            report,
            opts,
            Comparison {
                label: format!("grad_f[{i:5}]      "),
                analytic: grad_f[i],
                fd: (f_pert - f0) / step,
                kind: "gradient",
                block: String::new(),
                row: -1,
                col: i as i64,
            },
        );
        if m == 0 {
            continue;
        }
        tnlp.eval_g(&xp, false, &mut g_pert);
        report.evaluations += 1;
        for row in 0..m {
            let fd = (g_pert[row] - g0[row]) / step;
            match entry_at.get(&(row, i)) {
                Some(&k) => push(
                    report,
                    opts,
                    Comparison {
                        label: format!("jac_g [{row:5},{i:5}]"),
                        analytic: jac[k],
                        fd,
                        kind: "jacobian",
                        block: String::new(),
                        row: row as i64,
                        col: i as i64,
                    },
                ),
                None => {
                    // Not in the sparsity structure, so the solver can
                    // never see it: a nonzero here is a *structural*
                    // error, invisible to any value comparison.
                    if fd.abs() > opts.tol {
                        report.missing_structure += 1;
                        report.note_missing("jacobian_structure", "", row as i64, i as i64, fd);
                        report.lines.push(format!(
                            "! jac_g [{row:5},{i:5}] is not in the sparsity \
                             structure, but its finite difference is \
                             {fd:.6e}"
                        ));
                    }
                }
            }
        }
    }
}

/// `eval_h` against differences of `eval_grad_f` (objective block) and
/// of each `eval_jac_g` row (constraint blocks).
fn check_second_order(
    tnlp: &mut dyn TNLP,
    fx: &Fixture,
    opts: &DerivativeTestOptions,
    report: &mut DerivativeTestReport,
) {
    let (n, m) = (fx.n, fx.m);
    if fx.h_rows.is_empty() {
        report
            .lines
            .push("eval_h declares no nonzeros; skipping the second-order test.".to_string());
        return;
    }
    let mut entry_at = std::collections::HashMap::new();
    for (k, (&r, &c)) in fx.h_rows.iter().zip(fx.h_cols.iter()).enumerate() {
        // The Hessian triplet is a lower triangle; normalize so a
        // lookup by either ordering finds it.
        let (r, c) = (r as usize, c as usize);
        entry_at.insert((r.max(c), r.min(c)), k);
    }

    // `derivative_test_first_index` selects the first *constraint*
    // here, with -1 meaning the objective's Hessian.
    let start: i64 = if opts.first_index < -1 {
        -1
    } else {
        opts.first_index as i64
    };
    let mut h_vals = vec![0.0; fx.h_rows.len()];
    let mut base = vec![0.0; n];
    let mut pert = vec![0.0; n];
    let mut jac0 = vec![0.0; fx.jac_rows.len()];
    let mut jac_p = vec![0.0; fx.jac_rows.len()];

    for block in start..(m as i64) {
        let is_obj = block < 0;
        let mut lambda = vec![0.0; m];
        let obj_factor = if is_obj {
            1.0
        } else {
            lambda[block as usize] = 1.0;
            0.0
        };
        if !tnlp.eval_h(
            Some(&fx.x),
            true,
            obj_factor,
            Some(&lambda),
            true,
            SparsityRequest::Values {
                values: &mut h_vals,
            },
        ) {
            continue;
        }
        report.evaluations += 1;

        // Baseline first derivative of whichever function this block
        // belongs to.
        if is_obj {
            tnlp.eval_grad_f(&fx.x, true, &mut base);
        } else {
            tnlp.eval_jac_g(
                Some(&fx.x),
                true,
                SparsityRequest::Values { values: &mut jac0 },
            );
            row_of_jacobian(&fx.jac_rows, &fx.jac_cols, &jac0, block as usize, &mut base);
        }
        report.evaluations += 1;

        let name = if is_obj {
            "obj".to_string()
        } else {
            format!("g[{block}]")
        };
        for i in 0..n {
            let Some((xp, step)) = fx.perturb(i, opts.perturbation) else {
                continue;
            };
            if is_obj {
                tnlp.eval_grad_f(&xp, true, &mut pert);
            } else {
                tnlp.eval_jac_g(
                    Some(&xp),
                    true,
                    SparsityRequest::Values { values: &mut jac_p },
                );
                row_of_jacobian(
                    &fx.jac_rows,
                    &fx.jac_cols,
                    &jac_p,
                    block as usize,
                    &mut pert,
                );
            }
            report.evaluations += 1;
            // Column `i` of this block's Hessian. Only the lower
            // triangle is stored, so compare rows `j >= i`.
            for j in i..n {
                let fd = (pert[j] - base[j]) / step;
                match entry_at.get(&(j, i)) {
                    Some(&k) => push(
                        report,
                        opts,
                        Comparison {
                            label: format!("h_{name}[{j:5},{i:5}]"),
                            analytic: h_vals[k],
                            fd,
                            kind: "hessian",
                            block: name.clone(),
                            row: j as i64,
                            col: i as i64,
                        },
                    ),
                    None => {
                        if fd.abs() > opts.tol {
                            report.missing_structure += 1;
                            report.note_missing("hessian_structure", &name, j as i64, i as i64, fd);
                            report.lines.push(format!(
                                "! h_{name}[{j:5},{i:5}] is not in the Hessian \
                                 sparsity structure, but its finite difference \
                                 is {fd:.6e}"
                            ));
                        }
                    }
                }
            }
        }
    }
}

/// Scatter row `row` of a sparse Jacobian triplet into a dense vector.
fn row_of_jacobian(
    rows: &[Index],
    cols: &[Index],
    values: &[Number],
    row: usize,
    out: &mut [Number],
) {
    out.fill(0.0);
    for (k, (&r, &c)) in rows.iter().zip(cols.iter()).enumerate() {
        if r as usize == row {
            out[c as usize] = values[k];
        }
    }
}

fn push(report: &mut DerivativeTestReport, opts: &DerivativeTestOptions, cmp: Comparison) {
    report.checked += 1;
    let flagged = cmp.suspicious(opts.tol);
    let rel = cmp.deviation() / cmp.fd.abs().max(1.0);
    if rel.is_finite() || rel.is_nan() {
        let slot = match cmp.kind {
            "gradient" => &mut report.summary.max_rel_error_gradient,
            "jacobian" => &mut report.summary.max_rel_error_jacobian,
            _ => &mut report.summary.max_rel_error_hessian,
        };
        // A NaN analytic value is the worst error there is; keep it visible
        // rather than letting `max` swallow it.
        *slot = Some(match *slot {
            Some(m) if m.is_nan() || (!rel.is_nan() && m >= rel) => m,
            _ => rel,
        });
    }
    if flagged {
        report.suspicious += 1;
        if report.summary.flagged.len() < MAX_FLAGGED_ENTRIES {
            report.summary.flagged.push(DerivativeEntry {
                kind: cmp.kind,
                block: cmp.block.clone(),
                row: cmp.row,
                col: cmp.col,
                analytic: Some(cmp.analytic),
                finite_difference: cmp.fd,
                relative_error: Some(rel),
            });
        }
    }
    if flagged || opts.print_all {
        report.lines.push(cmp.format(flagged));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tnlp::{IpoptCq, IpoptData, NlpInfo, Solution};

    /// `min x0² + 3·x0·x1  s.t.  x0·x1 = 1`, with each derivative
    /// independently corruptible so the checker can be shown to catch
    /// exactly the thing that is wrong and nothing else.
    #[derive(Default)]
    struct Quad {
        bad_grad: bool,
        bad_jac: bool,
        bad_hess: bool,
        drop_jac_entry: bool,
    }

    impl TNLP for Quad {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 1,
                nnz_jac_g: if self.drop_jac_entry { 1 } else { 2 },
                nnz_h_lag: 2,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[-10.0, -10.0]);
            b.x_u.copy_from_slice(&[10.0, 10.0]);
            b.g_l[0] = 1.0;
            b.g_u[0] = 1.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.copy_from_slice(&[2.0, 3.0]);
            true
        }
        fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
            Some(x[0] * x[0] + 3.0 * x[0] * x[1])
        }
        fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
            g[0] = 2.0 * x[0] + 3.0 * x[1];
            g[1] = 3.0 * x[0];
            if self.bad_grad {
                g[1] += 0.5; // a constant offset a solver would chase forever
            }
            true
        }
        fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] * x[1];
            true
        }
        fn eval_jac_g(
            &mut self,
            x: Option<&[Number]>,
            _n: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                    if !self.drop_jac_entry {
                        irow[1] = 0;
                        jcol[1] = 1;
                    }
                }
                SparsityRequest::Values { values } => {
                    let x = x.expect("values call needs x");
                    values[0] = x[1];
                    if !self.drop_jac_entry {
                        values[1] = x[0];
                    }
                    if self.bad_jac {
                        values[0] *= 2.0;
                    }
                }
            }
            true
        }
        fn eval_h(
            &mut self,
            _x: Option<&[Number]>,
            _n: bool,
            obj_factor: Number,
            lambda: Option<&[Number]>,
            _nl: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    // lower triangle: (0,0) and (1,0)
                    irow[0] = 0;
                    jcol[0] = 0;
                    irow[1] = 1;
                    jcol[1] = 0;
                }
                SparsityRequest::Values { values } => {
                    let l = lambda.map(|l| l[0]).unwrap_or(0.0);
                    values[0] = 2.0 * obj_factor;
                    values[1] = 3.0 * obj_factor + l;
                    if self.bad_hess {
                        values[1] += 1.0;
                    }
                }
            }
            true
        }
        fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    fn opts(mode: DerivativeTest) -> DerivativeTestOptions {
        DerivativeTestOptions {
            mode,
            // Loose enough that a correct derivative is never flagged by
            // one-sided-difference truncation, tight enough to catch the
            // O(1) corruptions above.
            perturbation: 1e-7,
            tol: 1e-4,
            ..Default::default()
        }
    }

    /// A model whose variables are all fixed gives the checker nothing
    /// to difference. Reporting "no suspicious derivatives found" there
    /// would be the checker committing the defect it exists to remove:
    /// a clean bill of health for a check that never ran.
    #[test]
    fn checking_nothing_does_not_report_clean() {
        #[derive(Default)]
        struct AllFixed;
        impl TNLP for AllFixed {
            fn get_nlp_info(&mut self) -> Option<NlpInfo> {
                Some(NlpInfo {
                    n: 1,
                    m: 0,
                    nnz_jac_g: 0,
                    nnz_h_lag: 1,
                    index_style: IndexStyle::C,
                })
            }
            fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
                b.x_l[0] = 3.0;
                b.x_u[0] = 3.0;
                true
            }
            fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
                sp.x[0] = 3.0;
                true
            }
            fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
                Some(x[0] * x[0])
            }
            fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
                g[0] = 2.0 * x[0];
                true
            }
            fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
                true
            }
            fn eval_jac_g(
                &mut self,
                _x: Option<&[Number]>,
                _n: bool,
                _m: SparsityRequest<'_>,
            ) -> bool {
                true
            }
            fn eval_h(
                &mut self,
                _x: Option<&[Number]>,
                _n: bool,
                _o: Number,
                _l: Option<&[Number]>,
                _nl: bool,
                mode: SparsityRequest<'_>,
            ) -> bool {
                if let SparsityRequest::Structure { irow, jcol } = mode {
                    irow[0] = 0;
                    jcol[0] = 0;
                }
                true
            }
            fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
        }
        let mut t = AllFixed;
        let r = run(&mut t, &opts(DerivativeTest::FirstOrder)).expect("ran");
        assert_eq!(r.checked, 0);
        let summary = r.lines.last().expect("a summary line");
        assert!(
            summary.contains("No derivatives could be checked"),
            "must not read as a pass; got: {summary}"
        );
        assert!(
            !summary.contains("No suspicious derivatives found"),
            "got: {summary}"
        );
    }

    #[test]
    fn none_runs_nothing() {
        let mut t = Quad::default();
        assert!(run(&mut t, &opts(DerivativeTest::None)).is_none());
    }

    #[test]
    fn correct_derivatives_are_clean() {
        let mut t = Quad::default();
        let r = run(&mut t, &opts(DerivativeTest::SecondOrder)).expect("ran");
        assert!(
            r.clean(),
            "correct derivatives must not be flagged:\n{}",
            r.lines.join("\n")
        );
        assert!(r.checked > 0, "the test must actually compare something");
    }

    #[test]
    fn a_wrong_objective_gradient_is_caught() {
        let mut t = Quad {
            bad_grad: true,
            ..Default::default()
        };
        let r = run(&mut t, &opts(DerivativeTest::FirstOrder)).expect("ran");
        assert_eq!(r.suspicious, 1, "report:\n{}", r.lines.join("\n"));
        assert!(r.lines.iter().any(|l| l.contains("grad_f[    1]")));
    }

    /// gh#990 item 3: the verdict is machine-readable, not only lines.
    #[test]
    fn the_summary_names_the_flagged_entry_and_the_worst_error_per_kind() {
        let mut t = Quad {
            bad_grad: true,
            ..Default::default()
        };
        let r = run(&mut t, &opts(DerivativeTest::SecondOrder)).expect("ran");
        let s = &r.summary;
        assert_eq!(s.mode, "second-order");
        assert_eq!((s.checked, s.suspicious), (r.checked, 1));
        assert!(!s.clean);
        assert_eq!(s.flagged.len(), 1);
        let e = &s.flagged[0];
        assert_eq!((e.kind, e.col, e.row), ("gradient", 1, -1));
        assert_eq!(s.max_rel_error_gradient, e.relative_error);
        assert!(s.max_rel_error_jacobian.is_some_and(|v| v < 1e-4));
        assert!(s.max_rel_error_hessian.is_some_and(|v| v < 1e-4));

        let mut ok = Quad::default();
        let r = run(&mut ok, &opts(DerivativeTest::FirstOrder)).expect("ran");
        assert!(r.summary.clean && r.summary.flagged.is_empty());
        assert!(r.summary.max_rel_error_hessian.is_none());
    }

    #[test]
    fn a_wrong_jacobian_entry_is_caught() {
        let mut t = Quad {
            bad_jac: true,
            ..Default::default()
        };
        let r = run(&mut t, &opts(DerivativeTest::FirstOrder)).expect("ran");
        assert_eq!(r.suspicious, 1, "report:\n{}", r.lines.join("\n"));
        assert!(r.lines.iter().any(|l| l.contains("jac_g [    0,    0]")));
    }

    #[test]
    fn a_wrong_hessian_entry_is_caught() {
        let mut t = Quad {
            bad_hess: true,
            ..Default::default()
        };
        let r = run(&mut t, &opts(DerivativeTest::OnlySecondOrder)).expect("ran");
        // The corrupted entry belongs to both the objective block and
        // every constraint block that shares it.
        assert!(r.suspicious >= 1, "report:\n{}", r.lines.join("\n"));
        assert!(r.lines.iter().any(|l| l.contains("h_obj[")));
    }

    /// The check upstream does not make: a derivative that exists but is
    /// missing from the sparsity structure. No value comparison can find
    /// it, because there is no value to compare.
    #[test]
    fn a_missing_sparsity_entry_is_caught() {
        let mut t = Quad {
            drop_jac_entry: true,
            ..Default::default()
        };
        let r = run(&mut t, &opts(DerivativeTest::FirstOrder)).expect("ran");
        assert_eq!(r.missing_structure, 1, "report:\n{}", r.lines.join("\n"));
        assert!(!r.clean());
        assert!(
            r.lines
                .iter()
                .any(|l| l.contains("not in the sparsity structure")),
        );
    }

    /// `first-order` must not evaluate `eval_h` at all — a model with a
    /// deliberately broken Hessian still passes the first-order test.
    #[test]
    fn first_order_ignores_the_hessian() {
        let mut t = Quad {
            bad_hess: true,
            ..Default::default()
        };
        let r = run(&mut t, &opts(DerivativeTest::FirstOrder)).expect("ran");
        assert!(r.clean(), "report:\n{}", r.lines.join("\n"));
    }

    /// `only-second-order` skips the first-order pass, so a broken
    /// gradient is not reported by it.
    #[test]
    fn only_second_order_skips_the_first_order_pass() {
        let mut t = Quad {
            bad_grad: true,
            ..Default::default()
        };
        let r = run(&mut t, &opts(DerivativeTest::OnlySecondOrder)).expect("ran");
        assert!(
            !r.lines.iter().any(|l| l.contains("grad_f")),
            "report:\n{}",
            r.lines.join("\n")
        );
    }

    #[test]
    fn print_all_reports_clean_entries_too() {
        let mut t = Quad::default();
        let quiet = run(&mut t, &opts(DerivativeTest::FirstOrder)).expect("ran");
        let mut t = Quad::default();
        let loud = run(
            &mut t,
            &DerivativeTestOptions {
                print_all: true,
                ..opts(DerivativeTest::FirstOrder)
            },
        )
        .expect("ran");
        assert!(loud.lines.len() > quiet.lines.len());
        assert_eq!(loud.checked, quiet.checked);
        assert!(loud.clean() && quiet.clean());
    }

    /// `derivative_test_first_index` narrows the first-order sweep to
    /// variables at or after the given index.
    #[test]
    fn first_index_narrows_the_first_order_sweep() {
        let mut t = Quad {
            bad_grad: true, // the error is in grad_f[1]
            ..Default::default()
        };
        let r = run(
            &mut t,
            &DerivativeTestOptions {
                first_index: 1,
                ..opts(DerivativeTest::FirstOrder)
            },
        )
        .expect("ran");
        assert_eq!(r.suspicious, 1, "the error at index 1 is still in range");
        let mut t = Quad {
            bad_grad: true,
            ..Default::default()
        };
        let narrowed = run(
            &mut t,
            &DerivativeTestOptions {
                first_index: 2, // past the last variable: nothing to check
                ..opts(DerivativeTest::FirstOrder)
            },
        )
        .expect("ran");
        assert_eq!(narrowed.checked, 0);
    }

    /// A model whose functions are undefined outside their bounds must
    /// not be evaluated out of domain by its own checker: the step is
    /// taken downward when up would leave the box.
    #[derive(Default)]
    struct AtUpperBound;
    impl TNLP for AtUpperBound {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 0,
                nnz_jac_g: 0,
                nnz_h_lag: 1,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = 0.0;
            b.x_u[0] = 1.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 1.0; // pinned at the upper bound
            true
        }
        fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
            assert!(
                x[0] <= 1.0 && x[0] >= 0.0,
                "evaluated outside the declared bounds at {}",
                x[0]
            );
            Some((1.0 - x[0]).sqrt())
        }
        fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
            // d/dx sqrt(1-x) = -1/(2 sqrt(1-x)); at x = 1 exactly this is
            // singular, so the checker's downward step is what makes the
            // comparison possible at all.
            g[0] = -0.5 / (1.0 - x[0]).max(1e-300).sqrt();
            true
        }
        fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
            true
        }
        fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool {
            true
        }
        fn eval_h(
            &mut self,
            _x: Option<&[Number]>,
            _n: bool,
            _o: Number,
            _l: Option<&[Number]>,
            _nl: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                }
                SparsityRequest::Values { values } => values[0] = 0.0,
            }
            true
        }
        fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn the_perturbation_stays_inside_the_box() {
        let mut t = AtUpperBound;
        // The assertion lives in eval_f: stepping up from x = 1 would
        // trip it.
        let r = run(&mut t, &opts(DerivativeTest::FirstOrder)).expect("ran");
        assert_eq!(r.checked, 1);
    }
}
