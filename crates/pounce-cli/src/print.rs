//! Ipopt-style banner / problem-stats / final-summary printing for
//! the `pounce` CLI. Output is structured to match upstream Ipopt's
//! console layout closely enough that anyone familiar with `ipopt`
//! can spot at a glance whether POUNCE is converging similarly.

use crate::counting_tnlp::CountingTnlp;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{IndexStyle, NlpInfo, SparsityRequest, TNLP};
use pounce_nlp::tnlp_adapter::{FixedVarTreatment, TNLPAdapter};
use std::cell::RefCell;
use std::rc::Rc;

/// Same sentinel Ipopt uses for "no bound": ±1e19. Only referenced by the
/// unit tests now that `collect_stats` derives bounds from the adapter
/// classification rather than re-thresholding raw bounds itself.
#[cfg(test)]
const BOUND_INF: f64 = 1.0e19;

#[derive(Debug, Clone, Copy)]
pub struct ProblemStats {
    pub n: i32,
    pub m: i32,
    pub nnz_jac_eq: i32,
    pub nnz_jac_ineq: i32,
    pub nnz_h_lag: i32,
    pub var_lower_only: i32,
    pub var_upper_only: i32,
    pub var_both: i32,
    pub var_free: i32,
    pub n_eq: i32,
    pub n_ineq: i32,
    pub ineq_lower_only: i32,
    pub ineq_upper_only: i32,
    pub ineq_both: i32,
}

/// Gather everything the banner block needs, reported over the **reduced**
/// problem that the algorithm actually solves — i.e. after
/// `fixed_variable_treatment` removes fixed (`x_l == x_u`) variables under
/// `make_parameter`. This mirrors Ipopt, whose banner is computed from the
/// post-`IpTNLPAdapter` problem; computing it from the raw TNLP instead made
/// pounce over-report variables and bucket fixed vars as "lower and upper
/// bounds" (#140).
///
/// To stay byte-for-byte consistent with the solve, the counts are taken from
/// a throwaway [`TNLPAdapter`] built with the same options — reusing the exact
/// production classification (including the `make_parameter → relax_bounds`
/// auto-switch). The Jacobian / Hessian nnz are read from the raw structure and
/// filtered to drop entries in fixed-variable columns (the columns Ipopt
/// removes). Returns `None` if any required TNLP call fails.
pub fn collect_stats(
    tnlp: &Rc<RefCell<dyn TNLP>>,
    lo_inf: Number,
    up_inf: Number,
    fixed_treatment: FixedVarTreatment,
) -> Option<ProblemStats> {
    let adapter =
        TNLPAdapter::new_with_options(Rc::clone(tnlp), lo_inf, up_inf, fixed_treatment).ok()?;
    let cls = adapter.classification();
    let info: NlpInfo = *adapter.nlp_info();
    let n_full_x = cls.n_full_x as usize;
    let m = info.m as usize;
    let one_based = matches!(info.index_style, IndexStyle::Fortran);

    // --- Variable bound buckets over the reduced (non-fixed) variable set.
    // `x_l_map` / `x_u_map` hold positions in `x_var` that carry a finite
    // lower / upper bound. A fixed var under `make_parameter` is absent from
    // both (it was dropped from `x_var`); under `relax_bounds` it lands in
    // both — matching Ipopt's banner in either mode.
    let nv = cls.n_x_var() as usize;
    let mut has_l = vec![false; nv];
    let mut has_u = vec![false; nv];
    for &p in &cls.x_l_map {
        has_l[p as usize] = true;
    }
    for &p in &cls.x_u_map {
        has_u[p as usize] = true;
    }
    let (mut var_lower_only, mut var_upper_only, mut var_both, mut var_free) = (0, 0, 0, 0);
    for k in 0..nv {
        match (has_l[k], has_u[k]) {
            (true, true) => var_both += 1,
            (true, false) => var_lower_only += 1,
            (false, true) => var_upper_only += 1,
            (false, false) => var_free += 1,
        }
    }

    // --- Constraint counts / inequality bound buckets, straight from the
    // classification (equality = `c_map`, inequality = `d_map`).
    let n_eq = cls.n_c;
    let n_ineq = cls.n_d;
    let nd = cls.n_d as usize;
    let mut has_dl = vec![false; nd];
    let mut has_du = vec![false; nd];
    for &p in &cls.d_l_map {
        has_dl[p as usize] = true;
    }
    for &p in &cls.d_u_map {
        has_du[p as usize] = true;
    }
    let (mut ineq_lower_only, mut ineq_upper_only, mut ineq_both) = (0, 0, 0);
    for k in 0..nd {
        match (has_dl[k], has_du[k]) {
            (true, true) => ineq_both += 1,
            (true, false) => ineq_lower_only += 1,
            (false, true) => ineq_upper_only += 1,
            // A "free" inequality has no finite bound on either side (e.g. an
            // `.nl` range row left fully open). It is still counted in
            // `n_ineq`, so bucket it under "both" or the printed breakdown
            // won't sum to the total.
            (false, false) => ineq_both += 1,
        }
    }

    // Which raw rows are equality rows (for the Jacobian split).
    let mut row_is_eq = vec![false; m];
    for &r in &cls.c_map {
        row_is_eq[r as usize] = true;
    }

    // --- Jacobian split: read the raw structure once, drop fixed-variable
    // columns (the ones `make_parameter` removes), tally per-row.
    let nnz_total = info.nnz_jac_g as usize;
    let (mut nnz_jac_eq, mut nnz_jac_ineq) = (0, 0);
    if nnz_total > 0 && m > 0 {
        let mut irow = vec![0_i32; nnz_total];
        let mut jcol = vec![0_i32; nnz_total];
        let mut t = tnlp.borrow_mut();
        if t.eval_jac_g(
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        ) {
            for k in 0..nnz_total {
                let col = if one_based {
                    (jcol[k] - 1) as usize
                } else {
                    jcol[k] as usize
                };
                // Skip nonzeros in fixed-variable columns: `full_to_var[col]`
                // is `-1` for a dropped fixed var.
                if col >= n_full_x || cls.full_to_var[col] < 0 {
                    continue;
                }
                let row = if one_based {
                    (irow[k] - 1) as usize
                } else {
                    irow[k] as usize
                };
                if row < m && row_is_eq[row] {
                    nnz_jac_eq += 1;
                } else {
                    nnz_jac_ineq += 1;
                }
            }
        }
    }

    // --- Hessian nnz over the reduced problem: drop any entry touching a
    // fixed-variable row or column. If the TNLP supplies no Hessian
    // (`eval_h` → false), fall back to the raw count.
    let mut nnz_h_lag = info.nnz_h_lag;
    let nnz_h = info.nnz_h_lag as usize;
    if nnz_h > 0 {
        let mut irow = vec![0_i32; nnz_h];
        let mut jcol = vec![0_i32; nnz_h];
        let mut t = tnlp.borrow_mut();
        if t.eval_h(
            None,
            true,
            1.0,
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        ) {
            let mut kept = 0_i32;
            for k in 0..nnz_h {
                let r = if one_based {
                    (irow[k] - 1) as usize
                } else {
                    irow[k] as usize
                };
                let c = if one_based {
                    (jcol[k] - 1) as usize
                } else {
                    jcol[k] as usize
                };
                if r < n_full_x
                    && c < n_full_x
                    && cls.full_to_var[r] >= 0
                    && cls.full_to_var[c] >= 0
                {
                    kept += 1;
                }
            }
            nnz_h_lag = kept;
        }
    }

    Some(ProblemStats {
        n: cls.n_x_var(),
        m: info.m,
        nnz_jac_eq,
        nnz_jac_ineq,
        nnz_h_lag,
        var_lower_only,
        var_upper_only,
        var_both,
        var_free,
        n_eq,
        n_ineq,
        ineq_lower_only,
        ineq_upper_only,
        ineq_both,
    })
}

/// POUNCE wordmark in block letters, printed above the copyright banner.
const LOGO: [&str; 5] = [
    "####    ###   #   #  #   #   ####  #####",
    "#   #  #   #  #   #  ##  #  #      #    ",
    "####   #   #  #   #  # # #  #      #### ",
    "#      #   #  #   #  #  ##  #      #    ",
    "#       ###    ###   #   #   ####  #####",
];

/// Width of the copyright banner's asterisk rules — wide enough to span
/// the longest banner text line. The wordmark is centered against this,
/// and a matching rule is printed above it.
const BANNER_WIDTH: usize = 80;

/// Print the branded POUNCE ASCII wordmark, mimicking the project logo.
///
/// Block letters get a top-lit **steel** sheen (light silver → dark
/// steel down the rows); three diagonal **molten claw** slashes rake
/// upper-right → lower-left, glowing bright gold at the top into deep
/// red at the bottom — the brand logo's look. Emitted through
/// `anstream::stdout()`, which strips the ANSI when stdout is redirected
/// or `NO_COLOR` is set (non-TTY sinks get the plain text), with a
/// 256-color downgrade on non-truecolor terminals. The metallic letters
/// are tuned for a dark terminal background.
pub fn print_logo() {
    use std::io::Write as _;
    let width = LOGO
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(1)
        .max(2);
    let mut out = anstream::stdout();
    // Leading rule matching the copyright banner's width, then a blank
    // line, then the centered wordmark. The rule is left in the terminal's
    // default color (like the banner's own rules) so it stays distinct on
    // any background. `anstream` strips the styling when stdout isn't a TTY.
    let _ = writeln!(out, "{}", "*".repeat(BANNER_WIDTH));
    let _ = writeln!(out);
    let pad = " ".repeat(BANNER_WIDTH.saturating_sub(width) / 2);
    for row in logo_rows(true) {
        let _ = writeln!(out, "{pad}{row}");
    }
    let _ = writeln!(out);
}

/// Render the POUNCE wordmark as styled rows (one `String` per line):
/// steel-sheen letters with three molten claw slashes, in the project
/// palette. Emits ANSI styling only when `color`; otherwise plain
/// `#`/`/` block characters. Shared by the solve header ([`print_logo`])
/// and the interactive debugger's open banner (rendered to stderr).
pub fn logo_rows(color: bool) -> Vec<String> {
    use pounce_common::style::{downgrade, truecolor_enabled, ALPHA_HOT, BRIGHT_YEL, TIGER_ORANGE};

    fn lerp(a: u8, b: u8, t: f64) -> u8 {
        (a as f64 + (b as f64 - a as f64) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    }
    fn mix(a: anstyle::RgbColor, b: anstyle::RgbColor, t: f64) -> anstyle::RgbColor {
        anstyle::RgbColor(lerp(a.0, b.0, t), lerp(a.1, b.1, t), lerp(a.2, b.2, t))
    }
    // Steel sheen (top-lit): light silver at the top row → dark steel at
    // the bottom. Molten ramp: gold → tiger-orange → deep red top-to-bottom.
    const STEEL_HI: anstyle::RgbColor = anstyle::RgbColor(0xd2, 0xd6, 0xdc);
    const STEEL_LO: anstyle::RgbColor = anstyle::RgbColor(0x5c, 0x60, 0x68);

    let rows = LOGO.len();
    let width = LOGO
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(1)
        .max(2);
    let vfrac = |r: usize| {
        if rows <= 1 {
            0.0
        } else {
            r as f64 / (rows - 1) as f64
        }
    };
    // Molten color for a claw cell at row `r` (0 = top, hottest).
    let molten = |r: usize| {
        let t = vfrac(r);
        if t < 0.5 {
            mix(BRIGHT_YEL, TIGER_ORANGE, t / 0.5)
        } else {
            mix(TIGER_ORANGE, ALPHA_HOT, (t - 0.5) / 0.5)
        }
    };

    let mut grid: Vec<Vec<Option<(char, anstyle::RgbColor)>>> = vec![vec![None; width]; rows];
    for (r, line) in LOGO.iter().enumerate() {
        let steel = mix(STEEL_HI, STEEL_LO, vfrac(r));
        for (c, ch) in line.chars().enumerate() {
            if ch != ' ' {
                grid[r][c] = Some((ch, steel));
            }
        }
    }
    // Three parallel molten claw slashes, upper-right → lower-left (`/`).
    for &start in &[width / 4, width / 4 + 6, width / 4 + 12] {
        for r in 0..rows {
            let c = start + (rows - 1 - r);
            if c < width {
                grid[r][c] = Some(('/', molten(r)));
            }
        }
    }

    let truecolor = truecolor_enabled();
    grid.iter()
        .map(|row| {
            let mut rendered = String::new();
            for cell in row {
                match cell {
                    Some((ch, rgb)) if color => {
                        let style = anstyle::Style::new()
                            .bold()
                            .fg_color(Some(downgrade(*rgb, truecolor)));
                        rendered.push_str(&format!(
                            "{}{}{}",
                            style.render(),
                            ch,
                            style.render_reset()
                        ));
                    }
                    Some((ch, _)) => rendered.push(*ch),
                    None => rendered.push(' '),
                }
            }
            rendered.trim_end().to_string()
        })
        .collect()
}

pub fn print_banner(linear_solver: &str) {
    use std::io::IsTerminal as _;

    // OSC 8 hyperlink so supporting terminals make the URL clickable;
    // only emitted to a TTY so redirected output stays plain text.
    const URL: &str = "https://github.com/jkitchin/pounce";
    let link = if std::io::stdout().is_terminal() {
        format!("\x1b]8;;{URL}\x1b\\{URL}\x1b]8;;\x1b\\")
    } else {
        URL.to_string()
    };

    let rule = "*".repeat(BANNER_WIDTH);
    println!("{rule}");
    println!("This program contains POUNCE, a pure-Rust interior-point optimization solver");
    println!("for nonlinear, conic, and global problems (its NLP core is ported from Ipopt).");
    println!("Released under the Eclipse Public License (EPL) — drop-in compatible with Ipopt.");
    println!("         For more information visit {link}");
    println!("{rule}");
    println!();
    println!(
        "This is POUNCE version {}, running with linear solver {}.",
        env!("CARGO_PKG_VERSION"),
        linear_solver
    );
    println!();
}

pub fn print_problem_stats(s: &ProblemStats) {
    println!(
        "Number of nonzeros in equality constraint Jacobian...: {:>8}",
        s.nnz_jac_eq
    );
    println!(
        "Number of nonzeros in inequality constraint Jacobian.: {:>8}",
        s.nnz_jac_ineq
    );
    println!(
        "Number of nonzeros in Lagrangian Hessian.............: {:>8}",
        s.nnz_h_lag
    );
    println!();
    println!(
        "Total number of variables............................: {:>8}",
        s.n
    );
    println!(
        "                     variables with only lower bounds: {:>8}",
        s.var_lower_only
    );
    println!(
        "                variables with lower and upper bounds: {:>8}",
        s.var_both
    );
    println!(
        "                     variables with only upper bounds: {:>8}",
        s.var_upper_only
    );
    println!(
        "Total number of equality constraints.................: {:>8}",
        s.n_eq
    );
    println!(
        "Total number of inequality constraints...............: {:>8}",
        s.n_ineq
    );
    println!(
        "        inequality constraints with only lower bounds: {:>8}",
        s.ineq_lower_only
    );
    println!(
        "   inequality constraints with lower and upper bounds: {:>8}",
        s.ineq_both
    );
    println!(
        "        inequality constraints with only upper bounds: {:>8}",
        s.ineq_upper_only
    );
    println!();
}

pub fn print_summary(
    status: ApplicationReturnStatus,
    stats: &SolveStatistics,
    counters: &CountingTnlp,
) {
    println!();
    println!();
    println!("Number of Iterations....: {}", stats.iteration_count);
    println!();
    println!("                                   (scaled)                 (unscaled)");
    let row = |label: &str, scaled: f64, unscaled: f64| {
        println!(
            "{label}:   {}    {}",
            fmt_ipopt(scaled),
            fmt_ipopt(unscaled)
        );
    };
    row(
        "Objective...............",
        stats.final_scaled_objective,
        stats.final_objective,
    );
    row(
        "Dual infeasibility......",
        stats.final_dual_inf,
        stats.final_dual_inf,
    );
    row(
        "Constraint violation....",
        stats.final_constr_viol,
        stats.final_constr_viol,
    );
    row("Variable bound violation", 0.0, 0.0);
    row(
        "Complementarity.........",
        stats.final_compl,
        stats.final_compl,
    );
    row(
        "Overall NLP error.......",
        stats.final_kkt_error,
        stats.final_kkt_error,
    );
    println!();
    println!();
    println!(
        "Number of objective function evaluations             = {}",
        counters.n_obj.get()
    );
    println!(
        "Number of objective gradient evaluations             = {}",
        counters.n_grad_f.get()
    );
    println!(
        "Number of equality constraint evaluations            = {}",
        counters.n_g.get()
    );
    println!(
        "Number of inequality constraint evaluations          = {}",
        counters.n_g.get()
    );
    println!(
        "Number of equality constraint Jacobian evaluations   = {}",
        counters.n_jac_g.get()
    );
    println!(
        "Number of inequality constraint Jacobian evaluations = {}",
        counters.n_jac_g.get()
    );
    println!(
        "Number of Lagrangian Hessian evaluations             = {}",
        counters.n_h.get()
    );
    println!(
        "Total seconds in POUNCE                              = {:.3}",
        stats.total_wallclock_time_secs
    );
    println!();
    println!("EXIT: {}", status_message(status));
    println!();
    println!(
        "POUNCE {}: {}",
        env!("CARGO_PKG_VERSION"),
        status_message(status)
    );
}

/// Emit an Ipopt-style end-of-run summary for the dedicated convex
/// (LP / QP / conic) IPM path. That path otherwise prints only a compact
/// one-line result, so the `Number of Iterations....:` and
/// `Objective...............:` lines the general NLP path emits are missing.
/// Downstream consumers that parse Ipopt's summary block — notably the
/// benchmark harness's `extract_obj`/`extract_iters` in
/// `benchmarks/scripts/run_nl_bench.sh` — then see a null objective and zero
/// iterations even though the solve succeeded. This prints the same labelled
/// lines (objective + KKT residual rows) so those consumers capture the real
/// values. The convex solver reports a single (unscaled, user-sense) objective
/// and residuals, so the "(scaled)"/"(unscaled)" columns carry the same value.
pub fn print_convex_summary(
    iterations: usize,
    objective: f64,
    primal_inf: f64,
    dual_inf: f64,
    complementarity: f64,
    kkt_error: f64,
) {
    println!();
    println!();
    println!("Number of Iterations....: {iterations}");
    println!();
    println!("                                   (scaled)                 (unscaled)");
    let row = |label: &str, v: f64| {
        println!("{label}:   {}    {}", fmt_ipopt(v), fmt_ipopt(v));
    };
    row("Objective...............", objective);
    row("Dual infeasibility......", dual_inf);
    row("Constraint violation....", primal_inf);
    row("Variable bound violation", 0.0);
    row("Complementarity.........", complementarity);
    row("Overall NLP error.......", kkt_error);
    println!();
}

/// Format a number in Ipopt's scientific notation: 16-digit mantissa,
/// signed 2-digit exponent (e.g. `3.7952009505566139e+03`). Rust's
/// `{:.16e}` is close but emits a 1-digit exponent without leading
/// sign, which makes side-by-side diffs against `ipopt` output messy.
pub fn fmt_ipopt(v: f64) -> String {
    if v.is_nan() {
        return "nan".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{:.16e}", v);
    let Some(e_pos) = s.rfind('e') else {
        return s;
    };
    let (mantissa, exp_part) = s.split_at(e_pos);
    let exp_str = &exp_part[1..];
    let (sign, digits) = if let Some(rest) = exp_str.strip_prefix('-') {
        ('-', rest)
    } else if let Some(rest) = exp_str.strip_prefix('+') {
        ('+', rest)
    } else {
        ('+', exp_str)
    };
    let padded = if digits.len() < 2 {
        format!("0{digits}")
    } else {
        digits.to_string()
    };
    format!("{mantissa}e{sign}{padded}")
}

pub fn status_message(s: ApplicationReturnStatus) -> &'static str {
    match s {
        ApplicationReturnStatus::SolveSucceeded => "Optimal Solution Found.",
        ApplicationReturnStatus::SolvedToAcceptableLevel => "Solved To Acceptable Level.",
        ApplicationReturnStatus::InfeasibleProblemDetected => {
            "Converged to a point of local infeasibility. Problem may be infeasible."
        }
        ApplicationReturnStatus::SearchDirectionBecomesTooSmall => {
            "Search Direction is becoming Too Small."
        }
        ApplicationReturnStatus::DivergingIterates => {
            "Iterates diverging; problem might be unbounded."
        }
        ApplicationReturnStatus::UserRequestedStop => "Stopping optimization at user request.",
        ApplicationReturnStatus::FeasiblePointFound => "Feasible Point Found.",
        ApplicationReturnStatus::MaximumIterationsExceeded => {
            "Maximum Number of Iterations Exceeded."
        }
        ApplicationReturnStatus::RestorationFailed => "Restoration Failed!",
        ApplicationReturnStatus::ErrorInStepComputation => "Error in step computation.",
        ApplicationReturnStatus::MaximumCpuTimeExceeded => "Maximum CPU time exceeded.",
        ApplicationReturnStatus::MaximumWallTimeExceeded => "Maximum wallclock time exceeded.",
        ApplicationReturnStatus::NotEnoughDegreesOfFreedom => "Not Enough Degrees of Freedom.",
        ApplicationReturnStatus::InvalidProblemDefinition => "Invalid Problem Definition.",
        ApplicationReturnStatus::InvalidOption => "Invalid Option.",
        ApplicationReturnStatus::InvalidNumberDetected => {
            "Invalid number in NLP function or derivative detected."
        }
        ApplicationReturnStatus::UnrecoverableException => "Unrecoverable Exception.",
        ApplicationReturnStatus::NonIpoptExceptionThrown => "Exception of type generic.",
        ApplicationReturnStatus::InsufficientMemory => "Insufficient memory.",
        ApplicationReturnStatus::InternalError => "INTERNAL ERROR: Unknown SolverReturn value.",
    }
}

#[cfg(test)]
mod inequality_tally_tests {
    //! Regression test for code review L26: the inequality bound-type
    //! breakdown (`lower_only` / `both` / `upper_only`) must always sum to
    //! `n_ineq`. A "free" inequality row (no finite bound on either side)
    //! previously fell through to a no-op arm, so the breakdown summed to
    //! *less* than the total whenever such a row was present.
    use super::*;
    use pounce_common::types::{Index, Number};
    use pounce_nlp::tnlp::{BoundsInfo, IndexStyle, IpoptCq, IpoptData, Solution, StartingPoint};

    /// Two free variables, three inequality rows of distinct bound types:
    /// row 0 lower-only, row 1 both, row 2 *free* (the bug trigger). No
    /// equality rows. The breakdown must sum to `n_ineq == 3`.
    struct FreeIneqRow;
    impl TNLP for FreeIneqRow {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 3,
                nnz_jac_g: 3,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.iter_mut().for_each(|v| *v = -BOUND_INF);
            b.x_u.iter_mut().for_each(|v| *v = BOUND_INF);
            // row 0: lower-only  [0, +inf)
            b.g_l[0] = 0.0;
            b.g_u[0] = BOUND_INF;
            // row 1: both        [0, 1]
            b.g_l[1] = 0.0;
            b.g_u[1] = 1.0;
            // row 2: free        (-inf, +inf) — the regression trigger
            b.g_l[2] = -BOUND_INF;
            b.g_u[2] = BOUND_INF;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.iter_mut().for_each(|v| *v = 0.0);
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
            grad_f.iter_mut().for_each(|v| *v = 0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.iter_mut().for_each(|v| *v = 0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    // one entry per row so the eq/ineq Jacobian split also
                    // visits each row.
                    irow.copy_from_slice(&[0, 1, 2]);
                    jcol.copy_from_slice(&[0, 0, 0]);
                }
                SparsityRequest::Values { values } => {
                    values.copy_from_slice(&[1.0, 1.0, 1.0]);
                }
            }
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn free_inequality_row_keeps_breakdown_summing_to_total() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(FreeIneqRow));
        let s = collect_stats(
            &tnlp,
            -BOUND_INF,
            BOUND_INF,
            FixedVarTreatment::MakeParameter,
        )
        .expect("collect_stats succeeds");

        assert_eq!(s.n_eq, 0, "no equality rows");
        assert_eq!(s.n_ineq, 3, "all three rows are inequalities");
        // The headline invariant L26 flagged: the three printed buckets must
        // account for every inequality row.
        let bucket_sum: Index = s.ineq_lower_only + s.ineq_both + s.ineq_upper_only;
        assert_eq!(
            bucket_sum, s.n_ineq,
            "ineq bound-type breakdown ({} lower + {} both + {} upper) must sum to n_ineq={}",
            s.ineq_lower_only, s.ineq_both, s.ineq_upper_only, s.n_ineq
        );
        // The free row is bucketed under "both" alongside the genuine
        // both-bounded row 1.
        assert_eq!(s.ineq_lower_only, 1);
        assert_eq!(s.ineq_upper_only, 0);
        assert_eq!(s.ineq_both, 2);
    }

    /// #140 regression. Three variables, the middle one fixed
    /// (`x_l == x_u`). Under the default `make_parameter` the banner must
    /// report the *reduced* problem: the fixed var is dropped from the total,
    /// is NOT bucketed as "lower and upper bounds", and its Jacobian column is
    /// excluded from the nnz tally — matching Ipopt (and the problem the
    /// algorithm actually solves). Previously the banner walked the raw TNLP
    /// and over-reported all three.
    struct OneFixedVar;
    impl TNLP for OneFixedVar {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 3,
                m: 1,
                nnz_jac_g: 3,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            // var 0: lower-only [0, +inf)   var 1: FIXED at 2   var 2: free
            b.x_l[0] = 0.0;
            b.x_u[0] = BOUND_INF;
            b.x_l[1] = 2.0;
            b.x_u[1] = 2.0;
            b.x_l[2] = -BOUND_INF;
            b.x_u[2] = BOUND_INF;
            // one equality row (keeps n_x_var=2 >= n_c=1, so no relax switch)
            b.g_l[0] = 0.0;
            b.g_u[0] = 0.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.iter_mut().for_each(|v| *v = 0.0);
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(0.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
            grad_f.iter_mut().for_each(|v| *v = 0.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.iter_mut().for_each(|v| *v = 0.0);
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                // The equality row touches all three columns, including the
                // fixed var (col 1) that must be filtered out.
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 0, 0]);
                    jcol.copy_from_slice(&[0, 1, 2]);
                }
                SparsityRequest::Values { values } => {
                    values.copy_from_slice(&[1.0, 1.0, 1.0]);
                }
            }
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn make_parameter_banner_reports_reduced_problem() {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OneFixedVar));
        let s = collect_stats(
            &tnlp,
            -BOUND_INF,
            BOUND_INF,
            FixedVarTreatment::MakeParameter,
        )
        .expect("collect_stats succeeds");

        // Fixed var removed: 3 raw vars → 2 optimized.
        assert_eq!(s.n, 2, "fixed variable must be dropped from the total");
        assert_eq!(s.var_both, 0, "fixed var must NOT count as lower-and-upper");
        assert_eq!(s.var_lower_only, 1, "var 0 is lower-only");
        assert_eq!(s.var_free, 1, "var 2 is free");
        assert_eq!(s.var_upper_only, 0);
        // The fixed column is excluded from the Jacobian nnz.
        assert_eq!(s.n_eq, 1);
        assert_eq!(
            s.nnz_jac_eq, 2,
            "fixed-var column dropped from the Jacobian"
        );
        assert_eq!(s.nnz_jac_ineq, 0);
    }

    #[test]
    fn relax_bounds_banner_keeps_fixed_variable() {
        // Under relax_bounds the fixed var stays in the optimization and is
        // reported as a lower-and-upper-bounded variable — matching Ipopt.
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(OneFixedVar));
        let s = collect_stats(&tnlp, -BOUND_INF, BOUND_INF, FixedVarTreatment::RelaxBounds)
            .expect("collect_stats succeeds");

        assert_eq!(s.n, 3, "relax_bounds keeps the fixed variable");
        assert_eq!(
            s.var_both, 1,
            "fixed var reported as lower-and-upper bounded"
        );
        assert_eq!(s.var_lower_only, 1);
        assert_eq!(s.var_free, 1);
        assert_eq!(s.nnz_jac_eq, 3, "all columns retained under relax_bounds");
    }
}
