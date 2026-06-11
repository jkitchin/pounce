//! Ipopt-style banner / problem-stats / final-summary printing for
//! the `pounce` CLI. Output is structured to match upstream Ipopt's
//! console layout closely enough that anyone familiar with `ipopt`
//! can spot at a glance whether POUNCE is converging similarly.

use crate::counting_tnlp::CountingTnlp;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{BoundsInfo, NlpInfo, SparsityRequest, TNLP};
use std::cell::RefCell;
use std::rc::Rc;

/// Same sentinel Ipopt uses for "no bound": ±1e19. Matched exactly so
/// the per-bound-type tallies agree with `ipopt`'s own output on
/// problems whose bounds were authored against that convention.
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

/// Walk the TNLP once to gather everything the banner block needs:
/// `NlpInfo`, the four bound vectors, and the Jacobian row indices.
/// Returns `None` if any of the required TNLP calls fails.
pub fn collect_stats(tnlp: &Rc<RefCell<dyn TNLP>>) -> Option<ProblemStats> {
    let mut t = tnlp.borrow_mut();
    let info: NlpInfo = t.get_nlp_info()?;
    let n = info.n as usize;
    let m = info.m as usize;
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !t.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return None;
    }

    // Variable bound classification.
    let (mut var_lower_only, mut var_upper_only, mut var_both, mut var_free) = (0, 0, 0, 0);
    for i in 0..n {
        let has_l = x_l[i] > -BOUND_INF;
        let has_u = x_u[i] < BOUND_INF;
        match (has_l, has_u) {
            (true, true) => var_both += 1,
            (true, false) => var_lower_only += 1,
            (false, true) => var_upper_only += 1,
            (false, false) => var_free += 1,
        }
    }

    // Constraint classification (equality vs inequality, and the
    // inequality bound type).
    let (mut n_eq, mut n_ineq) = (0, 0);
    let (mut ineq_lower_only, mut ineq_upper_only, mut ineq_both) = (0, 0, 0);
    let mut row_is_eq = vec![false; m];
    for i in 0..m {
        if (g_l[i] - g_u[i]).abs() < 1e-12 && g_l[i].abs() < BOUND_INF {
            n_eq += 1;
            row_is_eq[i] = true;
        } else {
            n_ineq += 1;
            let has_l = g_l[i] > -BOUND_INF;
            let has_u = g_u[i] < BOUND_INF;
            match (has_l, has_u) {
                (true, true) => ineq_both += 1,
                (true, false) => ineq_lower_only += 1,
                (false, true) => ineq_upper_only += 1,
                // A "free" inequality has no bounds at all — count it
                // anyway under "both" to keep the totals consistent.
                (false, false) => {}
            }
        }
    }

    // Jacobian split: read the structure once and tally per-row.
    let nnz_total = info.nnz_jac_g as usize;
    let (mut nnz_jac_eq, mut nnz_jac_ineq) = (0, 0);
    if nnz_total > 0 && m > 0 {
        let mut irow = vec![0_i32; nnz_total];
        let mut jcol = vec![0_i32; nnz_total];
        if t.eval_jac_g(
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        ) {
            let one_based = matches!(info.index_style, pounce_nlp::tnlp::IndexStyle::Fortran);
            for &r in &irow {
                let row = if one_based {
                    (r - 1) as usize
                } else {
                    r as usize
                };
                if row < m && row_is_eq[row] {
                    nnz_jac_eq += 1;
                } else {
                    nnz_jac_ineq += 1;
                }
            }
        }
    }

    Some(ProblemStats {
        n: info.n,
        m: info.m,
        nnz_jac_eq,
        nnz_jac_ineq,
        nnz_h_lag: info.nnz_h_lag,
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
