//! gh#979 — the captured iteration history must reproduce the table the
//! solver prints, line for line, restoration rows included.
//!
//! Two defects, one test each, both read against the *printed* table rather
//! than against a number worked out here, because "the report says what the
//! console says" is the contract a caller annotating a log relies on:
//!
//! 1. `inf_pr` was the internal slack-form residual `‖d(x) − s‖` with the
//!    slack pushed off its bound (what the filter reads), not the printed
//!    violation of the user's model. At the Rosenbrock start below the two
//!    are 0.955 and 0.94.
//! 2. The restoration phase's `r`-suffixed rows were dropped; only the
//!    main-phase `R` row that entered restoration was recorded.
//!
//! Each test re-runs this binary as a child with `print_level=5` so the
//! table and the captured records come from the *same* solve; the child
//! prints the records after the table in a tagged form and the parent
//! matches them against the table rows by the digits the table prints.

use std::cell::RefCell;
use std::process::Command;
use std::rc::Rc;

use pounce_rs::prelude::*;
use pounce_rs::{IterPhase, IterRecord};

const CHILD_ENV: &str = "POUNCE_ISSUE979_CHILD";
const REC_TAG: &str = "ISSUE979-REC";

type Dense1 = fn(&[f64]) -> f64;

/// A dense two-variable TNLP with exact derivatives, so the solve takes the
/// exact-Hessian path the issue's tables were printed on.
struct Dense2 {
    m: usize,
    x0: [f64; 2],
    xl: [f64; 2],
    xu: [f64; 2],
    gl: Vec<f64>,
    gu: Vec<f64>,
    f: Dense1,
    grad: fn(&[f64]) -> [f64; 2],
    g: fn(&[f64], &mut [f64]),
    /// Row-major `m × 2`.
    jac: fn(&[f64], &mut [f64]),
    /// Lower triangle `(0,0), (1,0), (1,1)` of `σ∇²f + Σ λ_j ∇²g_j`.
    hess: fn(&[f64], f64, &[f64]) -> [f64; 3],
}

impl TNLP for Dense2 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: self.m as _,
            nnz_jac_g: (2 * self.m) as _,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.xl);
        b.x_u.copy_from_slice(&self.xu);
        b.g_l.copy_from_slice(&self.gl);
        b.g_u.copy_from_slice(&self.gu);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }
    fn eval_f(&mut self, x: &[f64], _new_x: bool) -> Option<f64> {
        Some((self.f)(x))
    }
    fn eval_grad_f(&mut self, x: &[f64], _new_x: bool, grad_f: &mut [f64]) -> bool {
        grad_f.copy_from_slice(&(self.grad)(x));
        true
    }
    fn eval_g(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        (self.g)(x, g);
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[f64]>, _new_x: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for j in 0..self.m {
                    irow[2 * j] = j as _;
                    jcol[2 * j] = 0;
                    irow[2 * j + 1] = j as _;
                    jcol[2 * j + 1] = 1;
                }
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                (self.jac)(x, values);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[f64]>,
        _new_x: bool,
        obj_factor: f64,
        lambda: Option<&[f64]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let (Some(x), Some(l)) = (x, lambda) else {
                    return false;
                };
                values.copy_from_slice(&(self.hess)(x, obj_factor, l));
            }
        }
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// `min (x0-1)² + 100 (x1-x0²)²  s.t.  x0² + x1² ≤ 1.5,  x ∈ [-2,2]²`.
fn rosenbrock_disk() -> Dense2 {
    Dense2 {
        m: 1,
        x0: [-1.2, 1.0],
        xl: [-2.0, -2.0],
        xu: [2.0, 2.0],
        gl: vec![-1e19],
        gu: vec![1.5],
        f: |x| (x[0] - 1.0).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
        grad: |x| {
            let r = x[1] - x[0] * x[0];
            [2.0 * (x[0] - 1.0) - 400.0 * x[0] * r, 200.0 * r]
        },
        g: |x, g| g[0] = x[0] * x[0] + x[1] * x[1],
        jac: |x, j| {
            j[0] = 2.0 * x[0];
            j[1] = 2.0 * x[1];
        },
        hess: |x, s, l| {
            [
                s * (2.0 - 400.0 * x[1] + 1200.0 * x[0] * x[0]) + 2.0 * l[0],
                s * (-400.0 * x[0]),
                s * 200.0 + 2.0 * l[0],
            ]
        },
    }
}

/// `min x0 + x1  s.t.  x0² + x1² = 1,  x0·x1 = 0.4,  x ∈ [-10,10]²` — the
/// issue's restoration example.
fn circle_hyperbola(x0: [f64; 2]) -> Dense2 {
    Dense2 {
        m: 2,
        x0,
        xl: [-10.0, -10.0],
        xu: [10.0, 10.0],
        gl: vec![1.0, 0.4],
        gu: vec![1.0, 0.4],
        f: |x| x[0] + x[1],
        grad: |_| [1.0, 1.0],
        g: |x, g| {
            g[0] = x[0] * x[0] + x[1] * x[1];
            g[1] = x[0] * x[1];
        },
        jac: |x, j| {
            j[0] = 2.0 * x[0];
            j[1] = 2.0 * x[1];
            j[2] = x[1];
            j[3] = x[0];
        },
        hess: |_, _, l| [2.0 * l[0], l[1], 2.0 * l[0]],
    }
}

/// Starting point for the restoration example. The issue does not give
/// one; from this one the solve enters restoration at iteration 2, runs
/// eleven `r` rows, and returns to converge at `-sqrt(1.8)`.
const CIRCLE_X0: [f64; 2] = [-0.7, -0.7];

/// Child side: solve with the table printed and the history captured, then
/// print the records after the table.
fn run_child(which: &str) {
    let problem = match which {
        "rosenbrock" => rosenbrock_disk(),
        "circle" => circle_hyperbola(CIRCLE_X0),
        other => panic!("unknown child problem {other}"),
    };
    let _scope = collector_scope();
    let mut app = IpoptApplication::new();
    assert!(app.initialize().is_ok());
    let opts = app.options_mut();
    let _ = opts.set_integer_value("print_level", 5, true, true);
    // Pin the NLP arm: the tables in the issue are interior-point tables.
    let _ = opts.set_string_value("solver_selection", "nlp", true, true);
    // Wire the restoration phase the way the CLI and Python frontends do;
    // a bare `IpoptApplication` has none.
    use pounce_rs::pounce_algorithm::application::{
        Ma57Config, default_backend_factory, feral_config_from_options,
    };
    use pounce_rs::pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
    use pounce_rs::pounce_restoration::resto_inner_solver::{
        InnerBackendFactoryFactory, make_default_restoration_factory_provider,
    };
    let feral_cfg = feral_config_from_options(app.options());
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        Box::new(move || default_backend_factory(feral_cfg.clone(), Ma57Config::default()))
    };
    let provider = make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        app.algorithm_builder_from_options(),
        bff_mint,
    );
    app.set_restoration_factory_provider(provider);
    app.enable_iter_history();
    let prob = Rc::new(RefCell::new(problem));
    let _ = app.optimize_tnlp(prob as Rc<RefCell<dyn TNLP>>);
    for r in &app.statistics().iterations {
        println!(
            "{REC_TAG} {} {} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {:e} {} {}",
            r.iter,
            r.phase.as_str(),
            r.objective,
            r.inf_pr,
            r.inf_pr_internal,
            r.inf_du,
            r.mu,
            r.d_norm,
            r.regularization,
            r.alpha_dual,
            r.alpha_primal,
            if r.alpha_primal_char == ' ' {
                '_'
            } else {
                r.alpha_primal_char
            },
            r.ls_trials,
        );
    }
}

/// C `%.Ne`, as the table prints it.
fn format_e(x: f64, precision: usize) -> String {
    let s = format!("{x:.precision$e}");
    let (mantissa, exp) = s.split_once('e').unwrap();
    let (sign, digits) = match exp.strip_prefix('-') {
        Some(rest) => ('-', rest),
        None => ('+', exp),
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

struct TableRow {
    iter: i32,
    restoration: bool,
    objective: String,
    inf_pr: String,
    inf_du: String,
    lg_mu: String,
    d_norm: String,
    lg_rg: String,
    alpha_du: String,
    alpha_pr: String,
    alpha_char: char,
    ls: i32,
}

/// Parse one printed iteration row, or `None` for any other line.
fn parse_row(line: &str) -> Option<TableRow> {
    // `%4d` then a space (main) or a literal `r` (restoration), then the
    // `%14.7e` objective, which can abut the `r` when it is negative.
    if line.len() < 5 {
        return None;
    }
    let (head, rest) = line.split_at(4);
    let iter: i32 = head.trim().parse().ok()?;
    let restoration = rest.starts_with('r');
    let toks: Vec<&str> = rest[1..].split_whitespace().collect();
    if toks.len() != 9 {
        return None;
    }
    let (alpha_pr, alpha_char) = toks[7].split_at(8);
    Some(TableRow {
        iter,
        restoration,
        objective: toks[0].into(),
        inf_pr: toks[1].into(),
        inf_du: toks[2].into(),
        lg_mu: toks[3].into(),
        d_norm: toks[4].into(),
        lg_rg: toks[5].into(),
        alpha_du: toks[6].into(),
        alpha_pr: alpha_pr.into(),
        alpha_char: alpha_char.chars().next().unwrap_or(' '),
        ls: toks[8].parse().ok()?,
    })
}

fn parse_rec(line: &str) -> Option<IterRecord> {
    let rest = line.strip_prefix(REC_TAG)?;
    let t: Vec<&str> = rest.split_whitespace().collect();
    let f = |i: usize| t[i].parse::<f64>().unwrap();
    Some(IterRecord {
        iter: t[0].parse().unwrap(),
        phase: IterPhase::from_name(t[1]),
        objective: f(2),
        inf_pr: f(3),
        inf_pr_internal: f(4),
        inf_du: f(5),
        mu: f(6),
        d_norm: f(7),
        regularization: f(8),
        alpha_dual: f(9),
        alpha_primal: f(10),
        alpha_primal_char: match t[11].chars().next().unwrap() {
            '_' => ' ',
            c => c,
        },
        ls_trials: t[12].parse().unwrap(),
    })
}

/// Run the child, returning the printed table rows and the records.
fn table_and_records(which: &str) -> (Vec<TableRow>, Vec<IterRecord>) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, which)
        .output()
        .expect("spawn child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "child failed:\n{stdout}");
    let rows: Vec<TableRow> = stdout.lines().filter_map(parse_row).collect();
    let recs: Vec<IterRecord> = stdout.lines().filter_map(parse_rec).collect();
    assert!(!rows.is_empty(), "no table rows in child output:\n{stdout}");
    (rows, recs)
}

/// Every printed column, record against row, by the digits the table prints.
fn assert_rows_match(rows: &[TableRow], recs: &[IterRecord]) {
    assert_eq!(
        rows.len(),
        recs.len(),
        "the report must carry one record per printed row; printed iters {:?}, recorded {:?}",
        rows.iter()
            .map(|r| format!("{}{}", r.iter, if r.restoration { "r" } else { "" }))
            .collect::<Vec<_>>(),
        recs.iter()
            .map(|r| format!(
                "{}{}",
                r.iter,
                if r.phase == IterPhase::Restoration {
                    "r"
                } else {
                    ""
                }
            ))
            .collect::<Vec<_>>(),
    );
    for (row, rec) in rows.iter().zip(recs) {
        let at = format!(
            "printed row {}{}",
            row.iter,
            if row.restoration { "r" } else { "" }
        );
        assert_eq!(rec.iter, row.iter, "{at}: iter");
        assert_eq!(
            rec.phase == IterPhase::Restoration,
            row.restoration,
            "{at}: phase"
        );
        assert_eq!(format_e(rec.objective, 7), row.objective, "{at}: objective");
        assert_eq!(format_e(rec.inf_pr, 2), row.inf_pr, "{at}: inf_pr");
        assert_eq!(format_e(rec.inf_du, 2), row.inf_du, "{at}: inf_du");
        assert_eq!(format!("{:.1}", rec.mu.log10()), row.lg_mu, "{at}: lg(mu)");
        assert_eq!(format_e(rec.d_norm, 2), row.d_norm, "{at}: ||d||");
        let lg_rg = if rec.regularization == 0.0 {
            "-".to_string()
        } else {
            format!("{:.1}", rec.regularization.log10())
        };
        assert_eq!(lg_rg, row.lg_rg, "{at}: lg(rg)");
        assert_eq!(format_e(rec.alpha_dual, 2), row.alpha_du, "{at}: alpha_du");
        assert_eq!(
            format_e(rec.alpha_primal, 2),
            row.alpha_pr,
            "{at}: alpha_pr"
        );
        assert_eq!(rec.alpha_primal_char, row.alpha_char, "{at}: alpha char");
        assert_eq!(rec.ls_trials, row.ls, "{at}: ls");
    }
}

/// Entry point the parent re-executes; a no-op when run as a normal test.
#[test]
fn child_entry() {
    if let Ok(which) = std::env::var(CHILD_ENV) {
        run_child(&which);
    }
}

#[test]
fn inf_pr_is_the_printed_violation_not_the_slack_residual() {
    let (rows, recs) = table_and_records("rosenbrock");
    assert_rows_match(&rows, &recs);
    // At x0, g = 2.44 against cu = 1.5: the printed violation is 0.94,
    // while the internal residual measures against the slack pushed off
    // its bound. Both must be present and they must differ, or this test
    // is not exercising the distinction it is named for.
    let r0 = &recs[0];
    assert!(
        (r0.inf_pr - 0.94).abs() < 1e-12,
        "iter 0 inf_pr = {}",
        r0.inf_pr
    );
    assert!(
        r0.inf_pr_internal > r0.inf_pr + 1e-3,
        "iter 0 inf_pr_internal = {} should be the slack-form residual",
        r0.inf_pr_internal
    );
}

#[test]
fn restoration_rows_are_recorded_and_tagged() {
    let (rows, recs) = table_and_records("circle");
    assert!(
        rows.iter().any(|r| r.restoration),
        "fixture no longer enters restoration — pick a start that does"
    );
    assert_rows_match(&rows, &recs);
    // The `R` row that hands over to restoration stays a main-phase row.
    let entry = recs
        .iter()
        .position(|r| r.phase == IterPhase::Restoration)
        .unwrap();
    assert!(entry > 0);
    assert_eq!(recs[entry - 1].phase, IterPhase::Main);
    // A restoration row's `inf_pr` is the original NLP's violation (what
    // the `r` row prints); its internal residual is the restoration
    // problem's own, which the slack variables drive to zero.
    let r = &recs[entry + 1];
    assert!(
        r.inf_pr_internal < 1e-2 * r.inf_pr,
        "restoration row {}: inf_pr {} vs internal {}",
        r.iter,
        r.inf_pr,
        r.inf_pr_internal
    );
}
