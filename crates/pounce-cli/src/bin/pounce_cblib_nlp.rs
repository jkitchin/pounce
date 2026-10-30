//! `pounce_cblib_nlp` — solve a CBLIB `.cbf` second-order-cone instance twice:
//! through the convex conic driver (`pounce-convex`, a global optimum), and
//! through the NLP interior-point method with the cones declared natively
//! (workstream C, `dev-notes/research/nlp-soc-design.md`). Prints one JSON
//! line with both results.
//!
//! ```text
//! pounce_cblib_nlp <file.cbf> [--max-iter N]
//! ```
//!
//! Only instances whose cones are the nonnegative orthant and second-order
//! cones are taken (the NLP arm carries nothing else); integer markers are
//! dropped, as the conic driver drops them, so both arms solve the same
//! continuous relaxation.

use pounce_algorithm::application::IpoptApplication;
use pounce_cli::cbf;
use pounce_common::types::{Index, Number};
use pounce_convex::{ConeSpec, QpOptions, QpProblem, QpStatus, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::soc::SocRows;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const INF: Number = 1e19;

/// `min ½xᵀPx + cᵀx  s.t.  Ax = b,  h − Gx ∈ K` as an NLP: the equality rows,
/// then one row per row of `G` (`g = h − Gx`): an orthant row reads `g ≥ 0`,
/// a second-order cone's rows are free and declared as a cone.
struct ConicNlp {
    prob: QpProblem,
    /// Per `G` row: `true` for an orthant row (`g >= 0`), `false` for a cone row.
    nonneg: Vec<bool>,
    jac: Vec<(usize, usize, Number)>,
    out: Rc<RefCell<Option<Vec<Number>>>>,
}

impl ConicNlp {
    fn new(prob: QpProblem, cones: &[ConeSpec]) -> Option<(Self, Vec<SocRows>)> {
        let m_eq = prob.b.len();
        let mut nonneg = Vec::new();
        let mut socs = Vec::new();
        for c in cones {
            match *c {
                ConeSpec::Nonneg(k) => nonneg.extend(std::iter::repeat_n(true, k)),
                ConeSpec::SecondOrder(k) => {
                    let start = m_eq + nonneg.len();
                    if k == 1 {
                        // A one-dimensional second-order cone is a bound.
                        nonneg.push(true);
                    } else {
                        socs.push(SocRows {
                            rows: (start..start + k).collect(),
                        });
                        nonneg.extend(std::iter::repeat_n(false, k));
                    }
                }
                _ => return None,
            }
        }
        if nonneg.len() != prob.h.len() {
            return None;
        }
        let mut jac: Vec<(usize, usize, Number)> =
            prob.a.iter().map(|t| (t.row, t.col, t.val)).collect();
        jac.extend(prob.g.iter().map(|t| (m_eq + t.row, t.col, -t.val)));
        let out = Rc::new(RefCell::new(None));
        Some((
            Self {
                prob,
                nonneg,
                jac,
                out,
            },
            socs,
        ))
    }

    fn m(&self) -> usize {
        self.prob.b.len() + self.prob.h.len()
    }
}

impl TNLP for ConicNlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.prob.n as Index,
            m: self.m() as Index,
            nnz_jac_g: self.jac.len() as Index,
            nnz_h_lag: self.prob.p_lower.len() as Index,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let n = self.prob.n;
        for j in 0..n {
            b.x_l[j] = self
                .prob
                .lb
                .get(j)
                .copied()
                .filter(|v| v.is_finite())
                .unwrap_or(-INF);
            b.x_u[j] = self
                .prob
                .ub
                .get(j)
                .copied()
                .filter(|v| v.is_finite())
                .unwrap_or(INF);
        }
        let m_eq = self.prob.b.len();
        for i in 0..m_eq {
            b.g_l[i] = self.prob.b[i];
            b.g_u[i] = self.prob.b[i];
        }
        for (r, &nn) in self.nonneg.iter().enumerate() {
            b.g_l[m_eq + r] = if nn { 0.0 } else { -INF };
            b.g_u[m_eq + r] = INF;
        }
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.fill(0.0);
        true
    }
    fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
        let mut f: Number = self.prob.c.iter().zip(x).map(|(c, v)| c * v).sum();
        for t in &self.prob.p_lower {
            let w = if t.row == t.col { 0.5 } else { 1.0 };
            f += w * t.val * x[t.row] * x[t.col];
        }
        Some(f)
    }
    fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&self.prob.c);
        for t in &self.prob.p_lower {
            g[t.row] += t.val * x[t.col];
            if t.row != t.col {
                g[t.col] += t.val * x[t.row];
            }
        }
        true
    }
    fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        let m_eq = self.prob.b.len();
        g.fill(0.0);
        for i in 0..self.prob.h.len() {
            g[m_eq + i] = self.prob.h[i];
        }
        for &(r, c, v) in &self.jac {
            g[r] += v * x[c];
        }
        true
    }
    fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c, _)) in self.jac.iter().enumerate() {
                    irow[k] = r as Index;
                    jcol[k] = c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                for (k, &(_, _, v)) in self.jac.iter().enumerate() {
                    values[k] = v;
                }
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _: Option<&[Number]>,
        _: bool,
        s: Number,
        _: Option<&[Number]>,
        _: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, t) in self.prob.p_lower.iter().enumerate() {
                    irow[k] = t.row as Index;
                    jcol[k] = t.col as Index;
                }
            }
            SparsityRequest::Values { values } => {
                for (k, t) in self.prob.p_lower.iter().enumerate() {
                    values[k] = s * t.val;
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _: &IpoptData, _: &IpoptCq) {
        *self.out.borrow_mut() = Some(sol.x.to_vec());
    }
}

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: pounce_cblib_nlp <file.cbf> [--max-iter N]");
        std::process::exit(2);
    };
    let max_iter: usize = args
        .iter()
        .position(|a| a == "--max-iter")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000);
    let name = std::path::Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().trim_end_matches(".cbf").to_string())
        .unwrap_or_default();
    let fail = |why: String| -> ! {
        println!("{}", serde_json::json!({"name": name, "skipped": why}));
        std::process::exit(0);
    };
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| fail(format!("read: {e}")));
    let model = cbf::parse(&text).unwrap_or_else(|e| fail(format!("parse: {e}")));
    let cp = model
        .to_conic()
        .unwrap_or_else(|e| fail(format!("map: {e}")));
    let n_soc = cp
        .cones
        .iter()
        .filter(|c| matches!(c, ConeSpec::SecondOrder(k) if *k > 1))
        .count();
    if n_soc == 0 {
        fail("no second-order cone".into());
    }

    // Convex arm.
    let opts = QpOptions {
        max_iter,
        ..QpOptions::default()
    };
    let t0 = std::time::Instant::now();
    let sol = solve_socp_ipm(&cp.prob, &cp.cones, &opts, backend);
    let convex_time = t0.elapsed().as_secs_f64();
    let convex_obj = cp.cbf_objective(sol.obj, model.minimize);
    let convex_ok = matches!(sol.status, QpStatus::Optimal);

    // NLP arm, cones declared natively.
    let Some((nlp, socs)) = ConicNlp::new(cp.prob.clone(), &cp.cones) else {
        fail("cones other than orthant / second-order".into());
    };
    let out = Rc::clone(&nlp.out);
    let mut app = IpoptApplication::new();
    let o = app.options_mut();
    let pl = std::env::var("POUNCE_PRINT_LEVEL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let _ = o.set_integer_value("print_level", pl, true, false);
    let _ = o.set_integer_value("max_iter", max_iter as Index, true, false);
    // `POUNCE_OPTS="k=v,k=v"`: extra options for the NLP arm (diagnosis).
    if let Ok(extra) = std::env::var("POUNCE_OPTS") {
        for kv in extra.split(',').filter(|s| !s.is_empty()) {
            if let Some((k, v)) = kv.split_once('=') {
                let _ = app.options_mut().read_from_str(&format!("{k} {v}\n"), true);
            }
        }
    }
    app.set_soc_constraints(socs);
    let t0 = std::time::Instant::now();
    let status = app.optimize_tnlp(Rc::new(RefCell::new(nlp)));
    let nlp_time = t0.elapsed().as_secs_f64();
    let x = out.borrow().clone();
    let nlp_iters = app.statistics().spent_iteration_count();
    let nlp_obj = x.as_ref().map(|x| {
        let mut f: Number = cp.prob.c.iter().zip(x).map(|(c, v)| c * v).sum();
        for t in &cp.prob.p_lower {
            let w = if t.row == t.col { 0.5 } else { 1.0 };
            f += w * t.val * x[t.row] * x[t.col];
        }
        cp.cbf_objective(f, model.minimize)
    });

    println!(
        "{}",
        serde_json::json!({
            "name": name,
            "n": cp.prob.n,
            "m_eq": cp.prob.b.len(),
            "m_cone": cp.prob.h.len(),
            "n_soc": n_soc,
            "convex": {"status": format!("{:?}", sol.status), "ok": convex_ok,
                       "obj": convex_obj, "iters": sol.iters, "time": convex_time},
            "nlp": {"status": status.upstream_name(), "obj": nlp_obj,
                    "iters": nlp_iters, "time": nlp_time},
        })
    );
}
