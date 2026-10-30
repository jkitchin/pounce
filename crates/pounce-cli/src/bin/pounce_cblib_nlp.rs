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

/// How a cone is written as smooth NLP rows — what a user had to do before
/// the NLP arm carried cones. Each cone row's affine form gets an auxiliary
/// variable `u` (`u = h − Gx`), and the cone becomes one inequality on `u`.
#[derive(Clone, Copy, PartialEq)]
enum Lowering {
    /// `u₀ − ‖u_{1..}‖ ≥ 0`; not differentiable at `u_{1..} = 0` (the
    /// subgradient 0 is used there, as most modeling layers would give).
    Norm,
    /// `u₀² − ‖u_{1..}‖² ≥ 0`, `u₀ ≥ 0`; smooth, gradient vanishes at the tip.
    Squared,
}

struct LoweredNlp {
    prob: QpProblem,
    mode: Lowering,
    /// `G` rows that are orthant rows (`h − Gx ≥ 0`).
    orth: Vec<usize>,
    /// Per cone, its `G` rows (head first).
    cones: Vec<Vec<usize>>,
    /// `G` as row lists.
    g_rows: Vec<Vec<(usize, Number)>>,
    /// Index of each cone row's `u` among the `u` variables.
    n_u: usize,
    jac: Vec<(usize, usize, Number)>,
    /// Row index of each cone's nonlinear row.
    cone_row0: usize,
    out: Rc<RefCell<Option<Vec<Number>>>>,
}

impl LoweredNlp {
    fn new(prob: QpProblem, cones: &[ConeSpec], mode: Lowering) -> Option<Self> {
        let mut orth = Vec::new();
        let mut cone_rows = Vec::new();
        let mut r = 0usize;
        for c in cones {
            match *c {
                ConeSpec::Nonneg(k) => {
                    orth.extend(r..r + k);
                    r += k;
                }
                ConeSpec::SecondOrder(k) if k == 1 => {
                    orth.push(r);
                    r += 1;
                }
                ConeSpec::SecondOrder(k) => {
                    cone_rows.push((r..r + k).collect::<Vec<_>>());
                    r += k;
                }
                _ => return None,
            }
        }
        if r != prob.h.len() {
            return None;
        }
        let mut g_rows = vec![Vec::new(); prob.h.len()];
        for t in &prob.g {
            g_rows[t.row].push((t.col, t.val));
        }
        let n = prob.n;
        let m_eq = prob.b.len();
        let n_u: usize = cone_rows.iter().map(|c| c.len()).sum();
        let mut jac: Vec<(usize, usize, Number)> =
            prob.a.iter().map(|t| (t.row, t.col, t.val)).collect();
        let mut row = m_eq;
        for &gr in &orth {
            for &(c, v) in &g_rows[gr] {
                jac.push((row, c, -v));
            }
            row += 1;
        }
        // link rows: u_k + G_r x = h_r
        let mut k = 0usize;
        for c in &cone_rows {
            for &gr in c {
                jac.push((row, n + k, 1.0));
                for &(col, v) in &g_rows[gr] {
                    jac.push((row, col, v));
                }
                row += 1;
                k += 1;
            }
        }
        let cone_row0 = row;
        let mut k0 = 0usize;
        for c in &cone_rows {
            for j in 0..c.len() {
                jac.push((row, n + k0 + j, 0.0));
            }
            row += 1;
            k0 += c.len();
        }
        Some(Self {
            prob,
            mode,
            orth,
            cones: cone_rows,
            g_rows,
            n_u,
            jac,
            cone_row0,
            out: Rc::new(RefCell::new(None)),
        })
    }

    fn m(&self) -> usize {
        self.cone_row0 + self.cones.len()
    }

    /// `(value, gradient over the cone's u, Hessian (row-major) over them)`.
    fn cone_fn(&self, u: &[Number]) -> (Number, Vec<Number>, Vec<Number>) {
        let m = u.len();
        let mut g = vec![0.0; m];
        let mut h = vec![0.0; m * m];
        match self.mode {
            Lowering::Squared => {
                let mut v = u[0] * u[0];
                g[0] = 2.0 * u[0];
                h[0] = 2.0;
                for j in 1..m {
                    v -= u[j] * u[j];
                    g[j] = -2.0 * u[j];
                    h[j * m + j] = -2.0;
                }
                (v, g, h)
            }
            Lowering::Norm => {
                let nrm = u[1..].iter().map(|v| v * v).sum::<Number>().sqrt();
                g[0] = 1.0;
                if nrm > 0.0 {
                    for j in 1..m {
                        g[j] = -u[j] / nrm;
                        for l in 1..m {
                            let d = if j == l { 1.0 } else { 0.0 };
                            h[j * m + l] = -(d / nrm - u[j] * u[l] / (nrm * nrm * nrm));
                        }
                    }
                }
                (u[0] - nrm, g, h)
            }
        }
    }

    fn cone_u<'a>(&self, x: &'a [Number], k: usize) -> (&'a [Number], usize) {
        let off: usize = self.cones[..k].iter().map(|c| c.len()).sum();
        let n = self.prob.n;
        (&x[n + off..n + off + self.cones[k].len()], n + off)
    }
}

impl TNLP for LoweredNlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let hu: usize = self.cones.iter().map(|c| c.len() * (c.len() + 1) / 2).sum();
        Some(NlpInfo {
            n: (self.prob.n + self.n_u) as Index,
            m: self.m() as Index,
            nnz_jac_g: self.jac.len() as Index,
            nnz_h_lag: (self.prob.p_lower.len() + hu) as Index,
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
        let mut k = n;
        for c in &self.cones {
            for j in 0..c.len() {
                b.x_l[k] = if j == 0 && self.mode == Lowering::Squared {
                    0.0
                } else {
                    -INF
                };
                b.x_u[k] = INF;
                k += 1;
            }
        }
        let m_eq = self.prob.b.len();
        let mut row = 0;
        for i in 0..m_eq {
            b.g_l[row] = self.prob.b[i];
            b.g_u[row] = self.prob.b[i];
            row += 1;
        }
        for &gr in &self.orth {
            b.g_l[row] = -self.prob.h[gr];
            b.g_u[row] = INF;
            row += 1;
        }
        for c in &self.cones {
            for &gr in c {
                b.g_l[row] = self.prob.h[gr];
                b.g_u[row] = self.prob.h[gr];
                row += 1;
            }
        }
        for _ in &self.cones {
            b.g_l[row] = 0.0;
            b.g_u[row] = INF;
            row += 1;
        }
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.fill(0.0);
        let mut k = self.prob.n;
        for c in &self.cones {
            for &gr in c {
                sp.x[k] = self.prob.h[gr];
                k += 1;
            }
        }
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
        g.fill(0.0);
        g[..self.prob.n].copy_from_slice(&self.prob.c);
        for t in &self.prob.p_lower {
            g[t.row] += t.val * x[t.col];
            if t.row != t.col {
                g[t.col] += t.val * x[t.row];
            }
        }
        true
    }
    fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g.fill(0.0);
        for t in &self.prob.a {
            g[t.row] += t.val * x[t.col];
        }
        let mut row = self.prob.b.len();
        for &gr in &self.orth {
            for &(c, v) in &self.g_rows[gr] {
                g[row] -= v * x[c];
            }
            row += 1;
        }
        let mut k = self.prob.n;
        for c in &self.cones {
            for &gr in c {
                g[row] = x[k];
                for &(col, v) in &self.g_rows[gr] {
                    g[row] += v * x[col];
                }
                row += 1;
                k += 1;
            }
        }
        for kk in 0..self.cones.len() {
            let (u, _) = self.cone_u(x, kk);
            g[row] = self.cone_fn(u).0;
            row += 1;
        }
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c, _)) in self.jac.iter().enumerate() {
                    irow[k] = r as Index;
                    jcol[k] = c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.unwrap();
                let lin = self.jac.len() - self.n_u;
                for (k, &(_, _, v)) in self.jac.iter().enumerate().take(lin) {
                    values[k] = v;
                }
                let mut k = lin;
                for kk in 0..self.cones.len() {
                    let (u, _) = self.cone_u(x, kk);
                    for gv in self.cone_fn(u).1 {
                        values[k] = gv;
                        k += 1;
                    }
                }
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _: bool,
        s: Number,
        lam: Option<&[Number]>,
        _: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for t in &self.prob.p_lower {
                    irow[k] = t.row as Index;
                    jcol[k] = t.col as Index;
                    k += 1;
                }
                for kk in 0..self.cones.len() {
                    let off: usize =
                        self.prob.n + self.cones[..kk].iter().map(|c| c.len()).sum::<usize>();
                    let m = self.cones[kk].len();
                    for i in 0..m {
                        for j in 0..=i {
                            irow[k] = (off + i) as Index;
                            jcol[k] = (off + j) as Index;
                            k += 1;
                        }
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let (x, lam) = (x.unwrap(), lam.unwrap());
                let mut k = 0;
                for t in &self.prob.p_lower {
                    values[k] = s * t.val;
                    k += 1;
                }
                for kk in 0..self.cones.len() {
                    let (u, _) = self.cone_u(x, kk);
                    let h = self.cone_fn(u).2;
                    let l = lam[self.cone_row0 + kk];
                    let m = u.len();
                    for i in 0..m {
                        for j in 0..=i {
                            values[k] = l * h[i * m + j];
                            k += 1;
                        }
                    }
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

    // NLP arms: cones declared natively ("after"), and the two smooth
    // lowerings a user had to write before ("before").
    let objective = |x: &[Number]| {
        let mut f: Number = cp.prob.c.iter().zip(x).map(|(c, v)| c * v).sum();
        for t in &cp.prob.p_lower {
            let w = if t.row == t.col { 0.5 } else { 1.0 };
            f += w * t.val * x[t.row] * x[t.col];
        }
        cp.cbf_objective(f, model.minimize)
    };
    let run =
        |tnlp: Rc<RefCell<dyn TNLP>>, out: Rc<RefCell<Option<Vec<Number>>>>, socs: Vec<SocRows>| {
            let mut app = IpoptApplication::new();
            let pl = std::env::var("POUNCE_PRINT_LEVEL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let _ = app
                .options_mut()
                .set_integer_value("print_level", pl, true, false);
            let _ = app
                .options_mut()
                .set_integer_value("max_iter", max_iter as Index, true, false);
            // `POUNCE_OPTS="k=v,k=v"`: extra options for the NLP arms (diagnosis).
            if let Ok(extra) = std::env::var("POUNCE_OPTS") {
                for kv in extra.split(',').filter(|s| !s.is_empty()) {
                    if let Some((k, v)) = kv.split_once('=') {
                        let _ = app.options_mut().read_from_str(&format!("{k} {v}\n"), true);
                    }
                }
            }
            app.set_soc_constraints(socs);
            let t0 = std::time::Instant::now();
            let status = app.optimize_tnlp(tnlp);
            let time = t0.elapsed().as_secs_f64();
            let obj = out.borrow().as_ref().map(|x| objective(&x[..cp.prob.n]));
            serde_json::json!({"status": status.upstream_name(), "obj": obj,
                           "iters": app.statistics().spent_iteration_count(), "time": time})
        };
    let Some((nlp, socs)) = ConicNlp::new(cp.prob.clone(), &cp.cones) else {
        fail("cones other than orthant / second-order".into());
    };
    let out = Rc::clone(&nlp.out);
    let native = run(Rc::new(RefCell::new(nlp)), out, socs);
    let lowered = |mode| {
        let l = LoweredNlp::new(cp.prob.clone(), &cp.cones, mode).expect("same cones");
        let out = Rc::clone(&l.out);
        run(Rc::new(RefCell::new(l)), out, Vec::new())
    };
    let norm = lowered(Lowering::Norm);
    let squared = lowered(Lowering::Squared);

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
            "nlp": native,
            "nlp_norm": norm,
            "nlp_squared": squared,
        })
    );
}
