//! CBLIB cross-check: solve each exponential-cone instance **twice** —
//! once as a conic program through the non-symmetric HSDE driver, once as a
//! smooth NLP through POUNCE's filter-IPM — and assert the two independent
//! solvers agree on the objective.
//!
//! The smooth NLP reuses the CBF variables: each `VAR EXP` triple
//! `(u₀, u₁, u₂)` (CBF order: `u₀ ≥ u₁·exp(u₂/u₁)`) becomes the constraint
//! `g = u₀ − u₁·exp(u₂/u₁) ≥ 0` with `u₁ ≥ 0`, supplied with its exact
//! gradient and Hessian; the `L=` / `L-` constraint rows stay linear. Because
//! the conic and NLP paths share no code, agreement is strong evidence the
//! exp-cone benchmark pipeline (parse → map → solve) is correct — the
//! validation strategy from `dev-notes/hsde.md`.

use pounce_algorithm::application::IpoptApplication;
use pounce_cli::cbf::{self, CbfModel, ConeKind};
use pounce_common::types::{Index, Number};
use pounce_convex::{solve_socp_ipm, QpOptions, QpStatus};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const INF: f64 = 1e20;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// A CBF power cone in smooth-NLP form: `|x_bnd| ≤ u₀^α · u₁^{1−α}`,
/// `u₀,u₁ ≥ 0`, modeled as the two constraints `φ ∓ x_bnd ≥ 0` with
/// `φ = u₀^α u₁^{1−α}`.
#[derive(Clone, Copy)]
struct PowCon {
    u0: usize,
    u1: usize,
    bnd: usize,
    alpha: f64,
}

/// The smooth-NLP form of a CBF instance (VAR exp / power cones).
struct CbfNlp {
    n: usize,
    lb: Vec<f64>,
    ub: Vec<f64>,
    x0: Vec<f64>,
    c: Vec<f64>,
    /// Linear constraint rows (`(col, coeff)` pairs) with their bounds.
    lin_rows: Vec<Vec<(usize, f64)>>,
    lin_gl: Vec<f64>,
    lin_gu: Vec<f64>,
    /// Each exp constraint's variable triple `(u₀, u₁, u₂)` in CBF order.
    exp: Vec<[usize; 3]>,
    /// Power cones (each → two NLP constraints `φ ∓ x_bnd ≥ 0`).
    pow: Vec<PowCon>,
    captured_obj: RefCell<Option<f64>>,
}

impl CbfNlp {
    /// Build from a parsed model. Errors (as a panic in this test harness) if
    /// the instance uses constraint-side exp/SOC cones, which this smooth
    /// form does not cover (the CBLIB GP instances put all exp cones on
    /// variables).
    fn from_model(m: &CbfModel) -> CbfNlp {
        let n = m.num_var;
        let mut lb = vec![-INF; n];
        let mut ub = vec![INF; n];
        let mut exp = Vec::new();
        let mut pow = Vec::new();

        // Variable cones → bounds and exp/power constraints.
        let mut v = 0usize;
        for cone in &m.var_cones {
            match cone.kind {
                ConeKind::Free => {}
                ConeKind::Nonneg => {
                    for j in 0..cone.dim {
                        lb[v + j] = 0.0;
                    }
                }
                ConeKind::Nonpos => {
                    for j in 0..cone.dim {
                        ub[v + j] = 0.0;
                    }
                }
                ConeKind::Zero => {
                    for j in 0..cone.dim {
                        lb[v + j] = 0.0;
                        ub[v + j] = 0.0;
                    }
                }
                ConeKind::Exp => {
                    // u₁ (the middle) must be ≥ 0 for the cone domain.
                    lb[v + 1] = 0.0;
                    exp.push([v, v + 1, v + 2]);
                }
                ConeKind::Pow => {
                    // CBF (x₀,x₁,x₂): x₀^β₀ x₁^β₁ ≥ |x₂|, x₀,x₁ ≥ 0.
                    lb[v] = 0.0;
                    lb[v + 1] = 0.0;
                    pow.push(PowCon {
                        u0: v,
                        u1: v + 1,
                        bnd: v + 2,
                        alpha: cone.alpha.expect("POW cone has α"),
                    });
                }
                ConeKind::SecondOrder => panic!("SOC var cone not supported in NLP cross-check"),
            }
            v += cone.dim;
        }

        // Constraint cones → linear rows with bounds (Ax + b ∈ K ⇒ bounds on
        // Ax). All CBLIB GP constraint cones are L= / L- / L+.
        let a_rows = {
            let mut rows = vec![Vec::new(); m.num_con];
            for &(r, col, val) in &m.a {
                rows[r].push((col, val));
            }
            rows
        };
        let mut lin_rows = Vec::new();
        let mut lin_gl = Vec::new();
        let mut lin_gu = Vec::new();
        let mut r = 0usize;
        for cone in &m.con_cones {
            for i in 0..cone.dim {
                let row = r + i;
                let (gl, gu) = match cone.kind {
                    ConeKind::Zero => (-m.b[row], -m.b[row]), // Ax = −b
                    ConeKind::Nonpos => (-INF, -m.b[row]),    // Ax ≤ −b
                    ConeKind::Nonneg => (-m.b[row], INF),     // Ax ≥ −b
                    other => panic!("CON cone {other:?} not supported in NLP cross-check"),
                };
                lin_rows.push(a_rows[row].clone());
                lin_gl.push(gl);
                lin_gu.push(gu);
            }
            r += cone.dim;
        }

        // Start: exp middles and power base vars at 1 (a generic interior of
        // the cone domain), everything else at 0 — independent of the conic
        // solution.
        let mut x0 = vec![0.0; n];
        for t in &exp {
            x0[t[1]] = 1.0;
        }
        for p in &pow {
            x0[p.u0] = 1.0;
            x0[p.u1] = 1.0;
        }
        // Respect fixed (Zero) variables.
        for j in 0..n {
            if lb[j] == ub[j] {
                x0[j] = lb[j];
            }
        }

        CbfNlp {
            n,
            lb,
            ub,
            x0,
            c: m.c.clone(),
            lin_rows,
            lin_gl,
            lin_gu,
            exp,
            pow,
            captured_obj: RefCell::new(None),
        }
    }

    fn n_lin(&self) -> usize {
        self.lin_rows.len()
    }

    /// Number of NLP constraints contributed by power cones (two each).
    fn n_pow_con(&self) -> usize {
        2 * self.pow.len()
    }
}

/// Evaluate one power cone: `φ = u₀^α · u₁^{1−α}` and `∂φ/∂u₀`, `∂φ/∂u₁`.
fn pow_pieces(x: &[f64], p: &PowCon) -> (f64, f64, f64) {
    let u0 = x[p.u0].max(1e-12);
    let u1 = x[p.u1].max(1e-12);
    let phi = u0.powf(p.alpha) * u1.powf(1.0 - p.alpha);
    (phi, p.alpha * phi / u0, (1.0 - p.alpha) * phi / u1)
}

/// Evaluate one exp constraint `g = u₀ − u₁·exp(u₂/u₁)` and its pieces.
/// Returns `(g, E, r)` with `E = exp(u₂/u₁)`, `r = u₂/u₁`.
fn exp_pieces(x: &[f64], t: &[usize; 3]) -> (f64, f64, f64) {
    let (u0, u1, u2) = (x[t[0]], x[t[1]], x[t[2]]);
    let u1 = u1.max(1e-12); // guard the domain during the line search
    let r = u2 / u1;
    let e = r.exp();
    (u0 - u1 * e, e, r)
}

impl TNLP for CbfNlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        // Jacobian: linear entries + 3 per exp + 6 per power cone (3 for each
        // of the two `φ ∓ x_bnd` constraints). Hessian: 3 per exp + 3 per
        // power cone (the φ curvature over (u₀,u₁)).
        let nnz_jac: usize = self.lin_rows.iter().map(|r| r.len()).sum::<usize>()
            + 3 * self.exp.len()
            + 6 * self.pow.len();
        Some(NlpInfo {
            n: self.n as Index,
            m: (self.n_lin() + self.exp.len() + self.n_pow_con()) as Index,
            nnz_jac_g: nnz_jac as Index,
            nnz_h_lag: (3 * self.exp.len() + 3 * self.pow.len()) as Index,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.lb);
        b.x_u.copy_from_slice(&self.ub);
        let nl = self.n_lin();
        for i in 0..nl {
            b.g_l[i] = self.lin_gl[i];
            b.g_u[i] = self.lin_gu[i];
        }
        // Exp and power constraints: g ≥ 0.
        let n_nonlin = self.exp.len() + self.n_pow_con();
        for k in 0..n_nonlin {
            b.g_l[nl + k] = 0.0;
            b.g_u[nl + k] = INF;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(self.c.iter().zip(x).map(|(&ci, &xi)| ci * xi).sum())
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad.copy_from_slice(&self.c);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let nl = self.n_lin();
        for (i, row) in self.lin_rows.iter().enumerate() {
            g[i] = row.iter().map(|&(c, val)| val * x[c]).sum();
        }
        for (k, t) in self.exp.iter().enumerate() {
            g[nl + k] = exp_pieces(x, t).0;
        }
        // Power cones: two constraints each, φ − x_bnd ≥ 0 and φ + x_bnd ≥ 0.
        let pbase = nl + self.exp.len();
        for (k, p) in self.pow.iter().enumerate() {
            let (phi, _, _) = pow_pieces(x, p);
            g[pbase + 2 * k] = phi - x[p.bnd];
            g[pbase + 2 * k + 1] = phi + x[p.bnd];
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let nl = self.n_lin();
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for (r, row) in self.lin_rows.iter().enumerate() {
                    for &(c, _) in row {
                        irow[k] = r as Index;
                        jcol[k] = c as Index;
                        k += 1;
                    }
                }
                for (e, t) in self.exp.iter().enumerate() {
                    for &col in t {
                        irow[k] = (nl + e) as Index;
                        jcol[k] = col as Index;
                        k += 1;
                    }
                }
                // Power cones: each contributes rows `g₊` then `g₋`, both with
                // nonzeros at (u₀, u₁, bnd).
                let pbase = nl + self.exp.len();
                for (e, p) in self.pow.iter().enumerate() {
                    for sign in 0..2 {
                        let row = (pbase + 2 * e + sign) as Index;
                        for &col in &[p.u0, p.u1, p.bnd] {
                            irow[k] = row;
                            jcol[k] = col as Index;
                            k += 1;
                        }
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("jac needs x");
                let mut k = 0;
                for row in &self.lin_rows {
                    for &(_, val) in row {
                        values[k] = val;
                        k += 1;
                    }
                }
                for t in &self.exp {
                    let (_, e, r) = exp_pieces(x, t);
                    values[k] = 1.0; // ∂g/∂u₀
                    values[k + 1] = e * (r - 1.0); // ∂g/∂u₁
                    values[k + 2] = -e; // ∂g/∂u₂
                    k += 3;
                }
                for p in &self.pow {
                    let (_, dphi0, dphi1) = pow_pieces(x, p);
                    // g₊ = φ − x_bnd: ∂/∂u₀, ∂/∂u₁, ∂/∂bnd = −1.
                    values[k] = dphi0;
                    values[k + 1] = dphi1;
                    values[k + 2] = -1.0;
                    // g₋ = φ + x_bnd: same φ grads, ∂/∂bnd = +1.
                    values[k + 3] = dphi0;
                    values[k + 4] = dphi1;
                    values[k + 5] = 1.0;
                    k += 6;
                }
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        // Objective is linear and linear constraints have no Hessian, so only
        // the exp and power constraints contribute. Exp: λ·∇²g over (u₁,u₂).
        // Power: (λ₊+λ₋)·∇²φ over (u₀,u₁).
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for t in &self.exp {
                    let (_, u1, u2) = (t[0], t[1], t[2]);
                    irow[k] = u1 as Index;
                    jcol[k] = u1 as Index;
                    irow[k + 1] = u2 as Index;
                    jcol[k + 1] = u1 as Index;
                    irow[k + 2] = u2 as Index;
                    jcol[k + 2] = u2 as Index;
                    k += 3;
                }
                for p in &self.pow {
                    // u₀ < u₁ (consecutive), so the cross term is row u₁, col u₀.
                    irow[k] = p.u0 as Index;
                    jcol[k] = p.u0 as Index;
                    irow[k + 1] = p.u1 as Index;
                    jcol[k + 1] = p.u0 as Index;
                    irow[k + 2] = p.u1 as Index;
                    jcol[k + 2] = p.u1 as Index;
                    k += 3;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("hess needs x");
                let lambda = lambda.expect("hess needs lambda");
                let nl = self.n_lin();
                let mut k = 0;
                for (e, t) in self.exp.iter().enumerate() {
                    let (_, ev, r) = exp_pieces(x, t);
                    let u1 = x[t[1]].max(1e-12);
                    let lam = lambda[nl + e];
                    // ∇²g over (u₁,u₂): [[−E r²/u₁, E r/u₁],[E r/u₁, −E/u₁]].
                    values[k] = lam * (-ev * r * r / u1); // (u₁,u₁)
                    values[k + 1] = lam * (ev * r / u1); // (u₂,u₁)
                    values[k + 2] = lam * (-ev / u1); // (u₂,u₂)
                    k += 3;
                }
                let pbase = nl + self.exp.len();
                for (e, p) in self.pow.iter().enumerate() {
                    let (phi, _, _) = pow_pieces(x, p);
                    let u0 = x[p.u0].max(1e-12);
                    let u1 = x[p.u1].max(1e-12);
                    let a = p.alpha;
                    // Both g₊ and g₋ share the Hessian ∇²φ (the ∓x_bnd term is
                    // linear), so the multipliers add.
                    let lam = lambda[pbase + 2 * e] + lambda[pbase + 2 * e + 1];
                    values[k] = lam * (a * (a - 1.0) * phi / (u0 * u0)); // (u₀,u₀)
                    values[k + 1] = lam * (a * (1.0 - a) * phi / (u0 * u1)); // (u₁,u₀)
                    values[k + 2] = lam * (-a * (1.0 - a) * phi / (u1 * u1)); // (u₁,u₁)
                    k += 3;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.captured_obj.borrow_mut() = Some(sol.obj_value);
    }
}

/// Solve the conic form; return `(status, cbf_objective)`.
fn solve_conic(m: &CbfModel) -> (QpStatus, f64) {
    let cp = m.to_conic().expect("to_conic");
    let opts = QpOptions {
        max_iter: 500,
        ..QpOptions::default()
    };
    let sol = solve_socp_ipm(&cp.prob, &cp.cones, &opts, backend);
    (sol.status, cp.cbf_objective(sol.obj, m.minimize))
}

/// Solve the smooth-NLP form; return its objective (CBF sense).
fn solve_nlp(m: &CbfModel) -> f64 {
    let nlp = CbfNlp::from_model(m);
    let mut app = IpoptApplication::new();
    app.initialize().expect("init");
    let _ = app
        .options_mut()
        .read_from_str("print_level 0\nmax_iter 1000\n", true);
    let rc = Rc::new(RefCell::new(nlp));
    let tnlp: Rc<RefCell<dyn TNLP>> = rc.clone();
    let status = app.optimize_tnlp(Rc::clone(&tnlp));
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "NLP solve failed: {status:?}"
    );
    let obj = rc.borrow().captured_obj.borrow().expect("obj");
    // NLP minimized cᵀx; add the CBF constant (and flip sign for MAX).
    let cp = m.to_conic().expect("to_conic");
    cp.cbf_objective(obj, m.minimize)
}

fn cross_check(label: &str, text: &str) {
    let m = cbf::parse(text).expect("parse");
    let (status, conic_obj) = solve_conic(&m);
    assert_eq!(status, QpStatus::Optimal, "{label}: conic status");
    let nlp_obj = solve_nlp(&m);
    let rel = (conic_obj - nlp_obj).abs() / (1.0 + nlp_obj.abs());
    eprintln!("[{label}] conic={conic_obj:.8}  nlp={nlp_obj:.8}  rel={rel:.2e}");
    assert!(
        rel < 1e-5,
        "{label}: conic {conic_obj} vs nlp {nlp_obj} (rel {rel:.2e})"
    );
}

#[test]
fn demb761_conic_matches_nlp() {
    cross_check("demb761", include_str!("data/cblib/demb761.cbf"));
}

#[test]
fn beck751_conic_matches_nlp() {
    cross_check("beck751", include_str!("data/cblib/beck751.cbf"));
}

#[test]
fn fang88_conic_matches_nlp() {
    cross_check("fang88", include_str!("data/cblib/fang88.cbf"));
}

#[test]
fn power_cone_conic_matches_nlp() {
    // The synthetic power-cone instance: conic (ConeSpec::Power) vs the
    // smooth |x| ≤ y^α z^{1−α} epigraph NLP. Both should hit x2 = 1.
    cross_check("pow3", include_str!("data/cblib/pow3_synthetic.cbf"));
}
