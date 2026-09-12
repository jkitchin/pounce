//! gh#936 scoping experiment: can soft reduced-Hessian modes be extracted
//! at sparse scale?
//!
//! # The question
//!
//! gh#936 proposes drawing `--minima` escape moves along the soft modes of
//! the reduced Hessian instead of isotropically in raw `x`-space, and names
//! the blocker itself: `compute_reduced_hessian_eigen` forms a dense `n×n`
//! matrix (`n` back-solves) and runs cyclic Jacobi on it (`O(n³)`), which is
//! why `minima/mod.rs` caps the PSD check at `PSD_MAX_N = 256`. The issue
//! asks for the matvec to be scoped *before* any hop logic is written:
//! "if the matvec is not cheaply available the whole idea may not pay for
//! itself."
//!
//! # What this measures
//!
//! Two facts the issue did not have, both of which change the answer:
//!
//! 1. **`B K⁻¹ Bᵀ` is the INVERSE of the reduced Hessian**, not the reduced
//!    Hessian. `reduced_hessian.rs`'s own unit test pins it: for
//!    `K = tridiag(-1,2,-1)` on rows `{0,2}` it returns `[[3/4,1/4],[1/4,3/4]]`,
//!    whose inverse `[[3/2,-1/2],[-1/2,3/2]]` is exactly the Schur complement
//!    of `K` onto that block. So the **soft** modes of the true reduced
//!    Hessian are the **dominant** eigenvectors of the operator we can apply —
//!    the end of the spectrum iterative methods reach first, not last.
//!
//! 2. **The matvec is one back-solve** against the factor the IPM already
//!    holds: scatter into the `x` block, `kkt_solve`, gather the `x` block.
//!    No refactorization (`PdSensBacksolver` retains the converged factor),
//!    and `kkt_solve_many` batches it.
//!
//! So this program runs a Lanczos iteration whose only operation is
//! `kkt_solve`, and reports **matvecs to converge the `k` softest modes** as
//! `n` grows. That is the number the whole idea lives or dies on: if it grows
//! with `n`, the approach is no better than the dense path; if it is flat,
//! soft modes cost a fixed handful of back-solves at any scale.
//!
//! # Why this fixture
//!
//! The 1-D chain `f = ½κ Σ(x_{i+1}-x_i)² + ½ε Σx_i² - bᵀx` has Hessian
//! `H = κL + εI` with `L = tridiag(-1,2,-1)`, whose eigenpairs are known in
//! closed form:
//!
//! ```text
//! λ_k = 4κ sin²(kπ / 2(n+1)) + ε      v_k(i) = sin((i+1)kπ / (n+1))
//! ```
//!
//! Three reasons that matters here. It is genuinely sparse. Its spectrum
//! spans `O(n²)` — a real sloppy cascade, which is the regime gh#936 is
//! about, rather than a synthetic one. And crucially **the ground truth is
//! analytic at every `n`**, so correctness can be checked at 100 000
//! variables rather than only below the 256-variable dense cap. A study that
//! could only validate where the dense path already works would be answering
//! the wrong question — that is the `CLAUDE.md` branch rule, and this fixture
//! is chosen to dodge it.
//!
//! Its soft modes are the long-wavelength modes of the chain, which is also
//! the cleanest possible instance of the collective-variable analogy the
//! issue borrows from MD.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p pounce-sensitivity --example soft_mode_scaling
//! ```

use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;
use std::time::Instant;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint,
};
use pounce_sensitivity::Solver;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// `min ½κ Σ_{i=0}^{n-2}(x_{i+1}-x_i)² + ½κ(x_0² + x_{n-1}²) + ½ε Σ x_i² - bᵀx`
///
/// The two endpoint terms impose Dirichlet conditions, which is what makes
/// the Hessian exactly `κ·tridiag(-1,2,-1) + εI` and so gives the analytic
/// eigenpairs quoted in the module docs.
///
/// `bounds` optionally puts a box on the first `n_bounded` variables tight
/// enough to be active at the solution, so the barrier `Σ = z/s` term is
/// live in the KKT factor.
struct ChainTnlp {
    n: usize,
    kappa: Number,
    eps: Number,
    b: Vec<Number>,
    /// Upper bound applied to the first `n_bounded` variables.
    n_bounded: usize,
    bound_u: Number,
}

impl ChainTnlp {
    fn new(n: usize, kappa: Number, eps: Number, n_bounded: usize, bound_u: Number) -> Self {
        // A right-hand side with content at every wavelength, so no mode is
        // accidentally absent from the solution.
        let b = (0..n)
            .map(|i| {
                let t = (i as Number + 1.0) / (n as Number + 1.0);
                (2.0 * PI * t).sin() + 0.3 * (17.0 * PI * t).sin() + 0.1
            })
            .collect();
        Self {
            n,
            kappa,
            eps,
            b,
            n_bounded,
            bound_u,
        }
    }

    /// Analytic eigenvalue `k` (1-indexed) of `H = κL + εI`, ascending in `k`.
    fn analytic_eigenvalue(&self, k: usize) -> Number {
        let s = (k as Number * PI / (2.0 * (self.n as Number + 1.0))).sin();
        4.0 * self.kappa * s * s + self.eps
    }

    /// Analytic eigenvector `k` (1-indexed), normalized.
    fn analytic_eigenvector(&self, k: usize) -> Vec<Number> {
        let n = self.n;
        let mut v: Vec<Number> = (0..n)
            .map(|i| ((i as Number + 1.0) * k as Number * PI / (n as Number + 1.0)).sin())
            .collect();
        let nrm = v.iter().map(|a| a * a).sum::<Number>().sqrt();
        v.iter_mut().for_each(|a| *a /= nrm);
        v
    }
}

impl TNLP for ChainTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as Index,
            m: 0,
            nnz_jac_g: 0,
            // Lower triangle of a tridiagonal matrix: n diagonal + (n-1) sub.
            nnz_h_lag: (2 * self.n - 1) as Index,
            index_style: IndexStyle::C,
        })
    }
    fn get_scaling_parameters(&mut self, _r: ScalingRequest<'_>) -> bool {
        false
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for i in 0..self.n {
            b.x_l[i] = -1.0e19;
            b.x_u[i] = if i < self.n_bounded {
                self.bound_u
            } else {
                1.0e19
            };
        }
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.iter_mut().for_each(|v| *v = 0.0);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        let mut f = 0.5 * self.kappa * (x[0] * x[0] + x[self.n - 1] * x[self.n - 1]);
        for i in 0..self.n - 1 {
            let d = x[i + 1] - x[i];
            f += 0.5 * self.kappa * d * d;
        }
        for i in 0..self.n {
            f += 0.5 * self.eps * x[i] * x[i] - self.b[i] * x[i];
        }
        Some(f)
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        // g = (κL + εI) x - b, with L = tridiag(-1, 2, -1).
        for i in 0..self.n {
            let left = if i == 0 { 0.0 } else { x[i - 1] };
            let right = if i + 1 == self.n { 0.0 } else { x[i + 1] };
            g[i] = self.kappa * (2.0 * x[i] - left - right) + self.eps * x[i] - self.b[i];
        }
        true
    }
    fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _mode: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for i in 0..self.n {
                    irow[k] = i as Index;
                    jcol[k] = i as Index;
                    k += 1;
                }
                for i in 0..self.n - 1 {
                    irow[k] = (i + 1) as Index;
                    jcol[k] = i as Index;
                    k += 1;
                }
            }
            SparsityRequest::Values { values } => {
                let mut k = 0;
                for _ in 0..self.n {
                    values[k] = obj * (2.0 * self.kappa + self.eps);
                    k += 1;
                }
                for _ in 0..self.n - 1 {
                    values[k] = obj * (-self.kappa);
                    k += 1;
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

// ---------------------------------------------------------------------------
// The operator: one back-solve per matvec
// ---------------------------------------------------------------------------

/// `v ↦ (K⁻¹)_xx v` — the `x`-block of the KKT inverse, which is the inverse
/// of the barrier-problem reduced Hessian. Counts its own back-solves.
struct KktXOperator<'a> {
    solver: &'a Solver,
    n_x: usize,
    kkt_dim: usize,
    matvecs: std::cell::Cell<usize>,
    rhs: RefCell<Vec<Number>>,
    lhs: RefCell<Vec<Number>>,
}

impl<'a> KktXOperator<'a> {
    fn new(solver: &'a Solver) -> Self {
        let kkt_dim = solver.kkt_dim().expect("converged factor");
        let n_x = solver.block_dims().expect("block dims")[0];
        Self {
            solver,
            n_x,
            kkt_dim,
            matvecs: std::cell::Cell::new(0),
            rhs: RefCell::new(vec![0.0; kkt_dim]),
            lhs: RefCell::new(vec![0.0; kkt_dim]),
        }
    }

    fn apply(&self, v: &[Number], out: &mut [Number]) {
        let mut rhs = self.rhs.borrow_mut();
        let mut lhs = self.lhs.borrow_mut();
        rhs.iter_mut().for_each(|r| *r = 0.0);
        rhs[..self.n_x].copy_from_slice(&v[..self.n_x]);
        self.solver
            .kkt_solve(&rhs, &mut lhs)
            .expect("kkt_solve against the converged factor");
        out[..self.n_x].copy_from_slice(&lhs[..self.n_x]);
        self.matvecs.set(self.matvecs.get() + 1);
        let _ = self.kkt_dim;
    }
}

// ---------------------------------------------------------------------------
// Lanczos with full reorthogonalization
// ---------------------------------------------------------------------------

fn dot(a: &[Number], b: &[Number]) -> Number {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[Number]) -> Number {
    dot(a, a).sqrt()
}

/// Symmetric tridiagonal eigendecomposition by QL implicit shifts, enough for
/// the small `m×m` Lanczos projection (`m` is tens, never `n`).
/// Returns `(eigenvalues ascending, eigenvectors column-major m×m)`.
fn tridiag_eigen(alpha: &[Number], beta: &[Number]) -> (Vec<Number>, Vec<Number>) {
    let m = alpha.len();
    let mut d = alpha.to_vec();
    let mut e = vec![0.0; m];
    e[..m - 1].copy_from_slice(&beta[..m - 1]);
    e[m - 1] = 0.0;
    let mut z = vec![0.0; m * m];
    for i in 0..m {
        z[i * m + i] = 1.0;
    }
    for l in 0..m {
        let mut iter = 0;
        loop {
            let mut mm = l;
            while mm + 1 < m {
                let dd = d[mm].abs() + d[mm + 1].abs();
                if e[mm].abs() <= f64::EPSILON * dd {
                    break;
                }
                mm += 1;
            }
            if mm == l {
                break;
            }
            iter += 1;
            if iter > 50 {
                break;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[mm] - d[l] + e[l] / (g + if g >= 0.0 { r.abs() } else { -r.abs() });
            let (mut s, mut c) = (1.0, 1.0);
            let mut p = 0.0;
            let mut i = mm;
            while i > l {
                let i1 = i - 1;
                let mut f = s * e[i1];
                let bb = c * e[i1];
                r = f.hypot(g);
                e[i] = r;
                if r == 0.0 {
                    d[i] -= p;
                    e[mm] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i] - p;
                r = (d[i1] - g) * s + 2.0 * c * bb;
                p = s * r;
                d[i] = g + p;
                g = c * r - bb;
                for k in 0..m {
                    f = z[i * m + k];
                    z[i * m + k] = s * z[i1 * m + k] + c * f;
                    z[i1 * m + k] = c * z[i1 * m + k] - s * f;
                }
                i -= 1;
            }
            if r == 0.0 && i > l {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[mm] = 0.0;
        }
    }
    // Sort ascending, carrying the columns.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());
    let vals: Vec<Number> = order.iter().map(|&i| d[i]).collect();
    let mut vecs = vec![0.0; m * m];
    for (newj, &oldj) in order.iter().enumerate() {
        for i in 0..m {
            vecs[newj * m + i] = z[oldj * m + i];
        }
    }
    (vals, vecs)
}

struct LanczosResult {
    /// Ritz values of the operator, descending (dominant first).
    ritz_values: Vec<Number>,
    /// Ritz vectors, column-major `n_x × k`, matching `ritz_values`.
    ritz_vectors: Vec<Number>,
    matvecs: usize,
    /// Seconds spent in back-solves.
    matvec_secs: f64,
    /// Seconds spent in full reorthogonalization — this program's own
    /// bookkeeping, `O(m²n)`, not a cost of the approach.
    reorth_secs: f64,
}

/// Lanczos on `op` for the `k` **dominant** eigenpairs — which, `op` being the
/// inverse reduced Hessian, are the `k` **softest** modes.
///
/// Full reorthogonalization: `m` is tens, so the `O(m·n)` cost per step is
/// negligible next to a back-solve, and it removes the spurious-copy failure
/// mode entirely.
fn lanczos_dominant(op: &KktXOperator, k: usize, m_max: usize, seed: u64) -> LanczosResult {
    let n = op.n_x;
    let mut rng = seed;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 11) as Number) / ((1u64 << 53) as Number) - 0.5
    };

    let mut q: Vec<Vec<Number>> = Vec::with_capacity(m_max);
    let mut v: Vec<Number> = (0..n).map(|_| next()).collect();
    let nrm = norm(&v);
    v.iter_mut().for_each(|a| *a /= nrm);
    q.push(v);

    let mut alpha: Vec<Number> = Vec::new();
    let mut beta: Vec<Number> = Vec::new();
    let mut w = vec![0.0; n];
    let mut matvec_secs = 0.0f64;
    let mut reorth_secs = 0.0f64;

    let m = m_max.min(n);
    for j in 0..m {
        let t_mv = Instant::now();
        op.apply(&q[j], &mut w);
        matvec_secs += t_mv.elapsed().as_secs_f64();
        let a = dot(&w, &q[j]);
        alpha.push(a);
        // Full reorthogonalization against every stored Lanczos vector.
        let t_re = Instant::now();
        for _ in 0..2 {
            for qi in q.iter() {
                let c = dot(&w, qi);
                for (wi, qv) in w.iter_mut().zip(qi) {
                    *wi -= c * qv;
                }
            }
        }
        reorth_secs += t_re.elapsed().as_secs_f64();
        let b = norm(&w);
        if j + 1 < m {
            beta.push(b);
            if b <= 1e-14 {
                break;
            }
            let mut qn = w.clone();
            qn.iter_mut().for_each(|a| *a /= b);
            q.push(qn);
        }
    }

    let mm = alpha.len();
    let (vals, vecs) = tridiag_eigen(&alpha, &beta);

    // Dominant end = largest Ritz values = softest modes of the true H.
    let kk = k.min(mm);
    let mut ritz_values = Vec::with_capacity(kk);
    let mut ritz_vectors = vec![0.0; n * kk];
    for c in 0..kk {
        let src = mm - 1 - c; // descending
        ritz_values.push(vals[src]);
        for i in 0..n {
            let mut acc = 0.0;
            for (jj, qj) in q.iter().enumerate().take(mm) {
                acc += vecs[src * mm + jj] * qj[i];
            }
            ritz_vectors[c * n + i] = acc;
        }
    }
    LanczosResult {
        ritz_values,
        ritz_vectors,
        matvecs: op.matvecs.get(),
        matvec_secs,
        reorth_secs,
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn solve_chain(tnlp: ChainTnlp) -> (Solver, Rc<RefCell<ChainTnlp>>, f64) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-10, true, false)
        .unwrap();
    app.initialize().unwrap();
    let rc = Rc::new(RefCell::new(tnlp));
    let dynamic: Rc<RefCell<dyn TNLP>> = rc.clone();
    let mut s = Solver::new(app, dynamic);
    let t0 = Instant::now();
    let st = s.solve();
    let solve_secs = t0.elapsed().as_secs_f64();
    assert!(
        matches!(
            st,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "chain solve failed: {st:?}"
    );
    (s, rc, solve_secs)
}

/// `|cos∠|` between a Ritz vector and the analytic mode, and the subspace
/// angle of the recovered `k`-dimensional space against the analytic one.
fn subspace_alignment(
    ritz: &[Number],
    n: usize,
    k: usize,
    analytic: &[Vec<Number>],
) -> (Vec<Number>, Number) {
    let mut per_mode = Vec::with_capacity(k);
    for c in 0..k {
        let col = &ritz[c * n..c * n + n];
        let cn = norm(col);
        let mut best: Number = 0.0;
        for a in analytic.iter() {
            best = best.max(dot(col, a).abs() / cn);
        }
        per_mode.push(best);
    }
    // Worst-case: how much of each analytic mode is captured by the full
    // recovered subspace (Gram-Schmidt the Ritz block, then project).
    let mut basis: Vec<Vec<Number>> = Vec::new();
    for c in 0..k {
        let mut col = ritz[c * n..c * n + n].to_vec();
        for bvec in basis.iter() {
            let d = dot(&col, bvec);
            for (ci, bi) in col.iter_mut().zip(bvec) {
                *ci -= d * bi;
            }
        }
        let nr = norm(&col);
        if nr > 1e-12 {
            col.iter_mut().for_each(|a| *a /= nr);
            basis.push(col);
        }
    }
    let mut worst: Number = 1.0;
    for a in analytic.iter().take(k) {
        let captured: Number = basis.iter().map(|b| dot(a, b).powi(2)).sum::<Number>();
        worst = worst.min(captured.sqrt());
    }
    (per_mode, worst)
}

fn run_case(n: usize, k: usize, m_max: usize, n_bounded: usize, eps: Number) {
    let kappa = 1.0;
    let bound_u = 0.05;
    let tnlp = ChainTnlp::new(n, kappa, eps, n_bounded, bound_u);
    let analytic_vals: Vec<Number> = (1..=k).map(|kk| tnlp.analytic_eigenvalue(kk)).collect();
    let analytic_vecs: Vec<Vec<Number>> = (1..=k).map(|kk| tnlp.analytic_eigenvector(kk)).collect();
    let stiffest = tnlp.analytic_eigenvalue(n);

    let (solver, _rc, solve_secs) = solve_chain(tnlp);

    let op = KktXOperator::new(&solver);
    let t0 = Instant::now();
    let res = lanczos_dominant(&op, k, m_max, 0x9E3779B97F4A7C15);
    let lanczos_secs = t0.elapsed().as_secs_f64();

    // A Ritz value of the inverse is 1/λ of the reduced Hessian.
    let recovered: Vec<Number> = res.ritz_values.iter().map(|r| 1.0 / r).collect();

    let (per_mode, worst) = subspace_alignment(&res.ritz_vectors, op.n_x, k, &analytic_vecs);

    let bd = solver.block_dims().expect("block dims");
    println!(
        "n = {n:>7}  (bounded vars: {n_bounded})   kkt blocks (x,s,y_c,y_d,z_l,z_u,v_l,v_u) = {bd:?}"
    );
    println!(
        "  NLP solve {:>8.3}s | {} Lanczos matvecs (= back-solves) in {:>7.3}s  ({:>7.1} µs each)",
        solve_secs,
        res.matvecs,
        lanczos_secs,
        1e6 * lanczos_secs / res.matvecs as f64
    );
    println!(
        "    of which back-solves {:>7.3}s, full reorthogonalization {:>7.3}s (this program's O(m^2 n) bookkeeping)",
        res.matvec_secs, res.reorth_secs
    );
    println!(
        "  back-solve budget as a fraction of one NLP solve: {:>6.2}%",
        100.0 * lanczos_secs / solve_secs
    );
    println!(
        "  condition number of H: {:.3e}  (softest {:.3e} .. stiffest {:.3e})",
        stiffest / analytic_vals[0],
        analytic_vals[0],
        stiffest
    );
    println!(
        "  soft-end relative gap (l2-l1)/l1: {:.3e}   [eps floor = {:.1e}]",
        (analytic_vals[1] - analytic_vals[0]) / analytic_vals[0],
        eps
    );
    if n_bounded == 0 {
        println!("  softest {k} eigenvalues:  recovered vs analytic");
        for i in 0..k.min(6) {
            let rel = (recovered[i] - analytic_vals[i]).abs() / analytic_vals[i];
            println!(
                "    mode {:>2}: {:>13.6e}  vs {:>13.6e}   rel err {:>9.2e}   |cos∠| {:>8.6}",
                i + 1,
                recovered[i],
                analytic_vals[i],
                rel,
                per_mode[i]
            );
        }
        println!("  worst-case subspace capture over the {k} softest modes: {worst:.6}");
    } else {
        println!(
            "  (bounds active — analytic modes no longer the ground truth; reporting spectrum)"
        );
        for i in 0..k.min(6) {
            println!(
                "    mode {:>2}: reduced-Hessian eigenvalue {:>13.6e}",
                i + 1,
                recovered[i]
            );
        }
        // gh#936 design note 3 asks whether a soft mode pointing into an
        // active bound has to be projected out or dropped by hand. An active
        // bound puts Sigma = z/s (large) on that x diagonal, so (K^-1)_xx is
        // *small* there and the dominant end should avoid it with no help.
        // Measure it: the share of each mode's squared norm carried by the
        // actively-bounded coordinates, against the share a direction with no
        // such preference would carry (n_bounded / n).
        let n_x = op.n_x;
        let neutral = n_bounded as Number / n_x as Number;
        println!(
            "  share of mode norm^2 on the {n_bounded} bounded coords (a direction with no preference would carry {neutral:.4}):"
        );
        for i in 0..k.min(6) {
            let col = &res.ritz_vectors[i * n_x..i * n_x + n_x];
            let tot: Number = col.iter().map(|a| a * a).sum();
            let on_b: Number = col[..n_bounded].iter().map(|a| a * a).sum();
            println!(
                "    mode {:>2}: {:>10.3e}   ({:>8.1}x less than neutral)",
                i + 1,
                on_b / tot,
                neutral / (on_b / tot).max(1e-300)
            );
        }
    }
    println!();
}

fn main() {
    println!("gh#936 — soft reduced-Hessian modes at sparse scale");
    println!("operator: v -> (K^-1)_xx v, one kkt_solve per matvec");
    println!("Lanczos targets the DOMINANT end, which is the SOFT end of H.\n");

    let k = 8;
    let m_max = 40;

    println!("=== A. fixed eps = 1e-6: the soft end collapses into the eps floor as n grows ===\n");
    for &n in &[1_000usize, 10_000, 50_000, 100_000] {
        run_case(n, k, m_max, 0, 1e-6);
    }

    println!("=== B. gap-matched (eps = 1e-12): the chain's own k^2 gaps survive at every n ===\n");
    for &n in &[1_000usize, 10_000, 50_000, 100_000] {
        run_case(n, k, m_max, 0, 1e-12);
    }

    println!("=== C. with active bounds (barrier Sigma live in the factor) ===\n");
    run_case(10_000, k, m_max, 200, 1e-12);

    println!("=== D. back-solve cost: single vs batched ===\n");
    cost_probe(100_000);

    println!("\n=== E. how many matvecs are actually needed? (n = 100000, gap-matched) ===\n");
    budget_sweep(100_000, k);
}

/// The Lanczos budget `m` was fixed at 40 above by choice, not by measurement.
/// Sweep it: the number that matters for gh#936 is the smallest `m` that
/// recovers the soft subspace, since that is what each escape move costs.
fn budget_sweep(n: usize, k: usize) {
    let tnlp = ChainTnlp::new(n, 1.0, 1e-12, 0, 0.05);
    let analytic_vecs: Vec<Vec<Number>> = (1..=k).map(|kk| tnlp.analytic_eigenvector(kk)).collect();
    let (solver, _rc, solve_secs) = solve_chain(tnlp);
    println!("  (one NLP solve on this model: {solve_secs:.3} s)");
    println!("   m   matvecs   worst-case subspace capture over the {k} softest modes");
    for &m in &[10usize, 12, 14, 16, 20, 24, 30, 40] {
        let op = KktXOperator::new(&solver);
        let res = lanczos_dominant(&op, k, m, 0x9E3779B97F4A7C15);
        let (_per, worst) = subspace_alignment(&res.ritz_vectors, op.n_x, k, &analytic_vecs);
        println!(
            "  {:>3}   {:>7}   {:>10.6}{}",
            m,
            res.matvecs,
            worst,
            if worst > 0.9999 {
                "   <-- converged"
            } else {
                ""
            }
        );
    }
}

/// Is the per-matvec cost an inherent sparse back-solve, or wrapper overhead?
/// `kkt_solve_many` shares one factor pass across many right-hand sides, so a
/// large gap between the two is overhead, not arithmetic.
fn cost_probe(n: usize) {
    let tnlp = ChainTnlp::new(n, 1.0, 1e-12, 0, 0.05);
    let (solver, _rc, solve_secs) = solve_chain(tnlp);
    let kkt_dim = solver.kkt_dim().unwrap();
    let n_x = solver.block_dims().unwrap()[0];
    let reps = 40usize;

    // Distinct right-hand sides on both paths: a repeated identical RHS can
    // hit the solver's last-solve fast path, which would flatter the batched
    // number for a reason that has nothing to do with batching.
    let rhs_at = |r: usize, i: usize| -> Number {
        (((i + 13 * r) % 7) as Number) - 3.0 + (r as Number) * 1e-3
    };
    let mut rhs = vec![0.0; kkt_dim];
    let mut lhs = vec![0.0; kkt_dim];
    let t0 = Instant::now();
    for r in 0..reps {
        rhs.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..n_x {
            rhs[i] = rhs_at(r, i);
        }
        solver.kkt_solve(&rhs, &mut lhs).unwrap();
    }
    let single = t0.elapsed().as_secs_f64();

    let mut rhs_flat = vec![0.0; reps * kkt_dim];
    for r in 0..reps {
        for i in 0..n_x {
            rhs_flat[r * kkt_dim + i] = rhs_at(r, i);
        }
    }
    let mut lhs_flat = vec![0.0; reps * kkt_dim];
    let t0 = Instant::now();
    solver
        .kkt_solve_many(&rhs_flat, &mut lhs_flat, reps)
        .unwrap();
    let batched = t0.elapsed().as_secs_f64();

    println!("n = {n}, kkt_dim = {kkt_dim}, {reps} right-hand sides");
    println!("  one NLP solve                 : {:>8.3} s", solve_secs);
    println!(
        "  {reps} x kkt_solve (one at a time): {:>8.3} s  ({:>8.1} us each)",
        single,
        1e6 * single / reps as f64
    );
    println!(
        "  kkt_solve_many (batched)      : {:>8.3} s  ({:>8.1} us each)  speedup {:.1}x",
        batched,
        1e6 * batched / reps as f64,
        single / batched
    );
    println!(
        "  batched extraction as a fraction of one NLP solve: {:.2}%",
        100.0 * batched / solve_secs
    );
}
