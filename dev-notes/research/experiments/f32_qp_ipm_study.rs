// f32-vs-f64 convergence study for a primal-dual interior-point QP solver.
//
// Goal: answer the Phase-0 go/no-go question from
// dev-notes/research/gpu-batched-layers.md — does an IPM converge in f32
// on QP-layer-shaped problems, and what accuracy floor does f32 hit?
//
// The GPU batched-layer beachhead would run exactly this arithmetic
// (a dense condensed-KKT SPD solve, per batch element) in f32 on the
// GPU. The GPU does not change the math, so a CPU f32 run is a faithful
// proxy for the numerics. We solve the canonical OptNet/qpth convex QP
//
//     min  1/2 x^T Q x + q^T x   s.t.  G x <= h
//
// by Mehrotra predictor-corrector, condensing the Newton system to the
// SPD matrix (Q + G^T diag(z/s) G) and factoring by Cholesky — the same
// reformulation the design note proposes for the GPU kernel.
//
// Problems are generated in f64. Each is solved once in f64 and once in
// f32 (data cast down). Accuracy is always judged by recomputing the
// duality gap and KKT residual in f64 from the iterate — so we measure
// the TRUE accuracy of the f32-computed point, not f32's rounded
// self-assessment.

use std::ops::{Add, Div, Mul, Neg, Sub};

// ---- minimal Real trait so the solver is generic over f32 / f64 -------

trait Real:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
}

impl Real for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn from_f64(x: f64) -> Self { x }
    fn to_f64(self) -> f64 { self }
    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn abs(self) -> Self { f64::abs(self) }
}

impl Real for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn from_f64(x: f64) -> Self { x as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn sqrt(self) -> Self { f32::sqrt(self) }
    fn abs(self) -> Self { f32::abs(self) }
}

// ---- tiny deterministic RNG (xorshift64*) -----------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1)) }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    // uniform in [-1, 1)
    fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    // uniform in [lo, hi)
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
}

// ---- problem (stored in f64) ------------------------------------------

struct Qp {
    n: usize,
    m: usize,
    q_mat: Vec<f64>, // n*n, row-major, SPD
    q_vec: Vec<f64>, // n
    g: Vec<f64>,     // m*n, row-major
    h: Vec<f64>,     // m
    // warm interior start
    x0: Vec<f64>,
    s0: Vec<f64>,
    z0: Vec<f64>,
}

fn gen_qp(n: usize, m: usize, rng: &mut Rng) -> Qp {
    // Q = (1/n) M^T M + I  (SPD, reasonably conditioned)
    let mut mmat = vec![0.0f64; n * n];
    for v in mmat.iter_mut() { *v = rng.unif(); }
    let mut q_mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n { acc += mmat[k * n + i] * mmat[k * n + j]; }
            q_mat[i * n + j] = acc / n as f64 + if i == j { 1.0 } else { 0.0 };
        }
    }
    let q_vec: Vec<f64> = (0..n).map(|_| rng.unif()).collect();
    let g: Vec<f64> = (0..m * n).map(|_| rng.unif()).collect();
    let x0: Vec<f64> = (0..n).map(|_| rng.unif()).collect();
    let s0: Vec<f64> = (0..m).map(|_| rng.range(0.5, 1.5)).collect();
    let z0: Vec<f64> = (0..m).map(|_| rng.range(0.5, 1.5)).collect();
    // h = G x0 + s0  =>  G x0 < h, strictly feasible interior start
    let mut h = vec![0.0f64; m];
    for i in 0..m {
        let mut acc = 0.0;
        for j in 0..n { acc += g[i * n + j] * x0[j]; }
        h[i] = acc + s0[i];
    }
    Qp { n, m, q_mat, q_vec, g, h, x0, s0, z0 }
}

// ---- dense Cholesky solve for SPD K (n*n, row-major), in-place rhs ----
// Returns false if a non-positive pivot is hit (factorization breakdown).

fn chol_solve<R: Real>(n: usize, k: &mut [R], rhs: &mut [R]) -> bool {
    // factor K = L L^T, store L in lower triangle of k
    for j in 0..n {
        let mut diag = k[j * n + j];
        for p in 0..j { diag = diag - k[j * n + p] * k[j * n + p]; }
        if !(diag.to_f64() > 0.0) { return false; }
        let ljj = diag.sqrt();
        k[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut s = k[i * n + j];
            for p in 0..j { s = s - k[i * n + p] * k[j * n + p]; }
            k[i * n + j] = s / ljj;
        }
    }
    // forward solve L y = rhs
    for i in 0..n {
        let mut s = rhs[i];
        for p in 0..i { s = s - k[i * n + p] * rhs[p]; }
        rhs[i] = s / k[i * n + i];
    }
    // back solve L^T x = y
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for p in (i + 1)..n { s = s - k[p * n + i] * rhs[p]; }
        rhs[i] = s / k[i * n + i];
    }
    true
}

struct SolveStats {
    converged: bool,
    iters: usize,
    best_gap: f64,   // smallest TRUE duality gap reached (f64-recomputed)
    resid_at_best: f64, // TRUE KKT residual (inf-norm) at that point
    breakdown: bool, // Cholesky breakdown / NaN
    // final iterate (f64), for warm-starting a refinement pass
    xf: Vec<f64>,
    sf: Vec<f64>,
    zf: Vec<f64>,
}

// ---- the IPM, generic over R ------------------------------------------

fn solve_qp<R: Real>(p: &Qp, gap_tol: f64, max_iter: usize) -> SolveStats {
    solve_qp_warm::<R>(p, gap_tol, max_iter, None)
}

fn solve_qp_warm<R: Real>(
    p: &Qp,
    gap_tol: f64,
    max_iter: usize,
    warm: Option<(&[f64], &[f64], &[f64])>,
) -> SolveStats {
    let n = p.n;
    let m = p.m;
    let c = |x: f64| R::from_f64(x);

    let q_mat: Vec<R> = p.q_mat.iter().map(|&v| c(v)).collect();
    let q_vec: Vec<R> = p.q_vec.iter().map(|&v| c(v)).collect();
    let g: Vec<R> = p.g.iter().map(|&v| c(v)).collect();
    let h: Vec<R> = p.h.iter().map(|&v| c(v)).collect();

    let (mut x, mut s, mut z): (Vec<R>, Vec<R>, Vec<R>) = match warm {
        Some((xw, sw, zw)) => (
            xw.iter().map(|&v| c(v)).collect(),
            sw.iter().map(|&v| c(v)).collect(),
            zw.iter().map(|&v| c(v)).collect(),
        ),
        None => (
            p.x0.iter().map(|&v| c(v)).collect(),
            p.s0.iter().map(|&v| c(v)).collect(),
            p.z0.iter().map(|&v| c(v)).collect(),
        ),
    };

    let mut best_gap = f64::INFINITY;
    let mut resid_at_best = f64::INFINITY;
    let mut converged = false;
    let mut iters_used = 0;

    for it in 0..max_iter {
        iters_used = it + 1;

        // residuals in R
        // r_d = Q x + q + G^T z
        let mut r_d = vec![R::zero(); n];
        for i in 0..n {
            let mut acc = q_vec[i];
            for j in 0..n { acc = acc + q_mat[i * n + j] * x[j]; }
            for k in 0..m { acc = acc + g[k * n + i] * z[k]; }
            r_d[i] = acc;
        }
        // r_p = G x + s - h
        let mut r_p = vec![R::zero(); m];
        for i in 0..m {
            let mut acc = s[i] - h[i];
            for j in 0..n { acc = acc + g[i * n + j] * x[j]; }
            r_p[i] = acc;
        }
        // mu = s.z / m
        let mut sz = R::zero();
        for i in 0..m { sz = sz + s[i] * z[i]; }
        let mu = sz / R::from_f64(m as f64);

        // --- TRUE accuracy check (recompute in f64 from current iterate) ---
        let (true_gap, true_resid) = true_metrics(p, &x, &s, &z);
        if true_gap < best_gap {
            best_gap = true_gap;
            resid_at_best = true_resid;
        }
        if true_gap < gap_tol && true_resid < (gap_tol * 100.0).max(1e-6) {
            converged = true;
            break;
        }

        // W = z/s  (diagonal)
        let mut w = vec![R::zero(); m];
        for i in 0..m { w[i] = z[i] / s[i]; }

        // condensed matrix K = Q + G^T W G   (n*n)
        let mut kmat = vec![R::zero(); n * n];
        for a in 0..n {
            for b in 0..n {
                let mut acc = q_mat[a * n + b];
                for r in 0..m { acc = acc + g[r * n + a] * w[r] * g[r * n + b]; }
                kmat[a * n + b] = acc;
            }
        }

        // ---- affine (predictor): r_c = s.z (sigma = 0) ----
        // ds = -r_p - G dx ; dz = -W*ds - z   (from S Z e residual, r_caff = s*z)
        // Build rhs_x = -r_d - G^T ( W r_p - r_caff/s )
        // with r_caff_i = s_i z_i  => r_caff/s = z. So term = W r_p - z.
        let mut rhs = vec![R::zero(); n];
        for a in 0..n {
            let mut acc = -r_d[a];
            for r in 0..m {
                let t = w[r] * r_p[r] - z[r];
                acc = acc - g[r * n + a] * t;
            }
            rhs[a] = acc;
        }
        let mut kfac = kmat.clone();
        if !chol_solve(n, &mut kfac, &mut rhs) {
            return SolveStats { converged, iters: iters_used, best_gap, resid_at_best, breakdown: true,
                xf: x.iter().map(|v| v.to_f64()).collect(),
                sf: s.iter().map(|v| v.to_f64()).collect(),
                zf: z.iter().map(|v| v.to_f64()).collect() };
        }
        let dx_aff = rhs.clone();
        // ds_aff = -r_p - G dx_aff
        let mut ds_aff = vec![R::zero(); m];
        for i in 0..m {
            let mut acc = -r_p[i];
            for j in 0..n { acc = acc - g[i * n + j] * dx_aff[j]; }
            ds_aff[i] = acc;
        }
        // dz_aff from complementarity affine: z + (z/s) ds + dz = 0 -> dz = -z - W ds
        let mut dz_aff = vec![R::zero(); m];
        for i in 0..m { dz_aff[i] = -z[i] - w[i] * ds_aff[i]; }

        // affine step length (fraction to boundary, tau=1 for the probe)
        let a_aff = step_len(&s, &ds_aff, &z, &dz_aff, R::one());
        // mu_aff
        let mut sz_aff = R::zero();
        for i in 0..m {
            sz_aff = sz_aff + (s[i] + a_aff * ds_aff[i]) * (z[i] + a_aff * dz_aff[i]);
        }
        let mu_aff = sz_aff / R::from_f64(m as f64);
        // sigma = (mu_aff/mu)^3
        let ratio = (mu_aff / mu).to_f64().max(0.0);
        let sigma = R::from_f64(ratio * ratio * ratio);

        // ---- corrector: r_c = s.z + ds_aff.dz_aff - sigma*mu*e ----
        // r_c/s term in rhs: (r_c)/s = z + (ds_aff*dz_aff - sigma*mu)/s
        let mut rhs2 = vec![R::zero(); n];
        for a in 0..n {
            let mut acc = -r_d[a];
            for r in 0..m {
                let rc = s[r] * z[r] + ds_aff[r] * dz_aff[r] - sigma * mu;
                let rc_over_s = rc / s[r];
                let t = w[r] * r_p[r] - rc_over_s;
                acc = acc - g[r * n + a] * t;
            }
            rhs2[a] = acc;
        }
        let mut kfac2 = kmat;
        if !chol_solve(n, &mut kfac2, &mut rhs2) {
            return SolveStats { converged, iters: iters_used, best_gap, resid_at_best, breakdown: true,
                xf: x.iter().map(|v| v.to_f64()).collect(),
                sf: s.iter().map(|v| v.to_f64()).collect(),
                zf: z.iter().map(|v| v.to_f64()).collect() };
        }
        let dx = rhs2.clone();
        let mut ds = vec![R::zero(); m];
        for i in 0..m {
            let mut acc = -r_p[i];
            for j in 0..n { acc = acc - g[i * n + j] * dx[j]; }
            ds[i] = acc;
        }
        let mut dz = vec![R::zero(); m];
        for i in 0..m {
            let rc = s[i] * z[i] + ds_aff[i] * dz_aff[i] - sigma * mu;
            // dz = -(rc + z*ds)/s
            dz[i] = -(rc + z[i] * ds[i]) / s[i];
        }

        let tau = R::from_f64(0.95);
        let alpha = step_len(&s, &ds, &z, &dz, tau);

        // check for NaN/inf
        if !alpha.to_f64().is_finite() {
            return SolveStats { converged, iters: iters_used, best_gap, resid_at_best, breakdown: true,
                xf: x.iter().map(|v| v.to_f64()).collect(),
                sf: s.iter().map(|v| v.to_f64()).collect(),
                zf: z.iter().map(|v| v.to_f64()).collect() };
        }

        for j in 0..n { x[j] = x[j] + alpha * dx[j]; }
        for i in 0..m { s[i] = s[i] + alpha * ds[i]; z[i] = z[i] + alpha * dz[i]; }
    }

    SolveStats { converged, iters: iters_used, best_gap, resid_at_best, breakdown: false,
        xf: x.iter().map(|v| v.to_f64()).collect(),
        sf: s.iter().map(|v| v.to_f64()).collect(),
        zf: z.iter().map(|v| v.to_f64()).collect() }
}

// fraction-to-boundary step length
fn step_len<R: Real>(s: &[R], ds: &[R], z: &[R], dz: &[R], tau: R) -> R {
    let mut a = R::one();
    for i in 0..s.len() {
        if ds[i].to_f64() < 0.0 {
            let cand = -tau * s[i] / ds[i];
            if cand < a { a = cand; }
        }
        if dz[i].to_f64() < 0.0 {
            let cand = -tau * z[i] / dz[i];
            if cand < a { a = cand; }
        }
    }
    if a.to_f64() < 0.0 { R::zero() } else { a }
}

// TRUE metrics: recompute gap and KKT residual in f64 from an R-iterate
fn true_metrics<R: Real>(p: &Qp, x: &[R], s: &[R], z: &[R]) -> (f64, f64) {
    let n = p.n; let m = p.m;
    let xf: Vec<f64> = x.iter().map(|v| v.to_f64()).collect();
    let sf: Vec<f64> = s.iter().map(|v| v.to_f64()).collect();
    let zf: Vec<f64> = z.iter().map(|v| v.to_f64()).collect();
    let mut resid = 0.0f64;
    // r_d = Q x + q + G^T z
    for i in 0..n {
        let mut acc = p.q_vec[i];
        for j in 0..n { acc += p.q_mat[i * n + j] * xf[j]; }
        for k in 0..m { acc += p.g[k * n + i] * zf[k]; }
        resid = resid.max(acc.abs());
    }
    // r_p = G x + s - h
    for i in 0..m {
        let mut acc = sf[i] - p.h[i];
        for j in 0..n { acc += p.g[i * n + j] * xf[j]; }
        resid = resid.max(acc.abs());
    }
    // gap = s.z/m  (also include any sign violation as residual)
    let mut sz = 0.0;
    for i in 0..m {
        sz += sf[i] * zf[i];
        if sf[i] < 0.0 { resid = resid.max(-sf[i]); }
        if zf[i] < 0.0 { resid = resid.max(-zf[i]); }
    }
    (sz / m as f64, resid)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let gap_tol = 1e-8;
    let max_iter = 80;
    let sizes = [(8usize, 8usize), (16, 16), (32, 32), (64, 64), (128, 128)];
    let n_inst = 60;

    println!("Mehrotra predictor-corrector QP IPM  (min 1/2 xQx+qx s.t. Gx<=h)");
    println!("condensed SPD Cholesky inner solve — the kernel a GPU batch would run\n");
    println!("Accuracy = fraction of {} instances whose f64-recomputed duality gap", n_inst);
    println!("reaches the threshold at any iteration (breakdown after reaching still counts).\n");

    // ---- Part 1: how accurate can each precision get? --------------------
    println!("PART 1 — achievable accuracy by precision");
    println!("{:>9} | {:>5} | {:>7} {:>7} {:>7} {:>7} | {:>11} | {:>10}",
             "size", "prec", "<1e-2", "<1e-4", "<1e-6", "<1e-8", "med best gap", "breakdown");
    println!("{}", "-".repeat(82));
    for &(n, m) in sizes.iter() {
        for &prec in ["f64", "f32"].iter() {
            let mut hit = [0usize; 4]; // 1e-2,1e-4,1e-6,1e-8
            let mut breakdown = 0usize;
            let mut best_gaps: Vec<f64> = Vec::new();
            for inst in 0..n_inst {
                let mut rng = Rng::new(0xC0FFEE ^ ((n as u64) << 16) ^ inst as u64);
                let p = gen_qp(n, m, &mut rng);
                let st = if prec == "f64" { solve_qp::<f64>(&p, gap_tol, max_iter) }
                         else { solve_qp::<f32>(&p, gap_tol, max_iter) };
                if st.best_gap < 1e-2 { hit[0] += 1; }
                if st.best_gap < 1e-4 { hit[1] += 1; }
                if st.best_gap < 1e-6 { hit[2] += 1; }
                if st.best_gap < 1e-8 { hit[3] += 1; }
                if st.breakdown { breakdown += 1; }
                best_gaps.push(st.best_gap);
            }
            let f = |k: usize| 100.0 * hit[k] as f64 / n_inst as f64;
            println!("{:>9} | {:>5} | {:>6.0}% {:>6.0}% {:>6.0}% {:>6.0}% | {:>11.1e} | {:>5}/{:<3}",
                     format!("{}x{}", n, m), prec, f(0), f(1), f(2), f(3),
                     median(&mut best_gaps), breakdown, n_inst);
        }
    }

    // ---- Part 2: f32-warm -> f64-refine hybrid (the note's mitigation) ---
    println!("\nPART 2 — hybrid: solve in f32, then warm-start a SHORT f64 refinement");
    println!("(models GPU f32 forward + a few f64 CPU clean-up iters; the Phase-4 plan)\n");
    println!("{:>9} | {:>14} | {:>14} | {:>18}",
             "size", "f32 reach<1e-8", "hybrid<1e-8", "med f64 refine iters");
    println!("{}", "-".repeat(64));
    for &(n, m) in sizes.iter() {
        let mut f32_ok = 0usize;
        let mut hybrid_ok = 0usize;
        let mut refine_iters: Vec<f64> = Vec::new();
        for inst in 0..n_inst {
            let mut rng = Rng::new(0xC0FFEE ^ ((n as u64) << 16) ^ inst as u64);
            let p = gen_qp(n, m, &mut rng);
            let f32st = solve_qp::<f32>(&p, gap_tol, max_iter);
            if f32st.best_gap < 1e-8 { f32_ok += 1; }
            // warm-start f64 from the f32 final iterate (nudge slacks/duals
            // positive so the warm point is interior, as a real handoff would)
            let nudge = |v: &[f64]| -> Vec<f64> {
                v.iter().map(|&x| if x < 1e-6 { 1e-6 } else { x }).collect()
            };
            let sw = nudge(&f32st.sf);
            let zw = nudge(&f32st.zf);
            let hy = solve_qp_warm::<f64>(&p, gap_tol, 30, Some((&f32st.xf, &sw, &zw)));
            if hy.converged { hybrid_ok += 1; refine_iters.push(hy.iters as f64); }
        }
        let pc = |k: usize| 100.0 * k as f64 / n_inst as f64;
        println!("{:>9} | {:>13.0}% | {:>13.0}% | {:>18.0}",
                 format!("{}x{}", n, m), pc(f32_ok), pc(hybrid_ok),
                 { let mut r = refine_iters.clone(); let v = median(&mut r); if v.is_nan() {0.0} else {v} });
    }
}
