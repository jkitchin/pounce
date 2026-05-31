//! Shared math for the GPU spike harnesses.
//!
//! All three Phase-0 steps (CPU baseline, GPU microbench, on-device
//! f32 accuracy) share: a float-generic dense QP IPM, a dense Cholesky
//! factor+solve that matches the WGSL kernel byte-for-byte in structure,
//! SPD-system generators, and f64-recomputed accuracy metrics.

use std::ops::{Add, Div, Mul, Neg, Sub};

// ---------------------------------------------------------------------
// Real: lets the solver and the dense kernels run in f32 or f64.
// ---------------------------------------------------------------------

pub trait Real:
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
}

impl Real for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn from_f64(x: f64) -> Self { x }
    fn to_f64(self) -> f64 { self }
    fn sqrt(self) -> Self { f64::sqrt(self) }
}

impl Real for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn from_f64(x: f64) -> Self { x as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn sqrt(self) -> Self { f32::sqrt(self) }
}

// ---------------------------------------------------------------------
// Deterministic RNG (xorshift64*). Problems are generated in f64 so the
// f32 and f64 solves see identical data.
// ---------------------------------------------------------------------

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// uniform in [-1, 1)
    pub fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    /// uniform in [lo, hi)
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
}

// ---------------------------------------------------------------------
// Dense SPD system K x = b  (the condensed-KKT shape the GPU factors).
// ---------------------------------------------------------------------

/// One n×n SPD system, row-major, with a right-hand side.
#[derive(Clone)]
pub struct SpdSystem {
    pub n: usize,
    pub mat: Vec<f64>, // n*n, row-major, SPD
    pub rhs: Vec<f64>, // n
}

/// Generate an SPD system `K = (1/n) MᵀM + jitter·I`. Smaller `jitter`
/// raises the condition number — the knob that drives the f32 accuracy
/// floor in Step 2.
pub fn gen_spd(n: usize, jitter: f64, rng: &mut Rng) -> SpdSystem {
    let mut m = vec![0.0f64; n * n];
    for v in m.iter_mut() {
        *v = rng.unif();
    }
    let mut mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += m[k * n + i] * m[k * n + j];
            }
            mat[i * n + j] = acc / n as f64 + if i == j { jitter } else { 0.0 };
        }
    }
    let rhs: Vec<f64> = (0..n).map(|_| rng.unif()).collect();
    SpdSystem { n, mat, rhs }
}

/// Dense Cholesky factor (in place, lower triangle) + forward/back
/// solve, generic over precision. Structurally identical to the WGSL
/// kernel so the CPU reference and the GPU result are comparable.
/// Returns false on a non-positive pivot (f32 breakdown).
pub fn chol_solve<R: Real>(n: usize, k: &mut [R], rhs: &mut [R]) -> bool {
    for j in 0..n {
        let mut diag = k[j * n + j];
        for p in 0..j {
            diag = diag - k[j * n + p] * k[j * n + p];
        }
        if !(diag.to_f64() > 0.0) {
            return false;
        }
        let ljj = diag.sqrt();
        k[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut s = k[i * n + j];
            for p in 0..j {
                s = s - k[i * n + p] * k[j * n + p];
            }
            k[i * n + j] = s / ljj;
        }
    }
    // forward L y = rhs
    for i in 0..n {
        let mut s = rhs[i];
        for p in 0..i {
            s = s - k[i * n + p] * rhs[p];
        }
        rhs[i] = s / k[i * n + i];
    }
    // back Lᵀ x = y
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for p in (i + 1)..n {
            s = s - k[p * n + i] * rhs[p];
        }
        rhs[i] = s / k[i * n + i];
    }
    true
}

/// Solve one SPD system in precision `R`, return the solution as f64.
pub fn solve_spd<R: Real>(sys: &SpdSystem) -> Option<Vec<f64>> {
    let n = sys.n;
    let mut k: Vec<R> = sys.mat.iter().map(|&v| R::from_f64(v)).collect();
    let mut b: Vec<R> = sys.rhs.iter().map(|&v| R::from_f64(v)).collect();
    if !chol_solve::<R>(n, &mut k, &mut b) {
        return None;
    }
    Some(b.iter().map(|v| v.to_f64()).collect())
}

/// TRUE relative residual ‖K x − b‖∞ / ‖b‖∞, recomputed in f64.
pub fn residual_inf(sys: &SpdSystem, x: &[f64]) -> f64 {
    let n = sys.n;
    let mut rmax = 0.0f64;
    let mut bmax = 0.0f64;
    for i in 0..n {
        let mut acc = -sys.rhs[i];
        for j in 0..n {
            acc += sys.mat[i * n + j] * x[j];
        }
        rmax = rmax.max(acc.abs());
        bmax = bmax.max(sys.rhs[i].abs());
    }
    if bmax > 0.0 {
        rmax / bmax
    } else {
        rmax
    }
}

/// One step of f64 iterative refinement against the f64 system — the
/// CPU "f64 tail" that recovers accuracy after an f32 solve.
pub fn refine_f64(sys: &SpdSystem, x: &mut [f64]) {
    let n = sys.n;
    // residual r = b − K x
    let mut r = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = sys.rhs[i];
        for j in 0..n {
            acc -= sys.mat[i * n + j] * x[j];
        }
        r[i] = acc;
    }
    // dx = K⁻¹ r  (fresh f64 factorization; spike-simple)
    let mut k: Vec<f64> = sys.mat.clone();
    if chol_solve::<f64>(n, &mut k, &mut r) {
        for i in 0..n {
            x[i] += r[i];
        }
    }
}

// ---------------------------------------------------------------------
// Dense inequality QP for the CPU baseline (Step 0):
//   min ½ xᵀQx + qᵀx  s.t.  G x ≤ h
// Mehrotra predictor-corrector with a condensed SPD inner solve.
// ---------------------------------------------------------------------

pub struct Qp {
    pub n: usize,
    pub m: usize,
    pub q_mat: Vec<f64>,
    pub q_vec: Vec<f64>,
    pub g: Vec<f64>,
    pub h: Vec<f64>,
    pub x0: Vec<f64>,
    pub s0: Vec<f64>,
    pub z0: Vec<f64>,
}

pub fn gen_qp(n: usize, m: usize, rng: &mut Rng) -> Qp {
    let mut mmat = vec![0.0f64; n * n];
    for v in mmat.iter_mut() {
        *v = rng.unif();
    }
    let mut q_mat = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += mmat[k * n + i] * mmat[k * n + j];
            }
            q_mat[i * n + j] = acc / n as f64 + if i == j { 1.0 } else { 0.0 };
        }
    }
    let q_vec: Vec<f64> = (0..n).map(|_| rng.unif()).collect();
    let g: Vec<f64> = (0..m * n).map(|_| rng.unif()).collect();
    let x0: Vec<f64> = (0..n).map(|_| rng.unif()).collect();
    let s0: Vec<f64> = (0..m).map(|_| rng.range(0.5, 1.5)).collect();
    let z0: Vec<f64> = (0..m).map(|_| rng.range(0.5, 1.5)).collect();
    let mut h = vec![0.0f64; m];
    for i in 0..m {
        let mut acc = 0.0;
        for j in 0..n {
            acc += g[i * n + j] * x0[j];
        }
        h[i] = acc + s0[i];
    }
    Qp { n, m, q_mat, q_vec, g, h, x0, s0, z0 }
}

pub struct QpStats {
    pub converged: bool,
    pub iters: usize,
    pub best_gap: f64,
}

/// Solve one QP in precision `R`. Returns convergence stats (gap judged
/// in f64). Self-contained Mehrotra IPM.
pub fn solve_qp<R: Real>(p: &Qp, gap_tol: f64, max_iter: usize) -> QpStats {
    let n = p.n;
    let m = p.m;
    let c = R::from_f64;
    let q_mat: Vec<R> = p.q_mat.iter().map(|&v| c(v)).collect();
    let q_vec: Vec<R> = p.q_vec.iter().map(|&v| c(v)).collect();
    let g: Vec<R> = p.g.iter().map(|&v| c(v)).collect();
    let h: Vec<R> = p.h.iter().map(|&v| c(v)).collect();
    let mut x: Vec<R> = p.x0.iter().map(|&v| c(v)).collect();
    let mut s: Vec<R> = p.s0.iter().map(|&v| c(v)).collect();
    let mut z: Vec<R> = p.z0.iter().map(|&v| c(v)).collect();

    let mut best_gap = f64::INFINITY;
    let mut converged = false;
    let mut iters_used = 0;

    for it in 0..max_iter {
        iters_used = it + 1;
        // r_d = Q x + q + Gᵀ z
        let mut r_d = vec![R::zero(); n];
        for i in 0..n {
            let mut acc = q_vec[i];
            for j in 0..n {
                acc = acc + q_mat[i * n + j] * x[j];
            }
            for k in 0..m {
                acc = acc + g[k * n + i] * z[k];
            }
            r_d[i] = acc;
        }
        // r_p = G x + s − h
        let mut r_p = vec![R::zero(); m];
        for i in 0..m {
            let mut acc = s[i] - h[i];
            for j in 0..n {
                acc = acc + g[i * n + j] * x[j];
            }
            r_p[i] = acc;
        }
        let mut sz = R::zero();
        for i in 0..m {
            sz = sz + s[i] * z[i];
        }
        let mu = sz / R::from_f64(m.max(1) as f64);

        // TRUE gap in f64
        let (true_gap, _) = qp_true_metrics(p, &x, &s, &z);
        if true_gap < best_gap {
            best_gap = true_gap;
        }
        if true_gap < gap_tol {
            converged = true;
            break;
        }

        let mut w = vec![R::zero(); m];
        for i in 0..m {
            w[i] = z[i] / s[i];
        }
        // K = Q + Gᵀ W G
        let mut kmat = vec![R::zero(); n * n];
        for a in 0..n {
            for b in 0..n {
                let mut acc = q_mat[a * n + b];
                for r in 0..m {
                    acc = acc + g[r * n + a] * w[r] * g[r * n + b];
                }
                kmat[a * n + b] = acc;
            }
        }
        // affine rhs: −r_d − Gᵀ(W r_p − z)
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
        if !chol_solve::<R>(n, &mut kfac, &mut rhs) {
            break;
        }
        let dx_aff = rhs.clone();
        let mut ds_aff = vec![R::zero(); m];
        for i in 0..m {
            let mut acc = -r_p[i];
            for j in 0..n {
                acc = acc - g[i * n + j] * dx_aff[j];
            }
            ds_aff[i] = acc;
        }
        let mut dz_aff = vec![R::zero(); m];
        for i in 0..m {
            dz_aff[i] = -z[i] - w[i] * ds_aff[i];
        }
        let a_aff = step_len(&s, &ds_aff, &z, &dz_aff, R::one());
        let mut sz_aff = R::zero();
        for i in 0..m {
            sz_aff = sz_aff + (s[i] + a_aff * ds_aff[i]) * (z[i] + a_aff * dz_aff[i]);
        }
        let mu_aff = sz_aff / R::from_f64(m.max(1) as f64);
        let ratio = (mu_aff / mu).to_f64().max(0.0);
        let sigma = R::from_f64(ratio * ratio * ratio);
        // corrector rhs
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
        if !chol_solve::<R>(n, &mut kfac2, &mut rhs2) {
            break;
        }
        let dx = rhs2.clone();
        let mut ds = vec![R::zero(); m];
        for i in 0..m {
            let mut acc = -r_p[i];
            for j in 0..n {
                acc = acc - g[i * n + j] * dx[j];
            }
            ds[i] = acc;
        }
        let mut dz = vec![R::zero(); m];
        for i in 0..m {
            let rc = s[i] * z[i] + ds_aff[i] * dz_aff[i] - sigma * mu;
            dz[i] = -(rc + z[i] * ds[i]) / s[i];
        }
        let tau = R::from_f64(0.95);
        let alpha = step_len(&s, &ds, &z, &dz, tau);
        if !alpha.to_f64().is_finite() {
            break;
        }
        for j in 0..n {
            x[j] = x[j] + alpha * dx[j];
        }
        for i in 0..m {
            s[i] = s[i] + alpha * ds[i];
            z[i] = z[i] + alpha * dz[i];
        }
    }

    QpStats { converged, iters: iters_used, best_gap }
}

fn step_len<R: Real>(s: &[R], ds: &[R], z: &[R], dz: &[R], tau: R) -> R {
    let mut a = R::one();
    for i in 0..s.len() {
        if ds[i].to_f64() < 0.0 {
            let cand = -tau * s[i] / ds[i];
            if cand < a {
                a = cand;
            }
        }
        if dz[i].to_f64() < 0.0 {
            let cand = -tau * z[i] / dz[i];
            if cand < a {
                a = cand;
            }
        }
    }
    if a.to_f64() < 0.0 {
        R::zero()
    } else {
        a
    }
}

fn qp_true_metrics<R: Real>(p: &Qp, x: &[R], s: &[R], z: &[R]) -> (f64, f64) {
    let n = p.n;
    let m = p.m;
    let xf: Vec<f64> = x.iter().map(|v| v.to_f64()).collect();
    let sf: Vec<f64> = s.iter().map(|v| v.to_f64()).collect();
    let zf: Vec<f64> = z.iter().map(|v| v.to_f64()).collect();
    let mut resid = 0.0f64;
    for i in 0..n {
        let mut acc = p.q_vec[i];
        for j in 0..n {
            acc += p.q_mat[i * n + j] * xf[j];
        }
        for k in 0..m {
            acc += p.g[k * n + i] * zf[k];
        }
        resid = resid.max(acc.abs());
    }
    let mut sz = 0.0;
    for i in 0..m {
        sz += sf[i] * zf[i];
    }
    (sz / m.max(1) as f64, resid)
}
