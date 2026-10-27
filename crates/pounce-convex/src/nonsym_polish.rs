//! Active-set **polish** of an exponential- or power-cone solution (gh#1085).
//!
//! The non-symmetric HSDE driver ([`crate::hsde_nonsym`]) cannot drive its
//! residuals much below `~1e-7` on an exp-cone model. Near the cone boundary
//! the `(z,z)` scaling block `M⁻¹` has eigenvalues `~μ`, `~1` and `~1/μ`, and
//! a dense 3×3 representation loses the `~μ` one to rounding once
//! `μ ≲ 1e-8` (absolute error `ε/μ`). The block then reads as indefinite and
//! the step collapses. Its objective is right by then, but its minimizer is
//! not: on a cond-18 Poisson regression the `Optimal` point was `1.8e-4` from
//! the minimizer, and tightening `tol` stalled at `1.5e-5`.
//!
//! The power cone has the same gap: the adversary's CES-utility model returned
//! `optimal` with `x` `6.6e-3` from its closed form, at an objective error of
//! `6e-8`.
//!
//! The polish leaves the barrier behind once the IPM has identified the
//! active set. At a strictly complementary optimum every exp or power block is
//! either
//!
//! - **inactive** — `s` interior to the cone, `z = 0`, or
//! - **active** — `s` on the smooth part of the cone's boundary, where the
//!   cone is locally one smooth inequality `ψ(s) ≥ 0`, so `ψ(s) = 0` and
//!   `z = λ·∇ψ(s)` with `λ ≥ 0` (`∇ψ` is the normal of the cone at `s`, and
//!   `⟨∇ψ(s), s⟩ = ψ(s) = 0` because `ψ` is 1-homogeneous):
//!   - exp: `ψ(s) = s₁·log(s₂/s₁) − s₀`, smooth where `s₁, s₂ > 0`;
//!   - power `K_α`: `ψ(s) = s₁^α·s₂^{1−α} − σ·s₀` with `σ = sign(s₀)`. Where
//!     `s₁, s₂ > 0` the boundary has `|s₀| = s₁^α s₂^{1−α} > 0`, so `σ` is
//!     fixed and only that one of the two inequalities `±s₀ ≤ s₁^α s₂^{1−α}`
//!     can bind. The boundary is non-smooth only where `s₁ = 0` or `s₂ = 0`.
//!
//! and every orthant row is active (`sᵢ = 0`) or inactive (`zᵢ = 0`). On that
//! guess the KKT conditions are a square system of smooth equations,
//!
//! ```text
//!   c + Px + Aᵀy + G_oᵀz_o + Σ_k λ_k G_kᵀ∇ψ(s_k) = 0
//!                                      Ax = b
//!                                     G_o x = h_o
//!                       ψ(h_k − G_k x) = 0      (each active block k)
//! ```
//!
//! and Newton's method on it converges quadratically from the IPM point.
//! The Jacobian is the symmetric KKT matrix
//! `[P + Σ λ_k G_kᵀ(−∇²ψ)G_k, Aᵀ, G_oᵀ, Jᵀ; A; G_o; J]` with
//! `J_k = −∇ψ(s_k)ᵀG_k`, in the unknowns `(x, y, z_o, −λ)`. `−∇²ψ` is PSD
//! (exp: ψ is the perspective of `log`; power: a weighted geometric mean minus
//! a linear term; both concave), which makes the `(x,x)` block PSD. Both ψ's
//! Hessians live on `(s₁, s₂)` only.
//!
//! The polish only proposes a point. It returns `None` whenever the guess
//! cannot be made (a block at a non-smooth boundary point — the apex, exp's
//! `s₁ = 0` face, a power cone's `s₁ = 0` or `s₂ = 0` face — or a second-order
//! block in the product), when the factorization
//! fails, when Newton stalls, or when the result breaks a sign condition (a
//! negative multiplier, an inactive row or block leaving its cone). The
//! driver then compares the proposal with the IPM point on the scale-relative
//! KKT certificate it already uses to judge optimality, and keeps the IPM
//! point unless the proposal does better.

use crate::hsde_nonsym::{NonsymCone, NsBlock};
use crate::ipm::{dot, inf_norm};
use crate::qp::QpProblem;
use pounce_common::types::Index;
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use std::collections::BTreeMap;

/// Newton iterations the polish may take. Convergence from an IPM point is
/// quadratic, so a good guess needs 2–4; the cap only bounds a bad one.
const MAX_NEWTON: usize = 20;

/// Static regularization of the polish KKT (`+δ` on the `(x,x)` block, `−δ`
/// on the multiplier blocks), so a model whose curvature lives only in its
/// constraints (an LP part) still factors. Newton recomputes the residual
/// exactly every iteration, so δ slows convergence by a factor of `δ` relative
/// to the curvature and does not move the point it converges to.
const POLISH_REG: f64 = 1e-12;

/// Residual (relative to the magnitude of the data) at which Newton stops:
/// a few ulps. It cannot do better, since the residuals are evaluated in
/// doubles.
const NEWTON_RTOL: f64 = 1e-15;

/// Residual (relative, as [`NEWTON_RTOL`]) Newton must have reached for the
/// polish to propose anything. A correct guess lands within a few ulps; a
/// wrong one stalls orders of magnitude above this (an inconsistent system
/// has no root to converge to).
const CONVERGED_RTOL: f64 = 1e-10;

/// How far below zero (relative to the magnitude of the multipliers) a
/// polished multiplier may sit and still be read as zero. A multiplier
/// converging to `0` on a weakly active constraint lands on either side of
/// it by rounding; anything past this is a wrong active-set guess.
const SIGN_SLACK: f64 = 1e-12;

/// A residual of the polish system and the `(∇ψ, ∇²ψ)` of each active block
/// it was evaluated with (the Jacobian reuses them).
type Residual = (Vec<f64>, Vec<([f64; 3], [f64; 3])>);

/// Which smooth boundary function ψ an active block is held on.
#[derive(Clone, Copy)]
enum Boundary {
    /// `ψ = s₁·log(s₂/s₁) − s₀`.
    Exp,
    /// `ψ = s₁^α s₂^{1−α} − σ·s₀`, `σ = ±1` the sign of `s₀` at the IPM point.
    Pow { alpha: f64, sigma: f64 },
}

/// Where one active exp/power block sits in the polish system.
struct ConeRow {
    /// Row offset of the block in `G`/`h`.
    off: usize,
    /// Its multiplier's position among the polish unknowns.
    col: usize,
    /// The boundary it is held on.
    kind: Boundary,
}

/// `ψ(s)`, its gradient and the `(1,2)` block of its Hessian (the only nonzero
/// one, lower triangle `[h₁₁; h₂₁, h₂₂]`), at a point with `s₁, s₂ > 0`.
fn psi(kind: Boundary, s: &[f64]) -> (f64, [f64; 3], [f64; 3]) {
    let (s0, s1, s2) = (s[0], s[1], s[2]);
    match kind {
        Boundary::Exp => {
            let l = (s2 / s1).ln();
            let g = [-1.0, l - 1.0, s1 / s2];
            // ∇²ψ on (s₁, s₂): [−1/s₁; 1/s₂, −s₁/s₂²].
            let h = [-1.0 / s1, 1.0 / s2, -s1 / (s2 * s2)];
            (s1 * l - s0, g, h)
        }
        Boundary::Pow { alpha, sigma } => {
            let om = 1.0 - alpha;
            let gm = s1.powf(alpha) * s2.powf(om);
            let g = [-sigma, alpha * gm / s1, om * gm / s2];
            // ∇²(s₁^α s₂^{1−α}) = α(1−α)·gm·[−1/s₁²; 1/(s₁s₂), −1/s₂²].
            let c = alpha * om * gm;
            let h = [-c / (s1 * s1), c / (s1 * s2), -c / (s2 * s2)];
            (gm - sigma * s0, g, h)
        }
    }
}

/// The boundary a block would be held on at `s`, and its primal margin (how
/// far `s` is inside the cone, in the units of `s`), or `None` at a
/// non-smooth boundary point (or outside the cone's `s₁, s₂ > 0` domain).
fn boundary_of(cone: &NonsymCone, s: &[f64]) -> Option<(Boundary, f64)> {
    if !(s[1] > 0.0 && s[2] > 0.0) {
        return None;
    }
    let kind = match cone {
        NonsymCone::Exp(_) => Boundary::Exp,
        NonsymCone::Power(c) => {
            // σ is only defined off the s₀ = 0 plane; an active block cannot
            // sit on it when s₁, s₂ > 0, and an inactive one does not use σ.
            let sigma = if s[0] < 0.0 { -1.0 } else { 1.0 };
            Boundary::Pow {
                alpha: c.alpha,
                sigma,
            }
        }
    };
    Some((kind, psi(kind, s).0))
}

/// Polish the IPM solution `(x, y, z)` (un-homogenized) of
/// `min ½xᵀPx + cᵀx s.t. Ax = b, Gx + s = h, s ∈ K` on its active set, `K` a
/// product of orthant, exponential and power blocks. Returns the polished
/// `(x, y, z)`, or `None` when the polish does not apply or does not converge
/// to a sign-feasible point (see the module docs). The caller decides whether
/// the proposal is better than what it has.
pub(crate) fn polish_nonsym<F>(
    prob: &QpProblem,
    specs: &[NsBlock],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    make_backend: &mut F,
) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)>
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let n = prob.n;
    let m_eq = prob.m_eq();
    let m_ineq = prob.m_ineq();
    if !specs.iter().any(|b| matches!(b, NsBlock::Nonsym(_))) {
        return None;
    }

    // s = h − Gx at the IPM point.
    let slack = |x: &[f64]| {
        let mut gx = vec![0.0; m_ineq];
        prob.g_mul(x, &mut gx);
        (0..m_ineq).map(|i| prob.h[i] - gx[i]).collect::<Vec<f64>>()
    };
    let s = slack(x);

    // --- active-set guess ---------------------------------------------------
    // Unknowns: x (n) | y (m_eq) | z_o (active orthant rows) | ν = −λ (active
    // exp/power blocks). At a strictly complementary optimum one of each (primal
    // margin, dual magnitude) pair is O(μ) and the other O(1), so comparing
    // them identifies the side; a wrong call fails the sign checks below.
    let mut orth_active: Vec<(usize, usize)> = Vec::new(); // (row, col)
    let mut orth_inactive: Vec<usize> = Vec::new();
    let mut cone_active: Vec<ConeRow> = Vec::new();
    let mut cone_inactive: Vec<(usize, Boundary)> = Vec::new();
    let mut col = n + m_eq;
    let mut off = 0;
    let mut lambda0: Vec<f64> = Vec::new();
    for b in specs {
        match b {
            NsBlock::Orthant(d) => {
                for i in off..off + d {
                    if s[i] < z[i] {
                        orth_active.push((i, col));
                        col += 1;
                    } else {
                        orth_inactive.push(i);
                    }
                }
                off += d;
            }
            NsBlock::Nonsym(nc) => {
                let sb = &s[off..off + 3];
                let zb = &z[off..off + 3];
                // Only the smooth part of the boundary (s₁, s₂ > 0) has a ψ
                // normal to linearize; the faces and the apex do not.
                let (kind, ps) = boundary_of(nc, sb)?;
                let (_, g, _) = psi(kind, sb);
                let zmag = inf_norm(zb);
                if ps < zmag {
                    // λ from the least-squares fit z ≈ λ∇ψ.
                    let lam = dot(zb, &g) / dot(&g, &g);
                    // NaN-safe: `!(lam > 0.0)` also rejects a NaN λ.
                    #[allow(clippy::neg_cmp_op_on_partial_ord)]
                    if !(lam > 0.0) {
                        return None;
                    }
                    cone_active.push(ConeRow { off, col, kind });
                    lambda0.push(lam);
                    col += 1;
                } else {
                    cone_inactive.push((off, kind));
                }
                off += 3;
            }
            // A second-order block's boundary has no single ψ here.
            NsBlock::SecondOrder(_) => return None,
        }
    }
    debug_assert_eq!(off, m_ineq);
    let dim = col;

    // --- constant sparsity: P, A, G_o once; per-exp-block entries -----------
    // The (x,x) block also carries Σ λ_k G_kᵀ(−∇²ψ)G_k, whose pattern is the
    // pairs of columns appearing in rows off+1, off+2 of block k.
    let mut row_of_g: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m_ineq];
    for t in &prob.g {
        row_of_g[t.row].push((t.col, t.val));
    }
    let mut entries: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut add = |r: usize, c: usize, v: f64| {
        let (r, c) = if r >= c { (r, c) } else { (c, r) };
        *entries.entry((r, c)).or_insert(0.0) += v;
    };
    for t in &prob.p_lower {
        add(t.row, t.col, t.val);
    }
    for j in 0..n {
        add(j, j, POLISH_REG);
    }
    for t in &prob.a {
        add(n + t.row, t.col, t.val);
    }
    for i in 0..m_eq {
        add(n + i, n + i, -POLISH_REG);
    }
    for &(row, c) in &orth_active {
        for &(j, v) in &row_of_g[row] {
            add(c, j, v);
        }
        add(c, c, -POLISH_REG);
    }
    // Pattern-only placeholders for the iteration-dependent entries.
    for e in &cone_active {
        for r in 0..3 {
            for &(j, _) in &row_of_g[e.off + r] {
                add(e.col, j, 0.0);
            }
        }
        add(e.col, e.col, -POLISH_REG);
        for r1 in 1..3 {
            for r2 in 1..3 {
                for &(j1, _) in &row_of_g[e.off + r1] {
                    for &(j2, _) in &row_of_g[e.off + r2] {
                        add(j1, j2, 0.0);
                    }
                }
            }
        }
    }
    let mut pos_of: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut airn: Vec<Index> = Vec::with_capacity(entries.len());
    let mut ajcn: Vec<Index> = Vec::with_capacity(entries.len());
    let mut base: Vec<f64> = Vec::with_capacity(entries.len());
    for (k, ((r, c), v)) in entries.into_iter().enumerate() {
        pos_of.insert((r, c), k);
        airn.push((r + 1) as Index);
        ajcn.push((c + 1) as Index);
        base.push(v);
    }
    let pos = |r: usize, c: usize| {
        let key = if r >= c { (r, c) } else { (c, r) };
        pos_of[&key]
    };

    // --- Newton -------------------------------------------------------------
    let mut xk = x.to_vec();
    let mut yk = y.to_vec();
    let mut zo: Vec<f64> = orth_active.iter().map(|&(i, _)| z[i]).collect();
    let mut lam = lambda0;

    // Scale for the stopping rule: magnitude of the data the residuals are
    // differences of.
    let data_scale = 1.0
        + inf_norm(&prob.c)
            .max(inf_norm(&prob.b))
            .max(inf_norm(&prob.h))
            .max(inf_norm(x));

    // F(x, y, z_o, λ) and the per-block (∇ψ, ∇²ψ) it was evaluated with.
    let residual = |xk: &[f64], yk: &[f64], zo: &[f64], lam: &[f64]| -> Option<Residual> {
        let sk = slack(xk);
        let mut r = vec![0.0; dim];
        let mut rd = vec![0.0; n];
        prob.p_mul(xk, &mut rd);
        for (rj, cj) in rd.iter_mut().zip(&prob.c) {
            *rj += cj;
        }
        prob.at_mul(yk, &mut rd);
        for (k, &(row, _)) in orth_active.iter().enumerate() {
            for &(j, v) in &row_of_g[row] {
                rd[j] += v * zo[k];
            }
        }
        let mut derivs = Vec::with_capacity(cone_active.len());
        for (k, e) in cone_active.iter().enumerate() {
            let sb = &sk[e.off..e.off + 3];
            if !(sb[1] > 0.0 && sb[2] > 0.0) {
                return None;
            }
            let (ps, g, h) = psi(e.kind, sb);
            for (q, gq) in g.iter().enumerate() {
                for &(j, v) in &row_of_g[e.off + q] {
                    rd[j] += lam[k] * gq * v;
                }
            }
            r[e.col] = ps;
            derivs.push((g, h));
        }
        r[..n].copy_from_slice(&rd);
        let mut ax = vec![0.0; m_eq];
        prob.a_mul(xk, &mut ax);
        for i in 0..m_eq {
            r[n + i] = ax[i] - prob.b[i];
        }
        for &(row, c) in &orth_active {
            r[c] = -sk[row];
        }
        if r.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some((r, derivs))
    };

    let (mut r, mut derivs) = residual(&xk, &yk, &zo, &lam)?;
    let mut rnorm = inf_norm(&r);
    let mut fact: Option<Factorization> = None;
    let mut vals = base.clone();
    for _ in 0..MAX_NEWTON {
        if rnorm <= NEWTON_RTOL * data_scale {
            break;
        }
        if crate::deadline::expired() {
            return None;
        }
        // Assemble the iteration-dependent entries: −∇ψᵀG_k in row e.col and
        // λ_k·G_kᵀ(−∇²ψ)G_k in the (x,x) block.
        vals.copy_from_slice(&base);
        for (k, e) in cone_active.iter().enumerate() {
            let (g, h) = &derivs[k];
            for (q, gq) in g.iter().enumerate() {
                for &(j, v) in &row_of_g[e.off + q] {
                    vals[pos(e.col, j)] -= gq * v;
                }
            }
            // −∇²ψ on (s₁, s₂): [1/s₁, −1/s₂; −1/s₂, s₁/s₂²] (PSD).
            let nh = [[-h[0], -h[1]], [-h[1], -h[2]]];
            for (a, r1) in (1..3).enumerate() {
                for (bq, r2) in (1..3).enumerate() {
                    let w = lam[k] * nh[a][bq];
                    for &(j1, v1) in &row_of_g[e.off + r1] {
                        for &(j2, v2) in &row_of_g[e.off + r2] {
                            // Each unordered (j1, j2) pair is visited from both
                            // sides; the lower triangle takes the j1 ≥ j2 half.
                            if j1 >= j2 {
                                vals[pos(j1, j2)] += w * v1 * v2;
                            }
                        }
                    }
                }
            }
        }
        match fact.as_mut() {
            Some(f) => f.refactor(&vals).ok()?,
            None => {
                fact = Some(
                    Factorization::new(
                        dim as Index,
                        airn.clone(),
                        ajcn.clone(),
                        vals.clone(),
                        make_backend(),
                    )
                    .ok()?,
                );
            }
        }
        let f = fact.as_mut()?;
        let mut step: Vec<f64> = r.iter().map(|v| -v).collect();
        f.solve_one(&mut step).ok()?;
        if step.iter().any(|v| !v.is_finite()) {
            return None;
        }

        // Backtrack on the residual norm: a full step from an IPM point is
        // expected, a shorter one guards an imperfect guess.
        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..30 {
            let xt: Vec<f64> = (0..n).map(|j| xk[j] + alpha * step[j]).collect();
            let yt: Vec<f64> = (0..m_eq).map(|i| yk[i] + alpha * step[n + i]).collect();
            let zt: Vec<f64> = orth_active
                .iter()
                .enumerate()
                .map(|(k, &(_, c))| zo[k] + alpha * step[c])
                .collect();
            // The unknown is ν = −λ.
            let lt: Vec<f64> = cone_active
                .iter()
                .enumerate()
                .map(|(k, e)| lam[k] - alpha * step[e.col])
                .collect();
            if let Some((rt, dt)) = residual(&xt, &yt, &zt, &lt) {
                let nt = inf_norm(&rt);
                if nt < rnorm {
                    accepted = Some((xt, yt, zt, lt, rt, dt, nt));
                    break;
                }
            }
            alpha *= 0.5;
        }
        match accepted {
            Some((xt, yt, zt, lt, rt, dt, nt)) => {
                xk = xt;
                yk = yt;
                zo = zt;
                lam = lt;
                r = rt;
                derivs = dt;
                rnorm = nt;
            }
            // No decrease along the Newton direction: at the rounding floor,
            // or the guess was wrong. Either way stop here; the checks below
            // and the caller's certificate decide.
            None => break,
        }
    }

    if rnorm > CONVERGED_RTOL * data_scale {
        return None;
    }

    // --- sign checks and the full dual ----------------------------------------
    let zmax = inf_norm(&zo).max(inf_norm(&lam)).max(1.0);
    let floor = SIGN_SLACK * zmax;
    if zo.iter().chain(lam.iter()).any(|&v| v < -floor) {
        return None;
    }
    let sk = slack(&xk);
    let smax = 1.0 + inf_norm(&sk);
    if orth_inactive.iter().any(|&i| sk[i] < -SIGN_SLACK * smax) {
        return None;
    }
    for &(o, kind) in &cone_inactive {
        let sb = &sk[o..o + 3];
        // A power block's margin is `s₁^α s₂^{1−α} − |s₀|`: re-read σ at the
        // polished point, where an inactive `s₀` may have crossed zero.
        let kind = match kind {
            Boundary::Pow { alpha, .. } => Boundary::Pow {
                alpha,
                sigma: if sb[0] < 0.0 { -1.0 } else { 1.0 },
            },
            k => k,
        };
        if !(sb[1] > 0.0 && sb[2] > 0.0) || psi(kind, sb).0 < -SIGN_SLACK * smax {
            return None;
        }
    }
    let mut zk = vec![0.0; m_ineq];
    for (k, &(row, _)) in orth_active.iter().enumerate() {
        zk[row] = zo[k].max(0.0);
    }
    for (k, e) in cone_active.iter().enumerate() {
        let l = lam[k].max(0.0);
        let (g, _) = &derivs[k];
        for q in 0..3 {
            zk[e.off + q] = l * g[q];
        }
    }
    Some((xk, yk, zk))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::Triplet;
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// `min t s.t. t ≥ exp(x), x = 0`: one exp block `(x, 1, t)`, optimum
    /// `t = 1` with dual `z = λ∇ψ(0, 1, 1) = (−1, −1, 1)`.
    fn epigraph() -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![0.0, 1.0],
            a: vec![Triplet::new(0, 0, 1.0)],
            b: vec![0.0],
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(2, 1, -1.0)],
            h: vec![0.0, 1.0, 0.0],
            lb: vec![],
            ub: vec![],
        }
    }

    #[test]
    fn converges_from_a_perturbed_point() {
        let prob = epigraph();
        let x = [1e-5, 1.0 + 3e-5];
        let z = [-1.0 + 2e-5, -1.0, 1.0 - 1e-5];
        let (px, _py, pz) =
            polish_nonsym(&prob, &[NsBlock::exp()], &x, &[0.9], &z, &mut backend).expect("polish");
        assert!(px[0].abs() < 1e-15 && (px[1] - 1.0).abs() < 1e-15, "{px:?}");
        for (q, want) in [-1.0, -1.0, 1.0].iter().enumerate() {
            assert!((pz[q] - want).abs() < 1e-14, "z = {pz:?}");
        }
    }

    /// Claim the binding block is slack (a tiny `z`, a large `ψ`): Newton
    /// then drops the only constraint holding `t` up, and the result is not
    /// a KKT point, so nothing is proposed.
    #[test]
    fn a_wrong_active_set_guess_proposes_nothing() {
        let prob = epigraph();
        let x = [0.0, 2.0]; // ψ = log 2 > ‖z‖
        let z = [-1e-3, -1e-3, 1e-3];
        assert!(polish_nonsym(&prob, &[NsBlock::exp()], &x, &[0.0], &z, &mut backend).is_none());
    }

    /// A second-order block has no single smooth ψ here, so the product is
    /// declined.
    #[test]
    fn second_order_blocks_are_declined() {
        let prob = epigraph();
        let x = [0.0, 1.0];
        let z = [-1.0, -1.0, 1.0];
        let mut p = prob.clone();
        p.h.extend_from_slice(&[0.0, 1.0, 1.0]);
        let mut zz = z.to_vec();
        zz.extend_from_slice(&[0.0, 1.0, 1.0]);
        assert!(
            polish_nonsym(
                &p,
                &[NsBlock::exp(), NsBlock::SecondOrder(3)],
                &x,
                &[0.0],
                &zz,
                &mut backend
            )
            .is_none()
        );
        // And a model with no exp/power block at all is not this polish's
        // business.
        let lp = QpProblem {
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![0.0],
            ..prob.clone()
        };
        assert!(
            polish_nonsym(
                &lp,
                &[NsBlock::Orthant(1)],
                &x,
                &[0.0],
                &[1.0],
                &mut backend
            )
            .is_none()
        );
    }
}
