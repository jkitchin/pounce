//! Sum-of-squares (SOS) **global lower bounds** for polynomial minimization
//! — the first step of polynomial global optimization on the SDP solver.
//!
//! For a polynomial `p(x)`, the SOS relaxation of `min_x p(x)` is
//!
//! ```text
//!   max γ   s.t.   p(x) − γ  is a sum of squares,
//! ```
//!
//! and `p(x) − γ` is SOS iff there is a PSD Gram matrix `Q ⪰ 0` with
//! `p(x) − γ = z(x)ᵀ Q z(x)`, where `z(x)` is the vector of monomials up to
//! degree `d = ⌈deg p / 2⌉`. Matching the coefficient of each monomial `xᵅ`
//! turns this into a semidefinite program:
//!
//! ```text
//!   max γ   s.t.   Σ_{βᵢ+βⱼ = α} Q_{ij} = p_α − γ·[α = 0],   Q ⪰ 0.
//! ```
//!
//! The optimal `γ*` is a **certified global lower bound**: `γ* ≤ min_x p(x)`
//! always, with equality whenever `p − p*` is itself SOS (e.g. univariate
//! polynomials, quadratics, and many low-degree cases — by Hilbert's
//! theorem not *every* nonnegative polynomial is SOS, so in general `γ*` can
//! be a strict lower bound). This is built as a conic program (one
//! [`crate::ConeSpec::Psd`] block plus coefficient-matching equalities) and
//! solved through [`crate::solve_socp_ipm`].

use crate::cones::psd::svec_index;
use crate::ipm::{solve_socp_ipm, QpOptions};
use crate::qp::{QpProblem, QpStatus, Triplet};
use crate::ConeSpec;
use pounce_linalg::symmetric_eigen;
use pounce_linsol::SparseSymLinearSolverInterface;
use std::collections::HashMap;

/// A sparse multivariate polynomial over `n_vars` variables: a list of
/// `(exponent vector, coefficient)` terms. The exponent vector has length
/// `n_vars`; e.g. over `(x, y)` the term `3·x²y` is `(vec![2, 1], 3.0)`.
#[derive(Debug, Clone)]
pub struct Polynomial {
    pub n_vars: usize,
    pub terms: Vec<(Vec<usize>, f64)>,
}

impl Polynomial {
    pub fn new(n_vars: usize, terms: Vec<(Vec<usize>, f64)>) -> Self {
        Polynomial { n_vars, terms }
    }

    /// Total degree (the largest term-exponent sum); `0` for a constant.
    pub fn degree(&self) -> usize {
        self.terms
            .iter()
            .map(|(e, _)| e.iter().sum::<usize>())
            .max()
            .unwrap_or(0)
    }

    /// Coefficients keyed by exponent vector (summing any duplicate terms).
    fn coeff_map(&self) -> HashMap<Vec<usize>, f64> {
        let mut m: HashMap<Vec<usize>, f64> = HashMap::new();
        for (e, c) in &self.terms {
            *m.entry(e.clone()).or_insert(0.0) += c;
        }
        m
    }
}

/// A constrained polynomial program `min p(x) s.t. gᵢ(x) ≥ 0, hⱼ(x) = 0`.
#[derive(Debug, Clone)]
pub struct PolyProblem {
    pub n_vars: usize,
    pub objective: Polynomial,
    /// Inequality constraints `gᵢ(x) ≥ 0`.
    pub inequalities: Vec<Polynomial>,
    /// Equality constraints `hⱼ(x) = 0`.
    pub equalities: Vec<Polynomial>,
}

impl PolyProblem {
    pub fn new(objective: Polynomial) -> Self {
        let n_vars = objective.n_vars;
        PolyProblem {
            n_vars,
            objective,
            inequalities: Vec::new(),
            equalities: Vec::new(),
        }
    }

    /// Add an inequality `g(x) ≥ 0`.
    pub fn ge(mut self, g: Polynomial) -> Self {
        self.inequalities.push(g);
        self
    }

    /// Add an equality `h(x) = 0`.
    pub fn eq(mut self, h: Polynomial) -> Self {
        self.equalities.push(h);
        self
    }
}

/// Result of the SOS relaxation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SosBound {
    /// The certified global lower bound `γ* ≤ min_x p(x)`.
    pub lower_bound: f64,
    /// Solve status of the underlying SDP.
    pub status: QpStatus,
}

/// All monomial exponent vectors over `n` variables with total degree
/// `≤ max_deg`, in a fixed (recursive) order.
fn monomials(n: usize, max_deg: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = vec![0usize; n];
    fn rec(pos: usize, remaining: usize, cur: &mut [usize], out: &mut Vec<Vec<usize>>) {
        if pos == cur.len() {
            out.push(cur.to_vec());
            return;
        }
        for e in 0..=remaining {
            cur[pos] = e;
            rec(pos + 1, remaining - e, cur, out);
        }
        cur[pos] = 0;
    }
    rec(0, max_deg, &mut cur, &mut out);
    out
}

/// Build and solve the unconstrained SOS lower-bound SDP for `p`, returning
/// the certified global lower bound. See the module docs for the model.
pub fn sos_lower_bound<F>(p: &Polynomial, mut make_backend: F) -> SosBound
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    sos_lower_bound_opts(p, &QpOptions::default(), &mut make_backend)
}

/// [`sos_lower_bound`] with explicit solver options.
pub fn sos_lower_bound_opts<F>(p: &Polynomial, opts: &QpOptions, make_backend: F) -> SosBound
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    sos_constrained_lower_bound_opts(&PolyProblem::new(p.clone()), None, opts, make_backend)
}

/// SOS / Lasserre lower bound for a **constrained** polynomial program
/// `min p s.t. gᵢ ≥ 0, hⱼ = 0` at relaxation order `order` (defaults to the
/// minimum admissible). Uses Putinar's representation
///
/// ```text
///   p(x) − γ = σ₀(x) + Σᵢ σᵢ(x) gᵢ(x) + Σⱼ λⱼ(x) hⱼ(x),
/// ```
///
/// with `σ₀, σᵢ` SOS (PSD Gram blocks; the *localizing* multipliers `σᵢ`
/// use the smaller basis of degree `d − ⌈deg gᵢ/2⌉`) and `λⱼ` free
/// polynomials. The returned `γ*` is a certified lower bound on `min p` over
/// the feasible set; raising `order` tightens it (the Lasserre hierarchy).
pub fn sos_constrained_lower_bound<F>(
    prob: &PolyProblem,
    order: Option<usize>,
    make_backend: F,
) -> SosBound
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    sos_constrained_lower_bound_opts(prob, order, &QpOptions::default(), make_backend)
}

/// The moment-side bookkeeping needed to recover the solution from the SDP
/// dual: the σ₀ monomial basis (= the moment-matrix index set) and the map
/// from a monomial `α` to the coefficient-matching equality whose dual
/// multiplier is the moment `y_α`.
struct MomentInfo {
    n_vars: usize,
    d: usize,
    basis0: Vec<Vec<usize>>,
    row_of: HashMap<Vec<usize>, usize>,
}

/// Build the SOS / Putinar SDP for `prob` at the given (clamped) order,
/// returning the conic program, its cones, and the moment bookkeeping.
///
/// `refine` selects the objective. `None` builds the ordinary lower-bound SDP
/// (`max γ` s.t. `p − γ` is in the Putinar cone) whose dual moments are the
/// analytic-center optimum. `Some(ε)` builds the **facial-reduction** SDP: the
/// objective polynomial is perturbed to `p + ε·θ` with the trace polynomial
/// `θ = Σ_{|β|≤d} x^{2β}`. Its dual moments then minimize `L(p) + ε·L(θ)` —
/// i.e. they pick the minimum-trace (lowest-rank) moment matrix among the
/// near-optimal ones, a standard nuclear-norm/low-rank surrogate. Because
/// `p + ε·θ` is coercive this stays as well-conditioned as the unperturbed
/// solve (unlike pinning `L(p)=γ*`, which is degenerate when `γ*≈0`), and the
/// recovered moment matrix is flat even when the optimum is non-unique. The
/// reported bound still comes from the unperturbed solve.
fn build_sos_sdp(
    prob: &PolyProblem,
    order: Option<usize>,
    refine: Option<f64>,
) -> (QpProblem, Vec<ConeSpec>, MomentInfo) {
    let n = prob.n_vars;
    let r2 = std::f64::consts::SQRT_2;

    // Minimum relaxation order, then honor a user-requested (larger) order.
    let mut d_min = prob.objective.degree().div_ceil(2);
    for g in &prob.inequalities {
        d_min = d_min.max(g.degree().div_ceil(2));
    }
    for h in &prob.equalities {
        d_min = d_min.max(h.degree().div_ceil(2));
    }
    let d = order.map_or(d_min, |o| o.max(d_min));
    let basis0 = monomials(n, d); // σ₀ basis = moment-matrix index set

    // Column layout: x = (γ, svec(Q₀), svec(Q₁)…, free λ coefficients…).
    let mut col = 1usize;
    let mut cones: Vec<ConeSpec> = Vec::new();
    let mut g_rows: Vec<Triplet> = Vec::new();
    let mut g_h: Vec<f64> = Vec::new();
    let mut by_mono: HashMap<Vec<usize>, Vec<(usize, f64)>> = HashMap::new();
    let unit = [(vec![0usize; n], 1.0)]; // weight ≡ 1 for σ₀

    // PSD (SOS) blocks: σ₀ (weight 1, basis degree d), then one localizing
    // multiplier per inequality (weight gᵢ, basis degree d − ⌈deg gᵢ/2⌉).
    let psd_specs = std::iter::once((d, &unit[..])).chain(
        prob.inequalities
            .iter()
            .map(|g| (d - g.degree().div_ceil(2), &g.terms[..])),
    );
    for (deg, weight) in psd_specs {
        let basis = monomials(n, deg);
        let bn = basis.len();
        let col_base = col;
        for i in 0..bn {
            for j in 0..=i {
                let coef0 = if i == j { 1.0 } else { r2 };
                let qcol = col_base + svec_index(bn, i, j);
                let base: Vec<usize> = basis[i].iter().zip(&basis[j]).map(|(a, b)| a + b).collect();
                for (delta, wc) in weight {
                    let alpha: Vec<usize> = base.iter().zip(delta).map(|(a, dd)| a + dd).collect();
                    by_mono.entry(alpha).or_default().push((qcol, coef0 * wc));
                }
            }
        }
        let sd = bn * (bn + 1) / 2;
        for k in 0..sd {
            let r = g_h.len();
            g_rows.push(Triplet::new(r, col_base + k, -1.0));
            g_h.push(0.0);
        }
        cones.push(ConeSpec::Psd(bn));
        col += sd;
    }

    // Free multipliers λⱼ for equalities: a free coefficient per monomial of
    // degree ≤ 2d − deg(hⱼ), contributing (× hⱼ's terms) with no cone.
    for h in &prob.equalities {
        let basis = monomials(n, 2 * d - h.degree());
        for nu in &basis {
            let lcol = col;
            col += 1;
            for (delta, hc) in &h.terms {
                let alpha: Vec<usize> = nu.iter().zip(delta).map(|(a, dd)| a + dd).collect();
                by_mono.entry(alpha).or_default().push((lcol, *hc));
            }
        }
    }

    let n_x = col;

    // Coefficient-matching RHS: the objective `p`, perturbed by `ε·θ` (with the
    // trace polynomial `θ = Σ_b x^{2b}`) when doing the facial-reduction solve.
    let pc = prob.objective.coeff_map();
    let mut rhs = pc.clone();
    if let Some(eps) = refine {
        for b in &basis0 {
            let dbl: Vec<usize> = b.iter().map(|e| 2 * e).collect();
            *rhs.entry(dbl).or_insert(0.0) += eps;
        }
    }

    // One coefficient-matching equality per distinct monomial; record the
    // monomial→row map so the equality duals can be read back as moments.
    let zero_exp = vec![0usize; n];
    let mut a: Vec<Triplet> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut row_of: HashMap<Vec<usize>, usize> = HashMap::new();
    for (alpha, terms) in &by_mono {
        let row = b.len();
        for &(c, coef) in terms {
            a.push(Triplet::new(row, c, coef));
        }
        if *alpha == zero_exp {
            a.push(Triplet::new(row, 0, 1.0)); // + γ
        }
        b.push(rhs.get(alpha).copied().unwrap_or(0.0));
        row_of.insert(alpha.clone(), row);
    }

    // Objective: maximize γ  ⇔  minimize −γ. (The refinement biases the dual
    // moments toward low trace purely through the perturbed RHS above.)
    let mut c = vec![0.0; n_x];
    c[0] = -1.0;

    let qp = QpProblem {
        n: n_x,
        p_lower: Vec::new(),
        c,
        a,
        b,
        g: g_rows,
        h: g_h,
        lb: Vec::new(),
        ub: Vec::new(),
    };
    (
        qp,
        cones,
        MomentInfo {
            n_vars: n,
            d,
            basis0,
            row_of,
        },
    )
}

/// [`sos_constrained_lower_bound`] with explicit solver options.
pub fn sos_constrained_lower_bound_opts<F>(
    prob: &PolyProblem,
    order: Option<usize>,
    opts: &QpOptions,
    make_backend: F,
) -> SosBound
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let (qp, cones, _moments) = build_sos_sdp(prob, order, None);
    let sol = solve_socp_ipm(&qp, &cones, opts, make_backend);
    SosBound {
        lower_bound: sol.x.first().copied().unwrap_or(f64::NEG_INFINITY),
        status: sol.status,
    }
}

/// The result of [`sos_minimize`]: the certified bound plus, when the moment
/// matrix is **flat** (exact relaxation), the global minimizer(s).
///
/// `is_exact` is a *sufficient* exactness certificate: when it holds,
/// `lower_bound` is provably the global minimum and `minimizers` are the
/// global optimizers.
///
/// An interior-point solver returns the **maximum-rank** (analytic-center)
/// optimal moment matrix, which is flat only when the optimal moment matrix is
/// unique — so a non-unique optimum would defeat flat truncation. To recover
/// these cases [`sos_minimize`] applies **facial reduction**: when the central
/// moment matrix is not flat it re-solves with a small trace penalty (a
/// low-rank surrogate) that collapses the spurious rank, so a non-unique but
/// exact optimum still certifies and all of its minimizers are extracted.
/// `is_exact` can still be `false` — e.g. when the relaxation order is too low
/// for flatness to be attainable (the moment-matrix rank exceeds the lower
/// basis dimension), or for a genuinely non-SOS-exact relaxation — but
/// `lower_bound` is a valid lower bound regardless.
#[derive(Debug, Clone, PartialEq)]
pub struct SosSolution {
    /// Certified global lower bound `γ*` (= the global minimum when `is_exact`).
    pub lower_bound: f64,
    pub status: QpStatus,
    /// `true` when the moment matrix is flat (`rank M_d = rank M_{d-1}`): the
    /// relaxation is then exact, so `lower_bound` is the global minimum.
    pub is_exact: bool,
    /// Number of global minimizers (the flat moment-matrix rank) when exact.
    pub num_minimizers: usize,
    /// The extracted global minimizers (all `num_minimizers` atoms) when the
    /// moment matrix is flat; recovered via the self-adjoint multiplication
    /// operators in the moment inner product (symmetric eigensolver only).
    pub minimizers: Vec<Vec<f64>>,
}

/// Solve `prob` by the SOS/Lasserre relaxation **and** recover the solution
/// from the moment matrix: certify exactness via flat truncation and extract
/// the global minimizer when it is unique. See [`SosSolution`].
pub fn sos_minimize<F>(prob: &PolyProblem, order: Option<usize>, mut make_backend: F) -> SosSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let (qp, cones, mi) = build_sos_sdp(prob, order, None);
    let sol = solve_socp_ipm(&qp, &cones, &QpOptions::default(), &mut make_backend);
    let lower_bound = sol.x.first().copied().unwrap_or(f64::NEG_INFINITY);
    if sol.status != QpStatus::Optimal {
        return SosSolution {
            lower_bound,
            status: sol.status,
            is_exact: false,
            num_minimizers: 0,
            minimizers: Vec::new(),
        };
    }

    let mut rec = recover_from_moments(&mi, &sol.y);

    // Facial reduction. The interior-point solver lands on the analytic-center
    // (maximum-rank) optimal moment matrix, which is flat only when the optimum
    // is unique; a non-unique optimum (free moment directions, or spurious
    // pseudo-moments invisible to a finite relaxation) inflates the rank and
    // defeats flat truncation. Re-solve with a small trace penalty `ε·θ` on the
    // objective (a low-rank / nuclear-norm surrogate): its moments collapse the
    // spurious rank, so an exact relaxation now certifies and the minimizers
    // can be extracted. The reported bound stays the unperturbed `γ*`.
    if !rec.is_exact {
        const TRACE_EPS: f64 = 1e-4;
        let (qp2, cones2, mi2) = build_sos_sdp(prob, order, Some(TRACE_EPS));
        let sol2 = solve_socp_ipm(&qp2, &cones2, &QpOptions::default(), &mut make_backend);
        if sol2.status == QpStatus::Optimal {
            let rec2 = recover_from_moments(&mi2, &sol2.y);
            if rec2.is_exact {
                rec = rec2;
            }
        }
    }

    SosSolution {
        lower_bound,
        status: sol.status,
        is_exact: rec.is_exact,
        num_minimizers: rec.num_minimizers,
        minimizers: rec.minimizers,
    }
}

/// Flat-truncation test + minimizer extraction from an SDP solution's moments.
struct Recovery {
    is_exact: bool,
    num_minimizers: usize,
    minimizers: Vec<Vec<f64>>,
}

/// Read the moment matrix out of the equality duals `y` (`y_α = y[row_of(α)]`,
/// with `y_0 = 1` by γ-stationarity up to a global sign), test flat truncation
/// (`rank M_d = rank M_{d−1}`), and extract the global minimizers when flat.
fn recover_from_moments(mi: &MomentInfo, y: &[f64]) -> Recovery {
    let moment = |alpha: &[usize]| -> f64 { y[mi.row_of[alpha]] };
    let zero = vec![0usize; mi.n_vars];
    let sign = if moment(&zero) < 0.0 { -1.0 } else { 1.0 };

    // Moment matrix M_d[i][j] = y_{basis0ᵢ + basis0ⱼ} (row-major).
    let big_n = mi.basis0.len();
    let mut m = vec![0.0; big_n * big_n];
    for i in 0..big_n {
        for j in 0..big_n {
            let a: Vec<usize> = mi.basis0[i]
                .iter()
                .zip(&mi.basis0[j])
                .map(|(p, q)| p + q)
                .collect();
            m[i * big_n + j] = sign * moment(&a);
        }
    }
    let rank_full = psd_rank(&m, big_n);

    // Flat truncation: compare with the rank on the degree-≤(d−1) sub-basis.
    let is_exact = if mi.d == 0 {
        true // a constant objective is trivially exact
    } else {
        let lower_idx: Vec<usize> = (0..big_n)
            .filter(|&i| mi.basis0[i].iter().sum::<usize>() < mi.d)
            .collect();
        let sub_n = lower_idx.len();
        let mut sub = vec![0.0; sub_n * sub_n];
        for (a, &ia) in lower_idx.iter().enumerate() {
            for (b, &ib) in lower_idx.iter().enumerate() {
                sub[a * sub_n + b] = m[ia * big_n + ib];
            }
        }
        psd_rank(&sub, sub_n) == rank_full
    };

    let num_minimizers = if is_exact { rank_full } else { 0 };
    let minimizers = if is_exact && rank_full >= 1 && mi.d >= 1 {
        extract_atoms(mi, rank_full, |alpha| sign * y[mi.row_of[alpha]])
    } else {
        Vec::new()
    };

    Recovery {
        is_exact,
        num_minimizers,
        minimizers,
    }
}

/// Extract the `r` global minimizers (atoms of the optimal measure) from a
/// flat moment matrix, using only the symmetric eigensolver.
///
/// Multiplication by a real variable `x_k` is **self-adjoint** in the moment
/// inner product `⟨f,g⟩ = L(fg)`, so whitening the degree-≤(d−1) moment
/// matrix `M` (`Wᵀ M W = I_r`) turns each multiplication operator into a
/// symmetric `r×r` matrix `B_k = Wᵀ M^{(k)} W`, where `M^{(k)}_{ij} =
/// y_{βᵢ+βⱼ+eₖ}` (a shifted moment matrix, available because flatness keeps
/// the degree ≤ 2d−1). The `B_k` commute, so a generic combination
/// `Σ cₖ Bₖ` is symmetric with the *common* eigenvectors `q_t`; the atoms'
/// coordinates are the Rayleigh quotients `x*_{t,k} = q_tᵀ Bₖ q_t`.
fn extract_atoms(mi: &MomentInfo, r: usize, moment: impl Fn(&[usize]) -> f64) -> Vec<Vec<f64>> {
    let n = mi.n_vars;
    // Quotient basis: monomials of degree ≤ d−1 (flatness ⇒ these span it).
    let sub: Vec<Vec<usize>> = mi
        .basis0
        .iter()
        .filter(|b| b.iter().sum::<usize>() < mi.d)
        .cloned()
        .collect();
    let s = sub.len();
    if s < r || r == 0 {
        return Vec::new();
    }
    let mono = |i: usize, j: usize, shift: Option<usize>| -> Vec<usize> {
        (0..n)
            .map(|t| sub[i][t] + sub[j][t] + usize::from(shift == Some(t)))
            .collect()
    };

    // M (s×s) and its top-r eigenpairs → whitening W (s×r), Wᵀ M W = I_r.
    let mut m = vec![0.0; s * s];
    for i in 0..s {
        for j in 0..s {
            m[i * s + j] = moment(&mono(i, j, None));
        }
    }
    let mut vals = vec![0.0; s];
    let mut vecs = vec![0.0; s * s]; // column-major eigenvectors, ascending
    if !symmetric_eigen(&m, s, &mut vals, &mut vecs) {
        return Vec::new();
    }
    // W column t ← eigenvector (s−1−t) scaled by 1/√λ.
    let mut w = vec![0.0; s * r]; // row-major s×r
    for t in 0..r {
        let e = s - 1 - t;
        let scale = 1.0 / vals[e].max(1e-12).sqrt();
        for i in 0..s {
            w[i * r + t] = vecs[e * s + i] * scale;
        }
    }

    // Whitened multiplication matrices B_k = Wᵀ M^{(k)} W  (r×r, symmetric).
    let mut bk: Vec<Vec<f64>> = Vec::with_capacity(n);
    for k in 0..n {
        let mut mk = vec![0.0; s * s];
        for i in 0..s {
            for j in 0..s {
                mk[i * s + j] = moment(&mono(i, j, Some(k)));
            }
        }
        // B = Wᵀ Mk W.
        let mut mw = vec![0.0; s * r]; // Mk · W
        for i in 0..s {
            for t in 0..r {
                let mut acc = 0.0;
                for j in 0..s {
                    acc += mk[i * s + j] * w[j * r + t];
                }
                mw[i * r + t] = acc;
            }
        }
        let mut b = vec![0.0; r * r];
        for a in 0..r {
            for c in 0..r {
                let mut acc = 0.0;
                for i in 0..s {
                    acc += w[i * r + a] * mw[i * r + c];
                }
                b[a * r + c] = acc;
            }
        }
        bk.push(b);
    }

    // Generic combination Σ cₖ Bₖ; its eigenvectors are the common atoms'
    // directions (cₖ = √(k+1) generically separates the combined eigenvalues).
    let mut comb = vec![0.0; r * r];
    for (k, b) in bk.iter().enumerate() {
        let ck = ((k + 1) as f64).sqrt();
        for idx in 0..r * r {
            comb[idx] += ck * b[idx];
        }
    }
    let mut cvals = vec![0.0; r];
    let mut cvecs = vec![0.0; r * r];
    if !symmetric_eigen(&comb, r, &mut cvals, &mut cvecs) {
        return Vec::new();
    }

    // Atom t: coordinate k = q_tᵀ B_k q_t (q_t orthonormal).
    let mut atoms = Vec::with_capacity(r);
    for t in 0..r {
        let q: Vec<f64> = (0..r).map(|i| cvecs[t * r + i]).collect();
        let atom: Vec<f64> = bk
            .iter()
            .map(|b| {
                let mut acc = 0.0;
                for a in 0..r {
                    for c in 0..r {
                        acc += q[a] * b[a * r + c] * q[c];
                    }
                }
                acc
            })
            .collect();
        atoms.push(atom);
    }
    atoms
}

/// Numerical rank of a symmetric PSD matrix (row-major `n×n`): the number of
/// eigenvalues exceeding `1e-6 · λ_max`.
fn psd_rank(mat: &[f64], n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut vals = vec![0.0; n];
    let mut vecs = vec![0.0; n * n];
    if !symmetric_eigen(mat, n, &mut vals, &mut vecs) {
        return n;
    }
    let max = vals.iter().cloned().fold(0.0_f64, f64::max);
    let tol = 1e-6 * max.max(1e-12);
    vals.iter().filter(|&&l| l > tol).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    #[test]
    fn monomial_count_is_binomial() {
        // #monomials over n vars of degree ≤ d is C(n+d, d).
        assert_eq!(monomials(1, 2).len(), 3); // 1, x, x²
        assert_eq!(monomials(2, 1).len(), 3); // 1, x, y
        assert_eq!(monomials(2, 2).len(), 6); // 1,x,y,x²,xy,y²
        assert_eq!(monomials(3, 2).len(), 10);
    }

    #[test]
    fn univariate_quartic_known_minimum() {
        // p(x) = x⁴ − 2x² + 3.  p' = 4x³ − 4x = 0 ⇒ x = 0, ±1; min at ±1 is
        // 1 − 2 + 3 = 2.  p − 2 = (x² − 1)² is SOS, so the bound is exact.
        let p = Polynomial::new(1, vec![(vec![4], 1.0), (vec![2], -2.0), (vec![0], 3.0)]);
        let r = sos_lower_bound(&p, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(
            (r.lower_bound - 2.0).abs() < 1e-5,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn shifted_paraboloid_two_vars() {
        // p(x,y) = (x−1)² + y² = x² − 2x + 1 + y².  Min 0 at (1, 0); SOS-exact.
        let p = Polynomial::new(
            2,
            vec![
                (vec![2, 0], 1.0),
                (vec![1, 0], -2.0),
                (vec![0, 0], 1.0),
                (vec![0, 2], 1.0),
            ],
        );
        let r = sos_lower_bound(&p, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(r.lower_bound.abs() < 1e-5, "bound = {}", r.lower_bound);
    }

    #[test]
    fn constant_polynomial() {
        // p ≡ 7: the global minimum (and SOS bound) is 7.
        let p = Polynomial::new(1, vec![(vec![0], 7.0)]);
        let r = sos_lower_bound(&p, backend);
        assert_eq!(r.status, QpStatus::Optimal);
        assert!(
            (r.lower_bound - 7.0).abs() < 1e-6,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn quadratic_lower_bound() {
        // p(x) = x² − 4x + 5 = (x−2)² + 1.  Min 1; basis degree d = 1.
        let p = Polynomial::new(1, vec![(vec![2], 1.0), (vec![1], -4.0), (vec![0], 5.0)]);
        let r = sos_lower_bound(&p, backend);
        assert_eq!(r.status, QpStatus::Optimal);
        assert!(
            (r.lower_bound - 1.0).abs() < 1e-5,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn constrained_linear_lower_bound() {
        // min x s.t. x − 1 ≥ 0  ⇒  min = 1 (the constraint binds).
        let prob = PolyProblem::new(Polynomial::new(1, vec![(vec![1], 1.0)]))
            .ge(Polynomial::new(1, vec![(vec![1], 1.0), (vec![0], -1.0)]));
        let r = sos_constrained_lower_bound(&prob, None, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(
            (r.lower_bound - 1.0).abs() < 1e-5,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn constrained_nonconvex_box() {
        // min −x s.t. 1 − x² ≥ 0  (x ∈ [−1,1])  ⇒  min = −1 at x = 1.
        // The localizing multiplier σ₁ (a nonneg scalar) makes the bound
        // exact — a nonconvex feasible-set bound from the SDP.
        let prob = PolyProblem::new(Polynomial::new(1, vec![(vec![1], -1.0)]))
            .ge(Polynomial::new(1, vec![(vec![0], 1.0), (vec![2], -1.0)]));
        let r = sos_constrained_lower_bound(&prob, None, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(
            (r.lower_bound + 1.0).abs() < 1e-5,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn constrained_equality_lower_bound() {
        // min x² + y² s.t. x + y − 2 = 0  ⇒  min = 2 at (1,1), via a free
        // multiplier λ(x,y) for the equality.
        let obj = Polynomial::new(2, vec![(vec![2, 0], 1.0), (vec![0, 2], 1.0)]);
        let prob = PolyProblem::new(obj).eq(Polynomial::new(
            2,
            vec![(vec![1, 0], 1.0), (vec![0, 1], 1.0), (vec![0, 0], -2.0)],
        ));
        let r = sos_constrained_lower_bound(&prob, None, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(
            (r.lower_bound - 2.0).abs() < 1e-5,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn extract_unique_minimizer_1d() {
        // p(x) = x² − 4x + 5 = (x−2)² + 1.  Unique min x* = 2, value 1.
        let p = Polynomial::new(1, vec![(vec![2], 1.0), (vec![1], -4.0), (vec![0], 5.0)]);
        let s = sos_minimize(&PolyProblem::new(p), None, backend);
        assert_eq!(s.status, QpStatus::Optimal);
        assert!(s.is_exact, "should be flat/exact");
        assert_eq!(s.num_minimizers, 1);
        assert_eq!(s.minimizers.len(), 1);
        assert!(
            (s.minimizers[0][0] - 2.0).abs() < 1e-4,
            "x* = {:?}",
            s.minimizers[0]
        );
        assert!((s.lower_bound - 1.0).abs() < 1e-5);
    }

    #[test]
    fn extract_unique_minimizer_2d() {
        // p(x,y) = (x−1)² + (y−2)².  Unique min (1, 2), value 0.
        let p = Polynomial::new(
            2,
            vec![
                (vec![2, 0], 1.0),
                (vec![1, 0], -2.0),
                (vec![0, 2], 1.0),
                (vec![0, 1], -4.0),
                (vec![0, 0], 5.0),
            ],
        );
        let s = sos_minimize(&PolyProblem::new(p), None, backend);
        assert_eq!(s.status, QpStatus::Optimal);
        assert!(s.is_exact);
        assert_eq!(s.num_minimizers, 1);
        let x = &s.minimizers[0];
        assert!(
            (x[0] - 1.0).abs() < 1e-4 && (x[1] - 2.0).abs() < 1e-4,
            "x* = {x:?}"
        );
    }

    #[test]
    fn extracts_two_global_minimizers() {
        // p(x) = x⁴ − 2x² + 3 has TWO global minimizers x = ±1 (value 2).
        // The relaxation is flat (moment-matrix rank 2) and the multi-atom
        // extraction recovers both points.
        let p = Polynomial::new(1, vec![(vec![4], 1.0), (vec![2], -2.0), (vec![0], 3.0)]);
        let s = sos_minimize(&PolyProblem::new(p), None, backend);
        assert_eq!(s.status, QpStatus::Optimal);
        assert!(s.is_exact, "flat truncation should hold");
        assert_eq!(s.num_minimizers, 2, "two atoms at ±1");
        assert_eq!(s.minimizers.len(), 2);
        let mut roots: Vec<f64> = s.minimizers.iter().map(|m| m[0]).collect();
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((roots[0] + 1.0).abs() < 1e-3, "min root {}", roots[0]);
        assert!((roots[1] - 1.0).abs() < 1e-3, "max root {}", roots[1]);
        assert!((s.lower_bound - 2.0).abs() < 1e-5);
    }

    #[test]
    fn facial_reduction_recovers_nonunique_minimizers() {
        // p(x,y) = (x²−1)² + y², global min 0 at (±1, 0). The objective is
        // SOS so the bound is exact (0), but the optimum is non-unique: the
        // interior-point solver lands on the analytic-center moment matrix,
        // whose rank is inflated by a spurious pseudo-moment direction
        // (L(y⁴) > 0 while L(y²) = 0), so plain flat truncation fails. The
        // facial-reduction (minimum-trace) re-solve collapses that rank and
        // recovers both minimizers.
        let p = Polynomial::new(
            2,
            vec![
                (vec![4, 0], 1.0),
                (vec![2, 0], -2.0),
                (vec![0, 0], 1.0),
                (vec![0, 2], 1.0),
            ],
        );
        let s = sos_minimize(&PolyProblem::new(p), None, backend);
        assert_eq!(s.status, QpStatus::Optimal);
        assert!(s.lower_bound.abs() < 1e-5, "bound = {}", s.lower_bound);
        assert!(s.is_exact, "facial reduction should certify exactness");
        assert_eq!(s.num_minimizers, 2, "two atoms at (±1, 0)");
        let mut xs: Vec<f64> = s.minimizers.iter().map(|m| m[0]).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((xs[0] + 1.0).abs() < 1e-2, "x⁻ = {}", xs[0]);
        assert!((xs[1] - 1.0).abs() < 1e-2, "x⁺ = {}", xs[1]);
        for atom in &s.minimizers {
            assert!(atom[1].abs() < 1e-2, "y = {}", atom[1]);
        }
    }
}
