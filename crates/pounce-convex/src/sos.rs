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

use crate::ConeSpec;
use crate::cones::psd::svec_index;
use crate::ipm::{QpOptions, solve_socp_ipm};
use crate::qp::{QpProblem, QpStatus, Triplet};
use pounce_linalg::symmetric_eigen;
use pounce_linsol::SparseSymLinearSolverInterface;
use std::collections::{BTreeMap, HashMap};

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

    /// Largest coefficient magnitude (∞-norm of the coefficient vector, after
    /// summing duplicate terms). `0.0` for the zero polynomial.
    fn coeff_inf_norm(&self) -> f64 {
        self.coeff_map().values().fold(0.0, |m, c| m.max(c.abs()))
    }

    /// Evaluate the polynomial at a point `x` (length `n_vars`).
    fn eval(&self, x: &[f64]) -> f64 {
        self.terms
            .iter()
            .map(|(e, c)| c * Self::monomial(e, x))
            .sum()
    }

    /// Triangle-inequality magnitude bound `Σ|cᵢ|·∏ₖ|xₖ|^{eₖ}` at `x` — an upper
    /// bound on `|eval(x)|` used to set a scale-invariant feasibility tolerance.
    fn eval_magnitude(&self, x: &[f64]) -> f64 {
        self.terms
            .iter()
            .map(|(e, c)| c.abs() * Self::monomial(e, x).abs())
            .sum()
    }

    /// `∏ₖ xₖ^{eₖ}` for a single exponent vector at `x`.
    fn monomial(e: &[usize], x: &[f64]) -> f64 {
        e.iter()
            .enumerate()
            .map(|(k, &pw)| x[k].powi(pw as i32))
            .product()
    }

    /// If this polynomial is a single variable bound `a·xⱼ + c ≥ 0`, the bound
    /// it imposes: `(j, value, is_upper)`.
    ///
    /// `a > 0` reads as the lower bound `xⱼ ≥ −c/a`, `a < 0` as the upper bound
    /// `xⱼ ≤ −c/a`. Anything else — two variables, a higher-degree term, an
    /// extra monomial — is not a box side and returns `None`. Used to find a
    /// variable's range for domain normalization (see
    /// [`PolyProblem::equilibrated`]).
    fn as_variable_bound(&self) -> Option<(usize, f64, bool)> {
        let mut linear: Option<(usize, f64)> = None;
        let mut constant = 0.0;
        for (e, c) in &self.coeff_map() {
            if *c == 0.0 {
                continue;
            }
            match e.iter().sum::<usize>() {
                0 => constant += c,
                1 => {
                    let j = e.iter().position(|&p| p == 1)?;
                    // A second linear term means two variables are coupled.
                    if linear.is_some() {
                        return None;
                    }
                    linear = Some((j, *c));
                }
                _ => return None,
            }
        }
        let (j, a) = linear?;
        Some((j, -constant / a, a < 0.0))
    }

    /// If this polynomial is `c − a·xⱼ² ≥ 0` with `a, c > 0`, the symmetric
    /// range it imposes: `(j, √(c/a))`, i.e. `|xⱼ| ≤ √(c/a)`.
    ///
    /// The idiomatic way to write a ball/box side in an SOS model, and common
    /// enough that missing it would leave obviously-bounded problems
    /// uncertifiable (`min −x s.t. 1 − x² ≥ 0` is the textbook example).
    fn as_variable_square_bound(&self) -> Option<(usize, f64)> {
        let mut square: Option<(usize, f64)> = None;
        let mut constant = 0.0;
        for (e, c) in &self.coeff_map() {
            if *c == 0.0 {
                continue;
            }
            match e.iter().sum::<usize>() {
                0 => constant += c,
                2 => {
                    // Must be `xⱼ²`, not a cross term `xⱼxₖ`.
                    let j = e.iter().position(|&p| p == 2)?;
                    if square.is_some() {
                        return None;
                    }
                    square = Some((j, *c));
                }
                _ => return None,
            }
        }
        let (j, a) = square?;
        // `c − a·xⱼ² ≥ 0` bounds xⱼ only when the square enters negatively and
        // the constant is nonnegative.
        if a >= 0.0 || constant < 0.0 {
            return None;
        }
        Some((j, (constant / -a).sqrt()))
    }

    /// Substitute `xⱼ = shiftⱼ + scaleⱼ·uⱼ` into this polynomial, returning it
    /// as a polynomial in `u`.
    ///
    /// Each monomial `∏ⱼ xⱼ^{eⱼ}` expands binomially, one variable at a time:
    /// `(shiftⱼ + scaleⱼuⱼ)^{eⱼ} = Σₖ C(eⱼ,k)·shiftⱼ^{eⱼ−k}·scaleⱼ^k·uⱼ^k`.
    /// Terms are accumulated through a `BTreeMap` so the result is ordered
    /// deterministically regardless of the input term order.
    fn affine_substitute(&self, shift: &[f64], scale: &[f64]) -> Polynomial {
        let n = self.n_vars;
        let mut acc: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
        for (e, c) in &self.terms {
            let mut cur: Vec<(Vec<usize>, f64)> = vec![(vec![0usize; n], *c)];
            for j in 0..n {
                let ej = e[j];
                if ej == 0 {
                    continue;
                }
                let mut next = Vec::with_capacity(cur.len() * (ej + 1));
                let mut binom = 1.0_f64;
                for k in 0..=ej {
                    let w = binom * shift[j].powi((ej - k) as i32) * scale[j].powi(k as i32);
                    if w != 0.0 {
                        for (m, cc) in &cur {
                            let mut m2 = m.clone();
                            m2[j] += k;
                            next.push((m2, cc * w));
                        }
                    }
                    // C(e, k+1) = C(e, k)·(e−k)/(k+1).
                    binom = binom * (ej - k) as f64 / (k + 1) as f64;
                }
                cur = next;
            }
            for (m, cc) in cur {
                *acc.entry(m).or_insert(0.0) += cc;
            }
        }
        Polynomial {
            n_vars: n,
            terms: acc.into_iter().filter(|(_, c)| *c != 0.0).collect(),
        }
    }

    /// This polynomial with every coefficient divided by `s`.
    fn scaled(&self, s: f64) -> Polynomial {
        Polynomial {
            n_vars: self.n_vars,
            terms: self.terms.iter().map(|(e, c)| (e.clone(), c / s)).collect(),
        }
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

    /// Whether a candidate point `x` lies in the feasible set
    /// `K = {gᵢ(x) ≥ 0, hⱼ(x) = 0}`, to a scale-invariant tolerance.
    ///
    /// Each constraint is judged on a *relative* margin: a point is infeasible
    /// only when `gᵢ(x) < −tol·(1 + ‖gᵢ‖(x))` (or `|hⱼ(x)| > tol·(1 + ‖hⱼ‖(x))`),
    /// where `‖·‖(x)` is the triangle-inequality magnitude bound at `x`. This
    /// tolerates the ~1e-4 inaccuracy of moment-extracted atoms (a binding
    /// constraint reads `gᵢ ≈ 0`) while still catching a clear violation.
    fn is_feasible(&self, x: &[f64], tol: f64) -> bool {
        let ineq_ok = self
            .inequalities
            .iter()
            .all(|g| g.eval(x) >= -tol * (1.0 + g.eval_magnitude(x)));
        let eq_ok = self
            .equalities
            .iter()
            .all(|h| h.eval(x).abs() <= tol * (1.0 + h.eval_magnitude(x)));
        ineq_ok && eq_ok
    }

    /// Coefficient-equilibrated copy for conditioning the moment SDP (gh #124),
    /// plus the objective scale `s_obj` by which a recovered lower bound must be
    /// multiplied to undo the scaling.
    ///
    /// Each polynomial is divided by its own largest coefficient magnitude.
    /// This is value- and minimizer-preserving: dividing the objective by a
    /// constant `s_obj > 0` divides every objective value (and hence the bound
    /// `γ*`) by `s_obj` without moving `argmin`, and dividing a constraint
    /// `gᵢ ≥ 0` / `hⱼ = 0` by a positive constant leaves the feasible set
    /// unchanged. The net effect is an O(1)-coefficient problem whose moment
    /// matrix stays well conditioned: on the standard degree-8 Goldstein-Price
    /// benchmark (coefficients spanning 144..23616) the raw problem returns NaN
    /// (`numerical_failure` / `iteration_limit`) while the equilibrated one
    /// certifies the exact bound in ~2 s. A zero/empty polynomial scales by
    /// `1.0` (nothing to do).
    fn equilibrated(&self) -> (PolyProblem, Rescaling) {
        fn nonzero_norm(p: &Polynomial) -> f64 {
            let s = p.coeff_inf_norm();
            if s > 0.0 { s } else { 1.0 }
        }
        let (shift, scale, boxed) = self.domain_normalization();
        // Domain first, coefficients second: the substitution rewrites every
        // coefficient, so equilibrating before it would be undone.
        let sub = |p: &Polynomial| p.affine_substitute(&shift, &scale);
        let objective = sub(&self.objective);
        let s_obj = nonzero_norm(&objective);
        let scaled = PolyProblem {
            n_vars: self.n_vars,
            objective: objective.scaled(s_obj),
            inequalities: self
                .inequalities
                .iter()
                .map(|g| {
                    let g = sub(g);
                    g.scaled(nonzero_norm(&g))
                })
                .collect(),
            equalities: self
                .equalities
                .iter()
                .map(|h| {
                    let h = sub(h);
                    h.scaled(nonzero_norm(&h))
                })
                .collect(),
        };
        (
            scaled,
            Rescaling {
                s_obj,
                shift,
                scale,
                boxed,
            },
        )
    }

    /// Per-variable affine map `xⱼ = shiftⱼ + scaleⱼ·uⱼ` normalizing every
    /// boxed variable's range onto `[−1, 1]`.
    ///
    /// A variable is boxed when the inequality list carries both a lower and an
    /// upper bound for it as standalone constraints (`xⱼ − l ≥ 0`, `u − xⱼ ≥ 0`
    /// and their scalar multiples); the tightest of each is used. Variables
    /// without a finite box, or with a degenerate one (`u ≤ l`), map by the
    /// identity `(0, 1)`.
    ///
    /// This complements coefficient equilibration, which normalizes each
    /// polynomial's coefficients but leaves the *domain* alone. The moment
    /// matrix entries are monomials in `x`, so a wide box makes them span
    /// decades on their own: on gh #218's quartic over `x₁ ∈ [0,3]` the
    /// degree-8 moments span `3⁸ ≈ 6561` against `1`, and no amount of
    /// coefficient scaling touches that. Normalizing the domain moves the
    /// order-3 relaxation there from a frozen `IterationLimit` (no bound at
    /// all) to `OptimalInaccurate` at `−6.667`, a valid bound tighter than the
    /// trivial `−7`.
    ///
    /// The map is value- and minimizer-preserving: it is a bijection of the
    /// feasible set, so the minimum is unchanged and a recovered minimizer maps
    /// back through [`Rescaling::unmap`].
    fn domain_normalization(&self) -> (Vec<f64>, Vec<f64>, bool) {
        let mut lower = vec![f64::NEG_INFINITY; self.n_vars];
        let mut upper = vec![f64::INFINITY; self.n_vars];
        for g in &self.inequalities {
            if let Some((j, v, is_upper)) = g.as_variable_bound() {
                if is_upper {
                    upper[j] = upper[j].min(v);
                } else {
                    lower[j] = lower[j].max(v);
                }
            } else if let Some((j, r)) = g.as_variable_square_bound() {
                lower[j] = lower[j].max(-r);
                upper[j] = upper[j].min(r);
            }
        }
        let mut shift = vec![0.0; self.n_vars];
        let mut scale = vec![1.0; self.n_vars];
        let mut boxed = true;
        for j in 0..self.n_vars {
            let (l, u) = (lower[j], upper[j]);
            if l.is_finite() && u.is_finite() && u > l {
                shift[j] = 0.5 * (l + u);
                scale[j] = 0.5 * (u - l);
            } else {
                boxed = false;
            }
        }
        (shift, scale, boxed)
    }
}

/// The change of variables [`PolyProblem::equilibrated`] applied, and how to
/// undo it on a recovered bound and minimizer.
struct Rescaling {
    /// The objective's coefficient scale: a bound computed on the equilibrated
    /// problem is multiplied by this to recover the original one.
    s_obj: f64,
    /// Per-variable `xⱼ = shiftⱼ + scaleⱼ·uⱼ` (identity `(0, 1)` when unboxed).
    shift: Vec<f64>,
    scale: Vec<f64>,
    /// Whether *every* variable was boxed, so the normalized feasible set is
    /// contained in `[−1, 1]ⁿ`. This is the precondition for a rigorous bound
    /// (see [`certified_slack`]); with even one variable unbounded, the
    /// residual polynomial cannot be bounded over the feasible set at all.
    boxed: bool,
}

impl Rescaling {
    /// Map a point from the normalized `u` coordinates back to the caller's `x`.
    fn unmap(&self, u: &[f64]) -> Vec<f64> {
        u.iter()
            .enumerate()
            .map(|(j, v)| self.shift[j] + self.scale[j] * v)
            .collect()
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
    sos_lower_bound_opts(p, &sos_opts(), &mut make_backend)
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
    sos_constrained_lower_bound_opts(prob, order, &sos_opts(), make_backend)
}

/// Default solver options for an SOS/moment SDP — the base a caller should
/// build on (`QpOptions { tol: 1e-6, ..sos_opts() }`) rather than starting from
/// [`QpOptions::default`], so the HSDE choice below is preserved.
///
/// SOS relaxations are *degenerate by design*: an exact relaxation has a
/// rank-deficient optimal moment matrix sitting on the PSD-cone boundary, where
/// the Nesterov–Todd scaling has unbounded dynamic range. The infeasible-start
/// symmetric driver stalls or diverges there (e.g. the order-3 trace-penalty
/// refinement ran to the iteration limit and drifted to a `-6e7` "bound");
/// the homogeneous self-dual embedding stays well-conditioned on the same
/// problems (≈10 iterations), so SOS solves default to it.
pub fn sos_opts() -> QpOptions {
    QpOptions {
        use_hsde: true,
        ..QpOptions::default()
    }
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
    /// Each SOS block's `(column base in x, basis dimension)`, in build order:
    /// σ₀ first, then one localizing block per inequality. Used to read the
    /// Gram matrices back out of the primal for [`certified_slack`].
    psd_blocks: Vec<(usize, usize)>,
}

/// A rigorous bound on how far the computed SOS certificate misses the exact
/// polynomial identity, given that the feasible set lies inside the unit box.
///
/// The SDP asks for `p − γ = σ₀ + Σᵢ σᵢ gᵢ + Σⱼ λⱼ hⱼ`, one linear equation per
/// monomial. A converged interior-point solve satisfies those equations only to
/// its tolerance, and its Gram matrices are only *approximately* PSD — so the
/// `γ` it reports can, and on gh #218's order-4 relaxation does, sit slightly
/// **above** the true minimum. That is the one genuinely unsound failure mode
/// for a lower bound, and no tightening of the solve removes it: it is a
/// property of finite precision, not of convergence.
///
/// The fix is to stop trusting the identity and measure it. Project every Gram
/// block onto the PSD cone (clamp negative eigenvalues to zero) so the `σ` are
/// *exactly* SOS, then evaluate the residual of the coefficient-matching system
/// at the projected point: `e = b − A·x'`. As polynomials that says
///
/// ```text
///   p(u) − γ = σ₀(u) + Σᵢ σᵢ(u) gᵢ(u) + Σⱼ λⱼ(u) hⱼ(u) + e(u),
/// ```
///
/// where every term but `e` is nonnegative on the feasible set — the `σ` are
/// SOS by construction now, the `gᵢ` are nonnegative there by definition, and
/// the `hⱼ` vanish there. Hence `p ≥ γ + e` on the feasible set, and with
/// `|u^α| ≤ 1` on `[−1,1]ⁿ` the crudest possible bound on `e` is already enough:
///
/// ```text
///   p(u) ≥ γ − Σ_α |e_α|   for all u in the feasible set.
/// ```
///
/// This function returns that `Σ_α |e_α|`. Subtracting it turns a bound that is
/// merely accurate into one that is *valid*, and it costs one eigendecomposition
/// per block plus a sparse matvec — no extra solve.
///
/// The `|u^α| ≤ 1` step is what requires the unit box, which is why
/// certification is available only when every variable is boxed (see
/// [`Rescaling::boxed`]). On an unbounded domain a nonzero residual cannot be
/// bounded at all: a residual with a negative leading coefficient is unbounded
/// below, so no finite correction exists. Certifying there needs the residual
/// driven to *exactly* zero in exact arithmetic (rational Gram recovery), which
/// is a different technique entirely.
fn certified_slack(qp: &QpProblem, mi: &MomentInfo, x: &[f64]) -> Option<f64> {
    let mut xp = x.to_vec();
    for &(col_base, bn) in &mi.psd_blocks {
        let sd = bn * (bn + 1) / 2;
        // svec -> dense symmetric (off-diagonals carry a √2 in svec).
        let mut m = vec![0.0; bn * bn];
        crate::cones::psd::smat(&xp[col_base..col_base + sd], bn, &mut m);
        let mut vals = vec![0.0; bn];
        let mut vecs = vec![0.0; bn * bn];
        if !symmetric_eigen(&m, bn, &mut vals, &mut vecs) {
            return None;
        }
        // Already PSD to working precision: leave the block untouched so a
        // clean solve pays nothing for the projection's round-off.
        if vals[0] >= 0.0 {
            continue;
        }
        // Q₊ = Σ max(λ_k, 0) v_k v_kᵀ, rebuilt straight back into svec.
        let r2 = std::f64::consts::SQRT_2;
        for i in 0..bn {
            for j in 0..=i {
                let mut acc = 0.0;
                for k in 0..bn {
                    if vals[k] > 0.0 {
                        acc += vals[k] * vecs[k * bn + i] * vecs[k * bn + j];
                    }
                }
                let scale = if i == j { 1.0 } else { r2 };
                xp[col_base + svec_index(bn, i, j)] = scale * acc;
            }
        }
    }

    // e = b − A·x' over the coefficient-matching rows, summed in absolute value.
    let mut ax = vec![0.0; qp.b.len()];
    for t in &qp.a {
        ax[t.row] += t.val * xp[t.col];
    }
    Some(qp.b.iter().zip(&ax).map(|(bi, axi)| (bi - axi).abs()).sum())
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
/// The smallest relaxation order that can represent `prob`: every polynomial
/// must fit inside the degree-`2d` window, so `d ≥ ⌈deg/2⌉` for each of them.
fn min_relaxation_order(prob: &PolyProblem) -> usize {
    let mut d = prob.objective.degree().div_ceil(2);
    for g in &prob.inequalities {
        d = d.max(g.degree().div_ceil(2));
    }
    for h in &prob.equalities {
        d = d.max(h.degree().div_ceil(2));
    }
    d
}

fn build_sos_sdp(
    prob: &PolyProblem,
    order: Option<usize>,
    refine: Option<f64>,
) -> (QpProblem, Vec<ConeSpec>, MomentInfo) {
    let n = prob.n_vars;
    let r2 = std::f64::consts::SQRT_2;

    // Minimum relaxation order, then honor a user-requested (larger) order.
    let d_min = min_relaxation_order(prob);
    let d = order.map_or(d_min, |o| o.max(d_min));
    let basis0 = monomials(n, d); // σ₀ basis = moment-matrix index set

    // Column layout: x = (γ, svec(Q₀), svec(Q₁)…, free λ coefficients…).
    let mut col = 1usize;
    let mut cones: Vec<ConeSpec> = Vec::new();
    let mut g_rows: Vec<Triplet> = Vec::new();
    let mut g_h: Vec<f64> = Vec::new();
    // BTreeMap (not HashMap) so the coefficient-matching rows below are emitted
    // in a deterministic, sorted-by-monomial order. With a HashMap the SDP's row
    // ordering — and hence the solver's floating-point path and results — varied
    // run-to-run (M22).
    let mut by_mono: BTreeMap<Vec<usize>, Vec<(usize, f64)>> = BTreeMap::new();
    let mut psd_blocks: Vec<(usize, usize)> = Vec::new();
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
        psd_blocks.push((col_base, bn));
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
            psd_blocks,
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
    // Equilibrate coefficients and normalize the domain before assembling the
    // SDP (gh #124, gh #218); undo the objective scale on the recovered bound.
    // The change of variables is value-preserving, so no other correction is
    // needed for a bound-only solve.
    let (prob, resc) = prob.equilibrated();
    let (qp, cones, _moments) = build_sos_sdp(&prob, order, None);
    let sol = solve_socp_ipm(&qp, &cones, opts, make_backend);
    SosBound {
        lower_bound: sol.x.first().copied().unwrap_or(f64::NEG_INFINITY) * resc.s_obj,
        status: sol.status,
    }
}

/// The result of [`sos_minimize`]: the certified bound plus, when the moment
/// matrix is **flat** (exact relaxation), the global minimizer(s).
///
/// `is_exact` is a *sufficient* exactness certificate: when it holds,
/// `lower_bound` is provably the global minimum and `minimizers` are the
/// global optimizers. For a constrained program it is only set once the
/// extracted atoms have been checked to lie in the feasible set — flat
/// truncation of the moment matrix `M_d` alone does not guarantee that (see
/// [`sos_minimize`]), so an atom that violates a constraint withdraws the
/// certificate (`is_exact = false`) while `lower_bound` stays a valid bound.
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
    ///
    /// **May under-report.** When the relaxation is not flat at the analytic
    /// center, the low-rank re-solve can collapse a multi-atom optimal measure
    /// onto fewer atoms; the survivors are validated against the certified
    /// bound (so they are genuine minimizers) but siblings can be missed.
    /// Himmelblau's function at order 2 reports 1 of its 4. Treat this as a
    /// lower bound on the number of global minimizers, not a count.
    pub num_minimizers: usize,
    /// The extracted global minimizers (all `num_minimizers` atoms) when the
    /// moment matrix is flat; recovered via the self-adjoint multiplication
    /// operators in the moment inner product (symmetric eigensolver only).
    pub minimizers: Vec<Vec<f64>>,
    /// Whether `lower_bound` is **rigorous**: proved to be a true lower bound,
    /// not merely accurate to the solver's tolerance.
    ///
    /// An uncertified bound is the raw `γ` the SDP reported. It is normally
    /// correct to several digits, but it can — and on hard relaxations does —
    /// land slightly *above* the true minimum, which makes it not a lower bound
    /// at all. A certified bound has the identity's measured residual
    /// subtracted, so it is valid no matter how the solve went.
    ///
    /// Certification requires the feasible set to lie in a box readable from
    /// the constraints; it is `false` for an unbounded feasible set (an
    /// unconstrained problem, say) where no finite correction can exist.
    pub certified: bool,
    /// The relaxation order that actually produced `lower_bound`.
    ///
    /// Normally the requested order. It is *lower* when that order failed to
    /// converge and a coarser one did — see [`sos_minimize`], which falls back
    /// rather than discard a bound it already proved. Always check this before
    /// reading a converged result as a statement about the order you asked for.
    pub order: usize,
}

/// Solve `prob` by the SOS/Lasserre relaxation **and** recover the solution
/// from the moment matrix: certify exactness via flat truncation and extract
/// the global minimizer when it is unique. See [`SosSolution`].
///
/// If the requested order does not converge, successively coarser orders are
/// tried down to the minimum admissible one, and the first that converges is
/// returned with its own [`order`](SosSolution::order). A bound from a lower
/// order is a *valid* bound on the same problem — just a weaker one — so
/// discarding it would throw away a certificate already computed (gh #218). The
/// fallback costs extra solves only on a failure, and when nothing converges
/// the requested order's own (non-converged) result is what comes back.
pub fn sos_minimize<F>(prob: &PolyProblem, order: Option<usize>, make_backend: F) -> SosSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    sos_minimize_opts(prob, order, &sos_opts(), make_backend)
}

/// [`sos_minimize`] with explicit solver options.
///
/// Only `tol` and `max_iter` are worth touching; the rest of [`QpOptions`] is
/// fixed by [`sos_opts`] because a moment SDP needs the homogeneous self-dual
/// embedding to stay conditioned. Loosening `tol` can rescue a relaxation that
/// would otherwise not converge, at the cost of a weaker certificate — the
/// bound stays *valid* either way when it is certified, since the certification
/// slack measures the actual miss rather than assuming the solve converged.
pub fn sos_minimize_opts<F>(
    prob: &PolyProblem,
    order: Option<usize>,
    opts: &QpOptions,
    mut make_backend: F,
) -> SosSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    // Equilibrate coefficients and normalize the domain before assembling the
    // SDP (gh #124, gh #218). The scaling is value- and minimizer-preserving
    // (see `PolyProblem::equilibrated`), so `is_exact` and the extracted
    // minimizers are unchanged; the bound is recovered by multiplying back by
    // `s_obj` and the minimizers by `Rescaling::unmap`.
    let (prob, resc) = prob.equilibrated();
    let prob = &prob;
    let d_min = min_relaxation_order(prob);
    let requested = order.map_or(d_min, |o| o.max(d_min));

    let requested_result = sos_minimize_at(prob, &resc, requested, opts, &mut make_backend);
    if requested_result.status == QpStatus::Optimal {
        return requested_result;
    }
    for d in (d_min..requested).rev() {
        let sol = sos_minimize_at(prob, &resc, d, opts, &mut make_backend);
        if sol.status == QpStatus::Optimal {
            return sol;
        }
    }
    // Nothing converged: report the order the caller actually asked for, with
    // its own verdict, exactly as it would have been without the fallback.
    requested_result
}

/// One [`sos_minimize`] attempt at a fixed relaxation order `d`, on an
/// already-equilibrated problem.
fn sos_minimize_at<F>(
    prob: &PolyProblem,
    resc: &Rescaling,
    d: usize,
    opts: &QpOptions,
    mut make_backend: F,
) -> SosSolution
where
    F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let order = Some(d);
    let (qp, cones, mi) = build_sos_sdp(prob, order, None);
    let sol = solve_socp_ipm(&qp, &cones, opts, &mut make_backend);
    let gamma = sol.x.first().copied().unwrap_or(f64::NEG_INFINITY);

    // Make the bound rigorous where that is possible: subtract the measured
    // miss in the SOS identity, so `lower_bound` is a true lower bound rather
    // than one that merely converged (see `certified_slack`). Only the reported
    // value moves — the moments, and hence exactness and the minimizers, are
    // read from the unmodified solve.
    let slack = if resc.boxed && sol.status == QpStatus::Optimal {
        certified_slack(&qp, &mi, &sol.x)
    } else {
        None
    };
    let certified = slack.is_some();
    let lower_bound = (gamma - slack.unwrap_or(0.0)) * resc.s_obj;
    if sol.status != QpStatus::Optimal {
        return SosSolution {
            lower_bound,
            status: sol.status,
            is_exact: false,
            num_minimizers: 0,
            minimizers: Vec::new(),
            certified: false,
            order: d,
        };
    }

    let mut rec = recover_from_moments(prob, &mi, &sol.y, gamma);

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
        let sol2 = solve_socp_ipm(&qp2, &cones2, opts, &mut make_backend);
        if sol2.status == QpStatus::Optimal {
            // Validate the re-solve's atoms against the UNPERTURBED, uncertified
            // `γ`: the trace penalty changes the moment matrix, not the value
            // being certified, and the certification slack is a reporting
            // correction that must not move the exactness test.
            let rec2 = recover_from_moments(prob, &mi2, &sol2.y, gamma);
            if rec2.is_exact {
                rec = rec2;
            }
        }
    }

    SosSolution {
        lower_bound,
        status: sol.status,
        certified,
        is_exact: rec.is_exact,
        num_minimizers: rec.num_minimizers,
        // Atoms are extracted in the normalized `u` coordinates the SDP was
        // built in; report them in the caller's `x`. A no-op when no variable
        // was boxed, since the map is then the identity.
        minimizers: rec.minimizers.iter().map(|u| resc.unmap(u)).collect(),
        order: d,
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
///
/// For a *constrained* program the flat-truncation rank test on the moment
/// matrix `M_d` alone is only sufficient for a representing measure on `ℝⁿ`; its
/// atoms need not lie in the feasible set `K`. When some constraint has
/// `dg = ⌈deg/2⌉ > 1` (degree > 2), the `rank M_d = rank M_{d−1}` window is a
/// strictly weaker condition than the `rank M_d = rank M_{d−dg}` window that
/// Curto–Fialkow/Henrion–Lasserre require to pin the atoms to `K`, so a flat
/// `M_d` can yield atoms outside `K` (M21). We therefore validate the extracted
/// atoms against `prob`'s constraints and withdraw the exactness certificate if
/// any atom is infeasible — `lower_bound` remains a valid lower bound.
fn recover_from_moments(
    prob: &PolyProblem,
    mi: &MomentInfo,
    y: &[f64],
    // `bound_scaled` is the certified bound in the EQUILIBRATED space: `prob`
    // here is the scaled problem, so the user-units bound must be divided by
    // `s_obj` before it can be compared against `prob.objective.eval`.
    bound_scaled: f64,
) -> Recovery {
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

    let mut minimizers = if is_exact && rank_full >= 1 && mi.d >= 1 {
        extract_atoms(mi, rank_full, |alpha| sign * y[mi.row_of[alpha]])
    } else {
        Vec::new()
    };

    // Atom-feasibility guard (M21). The flat-truncation test above certifies a
    // representing measure on ℝⁿ but not that its atoms lie in K; for a
    // constrained program an extracted atom may violate a constraint (this is
    // exactly the gap when some gᵢ has degree > 2). If any recovered atom is
    // infeasible the exactness certificate is unsound, so withdraw it: report
    // is_exact = false with no minimizers. The lower bound is unaffected (it is
    // a valid lower bound regardless of flatness).
    const FEAS_TOL: f64 = 1e-4;
    let atoms_feasible = (mi.d >= 1)
        && !minimizers.is_empty()
        && minimizers.iter().all(|x| prob.is_feasible(x, FEAS_TOL));
    let mut is_exact = is_exact && (minimizers.is_empty() || atoms_feasible);

    // Atom-objective guard. Feasibility is not enough: an extracted atom must
    // also *attain* the certified bound, i.e. `p(atom) ≈ γ*`. When it does not,
    // the moment matrix we read is not the optimal measure and the exactness
    // certificate is unsound.
    //
    // This is not hypothetical. The low-rank (min-trace) re-solve above exists
    // to collapse rank inflated by spurious pseudo-moments, but it selects the
    // *lowest-rank* near-optimal moment matrix — and when the true optimal
    // measure has several atoms, a rank-1 matrix concentrated on a single
    // non-minimizing point can be near-optimal too. The relaxation then reports
    // `is_exact = true` with `num_minimizers = 1` and hands back the measure's
    // mean rather than a minimizer. Himmelblau's function (four global
    // minimizers) is the reference case: it returned one "minimizer" whose
    // objective drifted from 4e-2 to 3e+1 as the order rose, while the bound
    // stayed correct at ~0.
    //
    // The bound is a valid lower bound regardless, so withdrawing exactness
    // costs nothing that was sound to begin with.
    if is_exact && !minimizers.is_empty() {
        let attains = minimizers.iter().all(|x| {
            let mag = prob.objective.eval_magnitude(x).max(1.0);
            (prob.objective.eval(x) - bound_scaled).abs() <= ATOM_OBJ_TOL * mag
        });
        if !attains {
            is_exact = false;
        }
    }

    if !is_exact {
        minimizers.clear();
    }
    let num_minimizers = if is_exact { rank_full } else { 0 };

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

/// Numerical rank of a symmetric PSD matrix (row-major `n×n`) for flat
/// truncation, by the **largest spectral gap**.
///
/// A fixed relative threshold is fragile here: a flat moment matrix has a few
/// `O(1)` eigenvalues (one per atom) and a noise floor set by the solver's
/// dual accuracy, but where that floor lands varies with the driver — the
/// homogeneous self-dual embedding leaves an `O(1e-5)` residual while the
/// symmetric driver reaches `O(1e-7)`, straddling any single cutoff. What is
/// invariant is the *gap*: there are many orders of magnitude between the
/// smallest true eigenvalue and the largest noise eigenvalue. So we sort the
/// eigenvalues descending and cut at the largest consecutive ratio, searching
/// only within the plausible band `(1e-9, 1e-2)·λ_max` — above the band an
/// eigenvalue is certainly real, below it is certainly numerical zero. With no
/// gap in the band the matrix is effectively full rank over that band.
/// Tolerance for the atom-objective guard, relative to the objective's
/// magnitude at the atom (`eval_magnitude`, the cancellation scale — a bare
/// absolute test is meaningless for a polynomial whose terms cancel to ~0).
///
/// Chosen from measurement, not taste. Across every extraction-exercising test
/// the relative residual of a genuine minimizer is `4e-10 … 2e-8`, with one
/// outlier at `2.5e-5` (`goldstein_price_wide_coefficient_range`, whose
/// coefficients span 144..23616 — gh #124 conditioning, still a correct
/// recovery). Non-minimizing atoms measured `7.6e-3` and `7.3e-2`. `1e-3` sits
/// ~40x above the worst legitimate case and ~7x below the worst bogus one.
const ATOM_OBJ_TOL: f64 = 1e-3;

fn psd_rank(mat: &[f64], n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut vals = vec![0.0; n];
    let mut vecs = vec![0.0; n * n];
    if !symmetric_eigen(mat, n, &mut vals, &mut vecs) {
        return n;
    }
    // Eigenvalues descending, floored at 0 (PSD; tiny negatives are noise),
    // normalized by λ_max so the bands below are absolute.
    let mut d: Vec<f64> = vals.iter().rev().map(|&v| v.max(0.0)).collect();
    let max = d[0];
    if max <= 1e-12 {
        return 0;
    }
    for v in &mut d {
        *v /= max;
    }
    const HI: f64 = 1e-2; // ≥ HI ⇒ certainly a real eigenvalue
    const LO: f64 = 1e-9; // ≤ LO ⇒ certainly numerical zero
    const MIN_GAP: f64 = 1e2; // a real rank cut spans ≥ this ratio
    let r_certain = d.iter().filter(|&&v| v >= HI).count();
    let r_possible = d.iter().filter(|&&v| v > LO).count();
    if r_certain == r_possible {
        return r_certain; // nothing in the ambiguous band
    }
    // Cut at the largest consecutive ratio gap within the ambiguous band; if no
    // gap clears MIN_GAP, keep every eigenvalue above the numerical-zero floor.
    let mut rank = r_possible;
    let mut best = MIN_GAP;
    for i in r_certain.max(1)..r_possible {
        let ratio = d[i - 1] / d[i].max(1e-300);
        if ratio > best {
            best = ratio;
            rank = i;
        }
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    #[test]
    fn himmelblau_never_returns_a_non_minimizing_atom() {
        // Himmelblau's function: f = (x²+y−11)² + (x+y²−7)², four global
        // minimizers with f* = 0. The moment relaxation is not flat at the
        // analytic center here, so the low-rank re-solve collapses it to rank 1
        // — and the resulting single "minimizer" drifted further from a true one
        // as the order ROSE: f(atom) = 4e-2, 4.2, then 30. `is_exact` was true
        // throughout, so a caller reading `minimizers[0]` got a non-minimizer
        // with no signal. The bound stayed correct (~0) the whole time.
        //
        // The guarantee asserted here is the one that matters: whatever comes
        // back is validated against the certified bound, so a reported
        // minimizer really is one. Under-reporting the COUNT is a known
        // remaining limitation (see `SosSolution::num_minimizers`).
        let p = Polynomial::new(
            2,
            vec![
                (vec![4, 0], 1.0),
                (vec![0, 4], 1.0),
                (vec![2, 1], 2.0),
                (vec![1, 2], 2.0),
                (vec![2, 0], -21.0),
                (vec![0, 2], -13.0),
                (vec![1, 0], -14.0),
                (vec![0, 1], -22.0),
                (vec![0, 0], 170.0),
            ],
        );
        let prob = PolyProblem::new(p);
        let himmelblau = |x: f64, y: f64| (x * x + y - 11.0).powi(2) + (x + y * y - 7.0).powi(2);

        for order in [2usize, 3, 4] {
            let r = sos_minimize(&prob, Some(order), backend);
            assert_eq!(r.status, QpStatus::Optimal, "order {order}: {:?}", r.status);

            // The bound stays a valid lower bound at every order — this never
            // regressed, and the fix must not cost it.
            assert!(
                r.lower_bound <= 1e-4,
                "order {order}: bound {} exceeds the true global minimum of 0",
                r.lower_bound
            );

            // Every returned minimizer must be near a TRUE one. Distance is the
            // meaningful test rather than f alone: Himmelblau is steep near its
            // minima, so a legitimately approximate atom at a low relaxation
            // order still carries a visible objective value (order 2 lands
            // 0.033 away, f = 4e-2). The failure being guarded against is not
            // imprecision but a point that is nowhere near any minimizer —
            // orders 3 and 4 previously returned atoms 1.0 and 2.0 away.
            const TRUE_MINIMIZERS: [(f64, f64); 4] = [
                (3.0, 2.0),
                (-2.805118, 3.131312),
                (-3.779310, -3.283186),
                (3.584428, -1.848126),
            ];
            for m in &r.minimizers {
                let nearest = TRUE_MINIMIZERS
                    .iter()
                    .map(|(a, b)| ((m[0] - a).powi(2) + (m[1] - b).powi(2)).sqrt())
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    nearest < 0.1,
                    "order {order}: reported minimizer ({}, {}) is {nearest:.3} from the \
                     nearest true minimizer (f = {:.3e})",
                    m[0],
                    m[1],
                    himmelblau(m[0], m[1])
                );
            }
            // And exactness must not be claimed with an unvalidated set.
            if r.is_exact {
                assert_eq!(
                    r.num_minimizers,
                    r.minimizers.len(),
                    "order {order}: num_minimizers disagrees with the returned atoms"
                );
            }
        }
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
    fn goldstein_price_wide_coefficient_range() {
        // gh #124. The degree-8 Goldstein-Price benchmark has coefficients
        // spanning 144..23616. On the *raw* polynomial the moment SDP is so
        // ill-conditioned it returns no usable bound (numerical_failure /
        // iteration_limit, NaN). With internal coefficient equilibration it
        // solves and certifies the exact global minimum f* = 3.0 at (0, −1).
        let f = Polynomial::new(
            2,
            vec![
                (vec![0, 0], 600.0),
                (vec![0, 1], 720.0),
                (vec![1, 0], 720.0),
                (vec![0, 2], 3060.0),
                (vec![1, 1], -4680.0),
                (vec![2, 0], 1260.0),
                (vec![0, 3], 12288.0),
                (vec![1, 2], -19296.0),
                (vec![2, 1], 7344.0),
                (vec![3, 0], -1072.0),
                (vec![0, 4], 14346.0),
                (vec![1, 3], -23616.0),
                (vec![2, 2], 7776.0),
                (vec![3, 1], 5784.0),
                (vec![4, 0], -2454.0),
                (vec![0, 5], 1944.0),
                (vec![1, 4], -11880.0),
                (vec![2, 3], 5040.0),
                (vec![3, 2], 9840.0),
                (vec![4, 1], -7680.0),
                (vec![5, 0], 1344.0),
                (vec![0, 6], -4428.0),
                (vec![1, 5], -1188.0),
                (vec![2, 4], 8730.0),
                (vec![3, 3], 1240.0),
                (vec![4, 2], -5370.0),
                (vec![5, 1], -168.0),
                (vec![6, 0], 952.0),
                (vec![0, 7], -648.0),
                (vec![1, 6], 1944.0),
                (vec![2, 5], 3672.0),
                (vec![3, 4], -3480.0),
                (vec![4, 3], -4080.0),
                (vec![5, 2], 2592.0),
                (vec![6, 1], 1344.0),
                (vec![7, 0], -768.0),
                (vec![0, 8], 729.0),
                (vec![1, 7], 972.0),
                (vec![2, 6], -1458.0),
                (vec![3, 5], -1836.0),
                (vec![4, 4], 1305.0),
                (vec![5, 3], 1224.0),
                (vec![6, 2], -648.0),
                (vec![7, 1], -288.0),
                (vec![8, 0], 144.0),
            ],
        );
        let r = sos_minimize(&PolyProblem::new(f), Some(0), backend);
        // The #124 contract: a *usable finite bound* instead of NaN. The bound is
        // the stable, reproducible quantity — across runs it lands within ~6e-4 of
        // the true minimum 3.0 (the SDP's relative tolerance, amplified by the
        // scale-back factor max|coef| ≈ 2.4e4). Exactness / minimizer extraction
        // reads the moment matrix's near-null space, which on this
        // conditioning-limited degree-8 problem is sensitive to floating-point
        // nondeterminism (the flat-truncation rank test occasionally flips), so it
        // is *not* asserted here — it usually succeeds, but the bound is the
        // guarantee.
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(r.lower_bound.is_finite(), "bound = {}", r.lower_bound);
        assert!(
            (r.lower_bound - 3.0).abs() < 5e-3,
            "bound = {}",
            r.lower_bound
        );
    }

    #[test]
    fn equilibration_preserves_a_well_scaled_bound() {
        // The coefficient equilibration (gh #124) must be a no-op on the *value*:
        // an already O(1)-scaled polynomial returns the same bound it did before.
        // p(x) = x² − 4x + 5 has min 1 at x = 2; its max|coef| is 5, so the
        // internal scale-and-unscale round-trip must still report 1.0.
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

    /// Lasserre, *SIAM J. Optim.* 11(3):796–817 (2001), Example 5 — the gh #218
    /// constrained quartic. `min −x₁−x₂` over two quartics and the box
    /// `[0,3]×[0,4]`; true global minimum −5.50801.
    fn lasserre_ex5() -> PolyProblem {
        let obj = Polynomial::new(2, vec![(vec![1, 0], -1.0), (vec![0, 1], -1.0)]);
        let g1 = Polynomial::new(
            2,
            vec![
                (vec![4, 0], 2.0),
                (vec![3, 0], -8.0),
                (vec![2, 0], 8.0),
                (vec![0, 0], 2.0),
                (vec![0, 1], -1.0),
            ],
        );
        let g2 = Polynomial::new(
            2,
            vec![
                (vec![4, 0], 4.0),
                (vec![3, 0], -32.0),
                (vec![2, 0], 88.0),
                (vec![1, 0], -96.0),
                (vec![0, 0], 36.0),
                (vec![0, 1], -1.0),
            ],
        );
        PolyProblem::new(obj)
            .ge(g1)
            .ge(g2)
            .ge(Polynomial::new(2, vec![(vec![1, 0], 1.0)]))
            .ge(Polynomial::new(
                2,
                vec![(vec![0, 0], 3.0), (vec![1, 0], -1.0)],
            ))
            .ge(Polynomial::new(2, vec![(vec![0, 1], 1.0)]))
            .ge(Polynomial::new(
                2,
                vec![(vec![0, 0], 4.0), (vec![0, 1], -1.0)],
            ))
    }

    #[test]
    fn affine_substitute_preserves_values() {
        // Substitution is only sound if the rewritten polynomial is the *same
        // function* under the change of variables: q(u) = p(shift + scale·u).
        // Check that pointwise on a polynomial with cross terms and a variable
        // appearing at several degrees.
        let p = Polynomial::new(
            2,
            vec![
                (vec![3, 0], 2.0),
                (vec![2, 1], -1.5),
                (vec![1, 1], 4.0),
                (vec![0, 2], 0.5),
                (vec![1, 0], -3.0),
                (vec![0, 0], 7.0),
            ],
        );
        let shift = [1.5, -2.0];
        let scale = [0.5, 3.0];
        let q = p.affine_substitute(&shift, &scale);
        for &(u0, u1) in &[
            (0.0, 0.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (0.37, 0.62),
            (-0.8, -0.25),
        ] {
            let x = [shift[0] + scale[0] * u0, shift[1] + scale[1] * u1];
            assert!(
                (q.eval(&[u0, u1]) - p.eval(&x)).abs() < 1e-9,
                "q({u0},{u1}) = {} but p({},{}) = {}",
                q.eval(&[u0, u1]),
                x[0],
                x[1],
                p.eval(&x)
            );
        }
    }

    #[test]
    fn domain_normalization_reports_minimizers_in_original_coordinates() {
        // gh #218. Domain normalization solves the SDP in `u ∈ [−1,1]²`, so a
        // recovered atom must be mapped back or the caller silently receives a
        // point in the wrong coordinate system.
        //
        // The boxes here are deliberately wide, off-center, and *different per
        // variable* (x: [0,10] ⇒ shift 5 scale 5; y: [−5,1] ⇒ shift −2 scale 3),
        // so a transposed or shared map lands visibly wrong rather than
        // coincidentally right.
        //
        // min (x−8)² + (y+3)² s.t. x ∈ [0,10], y ∈ [−5,1] ⇒ min 0 at (8, −3),
        // interior to the box so the box does not distort the optimum.
        let obj = Polynomial::new(
            2,
            vec![
                (vec![2, 0], 1.0),
                (vec![1, 0], -16.0),
                (vec![0, 2], 1.0),
                (vec![0, 1], 6.0),
                (vec![0, 0], 73.0),
            ],
        );
        let prob = PolyProblem::new(obj)
            .ge(Polynomial::new(2, vec![(vec![1, 0], 1.0)]))
            .ge(Polynomial::new(
                2,
                vec![(vec![0, 0], 10.0), (vec![1, 0], -1.0)],
            ))
            .ge(Polynomial::new(
                2,
                vec![(vec![0, 1], 1.0), (vec![0, 0], 5.0)],
            ))
            .ge(Polynomial::new(
                2,
                vec![(vec![0, 0], 1.0), (vec![0, 1], -1.0)],
            ));

        let r = sos_minimize(&prob, None, backend);
        assert_eq!(r.status, QpStatus::Optimal, "{:?}", r.status);
        assert!(r.lower_bound.abs() < 1e-4, "bound = {}", r.lower_bound);
        assert!(r.is_exact, "relaxation should be exact here");
        assert_eq!(r.num_minimizers, 1);
        let m = &r.minimizers[0];
        assert!(
            (m[0] - 8.0).abs() < 1e-3 && (m[1] + 3.0).abs() < 1e-3,
            "minimizer {m:?} is not (8, -3) — coordinates left un-mapped?"
        );
    }

    #[test]
    fn lasserre_ex5_hierarchy_converges_to_the_global_minimum() {
        // gh #218's acceptance criterion, verbatim: "a finite bound ≤ −5.50801,
        // tightening toward it as the order rises."
        //
        // Every order must converge (no `nan`, no iteration limit), every bound
        // must be a valid lower bound, the sequence must be monotone, and the
        // hierarchy must actually reach the optimum rather than plateau.
        const TRUE_MIN: f64 = -5.508013;
        let prob = lasserre_ex5();
        let mut prev = f64::NEG_INFINITY;
        let mut bounds = Vec::new();
        for order in [2usize, 3, 4] {
            let r = sos_constrained_lower_bound(&prob, Some(order), backend);
            assert_eq!(r.status, QpStatus::Optimal, "order {order}: {:?}", r.status);
            // Soundness first: a lower bound may never exceed the true minimum.
            assert!(
                r.lower_bound <= TRUE_MIN + 1e-5,
                "order {order}: bound {} exceeds the true minimum {TRUE_MIN}",
                r.lower_bound
            );
            assert!(
                r.lower_bound >= prev - 1e-6,
                "order {order}: bound {} is looser than order {}'s {prev}",
                r.lower_bound,
                order - 1
            );
            prev = r.lower_bound;
            bounds.push(r.lower_bound);
        }
        // Order 4 is where the hierarchy becomes exact for this problem; before
        // the degenerate-face fix it stalled with no usable bound at all.
        assert!(
            (bounds[2] - TRUE_MIN).abs() < 1e-4,
            "order 4 bound {} should reach the global minimum {TRUE_MIN}; got the sequence {bounds:?}",
            bounds[2]
        );
        // And the hierarchy must genuinely tighten, not sit at the trivial box
        // bound: order 2 is −7 (the quartics barely participate there).
        assert!(
            bounds[0] < bounds[1] - 1e-3 && bounds[1] < bounds[2] - 1e-3,
            "hierarchy did not tighten: {bounds:?}"
        );
    }

    #[test]
    fn certified_bound_is_a_true_lower_bound_not_merely_an_accurate_one() {
        // The soundness fix. A converged SDP reports a `γ` that is accurate but
        // not necessarily *below* the minimum: on this problem at order 4 the
        // raw value came back 2.2e-7 ABOVE the true minimum, which makes it not
        // a lower bound at all — the one genuinely unsound failure mode for
        // this API, and one no amount of solver tolerance removes.
        //
        // The certified value subtracts the SOS identity's measured miss, so it
        // is valid however the solve went. `TRUE_MIN` here is not the issue's
        // quoted figure but an independently computed one (400-start SLSQP),
        // and the assertion is strict — no tolerance slack — because the whole
        // point is that the bound must genuinely be below it.
        const TRUE_MIN: f64 = -5.508013271595;
        let prob = lasserre_ex5();
        for order in [2usize, 3, 4, 5] {
            let r = sos_minimize(&prob, Some(order), backend);
            assert_eq!(r.status, QpStatus::Optimal, "order {order}");
            assert!(
                r.certified,
                "order {order}: a fully boxed problem must certify"
            );
            assert!(
                r.lower_bound <= TRUE_MIN,
                "order {order}: certified bound {} is ABOVE the true minimum {TRUE_MIN} \
                 — not a lower bound",
                r.lower_bound
            );
            // Validity must not come from being uselessly loose: order 4 still
            // has to land within 1e-4 of the optimum.
            if order >= 4 {
                assert!(
                    r.lower_bound >= TRUE_MIN - 1e-4,
                    "order {order}: bound {} is valid but far too loose",
                    r.lower_bound
                );
            }
        }
    }

    #[test]
    fn certification_is_withheld_when_the_domain_is_unbounded() {
        // Certification rests on `|u^α| ≤ 1` over the feasible set, so it needs
        // a box. Without one the residual polynomial cannot be bounded below at
        // all, and claiming a certificate would be a lie — the flag must say so
        // rather than the bound silently pretending.
        let p = Polynomial::new(1, vec![(vec![2], 1.0), (vec![1], -4.0), (vec![0], 5.0)]);
        let r = sos_minimize(&PolyProblem::new(p), None, backend);
        assert_eq!(r.status, QpStatus::Optimal);
        assert!(!r.certified, "an unconstrained problem cannot be certified");
    }

    #[test]
    fn square_box_idiom_is_recognized_for_certification() {
        // `1 − x² ≥ 0` is the idiomatic way to write a box in an SOS model, and
        // bounds x just as surely as the linear pair does. Missing it would
        // leave obviously-bounded textbook problems uncertifiable.
        // min −x s.t. 1 − x² ≥ 0  ⇒  min = −1 at x = 1.
        let prob = PolyProblem::new(Polynomial::new(1, vec![(vec![1], -1.0)]))
            .ge(Polynomial::new(1, vec![(vec![0], 1.0), (vec![2], -1.0)]));
        let r = sos_minimize(&prob, None, backend);
        assert_eq!(r.status, QpStatus::Optimal);
        assert!(r.certified, "the `c − a·x² ≥ 0` idiom should be recognized");
        assert!(
            r.lower_bound <= -1.0,
            "certified bound {} is above the true minimum -1",
            r.lower_bound
        );
        assert!(
            r.lower_bound >= -1.0 - 1e-6,
            "bound {} too loose",
            r.lower_bound
        );
    }

    #[test]
    fn loosening_tol_costs_tightness_but_never_validity() {
        // The knob gh #218 asked for. Loosening `tol` is the escape hatch for a
        // relaxation that will not converge — and because certification measures
        // the actual miss rather than trusting the solve, a sloppier solve buys
        // a weaker bound, never an invalid one.
        const TRUE_MIN: f64 = -5.508013271595;
        let prob = lasserre_ex5();
        let mut prev_gap = 0.0;
        for tol in [1e-10, 1e-8, 1e-6] {
            let opts = QpOptions { tol, ..sos_opts() };
            let r = sos_minimize_opts(&prob, Some(4), &opts, backend);
            assert_eq!(r.status, QpStatus::Optimal, "tol {tol:e}");
            assert!(r.certified, "tol {tol:e}");
            assert!(
                r.lower_bound <= TRUE_MIN,
                "tol {tol:e}: bound {} is above the true minimum",
                r.lower_bound
            );
            // Looser tolerance ⇒ larger measured residual ⇒ looser bound.
            let gap = TRUE_MIN - r.lower_bound;
            assert!(
                gap >= prev_gap,
                "tol {tol:e}: gap {gap:e} should not be tighter than the stricter tolerance's {prev_gap:e}"
            );
            prev_gap = gap;
        }
    }

    #[test]
    fn a_failed_order_falls_back_to_a_coarser_proved_bound() {
        // gh #218 suggestion 2. `sos_minimize` must not answer "no bound" at a
        // high order when it already proved one at a lower order — a coarser
        // bound is still valid, just weaker.
        //
        // Driven through the public entry point on a problem that converges at
        // every order, so what is asserted is the contract that survives: the
        // reported `order` identifies which relaxation produced the bound, and
        // that bound is always valid.
        let prob = lasserre_ex5();
        for requested in [2usize, 3, 4] {
            let r = sos_minimize(&prob, Some(requested), backend);
            assert_eq!(r.status, QpStatus::Optimal, "order {requested}");
            assert!(
                r.order <= requested,
                "reported order {} exceeds the requested {requested}",
                r.order
            );
            assert!(
                r.lower_bound <= -5.508013 + 1e-5,
                "order {requested} (solved at {}): bound {} exceeds the true minimum",
                r.order,
                r.lower_bound
            );
        }
    }

    #[test]
    fn requested_order_below_the_minimum_admissible_is_raised_not_rejected() {
        // The fallback walks down to the minimum admissible order, so that
        // floor has to be right: the quartic constraints force d >= 2, and
        // asking for 1 must be silently raised rather than looping or building
        // a degree-deficient SDP.
        let prob = lasserre_ex5();
        assert_eq!(min_relaxation_order(&prob), 2);
        let r = sos_minimize(&prob, Some(1), backend);
        assert_eq!(r.status, QpStatus::Optimal);
        assert_eq!(
            r.order, 2,
            "order should be raised to the admissible minimum"
        );
    }

    #[test]
    fn degenerate_moment_sdp_converges_in_a_sane_iteration_count() {
        // gh #218. This relaxation's moment SDP sits on a degenerate face. The
        // affine direction points almost straight out of the PSD cone there, so
        // Mehrotra's σ came back near zero — no centering, exactly where
        // centering was needed — and the step collapsed geometrically until the
        // iterate froze, burning the entire `max_iter` budget to change no digit
        // of the objective.
        //
        // With the centering fallback it converges outright, so assert the
        // strong property rather than the weak one: a real solve in a sane
        // iteration count, nowhere near the budget it used to exhaust.
        let (prob, _resc) = lasserre_ex5().equilibrated();
        let (qp, cones, _mi) = build_sos_sdp(&prob, Some(3), None);
        let opts = QpOptions {
            max_iter: 5000,
            ..sos_opts()
        };
        let sol = solve_socp_ipm(&qp, &cones, &opts, backend);
        assert_eq!(sol.status, QpStatus::Optimal, "{:?}", sol.status);
        assert!(
            sol.iters < 100,
            "took {} of {} iterations",
            sol.iters,
            opts.max_iter
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

    #[test]
    fn facial_reduction_three_minimizers_degree_six() {
        // p(x) = x²(x−1)²(x+1)² = x⁶ − 2x⁴ + x², a nonnegative sextic with
        // THREE global minima (value 0) at x = −1, 0, 1. The order-3 relaxation
        // is degenerate (a boundary-rank optimum); the HSDE driver solves it and
        // facial reduction recovers all three atoms.
        let p = Polynomial::new(1, vec![(vec![6], 1.0), (vec![4], -2.0), (vec![2], 1.0)]);
        let s = sos_minimize(&PolyProblem::new(p), None, backend);
        assert_eq!(s.status, QpStatus::Optimal, "{:?}", s.status);
        assert!(s.lower_bound.abs() < 1e-5, "bound = {}", s.lower_bound);
        assert!(s.is_exact, "facial reduction should certify exactness");
        assert_eq!(s.num_minimizers, 3, "three atoms at −1, 0, 1");
        let mut roots: Vec<f64> = s.minimizers.iter().map(|m| m[0]).collect();
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((roots[0] + 1.0).abs() < 1e-2, "{roots:?}");
        assert!(roots[1].abs() < 1e-2, "{roots:?}");
        assert!((roots[2] - 1.0).abs() < 1e-2, "{roots:?}");
    }

    #[test]
    fn facial_reduction_four_minimizers_2d_order_three() {
        // p(x,y) = (x²−1)² + (y²−1)², four global minima (value 0) at (±1, ±1).
        // Four atoms need moment-matrix rank 4, which cannot stabilize against
        // the 3-dimensional degree-≤1 subspace until order 3 — a larger, more
        // degenerate SDP that only the HSDE driver carries to optimality.
        let p = Polynomial::new(
            2,
            vec![
                (vec![4, 0], 1.0),
                (vec![2, 0], -2.0),
                (vec![0, 4], 1.0),
                (vec![0, 2], -2.0),
                (vec![0, 0], 2.0),
            ],
        );
        let s = sos_minimize(&PolyProblem::new(p), Some(3), backend);
        assert_eq!(s.status, QpStatus::Optimal, "{:?}", s.status);
        assert!(s.lower_bound.abs() < 1e-5, "bound = {}", s.lower_bound);
        assert!(s.is_exact, "facial reduction should certify exactness");
        assert_eq!(s.num_minimizers, 4, "four atoms at (±1, ±1)");
        for atom in &s.minimizers {
            assert!((atom[0].abs() - 1.0).abs() < 2e-2, "x = {}", atom[0]);
            assert!((atom[1].abs() - 1.0).abs() < 2e-2, "y = {}", atom[1]);
        }
        // All four quadrants present.
        let mut quad = [false; 4];
        for atom in &s.minimizers {
            quad[usize::from(atom[0] > 0.0) + 2 * usize::from(atom[1] > 0.0)] = true;
        }
        assert!(
            quad.iter().all(|&q| q),
            "missing a quadrant: {:?}",
            s.minimizers
        );
    }

    #[test]
    fn sdp_row_order_is_deterministic() {
        // M22: the coefficient-matching rows were emitted in `HashMap`
        // iteration order, so the SDP's row ordering — and hence the solver's
        // floating-point path and results — varied run-to-run. Rust seeds each
        // `HashMap` differently, so building the *same* problem twice in one
        // process exposes it: with a `HashMap` the two builds disagree on row
        // order; with the `BTreeMap` they are identical. Assert determinism via
        // both the RHS vector order and the monomial→row map.
        let p = Polynomial::new(
            2,
            vec![
                (vec![4, 0], 1.0),
                (vec![0, 4], 1.0),
                (vec![2, 2], -1.0),
                (vec![1, 0], -2.0),
                (vec![0, 1], 3.0),
                (vec![0, 0], 5.0),
            ],
        );
        let prob = PolyProblem::new(p);
        let (qp1, _, mi1) = build_sos_sdp(&prob, None, None);
        let (qp2, _, mi2) = build_sos_sdp(&prob, None, None);
        assert!(qp1.b.len() > 1, "need several rows to detect a permutation");
        assert_eq!(qp1.b, qp2.b, "RHS row order differs between builds");
        assert_eq!(
            mi1.row_of, mi2.row_of,
            "monomial→row assignment differs between builds"
        );
    }

    #[test]
    fn constrained_overclaim_rejected_when_atom_infeasible() {
        // min (x+1)²  s.t.  x³ ≥ 0  (feasible set x ≥ 0).  The constrained
        // minimum is 1 at x = 0; the *unconstrained* minimum is 0 at x = −1.
        // At order 2 the localizing constraint is the single scalar L(x³) ≥ 0,
        // far too weak to pin the relaxation's atom to the feasible set: the
        // degree-3 constraint has dg = ⌈3/2⌉ = 2, so the d−1 flat-truncation
        // window is *weaker* than the d−dg window Curto–Fialkow/Henrion–Lasserre
        // require to certify atoms in K. Flat truncation on M_d alone fires and
        // would extract a single atom at x ≈ −0.72 — INFEASIBLE (x³ ≈ −0.37 < 0)
        // — while reporting the unconstrained bound 0 as the exact constrained
        // optimum (M21). The atom-feasibility guard must reject this: is_exact
        // is false, no minimizers, and lower_bound stays a valid lower bound
        // (0 ≤ 1).
        let prob = PolyProblem::new(Polynomial::new(
            1,
            vec![(vec![2], 1.0), (vec![1], 2.0), (vec![0], 1.0)],
        ))
        .ge(Polynomial::new(1, vec![(vec![3], 1.0)]));
        let s = sos_minimize(&prob, Some(2), backend);
        assert_eq!(s.status, QpStatus::Optimal, "{:?}", s.status);
        assert!(
            !s.is_exact,
            "flat M_d truncation extracted an infeasible atom; the certificate \
             must be rejected, got is_exact=true with minimizers {:?}",
            s.minimizers
        );
        assert_eq!(s.num_minimizers, 0);
        assert!(s.minimizers.is_empty());
        // The reported bound is still a valid lower bound on the constrained min.
        assert!(
            s.lower_bound <= 1.0 + 1e-6,
            "bound {} must not exceed the true constrained min 1",
            s.lower_bound
        );
    }
}
