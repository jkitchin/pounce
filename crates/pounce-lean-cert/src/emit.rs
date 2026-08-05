//! The certificate driver: a neutral f64 QP solve → an exact-rational
//! `pounce.lean-cert/v1` [`Certificate`], or a typed refusal.
//!
//! This layer is deliberately free of `pounce-nl`/`pounce-cli` types so the
//! published crate stays light and the whole pipeline is unit-testable from
//! plain numbers. The CLI's `certify` subcommand reads the `.nl`/`.sol`,
//! classifies the QP, and hands the extracted data here as a [`QpInput`].
//!
//! Pipeline: validate the supported slice → lossless `f64 → ℚ` → normalize
//! constraints to `A x ≥ b` → detect the active set from the float point →
//! [`refine_kkt`] for the exact optimizer + duals → [`ldlt`] for the PSD witness
//! → assemble → **exact self-check gate**. Any off-slice input or failed exact
//! check returns [`EmitError`] instead of an unsound certificate.

use crate::ldlt::{LdlError, ldlt};
use crate::linalg::dot;
use crate::rational::{Bound, Rat, RatError};
use crate::refine::{RefineError, refine_kkt_eq};
use crate::schema::{
    Binding, Candidate, Certificate, Constraint, Entry, Farkas, FeasibleWitness, HessianPsd,
    Objective, Problem, Recession, SCHEMA_TAG, SparseMatrix, Toolchain, VALIDATED_LEAN,
    VALIDATED_MATHLIB, VarBounds, Witnesses,
};
use num_rational::BigRational;
use num_traits::Zero;

/// A single linear constraint `lower ≤ coeffs·x ≤ upper` (bounds may be `±inf`).
#[derive(Clone, Debug)]
pub struct LinearConstraint {
    pub name: String,
    pub coeffs: Vec<f64>,
    pub lower: f64,
    pub upper: f64,
}

/// Neutral input: a convex-QP solve in `f64`, exactly as POUNCE produces it.
#[derive(Clone, Debug)]
pub struct QpInput {
    pub n: usize,
    /// Objective Hessian, lower triangle (`row ≥ col`) as `(i, j, value)`.
    pub q_lower: Vec<(usize, usize, f64)>,
    /// `true` ⇒ `f = ½xᵀQx + cᵀx + k` (POUNCE's convention).
    pub half_quadratic: bool,
    pub c: Vec<f64>,
    pub constant: f64,
    pub constraints: Vec<LinearConstraint>,
    pub var_lower: Vec<f64>,
    pub var_upper: Vec<f64>,
    /// The solver's float primal `x̃`, used only to *guess* the active set
    /// (the exact point is recomputed; a wrong guess is caught, not trusted).
    pub x_float: Vec<f64>,
    /// A normalized row is treated as active when `|A_i x̃ − b_i| ≤ active_tol`.
    pub active_tol: f64,
}

/// Provenance for the certificate's `binding` block.
#[derive(Clone, Debug)]
pub struct CertMeta {
    pub nl_sha256: String,
    pub sol_sha256: String,
    pub solver: String,
}

/// Why a certificate could not be emitted. Every variant means "refuse",
/// never "emit something weaker".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmitError {
    /// A vector length did not match `n_vars`.
    DimensionMismatch,
    /// A value that must be finite was `±inf`/`NaN`.
    NonFinite,
    /// Constraint `constraint` has no finite bound at all; degenerate.
    FreeConstraint { constraint: usize },
    /// A `q_lower` entry sat above the diagonal (`j > i`).
    QNotLowerTriangle { i: usize, j: usize },
    /// `Q` was not PSD / not expressible as unit-lower `LDLᵀ`.
    Ldl(LdlError),
    /// The exact active-set KKT refinement failed.
    Refine(RefineError),
    /// The exact Farkas refinement failed (infeasible verdict).
    Farkas(crate::refine_farkas::FarkasError),
    /// The exact recession refinement failed (unbounded verdict).
    Recession(crate::refine_recession::RecessionError),
    /// Defensive: an assembled cert failed its own exact recheck.
    SelfCheck(&'static str),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for EmitError {}

impl From<RatError> for EmitError {
    fn from(_: RatError) -> Self {
        EmitError::NonFinite
    }
}
impl From<LdlError> for EmitError {
    fn from(e: LdlError) -> Self {
        EmitError::Ldl(e)
    }
}
impl From<RefineError> for EmitError {
    fn from(e: RefineError) -> Self {
        EmitError::Refine(e)
    }
}

/// f64 → exact `BigRational` (errors on non-finite).
fn br(x: f64) -> Result<BigRational, EmitError> {
    Ok(Rat::from_f64(x)?.0)
}

/// Densify a symmetric lower-triangle triplet list into an `n×n` matrix,
/// optionally doubling every entry (to form `2Q` when `!half_quadratic`).
#[allow(clippy::needless_range_loop)]
fn densify_sym(
    n: usize,
    lower: &[(usize, usize, BigRational)],
    double: bool,
) -> Vec<Vec<BigRational>> {
    let mut m = vec![vec![BigRational::zero(); n]; n];
    for (i, j, v) in lower {
        let val = if double { v + v } else { v.clone() };
        m[*i][*j] = val.clone();
        if i != j {
            m[*j][*i] = val;
        }
    }
    m
}

/// `½·xᵀMx + cᵀx + k` over ℚ, where `m` is the dense Hessian-of-record.
#[allow(clippy::needless_range_loop)]
fn objective_value(
    m: &[Vec<BigRational>],
    c: &[BigRational],
    constant: &BigRational,
    x: &[BigRational],
) -> BigRational {
    let n = x.len();
    let mut xmx = BigRational::zero();
    for i in 0..n {
        for j in 0..n {
            xmx += &x[i] * &m[i][j] * &x[j];
        }
    }
    BigRational::new(1.into(), 2.into()) * xmx + dot(c, x) + constant
}

/// The shared **Frontend**: everything derivable from the problem alone (no
/// solution, no witnesses). [`emit_certificate`] runs this and then the witness
/// half; a consumer runs *only* this on its own `.nl` and checks the result
/// equals the certificate's `problem` block (see [`canonical_problem`]). Both
/// paths therefore produce the cert `problem` from the identical code.
struct ProblemView {
    /// The cert `problem` block (objective, var_bounds, constraints).
    problem: Problem,
    /// Expanded one-sided/equality rows (for the witness-side normalization).
    all_constraints: Vec<LinearConstraint>,
    /// Dense cert `Q` over ℚ (for the `LDLᵀ` witness).
    q_dense: Vec<Vec<BigRational>>,
    /// Dense Hessian-of-record `M` over ℚ (for the KKT solve).
    m_dense: Vec<Vec<BigRational>>,
    /// Linear objective term and constant over ℚ.
    c_rat: Vec<BigRational>,
    constant_rat: BigRational,
}

/// Re-derive the certificate `problem` block from the problem fields of `input`
/// (the `x_float`/`active_tol` solution hints are ignored). Pure and
/// deterministic — this is the consumer-side root of trust.
pub fn problem_block(input: &QpInput) -> Result<Problem, EmitError> {
    Ok(build_problem_view(input)?.problem)
}

/// A canonical JSON view of a `problem` block, for deciding whether two
/// certificates describe the **same optimization problem**: `Q` entries sorted
/// by `(i, j)`, constraints sorted by `(coeffs, lower, upper)` with their
/// (advisory) names blanked, and every rational already reduced by [`Rat`].
/// Two problems are the same iff their canonical views are equal — order- and
/// name-insensitive, value-exact over ℚ.
pub fn canonical_problem(p: &Problem) -> serde_json::Value {
    let mut p = p.clone();
    if let Some(obj) = p.objective.as_mut() {
        obj.q.entries.sort_by_key(|e| (e.i, e.j));
    }
    if let Some(cons) = p.constraints.as_mut() {
        for c in cons.iter_mut() {
            c.name = String::new();
        }
        cons.sort_by_cached_key(|c| serde_json::to_string(c).unwrap_or_default());
    }
    // `sos-poly`: a polynomial is its term *set*, so order must not distinguish
    // two spellings of the same objective. Exponent vectors are unique per term
    // (the extractor merges like monomials), so they alone are a total key.
    if let Some(poly) = p.polynomial.as_mut() {
        poly.terms.sort_by_key(|t| t.exponents.clone());
    }
    // Same for each `gₖ`, but *within* a constraint only: the order of the
    // constraints themselves is the certificate's `multiplier` numbering, so
    // sorting that list would renumber the witnesses.
    if let Some(cons) = p.poly_constraints.as_mut() {
        for g in cons.iter_mut() {
            g.terms.sort_by_key(|t| t.exponents.clone());
        }
    }
    serde_json::to_value(&p).unwrap_or(serde_json::Value::Null)
}

fn build_problem_view(input: &QpInput) -> Result<ProblemView, EmitError> {
    let n = input.n;
    if input.c.len() != n || input.var_lower.len() != n || input.var_upper.len() != n {
        return Err(EmitError::DimensionMismatch);
    }

    // Expand each general constraint. An equality (`l == u`) is kept whole — it
    // becomes an `E x = d` row with a free-sign multiplier `μ`, handled by the
    // `global_min_of_kkt_eq` theorem. A two-sided range (`l ≠ u`) splits into
    // `a·x ≥ l` (`{c}_lo`) and `a·x ≤ u` (`{c}_hi`); at most one can be active
    // at the optimum, so the split is non-degenerate. One-sided rows pass
    // through unchanged.
    let mut all_constraints: Vec<LinearConstraint> = Vec::with_capacity(input.constraints.len());
    for (ci, con) in input.constraints.iter().enumerate() {
        if con.coeffs.len() != n {
            return Err(EmitError::DimensionMismatch);
        }
        match (con.lower.is_finite(), con.upper.is_finite()) {
            (true, false) | (false, true) => all_constraints.push(con.clone()),
            // Equality: kept as-is (lower == upper), routed to `E x = d`.
            (true, true) if con.lower == con.upper => all_constraints.push(con.clone()),
            // Two-sided range: split into two one-sided rows.
            (true, true) => {
                all_constraints.push(LinearConstraint {
                    name: format!("{}_lo", con.name),
                    coeffs: con.coeffs.clone(),
                    lower: con.lower,
                    upper: f64::INFINITY,
                });
                all_constraints.push(LinearConstraint {
                    name: format!("{}_hi", con.name),
                    coeffs: con.coeffs.clone(),
                    lower: f64::NEG_INFINITY,
                    upper: con.upper,
                });
            }
            (false, false) => return Err(EmitError::FreeConstraint { constraint: ci }),
        }
    }

    // Finite variable bounds are linear inequalities too, so fold them into the
    // constraint system as one-sided rows: `xᵢ ≥ lᵢ` → `eᵢ·x ≥ lᵢ`, and
    // `xᵢ ≤ uᵢ` → `−eᵢ·x ≥ −uᵢ`. The reusable convex-QP KKT theorem is stated
    // over an arbitrary `A x ≥ b`, so a bound multiplier is just one more dual —
    // nothing special-cases bounds on the Lean side. The certificate therefore
    // emits `var_bounds` as all-infinite in v1 and carries the bounds (with
    // descriptive names) in `constraints`.
    for v in 0..n {
        let (lo, hi) = (input.var_lower[v], input.var_upper[v]);
        if lo.is_nan() || hi.is_nan() {
            return Err(EmitError::NonFinite);
        }
        let unit = || {
            let mut u = vec![0.0; n];
            u[v] = 1.0;
            u
        };
        if lo.is_finite() && hi.is_finite() && lo == hi {
            // Fixed variable xᵥ = lo → an equality row.
            all_constraints.push(LinearConstraint {
                name: format!("var{v}_fix"),
                coeffs: unit(),
                lower: lo,
                upper: hi,
            });
        } else {
            if lo != f64::NEG_INFINITY {
                all_constraints.push(LinearConstraint {
                    name: format!("var{v}_lb"),
                    coeffs: unit(),
                    lower: lo,
                    upper: f64::INFINITY,
                });
            }
            if hi != f64::INFINITY {
                all_constraints.push(LinearConstraint {
                    name: format!("var{v}_ub"),
                    coeffs: unit(),
                    lower: f64::NEG_INFINITY,
                    upper: hi,
                });
            }
        }
    }

    // Objective Hessian → rational lower triangle + dense Q (cert) and M (KKT).
    let mut q_lower_rat: Vec<(usize, usize, BigRational)> = Vec::with_capacity(input.q_lower.len());
    for &(i, j, v) in &input.q_lower {
        if j > i {
            return Err(EmitError::QNotLowerTriangle { i, j });
        }
        q_lower_rat.push((i, j, br(v)?));
    }
    let q_dense = densify_sym(n, &q_lower_rat, false);
    // Hessian of record: M = Q (half_quadratic) or 2Q (full quadratic).
    let m_dense = densify_sym(n, &q_lower_rat, !input.half_quadratic);

    let c_rat: Vec<BigRational> = input.c.iter().map(|&v| br(v)).collect::<Result<_, _>>()?;
    let constant_rat = br(input.constant)?;

    // Assemble the cert `problem` block (canonical: Q lower-triangle sorted by
    // (i,j), var_bounds all-infinite, constraints in expansion order).
    let mut q_entries: Vec<Entry> = q_lower_rat
        .iter()
        .map(|(i, j, v)| Entry {
            i: *i,
            j: *j,
            val: Rat(v.clone()),
        })
        .collect();
    q_entries.sort_by_key(|e| (e.i, e.j));

    let constraints: Vec<Constraint> = all_constraints
        .iter()
        .map(|con| {
            Ok(Constraint {
                name: con.name.clone(),
                coeffs: con
                    .coeffs
                    .iter()
                    .map(|&v| Rat::from_f64(v))
                    .collect::<Result<_, _>>()?,
                lower: Bound::from_f64(con.lower)?,
                upper: Bound::from_f64(con.upper)?,
            })
        })
        .collect::<Result<_, EmitError>>()?;

    // v1 folds finite variable bounds into `constraints`, so the cert's declared
    // `var_bounds` are always the infinite sentinels.
    let var_bounds = VarBounds {
        lower: vec![Bound::NegInf; n],
        upper: vec![Bound::PosInf; n],
    };

    let problem = Problem {
        n_vars: n,
        objective: Some(Objective {
            kind: "quadratic".to_string(),
            half_quadratic: input.half_quadratic,
            q: SparseMatrix::symmetric(n, n, q_entries),
            c: c_rat.iter().map(|v| Rat(v.clone())).collect(),
            constant: Rat(constant_rat.clone()),
        }),
        var_bounds: Some(var_bounds),
        constraints: Some(constraints),
        polynomial: None,
        poly_constraints: None,
        neighborhood: None,
    };

    Ok(ProblemView {
        problem,
        all_constraints,
        q_dense,
        m_dense,
        c_rat,
        constant_rat,
    })
}

/// Which system an expanded constraint row landed in, so a cert-order dual can
/// be read back out of the right multiplier vector.
enum RowRef {
    Ineq(usize),
    Eq(usize),
}

/// The expanded rows split into `A x ≥ b` (multipliers `λ ≥ 0`) and `E x = d`
/// (free-sign `μ`), plus the float slacks used to guess which inequalities are
/// active.
struct NormalizedRows {
    a_rows: Vec<Vec<BigRational>>,
    b_rhs: Vec<BigRational>,
    e_rows: Vec<Vec<BigRational>>,
    d_rhs: Vec<BigRational>,
    /// `A_i x̃ − b_i` in f64, aligned with `a_rows`. Float, and only ever used
    /// to *propose* an active set.
    nslack: Vec<f64>,
    /// One entry per expanded constraint, in cert order.
    row_ref: Vec<RowRef>,
}

/// Normalize expanded constraints to the two systems the Lean side speaks.
///
/// A `≤` row is negated rather than given a sign convention of its own, so
/// every inequality multiplier is `≥ 0` and one KKT theorem covers all of them.
fn normalize_rows(
    n: usize,
    all_constraints: &[LinearConstraint],
    x_float: &[f64],
) -> Result<NormalizedRows, EmitError> {
    let mut out = NormalizedRows {
        a_rows: Vec::new(),
        b_rhs: Vec::new(),
        e_rows: Vec::new(),
        d_rhs: Vec::new(),
        nslack: Vec::new(),
        row_ref: Vec::with_capacity(all_constraints.len()),
    };
    for con in all_constraints {
        if con.coeffs.len() != n {
            return Err(EmitError::DimensionMismatch);
        }
        let ax: f64 = con.coeffs.iter().zip(x_float).map(|(a, x)| a * x).sum();
        let (lo_fin, hi_fin) = (con.lower.is_finite(), con.upper.is_finite());
        if lo_fin && hi_fin {
            // Equality `a·x = lower` (ranges were split upstream, so lower == upper).
            if con.lower != con.upper {
                return Err(EmitError::SelfCheck("range survived expansion"));
            }
            let row = con
                .coeffs
                .iter()
                .map(|&v| br(v))
                .collect::<Result<Vec<_>, _>>()?;
            out.row_ref.push(RowRef::Eq(out.e_rows.len()));
            out.e_rows.push(row);
            out.d_rhs.push(br(con.lower)?);
        } else if lo_fin {
            // lower ≤ a·x  ⇒  a·x ≥ lower
            let row = con
                .coeffs
                .iter()
                .map(|&v| br(v))
                .collect::<Result<Vec<_>, _>>()?;
            out.row_ref.push(RowRef::Ineq(out.a_rows.len()));
            out.a_rows.push(row);
            out.b_rhs.push(br(con.lower)?);
            out.nslack.push(ax - con.lower);
        } else if hi_fin {
            // a·x ≤ upper  ⇒  −a·x ≥ −upper
            let row = con
                .coeffs
                .iter()
                .map(|&v| br(v).map(|r| -r))
                .collect::<Result<Vec<_>, _>>()?;
            out.row_ref.push(RowRef::Ineq(out.a_rows.len()));
            out.a_rows.push(row);
            out.b_rhs.push(-br(con.upper)?);
            out.nslack.push(con.upper - ax);
        } else {
            return Err(EmitError::SelfCheck("free row after expansion"));
        }
    }
    Ok(out)
}

/// Build an exact `verdict = "infeasible"` certificate from a solve that
/// terminated primal-infeasible, or refuse.
///
/// `y_float` is the solver's dual ray, already in the `λ ≥ 0` convention (AMPL
/// duals are negated by the caller). It is used **only to identify the
/// certificate's support** — the ray itself is recomputed exactly by
/// [`crate::refine_farkas::refine_farkas`], because a diverging float ray
/// satisfies `Aᵀy = 0` only relative to its own magnitude and so is not a
/// certificate over ℚ at all.
///
/// v1 restriction, enforced rather than assumed: every constraint must be a
/// one-sided `a·x ≥ l` row, and the row count must match `y_float`. Bound
/// folding and range splitting change the row set, which would break the
/// correspondence between `y_float` and the emitted rows; rather than guess a
/// mapping, those inputs are refused.
pub fn emit_infeasible_certificate(
    input: &QpInput,
    meta: &CertMeta,
    y_float: &[f64],
    support_tol: f64,
) -> Result<Certificate, EmitError> {
    let problem = problem_block(input)?;

    // Every emitted row must be `a·x ≥ lower`, so the cert's constraint order is
    // exactly the order the Farkas ray indexes.
    let rows = problem.constraint_rows();
    let mut a_rows: Vec<Vec<BigRational>> = Vec::with_capacity(rows.len());
    let mut b_vec: Vec<BigRational> = Vec::with_capacity(rows.len());
    for con in rows {
        let (Some(lo), None) = (con.lower.finite(), con.upper.finite()) else {
            return Err(EmitError::SelfCheck(
                "infeasible v1 accepts one-sided `a·x ≥ l` rows only",
            ));
        };
        a_rows.push(con.coeffs.iter().map(|r| r.inner().clone()).collect());
        b_vec.push(lo.inner().clone());
    }
    if a_rows.len() != y_float.len() {
        return Err(EmitError::SelfCheck(
            "dual ray length does not match the emitted row count",
        ));
    }

    let y = crate::refine_farkas::refine_farkas(&a_rows, &b_vec, y_float, support_tol)
        .map_err(EmitError::Farkas)?;

    Ok(Certificate {
        schema: SCHEMA_TAG.to_string(),
        verdict: "infeasible".to_string(),
        problem_class: "qp-convex".to_string(),
        tolerance: Rat(BigRational::zero()),
        bound: None,
        binding: Binding {
            nl_sha256: meta.nl_sha256.clone(),
            sol_sha256: meta.sol_sha256.clone(),
            solver: meta.solver.clone(),
        },
        toolchain: Toolchain {
            lean: VALIDATED_LEAN.to_string(),
            mathlib: VALIDATED_MATHLIB.to_string(),
        },
        problem,
        candidate: None,
        witnesses: Witnesses {
            duals: None,
            hessian_psd: None,
            active_set: None,
            farkas: Some(Farkas {
                y: y.into_iter().map(Rat).collect(),
            }),
            recession: None,
            sos: None,
            feasible_witness: None,
        },
    })
}

/// Build an exact `verdict = "unbounded"` certificate, or refuse.
///
/// `x_float` is the solver's diverging primal iterate, which serves as **both**
/// witnesses: it is the feasible point `x₀`, and it is the hint for the
/// recession direction `d`. The two roles are verified independently and
/// exactly below — they are separate obligations in the Lean theorem, and a
/// certificate that conflated them would be proving less than it claims.
///
/// The two roles also diverge in how much of the float survives. `x₀` must be
/// *exactly* feasible, so it is the iterate verbatim; there is nothing to
/// refine, because feasibility is a set of inequalities and an inequality with
/// margin survives the lossless f64→ℚ conversion. `d` must satisfy the
/// **equality** `Q d = 0`, which a float direction misses whenever `Q ≠ 0`, so
/// it is recomputed exactly by [`crate::refine_recession`] — the same reason
/// the Farkas path cannot copy the solver's ray.
///
/// This covers any `Q`, including an indefinite one: the recession theorem
/// needs `Q` symmetric and flat along `d`, not convex. Where `Q = 0` the
/// refinement is the identity and this reduces to the original LP slice.
pub fn emit_unbounded_certificate(
    input: &QpInput,
    meta: &CertMeta,
    x_float: &[f64],
) -> Result<Certificate, EmitError> {
    let view = build_problem_view(input)?;
    let problem = view.problem;
    let n = problem.n_vars;
    if x_float.len() != n {
        return Err(EmitError::SelfCheck("iterate length != n_vars"));
    }
    // `objective` is absent only for `sos-poly`, which never reaches this path
    // (it has no candidate point and no constraint rows to recess along).
    let obj = problem
        .objective
        .as_ref()
        .ok_or(EmitError::SelfCheck("unbounded requires a QP objective"))?;

    // Every row must be `a·x ≥ lower`, matching the Lean feasible-set encoding.
    let mut a_rows: Vec<Vec<BigRational>> = Vec::new();
    let mut b_vec: Vec<BigRational> = Vec::new();
    for con in problem.constraint_rows() {
        let (Some(lo), None) = (con.lower.finite(), con.upper.finite()) else {
            return Err(EmitError::SelfCheck(
                "unbounded v1 accepts one-sided `a·x ≥ l` rows only",
            ));
        };
        a_rows.push(con.coeffs.iter().map(|r| r.inner().clone()).collect());
        b_vec.push(lo.inner().clone());
    }

    let v: Vec<BigRational> = x_float
        .iter()
        .map(|&x| Rat::from_f64(x).map(|r| r.inner().clone()))
        .collect::<Result<_, _>>()?;
    let c: Vec<BigRational> = obj.c.iter().map(|r| r.inner().clone()).collect();

    // --- x₀: exactly feasible, verbatim -------------------------------------
    for (i, row) in a_rows.iter().enumerate() {
        if dot(row, &v) < b_vec[i] {
            return Err(EmitError::SelfCheck("iterate is not exactly feasible"));
        }
    }

    // --- d: exactly flat, feasible to travel, and descending ----------------
    // `q_dense` rather than `m_dense`: this is the matrix the generated Lean
    // will carry, so it is the one whose kernel the proof will be about. (The
    // two differ by a positive factor and so have the same kernel — matching
    // the Lean file is about the recheck being the check, not about the math.)
    let d = crate::refine_recession::refine_recession(&view.q_dense, &c, &a_rows, &v)
        .map_err(EmitError::Recession)?;

    Ok(Certificate {
        schema: SCHEMA_TAG.to_string(),
        verdict: "unbounded".to_string(),
        problem_class: "qp-convex".to_string(),
        tolerance: Rat(BigRational::zero()),
        bound: None,
        binding: Binding {
            nl_sha256: meta.nl_sha256.clone(),
            sol_sha256: meta.sol_sha256.clone(),
            solver: meta.solver.clone(),
        },
        toolchain: Toolchain {
            lean: VALIDATED_LEAN.to_string(),
            mathlib: VALIDATED_MATHLIB.to_string(),
        },
        problem,
        candidate: None,
        witnesses: Witnesses {
            duals: None,
            hessian_psd: None,
            active_set: None,
            farkas: None,
            recession: Some(Recession {
                x0: v.into_iter().map(Rat).collect(),
                d: d.into_iter().map(Rat).collect(),
            }),
            sos: None,
            feasible_witness: None,
        },
    })
}

/// Round `v > 0` **up** to one significant decimal digit; `0` stays `0`.
///
/// A tolerance is a declared bound, and a declaration reads better as `4/10^9`
/// than as the 60-digit fraction an f64 residual actually is. Rounding *up* is
/// what keeps it a bound: the true residual stays below the number published.
fn round_up_1sig(v: &BigRational) -> BigRational {
    if v.is_zero() {
        return BigRational::zero();
    }
    let ten = BigRational::from_integer(10.into());
    let one = BigRational::from_integer(1.into());
    let mut scale = one.clone();
    // Smallest power of ten that lifts `v` to at least 1 — i.e. its leading
    // digit. Terminates because `v > 0`.
    while &scale * v < one {
        scale *= &ten;
    }
    let scaled = &scale * v;
    BigRational::from_integer(scaled.ceil().to_integer()) / scale
}

/// Build an exact `verdict = "feasible"` certificate, or refuse.
///
/// This is the weakest verdict in the schema and the only one that says nothing
/// about optimality: *the reported point misses feasibility by at most ε, and a
/// genuinely feasible point exists within ε of it*. It is worth having because a
/// solve that cannot be certified optimal — a nonconvex model, a `Q` that is not
/// PSD, an active set that will not refine — has still produced something, and
/// "this is a real feasible point of the real problem" is a claim a reviewer can
/// use.
///
/// Unlike every other verdict, ε is **nonzero**: the candidate is the solver's
/// float `x̃` verbatim (rational image), residual and all. The exactness lives in
/// `witnesses.feasible_witness`, which [`crate::refine_feasible`] computes by
/// projecting `x̃` onto the active face over ℚ, and in ε itself, which is a
/// rational bound on residuals computed exactly rather than a solver tolerance
/// copied across.
///
/// Refuses when no exactly-feasible point can be produced — a wrong active set
/// or an infeasible problem — rather than publishing a bare ε-feasibility claim
/// about a point that may have no feasible neighbourhood at all.
pub fn emit_feasible_certificate(
    input: &QpInput,
    meta: &CertMeta,
) -> Result<Certificate, EmitError> {
    let n = input.n;
    let view = build_problem_view(input)?;
    if input.x_float.len() != n {
        return Err(EmitError::DimensionMismatch);
    }
    let ProblemView {
        problem,
        all_constraints,
        m_dense,
        c_rat,
        constant_rat,
        ..
    } = view;

    let NormalizedRows {
        a_rows,
        b_rhs,
        e_rows,
        d_rhs,
        nslack,
        ..
    } = normalize_rows(n, &all_constraints, &input.x_float)?;

    // The candidate is the float point itself. Nothing is refined into it: the
    // whole point of this verdict is to certify what the solver actually
    // returned, with the correction shipped alongside as a separate witness.
    let xstar: Vec<BigRational> = input
        .x_float
        .iter()
        .map(|&v| br(v))
        .collect::<Result<_, _>>()?;

    // Rows the float solve sits on are the ones to pin; the projection is
    // verified against every row afterwards, so this is a guess like any other.
    let active: Vec<usize> = (0..a_rows.len())
        .filter(|&i| nslack[i].abs() <= input.active_tol)
        .collect();
    let xhat =
        crate::refine_feasible::refine_feasible(&a_rows, &b_rhs, &e_rows, &d_rhs, &active, &xstar)?;

    // ε must cover both claims the consumer makes: that `x*` is ε-feasible, and
    // that `x̂` is within ε of `x*`. Neither implies the other — a row with large
    // coefficients turns a small step into a large residual, and a nearly
    // parallel pair of rows the reverse — so ε is the max of the two.
    let mut worst = BigRational::zero();
    for (i, row) in a_rows.iter().enumerate() {
        let violation = &b_rhs[i] - dot(row, &xstar);
        if violation > worst {
            worst = violation;
        }
    }
    for (j, row) in e_rows.iter().enumerate() {
        let residual = dot(row, &xstar) - &d_rhs[j];
        let residual = if residual < BigRational::zero() {
            -residual
        } else {
            residual
        };
        if residual > worst {
            worst = residual;
        }
    }
    for (xh, xs) in xhat.iter().zip(&xstar) {
        let step = xh - xs;
        let step = if step < BigRational::zero() {
            -step
        } else {
            step
        };
        if step > worst {
            worst = step;
        }
    }
    let eps = round_up_1sig(&worst);

    let objective = objective_value(&m_dense, &c_rat, &constant_rat, &xstar);

    Ok(Certificate {
        schema: SCHEMA_TAG.to_string(),
        verdict: "feasible".to_string(),
        problem_class: "qp-convex".to_string(),
        tolerance: Rat(eps),
        bound: None,
        binding: Binding {
            nl_sha256: meta.nl_sha256.clone(),
            sol_sha256: meta.sol_sha256.clone(),
            solver: meta.solver.clone(),
        },
        toolchain: Toolchain {
            lean: VALIDATED_LEAN.to_string(),
            mathlib: VALIDATED_MATHLIB.to_string(),
        },
        problem,
        candidate: Some(Candidate {
            x: xstar.into_iter().map(Rat).collect(),
            objective: Rat(objective),
        }),
        witnesses: Witnesses {
            duals: None,
            hessian_psd: None,
            active_set: None,
            farkas: None,
            recession: None,
            sos: None,
            feasible_witness: Some(FeasibleWitness {
                xhat: xhat.into_iter().map(Rat).collect(),
            }),
        },
    })
}

/// Build an exact certificate from a neutral QP solve, or refuse.
pub fn emit_certificate(input: &QpInput, meta: &CertMeta) -> Result<Certificate, EmitError> {
    let n = input.n;
    let view = build_problem_view(input)?;
    if input.x_float.len() != n {
        return Err(EmitError::DimensionMismatch);
    }
    let ProblemView {
        problem,
        all_constraints,
        q_dense,
        m_dense,
        c_rat,
        constant_rat,
    } = view;

    // Route each row to the inequality system `A x ≥ b` (λ ≥ 0) or the equality
    // system `E x = d` (free-sign μ), remembering which the cert constraint maps
    // to so its single dual lands in the right place.
    let NormalizedRows {
        a_rows,
        b_rhs,
        e_rows,
        d_rhs,
        nslack,
        row_ref,
    } = normalize_rows(n, &all_constraints, &input.x_float)?;

    // Active-set guess from the float point; refinement validates it exactly.
    let active: Vec<usize> = (0..a_rows.len())
        .filter(|&i| nslack[i].abs() <= input.active_tol)
        .collect();

    // Exact optimizer + duals (λ for inequalities, free-sign μ for equalities),
    // then the exact PSD factorization of the cert Q.
    let refined = refine_kkt_eq(&m_dense, &c_rat, &a_rows, &b_rhs, &active, &e_rows, &d_rhs)?;
    let factor = ldlt(&q_dense)?;

    // Per cert constraint: its dual (λ or μ) and whether it is active. Equalities
    // are always active; inequalities are active iff their normalized slack was 0.
    let active_ineq: std::collections::HashSet<usize> = active.iter().copied().collect();
    let mut duals_cert: Vec<BigRational> = Vec::with_capacity(row_ref.len());
    let mut active_set: Vec<usize> = Vec::new();
    for (cert_idx, rref) in row_ref.iter().enumerate() {
        match rref {
            RowRef::Ineq(ai) => {
                duals_cert.push(refined.lambda[*ai].clone());
                if active_ineq.contains(ai) {
                    active_set.push(cert_idx);
                }
            }
            RowRef::Eq(ei) => {
                duals_cert.push(refined.mu[*ei].clone());
                active_set.push(cert_idx);
            }
        }
    }
    active_set.sort_unstable();

    // ---- assemble the certificate (problem block comes from the Frontend) ----
    let mut l_entries: Vec<Entry> = factor
        .l_below
        .iter()
        .map(|(i, j, v)| Entry {
            i: *i,
            j: *j,
            val: Rat(v.clone()),
        })
        .collect();
    l_entries.sort_by_key(|e| (e.i, e.j));

    let objective = objective_value(&m_dense, &c_rat, &constant_rat, &refined.x);

    let cert = Certificate {
        schema: SCHEMA_TAG.to_string(),
        verdict: "global-min".to_string(),
        problem_class: "qp-convex".to_string(),
        // Exact slice: feasibility holds with zero residual.
        tolerance: Rat::zero(),
        bound: None,
        binding: Binding {
            nl_sha256: meta.nl_sha256.clone(),
            sol_sha256: meta.sol_sha256.clone(),
            solver: meta.solver.clone(),
        },
        toolchain: Toolchain {
            lean: VALIDATED_LEAN.to_string(),
            mathlib: VALIDATED_MATHLIB.to_string(),
        },
        problem,
        candidate: Some(Candidate {
            x: refined.x.iter().map(|v| Rat(v.clone())).collect(),
            objective: Rat(objective.clone()),
        }),
        witnesses: Witnesses {
            duals: Some(duals_cert.iter().map(|v| Rat(v.clone())).collect()),
            hessian_psd: Some(HessianPsd {
                of: "Q".to_string(),
                l: SparseMatrix::unit_lower(n, n, l_entries),
                d: factor.d.iter().map(|v| Rat(v.clone())).collect(),
            }),
            active_set: Some(active_set),
            farkas: None,
            recession: None,
            sos: None,
            feasible_witness: None,
        },
    };

    self_check(
        &m_dense,
        &c_rat,
        &a_rows,
        &b_rhs,
        &refined.x,
        &refined.lambda,
        &e_rows,
        &d_rhs,
        &refined.mu,
    )?;
    Ok(cert)
}

/// Exact, independent recheck of the assembled certificate's load-bearing KKT
/// claims. `refine_kkt`/`ldlt` already enforce these by construction; this is a
/// belt-and-suspenders gate so a future shape bug can never ship a bad cert.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn self_check(
    m: &[Vec<BigRational>],
    c: &[BigRational],
    a: &[Vec<BigRational>],
    b: &[BigRational],
    x: &[BigRational],
    lambda: &[BigRational],
    e: &[Vec<BigRational>],
    d: &[BigRational],
    mu: &[BigRational],
) -> Result<(), EmitError> {
    let n = x.len();

    // Inequalities: dual sign, primal feasibility, and complementarity.
    for (i, row) in a.iter().enumerate() {
        let slack = dot(row, x) - &b[i];
        if lambda[i] < BigRational::zero() {
            return Err(EmitError::SelfCheck("dual sign"));
        }
        if slack < BigRational::zero() {
            return Err(EmitError::SelfCheck("primal feasibility"));
        }
        if !(&lambda[i] * &slack).is_zero() {
            return Err(EmitError::SelfCheck("complementarity"));
        }
    }

    // Equalities must hold exactly (μ is free-sign, so no sign check).
    for (j, row) in e.iter().enumerate() {
        if dot(row, x) != d[j] {
            return Err(EmitError::SelfCheck("equality residual"));
        }
    }

    // Stationarity: (M x)_i + c_i = (Aᵀλ)_i + (Eᵀμ)_i for every variable, exactly.
    for i in 0..n {
        let mx: BigRational = (0..n).map(|j| &m[i][j] * &x[j]).sum();
        let atl: BigRational = a
            .iter()
            .enumerate()
            .map(|(k, row)| &row[i] * &lambda[k])
            .sum();
        let etm: BigRational = e.iter().enumerate().map(|(k, row)| &row[i] * &mu[k]).sum();
        if &mx + &c[i] != &atl + &etm {
            return Err(EmitError::SelfCheck("stationarity"));
        }
    }
    Ok(())
}
