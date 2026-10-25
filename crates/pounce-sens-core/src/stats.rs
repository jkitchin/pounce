//! Parameter covariance, the information matrix and identifiability, from
//! one converged solve.
//!
//! [`covariance`] is the asymptotic covariance of a least-squares estimate,
//! and [`information`] is the reduced Hessian it inverts, which stays finite
//! when the block is rank-deficient and the covariance does not. Both answer
//! about a *block*: model variables, by full-x index. The declared fitted
//! parameters are the default block.
//!
//! Everything here runs over any [`BarrierIterate`] through
//! [`ConvergedSolve`]. What a caller should be told about the answer comes
//! back as [`StatsFinding`]s beside it, in the order the calculation meets
//! them, and a failure as a [`StatsError`]. Neither carries prose about
//! names: the caller's front end holds the names and writes the messages.

use pounce_common::types::{Index, Number};

use crate::activity::{
    AMBIGUOUS, ActivityReport, EQUALITY, FIXED, INACTIVE, STRONGLY_ACTIVE, UNBOUNDED, UNIDENTIFIED,
    WEAKLY_ACTIVE,
};
use crate::barrier::BarrierIterate;
use crate::converged::{ConvergedSolve, SolverError};
use crate::dense::{self, Dense};
use crate::model::{Declarations, SolveRecord};
use crate::step::{StepError, check_margins, refuse_on_pdpert};

/// Which information matrix a statistic is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HessianForm {
    /// The exact reduced Hessian of the Lagrangian off the held factor: the
    /// observed information.
    Lagrangian,
    /// `2 JᵀJ` from the residual Jacobian: the expected information,
    /// positive semidefinite by construction. Needs declared residuals.
    GaussNewton,
}

/// Why a block has no inverse to classify its members with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deficiency {
    /// More coordinates than the fit has degrees of freedom.
    Count,
    /// Linearly dependent coordinates.
    Dependent,
}

/// Where a model variable the engine removed was asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemovedAt {
    /// As block member `k`.
    Member(usize),
    /// As a residual.
    Residual,
    /// As a fitted parameter, building a sub-block's marginal.
    Fitted,
    /// As a candidate for the variables the answer is conditional on.
    ConditionedOn,
}

/// What a caller should be told about a statistic, beside the answer.
#[derive(Debug, Clone, PartialEq)]
pub enum StatsFinding {
    /// The held factor carries an inertia correction, so the answer is
    /// regularized rather than exact.
    Perturbed {
        /// `(δ_x, δ_s, δ_c, δ_d)`.
        perturbations: [Number; 4],
    },
    /// Block member `member` is held by its bound (strongly active): its
    /// direction is projected out.
    MemberHeld {
        /// Position in the block.
        member: usize,
    },
    /// Block member `member` sits on its bound with a vanishing multiplier
    /// (weakly active): it is kept with finite variance.
    MemberWeak {
        /// Position in the block.
        member: usize,
    },
    /// Block member `member` has ambiguous bound activity: it is kept.
    MemberAmbiguous {
        /// Position in the block.
        member: usize,
    },
    /// Block member `member` has curvature below the model's noise scale:
    /// it is kept, and its variance is large.
    MemberUnidentified {
        /// Position in the block.
        member: usize,
    },
    /// Model row `row` is strongly active and involves variables outside
    /// the block, so the direction it pins is not projected.
    RowMixedHeld {
        /// Full-g row.
        row: usize,
    },
    /// Model row `row` has activity `status` and involves variables outside
    /// the block: it is kept unprojected and uncorrected.
    RowMixedKept {
        /// Full-g row.
        row: usize,
        /// Its activity code.
        status: i8,
    },
    /// Model row `row` is strongly active and pins the block combination
    /// `combo`: the variance along it is projected to zero.
    RowPins {
        /// Full-g row.
        row: usize,
        /// `(member, coefficient)` of the row's unit normal over the block.
        combo: Vec<(usize, Number)>,
        /// The conditional information along the combination.
        information: Number,
    },
    /// Model row `row` is weakly active on `combo`: kept unprojected.
    RowWeak {
        /// Full-g row.
        row: usize,
        /// See [`Self::RowPins`].
        combo: Vec<(usize, Number)>,
    },
    /// Model row `row` has ambiguous activity on `combo`: kept unprojected.
    RowAmbiguous {
        /// Full-g row.
        row: usize,
        /// See [`Self::RowPins`].
        combo: Vec<(usize, Number)>,
    },
    /// Model row `row` has curvature below the block's noise scale on
    /// `combo`: kept unprojected.
    RowUnidentified {
        /// Full-g row.
        row: usize,
        /// See [`Self::RowPins`].
        combo: Vec<(usize, Number)>,
    },
    /// A count of data points was given beside a higher-precedence noise
    /// source and is ignored.
    NDataIgnored,
    /// Block members with bound activity in a rank-deficient block, whose
    /// membership handling is unavailable.
    DeficientMembers {
        /// Positions in the block.
        members: Vec<usize>,
        /// Why the block is rank-deficient.
        reason: Deficiency,
    },
    /// The covariance has a negative variance on its diagonal.
    NegativeVariance,
    /// The Lagrangian information's free block is indefinite.
    Indefinite,
    /// A sub-block's exact Schur route found the free fitted coordinates
    /// outside it singular, and the reduction off the factor is used.
    SubblockFallback,
}

/// Why a statistic has no answer.
#[derive(Debug, Clone)]
pub enum StatsError {
    /// A calculation over the factor failed.
    Solver(SolverError),
    /// A margin or the inertia-correction cap refused the call.
    Step(StepError),
    /// A dense eigen- or singular value decomposition did not converge.
    Numeric(String),
    /// The requested block of the inverse KKT matrix is singular: its
    /// members are linearly dependent.
    SingularBlock,
    /// The noise variance has to be estimated from the residuals, and the
    /// solve recorded no point to read them at.
    NoSolvePoint,
    /// Model variable `full_x` was removed from the solve as fixed, so it
    /// has no row in the factor.
    Removed {
        /// The model variable.
        full_x: usize,
        /// Where it was asked for.
        at: RemovedAt,
    },
    /// No block was given and no fitted parameters were declared.
    NoBlock,
    /// The Gauss-Newton form needs declared residuals.
    GaussNewtonNeedsResiduals,
    /// The noise variance has to be estimated from the residuals, and no
    /// fitted parameters were declared to count the degrees of freedom.
    NoFitForResiduals,
    /// Residual group `group` has no more members than there are fitted
    /// parameters.
    GroupTooSmall {
        /// Its position among the declared groups.
        group: usize,
        /// Its member count.
        members: usize,
        /// The fitted parameter count.
        n_fit: usize,
    },
    /// A count of data points was given and no fitted parameters were
    /// declared to count the degrees of freedom.
    NDataNoFit,
    /// The count of data points does not exceed the fitted parameters.
    NDataTooSmall {
        /// The count given.
        n_data: Number,
        /// The fitted parameter count.
        n_fit: usize,
    },
    /// The solve reported no usable objective value to estimate the noise
    /// variance from.
    NoObjectiveValue {
        /// The value it reported.
        obj: Number,
    },
    /// No noise source was given.
    NoiseUnknown,
    /// The block is rank-deficient, so only the homoscedastic Lagrangian
    /// marginal covariance exists.
    DeficientNeedsLagrangian {
        /// Why it is rank-deficient.
        reason: Deficiency,
    },
    /// The Gauss-Newton matrix `JᵀJ` is singular.
    GaussNewtonSingular,
    /// The reduced Hessian restricted to the free members is singular.
    FreeSingular,
    /// The block is rank-deficient and carries no information matrix.
    InformationDeficient {
        /// Why it is rank-deficient.
        reason: Deficiency,
    },
    /// The Lagrangian information needs the square estimation structure.
    NotSquare {
        /// Variables in the factor.
        n_var: usize,
        /// Equality rows.
        n_eq: usize,
        /// Block members.
        n_params: usize,
    },
    /// The free block is singular, so the pinned members' conditional
    /// information is not defined.
    PinnedUndefined,
}

impl From<SolverError> for StatsError {
    fn from(e: SolverError) -> Self {
        StatsError::Solver(e)
    }
}

impl From<StepError> for StatsError {
    fn from(e: StepError) -> Self {
        StatsError::Step(e)
    }
}

/// An answer, and everything the caller should be told about it.
///
/// The findings are there whether or not the answer is: a statistic that
/// warns about a member and then fails on the noise variance has said both.
#[derive(Debug)]
pub struct Outcome<T> {
    /// What the calculation met, in order.
    pub findings: Vec<StatsFinding>,
    /// The answer.
    pub result: Result<T, StatsError>,
}

/// The block a statistic answers about: model variables by full-x index.
/// `None` asks about the declared fitted parameters.
pub type Block<'a> = Option<&'a [usize]>;

/// The covariance, and the noise variance it was scaled by.
#[derive(Debug, Clone, PartialEq)]
pub struct Covariance {
    /// The covariance over the block, in its order.
    pub matrix: Dense,
    /// The noise variance per residual group, in their order, or one value
    /// when no groups were declared.
    pub sigma_sq: Vec<Number>,
    /// The strongly active model variables outside the block, which the
    /// matrix is conditional on. See [`conditioned_on`].
    pub conditioned_on: Vec<usize>,
}

/// The information matrix, and the bounds it is conditional on.
#[derive(Debug, Clone, PartialEq)]
pub struct Information {
    /// The information over the block, in its order.
    pub matrix: Dense,
    /// The strongly active model variables outside the block, which the
    /// matrix is conditional on. See [`conditioned_on`].
    pub conditioned_on: Vec<usize>,
}

/// A member's activity, at the reduced level: item 0's rule with `q` the
/// curvature left at the block. Mirrors the classifier's bands, with the
/// one difference that a `q` below the floor is unidentified only when the
/// ratio is small too, since a huge `Σ` cancelling inside the reduced
/// curvature would otherwise misfile a strongly active entry.
pub fn classify_ratio(r: Number, mu: Number) -> i8 {
    if mu > 1e-4 {
        if r < 1e-1 {
            return INACTIVE;
        }
        if r > 1e1 {
            return STRONGLY_ACTIVE;
        }
        return AMBIGUOUS;
    }
    if r < mu.sqrt() {
        return INACTIVE;
    }
    if r > 1.0 / mu.sqrt() {
        return STRONGLY_ACTIVE;
    }
    if (1e-1..=1e1).contains(&r) {
        return WEAKLY_ACTIVE;
    }
    AMBIGUOUS
}

/// What the statistics read off the converged solve, gathered once.
struct Context<'s, 'a, E: BarrierIterate + ?Sized> {
    conv: &'s ConvergedSolve<'a, E>,
    decl: Declarations<'s>,
    /// Each model variable's row in the factor, `None` for a removed one.
    to_var: Vec<Option<usize>>,
    /// Variables in the factor.
    n_var: usize,
    /// Equality rows.
    n_eq: usize,
    dim: usize,
    /// The classification of the solve, computed when first needed, so
    /// the members, the rows and the conditioned bounds read the same one.
    act: std::cell::OnceCell<ActivityReport>,
}

impl<'s, 'a, E: BarrierIterate + ?Sized> Context<'s, 'a, E> {
    fn new(conv: &'s ConvergedSolve<'a, E>, decl: Declarations<'s>) -> Result<Self, StatsError> {
        let dims = conv.engine().block_dims();
        let n_full = conv.n_full_x()?;
        let rows: Vec<Index> = (0..n_full as Index).collect();
        let to_var = conv
            .x_primal_rows(&rows)?
            .into_iter()
            .map(|r| r.map(|r| r as usize))
            .collect();
        Ok(Self {
            conv,
            decl,
            to_var,
            n_var: dims[0],
            n_eq: dims[2],
            dim: conv.engine().dim(),
            act: std::cell::OnceCell::new(),
        })
    }

    fn act(&self) -> Result<&ActivityReport, StatsError> {
        if let Some(act) = self.act.get() {
            return Ok(act);
        }
        let act = self.conv.classify_activity()?;
        Ok(self.act.get_or_init(|| act))
    }

    fn primal_row(&self, full: usize, at: RemovedAt) -> Result<usize, StatsError> {
        self.to_var
            .get(full)
            .copied()
            .flatten()
            .ok_or(StatsError::Removed { full_x: full, at })
    }

    /// A model-space vector from a factor-space (var-x) one: variables the
    /// engine removed get zero, since a fixed variable does not move.
    fn scatter_x(&self, dx_var: &[Number]) -> Vec<Number> {
        self.to_var
            .iter()
            .map(|r| r.map_or(0.0, |r| dx_var[r]))
            .collect()
    }

    /// The columns of the inverse KKT matrix for the given factor rows.
    fn unit_columns(&self, krows: &[usize]) -> Result<Vec<Vec<Number>>, StatsError> {
        krows
            .iter()
            .map(|&r| {
                let mut e = vec![0.0; self.dim];
                e[r] = 1.0;
                let mut z = vec![0.0; self.dim];
                self.conv.kkt_solve(&e, &mut z)?;
                Ok(z)
            })
            .collect()
    }
}

/// The block of the inverse KKT matrix on `krows`, symmetrized.
fn block_of(zcols: &[Vec<Number>], krows: &[usize]) -> Dense {
    let n = krows.len();
    Dense::from_fn(n, n, |i, j| zcols[j][krows[i]]).symmetrized()
}

fn minv(m: &Dense) -> Result<Dense, StatsError> {
    dense::inv(m).ok_or(StatsError::SingularBlock)
}

/// The reduced Hessian over the block by tangent recovery: the x-blocks of
/// the inverse-KKT columns are `T M`, so `T = Z_x M⁻¹` exactly, with the
/// factor's barrier weight cancelling multiplicatively rather than by
/// subtraction. Then `R = Tᵀ H T` with the exact Lagrangian Hessian
/// (covariance roadmap item 2).
///
/// Machine-exact for equality and variable-bound activity. A binding
/// inequality row couples through its slack barrier with large but finite
/// weight, tilting the recovered tangent along that row's normal by about
/// `1e-6` relative at practical `μ`. Requires the square estimation
/// structure, which the callers check.
fn tangent_reduced_hessian<E: BarrierIterate + ?Sized>(
    cx: &Context<'_, '_, E>,
    m: &Dense,
    zcols: &[Vec<Number>],
) -> Result<Dense, StatsError> {
    let k = zcols.len();
    let zx = Dense::from_fn(cx.n_var, k, |i, j| zcols[j][i]);
    let tv = zx.mul(&minv(m)?);
    let t: Vec<Vec<Number>> = (0..k)
        .map(|j| {
            let col: Vec<Number> = (0..cx.n_var).map(|i| tv.at(i, j)).collect();
            cx.scatter_x(&col)
        })
        .collect();
    let ht: Vec<Vec<Number>> = t
        .iter()
        .map(|c| cx.conv.hessian_vec(c))
        .collect::<Result<_, _>>()?;
    let r = Dense::from_fn(k, k, |i, j| dense::dot(&t[i], &ht[j]));
    Ok(r.symmetrized())
}

/// What classifying the block's members and the rows binding on it left:
/// item 1's membership, value corrections and binding normals.
struct Classified {
    sig_fit: Vec<Number>,
    floor: Number,
    active: Vec<usize>,
    free: Vec<usize>,
    r_corr: Dense,
    bind_normals: Vec<Vec<Number>>,
    row_corrections: Vec<(Number, Vec<Number>)>,
    r_exact: Option<Dense>,
}

/// Item 1's classification at the reduced block: which members their bound
/// holds, which rows bind on the block, and the corrections the kept ones
/// need.
///
/// The classifier's per-coordinate rule scales `Σ` by the coordinate's own
/// curvature, which is zero for a fitted parameter written in the
/// residual-variable idiom, so the same rule runs here on the reduced
/// block, where the curvature is: `q` is the reduced Hessian's diagonal with
/// the member's own barrier term removed. A weakly active member is kept and
/// reported rather than pinned. No slack threshold can make that
/// distinction, since slack and multiplier are both `O(√μ)` there.
fn classify_block<E: BarrierIterate + ?Sized>(
    cx: &Context<'_, '_, E>,
    rows: &[usize],
    m: &Dense,
    zcols: &[Vec<Number>],
    findings: &mut Vec<StatsFinding>,
) -> Result<Classified, StatsError> {
    let n_params = rows.len();
    let mut r_exact: Option<Dense> = None;
    let act = cx.act()?;
    let mu = act.mu;
    // M, and so R_W, is natural-units by the kkt_solve contract, and so are
    // the report's diagonals and the row normals, so everything composes
    // without scale factors
    let r_w = minv(m)?;
    let sig_fit: Vec<Number> = rows.iter().map(|&r| act.var_sigma[r]).collect();
    let diag = r_w.diagonal();
    let q_red: Vec<Number> = (0..n_params)
        .map(|i| (diag[i] - sig_fit[i]).abs())
        .collect();
    let floor = Number::EPSILON.sqrt() * diag.iter().fold(0.0_f64, |a, &b| a.max(b.abs())).max(1.0);
    let mut active = Vec::new();
    for i in 0..n_params {
        let st = act.var_status[rows[i]];
        if st == UNBOUNDED || st == FIXED {
            continue;
        }
        let ri = sig_fit[i] / q_red[i].max(floor);
        let status = if q_red[i] < floor && ri <= 1e1 {
            // curvature AND barrier weight both below scale: the bound
            // question does not arise, but the direction is poorly
            // identified
            UNIDENTIFIED
        } else {
            classify_ratio(ri, mu)
        };
        match status {
            STRONGLY_ACTIVE => {
                active.push(i);
                findings.push(StatsFinding::MemberHeld { member: i });
            }
            WEAKLY_ACTIVE => findings.push(StatsFinding::MemberWeak { member: i }),
            AMBIGUOUS => findings.push(StatsFinding::MemberAmbiguous { member: i }),
            UNIDENTIFIED => findings.push(StatsFinding::MemberUnidentified { member: i }),
            _ => {}
        }
    }

    // A strongly active inequality row whose normal touches the block pins
    // a DIRECTION of it, which no per-member disposition can state, so the
    // free block is reduced on the null space of the binding normals. Rows
    // classify at the reduced level exactly as the members do: the row's
    // barrier weight against the curvature along its own normal.
    let mut r_corr = r_w.sub(&Dense::from_fn(n_params, n_params, |i, j| {
        if i == j { sig_fit[i] } else { 0.0 }
    }));
    // columns the declared-parameter pins hold constant: a row's support
    // there contributes nothing through elimination, so it does not make
    // the row mixed. A pin row's normal is the unit vector of its variable.
    let mut pin_cols = std::collections::BTreeSet::new();
    for &pr in cx.decl.pins {
        let pn = cx.conv.row_normal(pr)?;
        pin_cols.extend(
            pn.iter()
                .enumerate()
                .filter(|(_, v)| **v != 0.0)
                .map(|(i, _)| i),
        );
    }
    let fit_cols: std::collections::BTreeSet<usize> = rows.iter().copied().collect();
    let mut bind_normals = Vec::new();
    let mut row_corrections = Vec::new();
    for (j, &rst) in act.row_status.iter().enumerate() {
        if rst == EQUALITY || rst == UNBOUNDED || rst == INACTIVE {
            // inactive rows carry O(mu) geometric weight, the same order as
            // every other accepted O(mu) term, and skipping them spares a
            // row normal per row on wide models
            continue;
        }
        let a_full = cx.conv.row_normal(j)?;
        let mut a: Vec<Number> = rows.iter().map(|&r| a_full[r]).collect();
        let na = dense::norm(&a);
        // A row whose normal also touches variables outside the block pins
        // a combination that reaches the block through the eliminated
        // variables, not along the restricted normal, so the restricted
        // projection would delete the wrong direction. It is kept
        // unprojected instead.
        let nf = dense::norm(&a_full);
        let outside: Vec<Number> = a_full
            .iter()
            .enumerate()
            .filter(|(i, v)| **v != 0.0 && !fit_cols.contains(i) && !pin_cols.contains(i))
            .map(|(_, v)| *v)
            .collect();
        let mut mixed = !outside.is_empty() && dense::norm(&outside) > 1e-8 * nf.max(1.0);
        if na <= 1e-12 * nf.max(1.0) {
            // entirely outside the block: the extreme mixed case
            mixed = true;
        }
        if mixed {
            // the reduced-level rule is unreliable here too: the row's
            // barrier weight lands through elimination on a direction the
            // restricted normal cannot see, so item 0's raw class stands
            if rst == STRONGLY_ACTIVE {
                findings.push(StatsFinding::RowMixedHeld { row: j });
            } else if rst == WEAKLY_ACTIVE || rst == AMBIGUOUS || rst == UNIDENTIFIED {
                findings.push(StatsFinding::RowMixedKept {
                    row: j,
                    status: rst,
                });
            }
            continue;
        }
        for v in a.iter_mut() {
            *v /= na;
        }
        // the row's slack elimination contributes Σ_j times the raw normal's
        // outer product to the reduced block, so in the unit-normal basis
        // the coefficient is Σ_j ‖raw normal‖²
        let sig_row = act.row_sigma[j] * na * na;
        let q_w = dense::dot(&r_corr.t().mul_vec(&a), &a);
        // |q|, matching the classifier: indefinite curvature classifies on
        // magnitude, and indefiniteness still surfaces downstream
        let q_row = (q_w - sig_row).abs();
        let ri = sig_row / q_row.max(floor);
        let status = if q_row < floor && ri <= 1e1 {
            UNIDENTIFIED
        } else {
            classify_ratio(ri, mu)
        };
        let combo: Vec<(usize, Number)> = (0..n_params)
            .filter(|&k| a[k].abs() > 1e-12)
            .map(|k| (k, a[k]))
            .collect();
        match status {
            STRONGLY_ACTIVE => {
                // the conditional information along the pinned combination,
                // by tangent recovery where the square structure allows it
                if r_exact.is_none() && cx.n_var - cx.n_eq == n_params {
                    r_exact = Some(tangent_reduced_hessian(cx, m, zcols)?);
                }
                let s_a = match &r_exact {
                    Some(r) => dense::dot(&r.t().mul_vec(&a), &a),
                    None => (q_w - sig_row).max(0.0),
                };
                bind_normals.push(a.clone());
                findings.push(StatsFinding::RowPins {
                    row: j,
                    combo,
                    information: s_a,
                });
            }
            WEAKLY_ACTIVE => findings.push(StatsFinding::RowWeak { row: j, combo }),
            AMBIGUOUS => findings.push(StatsFinding::RowAmbiguous { row: j, combo }),
            UNIDENTIFIED => findings.push(StatsFinding::RowUnidentified { row: j, combo }),
            _ => {}
        }
        if status == WEAKLY_ACTIVE || status == AMBIGUOUS {
            // applied after the loop, so every row classifies against the
            // same snapshot whatever order the rows are visited in
            row_corrections.push((sig_row, a));
        }
    }
    for (w, a) in &row_corrections {
        // a kept row's own barrier weight comes off the reduced block
        let outer = Dense::from_fn(n_params, n_params, |i, j| a[i] * a[j]);
        r_corr = r_corr.sub(&outer.scale(*w));
    }
    let free = (0..n_params).filter(|i| !active.contains(i)).collect();
    Ok(Classified {
        sig_fit,
        floor,
        active,
        free,
        r_corr,
        bind_normals,
        row_corrections,
        r_exact,
    })
}

/// The projection basis over the free members: the null space of the
/// binding normals restricted to them. A normal that vanished with the
/// pinned members is dropped, since the free restriction already excludes
/// its content.
fn free_nullspace(bind_normals: &[Vec<Number>], free: &[usize]) -> Result<Dense, StatsError> {
    let nf = free.len();
    let mut rows: Vec<Vec<Number>> = Vec::new();
    for a in bind_normals {
        let af: Vec<Number> = free.iter().map(|&i| a[i]).collect();
        let nn = dense::norm(&af);
        if nn > 1e-12 {
            rows.push(af.iter().map(|v| v / nn).collect());
        }
    }
    let a = Dense::from_fn(rows.len(), nf, |i, j| rows[i][j]);
    dense::nullspace(&a).map_err(StatsError::Numeric)
}

/// The marginal information of a proper sub-block of the fitted set, by
/// Schur complement of the exact tangent `R` over the fitted block, so a
/// pinned member costs no digits. Pinned fitted variables outside the block
/// are conditioned on and free ones profiled out. `None` when it does not
/// apply: the block is not inside a square fitted set, or the fitted level
/// carries binding rows, whose projection does not compose simply with
/// marginalization.
fn subblock_information<E: BarrierIterate + ?Sized>(
    cx: &Context<'_, '_, E>,
    rows: &[usize],
    dof: usize,
    findings: &mut Vec<StatsFinding>,
) -> Result<Option<Dense>, StatsError> {
    let fit = cx.decl.fit_rows;
    if fit.len() != dof {
        return Ok(None);
    }
    let pos = |r: usize| fit.iter().position(|&f| f == r);
    if rows.iter().any(|&r| pos(r).is_none()) {
        return Ok(None);
    }
    let fkrows: Vec<usize> = fit
        .iter()
        .map(|&r| cx.primal_row(r, RemovedAt::Fitted))
        .collect::<Result<_, _>>()?;
    let zcols_f = cx.unit_columns(&fkrows)?;
    let m_f = block_of(&zcols_f, &fkrows);
    // the fitted-level classification is scaffolding here: what it finds
    // belongs to the block-level pass the caller runs
    let mut discarded = Vec::new();
    let cb_f = classify_block(cx, fit, &m_f, &zcols_f, &mut discarded)?;
    if !cb_f.bind_normals.is_empty() {
        return Ok(None);
    }
    let r_f = match cb_f.r_exact {
        Some(r) => r,
        None => tangent_reduced_hessian(cx, &m_f, &zcols_f)?,
    };
    let keep: Vec<usize> = rows
        .iter()
        .map(|&r| pos(r).expect("checked above"))
        .collect();
    let o_free: Vec<usize> = (0..dof)
        .filter(|i| !keep.contains(i) && !cb_f.active.contains(i))
        .collect();
    let sel: Vec<usize> = keep.iter().chain(&o_free).copied().collect();
    let rs = r_f.select(&sel, &sel);
    let nb = keep.len();
    let head: Vec<usize> = (0..nb).collect();
    let r_b = if o_free.is_empty() {
        rs.select(&head, &head)
    } else {
        let tail: Vec<usize> = (nb..sel.len()).collect();
        let a = rs.select(&head, &head);
        let b = rs.select(&head, &tail);
        let d = rs.select(&tail, &tail);
        match dense::solve(&d, &b.t()) {
            Some(x) => a.sub(&b.mul(&x)),
            None => {
                findings.push(StatsFinding::SubblockFallback);
                return Ok(None);
            }
        }
    };
    Ok(Some(r_b.symmetrized()))
}

/// The strongly active model variables outside `rows`. Their `Σ` stays in
/// the held factor and drives the coupling through them to zero as `μ`
/// falls, so a block's numbers are conditional on those bounds rather than
/// marginal over them.
///
/// Each candidate is classified as a singleton block: one back-solve gives
/// `(K⁻¹)_ii`, the reduced curvature is `|1/(K⁻¹)_ii − Σ_i|`, and the ratio
/// bands decide. A candidate first has to clear `Σ > √μ`, so only near-bound
/// variables pay the back-solve. Returns full-x indices in order.
pub fn conditioned_on<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    rows: &[usize],
) -> Result<Vec<usize>, StatsError> {
    let cx = Context::new(conv, Declarations::default())?;
    conditioned_on_with(&cx, rows)
}

fn conditioned_on_with<E: BarrierIterate + ?Sized>(
    cx: &Context<'_, '_, E>,
    rows: &[usize],
) -> Result<Vec<usize>, StatsError> {
    let act = cx.act()?;
    let mu = act.mu;
    let pre = if mu > 0.0 { mu.sqrt() } else { 0.0 };
    let inside: std::collections::HashSet<usize> = rows.iter().copied().collect();
    let mut out = Vec::new();
    for (idx, &st) in act.var_status.iter().enumerate() {
        if inside.contains(&idx)
            || st == UNBOUNDED
            || st == FIXED
            || st == EQUALITY
            || st == INACTIVE
        {
            continue;
        }
        let sig = act.var_sigma[idx];
        let pinned = if st == STRONGLY_ACTIVE {
            true
        } else if sig <= pre {
            continue;
        } else {
            let krow = cx.primal_row(idx, RemovedAt::ConditionedOn)?;
            let kii = cx.unit_columns(&[krow])?[0][krow];
            if kii == 0.0 {
                continue;
            }
            let q_red = (1.0 / kii - sig).abs();
            // clamped to the floor rather than refused, as the block-level
            // rule does
            let floor = Number::EPSILON.sqrt() * (1.0 / kii).abs().max(1.0);
            classify_ratio(sig / q_red.max(floor), mu) == STRONGLY_ACTIVE
        };
        if pinned {
            out.push(idx);
        }
    }
    Ok(out)
}

/// The block and its factor rows, with the inverse-KKT columns over it.
struct BlockSetup {
    rows: Vec<usize>,
    zcols: Vec<Vec<Number>>,
    m: Dense,
}

fn setup_block<E: BarrierIterate + ?Sized>(
    cx: &Context<'_, '_, E>,
    block: Block<'_>,
) -> Result<BlockSetup, StatsError> {
    let rows: Vec<usize> = match block {
        Some(b) => b.to_vec(),
        None => {
            if cx.decl.fit_rows.is_empty() {
                return Err(StatsError::NoBlock);
            }
            cx.decl.fit_rows.to_vec()
        }
    };
    // `rows` is full-x, the space of the activity report and the row
    // normals, and `krows` the same variables as factor rows. They differ
    // exactly when the model has a removed variable.
    let krows: Vec<usize> = rows
        .iter()
        .enumerate()
        .map(|(k, &r)| cx.primal_row(r, RemovedAt::Member(k)))
        .collect::<Result<_, _>>()?;
    let zcols = cx.unit_columns(&krows)?;
    let m = block_of(&zcols, &krows);
    Ok(BlockSetup { rows, zcols, m })
}

fn perturbation_finding<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    findings: &mut Vec<StatsFinding>,
) {
    if let Some(p) = conv.kkt_perturbations()
        && p.iter().any(|&v| v != 0.0)
    {
        findings.push(StatsFinding::Perturbed { perturbations: p });
    }
}

/// The residual Jacobians `d r_g / d p` over the whole block, one per
/// group, recovered from the same back-solves: the residual rows of the
/// inverse-KKT columns are `J M`, so `J = Z_r M⁻¹` exactly and the factor's
/// barrier weight cancels whatever `Σ` is.
fn group_jacobians<E: BarrierIterate + ?Sized>(
    cx: &Context<'_, '_, E>,
    setup: &BlockSetup,
) -> Result<Vec<Dense>, StatsError> {
    let mi = minv(&setup.m)?;
    let n_params = setup.rows.len();
    cx.decl
        .res_groups
        .iter()
        .map(|rws| {
            let rrows: Vec<usize> = rws
                .iter()
                .map(|&r| cx.primal_row(r, RemovedAt::Residual))
                .collect::<Result<_, _>>()?;
            let zr = Dense::from_fn(rrows.len(), n_params, |i, j| setup.zcols[j][rrows[i]]);
            Ok(zr.mul(&mi))
        })
        .collect()
}

/// Asymptotic covariance of a least-squares estimate, from one solve.
///
/// Assumes the objective is the plain sum of squared residuals. The block
/// of the inverse KKT matrix, one back-solve per member, equals the inverse
/// reduced Hessian `inv(∂²f*/∂p²)`, and for `f = SSR` the covariance is
/// `2 σ² (K⁻¹)_pp`. The noise variance comes from the first of these that
/// is given:
///
/// - `known`, one value per declared residual group in their order, or a
///   single value when none were declared
/// - an estimate per declared residual group, `SSR_g / (n_g − n_fit)`
/// - the solve-time objective value over `n_data − n_fit`
///
/// With several groups of unequal variance the heteroscedastic sandwich is
/// reported.
///
/// [`HessianForm::Lagrangian`] inverts the exact reduced Hessian, the
/// observed information. [`HessianForm::GaussNewton`] rebuilds the expected
/// information from the residual Jacobian, recovered from the same
/// back-solves. They agree for linear models.
///
/// Bound and row activity is classified from the solve's own barrier
/// geometry. A strongly active bound pins its member, which gets zero
/// variance. A weakly active one is kept with its variance corrected for
/// the barrier weight the factor carries. A strongly active row involving
/// the block projects the matrix onto its null space. An explicit block
/// that is rank-deficient gets the homoscedastic Lagrangian marginal
/// `2 σ² M` without the membership handling.
///
/// `max_pdpert` refuses when the factor carries a larger inertia correction,
/// or when the engine does not report its correction at all.
#[allow(clippy::too_many_arguments)]
pub fn covariance<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    decl: Declarations<'_>,
    rec: SolveRecord<'_>,
    block: Block<'_>,
    known: Option<&[Number]>,
    n_data: Option<Number>,
    hessian: HessianForm,
    max_pdpert: Option<Number>,
) -> Outcome<Covariance> {
    let mut findings = Vec::new();
    let result = covariance_inner(
        conv,
        decl,
        rec,
        block,
        known,
        n_data,
        hessian,
        max_pdpert,
        &mut findings,
    );
    Outcome { findings, result }
}

#[allow(clippy::too_many_arguments)]
fn covariance_inner<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    decl: Declarations<'_>,
    rec: SolveRecord<'_>,
    block: Block<'_>,
    known: Option<&[Number]>,
    n_data: Option<Number>,
    hessian: HessianForm,
    max_pdpert: Option<Number>,
    findings: &mut Vec<StatsFinding>,
) -> Result<Covariance, StatsError> {
    let explicit = block.is_some();
    let n_fit = decl.fit_rows.len();
    if block.is_none() && decl.fit_rows.is_empty() {
        return Err(StatsError::NoBlock);
    }
    check_margins(None, max_pdpert)?;
    refuse_on_pdpert(conv, max_pdpert)?;
    perturbation_finding(conv, findings);
    let cx = Context::new(conv, decl)?;
    let setup = setup_block(&cx, block)?;
    let n_params = setup.rows.len();

    // An explicit block with more coordinates than the fit has degrees of
    // freedom has a well-defined marginal covariance, 2 σ² M, and no
    // inverse to classify its members with. A singular M does not reliably
    // fail to invert, so the count and the rank are checked first.
    let mut deficient = None;
    if explicit {
        if n_params > cx.n_var - cx.n_eq {
            deficient = Some(Deficiency::Count);
        } else if dense::rank_deficient(&setup.m).map_err(StatsError::Numeric)? {
            deficient = Some(Deficiency::Dependent);
        }
    }
    let cb = if deficient.is_some() {
        None
    } else {
        match classify_block(&cx, &setup.rows, &setup.m, &setup.zcols, findings) {
            Ok(cb) => Some(cb),
            Err(StatsError::SingularBlock) if explicit => {
                deficient = Some(Deficiency::Dependent);
                None
            }
            Err(e) => return Err(e),
        }
    };

    // the noise variance per group
    let groups = decl.res_groups;
    if hessian == HessianForm::GaussNewton && groups.is_empty() {
        return Err(StatsError::GaussNewtonNeedsResiduals);
    }
    if n_data.is_some() && (known.is_some() || !groups.is_empty()) {
        findings.push(StatsFinding::NDataIgnored);
    }
    let group_sigma: Vec<Number> = if let Some(v) = known {
        v.to_vec()
    } else if !groups.is_empty() {
        if n_fit == 0 {
            return Err(StatsError::NoFitForResiduals);
        }
        if groups.iter().flatten().any(|&r| r >= rec.x.len()) {
            return Err(StatsError::NoSolvePoint);
        }
        let mut out = Vec::with_capacity(groups.len());
        for (g, rws) in groups.iter().enumerate() {
            let n_g = rws.len();
            if n_g <= n_fit {
                return Err(StatsError::GroupTooSmall {
                    group: g,
                    members: n_g,
                    n_fit,
                });
            }
            let ssr: Number = rws.iter().map(|&r| rec.x[r] * rec.x[r]).sum();
            out.push(ssr / (n_g - n_fit) as Number);
        }
        out
    } else if let Some(n_data) = n_data {
        if n_fit == 0 {
            return Err(StatsError::NDataNoFit);
        }
        if n_data <= n_fit as Number {
            return Err(StatsError::NDataTooSmall { n_data, n_fit });
        }
        // the objective value AT THE SOLVE, never re-evaluated: a modelling
        // layer reads its current values, so anything written after the
        // solve would rescale the covariance
        if !rec.obj.is_finite() {
            return Err(StatsError::NoObjectiveValue { obj: rec.obj });
        }
        // in the minimized sense: `maximize -SSR` is the same least-squares
        // problem spelled the other way round
        let ssr = rec.obj_sign * rec.obj;
        vec![ssr / (n_data - n_fit as Number)]
    } else {
        return Err(StatsError::NoiseUnknown);
    };

    // pooled when there is one group or all are equal to relative
    // tolerance, the heteroscedastic sandwich otherwise
    let homoscedastic = group_sigma.len() <= 1 || {
        let max = group_sigma
            .iter()
            .copied()
            .fold(Number::NEG_INFINITY, Number::max);
        let min = group_sigma
            .iter()
            .copied()
            .fold(Number::INFINITY, Number::min);
        let scale = group_sigma.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        max - min <= 1e-12 * scale
    };

    let Some(cb) = cb else {
        // the marginal-only path for a rank-deficient explicit block: the
        // Gauss-Newton and per-group routes profile Jacobians through
        // inv(M), so only the homoscedastic Lagrangian marginal exists
        let reason = deficient.expect("set whenever the classification was skipped");
        if hessian == HessianForm::GaussNewton || !homoscedastic {
            return Err(StatsError::DeficientNeedsLagrangian { reason });
        }
        let act = cx.act()?;
        let flagged: Vec<usize> = (0..n_params)
            .filter(|&i| {
                let st = act.var_status[setup.rows[i]];
                !(st == INACTIVE || st == UNBOUNDED || st == FIXED || st == EQUALITY)
            })
            .collect();
        if !flagged.is_empty() {
            findings.push(StatsFinding::DeficientMembers {
                members: flagged,
                reason,
            });
        }
        let cov = setup.m.scale(2.0 * group_sigma[0]).symmetrized();
        return Ok(Covariance {
            matrix: cov,
            sigma_sq: group_sigma,
            conditioned_on: conditioned_on_with(&cx, &setup.rows)?,
        });
    };

    // The covariance is computed over the free (off-bound) members and
    // embedded with zero rows and columns for the pinned ones: the
    // covariance conditional on the active set. The information matrix is
    // restricted and then inverted, not the inverse restricted.
    let free = &cb.free;
    let embed = |cov_ff: &Dense| -> Dense {
        if free.len() == n_params {
            return cov_ff.clone();
        }
        let mut full = Dense::zeros(n_params, n_params);
        for (a, &i) in free.iter().enumerate() {
            for (b, &j) in free.iter().enumerate() {
                full.set(i, j, cov_ff.at(a, b));
            }
        }
        full
    };
    let projected_inverse = |g: &Dense| -> Result<Option<Dense>, StatsError> {
        let zb = free_nullspace(&cb.bind_normals, free)?;
        if zb.cols == 0 {
            return Ok(Some(Dense::zeros(free.len(), free.len())));
        }
        Ok(dense::inv(&zb.t().mul(g).mul(&zb)).map(|i| zb.mul(&i).mul(&zb.t())))
    };
    let free_jacobians = || -> Result<Vec<Dense>, StatsError> {
        Ok(group_jacobians(&cx, &setup)?
            .into_iter()
            .map(|jg| {
                let rr: Vec<usize> = (0..jg.rows).collect();
                jg.select(&rr, free)
            })
            .collect())
    };

    let cov = if free.is_empty() {
        Dense::zeros(n_params, n_params)
    } else if hessian == HessianForm::GaussNewton {
        // H_GN = 2 JᵀJ in place of the exact reduced Hessian. Pooled:
        // cov = 2 s² inv(H_GN) = s² inv(JᵀJ). Grouped:
        // cov = inv(JᵀJ) (Σ_g s_g² J_gᵀ J_g) inv(JᵀJ).
        let js = free_jacobians()?;
        let mut g = Dense::zeros(free.len(), free.len());
        for jg in &js {
            g = g.add(&jg.t().mul(jg));
        }
        let ginv = projected_inverse(&g)?.ok_or(StatsError::GaussNewtonSingular)?;
        if homoscedastic {
            embed(&ginv.scale(group_sigma[0]))
        } else {
            let mut b = Dense::zeros(free.len(), free.len());
            for (jg, &s) in js.iter().zip(&group_sigma) {
                b = b.add(&jg.t().mul(jg).scale(s));
            }
            embed(&ginv.mul(&b).mul(&ginv))
        }
    } else {
        let all_free = free.len() == n_params;
        let quiet = cb.sig_fit.iter().fold(0.0_f64, |a, &b| a.max(b.abs())) <= cb.floor;
        let mc = if cb.bind_normals.is_empty() && cb.row_corrections.is_empty() && all_free && quiet
        {
            // nothing active and nothing to correct above noise scale: M is
            // already the answer, and the inv(inv(M)) round trip is spared
            setup.m.clone()
        } else {
            // R_corr carries item 1's value corrections: the members' own
            // barrier diagonal and the kept rows' barrier weight removed.
            // Pinned members are excluded by the free restriction and
            // binding directions annihilated by the projection basis.
            let rff = cb.r_corr.select(free, free);
            projected_inverse(&rff)?.ok_or(StatsError::FreeSingular)?
        };
        if homoscedastic {
            embed(&mc.scale(2.0 * group_sigma[0]))
        } else {
            // the heteroscedastic sandwich cov = A⁻¹ B A⁻¹ with A = ∂²f/∂p²,
            // which is 4 M (Σ_g σ_g² J_gᵀ J_g) M here
            let js = free_jacobians()?;
            let mut b = Dense::zeros(free.len(), free.len());
            for (jg, &s) in js.iter().zip(&group_sigma) {
                b = b.add(&jg.t().mul(jg).scale(s));
            }
            embed(&mc.mul(&b).mul(&mc).scale(4.0))
        }
    };
    let cov = cov.symmetrized();
    if cov
        .diagonal()
        .iter()
        .fold(Number::INFINITY, |a, &b| a.min(b))
        < 0.0
    {
        findings.push(StatsFinding::NegativeVariance);
    }
    Ok(Covariance {
        matrix: cov,
        sigma_sq: group_sigma,
        conditioned_on: conditioned_on_with(&cx, &setup.rows)?,
    })
}

/// The information matrix over the block: the reduced Hessian
/// [`covariance`] inverts, in natural units with no `σ²`, from the same
/// solve. For a homoscedastic Lagrangian fit the covariance equals
/// `2 σ² inv(information)` on the free block.
///
/// [`HessianForm::Lagrangian`] builds the observed information by tangent
/// recovery, which needs the square estimation structure.
/// [`HessianForm::GaussNewton`] is `2 JᵀJ` with `J` recovered over the
/// whole block. A strongly active member's entry is `S`, the reduction onto
/// the pinned set, not a zero row, and binding rows project the free block
/// on both sides. An indefinite Lagrangian block is returned with a
/// finding. A smaller explicit block gets its marginal by Schur complement
/// where it sits inside a square fitted set.
pub fn information<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    decl: Declarations<'_>,
    block: Block<'_>,
    hessian: HessianForm,
    max_pdpert: Option<Number>,
) -> Outcome<Information> {
    let mut findings = Vec::new();
    let result = information_inner(conv, decl, block, hessian, max_pdpert, &mut findings);
    Outcome { findings, result }
}

fn information_inner<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    decl: Declarations<'_>,
    block: Block<'_>,
    hessian: HessianForm,
    max_pdpert: Option<Number>,
    findings: &mut Vec<StatsFinding>,
) -> Result<Information, StatsError> {
    let explicit = block.is_some();
    if block.is_none() && decl.fit_rows.is_empty() {
        return Err(StatsError::NoBlock);
    }
    check_margins(None, max_pdpert)?;
    refuse_on_pdpert(conv, max_pdpert)?;
    perturbation_finding(conv, findings);
    let cx = Context::new(conv, decl)?;
    let setup = setup_block(&cx, block)?;
    let n_params = setup.rows.len();
    let dof = cx.n_var - cx.n_eq;

    if explicit {
        if n_params > dof {
            return Err(StatsError::InformationDeficient {
                reason: Deficiency::Count,
            });
        }
        if dense::rank_deficient(&setup.m).map_err(StatsError::Numeric)? {
            return Err(StatsError::InformationDeficient {
                reason: Deficiency::Dependent,
            });
        }
    }
    let cb = match classify_block(&cx, &setup.rows, &setup.m, &setup.zcols, findings) {
        Ok(cb) => cb,
        Err(StatsError::SingularBlock) if explicit => {
            return Err(StatsError::InformationDeficient {
                reason: Deficiency::Dependent,
            });
        }
        Err(e) => return Err(e),
    };
    let (free, active) = (&cb.free, &cb.active);

    let r = if hessian == HessianForm::GaussNewton {
        if decl.res_groups.is_empty() {
            return Err(StatsError::GaussNewtonNeedsResiduals);
        }
        // J over the whole block, sliced last, so the pinned rows exist for
        // S below
        let mut r = Dense::zeros(n_params, n_params);
        for jg in group_jacobians(&cx, &setup)? {
            r = r.add(&jg.t().mul(&jg).scale(2.0));
        }
        r
    } else if let Some(r) = cb.r_exact.clone() {
        r
    } else if dof == n_params {
        tangent_reduced_hessian(&cx, &setup.m, &setup.zcols)?
    } else if explicit {
        // a sub-block of the fitted set gets its marginal by Schur
        // complement of the exact tangent R, and any other explicit block
        // reduces off the factor with item 1's corrections
        match subblock_information(&cx, &setup.rows, dof, findings)? {
            Some(r) => r,
            None => cb.r_corr.clone(),
        }
    } else {
        return Err(StatsError::NotSquare {
            n_var: cx.n_var,
            n_eq: cx.n_eq,
            n_params,
        });
    };

    let mut info = Dense::zeros(n_params, n_params);
    if !free.is_empty() {
        let rff = r.select(free, free);
        let zb = free_nullspace(&cb.bind_normals, free)?;
        let info_ff = if zb.cols == rff.rows {
            rff
        } else {
            // projected on both sides: the pseudo-inverse of the projected
            // covariance
            zb.mul(&zb.t().mul(&rff).mul(&zb)).mul(&zb.t())
        };
        for (a, &i) in free.iter().enumerate() {
            for (b, &j) in free.iter().enumerate() {
                info.set(i, j, info_ff.at(a, b));
            }
        }
        if hessian == HessianForm::Lagrangian
            && dense::indefinite(&info_ff).map_err(StatsError::Numeric)?
        {
            findings.push(StatsFinding::Indefinite);
        }
    }
    if !active.is_empty() {
        // the pinned set's entry is S, the reduction onto the pinned block,
        // not a zero row: zero information is the opposite of what a pinned
        // member carries. Conditional on the rest of the pinned set.
        let s = if free.is_empty() {
            r.select(active, active)
        } else {
            // rank-gated before the solve, since whether a singular system
            // raises depends on the build, and diagonally scaled, so an
            // ill-scaled but well-determined free block keeps its answer
            let r_ff = r.select(free, free);
            if dense::rank_deficient(&r_ff).map_err(StatsError::Numeric)? {
                return Err(StatsError::PinnedUndefined);
            }
            let x =
                dense::solve(&r_ff, &r.select(free, active)).ok_or(StatsError::PinnedUndefined)?;
            r.select(active, active)
                .sub(&r.select(active, free).mul(&x))
        };
        for (a, &i) in active.iter().enumerate() {
            for (b, &j) in active.iter().enumerate() {
                info.set(i, j, s.at(a, b));
            }
        }
    }
    Ok(Information {
        matrix: info,
        conditioned_on: conditioned_on_with(&cx, &setup.rows)?,
    })
}

/// The standard errors and the correlation matrix of a covariance. An entry
/// whose scale is undefined, such as a member projected to zero variance,
/// reports correlation 0.
pub fn correlation(cov: &Dense) -> (Vec<Number>, Dense) {
    let se: Vec<Number> = cov.diagonal().iter().map(|v| v.sqrt()).collect();
    let corr = Dense::from_fn(cov.rows, cov.cols, |i, j| {
        let c = cov.at(i, j) / (se[i] * se[j]);
        if c.is_finite() { c } else { 0.0 }
    });
    (se, corr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ratio_classifies_by_the_bands_of_its_barrier_level() {
        // a loose solve keeps the fixed bands at 0.1 and 10
        assert_eq!(classify_ratio(0.05, 1e-3), INACTIVE);
        assert_eq!(classify_ratio(0.5, 1e-3), AMBIGUOUS);
        assert_eq!(classify_ratio(20.0, 1e-3), STRONGLY_ACTIVE);
        // a tight one moves the outer edges to √μ and 1/√μ
        assert_eq!(classify_ratio(1e-5, 1e-8), INACTIVE);
        assert_eq!(classify_ratio(2e4, 1e-8), STRONGLY_ACTIVE);
        assert_eq!(classify_ratio(5.0, 1e-8), WEAKLY_ACTIVE);
        assert_eq!(classify_ratio(50.0, 1e-8), AMBIGUOUS);
        assert_eq!(classify_ratio(1e-2, 1e-8), AMBIGUOUS);
    }

    #[test]
    fn an_undefined_correlation_is_zero() {
        let cov = Dense::from_fn(3, 3, |i, j| {
            [[4.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]][i][j]
        });
        let (se, corr) = correlation(&cov);
        assert_eq!(se, vec![2.0, 0.0, 1.0]);
        assert_eq!(corr.at(0, 2), 0.5);
        assert_eq!(corr.at(0, 1), 0.0);
        assert_eq!(corr.at(1, 1), 0.0);
        assert_eq!(corr.at(2, 2), 1.0);
    }
}
