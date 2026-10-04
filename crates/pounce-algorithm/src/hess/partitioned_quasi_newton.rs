//! Partitioned quasi-Newton Hessian — a **per-constraint element**
//! updater (Griewank & Toint partitioned updating; Asprion, Chinellato &
//! Guzzella, *J. Appl. Math.* 2014, doi:10.1155/2014/341716, applied it to
//! direct-collocation trajectory optimization).
//!
//! # Why this exists
//!
//! [`crate::hess::lim_mem_quasi_newton::LimMemQuasiNewtonUpdater`]
//! approximates the whole Lagrangian Hessian with `m` curvature pairs and
//! publishes it as a [`pounce_linalg::low_rank_update_sym_matrix::LowRankUpdateSymMatrix`],
//! which [`crate::kkt::low_rank_aug_system_solver::LowRankAugSystemSolver`]
//! applies by Sherman-Morrison-Woodbury. Two consequences:
//!
//! * the `(1,1)` block the sparse factorization sees is **diagonal**, so
//!   whatever block structure the model's Hessian has is invisible to the
//!   linear solver's ordering — and the Schur path
//!   ([`crate::kkt::SchurAugSystemSolver`]) is bypassed entirely
//!   (`alg_builder.rs`, the `is_lbfgs` branch);
//! * `m` pairs cannot represent the curvature of a 60 000-variable
//!   collocation model, and the iteration count shows it.
//!
//! This updater takes the other route. The Lagrangian is a sum of
//! *element functions* with small support,
//!
//! ```text
//!     L(x, y) = f(x) + Σ_j (y_c)_j c_j(x) + Σ_j (y_d)_j d_j(x)
//! ```
//!
//! so its Hessian is the assembly of the elements' own Hessians:
//!
//! ```text
//!     ∇²L = ∇²f + Σ_j (y_c)_j ∇²c_j + Σ_j (y_d)_j ∇²d_j
//! ```
//!
//! We keep one small dense symmetric `B_e` per element, update each from
//! that element's **own** curvature pair, and scatter-add the weighted
//! blocks into a [`SymTMatrix`] — the same type the exact-Hessian path
//! publishes, so the whole downstream KKT machinery is unchanged and the
//! factorization sees the true block-banded pattern.
//!
//! # Why per-constraint, and not per-primal-block
//!
//! Asprion et al. partition the *Lagrangian* by primal stage blocks.
//! That is fewer, larger elements, but each element's target moves as the
//! multipliers move. Splitting per constraint row instead gives each
//! `B_e` a **multiplier-independent** target — `∇²c_j` is a property of
//! the model, not of the iterate — so the approximation converges instead
//! of chasing `y`. It also needs no new user-facing structure hook: an
//! element's support is a row of the constraint Jacobian, whose pattern
//! every TNLP already declares.
//!
//! # Why SR1 is the default here
//!
//! An individual constraint is not convex, so `sᵀy > 0` fails routinely
//! and Powell-damped BFGS would force each `∇²c_j` model positive
//! semidefinite — then multiply it by a multiplier of either sign. SR1
//! carries the sign, and the indefiniteness reaches the IPM's inertia
//! check, which is exactly what
//! `dev-notes/issue-131-monotone-lbfgs-stall.md` records the damped path
//! hiding. [`UpdateType::Bfgs`] is accepted for comparison.
//!
//! # Elements with a declared, incomplete pattern
//!
//! When the model declares its Hessian structure, an element keeps only
//! the declared entries of its support (`Element::allowed`), and neither
//! formula above is used on it: a rank-one or rank-two term fills every
//! position, including the ones the model says are structurally zero.
//! Such an element takes [`sparse_secant_update`] (Toint's minimum-change
//! symmetric secant update restricted to the pattern) from `B = 0`, with
//! no `γI` seeding. On a collocation row the declared group is a star —
//! `k` free entries on `k` coordinates — so one pair determines the
//! element exactly. gh#989 item 5: the dense updates' spurious entries,
//! weighted by multipliers of either sign, gave the assembled `W` the wrong
//! inertia on a Radau batch reactor, and the IPM spent the solve on a
//! `δ_w` decaying by `1/3` per iteration (16/28/86 iterations at
//! N = 25/100/400 against exact 12/16/18; `bfgs` at the cap). With the
//! pattern kept: 13/15/21, `bfgs` and `blocks` the same.
//!
//! # Bounded element size
//!
//! An element with `k` nonzeros costs `k(k+1)/2` stored reals. A row that
//! touches most of `x` — a global resource constraint, or an objective
//! that sums over every stage — would blow that up, so elements wider
//! than `max_element` degrade to a **diagonal** approximation
//! (Dennis-Wolkowicz weak secant, `sᵀBs = sᵀy`) rather than being
//! dropped: a separable objective is then still represented exactly, and
//! a coupled one is represented approximately instead of not at all.

use crate::hess::lim_mem_quasi_newton::UpdateType;
use crate::hess::r#trait::HessianUpdater;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};
use pounce_linalg::Vector;
use pounce_linalg::compound_vector::CompoundVector;
use pounce_linalg::dense_vector::DenseVector;
use pounce_linalg::triplet::{GenTMatrix, SymTMatrix, SymTMatrixSpace};
use std::rc::Rc;

/// Relative safeguard on the SR1 denominator: the update is skipped when
/// `|wᵀs| ≤ SR1_SAFEGUARD · ‖s‖ · ‖w‖` (Nocedal & Wright §6.2, eq. 6.26,
/// which suggests `r ∈ [1e-8, 1e-4]`).
///
/// This is a *direction* test only — it rejects a pair carrying no usable
/// curvature along `s`. It is deliberately **not** the magnitude control,
/// and trying to make it one was measured to fail from both ends. The
/// rank-1 term `w wᵀ / wᵀs` is bounded only by `‖w‖ / (r ‖s‖)`, so at
/// `r = 1e-8` it permits a correction `1e8` times the curvature the data
/// implies — on `benchmarks/large_scale` `laptime` the implied element
/// curvature `‖y_e‖/‖s_e‖` stayed a healthy 9–33 every iteration while
/// single-update block changes reached `1.7e8` and the assembled `W` ran
/// four orders of magnitude over the exact Lagrangian Hessian. But
/// tightening it to `1e-4` overshoots just as badly in the other
/// direction: element supports are small (`k ≈ 9` here) and `w` is
/// routinely near-orthogonal to `s`, so the test then rejected 9 949 of
/// 9 950 updates, the blocks never learned anything, and `W` read `0.14`
/// where the exact Hessian read `2.1e3`. [`DEFAULT_CURVATURE_CAP`] is
/// where the magnitude is bounded, in the units of the thing being
/// modelled; this stays loose so that pairs still reach it.
const SR1_SAFEGUARD: Number = 1e-8;

/// Powell damping threshold for [`UpdateType::Bfgs`], matching the
/// hard-coded `0.2` of the limited-memory path.
const POWELL_THETA: Number = 0.2;

/// Relative floor on the BFGS denominators `sᵀr` and `sᵀBs`. The
/// limited-memory path tests only `> 0`, which is safe there because its
/// `s` is a whole primal step; restricted to one element's support the
/// same quantity goes arbitrarily small, and `r rᵀ / sᵀr` then blows up
/// exactly as the SR1 term does. Same measurement as
/// [`SR1_SAFEGUARD`]: unfloored, damped BFGS reached block changes of
/// `2.9e8` — worse than SR1, because `> 0` is no floor at all.
const BFGS_DENOM_FLOOR: Number = 1e-8;

/// Cap on a single update's magnitude, as a multiple of the curvature the
/// element's own secant pair implies (`‖y_e‖ / ‖s_e‖`).
///
/// **Off by default, because every finite value measured was worse than
/// off.** The idea was to bound the update in the units of the quantity
/// being modelled, since the relative denominator floors bound only the
/// ratio that produced the term. It does bound it — and it makes the
/// solver worse, non-monotonically. On `benchmarks/large_scale`
/// `laptime` at `N = 80`, `max_iter = 1200` (true optimum 65.462928):
///
/// | cap | status | iters | wall | objective |
/// |---|---|---|---|---|
/// | 1e1 | ErrorInStepComputation | 1071 | 179 s | 65.518586 |
/// | 1e2 | MaxIter | 1200 | 214 s | 67.202124 |
/// | 1e6 | MaxIter | 1200 | 232 s | 80.398129 |
/// | off | **Optimal** | 559 | 50 s | 65.462802 |
///
/// `1e6` being worse than both `1e1` and off is the shape of the result:
/// this is not "less capping is better". Rejection is *selective* — it
/// drops precisely the elements whose curvature is moving fastest,
/// leaving those blocks stale while their neighbours update. The
/// assembled `W` is then internally inconsistent, which costs more than a
/// uniformly noisy but coherent model.
///
/// Kept as a knob rather than deleted so the non-monotonicity can be
/// re-measured against a different element decomposition, where it may
/// behave differently. Do not turn it on without measuring.
const DEFAULT_CURVATURE_CAP: Number = Number::INFINITY;

/// Which gradient an element reads to form its curvature pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElementSource {
    /// The objective: gradient entries come from `grad_f` directly.
    Objective,
    /// A row of `J_c`; entries come from the equality Jacobian's values.
    EqRow,
    /// A row of `J_d`; entries come from the inequality Jacobian's values.
    IneqRow,
    /// A contiguous block of primal variables under
    /// [`ElementMode::PrimalBlock`]. The element function is the
    /// **Lagrangian itself**, restricted to the block, so its gradient is
    /// `∇f + J_cᵀ y_c + J_dᵀ y_d` restricted, and its assembly weight is
    /// 1 — the multiplier is already inside the function being modelled.
    LagrangianBlock,
}

/// How the Lagrangian is split into elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementMode {
    /// One element per constraint row (plus the objective). Each block
    /// has a multiplier-independent target, and needs no assumption about
    /// how the model orders its variables — but there are as many blocks
    /// as constraints, each approximating a `∇²c_j` with no sign
    /// structure, and the errors accumulate through the weighted sum.
    PerConstraint,
    /// One element per contiguous block of primal variables, modelling
    /// the Lagrangian's restriction to that block with damped BFGS.
    /// This is Asprion, Chinellato & Guzzella's partition: a direct
    /// collocation transcription orders its variables by stage, so the
    /// Lagrangian Hessian really is close to block diagonal in this
    /// partition, and the block count is the stage count rather than the
    /// constraint count.
    PrimalBlock,
}

/// One element function's quasi-Newton state.
#[derive(Debug)]
struct Element {
    source: ElementSource,
    /// 0-based row index within the element's source block, i.e. the
    /// position in `y_c` / `y_d` whose multiplier weights this element.
    /// Stored rather than inferred from the element's position in the
    /// table: a constraint row with no Jacobian entries produces no
    /// element, so counting elements and counting multipliers diverge
    /// the moment a model has one.
    row: u32,
    /// 0-based var-x columns this element touches, ascending and
    /// deduplicated. `k = support.len()`.
    support: Vec<Index>,
    /// `true` when `b` holds the packed lower triangle (`k(k+1)/2`
    /// reals); `false` when the element degraded to a diagonal
    /// approximation (`k` reals).
    dense: bool,
    /// `B_e`. Packed lower triangle in local coordinates — entry
    /// `(a, c)` with `a >= c` at `a(a+1)/2 + c` — or the diagonal alone
    /// when `!dense`.
    b: Vec<Number>,
    /// This element's gradient at the previous iterate, local coords.
    prev_g: Vec<Number>,
    /// Whether `prev_g` has been written at least once.
    has_prev: bool,
    /// Whether `b` has taken an accepted curvature pair (drives the
    /// one-time scalar seeding).
    seeded: bool,
    /// `(position in the source Jacobian's triplet arrays, local index)`
    /// for every triplet belonging to this element's row. Empty for
    /// [`ElementSource::Objective`], which reads `grad_f` by support
    /// index. Duplicated `(row, col)` triplets are summed, matching the
    /// triplet contract.
    entries: Vec<(u32, u32)>,
    /// Position in the assembled matrix's value array for each entry of
    /// `b`, in the same packing. `u32::MAX` marks a packed position the
    /// element's declared pattern excludes (never assembled).
    map: Vec<u32>,
    /// Which packed positions of `b` the model's declared Hessian pattern
    /// allows, when the element came from [`PartitionedQuasiNewtonUpdater::push_split`]
    /// and that pattern is **not** complete on the element. `None` means
    /// every position is free (the dense formulas apply). When `Some`, the
    /// element is updated with the pattern-constrained secant update
    /// ([`sparse_secant_update`]) and the excluded entries stay exactly
    /// zero (gh#989 item 5).
    allowed: Option<Vec<bool>>,
}

impl Element {
    fn k(&self) -> usize {
        self.support.len()
    }
}

/// One-time structural census, for the measurement write-up and for
/// `POUNCE_PARTITIONED_DEBUG`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PartitionStats {
    pub elements: usize,
    pub dense_elements: usize,
    pub diagonal_elements: usize,
    pub max_support: usize,
    pub total_support: usize,
    /// Nonzeros in the assembled lower triangle, diagonal included.
    pub assembled_nnz: usize,
    /// Reals held across all `B_e`.
    pub stored_reals: usize,
}

pub struct PartitionedQuasiNewtonUpdater {
    pub update_type: UpdateType,
    /// How the Lagrangian is split. See [`ElementMode`].
    pub mode: ElementMode,
    /// Target width of a primal block under [`ElementMode::PrimalBlock`]
    /// (`partitioned_block_size`).
    pub block_size: usize,
    /// Widest element that keeps a dense block; wider ones degrade to a
    /// diagonal approximation. Option `partitioned_max_element`.
    pub max_element: usize,
    /// Floor placed on assembled diagonal entries no active element
    /// covers, so the `(1, 1)` block never presents a structurally empty
    /// row to the factorization. Shares
    /// `limited_memory_init_val_min`'s default and its rationale — see
    /// the gh#624 discussion in
    /// [`crate::hess::lim_mem_quasi_newton`].
    pub init_val_min: Number,
    /// Clamp on the magnitude of the per-element scalar seeding.
    pub init_val_max: Number,
    /// Multiple of the identity published on the **first** iteration,
    /// before any element has seen a curvature pair. Mirrors
    /// `limited_memory_init_val` and the empty-history branch of the
    /// limited-memory updater: publishing the honest all-zero `W` there
    /// instead hands the very first KKT solve a `(1,1)` block with no
    /// curvature at all, and the whole trajectory is set by whatever
    /// `delta_x` the inertia correction then has to invent.
    pub init_val: Number,
    /// Support of the objective element, in the compressed `x_var`
    /// space, when the TNLP could state it
    /// (`TNLPAdapter::objective_nonlinear_vars`). `None` falls back to
    /// the first `∇f`'s nonzeros — see `build_structure`.
    pub objective_vars: Option<Vec<Index>>,
    /// Multiple of the implied curvature `‖y_e‖/‖s_e‖` that a single
    /// update to an element block may reach
    /// (`partitioned_curvature_cap`). See [`DEFAULT_CURVATURE_CAP`].
    pub curvature_cap: Number,
    /// The TNLP's declared Lagrangian Hessian sparsity, `(row >= col)` in the
    /// compressed variable space. When present, each per-constraint (and
    /// objective) element is split into the connected groups of that pattern
    /// restricted to its support, and coordinates the pattern never couples
    /// are dropped. See [`Self::push_split`].
    pub declared_pattern: Option<Vec<(Index, Index)>>,

    /// Element table, built on the first call and fixed thereafter.
    elements: Vec<Element>,
    /// Assembled pattern, built with the element table.
    space: Option<Rc<SymTMatrixSpace>>,
    /// Assembled positions of the `n` diagonal entries.
    diag_pos: Vec<u32>,
    /// Coordinates no active element covers, which take `init_val_min`.
    uncovered: Vec<Index>,
    /// `x` at the previous call.
    prev_x: Option<Vec<Number>>,
    /// `∇f`, and the Jacobians' triplet values, at the previous call.
    /// [`ElementMode::PrimalBlock`] needs the change in the *Lagrangian*
    /// gradient, which is not a per-element quantity, so it is formed
    /// once per call from these rather than element by element.
    prev_grad_f: Option<Vec<Number>>,
    prev_jac_c: Option<Vec<Number>>,
    prev_jac_d: Option<Vec<Number>>,
    stats: PartitionStats,
    /// One-shot latch for `POUNCE_HESS_PATTERN_CENSUS`.
    census_done: bool,
    /// Curvature pairs accepted / skipped, cumulative — a cheap health
    /// signal for the write-up.
    pub accepted_updates: u64,
    pub skipped_updates: u64,
    /// Per-call diagnostics for `POUNCE_PARTITIONED_ORACLE`: the element
    /// with the largest implied curvature `‖y_e‖ / ‖s_e‖`, and the
    /// largest single-update change to any block.
    dbg: DebugPeaks,
}

impl PartitionedQuasiNewtonUpdater {
    pub fn new(update_type: UpdateType) -> Self {
        Self {
            update_type,
            mode: ElementMode::PerConstraint,
            block_size: 64,
            max_element: 64,
            init_val_min: 1e-8,
            init_val_max: 1e8,
            init_val: 1.0,
            objective_vars: None,
            curvature_cap: DEFAULT_CURVATURE_CAP,
            declared_pattern: None,
            elements: Vec::new(),
            space: None,
            diag_pos: Vec::new(),
            uncovered: Vec::new(),
            prev_x: None,
            prev_grad_f: None,
            prev_jac_c: None,
            prev_jac_d: None,
            stats: PartitionStats::default(),
            census_done: false,
            accepted_updates: 0,
            skipped_updates: 0,
            dbg: DebugPeaks::default(),
        }
    }

    pub fn stats(&self) -> PartitionStats {
        self.stats
    }

    /// Build the element table and the assembled sparsity pattern. Runs
    /// once; both are structural and every later call reuses them, so
    /// the backend's symbolic factorization is done a single time.
    fn build_structure(
        &mut self,
        n: usize,
        grad_f: &[Number],
        jac_c: &GenTMatrix,
        jac_d: &GenTMatrix,
    ) {
        let mut elements: Vec<Element> = Vec::new();

        if self.mode == ElementMode::PrimalBlock {
            // Contiguous blocks in the model's own variable order.
            //
            // The ordering assumption is load-bearing and is *checked*
            // rather than trusted: `POUNCE_PARTITIONED_ORACLE` reports the
            // fraction of the exact Hessian's Frobenius mass that falls
            // inside the block-diagonal pattern. A transcription that
            // orders by stage — which is what every direct-collocation
            // writer does, and what `laptime` does — puts nearly all of it
            // inside. One that does not will show a low fraction, and the
            // partition is then simply wrong for that model rather than
            // silently poor.
            let bs = self.block_size.max(1);
            if let Some(pat) = self.declared_pattern.as_ref() {
                // gh#989 item 5: with a declared Hessian pattern the blocks
                // are not guessed from the variable order — they are the
                // connected components of that pattern, which are exactly
                // the Lagrangian Hessian's diagonal blocks (no declared entry
                // crosses a component, so the block-diagonal model has no
                // truncation error at all). A writer that does not order by
                // stage — an AMPL `.nl`, which puts nonlinear variables
                // first, is one — no longer breaks the partition: on the
                // Radau batch reactor at N = 400 contiguous blocks of 64 hit
                // the iteration cap. A variable no declared entry mentions is
                // linear in the Lagrangian and belongs to no block.
                //
                // Only when EVERY component fits in `partitioned_block_size`.
                // Chunking a wide component cuts declared coupling, and that
                // was measured to be worse than the contiguous partition:
                // `laptime` at N = 80 is one 3 280-variable component, and
                // chunking it hit the iteration cap where contiguous blocks
                // take 166. Such a model keeps the contiguous partition.
                let mut parent: Vec<usize> = (0..n).collect();
                fn root(parent: &mut [usize], mut a: usize) -> usize {
                    while parent[a] != a {
                        parent[a] = parent[parent[a]];
                        a = parent[a];
                    }
                    a
                }
                let mut mentioned = vec![false; n];
                let mut adj: std::collections::HashMap<Index, Vec<Index>> = Default::default();
                for &(r, c) in pat {
                    let (ru, cu) = (r as usize, c as usize);
                    if ru >= n || cu >= n {
                        continue;
                    }
                    mentioned[ru] = true;
                    mentioned[cu] = true;
                    adj.entry(r).or_default().push(c);
                    if r != c {
                        adj.entry(c).or_default().push(r);
                    }
                    let (a, b) = (root(&mut parent, ru), root(&mut parent, cu));
                    if a != b {
                        parent[a.max(b)] = a.min(b);
                    }
                }
                let mut comp_of_root: std::collections::HashMap<usize, usize> = Default::default();
                let mut comps: Vec<Vec<Index>> = Vec::new();
                for (i, _) in mentioned.iter().enumerate().filter(|(_, m)| **m) {
                    let r = root(&mut parent, i);
                    let g = *comp_of_root.entry(r).or_insert_with(|| {
                        comps.push(Vec::new());
                        comps.len() - 1
                    });
                    comps[g].push(i as Index);
                }
                if comps.iter().all(|c| c.len() <= bs) {
                    for comp in comps {
                        let mut el = Self::make_element(
                            ElementSource::LagrangianBlock,
                            0,
                            comp,
                            Vec::new(),
                            usize::MAX,
                        );
                        el.allowed = pattern_mask(&el.support, &adj);
                        elements.push(el);
                    }
                    self.finish_structure(n, elements);
                    return;
                }
            }
            let mut start = 0usize;
            while start < n {
                let end = (start + bs).min(n);
                let support: Vec<Index> = (start..end).map(|i| i as Index).collect();
                elements.push(Self::make_element(
                    ElementSource::LagrangianBlock,
                    0,
                    support,
                    Vec::new(),
                    usize::MAX,
                ));
                start = end;
            }
            self.finish_structure(n, elements);
            return;
        }

        // Declared-pattern adjacency (gh#989 item 5): coordinate -> every
        // coordinate it is coupled to (itself included when the diagonal is
        // declared), from the lower-triangle pairs.
        let adjacency: Option<std::collections::HashMap<Index, Vec<Index>>> =
            self.declared_pattern.as_ref().map(|pat| {
                let mut m: std::collections::HashMap<Index, Vec<Index>> = Default::default();
                for &(r, c) in pat {
                    m.entry(r).or_default().push(c);
                    if r != c {
                        m.entry(c).or_default().push(r);
                    }
                }
                m
            });
        // ---- objective element -------------------------------------
        //
        // Every constraint element takes its support from a row of the
        // Jacobian, whose pattern the TNLP is obliged to declare. The
        // objective has no such declaration, so the support comes from
        // `get_objective_variables_linearity` via
        // `TNLPAdapter::objective_nonlinear_vars` — the variables the
        // objective is *nonlinear* in, which is exactly the rows `∇²f`
        // can occupy.
        //
        // When the TNLP declines, the fallback is the first `∇f`'s
        // nonzeros, and that pattern is *value-derived*: a coordinate
        // whose `∂f/∂x_i` happens to vanish at the starting point is
        // excluded for the whole solve. On `laptime`, which declares 321
        // objective gradient nonzeros, the fallback captures 161 — so
        // this is the live case, not a corner. Widening to all of `x` is
        // not the alternative: that is an `n × n` element.
        let obj_support: Vec<Index> = match self.objective_vars.clone() {
            Some(v) => v,
            None => (0..n)
                .filter(|&i| grad_f[i] != 0.0)
                .map(|i| i as Index)
                .collect(),
        };
        if !obj_support.is_empty() {
            let mut sup = obj_support;
            sup.sort_unstable();
            sup.dedup();
            self.push_split(
                &mut elements,
                adjacency.as_ref(),
                ElementSource::Objective,
                0,
                sup,
                Vec::new(),
            );
        }

        // ---- one element per constraint row -------------------------
        for (source, jac) in [
            (ElementSource::EqRow, jac_c),
            (ElementSource::IneqRow, jac_d),
        ] {
            let n_rows = jac.space().n_rows() as usize;
            let irows = jac.irows();
            let jcols = jac.jcols();
            // Bucket triplet positions by row. `irows` is 1-based.
            let mut row_counts = vec![0u32; n_rows + 1];
            for &i in irows {
                row_counts[i as usize] += 1;
            }
            let mut row_start = vec![0u32; n_rows + 2];
            for r in 0..=n_rows {
                row_start[r + 1] = row_start[r] + row_counts[r];
            }
            let mut cursor = row_start.clone();
            let mut by_row = vec![0u32; irows.len()];
            for (pos, &i) in irows.iter().enumerate() {
                let r = i as usize;
                by_row[cursor[r] as usize] = pos as u32;
                cursor[r] += 1;
            }

            for r in 1..=n_rows {
                let slice = &by_row[row_start[r] as usize..row_start[r + 1] as usize];
                if slice.is_empty() {
                    continue;
                }
                // Support = sorted unique columns of this row (0-based).
                let mut cols: Vec<Index> = slice.iter().map(|&p| jcols[p as usize] - 1).collect();
                cols.sort_unstable();
                cols.dedup();
                // Local index of every triplet position in the row.
                // Duplicated `(row, col)` triplets land on the same local
                // index and are summed when the gradient is read.
                let entries: Vec<(u32, u32)> = slice
                    .iter()
                    .map(|&p| {
                        let c = jcols[p as usize] - 1;
                        let local = cols.partition_point(|&x| x < c) as u32;
                        (p, local)
                    })
                    .collect();
                self.push_split(
                    &mut elements,
                    adjacency.as_ref(),
                    source,
                    (r - 1) as u32,
                    cols,
                    entries,
                );
            }
        }

        self.finish_structure(n, elements);
    }

    /// Build the assembled pattern, the per-element scatter maps and the
    /// census from a finished element table. Shared by both
    /// [`ElementMode`]s.
    fn finish_structure(&mut self, n: usize, mut elements: Vec<Element>) {
        // ---- assembled pattern --------------------------------------
        //
        // Union of each active element's own lower triangle, plus the
        // full diagonal so no primal row of the `(1,1)` block is
        // structurally empty.
        let mut pairs: Vec<(Index, Index)> = Vec::new();
        for i in 0..n {
            pairs.push((i as Index, i as Index));
        }
        for e in &elements {
            if e.dense {
                for a in 0..e.k() {
                    for c in 0..=a {
                        if e.allowed.as_ref().is_none_or(|m| m[a * (a + 1) / 2 + c]) {
                            pairs.push((e.support[a], e.support[c]));
                        }
                    }
                }
            } else {
                for a in 0..e.k() {
                    pairs.push((e.support[a], e.support[a]));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();

        // Lookup from (row, col) to assembled position, by binary search
        // over the sorted pair list.
        let find =
            |row: Index, col: Index| -> u32 { pairs.partition_point(|&p| p < (row, col)) as u32 };
        for e in &mut elements {
            if e.dense {
                let mut map = Vec::with_capacity(e.k() * (e.k() + 1) / 2);
                for a in 0..e.k() {
                    for c in 0..=a {
                        // The full diagonal is always assembled, so a
                        // diagonal position maps even when the declared
                        // pattern excludes it (the BFGS convexification
                        // shift writes there).
                        let ok =
                            a == c || e.allowed.as_ref().is_none_or(|m| m[a * (a + 1) / 2 + c]);
                        map.push(if ok {
                            find(e.support[a], e.support[c])
                        } else {
                            u32::MAX
                        });
                    }
                }
                e.map = map;
            } else {
                e.map = (0..e.k())
                    .map(|a| find(e.support[a], e.support[a]))
                    .collect();
            }
        }
        self.diag_pos = (0..n).map(|i| find(i as Index, i as Index)).collect();

        let mut covered = vec![false; n];
        for e in &elements {
            for &i in &e.support {
                covered[i as usize] = true;
            }
        }
        self.uncovered = (0..n)
            .filter(|&i| !covered[i])
            .map(|i| i as Index)
            .collect();

        // 1-based triplets, lower triangle, matching the exact-Hessian
        // path's convention (`orig_ipopt_nlp.rs` pushes `i_var + 1`).
        let irows: Vec<Index> = pairs.iter().map(|&(r, _)| r + 1).collect();
        let jcols: Vec<Index> = pairs.iter().map(|&(_, c)| c + 1).collect();

        self.stats = PartitionStats {
            elements: elements.len(),
            dense_elements: elements.iter().filter(|e| e.dense).count(),
            diagonal_elements: elements.iter().filter(|e| !e.dense).count(),
            max_support: elements.iter().map(|e| e.k()).max().unwrap_or(0),
            total_support: elements.iter().map(|e| e.k()).sum(),
            assembled_nnz: pairs.len(),
            stored_reals: elements.iter().map(|e| e.b.len()).sum(),
        };
        self.space = Some(SymTMatrixSpace::new(n as Index, irows, jcols));
        self.elements = elements;

        if std::env::var("POUNCE_PARTITIONED_DEBUG").is_ok() {
            eprintln!("partitioned-qn: {:?}", self.stats);
        }
    }

    /// Push the element for `(source, row, support, entries)`, split along the
    /// declared Hessian pattern when one is available (gh#989 item 5).
    ///
    /// `∇²c_j` lives in `support × support`, and the declared Lagrangian
    /// pattern is a superset of it, so the connected components of the pattern
    /// restricted to `support` are independent blocks of `∇²c_j`:
    /// `y_comp = B_comp s_comp` holds exactly, one element per component is
    /// the same model without the spurious cross-component entries a single
    /// dense `k × k` block fills in, and a support coordinate the pattern never
    /// couples (a linear one) belongs to no component and is dropped.
    /// `entries` are `(Jacobian triplet position, local index)`; they are
    /// remapped to each component's own local index.
    fn push_split(
        &self,
        elements: &mut Vec<Element>,
        adjacency: Option<&std::collections::HashMap<Index, Vec<Index>>>,
        source: ElementSource,
        row: u32,
        support: Vec<Index>,
        entries: Vec<(u32, u32)>,
    ) {
        let Some(adj) = adjacency else {
            elements.push(Self::make_element(
                source,
                row,
                support,
                entries,
                self.max_element,
            ));
            return;
        };
        let k = support.len();
        // Union-find over local indices, joined by pattern pairs inside the
        // support; `mentioned[a]` marks a coordinate with any in-support entry.
        let mut parent: Vec<usize> = (0..k).collect();
        fn find(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        let mut mentioned = vec![false; k];
        for a in 0..k {
            if let Some(nbrs) = adj.get(&support[a]) {
                for &c in nbrs {
                    if let Ok(b) = support.binary_search(&c) {
                        mentioned[a] = true;
                        mentioned[b] = true;
                        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                        if ra != rb {
                            parent[ra.max(rb)] = ra.min(rb);
                        }
                    }
                }
            }
        }
        // Group by root, in ascending order of the first member.
        let mut group_of: Vec<Option<usize>> = vec![None; k];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut root_to_group: std::collections::HashMap<usize, usize> = Default::default();
        for a in 0..k {
            if !mentioned[a] {
                continue;
            }
            let r = find(&mut parent, a);
            let g = *root_to_group.entry(r).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(a);
            group_of[a] = Some(g);
        }
        let mut local_in_group = vec![0u32; k];
        for g in &groups {
            for (pos, &a) in g.iter().enumerate() {
                local_in_group[a] = pos as u32;
            }
        }
        let mut group_entries: Vec<Vec<(u32, u32)>> = vec![Vec::new(); groups.len()];
        for &(p, local) in &entries {
            if let Some(g) = group_of[local as usize] {
                group_entries[g].push((p, local_in_group[local as usize]));
            }
        }
        for (g, members) in groups.iter().enumerate() {
            let sub: Vec<Index> = members.iter().map(|&a| support[a]).collect();
            let ent = std::mem::take(&mut group_entries[g]);
            let mut el = Self::make_element(source, row, sub, ent, self.max_element);
            if el.dense {
                // The declared pattern restricted to this group. A connected
                // group is usually NOT complete — a collocation row's group
                // is a star (the control coupled to each state, no
                // state–state entries) — and a dense secant update fills the
                // missing entries with values the true `∇²c_j` does not have.
                el.allowed = pattern_mask(&el.support, adj);
            }
            elements.push(el);
        }
    }

    fn make_element(
        source: ElementSource,
        row: u32,
        support: Vec<Index>,
        entries: Vec<(u32, u32)>,
        max_element: usize,
    ) -> Element {
        let k = support.len();
        let dense = k <= max_element;
        let b_len = if dense { k * (k + 1) / 2 } else { k };
        Element {
            source,
            row,
            support,
            dense,
            b: vec![0.0; b_len],
            prev_g: vec![0.0; k],
            has_prev: false,
            seeded: false,
            entries,
            map: Vec::new(),
            allowed: None,
        }
    }
}

/// The declared pattern restricted to `support`, packed like an element's
/// `b` (`(a, c)`, `a >= c`, at `a(a+1)/2 + c`). `None` when the restriction
/// is complete, so the dense update formulas apply unchanged.
fn pattern_mask(
    support: &[Index],
    adj: &std::collections::HashMap<Index, Vec<Index>>,
) -> Option<Vec<bool>> {
    let k = support.len();
    let mut allowed = vec![false; k * (k + 1) / 2];
    let mut complete = true;
    for a in 0..k {
        let nbrs = adj.get(&support[a]);
        for c in 0..=a {
            let ok = nbrs.is_some_and(|v| v.contains(&support[c]));
            allowed[a * (a + 1) / 2 + c] = ok;
            complete &= ok;
        }
    }
    if complete { None } else { Some(allowed) }
}

#[derive(Debug, Clone, Copy, Default)]
struct DebugPeaks {
    ratio: Number,
    ratio_s: Number,
    ratio_y: Number,
    ratio_k: usize,
    delta: Number,
    step_norm: Number,
}

/// `out = B s` for a packed lower triangle.
fn packed_mult(b: &[Number], s: &[Number], out: &mut [Number]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    let mut p = 0usize;
    for a in 0..s.len() {
        for c in 0..=a {
            let v = b[p];
            p += 1;
            if v == 0.0 {
                continue;
            }
            out[a] += v * s[c];
            if c != a {
                out[c] += v * s[a];
            }
        }
    }
}

fn dot(a: &[Number], b: &[Number]) -> Number {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Pattern-constrained symmetric secant update (Toint, *Math. Comp.* 31,
/// 1977; the sparse analogue of PSB). Finds the correction `E` of least
/// Frobenius norm that is symmetric, vanishes outside `allowed`, and makes
/// `(B + E) s = y`. Writing `E_ac = λ_a s_c + λ_c s_a` on the allowed
/// positions, the secant condition is the `k × k` system
///
/// ```text
///   Q λ = r,   r = y − B s,
///   Q_aa = Σ_{c ∈ S_a} s_c² + [a ∈ S_a] s_a²,   Q_ac = s_a s_c  (c ∈ S_a, c ≠ a)
/// ```
///
/// `Q` is positive semidefinite; it is solved in the minimum-norm least
/// squares sense, so a row with no step component (`Q_aa = 0`) takes no
/// correction instead of a division by zero.
///
/// Why this and not SR1 on these elements (gh#989 item 5): a collocation
/// row's declared group is a *star* — the control coupled to each state,
/// no state–state entries — so the true `∇²c_j` has `k` free entries on a
/// `k`-coordinate support and one secant pair (`k` equations) determines
/// it. The rank-1 term instead writes `w wᵀ/wᵀs` into all `k(k+1)/2`
/// positions; those spurious entries, multiplied by a multiplier of either
/// sign, are what gave the assembled `W` the wrong inertia on the batch
/// reactor and the IPM a `δ_w` that decayed by `1/3` per iteration.
///
/// Returns `true` when a correction was applied.
fn sparse_secant_update(
    b: &mut [Number],
    allowed: &[bool],
    s: &[Number],
    y: &[Number],
    max_delta: Number,
) -> bool {
    let k = s.len();
    let mut bs = vec![0.0; k];
    packed_mult(b, s, &mut bs);
    let r: Vec<Number> = y.iter().zip(bs.iter()).map(|(a, c)| a - c).collect();
    let r_norm = dot(&r, &r).sqrt();
    let y_norm = dot(y, y).sqrt();
    if !r_norm.is_finite() || r_norm <= 1e-14 * y_norm.max(Number::MIN_POSITIVE) || r_norm == 0.0 {
        return false;
    }
    let mut q = vec![0.0; k * k];
    for a in 0..k {
        for c in 0..=a {
            if !allowed[a * (a + 1) / 2 + c] {
                continue;
            }
            if a == c {
                q[a * k + a] += 2.0 * s[a] * s[a];
            } else {
                q[a * k + a] += s[c] * s[c];
                q[c * k + c] += s[a] * s[a];
                q[a * k + c] += s[a] * s[c];
                q[c * k + a] += s[a] * s[c];
            }
        }
    }
    let Some(lam) = min_norm_solve(&mut q, &r, k) else {
        return false;
    };
    let mut e = vec![0.0; b.len()];
    let mut e_max = 0.0_f64;
    for a in 0..k {
        for c in 0..=a {
            let p = a * (a + 1) / 2 + c;
            if allowed[p] {
                let v = lam[a] * s[c] + lam[c] * s[a];
                e[p] = v;
                e_max = e_max.max(v.abs());
            }
        }
    }
    if !e_max.is_finite() || e_max > max_delta {
        return false;
    }
    for (bv, ev) in b.iter_mut().zip(e.iter()) {
        *bv += ev;
    }
    true
}

/// Minimum-norm solution of `Q x = r` for a small symmetric positive
/// semidefinite `Q` (row-major, overwritten), by cyclic Jacobi
/// eigendecomposition with eigenvalues below `1e-12 · λ_max` treated as
/// zero. `None` only when `Q` is entirely zero or non-finite.
fn min_norm_solve(q: &mut [Number], r: &[Number], k: usize) -> Option<Vec<Number>> {
    let mut v = vec![0.0; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }
    for _sweep in 0..60 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for i in 0..k {
            diag += q[i * k + i] * q[i * k + i];
            for j in 0..i {
                off += q[i * k + j] * q[i * k + j];
            }
        }
        if !(off.is_finite() && diag.is_finite()) {
            return None;
        }
        if off <= 1e-30 * diag {
            break;
        }
        for p in 0..k {
            for qq in (p + 1)..k {
                let apq = q[p * k + qq];
                if apq == 0.0 {
                    continue;
                }
                let app = q[p * k + p];
                let aqq = q[qq * k + qq];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for i in 0..k {
                    let aip = q[i * k + p];
                    let aiq = q[i * k + qq];
                    q[i * k + p] = c * aip - sn * aiq;
                    q[i * k + qq] = sn * aip + c * aiq;
                }
                for i in 0..k {
                    let api = q[p * k + i];
                    let aqi = q[qq * k + i];
                    q[p * k + i] = c * api - sn * aqi;
                    q[qq * k + i] = sn * api + c * aqi;
                }
                for i in 0..k {
                    let vip = v[i * k + p];
                    let viq = v[i * k + qq];
                    v[i * k + p] = c * vip - sn * viq;
                    v[i * k + qq] = sn * vip + c * viq;
                }
            }
        }
    }
    let lmax = (0..k).map(|i| q[i * k + i].abs()).fold(0.0_f64, f64::max);
    if lmax <= 0.0 || !lmax.is_finite() {
        return None;
    }
    let mut x = vec![0.0; k];
    for j in 0..k {
        let lj = q[j * k + j];
        if lj.abs() <= 1e-12 * lmax {
            continue;
        }
        let coef: Number = (0..k).map(|i| v[i * k + j] * r[i]).sum::<Number>() / lj;
        for i in 0..k {
            x[i] += coef * v[i * k + j];
        }
    }
    Some(x)
}

/// Apply one curvature pair to a single element. Returns `true` when the
/// update was accepted.
fn update_element(
    e: &mut Element,
    s: &[Number],
    y: &[Number],
    update_type: UpdateType,
    init_val_min: Number,
    init_val_max: Number,
    curvature_cap: Number,
) -> bool {
    let sts = dot(s, s);
    if !(sts > 0.0) || !sts.is_finite() {
        return false;
    }
    let sty = dot(s, y);
    if !sty.is_finite() {
        return false;
    }
    // The curvature this pair actually implies, and the ceiling any
    // single update to this block is allowed to reach.
    let s_norm = sts.sqrt();
    let implied = dot(y, y).sqrt() / s_norm;
    let max_delta = curvature_cap * implied;

    // Pattern-constrained element: no scalar seeding (a `γI` start writes
    // curvature onto diagonal entries the model declares structurally zero)
    // and no rank-1 formula (which fills every excluded entry). The sparse
    // secant update starts from `B = 0` and keeps the pattern exactly.
    if let Some(allowed) = e.allowed.as_ref() {
        let ok = sparse_secant_update(&mut e.b, allowed, s, y, max_delta);
        e.seeded |= ok;
        return ok;
    }

    // One-time scalar seeding: `B_e ← γ I` with γ the `scalar1` ratio of
    // this element's own first pair, so the block starts at the right
    // order of magnitude instead of at zero. BFGS needs γ > 0 to have a
    // positive-definite base; SR1 takes either sign, which is the point
    // of using it here.
    if !e.seeded {
        let mut gamma = sty / sts;
        if !gamma.is_finite() || gamma == 0.0 {
            gamma = if update_type == UpdateType::Bfgs {
                1.0
            } else {
                0.0
            };
        }
        if update_type == UpdateType::Bfgs && gamma <= 0.0 {
            gamma = 1.0;
        }
        if gamma != 0.0 {
            let mag = gamma.abs().clamp(init_val_min, init_val_max);
            gamma = gamma.signum() * mag;
        }
        if e.dense {
            for a in 0..e.k() {
                e.b[a * (a + 1) / 2 + a] = gamma;
            }
        } else {
            e.b.iter_mut().for_each(|v| *v = gamma);
        }
        e.seeded = true;
    }

    if !e.dense {
        // Diagonal element: Dennis-Wolkowicz weak secant update, the
        // minimum-Frobenius diagonal correction satisfying `sᵀBs = sᵀy`.
        let s_bs: Number = (0..e.k()).map(|a| e.b[a] * s[a] * s[a]).sum();
        let denom: Number = s.iter().map(|v| v * v * v * v).sum();
        if !(denom > 0.0) || !denom.is_finite() {
            return false;
        }
        let scale = (sty - s_bs) / denom;
        if !scale.is_finite() {
            return false;
        }
        for a in 0..e.k() {
            e.b[a] += scale * s[a] * s[a];
        }
        return true;
    }

    let mut bs = vec![0.0; e.k()];
    packed_mult(&e.b, s, &mut bs);

    match update_type {
        UpdateType::Sr1 => {
            // w = y − Bs;  B += w wᵀ / (wᵀ s)
            let w: Vec<Number> = y.iter().zip(bs.iter()).map(|(a, b)| a - b).collect();
            let den = dot(&w, s);
            let w_norm = dot(&w, &w).sqrt();
            let s_norm = sts.sqrt();
            // `<=`, not `<`. When the element's model already reproduces
            // its own curvature — a linear constraint, whose `y` is
            // identically zero, or a block the previous pair already
            // matched — `w` is exactly zero and both sides are zero. A
            // strict comparison lets that through and the rank-1 term
            // divides 0 by 0, publishing a NaN Hessian; the IPM then
            // reports a converged-looking restoration failure rather than
            // anything that names the cause. A linear element is the
            // common case, not a corner: every linear constraint row in
            // the model hits this on its first pair.
            if !den.is_finite() || w_norm == 0.0 || den.abs() <= SR1_SAFEGUARD * s_norm * w_norm {
                return false;
            }
            // `‖w wᵀ/den‖_max = max|w|² / |den|`; reject before writing.
            let w_max = w.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            if w_max * w_max / den.abs() > max_delta {
                return false;
            }
            let mut p = 0usize;
            for a in 0..e.k() {
                for c in 0..=a {
                    e.b[p] += w[a] * w[c] / den;
                    p += 1;
                }
            }
            true
        }
        UpdateType::Bfgs => {
            // Powell-damped BFGS on the element block.
            let s_bs = dot(s, &bs);
            let bs_norm = dot(&bs, &bs).sqrt();
            if !(s_bs > 0.0) || !s_bs.is_finite() || s_bs <= BFGS_DENOM_FLOOR * s_norm * bs_norm {
                return false;
            }
            let theta = if sty >= POWELL_THETA * s_bs {
                1.0
            } else {
                (1.0 - POWELL_THETA) * s_bs / (s_bs - sty)
            };
            if !theta.is_finite() {
                return false;
            }
            let r: Vec<Number> = y
                .iter()
                .zip(bs.iter())
                .map(|(yy, bb)| theta * yy + (1.0 - theta) * bb)
                .collect();
            let sr = dot(s, &r);
            let r_norm = dot(&r, &r).sqrt();
            if !(sr > 0.0) || !sr.is_finite() || sr <= BFGS_DENOM_FLOOR * s_norm * r_norm {
                return false;
            }
            let r_max = r.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let bs_max = bs.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            if r_max * r_max / sr + bs_max * bs_max / s_bs > max_delta {
                return false;
            }
            let mut p = 0usize;
            for a in 0..e.k() {
                for c in 0..=a {
                    e.b[p] += r[a] * r[c] / sr - bs[a] * bs[c] / s_bs;
                    p += 1;
                }
            }
            true
        }
    }
}

impl HessianUpdater for PartitionedQuasiNewtonUpdater {
    fn update_hessian(&mut self, data: &IpoptDataHandle, cq: &IpoptCqHandle) -> bool {
        let (curr_x, curr_y_c, curr_y_d) = match data.borrow().curr.as_ref() {
            Some(c) => (c.x.clone(), c.y_c.clone(), c.y_d.clone()),
            None => return true,
        };
        let curr_grad_f = cq.borrow().curr_grad_f();
        let curr_jac_c = cq.borrow().curr_jac_c();
        let curr_jac_d = cq.borrow().curr_jac_d();

        let (Some(jac_c), Some(jac_d)) = (
            curr_jac_c.as_any().downcast_ref::<GenTMatrix>(),
            curr_jac_d.as_any().downcast_ref::<GenTMatrix>(),
        ) else {
            // Not the plain NLP path (the restoration sub-IPM carries a
            // different Jacobian shape). The builder downgrades
            // restoration to the limited-memory updater, so this is a
            // guard, not a live branch.
            return false;
        };

        let x = flat(&*curr_x);
        let grad_f = flat(&*curr_grad_f);
        let n = x.len();

        if self.space.is_none() {
            self.build_structure(n, &grad_f, jac_c, jac_d);
        }
        let y_c_now = flat(&*curr_y_c);
        let y_d_now = flat(&*curr_y_d);

        // ---- curvature pairs, one per element -----------------------
        //
        // `s` is shared (the primal step); each element gets its own `y`
        // from its own gradient's change, so no element's pair is
        // contaminated by another's curvature — the property the
        // monolithic L-BFGS `y` cannot have.
        let s_full: Option<Vec<Number>> = self
            .prev_x
            .as_ref()
            .map(|p| x.iter().zip(p.iter()).map(|(a, b)| a - b).collect());

        // In `PrimalBlock` mode every element reads the same vector: the
        // Lagrangian gradient's *change*. It is formed once here, using
        // upstream's convention that BOTH Jacobians are dotted against the
        // CURRENT multipliers, so `y` is the difference of one fixed
        // function rather than of two different Lagrangians
        // (`IpLimMemQuasiNewtonUpdater.cpp:284-308`, and the same reasoning
        // the limited-memory updater records).
        let lagrangian_dy: Option<Vec<Number>> = if self.mode == ElementMode::PrimalBlock {
            match (
                self.prev_grad_f.as_ref(),
                self.prev_jac_c.as_ref(),
                self.prev_jac_d.as_ref(),
            ) {
                (Some(pg), Some(pc), Some(pd)) => {
                    let mut dy = vec![0.0; n];
                    for i in 0..n {
                        dy[i] = grad_f[i] - pg[i];
                    }
                    for (jac, prev, mult) in [(jac_c, pc, &y_c_now), (jac_d, pd, &y_d_now)] {
                        let (ir, jc, cur) = (jac.irows(), jac.jcols(), jac.values());
                        for k in 0..ir.len() {
                            let row = (ir[k] - 1) as usize;
                            let col = (jc[k] - 1) as usize;
                            dy[col] += (cur[k] - prev[k]) * mult[row];
                        }
                    }
                    Some(dy)
                }
                _ => None,
            }
        } else {
            None
        };

        let oracle = std::env::var("POUNCE_PARTITIONED_ORACLE").is_ok();
        if oracle {
            self.dbg = DebugPeaks {
                step_norm: s_full.as_ref().map(|v| dot(v, v).sqrt()).unwrap_or(0.0),
                ..DebugPeaks::default()
            };
        }
        let mut s_loc: Vec<Number> = Vec::new();
        let mut y_loc: Vec<Number> = Vec::new();
        let mut g_loc: Vec<Number> = Vec::new();
        for e in &mut self.elements {
            let k = e.k();
            g_loc.clear();
            g_loc.resize(k, 0.0);
            match e.source {
                ElementSource::Objective => {
                    for (a, &i) in e.support.iter().enumerate() {
                        g_loc[a] = grad_f[i as usize];
                    }
                }
                ElementSource::EqRow => {
                    let v = jac_c.values();
                    for &(pos, local) in &e.entries {
                        g_loc[local as usize] += v[pos as usize];
                    }
                }
                ElementSource::IneqRow => {
                    let v = jac_d.values();
                    for &(pos, local) in &e.entries {
                        g_loc[local as usize] += v[pos as usize];
                    }
                }
                ElementSource::LagrangianBlock => {}
            }

            let pair_ready = match e.source {
                // The block's `y` comes from the shared Lagrangian
                // difference, not from a stored per-element gradient.
                ElementSource::LagrangianBlock => lagrangian_dy.is_some(),
                _ => e.has_prev,
            };
            if let (Some(s_full), true) = (s_full.as_ref(), pair_ready) {
                s_loc.clear();
                y_loc.clear();
                for (a, &i) in e.support.iter().enumerate() {
                    s_loc.push(s_full[i as usize]);
                    y_loc.push(match e.source {
                        ElementSource::LagrangianBlock => {
                            lagrangian_dy.as_ref().expect("checked above")[i as usize]
                        }
                        _ => g_loc[a] - e.prev_g[a],
                    });
                }
                let before = e.b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                if update_element(
                    e,
                    &s_loc,
                    &y_loc,
                    self.update_type,
                    self.init_val_min,
                    self.init_val_max,
                    self.curvature_cap,
                ) {
                    self.accepted_updates += 1;
                } else {
                    self.skipped_updates += 1;
                }
                if oracle {
                    let sn = dot(&s_loc, &s_loc).sqrt();
                    let yn = dot(&y_loc, &y_loc).sqrt();
                    let r = if sn > 0.0 { yn / sn } else { 0.0 };
                    if r > self.dbg.ratio {
                        self.dbg = DebugPeaks {
                            ratio: r,
                            ratio_s: sn,
                            ratio_y: yn,
                            ratio_k: e.k(),
                            ..self.dbg
                        };
                    }
                    let after = e.b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                    self.dbg.delta = self.dbg.delta.max((after - before).abs());
                }
            }

            e.prev_g.copy_from_slice(&g_loc);
            e.has_prev = true;
        }
        self.prev_x = Some(x);
        if self.mode == ElementMode::PrimalBlock {
            self.prev_grad_f = Some(grad_f.clone());
            self.prev_jac_c = Some(jac_c.values().to_vec());
            self.prev_jac_d = Some(jac_d.values().to_vec());
        }

        // ---- assemble ------------------------------------------------
        //
        //   W = ∇²f + Σ_j (y_c)_j ∇²c_j + Σ_j (y_d)_j ∇²d_j
        //
        // with `obj_factor = 1`, matching
        // `IpoptCalculatedQuantities::curr_exact_hessian`.
        let y_c = flat(&*curr_y_c);
        let y_d = flat(&*curr_y_d);
        let space = Rc::clone(self.space.as_ref().expect("structure built above"));
        let mut w = SymTMatrix::new(Rc::clone(&space));
        // Before any element has taken a pair there is no curvature
        // information anywhere, and `init_val · I` is the same opening
        // model the limited-memory path's empty-history branch uses.
        // Publishing the honest all-zero `W` instead hands the first KKT
        // solve a `(1,1)` block with no curvature at all and lets the
        // inertia correction invent the scale.
        let any_seeded = self.elements.iter().any(|e| e.seeded);
        {
            let vals = w.values_mut();
            vals.iter_mut().for_each(|v| *v = 0.0);
            if !any_seeded {
                for &p in &self.diag_pos {
                    vals[p as usize] = self.init_val;
                }
            }
            for e in self.elements.iter().filter(|_| any_seeded) {
                let weight = match e.source {
                    ElementSource::Objective => 1.0,
                    ElementSource::EqRow => y_c[e.row as usize],
                    ElementSource::IneqRow => y_d[e.row as usize],
                    // The multiplier is already inside the modelled
                    // function; weighting again would square it.
                    ElementSource::LagrangianBlock => 1.0,
                };
                if weight == 0.0 || !weight.is_finite() {
                    continue;
                }
                for (p, &m) in e.map.iter().enumerate() {
                    if m != u32::MAX {
                        vals[m as usize] += weight * e.b[p];
                    }
                }
            }
            // Coordinates no element covers get the same nonzero floor
            // the masked limited-memory diagonal uses, for the same
            // reason: a structurally empty `(1,1)` row is carried by
            // `Σ_x` alone and costs the factorization a near-singular
            // pivot on every one of them.
            if any_seeded {
                for &i in &self.uncovered {
                    vals[self.diag_pos[i as usize] as usize] = self.init_val_min;
                }
            }
        }
        // Direct correctness oracle (`POUNCE_PARTITIONED_ORACLE`). On a
        // model that *does* supply second derivatives — every `.nl`, for
        // instance — the exact Lagrangian Hessian at this very iterate,
        // with this very `obj_factor` and these very multipliers, is one
        // call away. Comparing against it is the only check in this
        // module that reads a number the updater did not produce: the
        // unit tests pin the update formulas against themselves, and a
        // self-consistently wrong assembly would pass all of them.
        //
        // Never on by default: it evaluates `eval_h` every iteration,
        // which is precisely the cost the updater exists to avoid.
        if std::env::var("POUNCE_PARTITIONED_ORACLE").is_ok() {
            let exact = cq.borrow().curr_exact_hessian();
            if let Some(t) = exact.as_any().downcast_ref::<SymTMatrix>() {
                // Feasibility census for a sparse finite-difference
                // Hessian: the number of directional derivatives such a
                // scheme needs is set by the coloring of this pattern,
                // and `rho_max` (the widest symmetric row) is its
                // practical lower bound. Printed once.
                if std::env::var("POUNCE_HESS_PATTERN_CENSUS").is_ok() && !self.census_done {
                    self.census_done = true;
                    let n_h = t.space().dim() as usize;
                    let mut deg = vec![0usize; n_h];
                    for (&i, &j) in t.irows().iter().zip(t.jcols().iter()) {
                        let (a, b) = ((i - 1) as usize, (j - 1) as usize);
                        deg[a] += 1;
                        if a != b {
                            deg[b] += 1;
                        }
                    }
                    let rho_max = deg.iter().copied().max().unwrap_or(0);
                    let mean = deg.iter().sum::<usize>() as f64 / n_h as f64;
                    let mut hist = [0usize; 8];
                    for &d in &deg {
                        let b = (d.saturating_sub(1) / 8).min(7);
                        hist[b] += 1;
                    }
                    eprintln!(
                        "hess-pattern: n={n_h} nnz={} rho_max={rho_max} mean_row={mean:.2} \
                         hist(1-8,9-16,...)={hist:?}",
                        t.nonzeros()
                    );
                }
                use std::collections::HashMap;
                let mut mine: HashMap<(Index, Index), Number> = HashMap::new();
                for ((&i, &j), &v) in space
                    .irows()
                    .iter()
                    .zip(space.jcols().iter())
                    .zip(w.values().iter())
                {
                    *mine.entry((i, j)).or_insert(0.0) += v;
                }
                let (mut max_exact, mut max_err) = (0.0_f64, 0.0_f64);
                let (mut num, mut den) = (0.0_f64, 0.0_f64);
                // Frobenius mass of the exact Hessian that falls INSIDE
                // this updater's pattern. For `ElementMode::PrimalBlock`
                // this is the direct test of the variable-ordering
                // assumption: a transcription that orders by stage puts
                // nearly all of it inside, one that does not shows a low
                // fraction and the partition is wrong for that model.
                let mut captured = 0.0_f64;
                let mut worst = ((0, 0), 0.0, 0.0);
                let mut seen: HashMap<(Index, Index), bool> = HashMap::new();
                for ((&i, &j), &v) in t
                    .irows()
                    .iter()
                    .zip(t.jcols().iter())
                    .zip(t.values().iter())
                {
                    seen.insert((i, j), true);
                    let m = mine.get(&(i, j)).copied().unwrap_or(0.0);
                    if mine.contains_key(&(i, j)) {
                        captured += v * v;
                    }
                    let e = (m - v).abs();
                    max_exact = max_exact.max(v.abs());

                    if e > max_err {
                        max_err = e;
                        worst = ((i, j), v, m);
                    }
                    num += e * e;
                    den += v * v;
                }
                // Entries this updater carries that the true Hessian does
                // not: the per-constraint pattern is `supp ⊗ supp`, an
                // over-estimate, and those entries must be ~0.
                let mut extra = 0.0_f64;
                for (&k, &v) in mine.iter() {
                    if !seen.contains_key(&k) {
                        extra = extra.max(v.abs());
                    }
                }
                eprintln!(
                    "partitioned-qn oracle: rel_fro={:.3e} max_abs_err={:.3e}                      max|exact|={:.3e} worst={:?} exact={:.6e} mine={:.6e}                      max|extra-pattern|={:.3e} pattern_captures={:.4}",
                    (num / den.max(1e-300)).sqrt(),
                    max_err,
                    max_exact,
                    worst.0,
                    worst.1,
                    worst.2,
                    extra,
                    (captured / den.max(1e-300)).sqrt()
                );
                eprintln!(
                    "  peaks: max|y_e|/|s_e|={:.3e} (|s_e|={:.3e} |y_e|={:.3e} k={}) \
                     max_block_delta={:.3e} |s|={:.3e} accepted={} skipped={}",
                    self.dbg.ratio,
                    self.dbg.ratio_s,
                    self.dbg.ratio_y,
                    self.dbg.ratio_k,
                    self.dbg.delta,
                    self.dbg.step_norm,
                    self.accepted_updates,
                    self.skipped_updates
                );
            }
        }
        if std::env::var("POUNCE_PARTITIONED_DUMP").is_ok() && space.nonzeros() <= 32 {
            eprintln!(
                "partitioned-qn W: seeded={any_seeded} irows={:?} jcols={:?} vals={:?}",
                space.irows(),
                space.jcols(),
                w.values()
            );
            eprintln!("  y_c={y_c:?} y_d={y_d:?}");
        }
        data.borrow_mut().w = Some(Rc::new(w));
        true
    }
}

/// Read a primal-space vector's values as a flat slice. Mirrors
/// `lim_mem_quasi_newton::expanded_of`; kept local so the two updaters do
/// not share mutable helper state.
fn flat(v: &dyn Vector) -> Vec<Number> {
    if let Some(dv) = v.as_any().downcast_ref::<DenseVector>() {
        return dv.expanded_values();
    }
    if let Some(cv) = v.as_any().downcast_ref::<CompoundVector>() {
        let mut out = Vec::with_capacity(cv.dim() as usize);
        for i in 0..cv.n_comps() {
            out.extend(flat(cv.comp(i)));
        }
        return out;
    }
    panic!("PartitionedQuasiNewtonUpdater: unsupported primal vector type");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#989 item 5: a row supported on `{0,1,2,3,4,5,6}` whose declared
    /// Hessian couples `(0,1)` and `(2,3)` and carries `(4,4)` alone splits
    /// into three elements, and coordinates 5 and 6 (no pattern entry: linear)
    /// are dropped. The Jacobian entries follow their coordinates.
    #[test]
    fn declared_pattern_splits_an_element_into_its_independent_blocks() {
        let u = PartitionedQuasiNewtonUpdater::new(UpdateType::Sr1);
        let pat: Vec<(Index, Index)> = vec![(1, 0), (3, 2), (4, 4), (9, 9)];
        let mut adj: std::collections::HashMap<Index, Vec<Index>> = Default::default();
        for &(r, c) in &pat {
            adj.entry(r).or_default().push(c);
            if r != c {
                adj.entry(c).or_default().push(r);
            }
        }
        let mut els = Vec::new();
        let entries: Vec<(u32, u32)> = (0..7).map(|i| (10 + i as u32, i as u32)).collect();
        u.push_split(
            &mut els,
            Some(&adj),
            ElementSource::EqRow,
            3,
            (0..7).collect(),
            entries,
        );
        let sup: Vec<Vec<Index>> = els.iter().map(|e| e.support.clone()).collect();
        assert_eq!(sup, vec![vec![0, 1], vec![2, 3], vec![4]]);
        assert_eq!(els[1].entries, vec![(12, 0), (13, 1)]);
        assert!(els.iter().all(|e| e.row == 3));
        // Without a declared pattern the row stays one dense element.
        let mut one = Vec::new();
        u.push_split(
            &mut one,
            None,
            ElementSource::EqRow,
            3,
            (0..7).collect(),
            Vec::new(),
        );
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].support.len(), 7);
    }

    /// gh#989 item 5: on a *star* pattern — a control `T` coupled to two
    /// states, no state–state or state diagonal entries, the shape of every
    /// Radau collocation row's group — the pattern-constrained secant update
    /// recovers the exact element Hessian from `B = 0` with a single pair,
    /// and the excluded entries stay exactly zero. A dense SR1 term from the
    /// same pair writes all six entries.
    #[test]
    fn sparse_secant_recovers_a_star_hessian_from_one_pair() {
        // H = [[a, b1, b2], [b1, 0, 0], [b2, 0, 0]], packed lower triangle.
        let (a, b1, b2) = (-3.0e-4, 2.0e-3, -7.0e-4);
        let h = [a, b1, 0.0, b2, 0.0, 0.0];
        let allowed = vec![true, true, false, true, false, false];
        let s = vec![0.3, -1.2, 0.8];
        let mut y = vec![0.0; 3];
        packed_mult(&h, &s, &mut y);
        let mut b = vec![0.0; 6];
        assert!(sparse_secant_update(
            &mut b,
            &allowed,
            &s,
            &y,
            Number::INFINITY
        ));
        for p in 0..6 {
            assert!(
                (b[p] - h[p]).abs() < 1e-15,
                "packed entry {p}: {} vs {}",
                b[p],
                h[p]
            );
        }
        // Already exact: a second pair along another direction is a no-op.
        let s2 = vec![-0.5, 0.1, 0.4];
        let mut y2 = vec![0.0; 3];
        packed_mult(&h, &s2, &mut y2);
        assert!(!sparse_secant_update(
            &mut b,
            &allowed,
            &s2,
            &y2,
            Number::INFINITY
        ));
    }

    /// The pattern-constrained update satisfies the secant equation and
    /// never writes an excluded position, including when the pattern
    /// cannot be determined by one pair; a coordinate with no step
    /// component takes no correction rather than a division by zero.
    #[test]
    fn sparse_secant_keeps_the_pattern_and_the_secant_equation() {
        // 4 coordinates, allowed: (0,0), (1,0), (2,1), (3,3), (2,2).
        let k = 4;
        let mut allowed = vec![false; k * (k + 1) / 2];
        for (r, c) in [(0, 0), (1, 0), (2, 1), (3, 3), (2, 2)] {
            allowed[r * (r + 1) / 2 + c] = true;
        }
        let mut b = vec![0.0; k * (k + 1) / 2];
        b[0] = 1.0;
        let s = vec![0.5, -0.25, 1.0, 0.0];
        let y = vec![0.1, 0.7, -0.4, 0.0];
        assert!(sparse_secant_update(
            &mut b,
            &allowed,
            &s,
            &y,
            Number::INFINITY
        ));
        let mut bs = vec![0.0; k];
        packed_mult(&b, &s, &mut bs);
        for a in 0..k {
            assert!(
                (bs[a] - y[a]).abs() < 1e-12,
                "row {a}: {} vs {}",
                bs[a],
                y[a]
            );
        }
        for (p, &ok) in allowed.iter().enumerate() {
            if !ok {
                assert_eq!(b[p], 0.0, "excluded packed entry {p} was written");
            }
        }
        assert_eq!(b[9], 0.0, "(3,3) has no step component and must stay 0");
    }

    /// `min_norm_solve` returns the minimum-norm solution on a singular
    /// PSD system (a rank-one `Q = v vᵀ`), not a blow-up.
    #[test]
    fn min_norm_solve_handles_a_singular_system() {
        let v = [1.0, 2.0, 0.0];
        let mut q = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                q[i * 3 + j] = v[i] * v[j];
            }
        }
        // r in range(Q): r = Q x* with x* = v/|v|^2 * 5 -> r = 5 v.
        let r = [5.0, 10.0, 0.0];
        let x = min_norm_solve(&mut q, &r, 3).expect("solvable");
        let want = [1.0, 2.0, 0.0];
        for i in 0..3 {
            assert!(
                (x[i] - want[i]).abs() < 1e-12,
                "{i}: {} vs {}",
                x[i],
                want[i]
            );
        }
    }

    /// gh#989 item 5: `partitioned_elements=blocks` with a declared pattern
    /// takes its blocks from the pattern's connected components, not from
    /// contiguous index ranges. Variables 0 and 3 are coupled, as are 1, 2
    /// and 5 (a star on 2); variable 4 appears in no entry and is linear.
    #[test]
    fn blocks_follow_the_declared_components_not_the_index_order() {
        let mut u = PartitionedQuasiNewtonUpdater::new(UpdateType::Bfgs);
        u.mode = ElementMode::PrimalBlock;
        u.block_size = 64;
        u.declared_pattern = Some(vec![(3, 0), (2, 1), (2, 2), (5, 2)]);
        let empty = |rows| {
            GenTMatrix::new(pounce_linalg::triplet::GenTMatrixSpace::new(
                rows,
                6,
                Vec::new(),
                Vec::new(),
            ))
        };
        u.build_structure(6, &[0.0; 6], &empty(0), &empty(0));
        let sup: Vec<Vec<Index>> = u.elements.iter().map(|e| e.support.clone()).collect();
        assert_eq!(sup, vec![vec![0, 3], vec![1, 2, 5]]);
        assert_eq!(u.uncovered, vec![4]);
        // The star keeps only its declared entries: (1,1), (5,5), (5,1) excluded.
        let star = &u.elements[1];
        assert_eq!(
            star.allowed.as_deref(),
            Some(&[false, true, true, false, true, false][..])
        );
    }

    /// `packed_mult` agrees with a dense symmetric product, off-diagonal
    /// fan-out included.
    #[test]
    fn packed_mult_matches_dense() {
        // B = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
        let b = vec![1.0, 2.0, 4.0, 3.0, 5.0, 6.0];
        let s = vec![1.0, -2.0, 0.5];
        let mut out = vec![0.0; 3];
        packed_mult(&b, &s, &mut out);
        let dense = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];
        for a in 0..3 {
            let want: Number = (0..3).map(|c| dense[a][c] * s[c]).sum();
            assert!(
                (out[a] - want).abs() < 1e-12,
                "row {a}: {} vs {want}",
                out[a]
            );
        }
    }

    /// SR1 satisfies the secant equation `B_+ s = y` exactly in one step
    /// when the denominator is safe — the property that makes it the
    /// right per-element update for a nonconvex constraint.
    #[test]
    fn sr1_satisfies_the_secant_equation() {
        let mut e = Element {
            source: ElementSource::EqRow,
            row: 0,
            support: vec![0, 1, 2],
            dense: true,
            b: vec![0.0; 6],
            prev_g: vec![0.0; 3],
            has_prev: true,
            seeded: true,
            entries: Vec::new(),
            map: Vec::new(),
            allowed: None,
        };
        let s = vec![1.0, 0.5, -0.25];
        // A deliberately indefinite target: SR1 must not sanitize it.
        let y = vec![-2.0, 1.0, 3.0];
        assert!(update_element(
            &mut e,
            &s,
            &y,
            UpdateType::Sr1,
            1e-8,
            1e8,
            1e12
        ));
        let mut bs = vec![0.0; 3];
        packed_mult(&e.b, &s, &mut bs);
        for a in 0..3 {
            assert!(
                (bs[a] - y[a]).abs() < 1e-10,
                "component {a}: {} vs {}",
                bs[a],
                y[a]
            );
        }
    }

    /// An element whose curvature is genuinely negative keeps a negative
    /// block under SR1. Damped BFGS does not, which is why SR1 is the
    /// default — see the module docs and issue #131.
    ///
    /// Note what carries the sign on a **one-dimensional** element: the
    /// scalar seeding `gamma = sᵀy/sᵀs` already satisfies the secant
    /// equation exactly, so the rank-1 term has nothing to add and
    /// `update_element` correctly declines. The property under test is
    /// the resulting curvature, not the return value.
    #[test]
    fn sr1_preserves_negative_curvature_where_bfgs_would_not() {
        let make = || Element {
            source: ElementSource::EqRow,
            row: 0,
            support: vec![0],
            dense: true,
            b: vec![0.0],
            prev_g: vec![0.0],
            has_prev: true,
            seeded: false,
            entries: Vec::new(),
            map: Vec::new(),
            allowed: None,
        };
        let s = vec![1.0];
        let y = vec![-3.0];

        let mut sr1 = make();
        update_element(&mut sr1, &s, &y, UpdateType::Sr1, 1e-8, 1e8, 1e12);
        assert!(sr1.b[0] < 0.0, "SR1 kept curvature {}", sr1.b[0]);
        // and it is the exact secant value, not merely the right sign
        assert!((sr1.b[0] + 3.0).abs() < 1e-12, "{}", sr1.b[0]);

        let mut bfgs = make();
        update_element(&mut bfgs, &s, &y, UpdateType::Bfgs, 1e-8, 1e8, 1e12);
        assert!(bfgs.b[0] > 0.0, "damped BFGS kept curvature {}", bfgs.b[0]);
    }

    /// **The scalar seeding always annihilates the first SR1 update**,
    /// for every element and every dimension. Seeding sets `B = γI` with
    /// `γ = sᵀy/sᵀs`, so `Bs = γs` and the SR1 denominator is
    /// `wᵀs = sᵀy − γ·sᵀs ≡ 0`. That is not a defect — the seeded block
    /// already satisfies the secant equation along `s` — but it means an
    /// element's *first* pair contributes only a multiple of the
    /// identity and no directional information whatsoever. With ~5 000
    /// elements each receiving one direction per iteration, that costs a
    /// full iteration of information per element; see
    /// `dev-notes/partitioned-quasi-newton-prototype.md`.
    #[test]
    fn scalar_seeding_leaves_the_first_sr1_update_with_nothing_to_do() {
        let mut e = Element {
            source: ElementSource::EqRow,
            row: 0,
            support: vec![0, 1],
            dense: true,
            b: vec![0.0; 3],
            prev_g: vec![0.0; 2],
            has_prev: true,
            seeded: false,
            entries: Vec::new(),
            map: Vec::new(),
            allowed: None,
        };
        let s = vec![1.0, 0.5];
        let y = vec![2.0, -4.0];
        // Skipped, and what is left behind is exactly the seeded scalar.
        assert!(!update_element(
            &mut e,
            &s,
            &y,
            UpdateType::Sr1,
            1e-8,
            1e8,
            1e12
        ));
        assert!(e.seeded);
        assert_eq!(e.b[1], 0.0, "off-diagonal must still be zero");
        assert!(
            (e.b[0] - e.b[2]).abs() < 1e-15,
            "block must be a multiple of I"
        );
    }

    /// The SR1-vs-BFGS contrast on a **two-dimensional** element, driven
    /// through the path the solver actually takes: a first pair to seed,
    /// then a second that the rank-1 term can act on. SR1 reproduces the
    /// second pair exactly and leaves the block indefinite; damped BFGS
    /// returns a positive definite block that does not.
    #[test]
    fn sr1_reaches_an_indefinite_block_where_bfgs_stays_definite() {
        let make = || Element {
            source: ElementSource::EqRow,
            row: 0,
            support: vec![0, 1],
            dense: true,
            b: vec![0.0; 3],
            prev_g: vec![0.0; 2],
            has_prev: true,
            seeded: false,
            entries: Vec::new(),
            map: Vec::new(),
            allowed: None,
        };
        let (s1, y1) = (vec![1.0, 0.0], vec![2.0, 0.0]);
        let (s2, y2) = (vec![0.0, 1.0], vec![0.0, -4.0]);

        let mut sr1 = make();
        update_element(&mut sr1, &s1, &y1, UpdateType::Sr1, 1e-8, 1e8, 1e12);
        assert!(update_element(
            &mut sr1,
            &s2,
            &y2,
            UpdateType::Sr1,
            1e-8,
            1e8,
            1e12
        ));
        let mut bs = vec![0.0; 2];
        packed_mult(&sr1.b, &s2, &mut bs);
        for a in 0..2 {
            assert!(
                (bs[a] - y2[a]).abs() < 1e-10,
                "component {a}: {} vs {}",
                bs[a],
                y2[a]
            );
        }
        // det < 0 ⇒ one eigenvalue of each sign: the indefiniteness the
        // inertia correction is supposed to see.
        let det = sr1.b[0] * sr1.b[2] - sr1.b[1] * sr1.b[1];
        assert!(det < 0.0, "SR1 block determinant {det}");

        let mut bfgs = make();
        update_element(&mut bfgs, &s1, &y1, UpdateType::Bfgs, 1e-8, 1e8, 1e12);
        assert!(update_element(
            &mut bfgs,
            &s2,
            &y2,
            UpdateType::Bfgs,
            1e-8,
            1e8,
            1e12
        ));
        let det_b = bfgs.b[0] * bfgs.b[2] - bfgs.b[1] * bfgs.b[1];
        assert!(
            bfgs.b[0] > 0.0 && det_b > 0.0,
            "damped BFGS block is positive definite: diag {} det {det_b}",
            bfgs.b[0]
        );
    }

    /// The diagonal fallback satisfies the weak secant condition
    /// `sᵀBs = sᵀy`, which is the whole contract it is asked for.
    #[test]
    fn diagonal_element_satisfies_the_weak_secant_condition() {
        let mut e = Element {
            source: ElementSource::Objective,
            row: 0,
            support: vec![0, 1, 2],
            dense: false,
            b: vec![0.0; 3],
            prev_g: vec![0.0; 3],
            has_prev: true,
            seeded: true,
            entries: Vec::new(),
            map: Vec::new(),
            allowed: None,
        };
        let s = vec![1.0, -2.0, 0.5];
        let y = vec![0.5, 1.0, -3.0];
        assert!(update_element(
            &mut e,
            &s,
            &y,
            UpdateType::Sr1,
            1e-8,
            1e8,
            1e12
        ));
        let s_bs: Number = (0..3).map(|a| e.b[a] * s[a] * s[a]).sum();
        assert!(
            (s_bs - dot(&s, &y)).abs() < 1e-10,
            "{s_bs} vs {}",
            dot(&s, &y)
        );
    }

    /// A parallel `y` that carries no new information leaves the block
    /// untouched rather than producing an unbounded rank-1 term.
    #[test]
    fn sr1_skips_a_degenerate_denominator() {
        let mut e = Element {
            source: ElementSource::EqRow,
            row: 0,
            support: vec![0, 1],
            dense: true,
            b: vec![2.0, 0.0, 2.0],
            prev_g: vec![0.0; 2],
            has_prev: true,
            seeded: true,
            entries: Vec::new(),
            map: Vec::new(),
            allowed: None,
        };
        let before = e.b.clone();
        let s = vec![1.0, 1.0];
        // y = B s exactly, so w = 0 and the denominator vanishes.
        let y = vec![2.0, 2.0];
        assert!(!update_element(
            &mut e,
            &s,
            &y,
            UpdateType::Sr1,
            1e-8,
            1e8,
            1e12
        ));
        assert_eq!(e.b, before);
    }
}
