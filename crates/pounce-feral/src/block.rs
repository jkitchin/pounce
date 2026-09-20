//! Block-parallel KKT solver for block-diagonal-plus-border ("arrowhead")
//! systems (structured-KKT Phase 5b).
//!
//! Given a partition of the KKT indices into `nblocks` blocks plus a shared
//! border — blocks couple to the border and never to each other — this
//! factorizes each block independently and in parallel, forms the border's
//! Schur complement, and solves by block elimination:
//!
//! ```text
//!   K = [ A_11          A_1b ]        S = A_bb − Σ_k A_bk A_kk⁻¹ A_kb
//!       [      ⋱        ⋮    ]
//!       [           A_KK A_Kb]        y_k = A_kk⁻¹ b_k
//!       [ A_b1 ⋯  A_bK  A_bb ]        S Δ_b = b_b − Σ_k A_bk y_k
//!                                     x_k = A_kk⁻¹ (b_k − A_kb Δ_b)
//! ```
//!
//! **Why this exists.** On an arrowhead KKT the elimination tree already
//! exposes the blocks, but feral's tree parallelism reaches only ~1.2× because
//! the tasks are per-supernode and too small to amortise: measured on a
//! 210k-variable N-1 SCOPF, 1.22e9 flops over 196 561 supernodes. As `nblocks`
//! coarse tasks the same factorization is ~10× faster. See
//! `dev-notes/kkt-scaling-phase0b.md`.
//!
//! **Inertia** comes from Haynsworth additivity,
//! `inertia(K) = Σ_k inertia(A_kk) + inertia(S)`, so the IPM's
//! inertia-correction ladder consumes it unchanged.
//!
//! **Two factorizations per block, for now.** The Schur complement is formed
//! by feral's `factorize_multifrontal_with_schur`, which produces it inside the
//! factorization rather than through `n_border` back-solves (measured 0.012 s
//! against 0.37 s). Those factors leave the border tail un-eliminated and feral
//! exposes no solve against them, so each block is factored a second time,
//! block-only, for the back-solves. A partial-solve entry point in feral would
//! remove that second factorization and take the whole KKT solve from ~4× to
//! ~6×.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use feral::symbolic::{SupernodeParams, SymbolicFactorization, symbolic_factorize_with_schur};
use feral::{CscMatrix, FactorStatus, NumericParams, Solver, factorize_multifrontal_with_schur};
use pounce_common::types::{Index, Number};
use pounce_linsol::ESymSolverStatus;
use pounce_linsol::summary::{BlockSummary, LinearSolverSummary};
use rayon::prelude::*;

use crate::{FeralConfig, configure_solver, factor_record};

/// One block's pattern, symbolic analysis and reusable buffers.
struct Block {
    /// Order of `A_kk`.
    n: usize,
    /// `A_kk` lower-triangle triplet in block-local indices, plus the
    /// input-nnz position feeding each value.
    rows: Vec<usize>,
    cols: Vec<usize>,
    src: Vec<usize>,
    /// `A_bk` entries: border position, block-local column, input position.
    cpl_b: Vec<usize>,
    cpl_k: Vec<usize>,
    cpl_src: Vec<usize>,
    /// Symbolic analysis of `[A_kk A_kb; A_bk 0]` with the border as its
    /// Schur tail; and of `A_kk` alone, held by its own solver.
    sym: Option<SymbolicFactorization>,
    solver: Solver,
}

/// Block-parallel solver over a caller-supplied partition.
///
/// Lifecycle mirrors [`crate::FeralSolverInterface`]:
/// [`Self::initialize_structure`] pins the pattern and the partition,
/// [`Self::values_array_mut`] receives the KKT nonzeros, [`Self::factor`]
/// factorizes every block and the border, and [`Self::backsolve`] applies the
/// block elimination in place.
pub struct FeralBlockSolver {
    cfg: FeralConfig,
    dim: usize,
    /// Block of each KKT index; `usize::MAX` for the border.
    label: Vec<usize>,
    /// Position of each KKT index inside its block, or in the border.
    local: Vec<usize>,
    border: Vec<usize>,
    blocks: Vec<Block>,
    /// Border-border entries: `(i, j, input position)`, lower triangle.
    ss: Vec<(usize, usize, usize)>,
    values: Vec<Number>,
    /// Dense border complement, column-major, and its factorization.
    s_dense: Vec<Number>,
    s_solver: Solver,
    negevals: Index,
    have_factor: bool,
    initialized: bool,
    last_status: ESymSolverStatus,
    sink: Option<Arc<Mutex<LinearSolverSummary>>>,
}

impl FeralBlockSolver {
    pub fn new(cfg: FeralConfig) -> Self {
        let s_solver = configure_solver(&cfg);
        Self {
            cfg,
            dim: 0,
            label: Vec::new(),
            local: Vec::new(),
            border: Vec::new(),
            blocks: Vec::new(),
            ss: Vec::new(),
            values: Vec::new(),
            s_dense: Vec::new(),
            s_solver,
            negevals: 0,
            have_factor: false,
            initialized: false,
            last_status: ESymSolverStatus::Success,
            sink: None,
        }
    }

    /// Install a shared summary sink (see
    /// [`crate::FeralSolverInterface::with_summary_sink`]).
    pub fn with_summary_sink(mut self, sink: Arc<Mutex<LinearSolverSummary>>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Pin the KKT pattern and the block partition.
    ///
    /// `ia` / `ja` are the **1-based lower-triangle** triplet the aug-system
    /// solver assembles. `labels` gives each KKT index its block, `< 0` for the
    /// border. Returns [`ESymSolverStatus::FatalError`] — the caller's signal
    /// to fall back — when the partition is unusable: wrong length, an empty
    /// border or no block, or an entry coupling two different blocks, which
    /// means the declared structure does not match the matrix.
    pub fn initialize_structure(
        &mut self,
        dim: Index,
        ia: &[Index],
        ja: &[Index],
        labels: &[i32],
    ) -> ESymSolverStatus {
        let dim = dim as usize;
        if labels.len() != dim || ia.len() != ja.len() {
            return self.fail();
        }
        let nblocks = match labels.iter().copied().max() {
            Some(m) if m >= 0 => m as usize + 1,
            _ => return self.fail(),
        };
        self.dim = dim;
        self.label = labels
            .iter()
            .map(|&l| if l < 0 { usize::MAX } else { l as usize })
            .collect();
        self.local = vec![usize::MAX; dim];
        self.border = Vec::new();
        let mut sizes = vec![0usize; nblocks];
        for i in 0..dim {
            if self.label[i] == usize::MAX {
                self.local[i] = self.border.len();
                self.border.push(i);
            } else {
                let b = self.label[i];
                self.local[i] = sizes[b];
                sizes[b] += 1;
            }
        }
        if self.border.is_empty() || sizes.iter().any(|&s| s == 0) {
            return self.fail();
        }
        let nb = self.border.len();
        self.blocks = sizes
            .iter()
            .map(|&n| Block {
                n,
                rows: Vec::new(),
                cols: Vec::new(),
                src: Vec::new(),
                cpl_b: Vec::new(),
                cpl_k: Vec::new(),
                cpl_src: Vec::new(),
                sym: None,
                solver: configure_solver(&self.cfg),
            })
            .collect();
        self.ss.clear();
        for p in 0..ia.len() {
            let (r, c) = (ia[p] as usize - 1, ja[p] as usize - 1);
            if r >= dim || c >= dim {
                return self.fail();
            }
            match (self.label[r], self.label[c]) {
                (usize::MAX, usize::MAX) => self.ss.push((self.local[r], self.local[c], p)),
                (br, bc) if br != usize::MAX && bc != usize::MAX => {
                    if br != bc {
                        // a block-to-block entry: the declared structure is
                        // not the matrix's structure
                        return self.fail();
                    }
                    let blk = &mut self.blocks[br];
                    blk.rows.push(self.local[r]);
                    blk.cols.push(self.local[c]);
                    blk.src.push(p);
                }
                (br, _) if br != usize::MAX => {
                    let blk = &mut self.blocks[br];
                    blk.cpl_b.push(self.local[c]);
                    blk.cpl_k.push(self.local[r]);
                    blk.cpl_src.push(p);
                }
                (_, bc) => {
                    let blk = &mut self.blocks[bc];
                    blk.cpl_b.push(self.local[r]);
                    blk.cpl_k.push(self.local[c]);
                    blk.cpl_src.push(p);
                }
            }
        }
        self.values = vec![0.0; ia.len()];
        self.s_dense = vec![0.0; nb * nb];
        // Symbolic analysis per block, once: the pattern is fixed across IPM
        // iterations. Values are irrelevant here, so ones carry the pattern.
        let cfg = self.cfg.clone();
        let ok = self
            .blocks
            .par_iter_mut()
            .map(|blk| {
                let m = match block_matrix(blk, nb, &[], true) {
                    Some(m) => m,
                    None => return false,
                };
                let sn = SupernodeParams {
                    preprocess: cfg.ordering_preprocess,
                    ..SupernodeParams::default()
                };
                let idx: Vec<usize> = (blk.n..blk.n + nb).collect();
                match symbolic_factorize_with_schur(&m, &sn, &idx) {
                    Ok(s) => {
                        blk.sym = Some(s);
                        true
                    }
                    Err(_) => false,
                }
            })
            .reduce(|| true, |a, b| a && b);
        if !ok {
            return self.fail();
        }
        self.initialized = true;
        self.have_factor = false;
        self.set_status(ESymSolverStatus::Success)
    }

    /// Slice the caller writes the KKT nonzeros into, in the order of the
    /// triplet passed to [`Self::initialize_structure`].
    pub fn values_array_mut(&mut self) -> &mut [Number] {
        &mut self.values
    }

    /// Factor every block (in parallel) and the border complement, and combine
    /// the inertias. `check_neg_evals` verifies the combined count against
    /// `num_neg_evals`.
    pub fn factor(&mut self, check_neg_evals: bool, num_neg_evals: Index) -> ESymSolverStatus {
        if !self.initialized {
            return self.set_status(ESymSolverStatus::FatalError);
        }
        let nb = self.border.len();
        let values = &self.values;
        let t0 = Instant::now();
        // Each block: factor with the border as Schur tail (gives this block's
        // contribution to S), and block-only (gives factors the back-solve can
        // use — feral cannot solve against the schur-tail factors).
        let results: Vec<Option<(Vec<Number>, (usize, usize, usize), bool)>> = self
            .blocks
            .par_iter_mut()
            .map(|blk| {
                let m_full = block_matrix(blk, nb, values, false)?;
                let sym = blk.sym.as_ref()?;
                let (f, inertia, mut schur) =
                    factorize_multifrontal_with_schur(&m_full, sym, &NumericParams::default())
                        .ok()?;
                // feral factors `D·A·D`, so the Schur block comes back scaled:
                // `S_scaled[i][j] = d_i · d_j · S[i][j]` over the border's own
                // scale factors. Congruence leaves the inertia alone, which is
                // why an unscaled S still gives the right pivot counts, but the
                // solve needs the true complement.
                let d = &f.scaling[blk.n..blk.n + nb];
                for j in 0..nb {
                    for i in 0..nb {
                        schur.data[i + j * nb] /= d[i] * d[j];
                    }
                }
                let m_blk = block_only_matrix(blk, values)?;
                let singular = match blk.solver.factor(&m_blk, None) {
                    FactorStatus::Success => {
                        blk.solver.inertia().map(|i| i.zero > 0).unwrap_or(false)
                    }
                    FactorStatus::Singular => true,
                    _ => return None,
                };
                Some((
                    schur.data,
                    (inertia.positive, inertia.negative, inertia.zero),
                    singular,
                ))
            })
            .collect();
        let mut inertia = (0usize, 0usize, 0usize);
        self.s_dense.iter_mut().for_each(|v| *v = 0.0);
        for (i, j, p) in &self.ss {
            self.s_dense[i + j * nb] += values[*p];
            if i != j {
                self.s_dense[j + i * nb] += values[*p];
            }
        }
        for r in &results {
            let Some((schur, inert, singular)) = r else {
                return self.set_status(ESymSolverStatus::FatalError);
            };
            if *singular {
                return self.set_status(ESymSolverStatus::Singular);
            }
            inertia = (
                inertia.0 + inert.0,
                inertia.1 + inert.1,
                inertia.2 + inert.2,
            );
            for (d, v) in self.s_dense.iter_mut().zip(schur.iter()) {
                *d += v;
            }
        }
        // the border complement, as a dense lower triangle
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..nb {
            for i in j..nb {
                r.push(i);
                c.push(j);
                v.push(self.s_dense[i + j * nb]);
            }
        }
        let s_mat = match CscMatrix::from_triplets(nb, &r, &c, &v) {
            Ok(m) => m,
            Err(_) => return self.set_status(ESymSolverStatus::FatalError),
        };
        let s_inertia = match self.s_solver.factor(&s_mat, None) {
            FactorStatus::Success => match self.s_solver.inertia() {
                Some(i) => {
                    if i.zero > 0 {
                        return self.set_status(ESymSolverStatus::Singular);
                    }
                    (i.positive, i.negative, i.zero)
                }
                None => (0, self.s_solver.num_negative_eigenvalues(), 0),
            },
            FactorStatus::Singular => return self.set_status(ESymSolverStatus::Singular),
            _ => return self.set_status(ESymSolverStatus::FatalError),
        };
        let secs = t0.elapsed().as_secs_f64();
        // Haynsworth: inertia(K) = Σ_k inertia(A_kk) + inertia(S)
        self.negevals = (inertia.1 + s_inertia.1) as Index;
        self.have_factor = true;
        self.record(secs, nb);
        if check_neg_evals && self.negevals != num_neg_evals {
            return self.set_status(ESymSolverStatus::WrongInertia);
        }
        self.set_status(ESymSolverStatus::Success)
    }

    /// Block-elimination back-substitution, in place, for `nrhs` right-hand
    /// sides packed column-major in KKT index order.
    pub fn backsolve(&self, nrhs: Index, rhs: &mut [Number]) -> ESymSolverStatus {
        let nrhs = nrhs as usize;
        if !self.have_factor || rhs.len() != self.dim * nrhs {
            return ESymSolverStatus::FatalError;
        }
        for col in 0..nrhs {
            let b = &mut rhs[col * self.dim..(col + 1) * self.dim];
            // y_k = A_kk⁻¹ b_k
            let ys: Vec<Option<Vec<Number>>> = self
                .blocks
                .par_iter()
                .enumerate()
                .map(|(k, blk)| {
                    let mut bk = vec![0.0; blk.n];
                    for i in 0..self.dim {
                        if self.label[i] == k {
                            bk[self.local[i]] = b[i];
                        }
                    }
                    blk.solver.solve(&bk).ok()
                })
                .collect();
            // S Δ_b = b_b − Σ_k A_bk y_k
            let mut rs: Vec<Number> = self.border.iter().map(|&i| b[i]).collect();
            for (k, blk) in self.blocks.iter().enumerate() {
                let Some(y) = ys[k].as_ref() else {
                    return ESymSolverStatus::FatalError;
                };
                for p in 0..blk.cpl_b.len() {
                    rs[blk.cpl_b[p]] -= self.values[blk.cpl_src[p]] * y[blk.cpl_k[p]];
                }
            }
            let Ok(db) = self.s_solver.solve(&rs) else {
                return ESymSolverStatus::FatalError;
            };
            // x_k = A_kk⁻¹ (b_k − A_kb Δ_b)
            let xs: Vec<Option<Vec<Number>>> = self
                .blocks
                .par_iter()
                .enumerate()
                .map(|(k, blk)| {
                    let mut bk = vec![0.0; blk.n];
                    for i in 0..self.dim {
                        if self.label[i] == k {
                            bk[self.local[i]] = b[i];
                        }
                    }
                    for p in 0..blk.cpl_b.len() {
                        bk[blk.cpl_k[p]] -= self.values[blk.cpl_src[p]] * db[blk.cpl_b[p]];
                    }
                    blk.solver.solve(&bk).ok()
                })
                .collect();
            for i in 0..self.dim {
                if self.label[i] == usize::MAX {
                    b[i] = db[self.local[i]];
                } else {
                    let Some(x) = xs[self.label[i]].as_ref() else {
                        return ESymSolverStatus::FatalError;
                    };
                    b[i] = x[self.local[i]];
                }
            }
        }
        ESymSolverStatus::Success
    }

    pub fn number_of_neg_evals(&self) -> Index {
        self.negevals
    }

    pub fn provides_inertia(&self) -> bool {
        true
    }

    pub fn system_dim(&self) -> Index {
        self.dim as Index
    }

    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn border_dim(&self) -> usize {
        self.border.len()
    }

    pub fn last_solve_status(&self) -> ESymSolverStatus {
        self.last_status
    }

    /// Raise every block's pivot threshold, as the monolithic backend does.
    pub fn increase_quality(&mut self) -> bool {
        let mut any = false;
        for blk in self.blocks.iter_mut() {
            any |= blk.solver.increase_quality();
        }
        any | self.s_solver.increase_quality()
    }

    fn record(&self, secs: f64, nb: usize) {
        let Some(sink) = self.sink.as_ref() else {
            return;
        };
        let Ok(mut guard) = sink.lock() else {
            return;
        };
        if guard.solver_name.is_empty() {
            guard.solver_name = "feral".to_string();
        }
        // the largest block's factorization stands for the factor record
        if let Some(stats) = self
            .blocks
            .iter()
            .max_by_key(|b| b.n)
            .and_then(|b| b.solver.last_factor_stats().map(|s| (b, s)))
        {
            let (blk, st) = stats;
            let mut rec = factor_record(&blk.solver, &st, secs);
            rec.ordering = rec.ordering.map(|o| format!("{o} (per block)"));
            guard.record(&rec);
        }
        let b = guard.blocks.get_or_insert_with(BlockSummary::default);
        b.n_blocks = self.blocks.len();
        b.border_dim = nb;
        b.largest_block = self.blocks.iter().map(|x| x.n).max().unwrap_or(0);
        b.n_factors += 1;
        b.factor_secs += secs;
    }

    fn fail(&mut self) -> ESymSolverStatus {
        self.initialized = false;
        self.set_status(ESymSolverStatus::FatalError)
    }

    fn set_status(&mut self, s: ESymSolverStatus) -> ESymSolverStatus {
        self.last_status = s;
        s
    }
}

/// `[A_kk A_kb; A_bk 0]` with the border trailing, for the Schur-tail
/// factorization. `pattern_only` fills ones (values are irrelevant to the
/// symbolic analysis) and keeps the border diagonal structurally present.
fn block_matrix(
    blk: &Block,
    nb: usize,
    values: &[Number],
    pattern_only: bool,
) -> Option<CscMatrix> {
    let dim = blk.n + nb;
    let cap = blk.rows.len() + blk.cpl_b.len() + nb;
    let (mut r, mut c, mut v) = (
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
    );
    for p in 0..blk.rows.len() {
        r.push(blk.rows[p]);
        c.push(blk.cols[p]);
        v.push(if pattern_only {
            1.0
        } else {
            values[blk.src[p]]
        });
    }
    for p in 0..blk.cpl_b.len() {
        r.push(blk.n + blk.cpl_b[p]); // border row, block column: lower triangle
        c.push(blk.cpl_k[p]);
        v.push(if pattern_only {
            1.0
        } else {
            values[blk.cpl_src[p]]
        });
    }
    for k in 0..nb {
        r.push(blk.n + k);
        c.push(blk.n + k);
        v.push(0.0);
    }
    CscMatrix::from_triplets(dim, &r, &c, &v).ok()
}

/// `A_kk` alone, for the back-solve.
fn block_only_matrix(blk: &Block, values: &[Number]) -> Option<CscMatrix> {
    let v: Vec<Number> = blk.src.iter().map(|&p| values[p]).collect();
    CscMatrix::from_triplets(blk.n, &blk.rows, &blk.cols, &v).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block-diagonal-plus-border KKT: `nblocks` tridiagonal blocks of
    /// `n` each, every block coupled to all `nb` border columns.
    fn arrowhead(
        nblocks: usize,
        n: usize,
        nb: usize,
    ) -> (usize, Vec<Index>, Vec<Index>, Vec<Number>, Vec<i32>) {
        let dim = nblocks * n + nb;
        let (mut ia, mut ja, mut va) = (Vec::new(), Vec::new(), Vec::new());
        let mut lab = vec![-1i32; dim];
        for b in 0..nblocks {
            let off = b * n;
            for i in 0..n {
                lab[off + i] = b as i32;
                ia.push((off + i + 1) as Index);
                ja.push((off + i + 1) as Index);
                va.push(4.0 + (i % 3) as Number);
                if i + 1 < n {
                    ia.push((off + i + 2) as Index);
                    ja.push((off + i + 1) as Index);
                    va.push(-1.0);
                }
            }
            for k in 0..nb {
                ia.push((nblocks * n + k + 1) as Index);
                ja.push((off + (k % n) + 1) as Index);
                va.push(0.5 + k as Number * 0.01);
            }
        }
        for k in 0..nb {
            ia.push((nblocks * n + k + 1) as Index);
            ja.push((nblocks * n + k + 1) as Index);
            va.push(-2.0);
        }
        (dim, ia, ja, va, lab)
    }

    fn oracle(
        dim: usize,
        ia: &[Index],
        ja: &[Index],
        va: &[Number],
        rhs: &[Number],
    ) -> (Vec<Number>, usize) {
        let rows: Vec<usize> = ia.iter().map(|&v| v as usize - 1).collect();
        let cols: Vec<usize> = ja.iter().map(|&v| v as usize - 1).collect();
        let m = CscMatrix::from_triplets(dim, &rows, &cols, va).unwrap();
        let mut s = Solver::new();
        assert!(matches!(s.factor(&m, None), FactorStatus::Success));
        (s.solve(rhs).unwrap(), s.num_negative_eigenvalues())
    }

    #[test]
    fn matches_the_monolithic_solve_and_its_inertia() {
        let (dim, ia, ja, va, lab) = arrowhead(6, 40, 5);
        let rhs: Vec<Number> = (0..dim).map(|i| 1.0 + (i % 5) as Number * 0.25).collect();
        let (x_ref, neg_ref) = oracle(dim, &ia, &ja, &va, &rhs);

        let mut s = FeralBlockSolver::new(FeralConfig::default());
        assert_eq!(
            s.initialize_structure(dim as Index, &ia, &ja, &lab),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&va);
        assert_eq!(s.factor(true, neg_ref as Index), ESymSolverStatus::Success);
        assert_eq!(
            s.number_of_neg_evals(),
            neg_ref as Index,
            "Haynsworth inertia"
        );
        assert_eq!(s.n_blocks(), 6);
        assert_eq!(s.border_dim(), 5);

        let mut b = rhs.clone();
        assert_eq!(s.backsolve(1, &mut b), ESymSolverStatus::Success);
        let err = b
            .iter()
            .zip(&x_ref)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-9, "solution mismatch {err:e}");
    }

    /// The assembled border complement against one computed by explicit
    /// solves, `S = A_bb - sum_k A_bk A_kk^-1 A_kb`.
    ///
    /// This is the test that catches the scaling: feral factors `D·A·D`, so
    /// the Schur block it returns is scaled by the border's own factors.
    /// Inertia is a congruence invariant, so a scaled S still combines to the
    /// right pivot counts and every inertia assertion passes — only a solve
    /// notices. Drop the unscaling in `factor` and this test fails while the
    /// others do not.
    #[test]
    fn the_border_complement_matches_explicit_solves() {
        let (dim, ia, ja, va, lab) = arrowhead(3, 12, 2);
        let mut s = FeralBlockSolver::new(FeralConfig::default());
        assert_eq!(
            s.initialize_structure(dim as Index, &ia, &ja, &lab),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&va);
        assert_eq!(s.factor(false, 0), ESymSolverStatus::Success);
        let nb = s.border.len();
        let mut expect = vec![0.0f64; nb * nb];
        for (i, j, p) in &s.ss {
            expect[i + j * nb] += s.values[*p];
            if i != j {
                expect[j + i * nb] += s.values[*p];
            }
        }
        for blk in &s.blocks {
            let mut cols = vec![0.0f64; blk.n * nb];
            for p in 0..blk.cpl_b.len() {
                cols[blk.cpl_k[p] + blk.cpl_b[p] * blk.n] += s.values[blk.cpl_src[p]];
            }
            for j in 0..nb {
                let w = blk
                    .solver
                    .solve(&cols[j * blk.n..(j + 1) * blk.n].to_vec())
                    .unwrap();
                for p in 0..blk.cpl_b.len() {
                    expect[blk.cpl_b[p] + j * nb] -= s.values[blk.cpl_src[p]] * w[blk.cpl_k[p]];
                }
            }
        }
        let err = expect
            .iter()
            .zip(s.s_dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let scale = expect.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1.0);
        assert!(err < 1e-9 * scale, "border complement mismatch {err:e}");
    }

    #[test]
    fn multi_rhs_matches_single_rhs() {
        let (dim, ia, ja, va, lab) = arrowhead(4, 25, 3);
        let mut s = FeralBlockSolver::new(FeralConfig::default());
        assert_eq!(
            s.initialize_structure(dim as Index, &ia, &ja, &lab),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&va);
        assert_eq!(s.factor(false, 0), ESymSolverStatus::Success);
        let mut two: Vec<Number> = (0..2 * dim).map(|i| 1.0 + (i % 7) as Number).collect();
        let first = two[..dim].to_vec();
        assert_eq!(s.backsolve(2, &mut two), ESymSolverStatus::Success);
        let mut one = first;
        assert_eq!(s.backsolve(1, &mut one), ESymSolverStatus::Success);
        let err = one
            .iter()
            .zip(&two[..dim])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-12, "multi-RHS disagrees with single: {err:e}");
    }

    /// A declared partition the matrix contradicts (an entry joining two
    /// blocks) is rejected, so the caller can fall back rather than factor a
    /// system whose structure it has misread.
    #[test]
    fn a_block_to_block_entry_is_rejected() {
        let (dim, mut ia, mut ja, mut va, lab) = arrowhead(3, 20, 2);
        ia.push(25);
        ja.push(5); // block 1 row, block 0 column
        va.push(0.1);
        let mut s = FeralBlockSolver::new(FeralConfig::default());
        assert_eq!(
            s.initialize_structure(dim as Index, &ia, &ja, &lab),
            ESymSolverStatus::FatalError
        );
    }

    #[test]
    fn a_partition_with_no_border_is_rejected() {
        let (dim, ia, ja, _va, mut lab) = arrowhead(3, 20, 2);
        for l in lab.iter_mut() {
            if *l < 0 {
                *l = 0;
            }
        }
        let mut s = FeralBlockSolver::new(FeralConfig::default());
        assert_eq!(
            s.initialize_structure(dim as Index, &ia, &ja, &lab),
            ESymSolverStatus::FatalError
        );
    }
}
