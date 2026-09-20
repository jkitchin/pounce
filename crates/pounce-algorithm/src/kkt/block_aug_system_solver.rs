//! Block-parallel augmented-system solver (structured-KKT Phase 5b).
//!
//! Wraps [`StdAugSystemSolver`] — reusing its KKT assembly and RHS packing —
//! but routes the solve through a [`FeralBlockSolver`] over a caller-supplied
//! partition of the KKT indices into blocks plus a shared border. Every block
//! factorizes independently and in parallel; the border's Schur complement
//! carries the coupling; inertia comes from Haynsworth additivity.
//!
//! **Why.** An arrowhead KKT (scenarios, contingencies, or any blocks sharing
//! a few global columns) is already well ordered by a generic method, but its
//! factorization is per-supernode-overhead bound, so feral's tree parallelism
//! reaches ~1.2×. As coarse block tasks the same factorization is ~10× faster
//! and a whole KKT solve ~4–6×. Measured in `dev-notes/kkt-scaling-phase0b.md`.
//!
//! **Fallback is first-class**, as in [`crate::kkt::SchurAugSystemSolver`]:
//! a partition that does not match the matrix (an entry joining two blocks),
//! a singular block, or any backend error routes the rest of the solve through
//! the monolithic path, which regularizes the full system correctly. The
//! solve report's `linear_solver.blocks` is present only when the block path
//! actually factored, so a silent fallback is visible.

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use crate::kkt::aug_system_solver::{AugSysCoeffs, AugSysRhs, AugSysSol, AugSystemSolver};
use crate::kkt::std_aug_system_solver::StdAugSystemSolver;
use pounce_common::diagnostics::DiagnosticsState;
use pounce_common::timing::TimingStatistics;
use pounce_common::types::{Index, Number};
use pounce_feral::{FeralBlockSolver, FeralConfig};
use pounce_linsol::summary::LinearSolverSummary;
use pounce_linsol::{ESymSolverStatus, FactorPattern};

pub struct BlockAugSystemSolver {
    /// KKT assembly + the fallback solver.
    inner: StdAugSystemSolver,
    blocks: FeralBlockSolver,
    /// Block of each KKT index, `< 0` for the border. Empty means "detect it
    /// from the assembled matrix".
    labels: Vec<i32>,
    detect: bool,
    /// `None` until the partition has been validated against a concrete KKT
    /// dimension; `Some(dim)` records what it was pinned for.
    decided_for_dim: Option<Index>,
    use_blocks: bool,
    have_factor: bool,
    negevals: Index,
    last_status: ESymSolverStatus,
    timing: Option<Rc<TimingStatistics>>,
}

impl BlockAugSystemSolver {
    /// Wrap `inner` with a block backend over `labels` (KKT-space; `< 0` is
    /// the border). An empty `labels` asks for detection from the assembled
    /// KKT — see [`detect_blocks`].
    pub fn new(inner: StdAugSystemSolver, labels: Vec<i32>, cfg: FeralConfig) -> Self {
        Self {
            inner,
            blocks: FeralBlockSolver::new(cfg),
            detect: labels.is_empty(),
            labels,
            decided_for_dim: None,
            use_blocks: false,
            have_factor: false,
            negevals: 0,
            last_status: ESymSolverStatus::Success,
            timing: None,
        }
    }

    /// Record the block path's factorizations into a shared summary sink.
    pub fn with_summary_sink(mut self, sink: Arc<Mutex<LinearSolverSummary>>) -> Self {
        self.blocks = self.blocks.with_summary_sink(sink);
        self
    }

    /// Decide once per KKT dimension whether the block path is usable.
    fn decide(&mut self, dim: Index) {
        if self.decided_for_dim == Some(dim) {
            return;
        }
        self.decided_for_dim = Some(dim);
        self.use_blocks = false;
        let (irn, jcn) = {
            let (a, b, _v) = self.inner.assembled_triplet();
            (a.to_vec(), b.to_vec())
        };
        if self.detect {
            let t = std::time::Instant::now();
            let found = detect_blocks(dim as usize, &irn, &jcn);
            tracing::info!(
                target: "pounce::kkt",
                secs = t.elapsed().as_secs_f64(),
                blocks = found.as_ref().map(|l: &Vec<i32>| l.iter().copied().max().unwrap_or(-1) + 1),
                border = found.as_ref().map(|l: &Vec<i32>| l.iter().filter(|&&x| x < 0).count()),
                "block detection"
            );
            match found {
                Some(l) => self.labels = l,
                None => {
                    tracing::warn!(
                        target: "pounce::kkt",
                        "no block structure found in the KKT; using the standard solver"
                    );
                    return;
                }
            }
        }
        if self.labels.len() != dim as usize {
            tracing::warn!(
                target: "pounce::kkt",
                labels = self.labels.len(), dim = dim as usize,
                "block structure has the wrong length for this KKT; using the standard solver"
            );
            return;
        }
        // Every block's symbolic analysis carries the whole border as its
        // Schur tail, and the border's complement is dense, so a big border
        // costs `n_blocks * border^2` memory and more than the monolithic
        // factorization in time. Refuse rather than crawl.
        let border = self.labels.iter().filter(|&&l| l < 0).count();
        let max_border = (dim as usize / 200).clamp(64, 4096);
        if border > max_border {
            tracing::warn!(
                target: "pounce::kkt",
                border, max_border,
                "block structure's border is too large to pay off; using the standard solver"
            );
            return;
        }
        let t = std::time::Instant::now();
        let st = self
            .blocks
            .initialize_structure(dim, &irn, &jcn, &self.labels);
        tracing::info!(
            target: "pounce::kkt", secs = t.elapsed().as_secs_f64(), "block structure initialized"
        );
        if st == ESymSolverStatus::Success {
            self.use_blocks = true;
            tracing::info!(
                target: "pounce::kkt",
                blocks = self.blocks.n_blocks(), border = self.blocks.border_dim(),
                "block-parallel KKT path engaged"
            );
        } else {
            tracing::warn!(
                target: "pounce::kkt",
                "block structure rejected by the backend; using the standard solver"
            );
        }
    }

    /// Factor + block back-solve for one RHS, assuming `inner` has assembled.
    fn block_solve_one(
        &mut self,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        let dim = self.inner.assembled_dim() as usize;
        let vals = self.inner.assembled_triplet().2.to_vec();
        self.blocks.values_array_mut().copy_from_slice(&vals);
        let status = {
            let _g = self
                .timing
                .as_deref()
                .map(|t| t.linear_system_factorization.guard());
            self.blocks.factor(check_neg_evals, num_neg_evals)
        };
        self.last_status = status;
        match status {
            ESymSolverStatus::Success => {
                self.negevals = self.blocks.number_of_neg_evals();
                let mut packed = vec![0.0; dim];
                self.inner.pack_rhs(rhs, &mut packed);
                let bstat = {
                    let _g = self
                        .timing
                        .as_deref()
                        .map(|t| t.linear_system_back_solve.guard());
                    self.blocks.backsolve(1, &mut packed)
                };
                if bstat != ESymSolverStatus::Success {
                    self.have_factor = false;
                    self.last_status = bstat;
                    return bstat;
                }
                self.inner.unpack_sol(&packed, sol);
                self.have_factor = true;
                ESymSolverStatus::Success
            }
            // The δ-perturbation loop reaches every block and the border, so a
            // wrong combined inertia is correctable by a re-factor, exactly as
            // on the monolithic path.
            ESymSolverStatus::WrongInertia => {
                self.negevals = self.blocks.number_of_neg_evals();
                self.have_factor = false;
                status
            }
            // Singular is surfaced, not fallen back on — unlike the Schur arm,
            // where a rank-deficient *eliminated* block is outside δ_c's reach.
            // Here every dual row lives inside a block, so `perturb_for_singular`
            // bumping δ_c reaches it and the next factor succeeds. A KKT whose
            // structurally empty constraint rows carry only δ_c on the diagonal
            // reads as singular on the monolithic path too (measured: 2 240 zero
            // pivots on a 64-contingency SCOPF), and that path reports rather
            // than gives up.
            ESymSolverStatus::Singular => {
                self.have_factor = false;
                status
            }
            other => {
                self.have_factor = false;
                other
            }
        }
    }
}

/// Find a block-diagonal-plus-border partition in an assembled KKT, or `None`.
///
/// The border of an arrowhead KKT is its high-degree columns: every block
/// touches them, so their degree grows with the block count while a block's own
/// columns keep a fixed degree. Peel the top-`k` by degree for geometrically
/// growing `k`, and accept the first `k` that leaves a partition worth having.
///
/// "Worth having" is deliberately strict, because engaging this path on a
/// matrix that merely *can* be cut buys nothing and costs a border solve:
///
/// * the peeled columns must stand out — the smallest peeled degree at least
///   `DEGREE_GAP` times the median degree. A chain has no such gap, and
///   removing any interior column of a chain splits it, so without this a
///   plain tridiagonal matrix "detects" as blocks;
/// * at least `MIN_BLOCKS` blocks, and a border under 5% of the matrix;
/// * no block holding half the interior.
///
/// A separating set found this way can be larger than necessary (the peel is
/// geometric), so it is then shrunk: a peeled column whose neighbours all lie
/// in one block belongs in that block.
///
/// Measured on a 210k-variable N-1 SCOPF this recovers the model's own
/// structure exactly — 259 shared columns, 65 blocks — without being told.
///
/// `irn` / `jcn` are the 1-based lower-triangle triplet.
pub fn detect_blocks(dim: usize, irn: &[Index], jcn: &[Index]) -> Option<Vec<i32>> {
    const DEGREE_GAP: u32 = 3;
    const MIN_BLOCKS: usize = 4;
    if dim < 4 * MIN_BLOCKS {
        return None;
    }
    let mut deg = vec![0u32; dim];
    for p in 0..irn.len() {
        let (r, c) = (irn[p] as usize - 1, jcn[p] as usize - 1);
        if r != c {
            deg[r] += 1;
            deg[c] += 1;
        }
    }
    let median = {
        let mut d = deg.clone();
        let mid = d.len() / 2;
        d.select_nth_unstable(mid);
        d[mid].max(1)
    };
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(deg[i]));
    let max_border = (dim / 20).max(1);
    // `MIN_BLOCKS` constrains the partition that comes out, not the peel that
    // goes in: an arrowhead with many blocks can still share very few columns.
    let mut k = 1usize;
    while k <= max_border {
        // the peeled set has to be structurally special, not merely first
        if deg[order[k - 1]] < DEGREE_GAP * median {
            return None;
        }
        let mut is_border = vec![false; dim];
        for &i in order.iter().take(k) {
            is_border[i] = true;
        }
        if let Some(mut labels) = components_without(dim, irn, jcn, &is_border, MIN_BLOCKS) {
            shrink_border(dim, irn, jcn, &mut labels);
            merge_small_blocks(&mut labels);
            return Some(labels);
        }
        k *= 2;
    }
    None
}

/// Connected components of the KKT graph minus `is_border`, as block labels,
/// or `None` when that is not a partition worth having.
fn components_without(
    dim: usize,
    irn: &[Index],
    jcn: &[Index],
    is_border: &[bool],
    min_blocks: usize,
) -> Option<Vec<i32>> {
    let mut parent: Vec<usize> = (0..dim).collect();
    fn find(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    for p in 0..irn.len() {
        let (r, c) = (irn[p] as usize - 1, jcn[p] as usize - 1);
        if r == c || is_border[r] || is_border[c] {
            continue;
        }
        let (ra, rb) = (find(&mut parent, r), find(&mut parent, c));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut label = vec![-1i32; dim];
    let mut root_to_block = std::collections::HashMap::new();
    let mut sizes: Vec<usize> = Vec::new();
    for i in 0..dim {
        if is_border[i] {
            continue;
        }
        let r = find(&mut parent, i);
        let b = *root_to_block.entry(r).or_insert_with(|| {
            sizes.push(0);
            sizes.len() - 1
        });
        label[i] = b as i32;
        sizes[b] += 1;
    }
    let interior: usize = sizes.iter().sum();
    if sizes.len() < min_blocks || interior == 0 {
        return None;
    }
    if *sizes.iter().max().unwrap() * 2 >= interior {
        return None; // one block holding half of everything is not a partition
    }
    Some(label)
}

/// Fold under-sized components into the smallest real block.
///
/// A KKT carries rows that couple to nothing — a constraint left structurally
/// empty by presolve keeps only its dual regularization on the diagonal — and
/// each becomes its own component. On a 64-contingency SCOPF there are 2 240 of
/// them beside 64 blocks of 1 320. They must not go to the border, which would
/// grow it from 32 to 2 272 columns and make its dense complement cost more
/// than the whole factorization; and being isolated, they can join any block
/// without creating block-to-block coupling.
fn merge_small_blocks(labels: &mut [i32]) {
    let nblocks = labels.iter().copied().max().unwrap_or(-1) + 1;
    if nblocks <= 0 {
        return;
    }
    let mut sizes = vec![0usize; nblocks as usize];
    for &l in labels.iter() {
        if l >= 0 {
            sizes[l as usize] += 1;
        }
    }
    let interior: usize = sizes.iter().sum();
    let real: Vec<usize> = (0..sizes.len())
        .filter(|&b| sizes[b] * sizes.len() * 4 >= interior)
        .collect();
    if real.is_empty() || real.len() == sizes.len() {
        return;
    }
    // remap: real blocks keep an id; everything else joins the smallest one
    let mut new_id = vec![usize::MAX; sizes.len()];
    let mut load: Vec<usize> = Vec::new();
    for (k, &b) in real.iter().enumerate() {
        new_id[b] = k;
        load.push(sizes[b]);
    }
    for b in 0..sizes.len() {
        if new_id[b] == usize::MAX {
            let smallest = load
                .iter()
                .enumerate()
                .min_by_key(|&(_, l)| *l)
                .map(|(i, _)| i)
                .unwrap_or(0);
            new_id[b] = smallest;
            load[smallest] += sizes[b];
        }
    }
    for l in labels.iter_mut() {
        if *l >= 0 {
            *l = new_id[*l as usize] as i32;
        }
    }
}

/// Give back every peeled column whose neighbours all lie in one block: the
/// geometric peel overshoots, and a border column that separates nothing only
/// makes the border system bigger.
///
/// Worklist, not a fixed point over the whole matrix: a column is re-examined
/// only when one of its border neighbours is resolved, so the pass is linear
/// in the nonzeros rather than (border × nonzeros), which on a 55k-variable
/// SCOPF was the difference between milliseconds and not finishing.
fn shrink_border(dim: usize, irn: &[Index], jcn: &[Index], labels: &mut [i32]) {
    // adjacency of the border columns only
    let mut deg = vec![0usize; dim];
    for p in 0..irn.len() {
        let (r, c) = (irn[p] as usize - 1, jcn[p] as usize - 1);
        if r != c && (labels[r] < 0 || labels[c] < 0) {
            deg[r] += 1;
            deg[c] += 1;
        }
    }
    let mut start = vec![0usize; dim + 1];
    for i in 0..dim {
        start[i + 1] = start[i] + deg[i];
    }
    let mut adj = vec![0usize; start[dim]];
    let mut fill = start.clone();
    for p in 0..irn.len() {
        let (r, c) = (irn[p] as usize - 1, jcn[p] as usize - 1);
        if r != c && (labels[r] < 0 || labels[c] < 0) {
            adj[fill[r]] = c;
            fill[r] += 1;
            adj[fill[c]] = r;
            fill[c] += 1;
        }
    }
    let mut queue: Vec<usize> = (0..dim).filter(|&i| labels[i] < 0).collect();
    let mut queued = vec![true; dim];
    for i in 0..dim {
        queued[i] = labels[i] < 0;
    }
    let mut head = 0usize;
    while head < queue.len() {
        let i = queue[head];
        head += 1;
        queued[i] = false;
        if labels[i] >= 0 {
            continue;
        }
        let mut seen: Option<i32> = None;
        let mut separates = false;
        for &j in &adj[start[i]..start[i + 1]] {
            if labels[j] < 0 {
                continue; // another border column: decided later, if at all
            }
            match seen {
                None => seen = Some(labels[j]),
                Some(b) if b == labels[j] => {}
                Some(_) => {
                    separates = true;
                    break;
                }
            }
        }
        if separates {
            continue;
        }
        if let Some(b) = seen {
            labels[i] = b;
            for &j in &adj[start[i]..start[i + 1]] {
                if labels[j] < 0 && !queued[j] {
                    queued[j] = true;
                    queue.push(j);
                }
            }
        }
    }
}

impl AugSystemSolver for BlockAugSystemSolver {
    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }

    fn number_of_neg_evals(&self) -> Index {
        if self.use_blocks {
            self.negevals
        } else {
            self.inner.number_of_neg_evals()
        }
    }

    fn system_dim(&self) -> Index {
        self.inner.system_dim()
    }

    fn kkt_triplets(&self) -> Option<(Index, Vec<Index>, Vec<Index>, Vec<Number>)> {
        self.inner.kkt_triplets()
    }

    fn l_factor(&self, want_values: bool) -> Option<FactorPattern> {
        // No single monolithic L on the block path.
        if self.use_blocks {
            None
        } else {
            self.inner.l_factor(want_values)
        }
    }

    fn increase_quality(&mut self) -> bool {
        self.have_factor = false;
        if self.use_blocks {
            self.blocks.increase_quality()
        } else {
            self.inner.increase_quality()
        }
    }

    fn last_solve_status(&self) -> ESymSolverStatus {
        if self.use_blocks {
            self.last_status
        } else {
            self.inner.last_solve_status()
        }
    }

    fn set_timing_stats(&mut self, timing: Rc<TimingStatistics>) {
        self.timing = Some(Rc::clone(&timing));
        self.inner.set_timing_stats(timing);
    }

    fn set_diagnostics(&mut self, diag: Rc<DiagnosticsState>) {
        self.inner.set_diagnostics(diag);
    }

    fn solve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        let s = self.inner.assemble(coeffs);
        if s != ESymSolverStatus::Success {
            self.last_status = s;
            return s;
        }
        let dim = self.inner.assembled_dim();
        self.decide(dim);

        if self.use_blocks {
            let st = self.block_solve_one(rhs, sol, check_neg_evals, num_neg_evals);
            match st {
                ESymSolverStatus::Success
                | ESymSolverStatus::WrongInertia
                | ESymSolverStatus::Singular => return st,
                _ => {
                    tracing::warn!(
                        target: "pounce::kkt",
                        status = ?st,
                        "block backend could not factor this KKT; falling back to the standard solver"
                    );
                    self.use_blocks = false;
                    return self
                        .inner
                        .solve(coeffs, rhs, sol, check_neg_evals, num_neg_evals);
                }
            }
        }
        self.inner
            .solve(coeffs, rhs, sol, check_neg_evals, num_neg_evals)
    }

    fn resolve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
    ) -> ESymSolverStatus {
        if self.use_blocks {
            if self.have_factor {
                let dim = self.inner.assembled_dim() as usize;
                let mut packed = vec![0.0; dim];
                self.inner.pack_rhs(rhs, &mut packed);
                let bstat = {
                    let _g = self
                        .timing
                        .as_deref()
                        .map(|t| t.linear_system_back_solve.guard());
                    self.blocks.backsolve(1, &mut packed)
                };
                if bstat == ESymSolverStatus::Success {
                    self.inner.unpack_sol(&packed, sol);
                }
                return bstat;
            }
            return self.solve(coeffs, rhs, sol, false, 0);
        }
        self.inner.resolve(coeffs, rhs, sol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tridiagonal blocks sharing two border columns, as a lower-triangle
    /// triplet.
    fn arrowhead() -> (usize, Vec<Index>, Vec<Index>) {
        let (n, nb, nblocks) = (10usize, 2usize, 8usize);
        let dim = nblocks * n + nb;
        let (mut ia, mut ja) = (Vec::new(), Vec::new());
        for b in 0..nblocks {
            let off = b * n;
            for i in 0..n {
                ia.push((off + i + 1) as Index);
                ja.push((off + i + 1) as Index);
                if i + 1 < n {
                    ia.push((off + i + 2) as Index);
                    ja.push((off + i + 1) as Index);
                }
            }
            for k in 0..nb {
                ia.push((nblocks * n + k + 1) as Index);
                ja.push((off + k + 1) as Index);
            }
        }
        for k in 0..nb {
            ia.push((nblocks * n + k + 1) as Index);
            ja.push((nblocks * n + k + 1) as Index);
        }
        (dim, ia, ja)
    }

    #[test]
    fn detection_finds_the_blocks_and_the_border() {
        let (dim, ia, ja) = arrowhead();
        let labels = detect_blocks(dim, &ia, &ja).expect("structure found");
        assert_eq!(labels.len(), dim);
        assert_eq!(labels.iter().filter(|&&l| l < 0).count(), 2, "border");
        let mut per = std::collections::HashMap::new();
        for &l in labels.iter().filter(|&&l| l >= 0) {
            *per.entry(l).or_insert(0) += 1;
        }
        assert_eq!(per.len(), 8, "eight blocks");
        assert!(per.values().all(|&c| c == 10), "blocks of ten: {per:?}");
    }

    /// A chain has no high-degree border, and cutting any interior column
    /// splits it — so a degree peel alone would "find" blocks in it. Detection
    /// must decline: engaging the block path here buys nothing and costs a
    /// border solve.
    #[test]
    fn detection_declines_a_matrix_without_structure() {
        let n = 400usize;
        let (mut ia, mut ja) = (Vec::new(), Vec::new());
        for i in 0..n {
            ia.push((i + 1) as Index);
            ja.push((i + 1) as Index);
            if i + 1 < n {
                ia.push((i + 2) as Index);
                ja.push((i + 1) as Index);
            }
        }
        assert!(detect_blocks(n, &ia, &ja).is_none());
    }
}
