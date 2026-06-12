//! Postsolve frame stack for the auxiliary-equality preprocessing
//! pass.
//!
//! PR 7 of the auxiliary-presolve port (issue #53). A
//! [`ReductionFrame`] captures one layer of variable + row
//! elimination:
//!
//! - `fixed_vars` — block variables fixed by the block solve.
//! - `fixed_values` — their values at the fixed point.
//! - `dropped_rows` — equality rows used to determine them.
//! - `var_map / row_map` — index maps between full and reduced space.
//!
//! The headline method is
//! [`ReductionFrame::recover_dropped_multipliers`], which solves the
//! full-space KKT stationarity equations at the fixed variables for
//! the missing multipliers. Assumption (matching ripopt v1): fixed
//! variables are interior to their original bounds at the optimum,
//! so `z_l = z_u = 0` for them.
//!
//! ripopt anchor: `src/reduction_frame.rs:101-231`.

use pounce_common::types::Number;

use crate::block_solve::{lu_factor_partial_pivot, lu_solve, BlockSolveError};

/// One layer of the postsolve stack. Built once per accepted block
/// elimination by PR 8's orchestrator.
#[derive(Debug, Default, Clone)]
pub struct ReductionFrame {
    /// Inner-variable indices fixed by this layer, in ascending order.
    pub fixed_vars: Vec<usize>,
    /// Their values at the block-solve fixed point.
    pub fixed_values: Vec<Number>,
    /// Inner equality-row indices dropped by this layer, in
    /// ascending order. `dropped_rows.len() == fixed_vars.len()`.
    pub dropped_rows: Vec<usize>,
    /// `var_map[i] = Some(reduced_idx)` if `i` survives this layer,
    /// `None` if `i` is in `fixed_vars`.
    pub var_map: Vec<Option<usize>>,
    /// Same for rows.
    pub row_map: Vec<Option<usize>>,
}

impl ReductionFrame {
    /// Build a frame from the (sorted) lists of fixed variables /
    /// values / dropped rows and the **full-space** problem shape.
    pub fn new(
        n_vars: usize,
        n_rows: usize,
        fixed_vars: Vec<usize>,
        fixed_values: Vec<Number>,
        dropped_rows: Vec<usize>,
    ) -> Self {
        assert_eq!(
            fixed_vars.len(),
            fixed_values.len(),
            "fixed_vars and fixed_values must be the same length"
        );
        assert_eq!(
            fixed_vars.len(),
            dropped_rows.len(),
            "fixed_vars and dropped_rows must be the same length (square block)"
        );

        // Mark fixed positions on flat `bool` vectors (O(1) lookup);
        // BTreeSet would cost O(log k) per probe. PR review #60.
        let mut is_fixed_var = vec![false; n_vars];
        for &i in &fixed_vars {
            is_fixed_var[i] = true;
        }
        let mut is_dropped_row = vec![false; n_rows];
        for &i in &dropped_rows {
            is_dropped_row[i] = true;
        }

        let mut var_map = vec![None; n_vars];
        let mut next_reduced = 0;
        for (i, slot) in var_map.iter_mut().enumerate().take(n_vars) {
            if is_fixed_var[i] {
                continue;
            }
            *slot = Some(next_reduced);
            next_reduced += 1;
        }

        let mut row_map = vec![None; n_rows];
        let mut next_reduced_row = 0;
        for (i, slot) in row_map.iter_mut().enumerate().take(n_rows) {
            if is_dropped_row[i] {
                continue;
            }
            *slot = Some(next_reduced_row);
            next_reduced_row += 1;
        }

        Self {
            fixed_vars,
            fixed_values,
            dropped_rows,
            var_map,
            row_map,
        }
    }

    pub fn n_full_vars(&self) -> usize {
        self.var_map.len()
    }

    pub fn n_full_rows(&self) -> usize {
        self.row_map.len()
    }

    pub fn n_reduced_vars(&self) -> usize {
        self.n_full_vars() - self.fixed_vars.len()
    }

    pub fn n_reduced_rows(&self) -> usize {
        self.n_full_rows() - self.dropped_rows.len()
    }

    /// Project a full-space `x` vector into reduced space (drop the
    /// fixed entries).
    pub fn project_x(&self, x_full: &[Number]) -> Vec<Number> {
        assert_eq!(x_full.len(), self.n_full_vars());
        self.var_map
            .iter()
            .zip(x_full.iter())
            .filter_map(|(slot, &v)| slot.map(|_| v))
            .collect()
    }

    /// Lift a reduced `x` back to full space, splicing the fixed
    /// values back into their original positions.
    pub fn lift_x(&self, x_reduced: &[Number]) -> Vec<Number> {
        assert_eq!(x_reduced.len(), self.n_reduced_vars());
        let mut out = vec![0.0; self.n_full_vars()];
        for (i, slot) in self.var_map.iter().enumerate() {
            if let Some(r) = slot {
                out[i] = x_reduced[*r];
            }
        }
        for (k, &i) in self.fixed_vars.iter().enumerate() {
            out[i] = self.fixed_values[k];
        }
        out
    }

    /// Project a full-space λ vector into reduced space.
    pub fn project_lambda(&self, lambda_full: &[Number]) -> Vec<Number> {
        assert_eq!(lambda_full.len(), self.n_full_rows());
        self.row_map
            .iter()
            .zip(lambda_full.iter())
            .filter_map(|(slot, &v)| slot.map(|_| v))
            .collect()
    }

    /// Lift a reduced λ back to full space, with zeros at dropped
    /// row indices. (Real values for dropped rows come from
    /// [`Self::recover_dropped_multipliers`].)
    pub fn lift_lambda(&self, lambda_reduced: &[Number]) -> Vec<Number> {
        assert_eq!(lambda_reduced.len(), self.n_reduced_rows());
        let mut out = vec![0.0; self.n_full_rows()];
        for (i, slot) in self.row_map.iter().enumerate() {
            if let Some(r) = slot {
                out[i] = lambda_reduced[*r];
            }
        }
        out
    }

    /// Recover the `k = fixed_vars.len()` dropped-row multipliers
    /// via dense LU on the full-space KKT stationarity equations at
    /// the fixed variables. Returns one entry per `self.dropped_rows`
    /// (in the same order).
    ///
    /// Assumption: fixed variables are interior to their original
    /// bounds at the optimum (so `z_l = z_u = 0` for them).
    ///
    /// # Inputs
    ///
    /// - `grad_f` — objective gradient at the full-space optimum
    ///   (length `n_full_vars`).
    /// - `jac_full_row_major` — dense full-space Jacobian
    ///   `(n_full_rows × n_full_vars)` at the optimum.
    /// - `lambda_full` — multipliers for kept rows; entries at
    ///   dropped-row positions are ignored.
    ///
    /// # Example
    ///
    /// ```
    /// use pounce_presolve::reduction_frame::ReductionFrame;
    ///
    /// // 1 var, 1 row, dropped:  c(x) = x - 3 = 0, obj f = 4 x.
    /// // Stationarity:  4 - 1 * λ = 0  →  λ = 4.
    /// let frame = ReductionFrame::new(1, 1, vec![0], vec![3.0], vec![0]);
    /// let grad_f = [4.0];
    /// let jac = [1.0];
    /// let lambda_full = [0.0]; // dropped, ignored
    /// let lam = frame
    ///     .recover_dropped_multipliers(&grad_f, &jac, &lambda_full)
    ///     .unwrap();
    /// assert!((lam[0] - 4.0).abs() < 1e-12);
    /// ```
    pub fn recover_dropped_multipliers(
        &self,
        grad_f: &[Number],
        jac_full_row_major: &[Number],
        lambda_full: &[Number],
    ) -> Result<Vec<Number>, BlockSolveError> {
        let n_vars = self.n_full_vars();
        let n_rows = self.n_full_rows();
        assert_eq!(
            jac_full_row_major.len(),
            n_rows * n_vars,
            "jac_full_row_major length mismatch"
        );
        // The recovery reads the Jacobian only at columns `i ∈ fixed_vars`,
        // so the full dense layout is just one indexing convention; see
        // `recover_dropped_multipliers_cols` for the column-compacted one.
        self.recover_core(grad_f, lambda_full, |row, col| {
            jac_full_row_major[row * n_vars + col]
        })
    }

    /// Same recovery as [`recover_dropped_multipliers`], but reads the
    /// Jacobian from a **column-compacted** dense buffer that holds only a
    /// subset of the full columns. The recovery touches the Jacobian only at
    /// the frame's `fixed_vars` columns (it never reads any other column), so
    /// a caller that knows which columns matter can materialize an
    /// `(n_full_rows × n_cols)` block instead of the full
    /// `(n_full_rows × n_full_vars)` one — the difference between O(m·k) and
    /// O(m·n) memory when `k = |fixed_vars|` is tiny next to `n` (issue M26).
    ///
    /// - `jac_cols_row_major` — dense `(n_full_rows × n_cols)`, row-major.
    /// - `n_cols` — number of compacted columns (the row stride).
    /// - `orig_to_compact` — length `n_full_vars`; maps an original column
    ///   index to its position in the compacted buffer. Only the entries at
    ///   `fixed_vars` are ever read, and the caller must place those columns
    ///   in the buffer; entries for absent columns may be any value.
    pub fn recover_dropped_multipliers_cols(
        &self,
        grad_f: &[Number],
        jac_cols_row_major: &[Number],
        n_cols: usize,
        orig_to_compact: &[usize],
        lambda_full: &[Number],
    ) -> Result<Vec<Number>, BlockSolveError> {
        let n_rows = self.n_full_rows();
        assert_eq!(
            jac_cols_row_major.len(),
            n_rows * n_cols,
            "jac_cols_row_major length mismatch"
        );
        assert_eq!(
            orig_to_compact.len(),
            self.n_full_vars(),
            "orig_to_compact length mismatch"
        );
        self.recover_core(grad_f, lambda_full, |row, col| {
            jac_cols_row_major[row * n_cols + orig_to_compact[col]]
        })
    }

    /// Shared core of the multiplier recovery. `get(row, col)` returns the
    /// full-space Jacobian entry `J[row][col]`; the two public wrappers differ
    /// only in how they lay out the matrix behind that accessor. `get` is
    /// invoked exclusively at `col ∈ fixed_vars`.
    fn recover_core(
        &self,
        grad_f: &[Number],
        lambda_full: &[Number],
        get: impl Fn(usize, usize) -> Number,
    ) -> Result<Vec<Number>, BlockSolveError> {
        let n_rows = self.n_full_rows();
        let k = self.fixed_vars.len();
        assert_eq!(grad_f.len(), self.n_full_vars(), "grad_f length mismatch");
        assert_eq!(lambda_full.len(), n_rows, "lambda_full length mismatch");

        if k == 0 {
            return Ok(Vec::new());
        }

        // Use `row_map` for O(1) "is row r dropped?" — set in
        // `new()`, no BTreeSet needed (PR review #60).
        // Build the k×k system M λ_dropped = rhs.
        //   M[i_idx][j_idx] = J[dropped_rows[j_idx]][fixed_vars[i_idx]]
        //   rhs[i_idx] = grad_f[fixed_vars[i_idx]]
        //              - Σ_{r kept} J[r][fixed_vars[i_idx]] * lambda_full[r]
        let mut matrix = vec![0.0; k * k];
        for (i_idx, &i) in self.fixed_vars.iter().enumerate() {
            for (j_idx, &dr) in self.dropped_rows.iter().enumerate() {
                matrix[i_idx * k + j_idx] = get(dr, i);
            }
        }

        let mut rhs = vec![0.0; k];
        for (i_idx, &i) in self.fixed_vars.iter().enumerate() {
            let mut sum = 0.0;
            for r in 0..n_rows {
                if self.row_map[r].is_none() {
                    // Row was dropped.
                    continue;
                }
                sum += get(r, i) * lambda_full[r];
            }
            rhs[i_idx] = grad_f[i] - sum;
        }

        let piv = lu_factor_partial_pivot(&mut matrix, k).map_err(|_| BlockSolveError::Singular)?;
        lu_solve(&matrix, &piv, &mut rhs, k);
        Ok(rhs)
    }
}

/// LIFO stack of `ReductionFrame`s. Bottom-most frame represents the
/// first elimination layer applied; top-most is the most recent.
/// `finalize_solution` lifts from top to bottom.
#[derive(Debug, Default, Clone)]
pub struct ReductionStack {
    frames: Vec<ReductionFrame>,
}

impl ReductionStack {
    /// True when no reduction has been pushed (the no-op fast path).
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Number of layers currently on the stack.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Push a frame onto the stack (most-recent end).
    pub fn push(&mut self, frame: ReductionFrame) {
        self.frames.push(frame);
    }

    /// Reference to the most-recently-pushed frame, if any.
    pub fn top(&self) -> Option<&ReductionFrame> {
        self.frames.last()
    }

    /// Iterate frames from top (most recent) to bottom (first). PR 8
    /// uses this order when lifting a reduced solution back to the
    /// original full space.
    pub fn iter_top_down(&self) -> impl Iterator<Item = &ReductionFrame> {
        self.frames.iter().rev()
    }

    /// Iterate frames in push order (bottom to top). Useful when
    /// projecting full → reduced through the layers in the same
    /// order they were applied.
    pub fn iter_bottom_up(&self) -> impl Iterator<Item = &ReductionFrame> {
        self.frames.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_new_builds_maps_correctly() {
        // 4 vars, 3 rows. fixed_vars=[1], dropped_rows=[0].
        let frame = ReductionFrame::new(4, 3, vec![1], vec![42.0], vec![0]);
        // var_map: [Some(0), None, Some(1), Some(2)]
        assert_eq!(frame.var_map, vec![Some(0), None, Some(1), Some(2)]);
        // row_map: [None, Some(0), Some(1)]
        assert_eq!(frame.row_map, vec![None, Some(0), Some(1)]);
        assert_eq!(frame.n_reduced_vars(), 3);
        assert_eq!(frame.n_reduced_rows(), 2);
    }

    #[test]
    fn frame_project_x_drops_fixed() {
        let frame = ReductionFrame::new(3, 1, vec![1], vec![20.0], vec![0]);
        let x_full = [10.0, 20.0, 30.0];
        assert_eq!(frame.project_x(&x_full), vec![10.0, 30.0]);
    }

    #[test]
    fn frame_lift_x_splices_fixed_values() {
        let frame = ReductionFrame::new(3, 1, vec![1], vec![20.0], vec![0]);
        let x_reduced = [10.0, 30.0];
        assert_eq!(frame.lift_x(&x_reduced), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn frame_project_lift_x_roundtrip() {
        let frame = ReductionFrame::new(4, 2, vec![0, 2], vec![1.0, 9.0], vec![0, 1]);
        let x_full = [1.0, 5.0, 9.0, 7.0];
        let reduced = frame.project_x(&x_full);
        let lifted = frame.lift_x(&reduced);
        assert_eq!(lifted, x_full);
    }

    #[test]
    fn frame_project_lambda_drops_dropped() {
        let frame = ReductionFrame::new(3, 3, vec![1], vec![20.0], vec![0]);
        let lambda_full = [1.0, 2.0, 3.0];
        assert_eq!(frame.project_lambda(&lambda_full), vec![2.0, 3.0]);
    }

    #[test]
    fn frame_lift_lambda_zeros_dropped() {
        let frame = ReductionFrame::new(3, 3, vec![1], vec![20.0], vec![0]);
        let lambda_reduced = [2.0, 3.0];
        assert_eq!(frame.lift_lambda(&lambda_reduced), vec![0.0, 2.0, 3.0]);
    }

    #[test]
    fn recover_multipliers_singleton_linear() {
        // 1 var, 1 row. c(x) = x - 3 = 0, f = 4 x.
        // Stationarity: 4 - 1 * λ = 0 → λ = 4.
        let frame = ReductionFrame::new(1, 1, vec![0], vec![3.0], vec![0]);
        let lam = frame
            .recover_dropped_multipliers(&[4.0], &[1.0], &[0.0])
            .unwrap();
        assert_eq!(lam.len(), 1);
        assert!((lam[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn recover_multipliers_2x2_linear() {
        // 2 vars, 2 rows, both dropped.
        // J = [[1, 0], [1, 1]]
        // grad_f = [2, 5]
        // Stationarity (per fixed var i):
        //   i=0: 2 - 1*λ0 - 1*λ1 = 0
        //   i=1: 5 - 0*λ0 - 1*λ1 = 0
        // → λ1 = 5, then λ0 = 2 - 5 = -3.
        //
        // Note our system is M λ = rhs with
        //   M[i][j] = J[dropped[j]][fixed[i]]
        //   M = [[1, 1], [0, 1]]
        //   rhs = grad_f - 0 (no kept rows) = [2, 5]
        // Solving M λ = rhs:
        //   row 0: λ0 + λ1 = 2
        //   row 1:         λ1 = 5
        //   → λ1 = 5, λ0 = -3. ✓
        let frame = ReductionFrame::new(2, 2, vec![0, 1], vec![1.0, 2.0], vec![0, 1]);
        let jac = [1.0, 0.0, 1.0, 1.0]; // row-major
        let grad_f = [2.0, 5.0];
        let lam = frame
            .recover_dropped_multipliers(&grad_f, &jac, &[0.0, 0.0])
            .unwrap();
        assert!((lam[0] - (-3.0)).abs() < 1e-12, "λ0 was {}", lam[0]);
        assert!((lam[1] - 5.0).abs() < 1e-12, "λ1 was {}", lam[1]);
    }

    #[test]
    fn recover_multipliers_with_kept_rows() {
        // 2 vars, 2 rows. Row 0 dropped, row 1 kept.
        // fixed_vars = [0]. Only x_0 is fixed.
        // J = [[2,  3],   ← dropped (touches fixed var x_0 with J=2)
        //      [4,  5]]   ← kept (touches x_0 with J=4, λ_kept = 0.5)
        // grad_f[0] = 10.
        // Stationarity at x_0: 10 - 2 * λ_dropped - 4 * 0.5 = 0
        //   → 10 - 2 λ_dropped - 2 = 0 → λ_dropped = 4.
        let frame = ReductionFrame::new(2, 2, vec![0], vec![1.0], vec![0]);
        let jac = [2.0, 3.0, 4.0, 5.0];
        let grad_f = [10.0, 0.0];
        let lambda_full = [0.0, 0.5]; // entry 0 ignored
        let lam = frame
            .recover_dropped_multipliers(&grad_f, &jac, &lambda_full)
            .unwrap();
        assert_eq!(lam.len(), 1);
        assert!((lam[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn recover_multipliers_singular_block_jacobian() {
        // 2x2 with rank-1 block Jacobian.
        let frame = ReductionFrame::new(2, 2, vec![0, 1], vec![0.0, 0.0], vec![0, 1]);
        let jac = [1.0, 2.0, 2.0, 4.0]; // rank-1
        let grad_f = [1.0, 2.0];
        let err = frame
            .recover_dropped_multipliers(&grad_f, &jac, &[0.0, 0.0])
            .unwrap_err();
        assert_eq!(err, BlockSolveError::Singular);
    }

    #[test]
    fn recover_only_reads_fixed_var_columns() {
        // M26 verification: the recovery indexes the Jacobian solely at the
        // frame's `fixed_vars` columns, so a caller need only materialize
        // those columns. Poison every *non-fixed* column with NaN and confirm
        // the recovered multipliers are byte-for-byte identical to the clean
        // run — proving the densified non-fixed columns are never read.
        //
        // 3 vars, 3 rows. fixed_vars = [0, 2], dropped_rows = [0, 1];
        // var 1 is free, so column 1 is the one allowed to be garbage.
        let frame = ReductionFrame::new(3, 3, vec![0, 2], vec![1.0, 2.0], vec![0, 1]);
        let grad_f = [10.0, 4.0, 7.0];
        let lambda_full = [0.0, 0.0, 0.5]; // row 2 kept
        let clean = [
            2.0, 1.0, 0.5, // row 0
            1.0, -1.0, 3.0, // row 1
            0.4, 1.0, 0.9, // row 2 (kept)
        ];
        let expected = frame
            .recover_dropped_multipliers(&grad_f, &clean, &lambda_full)
            .unwrap();

        let mut poisoned = clean;
        for r in 0..3 {
            poisoned[r * 3 + 1] = Number::NAN; // column 1 = the free var
        }
        let got = frame
            .recover_dropped_multipliers(&grad_f, &poisoned, &lambda_full)
            .unwrap();

        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(g.is_finite(), "recovered multiplier went NaN: {g}");
            assert_eq!(g.to_bits(), e.to_bits(), "got {g}, expected {e}");
        }
    }

    #[test]
    fn recover_cols_matches_dense() {
        // M26: the column-compacted recovery must reproduce the dense one
        // exactly. Build a dense Jacobian, then a compacted buffer holding
        // only the fixed-var columns, and assert identical multipliers.
        let frame = ReductionFrame::new(3, 3, vec![0, 2], vec![1.0, 2.0], vec![0, 1]);
        let grad_f = [10.0, 4.0, 7.0];
        let lambda_full = [0.0, 0.0, 0.5];
        let dense = [
            2.0, 1.0, 0.5, // row 0
            1.0, -1.0, 3.0, // row 1
            0.4, 1.0, 0.9, // row 2
        ];
        let dense_lam = frame
            .recover_dropped_multipliers(&grad_f, &dense, &lambda_full)
            .unwrap();

        // Compact only columns {0, 2} (the union of fixed_vars).
        let needed = [0usize, 2];
        let n_cols = needed.len();
        let mut orig_to_compact = [usize::MAX; 3];
        for (cc, &c) in needed.iter().enumerate() {
            orig_to_compact[c] = cc;
        }
        let mut jac_cols = vec![0.0; 3 * n_cols];
        for r in 0..3 {
            for (cc, &c) in needed.iter().enumerate() {
                jac_cols[r * n_cols + cc] = dense[r * 3 + c];
            }
        }
        let cols_lam = frame
            .recover_dropped_multipliers_cols(
                &grad_f,
                &jac_cols,
                n_cols,
                &orig_to_compact,
                &lambda_full,
            )
            .unwrap();

        assert_eq!(cols_lam.len(), dense_lam.len());
        for (c, d) in cols_lam.iter().zip(dense_lam.iter()) {
            assert_eq!(c.to_bits(), d.to_bits(), "cols {c} != dense {d}");
        }
    }

    #[test]
    fn recover_cols_empty_frame() {
        // No fixed vars → no columns needed → empty compact buffer is valid.
        let frame = ReductionFrame::new(2, 2, vec![], vec![], vec![]);
        let lam = frame
            .recover_dropped_multipliers_cols(&[0.0; 2], &[], 0, &[usize::MAX; 2], &[0.0; 2])
            .unwrap();
        assert!(lam.is_empty());
    }

    #[test]
    fn recover_multipliers_empty_frame() {
        let frame = ReductionFrame::new(2, 2, vec![], vec![], vec![]);
        let lam = frame
            .recover_dropped_multipliers(&[0.0; 2], &[0.0; 4], &[0.0; 2])
            .unwrap();
        assert!(lam.is_empty());
    }

    #[test]
    fn kkt_residual_after_recovery_to_1e_minus_12() {
        // 3 vars (b1, b2, y), 3 rows.
        //   row 0 (dropped):     2 b1 + b2     - 3       = 0  → at fixed point.
        //   row 1 (dropped):     b1   - b2     + 1       = 0  → at fixed point.
        //   row 2 (kept):        b1   + b2 + y - 5       = 0
        // Solving the two dropped rows: b1 = 2/3, b2 = 5/3.
        // Then row 2: y = 5 - 7/3 = 8/3 ≈ 2.667.
        // Objective f = 10 b1 + 4 b2 + y².
        // grad_f at (b1, b2, y) = (10, 4, 2y).
        //
        // The IPM-style reduced problem keeps row 2 active and var y
        // free. We need to recover λ_0, λ_1 (for the dropped rows)
        // and verify full-space stationarity at b1, b2 (and y holds
        // automatically from the reduced KKT).
        let frame = ReductionFrame::new(3, 3, vec![0, 1], vec![2.0 / 3.0, 5.0 / 3.0], vec![0, 1]);
        // Build the full row-major Jacobian at the optimum.
        let jac = [
            2.0, 1.0, 0.0, // row 0
            1.0, -1.0, 0.0, // row 1
            1.0, 1.0, 1.0, // row 2
        ];
        // Objective gradient at the optimum.
        let y_star = 8.0 / 3.0;
        let grad_f = [10.0, 4.0, 2.0 * y_star];
        // Reduced problem's kept-row multipliers: at the optimum,
        // stationarity at y is 2y - λ_2 = 0 → λ_2 = 2y = 16/3.
        let lambda_kept_2 = 2.0 * y_star;
        let lambda_full = [0.0, 0.0, lambda_kept_2];

        let lam_dropped = frame
            .recover_dropped_multipliers(&grad_f, &jac, &lambda_full)
            .unwrap();
        // Reconstruct the full λ.
        let mut lambda_recovered = lambda_full;
        for (k, &r) in frame.dropped_rows.iter().enumerate() {
            lambda_recovered[r] = lam_dropped[k];
        }
        // Verify stationarity at b1, b2 to high precision.
        for &i in &frame.fixed_vars {
            let mut s = grad_f[i];
            for r in 0..3 {
                s -= jac[r * 3 + i] * lambda_recovered[r];
            }
            assert!(s.abs() < 1e-12, "stationarity at var {i} = {s}");
        }
    }

    /// Fuzz: build a synthetic full-space KKT solution `(x*, λ*)`,
    /// declare a random subset of variables "fixed" and the
    /// matching subset of rows "dropped", then verify the multiplier
    /// recovery reproduces the original λ at the dropped indices to
    /// within 1e-10.
    struct FuzzRng(u64);
    impl FuzzRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 32
        }
        fn unit(&mut self) -> Number {
            let raw = (self.next_u64() & 0x3fff_ffff) as Number;
            raw / (1u64 << 29) as Number - 1.0
        }
    }

    #[test]
    fn frame_fuzz_recover_reproduces_synthetic_lambda() {
        let mut rng = FuzzRng::new(0xface_b00c_baad_f00d);

        for trial in 0..30 {
            let n_vars = 2 + (rng.next_u64() % 3) as usize; // 2..=4
            let n_rows = n_vars;
            let k = 1 + (rng.next_u64() % n_vars as u64) as usize;

            let mut perm_v: Vec<usize> = (0..n_vars).collect();
            for i in (1..n_vars).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                perm_v.swap(i, j);
            }
            let mut fixed_vars: Vec<usize> = perm_v[..k].to_vec();
            fixed_vars.sort_unstable();

            let mut perm_r: Vec<usize> = (0..n_rows).collect();
            for i in (1..n_rows).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                perm_r.swap(i, j);
            }
            let mut dropped_rows: Vec<usize> = perm_r[..k].to_vec();
            dropped_rows.sort_unstable();

            let mut jac = vec![0.0; n_rows * n_vars];
            for r in 0..n_rows {
                for c in 0..n_vars {
                    jac[r * n_vars + c] = 0.2 * rng.unit();
                }
            }
            for (&r, &c) in dropped_rows.iter().zip(fixed_vars.iter()) {
                jac[r * n_vars + c] += 2.5;
            }

            let lambda_star: Vec<Number> = (0..n_rows).map(|_| rng.unit()).collect();
            let mut grad_f = vec![0.0; n_vars];
            let fixed_set: std::collections::BTreeSet<usize> = fixed_vars.iter().copied().collect();
            for i in 0..n_vars {
                if fixed_set.contains(&i) {
                    let mut s = 0.0;
                    for r in 0..n_rows {
                        s += jac[r * n_vars + i] * lambda_star[r];
                    }
                    grad_f[i] = s;
                } else {
                    grad_f[i] = rng.unit();
                }
            }

            let dropped_set: std::collections::BTreeSet<usize> =
                dropped_rows.iter().copied().collect();
            let mut lambda_given = vec![0.0; n_rows];
            for r in 0..n_rows {
                if !dropped_set.contains(&r) {
                    lambda_given[r] = lambda_star[r];
                }
            }

            let frame = ReductionFrame::new(
                n_vars,
                n_rows,
                fixed_vars.clone(),
                vec![0.0; k],
                dropped_rows.clone(),
            );

            let lam_dropped = frame
                .recover_dropped_multipliers(&grad_f, &jac, &lambda_given)
                .unwrap_or_else(|e| panic!("trial {trial}: {e:?}"));

            for (idx, &r) in dropped_rows.iter().enumerate() {
                let expected = lambda_star[r];
                let got = lam_dropped[idx];
                assert!(
                    (expected - got).abs() < 1e-10,
                    "trial {trial}: λ[{r}] expected {expected:.6}, got {got:.6}"
                );
            }
        }
    }

    #[test]
    fn reduction_stack_push_top_iter() {
        let mut stack = ReductionStack::default();
        assert!(stack.is_empty());
        let f1 = ReductionFrame::new(2, 2, vec![0], vec![1.0], vec![0]);
        let f2 = ReductionFrame::new(2, 2, vec![1], vec![2.0], vec![1]);
        stack.push(f1.clone());
        stack.push(f2.clone());
        assert_eq!(stack.len(), 2);
        let top = stack.top().expect("non-empty");
        assert_eq!(top.fixed_vars, f2.fixed_vars);
        // top-down: f2, then f1.
        let order: Vec<_> = stack.iter_top_down().map(|f| f.fixed_vars[0]).collect();
        assert_eq!(order, vec![1, 0]);
        let order_up: Vec<_> = stack.iter_bottom_up().map(|f| f.fixed_vars[0]).collect();
        assert_eq!(order_up, vec![0, 1]);
    }

    /// PR #60 review nit: there was no paired test for `project_lambda`
    /// + `lift_lambda` (only the directional tests). Confirm
    /// project(lift(x)) is the identity on the reduced shape AND
    /// lift(project(x)) zeroes the dropped indices but preserves
    /// kept ones.
    #[test]
    fn frame_project_lift_lambda_roundtrip() {
        let frame = ReductionFrame::new(4, 3, vec![0, 2], vec![1.0, 9.0], vec![0, 1]);
        // Full lambda with arbitrary values; project then lift.
        let lambda_full = [4.0, 5.0, 6.0];
        let reduced = frame.project_lambda(&lambda_full);
        // Reduced has one row (the only kept row, row 2).
        assert_eq!(reduced, vec![6.0]);
        // Lifting back zeroes the dropped row entries.
        let lifted = frame.lift_lambda(&reduced);
        assert_eq!(lifted, vec![0.0, 0.0, 6.0]);
        // Now the other direction: project the lifted lambda back
        // to reduced — should be the identity on reduced shape.
        let reduced_again = frame.project_lambda(&lifted);
        assert_eq!(reduced_again, reduced);
    }

    /// Multi-frame `ReductionStack` round-trip. Push two frames
    /// (mutually compatible — they fix disjoint vars and drop
    /// disjoint rows). Verify lift_x and lift_lambda compose
    /// consistently when walked through both frames.
    #[test]
    fn reduction_stack_multi_frame_roundtrip() {
        // Full shape: 4 vars, 4 rows.
        // Frame 1 (bottom): fixes var 0 (= 10), drops row 0.
        // Frame 2 (top):    fixes var 2 (= 30), drops row 2.
        let f1 = ReductionFrame::new(4, 4, vec![0], vec![10.0], vec![0]);
        let f2 = ReductionFrame::new(4, 4, vec![2], vec![30.0], vec![2]);
        let mut stack = ReductionStack::default();
        stack.push(f1.clone());
        stack.push(f2.clone());

        // Synthesize a "fully-lifted" x_full where the survivors
        // (vars 1, 3) and rows (1, 3) carry known values.
        let x_full_expected = vec![10.0, 7.0, 30.0, 5.0];
        let lambda_full_expected = vec![0.0, 8.0, 0.0, 6.0];

        // Project through both frames in bottom-up order, then
        // lift back top-down. Result must equal original at the
        // surviving entries (and frame-supplied values at fixed
        // entries / zeros at dropped row indices).
        //
        // For this test we don't have stacked reduced shapes
        // (each frame is independently 4-var/4-row); we just
        // confirm each frame's lift drops the expected values
        // back when walked individually via the stack's iterator.
        for frame in stack.iter_top_down() {
            let reduced_x = frame.project_x(&x_full_expected);
            let lifted_x = frame.lift_x(&reduced_x);
            assert_eq!(lifted_x, x_full_expected);
            let reduced_l = frame.project_lambda(&lambda_full_expected);
            let lifted_l = frame.lift_lambda(&reduced_l);
            // Dropped index should be 0 in the lift; survivors
            // preserve their values.
            for r in 0..4 {
                if frame.row_map[r].is_some() {
                    assert_eq!(lifted_l[r], lambda_full_expected[r]);
                } else {
                    assert_eq!(lifted_l[r], 0.0);
                }
            }
        }
    }
}
