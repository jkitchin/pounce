//! `SchurData` trait surface and the `IndexSchurData` flavor.
//!
//! Direct port of upstream
//! [`SensSchurData.hpp`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp)
//! (interface) and
//! [`SensIndexSchurData.{hpp,cpp}`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.hpp)
//! (the index-only specialization).
//!
//! # What `SchurData` represents
//!
//! `SchurData` is the matrix `B` in the augmented system
//!
//! ```text
//! ⎡ K   A ⎤
//! ⎣ B   0 ⎦
//! ```
//!
//! used by the sIPOPT step calculation (Pirnay, López-Negrete & Biegler 2012,
//! §2, eq. 4). `K` is the converged KKT matrix from the original IPM solve;
//! `A` and `B` carry the parameter-perturbation rows. The trait is the
//! minimum surface every backend needs to expose so the
//! `PCalculator` / `SchurDriver` family can stay matrix-shape-agnostic
//! ([`SensSchurData.hpp:17-25`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp)).
//!
//! # `IndexSchurData`
//!
//! Specialization for parameter rows whose only non-zero entries are
//! ±1 ([`SensIndexSchurData.hpp:15-19`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.hpp)).
//! Most parametric / reduced-Hessian use cases fit this shape — the
//! parameter just picks out a subset of primal/dual variables — so the
//! ±1 sparsification is what production sIPOPT runs on. Pounce's
//! port stores parallel `Vec<i32>` arrays of indices and signs, same
//! as upstream's `idx_` / `val_`
//! ([`SensIndexSchurData.hpp:127-128`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.hpp)).

use pounce_common::types::{Index, Number};

/// Minimum surface for any matrix that lives in the augmented sIPOPT
/// system's `A` / `B` slots. The numerical drivers in this crate
/// (`PCalculator`, `SchurDriver`, `SensStepCalc`) consume `SchurData`
/// objects and never touch the storage shape directly.
///
/// Mirrors `Ipopt::SchurData` from
/// [`SensSchurData.hpp:29-178`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp).
///
/// # Lifecycle
///
/// A `SchurData` instance starts uninitialized. Exactly one of the
/// `set_*` methods is called to populate it; subsequent reads via
/// `nrows`, `multiply`, `trans_multiply`, etc. require an
/// initialized instance. Upstream enforces this via `Set_Initialized`
/// asserts in DBG builds
/// ([`SensIndexSchurData.cpp:59-60`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.cpp));
/// pounce mirrors that invariant via `Result` returns on the read
/// surface so a mis-ordered call surfaces as `Err(SchurDataError::NotInitialized)`
/// instead of a panic.
pub trait SchurData {
    /// Number of rows the schur matrix has, i.e. the row count of `B`.
    /// Upstream `GetNRowsAdded()`
    /// ([`SensSchurData.hpp:77-80`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp)).
    fn nrows(&self) -> Index;

    /// `true` if one of the `set_*` methods has been called and the
    /// rest of the surface is safe to call. Upstream `Is_Initialized()`
    /// ([`SensSchurData.hpp:82-85`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp)).
    fn is_initialized(&self) -> bool;

    /// Set rows from a 0/1 flag array of length `dim`. For each `i`
    /// with `flags[i] == 1`, add a row whose only non-zero column is
    /// `i` with sign `sign(v)`. The magnitude of `v` is collapsed to
    /// ±1 — this trait only ever stores signs, mirroring upstream's
    /// `SetData_Flag(dim, flags, v)`
    /// ([`SensSchurData.hpp:45-49`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp),
    /// [`SensIndexSchurData.cpp:51-78`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.cpp)).
    ///
    /// Returns `Err(_)` if the instance was already initialized.
    fn set_from_flags(&mut self, flags: &[Index], v: Number) -> Result<(), SchurDataError>;

    /// Set rows from a list of column indices. Each `cols[k]` becomes
    /// row `k` of `B`, with the single non-zero entry at column
    /// `cols[k]` carrying sign `sign(v)`. Upstream `SetData_List`
    /// ([`SensSchurData.hpp:64-67`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp),
    /// [`SensIndexSchurData.cpp:149-167`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.cpp)).
    ///
    /// Returns `Err(_)` if already initialized.
    fn set_from_list(&mut self, cols: &[Index], v: Number) -> Result<(), SchurDataError>;

    /// Row-`i` access: return the parallel arrays `(indices, factors)`
    /// such that row `i` of `B` has non-zero column entries
    /// `factors[j]` at columns `indices[j]`. For `IndexSchurData`
    /// `indices` has length 1 and `factors[0] == ±1.0`. Upstream
    /// `GetMultiplyingVectors`
    /// ([`SensSchurData.hpp:93-104`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp),
    /// [`SensIndexSchurData.cpp:199-212`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.cpp)).
    fn multiplying_row(&self, i: Index) -> Result<(Vec<Index>, Vec<Number>), SchurDataError>;

    /// Apply `u = B v` for a `v` of length `n_full` and pre-sized
    /// `u` buffer (length must equal `self.nrows()`). Upstream
    /// `Multiply(IteratesVector, Vector)`
    /// ([`SensSchurData.hpp:106-110`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp),
    /// [`SensIndexSchurData.cpp:214-251`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.cpp)).
    ///
    /// In pounce we operate on flat `&[Number]` instead of upstream's
    /// `IteratesVector` block layout because Phase-A `SchurData` is
    /// shape-agnostic. Phase-B reconstructs the block layout where
    /// the algorithm-side needs it.
    fn multiply(&self, v: &[Number], u: &mut [Number]) -> Result<(), SchurDataError>;

    /// Apply `v = Bᵀ u` for a `u` of length `self.nrows()` and a
    /// pre-sized `v` buffer (length `n_full`). Upstream
    /// `TransMultiply`
    /// ([`SensSchurData.hpp:112-116`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp),
    /// [`SensIndexSchurData.cpp:253-307`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.cpp)).
    fn trans_multiply(&self, u: &[Number], v: &mut [Number]) -> Result<(), SchurDataError>;
}

/// Failure modes returned by [`SchurData`] read/write entry points.
/// Pounce returns these as `Err(_)` where upstream's debug asserts
/// would `DBG_ASSERT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchurDataError {
    /// A read method was called before a `set_*` method initialized
    /// the instance. Upstream asserts `Is_Initialized()` in DBG builds
    /// (e.g. [`SensIndexSchurData.cpp:176`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.cpp)).
    NotInitialized,
    /// `set_*` called twice on the same instance, or `flags` contained
    /// values other than 0/1. Upstream:
    /// [`SensIndexSchurData.cpp:59-69`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.cpp).
    AlreadyInitialized,
    /// A row index was out of range (e.g. `multiplying_row(i)` with
    /// `i >= nrows()`).
    RowOutOfRange,
    /// Caller-supplied buffer length didn't match the expected shape.
    DimensionMismatch,
    /// `v == 0` passed to a sign-only `set_*` API. Upstream asserts
    /// `v != 0` ([`SensIndexSchurData.cpp:61`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.cpp)).
    ZeroSign,
}

/// Specialization for `B` matrices whose non-zero entries are ±1
/// (the parametric / reduced-Hessian common case). Storage is two
/// parallel arrays mirroring upstream's `idx_` and `val_`
/// ([`SensIndexSchurData.hpp:127-128`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.hpp)).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexSchurData {
    idx: Vec<Index>,
    /// Stored as ±1 values, matching upstream's `Index`-typed `val_`.
    /// Kept as `Index` (not `Number`) so the sign is exact and small.
    val: Vec<Index>,
    initialized: bool,
}

impl IndexSchurData {
    /// Empty / uninitialized instance. Must be populated via one of
    /// the `set_from_*` methods before being read.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct directly from pre-built `(idx, val)` arrays. `val`
    /// must contain only ±1 entries. Mirrors upstream's two-arg
    /// constructor
    /// ([`SensIndexSchurData.cpp:24-36`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.cpp)).
    pub fn from_parts(idx: Vec<Index>, val: Vec<Index>) -> Result<Self, SchurDataError> {
        if idx.len() != val.len() {
            return Err(SchurDataError::DimensionMismatch);
        }
        if val.iter().any(|&v| v != 1 && v != -1) {
            return Err(SchurDataError::ZeroSign);
        }
        Ok(Self {
            idx,
            val,
            initialized: true,
        })
    }

    /// Column indices the rows refer to (one index per row). Upstream
    /// `GetColIndices()`
    /// ([`SensIndexSchurData.hpp:114`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.hpp)).
    pub fn col_indices(&self) -> &[Index] {
        &self.idx
    }

    /// Per-row ±1 sign carried by the single non-zero entry in that
    /// row. Pounce-specific accessor; upstream exposes the data only
    /// through the `multiplying_*` / `multiply` APIs.
    pub fn signs(&self) -> &[Index] {
        &self.val
    }
}

impl SchurData for IndexSchurData {
    fn nrows(&self) -> Index {
        self.val.len() as Index
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn set_from_flags(&mut self, flags: &[Index], v: Number) -> Result<(), SchurDataError> {
        if self.initialized {
            return Err(SchurDataError::AlreadyInitialized);
        }
        if v == 0.0 {
            return Err(SchurDataError::ZeroSign);
        }
        let w: Index = if v > 0.0 { 1 } else { -1 };
        for (i, &f) in flags.iter().enumerate() {
            match f {
                0 => {}
                1 => {
                    self.idx.push(i as Index);
                    self.val.push(w);
                }
                _ => return Err(SchurDataError::AlreadyInitialized), // upstream asserts flag ∈ {0,1}
            }
        }
        self.initialized = true;
        Ok(())
    }

    fn set_from_list(&mut self, cols: &[Index], v: Number) -> Result<(), SchurDataError> {
        if self.initialized {
            return Err(SchurDataError::AlreadyInitialized);
        }
        if v == 0.0 {
            return Err(SchurDataError::ZeroSign);
        }
        let w: Index = if v > 0.0 { 1 } else { -1 };
        self.idx.extend_from_slice(cols);
        self.val.resize(cols.len(), w);
        self.initialized = true;
        Ok(())
    }

    fn multiplying_row(
        &self,
        i: Index,
    ) -> Result<(Vec<Index>, Vec<Number>), SchurDataError> {
        if !self.initialized {
            return Err(SchurDataError::NotInitialized);
        }
        let i_us = i as usize;
        if i_us >= self.idx.len() {
            return Err(SchurDataError::RowOutOfRange);
        }
        Ok((vec![self.idx[i_us]], vec![self.val[i_us] as Number]))
    }

    fn multiply(&self, v: &[Number], u: &mut [Number]) -> Result<(), SchurDataError> {
        if !self.initialized {
            return Err(SchurDataError::NotInitialized);
        }
        if u.len() != self.idx.len() {
            return Err(SchurDataError::DimensionMismatch);
        }
        for (i, slot) in u.iter_mut().enumerate() {
            let col = self.idx[i] as usize;
            if col >= v.len() {
                return Err(SchurDataError::DimensionMismatch);
            }
            *slot = (self.val[i] as Number) * v[col];
        }
        Ok(())
    }

    fn trans_multiply(&self, u: &[Number], v: &mut [Number]) -> Result<(), SchurDataError> {
        if !self.initialized {
            return Err(SchurDataError::NotInitialized);
        }
        if u.len() != self.idx.len() {
            return Err(SchurDataError::DimensionMismatch);
        }
        for slot in v.iter_mut() {
            *slot = 0.0;
        }
        for (i, &row_u) in u.iter().enumerate() {
            let col = self.idx[i] as usize;
            if col >= v.len() {
                return Err(SchurDataError::DimensionMismatch);
            }
            v[col] += (self.val[i] as Number) * row_u;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three-variable example: select variables 1 and 3 with sign +1.
    /// `B = [[0 1 0 0], [0 0 0 1]]`.
    #[test]
    fn set_from_flags_round_trip() {
        let mut s = IndexSchurData::new();
        let flags = [0, 1, 0, 1];
        s.set_from_flags(&flags, 1.0).expect("init");
        assert_eq!(s.nrows(), 2);
        assert_eq!(s.col_indices(), &[1, 3]);
        assert_eq!(s.signs(), &[1, 1]);
        assert!(s.is_initialized());
    }

    #[test]
    fn set_from_flags_negative_sign_records_minus_one() {
        let mut s = IndexSchurData::new();
        s.set_from_flags(&[1, 0, 1], -2.5).expect("init");
        assert_eq!(s.signs(), &[-1, -1]);
    }

    #[test]
    fn set_from_flags_rejects_double_init() {
        let mut s = IndexSchurData::new();
        s.set_from_flags(&[1, 0], 1.0).expect("first init");
        assert_eq!(
            s.set_from_flags(&[0, 1], 1.0),
            Err(SchurDataError::AlreadyInitialized),
        );
    }

    #[test]
    fn set_from_flags_rejects_zero_sign() {
        let mut s = IndexSchurData::new();
        assert_eq!(
            s.set_from_flags(&[1, 0, 1], 0.0),
            Err(SchurDataError::ZeroSign),
        );
    }

    #[test]
    fn set_from_list_records_each_column_once() {
        let mut s = IndexSchurData::new();
        s.set_from_list(&[2, 0, 4], 1.0).expect("init");
        assert_eq!(s.nrows(), 3);
        assert_eq!(s.col_indices(), &[2, 0, 4]);
        assert_eq!(s.signs(), &[1, 1, 1]);
    }

    #[test]
    fn from_parts_validates_signs() {
        // Mixed ±1 OK
        let ok = IndexSchurData::from_parts(vec![0, 2], vec![1, -1]).expect("ok");
        assert_eq!(ok.signs(), &[1, -1]);
        // Length mismatch
        assert_eq!(
            IndexSchurData::from_parts(vec![0, 2], vec![1]),
            Err(SchurDataError::DimensionMismatch),
        );
        // Non-±1
        assert_eq!(
            IndexSchurData::from_parts(vec![0], vec![2]),
            Err(SchurDataError::ZeroSign),
        );
    }

    #[test]
    fn multiply_picks_selected_columns_with_signs() {
        // B = [[0 1 0 0], [0 0 0 -1]]
        let s = IndexSchurData::from_parts(vec![1, 3], vec![1, -1]).expect("ok");
        let v = [10.0, 20.0, 30.0, 40.0];
        let mut u = [0.0; 2];
        s.multiply(&v, &mut u).expect("ok");
        // u[0] = +1·v[1] = 20
        // u[1] = -1·v[3] = -40
        assert_eq!(u, [20.0, -40.0]);
    }

    #[test]
    fn trans_multiply_scatters_with_signs() {
        // B from previous test; Bᵀ u with u = [3, 5] should produce
        // v = (0, 3, 0, -5).
        let s = IndexSchurData::from_parts(vec![1, 3], vec![1, -1]).expect("ok");
        let u = [3.0, 5.0];
        let mut v = [0.0; 4];
        s.trans_multiply(&u, &mut v).expect("ok");
        assert_eq!(v, [0.0, 3.0, 0.0, -5.0]);
    }

    #[test]
    fn trans_multiply_overwrites_caller_buffer() {
        // Existing entries in v are zeroed before the scatter.
        let s = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).expect("ok");
        let u = [1.0, 2.0];
        let mut v = [99.0, 99.0, 99.0, 99.0];
        s.trans_multiply(&u, &mut v).expect("ok");
        assert_eq!(v, [1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn multiply_rejects_uninitialized() {
        let s = IndexSchurData::new();
        let v = [0.0];
        let mut u = [0.0];
        assert_eq!(
            s.multiply(&v, &mut u),
            Err(SchurDataError::NotInitialized),
        );
    }

    #[test]
    fn multiplying_row_out_of_range() {
        let s = IndexSchurData::from_parts(vec![0], vec![1]).expect("ok");
        assert_eq!(
            s.multiplying_row(2),
            Err(SchurDataError::RowOutOfRange),
        );
    }

    #[test]
    fn multiplying_row_returns_single_entry_for_index_schur_data() {
        // Per upstream `SensIndexSchurData.cpp:199-212`, `IndexSchurData`
        // rows always have exactly one non-zero entry — pounce mirrors
        // that contract.
        let s = IndexSchurData::from_parts(vec![5, 7], vec![1, -1]).expect("ok");
        let (idxs, facs) = s.multiplying_row(0).expect("ok");
        assert_eq!(idxs, &[5]);
        assert_eq!(facs, &[1.0]);
        let (idxs, facs) = s.multiplying_row(1).expect("ok");
        assert_eq!(idxs, &[7]);
        assert_eq!(facs, &[-1.0]);
    }
}
