//! POUNCE linear algebra primitives.
//!
//! Port of Ipopt's `src/LinAlg/`: BLAS-1 (this phase), Vector / Matrix
//! abstractions and concrete implementations (added incrementally
//! through Phase 2), triplet storage and triplet→CSC conversion
//! (Phase 4).

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod blas1;
pub mod compound_matrix;
pub mod compound_vector;
pub mod dense_gen_matrix;
pub mod dense_sym_matrix;
pub mod dense_vector;
pub mod diag_matrix;
pub mod eigen;
pub mod expansion_matrix;
pub mod low_rank_update_sym_matrix;
pub mod matrix;
pub mod multi_vector_matrix;
pub mod scaled_matrix;
pub mod special_matrix;
pub mod sum_matrix;
pub mod transpose_matrix;
pub mod triplet;
pub mod triplet_convert;
pub mod vector;

pub use compound_matrix::{
    CompoundMatrix, CompoundMatrixSpace, CompoundSymMatrix, CompoundSymMatrixSpace,
};
pub use compound_vector::{CompoundVector, CompoundVectorSpace};
pub use dense_gen_matrix::{DenseGenMatrix, DenseGenMatrixSpace};
pub use dense_sym_matrix::{DenseSymMatrix, DenseSymMatrixSpace};
pub use dense_vector::{DenseVector, DenseVectorSpace};
pub use diag_matrix::DiagMatrix;
pub use eigen::symmetric_eigen;
pub use expansion_matrix::{ExpansionMatrix, ExpansionMatrixSpace};
pub use low_rank_update_sym_matrix::{LowRankUpdateSymMatrix, LowRankUpdateSymMatrixSpace};
pub use matrix::{Matrix, MatrixCache, SymMatrix};
pub use multi_vector_matrix::{MultiVectorMatrix, MultiVectorMatrixSpace};
pub use scaled_matrix::{
    ScaledMatrix, ScaledMatrixSpace, ScalingReciprocal, SymScaledMatrix, SymScaledMatrixSpace,
};
pub use special_matrix::{IdentityMatrix, ZeroMatrix, ZeroSymMatrix};
pub use sum_matrix::{SumMatrix, SumSymMatrix};
pub use transpose_matrix::TransposeMatrix;
pub use triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
pub use triplet_convert::{TriFull, TripletToCsrConverter};
pub use vector::{Vector, VectorCache};
