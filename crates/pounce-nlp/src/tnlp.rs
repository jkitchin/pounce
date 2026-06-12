//! User-facing `TNLP` trait — port of `Interfaces/IpTNLP.{hpp,cpp}`.
//!
//! The Rust shape replaces upstream's two-call `(iRow,jCol,values)`
//! convention with [`SparsityRequest`], a request enum carrying the
//! caller-supplied buffers. This is more typesafe (no NULL pointers,
//! buffer length is type-checked) and matches the eight-method API
//! upstream documents.
//!
//! The `IpoptData` / `IpoptCalculatedQuantities` / `IteratesVector`
//! parameters of `intermediate_callback` and `finalize_solution` are
//! introduced as opaque [`IpoptData`] / [`IpoptCq`] types; their full
//! field set lands in Phase 5.
//!
//! Trait objects: `dyn TNLP` is supported. Concrete callers store the
//! TNLP behind an `Rc<RefCell<dyn TNLP>>` (so eval methods can mutate
//! internal caches) — `pounce_algorithm::IpoptApplication` handles
//! wrapping.

use crate::alg_types::SolverReturn;
use crate::return_codes::AlgorithmMode;
use pounce_common::types::{Index, Number};
use std::collections::BTreeMap;

/// Linearity tags. Mirrors `TNLP::LinearityType` upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linearity {
    Linear,
    NonLinear,
}

/// Index style for triplet I/O. Mirrors `TNLP::IndexStyleEnum`.
/// `Fortran` (1-based) is what MUMPS / HSL want directly; `C`
/// (0-based) is more natural for Rust user code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexStyle {
    C = 0,
    Fortran = 1,
}

/// Problem dimensions returned by [`TNLP::get_nlp_info`].
#[derive(Debug, Clone, Copy)]
pub struct NlpInfo {
    pub n: Index,
    pub m: Index,
    pub nnz_jac_g: Index,
    pub nnz_h_lag: Index,
    pub index_style: IndexStyle,
}

/// Variable / constraint metadata buckets, mirroring upstream's
/// `(StringMetaDataMapType, IntegerMetaDataMapType, NumericMetaDataMapType)`.
#[derive(Debug, Default, Clone)]
pub struct MetaData {
    pub strings: BTreeMap<String, Vec<String>>,
    pub integers: BTreeMap<String, Vec<Index>>,
    pub numerics: BTreeMap<String, Vec<Number>>,
}

/// Conventional [`MetaData::strings`] key for per-index human-readable
/// names (one entry per variable, or per constraint, in original
/// problem order). Mirrors upstream Ipopt's `"idx_names"` metadata
/// key. Carrying names this far lets the debugger report a near-singular
/// Jacobian row as the `mass_balance` equation instead of "row 3" —
/// the model-vs-index gap Lee et al. (2024,
/// <https://doi.org/10.69997/sct.147875>) flag as a key roadblock for
/// debugging equation-oriented models.
pub const IDX_NAMES: &str = "idx_names";

/// Bound-data target buffers passed into [`TNLP::get_bounds_info`].
#[derive(Debug)]
pub struct BoundsInfo<'a> {
    pub x_l: &'a mut [Number],
    pub x_u: &'a mut [Number],
    pub g_l: &'a mut [Number],
    pub g_u: &'a mut [Number],
}

/// Starting-point target buffers passed into [`TNLP::get_starting_point`].
/// Each `init_*` flag matches upstream — mostly false unless warm-starting.
#[derive(Debug)]
pub struct StartingPoint<'a> {
    pub init_x: bool,
    pub x: &'a mut [Number],
    pub init_z: bool,
    pub z_l: &'a mut [Number],
    pub z_u: &'a mut [Number],
    pub init_lambda: bool,
    pub lambda: &'a mut [Number],
}

/// Scaling-factor target buffers passed into [`TNLP::get_scaling_parameters`].
#[derive(Debug)]
pub struct ScalingRequest<'a> {
    pub obj_scaling: &'a mut Number,
    pub use_x_scaling: &'a mut bool,
    pub x_scaling: &'a mut [Number],
    pub use_g_scaling: &'a mut bool,
    pub g_scaling: &'a mut [Number],
}

/// Mode discriminator for the structure / values calls of
/// [`TNLP::eval_jac_g`] and [`TNLP::eval_h`]. Replaces upstream's
/// `iRow != NULL` heuristic.
#[derive(Debug)]
pub enum SparsityRequest<'a> {
    /// First call: fill `irow` and `jcol` with the structure (the
    /// numbering style is whatever was returned in
    /// [`NlpInfo::index_style`]). The values array is absent.
    Structure {
        irow: &'a mut [Index],
        jcol: &'a mut [Index],
    },
    /// Subsequent calls: fill `values` with the entries of the matrix
    /// at the current `x` (and, for the Hessian, `lambda`,
    /// `obj_factor`).
    Values { values: &'a mut [Number] },
}

/// Solution as passed to [`TNLP::finalize_solution`].
#[derive(Debug)]
pub struct Solution<'a> {
    pub status: SolverReturn,
    pub x: &'a [Number],
    pub z_l: &'a [Number],
    pub z_u: &'a [Number],
    pub g: &'a [Number],
    pub lambda: &'a [Number],
    pub obj_value: Number,
}

/// Per-iteration callback payload for [`TNLP::intermediate_callback`].
#[derive(Debug, Clone, Copy)]
pub struct IterStats {
    pub mode: AlgorithmMode,
    pub iter: Index,
    pub obj_value: Number,
    pub inf_pr: Number,
    pub inf_du: Number,
    pub mu: Number,
    pub d_norm: Number,
    pub regularization_size: Number,
    pub alpha_du: Number,
    pub alpha_pr: Number,
    pub ls_trials: Index,
}

/// Forward-declared placeholder for `IpoptData`. Phase 5 fills this
/// in with the full mutable iterate-state structure; for Phase 3 it
/// is opaque.
#[derive(Debug, Default)]
pub struct IpoptData {
    _private: (),
}

/// Forward-declared placeholder for `IpoptCalculatedQuantities`.
/// Phase 5 fills this in.
#[derive(Debug, Default)]
pub struct IpoptCq {
    _private: (),
}

/// User-facing NLP interface — port of `class TNLP`. Object-safe.
///
/// Defaults provided for every method that upstream documents as
/// "default returns false / does nothing", so simple problems only
/// override the eight pure-virtual methods.
pub trait TNLP {
    /// **Required.** Problem dimensions and triplet index style.
    fn get_nlp_info(&mut self) -> Option<NlpInfo>;

    /// **Required.** Variable / constraint bounds.
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool;

    /// **Required.** Initial primal (and optionally dual) point.
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool;

    /// **Required.** Objective value at `x`.
    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number>;

    /// **Required.** Objective gradient at `x` into `grad_f`.
    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool;

    /// **Required.** Constraint values `g(x)`.
    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool;

    /// **Required.** Jacobian of `g`. Sparsity vs. values selected by
    /// `mode`. `x` and `new_x` are unused on the structure call.
    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool;

    /// **Required for exact Hessian, optional for L-BFGS.** Hessian
    /// of the Lagrangian. Default returns false (signals to %Ipopt
    /// that quasi-Newton must be used).
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        false
    }

    /// **Required.** Receives the final iterate after solve.
    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq);

    // ---- Optional methods (defaults match upstream's "do nothing") ----

    /// Provide variable/constraint metadata (e.g. `idx_names`).
    /// Default: no metadata.
    fn get_var_con_metadata(&mut self, _var: &mut MetaData, _con: &mut MetaData) -> bool {
        false
    }

    /// User-supplied scaling, used only when
    /// `nlp_scaling_method=user-scaling`. Default: declines.
    fn get_scaling_parameters(&mut self, _req: ScalingRequest<'_>) -> bool {
        false
    }

    /// Variable linearity tags (used by Bonmin, not by Ipopt).
    fn get_variables_linearity(&mut self, _types: &mut [Linearity]) -> bool {
        false
    }

    /// Per-variable linearity with respect to the **objective only** (a
    /// pounce extension; upstream has no objective-scoped query).
    /// `NonLinear` iff the objective's nonlinear part depends on the
    /// variable; a variable that enters the objective only linearly (or
    /// not at all) is `Linear` even when it is nonlinear in a
    /// constraint. Consumed by presolve's Phase-0 objective-coupling
    /// guard, which must not mistake constraint-only nonlinearity for
    /// objective coupling. Default: declines (slice untouched).
    fn get_objective_variables_linearity(&mut self, _types: &mut [Linearity]) -> bool {
        false
    }

    /// Constraint linearity tags. Used by adaptive-mu's
    /// `nlp_scaling_method=equilibration-based`.
    fn get_constraints_linearity(&mut self, _types: &mut [Linearity]) -> bool {
        false
    }

    /// Number of variables that appear nonlinearly. Returning -1
    /// means "treat all as nonlinear" (the Ipopt default).
    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        -1
    }

    /// List of nonlinear variable indices, in the index style
    /// returned from [`Self::get_nlp_info`].
    fn get_list_of_nonlinear_variables(&mut self, _pos_nonlin_vars: &mut [Index]) -> bool {
        false
    }

    /// Per-iteration intermediate callback. Returning false requests
    /// early termination with `User_Requested_Stop`.
    fn intermediate_callback(
        &mut self,
        _stats: IterStats,
        _ip_data: &IpoptData,
        _ip_cq: &IpoptCq,
    ) -> bool {
        true
    }

    /// Final metadata pass — called just before
    /// [`Self::finalize_solution`]. Default does nothing.
    fn finalize_metadata(&mut self, _var: &MetaData, _con: &MetaData) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny `min x[0]^2 + x[1]^2  s.t. x[0] + x[1] = 1` problem.
    /// Used as a smoke test that the trait is object-safe and the
    /// defaults compile.
    struct Mini;
    impl TNLP for Mini {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 2,
                m: 1,
                nnz_jac_g: 2,
                nnz_h_lag: 2,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.iter_mut().for_each(|v| *v = -1e19);
            b.x_u.iter_mut().for_each(|v| *v = 1e19);
            b.g_l[0] = 1.0;
            b.g_u[0] = 1.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            assert!(sp.init_x);
            sp.x[0] = 0.5;
            sp.x[1] = 0.5;
            true
        }
        fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
            Some(x[0] * x[0] + x[1] * x[1])
        }
        fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
            grad_f[0] = 2.0 * x[0];
            grad_f[1] = 2.0 * x[1];
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] + x[1];
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _new_x: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow.copy_from_slice(&[0, 0]);
                    jcol.copy_from_slice(&[0, 1]);
                }
                SparsityRequest::Values { values } => {
                    values.copy_from_slice(&[1.0, 1.0]);
                }
            }
            true
        }
        fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    #[test]
    fn tnlp_is_object_safe() {
        // The trait must be usable behind `dyn`; this also exercises
        // every default-impl method to make sure they compile.
        let mut t: Box<dyn TNLP> = Box::new(Mini);
        let info = t.get_nlp_info().expect("get_nlp_info");
        assert_eq!(info.n, 2);
        assert_eq!(info.m, 1);
        assert_eq!(info.index_style, IndexStyle::C);

        let mut x_l = [0.0; 2];
        let mut x_u = [0.0; 2];
        let mut g_l = [0.0; 1];
        let mut g_u = [0.0; 1];
        assert!(t.get_bounds_info(BoundsInfo {
            x_l: &mut x_l,
            x_u: &mut x_u,
            g_l: &mut g_l,
            g_u: &mut g_u
        }));
        assert_eq!(g_l[0], 1.0);

        let mut grad = [0.0; 2];
        assert!(t.eval_grad_f(&[3.0, 4.0], true, &mut grad));
        assert_eq!(grad, [6.0, 8.0]);

        // exact-Hessian default returns false
        let mut tmp_v = [0.0; 0];
        assert!(!t.eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Values { values: &mut tmp_v }
        ));

        // Quasi-Newton info default
        assert_eq!(t.get_number_of_nonlinear_variables(), -1);
    }

    #[test]
    fn sparsity_request_round_trip() {
        let mut t = Mini;
        let mut irow = [0; 2];
        let mut jcol = [0; 2];
        assert!(t.eval_jac_g(
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol
            }
        ));
        assert_eq!(irow, [0, 0]);
        assert_eq!(jcol, [0, 1]);

        let mut vals = [0.0; 2];
        assert!(t.eval_jac_g(
            Some(&[1.0, 2.0]),
            true,
            SparsityRequest::Values { values: &mut vals }
        ));
        assert_eq!(vals, [1.0, 1.0]);
    }
}
