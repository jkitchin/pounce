//! NLP traits consumed by the algorithm core — port of `IpNLP.hpp` /
//! `IpIpoptNLP.hpp`.
//!
//! These traits live in `pounce-nlp` (rather than `pounce-algorithm`)
//! so that the concrete [`crate::orig_ipopt_nlp::OrigIpoptNlp`], which
//! wraps a `TNLPAdapter` from this same crate, can implement them
//! without forcing `pounce-nlp` to depend on `pounce-algorithm` (the
//! reverse dependency already exists). `pounce-algorithm` re-exports
//! both traits from its own `ipopt_nlp` module so the rest of the
//! algorithm-side code continues to use the canonical
//! `crate::ipopt_nlp::IpoptNlp` path.

use pounce_common::types::{Index, Number};
use pounce_linalg::{DenseVector, Matrix, SymMatrix, Vector};
use std::rc::Rc;

/// Lower-level NLP interface (post-`TNLPAdapter`). Equality and
/// inequality constraints are already separated; bounds are already
/// classified into `x_l_map` / `x_u_map` / etc.
///
/// This is the equivalent of upstream `Ipopt::NLP`.
pub trait Nlp {
    fn n(&self) -> Index;
    fn m_eq(&self) -> Index;
    fn m_ineq(&self) -> Index;

    fn eval_f(&mut self, x: &dyn Vector) -> Number;
    fn eval_grad_f(&mut self, x: &dyn Vector, g: &mut dyn Vector);
    fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector);
    fn eval_d(&mut self, x: &dyn Vector, d: &mut dyn Vector);
    fn eval_jac_c(&mut self, x: &dyn Vector) -> Rc<dyn Matrix>;
    fn eval_jac_d(&mut self, x: &dyn Vector) -> Rc<dyn Matrix>;
    fn eval_h(
        &mut self,
        x: &dyn Vector,
        obj_factor: Number,
        y_c: &dyn Vector,
        y_d: &dyn Vector,
    ) -> Rc<dyn SymMatrix>;
}

/// Algorithm-side NLP (adds scaling-aware variants and provides the
/// bound expansion matrices `Px_L`, `Px_U`, `Pd_L`, `Pd_U`). Mirrors
/// upstream `Ipopt::IpoptNLP`.
pub trait IpoptNlp: Nlp {
    fn x_l(&self) -> &dyn Vector;
    fn x_u(&self) -> &dyn Vector;
    fn d_l(&self) -> &dyn Vector;
    fn d_u(&self) -> &dyn Vector;

    /// Bound expansion matrices: `Px_L` extracts the
    /// `x` components that have a finite lower bound, etc.
    fn px_l(&self) -> Rc<dyn Matrix>;
    fn px_u(&self) -> Rc<dyn Matrix>;
    fn pd_l(&self) -> Rc<dyn Matrix>;
    fn pd_u(&self) -> Rc<dyn Matrix>;

    /// Fill `x` with the initial primal values (mirrors upstream
    /// `IpoptNLP::GetStartingPoint`'s `init_x` flag). Default impl
    /// leaves `x` at its current contents (typically the zero vector
    /// produced by `make_new`).
    fn get_starting_x(&mut self, _x: &mut dyn Vector) -> bool {
        true
    }

    /// Fill `y_c` / `y_d` with initial multiplier guesses (mirrors
    /// `IpoptNLP::GetStartingPoint`'s `init_lambda` flag). Default
    /// impl leaves them at their current contents (zeros).
    fn get_starting_y(&mut self, _y_c: &mut dyn Vector, _y_d: &mut dyn Vector) -> bool {
        true
    }

    /// Fill `z_l` / `z_u` / `v_l` / `v_u` with initial bound-multiplier
    /// guesses (mirrors `init_z`). Default impl leaves them at zeros.
    #[allow(clippy::too_many_arguments)]
    fn get_starting_z(
        &mut self,
        _z_l: &mut dyn Vector,
        _z_u: &mut dyn Vector,
        _v_l: &mut dyn Vector,
        _v_u: &mut dyn Vector,
    ) -> bool {
        true
    }

    /// Lift a compressed `x_var` (length `n_x_var`) to the full-x
    /// length (`n_full_x` = user TNLP's `n`), splicing fixed-variable
    /// values back in. Used at finalize-solution time to hand the user
    /// a full-length x. Default impl returns x as-is, valid when the
    /// problem has no fixed variables.
    fn lift_x_to_full(&self, x: &dyn Vector) -> Vec<Number> {
        let dx = x
            .as_any()
            .downcast_ref::<DenseVector>()
            .expect("IpoptNlp::lift_x_to_full expects DenseVector");
        dx.expanded_values().to_vec()
    }

    /// Pack the algorithm-side `(y_c, y_d)` constraint multipliers into
    /// the user TNLP's `lambda` array (length `n_full_g`, ordered by
    /// the original `g` index). Used by `GetIpoptCurrentIterate` and
    /// `finalize_solution`. Default impl returns an empty vector — the
    /// canonical `OrigIpoptNlp` implementation overrides it to perform
    /// the c/d-split inverse and scaling unwind.
    fn pack_lambda_for_user(&self, _y_c: &dyn Vector, _y_d: &dyn Vector) -> Vec<Number> {
        Vec::new()
    }

    /// Pack the algorithm-side `(c, d)` constraint values into the user
    /// TNLP's `g` array (length `n_full_g`, ordered by the original `g`
    /// index, in user-unscaled space). Default impl returns an empty
    /// vector; `OrigIpoptNlp` overrides.
    fn pack_g_for_user(&self, _c: &dyn Vector, _d: &dyn Vector) -> Vec<Number> {
        Vec::new()
    }

    /// Expand a compressed lower-bound-multiplier vector
    /// (length = number of finite-lower-bound free variables) into the
    /// user TNLP's full-`n` length `z_L` array. Default impl returns an
    /// empty vector; `OrigIpoptNlp` overrides.
    fn pack_z_l_for_user(&self, _z_l: &dyn Vector) -> Vec<Number> {
        Vec::new()
    }

    /// Expand a compressed upper-bound-multiplier vector into the user
    /// TNLP's full-`n` length `z_U` array. Default impl returns an
    /// empty vector; `OrigIpoptNlp` overrides.
    fn pack_z_u_for_user(&self, _z_u: &dyn Vector) -> Vec<Number> {
        Vec::new()
    }

    /// Number of variables `n` as the user TNLP declared it (= `n_full_x`,
    /// before fixed-variable elimination). Used by inspector entry
    /// points that need to size full-`n` buffers. Default impl returns
    /// 0; `OrigIpoptNlp` overrides.
    fn n_full_x(&self) -> Index {
        0
    }

    /// Number of constraints `m` as the user TNLP declared it (= `n_full_g`).
    /// Default impl returns 0; `OrigIpoptNlp` overrides.
    fn n_full_g(&self) -> Index {
        0
    }

    /// Lift the algorithm-side `(y_c, y_d)` multipliers back to the
    /// user TNLP's `lambda` array (length `m_full = n_c + n_d`),
    /// matching upstream `IpOrigIpoptNLP::FinalizeSolution`. Sibling
    /// to `pack_lambda_for_user`; added by pounce#11 for the
    /// `finalize_solution` path. Default returns empty; `OrigIpoptNlp`
    /// overrides.
    fn finalize_solution_lambda(
        &self,
        _y_c: &dyn Vector,
        _y_d: &dyn Vector,
    ) -> Vec<Number> {
        Vec::new()
    }

    /// Lift compressed `z_l` back to full-x. Sibling to
    /// `pack_z_l_for_user`; added by pounce#11. Default returns empty.
    fn finalize_solution_z_l(&self, _z_l: &dyn Vector) -> Vec<Number> {
        Vec::new()
    }

    /// Lift compressed `z_u` back to full-x. Sibling to
    /// `pack_z_u_for_user`; added by pounce#11. Default returns empty.
    fn finalize_solution_z_u(&self, _z_u: &dyn Vector) -> Vec<Number> {
        Vec::new()
    }
}
