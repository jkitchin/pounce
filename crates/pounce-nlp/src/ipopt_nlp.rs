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

/// Human-readable names projected into the algorithm's *split* space —
/// the index space the debugger reports residuals in, where equality and
/// inequality constraints are separated and fixed variables are removed.
///
/// Each vector is indexed by the split-space position (`x_var[j]` is the
/// `j`-th free variable, `eq[k]` the `k`-th equality constraint, `ineq[k]`
/// the `k`-th inequality), and each entry is `Some(name)` when the model
/// carried one or `None` to fall back to an index label. Producing this
/// requires composing the TNLP's original-order names with the
/// fixed-variable and c/d-split permutations, which is why it lives on
/// the NLP rather than being read directly off the TNLP.
///
/// Names are what turn "variables 1, 132, 439 in equations 3, 15" into a
/// model-level diagnosis — the gap Lee et al. (2024,
/// <https://doi.org/10.69997/sct.147875>) call out for equation-oriented
/// model debugging.
#[derive(Debug, Clone, Default)]
pub struct SplitNames {
    /// Names of the free variables, in algorithm-side `x` order (`n()`).
    pub x_var: Vec<Option<String>>,
    /// Names of the equality constraints, in `c` order (`m_eq()`).
    pub eq: Vec<Option<String>>,
    /// Names of the inequality constraints, in `d` order (`m_ineq()`).
    pub ineq: Vec<Option<String>>,
}

impl SplitNames {
    /// Whether any entry carries a name. An all-`None` projection (e.g.
    /// the model shipped no `.col`/`.row` files, or presolve declined to
    /// forward names) is reported as "no names available" so the debugger
    /// falls back to index labels rather than printing blanks.
    pub fn any_present(&self) -> bool {
        self.x_var
            .iter()
            .chain(self.eq.iter())
            .chain(self.ineq.iter())
            .any(Option::is_some)
    }
}

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
    /// Per-evaluation call counts accumulated over the solve, ordered
    /// `[f, grad_f, c, d, jac_c, jac_d, h]`. Populates the end-of-run
    /// summary's evaluation tallies (#206). Default is all zeros for
    /// implementors that do not count; [`OrigIpoptNlp`] reports its live
    /// counters.
    fn eval_counts(&self) -> [Index; 7] {
        [0; 7]
    }

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

    /// Replace the `x_L / x_U / d_L / d_U` bounds in place. Invoked by the
    /// algorithm's accept step when the safe-slack mechanism moved one or
    /// more bounds (port of `IpoptNLP::AdjustVariableBounds`,
    /// `IpOrigIpoptNLP.cpp:990-1001`). Default is a no-op for NLP
    /// implementations that do not own mutable bound storage.
    fn adjust_variable_bounds(
        &mut self,
        _new_x_l: &dyn Vector,
        _new_x_u: &dyn Vector,
        _new_d_l: &dyn Vector,
        _new_d_u: &dyn Vector,
    ) {
    }

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
    fn finalize_solution_lambda(&self, _y_c: &dyn Vector, _y_d: &dyn Vector) -> Vec<Number> {
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

    /// Map a 0-based **full-x** index (user-TNLP space, length
    /// `n_full_x()`) to a 0-based **var-x** index (algorithm-side,
    /// length `n()`). Returns `None` when the variable was eliminated
    /// because `x_l[i] == x_u[i]` under
    /// `fixed_variable_treatment = make_parameter`.
    ///
    /// Default impl assumes no fixed variables (identity mapping). The
    /// `OrigIpoptNlp` implementation consults
    /// `BoundClassification::full_to_var`.
    fn full_x_to_var_x(&self, full_idx: Index) -> Option<Index> {
        Some(full_idx)
    }

    /// Map a 0-based **full-g** index (user-TNLP space, length
    /// `n_full_g()`) to a 0-based position in the c-block (algorithm-side
    /// equality multiplier vector `y_c`, length `m_eq()`). Returns
    /// `None` when the constraint is an inequality (lives in `d`, not
    /// `c`).
    ///
    /// Default impl assumes the c-block matches the user's g order
    /// (no c/d split); `OrigIpoptNlp` overrides via
    /// `BoundClassification::c_map`.
    fn full_g_to_c_block(&self, full_idx: Index) -> Option<Index> {
        Some(full_idx)
    }

    /// Inverse of [`Self::full_x_to_var_x`]: map a 0-based var-x index
    /// (length `n()`) to the corresponding full-x index (length
    /// `n_full_x()`). Used when scattering a compressed step or
    /// iterate back into the user's full-x array.
    ///
    /// Default impl assumes no fixed variables (identity); `OrigIpoptNlp`
    /// returns `classification.x_not_fixed_map[var_idx]`.
    fn var_x_to_full_x(&self, var_idx: Index) -> Index {
        var_idx
    }

    /// Effective objective scaling factor (`df_` upstream): the value
    /// `f` is multiplied by inside [`Self::eval_f`]. Used to recover the
    /// unscaled objective for display. Default `1.0` (no scaling);
    /// `OrigIpoptNlp` overrides.
    fn obj_scaling_factor(&self) -> Number {
        1.0
    }

    /// The **solver-computed** part of the objective scale, before the user's
    /// constant `obj_scaling_factor` is multiplied in.
    ///
    /// [`Self::obj_scaling_factor`] returns the product `df * user_factor`,
    /// which is the right thing for unscaling a residual but the wrong thing
    /// for asking *why* the scale is small. `df` is what gradient-based scaling
    /// computed and clamped at `nlp_scaling_min_value`; the user factor is a
    /// deliberate choice. Only the former can mask a certificate (gh #200), so
    /// the termination logic keys on this rather than on the product.
    /// Default `1.0`; `OrigIpoptNlp` overrides.
    fn computed_obj_scaling_factor(&self) -> Number {
        1.0
    }

    /// Per-row scaling vector for the equality block (`dc_` upstream):
    /// the factor each `c` row is multiplied by inside [`Self::eval_c`]
    /// / [`Self::eval_jac_c`]. `None` ⇔ no row scaling (all 1.0);
    /// length `m_eq()` when present. Together with
    /// [`Self::obj_scaling_factor`] and [`Self::d_scale_vec`] this is
    /// what lets `pounce-sensitivity` undo the NLP scaling baked into
    /// the converged KKT factor (pounce#128). Default `None`;
    /// `OrigIpoptNlp` overrides.
    fn c_scale_vec(&self) -> Option<Vec<Number>> {
        None
    }

    /// Per-row scaling vector for the inequality block (`dd_`
    /// upstream), same convention as [`Self::c_scale_vec`]. Length
    /// `m_ineq()` when present. Default `None`; `OrigIpoptNlp`
    /// overrides.
    fn d_scale_vec(&self) -> Option<Vec<Number>> {
        None
    }

    /// Human-readable variable / constraint names projected into the
    /// algorithm's split space (free variables, equalities, inequalities),
    /// or `None` when the model carries no names. The debugger uses this to
    /// label residuals by model name (`mass_balance`) rather than index
    /// (`c[3]`) — see [`SplitNames`] and Lee et al. (2024,
    /// <https://doi.org/10.69997/sct.147875>).
    ///
    /// Default returns `None`; `OrigIpoptNlp` overrides by pulling
    /// `idx_names` metadata from the underlying TNLP and composing it with
    /// the bound / c-d-split permutations.
    fn split_space_names(&self) -> Option<SplitNames> {
        None
    }
}
