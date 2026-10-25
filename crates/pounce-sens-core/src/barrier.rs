//! What the corrector and the activity classification read off a converged
//! barrier iterate, beyond the back-solve [`SensBacksolver`] provides.
//!
//! Both work in the compound layout every engine here shares with pounce's
//! interior-point solver: one flat vector holding eight blocks in the order
//!
//! ```text
//! x, s, y_c, y_d, z_l, z_u, v_l, v_u
//! ```
//!
//! where `s` is the slack of the inequality rows, `y_c` and `y_d` are the
//! multipliers of the equality and inequality rows, `z_l` and `z_u` the
//! multipliers of the finite variable bounds, and `v_l` and `v_u` those of
//! the finite inequality-row bounds. The signs follow Ipopt's:
//!
//! ```text
//! ∇ₓL = ∇f + J_cᵀ y_c + J_dᵀ y_d − P_xl z_l + P_xu z_u
//! ∇ₛL = −y_d − P_dl v_l + P_du v_u
//! ```
//!
//! An engine that implements [`BarrierIterate`] gets [`crate::corrector`] and
//! [`crate::activity`] over its own converged point. pounce's interior-point
//! solver implements it in `pounce-sensitivity`, and an engine outside pounce
//! implements it from its own final iterate, its own model callbacks and its
//! own factorization of the KKT matrix.

use std::rc::Rc;

use pounce_common::types::Number;
use pounce_linalg::{Matrix, SymMatrix};

use crate::backsolver::SensBacksolver;

/// A converged iterate of a barrier method, with the evaluations the
/// corrector and the activity classification need.
///
/// Two frames appear below. The **natural** frame is the model as the user
/// wrote it, and it is the frame [`SensBacksolver::solve`] answers in. The
/// **engine** frame is whatever the engine iterated in, which differs from the
/// natural one when it scaled the model. [`SensBacksolver::natural_units_factor`]
/// carries a quantity from the engine frame into the natural one, and
/// [`Self::scaled_rhs_factor`] carries a right-hand side the other way. An
/// engine that did not scale returns `None` from both and has one frame.
pub trait BarrierIterate: SensBacksolver {
    /// Lengths of the eight blocks, in the order `x, s, y_c, y_d, z_l, z_u,
    /// v_l, v_u`. They sum to [`SensBacksolver::dim`].
    fn block_dims(&self) -> [usize; 8];

    /// Where each block starts in the flat vector, with the total length in
    /// the last entry.
    fn offsets(&self) -> [usize; 9] {
        let d = self.block_dims();
        let mut o = [0usize; 9];
        for i in 0..8 {
            o[i + 1] = o[i] + d[i];
        }
        o
    }

    /// The barrier parameter the converged point sits on, as the level of
    /// the complementarity products of the slacks and multipliers this engine
    /// reports. These are the products [`Self::barrier_residual`] writes, at
    /// the point [`Self::converged_iterate`] writes, in the engine frame.
    ///
    /// An engine reports the multipliers and `μ` in the frame its own
    /// iteration ran in, and sets [`ActivityData::obj_scale`] to the factor it
    /// multiplied the objective by. The classification's ratios do not depend
    /// on that frame, but its band edges come from `√μ`, which does. In that
    /// frame an engine classifies as pounce does. Ipopt
    /// multiplies the objective, and with it every bound multiplier, by a
    /// scaling factor `f` it prints, and its `.sol` reports the multipliers
    /// divided by `f` again. An engine built from Ipopt's result therefore
    /// multiplies those multipliers by `f`, returns the `μ` Ipopt printed, and
    /// sets `obj_scale` to `f`.
    ///
    /// Reporting the unscaled multipliers with `μ / f` instead is consistent,
    /// since each product then sits at `μ / f`. But the band edges move by
    /// `1/√f` from pounce's, and the corrector measures its residual in a
    /// different frame.
    fn barrier_mu(&self) -> Number;

    /// Writes the converged iterate into `out`, in the engine frame. The error
    /// says why there is none, or that `out` is the wrong length.
    fn converged_iterate(&self, out: &mut [Number]) -> Result<(), String>;

    /// Per-row factor `E` that takes a natural-units right-hand side into the
    /// engine frame, indexed like the compound vector. `None` is the
    /// identity.
    ///
    /// The corrector reads its residual in the engine frame and hands it to
    /// a solve that expects natural units, so it divides by this first.
    fn scaled_rhs_factor(&self) -> Option<&[Number]> {
        None
    }

    /// Writes the barrier KKT residual at `point` into `out`, block by block
    /// in the compound layout and in the engine frame. `point` is in the
    /// engine frame too.
    ///
    /// The eight blocks are `∇ₓL`, `∇ₛL`, `c(x)`, `d(x) − s`, and the four
    /// complementarity products `(x − x_l) z_l`, `(x_u − x) z_u`,
    /// `(s − d_l) v_l` and `(d_u − s) v_u`, each over the bounds that exist.
    /// The products carry no barrier shift. The corrector subtracts `μ`
    /// itself, and it moves the equality rows a parametric perturbation
    /// pinned, so `c(x)` here is measured against the unperturbed
    /// right-hand side.
    ///
    /// A point outside the domain of the model's functions writes non-finite
    /// entries rather than failing. The error is for a malformed call.
    fn barrier_residual(&self, point: &[Number], out: &mut [Number]) -> Result<(), String>;

    /// The Newton operator of the barrier system at a predicted point, for
    /// the corrector to iterate against.
    ///
    /// `predicted` is a compound vector in natural units. The operator is
    /// the Jacobian of the residual [`Self::barrier_residual`] describes,
    /// evaluated there, with two changes:
    ///
    /// * Each `(row, add)` in `pinned` names a primal row, a row of the `x`
    ///   or `s` block, whose diagonal gets `add` on top of what the barrier
    ///   puts there. These are bounds the step brought into the active set.
    /// * With `exact_hessian` the Hessian of the Lagrangian is evaluated at
    ///   `predicted`, with its multipliers. Without it the engine keeps the
    ///   Hessian it last held, which is all a quasi-Newton engine has.
    ///
    /// A bound the corrector took out of the active set needs nothing from
    /// the engine. Its multiplier is zero at `predicted` and the corrector
    /// zeroes its row of every right-hand side, so with a positive slack the
    /// bound adds nothing to the diagonal and its row returns a zero step.
    ///
    /// The returned operator answers in natural units like
    /// [`SensBacksolver::solve`]. An engine whose evaluations depend on state
    /// it changed to build the operator restores that state when the operator
    /// is dropped.
    fn corrector_operator<'a>(
        &'a self,
        predicted: &[Number],
        pinned: &[(usize, Number)],
        exact_hessian: bool,
    ) -> Result<Box<dyn KktOperator + 'a>, String>;

    /// Everything the activity classification reads off the converged point.
    fn activity_data(&self) -> Result<ActivityData, String>;

    /// How the engine's variables and rows correspond to the model's.
    fn model_maps(&self) -> Result<ModelMaps, String>;

    /// The scaling the engine applied to the model.
    /// [`EngineScaling::none`] is the answer for an engine that applied none.
    fn scaling(&self) -> EngineScaling;

    /// The bounds the engine iterated against over the primal prefix of the
    /// compound vector, `x` then `s`, in the engine frame: the variables'
    /// bounds, then the inequality rows' limits. An entry with no bound on a
    /// side holds an infinity there.
    ///
    /// These are the bounds the slacks of [`Self::activity_data`] are
    /// measured from, so they carry whatever relaxation the engine applied.
    fn primal_bounds(&self) -> Result<(Vec<Number>, Vec<Number>), String>;

    /// The perturbations `(δ_x, δ_s, δ_c, δ_d)` the engine added to its KKT
    /// matrix to correct its inertia, as its factor holds them. `None` when
    /// the engine does not report them, which is the default.
    ///
    /// All four zero means every back-solve inverts the exact KKT matrix.
    /// Anything else perturbs every sensitivity read off the factor, the
    /// covariance in particular.
    fn kkt_perturbations(&self) -> Option<[Number; 4]> {
        None
    }

    /// An operator for the system with the bounds in `released`, compound
    /// bound-multiplier rows, taken out of the active set, for a calculation
    /// that solves against one released system many times.
    ///
    /// Each solve answers as [`SensBacksolver::solve_released`] does. The
    /// default calls it on every solve. An engine that can factor the
    /// released system once and back-solve against it thereafter overrides
    /// this, and the answers do not change.
    fn released_operator<'a>(
        &'a self,
        released: &[usize],
    ) -> Result<Box<dyn KktOperator + 'a>, String> {
        Ok(Box::new(ReleasedBySolve {
            bs: self,
            released: released.to_vec(),
        }))
    }
}

/// [`BarrierIterate::released_operator`]'s default: every solve is a call
/// to [`SensBacksolver::solve_released`].
struct ReleasedBySolve<'a, B: SensBacksolver + ?Sized> {
    bs: &'a B,
    released: Vec<usize>,
}

impl<B: SensBacksolver + ?Sized> KktOperator for ReleasedBySolve<'_, B> {
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.bs.solve_released(&self.released, rhs, lhs)
    }
}

/// A factored KKT operator, solved against in natural units. The corrector
/// iterates against one (see [`BarrierIterate::corrector_operator`]), and
/// the bound-aware steps solve against a released one (see
/// [`BarrierIterate::released_operator`]).
pub trait KktOperator {
    /// Solves `K · lhs = rhs` in natural units. Returns `false` on failure.
    ///
    /// The solve has to be refined against the matrix, as Ipopt's and
    /// pounce's are. A bound held on its constraint gives `K` a diagonal entry
    /// many orders above the rest, and an unrefined solve loses the digits
    /// that entry costs. Where a step direction is roundoff-sized, that can
    /// change its sign, and the corrector's iterations then end differently
    /// from pounce's.
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool;
}

/// The bounds on one side of one block: which entries carry a finite bound,
/// and at each of them the multiplier and the slack, in that order.
///
/// `pos` indexes the block the bound constrains, the `x` block for a
/// variable bound and the `s` block for an inequality-row bound. `mult[k]`
/// and `slack[k]` belong to entry `pos[k]`. The slack is the distance to the
/// bound the engine iterated against, which is the declared bound widened by
/// whatever relaxation the engine applied.
#[derive(Debug, Clone, Default)]
pub struct BoundSide {
    /// Entry of the constrained block, per bound.
    pub pos: Vec<usize>,
    /// The bound's multiplier, per bound.
    pub mult: Vec<Number>,
    /// The distance from the entry to its bound, per bound.
    pub slack: Vec<Number>,
}

/// How an engine's variables and rows correspond to the model's.
///
/// An engine may drop variables whose bounds coincide, and it splits the
/// model's rows into equality rows, the `y_c` block, and inequality rows, the
/// `s` and `y_d` blocks, so neither numbering is the model's. Every result a
/// caller reads in the model's indices goes through these maps.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMaps {
    /// The number of variables the model declares, counting any the engine
    /// removed because their bounds coincide.
    pub n_full_x: usize,
    /// The model's index of each variable the engine kept.
    pub var_x_to_full_x: Vec<usize>,
    /// For each of the model's rows, its position among the equality rows,
    /// or `None` for an inequality row.
    pub full_g_to_c: Vec<Option<usize>>,
    /// For each of the model's rows, its position among the inequality rows,
    /// or `None` for an equality row.
    pub full_g_to_d: Vec<Option<usize>>,
}

impl ModelMaps {
    /// The number of rows the model declares.
    pub fn n_full_g(&self) -> usize {
        self.full_g_to_c.len()
    }

    /// The engine's index of each of the model's variables, `None` for one
    /// the engine removed.
    pub fn full_x_to_var_x(&self) -> Vec<Option<usize>> {
        let mut out = vec![None; self.n_full_x];
        for (i, &f) in self.var_x_to_full_x.iter().enumerate() {
            if let Some(slot) = out.get_mut(f) {
                *slot = Some(i);
            }
        }
        out
    }

    /// Checks the maps against the engine's block lengths: one model
    /// variable per engine variable, each named once, and each model row in
    /// exactly one of the two row blocks, filling each block once.
    pub fn check(&self, dims: &[usize; 8]) -> Result<(), String> {
        let (n, m_c, m_d) = (dims[0], dims[2], dims[3]);
        if self.var_x_to_full_x.len() != n {
            return Err(format!(
                "model maps: {} entries in the variable map for {n} variables",
                self.var_x_to_full_x.len()
            ));
        }
        let mut seen = vec![false; self.n_full_x];
        for &i in &self.var_x_to_full_x {
            match seen.get_mut(i) {
                Some(slot) if !*slot => *slot = true,
                _ => {
                    return Err(format!(
                        "model maps: the variable map names variable {i} twice or out of {}",
                        self.n_full_x
                    ));
                }
            }
        }
        if self.full_g_to_d.len() != self.full_g_to_c.len() {
            return Err(format!(
                "model maps: {} rows in the equality map and {} in the inequality map",
                self.full_g_to_c.len(),
                self.full_g_to_d.len()
            ));
        }
        let mut seen_c = vec![false; m_c];
        let mut seen_d = vec![false; m_d];
        for (g, (c, d)) in self.full_g_to_c.iter().zip(&self.full_g_to_d).enumerate() {
            let (block, seen, p) = match (c, d) {
                (Some(p), None) => ("equality", &mut seen_c, *p),
                (None, Some(p)) => ("inequality", &mut seen_d, *p),
                _ => {
                    return Err(format!(
                        "model maps: row {g} is not in exactly one of the two row blocks"
                    ));
                }
            };
            match seen.get_mut(p) {
                Some(slot) if !*slot => *slot = true,
                _ => {
                    return Err(format!(
                        "model maps: the row map names {block} row {p} twice or out of {}",
                        seen.len()
                    ));
                }
            }
        }
        if seen_c.iter().chain(&seen_d).any(|s| !s) {
            return Err("model maps: the row map leaves an engine row unnamed".into());
        }
        Ok(())
    }
}

/// The scaling an engine applied to the model, as the factors it multiplied
/// each quantity by.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineScaling {
    /// The factor the engine multiplied the objective by, `1.0` when it did
    /// not, and negative for a maximization it turned into a minimization.
    pub objective: Number,
    /// The engine's change of variables `x̃ = d ⊙ x`, one factor per
    /// variable the engine kept, when it made one.
    pub variables: Option<Vec<Number>>,
    /// The factor the engine multiplied each equality row by, when it did.
    pub equality_rows: Option<Vec<Number>>,
    /// The factor the engine multiplied each inequality row by, when it did.
    pub inequality_rows: Option<Vec<Number>>,
}

impl EngineScaling {
    /// No scaling at all.
    pub fn none() -> Self {
        Self {
            objective: 1.0,
            variables: None,
            equality_rows: None,
            inequality_rows: None,
        }
    }

    /// Checks each factor vector against the block it scales.
    pub fn check(&self, dims: &[usize; 8]) -> Result<(), String> {
        for (what, v, len) in [
            ("variable", &self.variables, dims[0]),
            ("equality row", &self.equality_rows, dims[2]),
            ("inequality row", &self.inequality_rows, dims[3]),
        ] {
            if let Some(v) = v
                && v.len() != len
            {
                return Err(format!(
                    "engine scaling: {} {what} scale factors for {len} entries",
                    v.len()
                ));
            }
        }
        Ok(())
    }
}

/// The converged quantities the activity classification reads, in the
/// engine frame. See [`crate::activity`] for what it does with them. The
/// barrier parameter comes from [`BarrierIterate::barrier_mu`], and the maps
/// and scaling from [`BarrierIterate::model_maps`] and
/// [`BarrierIterate::scaling`].
pub struct ActivityData {
    /// Lower variable bounds.
    pub x_l: BoundSide,
    /// Upper variable bounds.
    pub x_u: BoundSide,
    /// Lower bounds of the inequality rows.
    pub d_l: BoundSide,
    /// Upper bounds of the inequality rows.
    pub d_u: BoundSide,
    /// The barrier diagonal of the `x` block, one entry per variable the
    /// engine kept, both sides summed.
    ///
    /// An engine that bounds the diagonal in no way of its own writes
    /// `mult / slack` from [`Self::x_l`] and [`Self::x_u`], summed. pounce
    /// writes the diagonal it factored instead, which differs from that sum
    /// where the gh#737 ceiling capped an entry or a crossover point took the
    /// declared-bound frame (gh#654).
    pub sigma_x: Vec<Number>,
    /// The barrier diagonal of the `s` block, one entry per inequality row,
    /// from [`Self::d_l`] and [`Self::d_u`] as for [`Self::sigma_x`].
    pub sigma_s: Vec<Number>,
    /// The exact Hessian of the Lagrangian at the converged point.
    pub hessian: Rc<dyn SymMatrix>,
    /// The Jacobian of the equality rows at the converged point.
    pub jac_c: Rc<dyn Matrix>,
    /// The Jacobian of the inequality rows at the converged point.
    pub jac_d: Rc<dyn Matrix>,
}

impl ActivityData {
    /// Checks the lengths agree with each other, so the classification can
    /// index without bounds failures on data an outside engine supplied.
    pub fn check(&self) -> Result<(), String> {
        let n = self.sigma_x.len();
        let m_d = self.sigma_s.len();
        let side = |name: &str, s: &BoundSide, len: usize| -> Result<(), String> {
            if s.mult.len() != s.pos.len() || s.slack.len() != s.pos.len() {
                return Err(format!(
                    "activity data: {name} has {} positions, {} multipliers and {} slacks",
                    s.pos.len(),
                    s.mult.len(),
                    s.slack.len()
                ));
            }
            if let Some(&p) = s.pos.iter().find(|&&p| p >= len) {
                return Err(format!(
                    "activity data: {name} names entry {p} of a block of {len}"
                ));
            }
            Ok(())
        };
        side("x_l", &self.x_l, n)?;
        side("x_u", &self.x_u, n)?;
        side("d_l", &self.d_l, m_d)?;
        side("d_u", &self.d_u, m_d)?;
        if self.hessian.n_rows() as usize != n || self.hessian.n_cols() as usize != n {
            return Err(format!(
                "activity data: the Hessian is {}x{}, expected {n}x{n}",
                self.hessian.n_rows(),
                self.hessian.n_cols()
            ));
        }
        if self.jac_d.n_rows() as usize != m_d || self.jac_d.n_cols() as usize != n {
            return Err(format!(
                "activity data: the inequality Jacobian is {}x{}, expected {m_d}x{n}",
                self.jac_d.n_rows(),
                self.jac_d.n_cols()
            ));
        }
        if self.jac_c.n_cols() as usize != n {
            return Err(format!(
                "activity data: the equality Jacobian has {} columns, expected {n}",
                self.jac_c.n_cols()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};

    /// Two variables, one equality row, one inequality row, a lower bound on
    /// each variable and an upper bound on the inequality row: consistent in
    /// every length.
    fn data() -> ActivityData {
        let hess = SymTMatrix::new(SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]));
        let jac = |m| GenTMatrix::new(GenTMatrixSpace::new(m, 2, vec![1, 1], vec![1, 2]));
        let side = |pos: Vec<usize>| BoundSide {
            mult: vec![1.0; pos.len()],
            slack: vec![1.0; pos.len()],
            pos,
        };
        ActivityData {
            x_l: side(vec![0, 1]),
            x_u: side(vec![]),
            d_l: side(vec![]),
            d_u: side(vec![0]),
            sigma_x: vec![1.0, 1.0],
            sigma_s: vec![1.0],
            hessian: Rc::new(hess),
            jac_c: Rc::new(jac(1)),
            jac_d: Rc::new(jac(1)),
        }
    }

    /// The block lengths of [`data`]'s engine: two variables, one slack,
    /// one equality and one inequality multiplier, two variable bounds and
    /// one row bound.
    const DIMS: [usize; 8] = [2, 1, 1, 1, 2, 0, 0, 1];

    /// Three model variables, of which the engine dropped the second, and
    /// two model rows, an inequality then an equality.
    fn maps() -> ModelMaps {
        ModelMaps {
            n_full_x: 3,
            var_x_to_full_x: vec![0, 2],
            full_g_to_c: vec![None, Some(0)],
            full_g_to_d: vec![Some(0), None],
        }
    }

    #[test]
    fn consistent_data_passes() {
        assert_eq!(data().check(), Ok(()));
        assert_eq!(maps().check(&DIMS), Ok(()));
        assert_eq!(EngineScaling::none().check(&DIMS), Ok(()));
    }

    #[test]
    fn a_bound_side_with_mismatched_lengths_is_refused() {
        let mut d = data();
        d.x_l.slack.pop();
        assert!(d.check().unwrap_err().contains("x_l"));
        let mut d = data();
        d.d_u.mult.push(1.0);
        assert!(d.check().unwrap_err().contains("d_u"));
    }

    #[test]
    fn a_bound_outside_its_block_is_refused() {
        let mut d = data();
        d.x_l.pos[1] = 2;
        assert!(d.check().unwrap_err().contains("entry 2"));
        let mut d = data();
        d.d_u.pos[0] = 1;
        assert!(d.check().unwrap_err().contains("entry 1"));
    }

    #[test]
    fn matrices_of_the_wrong_shape_are_refused() {
        let mut d = data();
        d.hessian = Rc::new(SymTMatrix::new(SymTMatrixSpace::new(3, vec![], vec![])));
        assert!(d.check().unwrap_err().contains("Hessian"));
        let mut d = data();
        d.jac_d = Rc::new(GenTMatrix::new(GenTMatrixSpace::new(1, 3, vec![], vec![])));
        assert!(d.check().unwrap_err().contains("inequality Jacobian"));
        let mut d = data();
        d.jac_c = Rc::new(GenTMatrix::new(GenTMatrixSpace::new(1, 3, vec![], vec![])));
        assert!(d.check().unwrap_err().contains("equality Jacobian"));
    }

    #[test]
    fn scale_factors_of_the_wrong_length_are_refused() {
        for (scaling, what) in [
            (
                EngineScaling {
                    variables: Some(vec![1.0]),
                    ..EngineScaling::none()
                },
                "variable scale",
            ),
            (
                EngineScaling {
                    equality_rows: Some(vec![1.0, 1.0]),
                    ..EngineScaling::none()
                },
                "equality row scale",
            ),
            (
                EngineScaling {
                    inequality_rows: Some(vec![]),
                    ..EngineScaling::none()
                },
                "inequality row scale",
            ),
        ] {
            assert!(scaling.check(&DIMS).unwrap_err().contains(what), "{what}");
        }
    }

    #[test]
    fn a_variable_map_that_repeats_overruns_or_has_the_wrong_length_is_refused() {
        let mut m = maps();
        m.var_x_to_full_x.pop();
        assert!(m.check(&DIMS).unwrap_err().contains("variable map for"));
        let mut m = maps();
        m.var_x_to_full_x[1] = 3;
        assert!(m.check(&DIMS).unwrap_err().contains("names variable 3"));
        let mut m = maps();
        m.var_x_to_full_x[1] = 0;
        assert!(
            m.check(&DIMS)
                .unwrap_err()
                .contains("names variable 0 twice")
        );
    }

    #[test]
    fn a_row_map_that_repeats_skips_splits_or_overruns_is_refused() {
        let mut m = maps();
        m.full_g_to_d = vec![Some(0), Some(0)];
        assert!(m.check(&DIMS).unwrap_err().contains("exactly one"));
        let mut m = maps();
        m.full_g_to_d = vec![Some(1), None];
        assert!(m.check(&DIMS).unwrap_err().contains("twice or out of"));
        let mut m = maps();
        m.full_g_to_c = vec![None, Some(0), Some(0)];
        assert!(m.check(&DIMS).unwrap_err().contains("equality map"));
        let mut m = maps();
        m.full_g_to_c = vec![None, None];
        m.full_g_to_d = vec![Some(0), None];
        assert!(m.check(&DIMS).unwrap_err().contains("exactly one"));
        let mut m = maps();
        m.full_g_to_c = vec![Some(0), Some(0)];
        m.full_g_to_d = vec![None, None];
        assert!(m.check(&DIMS).unwrap_err().contains("twice or out of"));
        let m = ModelMaps {
            full_g_to_c: vec![Some(0)],
            full_g_to_d: vec![None],
            ..maps()
        };
        assert!(m.check(&DIMS).unwrap_err().contains("unnamed"));
    }

    #[test]
    fn the_inverse_variable_map_marks_a_dropped_variable() {
        assert_eq!(maps().full_x_to_var_x(), vec![Some(0), None, Some(1)]);
        assert_eq!(maps().n_full_g(), 2);
    }
}
