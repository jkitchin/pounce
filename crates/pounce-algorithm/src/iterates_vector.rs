//! Eight-component iterate — port of
//! `Algorithm/IpIteratesVector.{hpp,cpp}`.
//!
//! Concrete struct with named fields rather than upstream's
//! `CompoundVector` slot-by-index. Same components, same indexing
//! convention:
//!
//! | slot | name | meaning                                  |
//! |------|------|------------------------------------------|
//! |  0   | x    | primal variables                          |
//! |  1   | s    | inequality slacks                         |
//! |  2   | y_c  | equality multipliers                      |
//! |  3   | y_d  | inequality multipliers                    |
//! |  4   | z_l  | x lower-bound multipliers                 |
//! |  5   | z_u  | x upper-bound multipliers                 |
//! |  6   | v_l  | s lower-bound multipliers                 |
//! |  7   | v_u  | s upper-bound multipliers                 |
//!
//! Components are `Rc<dyn Vector>` to keep upstream's shared-ownership
//! semantics (the same `x` lives in `curr`, `delta`, etc., until
//! someone replaces it via `set_*`).

use pounce_linalg::Vector;
use std::rc::Rc;

/// Eight-component iterate vector. Cheap to clone via `Rc`.
#[derive(Clone)]
pub struct IteratesVector {
    pub x: Rc<dyn Vector>,
    pub s: Rc<dyn Vector>,
    pub y_c: Rc<dyn Vector>,
    pub y_d: Rc<dyn Vector>,
    pub z_l: Rc<dyn Vector>,
    pub z_u: Rc<dyn Vector>,
    pub v_l: Rc<dyn Vector>,
    pub v_u: Rc<dyn Vector>,
}

impl std::fmt::Debug for IteratesVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IteratesVector")
            .field("x_dim", &self.x.dim())
            .field("s_dim", &self.s.dim())
            .field("y_c_dim", &self.y_c.dim())
            .field("y_d_dim", &self.y_d.dim())
            .field("z_l_dim", &self.z_l.dim())
            .field("z_u_dim", &self.z_u.dim())
            .field("v_l_dim", &self.v_l.dim())
            .field("v_u_dim", &self.v_u.dim())
            .finish()
    }
}

impl IteratesVector {
    /// Construct from eight already-allocated component vectors.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: Rc<dyn Vector>,
        s: Rc<dyn Vector>,
        y_c: Rc<dyn Vector>,
        y_d: Rc<dyn Vector>,
        z_l: Rc<dyn Vector>,
        z_u: Rc<dyn Vector>,
        v_l: Rc<dyn Vector>,
        v_u: Rc<dyn Vector>,
    ) -> Self {
        Self {
            x,
            s,
            y_c,
            y_d,
            z_l,
            z_u,
            v_l,
            v_u,
        }
    }

    /// Clone this iterate with `x` swapped out and every other
    /// component shared. `Rc` makes it a pointer copy, so this is the
    /// cheap way to re-stage a candidate `x` for a CQ evaluation
    /// without rebuilding the other seven blocks.
    pub fn with_x(&self, x: Rc<dyn Vector>) -> Self {
        Self { x, ..self.clone() }
    }

    /// Total dimension across all eight components.
    pub fn dim(&self) -> i32 {
        self.x.dim()
            + self.s.dim()
            + self.y_c.dim()
            + self.y_d.dim()
            + self.z_l.dim()
            + self.z_u.dim()
            + self.v_l.dim()
            + self.v_u.dim()
    }

    /// Max-norm across all eight components — port of
    /// `IteratesVector::Amax()` (which itself is `CompoundVector::Amax`,
    /// the max of per-block `Amax`).
    pub fn amax(&self) -> pounce_common::types::Number {
        let mut m = self.x.amax();
        for v in [
            &self.s, &self.y_c, &self.y_d, &self.z_l, &self.z_u, &self.v_l, &self.v_u,
        ] {
            let a = v.amax();
            if a > m {
                m = a;
            }
        }
        m
    }

    /// Allocate a fresh, zero-initialized iterate with the same shape.
    /// Equivalent to upstream `MakeNewIteratesVector(true)`.
    pub fn make_new_zeroed(&self) -> IteratesVectorMut {
        IteratesVectorMut {
            x: self.x.make_new(),
            s: self.s.make_new(),
            y_c: self.y_c.make_new(),
            y_d: self.y_d.make_new(),
            z_l: self.z_l.make_new(),
            z_u: self.z_u.make_new(),
            v_l: self.v_l.make_new(),
            v_u: self.v_u.make_new(),
        }
    }

    /// Deep copy — equivalent to upstream `MakeNewIteratesVectorCopy()`.
    pub fn deep_copy(&self) -> IteratesVectorMut {
        let mut out = self.make_new_zeroed();
        out.x.copy(&*self.x);
        out.s.copy(&*self.s);
        out.y_c.copy(&*self.y_c);
        out.y_d.copy(&*self.y_d);
        out.z_l.copy(&*self.z_l);
        out.z_u.copy(&*self.z_u);
        out.v_l.copy(&*self.v_l);
        out.v_u.copy(&*self.v_u);
        out
    }
}

/// Owned, mutable variant — used as the working-storage form of an
/// IteratesVector (typical use: a freshly-allocated solution slot the
/// solver writes into). Convertible into `IteratesVector` via `freeze`.
pub struct IteratesVectorMut {
    pub x: Box<dyn Vector>,
    pub s: Box<dyn Vector>,
    pub y_c: Box<dyn Vector>,
    pub y_d: Box<dyn Vector>,
    pub z_l: Box<dyn Vector>,
    pub z_u: Box<dyn Vector>,
    pub v_l: Box<dyn Vector>,
    pub v_u: Box<dyn Vector>,
}

impl IteratesVectorMut {
    /// Convert into the shareable `Rc`-backed form.
    pub fn freeze(self) -> IteratesVector {
        IteratesVector::new(
            Rc::from(self.x),
            Rc::from(self.s),
            Rc::from(self.y_c),
            Rc::from(self.y_d),
            Rc::from(self.z_l),
            Rc::from(self.z_u),
            Rc::from(self.v_l),
            Rc::from(self.v_u),
        )
    }

    pub fn amax(&self) -> pounce_common::types::Number {
        let mut m = self.x.amax();
        for v in [
            &self.s, &self.y_c, &self.y_d, &self.z_l, &self.z_u, &self.v_l, &self.v_u,
        ] {
            let a = v.amax();
            if a > m {
                m = a;
            }
        }
        m
    }

    /// Scale every component by `alpha` — port of `IteratesVector::Scal`.
    pub fn scal(&mut self, alpha: pounce_common::types::Number) {
        self.x.scal(alpha);
        self.s.scal(alpha);
        self.y_c.scal(alpha);
        self.y_d.scal(alpha);
        self.z_l.scal(alpha);
        self.z_u.scal(alpha);
        self.v_l.scal(alpha);
        self.v_u.scal(alpha);
    }

    /// `self += alpha * other` per component — port of `IteratesVector::Axpy`.
    pub fn axpy(&mut self, alpha: pounce_common::types::Number, other: &IteratesVector) {
        self.x.axpy(alpha, &*other.x);
        self.s.axpy(alpha, &*other.s);
        self.y_c.axpy(alpha, &*other.y_c);
        self.y_d.axpy(alpha, &*other.y_d);
        self.z_l.axpy(alpha, &*other.z_l);
        self.z_u.axpy(alpha, &*other.z_u);
        self.v_l.axpy(alpha, &*other.v_l);
        self.v_u.axpy(alpha, &*other.v_u);
    }

    /// `self = a*self + b*other` per component — port of
    /// `IteratesVector::AddOneVector` (when called on `self`).
    pub fn add_one_vector(
        &mut self,
        a: pounce_common::types::Number,
        other: &IteratesVector,
        b: pounce_common::types::Number,
    ) {
        self.x.add_one_vector(a, &*other.x, b);
        self.s.add_one_vector(a, &*other.s, b);
        self.y_c.add_one_vector(a, &*other.y_c, b);
        self.y_d.add_one_vector(a, &*other.y_d, b);
        self.z_l.add_one_vector(a, &*other.z_l, b);
        self.z_u.add_one_vector(a, &*other.z_u, b);
        self.v_l.add_one_vector(a, &*other.v_l, b);
        self.v_u.add_one_vector(a, &*other.v_u, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linalg::dense_vector::DenseVectorSpace;

    fn zero_vec(n: i32) -> Rc<dyn Vector> {
        let space = DenseVectorSpace::new(n);
        Rc::new(space.make_new_dense())
    }

    #[test]
    fn iterates_vector_dim_sums_components() {
        let iv = IteratesVector::new(
            zero_vec(4),
            zero_vec(1),
            zero_vec(1),
            zero_vec(1),
            zero_vec(4),
            zero_vec(4),
            zero_vec(1),
            zero_vec(1),
        );
        assert_eq!(iv.dim(), 4 + 1 + 1 + 1 + 4 + 4 + 1 + 1);
    }
}
