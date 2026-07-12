//! Block vector — port of `LinAlg/IpCompoundVector.{hpp,cpp}`.
//!
//! Stacks zero-or-more component vectors into a single virtual vector.
//! Every operation dispatches block-by-block in the order the
//! components were registered, matching upstream's iteration order
//! exactly so that summed reductions (`nrm2`, `dot`, `sum`) preserve
//! bit-equivalence under the same component layout.
//!
//! Component construction uses a `Vec` of factory closures rather
//! than a `VectorSpace` trait — see the docstring on
//! [`CompoundVectorSpace::set_comp`] for usage.

use crate::vector::{Vector, VectorCache};
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

type CompFactory = Box<dyn Fn() -> Box<dyn Vector>>;

/// Vector space describing the block layout. Constructed by
/// [`CompoundVectorSpace::new`], populated with [`set_comp`], then
/// passed to [`CompoundVector::new`] to create new compound vectors.
pub struct CompoundVectorSpace {
    total_dim: Index,
    n_comp_spaces: Index,
    /// Component dimensions; `Index::MIN` until set.
    comp_dims: RefCell<Vec<Index>>,
    factories: RefCell<Vec<Option<CompFactory>>>,
}

impl std::fmt::Debug for CompoundVectorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompoundVectorSpace")
            .field("total_dim", &self.total_dim)
            .field("n_comp_spaces", &self.n_comp_spaces)
            .field("comp_dims", &self.comp_dims.borrow())
            .finish()
    }
}

impl CompoundVectorSpace {
    pub fn new(n_comp_spaces: Index, total_dim: Index) -> Rc<Self> {
        let mut factories: Vec<Option<CompFactory>> = Vec::with_capacity(n_comp_spaces as usize);
        for _ in 0..n_comp_spaces {
            factories.push(None);
        }
        Rc::new(Self {
            total_dim,
            n_comp_spaces,
            comp_dims: RefCell::new(vec![0; n_comp_spaces as usize]),
            factories: RefCell::new(factories),
        })
    }

    pub fn dim(&self) -> Index {
        self.total_dim
    }

    pub fn n_comp_spaces(&self) -> Index {
        self.n_comp_spaces
    }

    pub fn comp_dim(&self, icomp: Index) -> Index {
        self.comp_dims.borrow()[icomp as usize]
    }

    /// Register the factory that builds a fresh component at slot
    /// `icomp`. The factory closure must capture (typically by `Rc`)
    /// any subspace it needs. Mirrors upstream
    /// `CompoundVectorSpace::SetCompSpace`.
    pub fn set_comp<F>(&self, icomp: Index, dim: Index, factory: F)
    where
        F: Fn() -> Box<dyn Vector> + 'static,
    {
        assert!(icomp < self.n_comp_spaces);
        self.comp_dims.borrow_mut()[icomp as usize] = dim;
        self.factories.borrow_mut()[icomp as usize] = Some(Box::new(factory));
    }
}

/// Compound (block) vector. Owns its components.
#[derive(Debug)]
pub struct CompoundVector {
    space: Rc<CompoundVectorSpace>,
    cache: VectorCache,
    comps: Vec<Box<dyn Vector>>,
}

impl CompoundVector {
    /// Construct, calling each registered factory once. Equivalent to
    /// upstream `CompoundVector(owner_space, /*create_new=*/true)`.
    pub fn new(space: Rc<CompoundVectorSpace>) -> Self {
        let n = space.n_comp_spaces() as usize;
        let mut comps: Vec<Box<dyn Vector>> = Vec::with_capacity(n);
        let factories = space.factories.borrow();
        let mut dim_check: Index = 0;
        for f in factories.iter() {
            let factory = match f.as_ref() {
                Some(fac) => fac,
                None => panic!(
                    "CompoundVectorSpace component not set — call set_comp on every component before constructing a CompoundVector"
                ),
            };
            let v = factory();
            dim_check += v.dim();
            comps.push(v);
        }
        debug_assert_eq!(dim_check, space.total_dim);
        drop(factories);
        Self {
            space,
            cache: VectorCache::new(),
            comps,
        }
    }

    pub fn n_comps(&self) -> Index {
        self.comps.len() as Index
    }

    pub fn comp(&self, i: Index) -> &dyn Vector {
        self.comps[i as usize].as_ref()
    }

    /// Mutable access to a component. Marks the compound as changed,
    /// matching upstream's `GetCompNonConst` which calls
    /// `ObjectChanged()` because the caller is about to mutate.
    pub fn comp_mut(&mut self, i: Index) -> &mut dyn Vector {
        self.cache.bump();
        self.comps[i as usize].as_mut()
    }

    pub fn space(&self) -> &Rc<CompoundVectorSpace> {
        &self.space
    }
}

fn downcast_compound(x: &dyn Vector) -> &CompoundVector {
    match x.as_any().downcast_ref::<CompoundVector>() {
        Some(v) => v,
        None => panic!("Vector argument is not a CompoundVector"),
    }
}

impl TaggedObject for CompoundVector {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Vector for CompoundVector {
    fn dim(&self) -> Index {
        self.space.total_dim
    }

    fn cache(&self) -> &VectorCache {
        &self.cache
    }

    fn make_new(&self) -> Box<dyn Vector> {
        Box::new(CompoundVector::new(Rc::clone(&self.space)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_tagged(&self) -> &dyn TaggedObject {
        self
    }

    fn as_dyn_vector(&self) -> &dyn Vector {
        self
    }

    fn copy_impl(&mut self, x: &dyn Vector) {
        let cx = downcast_compound(x);
        debug_assert_eq!(self.n_comps(), cx.n_comps());
        for i in 0..self.comps.len() {
            self.comps[i].copy(cx.comps[i].as_ref());
        }
    }

    fn scal_impl(&mut self, alpha: Number) {
        for c in &mut self.comps {
            c.scal(alpha);
        }
    }

    fn axpy_impl(&mut self, alpha: Number, x: &dyn Vector) {
        let cx = downcast_compound(x);
        debug_assert_eq!(self.n_comps(), cx.n_comps());
        for i in 0..self.comps.len() {
            self.comps[i].axpy(alpha, cx.comps[i].as_ref());
        }
    }

    fn dot_impl(&self, x: &dyn Vector) -> Number {
        let cx = downcast_compound(x);
        debug_assert_eq!(self.n_comps(), cx.n_comps());
        let mut s = 0.0;
        for i in 0..self.comps.len() {
            s += self.comps[i].dot(cx.comps[i].as_ref());
        }
        s
    }

    fn nrm2_impl(&self) -> Number {
        let mut sum_sq = 0.0;
        for c in &self.comps {
            let n = c.nrm2();
            sum_sq += n * n;
        }
        sum_sq.sqrt()
    }

    fn asum_impl(&self) -> Number {
        let mut s = 0.0;
        for c in &self.comps {
            s += c.asum();
        }
        s
    }

    fn amax_impl(&self) -> Number {
        let mut m: Number = 0.0;
        for c in &self.comps {
            let v = c.amax();
            if v > m {
                m = v;
            }
        }
        m
    }

    fn set_impl(&mut self, value: Number) {
        for c in &mut self.comps {
            c.set(value);
        }
    }

    fn element_wise_divide_impl(&mut self, x: &dyn Vector) {
        let cx = downcast_compound(x);
        for i in 0..self.comps.len() {
            self.comps[i].element_wise_divide(cx.comps[i].as_ref());
        }
    }
    fn element_wise_multiply_impl(&mut self, x: &dyn Vector) {
        let cx = downcast_compound(x);
        for i in 0..self.comps.len() {
            self.comps[i].element_wise_multiply(cx.comps[i].as_ref());
        }
    }
    fn element_wise_select_impl(&mut self, x: &dyn Vector) {
        let cx = downcast_compound(x);
        for i in 0..self.comps.len() {
            self.comps[i].element_wise_select(cx.comps[i].as_ref());
        }
    }
    fn element_wise_max_impl(&mut self, x: &dyn Vector) {
        let cx = downcast_compound(x);
        for i in 0..self.comps.len() {
            self.comps[i].element_wise_max(cx.comps[i].as_ref());
        }
    }
    fn element_wise_min_impl(&mut self, x: &dyn Vector) {
        let cx = downcast_compound(x);
        for i in 0..self.comps.len() {
            self.comps[i].element_wise_min(cx.comps[i].as_ref());
        }
    }
    fn element_wise_reciprocal_impl(&mut self) {
        for c in &mut self.comps {
            c.element_wise_reciprocal();
        }
    }
    fn element_wise_abs_impl(&mut self) {
        for c in &mut self.comps {
            c.element_wise_abs();
        }
    }
    fn element_wise_sqrt_impl(&mut self) {
        for c in &mut self.comps {
            c.element_wise_sqrt();
        }
    }
    fn element_wise_sgn_impl(&mut self) {
        for c in &mut self.comps {
            c.element_wise_sgn();
        }
    }
    fn add_scalar_impl(&mut self, scalar: Number) {
        for c in &mut self.comps {
            c.add_scalar(scalar);
        }
    }

    fn max_impl(&self) -> Number {
        debug_assert!(!self.comps.is_empty() && self.dim() > 0);
        let mut m = -Number::MAX;
        for c in &self.comps {
            if c.dim() != 0 {
                let v = c.max();
                if v > m {
                    m = v;
                }
            }
        }
        m
    }

    fn min_impl(&self) -> Number {
        debug_assert!(!self.comps.is_empty() && self.dim() > 0);
        let mut m = Number::MAX;
        for c in &self.comps {
            if c.dim() != 0 {
                let v = c.min();
                if v < m {
                    m = v;
                }
            }
        }
        m
    }

    fn sum_impl(&self) -> Number {
        let mut s = 0.0;
        for c in &self.comps {
            s += c.sum();
        }
        s
    }

    fn sum_logs_impl(&self) -> Number {
        let mut s = 0.0;
        for c in &self.comps {
            s += c.sum_logs();
        }
        s
    }

    fn add_two_vectors_impl(
        &mut self,
        a: Number,
        v1: &dyn Vector,
        b: Number,
        v2: &dyn Vector,
        c: Number,
    ) {
        let cv1 = downcast_compound(v1);
        let cv2 = downcast_compound(v2);
        debug_assert_eq!(self.n_comps(), cv1.n_comps());
        debug_assert_eq!(self.n_comps(), cv2.n_comps());
        for i in 0..self.comps.len() {
            self.comps[i].add_two_vectors(a, cv1.comps[i].as_ref(), b, cv2.comps[i].as_ref(), c);
        }
    }

    fn frac_to_bound_impl(&self, delta: &dyn Vector, tau: Number) -> Number {
        let cd = downcast_compound(delta);
        debug_assert_eq!(self.n_comps(), cd.n_comps());
        let mut alpha: Number = 1.0;
        for i in 0..self.comps.len() {
            let a = self.comps[i].frac_to_bound(cd.comps[i].as_ref(), tau);
            if a < alpha {
                alpha = a;
            }
        }
        alpha
    }

    fn add_vector_quotient_impl(&mut self, a: Number, z: &dyn Vector, s: &dyn Vector, c: Number) {
        let cz = downcast_compound(z);
        let cs = downcast_compound(s);
        for i in 0..self.comps.len() {
            self.comps[i].add_vector_quotient(a, cz.comps[i].as_ref(), cs.comps[i].as_ref(), c);
        }
    }

    fn has_valid_numbers_impl(&self) -> bool {
        for c in &self.comps {
            if !c.has_valid_numbers() {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_vector::{DenseVector, DenseVectorSpace};

    fn make_2block_space(d1: Index, d2: Index) -> Rc<CompoundVectorSpace> {
        let space = CompoundVectorSpace::new(2, d1 + d2);
        let s1 = DenseVectorSpace::new(d1);
        let s2 = DenseVectorSpace::new(d2);
        space.set_comp(0, d1, {
            let s = Rc::clone(&s1);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
        space.set_comp(1, d2, {
            let s = Rc::clone(&s2);
            move || Box::new(DenseVector::new(Rc::clone(&s)))
        });
        space
    }

    fn fill_dense(v: &mut dyn Vector, vals: &[Number]) {
        let dv = v
            .as_any_mut()
            .downcast_mut::<DenseVector>()
            .expect("DenseVector");
        dv.set_values(vals);
    }

    #[test]
    fn nrm2_combines_blocks() {
        let space = make_2block_space(2, 3);
        let mut v = CompoundVector::new(space);
        fill_dense(v.comp_mut(0), &[3.0, 4.0]); // nrm2 = 5
        fill_dense(v.comp_mut(1), &[0.0, 0.0, 12.0]); // nrm2 = 12
        // sqrt(25 + 144) = 13
        assert!((v.nrm2() - 13.0).abs() < 1e-15);
    }

    #[test]
    fn dot_routes_to_blocks() {
        let space = make_2block_space(2, 2);
        let mut x = CompoundVector::new(Rc::clone(&space));
        fill_dense(x.comp_mut(0), &[1.0, 2.0]);
        fill_dense(x.comp_mut(1), &[3.0, 4.0]);
        let mut y = CompoundVector::new(Rc::clone(&space));
        fill_dense(y.comp_mut(0), &[10.0, 20.0]);
        fill_dense(y.comp_mut(1), &[100.0, 1000.0]);
        // 1*10 + 2*20 + 3*100 + 4*1000 = 10 + 40 + 300 + 4000 = 4350
        assert_eq!(x.dot(&y), 4350.0);
    }

    #[test]
    fn axpy_propagates_to_blocks() {
        let space = make_2block_space(2, 1);
        let mut x = CompoundVector::new(Rc::clone(&space));
        fill_dense(x.comp_mut(0), &[1.0, 1.0]);
        fill_dense(x.comp_mut(1), &[1.0]);
        let mut y = CompoundVector::new(Rc::clone(&space));
        fill_dense(y.comp_mut(0), &[10.0, 20.0]);
        fill_dense(y.comp_mut(1), &[30.0]);
        y.axpy(2.0, &x);
        let dy0 = y.comp(0).as_any().downcast_ref::<DenseVector>().unwrap();
        let dy1 = y.comp(1).as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dy0.values(), &[12.0, 22.0]);
        assert_eq!(dy1.values(), &[32.0]);
    }

    #[test]
    fn asum_sums_block_asums() {
        let space = make_2block_space(2, 2);
        let mut x = CompoundVector::new(space);
        fill_dense(x.comp_mut(0), &[-1.0, 2.0]); // asum = 3
        fill_dense(x.comp_mut(1), &[3.0, -4.0]); // asum = 7
        assert_eq!(x.asum(), 10.0);
    }

    #[test]
    fn amax_takes_max_across_blocks() {
        let space = make_2block_space(2, 3);
        let mut x = CompoundVector::new(space);
        fill_dense(x.comp_mut(0), &[1.0, -2.0]);
        fill_dense(x.comp_mut(1), &[0.5, -10.0, 3.0]);
        assert_eq!(x.amax(), 10.0);
    }

    #[test]
    fn frac_to_bound_takes_min_across_blocks() {
        let space = make_2block_space(2, 1);
        let mut x = CompoundVector::new(Rc::clone(&space));
        fill_dense(x.comp_mut(0), &[1.0, 2.0]);
        fill_dense(x.comp_mut(1), &[3.0]);
        let mut delta = CompoundVector::new(space);
        fill_dense(delta.comp_mut(0), &[-2.0, 0.0]); // alpha = tau/2 * 1 = 0.5
        fill_dense(delta.comp_mut(1), &[-1.5]); // alpha = tau/1.5 * 3 = 2*tau
        // Min(0.5, 2*tau) for tau=1 → 0.5
        let alpha = x.frac_to_bound(&delta, 1.0);
        assert!((alpha - 0.5).abs() < 1e-15);
    }

    #[test]
    fn make_new_creates_uninitialized_compound() {
        let space = make_2block_space(2, 1);
        let mut x = CompoundVector::new(Rc::clone(&space));
        fill_dense(x.comp_mut(0), &[1.0, 2.0]);
        fill_dense(x.comp_mut(1), &[3.0]);
        let y = x.make_new();
        let cy = y.as_any().downcast_ref::<CompoundVector>().unwrap();
        assert_eq!(cy.n_comps(), 2);
        assert_eq!(cy.comp(0).dim(), 2);
        assert_eq!(cy.comp(1).dim(), 1);
    }
}
