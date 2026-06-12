//! Crate-internal vector helpers shared by [`crate::convenience`] and
//! [`crate::solver`].

use pounce_common::types::Number;

/// Flatten a `pounce_linalg::Vector` trait object into a plain
/// `Vec<Number>`, handling the two concrete impls a converged iterate
/// can carry: `DenseVector` and (possibly nested) `CompoundVector`.
pub(crate) fn dense_to_vec(v: &dyn pounce_linalg::Vector) -> Vec<Number> {
    let any = v.as_any();
    // `expanded_values` materializes a homogeneous vector instead of tripping
    // `DenseVector::values`'s `!homogeneous` debug_assert (L16).
    if let Some(d) = any.downcast_ref::<pounce_linalg::dense_vector::DenseVector>() {
        return d.expanded_values();
    }
    // M9/F8: a `CompoundVector` (e.g. a partitioned primal `x`) is the other
    // concrete `Vector` impl the iterate can carry. Flatten its components in
    // order — recursively, so nested compounds work — instead of silently
    // fabricating a zero vector, which previously poisoned `SensResult.x` /
    // the KKT residual extraction with zeros.
    if let Some(c) = any.downcast_ref::<pounce_linalg::compound_vector::CompoundVector>() {
        let mut out = Vec::with_capacity(v.dim() as usize);
        for i in 0..c.n_comps() {
            out.extend(dense_to_vec(c.comp(i)));
        }
        return out;
    }
    // No generic element accessor exists on the `Vector` trait, so an
    // unrecognized concrete impl still falls back to zeros — but assert in
    // debug builds so a newly-added `Vector` type is caught by tests rather
    // than silently emitting a zero vector in release.
    debug_assert!(
        false,
        "dense_to_vec: unhandled Vector impl, returning zeros (dim {})",
        v.dim()
    );
    vec![0.0; v.dim() as usize]
}

#[cfg(test)]
mod tests {
    use super::dense_to_vec;
    use pounce_common::types::{Index, Number};
    use pounce_linalg::compound_vector::{CompoundVector, CompoundVectorSpace};
    use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
    use pounce_linalg::vector::Vector;
    use std::rc::Rc;

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
        v.as_any_mut()
            .downcast_mut::<DenseVector>()
            .expect("DenseVector")
            .set_values(vals);
    }

    // F8 (M9): a partitioned primal `x` reaches `dense_to_vec` as a
    // `CompoundVector`. The pre-fix code matched only `DenseVector` and the
    // `None` arm fabricated `vec![0.0; dim]`, silently poisoning `SensResult.x`
    // and the KKT residual with zeros. This asserts the real component values
    // are flattened in order instead.
    #[test]
    fn dense_to_vec_flattens_compound_vector_components() {
        let space = make_2block_space(2, 3);
        let mut v = CompoundVector::new(space);
        fill_dense(v.comp_mut(0), &[1.5, -2.0]);
        fill_dense(v.comp_mut(1), &[7.0, 0.25, -9.0]);

        let flat = dense_to_vec(&v as &dyn Vector);

        assert_eq!(flat, vec![1.5, -2.0, 7.0, 0.25, -9.0]);
        // Guard against the regression specifically: the old zero-fill would
        // have produced an all-zero vector of the same length.
        assert_ne!(flat, vec![0.0; 5]);
    }

    #[test]
    fn dense_to_vec_handles_plain_dense_vector() {
        let space = DenseVectorSpace::new(3);
        let mut d = DenseVector::new(space);
        d.set_values(&[3.0, 4.0, 5.0]);
        assert_eq!(dense_to_vec(&d as &dyn Vector), vec![3.0, 4.0, 5.0]);
    }
}
