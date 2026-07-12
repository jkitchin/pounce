//! Dense (contiguous) vector — port of `LinAlg/IpDenseVector.{hpp,cpp}`.
//!
//! Matches upstream's homogeneous-value optimization: when every entry
//! has the same value, only the scalar is stored, and the underlying
//! `Vec<Number>` is empty. Mutating any single element materializes
//! the storage (`set_values_from_scalar`) and clears the homogeneous
//! flag, exactly as in `DenseVector::Values()`.

use crate::blas1;
use crate::vector::{Vector, VectorCache};
use pounce_common::tagged::{Tag, TaggedObject};
use pounce_common::types::{Index, Number};
use std::any::Any;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

/// Vector space for `DenseVector`. Owns the dimension and any metadata
/// (string / integer / numeric maps keyed by tag string, mirroring
/// upstream `DenseVectorSpace::{string,integer,numeric}_meta_data_`).
#[derive(Debug, Default)]
pub struct DenseVectorSpace {
    dim: Index,
    string_meta: RefCell<BTreeMap<String, Vec<String>>>,
    integer_meta: RefCell<BTreeMap<String, Vec<Index>>>,
    numeric_meta: RefCell<BTreeMap<String, Vec<Number>>>,
}

impl DenseVectorSpace {
    pub fn new(dim: Index) -> Rc<Self> {
        Rc::new(Self {
            dim,
            string_meta: RefCell::new(BTreeMap::new()),
            integer_meta: RefCell::new(BTreeMap::new()),
            numeric_meta: RefCell::new(BTreeMap::new()),
        })
    }

    pub fn dim(&self) -> Index {
        self.dim
    }

    pub fn make_new_dense(self: &Rc<Self>) -> DenseVector {
        DenseVector::new(Rc::clone(self))
    }

    pub fn has_string_meta(&self, tag: &str) -> bool {
        self.string_meta.borrow().contains_key(tag)
    }
    pub fn set_string_meta(&self, tag: &str, data: Vec<String>) {
        self.string_meta.borrow_mut().insert(tag.to_string(), data);
    }
    pub fn get_string_meta(&self, tag: &str) -> Option<Vec<String>> {
        self.string_meta.borrow().get(tag).cloned()
    }

    pub fn has_integer_meta(&self, tag: &str) -> bool {
        self.integer_meta.borrow().contains_key(tag)
    }
    pub fn set_integer_meta(&self, tag: &str, data: Vec<Index>) {
        self.integer_meta.borrow_mut().insert(tag.to_string(), data);
    }
    pub fn get_integer_meta(&self, tag: &str) -> Option<Vec<Index>> {
        self.integer_meta.borrow().get(tag).cloned()
    }

    pub fn has_numeric_meta(&self, tag: &str) -> bool {
        self.numeric_meta.borrow().contains_key(tag)
    }
    pub fn set_numeric_meta(&self, tag: &str, data: Vec<Number>) {
        self.numeric_meta.borrow_mut().insert(tag.to_string(), data);
    }
    pub fn get_numeric_meta(&self, tag: &str) -> Option<Vec<Number>> {
        self.numeric_meta.borrow().get(tag).cloned()
    }
}

/// Dense vector — port of `IpDenseVector`.
#[derive(Debug)]
pub struct DenseVector {
    space: Rc<DenseVectorSpace>,
    cache: VectorCache,
    /// Storage. Empty until materialized; otherwise length == dim.
    values: Vec<Number>,
    initialized: bool,
    homogeneous: bool,
    scalar: Number,
}

impl DenseVector {
    pub fn new(space: Rc<DenseVectorSpace>) -> Self {
        let dim = space.dim();
        // Upstream: `if (Dim() == 0) { initialized_ = true; homogeneous_ = true; scalar_ = 0.; }`
        let (initialized, homogeneous, scalar) = if dim == 0 {
            (true, true, 0.0)
        } else {
            (false, false, 0.0)
        };
        Self {
            space,
            cache: VectorCache::new(),
            values: Vec::new(),
            initialized,
            homogeneous,
            scalar,
        }
    }

    pub fn space(&self) -> &Rc<DenseVectorSpace> {
        &self.space
    }

    pub fn is_homogeneous(&self) -> bool {
        self.homogeneous
    }

    pub fn scalar(&self) -> Number {
        debug_assert!(self.homogeneous);
        self.scalar
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Read-only slice into materialized values. Panics if currently
    /// homogeneous — mirrors upstream's DBG_ASSERT in
    /// `DenseVector::Values() const`. Use `expanded_values` to always
    /// get a slice.
    pub fn values(&self) -> &[Number] {
        debug_assert!(self.initialized && !self.homogeneous);
        &self.values
    }

    /// Mutable slice. Materializes a homogeneous vector first and
    /// bumps the change tag, matching upstream's non-const `Values()`.
    pub fn values_mut(&mut self) -> &mut [Number] {
        if self.initialized && self.homogeneous {
            self.materialize_from_scalar();
        }
        self.ensure_storage();
        self.cache.bump();
        self.initialized = true;
        self.homogeneous = false;
        &mut self.values
    }

    /// Always returns a fully-materialized slice. Allocates a copy if
    /// the vector is homogeneous (upstream caches this in
    /// `expanded_values_`; we just allocate on the fly).
    pub fn expanded_values(&self) -> Vec<Number> {
        if self.homogeneous {
            vec![self.scalar; self.space.dim() as usize]
        } else {
            self.values.clone()
        }
    }

    pub fn set_values(&mut self, x: &[Number]) {
        let dim = self.space.dim() as usize;
        assert_eq!(x.len(), dim);
        self.ensure_storage();
        self.values[..dim].copy_from_slice(x);
        self.initialized = true;
        self.homogeneous = false;
        self.cache.bump();
    }

    /// Equivalent to upstream `DenseVector::CopyToPos`.
    pub fn copy_to_pos(&mut self, pos: Index, x: &dyn Vector) {
        let pos = pos as usize;
        let dim_x = x.dim() as usize;
        assert!(pos + dim_x <= self.space.dim() as usize);
        let dense_x = downcast_dense(x);
        if self.homogeneous && self.initialized {
            self.materialize_from_scalar();
        }
        self.ensure_storage();
        self.homogeneous = false;
        if dense_x.homogeneous {
            for v in &mut self.values[pos..pos + dim_x] {
                *v = dense_x.scalar;
            }
        } else {
            self.values[pos..pos + dim_x].copy_from_slice(&dense_x.values[..dim_x]);
        }
        self.initialized = true;
        self.cache.bump();
    }

    /// Equivalent to upstream `DenseVector::CopyFromPos`.
    pub fn copy_from_pos(&mut self, pos: Index, x: &dyn Vector) {
        let pos = pos as usize;
        let dim = self.space.dim() as usize;
        assert!(pos + dim <= x.dim() as usize);
        let dense_x = downcast_dense(x);
        if dense_x.homogeneous {
            self.set(dense_x.scalar);
        } else {
            self.ensure_storage();
            self.values[..dim].copy_from_slice(&dense_x.values[pos..pos + dim]);
            self.initialized = true;
            self.homogeneous = false;
            self.cache.bump();
        }
    }

    pub fn ensure_storage(&mut self) {
        let dim = self.space.dim() as usize;
        if self.values.len() != dim {
            self.values.resize(dim, 0.0);
        }
    }

    /// Upstream `DenseVector::set_values_from_scalar`.
    fn materialize_from_scalar(&mut self) {
        debug_assert!(self.homogeneous);
        let dim = self.space.dim() as usize;
        self.values.clear();
        self.values.resize(dim, self.scalar);
        self.homogeneous = false;
        self.initialized = true;
    }
}

fn downcast_dense(x: &dyn Vector) -> &DenseVector {
    match x.as_any().downcast_ref::<DenseVector>() {
        Some(v) => v,
        None => panic!(
            "Vector argument is not a DenseVector — mixed-type linear algebra is not supported in v1.0"
        ),
    }
}

impl TaggedObject for DenseVector {
    fn get_tag(&self) -> Tag {
        self.cache.tag()
    }
}

impl Vector for DenseVector {
    fn dim(&self) -> Index {
        self.space.dim()
    }

    fn cache(&self) -> &VectorCache {
        &self.cache
    }

    fn make_new(&self) -> Box<dyn Vector> {
        Box::new(DenseVector::new(Rc::clone(&self.space)))
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
        let dx = downcast_dense(x);
        debug_assert!(dx.initialized);
        debug_assert_eq!(self.space.dim(), dx.space.dim());
        self.homogeneous = dx.homogeneous;
        if dx.homogeneous {
            self.scalar = dx.scalar;
            self.values.clear();
        } else {
            self.ensure_storage();
            let dim = self.space.dim() as usize;
            self.values[..dim].copy_from_slice(&dx.values[..dim]);
        }
        self.initialized = true;
    }

    fn scal_impl(&mut self, alpha: Number) {
        debug_assert!(self.initialized);
        if self.homogeneous {
            self.scalar *= alpha;
        } else {
            blas1::scal(alpha, &mut self.values, 1, self.space.dim());
        }
    }

    fn axpy_impl(&mut self, alpha: Number, x: &dyn Vector) {
        debug_assert!(self.initialized);
        let dx = downcast_dense(x);
        debug_assert!(dx.initialized);
        let dim = self.space.dim();
        if dim == 0 {
            return;
        }
        if self.homogeneous {
            if dx.homogeneous {
                self.scalar += alpha * dx.scalar;
            } else {
                let s0 = self.scalar;
                self.homogeneous = false;
                self.ensure_storage();
                let n = dim as usize;
                for i in 0..n {
                    self.values[i] = s0 + alpha * dx.values[i];
                }
            }
        } else if dx.homogeneous {
            if dx.scalar != 0.0 {
                let inc = alpha * dx.scalar;
                for v in &mut self.values[..dim as usize] {
                    *v += inc;
                }
            }
        } else {
            blas1::axpy(alpha, &dx.values, 1, &mut self.values, 1, dim);
        }
    }

    fn dot_impl(&self, x: &dyn Vector) -> Number {
        debug_assert!(self.initialized);
        let dx = downcast_dense(x);
        debug_assert!(dx.initialized);
        let dim = self.space.dim();
        let n = dim as usize;
        if dim == 0 {
            return 0.0;
        }
        match (self.homogeneous, dx.homogeneous) {
            (true, true) => (n as Number) * self.scalar * dx.scalar,
            (true, false) => {
                // Σ scalar * dx_i = scalar * Σ dx_i
                let mut s = 0.0;
                for v in &dx.values[..n] {
                    s += self.scalar * v;
                }
                s
            }
            (false, true) => {
                let mut s = 0.0;
                for v in &self.values[..n] {
                    s += dx.scalar * v;
                }
                s
            }
            (false, false) => blas1::dot(&self.values, 1, &dx.values, 1, dim),
        }
    }

    fn nrm2_impl(&self) -> Number {
        debug_assert!(self.initialized);
        if self.homogeneous {
            (self.space.dim() as Number).sqrt() * self.scalar.abs()
        } else {
            blas1::nrm2(&self.values, 1, self.space.dim())
        }
    }

    fn asum_impl(&self) -> Number {
        debug_assert!(self.initialized);
        if self.homogeneous {
            (self.space.dim() as Number) * self.scalar.abs()
        } else {
            blas1::asum(&self.values, 1, self.space.dim())
        }
    }

    fn amax_impl(&self) -> Number {
        debug_assert!(self.initialized);
        if self.space.dim() == 0 {
            return 0.0;
        }
        if self.homogeneous {
            return self.scalar.abs();
        }
        let i = blas1::iamax(&self.values, 1, self.space.dim()) as usize;
        self.values[i].abs()
    }

    fn set_impl(&mut self, value: Number) {
        self.initialized = true;
        self.homogeneous = true;
        self.scalar = value;
        // Free dense storage like upstream.
        self.values.clear();
        self.values.shrink_to_fit();
    }

    fn element_wise_divide_impl(&mut self, x: &dyn Vector) {
        debug_assert!(self.initialized);
        let dx = downcast_dense(x);
        debug_assert!(dx.initialized);
        let n = self.space.dim() as usize;
        if n == 0 {
            return;
        }
        match (self.homogeneous, dx.homogeneous) {
            (true, true) => self.scalar /= dx.scalar,
            (true, false) => {
                let s0 = self.scalar;
                self.homogeneous = false;
                self.ensure_storage();
                for i in 0..n {
                    self.values[i] = s0 / dx.values[i];
                }
            }
            (false, true) => {
                for v in &mut self.values[..n] {
                    *v /= dx.scalar;
                }
            }
            (false, false) => {
                for i in 0..n {
                    self.values[i] /= dx.values[i];
                }
            }
        }
    }

    fn element_wise_multiply_impl(&mut self, x: &dyn Vector) {
        debug_assert!(self.initialized);
        let dx = downcast_dense(x);
        debug_assert!(dx.initialized);
        let n = self.space.dim() as usize;
        if n == 0 {
            return;
        }
        match (self.homogeneous, dx.homogeneous) {
            (true, true) => self.scalar *= dx.scalar,
            (true, false) => {
                let s0 = self.scalar;
                self.homogeneous = false;
                self.ensure_storage();
                for i in 0..n {
                    self.values[i] = s0 * dx.values[i];
                }
            }
            (false, true) => {
                if dx.scalar != 1.0 {
                    for v in &mut self.values[..n] {
                        *v *= dx.scalar;
                    }
                }
            }
            (false, false) => {
                for i in 0..n {
                    self.values[i] *= dx.values[i];
                }
            }
        }
    }

    fn element_wise_select_impl(&mut self, x: &dyn Vector) {
        debug_assert!(self.initialized);
        let dx = downcast_dense(x);
        debug_assert!(dx.initialized);
        let n = self.space.dim() as usize;
        if n == 0 {
            return;
        }
        if self.homogeneous {
            if self.scalar == 0.0 {
                return;
            }
            if dx.homogeneous {
                self.scalar *= dx.scalar;
            } else {
                let s0 = self.scalar;
                self.homogeneous = false;
                self.ensure_storage();
                for i in 0..n {
                    self.values[i] = s0 * dx.values[i];
                }
            }
        } else if dx.homogeneous {
            if dx.scalar != 1.0 {
                for v in &mut self.values[..n] {
                    if *v > 0.0 {
                        *v = dx.scalar;
                    } else if *v < 0.0 {
                        *v = -dx.scalar;
                    }
                }
            }
        } else {
            for i in 0..n {
                if self.values[i] > 0.0 {
                    self.values[i] = dx.values[i];
                } else if self.values[i] < 0.0 {
                    self.values[i] = -dx.values[i];
                }
            }
        }
    }

    fn element_wise_max_impl(&mut self, x: &dyn Vector) {
        debug_assert!(self.initialized);
        let dx = downcast_dense(x);
        debug_assert!(dx.initialized);
        let n = self.space.dim() as usize;
        if n == 0 {
            return;
        }
        match (self.homogeneous, dx.homogeneous) {
            (true, true) => self.scalar = self.scalar.max(dx.scalar),
            (true, false) => {
                let s0 = self.scalar;
                self.homogeneous = false;
                self.ensure_storage();
                for i in 0..n {
                    self.values[i] = s0.max(dx.values[i]);
                }
            }
            (false, true) => {
                for v in &mut self.values[..n] {
                    *v = (*v).max(dx.scalar);
                }
            }
            (false, false) => {
                for i in 0..n {
                    self.values[i] = self.values[i].max(dx.values[i]);
                }
            }
        }
    }

    fn element_wise_min_impl(&mut self, x: &dyn Vector) {
        debug_assert!(self.initialized);
        let dx = downcast_dense(x);
        debug_assert!(dx.initialized);
        let n = self.space.dim() as usize;
        if n == 0 {
            return;
        }
        match (self.homogeneous, dx.homogeneous) {
            (true, true) => self.scalar = self.scalar.min(dx.scalar),
            (true, false) => {
                let s0 = self.scalar;
                self.homogeneous = false;
                self.ensure_storage();
                for i in 0..n {
                    self.values[i] = s0.min(dx.values[i]);
                }
            }
            (false, true) => {
                for v in &mut self.values[..n] {
                    *v = (*v).min(dx.scalar);
                }
            }
            (false, false) => {
                for i in 0..n {
                    self.values[i] = self.values[i].min(dx.values[i]);
                }
            }
        }
    }

    fn element_wise_reciprocal_impl(&mut self) {
        debug_assert!(self.initialized);
        let n = self.space.dim() as usize;
        if n == 0 {
            return;
        }
        if self.homogeneous {
            self.scalar = 1.0 / self.scalar;
        } else {
            for v in &mut self.values[..n] {
                *v = 1.0 / *v;
            }
        }
    }

    fn element_wise_abs_impl(&mut self) {
        debug_assert!(self.initialized);
        if self.homogeneous {
            self.scalar = self.scalar.abs();
        } else {
            for v in &mut self.values[..self.space.dim() as usize] {
                *v = v.abs();
            }
        }
    }

    fn element_wise_sqrt_impl(&mut self) {
        debug_assert!(self.initialized);
        if self.homogeneous {
            self.scalar = self.scalar.sqrt();
        } else {
            for v in &mut self.values[..self.space.dim() as usize] {
                *v = v.sqrt();
            }
        }
    }

    fn element_wise_sgn_impl(&mut self) {
        debug_assert!(self.initialized);
        let sgn = |v: Number| -> Number {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        };
        if self.homogeneous {
            self.scalar = sgn(self.scalar);
        } else {
            for v in &mut self.values[..self.space.dim() as usize] {
                *v = sgn(*v);
            }
        }
    }

    fn add_scalar_impl(&mut self, scalar: Number) {
        debug_assert!(self.initialized);
        if self.homogeneous {
            self.scalar += scalar;
        } else {
            for v in &mut self.values[..self.space.dim() as usize] {
                *v += scalar;
            }
        }
    }

    fn max_impl(&self) -> Number {
        debug_assert!(self.initialized);
        let n = self.space.dim() as usize;
        if n == 0 {
            return -Number::MAX;
        }
        if self.homogeneous {
            return self.scalar;
        }
        let mut m = self.values[0];
        for &v in &self.values[1..n] {
            if v > m {
                m = v;
            }
        }
        m
    }

    fn min_impl(&self) -> Number {
        debug_assert!(self.initialized);
        let n = self.space.dim() as usize;
        if n == 0 {
            return Number::MAX;
        }
        if self.homogeneous {
            return self.scalar;
        }
        let mut m = self.values[0];
        for &v in &self.values[1..n] {
            if v < m {
                m = v;
            }
        }
        m
    }

    fn sum_impl(&self) -> Number {
        debug_assert!(self.initialized);
        let n = self.space.dim() as usize;
        if self.homogeneous {
            (n as Number) * self.scalar
        } else {
            let mut s = 0.0;
            for &v in &self.values[..n] {
                s += v;
            }
            s
        }
    }

    fn sum_logs_impl(&self) -> Number {
        debug_assert!(self.initialized);
        let n = self.space.dim() as usize;
        if n == 0 {
            return 0.0;
        }
        if self.homogeneous {
            (n as Number) * self.scalar.ln()
        } else {
            let mut s = 0.0;
            for &v in &self.values[..n] {
                s += v.ln();
            }
            s
        }
    }

    fn frac_to_bound_impl(&self, delta: &dyn Vector, tau: Number) -> Number {
        debug_assert_eq!(self.space.dim(), delta.dim());
        debug_assert!(tau >= 0.0);
        let dd = downcast_dense(delta);
        let n = self.space.dim() as usize;
        if n == 0 {
            return 1.0;
        }
        let mut alpha: Number = 1.0;
        match (self.homogeneous, dd.homogeneous) {
            (true, true) => {
                if dd.scalar < 0.0 {
                    alpha = alpha.min(-tau / dd.scalar * self.scalar);
                }
            }
            (true, false) => {
                for &d in &dd.values[..n] {
                    if d < 0.0 {
                        alpha = alpha.min(-tau / d * self.scalar);
                    }
                }
            }
            (false, true) => {
                if dd.scalar < 0.0 {
                    let f = -tau / dd.scalar;
                    for &x in &self.values[..n] {
                        alpha = alpha.min(f * x);
                    }
                }
            }
            (false, false) => {
                for i in 0..n {
                    let d = dd.values[i];
                    if d < 0.0 {
                        alpha = alpha.min(-tau / d * self.values[i]);
                    }
                }
            }
        }
        debug_assert!(alpha >= 0.0);
        alpha
    }

    fn add_two_vectors_impl(
        &mut self,
        a: Number,
        v1: &dyn Vector,
        b: Number,
        v2: &dyn Vector,
        c: Number,
    ) {
        let n = self.space.dim() as usize;
        if n == 0 {
            debug_assert!(self.initialized);
            return;
        }
        let dv1 = if a != 0.0 {
            Some(downcast_dense(v1))
        } else {
            None
        };
        let dv2 = if b != 0.0 {
            Some(downcast_dense(v2))
        } else {
            None
        };
        let homog_v1 = dv1.map(|d| d.homogeneous).unwrap_or(true);
        let homog_v2 = dv2.map(|d| d.homogeneous).unwrap_or(true);
        let s_v1 = dv1.map(|d| d.scalar).unwrap_or(0.0);
        let s_v2 = dv2.map(|d| d.scalar).unwrap_or(0.0);

        // All-homogeneous fast path — result stays homogeneous.
        if (c == 0.0 || self.homogeneous) && homog_v1 && homog_v2 {
            let prev = if c == 0.0 { 0.0 } else { c * self.scalar };
            self.scalar = prev + a * s_v1 + b * s_v2;
            self.homogeneous = true;
            self.initialized = true;
            return;
        }

        // Materialize self if needed. With c == 0 we are about to overwrite.
        if c == 0.0 {
            self.ensure_storage();
            self.homogeneous = false;
        } else if self.homogeneous {
            self.materialize_from_scalar();
        }

        // Get expanded slices of v1/v2 (allocate when homogeneous; the
        // slow path is rare in practice).
        let v1_arr: Option<Vec<Number>> = dv1.map(|d| {
            if d.homogeneous {
                vec![d.scalar; n]
            } else {
                d.values[..n].to_vec()
            }
        });
        let v2_arr: Option<Vec<Number>> = dv2.map(|d| {
            if d.homogeneous {
                vec![d.scalar; n]
            } else {
                d.values[..n].to_vec()
            }
        });

        // Single fused expression. IEEE multiplication by 0 / 1 / -1
        // is exact, so this is bit-equivalent to upstream's 64-case
        // dispatch in `IpDenseVector.cpp:843-1322`.
        if c == 0.0 {
            for i in 0..n {
                let v1i = v1_arr.as_ref().map(|v| v[i]).unwrap_or(0.0);
                let v2i = v2_arr.as_ref().map(|v| v[i]).unwrap_or(0.0);
                self.values[i] = a * v1i + b * v2i;
            }
        } else {
            for i in 0..n {
                let v1i = v1_arr.as_ref().map(|v| v[i]).unwrap_or(0.0);
                let v2i = v2_arr.as_ref().map(|v| v[i]).unwrap_or(0.0);
                self.values[i] = a * v1i + b * v2i + c * self.values[i];
            }
        }
        self.initialized = true;
    }

    fn add_vector_quotient_impl(&mut self, a: Number, z: &dyn Vector, s: &dyn Vector, c: Number) {
        debug_assert_eq!(self.space.dim(), z.dim());
        debug_assert_eq!(self.space.dim(), s.dim());
        let dz = downcast_dense(z);
        let ds = downcast_dense(s);
        debug_assert!(dz.initialized && ds.initialized);
        let n = self.space.dim() as usize;
        if n == 0 {
            return;
        }
        let homog_z = dz.homogeneous;
        let homog_s = ds.homogeneous;
        if (c == 0.0 || self.homogeneous) && homog_z && homog_s {
            self.scalar = if c == 0.0 {
                a * dz.scalar / ds.scalar
            } else {
                c * self.scalar + a * dz.scalar / ds.scalar
            };
            self.initialized = true;
            self.homogeneous = true;
            self.values.clear();
            return;
        }
        // Materialize self if needed.
        if c == 0.0 {
            self.ensure_storage();
            self.homogeneous = false;
        } else if self.homogeneous {
            self.materialize_from_scalar();
        }
        let z_arr: Vec<Number> = if homog_z {
            vec![dz.scalar; n]
        } else {
            dz.values[..n].to_vec()
        };
        let s_arr: Vec<Number> = if homog_s {
            vec![ds.scalar; n]
        } else {
            ds.values[..n].to_vec()
        };
        if c == 0.0 {
            for i in 0..n {
                self.values[i] = a * z_arr[i] / s_arr[i];
            }
        } else {
            for i in 0..n {
                self.values[i] = c * self.values[i] + a * z_arr[i] / s_arr[i];
            }
        }
        self.initialized = true;
        self.homogeneous = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(space: &Rc<DenseVectorSpace>, vals: &[Number]) -> DenseVector {
        let mut v = DenseVector::new(Rc::clone(space));
        v.set_values(vals);
        v
    }

    #[test]
    fn axpy_basic() {
        let s = DenseVectorSpace::new(3);
        let x = vec_of(&s, &[1.0, 2.0, 3.0]);
        let mut y = vec_of(&s, &[10.0, 20.0, 30.0]);
        y.axpy(2.0, &x);
        assert_eq!(y.values(), &[12.0, 24.0, 36.0]);
    }

    #[test]
    fn dot_homogeneous_pair() {
        let s = DenseVectorSpace::new(4);
        let mut x = DenseVector::new(Rc::clone(&s));
        x.set(2.0); // homogeneous 2
        let mut y = DenseVector::new(Rc::clone(&s));
        y.set(3.0); // homogeneous 3
        // 4 entries of 2*3 = 24
        assert_eq!(x.dot(&y), 24.0);
    }

    #[test]
    fn dot_mixed_homog_dense() {
        let s = DenseVectorSpace::new(3);
        let mut x = DenseVector::new(Rc::clone(&s));
        x.set(2.0);
        let y = vec_of(&s, &[1.0, 2.0, 3.0]);
        // 2*(1+2+3) = 12
        assert_eq!(x.dot(&y), 12.0);
        assert_eq!(y.dot(&x), 12.0);
    }

    #[test]
    fn nrm2_homogeneous_uses_sqrt_n() {
        let s = DenseVectorSpace::new(4);
        let mut x = DenseVector::new(Rc::clone(&s));
        x.set(3.0);
        // sqrt(4) * 3 = 6
        assert!((x.nrm2() - 6.0).abs() < 1e-15);
    }

    #[test]
    fn nrm2_cache_invalidated_by_mutation() {
        let s = DenseVectorSpace::new(2);
        let mut x = vec_of(&s, &[3.0, 4.0]);
        assert_eq!(x.nrm2(), 5.0);
        x.scal(2.0);
        assert!((x.nrm2() - 10.0).abs() < 1e-15);
    }

    #[test]
    fn dot_cache_hits_after_first_call() {
        let s = DenseVectorSpace::new(3);
        let x = vec_of(&s, &[1.0, 2.0, 3.0]);
        let y = vec_of(&s, &[1.0, 1.0, 1.0]);
        assert_eq!(x.dot(&y), 6.0);
        // Second call should be cached but still produce the same value.
        assert_eq!(x.dot(&y), 6.0);
    }

    #[test]
    fn dot_self_uses_nrm2_squared_path() {
        let s = DenseVectorSpace::new(2);
        let x = vec_of(&s, &[3.0, 4.0]);
        // Pass x as both args — cache shortcut should compute 5*5 = 25.
        assert_eq!(x.dot(&x), 25.0);
    }

    #[test]
    fn add_two_vectors_all_homogeneous() {
        let s = DenseVectorSpace::new(5);
        let mut y = DenseVector::new(Rc::clone(&s));
        y.set(1.0);
        let mut v1 = DenseVector::new(Rc::clone(&s));
        v1.set(2.0);
        let mut v2 = DenseVector::new(Rc::clone(&s));
        v2.set(3.0);
        // y = 4*v1 + 5*v2 + 0.5*y = 4*2 + 5*3 + 0.5 = 8 + 15 + 0.5 = 23.5
        y.add_two_vectors(4.0, &v1, 5.0, &v2, 0.5);
        assert!(y.is_homogeneous());
        assert_eq!(y.scalar(), 23.5);
    }

    #[test]
    fn add_two_vectors_mixed_dense_overrides_homog() {
        let s = DenseVectorSpace::new(3);
        let mut y = DenseVector::new(Rc::clone(&s));
        y.set(0.0);
        let v1 = vec_of(&s, &[1.0, 2.0, 3.0]);
        let v2 = vec_of(&s, &[10.0, 10.0, 10.0]);
        // y = 1*v1 + 1*v2 + 0*y = [11, 12, 13]
        y.add_two_vectors(1.0, &v1, 1.0, &v2, 0.0);
        assert!(!y.is_homogeneous());
        assert_eq!(y.values(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn frac_to_bound_basic() {
        let s = DenseVectorSpace::new(3);
        let x = vec_of(&s, &[1.0, 2.0, 3.0]);
        let delta = vec_of(&s, &[-2.0, 1.0, -1.5]);
        // negative components: i=0 → -tau/-2 * 1 = tau/2
        //                       i=2 → -tau/-1.5 * 3 = tau*2
        // alpha = min(1, tau/2, tau*2). tau=1 → alpha = 0.5
        let alpha = x.frac_to_bound(&delta, 1.0);
        assert!((alpha - 0.5).abs() < 1e-15);
    }

    #[test]
    fn element_wise_divide_homog_dense() {
        let s = DenseVectorSpace::new(3);
        let mut y = DenseVector::new(Rc::clone(&s));
        y.set(6.0);
        let x = vec_of(&s, &[1.0, 2.0, 3.0]);
        y.element_wise_divide(&x);
        assert!(!y.is_homogeneous());
        assert_eq!(y.values(), &[6.0, 3.0, 2.0]);
    }

    #[test]
    fn element_wise_sgn_handles_all_three_signs() {
        let s = DenseVectorSpace::new(3);
        let mut x = vec_of(&s, &[-2.5, 0.0, 7.0]);
        x.element_wise_sgn();
        assert_eq!(x.values(), &[-1.0, 0.0, 1.0]);
    }

    #[test]
    fn sum_and_max_min_homogeneous() {
        let s = DenseVectorSpace::new(4);
        let mut x = DenseVector::new(Rc::clone(&s));
        x.set(2.5);
        assert_eq!(x.sum(), 10.0);
        assert_eq!(x.max(), 2.5);
        assert_eq!(x.min(), 2.5);
    }

    #[test]
    fn has_valid_numbers_detects_nan() {
        let s = DenseVectorSpace::new(3);
        let bad = vec_of(&s, &[1.0, Number::NAN, 2.0]);
        assert!(!bad.has_valid_numbers());
        let good = vec_of(&s, &[1.0, 2.0, 3.0]);
        assert!(good.has_valid_numbers());
    }

    #[test]
    fn copy_to_pos_pastes_into_subrange() {
        let s_big = DenseVectorSpace::new(5);
        let s_small = DenseVectorSpace::new(2);
        let mut y = DenseVector::new(Rc::clone(&s_big));
        y.set(0.0);
        let x = vec_of(&s_small, &[7.0, 8.0]);
        y.copy_to_pos(2, &x);
        assert_eq!(y.values(), &[0.0, 0.0, 7.0, 8.0, 0.0]);
    }

    #[test]
    fn make_new_copy_clones_values() {
        let s = DenseVectorSpace::new(3);
        let x = vec_of(&s, &[1.0, 2.0, 3.0]);
        let y = x.make_new_copy();
        let dy = y.as_any().downcast_ref::<DenseVector>().unwrap();
        assert_eq!(dy.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_vector_quotient_all_homogeneous() {
        let s = DenseVectorSpace::new(4);
        let mut y = DenseVector::new(Rc::clone(&s));
        y.set(1.0);
        let mut z = DenseVector::new(Rc::clone(&s));
        z.set(6.0);
        let mut sd = DenseVector::new(Rc::clone(&s));
        sd.set(2.0);
        // y = 2 * z/sd + 0.5 * y = 2*6/2 + 0.5 = 6.5
        y.add_vector_quotient(2.0, &z, &sd, 0.5);
        assert!(y.is_homogeneous());
        assert_eq!(y.scalar(), 6.5);
    }

    #[test]
    fn dim_zero_is_consistent() {
        let s = DenseVectorSpace::new(0);
        let x = DenseVector::new(Rc::clone(&s));
        assert!(x.is_initialized());
        assert!(x.is_homogeneous());
        assert_eq!(x.nrm2(), 0.0);
    }
}
