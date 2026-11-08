//! Shared driver for the `PounceQp*` tests (gh#1147 S3-I3): a handle that
//! frees itself, and one wrapper per export over Rust slices.

#![allow(dead_code)]

use pounce_cinterface::*;
use std::ffi::CString;
use std::ptr::{null, null_mut};

pub type Trip = (Index, Index, f64);

pub const TRUE: Bool = 1;
pub const FALSE: Bool = 0;

pub fn cs(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// Owns a handle so a failing assertion still frees it.
pub struct Qp {
    pub h: PounceQp,
    pub n: usize,
    pub m_eq: usize,
    pub m_ineq: usize,
}

impl Drop for Qp {
    fn drop(&mut self) {
        unsafe { PounceFreeQp(self.h) }
    }
}

fn split(t: &[Trip]) -> (Vec<Index>, Vec<Index>, Vec<f64>) {
    (
        t.iter().map(|e| e.0).collect(),
        t.iter().map(|e| e.1).collect(),
        t.iter().map(|e| e.2).collect(),
    )
}

impl Qp {
    pub fn new(n: usize, m_eq: usize, m_ineq: usize, style: Index) -> Qp {
        let h = unsafe { PounceCreateQp(n as Index, m_eq as Index, m_ineq as Index, style) };
        assert!(
            !h.is_null(),
            "PounceCreateQp({n}, {m_eq}, {m_ineq}, {style})"
        );
        Qp { h, n, m_eq, m_ineq }
    }

    pub fn objective(&self, p: &[Trip], c: &[f64]) -> Bool {
        let (r, cc, v) = split(p);
        unsafe {
            PounceQpSetObjective(
                self.h,
                r.len() as Index,
                r.as_ptr(),
                cc.as_ptr(),
                v.as_ptr(),
                c.as_ptr(),
            )
        }
    }

    pub fn equalities(&self, a: &[Trip], b: &[f64]) -> Bool {
        let (r, c, v) = split(a);
        unsafe {
            PounceQpSetEqualities(
                self.h,
                r.len() as Index,
                r.as_ptr(),
                c.as_ptr(),
                v.as_ptr(),
                b.as_ptr(),
            )
        }
    }

    pub fn inequalities(&self, g: &[Trip], h: &[f64]) -> Bool {
        let (r, c, v) = split(g);
        unsafe {
            PounceQpSetInequalities(
                self.h,
                r.len() as Index,
                r.as_ptr(),
                c.as_ptr(),
                v.as_ptr(),
                h.as_ptr(),
            )
        }
    }

    /// `None` passes NULL for that side.
    pub fn bounds(&self, lb: Option<&[f64]>, ub: Option<&[f64]>) -> Bool {
        unsafe {
            PounceQpSetBounds(
                self.h,
                lb.map_or(null(), |v| v.as_ptr()),
                ub.map_or(null(), |v| v.as_ptr()),
            )
        }
    }

    pub fn cone(&self, kind: i32, dim: Index, alpha: f64) -> i32 {
        unsafe { PounceQpAddCone(self.h, kind, dim, alpha) }
    }

    pub fn str_opt(&self, k: &str, v: &str) -> Bool {
        unsafe { PounceQpSetStrOption(self.h, cs(k).as_ptr(), cs(v).as_ptr()) }
    }

    pub fn num_opt(&self, k: &str, v: f64) -> Bool {
        unsafe { PounceQpSetNumOption(self.h, cs(k).as_ptr(), v) }
    }

    pub fn solve(&self) -> i32 {
        unsafe { PounceQpSolve(self.h) }
    }

    pub fn solution(&self) -> Sol {
        let mut s = Sol {
            x: vec![f64::NAN; self.n],
            y: vec![f64::NAN; self.m_eq],
            z: vec![f64::NAN; self.m_ineq],
            z_lb: vec![f64::NAN; self.n],
            z_ub: vec![f64::NAN; self.n],
            obj: f64::NAN,
            iters: -1,
        };
        let ok = unsafe {
            PounceQpGetSolution(
                self.h,
                s.x.as_mut_ptr(),
                s.y.as_mut_ptr(),
                s.z.as_mut_ptr(),
                s.z_lb.as_mut_ptr(),
                s.z_ub.as_mut_ptr(),
                &mut s.obj,
                &mut s.iters,
            )
        };
        assert_eq!(ok, TRUE, "PounceQpGetSolution");
        s
    }

    pub fn residuals(&self) -> PounceQpResiduals {
        let mut r = PounceQpResiduals::default();
        assert_eq!(unsafe { PounceQpGetResiduals(self.h, &mut r) }, TRUE);
        r
    }

    pub fn certificate(&self) -> PounceQpCertificate {
        let mut c = PounceQpCertificate::default();
        assert_eq!(unsafe { PounceQpGetCertificate(self.h, &mut c) }, TRUE);
        c
    }

    /// `PounceQpGetSolution` with every output NULL: whether a result exists.
    pub fn has_result(&self) -> bool {
        let n = null_mut();
        unsafe { PounceQpGetSolution(self.h, n, n, n, n, n, null_mut(), null_mut()) == TRUE }
    }
}

#[derive(Debug, Clone)]
pub struct Sol {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub z_lb: Vec<f64>,
    pub z_ub: Vec<f64>,
    pub obj: f64,
    pub iters: Index,
}

pub fn close(what: &str, a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (x, y) in a.iter().zip(b) {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "{what}: {a:?} vs {b:?}"
        );
    }
}
