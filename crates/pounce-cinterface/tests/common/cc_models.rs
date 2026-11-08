//! The gh#1147 complementarity models as C callbacks, and readers for the
//! outcome getters, shared by `complementarity_getters.rs` and
//! `complementarity_report.rs`. Include after `qpec`:
//!
//! ```ignore
//! #[path = "common/qpec.rs"]
//! mod qpec;
//! #[path = "common/cc_models.rs"]
//! mod cc_models;
//! ```
//!
//! | class          | model                                   | reported        |
//! |----------------|-----------------------------------------|-----------------|
//! | `S`            | `QpecSmall` (`common/qpec.rs`)          | answer          |
//! | `W`            | `ralph1` (`fixtures_mpcc/ralph1.nl`)    | weak fallback   |
//! | `none`         | `infeasible_pair` (`fixtures_mpcc/`)    | verdict         |
//! | `not searched` | 605 active bounds, a duplicate row      | answer          |

#![allow(dead_code)]

use crate::qpec;
use pounce_cinterface::*;
use std::ffi::{CStr, c_char, c_void};
use std::ptr;

pub const INF: Number = 2e19;

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// A linear model through `user_data`: `f = c·x`, `g = J x` with `J` as
/// `(row, col, value)` triplets, no Hessian. `ralph1`, `infeasible_pair`
/// and the not-searched model are all of this form.
pub struct Linear {
    pub c: Vec<Number>,
    pub jac: Vec<(Index, Index, Number)>,
    pub m: usize,
}

pub unsafe fn linear<'a>(u: *mut c_void) -> &'a Linear {
    unsafe { &*(u as *const Linear) }
}

pub unsafe extern "C" fn lin_f(
    n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, n as usize);
        *obj = linear(u).c.iter().zip(x).map(|(c, x)| c * x).sum();
        1
    }
}

pub unsafe extern "C" fn lin_grad_f(
    n: Index,
    _x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        std::slice::from_raw_parts_mut(grad, n as usize).copy_from_slice(&linear(u).c);
        1
    }
}

pub unsafe extern "C" fn lin_g(
    n: Index,
    x: *const Number,
    _new_x: Bool,
    m: Index,
    g: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, n as usize);
        let g = std::slice::from_raw_parts_mut(g, m as usize);
        g.fill(0.0);
        for &(i, j, v) in &linear(u).jac {
            g[i as usize] += v * x[j as usize];
        }
        1
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn lin_jac_g(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    _m: Index,
    nele: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let jac = &linear(u).jac;
        if values.is_null() {
            let r = std::slice::from_raw_parts_mut(i_row, nele as usize);
            let c = std::slice::from_raw_parts_mut(j_col, nele as usize);
            for (k, &(i, j, _)) in jac.iter().enumerate() {
                r[k] = i;
                c[k] = j;
            }
        } else {
            let v = std::slice::from_raw_parts_mut(values, nele as usize);
            for (k, &(_, _, a)) in jac.iter().enumerate() {
                v[k] = a;
            }
        }
        1
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn no_h(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    _obj_factor: Number,
    _m: Index,
    _lambda: *const Number,
    _new_lambda: Bool,
    _nele_hess: Index,
    _i_row: *mut Index,
    _j_col: *mut Index,
    _values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    1
}

/// A model, its handle and its start.
pub struct Model {
    pub prob: IpoptProblem,
    /// Boxed so the pointer handed to `IpoptSolve` stays put.
    pub data: Option<Box<Linear>>,
    pub x0: Vec<Number>,
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { FreeIpoptProblem(self.prob) };
    }
}

impl Model {
    pub fn linear(
        lin: Linear,
        x_l: &[Number],
        x_u: &[Number],
        g_l: &[Number],
        g_u: &[Number],
        pairs: &[(Index, Index)],
        x0: Vec<Number>,
    ) -> Model {
        assert_eq!(lin.m, g_l.len());
        let prob = unsafe {
            CreateIpoptProblem(
                x_l.len() as Index,
                x_l.as_ptr(),
                x_u.as_ptr(),
                g_l.len() as Index,
                g_l.as_ptr(),
                g_u.as_ptr(),
                lin.jac.len() as Index,
                0,
                0,
                Some(lin_f),
                Some(lin_g),
                Some(lin_grad_f),
                Some(lin_jac_g),
                Some(no_h),
            )
        };
        assert!(!prob.is_null(), "CreateIpoptProblem returned NULL");
        let m = Model {
            prob,
            data: Some(Box::new(lin)),
            x0,
        };
        m.declare(pairs);
        m
    }

    /// `QpecSmall` with its pairs, from `x0`.
    pub fn qpec(x0: [Number; qpec::N]) -> Model {
        let m = Model {
            prob: qpec::create_qpec(0),
            data: None,
            x0: x0.to_vec(),
        };
        m.declare(&qpec::PAIRS);
        m
    }

    /// `ralph1`: `min 2x - y`, `x >= 0`, `0 <= y ⟂ y - x >= 0`, from `(1, 2)`.
    pub fn ralph1() -> Model {
        let lin = Linear {
            c: vec![2.0, -1.0],
            jac: vec![(0, 0, -1.0), (0, 1, 1.0)],
            m: 1,
        };
        Model::linear(
            lin,
            &[0.0, 0.0],
            &[INF, INF],
            &[0.0],
            &[INF],
            &[(0, 1)],
            vec![1.0, 2.0],
        )
    }

    /// `infeasible_pair`: `min x + y`, rows `x >= 1`, `y >= 1`, and
    /// `0 <= x ⟂ y >= 0` (row 2), from `(2, 2)`.
    pub fn infeasible_pair() -> Model {
        let lin = Linear {
            c: vec![1.0, 1.0],
            jac: vec![(0, 0, 1.0), (1, 1, 1.0), (2, 1, 1.0)],
            m: 3,
        };
        Model::linear(
            lin,
            &[0.0, 0.0],
            &[INF, INF],
            &[1.0, 1.0, 0.0],
            &[INF, INF, INF],
            &[(2, 0)],
            vec![2.0, 2.0],
        )
    }

    /// `min Σ x`, `x >= 0` (605 variables), row 0 `x_0 >= 0` (duplicating a
    /// bound, so the multipliers are a set), row 1 `x_1 >= 0 ⟂ x_2 >= 0`.
    /// At the answer `x = 0` every bound is active: more sign-constrained
    /// columns than the set search takes. From all ones.
    pub fn not_searched() -> Model {
        const NS: usize = 605;
        let lin = Linear {
            c: vec![1.0; NS],
            jac: vec![(0, 0, 1.0), (1, 1, 1.0)],
            m: 2,
        };
        Model::linear(
            lin,
            &[0.0; NS],
            &[INF; NS],
            &[0.0, 0.0],
            &[INF, INF],
            &[(1, 2)],
            vec![1.0; NS],
        )
    }

    pub fn declare(&self, pairs: &[(Index, Index)]) {
        let (rows, vars): (Vec<Index>, Vec<Index>) = pairs.iter().copied().unzip();
        let rc = unsafe {
            SetPounceComplementarityPairs(
                self.prob,
                pairs.len() as Index,
                rows.as_ptr(),
                vars.as_ptr(),
                ptr::null_mut(),
            )
        };
        assert_eq!(rc, POUNCE_CC_OK);
    }

    pub fn option(&self, key: &str, value: &str) {
        let k = std::ffi::CString::new(key).unwrap();
        let v = std::ffi::CString::new(value).unwrap();
        assert_eq!(
            unsafe { AddIpoptStrOption(self.prob, k.as_ptr(), v.as_ptr()) },
            1,
            "{key}={value} refused"
        );
    }

    /// Solve quietly from the model's start; the status and the point.
    pub fn solve(&self) -> (Index, Vec<Number>) {
        let pl = std::ffi::CString::new("print_level").unwrap();
        unsafe { AddIpoptIntOption(self.prob, pl.as_ptr(), 0) };
        let mut x = self.x0.clone();
        let u = self
            .data
            .as_ref()
            .map_or(ptr::null_mut(), |d| &**d as *const Linear as *mut c_void);
        let status = unsafe {
            IpoptSolve(
                self.prob,
                x.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                u,
            )
        };
        (status, x)
    }
}

// ---------------------------------------------------------------------------
// Reading the getters
// ---------------------------------------------------------------------------

/// A struct filled with a byte pattern no getter writes, so "untouched" is
/// observable.
pub fn sentinel<T: Copy>() -> T {
    let mut v = std::mem::MaybeUninit::<T>::uninit();
    unsafe {
        ptr::write_bytes(v.as_mut_ptr(), 0xA5, 1);
        v.assume_init()
    }
}

pub fn bytes_of<T>(v: &T) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, size_of::<T>()) }.to_vec()
}

pub fn stats(prob: IpoptProblem) -> Option<PounceComplementarityStats> {
    let mut s: PounceComplementarityStats = sentinel();
    s.struct_size = size_of::<PounceComplementarityStats>() as Index;
    let before = bytes_of(&s);
    let ok = unsafe { GetPounceComplementarityStats(prob, &mut s) };
    if ok == 0 {
        assert_eq!(bytes_of(&s), before, "a FALSE stats call wrote the struct");
        return None;
    }
    assert_eq!(ok, 1);
    assert_eq!(
        s.struct_size,
        size_of::<PounceComplementarityStats>() as Index
    );
    Some(s)
}

pub fn solve_record(prob: IpoptProblem, k: Index) -> Option<PounceComplementaritySolve> {
    let mut s: PounceComplementaritySolve = sentinel();
    s.struct_size = size_of::<PounceComplementaritySolve>() as Index;
    let before = bytes_of(&s);
    let ok = unsafe { GetPounceComplementaritySolve(prob, k, &mut s) };
    if ok == 0 {
        assert_eq!(bytes_of(&s), before, "a FALSE solve call wrote the record");
        return None;
    }
    assert_eq!(ok, 1);
    assert_eq!(
        s.struct_size,
        size_of::<PounceComplementaritySolve>() as Index
    );
    Some(s)
}

/// The text, read the documented way: size it, then fill an exact buffer.
pub fn text(prob: IpoptProblem, which: i32) -> Option<String> {
    let len = unsafe { GetPounceComplementarityText(prob, which, ptr::null_mut(), 0) };
    if len < 0 {
        assert_eq!(len, -1);
        return None;
    }
    let mut buf = vec![0x7f as c_char; len as usize + 1];
    let again = unsafe { GetPounceComplementarityText(prob, which, buf.as_mut_ptr(), len + 1) };
    assert_eq!(
        again, len,
        "the fill must return the length the size query did"
    );
    assert_eq!(buf[len as usize], 0, "no NUL at buf[len]");
    let s = unsafe { CStr::from_ptr(buf.as_ptr()) };
    assert_eq!(s.to_bytes().len(), len as usize, "an interior NUL");
    Some(s.to_str().expect("UTF-8").to_string())
}

pub fn label(chars: &[c_char]) -> &str {
    let end = chars
        .iter()
        .position(|&c| c == 0)
        .expect("label is NUL-terminated");
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(chars.as_ptr() as *const u8, end) };
    std::str::from_utf8(bytes).expect("UTF-8 label")
}

/// Bitwise equality, with every NaN equal to every NaN (the getters' and
/// the reference's "not measured").
pub fn same(a: Number, b: Number) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

#[track_caller]
pub fn assert_same(what: &str, got: Number, want: Number) {
    assert!(same(got, want), "{what}: C {got:e}, Python {want:e}");
}

pub fn class_code(name: &str) -> i32 {
    match name {
        "S" => POUNCE_CC_CLASS_S,
        "W" => POUNCE_CC_CLASS_W,
        "none" => POUNCE_CC_CLASS_NONE,
        "not_searched" => POUNCE_CC_CLASS_NOT_SEARCHED,
        other => panic!("unknown class {other}"),
    }
}

pub fn b_code(name: &str) -> i32 {
    match name {
        "B" => POUNCE_CC_B_VERDICT_B,
        "not_B" => POUNCE_CC_B_VERDICT_NOT_B,
        "not_checked" => POUNCE_CC_B_VERDICT_NOT_CHECKED,
        other => panic!("unknown verdict {other}"),
    }
}

pub const STAGES: [&str; 5] = [
    "exact product",
    "Scholtes continuation",
    "l1 penalty",
    "branch NLP",
    "branch polish",
];
