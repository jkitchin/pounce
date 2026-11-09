//! gh#1147 Stage 2 — solving after `SetPounceBounds` (plan §6.4, S2-I3).
//!
//! The setter's contract for a solve is that an updated handle is the same
//! problem as a handle created with those bounds: same status, point,
//! objective and iteration count, to the bit, since `IpoptSolve` builds
//! its solver from the handle's bounds either way. Held on three models —
//! a plain NLP, native declared pairs, and MOI `Complements` declarations
//! (whose row edits the setter re-derives) — on both Hessian legs, because
//! the L-BFGS leg is what the Python frontend and CasADi select when no
//! exact Hessian is available (CLAUDE.md, the fixture-sweep section).
//!
//! Plus: an explicit warm re-solve after an update reaches the updated
//! problem's answer, and the previous solve's statistics stay readable
//! until the next solve.
//!
//! Model: `QpecSmall` (`common/qpec.rs`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/qpec.rs"]
mod qpec;

use pounce_cinterface::*;
use qpec::{G_L, G_U, INF, M, N, PAIRS, X_L, X_U, create};
use std::ffi::CString;
use std::ptr;

const SOLVE_SUCCEEDED: Index = 0;

fn opt_str(p: IpoptProblem, k: &str, v: &str) {
    let (k, v) = (CString::new(k).unwrap(), CString::new(v).unwrap());
    assert_eq!(unsafe { AddIpoptStrOption(p, k.as_ptr(), v.as_ptr()) }, 1);
}

fn opt_int(p: IpoptProblem, k: &str, v: Index) {
    let k = CString::new(k).unwrap();
    assert_eq!(unsafe { AddIpoptIntOption(p, k.as_ptr(), v) }, 1);
}

#[derive(Debug, PartialEq)]
struct Outcome {
    status: Index,
    x: [Number; N],
    obj: Number,
    iters: Index,
    mult_g: [Number; M],
    mult_x_l: [Number; N],
    mult_x_u: [Number; N],
}

fn solve_from(p: IpoptProblem, x0: [Number; N], warm: Option<&Outcome>) -> Outcome {
    let mut x = x0;
    let mut obj = Number::NAN;
    let mut mult_g = warm.map_or([0.0; M], |w| w.mult_g);
    let mut mult_x_l = warm.map_or([0.0; N], |w| w.mult_x_l);
    let mut mult_x_u = warm.map_or([0.0; N], |w| w.mult_x_u);
    let status = unsafe {
        IpoptSolve(
            p,
            x.as_mut_ptr(),
            ptr::null_mut(),
            &mut obj,
            mult_g.as_mut_ptr(),
            mult_x_l.as_mut_ptr(),
            mult_x_u.as_mut_ptr(),
            ptr::null_mut(),
        )
    };
    Outcome {
        status,
        x,
        obj,
        iters: unsafe { GetIpoptIterCount(p) },
        mult_g,
        mult_x_l,
        mult_x_u,
    }
}

const START: [Number; N] = [0.0, 0.5, 0.5];

/// How the model is declared on a fresh handle.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Nlp,
    NativePairs,
    Complements,
}

/// A configured handle of `kind` with the given bounds. For `Complements`
/// the rows are created free and `[F_r, x_v]` declared for each pair, so
/// `g_l`/`g_u` are ignored.
fn build(
    kind: Kind,
    lbfgs: bool,
    x_l: &[Number; N],
    x_u: &[Number; N],
    g_l: &[Number; M],
    g_u: &[Number; M],
) -> IpoptProblem {
    let p = match kind {
        Kind::Complements => create(x_l, x_u, &[-INF; M], &[INF; M], 0),
        _ => create(x_l, x_u, g_l, g_u, 0),
    };
    opt_int(p, "print_level", 0);
    if lbfgs {
        opt_str(p, "hessian_approximation", "limited-memory");
    }
    for &(r, v) in &PAIRS {
        match kind {
            Kind::Nlp => {}
            Kind::NativePairs => {
                assert_eq!(unsafe { AddPounceComplementarityPair(p, r, v) }, POUNCE_OK)
            }
            Kind::Complements => {
                assert_eq!(
                    unsafe { AddPounceComplementsConstraint(p, r, v) },
                    POUNCE_COMPLEMENTS_PAIR_LOWER
                )
            }
        }
    }
    p
}

/// The update every parity case applies: x's box shrinks from `[0, 2]` to
/// `[0.25, 0.75]`, which moves the answer off `(1, 1, 0)`; y1's lower bound
/// moves to 0.1 (a paired bound's value: allowed); for the declared-row
/// models row 1's lower side moves too.
fn new_bounds(kind: Kind) -> ([Number; N], [Number; N], [Number; M], [Number; M]) {
    let x_l = [0.25, 0.1, X_L[2]];
    let x_u = [0.75, X_U[1], X_U[2]];
    let (g_l, g_u) = match kind {
        Kind::Complements => ([-INF; M], [INF; M]),
        _ => ([G_L[0], -0.2], G_U),
    };
    (x_l, x_u, g_l, g_u)
}

fn update_dense(
    p: IpoptProblem,
    x_l: &[Number; N],
    x_u: &[Number; N],
    g_l: &[Number; M],
    g_u: &[Number; M],
) {
    let mut bad = -42;
    let code = unsafe {
        SetPounceBounds(
            p,
            N as Index,
            ptr::null(),
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            ptr::null(),
            g_l.as_ptr(),
            g_u.as_ptr(),
            0,
            &mut bad,
        )
    };
    assert_eq!((code, bad), (POUNCE_OK, -1));
}

#[test]
fn an_updated_handle_solves_exactly_like_a_fresh_one() {
    for kind in [Kind::Nlp, Kind::NativePairs, Kind::Complements] {
        for lbfgs in [false, true] {
            let (x_l, x_u, g_l, g_u) = new_bounds(kind);
            // Reference: created with the new bounds.
            let fresh = build(kind, lbfgs, &x_l, &x_u, &g_l, &g_u);
            let want = solve_from(fresh, START, None);
            unsafe { FreeIpoptProblem(fresh) };
            assert_eq!(want.status, SOLVE_SUCCEEDED, "{kind:?} lbfgs={lbfgs}");
            assert!(
                (want.x[0] - 0.75).abs() < 1e-6,
                "{kind:?} lbfgs={lbfgs}: the update should move x to its new upper bound, got {:?}",
                want.x
            );

            // Created with the original bounds, solved, then updated.
            let p = build(kind, lbfgs, &X_L, &X_U, &G_L, &G_U);
            let before = solve_from(p, START, None);
            assert_eq!(before.status, SOLVE_SUCCEEDED);
            update_dense(p, &x_l, &x_u, &g_l, &g_u);
            let got = solve_from(p, START, None);
            assert_eq!(got, want, "{kind:?} lbfgs={lbfgs}");
            unsafe { FreeIpoptProblem(p) };
        }
    }
}

#[test]
fn a_sparse_update_equals_the_dense_one() {
    // Only the entries that change, in a scrambled order.
    for kind in [Kind::Nlp, Kind::NativePairs] {
        let (x_l, x_u, g_l, g_u) = new_bounds(kind);
        let dense = build(kind, false, &X_L, &X_U, &G_L, &G_U);
        update_dense(dense, &x_l, &x_u, &g_l, &g_u);
        let want = solve_from(dense, START, None);
        unsafe { FreeIpoptProblem(dense) };

        let p = build(kind, false, &X_L, &X_U, &G_L, &G_U);
        let (xi, xl, xu) = ([1, 0], [x_l[1], x_l[0]], [x_u[1], x_u[0]]);
        let (gi, gl, gu) = ([1], [g_l[1]], [g_u[1]]);
        let code = unsafe {
            SetPounceBounds(
                p,
                2,
                xi.as_ptr(),
                xl.as_ptr(),
                xu.as_ptr(),
                1,
                gi.as_ptr(),
                gl.as_ptr(),
                gu.as_ptr(),
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(code, POUNCE_OK);
        assert_eq!(solve_from(p, START, None), want, "{kind:?}");
        unsafe { FreeIpoptProblem(p) };
    }
}

#[test]
fn an_explicit_warm_resolve_after_an_update_reaches_the_new_answer() {
    let (x_l, x_u, g_l, g_u) = new_bounds(Kind::Nlp);
    let fresh = build(Kind::Nlp, false, &x_l, &x_u, &g_l, &g_u);
    let cold = solve_from(fresh, START, None);
    unsafe { FreeIpoptProblem(fresh) };

    let p = build(Kind::Nlp, false, &X_L, &X_U, &G_L, &G_U);
    let first = solve_from(p, START, None);
    assert_eq!(first.status, SOLVE_SUCCEEDED);
    update_dense(p, &x_l, &x_u, &g_l, &g_u);
    opt_str(p, "warm_start_init_point", "yes");
    // The previous point, projected into the new box by the solver.
    let warm = solve_from(p, first.x, Some(&first));
    assert_eq!(warm.status, SOLVE_SUCCEEDED);
    for (a, b) in warm.x.iter().zip(&cold.x) {
        assert!(
            (a - b).abs() < 1e-6,
            "warm {:?} vs cold {:?}",
            warm.x,
            cold.x
        );
    }
    assert!((warm.obj - cold.obj).abs() < 1e-8);
    unsafe { FreeIpoptProblem(p) };
}

#[test]
fn the_previous_solve_stays_readable_until_the_next() {
    let p = build(Kind::NativePairs, false, &X_L, &X_U, &G_L, &G_U);
    let first = solve_from(p, START, None);
    let mut stats: PounceComplementarityStats = unsafe { std::mem::zeroed() };
    stats.struct_size = std::mem::size_of::<PounceComplementarityStats>() as Index;
    assert_eq!(unsafe { GetPounceComplementarityStats(p, &mut stats) }, 1);
    let total = stats.total_iterations;
    let (x_l, x_u, g_l, g_u) = new_bounds(Kind::NativePairs);
    update_dense(p, &x_l, &x_u, &g_l, &g_u);
    assert_eq!(unsafe { GetIpoptIterCount(p) }, first.iters);
    assert_eq!(unsafe { GetPounceComplementarityStats(p, &mut stats) }, 1);
    assert_eq!(stats.total_iterations, total);
    unsafe { FreeIpoptProblem(p) };
}

#[test]
fn the_reported_row_multipliers_are_the_models() {
    // MOI Complements QpecSmall with x in [0.25, 0.75]: the answer
    // (3/4, 7/8, 1/8) has both pairs non-degenerate, so its multipliers
    // are unique: λ = (1/8, -1/8), z_U[x] = 3/4 (∇f + Jᵀλ - z_L + z_U = 0).
    // They were reported as the lowered model's row/product split,
    // [-2.2e7, -1.8e7]; fixed in the driver for every frontend.
    for lbfgs in [false, true] {
        let (x_l, x_u, g_l, g_u) = new_bounds(Kind::Complements);
        let p = build(Kind::Complements, lbfgs, &x_l, &x_u, &g_l, &g_u);
        let o = solve_from(p, START, None);
        unsafe { FreeIpoptProblem(p) };
        assert_eq!(o.status, SOLVE_SUCCEEDED);
        let ctx = format!("lbfgs={lbfgs}: {o:?}");
        assert!((o.mult_g[0] - 0.125).abs() < 1e-5, "{ctx}");
        assert!((o.mult_g[1] + 0.125).abs() < 1e-5, "{ctx}");
        assert!((o.mult_x_u[0] - 0.75).abs() < 1e-5, "{ctx}");
        assert!(o.mult_x_l.iter().all(|z| z.abs() < 1e-6), "{ctx}");
    }
}
