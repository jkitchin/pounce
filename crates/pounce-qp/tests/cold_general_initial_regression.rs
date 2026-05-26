//! Regression for PR #50 review finding C1 (`cold_general_initial`
//! RHS allegedly missing `−Hx`).
//!
//! `cold_general_initial` solves the equality-relaxed KKT system
//!
//!     [H  Aᵀ_eq] [x  ]   [-g ]
//!     [A_eq 0  ] [λ_eq] = [b_eq]
//!
//! directly for `x` — there is no "current x" to subtract. The
//! result `x*` is the constrained minimizer of `½ xᵀ H x + gᵀ x`
//! over `Ax = b_eq`. RHS = `[−g; b]` is the correct primal-direct
//! form (the `H` term is in the matrix factor, not the RHS).
//!
//! Compare to `solve_general`'s inner loop, which solves for a
//! *step* `p` from a current `x`: there the RHS is `−(Hx + g)`.
//! Different formulations; both are right.
//!
//! Fixture: `min ½(4x₁² + 4x₂²) − 2x₁ − 2x₂  s.t.  x₁ + x₂ = 1`.
//! Closed form: KKT 4x_i − 2 + λ = 0, x₁ + x₂ = 1
//!   ⇒ x_i = 0.5, λ = 0.
//! f* = ½ · (4·0.25 + 4·0.25) + (−2·0.5 + −2·0.5) = 1 − 2 = −1.
//!
//! If C1 were a real bug, `cold_general_initial` would return a
//! non-zero primal whose H-shifted RHS was missed — concretely,
//! `x ≠ 0.5`. The test below pins the exact closed form and so
//! catches any future regression in the RHS construction.

use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver};
use std::rc::Rc;

#[test]
fn cold_general_initial_solves_qp_with_nonzero_hessian_at_solution() {
    let h_space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
    let mut h = SymTMatrix::new(Rc::clone(&h_space));
    h.set_values(&[4.0, 4.0]);
    let a_space = GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
    let mut a = GenTMatrix::new(Rc::clone(&a_space));
    a.set_values(&[1.0, 1.0]);
    let g = [-2.0, -2.0];
    let bl = [1.0];
    let bu = [1.0];
    let xl = [-1e20, -1e20];
    let xu = [1e20, 1e20];
    let qp = QpProblem {
        n: 2,
        m: 1,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert!((sol.x[0] - 0.5).abs() < 1e-10, "x[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 0.5).abs() < 1e-10, "x[1] = {}", sol.x[1]);
    assert!((sol.obj - (-1.0)).abs() < 1e-10, "obj = {}", sol.obj);
}
