//! An approximate Farkas ray must not certify a feasible model whose
//! feasible points lie far out, on either convex engine.
//!
//! `data/false_infeas_lp_no_box.txt`: `min cᵀx s.t. Gx ≤ h`, 18 × 18,
//! no variable bounds, data of order `1e6`. `G` is nonsingular and `Gx = h`
//! has an exact rational solution (`‖x‖∞ ≈ 1.07e9`), so the model is
//! feasible; with this `c` it is unbounded (HiGHS agrees). The HSDE used to
//! return `PrimalInfeasible` at iteration 9 off a ray with
//! `hᵀz/‖z‖ = −1.9e-5` and `‖Gᵀz‖ = 5.5e-12`. Both pass the absolute
//! tolerances, but the ray excludes feasible points only out to
//! `|hᵀz|/‖Gᵀz‖ ≈ 3.4e6`, about 1× the data's own scale. See
//! `ipm::FarkasMargin`.
//!
//! The complementary branch, which must stay certified with no box, is
//! pinned by the `detect_infeasibility_tests` unit tests in `ipm.rs` and by
//! `crates/pounce-cli/tests/farkas_far_feasible_not_infeasible.rs`.

use pounce_convex::{
    ActiveSetOverrides, QpOptions, QpProblem, QpStatus, Triplet, solve_qp_active_set, solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// Parse `n`, `c`, `h` and one `G` line per row.
fn load(zero_objective: bool) -> QpProblem {
    let text = include_str!("data/false_infeas_lp_no_box.txt");
    let nums = |rest: &str| -> Vec<f64> {
        rest.split_whitespace()
            .map(|v| v.parse::<f64>().unwrap_or(f64::NAN))
            .collect()
    };
    let (mut n, mut c, mut h, mut g) = (0, vec![], vec![], vec![]);
    let mut row = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let (key, rest) = line.split_once(' ').unwrap_or((line, ""));
        match key {
            "n" => n = rest.trim().parse().unwrap_or(0),
            "c" => c = nums(rest),
            "h" => h = nums(rest),
            "G" => {
                for (j, v) in nums(rest).into_iter().enumerate() {
                    if v != 0.0 {
                        g.push(Triplet::new(row, j, v));
                    }
                }
                row += 1;
            }
            _ => {}
        }
    }
    assert_eq!((n, c.len(), h.len(), row), (18, 18, 18, 18));
    if zero_objective {
        c = vec![0.0; n];
    }
    QpProblem {
        n,
        p_lower: vec![],
        c,
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![],
        ub: vec![],
    }
}

fn ipm(prob: &QpProblem) -> QpStatus {
    let mut mk = backend;
    solve_qp_ipm(prob, &QpOptions::default(), &mut mk).status
}

fn active_set(prob: &QpProblem) -> QpStatus {
    let mut mk = backend;
    solve_qp_active_set(
        prob,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        &mut mk,
    )
    .status
}

#[test]
fn ipm_does_not_certify_the_far_feasible_lp() {
    let s = ipm(&load(false));
    assert_ne!(s, QpStatus::PrimalInfeasible);
    assert_eq!(s, QpStatus::DualInfeasible, "the LP is unbounded");
    assert_eq!(ipm(&load(true)), QpStatus::Optimal);
}

#[test]
fn active_set_does_not_certify_the_far_feasible_lp() {
    assert_ne!(active_set(&load(false)), QpStatus::PrimalInfeasible);
    assert_ne!(active_set(&load(true)), QpStatus::PrimalInfeasible);
}
