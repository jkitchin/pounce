//! gh#971 — on a convex QP whose equality block carries one **consistent,
//! redundant** row (row 3 = row 1 + row 2), `qp-active-set` walked the
//! equality multipliers along `null(Aᵀ)` until they hit exactly `±1e6`, and
//! ended `optimal_inaccurate` / `Solved_To_Acceptable_Level` with
//! `|Ax − b| = 3.1e-6`. Dropping the redundant row — which leaves the feasible
//! set unchanged — gave `optimal` with `|y| ≈ 1`.
//!
//! ## The chain
//!
//! The cold homotopy reaches `t = 1` off the target, and its corrector falls
//! through to l1-elastic mode. The elastic reformulation gives every row its
//! own slack pair, so the rank-deficient equality block no longer makes any
//! KKT matrix singular and none of the rank guards fire. What is left is a
//! multiplier with a null space, bounded only by the slack pairs' own
//! multipliers `|λᵢ| ≤ γ = 1e6`. `±1e6` is that cap, not a converged value.
//!
//! The fix rank-reveals the equality rows before building the reformulation
//! and runs elastic on an independent subset, accepting the result only when
//! every dropped row holds at it — so an *inconsistent* dependent row still
//! goes through the unpruned solve, which owns the infeasibility verdict.
//!
//! Data: `numpy.random.default_rng(seed)` exactly as in the issue's
//! reproduction, printed with `repr` so every digit is the generator's.
//! Oracle: Clarabel, `-0.30573428035208106` (seed 51) and
//! `-0.2499623364970595` (seed 155).

use pounce_convex::{
    ActiveSetOverrides, QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_active_set,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn active_set(p: &QpProblem) -> QpSolution {
    let mut mk = backend;
    solve_qp_active_set(
        p,
        &QpOptions::default(),
        &ActiveSetOverrides::default(),
        &mut mk,
    )
}

struct Case {
    c: [f64; 4],
    a: [[f64; 4]; 3],
    b: [f64; 3],
    v: [f64; 4],
}

fn problem(k: &Case, rows: usize) -> QpProblem {
    let mut p_lower = Vec::new();
    for i in 0..4 {
        for j in 0..=i {
            p_lower.push(Triplet::new(i, j, k.v[i] * k.v[j]));
        }
    }
    let mut a = Vec::new();
    for (i, r) in k.a.iter().take(rows).enumerate() {
        for (j, &v) in r.iter().enumerate() {
            a.push(Triplet::new(i, j, v));
        }
    }
    QpProblem {
        n: 4,
        p_lower,
        c: k.c.to_vec(),
        a,
        b: k.b[..rows].to_vec(),
        g: vec![],
        h: vec![],
        lb: vec![-2.0; 4],
        ub: vec![2.0; 4],
    }
}

const SEED51: Case = Case {
    c: [
        -0.5751108303105836,
        -0.6220993166194774,
        0.08025479526353546,
        1.2524723497116306,
    ],
    a: [
        [
            -0.32389662147366177,
            -1.1016437567606532,
            -0.7987351061217752,
            1.7774511121114445,
        ],
        [
            -0.3504098615772675,
            -1.1836250641849182,
            -0.3022892878974984,
            0.29827317660283587,
        ],
        [
            -0.6743064830509293,
            -2.2852688209455714,
            -1.1010243940192737,
            2.07572428871428,
        ],
    ],
    b: [1.5775062145316032, 0.004979077047641223, 1.5824852915792444],
    v: [
        1.508860040333287,
        0.2870095032733602,
        0.30000190493599066,
        -0.6790304942926411,
    ],
};

const SEED155: Case = Case {
    c: [
        0.41533064386674734,
        -1.389847552821441,
        -0.41502780761130875,
        0.1039154844102442,
    ],
    a: [
        [
            1.8207619861986173,
            -2.7456402747587245,
            0.7995850935329225,
            0.8712283527052398,
        ],
        [
            0.9770683785838647,
            -0.9210003765043308,
            -0.40674360602557913,
            -0.10211300182846439,
        ],
        [
            2.797830364782482,
            -3.666640651263055,
            0.3928414875073434,
            0.7691153508767754,
        ],
    ],
    b: [
        -0.37147458000233335,
        -0.08041693825355356,
        -0.45189151825588647,
    ],
    v: [
        -1.8273390143890733,
        -0.24535960128069498,
        0.18582556752464716,
        0.4180888279857321,
    ],
};

/// `max |Ax − b|` over the equality rows.
fn eq_residual(p: &QpProblem, x: &[f64]) -> f64 {
    let mut ax = vec![0.0; p.b.len()];
    for t in &p.a {
        ax[t.row] += t.val * x[t.col];
    }
    ax.iter()
        .zip(&p.b)
        .fold(0.0_f64, |w, (a, b)| w.max((a - b).abs()))
}

/// `‖Px + c + Aᵀy − z‖∞` with the bound multipliers read off the sign of the
/// residual at an active bound — the stationarity the ±1e6 multipliers only
/// satisfied up to their own round-off.
fn stationarity(p: &QpProblem, s: &QpSolution) -> f64 {
    let mut r = p.c.clone();
    for t in &p.p_lower {
        r[t.row] += t.val * s.x[t.col];
        if t.row != t.col {
            r[t.col] += t.val * s.x[t.row];
        }
    }
    for t in &p.a {
        r[t.col] += t.val * s.y[t.row];
    }
    r.iter()
        .enumerate()
        .filter(|&(j, _)| (s.x[j] - p.lb[j]).abs() >= 1e-9 && (s.x[j] - p.ub[j]).abs() >= 1e-9)
        .fold(0.0_f64, |w, (_, v)| w.max(v.abs()))
}

fn check(k: &Case, f_star: f64, seed: u64) {
    let full = problem(k, 3);
    let sol = active_set(&full);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "seed {seed}: gh#971 ended {:?} with y = {:?}",
        sol.status,
        sol.y
    );
    let ymax = sol.y.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        ymax < 10.0,
        "seed {seed}: equality multipliers drifted along null(Aᵀ): y = {:?}",
        sol.y
    );
    let res = eq_residual(&full, &sol.x);
    assert!(res < 1e-8, "seed {seed}: |Ax − b| = {res:e}");
    let st = stationarity(&full, &sol);
    assert!(st < 1e-8, "seed {seed}: stationarity residual {st:e}");
    assert!(
        (sol.obj - f_star).abs() < 1e-8,
        "seed {seed}: objective {} against Clarabel {f_star}",
        sol.obj
    );

    // The same engine on the two independent rows — the issue's second oracle.
    let dropped = active_set(&problem(k, 2));
    assert_eq!(dropped.status, QpStatus::Optimal);
    assert!((sol.obj - dropped.obj).abs() < 1e-8);
}

const SEED2: Case = Case {
    c: [
        0.18905338179353307,
        -0.5227484414807474,
        -0.41306354339189344,
        -2.4414673826398556,
    ],
    a: [
        [
            1.799707382720902,
            1.1441658720372287,
            -0.32542283686782436,
            0.7738065867276614,
        ],
        [
            0.28121066979764925,
            -0.5538228364240524,
            0.9775674511260357,
            -0.31055654665915255,
        ],
        [
            2.0809180525185513,
            0.5903430356131764,
            0.6521446142582114,
            0.4632500400685089,
        ],
    ],
    b: [
        0.4012918483643172,
        -0.45910025556724554,
        -0.05780840720292829,
    ],
    v: [
        0.5452887139646817,
        -0.6071856998706371,
        0.12682784711186987,
        -0.8922740434297903,
    ],
};

#[test]
fn seed_51_redundant_row_is_solved_with_bounded_multipliers() {
    check(&SEED51, -0.30573428035208106, 51);
}

#[test]
fn seed_155_redundant_row_is_solved_with_bounded_multipliers() {
    check(&SEED155, -0.2499623364970595, 155);
}

/// Seed 2 takes the polish branch. The prune keeps rows 0 and 2, and the
/// elastic solve on that *full-rank* pair stalls in the Schur-update path: an
/// elastic slack active at its bound carries the SMW solve's pin-row error
/// (`−1.26e-11`), the model step cap takes `α = 5195` along a near-flat
/// δ-shifted direction, and the slack lands at `−6.6e-8`. Phase-1 ends
/// `MaxIter`, the consistency gate refuses it, and the full-row fallback
/// returns `Optimal` with `λ = (1e6, 1e6, −1e6)` — the same ±1e6 shadow prices
/// under a status the issue's count never flagged (105 of the generator's 200
/// seeds were in that class). Its feasible point now seeds a warm phase-2 on
/// the independent rows. The stall itself is not fixed here; see
/// `solve_elastic`'s doc for the repairs that were measured and rejected.
/// Oracle: the same engine on the two independent rows.
#[test]
fn seed_2_full_row_fallback_is_polished_onto_the_independent_rows() {
    let sol = active_set(&problem(&SEED2, 3));
    let two = active_set(&problem(&SEED2, 2));
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(two.status, QpStatus::Optimal);
    let ymax = sol.y.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(ymax < 10.0, "y = {:?}", sol.y);
    let full = problem(&SEED2, 3);
    assert!(eq_residual(&full, &sol.x) < 1e-8);
    assert!(stationarity(&full, &sol) < 1e-8);
    assert!(
        (sol.obj - two.obj).abs() < 1e-8,
        "obj {} vs two-row {}",
        sol.obj,
        two.obj
    );
}

/// The prune must not swallow an *inconsistent* dependent row: shifting row
/// 3's right-hand side by 0.5 makes the equality block contradictory with the
/// same rank. End to end this must not come back a success. Note that it
/// stays green with the engine's consistency check removed, because this
/// driver's `verify_status` re-derives the verdict and catches the violated
/// row on its own; the engine-level guard is pinned, mutation-checked, by
/// `pounce-qp`'s `issue_971_inconsistent_dependent_equality_is_not_pruned_to_optimal`.
#[test]
fn an_inconsistent_dependent_row_is_not_pruned_into_a_solution() {
    let mut p = problem(&SEED155, 3);
    p.b[2] += 0.5;
    let sol = active_set(&p);
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "an inconsistent equality block was reported optimal at x = {:?}",
        sol.x
    );
    assert_ne!(sol.status, QpStatus::OptimalInaccurate);
}
