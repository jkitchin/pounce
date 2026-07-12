//! Bound-tightening presolve: domain propagation shrinks variable boxes,
//! and an *active* tightened bound's multiplier is re-attributed to the row
//! that implied it (the multiplier on a non-real bound belongs to the
//! constraint, not the variable). Because that dual recovery is the subtle
//! part, this suite leans on **randomized KKT roundtrip** testing: many
//! random tightening-rich problems are solved with and without presolve,
//! and the postsolved `(x, y, z, z_lb, z_ub)` is checked to be a valid KKT
//! point of the *original* problem (and to match the direct primal).

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn direct(prob: &QpProblem) -> QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

fn with_presolve(prob: &QpProblem) -> QpSolution {
    solve_with_presolve(prob, |r| solve_qp_ipm(r, &QpOptions::default(), backend))
}

/// Bound-aware KKT validity to tolerance `tol`.
fn assert_kkt(prob: &QpProblem, sol: &QpSolution, tol: f64) {
    let n = prob.n;
    let mut g = prob.c.clone();
    prob.p_mul(&sol.x, &mut g);
    prob.at_mul(&sol.y, &mut g);
    prob.gt_mul(&sol.z, &mut g);
    for i in 0..n {
        let stat = g[i] + sol.z_ub[i] - sol.z_lb[i];
        assert!(stat.abs() < tol, "stationarity[{i}] = {stat}");
        assert!(
            sol.z_lb[i] > -tol && sol.z_ub[i] > -tol,
            "bound dual sign [{i}]"
        );
        assert!(
            sol.x[i] >= prob.lb_of(i) - tol && sol.x[i] <= prob.ub_of(i) + tol,
            "box [{i}]: {} in [{}, {}]",
            sol.x[i],
            prob.lb_of(i),
            prob.ub_of(i)
        );
        assert!(
            (sol.z_lb[i] * (sol.x[i] - prob.lb_of(i))).abs() < 1e-4,
            "lb comp [{i}]"
        );
        assert!(
            (sol.z_ub[i] * (prob.ub_of(i) - sol.x[i])).abs() < 1e-4,
            "ub comp [{i}]"
        );
    }
    let mut gx = vec![0.0; prob.m_ineq()];
    prob.g_mul(&sol.x, &mut gx);
    for i in 0..prob.m_ineq() {
        let slack = prob.h[i] - gx[i];
        assert!(slack > -tol, "Gx≤h row {i}: slack {slack}");
        assert!(sol.z[i] > -tol, "z[{i}] < 0");
        assert!((sol.z[i] * slack).abs() < 1e-4, "ineq comp row {i}");
    }
    let mut ax = vec![0.0; prob.m_eq()];
    prob.a_mul(&sol.x, &mut ax);
    for (i, (&axi, &bi)) in ax.iter().zip(&prob.b).enumerate() {
        assert!((axi - bi).abs() < tol, "Ax=b row {i}: {axi} vs {bi}");
    }
}

/// Tiny deterministic LCG, so the randomized sweep is reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unif(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
}

/// A specific hand-checked case: a singleton inequality tightens a box and
/// the bound is active, so the multiplier must move to the row.
#[test]
fn singleton_inequality_tightens_and_reattributes() {
    // min ½·2·(x0−5)² + ½·2·(x1−5)²  (via c=−10) s.t.  2·x0 ≤ 3,  0 ≤ x ≤ 10.
    // 2x0 ≤ 3 ⇒ x0 ≤ 1.5 (tightened); the objective pulls x0 to 5, so the
    // tightened bound is active. x1 is unconstrained ⇒ x1 = 5.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-10.0, -10.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 2.0)],
        h: vec![3.0],
        lb: vec![0.0, 0.0],
        ub: vec![10.0, 10.0],
    };
    match presolve(&prob) {
        PresolveOutcome::Reduced(ps) => assert!(ps.stats().tightened_bounds >= 1),
        other => panic!(
            "expected Reduced, got {:?}",
            matches!(other, PresolveOutcome::Reduced(_))
        ),
    }
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 1.5).abs() < 1e-5, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 5.0).abs() < 1e-5, "x1={}", sol.x[1]);
    assert_kkt(&prob, &sol, 1e-5);
    // The force holding x0 is the row, not the (slack) real bound: the
    // inequality multiplier is positive and the bound multiplier ~0.
    assert!(
        sol.z[0] > 0.1,
        "row multiplier should carry the force: {}",
        sol.z[0]
    );
    assert!(
        sol.z_ub[0].abs() < 1e-5,
        "real bound slack ⇒ z_ub≈0: {}",
        sol.z_ub[0]
    );
    let d = direct(&prob);
    assert!((sol.obj - d.obj).abs() < 1e-5);
}

/// Two-variable forcing-via-tightening: x0 − x1 ≤ −4 with 0≤x≤5 tightens
/// x0's upper toward 1 (when x1 at its min) — the other variable sits at
/// its activity bound, exercising the re-attribution's other-column path.
#[test]
fn pair_inequality_tightening() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-10.0, 6.0], // pull x0 up, push x1 down
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)],
        h: vec![-4.0],
        lb: vec![0.0, 0.0],
        ub: vec![5.0, 5.0],
    };
    let sol = with_presolve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_kkt(&prob, &sol, 1e-5);
    let d = direct(&prob);
    for i in 0..2 {
        assert!(
            (sol.x[i] - d.x[i]).abs() < 1e-5,
            "x[{i}]: {} vs {}",
            sol.x[i],
            d.x[i]
        );
    }
}

/// Randomized sweep: many tightening-rich problems, each KKT-validated and
/// primal-matched against a direct solve. Constraints live on disjoint
/// variable groups (singletons and pairs) so tightening fires often.
#[test]
fn randomized_bound_tightening_roundtrip() {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let mut total_tightened = 0usize;
    let mut checked = 0usize;

    for _ in 0..300 {
        let n = 6usize;
        // Strictly convex diagonal P and random linear cost.
        let p_lower: Vec<Triplet> = (0..n)
            .map(|i| Triplet::new(i, i, rng.unif(0.5, 3.0)))
            .collect();
        let c: Vec<f64> = (0..n).map(|_| rng.unif(-8.0, 8.0)).collect();
        let lb = vec![0.0; n];
        let ub = vec![10.0; n];

        // Disjoint constraint groups: a singleton on x0, x1; a pair on
        // (x2,x3); a pair on (x4,x5). Coefficients/RHS random but in a
        // range that often (not always) tightens.
        let mut g = Vec::new();
        let mut h = Vec::new();
        // singletons
        g.push(Triplet::new(0, 0, rng.unif(1.0, 3.0)));
        h.push(rng.unif(1.0, 12.0));
        g.push(Triplet::new(1, 1, rng.unif(1.0, 3.0)));
        h.push(rng.unif(1.0, 12.0));
        // pair (x2, x3)
        let s = if rng.unif(0.0, 1.0) < 0.5 { 1.0 } else { -1.0 };
        g.push(Triplet::new(2, 2, rng.unif(1.0, 2.0)));
        g.push(Triplet::new(2, 3, s * rng.unif(1.0, 2.0)));
        h.push(rng.unif(-3.0, 8.0));
        // pair (x4, x5)
        g.push(Triplet::new(3, 4, rng.unif(1.0, 2.0)));
        g.push(Triplet::new(3, 5, rng.unif(1.0, 2.0)));
        h.push(rng.unif(2.0, 14.0));

        let prob = QpProblem {
            n,
            p_lower,
            c,
            a: vec![],
            b: vec![],
            g,
            h,
            lb,
            ub,
        };

        // Skip presolve-detected infeasible instances (random RHS can make
        // a group infeasible); the direct solve agrees by status.
        match presolve(&prob) {
            PresolveOutcome::Infeasible => {
                assert_eq!(direct(&prob).status, QpStatus::PrimalInfeasible);
                continue;
            }
            PresolveOutcome::Unbounded => continue,
            PresolveOutcome::Reduced(ps) => total_tightened += ps.stats().tightened_bounds,
        }

        let sol = with_presolve(&prob);
        let d = direct(&prob);
        if sol.status != QpStatus::Optimal || d.status != QpStatus::Optimal {
            continue;
        }
        assert_kkt(&prob, &sol, 1e-4);
        for i in 0..n {
            assert!(
                (sol.x[i] - d.x[i]).abs() < 1e-4,
                "primal x[{i}]: presolve {} vs direct {}",
                sol.x[i],
                d.x[i]
            );
        }
        assert!(
            (sol.obj - d.obj).abs() < 1e-4,
            "obj {} vs {}",
            sol.obj,
            d.obj
        );
        checked += 1;
    }

    assert!(checked > 50, "too few optimal instances checked: {checked}");
    assert!(total_tightened > 0, "no bound tightening exercised");
}

/// Randomized sweep with **overlapping** constraints (consecutive rows
/// share a variable, forming a chain). Here tightening sources overlap, so
/// no single round can use them all — the fixpoint must resolve them across
/// rounds while keeping the re-attributed duals correct. KKT-validated.
#[test]
fn randomized_overlapping_tightening_roundtrip() {
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    let mut checked = 0usize;
    let mut total_tightened = 0usize;

    for _ in 0..300 {
        let n = 6usize;
        let p_lower: Vec<Triplet> = (0..n)
            .map(|i| Triplet::new(i, i, rng.unif(0.5, 3.0)))
            .collect();
        let c: Vec<f64> = (0..n).map(|_| rng.unif(-8.0, 8.0)).collect();

        // Chain of overlapping pair inequalities: row i couples x_i, x_{i+1}.
        let mut g = Vec::new();
        let mut h = Vec::new();
        for i in 0..n - 1 {
            let s = if rng.unif(0.0, 1.0) < 0.5 { 1.0 } else { -1.0 };
            g.push(Triplet::new(i, i, rng.unif(1.0, 2.0)));
            g.push(Triplet::new(i, i + 1, s * rng.unif(1.0, 2.0)));
            h.push(rng.unif(-2.0, 10.0));
        }

        let prob = QpProblem {
            n,
            p_lower,
            c,
            a: vec![],
            b: vec![],
            g,
            h,
            lb: vec![0.0; n],
            ub: vec![10.0; n],
        };

        match presolve(&prob) {
            PresolveOutcome::Infeasible => {
                assert_eq!(direct(&prob).status, QpStatus::PrimalInfeasible);
                continue;
            }
            PresolveOutcome::Unbounded => continue,
            PresolveOutcome::Reduced(ps) => total_tightened += ps.stats().tightened_bounds,
        }

        let sol = with_presolve(&prob);
        let d = direct(&prob);
        if sol.status != QpStatus::Optimal || d.status != QpStatus::Optimal {
            continue;
        }
        assert_kkt(&prob, &sol, 1e-4);
        for i in 0..n {
            assert!(
                (sol.x[i] - d.x[i]).abs() < 1e-4,
                "primal x[{i}]: presolve {} vs direct {}",
                sol.x[i],
                d.x[i]
            );
        }
        checked += 1;
    }

    assert!(checked > 50, "too few optimal instances: {checked}");
    assert!(total_tightened > 0, "no overlapping tightening exercised");
}
