//! gh#1181: an infeasible or unbounded verdict carries its certificate, in
//! the caller's own rows and columns, through presolve.
//!
//! Every certificate is checked here by [`check_farkas`] / [`check_recession`]
//! — written against the *original* problem data with nothing borrowed from
//! `pounce_convex::certificate` — so a certificate the module wrongly marks
//! `verified` fails the test, not just one it wrongly refuses.
//!
//! One fixture per branch the certificate can take:
//!
//! * a genuinely infeasible network LP (GALENET), with and without presolve;
//! * a genuinely unbounded LP and QP, with and without presolve;
//! * an infeasibility presolve proves by itself, so no engine ran and there is
//!   no ray to lift — the certificate has to come from the unreduced re-solve;
//! * a feasible model a hair's width from infeasible, which must come back
//!   optimal with **no** certificate.

use pounce_convex::pipeline::{solve_presolved, solve_presolved_with_ray};
use pounce_convex::{
    Certificate, CertificateSource, QpOptions, QpProblem, QpSolution, QpStatus, Triplet,
    certificate, certificate_or_resolve, solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

const INF: f64 = f64::INFINITY;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn plain(prob: &QpProblem) -> QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

/// Presolve → solve → postsolve, then the certificate with the unreduced
/// re-solve as its fallback: what `solve_qp(qp_presolve=True)` does.
fn presolved(prob: &QpProblem) -> (QpSolution, Option<Certificate>) {
    let o = QpOptions::default();
    let (sol, _, ray) = solve_presolved_with_ray(prob, &o, |p, po| solve_qp_ipm(p, po, backend));
    let lift = ray.as_ref().map(|(y, z)| (y.as_slice(), z.as_slice()));
    let cert = certificate_or_resolve(prob, &[], &sol, lift, || Some(plain(prob))).certificate;
    (sol, cert)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `(y, z, z_lb, z_ub)` proves `{Ax = b, Gx ≤ h, lb ≤ x ≤ ub}` empty:
/// `Aᵀy + Gᵀz − z_lb + z_ub = 0`, every multiplier of an inequality or bound
/// non-negative and zero on an infinite bound, and
/// `bᵀy + hᵀz − lbᵀz_lb + ubᵀz_ub < 0`.
fn check_farkas(p: &QpProblem, c: &Certificate) {
    let Certificate::PrimalInfeasible(f) = c else {
        panic!("expected a Farkas certificate, got {c:?}");
    };
    assert!(f.verified, "{f:?}");
    let n = p.n;
    let lb = |j: usize| p.lb.get(j).copied().unwrap_or(-INF);
    let ub = |j: usize| p.ub.get(j).copied().unwrap_or(INF);
    let mut r = vec![0.0; n];
    for t in &p.a {
        r[t.col] += t.val * f.y[t.row];
    }
    for t in &p.g {
        r[t.col] += t.val * f.z[t.row];
    }
    let mut value = dot(&p.b, &f.y) + dot(&p.h, &f.z);
    for j in 0..n {
        r[j] += f.z_ub[j] - f.z_lb[j];
        assert!(f.z_lb[j] >= 0.0 && f.z_ub[j] >= 0.0);
        if f.z_lb[j] > 0.0 {
            assert!(lb(j).is_finite(), "z_lb[{j}] on an infinite bound");
            value -= lb(j) * f.z_lb[j];
        }
        if f.z_ub[j] > 0.0 {
            assert!(ub(j).is_finite(), "z_ub[{j}] on an infinite bound");
            value += ub(j) * f.z_ub[j];
        }
    }
    assert!(f.z.iter().all(|&v| v >= 0.0), "z ∉ K*: {:?}", f.z);
    let resid = r.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(resid <= 1e-9, "Aᵀy + Gᵀz − z_lb + z_ub = {r:?}");
    assert!(value < -1e-6, "Farkas value {value} is not negative");
    assert!((value - f.value).abs() <= 1e-12 * value.abs().max(1.0));
}

/// `d` takes every feasible point to `−∞`: `Pd = 0`, `Ad = 0`, `Gd ≤ 0`,
/// `d ≥ 0` on a finite lower bound and `≤ 0` on a finite upper one, `cᵀd < 0`.
fn check_recession(p: &QpProblem, c: &Certificate) {
    let Certificate::DualInfeasible(r) = c else {
        panic!("expected a recession certificate, got {c:?}");
    };
    assert!(r.verified, "{r:?}");
    let d = &r.d;
    let mut pd = vec![0.0; p.n];
    for t in &p.p_lower {
        pd[t.row] += t.val * d[t.col];
        if t.row != t.col {
            pd[t.col] += t.val * d[t.row];
        }
    }
    let mut ad = vec![0.0; p.m_eq()];
    for t in &p.a {
        ad[t.row] += t.val * d[t.col];
    }
    let mut gd = vec![0.0; p.m_ineq()];
    for t in &p.g {
        gd[t.row] += t.val * d[t.col];
    }
    let tol = 1e-9;
    assert!(pd.iter().all(|v| v.abs() <= tol), "Pd = {pd:?}");
    assert!(ad.iter().all(|v| v.abs() <= tol), "Ad = {ad:?}");
    assert!(gd.iter().all(|&v| v <= tol), "Gd = {gd:?}");
    for (j, &dj) in d.iter().enumerate() {
        if p.lb.get(j).is_some_and(|v| v.is_finite()) {
            assert!(dj >= -tol, "d[{j}] = {dj} crosses a lower bound");
        }
        if p.ub.get(j).is_some_and(|v| v.is_finite()) {
            assert!(dj <= tol, "d[{j}] = {dj} crosses an upper bound");
        }
    }
    let cd = dot(&p.c, d);
    assert!(cd < -1e-6, "cᵀd = {cd} is not negative");
}

/// GALENET (netlib `infeas`), a transportation network: three supplies
/// `S1..S3 ≤ 20`, two transshipment nodes, three demands `D6 ≥ 10`,
/// `D7 ≥ 20`, `D8 ≥ 30`, and capacities on all eight arcs. Infeasible because
/// node 5 can receive at most `T25 + T35 ≤ 10 + 10 = 20` but must send
/// `T57 + T58 ≥ D8 = 30`. No objective.
///
/// Columns `T14 T24 T25 T35 T46 T47 T57 T58`.
fn galenet() -> QpProblem {
    let (t14, t24, t25, t35, t46, t47, t57, t58) = (0, 1, 2, 3, 4, 5, 6, 7);
    let g = vec![
        // Supplies, `≤`.
        Triplet::new(0, t14, 1.0),
        Triplet::new(1, t24, 1.0),
        Triplet::new(1, t25, 1.0),
        Triplet::new(2, t35, 1.0),
        // Demands, `≥` written as `−row ≤ −rhs`.
        Triplet::new(3, t46, -1.0),
        Triplet::new(4, t47, -1.0),
        Triplet::new(4, t57, -1.0),
        Triplet::new(5, t58, -1.0),
    ];
    let a = vec![
        // Node 4 and node 5: inflow − outflow = 0.
        Triplet::new(0, t14, 1.0),
        Triplet::new(0, t24, 1.0),
        Triplet::new(0, t46, -1.0),
        Triplet::new(0, t47, -1.0),
        Triplet::new(1, t25, 1.0),
        Triplet::new(1, t35, 1.0),
        Triplet::new(1, t57, -1.0),
        Triplet::new(1, t58, -1.0),
    ];
    QpProblem {
        n: 8,
        p_lower: vec![],
        c: vec![0.0; 8],
        a,
        b: vec![0.0, 0.0],
        g,
        h: vec![20.0, 20.0, 20.0, -10.0, -20.0, -30.0],
        lb: vec![0.0; 8],
        ub: vec![30.0, 20.0, 10.0, 10.0, 10.0, 2.0, 20.0, 30.0],
    }
}

#[test]
fn galenet_carries_a_farkas_certificate() {
    let p = galenet();
    let sol = plain(&p);
    assert_eq!(sol.status, QpStatus::PrimalInfeasible);
    let c = certificate(&p, &[], &sol).expect("an infeasible verdict carries a certificate");
    check_farkas(&p, &c);
    assert_eq!(c.source(), CertificateSource::Solve);
}

#[test]
fn galenet_certificate_survives_presolve() {
    let p = galenet();
    let (sol, c) = presolved(&p);
    assert_eq!(sol.status, QpStatus::PrimalInfeasible);
    check_farkas(&p, &c.expect("certificate"));
}

/// `min −x₀ − 2x₁ + x₂` over `x ≥ 0`, `x₀ − x₁ = 0`, `x₀ + x₁ − x₂ ≥ 1`,
/// `x₂ ≤ 4`: `d = (1, 1, 0)` is an exact recession direction.
fn unbounded_lp() -> QpProblem {
    QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![-1.0, -2.0, 1.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)],
        b: vec![0.0],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(0, 1, -1.0),
            Triplet::new(0, 2, 1.0),
        ],
        h: vec![-1.0],
        lb: vec![0.0; 3],
        ub: vec![INF, INF, 4.0],
    }
}

/// The LP above plus curvature on `x₂` only, which `d` does not move.
fn unbounded_qp() -> QpProblem {
    QpProblem {
        p_lower: vec![Triplet::new(2, 2, 1.0)],
        ..unbounded_lp()
    }
}

#[test]
fn unbounded_models_carry_a_recession_certificate() {
    for p in [unbounded_lp(), unbounded_qp()] {
        let sol = plain(&p);
        assert_eq!(sol.status, QpStatus::DualInfeasible);
        check_recession(&p, &certificate(&p, &[], &sol).expect("certificate"));
        let (sol, c) = presolved(&p);
        assert_eq!(sol.status, QpStatus::DualInfeasible);
        check_recession(&p, &c.expect("certificate"));
    }
}

/// `x₀ + x₁ ≥ 3` over `[0, 1]²`: the row's largest activity is 2, so
/// presolve's activity screen proves infeasibility and no engine runs. The
/// pipeline returns a zero point with the verdict — no ray — so the
/// certificate has to come from re-solving the original problem.
#[test]
fn presolve_verdict_gets_its_certificate_from_the_unreduced_problem() {
    let p = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(0, 1, -1.0)],
        h: vec![-3.0],
        lb: vec![0.0, 0.0],
        ub: vec![1.0, 1.0],
    };
    let o = QpOptions::default();
    let (sol, report) = solve_presolved(&p, &o, |q, qo| solve_qp_ipm(q, qo, backend));
    assert_eq!(report.outcome, "infeasible", "{report:?}");
    assert_eq!(sol.status, QpStatus::PrimalInfeasible);
    assert!(
        certificate(&p, &[], &sol).is_none(),
        "a presolve verdict has no ray of its own"
    );
    let c = certificate_or_resolve(&p, &[], &sol, None, || Some(plain(&p)))
        .certificate
        .expect("certificate");
    assert_eq!(c.source(), CertificateSource::UnreducedResolve);
    check_farkas(&p, &c);
}

/// A presolved solve whose lifted ray is *not* a certificate of the original
/// problem must not be passed off as one: whatever is returned, verified,
/// checks against the original data. `x₂ = 5` is a singleton row presolve
/// fixes and substitutes; the remaining rows are infeasible only together.
#[test]
fn ray_lifted_through_a_fixing_still_checks() {
    let p = QpProblem {
        n: 3,
        p_lower: vec![Triplet::new(0, 0, 1.0)],
        c: vec![0.0; 3],
        a: vec![Triplet::new(0, 2, 1.0)],
        b: vec![5.0],
        g: vec![
            // x₀ + x₁ + x₂ ≤ 6
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
            // −x₀ + x₁ ≤ −1  (x₀ ≥ 1 + x₁)
            Triplet::new(1, 0, -1.0),
            Triplet::new(1, 1, 1.0),
            // x₀ − x₁ + x₂ ≥ 7.5 → −x₀ + x₁ − x₂ ≤ −7.5
            Triplet::new(2, 0, -1.0),
            Triplet::new(2, 1, 1.0),
            Triplet::new(2, 2, -1.0),
        ],
        h: vec![6.0, -1.0, -7.5],
        lb: vec![-INF, 0.0, -INF],
        ub: vec![INF, INF, INF],
    };
    // With x₂ = 5 the last two rows read x₀ + x₁ ≤ 1 and x₀ − x₁ ≥ 2.5, so
    // x₀ ≤ 1 − x₁ ≤ 1 and x₀ ≥ 2.5 + x₁ ≥ 2.5: empty.
    let (sol, c) = presolved(&p);
    assert_eq!(sol.status, QpStatus::PrimalInfeasible);
    check_farkas(&p, &c.expect("certificate"));
}

/// Feasible, but only on a sliver of width `1e-7`: `1 − 1e-7 ≤ x₀ + x₁ ≤ 1`
/// over `x ≥ 0`. A loose certificate tolerance would call this infeasible;
/// it must come back optimal and carry no certificate, with or without
/// presolve.
#[test]
fn feasible_sliver_is_optimal_with_no_certificate() {
    let w = 1e-7;
    let p = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
        c: vec![-1.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(1, 1, -1.0),
        ],
        h: vec![1.0, -(1.0 - w)],
        lb: vec![0.0, 0.0],
        ub: vec![INF, INF],
    };
    let sol = plain(&p);
    assert!(
        matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
        "{:?}",
        sol.status
    );
    assert!(certificate(&p, &[], &sol).is_none());
    let (sol, c) = presolved(&p);
    assert!(
        matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
        "{:?}",
        sol.status
    );
    assert!(c.is_none());
}

/// The lift itself, on a model where it is the only ray that checks.
///
/// `x₀ + x₁ − x₂ ≥ 1` (row 0) against `x₀ + x₁ − x₂ ≤ −1.5` (row 1, written
/// doubled) is infeasible by itself; rows 2 and 3 and the box give presolve
/// bounds to tighten and dominated columns to fix, and
/// nothing to prove infeasible, so the IPM reaches the verdict on the reduced
/// problem. The ordinary postsolve reconstructs the removed multipliers from
/// stationarity *with* `c` and decides bound activity from the lifted `x`,
/// and the ray it returns does not check against the original rows; the
/// objective-free lift (`Presolve::postsolve_farkas`) does, so no re-solve is
/// needed — the `resolve` closure here panics if called. Found by a
/// randomized search over small boxed LPs; the NETLIB-infeas models
/// `ceria3d`, `forest6`, `klein3`, `mondou2`, `pang`, `qual`, `refinery` and
/// `vol1` are the same case at scale.
///
/// Re-searched twice for gh#1246. The original model stopped reaching the
/// case once presolve stopped handing the solver the bounds its rows imply:
/// with those sides gone, the plain postsolve's ray checks on it. That is the
/// improvement, but it left this test pinning nothing. The second model
/// stopped reaching it when convex dominance arrived. Its full catalog then
/// proved the model infeasible, the confirming re-derivation could not, and
/// the reduction it fell back to has no dominated column.
#[test]
fn objective_free_lift_is_a_certificate_where_the_plain_postsolve_is_not() {
    let p = QpProblem {
        n: 5,
        p_lower: vec![],
        c: vec![-1.0, 0.0, -1.0, 0.0, 0.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(0, 1, -1.0),
            Triplet::new(0, 2, 1.0),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(1, 2, -2.0),
            Triplet::new(2, 0, -1.0),
            Triplet::new(2, 1, 1.0),
            Triplet::new(2, 4, 2.0),
            Triplet::new(3, 3, -1.0),
        ],
        h: vec![-1.0, -3.0, 2.0, 0.0],
        lb: vec![0.0; 5],
        ub: vec![10.0, INF, INF, 10.0, 10.0],
    };
    let o = QpOptions::default();
    let (sol, report, lift) =
        solve_presolved_with_ray(&p, &o, |q, qo| solve_qp_ipm(q, qo, backend));
    assert_eq!(sol.status, QpStatus::PrimalInfeasible);
    let stats = report.stats.expect("presolve reduced the model");
    assert!(
        stats.tightened_bounds > 0 && stats.dominated_cols > 0,
        "{stats:?}"
    );
    assert!(
        !certificate(&p, &[], &sol).is_some_and(|c| c.verified()),
        "the plain postsolve's ray is not expected to check here; if it now \
         does, this fixture no longer reaches the case it pins"
    );
    let (y, z) = lift.expect("a reduced primal-infeasible solve has a lifted ray");
    let c = certificate_or_resolve(&p, &[], &sol, Some((&y, &z)), || {
        panic!("the lifted ray should have checked")
    })
    .certificate
    .expect("certificate");
    assert_eq!(c.source(), CertificateSource::Solve);
    check_farkas(&p, &c);
}

/// A bound at `±1e20` is no bound, by the engine's convention (`BOUND_INF`,
/// and `NEG_INF`/`POS_INF`'s docs), and the certificate must read it the same
/// way (gh#1181 review). Reading `−1e20` with `is_finite()` as a real bound
/// put the Farkas ray's rounding leftover on it — `value ≈ +1.9e9` — and
/// called a recession direction moving toward `−1e20` a bound crossing, so
/// both certificates came back "not verified" on correct verdicts, and
/// `qp_certificate=yes` paid for a pointless re-solve. Each model is solved
/// with the sentinel and with `±∞`; both must give the same verified result.
#[test]
fn bounds_at_the_bound_inf_sentinel_are_no_bounds() {
    for big in [1e20, INF] {
        // x₀ + x₁ ≤ 0 and x₀ + x₁ ≥ 1, x₀ ∈ [0, 10], x₁ free.
        let farkas = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![0.0, 0.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, -1.0),
                Triplet::new(1, 1, -1.0),
            ],
            h: vec![0.0, -1.0],
            lb: vec![0.0, -big],
            ub: vec![10.0, big],
        };
        let sol = plain(&farkas);
        assert_eq!(sol.status, QpStatus::PrimalInfeasible, "bound {big}");
        let c = certificate(&farkas, &[], &sol).expect("certificate");
        assert!(c.verified(), "bound {big}: {c:?}");
        let Certificate::PrimalInfeasible(f) = &c else {
            unreachable!()
        };
        assert_eq!(
            f.z_lb[1], 0.0,
            "bound {big}: a multiplier on an absent bound"
        );
        assert_eq!(
            f.z_ub[1], 0.0,
            "bound {big}: a multiplier on an absent bound"
        );

        // min x₀ s.t. x₀ − x₁ = 0, both free: unbounded along d = −(1, 1).
        let recession = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)],
            b: vec![0.0],
            g: vec![],
            h: vec![],
            lb: vec![-big, -big],
            ub: vec![big, big],
        };
        let sol = plain(&recession);
        assert_eq!(sol.status, QpStatus::DualInfeasible, "bound {big}");
        let c = certificate(&recession, &[], &sol).expect("certificate");
        let Certificate::DualInfeasible(r) = &c else {
            unreachable!()
        };
        assert!(r.verified, "bound {big}: {r:?}");
        assert_eq!(r.bound_violation, 0.0, "bound {big}");
    }
}
