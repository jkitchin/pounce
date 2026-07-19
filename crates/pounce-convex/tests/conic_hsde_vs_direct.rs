//! Randomized differential test: the HSDE driver against the direct one, over
//! second-order and semidefinite cones.
//!
//! The gh #218 centering fallback lives in the **HSDE** driver (`hsde.rs`),
//! which serves the symmetric cones — orthant, second-order, PSD. Orthant
//! coverage is strong already (the NETLIB LP and Maros–Mészáros QP benchmark
//! suites run through it), and the exponential/power cones are untouched
//! because they route to the separate non-symmetric driver. That leaves SOC and
//! PSD, whose only coverage was a handful of closed-form instances.
//!
//! The direct symmetric driver (`use_hsde: false`, in `ipm.rs`) solves the same
//! cones by an independent path that the change does not touch, which makes it
//! a genuine reference rather than a self-comparison: the two drivers agree on
//! the optimal value of any problem with a bounded optimum and a Slater point,
//! so a disagreement is a real defect in one of them.
//!
//! Instances are generated so that both properties hold by construction — a
//! strictly feasible point is planted, and every variable is boxed, so the
//! feasible set is compact and nonempty. The RNG is a fixed-seed xorshift, so
//! failures are reproducible and the suite is deterministic.

use pounce_convex::{ConeSpec, QpOptions, QpProblem, QpStatus, Triplet, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// xorshift64*, so the suite is deterministic and a failure is reproducible
/// from its seed alone.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `[-1, 1)`.
    fn signed(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
    }
}

const BOX: f64 = 10.0;

/// A random conic program with a planted strictly feasible point and a compact
/// feasible set.
///
/// Every variable carries `|xⱼ| ≤ BOX` as a pair of orthant rows, so the
/// feasible set is bounded and the optimum is attained. The remaining cone
/// blocks get their `h` set to `G·x₀ + s₀` for a random interior `s₀`, which
/// makes `x₀` a Slater point. Both properties together are exactly the
/// hypotheses under which the two drivers must agree.
fn random_instance(rng: &mut Rng, n: usize, blocks: &[ConeSpec], m_eq: usize) -> QpProblem {
    let x0: Vec<f64> = (0..n).map(|_| rng.signed() * 0.5 * BOX).collect();

    // Equalities through the planted point: A x = A x₀.
    let mut a = Vec::new();
    let mut b = vec![0.0; m_eq];
    for r in 0..m_eq {
        for (j, x0j) in x0.iter().enumerate() {
            let v = rng.signed();
            if v.abs() > 0.3 {
                a.push(Triplet::new(r, j, v));
                b[r] += v * x0j;
            }
        }
    }

    let mut g = Vec::new();
    let mut h = Vec::new();
    let mut cones = Vec::new();

    // Variable box as two orthant rows per variable: BOX ∓ xⱼ ≥ 0.
    for j in 0..n {
        let r = h.len();
        g.push(Triplet::new(r, j, 1.0));
        h.push(BOX);
        let r = h.len();
        g.push(Triplet::new(r, j, -1.0));
        h.push(BOX);
    }
    cones.push(ConeSpec::Nonneg(2 * n));

    for spec in blocks {
        let dim = match spec {
            ConeSpec::Nonneg(k) | ConeSpec::SecondOrder(k) => *k,
            ConeSpec::Psd(k) => k * (k + 1) / 2,
            // Exponential/power cones route to the non-symmetric driver, which
            // this change does not touch, so they are out of scope here.
            other => panic!("unsupported cone in this generator: {other:?}"),
        };
        let row0 = h.len();
        // Random sparse G rows, then h = G·x₀ + s₀ with s₀ interior.
        let mut gx0 = vec![0.0; dim];
        for (i, gx) in gx0.iter_mut().enumerate() {
            for (j, x0j) in x0.iter().enumerate() {
                let v = rng.signed();
                if v.abs() > 0.5 {
                    g.push(Triplet::new(row0 + i, j, v));
                    *gx += v * x0j;
                }
            }
        }
        let s0 = interior_point(rng, spec, dim);
        for i in 0..dim {
            h.push(gx0[i] + s0[i]);
        }
        cones.push(spec.clone());
    }

    QpProblem {
        n,
        p_lower: Vec::new(),
        c: (0..n).map(|_| rng.signed()).collect(),
        a,
        b,
        g,
        h,
        lb: Vec::new(),
        ub: Vec::new(),
    }
}

/// A point strictly inside `spec` — the planted slack `s₀`, which is what makes
/// the generated instance Slater-feasible.
fn interior_point(rng: &mut Rng, spec: &ConeSpec, dim: usize) -> Vec<f64> {
    match spec {
        ConeSpec::Nonneg(_) => (0..dim).map(|_| 1.0 + rng.signed().abs()).collect(),
        ConeSpec::SecondOrder(_) => {
            // (t, v) with t > ‖v‖: build v, then set t above its norm.
            let mut s = vec![0.0; dim];
            let mut nrm = 0.0;
            for si in s.iter_mut().skip(1) {
                *si = rng.signed();
                nrm += *si * *si;
            }
            s[0] = nrm.sqrt() + 1.0 + rng.signed().abs();
            s
        }
        ConeSpec::Psd(k) => {
            // svec of `L Lᵀ + 2I`, which is positive definite by construction.
            let k = &(*k);
            let l: Vec<f64> = (0..k * k).map(|_| rng.signed()).collect();
            let mut m = vec![0.0; k * k];
            for i in 0..*k {
                for j in 0..*k {
                    let mut acc = 0.0;
                    for p in 0..*k {
                        acc += l[i * k + p] * l[j * k + p];
                    }
                    m[i * k + j] = acc + if i == j { 2.0 } else { 0.0 };
                }
            }
            // svec: lower triangle column by column, off-diagonals × √2.
            let r2 = std::f64::consts::SQRT_2;
            let mut s = Vec::with_capacity(dim);
            for j in 0..*k {
                for i in j..*k {
                    s.push(if i == j {
                        m[i * k + j]
                    } else {
                        r2 * m[i * k + j]
                    });
                }
            }
            s
        }
        other => panic!("unsupported cone in this generator: {other:?}"),
    }
}

fn solve(prob: &QpProblem, cones: &[ConeSpec], use_hsde: bool) -> pounce_convex::QpSolution {
    let opts = QpOptions {
        use_hsde,
        max_iter: 500,
        ..QpOptions::default()
    };
    solve_socp_ipm(prob, cones, &opts, backend)
}

/// Run `count` random instances of the given shape through both drivers and
/// require they agree on the optimal value.
fn agree_on(name: &str, seed: u64, count: usize, mut shape: impl FnMut(&mut Rng) -> Vec<ConeSpec>) {
    let mut rng = Rng(seed);
    let (mut compared, mut hsde_only, mut direct_only) = (0usize, 0usize, 0usize);
    for case in 0..count {
        let blocks = shape(&mut rng);
        let n = rng.range(3, 8);
        let m_eq = rng.range(0, 2);
        let prob = random_instance(&mut rng, n, &blocks, m_eq);
        let mut cones = vec![ConeSpec::Nonneg(2 * n)];
        cones.extend(blocks.iter().cloned());

        let a = solve(&prob, &cones, true);
        let b = solve(&prob, &cones, false);

        match (a.status, b.status) {
            (QpStatus::Optimal, QpStatus::Optimal) => {
                // Neither driver may claim optimality without a usable answer.
                // This used to hold for HSDE only — the direct driver returned
                // `Optimal` with `obj = NaN` on some PSD instances (gh #222),
                // and the comparison had to skip those. Since that fix the
                // guarantee is symmetric, so assert it on both.
                assert!(
                    a.obj.is_finite(),
                    "{name} case {case}: HSDE reported Optimal with a non-finite objective"
                );
                assert!(
                    b.obj.is_finite(),
                    "{name} case {case}: direct driver reported Optimal with a non-finite objective"
                );
                compared += 1;
                let scale = 1.0_f64.max(a.obj.abs()).max(b.obj.abs());
                assert!(
                    (a.obj - b.obj).abs() <= 1e-6 * scale,
                    "{name} case {case}: HSDE obj {} vs direct obj {} (n={n}, m_eq={m_eq}, cones={cones:?})",
                    a.obj,
                    b.obj
                );
            }
            // A driver that solves one the other cannot is an improvement, not
            // a regression — but count them so a shape that silently stopped
            // solving anything cannot masquerade as agreement.
            (QpStatus::Optimal, _) => hsde_only += 1,
            (_, QpStatus::Optimal) => direct_only += 1,
            _ => {}
        }
    }
    // Guard the premise: if the generator drifted into producing problems
    // neither driver solves, every assertion above would vacuously pass.
    assert!(
        compared * 4 >= count * 3,
        "{name}: only {compared}/{count} instances were solved by both drivers \
         (hsde-only {hsde_only}, direct-only {direct_only}) — too few to be evidence"
    );
    println!(
        "{name}: {compared}/{count} compared, hsde-only {hsde_only}, direct-only {direct_only}"
    );
}

#[test]
fn second_order_cones_agree_across_drivers() {
    agree_on("soc", 0x5EED_1234, 120, |rng| {
        let k = rng.range(1, 3);
        (0..k)
            .map(|_| ConeSpec::SecondOrder(rng.range(2, 6)))
            .collect()
    });
}

#[test]
fn psd_cones_agree_across_drivers() {
    agree_on("psd", 0xBEEF_0001, 80, |rng| {
        vec![ConeSpec::Psd(rng.range(2, 5))]
    });
}

#[test]
fn mixed_soc_and_psd_cones_agree_across_drivers() {
    agree_on("mixed", 0xC0FFEE_02, 80, |rng| {
        vec![
            ConeSpec::SecondOrder(rng.range(2, 5)),
            ConeSpec::Psd(rng.range(2, 4)),
        ]
    });
}

/// The degenerate tier: a planted slack sitting *on* the cone boundary rather
/// than strictly inside it, which is the regime the gh #218 centering fallback
/// exists for. Slater no longer holds, so the two drivers may legitimately
/// disagree on which optimal point they return — but when both converge they
/// must still agree on the optimal *value*.
#[test]
fn boundary_hugging_instances_agree_across_drivers() {
    let mut rng = Rng(0xDEAD_BEEF);
    let mut compared = 0usize;
    for case in 0..80 {
        let n = rng.range(3, 6);
        let blocks = vec![ConeSpec::Psd(rng.range(2, 4))];
        let m_eq = rng.range(0, 1);
        let mut prob = random_instance(&mut rng, n, &blocks, m_eq);
        // Pull the planted PSD slack onto the boundary: drop `h` by the 2I that
        // `interior_point` added, leaving a singular (rank-deficient) slack.
        let base = 2 * n;
        if let ConeSpec::Psd(k) = blocks[0] {
            let mut idx = base;
            for j in 0..k {
                for i in j..k {
                    if i == j {
                        prob.h[idx] -= 2.0;
                    }
                    idx += 1;
                }
            }
        }
        let mut cones = vec![ConeSpec::Nonneg(2 * n)];
        cones.extend(blocks.iter().cloned());

        let a = solve(&prob, &cones, true);
        let b = solve(&prob, &cones, false);
        if a.status == QpStatus::Optimal && b.status == QpStatus::Optimal {
            assert!(
                a.obj.is_finite() && b.obj.is_finite(),
                "degenerate case {case}: a driver reported Optimal with a non-finite objective"
            );
            compared += 1;
            let scale = 1.0_f64.max(a.obj.abs()).max(b.obj.abs());
            assert!(
                (a.obj - b.obj).abs() <= 1e-5 * scale,
                "degenerate case {case}: HSDE obj {} vs direct obj {}",
                a.obj,
                b.obj
            );
        }
    }
    assert!(
        compared >= 20,
        "only {compared}/80 degenerate instances solved by both — too few to be evidence"
    );
    println!("degenerate: {compared}/80 compared");
}

/// The gh #222 instance, verbatim.
///
/// The direct driver diverges here — which is expected of it on a degenerate
/// face, and precisely why the HSDE embedding is the default (see `sos_opts`).
/// The defect was that it reported the divergence as `Optimal`, handing back
/// `x = [NaN, NaN]` under a status that says the answer is usable.
///
/// The mechanism was `inf_norm`: `f64::max` ignores `NaN`, so the ∞-norm of an
/// all-`NaN` iterate came out `0.0` and `res < tol` passed. Both the norm and
/// an explicit exit guard now prevent that.
#[test]
fn a_diverged_solve_is_never_reported_as_optimal() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![-0.4587520836280501, -0.28511097640465644],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(2, 1, 1.0),
            Triplet::new(3, 1, -1.0),
            Triplet::new(4, 1, 0.9144362258975529),
            Triplet::new(5, 0, 0.5271709449981303),
            Triplet::new(5, 1, -0.6408278339462257),
            Triplet::new(6, 0, -0.9444454430308236),
            Triplet::new(6, 1, -0.9246001786463665),
            Triplet::new(8, 0, -0.7591854473470727),
            Triplet::new(8, 1, -0.533518750473565),
            Triplet::new(9, 0, -0.7290269156600226),
            Triplet::new(9, 1, 0.6420378359547312),
        ],
        h: vec![
            10.0,
            10.0,
            10.0,
            10.0,
            2.3063630770492667,
            2.5412241108987312,
            -2.3996656180605522,
            2.6724192569915712,
            -2.0785098897371794,
            -0.09460366494727701,
        ],
        lb: vec![],
        ub: vec![],
    };
    let cones = [ConeSpec::Nonneg(4), ConeSpec::Psd(3)];

    let direct = solve(&prob, &cones, false);
    assert!(
        !matches!(
            direct.status,
            QpStatus::Optimal | QpStatus::OptimalInaccurate
        ),
        "direct driver claimed {:?} with x = {:?}",
        direct.status,
        direct.x
    );

    // The instance itself is solvable, and HSDE solves it — so the honest
    // failure above is a property of that driver, not of the problem. Pin the
    // answer so a future change cannot quietly break the case that works.
    let hsde = solve(&prob, &cones, true);
    assert_eq!(hsde.status, QpStatus::Optimal, "{:?}", hsde.status);
    assert!(
        (hsde.obj - -3.0311688992249475).abs() < 1e-9,
        "{}",
        hsde.obj
    );
    assert!(hsde.x.iter().all(|v| v.is_finite()));
}
