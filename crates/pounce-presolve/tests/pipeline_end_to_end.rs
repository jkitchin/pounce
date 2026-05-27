//! End-to-end pipeline fuzz for PRs 2-7 of issue #53.
//!
//! Builds random linear problems with a known full-space optimum,
//! runs the full pipeline (incidence → matching → DM → components
//! → BTF → coupling → block solve → reduction frame → multiplier
//! recovery), and verifies the reconstructed solution matches the
//! analytical answer. This is PR 8 in miniature: when the
//! orchestrator gets wired in PR 8, it executes the same sequence
//! inside the TNLP wrapper.
//!
//! Problem template: `A x = b` where `A` is a random lower-triangular
//! matrix with strong diagonal (so it's always nonsingular). The
//! analytical solution is `x* = A^{-1} b`. For the multiplier check,
//! we also pick a random objective gradient `g`; then the dropped-row
//! multipliers should satisfy `A^T λ = g`, i.e., `λ = (A^T)^{-1} g`.
//!
//! Linear lower-triangular A is convenient because:
//! - HK always finds the diagonal matching.
//! - DM puts everything in `square`.
//! - BTF emits N singleton blocks in order.
//! - Each block solve is a one-iteration Newton (linear system).

#![allow(clippy::expect_used, clippy::unwrap_used)]

use pounce_common::types::Number;
use pounce_presolve::matching::hopcroft_karp;
use pounce_presolve::{
    classify_block, AuxiliaryCouplingClass, BlockEquations, BlockSolveOptions, BlockSolver,
    BlockTriangularForm, DampedNewtonSolver, DulmageMendelsohnPartition, EqualityIncidence,
    InequalityIncidence, ProbeView, ReductionFrame, SquareComponents,
};

/// Deterministic LCG-based random for reproducible fuzz.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 32
    }
    fn unit(&mut self) -> Number {
        let raw = (self.next() & 0x3fff_ffff) as Number;
        raw / (1u64 << 29) as Number - 1.0
    }
}

/// Wrap a captured `(A, b)` so the Newton solver can call it via
/// the `BlockEquations` trait.
struct LinearBlockEqs {
    a: Vec<Number>,
    b: Vec<Number>,
    n: usize,
}
impl BlockEquations for LinearBlockEqs {
    fn dim(&self) -> usize {
        self.n
    }
    fn eval(&mut self, x: &[Number], f: &mut [Number]) -> bool {
        for i in 0..self.n {
            let mut s = -self.b[i];
            for j in 0..self.n {
                s += self.a[i * self.n + j] * x[j];
            }
            f[i] = s;
        }
        true
    }
    fn jacobian(&mut self, _x: &[Number], j: &mut [Number]) -> bool {
        j.copy_from_slice(&self.a);
        true
    }
}

/// Run incidence → matching → DM → components → BTF → coupling →
/// block solve → frame → multiplier recovery on one random
/// lower-triangular linear system, and verify the reconstructed
/// (x*, λ*) matches the analytical answer.
fn one_trial(seed: u64) {
    let mut rng = Rng::new(seed);
    let n = 2 + (rng.next() % 4) as usize; // 2..=5

    // A: lower-triangular, strong diagonal.
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            a[i * n + j] = if i == j {
                2.0 + rng.unit().abs()
            } else {
                0.3 * rng.unit()
            };
        }
    }
    let b: Vec<Number> = (0..n).map(|_| rng.unit()).collect();
    // Analytical x* by forward substitution on A x = b (lower-tri).
    let mut x_star = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= a[i * n + j] * x_star[j];
        }
        x_star[i] = s / a[i * n + i];
    }
    // Pick random objective gradient g for multiplier check.
    let grad_f: Vec<Number> = (0..n).map(|_| rng.unit()).collect();
    // Analytical λ from A^T λ = g (back substitution since A^T is
    // upper-triangular).
    let mut lambda_star = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = grad_f[i];
        for j in (i + 1)..n {
            s -= a[j * n + i] * lambda_star[j];
        }
        lambda_star[i] = s / a[i * n + i];
    }

    // Build the TNLP-style probe: every row is an equality (g_l = g_u
    // = b[i]). Convert A's sparsity to (irow, jcol) triples.
    let mut jac_irow: Vec<i32> = Vec::new();
    let mut jac_jcol: Vec<i32> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if a[i * n + j] != 0.0 {
                jac_irow.push(i as i32);
                jac_jcol.push(j as i32);
            }
        }
    }
    let g_l: Vec<Number> = b.clone();
    let g_u: Vec<Number> = b.clone();
    let probe = ProbeView {
        n_vars: n,
        m_rows: n,
        jac_irow: &jac_irow,
        jac_jcol: &jac_jcol,
        jac_values: None,
        g_l: &g_l,
        g_u: &g_u,
        linearity: None,
        one_based: false,
        eq_tol: 1e-12,
        excluded_vars: None,
        excluded_rows: None,
    };

    // Stage 1: incidence + matching.
    let inc = EqualityIncidence::from_probe(&probe);
    assert_eq!(
        inc.n_eq_rows(),
        n,
        "trial {seed}: all rows should be equalities"
    );
    let matching = hopcroft_karp(&inc);
    assert_eq!(
        matching.size, n,
        "trial {seed}: lower-tri matrix is nonsingular → perfect matching"
    );

    // Stage 2: DM partition.
    let dm = DulmageMendelsohnPartition::from_matching(&inc, &matching);
    assert_eq!(
        dm.square_rows.len(),
        n,
        "trial {seed}: nonsingular square → all square"
    );
    assert!(dm.over_rows.is_empty() && dm.under_rows.is_empty());

    // Stage 3: components + BTF.
    let comps = SquareComponents::of_square_part(&inc, &matching, &dm);
    assert!(!comps.components.is_empty());

    // Stage 4: build inequality incidence (will be empty for this
    // problem) and verify all blocks classify as PureEquality.
    let ineq = InequalityIncidence::from_probe(&probe);
    assert_eq!(ineq.n_ineq_rows(), 0);

    // Stage 5: walk each component's BTF, solve each block via
    // Newton, accumulate (fixed_vars, fixed_values, dropped_rows).
    let mut all_fixed_vars: Vec<usize> = Vec::new();
    let mut all_fixed_values: Vec<Number> = Vec::new();
    let mut all_dropped_rows: Vec<usize> = Vec::new();
    let mut x_running = vec![0.0; n]; // Working solution, filled as blocks solve.

    for comp in &comps.components {
        let btf = BlockTriangularForm::of_component(&inc, &matching, comp);
        for block in &btf.blocks {
            assert_eq!(
                classify_block(block, &ineq, &Default::default()),
                AuxiliaryCouplingClass::PureEquality,
                "trial {seed}: block should be pure equality"
            );
            let k = block.eq_rows.len();
            // Extract the (block rows × block cols) submatrix and a
            // RHS adjusted for already-solved variables.
            let mut a_block = vec![0.0; k * k];
            let mut b_block = vec![0.0; k];
            for (ii, &r) in block.eq_rows.iter().enumerate() {
                let mut residual_from_earlier = b[r];
                for j in 0..n {
                    if block.cols.contains(&j) {
                        let jj = block.cols.iter().position(|c| *c == j).unwrap();
                        a_block[ii * k + jj] = a[r * n + j];
                    } else {
                        residual_from_earlier -= a[r * n + j] * x_running[j];
                    }
                }
                b_block[ii] = residual_from_earlier;
            }

            let mut eqs = LinearBlockEqs {
                a: a_block,
                b: b_block,
                n: k,
            };
            let opt = BlockSolveOptions::default();
            let mut solver = DampedNewtonSolver;
            let out = solver
                .solve(&vec![0.0; k], &mut eqs, &opt)
                .unwrap_or_else(|e| panic!("trial {seed}: block solve {e:?}"));

            for (ii, &c) in block.cols.iter().enumerate() {
                x_running[c] = out.x[ii];
            }
            // Record for the frame.
            for &r in &block.eq_rows {
                all_dropped_rows.push(r);
            }
            for (ii, &c) in block.cols.iter().enumerate() {
                all_fixed_vars.push(c);
                all_fixed_values.push(out.x[ii]);
            }
        }
    }

    // Sort & sanity-check.
    let mut order: Vec<usize> = (0..all_fixed_vars.len()).collect();
    order.sort_by_key(|&i| all_fixed_vars[i]);
    let fixed_vars_sorted: Vec<usize> = order.iter().map(|&i| all_fixed_vars[i]).collect();
    let fixed_values_sorted: Vec<Number> = order.iter().map(|&i| all_fixed_values[i]).collect();
    let mut dropped_rows_sorted = all_dropped_rows.clone();
    dropped_rows_sorted.sort_unstable();

    // Stage 6: verify the running x matches the analytical x_star.
    let mut max_err: Number = 0.0;
    for i in 0..n {
        max_err = max_err.max((x_running[i] - x_star[i]).abs());
    }
    assert!(
        max_err < 1e-9,
        "trial {seed}: forward solve disagreement {max_err:.3e}"
    );

    // Stage 7: build the reduction frame, recover the dropped-row
    // multipliers, check against the analytical λ*.
    let frame = ReductionFrame::new(
        n,
        n,
        fixed_vars_sorted.clone(),
        fixed_values_sorted,
        dropped_rows_sorted.clone(),
    );
    // For this problem ALL rows are dropped → lambda_given is zero.
    let lambda_given = vec![0.0; n];
    let lam_dropped = frame
        .recover_dropped_multipliers(&grad_f, &a, &lambda_given)
        .unwrap_or_else(|e| panic!("trial {seed}: recover {e:?}"));

    // Compare to analytical λ at the dropped indices.
    for (idx, &r) in frame.dropped_rows.iter().enumerate() {
        let expected = lambda_star[r];
        let got = lam_dropped[idx];
        assert!(
            (expected - got).abs() < 1e-9,
            "trial {seed}: λ[{r}] expected {expected:.6}, got {got:.6}"
        );
    }

    // Final full-space KKT residual check.
    let mut lambda_full = vec![0.0; n];
    for (idx, &r) in frame.dropped_rows.iter().enumerate() {
        lambda_full[r] = lam_dropped[idx];
    }
    let mut max_kkt: Number = 0.0;
    for i in 0..n {
        let mut s = grad_f[i];
        for r in 0..n {
            s -= a[r * n + i] * lambda_full[r];
        }
        max_kkt = max_kkt.max(s.abs());
    }
    assert!(max_kkt < 1e-9, "trial {seed}: KKT residual {max_kkt:.3e}");
}

#[test]
fn end_to_end_pipeline_fuzz() {
    let seeds: [u64; 20] = [
        0x1111_1111_1111_1111,
        0x2222_2222_2222_2222,
        0x3333_3333_3333_3333,
        0x4444_4444_4444_4444,
        0x5555_5555_5555_5555,
        0x6666_6666_6666_6666,
        0x7777_7777_7777_7777,
        0x8888_8888_8888_8888,
        0x9999_9999_9999_9999,
        0xaaaa_aaaa_aaaa_aaaa,
        0xbbbb_bbbb_bbbb_bbbb,
        0xcccc_cccc_cccc_cccc,
        0xdddd_dddd_dddd_dddd,
        0xeeee_eeee_eeee_eeee,
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0xdead_beef_dead_beef,
        0xcafe_b00b_cafe_b00b,
        0xfeed_face_feed_face,
        0xbaad_f00d_baad_f00d,
    ];
    for &s in &seeds {
        one_trial(s);
    }
}
