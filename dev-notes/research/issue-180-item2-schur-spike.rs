//! pounce#180 item-2 Phase-0 spike.  [REFERENCE ARTIFACT — not a workspace member]
//!
//! Standalone reproducer. To run: drop this into a fresh crate with
//! `feral = "0.13.0"` as the only dependency and `cargo run --release`.
//! Results are summarised in `issue-180-item2-schur-kkt-scope.md` § Phase-0.
//!
//! Question 1 (correctness): does a block Schur backsolve reconstruct the
//!   exact KKT solution that a monolithic factorization gives?
//! Question 2 (inertia): does inertia(A_FF) + inertia(S) — Sylvester / Haynsworth
//!   additivity — equal the true inertia of the full system?
//! Question 3 (the dense trap): where does cost live as n grows? We must confirm
//!   the SPARSE dimension (n_F, the eliminated block) stays ~linear and the only
//!   n_schur-dependent cost (dense S: O(n_s^2) store, O(n_s^3) factor, plus
//!   n_s backsolves to form it) is CONFINED to the intended-small Schur block.
//!
//! Design under test = "self-formed Schur": factor A_FF standalone (feral
//! `Solver::factor` — proven), form S = A_SS - A_FS^T A_FF^{-1} A_FS via n_schur
//! backsolves, factor S densely. Uses ONLY feral's public factor/solve; does NOT
//! rely on `factorize_multifrontal_with_schur`'s un-eliminated partial factors
//! being solvable. We ALSO cross-check S against feral's native Schur extraction.

use std::time::Instant;

use feral::inertia::Inertia;
use feral::scaling::ScalingStrategy;
use feral::sparse::csc::CscMatrix;
use feral::symbolic::{symbolic_factorize_with_schur, SupernodeParams};
use feral::{factorize_multifrontal_with_schur, NumericParams, Solver};

/// A KKT-shaped symmetric-indefinite test matrix, held as lower-triangle
/// pieces so we can assemble the full system OR the F/S blocks on demand.
///
/// F = indices `0..n_f` (the "eliminated" / block-triangular submatrix):
///   tridiagonal SPD (diag 2+something, off-diag -1) → sparse, banded,
///   near-linear to factor; all-positive inertia.
/// S = indices `n_f..n_f+n_s` (the Schur tail): `A_SS` negative-definite-ish,
///   coupled to F by a sparse `A_FS` (a few entries per Schur column).
/// The full matrix then has inertia `(n_f, n_s, 0)` — a clean Sylvester target.
struct Kkt {
    n_f: usize,
    n_s: usize,
    /// A_FF lower-triangle triplets, indices in `0..n_f`.
    ff: Vec<(usize, usize, f64)>,
    /// Coupling entries, stored as A_SF: `(schur s in 0..n_s, f in 0..n_f, val)`.
    /// By symmetry these are also A_FS[f, s]. ~`coupling_deg` per Schur column.
    coupling: Vec<(usize, usize, f64)>,
    /// A_SS lower-triangle triplets, indices in `0..n_s`.
    ss: Vec<(usize, usize, f64)>,
}

impl Kkt {
    fn build(n_f: usize, n_s: usize, coupling_deg: usize) -> Self {
        Self::build_ex(n_f, n_s, coupling_deg, false)
    }

    /// `f_indef=true` makes the eliminated block A_FF *indefinite* (first half
    /// +diag, second half −diag) — a diagonally-dominant, nonsingular saddle,
    /// mimicking a real KKT eliminated block that contains constraint rows. The
    /// point is to confirm Sylvester additivity when BOTH A_FF and S contribute
    /// negative eigenvalues, not just when A_FF is SPD.
    fn build_ex(n_f: usize, n_s: usize, coupling_deg: usize, f_indef: bool) -> Self {
        let mut ff = Vec::with_capacity(2 * n_f);
        for i in 0..n_f {
            let d = if f_indef && i >= n_f / 2 { -4.0 } else { 4.0 };
            ff.push((i, i, d));
            if i > 0 {
                ff.push((i, i - 1, -1.0));
            }
        }
        // Coupling: spread each Schur column across `coupling_deg` F-rows,
        // deterministically (no RNG — vary the stride by index).
        let mut coupling = Vec::with_capacity(n_s * coupling_deg.max(1));
        if n_f > 0 {
            for s in 0..n_s {
                for k in 0..coupling_deg {
                    let f = ((s * 7 + k * 101 + 3) * 2_654_435_761usize) % n_f;
                    coupling.push((s, f, 0.5 + 0.1 * (k as f64)));
                }
            }
        }
        // A_SS: negative diagonal (drives the Schur complement negative-def).
        let mut ss = Vec::with_capacity(n_s);
        for i in 0..n_s {
            ss.push((i, i, -1.0));
        }
        Kkt { n_f, n_s, ff, coupling, ss }
    }

    fn n(&self) -> usize {
        self.n_f + self.n_s
    }

    /// Full lower-triangle triplets of M (F first, S as the tail).
    fn full_lower(&self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for &(i, j, x) in &self.ff {
            r.push(i);
            c.push(j);
            v.push(x);
        }
        // Coupling lands as (row = n_f + s, col = f) — strictly lower since
        // the Schur tail has the largest indices.
        for &(s, f, x) in &self.coupling {
            r.push(self.n_f + s);
            c.push(f);
            v.push(x);
        }
        for &(i, j, x) in &self.ss {
            r.push(self.n_f + i);
            c.push(self.n_f + j);
            v.push(x);
        }
        (r, c, v)
    }

    fn ff_lower(&self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut r = Vec::new();
        let mut c = Vec::new();
        let mut v = Vec::new();
        for &(i, j, x) in &self.ff {
            r.push(i);
            c.push(j);
            v.push(x);
        }
        (r, c, v)
    }
}

fn factor_identity_scaled(csc: &CscMatrix) -> (Solver, Inertia, usize, f64) {
    let mut s = Solver::new().with_scaling(ScalingStrategy::Identity);
    let t = Instant::now();
    let status = s.factor(csc, None);
    let dt = t.elapsed().as_secs_f64();
    match status {
        feral::numeric::solver::FactorStatus::Success => {}
        other => panic!("factor failed: {other:?}"),
    }
    let inertia = s.inertia().cloned().unwrap_or(Inertia {
        positive: 0,
        negative: s.num_negative_eigenvalues(),
        zero: 0,
    });
    let nnz_l = s.last_factor_stats().map(|st| st.nnz_l).unwrap_or(0);
    (s, inertia, nnz_l, dt)
}

/// Dense apply pieces used by the block backsolve, all O(nnz(coupling)).
fn apply_afs(k: &Kkt, x_s: &[f64], out_f: &mut [f64]) {
    // out_f += A_FS x_s  (A_FS[f,s] = coupling(s,f))
    for &(s, f, v) in &k.coupling {
        out_f[f] += v * x_s[s];
    }
}
fn apply_asf(k: &Kkt, x_f: &[f64], out_s: &mut [f64]) {
    // out_s += A_SF x_f  (A_SF[s,f] = coupling(s,f))
    for &(s, f, v) in &k.coupling {
        out_s[s] += v * x_f[f];
    }
}

struct SchurResult {
    x: Vec<f64>,
    inertia: Inertia,
    // timing / size instrumentation
    t_factor_ff: f64,
    t_form_s: f64,
    t_factor_s: f64,
    t_backsolve: f64,
    nnz_l_ff: usize,
    w_store: usize, // n_f * n_s intermediate (dense columns of A_FF^{-1} A_FS)
    native_s_matches: Option<bool>,
}

/// The design under test.
fn schur_solve(k: &Kkt, b: &[f64]) -> SchurResult {
    let n_f = k.n_f;
    let n_s = k.n_s;

    // 1. Factor A_FF (sparse, the big block).
    let (rf, cf, vf) = k.ff_lower();
    let ff_csc = CscMatrix::from_triplets(n_f, &rf, &cf, &vf).unwrap();
    let (ff, inertia_ff, nnz_l_ff, t_factor_ff) = factor_identity_scaled(&ff_csc);

    // 2. Form S = A_SS - A_FS^T A_FF^{-1} A_FS.
    let t = Instant::now();
    //   Build A_FS as a dense n_f x n_s column-major buffer (n_s solves).
    let mut afs = vec![0.0f64; n_f * n_s];
    for &(s, f, v) in &k.coupling {
        afs[f + s * n_f] = v; // column s, row f
    }
    let w = ff.solve_many(&afs, n_s).unwrap(); // W = A_FF^{-1} A_FS, n_f x n_s
    //   S = A_SS - A_FS^T W  (n_s x n_s, dense — CONFINED to n_s).
    let mut s_dense = vec![0.0f64; n_s * n_s];
    for &(i, j, x) in &k.ss {
        s_dense[i + j * n_s] += x;
    }
    // A_FS^T W: for each coupling (s,f,v): row s of A_FS^T; subtract v*W[f,:].
    for &(s, f, v) in &k.coupling {
        for j in 0..n_s {
            s_dense[s + j * n_s] -= v * w[f + j * n_f];
        }
    }
    let t_form_s = t.elapsed().as_secs_f64();

    // 3. Factor S densely (as a small sparse matrix via feral — symmetric
    //    indefinite Bunch-Kaufman, gives its inertia). Lower triangle only.
    let t = Instant::now();
    let (mut sr, mut sc, mut sv) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n_s {
        for i in j..n_s {
            let val = s_dense[i + j * n_s];
            if val != 0.0 || i == j {
                sr.push(i);
                sc.push(j);
                sv.push(val);
            }
        }
    }
    let s_csc = CscMatrix::from_triplets(n_s, &sr, &sc, &sv).unwrap();
    let (s_solver, inertia_s, _n, _feral_dt) = factor_identity_scaled(&s_csc);
    let t_factor_s = t.elapsed().as_secs_f64();

    // 4. Block backsolve for M x = b.
    let t = Instant::now();
    let b_f = &b[..n_f];
    let b_s = &b[n_f..];
    //   u = A_FF^{-1} b_F
    let u = ff.solve(b_f).unwrap();
    //   r_S = b_S - A_SF u
    let mut r_s = b_s.to_vec();
    let mut asf_u = vec![0.0; n_s];
    apply_asf(k, &u, &mut asf_u);
    for i in 0..n_s {
        r_s[i] -= asf_u[i];
    }
    //   x_S = S^{-1} r_S
    let x_s = s_solver.solve(&r_s).unwrap();
    //   x_F = A_FF^{-1} (b_F - A_FS x_S)
    let mut rhs_f = b_f.to_vec();
    let mut afs_xs = vec![0.0; n_f];
    apply_afs(k, &x_s, &mut afs_xs);
    for f in 0..n_f {
        rhs_f[f] -= afs_xs[f];
    }
    let x_f = ff.solve(&rhs_f).unwrap();
    let t_backsolve = t.elapsed().as_secs_f64();

    let mut x = Vec::with_capacity(n_f + n_s);
    x.extend_from_slice(&x_f);
    x.extend_from_slice(&x_s);

    let inertia = Inertia {
        positive: inertia_ff.positive + inertia_s.positive,
        negative: inertia_ff.negative + inertia_s.negative,
        zero: inertia_ff.zero + inertia_s.zero,
    };

    // Cross-check S against feral's NATIVE Schur extraction (Design A), where
    // the F3.2b single-supernode constraint permits it.
    let native_s_matches = native_schur_check(k, &s_dense);

    SchurResult {
        x,
        inertia,
        t_factor_ff,
        t_form_s,
        t_factor_s,
        t_backsolve,
        nnz_l_ff,
        w_store: n_f * n_s,
        native_s_matches,
    }
}

/// Feed the full lower-triangle matrix to feral's own Schur factorization and
/// compare its dense S to ours. Returns None if feral rejects the partition
/// (e.g. the F3.2b single-supernode scope limit) — which is itself a finding.
fn native_schur_check(k: &Kkt, our_s: &[f64]) -> Option<bool> {
    let (r, c, v) = k.full_lower();
    let m = CscMatrix::from_triplets(k.n(), &r, &c, &v).ok()?;
    let schur_indices: Vec<usize> = (k.n_f..k.n()).collect();
    let snode = SupernodeParams::default();
    let sym = symbolic_factorize_with_schur(&m, &snode, &schur_indices).ok()?;
    let params = NumericParams {
        scaling: ScalingStrategy::Identity,
        ..NumericParams::default()
    };
    let (_factors, _inertia_ff, schur_block) =
        factorize_multifrontal_with_schur(&m, &sym, &params).ok()?;
    // Compare diagonals (both symmetric); tolerate ordering by comparing the
    // symmetric entrywise max-abs difference over the n_s x n_s block.
    let n_s = k.n_s;
    if schur_block.dim != n_s {
        return Some(false);
    }
    let mut max_diff = 0.0f64;
    for i in 0..n_s {
        for j in 0..n_s {
            let d = (schur_block.get(i, j) - our_s[i + j * n_s]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    Some(max_diff < 1e-9)
}

/// Oracle: monolithic factorization of the full system.
fn oracle(k: &Kkt, b: &[f64]) -> (Vec<f64>, Inertia, usize, f64, f64) {
    let (r, c, v) = k.full_lower();
    let m = CscMatrix::from_triplets(k.n(), &r, &c, &v).unwrap();
    let (solver, inertia, nnz_l, t_factor) = factor_identity_scaled(&m);
    let t = Instant::now();
    let x = solver.solve(b).unwrap();
    let t_solve = t.elapsed().as_secs_f64();
    (x, inertia, nnz_l, t_factor, t_solve)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + ((i % 7) as f64) * 0.3 - ((i % 3) as f64) * 0.5).collect()
}

fn run_case(label: &str, n_f: usize, n_s: usize, coupling_deg: usize, verbose: bool) {
    run_case_ex(label, n_f, n_s, coupling_deg, verbose, false)
}

fn run_case_ex(label: &str, n_f: usize, n_s: usize, coupling_deg: usize, verbose: bool, f_indef: bool) {
    let k = Kkt::build_ex(n_f, n_s, coupling_deg, f_indef);
    let b = rhs(k.n());

    let (x_true, inertia_true, nnz_l_full, t_full_factor, t_full_solve) = oracle(&k, &b);
    let res = schur_solve(&k, &b);

    let sol_err = max_abs_diff(&res.x, &x_true);
    let inertia_ok = res.inertia == inertia_true;
    let schur_total = res.t_factor_ff + res.t_form_s + res.t_factor_s + res.t_backsolve;

    println!("── {label}  (n_f={n_f}, n_s={n_s}, n={})", k.n());
    println!(
        "   correctness: max|x_schur - x_oracle| = {sol_err:.3e}   {}",
        if sol_err < 1e-7 { "OK" } else { "*** FAIL ***" }
    );
    println!(
        "   inertia:     schur={:?}  oracle={:?}   {}",
        (res.inertia.positive, res.inertia.negative, res.inertia.zero),
        (inertia_true.positive, inertia_true.negative, inertia_true.zero),
        if inertia_ok { "OK (Sylvester)" } else { "*** FAIL ***" }
    );
    println!(
        "   native S cross-check (feral factorize_multifrontal_with_schur): {}",
        match res.native_s_matches {
            Some(true) => "MATCHES our S".to_string(),
            Some(false) => "*** DIFFERS ***".to_string(),
            None => "rejected (F3.2b scope limit / not applicable)".to_string(),
        }
    );
    if verbose {
        println!(
            "   sparsity:  nnz(L_AFF)={}  (full-M nnz(L)={})   dense S: {}x{}={} f64 ({} KiB)",
            res.nnz_l_ff,
            nnz_l_full,
            n_s,
            n_s,
            n_s * n_s,
            (n_s * n_s * 8) / 1024
        );
        println!(
            "   W intermediate (A_FF^-1 A_FS, dense): {} f64 ({} KiB)",
            res.w_store,
            (res.w_store * 8) / 1024
        );
        println!(
            "   time[s]: factorAFF={:.4} formS={:.4} factorS={:.4} backsolve={:.4} | schurTOTAL={:.4}  ||  oracle factor={:.4} solve={:.4}",
            res.t_factor_ff,
            res.t_form_s,
            res.t_factor_s,
            res.t_backsolve,
            schur_total,
            t_full_factor,
            t_full_solve
        );
    }
    assert!(sol_err < 1e-6, "solution mismatch");
    assert!(inertia_ok, "inertia (Sylvester) mismatch");
    println!();
}

fn main() {
    println!("=== pounce#180 item-2 Phase-0 spike: Schur KKT backsolve + inertia + scaling ===\n");

    println!("## Correctness & Sylvester inertia (small, thorough) ##\n");
    run_case("tiny", 6, 2, 2, true);
    run_case("small", 50, 4, 3, true);
    run_case("medium", 500, 8, 4, true);

    println!("## Indefinite eliminated block A_FF (both A_FF and S carry negatives) ##\n");
    run_case_ex("indef small", 50, 4, 3, true, true);
    run_case_ex("indef medium", 2000, 16, 4, true, true);

    println!("## Sweep A: grow the SPARSE block n_F, hold n_S=8 (should stay ~linear) ##\n");
    for &n_f in &[1_000usize, 4_000, 16_000, 64_000, 256_000] {
        run_case(&format!("A n_f={n_f}"), n_f, 8, 4, true);
    }

    println!("## Sweep B: hold n_F=8000, grow the DENSE Schur block n_S (find the trap) ##\n");
    for &n_s in &[2usize, 8, 32, 128, 512, 1024] {
        run_case(&format!("B n_s={n_s}"), 8_000, n_s, 4, true);
    }

    println!("All correctness + inertia assertions passed.");
}
