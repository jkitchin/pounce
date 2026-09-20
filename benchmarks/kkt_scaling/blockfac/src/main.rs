//! Does block-granular parallelism recover what feral's elimination tree
//! cannot on an arrowhead KKT?
//!
//! Input: `kkt.bin` from `../export_kkt_blocks.py` — a KKT dumped by
//! `pounce --dump kkt:N`, its lower triangle in 0-based triplets, plus a block
//! label per index (-1 = the shared border).
//!
//!     pounce model.nl out.sol --dump kkt:10 --dump-dir dump
//!     python ../export_kkt_blocks.py dump/iter_010/kkt_solve_001.jsonl kkt.bin
//!     cargo run --release -- kkt.bin
//!
//! Compares, on the same matrix:
//!   monolithic  — one feral factorization (what pounce does today);
//!   block       — each block factored independently (rayon), then the dense
//!                 Schur complement on the border, factored on its own.
//! Reports wall time per phase and checks the inertias agree
//! (Haynsworth: inertia(K) = sum_k inertia(A_kk) + inertia(S)).
use feral::symbolic::OrderingMethod;
use feral::{CscMatrix, FactorStatus, Solver};
use rayon::prelude::*;
use std::io::Read;
use std::time::Instant;

#[allow(clippy::type_complexity)]
type Kkt = (usize, Vec<i32>, Vec<i32>, Vec<f64>, Vec<i32>, usize, Vec<f64>, Vec<f64>);

fn read_kkt(path: &str) -> Kkt {
    let mut f = std::fs::File::open(path).expect("open kkt.bin");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let mut off = 0usize;
    let mut take_i64 = || {
        let v = i64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        off += 8;
        v as usize
    };
    let (n, nnz, nblocks) = (take_i64(), take_i64(), take_i64());
    let mut take = |count: usize, width: usize| {
        let s = off;
        off += count * width;
        &buf[s..off]
    };
    let irn: Vec<i32> = take(nnz, 4).chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let jcn: Vec<i32> = take(nnz, 4).chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let vals: Vec<f64> = take(nnz, 8).chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let lab: Vec<i32> = take(n, 4).chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let f64s = |b: &[u8]| -> Vec<f64> {
        b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    };
    let rhs = f64s(take(n, 8));
    let sol = f64s(take(n, 8));
    (n, irn, jcn, vals, lab, nblocks, rhs, sol)
}

fn sizes_max_local(n: usize) -> usize {
    n
}

fn order_from(name: &str) -> OrderingMethod {
    match name {
        "metis" => OrderingMethod::MetisND,
        "amd" => OrderingMethod::Amd,
        "amf" => OrderingMethod::Amf,
        _ => OrderingMethod::Auto,
    }
}

fn solver(ordering: OrderingMethod, parallel: bool) -> Solver {
    Solver::new().with_ordering(ordering).with_parallel(parallel)
}

fn inertia_of(s: &Solver) -> (usize, usize, usize) {
    s.inertia().map(|i| (i.positive, i.negative, i.zero)).unwrap_or((0, 0, 0))
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "../scopf/kkt.bin".into());
    let (n, irn, jcn, vals, lab, nblocks, rhs, pounce_sol) = read_kkt(&path);
    let border: Vec<usize> = (0..n).filter(|&i| lab[i] < 0).collect();
    let nb = border.len();
    println!("KKT n={n} nnz={} blocks={nblocks} border={nb}", irn.len());

    // ---- monolithic ------------------------------------------------------
    let rows: Vec<usize> = irn.iter().map(|&v| v as usize).collect();
    let cols: Vec<usize> = jcn.iter().map(|&v| v as usize).collect();
    let t = Instant::now();
    let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).expect("build");
    let build_s = t.elapsed().as_secs_f64();
    println!("build {build_s:.2}s");
    let mut mono_s = f64::INFINITY;
    let mut mono_inertia = (0, 0, 0);
    let mut best = String::new();
    for name in ["auto", "amf", "amd", "metis"] {
        for par in [false, true] {
            let mut m = solver(order_from(name), par);
            let t = Instant::now();
            let st = m.factor(&a, None);
            let first = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let _ = m.factor(&a, None);            // pattern cached, as in an IPM
            let secs = t.elapsed().as_secs_f64();
            println!("  monolithic {name:>5} parallel={par:<5}: first {first:6.2}s  refactor {secs:6.2}s {st:?}");
            if secs < mono_s {
                mono_s = secs;
                mono_inertia = inertia_of(&m);
                best = format!("{name} parallel={par}");
            }
        }
    }
    println!("monolithic best: {best} at {mono_s:.2}s, inertia {mono_inertia:?}");
    let mut mono_solver = solver(order_from(best.split_whitespace().next().unwrap()), true);
    assert!(matches!(mono_solver.factor(&a, None), FactorStatus::Success));

    // ---- block-parallel --------------------------------------------------
    // local index of every unknown inside its block, and border position
    let mut local = vec![usize::MAX; n];
    let mut sizes = vec![0usize; nblocks];
    for i in 0..n {
        if lab[i] >= 0 {
            let b = lab[i] as usize;
            local[i] = sizes[b];
            sizes[b] += 1;
        }
    }
    let mut bpos = vec![usize::MAX; n];
    for (k, &i) in border.iter().enumerate() {
        bpos[i] = k;
    }
    // scatter the triplets: block-local, block-border coupling, border-border
    let mut blk: Vec<(Vec<usize>, Vec<usize>, Vec<f64>)> = (0..nblocks).map(|_| (vec![], vec![], vec![])).collect();
    let mut cpl: Vec<(Vec<usize>, Vec<usize>, Vec<f64>)> = (0..nblocks).map(|_| (vec![], vec![], vec![])).collect();
    let mut ss = vec![0.0f64; nb * nb];
    for e in 0..rows.len() {
        let (r, c, v) = (rows[e], cols[e], vals[e]);
        match (lab[r], lab[c]) {
            (br, bc) if br >= 0 && bc >= 0 => {
                assert_eq!(br, bc, "a block-to-block entry must not exist");
                let b = br as usize;
                blk[b].0.push(local[r]);
                blk[b].1.push(local[c]);
                blk[b].2.push(v);
            }
            (br, _) if br >= 0 => {
                let b = br as usize;             // row in block, col on border
                cpl[b].0.push(local[r]);
                cpl[b].1.push(bpos[c]);
                cpl[b].2.push(v);
            }
            (_, bc) if bc >= 0 => {
                let b = bc as usize;             // col in block, row on border
                cpl[b].0.push(local[c]);
                cpl[b].1.push(bpos[r]);
                cpl[b].2.push(v);
            }
            _ => {
                let (i, j) = (bpos[r], bpos[c]); // border-border (lower triangle)
                ss[i + j * nb] += v;
                if i != j {
                    ss[j + i * nb] += v;
                }
            }
        }
    }

    // pass 1: build + factor every block (parallel), keeping the factors
    let t_build = Instant::now();
    let mut solvers: Vec<(Solver, CscMatrix)> = (0..nblocks)
        .into_par_iter()
        .map(|b| {
            let (ref r, ref c, ref v) = blk[b];
            let m = CscMatrix::from_triplets(sizes[b], r, c, v).expect("block build");
            let mut s = solver(order_from(&std::env::var("BLOCK_ORDERING").unwrap_or("auto".into())), false);
            let st = s.factor(&m, None);
            assert!(matches!(st, FactorStatus::Success), "block {b}: {st:?}");
            (s, m)
        })
        .collect();
    println!("  pass 0 (symbolic + first factor, parallel): {:.2}s", t_build.elapsed().as_secs_f64());
    let t_fac = Instant::now();
    solvers.par_iter_mut().for_each(|(s, m)| {
        let _ = s.factor(m, None);
    });
    let fac_wall = t_fac.elapsed().as_secs_f64();
    let t_schur = Instant::now();
    let _schur_parts: Vec<(Vec<f64>, (usize, usize, usize))> = solvers
        .par_iter()
        .enumerate()
        .map(|(b, (s, _))| {
            let mut rhs = vec![0.0f64; sizes[b] * nb];
            for e in 0..cpl[b].0.len() {
                rhs[cpl[b].0[e] + cpl[b].1[e] * sizes[b]] += cpl[b].2[e];
            }
            let w = s.solve_many(&rhs, nb).expect("solve_many");
            let mut sk = vec![0.0f64; nb * nb];
            for e in 0..cpl[b].0.len() {
                let (li, bj, val) = (cpl[b].0[e], cpl[b].1[e], cpl[b].2[e]);
                for j in 0..nb {
                    sk[bj + j * nb] += val * w[li + j * sizes[b]];
                }
            }
            (sk, inertia_of(s))
        })
        .collect();
    let schur_wall = t_schur.elapsed().as_secs_f64();
    println!("  steady state, parallel wall: block refactor {fac_wall:.3}s  schur formation {schur_wall:.3}s");
    // --- native path: feral forms each block's Schur contribution inside the
    // factorization (no dense multi-RHS solve) -----------------------------
    {
        use feral::symbolic::{symbolic_factorize_with_schur, SupernodeParams};
        use feral::{factorize_multifrontal_with_schur, NumericParams};
        let schur_idx: Vec<usize> = (0..nb).map(|k| sizes_max_local(sizes[0]) + k).collect();
        let _ = &schur_idx;
        let t_sym = Instant::now();
        let prepared: Vec<(CscMatrix, feral::symbolic::SymbolicFactorization)> = (0..nblocks)
            .into_par_iter()
            .map(|b| {
                let nloc = sizes[b];
                let dim = nloc + nb;
                let (mut r, mut c, mut v) = (vec![], vec![], vec![]);
                for e in 0..blk[b].0.len() {
                    r.push(blk[b].0[e]);
                    c.push(blk[b].1[e]);
                    v.push(blk[b].2[e]);
                }
                for e in 0..cpl[b].0.len() {
                    r.push(nloc + cpl[b].1[e]);   // border row (lower triangle)
                    c.push(cpl[b].0[e]);          // block column
                    v.push(cpl[b].2[e]);
                }
                for k in 0..nb {
                    r.push(nloc + k);             // keep the border diagonal present
                    c.push(nloc + k);
                    v.push(0.0);
                }
                let m = CscMatrix::from_triplets(dim, &r, &c, &v).expect("block+border build");
                let idx: Vec<usize> = (nloc..dim).collect();
                let sym = symbolic_factorize_with_schur(&m, &SupernodeParams::default(), &idx)
                    .expect("symbolic with schur");
                (m, sym)
            })
            .collect();
        println!("  native: symbolic with schur tail (parallel): {:.2}s", t_sym.elapsed().as_secs_f64());
        let t_nat = Instant::now();
        let parts: Vec<(Vec<f64>, (usize, usize, usize), feral::numeric::factorize::SparseFactors)> = prepared
            .par_iter()
            .map(|(m, sym)| {
                let (f, inertia, sb) =
                    factorize_multifrontal_with_schur(m, sym, &NumericParams::default())
                        .expect("factor with schur");
                (sb.data, (inertia.positive, inertia.negative, inertia.zero), f)
            })
            .collect();
        // Can those factors solve A_kk alone? (the tail is not eliminated).
        // Compare against the block-only Solver factors on block 0.
        {
            let b = 0usize;
            let nloc = sizes[b];
            let mut rhs_full = vec![0.0f64; nloc + nb];
            for i in 0..nloc {
                rhs_full[i] = 1.0 + (i % 7) as f64;
            }
            let x_schur = feral::solve_sparse(&parts[b].2, &rhs_full).expect("solve with schur factors");
            let x_ref = solvers[b].0.solve(&rhs_full[..nloc].to_vec()).expect("block solve");
            let err = x_ref
                .iter()
                .zip(x_schur.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            let scale = x_ref.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1.0);
            println!("  schur-tail factors vs block-only factors on block 0: max |diff| {:.2e} (scale {:.2e})", err, scale);
        }
        let nat_wall = t_nat.elapsed().as_secs_f64();
        let mut s_nat = ss.clone();
        let mut inat = (0usize, 0usize, 0usize);
        for (sk, i, _) in &parts {
            inat = (inat.0 + i.0, inat.1 + i.1, inat.2 + i.2);
            for (d, v) in s_nat.iter_mut().zip(sk.iter()) {
                *d += v;            // native returns the Schur complement itself
            }
        }
        let (mut r, mut c, mut v) = (vec![], vec![], vec![]);
        for j in 0..nb {
            for i in j..nb {
                r.push(i);
                c.push(j);
                v.push(s_nat[i + j * nb]);
            }
        }
        let sm = CscMatrix::from_triplets(nb, &r, &c, &v).expect("S build");
        let mut ssolve = solver(OrderingMethod::Amd, false);
        let t_sf = Instant::now();
        let st = ssolve.factor(&sm, None);
        let si = inertia_of(&ssolve);
        let comb = (inat.0 + si.0, inat.1 + si.1, inat.2 + si.2);
        println!(
            "  native block factor+schur (parallel): {nat_wall:.3}s + border factor {sf:.3}s {st:?}  inertia {comb:?}",
            sf = t_sf.elapsed().as_secs_f64()
        );
        println!(
            "NATIVE vs monolithic refactor: {:.2}x  (inertia match {})",
            mono_s / (nat_wall + t_sf.elapsed().as_secs_f64()),
            comb == mono_inertia
        );
    }

    let t_all = Instant::now();
    let per: Vec<(f64, f64, (usize, usize, usize), Vec<f64>)> = (0..nblocks)
        .into_par_iter()
        .map(|b| {
            let (ref r, ref c, ref v) = blk[b];
            let m = CscMatrix::from_triplets(sizes[b], r, c, v).expect("block build");
            let mut s = solver(order_from(&std::env::var("BLOCK_ORDERING").unwrap_or("auto".into())), false);
            let st = s.factor(&m, None);           // first factor: pays symbolic
            let t = Instant::now();
            let _ = s.factor(&m, None);            // refactor, as in an IPM
            let fac = t.elapsed().as_secs_f64();
            assert!(matches!(st, FactorStatus::Success), "block {b}: {st:?}");
            // W = A_kk^{-1} A_kb, then S_k = A_kb^T W  (dense, nb columns)
            let t = Instant::now();
            let mut rhs = vec![0.0f64; sizes[b] * nb];
            for e in 0..cpl[b].0.len() {
                rhs[cpl[b].0[e] + cpl[b].1[e] * sizes[b]] += cpl[b].2[e];
            }
            let w = s.solve_many(&rhs, nb).expect("solve_many");
            let mut sk = vec![0.0f64; nb * nb];
            for e in 0..cpl[b].0.len() {
                let (li, bj, val) = (cpl[b].0[e], cpl[b].1[e], cpl[b].2[e]);
                for j in 0..nb {
                    sk[bj + j * nb] += val * w[li + j * sizes[b]];
                }
            }
            let schur = t.elapsed().as_secs_f64();
            (fac, schur, inertia_of(&s), sk)
        })
        .collect();
    let mut s_dense = ss.clone();
    let mut inert = (0usize, 0usize, 0usize);
    let (mut fac_sum, mut schur_sum, mut fac_max, mut schur_max) = (0.0, 0.0, 0.0f64, 0.0f64);
    for (fac, schur, i, sk) in &per {
        fac_sum += fac;
        schur_sum += schur;
        fac_max = fac_max.max(*fac);
        schur_max = schur_max.max(*schur);
        inert = (inert.0 + i.0, inert.1 + i.1, inert.2 + i.2);
        for (d, v) in s_dense.iter_mut().zip(sk.iter()) {
            *d -= v;
        }
    }
    // factor the dense border system as a sparse matrix
    let (mut r, mut c, mut v) = (vec![], vec![], vec![]);
    for j in 0..nb {
        for i in j..nb {
            r.push(i);
            c.push(j);
            v.push(s_dense[i + j * nb]);
        }
    }
    let t = Instant::now();
    let sm = CscMatrix::from_triplets(nb, &r, &c, &v).expect("S build");
    let mut border_solver = solver(OrderingMethod::Amd, false);
    let st = border_solver.factor(&sm, None);
    let s_fac = t.elapsed().as_secs_f64();
    let si = inertia_of(&border_solver);
    let total = t_all.elapsed().as_secs_f64();
    let combined = (inert.0 + si.0, inert.1 + si.1, inert.2 + si.2);
    println!(
        "block-parallel ({} threads): total {total:.2}s  [block factors: sum {fac_sum:.2}s, slowest {fac_max:.2}s]  \
         [schur forms: sum {schur_sum:.2}s, slowest {schur_max:.2}s]  [border {nb}x{nb} factor {s_fac:.3}s {st:?}]",
        rayon::current_num_threads()
    );
    println!("inertia: monolithic {mono_inertia:?}  blocks+schur {combined:?}  match {}", combined == mono_inertia);
    println!(
        "DENSE-RHS block path vs monolithic refactor: {:.2}x (the multi-RHS Schur \
         formation is what costs; see the native line above)",
        mono_s / total
    );

    // ---- back-solve: the other half of a KKT solve -----------------------
    // K x = b by block elimination:
    //   y_k = A_kk^-1 b_k            (parallel)
    //   S dx_s = b_s - sum_k A_sk y_k
    //   x_k = A_kk^-1 (b_k - A_ks dx_s)   (parallel)
    // `solvers` already holds each block's factors; the border system is the
    // Schur complement formed above.
    let residual = |x: &[f64]| -> f64 {
        let mut r = rhs.clone();
        for e in 0..rows.len() {
            let (i, j, v) = (rows[e], cols[e], vals[e]);
            r[i] -= v * x[j];
            if i != j {
                r[j] -= v * x[i];
            }
        }
        let num = r.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let den = rhs.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1.0);
        num / den
    };
    let t = Instant::now();
    let x_mono = mono_solver.solve(&rhs).expect("monolithic solve");
    let mono_solve = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let y: Vec<Vec<f64>> = solvers
        .par_iter()
        .enumerate()
        .map(|(b, (s, _))| {
            let mut bk = vec![0.0f64; sizes[b]];
            for i in 0..n {
                if lab[i] as usize == b && lab[i] >= 0 {
                    bk[local[i]] = rhs[i];
                }
            }
            s.solve(&bk).expect("block solve")
        })
        .collect();
    let mut rs: Vec<f64> = border.iter().map(|&i| rhs[i]).collect();
    for b in 0..nblocks {
        for e in 0..cpl[b].0.len() {
            rs[cpl[b].1[e]] -= cpl[b].2[e] * y[b][cpl[b].0[e]];
        }
    }
    let dx_s = border_solver.solve(&rs).expect("border solve");
    let x_blocks: Vec<Vec<f64>> = solvers
        .par_iter()
        .enumerate()
        .map(|(b, (s, _))| {
            let mut bk = vec![0.0f64; sizes[b]];
            for i in 0..n {
                if lab[i] >= 0 && lab[i] as usize == b {
                    bk[local[i]] = rhs[i];
                }
            }
            for e in 0..cpl[b].0.len() {
                bk[cpl[b].0[e]] -= cpl[b].2[e] * dx_s[cpl[b].1[e]];
            }
            s.solve(&bk).expect("block solve 2")
        })
        .collect();
    let block_solve = t.elapsed().as_secs_f64();
    let mut x_blk = vec![0.0f64; n];
    for i in 0..n {
        if lab[i] >= 0 {
            x_blk[i] = x_blocks[lab[i] as usize][local[i]];
        } else {
            x_blk[i] = dx_s[bpos[i]];
        }
    }
    println!(
        "back-solve: monolithic {mono_solve:.3}s (residual {:.2e})  block-parallel {block_solve:.3}s \
         (residual {:.2e})  pounce's own solution residual {:.2e}",
        residual(&x_mono),
        residual(&x_blk),
        residual(&pounce_sol)
    );
}
