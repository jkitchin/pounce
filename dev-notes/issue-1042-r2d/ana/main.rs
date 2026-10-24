use feral::numeric::factorize::NumericParams;
use feral::scaling::ScalingStrategy;
use feral::symbolic::{OrderingMethod, SupernodeParams};
use feral::Solver;
use std::path::Path;

fn run(label: &str, csc: &feral::CscMatrix, pivtol: f64, scaling: ScalingStrategy, ord: Option<OrderingMethod>) {
    let mut np = NumericParams::default();
    np.bk.pivot_threshold = pivtol;
    let mut s = Solver::with_params(np, SupernodeParams::default()).with_parallel(false).with_scaling(scaling);
    if let Some(o) = ord {
        s = s.with_ordering(o);
    }
    let st = s.factor(csc, None);
    let i = s.inertia().cloned();
    let fs = s.last_factor_stats();
    let mut piv: Vec<f64> = Vec::new();
    if let Some(f) = s.factors() {
        for nf in &f.node_factors {
            let ff = &nf.frontal_factors;
            let mut k = 0;
            while k < ff.nelim {
                if k + 1 < ff.nelim && ff.d_subdiag[k] != 0.0 {
                    let (a, b, c) = (ff.d_diag[k], ff.d_subdiag[k], ff.d_diag[k + 1]);
                    let tr = a + c; let det = a * c - b * b;
                    let l = (tr.abs() + (tr * tr - 4.0 * det).max(0.0).sqrt()) * 0.5;
                    piv.push(l); piv.push(if l > 0.0 { det.abs() / l } else { 0.0 });
                    k += 2;
                } else { piv.push(ff.d_diag[k].abs()); k += 1; }
            }
        }
    }
    piv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pv: Vec<String> = piv.iter().take(5).map(|x| format!("{x:.2e}")).collect();
    println!(
        "  {label:<34} status={:?} inertia(+,-,0)={:?} min_piv={:.3e} max_piv={:.3e} n_tiny={:?} smallest_pivots={}",
        st,
        i.map(|i| (i.positive, i.negative, i.zero)),
        s.min_pivot_magnitude().unwrap_or(f64::NAN),
        s.max_pivot_magnitude().unwrap_or(f64::NAN),
        fs.map(|f| f.n_tiny),
        pv.join(","),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m = feral::read_mtx(Path::new(&args[1])).unwrap();
    let n = m.n;
    let csc = m.to_csc().unwrap();
    let pivtol_run: f64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1e-8);
    // dense exact inertia
    #[cfg(feature = "dense")]
    {
        let mut a = faer::Mat::<f64>::zeros(n, n);
        for &(i, j, v) in &m.entries {
            a[(i, j)] += v;
            if i != j {
                a[(j, i)] += v;
            }
        }
        let mut knorm: f64 = 0.0;
        for i in 0..n {
            let mut r = 0.0;
            for j in 0..n {
                r += a[(i, j)].abs();
            }
            knorm = knorm.max(r);
        }
        let ev = a.self_adjoint_eigenvalues(faer::Side::Lower).unwrap();
        let neg = ev.iter().filter(|&&x| x < 0.0).count();
        let pos = ev.iter().filter(|&&x| x > 0.0).count();
        let mut abs: Vec<f64> = ev.iter().map(|x| x.abs()).collect();
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t1 = n as f64 * f64::EPSILON * knorm;
        let t2 = 1e-10 * knorm;
        println!("dense: n={n} ||K||inf={knorm:.4e} inertia(+,-,0)=({pos},{neg},{})", n - pos - neg);
        println!("  #|l|<n*eps*||K||={} (thr {t1:.2e})  #|l|<1e-10*||K||={} (thr {t2:.2e})",
            abs.iter().filter(|&&x| x < t1).count(), abs.iter().filter(|&&x| x < t2).count());
        let mut evs: Vec<f64> = ev.clone();
        evs.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        println!("  smallest |l| (signed): {:?}", &evs[..6.min(n)].iter().map(|x| format!("{x:.4e}")).collect::<Vec<_>>());
        println!("  smallest |l| / ||K||: {:.3e}", abs[0] / knorm);
        // eigenvalues near zero split by sign
        let neg_small: Vec<f64> = ev.iter().cloned().filter(|x| *x < 0.0 && x.abs() < 1e-6 * knorm).collect();
        let pos_small: Vec<f64> = ev.iter().cloned().filter(|x| *x > 0.0 && x.abs() < 1e-6 * knorm).collect();
        println!("  |l|<1e-6||K||: neg {:?} pos {:?}", neg_small.iter().map(|x| format!("{x:.3e}")).collect::<Vec<_>>(), pos_small.iter().map(|x| format!("{x:.3e}")).collect::<Vec<_>>());
    }
    let _ = n;
    println!("feral {}:", option_env!("FERAL_VER").unwrap_or("(version pinned in Cargo.toml)"));
    for (lab, sc) in [("Auto", ScalingStrategy::Auto), ("Identity", ScalingStrategy::Identity), ("InfNorm", ScalingStrategy::InfNorm)] {
        for pt in [1e-8, pivtol_run, 0.01, 0.1, 0.5] {
            run(&format!("{lab} pivtol={pt:e}"), &csc, pt, sc.clone(), None);
        }
    }
    if let Some(p) = args.get(2).filter(|s| !s.is_empty() && *s != "-") {
        let perm: Vec<usize> = std::fs::read_to_string(p).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
        for pt in [1e-8, pivtol_run] {
            run(&format!("Auto replay-perm pivtol={pt:e}"), &csc, pt, ScalingStrategy::Auto, Some(OrderingMethod::External(perm.clone())));
        }
        if let Some(sp) = args.get(4) {
            let sc: Vec<f64> = std::fs::read_to_string(sp).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
            for pt in [1e-8, pivtol_run, 0.01, 0.1, 0.5] {
                run(&format!("replay perm+scaling pivtol={pt:e}"), &csc, pt, ScalingStrategy::External(sc.clone()), Some(OrderingMethod::External(perm.clone())));
            }
        }
    }
}
