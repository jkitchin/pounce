//! gpu_spike — Phase-0 Apple-Silicon / cross-vendor GPU spike harnesses
//! for the pounce gpu-batched-layers roadmap (dev-notes/research).
//!
//! Three steps, selected by the first argument:
//!   baseline   Step 0  CPU batched-QP throughput (the bar GPU must beat)
//!   microbench Step 1  GPU vs CPU batched dense Cholesky+solve, crossover
//!   accuracy   Step 2  on-device f32 accuracy + run-to-run variation,
//!                      and the f64-refinement-tail recovery
//!   all        run baseline + microbench + accuracy
//!
//! ON/OFF for performance measurement:
//!   * compile-time:  build with/without `--features gpu`
//!   * runtime:       `--device cpu|gpu|both` forces a side even when the
//!                    gpu feature is compiled in (A/B the same binary)
//!
//! Examples:
//!   cargo run --release -- baseline -b 1024 -n 32 -m 32
//!   cargo run --release --features gpu -- microbench --batches 256,1024,4096 --dims 16,32,64
//!   cargo run --release --features gpu -- accuracy -n 48 --jitter 1e-3 -r 20 --device gpu

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use pounce_gpu_spike::*;

#[cfg(feature = "gpu")]
mod gpu;

struct Args {
    step: String,
    batches: Vec<usize>,
    dims: Vec<usize>,
    cons: usize,
    threads: usize,
    repeats: usize,
    jitter: f64,
    device: String, // cpu | gpu | both
    backend: String, // vulkan | metal | dx12 | gl | all
}

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let default_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let default_device = if cfg!(feature = "gpu") { "both" } else { "cpu" };
    let mut a = Args {
        step: "all".into(),
        batches: vec![1024],
        dims: vec![32],
        cons: 32,
        threads: default_threads,
        repeats: 5,
        jitter: 1.0,
        device: default_device.into(),
        backend: "all".into(),
    };
    let mut i = 0;
    if !raw.is_empty() && !raw[0].starts_with('-') {
        a.step = raw[0].clone();
        i = 1;
    }
    while i < raw.len() {
        let key = raw[i].clone();
        let mut take = || -> String {
            i += 1;
            raw.get(i).cloned().unwrap_or_default()
        };
        match key.as_str() {
            "-b" | "--batch" | "--batches" => a.batches = parse_list(&take()),
            "-n" | "--dim" | "--dims" => a.dims = parse_list(&take()),
            "-m" | "--cons" => a.cons = take().parse().unwrap_or(a.cons),
            "-t" | "--threads" => a.threads = take().parse().unwrap_or(a.threads),
            "-r" | "--repeats" => a.repeats = take().parse().unwrap_or(a.repeats),
            "--jitter" => a.jitter = take().parse().unwrap_or(a.jitter),
            "--device" => a.device = take(),
            "--backend" => a.backend = take(),
            _ => eprintln!("warning: ignoring unknown arg `{}`", key),
        }
        drop(take);
        i += 1;
    }
    a
}

fn best(d: &[Duration]) -> Duration {
    d.iter().copied().min().unwrap_or_default()
}

// --------------------------------------------------------------------
// Step 0 — CPU batched-QP baseline (the throughput bar)
// --------------------------------------------------------------------

fn step_baseline(a: &Args) {
    println!("== Step 0: CPU batched-QP baseline (f64 IPM, {} threads) ==", a.threads);
    println!(
        "{:>8} {:>5} {:>5} | {:>10} {:>12} {:>8}",
        "batch", "n", "m", "wall(ms)", "solves/sec", "med_it"
    );
    for &n in &a.dims {
        for &b in &a.batches {
            // build B identical-structure QPs (different RNG seeds = different data)
            let qps: Vec<Qp> = (0..b)
                .map(|k| gen_qp(n, a.cons, &mut Rng::new(0x5EED ^ ((n as u64) << 20) ^ k as u64)))
                .collect();
            let mut runs = Vec::new();
            let mut iters_sample = Vec::new();
            for _ in 0..a.repeats {
                let t0 = Instant::now();
                let total_iters = AtomicUsize::new(0);
                let solved = AtomicUsize::new(0);
                let chunk = (b + a.threads - 1) / a.threads.max(1);
                thread::scope(|sc| {
                    for ch in qps.chunks(chunk.max(1)) {
                        let total_iters = &total_iters;
                        let solved = &solved;
                        sc.spawn(move || {
                            let mut li = 0usize;
                            let mut ls = 0usize;
                            for qp in ch {
                                let st = solve_qp::<f64>(qp, 1e-8, 100);
                                li += st.iters;
                                if st.converged {
                                    ls += 1;
                                }
                            }
                            total_iters.fetch_add(li, Ordering::Relaxed);
                            solved.fetch_add(ls, Ordering::Relaxed);
                        });
                    }
                });
                runs.push(t0.elapsed());
                iters_sample.push(total_iters.load(Ordering::Relaxed) as f64 / b as f64);
                let _ = solved;
            }
            let wall = best(&runs);
            let per_sec = b as f64 / wall.as_secs_f64();
            let med_it = iters_sample[iters_sample.len() / 2];
            println!(
                "{:>8} {:>5} {:>5} | {:>10.2} {:>12.0} {:>8.0}",
                b,
                n,
                a.cons,
                wall.as_secs_f64() * 1e3,
                per_sec,
                med_it
            );
        }
    }
    println!();
}

// --------------------------------------------------------------------
// Step 1 — GPU vs CPU batched dense Cholesky+solve (crossover)
// --------------------------------------------------------------------

fn microbench_cpu(systems: &[SpdSystem], threads: usize) -> Duration {
    let t0 = Instant::now();
    let chunk = (systems.len() + threads - 1) / threads.max(1);
    thread::scope(|sc| {
        for ch in systems.chunks(chunk.max(1)) {
            sc.spawn(move || {
                for sys in ch {
                    let mut k: Vec<f32> = sys.mat.iter().map(|&v| v as f32).collect();
                    let mut b: Vec<f32> = sys.rhs.iter().map(|&v| v as f32).collect();
                    let _ = chol_solve::<f32>(sys.n, &mut k, &mut b);
                }
            });
        }
    });
    t0.elapsed()
}

fn step_microbench(a: &Args) {
    println!(
        "== Step 1: batched dense Cholesky+solve (f32) — CPU vs GPU crossover ==\n\
         (spike kernel is global-memory, unoptimized: GPU number is a lower bound)"
    );
    let want_cpu = a.device != "gpu";

    #[cfg(feature = "gpu")]
    let ctx = if a.device != "cpu" {
        match gpu::GpuContext::new(&a.backend) {
            Some(c) => {
                println!("GPU: {} via {}", c.adapter_name, c.backend);
                Some(c)
            }
            None => {
                println!("GPU: no usable adapter on `{}` — CPU only (fallback)", a.backend);
                None
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "gpu"))]
    if a.device != "cpu" {
        println!("GPU: built without --features gpu — CPU only");
    }

    println!(
        "\n{:>8} {:>5} | {:>11} {:>11} | {:>9}",
        "batch", "n", "cpu(ms)", "gpu(ms)", "speedup"
    );
    for &n in &a.dims {
        for &b in &a.batches {
            let systems: Vec<SpdSystem> = (0..b)
                .map(|k| gen_spd(n, a.jitter, &mut Rng::new(0xA11CE ^ ((n as u64) << 20) ^ k as u64)))
                .collect();

            let cpu_ms = if want_cpu {
                let mut runs = Vec::new();
                for _ in 0..a.repeats {
                    runs.push(microbench_cpu(&systems, a.threads));
                }
                Some(best(&runs).as_secs_f64() * 1e3)
            } else {
                None
            };

            #[allow(unused_mut)]
            let mut gpu_ms: Option<f64> = None;
            #[cfg(feature = "gpu")]
            if let Some(ctx) = &ctx {
                let mut runs = Vec::new();
                let mut ok = true;
                for _ in 0..a.repeats {
                    let mut mats: Vec<f32> = Vec::with_capacity(b * n * n);
                    let mut rhs: Vec<f32> = Vec::with_capacity(b * n);
                    for sys in &systems {
                        mats.extend(sys.mat.iter().map(|&v| v as f32));
                        rhs.extend(sys.rhs.iter().map(|&v| v as f32));
                    }
                    match ctx.solve(n, b, &mut mats, &mut rhs) {
                        Ok(d) => runs.push(d),
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    gpu_ms = Some(best(&runs).as_secs_f64() * 1e3);
                }
            }

            let speedup = match (cpu_ms, gpu_ms) {
                (Some(c), Some(g)) if g > 0.0 => format!("{:.2}x", c / g),
                _ => "-".into(),
            };
            println!(
                "{:>8} {:>5} | {:>11} {:>11} | {:>9}",
                b,
                n,
                cpu_ms.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
                gpu_ms.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "-".into()),
                speedup
            );
        }
    }
    println!();
}

// --------------------------------------------------------------------
// Step 2 — on-device f32 accuracy + run-to-run variation + f64 tail
// --------------------------------------------------------------------

fn step_accuracy(a: &Args) {
    let n = *a.dims.first().unwrap_or(&32);
    let b = *a.batches.first().unwrap_or(&256);
    println!(
        "== Step 2: f32 accuracy & determinism (n={}, batch={}, jitter={:.0e}) ==",
        n, b, a.jitter
    );
    let systems: Vec<SpdSystem> = (0..b)
        .map(|k| gen_spd(n, a.jitter, &mut Rng::new(0xACC ^ ((n as u64) << 20) ^ k as u64)))
        .collect();

    // CPU reference: f64 (truth) and f32 (the floor f32 *should* reach).
    let mut f64_res = 0.0f64;
    let mut f32_res = 0.0f64;
    let mut f32_tail_res = 0.0f64;
    for sys in &systems {
        if let Some(x) = solve_spd::<f64>(sys) {
            f64_res = f64_res.max(residual_inf(sys, &x));
        }
        if let Some(mut x) = solve_spd::<f32>(sys) {
            f32_res = f32_res.max(residual_inf(sys, &x));
            refine_f64(sys, &mut x); // the f64 tail
            f32_tail_res = f32_tail_res.max(residual_inf(sys, &x));
        }
    }
    println!("  CPU f64 reference   max rel-resid: {:.2e}", f64_res);
    println!("  CPU f32             max rel-resid: {:.2e}", f32_res);
    println!("  CPU f32 + f64 tail  max rel-resid: {:.2e}", f32_tail_res);

    gpu_accuracy(a, &systems, n, b);
    println!();
}

#[cfg(feature = "gpu")]
fn gpu_accuracy(a: &Args, systems: &[SpdSystem], n: usize, b: usize) {
    if a.device == "cpu" {
        return;
    }
    let ctx = match gpu::GpuContext::new(&a.backend) {
        Some(c) => {
            println!("  GPU: {} via {}", c.adapter_name, c.backend);
            c
        }
        None => {
            println!("  GPU: no usable adapter — skipped (CPU-only fallback)");
            return;
        }
    };
    // Run the batch `repeats` times; record per-run max residual and the
    // max element-wise spread between runs (the determinism signal).
    let mut first: Option<Vec<f32>> = None;
    let mut max_run_res = 0.0f64;
    let mut max_spread = 0.0f64;
    let mut max_tail_res = 0.0f64;
    for _ in 0..a.repeats {
        let mut mats: Vec<f32> = Vec::with_capacity(b * n * n);
        let mut rhs: Vec<f32> = Vec::with_capacity(b * n);
        for sys in systems {
            mats.extend(sys.mat.iter().map(|&v| v as f32));
            rhs.extend(sys.rhs.iter().map(|&v| v as f32));
        }
        if ctx.solve(n, b, &mut mats, &mut rhs).is_err() {
            println!("  GPU: solve error — skipped");
            return;
        }
        // accuracy + f64 tail, per system
        for (e, sys) in systems.iter().enumerate() {
            let mut x: Vec<f64> = rhs[e * n..(e + 1) * n].iter().map(|&v| v as f64).collect();
            max_run_res = max_run_res.max(residual_inf(sys, &x));
            refine_f64(sys, &mut x);
            max_tail_res = max_tail_res.max(residual_inf(sys, &x));
        }
        match &first {
            None => first = Some(rhs.clone()),
            Some(f0) => {
                for (a0, a1) in f0.iter().zip(rhs.iter()) {
                    max_spread = max_spread.max((*a0 as f64 - *a1 as f64).abs());
                }
            }
        }
    }
    println!("  GPU f32             max rel-resid: {:.2e}", max_run_res);
    println!("  GPU f32 + f64 tail  max rel-resid: {:.2e}", max_tail_res);
    println!(
        "  GPU run-to-run spread (max |Δx| over {} runs): {:.2e}  {}",
        a.repeats,
        max_spread,
        if max_spread == 0.0 { "(bitwise-deterministic)" } else { "(nondeterministic reductions)" }
    );
}

#[cfg(not(feature = "gpu"))]
fn gpu_accuracy(a: &Args, _systems: &[SpdSystem], _n: usize, _b: usize) {
    if a.device != "cpu" {
        println!("  GPU: built without --features gpu — on-device check skipped");
    }
}

fn main() {
    let a = parse_args();
    println!(
        "gpu_spike | step={} device={} backend={} threads={} repeats={} | gpu-feature={}\n",
        a.step,
        a.device,
        a.backend,
        a.threads,
        a.repeats,
        cfg!(feature = "gpu")
    );
    match a.step.as_str() {
        "baseline" => step_baseline(&a),
        "microbench" => step_microbench(&a),
        "accuracy" => step_accuracy(&a),
        "all" => {
            step_baseline(&a);
            step_microbench(&a);
            step_accuracy(&a);
        }
        other => {
            eprintln!("unknown step `{}` (use: baseline | microbench | accuracy | all)", other);
            std::process::exit(2);
        }
    }
}
