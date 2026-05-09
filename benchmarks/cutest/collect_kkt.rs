// Binary: collect KKT matrices from ripopt CUTEst runs for use as FERAL benchmarks.
//
// For each CUTEst problem, runs ripopt with kkt_dump_dir enabled and writes one
// .mtx + .json pair per IPM iteration to a structured output directory.
//
// Usage:
//   cargo run --bin collect_kkt --features cutest --release -- --output /path/to/feral/data/matrices/kkt/
//   cargo run --bin collect_kkt --features cutest --release -- --output /path/ ROSENBR HS35 HS71
//
// Output layout:
//   {output}/{problem_name}/{problem_name}_{iter:04}.mtx
//   {output}/{problem_name}/{problem_name}_{iter:04}.json
//
// Env vars:
//   CUTEST_TIMEOUT  — per-problem timeout in seconds (default: 60)
//   CUTEST_MAX_N    — skip problems with n > this value (default: unlimited)

mod cutest_ffi;
mod cutest_problem;

use cutest_problem::CutestProblem;
use ripopt::SolverOptions;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Subprocess mode: --single PROBLEM (KKT_OUTPUT_DIR must be set in env)
    if args.len() >= 3 && args[1] == "--single" {
        let problem_name = &args[2];
        let output_dir = std::env::var("KKT_OUTPUT_DIR").unwrap_or_else(|_| {
            eprintln!("collect_kkt --single: KKT_OUTPUT_DIR not set");
            std::process::exit(1);
        });
        run_single(problem_name, Path::new(&output_dir));
        return;
    }

    // Main mode
    let (output_dir, problem_names) = parse_args(&args[1..]);
    run_all(&output_dir, &problem_names);
}

/// Subprocess entry point: load one problem, solve with dump enabled, exit.
/// The solve result is ignored — we want KKT matrices from all iterations,
/// including those from problems that fail to converge.
fn run_single(name: &str, output_dir: &Path) {
    let suite_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks").join("cutest");
    let problems_dir = suite_dir.join("problems");

    let lib_path = problems_dir.join(format!("lib{}.{}", name, std::env::consts::DLL_EXTENSION));
    let outsdif_path = problems_dir.join(format!("{}_OUTSDIF.d", name));

    let problem = match CutestProblem::load(
        name,
        lib_path.to_str().unwrap(),
        outsdif_path.to_str().unwrap(),
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("SKIP {} (load failed: {})", name, e);
            std::process::exit(1);
        }
    };

    let problem_output_dir = output_dir.join(name);
    if let Err(e) = std::fs::create_dir_all(&problem_output_dir) {
        eprintln!("SKIP {} (cannot create output dir: {})", name, e);
        std::process::exit(1);
    }

    let options = SolverOptions {
        tol: 1e-8,
        max_iter: 3000,
        mu_strategy_adaptive: true,
        max_wall_time: 30.0,
        print_level: 0,
        kkt_dump_dir: Some(problem_output_dir),
        kkt_dump_name: name.to_string(),
        ..SolverOptions::default()
    };

    // Ignore solve result — dump fires automatically on each successful factorization.
    let _ = ripopt::solve(&problem, &options);
}

/// Main mode: iterate problems, spawn one subprocess per problem with timeout.
fn run_all(output_dir: &Path, problem_names: &[String]) {
    let self_exe = std::env::current_exe().expect("cannot find self executable");

    let timeout_secs: u64 = std::env::var("CUTEST_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let max_n: usize = std::env::var("CUTEST_MAX_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let suite_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks").join("cutest");
    let problems_dir = suite_dir.join("problems");

    std::fs::create_dir_all(output_dir).unwrap_or_else(|e| {
        eprintln!("Cannot create output dir {}: {}", output_dir.display(), e);
        std::process::exit(1);
    });

    eprintln!("Collecting KKT matrices → {}", output_dir.display());
    eprintln!("{} problems to process", problem_names.len());

    let mut n_done = 0usize;
    let mut n_skipped = 0usize;

    for name in problem_names {
        let lib_path =
            problems_dir.join(format!("lib{}.{}", name, std::env::consts::DLL_EXTENSION));
        let outsdif_path = problems_dir.join(format!("{}_OUTSDIF.d", name));

        if !lib_path.exists() || !outsdif_path.exists() {
            eprintln!("  SKIP {} (not prepared — run prepare.sh first)", name);
            n_skipped += 1;
            continue;
        }

        // Quick dimension check before spawning subprocess.
        let (n, m) = match CutestProblem::load(
            name,
            lib_path.to_str().unwrap(),
            outsdif_path.to_str().unwrap(),
        ) {
            Ok(p) => {
                let dims = (p.n, p.m);
                p.cleanup();
                dims
            }
            Err(e) => {
                eprintln!("  SKIP {} (load failed: {})", name, e);
                n_skipped += 1;
                continue;
            }
        };

        if n > max_n {
            eprintln!("  SKIP {} (n={} > max_n={})", name, n, max_n);
            n_skipped += 1;
            continue;
        }

        eprint!("  {} (n={}, m={}) ... ", name, n, m);

        let output = std::process::Command::new("timeout")
            .arg(format!("{}s", timeout_secs))
            .arg(&self_exe)
            .arg("--single")
            .arg(name)
            .env("KKT_OUTPUT_DIR", output_dir)
            .stderr(std::process::Stdio::piped())
            .output();

        match output {
            Ok(out) => {
                let exit_code = out.status.code();
                let label = if !out.status.success() {
                    if exit_code == Some(124) {
                        "TIMEOUT"
                    } else {
                        "CRASH"
                    }
                } else {
                    "ok"
                };

                // Count .mtx files written regardless of exit status —
                // a timed-out solve still produces matrices from completed iterations.
                let n_matrices = count_mtx_files(&output_dir.join(name));
                eprintln!("{} ({} matrices)", label, n_matrices);
                n_done += 1;
            }
            Err(e) => {
                eprintln!("SPAWN_ERROR({})", e);
                n_skipped += 1;
            }
        }
    }

    // Final tally
    let total: usize = std::fs::read_dir(output_dir)
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .map(|e| count_mtx_files(&e.path()))
                .sum()
        })
        .unwrap_or(0);

    eprintln!();
    eprintln!(
        "Done: {} problems processed, {} skipped",
        n_done, n_skipped
    );
    eprintln!("Total KKT matrices collected: {}", total);
    eprintln!("Output directory: {}", output_dir.display());
}

fn count_mtx_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map_or(false, |x| x == "mtx"))
                .count()
        })
        .unwrap_or(0)
}

fn parse_args(args: &[String]) -> (PathBuf, Vec<String>) {
    let mut output_dir: Option<PathBuf> = None;
    let mut problem_names: Vec<String> = Vec::new();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--output" if i + 1 < args.len() => {
                i += 1;
                output_dir = Some(PathBuf::from(&args[i]));
            }
            arg if arg.starts_with("--output=") => {
                output_dir = Some(PathBuf::from(&arg["--output=".len()..]));
            }
            other => {
                problem_names.push(other.to_string());
            }
        }
        i += 1;
    }

    let output_dir = output_dir.unwrap_or_else(|| {
        eprintln!("Usage: collect_kkt --output <dir> [PROBLEM1 PROBLEM2 ...]");
        eprintln!();
        eprintln!("  --output <dir>   Destination for KKT matrices (required)");
        eprintln!("  PROBLEM...       Problem names; reads benchmarks/cutest/problem_list.txt if omitted");
        eprintln!();
        eprintln!("Env vars:");
        eprintln!("  CUTEST_TIMEOUT   Per-problem timeout in seconds (default: 60)");
        eprintln!("  CUTEST_MAX_N     Skip problems with n > this value (default: unlimited)");
        std::process::exit(1);
    });

    if problem_names.is_empty() {
        let suite_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks").join("cutest");
        let list_path = suite_dir.join("problem_list.txt");
        if list_path.exists() {
            let contents =
                std::fs::read_to_string(&list_path).expect("Failed to read problem_list.txt");
            problem_names = contents
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect();
        } else {
            eprintln!(
                "No problems specified and {} not found.",
                list_path.display()
            );
            std::process::exit(1);
        }
    }

    (output_dir, problem_names)
}
