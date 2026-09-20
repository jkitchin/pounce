//! Structured-KKT Phase 5b: the declared block partition must reach the
//! *restoration* sub-IPM, not just the main solve.
//!
//! Restoration's inner `AlgorithmBuilder` is minted by the frontend before the
//! solve starts — before the KKT layout, and therefore the mapping from a
//! model-space declaration onto it, exists — so the labels cannot be installed
//! on it directly. `IpoptApplication` publishes them into a shared cell as it
//! builds the outer algorithm and the inner builder reads that cell when
//! restoration runs (`AlgorithmBuilder::kkt_blocks_shared`).
//!
//! That this is *sound* is the part worth stating: `AugRestoSystemSolver`
//! reduces the 8-block restoration KKT onto the original 4-block system before
//! delegating to its inner solver, so the matrix that inner solver factors has
//! the outer system's dimension and sparsity and the outer labels describe it
//! exactly. It is not an approximation and not a re-derivation.
//!
//! Measured on the model this was built for (an infeasible 209k-variable
//! corrective N-1 SCOPF, `dev-notes/kkt-scaling-phase0b.md`): restoration's
//! factorization time falls 35.0s -> 11.4s, against 2.8s for the whole main
//! solve — i.e. restoration was by far the larger share, and is why the option
//! exists at all.
//!
//! Mutation table:
//!
//! | Break | Which test goes red |
//! |---|---|
//! | Do not publish the labels (drop the `kkt_blocks_published` write) | `the_partition_reaches_restoration` |
//! | Do not read the cell in `build_with_backend` | `the_partition_reaches_restoration` |
//! | Ignore `kkt_block_restoration=no` | `the_option_turns_the_restoration_path_off` |
//! | Publish stale labels across solves (drop the per-solve reset) | `a_second_solve_does_not_inherit_the_first_ones_labels` |
//! | Drop the median-block-size guard | `tiny_blocks_are_refused_by_default` |

use pounce_algorithm::application::{
    IpoptApplication, Ma57Config, default_backend_factory_with_sink, feral_config_from_options,
};
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    InnerBackendFactoryFactory, make_default_restoration_factory_provider,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Wire the restoration phase the way every frontend does, with its own
/// summary sink so `linear_solver.restoration` is populated. A bare
/// `IpoptApplication` has no restoration factory at all — it reports
/// `RestorationFailed` the moment the line search needs one — so a test about
/// restoration's linear algebra has to do this first.
fn wire_restoration(app: &mut IpoptApplication) {
    let feral_cfg = feral_config_from_options(app.options());
    let sink = app.restoration_summary_sink();
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        let sink = std::sync::Arc::clone(&sink);
        Box::new(move || {
            default_backend_factory_with_sink(
                feral_cfg.clone(),
                Ma57Config::default(),
                std::sync::Arc::clone(&sink),
            )
        })
    };
    let provider = make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        app.algorithm_builder_from_options(),
        bff_mint,
    );
    app.set_restoration_factory_provider(provider);
}

/// `k` independent two-variable cells sharing one linking variable `z`, with
/// an infeasible pair of rows in every cell:
///
/// ```text
/// min  sum_i x_i^2 + z^2
/// s.t. x_{2j} + x_{2j+1} + z == 1      (cell j, equality)
///      x_{2j}^2 + x_{2j+1}^2 >= 4      (cell j, inequality)
///      0 <= x_i <= 1,  0 <= z <= 1
/// ```
///
/// The equality holds the cell's two variables inside `[0, 1]` with a sum of
/// at most 1, so the largest `x_{2j}^2 + x_{2j+1}^2` can reach is 1: every
/// cell is infeasible on its own, which is what drives the solve into
/// restoration. The rows are nonlinear on purpose — a linear contradiction is
/// answered by presolve's certification long before the IPM runs.
///
/// Arrowhead by construction: cell `j`'s rows touch only cell `j`'s variables
/// and `z`, so the declaration is exact rather than approximate.
struct Cells {
    k: usize,
}

impl Cells {
    fn n(&self) -> usize {
        2 * self.k + 1
    }

    fn m(&self) -> usize {
        2 * self.k
    }

    /// The declaration: both variables of cell `j` and both of its rows carry
    /// label `j`; the linking variable is shared (`-1`).
    fn declaration(&self) -> (Vec<i32>, Vec<i32>) {
        let mut vars: Vec<i32> = (0..self.k as i32).flat_map(|j| [j, j]).collect();
        vars.push(-1);
        let cons: Vec<i32> = (0..self.k as i32).flat_map(|j| [j, j]).collect();
        (vars, cons)
    }
}

impl TNLP for Cells {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n() as i32,
            m: self.m() as i32,
            // equality: two cell columns + z; inequality: two cell columns.
            nnz_jac_g: (5 * self.k) as i32,
            // Diagonal: the objective is separable and the only nonlinear row
            // is `x^2 + y^2`, whose Hessian is diagonal too.
            nnz_h_lag: self.n() as i32,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(0.0);
        b.x_u.fill(1.0);
        for j in 0..self.k {
            b.g_l[2 * j] = 1.0;
            b.g_u[2 * j] = 1.0;
            b.g_l[2 * j + 1] = 4.0;
            b.g_u[2 * j + 1] = 2e19;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.fill(0.5);
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        for j in 0..self.k {
            types[2 * j] = Linearity::Linear;
            types[2 * j + 1] = Linearity::NonLinear;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x.iter().map(|v| v * v).sum())
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        for (g, v) in grad.iter_mut().zip(x) {
            *g = 2.0 * v;
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let z = x[self.n() - 1];
        for j in 0..self.k {
            let (a, b) = (x[2 * j], x[2 * j + 1]);
            g[2 * j] = a + b + z;
            g[2 * j + 1] = a * a + b * b;
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let z_col = (self.n() - 1) as i32;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for j in 0..self.k {
                    let (a, b) = ((2 * j) as i32, (2 * j + 1) as i32);
                    let base = 5 * j;
                    irow[base] = 2 * j as i32;
                    jcol[base] = a;
                    irow[base + 1] = 2 * j as i32;
                    jcol[base + 1] = b;
                    irow[base + 2] = 2 * j as i32;
                    jcol[base + 2] = z_col;
                    irow[base + 3] = 2 * j as i32 + 1;
                    jcol[base + 3] = a;
                    irow[base + 4] = 2 * j as i32 + 1;
                    jcol[base + 4] = b;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("values mode carries x");
                for j in 0..self.k {
                    let base = 5 * j;
                    values[base] = 1.0;
                    values[base + 1] = 1.0;
                    values[base + 2] = 1.0;
                    values[base + 3] = 2.0 * x[2 * j];
                    values[base + 4] = 2.0 * x[2 * j + 1];
                }
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..self.n() {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                values.fill(2.0 * obj_factor);
                if let Some(l) = lambda {
                    for j in 0..self.k {
                        values[2 * j] += 2.0 * l[2 * j + 1];
                        values[2 * j + 1] += 2.0 * l[2 * j + 1];
                    }
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

struct Run {
    status: ApplicationReturnStatus,
    main_blocks: Option<usize>,
    resto_blocks: Option<usize>,
    resto_factors: u64,
}

/// Solve the `k`-cell model. `declare` installs the model-space partition;
/// `resto` is the `kkt_block_restoration` setting, `None` leaving it default.
fn solve(k: usize, declare: bool, resto: Option<&str>) -> Run {
    solve_with(k, declare, resto, Some(0))
}

/// `min_block_size` is passed through to `kkt_block_min_size`. This fixture's
/// blocks are two columns wide — far below the size at which the block path
/// pays — so every test here that wants the path *engaged* has to lower the
/// guard. `tiny_blocks_are_refused_by_default` is the one that does not.
fn solve_with(k: usize, declare: bool, resto: Option<&str>, min_block_size: Option<i32>) -> Run {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    if let Some(n) = min_block_size {
        app.options_mut()
            .set_integer_value("kkt_block_min_size", n, true, false)
            .unwrap();
    }
    if let Some(v) = resto {
        app.options_mut()
            .set_string_value("kkt_block_restoration", v, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    wire_restoration(&mut app);
    let problem = Cells { k };
    if declare {
        let (vars, cons) = problem.declaration();
        app.set_block_structure(vars, cons);
    }
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(problem));
    let status = app.optimize_tnlp(tnlp);
    let summary = app.linear_solver_summary();
    let resto_summary = summary.as_ref().and_then(|s| s.restoration.as_ref());
    Run {
        status,
        main_blocks: summary
            .as_ref()
            .and_then(|s| s.blocks.as_ref())
            .map(|b| b.n_blocks),
        resto_blocks: resto_summary
            .and_then(|r| r.blocks.as_ref())
            .map(|b| b.n_blocks),
        resto_factors: resto_summary.map(|r| r.n_factors).unwrap_or(0),
    }
}

/// The model is infeasible however it is solved, and every cell is infeasible
/// on its own — so the solve reaches restoration, which is the precondition
/// for everything below. Without this the other tests could pass vacuously.
#[test]
fn the_model_is_infeasible_and_reaches_restoration() {
    let plain = solve(6, false, None);
    assert!(
        matches!(
            plain.status,
            ApplicationReturnStatus::RestorationFailed
                | ApplicationReturnStatus::InfeasibleProblemDetected
        ),
        "the model has an empty feasible set; got {:?}",
        plain.status
    );
    assert!(
        plain.resto_factors > 0,
        "the fixture must enter restoration for the rest of this file to mean \
         anything; it factored {} times there",
        plain.resto_factors
    );
    assert_eq!(
        plain.main_blocks, None,
        "no declaration, so nothing should have taken the block path"
    );
}

/// The point of the file: a declaration installed on the application reaches
/// the restoration sub-IPM's own factorizations, and does not change the
/// verdict.
#[test]
fn the_partition_reaches_restoration() {
    let plain = solve(6, false, None);
    let declared = solve(6, true, None);
    assert_eq!(
        declared.status, plain.status,
        "the partition must not change the answer"
    );
    assert_eq!(
        declared.main_blocks,
        Some(6),
        "the main solve takes the block path"
    );
    assert_eq!(
        declared.resto_blocks,
        Some(6),
        "and so does restoration — the reduced restoration system is the \
         original one, so the same labels describe it"
    );
}

/// `kkt_block_restoration=no` leaves restoration monolithic while the main
/// solve keeps the block path. This is how restoration's own contribution was
/// measured, so it has to actually do that.
#[test]
fn the_option_turns_the_restoration_path_off() {
    let plain = solve(6, false, None);
    let off = solve(6, true, Some("no"));
    assert_eq!(off.status, plain.status);
    assert_eq!(
        off.main_blocks,
        Some(6),
        "the option is about restoration only"
    );
    assert_eq!(
        off.resto_blocks, None,
        "restoration must factor monolithically"
    );
    assert!(
        off.resto_factors > 0,
        "and it must still be restoration doing the factoring"
    );
}

/// The published labels describe *this* solve's KKT. A second solve on the
/// same application with no declaration must not inherit them: which variables
/// are fixed, and so the whole layout, can differ.
#[test]
fn a_second_solve_does_not_inherit_the_first_ones_labels() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("kkt_block_min_size", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    wire_restoration(&mut app);

    let first = Cells { k: 6 };
    let (vars, cons) = first.declaration();
    app.set_block_structure(vars, cons);
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(first));
    app.optimize_tnlp(tnlp);
    assert_eq!(
        app.linear_solver_summary()
            .and_then(|s| s.blocks.map(|b| b.n_blocks)),
        Some(6),
        "first solve declared a partition"
    );

    app.clear_block_structure();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Cells { k: 6 }));
    app.optimize_tnlp(tnlp);
    let summary = app.linear_solver_summary().expect("a second summary");
    assert!(
        summary.blocks.is_none(),
        "the declaration was cleared; the main solve must be monolithic"
    );
    assert!(
        summary
            .restoration
            .as_ref()
            .and_then(|r| r.blocks.as_ref())
            .is_none(),
        "and restoration must not still be reading the first solve's labels"
    );
}

/// The guard the other tests in this file switch off. This fixture's blocks
/// are two columns wide, and at that size the block path is *slower* than the
/// monolithic one — measured through discopt on a 32-block arrowhead,
/// factorization runs 0.31x at 60 columns per block and does not reach parity
/// until ~250. A declaration is not a reason to take a slower path, so pounce
/// refuses it at the default `kkt_block_min_size` and says so in the log.
///
/// This is the one test here that leaves the option alone.
#[test]
fn tiny_blocks_are_refused_by_default() {
    let declared = solve_with(6, true, None, None);
    assert_eq!(
        declared.main_blocks, None,
        "two-column blocks are below the size at which the partition pays"
    );
    assert_eq!(
        declared.resto_blocks, None,
        "and restoration inherits the refusal, not the labels"
    );
    assert_eq!(
        declared.status,
        solve(6, false, None).status,
        "refusing the partition must not change the answer either"
    );
    // The same declaration is honored once the guard is lowered, so what the
    // test above pins is the guard and not some other refusal.
    assert_eq!(solve_with(6, true, None, Some(0)).main_blocks, Some(6));
}
