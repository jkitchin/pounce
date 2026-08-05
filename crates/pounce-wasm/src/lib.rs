//! WebAssembly entry points for POUNCE.
//!
//! POUNCE's solver core is pure Rust with no C/Fortran dependency on the
//! default build, so the whole `.nl` → AD tape → interior-point pipeline
//! compiles to `wasm32-wasip1` unchanged. This crate is the thin C-ABI
//! shim a browser page talks to: hand it the bytes of an AMPL `.nl` file,
//! get back a JSON problem summary, then ask it to solve.
//!
//! # ABI
//!
//! Everything crosses the boundary as UTF-8 in linear memory, because that
//! is all the raw `WebAssembly` API can express without a bindings
//! generator (no `wasm-bindgen` dependency — see `web/README.md`):
//!
//! * [`pounce_alloc`] / [`pounce_dealloc`] — let JS place input bytes in
//!   wasm memory.
//! * [`pounce_load`] — parse a `.nl` (plus optional `.col` / `.row` name
//!   files), keep the built [`NlTnlp`] in a per-instance slot, and return
//!   a JSON summary. Returns `{"error": …}` on a bad file.
//! * [`pounce_solve`] — solve the loaded problem with an `ipopt.opt`-style
//!   options string, returning a JSON result.
//! * [`pounce_builder_regression`] — solve the constrained two-variable
//!   builder fixture used by the Wasm regression smoke test. This deliberately
//!   exercises `pounce_rs::builder::Nlp`.
//! * [`pounce_solution_sol`] — format the last solve as an AMPL `.sol` file,
//!   the same bytes `pounce model.nl` writes, so a browser result can be read
//!   back by AMPL or Pyomo.
//! * [`pounce_free_payload`] — release a returned payload.
//!
//! Every returned payload is a little-endian `u32` byte count followed by
//! that many UTF-8 bytes, so the caller never has to scan for a terminator.
//! Every entry point catches panics, so a malformed model surfaces as a JSON
//! error rather than a trapped instance the page would have to rebuild.
//!
//! The solver's own console output (banner, iteration table, exit line) is
//! written to stdout as usual; under WASI the host shim receives it through
//! `fd_write`, which is how the demo page streams the live iteration log.

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_nl::nl_reader::{NlProblem, NlTnlp, parse_nl_text};
use pounce_nl::sol_writer::{
    SolSuffix, SolSuffixTarget, SolSuffixValues, SolutionFile, format_sol,
};
use pounce_nlp::expression_provider::ExpressionProvider;
use pounce_nlp::tnlp::{Linearity, TNLP};
use pounce_rs::builder::{Nlp as BuilderNlp, Problem as BuilderProblem};

/// Bound on how many per-variable / per-constraint entries a JSON payload
/// carries. A million-variable model would otherwise serialize a JSON array
/// the page can neither render nor afford to parse; the arrays are for
/// display, and the summary reports the true counts separately.
const PREVIEW_LIMIT: usize = 2000;

/// What a `.sol` download needs from the last solve: the primal and dual
/// blocks in original `.nl` order, and the status the file reports.
struct SolveOutcome {
    x: Vec<f64>,
    g: Vec<f64>,
    lambda: Vec<f64>,
    /// Bound multipliers in Ipopt's internal convention (both `>= 0`); the
    /// `.sol` writer applies the sign flip AMPL expects on the upper block.
    z_l: Vec<f64>,
    z_u: Vec<f64>,
    status: pounce_nlp::return_codes::ApplicationReturnStatus,
}

thread_local! {
    /// The last completed solve, kept for [`pounce_solution_sol`]. Cleared
    /// whenever a new model is loaded, so a `.sol` can never describe a
    /// different model than the one on screen.
    static LAST_SOLVE: RefCell<Option<SolveOutcome>> = const { RefCell::new(None) };

    /// The currently loaded model. A wasm instance drives one model at a
    /// time (the demo runs one instance per worker), so a single slot is
    /// enough and keeps the ABI handle-free.
    static LOADED: RefCell<Option<Rc<RefCell<NlTnlp>>>> = const { RefCell::new(None) };
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------
//
// Two allocation shapes cross the boundary, and both carry their length
// explicitly rather than relying on a sentinel:
//
// * caller-owned input buffers — `pounce_alloc(len)` / `pounce_dealloc(ptr,
//   len)`, allocated and freed with the same `Layout`;
// * module-owned return payloads — a `u32` little-endian length followed by
//   that many UTF-8 bytes, released with `pounce_free_payload`.
//
// The length prefix exists because scanning for a NUL terminator makes the
// reader's correctness depend on the payload never containing a zero byte
// and on the scan staying inside the buffer; a truncated read then surfaces
// as an unrelated JSON parse error rather than as the memory bug it is.
// Reading a length can't drift like that.

/// Bytes of the little-endian `u32` length prefix on every returned payload.
const PREFIX: usize = 4;

fn layout_for(len: usize) -> Option<std::alloc::Layout> {
    std::alloc::Layout::from_size_align(len, 1).ok()
}

/// Allocate `len` bytes in wasm linear memory for the caller to write into.
/// Returns null when `len` is 0 or the allocation fails.
///
/// # Safety
/// The returned pointer must be released with [`pounce_dealloc`] using the
/// same `len`, or handed to [`pounce_load`], which takes no ownership.
#[unsafe(no_mangle)]
pub extern "C" fn pounce_alloc(len: usize) -> *mut u8 {
    match layout_for(len) {
        // SAFETY: `layout_for` rejects a zero size, so the layout is valid
        // for `alloc`. A null return is propagated to the caller as-is.
        Some(layout) if len > 0 => unsafe { std::alloc::alloc(layout) },
        _ => std::ptr::null_mut(),
    }
}

/// Release a buffer obtained from [`pounce_alloc`].
///
/// # Safety
/// `ptr`/`len` must come from a single [`pounce_alloc`] call and must not
/// have been freed already.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pounce_dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    if let Some(layout) = layout_for(len) {
        // SAFETY: the caller guarantees ptr came from `pounce_alloc` with
        // this same `len`, hence this same layout.
        unsafe { std::alloc::dealloc(ptr, layout) }
    }
}

/// Release a payload returned by [`pounce_load`], [`pounce_solve`], or
/// [`pounce_solution_sol`].
///
/// # Safety
/// `ptr` must be a payload pointer this module returned and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pounce_free_payload(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: every payload starts with its own length prefix, written by
    // `to_payload`, so the original allocation size is recoverable here.
    let len = unsafe { std::ptr::read_unaligned(ptr.cast::<u32>()) } as usize;
    if let Some(layout) = layout_for(PREFIX + len) {
        // SAFETY: same allocation, same layout as `to_payload` used.
        unsafe { std::alloc::dealloc(ptr, layout) }
    }
}

/// Move `s` into a caller-owned `[u32 length][UTF-8 bytes]` allocation.
/// Returns null only if the allocation fails.
fn to_payload(s: &str) -> *mut u8 {
    let bytes = s.as_bytes();
    let Ok(len) = u32::try_from(bytes.len()) else {
        // A payload larger than 4 GiB cannot be described by the prefix —
        // and cannot exist in a 32-bit address space either.
        return std::ptr::null_mut();
    };
    let Some(layout) = layout_for(PREFIX + bytes.len()) else {
        return std::ptr::null_mut();
    };
    // SAFETY: `layout` has a non-zero size (PREFIX > 0).
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        return ptr;
    }
    // SAFETY: `ptr` owns `PREFIX + bytes.len()` bytes; the prefix is written
    // unaligned because the allocation has align 1, and the payload follows.
    unsafe {
        std::ptr::write_unaligned(ptr.cast::<u32>(), len);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(PREFIX), bytes.len());
    }
    ptr
}

fn error_json(msg: impl std::fmt::Display) -> *mut u8 {
    to_payload(&serde_json::json!({ "error": msg.to_string() }).to_string())
}

/// Borrow `len` bytes at `ptr` as `&str`. Empty when `ptr` is null or `len`
/// is 0, so optional inputs can be passed as `(0, 0)`.
///
/// # Safety
/// `ptr`/`len` must describe an initialized, readable region that stays
/// valid for the call.
unsafe fn str_from_parts<'a>(ptr: *const u8, len: usize) -> Result<&'a str, String> {
    if ptr.is_null() || len == 0 {
        return Ok("");
    }
    // SAFETY: the caller guarantees the region is valid and initialized.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).map_err(|e| format!("input is not valid UTF-8: {e}"))
}

/// Run `f`, converting a panic into a JSON error string. `.nl` input is
/// arbitrary user data; a panic inside the parser or the solver must not
/// poison the wasm instance for the rest of the page's lifetime.
fn guarded(what: &str, f: impl FnOnce() -> *mut u8) -> *mut u8 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(p) => p,
        Err(_) => error_json(format!("{what} panicked (see the console log for details)")),
    }
}

// ---------------------------------------------------------------------------
// Load + summarize
// ---------------------------------------------------------------------------

/// Parse a `.nl` file and report what is in it.
///
/// `col_*` / `row_*` are the optional sibling `.col` / `.row` name files
/// AMPL writes under `option auxfiles rc;`; pass `(null, 0)` when absent.
/// The parsed model is retained for a following [`pounce_solve`].
///
/// # Safety
/// Each pointer/length pair must describe a readable region valid for the
/// call, or be `(null, 0)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pounce_load(
    nl_ptr: *const u8,
    nl_len: usize,
    col_ptr: *const u8,
    col_len: usize,
    row_ptr: *const u8,
    row_len: usize,
) -> *mut u8 {
    guarded("load", || {
        // SAFETY: forwarded from this function's own safety contract.
        let (nl, col, row) = unsafe {
            (
                str_from_parts(nl_ptr, nl_len),
                str_from_parts(col_ptr, col_len),
                str_from_parts(row_ptr, row_len),
            )
        };
        let (nl, col, row) = match (nl, col, row) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return error_json(e),
        };
        if nl.trim().is_empty() {
            return error_json("empty .nl input");
        }

        let mut prob = match parse_nl_text(nl) {
            Ok(p) => p,
            Err(e) => return error_json(format!("could not parse .nl file: {e}")),
        };
        attach_names(&mut prob, col, row);

        let tnlp = match NlTnlp::try_new(prob) {
            Ok(t) => t,
            Err(e) => return error_json(e),
        };
        let tnlp = Rc::new(RefCell::new(tnlp));
        let summary = summarize(&mut tnlp.borrow_mut());
        LAST_SOLVE.with(|slot| *slot.borrow_mut() = None);
        LOADED.with(|slot| *slot.borrow_mut() = Some(Rc::clone(&tnlp)));
        to_payload(&summary.to_string())
    })
}

/// Scatter `.col` / `.row` line-per-name text onto the parsed problem, the
/// same way [`pounce_nl::nl_reader::read_nl_file`] does for files on disk.
/// A name file of the wrong length is ignored rather than mislabeling rows.
fn attach_names(prob: &mut NlProblem, col: &str, row: &str) {
    let lines = |txt: &str| -> Vec<String> {
        txt.lines()
            .map(str::trim_end)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect()
    };
    let var_names = lines(col);
    if var_names.len() == prob.n {
        prob.var_names = var_names;
    }
    let con_names = lines(row);
    if con_names.len() == prob.m {
        prob.con_names = con_names;
    }
}

/// Classification of a `[lo, hi]` pair, shared by the variable and
/// constraint tallies. `INF` here is AMPL's 1e19 sentinel convention.
const INF: f64 = 1.0e19;

#[derive(Default)]
struct BoundTally {
    free: usize,
    lower_only: usize,
    upper_only: usize,
    boxed: usize,
    fixed: usize,
}

impl BoundTally {
    fn count(lo: &[f64], hi: &[f64]) -> Self {
        let mut t = Self::default();
        for (l, u) in lo.iter().zip(hi.iter()) {
            let has_l = *l > -INF;
            let has_u = *u < INF;
            match (has_l, has_u) {
                (false, false) => t.free += 1,
                (true, false) => t.lower_only += 1,
                (false, true) => t.upper_only += 1,
                (true, true) if l == u => t.fixed += 1,
                (true, true) => t.boxed += 1,
            }
        }
        t
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "free": self.free,
            "lower_only": self.lower_only,
            "upper_only": self.upper_only,
            "boxed": self.boxed,
            "fixed": self.fixed,
        })
    }
}

fn preview<T: Clone>(v: &[T]) -> (&[T], bool) {
    if v.len() > PREVIEW_LIMIT {
        (&v[..PREVIEW_LIMIT], true)
    } else {
        (v, false)
    }
}

/// Build the JSON problem summary shown before a solve: sizes, sparsity,
/// how the bounds break down, and how much of the model is nonlinear.
fn summarize(tnlp: &mut NlTnlp) -> serde_json::Value {
    let info = tnlp.get_nlp_info();
    let (n, m, nnz_jac, nnz_hess) = match info {
        Some(i) => (
            i.n as usize,
            i.m as usize,
            i.nnz_jac_g as usize,
            i.nnz_h_lag as usize,
        ),
        None => (0, 0, 0, 0),
    };

    let mut var_lin = vec![Linearity::Linear; n];
    let n_nonlinear_vars = if tnlp.get_variables_linearity(&mut var_lin) {
        var_lin
            .iter()
            .filter(|l| **l == Linearity::NonLinear)
            .count()
    } else {
        0
    };
    let mut con_lin = vec![Linearity::Linear; m];
    let n_nonlinear_cons = if tnlp.get_constraints_linearity(&mut con_lin) {
        con_lin
            .iter()
            .filter(|l| **l == Linearity::NonLinear)
            .count()
    } else {
        0
    };

    let prob = tnlp.problem();
    let var_bounds = BoundTally::count(&prob.x_l, &prob.x_u);
    let con_bounds = BoundTally::count(&prob.g_l, &prob.g_u);
    // A constraint whose bounds coincide is an equality; the rest of the
    // tally reads as inequalities (one-sided or ranged).
    let n_equality = con_bounds.fixed;

    let (var_names, var_names_truncated) = preview(&prob.var_names);
    let (con_names, con_names_truncated) = preview(&prob.con_names);
    let (x0, x0_truncated) = preview(&prob.x0);

    serde_json::json!({
        "n_vars": n,
        "n_cons": m,
        "n_objs": prob.num_obj,
        "sense": if prob.minimize { "minimize" } else { "maximize" },
        "nnz_jac": nnz_jac,
        "nnz_hess": nnz_hess,
        "jac_density": if n * m > 0 { nnz_jac as f64 / (n as f64 * m as f64) } else { 0.0 },
        "n_nonlinear_vars": n_nonlinear_vars,
        "n_nonlinear_cons": n_nonlinear_cons,
        "n_equality_cons": n_equality,
        "n_inequality_cons": m - n_equality,
        "degrees_of_freedom": n as i64 - n_equality as i64,
        "var_bounds": var_bounds.to_json(),
        "con_bounds": con_bounds.to_json(),
        "external_funcs": prob.imported_funcs.iter().map(|f| f.name.clone()).collect::<Vec<_>>(),
        "var_names": var_names,
        "con_names": con_names,
        "x0": x0,
        "truncated": var_names_truncated || con_names_truncated || x0_truncated,
        "preview_limit": PREVIEW_LIMIT,
    })
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------

/// Solve the model most recently loaded by [`pounce_load`].
///
/// `opts_*` is `ipopt.opt`-style text (`name value` per line, `#` comments)
/// — the same option names the CLI and the Python API take. Pass
/// `(null, 0)` for defaults.
///
/// # Safety
/// `opts_ptr`/`opts_len` must describe a readable region valid for the
/// call, or be `(null, 0)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pounce_solve(opts_ptr: *const u8, opts_len: usize) -> *mut u8 {
    guarded("solve", || {
        // SAFETY: forwarded from this function's own safety contract.
        let opts = match unsafe { str_from_parts(opts_ptr, opts_len) } {
            Ok(s) => s,
            Err(e) => return error_json(e),
        };
        let Some(tnlp) = LOADED.with(|slot| slot.borrow().clone()) else {
            return error_json("no model loaded — call pounce_load first");
        };
        to_payload(&solve_loaded(tnlp, opts).to_string())
    })
}

fn solve_loaded(tnlp: Rc<RefCell<NlTnlp>>, opts: &str) -> serde_json::Value {
    let mut app = IpoptApplication::new();
    if let Err(e) = app.initialize_with_options_str(opts) {
        return serde_json::json!({ "error": format!("bad options: {}", e.message) });
    }

    // Wrap in the auxiliary presolve exactly as the CLI does, so a model
    // solved in the browser takes the same path — and reaches the same
    // answer — as `pounce model.nl` on the command line. `NlTnlp` doubles as
    // the `ExpressionProvider` presolve needs for FBBT.
    let presolve_opts = match pounce_presolve::PresolveOptions::from_options_list(app.options()) {
        Ok(o) => o,
        Err(e) => return serde_json::json!({ "error": format!("presolve setup failed: {e}") }),
    };
    let presolve = presolve_opts.enabled.then(|| {
        app.set_presolve_already_applied(true);
        Rc::new(RefCell::new(
            pounce_presolve::PresolveTnlp::with_expression_provider(
                Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>,
                Rc::clone(&tnlp) as Rc<RefCell<dyn ExpressionProvider>>,
                presolve_opts,
            ),
        ))
    });
    // Phase 6 (#487) stacks on top of the rest of presolve: it is the one
    // pass that removes columns, so it has to be the outermost layer. The
    // browser reads its results off `NlTnlp` below, which receives the
    // full-space solution from `finalize_solution`, so nothing else here
    // needs to know the reduced problem existed.
    let elim = match (&presolve, presolve_opts.linear_eq_reduction) {
        (Some(p), true) => Some(Rc::new(RefCell::new(
            pounce_presolve::LinearEqElimTnlp::new(
                Rc::clone(p) as Rc<RefCell<dyn TNLP>>,
                presolve_opts,
            ),
        ))),
        _ => None,
    };
    let target: Rc<RefCell<dyn TNLP>> = match (&elim, &presolve) {
        (Some(e), _) => Rc::clone(e) as Rc<RefCell<dyn TNLP>>,
        (None, Some(p)) => Rc::clone(p) as Rc<RefCell<dyn TNLP>>,
        (None, None) => Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>,
    };

    let status = app.optimize_tnlp(target);
    let stats = app.statistics();
    let presolve_report = presolve.as_ref().map(|p| {
        let h = p.borrow();
        let tr = h.tighten_report();
        serde_json::json!({
            "tightened_bounds": tr.n_tightened,
            "newly_finite_bounds": tr.n_new_finite,
            "dropped_rows": h.n_dropped_rows(),
            "eliminated_columns": elim
                .as_ref()
                .map(|e| e.borrow().n_eliminated_vars())
                .unwrap_or(0),
            "eliminated_rows": elim
                .as_ref()
                .map(|e| e.borrow().n_eliminated_rows())
                .unwrap_or(0),
        })
    });

    let mut t = tnlp.borrow_mut();
    let x: Vec<f64> = t.final_x().map(<[f64]>::to_vec).unwrap_or_default();
    let lambda: Vec<f64> = t.final_lambda().map(<[f64]>::to_vec).unwrap_or_default();
    let (z_l, z_u) = t
        .final_bound_multipliers()
        .map(|(l, u)| (l.to_vec(), u.to_vec()))
        .unwrap_or_default();
    let objective = t.final_obj();
    // Constraint values at the returned point, so the page can show which
    // rows are tight or violated without re-evaluating the model in JS.
    let m = t.problem().m;
    let mut g = vec![0.0; m];
    if x.is_empty() || !t.eval_g(&x, true, &mut g) {
        g.clear();
    }
    let (g_l, g_u) = (t.problem().g_l.clone(), t.problem().g_u.clone());

    // Remember what a `.sol` download would need: AMPL's file carries the
    // primal and dual blocks plus the solver's own status line, and the
    // multipliers are not reconstructible from the JSON payload (which
    // truncates long vectors for display).
    LAST_SOLVE.with(|slot| {
        *slot.borrow_mut() = Some(SolveOutcome {
            x: x.clone(),
            g: g.clone(),
            lambda: lambda.clone(),
            z_l,
            z_u,
            status,
        })
    });

    let (x_prev, x_truncated) = preview(&x);
    let (g_prev, g_truncated) = preview(&g);
    let (g_l_prev, _) = preview(&g_l);
    let (g_u_prev, _) = preview(&g_u);
    let (lambda_prev, _) = preview(&lambda);

    serde_json::json!({
        "status": format!("{status:?}"),
        "status_code": status.as_int(),
        "success": status.as_int() >= 0,
        "objective": objective,
        "iterations": stats.iteration_count,
        "wall_time_secs": stats.total_wallclock_time_secs,
        "dual_infeasibility": stats.final_unscaled_dual_inf,
        "constraint_violation": stats.final_unscaled_constr_viol,
        "complementarity": stats.final_unscaled_compl,
        "kkt_error": stats.final_unscaled_kkt_error,
        "restoration_calls": stats.restoration_calls,
        "presolve": presolve_report,
        "evals": {
            "objective": stats.num_obj_evals,
            "objective_grad": stats.num_obj_grad_evals,
            "constraints": stats.num_constr_evals,
            "constraint_jac": stats.num_constr_jac_evals,
            "hessian": stats.num_hess_evals,
        },
        "x": x_prev,
        // POUNCE's Lagrange multipliers (L = f + λ'g). AMPL's `.sol` dual
        // block is the marginal value dobj/db = -λ, and `pounce_solution_sol`
        // applies that negation; this field is the raw multiplier.
        "lambda": lambda_prev,
        "g": g_prev,
        "g_l": g_l_prev,
        "g_u": g_u_prev,
        "truncated": x_truncated || g_truncated,
    })
}

// ---------------------------------------------------------------------------
// AMPL .sol export
// ---------------------------------------------------------------------------

/// Format the last solve as an AMPL `.sol` file — the same writer, and so
/// the same bytes, as `pounce model.nl` produces on disk. Lets a browser
/// result be read back by AMPL (`solve_result_num`, `_var.X`, `_con.dual`)
/// or by Pyomo's `.sol` reader.
///
/// Returns null when no solve has completed since the model was loaded.
#[unsafe(no_mangle)]
pub extern "C" fn pounce_solution_sol() -> *mut u8 {
    guarded("sol export", || {
        LAST_SOLVE.with(|slot| match &*slot.borrow() {
            None => std::ptr::null_mut(),
            Some(outcome) => {
                let message = format!("POUNCE {}: {:?}", env!("CARGO_PKG_VERSION"), outcome.status);
                // Reduced costs, exactly as the CLI writes them (gh #296):
                // `ipopt_zL_out = +z_l`, `ipopt_zU_out = -z_u`, so Pyomo's
                // `model.ipopt_zL_out` / AMPL's `.rc` read the same numbers
                // from a browser solve as from a command-line one.
                let suffixes = [
                    SolSuffix {
                        name: "ipopt_zL_out".to_string(),
                        target: SolSuffixTarget::Var,
                        values: SolSuffixValues::Real(outcome.z_l.clone()),
                    },
                    SolSuffix {
                        name: "ipopt_zU_out".to_string(),
                        target: SolSuffixTarget::Var,
                        values: SolSuffixValues::Real(outcome.z_u.iter().map(|&z| -z).collect()),
                    },
                ];
                to_payload(&format_sol(&SolutionFile {
                    message: &message,
                    x: &outcome.x,
                    mult_g: &outcome.lambda,
                    solve_result_num: pounce_solve_report::status_to_solve_result_num(
                        outcome.status,
                    ),
                    suffixes: &suffixes,
                }))
            }
        })
    })
}

/// Format the last solve as CSV: one row per variable and per constraint,
/// with the model's own names when a `.col` / `.row` file was supplied.
///
/// Unlike the JSON the page renders from — which truncates long vectors so a
/// large model stays displayable — this carries every row.
///
/// Returns null when no solve has completed since the model was loaded.
#[unsafe(no_mangle)]
pub extern "C" fn pounce_solution_csv() -> *mut u8 {
    guarded("csv export", || {
        let Some(tnlp) = LOADED.with(|slot| slot.borrow().clone()) else {
            return std::ptr::null_mut();
        };
        LAST_SOLVE.with(|slot| match &*slot.borrow() {
            None => std::ptr::null_mut(),
            Some(outcome) => {
                let t = tnlp.borrow();
                let prob = t.problem();
                let mut out = String::from("kind,index,name,value,lower,upper,multiplier\n");
                for (i, v) in outcome.x.iter().enumerate() {
                    let name = prob.var_names.get(i).cloned().unwrap_or_default();
                    let (lo, hi) = (prob.x_l[i], prob.x_u[i]);
                    out.push_str(&format!(
                        "variable,{i},{},{v:.17e},{lo:.17e},{hi:.17e},\n",
                        csv_field(&name, i, "x")
                    ));
                }
                for (i, v) in outcome.g.iter().enumerate() {
                    let name = prob.con_names.get(i).cloned().unwrap_or_default();
                    let (lo, hi) = (prob.g_l[i], prob.g_u[i]);
                    let mult = outcome.lambda.get(i).copied().unwrap_or(0.0);
                    out.push_str(&format!(
                        "constraint,{i},{},{v:.17e},{lo:.17e},{hi:.17e},{mult:.17e}\n",
                        csv_field(&name, i, "c")
                    ));
                }
                to_payload(&out)
            }
        })
    })
}

/// A quoted CSV name field, falling back to `x[i]` / `c[i]` when the model
/// shipped no name files. Embedded quotes are doubled, per RFC 4180.
fn csv_field(name: &str, index: usize, fallback: &str) -> String {
    let text = if name.is_empty() {
        format!("{fallback}[{index}]")
    } else {
        name.to_string()
    };
    format!("\"{}\"", text.replace('"', "\"\""))
}

struct BuilderRegressionProblem;

impl BuilderProblem for BuilderRegressionProblem {
    fn objective(&self, x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    fn n_constraints(&self) -> usize {
        2
    }

    fn constraints(&self, x: &[f64], out: &mut [f64]) {
        out[0] = x[0].powi(2) + x[1].powi(2);
        out[1] = x[0] + x[1];
    }
}

/// Run the constrained builder fixture used by the Wasm regression smoke.
#[unsafe(no_mangle)]
pub extern "C" fn pounce_builder_regression() -> *mut u8 {
    guarded("builder regression", || {
        let solution = BuilderNlp::new(BuilderRegressionProblem)
            .var_bounds(&[-3.0, -3.0], &[3.0, 3.0])
            .constraint_bounds(&[-2.0e19, 0.5], &[4.0, 2.0e19])
            .x0(&[-1.2, 1.0])
            .option_num("tol", 1.0e-3)
            .option_int("print_level", 0)
            .solve();
        to_payload(
            &serde_json::json!({
                "success": solution.success,
                "status": format!("{:?}", solution.status),
                "objective": solution.objective,
                "x": solution.x,
                "iterations": solution.stats.iteration_count,
            })
            .to_string(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two variables, one nonlinear equality, one linear inequality —
    /// small enough to keep inline, big enough that every field of the
    /// summary has something to say.
    const SIMPLE_NL: &str = include_str!("../tests/simple.nl");

    /// Read a returned payload the way the JS side does — length prefix
    /// first, then that many bytes — and release it.
    fn take_payload(ptr: *mut u8) -> Option<String> {
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ptr` is a payload this module just returned, so the
        // prefix and the bytes behind it are initialized and owned.
        let text = unsafe {
            let len = std::ptr::read_unaligned(ptr.cast::<u32>()) as usize;
            let bytes = std::slice::from_raw_parts(ptr.add(PREFIX), len);
            String::from_utf8_lossy(bytes).into_owned()
        };
        unsafe { pounce_free_payload(ptr) };
        Some(text)
    }

    /// Call the C ABI the way JS does — bytes in, JSON out — and hand back
    /// the parsed payload.
    fn call_load(nl: &str, col: &str, row: &str) -> serde_json::Value {
        let ptr = unsafe {
            pounce_load(
                nl.as_ptr(),
                nl.len(),
                col.as_ptr(),
                col.len(),
                row.as_ptr(),
                row.len(),
            )
        };
        let s = take_payload(ptr).expect("load must return a payload");
        serde_json::from_str(&s).expect("entry points must return JSON")
    }

    fn call_solve(opts: &str) -> serde_json::Value {
        let ptr = unsafe { pounce_solve(opts.as_ptr(), opts.len()) };
        let s = take_payload(ptr).expect("solve must return a payload");
        serde_json::from_str(&s).expect("entry points must return JSON")
    }

    #[test]
    fn load_reports_problem_shape() {
        let s = call_load(SIMPLE_NL, "alpha\nbeta\n", "ring\nline\n");
        assert_eq!(s["n_vars"], 2);
        assert_eq!(s["n_cons"], 2);
        assert_eq!(s["sense"], "minimize");
        assert_eq!(s["var_names"][0], "alpha");
        assert_eq!(s["con_names"][1], "line");
        // One nonlinear row (the circle), one linear row.
        assert_eq!(s["n_nonlinear_cons"], 1);
        assert!(s["nnz_jac"].as_u64().unwrap_or(0) > 0);
    }

    #[test]
    fn name_files_of_the_wrong_length_are_ignored() {
        let s = call_load(SIMPLE_NL, "only_one\n", "");
        assert!(
            s["var_names"]
                .as_array()
                .map(Vec::is_empty)
                .unwrap_or(false),
            "a .col file that does not match n must not be applied"
        );
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        assert!(call_load("not an nl file at all", "", "")["error"].is_string());
        assert!(call_load("", "", "")["error"].is_string());
    }

    #[test]
    fn solve_runs_the_loaded_model() {
        call_load(SIMPLE_NL, "", "");
        let r = call_solve("print_level 0\n");
        assert_eq!(r["success"], true, "solve payload: {r}");
        assert!(r["iterations"].as_i64().unwrap_or(0) > 0);
        // min x0 s.t. x0^2 + x1^2 == 1, x0 + x1 >= 0  ⇒  x0 = -1/√2.
        let obj = r["objective"].as_f64().unwrap_or(f64::NAN);
        assert!(
            (obj + 0.5f64.sqrt()).abs() < 1e-6,
            "unexpected objective {obj}"
        );
        assert_eq!(r["x"].as_array().map(Vec::len), Some(2));
        assert_eq!(r["g"].as_array().map(Vec::len), Some(2));
    }

    #[test]
    fn solving_with_no_model_loaded_is_an_error() {
        LOADED.with(|slot| *slot.borrow_mut() = None);
        assert!(call_solve("")["error"].is_string());
    }

    #[test]
    fn round_trips_a_payload_of_every_size_near_the_prefix() {
        // The length prefix is the whole reason a payload is readable; an
        // off-by-one there would corrupt short payloads first.
        for len in [0usize, 1, 3, 4, 5, 4096] {
            let text = "x".repeat(len);
            let ptr = to_payload(&text);
            assert!(!ptr.is_null(), "allocation failed at len {len}");
            assert_eq!(take_payload(ptr).as_deref(), Some(text.as_str()));
        }
    }

    #[test]
    fn sol_export_matches_the_solve() {
        call_load(SIMPLE_NL, "", "");
        assert!(
            pounce_solution_sol().is_null(),
            "a .sol before any solve would describe nothing"
        );
        let r = call_solve("print_level 0\n");
        let sol = take_payload(pounce_solution_sol()).expect("sol after a solve");

        assert!(sol.starts_with("POUNCE "), "missing status line:\n{sol}");
        assert!(sol.contains("SolveSucceeded"), "unexpected status:\n{sol}");
        // Four-integer count block for m = 2, n = 2, then `objno 0 0`
        // (SolveSucceeded). The dual block is written before the primal.
        assert!(sol.contains("\n2\n2\n2\n2\n"), "count block wrong:\n{sol}");
        assert!(sol.contains("\nobjno 0 0\n"), "objno wrong:\n{sol}");
        // Reduced costs ride along in the same suffix blocks the CLI writes,
        // so Pyomo's `ipopt_zL_out` / `ipopt_zU_out` are populated.
        assert!(
            sol.contains("\nipopt_zL_out\n"),
            "missing zL suffix:\n{sol}"
        );
        assert!(
            sol.contains("\nipopt_zU_out\n"),
            "missing zU suffix:\n{sol}"
        );
        // The primal block must be the x the JSON reported.
        let x0 = r["x"][0].as_f64().unwrap_or(f64::NAN);
        assert!(
            sol.contains(&format!("{x0:.17e}")),
            "x[0] = {x0} missing from the primal block:\n{sol}"
        );
    }

    #[test]
    fn csv_export_covers_every_row() {
        call_load(SIMPLE_NL, "alpha\nbeta\n", "ring\nline\n");
        assert!(pounce_solution_csv().is_null(), "csv before a solve");
        call_solve("print_level 0\n");
        let csv = take_payload(pounce_solution_csv()).expect("csv after a solve");

        let lines: Vec<&str> = csv.trim_end().split('\n').collect();
        // Header + 2 variables + 2 constraints, with the model's own names.
        assert_eq!(lines.len(), 5, "unexpected csv:\n{csv}");
        assert!(lines[0].starts_with("kind,index,name,value"));
        assert!(
            lines[1].starts_with("variable,0,\"alpha\","),
            "{}",
            lines[1]
        );
        assert!(
            lines[3].starts_with("constraint,0,\"ring\","),
            "{}",
            lines[3]
        );
        // Bounds ride along so a reader can see which rows are tight.
        assert_eq!(lines[4].split(',').count(), 7);
    }

    #[test]
    fn csv_falls_back_to_index_labels_without_name_files() {
        call_load(SIMPLE_NL, "", "");
        call_solve("print_level 0\n");
        let csv = take_payload(pounce_solution_csv()).expect("csv");
        assert!(csv.contains("variable,0,\"x[0]\","), "{csv}");
        assert!(csv.contains("constraint,1,\"c[1]\","), "{csv}");
    }

    #[test]
    fn loading_a_new_model_invalidates_the_previous_sol() {
        call_load(SIMPLE_NL, "", "");
        call_solve("print_level 0\n");
        assert!(take_payload(pounce_solution_sol()).is_some());
        // A fresh load must not leave the old solve downloadable — that .sol
        // would carry the previous model's x against the new model's name
        // files.
        call_load(SIMPLE_NL, "", "");
        assert!(
            pounce_solution_sol().is_null(),
            "stale .sol survived a reload"
        );
    }

    #[test]
    fn bad_options_are_reported() {
        call_load(SIMPLE_NL, "", "");
        assert!(call_solve("max_iter not_an_integer\n")["error"].is_string());
    }
}
